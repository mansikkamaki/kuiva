"""The variational quantum eigensolver: the secondary, exploratory CI path.

**Orchestration, not a registered kernel.**

Why it is here at all, given that SQD is the primary path
---------------------------------------------------------
SQD (:mod:`kuiva.qc.sqd`) uses the device as a *sampler* and does the eigensolve and the RDMs
classically, which is what makes the ci_solver contract satisfiable on near-term hardware. VQE does
the opposite: the state stays in the device, its energy is *measured*, and a classical
optimizer moves the circuit parameters. Kept in scope (a user decision) for two reasons and
no others:

* it is the more genuinely quantum of the two — the CI step itself runs on the device rather
  than the configuration *selection* — and the brief asks for that question to be studied;
* it is the **validation partner** for the whole mapping layer. SQD never touches
  :mod:`kuiva.qc.mapping` (its subspace Hamiltonian is built by ``ci/strings.py``), so a
  passing SQD test says nothing about the qubit Hamiltonian. A statevector VQE that reaches
  ``FullCISolver``'s energy on a small active space is an end-to-end check of the mapping, the
  circuits and the estimator together.

⚠ **What this costs, stated up front because it is the finding rather than a caveat.** The
energy is ``sum_t c_t <P_t>`` over ``O(n^4)`` Pauli strings, and the RDMs — which the optimizer
contract *requires* — are another ``O(n^4)`` operators of ``O(n^4)`` strings. On a simulator
that is a vector operation. On hardware it is tomography, it is the single worst NISQ
bottleneck, and it is exactly what SQD was chosen to avoid. :meth:`VQESolver.report` prints the
string and circuit counts for that reason; they are the honest price of this path.

The contract, and what is deliberately not claimed
----------------------------------------------------
A plain ``ci_solver(ints) -> (energy, gamma, Gamma)``, so it plugs into
``optimize_orbitals`` unchanged and ``mcscf.adaptive.as_adaptive_solver`` wraps it if an
event-gated driver is wanted. ⚠ It is **not** an :class:`AdaptiveCISolver`: there is no
``propose``, because a VQE has no notion of proposing a different *space* — its space is the
whole sector and what changes is the state within it. The design defers that
conformance deliberately, and inventing a ``propose`` that re-ran the optimizer would be a
different object wearing the protocol's name.

⚠ **The optimizer tolerance is a noise floor on ``E(kappa)``**, exactly as it is for the cheap
cheap CI: a VQE energy is converged only to whatever the classical optimizer reached, so an
orbital optimizer built on it sees a surface with that much noise on it and cannot converge its
gradient below it. Warm starting from the previous macro-iteration's parameters is what keeps
the surface *approximately* smooth; it does not make it exactly so, and no claim here says it
does.

⚠ **A shot-based VQE energy is not variational.** The exact expectation value of a trial state
is an upper bound on the ground-state energy; a *sampled estimate* of it is a random variable
that lands below the true energy about half the time when the two are within the standard
error. Nothing here reports a shot-based energy as a bound, and a caller must not either.

Gradients
---------
Two, and the choice is not a matter of taste:

* ``gradient="parameter-shift"`` — exact, hardware-realizable, ``2`` circuit evaluations per
  *gate* (:func:`kuiva.qc.fermionic.parameter_shift_gradient`). Expensive: a UCC doubles
  amplitude drives 8 rotations, so one parameter costs 16 energies.
* ``gradient=None`` with a derivative-free optimizer (COBYLA, the default) — what a shot-noisy
  objective actually tolerates, and what most published VQE work uses for that reason.

⚠ A finite-difference gradient is **not** offered. On a shot-noisy energy its error is the
shot noise divided by the step size, which is unbounded as the step shrinks; the shift rule has
no step to divide by. Offering both would invite the wrong one to be picked on a simulator and
then carried to hardware.

References
----------------------------
* A. Peruzzo, J. McClean, P. Shadbolt, M.-H. Yung, X.-Q. Zhou, P. J. Love, A. Aspuru-Guzik,
  J. L. O'Brien, "A variational eigenvalue solver on a photonic quantum processor",
  *Nat. Commun.* **5**, 4213 (2014), doi:10.1038/ncomms5213.
* J. R. McClean, J. Romero, R. Babbush, A. Aspuru-Guzik, "The theory of variational hybrid
  quantum-classical algorithms", *New J. Phys.* **18**, 023023 (2016),
  doi:10.1088/1367-2630/18/2/023023.
* The parameter-shift rule: K. Mitarai, M. Negoro, M. Kitagawa, K. Fujii, "Quantum circuit
  learning", *Phys. Rev. A* **98**, 032309 (2018), doi:10.1103/PhysRevA.98.032309; M. Schuld,
  V. Bergholm, C. Gogolin, J. Izaac, N. Killoran, "Evaluating analytic gradients on quantum
  hardware", *Phys. Rev. A* **99**, 032331 (2019), doi:10.1103/PhysRevA.99.032331.
* COBYLA: M. J. D. Powell, "A direct search optimization method that models the objective and
  constraint functions by linear interpolation", in *Advances in Optimization and Numerical
  Analysis* (1994), doi:10.1007/978-94-015-8330-5_4.
* The classical half — determinant algebra and RDMs — is unchanged Kuiva: ``ci/strings.py``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..ci.strings import CASSpace, cas_dimension, rdm12
from ..util.logging import get_logger
from ..util.timing import timer
from .ansatz import UCCStrategy, VariationalAnsatz
from .backend import BackendProvenance, get_backend, require_primitives
from .fermionic import parameter_shift_gradient
from .mapping import qubit_hamiltonian, qwc_groups, rdm_measurement

log = get_logger(__name__)

#: Primitives this algorithm needs of a backend. ⚠ ``statevector`` is needed *in addition*
#: whenever ``rdm_source="statevector"``, and the constructor checks for it then — declared per
#: instance rather than in the registry, because it depends on how the solver was configured.
REQUIRED_PRIMITIVES = ("estimate",)

#: Below this the state has strayed too far off the requested particle number for its RDMs to
#: mean anything. Not a tolerance to tune: a number-conserving ansatz leaks ``0`` and a
#: hardware-efficient one leaks most of its norm, so anything in between is a broken circuit.
SECTOR_WEIGHT_FLOOR = 0.99


class VQESolver:
    """A variational quantum eigensolver satisfying the plain ``ci_solver`` contract.

    Parameters
    ----------
    n_elec : int — electrons in the active space.
    backend : str or object — a registered backend name or an instance.
    ansatz : VariationalAnsatz, optional
        Defaults to :class:`~kuiva.qc.ansatz.UCCStrategy` — the Stage-C ansatz, chosen as the
        default here because a VQE over a *hardware-efficient* circuit does not conserve
        particle number and its RDMs would not describe an ``n_elec`` state at all.
    shots : int, optional
        Shots per Pauli estimate. ``None`` asks the backend for exact expectation values and is
        legal only on a simulator. ⚠ See the module docstring on why a shot-based energy is not
        a variational bound.
    optimizer : ``"cobyla"``, ``"l-bfgs-b"`` or ``"powell"``.
    gradient : ``"parameter-shift"`` or ``None``.
    rdm_source : ``"statevector"`` or ``"estimate"``.
        ``"statevector"`` projects the exact amplitudes onto the determinant space and calls
        ``ci/strings.rdm12`` — Kuiva's own validated RDM code, reused rather than re-derived,
        and simulator-only. ``"estimate"`` measures the RDM operators
        (:func:`kuiva.qc.mapping.rdm_measurement`), which is what hardware would have to do and
        is here to make that cost visible.
    maxiter : int — classical optimizer iterations. A budget, not a convergence promise.
    """

    def __init__(self, n_elec: int, *, backend: Any = "stub",
                 ansatz: Optional[VariationalAnsatz] = None, shots: Optional[int] = None,
                 seed: Optional[int] = 0, optimizer: str = "cobyla",
                 gradient: Optional[str] = None, rdm_source: str = "statevector",
                 maxiter: int = 200, mapping: str = "jordan_wigner",
                 initial_params: Optional[np.ndarray] = None) -> None:
        self.n_elec = int(n_elec)
        self.backend = get_backend(backend) if isinstance(backend, str) else backend
        needed = list(REQUIRED_PRIMITIVES)
        if str(rdm_source) == "statevector":
            needed.append("statevector")
        require_primitives(self.backend, needed, algorithm="vqe")
        self.ansatz = UCCStrategy(self.n_elec) if ansatz is None else ansatz
        self.shots = None if shots is None else int(shots)
        self.seed = None if seed is None else int(seed)
        self.optimizer = str(optimizer).lower()
        if gradient not in (None, "parameter-shift"):
            raise ValueError("gradient must be None or 'parameter-shift', got {!r}; a finite "
                             "difference is deliberately not offered (module docstring)"
                             .format(gradient))
        self.gradient = gradient
        if str(rdm_source) not in ("statevector", "estimate"):
            raise ValueError("rdm_source must be 'statevector' or 'estimate', got {!r}"
                             .format(rdm_source))
        self.rdm_source = str(rdm_source)
        self.maxiter = int(maxiter)
        self.mapping = str(mapping)

        self.params: Optional[np.ndarray] = (None if initial_params is None
                                             else np.asarray(initial_params, dtype=np.float64))
        #: Provenance of the most recent backend call (an energy whose record does not
        #: say which device and how many shots produced it is not interpretable).
        self.provenance: Optional[BackendProvenance] = None
        self.last_energy: Optional[float] = None
        self.last_operator = None
        self.n_energy_evaluations = 0
        self.n_solves = 0
        self.history: list = []

    # -- the objective ----------------------------------------------------------------------

    def _active(self, ints):
        return (np.ascontiguousarray(ints.h_active_effective()), np.asarray(ints.active_eri()),
                float(getattr(ints, "e_core", 0.0)))

    def _draw_seed(self) -> Optional[int]:
        """Seed for the next backend call, advanced per evaluation.

        ⚠ Advanced for the same reason ``SQDSolver.draw_seed`` advances: a fixed seed makes
        every shot-based estimate replay one record, so the optimizer would be walking a
        *deterministic* function that is not the energy. Still derived from one number, so the
        whole run replays.
        """
        return None if self.seed is None else self.seed + self.n_energy_evaluations

    def energy_of(self, circuit, operator, e_core: float) -> float:
        """``<circuit| H |circuit> + e_core`` through the backend's ``estimate`` primitive."""
        result = self.backend.estimate(circuit, operator.x_masks, operator.z_masks,
                                       shots=self.shots, seed=self._draw_seed())
        self.provenance = result.provenance
        self.n_energy_evaluations += 1
        value, _variance = result.combine(operator.coeffs)
        return float(value) + float(e_core)

    def optimize(self, ints, *, params: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """Minimize the energy over the ansatz parameters; return ``(energy, params)``.

        Warm-started from :attr:`params` unless ``params=`` is given — which is what keeps the
        energy an approximately smooth function of the integrals across macro-iterations.
        """
        from scipy.optimize import minimize

        h, eri, e_core = self._active(ints)
        n_qubit = int(h.shape[0])
        operator = qubit_hamiltonian(ints, mapping=self.mapping)
        self.last_operator = operator
        if params is None:
            params = (self.ansatz.initial_parameters(n_qubit, h=h, eri=eri)
                      if self.params is None else self.params)
        params = np.asarray(params, dtype=np.float64)

        def objective(x):
            return self.energy_of(self.ansatz.build(n_qubit, x, h=h, eri=eri).circuit,
                                  operator, e_core)

        def jacobian(x):
            compiled = self.ansatz.build(n_qubit, x, h=h, eri=eri)
            return parameter_shift_gradient(
                compiled, lambda circuit: self.energy_of(circuit, operator, e_core))

        with timer("VQE optimization"):
            if self.optimizer == "cobyla":
                out = minimize(objective, params, method="COBYLA",
                               options={"maxiter": self.maxiter})
            elif self.optimizer == "powell":
                out = minimize(objective, params, method="Powell",
                               options={"maxiter": self.maxiter})
            elif self.optimizer == "l-bfgs-b":
                if self.gradient != "parameter-shift":
                    raise ValueError(
                        "optimizer='l-bfgs-b' needs gradient='parameter-shift'; SciPy would "
                        "otherwise build a finite-difference gradient of a possibly shot-noisy "
                        "energy, which is the one thing this module refuses (module docstring)")
                out = minimize(objective, params, method="L-BFGS-B", jac=jacobian,
                               options={"maxiter": self.maxiter})
            else:
                raise ValueError("unknown optimizer {!r}; implemented: cobyla, powell, "
                                 "l-bfgs-b".format(self.optimizer))
        self.params = np.asarray(out.x, dtype=np.float64)
        self.last_energy = float(out.fun)
        self.history.append({"energy": float(out.fun), "nit": int(getattr(out, "nit", -1)),
                             "nfev": int(getattr(out, "nfev", -1)),
                             "success": bool(out.success), "message": str(out.message)})
        if not out.success:
            log.warning("the VQE optimizer stopped without converging (%s); the energy is the "
                        "best iterate reached in %d evaluations, and it is a noise floor on "
                        "any orbital optimization built on it", out.message, self.maxiter)
        log.debug("VQE: %d parameters, %d energy evaluations, E = %.10f Eh",
                  params.size, self.n_energy_evaluations, self.last_energy)
        return self.last_energy, self.params

    # -- densities ---------------------------------------------------------------------------

    def state(self, ints, params: Optional[np.ndarray] = None) -> np.ndarray:
        """The optimized state's amplitudes. Simulator only, by declared capability."""
        h, eri, _ = self._active(ints)
        n_qubit = int(h.shape[0])
        params = self.params if params is None else params
        compiled = self.ansatz.build(n_qubit, params, h=h, eri=eri)
        return self.backend.statevector(compiled.circuit)

    def _sector(self, psi: np.ndarray, n_qubit: int) -> Tuple[np.ndarray, float]:
        """Project onto the ``n_elec`` sector; return ``(ci vector, weight kept)``.

        ⚠ **The weight is checked, not assumed.** A number-conserving ansatz keeps 1 exactly;
        anything less means the RDMs describe a renormalized *projection* of the optimized
        state rather than the state whose energy was reported, and the two are different
        objects. :data:`SECTOR_WEIGHT_FLOOR` is where that stops being a rounding question.
        """
        dets = CASSpace(n_qubit, self.n_elec).determinants()
        ci = np.asarray(psi, dtype=np.complex128)[dets.masks.astype(np.int64)]
        weight = float(np.vdot(ci, ci).real)
        return ci / np.sqrt(max(weight, 1e-300)), weight

    def rdms(self, ints, params: Optional[np.ndarray] = None):
        """``(gamma, Gamma)`` of the optimized state, by whichever route was configured."""
        h, eri, _ = self._active(ints)
        n_qubit = int(h.shape[0])
        params = self.params if params is None else params
        if self.rdm_source == "statevector":
            psi = self.state(ints, params)
            ci, weight = self._sector(psi, n_qubit)
            if weight < SECTOR_WEIGHT_FLOOR:
                log.warning("only %.3f of the optimized state lies in the %d-electron sector; "
                            "its RDMs are those of the renormalized projection, which is not "
                            "the state whose energy was reported. Use a particle-conserving "
                            "ansatz (kuiva.qc.ansatz.UCCStrategy).", weight, self.n_elec)
            dets = CASSpace(n_qubit, self.n_elec).determinants()
            return rdm12(dets, ci[:, None], np.array([1.0]))
        plan = rdm_measurement(n_qubit, rank=2)
        compiled = self.ansatz.build(n_qubit, params, h=h, eri=eri)
        result = self.backend.estimate(compiled.circuit, plan.x_masks, plan.z_masks,
                                       shots=self.shots, seed=self._draw_seed())
        self.provenance = result.provenance
        return plan.gamma(result.values), plan.gamma2(result.values)

    # -- the ci_solver contract ---------------------------------------------------------------------

    def solve(self, ints):
        """``(energy, gamma, Gamma)``: optimize, then take the densities of the optimum."""
        energy, params = self.optimize(ints)
        gamma, gamma2 = self.rdms(ints, params)
        self.n_solves += 1
        return energy, gamma, gamma2

    __call__ = solve

    # -- reporting -------------------------------------------------------------------------

    def cost(self, n_qubit: int, *, rank: int = 2) -> Dict[str, int]:
        """The measurement cost of one energy and one RDM set — the point of VQE's measurement-cost caveat.

        Circuit counts are **qubit-wise commuting groups**, which is what a shot-based
        estimator actually runs; the string counts are what an ungrouped one would.
        """
        out: Dict[str, int] = {}
        if self.last_operator is not None:
            out["energy_strings"] = self.last_operator.n_terms
            out["energy_circuits"] = len(qwc_groups(self.last_operator))
        plan = rdm_measurement(int(n_qubit), rank=rank)
        out["rdm_strings"] = plan.n_strings
        out["cas_dimension"] = cas_dimension(int(n_qubit), self.n_elec)
        return out

    def report(self) -> Dict[str, Any]:
        return {
            "algorithm": "vqe",
            "ansatz": type(self.ansatz).__name__,
            "optimizer": self.optimizer,
            "gradient": self.gradient or "none (derivative-free)",
            "shots": self.shots,
            "rdm_source": self.rdm_source,
            "n_parameters": 0 if self.params is None else int(self.params.size),
            "n_energy_evaluations": self.n_energy_evaluations,
            "n_solves": self.n_solves,
            "energy": self.last_energy,
            "provenance": None if self.provenance is None else self.provenance.as_dict(),
        }

    def __repr__(self) -> str:
        return "VQESolver(n_elec={}, ansatz={}, optimizer={}, backend={}, shots={})".format(
            self.n_elec, type(self.ansatz).__name__, self.optimizer,
            getattr(self.backend, "name", "?"), self.shots)


__all__ = ["REQUIRED_PRIMITIVES", "SECTOR_WEIGHT_FLOOR", "VQESolver"]
