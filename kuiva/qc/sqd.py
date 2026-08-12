"""Sample-based quantum diagonalization as a ``ci_solver`` for the shared orbital optimizer.

**Orchestration, not a registered kernel.**

What the algorithm is
---------------------
A quantum circuit is used **only as a configuration sampler**. It is measured in the
computational (Jordan-Wigner) basis; the bitstrings are recovered onto the right particle
number classically (:mod:`kuiva.qc.recovery`); and the Hamiltonian is then diagonalized
**classically, in that sampled subspace** — by literally the same
``ci/strings.hamiltonian_matrix`` machinery ``mcscf/preopt.py``'s CIPSI-style selected CI uses
on a *classically* chosen determinant list. SQD changes only where the list comes from.

Three consequences, and they are why this is the right primary algorithm for Kuiva rather
than merely the currently popular one:

* **The RDMs come from the classical diagonalization**, not from noisy on-hardware
  tomography. That is what makes the ``ci_solver(ints) -> (energy, gamma, Gamma)`` contract
  satisfiable at all on near-term hardware, and it is the single worst NISQ bottleneck a VQE
  path would hit.
* **State averaging is close to free**: the subspace diagonalization yields several
  low eigenpairs at once, and the same ``state_average_weights`` gate every other solver in
  Kuiva goes through is applied to them. A VQE ground-state loop gives one state per
  optimization and needs a genuinely separate excited-state extension.
* **It matches the published VTT Q50/LUMI operating model** — quantum device as sampler,
  classical HPC doing the heavy pre- and post-processing.

⚠ What a good result here does and does not prove
--------------------------------------------------
If the recovered subspace happens to cover the whole CAS space, this solver **is** the full
CI, and reproducing :class:`~kuiva.mcscf.casci.FullCISolver` proves the *pipeline* — sampling,
recovery, subspace solve, RDMs, contract conformance — and nothing about the algorithm.
:func:`kuiva.qc.recovery.subspace_fraction` exists to make that distinction unavoidable, and
every test and example here reports it. ⚠ **It does not even prove the qubit mapping**: the
subspace Hamiltonian is built by ``ci/strings.hamiltonian_matrix``, so :mod:`kuiva.qc.mapping`
is never called on this path and every test here would pass with it arbitrarily wrong.
:mod:`kuiva.qc.vqe` is what covers that. The interesting claim — that a *realistic* circuit
concentrates its shots on the configurations that matter — needs a system large enough for the
question to be interesting and hardware to run it on, and is not made anywhere in this
module.

What is honestly true at any subspace size is weaker and still useful: the subspace energy is
a **variational upper bound**, because the space is a subspace.

⚠ **Monotone needs *nested* spaces, and more shots do not give them.** Two independent draws
at different shot counts are not nested — the larger one can miss a configuration the smaller
one found — so a shot ladder is *not* guaranteed to fall, and asserting that it does is
asserting a property of a random draw. Two things here **are** nested by construction and are
therefore monotone as a theorem: raising ``max_determinants`` at a fixed draw (the cap
truncates one sorted tally), and accumulating proposals with ``accumulate=True``. Use those
when the claim being made is convergence rather than sampling luck.

The adaptive-solver contract, and how the pieces map onto it
-----------------------------------------------------
A sampling solver re-selects its space every time it is asked to run, which is exactly the
shape ``mcscf/adaptive.py`` was built for:

``solve(ints)``
    Diagonalize in the **incumbent** subspace. **No sampling.** Deterministic and smooth in
    ``ints`` at fixed subspace, which is the promise the trust region, the quadratic model and
    the accept/reject test all rest on. ⚠ A ``solve`` that re-sampled would defeat the whole
    of :mod:`kuiva.mcscf.events` silently, and it is the one thing this class must not do.
``propose(ints)``
    Draw fresh shots, recover, optionally union with the incumbent, diagonalize, and hand the
    result back **without adopting it**. The controller adopts only on a variational
    improvement at the same integrals.
``adopt`` / ``space_key``
    ``space_key`` is ``mcscf.adaptive.array_key`` over the sorted recovered mask list —
    reuse, not reinvention, and sorted because a space is identified by what it spans.

⚠ **``accumulate`` (default on) is what makes this converge, and it is also what makes
``propose`` almost always win.** Unioning the fresh sample with the incumbent gives a superset
whenever the cap does not bind, and a superset's variational energy can only go down; that is
the self-consistent configuration-recovery loop of the SQD literature expressed through
Kuiva's event gate. With ``accumulate=False`` each proposal is an independent draw, the
comparison is a genuine competition between two spaces, and the space does not grow — useful
for studying the sampler, wrong for converging a calculation.

⚠ **The first ``solve`` samples, because it must.** There has to be an incumbent space before
there can be a fixed one. That single call is the only sampling ``solve`` ever does, and its
provenance is recorded like any other.

⚠ What a proposal means depends on which circuit strategy is running
----------------------------------------------------------------------
**The Stage-A and Stage-B circuits of :mod:`kuiva.qc.ansatz` do not depend on the
integrals.** They are built from the mode count and a reference occupation and nothing else,
so as the orbitals move the sampling *distribution* does not. What a proposal therefore adds
is **statistical coverage** — more independent draws from the same distribution — and not a
re-selection informed by the current orbitals, which is what ``CheapCISolver`` does and what
event gating was measured on.

⚠ **The Stage-C ansaetze and the SKQD time-evolution strategy do read the integrals**, so with
either of those a proposal *is* a re-selection in the sense the adaptive protocol means. Nothing in this
driver changed to make that true: the seam was always correct — ``h`` and ``eri`` reach
:meth:`SQDSolver.sample_space` and are handed to the strategy — which is why a caller has to
know which rung is running before reading anything into a proposal's acceptance rate.

Two consequences worth stating because both were live defects rather than predictions:

* **The per-draw seed advances** (:meth:`SQDSolver.draw_seed`). With one fixed seed the
  backend replays an identical shot record, and since the circuit is fixed too, every
  proposal reproduces the incumbent space, returns ``None``, and the whole event machinery
  runs at zero effect while spending shots.
* **Accumulation carries counts, not immunity.** Prior configurations enter the next
  recovery with their accumulated shot counts *added* to the new draw's; giving them a count
  above every sampled one instead freezes the space at the first draw whenever a cap binds.

References
----------------------------
* Sample-based quantum diagonalization / quantum-selected CI: "Chemistry beyond the scale of
  exact diagonalization on a quantum-centric supercomputer", *Sci. Adv.* (2025),
  doi:10.1126/sciadv.adu9991; "Localized sample-based quantum diagonalization for strongly
  correlated chemistry", *PNAS* (2025), doi:10.1073/pnas.2603914123; the reference
  implementation ``qiskit-addon-sqd``, github.com/Qiskit/qiskit-addon-sqd — read for the
  recovery and subspace-diagonalization recipes, **not** adopted as-is, since it assumes the
  spin separability Kuiva's Hamiltonian does not have.
* Quantum subspace/Krylov methods, the family the pluggable circuit strategy keeps this open
  to: M. Motta et al., *Electron. Struct.* **6**, 013001 (2024).
* The classical half is unchanged Kuiva: the determinant algebra of ``ci/strings.py`` and the
  selected-space solve of ``mcscf/preopt.py``, whose references are in those modules.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Hashable, Optional, Sequence, Tuple

import numpy as np

from ..ci.strings import Determinants, connections, determinant_memory_gb, rdm12
from ..mcscf.adaptive import Proposal, SolverFailure, array_key
from ..mcscf.preopt import solve_fixed_space
from ..rdm.rdm import DEFAULT_DEGENERACY_TOL, state_average_weights
from ..util import resources as res
from ..util.logging import get_logger
from ..util.timing import timer
from .ansatz import ReferenceExcitationStrategy, resolve_strategy
from .backend import BackendProvenance, get_backend, require_primitives
from .recovery import recover_configurations, subspace_fraction

log = get_logger(__name__)

#: Primitives this algorithm needs of a backend. Sampling and nothing else — which is what
#: makes SQD runnable on a device that offers only counts, and why it is the primary path.
REQUIRED_PRIMITIVES = ("sample",)

#: Default shots per proposal. Not a physical constant: it is the knob that trades subspace
#: coverage against cost, and the honest figure to quote beside any energy from this solver.
DEFAULT_SHOTS = 20_000


def subspace_gb(ndet: int, n_spinor: int, n_states: int = 1) -> float:
    """Size [GB] of the resident arrays one sampled-subspace solve holds.

    The determinant list and its CI vectors, plus the state-averaged 2-RDM. ⚠ The sparse
    Hamiltonian is **not** counted here: its size depends on the connection structure, which
    is not known until the pair search has run, and ``ci/strings.hamiltonian_matrix`` requires
    it against the configured memory limit itself at the point where it *is* known. Counting a guess for it
    here would be padding, which the exact-sizing rule forbids.

    This is the sizing obligation: the published SQD reach is in
    *qubit* count, but the recovered-subspace diagonalization is still a classical CI solve
    and inherits a classical CI's memory ceiling.
    """
    return (determinant_memory_gb(int(ndet), int(n_states))
            + res.rdm_gb(int(n_spinor), 2))


class SQDSolver:
    """Sampled-subspace CI as an :class:`~kuiva.mcscf.adaptive.AdaptiveCISolver`.

    Parameters
    ----------
    n_elec : int — electrons in the active space.
    n_states : int
        Roots to solve for and average over. ⚠ With an odd electron count Kramers' theorem
        makes every level at least doubly degenerate, so an odd count splits a pair and is
        refused where the weights are built (:func:`kuiva.rdm.rdm.state_average_weights`),
        exactly as for every other solver in Kuiva.
    backend : str or object
        A registered backend name (``"stub"``, ``"qiskit_aer"``) or a constructed backend.
        ⚠ The pairing is validated **at construction** against
:data:`REQUIRED_PRIMITIVES`, naming both sides on a refusal.
    strategy : str or object
        A :mod:`kuiva.qc.ansatz` circuit strategy, or a name to construct one from. Defaults
        to the Stage-A reference-excitation strategy.
    shots : int — shots per sampling call.
    accumulate : bool
        Union each fresh sample with the incumbent space (module docstring). On by default.
    max_determinants : int, optional
        Cap on the subspace. ⚠ Truncation is by **shot count**, which is the only ordering
        available before the CI is solved.
    recovery : str
        ``"repair"``, ``"project"`` or ``"kramers"`` (:mod:`kuiva.qc.recovery`). ⚠ **On a
        Kramers-paired active space** — which is every active space Kuiva builds from a
        restricted reference — ``"kramers"`` is the right choice and ``"repair"`` is the
        cheap one: an arbitrary sampled subspace is not time-reversal invariant and splits a
        Kramers degeneracy by an arbitrary amount, landing inside the 1e-8..1e-6 Eh band
        reserved for a *genuine* numerical splitting. It is nonetheless **not the default**,
        because the closure presumes a Kramers-paired set, an unrestricted reference
        does not give one, and a mask array cannot advertise which it is.
    seed : int, optional
        Backend seed. ⚠ Recorded in the provenance and honoured per call, so a simulator run
        is replayable; ``None`` is recorded as ``None`` rather than as a number nothing used.
    weights, degeneracy_tol, on_split, enforce_kramers
        The state-averaging policy, with the same meanings as in
        :class:`~kuiva.mcscf.casci.FullCISolver`.
    """

    KEY_PREFIX = "sqd"

    def __init__(self, n_elec: int, *, n_states: int = 1, backend: Any = "stub",
                 strategy: Any = None, shots: int = DEFAULT_SHOTS,
                 accumulate: bool = True, max_determinants: Optional[int] = None,
                 recovery: str = "repair", seed: Optional[int] = 0,
                 weights: Optional[Sequence[float]] = None,
                 degeneracy_tol: float = DEFAULT_DEGENERACY_TOL,
                 on_split: str = "raise", enforce_kramers: bool = True) -> None:
        self.n_elec = int(n_elec)
        self.n_states = int(n_states)
        self.backend = get_backend(backend) if isinstance(backend, str) else backend
        require_primitives(self.backend, REQUIRED_PRIMITIVES, algorithm="sqd")
        if strategy is None:
            strategy = ReferenceExcitationStrategy(self.n_elec)
        elif isinstance(strategy, str):
            strategy = resolve_strategy(strategy, n_elec=self.n_elec)
        self.strategy = strategy
        self.shots = int(shots)
        self.accumulate = bool(accumulate)
        self.max_determinants = None if max_determinants is None else int(max_determinants)
        self.recovery = str(recovery)
        self.seed = None if seed is None else int(seed)
        self.requested_weights = None if weights is None else np.asarray(weights, float)
        self.degeneracy_tol = float(degeneracy_tol)
        self.on_split = on_split
        self.enforce_kramers = bool(enforce_kramers)

        self._dets: Optional[Determinants] = None
        self._counts: Optional[np.ndarray] = None
        self._conn = None
        self._key: Optional[str] = None
        self._candidate: Optional[Tuple[str, Determinants, Any, Any]] = None
        #: The most recent solve, for callers that want the spectrum or the CI vectors rather
        #: than the RDMs the optimizer asked for.
        self.last = None
        #: The most recent recovery, for callers that want the yield rather than the space.
        self.last_recovery = None
        #: Provenance of the most recent sampling call — the provenance rule: an energy whose record
        #: does not say which device and how many shots produced it is not interpretable.
        self.provenance: Optional[BackendProvenance] = None
        self.n_solves = 0
        self.n_samples = 0
        self.n_proposals = 0
        self.n_adoptions = 0

    # -- sampling -------------------------------------------------------------------------

    def draw_seed(self) -> Optional[int]:
        """Seed for the **next** sampling call: ``seed + n_samples``, or ``None``.

        ⚠ **Advancing the seed between draws is not a detail, it is what makes ``propose``
        able to do anything at all.** With a fixed per-call seed the backend replays the
        identical shot record every time — and since the Stage-A/B circuits do not depend on
        the integrals either (see the class docstring's warning), every proposal reproduces
        the incumbent space exactly, returns ``None``, and the event machinery spends samples
        to learn nothing. That was measured, not imagined: 0 adoptions out of every proposal
        on a Ti(2+) CAS(2,10) run.

        Derived from the constructor's seed rather than drawn from entropy, so the whole
        trajectory is still replayable from one number — the provenance-travels-with-the-result rule — and every
        draw's own seed is recorded in its :class:`~kuiva.qc.backend.BackendProvenance`.
        """
        return None if self.seed is None else self.seed + self.n_samples

    def sample_space(self, n_qubit: int, *, prior=None, shots: Optional[int] = None,
                     h: Optional[np.ndarray] = None, eri: Optional[np.ndarray] = None):
        """Draw shots, recover configurations, return a
        :class:`~kuiva.qc.recovery.RecoveryResult`.

        Public because the sampler is worth studying on its own — the yield, the subspace
        fraction and the shot ladder are the honest characterization of this algorithm, and
        they should not require driving a whole CASSCF to see.

        ``h`` and ``eri`` are handed to the strategy so an integral-dependent circuit is
        possible through the same seam. ⚠ Neither Stage-A nor Stage-B uses them, which is the
        limitation recorded in the class docstring, not an unused argument.
        """
        shots = self.shots if shots is None else int(shots)
        plan = self.strategy.plan(n_qubit, n_elec=self.n_elec, h=h, eri=eri)
        allocation = plan.allocate(shots)
        seed = self.draw_seed()
        masks, counts = [], []
        with timer("SQD sampling"):
            for index, (circuit, n) in enumerate(zip(plan.circuits, allocation)):
                # ⚠ Offset per circuit as well as per draw: two identical circuits in one plan
                # would otherwise replay the same record, which is the same defect
                # :meth:`draw_seed` exists for, one level down.
                result = self.backend.sample(
                    circuit, n, seed=None if seed is None else seed + index)
                masks.append(result.masks)
                counts.append(result.counts)
                # ⚠ The provenance of the LAST circuit in the plan. Exact for the single-circuit
                # plans every current strategy produces; a multi-circuit strategy that wants
                # per-circuit provenance has to widen this rather than assume it.
                self.provenance = result.provenance
        self.n_samples += 1
        recovered = recover_configurations(
            np.concatenate(masks), np.concatenate(counts), self.n_elec, n_qubit,
            policy=self.recovery, max_determinants=self.max_determinants, prior=prior)
        self.last_recovery = recovered
        return recovered

    # -- the solve ------------------------------------------------------------------------

    def _active(self, ints):
        h = np.ascontiguousarray(ints.h_active_effective())
        eri = np.asarray(ints.active_eri())
        return h, eri, float(getattr(ints, "e_core", 0.0))

    def _install(self, dets: Determinants, counts: np.ndarray, conn=None) -> None:
        """Make ``dets`` the incumbent space, carrying the shot evidence behind it.

        The connection search is the expensive classical part and depends only on the
        determinants, not on the integrals, so it is done once here rather than per solve —
        the same trade ``CheapCISolver._install`` makes; a proposal that already paid for it
        hands it over rather than repeating it.
        """
        res.require("SQD subspace ({} determinants)".format(dets.ndet),
                    subspace_gb(dets.ndet, dets.n_spinor, self.n_states),
                    note="{} spinors, {} electrons, {} states".format(
                        dets.n_spinor, dets.n_elec, self.n_states),
                    advice=["lower max_determinants",
                            "fewer shots, or a circuit strategy that concentrates more "
                            "(kuiva.qc.ansatz)"])
        self._dets = dets
        self._counts = np.ascontiguousarray(counts, dtype=np.int64)
        self._conn = connections(dets) if conn is None else conn
        self._key = array_key(dets.masks)
        self._candidate = None

    def _prior(self):
        """The incumbent space and its accumulated shot counts, or ``None``."""
        if not self.accumulate or self._dets is None:
            return None
        return (self._dets.masks, self._counts)

    def _diagonalize(self, dets: Determinants, conn, h, eri, e_core: float):
        """Solve in a **given** space and apply the state-averaging gate. No sampling, no selection.

        The eigensolve is ``mcscf.preopt.solve_fixed_space``: reuse rather than reinvention,
        and deliberately so — that routine carries the hard-won ARPACK hardening (a subspace
        large enough for a Kramers-degenerate pair to separate, a tolerance set from an
        absolute energy target, a dense fallback, and :class:`SolverFailure` as the last
        resort) which a Kramers-structured, state-averaged subspace needs exactly as much as
        the cheap CI does.

        ⚠ **The equalized weights produce the reported energy *and* the RDMs.** Taking the
        energy from the requested weights and the densities from the equalized ones is a
        discrepancy visible only when a block is genuinely degenerate — i.e. always in
        production and never in a test on random integrals.
        """
        if self.n_states > dets.ndet:
            raise SolverFailure(
                "the recovered subspace holds {} determinants but {} states were asked for; "
                "sample more shots or use a circuit strategy that spreads further"
                .format(dets.ndet, self.n_states))
        ci = solve_fixed_space(dets, h, eri, n_states=self.n_states,
                               state_weights=self.requested_weights, conn=conn)
        if self.enforce_kramers and self.n_states > 1:
            weights = state_average_weights(ci.energies, self.n_elec, self.requested_weights,
                                            tol=self.degeneracy_tol, on_split=self.on_split)
            if not np.allclose(weights, ci.weights, rtol=0.0, atol=1e-14):
                gamma, gamma2 = rdm12(dets, ci.civecs, weights, conn)
                ci = replace(ci, gamma=gamma, gamma2=gamma2, weights=weights)
        energy = float(np.dot(ci.weights, ci.energies)) + e_core
        return energy, ci

    # -- the AdaptiveCISolver contract ------------------------------------------------------

    def solve(self, ints):
        """``(energy, gamma, Gamma)`` in the **incumbent** subspace — the ci_solver contract.

        ⚠ Samples exactly once, on the first call, because there has to be an incumbent space
        before there can be a fixed one. Every later call is a deterministic function of the
        integrals, which is the promise :mod:`kuiva.mcscf.events` rests on.
        """
        h, eri, e_core = self._active(ints)
        n_qubit = int(h.shape[0])
        if self._dets is None:
            first = self.sample_space(n_qubit, h=h, eri=eri)
            self._install(first.determinants, first.counts)
        elif self._dets.n_spinor != n_qubit:
            raise ValueError("these integrals carry {} active spinors; the incumbent subspace "
                             "was sampled for {}".format(n_qubit, self._dets.n_spinor))
        energy, ci = self._diagonalize(self._dets, self._conn, h, eri, e_core)
        self.n_solves += 1
        self.last = ci
        return energy, ci.gamma, ci.gamma2

    __call__ = solve

    def propose(self, ints) -> Optional[Proposal]:
        """A freshly sampled subspace at these integrals, evaluated but **not** adopted."""
        h, eri, e_core = self._active(ints)
        n_qubit = int(h.shape[0])
        self.n_proposals += 1
        recovered = self.sample_space(n_qubit, prior=self._prior(), h=h, eri=eri)
        dets = recovered.determinants
        key = array_key(dets.masks)
        if key == self._key:
            return None                     # the draw reproduced the incumbent space
        conn = connections(dets)
        energy, ci = self._diagonalize(dets, conn, h, eri, e_core)
        self._candidate = (key, recovered, conn, ci)
        return Proposal(energy=energy, gamma=ci.gamma, gamma2=ci.gamma2, key=key,
                        label=self._label(recovered))

    def adopt(self, key: Hashable) -> None:
        if self._candidate is None or self._candidate[0] != key:
            raise ValueError("no proposal with key {!r} is pending; adopt() takes the key of "
                             "the most recent propose()".format(key))
        _, recovered, conn, ci = self._candidate
        self._install(recovered.determinants, recovered.counts, conn=conn)
        self.last = ci
        self.n_adoptions += 1

    def space_key(self) -> Optional[str]:
        """Identity of the incumbent subspace, or ``None`` before the first solve.

        ⚠ The **determinant list**, not the shot counts or the seed: two draws that recovered
        the same configurations span the same space and therefore *are* the same surface, and
        keying on anything finer would declare a chart change at every proposal and clear the
        curvature memory for nothing (the chart-scoped-curvature rule, the same one ``DMRGSolver`` follows in
        keying on caps rather than on observed bond dimensions).
        """
        return self._key

    # -- reporting -------------------------------------------------------------------------

    def _label(self, recovered) -> str:
        if self._dets is None:
            return "{} configurations".format(recovered.ndet)
        shared = int(np.intersect1d(recovered.determinants.masks, self._dets.masks).size)
        return "{}/{} shared".format(shared, recovered.ndet)

    @property
    def n_determinants(self) -> int:
        return 0 if self._dets is None else self._dets.ndet

    def coverage(self) -> float:
        """Incumbent subspace as a fraction of the complete CAS space (see "⚠ What a good
        result here does and does not prove" in the module docstring). ``1.0`` means this
        solver is a full CI."""
        if self._dets is None:
            return 0.0
        return subspace_fraction(self._dets.ndet, self._dets.n_spinor, self.n_elec)

    def report(self) -> Dict[str, Any]:
        """Everything a caller needs to state what produced an energy, as plain data."""
        return {
            "algorithm": "sqd",
            "n_determinants": self.n_determinants,
            "coverage": self.coverage(),
            "shots": self.shots,
            "strategy": getattr(self.strategy, "__class__").__name__,
            "recovery": self.recovery,
            "accumulate": self.accumulate,
            "n_solves": self.n_solves,
            "n_samples": self.n_samples,
            "n_proposals": self.n_proposals,
            "n_adoptions": self.n_adoptions,
            "yield": (None if self.last_recovery is None
                      else self.last_recovery.yield_fraction),
            "provenance": (None if self.provenance is None
                           else self.provenance.as_dict()),
        }

    def __repr__(self) -> str:
        return "SQDSolver(n_elec={}, n_states={}, ndet={}, backend={}, key={})".format(
            self.n_elec, self.n_states, self.n_determinants,
            getattr(self.backend, "name", "?"),
            None if self._key is None else self._key[:8])


__all__ = ["DEFAULT_SHOTS", "REQUIRED_PRIMITIVES", "SQDSolver", "subspace_gb"]
