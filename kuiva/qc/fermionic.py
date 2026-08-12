"""Compiling fermionic operators into circuits: Pauli exponentials, Givens networks, Trotter.

**Orchestration and Pauli algebra, not a registered kernel.** It runs once per circuit build, and
what it costs is dwarfed by the sampling or the eigensolve around it.

What this module is for
-----------------------
Everything above the mapping layer needs the same primitive: *given an operator, give me a
circuit that exponentiates it*. A Stage-C UCC ansatz exponentiates excitation generators, a
cluster-Jastrow ansatz exponentiates one-body rotations and number-number interactions, and
SKQD exponentiates the Hamiltonian itself. Writing that three times would be three chances to
get a sign wrong in a way that is Hermitian, plausible and invisible — Hermitian, plausible and invisible is the worst shape — so it is
written once, here, and validated against ``scipy.linalg.expm`` of the dense operator.

The one primitive, and its convention
--------------------------------------
``exp(-i theta P / 2)`` for a Pauli string ``P``, built from the six-gate vocabulary of
:mod:`kuiva.qc.circuits`:

1. rotate each qubit's Pauli into ``Z`` — ``h`` for an ``X`` factor, ``rx(pi/2)`` for a ``Y``;
2. a ``cx`` ladder up the support, accumulating parity into the highest qubit;
3. ``rz(theta)`` there;
4. the ladder and the basis change, undone.

⚠ **The ``Y`` basis change is the one thing here that cannot be argued, only checked.**
``Rx(pi/2) Z Rx(pi/2)^dag = -Y``, so the rotation that takes ``Y`` into ``Z`` is ``rx(+pi/2)``
applied *first* and ``rx(-pi/2)`` after — the opposite order from the naive reading, and the
two differ by the sign of every ``Y``-carrying term. ``tests/test_qc_fermionic.py`` compares
against ``expm`` for every Pauli string on three qubits for exactly this reason.

⚠ **An identity string is skipped and its phase returned, never silently dropped.**
``exp(-i theta I / 2)`` is a global phase, which the vocabulary cannot express (and which no
measurement can see). :class:`CompiledCircuit` carries it so a statevector comparison stays an
equality rather than a similarity, and so a caller who *does* care — a controlled application,
a Krylov overlap — has the number rather than a footnote.

Exactness: when a Trotter product is not an approximation
----------------------------------------------------------
``exp(sum_t A_t)`` factorizes into ``prod_t exp(A_t)`` **exactly** when the terms commute.
That is not a rare special case here, it is the normal one for an *ansatz*:

* a single excitation generator ``t a_p^dag a_q - h.c.`` maps to **2** Pauli strings, and they
  commute;
* a double excitation generator maps to **8**, and they all commute;
* a Jastrow ``sum J_pq n_p n_q`` is diagonal, so everything commutes.

So every ansatz circuit in :mod:`kuiva.qc.ansatz` is an *exact* exponential of its generator,
and only the product **across** generators (disentangled UCC) and the time evolution of the
Hamiltonian (SKQD) carry Trotter error. :func:`exponential_circuit` takes ``require_exact`` and
checks the claim through :meth:`~kuiva.qc.mapping.PauliSum.all_commute` rather than asserting
it in prose.

Orbital rotations are compiled exactly, not Trotterized
--------------------------------------------------------
A one-body rotation ``exp(sum_pq kappa_pq a_p^dag a_q)`` is the single most useful fermionic
unitary — it is what an orbital rotation *is*, it is the outer layer of every cluster-Jastrow
ansatz, and Trotterizing it would be both approximate and deep. :func:`orbital_rotation_circuit`
instead decomposes the ``n x n`` mode unitary into **adjacent-mode** Givens rotations plus final
phases (Reck-style triangular elimination) and emits each exactly. Adjacency is what makes the
Jordan-Wigner string vanish: modes ``p`` and ``p+1`` have nothing between them, so a two-qubit
rotation on neighbouring qubits is the whole operation — which is also why the circuit maps onto
a linear device with no routing.

⚠ **The global phase is real and is returned.** The mode-phase factors are emitted as
``rz(a)`` on one mode and ``rz(-a)`` on its partner, which is exactly phase-free by
construction; the *final* diagonal cannot be, and its phase is ``arg(det u)/2``. Anyone
comparing amplitudes against the Slater-determinant minors ``det(u[out, in])`` needs it.

⚠ **What is deliberately not here: double factorization.** The efficient Trotter step for a
chemistry Hamiltonian writes the two-body operator as ``sum_mu lambda_mu O_mu^2`` with each
``O_mu`` one-body, so each factor becomes *orbital rotation, diagonal phases, rotation back* —
using precisely this module's Givens machinery, and reducing an SKQD step from ``O(n^4)`` Pauli
exponentials to ``O(n)`` rotations. It is not implemented, and the reason is a trap worth
recording rather than rediscovering: with **complex** spinor integrals the factorization is a
Takagi problem, not an eigendecomposition. ``M[(pq),(rs)] = (pq|rs)`` is complex *symmetric*;
what is Hermitian is ``N = M S`` with ``S`` the index swap, and ``N`` obeys ``N* = S N S``, so
its eigenvectors must be chosen adapted to that antiunitary before the ``L^mu`` come out
Hermitian and the ``exp(-i t O_mu^2)`` trick applies at all. ``numpy.linalg.eigh`` will not do
that for you, and an unadapted eigenbasis yields a non-Hermitian ``L^mu``, a non-unitary
"rotation", and a plausible wrong circuit. Deferred deliberately: it is a *cost* optimization
whose benefit appears at hardware scale, and the generic path below is correct at every scale.

References
----------------------------
* The Pauli-exponential circuit construction: M. A. Nielsen, I. L. Chuang, "Quantum Computation
  and Quantum Information", Cambridge (2010), sec. 4.7.3.
* Adjacent-mode Givens networks for fermionic basis rotations, and their linear depth and
  connectivity: I. D. Kivlichan, J. McClean, N. Wiebe, C. Gidney, A. Aspuru-Guzik, G. K.-L.
  Chan, R. Babbush, "Quantum simulation of electronic structure with linear depth and
  connectivity", *Phys. Rev. Lett.* **120**, 110501 (2018),
  doi:10.1103/PhysRevLett.120.110501. The triangular elimination itself is M. Reck, A.
  Zeilinger, H. J. Bernstein, P. Bertani, *Phys. Rev. Lett.* **73**, 58 (1994),
  doi:10.1103/PhysRevLett.73.58; the depth-optimal rectangular variant is W. R. Clements et
  al., *Optica* **3**, 1460 (2016), doi:10.1364/OPTICA.3.001460, and is not used here.
* Trotter product formulas: H. F. Trotter, *Proc. Am. Math. Soc.* **10**, 545 (1959);
  M. Suzuki, *Commun. Math. Phys.* **51**, 183 (1976), doi:10.1007/BF01609348.
* Unitary coupled cluster and its disentangled (single-Trotter-step) circuit form: J. Romero,
  R. Babbush, J. R. McClean, C. Hempel, P. J. Love, A. Aspuru-Guzik, *Quantum Sci. Technol.*
  **4**, 014008 (2018), doi:10.1088/2058-9565/aad3e4; F. A. Evangelista, G. K.-L. Chan, G. E.
  Scuseria, "Exact parameterization of fermionic wave functions via unitary coupled cluster
  theory", *J. Chem. Phys.* **151**, 244112 (2019), doi:10.1063/1.5133059 — the statement that
  the disentangled product is a *different* (and still universal) ansatz, not an approximation
  to a single exponential, which is why this module never calls it one.
* Double factorization, deferred above: M. Motta et al., "Low rank representations for quantum
  simulation of electronic structure", *npj Quantum Inf.* **7**, 83 (2021),
  doi:10.1038/s41534-021-00416-z.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..util.logging import get_logger
from .circuits import CircuitSpec, Gate
from .mapping import PauliSum, fermionic_operator

log = get_logger(__name__)

_U64 = np.uint64

#: Below this, an angle is not worth a gate. Twelve orders below the 1e-8 Eh suite tolerance and the
#: same spirit as ``mapping.INTEGRAL_SCREEN_TOL``: it removes structural zeros (an amplitude
#: that is exactly zero by symmetry) and nothing that carries physics. It is a screening
#: parameter, not an accuracy knob.
ANGLE_SCREEN_TOL = 1e-14


# --- A circuit that remembers where its parameters went ---------------------------------------

@dataclass(frozen=True)
class CompiledCircuit:
    """A circuit, the global phase it could not express, and its parameter map.

    ⚠ **The phase convention, stated once:** the operator this circuit was asked to realize is

        ``target = exp(1j * global_phase) * U_circuit``

    so a consumer that needs an exact statevector multiplies the simulated one by
    ``exp(1j * global_phase)``. A sampler never needs it (``|psi|^2`` is blind to it) and a VQE
    never needs it (``<psi|H|psi>`` is too); it exists so that the *tests* can be equalities.

    Attributes
    ----------
    circuit : CircuitSpec
    global_phase : float
    gate_param : ndarray (n_gates,) int64
        Which ansatz parameter drives each gate's angle, or ``-1`` where the gate is fixed.
    gate_scale : ndarray (n_gates,) float64
        ``d(gate angle) / d(parameter)``. ⚠ Always **linear**, and that is a requirement rather
        than an observation: it is what makes the parameter-shift rule of
        :func:`parameter_shift_gradient` exact. A builder that made a gate angle a nonlinear
        function of a parameter would break the gradient silently, so nothing here offers one.
    n_params : int
    """

    circuit: CircuitSpec
    global_phase: float = 0.0
    gate_param: Optional[np.ndarray] = None
    gate_scale: Optional[np.ndarray] = None
    n_params: int = 0

    def __post_init__(self) -> None:
        n = self.circuit.n_gates
        param = (np.full(n, -1, dtype=np.int64) if self.gate_param is None
                 else np.ascontiguousarray(self.gate_param, dtype=np.int64))
        scale = (np.zeros(n, dtype=np.float64) if self.gate_scale is None
                 else np.ascontiguousarray(self.gate_scale, dtype=np.float64))
        if param.shape != (n,) or scale.shape != (n,):
            raise ValueError("gate_param/gate_scale must have one entry per gate ({})".format(n))
        object.__setattr__(self, "gate_param", param)
        object.__setattr__(self, "gate_scale", scale)
        object.__setattr__(self, "global_phase", float(self.global_phase))
        object.__setattr__(self, "n_params", int(self.n_params))

    @property
    def n_qubit(self) -> int:
        return self.circuit.n_qubit

    def parameterized_gates(self) -> np.ndarray:
        """Indices of the gates whose angle is driven by a parameter."""
        return np.flatnonzero(self.gate_param >= 0)

    def shifted(self, gate: int, delta: float) -> CircuitSpec:
        """The circuit with **one** gate's angle moved by ``delta``.

        The whole mechanism behind :func:`parameter_shift_gradient`: every parameterized gate
        in this vocabulary is ``exp(-i theta P / 2)`` with ``P^2 = I``, so shifting its angle
        by ``+-pi/2`` gives the derivative exactly — no step size, no finite-difference error.
        """
        gate = int(gate)
        old = self.circuit.gates[gate]
        if len(old.params) != 1:
            raise ValueError("gate {} ({!r}) carries no angle to shift".format(gate, old))
        gates = list(self.circuit.gates)
        gates[gate] = Gate(old.name, old.qubits, (old.params[0] + float(delta),))
        return CircuitSpec(self.circuit.n_qubit, tuple(gates))

    def __repr__(self) -> str:
        return "CompiledCircuit({!r}, phase={:.6g}, n_params={})".format(
            self.circuit, self.global_phase, self.n_params)


class CircuitBuilder:
    """Accumulates gates, the global phase, and the parameter map.

    Not a public abstraction so much as the only way to keep the parameter bookkeeping beside
    the gate that carries it — a separate pass that *recomputed* which gate belongs to which
    parameter would be a second convention, and the gradient would be wrong rather than
    obviously wrong.
    """

    def __init__(self, n_qubit: int) -> None:
        self.n_qubit = int(n_qubit)
        self._gates: List[Gate] = []
        self._param: List[int] = []
        self._scale: List[float] = []
        self.global_phase = 0.0

    def add(self, name: str, qubits: Sequence[int], params: Sequence[float] = (), *,
            param_index: int = -1, scale: float = 0.0) -> "CircuitBuilder":
        self._gates.append(Gate(name, tuple(qubits), tuple(params)))
        self._param.append(int(param_index))
        self._scale.append(float(scale))
        return self

    def extend(self, other: "CircuitBuilder") -> "CircuitBuilder":
        if other.n_qubit != self.n_qubit:
            raise ValueError("cannot append a {}-qubit block to a {}-qubit circuit"
                             .format(other.n_qubit, self.n_qubit))
        self._gates.extend(other._gates)
        self._param.extend(other._param)
        self._scale.extend(other._scale)
        self.global_phase += other.global_phase
        return self

    def prepare(self, mask: int) -> "CircuitBuilder":
        """Prepare the occupation mask ``|mask>`` — ``x`` gates, per the mask convention."""
        for gate in CircuitSpec.prepare(int(mask), self.n_qubit).gates:
            self.add(gate.name, gate.qubits)
        return self

    def result(self, n_params: int = 0) -> CompiledCircuit:
        return CompiledCircuit(
            circuit=CircuitSpec(self.n_qubit, tuple(self._gates)),
            global_phase=self.global_phase,
            gate_param=np.array(self._param, dtype=np.int64),
            gate_scale=np.array(self._scale, dtype=np.float64),
            n_params=int(n_params))

    def __len__(self) -> int:
        return len(self._gates)


# --- The one primitive ------------------------------------------------------------------------

def pauli_exponential(builder: CircuitBuilder, x: int, z: int, theta: float, *,
                      param_index: int = -1, scale: float = 0.0) -> CircuitBuilder:
    """Append ``exp(-i theta P(x, z) / 2)`` to ``builder``. See the module docstring.

    An identity string (``x == z == 0``) contributes ``exp(-i theta / 2)``, which is added to
    the builder's global phase and emits no gate.
    """
    x, z, theta = int(x), int(z), float(theta)
    support = [k for k in range(builder.n_qubit) if (x >> k) & 1 or (z >> k) & 1]
    if not support:
        builder.global_phase += -0.5 * theta
        return builder
    # 1. basis change into Z on every qubit of the support. Applied FIRST, so this is V^dag.
    for k in support:
        has_x, has_z = (x >> k) & 1, (z >> k) & 1
        if has_x and has_z:
            builder.add("rx", (k,), (0.5 * np.pi,))      # Y -> Z; see the docstring warning
        elif has_x:
            builder.add("h", (k,))                       # X -> Z
    # 2. parity ladder up the support.
    for a, b in zip(support[:-1], support[1:]):
        builder.add("cx", (a, b))
    # 3. the rotation itself — the only gate that carries the angle.
    builder.add("rz", (support[-1],), (theta,), param_index=param_index, scale=scale)
    # 4. and back.
    for a, b in zip(support[-2::-1], support[:0:-1]):
        builder.add("cx", (a, b))
    for k in support:
        has_x, has_z = (x >> k) & 1, (z >> k) & 1
        if has_x and has_z:
            builder.add("rx", (k,), (-0.5 * np.pi,))
        elif has_x:
            builder.add("h", (k,))
    return builder


def pauli_exponential_circuit(x: int, z: int, theta: float, n_qubit: int) -> CompiledCircuit:
    """``exp(-i theta P(x, z) / 2)`` as a standalone circuit."""
    return pauli_exponential(CircuitBuilder(n_qubit), x, z, theta).result()


def exponential_circuit(builder: CircuitBuilder, operator: PauliSum, scale: float = 1.0, *,
                        require_exact: bool = False, param_index: int = -1,
                        param_scale: float = 1.0,
                        tol: float = ANGLE_SCREEN_TOL) -> CircuitBuilder:
    """Append ``prod_t exp(-i scale c_t P_t)`` — the first-order product of the terms.

    Equals ``exp(-i scale * operator)`` **exactly** when the terms commute, and is a first-order
    Trotter approximation otherwise. ``require_exact=True`` checks the commutation rather than
    trusting the caller (see the module docstring); it is ``O(n_terms^2)`` and belongs on an
    ansatz generator, not on a Hamiltonian.

    ``param_index >= 0`` marks every emitted rotation as driven by that ansatz parameter, with
    the chain rule ``d(gate angle)/d(param) = 2 c_t * param_scale`` — used when ``scale`` is
    ``param_scale`` times the parameter. ⚠ ``param_scale`` is not decoration: a block emitted
    with a *negated* angle (the inverse half of a conjugation) has a negated derivative too,
    and a gradient that missed it would be plausible and wrong.
    """
    if require_exact and not operator.all_commute():
        raise ValueError(
            "exponential_circuit(require_exact=True) was given {} Pauli strings that do not all "
            "commute, so the product is a Trotter approximation and not the exponential it was "
            "asked for. Split the operator into commuting groups, or use trotter_circuit() and "
            "say so.".format(operator.n_terms))
    for c, x, z in zip(operator.coeffs, operator.x_masks, operator.z_masks):
        angle = 2.0 * float(scale) * float(c)
        if abs(angle) <= tol and param_index < 0:
            continue
        pauli_exponential(builder, int(x), int(z), angle,
                          param_index=param_index, scale=2.0 * float(c) * float(param_scale))
    return builder


def trotter_circuit(operator: PauliSum, time: float, *, steps: int = 1,
                    reference: Optional[int] = None) -> CompiledCircuit:
    """``exp(-i * operator * time)`` by a first-order product formula in ``steps`` slices.

    ⚠ **First order, and the error is stated rather than tuned away:** the deviation from the
    true evolution is ``O(t^2 / steps)`` with a prefactor set by the commutators of the terms,
    and for a chemistry Hamiltonian that prefactor is not small. A second-order (symmetric)
    formula would halve nothing about the *scaling* of the term count, which is what actually
    binds here; the honest control is ``steps``, and
    ``tests/test_qc_fermionic.py::test_trotter_error_falls_as_one_over_steps`` measures it.

    ⚠ **The product does not conserve particle number even when ``operator`` does.** ``H``
    commutes with ``N``; the individual Pauli strings it decomposes into do not, so each factor
    moves a little weight off the sector and only the exact sum puts it back. The leak is a
    Trotter error and falls with ``steps`` like any other, but it is not zero, and a caller who
    assumed "the Hamiltonian conserves N, so the circuit does" would be wrong. A
    number-conserving decomposition — over Hermitian *fermionic* terms, or via the double
    factorization this module's docstring defers — is what removes it.

    ``reference`` prepends the preparation of an occupation mask, so the result is
    ``exp(-iHt)|ref>`` — the state a sample-based Krylov method measures.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be at least 1, got {}".format(steps))
    builder = CircuitBuilder(operator.n_qubit)
    if reference is not None:
        builder.prepare(int(reference))
    for _ in range(steps):
        exponential_circuit(builder, operator, float(time) / steps)
    return builder.result()


# --- Fermionic generators ----------------------------------------------------------------------
#
# Each returns the **Hermitian** operator ``A`` for which ``exp(-i A)`` is the intended unitary.
# An excitation generator ``G = T - T^dag`` is anti-Hermitian, so ``A = i G``; passing ``G``
# itself to the mapping would (correctly) be refused as non-Hermitian, which is the check
# working rather than an inconvenience.

def single_excitation_generator(p: int, q: int, amplitude: complex,
                                n_qubit: int) -> PauliSum:
    """``A`` with ``exp(-i A) = exp(t a_p^dag a_q - t^* a_q^dag a_p)``, ``t = amplitude``.

    ⚠ **The amplitude is complex, and that is the whole point.** Every published UCC/LUCJ
    implementation takes it real, because a spin-separable non-relativistic Hamiltonian has
    real amplitudes; Kuiva's does not, and the imaginary part is what lets the ansatz
    mix Kramers partners — the same thing the complex ``kappa`` of ``mcscf/orbopt.py`` does for
    the classical orbital rotation.
    """
    t = complex(amplitude)
    return fermionic_operator([1j * t, -1j * np.conj(t)],
                              [[int(p), int(q)], [int(q), int(p)]],
                              (True, False), int(n_qubit))


def double_excitation_generator(p: int, q: int, r: int, s: int, amplitude: complex,
                                n_qubit: int) -> PauliSum:
    """``A`` with ``exp(-i A) = exp(t a_p^dag a_q^dag a_r a_s - h.c.)``.

    The excitation takes the pair of modes ``(s, r)`` to ``(p, q)``. Complex ``t``, for the
    reason in :func:`single_excitation_generator`.
    """
    t = complex(amplitude)
    return fermionic_operator([1j * t, -1j * np.conj(t)],
                              [[int(p), int(q), int(r), int(s)],
                               [int(s), int(r), int(q), int(p)]],
                              (True, True, False, False), int(n_qubit))


@lru_cache(maxsize=4096)
def _real_excitation_generator(creations: Tuple[int, ...], annihilations: Tuple[int, ...],
                              n_qubit: int) -> PauliSum:
    """``i (D - D^dag)`` for a unit real amplitude, cached on the mode pattern.

    Memoized because a VQE rebuilds the *same* ansatz thousands of times with different angles,
    and the Pauli algebra depends only on the mode pattern — measured at roughly half the cost
    of one energy evaluation before it was cached. ⚠ A pure-function cache in orchestration
    code, not a kernel: the kernel contract's B2 rule forbids hash-based addressing *inside* a kernel, and
    nothing here is one.
    """
    return fermionic_operator(
        [1j, -1j],
        [list(creations) + list(annihilations),
         list(reversed(annihilations)) + list(reversed(creations))],
        tuple([True] * len(creations) + [False] * len(annihilations)),
        int(n_qubit))


def excitation_circuit(builder: CircuitBuilder, creations: Sequence[int],
                       annihilations: Sequence[int], magnitude: float, phase: float = 0.0, *,
                       magnitude_index: int = -1, phase_index: int = -1) -> CircuitBuilder:
    """Append ``exp(t D - t^* D^dag)`` **exactly**, with ``t = magnitude * exp(i * phase)``
    and ``D = a_c1^dag ... a_a1 ...``.

    ⚠ **This is where the literature stops transferring, and the finding is measured rather
    than argued** (the research content of this package). With a *real* amplitude an excitation generator maps
    to 2 (singles) or 8 (doubles) Pauli strings and **they all commute**, which is exactly why
    every published UCC circuit is a plain product of Pauli exponentials. With a **complex**
    amplitude — which a spin-orbit-coupled spinor Hamiltonian forces — the count doubles to 4
    and 16, and the two halves **do not commute with each other**: the real part contributes
    ``XY-YX``-type strings and the imaginary part ``XX+YY``-type ones, which anticommute on the
    qubits where they differ. A naive product would therefore be a *Trotter approximation* of
    something the literature treats as exact, and nothing in the resulting energy would say so.
    ``tests/test_qc_fermionic.py`` pins both halves of that statement.

    The exact route is a conjugation rather than a longer product. ``D`` raises the occupation
    of its first creation mode ``p`` by one, so a phase gate there rotates its amplitude::

        exp(i phi n_p) D exp(-i phi n_p) = exp(i phi) D
        =>  exp(t D - t^* D^dag) = exp(i phi n_p) exp(|t| (D - D^dag)) exp(-i phi n_p)

    — a **real** generator (commuting strings, exact product) between two ``rz`` gates whose
    global phases cancel exactly. Two extra single-qubit gates buy back exactness; the price is
    that the two parameters of a complex amplitude are ``(magnitude, phase)`` rather than
    ``(real, imaginary)``, which is also the better VQE parameterization since ``phase = 0``
    recovers the real-amplitude ansatz the literature optimizes.

    ``magnitude_index`` / ``phase_index`` mark the emitted rotations for
    :func:`parameter_shift_gradient`.
    """
    creations = [int(c) for c in creations]
    annihilations = [int(a) for a in annihilations]
    if not creations or len(creations) != len(annihilations):
        raise ValueError("an excitation needs equally many creations and annihilations, got "
                         "{} and {}".format(creations, annihilations))
    # Any creation mode absent from the annihilations will do as the phase carrier: the
    # conjugation only needs *one* mode whose occupation D changes by exactly +1.
    carrier = next((c for c in creations if c not in annihilations), None)
    if carrier is None:
        raise ValueError(
            "every mode this excitation creates it also annihilates ({} -> {}), so it changes "
            "no occupation by one and the phase-gate conjugation that makes a complex "
            "amplitude exact does not apply. Such a generator is a number operator in "
            "disguise; drop it from the ansatz rather than Trotter-splitting it."
            .format(annihilations, creations))
    if abs(phase) > ANGLE_SCREEN_TOL or phase_index >= 0:
        builder.add("rz", (carrier,), (-float(phase),), param_index=phase_index, scale=-1.0)
    exponential_circuit(builder,
                        _real_excitation_generator(tuple(creations), tuple(annihilations),
                                                   builder.n_qubit),
                        float(magnitude), require_exact=True, param_index=magnitude_index)
    if abs(phase) > ANGLE_SCREEN_TOL or phase_index >= 0:
        builder.add("rz", (carrier,), (float(phase),), param_index=phase_index, scale=1.0)
    return builder


def one_body_generator(kappa: np.ndarray) -> PauliSum:
    """``A`` with ``exp(-i A) = exp(sum_pq kappa_pq a_p^dag a_q)`` for anti-Hermitian ``kappa``.

    Refuses a ``kappa`` that is not anti-Hermitian rather than projecting it: the projection
    would give a unitary, of plausible magnitude, that is not the rotation the caller asked
    for. (``mcscf/orbopt.py``'s ``unitary_from_antihermitian`` makes the same demand of the
    classical path.)
    """
    kappa = np.ascontiguousarray(kappa, dtype=np.complex128)
    n = int(kappa.shape[0])
    if kappa.shape != (n, n):
        raise ValueError("kappa must be square, got {}".format(kappa.shape))
    defect = float(np.abs(kappa + kappa.conj().T).max()) if n else 0.0
    if defect > 1e-12 * max(1.0, float(np.abs(kappa).max())):
        raise ValueError("kappa must be anti-Hermitian (max |kappa + kappa^dag| = {:.3e}); a "
                         "Hermitian part would exponentiate to something that is not unitary "
                         "on the mode space".format(defect))
    p, q = (idx.reshape(-1) for idx in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
    return fermionic_operator(1j * kappa[p, q], np.stack([p, q], axis=1), (True, False), n)


def jastrow_generator(coupling: np.ndarray) -> PauliSum:
    """``A`` with ``exp(-i A) = exp(i sum_{p <= q} J_pq n_p n_q)`` for real symmetric ``J``.

    The diagonal half of a cluster-Jastrow ansatz. Every term is a product of number operators,
    hence diagonal, hence mutually commuting — so :func:`exponential_circuit` realizes it
    exactly and ``require_exact=True`` passes.
    """
    j = np.ascontiguousarray(coupling, dtype=np.float64)
    n = int(j.shape[0])
    if j.shape != (n, n):
        raise ValueError("the Jastrow coupling must be square, got {}".format(j.shape))
    if float(np.abs(j - j.T).max() if n else 0.0) > 1e-12 * max(1.0, float(np.abs(j).max())):
        raise ValueError("the Jastrow coupling must be symmetric: n_p n_q = n_q n_p, so an "
                         "antisymmetric part is discarded by the physics and would make the "
                         "parameter count a fiction")
    rows, coeffs = [], []
    for p in range(n):
        for q in range(p, n):
            if abs(j[p, q]) <= ANGLE_SCREEN_TOL:
                continue
            rows.append([p, p, q, q])
            # exp(+i J n n) = exp(-i A) => A = -J n_p n_q.
            coeffs.append(-j[p, q])
    if not rows:
        return PauliSum(np.zeros(0), np.zeros(0, _U64), np.zeros(0, _U64), n)
    return fermionic_operator(np.array(coeffs, dtype=np.complex128), np.array(rows),
                              (True, False, True, False), n)


# --- Exact compilation of a one-body rotation ---------------------------------------------------

@dataclass(frozen=True)
class GivensRotation:
    """One adjacent-mode ``SU(2)``: modes ``(mode, mode + 1)``, ``D1 R(theta) D2``.

    ``D1 = diag(e^{i a}, e^{-i a})`` and ``D2 = diag(e^{i b}, e^{-i b})`` are traceless-generator
    phase pairs — chosen over the more obvious ``diag(1, e^{i phi})`` because ``rz(a)`` on one
    mode and ``rz(-a)`` on the other reproduce them with the global phases cancelling
    **exactly**, which keeps the compiled circuit's only phase defect in one place (the final
    diagonal) instead of accumulating one per rotation.
    """

    mode: int
    theta: float
    phase_a: float
    phase_b: float

    def matrix(self) -> np.ndarray:
        """The ``2 x 2`` mode-space unitary."""
        c, s = np.cos(self.theta), np.sin(self.theta)
        d1 = np.array([np.exp(1j * self.phase_a), np.exp(-1j * self.phase_a)])
        d2 = np.array([np.exp(1j * self.phase_b), np.exp(-1j * self.phase_b)])
        return d1[:, None] * np.array([[c, s], [-s, c]], dtype=np.complex128) * d2[None, :]


def _su2_parameters(g: np.ndarray) -> Tuple[float, float, float]:
    """``(theta, a, b)`` with ``g = diag(e^{ia},e^{-ia}) R(theta) diag(e^{ib},e^{-ib})``.

    Valid for any ``g`` in ``SU(2)`` written as ``[[alpha, beta], [-beta^*, alpha^*]]``:
    ``|alpha| = cos theta``, ``|beta| = sin theta``, ``a + b = arg alpha``, ``a - b = arg beta``.
    """
    alpha, beta = complex(g[0, 0]), complex(g[0, 1])
    theta = float(np.arctan2(abs(beta), abs(alpha)))
    arg_a = float(np.angle(alpha)) if abs(alpha) > 1e-14 else 0.0
    arg_b = float(np.angle(beta)) if abs(beta) > 1e-14 else 0.0
    return theta, 0.5 * (arg_a + arg_b), 0.5 * (arg_a - arg_b)


def givens_decomposition(u: np.ndarray) -> Tuple[Tuple[GivensRotation, ...], np.ndarray]:
    """Factor a unitary ``u`` into adjacent-mode rotations and a final diagonal.

    Returns ``(rotations, phases)`` with

        ``u = G_1^dag G_2^dag ... G_m^dag diag(exp(i * phases))``

    where ``rotations`` is ``(G_1, ..., G_m)`` in the order the elimination applied them. ⚠ The
    reconstruction is asserted in the tests: a decomposition that is *nearly* right produces a
    circuit that is unitary, particle-conserving and wrong.

    Triangular (Reck) elimination: zero the sub-diagonal column by column from the bottom,
    each time with a rotation on two neighbouring rows. ``O(n^2)`` rotations — not the
    depth-optimal rectangular arrangement, which buys parallelism on hardware and nothing here.
    """
    w = np.array(u, dtype=np.complex128, copy=True)
    n = int(w.shape[0])
    if w.shape != (n, n):
        raise ValueError("an orbital rotation must be square, got {}".format(w.shape))
    defect = float(np.abs(w.conj().T @ w - np.eye(n)).max()) if n else 0.0
    if defect > 1e-10:
        raise ValueError("the mode matrix is not unitary (max |u^dag u - 1| = {:.3e}); a "
                         "Givens network can only represent a unitary".format(defect))
    rotations: List[GivensRotation] = []
    for col in range(n - 1):
        for row in range(n - 1, col, -1):
            a, b = w[row - 1, col], w[row, col]
            r = float(np.hypot(abs(a), abs(b)))
            if r <= 1e-15:
                continue
            g = np.array([[np.conj(a) / r, np.conj(b) / r], [-b / r, a / r]],
                         dtype=np.complex128)
            w[row - 1:row + 1, :] = g @ w[row - 1:row + 1, :]
            theta, pa, pb = _su2_parameters(g)
            rotations.append(GivensRotation(mode=row - 1, theta=theta, phase_a=pa, phase_b=pb))
    phases = np.angle(np.diag(w))
    off = float(np.abs(w - np.diag(np.diag(w))).max()) if n else 0.0
    if off > 1e-9:
        raise ValueError("the elimination left {:.3e} off the diagonal; a triangular unitary "
                         "is diagonal, so this is a numerical failure rather than a "
                         "convention question".format(off))
    return tuple(rotations), np.ascontiguousarray(phases, dtype=np.float64)


def su2_mode_rotation(builder: CircuitBuilder, mode: int, theta: float, phase_a: float = 0.0,
                      phase_b: float = 0.0, *,
                      param_indices: Sequence[int] = (-1, -1, -1),
                      param_scales: Sequence[float] = (1.0, 1.0, 1.0)) -> CircuitBuilder:
    """The general ``SU(2)`` rotation of modes ``(mode, mode+1)``: ``D1 R(theta) D2``.

    The **complete** number-preserving two-mode unitary, up to a phase that a Givens network
    carries in its final diagonal — three real parameters, which is ``dim SU(2)``. Adjacent
    modes, so there is no Jordan-Wigner string and the whole thing is two qubits wide.

    This is both the building block :func:`orbital_rotation_circuit` compiles into and the
    natural *variational* unit: ``param_indices = (theta, phase_a, phase_b)`` marks all three
    for :func:`parameter_shift_gradient`, and every one of them drives its gates **linearly**,
    which a parameterization through ``expm`` of a ``kappa`` matrix would not. That is the
    reason a variational ansatz here is parameterized by Givens angles rather than by the
    entries of an anti-Hermitian generator.
    """
    p, q = int(mode), int(mode) + 1
    it, ia, ib = (int(i) for i in param_indices)
    st, sa, sb = (float(s) for s in param_scales)
    if abs(phase_b) > ANGLE_SCREEN_TOL or ib >= 0:      # D2 acts first
        builder.add("rz", (p,), (float(phase_b),), param_index=ib, scale=sb)
        builder.add("rz", (q,), (-float(phase_b),), param_index=ib, scale=-sb)
    if abs(theta) > ANGLE_SCREEN_TOL or it >= 0:
        exponential_circuit(builder, _real_excitation_generator((p,), (q,), builder.n_qubit),
                            float(theta), require_exact=True, param_index=it, param_scale=st)
    if abs(phase_a) > ANGLE_SCREEN_TOL or ia >= 0:
        builder.add("rz", (p,), (float(phase_a),), param_index=ia, scale=sa)
        builder.add("rz", (q,), (-float(phase_a),), param_index=ia, scale=-sa)
    return builder


def orbital_rotation_circuit(u: np.ndarray, *, builder: Optional[CircuitBuilder] = None
                             ) -> CompiledCircuit:
    """Compile ``U(u)``, the fermionic unitary with ``U a_p^dag U^dag = sum_q u_qp a_q^dag``.

    Exact, not Trotterized. The circuit's unitary times ``exp(1j * global_phase)`` is ``U(u)``,
    and that phase is ``arg(det u) / 2`` — the final diagonal is the only part a phase-free
    emission cannot cover.

    ⚠ **The reference check is the Slater-determinant minor, not another circuit.** Applied to
    an occupation mask, the amplitude on mask ``I`` must be ``det(u[occ(I), occ(ref)])`` — an
    independent formula from a different branch of the theory, which is what makes it able to
    fail.
    """
    rotations, phases = givens_decomposition(u)
    n = int(np.asarray(u).shape[0])
    builder = CircuitBuilder(n) if builder is None else builder
    # u = G_1^dag ... G_m^dag D, and U is a homomorphism, so D acts first and the daggered
    # rotations follow in reverse order.
    for k, phi in enumerate(phases):
        if abs(phi) > ANGLE_SCREEN_TOL:
            builder.add("rz", (k,), (float(phi),))
            builder.global_phase += 0.5 * float(phi)
    for rot in reversed(rotations):
        theta, pa, pb = _su2_parameters(rot.matrix().conj().T)
        su2_mode_rotation(builder, rot.mode, theta, pa, pb)
    return builder.result()


# --- Gradients ------------------------------------------------------------------------------

def parameter_shift_gradient(compiled: CompiledCircuit, energy: "callable") -> np.ndarray:
    """Exact gradient of ``energy(circuit)`` with respect to the ansatz parameters.

    Every parameterized gate is ``exp(-i theta P / 2)`` with ``P^2 = I``, so

        ``dE/dtheta_g = [E(theta_g + pi/2) - E(theta_g - pi/2)] / 2``

    holds **exactly** — no step size, no cancellation error — and the chain rule sums the gate
    derivatives into each parameter through :attr:`CompiledCircuit.gate_scale`. A parameter
    driving several gates (a UCC amplitude drives 2 or 8 rotations, a Givens angle drives 2)
    costs ``2`` evaluations per gate, which is the honest cost of the rule and the reason a
    hardware VQE is expensive.

    ⚠ **This is the hardware-relevant gradient, and it is not a finite difference.** A
    finite-difference gradient on a *shot-noisy* energy is dominated by the noise divided by
    the step; the shift rule has no step to divide by. Which is why it is here rather than a
    ``scipy`` numerical Jacobian.

    ``energy`` takes a :class:`~kuiva.qc.circuits.CircuitSpec` and returns a float.
    """
    grad = np.zeros(compiled.n_params, dtype=np.float64)
    half = 0.5 * np.pi
    for g in compiled.parameterized_gates().tolist():
        j = int(compiled.gate_param[g])
        plus = energy(compiled.shifted(g, half))
        minus = energy(compiled.shifted(g, -half))
        grad[j] += float(compiled.gate_scale[g]) * 0.5 * (plus - minus)
    return grad


__all__ = ["ANGLE_SCREEN_TOL", "CircuitBuilder", "CompiledCircuit", "GivensRotation",
           "double_excitation_generator", "excitation_circuit",
           "exponential_circuit", "givens_decomposition",
           "jastrow_generator", "one_body_generator", "orbital_rotation_circuit",
           "parameter_shift_gradient", "pauli_exponential", "pauli_exponential_circuit",
           "single_excitation_generator", "su2_mode_rotation", "trotter_circuit"]
