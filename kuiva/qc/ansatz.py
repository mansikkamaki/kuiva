"""Circuit strategies: what a sampled-subspace algorithm actually measures.

**Orchestration, not a registered kernel.**

The strategy seam, and why the driver does not own the circuit
---------------------------------------------------------------
The *circuit being sampled* is a pluggable strategy inside **one**
sampled-subspace driver, because the family is moving faster than the hardware: plain SQD
samples an ansatz state, SKQD samples (approximately Trotterized) time-evolved states, and a
scheme that appears next year will sample something else. Everything downstream — recovery,
subspace diagonalization, RDMs, ``AdaptiveCISolver`` conformance — is shared code that never
learns which strategy ran, so a new one is a class here and nothing else.

A strategy answers exactly one question: *given these active-space integrals, which circuits
should be sampled and with what share of the shots?* It returns :class:`SamplingPlan` —
:class:`~kuiva.qc.circuits.CircuitSpec` objects and shot weights, and nothing else.

The four rungs, and what each is *for*
---------------------------------------
  Stage A  :class:`ReferenceExcitationStrategy` — a reference determinant with a small
           rotation on each mode. Physically it is "excitations from a reference, ordered by
           rank"; numerically it is a distribution that *concentrates* on the reference and
           its low excitations, which is the structure SQD relies on. Cheap, transparent, and
           the right thing for validating the pipeline.
  Stage B  :class:`HardwareEfficientStrategy` — alternating single-qubit rotation layers and
           a ``cx`` ladder, with **no assumed spin structure**. The safest generalization to a
           non-separable spinor space and the least chemically informed; it is what gets
           real SQD-shaped behaviour with nothing chemical assumed.
  Stage C  :class:`UCCStrategy` and :class:`ClusterJastrowStrategy` — the research rung
           (the research half): unitary coupled cluster and cluster-Jastrow circuits built
           from **complex spinor** excitation generators, generalizing the LUCJ/UCCSD families
           past the spin separability every published implementation assumes. ⚠ Implemented
           and exact; that is **not** the same as validated as a good ansatz, and no docstring
           here may read as though it were.
  SKQD     :class:`TimeEvolutionStrategy` — not an ansatz at all: the circuit comes from the
           Hamiltonian, so there is nothing to guess.

⚠ **Stage A and Stage B do not conserve particle number, Stage C does, and that is the largest
practical difference between them.** The vocabulary's bare single-qubit rotations do not
conserve it and a hardware-efficient ansatz is defined by not caring, so :mod:`kuiva.qc.recovery`
has to repair 60-70% of the shots; a Stage-C circuit is built from number-conserving generators
and its recovery yield is **1** exactly. What Stage C costs instead is depth.

⚠ **Only Stage C and SKQD read the integrals.** A Stage-A or Stage-B circuit is built from a
mode count and a reference occupation, so under event gating a proposal made with one
adds statistical coverage and nothing else. That limitation is a property of those rungs, not
of the driver, and the two later ones remove it.

⚠ **Parameters are inputs here, not something this module optimizes.** A strategy takes the
angles it is given, derives them from the integrals (:func:`mp2_amplitudes`), or draws them
from a fixed seed so a run is replayable (the provenance-travels-with-the-result rule). Optimizing them
variationally is :mod:`kuiva.qc.vqe`, a *different algorithm* on the same boundary — and
:class:`VariationalAnsatz` is the seam that lets one object serve both.

References
----------------------------
* Sampled-subspace diagonalization, and the role of the sampling circuit in it: "Chemistry
  beyond the scale of exact diagonalization on a quantum-centric supercomputer", *Sci. Adv.*
  (2025), doi:10.1126/sciadv.adu9991.
* Hardware-efficient ansaetze: A. Kandala et al., "Hardware-efficient variational quantum
  eigensolver for small molecules and quantum magnets", *Nature* **549**, 242 (2017),
  doi:10.1038/nature23879.
* The ansatz families this module deliberately does **not** import unmodified, because they
  encode spin separability structurally: J. Romero et al., *Quantum Sci. Technol.* **4**,
  014008 (2018), doi:10.1088/2058-9565/aad3e4 (UCC); the LUCJ/``ffsim`` line described in
  the ansatz research notes in this package's docstrings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

from ..util.logging import get_logger
from .circuits import CircuitSpec

if TYPE_CHECKING:                            # pragma: no cover - typing only
    from .fermionic import CompiledCircuit

log = get_logger(__name__)


@dataclass(frozen=True)
class SamplingPlan:
    """Circuits to sample and the share of the shot budget each gets.

    ``weights`` are normalized on construction and turned into integer shot counts by
    :meth:`allocate`, which is where the rounding is decided **once** rather than in each
    driver. A plan with one circuit — the common case — allocates the whole budget to it.
    """

    circuits: Tuple[CircuitSpec, ...]
    weights: Tuple[float, ...]
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "circuits", tuple(self.circuits))
        object.__setattr__(self, "weights", tuple(float(w) for w in self.weights))
        if not self.circuits:
            raise ValueError("a sampling plan must contain at least one circuit")
        if len(self.circuits) != len(self.weights):
            raise ValueError("{} circuits but {} weights".format(
                len(self.circuits), len(self.weights)))
        if any(w < 0.0 for w in self.weights) or not sum(self.weights) > 0.0:
            raise ValueError("shot weights must be non-negative and not all zero")
        total = sum(self.weights)
        object.__setattr__(self, "weights", tuple(w / total for w in self.weights))
        widths = {c.n_qubit for c in self.circuits}
        if len(widths) != 1:
            raise ValueError("every circuit in a plan must have the same width; got {}"
                             .format(sorted(widths)))

    @property
    def n_qubit(self) -> int:
        return self.circuits[0].n_qubit

    def allocate(self, shots: int) -> Tuple[int, ...]:
        """Integer shots per circuit, summing **exactly** to ``shots``.

        Largest-remainder apportionment, so the total is exact and no circuit silently gets
        zero shots because of rounding — a circuit in the plan that is never measured is a
        strategy decision the strategy did not make.
        """
        shots = int(shots)
        if shots < len(self.circuits):
            raise ValueError("{} shots cannot be shared among {} circuits; every circuit in "
                             "a plan must get at least one".format(shots, len(self.circuits)))
        exact = np.asarray(self.weights) * shots
        base = np.floor(exact).astype(np.int64)
        base = np.maximum(base, 1)
        deficit = shots - int(base.sum())
        if deficit > 0:
            order = np.argsort(-(exact - np.floor(exact)))
            for k in range(deficit):
                base[order[k % base.size]] += 1
        elif deficit < 0:                       # the `maximum(base, 1)` floor overshot
            order = np.argsort(-base)
            for k in range(-deficit):
                base[order[k % base.size]] = max(1, base[order[k % base.size]] - 1)
        if int(base.sum()) != shots:            # pragma: no cover - defensive
            raise ValueError("shot apportionment lost {} shots".format(
                shots - int(base.sum())))
        return tuple(int(b) for b in base)

    def __repr__(self) -> str:
        return "SamplingPlan(n_circuits={}, n_qubit={}, label={!r})".format(
            len(self.circuits), self.n_qubit, self.label)


def aufbau_mask(n_elec: int) -> int:
    """The lowest-index occupation mask — the reference every strategy here starts from.

    Deliberately the same convention as :meth:`kuiva.ci.strings.Determinants.aufbau`. ⚠ It is
    a *reference*, not a guess at the ground state: in an active space chosen for strong
    correlation the aufbau determinant may carry very little of the true weight, which is the
    whole reason a sampled subspace is worth anything.
    """
    return (1 << int(n_elec)) - 1


# --- Stage A ------------------------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceExcitationStrategy:
    """A reference determinant with a small ``ry`` rotation on each mode (Stage A).

    ``ry(theta)`` on a mode takes ``|0> -> cos|0> + sin|1>``, so applying it to every qubit of
    ``|reference>`` gives a product state whose amplitude on a configuration differing from
    the reference in ``k`` modes goes as ``(tan(theta/2)-ish)^k``. The distribution therefore
    **concentrates on the reference and its low-rank excitations**, in the same hierarchy a
    selected CI would build classically — which is exactly the structure a sampled subspace
    needs, and exactly why the pipeline can be validated without solving the ansatz research problem.

    ⚠ **``entangle`` is off by default, and the reason is worth stating rather than hiding.**
    A ``cx`` ladder makes the state genuinely non-product, but ``cx`` **does not conserve
    particle number**: applied to the aufbau reference it moves it off ``n_elec`` outright,
    and the measured yield of valid configurations collapses (0.44 -> 0.07 on an 8-spinor,
    4-electron test). That is not a defect in the ladder, it is the package's research point in miniature —
    a circuit family that assumes nothing about the Hamiltonian also preserves none of its
    symmetries — and it is Stage B's whole cost. Stage A exists to be the *clean* rung, so it
    is a product state by default and the ladder is available for anyone who wants to watch
    the yield fall.

    Parameters
    ----------
    n_elec : int — electrons, fixing the reference through :func:`aufbau_mask`.
    theta : float
        Rotation angle [rad]. ⚠ The knob that trades subspace *size* against
        *concentration*, and the honest way to read it: small ``theta`` samples few
        configurations very reliably, large ``theta`` approaches a uniform product state and
        loses the hierarchy. It is not a variational parameter and nothing here optimizes it.
    reference : int, optional — occupation mask to excite from; aufbau by default.
    layers : int — how many rotation blocks. More layers spread further.
    entangle : bool — append a ``cx`` ladder to each block; see the warning above.
    """

    n_elec: int
    theta: float = 0.35
    reference: Optional[int] = None
    layers: int = 1
    entangle: bool = False

    def plan(self, n_qubit: int, **_) -> SamplingPlan:
        """The :class:`SamplingPlan`. Extra keyword arguments (integrals, and whatever a
        later strategy wants) are accepted and ignored, so the seam stays uniform."""
        n_qubit = int(n_qubit)
        mask = aufbau_mask(self.n_elec) if self.reference is None else int(self.reference)
        circuit = CircuitSpec.prepare(mask, n_qubit)
        for _layer in range(max(1, int(self.layers))):
            for q in range(n_qubit):
                circuit = circuit.with_gate("ry", (q,), (float(self.theta),))
            if self.entangle:
                for q in range(n_qubit - 1):
                    circuit = circuit.with_gate("cx", (q, q + 1))
        return SamplingPlan((circuit,), (1.0,),
                            label="reference+ry(theta={:.3f}) x{}".format(self.theta,
                                                                         self.layers))

    @property
    def requires(self) -> Tuple[str, ...]:
        return ("sample",)


# --- Stage B ------------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareEfficientStrategy:
    """Alternating rotation layers and a ``cx`` ladder, with no assumed spin structure.

    The Stage-B rung: the safest generalization to a complex, non-separable spinor space
    precisely *because* it assumes nothing about the Hamiltonian — and, for the same reason,
    the least chemically informed circuit anyone would run. It is here to produce real
    SQD-shaped behaviour (a broad, structured distribution over configurations) while the ansatz layer's
    open problem stays open.

    Parameters are taken as given or drawn once from a **fixed seed**, never optimized here:
    a strategy that quietly re-drew its angles per call would make :meth:`SQDSolver.solve`
    non-deterministic, which the adaptive-solver contract forbids outright.

    Parameters
    ----------
    n_elec : int — electrons; only used to pick the starting reference.
    layers : int — rotation+entangler blocks.
    params : ndarray, optional
        ``(layers, n_qubit, 2)`` angles (``ry`` then ``rz`` per qubit per layer). Drawn from
        ``seed`` when omitted, at the first :meth:`plan` call, and **cached** — see above.
    scale : float — standard deviation of the drawn angles [rad].
    """

    n_elec: int
    layers: int = 2
    params: Optional[np.ndarray] = None
    seed: int = 0
    scale: float = 0.6
    reference: Optional[int] = None

    def angles(self, n_qubit: int) -> np.ndarray:
        """``(layers, n_qubit, 2)`` rotation angles — given, or drawn from the fixed seed."""
        shape = (int(self.layers), int(n_qubit), 2)
        if self.params is not None:
            angles = np.asarray(self.params, dtype=np.float64)
            if angles.shape != shape:
                raise ValueError("params must have shape {}, got {}".format(shape,
                                                                            angles.shape))
            return angles
        return np.random.default_rng(self.seed).normal(0.0, float(self.scale), size=shape)

    def plan(self, n_qubit: int, **_) -> SamplingPlan:
        n_qubit = int(n_qubit)
        mask = aufbau_mask(self.n_elec) if self.reference is None else int(self.reference)
        angles = self.angles(n_qubit)
        circuit = CircuitSpec.prepare(mask, n_qubit)
        for layer in range(angles.shape[0]):
            for q in range(n_qubit):
                circuit = circuit.with_gate("ry", (q,), (float(angles[layer, q, 0]),))
                circuit = circuit.with_gate("rz", (q,), (float(angles[layer, q, 1]),))
            for q in range(n_qubit - 1):
                circuit = circuit.with_gate("cx", (q, q + 1))
        return SamplingPlan((circuit,), (1.0,),
                            label="hardware-efficient ({} layer{}, seed {})".format(
                                angles.shape[0], "" if angles.shape[0] == 1 else "s",
                                self.seed))

    @property
    def requires(self) -> Tuple[str, ...]:
        return ("sample",)


# --- Stage C: complex spinor excitation generators -------------------------------------------

class VariationalAnsatz:
    """Base class for a circuit family that is *also* a set of variational parameters.

    Two consumers, one object. A sampled-subspace solver asks for :meth:`plan` and measures the
    circuit at whatever parameters it has; a VQE asks for :meth:`build` and moves them. Keeping
    them the same class is what makes the comparison in ``examples/13`` a comparison of
    *algorithms* on one ansatz rather than of two different circuits.

    A subclass provides :meth:`n_parameters`, :meth:`initial_parameters` and :meth:`build`.
    """

    #: Backend primitives a *sampling* use of this ansatz needs.
    requires: Tuple[str, ...] = ("sample",)

    def n_parameters(self, n_qubit: int) -> int:
        raise NotImplementedError

    def initial_parameters(self, n_qubit: int, *, h=None, eri=None) -> np.ndarray:
        raise NotImplementedError

    def build(self, n_qubit: int, params: Optional[np.ndarray] = None, *, h=None,
              eri=None) -> "CompiledCircuit":
        raise NotImplementedError

    def plan(self, n_qubit: int, *, params: Optional[np.ndarray] = None, h=None, eri=None,
             **_) -> SamplingPlan:
        """One circuit, the whole shot budget, built at ``params`` (initial if omitted)."""
        compiled = self.build(int(n_qubit), params, h=h, eri=eri)
        return SamplingPlan((compiled.circuit,), (1.0,), label=self.label(compiled))

    def label(self, compiled: "CompiledCircuit") -> str:
        return "{} ({} gates, {} parameters)".format(
            type(self).__name__, compiled.circuit.n_gates, compiled.n_params)


def fock_diagonal(h: np.ndarray, eri: np.ndarray, reference: int) -> np.ndarray:
    """Orbital energies of the reference determinant: ``F_pp`` over the active space.

    ``F_pq = h_pq + sum_{i occupied} [(pq|ii) - (pi|iq)]``, in the chemists' notation
    ``ci/strings.py`` and ``CASIntegrals`` use. ⚠ ``h`` here is already the *inactive* Fock
    restricted to the active spinors, so this adds only the active occupied contribution — the
    inactive one is in ``h`` twice over if you add it again.
    """
    h = np.ascontiguousarray(h, dtype=np.complex128)
    n = int(h.shape[0])
    occ = [p for p in range(n) if (int(reference) >> p) & 1]
    fock = h.copy()
    for i in occ:
        fock += eri[:, :, i, i] - eri[:, i, i, :]
    return np.ascontiguousarray(np.real(np.diag(fock)))


def mp2_amplitudes(h: np.ndarray, eri: np.ndarray, reference: int, *,
                   max_amplitude: float = 0.5) -> Tuple[Dict, int]:
    """First-order (MP2) singles and doubles amplitudes for the reference determinant.

    Returns ``({(creations, annihilations): t}, n_clipped)`` with complex ``t``.

    ⚠ **In an active space the MP2 denominators are small by construction, and that is not a
    numerical accident — it is why the space is active.** A CAS is chosen precisely for its
    near-degeneracies, so ``eps_i + eps_j - eps_a - eps_b`` can be anything down to zero, and
    the perturbative amplitude it produces can be enormous or meaningless. Amplitudes are
    therefore **clipped in magnitude** (never the denominator, which would silently change the
    ratio between two amplitudes) and the count is returned so a caller can report it. What
    this buys is a *starting point* whose distribution concentrates on the low excitations of
    the reference; it is not a claim that MP2 describes the state.
    """
    h = np.ascontiguousarray(h, dtype=np.complex128)
    eri = np.asarray(eri)
    n = int(h.shape[0])
    ref = int(reference)
    occ = [p for p in range(n) if (ref >> p) & 1]
    virt = [p for p in range(n) if not (ref >> p) & 1]
    eps = fock_diagonal(h, eri, ref)
    fock = h + sum(eri[:, :, i, i] - eri[:, i, i, :] for i in occ) if occ else h
    amplitudes: Dict = {}
    clipped = 0

    def _store(key, value, denom):
        nonlocal clipped
        if abs(denom) < 1e-12:
            t = 0.0 + 0.0j
        else:
            t = complex(value / denom)
        if abs(t) > max_amplitude:
            t = t / abs(t) * max_amplitude
            clipped += 1
        if abs(t) > 0.0:
            amplitudes[key] = t

    for i in occ:
        for a in virt:
            _store(((a,), (i,)), fock[a, i], eps[i] - eps[a])
    for x, i in enumerate(occ):
        for j in occ[x + 1:]:
            for y, a in enumerate(virt):
                for b in virt[y + 1:]:
                    # <ij||ab> = (ia|jb) - (ib|ja) in chemists' notation.
                    value = eri[i, a, j, b] - eri[i, b, j, a]
                    _store(((a, b), (j, i)), value, eps[i] + eps[j] - eps[a] - eps[b])
    return amplitudes, clipped


@dataclass(frozen=True)
class UCCStrategy(VariationalAnsatz):
    """Disentangled UCC from **complex spinor** excitation generators — the Stage-C ansatz.

    ``prod_k exp(t_k D_k - t_k^* D_k^dag) |reference>`` over singles and doubles built on the
    *spinor* index, with no assumed spin structure and complex amplitudes. This is the
    generalization that is the research content of this package: every published UCC/LUCJ
    implementation assumes a spin-separable, real, non-relativistic Hamiltonian, and Kuiva's is
    none of those.

    Three things it buys, and one it costs:

    * ⚠ **Particle number is conserved exactly**, so configuration recovery has nothing to
      repair and the yield is 1. The Stage-A/B circuits pay 40-95% of their shots to
      :mod:`kuiva.qc.recovery`; this pays none. That is the largest single practical difference
      between a chemically structured ansatz and a hardware-efficient one.
    * ⚠ **It depends on the integrals** (through :func:`mp2_amplitudes`), so an
      ``AdaptiveCISolver`` proposal made with it is a genuine re-selection informed by the
      current orbitals rather than another draw from a fixed distribution — the limitation
      ``kuiva/qc/sqd.py`` records for Stage A and B.
    * **Each generator is exponentiated exactly**, complex amplitude included, by the
      phase-conjugation identity of :func:`kuiva.qc.fermionic.excitation_circuit`. Only the
      product *across* generators is a Trotter (disentangled-UCC) choice, which is a different
      ansatz rather than an approximation — Evangelista et al. (2019), cited in that module.
    * The cost is depth: a doubles generator is 8 Pauli exponentials of weight up to ``n``.

    Parameters
    ----------
    n_elec : int — electrons, fixing the reference through :func:`aufbau_mask`.
    reference : int, optional — occupation mask to excite from.
    doubles, singles : bool — which excitation ranks to include.
    generalized : bool
        Excite between **all** mode pairs rather than only occupied-to-virtual. The
        ``k-UpCCGSD`` idea: a generalized, layered product is far more expressive than plain
        UCCSD at the same rank, which is what makes a statevector VQE able to *reach* the full
        CI on a small space and therefore able to validate the mapping end to end. ⚠ The
        excitation count grows as ``n^4`` rather than ``n_occ^2 n_virt^2``, so this is a
        validation and small-space tool, not a production choice.
    layers : int
        Repeat the excitation list with independent parameters. One layer of a *disentangled*
        product is not the exponential of a sum (see :mod:`kuiva.qc.fermionic`), so layers are
        genuinely more expressive rather than redundant.
    real_amplitudes : bool
        Freeze every phase at zero, i.e. run the ansatz **the literature runs**. ⚠ This is the
        control, not a cheaper option: with SOC on, the difference between this and the complex
        ansatz is the question Stage C exists to answer, and it is measured in ``examples/13``.
    max_amplitude : float — magnitude clip on the MP2 guess (see :func:`mp2_amplitudes`).
    seed : int
        Used for the amplitudes MP2 does not define — every amplitude when no integrals are
        given, and the generalized/second-layer ones always. ⚠ Small and nonzero rather than
        zero: the *phase* of an excitation with zero magnitude has an identically zero
        gradient, so a zero start would leave half the parameters frozen and the optimizer
        would report convergence.
    """

    n_elec: int
    reference: Optional[int] = None
    singles: bool = True
    doubles: bool = True
    generalized: bool = False
    layers: int = 1
    real_amplitudes: bool = False
    max_amplitude: float = 0.5
    seed: int = 0
    jitter: float = 0.05

    def reference_mask(self) -> int:
        return aufbau_mask(self.n_elec) if self.reference is None else int(self.reference)

    def excitations(self, n_qubit: int) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]:
        """The ordered excitation list of **one** layer — structure, independent of amplitudes.

        ⚠ Only one of each generator and its Hermitian conjugate appears: they are the same
        anti-Hermitian generator, so including both would double-count the parameter and make
        the ansatz's dimension a fiction.
        """
        ref = self.reference_mask()
        modes = list(range(n_qubit))
        occ = modes if self.generalized else [p for p in modes if (ref >> p) & 1]
        virt = modes if self.generalized else [p for p in modes if not (ref >> p) & 1]
        out = []
        if self.singles:
            out.extend(((a,), (i,)) for i in occ for a in virt if a > i or not self.generalized)
        if self.doubles:
            for x, i in enumerate(occ):
                for j in occ[x + 1:]:
                    for y, a in enumerate(virt):
                        for b in virt[y + 1:]:
                            if self.generalized and (a, b) <= (i, j):
                                continue          # the h.c. of a generator already in the list
                            if set((a, b)) <= set((i, j)):
                                continue          # a number operator in disguise
                            out.append(((a, b), (j, i)))
        return tuple(out) * max(1, int(self.layers))

    def n_parameters(self, n_qubit: int) -> int:
        k = len(self.excitations(n_qubit))
        return k if self.real_amplitudes else 2 * k

    def initial_parameters(self, n_qubit: int, *, h=None, eri=None) -> np.ndarray:
        """MP2 amplitudes where integrals are available, a seeded draw where they are not."""
        order = self.excitations(n_qubit)
        params = np.zeros(self.n_parameters(n_qubit), dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        jitter = rng.normal(0.0, float(self.jitter), size=len(order))
        amplitudes: Dict = {}
        if h is not None and eri is not None:
            amplitudes, clipped = mp2_amplitudes(h, eri, self.reference_mask(),
                                                 max_amplitude=self.max_amplitude)
            if clipped:
                log.warning("%d of %d MP2 amplitudes hit the |t| <= %.3g clip; in an active "
                            "space the denominators are small by construction, so this is "
                            "expected and the amplitudes are a starting point rather than a "
                            "description", clipped, len(amplitudes) + clipped,
                            self.max_amplitude)
        seen = set()
        for k, key in enumerate(order):
            # ⚠ A repeated layer must not repeat the amplitude: the same excitation applied
            # twice with the same angle is one rotation of twice the angle, and the extra
            # layer would contribute nothing while claiming parameters.
            t = amplitudes.get(key, 0.0 + 0.0j) if key not in seen else 0.0 + 0.0j
            seen.add(key)
            magnitude = float(abs(t)) if abs(t) > 0.0 else float(jitter[k])
            if self.real_amplitudes:
                params[k] = float(np.real(t)) if abs(t) > 0.0 else float(jitter[k])
            else:
                params[2 * k] = magnitude
                params[2 * k + 1] = float(np.angle(t)) if abs(t) > 0.0 else 0.0
        return params

    def build(self, n_qubit: int, params: Optional[np.ndarray] = None, *, h=None,
              eri=None) -> "CompiledCircuit":
        from .fermionic import CircuitBuilder, excitation_circuit

        n_qubit = int(n_qubit)
        order = self.excitations(n_qubit)
        if params is None:
            params = self.initial_parameters(n_qubit, h=h, eri=eri)
        params = np.asarray(params, dtype=np.float64)
        if params.size != self.n_parameters(n_qubit):
            raise ValueError("this ansatz takes {} parameters, got {}".format(
                self.n_parameters(n_qubit), params.size))
        builder = CircuitBuilder(n_qubit).prepare(self.reference_mask())
        for k, (creations, annihilations) in enumerate(order):
            if self.real_amplitudes:
                excitation_circuit(builder, creations, annihilations, params[k], 0.0,
                                   magnitude_index=k)
            else:
                excitation_circuit(builder, creations, annihilations,
                                   params[2 * k], params[2 * k + 1],
                                   magnitude_index=2 * k, phase_index=2 * k + 1)
        return builder.result(self.n_parameters(n_qubit))


@dataclass(frozen=True)
class ClusterJastrowStrategy(VariationalAnsatz):
    """``prod_mu exp(K_mu) exp(i J_mu) exp(-K_mu) |reference>`` — LUCJ, generalized.

    The other Stage-C family: a local unitary cluster-Jastrow ansatz, with the one-body
    rotations taken over **complex spinors** instead of the real, spin-blocked orbitals every
    published LUCJ implementation uses.

    Two design decisions worth stating:

    * ⚠ **The rotations are parameterized by Givens angles, not by an anti-Hermitian
      ``kappa``.** A ``kappa`` reaches the gates through ``expm`` and a Givens decomposition,
      which is not a linear map, and the parameter-shift gradient of
      :mod:`kuiva.qc.fermionic` would then be wrong — quietly, since it would still return a
      plausible vector. Angles keep every gate linear in its parameter and lose nothing: a
      brick-wall network of adjacent ``SU(2)`` rotations spans the whole mode unitary group at
      sufficient depth.
    * **The Jastrow is nearest-neighbour by default**, which is where "local" in LUCJ comes
      from: a ``n_p n_q`` term is a two-qubit ``ZZ`` rotation, so restricting to neighbouring
      modes is what keeps the circuit runnable on a linear device. ``jastrow="full"`` lifts it.

    Number-conserving, like :class:`UCCStrategy`, so recovery has nothing to repair.

    Parameters
    ----------
    n_elec : int
    layers : int — how many ``exp(K) exp(iJ) exp(-K)`` blocks.
    depth : int — brick-wall rotation layers inside each ``K``.
    jastrow : ``"neighbour"`` or ``"full"``.
    seed, scale : the fixed-seed initial draw (see :class:`HardwareEfficientStrategy`).
    """

    n_elec: int
    layers: int = 1
    depth: int = 2
    jastrow: str = "neighbour"
    seed: int = 0
    scale: float = 0.3
    reference: Optional[int] = None

    def reference_mask(self) -> int:
        return aufbau_mask(self.n_elec) if self.reference is None else int(self.reference)

    def _pairs(self, n_qubit: int) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(range(d % 2, n_qubit - 1, 2)) for d in range(int(self.depth)))

    def _jastrow_pairs(self, n_qubit: int) -> Tuple[Tuple[int, int], ...]:
        if self.jastrow == "neighbour":
            return tuple((p, q) for p in range(n_qubit) for q in (p, p + 1) if q < n_qubit)
        if self.jastrow == "full":
            return tuple((p, q) for p in range(n_qubit) for q in range(p, n_qubit))
        raise ValueError("jastrow must be 'neighbour' or 'full', got {!r}".format(self.jastrow))

    def _layout(self, n_qubit: int) -> Tuple[int, int]:
        """``(rotation parameters, Jastrow parameters)`` per layer."""
        return (3 * sum(len(row) for row in self._pairs(n_qubit)),
                len(self._jastrow_pairs(n_qubit)))

    def n_parameters(self, n_qubit: int) -> int:
        rot, jas = self._layout(int(n_qubit))
        return int(self.layers) * (rot + jas)

    def initial_parameters(self, n_qubit: int, *, h=None, eri=None) -> np.ndarray:
        """A fixed-seed draw. ⚠ **No integral-derived initialization is offered, on purpose.**
        LUCJ amplitudes are normally fitted to a classical CCSD ``t2``, which Kuiva does not
        have and which is not defined for a complex spinor Hamiltonian without its own piece of
        research. Seeding from a distribution and letting VQE move it is honest; seeding from
        something that *looks* like a fit would not be."""
        rng = np.random.default_rng(self.seed)
        return rng.normal(0.0, float(self.scale), size=self.n_parameters(int(n_qubit)))

    def build(self, n_qubit: int, params: Optional[np.ndarray] = None, *, h=None,
              eri=None) -> "CompiledCircuit":
        from .fermionic import (CircuitBuilder, exponential_circuit, jastrow_generator,
                                su2_mode_rotation)

        n_qubit = int(n_qubit)
        if params is None:
            params = self.initial_parameters(n_qubit)
        params = np.asarray(params, dtype=np.float64)
        if params.size != self.n_parameters(n_qubit):
            raise ValueError("this ansatz takes {} parameters, got {}".format(
                self.n_parameters(n_qubit), params.size))
        rot_count, jas_count = self._layout(n_qubit)
        pairs, jpairs = self._pairs(n_qubit), self._jastrow_pairs(n_qubit)
        builder = CircuitBuilder(n_qubit).prepare(self.reference_mask())
        base = 0
        for _layer in range(int(self.layers)):
            rot = params[base:base + rot_count]
            jas = params[base + rot_count:base + rot_count + jas_count]
            self._rotation(builder, pairs, rot, base, inverse=True)     # exp(-K), acts first
            coupling = np.zeros((n_qubit, n_qubit), dtype=np.float64)
            for k, (p, q) in enumerate(jpairs):
                coupling[p, q] = coupling[q, p] = jas[k]
            generator = jastrow_generator(coupling)
            if generator.n_terms:
                exponential_circuit(builder, generator, 1.0, require_exact=True)
            self._rotation(builder, pairs, rot, base, inverse=False)    # exp(K)
            base += rot_count + jas_count
        return builder.result(self.n_parameters(n_qubit))

    @staticmethod
    def _rotation(builder, pairs, angles, offset: int, *, inverse: bool) -> None:
        """The brick-wall Givens network, or its inverse — the *same* parameters either way.

        ⚠ Sharing the parameters is what makes the block a conjugation ``exp(K) . exp(-K)``
        rather than two unrelated rotations. The inverse of ``D1 R(theta) D2`` is
        ``D2^dag R(-theta) D1^dag``, so the whole network reverses, the two phases **swap
        places** as well as changing sign, and every parameter's chain-rule scale flips.
        Negating ``theta`` alone would leave a plausible circuit that is not a conjugation, and
        forgetting the scales would leave a plausible gradient that is not one either.
        """
        from .fermionic import su2_mode_rotation

        schedule = [(mode, 3 * k)
                    for k, mode in enumerate(m for row in pairs for m in row)]
        for mode, base in (reversed(schedule) if inverse else schedule):
            theta, pa, pb = (float(v) for v in angles[base:base + 3])
            i0 = offset + base
            if inverse:
                su2_mode_rotation(builder, mode, -theta, -pb, -pa,
                                  param_indices=(i0, i0 + 2, i0 + 1),
                                  param_scales=(-1.0, -1.0, -1.0))
            else:
                su2_mode_rotation(builder, mode, theta, pa, pb,
                                  param_indices=(i0, i0 + 1, i0 + 2))


# --- SKQD: sampling a time-evolved state ------------------------------------------------------

@dataclass(frozen=True)
class TimeEvolutionStrategy:
    """Sample ``exp(-i H t_k) |reference>`` for a ladder of times — the SKQD circuit source.

    Sample-based **Krylov** quantum diagonalization: instead of sampling one ansatz state, draw
    from a set of time-evolved states, whose span is (approximately) a Krylov space of ``H``.
    The appeal, and the reason it strengthens the SQD case rather
    than competing with it, is that it **sheds the ansatz-design problem** — the circuit is
    determined by the Hamiltonian, so there is nothing to guess and nothing to optimize. Every
    step downstream (recovery, subspace diagonalization, RDMs, protocol conformance) is
    unchanged, which is exactly why the circuit source is a pluggable strategy.

    ⚠ **This is the first strategy that reads the integrals**, so it is the first for which a
    ``propose`` under event gating is a re-selection informed by the current orbitals
    rather than another draw from a fixed distribution.

    ⚠ **First-order Trotter, and the error is real.** The circuit is
    :func:`kuiva.qc.fermionic.trotter_circuit`, whose deviation from ``exp(-iHt)`` is
    ``O(t^2/steps)``. For a *sampling* application that error is far less damaging than it
    would be for a phase estimation — a slightly wrong state still concentrates on the right
    configurations, and the subspace diagonalization that follows is variational whatever was
    sampled — but it is not zero, and no claim here may treat the sampled state as the exact
    Krylov vector.

    ⚠ **A Pauli-Trotterized evolution does NOT conserve particle number, and that surprises
    people.** ``H`` conserves it and so does ``exp(-iHt)``; the individual **Pauli strings**
    ``H`` decomposes into do not, so a product of their exponentials leaks. It is a Trotter
    error like any other and vanishes with the slice count — measured at 11% of the norm for
    ``t = 0.5`` in one slice, 0.2% in two and 4e-5 in eight — but it is not zero, and it means
    SKQD pays a configuration-recovery yield exactly as a hardware-efficient ansatz does,
    where a naive reading of "the Hamiltonian conserves N" would predict none. The fix is a
    **number-conserving** decomposition — Trotterizing over Hermitian *fermionic* terms, or the
    double factorization :mod:`kuiva.qc.fermionic` records as deliberately deferred, whose
    factors are orbital rotations and diagonal phases and therefore conserve ``N`` exactly.
    The two deferrals are the same piece of work, and it buys symmetry as well as cost.

    ⚠ **Cost is the binding constraint, not accuracy.** The Hamiltonian has ``O(n^4)`` Pauli
    terms and each becomes a Pauli exponential of weight up to ``n``, so a Trotter step is tens
    of thousands of gates at ten spinors. ``screen`` drops small Pauli coefficients — which is
    a **truncation of the Hamiltonian**, reported and not a tidy-up — and the efficient answer
    is the double factorization ``kuiva/qc/fermionic.py`` records as deliberately deferred.

    Parameters
    ----------
    n_elec : int
    times : sequence of float
        Evolution times [1/Eh]. The ladder that spans the Krylov space; ``t = 0`` (the bare
        reference) is *not* added automatically, because a Krylov space that includes it is a
        different space and the caller should say so.
    steps : int — Trotter slices **per unit time**, so long times are not silently cruder.
    screen : float — drop Pauli terms below this coefficient before compiling.
    """

    n_elec: int
    times: Tuple[float, ...] = (0.5, 1.0, 1.5)
    steps: int = 1
    screen: float = 0.0
    reference: Optional[int] = None
    mapping: str = "jordan_wigner"

    @property
    def requires(self) -> Tuple[str, ...]:
        return ("sample",)

    def plan(self, n_qubit: int, *, h=None, eri=None, **_) -> SamplingPlan:
        from .fermionic import trotter_circuit
        from .mapping import resolve_mapping

        if h is None or eri is None:
            raise ValueError(
                "TimeEvolutionStrategy samples exp(-iHt), so it needs the integrals; the "
                "sampled-subspace driver passes them, but a direct plan() call must too. This "
                "is the first strategy for which that is true — the Stage-A/B circuits ignore "
                "them, which is exactly the limitation this one removes.")
        operator = resolve_mapping(self.mapping)(h, eri)
        if self.screen > 0.0:
            kept = operator.drop_below(self.screen)
            log.warning("SKQD: screening dropped %d of %d Pauli terms (%.3g of the 1-norm); "
                        "this is a truncation of the Hamiltonian being evolved, not of the one "
                        "being diagonalized afterwards", operator.n_terms - kept.n_terms,
                        operator.n_terms, 1.0 - kept.one_norm / max(operator.one_norm, 1e-30))
            operator = kept
        mask = aufbau_mask(self.n_elec) if self.reference is None else int(self.reference)
        circuits = tuple(
            trotter_circuit(operator, float(t),
                            steps=max(1, int(round(self.steps * abs(float(t))))),
                            reference=mask).circuit
            for t in self.times)
        return SamplingPlan(circuits, (1.0,) * len(circuits),
                            label="SKQD: exp(-iHt) at t = {}".format(
                                ", ".join("{:.3g}".format(t) for t in self.times)))


# --- The registry -------------------------------------------------------------------------

_STRATEGIES = {
    "reference": ReferenceExcitationStrategy,
    "hardware_efficient": HardwareEfficientStrategy,
    "ucc": UCCStrategy,
    "cluster_jastrow": ClusterJastrowStrategy,
    "time_evolution": TimeEvolutionStrategy,
}


def available_strategies() -> Tuple[str, ...]:
    return tuple(sorted(_STRATEGIES))


def resolve_strategy(name: str, **kwargs):
    """Construct a circuit strategy by name.

    ⚠ Only implemented strategies are registered. A Stage-C ansatz name that resolved to a
    Stage-B circuit would be the worst possible failure here — a research claim that quietly
    ran a different ansatz — so nothing is registered until it exists, exactly as
    ``amf/backend.py`` keeps ``"kuiva"`` unregistered.
    """
    key = str(name).lower()
    if key not in _STRATEGIES:
        raise ValueError("unknown circuit strategy {!r}; registered: {}"
                         .format(name, ", ".join(available_strategies())))
    return _STRATEGIES[key](**kwargs)


def register_strategy(name: str, factory) -> None:
    """Register a circuit strategy under ``name``."""
    _STRATEGIES[str(name).lower()] = factory


__all__ = ["ClusterJastrowStrategy", "HardwareEfficientStrategy",
           "ReferenceExcitationStrategy", "SamplingPlan", "TimeEvolutionStrategy",
           "UCCStrategy", "VariationalAnsatz", "aufbau_mask", "available_strategies",
           "fock_diagonal", "mp2_amplitudes", "register_strategy", "resolve_strategy"]
