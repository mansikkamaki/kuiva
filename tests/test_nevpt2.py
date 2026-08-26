"""SC-NEVPT2: the complete eight-class correction and its frozen-core option.

The tests are ordered by what they can *fail on*, which is the criterion that matters rather
than how tightly they agree:

1. **Against brute force** (:mod:`fockspace`) — the strongest checks here. Every contraction
   primitive and every class norm and energy is compared with the same quantity built from
   explicit ladder-operator strings over a tiny Fock space, with **complex 4-fold** integrals.
   That reference shares no code with ``kuiva.pt``, so it can see an error in the spinor
   derivation; an internal consistency check could not.
2. **Theorems** — invariance under rotations of whole subspaces *and* inside a degenerate
   ``eps`` block, the null case, Kramers equality, size consistency, the Hylleraas hierarchy,
   Hermiticity.
3. **Against PySCF** (Tier 1) — the same per-class numbers *and the total* in the scalar
   limit, which validates the whole substrate: the integral blocks, the pseudo-canonicalization
   and the driver, against an independent implementation of the same method.
4. **With spin-orbit coupling on** — the committed brute-force record of
   ``tests/reference/nevpt2_uncontracted.json``, which is the only like-for-like check of the
   SOC-on algebra that exists and simultaneously checks that the
   frozen-core option restricts *labels* and nothing else.
5. **Structure** — the registry, the resource sizing, the warnings that are behaviour.

⚠ Several checks come in pairs with a *companion that must fail*: a guard that cannot fail
proves nothing (a guard that cannot fail proves nothing). Two instances here — the group-complete small-norm cutoff,
and the group-complete *contraction*, whose absence is invisible on any system without a
degenerate ``eps`` block and which is why one fixture partition imposes one.
"""
from __future__ import annotations

import numpy as np
import pytest

from fockspace import ReferenceNEVPT2
from kuiva.ci.strings import CASSpace
from kuiva.mcscf.orbopt import CASIntegrals, OrbitalSpaces, averaged_fock
from kuiva.pt import blocks as ptblocks
from kuiva.pt.classes import (ClassContext, available_classes, class_energy,
                              denominator_groups, excitation_class, group_complete_mask,
                              implemented_classes, label_buffer_gb)
from kuiva.pt.contractions import CIContractionProvider, koopmans_annihilation
from kuiva.pt.nevpt2 import pseudo_canonicalize, sc_nevpt2

#: The Fock-space reference is exact, so the only error is rounding on ``2**8`` amplitudes.
BRUTE_FORCE_TOL = 1e-12
#: Tier-1 cross-implementation lock on a per-class energy. ⚠ Absolute, and it may be, because
#: what is compared is a *correlation* contribution of order 1e-4 Eh produced from the same
#: orbitals and the same CI state — not a total energy, which is locked relatively.
PYSCF_CLASS_TOL = 1e-11
#: Invariance under a unitary mixing of whole subspaces is a theorem, not a tolerance; this is
#: a rounding budget for one extra integral transform.
INVARIANCE_TOL = 1e-12


# --- fixtures and helpers ------------------------------------------------------------------

class StubBlocks:
    """The three-index interface, served from an explicit complex factorization.

    ⚠ Why a stub rather than the real :class:`kuiva.pt.blocks.IntegralBlocks`: the point of the
    brute-force comparison is to drive the class algebra with **complex, SOC-like** integrals
    that have exactly 4-fold symmetry, and no real AO basis can produce those. Building the
    same factorization the classes will see at run time — ``B^P`` sliced by orbital space — is
    what keeps the test on the algebra instead of on the front-end. The front-end path is what
    the Tier-1 test below covers.
    """

    def __init__(self, factor: np.ndarray, spaces: dict) -> None:
        self.factor = factor
        self.spaces = spaces

    def size(self, name: str) -> int:
        return len(self.spaces[name])

    def three_index(self, bra: str, ket: str) -> np.ndarray:
        rows = self.spaces[bra]
        cols = self.spaces[ket]
        return np.ascontiguousarray(self.factor[:, rows, :][:, :, cols])


def factorized_integrals(n: int, seed: int, scale: float = 0.3):
    """``(h, eri, B)`` with ``eri = sum_P B^P_pq B^P_rs`` — the same object twice.

    Mirrors :func:`fockspace.random_integrals` exactly, and additionally returns the factor, so
    that the brute-force reference and the class evaluation are demonstrably fed the *same*
    two-electron integrals rather than two that agree numerically.
    """
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = 0.5 * (h + h.conj().T)
    naux = 2 * n
    b = rng.normal(size=(naux, n, n)) + 1j * rng.normal(size=(naux, n, n))
    b = 0.5 * (b + b.transpose(0, 2, 1).conj())
    b = b * np.sqrt(scale / naux)
    return h, np.tensordot(b, b, axes=([0], [0])), b


#: ``(inactive, active, virtual, n_active_elec, degenerate_eps)``. ⚠ The third entry pairs the
#: ``eps`` into **exactly degenerate** groups, which is the only configuration that exercises
#: the group-complete contraction of :mod:`kuiva.pt.classes` — with generic ``eps`` every group
#: is a singleton and the rule is invisible. It is how a real calculation always looks
#: (Kramers pairs), and how the requirement was missed on the first pass.
PARTITIONS = [
    pytest.param(([0, 1], [2, 3, 4, 5], [6, 7], 2, False), id="2c-4a-2v-2e"),
    pytest.param(([0, 1, 2], [3, 4, 5], [6, 7], 2, False), id="3c-3a-2v-2e"),
    pytest.param(([0, 1], [2, 3, 4, 5], [6, 7], 2, True), id="2c-4a-2v-2e-degenerate-eps"),
]


@pytest.fixture(params=PARTITIONS, scope="module")
def brute(request):
    """A tiny complex system, solved both ways: the reference, and kuiva's context."""
    inactive, active, virtual, n_active_elec, degenerate = request.param
    n = len(inactive) + len(active) + len(virtual)
    h, eri, factor = factorized_integrals(n, seed=1 + n)
    # eps are *given* to both sides. They need not diagonalize anything: the Dyall eps term is
    # a number operator, so a perturber with a fixed hole/particle pattern is an eigenfunction
    # of it whatever the values are, on both sides of the comparison.
    rng = np.random.default_rng(7)
    eps_i = np.sort(rng.normal(loc=-2.0, size=len(inactive)))
    eps_v = np.sort(rng.normal(loc=+1.5, size=len(virtual)))
    if degenerate:
        eps_i[:] = eps_i[0]
        eps_v[:] = eps_v[0]
    ref = ReferenceNEVPT2(h, eri, inactive, active, virtual, eps_i, eps_v, n_active_elec)

    space = CASSpace(len(active), n_active_elec)
    civec = np.ascontiguousarray(ref.cas_vectors[:, 0].astype(np.complex128))
    provider = CIContractionProvider(space, civec, ref.f_inactive[np.ix_(active, active)],
                                     eri[np.ix_(active, active, active, active)])
    stub = StubBlocks(factor, {"inactive": inactive, "active": active, "virtual": virtual,
                               "all": list(range(n))})
    ctx = ClassContext(blocks=stub, provider=provider, eps_inactive=eps_i, eps_virtual=eps_v,
                       fock_vi=ref.f_inactive[np.ix_(virtual, inactive)],
                       fock_va=ref.f_inactive[np.ix_(virtual, active)],
                       fock_ai=ref.f_inactive[np.ix_(active, inactive)])
    return ref, ctx


#: Every class, in the reporting order. Used wherever a test must cover the whole partition —
#: written out rather than taken from :func:`implemented_classes` so that a class quietly
#: losing its energy status fails the tests instead of shrinking their coverage.
ALL_CLASSES = ["Sijrs", "Srsi", "Sijr", "Srs", "Sij", "Sir", "Sr", "Si"]


# --- 1. against brute force ------------------------------------------------------------------

def test_the_reference_is_self_consistent(brute):
    """⚠ Guard the guard: ``H_D |Psi_0> = E_0 |Psi_0>`` must hold to machine precision.

    The brute-force module is the authority for every number below it, so its own construction
    — the inactive Fock, the core energy, the Dyall constant and every ladder-operator sign —
    is asserted first. A reference nobody checks is not a reference.
    """
    ref, _ = brute
    assert ref.dyall_residual(0) < 1e-12


def test_contraction_primitives_match_explicit_operators(brute):
    """``gamma``, ``Gamma``, the hole pair matrix and both Koopmans matrices, term by term.

    The hole pair matrix is the one worth naming: six ``delta`` terms with three sign patterns,
    every one of them invisible to a Hermiticity or trace check on the result.
    """
    ref, ctx = brute
    provider = ctx.provider
    assert np.allclose(provider.rdm1(), ref.rdm1(), atol=BRUTE_FORCE_TOL)
    assert np.allclose(provider.rdm2(), ref.rdm2(), atol=BRUTE_FORCE_TOL)
    assert np.allclose(provider.hole_pair_matrix(), ref.hole_pair(), atol=BRUTE_FORCE_TOL)
    assert np.allclose(provider.koopmans_annihilation(), ref.koopmans_annihilation(),
                       atol=BRUTE_FORCE_TOL)
    assert np.allclose(provider.koopmans_creation(), ref.koopmans_creation(),
                       atol=BRUTE_FORCE_TOL)


def test_derived_primitives_are_hermitian(brute):
    """Every kernel of a perturber norm is Hermitian, so the norms are real."""
    _, ctx = brute
    provider = ctx.provider
    for name, matrix in (("hole 1-RDM", provider.hole_rdm1()),
                         ("pair", provider.pair_matrix()),
                         ("hole pair", provider.hole_pair_matrix()),
                         ("K", provider.koopmans_annihilation()),
                         ("K'", provider.koopmans_creation())):
        assert np.allclose(matrix, matrix.conj().T, atol=1e-12), name


@pytest.mark.parametrize("name", ALL_CLASSES)
def test_class_norm_matches_brute_force(brute, name):
    """``sum_l N_l = <Psi_0| H P_class H |Psi_0>``, against explicit projection.

    ⚠ This is the check the whole spinor derivation rests on. The perturber norm is
    convention-free — ``P_l`` is a projector, so no scaling of ``|Psi_l>`` can change it — which
    is why two implementations that share nothing must produce the same number.
    """
    ref, ctx = brute
    reference_norm, _ = ref.by_name(name)
    result = excitation_class(name).evaluate(ctx)
    assert result.norm == pytest.approx(reference_norm, abs=BRUTE_FORCE_TOL,
                                        rel=BRUTE_FORCE_TOL)


@pytest.mark.parametrize("name", ALL_CLASSES)
def test_class_energy_matches_brute_force(brute, name):
    """``E^k = -sum_l N_l / dE_l``, denominators included.

    Where the norm test checks the perturber, this checks the *Koopmans* algebra as well: every
    denominator but ``Sijrs``'s carries a ``K/N``, so a wrong commutator, a wrong ladder-string
    sign or a wrong ``H_act`` on a shifted electron-number space shows up here and nowhere
    above. ``Sir`` additionally exercises the augmented basis, whose first row and column
    behave differently in the overlap and in the Koopmans matrix.
    """
    ref, ctx = brute
    _, reference_energy = ref.by_name(name)
    result = excitation_class(name).evaluate(ctx)
    assert result.energy is not None
    assert result.energy == pytest.approx(reference_energy, abs=BRUTE_FORCE_TOL,
                                          rel=BRUTE_FORCE_TOL)


def test_a_coincident_pair_label_has_a_vanishing_perturber(brute):
    """⚠ The claim that makes ``sum_{a<b} = 1/2 sum_{a,b}`` exact, asserted rather than assumed.

    The classes enumerate ordered pairs and divide by two; that is only correct because the
    perturber of a label with two equal external indices vanishes identically — it contains
    ``a+_a a+_a``. The formula has to reproduce that on its own, from the symmetry of the
    integrals against the antisymmetry of the density kernel, and if it did not, the diagonal
    would contribute a spurious positive norm.
    """
    _, ctx = brute
    b_va = ctx.blocks.three_index("virtual", "active")
    na = ctx.n_active
    kernel = ctx.provider.pair_matrix()
    for a in range(ctx.n_virtual):
        v = np.tensordot(b_va[:, a:a + 1, :], b_va[:, a:a + 1, :], axes=([0], [0]))
        x = v.transpose(0, 2, 1, 3).reshape(1, na * na)
        assert abs((x.conj() @ kernel @ x.ravel()).item()) < 1e-13


@pytest.mark.parametrize("name", ALL_CLASSES)
def test_the_contracted_energy_sits_above_the_uncontracted_one(brute, name):
    """``E2_unc <= E2_SC <= 0`` per class — the Hylleraas hierarchy, and its proof.

    ⚠ The plan lists this assertion as *conditional on a proof* that the implemented per-class
    formulas really are Hylleraas minima. They are, and the argument is short enough to record
    where it is used:

    * ``H_D`` preserves the hole/particle pattern (the ``eps`` term is a number operator and
      ``H_act`` touches only active modes), **which is exactly what the pseudo-canonicalization
      buys**, so different label sets neither couple through ``H_D`` nor overlap. The
      minimization therefore decouples label by label.
    * Over the one-dimensional span of ``|Psi_l>``, the Hylleraas functional
      ``2 Re(c* N_l) + |c|^2 N_l dE_l`` is stationary at ``c = -1/dE_l`` with value
      ``-N_l / dE_l`` — which *is* the implemented formula. So SC is the exact minimum in its
      space, and the uncontracted energy is the minimum in a space containing it.
    * ⚠ The stationary point is a *minimum* only while ``H_D - E_0`` is positive definite on
      the class subspace. A genuine intruder makes the functional unbounded below and the
      inequality vacuous, so the test asserts positive denominators as part of the claim rather
      than assuming them.

    The conclusion is that the assertion is available for every class implemented here, and
    that it must be re-derived rather than inherited for any class whose perturber space is not
    one-dimensional per label set (i.e. for FIC).
    """
    ref, ctx = brute
    holes, particles = ref.CLASS_PATTERN[name]
    denominators = [d for _, d in ref.class_terms(holes, particles)]
    if not denominators:
        pytest.skip("this class has no perturbers in this partition")
    assert min(denominators) > 0.0, "the hierarchy needs no intruder"

    contracted = excitation_class(name).evaluate(ctx).energy
    uncontracted = ref.class_energy_uncontracted(holes, particles)
    assert contracted <= 0.0
    assert uncontracted <= contracted + BRUTE_FORCE_TOL


def test_what_the_strong_contraction_actually_costs():
    """⚠ The companion the inequality above needs: sometimes the gap is exactly zero, and that
    is a fact about the classes rather than a vacuous test.

    * ``Sijrs`` is **not contracted at all**. Its perturber
      ``a+_a a+_b a_j a_i |Psi_0>`` leaves the active part untouched, so it is already an
      eigenvector of ``H_D`` inside the class subspace and the uncontracted minimization has
      nothing to add. The gap is zero to machine precision, and an implementation that produced
      a nonzero one would be wrong.
    * ``Srsi`` and ``Sijr`` contract a genuinely multi-dimensional space — **provided the
      active space is big enough to have one**. With three active spinors and two electrons,
      ``Sijr``'s ``N_act + 1 = 3`` sector is a single determinant and the gap is zero for that
      reason alone, which is why this test fixes a four-spinor active space instead of reusing
      the parametrized fixture.
    """
    inactive, active, virtual, n_elec = [0, 1], [2, 3, 4, 5], [6, 7], 2
    h, eri, factor = factorized_integrals(8, seed=9)
    rng = np.random.default_rng(2)
    eps_i = np.sort(rng.normal(loc=-2.0, size=len(inactive)))
    eps_v = np.sort(rng.normal(loc=+1.5, size=len(virtual)))
    ref = ReferenceNEVPT2(h, eri, inactive, active, virtual, eps_i, eps_v, n_elec)
    space = CASSpace(len(active), n_elec)
    provider = CIContractionProvider(space,
                                     np.ascontiguousarray(ref.cas_vectors[:, 0].astype(complex)),
                                     ref.f_inactive[np.ix_(active, active)],
                                     eri[np.ix_(active, active, active, active)])
    ctx = ClassContext(blocks=StubBlocks(factor, {"inactive": inactive, "active": active,
                                                  "virtual": virtual, "all": list(range(8))}),
                       provider=provider, eps_inactive=eps_i, eps_virtual=eps_v,
                       fock_vi=ref.f_inactive[np.ix_(virtual, inactive)],
                       fock_va=ref.f_inactive[np.ix_(virtual, active)],
                       fock_ai=ref.f_inactive[np.ix_(active, inactive)])

    gaps = {}
    for name in ("Sijrs", "Srsi", "Sijr", "Sr", "Si"):
        holes, particles = ref.CLASS_PATTERN[name]
        contracted = excitation_class(name).evaluate(ctx).energy
        gaps[name] = contracted - ref.class_energy_uncontracted(holes, particles)
    assert abs(gaps["Sijrs"]) < 1e-14, "S(0) is uncontracted by construction"
    for name in ("Srsi", "Sijr", "Sr", "Si"):
        assert gaps[name] > 1e-9, name
    # ⚠ The primed classes pay the *most*, by two orders of magnitude, and that is the point of
    # recording it: they contract the largest perturber space onto one function per label, so
    # they are where a future FIC treatment would buy the most.
    assert min(gaps["Sr"], gaps["Si"]) > 100 * max(gaps["Srsi"], gaps["Sijr"])


def test_the_two_routes_to_the_overlap_kernels_agree(brute):
    """⚠ The check that validates the ladder-string machinery against the density algebra.

    Each ``(+-2)`` class has its overlap kernel built **twice** by routes that share no code:
    once by anticommuting operators into normal order and contracting the 1- and 2-RDM
    (:func:`kuiva.pt.contractions.pair_matrix`, ``hole_pair_matrix``), and once as the Gram
    matrix of vectors built by applying ladder operators between electron-number spaces. Every
    fermionic sign in the second route has to be right for them to agree, and none of those
    signs is visible in a Hermiticity or trace check.

    The same holds for the ``Sir`` augmented overlap, whose ``n^2`` block is
    ``<E_ut E_vw>`` — reachable from the 2-RDM as well.
    """
    _, ctx = brute
    provider = ctx.provider
    n = ctx.n_active
    gram_pair = provider._pair_gram()[0]                  # noqa: SLF001 - the point of the test
    gram_hole = provider._hole_pair_gram()[0]             # noqa: SLF001
    assert np.allclose(gram_pair, provider.pair_matrix(), atol=BRUTE_FORCE_TOL)
    assert np.allclose(gram_hole, provider.hole_pair_matrix(), atol=BRUTE_FORCE_TOL)

    overlap = provider.excitation_overlap()
    assert overlap.shape == (n * n + 1, n * n + 1)
    assert overlap[0, 0] == pytest.approx(1.0, abs=1e-12)          # the reference is normalized
    assert np.allclose(overlap[0, 1:].reshape(n, n), provider.rdm1(), atol=BRUTE_FORCE_TOL)
    # <E_ut E_vw> = delta_tv gamma_uw + Gamma_utvw, from the 2-RDM and nothing else.
    gamma, gamma2 = provider.rdm1(), provider.rdm2()
    expected = (np.einsum("tv,uw->tuvw", np.eye(n), gamma)
                + np.einsum("utvw->tuvw", gamma2)).reshape(n * n, n * n)
    assert np.allclose(overlap[1:, 1:], expected, atol=BRUTE_FORCE_TOL)


def test_the_group_complete_contraction_is_load_bearing(brute):
    """⚠ The companion that must fail: a per-label contraction gives a *different* answer.

    Re-running the class energies with ``groups=None`` — the natural first implementation —
    must give a *different* answer here, or the degenerate partition would be a decoration
    rather than the only configuration that exercises the rule at all.

    ⚠ **On this synthetic system the rule moves several classes, and on a real one it moves
    only ``Sir``. Both facts are real and the difference is the point.** Here the ``eps``
    degeneracy is imposed by hand on integrals with no matching symmetry, so the denominators
    genuinely vary inside a group and every class feels the contraction. In a molecule the
    degeneracy *comes from* a symmetry the integrals share — Kramers, or a point group — so the
    denominators are constant inside the group and the rule is a no-op everywhere except
    ``Sir``, whose same-spin and spin-flip perturbers are not related by any symmetry. The
    molecular statement is what the PySCF comparison measures; this one is what proves the code
    is doing the contraction at all.
    """
    ref, ctx = brute
    if len(set(ctx.group_virtual)) == ctx.n_virtual and \
            len(set(ctx.group_inactive)) == ctx.n_inactive:
        pytest.skip("no degenerate eps in this partition, so there is nothing to lump")

    from kuiva.pt import classes as ptclasses

    seen = {}
    original = ptclasses.class_energy

    def ungrouped(name, norms, denominators, context, *, groups=None, **kwargs):
        result = original(name, norms, denominators, context, groups=None, **kwargs)
        seen[name] = result.energy
        return result

    lumped = [n for n in ALL_CLASSES if n != "Sijrs"]
    ptclasses.class_energy = ungrouped
    try:
        per_label = {n: excitation_class(n).evaluate(ctx).energy for n in lumped}
    finally:
        ptclasses.class_energy = original
    grouped = {n: excitation_class(n).evaluate(ctx).energy for n in lumped}

    assert per_label["Sir"] != pytest.approx(grouped["Sir"], rel=1e-9), \
        "the degenerate partition must actually exercise the contraction rule"
    # ...and the grouped value is the one the independent reference produces, class by class.
    for name, energy in grouped.items():
        assert energy == pytest.approx(ref.by_name(name)[1], abs=BRUTE_FORCE_TOL,
                                       rel=BRUTE_FORCE_TOL), name


def test_the_perturber_primitive_returns_the_gram_matrix_it_claims_to(brute):
    """⚠ The FIC-readiness claim of :mod:`kuiva.pt.contractions`, asserted rather than asserted.

    Strong contraction only reads the *diagonal* of the single-external Gram pair, so the
    ``full=True`` path — the one that with the identity as coefficients gives the internally
    contracted metric and per-class Hamiltonian — would otherwise be an untested promise. Two
    things are checked: that its diagonal is the number the SC path uses, and that with the
    identity it reproduces the plain Gram matrix of the ``n_act^3`` ladder strings, computed
    here by an explicit loop over :func:`kuiva.ci.strings.apply_ladder`.
    """
    from kuiva.ci.strings import apply_ladder

    _, ctx = brute
    provider = ctx.provider
    n = ctx.n_active
    rng = np.random.default_rng(11)
    rows = 3
    w1 = rng.normal(size=(rows, n)) + 1j * rng.normal(size=(rows, n))
    w3 = rng.normal(size=(rows, n, n, n)) + 1j * rng.normal(size=(rows, n, n, n))

    for method in (provider.annihilation_perturbers, provider.creation_perturbers):
        norm, koop, _ = method(w1, w3)
        overlap_full, koop_full, _ = method(w1, w3, full=True)
        assert np.allclose(np.diag(overlap_full).real, norm, atol=1e-12)
        assert np.allclose(np.diag(koop_full).real, koop, atol=1e-12)
        # A Gram matrix is Hermitian positive semidefinite whatever the coefficients are.
        assert np.allclose(overlap_full, overlap_full.conj().T, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(overlap_full)) > -1e-12

    # ...and the identity gives the string-basis metric, built here without the provider.
    strings = np.zeros((n ** 3, provider.shifted_ndet(-1)), dtype=complex)
    minus1 = provider.spaces.get(-1)
    minus2 = provider.spaces.get(-2)
    if not minus2.empty:
        for t in range(n):
            at = apply_ladder(provider.space.masks, minus1.masks, t, provider.civec)
            for v in range(n):
                av = apply_ladder(minus1.masks, minus2.masks, v, at)
                for u in range(n):
                    strings[(t * n + u) * n + v] = apply_ladder(
                        minus2.masks, minus1.masks, u, av, dagger=True)
        eye = np.eye(n ** 3, dtype=complex).reshape(n ** 3, n, n, n)
        metric, _, _ = provider.annihilation_perturbers(
            np.zeros((n ** 3, n), dtype=complex), eye, full=True)
        assert np.allclose(metric, strings.conj() @ strings.T, atol=BRUTE_FORCE_TOL)


def test_a_primed_class_refuses_without_its_one_body_fock_block(brute):
    """The one-body half of a primed perturber is not optional, so its absence must raise.

    ⚠ Silently treating a missing ``f^I`` block as zero would drop the whole one-body term of
    ``Sr``/``Si`` and leave a plausible, smaller, wrong class energy — the worst shape of error —
    says to guard against structurally rather than by review.
    """
    from dataclasses import replace

    _, ctx = brute
    with pytest.raises(ValueError, match="virtual, active"):
        excitation_class("Sr").evaluate(replace(ctx, fock_va=None))
    with pytest.raises(ValueError, match="active, inactive"):
        excitation_class("Si").evaluate(replace(ctx, fock_ai=None))


def test_the_koopmans_matrix_of_the_reference_row_vanishes(brute):
    """``(H_act - E)|Psi> = 0``, so the augmented Koopmans matrix has a zero first row.

    Free, and it fails on a state that is not an eigenvector of the Hamiltonian it is being
    contracted with — the same inconsistency :func:`koopmans_annihilation`'s Hermiticity check
    catches, reached from a different direction.
    """
    _, ctx = brute
    koopmans = ctx.provider.excitation_koopmans()
    assert np.max(np.abs(koopmans[0, :])) < 1e-10
    assert np.max(np.abs(koopmans[:, 0])) < 1e-10


# --- 2. theorems ------------------------------------------------------------------------------

def test_koopmans_hermiticity_is_a_real_check_and_warns_when_it_fails(brute, kuiva_caplog):
    """⚠ A guard that cannot fail proves nothing: break the pairing and it must complain.

    ``K`` is Hermitian *only* because the density matrices come from an exact eigenvector of the
    active Hamiltonian they are contracted with. Feeding a mismatched Hamiltonian must therefore
    be detected — that is what makes the silent pass on the matching pair informative.
    """
    ref, ctx = brute
    active = ref.active
    gamma, gamma2 = ctx.provider.rdm1(), ctx.provider.rdm2()
    h_act = ref.f_inactive[np.ix_(active, active)]
    eri_act = ref.eri[np.ix_(active, active, active, active)]

    kuiva_caplog.clear()
    koopmans_annihilation(h_act, eri_act, gamma, gamma2)
    assert not [r for r in kuiva_caplog.records if r.levelname == "WARNING"]

    kuiva_caplog.clear()
    perturbed = h_act + np.diag(np.linspace(0.1, 0.3, h_act.shape[0]))
    koopmans_annihilation(perturbed, eri_act, gamma, gamma2)
    assert any("not Hermitian" in r.getMessage() for r in kuiva_caplog.records)


def test_denominator_groups_collect_exact_degeneracies():
    groups = denominator_groups(np.array([1.0, 2.0, 1.0, 3.0, 2.0]))
    assert groups[0] == groups[2]
    assert groups[1] == groups[4]
    assert len({int(g) for g in groups}) == 3


def test_the_small_norm_cutoff_never_splits_a_degenerate_pair():
    """The group-complete cutoff, and its companion that must fail.

    Two perturbers with the *same* denominator — which is what time reversal guarantees — where
    only one is below the cutoff. The group-complete rule keeps both; the element-wise rule that
    would be the obvious implementation keeps one, which is exactly how a Kramers splitting gets
    manufactured out of a numerical threshold.
    """
    norms = np.array([1.0e-9, 5.0e-4, 1.0e-16])
    denominators = np.array([2.0, 2.0, 7.0])
    keep = group_complete_mask(norms, denominators, cutoff=1e-6)
    assert list(keep) == [True, True, False], "a degenerate group was cut in half"
    element_wise = np.abs(norms) > 1e-6
    assert list(element_wise) == [False, True, False], "the companion rule must disagree"


def test_a_dropped_group_is_reported_and_changes_the_energy():
    ctx = ClassContext(blocks=None, provider=None, eps_inactive=np.zeros(0),
                       eps_virtual=np.zeros(0), norm_cutoff=1e-6)
    norms = np.array([1.0e-9, 1.0e-9, 5.0e-4])
    denominators = np.array([2.0, 2.0, 3.0])
    result = class_energy("test", norms, denominators, ctx)
    assert result.n_dropped == 2
    assert result.energy == pytest.approx(-5.0e-4 / 3.0)


def test_level_shifts_reduce_the_magnitude_of_every_contribution():
    ctx = ClassContext(blocks=None, provider=None, eps_inactive=np.zeros(0),
                       eps_virtual=np.zeros(0))
    norms = np.array([1.0e-3, 2.0e-3])
    denominators = np.array([0.5, 1.5])
    plain = class_energy("test", norms, denominators, ctx).energy
    real_shift = class_energy("test", norms, denominators,
                              ClassContext(blocks=None, provider=None,
                                           eps_inactive=np.zeros(0), eps_virtual=np.zeros(0),
                                           shift=0.2)).energy
    imag_shift = class_energy("test", norms, denominators,
                              ClassContext(blocks=None, provider=None,
                                           eps_inactive=np.zeros(0), eps_virtual=np.zeros(0),
                                           shift=0.2, imaginary_shift=True)).energy
    assert plain < real_shift < 0.0
    assert plain < imag_shift < 0.0


# --- 3. the driver, on a real front-end -------------------------------------------------------

def spinor_setup(atom, basis="sto-3g", *, charge=0, spin=0, ncas=2, nelecas=2):
    """A scalar SCF turned into the spinor problem, with SOC off.

    ⚠ The inactive count comes from the **electron** count, never from ``mo_occ > 0``:
    an ROHF singly occupied orbital has ``occ > 0`` while holding one electron, and the
    expression that ignores that puts extra electrons in the calculation *and* can split a
    Kramers pair across a space boundary.
    """
    pytest.importorskip("pyscf.mcscf")
    from pyscf import ao2mo, gto, scf

    from kuiva.integrals.transform import ThreeIndexAO
    from kuiva.spinor.expand import spin_block_diagonal

    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).run(conv_tol=1e-12)

    nao, nmo = mol.nao, mf.mo_coeff.shape[1]
    n_inactive_elec = mol.nelectron - nelecas
    assert n_inactive_elec % 2 == 0, "an odd inactive electron count would split a Kramers pair"
    ncore = n_inactive_elec // 2
    factors = ThreeIndexAO.from_eri(ao2mo.restore(1, mf._eri, nao), nao, 1e-13, report=False)
    h_ao = spin_block_diagonal(mf.get_hcore())
    coeff = np.zeros((2 * nao, 2 * nmo), dtype=np.complex128)
    coeff[:nao, 0::2] = mf.mo_coeff                     # interleaved Kramers pairs
    coeff[nao:, 1::2] = mf.mo_coeff
    spaces = OrbitalSpaces.from_counts(2 * ncore, 2 * ncas, 2 * nmo)
    return dict(mol=mol, mf=mf, ncore=ncore, ncas=ncas, factors=factors, h_ao=h_ao,
                coeff=coeff, spaces=spaces, nelecas=nelecas, e_nuc=mol.energy_nuc())


def lih_setup(basis="sto-3g", distance=1.6, ncas=2, nelecas=2):
    """LiH plus the PySCF ``CASCI`` object the Tier-1 comparison is made against.

    ⚠ **CASCI, not CASSCF, and the difference matters for the Tier-1 lock.** At converged
    CASSCF orbitals PySCF canonicalizes with a density that is not exactly the CASCI density at
    those orbitals, so its ``eps`` differ from a freshly built Fock's in the seventh digit and
    the class energies with them — measured here, and it is a property of the reference, not of
    Kuiva. At *fixed* orbitals both codes solve the same unique CI problem and the comparison is
    exact, which is what a correctness test wants ("the test is correctness, not accuracy").
    """
    from pyscf import mcscf

    setup = spinor_setup("Li 0 0 0; H 0 0 {}".format(distance), basis,
                         ncas=ncas, nelecas=nelecas)
    setup["mc"] = mcscf.CASCI(setup["mf"], ncas, nelecas)
    setup["mc"].kernel()
    return setup


@pytest.fixture(scope="module")
def lih():
    return lih_setup()


def kuiva_casci(setup, coeff=None, n_states=1):
    from kuiva.mcscf.casci import casci
    return casci(setup["factors"], setup["h_ao"],
                 setup["coeff"] if coeff is None else coeff,
                 setup["spaces"], setup["nelecas"], n_states=n_states,
                 e_nuc=setup["e_nuc"], report=False, enforce_kramers=False)


def test_pseudo_canonicalization_diagonalizes_the_fock_and_leaves_the_active_space_alone(lih):
    """⚠ Tested through the object that gets used downstream, through the object used downstream.

    Rotate the coefficients, **rebuild** the Fock from them, and assert that the inactive and
    virtual blocks are diagonal with the reported spectrum. Checking the eigendecomposition in
    isolation would pass with the conjugation the wrong way round, because a density and a
    coefficient matrix transform oppositely and both look plausible afterwards.
    """
    result = kuiva_casci(lih)
    spaces = lih["spaces"]
    canonical = pseudo_canonicalize(lih["factors"], lih["h_ao"], lih["coeff"], spaces,
                                    result.gamma, e_nuc=lih["e_nuc"])
    ints = CASIntegrals.build(lih["factors"], lih["h_ao"], canonical.coeff, spaces,
                              e_nuc=lih["e_nuc"])
    fock = averaged_fock(ints, lih["factors"], canonical.coeff, result.gamma)
    for idx, eps in ((spaces.inactive, canonical.eps_inactive),
                     (spaces.virtual, canonical.eps_virtual)):
        block = fock[np.ix_(idx, idx)]
        assert np.allclose(np.diag(block).real, eps, atol=1e-10)
        assert np.max(np.abs(block - np.diag(np.diag(block)))) < 1e-10

    # The active space is untouched, so the CI vectors solved before it are still exact.
    assert canonical.active_drift < 1e-10
    assert np.allclose(canonical.coeff[:, spaces.active], lih["coeff"][:, spaces.active])


def test_e2_is_invariant_under_rotations_inside_the_inactive_and_virtual_spaces(lih):
    """A theorem both SC and FIC must satisfy: the classes contract over complete label sets.

    It catches any formula that quietly assumed canonical labels beyond the ``eps`` themselves,
    and it is the reason the driver owns the canonicalization instead of trusting its input.
    """
    reference = sc_nevpt2(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"],
                          kuiva_casci(lih).vectors, lih["nelecas"],
                          energies=kuiva_casci(lih).energies, e_nuc=lih["e_nuc"], report=False)
    rng = np.random.default_rng(4)
    mixed = lih["coeff"].copy()
    for idx in (lih["spaces"].inactive, lih["spaces"].virtual):
        kappa = (rng.normal(size=(idx.size,) * 2) + 1j * rng.normal(size=(idx.size,) * 2))
        kappa = kappa - kappa.conj().T
        w, v = np.linalg.eigh(1j * kappa)
        mixed[:, idx] = mixed[:, idx] @ ((v * np.exp(-1j * w)) @ v.conj().T)
    rotated_ci = kuiva_casci(lih, coeff=mixed)
    rotated = sc_nevpt2(lih["factors"], lih["h_ao"], mixed, lih["spaces"], rotated_ci.vectors,
                        lih["nelecas"], energies=rotated_ci.energies, e_nuc=lih["e_nuc"],
                        report=False)
    for name in implemented_classes():
        assert rotated.class_energies[name][0] == pytest.approx(
            reference.class_energies[name][0], abs=INVARIANCE_TOL, rel=1e-9), name
    for name in implemented_classes():
        assert rotated.class_norms[name][0] == pytest.approx(
            reference.class_norms[name][0], abs=INVARIANCE_TOL, rel=1e-9), name


def test_e2_is_invariant_inside_a_degenerate_eps_block(lih):
    """⚠ **The requirement that shaped the contraction**, and the one that a subspace-rotation
    test cannot see.

    The pseudo-canonicalization determines the orbitals only up to a unitary *inside* each
    degenerate ``eps`` block, and with SOC off every block is at least a Kramers pair, so this
    freedom is always present. Mixing whole subspaces (the test above) is undone by the
    canonicalization and therefore says nothing about it; mixing inside a block is not undone
    and is what a per-spinor contraction fails.

    Measured before the fix: ``Sir (0')`` moved by 1.3e-11 Eh on a 3.3e-6 Eh class energy —
    4e-6 relative, entirely arbitrary, and invisible in every other class. Contracting over the
    whole group makes it a theorem: the group's accumulated norm and denominator are traces of
    Gram matrices over the group, which no unitary can move.
    """
    from kuiva.pt.classes import degeneracy_groups

    result = kuiva_casci(lih)
    spaces = lih["spaces"]
    canonical = pseudo_canonicalize(lih["factors"], lih["h_ao"], lih["coeff"], spaces,
                                    result.gamma, e_nuc=lih["e_nuc"])
    # Both blocks must actually be degenerate, or the test proves nothing.
    assert canonical.eps_inactive.size > 1
    assert len(set(degeneracy_groups(canonical.eps_virtual))) < canonical.eps_virtual.size

    def run(coeff):
        states = kuiva_casci(lih, coeff=coeff)
        return sc_nevpt2(lih["factors"], lih["h_ao"], coeff, spaces, states.vectors,
                         lih["nelecas"], energies=states.energies, e_nuc=lih["e_nuc"],
                         report=False)

    reference = run(canonical.coeff)
    rng = np.random.default_rng(5)
    mixed = canonical.coeff.copy()
    for idx, eps in ((spaces.inactive, canonical.eps_inactive),
                     (spaces.virtual, canonical.eps_virtual)):
        for group in set(degeneracy_groups(eps)):
            block = idx[degeneracy_groups(eps) == group]
            if block.size < 2:
                continue
            kappa = rng.normal(size=(block.size,) * 2) + 1j * rng.normal(size=(block.size,) * 2)
            kappa = kappa - kappa.conj().T
            w, v = np.linalg.eigh(1j * kappa)
            mixed[:, block] = mixed[:, block] @ ((v * np.exp(-1j * w)) @ v.conj().T)
    rotated = run(mixed)
    for name in implemented_classes():
        assert rotated.class_energies[name][0] == pytest.approx(
            reference.class_energies[name][0], abs=1e-15, rel=1e-11), name


def test_a_full_active_space_has_an_empty_first_order_interacting_space(lih):
    """The null test: CAS over every spinor leaves no perturber, so ``E2 = 0`` class by class.

    Exact, not approximate — there is nothing to sum — so it is the cheapest end-to-end check
    that the label enumeration is right, and the one to run first when something breaks.
    """
    from kuiva.mcscf.casci import casci

    n_orb = lih["coeff"].shape[1]
    spaces = OrbitalSpaces(inactive=np.zeros(0, dtype=int), active=np.arange(n_orb),
                           virtual=np.zeros(0, dtype=int), n_orb=n_orb)
    n_elec = lih["mol"].nelectron
    result = casci(lih["factors"], lih["h_ao"], lih["coeff"], spaces, n_elec,
                   e_nuc=lih["e_nuc"], report=False, enforce_kramers=False)
    corrected = sc_nevpt2(lih["factors"], lih["h_ao"], lih["coeff"], spaces, result.vectors,
                          n_elec, energies=result.energies, e_nuc=lih["e_nuc"], report=False)
    assert corrected.e2[0] == 0.0
    for name in implemented_classes():
        assert corrected.class_energies[name][0] == 0.0
        assert corrected.class_norms[name][0] == pytest.approx(0.0, abs=1e-14)


def test_batching_does_not_change_the_answer(lih, monkeypatch):
    """The transient budget decides the block size and nothing else (kernel rule B7).

    ⚠ **To rounding, not bitwise, and the distinction is a statement about the code.** Changing
    the batch size reorders two floating-point reductions — the auxiliary blocking inside
    ``transform_3c`` and, for ``Sijrs`` alone, the per-batch accumulation of the energy sum, the
    one class that is summed batch by batch because it needs no group-complete cutoff. Both are
    the kind of reduction kernel rule B10 requires to be *named* rather than assumed stable, so
    the tolerance here is a rounding budget and nothing looser.
    """
    result = kuiva_casci(lih)
    args = (lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"], result.vectors,
            lih["nelecas"])
    kwargs = dict(energies=result.energies, e_nuc=lih["e_nuc"], report=False)
    whole = sc_nevpt2(*args, **kwargs)
    monkeypatch.setattr(ptblocks.res, "transient_gb", lambda **kw: 1e-9)
    split = sc_nevpt2(*args, **kwargs)
    for name in implemented_classes():
        assert split.class_energies[name][0] == pytest.approx(
            whole.class_energies[name][0], rel=1e-13), name
        assert split.class_norms[name][0] == pytest.approx(
            whole.class_norms[name][0], rel=1e-13), name


def test_the_state_independent_class_is_computed_once_and_is_the_same_number(li2):
    """``Sijrs`` contains no active-space operator, so with the state-averaged Fock every state
    gets the same answer — and it is the most expensive class on a real system, so it is
    computed once rather than once per state.

    ⚠ Both halves are asserted, because either alone is a trap: that the reuse *happened* (the
    same object appears in every state's table, so a regression that silently recomputes it
    costs time nobody measures), and that the reused number is **bitwise** what recomputing
    per state gives — checked by turning the declaration off and running again, which is the
    only way to see that the reuse is not hiding a stale context.
    """
    from dataclasses import replace as _replace

    from kuiva.pt import classes as ptclasses

    two = kuiva_casci(li2, n_states=2)
    args = (li2["factors"], li2["h_ao"], li2["coeff"], li2["spaces"])
    kwargs = dict(e_nuc=li2["e_nuc"], report=False)
    paired = sc_nevpt2(*args, two.vectors, li2["nelecas"], energies=two.energies, **kwargs)
    assert paired.per_state[0]["Sijrs"] is paired.per_state[1]["Sijrs"]

    spec = excitation_class("Sijrs")
    ptclasses.register_class(_replace(spec, state_independent=False))
    try:
        recomputed = sc_nevpt2(*args, two.vectors, li2["nelecas"], energies=two.energies,
                               **kwargs)
    finally:
        ptclasses.register_class(spec)
    assert recomputed.per_state[0]["Sijrs"] is not recomputed.per_state[1]["Sijrs"]
    for state in (0, 1):
        assert (recomputed.class_energies["Sijrs"][state]
                == paired.class_energies["Sijrs"][state])

    # ...and a state-specific Fock moves the orbitals between states, so nothing is shared.
    specific = sc_nevpt2(*args, two.vectors, li2["nelecas"], energies=two.energies,
                         fock="state-specific", **kwargs)
    assert specific.per_state[0]["Sijrs"] is not specific.per_state[1]["Sijrs"]


def test_kramers_partners_get_the_same_correction():
    """⚠ With the time-even state-averaged Fock this is a theorem, not a tolerance.

    A one-electron active space on a Li atom: the lowest level is a Kramers doublet and the two
    members must receive identical ``E2``. It is the check that would catch a state-specific
    Fock leaking into the default path, and the SC-NEVPT2 instance of the Kramers-degeneracy
    requirement the CI itself carries.

    ⚠ Two roots, not more, and for a reason: **any** orthonormal basis of a *two*-dimensional
    degenerate space is a Kramers pair (``T`` maps a member to something in the space and
    orthogonal to it, since ``T^2 = -1``), so the theorem holds however the eigensolver mixed
    them. That argument fails for a four-fold manifold, where the arbitrary mixing inside the
    block is real and is mechanism M1 of the multiplet diagnosis below rather than a bug.
    """
    setup = spinor_setup("Li 0 0 0", "sto-3g", spin=1, ncas=2, nelecas=1)
    result = kuiva_casci(setup, n_states=2)
    corrected = sc_nevpt2(setup["factors"], setup["h_ao"], setup["coeff"], setup["spaces"],
                          result.vectors, setup["nelecas"], energies=result.energies,
                          e_nuc=setup["e_nuc"], report=False)
    assert corrected.e2.size == 2
    for name in implemented_classes():
        pair = corrected.class_energies[name]
        assert pair[0] == pytest.approx(pair[1], abs=1e-12, rel=1e-10), name
    assert corrected.e2[0] == pytest.approx(corrected.e2[1], abs=1e-12)


# --- frozen core and deleted virtuals -----------------------------------------------------

def test_freezing_a_core_orbital_restricts_the_labels_and_nothing_else():
    """⚠ the frozen-core semantics, against brute force: a frozen spinor keeps its **mean field**.

    It stays in ``F^I`` and in ``e_core`` and disappears only from the ``i, j`` label ranges of
    every class. The distinction is the whole approximation: an implementation that instead
    projected the orbital out of ``H`` would change ``H_act``, the reference energy and every
    class, and would agree with nothing here. The reference restricts its own label
    enumeration and nothing else, so the two agree only if Kuiva does the same.
    """
    inactive, active, virtual, n_elec = [0, 1, 2], [3, 4, 5], [6, 7], 2
    h, eri, factor = factorized_integrals(8, seed=11)
    rng = np.random.default_rng(3)
    eps_i = np.sort(rng.normal(loc=-2.0, size=len(inactive)))
    eps_v = np.sort(rng.normal(loc=+1.5, size=len(virtual)))
    frozen = [inactive[0]]
    core = [i for i in inactive if i not in frozen]

    ref = ReferenceNEVPT2(h, eri, inactive, active, virtual, eps_i, eps_v, n_elec,
                          frozen=frozen)
    space = CASSpace(len(active), n_elec)
    provider = CIContractionProvider(
        space, np.ascontiguousarray(ref.cas_vectors[:, 0].astype(complex)),
        ref.f_inactive[np.ix_(active, active)], eri[np.ix_(active, active, active, active)])
    ctx = ClassContext(
        blocks=StubBlocks(factor, {"inactive": core, "active": active, "virtual": virtual,
                                   "all": list(range(8))}),
        provider=provider, eps_inactive=eps_i[1:], eps_virtual=eps_v,
        fock_vi=ref.f_inactive[np.ix_(virtual, core)],
        fock_va=ref.f_inactive[np.ix_(virtual, active)],
        fock_ai=ref.f_inactive[np.ix_(active, core)])
    for name in ALL_CLASSES:
        norm, energy = ref.by_name(name)
        got = excitation_class(name).evaluate(ctx)
        assert got.norm == pytest.approx(norm, abs=BRUTE_FORCE_TOL, rel=BRUTE_FORCE_TOL), name
        assert got.energy == pytest.approx(energy, abs=BRUTE_FORCE_TOL,
                                           rel=BRUTE_FORCE_TOL), name


def test_a_threshold_that_would_split_a_degenerate_shell_is_refused():
    """⚠ Refuse, never round. Freezing half a shell is the group-completeness rule
    broken at the orbital level, and the resulting Kramers splitting is indistinguishable from
    a physical one downstream.

    The companion that must pass: a threshold placed *between* two groups is accepted, so the
    refusal is about the degeneracy and not about thresholds in general.

    ⚠ The members are made to differ by 2e-10 rather than being bitwise equal, because that is
    what a real Kramers pair does — no threshold can fall between two identical numbers, and a
    test built on identical numbers would prove the guard unreachable rather than working.
    """
    from kuiva.pt.nevpt2 import select_correlated

    eps_i = np.array([-2.5, -2.5 + 2e-10, -1.0, -1.0 + 2e-10])
    eps_v = np.array([0.5, 0.5 + 2e-10, 0.5 + 3e-10, 0.5 + 4e-10, 2.0, 2.0])
    ok = select_correlated(eps_i, eps_v, frozen_core=-1.5, deleted_virtual=1.0)
    assert list(ok.inactive) == [2, 3] and ok.n_frozen == 2
    assert list(ok.virtual) == [0, 1, 2, 3] and ok.n_deleted == 2
    with pytest.raises(ValueError, match="cuts through a degenerate group"):
        select_correlated(eps_i, eps_v, frozen_core=-2.5 + 1e-10)
    with pytest.raises(ValueError, match="cuts through a degenerate group"):
        select_correlated(eps_i, eps_v, deleted_virtual=0.5 + 2.5e-10)


@pytest.fixture(scope="module")
def li2():
    """Li2 at its experimental bond length: two inactive Kramers pairs, so a core can be
    frozen without emptying the space. LiH has only one and cannot exercise the frozen-core machinery at all."""
    return spinor_setup("Li 0 0 0; Li 0 0 2.673", "sto-3g", ncas=2, nelecas=2)


def test_a_threshold_below_every_core_orbital_is_a_bitwise_no_op(li2):
    """The trivial limit, asserted because it is the one an implementation can get wrong for
    free: selecting *everything* must reproduce the unrestricted run exactly, not to rounding.
    """
    result = kuiva_casci(li2)
    args = (li2["factors"], li2["h_ao"], li2["coeff"], li2["spaces"], result.vectors,
            li2["nelecas"])
    kwargs = dict(energies=result.energies, e_nuc=li2["e_nuc"], report=False)
    plain = sc_nevpt2(*args, **kwargs)
    selected = sc_nevpt2(*args, frozen_core=-1e6, deleted_virtual=1e6, **kwargs)
    assert selected.n_frozen == 0 and selected.n_deleted == 0
    for name in ALL_CLASSES:
        assert selected.class_energies[name][0] == plain.class_energies[name][0], name


def test_freezing_removes_exactly_the_classes_with_a_core_label(li2, kuiva_caplog):
    """Freezing *every* core spinor zeroes the six classes with an ``i`` label and leaves the
    two without one **bitwise** unchanged — which is the sharpest statement available that the
    frozen selection touches the label ranges and nothing in the active space or the Fock.
    """
    result = kuiva_casci(li2)
    args = (li2["factors"], li2["h_ao"], li2["coeff"], li2["spaces"], result.vectors,
            li2["nelecas"])
    kwargs = dict(energies=result.energies, e_nuc=li2["e_nuc"], report=False)
    plain = sc_nevpt2(*args, **kwargs)
    kuiva_caplog.clear()
    allfrozen = sc_nevpt2(*args, frozen_core=1e6, **kwargs)
    assert allfrozen.n_frozen == li2["spaces"].n_inactive
    assert any("removes every spinor" in r.getMessage() for r in kuiva_caplog.records)
    for name in ("Sijrs", "Srsi", "Sijr", "Sij", "Sir", "Si"):
        assert allfrozen.class_energies[name][0] == 0.0, name
    for name in ("Srs", "Sr"):
        assert allfrozen.class_energies[name][0] == plain.class_energies[name][0], name
    # ...and the reference energy is untouched: freezing changes the perturbation, not the
    # wavefunction it corrects.
    assert allfrozen.e_casscf[0] == plain.e_casscf[0]


def test_freezing_a_shell_shrinks_the_correction_and_reports_it(li2):
    """The ordinary case: freeze the lower Li 1s Kramers pair of the two."""
    result = kuiva_casci(li2)
    args = (li2["factors"], li2["h_ao"], li2["coeff"], li2["spaces"], result.vectors,
            li2["nelecas"])
    kwargs = dict(energies=result.energies, e_nuc=li2["e_nuc"], report=False)
    plain = sc_nevpt2(*args, **kwargs)
    assert plain.eps_inactive.size == 4
    threshold = 0.5 * float(plain.eps_inactive[1] + plain.eps_inactive[2])
    frozen = sc_nevpt2(*args, frozen_core=threshold, **kwargs)
    assert frozen.n_frozen == 2
    assert abs(frozen.e2[0]) < abs(plain.e2[0])
    # The classes with no core label cannot move at all.
    for name in ("Srs", "Sr"):
        assert frozen.class_energies[name][0] == plain.class_energies[name][0], name


# --- size consistency --------------------------------------------------------------------

def test_e2_is_size_consistent_on_a_non_interacting_pair():
    """``E2(AB) = E2(A) + E2(B)`` for a product-structured CAS — a theorem, and the one
    property a perturbation theory can lose without any test noticing.

    ⚠ **Two Li2 molecules, not two LiH**, and 50 A rather than the suite's 25 A. LiH carries a
    6 D dipole, so a pair of them still interacts by 3e-5 Eh at 25 A *at the CASCI level* — the
    reference itself is then not additive and there is nothing left to measure. Li2 is
    non-polar, and the residual falls by two orders between 25 and 50 A, which is what says the
    remainder is physical interaction rather than a size-consistency defect.

    The tolerance is therefore stated against the reference's own non-additivity as well as in
    absolute terms: the perturbation may not add size-inconsistency the CASCI did not have.
    """
    geom = "Li 0 0 0; Li 0 0 2.673"
    far = geom + "; Li 50.0 0 0; Li 50.0 0 2.673"

    def run(atom, ncas, nelecas):
        setup = spinor_setup(atom, "sto-3g", ncas=ncas, nelecas=nelecas)
        states = kuiva_casci(setup)
        return states, sc_nevpt2(setup["factors"], setup["h_ao"], setup["coeff"],
                                 setup["spaces"], states.vectors, setup["nelecas"],
                                 energies=states.energies, e_nuc=setup["e_nuc"], report=False)

    mono_ci, mono = run(geom, 2, 2)
    pair_ci, pair = run(far, 4, 4)
    reference_defect = abs(2 * mono_ci.total_energies[0] - pair_ci.total_energies[0])
    assert reference_defect < 1e-7, "the CASCI reference must itself be additive"
    assert abs(2 * mono.e2[0] - pair.e2[0]) < 1e-9
    assert abs(2 * mono.e2[0] - pair.e2[0]) < reference_defect
    for name in ALL_CLASSES:
        assert 2 * mono.class_energies[name][0] == pytest.approx(
            pair.class_energies[name][0], abs=1e-9), name


# --- the SOC-on brute-force reference ----------------------------------------------------

def test_every_class_matches_the_uncontracted_reference_with_soc_on():
    """⚠ The only like-for-like check of the SOC-on algebra that exists.

    ``tests/reference/nevpt2_uncontracted.json`` holds HI's eight class energies as computed by
    explicit projection over a Fock space of the eight retained spinors — dense ladder
    operators, ``H_D`` as a matrix, no NEVPT2 formula anywhere — from integrals that came
    through the front end with ``with_soc=True``. The stored ``eri_imaginary_over_real`` says
    the integrals are genuinely complex rather than complex-by-dtype, which is what makes this
    a spin-orbit check and not an arithmetic formality.

    ⚠ It is a *stored* comparison, so it also locks the frozen-core and deleted-virtual
    thresholds it was generated with: what is being re-run is the whole production path down to
    the integral blocks. ``tests/generate/nevpt2_uncontracted.py --check`` regenerates it.
    """
    import json
    import pathlib

    stored = json.loads(
        (pathlib.Path(__file__).resolve().parent
         / "reference/nevpt2_uncontracted.json").read_text())
    assert stored["with_soc"] and stored["eri_imaginary_over_real"] > 1e-3
    assert stored["dyall_residual"] < 1e-12, "the reference's own consistency check"
    assert set(stored["classes"]) == set(ALL_CLASSES)
    for name, entry in stored["classes"].items():
        assert entry["energy_kuiva"] == pytest.approx(entry["energy"], abs=1e-14,
                                                      rel=1e-10), name
        assert entry["norm_kuiva"] == pytest.approx(entry["norm"], abs=1e-14, rel=1e-10), name
        # The Hylleraas hierarchy, with SOC on and on a real molecule.
        assert entry["energy_uncontracted"] <= entry["energy"] + 1e-14, name
        assert entry["energy"] <= 1e-14, name
    assert stored["e2_kuiva"] == pytest.approx(stored["e2_reference"], rel=1e-10)


@pytest.mark.slow
def test_the_soc_on_reference_still_reproduces_itself():
    """Re-run the generator and compare against what it committed (a stored reference that
    is never regenerated is a number nobody can defend)."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "generate"))
    import json
    import pathlib

    import nevpt2_uncontracted as gen

    stored = json.loads(
        (pathlib.Path(__file__).resolve().parent
         / "reference/nevpt2_uncontracted.json").read_text())
    fresh = gen.build(stored["key"])
    for name, entry in stored["classes"].items():
        assert fresh["classes"][name]["energy"] == pytest.approx(entry["energy"], rel=1e-9), name
        assert fresh["classes"][name]["energy_kuiva"] == pytest.approx(
            entry["energy_kuiva"], rel=1e-9), name


# --- the multiplet-splitting diagnosis: M1, M2 and the C1 reporting discipline -------------

def beryllium_setup():
    """Be / 6-31G, CAS(2 e, 4 spatial = 8 spinors) — an **exactly** 9-fold degenerate manifold.

    ⚠ **Every ingredient of that sentence is load-bearing for M1, and three cheaper systems are
    not usable.**

    * **Closed-shell RHF**, so the reference density is spherical and the three 2p orbitals are
      degenerate *by symmetry rather than by convergence*: the ``3P`` block of ``2s2p`` comes
      out degenerate to 0.0 cm^-1, not to the Davidson noise floor. An open-shell atom (C, O)
      at aufbau ROHF orbitals has a non-spherical density, the p orbitals split, and there is
      no degenerate manifold to rotate at all — that is why the *measurement* script has to
      run a state-averaged CASSCF and why this test cannot.
    * **Nine-fold**, so M1 is live. A Kramers doublet's two members are related by
      time reversal whatever basis the eigensolver returned, so M1 is zero there by theorem
      (:func:`test_kramers_partners_get_the_same_correction`).
    * **Two active electrons, not one.** Measured: with a *one*-electron active space (Li, a
      4-fold 2p manifold) the rotation moves nothing — 1e-11 cm^-1, i.e. rounding — because a
      one-electron state's density carries no correlation for the perturbation to see
      state-specifically. M1 needs a many-electron manifold to exist.
    * **6-31G, not STO-3G**: with no virtual spinors at all six of the eight classes are empty
      and the rotation again moves nothing (2e-13 cm^-1, measured). A minimal basis would give
      a green test of nothing.
    """
    setup = spinor_setup("Be 0 0 0", "6-31g", spin=0, ncas=4, nelecas=2)
    assert setup["spaces"].n_virtual > 0, "no perturber space: see the docstring"
    return setup


def _haar(n, rng):
    """Haar-distributed U(n). The diagonal phase fix is what makes it Haar rather than
    whatever convention LAPACK's QR happens to leave behind — and a biased 'random' rotation
    would be biased toward exactly the basis this test varies."""
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


def test_the_ci_block_basis_moves_the_members_and_not_the_barycentre():
    """Mechanism M1, and the whole justification for the C1 reporting discipline.

    The CI returns an **arbitrary** unitary mix of a degenerate manifold. SC-NEVPT2 is state
    specific through the per-state RDMs that build every norm and every Koopmans matrix, so the
    individual ``E2`` depend on that arbitrary choice — while the manifold's barycentre, to
    measurement, does not.

    The rotated vectors are still exact eigenvectors of ``H_act`` with the same eigenvalue, so
    the reference spectrum, ``H0`` and every integral are bitwise unchanged and the difference
    is the mechanism with nothing subtracted and nothing modelled.

    ⚠ **This is a guard that can fail in both directions, which is the point.** The first
    assertion is that the members *do* move: a version of this test on a system where they do
    not (a one-electron CAS, or a minimal basis with no virtuals — both measured, see
    :func:`beryllium_setup`) would pass the invariance half while proving nothing. The second
    is that the barycentre is orders more stable.

    ⚠ **The barycentre's stability is a measurement, never a theorem, and must not be quoted as
    one.** ``E2 = -sum_l N_l / dE_l`` is not linear in the state's density — the denominators
    are state dependent too — so no symmetry argument makes the mean of a manifold invariant.
    What is asserted is the ratio actually observed, with three orders of headroom.
    """
    setup = beryllium_setup()
    result = kuiva_casci(setup, n_states=12)

    def correct(vectors):
        return sc_nevpt2(setup["factors"], setup["h_ao"], setup["coeff"], setup["spaces"],
                         vectors, setup["nelecas"], energies=result.energies,
                         e_nuc=setup["e_nuc"], report=False)

    base = correct(result.vectors)
    live = [b for b in base.multiplets() if b.size >= 3]
    assert live, "no manifold to rotate: the fixture no longer tests what it claims"
    block = max(live, key=lambda b: b.size)
    # ⚠ 1e-9 cm^-1, not "converged to the CI threshold": the degeneracy here is a symmetry of
    # the matrix being diagonalized, so what is left is the eigensolver's rounding. A
    # state-averaged CASSCF on an open-shell atom leaves 1e-6..1e-4 instead, which is why the
    # measurement script reports the reference spread beside the corrected one and this fixture
    # does not have to.
    assert block.size == 9 and block.reference_spread_cm < 1.0e-9

    from kuiva.props.multiplet import HARTREE_TO_CM
    members = [np.asarray(base.e2)]
    for trial in range(2):
        rng = np.random.default_rng(20260809 + trial)
        rotated = np.array(result.vectors, dtype=np.complex128, copy=True)
        sl = slice(block.start, block.start + block.size)
        rotated[sl] = _haar(block.size, rng) @ rotated[sl]
        # The rotation must not have moved the *space*: same eigenvalues, still normalized.
        assert np.allclose(rotated[sl] @ rotated[sl].conj().T, np.eye(block.size), atol=1e-12)
        members.append(np.asarray(correct(rotated).e2))
    stack = np.array(members)[:, block.start:block.start + block.size] * HARTREE_TO_CM

    across = float(np.max(np.ptp(stack, axis=0)))       # how far one member moves: M1 itself
    barycentre = float(np.ptp(np.mean(stack, axis=1)))  # what C1 would report instead
    assert across > 1.0e-3, ("the members did not move, so this system cannot see M1 and the "
                             "invariance below proves nothing (measured: 4e-01 cm^-1)")
    assert barycentre < 1.0e-3 * across, (
        "the barycentre moved comparably to the members, which is the premise of the C1 "
        "reporting discipline failing (measured ratio: 1e-04)")


def test_manifolds_are_grouped_by_the_reference_spectrum_not_the_corrected_one():
    """⚠ Grouping the corrected energies would hide exactly the splitting this exists to find.

    A manifold that the correction splits would *become* two manifolds under corrected-spectrum
    grouping, each with zero internal spread, and the diagnostic would report a clean result
    for the one case it is there to catch. Built by hand, because the whole point is a spectrum
    no converged calculation of this size produces.
    """
    from kuiva.pt.nevpt2 import NEVPT2Result

    e_ref = np.array([0.0, 0.0, 0.0, 1.0])                    # a 3-fold manifold and a singlet
    e2 = np.array([0.0, 1.0e-5, 2.0e-5, 0.0])                 # ... which the correction splits
    result = NEVPT2Result(e_casscf=e_ref, e2=e2, complete=True)
    blocks = result.multiplets()
    assert [(b.start, b.size) for b in blocks] == [(0, 3), (3, 1)]
    from kuiva.props.multiplet import HARTREE_TO_CM
    assert blocks[0].reference_spread_cm == 0.0
    assert blocks[0].corrected_spread_cm == pytest.approx(2.0e-5 * HARTREE_TO_CM)
    # The barycentre correction is the mean, and it is what `e2` reports.
    assert blocks[0].e2 == pytest.approx(1.0e-5)


def test_a_manifold_split_beyond_physical_significance_is_warned_about(kuiva_caplog):
    """⚠ 0.1 cm^-1 is not a numerical tolerance — it is the size at which a
    splitting already implies different physics. The warning therefore says the members are not
    one level, and the barycentre printed beside them is not a summary of them.

    The companion half matters as much: a manifold split *below* it must **not** warn, or the
    warning would fire on every converged calculation's Davidson noise and mean nothing.
    """
    from kuiva.pt.nevpt2 import MULTIPLET_TOL_CM, NEVPT2Result, _report_multiplets
    from kuiva.props.multiplet import HARTREE_TO_CM

    def result_with(spread_cm):
        return NEVPT2Result(e_casscf=np.zeros(3), complete=True,
                            e2=np.array([0.0, 0.5, 1.0]) * spread_cm / HARTREE_TO_CM)

    kuiva_caplog.clear()
    _report_multiplets(result_with(10.0 * MULTIPLET_TOL_CM))
    assert any("split by more than" in r.getMessage() for r in kuiva_caplog.records)

    kuiva_caplog.clear()
    _report_multiplets(result_with(0.01 * MULTIPLET_TOL_CM))
    assert not [r for r in kuiva_caplog.records if r.levelno >= 30]


def test_the_manifold_report_refuses_a_spectrum_it_cannot_group():
    """Without the reference energies there is no spectrum to find manifolds in, and grouping
    the corrected one instead is the defect above. Refused, not guessed."""
    from kuiva.pt.nevpt2 import NEVPT2Result

    with pytest.raises(ValueError, match="reference state energies are not available"):
        NEVPT2Result(e_casscf=np.array([np.nan, np.nan]), e2=np.zeros(2)).multiplets()
    with pytest.raises(ValueError, match="not ascending"):
        NEVPT2Result(e_casscf=np.array([1.0, 0.0]), e2=np.zeros(2)).multiplets()


def test_the_default_report_prints_the_manifold_table_beside_the_per_state_one(kuiva_caplog):
    """⚠ ``report=True`` is the *default* and therefore the production output path, and every
    other test in this file passes ``report=False``. The C1 discipline is that the manifold
    table appears **beside** the per-state table and never instead of it, which is a statement
    about output and is tested as one.
    """
    setup = spinor_setup("Li 0 0 0", "sto-3g", spin=1, ncas=2, nelecas=1)
    result = kuiva_casci(setup, n_states=2)
    kuiva_caplog.clear()
    sc_nevpt2(setup["factors"], setup["h_ao"], setup["coeff"], setup["spaces"], result.vectors,
              setup["nelecas"], energies=result.energies, e_nuc=setup["e_nuc"], report=True)
    text = "\n".join(r.getMessage() for r in kuiva_caplog.records)
    assert "degenerate manifolds of the reference spectrum" in text
    assert "state 0" in text and "state 1" in text, "the per-state tables were replaced"
    assert "E2 (bary) [Eh]" in text and "spread [cm^-1]" in text


def multiplet_record():
    import json
    import pathlib

    return json.loads((pathlib.Path(__file__).resolve().parent
                       / "reference/nevpt2_multiplet.json").read_text())


def test_the_committed_multiplet_measurement_holds_its_claims():
    """⚠ The measurement protocol's verdict, locked: **the correction does not split a degenerate
    manifold by a physically meaningful amount** (``tests/reference/nevpt2_multiplet.json``,
    generated by ``tests/generate/nevpt2_multiplet.py``).

    Three claims, and the middle one is what makes the first interpretable:

    1. Every degenerate manifold of every measured system stays degenerate to well inside
       the 0.1 cm^-1 physical-degeneracy — and, on the free ions, to at or below the spread the **reference
       CI** already carries, so the perturbation adds nothing.
    2. ⚠ **The protocol can detect a splitting**, demonstrated by producing one: the
       state-specific Fock of mechanism M2 splits a manifold *larger than a Kramers pair* by
       0.7 to 340 cm^-1, four to six orders more. Without this the first claim would be
       indistinguishable from a measurement that is simply blind (the guard-that-can-fail
       pattern), and it is simultaneously the measurement of what the time-even Dyall Fock is worth.

       ⚠ **And the complementary half, which is a theorem and was measured rather than
       assumed: M2 cannot split a Kramers DOUBLET.** The partner's density is the time reverse
       of the state's, so its Fock is the time reverse, every ``eps`` is identical and so is
       every class energy — ``ticl3``, whose ligand field leaves nothing but doublets, gives
       4e-07 cm^-1 where the free ions give tens. A test asserting a large M2 everywhere would
       fail on it, and would be asserting the wrong thing.
    3. M1 moves the members far more than it moves the barycentre — the C1 premise, on real
       systems rather than the fixture above.
    """
    from kuiva.pt.nevpt2 import MULTIPLET_TOL_CM

    stored = multiplet_record()
    assert stored["records"], "no systems measured"
    saw_a_split = False
    for record in stored["records"]:
        key, blocks = record["key"], record["manifolds"]
        existence = record["existence"]
        assert record["casscf_converged"], key
        # 1. the physical requirement, against the reference's own spread.
        for block, spread in zip(blocks, existence["corrected_spread_cm"]):
            if block["size"] > 1:
                assert spread < MULTIPLET_TOL_CM, (key, block)
        # 2. the guard that can fail, and its theorem-backed complement.
        worst_default = max(existence["corrected_spread_cm"])
        worst_m2 = max(record["m2"]["corrected_spread_cm"])
        if any(b["size"] > 2 for b in blocks):
            assert worst_m2 > MULTIPLET_TOL_CM, key
            assert worst_m2 > 100.0 * worst_default, key
            saw_a_split = True
        else:
            assert worst_m2 < MULTIPLET_TOL_CM, (
                "{}: every manifold here is a Kramers doublet, whose two members get "
                "time-reversed Focks and therefore identical E2 even state-specifically"
                .format(key))
        # 3. M1: the members move, the barycentre does not.
        for entry in record["m1"].get("per_manifold", []):
            assert entry["barycentre_spread_cm"] < 0.05 * entry["across_rotation_cm"], (
                key, entry["start"])
            assert entry["across_rotation_cm"] < MULTIPLET_TOL_CM, (key, entry["start"])
    assert saw_a_split, ("no measured system has a manifold larger than a Kramers pair, so "
                         "nothing in this record demonstrates that the protocol can see a "
                         "splitting at all")


def test_the_ligand_field_system_is_what_scopes_m4():
    """⚠ M4 is not an implementation artefact and no invariance repairs it, so the record has
    to *scope* it rather than assert it away.

    A state-specific perturbation shifts two nearby multiplets independently, without letting
    them interact; ``|differential shift| / gap`` reaching 1 means it has moved a pair through
    each other, which is the regime a quasi-degenerate treatment exists for. Free-ion multiplets
    are thousands of cm^-1 apart and cannot exhibit it — which is precisely why a ligand-field
    system is in the sweep, and why its absence would make the scoping vacuous.
    """
    stored = multiplet_record()
    by_key = {r["key"]: r for r in stored["records"]}
    assert "ticl3" in by_key, (
        "the ligand-field system is missing from the record, so nothing in it bounds M4 at "
        "small gaps: regenerate with `nevpt2_multiplet.py --only ticl3 --merge`")
    for record in stored["records"]:
        ratios = [p["crossing_ratio"] for p in record["m4"]["pairs"]]
        assert ratios, record["key"]
        # Below 1 the ordering of the multiplets survives the correction. This is the claim the
        # measurement supports; it is a bound on the systems measured, never a general one.
        assert max(ratios) < 1.0, record["key"]


@pytest.mark.slow
def test_the_multiplet_measurement_still_reproduces_itself():
    """A stored reference that is never regenerated is a number nobody can defend.

    Only the cheapest system, and only the ``existence`` half: M1's rotations multiply the cost
    by the rotation count and its numbers are locked by the fast test above.
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "generate"))

    import nevpt2_multiplet as gen

    stored = {r["key"]: r for r in multiplet_record()["records"]}["o"]
    fresh = gen.run_system(gen.resolve("o"), n_rotations=0, with_m2=False)
    assert [(b["start"], b["size"]) for b in fresh["manifolds"]] == \
           [(b["start"], b["size"]) for b in stored["manifolds"]]
    for got, want in zip(fresh["existence"]["e2"], stored["existence"]["e2"]):
        assert got == pytest.approx(want, rel=1e-8)


# --- Tier 1: against PySCF's mrpt.NEVPT, class by class ----------------------------------------

def test_per_class_values_match_pyscf_in_the_scalar_limit():
    """⚠ The strongest external check there is (Tier 1).

    With SOC off the two-component problem is two copies of a scalar one, so Kuiva's spinor
    SC-NEVPT2 must reproduce PySCF's ``mrpt.NEVPT`` **per class** — the same eight-way partition
    of the first-order interacting space, so the comparison is line for line and not merely a
    total. It validates the whole substrate: the integral blocks, the pseudo-canonicalization
    and the driver, against an independent implementation of the same method.

    Kuiva starts from the **uncanonicalized** CASCI orbitals and canonicalizes them itself,
    while PySCF additionally rotates its active space to natural orbitals. Agreement therefore
    also demonstrates the invariance of the class quantities under an active-space rotation —
    the property that will let FIC and a DMRG reference plug into the same classes.

    ⚠ It is also the only external check on the **contraction group**. PySCF labels by spatial
    orbital and Kuiva by degenerate-``eps`` group, which in the scalar limit are different
    partitions — Kuiva's is the coarser one wherever two spatial orbitals are symmetry
    degenerate, as LiH's virtuals are. That they agree to 1e-14 relative on every class says the
    denominators are constant across a symmetry-degenerate group too, which is what makes the
    two definitions the same quantity here. ⚠ With a *per-spinor* contraction — the natural
    first implementation — ``Sir (0')`` disagreed with PySCF by 4e-6 relative, and that
    disagreement was the symptom of the invariance defect, not a methodological difference.
    """
    from pyscf import fci
    from pyscf.mrpt import nevpt2 as pyscf_nevpt

    setup = lih_setup()
    result = kuiva_casci(setup)
    corrected = sc_nevpt2(setup["factors"], setup["h_ao"], setup["coeff"], setup["spaces"],
                          result.vectors, setup["nelecas"], energies=result.energies,
                          e_nuc=setup["e_nuc"], report=False)

    reference = pyscf_nevpt.NEVPT(setup["mc"])
    reference.kernel()
    ncas = setup["mc"].ncas
    ci = reference.load_ci()
    dm1, dm2, dm3 = fci.rdm.make_dm123("FCI3pdm_kern_sf", ci, ci, ncas, reference.nelecas)
    dms = {"1": dm1, "2": dm2, "3": dm3, "4": None}
    eris = pyscf_nevpt._ERIS(reference, reference.mo_coeff)

    # The reference reports Kuiva's own CASCI energy, so a discrepancy in the correction
    # cannot be blamed on a different reference state.
    assert result.total_energies[0] == pytest.approx(setup["mc"].e_tot, abs=1e-10)

    expected = {
        "Sijrs": pyscf_nevpt.Sijrs(reference, eris),
        "Srsi": pyscf_nevpt.Srsi(reference, dms, eris),
        "Sijr": pyscf_nevpt.Sijr(reference, dms, eris),
        "Srs": pyscf_nevpt.Srs(reference, dms, eris),
        "Sij": pyscf_nevpt.Sij(reference, dms, eris),
        "Sir": pyscf_nevpt.Sir(reference, dms, eris),
        # ⚠ The two primed classes are where the two codes' *routes* diverge most: PySCF
        # contracts a stored 3-RDM against the integrals and builds the rank-4 piece from the
        # CI vector on the fly, while Kuiva forms one perturber vector per virtual (or per
        # core hole) and never has a rank-3 or rank-4 object at all. Agreement here is the
        # only external evidence that the label-first contraction is the same quantity.
        "Sr": pyscf_nevpt.Sr(reference, ci, dms, eris),
        "Si": pyscf_nevpt.Si(reference, ci, dms, eris),
    }
    assert set(expected) == set(implemented_classes()) == set(ALL_CLASSES), \
        "every implemented class must be compared, or the test decays as classes land"
    for name, (norm, energy) in expected.items():
        assert corrected.class_norms[name][0] == pytest.approx(norm, abs=PYSCF_CLASS_TOL,
                                                               rel=1e-9), name + " norm"
        assert corrected.class_energies[name][0] == pytest.approx(
            energy, abs=PYSCF_CLASS_TOL, rel=1e-9), name + " energy"

    # ⚠ And the *total*, which is what a user reads. It is not implied by the eight class
    # comparisons: PySCF's `e_corr` is assembled by its own driver, so this also locks the
    # sign convention and the assembly on both sides.
    assert corrected.complete
    assert corrected.e2[0] == pytest.approx(reference.e_corr, abs=PYSCF_CLASS_TOL, rel=1e-9)


# --- 4. structure ------------------------------------------------------------------------------

def test_corrected_energies_reach_the_dump_only_with_their_protocol(lih):
    """⚠ The dump file is a contract, and a corrected one is a **hybrid** that must say so.

    Substituting `E2`-corrected diagonal energies leaves `H` at second-order perturbation
    theory and every `mu` element at the CASSCF states it was built from. That is a legitimate
    protocol and it is *not* the default; what this asserts is that the substitution cannot be
    made without the record, because a file that does not say it outlives the session and
    nothing in it would reveal the mixture.
    """
    from kuiva.props.dump import PropertyMatrices
    from kuiva.pt.nevpt2 import DUMP_PROTOCOL, corrected_property_matrices

    result = kuiva_casci(lih)
    corrected = sc_nevpt2(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"],
                          result.vectors, lih["nelecas"], energies=result.energies,
                          e_nuc=lih["e_nuc"], report=False)
    n = corrected.e2.size
    plain = PropertyMatrices(energies=np.asarray(corrected.e_casscf),
                             mu=np.zeros((3, n, n), dtype=complex),
                             l=np.zeros((3, n, n), dtype=complex),
                             s=np.zeros((3, n, n), dtype=complex))
    merged = corrected_property_matrices(plain, corrected)
    assert np.allclose(merged.energies, corrected.total_energies)
    assert DUMP_PROTOCOL in merged.comments
    assert merged.provenance["nevpt2"]["flavor"] == "SC-NEVPT2"
    assert merged.provenance["nevpt2"]["complete"] is True
    # ...and the original is untouched: `PropertyMatrices` is frozen and this returns a copy.
    assert np.allclose(plain.energies, corrected.e_casscf)
    assert plain.comments == ()

    with pytest.raises(ValueError, match="same spectrum in the same order"):
        corrected_property_matrices(
            PropertyMatrices(energies=np.zeros(n + 1), mu=np.zeros((3, 1, 1), dtype=complex),
                             l=np.zeros((3, 1, 1), dtype=complex),
                             s=np.zeros((3, 1, 1), dtype=complex)), corrected)


def test_the_dependency_runs_one_way():
    """⚠ ``kuiva.pt`` is a post-processing stage: nothing in the calculation path imports it.

    The mirror of ``test_x2c_decouple.py`` and ``test_qc_skeleton.py``. Asserted from the
    sources rather than by habit, because a back-edge — even a convenience re-export in an
    ``__init__`` — is invisible until it forces an import cycle or drags the perturbation layer
    into a CASSCF that never asked for it. If NEVPT2-corrected energies reach the property dump, the
    driver hands them *to* the dump; ``props/`` does not learn that ``pt/`` exists.

    ⚠ One file gets a weaker rule, not an exemption: ``interface/stages.py`` **is** the driver
    the docstring above speaks of — its ``NEVPT2`` and ``PropertyDump`` stages are what hands
    the correction to the dump — so it may import ``kuiva.pt`` **inside the methods that run
    it**, and only there. At module level the rule stands, so importing the class layer still
    costs nothing on the calculation path.
    """
    import ast
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for package in ("ci", "mcscf", "rdm", "x2c", "dmrg", "amf", "integrals", "interface",
                    "props", "spinor", "orth", "basis", "io", "util", "qc"):
        for path in sorted((repo / "kuiva" / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text())
            driver = path.relative_to(repo).as_posix() == "kuiva/interface/stages.py"
            nodes = ast.iter_child_nodes(tree) if driver else ast.walk(tree)
            for node in nodes:
                if isinstance(node, ast.Import):
                    offenders += [(path.name, a.name) for a in node.names
                                  if a.name.split(".")[:2] == ["kuiva", "pt"]]
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if ((node.level == 0 and module.startswith("kuiva.pt"))
                            or (node.level > 0 and module.split(".")[0] == "pt")):
                        offenders.append((path.name, module))
    assert offenders == [], "nothing in the calculation path may import kuiva.pt: {}".format(
        offenders)


def test_the_registry_covers_the_whole_partition():
    """All eight, in Angeli's order, with a status each — see the module docstring on why the
    whole partition is registered rather than only what is implemented."""
    names = available_classes()
    assert set(names) == set(ALL_CLASSES)
    assert len(names) == 8
    deltas = {n: excitation_class(n).delta_n_act for n in names}
    assert sorted(deltas.values()) == [-2, -1, -1, 0, 0, 1, 1, 2]
    assert set(implemented_classes()) == set(ALL_CLASSES)


def test_an_unknown_class_name_is_refused_naming_what_exists():
    with pytest.raises(ValueError, match="unknown NEVPT2 excitation class"):
        excitation_class("Stuvw")


def test_higher_rank_densities_are_refused_rather_than_approximated(brute):
    """⚠ Must refuse, not return a cumulant or a zero, and say *why* it is not needed.

    Neither ``rdm3`` nor ``contract_rdm4`` is a missing feature: the rank-3 requirements are
    served as Gram matrices of ladder-string vectors, and the rank-4 one never becomes an
    object at all because the primed classes contract the integrals into one perturber vector
    per external label first. A stub that read "not implemented yet" would invite someone to
    implement the ``n_act^6`` object the design rejects.
    """
    _, ctx = brute
    with pytest.raises(NotImplementedError, match="no 3-RDM is built"):
        ctx.provider.rdm3()
    with pytest.raises(NotImplementedError, match="no rank-4 quantity is contracted"):
        ctx.provider.contract_rdm4(None)


def test_a_partial_total_says_so(lih, kuiva_caplog):
    """⚠ The whole partition is computed by default, so this asserts the *guard*, not a gap.

    ``E2`` is complete today. The mechanism that refuses to call an incomplete sum ``E2``
    still has to work, because ``classes=`` can restrict the set and a future class could be
    added with a non-``"energy"`` status — either of which would otherwise shrink the total
    silently. Restricting the set is the only way to reach that branch now, and it must
    produce a ``PARTIAL`` warning and ``nan`` (never zero) for what was not computed.
    """
    result = kuiva_casci(lih)
    kuiva_caplog.clear()
    corrected = sc_nevpt2(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"],
                          result.vectors, lih["nelecas"], energies=result.energies,
                          e_nuc=lih["e_nuc"], report=False)
    assert corrected.complete
    assert corrected.missing == ()
    assert not any("PARTIAL" in r.getMessage() for r in kuiva_caplog.records)

    kuiva_caplog.clear()
    restricted = sc_nevpt2(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"],
                           result.vectors, lih["nelecas"], energies=result.energies,
                           e_nuc=lih["e_nuc"], report=False,
                           classes=["Sijrs", "Sr"])
    assert not restricted.complete
    assert set(restricted.missing) == set(ALL_CLASSES) - {"Sijrs", "Sr"}
    assert any("PARTIAL" in r.getMessage() for r in kuiva_caplog.records)
    assert restricted.e2[0] == pytest.approx(
        corrected.class_energies["Sijrs"][0] + corrected.class_energies["Sr"][0], rel=1e-12)
    # ⚠ nan, never zero: a caller summing the dictionary must not get a plausible total.
    assert np.isnan(restricted.class_energies["Srs"][0])


def test_the_state_specific_fock_warns_at_the_point_of_selection(lih, kuiva_caplog):
    """A benchmark-grade option must announce itself, as X2C-mmf and DLU must."""
    result = kuiva_casci(lih)
    kuiva_caplog.clear()
    sc_nevpt2(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"], result.vectors,
              lih["nelecas"], energies=result.energies, e_nuc=lih["e_nuc"],
              fock="state-specific", report=False)
    assert any("per state" in r.getMessage() for r in kuiva_caplog.records)
    with pytest.raises(ValueError, match="fock must be one of"):
        sc_nevpt2(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"], result.vectors,
                  lih["nelecas"], energies=result.energies, e_nuc=lih["e_nuc"],
                  fock="canonical", report=False)


def test_more_than_one_state_needs_its_energies_whatever_the_fock_is(lih):
    """⚠ The state-averaging gate is about the *state selection*, not about the choice of H0.

    A count that splits a degenerate block is as fatal to a state-specific run as to an
    averaged one, so the refusal does not become optional when the averaged density is not the
    thing being built.
    """
    result = kuiva_casci(lih, n_states=2)
    for fock in ("state-averaged", "state-specific"):
        with pytest.raises(ValueError, match="needs the state energies"):
            sc_nevpt2(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"], result.vectors,
                      lih["nelecas"], e_nuc=lih["e_nuc"], fock=fock, report=False)


def test_a_level_shift_warns(lih, kuiva_caplog):
    result = kuiva_casci(lih)
    kuiva_caplog.clear()
    sc_nevpt2(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"], result.vectors,
              lih["nelecas"], energies=result.energies, e_nuc=lih["e_nuc"], shift=0.1,
              report=False)
    assert any("level shift" in r.getMessage() for r in kuiva_caplog.records)


# --- resource sizing (exact, unpadded, two-sided against a real array) -------------------

def test_label_buffer_sizing_is_exact():
    n = 4321
    gb = label_buffer_gb(n)
    real = (2 * np.empty(n, dtype=np.float64).nbytes
            + np.empty(n, dtype=np.int64).nbytes) / 1024.0 ** 3
    assert gb == pytest.approx(real, rel=0, abs=0)
    assert gb <= real                      # ⚠ never padded: a safety factor here is a defect


def test_perturber_vector_sizing_is_exact():
    """The single-external perturber set is a resident array, so it is sized exactly.

    ⚠ Two-sided and unpadded: a sizing function that grows a safety factor must fail here. The
    factor of two is the contract, not a margin — ``H_act`` applied to the set is the same shape
    and lives at the same time.
    """
    from kuiva.pt.contractions import perturber_vector_gb

    n_labels, ndet = 37, 4321
    gb = perturber_vector_gb(ndet, n_labels)
    real = 2 * np.empty((n_labels, ndet), dtype=np.complex128).nbytes / 1024.0 ** 3
    assert gb == pytest.approx(real, rel=0, abs=0)
    assert gb <= real


def test_three_index_block_sizing_is_exact(lih):
    blocks = ptblocks.IntegralBlocks(lih["factors"], lih["coeff"], lih["spaces"])
    try:
        block = blocks.three_index("virtual", "inactive")
        gb = blocks.block_gb("virtual", "inactive")
        assert gb == pytest.approx(block.nbytes / 1024.0 ** 3, rel=0, abs=0)
        assert block.shape == (blocks.naux, lih["spaces"].n_virtual,
                               lih["spaces"].n_inactive)
        assert blocks.three_index("virtual", "inactive") is block      # cached
    finally:
        blocks.release()


def test_the_planned_block_pairs_are_the_ones_a_real_run_asks_for(lih, monkeypatch):
    """⚠ The memory pre-flight is only as good as this list.

    It budgets the **sum** of the blocks SC-NEVPT2 holds at once, taken from
    ``SC_NEVPT2_BLOCK_PAIRS``. A class that starts asking for a fifth pair without extending
    that tuple would leave the pre-flight describing a calculation that is no longer the one
    running — and an under-estimating plan is worse than a pessimistic one, because the whole
    mechanism is refuse-before-allocate. So the list is checked against what a run of every
    implemented class actually requests, not against a reading of the sources.
    """
    requested = set()
    original = ptblocks.IntegralBlocks.three_index

    def recording(self, bra, ket):
        requested.add((bra, ket))
        return original(self, bra, ket)

    monkeypatch.setattr(ptblocks.IntegralBlocks, "three_index", recording)
    result = kuiva_casci(lih)
    sc_nevpt2(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"], result.vectors,
              lih["nelecas"], energies=result.energies, e_nuc=lih["e_nuc"], report=False)
    assert requested == set(ptblocks.SC_NEVPT2_BLOCK_PAIRS)


def test_the_planned_block_total_matches_the_blocks_actually_held(lih):
    """The sum over the four pairs, two-sided against the arrays the cache ends up holding."""
    spaces = lih["spaces"]
    blocks = ptblocks.IntegralBlocks(lih["factors"], lih["coeff"], spaces)
    try:
        for bra, ket in ptblocks.SC_NEVPT2_BLOCK_PAIRS:
            blocks.three_index(bra, ket)
        held = sum(b.nbytes for b in blocks._cache.values()) / 1024.0 ** 3
        estimate = ptblocks.nevpt2_blocks_memory_gb(
            blocks.naux, spaces.n_inactive, spaces.n_active, spaces.n_virtual)
        assert estimate == pytest.approx(held, rel=0, abs=0)
        assert estimate <= held                # ⚠ never padded, exactly as the sizing rule says
    finally:
        blocks.release()


def test_batch_slices_cover_everything_exactly_once():
    slices = ptblocks.batch_slices(10, 1.0, budget_gb=3.0)
    covered = [i for s in slices for i in range(s.start, s.stop)]
    assert covered == list(range(10))
    assert all(s.stop > s.start for s in slices)
    assert ptblocks.batch_slices(0, 1.0) == []
    # A single item that does not fit is still attempted — the resident checks refuse, not this.
    assert ptblocks.batch_slices(3, 100.0, budget_gb=1e-6) == [slice(0, 1), slice(1, 2),
                                                               slice(2, 3)]


# --- the intruder diagnostic: computed, printed, and now actually compared ------------------

def test_class_energy_records_the_signed_denominator_as_well_as_the_absolute_one():
    """⚠ The two answer different questions and the absolute one cannot see the worse case.

    A small ``|dE|`` says the sum is badly conditioned. A **negative** ``dE`` says a perturber
    has fallen below the reference, which makes the class energy wrong in sign as well as in
    size — and ``|dE|`` for that same term can be perfectly comfortable.
    """
    from kuiva.pt.classes import ClassContext, class_energy

    ctx = ClassContext(blocks=None, provider=None, eps_inactive=np.zeros(0),
                       eps_virtual=np.zeros(0), fock_vi=None, fock_va=None, fock_ai=None)
    result = class_energy("Sr", np.array([1.0, 1.0, 1.0]), np.array([2.0, -0.4, 5.0]), ctx,
                          cutoff=False)
    assert result.min_denominator == pytest.approx(0.4)      # |dE|: looks merely tight
    assert result.min_signed_denominator == pytest.approx(-0.4)   # ...and is actually below 0


@pytest.mark.parametrize("min_abs, min_signed, expect", [
    (2.5, 2.5, None),                                  # comfortably outside the band
    (0.05, 0.05, "intruder band"),                     # the conventional warning tier
    (1e-9, 1e-9, "essentially zero"),                  # a vanishing denominator
    (0.4, -0.4, "wrong in sign"),                      # a perturber below the reference
])
def test_a_small_denominator_now_warns_instead_of_only_being_printed(
        min_abs, min_signed, expect, kuiva_caplog):
    """⚠ **The defect.** ``min_denominator`` was computed for every class and printed in the
    per-class table, and was compared against nothing at all — so the one number saying
    whether a class's ``E2`` can be believed reached the user only if the user already knew
    what value to be alarmed by. The bounds are fixed in advance, not fitted to the shipped
    systems.
    """
    from kuiva.pt.classes import ClassResult
    from kuiva.pt.nevpt2 import _warn_on_intruder

    entry = ClassResult(name="Sijrs", norm=1.0, energy=-0.1, n_perturbers=3,
                        min_denominator=min_abs, min_signed_denominator=min_signed)
    _warn_on_intruder(0, "Sijrs", entry, shifted=False)

    warnings = [r.message for r in kuiva_caplog.records if r.levelname == "WARNING"]
    if expect is None:
        assert not warnings
    else:
        assert any(expect in m for m in warnings), warnings


def test_the_bands_are_the_pre_registered_ones(kuiva_caplog):
    """The constants are the contract; a test that read them from the code it checks would
    pass no matter what they became."""
    from kuiva.pt.nevpt2 import INTRUDER_SEVERE_EH, INTRUDER_WARN_EH

    assert INTRUDER_WARN_EH == 0.1
    assert INTRUDER_SEVERE_EH == 1.0e-6


def test_an_applied_shift_does_not_silence_the_intruder_warning(kuiva_caplog):
    """⚠ A level shift bounds the damage; it does not remove the intruder. Going quiet because
    the symptom was treated is how a shifted run gets quoted as a clean one."""
    from kuiva.pt.classes import ClassResult
    from kuiva.pt.nevpt2 import _warn_on_intruder

    entry = ClassResult(name="Sr", norm=1.0, energy=-0.1, n_perturbers=3,
                        min_denominator=0.01, min_signed_denominator=0.01)
    _warn_on_intruder(2, "Sr", entry, shifted=True)
    messages = [r.message for r in kuiva_caplog.records if r.levelname == "WARNING"]
    assert any("intruder band" in m and "level shift is applied" in m for m in messages)


def test_a_class_with_no_denominators_says_nothing(kuiva_caplog):
    """A class that reports only a norm formed no denominators; there is nothing to judge."""
    from kuiva.pt.classes import ClassResult
    from kuiva.pt.nevpt2 import _warn_on_intruder

    _warn_on_intruder(0, "Sij", ClassResult(name="Sij", norm=1.0), shifted=False)
    assert not [r for r in kuiva_caplog.records if r.levelname == "WARNING"]


def test_the_shifted_family_holds_one_workspace_at_a_time():
    """⚠ The aggregate this kills: five shifted SigmaOperators, each with its own
    workspace, all live at once — measured at 4.6 GB against the largest single one's
    1.15 GB at a 20-spinor half-filled active space, for operators the classes only ever
    apply one at a time. At most one lives now; a sibling's reservation dies with its
    dropped buffer; and the build counter pins that switches happen at class boundaries,
    not in inner loops."""
    from kuiva.pt.contractions import ShiftedSpaces
    from test_ci_strings import random_spinor_integrals

    na, k = 6, 3
    h, eri = random_spinor_integrals(na, seed=2)
    family = ShiftedSpaces(na, k, h, eri)
    rng = np.random.default_rng(0)

    def vec(delta):
        m = family.get(delta).ndet
        v = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        return v / np.linalg.norm(v)

    family.get(-1).apply_h(vec(-1))
    assert family.get(-1)._sigma is not None
    family.get(0).apply_h(vec(0))
    assert family.get(-1)._sigma is None       # the sibling made room
    assert family.get(0)._sigma is not None
    family.get(-1).apply_h(vec(-1))            # and comes back on demand
    assert family.get(0)._sigma is None
    assert family.n_sigma_builds == 3          # one per switch, never per application
    family.release()
    assert all(s._sigma is None for s in family._spaces.values())
