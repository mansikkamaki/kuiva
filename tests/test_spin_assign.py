"""Tier 0/1: ``<S^2>`` over CI states, and the term assignment built on it.

What these tests are designed to fail on
----------------------------------------
``<S^2>`` is computed as ``sum_k || S_k |I> ||^2`` over the *CAS determinant space*, plus an
explicit correction for the part of ``S_k|I>`` that leaves it (``kuiva.props.spin``). Both
halves fail quietly if they are wrong:

* the in-space half gives the right answer for every spin-separable model, so a model test
  alone would pass while the correction was missing entirely;
* the correction is a sum of four second-quantized families with signs and 1-RDM
  contractions in it, and getting one wrong shifts ``<S^2>`` by an amount that still looks
  like a plausible spin.

So the load-bearing test here (:func:`test_the_out_of_cas_correction_matches_a_full_space_ci`)
compares against a **brute-force full-space** construction: an explicit determinant-basis
matrix of the one-body ``S_k`` over *every* orbital, squared and taken as an expectation
value, with the CAS state embedded in that space. It shares no code with the implementation
and it fails by 1.17 out of 1.26 if the correction is dropped, which is measured in the test
rather than asserted from theory.

The assignment layer is tested on what it must **refuse**: a block whose evidence does not
add up gets ``"?"``. A term labeller that always produces a label is the failure mode this
project cares about, so the negative cases outnumber the positive ones.

Tolerances: the model and brute-force comparisons are exact arithmetic on small matrices, so
1e-12 (they reproduce to 1e-16). The atom tests assert ``<S^2>`` to 1e-8 of an exact
half-integer value, which is a physical requirement rather than a numerical one — spin is a
good quantum number with spin-orbit coupling off, and a value that is not integral there
means something is wrong with the calculation.
"""
import importlib
import itertools

import numpy as np
import pytest

import kuiva
from kuiva.props.assign import (LANDE_FIT_TOL, Assignment, TermAssignment, assign_terms,
                                term_letter)
from kuiva.props.multiplet import Multiplet
from kuiva.props.spin import SpinAnalysis, spin_analysis, spin_from_s_squared, spin_squared_states

casci_mod = importlib.import_module("kuiva.mcscf.casci")

EXACT = 1e-12
PHYSICAL = 1e-8                      # a spin-free <S^2> is integral or the run is wrong


# --- the machinery, on models with known answers --------------------------------------------

def spin_matrices(n_spinor: int) -> np.ndarray:
    """``S_x, S_y, S_z`` over ``n_spinor`` spinors in the interleaved Kramers convention.

    Written out here rather than imported so the test does not check the implementation
    against itself: this is the textbook ``sigma/2`` on each Kramers pair.
    """
    sx = np.zeros((n_spinor, n_spinor), complex)
    sy = np.zeros_like(sx)
    sz = np.zeros_like(sx)
    for p in range(n_spinor // 2):
        u, b = 2 * p, 2 * p + 1
        sx[u, b] = sx[b, u] = 0.5
        sy[u, b] = -0.5j
        sy[b, u] = 0.5j
        sz[u, u], sz[b, b] = 0.5, -0.5
    return np.stack([sx, sy, sz])


def spin_free_integrals(h_spatial: np.ndarray, g_spatial: np.ndarray):
    """Lift a spin-free ``(h, g)`` over spatial orbitals into the spinor convention."""
    n = 2 * h_spatial.shape[0]
    h = np.zeros((n, n), complex)
    eri = np.zeros((n,) * 4, complex)
    for p, q in itertools.product(range(h_spatial.shape[0]), repeat=2):
        for s in range(2):
            h[2 * p + s, 2 * q + s] = h_spatial[p, q]
    for p, q, r, s in itertools.product(range(h_spatial.shape[0]), repeat=4):
        for a, b in itertools.product(range(2), repeat=2):
            eri[2 * p + a, 2 * q + a, 2 * r + b, 2 * s + b] = g_spatial[p, q, r, s]
    return h, eri


def symmetrized_eri(rng, n: int) -> np.ndarray:
    g = rng.normal(size=(n,) * 4)
    g = 0.5 * (g + g.transpose(1, 0, 2, 3))
    g = 0.5 * (g + g.transpose(2, 3, 0, 1))
    return 0.5 * (g + g.transpose(1, 0, 3, 2))


def test_two_electrons_in_two_orbitals_give_one_triplet_and_three_singlets():
    """The smallest complete check: ``<S^2>`` is 2 on the triplet and 0 on every singlet.

    A spin-free Hamiltonian over two spatial orbitals has exactly this spectrum whatever the
    integrals are, so the assertion is on the *structure* rather than on a fitted number.
    """
    rng = np.random.default_rng(3)
    h_sp = rng.normal(size=(2, 2))
    h_sp = 0.5 * (h_sp + h_sp.T)
    h, eri = spin_free_integrals(h_sp, symmetrized_eri(rng, 2))
    solver = casci_mod.FullCISolver(4, 2, n_states=6, enforce_kramers=False)
    result = solver.solve_active(h, eri)
    s2, leak = spin_squared_states(solver, spin_matrices(4), vectors=result.vectors)
    assert np.allclose(np.sort(np.round(s2, 10)), [0.0, 0.0, 0.0, 2.0, 2.0, 2.0], atol=EXACT)
    assert np.max(np.abs(leak)) == 0.0            # no leak blocks given: identically zero


def test_a_single_electron_gives_three_quarters():
    """``S(S+1) = 3/4`` for one electron, whatever the orbitals — including the ``t_k`` path."""
    solver = casci_mod.FullCISolver(2, 1, n_states=2, enforce_kramers=False)
    result = solver.solve_active(np.diag([-1.0, -1.0]).astype(complex),
                                 np.zeros((2,) * 4, complex))
    s2, _ = spin_squared_states(solver, spin_matrices(2), vectors=result.vectors)
    assert np.allclose(s2, 0.75, atol=EXACT)


def test_a_non_hermitian_operator_is_refused_because_the_square_identity_needs_one():
    """⚠ ``<A^2> = ||A|I>||^2`` holds only for Hermitian ``A``; a caller cannot be trusted to
    know that, so the boundary asserts it."""
    solver = casci_mod.FullCISolver(2, 1, n_states=1, enforce_kramers=False)
    solver.solve_active(np.diag([-1.0, -1.0]).astype(complex), np.zeros((2,) * 4, complex))
    bad = np.zeros((1, 2, 2), complex)
    bad[0, 0, 1] = 1.0                                  # a^dag_0 a_1 alone: not Hermitian
    with pytest.raises(ValueError, match="Hermitian"):
        solver.one_body_moments(bad)


def test_one_body_moments_reproduce_the_transition_density_diagonal():
    """The 1-RDMs this returns must be the ones the excitation map already produces.

    Two consumers of one intermediate that disagree is the failure this rules out; the
    transition densities are the independently validated route.
    """
    rng = np.random.default_rng(11)
    h_sp = rng.normal(size=(3, 3))
    h_sp = 0.5 * (h_sp + h_sp.T)
    h, eri = spin_free_integrals(h_sp, symmetrized_eri(rng, 3))
    solver = casci_mod.FullCISolver(6, 3, n_states=4, enforce_kramers=False)
    result = solver.solve_active(h, eri)
    _, _, rdm1 = solver.one_body_moments(spin_matrices(6), result.vectors)
    tdm = solver.transition_densities(result.vectors)
    for i in range(4):
        assert np.abs(rdm1[i] - tdm[i, i]).max() < EXACT


# --- the out-of-CAS correction, against a full-space construction ---------------------------

def determinants(n_orb: int, n_elec: int):
    return [tuple(c) for c in itertools.combinations(range(n_orb), n_elec)]


def one_body_determinant_matrix(op: np.ndarray, n_orb: int, n_elec: int) -> np.ndarray:
    """``<K| sum_pq op_pq a^dag_p a_q |J>`` by explicit second quantization.

    Deliberately naive and written from the anticommutation rules, so it shares nothing with
    ``ci/strings.py`` beyond the ascending-index determinant ordering (which the agreement in
    :func:`test_the_out_of_cas_correction_matches_a_full_space_ci` also confirms).
    """
    ds = determinants(n_orb, n_elec)
    index = {d: i for i, d in enumerate(ds)}
    m = np.zeros((len(ds), len(ds)), complex)
    for j, d in enumerate(ds):
        for q in d:
            sign_q = (-1) ** d.index(q)
            rest = tuple(x for x in d if x != q)
            for p in range(n_orb):
                if p in rest:
                    continue
                new = tuple(sorted(rest + (p,)))
                m[index[new], j] += (-1) ** new.index(p) * sign_q * op[p, q]
    return m


def test_the_out_of_cas_correction_matches_a_full_space_ci():
    """⚠ The load-bearing test: the correction is exact, and dropping it is not small.

    A random unitary is applied to the spin operator so that the orbital spaces are **not**
    spin-separable — the case a converged spin-orbit CASSCF is genuinely in. ``<S^2>`` of a
    CAS state is then compared against the same expectation value taken in the full
    six-spinor determinant space, where no separation is assumed at all.
    """
    rng = np.random.default_rng(7)
    n_orb, n_elec = 6, 4
    inactive, active, virtual = [0, 1], [2, 3], [4, 5]

    a = rng.normal(size=(n_orb, n_orb)) + 1j * rng.normal(size=(n_orb, n_orb))
    q, _ = np.linalg.qr(a)
    s_mo = np.stack([q.conj().T @ sk @ q for sk in spin_matrices(n_orb)])

    h = rng.normal(size=(2, 2))
    h = 0.5 * (h + h.T)
    solver = casci_mod.FullCISolver(2, 2, n_states=1, enforce_kramers=False)
    ci = solver.solve_active(h.astype(complex), symmetrized_eri(rng, 2).astype(complex))

    s_act = np.stack([sk[np.ix_(active, active)] for sk in s_mo])
    trace = np.array([np.real(np.trace(sk[np.ix_(inactive, inactive)])) for sk in s_mo])
    blocks = (np.stack([sk[np.ix_(active, inactive)] for sk in s_mo]),
              np.stack([sk[np.ix_(virtual, active)] for sk in s_mo]),
              np.stack([sk[np.ix_(virtual, inactive)] for sk in s_mo]))
    corrected, leak = spin_squared_states(solver, s_act, inactive_trace=trace,
                                          leak_blocks=blocks, vectors=ci.vectors)
    bare, _ = spin_squared_states(solver, s_act, inactive_trace=trace, vectors=ci.vectors)

    full = determinants(n_orb, n_elec)
    index = {d: i for i, d in enumerate(full)}
    psi = np.zeros(len(full), complex)
    for j, ad in enumerate(determinants(2, 2)):
        psi[index[tuple(sorted(tuple(inactive) + tuple(active[x] for x in ad)))]] = ci.vectors[0, j]
    psi /= np.linalg.norm(psi)
    exact = sum(float(np.real(np.vdot(psi, m @ m @ psi)))
                for m in (one_body_determinant_matrix(s_mo[k], n_orb, n_elec)
                          for k in range(3)))

    assert abs(corrected[0] - exact) < EXACT
    assert abs(corrected[0] - bare[0] - leak[0]) < EXACT
    # ⚠ measured, not assumed: without the correction this test is wrong by most of the value.
    assert abs(bare[0] - exact) > 1.0


def test_a_spin_separable_set_has_an_identically_zero_correction():
    """The normal case: whole Kramers pairs per space, so ``S`` never leaves the CAS space.

    The counterpart of the test above — the correction must not *add* anything where the
    physics says it is zero, or every ordinary spectrum would be shifted.
    """
    n_orb, active = 6, [2, 3]
    s_mo = spin_matrices(n_orb)
    inactive, virtual = [0, 1], [4, 5]
    for rows, cols in ((active, inactive), (virtual, active), (virtual, inactive)):
        assert np.abs(s_mo[:, rows][:, :, cols]).max() == 0.0


# --- the atoms, end to end ------------------------------------------------------------------

@pytest.fixture(scope="module")
def oxygen_spin_free():
    """O ``2p^4`` with spin-orbit coupling *off*: the textbook ``3P / 1D / 1S``.

    ``screening="none"`` and ``with_soc=False``: nothing here is a statement about spin-orbit
    coupling, and the suite may not depend on a warm AMF cache.
    """
    mol = kuiva.Molecule([("O", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=2)
    scf = kuiva.ScalarSCF(mol, memory_gb=8.0, screening="none", with_soc=False).run()
    ref = kuiva.Reference(scf).run()
    return kuiva.CASSCF(ref, character=("O", "p"), n_active=6, n_states=15,
                        report=False).run()


def test_the_spin_free_oxygen_terms_have_integral_spin(oxygen_spin_free):
    """⚠ Asserted as a **physical** requirement: with spin-orbit coupling off, ``S`` is a good
    quantum number, so ``<S^2>`` is ``S(S+1)`` exactly and not to whatever the code produces.
    """
    spin = oxygen_spin_free.spin_analysis()
    assert not spin.has_soc
    assert [n for _, n in spin.blocks] == [9, 5, 1]
    assert np.allclose(spin.block_s_squared, [2.0, 0.0, 0.0], atol=PHYSICAL)
    assert np.max(spin.purity()) < PHYSICAL
    # the orbital spaces are spin-separable here, so the out-of-space term is exactly zero
    assert np.max(np.abs(spin.block_leak)) < 1e-10


def test_oxygen_is_assigned_3p_1d_1s(oxygen_spin_free):
    """The exit criterion: dimension and ``<S^2>`` together fix ``L``, with no g involved."""
    assignment = oxygen_spin_free.assign(report=False)
    assert assignment.labels() == ("^3P", "^1D", "^1S")
    assert all(t.residual == 0.0 for t in assignment.terms)
    assert all("2L+1" in " ".join(t.evidence) for t in assignment.terms)


@pytest.fixture(scope="module")
def boron_soc():
    """B ``2p^1`` with spin-orbit coupling on: ``2P_1/2`` below ``2P_3/2``.

    The cheapest system with an analytic two-component target — the Landé factors 2/3 and
    4/3 are fixed by theory and by no program's conventions.
    """
    mol = kuiva.Molecule([("B", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=1)
    scf = kuiva.ScalarSCF(mol, memory_gb=8.0, screening="none").run()
    ref = kuiva.Reference(scf).run()
    return kuiva.CASSCF(ref, character=("B", "p"), n_active=6, n_states=6,
                        report=False).run()


def test_boron_levels_are_assigned_from_the_inverted_lande_factor(boron_soc):
    """``J`` from the dimension, ``S`` from ``<S^2>``, ``L`` from ``g`` — all three needed."""
    assignment = boron_soc.assign(report=False)
    assert assignment.has_soc
    assert assignment.labels() == ("^2P_1/2", "^2P_3/2")
    assert [t.size for t in assignment.terms] == [2, 4]
    assert [t.j for t in assignment.terms] == [0.5, 1.5]
    # The residual is the bare-operator picture-change error, ~1e-3 relative; it is the
    # quantity that would grow if the Lande inversion were wrong, so it is asserted on.
    assert all(t.residual < LANDE_FIT_TOL for t in assignment.terms)
    assert all(t.orbital == 1 for t in assignment.terms)


def test_the_boron_spin_stays_pure_and_the_leak_stays_small(boron_soc):
    """A one-electron system cannot mix spins, so ``<S^2>`` is 3/4 even with SOC on."""
    spin = boron_soc.spin_analysis()
    assert spin.has_soc
    assert np.allclose(spin.block_s_squared, 0.75, atol=1e-6)
    # ⚠ The orbitals of a converged general-complex CASSCF are NOT exactly spin-separable,
    # so this is nonzero -- and it must stay far below the 0.01 that would move a label.
    assert 0.0 < spin.leakage < 1e-2
    assert np.max(np.abs(spin.block_leak)) < 1e-3


def test_the_blocking_follows_the_spectrum_it_is_given(boron_soc):
    """⚠ ``energies=`` exists so that a NEVPT2-corrected file blocks **one** spectrum.

    There the stored ``H`` is the perturbed spectrum while the states are still the CASSCF
    ones; without this the ``<S^2>`` blocking and ``analyse_spectrum``'s would group
    different levels and could not be paired at all. Tested by handing it a spectrum that
    splits a block: the grouping must follow, and the per-state values must not move, since
    they are properties of the states and not of the energies.
    """
    from kuiva.interface.api import spin_analysis as api_spin
    reference = boron_soc.reference_stage.reference
    plain = api_spin(reference, boron_soc.outcome)
    assert [n for _, n in plain.blocks] == [2, 4]

    shifted = np.array(boron_soc.outcome.ci.total_energies, dtype=float)
    shifted[4:] += 1.0e-3                                    # ~220 cm^-1, splits the quartet
    split = api_spin(reference, boron_soc.outcome, energies=shifted)
    assert [n for _, n in split.blocks] == [2, 2, 2]
    assert np.allclose(np.sort(split.state_s_squared),
                       np.sort(plain.state_s_squared), atol=1e-12)


def test_spin_analysis_refuses_a_solver_without_one_body_moments():
    """A solver providing no ``one_body_moments`` must say so, not guess.

    Both real solvers provide it now — the CI through the excitation map, the network
    through per-root densities — so the refusal guards whatever third solver comes next.
    """
    class Stub:
        pass

    with pytest.raises(NotImplementedError, match="one_body_moments"):
        spin_squared_states(Stub(), spin_matrices(2))


# --- the assignment layer's refusals --------------------------------------------------------

def multiplet(start, size, energy=0.0, g=()):
    return Multiplet(start=start, size=size, energy_cm=energy, spread_cm=0.0, g_values=g)


def spin_of(blocks, values, has_soc=True):
    return SpinAnalysis(blocks=tuple(blocks), block_s_squared=np.asarray(values, dtype=float),
                        state_s_squared=np.zeros(sum(n for _, n in blocks)),
                        energies_cm=np.zeros(len(blocks)), has_soc=has_soc)


def test_a_strongly_spin_mixed_block_gets_no_label():
    """⚠ The behaviour that matters: no term is offered where the evidence does not support
    one. A labeller that always produces a label is the failure this project refuses."""
    m = [multiplet(0, 4, g=(1.0, 1.0, 1.0))]
    result = assign_terms(m, spin_of([(0, 4)], [1.4]))
    assert result.labels() == ("?",)
    assert "strongly spin-mixed" in " ".join(result.terms[0].evidence)


def test_a_crystal_field_doublet_is_not_forced_into_a_free_ion_level():
    """A ``g = 8`` Kramers doublet of a complex is not a ``2J+1`` manifold; the inverted ``L``
    is nowhere near an integer and the label is withheld."""
    m = [multiplet(0, 2, g=(0.01, 0.02, 8.0))]
    result = assign_terms(m, spin_of([(0, 2)], [0.75]))
    assert result.labels() == ("?",)
    assert any("L" in line for line in result.terms[0].evidence)


def test_a_block_failing_the_triangle_condition_is_refused():
    """``|L-S| <= J <= L+S`` is checked, so an inverted ``L`` that is integral but impossible
    still gets no label."""
    # S = 1/2, J = 1/2; a g that inverts to L = 3 violates |L-S| <= J.
    from kuiva.props.multiplet import lande_g
    g = lande_g(3.0, 0.5, 0.5)
    result = assign_terms([multiplet(0, 2, g=(g, g, g))], spin_of([(0, 2)], [0.75]))
    assert result.labels() == ("?",)
    assert "triangle condition" in " ".join(result.terms[0].evidence)


def test_a_non_kramers_pseudo_doublet_is_not_offered_a_free_ion_level():
    """⚠ Its ``g_z`` is a Griffith / Abragam-Bleaney effective-spin quantity with the
    transverse components zero **by convention**, so inverting the Landé formula on its
    isotropic average would produce a term out of a convention rather than out of the states.
    """
    m = [Multiplet(start=0, size=2, energy_cm=0.0, spread_cm=0.0, g_values=(0.0, 0.0, 17.5),
                   non_kramers=True, tunnelling_gap_cm=0.004)]
    result = assign_terms(m, spin_of([(0, 2)], [6.0]))
    assert result.labels() == ("?",)
    assert "non-Kramers pseudo-doublet" in " ".join(result.terms[0].evidence)


def test_a_j_zero_block_says_why_it_cannot_be_assigned():
    """A singlet carries no moment, so ``L`` cannot be inverted from a g that does not exist."""
    result = assign_terms([multiplet(0, 1)], spin_of([(0, 1)], [0.0]))
    assert result.labels() == ("?",)
    assert "no magnetic moment" in " ".join(result.terms[0].evidence)


def test_mismatched_blockings_are_refused_rather_than_zipped():
    """The two analyses must be the same grouping of the same spectrum; silently pairing block
    1 of one with block 1 of the other would attach every label to the wrong states."""
    with pytest.raises(ValueError, match="blocking"):
        assign_terms([multiplet(0, 2), multiplet(2, 4)], spin_of([(0, 6)], [0.75]))


def test_term_letters_skip_j_and_refuse_beyond_the_table():
    assert [term_letter(i) for i in range(5)] == ["S", "P", "D", "F", "G"]
    assert "J" not in "".join(term_letter(i) for i in range(16))
    with pytest.raises(ValueError, match="term letter"):
        term_letter(99)


def test_spin_from_s_squared_clamps_rather_than_returning_nan():
    """⚠ A roundoff-negative ``<S^2>`` around a singlet must read as ``S = 0``; a ``nan``
    propagating into a multiplicity would print as "could not be measured"."""
    assert spin_from_s_squared(-1e-14) == 0.0
    assert spin_from_s_squared(0.75) == pytest.approx(0.5)
    assert spin_from_s_squared(2.0) == pytest.approx(1.0)


def test_an_irrep_label_is_a_property_of_the_block_not_of_a_state():
    """Members disagreeing means the block has no single label — reported as none, not as the
    first member's (:mod:`kuiva.symm.classify` makes the same call for the same reason)."""
    m = [multiplet(0, 2, g=(0.666, 0.666, 0.666)), multiplet(2, 2, g=(0.666,) * 3)]
    spin = spin_of([(0, 2), (2, 2)], [0.75, 0.75])
    result = assign_terms(m, spin, irreps=["E1g", "E1g", "E1g", "E1u"])
    assert result.terms[0].irrep == "E1g"
    assert result.terms[1].irrep == ""


def test_the_report_never_claims_more_than_it_measured(kuiva_caplog):
    """The report must say the labels are inferences, in the output stream, every time."""
    import logging
    result = assign_terms([multiplet(0, 9)], spin_of([(0, 9)], [2.0], has_soc=False))
    with kuiva_caplog.at_level(logging.INFO):
        result.report()
    text = kuiva_caplog.text
    assert "INFERENCE" in text and "^3P" in text
