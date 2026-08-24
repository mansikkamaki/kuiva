"""Tier-0 tests for :mod:`kuiva.props.multiplet` (Tier 0).

Analytic, no external data, milliseconds. This module carries more weight than its size
suggests: every Tier-2 comparison is expressed through its invariants, so if the invariants
were wrong the cross-code tests would compare nonsense and still pass. The tests here build
magnetic-moment matrices with a **known** g tensor from angular-momentum algebra and check
that the machinery recovers it - and, crucially, that it keeps recovering it after the
degenerate states are scrambled by a random unitary, which is exactly what different programs
(and different runs) do to them.
"""
from __future__ import annotations

import numpy as np
import pytest

from kuiva.props.multiplet import (
    G_ELECTRON, HARTREE_TO_CM, PSEUDO_DOUBLET_HINT_CM, analyse_spectrum, block_moment_tensor,
    degeneracy_pattern, degenerate_blocks, lande_g, magnetic_moment_matrices,
    multiplet_g_axes, multiplet_g_values, g_determinant_sign, axis_is_defined,
    AXIS_DEFINED_RTOL,
)

PAULI = [np.array([[0, 1], [1, 0]], dtype=complex),
         np.array([[0, -1j], [1j, 0]], dtype=complex),
         np.array([[1, 0], [0, -1]], dtype=complex)]


def angular_momentum_matrices(j: float):
    """``(J_x, J_y, J_z)`` in the ``|j, m>`` basis, m descending. Standard construction."""
    dim = int(round(2 * j + 1))
    m = np.array([j - k for k in range(dim)])
    jz = np.diag(m).astype(complex)
    jp = np.zeros((dim, dim), dtype=complex)
    for k in range(1, dim):                       # J_+ |j,m> = sqrt(j(j+1)-m(m+1)) |j,m+1>
        mm = m[k]
        jp[k - 1, k] = np.sqrt(j * (j + 1) - mm * (mm + 1))
    jm = jp.conj().T
    return (jp + jm) / 2.0, (jp - jm) / (2.0j), jz


def random_unitary(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    return q * (np.diag(r) / np.abs(np.diag(r)))


# --- blocking -----------------------------------------------------------------------------
def test_degenerate_blocks_groups_by_tolerance():
    e = [0.0, 0.0, 0.0, 2400.0, 2400.0]
    assert degenerate_blocks(e, tol_cm=1.0) == [(0, 3), (3, 2)]


def test_degenerate_blocks_splits_just_above_tolerance():
    """A gap larger than the tolerance must start a new block - the boundary behaviour that
    decides whether a real (small) crystal-field splitting is seen or absorbed."""
    assert degenerate_blocks([0.0, 0.5], tol_cm=1.0) == [(0, 2)]
    assert degenerate_blocks([0.0, 1.5], tol_cm=1.0) == [(0, 1), (1, 1)]


def test_analyse_spectrum_sorts_unsorted_input():
    """Energies may arrive in any order; blocks must still come out in energy order."""
    e_hartree = np.array([1.0, 0.0, 1.0, 0.0]) * (2400.0 / HARTREE_TO_CM)
    mults = analyse_spectrum(e_hartree)
    assert degeneracy_pattern(mults) == (2, 2)
    assert mults[0].energy_cm == pytest.approx(0.0)
    assert mults[1].energy_cm == pytest.approx(2400.0, abs=1e-6)


# --- g values from a known moment operator --------------------------------------------------
@pytest.mark.parametrize("g_expected", [2.0, 6 / 7, 4 / 3, 20.0])
def test_kramers_doublet_g_is_recovered(g_expected):
    """Build mu = -(1/2) g sigma for an isotropic doublet and recover g.

    This is the Abragam-Bleaney pseudospin convention (see the module docstring); getting the
    factor wrong by 2 is the classic error, and this pins it down.
    """
    mu = np.array([-0.5 * g_expected * p for p in PAULI])
    m = block_moment_tensor(mu, 0, 2)
    assert multiplet_g_values(m, 2) == pytest.approx([g_expected] * 3, rel=1e-12)


@pytest.mark.parametrize("j,g_j", [(2.5, 6 / 7), (3.5, 8 / 7), (7.5, 4 / 3), (0.5, 2.0)])
def test_free_ion_multiplet_g_is_recovered(j, g_j):
    """For a full ``2J+1`` multiplet with ``mu = -g_J J``, the same routine must return g_J.

    That one formula covers both the Kramers-doublet and the free-ion case is the load-bearing
    claim of :func:`multiplet_g_values`; here it is checked against explicitly constructed
    angular-momentum matrices for J from 1/2 to 15/2.
    """
    jx, jy, jz = angular_momentum_matrices(j)
    mu = np.array([-g_j * x for x in (jx, jy, jz)])
    dim = int(round(2 * j + 1))
    m = block_moment_tensor(mu, 0, dim)
    assert multiplet_g_values(m, dim) == pytest.approx([g_j] * 3, rel=1e-10)


def test_anisotropic_g_tensor_is_recovered():
    """A deliberately anisotropic doublet: principal values must come back in ascending order
    regardless of the axis ordering used to build it."""
    g = [1.5, 4.0, 12.0]
    mu = np.array([-0.5 * g[i] * PAULI[i] for i in range(3)])
    m = block_moment_tensor(mu, 0, 2)
    assert multiplet_g_values(m, 2) == pytest.approx(sorted(g), rel=1e-12)


# --- the invariance that the whole Tier-2 comparison depends on ------------------------------
@pytest.mark.parametrize("j", [0.5, 2.5, 7.5])
def test_invariants_survive_arbitrary_mixing_of_degenerate_states(j):
    """Scramble a degenerate block with a random unitary; the moment tensor must not move.

    Degenerate SOC eigenvectors are defined only up to such a mixing, so this is precisely the
    freedom meant when the phases are called arbitrary. If this test failed, no
    cross-code comparison of magnetic moments would mean anything.
    """
    jx, jy, jz = angular_momentum_matrices(j)
    mu = np.array([-1.25 * x for x in (jx, jy, jz)])
    dim = int(round(2 * j + 1))
    m_ref = block_moment_tensor(mu, 0, dim)

    u = random_unitary(dim, seed=17)
    mu_rot = np.array([u.conj().T @ x @ u for x in mu])
    m_rot = block_moment_tensor(mu_rot, 0, dim)

    assert m_rot == pytest.approx(m_ref, abs=1e-10)
    assert multiplet_g_values(m_rot, dim) == pytest.approx(
        multiplet_g_values(m_ref, dim), abs=1e-10)


def test_invariants_survive_arbitrary_phases():
    """The weaker but even more common case: each state picking up its own phase."""
    jx, jy, jz = angular_momentum_matrices(2.5)
    mu = np.array([-0.857 * x for x in (jx, jy, jz)])
    m_ref = block_moment_tensor(mu, 0, 6)
    rng = np.random.default_rng(3)
    phase = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, 6)))
    mu_ph = np.array([phase.conj().T @ x @ phase for x in mu])
    assert block_moment_tensor(mu_ph, 0, 6) == pytest.approx(m_ref, abs=1e-12)


# --- moment construction and analytic g ------------------------------------------------------
def test_magnetic_moment_matrices_combine_l_and_s():
    """mu = -(L + g_e S): a pure spin-1/2 with no orbital part must give g = g_e."""
    s = np.array([0.5 * p for p in PAULI])
    l = np.zeros_like(s)
    mu = magnetic_moment_matrices(l, s)
    g = multiplet_g_values(block_moment_tensor(mu, 0, 2), 2)
    assert g == pytest.approx([G_ELECTRON] * 3, rel=1e-12)


def test_magnetic_moment_matrices_rejects_bad_shapes():
    with pytest.raises(ValueError):
        magnetic_moment_matrices(np.zeros((3, 4, 4)), np.zeros((3, 5, 5)))


@pytest.mark.parametrize("l,s,j,expected,label", [
    (3, 0.5, 2.5, 6 / 7, "Ce(3+) 2F5/2"),
    (3, 0.5, 3.5, 8 / 7, "Yb(3+) 2F7/2"),
    (5, 2.5, 7.5, 4 / 3, "Dy(3+) 6H15/2"),
    (0, 0.5, 0.5, 2.0, "pure spin"),
])
def test_lande_g_matches_textbook_values(l, s, j, expected, label):
    assert lande_g(l, s, j) == pytest.approx(expected, rel=1e-12), label


def test_non_degenerate_level_reports_no_moment_rather_than_a_zero_one():
    """⚠ A J = 0 block has no magnetic moment, and the routine says so with an EMPTY tuple.

    It used to return ``(0.0, 0.0, 0.0)``, which reads as a *measurement* of zero. The two
    statements coincide for a genuine J = 0 level and diverge completely for the case that
    matters -- half of a tunnelling-split non-Kramers pseudo-doublet, where the pair carries a
    large axial moment and each singlet alone carries none. Printing g = 0 for a Tb/Ho-type
    ground state is a plausible wrong answer, silently. ``()`` is already the value a block
    analysed without moment matrices returns, so the reporting path needed no new case.
    """
    assert multiplet_g_values(np.zeros((3, 3)), 1) == ()
    # ...and it is still not a division by zero, whatever the tensor holds.
    assert multiplet_g_values(np.eye(3), 1) == ()


def test_analyse_spectrum_end_to_end():
    """A two-level spectrum with different g per level, as Tier 2 consumes it."""
    dim = 6
    jx, jy, jz = angular_momentum_matrices(2.5)
    mu = np.zeros((3, dim + 2, dim + 2), dtype=complex)
    for i, x in enumerate((jx, jy, jz)):
        mu[i, :dim, :dim] = -(6 / 7) * x
    for i, p in enumerate(PAULI):
        mu[i, dim:, dim:] = -0.5 * 2.0 * p
    e = np.concatenate([np.zeros(dim), np.full(2, 2400.0 / HARTREE_TO_CM)])
    mults = analyse_spectrum(e, mu=mu)
    assert degeneracy_pattern(mults) == (6, 2)
    assert mults[0].g_values == pytest.approx([6 / 7] * 3, rel=1e-10)
    assert mults[1].g_values == pytest.approx([2.0] * 3, rel=1e-10)
    assert mults[1].energy_cm == pytest.approx(2400.0, abs=1e-6)
    assert mults[0].j == 2.5 and mults[1].j == 0.5


# --- the non-Kramers pseudo-doublet -------------------------------------------------------

def tunnelling_split_doublet(g_z: float, gap_cm: float, seed: int = 0):
    """A model Tb/Ho ground state: two singlets, split by ``gap_cm``, whose PAIR carries an
    axial moment ``g_z`` and nothing transverse.

    Built the way the physics builds it. The two states are ``(|+m> +- |-m>)/sqrt(2)``, so
    ``mu_z`` is purely **off-diagonal** between them: each singlet on its own has zero moment
    in every direction, while the two together carry the whole of it. That is exactly why a
    per-state analysis of this spectrum is worthless and a per-*pair* one is not.
    """
    mu = np.zeros((3, 4, 4), dtype=complex)
    # states 0,1 = the tunnelling-split pair; 2,3 = an ordinary distant pair, for contrast.
    off = 0.5 * g_z                       # <0|mu_z|1>; Tr(mu_z^2) over the block = 2*off^2
    mu[2, 0, 1] = off
    mu[2, 1, 0] = off
    jx, jy, jz = angular_momentum_matrices(0.5)
    mu[:, 2:, 2:] = np.array([jx, jy, jz]) * 2.0
    energies = np.array([0.0, gap_cm, 5000.0, 5000.0]) / HARTREE_TO_CM
    if seed:                              # scramble the far pair, which is genuinely degenerate
        u = random_unitary(2, seed)
        mu[:, 2:, 2:] = np.einsum("ba,ibc,cd->iad", u.conj(), mu[:, 2:, 2:], u)
    return energies, mu


def test_a_split_pseudo_doublet_reports_no_g_rather_than_zero(kuiva_caplog):
    """⚠ **The defect this replaces.** Two singlets 8 cm^-1 apart never group at the default
    1 cm^-1 tolerance, so each arrived as a size-1 block and was handed back g = (0, 0, 0) --
    with no message. For a Tb/Ho-type SMM, whose entire magnetism lives in that pair, that is
    a silent wrong answer of the exact kind this project refuses to emit.
    """
    energies, mu = tunnelling_split_doublet(g_z=17.5, gap_cm=8.0)
    blocks = analyse_spectrum(energies, mu)

    assert [m.size for m in blocks] == [1, 1, 2]          # ungrouped, as before
    assert blocks[0].g_values == () and blocks[1].g_values == ()   # NOT (0.0, 0.0, 0.0)
    assert all(not m.non_kramers for m in blocks)
    assert np.isnan(blocks[0].g_z)                        # g_z is for grouped blocks only

    # ...and the reader is told why two singlets are sitting on top of each other.
    assert any("pseudo-doublet" in r.message and "pseudo_doublet_tol_cm" in r.message
               for r in kuiva_caplog.records)


def test_grouping_the_pair_recovers_g_z_and_the_gap():
    """The opt-in half: with the pair grouped, g_z is the axial moment the pair carries and
    the transverse residual is zero by symmetry (Griffith; Abragam & Bleaney)."""
    energies, mu = tunnelling_split_doublet(g_z=17.5, gap_cm=8.0)
    blocks = analyse_spectrum(energies, mu, pseudo_doublet_tol_cm=20.0)

    assert [m.size for m in blocks] == [2, 2]
    ground = blocks[0]
    assert ground.non_kramers
    assert ground.tunnelling_gap_cm == pytest.approx(8.0, rel=1e-9)
    assert ground.g_z == pytest.approx(17.5, rel=1e-10)
    assert ground.g_transverse_residual == pytest.approx(0.0, abs=1e-12)

    # The genuinely degenerate pair further up is untouched and is not marked.
    assert not blocks[1].non_kramers and blocks[1].tunnelling_gap_cm is None


def test_the_transverse_residual_is_a_check_that_can_fail():
    """⚠ A diagnostic whose only possible value is the one you assumed is not a diagnostic.

    Group two states that are *not* a pseudo-doublet -- here an ordinary isotropic pair
    artificially split -- and the residual comes back comparable to g_z, which is the signal
    that the ``pseudo_doublet_tol_cm`` request was wrong.
    """
    jx, jy, jz = angular_momentum_matrices(0.5)
    mu = np.zeros((3, 2, 2), dtype=complex)
    mu[:] = np.array([jx, jy, jz]) * 2.0                  # isotropic: g = 2 in every direction
    energies = np.array([0.0, 8.0]) / HARTREE_TO_CM

    grouped = analyse_spectrum(energies, mu, pseudo_doublet_tol_cm=20.0)[0]
    assert grouped.non_kramers
    assert grouped.g_transverse_residual == pytest.approx(grouped.g_z, rel=1e-9)


def test_three_close_singlets_are_not_made_into_one_and_a_half_doublets():
    """A greedy half-open pass would pair 0-1 and then 1-2. A singlet consumed by one pair
    cannot join another, and three near-equal singlets are not a doublet in any case."""
    energies = np.array([0.0, 2.0, 4.0, 5000.0]) / HARTREE_TO_CM
    blocks = analyse_spectrum(energies, pseudo_doublet_tol_cm=10.0)
    assert [m.size for m in blocks] == [2, 1, 1]
    assert [m.non_kramers for m in blocks] == [True, False, False]
    assert sum(m.size for m in blocks) == 4               # every state accounted for, once


def test_distant_singlets_neither_group_nor_warn(kuiva_caplog):
    """The advisory hint is bounded, or every even-electron spectrum would carry it."""
    far = 5.0 * PSEUDO_DOUBLET_HINT_CM
    energies = np.array([0.0, far, 2 * far]) / HARTREE_TO_CM
    blocks = analyse_spectrum(energies)
    assert [m.size for m in blocks] == [1, 1, 1]
    assert not any("pseudo-doublet" in r.message for r in kuiva_caplog.records)


# --- principal axes and the sign M cannot carry --------------------------------------------

def doublet_with_g(g_diag, seed=0):
    """A Kramers doublet built from a KNOWN g tensor: mu_i = -(1/2) sum_k g_ik sigma_k.

    Optionally scrambled by a random unitary inside the block, which is what a different
    program (or a different run) does to these states and what every invariant here must
    survive.
    """
    g = np.diag(np.asarray(g_diag, dtype=float))
    mu = np.stack([-0.5 * sum(g[i, k] * PAULI[k] for k in range(3)) for i in range(3)])
    if seed:
        u = random_unitary(2, seed)
        mu = np.stack([u.conj().T @ m @ u for m in mu])
    return g, mu


def test_the_principal_axes_are_recovered_and_are_a_proper_rotation():
    """The axes were computed inside ``multiplet_g_values`` and thrown away; they are the
    quantity a crystal-field analysis actually wants."""
    # An axial doublet whose easy axis is deliberately NOT a Cartesian direction.
    theta = 0.7
    rot = np.array([[np.cos(theta), 0.0, np.sin(theta)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(theta), 0.0, np.cos(theta)]])
    g = rot @ np.diag([1.0, 2.0, 18.0]) @ rot.T
    mu = np.stack([-0.5 * sum(g[i, k] * PAULI[k] for k in range(3)) for i in range(3)])

    block = analyse_spectrum([0.0, 0.0], mu)[0]
    assert block.g_values == pytest.approx((1.0, 2.0, 18.0), rel=1e-10)
    assert block.easy_axis_is_defined()
    # The easy axis is a LINE, so compare up to sign.
    assert abs(float(np.dot(block.easy_axis, rot[:, 2]))) == pytest.approx(1.0, abs=1e-10)
    assert np.linalg.det(block.g_axes) == pytest.approx(1.0, abs=1e-12)   # proper rotation
    assert block.axiality == pytest.approx(18.0 / 1.5, rel=1e-10)


def test_an_easy_axis_is_refused_where_two_g_values_coincide():
    """⚠ The check that stops an arbitrary vector being quoted as an easy axis. An easy-plane
    or isotropic block has no easy *direction*: the eigensolver returns some pair spanning the
    degenerate plane, self-consistently and meaninglessly."""
    _, isotropic = doublet_with_g([2.0, 2.0, 2.0])
    _, easy_plane = doublet_with_g([9.0, 9.0, 0.5])
    _, axial = doublet_with_g([0.5, 0.6, 9.0])

    assert not analyse_spectrum([0.0, 0.0], isotropic)[0].easy_axis_is_defined()
    assert not analyse_spectrum([0.0, 0.0], easy_plane)[0].easy_axis_is_defined()
    assert analyse_spectrum([0.0, 0.0], axial)[0].easy_axis_is_defined()


def test_the_axis_tolerance_is_loose_enough_for_a_real_isotropic_doublet():
    """⚠ **Why the threshold is 1% and not machine epsilon.**

    A free ion's ``j = 1/2`` doublet is isotropic *by symmetry*, and a real calculation still
    returns it anisotropic in the last few digits — measured on example 2's boron 2p1:
    ``g = (0.6656, 0.6656, 0.6665)``, a 1.3e-3 relative spread from basis and picture-change
    anisotropy rather than from physics. At a tight tolerance that block would be reported as
    having an easy axis, and the direction printed beside it would be an artefact of those
    digits — next to an axiality of 1.00, which says the opposite.
    """
    boron_like = (0.66557, 0.66557, 0.66646)             # example 2, to the printed precision
    assert not axis_is_defined(boron_like)

    # ...while a genuinely axial SMM doublet, and even a merely rhombic one, keep theirs.
    assert axis_is_defined((0.01, 0.02, 19.0))           # Ising-like
    assert axis_is_defined((1.5, 2.0, 2.5))              # rhombic, 20% separation
    assert not axis_is_defined((0.5, 9.0, 9.0))          # easy plane: no easy *axis*

    # The boundary is where it says it is, and it is a *relative* one.
    assert axis_is_defined((1.0, 1.0, 1.0 + 2.0 * AXIS_DEFINED_RTOL))
    assert not axis_is_defined((1.0, 1.0, 1.0 + 0.5 * AXIS_DEFINED_RTOL))


@pytest.mark.parametrize("g_diag, sign", [([2.0, 2.0, 2.0], +1), ([-2.0, 2.0, 2.0], -1),
                                          ([1.0, 3.0, 18.0], +1), ([1.0, 3.0, -18.0], -1)])
def test_the_sign_of_det_g_is_recovered_from_a_third_order_invariant(g_diag, sign):
    """⚠ ``M = Tr(mu_i mu_j)`` is **quadratic** in mu, so it fixes |det g| and loses the sign.

    The sign is a property of the states rather than a convention, and it comes back from
    ``det(g) = 2i Tr(mu_x [mu_y, mu_z]) / mu_B^3`` — a trace of block-restricted operators,
    hence invariant under any mixing inside the block, exactly like M itself.
    """
    g, mu = doublet_with_g(g_diag)
    assert np.sign(np.linalg.det(g)) == sign
    assert g_determinant_sign(mu, 0, 2) == pytest.approx(float(sign))


def test_the_sign_survives_the_scrambling_that_makes_it_worth_having():
    """The point of an invariant: it is the same after the arbitrary unitary mixing that a
    different program's eigensolver applies to a degenerate block. The |g| values agree too —
    which is what shows M genuinely cannot see the sign, rather than merely not reporting it."""
    plain = analyse_spectrum([0.0, 0.0], doublet_with_g([1.0, 3.0, -18.0])[1])[0]
    for seed in (2, 3, 4):
        mixed = analyse_spectrum([0.0, 0.0], doublet_with_g([1.0, 3.0, -18.0], seed)[1])[0]
        assert mixed.g_values == pytest.approx(plain.g_values, rel=1e-10)
        assert mixed.g_sign == plain.g_sign == -1.0
    # ...and the positive-determinant tensor with the SAME |g| is indistinguishable in M.
    positive = analyse_spectrum([0.0, 0.0], doublet_with_g([1.0, 3.0, 18.0])[1])[0]
    assert positive.g_values == pytest.approx(plain.g_values, rel=1e-10)
    assert np.allclose(positive.m_tensor, plain.m_tensor)          # M cannot tell them apart
    assert positive.g_sign == +1.0 and plain.g_sign == -1.0        # the third-order one can


def test_the_sign_is_not_reported_where_it_is_not_defined():
    """⚠ Never a guess. Only a two-dimensional block has a ``det(g)``; for a larger multiplet
    the three-fold product is not the determinant of anything."""
    j = 2.5
    jx, jy, jz = angular_momentum_matrices(j)
    mu = -np.stack([jx, jy, jz]) * (6.0 / 7.0)
    quartet = analyse_spectrum([0.0] * int(2 * j + 1), mu)[0]
    assert quartet.size == 6 and quartet.g_values          # g values, yes
    assert quartet.g_sign == 0.0                           # a sign, no
    assert g_determinant_sign(mu, 0, 6) == 0.0
