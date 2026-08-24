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
    multiplet_g_values,
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
