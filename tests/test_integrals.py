"""Tier-0/1 tests for the integral factorization and the AO -> spinor-MO transform.

Two kinds of test, and the distinction matters:

* **synthetic** — a random positive-semidefinite "ERI" tensor with the right permutational
  symmetry. No PySCF, instant, and it exercises the Cholesky decomposition and the transform
  against a brute-force ``einsum`` reference in full generality (complex coefficients, mixed
  orbital blocks). This is where correctness of the *algebra* is established.
* **real integrals** — a small molecule through the front-end. This is where correctness of
  the *conventions* is established: the 8-fold packing order of PySCF's ``aosym="s8"`` array,
  the packing of its DF factors, and the agreement of the DF and Cholesky routes. A wrong
  index formula passes every synthetic test and fails these.
"""
import numpy as np
import pytest

from kuiva.integrals.transform import (DEFAULT_CHOLESKY_TOL, SpinorMOIntegrals, ThreeIndexAO,
                                       assemble_4c, check_permutational_symmetry, npair_of,
                                       pivoted_cholesky, shell_pair_orbits, transform_1e,
                                       transform_3c)
from kuiva.interface import Molecule
from kuiva.interface.pyscf_bridge import run_scalar_x2c
from kuiva.orth.canonical import canonical_orthogonalization
from kuiva.spinor.expand import expand_scalar_mos, spin_block_diagonal, time_reverse

CD_TOL = 1e-8              # tight decomposition for the tests: we compare against exact
ALG_TOL = 1e-10            # algebraic identities hold to rounding


# --- helpers ------------------------------------------------------------------------------
def synthetic_eri(nao, nvec=None, seed=0, decay=None):
    """A random ERI-like 4-index tensor with full 8-fold symmetry and positive semidefinite
    pair matrix — i.e. everything the decomposition is allowed to assume, and nothing else.

    ``decay`` scales the generating vectors geometrically, giving the pair matrix a decaying
    spectrum like a real ERI matrix. Without it the matrix is full rank and well conditioned,
    and the Cholesky decomposition never truncates.
    """
    npair = npair_of(nao)
    nvec = nvec or npair
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((npair, nvec))
    if decay:
        a = a * decay ** np.arange(nvec)
    m = a @ a.T                                        # (npair, npair), SPD, symmetric
    i, j = np.tril_indices(nao)
    eri = np.zeros((nao,) * 4)
    eri[i[:, None], j[:, None], i[None, :], j[None, :]] = m
    eri[j[:, None], i[:, None], i[None, :], j[None, :]] = m
    eri[i[:, None], j[:, None], j[None, :], i[None, :]] = m
    eri[j[:, None], i[:, None], j[None, :], i[None, :]] = m
    return eri


def pack_s8(eri, nao):
    """Pack a full 4-index array the way PySCF's ``aosym="s8"`` does (our reference for the
    index formula in :func:`kuiva.integrals.transform._s8_column`)."""
    i, j = np.tril_indices(nao)
    m = eri[i[:, None], j[:, None], i[None, :], j[None, :]]
    ij = np.tril_indices(npair_of(nao))
    return m[ij]


def brute_force_spinor_eri(eri_ao, c):
    """Reference transform: explicit spin sum, no factorization, no cleverness.

    ``(pq|rs) = sum_{sigma,tau} C^{sigma*}_{mu p} C^{sigma}_{nu q} (mu nu|la ka)
                C^{tau*}_{la r} C^{tau}_{ka s}``
    """
    nao = eri_ao.shape[0]
    ca, cb = c[:nao], c[nao:]
    t = (np.einsum("mnkl,kr,ls->mnrs", eri_ao, ca.conj(), ca, optimize=True) +
         np.einsum("mnkl,kr,ls->mnrs", eri_ao, cb.conj(), cb, optimize=True))
    return (np.einsum("mnrs,mp,nq->pqrs", t, ca.conj(), ca, optimize=True) +
            np.einsum("mnrs,mp,nq->pqrs", t, cb.conj(), cb, optimize=True))


def random_spinors(nao, nspinor, seed=0, complex_=True):
    """A random orthonormal spinor set in an orthonormal scalar basis (as after a CASSCF
    rotation: dense, complex, no Kramers structure)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((2 * nao, 2 * nao))
    if complex_:
        a = a + 1j * rng.standard_normal((2 * nao, 2 * nao))
    q, _ = np.linalg.qr(a)
    return np.ascontiguousarray(q[:, :nspinor])


# --- Cholesky decomposition ----------------------------------------------------------------
def test_pivoted_cholesky_reconstructs_a_psd_matrix():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((40, 12))
    m = a @ a.T                                        # rank 12 of 40: must stop at 12 vectors
    lvec, piv, resid = pivoted_cholesky(np.diag(m).copy(), lambda q: m[:, q], 1e-10)
    assert lvec.shape[0] == 12                         # exact rank detection
    assert np.max(np.abs(lvec.T @ lvec - m)) < 1e-8
    assert resid < 1e-10
    assert len(set(piv.tolist())) == len(piv)          # no pivot chosen twice


@pytest.mark.parametrize("tol", [1e-2, 1e-4, 1e-6])
def test_cholesky_error_is_bounded_by_the_threshold(tol):
    """The property that makes truncation safe: the residual bounds every matrix element.

    ``|M_ij - (L^T L)_ij| <= sqrt(d_i d_j) <= tol`` (Koch, Sanchez de Meras & Pedersen 2003).
    This is what lets the threshold be read as an error in Eh on any two-electron integral,
    which is the claim the module docstring makes to the user.
    """
    nao = 6
    eri = synthetic_eri(nao, seed=2, decay=0.75)
    f = ThreeIndexAO.from_eri(eri, nao, tol=tol, report=False)
    approx = assemble_4c(f.unpack(slice(None)).reshape(f.naux, nao, nao))
    assert np.max(np.abs(approx - eri)) <= tol
    assert f.naux < npair_of(nao)                      # truncation actually happened
    assert f.residual <= tol


@pytest.mark.parametrize("form", ["s8", "s4", "s1"])
def test_all_eri_input_forms_agree(form):
    nao = 5
    eri = synthetic_eri(nao, seed=3)
    if form == "s1":
        arg = eri
    elif form == "s8":
        arg = pack_s8(eri, nao)
    else:
        i, j = np.tril_indices(nao)
        arg = eri[i[:, None], j[:, None], i[None, :], j[None, :]]
    f = ThreeIndexAO.from_eri(arg, nao, tol=CD_TOL, report=False)
    got = assemble_4c(f.unpack(slice(None)).reshape(f.naux, nao, nao))
    assert np.max(np.abs(got - eri)) < 1e-7


def test_indefinite_matrix_is_reported(kuiva_caplog):
    m = np.diag([1.0, -1.0, 0.5])
    with kuiva_caplog.at_level("ERROR"):
        pivoted_cholesky(np.diag(m).copy(), lambda q: m[:, q], 1e-10)
    assert any("positive semidefinite" in r.message for r in kuiva_caplog.records)


# --- The orbit-complete (block) path -------------------------------------------------------
#
# What it exists for is symmetry (tested against real atoms further down); what is tested here
# is that it is a *correct decomposition* — the algebra, on synthetic matrices, in full
# generality. The two are separable and must be, because a routine that preserves symmetry
# while factorizing badly is exactly the failure mode this path was written to avoid.

def _synthetic_psd(n, rank, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, rank))
    return a @ a.T


def test_plain_path_is_bitwise_unchanged_when_no_orbits_are_given():
    """⚠ The property that makes this change safe to land before the default flips: passing
    no labelling must reproduce the validated path **exactly**, not merely closely."""
    m = _synthetic_psd(40, 12, seed=11)
    args = (np.diag(m).copy(), lambda q: m[:, q], 1e-10)
    a = pivoted_cholesky(*args)
    b = pivoted_cholesky(*args, orbits=None)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]


@pytest.mark.parametrize("tol", [1e-2, 1e-6, 1e-10])
def test_block_path_reconstructs_a_psd_matrix(tol):
    """The block path is a decomposition of the same quality as the plain one: the residual
    still bounds every element by Cauchy-Schwarz, which is the whole accuracy contract."""
    n, m = 48, _synthetic_psd(48, 20, seed=12)
    orbits = np.arange(n) // 4                           # 12 orbits of 4
    lvec, _, resid = pivoted_cholesky(np.diag(m).copy(), lambda q: m[:, q], tol,
                                      orbits=orbits)
    assert resid <= tol
    assert np.max(np.abs(lvec.T @ lvec - m)) <= tol


def test_block_path_matches_the_plain_path_in_accuracy_but_not_in_vectors():
    """⚠ Both halves matter. Equal accuracy is the claim; a *different* vector set is the
    evidence that the block path is actually doing something rather than silently falling
    through to plain pivoting."""
    n, tol = 48, 1e-8
    m = _synthetic_psd(n, 30, seed=13)
    orbits = np.arange(n) // 6
    plain, _, r_plain = pivoted_cholesky(np.diag(m).copy(), lambda q: m[:, q], tol)
    block, _, r_block = pivoted_cholesky(np.diag(m).copy(), lambda q: m[:, q], tol,
                                         orbits=orbits)
    assert r_plain <= tol and r_block <= tol
    assert np.max(np.abs(block.T @ block - m)) <= tol
    assert np.max(np.abs(plain.T @ plain - m)) <= tol
    assert block.shape[0] != plain.shape[0] or \
        not np.allclose(np.sort(np.abs(block).sum(axis=1)),
                        np.sort(np.abs(plain).sum(axis=1)))


def test_block_path_survives_a_rank_deficient_orbit():
    """The stability case the plain path gets for free and this one does not: an orbit whose
    Gram matrix is singular. Whole degenerate groups must drop out without the ``1/sqrt(s)``
    blowing up — the failure that cost 251 Eh on Lu(3+)."""
    n = 24
    base = _synthetic_psd(n, 24, seed=14)
    dup = np.eye(n)
    dup[:, 1] = dup[:, 0]                                # columns 0 and 1 identical => rank drop
    m = dup.T @ base @ dup
    orbits = np.arange(n) // 4
    lvec, _, resid = pivoted_cholesky(np.diag(m).copy(), lambda q: m[:, q], 1e-10,
                                      orbits=orbits)
    assert np.all(np.isfinite(lvec))
    assert resid <= 1e-10
    assert np.max(np.abs(lvec.T @ lvec - m)) < 1e-8


def test_block_path_never_splits_an_orbit_at_the_vector_limit():
    """⚠ Truncating inside an orbit is exactly what breaks the invariance, so the limit stops
    the decomposition short rather than emitting a partial orbit."""
    n = 40
    m = _synthetic_psd(n, 40, seed=15)
    orbits = np.arange(n) // 8                           # 5 orbits of 8
    lvec, _, _ = pivoted_cholesky(np.diag(m).copy(), lambda q: m[:, q], 1e-12,
                                  max_vectors=20, orbits=orbits)
    assert lvec.shape[0] <= 20
    assert lvec.shape[0] % 8 == 0                        # whole orbits only


@pytest.mark.parametrize("rtol", [1e-12, 1e-6, 1e-2, 1.0])
def test_block_path_cannot_produce_a_nan_at_any_grouping_tolerance(rtol):
    """⚠ The stability floor, and why it is a floor on the group's *smallest* member.

    A grouping tolerance loose enough to hold Lu(3+)'s degeneracies together merged distinct
    eigenvalues on Ar, and the merged group — kept because its largest member cleared the
    accuracy floor — contained a rounding-negative one, so ``s^-1/2`` produced NaN. Dropping
    a group whose smallest member is noise makes every ``degeneracy_rtol`` safe, at worst
    costing vectors. The accuracy is then reported honestly through the residual rather than
    silently assumed.
    """
    n = 24
    base = _synthetic_psd(n, 24, seed=17)
    dup = np.eye(n)
    dup[:, 1] = dup[:, 0]
    dup[:, 5] = dup[:, 4]                                # two exact rank drops, in two orbits
    m = dup.T @ base @ dup
    lvec, _, resid = pivoted_cholesky(np.diag(m).copy(), lambda q: m[:, q], 1e-10,
                                      orbits=np.arange(n) // 4, degeneracy_rtol=rtol)
    assert np.all(np.isfinite(lvec))
    assert resid >= 0.0
    # whatever was kept must still be a faithful decomposition of what it spans
    assert np.max(np.abs(lvec.T @ lvec - m)) <= max(resid, 1e-8)


# --- The orbit labelling itself ------------------------------------------------------------

def _layout(atom, charge=0):
    from kuiva.interface.pyscf_bridge import ao_layout
    from pyscf import gto
    mol = gto.M(atom=atom, charge=charge, spin=0, basis="x2c-SVPall-2c", verbose=0)
    return mol, ao_layout(mol)


def _orbit_sizes(labels):
    return np.bincount(np.unique(labels, return_inverse=True)[1])


def test_shell_pair_orbits_is_an_exact_cover_of_complete_shell_products():
    """⚠ A partition, not a suggestion: a duplicated column would be projected out twice and
    a missing one never at all, and both leave a plausible, wrong factorization. Each orbit
    must also be a **complete** shell-pair product — a partial one is not an invariant
    subspace and the whole invariance argument fails on it."""
    mol, lay = _layout("Ar 0 0 0")
    orbits = shell_pair_orbits(lay.ao_shell, lay.ao_atom)
    assert orbits.shape == (npair_of(mol.nao),)
    i, j = np.tril_indices(mol.nao)
    nshell_of = {int(s): 2 * int(lay.ao_l[np.nonzero(lay.ao_shell == s)[0][0]]) + 1
                 for s in np.unique(lay.ao_shell)}

    covered = 0
    for label in np.unique(orbits):
        members = np.nonzero(orbits == label)[0]
        shell_pairs = {tuple(sorted((int(lay.ao_shell[i[m]]), int(lay.ao_shell[j[m]]))))
                       for m in members}
        assert len(shell_pairs) == 1, "orbit {} mixes shell pairs {}".format(label, shell_pairs)
        b, a = sorted(shell_pairs.pop())
        na, nb = nshell_of[a], nshell_of[b]
        expected = na * (na + 1) // 2 if a == b else na * nb
        assert members.size == expected, (label, a, b, members.size, expected)
        covered += members.size
    assert covered == npair_of(mol.nao)


def test_shell_pair_orbits_keeps_off_atom_pairs_out_of_the_grouping():
    """⚠ One-centre by decision: only pairs on the same atom carry a spherical
    symmetry to preserve. Everything else is a singleton, for which the block step *is* an
    ordinary rank-one update, so a molecule keeps plain pivoting between its centres."""
    mol, lay = _layout("Ti 0 0 0; O 0 0 3.0")          # two centres; no SCF is needed here
    i, j = np.tril_indices(mol.nao)
    same = lay.ao_atom[i] == lay.ao_atom[j]

    one_centre = shell_pair_orbits(lay.ao_shell, lay.ao_atom)
    per_pair = _orbit_sizes(one_centre)[np.unique(one_centre, return_inverse=True)[1]]
    assert per_pair[~same].max() == 1                    # every off-atom pair is a singleton
    assert per_pair[same].max() > 1                      # on-atom pairs are grouped

    everything = shell_pair_orbits(lay.ao_shell, lay.ao_atom, one_centre=False)
    assert np.unique(everything).size < np.unique(one_centre).size    # strictly coarser
    assert one_centre.size == everything.size == npair_of(mol.nao)


def test_a_general_contraction_is_split_into_separate_orbits():
    """⚠ Two radial functions of the same ``l`` on the same atom are **not** related by any
    symmetry, so they may not share an orbit. ``AOLayout`` splits general contractions into
    one :class:`Shell` per contracted function, which is what makes this automatic — a
    labelling built from raw ``libcint`` shells would merge them."""
    _, lay = _layout("Ar 0 0 0")
    # every shell index covers exactly 2l+1 consecutive AOs of one l
    for s in np.unique(lay.ao_shell):
        aos = np.nonzero(lay.ao_shell == s)[0]
        l = int(lay.ao_l[aos[0]])
        assert aos.size == 2 * l + 1
        assert np.all(lay.ao_l[aos] == l)
    assert np.unique(lay.ao_shell).size > np.unique(lay.ao_l).size   # several radial per l


def test_the_front_end_pivots_on_symmetry_orbits_by_default(hf_conv):
    """⚠ The plumbing, which is what makes the symmetry statement true of a *calculation*
    rather than of a routine nobody calls that way. ``ScalarX2CData.ao_layout`` stopped being
    analysis-only metadata when this landed."""
    factors = ThreeIndexAO.from_scalar_data(hf_conv, report=False)
    assert factors.orbit_complete
    plain = ThreeIndexAO.from_scalar_data(hf_conv, report=False, orbit_pivots=False)
    assert not plain.orbit_complete
    assert factors.residual <= DEFAULT_CHOLESKY_TOL and plain.residual <= DEFAULT_CHOLESKY_TOL


def test_a_missing_layout_falls_back_to_plain_pivoting_with_a_warning(hf_conv, kuiva_caplog):
    """⚠ Falling back silently is the failure mode to avoid: the factorization would still be
    faithful to ``tol``, but its spherical symmetry would be back to depending on the
    threshold, and a free-ion splitting produced that way is not distinguishable after the
    fact from a physical one."""
    import dataclasses

    stripped = dataclasses.replace(hf_conv, ao_layout=None)
    with kuiva_caplog.at_level("WARNING"):
        factors = ThreeIndexAO.from_scalar_data(stripped, report=False)
    assert not factors.orbit_complete
    assert any("plain column pivoting" in r.message for r in kuiva_caplog.records)


def test_block_path_rejects_a_mislabelled_column_set():
    m = _synthetic_psd(10, 10, seed=16)
    with pytest.raises(ValueError, match="one label per column"):
        pivoted_cholesky(np.diag(m).copy(), lambda q: m[:, q], 1e-10,
                         orbits=np.arange(9))


# --- The transform: against brute force -----------------------------------------------------
@pytest.mark.parametrize("complex_", [False, True])
def test_transform_matches_brute_force(complex_):
    nao, nsp = 6, 5
    eri = synthetic_eri(nao, seed=4)
    f = ThreeIndexAO.from_eri(eri, nao, tol=CD_TOL, report=False)
    c = random_spinors(nao, nsp, seed=5, complex_=complex_)
    b = transform_3c(f, c, c)
    got = assemble_4c(b)
    ref = brute_force_spinor_eri(eri, c)
    assert np.max(np.abs(got - ref)) < 1e-7


def test_transform_with_different_bra_and_ket_spaces():
    """Mixed blocks (the CASSCF gradient and NEVPT2 classes are all mixed) must work too."""
    nao = 6
    eri = synthetic_eri(nao, seed=6)
    f = ThreeIndexAO.from_eri(eri, nao, tol=CD_TOL, report=False)
    c = random_spinors(nao, 2 * nao, seed=7)
    occ, vir = c[:, :3], c[:, 3:7]
    b_ov = transform_3c(f, occ, vir)
    b_vo = transform_3c(f, vir, occ)
    got = assemble_4c(b_ov, b_vo)                      # (ov|vo)
    ref = brute_force_spinor_eri(eri, c)[:3, 3:7, 3:7, :3]
    assert np.max(np.abs(got - ref)) < 1e-7


def test_aux_blocking_does_not_change_the_result():
    """The blocked loop is a memory strategy, not an approximation."""
    nao = 6
    eri = synthetic_eri(nao, seed=8)
    f = ThreeIndexAO.from_eri(eri, nao, tol=CD_TOL, report=False)
    c = random_spinors(nao, 6, seed=9)
    whole = transform_3c(f, c, c, aux_blocksize=f.naux)
    for nb in (1, 3, 7):
        assert np.max(np.abs(transform_3c(f, c, c, aux_blocksize=nb) - whole)) < ALG_TOL


def test_real_and_complex_paths_agree():
    """The real fast path must be exactly the complex path, not merely close to it."""
    nao = 5
    eri = synthetic_eri(nao, seed=10)
    f = ThreeIndexAO.from_eri(eri, nao, tol=CD_TOL, report=False)
    c_real = random_spinors(nao, 4, seed=11, complex_=False)
    b_fast = transform_3c(f, c_real, c_real)                        # takes the real GEMM path
    b_slow = transform_3c(f, c_real + 0j * 1e-300, c_real.astype(complex) + 1e-300j)
    assert np.max(np.abs(b_fast - b_slow)) < 1e-12


def test_real_dtype_request_with_complex_coefficients_is_refused():
    nao = 4
    f = ThreeIndexAO.from_eri(synthetic_eri(nao, seed=12), nao, tol=CD_TOL, report=False)
    c = random_spinors(nao, 3, seed=13, complex_=True)
    with pytest.raises(ValueError, match="imaginary part"):
        transform_3c(f, c, c, dtype=np.float64)


def test_result_dtype_is_complex_regardless_of_input():
    nao = 4
    f = ThreeIndexAO.from_eri(synthetic_eri(nao, seed=14), nao, tol=CD_TOL, report=False)
    c = random_spinors(nao, 3, seed=15, complex_=False)
    assert transform_3c(f, c, c).dtype == np.complex128


# --- Symmetries the transformed integrals must have -------------------------------------------
def test_four_fold_permutational_symmetry():
    """Complex spinor integrals have 4-fold symmetry, not 8-fold."""
    nao = 5
    eri = synthetic_eri(nao, seed=16)
    f = ThreeIndexAO.from_eri(eri, nao, tol=CD_TOL, report=False)
    c = random_spinors(nao, 4, seed=17)
    g = assemble_4c(transform_3c(f, c, c))
    herm, swap = check_permutational_symmetry(g)
    assert herm < ALG_TOL and swap < ALG_TOL
    # ... and 8-fold does NOT hold: (pq|rs) != (qp|rs) for complex spinors.
    assert np.max(np.abs(g - np.transpose(g, (1, 0, 2, 3)))) > 1e-6


def test_three_index_block_is_hermitian():
    nao = 5
    f = ThreeIndexAO.from_eri(synthetic_eri(nao, seed=18), nao, tol=CD_TOL, report=False)
    c = random_spinors(nao, 4, seed=19)
    b = transform_3c(f, c, c)
    assert np.max(np.abs(b - np.conj(np.transpose(b, (0, 2, 1))))) < ALG_TOL


def test_assemble_4c_conjugation_convention():
    """Guard the documented trap: conjugating in the assembly gives a *different*, plausible
    answer. If someone 'fixes' assemble_4c by adding a .conj(), this fails."""
    nao = 5
    f = ThreeIndexAO.from_eri(synthetic_eri(nao, seed=20), nao, tol=CD_TOL, report=False)
    c = random_spinors(nao, 4, seed=21)
    b = transform_3c(f, c, c)
    right = assemble_4c(b)
    wrong = np.tensordot(b.conj(), b, axes=([0], [0]))
    assert np.max(np.abs(right - brute_force_spinor_eri(
        synthetic_eri(nao, seed=20), c))) < 1e-7
    assert np.max(np.abs(right - wrong)) > 1e-3


def test_invariance_under_a_unitary_rotation_of_the_spinors():
    """A Tier-0 invariance: the CI energy cannot depend on the basis of the active
    space, so the fully contracted trace of the integrals must be rotation invariant."""
    nao = 5
    f = ThreeIndexAO.from_eri(synthetic_eri(nao, seed=22), nao, tol=CD_TOL, report=False)
    c = random_spinors(nao, 6, seed=23)
    rng = np.random.default_rng(24)
    u, _ = np.linalg.qr(rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6)))
    g0 = assemble_4c(transform_3c(f, c, c))
    g1 = assemble_4c(transform_3c(f, c @ u, c @ u))
    inv0 = np.einsum("ppqq->", g0) + 0j, np.einsum("pqqp->", g0) + 0j
    inv1 = np.einsum("ppqq->", g1) + 0j, np.einsum("pqqp->", g1) + 0j
    assert abs(inv0[0] - inv1[0]) < 1e-8 and abs(inv0[1] - inv1[1]) < 1e-8


# --- Kramers structure of the SOC-free guess --------------------------------------------------
def test_spin_forbidden_blocks_vanish_for_the_guess():
    """With a Kramers-paired, spin-pure guess, B^P_{p pbar} = 0: the AO integrals are
    spin-free, so a pair of opposite-spin spinors cannot contribute to a density."""
    nao = 5
    f = ThreeIndexAO.from_eri(synthetic_eri(nao, seed=25), nao, tol=CD_TOL, report=False)
    mo = np.linalg.qr(np.random.default_rng(26).standard_normal((nao, nao)))[0]
    sb = expand_scalar_mos(mo, basis="ao")
    b = transform_3c(f, sb.c, sb.c)
    assert np.max(np.abs(b[:, 0::2, 1::2])) < 1e-12    # unbarred-barred: spin forbidden
    assert np.max(np.abs(b[:, 1::2, 0::2])) < 1e-12
    assert np.max(np.abs(b[:, 0::2, 0::2])) > 1e-3     # same-spin blocks are not zero


def test_spinor_integrals_reduce_to_the_scalar_ones():
    """(2p 2p|2q 2q) over spinors must equal the scalar (pp|qq): the SOC-free guess adds no
    physics, only a doubling of the basis. This is the 'same problem in a doubled
    basis' check at the integral level."""
    nao = 5
    eri = synthetic_eri(nao, seed=27)
    f = ThreeIndexAO.from_eri(eri, nao, tol=CD_TOL, report=False)
    mo = np.linalg.qr(np.random.default_rng(28).standard_normal((nao, nao)))[0]
    sb = expand_scalar_mos(mo, basis="ao")
    g = assemble_4c(transform_3c(f, sb.c, sb.c))
    scalar = np.einsum("mnkl,mp,nq,kr,ls->pqrs", eri, mo, mo, mo, mo, optimize=True)
    for p in range(nao):
        for q in range(nao):
            assert abs(g[2 * p, 2 * p, 2 * q, 2 * q] - scalar[p, p, q, q]) < 1e-7
            # and the Kramers partner sees exactly the same interaction
            assert abs(g[2 * p + 1, 2 * p + 1, 2 * q, 2 * q] - scalar[p, p, q, q]) < 1e-7


def test_kramers_degenerate_one_electron_energies():
    """h_pp = h_pbar,pbar for a time-reversal-even one-electron operator."""
    nbas = 6
    rng = np.random.default_rng(29)
    h = rng.standard_normal((nbas, nbas))
    h = h + h.T
    mo = np.linalg.qr(rng.standard_normal((nbas, nbas)))[0]
    sb = expand_scalar_mos(mo, basis="ao")
    h_sp = transform_1e(h, sb.c)
    assert np.max(np.abs(np.diag(h_sp)[0::2] - np.diag(h_sp)[1::2])) < 1e-12
    # the same operator lifted explicitly to two components must give the same matrix
    assert np.max(np.abs(h_sp - transform_1e(spin_block_diagonal(h), sb.c))) < 1e-12


def test_transform_1e_rejects_a_mismatched_operator():
    with pytest.raises(ValueError, match="matches neither"):
        transform_1e(np.eye(7), np.zeros((12, 3), dtype=complex))


# --- Real integrals: conventions ---------------------------------------------------------------
@pytest.fixture(scope="module")
def hf_conv():
    mol = Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917", basis="x2c-SVPall-2c")
    return run_scalar_x2c(mol, fitting="conventional", screening="none")


@pytest.fixture(scope="module")
def hf_df():
    mol = Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917", basis="x2c-SVPall-2c")
    return run_scalar_x2c(mol, fitting="df", screening="none")


def test_s8_packing_convention_matches_pyscf(hf_conv):
    """The index formula for PySCF's 8-fold-packed array. A silent off-by-one here would
    corrupt every two-electron integral in the program."""
    from kuiva.interface.pyscf_bridge import build_mole

    mol = build_mole(Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917", basis="x2c-SVPall-2c"))
    full = mol.intor("int2e")                          # (nao,)*4, unpacked reference
    nao = mol.nao
    f = ThreeIndexAO.from_eri(hf_conv.eri, nao, tol=1e-10, report=False)
    approx = assemble_4c(f.unpack(slice(None)).reshape(f.naux, nao, nao))
    assert np.max(np.abs(approx - full)) < 1e-8


def test_df_and_cholesky_routes_agree(hf_conv, hf_df, kuiva_caplog):
    """The two factorization routes give the same integrals through the same interface — but only to
    within the *fitting* error, and this test records how big that actually is.

    With the registry's recommended Coulomb-fitting auxiliary (``x2c-JFIT``) the worst
    active-space integral differs by ~1.7e-3 Eh. That is a property of J-fitting sets used
    for correlated integrals, not a bug, and it is why ``from_df`` warns: the Cholesky route
    bounds its error by a threshold, a density fit does not bound its error at all. If this
    tolerance ever has to be *loosened*, something has gone wrong; if a JK-fitting auxiliary
    is adopted it should be tightened.
    """
    f_cd = ThreeIndexAO.from_scalar_data(hf_conv, DEFAULT_CHOLESKY_TOL, report=False)
    with kuiva_caplog.at_level("WARNING"):
        f_df = ThreeIndexAO.from_scalar_data(hf_df, report=False)
    assert f_cd.origin == "cholesky" and f_df.origin == "df"
    assert any("Coulomb-fitting auxiliary" in r.message for r in kuiva_caplog.records)
    ob = canonical_orthogonalization(hf_conv.s_ao)
    sb = expand_scalar_mos(ob.to_working(hf_conv.mo_coeff)).transform_scalar_basis(ob.x, "ao")
    act = np.arange(6, 16)
    g_cd = assemble_4c(transform_3c(f_cd, sb.take(act), sb.take(act)))
    g_df = assemble_4c(transform_3c(f_df, sb.take(act), sb.take(act)))
    err = np.max(np.abs(g_cd - g_df))
    assert 1e-4 < err < 5e-3                           # the measured J-fitting error


def test_full_pipeline_integrals_are_hermitian(hf_conv):
    """Ingestion -> working basis -> spinors -> integrals, with the invariants that must
    hold at the end of it."""
    ob = canonical_orthogonalization(hf_conv.s_ao)
    sb = expand_scalar_mos(ob.to_working(hf_conv.mo_coeff), hf_conv.mo_energy,
                           hf_conv.mo_occ)
    f = ThreeIndexAO.from_scalar_data(hf_conv, report=False)
    c_ao = sb.transform_scalar_basis(ob.x, "ao").take(np.arange(4, 14))
    ints = SpinorMOIntegrals.build(f, hf_conv.h_x2c, c_ao, e_nuc=hf_conv.e_nuc,
                                   report=False)
    assert ints.n == 10
    assert ints.hermiticity_error() < 1e-10
    herm, swap = check_permutational_symmetry(ints.eri())
    assert herm < 1e-10 and swap < 1e-10
    # Kramers degeneracy of the one-electron energies survives the whole pipeline.
    assert np.max(np.abs(np.diag(ints.h)[0::2] - np.diag(ints.h)[1::2])) < 1e-10


def test_time_reversed_orbitals_give_conjugate_integrals(hf_conv):
    """Time reversal maps (pq|rs) to its complex conjugate: a physical check on the
    transform that involves both the spin blocking and the conjugation convention."""
    ob = canonical_orthogonalization(hf_conv.s_ao)
    sb = expand_scalar_mos(ob.to_working(hf_conv.mo_coeff)).transform_scalar_basis(ob.x, "ao")
    f = ThreeIndexAO.from_scalar_data(hf_conv, report=False)
    rng = np.random.default_rng(30)
    n = 6
    u, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    c = sb.c[:, 4:4 + n] @ u
    g = assemble_4c(transform_3c(f, c, c))
    g_t = assemble_4c(transform_3c(f, time_reverse(c), time_reverse(c)))
    assert np.max(np.abs(g_t - g.conj())) < 1e-9


# --- ⚠ The Cholesky error is ANISOTROPIC, and that is what sets the threshold -------------
#
# A spherical atom's integrals commute with SO(3); a pivoted Cholesky factorization of them
# does not, because the pivots are particular AO pairs chosen in a particular order. The
# rank-truncated ``(pq|rs)`` therefore treats the ``m`` components of a shell inequivalently
# at the size of the threshold, splitting degeneracies symmetry makes exact.
#
# ⚠ **Two independent channels carry the damage, and neither dominates** (measured; see
# measured). A guard on one of them alone understates the
# worst case by up to two orders of magnitude, which is what the original single-shell,
# Fock-only version of this guard did:
#
#   channel F (inactive Fock)  the block of ``J - K/2`` over a complete shell must be a
#                              multiple of the identity -- Schur's lemma on the one-particle
#                              operator. This is what a CASSCF with ONE active electron sees.
#   channel A (active ERI)     the shell-only integrals as a matrix ``M_{(tu),(vw)}`` on the
#                              ``(2l+1)^2`` pair space. ``D^l (x) D^l = (+)_L D^L`` with each
#                              ``L`` exactly once, so ``M`` is block-scalar: its eigenvalues
#                              come in degenerate groups of ``2L+1``, the odd-``L``
#                              (antisymmetric) part being exactly zero. This is what a
#                              many-electron active space sees, and it is the channel a
#                              Fock-only probe is blind to.
#
# ⚠ **And it must be measured on every complete shell, not one.** Different radial functions
# of the same ``l`` break by amounts two orders of magnitude apart, so probing whichever
# contraction happens to be last measures nothing in particular.

_SPHERICAL_ATOMS: dict = {}


def _spherical_atom(atom: str, charge: int = 0):
    """A converged, **spherical** closed-shell atom: ``(mol, density, eri8)``, cached.

    ⚠ **Plain RHF is not enough, and assuming it is has invalidated real measurements.**
    Unconstrained RHF on Lu(3+) — nominally [Xe]4f14, "closed shell, hence spherical" —
    converges to a symmetry-*broken* solution whose 4f shell is split by 0.03 Eh and which
    lies 0.23 Eh *below* the symmetric one. SO(3) symmetry is imposed on the SCF, and the
    exact-integral control in :func:`_shell_symmetry_spread` is what proves it took.
    """
    from pyscf import gto, scf

    key = (atom, charge)
    if key not in _SPHERICAL_ATOMS:
        mol = gto.M(atom="{} 0 0 0".format(atom), charge=charge, spin=0,
                    basis="x2c-SVPall-2c", verbose=0, symmetry=True)
        assert mol.topgroup == "SO3", mol.topgroup
        mf = scf.RHF(mol)
        mf.kernel(dm0=scf.hf.get_init_guess(mol, "atom"))
        assert mf.converged, "{}{:+d} SCF did not converge".format(atom, charge)
        _SPHERICAL_ATOMS[key] = (mol, mf.make_rdm1(), mol.intor("int2e", aosym="s8"))
    return _SPHERICAL_ATOMS[key]


def _complete_shells(mol, l_probe):
    """Every complete ``l`` shell of the AO basis — one per contraction, orthonormal."""
    ao_l = np.concatenate([[mol.bas_angular(b)] * (2 * mol.bas_angular(b) + 1)
                           * mol.bas_nctr(b) for b in range(mol.nbas)])
    n, ovlp = 2 * l_probe + 1, mol.intor("int1e_ovlp")
    idx = np.nonzero(ao_l == l_probe)[0]
    shells = []
    for k in range(0, len(idx), n):
        shell = np.zeros((mol.nao, n))
        shell[idx[k:k + n], np.arange(n)] = 1.0
        shell /= np.sqrt(np.diag(shell.T @ ovlp @ shell))
        # different m of one contracted shell are orthogonal by symmetry; rely on it
        assert np.abs(shell.T @ ovlp @ shell - np.eye(n)).max() < 1e-12
        shells.append(shell)
    return shells


def _fock_spread(j, k, shell):
    """Channel F: spread of the diagonal of ``J - K/2`` restricted to ``shell`` [Eh]."""
    diag = np.diag(shell.T @ (j - 0.5 * k) @ shell)
    return float(np.abs(diag - diag.mean()).max())


def _group_spread(pair, bounds):
    """Channel A: worst spread inside a degenerate eigenvalue group of ``(tu|vw)`` [Eh]."""
    w = np.sort(np.linalg.eigvalsh(pair))
    return max(float(w[bounds[i]:bounds[i + 1]].ptp()) for i in range(len(bounds) - 1))


def _eri_groups(pair, l_probe):
    """Eigenvalue-group index boundaries of ``(tu|vw)``, taken from the **exact** spectrum
    and reused on the low-rank one (whose perturbation is far below the group gaps).

    ⚠ The *ordering* of the ``L`` blocks is empirical (the Slater–Condon ``F^k`` fall with
    ``k``), never a theorem, so the boundaries are found by clustering and only *checked*
    against the SO(3) multiplicities — a mismatch means the probe is invalid, not that the
    factorization is bad.
    """
    w = np.sort(np.linalg.eigvalsh(pair))
    cut = np.nonzero(np.diff(w) > 1e-6 * max(float(np.abs(w).max()), 1.0))[0] + 1
    bounds = [0] + list(cut) + [w.size]
    sizes = sorted(bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1))
    n = 2 * l_probe + 1
    expected = sorted([n * (n - 1) // 2] + [2 * L + 1 for L in range(0, 2 * l_probe + 1, 2)])
    assert sizes == expected, \
        "eigenvalue groups {} do not match the SO(3) multiplicities {}: the exact spectrum " \
        "is not the symmetric one, so this probe is invalid".format(sizes, expected)
    return bounds


def _shell_symmetry_spread(tol: float, atom: str = "Ar", charge: int = 0,
                           angular=(1, 2), pivots: str = "orbit"):
    """``(exact, cholesky)`` worst symmetry breaking [Eh] over both channels and every
    complete shell of every ``l`` in ``angular``.

    The ``exact`` figure is the control reproducibility requires of any measurement of this invariant:
    if the exact integrals already break it, the density is not spherical and the Cholesky
    number means nothing.

    ``pivots`` is ``"orbit"`` — complete symmetry orbits, what the front-end now does by
    default — or ``"column"`` for plain pivoting, which is only a control here. The labelling
    is built from the **same** ``mol`` the integrals come from, so no basis bookkeeping can
    drift between the two.
    """
    from pyscf import ao2mo, scf

    from kuiva.interface.pyscf_bridge import ao_layout

    mol, density, eri8 = _spherical_atom(atom, charge)
    if pivots == "orbit":
        layout = ao_layout(mol)
        orbits = shell_pair_orbits(layout.ao_shell, layout.ao_atom)
    elif pivots == "column":
        orbits = None
    else:
        raise ValueError("pivots must be 'orbit' or 'column', got {!r}".format(pivots))
    j_ex, k_ex = scf.hf.dot_eri_dm(eri8, density, hermi=1)
    factors = ThreeIndexAO.from_eri(eri8, mol.nao, tol, orbits=orbits, report=False)
    assert factors.residual <= tol                    # the accuracy contract, always
    lsq = factors.unpack(slice(None))
    j_ch = np.tensordot(np.tensordot(lsq, density, axes=([1, 2], [0, 1])), lsq, axes=(0, 0))
    k_ch = np.matmul(np.matmul(lsq, density), lsq).sum(axis=0)

    worst_exact = worst_chol = 0.0
    for l_probe in angular:
        for shell in _complete_shells(mol, l_probe):
            n = shell.shape[1]
            pair_ex = ao2mo.incore.full(eri8, shell, compact=False).reshape(n * n, n * n)
            pair_ex = 0.5 * (pair_ex + pair_ex.T)
            b = np.matmul(shell.T, np.matmul(lsq, shell)).reshape(lsq.shape[0], -1)
            bounds = _eri_groups(pair_ex, l_probe)
            worst_exact = max(worst_exact, _fock_spread(j_ex, k_ex, shell),
                              _group_spread(pair_ex, bounds))
            worst_chol = max(worst_chol, _fock_spread(j_ch, k_ch, shell),
                             _group_spread(b.T @ b, bounds))
    return worst_exact, worst_chol


#: Degeneracy of a complete atomic shell, in Eh. 1e-10 Eh is 2e-5 cm^-1 — two thousand times
#: below the ~0.1 cm^-1 at which a splitting starts to mean different physics, and five orders
#: above the machine-precision value the default threshold delivers on Ar.
SHELL_DEGENERACY_TOL = 1e-10

#: ⚠ **A threshold deliberately far looser than anything anyone would run**, for the tests
#: that assert the symmetry is *structural*. At 1e-6 plain pivoting breaks Ar's shells by
#: 7e-7 Eh (0.16 cm^-1) and Lu(3+)'s f shells by 0.10 cm^-1; the orbit path must be at machine
#: precision there. This is the whole Stage-3 claim, and it is why the number is 100x the
#: default rather than 2x.
LOOSE_TOL = 1e-6


def test_the_default_factorization_preserves_atomic_shell_degeneracy():
    """⚠ **The regression guard for a real defect.** With plain column pivoting at the old 1e-6
    default this broke Ar's shells by 6.4e-7 Eh = 0.14 cm^-1, which propagated straight into a
    free-ion ``2J+1`` multiplet splitting of 0.23 cm^-1 in Ce(3+) — a quantity symmetry makes
    exactly zero. Orbit pivoting takes Yb(3+) from 0.00215 to 0.000051 cm^-1 at the *same*
    threshold.

    ⚠ **Dy(3+)'s 44.85 cm^-1 was NOT this**, and the earlier version of this docstring said it
    was. That splitting is reproduced to six figures by the orbit-complete factorization; its
    cause was the state-average count. Two mechanisms that both split
    a degeneracy symmetry makes exact are still two mechanisms.

    This runs what the front-end runs: the default threshold **and** the default pivot
    selection."""
    exact, cholesky = _shell_symmetry_spread(DEFAULT_CHOLESKY_TOL)
    assert exact < SHELL_DEGENERACY_TOL, \
        "the exact integrals already break the shell degeneracy ({:.2e} Eh); the test system " \
        "is not spherical and this test is measuring the wrong thing".format(exact)
    assert cholesky < SHELL_DEGENERACY_TOL, (
        "the Cholesky factorization at tol = {:.0e} splits a complete atomic shell by "
        "{:.2e} Eh ({:.3f} cm^-1). That degeneracy is exact by Schur's lemma, so this is a "
        "symmetry-breaking artifact of the pivoted decomposition and it propagates directly "
        "into free-ion J-multiplet splittings.".format(DEFAULT_CHOLESKY_TOL, cholesky,
                                                      cholesky * 219474.6313632))


def test_the_symmetry_no_longer_depends_on_the_threshold():
    """⚠ **The statement Stage 3 exists to make.** The threshold used to be doing two
    unrelated jobs — bounding the energy error *and* hiding a symmetry violation — and the
    1e-8 default was set by the second. With complete symmetry orbits as pivots the invariance
    is exact by construction, so a threshold **100x looser than the default** must preserve a
    complete shell's degeneracy just as well.

    If this ever fails while the test above passes, the symmetry has silently gone back to
    being bought with the threshold.
    """
    exact, cholesky = _shell_symmetry_spread(LOOSE_TOL)
    assert exact < SHELL_DEGENERACY_TOL, exact
    assert cholesky < SHELL_DEGENERACY_TOL, (
        "the orbit-complete factorization at a deliberately loose tol = {:.0e} splits a "
        "complete atomic shell by {:.2e} Eh ({:.4f} cm^-1); the symmetry is threshold-"
        "dependent again".format(LOOSE_TOL, cholesky, cholesky * 219474.6313632))


def test_plain_column_pivoting_still_breaks_it():
    """⚠ The control that makes both tests above meaningful: the invariant must be able to
    fail. It is measured on the **plain** path, which is the only thing left that breaks it —
    at 1e-6 the same shells split by ~7e-7 Eh, three orders above the bound.

    A guard nothing can fail proves nothing, and once the orbit path became the default this
    is where the sensitivity had to move.
    """
    _, loose = _shell_symmetry_spread(LOOSE_TOL, pivots="column")
    assert loose > 100 * SHELL_DEGENERACY_TOL, \
        "plain column pivoting at tol = {:.0e} no longer breaks the shell degeneracy " \
        "({:.2e} Eh); the invariant has stopped being sensitive and the guards above prove " \
        "nothing".format(LOOSE_TOL, loose)


@pytest.mark.slow
@pytest.mark.parametrize("tol", [DEFAULT_CHOLESKY_TOL, LOOSE_TOL])
def test_f_shell_degeneracy_is_structural(tol):
    """⚠ **The f probe, and it is a different statement from the Ar one.**

    Ar's small, well-conditioned d and p shells reached machine precision under plain pivoting
    at the default threshold; Lu(3+)'s 4f shells never did — they broke by 0.0038 cm^-1 at 1e-8
    and 0.100 cm^-1 at 1e-6, i.e. the margin against the 0.01 cm^-1 physical requirement
    was a factor of 2.6 and it was bought entirely with the threshold. With orbit pivots both
    thresholds are at machine precision, which is the point.

    ⚠ **Not tested at 1e-4**, and deliberately: Lu(3+) retains 0.23 cm^-1 there (170x better
    than plain, not machine precision — measured). Every integral in that
    factorization already carries 1e-4 Eh of error, so it is outside the usable range, but the
    claim being made here is "structural at any *usable* threshold", not "at any threshold".

    Lu(3+) is the right f probe because [Xe]4f14 is genuinely closed-shell — but see
    :func:`_spherical_atom` for why its SCF still needs SO(3) imposed.
    """
    exact, cholesky = _shell_symmetry_spread(tol, "Lu", 3, angular=(3,))
    assert exact < SHELL_DEGENERACY_TOL, \
        "the exact integrals already break the f shell degeneracy ({:.2e} Eh); the Lu(3+) " \
        "reference is not spherical and this test is measuring the wrong thing".format(exact)
    assert cholesky < SHELL_DEGENERACY_TOL, (
        "the orbit-complete factorization at tol = {:.0e} splits a complete atomic f shell by "
        "{:.2e} Eh ({:.4f} cm^-1); a free ion's 2J+1 multiplets stop being degenerate to a "
        "physically meaningful accuracy at 0.01 cm^-1, and this is supposed "
        "to be structural rather than threshold-dependent."
        .format(tol, cholesky, cholesky * 219474.6313632))
