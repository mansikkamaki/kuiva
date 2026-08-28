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
                                       _s8_column, _s8_diagonal,
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


@pytest.mark.parametrize("orbits", [None, "blocks"])
def test_growing_the_factor_array_changes_nothing_but_the_allocation(orbits):
    """⚠ The factor array is grown as the decomposition fills it, and that must be invisible.

    Allocating for the worst case instead — one vector per column — is an array *larger than
    the two-electron integrals it factorizes* (27.5 GB against 13.7 GB at nao = 348), and it
    was the real ceiling on system size. Growing it is only safe if the result does not depend
    on the capacity it started from, so this asserts **bitwise** equality across starting
    capacities from one row to the full worst case.
    """
    n = 48
    m = _synthetic_psd(n, 20, seed=15)
    orb = None if orbits is None else np.arange(n) // 6
    args = (np.diag(m).copy(), lambda q: m[:, q], 1e-10)
    reference = pivoted_cholesky(*args, orbits=orb, initial_vectors=n)
    for capacity in (1, 2, 7, 13, None):
        got = pivoted_cholesky(*args, orbits=orb, initial_vectors=capacity)
        assert np.array_equal(got[0], reference[0]), "capacity {}".format(capacity)
        assert np.array_equal(got[1], reference[1]) and got[2] == reference[2]


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
    factors = ThreeIndexAO.from_scalar_data(hf_conv, report=False, release_eri=False)
    assert factors.orbit_complete
    plain = ThreeIndexAO.from_scalar_data(hf_conv, report=False, orbit_pivots=False,
                                          release_eri=False)
    assert not plain.orbit_complete
    assert factors.residual <= DEFAULT_CHOLESKY_TOL and plain.residual <= DEFAULT_CHOLESKY_TOL


def test_a_missing_layout_falls_back_to_plain_pivoting_with_a_warning(hf_conv, kuiva_caplog):
    """⚠ Falling back silently is the failure mode to avoid: the factorization would still be
    faithful to ``tol``, but its spherical symmetry would be back to depending on the
    threshold, and a free-ion splitting produced that way is not distinguishable after the
    fact from a physical one."""
    import dataclasses

    # ⚠ release_eri=False on a ``replace`` copy for a reason: the copy shares both the
    # array and its ledger entry with the fixture, so releasing here would empty the
    # fixture's reservation while every later test still holds its array.
    stripped = dataclasses.replace(hf_conv, ao_layout=None)
    with kuiva_caplog.at_level("WARNING"):
        factors = ThreeIndexAO.from_scalar_data(stripped, report=False, release_eri=False)
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


def test_coulomb_exchange_right_half_is_a_pure_cache():
    """Passing the precomputed half transform must change nothing but the cost."""
    from kuiva.integrals.transform import coulomb_exchange, half_transform
    nao = 6
    eri = synthetic_eri(nao, seed=11)
    f = ThreeIndexAO.from_eri(eri, nao, tol=CD_TOL, report=False)
    left = random_spinors(nao, 4, seed=12)
    right = random_spinors(nao, 4, seed=13)
    j_ref, k_ref = coulomb_exchange(f, left, orbitals_right=right)
    j_c, k_c = coulomb_exchange(f, left, orbitals_right=right,
                                right_half=half_transform(f, right))
    scale = max(1.0, float(np.max(np.abs(k_ref))))
    assert np.max(np.abs(j_c - j_ref)) < 1e-13 * scale
    assert np.max(np.abs(k_c - k_ref)) < 1e-13 * scale
    with pytest.raises(ValueError):                     # J needs the raw coefficients too
        coulomb_exchange(f, left, right_half=half_transform(f, right))
    with pytest.raises(ValueError):                     # a wrong-shape cache is refused
        coulomb_exchange(f, left, orbitals_right=right,
                         right_half=half_transform(f, right[:, :2]))


def test_one_sided_jk_hermitized_equals_the_symmetrized_build():
    """J and K are linear in the density and map D^dag to the Hermitian adjoint.

    This identity is what lets the orbital Hessian build its transition densities
    one-sided (A = X C^dag alone, Hermitized afterwards) at half the half-transform and
    pair-contraction cost of the symmetrized [X | C], [C | X] build. It holds only over
    real AO integrals, which every factorization here produces.
    """
    from kuiva.integrals.transform import coulomb_exchange
    nao = 6
    eri = synthetic_eri(nao, seed=14)
    f = ThreeIndexAO.from_eri(eri, nao, tol=CD_TOL, report=False)
    x = random_spinors(nao, 3, seed=15)
    c = random_spinors(nao, 3, seed=16)
    j2, k2 = coulomb_exchange(f, np.concatenate([x, c], axis=1),
                              orbitals_right=np.concatenate([c, x], axis=1))
    j1, k1 = coulomb_exchange(f, x, orbitals_right=c)
    scale = max(1.0, float(np.max(np.abs(k2))))
    assert np.max(np.abs((j1 + j1.conj().T) - j2)) < 1e-13 * scale
    assert np.max(np.abs((k1 + k1.conj().T) - k2)) < 1e-13 * scale


def test_diagonal_pair_blocks_match_brute_force():
    """b_diag and the exchange sums against the brute-force spinor integrals:
    ``(pp|qq) = sum_P b_diag[P,p] b_diag[P,q]`` and ``s2[p,j] = (p c_j | c_j p)``."""
    from kuiva.integrals.transform import diagonal_pair_blocks
    nao = 6
    eri = synthetic_eri(nao, seed=19)
    f = ThreeIndexAO.from_eri(eri, nao, tol=CD_TOL, report=False)
    c = random_spinors(nao, 2 * nao, seed=20)
    cols = np.array([1, 4, 7])
    b_diag, s2 = diagonal_pair_blocks(f, c, cols)
    ref = brute_force_spinor_eri(eri, c)
    n = c.shape[1]
    ppqq = b_diag.T @ b_diag
    for p in range(n):
        for q in range(n):
            assert abs(ppqq[p, q] - np.real(ref[p, p, q, q])) < 1e-7
    for p in range(n):
        for j, q in enumerate(cols):
            assert abs(s2[p, j] - np.real(ref[p, q, q, p])) < 1e-7


def test_diagonal_pair_blocks_rows_and_half_are_pure_restructurings():
    """``rows=`` restricts, ``half=`` reuses — neither may change a number."""
    from kuiva.integrals.transform import diagonal_pair_blocks, half_transform
    nao = 6
    eri = synthetic_eri(nao, seed=21)
    f = ThreeIndexAO.from_eri(eri, nao, tol=CD_TOL, report=False)
    c = random_spinors(nao, 2 * nao, seed=22)
    n = c.shape[1]
    cols = np.array([0, 2, 5])
    rows = np.array([3, 8, 9, 11])
    b_ref, s2_ref = diagonal_pair_blocks(f, c, cols)
    b_r, s2_r = diagonal_pair_blocks(f, c, cols, rows=rows)
    assert np.max(np.abs(b_r - b_ref)) < 1e-12
    assert np.max(np.abs(s2_r - s2_ref[rows])) < 1e-12
    have = np.array([0, 1, 2, 4, 5, 6])                # covers cols
    w = half_transform(f, np.ascontiguousarray(c[:, have]))
    b_h, s2_h = diagonal_pair_blocks(f, c, cols, rows=rows, half=(have, w))
    scale = max(1.0, float(np.max(np.abs(b_ref))))
    assert np.max(np.abs(b_h - b_ref)) < 1e-12 * scale
    assert np.max(np.abs(s2_h - s2_ref[rows])) < 1e-12 * scale
    with pytest.raises(ValueError):                     # cols outside the cached columns
        diagonal_pair_blocks(f, c, np.array([3]), half=(have, w))


def test_half_transform_sizing_is_exact():
    """Two-sided pin of the sizing function against a real result's nbytes — sizing
    functions are exact and never pad, like every other one in the resource budget."""
    from kuiva.integrals.transform import half_transform, half_transform_memory_gb
    nao, k = 6, 4
    eri = synthetic_eri(nao, seed=17)
    f = ThreeIndexAO.from_eri(eri, nao, tol=CD_TOL, report=False)
    w = half_transform(f, random_spinors(nao, k, seed=18))
    nbytes = sum(a.nbytes for a in w)
    assert all(a.dtype == np.complex128 for a in w)
    got = half_transform_memory_gb(f.naux, nao, k) * 1024.0 ** 3
    assert got == pytest.approx(nbytes, rel=0.0, abs=0.5)


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


# --- the integral-direct route ------------------------------------------------------------
# ⚠ The acceptance criterion for this route was fixed before it was written and is **bitwise**,
# not a tolerance: it evaluates the same matrix elements the stored route reads, so the same
# pivots in the same order must come out of the same arithmetic. A threshold comparison here
# would hide exactly the failure that matters — an evaluator that serves the right numbers in
# the wrong order, which changes which columns are selected while every norm stays plausible.


@pytest.fixture(scope="module")
def hf_mole():
    from kuiva.interface.pyscf_bridge import build_mole
    return build_mole(Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917",
                                               basis="x2c-SVPall-2c"))


def _direct_source(mol):
    from kuiva.interface.pyscf_bridge import DirectERIMatrix
    return DirectERIMatrix(mol)


def _stored_matrix(mol):
    """The same matrix the direct evaluator serves, materialized: ``(mu nu | la ka)``."""
    i, j = np.tril_indices(mol.nao)
    full = mol.intor("int2e")
    return full[i[:, None], j[:, None], i[None, :], j[None, :]]


def test_the_direct_evaluator_serves_the_matrix_element_for_element(hf_mole):
    """Every column and the diagonal, against the stored matrix — bitwise.

    ⚠ Including the diagonal *as a separate check*: the plain path divides a column by the
    square root of a diagonal element, so a diagonal that came from a different evaluation
    than its column would make a factorization that is nearly, but not exactly, the one
    asked for.
    """
    src = _direct_source(hf_mole)
    m = _stored_matrix(hf_mole)
    assert np.array_equal(src.diagonal(), np.diag(m))
    for q in range(m.shape[1]):
        assert np.array_equal(src.column(q), m[:, q]), "column {} differs".format(q)
    assert src.diagonal()[3] == src.column(3)[3]


def test_the_direct_route_reproduces_the_stored_route_bitwise(hf_mole):
    """The acceptance criterion: same factors, bit for bit, on both pivot rules."""
    from kuiva.interface.pyscf_bridge import ao_layout

    nao = hf_mole.nao
    m = _stored_matrix(hf_mole)
    layout = ao_layout(hf_mole)
    orbits = shell_pair_orbits(layout.ao_shell, layout.ao_atom)
    for label, orb in (("orbit-complete", orbits), ("plain pivoting", None)):
        src = _direct_source(hf_mole)
        direct = ThreeIndexAO.from_matrix(src.diagonal(), src.column, nao, CD_TOL,
                                          orbits=orb, report=False)
        stored = ThreeIndexAO.from_eri(m, nao, CD_TOL, orbits=orb, report=False)
        assert direct.naux == stored.naux, label
        assert np.array_equal(direct.l_packed, stored.l_packed), label
        assert direct.residual == stored.residual, label


def test_the_direct_route_never_forms_the_integral_array(hf_mole, monkeypatch):
    """The point of the route. Asserted by refusing the call that would build the array.

    ⚠ Not by checking a container field afterwards: the array is what bounds the size of
    system that fits, so what has to be true is that it is never *requested*, and only the
    call itself can say that.
    """
    from kuiva.interface import pyscf_bridge as br

    real_intor = type(hf_mole).intor

    def guarded(self, name, *args, **kwargs):
        if name.startswith("int2e") and kwargs.get("aosym") in ("s8", "s4"):
            raise AssertionError("the direct route asked for a packed ERI array ({})"
                                 .format(kwargs.get("aosym")))
        return real_intor(self, name, *args, **kwargs)

    monkeypatch.setattr(type(hf_mole), "intor", guarded)
    factors = br._direct_cholesky(hf_mole, tol=CD_TOL, report=False)
    assert factors.origin == "cholesky" and factors.orbit_complete


def test_the_direct_route_agrees_with_the_shipped_stored_route_on_the_integrals(hf_conv,
                                                                                hf_mole):
    """⚠ Bitwise against the *8-fold packed* array is unattainable, and not for our reasons.

    PySCF's ``aosym="s8"`` fill and its sliced ``s2ij`` fill traverse the same integrals
    differently and disagree in the last bits for a sixth of the elements, so the two
    decompositions can pick different — equally valid — pivots inside a degenerate orbit.
    What must agree is the object the rest of the program consumes, and it agrees to machine
    precision: ``L`` is a basis of the factorization, not an observable, and any orthogonal
    mixing of its vectors reproduces the same integrals.
    """
    from kuiva.interface.pyscf_bridge import ao_layout

    nao = hf_mole.nao
    layout = ao_layout(hf_mole)
    orbits = shell_pair_orbits(layout.ao_shell, layout.ao_atom)
    src = _direct_source(hf_mole)
    direct = ThreeIndexAO.from_matrix(src.diagonal(), src.column, nao, CD_TOL,
                                      orbits=orbits, report=False)
    stored = ThreeIndexAO.from_eri(hf_conv.eri, nao, CD_TOL, orbits=orbits, report=False)
    assert direct.naux == stored.naux
    g_direct = direct.l_packed.T @ direct.l_packed
    g_stored = stored.l_packed.T @ stored.l_packed
    assert np.max(np.abs(g_direct - g_stored)) < 1e-13
    assert abs(direct.residual - stored.residual) < 1e-15


def test_the_direct_batch_serves_a_whole_orbit_from_one_evaluation(hf_mole):
    """The cost argument: one integral batch serves many columns, and with room to cache them
    the decomposition never evaluates a shell pair twice.

    ⚠ The second half is stated *for a cache that can hold them all*, because the cache is
    bounded and a smaller one re-evaluates — that is the time-for-memory trade the route is
    built on, not a defect. What must hold at any cache size is that the factors come out the
    same, which is the next test.
    """
    from kuiva.interface.pyscf_bridge import DirectERIMatrix, ao_layout

    layout = ao_layout(hf_mole)
    orbits = shell_pair_orbits(layout.ao_shell, layout.ao_atom)
    src = DirectERIMatrix(hf_mole, cache_gb=1.0)          # room for every batch
    ThreeIndexAO.from_matrix(src.diagonal(), src.column, hf_mole.nao, CD_TOL,
                             orbits=orbits, report=False)
    assert src.n_columns > src.n_batches
    assert src.n_batches <= hf_mole.nbas * (hf_mole.nbas + 1) // 2


def test_the_direct_factors_do_not_depend_on_the_batch_cache(hf_mole):
    """⚠ A cache may change what a run costs and may not change what it produces.

    The cache is sized from the transient budget, so the same calculation on two machines —
    or the same machine at two memory limits — takes different numbers of integral batch
    evaluations. If that moved the factors, every result would carry a dependence on the free
    memory of the box it ran on, which is the least reproducible input there is.
    """
    from kuiva.interface.pyscf_bridge import DirectERIMatrix, ao_layout

    layout = ao_layout(hf_mole)
    orbits = shell_pair_orbits(layout.ao_shell, layout.ao_atom)
    runs = []
    for cache_gb in (1e-9, 1e-4, 1.0):                    # none, a few batches, all of them
        src = DirectERIMatrix(hf_mole, cache_gb=cache_gb)
        runs.append((src, ThreeIndexAO.from_matrix(src.diagonal(), src.column, hf_mole.nao,
                                                   CD_TOL, orbits=orbits, report=False)))
    assert runs[0][0].n_batches > runs[-1][0].n_batches   # the cost did change
    for src, factors in runs[1:]:
        assert np.array_equal(factors.l_packed, runs[0][1].l_packed)


def test_the_direct_cache_holds_what_the_plan_says_and_no_more(hf_mole):
    """⚠ The evaluator may not hold more than the memory plan carries for it.

    Sized from the free budget and left unstated, the cache was measured holding gigabytes of
    integral batches under a plan that claimed one batch. A buffer nobody can predict the size
    of is not a transient, it is an unaccounted array — so the plan states
    ``min(what it wants, what it is allowed)`` and this asserts the evaluator obeys it.
    """
    from kuiva.interface import pyscf_bridge as br
    from kuiva.util import resources as res

    src = _direct_source(hf_mole)
    for k in range(hf_mole.nbas):
        for l in range(k + 1):
            src._batch(k, l)
    held = sum(b.nbytes for b in src._cache.values()) / 1024.0 ** 3
    planned = [a for a in br.memory_plan(
        hf_mole.nao, conventional=False, direct=True, shell_ao_max=src.shell_ao_max,
        n_shells=hf_mole.nbas)[1].allocations if "shell-pair batch" in a.label][0]
    assert held <= planned.gb + 1e-12
    assert held <= res.transient_gb() + 1e-12


def test_the_direct_route_carries_its_factors_through_the_front_end(kuiva_caplog):
    """End to end: the container carries factors instead of integrals, and the threshold
    that built them is the front end's, which is reported if a later call disagrees."""
    mol = Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917", basis="x2c-SVPall-2c")
    data = run_scalar_x2c(mol, fitting="cholesky-direct", screening="none",
                          cholesky_tol=1e-6)
    assert data.fit_route == "direct"
    assert data.eri is None and data.df_cderi is None
    assert data.factors is not None and data.factors.tol == 1e-6
    assert ThreeIndexAO.from_scalar_data(data, 1e-6, report=False) is data.factors
    with kuiva_caplog.at_level("WARNING"):
        ThreeIndexAO.from_scalar_data(data, 1e-10, report=False)
    assert any("integral-direct" in r.message for r in kuiva_caplog.records)


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
    f_cd = ThreeIndexAO.from_scalar_data(hf_conv, DEFAULT_CHOLESKY_TOL, report=False,
                                         release_eri=False)
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
    f = ThreeIndexAO.from_scalar_data(hf_conv, report=False, release_eri=False)
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
    f = ThreeIndexAO.from_scalar_data(hf_conv, report=False, release_eri=False)
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


# --- Scratch residence of the factor rows -------------------------------------------------
# The spill is a statement about memory, never about numbers: every test here asserts
# bitwise equality against the in-core path, because the two run the same arithmetic on the
# same rows and anything less than bitwise would mean they do not.


@pytest.fixture
def scratch_limits(monkeypatch, tmp_path):
    """Global limits with a test-owned scratch directory.

    ⚠ A spill in a test must never land in the site scratch, and must not depend on one
    being configured — the scratch directory has no built-in default (the same rule as the
    memory limit), so every spilling test states its own.
    """
    from kuiva.util import resources as res
    monkeypatch.setattr(res.BUDGET, "_limits",
                        res.ResourceLimits(memory_gb=8.0, source="test",
                                           scratch_dir=str(tmp_path)))
    return tmp_path


def _random_factors(nao=14, naux=40, seed=3):
    rng = np.random.default_rng(seed)
    return ThreeIndexAO(l_packed=rng.standard_normal((naux, npair_of(nao))),
                        nao=nao, origin="cholesky")


def test_spilled_factors_unpack_bitwise_including_non_sequential_reads(scratch_limits):
    """Sequential blocks are the optimized walk; a backwards jump (a second consumer
    starting over) must be served correctly through the same reader, just without the
    prefetch hit."""
    keep = _random_factors()
    spilled = ThreeIndexAO(l_packed=keep.l_packed.copy(), nao=keep.nao, origin="cholesky")
    assert spilled.spill_to_scratch() is spilled and spilled.is_spilled
    assert spilled.spill_to_scratch() is spilled           # idempotent
    for sl in [slice(0, 13), slice(13, 26), slice(26, 40),  # the sequential walk
               slice(5, 20), slice(0, 40), slice(39, 40)]:  # jumps and edges
        np.testing.assert_array_equal(keep.unpack(sl), spilled.unpack(sl))
    with pytest.raises(ValueError):
        spilled.unpack(slice(0, 10, 2))                    # strided reads are refused


def test_spilled_factors_transform_and_jk_bitwise(scratch_limits):
    rng = np.random.default_rng(4)
    keep = _random_factors()
    spilled = ThreeIndexAO(l_packed=keep.l_packed.copy(), nao=keep.nao, origin="cholesky")
    spilled.spill_to_scratch()
    c = rng.standard_normal((2 * keep.nao, 6)) + 1j * rng.standard_normal((2 * keep.nao, 6))
    c = np.ascontiguousarray(c)
    np.testing.assert_array_equal(transform_3c(keep, c, c, aux_blocksize=7),
                                  transform_3c(spilled, c, c, aux_blocksize=7))
    from kuiva.integrals.transform import coulomb_exchange
    occ = np.ascontiguousarray(c[:, :3])
    j1, k1 = coulomb_exchange(keep, occ)
    j2, k2 = coulomb_exchange(spilled, occ)
    np.testing.assert_array_equal(j1, j2)
    np.testing.assert_array_equal(k1, k2)


def test_spill_releases_the_ram_reservation_and_deletes_its_file_with_the_object(
        scratch_limits):
    """The two lifetime halves of the design: the ledger stops carrying rows that are no
    longer in RAM, and the scratch file cannot outlive the object that owns it."""
    import gc
    import os as _os
    from kuiva.util import resources as res

    base = res.BUDGET.resident_gb()
    f = _random_factors()
    assert res.BUDGET.resident_gb() > base                 # the in-core reservation
    f.spill_to_scratch()
    assert res.BUDGET.resident_gb() == pytest.approx(base)  # given back on spill
    path = f._store.path
    assert _os.path.exists(path)
    del f
    gc.collect()
    assert not _os.path.exists(path)


def test_spilled_metadata_and_stream_accounting(scratch_limits):
    f = _random_factors()
    naux, npair, gb = f.naux, f.npair, f.memory_gb
    assert f.stream_row_bytes == 0.0
    f.spill_to_scratch()
    assert (f.naux, f.npair) == (naux, npair)
    assert f.memory_gb == pytest.approx(gb)                # the data size, wherever it lives
    assert f.stream_row_bytes == pytest.approx(2.0 * npair * 8.0)
    assert f.l_packed is None


def test_the_stored_array_is_released_once_the_factors_exist():
    """⚠ The mechanism, not only the observable: the array goes *and* its ledger entry goes.

    Half a release is worse than none — a dropped array whose reservation stays behind is a
    budget that refuses calculations the machine would have run, and an array kept under a
    released reservation is the same lie in the other direction. Both halves are asserted
    here, together with the one thing that must not change: the factors themselves.
    """
    from kuiva.util import resources as res

    mol = Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917", basis="x2c-SVPall-2c")
    data = run_scalar_x2c(mol, fitting="conventional", screening="none")
    eri_gb = float(np.asarray(data.eri).nbytes) / 1024.0 ** 3
    held = res.BUDGET.resident_gb()
    assert any("ERI" in a.label for a in res.BUDGET._resident)

    kept = ThreeIndexAO.from_scalar_data(data, CD_TOL, report=False, release_eri=False)
    assert data.eri is not None and not data.eri_released

    obj = ThreeIndexAO.from_scalar_data(data, CD_TOL, report=False)
    assert data.eri is None and data.eri_released
    assert not any("ERI" in a.label for a in res.BUDGET._resident)
    # The ledger gave back exactly the array, and nothing else: both factorizations are
    # still live and still reserved, so the difference is the ERI line alone.
    assert res.BUDGET.resident_gb() == pytest.approx(
        held - eri_gb + kept.memory_gb + obj.memory_gb, abs=1e-9)
    np.testing.assert_array_equal(kept.l_packed, obj.l_packed)
    assert data.release_eri() == 0.0                      # idempotent


def test_a_second_factorization_after_the_release_refuses_with_the_knob():
    """⚠ A refusal that teaches, not a container that merely looks empty: the message has to
    name ``release_eri=False``, because "factorize this ingested data twice" is a real thing
    to want (a threshold series) and the container cannot tell the user that by itself."""
    mol = Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917", basis="x2c-SVPall-2c")
    data = run_scalar_x2c(mol, fitting="conventional", screening="none")
    ThreeIndexAO.from_scalar_data(data, CD_TOL, report=False)
    with pytest.raises(ValueError, match="release_eri=False"):
        ThreeIndexAO.from_scalar_data(data, 1e-6, report=False)
    # And a container that never had the array still gets the generic complaint, so the two
    # states stay distinguishable.
    import dataclasses
    empty = dataclasses.replace(data, eri_released=False)
    with pytest.raises(ValueError, match="neither conventional ERIs nor DF"):
        ThreeIndexAO.from_scalar_data(empty, CD_TOL, report=False)


def test_front_end_scratch_residence_is_recorded_and_numerically_inert(scratch_limits):
    """The whole axis end to end: an explicit factors="scratch" run must carry the record,
    hand back spilled factors, and change nothing numerical. ⚠ Bitwise identity of the spill
    itself (write the rows, read them back) is asserted in the unit tests above; *across two
    separate runs* the last bits belong to threaded-reduction nondeterminism, not to the
    residence, so here the comparison is a tight tolerance on converged quantities."""
    mol = Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917", basis="x2c-SVPall-2c")
    core = run_scalar_x2c(mol, fitting="cholesky-direct", screening="none")
    spill = run_scalar_x2c(mol, fitting="cholesky-direct", screening="none",
                           factors="scratch")
    assert core.factor_residence == "in-core" and not core.factors.is_spilled
    assert spill.factor_residence == "scratch" and spill.factors.is_spilled
    assert spill.e_scf == pytest.approx(core.e_scf, abs=1e-9)
    full = slice(0, core.factors.naux)
    np.testing.assert_allclose(spill.factors.unpack(full), core.factors.unpack(full),
                               rtol=0.0, atol=1e-12)
    # And the stored route honours the same record downstream, where it decomposes.
    # Compared against an in-core factorization of the *same* ingested data — the stored
    # and direct routes are documented to differ in the last bits of the integrals
    # themselves, so cross-route factor comparison would measure that, not the spill.
    conv = run_scalar_x2c(mol, fitting="conventional", screening="none",
                          factors="scratch")
    obj = ThreeIndexAO.from_scalar_data(conv, CD_TOL, report=False, release_eri=False)
    assert obj.is_spilled
    object.__setattr__(conv, "factor_residence", "in-core")
    ref = ThreeIndexAO.from_scalar_data(conv, CD_TOL, report=False)
    assert not ref.is_spilled
    np.testing.assert_array_equal(obj.unpack(slice(0, obj.naux)),
                                  ref.unpack(slice(0, ref.naux)))


# --- The out-of-core decomposition --------------------------------------------------------

def _reconstruct(f):
    """The two-electron integrals a factorization stands for, as a dense array."""
    blk = f.unpack(slice(0, f.naux))
    return np.einsum("pij,pkl->ijkl", blk, blk)


def test_the_streamed_decomposition_reproduces_the_in_core_factorization(scratch_limits):
    """⚠ The claim is about the **integrals**, not about the factor rows, and the difference
    matters. The orbit path emits an eigenbasis of each orbit's Gram matrix, so two runs that
    take equal-diagonal orbits in a different order produce different vectors spanning the
    same space — an O(1) element-wise difference in ``L`` and none at all in ``(pq|rs)``.
    What is asserted is therefore the vector *count*, the residual bound, and the
    reconstructed integrals; the same discipline that forbids anything downstream from
    depending on a rotation inside a degenerate block.

    The tight budget is the point of the test: it forces several update passes, which is the
    whole out-of-core mechanism (one pass would just be the in-core algorithm with a file).
    """
    mol = Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917", basis="x2c-SVPall-2c")
    data = run_scalar_x2c(mol, fitting="conventional", screening="none")
    orbits = shell_pair_orbits(data.ao_layout.ao_shell, data.ao_layout.ao_atom)
    core = ThreeIndexAO.from_eri(data.eri, data.nao, CD_TOL, orbits=orbits, report=False)
    exact = _reconstruct(core)

    from kuiva.integrals.transform import streamed_cholesky

    n = npair_of(data.nao)
    seen_passes = []
    for budget in (4e-4, 1e-3, 1e-2):
        store, _piv, residual, stats = streamed_cholesky(
            _s8_diagonal(data.eri, n), lambda q: _s8_column(data.eri, n, q), data.nao,
            CD_TOL, orbits=orbits, budget_gb=budget)
        obj = ThreeIndexAO(l_packed=None, nao=data.nao, origin="cholesky", tol=CD_TOL,
                           residual=residual, orbit_complete=True, _store=store,
                           stream_stats=stats)
        assert obj.naux == core.naux                       # the same pivot sequence
        assert obj.residual <= CD_TOL
        np.testing.assert_allclose(_reconstruct(obj), exact, rtol=0.0, atol=1e-13)
        seen_passes.append(stats["passes"])
    # Budget-independent numbers, budget-dependent IO: that is what an out-of-core route is.
    assert max(seen_passes) > min(seen_passes) and max(seen_passes) > 1


def test_a_tight_budget_forces_several_passes_and_re_reads_no_column_twice(scratch_limits):
    """The retention rule, which is what keeps a narrow budget from becoming quadratic.

    A pass ends the moment the true argmax leaves its candidate set — that is what keeps the
    pivot order exactly the in-core one — so without retaining the candidates it did not
    consume, a pass that ended early would throw away every column it had just evaluated. On
    the integral-direct route those are integrals, re-evaluated from scratch.
    """
    from kuiva.integrals.transform import streamed_cholesky

    mol = Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917", basis="x2c-SVPall-2c")
    data = run_scalar_x2c(mol, fitting="conventional", screening="none")
    n = npair_of(data.nao)
    orbits = shell_pair_orbits(data.ao_layout.ao_shell, data.ao_layout.ao_atom)
    store, _piv, residual, stats = streamed_cholesky(
        _s8_diagonal(data.eri, n), lambda q: _s8_column(data.eri, n, q), data.nao, CD_TOL,
        orbits=orbits, budget_gb=4e-4)
    assert stats["passes"] > 1 and stats["gb_read"] > 0.0
    # Every column is evaluated at least once; retention holds the re-evaluations to a small
    # multiple rather than one set per pass.
    assert n <= stats["columns"] < n + stats["passes"] * stats["candidate_columns"]
    assert residual <= CD_TOL
    store.close()


def test_a_streamed_factorization_is_a_spilled_store_that_never_had_a_reservation(
        scratch_limits):
    """⚠ The ledger half: the in-core route reserves the factor array and the spill gives it
    back, so a factorization that never allocated it must never have reserved it either — an
    unbalanced ledger refuses calculations the machine would have run."""
    import gc
    import os as _os
    from kuiva.util import resources as res

    mol = Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917", basis="x2c-SVPall-2c")
    data = run_scalar_x2c(mol, fitting="conventional", screening="none")
    base = res.BUDGET.resident_gb()
    obj = ThreeIndexAO.from_eri(data.eri, data.nao, CD_TOL, report=False,
                                residence="streamed")
    assert obj.is_spilled and obj.l_packed is None and obj._reservation is None
    assert res.BUDGET.resident_gb() == pytest.approx(base)
    assert obj.memory_gb > 0.0 and obj.stream_stats is not None
    path = obj._store.path
    assert _os.path.exists(path)
    del obj
    gc.collect()
    assert not _os.path.exists(path)           # the file cannot outlive the factors


def test_the_streamed_decomposition_refuses_rather_than_split_an_orbit(monkeypatch, tmp_path):
    """⚠ The whole-orbit rule at the one place a *budget* could break it: an orbit is the
    unit the block path is invariant in, so a working set that cannot hold the widest one is
    refused — never quietly given a partial orbit, which would look like ordinary blocking
    and would silently return the threshold-dependent symmetry the orbit path exists to
    remove."""
    from kuiva.integrals.transform import streamed_cholesky
    from kuiva.util import resources as res

    mol = Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917", basis="x2c-SVPall-2c")
    data = run_scalar_x2c(mol, fitting="conventional", screening="none")
    n = npair_of(data.nao)
    orbits = shell_pair_orbits(data.ao_layout.ao_shell, data.ao_layout.ao_atom)
    monkeypatch.setattr(res.BUDGET, "_limits",
                        res.ResourceLimits(memory_gb=1e-5, source="test",
                                           scratch_dir=str(tmp_path)))
    with pytest.raises(res.MemoryLimitError, match="streamed Cholesky working set"):
        streamed_cholesky(_s8_diagonal(data.eri, n),
                          lambda q: _s8_column(data.eri, n, q), data.nao, CD_TOL,
                          orbits=orbits, budget_gb=1e-6)


def test_front_end_streamed_residence_is_recorded_and_numerically_inert(scratch_limits):
    """The axis end to end on the route that will use it: ``fitting="cholesky-direct"``
    never forms the integral array and ``factors="streamed"`` never forms the factor array,
    so the whole front end runs without either — and produces the same integrals."""
    mol = Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917", basis="x2c-SVPall-2c")
    core = run_scalar_x2c(mol, fitting="cholesky-direct", screening="none")
    flow = run_scalar_x2c(mol, fitting="cholesky-direct", screening="none",
                          factors="streamed")
    assert flow.factor_residence == "streamed" and flow.eri is None
    assert flow.factors.is_spilled and flow.factors.l_packed is None
    assert flow.factors.naux == core.factors.naux
    np.testing.assert_allclose(_reconstruct(flow.factors), _reconstruct(core.factors),
                               rtol=0.0, atol=1e-13)
    # And the stored route honours the same record where it decomposes, downstream.
    conv = run_scalar_x2c(mol, fitting="conventional", screening="none",
                          factors="streamed")
    obj = ThreeIndexAO.from_scalar_data(conv, CD_TOL, report=False)
    assert obj.is_spilled and obj.stream_stats is not None and conv.eri is None


def test_an_explicit_streamed_request_without_a_scratch_directory_refuses_early(monkeypatch):
    """Same rule as the spill: no scratch directory is a configuration error, raised before
    the SCF is paid for rather than after the integrals have been evaluated."""
    from kuiva.util import resources as res
    monkeypatch.setattr(res.BUDGET, "_limits",
                        res.ResourceLimits(memory_gb=8.0, source="test"))
    mol = Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917", basis="x2c-SVPall-2c")
    with pytest.raises(res.ConfigurationError, match="scratch"):
        run_scalar_x2c(mol, fitting="cholesky-direct", screening="none", factors="streamed")


def test_an_explicit_scratch_request_without_a_scratch_directory_refuses_early(monkeypatch):
    """⚠ The no-built-in-default rule, at the front end: factors="scratch" with no scratch
    directory configured refuses before the SCF is paid for, and the refusal teaches the
    knob. ("auto" never resolves to scratch in that state — tested with the plan.)"""
    from kuiva.util import resources as res
    monkeypatch.setattr(res.BUDGET, "_limits",
                        res.ResourceLimits(memory_gb=8.0, source="test"))
    mol = Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917", basis="x2c-SVPall-2c")
    with pytest.raises(res.ConfigurationError, match="scratch"):
        run_scalar_x2c(mol, fitting="cholesky-direct", screening="none", factors="scratch")
