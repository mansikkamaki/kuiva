"""Tier-0/1 tests for the determinant machinery.

The whole cheap CI, and eventually the full CI, rests on the Slater-Condon phases in
``ci/strings.py``. A sign error there is invisible in norms, plausible in magnitude, and
poisons every RDM downstream. So the strategy is deliberately redundant:

* against a **brute-force** implementation that applies second-quantized operators one at a
  time to explicit occupation lists — slow, obviously correct, and completely independent of
  the bit arithmetic being tested;
* against **PySCF's FCI** for a real spin-free system, where the spinor CI must reproduce the
  spatial answer exactly (this catches convention errors the brute force shares);
* against **exact identities**: hermiticity, ``Tr gamma = N``, and
  ``sum_r Gamma_pqrr = (N-1) gamma_pq``.
"""
import itertools

import numpy as np
import pytest

from kuiva.ci.strings import (Determinants, connections, diagonal_energies,
                              hamiltonian_matrix, occupation_correlations, rdm12,
                              single_excitation_operator)

ALG_TOL = 1e-10


# --- helpers -------------------------------------------------------------------------------
def random_spinor_integrals(n, seed=0, scale=1.0):
    """Hermitian ``h`` and an ERI with the 4-fold symmetry of complex spinor integrals."""
    r = np.random.default_rng(seed)
    h = r.standard_normal((n, n)) + 1j * r.standard_normal((n, n))
    h = 0.5 * (h + h.conj().T)
    ell = r.standard_normal((3 * n, n, n))
    ell = 0.5 * (ell + ell.transpose(0, 2, 1))                # real symmetric AO factors
    c = r.standard_normal((n, n)) + 1j * r.standard_normal((n, n))
    q, _ = np.linalg.qr(c)
    b = np.einsum("mp,Pmn,nq->Ppq", q.conj(), ell, q)
    return h, scale * np.tensordot(b, b, axes=([0], [0]))


def _apply(occ, ops):
    """Apply ``[('a'|'c', p), ...]`` right-to-left order as written; return (sign, occ)."""
    occ = list(occ)
    sign = 1
    for kind, p in ops:
        if kind == "a":
            if p not in occ:
                return None
            sign *= (-1) ** sorted(occ).index(p)
            occ.remove(p)
        else:
            if p in occ:
                return None
            sign *= (-1) ** sum(1 for q in occ if q < p)
            occ.append(p)
    return sign, tuple(sorted(occ))


def brute_hamiltonian(occ_lists, n, h, eri):
    """H by explicit operator application — independent of every bit trick under test."""
    idx = {d: i for i, d in enumerate(occ_lists)}
    mat = np.zeros((len(occ_lists),) * 2, dtype=complex)
    for j, dj in enumerate(occ_lists):
        for p, q in itertools.product(range(n), repeat=2):
            res = _apply(dj, [("a", q), ("c", p)])
            if res and res[1] in idx:
                mat[idx[res[1]], j] += res[0] * h[p, q]
        for p, q, r, s in itertools.product(range(n), repeat=4):
            res = _apply(dj, [("a", q), ("a", s), ("c", r), ("c", p)])
            if res and res[1] in idx:
                mat[idx[res[1]], j] += 0.5 * res[0] * eri[p, q, r, s]
    return mat


def brute_rdms(occ_lists, n, vec):
    idx = {d: i for i, d in enumerate(occ_lists)}
    g1 = np.zeros((n, n), dtype=complex)
    g2 = np.zeros((n,) * 4, dtype=complex)
    for j, dj in enumerate(occ_lists):
        for p, q in itertools.product(range(n), repeat=2):
            res = _apply(dj, [("a", q), ("c", p)])
            if res and res[1] in idx:
                g1[p, q] += np.conj(vec[idx[res[1]]]) * vec[j] * res[0]
        for p, q, r, s in itertools.product(range(n), repeat=4):
            res = _apply(dj, [("a", q), ("a", s), ("c", r), ("c", p)])
            if res and res[1] in idx:
                g2[p, q, r, s] += np.conj(vec[idx[res[1]]]) * vec[j] * res[0]
    return g1, g2


def full_space(n, nelec):
    occ = list(itertools.combinations(range(n), nelec))
    return occ, Determinants.from_occupations(occ, n)


# --- representation --------------------------------------------------------------------------
def test_determinant_bookkeeping():
    occ, dets = full_space(6, 3)
    assert dets.ndet == 20 and dets.n_elec == 3
    assert dets.occupations().sum(axis=1).tolist() == [3] * 20
    assert dets.position(dets.masks[7]) == 7
    assert dets.position(np.uint64(0)) == -1


def test_mixed_particle_number_rejected():
    with pytest.raises(ValueError, match="mixes particle numbers"):
        Determinants.from_occupations([(0, 1), (0, 1, 2)], 4)


def test_duplicates_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        Determinants.from_occupations([(0, 1), (0, 1)], 4)


def test_spinor_count_limit():
    with pytest.raises(ValueError, match="multi-word"):
        Determinants(masks=np.zeros(0, dtype=np.uint64), n_spinor=65, n_elec=0)


def test_connections_find_every_interacting_pair():
    """Exhaustive check that the blocked XOR search misses nothing."""
    occ, dets = full_space(7, 3)
    conn = connections(dets, block=3)
    found = set(zip(conn.single_i.tolist(), conn.single_j.tolist()))
    found |= set(zip(conn.double_i.tolist(), conn.double_j.tolist()))
    expected = set()
    for i in range(dets.ndet):
        for j in range(i + 1, dets.ndet):
            rank = len(set(occ[i]) - set(occ[j]))
            if rank in (1, 2):
                expected.add((i, j))
    assert found == expected


# --- Slater-Condon ------------------------------------------------------------------------------
def test_hamiltonian_matches_brute_force():
    n, nelec = 6, 3
    h, eri = random_spinor_integrals(n, seed=1)
    occ, dets = full_space(n, nelec)
    got = hamiltonian_matrix(dets, h, eri).toarray()
    assert np.max(np.abs(got - brute_hamiltonian(occ, n, h, eri))) < 1e-10


def test_hamiltonian_is_hermitian():
    n, nelec = 7, 4
    h, eri = random_spinor_integrals(n, seed=2)
    _, dets = full_space(n, nelec)
    mat = hamiltonian_matrix(dets, h, eri).toarray()
    assert np.max(np.abs(mat - mat.conj().T)) < ALG_TOL


def test_diagonal_energies_match_the_matrix():
    n, nelec = 6, 3
    h, eri = random_spinor_integrals(n, seed=3)
    _, dets = full_space(n, nelec)
    mat = hamiltonian_matrix(dets, h, eri).toarray()
    assert np.max(np.abs(diagonal_energies(dets, h, eri) - np.diag(mat).real)) < ALG_TOL


@pytest.mark.parametrize("seed", [4, 5])
def test_rdms_match_brute_force(seed):
    n, nelec = 6, 3
    h, eri = random_spinor_integrals(n, seed=seed)
    occ, dets = full_space(n, nelec)
    _, v = np.linalg.eigh(hamiltonian_matrix(dets, h, eri).toarray())
    vec = v[:, 0]
    g1, g2 = rdm12(dets, vec)
    b1, b2 = brute_rdms(occ, n, vec)
    assert np.max(np.abs(g1 - b1)) < ALG_TOL
    assert np.max(np.abs(g2 - b2)) < ALG_TOL


def test_energy_from_rdms_equals_the_eigenvalue():
    """The sharpest single check: it ties the Hamiltonian and both RDMs to one number, and it
    holds only if every phase is consistent between them."""
    n, nelec = 7, 3
    h, eri = random_spinor_integrals(n, seed=6)
    _, dets = full_space(n, nelec)
    w, v = np.linalg.eigh(hamiltonian_matrix(dets, h, eri).toarray())
    g1, g2 = rdm12(dets, v[:, 0])
    e = np.einsum("pq,pq->", h, g1) + 0.5 * np.einsum("pqrs,pqrs->", eri, g2)
    assert abs(e.imag) < ALG_TOL
    assert abs(e.real - w[0]) < 1e-9


def test_rdm_exact_identities():
    n, nelec = 7, 4
    h, eri = random_spinor_integrals(n, seed=7)
    _, dets = full_space(n, nelec)
    _, v = np.linalg.eigh(hamiltonian_matrix(dets, h, eri).toarray())
    g1, g2 = rdm12(dets, v[:, 0])
    assert abs(np.trace(g1).real - nelec) < 1e-10
    assert np.max(np.abs(g1 - g1.conj().T)) < ALG_TOL
    # sum_r Gamma_pqrr = (N-1) gamma_pq
    assert np.max(np.abs(np.einsum("pqrr->pq", g2) - (nelec - 1) * g1)) < 1e-10
    # Gamma_pqrs = Gamma_rspq and Gamma_pqrs* = Gamma_qpsr
    assert np.max(np.abs(g2 - g2.transpose(2, 3, 0, 1))) < ALG_TOL
    assert np.max(np.abs(g2 - g2.conj().transpose(1, 0, 3, 2))) < ALG_TOL
    # antisymmetry under exchanging the two annihilators
    assert np.max(np.abs(g2 + g2.transpose(0, 3, 2, 1))) < ALG_TOL


def test_state_averaged_rdm_is_the_weighted_mean():
    n, nelec = 6, 3
    h, eri = random_spinor_integrals(n, seed=8)
    _, dets = full_space(n, nelec)
    _, v = np.linalg.eigh(hamiltonian_matrix(dets, h, eri).toarray())
    w = np.array([0.7, 0.3])
    g_avg, _ = rdm12(dets, v[:, :2], w, with_2rdm=False)
    g0, _ = rdm12(dets, v[:, 0], with_2rdm=False)
    g1, _ = rdm12(dets, v[:, 1], with_2rdm=False)
    assert np.max(np.abs(g_avg - (0.7 * g0 + 0.3 * g1))) < ALG_TOL


def test_occupation_correlations_agree_with_the_2rdm():
    n, nelec = 6, 3
    h, eri = random_spinor_integrals(n, seed=9)
    _, dets = full_space(n, nelec)
    _, v = np.linalg.eigh(hamiltonian_matrix(dets, h, eri).toarray())
    _, g2 = rdm12(dets, v[:, 0])
    nn = occupation_correlations(dets, v[:, 0])
    off = ~np.eye(n, dtype=bool)
    assert np.max(np.abs(np.einsum("ppqq->pq", g2)[off] - nn[off])) < ALG_TOL


def test_single_excitation_operator_reproduces_the_rdm():
    n, nelec = 6, 3
    h, eri = random_spinor_integrals(n, seed=10)
    _, dets = full_space(n, nelec)
    _, v = np.linalg.eigh(hamiltonian_matrix(dets, h, eri).toarray())
    vec = v[:, 0].astype(complex)
    g1, _ = rdm12(dets, vec, with_2rdm=False)
    for p, q in [(0, 0), (1, 3), (4, 2)]:
        op = single_excitation_operator(dets, p, q)
        assert abs(np.vdot(vec, op @ vec) - g1[p, q]) < 1e-10


# --- Tier 1: against PySCF's FCI -------------------------------------------------------------------
def test_spinor_fci_reproduces_pyscf_fci():
    """A spin-free Hamiltonian in a Kramers-paired spinor basis must give exactly the spatial
    answer. This is the 'same problem in a doubled basis' check at the CI level, and it
    catches convention errors that the brute-force reference shares with the code under test.
    """
    from pyscf import fci, gto, scf, ao2mo

    mol = gto.M(atom="Li 0 0 0; H 0 0 1.60", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    ncas, nelecas = 4, 4
    ncore = (mol.nelectron - nelecas) // 2
    mo = mf.mo_coeff[:, ncore:ncore + ncas]
    h_core = mf.get_hcore()
    # frozen-core effective one-electron Hamiltonian
    mo_core = mf.mo_coeff[:, :ncore]
    dm_core = 2 * mo_core @ mo_core.T
    vj, vk = mf.get_jk(mol, dm_core)
    h_eff = mo.T @ (h_core + vj - 0.5 * vk) @ mo
    e_core = np.einsum("ij,ji->", h_core + 0.5 * (vj - 0.5 * vk), dm_core) + mol.energy_nuc()
    eri_cas = ao2mo.restore(1, ao2mo.kernel(mol, mo), ncas)
    e_ref = fci.direct_spin1.kernel(h_eff, eri_cas, ncas, nelecas)[0] + e_core

    # Kramers expansion: spinor 2p / 2p+1 carry the same spatial orbital
    n = 2 * ncas
    h_s = np.zeros((n, n), dtype=complex)
    eri_s = np.zeros((n,) * 4, dtype=complex)
    for p in range(ncas):
        for q in range(ncas):
            for s in (0, 1):
                h_s[2 * p + s, 2 * q + s] = h_eff[p, q]
    for p, q, r, s in itertools.product(range(ncas), repeat=4):
        for s1 in (0, 1):
            for s2 in (0, 1):
                eri_s[2 * p + s1, 2 * q + s1, 2 * r + s2, 2 * s + s2] = eri_cas[p, q, r, s]

    occ, dets = full_space(n, nelecas)
    from scipy.sparse.linalg import eigsh
    e0 = eigsh(hamiltonian_matrix(dets, h_s, eri_s), k=1, which="SA")[0][0]
    assert abs(e0 + e_core - e_ref) < 1e-9


# --- ladder operators between adjacent determinant spaces (the Gram route to ladder operators) -----------------

def test_ladder_map_matches_dense_ladder_operators():
    """⚠ Validated against an independent construction, because every sign here is silent.

    ``ladder_map`` is the only place ``a_p`` and ``a_p^dag`` are written as a map *between*
    determinant spaces, and :mod:`kuiva.pt.contractions` builds its whole Gram route on it: a
    wrong parity would produce Hermitian, positive, correctly-traced and wrong overlap
    matrices. ``tests/fockspace.py`` builds the same operators as dense matrices over the full
    Fock space from the occupation patterns directly and shares no code with this one.

    Run for **every mode and every electron count**, both directions, because the parity
    depends on how many modes lie below ``p`` and a rule that is right for ``p = 0`` says
    nothing about ``p = 3``.
    """
    from fockspace import FockSpace

    from kuiva.ci.strings import CASSpace, apply_ladder

    n = 6
    fock = FockSpace(n)
    rng = np.random.default_rng(0)
    for k in range(1, n):
        space = CASSpace(n, k, build_map=False)
        vector = rng.normal(size=space.ndet) + 1j * rng.normal(size=space.ndet)
        embedded = np.zeros(fock.dim, dtype=complex)
        embedded[space.masks.astype(int)] = vector
        for mode in range(n):
            if k - 1 >= 1:
                lower = CASSpace(n, k - 1, build_map=False)
                expected = (fock.create(mode).T @ embedded)[lower.masks.astype(int)]
                assert np.allclose(apply_ladder(space.masks, lower.masks, mode, vector),
                                   expected), ("a_{}".format(mode), k)
            if k + 1 <= n:
                upper = CASSpace(n, k + 1, build_map=False)
                expected = (fock.create(mode) @ embedded)[upper.masks.astype(int)]
                assert np.allclose(apply_ladder(space.masks, upper.masks, mode, vector,
                                                dagger=True), expected), ("a+", mode, k)


def test_the_vacuum_is_a_legal_target_space():
    """A two-electron active space annihilates to it, so it is reachable from ``CAS(2, n)``.

    :class:`~kuiva.ci.strings.CASSpace` cannot represent zero electrons, so the caller passes
    the mask array directly — which is why :func:`~kuiva.ci.strings.ladder_map` takes masks
    rather than spaces.
    """
    from fockspace import FockSpace

    from kuiva.ci.strings import CASSpace, apply_ladder

    n = 5
    fock = FockSpace(n)
    space = CASSpace(n, 1, build_map=False)
    rng = np.random.default_rng(3)
    vector = rng.normal(size=space.ndet) + 1j * rng.normal(size=space.ndet)
    embedded = np.zeros(fock.dim, dtype=complex)
    embedded[space.masks.astype(int)] = vector
    vacuum = np.zeros(1, dtype=np.uint64)
    for mode in range(n):
        got = apply_ladder(space.masks, vacuum, mode, vector)
        assert abs(got[0] - (fock.create(mode).T @ embedded)[0]) < 1e-14


def test_a_ladder_between_non_adjacent_spaces_is_refused():
    """⚠ Refused, not silently empty: dropping the terms would give a plausible small answer."""
    from kuiva.ci.strings import CASSpace, ladder_map

    n = 5
    with pytest.raises(ValueError, match="maps out of the target"):
        ladder_map(CASSpace(n, 3, build_map=False).masks,
                   CASSpace(n, 1, build_map=False).masks, 0)
