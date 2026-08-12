"""Stage 1: the Jordan-Wigner mapping against the CI Hamiltonian .

Why this file has to be airtight before anything else in :mod:`kuiva.qc` is trusted
------------------------------------------------------------------------------------
A sign or ordering error in the fermion-to-qubit mapping produces a Hermitian operator of
plausible magnitude with a plausible spectrum, and every structural check — Hermiticity,
particle-number conservation, trace conditions — passes on the wrong one. That is the worst
description of the defects that cost the most, and it is why the reference here is
``ci/strings.hamiltonian_matrix``: a **genuinely independent implementation** of the same
operator, by a different algorithm (an ``O(N^2)`` XOR/popcount search over determinant pairs
plus the Slater-Condon rules) with its own phase bookkeeping. The two sides share no code, so
the comparison **can fail**, which is the measure of a check's worth.

The comparison is exact, with no tolerance to argue about: both sides sum the same products of
the same integrals, so anything above rounding is a defect and not a method difference. The
tolerance below is a floating-point statement, nothing more.

⚠ **The integrals are complex on purpose.** Every published quantum-chemistry mapping is
validated on real, spin-separable, non-relativistic integrals; Kuiva's are none of those
, and the imaginary parts of ``h_pq`` and ``(pq|rs)`` are precisely what generate the
``XY - YX``-type Pauli terms that a real-integral test never exercises.
"""
import numpy as np
import pytest

from kuiva.ci.strings import CASSpace, hamiltonian_matrix, popcount
from kuiva.qc.mapping import (HERMITICITY_TOL, MAX_QUBITS, PauliSum, available_mappings,
                              dense_matrix_gb, dense_matrix_workspace_gb, jordan_wigner,
                              fermionic_operator, jordan_wigner_terms, pauli_apply,
                              pauli_commute, pauli_expectation,
                              pauli_label, pauli_terms_gb, qubit_hamiltonian, qwc_groups,
                              rdm_measurement,
                              register_mapping, resolve_mapping)
from test_ci_strings import random_spinor_integrals

#: Machine precision. See the module docstring for why nothing looser would be meaningful.
EXACT = 1e-11

#: Active spaces small enough for the ``2^n`` dense matrix, large enough that the JW parity
#: string is non-trivial for most modes. Six spinors is already 562 Pauli terms.
SPACES = (2, 3, 4, 5, 6)


def _dense_reference(n, k, h, eri):
    """The CI Hamiltonian over the ``k``-electron sector, and the masks it is indexed by."""
    dets = CASSpace(n, k).determinants()
    return hamiltonian_matrix(dets, h, eri).toarray(), dets.masks.astype(np.int64)


# --- the load-bearing comparison ----------------------------------------------------------

@pytest.mark.parametrize("n", SPACES)
def test_jordan_wigner_reproduces_the_ci_hamiltonian_in_every_sector(n):
    """The Stage-1 exit criterion, for every electron number at once.

    The determinant mask **is** the computational-basis index (the determinant convention meets Jordan-Wigner), so
    the CI Hamiltonian of the ``k``-electron sector is a plain submatrix of the qubit
    Hamiltonian — no permutation, no phase convention to reconcile. If that correspondence
    were even slightly off, this comparison would fail rather than merely disagree in the
    last digits.
    """
    h, eri = random_spinor_integrals(n, seed=100 + n)
    dense = jordan_wigner(h, eri).to_dense()
    for k in range(1, n + 1):
        ref, masks = _dense_reference(n, k, h, eri)
        assert np.abs(dense[np.ix_(masks, masks)] - ref).max() < EXACT, (n, k)


@pytest.mark.parametrize("n", SPACES)
def test_the_mapped_hamiltonian_conserves_particle_number(n):
    """Every matrix element between sectors of different ``N`` is exactly zero.

    Not implied by the test above, which only ever looks *inside* a sector: a mapping that
    dropped a parity string would leak between them and still reproduce each block. The
    fermionic operator conserves ``N`` by construction, so this is a structural property of
    the mapping and gets machine precision rather than a tolerance.
    """
    h, eri = random_spinor_integrals(n, seed=200 + n)
    dense = jordan_wigner(h, eri).to_dense()
    number = popcount(np.arange(1 << n, dtype=np.uint64))
    off_sector = number[:, None] != number[None, :]
    assert np.abs(dense[off_sector]).max() < EXACT


def test_the_parity_string_is_load_bearing():
    """⚠ A guard that cannot fail proves nothing (the guard rule, applied here).

    Drop the Jordan-Wigner parity string — keep every other part of the mapping — and the
    comparison above must break. Without this, "the mapping agrees" would be consistent with
    a test insensitive to the one piece of bookkeeping that is genuinely easy to get wrong.
    """
    from kuiva.qc import mapping as m

    n = 5
    h, eri = random_spinor_integrals(n, seed=7)
    ref, masks = _dense_reference(n, 3, h, eri)

    def _no_parity(modes, dagger):
        """:func:`_ladder_terms` with ``below_mask`` replaced by zero."""
        p = np.asarray(modes, dtype=np.int64)
        bit = m.mode_bit(p)
        zero = np.zeros_like(bit)
        x = np.stack([bit, bit], axis=1)
        z = np.stack([zero, zero | bit], axis=1)
        second = 0.5 if dagger else -0.5
        c = np.stack([np.full(p.shape, 0.5), np.full(p.shape, second)],
                     axis=1).astype(np.complex128)
        return c, x, z

    original = m._ladder_terms
    try:
        m._ladder_terms = _no_parity
        broken = m.jordan_wigner(h, eri).to_dense()
    finally:
        m._ladder_terms = original
    assert np.abs(broken[np.ix_(masks, masks)] - ref).max() > 1e-3


def test_a_real_hamiltonian_is_the_easy_case_and_the_complex_one_is_not():
    """The imaginary parts are what generate the terms a real-integral test never sees.

    Recorded as a measurement rather than an assertion of principle: with SOC-like complex
    integrals the mapping produces strictly more Pauli strings than the real Hamiltonian of
    the same size, and those extra strings are the ``XY``/``YX``-type ones the
    literature never exercises.
    """
    n = 4
    h, eri = random_spinor_integrals(n, seed=11)
    complex_terms = jordan_wigner(h, eri).n_terms
    real_terms = jordan_wigner(h.real.astype(np.complex128),
                               eri.real.astype(np.complex128)).n_terms
    assert complex_terms > real_terms


# --- structure of the representation -------------------------------------------------------

def test_the_coefficients_are_real_and_the_operator_is_hermitian():
    n = 5
    h, eri = random_spinor_integrals(n, seed=3)
    ps = jordan_wigner(h, eri)
    assert ps.coeffs.dtype == np.float64
    dense = ps.to_dense()
    assert np.abs(dense - dense.conj().T).max() < EXACT


def test_the_terms_are_unique_and_canonically_ordered():
    """Two mappings of the same Hamiltonian must compare element by element, which needs a
    canonical form: unique strings, sorted by ``(x, z)``."""
    h, eri = random_spinor_integrals(4, seed=5)
    ps = jordan_wigner(h, eri)
    keys = list(zip(ps.x_masks.tolist(), ps.z_masks.tolist()))
    assert len(set(keys)) == len(keys)
    assert keys == sorted(keys)


def test_a_non_hermitian_input_is_refused_not_mapped():
    """⚠ Refused rather than warned about (the refuse-never-reconcile culture of the method surface).

    Every coefficient of a Hermitian operator in the Pauli basis is real, so a complex one is
    not a small error to be tolerated — it is a different operator, and the message says which
    input property failed.
    """
    n = 4
    h, eri = random_spinor_integrals(n, seed=13)
    h[0, 1] += 0.5                                    # breaks h_pq = h_qp^*
    with pytest.raises(ValueError, match="complex Pauli coefficients"):
        jordan_wigner(h, eri)


def test_e_core_is_excluded_and_the_identity_term_is_not_it():
    """⚠ The qubit Hamiltonian is the active-space operator alone, exactly as the sigma
    operator and the TTNO are. Its identity string comes from normal ordering and has
    nothing to do with the inactive energy — asserted here because the two are the same kind
    of number and adding both is a plausible mistake that shifts every energy."""
    n = 4
    h, eri = random_spinor_integrals(n, seed=17)

    class _Ints:
        e_core = -123.456

        def h_active_effective(self):
            return h

        def active_eri(self):
            return eri

    direct = jordan_wigner(h, eri)
    through = qubit_hamiltonian(_Ints())
    assert np.array_equal(direct.coeffs, through.coeffs)
    assert np.array_equal(direct.x_masks, through.x_masks)
    assert abs(direct.identity_coefficient - _Ints.e_core) > 1.0


def test_an_empty_hamiltonian_maps_to_an_empty_sum():
    n = 3
    zeros_h = np.zeros((n, n), dtype=np.complex128)
    zeros_eri = np.zeros((n,) * 4, dtype=np.complex128)
    ps = jordan_wigner(zeros_h, zeros_eri)
    assert ps.n_terms == 0
    assert ps.identity_coefficient == 0.0
    assert np.abs(ps.to_dense()).max() == 0.0


def test_pauli_labels_put_qubit_zero_first():
    """⚠ Kuiva's order, matching the mask bit order and a determinant. Qiskit's is the
    reverse, and exactly one place (the adapter) is allowed to know that."""
    assert pauli_label(0b0001, 0b0000, 4) == "XIII"
    assert pauli_label(0b0000, 0b1000, 4) == "IIIZ"
    assert pauli_label(0b0010, 0b0010, 4) == "IYII"
    assert pauli_label(0, 0, 3) == "III"


# --- the Pauli algebra the backends share --------------------------------------------------

def _kron_pauli(label):
    """Dense matrix of a Pauli label by explicit Kronecker products, qubit 0 first.

    ⚠ Deliberately **not** built from the symplectic phase rule the library uses: the independence point
    is that a check whose two sides share an implementation cannot see an error in it, and the
    ``i^popcount(x & z)`` phase is exactly the thing worth checking independently. The
    little-endian statevector convention makes qubit 0 the *last* Kronecker factor.
    """
    single = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
              "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.array([[1, 0], [0, -1]])}
    out = np.array([[1.0 + 0.0j]])
    for char in reversed(label):
        out = np.kron(out, single[char])
    return out


@pytest.mark.parametrize("label", ["I", "X", "Y", "Z", "XY", "YZ", "ZX", "YY", "XYZ", "IZY"])
def test_the_symplectic_phase_convention_against_explicit_kronecker_products(label):
    """The one place the ``P(x, z) = i^popcount(x&z) X^x Z^z`` convention is checked against
    something that does not use it."""
    n = len(label)
    x = sum(1 << k for k, c in enumerate(label) if c in "XY")
    z = sum(1 << k for k, c in enumerate(label) if c in "ZY")
    ps = PauliSum(np.array([1.0]), np.array([x], dtype=np.uint64),
                  np.array([z], dtype=np.uint64), n)
    assert np.abs(ps.to_dense() - _kron_pauli(label)).max() < EXACT
    assert ps.labels() == (label,)


def test_pauli_apply_and_expectation_agree_with_the_dense_matrix():
    """The three consumers of one phase rule — the dense matrix, the operator action and the
    expectation value — must be the same operator."""
    n = 4
    h, eri = random_spinor_integrals(n, seed=23)
    ps = jordan_wigner(h, eri)
    dense = ps.to_dense()
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
    psi /= np.linalg.norm(psi)
    assert np.abs(pauli_apply(ps.x_masks, ps.z_masks, ps.coeffs, psi)
                  - dense @ psi).max() < EXACT
    per_term = pauli_expectation(ps.x_masks, ps.z_masks, psi)
    assert abs(float(ps.coeffs @ per_term)
               - float(np.real(np.vdot(psi, dense @ psi)))) < EXACT


def test_the_ground_state_energy_matches_a_full_ci_in_the_same_space():
    """End to end, and the number a solver will actually be judged on.

    The lowest eigenvalue of the qubit Hamiltonian restricted to a particle-number sector is
    the full-CI energy of that sector. ⚠ It excludes ``e_core``, which is the driver's to add.
    """
    n, k = 6, 3
    h, eri = random_spinor_integrals(n, seed=29)
    dense = jordan_wigner(h, eri).to_dense()
    ref, masks = _dense_reference(n, k, h, eri)
    assert abs(np.linalg.eigvalsh(dense[np.ix_(masks, masks)])[0]
               - np.linalg.eigvalsh(ref)[0]) < EXACT


# --- sizing ---------------------------------------------------------------

@pytest.mark.parametrize("n_terms", [0, 1, 137, 4096])
def test_the_sizing_function_is_exact_on_both_sides(n_terms):
    """⚠ Bounded on both sides, so a sizing function that grows a safety factor fails."""
    ps = PauliSum(np.zeros(n_terms), np.zeros(n_terms, dtype=np.uint64),
                  np.zeros(n_terms, dtype=np.uint64), 8)
    actual = (ps.coeffs.nbytes + ps.x_masks.nbytes + ps.z_masks.nbytes) / (1 << 30)
    assert pauli_terms_gb(n_terms) == pytest.approx(actual, rel=0.0, abs=1e-18)


@pytest.mark.parametrize("n_qubit", [1, 4, 8])
def test_the_dense_sizing_functions_account_for_the_whole_peak(n_qubit):
    """⚠ The *workspace* is the larger half and is counted, not ignored.

    Scattered accumulation is real-valued (``np.bincount``), so the real and imaginary parts
    are summed in separate ``2^n x 2^n`` float arrays and combined once. Reporting only the
    complex result would under-account the requirement by a factor of two, which the accounting rule makes a
    defect rather than a rounding.
    """
    dim = 1 << n_qubit
    result = np.zeros((dim, dim), dtype=np.complex128)
    workspace = np.zeros((dim, dim), dtype=np.float64)
    assert dense_matrix_gb(n_qubit) == pytest.approx(result.nbytes / (1 << 30),
                                                     rel=0.0, abs=1e-18)
    assert dense_matrix_workspace_gb(n_qubit) == pytest.approx(2 * workspace.nbytes / (1 << 30),
                                                               rel=0.0, abs=1e-18)


def test_the_dense_matrix_is_refused_rather_than_thrashed():
    """The refusal is raised before the allocation, with advice naming the knob."""
    from kuiva.util import resources

    resources.configure(memory_gb=0.5, allow_overcommit=False)
    try:
        ps = PauliSum(np.array([1.0]), np.array([0], dtype=np.uint64),
                      np.array([0], dtype=np.uint64), 20)
        with pytest.raises(resources.MemoryLimitError):
            ps.to_dense()
    finally:
        resources.reset()


def test_the_pre_collapse_term_count_is_exact():
    """:func:`jordan_wigner_terms` is a statement about the buffer, not an estimate of it: it
    must equal the number of terms the unscreened mapping actually generates."""
    n = 4
    h, eri = random_spinor_integrals(n, seed=31)
    n_one = int((np.abs(h) > 0).sum())
    n_two = int((np.abs(eri) > 0).sum())
    assert jordan_wigner_terms(n_one, n_two) == 4 * n_one + 16 * n_two
    # And collapsing can only shrink it, which is what makes the bound legitimate.
    assert jordan_wigner(h, eri, tol=0.0).n_terms <= jordan_wigner_terms(n_one, n_two)


# --- the registry ---------------------------------------------------------------------------

def test_only_implemented_mappings_are_registered():
    """⚠ ``amf/backend.py``'s decision, restated: a name that resolves to something
    non-functional fails further from its cause than a name that does not resolve."""
    assert "jordan_wigner" in available_mappings()
    assert "bravyi_kitaev" not in available_mappings()
    assert "parity" not in available_mappings()
    with pytest.raises(ValueError, match="unknown fermion-to-qubit mapping"):
        resolve_mapping("bravyi_kitaev")


def test_a_registered_mapping_is_what_the_resolver_returns():
    sentinel = object()

    def _fake(h, eri, tol=0.0):
        return sentinel

    register_mapping("test_only_mapping", _fake)
    try:
        assert resolve_mapping("test_only_mapping") is _fake
        assert "test_only_mapping" in available_mappings()
    finally:
        from kuiva.qc import mapping as m
        m._MAPPINGS.pop("test_only_mapping", None)


def test_the_qubit_ceiling_is_the_determinant_ceiling():
    """One limit, from one place: both are the single ``uint64`` mask of ``ci/strings.py``."""
    from kuiva.ci.strings import DEFAULT_MAX_SPINORS

    assert MAX_QUBITS == DEFAULT_MAX_SPINORS
    assert HERMITICITY_TOL > 0.0


# --- Stage 5: the generalized ladder mapper, grouping and RDM operators -------------------------

def test_the_hamiltonian_and_an_arbitrary_operator_go_through_one_path():
    """``fermionic_operator`` is what :func:`jordan_wigner` is now built from, so assembling
    the Hamiltonian from its two blocks by hand must reproduce it term for term.

    The point is not the arithmetic but the *sharing*: an excitation generator and an RDM
    operator use the same function, so a sign convention cannot differ between the operator
    that is diagonalized and the operator that is exponentiated.
    """
    n = 4
    h, eri = _integrals(n, seed=21)
    reference = jordan_wigner(h, eri)
    p, q = (idx.reshape(-1) for idx in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
    one = fermionic_operator(h[p, q], np.stack([p, q], axis=1), (True, False), n)
    grid = np.meshgrid(*(np.arange(n),) * 4, indexing="ij")
    a, b, c, d = (g.reshape(-1) for g in grid)
    two = fermionic_operator(0.5 * eri[a, b, c, d], np.stack([a, c, d, b], axis=1),
                             (True, True, False, False), n)
    rebuilt = one.plus(two).drop_below(1e-14)
    assert rebuilt.n_terms == reference.n_terms
    assert np.abs(rebuilt.coeffs - reference.coeffs).max() < 1e-13


def test_a_non_hermitian_ladder_sum_is_refused():
    """``a_p^dag a_q`` alone is not Hermitian, and the check that catches it is the mapping's
    own load-bearing one — so it must fire here too, not only on a bad Hamiltonian."""
    with pytest.raises(ValueError, match="complex Pauli coefficients"):
        fermionic_operator([1.0], [[0, 1]], (True, False), 3)


@pytest.mark.parametrize("n", [4, 6])
def test_qubit_wise_commuting_groups_are_a_valid_partition(n):
    """Every group is measurable with one circuit, and every term is in exactly one group."""
    operator = jordan_wigner(*_integrals(n, seed=22))
    groups = qwc_groups(operator)
    assert sorted(int(t) for g in groups for t in g) == list(range(operator.n_terms))
    for members in groups:
        for i in members.tolist():
            for j in members.tolist():
                overlap = int((operator.x_masks[i] | operator.z_masks[i])
                              & (operator.x_masks[j] | operator.z_masks[j]))
                assert int(operator.x_masks[i]) & overlap == int(operator.x_masks[j]) & overlap
                assert int(operator.z_masks[i]) & overlap == int(operator.z_masks[j]) & overlap
    # Grouping is worth doing: the circuit count falls by a factor, and it is reported rather
    # than asserted tightly, because a greedy partition is not canonical.
    assert len(groups) < operator.n_terms


def test_qubit_wise_commuting_implies_commuting_but_not_the_reverse():
    """The two notions, and the reason the cheap one is used: X_0 Z_1 and Z_0 X_1 commute
    (two anticommuting factors) but are not qubit-wise commuting, so they cannot share a
    measurement basis."""
    assert bool(pauli_commute(0b01, 0b10, 0b10, 0b01))
    operator = PauliSum(np.array([1.0, 1.0]), np.array([0b01, 0b10], np.uint64),
                        np.array([0b10, 0b01], np.uint64), 2)
    assert operator.all_commute()
    assert len(qwc_groups(operator)) == 2


def test_the_rdm_operators_reproduce_the_classical_rdms_exactly():
    """The measurement plan against ``ci/strings.rdm12`` on the same CI vector — independent
    code paths, so the comparison can fail."""
    from kuiva.ci.strings import rdm12

    n, n_elec = 4, 2
    h, eri = _integrals(n, seed=23)
    dets = CASSpace(n, n_elec).determinants()
    _values, vectors = np.linalg.eigh(hamiltonian_matrix(dets, h, eri).toarray())
    ci = vectors[:, 0]
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[dets.masks.astype(np.int64)] = ci

    plan = rdm_measurement(n, rank=2)
    measured = pauli_expectation(plan.x_masks, plan.z_masks, psi)
    gamma, gamma2 = rdm12(dets, ci[:, None], np.array([1.0]))
    assert np.abs(plan.gamma(measured) - gamma).max() < 1e-13
    assert np.abs(plan.gamma2(measured) - gamma2).max() < 1e-13


def _integrals(n, seed):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = h + h.conj().T
    eri = rng.normal(size=(n,) * 4) + 1j * rng.normal(size=(n,) * 4)
    eri = eri + eri.transpose(2, 3, 0, 1)
    eri = 0.2 * (eri + eri.conj().transpose(1, 0, 3, 2))
    return h, eri
