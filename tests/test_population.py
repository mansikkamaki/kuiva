"""Tier-0/1 tests for the AO layout and Loewdin population analysis.

Three groups of checks, chosen by what each can *fail on*:

* **The layout** — asserted against PySCF's own AO ordering and molden permutation, because
  it is bookkeeping, and bookkeeping is only ever validated by an independent statement of the
  same thing. The p-shell ordering is singled out: ``libcint`` orders p as ``px, py, pz``
  while every other ``l`` runs ``-l..+l``, and getting it wrong is invisible in any total
  population while being wrong in every per-AO row.
* **Sum rules** — populations sum to the electron count, a spinor's own populations sum to 1.
  Cheap, and they catch a wrong metric immediately.
* **The conjugation trap ** — the AO density is ``C gamma^T C^dag``. It needs a
  **non-diagonal complex** ``gamma``, because for the natural spinors a caller normally has,
  the two forms agree. ⚠ And it needs the **spin** density: the charge population is
  algebraically identical under both conventions, so a charge or a sum rule cannot see the
  trap at all. Both halves of that statement are asserted, the second so that a future reader
  does not conclude from a passing charge test that this line is covered.
* **Spin-density sign conventions** — pinned against known one-spinor cases (a pure alpha
  spinor is ``s_z = +1/2``, ``(alpha + i beta)/sqrt2`` points along ``+y``). The convention requires
  exactly this of anything involving ``sigma``: a sign error is invisible to every norm.
"""
import numpy as np
import pytest

from kuiva.basis.layout import molden_ao_order, shell_m_values
from kuiva.interface.api import Molecule
from kuiva.interface.pyscf_bridge import ao_layout, run_scalar_x2c
from kuiva.orth.canonical import sqrt_overlap
from kuiva.props.population import (atomic_populations, degenerate_groups, frontier_columns,
                                    kramers_pair_groups, lowdin_analysis,
                                    lowdin_coefficients, orbital_populations, select_columns)
from kuiva.spinor.expand import expand_scalar_mos, time_reverse

#: Sum rules are exact linear algebra; 1e-10 leaves room for the S^{1/2} eigendecomposition.
SUM_TOL = 1e-10


@pytest.fixture(scope="module")
def ticl():
    """A two-atom heavy/light system: two elements, d and p shells, f polarization."""
    from pyscf import gto
    return gto.M(atom=[["Ti", (0, 0, 0)], ["Cl", (0, 0, 2.3)]], basis="x2c-SVPall-2c",
                 spin=1, verbose=0)


# --- The AO layout --------------------------------------------------------------------------

def test_layout_covers_the_basis(ticl):
    lay = ao_layout(ticl)
    assert lay.nao == ticl.nao
    assert lay.natm == 2
    assert sum(sh.nao for sh in lay.shells) == ticl.nao
    assert np.array_equal(lay.atom_charges, [22.0, 17.0])


def test_layout_atom_assignment_matches_pyscf(ticl):
    lay = ao_layout(ticl)
    slices = ticl.aoslice_by_atom()
    for ia in range(ticl.natm):
        expected = np.arange(slices[ia][2], slices[ia][3])
        assert np.array_equal(lay.atom_indices(ia), expected)


def test_layout_labels_match_pyscf(ticl):
    lay = ao_layout(ticl)
    assert list(lay.ao_labels) == [lbl.split()[2] for lbl in ticl.ao_labels()]


def test_p_shells_are_ordered_x_y_z_not_minus_l_to_l():
    """⚠ The one ordering rule that is not ``-l..+l`` (see :mod:`kuiva.basis.layout`)."""
    assert shell_m_values(1) == [1, -1, 0]
    assert shell_m_values(2) == [-2, -1, 0, 1, 2]
    assert shell_m_values(0) == [0]


def test_m_labels_agree_with_pyscf_for_p_and_d(ticl):
    lay = ao_layout(ticl)
    by_label = {lay.ao_labels[i]: int(lay.ao_m[i]) for i in range(lay.nao)}
    assert by_label["2px"] == 1 and by_label["2py"] == -1 and by_label["2pz"] == 0
    assert by_label["3dxy"] == -2 and by_label["3dz^2"] == 0 and by_label["3dx2-y2"] == 2


def test_molden_permutation_matches_pyscf(ticl):
    """The whole molden AO-ordering convention, against an independent implementation."""
    from pyscf.tools import molden
    lay = ao_layout(ticl)
    assert np.array_equal(molden_ao_order(lay.shells, max_l=None),
                          np.asarray(molden.order_ao_index(ticl)))


def test_molden_permutation_is_a_permutation(ticl):
    lay = ao_layout(ticl)
    order = molden_ao_order(lay.shells, max_l=None)
    assert np.array_equal(np.sort(order), np.arange(lay.nao))


def test_high_l_functions_are_omitted_above_max_l(ticl):
    lay = ao_layout(ticl)
    kept = molden_ao_order(lay.shells, max_l=2)
    assert np.all(lay.ao_l[kept] <= 2)
    assert kept.size == int(np.count_nonzero(lay.ao_l <= 2))


def test_cartesian_basis_refused():
    from pyscf import gto
    mol = gto.M(atom=[["O", (0, 0, 0)]], basis="sto-3g", spin=2, cart=True, verbose=0)
    with pytest.raises(NotImplementedError, match="Cartesian"):
        ao_layout(mol)


def test_group_by_ao_type_partitions_the_basis(ticl):
    lay = ao_layout(ticl)
    groups = lay.group_by_ao_type()
    covered = np.concatenate(list(groups.values()))
    assert np.array_equal(np.sort(covered), np.arange(lay.nao))


# --- S^{1/2} ------------------------------------------------------------------------------

def test_sqrt_overlap_squares_back(ticl):
    s = ticl.intor("int1e_ovlp")
    root = sqrt_overlap(s)
    assert np.max(np.abs(root @ root - s)) < 1e-10
    assert np.max(np.abs(root - root.T)) < 1e-12


def test_sqrt_overlap_survives_a_singular_metric():
    """⚠ The reason this is not ``symmetric_orthogonalization``, which raises here: the square
    root is well defined for a singular ``S``, only the inverse root is not."""
    s = np.diag([1.0, 1.0, 0.0])
    assert np.max(np.abs(sqrt_overlap(s) - s)) < 1e-14


# --- Sum rules ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def water():
    """A closed-shell scalar reference; SOC ingestion is off (pure cost when unused)."""
    mol = Molecule.from_xyz_string("O 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24",
                                   basis="x2c-SVPall-2c")
    return run_scalar_x2c(mol, screening="none", memory_gb=4.0)


@pytest.fixture(scope="module")
def water_spinors(water):
    """AO-basis spinors and occupations for the water reference."""
    sb = expand_scalar_mos(water.mo_coeff, water.mo_energy, water.mo_occ, basis="ao")
    return sb.c, sb.occ, water.s_ao, water.ao_layout


def test_populations_sum_to_the_electron_count(water_spinors):
    c, occ, s, lay = water_spinors
    pops = atomic_populations(c, s, lay, occupation=occ)
    assert pops.n_electrons == pytest.approx(10.0, abs=SUM_TOL)
    assert pops.atomic_charge().sum() == pytest.approx(0.0, abs=SUM_TOL)


def test_each_spinor_population_sums_to_one(water_spinors):
    c, _, s, lay = water_spinors
    pops = orbital_populations(c, s, lay, group="none")
    assert np.allclose(pops.ao.sum(axis=0), 1.0, atol=SUM_TOL)


def test_a_kramers_pair_holds_two_spinors(water_spinors):
    c, _, s, lay = water_spinors
    pops = orbital_populations(c, s, lay, group="kramers")
    assert np.allclose(pops.ao.sum(axis=0), 2.0, atol=SUM_TOL)
    assert pops.n_groups == c.shape[1] // 2


def test_water_charges_are_chemically_sensible(water_spinors):
    """A Tier-1 sanity check, not an accuracy claim: oxygen is negative, hydrogens positive
    and equal by symmetry of the input geometry to within its asymmetry."""
    c, occ, s, lay = water_spinors
    q = atomic_populations(c, s, lay, occupation=occ).atomic_charge()
    assert q[0] < -0.05 and q[1] > 0.0 and q[2] > 0.0
    assert abs(q[1] - q[2]) < 0.01


def test_closed_shell_has_no_spin_density(water_spinors):
    c, occ, s, lay = water_spinors
    pops = atomic_populations(c, s, lay, occupation=occ)
    assert np.max(np.abs(pops.atomic_spin())) < SUM_TOL


# --- Spin: the convention, which no norm-based check can see -------------------------------

def spin_of(c_col, s, lay):
    return atomic_populations(c_col, s, lay, occupation=np.ones(c_col.shape[1])).atomic_spin()


def test_pure_alpha_spinor_has_spin_up_one_half(water_spinors):
    """⚠ Fixes the sign of the whole spin-density convention. Validated against a known case,
    per the convention's rule for anything involving sigma."""
    c, _, s, lay = water_spinors
    nao = s.shape[0]
    alpha = np.zeros((2 * nao, 1), dtype=complex)
    alpha[:nao, 0] = c[:nao, 0]                       # spinor 0 is (phi, 0) by construction
    total = spin_of(alpha, s, lay).sum(axis=1)
    assert total[2] == pytest.approx(0.5, abs=1e-8)
    assert abs(total[0]) < 1e-10 and abs(total[1]) < 1e-10


def test_pure_beta_spinor_has_spin_down(water_spinors):
    c, _, s, lay = water_spinors
    nao = s.shape[0]
    beta = np.zeros((2 * nao, 1), dtype=complex)
    beta[nao:, 0] = c[:nao, 0]
    assert spin_of(beta, s, lay).sum(axis=1)[2] == pytest.approx(-0.5, abs=1e-8)


def test_spin_along_x_and_y(water_spinors):
    """``(alpha + beta)/sqrt2`` points along +x and ``(alpha + i beta)/sqrt2`` along +y.

    This is what pins ``s_x = +Re D_ab`` and ``s_y = -Im D_ab``; a swapped sign on ``s_y``
    passes every sum rule and every norm."""
    c, _, s, lay = water_spinors
    nao = s.shape[0]
    phi = c[:nao, 0]
    for phase, axis in ((1.0, 0), (1j, 1)):
        col = np.zeros((2 * nao, 1), dtype=complex)
        col[:nao, 0] = phi / np.sqrt(2.0)
        col[nao:, 0] = phase * phi / np.sqrt(2.0)
        total = spin_of(col, s, lay).sum(axis=1)
        assert total[axis] == pytest.approx(0.5, abs=1e-8)
        assert abs(total[2]) < 1e-10


def test_kramers_pair_has_exactly_zero_spin(water_spinors):
    """⚠ The result that will otherwise be read as a bug: a state-averaged Kramers pair has
    no spin density anywhere (module docstring)."""
    c, _, s, lay = water_spinors
    nao = s.shape[0]
    col = c[:, 4:5].copy()
    pair = np.hstack([col, time_reverse(col)])
    assert np.max(np.abs(spin_of(pair, s, lay))) < SUM_TOL


# --- The conjugation trap ---------------------------------------------------------

def random_gamma(n, seed=2, electrons=3.0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    gamma = a @ a.conj().T                             # Hermitian positive definite
    return gamma * (electrons / np.trace(gamma).real)


def test_charge_population_cannot_see_the_conjugation_trap(water_spinors):
    """⚠ The conjugation trap, and the reason the next test exists.

    ``C gamma^T C^dag`` and ``C gamma C^dag`` differ, but their *same-spin* diagonals are
    complex conjugates of each other, and a population is the real part of one. So the charge
    populations are equal under the wrong convention — identically, up to the summation order
    inside the contraction — and this is asserted here so that nobody later mistakes a passing
    charge sum rule for a validation of the density build.
    """
    c, _, s, lay = water_spinors
    gamma = random_gamma(6)
    sub = np.ascontiguousarray(c[:, :6])
    right = atomic_populations(sub, s, lay, dm=gamma)
    wrong = atomic_populations(sub, s, lay, dm=gamma.T)
    assert np.max(np.abs(right.ao_population - wrong.ao_population)) < 1e-15
    assert right.n_electrons == pytest.approx(float(np.trace(gamma).real), abs=SUM_TOL)


def test_density_matrix_uses_gamma_transpose(water_spinors):
    """⚠ The conjugation trap: the AO density reproducing ``gamma`` is ``C gamma^T C^dag``.

    The **spin** density is where the two conventions differ (68% on this system), because it
    comes from the off-diagonal spin block, where the cancellation of the test above does not
    happen. The reference is built independently here, straight from the definition
    ``rho_{ss'} = C^s gamma^T C^{s'dag}``, rather than by calling the same code twice.
    """
    c, _, s, lay = water_spinors
    n = 6
    gamma = random_gamma(n)
    sub = np.ascontiguousarray(c[:, :n])
    nao = s.shape[0]

    pops = atomic_populations(sub, s, lay, dm=gamma)

    root = sqrt_overlap(s)
    ca, cb = root @ sub[:nao], root @ sub[nao:]
    rho_ab = np.diag(ca @ gamma.T @ cb.conj().T)
    expected = np.stack([rho_ab.real, -rho_ab.imag,
                         np.diag(ca @ gamma.T @ ca.conj().T).real
                         - np.diag(cb @ gamma.T @ cb.conj().T).real])
    expected[2] *= 0.5
    assert np.max(np.abs(pops.ao_spin - expected)) < SUM_TOL

    wrong = atomic_populations(sub, s, lay, dm=gamma.T)
    assert np.max(np.abs(wrong.ao_spin - pops.ao_spin)) > 0.1 * np.max(np.abs(pops.ao_spin))


def test_dm_and_occupation_agree_for_a_diagonal_gamma(water_spinors):
    """The two entry points must coincide where both are defined -- and this is exactly the
    case in which the conjugation trap is invisible."""
    c, occ, s, lay = water_spinors
    n = 8
    sub, w = np.ascontiguousarray(c[:, :n]), occ[:n]
    by_dm = atomic_populations(sub, s, lay, dm=np.diag(w).astype(complex))
    by_occ = atomic_populations(sub, s, lay, occupation=w)
    assert np.max(np.abs(by_dm.ao_population - by_occ.ao_population)) < SUM_TOL


def test_dm_and_occupation_are_mutually_exclusive(water_spinors):
    c, occ, s, lay = water_spinors
    with pytest.raises(ValueError, match="exactly one"):
        atomic_populations(c, s, lay)
    with pytest.raises(ValueError, match="exactly one"):
        atomic_populations(c, s, lay, dm=np.eye(c.shape[1]), occupation=occ)


# --- Invariance: which rows mean something -------------------------------------------------

def test_block_populations_are_invariant_under_mixing(water_spinors):
    """⚠ The invariance discipline applied here: an individual spinor's population inside a
    degenerate manifold is arbitrary, the block sum is not."""
    c, _, s, lay = water_spinors
    block = np.ascontiguousarray(c[:, 2:6])
    rng = np.random.default_rng(7)
    a = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    u = np.linalg.qr(a)[0]

    before = orbital_populations(block, s, lay, group=[[0, 1, 2, 3]]).ao
    after = orbital_populations(block @ u, s, lay, group=[[0, 1, 2, 3]]).ao
    assert np.max(np.abs(before - after)) < SUM_TOL

    # ... and the per-spinor rows really are basis-dependent, which is why they are not default
    per_before = orbital_populations(block, s, lay, group="none").ao
    per_after = orbital_populations(block @ u, s, lay, group="none").ao
    assert np.max(np.abs(per_before - per_after)) > 1e-3


def test_lowdin_coefficients_are_orthonormal(water_spinors):
    """``S^{1/2} C`` has orthonormal columns when ``C`` is orthonormal in the ``S`` metric —
    which is what makes ``|.|^2`` a population that sums to 1."""
    c, _, s, _ = water_spinors
    ct = lowdin_coefficients(c, s)
    nao = s.shape[0]
    gram = ct[:nao].conj().T @ ct[:nao] + ct[nao:].conj().T @ ct[nao:]
    assert np.max(np.abs(gram - np.eye(c.shape[1]))) < 1e-10


# --- Grouping and selection ---------------------------------------------------------------

def test_kramers_pair_groups():
    assert [g.tolist() for g in kramers_pair_groups(6)] == [[0, 1], [2, 3], [4, 5]]
    with pytest.raises(ValueError, match="even number"):
        kramers_pair_groups(5)


def test_degenerate_groups_uses_adjacency():
    groups = degenerate_groups(np.array([1.0, 1.0, 0.5, 0.5, 0.5, 1.0]))
    assert [g.tolist() for g in groups] == [[0, 1], [2, 3, 4], [5]]


def test_frontier_columns_straddles_the_gap():
    occ = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    assert frontier_columns(occ, 1, 1, pairs=False).tolist() == [3, 4]


def test_frontier_columns_returns_whole_kramers_pairs():
    """⚠ A frontier taken spinor by spinor splits pairs: the HOMO and LUMO are adjacent
    columns of *different* pairs, and half a pair is a basis-dependent object."""
    occ = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    cols = frontier_columns(occ, 1, 1)
    assert cols.tolist() == [2, 3, 4, 5]
    assert cols.size % 2 == 0 and np.all(cols[0::2] % 2 == 0)


def test_select_columns_levels():
    assert select_columns("active", 10, active=[2, 3]).tolist() == [2, 3]
    assert select_columns("all", 4).tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="unknown analysis level"):
        select_columns("everything", 4)


def test_level_all_warns(kuiva_caplog):
    """"Proceeds but the user should know" is behaviour, and large output is that case."""
    select_columns("all", 200)
    assert any("level='all'" in r.getMessage() for r in kuiva_caplog.records)


def test_kramers_grouping_warns_on_unpaired_occupations(water_spinors, kuiva_caplog):
    c, _, s, lay = water_spinors
    occ = np.zeros(c.shape[1])
    occ[0] = 1.0                                       # partners with different occupations
    orbital_populations(c, s, lay, columns=[0, 1, 2, 3], group="kramers", occupation=occ)
    assert any("not Kramers paired" in r.getMessage() for r in kuiva_caplog.records)


# --- The driver and its report ------------------------------------------------------------

def test_analysis_reports_without_an_active_space(water_spinors, kuiva_caplog):
    """``level="active"`` with no active space gives the charges and no orbital table, rather
    than raising: a caller with no active space yet still wants the charges."""
    c, occ, s, lay = water_spinors
    atomic, orbital = lowdin_analysis(c, s, lay, occupation=occ, report=True)
    assert orbital is None
    assert atomic.n_electrons == pytest.approx(10.0, abs=SUM_TOL)
    assert any("Loewdin atomic populations" in r.getMessage() for r in kuiva_caplog.records)


def test_reduced_populations_identify_the_oxygen_lone_pair(water_spinors):
    """What the analysis is *for*: naming an orbital without looking at it."""
    c, occ, s, lay = water_spinors
    homo = int(np.nonzero(occ > 0.5)[0][-1])
    pops = orbital_populations(c, s, lay, columns=[homo - 1, homo], group="kramers")
    on_oxygen = pops.by_atom()[0] / pops.ao.sum(axis=0)
    assert on_oxygen[0] > 0.85                         # a lone pair, not a bonding orbital
    p_character = pops.by_angular_momentum()[(0, 1)] / pops.ao.sum(axis=0)
    assert p_character[0] > 0.85


def test_normalized_columns_sum_to_one(water_spinors):
    c, _, s, lay = water_spinors
    pops = orbital_populations(c, s, lay, columns=range(8))
    assert np.allclose(pops.normalized().sum(axis=0), 1.0, atol=SUM_TOL)


def test_by_ao_type_and_by_atom_agree(water_spinors):
    c, _, s, lay = water_spinors
    pops = orbital_populations(c, s, lay, columns=range(6))
    by_type = pops.by_ao_type()
    per_atom = np.zeros_like(pops.by_atom())
    for (ia, _), value in by_type.items():
        per_atom[ia] += value
    assert np.max(np.abs(per_atom - pops.by_atom())) < SUM_TOL


def test_reduced_table_accounts_for_the_whole_orbital(water_spinors, kuiva_caplog):
    """The 'remainder' row exists so the printed percentages always add to 100."""
    c, _, s, lay = water_spinors
    pops = orbital_populations(c, s, lay, columns=[0, 1])
    pops.report(tolerance=0.4)
    text = "\n".join(r.getMessage() for r in kuiva_caplog.records)
    assert "remainder" in text
