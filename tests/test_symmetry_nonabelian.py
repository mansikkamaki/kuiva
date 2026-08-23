"""Tier-0/Tier-1 tests for the non-abelian CLASSIFICATION layer.

What each group can fail on, which is the test of a check's worth:

* the **group construction**, against the theorems a finite group must satisfy — the double
  group is twice the point group, the characters are orthogonal both ways, and the dimensions
  square-sum to the order. The character tables here are *computed* rather than transcribed,
  so these are the checks that make them trustworthy;
* the irrep **names**, against the published tables. That is the one place transcription
  belongs: in a test that can fail, rather than in the code that would then be right by
  definition. The sources are Koster-Dimmock-Wheeler-Statz and Altmann-Herzig, cited where
  the naming rules are implemented;
* the general operator construction against the **landed abelian one**. For the four
  operations both modules can build, the Ivanic-Ruedenberg harmonic blocks and the
  permutation-and-sign form must agree element for element — a genuinely independent
  construction of the same matrices, and the only thing that keeps one convention rather than
  two that agree until one is changed;
* the **subduction** against the textbook correlation tables, computed rather than stored;
* ``U(g)`` on a CI vector against the matrix exponential of the one-body operator built
  straight from the determinant strings. Two constructions with no line of code in common, one
  of them exact by definition;
* the **classification of a real spectrum**, on a molecule whose multiplets are known from
  physics: ``N2+`` in a pi/sigma active space has a ``2 Sigma g`` ground state and a
  ``2 Pi u`` excited term whose spin-orbit components are ``Omega = 1/2`` and ``3/2``, which
  in the finite group used are ``E1/2g``, ``E1/2u`` and ``E3/2u``;
* the **refusals**: an active space the group takes states out of, and a state count that cuts
  a multiplet theory fixes. Each of those otherwise produces a plausible number under a
  meaningless label;
* that classification **changes no number**, which is the whole claim of the layer being a
  labelling.
"""
import logging

import numpy as np
import pytest

from kuiva.basis.layout import shell_m_values
from kuiva.ci.strings import CASSpace
from kuiva.interface.api import Molecule, casci, spinor_reference
from kuiva.symm.classify import (StateClassifier, apply_orbital_rotation,
                                 assert_multiplet_boundary, givens_factors)
from kuiva.symm.double import candidate_groups, detect_point_group, double_group, is_linear
from kuiva.symm.groups import C1, C2H_Z, C2Z, CS_XY, CI
from kuiva.symm.operators import SPIN_FACTOR, ao_operation
from kuiva.symm.rotations import (ao_transform, harmonic_rotation, operation_name, reflection,
                                  rotation, spin_factor)

#: Groups the suite exercises. Everything the detector can return for a system in this
#: project, plus the two families whose naming rules have their own branch.
GROUPS = ("C1", "Ci", "Cs", "C2", "C2h", "C2v", "D2", "D2h", "C3", "C3v", "D3", "D3h",
          "C4v", "D4h", "C6v", "D6h", "D2d", "S4")


# --- Systems ---------------------------------------------------------------------------------

def n2_cation(point_group="auto"):
    """``N2+`` on the ``z`` axis: 13 electrons, so its states are Kramers doublets, and a real
    ``2 Sigma`` / ``2 Pi`` spectrum a few hundredths of a hartree apart."""
    return Molecule(atoms=[("N", (0.0, 0.0, 0.56)), ("N", (0.0, 0.0, -0.56))],
                    basis="x2c-SVPall-2c", charge=1, spin=1, point_group=point_group)


@pytest.fixture(scope="module")
def cation_reference():
    # screening="none": the classification layer changes no scalar quantity, so the
    # four-component atomic solve the default would pay for is pure cost here.
    return spinor_reference(n2_cation(), screening="none", memory_gb=6.0)


#: The pi_u and pi_g* shells plus the sigma_g HOMO: closed under every operation of the
#: classification group, which is what the layer needs and what a partial shell would break.
CATION_ACTIVE = list(range(8, 18))
CATION_ELEC = 5


# --- The group construction ------------------------------------------------------------------

@pytest.mark.parametrize("name", GROUPS)
def test_the_double_group_is_twice_the_point_group(name):
    """Every spatial operation appears twice, once with each sign of the spin factor."""
    group = double_group(name)
    spatial = {tuple(np.round(e.cart.ravel(), 6) + 0.0) for e in group.elements}
    assert group.order == 2 * len(spatial)


@pytest.mark.parametrize("name", GROUPS)
def test_the_computed_character_table_is_orthogonal_both_ways(name):
    """Row and column orthogonality plus ``sum d^2 = |G|`` — the three statements that pin a
    character table, and the ones that make a *computed* table as trustworthy as a printed
    one."""
    group = double_group(name)
    sizes = np.array([len(c) for c in group.classes], dtype=float)
    chi = group.characters
    rows = (chi * sizes[None, :]) @ np.conj(chi).T / group.order
    assert np.allclose(rows, np.eye(group.n_irreps), atol=1e-9)
    columns = np.conj(chi).T @ chi
    assert np.allclose(columns, np.diag(group.order / sizes), atol=1e-8)
    assert int(np.sum(group.dimensions ** 2)) == group.order


@pytest.mark.parametrize("name", GROUPS)
def test_exactly_half_the_group_order_lives_in_the_fermion_irreps(name):
    """``sum d^2`` over the double-valued irreps is ``|G| / 2`` — the spinor half of the
    regular representation, and a check that the ``Ebar`` grading is being read right."""
    group = double_group(name)
    fermion = [r for r in range(group.n_irreps) if group.is_fermion(r)]
    assert int(sum(int(group.dimensions[r]) ** 2 for r in fermion)) == group.order // 2


#: Irrep names as the published double-group tables spell them. ⚠ This is the transcription,
#: deliberately in the test rather than in the code: Koster, Dimmock, Wheeler & Statz,
#: "Properties of the Thirty-Two Point Groups" (1963), and Altmann & Herzig, "Point-Group
#: Theory Tables" (1994), with the Mulliken spelling of the single-valued rows.
PUBLISHED_NAMES = {
    "C2v": ("A1", "A2", "B1", "B2", "E1/2"),
    "C3v": ("A1", "A2", "E", "E1/2", "1E3/2", "2E3/2"),
    "D3": ("A1", "A2", "E", "E1/2", "1E3/2", "2E3/2"),
    "D2": ("A", "B1", "B2", "B3", "E1/2"),
    "D2h": ("Ag", "Au", "B1g", "B1u", "B2g", "B2u", "B3g", "B3u", "E1/2g", "E1/2u"),
    "C2h": ("Ag", "Au", "Bg", "Bu", "1E1/2g", "2E1/2g", "1E1/2u", "2E1/2u"),
    "C4v": ("A1", "A2", "B1", "B2", "E", "E1/2", "E3/2"),
    "C6v": ("A1", "A2", "B1", "B2", "E1", "E2", "E1/2", "E3/2", "E5/2"),
    "D2d": ("A1", "A2", "B1", "B2", "E", "1E1/2", "2E1/2"),
}


@pytest.mark.parametrize("name", sorted(PUBLISHED_NAMES))
def test_the_irrep_names_match_the_published_tables(name):
    """The naming rules are Mulliken's for the single-valued irreps and ``|omega|`` on the
    principal axis for the double-valued ones; this is what says they were applied right."""
    group = double_group(name)
    assert sorted(group.irrep_names) == sorted(PUBLISHED_NAMES[name])


def test_the_abelian_label_group_names_agree_with_the_classification_group():
    """⚠ One vocabulary, two implementations of it. The abelian label groups spell their
    irreps from a stored table and the classification groups from computed characters; where
    the two describe the same group they must produce the same bytes, or a user reading a
    correspondence table would see two names for one irrep."""
    for abelian, full in ((C1, "C1"), (CI, "Ci"), (C2Z, "C2"), (CS_XY, "Cs"), (C2H_Z, "C2h")):
        stored = sorted(abelian.names.values())
        computed = sorted(double_group(full).irrep_names)
        assert stored == computed, full


# --- The operator matrices --------------------------------------------------------------------

@pytest.mark.parametrize("l", list(range(7)))
def test_a_harmonic_block_is_orthogonal_and_a_homomorphism(l):
    """``D_l(R1 R2) = D_l(R1) D_l(R2)`` and ``D_l D_l^T = 1``. The recursion is the only place
    a real-harmonic convention above ``l = 1`` is defined, so these are what pin it."""
    r1 = rotation([0.0, 0.0, 1.0], 2.0 * np.pi / 3.0)
    r2 = rotation([1.0, 1.0, 0.0], np.pi / 2.0)
    d1, d2 = harmonic_rotation(l, r1), harmonic_rotation(l, r2)
    assert np.allclose(d1 @ d1.T, np.eye(2 * l + 1), atol=1e-12)
    assert np.allclose(d1 @ d2, harmonic_rotation(l, r1 @ r2), atol=1e-10)


@pytest.mark.parametrize("l", list(range(7)))
def test_the_harmonic_blocks_reproduce_the_landed_sign_table(l):
    """⚠ The check that keeps one convention rather than two. For ``C2(z)``, ``i`` and
    ``sigma(xy)`` the general recursion must reproduce the ``(-1)^|m|`` / ``(-1)^l`` signs the
    abelian implementation states as a table — including the ``px, py, pz`` order of a p
    shell, which is the case that catches people."""
    m = np.abs(np.asarray(shell_m_values(l)))
    for spatial, cart in (((1, 0), np.diag([-1.0, -1.0, 1.0])),
                          ((0, 1), -np.eye(3)),
                          ((1, 1), np.diag([1.0, 1.0, -1.0]))):
        block = harmonic_rotation(l, cart)
        assert np.allclose(block, np.diag(np.diag(block)), atol=1e-12), spatial
        exponent = np.zeros_like(m)
        if spatial[0]:
            exponent = exponent + m
        if spatial[1]:
            exponent = exponent + l
        assert np.allclose(np.diag(block), np.where(exponent % 2 == 0, 1.0, -1.0), atol=1e-12)


def test_the_spin_factors_agree_with_the_landed_branch():
    """The ``theta`` in ``[0, pi)`` branch, arrived at from an axis-angle decomposition rather
    than from a table. A different branch would swap every fermion irrep with its partner and
    leave every number plausible."""
    for spatial, cart in (((0, 0), np.eye(3)), ((1, 0), np.diag([-1.0, -1.0, 1.0])),
                          ((0, 1), -np.eye(3)), ((1, 1), np.diag([1.0, 1.0, -1.0]))):
        assert np.allclose(spin_factor(cart), SPIN_FACTOR[spatial], atol=1e-12), spatial


def test_the_general_ao_operator_equals_the_abelian_one(cation_reference):
    """Two independent constructions of ``U(g)``: a permutation with a sign, and a shell
    permutation with real harmonic blocks. Element for element, on a real molecule's basis."""
    layout = cation_reference.data.ao_layout
    for spatial, cart in (((1, 0), np.diag([-1.0, -1.0, 1.0])),
                          ((0, 1), -np.eye(3)), ((1, 1), np.diag([1.0, 1.0, -1.0]))):
        abelian = ao_operation(layout, spatial)
        general = ao_transform(layout, cart)
        assert (abelian is None) == (general is None), spatial
        if abelian is None:
            continue
        assert np.allclose(general.matrix(), abelian.matrix(), atol=1e-12), spatial
        assert np.allclose(general.two_component(), abelian.two_component(), atol=1e-12)


def test_operation_names_are_lab_frame_geometry():
    """⚠ Every table prints these, and a wrong axis convention is otherwise silent."""
    cases = [(np.eye(3), "E"), (rotation([0, 0, 1], 2 * np.pi / 3), "C3(z)"),
             (rotation([0, 0, 1], 4 * np.pi / 3), "C3^2(z)"), (-np.eye(3), "i"),
             (reflection([0, 0, 1]), "sigma(xy)"), (reflection([0, 1, 0]), "sigma(xz)"),
             (rotation([1, 0, 0], np.pi), "C2(x)"),
             (reflection([0, 0, 1]) @ rotation([0, 0, 1], np.pi / 2), "S4(z)")]
    for cart, expected in cases:
        assert operation_name(cart) == expected


def test_the_characters_are_reproduced_by_the_operator_traces(cation_reference):
    """⚠ **The check the printed tables exist for, in its non-abelian form.** For a degenerate
    block of the run's own orbitals, ``tr U(g)`` computed from the AO representation matrices
    must be an integer combination of the group's characters — so the printed table and the
    matrices the calculation uses cannot drift apart."""
    data = cation_reference.data
    group = data.symmetry.full_group
    assert group is not None
    layout = data.ao_layout
    c = cation_reference.spinors_in_ao()
    nao = data.s_ao.shape[0]
    s2 = np.kron(np.eye(2), np.asarray(data.s_ao))
    energies = np.asarray(cation_reference.spinors.energy)
    traces = np.zeros(group.n_irreps, dtype=complex)
    block = slice(8, 12)                       # the pi_u shell: four spinors, one manifold
    assert np.ptp(energies[block]) < 1e-8
    for k in range(group.n_irreps):
        element = group.elements[group.class_representative(k)]
        transform = ao_transform(layout, element.cart)
        moved = np.empty_like(c)
        upper, lower = transform.apply(c[:nao]), transform.apply(c[nao:])
        moved[:nao] = element.spin[0, 0] * upper + element.spin[0, 1] * lower
        moved[nao:] = element.spin[1, 0] * upper + element.spin[1, 1] * lower
        u = np.conj(c).T @ (s2 @ moved)
        traces[k] = np.trace(u[block, block])
    sizes = np.array([len(m) for m in group.classes], dtype=float)
    weights = (np.conj(group.characters) * sizes[None, :]) @ traces / group.order
    assert np.allclose(weights, np.rint(np.real(weights)), atol=1e-6), weights
    assert np.all(np.rint(np.real(weights)) >= 0)
    assert int(np.sum(np.rint(np.real(weights)) * group.dimensions)) == 4


# --- The subduction ---------------------------------------------------------------------------

def test_the_subduction_reproduces_the_textbook_correlation():
    """Computed subduction, checked against the correlation tables by hand. ⚠ A multiplet that
    lands in *two* abelian sectors is exactly the case a per-irrep count can cut in half with
    every abelian check still passing, so this is the table that has to be right."""
    c2v = double_group("C2v")
    sub = c2v.subduction(C2Z)
    named = {c2v.irrep_names[r]: sorted(C2Z.irrep_name(t) for t in d)
             for r, d in sub.items()}
    assert named["A1"] == ["A"] and named["A2"] == ["A"]
    assert named["B1"] == ["B"] and named["B2"] == ["B"]
    assert named["E1/2"] == sorted(["1E1/2", "2E1/2"])

    d2h = double_group("D2h")
    sub = d2h.subduction(C2H_Z)
    named = {d2h.irrep_names[r]: sorted(C2H_Z.irrep_name(t) for t in d)
             for r, d in sub.items()}
    assert named["Ag"] == ["Ag"] and named["B1g"] == ["Ag"]
    assert named["B2g"] == ["Bg"] and named["B3u"] == ["Bu"]
    assert named["E1/2g"] == sorted(["1E1/2g", "2E1/2g"])


def test_every_subduction_conserves_the_dimension():
    """A theorem, on every supported group: an irrep restricted to a subgroup decomposes into
    exactly ``dim`` one-dimensional abelian ones."""
    for name in GROUPS:
        group = double_group(name)
        for abelian in (C1, CI, C2Z, CS_XY, C2H_Z):
            try:
                sub = group.subduction(abelian)
            except ValueError:
                continue                       # not a subgroup of this one; nothing to check
            for r, decomposition in sub.items():
                assert sum(decomposition.values()) == int(group.dimensions[r]), (name, r)


# --- U(g) on a CI vector -----------------------------------------------------------------------

def _one_body_matrix(space, kappa):
    """``sum_pq kappa_pq a_p^dag a_q`` in the determinant basis, from the strings themselves.

    Deliberately naive and deliberately independent of everything in ``kuiva.symm``: this is
    the reference the Givens factorization is checked against.
    """
    masks = [int(m) for m in space.masks]
    position = {m: i for i, m in enumerate(masks)}
    matrix = np.zeros((len(masks), len(masks)), dtype=np.complex128)
    for j, mj in enumerate(masks):
        for p in range(space.n_spinor):
            for q in range(space.n_spinor):
                if not (mj >> q) & 1:
                    continue
                intermediate = mj ^ (1 << q)
                if (intermediate >> p) & 1:
                    continue
                sign = (-1) ** (bin(mj & ((1 << q) - 1)).count("1")
                                + bin(intermediate & ((1 << p) - 1)).count("1"))
                matrix[position[intermediate | (1 << p)], j] += sign * kappa[p, q]
    return matrix


@pytest.mark.parametrize("n_spinor,n_elec", [(4, 2), (6, 3), (6, 4), (8, 3)])
def test_the_ci_orbital_rotation_matches_the_matrix_exponential(n_spinor, n_elec):
    """⚠ The check ``U(g)`` on a CI vector rests on. ``exp`` of the one-body operator built
    from the determinant strings is the same operator by definition and shares no code with
    the two-mode factorization; agreement to roundoff is the only acceptable result."""
    from scipy.linalg import expm
    rng = np.random.default_rng(3)
    space = CASSpace(n_spinor, n_elec)
    a = rng.normal(size=(n_spinor, n_spinor)) + 1j * rng.normal(size=(n_spinor, n_spinor))
    kappa = (a - np.conj(a).T) / 4.0
    reference = expm(_one_body_matrix(space, kappa))
    vectors = (rng.normal(size=(2, space.ndet)) + 1j * rng.normal(size=(2, space.ndet)))
    got = apply_orbital_rotation(space, expm(kappa), vectors)
    assert np.allclose(got, (reference @ vectors.T).T, atol=1e-11)


def test_the_ci_orbital_rotation_is_a_homomorphism_and_unitary():
    """``U(u1 u2) = U(u1) U(u2)`` and it preserves the norm — the two properties every use of
    it in the classification depends on."""
    from scipy.linalg import expm
    rng = np.random.default_rng(11)
    space = CASSpace(8, 5)
    def random_unitary():
        a = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        return expm((a - np.conj(a).T) / 4.0)
    u1, u2 = random_unitary(), random_unitary()
    v = rng.normal(size=space.ndet) + 1j * rng.normal(size=space.ndet)
    left = apply_orbital_rotation(space, u1 @ u2, v)
    right = apply_orbital_rotation(space, u1, apply_orbital_rotation(space, u2, v))
    assert np.allclose(left, right, atol=1e-11)
    assert abs(np.linalg.norm(left) - np.linalg.norm(v)) < 1e-11


def test_the_givens_factorization_reproduces_the_matrix():
    """The factorization itself, before it ever touches a determinant."""
    from scipy.linalg import expm
    rng = np.random.default_rng(5)
    a = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
    u = expm((a - np.conj(a).T) / 3.0)
    factors, diagonal = givens_factors(u)
    rebuilt = np.diag(diagonal)
    for p, g in reversed(factors):
        block = np.eye(7, dtype=np.complex128)
        block[p:p + 2, p:p + 2] = g
        rebuilt = block @ rebuilt
    assert np.allclose(rebuilt, u, atol=1e-12)
    # ⚠ Every factor acts on (p, p+1) and on nothing else: that is what makes the fermionic
    # phase of the swap +1 and removes the one place a sign error would be invisible.
    assert all(0 <= p < 6 for p, _ in factors)


# --- Detection ---------------------------------------------------------------------------------

def test_detection_is_frame_dependent_and_a_linear_molecule_is_flagged(cation_reference):
    """⚠ The molecule is never reoriented, so the group is the one the input frame has — and a
    linear molecule's true group is infinite, so what the layer uses is a finite subgroup and
    says so."""
    layout = cation_reference.data.ao_layout
    assert is_linear(layout)
    assert detect_point_group(layout) == "D6h"
    assert cation_reference.data.symmetry.full_group.name == "D6h"
    assert cation_reference.data.symmetry.axial_truncation


def test_a_group_the_geometry_does_not_have_is_refused():
    """⚠ A named classification group is verified, never assumed. Water has ``C2v`` and not
    ``D3h``, and a label read off an operation that is not a symmetry is a number with no
    meaning."""
    from kuiva.interface.pyscf_bridge import ao_layout, build_mole
    from kuiva.symm.assign import resolve_classification
    water = Molecule(atoms=[("O", (0.0, 0.0, 0.117)), ("H", (0.0, 0.757, -0.469)),
                            ("H", (0.0, -0.757, -0.469))], basis="x2c-SVPall-2c")
    layout = ao_layout(build_mole(water))
    assert detect_point_group(layout) == "C2v"
    with pytest.raises(ValueError, match="does not have every operation"):
        resolve_classification(layout, C2Z, "D3h")


def test_the_classification_group_must_contain_the_label_group(cation_reference):
    """⚠ Without the abelian label group *inside* the classification group there is no
    correspondence between their irreps at all, and the two would be two vocabularies rather
    than one. A linear molecule is where this bites: it has ``D6d`` and ``D6h`` alike, both of
    order 48, and only ``D6h`` contains the inversion the label group ``C2h(z)`` needs."""
    from kuiva.symm.assign import resolve_classification
    layout = cation_reference.data.ao_layout
    from kuiva.symm.double import has_group
    assert has_group(layout, "D6d")                       # the geometry does have it
    group, _ = resolve_classification(layout, C2H_Z, "auto")
    assert group.name == "D6h"
    assert double_group("D6h").order == double_group("D6d").order
    with pytest.raises(ValueError, match="not a subgroup"):
        resolve_classification(layout, C2H_Z, "D6d")


def test_the_candidate_list_is_ordered_largest_first():
    orders = [double_group(n).order for n in candidate_groups()]
    assert orders == sorted(orders, reverse=True)


# --- Classification of a real spectrum ----------------------------------------------------------

@pytest.fixture(scope="module")
def cation_states(cation_reference):
    return casci(cation_reference, active=CATION_ACTIVE, n_active_elec=CATION_ELEC,
                 n_states=4, report=False)


def test_the_multiplets_of_the_cation_are_the_ones_physics_predicts(cation_states):
    """⚠ **What this can fail on.** ``N2+`` has an ``X 2 Sigma g+`` ground state and an
    ``A 2 Pi u`` term above it; with spin-orbit coupling the ``Pi`` term splits into
    ``Omega = 3/2`` and ``1/2`` components. In the finite group the layer uses those are
    ``E1/2g`` and ``E3/2u``/``E1/2u`` — an assignment fixed by the spectroscopy of the
    molecule and by nothing in this program.
    """
    multiplets = cation_states.multiplets
    assert multiplets is not None
    assert multiplets[0] == multiplets[1] == "E1/2g"
    assert multiplets[2] == multiplets[3]
    assert multiplets[2] in ("E3/2u", "E1/2u")
    assert cation_states.classification.max_residual < 1e-8


def test_every_block_is_a_kramers_doublet_and_carries_one_multiplet(cation_states):
    """13 electrons: every level is at least doubly degenerate, and a doublet of a group whose
    fermion irreps are two-dimensional is exactly one of them."""
    classification = cation_states.classification
    for block in classification.blocks:
        assert block.size % 2 == 0
        assert block.classified
        assert sum(classification.group.dimensions[r] * m
                   for r, m in block.multiplicities.items()) == block.size


def test_classification_changes_no_number(cation_reference, cation_states):
    """The whole claim of the layer: it labels and never adapts."""
    plain = casci(cation_reference, active=CATION_ACTIVE, n_active_elec=CATION_ELEC,
                  n_states=4, report=False, classify=False)
    assert plain.multiplets is None
    assert np.array_equal(plain.energies, cation_states.energies)
    assert np.allclose(plain.gamma, cation_states.gamma, atol=0, rtol=0)


def test_an_active_space_the_group_takes_states_out_of_is_refused(cation_reference, caplog):
    """⚠ Half a degenerate shell is not a space any operator of the full group preserves, so
    there is no operator on its CI space at all. The layer says so and **degrades**: the
    calculation itself is unaffected, which is what stops a labelling from breaking a run."""
    with caplog.at_level(logging.WARNING, logger="kuiva"):
        # Spinors 8 and 9 are one Kramers pair of the fourfold pi_u shell: every operation
        # of the group that turns about z maps them onto the other pair, which is outside.
        result = casci(cation_reference, active=[8, 9], n_active_elec=1,
                       n_states=2, report=False)
    assert result.multiplets is None
    assert any("not closed under" in record.getMessage() for record in caplog.records)
    assert result.energies.size == 2                     # and the calculation still ran


def test_a_state_count_that_cuts_a_multiplet_is_refused(cation_reference):
    """⚠ **What the layer is for.** A block that is a fragment of a multiplet does not
    decompose into whole irreps, and the gate refuses rather than averaging over a fragment
    whose content depends on the eigensolver's arbitrary rotation inside it."""
    classifier = StateClassifier(
        cation_reference.data.symmetry.full_group, cation_reference.data.ao_layout,
        cation_reference.spinors_in_ao(),
        np.asarray(cation_reference.data.s_ao),
        _spaces(cation_reference), CASSpace(len(CATION_ACTIVE), CATION_ELEC))
    result = casci(cation_reference, active=CATION_ACTIVE, n_active_elec=CATION_ELEC,
                   n_states=4, report=False)
    # Take three of the four states: the last block is then half a Kramers doublet, which is
    # half a two-dimensional multiplet and decomposes into no whole irrep at all.
    partial = classifier.classify(result.vectors[:3], result.energies[:3])
    assert partial.unclassified
    with pytest.raises(ValueError, match="do not decompose into whole irreps"):
        assert_multiplet_boundary(partial)
    assert_multiplet_boundary(partial, on_split="warn")        # advisory form must not raise


def _spaces(reference):
    from kuiva.interface.api import active_space_for
    return active_space_for(reference, active=CATION_ACTIVE,
                            n_active_elec=CATION_ELEC).spaces
