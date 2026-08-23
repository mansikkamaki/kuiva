"""Tier-0/Tier-1 tests for abelian double-group symmetry.

What each group can actually fail on, which is the test of a check's worth:

* the **label algebra** against the group's own characters — composition, conjugation, the
  boson/fermion grading, and the spin-1/2 shift that turns a scalar label into a spinor one.
  These are cheap and they pin the vocabulary every other layer consumes;
* the **printed character table against the run's own operator matrices**. This is the check
  D-B7 exists for: ``tr U(g)`` computed from the AO representation matrices on the labelled
  orbitals must reproduce the characters the table prints, so a wrong axis convention or a
  wrong branch of the spinor representation shows up in the output rather than silently
  mislabelling everything. It shares no line of code with the analytic composition rule;
* the **labels against PySCF's own** ``orbsym`` on a molecule PySCF can label — a genuinely
  external implementation of the same classification, which is the only kind of agreement
  that can fail on this code alone;
* **refusals**: a geometry that does not have the operation asked for, an orbital set that is
  not symmetry-pure, an irrep that does not exist, a state count in an empty sector. Each of
  those otherwise produces a plausible number attached to a meaningless label;
* the **per-irrep selection against the general path**: the states a sector request returns
  must be exactly the states of that irrep in the plain lowest-n spectrum, on the same
  integrals. The general path is the reference path and this is the comparison that can fail
  on the new code alone;
* the **rotation mask**, asserted structurally (no inter-irrep parameters exist) *and* by
  outcome (a masked CASSCF reaches the unconstrained energy where the symmetry is not broken);
* the **tensor network**, where the label is a conserved quantum number rather than a
  classification: a labelled solve must reproduce the exact CI in the sector it targets, and
  a different sector must reproduce that sector's lowest CI root — which is a statement no
  amount of internal consistency can substitute for.
"""
import ast
import logging
from pathlib import Path

import numpy as np
import pytest

from kuiva.ci.davidson import davidson, davidson_sector
from kuiva.dmrg.block import QuantumNumber
from kuiva.interface.api import (Molecule, active_space_for, casci, casscf,
                                 spinor_reference)
from kuiva.mcscf.casci import FullCISolver
from kuiva.mcscf.orbopt import CASIntegrals, OrbitalSpaces
from kuiva.symm import (analyze, ao_operation, detect_operations, group_operations,
                        label_scalar_orbitals, resolve_group)
from kuiva.symm.groups import C1, C2H_Z, C2Z, CI, CS_XY, GROUPS
from kuiva.symm.report import character_text
from kuiva.symm.sectors import (SectorTable, assert_sector_symmetry, determinant_labels,
                                mode_bases, resolve_state_request, sector_violation)

#: Agreement demanded between a per-irrep solve and the same states picked out of the general
#: lowest-n spectrum. They are eigenvalues of the same operator reached by two routes, so the
#: only difference is the Davidson residual's second-order effect on a Ritz value.
ENERGY_TOL = 1e-10


# --- Systems ---------------------------------------------------------------------------------

def n2_molecule(point_group="auto"):
    """N2 on the z axis: D(inf)h, whose largest abelian-double-group subgroup here is C2h(z)."""
    return Molecule(atoms=[("N", (0.0, 0.0, 0.55)), ("N", (0.0, 0.0, -0.55))],
                    basis="x2c-SVPall-2c", point_group=point_group)


@pytest.fixture(scope="module")
def n2_reference():
    # screening="none": the symmetry work changes no scalar quantity, so the four-component
    # atomic solve the default would pay for is pure cost here (and the suite may not depend
    # on a warm cache).
    return spinor_reference(n2_molecule(), screening="none", memory_gb=6.0)


@pytest.fixture(scope="module")
def n2_active(n2_reference):
    space = active_space_for(n2_reference, active=list(range(8, 16)), n_active_elec=6)
    ints = CASIntegrals.build(n2_reference.factors, n2_reference.h_one_electron(),
                              n2_reference.spinors_in_ao(), space.spaces,
                              e_nuc=n2_reference.data.e_nuc)
    return space, ints


# --- Dependency direction, asserted from the sources ------------------------------------------

REPO = Path(__file__).resolve().parent.parent


def _module_files(root: Path):
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def test_the_label_vocabulary_depends_on_no_consumer():
    """⚠ Nothing in :mod:`kuiva.symm` may import a package that consumes labels.

    A shared primitive living inside its first consumer makes every later consumer look like a
    special case of it, and there are four here — the front end, the CI, the orbital optimizer
    and the tensor network. The one exception is deliberate and is checked to stay function-
    local: the tensor network's ``QuantumNumber``/``ModeBasis`` are *constructed* on its behalf,
    so ``kuiva.dmrg`` is imported **inside** the function that builds them and never at module
    scope, which is what keeps ``import kuiva.symm`` from dragging the network layer in.
    """
    forbidden = ("ci", "mcscf", "rdm", "dmrg", "interface", "props", "pt", "qc", "amf",
                 "integrals", "extras", "io", "orth", "x2c")
    offenders = []
    for path in _module_files(REPO / "kuiva" / "symm"):
        tree = ast.parse(path.read_text())
        # direct children of the module body only -- an import nested in a function is the
        # deliberate exception this test exists to allow
        module_level = {id(n) for n in tree.body
                        if isinstance(n, (ast.Import, ast.ImportFrom))}
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[1] for a in node.names
                         if a.name.split(".")[0] == "kuiva" and "." in a.name]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level > 0:
                    names = [module.split(".")[0]] if module else []
                elif module.startswith("kuiva."):
                    names = [module.split(".")[1]]
            if id(node) not in module_level:
                continue                              # function-local: the declared exception
            for name in names:
                if name in forbidden:
                    offenders.append((path.relative_to(REPO).as_posix(), name))
    assert offenders == [], (
        "kuiva.symm must not import a consumer of the labels at module scope: {}"
        .format(offenders))


def test_importing_the_vocabulary_costs_nothing():
    """``import kuiva.symm`` must not pull in PySCF, the CI or the tensor network."""
    import subprocess
    import sys as _sys
    code = ("import sys, kuiva.symm; "
            "print([m for m in ('pyscf', 'kuiva.dmrg', 'kuiva.mcscf', 'kuiva.ci', "
            "'kuiva.interface') if m in sys.modules])")
    out = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(REPO))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout


# --- The label vocabulary --------------------------------------------------------------------

@pytest.mark.parametrize("group", list(GROUPS.values()), ids=lambda g: g.name)
def test_every_label_composes_and_conjugates_inside_the_group(group):
    labels = group.labels()
    assert len(labels) == group.order
    for a in labels:
        assert group.compose(a, group.identity()) == a
        assert group.compose(a, group.conjugate(a)) == group.identity()
        assert group.conjugate(group.conjugate(a)) == a
        # conjugation preserves the grading: a Kramers partner is a spinor too
        assert group.is_fermion(a) == group.is_fermion(group.conjugate(a))


@pytest.mark.parametrize("group", list(GROUPS.values()), ids=lambda g: g.name)
def test_the_character_table_is_orthogonal(group):
    """Row orthogonality of the stored table — the definition of a character table, and the
    cheapest thing that fails if a name, a modulus or a generator order is wrong."""
    elements = group.elements()
    chi = np.array([[group.character(lab, e) for e in elements] for lab in group.labels()])
    gram = chi.conj() @ chi.T / len(elements)
    assert np.allclose(gram, np.eye(len(group.labels())), atol=1e-12)


@pytest.mark.parametrize("group", list(GROUPS.values()), ids=lambda g: g.name)
def test_exactly_half_the_irreps_are_fermion_irreps(group):
    """``chi(Ebar) = -1`` on exactly half of them: the double group is a two-fold central
    extension, so a group that got its grading wrong fails here rather than at a CI solve."""
    fermion = group.labels(fermion=True)
    assert len(fermion) * 2 == group.order
    for lab in fermion:
        ebar = next(e for e in group.elements() if group.element_name(e) == "Ebar")
        assert group.character(lab, ebar).real == pytest.approx(-1.0)


@pytest.mark.parametrize("group", list(GROUPS.values()), ids=lambda g: g.name)
def test_irrep_names_are_unique_and_round_trip(group):
    names = [group.irrep_name(t) for t in group.labels()]
    assert len(set(names)) == len(names)
    for lab, name in zip(group.labels(), names):
        assert group.label_of(name) == lab
        assert group.label_of(name.lower()) == lab
        assert group.label_of(lab) == lab


@pytest.mark.parametrize("group", list(GROUPS.values()), ids=lambda g: g.name)
def test_a_scalar_orbital_expands_into_a_conjugate_pair_of_fermion_labels(group):
    """The spin-1/2 shift is what turns a boson label into two spinor labels, and the two
    must be conjugate — time reversal is antiunitary and commutes with every spatial
    operation. A wrong shift breaks this and nothing else."""
    boson = np.array(group.labels(fermion=False), dtype=int)
    spinor = group.spinor_labels(boson)
    assert spinor.shape == (2 * boson.shape[0], group.width)
    for k in range(boson.shape[0]):
        a, b = tuple(spinor[2 * k]), tuple(spinor[2 * k + 1])
        assert group.is_fermion(a) and group.is_fermion(b)
        assert group.conjugate(a) == b


def test_characters_render_in_ascii():
    """The output stream is ASCII, and a character table full of Unicode is a table
    nobody can grep over ssh."""
    for chi, text in ((1 + 0j, "1"), (-1 + 0j, "-1"), (1j, "i"), (-1j, "-i")):
        assert character_text(chi) == text
        assert text.isascii()


def test_a_larger_group_reduces_to_its_abelian_double_subgroup():
    """C2v, D2 and D2h have two-dimensional fermion irreps, which one integer cannot label."""
    for name, expected, reduced in (("D2h", "C2h(z)", True), ("D2", "C2(z)", True),
                                    ("C2v", "C2(z)", True), ("C2h", "C2h(z)", False),
                                    ("Ci", "Ci", False), ("C1", "C1", False)):
        group, was_reduced = resolve_group(name)
        assert group.name == expected
        assert was_reduced is reduced


def test_an_unknown_group_is_refused_naming_what_exists():
    with pytest.raises(ValueError, match="unknown point group"):
        resolve_group("Oh")


# --- Operator matrices and detection ---------------------------------------------------------

def test_the_ao_signs_reproduce_the_known_p_orbital_behaviour(n2_reference):
    """p functions are where the ``m`` convention bites: the integral library lays a p shell
    out as ``px, py, pz`` (``m = +1, -1, 0``), not as ``-l..+l``. Deriving the sign from the
    position in the shell instead mislabels every p function while leaving every shell *sum*
    unchanged — invisible in any total, wrong in every row."""
    from kuiva.symm.operators import ao_signs
    layout = n2_reference.ao_layout
    is_p = layout.ao_l == 1
    for spatial, expect in (((1, 0), {1: -1.0, -1: -1.0, 0: 1.0}),      # C2(z)
                            ((0, 1), {1: -1.0, -1: -1.0, 0: -1.0}),     # i
                            ((1, 1), {1: 1.0, -1: 1.0, 0: -1.0})):      # sigma(xy)
        sign = ao_signs(layout, spatial)
        for m, value in expect.items():
            sel = is_p & (layout.ao_m == m)
            assert np.all(sign[sel] == value)


def test_detection_is_frame_dependent_and_says_so_by_finding_less():
    """⚠ The molecule is never reoriented, so the same molecule in two frames is labelled by
    two different groups. That is the documented behaviour and the cost of not moving the
    gauge origin, and it is asserted rather than left to the docstring.

    C2v water, twice: with the two-fold axis along ``z`` the detected operation is ``C2(z)``;
    with it along ``x`` the axis is invisible here and what is left is the molecular plane,
    ``sigma(xy)``. Neither group contains the other, so this is a genuine relabelling and not
    merely a coarser one.
    """
    from kuiva.interface.pyscf_bridge import ao_layout, build_mole
    on_z = Molecule(atoms=[("O", (0.0, 0.0, 0.116)), ("H", (0.0, 0.75, -0.47)),
                           ("H", (0.0, -0.75, -0.47))], basis="x2c-SVPall-2c")
    on_x = Molecule(atoms=[("O", (0.116, 0.0, 0.0)), ("H", (-0.47, 0.75, 0.0)),
                           ("H", (-0.47, -0.75, 0.0))], basis="x2c-SVPall-2c")
    assert detect_operations(ao_layout(build_mole(on_z))) == ("C2(z)",)
    assert detect_operations(ao_layout(build_mole(on_x))) == ("sigma(xy)",)


def test_a_group_the_geometry_does_not_have_is_refused(n2_reference):
    """Water with its C2 axis along ``z`` has no inversion; asking for C2h must refuse rather
    than label everything ``g``."""
    from kuiva.interface.pyscf_bridge import ao_layout, build_mole
    water = Molecule(atoms=[("O", (0.0, 0.0, 0.116)), ("H", (0.0, 0.75, -0.47)),
                            ("H", (0.0, -0.75, -0.47))], basis="x2c-SVPall-2c")
    layout = ao_layout(build_mole(water))
    with pytest.raises(ValueError, match="does not have i"):
        group_operations(layout, C2H_Z)


def test_a_per_atom_basis_breaks_a_symmetry_the_geometry_has():
    """Two atoms are images of each other only if they carry the same basis, or a symmetric
    geometry with an asymmetric basis would be labelled with an operation that is not one."""
    from kuiva.interface.pyscf_bridge import ao_layout, build_mole
    mol = Molecule(atoms=[("N", (0.0, 0.0, 0.55)), ("N", (0.0, 0.0, -0.55))],
                   basis={1: "x2c-SVPall-2c", 2: "x2c-TZVPall-2c"})
    ops = detect_operations(ao_layout(build_mole(mol)))
    assert "i" not in ops and "sigma(xy)" not in ops
    assert ops == ("C2(z)",)


# --- The self-consistency of the printed table ------------------------------------------------

def test_the_printed_characters_are_reproduced_by_the_operator_matrices(n2_reference):
    """⚠ **The check D-B7's table exists for.** ``<psi|U(g)|psi>`` computed from the AO
    representation matrices — spatial permutation-and-sign times the spin-1/2 factor — must
    equal the character the table prints for that orbital's label, for **every** element of
    the group including the barred ones. It shares no code with the analytic composition rule
    in :meth:`Group.spinor_labels`, so a wrong branch of the spinor representation, a wrong
    axis or a wrong stored table breaks it.
    """
    data = n2_reference.data
    symmetry = data.symmetry
    group = symmetry.group
    layout = data.ao_layout
    labels = symmetry.spinor_labels()

    c = n2_reference.spinors_in_ao()
    s2 = np.kron(np.eye(2), np.asarray(data.s_ao))
    ops = {}
    for spatial in ((0, 0), (1, 0), (0, 1), (1, 1)):
        op = ao_operation(layout, spatial)
        if op is not None:
            ops[spatial] = op.two_component()

    for element in group.elements():
        spatial, ebar = group.element_spatial(element)
        u = ops[spatial] * (-1.0 if ebar else 1.0)
        measured = np.einsum("ip,ij,jp->p", c.conj(), s2 @ u, c)
        expected = np.array([group.character(t, element) for t in labels.tuples()])
        assert np.allclose(measured, expected, atol=1e-8), group.element_name(element)


def test_the_labels_agree_with_pyscfs_own_orbsym():
    """An external implementation of the same classification. PySCF is allowed to reorient
    for this — it is a separate ``Mole`` built only for the comparison — which is exactly
    what Kuiva refuses to do to the ``Mole`` its own operators come from."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, symm
    from kuiva.interface.pyscf_bridge import ao_layout, build_mole

    mol = gto.M(atom="N 0 0 0.55; N 0 0 -0.55", basis="sto-3g", symmetry="D2h",
                verbose=0)
    mf = mol.RHF().run()
    orbsym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)

    # the layout is built from the same Mole PySCF labelled, so the two are certainly in one
    # frame; Kuiva's own front end never lets PySCF reorient, which is the point of D-B1
    layout = ao_layout(mol)
    labels, _, _ = label_scalar_orbitals(np.asarray(mf.mo_coeff),
                                         np.asarray(mol.intor("int1e_ovlp")),
                                         layout, C2H_Z, mo_energy=np.asarray(mf.mo_energy))
    names = [C2H_Z.irrep_name(tuple(row)) for row in labels]
    # D2h names map onto the C2h(z) subgroup by their behaviour under C2(z) and i:
    # Ag/B1g -> Ag, B2g/B3g -> Bg, Au/B1u -> Au, B2u/B3u -> Bu.
    subduced = {"Ag": "Ag", "B1g": "Ag", "B2g": "Bg", "B3g": "Bg",
                "Au": "Au", "B1u": "Au", "B2u": "Bu", "B3u": "Bu"}
    assert names == [subduced[str(x)] for x in orbsym]


def test_orbitals_that_are_not_symmetry_pure_are_refused(n2_reference):
    """A mixture of two irreps is an eigenvector of nothing, and a label read off it would be
    the dominant component dressed up as a quantum number."""
    data = n2_reference.data
    c = np.array(data.mo_coeff, dtype=float, copy=True)
    labels = data.symmetry.scalar[0].labels
    # find two orbitals of different irreps and mix them; a degenerate-block rotation cannot
    # undo it because they are not degenerate
    i, j = next((a, b) for a in range(c.shape[1]) for b in range(a + 1, c.shape[1])
                if not np.array_equal(labels[a], labels[b]))
    mix = (c[:, i] + c[:, j]) / np.sqrt(2.0)
    c[:, i] = mix
    with pytest.raises(ValueError, match="not an eigenvector"):
        label_scalar_orbitals(c, np.asarray(data.s_ao), data.ao_layout, C2H_Z,
                              mo_energy=np.asarray(data.mo_energy))


def test_symmetry_adaptation_changes_no_observable():
    """The one thing that makes rotating a degenerate block legitimate: the occupied density
    is invariant, so the SCF energy and every observable are."""
    from kuiva.interface.pyscf_bridge import run_scalar_x2c
    mol = n2_molecule(point_group=None)
    plain = run_scalar_x2c(mol, screening="none", memory_gb=6.0)
    labelled = run_scalar_x2c(n2_molecule("auto"), screening="none", memory_gb=6.0)
    occ = plain.mo_occ > 0
    d_plain = (plain.mo_coeff[:, occ] * plain.mo_occ[occ]) @ plain.mo_coeff[:, occ].T
    d_lab = (labelled.mo_coeff[:, occ] * labelled.mo_occ[occ]) @ labelled.mo_coeff[:, occ].T
    assert np.max(np.abs(d_plain - d_lab)) < 1e-10
    assert labelled.e_scf == pytest.approx(plain.e_scf, abs=1e-12)


def test_an_unrestricted_reference_labels_both_spin_sets():
    """⚠ An unrestricted spinor set is orthonormal but **not** Kramers paired: spinor ``2p``
    is the ``p``-th alpha orbital and ``2p+1`` the ``p``-th beta one, two different orbitals.
    So the two spin shifts are applied to two different label sets, and the pair is *not*
    conjugate — the same statement as ``kramers_paired = False``, and the reason this cannot
    go through the restricted path.
    """
    mol = Molecule(atoms=[("O", (0.0, 0.0, 0.6)), ("H", (0.0, 0.0, -0.4))],
                   basis="x2c-SVPall-2c", charge=0, spin=1, point_group="auto")
    ref = spinor_reference(mol, reference="uhf", screening="none", memory_gb=6.0)
    symmetry = ref.data.symmetry
    assert symmetry.unrestricted and len(symmetry.scalar) == 2
    labels = ref.spinor_labels
    assert len(labels) == ref.nspinor
    assert not ref.spinors.kramers_paired
    group = labels.group
    # every spinor carries a fermion label, whichever spin set it came from
    assert all(group.is_fermion(t) for t in labels.tuples())


# --- Determinant sectors -----------------------------------------------------------------

def test_a_determinants_label_is_the_group_sum_of_its_occupied_spinors():
    labels = np.array([[1], [3], [1], [3]], dtype=int)            # C2(z) fermion labels
    occ = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0]], dtype=bool)
    got = determinant_labels(occ, labels, C2Z.moduli)
    assert [tuple(r) for r in got] == [(0,), (2,), (1,)]


def test_the_sectors_partition_the_determinant_space(n2_active):
    space, _ = n2_active
    from kuiva.ci.strings import CASSpace
    cas = CASSpace(8, 6, build_map=False)
    table = SectorTable.build(cas.occupations(), space.labels.labels, space.labels.group)
    assert sum(table.sizes().values()) == cas.ndet
    assert set(np.unique(table.sector_of)) == set(range(table.n_sectors))


def test_an_empty_or_unknown_sector_is_refused_naming_the_available_ones(n2_active):
    space, _ = n2_active
    from kuiva.ci.strings import CASSpace
    cas = CASSpace(8, 6, build_map=False)
    table = SectorTable.build(cas.occupations(), space.labels.labels, space.labels.group)
    with pytest.raises(ValueError, match="is not an irrep"):
        resolve_state_request({"Xx": 1}, table)
    with pytest.raises(ValueError, match="holds no determinant"):
        resolve_state_request({"1E1/2g": 1}, table)      # even N: no fermion sectors exist
    with pytest.raises(ValueError, match="either requested or left out"):
        resolve_state_request({"Ag": 0}, table)


def test_symmetry_pure_integrals_conserve_the_sector_and_broken_ones_do_not(n2_active):
    space, ints = n2_active
    h = np.ascontiguousarray(ints.h_active_effective())
    eri = ints.active_eri()
    labels, moduli = space.labels.labels, space.labels.group.moduli
    err_h, err_eri = sector_violation(h, eri, labels, moduli)
    assert max(err_h, err_eri) < 1e-10
    assert_sector_symmetry(h, eri, labels, moduli)
    # a control that must fail, or the check is measuring nothing
    broken = h.copy()
    i, j = next((a, b) for a in range(h.shape[0]) for b in range(h.shape[0])
                if not np.array_equal(labels[a], labels[b]))
    broken[i, j] = broken[j, i] = 0.5 * np.max(np.abs(h))
    with pytest.raises(ValueError, match="do not conserve the irrep label"):
        assert_sector_symmetry(broken, eri, labels, moduli)


# --- The sector-restricted eigensolver -----------------------------------------------------

def test_a_sector_solve_reproduces_that_sectors_roots_of_the_full_spectrum():
    """A dense control, where 'the sector's roots' is unambiguous."""
    rng = np.random.default_rng(11)
    n = 60
    mask = np.zeros(n, dtype=bool)
    mask[::3] = True
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    a = a + a.conj().T
    a[np.ix_(mask, ~mask)] = 0.0                     # block diagonal over the two sectors
    a[np.ix_(~mask, mask)] = 0.0
    a += np.diag(np.arange(n, dtype=float))
    inside = np.linalg.eigvalsh(a[np.ix_(mask, mask)])
    got = davidson_sector(lambda c: a @ c, np.real(np.diag(a)), mask, 4, conv_tol=1e-11)
    assert np.allclose(got.energies, inside[:4], atol=1e-9)
    # the vectors come back full length, zero outside the sector
    assert got.vectors.shape[1] == n
    assert np.max(np.abs(got.vectors[:, ~mask])) == 0.0


def test_a_sector_solve_cannot_miss_the_sector_a_biased_guess_misses():
    """⚠ The **recorded** converged-but-not-lowest failure, reproduced and then fixed
    structurally.

    A Krylov method cannot leave the invariant subspaces its starting vectors lie in. Here
    every low **diagonal** element sits in sector A while sector B's strong internal coupling
    puts its ground state 40 units *below* everything in A — so a guess made of unit vectors on
    the lowest diagonals converges, reports converged residuals, and returns the wrong three
    lowest eigenvalues. Nothing about the answer looks wrong.

    Solving sector B on its own cannot fail this way, because the sector *is* the invariant
    subspace that was being missed. That is the whole claim, and it is asserted against a dense
    ``eigvalsh`` of the same matrix rather than against the solver's own opinion.
    """
    rng = np.random.default_rng(5)
    n = 48
    mask = np.zeros(n, dtype=bool)
    mask[n // 2:] = True
    diagonal = np.concatenate([np.arange(n // 2, dtype=float),
                               50.0 + np.arange(n // 2, dtype=float)])
    a = np.diag(diagonal).astype(np.complex128)
    coupling = (rng.standard_normal((n // 2, n // 2))
                + 1j * rng.standard_normal((n // 2, n // 2)))
    a[n // 2:, n // 2:] += 6.0 * (coupling + coupling.conj().T)
    exact = np.linalg.eigvalsh(a)

    # a guess that spans only sector A -- the biased set, and supplying it is what suppresses
    # the generic starting vectors that otherwise mitigate this
    guess = np.zeros((6, n), dtype=np.complex128)
    for k in range(6):
        guess[k, k] = 1.0
    biased = davidson(lambda c: a @ c, diagonal, 3, guess=guess, conv_tol=1e-10,
                      dense_max_det=0)
    assert biased.converged
    assert biased.energies[0] > exact[0] + 1.0          # converged, and not the lowest

    found = davidson_sector(lambda c: a @ c, diagonal, mask, 1, conv_tol=1e-10)
    assert found.energies[0] == pytest.approx(exact[0], abs=1e-9)


def test_per_irrep_selection_reproduces_a_dense_diagonalization_of_the_sector(n2_active):
    """The exit criterion, against a reference that shares nothing with the sector solver:
    the explicit Hamiltonian matrix, sliced to the sector's determinants and diagonalized.

    That is a genuinely independent route — no Davidson, no compression, no guess — so it can
    fail on the sector machinery alone. It also asserts the blocking itself: the union of the
    per-sector spectra must be the whole spectrum, which is what "``H`` is block diagonal over
    the sectors" means numerically.
    """
    from kuiva.ci.strings import CASSpace, hamiltonian_matrix

    space, ints = n2_active
    cas = CASSpace(8, 6, build_map=False)
    h = np.ascontiguousarray(ints.h_active_effective())
    eri = ints.active_eri()
    dense = np.asarray(hamiltonian_matrix(cas.determinants(), h, eri).todense())
    table = SectorTable.build(cas.occupations(), space.labels.labels, space.labels.group)

    want = {"Ag": 2, "Bu": 3}
    solver = FullCISolver(8, 6, n_states=want, symmetry=space.labels, enforce_kramers=False)
    solver.solve(ints)

    expected, union = [], []
    for name in table.sectors:
        idx = table.indices(name)
        block = np.linalg.eigvalsh(dense[np.ix_(idx, idx)])
        union.extend(block)
        label = table.name(name)
        if label in want:
            expected.extend(block[:want[label]])
    assert np.allclose(np.sort(union), np.linalg.eigvalsh(dense), atol=1e-9)
    assert np.allclose(np.sort(expected), solver.last.energies, atol=ENERGY_TOL)
    assert sorted(solver.last.irreps) == sorted(
        [n for n, k in want.items() for _ in range(k)])


def test_a_degenerate_block_of_two_sectors_is_classified_as_the_block(n2_active):
    """⚠ A single state inside a degenerate block has no sector, and the classification says
    so rather than picking the dominant half. Two conjugate sectors meeting at one energy is
    the normal case for a molecule whose real group is bigger than the abelian one used, and
    the eigensolver may return any rotation of the block — which is freedom, not impurity, so
    the leakage stays zero.
    """
    space, ints = n2_active
    solver = FullCISolver(8, 6, n_states=4, symmetry=space.labels, enforce_kramers=False)
    solver.solve(ints)
    assert solver.last.sector_leakage < 1e-10
    # every reported name is either one irrep or an explicit sum of the block's irreps
    for name in solver.last.irreps:
        for part in name.split(" + "):
            assert part in [space.labels.group.irrep_name(t)
                            for t in space.labels.group.labels()]


def test_a_per_irrep_solve_costs_fewer_applications_of_h(n2_active):
    """The other half of the point: a sector's ``n`` roots cost ``n`` roots, not however many
    roots of the full spectrum happen to lie below them."""
    space, ints = n2_active
    general = FullCISolver(8, 6, n_states=5, symmetry=space.labels, enforce_kramers=False)
    general.solve(ints)
    per_irrep = FullCISolver(8, 6, n_states={"Ag": 2, "Bu": 3}, symmetry=space.labels,
                             enforce_kramers=False)
    per_irrep.solve(ints)
    assert per_irrep.last.n_apply < general.last.n_apply


def test_the_state_table_names_the_irrep_and_reports_the_leakage(n2_active):
    space, ints = n2_active
    solver = FullCISolver(8, 6, n_states=4, symmetry=space.labels, enforce_kramers=False)
    result = solver.solve_active(np.ascontiguousarray(ints.h_active_effective()),
                                 ints.active_eri(), e_core=ints.e_core)
    assert result.irreps is not None and len(result.irreps) == 4
    assert result.sector_leakage < 1e-12


def test_a_per_irrep_request_without_labels_is_refused():
    with pytest.raises(ValueError, match="needs irrep labels"):
        FullCISolver(8, 6, n_states={"Ag": 1})


def test_kramers_restricted_and_per_irrep_selection_combine_over_conjugate_pairs(n2_active):
    """⚠ The two symmetries commute, but they do not act on the same object: time reversal
    **conjugates** an irrep label, so a sector is time-reversal-closed only when it is
    self-conjugate and otherwise only the union of a conjugate pair is. The combination is
    therefore stated over pairs, and asking for ``n`` states of an irrep returns the ``n``
    time-reversed partners in its conjugate as well.

    ``tests/test_symmetry_kramers.py`` owns the combination cases; this is the one assertion
    that belongs beside the per-irrep selection itself.
    """
    space, ints = n2_active
    labels = space.labels
    solver = FullCISolver(len(labels), 5, n_states={"1E1/2u": 2}, symmetry=labels,
                          kramers="restricted")
    assert solver.n_states == 4
    with pytest.raises(ValueError, match="conjugate pair"):
        FullCISolver(len(labels), 5, n_states={"1E1/2u": 2, "2E1/2u": 2}, symmetry=labels,
                     kramers="restricted")


def test_the_symmetry_mode_is_part_of_the_space_key(n2_active):
    """A per-irrep solver is a different surface, so the optimizer's curvature memory must
    not be transported onto it — and a solver *without* one keeps the key it always had, so
    no checkpoint written before symmetry existed is silently downgraded to a cold start."""
    space, _ = n2_active
    plain = FullCISolver(8, 6, n_states=2)
    classified = FullCISolver(8, 6, n_states=2, symmetry=space.labels)
    selected = FullCISolver(8, 6, n_states={"Ag": 2}, symmetry=space.labels,
                            enforce_kramers=False)
    assert plain.space_key() == classified.space_key()
    assert selected.space_key() != plain.space_key()
    assert "Ag=2" in selected.space_key()


def test_the_boundary_diagnostic_reports_the_tightest_sector(n2_active):
    from kuiva.mcscf.casci import state_average_boundary
    space, ints = n2_active
    solver = FullCISolver(8, 6, n_states={"Ag": 1, "Bu": 2}, symmetry=space.labels,
                          enforce_kramers=False)
    solver.solve(ints)
    report = state_average_boundary(solver, ints, margin=2, where="test orbitals")
    assert report.sector in ("Ag", "Bu")
    assert report.gap_cm is not None and report.gap_cm > 0.0


# --- The orbital rotation mask ---------------------------------------------------------------

def test_the_rotation_mask_removes_every_inter_irrep_parameter(n2_active, n2_reference):
    space, _ = n2_active
    labels = n2_reference.spinor_labels.labels
    spaces = space.spaces
    rows, cols = spaces.rotation_pairs()
    m_rows, m_cols = spaces.rotation_pairs(labels=labels)
    assert m_rows.size < rows.size
    assert np.all(labels[m_rows] == labels[m_cols])
    # and nothing legal was dropped
    keep = np.all(labels[rows] == labels[cols], axis=1)
    assert m_rows.size == int(keep.sum())


@pytest.mark.slow
def test_a_masked_casscf_reaches_the_unconstrained_energy(n2_reference):
    """Where the symmetry is not spontaneously broken the constraint costs nothing, and the
    labels are then exact at convergence rather than only at the start."""
    kwargs = dict(active=list(range(8, 16)), n_active_elec=6, n_states=1, max_iter=40,
                  report=False)
    masked = casscf(n2_reference, preserve_symmetry=True, **kwargs)
    free = casscf(n2_reference, **kwargs)
    assert masked.converged and free.converged
    assert masked.energy == pytest.approx(free.energy, abs=1e-7)
    assert masked.ci.sector_leakage < 1e-12


# --- The tensor-network quantum number -------------------------------------------------------

def test_a_cyclic_quantum_number_reduces_and_propagates_its_moduli():
    """⚠ A finite cyclic group is not a subgroup of the integers: without the modulus, terms
    the group allows (a shift of exactly one full turn) look forbidden, and every one of them
    would be dropped from the Hamiltonian while staying Hermitian and plausible."""
    a = QuantumNumber(1, 3, moduli=(None, 4))
    b = QuantumNumber(1, 3, moduli=(None, 4))
    assert tuple(a + b) == (2, 2)
    assert (a + b).moduli == (None, 4)
    assert tuple(a + b + b) == (3, 1)
    assert tuple(-a) == (-1, 1)
    # an unlabelled identity still works, which is what keeps every existing call site valid
    assert tuple(a + QuantumNumber.zero(2)) == (1, 3)
    with pytest.raises(ValueError, match="moduli differ"):
        a + QuantumNumber(1, 1, moduli=(None, 2))


def test_mode_bases_carry_the_label_of_the_occupied_spinor():
    labels = np.array([[1], [3]], dtype=int)
    bases = mode_bases(labels, C2Z)
    assert tuple(bases[0].charges[0]) == (0, 0)
    assert tuple(bases[0].charges[1]) == (1, 1)
    assert tuple(bases[1].charges[1]) == (1, 3)
    assert bases[0].charges[1].moduli == (None, 4)


@pytest.mark.slow
def test_a_labelled_network_reproduces_the_exact_ci_in_the_sector_it_targets(n2_active):
    """The claim the widening is for: the label is a **conserved quantum number** of the
    network, not a classification of its output. A solve targeted at one sector must equal
    that sector's lowest CI root, and the default target must equal the global ground state.
    """
    from kuiva.dmrg.solver import DMRGSolver
    space, ints = n2_active
    exact = FullCISolver(8, 6, n_states=1)
    e_exact, gamma, _ = exact.solve(ints)

    network = DMRGSolver(6, max_bond=64, n_roots=1, max_sweeps=40,
                         symmetry=space.labels, seed=3)
    e_net, gamma_net, _ = network.solve(ints)
    assert e_net == pytest.approx(e_exact, abs=1e-10)
    assert np.max(np.abs(gamma_net - gamma)) < 1e-8
    assert tuple(network._state.charge)[0] == 6

    excited = FullCISolver(8, 6, n_states={"Bu": 1}, symmetry=space.labels,
                           enforce_kramers=False)
    e_bu, _, _ = excited.solve(ints)
    targeted = DMRGSolver(6, max_bond=64, n_roots=1, max_sweeps=40,
                          symmetry=space.labels, sector="Bu", seed=3)
    e_net_bu, _, _ = targeted.solve(ints)
    assert e_net_bu == pytest.approx(e_bu, abs=1e-10)
    assert e_net_bu > e_net
