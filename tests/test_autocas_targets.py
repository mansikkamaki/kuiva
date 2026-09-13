"""Tier 0: the automatic-selection vocabulary, its refusals, and the one-way dependency.

Why these tests and not others
------------------------------
A target is a *statement*, and the two things that can go wrong with a statement are both
silent. It can be **misread** -- ``("shell", "Dy")`` taken as one centre when the molecule has
two, or ``"Dy"`` split into two atom keys ``"D"`` and ``"y"`` that name nothing -- which gives
a perfectly well-formed active space that is not the one asked for. And it can be
**unreproducible**: a description carrying spinor indices says nothing outside the orbital set
it came from, so a stored product built on one is not checkable at all.

Every test here fails on one of those. The round-trip tests are the sharp form of the first
(a spelling that parses to something it cannot write back is a spelling that was misread), and
the no-index test is the sharp form of the second.

Pure text and geometry throughout: no SCF, no integrals, so this runs in the default suite --
which is where a check on a user-facing vocabulary has to run.
"""
import ast
import re
from pathlib import Path

import numpy as np
import pytest

from kuiva.autocas import candidates as cand
from kuiva.autocas import targets as tg
from kuiva.basis.layout import Shell, build_layout

REPO = Path(__file__).resolve().parents[1]


def _layout(atoms):
    """A minimal :class:`AOLayout` -- symbols, coordinates [bohr] and one s shell each.

    Enough for the atom addressing and the contact detection, which are the parts of the
    vocabulary that touch the molecule at all.
    """
    symbols = [a for a, _ in atoms]
    coords = np.array([xyz for _, xyz in atoms], dtype=float)
    shells = [Shell(atom=i, l=0, exponents=np.array([1.0]), coefficients=np.array([1.0]))
              for i in range(len(atoms))]
    return build_layout(symbols, [1.0] * len(atoms), coords, shells,
                        ["1s"] * len(atoms))


#: Ti2Cl6's connectivity, in bohr: two titaniums, two bridging chlorines, four terminal ones.
#: The same model geometry the validation suite's dimer uses, which is what makes the contact
#: detection test a statement about a real case.
ANG = 1.0 / 0.52917721092
TI2CL6 = _layout([("Ti", (+1.75 * ANG, 0.0, 0.0)), ("Ti", (-1.75 * ANG, 0.0, 0.0)),
                  ("Cl", (0.0, +1.715 * ANG, 0.0)), ("Cl", (0.0, -1.715 * ANG, 0.0)),
                  ("Cl", (+3.435 * ANG, 0.0, +1.414 * ANG)),
                  ("Cl", (+3.435 * ANG, 0.0, -1.414 * ANG)),
                  ("Cl", (-3.435 * ANG, 0.0, +1.414 * ANG)),
                  ("Cl", (-3.435 * ANG, 0.0, -1.414 * ANG))])


# --- the spellings --------------------------------------------------------------------------

SPELLINGS = [
    "shells",
    ("shell", "Dy"),
    ("shell", ("Ti1", "Ti2")),
    ("shell", "Fe", "d"),
    ("frontier", ("N1", "N2")),
    ("frontier", 3, 1, 1),
    ("bridge", ("Dy1", "Dy2")),
    ("bridge", (1, 2), ("O5",)),
    ("bridge", (1, 2), ("O5",), 2),
    ("bonding", "Fe"),
    ("bonding", "Fe", 3),
    ("double", "Ce"),
    ("double", "Ce", "f"),
]


@pytest.mark.parametrize("spec", SPELLINGS)
def test_every_spelling_parses_and_writes_itself_back(spec):
    """⚠ The round trip is the test that the spelling was *read* the way it was written.

    A parser that drops an argument, pools a fragment it should not, or defaults a count it
    was given still produces a valid target -- and the calculation then runs on a statement
    nobody made. Writing the spelling back and re-parsing it is what catches that, and it is
    also what lets a resolved request be printed as something a user could type.
    """
    target = tg.parse(spec)
    again = tg.parse(target.spelling())
    assert again == target, (spec, target.spelling())


def test_the_bare_spelling_is_the_detected_shell_set():
    target = tg.parse("shells")
    assert isinstance(target, tg.Shell) and target.detected and target.l is None


def test_an_element_symbol_is_one_key_and_not_a_sequence_of_letters():
    """``"Dy"`` is one atom key. ``tuple("Dy")`` is ``("D", "y")``, which names nothing."""
    assert tg.parse(("shell", "Dy")).atoms == ("Dy",)
    assert tg.parse(("shell", ("Dy1", "Dy2"))).atoms == ("Dy1", "Dy2")


def test_counts_and_shells_survive_the_parse():
    frontier = tg.parse(("frontier", ("N1", "N2"), 2, 1))
    assert (frontier.n_occ, frontier.n_vir) == (2, 1)
    assert tg.parse(("bonding", "Fe", 3)).n_pairs == 3
    assert tg.parse(("bridge", (1, 2), (3,), 2)).n_pairs == 2
    assert tg.parse(("shell", "Fe", "d")).l == 2
    assert tg.parse(("double", "Ce", "f")).l == 3


# --- the refusals ---------------------------------------------------------------------------

def test_an_unknown_class_lists_the_five_that_exist():
    with pytest.raises(ValueError, match="unknown target class"):
        tg.parse(("orbitals", "Fe"))
    with pytest.raises(ValueError, match="shells"):
        tg.parse("everything")


def test_the_wrong_number_of_arguments_shows_the_accepted_forms():
    for spec in [("shell", "Fe", "d", 2), ("frontier", "N", 1), ("bonding",),
                 ("double", "Ce", "f", 1), ("bridge",)]:
        with pytest.raises(ValueError):
            tg.parse(spec)


def test_a_bridge_is_between_exactly_two_sites():
    """A pathway between three centres is three bridges, and which pair each connects is the
    statement -- so a three-site spelling is refused rather than reduced."""
    with pytest.raises(ValueError, match="exactly two sites"):
        tg.parse(("bridge", ("Dy1", "Dy2", "Dy3")))


def test_a_principal_quantum_number_is_refused_as_a_shell():
    """⚠ Inherited from the one angular-momentum resolver: ``"4f"`` counts shells within the
    *basis* and is therefore basis-dependent, which is the trap the whole ``(atom, l)``
    addressing exists to avoid."""
    with pytest.raises(ValueError, match="principal quantum number"):
        tg.parse(("shell", "Ce", "4f"))


def test_a_negative_count_is_refused():
    with pytest.raises(ValueError, match="negative"):
        tg.parse(("bonding", "Fe", -1))


def test_an_empty_target_list_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="physical statement"):
        tg.parse_targets([])


# --- the ordering ---------------------------------------------------------------------------

def test_targets_come_back_in_the_fixed_priority_order():
    """⚠ Priority is a statement about physics -- the mechanism before the correlation
    refinement -- and it is what makes a dropped class attributable, so it is imposed here
    rather than taken from the order the user happened to write."""
    order = tg.parse_targets([("bonding", "Fe"), ("double", "Fe"), "shells",
                              ("bridge", (1, 2)), ("frontier", 3)])
    assert [t.cls for t in order] == ["shell", "frontier", "bridge", "bonding", "double"]
    assert [t.priority for t in order] == [1, 2, 3, 4, 5]


def test_two_targets_of_one_class_keep_the_order_they_were_written_in():
    order = tg.parse_targets([("bridge", (1, 2)), ("bridge", (1, 3))])
    assert [t.sites[1] for t in order] == [(2,), (3,)]


def test_a_single_tuple_target_is_one_target_and_not_a_list_of_two():
    """``targets=("shell", "Fe")`` is a target. Read as a list it would be two nonsense
    entries, and the failure would be a refusal rather than a wrong space -- but only
    because "Fe" happens not to be a class name."""
    assert tg.parse_targets(("shell", "Fe")) == (tg.Shell(atoms=("Fe",)),)
    assert tg.parse_targets(None) == (tg.Shell(),)


def test_the_default_target_set_is_the_detected_shells():
    assert tg.parse_targets(None)[0].detected


# --- the statements -------------------------------------------------------------------------

@pytest.mark.parametrize("spec", SPELLINGS)
def test_no_statement_contains_an_index_list(spec):
    """⚠ A description that carries spinor indices is not reproducible outside the orbital
    set it came from, and it is what reaches a property dump's header."""
    text = tg.parse(spec).statement(("1 Ti", "2 Ti"))
    assert not re.search(r"\[\s*\d", text), text
    assert "spinor" not in text.lower()
    assert text and text[0].islower()


def test_a_statement_names_the_atoms_it_resolved_to():
    text = tg.parse(("shell", ("Ti1", "Ti2"), "d")).statement(("1 Ti", "2 Ti"))
    assert "1 Ti and 2 Ti" in text and "d shell" in text


# --- atom addressing ------------------------------------------------------------------------

def test_an_element_key_pools_every_atom_of_that_element():
    """⚠ Deliberately different from the character selection, which refuses an ambiguous
    symbol: there a statement has to say which centre it means, here the pooling *is* the
    statement and the atoms it resolved to are printed with it."""
    assert tg.resolve_atoms(TI2CL6, "Ti") == (0, 1)
    assert tg.resolve_atoms(TI2CL6, "Cl") == (2, 3, 4, 5, 6, 7)


def test_a_label_and_a_number_are_one_atom_and_are_one_based():
    assert tg.resolve_atoms(TI2CL6, "Ti2") == (1,)
    assert tg.resolve_atoms(TI2CL6, 1) == (0,)
    assert tg.resolve_atoms(TI2CL6, (3, 4)) == (2, 3)


def test_a_label_naming_the_wrong_element_is_refused_not_reinterpreted():
    with pytest.raises(ValueError, match="names a"):
        tg.resolve_atoms(TI2CL6, "Cl1")


def test_a_key_naming_no_atom_of_this_molecule_is_refused():
    with pytest.raises(ValueError, match="no atom"):
        tg.resolve_atoms(TI2CL6, "Fe")
    with pytest.raises(ValueError, match="out of range"):
        tg.resolve_atoms(TI2CL6, 99)


def test_atom_labels_are_the_output_side_one_based_form():
    assert tg.atom_labels(TI2CL6, (0, 1)) == ("1 Ti", "2 Ti")


# --- bridge detection by contact ------------------------------------------------------------

def test_the_bridging_atoms_of_the_dimer_are_detected_by_contact():
    """Only the two bridging chlorines touch both titaniums; the terminal ones touch one.

    Geometry alone, so this is a statement about the criterion rather than about an SCF.
    """
    assert cand.bridge_atoms_by_contact(TI2CL6, (0,), (1,)) == (2, 3)


def test_a_detection_that_finds_nothing_asks_for_the_atoms_by_name():
    far = _layout([("Ti", (0.0, 0.0, 0.0)), ("Ti", (40.0, 0.0, 0.0)),
                   ("Cl", (0.0, 4.0, 0.0))])
    with pytest.raises(ValueError, match="Name the bridging atoms|no ligand is within"):
        cand.bridge_atoms_by_contact(far, (0,), (1,))


def test_an_element_with_no_tabulated_radius_is_refused_rather_than_guessed():
    """The radii stop where the published table stops (Cm), and a guessed radius would
    decide which atoms are offered as a pathway."""
    exotic = _layout([("U", (0.0, 0.0, 0.0)), ("U", (7.0, 0.0, 0.0)),
                      ("Es", (3.5, 0.0, 0.0))])
    with pytest.raises(ValueError, match="no covalent radius"):
        cand.bridge_atoms_by_contact(exotic, (0,), (1,))


def test_the_radii_table_is_the_published_one_where_it_is_checked():
    """A handful of values against Cordero et al. (2008), including the three the paper
    gives per spin state and the sp3 carbon -- the ones a transcription would get wrong."""
    for symbol, radius in (("H", 0.31), ("C", 0.76), ("O", 0.66), ("Cl", 1.02),
                           ("Ti", 1.60), ("Fe", 1.32), ("Dy", 1.92), ("U", 1.96)):
        assert cand.COVALENT_RADII_ANGSTROM[symbol] == pytest.approx(radius)


# --- the package ----------------------------------------------------------------------------

def _module_files(package_dir):
    return sorted(p for p in package_dir.rglob("*.py") if "__pycache__" not in p.parts)


def _top_level_imports(tree):
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def test_the_dependency_runs_one_way():
    """⚠ Nothing in the calculation path may import ``kuiva.autocas``.

    The mirror of the same test for ``kuiva.qc`` and ``kuiva.extras``. Selection happens
    *before* a calculation; a back-edge -- even a convenience re-export in an ``__init__`` --
    would put it inside one, and the stage that reaches this package does so lazily for
    exactly that reason.
    """
    offenders = []
    for package in ("ci", "mcscf", "rdm", "x2c", "dmrg", "amf", "integrals", "props",
                    "spinor", "orth", "basis", "io", "util", "pt", "qc", "extras", "symm"):
        for path in _module_files(REPO / "kuiva" / package):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[:2] == ["kuiva", "autocas"]:
                            offenders.append((path.relative_to(REPO).as_posix(), alias.name))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    absolute = node.level == 0 and module.startswith("kuiva.autocas")
                    relative = node.level > 0 and module.split(".")[0] == "autocas"
                    if absolute or relative:
                        offenders.append((path.relative_to(REPO).as_posix(), module))
    assert offenders == [], (
        "nothing in the calculation path may import kuiva.autocas: {}".format(offenders))


def test_importing_the_package_costs_nothing():
    """``import kuiva.autocas`` must not pull the front end in.

    Asserted from the sources, because on a machine that has PySCF installed the failure is
    invisible -- which is every machine that runs this suite.
    """
    import kuiva.autocas

    assert "parse_targets" in dir(kuiva.autocas)
    assert kuiva.autocas.parse_targets is tg.parse_targets
    offenders = []
    for path in _module_files(REPO / "kuiva" / "autocas"):
        for name in _top_level_imports(ast.parse(path.read_text())):
            if name.split(".")[0] in ("pyscf", "h5py") or name.startswith("kuiva.interface"):
                offenders.append((path.relative_to(REPO).as_posix(), name))
    assert offenders == [], (
        "kuiva.autocas must not import the front end at module scope; move these inside the "
        "function that needs them: {}".format(offenders))
