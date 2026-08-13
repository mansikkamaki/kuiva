"""Shell-resolved configurations, and the boundary ``kuiva.extras`` is defined by.

Three things are worth testing here and they are of different kinds.

1. **One grammar.** ``AtomicConfiguration`` and ``ShellConfiguration`` read the same string
   through the same function, so the test that matters is not that each parses correctly but
   that the two **agree** — including on the noble-gas cores, where one of them derives
   principal quantum numbers the other never had. Two grammars for one notation would drift
   silently, and the string is provenance that ends up in a stored file.

2. **What is refused.** A shell-resolved configuration is exactly as expressive as the per-``l``
   form it converts to, and the class's whole safety property is that the two agree about
   which configurations exist. A core hole accepted here would become a mean field of a
   different state, with nothing in the output to say so.

3. **The package boundary**, asserted from the sources as ``kuiva.qc``'s is: nothing in the
   calculation path may import ``kuiva.extras``, and nothing here may pull PySCF in at import
   time.

All of it is arithmetic and string handling — no SCF, no integrals, instant.
"""
import ast
import pathlib

import pytest

from kuiva.amf.configuration import AtomicConfiguration, parse_shell_terms
from kuiva.extras.shells import ShellConfiguration, shell_capacity, shell_label

REPO = pathlib.Path(__file__).resolve().parents[1]
EXTRAS_DIR = REPO / "kuiva" / "extras"

#: The configuration the feature exists for, and the shape of every Ln(I) in the series.
DY_I = "[Xe] 4f9 5d1 6s1"


# --- 1. The shared grammar -----------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("1s2", ((1, 0, 2),)),
    ("1s2 2s2 2p6", ((1, 0, 2), (2, 0, 2), (2, 1, 6))),
    ("[He]2s2 2p4", ((1, 0, 2), (2, 0, 2), (2, 1, 4))),
    ("[Ar]3d1", ((1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 6), (3, 2, 1))),
    # order and separators are free; repeated shells sum
    ("6s1, 4f9", ((4, 3, 9), (6, 0, 1))),
    ("2p3 2p3", ((2, 1, 6),)),
])
def test_the_reader_resolves_shells(text, expected):
    assert parse_shell_terms(text) == expected


def test_an_explicitly_empty_shell_survives_the_reader():
    """``"[Xe]4f0"`` is how a caller writes "this shell is empty" — Ce(IV) is stated that way
    — and dropping the term would be a silent edit of a provenance string. Which consumers
    keep it is their rule: the shell-resolved configuration drops it as unoccupied."""
    terms = parse_shell_terms("[Xe]4f0")
    assert (4, 3, 0) in terms
    assert sum(q for _, _, q in terms) == 54
    assert ShellConfiguration.parse("[Xe]4f0") == ShellConfiguration.parse("[Xe]")


def test_a_noble_gas_core_gets_the_right_principal_quantum_numbers():
    """⚠ The core's ``n`` are **derived** from per-``l`` counts, and the derivation is only
    safe because a noble gas is aufbau-filled. Xenon is the case the lanthanide work runs
    through: 4d is full and 4f is empty, so a rule that merely counted shells per channel in
    order would still have to put the twentieth d electron in 4d and no f electron anywhere.
    """
    xe = dict(((n, l), q) for n, l, q in parse_shell_terms("[Xe]"))
    assert xe == {(1, 0): 2, (2, 0): 2, (3, 0): 2, (4, 0): 2, (5, 0): 2,
                  (2, 1): 6, (3, 1): 6, (4, 1): 6, (5, 1): 6,
                  (3, 2): 10, (4, 2): 10}
    assert sum(xe.values()) == 54
    rn = dict(((n, l), q) for n, l, q in parse_shell_terms("[Rn]"))
    assert rn[(4, 3)] == 14 and rn[(5, 2)] == 10 and sum(rn.values()) == 86


@pytest.mark.parametrize("text", [
    "1s2", "1s2 2s2 2p6", "[He]2s2 2p4", "[Ar]3d1", "[Xe]4f9", "[Xe]4f13",
    "[Xe]4f14 5d10 6s2 6p3", "[Rn]5f4", DY_I, "[Ar] 3d2 4s1",
])
def test_the_two_configuration_classes_read_one_string_the_same_way(text):
    """⚠ **The point of the shared reader.** The shell-resolved and per-``l`` forms are
    different objects for different purposes, but they must never disagree about what a
    configuration string says — the same string reaches an AMF cache key through one of them
    and a Slater-Condon parameter label through the other.
    """
    shell = ShellConfiguration.parse(text)
    atomic = AtomicConfiguration.parse(text)
    assert shell.to_atomic() == atomic
    assert shell.n_electrons == atomic.n_electrons
    # ...and the conversion loses nothing: the per-l form resolves back to the same shells.
    assert atomic.shells() == shell.shells


def test_the_per_l_form_of_the_target_configuration():
    """Dy(I) 4f9 5d1 6s1, the configuration the whole feature is built for."""
    config = ShellConfiguration.parse(DY_I)
    assert config.n_electrons == 65                # Dy is Z = 66, so this is the monocation
    assert config.charge("Dy") == 1
    # ordered by (n, l), so 4f precedes 5s — the canonical order, not the spectroscopic one
    assert config.labels[-5:] == ("4f", "5s", "5p", "5d", "6s")
    assert config.as_dict()["4f"] == 9
    assert config.occupation(4, 3) == 9
    assert config.occupation(5, 3) == 0
    assert config.open_shells() == ((4, 3, 9), (5, 2, 1), (6, 0, 1))
    assert not config.is_closed_shell
    assert config.to_atomic().canonical == "s11 p24 d21 f9"


def test_the_identity_is_the_shells_and_not_the_spelling():
    """Canonical and hashable, as the per-``l`` form is: order, separators and empty shells
    are spelling. The label is provenance and stays out of the identity."""
    a = ShellConfiguration.parse(DY_I)
    b = ShellConfiguration.parse("[Xe] 6s1 5d1 4f9 5f0")
    assert a == b and hash(a) == hash(b)
    assert a.label != b.label                      # ...and the spelling is still recorded
    assert str(a) == DY_I
    assert ShellConfiguration(a.shells) == a       # round trip through the tuple form


def test_a_closed_shell_configuration_is_recognized_as_one():
    assert ShellConfiguration.parse("[Xe]").is_closed_shell
    assert ShellConfiguration.parse("[Xe]4f14").is_closed_shell
    assert ShellConfiguration.parse("[Xe]4f14 6s2").open_shells() == ()
    assert not ShellConfiguration.parse("[Xe]4f13").is_closed_shell


# --- 2. What is refused --------------------------------------------------------------------

@pytest.mark.parametrize("text,message", [
    ("[Fe]3d6", "not a noble gas"),
    ("f9", "cannot read"),
    ("4f9x", "cannot read"),
    ("", "empty configuration"),
    ("1p6", "no 1p shell"),                        # a typo the grammar can catch on its own
    ("2d1", "no 2d shell"),
])
def test_the_reader_refuses_what_it_cannot_read(text, message):
    with pytest.raises(ValueError, match=message):
        parse_shell_terms(text)
    with pytest.raises(ValueError, match=message):
        AtomicConfiguration.parse(text)            # the shared reader, same refusals


@pytest.mark.parametrize("text,message", [
    ("3d10 5d1", "aufbau filling"),                # 4d missing entirely
    ("1s2 3s1", "aufbau filling"),
    ("3d9 4d1", "not full"),                       # a hole below an occupied shell
    ("3d10 4d9 5d1", "not full"),                  # ...two shells down, still refused
    ("4f15", "holds at most 14"),
    ("1s3", "holds at most 2"),
    ("[Xe]4d2", "holds at most 10"),               # the core's 4d10 plus two more
])
def test_a_configuration_with_no_spherical_ensemble_is_refused(text, message):
    """⚠ **The class's safety property.** Every accepted configuration converts to the
    per-``l`` form without loss, and every refused one is a configuration that form would put
    the electrons somewhere else in — a 3d hole below an occupied 4d becomes 3d10 4d0 the
    moment the principal quantum numbers are summed away. Accepting it would produce a mean
    field of a state nobody asked for, with nothing in the output to say so.
    """
    with pytest.raises(ValueError, match=message):
        ShellConfiguration.parse(text)


def test_an_oxidation_state_is_not_a_shell_configuration():
    """⚠ Deliberately narrower than ``AtomicConfiguration.coerce``, which does accept one.

    Which shells the electrons of Dy(1+) occupy — 4f9 5d1 6s1 or 4f10 6s1 — is chemistry, and
    it changes every shell-resolved quantity. The per-``l`` form can be derived from an
    oxidation state because it does not have to answer that question; this one cannot.
    """
    with pytest.raises(ValueError, match="oxidation state"):
        ShellConfiguration.coerce("+1")
    with pytest.raises(ValueError, match="oxidation state"):
        ShellConfiguration.coerce("3+")
    assert AtomicConfiguration.coerce("+3", element="Dy").n_electrons == 63   # still works

    assert ShellConfiguration.coerce(DY_I) == ShellConfiguration.parse(DY_I)
    assert ShellConfiguration.coerce(ShellConfiguration.parse(DY_I)).n_electrons == 65
    assert ShellConfiguration.coerce([(1, 0, 2), (2, 0, 1)]).canonical == "1s2 2s1"


def test_the_helpers_say_what_a_shell_is():
    assert [shell_capacity(l) for l in range(4)] == [2, 6, 10, 14]
    assert shell_label(4, 3) == "4f" and shell_label(6, 0) == "6s"


# --- 3. The package boundary ---------------------------------------------------------------

def _module_files(package_dir):
    return sorted(p for p in package_dir.rglob("*.py") if "__pycache__" not in p.parts)


def _top_level_imports(tree):
    """Module names imported at **module scope** — i.e. at import time, not inside a call."""
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def test_the_dependency_runs_one_way():
    """⚠ Nothing in the calculation path may import ``kuiva.extras``.

    The mirror of ``test_qc_skeleton.py::test_the_dependency_runs_one_way``. A special-purpose
    method is allowed to depend on the pipeline; the pipeline depending back on it — even
    through a convenience re-export in an ``__init__`` — would make a side feature part of
    every calculation, which is the one thing the package layout exists to prevent.
    """
    offenders = []
    for package in ("ci", "mcscf", "rdm", "x2c", "dmrg", "amf", "integrals", "interface",
                    "props", "spinor", "orth", "basis", "io", "util", "pt", "qc"):
        for path in _module_files(REPO / "kuiva" / package):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[:2] == ["kuiva", "extras"]:
                            offenders.append((path.relative_to(REPO).as_posix(), alias.name))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    absolute = node.level == 0 and module.startswith("kuiva.extras")
                    relative = node.level > 0 and module.split(".")[0] == "extras"
                    if absolute or relative:
                        offenders.append((path.relative_to(REPO).as_posix(), module))
    assert offenders == [], (
        "nothing in the calculation path may import kuiva.extras: {}".format(offenders))


def test_importing_the_package_costs_nothing():
    """``import kuiva.extras`` must not pull the front-end in.

    The names are resolved lazily and no module here imports PySCF at module scope, so a run
    that never touches a special feature never pays for one. Asserted from the sources
    because the failure is invisible on a machine that has PySCF installed — which is every
    machine that runs this suite.
    """
    import kuiva.extras

    assert "ShellConfiguration" in dir(kuiva.extras)
    assert kuiva.extras.ShellConfiguration is ShellConfiguration

    offenders = []
    for path in _module_files(EXTRAS_DIR):
        for name in _top_level_imports(ast.parse(path.read_text())):
            if name.split(".")[0] in ("pyscf", "h5py"):
                offenders.append((path.relative_to(REPO).as_posix(), name))
    assert offenders == [], (
        "kuiva.extras must not import the front-end at module scope; move these inside the "
        "function that needs them: {}".format(offenders))
