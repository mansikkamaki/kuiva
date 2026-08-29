"""The character selection must land on the VALENCE shell, not on a core one.

Why this test exists
--------------------
``character=(atom, l)`` takes the **lowest** Kramers pairs of that character. That is the
valence shell only while nothing of the same ``l`` is filled below it — true for every ``d``
and ``f`` system in the suite (3d and 4f are the lowest of their kind) and for boron's 2p, and
false for every other member of the np¹ series: Al's 3p¹ selects 2p, Ga's 4p¹ selects 2p, and
so on down to Tl.

⚠ **The failure is silent twice over, which is what makes it worth a test rather than a
comment.** The calculation converges and reports an ordinary ²P doublet — Ga's came out at
249 400 cm⁻¹ against an experimental 826 — and the **g values do not notice**, because a p¹
shell is Landé 2/3 whichever shell it occupies. A study whose only observable is ``g``
therefore cannot verify its own active space, and one did not for some time.

What is asserted is the structural invariant, not a number: for a system with one electron in
its valence shell, the selected active space must begin exactly where the inactive spinors end
(``nelec_total - nelecas``). A core-shell selection puts it far below that and fails here
immediately, at SCF cost only — no CASSCF, and no comparison with experiment needed.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "generate"))

import systems as sysdef                                                     # noqa: E402

from kuiva.interface import api                                              # noqa: E402

#: One electron in the valence shell, so the frontier invariant below is exact. Kept to the
#: light members: the heavy ones need the same SCF and prove nothing extra about the *rule*,
#: and the default suite stays laptop-fast.
ONE_ELECTRON_VALENCE = ("b", "al", "ga")


@pytest.mark.parametrize("key", ONE_ELECTRON_VALENCE)
def test_character_selection_lands_on_the_valence_shell(key):
    system = sysdef.get(key)
    molecule = api.Molecule(atoms=system.atoms, basis=system.basis,
                            charge=system.charge, spin=system.spin)
    # screening="none": the two-electron picture change changes no orbital *ordering* and
    # would cost a four-component atomic solve per element for nothing here.
    reference = api.spinor_reference(molecule, screening="none", memory_gb=4.0)
    space = api.active_space_for(reference, **sysdef.character_selection(system))

    n_inactive = reference.data.nelec_total - system.nelecas
    active = sorted(int(i) for i in space.spaces.active)
    assert active[0] == n_inactive, (
        "{}: the active space starts at spinor {} but the frontier is at {} — the selection "
        "landed on a core {} shell. Set active_skip_pairs on the system; a plain (atom, l) "
        "form takes the lowest pairs of that character."
        .format(key, active[0], n_inactive, system.active_l))
    assert active == list(range(n_inactive, n_inactive + 2 * system.ncas))


def test_every_np1_member_declares_its_skip():
    """The skip count is a property of the system, so no consumer can forget it.

    Boron is the one member whose valence p shell *is* the lowest, hence skip 0; every other
    member must declare a nonzero one, and the counts are the filled p shells below it.
    """
    expected = {"b": 0, "al": 3, "ga": 6, "in": 9, "tl": 12}
    for key, skip in expected.items():
        assert sysdef.get(key).active_skip_pairs == skip, key


def test_the_experimental_guard_covers_the_series():
    """Every np¹ member has a published splitting to check a computed one against.

    The g values cannot catch a core-shell selection; this table is what can, and a member
    without an entry would be a member no cheap check protects.
    """
    for key in ("b", "al", "ga", "in", "tl"):
        assert key in sysdef.EXPERIMENTAL_SPLITTING_CM, key
        assert sysdef.EXPERIMENTAL_SPLITTING_CM[key] > 0.0
