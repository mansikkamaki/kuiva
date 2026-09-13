"""Tier 0: Hund ground terms and the theoretical floor of a proposed state count.

What these tests are chosen to fail on
--------------------------------------
The floor is arithmetic on two quantum numbers, so the way it goes wrong is by being
*plausible*: a ``J = L + S`` used below half filling gives 6 instead of 4 for Ti(3+), which is
a state count that averages a complete-looking manifold that is not one, and every downstream
check passes on it. So the ground level of **every** ``d^n`` and ``f^n`` is asserted against
the tabulated free-ion terms rather than a few representative ones -- the particle-hole
mirror at half filling is exactly where an off-by-one lives.

⚠ The second set of tests is the one that cannot be satisfied by accident: the floor has to be
a **manifold boundary of the committed reference spectra**. Those degeneracy patterns were
produced by the CASSCF and stored (``tests/reference/tier2_kuiva.json``); a floor that is not
a prefix sum of one of them is a count that would cut a manifold of a system this suite has
actually computed. Nothing in :mod:`kuiva.autocas.multiplets` knows those numbers exist.

Pure arithmetic and stored JSON: no SCF, so it runs in the default suite.
"""
import json
from pathlib import Path

import pytest

from kuiva.autocas import multiplets as mult

REFERENCE = Path(__file__).resolve().parent / "reference" / "tier2_kuiva.json"

#: Free-ion ground terms of the ``d`` shell (Condon & Shortley; any inorganic text).
D_TERMS = {0: "1S0", 1: "2D3/2", 2: "3F2", 3: "4F3/2", 4: "5D0", 5: "6S5/2", 6: "5D4",
           7: "4F9/2", 8: "3F4", 9: "2D5/2", 10: "1S0"}

#: Free-ion ground terms of the ``f`` shell -- the lanthanide series, Ce(3+) to Yb(3+).
F_TERMS = {0: "1S0", 1: "2F5/2", 2: "3H4", 3: "4I9/2", 4: "5I4", 5: "6H5/2", 6: "7F0",
           7: "8S7/2", 8: "7F6", 9: "6H15/2", 10: "5I8", 11: "4I15/2", 12: "3H6",
           13: "2F7/2", 14: "1S0"}


@pytest.mark.parametrize("n,symbol", sorted(D_TERMS.items()))
def test_every_d_shell_ground_term(n, symbol):
    assert mult.hund_ground_term(2, n).symbol == symbol


@pytest.mark.parametrize("n,symbol", sorted(F_TERMS.items()))
def test_every_f_shell_ground_term(n, symbol):
    assert mult.hund_ground_term(3, n).symbol == symbol


def test_the_four_cases_the_plan_names_come_out_as_stated():
    """Ti(3+), Dy(3+), Yb(3+), Ce(3+) levels and the Fe(2+)/Mn(2+)/Gd(3+) floors."""
    assert mult.hund_ground_term(2, 1).level_dimension == 4          # Ti(3+) 2D3/2
    assert mult.hund_ground_term(3, 9).level_dimension == 16         # Dy(3+) 6H15/2
    assert mult.hund_ground_term(3, 13).level_dimension == 8         # Yb(3+) 2F7/2
    assert mult.hund_ground_term(3, 1).level_dimension == 6          # Ce(3+) 2F5/2
    assert mult.shell_floor(2, 6) == 5                               # Fe(2+) 5D, spin floor
    assert mult.shell_floor(2, 5) == 6                               # Mn(2+) 6S
    assert mult.shell_floor(3, 7) == 8                               # Gd(3+) 8S7/2


def test_the_two_regimes_are_different_numbers_and_stay_separate():
    """⚠ A count that bounds ``2S+1`` terms generally cuts the ``2J+1`` multiplets of the
    same system with spin-orbit coupling on, and the converse bites equally -- so the f block
    takes the level and the d block the spin multiplicity, and neither is a default for the
    other."""
    term = mult.hund_ground_term(2, 1)
    assert (term.spin_multiplicity, term.level_dimension) == (2, 4)
    assert mult.shell_floor(2, 1) == 2                      # d block: the spin multiplicity
    assert mult.shell_floor(2, 1, regime="level") == 4      # asked for explicitly
    assert mult.shell_floor(3, 9) == 16                     # f block: the level
    assert mult.shell_floor(3, 9, regime="spin") == 6


def test_the_term_dimension_is_the_spin_free_manifold():
    """``(2S+1)(2L+1)`` -- what a spin-free calculation's complete manifold would be, which
    is a third count again and is why it has its own name."""
    assert mult.hund_ground_term(3, 9).term_dimension == 6 * 11      # 6H
    assert mult.hund_ground_term(2, 6).term_dimension == 5 * 5       # 5D


def test_a_closed_or_empty_shell_contributes_one_state_and_not_a_manifold():
    for l in (2, 3):
        for n in (0, 4 * l + 2):
            term = mult.hund_ground_term(l, n)
            assert term.symbol == "1S0" and term.level_dimension == 1


def test_an_odd_electron_count_never_gets_an_odd_spin_floor():
    """Kramers makes every level of an odd-electron system at least two-fold, so an odd floor
    there is a count that cannot be a manifold boundary at all."""
    for l in (2, 3):
        for n in range(1, 4 * l + 2, 2):
            assert mult.shell_floor(l, n, regime="spin") % 2 == 0


def test_impossible_occupations_are_refused():
    with pytest.raises(ValueError, match="holds 0"):
        mult.hund_ground_term(2, 11)
    with pytest.raises(ValueError, match="holds 0"):
        mult.hund_ground_term(3, -1)
    with pytest.raises(ValueError, match="regime"):
        mult.shell_floor(2, 1, regime="whatever")


def test_the_coupled_floor_is_the_product_and_is_printed_whatever_its_size():
    """⚠ Two Dy(3+) give 256, and that number is reported rather than reduced: what to do
    about a manifold no solver can average whole is a decision, not an arithmetic step."""
    assert mult.coupled_floor([16, 16]) == 256
    assert mult.coupled_floor([2, 2]) == 4
    with pytest.raises(ValueError, match="no centres"):
        mult.coupled_floor([])
    with pytest.raises(ValueError, match="at least one state"):
        mult.coupled_floor([4, 0])


def test_the_l_letters_skip_j_as_the_convention_does():
    assert [mult.term_letter(l) for l in range(8)] == list("SPDFGHIK")


# --- against the committed reference spectra ------------------------------------------------

#: ``system -> (l, electrons in the shell, floor)``. The electron counts are the committed
#: systems' own (``tests/generate/systems.py``: ``nelecas`` in a shell of ``ncas`` orbitals).
COMMITTED = {
    "ce3p": (3, 1, 6),        # free Ce(3+): the ground level is the first block
    "yb3p": (3, 13, 8),       # free Yb(3+)
    "dy3p": (3, 9, 16),       # free Dy(3+)
    "cecl3": (3, 1, 6),       # f^1 in a ligand field: three Kramers doublets
    "ticl3": (2, 1, 2),       # d^1 in a ligand field: the spin floor, one doublet
}


@pytest.mark.parametrize("key", sorted(COMMITTED))
def test_the_floor_is_a_manifold_boundary_of_the_committed_spectrum(key):
    """⚠ The test that cannot pass by accident.

    The degeneracy patterns come from the stored CASSCF references; the floor comes from
    Hund's rules and knows nothing about them. A floor that is not a prefix sum of the
    pattern is a state count that would cut a manifold of a system this suite has computed --
    which is the failure the floor exists to prevent.
    """
    records = json.loads(REFERENCE.read_text())["records"]
    pattern = records["{}/x2c-SVPall-2c".format(key)]["degeneracy_pattern"]
    l, n, expected = COMMITTED[key]
    floor = mult.shell_floor(l, n)
    assert floor == expected
    boundaries = {sum(pattern[:k]) for k in range(1, len(pattern) + 1)}
    assert floor in boundaries, (
        "the floor {} for {} is not a manifold boundary of the committed pattern {}"
        .format(floor, key, pattern))


def test_the_coupled_floor_is_a_boundary_of_the_dimers_committed_spectrum():
    """Ti2Cl6: two coupled d^1 centres, so the exchange manifold is 2 x 2 = 4 states."""
    records = json.loads(REFERENCE.read_text())["records"]
    pattern = records["ti2cl6/x2c-SVPall-2c"]["degeneracy_pattern"]
    floor = mult.coupled_floor([mult.shell_floor(2, 1)] * 2)
    assert floor == 4
    boundaries = {sum(pattern[:k]) for k in range(1, len(pattern) + 1)}
    assert floor in boundaries, (floor, pattern[:6])
