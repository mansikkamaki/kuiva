"""Tier 0/1: resolving a target into candidate Kramers pairs.

What is asserted, and why each one can fail
-------------------------------------------
A candidate set is columns plus a statement, and both can be wrong in ways that produce a
calculation rather than an error:

* **the shell must be the space the committed references use.** ``ticl3``'s automatic shell is
  compared element for element against the ``character=("Ti", "d")`` selection the Tier-2
  reference is defined by. Two constructions of "the 3d shell of titanium" that disagree are
  two definitions of it, and the one in the reference files is the one that has been
  validated externally.
* **the electron count must come from the right place.** It is measured off the selected
  pairs' reference occupations -- except for a shell that is *empty* in the reference, where
  there is nothing to measure and the ion's reference state supplies it loudly. Both branches
  are asserted, including the refusal for the case where the reference state is only the
  neutral-atom default.
* **a bounded count may not split a tie.** TiCl3's two Ti-Cl sigma bonding pairs are exactly
  degenerate, so a request for one bonding pair has to come back with two.
* ⚠ **the tie rule must not fire when there is no tie**, which is the defect this file's
  mechanism test exists for: reading a *selection order* as if it were a spectrum reported
  "a degenerate group of 3" for the values 0.505..0.766 and rounded every bounded count out
  to its whole pool. A guard that always fires is the same defect as one that never can.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "generate"))

import systems as sysdef                                                    # noqa: E402

from kuiva.autocas import candidates as cand                                # noqa: E402
from kuiva.autocas import centres as ctr                                    # noqa: E402
from kuiva.autocas import targets as tg                                     # noqa: E402
from kuiva.interface import api                                             # noqa: E402
from test_autocas_centres import reference_for                              # noqa: E402


@pytest.fixture(scope="module")
def ticl3():
    return reference_for("ticl3")


@pytest.fixture(scope="module")
def fecl2():
    """The high-spin ROHF case: five *occupied* d pairs holding six electrons.

    In the default suite (2.5 s wall) although the system is marked slow -- what is slow
    about ``fecl2`` is its CASSCF, and this is the only committed system whose shell has
    several occupied pairs, i.e. the case the count-stated cut has to split correctly.
    """
    return reference_for("fecl2")


def _principal_overlaps(reference, coeff_a, columns_a, coeff_b, columns_b) -> np.ndarray:
    """Singular values of the overlap between two sets of spinor columns.

    ⚠ **The comparison two active-space constructions actually admit.** An index list is only
    meaningful inside the orbital set it came from, and AVAS *rotates*: it gathers the
    character-carrying pairs at the inner edge of their occupation group, so a shell whose
    canonical orbital sits at spinor 42 comes back at spinor 54 having moved nowhere
    physically. What has to agree is the **span**, and a span is compared through its
    principal overlaps -- all ones when the two sets span the same subspace.
    """
    from kuiva.spinor.expand import spin_block_diagonal

    s2 = spin_block_diagonal(np.asarray(reference.data.s_ao))
    a = np.asarray(coeff_a)[:, np.asarray(columns_a, dtype=int)]
    b = np.asarray(coeff_b)[:, np.asarray(columns_b, dtype=int)]
    return np.linalg.svd(a.conj().T @ s2 @ b, compute_uv=False)


def _whole_pairs(columns) -> bool:
    """Every column comes with its Kramers partner, and no pair is split."""
    columns = np.sort(np.asarray(columns, dtype=int))
    return (columns.size % 2 == 0
            and np.array_equal(columns[0::2] % 2, np.zeros(columns.size // 2, dtype=int))
            and np.array_equal(columns[1::2], columns[0::2] + 1))


# --- the shell ------------------------------------------------------------------------------

def test_the_shell_spans_the_space_the_committed_reference_is_defined_by(ticl3):
    """⚠ The same **span** as ``character=("Ti", "d")``, which is how the Tier-2 reference
    states its active space, and element for element too on this system.

    A shell construction that agreed only in *size* would be a different calculation with the
    same name. The span is what has to agree in general: see
    :func:`test_a_high_spin_shell_spans_the_committed_space_from_different_columns`, where the
    rotation relocates a pair and the index lists differ while the space does not.
    """
    system = sysdef.get("ticl3")
    stated = api.active_space_for(ticl3, **sysdef.character_selection(system))
    detection = ctr.detect_centres(ticl3, report=False)
    construction = cand.shell_candidates(ticl3, detection.centres, report=False)
    shell = construction.shell
    assert list(shell.columns) == list(stated.spaces.active)
    assert shell.n_pairs == system.ncas
    assert shell.electrons == system.nelecas
    # ⚠ The same *pairs*, and a span that differs by a few per cent -- which is AVAS doing
    # its job rather than a disagreement: TiCl3's 3d is covalent (the selected projections run
    # 0.73-0.91), so the rotation trades ligand admixture in from outside the block. Measured
    # principal overlaps: 1.000, 1.000, 0.975 (x4), 0.961 (x4).
    overlaps = _principal_overlaps(ticl3, ticl3.spinors_in_ao(), stated.spaces.active,
                                   construction.coeff, shell.columns)
    assert float(np.min(overlaps)) > 0.95, overlaps


def test_the_count_stated_shell_is_what_a_threshold_of_0_4_would_have_chosen(ticl3):
    """⚠ The two AVAS selection rules on the real system, at the threshold where they agree.

    TiCl3's Ti 3d projections are 0.912, 0.906, 0.906, 0.735, 0.735 and then 0.265, so
    "everything above 0.4" and "the five pairs of the shell" are the same set. That they
    agree *here* is what makes the count-stated rule a restatement of AVAS rather than a
    second construction of the shell -- and 0.2, the published default, would have taken the
    two sigma pairs as well.
    """
    from kuiva.mcscf.avas import avas_projection

    reference, data = ticl3, ticl3.data
    common = dict(atom=[0], l="d", occupation=np.asarray(reference.spinors.occ))
    by_threshold = avas_projection(reference.spinors_in_ao(), data.s_ao,
                                   reference.ao_layout, data.atomic_reference,
                                   threshold=0.4, **common)
    detection = ctr.detect_centres(ticl3, report=False)
    shell = cand.shell_candidates(ticl3, detection.centres, report=False).shell
    assert list(np.asarray(by_threshold.selected) * 2) == list(shell.columns[0::2])
    by_default = avas_projection(reference.spinors_in_ao(), data.s_ao, reference.ao_layout,
                                 data.atomic_reference, threshold=0.2, **common)
    assert np.size(by_default.selected) == 7, "the published default takes the sigma pairs"


def test_a_count_that_cuts_a_degenerate_pair_warns_through_the_gap(ticl3, kuiva_caplog):
    """⚠ The honesty check of the count-stated mode, on a real degeneracy: TiCl3's two Ti-Cl
    sigma pairs project identically, so a six-pair count cuts between them and the eigenvalue
    gap at the cut is exactly zero. The count, and not the electronic structure, chose."""
    from kuiva.mcscf.avas import avas_projection

    result = avas_projection(ticl3.spinors_in_ao(), ticl3.data.s_ao, ticl3.ao_layout,
                             ticl3.data.atomic_reference, atom=[0], l="d",
                             occupation=np.asarray(ticl3.spinors.occ), n_pairs=6)
    assert abs(result.gap) < 1e-8
    assert any("requested count" in r.message for r in kuiva_caplog.records)


def test_a_high_spin_shell_spans_the_committed_space_from_different_columns(fecl2):
    """⚠ The other end of the count-stated cut's range, and the case that shows why a *span*
    is what two constructions can be compared through.

    Every one of FeCl2's five Fe 3d pairs is occupied -- one doubly, four singly -- so the cut
    takes no empty pair at all, and the ROHF occupations give the committed reference's six
    electrons. But the doubly occupied 3d pair is a **deep** canonical orbital (spinor 42,
    eps = -0.61, d character 1.000, six pairs below the singly occupied ones), and AVAS
    gathers it at the inner edge of the doubly occupied group instead (spinor 54). The index
    lists therefore differ while the space does not, which an element-for-element comparison
    would report as a disagreement.
    """
    system = sysdef.get("fecl2")
    stated = api.active_space_for(fecl2, **sysdef.character_selection(system))
    detection = ctr.detect_centres(fecl2, report=False)
    construction = cand.shell_candidates(fecl2, detection.centres, bonding=2, report=False)
    shell = construction.shell
    # ⚠ Here the spans agree to machine precision *because* the shell is ionic: the five Fe
    # 3d pairs carry 0.91-1.00 of the character already, so the rotation has nothing to mix
    # in and moves only the columns. On a covalent shell (TiCl3, above) the span itself moves.
    overlaps = _principal_overlaps(fecl2, fecl2.spinors_in_ao(), stated.spaces.active,
                                   construction.coeff, shell.columns)
    assert np.abs(overlaps - 1.0).max() < 1e-8, overlaps
    assert list(shell.columns) != list(stated.spaces.active), (
        "the rotation relocates the deep pair; if this ever becomes an equality the test "
        "above it is the one that matters")
    assert shell.n_pairs == system.ncas and shell.electrons == system.nelecas
    assert np.all(shell.occupations > 0.5), "no empty pair was taken"
    assert sorted(shell.occupations.tolist()) == [1.0, 1.0, 1.0, 1.0, 2.0]
    assert construction.bonding.n_pairs >= 1
    cand.check_disjoint(construction.sets)


def test_the_shell_is_whole_pairs_fixed_and_ranked_by_its_projection(ticl3):
    detection = ctr.detect_centres(ticl3, report=False)
    shell = cand.shell_candidates(ticl3, detection.centres, report=False).shell
    assert _whole_pairs(shell.columns)
    assert shell.fixed, "a shell is taken whole or not at all"
    assert shell.ranking_name == "AVAS projection"
    assert shell.ranking.size == shell.n_pairs
    assert float(shell.ranking.min()) > 0.5
    assert shell.priority == 1


def test_asking_for_the_double_shell_does_not_change_what_the_valence_shell_holds(ticl3):
    """⚠ The mechanism, on the system that exposed it. The two-shell selection of a ``d^1``
    TiCl3 contains doubly occupied Cl sigma pairs with some 3d in them; splitting it
    "occupied pairs first" put them in the valence shell (CAS(5, 10), a 6S floor) and left
    them in the "correlating shell" too. The valence shell has to hold what it holds without
    the double shell, and the correlating shell has to be empty -- by construction, for every
    reference, not by the luck of one whose shell happens to be the whole occupied part."""
    centres = ctr.detect_centres(ticl3, report=False).centres
    single = cand.shell_candidates(ticl3, centres, report=False)
    both = cand.shell_candidates(ticl3, centres, double=True, report=False)
    assert both.shell.n_pairs == single.shell.n_pairs == 5
    assert both.shell.electrons == single.shell.electrons == 1.0
    assert both.double.n_pairs == 5 and both.double.electrons == 0.0
    assert np.all(both.double.occupations == 0.0)
    # Ranked by the projection onto the valence shell alone, and separated by it.
    assert float(both.shell.ranking.min()) > float(both.double.ranking.max()) + 0.05
    cand.check_disjoint(both.sets)


def test_the_shell_statement_is_physical_and_carries_the_reference_state(ticl3):
    """⚠ What reaches a property dump's header. An index list there would be unreadable
    outside the orbital set it came from."""
    detection = ctr.detect_centres(ticl3, report=False)
    shell = cand.shell_candidates(ticl3, detection.centres, report=False).shell
    assert "d projection on 1 Ti" in shell.description
    assert "count-stated AVAS" in shell.description
    assert "Ti neutral ground" in shell.description       # the reference state, stated
    assert "gap at the cut" in shell.description
    assert "[" not in shell.description


def test_the_rotated_orbitals_and_not_the_inputs_are_what_the_columns_index(ticl3):
    """⚠ AVAS rotates: a selection read against the unrotated columns is a different active
    space wearing the same description."""
    detection = ctr.detect_centres(ticl3, report=False)
    construction = cand.shell_candidates(ticl3, detection.centres, report=False)
    start = ticl3.spinors_in_ao()
    assert construction.coeff.shape == start.shape
    assert np.abs(construction.coeff - start).max() > 1e-3, "the rotation is not the identity"
    # the rotation preserves the density, which is what makes it safe at all
    occ = np.asarray(ticl3.spinors.occ)
    before = (start * occ) @ start.conj().T
    after = (construction.coeff * occ) @ construction.coeff.conj().T
    assert np.abs(before - after).max() < 1e-10


def test_shells_of_two_different_l_are_refused_rather_than_composed(ticl3):
    """⚠ Two projectors mean two rotations, and the second re-mixes the pairs the first
    selected -- they are degenerate at zero projection in it, so their basis is whatever the
    diagonalization returns. The refusal names what lifting it would take."""
    centre = ctr.detect_centres(ticl3, report=False).centres[0]
    from dataclasses import replace

    other = replace(centre, l=3)
    with pytest.raises(ValueError, match="more than one angular momentum"):
        cand.shell_candidates(ticl3, [centre, other], report=False)


def test_no_centres_is_a_refusal_and_not_an_empty_space(ticl3):
    with pytest.raises(ValueError, match="no centres"):
        cand.shell_candidates(ticl3, [], report=False)


def test_the_reports_print_the_statement_and_not_an_index_list(ticl3, kuiva_caplog):
    """The output grammar path, which is behaviour rather than decoration: a run's record of
    how its active space was chosen is what a reader has to be able to check."""
    detection = ctr.detect_centres(ticl3, report=True)
    construction = cand.shell_candidates(ticl3, detection.centres, bonding=2, report=True)
    construction.shell.report()
    construction.bonding.report()
    text = "\n".join(r.getMessage() for r in kuiva_caplog.records)
    assert "magnetic centres" in text and "1 Ti" in text
    assert "count stated" in text                       # the AVAS rule, in the report
    assert "shell candidates" in text and "bonding candidates" in text
    assert "[72," not in text, "no index list in a printed statement"


# --- the electron count (the mechanism, without an SCF) --------------------------------------

def _centre(*, stated, trusted, channels=(8.0, 12.0, 1.0, 0.0)):
    return ctr.Centre(atoms=(0,), l=2, labels=("1 Ti",), population=0.9, pooled=False,
                      reference_state="Ti(3+) [Ar]3d1", reference_channels=channels,
                      stated=stated, trusted_state=trusted)


def test_an_ordinary_shell_takes_its_electron_count_from_the_orbitals():
    """⚠ Not from the reference state, whose default is the *neutral* atom outside the f
    block -- a neutral Ti claims d2 of a d1 complex."""
    count, note = cand._shell_electrons([_centre(stated=False, trusted=False)],
                                        np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert count is None and note == ""


def test_a_shell_empty_in_the_reference_takes_the_ions_count_loudly(kuiva_caplog):
    """⚠ Measured on CeCl3: its ROHF puts the valence electron in a Ce 5d, so every 4f pair
    is empty and the measurement says CAS(0, 14) -- one determinant and no physics. The
    f-block default reference state is M(3+), which is a statement about the ion."""
    count, note = cand._shell_electrons([_centre(stated=False, trusted=True)],
                                        np.zeros(7))
    assert count == 1.0
    assert "left OUTSIDE the active space" in note
    assert any("reference state instead" in r.message for r in kuiva_caplog.records)


def test_an_empty_shell_with_only_a_neutral_atom_default_is_refused():
    """⚠ The other half of the same decision: a neutral titanium says "d2" about a d0
    complex just as confidently, so the count is not taken from it -- the refusal names
    ``configuration=``."""
    with pytest.raises(ValueError, match="NEUTRAL-ATOM default"):
        cand._shell_electrons([_centre(stated=False, trusted=False)], np.zeros(5))
    with pytest.raises(ValueError, match="configuration="):
        cand._shell_electrons([_centre(stated=False, trusted=False)], np.zeros(5))


def test_a_full_shell_is_the_same_decision_in_the_other_direction():
    count, _ = cand._shell_electrons([_centre(stated=True, trusted=True)],
                                     np.full(5, 2.0))
    assert count == 1.0, "the stated d1 reference against a shell measured as filled"


# --- the tie rule (the mechanism) -------------------------------------------------------------

def test_a_cut_inside_a_tie_is_rounded_outward():
    values = np.array([0.9, 0.4124, 0.4124, 0.1])
    assert cand._round_out(values, 2)[0] == 3
    assert "split a tie" in cand._round_out(values, 2)[1]
    assert cand._round_out(values, 1) == (1, None)
    assert cand._round_out(values, 3) == (3, None)


def test_the_tie_rule_does_not_fire_on_a_selection_order_that_is_not_a_spectrum():
    """⚠ The mechanism test for the guard that always fired.

    A ligand class orders its candidates by proximity to the Fermi level and ranks them by a
    population, so the values it hands the tie rule are **not monotone**. Read as a spectrum
    they looked like one degenerate group spanning 0.505..0.766 -- the Ti2Cl6 numbers -- and
    every bounded count rounded out to its whole pool.
    """
    values = np.array([0.5052, 0.5272, 0.7655])
    assert cand._round_out(values, 2) == (2, None)
    assert cand._round_out(values, 1) == (1, None)


def test_the_tie_rule_is_scale_free_and_takes_the_projects_own_tolerance():
    values = np.array([1e-6, 1e-6 * (1 + 1e-9), 1.0])
    assert cand._round_out(values, 1)[0] == 2, "relative, not absolute"
    assert cand._round_out(values, 1, rtol=0.0)[0] == 1


# --- the bonding class ------------------------------------------------------------------------

def test_the_bonding_pairs_are_the_sigma_combinations_below_the_shell_cut(ticl3):
    detection = ctr.detect_centres(ticl3, report=False)
    construction = cand.shell_candidates(ticl3, detection.centres, bonding=2, report=False)
    bonding = construction.bonding
    assert bonding.n_pairs == 2 and _whole_pairs(bonding.columns)
    assert np.allclose(bonding.occupations, 2.0), "they are occupied bonding combinations"
    assert bonding.electrons == 4
    assert float(bonding.ranking.max()) < float(construction.shell.ranking.min()), (
        "below the shell's cut, by construction")
    assert bonding.priority == 4 and not bonding.fixed


def test_asking_for_one_bonding_pair_returns_the_degenerate_two(ticl3):
    """⚠ The tie rule on a real degeneracy: TiCl3's two Ti-Cl sigma pairs are the ``e'``
    partners of D3h and project identically, so keeping one of them would select a subspace
    the molecule's symmetry does not leave invariant."""
    detection = ctr.detect_centres(ticl3, report=False)
    bonding = cand.shell_candidates(ticl3, detection.centres, bonding=1,
                                    report=False).bonding
    assert bonding.n_pairs == 2
    assert np.allclose(bonding.ranking, bonding.ranking[0])
    assert any("split a tie" in note for note in bonding.notes)


def test_the_antibonding_partners_are_reported_as_already_active(ticl3):
    """⚠ The shell takes every pair carrying the character, antibonding ones included, so
    what is left in the empty group has no shell character at all (0.012 on TiCl3 against
    the occupied partners' 0.265). Adding those would be pure ligand virtual orbitals in the
    name of correlating a bond whose partner is active already."""
    detection = ctr.detect_centres(ticl3, report=False)
    bonding = cand.shell_candidates(ticl3, detection.centres, bonding=2,
                                    report=False).bonding
    assert np.all(bonding.occupations > 1.5), "no empty pair was taken"
    assert any("active already" in note for note in bonding.notes)


def test_an_ionic_bond_gives_an_empty_class_that_says_why(ticl3):
    detection = ctr.detect_centres(ticl3, report=False)
    bonding = cand.shell_candidates(ticl3, detection.centres, bonding=2, bonding_floor=0.9,
                                    report=False).bonding
    assert bonding.empty and bonding.n_pairs == 0
    assert "ionic at this reference" in bonding.description


# --- the ligand classes ------------------------------------------------------------------------

def test_a_fragment_with_no_singly_occupied_pair_is_not_a_radical(ticl3):
    """⚠ A frontier target is a claim about the molecule; the chlorines of TiCl3 do not
    support it, and the refusal prints what their populations actually are."""
    with pytest.raises(ValueError, match="not a radical in this reference"):
        cand.frontier_candidates(ticl3, tg.parse(("frontier", "Cl")), report=False)


def test_a_frontier_target_can_ask_for_closed_shell_neighbours_instead(ticl3):
    target = tg.parse(("frontier", "Cl", 1, 1))
    built = cand.frontier_candidates(ticl3, target, report=False)
    candidates = built.candidates
    assert candidates.n_pairs >= 2 and _whole_pairs(candidates.columns)
    assert np.any(candidates.occupations > 1.5) and np.any(candidates.occupations < 0.5)
    assert candidates.ranking_name == "fragment population"
    assert float(candidates.ranking.min()) >= cand.DEFAULT_FRAGMENT_THRESHOLD
    assert "after rotating the unclaimed pairs" in candidates.description


def test_the_frontier_rotation_is_unique_where_the_column_basis_is_not(ticl3):
    """⚠ The mechanism test for the run-to-run instability.

    A rotation applied to an *arbitrary* basis of the free span must give the same answer as
    one applied to any other basis of it, because it diagonalizes a physical operator on that
    span. Here the input basis is scrambled deliberately -- a random unitary inside the
    doubly occupied group, which is what a degenerate diagonalization is free to return -- and
    the selected populations have to come back identical. Without the rotation the selection
    changed between identical runs on Ti2Cl6.
    """
    target = tg.parse(("frontier", "Cl", 1, 1))
    first = cand.frontier_candidates(ticl3, target, report=False)

    coeff = np.array(ticl3.spinors_in_ao(), copy=True)
    occ = np.asarray(ticl3.spinors.occ, dtype=float)
    pair_occ = cand._pair_occupations(occ)
    group = np.nonzero(pair_occ > 1.5)[0]
    rng = np.random.default_rng(17)
    u = np.linalg.qr(rng.normal(size=(group.size, group.size))
                     + 1j * rng.normal(size=(group.size, group.size)))[0]
    full = np.eye(pair_occ.size, dtype=complex)
    full[np.ix_(group, group)] = u
    from kuiva.spinor.expand import rotate_kramers_pairs

    scrambled = rotate_kramers_pairs(coeff, full, np.arange(coeff.shape[1]))
    second = cand.frontier_candidates(ticl3, target, coeff=scrambled, occupation=occ,
                                      report=False)
    assert np.allclose(np.sort(first.candidates.ranking),
                       np.sort(second.candidates.ranking), atol=1e-8), (
        first.candidates.ranking, second.candidates.ranking)


def test_the_fragment_rotation_moves_no_density_and_keeps_the_pairing(ticl3):
    """The two invariances that make a rotation inside occupation groups safe at all -- the
    same pair the projection itself has to satisfy, and the conjugate-partner trap with it."""
    from kuiva.spinor.expand import spin_block_diagonal, time_reverse

    coeff = ticl3.spinors_in_ao()
    occ = np.asarray(ticl3.spinors.occ, dtype=float)
    rotated, populations = cand.fragment_rotation(
        ticl3, coeff, (1, 2, 3), pair_occupations=cand._pair_occupations(occ))
    s2 = spin_block_diagonal(np.asarray(ticl3.data.s_ao))
    before = (coeff * occ) @ coeff.conj().T
    after = (rotated * occ) @ rotated.conj().T
    assert np.abs(before - after).max() < 1e-10, "the reference density may not move"
    gram = rotated.conj().T @ s2 @ rotated
    assert np.abs(gram - np.eye(gram.shape[0])).max() < 1e-9
    overlap = np.sum(np.conj(rotated[:, 1::2]) * (s2 @ time_reverse(rotated[:, ::2])), axis=0)
    assert np.abs(1.0 - np.abs(overlap)).max() < 1e-9, "Kramers pairing survives"
    assert populations.min() > -1e-10 and populations.max() < 1.0 + 1e-10


def test_a_protected_quantity_confines_the_rotation_to_its_degenerate_blocks(ticl3):
    """⚠ The rule that keeps two ligand classes composable.

    A later bridge round ranks by the shell's projections, so a frontier rotation may not mix
    pairs whose projections differ -- that would destroy the number. Pairs with a *unique*
    projection must therefore come back untouched, column for column, while the null space
    (all of it degenerate at zero) is free to rotate.
    """
    detection = ctr.detect_centres(ticl3, report=False)
    construction = cand.shell_candidates(ticl3, detection.centres, report=False)
    values = np.asarray(construction.avas.eigenvalues)
    pair_occ = np.asarray(construction.avas.occupations)
    shell_pairs = np.asarray(construction.shell.columns, dtype=int)[0::2] // 2
    rotated, _ = cand.fragment_rotation(
        ticl3, construction.coeff, (1, 2, 3), pair_occupations=pair_occ,
        frozen=shell_pairs, protect=values, rtol=1e-6)

    free = [p for p in range(pair_occ.size) if p not in set(shell_pairs.tolist())]
    unique = [p for p in free
              if np.count_nonzero(np.abs(values[free] - values[p]) < 1e-8 * max(values[p], 1.0)) == 1
              and values[p] > 1e-3]
    assert unique, "the projection spectrum has non-degenerate members outside the shell"
    for p in unique:
        assert np.abs(rotated[:, 2 * p:2 * p + 2]
                      - construction.coeff[:, 2 * p:2 * p + 2]).max() < 1e-12, p
    # and something did move: the null space is degenerate and therefore free
    assert np.abs(rotated - construction.coeff).max() > 1e-3


def test_the_degenerate_blocks_are_the_projects_own_grouping():
    values = np.array([0.9, 0.5, 0.5, 0.5, 0.0, 0.0])
    blocks = cand._degenerate_blocks(np.arange(6), values, 1e-6)
    assert [b.tolist() for b in blocks] == [[0], [1, 2, 3], [4, 5]]
    assert [b.tolist() for b in cand._degenerate_blocks(np.arange(6), None, 1e-6)] == [
        list(range(6))], "no protected quantity means one block"
    assert cand._degenerate_blocks(np.zeros(0, dtype=int), values, 1e-6) == []


def test_asking_for_more_pairs_than_the_fragment_has_is_refused_with_the_count(ticl3):
    with pytest.raises(ValueError, match="were asked for"):
        cand.frontier_candidates(ticl3, tg.parse(("frontier", "Ti", 0, 99)), report=False)


def test_a_bridge_whose_two_sites_share_an_atom_is_refused(ticl3):
    detection = ctr.detect_centres(ticl3, report=False)
    construction = cand.shell_candidates(ticl3, detection.centres, report=False)
    with pytest.raises(ValueError, match="share atom"):
        cand.bridge_candidates(ticl3, tg.parse(("bridge", ("Ti", "Ti"))),
                               construction.avas, report=False)


def test_a_bridging_atom_that_is_also_a_site_atom_is_refused(ticl3):
    detection = ctr.detect_centres(ticl3, report=False)
    construction = cand.shell_candidates(ticl3, detection.centres, report=False)
    with pytest.raises(ValueError, match="also site atoms"):
        cand.bridge_candidates(ticl3, tg.parse(("bridge", (1, 2), (2,))),
                               construction.avas, report=False)


def test_a_bridge_with_nothing_both_on_it_and_mixed_with_the_shell_is_refused(ticl3):
    """⚠ Two ways to have no pathway, and the message distinguishes them: nothing sits on
    those atoms, or what does is orthogonal to the metal shell -- which is a statement about
    the molecule and not a threshold to lower."""
    detection = ctr.detect_centres(ticl3, report=False)
    construction = cand.shell_candidates(ticl3, detection.centres, report=False)
    with pytest.raises(ValueError, match="no Kramers pair is both on"):
        cand.bridge_candidates(ticl3, tg.parse(("bridge", (1, 2), (3,))),
                               construction.avas, threshold=0.999, report=False)
    with pytest.raises(ValueError, match="no superexchange pathway"):
        cand.bridge_candidates(ticl3, tg.parse(("bridge", (1, 2), (3,))),
                               construction.avas, floor=0.99, report=False)


# --- the overlap refusal -----------------------------------------------------------------------

def _set(cls, columns):
    columns = np.asarray(columns, dtype=int)
    return cand.CandidateSet(cls=cls, columns=columns, description="the {} set".format(cls),
                             occupations=np.zeros(columns.size // 2),
                             ranking=np.zeros(columns.size // 2))


def test_two_classes_claiming_one_pair_are_refused_naming_both_statements():
    with pytest.raises(ValueError, match="both claim spinor"):
        cand.check_disjoint([_set("bridge", [4, 5, 6, 7]), _set("bonding", [6, 7, 8, 9])])
    cand.check_disjoint([_set("bridge", [4, 5]), _set("bonding", [6, 7])])
    cand.check_disjoint([_set("bridge", [4, 5]), _set("bonding", [])])


# --- the slow systems ---------------------------------------------------------------------------

@pytest.mark.slow
def test_the_cerium_f_shell_is_empty_in_the_reference_and_still_holds_one_electron():
    """⚠ The case the electron-count decision exists for, end to end on a real molecule."""
    reference = reference_for("cecl3")
    detection = ctr.detect_centres(reference, report=False)
    construction = cand.shell_candidates(reference, detection.centres, double=True,
                                         report=False)
    shell = construction.shell
    assert shell.n_pairs == 7 and _whole_pairs(shell.columns)
    assert np.allclose(shell.occupations, 0.0), "every 4f pair is empty in this ROHF"
    assert shell.electrons == 1.0, "the Ce(3+) reference state supplies the count"
    assert any("reference state instead" in note for note in shell.notes)
    double = construction.double
    assert double.n_pairs == 7 and double.fixed and double.electrons == 0.0
    assert float(double.ranking.max()) <= float(shell.ranking.min()) + 1e-12
    cand.check_disjoint(construction.sets)


@pytest.mark.slow
def test_the_dimers_pooled_shell_spans_the_committed_dimer_space():
    reference = reference_for("ti2cl6")
    system = sysdef.get("ti2cl6")
    stated = api.active_space_for(reference, **sysdef.character_selection(system))
    detection = ctr.detect_centres(reference, report=False)
    construction = cand.shell_candidates(reference, detection.centres, report=False)
    shell = construction.shell
    assert list(shell.columns) == list(stated.spaces.active)
    assert shell.n_pairs == system.ncas and shell.electrons == system.nelecas
    # ⚠ The same columns and **not** the same space: measured principal overlaps 0.994 down
    # to 0.896 on nine of the ten pairs, and 5e-11 on the tenth. The character rule's tenth
    # "d" pair and the rotation's tenth d-like combination are different orbitals -- the
    # rotation built its own out of the virtual space, which is what AVAS is for on a
    # covalently bridged dimer. Asserted as a count, because which pair it is is not the
    # point and the two numbers either side of the gap are far apart.
    overlaps = _principal_overlaps(reference, reference.spinors_in_ao(),
                                   stated.spaces.active, construction.coeff, shell.columns)
    assert np.count_nonzero(overlaps > 0.85) == 18, overlaps
    assert np.count_nonzero(overlaps < 1e-6) == 2, overlaps


@pytest.mark.slow
def test_the_bridge_orbitals_of_the_dimer_are_the_pairs_that_mix_with_the_metal():
    """⚠ The measurement that fixed the ranking: ordered by *population* the two "bridge"
    pairs came back as deep chlorine orbitals (columns 52 and 122, populations 0.72 and
    0.66); ordered by proximity to the Fermi level they are the pairs just below the shell
    cut, which are the ones carrying metal character."""
    reference = reference_for("ti2cl6")
    detection = ctr.detect_centres(reference, report=False)
    construction = cand.shell_candidates(reference, detection.centres, bonding=2,
                                         report=False)
    bridge = cand.bridge_candidates(
        reference, tg.parse(("bridge", (1, 2))), construction.avas,
        exclude=np.concatenate([construction.shell.columns,
                                construction.bonding.columns]), report=False)
    assert bridge.n_pairs >= 1 and _whole_pairs(bridge.columns)
    assert "3 Cl+4 Cl" in bridge.description, "the bridging chlorines, detected by contact"
    assert "covalent contact" in bridge.description
    # ⚠ Every candidate mixes with the metal shell -- which is what a pathway is, and what
    # keeps the selection out of the projection's arbitrary null space.
    assert float(bridge.ranking.min()) >= cand.DEFAULT_BONDING_FLOOR
    assert bridge.ranking_name == "AVAS projection onto the shell"
    assert int(np.max(bridge.columns)) < int(np.min(construction.shell.columns))
    cand.check_disjoint([construction.shell, construction.bonding, bridge])
