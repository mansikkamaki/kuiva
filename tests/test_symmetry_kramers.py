"""The combination: Kramers-restricted CI **and** abelian double-group labels at once.

Two symmetries of the same Hamiltonian, imposed together. They commute, so the combination is
well posed — but they do not act on the same object, and the whole content of this file is the
one structural fact that follows:

⚠ **Time reversal CONJUGATES an irrep label.** ``T`` maps the determinants of a sector onto
those of its conjugate sector, so a sector is time-reversal-closed only when it is
self-conjugate, and otherwise only the **union of a conjugate pair** is. The indivisible unit
of a Kramers-restricted per-irrep selection is therefore that pair, not a sector: asking for
``n`` states of an irrep also returns the ``n`` time-reversed partners in its conjugate, and
there is no calculation in which one exists without the other.

What each test here can fail on:

* the combined solve against the **general path's own spectrum** — the reference path,
  restricted to the same sectors, on the same integrals. Nothing in the combination can be
  right if that disagrees;
* the pair degeneracy, asserted at **exactly zero** rather than at a band, because in this
  mode it is a property of the subspace and not a numerical outcome. ⚠ That it comes out exact
  is a side effect and never the argument for the mode: the mode is for the factor of two;
* the **state-averaging gate**, which the combination reaches by a different route than either
  half. A plain per-irrep selection of two states of one sector *splits Kramers pairs* — the
  partners are in the conjugate sector and are not selected — and the gate refuses it. The
  combined mode is what makes that request well defined, and this file pins both halves of
  that statement;
* the **refusals** the combination adds: both members of a conjugate pair requested (they are
  one unit), an odd count in a self-conjugate sector (it would split a pair inside it), and an
  active space whose conjugate sector is missing altogether;
* the classification of the resulting states, which must be **per degenerate block**: a
  Kramers pair spanning two conjugate sectors is exactly degenerate, so the eigensolver may
  return any rotation inside it and half of each state lands in each sector. Calling that
  impure orbitals would be a warning about the eigensolver's freedom.
"""
import numpy as np
import pytest

from kuiva.ci.davidson import davidson_kramers_sector
from kuiva.ci.strings import CASSpace
from kuiva.interface.api import Molecule, active_space_for, spinor_reference
from kuiva.mcscf.casci import FullCISolver
from kuiva.mcscf.orbopt import CASIntegrals

#: The two paths are the same eigenproblem reached two ways, so the only difference is the
#: second-order effect of a Davidson residual on a Ritz value.
ENERGY_TOL = 1e-10

#: An odd electron count in a Kramers-paired active space, which is what the restricted mode
#: is defined for. The active space is a window of N2's own spinors; the electron count is the
#: solver's, so this is one well-defined Hamiltonian eigenproblem rather than a claim about
#: the neutral molecule.
N_ACTIVE_ELEC = 5


@pytest.fixture(scope="module")
def n2_labelled():
    """``(active integrals, active-spinor labels)`` for N2 in ``C2h(z)``."""
    molecule = Molecule(atoms=[("N", (0.0, 0.0, 0.55)), ("N", (0.0, 0.0, -0.55))],
                        basis="x2c-SVPall-2c", point_group="auto")
    # screening="none": the symmetry work changes no scalar quantity, so the four-component
    # atomic solve the default would pay for is pure cost here.
    reference = spinor_reference(molecule, screening="none", memory_gb=6.0)
    space = active_space_for(reference, active=list(range(8, 16)), n_active_elec=6)
    ints = CASIntegrals.build(reference.factors, reference.h_one_electron(),
                              reference.spinors_in_ao(), space.spaces,
                              e_nuc=reference.data.e_nuc)
    return (np.ascontiguousarray(ints.h_active_effective()), ints.active_eri(), space.labels)


def _solver(labels, **kwargs):
    return FullCISolver(len(labels), N_ACTIVE_ELEC, symmetry=labels, **kwargs)


def _sector_names(labels):
    return [t for t in labels.group.labels(fermion=True)]


# --- The combined solve against the general path ------------------------------------------

def test_a_combined_solve_reproduces_the_general_sector_spectrum(n2_labelled):
    """⚠ The comparison that can fail on the new code alone: the general complex path is the
    reference path, and the energies of a conjugate sector pair are the same numbers however
    they are reached."""
    h, eri, labels = n2_labelled
    group = labels.group
    reference = _solver(labels, n_states=16)
    plain = reference.solve_active(h, eri)
    table = reference._sectors
    for label in table.sectors:
        conjugate = group.conjugate(label)
        if group.irrep_name(label) > group.irrep_name(conjugate):
            continue                                   # each pair once
        name = table.name(label)
        combined = _solver(labels, n_states={name: 2}, kramers="restricted")
        result = combined.solve_active(h, eri)
        assert combined.n_states == 4                  # 2 of the irrep, 2 of its conjugate
        # the same roots picked out of the general lowest-n spectrum by their sector weights
        weight = table.sector_weights(plain.vectors)
        columns = [table.sectors.index(label), table.sectors.index(conjugate)]
        inside = np.nonzero(weight[:, columns].sum(axis=1) > 1.0 - 1e-8)[0]
        assert np.allclose(np.sort(result.energies),
                           np.sort(plain.energies[inside][:4]), atol=ENERGY_TOL)


def test_the_pair_degeneracy_is_exact_and_the_states_span_the_conjugate_pair(n2_labelled):
    """The pairing is a property of the subspace here, so it is asserted **at zero**. ⚠ It is
    a side effect of the mode and never the argument for it (that is the factor of two): the
    general path's own Kramers splitting is 1e-15..1e-13 Eh and is a numerical statement."""
    h, eri, labels = n2_labelled
    group = labels.group
    combined = _solver(labels, n_states={"1E1/2u": 3}, kramers="restricted")
    result = combined.solve_active(h, eri)
    assert combined.n_states == 6
    for pair in range(3):
        assert result.energies[2 * pair] == result.energies[2 * pair + 1]
    weight = combined._sectors.sector_weights(result.vectors)
    unit = [combined._sectors.sectors.index(t)
            for t in (group.label_of("1E1/2u"), group.conjugate(group.label_of("1E1/2u")))]
    assert np.allclose(weight[:, unit].sum(axis=1), 1.0, atol=1e-10)
    # ⚠ and the two sectors carry equal total weight: the states are the pair, whatever
    # rotation inside each doublet the eigensolver happened to return
    assert abs(weight[:, unit[0]].sum() - weight[:, unit[1]].sum()) < 1e-8


def test_the_rdms_and_the_energy_agree_with_the_general_path(n2_labelled):
    """The solver contract: whatever a mode does internally, what comes out is the same objects
    in the same convention. Here the two routes select the *same* states, so the
    state-averaged density must agree to the RDM tolerance.

    ⚠ **The count is four pairs, and the reason is the rule this project applies everywhere
    else: a state count must land on a manifold boundary.** At three pairs it does not — in
    this spectrum the third pair sits **9.07e-07 Eh** from the fourth, and a density averaged
    over one of two levels that close inherits the eigenvector resolution floor
    ``~eps*||H||/gap`` ≈ 2.7e-09, which is *above* the 1e-9 this test asserts. Measured
    consequence, and it is why this test failed intermittently before 2026-08-29: at three
    pairs ``d(2-RDM)`` swings 1.8e-10 … 8.4e-10 with nothing but BLAS reduction order, up to
    **7x larger** than the 1-RDM error beside it. At four pairs the boundary gap is 3.6e-02 Eh,
    the two-particle error *equals* the one-particle error (6.1e-11 … 1.9e-10 over one to eight
    threads — pure rounding), and the tolerance below is unchanged. Tightening the Davidson
    criterion does nothing: both solves are already converged past 1e-12, so this was never a
    convergence question.
    """
    h, eri, labels = n2_labelled
    n_pairs, n_sel = 4, 8
    combined = _solver(labels, n_states={"1E1/2u": n_pairs}, kramers="restricted")
    result = combined.solve_active(h, eri)
    # The general path's own states, restricted to the same conjugate pair of sectors and
    # averaged with the same equalized weights. ⚠ A state-averaged density over *complete*
    # degenerate blocks is invariant under the rotation the eigensolver chose inside them,
    # which is what makes this comparison meaningful at all.
    from kuiva.rdm.rdm import cas_rdms
    general = _solver(labels, n_states=16, kramers="general")
    plain = general.solve_active(h, eri)
    table = general._sectors
    group = labels.group
    unit = [table.sectors.index(group.label_of(n)) for n in ("1E1/2u", "2E1/2u")]
    weight = table.sector_weights(plain.vectors)
    qualifying = np.nonzero(weight[:, unit].sum(axis=1) > 1.0 - 1e-8)[0]
    inside = qualifying[:n_sel]
    assert inside.size == n_sel
    # ⚠ The guard that keeps the assertions below meaningful, and the one that would have
    # caught this: the count must not cut a near-degenerate group. A basis, geometry or
    # threshold change that closes this gap makes the RDM comparison a measurement of
    # eigenvector resolution rather than of the solver contract — and it fails *here*, saying
    # so, instead of failing one run in several on a tolerance that looks arbitrary.
    rest = [int(i) for i in qualifying if int(i) not in set(inside.tolist())]
    boundary_gap = (abs(plain.energies[rest[0]] - plain.energies[inside[-1]]) if rest
                    else float("inf"))
    assert boundary_gap > 1.0e-3, (
        "the {}-state selection ends {:.3e} Eh from the next state of the same sector pair; "
        "two levels that close are resolved as vectors only to ~eps*||H||/gap, so the RDM "
        "comparison below would measure that and not the solver contract"
        .format(n_sel, boundary_gap))
    assert np.allclose(np.sort(result.energies), np.sort(plain.energies[inside]),
                       atol=ENERGY_TOL)
    gamma, gamma2 = cas_rdms(general.space, plain.vectors[inside],
                             np.full(n_sel, 1.0 / n_sel), enforce_kramers=False)
    assert np.abs(result.gamma - gamma).max() < 1e-9
    assert np.abs(result.gamma2 - gamma2).max() < 1e-9


def test_the_states_come_back_in_the_general_convention(n2_labelled):
    """Hermitian densities, the right trace, ascending energies — nothing downstream learns
    that two symmetries were used."""
    h, eri, labels = n2_labelled
    result = _solver(labels, n_states={"1E1/2g": 2},
                     kramers="restricted").solve_active(h, eri)
    assert np.all(np.diff(result.energies) >= -1e-12)
    assert np.abs(result.gamma - result.gamma.conj().T).max() < 1e-12
    assert abs(np.trace(result.gamma).real - N_ACTIVE_ELEC) < 1e-10
    assert abs(result.weights.sum() - 1.0) < 1e-12


# --- What the combination refuses ------------------------------------------------------------

def test_both_members_of_a_conjugate_pair_cannot_be_requested(n2_labelled):
    """They are one unit; counting them separately would count the same states twice."""
    _, _, labels = n2_labelled
    with pytest.raises(ValueError, match="conjugate pair"):
        _solver(labels, n_states={"1E1/2u": 2, "2E1/2u": 2}, kramers="restricted")


def test_an_odd_count_in_a_self_conjugate_sector_is_refused():
    """⚠ A self-conjugate sector *is* time-reversal-closed, so its own states pair up inside
    it and an odd count splits one. ``Ci`` is where this happens: conjugation is the identity
    on its labels."""
    from kuiva.symm.assign import OrbitalLabels
    from kuiva.symm.groups import CI
    labels = OrbitalLabels(group=CI, labels=CI.spinor_labels(
        np.array([[0, 0], [0, 1], [0, 0], [0, 1]])))
    assert CI.conjugate((1, 0)) == (1, 0)
    with pytest.raises(ValueError, match="self-conjugate"):
        FullCISolver(8, 5, n_states={"E1/2g": 3}, symmetry=labels, kramers="restricted")
    solver = FullCISolver(8, 5, n_states={"E1/2g": 4}, symmetry=labels, kramers="restricted")
    assert solver.n_states == 4                       # the sector's own, not doubled


def test_an_even_electron_count_is_still_the_other_theorem(n2_labelled):
    """The restricted mode is the odd-``N`` theorem whatever the labels say."""
    _, _, labels = n2_labelled
    # An even-electron determinant space has boson sectors, so the request has to name one
    # for the electron-count refusal to be the one that fires.
    with pytest.raises(ValueError, match="odd-electron theorem"):
        FullCISolver(len(labels), 6, n_states={"Ag": 2}, symmetry=labels,
                     kramers="restricted")


def test_a_subspace_that_is_not_time_reversal_closed_is_refused(n2_labelled):
    """The eigensolver's own refusal, one level down: a bare sector is not ``T``-closed."""
    h, eri, labels = n2_labelled
    solver = _solver(labels, n_states=2)
    space = CASSpace(len(labels), N_ACTIVE_ELEC)
    mask = solver._sectors.mask("1E1/2u")
    with pytest.raises(ValueError, match="not closed under time reversal"):
        space.kramers().restrict(mask)


# --- The gate, which is what the combination is for -------------------------------------------

def test_a_plain_per_irrep_selection_of_the_same_states_splits_kramers_pairs(n2_labelled):
    """⚠ **The failure the combination removes.** Asking the general path for two states of one
    sector selects two states whose time-reversed partners live in the *conjugate* sector and
    are not selected — so the average is over half of each Kramers pair, and the gate refuses
    it. Under ``kramers="restricted"`` the same request is well defined, because the unit of
    selection is the conjugate pair."""
    h, eri, labels = n2_labelled
    plain = _solver(labels, n_states={"1E1/2u": 2})
    with pytest.raises(ValueError, match="split a Kramers-degenerate block"):
        plain.solve_active(h, eri)
    combined = _solver(labels, n_states={"1E1/2u": 2}, kramers="restricted")
    combined.solve_active(h, eri)                     # the same request, now well posed


def test_the_gate_still_runs_and_equalizes_whole_blocks(n2_labelled):
    """The combination reimplements no part of the state-averaging gate: the same code, on
    the same objects. A mode that agreed numerically while running its own gate would still be
    wrong."""
    h, eri, labels = n2_labelled
    result = _solver(labels, n_states={"1E1/2u": 2},
                     kramers="restricted").solve_active(h, eri)
    for start in range(0, result.energies.size, 2):
        assert result.weights[start] == result.weights[start + 1]


def test_classification_of_a_combined_solve_is_per_degenerate_block(n2_labelled):
    """⚠ A Kramers pair spanning two conjugate sectors is exactly degenerate, so any rotation
    inside it is as good an eigenvector and a *single* state has no sector. The block is what
    is invariant, and the leakage reported must be zero — it is a statement about the
    orbitals, not about the eigensolver's freedom."""
    h, eri, labels = n2_labelled
    result = _solver(labels, n_states={"1E1/2u": 2},
                     kramers="restricted").solve_active(h, eri)
    assert result.irreps is not None
    assert result.sector_leakage < 1e-9
    for start in range(0, len(result.irreps), 2):
        assert result.irreps[start] == result.irreps[start + 1]
        assert "1E1/2u" in result.irreps[start] and "2E1/2u" in result.irreps[start]


# --- Bookkeeping ------------------------------------------------------------------------------

def test_the_space_key_distinguishes_the_combination(n2_labelled):
    """⚠ Curvature memory is chart-scoped, so a mode switch must not silently reuse a warm
    start from the other one. The key moves only for the non-default modes, so a run that
    never used either keeps every checkpoint it ever wrote."""
    _, _, labels = n2_labelled
    plain = _solver(labels, n_states=4).space_key()
    sector = _solver(labels, n_states={"1E1/2u": 4}).space_key()
    combined = _solver(labels, n_states={"1E1/2u": 2}, kramers="restricted").space_key()
    kramers = FullCISolver(len(labels), N_ACTIVE_ELEC, n_states=4,
                           kramers="restricted").space_key()
    assert len({plain, sector, combined, kramers}) == 4
    assert FullCISolver(len(labels), N_ACTIVE_ELEC, n_states=4).space_key() == \
        FullCISolver(len(labels), N_ACTIVE_ELEC, n_states=4).space_key()


def test_the_boundary_diagnostic_runs_in_the_combined_mode(n2_labelled):
    """The per-sector boundary report is the same code on the same objects; in this mode its
    spectrum comes from a conjugate-pair solve and is thinned back to the requested sector's
    own states, so the index it cuts on still means what it meant."""
    h, eri, labels = n2_labelled
    solver = _solver(labels, n_states={"1E1/2u": 2}, kramers="restricted")
    solver.solve_active(h, eri)
    spectra = solver.sector_spectra(h, eri, 2)
    label = labels.group.label_of("1E1/2u")
    assert spectra[label].size >= 4
    assert np.all(np.diff(spectra[label]) >= -1e-12)
