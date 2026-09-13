"""Resolving a target into candidate Kramers pairs, with the statement that produced them.

One constructor per target class. Each returns a :class:`CandidateSet` -- whole Kramers
pairs, whole degenerate groups, a description another program could reproduce -- over **one**
orbital set, which is what makes the columns of different classes comparable at all.

Where the orbitals come from
----------------------------
The shell classes *rotate*: a shell is an AVAS projection onto the free-atom reference
orbitals (:func:`kuiva.mcscf.avas.avas`), in its **count-stated** mode, because a shell is
``2l+1`` pairs per centre whatever the next pair's projection is. The ligand classes do not
rotate: they select pairs of the set they are given by Loewdin population on a fragment.

⚠ **Therefore the shell is constructed once, for every class that reads its projection.**
The bonding partners and the bridge candidates are read off that call's eigenvalue spectrum,
and the double shell is built *after* it, by a second rotation confined to the empty pairs the
shell left over at (near) zero projection -- so asking for it changes neither the shell nor
the spectrum the other classes read. :func:`shell_candidates` is where all three come from.

⚠ **One rotation per orbital set, never two composed -- and shells of different ``l`` are
one projection onto the union.** A heterometallic 3d/4f pair (or a 4f shell with a 5d partner)
is not two AVAS calls: the second call's rotation mixes the pairs the first call selected,
because within an occupation group the pairs outside the second projector's span are
degenerate at eigenvalue zero and their basis is whatever the diagonalization returns
(measured: a sequential O 2p / H 1s pair of calls on water lost part of the first selection,
by a different amount on two identical runs, and a Ti 3d / Ce 4f pair lost all of it). It is one count-stated projection
onto the **union** of the centres' reference spans (``shells=`` of
:func:`kuiva.mcscf.avas.avas_projection`), with each selected pair attributed to the centre
whose shell it projects onto most -- a diagonal read in the rotated orbitals, not a rotation.
Everything a shell carries per centre (its pairs, its electron count, its floor) follows that
attribution, and a count that does not come out as ``2l+1`` per atom **warns**.

⚠ **Shells of one ``l`` keep the single projection they always had**, bitwise, and their
per-centre attribution by Loewdin population: the union form is taken only where the centres
have more than one ``l``. Two equivalent centres of one ``l`` have no shell-by-shell
projection to tell them apart anyway -- only a localization can, later.

⚠ **No ligand class may select a pair out of the projection's null space, and that is a
correctness rule rather than a preference.** The pairs of an occupation group that AVAS did
not select are degenerate at *zero* projection, so the diagonalization returns an arbitrary
basis of their span -- and "arbitrary" turned out to mean **not reproducible between
identical runs**: measured on Ti2Cl6, three runs of the same script at the same thread width
selected three different bridge pairs (populations 0.73, 0.63 and one whose neighbour had
0.60), because the SCF's own last bits move and the degenerate block reshuffles. An active
space that changes from run to run is not an active space.

The two classes handle it in the only two ways that give a unique answer, and neither is a
threshold on a basis nobody chose:

* **Bridge ranks by the projection onto the shell and refuses the null space.** A pair whose
  projection exceeds :data:`DEFAULT_BONDING_FLOOR` is a non-degenerate eigenvector and is
  therefore unique -- and it is also the only kind of bridge orbital that means anything: a
  ligand pair *orthogonal* to the metal shell carries no superexchange. Same spectrum as the
  bonding class, one filter apart (which atoms the ligand character sits on).
* **Frontier rotates the free complement onto its own fragment.** The *span* of what AVAS did
  not select is unique even when its basis is not, so diagonalizing a physical operator on
  that span -- the fragment's Loewdin population -- fixes the basis uniquely and by
  construction leaves the claimed pairs alone. The eigenvalues *are* the populations the
  class selects on. ⚠ And the rotation is confined to pairs whose **projection is degenerate**
  (:func:`fragment_rotation`'s ``protect``): mixing two pairs of different projection would
  destroy the very number a later bridge round ranks by, while mixing degenerate ones
  destroys nothing, because there no basis was defined to begin with.

Rules every constructor obeys
-----------------------------
* **Whole Kramers pairs**, always: a space is defined on spatial orbitals and a pair may not
  be split across a boundary.
* **Whole degenerate groups**: a bounded count is rounded *outward* rather than split a tie
  in the quantity the class ranks by (:func:`_round_out`, over
  :mod:`kuiva.util.degeneracy`'s one definition of how close is close), and the set says so.
  The ranking quantity rather than the orbital energy, because after a rotation an orbital
  energy is not defined while the quantity the class ranked by always is -- and for symmetry
  partners, which is the case the rule exists for, the two agree to machine precision.
* **A description and never an index list.** The columns mean nothing outside the orbital set
  they came from; the statement is what a reference file or a property dump can carry.

References
----------
* B. Cordero, V. Gomez, A. E. Platero-Prats, M. Reves, J. Echeverria, E. Cremades,
  F. Barragan, S. Alvarez, "Covalent radii revisited", Dalton Trans. 2832 (2008),
  doi:10.1039/b801115j -- :data:`COVALENT_RADII_ANGSTROM`, used only to decide which atoms
  *touch* both sites when bridging atoms are not named.
* The AVAS projection and the SPADE fragment populations are cited where they are
  implemented (:mod:`kuiva.mcscf.avas`, :mod:`kuiva.mcscf.localize`); nothing here
  re-derives either.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util.degeneracy import DEFAULT_GROUP_RTOL, relative_gap
from ..util.logging import get_logger
from . import centres as ctr
from . import targets as tg

log = get_logger(__name__)

__all__ = ["COVALENT_RADII_ANGSTROM", "CandidateSet", "DEFAULT_BONDING_FLOOR",
           "DEFAULT_CONTACT_SCALE", "DEFAULT_FRAGMENT_THRESHOLD", "DEFAULT_RANKING_RTOL",
           "LigandConstruction", "ShellConstruction", "bonding_candidates",
           "bridge_atoms_by_contact", "bridge_candidates", "check_disjoint",
           "fragment_rotation", "frontier_candidates", "shell_candidates"]

#: Fraction of a Kramers pair's Loewdin population that has to sit on a fragment for the pair
#: to be a candidate of that fragment. The same 0.5 the centre detection and the site
#: partition use: below half, the orbital is not the fragment's orbital.
DEFAULT_FRAGMENT_THRESHOLD = 0.5

#: A bonding partner has to carry at least this much of the shell's character. Below it the
#: class is **empty and says so**: an ionic metal-ligand bond has no bonding combination to
#: correlate, and an empty class is information rather than a failure.
DEFAULT_BONDING_FLOOR = 0.05

#: Two atoms are in contact when their distance is within this multiple of the sum of their
#: covalent radii. Generous on purpose: it decides which atoms are *offered* as bridge
#: candidates, the candidate count bounds what is taken, and naming the atoms overrides it.
DEFAULT_CONTACT_SCALE = 1.3

#: Relative gap below which two neighbouring ranking values are one degenerate group. The
#: project's own default, so that "what is a degenerate group" has one answer.
DEFAULT_RANKING_RTOL = DEFAULT_GROUP_RTOL

#: Bohr per Angstrom (CODATA 2018), the literal :mod:`kuiva.interface.environment` uses. The
#: radii below are published in Angstrom and kept in that unit so they can be checked against
#: the paper; the geometry is in bohr.
_BOHR_PER_ANGSTROM = 1.0 / 0.52917721092

#: Covalent radii [Angstrom] from Cordero et al. (2008); the low-spin values for Mn, Fe and
#: Co and the sp3 value for carbon, which are the paper's own defaults. ⚠ The table stops at
#: Cm, where the paper does: an element beyond it is refused with the advice to name the
#: bridging atoms, never given a guessed radius.
COVALENT_RADII_ANGSTROM: Dict[str, float] = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39,
    "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20,
    "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47,
    "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39,
    "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99,
    "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89,
    "Tm": 1.90, "Yb": 1.87, "Lu": 1.87, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51,
    "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46,
    "Bi": 1.48, "Po": 1.40, "At": 1.50, "Rn": 1.50,
    "Fr": 2.60, "Ra": 2.21, "Ac": 2.15, "Th": 2.06, "Pa": 2.00, "U": 1.96, "Np": 1.90,
    "Pu": 1.87, "Am": 1.80, "Cm": 1.69,
}


@dataclass(frozen=True)
class CandidateSet:
    """Kramers pairs one target resolved to, and the statement that produced them.

    ``columns`` are spinor indices into the orbital set the constructor was given -- whole
    pairs, sorted. ``occupations`` are the **reference** electron counts of those pairs (2 for
    a closed pair, 1 for a singly occupied one, 0 for an empty one), which is where the
    active space's electron count comes from: measured off the selected orbitals, never
    inferred from an oxidation state.

    ``ranking`` is the quantity the selection ordered by, one value per selected pair, in
    column order -- an AVAS projection for the shell classes, a fragment population for the
    ligand ones. It is what a report prints beside each pair and what the degenerate-group
    rounding was applied to.
    """

    cls: str
    columns: np.ndarray
    description: str
    occupations: np.ndarray
    ranking: np.ndarray
    ranking_name: str = ""
    fixed: bool = False
    #: An electron count that is **not** the sum of :attr:`occupations`, set only where the
    #: measurement cannot supply one (:func:`_shell_electrons`). ``None`` otherwise, which is
    #: every ordinary case.
    stated_electrons: Optional[float] = None
    fragments: Tuple[Tuple[int, ...], ...] = ()
    #: What the selection did beyond the plain rule: a degenerate group it had to round out
    #: to, a gap at the cut, an empty class's reason. Printed, never silent.
    notes: Tuple[str, ...] = ()
    #: Electrons per centre, in the order of the centres the shell was built for; set on a
    #: shell only. ⚠ What the floors and the Hund product are computed from: with centres of
    #: one ``l`` the shell's electrons divided over its atoms, with centres of different ``l``
    #: the count of the pairs attributed to each (a Ti d^1 beside a Ce f^1 is not two
    #: "one-electron atoms" of one kind).
    centre_electrons: Tuple[float, ...] = ()

    @property
    def priority(self) -> int:
        return tg.PRIORITY[self.cls]

    @property
    def n_pairs(self) -> int:
        return int(np.size(self.columns)) // 2

    @property
    def electrons(self) -> float:
        """Electrons the class contributes -- measured, unless :attr:`stated_electrons` is
        set because the measurement could not say (a shell empty in the reference)."""
        if self.stated_electrons is not None:
            return float(self.stated_electrons)
        return float(np.sum(self.occupations))

    @property
    def empty(self) -> bool:
        return self.n_pairs == 0

    def report(self, logger=None) -> None:
        logger = logger or log
        out.entry(logger, "{} candidates".format(self.cls),
                  "{} pair(s), {:.0f} e".format(self.n_pairs, self.electrons), "",
                  self.description)
        for note in self.notes:
            out.note(logger, "  {}".format(note))

    def __repr__(self) -> str:
        return "CandidateSet({}, {} pairs{})".format(self.cls, self.n_pairs,
                                                     ", fixed" if self.fixed else "")


@dataclass(frozen=True)
class LigandConstruction:
    """A ligand class's candidates and the orbital set its columns index.

    ⚠ ``coeff`` is the set **after** the fragment rotation (:func:`fragment_rotation`), which
    is what the columns refer to and what a probe must start from. The rotation is inside
    occupation groups and leaves the claimed pairs alone, so composing it after the shell's
    changes no earlier selection and no density.
    """

    coeff: np.ndarray
    candidates: CandidateSet
    #: Population on the fragment of every Kramers pair of :attr:`coeff`, in column order.
    populations: np.ndarray

    @property
    def n_pairs(self) -> int:
        return self.candidates.n_pairs


@dataclass(frozen=True)
class ShellConstruction:
    """The one AVAS call a target set's shells, double shells and bonding partners come from.

    ⚠ ``coeff`` -- **not** the orbitals handed in -- is the set every :class:`CandidateSet`
    here indexes into, and the set a later probe must start from. AVAS rotates, and a
    selection read against the unrotated columns is a different active space wearing the
    same description.
    """

    coeff: np.ndarray
    occupation: np.ndarray
    avas: object
    shell: CandidateSet
    centres: Tuple[ctr.Centre, ...]
    double: Optional[CandidateSet] = None
    bonding: Optional[CandidateSet] = None

    @property
    def sets(self) -> Tuple[CandidateSet, ...]:
        return tuple(s for s in (self.shell, self.double, self.bonding) if s is not None)


# --- shared helpers ------------------------------------------------------------------------

def _pair_columns(pairs: Sequence[int]) -> np.ndarray:
    """Pair indices -> spinor columns, both partners, sorted. The one place this is done."""
    pairs = np.unique(np.asarray(pairs, dtype=int))
    if pairs.size == 0:
        return np.zeros(0, dtype=int)
    return np.sort(np.concatenate([[2 * int(p), 2 * int(p) + 1] for p in pairs]))


def _excluded_pairs(exclude) -> set:
    """Spinor columns already claimed -> the pair indices they occupy.

    ⚠ One convention for ``exclude=`` across every constructor: **spinor columns**, which is
    what a caller holds (a :attr:`CandidateSet.columns`). Two constructors taking two
    conventions for the same argument is the kind of mistake that quietly excludes the wrong
    half of an orbital set.
    """
    if exclude is None:
        return set()
    return set(int(c) // 2 for c in np.asarray(exclude, dtype=int).ravel())


def _pair_occupations(occupation: np.ndarray) -> np.ndarray:
    """Electrons per Kramers pair: 2 closed, 1 singly occupied, 0 empty."""
    occ = np.asarray(occupation, dtype=float).ravel()
    if occ.size % 2:
        raise ValueError("a Kramers-paired spinor set has an even number of columns; got {}"
                         .format(occ.size))
    return occ[0::2] + occ[1::2]


def _round_out(values_in_order: np.ndarray, n_keep: int, *,
               rtol: float = DEFAULT_RANKING_RTOL) -> Tuple[int, Optional[str]]:
    """Extend ``n_keep`` so the cut does not split a tie in the ranking quantity.

    The project's rule for any truncation of a symmetric quantity, in its *selection*
    instance: keeping some members of a degenerate group and not the rest selects a subspace
    that is not invariant under whatever made them degenerate, which for symmetry partners
    means an active space that breaks a degeneracy the molecule has exactly. A bounded count
    can therefore come out larger than it was asked for, and the note says so.

    ⚠ **The values are in the order the class selects in, which need not be monotone**, so
    degeneracy is tested as an *absolute* relative gap between the last kept value and the
    next one rather than as a group of a descending spectrum. That is the same definition
    (:func:`kuiva.util.degeneracy.relative_gap`, one implementation of "how close is close")
    applied where a selection order is not an eigenvalue order -- a bridge is ordered by
    proximity to the Fermi level and ranked by a population, and reading consecutive gaps of
    *that* as a spectrum said "a degenerate group of 3" about the values 0.505..0.766 on
    Ti2Cl6 and rounded every bounded count out to its whole pool. A guard that always fires
    is the same defect as one that never can.
    """
    values = np.asarray(values_in_order, dtype=float)
    n_keep = int(n_keep)
    if n_keep <= 0 or n_keep >= values.size:
        return max(0, min(n_keep, values.size)), None
    stop = n_keep
    while stop < values.size and abs(relative_gap(values[stop - 1], values[stop])) <= rtol:
        stop += 1
    if stop <= n_keep:
        return n_keep, None
    return stop, ("the cut at {} would have split a tie in the {} (values {:.4f} and "
                  "{:.4f}); rounded outward to {} so the degenerate group is taken whole"
                  .format(n_keep, "ranking", float(values[n_keep - 1]),
                          float(values[stop - 1]), stop))


def _rank_pool(pool: np.ndarray, *, prefer_high_index: bool) -> np.ndarray:
    """Order ``pool`` **nearest the Fermi level first** -- the ligand classes' rule.

    Occupied pairs from the gap downwards, empty pairs from the gap upwards. The population
    on the fragment is the *filter* (a pair below the threshold is not the fragment's pair at
    all) and this is the *order*, which is the way round it has to be:

    ⚠ **Ranking by the population instead selects core orbitals.** Measured on Ti2Cl6 --
    ranked by population, the two "bridge" pairs came back at columns 52 and 122, deep
    chlorine orbitals with 0.72 and 0.66 on the bridging atoms and no role in any exchange
    pathway, while the pairs just below the shell cut (0.51, 0.53) are the ones that mix with
    the metal d. The most localized orbital on a fragment is its most *core* one.

    ⚠ In an AVAS-rotated set this ordering is still the right one, and for a better reason
    than continuity: within an occupation group AVAS orders the pairs by projection onto the
    metal shell, so the pairs nearest the gap are the ones carrying metal character -- which
    among the fragment's own pairs is exactly what a superexchange pathway is.
    """
    pool = np.asarray(pool, dtype=int)
    if pool.size == 0:
        return pool
    return pool[np.argsort(-pool if prefer_high_index else pool)]


def _fragment_pair_populations(reference, coeff, atoms: Sequence[int]) -> np.ndarray:
    """Loewdin population of each Kramers pair on ``atoms`` -- a fraction of the pair.

    Through :func:`kuiva.mcscf.localize.fragment_populations`, which is the project's one
    implementation of "how much of this orbital is on these atoms" (and the one that knows a
    spinor's rows are spin-blocked, so an atom owns two row ranges).
    """
    from ..mcscf.localize import fragment_populations

    pops = fragment_populations(coeff, reference.data.s_ao, reference.ao_layout,
                                [list(int(a) for a in atoms)])
    per_spinor = np.asarray(pops)[:, 0]
    return 0.5 * (per_spinor[0::2] + per_spinor[1::2])


def fragment_rotation(reference, coeff, atoms: Sequence[int], *, pair_occupations,
                      frozen: Sequence[int] = (), protect: Optional[np.ndarray] = None,
                      rtol: float = DEFAULT_RANKING_RTOL):
    """Rotate the unclaimed Kramers pairs onto a fragment; return ``(coeff, populations)``.

    Within each group of equal occupation, the pairs **not** in ``frozen`` are rotated so that
    the fragment's Loewdin population operator is diagonal over them; the eigenvalues are
    each rotated pair's population on ``atoms``, and the frozen pairs are untouched.

    ⚠ ``protect`` is a per-pair quantity that must survive the rotation -- in practice the
    shell's AVAS projections -- and it narrows the rotation to blocks of pairs whose value is
    **degenerate**. That is the honest boundary: mixing two pairs whose projections differ
    destroys a number another class ranks by (a later bridge round reads exactly those
    projections), while mixing pairs that are degenerate in it destroys nothing, because
    within such a block no basis was defined in the first place. Without ``protect`` the whole
    free part of a group is one block.

    ⚠ **This is what makes a ligand selection reproducible at all.** What AVAS leaves behind
    in an occupation group is a *span*, not a basis: its pairs are degenerate at zero
    projection, so which combinations come back is whatever the diagonalization returned, and
    measured on Ti2Cl6 that differed between identical runs. The span itself is unique -- it
    is the orthogonal complement of the claimed pairs inside the group -- so diagonalizing a
    physical operator on it gives one answer, every time.

    ⚠ **The density does not move**, for the same reason AVAS's rotation does not: it stays
    inside groups of equal occupation. And the barred partners take the **conjugate** rotation
    (:func:`kuiva.spinor.expand.rotate_kramers_pairs`), which is what keeps the set Kramers
    paired -- the same trap, in the same shape, as in the projection itself.

    The population is measured through :func:`kuiva.mcscf.localize.fragment_populations`'s own
    metric (``S^{1/2}`` from :func:`kuiva.orth.canonical.sqrt_overlap`), so "how much of this
    orbital is on these atoms" has one definition in the project and this is an application of
    it rather than a second one.
    """
    from ..orth.canonical import sqrt_overlap
    from ..spinor.expand import fold_to_kramers_pairs, rotate_kramers_pairs

    coeff = np.ascontiguousarray(coeff, dtype=np.complex128)
    layout = reference.ao_layout
    root = sqrt_overlap(np.asarray(reference.data.s_ao))
    nao = root.shape[0]
    if coeff.shape[0] != 2 * nao:
        raise ValueError("the spinors span {} rows against {} AOs".format(coeff.shape[0], nao))
    # ⚠ Spin-blocked rows: an atom owns TWO row ranges. Halving every population by using one
    # of them looks like a disappointing fragment rather than a bug.
    ao = np.concatenate([np.asarray(layout.atom_indices(int(a))) for a in atoms])
    t = np.vstack([root @ coeff[:nao], root @ coeff[nao:]])
    rows = np.concatenate([ao, ao + nao])
    operator = t[rows].conj().T @ t[rows]
    m_pair, residual = fold_to_kramers_pairs(operator)
    if residual > 1.0e-8:
        raise ValueError(
            "the fragment population operator departs from Kramers-pair structure by {:.3e}: "
            "these orbitals are not the paired set it can be folded onto (the operator itself "
            "is spin-free, so this is a statement about the orbitals). The usual cause is an "
            "unrestricted reference".format(residual))

    pair_occ = np.asarray(pair_occupations, dtype=float)
    npair = pair_occ.size
    claimed = set(int(p) for p in np.asarray(frozen, dtype=int).ravel())
    populations = np.real(np.diag(m_pair)).copy()
    rotation = np.eye(npair, dtype=np.complex128)
    for value in sorted(set(np.round(pair_occ, 8).tolist()), reverse=True):
        group = np.array([p for p in np.nonzero(np.round(pair_occ, 8) == value)[0]
                          if int(p) not in claimed], dtype=int)
        if group.size < 2:
            continue                                  # nothing to rotate; one pair is unique
        for block in _degenerate_blocks(group, protect, rtol):
            if block.size < 2:
                continue
            w, v = np.linalg.eigh(m_pair[np.ix_(block, block)])
            order = np.argsort(-w)                    # most localized on the fragment first
            populations[block] = w[order]
            rotation[np.ix_(block, block)] = v[:, order]
    rotated = rotate_kramers_pairs(coeff, rotation, np.arange(2 * npair))
    return np.ascontiguousarray(rotated), populations


def _degenerate_blocks(pairs: np.ndarray, protect, rtol: float) -> List[np.ndarray]:
    """``pairs`` split into runs whose ``protect`` value is degenerate (all of it if none).

    The cut is the project's own consecutive-relative-gap rule
    (:func:`kuiva.util.degeneracy.group_bounds`) on the protected quantity sorted descending,
    which is a spectrum and therefore the case that rule is written for.
    """
    pairs = np.asarray(pairs, dtype=int)
    if pairs.size == 0:
        return []
    if protect is None:
        return [pairs]
    from ..util.degeneracy import group_bounds

    values = np.asarray(protect, dtype=float)[pairs]
    order = np.argsort(-values, kind="stable")
    ordered, spectrum = pairs[order], values[order]
    blocks, start = [], 0
    while start < ordered.size:
        _, stop = group_bounds(spectrum, start, rtol)
        blocks.append(np.sort(ordered[start:stop]))
        start = stop
    return blocks


def _population_table(logger, values, occupations, kept) -> None:
    """The pairs by population on the fragment, either side of the cut. What a refusal or a
    report has to show: "no pair carries 50%" is a number, and the useful form of it is the
    table of what the pairs actually carry."""
    table = out.Table(logger, [out.col_count("pair", 8),
                               out.Column("occupation", "{:.1f}", 12),
                               out.Column("population", "{:.4f}", 12),
                               out.Column("candidate", "{}", 11, "<")]).start()
    order = np.argsort(-np.asarray(values))
    for p in order[:max(len(kept) + 4, 8)]:
        table.row(int(p), float(occupations[p]), float(values[p]),
                  "yes" if int(p) in set(int(k) for k in kept) else "no")
    table.end("Kramers pairs by population on the fragment")


# --- 1. the shell classes ------------------------------------------------------------------

def shell_candidates(reference, centres: Sequence[ctr.Centre], *, coeff=None,
                     occupation=None, double: bool = False, bonding: int = 0,
                     bonding_floor: float = DEFAULT_BONDING_FLOOR,
                     rtol: float = DEFAULT_RANKING_RTOL,
                     report: bool = True) -> ShellConstruction:
    """The valence shells of ``centres``, and whatever else that AVAS call defines.

    Parameters
    ----------
    reference
        A finished spinor reference (duck-typed: ``data``, ``ao_layout``, ``spinors``,
        ``spinors_in_ao()``), with ``atomic_reference=True`` on its scalar SCF.
    centres
        :class:`~kuiva.autocas.centres.Centre` objects. Of one ``l`` they are one AVAS
        projection as always; of several, one projection onto the **union** of their
        reference spans with every selected pair attributed to a centre -- see the module
        docstring for why that and not two projections.
    coeff, occupation
        The orbital set to rotate, defaulting to the reference's own guess spinors.
    double
        Also offer the correlating shell: the call is made with ``n_shells=2`` and twice the
        count. The **valence shell** is the pairs of that selection with the largest
        projection onto the *valence reference shell alone* (a diagonal read in the rotated
        orbitals, so nothing is rotated twice); the
        :class:`~kuiva.autocas.targets.DoubleShell` candidate set is the next **empty** pairs by
        the two-shell projection -- empty by construction, since an occupied ligand pair with
        a little valence character can outrank a second-shell virtual. ⚠ It changes the projector,
        so the shell's own pairs are selected from the two-shell projection; the count is the
        same and the description states which projection produced it.
    bonding
        Offer up to this many metal-ligand bonding pairs per centre, with their antibonding
        partners, from the same spectrum.
    """
    from ..mcscf.avas import avas_projection, projection_pair_matrix

    centres = tuple(centres)
    if not centres:
        raise ValueError("no centres: there is no shell to construct")
    ells = sorted({int(c.l) for c in centres})
    mixed = len(ells) > 1
    ell = ells[0]
    atoms = tuple(sorted({int(a) for c in centres for a in c.atoms}))
    # ⚠ The one statement of what is projected onto, shared by every call below: the single
    # (atoms, l) form where the centres share an l -- bitwise what it always was -- and one
    # union entry per centre where they do not, so the attribution has a row per centre.
    if mixed:
        def target(n_shells):
            return dict(shells=[(list(c.atoms), int(c.l), int(n_shells)) for c in centres])
    else:
        def target(n_shells):
            return dict(atom=list(atoms), l=ell, n_shells=int(n_shells))
    if coeff is None:
        coeff = reference.spinors_in_ao()
        if occupation is None:
            occupation = np.asarray(reference.spinors.occ, dtype=float)
    if occupation is None:
        raise ValueError("give occupation= beside coeff=: the AVAS rotation happens inside "
                         "groups of equal occupation and cannot infer them from orbitals")

    n_shell_pairs = sum(c.n_pairs for c in centres)

    # ⚠ **`avas_projection`, not `avas`**: the space -- and with it the electron count -- is
    # this function's decision, because a shell that is empty in the reference has no aufbau
    # count and `active_space` can only refuse one (see `_shell_electrons`).
    result = avas_projection(coeff, reference.data.s_ao, reference.ao_layout,
                             reference.data.atomic_reference, occupation=occupation,
                             n_pairs=n_shell_pairs, **target(1))
    if report:
        result.report(log)
    values = np.asarray(result.eigenvalues, dtype=float)
    pair_occ = np.asarray(result.occupations, dtype=float)
    selected = np.asarray(result.selected, dtype=int)
    shell_pairs = np.sort(selected)
    ranking, ranking_name = values, "AVAS projection"

    double_pairs = np.zeros(0, dtype=int)
    double_values = np.zeros(0)
    if double:
        # ⚠ **The valence shell is the one-shell construction, bitwise, and the correlating
        # shell is built in what it left over.** Asking for a class may not change the core
        # the class is judged against. This was once one AVAS call with two shells projected,
        # the valence shell read back off it -- and on TiCl3 that shell was a different span
        # (principal overlaps 0.87-0.91 with the one-shell one) whose empty d pairs put the
        # ligand-field states at 28 800 cm^-1 instead of 4 300: a double-shell round then
        # compared a broken core with a repaired one and "kept" the class by 24 000 cm^-1.
        #
        # The second rotation is confined to the EMPTY pairs the shell did not select AND whose
        # one-shell projection is below the bonding floor. Those are degenerate at (near) zero
        # projection, so no basis of them was defined and rotating them destroys nothing --
        # the frontier class's rule. The empty pairs above the floor are non-degenerate
        # eigenvectors carrying shell character, which the bonding and bridge classes read,
        # and they are left as they are. Inside the pool the two-shell projector is
        # diagonalized: its eigenvectors are unique wherever the projection is not zero.
        from ..spinor.expand import rotate_kramers_pairs

        pool = np.array([p for p in np.nonzero(pair_occ <= 1.0e-8)[0]
                         if int(p) not in set(shell_pairs.tolist())
                         and values[p] < DEFAULT_BONDING_FLOOR], dtype=int)
        m_two, _, _ = projection_pair_matrix(
            result.coeff, reference.data.s_ao, reference.ao_layout,
            reference.data.atomic_reference, **target(2))
        w, v = np.linalg.eigh(m_two[np.ix_(pool, pool)])
        order = np.argsort(-w, kind="stable")
        w, v = w[order], v[:, order]
        rotated = rotate_kramers_pairs(result.coeff, v, _pair_columns(pool))
        from dataclasses import replace as _replace

        # The rotated pool's one-shell projections are its diagonal now; each lies inside the
        # pool's eigenvalue range, so none can cross the floor the other classes read.
        m_one, _, _ = projection_pair_matrix(
            rotated, reference.data.s_ao, reference.ao_layout,
            reference.data.atomic_reference, **target(1))
        values = values.copy()
        values[pool] = np.real(np.diag(m_one))[pool]
        result = _replace(result, coeff=rotated, eigenvalues=values)
        ranking = values
        take = min(n_shell_pairs, pool.size)
        double_pairs = np.sort(pool[:take])          # the pool columns now hold ordered vectors
        double_values = w[:take][np.argsort(pool[:take])]
        if take < n_shell_pairs:
            raise ValueError(
                "the basis leaves only {} empty Kramers pairs outside the {} for a correlating "
                "shell of {}".format(pool.size, _shells_phrase(centres), n_shell_pairs))
        if float(double_values.min()) < DEFAULT_BONDING_FLOOR:
            log.warning("the correlating shell of the %s reaches down to empty pairs "
                        "carrying only %.3f of the two-shell projection: the basis does not "
                        "hold a second shell for them to be, so the class is weakly defined",
                        _shells_phrase(centres), float(double_values.min()))

    where = " and ".join(c.where for c in centres)
    state = centres[0].reference_state
    shell_note = ("gap at the cut {:.3f}".format(result.gap) if np.isfinite(result.gap)
                  else "no pair was dropped, so there is no gap to report")
    if mixed:
        shell_fragments = _attribute_by_projection(reference, result.coeff, centres,
                                                   shell_pairs, n_shells=1)
        description = (
            "the {} Kramers pairs of largest projection onto the union of the {} (one "
            "count-stated AVAS on the free-atom reference, {}; each pair attributed to the "
            "shell it projects onto most, {}; {})".format(
                shell_pairs.size, _shells_phrase(centres),
                ", ".join("{}: {}".format(c.where, c.reference_state) for c in centres),
                " + ".join(str(len(g) // 2) for g in shell_fragments), shell_note))
        electrons, electron_note, per_centre = _attributed_electrons(
            centres, shell_fragments, pair_occ)
    else:
        shell_fragments = _attribute(reference, result.coeff, centres, shell_pairs)
        description = (
            "the {} Kramers pairs of largest {} projection on {} (count-stated AVAS on the "
            "free-atom reference, {}; {})".format(
                shell_pairs.size, tg.angular_momentum_letter(ell), where, state, shell_note))
        electrons, electron_note = _shell_electrons(centres, pair_occ[shell_pairs])
        total = float(electrons if electrons is not None else np.sum(pair_occ[shell_pairs]))
        n_atoms = sum(len(c.atoms) for c in centres)
        per_centre = tuple(total * len(c.atoms) / n_atoms for c in centres)
    shell = CandidateSet(
        cls="shell", columns=_pair_columns(shell_pairs), description=description,
        occupations=pair_occ[shell_pairs], ranking=ranking[shell_pairs],
        ranking_name=ranking_name, fixed=True, stated_electrons=electrons,
        fragments=shell_fragments,
        notes=(shell_note,) + ((electron_note,) if electron_note else ()),
        centre_electrons=per_centre)

    _cross_check(centres, shell)

    double_set = None
    if double:
        double_set = CandidateSet(
            cls="double", columns=_pair_columns(double_pairs),
            description=("the {} empty Kramers pairs of largest two-shell projection onto "
                         "the {} outside the valence shells -- the correlating shells "
                         "(projections {:.3f}..{:.3f}, {})"
                         .format(double_pairs.size, _shells_phrase(centres),
                                 float(double_values.max()), float(double_values.min()),
                                 ", ".join(c.reference_state for c in centres))
                         if mixed else
                         "the {} empty Kramers pairs of largest two-shell {} projection on {} "
                         "outside the valence shell -- the correlating shell (projections "
                         "{:.3f}..{:.3f}, {})"
                         .format(double_pairs.size, tg.angular_momentum_letter(ell), where,
                                 float(double_values.max()), float(double_values.min()),
                                 state)),
            occupations=pair_occ[double_pairs], ranking=double_values,
            ranking_name="AVAS projection onto two shells, outside the valence shell",
            fixed=True,
            fragments=(_attribute_by_projection(reference, result.coeff, centres,
                                                double_pairs, n_shells=2)
                       if mixed else
                       _attribute(reference, result.coeff, centres, double_pairs)),
            notes=("taken whole or not at all: at this level a correlating shell is empty "
                   "whatever it is worth, so only the spectrum can decide it",))
    bonding_set = None
    if bonding:
        bonding_set = bonding_candidates(
            result, centres, n_pairs=int(bonding), floor=float(bonding_floor), rtol=rtol,
            exclude=_pair_columns(np.concatenate([shell_pairs, double_pairs])))
    return ShellConstruction(coeff=result.coeff, occupation=np.asarray(occupation, float),
                             avas=result, shell=shell, centres=centres, double=double_set,
                             bonding=bonding_set)


def _shell_electrons(centres, occupations) -> Tuple[Optional[float], str]:
    """How many electrons the shell holds: measured, except where it cannot be.

    ⚠ **The count is the reference occupation of the selected pairs**, because the per-atom
    reference state defaults to the *neutral* atom outside the f block and a transition-metal
    ion's d count is therefore wrong by default there.

    ⚠ **The exception is a shell that is entirely empty (or entirely full) in the reference**,
    where there is no count to measure at all: the scalar SCF has put the electrons somewhere
    else. Measured live on CeCl3 -- its ROHF valence electron sits in a Ce **5d** orbital, so
    all seven 4f pairs are empty and the measurement says CAS(0, 14), a single determinant
    with no physics in it. Where the centre's reference state is a statement about *this ion*
    (the user stated it, or the element is in the f block, whose default is M(3+)) that count
    is used instead, **loudly**: the orbitals the electrons came from end up in the virtual
    space, which is a real change to the calculation and exactly what the committed CeCl3
    reference does. Where it is only the neutral-atom fallback the shell is **refused**, with
    ``configuration=`` named -- a neutral Ti says "d2" about a d0 complex just as confidently.
    """
    occupations = np.asarray(occupations, dtype=float)
    measured = float(np.sum(occupations))
    capacity = 2.0 * occupations.size
    expected = float(sum(c.open_electrons * len(c.atoms) for c in centres))
    if occupations.size == 0 or measured not in (0.0, capacity):
        return None, ""
    if abs(expected - measured) < 1e-8 or expected in (0.0, capacity):
        return None, ""
    where = " and ".join(c.where for c in centres)
    states = ", ".join("{}: {}".format(c.where, c.reference_state) for c in centres)
    if not all(c.trusted_state for c in centres):
        raise ValueError(
            "the shell of {} is {} in this reference, so its electron count cannot be "
            "measured from the orbitals -- and the reference state ({}) is the NEUTRAL-ATOM "
            "default, which says nothing about this ion: a neutral titanium claims d2 of a "
            "d0 complex just as confidently. Either this shell is genuinely {} and is an "
            "inactive block rather than an active space, or the oxidation state has to be "
            "stated: configuration={{\"{}\": \"+n\"}} on the scalar SCF"
            .format(where, "empty" if measured == 0.0 else "full", states,
                    "empty" if measured == 0.0 else "full",
                    centres[0].labels[0].split()[-1]))
    note = ("the shell is {} in this reference ({:.0f} electrons by occupation), so its "
            "electron count comes from the reference state instead ({}): {:.0f} electrons. "
            "The orbitals the scalar SCF put them in are left OUTSIDE the active space"
            .format("empty" if measured == 0.0 else "full", measured, states, expected))
    log.warning("%s", note)
    return expected, note


def _shells_phrase(centres) -> str:
    """``"d shell of Ti1 and f shell of Ce2"`` -- the shells as a statement, one per centre."""
    return " and ".join("{} shell of {}".format(tg.angular_momentum_letter(c.l), c.where)
                        for c in centres)


def _attribute_by_projection(reference, coeff, centres, pairs, *,
                             n_shells: int) -> Tuple[Tuple[int, ...], ...]:
    """Which centre each pair belongs to, by its projection onto each centre's shell alone.

    The attribution for centres of **different** ``l``, where a Loewdin population cannot do
    it: a 4f and a 5d on one lanthanide sit on the same atom. Each pair goes to the centre
    whose reference span it projects onto most (:func:`kuiva.mcscf.avas.component_projections`,
    a diagonal read that rotates nothing), and a count that does not come out as ``2l+1`` per
    atom **warns** -- for the valence shells (``n_shells=1``) that means the union's pairs do
    not separate into the shells the statement names, which is a fact about the molecule and
    not an error; the space is still the union that was asked for.
    """
    from ..mcscf.avas import component_projections

    pairs = np.asarray(pairs, dtype=int)
    parts, _ = component_projections(
        coeff, reference.data.s_ao, reference.ao_layout, reference.data.atomic_reference,
        shells=[(list(c.atoms), int(c.l), int(n_shells)) for c in centres])
    owner = np.argmax(parts[:, pairs], axis=0) if pairs.size else np.zeros(0, dtype=int)
    groups = []
    for i, centre in enumerate(centres):
        mine = pairs[owner == i]
        if mine.size != centre.n_pairs:
            log.warning("%d of the %d pairs selected by the union projection project most onto "
                        "the %s shell of %s, which is %d pairs: the union's orbitals do not "
                        "separate into the shells the statement names. The space is still the "
                        "union that was asked for; what is not reliable is the per-centre "
                        "attribution, and with it the per-centre electron count and floor",
                        mine.size, pairs.size, tg.angular_momentum_letter(centre.l),
                        centre.where, centre.n_pairs)
        groups.append(tuple(int(c) for c in _pair_columns(mine)))
    return tuple(groups)


def _attributed_electrons(centres, groups, pair_occ) -> Tuple[Optional[float], str,
                                                               Tuple[float, ...]]:
    """``(stated total or None, note, per-centre electrons)`` for a union shell.

    :func:`_shell_electrons` applied **per centre**, to the pairs attributed to it -- the
    empty-shell decision is a statement about one ion and has to be taken where that ion's
    pairs are. ⚠ The case it exists for is the ordinary one for a heterometallic lanthanide
    complex: the scalar ROHF puts the Ce(3+) electron in a 5d orbital, so the Ce 4f pairs are
    empty while the Ti 3d pair beside them holds its electron, and the union as a whole is
    neither empty nor full -- a whole-shell test would measure CAS(1, 24) and lose the ion.
    """
    pair_occ = np.asarray(pair_occ, dtype=float)
    per_centre: List[float] = []
    notes: List[str] = []
    stated = False
    for centre, group in zip(centres, groups):
        pairs = np.asarray(group, dtype=int)[0::2] // 2
        electrons, note = _shell_electrons([centre], pair_occ[pairs])
        if electrons is None:
            per_centre.append(float(np.sum(pair_occ[pairs])))
        else:
            stated = True
            per_centre.append(float(electrons))
            notes.append(note)
    total = float(sum(per_centre))
    return (total if stated else None), "; ".join(notes), tuple(per_centre)


def _attribute(reference, coeff, centres, pairs) -> Tuple[Tuple[int, ...], ...]:
    """Which centre each selected pair belongs to -- the ``fragments`` of the active space.

    One centre (pooled or not) owns all of them. With several centres of the same ``l`` each
    pair goes to the centre carrying most of its population, and a count that does not come
    out as ``2l+1`` per centre **warns**: it means the orbitals do not separate the way the
    statement says they do, which is a fact about the molecule and not an error.
    """
    pairs = np.asarray(pairs, dtype=int)
    if len(centres) == 1:
        return (tuple(int(c) for c in _pair_columns(pairs)),)
    populations = np.stack([_fragment_pair_populations(reference, coeff, c.atoms)[pairs]
                            for c in centres])
    owner = np.argmax(populations, axis=0)
    groups = []
    for i, centre in enumerate(centres):
        mine = pairs[owner == i]
        if mine.size != centre.n_pairs:
            log.warning("%d of the selected pairs sit mostly on %s, but its %s shell is %d "
                        "pairs: the shells of the separate centres are not separated by "
                        "these orbitals. The space is still the union that was asked for; "
                        "what is not reliable is the per-centre attribution",
                        mine.size, centre.where, tg.angular_momentum_letter(centre.l),
                        centre.n_pairs)
        groups.append(tuple(int(c) for c in _pair_columns(mine)))
    return tuple(groups)


def _cross_check(centres, shell: CandidateSet) -> None:
    """The measured shell electron count against each centre's stated reference state.

    ⚠ The count that *defines* the space is this measured one; the configuration statement is
    only allowed to disagree with it (:func:`kuiva.autocas.centres.check_configuration`).
    """
    electrons_of_pair = {int(c) // 2: float(o)
                         for c, o in zip(np.asarray(shell.columns, dtype=int)[0::2],
                                         np.asarray(shell.occupations, dtype=float))}
    groups = shell.fragments or ((tuple(int(c) for c in shell.columns),),)
    for centre, group in zip(centres, groups):
        electrons = sum(electrons_of_pair.get(int(c) // 2, 0.0)
                        for c in np.asarray(group, dtype=int)[0::2])
        ctr.check_configuration(centre, electrons / max(len(centre.atoms), 1))


def bonding_candidates(avas_result, centres: Sequence[ctr.Centre], *, n_pairs: int,
                       floor: float = DEFAULT_BONDING_FLOOR,
                       rtol: float = DEFAULT_RANKING_RTOL,
                       exclude: Optional[Sequence[int]] = None) -> CandidateSet:
    """The metal-ligand bonding pairs just below a shell's projection cut, with partners.

    Read off the shell's own AVAS spectrum: the *occupied* pairs of largest remaining
    projection are the bonding combinations that carry the shell's character, and for each of
    them the *empty* pair of largest remaining projection is its antibonding partner. Where
    the next occupied projection is below ``floor`` the class is **empty and says why** --
    the bond is ionic at this reference and there is no combination to correlate.

    ``exclude`` is the **spinor columns** already claimed (the shell's, and the double
    shell's where there is one); it defaults to the projection's own selection.
    """
    values = np.asarray(avas_result.eigenvalues, dtype=float)
    pair_occ = np.asarray(avas_result.occupations, dtype=float)
    taken = (_excluded_pairs(exclude) if exclude is not None
             else set(int(p) for p in np.asarray(avas_result.selected, dtype=int)))
    want = int(n_pairs) * max(len(centres), 1)
    where = " and ".join(c.where for c in centres)

    def _rank(mask) -> np.ndarray:
        pool = np.array([p for p in np.nonzero(mask)[0] if int(p) not in taken], dtype=int)
        return pool[np.argsort(-values[pool], kind="stable")] if pool.size else pool

    occupied = _rank(pair_occ > 0.5)
    empty = _rank(pair_occ <= 0.5)
    notes: List[str] = []
    if occupied.size == 0 or values[occupied[0]] < float(floor):
        best = float(values[occupied[0]]) if occupied.size else 0.0
        return CandidateSet(
            cls="bonding", columns=np.zeros(0, dtype=int),
            description=("no metal-ligand bonding combination of {} carries {:.0%} of the "
                         "shell's character (largest remaining occupied projection {:.4f}): "
                         "the bond is ionic at this reference".format(where, floor, best)),
            occupations=np.zeros(0), ranking=np.zeros(0), ranking_name="AVAS projection",
            notes=("empty class: the largest remaining occupied projection is {:.4f}, below "
                   "the {:.2f} floor".format(best, floor),))
    n_occ = int(min(want, np.count_nonzero(values[occupied] >= float(floor))))
    n_occ, note = _round_out(values[occupied], n_occ, rtol=rtol)
    if note:
        notes.append("bonding, occupied: " + note)
    # ⚠ The antibonding partner is usually **already in the shell**, and the floor is what
    # says so rather than a rule. The shell takes every pair carrying the character,
    # antibonding ones included, so what is left in the empty group after it is a ligand
    # orbital with no shell character at all -- measured on TiCl3: 0.0124 and 0.0000 against
    # the occupied partners' 0.265. Adding those would be two pairs of pure ligand virtual
    # in the name of correlating a bond whose partner is active already.
    n_vir = int(min(n_occ, np.count_nonzero(values[empty] >= float(floor))))
    if n_vir:
        n_vir, note = _round_out(values[empty], n_vir, rtol=rtol)
        if note:
            notes.append("bonding, antibonding partners: " + note)
    elif empty.size:
        notes.append("no empty pair outside the shell carries {:.0%} of the shell character "
                     "(largest {:.4f}), so the antibonding partners are the shell's own "
                     "pairs and are active already".format(floor, float(values[empty[0]])))
    pairs = np.sort(np.concatenate([occupied[:n_occ], empty[:n_vir]]))
    return CandidateSet(
        cls="bonding", columns=_pair_columns(pairs),
        description=("the {} occupied Kramers pairs of {} below the shell's projection cut"
                     "{} (projections {:.4f}..{:.4f})"
                     .format(n_occ, where,
                             " with {} antibonding partner(s)".format(n_vir) if n_vir else
                             " (their antibonding partners are shell pairs already)",
                             float(values[pairs].max()), float(values[pairs].min()))),
        occupations=pair_occ[pairs], ranking=values[pairs],
        ranking_name="AVAS projection",
        fragments=(tuple(int(c) for c in _pair_columns(pairs)),), notes=tuple(notes))


def check_disjoint(sets: Sequence[CandidateSet]) -> None:
    """Refuse two classes that claim the same Kramers pair, naming both statements.

    The same rule the union-of-fragments character selection applies, and for the same
    reason: a pair that satisfies two classes at once means they are not the disjoint
    statement they were written as. ⚠ The **shell** is the stated exception and it cannot
    reach here, by construction rather than by a special case -- the ligand classes are given
    the shell's columns as ``exclude=`` and are defined as "not the shell", so a bonding
    combination that *is* a shell member is simply already in.
    """
    live = [s for s in sets if s is not None and not s.empty]
    for i, first in enumerate(live):
        for second in live[i + 1:]:
            shared = np.intersect1d(np.asarray(first.columns, dtype=int),
                                    np.asarray(second.columns, dtype=int))
            if shared.size:
                raise ValueError(
                    "the {} and {} candidates both claim spinor(s) {}: '{}' and '{}' are not "
                    "the disjoint statement they were written as. Lower one of the counts, "
                    "name the atoms of the narrower class, or drop the class that is already "
                    "contained in the other"
                    .format(first.cls, second.cls, shared.tolist(), first.description,
                            second.description))


# --- 2. the ligand classes -----------------------------------------------------------------

def frontier_candidates(reference, target: tg.Frontier, *, coeff=None, occupation=None,
                        exclude: Optional[Sequence[int]] = None,
                        protect: Optional[np.ndarray] = None,
                        threshold: float = DEFAULT_FRAGMENT_THRESHOLD,
                        rtol: float = DEFAULT_RANKING_RTOL,
                        report: bool = True) -> LigandConstruction:
    """A fragment's singly occupied pairs, plus ``n_occ``/``n_vir`` neighbours of it.

    The free pairs are **rotated onto the fragment** first (:func:`fragment_rotation`), so
    the candidates are the combinations of largest Loewdin population there rather than
    whichever basis of the complement an earlier rotation happened to return -- see the module
    docstring for why that is a correctness rule and not tidiness. ``exclude`` is the spinor
    columns already claimed; they are frozen by the rotation and stay exactly where they are.

    A fragment carrying **no** singly occupied pair when no neighbours were asked for is
    refused with the populations printed: that is a bridge which is not a radical, and the
    target was a claim about the molecule that the molecule does not support.

    ``protect`` (in practice the shell's AVAS projections) confines the rotation to blocks of
    pairs whose projection is degenerate, so a later class can still rank by it -- see
    :func:`fragment_rotation`.
    """
    layout = reference.ao_layout
    atoms = tg.resolve_atoms(layout, target.atoms)
    labels = tg.atom_labels(layout, atoms)
    if coeff is None:
        coeff = reference.spinors_in_ao()
        if occupation is None:
            occupation = np.asarray(reference.spinors.occ, dtype=float)
    if occupation is None:
        raise ValueError("give occupation= beside coeff=: the rotation happens inside groups "
                         "of equal occupation and cannot infer them from orbitals")
    pair_occ = _pair_occupations(occupation)
    taken = _excluded_pairs(exclude)
    coeff, population = fragment_rotation(reference, coeff, atoms,
                                          pair_occupations=pair_occ,
                                          frozen=sorted(taken), protect=protect,
                                          rtol=rtol)

    open_pairs = np.array([p for p in np.nonzero(np.abs(pair_occ - 1.0) < 0.25)[0]
                           if population[p] >= threshold and int(p) not in taken], dtype=int)
    if open_pairs.size == 0 and not (target.n_occ or target.n_vir):
        if report:
            _population_table(log, population, pair_occ, ())
        singly = np.abs(pair_occ - 1.0) < 0.25
        raise ValueError(
            "no singly occupied Kramers pair carries {:.0%} of its population on {}: the "
            "largest population of a singly occupied pair there is {:.3f}. This fragment is "
            "not a radical in this reference, so a frontier target is a statement the "
            "molecule does not support -- ask for occupied/empty pairs explicitly "
            "((\"frontier\", atoms, n_occ, n_vir)) if the closed-shell orbitals are what is "
            "meant".format(threshold, "+".join(labels),
                           float(np.max(population[singly])) if np.any(singly) else 0.0))

    notes: List[str] = []
    chosen = [open_pairs]
    for count, mask, name in ((int(target.n_occ), pair_occ > 1.5, "occupied"),
                              (int(target.n_vir), pair_occ <= 0.5, "empty")):
        if not count:
            continue
        pool = np.array([p for p in np.nonzero(mask)[0]
                         if population[p] >= threshold and int(p) not in taken], dtype=int)
        if pool.size < count:
            raise ValueError(
                "{} {} Kramers pair(s) carry {:.0%} of their population on {} after the "
                "rotation, but {} were asked for. Lower the fragment threshold deliberately, "
                "or ask for fewer".format(pool.size, name, threshold, "+".join(labels),
                                          count))
        # Most localized on the fragment first: after the rotation that is the one scalar the
        # class is defined by, and it is unique -- which the column order is not.
        order = pool[np.argsort(-population[pool], kind="stable")]
        take, note = _round_out(population[order], count, rtol=rtol)
        if note:
            notes.append("frontier, {}: {}".format(name, note))
        chosen.append(order[:take])
    pairs = np.unique(np.concatenate(chosen)) if chosen else np.zeros(0, dtype=int)
    extra = ("" if not (target.n_occ or target.n_vir) else
             " plus the {} occupied and {} empty pair(s) of largest population there"
             .format(target.n_occ, target.n_vir))
    description = ("the {} singly occupied Kramers pair(s) with at least {:.0%} of their "
                   "Loewdin population on {}{}, after rotating the unclaimed pairs onto that "
                   "fragment".format(open_pairs.size, threshold, "+".join(labels), extra))
    if report:
        _population_table(log, population, pair_occ, pairs)
    candidates = CandidateSet(cls="frontier", columns=_pair_columns(pairs),
                              description=description, occupations=pair_occ[pairs],
                              ranking=population[pairs],
                              ranking_name="fragment population",
                              fragments=(tuple(int(c) for c in _pair_columns(pairs)),),
                              notes=tuple(notes))
    return LigandConstruction(coeff=coeff, candidates=candidates, populations=population)


def bridge_atoms_by_contact(layout, site_a: Sequence[int], site_b: Sequence[int], *,
                            scale: float = DEFAULT_CONTACT_SCALE,
                            barriers: Sequence[int] = ()) -> Tuple[int, ...]:
    """The atoms of every **bridging ligand** between two sites, detected.

    Two atoms are in contact when their distance is within ``scale`` times the sum of their
    covalent radii (Cordero et al. 2008). The non-site atoms fall apart into ligands -- the
    connected components of that contact graph -- and a ligand bridges when some atom of it
    touches ``site_a`` and some atom of it (the same or another) touches ``site_b``. The
    bridging atoms are all the atoms of all bridging ligands.

    ⚠ **A ligand, not an atom that touches both sites.** The atom rule sees a mu-oxo or a
    mu-chloride and nothing else: a syn-syn carboxylate bridges through O-C-O, each oxygen
    touching one metal, and on ``mn3_linear`` (each Mn pair bridged by one hydroxide and two
    formates) it offered only the hydroxide oxygen -- where no orbital that mixes with the
    shell holds half its population, so the class was refused although the pathway is
    there. ``barriers`` are atoms a ligand may not be connected *through* (the sites always,
    and every other magnetic centre: through a metal the whole molecule is one component).

    A detection that finds nothing is a refusal rather than an empty set: the target claimed a
    pathway between the two sites, and if no ligand touches both, the atoms have to be named.
    """
    coords = np.asarray(layout.coords_bohr, dtype=float)
    site_a = tuple(int(a) for a in site_a)
    site_b = tuple(int(b) for b in site_b)
    sites = set(site_a) | set(site_b)
    blocked = sites | set(int(a) for a in barriers)

    def radius(atom: int) -> float:
        from ..basis.ghosts import is_ghost, normalize_symbol

        symbol = normalize_symbol(str(layout.atom_symbols[int(atom)]))
        if is_ghost(symbol):
            return 0.0                            # a ghost bridges nothing: no nucleus
        try:
            return COVALENT_RADII_ANGSTROM[symbol] * _BOHR_PER_ANGSTROM
        except KeyError:
            raise ValueError(
                "no covalent radius is tabulated for {} (Cordero et al. 2008 stop at Cm), so "
                "the bridging atoms cannot be detected by contact here -- name them: "
                "(\"bridge\", (site_a, site_b), atoms)".format(symbol))

    def in_contact(i: int, j: int) -> bool:
        return bool(np.linalg.norm(coords[i] - coords[j]) <= scale * (radius(i) + radius(j)))

    def touches(atom: int, site: Sequence[int]) -> bool:
        return any(in_contact(atom, s) for s in site)

    ligand_atoms = [ia for ia in range(layout.natm) if ia not in blocked and radius(ia) > 0.0]
    unvisited, found_atoms = set(ligand_atoms), []
    for start in ligand_atoms:
        if start not in unvisited:
            continue
        unvisited.discard(start)
        component, frontier = [start], [start]
        while frontier:
            atom = frontier.pop()
            for other in [o for o in unvisited if in_contact(atom, o)]:
                unvisited.discard(other)
                component.append(other)
                frontier.append(other)
        if any(touches(a, site_a) for a in component) and \
                any(touches(a, site_b) for a in component):
            found_atoms.extend(component)
    found = tuple(sorted(found_atoms))
    if not found:
        raise ValueError(
            "no ligand is within {:.2f} x (r_cov + r_cov) of both {} and {}, so this molecule "
            "has no bridging ligand to detect: the two sites are either directly bonded or too "
            "far apart for a contact criterion. Name the bridging atoms: "
            "(\"bridge\", (site_a, site_b), atoms)"
            .format(scale, "+".join(layout.atom_label(a) for a in site_a),
                    "+".join(layout.atom_label(b) for b in site_b)))
    return found


def bridge_candidates(reference, target: tg.Bridge, projection, *,
                      exclude: Optional[Sequence[int]] = None,
                      barriers: Sequence[int] = (),
                      threshold: float = DEFAULT_FRAGMENT_THRESHOLD,
                      floor: float = DEFAULT_BONDING_FLOOR,
                      scale: float = DEFAULT_CONTACT_SCALE,
                      rtol: float = DEFAULT_RANKING_RTOL,
                      report: bool = True) -> CandidateSet:
    """Ligand pairs on the bridge that mix with the metal shell: the superexchange pathway.

    ``projection`` is the shell's own AVAS result (:attr:`ShellConstruction.avas`), and the
    candidates are read off **its** spectrum: the pairs whose projection onto the shell
    exceeds ``floor`` -- so they are non-degenerate and therefore uniquely defined -- and
    whose Loewdin population on the bridging atoms exceeds ``threshold``. The same spectrum
    the bonding class reads, one filter apart: which atoms the ligand character sits on.

    ⚠ **The two things this rule is instead of.** Ranking by the *population* selects the most
    localized orbital on the fragment, which is its most **core** one -- measured on Ti2Cl6:
    deep chlorine pairs at spinors 52 and 122 with 0.72 and 0.66 on the bridge and no part in
    any exchange pathway. And selecting anything from *below* the floor selects out of the
    projection's null space, whose basis is arbitrary: three identical runs picked three
    different pairs there. A ligand pair orthogonal to the metal shell carries no
    superexchange, so the floor costs nothing physical.

    ⚠ The candidates are **offered**, not ranked: which of them mediates the exchange is a
    statement about how it entangles the two sites' shells, and only the probe can measure
    that.
    """
    layout = reference.ao_layout
    site_a = tg.resolve_atoms(layout, target.sites[0])
    site_b = tg.resolve_atoms(layout, target.sites[1])
    shared = set(site_a) & set(site_b)
    if shared:
        raise ValueError(
            "the two sites of a bridge share atom(s) {}: a bridge is between two centres, so "
            "which atoms belong to which side is the statement"
            .format(", ".join(layout.atom_label(a) for a in sorted(shared))))
    if target.atoms is None:
        atoms = bridge_atoms_by_contact(layout, site_a, site_b, scale=scale,
                                        barriers=barriers)
        how = ("the ligands in covalent contact with both sites, detected (contact within "
               "{:.2f} x the sum of the covalent radii)".format(scale))
    else:
        atoms = tg.resolve_atoms(layout, target.atoms)
        overlap = set(atoms) & (set(site_a) | set(site_b))
        if overlap:
            raise ValueError(
                "the bridging atom(s) {} are also site atoms: the bridge is what lies "
                "*between* the sites, and a shell of a site is a shell target"
                .format(", ".join(layout.atom_label(a) for a in sorted(overlap))))
        how = "named"
    labels = tg.atom_labels(layout, atoms)

    values = np.asarray(projection.eigenvalues, dtype=float)
    pair_occ = np.asarray(projection.occupations, dtype=float)
    population = _fragment_pair_populations(reference, projection.coeff, atoms)
    taken = (_excluded_pairs(exclude) if exclude is not None
             else set(int(p) for p in np.asarray(projection.selected, dtype=int)))

    pool = np.array([p for p in range(pair_occ.size)
                     if population[p] >= threshold and values[p] >= floor
                     and int(p) not in taken], dtype=int)
    if pool.size == 0:
        if report:
            _population_table(log, population, pair_occ, ())
        on_bridge = np.array([p for p in range(pair_occ.size)
                              if population[p] >= threshold and int(p) not in taken],
                             dtype=int)
        raise ValueError(
            "no Kramers pair is both on {} (at least {:.0%} of its Loewdin population) and "
            "mixed with the shell (a projection of at least {:.2f}): {} pair(s) clear the "
            "population cut and the largest projection among them is {:.4f}. Either the "
            "bridging atoms are not these ({}), or they are orthogonal to the metal shell at "
            "this reference -- which is a statement that there is no superexchange pathway to "
            "offer, not a threshold to lower. ⚠ Where the molecule has EQUIVALENT pathways "
            "the orbitals are their symmetric and antisymmetric combinations, exactly as for "
            "equivalent centres, and no pair holds half its population on one of them: state "
            "one bridge between the sublattices instead, e.g. ('bridge', ((1, 3), (2,))) for "
            "the two pathways of a linear trimer"
            .format("+".join(labels), threshold, floor, on_bridge.size,
                    float(np.max(values[on_bridge])) if on_bridge.size else 0.0, how))

    want = int(target.n_pairs)
    occupied = pool[pair_occ[pool] > 1.0 - 1e-8]
    empty = pool[pair_occ[pool] <= 1.0 - 1e-8]
    notes: List[str] = []
    chosen: List[np.ndarray] = []
    # Half the budget either side of the gap, occupied first: a superexchange pathway needs
    # both the ligand donor orbital and the acceptor it delocalizes into, and a budget spent
    # entirely on one side of the gap cannot describe one.
    for side, count, name in ((occupied, want - want // 2, "occupied"),
                              (empty, want // 2, "empty")):
        count = int(min(count, side.size))
        if count == 0:
            continue
        order = side[np.argsort(-values[side], kind="stable")]
        take, note = _round_out(values[order], count, rtol=rtol)
        if note:
            notes.append("bridge, {}: {}".format(name, note))
        chosen.append(order[:take])
    pairs = np.unique(np.concatenate(chosen)) if chosen else np.zeros(0, dtype=int)
    description = ("at most {} Kramers pairs on {} ({}) that mix with the shell, occupied and "
                   "empty, of largest projection onto it above {:.2f} among those with at "
                   "least {:.0%} of their population there (projections {:.4f}..{:.4f}, "
                   "populations {:.3f}..{:.3f})"
                   .format(want, "+".join(labels), how, floor, threshold,
                           float(values[pairs].max()), float(values[pairs].min()),
                           float(population[pairs].max()), float(population[pairs].min())))
    if report:
        _population_table(log, population, pair_occ, pairs)
    return CandidateSet(cls="bridge", columns=_pair_columns(pairs), description=description,
                        occupations=pair_occ[pairs], ranking=values[pairs],
                        ranking_name="AVAS projection onto the shell",
                        fragments=(tuple(int(c) for c in _pair_columns(pairs)),),
                        notes=tuple(notes))
