"""AVAS: an active space from the projection onto atomic valence orbitals.

Why the character selection is not enough
-----------------------------------------
:func:`kuiva.mcscf.casci.active_space_by_character` takes the *lowest orbitals carrying a
given (atom, l) character*. That is exact, reproducible, and it presupposes that some
orbitals in the converged set **are** the target orbitals. Where the metal-ligand bond is
covalent no single canonical orbital is "the 3d": the d character is spread over several
bonding and antibonding combinations, none of which clears a character threshold, and the
useful active space is a *rotation* of them rather than a subset.

AVAS (Sayfutyarova, Sun, Chan & Knizia) constructs that rotation. Project each orbital onto
the span of a set of reference atomic valence orbitals, diagonalize the projector separately
within the occupied and within the virtual space, and take the eigenvectors of large
eigenvalue: they are the combinations carrying the target character, and the eigenvalue says
how much of it each carries. Because the rotation stays **inside** the occupied space and
inside the virtual space, the reference density -- and therefore the SCF energy -- is
unchanged; only the orbitals a later CASSCF starts from move.

⚠ The reference set here is not MINAO
-------------------------------------
The published method projects onto a fixed minimal basis (MINAO). This implementation
projects onto the **free-atom orbitals Kuiva already computes** for the atomic-reference
charges (``atomic_reference=True`` on the front end; :mod:`kuiva.props.population`), taken at
the same per-element reference state the atomic mean field uses -- neutral, or M(3+) on the f
block. Three reasons, and the deviation is stated because it is a deviation:

* they live in the calculation's **own** basis, so the projection needs no second basis set
  and no cross-basis overlap;
* one element then has one reference state across the whole program, rather than a second
  notion of "the atomic orbitals of Ti" that can disagree with the first;
* for an ion they are a better reference than a neutral minimal basis.

The consequence to be aware of: eigenvalues are **not** numerically comparable with a MINAO
AVAS from another program, though the orbitals they select agree in every case tested. The
threshold's meaning is unchanged (fraction of the target character in a rotated orbital).

⚠ Whole Kramers pairs, always
-----------------------------
The projector is spin-free, so it says nothing that distinguishes a pair's two members. It is
folded onto the pair space (:func:`kuiva.spinor.expand.fold_to_kramers_pairs`), diagonalized
there, and the rotation is lifted back with the barred partners taking the **conjugate**
rotation -- so the output set is Kramers paired by construction and the active space is whole
pairs, as the spinor conventions require. The fold's residual is measured and reported; a set
that is not pair-structured cannot be folded and is refused rather than averaged.

⚠ The rotation is within groups of equal occupation, not within "occupied"
--------------------------------------------------------------------------
Mixing a doubly occupied orbital with a singly occupied one changes the density, which is the
one thing this transformation must not do. Pairs are therefore grouped by their occupation
and rotated within each group -- two groups for a closed shell, three for an ROHF open shell.
Every group is offered to the threshold, so a singly occupied orbital of the right character
is selected on its merits rather than by a rule about open shells.

References
----------
* E. R. Sayfutyarova, Q. Sun, G. K.-L. Chan, G. Knizia, "Automated Construction of Molecular
  Active Spaces from Atomic Valence Orbitals", J. Chem. Theory Comput. 13, 4063 (2017),
  doi:10.1021/acs.jctc.7b00128. The projection, the occupied/virtual separation and the
  eigenvalue threshold follow this work; the reference set does not (above).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..spinor.expand import fold_to_kramers_pairs, rotate_kramers_pairs, spin_block_diagonal
from ..util import output as out
from ..util import resources as res
from ..util.logging import get_logger

log = get_logger(__name__)

#: Default eigenvalue cut. An orbital enters the active space when this fraction of it lies
#: in the reference span. 0.2 is the published default and behaves the same here; ⚠ it is a
#: **selection** knob and not a tolerance — the right value is the one that puts the gap in
#: the eigenvalue spectrum between the kept and the dropped orbitals, which
#: :meth:`AVASResult.report` prints for exactly that reason.
DEFAULT_AVAS_THRESHOLD = 0.2

#: Largest departure from Kramers-pair structure (see
#: :func:`~kuiva.spinor.expand.fold_to_kramers_pairs`) tolerated before the projection is
#: refused. A spin-free projector on a properly paired set gives ~1e-15 here.
PAIR_FOLD_TOL = 1.0e-8

#: An atomic reference orbital counts as occupied above this. Matches
#: :data:`kuiva.props.population.REFERENCE_OCC_THRESHOLD`: average of configuration fills
#: whole shells equally, so any cut below the smallest fractional filling separates shells
#: rather than splitting one.
REFERENCE_OCC_THRESHOLD = 1e-8


@dataclass(frozen=True)
class AVASResult:
    """The rotated orbitals and the active space AVAS selected.

    Attributes
    ----------
    coeff : ndarray ``(2*nao, nspinor)``
        The rotated spinors, Kramers paired. ⚠ These are what a CASSCF must start from — the
        selection indexes *these* columns, not the ones handed in.
    space : :class:`kuiva.mcscf.casci.ActiveSpace`
        The active space, in the rotated set's numbering.
    eigenvalues : ndarray ``(npair,)``
        Projection eigenvalue of every Kramers pair of the rotated set, in column order.
    selected : ndarray
        Pair indices taken into the active space.
    occupations : ndarray ``(npair,)``
        Electrons in each Kramers pair of the rotated set — 2 for a closed pair, 1 for an
        ROHF singly occupied one. Unchanged by the rotation, by construction.
    gap : float
        Distance between the smallest kept and the largest dropped eigenvalue — how clean the
        cut was. ⚠ A small gap means the threshold, not the physics, chose the active space.
    fold_residual : float
        How far the projector was from Kramers-pair structured; ~1e-15 for a paired set.
    reference : str
        What was projected onto, as a sentence — this is the active space's *description*
        and it is what reaches the property dump's header.
    """

    coeff: np.ndarray
    space: object
    eigenvalues: np.ndarray
    selected: np.ndarray
    occupations: np.ndarray
    gap: float = float("nan")
    fold_residual: float = 0.0
    reference: str = ""

    @property
    def n_pairs(self) -> int:
        return int(np.size(self.selected))

    def report(self, logger=None, *, context: int = 3) -> None:
        """The INFO summary: the selected pairs and the eigenvalues either side of the cut."""
        logger = logger or log
        out.subsection(logger, "AVAS active-space construction")
        out.entries(logger, [
            ("reference orbitals", self.reference),
            ("Kramers pairs selected", self.n_pairs),
            ("eigenvalue gap at the cut", self.gap, "", "small = the threshold chose, not "
             "the physics", "{:.3f}"),
            ("Kramers-pair fold residual", self.fold_residual, "", "", "{:.2e}"),
        ])
        chosen = set(int(x) for x in np.asarray(self.selected).ravel())
        order = np.argsort(-np.asarray(self.eigenvalues))
        shown = [int(i) for i in order[:len(chosen) + context]]
        table = out.Table(logger, [out.col_count("pair", 7),
                                   out.Column("occupation", "{:.3f}", 12),
                                   out.Column("projection", "{:.6f}", 12),
                                   out.Column("active", "{:s}", 8)]).start()
        for i in shown:
            table.row(i, float(self.occupations[i]), float(self.eigenvalues[i]),
                      "yes" if i in chosen else "no")
        table.end("{} pair(s) shown of {}".format(len(shown), np.size(self.eigenvalues)))
        self.space.report(logger)

    def __repr__(self) -> str:
        return "AVASResult(pairs={}, gap={:.3f})".format(self.n_pairs, self.gap)


def _reference_columns(layout, reference, atoms: Sequence[int], ell: int,
                       n_shells: int) -> Tuple[np.ndarray, str]:
    """The atomic valence orbitals of ``(atoms, ell)``, placed in the molecular AO basis.

    Each atom's reference orbitals are classified by their Mulliken population per angular
    momentum — exact, because the atomic SCF is constrained spherical (so an atomic orbital
    has support on one ``l``) — and the lowest ``n_shells * (2*ell+1)`` of the target ``l``
    are taken, occupied shells first. That is "the 3d of titanium" without ever naming a
    principal quantum number: the ordering is the atomic solution's own.
    """
    nao = layout.nao
    ao_l = np.asarray(layout.ao_l)
    columns: List[np.ndarray] = []
    labels: List[str] = []
    for ia in atoms:
        idx = np.asarray(layout.atom_indices(ia))
        sym = str(layout.atom_symbols[ia]).capitalize()
        try:
            entry = (reference.entry_for_atom(ia, sym)
                     if hasattr(reference, "entry_for_atom") else reference[sym])
        except KeyError:
            raise ValueError("the ingested atomic reference has no entry for atom {} ({}); "
                             "it was built for a different molecule"
                             .format(ia + 1, sym))
        if entry.c.shape[0] != idx.size:
            raise ValueError(
                "the atomic reference for {} spans {} functions but this molecule gives the "
                "atom {}: the reference was built in a different basis"
                .format(sym, entry.c.shape[0], idx.size))
        local_l = ao_l[idx]
        if not np.any(local_l == ell):
            raise ValueError("{} has no l = {} functions in this basis"
                             .format(layout.atom_label(ia), ell))
        # Mulliken weight of each atomic orbital on the target l, in the atom's own metric.
        # The atomic block of the molecular overlap is the free atom's overlap: same basis,
        # same ordering (that is what makes the block placement below exact).
        pure = np.zeros(entry.c.shape[1])
        s_atom = _atom_overlap(entry)
        for j in range(entry.c.shape[1]):
            w = np.real(entry.c[:, j] * (s_atom @ entry.c[:, j]))
            pure[j] = float(w[local_l == ell].sum() / max(w.sum(), 1e-30))
        want = n_shells * (2 * ell + 1)
        # Occupied shells first, then the atomic virtuals, each group in its own order --
        # the same two-tier ordering the atomic-reference charges use, and for the same
        # reason: average of configuration fills whole shells, so this cuts between shells.
        occupied = np.asarray(entry.occ) > REFERENCE_OCC_THRESHOLD
        candidates = [j for j in range(entry.c.shape[1]) if pure[j] > 0.9 and occupied[j]]
        candidates += [j for j in range(entry.c.shape[1]) if pure[j] > 0.9 and not occupied[j]]
        if len(candidates) < want:
            raise ValueError(
                "{} offers only {} atomic orbital(s) of l = {} but {} shell(s) were asked "
                "for ({} orbitals). The reference basis has to carry them"
                .format(layout.atom_label(ia), len(candidates), ell, n_shells, want))
        take = candidates[:want]
        block = np.zeros((nao, len(take)))
        block[idx, :] = np.real(entry.c[:, take])
        columns.append(block)
        labels.append("{} ({}, {} shell(s) of l={})"
                      .format(layout.atom_label(ia), entry.configuration, n_shells, ell))
    return np.concatenate(columns, axis=1), "; ".join(labels)


def _atom_overlap(entry) -> np.ndarray:
    """``S`` over one atom's AO block, recovered from its orthonormal reference orbitals.

    The atomic MOs satisfy ``C^T S C = 1`` and ``C`` is square and invertible, so
    ``S = (C C^T)^-1``. Reconstructing it beats carrying a second copy that could disagree
    with the coefficients it belongs to.
    """
    c = np.real(np.asarray(entry.c))
    return np.linalg.inv(c @ c.T)


def avas(coeff_ao: np.ndarray, s_ao: np.ndarray, layout, reference, n_elec_total: int, *,
         atom, l, occupation: np.ndarray, n_shells: int = 1,
         threshold: float = DEFAULT_AVAS_THRESHOLD,
         n_active_elec: Optional[int] = None,
         max_pairs: Optional[int] = None):
    """Rotate ``coeff_ao`` onto atomic valence orbitals and select an active space.

    Parameters
    ----------
    coeff_ao : ``(2*nao, nspinor)`` complex — Kramers-paired spinors in the AO basis.
    s_ao : ``(nao, nao)`` — the scalar AO overlap.
    layout : :class:`kuiva.basis.layout.AOLayout`.
    reference : :class:`kuiva.basis.reference.AtomicReferenceSet` — the front end's
        ``atomic_reference=True`` product. Without one this cannot run, and the message says
        which knob to set.
    n_elec_total : int — electrons in the molecule.
    atom, l : the target character, addressed exactly as
        :func:`kuiva.mcscf.casci.active_space_by_character` addresses it (an index, a unique
        element symbol, or a sequence of either whose reference orbitals are pooled).
    occupation : ``(nspinor,)`` — spinor occupations of ``coeff_ao``. The rotation happens
        within groups of equal occupation, never across them.
    n_shells : int
        Shells of the target ``l`` to project onto. ``2`` is the **double shell**: the target
        shell plus its correlating partner, which is what a Ln/An calculation needs and what
        no character threshold finds (the correlating shell is diffuse and covalent).
    threshold : float — projection eigenvalue above which a pair enters the active space.
    max_pairs : int, optional
        Refuse rather than return more than this many pairs. ⚠ Worth setting: an AVAS whose
        threshold is slightly too low returns a perfectly plausible active space one or two
        pairs too large, and the cost of that is discovered only when the CI runs.
    """
    # ⚠ The atom and angular-momentum resolution is **imported, not re-derived**: `avas` and
    # `active_space_by_character` are two routes to one object and must accept the same
    # `(atom, l)` spellings, with the same refusals (an ambiguous element symbol, a principal
    # quantum number). Two resolvers would pass every numerical test and still be two APIs.
    from .casci import _angular_momentum, _atom_indices, active_space

    if reference is None:
        raise ValueError(
            "AVAS projects onto the free-atom reference orbitals, which only the front end "
            "can compute: re-run the scalar SCF with atomic_reference=True (they are cached "
            "per element, so the cost is one small atomic SCF per unique element, once per "
            "process).")
    c = np.ascontiguousarray(coeff_ao, dtype=np.complex128)
    nao = int(np.shape(s_ao)[0])
    if c.shape[0] != 2 * nao:
        raise ValueError("the spinors span {} rows against {} AOs"
                         .format(c.shape[0], nao))
    nspinor = c.shape[1]
    if nspinor % 2:
        raise ValueError("a Kramers-paired spinor set has an even number of columns; got {}"
                         .format(nspinor))
    ell = _angular_momentum(l)
    atoms = _atom_indices(layout, atom)
    ref, ref_label = _reference_columns(layout, reference, atoms, ell, int(n_shells))

    # The metric-aware projector onto the reference span, in the AO basis:
    #   P = S A (A^T S A)^-1 A^T S,   so that <p|P|q> = C_p^T P C_q.
    # The pseudo-inverse guards the case the reference vectors of two atoms are nearly
    # linearly dependent (close centres, diffuse reference), which is a property of the
    # molecule rather than a mistake.
    # Two arrays of the two-component AO dimension live at once here: the lifted projector
    # and the congruence's intermediate. Both are transient, but on a large molecule they are
    # the biggest thing this function touches, so the budget is asked before they exist
    # rather than after. Exact, not padded: 1_2 (x) P is real and (2*nao)^2, the intermediate
    # is complex and (2*nao) x nspinor.
    res.require("AVAS projection ({} AOs, {} spinors)".format(nao, nspinor),
                res.array_gb((2 * nao, 2 * nao), np.float64)
                + res.array_gb((2 * nao, nspinor), np.complex128),
                note="the spin-blocked reference projector and the congruence intermediate",
                advice=["project onto fewer centres, or select the active space by character"])
    sa = np.asarray(s_ao) @ ref
    p_ao = sa @ np.linalg.pinv(ref.T @ sa, rcond=1e-10) @ sa.T
    m = c.conj().T @ spin_block_diagonal(p_ao) @ c
    m_pair, residual = fold_to_kramers_pairs(m)
    if residual > PAIR_FOLD_TOL:
        raise ValueError(
            "the projection matrix departs from Kramers-pair structure by {:.3e}, so these "
            "orbitals are not the paired set AVAS folds onto pairs (the projector itself is "
            "spin-free, so this is a statement about the orbitals). The usual cause is an "
            "UNRESTRICTED reference, whose spinors are orthonormal but not Kramers paired -- "
            "spinors 2p and 2p+1 are then the p-th alpha and p-th beta orbital and need not "
            "describe the same thing. Use a restricted or ROHF reference, or select the "
            "active space explicitly by spinor index".format(residual))

    occ = np.asarray(occupation, dtype=float).ravel()
    if occ.size != nspinor:
        raise ValueError("{} occupations for {} spinors".format(occ.size, nspinor))
    # Electrons in the pair, not the per-spinor mean: 2 for a closed pair, 1 for an ROHF
    # singly occupied one, which is what the report prints and what the groups are cut on.
    pair_occ = occ[0::2] + occ[1::2]
    npair = nspinor // 2

    # Rotate inside each group of equal occupation: mixing orbitals of different occupation
    # would change the density, which is the one thing this transformation may not do.
    #
    # ⚠ **The ordering inside a group is what puts the active space where it belongs.** An
    # occupied group is ordered by *ascending* projection and an empty one by *descending*,
    # so the orbitals carrying the character sit at the inner edge of each group and the
    # selection comes out as one contiguous block straddling the occupied/virtual boundary
    # -- the standard AVAS layout. Sorting every group the same way instead puts the most
    # d-like *occupied* orbital at column 0, below the core: a perfectly valid active space
    # on paper, with an orbital ordering nothing downstream expects and no reader can read.
    eigenvalues = np.zeros(npair)
    rotation = np.zeros((npair, npair), dtype=np.complex128)
    for value in sorted(set(np.round(pair_occ, 8).tolist()), reverse=True):
        group = np.nonzero(np.round(pair_occ, 8) == value)[0]
        w, v = np.linalg.eigh(m_pair[np.ix_(group, group)])
        order = np.argsort(w) if value > 0.0 else np.argsort(-w)
        eigenvalues[group] = w[order]
        rotation[np.ix_(group, group)] = v[:, order]

    keep = np.nonzero(eigenvalues >= float(threshold))[0]
    if keep.size == 0:
        best = np.sort(eigenvalues)[::-1][:5]
        raise ValueError(
            "no orbital carries {:.2f} of the {} character: the largest projections are {}. "
            "Lower `threshold`, or check that the reference shell is the one meant"
            .format(threshold, ref_label, np.round(best, 4).tolist()))
    if max_pairs is not None and keep.size > int(max_pairs):
        raise ValueError(
            "AVAS selected {} Kramers pairs at threshold {:.2f}, above the max_pairs = {} "
            "asked for; the projections at the cut are {}. Raise the threshold or the limit"
            .format(keep.size, threshold, max_pairs,
                    np.round(np.sort(eigenvalues)[::-1][:keep.size + 2], 4).tolist()))
    if int(keep.max() - keep.min()) != keep.size - 1:
        # Not reachable through the ordering above; asserted because a non-contiguous active
        # block is exactly the symptom of that ordering having been changed, and it is the
        # kind of thing that produces a valid-looking calculation nobody can read.
        log.warning("the AVAS selection is not a contiguous block of Kramers pairs (%s): the "
                    "active orbitals do not sit together around the Fermi level, which "
                    "usually means an occupation group was ordered the wrong way",
                    keep.tolist())
    dropped = np.setdiff1d(np.arange(npair), keep)
    gap = (float(eigenvalues[keep].min() - eigenvalues[dropped].max())
           if dropped.size else float("nan"))
    if np.isfinite(gap) and gap < 0.05:
        log.warning("the AVAS eigenvalue gap at the cut is only %.3f (kept %.4f, dropped "
                    "%.4f): the threshold and not the electronic structure is what chose "
                    "this active space, and a small change to either would choose a "
                    "different one. Look at the spectrum before trusting the selection",
                    gap, float(eigenvalues[keep].min()), float(eigenvalues[dropped].max()))

    rotated = rotate_kramers_pairs(c, rotation, np.arange(nspinor))
    columns = np.concatenate([[2 * int(g), 2 * int(g) + 1] for g in keep])
    description = "AVAS: {} pairs projected onto {} at threshold {:.2f}".format(
        keep.size, ref_label, threshold)
    space = active_space(columns, nspinor, n_elec_total, n_active_elec=n_active_elec,
                         description=description)
    log.debug("AVAS: eigenvalues %s, kept pairs %s",
              np.round(np.sort(eigenvalues)[::-1][:keep.size + 3], 4).tolist(),
              keep.tolist())
    return AVASResult(coeff=rotated, space=space, eigenvalues=eigenvalues, selected=keep,
                      occupations=pair_occ, gap=gap, fold_residual=residual,
                      reference=ref_label)


__all__ = ["AVASResult", "DEFAULT_AVAS_THRESHOLD", "PAIR_FOLD_TOL", "avas"]
