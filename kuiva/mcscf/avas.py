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

⚠ Two selection rules, because a shell has a count and not a threshold
----------------------------------------------------------------------
``threshold=`` is the published rule: every pair above a projection cut. ``n_pairs=`` is the
**count-stated** rule: exactly the ``k`` pairs of largest projection, which is what a
*shell* is — a Dy 4f shell is seven pairs whatever the eighth pair's projection is, and the
default threshold has been measured taking two ligand sigma pairs *more* than a 3d shell on
TiCl3. The two are exclusive, both report the eigenvalue **gap at the cut**, and both warn
when it is small: a count that lands inside a tight group of projections chose the space by
its count rather than by the electronic structure, exactly as a threshold in the same place
would have. Which rule ran is recorded on the result (:attr:`AVASResult.mode`) and written
into the active space's description, because the two are different physical statements and a
reader of a stored product has to be able to tell them apart.

The count-stated rule is a **departure from the published method**, which states a
threshold; it is what lets the automatic shell construction say "the valence shell" without
a knob in front of it.

⚠ Several shells are ONE projection onto the union, never two projections in sequence
-----------------------------------------------------------------------------------------
``shells=[(atom, l), ...]`` projects onto the span of every listed reference shell at once —
a 3d on one centre and a 4f on another, or a 4f with its own 5d. Two calls in sequence are
**not** the same thing and are not a supported way to get it: within an occupation group the
pairs outside the second projector's span are degenerate at eigenvalue zero in it, so the
second rotation returns an arbitrary basis of them and re-mixes the pairs the first call
selected. One projector, one diagonalization per occupation group, and the selection is the
invariant span of the largest eigenvalues whatever order the shells were listed in. A union
reference set is the published method's own form (Sayfutyarova et al. project onto any list of
atomic valence orbitals, a metal 3d together with ligand 2p among their examples); what is
added here is the count over the union and the per-shell attribution below.

What the union does not say by itself is which shell a selected pair belongs to, and that is
answered by a diagonal read, not a rotation: :attr:`AVASResult.component_projections` holds
every rotated pair's projection onto each listed shell *separately*, so a caller attributes a
pair to the shell it projects onto most without rotating anything a second time. ⚠ The
separate projections need not sum to the union's where two shells' spans overlap (neighbouring
atoms); the attribution is an ``argmax`` and says nothing more than that.

References
----------
* E. R. Sayfutyarova, Q. Sun, G. K.-L. Chan, G. Knizia, "Automated Construction of Molecular
  Active Spaces from Atomic Valence Orbitals", J. Chem. Theory Comput. 13, 4063 (2017),
  doi:10.1021/acs.jctc.7b00128. The projection, the occupied/virtual separation and the
  eigenvalue threshold follow this work; the reference set and the count-stated rule above do
  not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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
    space : :class:`kuiva.mcscf.casci.ActiveSpace` or None
        The active space, in the rotated set's numbering. ``None`` from
        :func:`avas_projection`, which is the projection **without** a space: see that
        function for the one case where the electron count cannot be derived here.
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
    mode : str
        Which rule produced :attr:`selected`: ``"threshold"`` (a projection cut) or
        ``"count"`` (the ``n_pairs`` form). The two are different physical statements —
        a *shell* has a count whatever the next pair's projection is — and a report that
        prints one for the other misstates how the space was chosen.
    cut : float
        The number the rule was stated with: the threshold, or the requested pair count.
    reference_statement : str
        The selection as one sentence -- what the active space's ``description`` is set to.
    components : tuple of str
        One label per shell of a ``shells=`` projection, in the order they were listed; empty
        for the single ``(atom, l)`` form.
    component_projections : ndarray ``(n_components, npair)`` or None
        Projection of every Kramers pair of the **rotated** set onto each listed shell on its
        own -- the diagonal read that attributes a selected pair to a shell. ``None`` for the
        single form, where there is nothing to attribute.
    """

    coeff: np.ndarray
    space: object
    eigenvalues: np.ndarray
    selected: np.ndarray
    occupations: np.ndarray
    gap: float = float("nan")
    fold_residual: float = 0.0
    reference: str = ""
    mode: str = "threshold"
    cut: float = float("nan")
    reference_statement: str = ""
    components: Tuple[str, ...] = ()
    component_projections: Optional[np.ndarray] = None

    @property
    def n_pairs(self) -> int:
        return int(np.size(self.selected))

    def owners(self, pairs=None) -> np.ndarray:
        """Index into :attr:`components` of the shell each pair projects onto most.

        ``pairs`` defaults to :attr:`selected`. ⚠ Ties go to the **first-listed** shell, so the
        attribution is deterministic; a caller that needs a count per shell checks it, because
        an ``argmax`` that does not come out as the shells' sizes means the orbitals do not
        separate the way the statement says.
        """
        if self.component_projections is None:
            raise ValueError("a single-shell projection has no components to attribute to; "
                             "project with shells=[...] for per-shell attribution")
        pairs = self.selected if pairs is None else pairs
        pairs = np.asarray(pairs, dtype=int).ravel()
        return np.argmax(np.asarray(self.component_projections)[:, pairs], axis=0)

    def report(self, logger=None, *, context: int = 3) -> None:
        """The INFO summary: the selected pairs and the eigenvalues either side of the cut."""
        logger = logger or log
        out.subsection(logger, "AVAS active-space construction")
        rule = ("count stated: {:.0f} pair(s)".format(self.cut) if self.mode == "count"
                else "projection threshold {:.3f}".format(self.cut))
        out.entries(logger, [
            ("reference orbitals", self.reference),
            ("selection rule", rule),
            ("Kramers pairs selected", self.n_pairs),
            ("eigenvalue gap at the cut", self.gap, "", "small = the {} chose, not "
             "the physics".format("count" if self.mode == "count" else "threshold"),
             "{:.3f}"),
            ("Kramers-pair fold residual", self.fold_residual, "", "", "{:.2e}"),
        ])
        chosen = set(int(x) for x in np.asarray(self.selected).ravel())
        order = np.argsort(-np.asarray(self.eigenvalues))
        shown = [int(i) for i in order[:len(chosen) + context]]
        columns = [out.col_count("pair", 7), out.Column("occupation", "{:.3f}", 12),
                   out.Column("projection", "{:.6f}", 12), out.Column("active", "{:s}", 8)]
        attributed = self.component_projections is not None
        if attributed:
            columns.append(out.col_count("shell", 7))
        table = out.Table(logger, columns).start()
        for i in shown:
            row = [i, float(self.occupations[i]), float(self.eigenvalues[i]),
                   "yes" if i in chosen else "no"]
            if attributed:
                row.append(int(self.owners([i])[0]) + 1)
            table.row(*row)
        table.end("{} pair(s) shown of {}".format(len(shown), np.size(self.eigenvalues)))
        for k, label in enumerate(self.components):
            out.note(logger, "shell {}: {}".format(k + 1, label))
        if self.space is not None:
            self.space.report(logger)

    def __repr__(self) -> str:
        return "AVASResult(pairs={}, {}, gap={:.3f})".format(self.n_pairs, self.mode,
                                                             self.gap)


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
        pure = reference_channel_weights(entry, local_l)[:, ell]
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


def reference_channel_weights(entry, local_l) -> np.ndarray:
    """``(n_orb, max_l+1)`` Mulliken weight of each free-atom reference orbital per ``l``.

    The atomic SCF behind an :class:`~kuiva.basis.reference.AtomicReferenceEntry` is
    constrained spherical, so every atomic orbital has support on **one** angular momentum
    and these weights come out as ones and zeros to rounding; they are computed rather than
    assumed because that is what makes "the 3d of titanium" a measurement on the atomic
    solution instead of a claim about the basis ordering.

    ``local_l`` is the ``ao_l`` column of the atom's own AO block. The metric is the free
    atom's overlap, recovered from the orbitals themselves (:func:`_atom_overlap`).

    Two consumers, deliberately one implementation: :func:`_reference_columns` picks the
    columns of a target ``l`` here, and :mod:`kuiva.autocas.centres` reads the reference
    state's **electron count per channel** off the same weights — an atom is a magnetic
    centre only if its own reference has an open shell of the target ``l``, and that is a
    statement about this solution, not about the label it was given.
    """
    c = np.real(np.asarray(entry.c))
    local_l = np.asarray(local_l, dtype=int)
    s_atom = _atom_overlap(entry)
    n_orb = c.shape[1]
    weights = np.zeros((n_orb, int(local_l.max()) + 1))
    for j in range(n_orb):
        w = c[:, j] * (s_atom @ c[:, j])
        total = max(float(w.sum()), 1e-30)
        for l in range(weights.shape[1]):
            weights[j, l] = float(w[local_l == l].sum() / total)
    return weights


def _atom_overlap(entry) -> np.ndarray:
    """``S`` over one atom's AO block, recovered from its orthonormal reference orbitals.

    The atomic MOs satisfy ``C^T S C = 1`` and ``C`` is square and invertible, so
    ``S = (C C^T)^-1``. Reconstructing it beats carrying a second copy that could disagree
    with the coefficients it belongs to.
    """
    c = np.real(np.asarray(entry.c))
    return np.linalg.inv(c @ c.T)


def _shell_list(atom, l, n_shells: int, shells) -> List[Tuple[object, object, int]]:
    """The projection's reference shells as ``[(atom, l, n_shells), ...]``, one form or the other.

    ``atom=``/``l=`` is the single-shell form and ``shells=`` the union form; giving both is
    refused, because a union that silently dropped the single statement (or the reverse) would
    not be the space its description names. An entry of ``shells`` is ``(atom, l)`` or
    ``(atom, l, n_shells)``, the two-element form taking the ``n_shells`` argument.
    """
    if shells is None:
        if atom is None or l is None:
            raise ValueError("give atom= and l= (one reference shell), or shells=[(atom, l), "
                             "...] (the union of several)")
        return [(atom, l, int(n_shells))]
    if atom is not None or l is not None:
        raise ValueError("give either atom=/l= or shells=, not both: a single shell and a "
                         "union of shells are two statements of what is projected onto")
    listed: List[Tuple[object, object, int]] = []
    for entry in shells:
        if not isinstance(entry, (tuple, list)) or len(entry) not in (2, 3):
            raise ValueError("a shells= entry is (atom, l) or (atom, l, n_shells); got {!r}"
                             .format(entry))
        listed.append((entry[0], entry[1],
                       int(entry[2]) if len(entry) == 3 else int(n_shells)))
    if not listed:
        raise ValueError("shells= is empty: there is nothing to project onto")
    return listed


def _resolved_shells(layout, shells) -> List[Tuple[List[int], int, int]]:
    """``[(atoms, ell, n_shells)]`` with the project's one ``(atom, l)`` resolver, and a refusal
    for an atom named twice at one ``l``: its reference orbitals would enter the union twice,
    which the pseudo-inverse would hide and the attribution could not."""
    from .casci import _angular_momentum, _atom_indices

    resolved: List[Tuple[List[int], int, int]] = []
    seen = set()
    for atom, l, n in shells:
        ell = _angular_momentum(l)
        atoms = [int(a) for a in _atom_indices(layout, atom)]
        for a in atoms:
            if (a, ell) in seen:
                raise ValueError(
                    "{} l = {} is named by more than one entry of shells=: a reference shell "
                    "enters the union once, and a pair cannot be attributed to two copies of "
                    "it".format(layout.atom_label(a), ell))
            seen.add((a, ell))
        resolved.append((atoms, ell, int(n)))
    return resolved


def projection_pair_matrix(coeff_ao: np.ndarray, s_ao: np.ndarray, layout, reference, *,
                           atom=None, l=None, n_shells: int = 1,
                           shells=None) -> Tuple[np.ndarray, float, str]:
    """``(M, fold residual, reference label)``: the AVAS projector folded onto Kramers pairs.

    ``M[p, q] = <p|P|q>`` over the Kramers pairs of ``coeff_ao``, with ``P`` the metric-aware
    projector onto the free-atom reference span of ``(atom, l)`` (``n_shells`` shells of it)
    -- or, with ``shells=``, onto the **union** of the listed spans (see the module
    docstring). **The one construction of that projector**: :func:`avas_projection`
    diagonalizes it, and a caller that needs the projection of an *already rotated* set onto a
    different number of reference shells reads its diagonal instead -- which rotates nothing,
    so it cannot re-mix a selection the way a second :func:`avas_projection` call would.
    """
    listed = _shell_list(atom, l, n_shells, shells)
    c = np.ascontiguousarray(coeff_ao, dtype=np.complex128)
    nao = int(np.shape(s_ao)[0])
    if c.shape[0] != 2 * nao:
        raise ValueError("the spinors span {} rows against {} AOs"
                         .format(c.shape[0], nao))
    nspinor = c.shape[1]
    if nspinor % 2:
        raise ValueError("a Kramers-paired spinor set has an even number of columns; got {}"
                         .format(nspinor))
    if reference is None:
        raise ValueError(
            "AVAS projects onto the free-atom reference orbitals, which only the front end "
            "can compute: re-run the scalar SCF with atomic_reference=True (they are cached "
            "per element, so the cost is one small atomic SCF per unique element, once per "
            "process).")
    blocks, labels = [], []
    for atoms, ell, n in _resolved_shells(layout, listed):
        block, label = _reference_columns(layout, reference, atoms, ell, n)
        blocks.append(block)
        labels.append(label)
    # One shell is the concatenation of one block: bitwise the single-shell construction.
    ref = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)
    ref_label = "; ".join(labels)

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
    return m_pair, float(residual), ref_label


def component_projections(coeff_ao: np.ndarray, s_ao: np.ndarray, layout, reference, *,
                          shells, n_shells: int = 1) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """``((n_shells_listed, npair), labels)``: every pair's projection onto each shell alone.

    The diagonal of :func:`projection_pair_matrix` for one entry of ``shells`` at a time --
    a read of the orbitals as they are, which is what attributes the pairs of a union
    selection to their shells without a second rotation.
    """
    listed = _shell_list(None, None, n_shells, shells)
    rows, labels = [], []
    for entry in listed:
        m, _, label = projection_pair_matrix(coeff_ao, s_ao, layout, reference,
                                             shells=[entry])
        rows.append(np.real(np.diag(m)))
        labels.append(label)
    return np.asarray(rows), tuple(labels)


def _separate_degenerate_shells(w: np.ndarray, v: np.ndarray, weighted: np.ndarray, *,
                                rtol: Optional[float] = None) -> np.ndarray:
    """``v`` with each degenerate run of ``w`` re-diagonalized on the weighted shell projector.

    ``w``/``v`` are one occupation group's union eigenpairs in selection order, ``weighted``
    that group's block of ``sum_k (k+1) P_k`` in the same pair frame the eigenvectors are
    expressed in. A run is consecutive values within the project's relative-gap tolerance
    (:mod:`kuiva.util.degeneracy`), so the (near-)zero null space -- whose relative gaps are
    large -- is left alone. Inside a run the new vectors are ordered by descending weight,
    i.e. by the shells' listing order reversed, which is deterministic.
    """
    from ..util.degeneracy import DEFAULT_GROUP_RTOL, relative_gap

    rtol = DEFAULT_GROUP_RTOL if rtol is None else float(rtol)
    v = np.array(v, copy=True)
    start = 0
    while start < w.size:
        stop = start + 1
        while stop < w.size and abs(relative_gap(w[stop - 1], w[stop])) <= rtol:
            stop += 1
        if stop - start > 1 and abs(float(w[start])) > rtol:
            block = v[:, start:stop]
            u_w, u = np.linalg.eigh(block.conj().T @ weighted @ block)
            v[:, start:stop] = block @ u[:, np.argsort(-u_w, kind="stable")]
        start = stop
    return v


def avas_projection(coeff_ao: np.ndarray, s_ao: np.ndarray, layout, reference, *,
                    atom=None, l=None, occupation: np.ndarray, n_shells: int = 1,
                    threshold: Optional[float] = None,
                    n_pairs: Optional[int] = None,
                    max_pairs: Optional[int] = None,
                    shells=None):
    """The AVAS rotation and selection **without** an active space: ``space`` is ``None``.

    Everything :func:`avas` does except the last step, and it exists for the one caller that
    cannot let this function derive the electron count: ⚠ **a shell that is entirely empty in
    the reference has no aufbau count**, and deriving one here refuses (an odd inactive
    count) or asserts a CAS with no electrons in it. Measured live: CeCl3's scalar ROHF puts
    its single valence electron in a Ce **5d** orbital, so every Ce 4f pair is empty and the
    f-shell selection is a perfectly correct set of seven empty pairs -- whose electron count
    is 1 and comes from the ion, not from the orbitals. Which electron count an empty shell
    gets is a physical decision (:mod:`kuiva.autocas.candidates`), not an arithmetic one, and
    this is the seam that lets the decision live where it is made.

    Every other caller wants :func:`avas`, whose parameters and traps are documented there.
    """
    # ⚠ The atom and angular-momentum resolution is **imported, not re-derived**: `avas` and
    # `active_space_by_character` are two routes to one object and must accept the same
    # `(atom, l)` spellings, with the same refusals (an ambiguous element symbol, a principal
    # quantum number). Two resolvers would pass every numerical test and still be two APIs.
    # (Both are resolved inside `projection_pair_matrix`, the one construction of P.)
    if n_pairs is not None:
        if threshold is not None:
            raise ValueError(
                "give either threshold= or n_pairs=, not both: a projection cut and a pair "
                "count are two different statements of what the active space IS (a "
                "threshold says 'everything this d-like', a count says 'the shell'), and a "
                "run that silently preferred one would not be reproducible from its "
                "description")
        if max_pairs is not None:
            raise ValueError("max_pairs bounds the threshold mode's selection; with n_pairs "
                             "the size is already stated exactly")
        if int(n_pairs) < 1:
            raise ValueError("n_pairs is a number of Kramers pairs and must be positive; "
                             "got {!r}".format(n_pairs))
    if reference is None:
        raise ValueError(
            "AVAS projects onto the free-atom reference orbitals, which only the front end "
            "can compute: re-run the scalar SCF with atomic_reference=True (they are cached "
            "per element, so the cost is one small atomic SCF per unique element, once per "
            "process).")
    c = np.ascontiguousarray(coeff_ao, dtype=np.complex128)
    nspinor = c.shape[1]
    m_pair, residual, ref_label = projection_pair_matrix(c, s_ao, layout, reference,
                                                         atom=atom, l=l, n_shells=n_shells,
                                                         shells=shells)

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
    #
    # ⚠ **A union is degenerate across its shells wherever they do not overlap**, and then the
    # eigenvectors are an arbitrary basis of the block. Two ions 25 A apart project at
    # eigenvalue 1 on BOTH shells, and the diagonalization returned pairs 96/4, 14/86 and
    # 85/15 per cent Ti/Ce: an attribution by argmax on such pairs is the diagonalization's
    # choice, not the molecule's. Inside every degenerate block of the union's eigenvalues the
    # basis is therefore fixed by a second diagonalization, of the shells' projectors weighted
    # by their listing position -- a rotation inside a block of equal eigenvalue and equal
    # occupation, so no union eigenvalue and no density moves, and it separates exactly the
    # shells whose spans the union could not tell apart.
    weighted = None
    if shells is not None and len(_shell_list(None, None, n_shells, shells)) > 1:
        listed = _shell_list(None, None, n_shells, shells)
        weighted = np.zeros_like(m_pair)
        for k, entry in enumerate(listed):
            m_k, _, _ = projection_pair_matrix(c, s_ao, layout, reference, shells=[entry])
            weighted += float(k + 1) * m_k
    for value in sorted(set(np.round(pair_occ, 8).tolist()), reverse=True):
        group = np.nonzero(np.round(pair_occ, 8) == value)[0]
        w, v = np.linalg.eigh(m_pair[np.ix_(group, group)])
        order = np.argsort(w) if value > 0.0 else np.argsort(-w)
        w, v = w[order], v[:, order]
        if weighted is not None:
            v = _separate_degenerate_shells(w, v, weighted[np.ix_(group, group)])
        eigenvalues[group] = w
        rotation[np.ix_(group, group)] = v

    if n_pairs is not None:
        # The count-stated cut. ⚠ Ordered by projection and then by **column index**, so the
        # selection is deterministic when two pairs project equally — which is not a corner
        # case but the symmetric one: the partners of an `e` or `t2` set project identically
        # to machine precision, and an unstable sort would return a different two of them per
        # run. A count that lands inside such a group still warns through the gap below.
        want = int(n_pairs)
        if want > npair:
            raise ValueError("n_pairs = {} asks for more Kramers pairs than the orbital set "
                             "has ({})".format(want, npair))
        keep = np.sort(np.argsort(-eigenvalues, kind="stable")[:want])
        mode, cut = "count", float(want)
    else:
        cut_value = DEFAULT_AVAS_THRESHOLD if threshold is None else float(threshold)
        keep = np.nonzero(eigenvalues >= cut_value)[0]
        mode, cut = "threshold", cut_value
        if keep.size == 0:
            best = np.sort(eigenvalues)[::-1][:5]
            raise ValueError(
                "no orbital carries {:.2f} of the {} character: the largest projections are "
                "{}. Lower `threshold`, or check that the reference shell is the one meant"
                .format(cut_value, ref_label, np.round(best, 4).tolist()))
        if max_pairs is not None and keep.size > int(max_pairs):
            raise ValueError(
                "AVAS selected {} Kramers pairs at threshold {:.2f}, above the max_pairs = "
                "{} asked for; the projections at the cut are {}. Raise the threshold or the "
                "limit".format(keep.size, cut_value, max_pairs,
                               np.round(np.sort(eigenvalues)[::-1][:keep.size + 2],
                                        4).tolist()))
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
                    "%.4f): the %s and not the electronic structure is what chose "
                    "this active space, and a small change to either would choose a "
                    "different one. Look at the spectrum before trusting the selection",
                    gap, float(eigenvalues[keep].min()), float(eigenvalues[dropped].max()),
                    "requested count" if mode == "count" else "threshold")

    rotated = rotate_kramers_pairs(c, rotation, np.arange(nspinor))
    union = shells is not None and len(_shell_list(None, None, n_shells, shells)) > 1
    parts, labels = None, ()
    if union:
        parts, labels = component_projections(rotated, s_ao, layout, reference, shells=shells,
                                              n_shells=n_shells)
    onto = "the union of {}".format(ref_label) if union else ref_label
    description = ("AVAS: the {} pairs of largest projection onto {} (count stated; gap at "
                   "the cut {:.3f})".format(keep.size, onto, gap) if mode == "count"
                   else "AVAS: {} pairs projected onto {} at threshold {:.2f}".format(
                       keep.size, onto, cut))
    if union:
        owner = np.argmax(parts[:, keep], axis=0)
        description += " -- per shell {}".format(
            " + ".join(str(int(np.count_nonzero(owner == k))) for k in range(len(labels))))
    log.debug("AVAS: eigenvalues %s, kept pairs %s",
              np.round(np.sort(eigenvalues)[::-1][:keep.size + 3], 4).tolist(),
              keep.tolist())
    return AVASResult(coeff=rotated, space=None, eigenvalues=eigenvalues, selected=keep,
                      occupations=pair_occ, gap=gap, fold_residual=residual,
                      reference=ref_label, mode=mode, cut=cut,
                      reference_statement=description, components=labels,
                      component_projections=parts)


def avas(coeff_ao: np.ndarray, s_ao: np.ndarray, layout, reference, n_elec_total: int, *,
         atom=None, l=None, occupation: np.ndarray, n_shells: int = 1,
         threshold: Optional[float] = None,
         n_pairs: Optional[int] = None,
         n_active_elec: Optional[int] = None,
         max_pairs: Optional[int] = None,
         shells=None):
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
    shells : sequence of ``(atom, l)`` or ``(atom, l, n_shells)``, optional
        **The union form**, exclusive with ``atom``/``l``: one projection onto the span of
        every listed reference shell -- a 3d centre and a 4f centre, or a 4f shell and its
        5d. ⚠ Not two calls in sequence, which re-mix each other's selection (module
        docstring). The result's :attr:`AVASResult.component_projections` attributes each
        pair to the shell it projects onto most; the count and the threshold apply to the
        union.
    occupation : ``(nspinor,)`` — spinor occupations of ``coeff_ao``. The rotation happens
        within groups of equal occupation, never across them.
    n_shells : int
        Shells of the target ``l`` to project onto. ``2`` is the **double shell**: the target
        shell plus its correlating partner, which is what a Ln/An calculation needs and what
        no character threshold finds (the correlating shell is diffuse and covalent).
    threshold : float, optional
        Projection eigenvalue above which a pair enters the active space; defaults to
        :data:`DEFAULT_AVAS_THRESHOLD`. Exclusive with ``n_pairs``.
    n_pairs : int, optional
        **The count-stated mode**: take exactly this many Kramers pairs, the ones of largest
        projection, instead of everything above a threshold. ⚠ It exists because a *shell*
        has a count and not a threshold — a Dy 4f shell is seven pairs whatever the eighth
        pair's projection is, and :data:`DEFAULT_AVAS_THRESHOLD` is a selection knob that has
        been measured taking two ligand sigma pairs *more* than a 3d shell. The eigenvalue
        **gap at the cut is still the honesty check** and still warns when it is small: a
        count cut inside a tight group of projections chose the space by the count and not by
        the electronic structure, exactly as a threshold cut in the same place would have.
        Exclusive with ``threshold`` and with ``max_pairs`` (both are second statements of
        the size).

        ⚠ **The count is global and reaches across the occupied/empty gap**, which is what a
        shell needs and is measured to be the right rule on both ends of the range it has to
        cover: on a free Dy(3+) ion the seven-pair cut takes the seven occupied 4f pairs
        (projections 1.000) and the fourteen-pair cut of the double shell splits 7 + 7
        exactly, while on CeCl3 -- whose scalar ROHF puts its valence electron in a Ce 5d
        orbital, leaving every 4f pair empty -- it correctly takes seven empty pairs. ⚠ The
        second case is also why an **empty** selection is not an error here: which electron
        count such a shell gets is a physical decision for the caller, which is what
        :func:`avas_projection` exists for.
    max_pairs : int, optional
        Refuse rather than return more than this many pairs. ⚠ Worth setting: an AVAS whose
        threshold is slightly too low returns a perfectly plausible active space one or two
        pairs too large, and the cost of that is discovered only when the CI runs.
    """
    from dataclasses import replace

    from .casci import active_space

    result = avas_projection(coeff_ao, s_ao, layout, reference, atom=atom, l=l,
                             occupation=occupation, n_shells=n_shells,
                             threshold=threshold, n_pairs=n_pairs, max_pairs=max_pairs,
                             shells=shells)
    columns = np.concatenate([[2 * int(g), 2 * int(g) + 1]
                              for g in np.asarray(result.selected, dtype=int)])
    space = active_space(columns, int(np.shape(result.coeff)[1]), n_elec_total,
                         n_active_elec=n_active_elec, description=result.reference_statement)
    return replace(result, space=space)


__all__ = ["AVASResult", "DEFAULT_AVAS_THRESHOLD", "PAIR_FOLD_TOL", "avas",
           "avas_projection", "component_projections", "reference_channel_weights"]
