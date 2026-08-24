"""Full-CI CASCI and CASSCF: the drivers that turn the CI machinery into a calculation.

Where this sits
---------------
``ci/`` holds the CI machinery proper — addressing, the sigma vector, the eigensolver — and
knows nothing about orbital spaces. ``mcscf/orbopt.py`` holds the optimizer and its
:class:`~kuiva.mcscf.orbopt.CASIntegrals`. This module is the seam: it presents the full CI
**as the optimizer's ``ci_solver`` callback**, so a CASSCF is the existing, validated
:func:`~kuiva.mcscf.orbopt.optimize_orbitals` driver with :class:`FullCISolver` plugged into
it and nothing else changed.

⚠ It lives in ``mcscf/`` rather than ``ci/`` for a structural reason: ``mcscf/preopt.py``
imports ``ci/strings.py``, so **``ci/`` may not import ``mcscf/``**, and the adapter needs
:class:`~kuiva.mcscf.orbopt.CASIntegrals`. Same rule that put
:class:`~kuiva.util.errors.SolverFailure` in ``util/``.

The full CI is a **static** solver, and that is why no new controller exists
---------------------------------------------------------------------------
A complete CAS space is fixed by construction, so :class:`FullCISolver` satisfies the
:class:`~kuiva.mcscf.adaptive.AdaptiveCISolver` protocol trivially: :meth:`~FullCISolver.propose`
returns ``None``, :meth:`~FullCISolver.space_key` never changes, and the event controller of
the event-gated driver would correctly degrade to the plain trust-region loop. The driver is therefore
:func:`~kuiva.mcscf.orbopt.optimize_orbitals` — the validated smooth-surface driver — and
**not** ``optimize_orbitals_events``, which would be correct but pointless here. Nothing in
:mod:`kuiva.mcscf.events` or :mod:`kuiva.mcscf.adaptive` needed changing for CASSCF.

⚠ One thing the surface being smooth does *not* buy: bitwise determinism is only as good as
the CI tolerance. The Davidson is warm started from the previous solve, so ``solve`` at the
same integrals twice in a row can differ at the residual tolerance (1e-8). That is far below
the optimizer's ``conv_energy``, which is what makes the accept/reject test meaningful, but it
is why a restart is validated to *converge to the same answer* rather than to reproduce a
trajectory bit for bit (:mod:`kuiva.io.checkpoint`).

Active-space selection
-------------------------------
"An active space stated as an orbital-index window is not a definition another program can
reproduce." Three ways in, in increasing order of how well they travel:

* :func:`active_space` — explicit spinor indices. Exact, and unreproducible by anyone who does
  not have the same orbitals.
* :func:`active_space_by_character` — the **lowest spinors of a given angular-momentum
  character on a given atom**, which is what reproducibility requires of a reference calculation and is
  what the committed Tier-2 records are defined by. Built on the Löwdin reduced orbital
  populations, so it needs the AO layout the front-end carries.
* :meth:`~kuiva.mcscf.preopt.PreoptResult.suggest_active_space` — from the cheap CI's
  occupations. ⚠ A **lower bound**: it finds partially *occupied* orbitals and cannot
  flag an empty one a better treatment would populate.

⚠ Two traps both of these carry, and both have produced plausible wrong answers here before:

* **The electron count fixes the inactive count, not the occupation pattern.** ``n_inactive =
  nelec_total - n_active_elec``, never ``2 * (mo_occ > 0).sum()``: an ROHF singly occupied MO
  has ``occ > 0`` while holding one electron, so that expression over-counts by one per open
  shell. An odd inactive count is refused outright — it would split a Kramers pair
  across the space boundary, which the pairing convention forbids.
* ⚠ **With an unrestricted reference an active space may not be a contiguous spinor range**
: spinors ``2p``/``2p+1`` are then the *p*-th alpha and *p*-th beta orbital and need
  not describe the same thing. :func:`active_space` warns when it is handed a contiguous range
  together with ``kramers_paired=False``.

References
----------
* CASSCF and the CI-plus-orbital-rotation split: B. O. Roos, P. R. Taylor, P. E. M. Siegbahn,
  Chem. Phys. 48, 157 (1980), doi:10.1016/0301-0104(80)80045-0; T. Helgaker, P. Jorgensen,
  J. Olsen, "Molecular Electronic-Structure Theory", Wiley (2000), ch. 12.
* Two-component (spinor) CASSCF, where the CI roots are already the spin-orbit eigenstates:
  H. J. Aa. Jensen, K. G. Dyall, T. Saue, K. Faegri, J. Chem. Phys. 104, 4083 (1996),
  doi:10.1063/1.471644; T. Fleig, J. Olsen, C. M. Marian, J. Chem. Phys. 114, 4775 (2001),
  doi:10.1063/1.1349076.
* Transition density matrices from the one-particle-excitation intermediate: J. Olsen,
  B. O. Roos, P. Jorgensen, H. J. Aa. Jensen, J. Chem. Phys. 89, 2185 (1988),
  doi:10.1063/1.455063.
* Active-space selection by orbital character and by natural occupation: C. J. Stein,
  M. Reiher, J. Chem. Theory Comput. 12, 1760 (2016), doi:10.1021/acs.jctc.6b00156.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from ..ci import kernels
from ..ci.davidson import (DEFAULT_CONV_TOL, DEFAULT_MAX_ITER, DavidsonResult, davidson,
                           davidson_kramers, davidson_kramers_sector,
                           davidson_sector)
from ..ci.sigma import SigmaOperator, assert_time_reversal, gather_block_size
from ..ci.strings import CASSpace, _check_array, _check_output, diagonal_energies
from ..props.multiplet import HARTREE_TO_CM
from ..rdm.rdm import DEFAULT_DEGENERACY_TOL, RDMBuilder, state_average_weights
from ..symm.sectors import SectorTable, assert_sector_symmetry, resolve_state_request
from ..util import output as out
from ..util import resources as res
from ..util.errors import SolverFailure
from ..util.logging import get_logger
from ..util.timing import timer
from .orbopt import CASIntegrals, CASSCFResult, OrbitalSpaces, optimize_orbitals

log = get_logger(__name__)

#: Fraction of a Kramers pair's Löwdin population that must sit on the requested
#: ``(atom, l)`` for :func:`active_space_by_character` to count it. Deliberately below a half:
#: a metal d orbital in a ligand field is 60-98% metal (measured on TiCl3: 94% Ti d), and
#: demanding more would reject exactly the covalently delocalized orbitals an active space is
#: chosen to describe.
DEFAULT_CHARACTER_THRESHOLD = 0.35


# --- The transition-density kernel (a registered kernel) -------------------------------------------------

@kernels.kernel("transition_density")
def transition_density_numpy(bra: np.ndarray, f: np.ndarray, block: int,
                             out: np.ndarray) -> None:
    """Accumulate ``out[I, p*n+q] += sum_K conj(bra[I, K]) f[K, p*n+q]``.

    With ``f`` the intermediate for **one ket state** ``J`` — ``F[K, pq] = sum_L <K|E_pq|L>
    c^J_L``, from :func:`kuiva.ci.sigma.sigma_gather_f_numpy` — this is the column of
    transition density matrices ``gamma^{IJ}_pq = <I|E_pq|J>`` for every bra ``I`` at once
    (one excitation map, one intermediate, three consumers). One GEMM, no new map and no
    second residency.

    Parameters
    ----------
    bra : ``(n_states, ndet)`` ``complex128``, C-contiguous — the bra CI vectors.
    f : ``(ndet, n**2)`` ``complex128``, C-contiguous — the ket's intermediate.
    block : int
        Determinant rows per chunk. ⚠ A **parameter** (B7), sized once by the caller outside
        this call: NumPy has no lazy conjugate, so without chunking the ``conj(bra)``
        temporary is a full copy of the bra stack. A compiled backend calls ``zgemm`` with
        ``transa='C'`` and needs neither.
    out : ``(n_states, n**2)`` ``complex128``, C-contiguous
        Caller-allocated and **caller-zeroed** — this accumulates, so that a caller building
        the whole ``(n_states, n_states)`` block does it one ket at a time into slices of one
        array. Must not alias ``bra`` or ``f``.

    Notes
    -----
    **Portability:** plain arrays and scalars (B1), no hashing (B2), rectangular (B3), dtypes
    and contiguity asserted on entry (B4/B5), caller-provided non-aliasing output (B6),
    blocking as a parameter (B7), no logging, timing, resource check or raise inside the loop
    (B8), no callbacks (B9).

    ⚠ **Reduction order (B10): yes.** The sum over determinants is split across blocks here
    and would be split across threads in a port, and BLAS picks its own order inside each
    block. The fixed **1e-13 relative** parity tolerance applies, not a bitwise one.
    """
    _check_array("transition_density", "bra", bra, np.complex128, 2)
    _check_array("transition_density", "f", f, np.complex128, 2)
    ndet, npair = f.shape
    n_states = bra.shape[0]
    _check_output("transition_density", "out", out, np.complex128, 2, (n_states, npair),
                  bra, f)
    if bra.shape != (n_states, ndet):
        raise ValueError("transition_density: bra has shape {}, expected {}"
                         .format(bra.shape, (n_states, ndet)))
    if block < 1:
        raise ValueError("transition_density: block must be positive, got {}".format(block))

    for lo in range(0, ndet, int(block)):
        hi = min(lo + int(block), ndet)
        out += np.conj(bra[:, lo:hi]) @ f[lo:hi]


# --- Active-space selection ----------------------------------------------------------------

@dataclass(frozen=True)
class ActiveSpace:
    """An active space, stated so that another program could reproduce it.

:attr:`description` is the physical statement — "the 10 lowest spinors of d
    character on Ti" — and is what goes into a reference file or a property dump, never the
    index list, which is meaningless outside the orbital set it was derived from.
    """

    spaces: OrbitalSpaces
    n_elec: int
    description: str = ""
    #: Per-fragment spinor indices when the space was selected as a union of character
    #: fragments (:func:`active_space_by_characters`); ``None`` otherwise. Metadata for
    #: provenance and for a fragment-wise model space — the union in :attr:`spaces` is what
    #: every calculation consumes.
    fragments: Optional[Tuple[Tuple[int, ...], ...]] = None
    #: Irrep labels of the **active** spinors (:class:`kuiva.symm.OrbitalLabels`), sliced
    #: from the reference's own labels in the order :attr:`spaces` lists them; ``None`` when
    #: the front end was not asked for symmetry. ⚠ An active space that is not closed under
    #: the group makes every sector count a lie — one member of a pair inside and its partner
    #: outside — so the selection checks closure and warns; see :func:`active_space_for`.
    labels: Optional[object] = None

    @property
    def n_active(self) -> int:
        return self.spaces.n_active

    def report(self, logger=None) -> None:
        logger = logger or log
        out.entries(logger, [
            ("active space", "CAS({}, {})".format(self.n_elec, self.n_active), "",
             self.description),
            ("inactive / active / virtual spinors", "{} / {} / {}".format(
                self.spaces.n_inactive, self.spaces.n_active, self.spaces.n_virtual)),
            ("determinants", _dimension(self.n_active, self.n_elec)),
        ])
        if self.labels is not None:
            from ..symm.report import sector_table
            sector_table(self.labels, logger, title="active spinors per fermion irrep")

    def __repr__(self) -> str:
        return "ActiveSpace(CAS({}, {}), {})".format(self.n_elec, self.n_active,
                                                     self.description or "explicit")


def _dimension(n_active: int, n_elec: int) -> int:
    from ..ci.strings import cas_dimension
    return cas_dimension(n_active, n_elec)


def active_space(active: Sequence[int], n_orb: int, n_elec_total: int, *,
                 n_active_elec: Optional[int] = None, kramers_paired: bool = True,
                 description: str = "") -> ActiveSpace:
    """Partition ``n_orb`` spinors around an explicit active list.

    The inactive set is the lowest-indexed non-active spinors, filled until the electrons that
    are not active are placed. With ``n_active_elec`` omitted it is derived by aufbau: the
    non-active spinors among the lowest ``n_elec_total`` are inactive and the rest of the
    electrons are active. Both routes end at the same statement,

        ``n_inactive = n_elec_total - n_active_elec``   (a real trap, stated as an equation)

    which is checked, not assumed — and the result is refused if it is **odd**, because an odd
    inactive count splits a Kramers pair across a space boundary.

    ⚠ ``kramers_paired=False`` (an unrestricted reference) additionally warns when the
    active list is a contiguous range, which is the natural thing to write and the wrong thing
    to do there: spinors ``2p``/``2p+1`` are the *p*-th alpha and *p*-th beta orbital and need
    not describe the same physical orbital.
    """
    act = np.unique(np.asarray(active, dtype=int).ravel())
    if act.size == 0:
        raise ValueError("the active space is empty")
    if act[0] < 0 or act[-1] >= n_orb:
        raise ValueError("active spinor indices must lie in [0, {}); got {}..{}"
                         .format(n_orb, int(act[0]), int(act[-1])))
    others = np.setdiff1d(np.arange(n_orb), act)

    if n_active_elec is None:
        # Aufbau: the electrons occupy the lowest n_elec_total spinors, and whichever of those
        # are not active are inactive.
        n_inactive = int(np.count_nonzero(others < n_elec_total))
        n_active_elec = int(n_elec_total) - n_inactive
    else:
        n_active_elec = int(n_active_elec)
        n_inactive = int(n_elec_total) - n_active_elec
    if not 0 <= n_active_elec <= act.size:
        raise ValueError(
            "CAS({}, {}) is impossible: {} electrons cannot occupy {} spinors. Check "
            "n_active_elec against the total electron count ({})"
            .format(n_active_elec, act.size, n_active_elec, act.size, n_elec_total))
    if n_inactive < 0 or n_inactive > others.size:
        raise ValueError("{} inactive spinors are needed for {} electrons with {} of them "
                         "active, but only {} spinors are outside the active space"
                         .format(n_inactive, n_elec_total, n_active_elec, others.size))
    if n_inactive % 2 != 0:
        raise ValueError(
            "an odd inactive count ({}) splits a Kramers pair across the active-space "
            "boundary, which the spinor conventions forbid: every orbital space is defined on "
            "*spatial* orbitals. {} electrons with {} active leaves {} inactive; ask for {} "
            "or {} active electrons instead"
            .format(n_inactive, n_elec_total, n_active_elec, n_inactive,
                    n_active_elec - 1, n_active_elec + 1))

    inactive = others[:n_inactive]
    virtual = others[n_inactive:]
    if not kramers_paired and act.size > 1 and np.array_equal(act, np.arange(act[0],
                                                                             act[-1] + 1)):
        log.warning("the active space is a contiguous spinor range, but this orbital set is "
                    "not Kramers paired (an unrestricted reference): spinors 2p and 2p+1 are "
                    "then the p-th alpha and p-th beta orbital and need not describe the same "
                    "thing. Select by orbital character per spin set instead ")
    spaces = OrbitalSpaces(inactive=inactive, active=act, virtual=virtual, n_orb=int(n_orb))
    return ActiveSpace(spaces=spaces, n_elec=int(n_active_elec),
                       description=description or "{} explicit spinors".format(act.size))


def active_space_by_character(coeff_ao: np.ndarray, overlap: np.ndarray, layout,
                              n_elec_total: int, *, atom, l, n_pairs: int,
                              n_active_elec: Optional[int] = None,
                              threshold: float = DEFAULT_CHARACTER_THRESHOLD,
                              occupation: Optional[np.ndarray] = None,
                              skip_pairs: int = 0) -> ActiveSpace:
    """The lowest ``n_pairs`` Kramers pairs of ``(atom, l)`` character.

    This is the selection rule the committed Tier-2 references are defined by, and the only
    one an independent implementation can reproduce: *"the five lowest pairs of d character on
    the titanium"*, not *"spinors 36-45"*.

    Parameters
    ----------
    coeff_ao : ``(2*nao, n_orb)`` complex — spinors in the AO basis.
    overlap : ``(nao, nao)`` — the scalar AO overlap.
    layout : :class:`kuiva.basis.layout.AOLayout` — from the front-end.
    atom : int, str, or a sequence of either
        Atom index, or an element symbol (which must be unique in the molecule) — or a
        **sequence** of those, whose populations are summed. The sequence form is what a
        multi-centre active space needs: *"the ten lowest Kramers pairs of d character on the
        two titaniums"* is one physical statement and one active space, and no single
        centre carries it. ⚠ An *ambiguous scalar* symbol still raises: in a dimer, ``"Ti"``
        alone does not say whether one centre or both is meant, and which it is is the whole
        point.
    l : int or str
        Angular momentum, as a number or a shell letter (``"s"``, ``"p"``, ``"d"``, ``"f"``).
        ⚠ **No principal quantum number**: those count shells *within the basis* and are
        therefore basis-dependent — the live trap where ``"6p"`` selects core orbitals in
        a segmented set and valence orbitals in a general one. ``(atom, l)`` is unambiguous.
    n_pairs : int
        Kramers pairs to take, i.e. ``2 * n_pairs`` spinors. Whole pairs always.
    threshold : float
        Minimum fraction of a pair's Löwdin population on ``(atom, l)``. See
        :data:`DEFAULT_CHARACTER_THRESHOLD`.
    skip_pairs : int
        Skip this many of the qualifying pairs before taking ``n_pairs`` — an **ordinal
        window** into the same character-ordered list the plain form takes its lowest pairs
        from. *"the 6th to 10th lowest d pairs on Ti"*.

        It exists for the **double shell**: a correlating shell of the same ``l`` cannot be
        named by the union form (:func:`active_space_by_characters` refuses two fragments
        that claim one pair, and without a window two ``(Ti, d)`` fragments claim exactly
        the same pairs), and pooling them into one ``n_pairs = 10`` selection cannot say
        which five are which. ⚠ It is **not** a principal quantum number in disguise: ``n``
        counts shells within the *basis* and is therefore basis-relative, while an ordinal
        within a stated character-and-threshold ordering is something an independent
        implementation reproduces from the description this writes.

    Raises
    ------
    ValueError
        If fewer than ``n_pairs`` pairs clear the threshold — with the populations of the best
        candidates in the message, because the useful answer to "I asked for five d pairs and
        got three" is *what the other two look like*, not a bare failure.
    """
    columns, description = _character_columns(coeff_ao, overlap, layout, atom=atom, l=l,
                                              n_pairs=n_pairs, threshold=threshold,
                                              occupation=occupation,
                                              skip_pairs=_check_skip(skip_pairs))
    return active_space(columns, int(np.shape(coeff_ao)[1]), n_elec_total,
                        n_active_elec=n_active_elec, description=description)


def _character_columns(coeff_ao: np.ndarray, overlap: np.ndarray, layout, *, atom, l,
                       n_pairs: int, threshold: float,
                       occupation: Optional[np.ndarray],
                       skip_pairs: int = 0) -> Tuple[np.ndarray, str]:
    """The selection core of :func:`active_space_by_character`: ``(columns, description)``."""
    from ..props.population import orbital_populations

    n_orb = int(np.shape(coeff_ao)[1])
    if n_orb % 2 != 0:
        raise ValueError("a Kramers-paired spinor set has an even number of columns; got {}"
                         .format(n_orb))
    centres = _atom_indices(layout, atom)
    ell = _angular_momentum(l)
    where = " and ".join(layout.atom_label(i) for i in centres)

    populations = orbital_populations(coeff_ao, overlap, layout, group="kramers",
                                      occupation=occupation)
    on = np.isin(np.asarray(layout.ao_atom), centres) & (np.asarray(layout.ao_l) == ell)
    if not np.any(on):
        raise ValueError("{} has no l = {} functions in this basis".format(where, ell))
    fraction = populations.normalized()[on].sum(axis=0)      # (n_pairs_total,)

    clears = np.nonzero(fraction >= threshold)[0]
    if clears.size < skip_pairs + n_pairs:
        order = np.argsort(-fraction)[:skip_pairs + n_pairs + 3]
        detail = ", ".join("pair {} ({:.0%})".format(populations.labels[g], fraction[g])
                           for g in order)
        raise ValueError(
            "only {} Kramers pair(s) carry at least {:.0%} of their Loewdin population on "
            "{} l={}, but {} were asked for{}. The best candidates are: {}. Lower `threshold` "
            "if the orbitals are more covalent than expected, or select explicitly"
            .format(clears.size, threshold, where, ell, n_pairs,
                    "" if not skip_pairs else " after skipping {}".format(skip_pairs),
                    detail))
    # The ordinal window. Counted over the pairs that *clear the threshold*, in orbital
    # order, so it names a position in the same list the plain form takes its lowest
    # `n_pairs` from -- "the 6th to 10th lowest d pairs on Ti", which an independent
    # implementation can reproduce. ⚠ Deliberately not a principal quantum number: those
    # count shells within the *basis* (see :func:`_angular_momentum`), and the whole reason
    # this window exists is to say "the second d shell" without making that claim.
    chosen = clears[skip_pairs:skip_pairs + n_pairs]          # in orbital order
    columns = np.concatenate([populations.groups[g] for g in chosen])
    description = ("{} lowest Kramers pairs of l={} character on {}".format(
        n_pairs, ell, where) if not skip_pairs else
        "Kramers pairs {}-{} (by ascending orbital order) of l={} character on {}".format(
            skip_pairs + 1, skip_pairs + n_pairs, ell, where))
    log.debug("character selection: pairs %s carry %s of the (%s, l=%d) population",
              chosen.tolist(), np.round(fraction[chosen], 3).tolist(), where, ell)
    return columns, description


def active_space_by_characters(coeff_ao: np.ndarray, overlap: np.ndarray, layout,
                               n_elec_total: int, *, fragments,
                               n_active_elec: Optional[int] = None,
                               threshold: float = DEFAULT_CHARACTER_THRESHOLD,
                               occupation: Optional[np.ndarray] = None) -> ActiveSpace:
    """A union of per-fragment character selections — one active space, several centres.

    ``fragments`` is a sequence of ``(atom, l, n_spinors)`` triples, each the argument set of
    :func:`active_space_by_character` (``n_spinors`` = ``2 * n_pairs``, whole Kramers pairs),
    optionally extended to ``(atom, l, n_spinors, skip_pairs)`` — the ordinal window of
    :func:`active_space_by_character`, which is how a **double shell** is written:
    ``[("Ti", "d", 10), ("Ti", "d", 10, 5)]`` is *"the five lowest d pairs on Ti, plus the
    next five"*, two fragments of the same ``l`` naming disjoint sets of pairs.

    The selections are made independently and unioned; a Kramers pair claimed by two
    fragments is **refused, not shared**, because a pair that clears two thresholds at once
    means the fragments are not the disjoint physical statement they were written as —
    lower the count, offset one of them with ``skip_pairs``, or pool the centres into one
    ``(atom, l)`` selection instead.

    ⚠ On a *symmetric* polynuclear system the canonical orbitals delocalize over the
    equivalent centres and no per-centre threshold is meaningful; that case is the pooled
    ``atom=(i, j, ...)`` form of :func:`active_space_by_character`. This union form is for
    fragments that are genuinely distinct — different elements, different shells.

    Returns an :class:`ActiveSpace` whose :attr:`~ActiveSpace.fragments` records each
    fragment's spinor indices, in the order given.
    """
    resolved: List[Tuple[np.ndarray, str]] = []
    for entry in fragments:
        entry = tuple(entry)
        if len(entry) == 3:
            (atom, l, n_spinors), skip = entry, 0
        elif len(entry) == 4:
            atom, l, n_spinors, skip = entry
        else:
            raise ValueError("each fragment is an (atom, l, n_spinors) triple, optionally "
                             "extended to (atom, l, n_spinors, skip_pairs); got {!r}"
                             .format(entry))
        n_spinors = int(n_spinors)
        if n_spinors <= 0 or n_spinors % 2 != 0:
            raise ValueError("a fragment selects whole Kramers pairs, so n_spinors must be "
                             "positive and even; got {} for {!r}".format(n_spinors, entry))
        resolved.append(_character_columns(coeff_ao, overlap, layout, atom=atom, l=l,
                                           n_pairs=n_spinors // 2, threshold=threshold,
                                           occupation=occupation,
                                           skip_pairs=_check_skip(skip)))

    for (cols_a, desc_a), (cols_b, desc_b) in itertools.combinations(resolved, 2):
        shared = np.intersect1d(cols_a, cols_b)
        if shared.size:
            raise ValueError(
                "fragments '{}' and '{}' both claim spinor(s) {}: the same Kramers pair "
                "clears both thresholds, so these are not disjoint fragments. Lower the "
                "counts, offset one fragment with skip_pairs (the fourth element of the "
                "fragment tuple -- how a second shell of the same l is named), raise "
                "`threshold`, or pool the centres into one (atom, l) selection"
                .format(desc_a, desc_b, shared.tolist()))

    columns = np.concatenate([cols for cols, _ in resolved])
    description = " + ".join(desc for _, desc in resolved)
    space = active_space(columns, int(np.shape(coeff_ao)[1]), n_elec_total,
                         n_active_elec=n_active_elec, description=description)
    # spaces.active is the sorted union; fragments keep the caller's grouping
    return ActiveSpace(spaces=space.spaces, n_elec=space.n_elec, description=description,
                       fragments=tuple(tuple(int(c) for c in cols) for cols, _ in resolved))


def _check_skip(skip_pairs) -> int:
    """Validate an ordinal window offset. One implementation, both selection entry points."""
    skip = int(skip_pairs)
    if skip < 0:
        raise ValueError("skip_pairs counts qualifying Kramers pairs to step over and "
                         "cannot be negative; got {!r}".format(skip_pairs))
    return skip


def _atom_index(layout, atom) -> int:
    if isinstance(atom, (int, np.integer)):
        return int(atom)
    symbols = [str(s).capitalize() for s in layout.atom_symbols]
    matches = [i for i, s in enumerate(symbols) if s == str(atom).capitalize()]
    if not matches:
        raise ValueError("no atom {!r} in this molecule (have {})".format(atom, symbols))
    if len(matches) > 1:
        raise ValueError("{!r} appears {} times ({}); give the atom index, or the list of "
                         "indices for a multi-centre active space, since which centre the "
                         "active space belongs to is the whole point"
                         .format(atom, len(matches), matches))
    return matches[0]


def _atom_indices(layout, atom) -> np.ndarray:
    """Resolve ``atom`` to a sorted array of atom indices. See
    :func:`active_space_by_character`'s ``atom`` parameter."""
    if isinstance(atom, (str, int, np.integer)):
        atom = [atom]
    indices = sorted({_atom_index(layout, a) for a in atom})
    if not indices:
        raise ValueError("the active space must sit on at least one atom")
    return np.asarray(indices, dtype=int)


def _angular_momentum(l) -> int:
    if isinstance(l, (int, np.integer)):
        return int(l)
    letters = "spdfghi"
    key = str(l).strip().lower()
    if len(key) == 1 and key in letters:
        return letters.index(key)
    raise ValueError("angular momentum must be a number or one of {!r}; got {!r}. ⚠ A "
                     "principal quantum number ('6p') is deliberately not accepted: it counts "
                     "shells within the basis and is basis-dependent "
                     .format(letters, l))


# --- The CI solver -------------------------------------------------------------------------

@dataclass
class CASCIResult:
    """One full-CI solve: the states, and everything built from them."""

    energies: np.ndarray                 # (n_states,) active-space eigenvalues [Eh]
    vectors: np.ndarray                  # (n_states, ndet) complex, rows are states
    weights: np.ndarray                  # (n_states,) state-averaging weights actually used
    gamma: np.ndarray                    # (n_act, n_act) state-averaged 1-RDM
    gamma2: np.ndarray                   # (n_act,)*4 state-averaged 2-RDM
    e_core: float                        # inactive + nuclear energy [Eh]
    n_apply: int                         # applications of H
    n_iter: int
    dense: bool = False
    #: The solver that produced this, so that the property dump can ask for transition
    #: densities without a second entry point — they need the excitation map and the ``F``
    #: workspace, which are the solver's and are deliberately not duplicated.
    solver: Optional["FullCISolver"] = None
    #: The orbitals and the orbital-space partition these states were solved at, when the
    #: caller knew them (the drivers below do; :meth:`FullCISolver.solve_active` does not).
    #: ⚠ The property dump needs *both* and they must be the matching pair: a moment matrix
    #: built from one orbital set and a state set solved at another is Hermitian, plausible
    #: and wrong.
    coeff: Optional[np.ndarray] = None
    spaces: Optional[OrbitalSpaces] = None
    #: The active space as a *physical* statement, for the property dump header. An index
    #: window is not a definition another program can reproduce, so the file records this.
    description: str = ""
    #: Irrep name of each state, when the orbitals carry labels; ``"?"`` where a state could
    #: not be classified. ``None`` without symmetry.
    irreps: Optional[Tuple[str, ...]] = None
    #: Largest weight any state carries outside its own sector — a measurement of how far the
    #: orbitals have drifted out of the symmetry, not a property of the states. ``None``
    #: without symmetry.
    sector_leakage: Optional[float] = None
    #: Name of the **full double group** multiplet each state belongs to, when the non-abelian
    #: classification layer ran; ``"?"`` for a block that does not decompose into whole
    #: irreps. ⚠ A property of the degenerate *block*, repeated on its members: a single state
    #: inside one has no irrep, because the eigensolver may return any rotation of the block.
    multiplets: Optional[Tuple[str, ...]] = None
    #: The full :class:`kuiva.symm.classify.Classification`, for a caller that wants the
    #: per-block residuals and traces rather than the names.
    classification: Optional[object] = None

    @property
    def total_energies(self) -> np.ndarray:
        """State energies including the inactive and nuclear contribution [Eh].

        ⚠ These, not :attr:`energies`, are the numbers a spectrum is read off. Because the CI
        is already two-component, they **are** the spin-orbit eigenstates — there is no
        separate spin-orbit mixing step, unlike a RASSI-style two-step treatment.
        """
        return self.energies + self.e_core

    @property
    def energy(self) -> float:
        """The state-averaged total energy — what the orbital optimizer minimizes [Eh]."""
        return float(np.dot(self.weights, self.energies)) + self.e_core

    def transition_densities(self) -> np.ndarray:
        """``gamma^{IJ}_pq = <I|E_pq|J>`` over these states — see
        :meth:`FullCISolver.transition_densities`, whose workspace it reuses."""
        if self.solver is None:
            raise RuntimeError("this result carries no solver, so the excitation map and the "
                               "F workspace the transition densities are built from are not "
                               "available; call FullCISolver.transition_densities directly")
        return self.solver.transition_densities(self.vectors)

    def excitation_energies_cm(self) -> np.ndarray:
        """Relative state energies [cm^-1], from the lowest root.

        ⚠ Relative, always: the *absolute* two-component total energy contains the inactive
        energy and the X2C picture-change shift, and comparing one against another program's
        is only meaningful when the Hamiltonians match exactly. Splittings are what the validation rules
        validates against.
        """
        return (self.energies - self.energies[0]) * HARTREE_TO_CM

    def __repr__(self) -> str:
        return "CASCIResult(n_states={}, E0={:.10f} Eh, <E>={:.10f} Eh)".format(
            self.energies.size, float(self.total_energies[0]), self.energy)


class FullCISolver:
    """Exact CI over a complete CAS space, as the optimizer's ``ci_solver``.

    Implements :class:`~kuiva.mcscf.adaptive.AdaptiveCISolver` as a **static** solver: the
    space is complete, so there is nothing to propose and the chart never changes.
    Call it directly — ``solver(ints)`` — or hand it to
    :func:`~kuiva.mcscf.orbopt.optimize_orbitals`; both reach the same
    :meth:`solve`.

    Everything expensive is built **once**: the determinant addressing, the excitation map,
    the ``F``/``G`` workspace and the RDM accumulator survive across macro-iterations, and a
    new set of integrals replaces two small arrays (:meth:`~kuiva.ci.sigma.SigmaOperator.set_integrals`).
    The RDM builder shares the sigma operator's ``f_buf``, so the 1.1 GB intermediate exists
    once rather than twice.

    Parameters
    ----------
    n_spinor, n_elec : int — the active space, ``C(n_spinor, n_elec)`` determinants.
    n_states : int
        Roots to solve for and average over. ⚠ With an odd electron count Kramers' theorem
        makes every level at least doubly degenerate, so an odd ``n_states`` splits a pair and
        is **refused where the RDMs are built** (:mod:`kuiva.rdm.rdm`), not here.
    weights : sequence, optional — state-averaging weights; uniform by default. They are
        equalized inside a degenerate block whatever is asked for.
    conv_tol : float
        Davidson residual tolerance. ⚠ Sized against the **1-RDM** error, not the energy:
        see :mod:`kuiva.ci.davidson`.
    enforce_kramers : bool
        The explicit escape hatch from the state-averaging policy. With it off no degeneracy check runs
        and the requested weights are used as given — a statement that the caller has some
        other reason to want them (a model Hamiltonian with no time-reversal symmetry, say),
        never a way to avoid the refusal on a physical system.
    kramers : ``"general"`` or ``"restricted"``
        The CI symmetry mode. ⚠ ``"general"`` is the **default and the reference path**, and
        stays so: every committed reference in this project is a general-complex result and
        none of them moves for this option existing.

        ``"restricted"`` is the time-reversal-adapted path: the eigensolver keeps its
        expansion subspace closed under ``T``, so one stored direction spans a whole Kramers
        pair and half as many applications of ``H`` deliver the same spectrum
        (:func:`~kuiva.ci.davidson.davidson_kramers`). The states it returns are
        pair-expanded and identical in shape and convention to the general path's, so nothing
        downstream — the state-averaging gate, the boundary diagnostic, ``props/``, NEVPT2 —
        learns a new convention.

        Refused at construction for an **even** electron count and for an odd ``n_states``.
        Both are structure, not policy: ``T^2 = +1`` for an even count, so there is no Kramers
        pairing to exploit and the saving there would be a different theorem (a real
        Hamiltonian, not a halved subspace); and a mode whose unit of work is the pair cannot
        deliver half of one.

        ⚠ It additionally requires the active-space integrals to commute with ``T``, which is
        a property of the *orbitals* and is checked at every solve
        (:func:`~kuiva.ci.sigma.assert_time_reversal`), not assumed.
    """

    #: A complete CAS space has exactly one surface, and its identity is its dimensions.
    KEY_PREFIX = "full-ci"
    #: The CI symmetry modes, in the vocabulary the rest of the program uses for them. The
    #: selected one reaches :meth:`space_key` when it is not the default, so that switching
    #: mode across a restart is a **chart change** — the optimizer's curvature memory is
    #: cleared rather than transported between two different solvers.
    KRAMERS_MODES = ("general", "restricted")

    def __init__(self, n_spinor: int, n_elec: int, *, n_states=1,
                 weights: Optional[Sequence[float]] = None,
                 conv_tol: float = DEFAULT_CONV_TOL, max_iter: int = DEFAULT_MAX_ITER,
                 max_subspace: Optional[int] = None, backend: Optional[str] = None,
                 block: Optional[int] = None,
                 degeneracy_tol: float = DEFAULT_DEGENERACY_TOL,
                 on_split: str = "raise", enforce_kramers: bool = True,
                 kramers: str = "general", symmetry: Optional[object] = None,
                 warm_start: bool = True) -> None:
        self.n_spinor = int(n_spinor)
        self.n_elec = int(n_elec)
        self.symmetry = symmetry
        self.state_request = None
        if isinstance(n_states, dict):
            if symmetry is None:
                raise ValueError(
                    "n_states={irrep: n} needs irrep labels for the active spinors, and this "
                    "solver was given none. Run the front end with point_group= so the "
                    "orbitals carry labels, or ask for a plain lowest-n state count")
            self.n_states = int(sum(int(v) for v in n_states.values()))
        else:
            self.n_states = int(n_states)
        self.requested_weights = None if weights is None else np.asarray(weights, float)
        self.conv_tol = float(conv_tol)
        self.max_iter = int(max_iter)
        self.max_subspace = max_subspace
        self.backend = backend
        self.degeneracy_tol = float(degeneracy_tol)
        self.on_split = on_split
        self.enforce_kramers = bool(enforce_kramers)
        self.warm_start = bool(warm_start)
        # ⚠ The mode is validated *after* the sectors are resolved, because in the combined
        # mode the resolution is what fixes the state count: a per-irrep request under
        # Kramers restriction also delivers the conjugate sector's partners, and the parity
        # check below is about the total.
        self.kramers = str(kramers)
        if self.kramers not in self.KRAMERS_MODES:
            raise ValueError("kramers must be one of {}, got {!r}"
                             .format(", ".join(map(repr, self.KRAMERS_MODES)), kramers))

        self.space = CASSpace(self.n_spinor, self.n_elec, backend=backend)
        if self.n_states > self.space.ndet:
            raise ValueError("asked for {} states of a {}-determinant space"
                             .format(self.n_states, self.space.ndet))
        self._sectors = None
        if symmetry is not None:
            if len(symmetry) != self.n_spinor:
                raise ValueError(
                    "{} labels for a {}-spinor active space; the labels must be the active "
                    "columns in the order the active space lists them"
                    .format(len(symmetry), self.n_spinor))
            self._sectors = SectorTable.build(self.space.occupations(), symmetry.labels,
                                              symmetry.group)
            if isinstance(n_states, dict):
                self.state_request = resolve_state_request(n_states, self._sectors)
                for label, count in self.state_request:
                    size = self._sectors.size(label)
                    if count > size:
                        raise ValueError(
                            "asked for {} states of {}, which holds {} determinants"
                            .format(count, self._sectors.name(label), size))
                out.entry(log, "state selection", "per irrep", "",
                          ", ".join("{}: {}".format(self._sectors.name(t), n)
                                    for t, n in self.state_request))
        self._conjugate_units = None
        if self.kramers == "restricted" and self.state_request is not None:
            self._conjugate_units = self._resolve_conjugate_units()
            self.n_states = int(sum(2 * pairs for _, _, pairs in self._conjugate_units))
        self.kramers = self._validate_kramers(self.kramers)
        self._kramers_map = self.space.kramers() if self.kramers == "restricted" else None
        if self._kramers_map is not None:
            # One entry, at the point of selection, naming what the mode changes and what it
            # does not: a non-default symmetry mode that announces itself is the same
            # discipline the cheap-end Hamiltonians follow.
            out.entry(log, "CI symmetry mode", "Kramers-restricted", "",
                      "time-reversal-closed Davidson subspace; {} pairs solved for {} "
                      "states, energies and RDMs in the general convention"
                      .format(self.n_states // 2, self.n_states))
        self._block = block
        self._sigma: Optional[SigmaOperator] = None
        self._rdms: Optional[RDMBuilder] = None
        self._guess: Optional[np.ndarray] = None
        #: The most recent solve, for callers that want the states rather than the RDMs the
        #: optimizer asked for — the spectrum, the CI vectors, the transition densities.
        self.last: Optional[CASCIResult] = None
        self.n_solves = 0

    def _resolve_conjugate_units(self, requests=None, *, strict: bool = True):
        """``[(label, mask, n_pairs)]`` for a per-irrep request under Kramers restriction.

        ⚠ **The indivisible unit of a Kramers-restricted per-irrep selection is a CONJUGATE
        PAIR of sectors, not a sector.** Time reversal conjugates an irrep label, so ``T``
        maps the determinants of ``lambda`` onto those of ``conj(lambda)``: a sector is
        ``T``-closed only when it is self-conjugate, and otherwise only the union of the two
        is. Solving inside a single non-self-conjugate sector would be solving in a subspace
        the operator does not preserve.

        The consequence for the user surface is stated where it is requested and again here:
        asking for ``n`` states of ``lambda`` also returns ``n`` states of
        ``conj(lambda)`` — they are the time-reversed partners and there is no calculation in
        which one exists without the other. Asking for both members separately is therefore
        refused rather than being silently counted twice.

        ``requests`` defaults to this solver's own selection; the boundary diagnostic passes
        its own (larger) counts through the same resolution, with ``strict=False`` so that an
        odd margin in a self-conjugate sector is rounded up to a whole pair rather than
        refused — a margin is a number of roots to look at, not a state average.
        """
        group = self._sectors.group
        units = []
        claimed = set()
        for label, count in (self.state_request if requests is None else requests):
            conjugate = group.conjugate(label)
            if label in claimed or conjugate in claimed:
                raise ValueError(
                    "{} and {} are a conjugate pair of irreps, and under "
                    "kramers='restricted' they are one indivisible unit: time reversal maps "
                    "each onto the other, so asking for states of one already returns the "
                    "partners in the other. Request only one of them"
                    .format(self._sectors.name(label),
                            self._sectors.name(conjugate if conjugate in claimed else label)))
            claimed.add(label)
            self_conjugate = conjugate == label
            mask = self._sectors.mask(label)
            if not self_conjugate:
                if conjugate not in self._sectors.sectors:
                    raise ValueError(
                        "the active space holds determinants of {} but none of its conjugate "
                        "{}, so the two do not form a time-reversal-closed subspace and "
                        "kramers='restricted' cannot solve in it. The active space is not "
                        "closed under time reversal"
                        .format(self._sectors.name(label), group.irrep_name(conjugate)))
                claimed.add(conjugate)
                mask = mask | self._sectors.mask(conjugate)
                pairs = int(count)
            else:
                if int(count) % 2 and strict:
                    raise ValueError(
                        "{} is self-conjugate, so its states come in Kramers pairs inside it "
                        "and a count of {} would split one; ask for {} or {}"
                        .format(self._sectors.name(label), count, count - 1, count + 1))
                pairs = (int(count) + 1) // 2
            available = int(np.count_nonzero(mask))
            if 2 * pairs > available:
                raise ValueError(
                    "asked for {} Kramers pairs ({} states) of the time-reversal-closed "
                    "subspace of {}, which holds {} determinants"
                    .format(pairs, 2 * pairs, self._sectors.name(label), available))
            units.append((label, mask, pairs))
        return units

    def _validate_kramers(self, kramers: str) -> str:
        """Check the symmetry mode against the space, refusing rather than degrading."""
        mode = str(kramers)
        if mode not in self.KRAMERS_MODES:
            raise ValueError("kramers must be one of {}, got {!r}"
                             .format(", ".join(map(repr, self.KRAMERS_MODES)), kramers))
        if mode == "general":
            return mode
        if self.n_spinor % 2 != 0:
            raise ValueError(
                "kramers='restricted' needs a Kramers-paired active space, which has an even "
                "number of spinors; got {}".format(self.n_spinor))
        if self.n_elec % 2 == 0:
            raise ValueError(
                "kramers='restricted' is the odd-electron theorem and this active space holds "
                "{} electrons. With an even count T^2 = +1: there is no Kramers degeneracy to "
                "pair states into and no halved subspace to win — what an even count offers "
                "instead is a basis in which H is real, which is a different construction and "
                "is not implemented. Use kramers='general'".format(self.n_elec))
        if self.n_states % 2 != 0:
            raise ValueError(
                "kramers='restricted' solves whole Kramers pairs, so n_states must be even; "
                "got {}. An odd count would split a pair, which is refused where the "
                "state-averaged RDMs are built in any case — ask for {} or {} states"
                .format(self.n_states, self.n_states - 1, self.n_states + 1))
        return mode

    def _davidson(self, sigma: SigmaOperator, diagonal: np.ndarray, n_states: int, *,
                  guess: Optional[np.ndarray], label: str, requests=None,
                  **kwargs) -> DavidsonResult:
        """Whichever eigensolver this solver's symmetry mode selects.

        ⚠ The two return **the same object in the same convention** — ascending energies,
        rows are states, pair-expanded. That is what keeps every caller below (the CASCI
        driver, :meth:`spectrum`, the boundary diagnostic) mode-agnostic, and it is the
        contract that makes the mode a non-default option rather than a second code path
        through the whole module.
        """
        if requests is not None:
            return self._davidson_sectors(sigma, diagonal, requests, guess=guess,
                                          label=label, **kwargs)
        if self._kramers_map is None:
            return davidson(sigma, diagonal, n_states, guess=guess,
                            conv_tol=self.conv_tol, max_iter=self.max_iter,
                            max_subspace=self.max_subspace, label=label, **kwargs)
        # Rounded up, then truncated by the caller: the boundary diagnostic asks for a margin
        # that need not be even, and a pair is this mode's indivisible unit of work.
        n_pairs = (int(n_states) + 1) // 2
        return davidson_kramers(sigma, self._kramers_map, diagonal, n_pairs, guess=guess,
                                conv_tol=self.conv_tol, max_iter=self.max_iter,
                                max_subspace=self.max_subspace, label=label, **kwargs)

    def _davidson_sectors(self, sigma: SigmaOperator, diagonal: np.ndarray, requests,
                          *, guess: Optional[np.ndarray], label: str,
                          **kwargs) -> DavidsonResult:
        """One :func:`~kuiva.ci.davidson.davidson_sector` solve per requested irrep.

        The sectors are independent problems — ``H`` does not connect them — so this is a
        sequence of small solves rather than one large one, and the roots come back **merged
        and sorted ascending**, in the same convention as every other path here. A caller
        therefore cannot tell from the result that the spectrum was assembled: the state
        table, the state-averaging gate and the RDM builder all see the ordinary object.

        ⚠ The merged ordering is what the gate then works on, deliberately. A degenerate block
        spanning two sectors — which happens whenever the physical group is larger than the
        abelian one being used — is one block to the gate, and it has to be, or a per-irrep
        selection would be a way to walk past the refusal that exists to stop exactly this.
        """
        energies: List[np.ndarray] = []
        vectors: List[np.ndarray] = []
        n_apply = 0
        n_iter = 0
        dense = True
        # ⚠ In the combined mode the unit of work is a **conjugate pair of sectors**, because
        # that is the smallest subspace time reversal preserves; see
        # :meth:`_resolve_conjugate_units` and :func:`~kuiva.ci.davidson.
        # davidson_kramers_sector`.
        units = (self._resolve_conjugate_units(requests, strict=False)
                 if self._kramers_map is not None else
                 [(sector, self._sectors.mask(sector), int(count))
                  for sector, count in requests])
        for sector, mask, count in units:
            solve = (davidson_kramers_sector if self._kramers_map is not None
                     else davidson_sector)
            extra = (self._kramers_map,) if self._kramers_map is not None else ()
            result = solve(
                sigma, *extra, diagonal, mask, int(count), guess=guess,
                conv_tol=self.conv_tol, max_iter=self.max_iter,
                max_subspace=self.max_subspace,
                label="{} {}".format(label, self._sectors.name(sector)), **kwargs)
            energies.append(result.energies)
            vectors.append(result.vectors)
            n_apply += result.n_apply
            n_iter = max(n_iter, result.n_iter)
            dense = dense and result.dense
        e = np.concatenate(energies)
        v = np.concatenate(vectors, axis=0)
        order = np.argsort(e, kind="stable")
        return DavidsonResult(energies=np.ascontiguousarray(e[order]),
                              vectors=np.ascontiguousarray(v[order]),
                              residuals=np.zeros(e.size), n_iter=n_iter, n_apply=n_apply,
                              converged=True, dense=dense)

    def state_irreps(self, vectors: Optional[np.ndarray] = None, energies=None):
        """``(names, leakage)`` for CI vectors — which irrep each state carries.

        ``None`` when this solver has no labels. ⚠ The classification is **per degenerate
        block**, because a single state inside one has no sector: two conjugate sectors meet
        at the same energy in every Kramers pair of a molecule whose group is bigger than the
        abelian one, and the eigensolver may return any rotation of the block. A block that
        decomposes into two sectors is named for both, and ``leakage`` measures how far its
        per-sector weights are from whole numbers — which is the part that *is* a statement
        about the orbitals. See :meth:`kuiva.symm.sectors.SectorTable.classify`.
        """
        if self._sectors is None:
            return None
        if vectors is None:
            vectors, energies = self.last.vectors, self.last.energies
        return self._sectors.classify(vectors, energies=energies,
                                      degeneracy_tol=self.degeneracy_tol)

    def _check_integrals(self, h: np.ndarray, eri: np.ndarray) -> None:
        """⚠ The Kramers-restricted path's one precondition, checked at **every** solve.

        ``[H, T] = 0`` is a property of the orbitals, not of the method, and a CASSCF rotates
        the orbitals at every macro-iteration. The general-complex optimizer is free to leave
        the Kramers-preserving subgroup; in practice it does not (the measured drift is
        roundoff), but "in practice" is not what a mode whose whole structure rests on the
        symmetry may run on. The check costs one pass over the active ``n^4`` — microseconds
        against a Davidson solve — and it is the only thing standing between a broken pairing
        and a converged, exactly degenerate, wrong spectrum.
        """
        if self._sectors is not None and self.state_request is not None:
            # Same discipline, same reason: a per-irrep spectrum solved on orbitals that have
            # drifted out of the symmetry is converged, degenerate-looking and wrong, and the
            # check costs one pass over the active n^4 against a Davidson solve. It runs only
            # when the sectors are actually *solved in*; labels used for classification alone
            # measure the drift instead of refusing on it.
            assert_sector_symmetry(h, eri, self.symmetry.labels, self.symmetry.group.moduli,
                                   what="per-irrep state selection")
        if self._kramers_map is None:
            return
        assert_time_reversal(h, eri, what="kramers='restricted'")

    # -- the AdaptiveCISolver contract ---------------------------------------------------
    def solve(self, ints: CASIntegrals) -> Tuple[float, np.ndarray, np.ndarray]:
        """``(energy, gamma, Gamma)`` at these integrals — the ci_solver contract."""
        result = self.casci(ints)
        return result.energy, result.gamma, result.gamma2

    __call__ = solve

    def propose(self, ints: CASIntegrals) -> None:
        """``None``, always: a complete CAS space has nothing to select."""
        return None

    def adopt(self, key: Hashable) -> None:
        raise ValueError("a complete CAS space never proposes an alternative, so there is "
                         "nothing to adopt (key {!r})".format(key))

    def space_key(self) -> str:
        """The identity of this solver's surface, as the checkpoint and the chart-scoping
        rules use it.

        ⚠ The symmetry mode is appended only when it is **not** the default, so that every
        checkpoint written before the mode existed still matches and its warm start is still a
        warm start. A key that changed for everyone would have silently downgraded every
        stored restart to a cold one — the cheapest possible way to make a non-default option
        cost something to people who never selected it.
        """
        key = "{}:{}:{}".format(self.KEY_PREFIX, self.n_spinor, self.n_elec)
        if self.kramers != "general":
            key = "{}:{}".format(key, self.kramers)
        if self.state_request is not None:
            # Same rule and the same reason as the symmetry mode above: the key moves only for
            # a solver that actually selects per irrep, so every checkpoint written without
            # symmetry still matches its own and keeps its warm start.
            key = "{}:{}[{}]".format(key, self._sectors.group.name,
                                     ",".join("{}={}".format(self._sectors.name(t), n)
                                              for t, n in self.state_request))
        return key

    # -- the solve itself ------------------------------------------------------------------
    def casci(self, ints: CASIntegrals, *, level: Optional[int] = None) -> CASCIResult:
        """Solve and keep everything, not just what the optimizer needs."""
        if ints.spaces.n_active != self.n_spinor:
            raise ValueError("these integrals carry {} active spinors; this solver was built "
                             "for {}".format(ints.spaces.n_active, self.n_spinor))
        h_eff = np.ascontiguousarray(ints.h_active_effective())
        eri = ints.active_eri()
        return self.solve_active(h_eff, eri, e_core=ints.e_core, level=level)

    def solve_active(self, h: np.ndarray, eri: np.ndarray, *, e_core: float = 0.0,
                     level: Optional[int] = None) -> CASCIResult:
        """The bare active-space problem, without any orbital-space context.

        Used by :meth:`casci` and directly by tests and by anyone holding an active-space
        Hamiltonian already. ``e_core`` is added to every state energy and is what makes the
        result comparable across macro-iterations.
        """
        self._check_integrals(h, eri)
        sigma = self._operator(h, eri)
        with timer("full CI: diagonal"):
            # <I|H|I> from the *unfolded* h: the h~ folding of ci/sigma.py belongs to the
            # E_pq resolution, not to a Slater-Condon diagonal element.
            diagonal = diagonal_energies(self.space, h, eri)
        guess = self._guess if self.warm_start else None
        kwargs: Dict[str, Any] = {}
        if level is not None:
            kwargs["level"] = level
        result: DavidsonResult = self._davidson(sigma, diagonal, self.n_states, guess=guess,
                                                label="CAS", requests=self.state_request,
                                                **kwargs)
        self._guess = result.vectors

        # ⚠ The weights are equalized *here*, once, and the same array is used for the energy
        # and for the RDMs. Letting the builder do it independently would leave the reported
        # state-averaged energy weighted differently from the density matrices it is supposed
        # to be the expectation value of -- a discrepancy at exactly the size of the weight
        # correction, i.e. invisible unless a block is actually degenerate.
        if self.enforce_kramers:
            weights = state_average_weights(result.energies, self.n_elec,
                                            self.requested_weights, tol=self.degeneracy_tol,
                                            on_split=self.on_split)
        elif self.requested_weights is None:
            weights = np.full(self.n_states, 1.0 / self.n_states)
        else:
            weights = self.requested_weights / np.sum(self.requested_weights)
        gamma, gamma2 = self._builder()(result.vectors, weights,
                                        energies=result.energies,
                                        enforce_kramers=self.enforce_kramers,
                                        degeneracy_tol=self.degeneracy_tol,
                                        on_split=self.on_split)
        self.n_solves += 1
        self.last = CASCIResult(energies=result.energies, vectors=result.vectors,
                                weights=weights, gamma=gamma, gamma2=gamma2,
                                e_core=float(e_core), n_apply=result.n_apply,
                                n_iter=result.n_iter, dense=result.dense, solver=self)
        if self._sectors is not None:
            # A per-irrep solve returns sector-pure states by construction, so its states are
            # classified one at a time; a general solve's degenerate blocks are not, and are
            # classified as blocks (:meth:`state_irreps`).
            # ⚠ A per-irrep solve returns sector-pure states, so each can be classified on
            # its own -- **except** in the combined Kramers-restricted mode, where a Kramers
            # pair spans a conjugate pair of sectors and the eigensolver may return any
            # rotation inside it. There the block is the only invariant unit, exactly as in a
            # plain lowest-n solve.
            names, leakage = self.state_irreps(
                result.vectors,
                energies=None if (self.state_request is not None
                                  and self._kramers_map is None) else result.energies)
            self.last.irreps = tuple(names)
            self.last.sector_leakage = float(np.max(leakage)) if leakage.size else 0.0
        return self.last

    def spectrum(self, h: np.ndarray, eri: np.ndarray, n_states: int, *,
                 e_core: float = 0.0, seed_warm_start: bool = False) -> np.ndarray:
        """``n_states`` eigenvalues at these integrals — **no RDMs, nothing stored**.

        For :func:`state_average_boundary`, which needs roots this solver is not averaging
        over. It reuses the sigma operator and its ``F``/``G`` workspace, so it adds a Davidson
        solve and **no new residency**; it deliberately leaves :attr:`last` and the warm start
        alone, because the caller's states are the ones the calculation is about.

        ⚠ Not a substitute for :meth:`solve_active`: no state averaging, no Kramers check, no
        density. Asking it for the states you intend to use would skip the state-averaging gate entirely.

        ``seed_warm_start`` is the one sanctioned exception to "leaves the warm start alone",
        and it exists for the **pre-flight** boundary check of :func:`casscf`. There the
        vectors this solve produces are eigenvectors at exactly the integrals the optimizer's
        first CI solve will use, so handing them over turns that solve into a couple of
        Davidson iterations and makes the whole pre-flight cost about the difference between
        solving ``n_states`` roots and ``n_states + margin`` of them. ⚠ Only pass it when the
        *next* solve is genuinely at these integrals; a stale guess is not wrong but it is not
        free either.
        """
        n_states = int(n_states)
        if not 1 <= n_states <= self.space.ndet:
            raise ValueError("asked for {} states of a {}-determinant space"
                             .format(n_states, self.space.ndet))
        self._check_integrals(h, eri)
        sigma = self._operator(h, eri)
        diagonal = diagonal_energies(self.space, h, eri)
        result: DavidsonResult = self._davidson(
            sigma, diagonal, n_states, guess=self._guess if self.warm_start else None,
            label="CAS boundary", requests=self._boundary_requests(n_states - self.n_states))
        if seed_warm_start and self.warm_start:
            # ⚠ Truncated to what was asked for. In the Kramers-restricted mode an odd request
            # is rounded up to a whole pair, and seeding the warm start with the extra state
            # would hand the next solve one more vector than it has roots -- harmless, but the
            # two paths are kept identical here on purpose.
            self._guess = result.vectors[:n_states]
        return np.asarray(result.energies[:n_states], dtype=float) + float(e_core)

    def sector_spectra(self, h: np.ndarray, eri: np.ndarray, extra: int, *,
                       e_core: float = 0.0) -> "Dict[Tuple[int, ...], np.ndarray]":
        """``{irrep label: ascending energies}`` with ``extra`` roots beyond the selection.

        The per-irrep form of :meth:`spectrum`, and what the boundary diagnostic reads: a
        selection that ends inside a near-degenerate cluster *of one sector's own states*
        breaks the average exactly as a plain count does, and only a per-sector spectrum can
        see it. No RDMs, nothing stored, and the warm start is left alone.
        """
        if self.state_request is None:
            raise RuntimeError("this solver does not select per irrep, so it has no "
                               "per-sector spectra; use spectrum()")
        self._check_integrals(h, eri)
        sigma = self._operator(h, eri)
        diagonal = diagonal_energies(self.space, h, eri)
        guess = self._guess if self.warm_start else None
        spectra: Dict[Tuple[int, ...], np.ndarray] = {}
        for request in self._boundary_requests(extra):
            label = request[0]
            result = self._davidson_sectors(sigma, diagonal, [request], guess=guess,
                                            label="CAS boundary")
            energies = np.asarray(result.energies, dtype=float)
            if self._kramers_map is not None \
                    and self._sectors.group.conjugate(label) != label:
                # ⚠ The solve ran on the conjugate PAIR of sectors, so every root came back
                # twice -- once in each sector. One of each is this sector's, which is what
                # the request counted and what the boundary index below has to be against.
                energies = energies[::2]
            spectra[label] = energies + float(e_core)
        return spectra

    def _boundary_requests(self, extra: int):
        """Per-sector counts for the boundary diagnostic: ``margin`` extra roots in **each**
        selected sector.

        ⚠ The boundary question is per sector, because the selection is: a sector whose count
        ends inside a near-degenerate cluster of *its own* states breaks the average exactly
        as a plain count does, and a margin spread over the union would leave some sectors
        with none. Nothing is averaged over the extra roots; they exist to be looked at.
        """
        if self.state_request is None:
            return None
        margin = max(1, int(extra))
        return [(label, min(count + margin, self._sectors.size(label)))
                for label, count in self.state_request]

    def transition_densities(self, vectors: Optional[np.ndarray] = None) -> np.ndarray:
        """``gamma^{IJ}_pq = <I|E_pq|J>`` over the stored states.

        Returns ``(n_states, n_states, n, n)``. The diagonal ``gamma^{II}`` reproduces the
        state 1-RDMs exactly, which is the cheapest check that this and :mod:`kuiva.rdm.rdm`
        agree; the off-diagonal blocks are what the magnetic-moment matrices are built
        from.

        ⚠ It reuses the sigma operator's ``f_buf``, so the caller's ``F`` is destroyed. That
        is deliberate — the alternative is a second 1.1 GB residency — and it is why this is a
        method rather than a free function.
        """
        if self._sigma is None:
            raise RuntimeError("solve first: the transition densities are built from the "
                               "sigma operator's excitation map and workspace")
        vectors = self.last.vectors if vectors is None else vectors
        vectors = np.atleast_2d(np.ascontiguousarray(vectors, dtype=np.complex128))
        n_states, n = vectors.shape[0], self.n_spinor
        arrays = self.space.excitation_arrays()
        gather = kernels.resolve("sigma_gather_f", self.backend)
        accumulate = kernels.resolve("transition_density", self.backend)
        block = gather_block_size(self.n_elec, self.space.n_empty, self.space.ndet,
                                  self._block)
        res.require("transition densities ({} states, {} spinors)".format(n_states, n),
                    res.array_gb((n_states, n_states, n, n), np.complex128),
                    note="the (I,J) block of gamma^{IJ}",
                    advice=["ask for fewer states"])
        # Indexed [ket, bra] so that each kernel call writes a **C-contiguous** slice — the
        # layout is part of the contract (B5) and a strided view is refused on entry. The
        # transposition to the conventional (bra, ket) order happens once, at the end.
        columns = np.zeros((n_states, n_states, n * n), dtype=np.complex128)
        with timer("transition densities"):
            for ket in range(n_states):
                gather(np.ascontiguousarray(vectors[ket]), *arrays, n, block,
                       self._sigma.f_buf)
                accumulate(vectors, self._sigma.f_buf, block, columns[ket])
        return np.ascontiguousarray(
            columns.transpose(1, 0, 2).reshape(n_states, n_states, n, n))

    def one_body_moments(self, ops: np.ndarray,
                         vectors: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``<I|A_k|I>``, ``<I|A_k A_k|I>`` and the state 1-RDMs, in one pass.

        Parameters
        ----------
        ops : ``(n_ops, n, n)`` complex — **Hermitian** one-electron operators over the
            active spinors. Hermiticity is asserted, because the square is evaluated as
            ``|| A|I> ||^2`` and that identity is what makes this cheap.
        vectors : ``(n_states, ndet)``, optional — the stored states by default.

        Returns
        -------
        ``(expect, square, rdm1)`` — ``(n_ops, n_states)`` real, ``(n_ops, n_states)`` real,
        ``(n_states, n, n)`` complex.

        Why the square is computed this way
        -----------------------------------
        ``A^2`` is a **two**-body operator, so the obvious route is one 2-RDM per state --
        the most expensive object the CI produces, for a diagnostic. But ``A`` is one-body,
        so ``A|I>`` is one contraction of the excitation map ``F[K,pq] = <K|E_pq|I>`` that
        :meth:`transition_densities` and the sigma vector already build, and the square is
        that vector's norm. This is the **fourth** consumer of the one intermediate: one
        gather per state, then a GEMV per operator.

        Everything here is **orchestration** and none of it is a port candidate: the gather
        is the already-registered ``sigma_gather_f`` kernel and the contraction is a BLAS
        GEMV, so this adds no kernel contract of its own.

        ⚠ The norm is taken over the **CAS determinant space only**. For a spin-like
        operator with matrix elements connecting the active space to the inactive or virtual
        one, ``A|I>`` has a component outside that space and this undercounts the square. The
        missing part is computable from ``rdm1`` and the off-diagonal operator blocks --
        which is why the 1-RDMs come back from here rather than being asked for separately
        (:func:`kuiva.props.spin.spin_analysis` is the caller that does it).

        ⚠ It reuses the sigma operator's ``f_buf``, so the caller's ``F`` is destroyed --
        the same contract, and for the same reason, as :meth:`transition_densities`.
        """
        if self._sigma is None:
            raise RuntimeError("solve first: these are built from the sigma operator's "
                               "excitation map and workspace")
        a = np.ascontiguousarray(ops, dtype=np.complex128)
        if a.ndim == 2:
            a = a[None]
        n = self.n_spinor
        if a.ndim != 3 or a.shape[1:] != (n, n):
            raise ValueError("the operators must be (n_ops, {0}, {0}); got {1}"
                             .format(n, np.shape(ops)))
        worst = float(np.max(np.abs(a - a.conj().transpose(0, 2, 1)))) if a.size else 0.0
        if worst > 1e-10:
            raise ValueError(
                "one_body_moments evaluates <A^2> as ||A|I>||^2, which holds only for a "
                "Hermitian A; the operators are non-Hermitian by {:.3e}".format(worst))
        vectors = self.last.vectors if vectors is None else vectors
        vectors = np.atleast_2d(np.ascontiguousarray(vectors, dtype=np.complex128))
        n_states, n_ops = vectors.shape[0], a.shape[0]
        arrays = self.space.excitation_arrays()
        gather = kernels.resolve("sigma_gather_f", self.backend)
        block = gather_block_size(self.n_elec, self.space.n_empty, self.space.ndet,
                                  self._block)
        flat = a.reshape(n_ops, n * n)
        expect = np.zeros((n_ops, n_states))
        square = np.zeros((n_ops, n_states))
        rdm1 = np.zeros((n_states, n, n), dtype=np.complex128)
        # One (ndet,) scratch column, reused: A|I> for one operator. Next to the f_buf it is
        # gathered from (ndet x n^2, already reserved) this is noise, so it is a transient
        # rather than a declared residency.
        work = np.empty(self.space.ndet, dtype=np.complex128)
        with timer("one-body state moments"):
            for i in range(n_states):
                gather(np.ascontiguousarray(vectors[i]), *arrays, n, block,
                       self._sigma.f_buf)
                bra = np.conj(vectors[i])
                rdm1[i] = (bra @ self._sigma.f_buf).reshape(n, n)
                for k in range(n_ops):
                    np.matmul(self._sigma.f_buf, flat[k], out=work)
                    expect[k, i] = float(np.real(bra @ work))
                    square[k, i] = float(np.real(np.vdot(work, work)))
        return expect, square, rdm1

    # -- internals -------------------------------------------------------------------------
    def _operator(self, h: np.ndarray, eri: np.ndarray) -> SigmaOperator:
        if self._sigma is None:
            self._sigma = SigmaOperator(self.space, h, eri, backend=self.backend,
                                        block=self._block)
        else:
            self._sigma.set_integrals(h, eri)
        return self._sigma

    def _builder(self) -> RDMBuilder:
        if self._rdms is None:
            assert self._sigma is not None
            self._rdms = RDMBuilder(self.space, backend=self.backend,
                                    f_buf=self._sigma.f_buf, block=self._block)
        return self._rdms

    @property
    def ndet(self) -> int:
        return self.space.ndet

    @property
    def n_apply(self) -> int:
        return 0 if self._sigma is None else self._sigma.n_apply

    def reset_guess(self) -> None:
        """Forget the Davidson warm start (a fresh problem, or a restart from disk)."""
        self._guess = None

    def set_guess(self, vectors: Optional[np.ndarray]) -> None:
        """Install CI vectors as the next solve's Davidson guess (the checkpoint stores
        exactly "what is needed to restart Davidson from a good guess")."""
        self._guess = (None if vectors is None else
                       np.atleast_2d(np.ascontiguousarray(vectors, dtype=np.complex128)))

    def __repr__(self) -> str:
        return "FullCISolver(CAS({}, {}), {} determinants, {} states, {} solves, {})".format(
            self.n_elec, self.n_spinor, self.ndet, self.n_states, self.n_solves,
            self.kramers)


# --- Drivers --------------------------------------------------------------------------------

def casci(factors, h_ao: np.ndarray, c_spinor: np.ndarray, spaces: OrbitalSpaces,
          n_elec: int, *, n_states=1, e_nuc: float = 0.0,
          weights: Optional[Sequence[float]] = None, report: bool = True,
          solver: Optional[FullCISolver] = None, classifier=None,
          **solver_kwargs) -> CASCIResult:
    """A full CI at **fixed orbitals** — the CASSCF energy without the orbital optimization.

    Everything a CASSCF macro-iteration does, once: transform, solve, build the RDMs. Useful
    on its own (a CASCI on optimized orbitals from elsewhere) and as the reference a CASSCF is
    checked against — the CASSCF energy must be lower, and at a full active space over *all*
    spinors both must equal the SCF energy.
    """
    c_spinor = np.ascontiguousarray(c_spinor)
    ints = CASIntegrals.build(factors, h_ao, c_spinor, spaces, e_nuc=e_nuc)
    solver = solver or FullCISolver(spaces.n_active, n_elec, n_states=n_states,
                                    weights=weights, **solver_kwargs)
    if report:
        out.subsection(log, "CASCI")
        out.entries(log, [
            ("active space", "CAS({}, {})".format(n_elec, spaces.n_active)),
            ("determinants", solver.ndet),
            ("states", solver.n_states, "",
             "" if solver.state_request is None else ", ".join(
                 "{}: {}".format(solver._sectors.name(t), n)
                 for t, n in solver.state_request)),
        ])
    result = solver.casci(ints, level=logging.INFO if report else None)
    result.coeff, result.spaces = c_spinor, spaces
    classify_multiplets(result, classifier, on_split=solver.on_split, report=report,
                        level=logging.INFO if report else logging.DEBUG)
    if report:
        _report_states(result)
    return result


def classify_multiplets(result: CASCIResult, classifier, *, on_split: str = "raise",
                        report: bool = True, level: Optional[int] = None,
                        rebuild_at=None) -> None:
    """Label a converged spectrum by the full double group and gate on whole multiplets.

    ⚠ **A failure to classify may never kill a calculation that would otherwise have run.**
    The layer needs the active space to be closed under the *full* group, which a perfectly
    legitimate active space often is not, so everything except the boundary refusal degrades
    to a warning and no labels. What does refuse is the one thing classification is for: a
    state count that cuts a multiplet whose dimension theory fixes
    (:func:`kuiva.symm.classify.assert_multiplet_boundary`).
    """
    from ..symm.classify import assert_multiplet_boundary
    from ..symm.classify import report as _report_multiplets
    if classifier is None:
        return
    try:
        # ⚠ Rebuilt **inside** the guard, not before it. An operator matrix belongs to the
        # orbitals it came from, so a converged CASSCF needs new ones -- and if the converged
        # orbitals have left the symmetry, building them is exactly where it fails, which is
        # a labelling that cannot be made rather than a calculation that went wrong.
        if rebuild_at is not None:
            classifier = classifier.rebuild(*rebuild_at)
        classification = classifier.classify(result.vectors, result.energies,
                                             degeneracy_tol=result.solver.degeneracy_tol
                                             if result.solver is not None else 1e-6)
    except Exception as exc:                       # noqa: BLE001 - advisory by design
        log.warning("the converged states could not be classified by the full double group "
                    "(%s); the calculation is unaffected and the abelian labels stand", exc)
        return
    result.classification = classification
    result.multiplets = classification.names()
    if report:
        _report_multiplets(classification, log, level=level)
    assert_multiplet_boundary(classification, on_split=on_split,
                              what="the state-averaging gate")


def _report_states(result: CASCIResult) -> None:
    """The state table. Energies in Eh and relative energies in cm^-1, because a
    spectrum is read in the second and a total energy is only meaningful in the first."""
    columns = [out.col_count("state", 7), out.col_energy("E [Eh]"),
               out.Column("rel [cm^-1]", out.CM_FMT, 14),
               out.Column("weight", "{:.4f}", 9)]
    if result.irreps is not None:
        columns.append(out.Column("irrep", "{}", max(10, max(map(len, result.irreps)))))
    if result.multiplets is not None:
        columns.append(out.Column("multiplet", "{}",
                                  max(12, max(map(len, result.multiplets)) + 1)))
    table = out.Table(log, columns)
    table.start()
    relative = result.excitation_energies_cm()
    for i, energy in enumerate(result.total_energies):
        row = [i, energy, relative[i], result.weights[i]]
        if result.irreps is not None:
            row.append(result.irreps[i])
        if result.multiplets is not None:
            row.append(result.multiplets[i])
        table.row(*row)
    footer = "{} applications of H in {} iterations".format(result.n_apply, result.n_iter)
    if result.sector_leakage is not None:
        footer += "; largest weight outside a state's own irrep {:.2e}".format(
            result.sector_leakage)
    table.end(footer)


#: Extra roots solved to measure the state-average boundary. Never averaged over.
#: 8 clears the largest ``2J+1`` manifold an f-shell ion puts near its ground term, so a
#: boundary landing inside a manifold cannot hide by having the whole remainder fall outside
#: the window.
BOUNDARY_MARGIN = 8

#: Above this, the averaged density is reported as **leaning on the spin-orbit structure**:
#: its stability rests entirely on the boundary gap rather than on an invariance of the
#: ensemble. Term-complete and full-space averages measure <=0.02 (machine zero for a free
#: ion); sub-manifold averages measure 0.13-0.71 at symmetric orbitals and down to ~0.07 at
#: their own converged (polarized) ones, so the split is wide but a d^1 doublet can approach
#: the line from above.
SPIN_LEANING_THRESHOLD = 0.05


def ensemble_spin_noninvariance(gamma: np.ndarray, spin_mo: np.ndarray) -> float:
    """``max_k ||[gamma, S_k]||_F / ||gamma||_F`` over the active space.

    Zero exactly when the state-averaged density commutes with the spin rotations, which is
    what a **term-complete** average delivers (the ensemble is invariant under separate
    spatial and spin rotations, so its density carries no spin-orbital entanglement) and
    what a sub-manifold average destroys: a single-J or single-Kramers-doublet ensemble
    weights the spin-orbit structure of its states, and its symmetric point is then
    protected only by the energy gap to the states left out — the mechanism whose measured
    instance is a free-ion g anisotropy 60x apart between two BLAS libraries on identical
    input. The split is wide: every term-complete or full-space average measured below
    0.02, every sub-manifold one at 0.13-0.71 at the symmetric orbitals (a d^1 ligand-field
    doublet reads ~0.07 at its own converged, polarized ones).

    ⚠ A large value is a statement of *exposure*, not a verdict. Measured on an instability
    ladder of free-ion and ligand-field ensembles: a leaning average over a 33 cm^-1 gap
    amplified a 1e-8 Kramers seed by x25-80 per macro-iteration into the state-averaging
    gate's refusal, while leaning averages over 226-4795 cm^-1 gaps damped every seed —
    ligand-field ground-doublet averages among them, so a warning on leaning alone would cry
    wolf on a legitimate standard protocol. And a *locally stable* leaning average can still
    converge from the scalar guess into a different basin with its manifold visibly split
    (measured on a d^6 J-only average: 1.9 cm^-1 of splitting behind clean diagnostics, an
    even electron count keeping the odd-block gate blind), which no static quantity in hand
    detects. Hence: reported beside the boundary gap, never gated on.

    Parameters
    ----------
    gamma : ``(n_act, n_act)`` complex — the state-averaged active 1-RDM.
    spin_mo : ``(3, n_act, n_act)`` complex — the spin operator over the active spinors,
        ``C_act^H  (S_AO x sigma_k/2)  C_act`` (:func:`kuiva.spinor.expand.spin_operator`
        transformed with the active columns).
    """
    gamma = np.asarray(gamma)
    scale = float(np.linalg.norm(gamma)) or 1.0
    return max(float(np.linalg.norm(gamma @ spin_mo[k] - spin_mo[k] @ gamma)) / scale
               for k in range(3))

#: ⚠ **Not a physical tolerance — a statement that the boundary is UNAMBIGUOUS.** Below this
#: the last averaged root and the first one left out are close enough that the selection can
#: reorder across them while the orbitals move, which is the self-reinforcing broken-average mechanism.
#: Measured there: 3.9 cm^-1 at the count that broke, 2058 at the count that works.
BOUNDARY_WARN_CM = 50.0


@dataclass
class BoundaryReport:
    """Where the state average stops, relative to the states it leaves out.

    A state-averaged CASSCF is exactly as symmetric as the set it averages over. A count that
    ends *inside* a near-degenerate manifold makes the averaged density non-invariant, the
    Fock operator built from it splits the shell, and the orbitals then optimize on the broken
    density — self-reinforcing, and every number it produces stays plausible. Nothing in the
    retained spectrum shows it: the only evidence is the root the average did **not** take.
    """

    n_states: int
    ndet: int
    margin: int
    #: Distance from the last averaged root to the first one left out [cm^-1]. ``None`` when
    #: the average spans the whole CI space — then there is no root left out and no boundary,
    #: and ⚠ that is a *pass*, not a gap of zero.
    gap_cm: Optional[float]
    #: The last averaged root and the ones beyond it, relative to the lowest state [cm^-1].
    next_cm: Tuple[float, ...] = ()
    warn_cm: float = BOUNDARY_WARN_CM
    #: Which orbitals this was measured at — :func:`casscf` measures it twice and the two are
    #: different statements. ⚠ Not decoration: a boundary can be clean at one and not at the
    #: other, and it is the **starting** one that decides whether the trajectory is safe.
    where: str = "converged orbitals"
    #: With a per-irrep selection, the irrep whose own boundary is the tightest — the one the
    #: gap above belongs to. ⚠ The question is per sector because the selection is: a count
    #: that ends inside a near-degenerate cluster of one irrep's states breaks the average
    #: exactly as a plain count does, and the union's spectrum cannot see it.
    sector: Optional[str] = None
    #: :func:`ensemble_spin_noninvariance` of the averaged density these states produced, or
    #: ``None`` where it was not measured (no spin operator in hand, or no density exists at
    #: the measurement point — the pre-flight check runs before the first RDM is built).
    #: ⚠ A clean gap and a zero here are **different statements**: the gap says the cut is
    #: unambiguous, this says the ensemble is one the term's symmetry leaves invariant, and
    #: only the two together make the average safe by construction rather than by gap.
    spin_noninvariance: Optional[float] = None

    @property
    def spans_full_ci(self) -> bool:
        return self.gap_cm is None

    @property
    def is_clean(self) -> bool:
        return self.gap_cm is None or self.gap_cm > self.warn_cm

    @property
    def leaning(self) -> Optional[bool]:
        """Whether the averaged density leans on the spin-orbit structure (``None`` =
        not measured)."""
        if self.spin_noninvariance is None:
            return None
        return self.spin_noninvariance > SPIN_LEANING_THRESHOLD

    def report(self, level: int = logging.INFO) -> None:
        """One or two entries at INFO, and a WARNING when the boundary is ambiguous."""
        if self.spans_full_ci:
            out.entry(log, "state-average boundary", "complete", "",
                      "all {} determinants averaged over".format(self.ndet), level=level)
        else:
            out.entry(log, "state-average boundary gap", self.gap_cm, "cm^-1",
                      ("tightest irrep {}, at the {}".format(self.sector, self.where)
                       if self.sector else
                       "to root {} of {}, at the {}".format(self.n_states + 1, self.ndet,
                                                            self.where)),
                      fmt=out.CM_FMT, level=level)
        if self.spin_noninvariance is not None:
            out.entry(log, "state-average spin invariance", self.spin_noninvariance, "",
                      "leaning on the spin-orbit structure" if self.leaning
                      else "spin-rotation invariant ensemble",
                      fmt=out.SCI_FMT, level=level)
        if self.spans_full_ci:
            return
        if not self.is_clean:
            mechanism = ""
            if self.leaning:
                mechanism = (
                    " And this averaged density is not spin-rotation invariant "
                    "(non-invariance {:.2f}), so its symmetric point is protected by "
                    "nothing but this gap: a Kramers-breaking seed grows by orders of "
                    "magnitude per macro-iteration in this regime (measured x25-80 at a "
                    "33 cm^-1 gap)."
                    .format(self.spin_noninvariance))
            log.warning(
                "at the %s the state average ends %.2f cm^-1 below the first root it leaves "
                "out, which is under the %.0f cm^-1 that makes a boundary unambiguous. "
                "Averaging over %d states may be cutting a degenerate manifold in half, and if "
                "it is, the averaged density is not symmetric, the orbitals optimize on that, "
                "and the result looks entirely plausible.%s Next energies "
                "[cm^-1]: %s. Either raise n_states to the next real gap or confirm the set is "
                "complete.",
                self.where, self.gap_cm, self.warn_cm, self.n_states, mechanism,
                ", ".join("{:.2f}".format(x) for x in self.next_cm))


def state_average_boundary(solver: FullCISolver, ints: CASIntegrals, *,
                           margin: int = BOUNDARY_MARGIN,
                           warn_cm: float = BOUNDARY_WARN_CM,
                           where: str = "converged orbitals",
                           seed_warm_start: bool = False,
                           spin_mo: Optional[np.ndarray] = None,
                           gamma: Optional[np.ndarray] = None) -> BoundaryReport:
    """Measure the state-average boundary at these integrals.

    Costs one Davidson solve at fixed orbitals, with no RDMs built and **no new residency**:
    it reuses the solver's sigma operator and workspace, and leaves ``solver.last`` and (unless
    ``seed_warm_start``) the warm start untouched, because the caller's states are the ones
    that matter.

    ⚠ The *caller* pays for the ``CASIntegrals`` as well, so from :func:`casscf` the whole
    check is an integral build plus a Davidson solve — about **one macro-iteration** of the
    optimization that just ran, not a free rider on it. That is the number to weigh when
    deciding whether to switch it off.

    ``spin_mo``/``gamma`` (the active-space spin operator and the state-averaged density)
    additionally measure :func:`ensemble_spin_noninvariance` — microseconds, no solve — so
    the report states not only whether the cut is unambiguous but whether the ensemble is
    one the symmetry leaves invariant at all. Omitted, the field stays ``None``.
    """
    leaning = (None if spin_mo is None or gamma is None
               else ensemble_spin_noninvariance(gamma, spin_mo))
    n_states, ndet = solver.n_states, solver.ndet
    if n_states >= ndet:
        return BoundaryReport(n_states=n_states, ndet=ndet, margin=margin, gap_cm=None,
                              warn_cm=warn_cm, where=where, spin_noninvariance=leaning)
    n_extra = max(1, min(int(margin), ndet - n_states))
    if solver.state_request is not None:
        return _sector_boundary(solver, ints, n_extra, warn_cm=warn_cm, where=where,
                                leaning=leaning)
    energies = solver.spectrum(np.ascontiguousarray(ints.h_active_effective()),
                               ints.active_eri(), n_states + n_extra,
                               seed_warm_start=seed_warm_start)
    relative = (energies - energies[0]) * HARTREE_TO_CM
    return BoundaryReport(n_states=n_states, ndet=ndet, margin=n_extra,
                          gap_cm=float(relative[n_states] - relative[n_states - 1]),
                          next_cm=tuple(float(x) for x in relative[n_states - 1:]),
                          warn_cm=warn_cm, where=where, spin_noninvariance=leaning)


def _sector_boundary(solver: FullCISolver, ints: CASIntegrals, n_extra: int, *,
                     warn_cm: float, where: str,
                     leaning: Optional[float]) -> BoundaryReport:
    """The boundary of a per-irrep selection: the **tightest** sector decides.

    Each requested irrep gets its own margin and its own gap; the report carries the smallest
    of them and names the irrep it belongs to, because one ambiguous sector is enough to break
    the averaged density and a mean over sectors would hide it.
    """
    spectra = solver.sector_spectra(np.ascontiguousarray(ints.h_active_effective()),
                                    ints.active_eri(), n_extra)
    reference = min(float(e[0]) for e in spectra.values())
    gap = None
    binding = None
    tail: Tuple[float, ...] = ()
    for label, count in solver.state_request:
        energies = spectra[label]
        if energies.size <= count:
            continue
        relative = (energies - reference) * HARTREE_TO_CM
        this = float(relative[count] - relative[count - 1])
        if gap is None or this < gap:
            gap = this
            binding = solver._sectors.name(label)
            tail = tuple(float(x) for x in relative[count - 1:])
    return BoundaryReport(n_states=solver.n_states, ndet=solver.ndet, margin=n_extra,
                          gap_cm=gap, next_cm=tail, warn_cm=warn_cm, where=where,
                          sector=binding, spin_noninvariance=leaning)


def _boundary_or_warning(solver: FullCISolver, factors, h_ao: np.ndarray,
                         c_spinor: np.ndarray, spaces: OrbitalSpaces, e_nuc: float,
                         boundary_check: int, *, where: str, level: int,
                         seed_warm_start: bool = False,
                         spin_ao_2c: Optional[np.ndarray] = None,
                         gamma: Optional[np.ndarray] = None) -> Optional[BoundaryReport]:
    """:func:`state_average_boundary` at these orbitals, **downgraded to a warning on failure**.

    ⚠ **A diagnostic may not kill the calculation it is diagnosing.** The extra roots are
    harder to converge than the ones the average uses — they are higher, so the diagonal
    preconditioner is worse there, and they are the ones with no warm start — so a Davidson
    that converges ``n_states`` roots can fail on ``n_states + margin``. Observed on dy3p:
    142 roots stalled at ``max|r| = 1.8e-08`` against a 1e-8 tolerance, which would have
    thrown away an entire converged CASSCF for the sake of a check that is advisory. The
    failure is reported, because "the boundary could not be measured" is a *weaker* statement
    than "the boundary is clean" and the two must not be confused.
    """
    if not boundary_check:
        return None
    spin_mo = None
    if spin_ao_2c is not None and gamma is not None:
        c_act = c_spinor[:, spaces.active]
        spin_mo = np.stack([c_act.conj().T @ spin_ao_2c[k] @ c_act for k in range(3)])
    try:
        report = state_average_boundary(
            solver, CASIntegrals.build(factors, h_ao, c_spinor, spaces, e_nuc=e_nuc),
            margin=int(boundary_check), where=where, seed_warm_start=seed_warm_start,
            spin_mo=spin_mo, gamma=gamma)
    except SolverFailure as exc:
        log.warning("could not measure the state-average boundary at the %s: %s. The average "
                    "may still be cutting a degenerate manifold and nothing here can now say "
                    "whether it does; lower boundary_check, or check the gap "
                    "by hand with a separate CASCI", where, exc)
        return None
    report.report(level=level)
    return report


@dataclass
class CASSCFOutcome:
    """A converged (or budget-stopped) CASSCF: the orbitals *and* the states.

    :attr:`orbital` is exactly what :func:`~kuiva.mcscf.orbopt.optimize_orbitals` returned —
    energy, coefficients, RDMs, convergence and the work bookkeeping — and :attr:`ci` is the
    last full CI solved on those orbitals, i.e. the spectrum. Keeping them separate rather
    than merging them keeps the optimizer's contract visible: it never knew there were states.
    """

    orbital: CASSCFResult
    ci: CASCIResult
    active: ActiveSpace
    solver: FullCISolver
    checkpoint_path: Optional[str] = None
    #: Where the state average stops relative to the states it leaves out, at the
    #: **converged** orbitals. ``None`` only when the check was switched off with
    #: ``boundary_check=0``.
    boundary: Optional[BoundaryReport] = None
    #: The same measurement at the orbitals the optimization **started** from. ⚠ This is the
    #: one that says whether the trajectory was safe; the converged one only says whether the
    #: answer is. They can disagree, and when they do it is this one that matters.
    boundary_initial: Optional[BoundaryReport] = None

    @property
    def energy(self) -> float:
        return self.orbital.energy

    @property
    def coeff(self) -> np.ndarray:
        return self.orbital.coeff

    @property
    def converged(self) -> bool:
        return self.orbital.converged

    @property
    def state_energies(self) -> np.ndarray:
        return self.ci.total_energies

    def __repr__(self) -> str:
        return "CASSCFOutcome(E={:.10f} Eh, |g|={:.2e}, converged={}, {} states)".format(
            self.energy, self.orbital.grad_norm, self.converged, self.ci.energies.size)


def casscf(factors, h_ao: np.ndarray, c_spinor: np.ndarray, spaces: OrbitalSpaces,
           n_elec: int, *, n_states=1, e_nuc: float = 0.0,
           weights: Optional[Sequence[float]] = None,
           solver: Optional[FullCISolver] = None,
           active: Optional[ActiveSpace] = None,
           solver_options: Optional[Dict[str, Any]] = None,
           boundary_check: int = BOUNDARY_MARGIN,
           spin_ao_2c: Optional[np.ndarray] = None, classifier=None,
           **optimizer_kwargs) -> CASSCFOutcome:
    """State-averaged two-component CASSCF: full CI in, orbital rotation out.

    A thin composition, deliberately: :class:`FullCISolver` is handed to the **existing**
    :func:`~kuiva.mcscf.orbopt.optimize_orbitals` and nothing about the optimizer changes.
    ``optimizer_kwargs`` go straight through (``mode``, ``max_iter``, ``conv_grad``,
    ``conv_energy``, ``callback``, ...).

    The state-average boundary is measured **twice**, and the two are different
    statements: :attr:`~CASSCFOutcome.boundary_initial` at the orbitals the optimization starts
    from, and :attr:`~CASSCFOutcome.boundary` at the ones it ends on. ⚠ An incomplete average
    does its damage *along the trajectory*, so the initial one is what says whether the run was
    safe — and it is the only one that exists at all if the odd-block state-averaging gate refuses part way
    through. ``boundary_check=0`` switches both off.

    ``spin_ao_2c`` (the two-component AO spin matrices,
    :func:`kuiva.spinor.expand.spin_operator` on the scalar overlap) additionally measures
    :func:`ensemble_spin_noninvariance` of the converged averaged density and reports it in
    :attr:`~CASSCFOutcome.boundary` — whether the ensemble is invariant under the symmetry,
    which a clean boundary gap deliberately does not claim. The api front-end always passes
    it; direct callers may omit it and lose only that field.

    ⚠ ``mode`` keeps the optimizer's own default (``"auto"``), which is the **robust** choice
    rather than the cheapest, and the measurement is on TlH — 9440
    complex rotation parameters, 12 state-averaged roots, and a CI of *28 determinants* —
    ``auto`` spends ~21 macro-iterations on quasi-Newton steps that do not converge before it
    escalates, while ``mode="second-order"`` converges in 12 at 22× the CPU. So what decides
    the mode is the **orbital** problem, not the CI cost: set ``mode="second-order"``
    explicitly for a heavy element or a large state average, and ⚠ give ``max_iter`` enough
    room to survive the escalation delay, or an expiring budget returns an unconverged iterate
    rather than a slow answer.
    """
    solver = solver or FullCISolver(spaces.n_active, n_elec, n_states=n_states,
                                    weights=weights, **(solver_options or {}))
    report_level = (logging.INFO if optimizer_kwargs.get("report", True) else logging.DEBUG)

    # ⚠ **The boundary is checked BEFORE the optimization as well, and that is the check that
    # can actually save the run**. An incomplete average does its damage *along* the
    # trajectory: the averaged density stops being invariant, the orbitals optimize on that,
    # and the defect is self-reinforcing — so by the time the converged check below runs, the
    # orbitals are already wrong, and if the odd-block state-averaging gate fires on the way there is no
    # converged check at all. Measured on dy3p's scalar CASSCF, where a count that is a
    # boundary of the *two-component* spectrum sits inside a 4-fold block of the spin-free one:
    # the Kramers pairing died at macro-iteration 3 with nothing before it to say why, while
    # this check reports a 0.00 cm^-1 gap at second zero.
    #
    # It is close to free despite being a full Davidson solve: the roots it finds are
    # eigenvectors at exactly the integrals the optimizer's first CI solve uses, so seeding the
    # warm start with them leaves that solve a couple of iterations. What is actually paid is
    # the extra ``margin`` roots and one integral build.
    boundary_initial = _boundary_or_warning(
        solver, factors, h_ao, c_spinor, spaces, e_nuc, boundary_check,
        where="starting orbitals", level=report_level, seed_warm_start=True)

    orbital = optimize_orbitals(factors, h_ao, np.ascontiguousarray(c_spinor), spaces,
                                solver, e_nuc=e_nuc, **optimizer_kwargs)
    if solver.last is None:                                   # pragma: no cover - defensive
        raise RuntimeError("the optimizer returned without ever solving the CI")

    # ⚠ The optimizer's *last* CI solve is not necessarily at the orbitals it returns: a
    # rejected trial step solves the CI at orbitals that are then thrown away, and a run that
    # stops on a rejection (max_iter, or a callback) leaves ``solver.last`` there. The states
    # this outcome reports — and the property matrices built from them — must belong to
    # ``orbital.coeff``, so the mismatch is detected and repaired rather than trusted. The
    # test is exact: a step is rejected only when it *raises* the energy by more than
    # ``conv_energy``, so equal energies mean the same point.
    if solver.last.energy != orbital.energy:
        log.debug("the optimizer's last CI solve was at rejected orbitals (E = %.10f Eh "
                  "against the returned %.10f Eh); re-solving at the returned orbitals so "
                  "the reported states belong to them", solver.last.energy, orbital.energy)
        final = CASIntegrals.build(factors, h_ao, orbital.coeff, spaces, e_nuc=e_nuc)
        solver.casci(final)
    solver.last.coeff, solver.last.spaces = orbital.coeff, spaces
    if active is not None:
        solver.last.description = active.description
    # ⚠ The classifier is built from the *starting* orbitals but applied to the converged
    # ones, so it is rebuilt here rather than reused: an operator matrix belongs to the
    # orbital set it was computed from, and a converged CASSCF has rotated them.
    classify_multiplets(solver.last, classifier, rebuild_at=(orbital.coeff, spaces),
                        on_split=solver.on_split,
                        report=optimizer_kwargs.get("report", True), level=report_level)
    if optimizer_kwargs.get("report", True):
        _report_states(solver.last)

    # ⚠ **Is the state average complete?** The only evidence is a root the average did
    # not take, so this solves a few extra at the converged orbitals and discards them. Cost is
    # one integral build plus one Davidson solve — about a macro-iteration. It runs *after* the
    # states are reported so a WARNING lands next to the spectrum it is about, and it never
    # touches ``solver.last``.
    #
    # With ``spin_ao_2c`` in hand it also measures whether the converged averaged density is
    # spin-rotation invariant — the *other* half of "is this ensemble safe", which a clean
    # gap says nothing about. Only here: the pre-flight check runs before any RDM exists,
    # and the leaning of an ensemble barely moves along a trajectory (measured flat to two
    # digits over every probe of the instability ladder), so one end states it.
    boundary = _boundary_or_warning(solver, factors, h_ao, orbital.coeff, spaces, e_nuc,
                                    boundary_check, where="converged orbitals",
                                    level=report_level, spin_ao_2c=spin_ao_2c,
                                    gamma=solver.last.gamma)
    _report_symmetry_drift(solver, optimizer_kwargs.get("labels"), level=report_level)
    return CASSCFOutcome(orbital=orbital, ci=solver.last, solver=solver,
                         active=active or ActiveSpace(spaces=spaces, n_elec=n_elec),
                         boundary=boundary, boundary_initial=boundary_initial)


#: Weight outside its own irrep above which a converged state is reported as having drifted
#: out of the symmetry. Well above the 1e-14 an exactly masked rotation leaves and far below
#: anything a genuinely broken symmetry produces, so the band between never has to be judged.
SYMMETRY_DRIFT_TOL = 1.0e-6


def _report_symmetry_drift(solver: FullCISolver, labels, *, level: int) -> None:
    """State whether the converged orbitals are still symmetry-pure — measured, not assumed.

    ⚠ A general-complex orbital optimizer is entitled to leave the symmetry-preserving
    subgroup; in practice it does not, but "in practice" is not what a label printed beside a
    state may rest on. With the irrep mask on (``labels=``) purity is **structural** — the
    rotation has no inter-irrep parameters at all — and this says so; without it the drift is
    measured from the converged states and warned about, which is the honest version of the
    same claim.
    """
    if solver._sectors is None:
        return
    leakage = solver.last.sector_leakage
    if leakage is None:
        return
    if labels is not None:
        out.entry(log, "irrep purity", "exact", "",
                  "rotation masked to within each irrep; largest measured leakage {:.1e}"
                  .format(leakage), level=level)
        return
    out.entry(log, "irrep purity", leakage, "",
              "largest weight of a converged state outside its own irrep",
              fmt=out.SCI_FMT, level=level)
    if leakage > SYMMETRY_DRIFT_TOL:
        log.warning("the converged orbitals are no longer symmetry-pure: a state carries "
                    "%.2e of its weight outside its own irrep, against %.1e. The irrep "
                    "labels printed beside the states describe the dominant sector and no "
                    "longer a conserved quantity; re-run with preserve_symmetry=True to "
                    "constrain the rotation, or read the labels as a classification only",
                    leakage, SYMMETRY_DRIFT_TOL)


__all__ = ["FullCISolver", "CASCIResult", "CASSCFOutcome", "ActiveSpace", "BoundaryReport",
           "active_space", "active_space_by_character", "casci", "casscf",
           "state_average_boundary", "ensemble_spin_noninvariance",
           "transition_density_numpy",
           "BOUNDARY_MARGIN", "BOUNDARY_WARN_CM", "SPIN_LEANING_THRESHOLD",
           "SYMMETRY_DRIFT_TOL",
           "DEFAULT_CHARACTER_THRESHOLD"]
