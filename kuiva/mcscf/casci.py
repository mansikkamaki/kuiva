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
from ..ci.davidson import DEFAULT_CONV_TOL, DEFAULT_MAX_ITER, DavidsonResult, davidson
from ..ci.sigma import SigmaOperator, gather_block_size
from ..ci.strings import CASSpace, _check_array, _check_output, diagonal_energies
from ..props.multiplet import HARTREE_TO_CM
from ..rdm.rdm import DEFAULT_DEGENERACY_TOL, RDMBuilder, state_average_weights
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

    @property
    def n_active(self) -> int:
        return self.spaces.n_active

    def report(self, logger=None) -> None:
        out.entries(logger or log, [
            ("active space", "CAS({}, {})".format(self.n_elec, self.n_active), "",
             self.description),
            ("inactive / active / virtual spinors", "{} / {} / {}".format(
                self.spaces.n_inactive, self.spaces.n_active, self.spaces.n_virtual)),
            ("determinants", _dimension(self.n_active, self.n_elec)),
        ])

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
                              occupation: Optional[np.ndarray] = None) -> ActiveSpace:
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

    Raises
    ------
    ValueError
        If fewer than ``n_pairs`` pairs clear the threshold — with the populations of the best
        candidates in the message, because the useful answer to "I asked for five d pairs and
        got three" is *what the other two look like*, not a bare failure.
    """
    columns, description = _character_columns(coeff_ao, overlap, layout, atom=atom, l=l,
                                              n_pairs=n_pairs, threshold=threshold,
                                              occupation=occupation)
    return active_space(columns, int(np.shape(coeff_ao)[1]), n_elec_total,
                        n_active_elec=n_active_elec, description=description)


def _character_columns(coeff_ao: np.ndarray, overlap: np.ndarray, layout, *, atom, l,
                       n_pairs: int, threshold: float,
                       occupation: Optional[np.ndarray]) -> Tuple[np.ndarray, str]:
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
    if clears.size < n_pairs:
        order = np.argsort(-fraction)[:n_pairs + 3]
        detail = ", ".join("pair {} ({:.0%})".format(populations.labels[g], fraction[g])
                           for g in order)
        raise ValueError(
            "only {} Kramers pair(s) carry at least {:.0%} of their Loewdin population on "
            "{} l={}, but {} were asked for. The best candidates are: {}. Lower `threshold` "
            "if the orbitals are more covalent than expected, or select explicitly"
            .format(clears.size, threshold, where, ell, n_pairs, detail))
    chosen = clears[:n_pairs]                                # lowest pairs, in orbital order
    columns = np.concatenate([populations.groups[g] for g in chosen])
    description = "{} lowest Kramers pairs of l={} character on {}".format(
        n_pairs, ell, where)
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
    :func:`active_space_by_character` (``n_spinors`` = ``2 * n_pairs``, whole Kramers pairs).
    The selections are made independently and unioned; a Kramers pair claimed by two
    fragments is **refused, not shared**, because a pair that clears two thresholds at once
    means the fragments are not the disjoint physical statement they were written as —
    lower the count or pool the centres into one ``(atom, l)`` selection instead.

    ⚠ On a *symmetric* polynuclear system the canonical orbitals delocalize over the
    equivalent centres and no per-centre threshold is meaningful; that case is the pooled
    ``atom=(i, j, ...)`` form of :func:`active_space_by_character`. This union form is for
    fragments that are genuinely distinct — different elements, different shells.

    Returns an :class:`ActiveSpace` whose :attr:`~ActiveSpace.fragments` records each
    fragment's spinor indices, in the order given.
    """
    resolved: List[Tuple[np.ndarray, str]] = []
    for entry in fragments:
        try:
            atom, l, n_spinors = entry
        except (TypeError, ValueError):
            raise ValueError(
                "each fragment is an (atom, l, n_spinors) triple; got {!r}".format(entry))
        n_spinors = int(n_spinors)
        if n_spinors <= 0 or n_spinors % 2 != 0:
            raise ValueError("a fragment selects whole Kramers pairs, so n_spinors must be "
                             "positive and even; got {} for {!r}".format(n_spinors, entry))
        resolved.append(_character_columns(coeff_ao, overlap, layout, atom=atom, l=l,
                                           n_pairs=n_spinors // 2, threshold=threshold,
                                           occupation=occupation))

    for (cols_a, desc_a), (cols_b, desc_b) in itertools.combinations(resolved, 2):
        shared = np.intersect1d(cols_a, cols_b)
        if shared.size:
            raise ValueError(
                "fragments '{}' and '{}' both claim spinor(s) {}: the same Kramers pair "
                "clears both thresholds, so these are not disjoint fragments. Lower the "
                "counts, raise `threshold`, or pool the centres into one (atom, l) selection"
                .format(desc_a, desc_b, shared.tolist()))

    columns = np.concatenate([cols for cols, _ in resolved])
    description = " + ".join(desc for _, desc in resolved)
    space = active_space(columns, int(np.shape(coeff_ao)[1]), n_elec_total,
                         n_active_elec=n_active_elec, description=description)
    # spaces.active is the sorted union; fragments keep the caller's grouping
    return ActiveSpace(spaces=space.spaces, n_elec=space.n_elec, description=description,
                       fragments=tuple(tuple(int(c) for c in cols) for cols, _ in resolved))


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
    """

    #: A complete CAS space has exactly one surface, and its identity is its dimensions.
    KEY_PREFIX = "full-ci"

    def __init__(self, n_spinor: int, n_elec: int, *, n_states: int = 1,
                 weights: Optional[Sequence[float]] = None,
                 conv_tol: float = DEFAULT_CONV_TOL, max_iter: int = DEFAULT_MAX_ITER,
                 max_subspace: Optional[int] = None, backend: Optional[str] = None,
                 block: Optional[int] = None,
                 degeneracy_tol: float = DEFAULT_DEGENERACY_TOL,
                 on_split: str = "raise", enforce_kramers: bool = True,
                 warm_start: bool = True) -> None:
        self.n_spinor = int(n_spinor)
        self.n_elec = int(n_elec)
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

        self.space = CASSpace(self.n_spinor, self.n_elec, backend=backend)
        if self.n_states > self.space.ndet:
            raise ValueError("asked for {} states of a {}-determinant space"
                             .format(self.n_states, self.space.ndet))
        self._block = block
        self._sigma: Optional[SigmaOperator] = None
        self._rdms: Optional[RDMBuilder] = None
        self._guess: Optional[np.ndarray] = None
        #: The most recent solve, for callers that want the states rather than the RDMs the
        #: optimizer asked for — the spectrum, the CI vectors, the transition densities.
        self.last: Optional[CASCIResult] = None
        self.n_solves = 0

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
        return "{}:{}:{}".format(self.KEY_PREFIX, self.n_spinor, self.n_elec)

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
        sigma = self._operator(h, eri)
        with timer("full CI: diagonal"):
            # <I|H|I> from the *unfolded* h: the h~ folding of ci/sigma.py belongs to the
            # E_pq resolution, not to a Slater-Condon diagonal element.
            diagonal = diagonal_energies(self.space, h, eri)
        guess = self._guess if self.warm_start else None
        kwargs: Dict[str, Any] = {}
        if level is not None:
            kwargs["level"] = level
        result: DavidsonResult = davidson(
            sigma, diagonal, self.n_states, guess=guess, conv_tol=self.conv_tol,
            max_iter=self.max_iter, max_subspace=self.max_subspace, label="CAS", **kwargs)
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
        sigma = self._operator(h, eri)
        diagonal = diagonal_energies(self.space, h, eri)
        result: DavidsonResult = davidson(
            sigma, diagonal, n_states, guess=self._guess if self.warm_start else None,
            conv_tol=self.conv_tol, max_iter=self.max_iter,
            max_subspace=self.max_subspace, label="CAS boundary")
        if seed_warm_start and self.warm_start:
            self._guess = result.vectors
        return np.asarray(result.energies, dtype=float) + float(e_core)

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
        return "FullCISolver(CAS({}, {}), {} determinants, {} states, {} solves)".format(
            self.n_elec, self.n_spinor, self.ndet, self.n_states, self.n_solves)


# --- Drivers --------------------------------------------------------------------------------

def casci(factors, h_ao: np.ndarray, c_spinor: np.ndarray, spaces: OrbitalSpaces,
          n_elec: int, *, n_states: int = 1, e_nuc: float = 0.0,
          weights: Optional[Sequence[float]] = None, report: bool = True,
          solver: Optional[FullCISolver] = None, **solver_kwargs) -> CASCIResult:
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
            ("states", n_states),
        ])
    result = solver.casci(ints, level=logging.INFO if report else None)
    result.coeff, result.spaces = c_spinor, spaces
    if report:
        _report_states(result)
    return result


def _report_states(result: CASCIResult) -> None:
    """The state table. Energies in Eh and relative energies in cm^-1, because a
    spectrum is read in the second and a total energy is only meaningful in the first."""
    table = out.Table(log, [out.col_count("state", 7), out.col_energy("E [Eh]"),
                            out.Column("rel [cm^-1]", out.CM_FMT, 14),
                            out.Column("weight", "{:.4f}", 9)])
    table.start()
    relative = result.excitation_energies_cm()
    for i, energy in enumerate(result.total_energies):
        table.row(i, energy, relative[i], result.weights[i])
    table.end("{} applications of H in {} iterations".format(result.n_apply, result.n_iter))


#: Extra roots solved to measure the state-average boundary. Never averaged over.
#: 8 clears the largest ``2J+1`` manifold an f-shell ion puts near its ground term, so a
#: boundary landing inside a manifold cannot hide by having the whole remainder fall outside
#: the window.
BOUNDARY_MARGIN = 8

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

    @property
    def spans_full_ci(self) -> bool:
        return self.gap_cm is None

    @property
    def is_clean(self) -> bool:
        return self.gap_cm is None or self.gap_cm > self.warn_cm

    def report(self, level: int = logging.INFO) -> None:
        """One entry at INFO, and a WARNING when the boundary is ambiguous."""
        if self.spans_full_ci:
            out.entry(log, "state-average boundary", "complete", "",
                      "all {} determinants averaged over".format(self.ndet), level=level)
            return
        out.entry(log, "state-average boundary gap", self.gap_cm, "cm^-1",
                  "to root {} of {}, at the {}".format(self.n_states + 1, self.ndet,
                                                       self.where),
                  fmt=out.CM_FMT, level=level)
        if not self.is_clean:
            log.warning(
                "at the %s the state average ends %.2f cm^-1 below the first root it leaves "
                "out, which is under the %.0f cm^-1 that makes a boundary unambiguous. "
                "Averaging over %d states may be cutting a degenerate manifold in half, and if "
                "it is, the averaged density is not symmetric, the orbitals optimize on that, "
                "and the result looks entirely plausible. Next energies "
                "[cm^-1]: %s. Either raise n_states to the next real gap or confirm the set is "
                "complete.",
                self.where, self.gap_cm, self.warn_cm, self.n_states,
                ", ".join("{:.2f}".format(x) for x in self.next_cm))


def state_average_boundary(solver: FullCISolver, ints: CASIntegrals, *,
                           margin: int = BOUNDARY_MARGIN,
                           warn_cm: float = BOUNDARY_WARN_CM,
                           where: str = "converged orbitals",
                           seed_warm_start: bool = False) -> BoundaryReport:
    """Measure the state-average boundary at these integrals.

    Costs one Davidson solve at fixed orbitals, with no RDMs built and **no new residency**:
    it reuses the solver's sigma operator and workspace, and leaves ``solver.last`` and (unless
    ``seed_warm_start``) the warm start untouched, because the caller's states are the ones
    that matter.

    ⚠ The *caller* pays for the ``CASIntegrals`` as well, so from :func:`casscf` the whole
    check is an integral build plus a Davidson solve — about **one macro-iteration** of the
    optimization that just ran, not a free rider on it. That is the number to weigh when
    deciding whether to switch it off.
    """
    n_states, ndet = solver.n_states, solver.ndet
    if n_states >= ndet:
        return BoundaryReport(n_states=n_states, ndet=ndet, margin=margin, gap_cm=None,
                              warn_cm=warn_cm, where=where)
    n_extra = max(1, min(int(margin), ndet - n_states))
    energies = solver.spectrum(np.ascontiguousarray(ints.h_active_effective()),
                               ints.active_eri(), n_states + n_extra,
                               seed_warm_start=seed_warm_start)
    relative = (energies - energies[0]) * HARTREE_TO_CM
    return BoundaryReport(n_states=n_states, ndet=ndet, margin=n_extra,
                          gap_cm=float(relative[n_states] - relative[n_states - 1]),
                          next_cm=tuple(float(x) for x in relative[n_states - 1:]),
                          warn_cm=warn_cm, where=where)


def _boundary_or_warning(solver: FullCISolver, factors, h_ao: np.ndarray,
                         c_spinor: np.ndarray, spaces: OrbitalSpaces, e_nuc: float,
                         boundary_check: int, *, where: str, level: int,
                         seed_warm_start: bool = False) -> Optional[BoundaryReport]:
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
    try:
        report = state_average_boundary(
            solver, CASIntegrals.build(factors, h_ao, c_spinor, spaces, e_nuc=e_nuc),
            margin=int(boundary_check), where=where, seed_warm_start=seed_warm_start)
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
           n_elec: int, *, n_states: int = 1, e_nuc: float = 0.0,
           weights: Optional[Sequence[float]] = None,
           solver: Optional[FullCISolver] = None,
           active: Optional[ActiveSpace] = None,
           solver_options: Optional[Dict[str, Any]] = None,
           boundary_check: int = BOUNDARY_MARGIN,
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
    if optimizer_kwargs.get("report", True):
        _report_states(solver.last)

    # ⚠ **Is the state average complete?** The only evidence is a root the average did
    # not take, so this solves a few extra at the converged orbitals and discards them. Cost is
    # one integral build plus one Davidson solve — about a macro-iteration. It runs *after* the
    # states are reported so a WARNING lands next to the spectrum it is about, and it never
    # touches ``solver.last``.
    boundary = _boundary_or_warning(solver, factors, h_ao, orbital.coeff, spaces, e_nuc,
                                    boundary_check, where="converged orbitals",
                                    level=report_level)
    return CASSCFOutcome(orbital=orbital, ci=solver.last, solver=solver,
                         active=active or ActiveSpace(spaces=spaces, n_elec=n_elec),
                         boundary=boundary, boundary_initial=boundary_initial)


__all__ = ["FullCISolver", "CASCIResult", "CASSCFOutcome", "ActiveSpace", "BoundaryReport",
           "active_space", "active_space_by_character", "casci", "casscf",
           "state_average_boundary", "transition_density_numpy",
           "BOUNDARY_MARGIN", "BOUNDARY_WARN_CM", "DEFAULT_CHARACTER_THRESHOLD"]
