"""Which atoms are the magnetic centres, measured rather than assumed.

The default target set is "the open d/f shells of this molecule", and that has to be decided
from the reference itself: an element symbol says nothing about whether *this* zinc has an
open shell, and a table of blocks would answer for the element instead of for the atom.

Two conditions, and both are load-bearing
-----------------------------------------
An atom is a **centre for ``l``** when

1. some Kramers pair in the frontier window carries more than ``threshold`` of its Loewdin
   population on ``(atom, l)``, with ``l`` in ``d`` or ``f``; **and**
2. the atom's own **free-atom reference** -- the orbitals ``atomic_reference=True`` computed,
   at the reference state the atomic mean field uses -- has an **open** shell of that ``l``.

⚠ Neither condition alone is enough, and both failure modes are measured rather than
imagined, on this project's own validation systems:

* Zn(2+) puts its **filled** 3d in the frontier window and clears condition 1 at 1.000. Its
  reference is ``d10``, closed, so condition 2 is what keeps a closed shell out of an active
  space. A filled shell taken as a target produces a perfectly well-formed calculation whose
  numbers are plausible and whose physics is absent.
* CeCl3's frontier orbitals carry 0.765 of their population on the cerium **5d** and clear
  condition 1 as convincingly as the 4f does at 0.919. The Ce(3+) reference has ``d`` closed
  (20 electrons: the filled 3d/4d) and ``f`` open, so condition 2 is again what picks the
  shell the physics is in.

⚠ **The reference state is read off the free-atom solution, never parsed from its label.**
The electrons per angular-momentum channel come from the atomic orbitals' own occupations
(:func:`kuiva.mcscf.avas.reference_channel_weights`), so a configuration whose label is
provenance rather than notation ("Madelung filling") is still a number here.

Equivalent centres are pooled
-----------------------------
For two symmetric metals the canonical orbitals answer "both": the SCF returns the symmetric
and antisymmetric combinations, each about half on each centre, and neither atom clears the
threshold on its own. Measured on the Ti2Cl6 dimer: 0.456 per titanium, 0.913 pooled. Such
atoms become **one** centre over the set, whose shell is ``2l+1`` pairs *per atom*; separating
the two shells into site orbitals is a localization, later and elsewhere
(:mod:`kuiva.mcscf.localize`), and it is a rotation that changes no number.

⚠ An unrestricted reference is refused
--------------------------------------
Its spinors are orthonormal but not Kramers paired, so pair populations are not populations
of anything and the shell construction downstream (AVAS) cannot fold them. The method runs on
the restricted or ROHF reference of the high-spin state, which is the standard reference for
a CASSCF on coupled centres; a broken-symmetry guess is an SCF-level device for a different
purpose.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util.logging import get_logger
from . import targets as tg

log = get_logger(__name__)

__all__ = ["Centre", "CentreDetection", "CENTRE_ANGULAR_MOMENTA",
           "DEFAULT_CENTRE_THRESHOLD", "DEFAULT_FRONTIER_ABOVE", "DEFAULT_FRONTIER_BELOW",
           "check_configuration", "check_reference", "detect_centres", "reference_channels",
           "shell_centre"]

#: Fraction of a Kramers pair's Loewdin population that must sit on ``(atom, l)`` for the
#: atom to be a centre. The same value as the character selection's own cut
#: (:data:`kuiva.mcscf.casci.DEFAULT_CHARACTER_THRESHOLD`), deliberately: "this orbital is
#: mostly on this atom's d shell" is one statement and should not have two numbers.
DEFAULT_CENTRE_THRESHOLD = 0.5

#: Kramers pairs below and above the Fermi level that the detection looks at. Wide enough to
#: hold a whole shell either side, narrow enough that a semi-core shell of the same ``l``
#: does not enter the window.
DEFAULT_FRONTIER_BELOW = 5
DEFAULT_FRONTIER_ABOVE = 5

#: The angular momenta a centre can be a centre *for*. ⚠ Not a convenience: an open ``p``
#: shell is a main-group radical and not a magnetic centre in the sense this method is
#: built for, and it is named as a :class:`~kuiva.autocas.targets.Frontier` target instead.
CENTRE_ANGULAR_MOMENTA = (2, 3)

#: An atomic reference channel is **open** when its occupation is this far from a filled
#: shell. The atomic SCF is an average of configuration, so an open channel's occupation is
#: exactly its electron count and a closed one is exactly ``4l+2`` per shell; the tolerance
#: only absorbs the SCF's own convergence.
_CLOSED_TOL = 1e-6


@dataclass(frozen=True)
class Centre:
    """One magnetic centre: which atoms, which shell, and what the evidence was.

    ``atoms`` are 0-based atom indices -- more than one when the centres are equivalent and
    the canonical orbitals delocalize over them (see the module docstring).
    """

    atoms: Tuple[int, ...]
    l: int
    labels: Tuple[str, ...]
    population: float
    pooled: bool
    reference_state: str
    reference_channels: Tuple[float, ...]
    stated: bool
    #: Whether the reference state is a statement about **this ion** rather than the
    #: neutral-atom fallback: the user stated it, or the element is in the f block, where the
    #: per-element default is M(3+) on chemistry. ⚠ It is what decides whether the reference
    #: state may supply an electron count the orbitals cannot
    #: (:func:`kuiva.autocas.candidates.shell_candidates`) -- outside the f block the default
    #: is the neutral atom, so a transition-metal ion's d count is wrong by default there and
    #: may not be the source of anything.
    trusted_state: bool = False

    @property
    def n_pairs(self) -> int:
        """Kramers pairs the shell of this centre occupies: ``2l+1`` per atom."""
        return (2 * self.l + 1) * len(self.atoms)

    @property
    def where(self) -> str:
        return " and ".join(self.labels)

    @property
    def reference_electrons(self) -> float:
        """Electrons the free-atom reference puts in this ``l`` channel (all shells of it)."""
        if self.l >= len(self.reference_channels):
            return 0.0
        return float(self.reference_channels[self.l])

    @property
    def open_electrons(self) -> int:
        """Electrons in the **open** shell of this ``l`` in the reference state.

        The channel total minus the filled shells below it, which for a reference state that
        is what it says it is (``[Xe] 4f9``) is the shell occupation itself.
        """
        return int(round(self.reference_electrons)) % (4 * self.l + 2)

    def statement(self) -> str:
        """The physical statement of the shell, for a description string."""
        return "the valence {} shell of {}".format(tg.angular_momentum_letter(self.l),
                                                   self.where)

    def __repr__(self) -> str:
        return "Centre({}, l={}, {} pairs{})".format(
            self.where, tg.angular_momentum_letter(self.l), self.n_pairs,
            ", pooled" if self.pooled else "")


@dataclass(frozen=True)
class CentreDetection:
    """What the detection found, and the evidence it found it on."""

    centres: Tuple[Centre, ...]
    #: Frontier spinor columns the populations were measured over.
    columns: np.ndarray
    #: ``{(atom, l): (n_pair,)}`` fraction of each frontier pair's population on that channel.
    fractions: Dict[Tuple[int, int], np.ndarray]
    #: Pair labels of the frontier window, in column order.
    pair_labels: Tuple[str, ...]
    threshold: float

    def report(self, logger=None) -> None:
        logger = logger or log
        out.subsection(logger, "magnetic centres")
        if not self.centres:
            out.entry(logger, "centres detected", "none",
                      note="no frontier pair carries {:.2f} of its population on an open "
                           "d or f shell".format(self.threshold))
            return
        table = out.Table(logger, [
            out.Column("centre", "{}", 20, "<"),
            out.Column("shell", "{}", 7, "<"),
            out.Column("pairs", "{}", 7),
            out.Column("frontier pop", "{:.3f}", 14),
            out.Column("ref e", "{}", 7),
            out.Column("reference state", "{}", 24, "<"),
            out.Column("stated", "{}", 8, "<"),
        ]).start()
        for c in self.centres:
            table.row(c.where + (" (pooled)" if c.pooled else ""),
                      tg.angular_momentum_letter(c.l), c.n_pairs, c.population,
                      c.open_electrons, c.reference_state,
                      "yes" if c.stated else "default")
        table.end("population is the largest frontier Kramers pair's, threshold {:.2f}; "
                  "'ref e' is the open-shell count of the free-atom reference state, which "
                  "is a cross-check and NOT the active space's electron count"
                  .format(self.threshold))


# --- the reference view --------------------------------------------------------------------
#
# ⚠ The reference is taken **duck-typed**, not imported: `kuiva.autocas` sits above the
# multireference layer and below nothing, and importing `kuiva.interface.api` for a type hint
# would put the front end (and PySCF) behind `import kuiva.autocas`. What is needed is five
# arrays and a layout, and every consumer here says so.

def check_reference(reference) -> None:
    """The two things an automatic selection needs of a reference, refused eagerly.

    A restricted or ROHF reference (an unrestricted one's spinors are not Kramers paired) and
    the **free-atom reference orbitals** (every shell it builds is an AVAS projection onto
    them). Both refusals name the knob that fixes them.

    ⚠ It reads two attributes and computes nothing, which is what lets the stage API call it
    at construction: a missing prerequisite has to fail before the expensive call, not an
    hour into it.
    """
    data = reference.data
    if getattr(data, "unrestricted", False):
        raise ValueError(
            "automatic active-space selection needs a restricted or ROHF reference: an "
            "unrestricted one gives spinors that are orthonormal but NOT Kramers paired, so "
            "a pair population is not a population of one orbital and the shell "
            "construction (AVAS) cannot fold them onto pairs. Run the scalar SCF with "
            "reference='rohf' (the high-spin state is the standard reference for a CASSCF on "
            "coupled centres); a broken-symmetry guess is an SCF-level device for a "
            "different purpose")
    if getattr(data, "atomic_reference", None) is None:
        raise ValueError(
            "automatic active-space selection needs the free-atom reference orbitals: every "
            "shell it builds is an AVAS projection onto them, and whether an atom has an "
            "OPEN shell of a given l is read off them. Re-run the scalar SCF with "
            "atomic_reference=True (one small atomic SCF per unique element, cached per "
            "process); a character threshold is not offered as a fallback, because two "
            "constructions of 'the shell' would be two definitions of it")


def _view(reference):
    """``(layout, s_ao, coeff, occupation, energy, atomic_reference, nelec_total)``."""
    check_reference(reference)
    data = reference.data
    return (reference.ao_layout, np.asarray(data.s_ao), reference.spinors_in_ao(),
            np.asarray(reference.spinors.occ, dtype=float),
            np.asarray(reference.spinors.energy, dtype=float),
            data.atomic_reference, int(data.nelec_total))


def reference_channels(layout, atomic_reference, atom: int) -> Optional[Tuple[float, ...]]:
    """Electrons per angular-momentum channel in one atom's free-atom reference.

    ``None`` for an atom that has no free atom at all -- a **ghost**, which carries basis
    functions, no nucleus and no electrons, and is skipped by every consumer that needs an
    element rather than a label.
    """
    from ..basis.ghosts import is_ghost
    from ..mcscf.avas import reference_channel_weights

    symbol = str(layout.atom_symbols[int(atom)])
    if is_ghost(symbol):
        return None
    try:
        entry = atomic_reference.entry_for_atom(int(atom), symbol)
    except KeyError:
        return None
    idx = np.asarray(layout.atom_indices(int(atom)))
    weights = reference_channel_weights(entry, np.asarray(layout.ao_l)[idx])
    occ = np.asarray(entry.occ, dtype=float)
    return tuple(float(x) for x in (occ[:, None] * weights).sum(axis=0))


def _reference_entry(layout, atomic_reference, atom: int):
    from ..basis.ghosts import is_ghost

    symbol = str(layout.atom_symbols[int(atom)])
    if is_ghost(symbol):
        return None
    try:
        return atomic_reference.entry_for_atom(int(atom), symbol)
    except KeyError:
        return None


def _open_channels(channels: Sequence[float]) -> Tuple[int, ...]:
    """The angular momenta of :data:`CENTRE_ANGULAR_MOMENTA` that are open in ``channels``."""
    open_l = []
    for l in CENTRE_ANGULAR_MOMENTA:
        if l >= len(channels):
            continue
        n = float(channels[l])
        if n < _CLOSED_TOL:
            continue                            # empty: there is no shell to be open
        shell = 4 * l + 2
        remainder = n % shell
        if min(remainder, shell - remainder) > _CLOSED_TOL:
            open_l.append(l)                    # neither a filled shell nor a multiple of one
    return tuple(open_l)


def _pair_fractions(coeff, s_ao, layout, columns, occupation):
    """``{(atom, l): (n_pair,)}`` -- each frontier pair's population fraction per channel."""
    from ..props.population import orbital_populations

    pops = orbital_populations(coeff, s_ao, layout, columns=columns, group="kramers",
                               occupation=occupation)
    frac = pops.normalized()
    ao_atom = np.asarray(layout.ao_atom)
    ao_l = np.asarray(layout.ao_l)
    out_map: Dict[Tuple[int, int], np.ndarray] = {}
    for ia in range(layout.natm):
        on_atom = ao_atom == ia
        for l in np.unique(ao_l[on_atom]):
            mask = on_atom & (ao_l == int(l))
            out_map[(ia, int(l))] = frac[mask].sum(axis=0)
    return out_map, pops


def _pooled_fraction(fractions, atoms: Sequence[int], l: int) -> np.ndarray:
    total = None
    for ia in atoms:
        part = fractions.get((int(ia), int(l)))
        if part is None:
            continue
        total = part.copy() if total is None else total + part
    if total is None:
        raise ValueError("no atom of the set carries l = {} functions in this basis"
                         .format(l))
    return total


def detect_centres(reference, *, threshold: float = DEFAULT_CENTRE_THRESHOLD,
                   n_below: int = DEFAULT_FRONTIER_BELOW,
                   n_above: int = DEFAULT_FRONTIER_ABOVE,
                   report: bool = True) -> CentreDetection:
    """The molecule's open d/f centres, by the two conditions of the module docstring.

    Atoms of one element that each fall below the threshold are **pooled** and tested
    together; atoms that clear it individually stay separate centres, which is the right
    answer for two inequivalent metals of the same element.
    """
    from ..props.population import frontier_columns

    layout, s_ao, coeff, occ, _energy, atomic_reference, _nelec = _view(reference)
    columns = frontier_columns(occ, int(n_below), int(n_above))
    fractions, pops = _pair_fractions(coeff, s_ao, layout, columns, occ)

    channels: Dict[int, Tuple[float, ...]] = {}
    open_l: Dict[int, Tuple[int, ...]] = {}
    for ia in range(layout.natm):
        ch = reference_channels(layout, atomic_reference, ia)
        if ch is None:
            continue
        channels[ia] = ch
        open_l[ia] = _open_channels(ch)

    from ..basis.ghosts import normalize_symbol
    symbols = [normalize_symbol(str(s)) for s in layout.atom_symbols]

    individual: List[Centre] = []
    qualified: Dict[Tuple[str, int], bool] = {}
    for ia in sorted(open_l):
        for l in open_l[ia]:
            value = fractions.get((ia, l))
            if value is None or value.size == 0:
                continue
            best = float(np.max(value))
            key = (symbols[ia], l)
            if best >= float(threshold):
                individual.append(_centre(layout, atomic_reference, (ia,), l, best,
                                          pooled=False, channels=channels))
                qualified[key] = True
            else:
                qualified.setdefault(key, False)
                log.debug("atom %s carries at most %.3f of a frontier pair on l = %d, below "
                          "the centre threshold %.2f", layout.atom_label(ia), best, l,
                          threshold)

    pooled: List[Centre] = []
    for (element, l), was_qualified in sorted(qualified.items()):
        if was_qualified:
            continue
        atoms = tuple(ia for ia in sorted(open_l)
                      if symbols[ia] == element and l in open_l[ia])
        if len(atoms) < 2:
            continue
        best = float(np.max(_pooled_fraction(fractions, atoms, l)))
        if best >= float(threshold):
            pooled.append(_centre(layout, atomic_reference, atoms, l, best, pooled=True,
                                  channels=channels))
            log.debug("atoms %s pooled into one l = %d centre: %.3f together against %.2f "
                      "individually", atoms, l, best, threshold)

    centres = tuple(sorted(individual + pooled, key=lambda c: (c.atoms[0], c.l)))
    detection = CentreDetection(centres=centres, columns=np.asarray(columns),
                                fractions=fractions, pair_labels=tuple(pops.labels),
                                threshold=float(threshold))
    if report:
        detection.report()
    return detection


def _centre(layout, atomic_reference, atoms, l, population, *, pooled, channels) -> Centre:
    from ..amf.configuration import is_f_block
    from ..basis.registry import z_of

    entry = _reference_entry(layout, atomic_reference, atoms[0])
    stated = False if entry is None else not bool(entry.is_default)
    f_block = bool(is_f_block(int(z_of(str(layout.atom_symbols[int(atoms[0])])))))
    return Centre(atoms=tuple(int(a) for a in atoms), l=int(l),
                  labels=tg.atom_labels(layout, atoms), population=float(population),
                  pooled=bool(pooled),
                  reference_state="" if entry is None else str(entry.configuration),
                  reference_channels=tuple(channels[int(atoms[0])]),
                  stated=stated, trusted_state=bool(stated or f_block))


def shell_centre(reference, atoms, l=None, *,
                 threshold: float = DEFAULT_CENTRE_THRESHOLD,
                 n_below: int = DEFAULT_FRONTIER_BELOW,
                 n_above: int = DEFAULT_FRONTIER_ABOVE) -> Centre:
    """The centre a named :class:`~kuiva.autocas.targets.Shell` target resolves to.

    ``l`` omitted is read off the free-atom reference: the one channel of ``d``/``f`` that is
    **open** there. A reference with both open (a neutral lanthanide's ``4f^n 5d^1``) is
    refused rather than guessed -- which shell the calculation is about is the statement, and
    two open channels means the molecule has not said.

    A **closed** shell is refused with its populations, because a shell with no open member
    is not a target but an inactive block: the calculation it produces is well formed,
    plausible and about nothing.
    """
    layout, s_ao, coeff, occ, _energy, atomic_reference, _nelec = _view(reference)
    from ..props.population import frontier_columns

    indices = tg.resolve_atoms(layout, atoms)
    labels = tg.atom_labels(layout, indices)
    channels: Dict[int, Tuple[float, ...]] = {}
    for ia in indices:
        ch = reference_channels(layout, atomic_reference, ia)
        if ch is None:
            raise ValueError(
                "{} has no free atom: it is a ghost (basis functions, no nucleus and no "
                "electrons), so it has no shell to make active".format(layout.atom_label(ia)))
        channels[ia] = ch

    open_sets = {ia: _open_channels(channels[ia]) for ia in indices}
    if l is None:
        common = sorted(set.intersection(*[set(v) for v in open_sets.values()]))
        if not common:
            raise ValueError(
                "{} has no open d or f shell in its free-atom reference state ({}: {} "
                "electrons per channel s, p, d, f): a shell with no open member is an "
                "inactive block, not an active space. State the shell explicitly if a "
                "closed one is really meant"
                .format(" and ".join(labels),
                        _state_label(layout, atomic_reference, indices[0]),
                        [round(x, 3) for x in channels[indices[0]]]))
        if len(common) > 1:
            raise ValueError(
                "{}'s reference state has more than one open shell among d and f ({}); which "
                "one the active space is about is the statement, so name it: "
                "(\"shell\", atoms, \"{}\")"
                .format(" and ".join(labels),
                        ", ".join(tg.angular_momentum_letter(x) for x in common),
                        tg.angular_momentum_letter(common[-1])))
        ell = int(common[0])
    else:
        from ..mcscf.casci import _angular_momentum
        ell = _angular_momentum(l)
        for ia in indices:
            if ell not in open_sets[ia]:
                n = (channels[ia][ell] if ell < len(channels[ia]) else 0.0)
                raise ValueError(
                    "{} has a CLOSED {} shell in its free-atom reference state ({}: {:.0f} "
                    "electrons in the {} channel, a filled {}). A filled shell taken as a "
                    "target gives a well-formed calculation with plausible numbers and no "
                    "physics in it -- the term structure and the Lande g of a shell do not "
                    "say which shell it is"
                    .format(layout.atom_label(ia), tg.angular_momentum_letter(ell),
                            _state_label(layout, atomic_reference, ia), n,
                            tg.angular_momentum_letter(ell),
                            tg.angular_momentum_letter(ell)))

    columns = frontier_columns(occ, int(n_below), int(n_above))
    fractions, _pops = _pair_fractions(coeff, s_ao, layout, columns, occ)
    population = float(np.max(_pooled_fraction(fractions, indices, ell)))
    pooled = len(indices) > 1
    if population < float(threshold):
        log.warning("no frontier Kramers pair carries more than %.3f of its population on "
                    "%s l = %d, against a centre threshold of %.2f: the shell was asked for "
                    "by name and is taken, but these orbitals are not where its character "
                    "is. Check the reference state and the geometry before reading the "
                    "spectrum", population, " and ".join(labels), ell, threshold)
    return _centre(layout, atomic_reference, indices, ell, population, pooled=pooled,
                   channels=channels)


def _state_label(layout, atomic_reference, atom: int) -> str:
    entry = _reference_entry(layout, atomic_reference, atom)
    return "?" if entry is None else str(entry.configuration)


def check_configuration(centre: Centre, measured_electrons: float, *,
                        tolerance: float = 1.0) -> Optional[str]:
    """Cross-check a shell's measured electron count against the stated reference state.

    The count that *defines* the active space is the measured one -- the reference occupation
    of the orbitals the shell construction selected -- for a reason that is not a preference:
    the per-atom ``configuration=`` statement defaults to the **neutral atom** outside the f
    block, so a transition-metal ion's d count is wrong by default there and cannot be the
    source of anything. What the statement is good for is *disagreeing*: a stated Ti(3+) whose
    shell measures two electrons is either the wrong oxidation state or the wrong orbitals,
    and either way the user wants to know.

    Returns the warning text (already logged) or ``None``. ⚠ Nothing is checked where no
    configuration was stated: the default is not evidence, and "the neutral atom disagrees
    with the ion" is not information.
    """
    if not centre.stated:
        return None
    expected = centre.open_electrons
    if abs(float(measured_electrons) - expected) <= float(tolerance):
        return None
    text = ("the {} shell of {} holds {:.1f} electrons in this reference, but the stated "
            "reference state {} has {} in its open {} shell. One of the two is not the "
            "calculation that was meant: check the oxidation state against the geometry, and "
            "the selected orbitals against the shell they were supposed to be"
            .format(tg.angular_momentum_letter(centre.l), centre.where,
                    float(measured_electrons), centre.reference_state, expected,
                    tg.angular_momentum_letter(centre.l)))
    log.warning("%s", text)
    return text
