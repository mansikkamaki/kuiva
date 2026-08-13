"""The atomic reference configuration.

An X2CAMF correction is the picture change of a **mean field**, and a mean field is taken over
some state. For a closed-shell atom there is only one, which is why the reference
configuration was at first no more than a free-form provenance string. For the target ions this project
actually targets — Ti(III) d1, Ce(III) 4f1, Dy(III) 4f9, Yb(III) 4f13, Bi 6p3 — there is a
choice, it changes the answer, and it has to be part of the cache key.

Why electrons-per-angular-momentum is the canonical form
--------------------------------------------------------
This module represents a configuration as **the number of electrons in each angular-momentum
channel** — Bi as ``(12, 27, 30, 14)``, i.e. 12 s, 27 p, 30 d, 14 f. That is not a
simplification of the spectroscopic notation for convenience; it is exactly the information
the average-of-configuration occupation needs and nothing more:

* within one ``l`` channel the shells fill in energy order, so ``N_l`` alone fixes how many
  spinors of that ``l`` are fully occupied (``N_l // (4l+2)`` shells) and how many electrons
  are left to spread over the frontier one (``N_l mod (4l+2)``);
* the fractional occupation that follows is spherical whichever ``j`` sub-shell it lands on,
  which is what makes the density-anisotropy guard of :mod:`kuiva.amf.pyscf_dhf` become an
  *assertion that averaging worked* rather than a closed-shell restriction.

It is also **canonical and hashable**, which the cache key of :mod:`kuiva.amf.atomic` requires:
two ways of writing
the same configuration produce the same tuple, so they share a cache entry, while two genuinely
different references cannot alias. A free-form string could do neither.

⚠ The one thing this form cannot express, stated rather than discovered
------------------------------------------------------------------------
A configuration with a **hole below an occupied shell of the same ``l``** — say a 3d hole with
4d occupied — is not representable: the per-``l`` count would put the electrons in the lower
shells. Every ground-state atom and ion is aufbau-ordered within each ``l`` channel, so this
costs nothing for the intended use, but a core-hole reference state is out of reach and must be
refused rather than silently re-ordered. :func:`parse` does not check it (it cannot — the
information is gone by then), so a caller building a core-excited reference has to know.

Ionic configurations are **stated, never guessed**
---------------------------------------------------
The neutral ground configuration comes from PySCF's aufbau table and needs no judgement. A
*general* ion's does: which electrons a cation loses is chemistry rather than arithmetic — the
per-``l`` totals carry no principal quantum number to remove them by — so Ti(3+) is [Ar]3d1
and not [Ar]4s1. Deriving one in general would be the same class of error as
the ``"6p"`` AO-label incident: plausible, silent, and element-dependent.

The default is therefore **the neutral atom** — the atomic mean field is a property of an
element and a molecular charge belongs to no single atom of it — **with one deliberate
exception, decided on chemistry** (see :func:`default_configuration`).

⚠ **The f elements default to the trivalent ion, and this is a recorded user decision.** Lanthanides are almost always encountered as Ln(3+), whose configuration is
unambiguously ``[Xe]4f^(Z-57)`` — no judgement needed, unlike a d-block ion. The neutral atom
is the *wrong* reference for them twice over:

* it is a state molecular chemistry essentially never sees; and
* **low-valent lanthanides do not keep their atomic configuration anyway.** The known Dy(II)
  complexes are 4f9(6s/5d)1, not 4f10 — so neutral Dy's 4f10 describes nothing that occurs in
  a molecule, while Dy(3+) 4f9 is what almost every target system of this project actually is.

It has a numerical benefit too, though **not the one it might appear to have**: Ln(3+) has a
single open ``f`` shell (or none, for La and Lu), whereas several neutral lanthanides have two
open shells at once — neutral Ce is [Xe]4f1 5d1 6s2, open in both ``d`` and ``f``. Averaging
over one open shell is a better-defined approximation than averaging over two. ⚠ It is **not**
a fix for the time-reversal problem seen in the lanthanide corrections: the occupations a
two-open-shell configuration produces were measured and are clean (neutral Ce: density
time-reversal odd at 1.7e-10 relative, anisotropy 1.5e-13).

Whichever reference is used, **the number of electrons in the configuration fixes the charge
state that is actually solved** — the configuration is the single source of truth, so the two
can never disagree. How much the choice matters is a measurement, not a judgement: for Ti(3+)
it moves a splitting by 0.21%.

References
----------
* Average-of-configuration Dirac-Hartree-Fock: I. P. Grant, "Relativistic Quantum Theory of
  Atoms and Molecules", Springer (2007), ch. 6-7; J. P. Desclaux, Comput. Phys. Commun. 9, 31
  (1975), doi:10.1016/0010-4655(75)90054-5 — the multiconfiguration average from which the
  fractional-occupation scheme descends.
* Its use as the atomic reference for a mean-field spin-orbit operator: B. A. Hess,
  C. M. Marian, U. Wahlgren, O. Gropen, Chem. Phys. Lett. 251, 365 (1996),
  doi:10.1016/0009-2614(96)00119-4; J. Liu, L. Cheng, J. Chem. Phys. 148, 144108 (2018),
  doi:10.1063/1.5023750.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from typing import Dict, Optional, Sequence, Tuple, Union

#: Angular-momentum letters, indexed by ``l``. Extended past ``l = 3`` because
#: ``cc-pVnZ-X2C`` carries ``g`` functions and a future reference state could occupy one.
SHELL_LETTERS = "spdfghi"

#: Atomic numbers of the noble gases, so ``[Xe]4f1`` can be written the way it is spoken.
NOBLE_GASES = {"He": 2, "Ne": 10, "Ar": 18, "Kr": 36, "Xe": 54, "Rn": 86}

#: ``Z`` ranges of the f blocks, and the noble-gas core each trivalent ion reduces to. The
#: trivalent configuration is then ``core + f^(Z - Z_first)`` with no judgement required, which
#: is exactly what makes it safe to derive when a general ion's is not (module docstring).
F_BLOCKS = ((57, 71, "Xe", 4), (89, 103, "Rn", 5))       # (Z_first, Z_last, core, n of nf)

_TERM = re.compile(r"^(\d+)([{}])(\d+)$".format(SHELL_LETTERS))
_CORE = re.compile(r"^\[([A-Za-z]+)\]")
#: ``"+3"``, ``"3+"``, ``"0"`` — an oxidation state rather than a configuration.
_OXIDATION = re.compile(r"^(?:\+(\d+)|(\d+)\+|([+-]?\d+))$")


def _shells(occupations: Sequence[int]) -> Dict[Tuple[int, int], int]:
    """``{(n, l): electrons}`` reconstructed from per-``l`` totals.

    The per-``l`` representation carries no principal quantum number, but one can be *derived*
    where the channel is aufbau-filled: the ``k``-th shell of angular momentum ``l`` is
    ``n = l + k``, so 1s, 2s, 3s... for ``l = 0`` and 4f, 5f... for ``l = 3``. That is enough
    to order shells by ``n``, which is what removing electrons from an atom needs.

    ⚠ Only valid under the same aufbau assumption the class docstring states: no hole below an
    occupied shell of the same ``l``.
    """
    out: Dict[Tuple[int, int], int] = {}
    for l, total in enumerate(occupations):
        degeneracy = 4 * l + 2
        k, left = 1, total
        while left > 0:
            out[(l + k, l)] = min(left, degeneracy)
            left -= min(left, degeneracy)
            k += 1
    return out


def parse_shell_terms(text: str) -> Tuple[Tuple[int, int, int], ...]:
    """``"[Xe] 4f9 5d1 6s1"`` -> ``((1, 0, 2), ..., (4, 3, 9), (5, 2, 1), (6, 0, 1))``.

    **The one grammar for a spectroscopic configuration string in this project.** It is
    shared deliberately: :meth:`AtomicConfiguration.parse` throws the principal quantum
    numbers away afterwards (its per-``l`` form is all an atomic mean field needs) while
    :class:`kuiva.extras.shells.ShellConfiguration` keeps them, and a second grammar for the
    same notation would be two spellings of ``"[Xe]4f9"`` that could disagree about what the
    user wrote. Terms are returned shell-resolved and **unvalidated beyond the grammar** —
    which occupations are physically admissible is the consuming class's rule, not the
    reader's.

    Accepts an optional leading noble-gas core, then ``nl^q`` terms separated by spaces or
    commas, in any order. Repeated shells are summed; a term with ``q = 0`` is kept, since it
    is how a caller writes "this shell is empty" (``"[Xe]4f0"``) and dropping it would be a
    silent edit of a provenance string.

    ⚠ **The principal quantum numbers of a noble-gas core are derived, and that is exact
    here for the reason it is not in general** (:func:`_shells`): a noble gas is aufbau-filled
    with no hole in any channel, so ``n = l + k`` for the ``k``-th shell of angular momentum
    ``l`` reproduces 1s 2s 2p 3s 3p 3d 4s ... exactly. The derivation would *not* be safe for
    an arbitrary per-``l`` count, which is why it is confined to the core here.

    Raises
    ------
    ValueError
        on an unreadable term, a core that is not a noble gas, an empty string, or ``n <= l``
        (there is no ``1p`` shell, so such a term can only be a typo).
    """
    text = text.strip()
    shells: Dict[Tuple[int, int], int] = {}
    core = _CORE.match(text)
    if core:
        symbol = core.group(1).capitalize()
        if symbol not in NOBLE_GASES:
            raise ValueError(
                "{!r} is not a noble gas, so [{}] is not a closed core. Known cores: "
                "{}.".format(core.group(1), core.group(1), ", ".join(sorted(NOBLE_GASES))))
        for key, q in sorted(_shells(_neutral_occupations(NOBLE_GASES[symbol])).items()):
            shells[key] = q
        text = text[core.end():]
    terms = text.replace(",", " ").split()
    if not terms and not core:
        raise ValueError("empty configuration {!r}".format(text))
    for term in terms:
        m = _TERM.match(term)
        if not m:
            raise ValueError(
                "cannot read {!r} as a shell occupation. Expected `nl^q` with the "
                "principal quantum number, e.g. '4f9' or '3d1', optionally after a "
                "noble-gas core such as '[Xe]'.".format(term))
        n, l, q = int(m.group(1)), SHELL_LETTERS.index(m.group(2)), int(m.group(3))
        if n <= l:
            raise ValueError(
                "there is no {}{} shell: the lowest shell of angular momentum {} is "
                "{}{}.".format(n, SHELL_LETTERS[l], SHELL_LETTERS[l], l + 1,
                               SHELL_LETTERS[l]))
        shells[(n, l)] = shells.get((n, l), 0) + q
    return tuple((n, l, q) for (n, l), q in sorted(shells.items()))


def _neutral_occupations(z: int) -> Tuple[int, ...]:
    """Electrons per ``l`` channel for the neutral ground configuration of element ``z``.

    Taken from ``pyscf.data.elements.CONFIGURATION``, which is already exactly this
    representation (``[n_s, n_p, n_d, n_f]``) — one of the rare cases where the convention a
    dependency happens to use is the one that was independently arrived at, so it is used
    directly rather than converted through spectroscopic notation and back.
    """
    from pyscf.data import elements

    if not 1 <= z < len(elements.CONFIGURATION):
        raise ValueError("no tabulated ground configuration for Z = {}".format(z))
    return tuple(int(n) for n in elements.CONFIGURATION[z])


@dataclass(frozen=True)
class AtomicConfiguration:
    """Electrons per angular-momentum channel — canonical, hashable, and cache-key safe.

    ``occupations[l]`` is the total number of electrons in all shells of angular momentum
    ``l``. Trailing zeros are stripped on construction so that ``(4, 6)`` and ``(4, 6, 0, 0)``
    are the *same* configuration and cannot occupy two cache entries.

    ``label`` is provenance and is deliberately **excluded from equality and hashing**: two
    descriptions of the same electron distribution are the same reference state, whatever
    either of them was called, and treating them as different would defeat the caching that
    makes an atomic mean field affordable at all.
    """

    occupations: Tuple[int, ...]

    def __init__(self, occupations: Sequence[int], label: str = "") -> None:
        occ = [int(n) for n in occupations]
        if any(n < 0 for n in occ):
            raise ValueError("negative occupation in {}".format(occupations))
        for l, n in enumerate(occ):
            if n > 2 * (2 * l + 1) * 20:                # 20 shells of one l is unphysical
                raise ValueError("implausible {} occupation {}".format(SHELL_LETTERS[l], n))
        while occ and occ[-1] == 0:
            occ.pop()
        object.__setattr__(self, "occupations", tuple(occ))
        object.__setattr__(self, "_label", label)

    # -- basic properties ------------------------------------------------------------------

    @property
    def n_electrons(self) -> int:
        return int(sum(self.occupations))

    @property
    def label(self) -> str:
        """Human-readable provenance. Not part of the identity — see the class docstring."""
        return getattr(self, "_label", "") or self.canonical

    @property
    def canonical(self) -> str:
        """``"s12 p27 d30 f14"`` — the identity, written out. Two configurations share a
        cache entry if and only if this string is the same."""
        return " ".join("{}{}".format(SHELL_LETTERS[l], n)
                        for l, n in enumerate(self.occupations) if n)

    def open_shells(self) -> Tuple[Tuple[int, int], ...]:
        """``(l, n_open_electrons)`` for every partially filled channel.

        Empty for a closed shell, which is what makes ``is_closed_shell`` a statement about
        the configuration rather than about the converged density (the two are checked
        independently — see :func:`kuiva.amf.pyscf_dhf.density_anisotropy`, which catches the
        case where a *correct* configuration was solved *incorrectly*).
        """
        return tuple((l, n % (4 * l + 2))
                     for l, n in enumerate(self.occupations) if n % (4 * l + 2))

    @property
    def is_closed_shell(self) -> bool:
        return not self.open_shells()

    def shells(self) -> Tuple[Tuple[int, int, int], ...]:
        """``((n, l, q), ...)`` — the per-``l`` counts resolved back into shells.

        ⚠ **A derivation under the aufbau assumption, not stored information** (:func:`_shells`
        and the module docstring): the ``n`` were discarded at construction, and putting them
        back presumes no hole below an occupied shell of the same ``l``. For a configuration
        that *was* built from shells this inverts :meth:`parse` exactly, which is what
        :class:`kuiva.extras.shells.ShellConfiguration` checks itself against to know that its
        per-``l`` form loses nothing.
        """
        return tuple((n, l, q) for (n, l), q in sorted(_shells(self.occupations).items()))

    def spinors_needed(self, l: int) -> int:
        """Spinors of angular momentum ``l`` the occupation requires the basis to supply.

        A partially filled channel needs the *whole* frontier shell present, not just the
        occupied part of it, because the electrons are spread over all ``4l+2`` of its
        spinors. A basis too small to hold it gives a wrong answer rather than a small one.
        """
        if l >= len(self.occupations):
            return 0
        degeneracy = 4 * l + 2
        full, remainder = divmod(self.occupations[l], degeneracy)
        return full * degeneracy + (degeneracy if remainder else 0)

    # -- construction ----------------------------------------------------------------------

    @classmethod
    def ground(cls, element: Union[str, int]) -> "AtomicConfiguration":
        """The **neutral** ground configuration of an element, from the aufbau table.

        The only ionic counterpart is :meth:`trivalent`, and only for the f block, where the
        ion is a closed noble-gas core plus an ``f`` shell and nothing has to be decided. For
        anything else, which electrons a cation loses is chemistry rather than arithmetic —
        Ti(3+) is [Ar]3d1 and not [Ar]4s1 — and the per-``l`` totals carry no principal
        quantum number to remove them by, so the configuration is stated, never derived.
        """
        from pyscf import gto

        z = int(gto.charge(element)) if isinstance(element, str) else int(element)
        symbol = element if isinstance(element, str) else str(element)
        return cls(_neutral_occupations(z), label="{} neutral ground".format(symbol))

    @classmethod
    def parse(cls, text: str) -> "AtomicConfiguration":
        """``"[Xe]4f1"``, ``"[Ar]3d1"``, ``"1s2 2s2 2p6 3s2 3p6 3d1"`` — all of these.

        A noble-gas core expands to that element's neutral configuration; the remaining terms
        are ``nl^q`` with the principal quantum number **required**. The ``n`` is then
        discarded, because the occupation only depends on the per-``l`` total (class
        docstring) — but writing ``f9`` instead of ``4f9`` is refused anyway, since a
        configuration a reader cannot check is not provenance.

        The reading itself is :func:`parse_shell_terms`, shared with the shell-resolved
        configuration of :mod:`kuiva.extras.shells`, which keeps the ``n`` this discards.
        """
        occ = []
        for _, l, q in parse_shell_terms(text):
            while len(occ) <= l:
                occ.append(0)
            occ[l] += q
        return cls(occ, label=_CORE.sub("", text.strip(), count=1).strip() or "closed core")

    @classmethod
    def trivalent(cls, element: Union[str, int]) -> "AtomicConfiguration":
        """The ``M(3+)`` configuration of an f-block element: ``[core] nf^(Z - Z_first)``.

        Defined **only** for the lanthanides and actinides, and that restriction is the point.
        Their trivalent configuration needs no judgement — Ce(3+) is [Xe]4f1, Dy(3+) is
        [Xe]4f9, Lu(3+) is [Xe]4f14 — because the ion is a closed noble-gas core plus an
        ``f`` shell, with nothing to decide about which electrons went. A d-block ion is not
        like that (Ti(3+) is [Ar]3d1, not [Ar]4s1), so this deliberately refuses to guess one.
        """
        from pyscf import gto

        z = int(gto.charge(element)) if isinstance(element, str) else int(element)
        symbol = element if isinstance(element, str) else str(element)
        for first, last, core, shell in F_BLOCKS:
            if first <= z <= last:
                occ = list(_neutral_occupations(NOBLE_GASES[core]))
                while len(occ) <= 3:
                    occ.append(0)
                occ[3] += z - first
                return cls(occ, label="{}(3+) [{}]{}f{}".format(symbol, core, shell,
                                                                z - first))
        raise ValueError(
            "no trivalent configuration is derived for {} (Z = {}): it is defined only for "
            "the f blocks, where the ion is a noble-gas core plus an f shell and nothing has "
            "to be decided. For a d-block or main-group ion, state the configuration."
            .format(symbol, z))

    @classmethod
    def for_oxidation_state(cls, element: Union[str, int],
                            oxidation_state: int) -> "AtomicConfiguration":
        """The configuration of ``element`` in a given oxidation state.

        **f block:** ``[core] nf^(Z - Z_core - q)`` exactly — the ion is a closed noble-gas
        core plus an ``f`` shell whatever ``q`` is, so nothing has to be decided. This covers
        the cases that motivated the feature: Th(IV) is [Rn]5f0, U(VI) is [Rn]5f0, Pu(IV) is
        [Rn]5f4, No(II) is [Rn]5f14, Eu(II) is [Xe]4f7, Ce(IV) is [Xe]4f0.

        **Everything else:** electrons are removed from the occupied shell of highest ``n``,
        ties broken by highest ``l``. That is the standard rule and it is reliable off the f
        block — Ti(III) [Ar]3d1, Fe(III) [Ar]3d5, Cu(II) [Ar]3d9, Bi(III) 6s2 (the inert pair),
        Pb(II) 6s2. ⚠ It is **not** reliable *on* the f block, which is why that case never
        reaches it: applied to Yb(3+) it would strip 5p before 4f and return [Xe]4f14 5s2 5p5
        instead of [Xe]4f13, because ordering by ``n`` is not ordering by orbital energy once
        4f drops below 5p.

        ⚠ **A derived low-valent f-block configuration is the formal ``f^n`` one, not
        necessarily the true ground state.** The known Dy(II) complexes are 4f9(6s/5d)1 rather
        than 4f10, and La(2+) is [Xe]5d1 rather than [Xe]4f1. For an atomic *mean field* that
        distinction is worth parts per million (the measured reference sensitivity is 13 ppm
        for Lu, 0.21% for Ti(3+)), so the formal configuration is a sound reference — but it is
        a reference, not a claim about the ion's spectroscopic ground state, and an explicit
        configuration string overrides it.
        """
        from pyscf import gto

        z = int(gto.charge(element)) if isinstance(element, str) else int(element)
        symbol = element if isinstance(element, str) else str(element)
        q = int(oxidation_state)
        if q >= z:
            raise ValueError("{}{:+d} would have {} electrons".format(symbol, q, z - q))

        for first, last, core, shell in F_BLOCKS:
            if first <= z <= last:
                n_f = z - NOBLE_GASES[core] - q
                if not 0 <= n_f <= 14:
                    raise ValueError(
                        "{}({:+d}) implies {} f electrons, which is outside 0-14. The f-block "
                        "rule [{}]{}f^(Z - Z_core - q) does not describe this state; give the "
                        "configuration explicitly.".format(symbol, q, n_f, core, shell))
                occ = list(_neutral_occupations(NOBLE_GASES[core]))
                while len(occ) <= 3:
                    occ.append(0)
                occ[3] += n_f
                return cls(occ, label="{}({:+d}) [{}]{}f{}".format(symbol, q, core, shell,
                                                                   n_f))

        shells = _shells(_neutral_occupations(z))
        for _ in range(q):
            # highest n, then highest l — the outermost electron the atom actually loses
            n, l = max(k for k, v in shells.items() if v > 0)
            shells[(n, l)] -= 1
        occ = [0] * (max(l for _, l in shells) + 1)
        for (_, l), v in shells.items():
            occ[l] += v
        return cls(occ, label="{}({:+d})".format(symbol, q))

    @classmethod
    def coerce(cls, value: Union[None, str, "AtomicConfiguration", Sequence[int]],
               element: Optional[str] = None) -> "AtomicConfiguration":
        """Accept whatever a caller has and return the canonical object.

        Accepts, in order: the object itself; ``None`` for :func:`default_configuration` — the
        neutral atom for most elements, the **trivalent ion** for the f blocks; an **oxidation
        state**, as an ``int`` or a string like ``"+2"`` / ``"3+"``, which is what a user
        reaches for when the element's oxidation state is the thing they know
        (:meth:`for_oxidation_state`); a configuration string; or per-``l`` counts.

        The oxidation-state form is why choosing a state needs no new parameter anywhere:
        ``amf_correction(mol, configuration="+4")`` is a Th(IV) reference, and it flows through
        the cache key and the provenance record like any other configuration.
        """
        if isinstance(value, AtomicConfiguration):
            return value
        if value is None:
            if element is None:
                raise ValueError("a default configuration needs the element")
            return default_configuration(element)
        if isinstance(value, bool):
            raise TypeError("a bool is not a configuration")
        if isinstance(value, int):
            if element is None:
                raise ValueError("an oxidation state needs the element")
            return cls.for_oxidation_state(element, value)
        if isinstance(value, str):
            match = _OXIDATION.match(value.strip())
            if match:
                if element is None:
                    raise ValueError("an oxidation state needs the element")
                digits = next(g for g in match.groups() if g is not None)
                return cls.for_oxidation_state(element, int(digits))
            return cls.parse(value)
        return cls(value)

    # -- reporting -------------------------------------------------------------------------

    def as_dict(self) -> Dict[str, int]:
        """``{"s": 12, "p": 27, ...}`` — for reference records and JSON provenance."""
        return {SHELL_LETTERS[l]: n for l, n in enumerate(self.occupations) if n}

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        return "AtomicConfiguration({}, {} electrons{})".format(
            self.canonical, self.n_electrons,
            "" if self.is_closed_shell else ", open: " + ", ".join(
                "{}^{}".format(SHELL_LETTERS[l], q) for l, q in self.open_shells()))


def is_f_block(element: Union[str, int]) -> bool:
    """True for La-Lu and Ac-Lr."""
    from pyscf import gto

    z = int(gto.charge(element)) if isinstance(element, str) else int(element)
    return any(first <= z <= last for first, last, _, _ in F_BLOCKS)


def default_configuration(element: Union[str, int]) -> AtomicConfiguration:
    """The atomic reference Kuiva uses when the caller names none.

    * **f block (La-Lu, Ac-Lr): the trivalent ion.** A user decision, on chemistry rather than numerics. Lanthanides are almost always encountered as
      Ln(3+); the neutral atom is a state molecular chemistry essentially never sees, and
      **low-valent lanthanides do not retain their atomic configuration anyway** — the known
      Dy(II) complexes are 4f9(6s/5d)1, not the 4f10 of the neutral atom. So neutral Dy's
      reference describes nothing that occurs in a molecule, while Dy(3+) 4f9 is what nearly
      every target system of this project actually is. It also avoids the two-open-shell
      references several neutral lanthanides have (neutral Ce is open in both ``d`` and ``f``).
    * **everything else: the neutral atom**, because there is no comparable single answer. A
      d-block metal's oxidation state is genuinely variable — Ti(III), Fe(II)/Fe(III),
      Cu(II) — and no default would be right more often than neutral. It matters little in any
      case: the measured neutral-vs-ionic sensitivity for Ti(3+) is **0.21%** on a splitting
      (``tests/reference/amf_sensitivity.json``), because the picture change is a core effect
      and valence electrons barely touch the core.

    ⚠ **+3 is not universal even across the f block, and is a default rather than a claim.**
    Thorium is usually tetravalent, uranium and plutonium high-valent, and the late actinides
    such as nobelium more stable divalent — so An(III) is not "the" actinide oxidation state in
    the way Ln(III) nearly is for lanthanides. It is kept as the default because a default has
    to be *one* thing, +3 is the most defensible single choice across both f blocks, and the
    reference is worth parts per million to the correction. What matters is that it is
    **overridable per element**: :meth:`AtomicConfiguration.for_oxidation_state`, reachable as
    ``configuration="+4"`` for Th(IV), ``"+6"`` for U(VI), ``"+2"`` for No(II).

    Any of it is overridden by an oxidation state or by naming a configuration outright, which
    remains the only way to get one this function will not derive.
    """
    if is_f_block(element):
        return AtomicConfiguration.trivalent(element)
    return AtomicConfiguration.ground(element)


# --- Sphericity: the constraint that DEFINES an average of configuration --------------------

def angular_channel_groups(ao_l, ao_m, ao_shell) -> Dict[int, np.ndarray]:
    """Group the real-harmonic AO functions of **one atom** by angular momentum.

    Returns ``{l: (n_radial, 2l+1) index array}``: entry ``[r, i]`` is the
    AO index of the ``r``-th radial function in the ``m = i - l`` channel, so a row is one
    radial function across all its ``m`` and a column is one ``m`` channel across all radial
    functions. The columns are ordered by **ascending m**, which is not the integral library's
    within-shell order (a ``p`` shell is stored ``px, py, pz`` = ``m = +1, -1, 0``); ordering
    them here is what lets everything downstream ignore that.

    Every ``m`` channel of one ``l`` must offer the same radial functions in the same order,
    which a spherical basis does by construction — a shortfall is refused rather than padded,
    because a channel with a function missing would make an atom's operator anisotropic for a
    reason that has nothing to do with physics.
    """
    ao_l = np.asarray(ao_l, dtype=int)
    ao_m = np.asarray(ao_m, dtype=int)
    ao_shell = np.asarray(ao_shell, dtype=int)
    groups: Dict[int, np.ndarray] = {}
    for l in sorted(set(ao_l.tolist())):
        columns = []
        for m in range(-l, l + 1):
            index = np.where((ao_l == l) & (ao_m == m))[0]
            columns.append(index[np.argsort(ao_shell[index], kind="stable")])
        sizes = sorted({int(c.size) for c in columns})
        if len(sizes) != 1 or sizes[0] == 0:
            raise ValueError(
                "the {} channel offers {} functions across its m values; a spherical basis "
                "offers the same radial functions in every m channel".format(
                    SHELL_LETTERS[l] if l < len(SHELL_LETTERS) else "l={}".format(l),
                    sorted(int(c.size) for c in columns)))
        groups[l] = np.stack(columns, axis=1)
    return groups


def spherical_projector(groups, dimension: int, *, blocks: int = 1):
    """Return ``P(A)``: the projection of an **atomic** matrix onto its spherical part.

    A rank-zero (scalar) operator on a spherically symmetric atom is diagonal in the angular
    label and **independent of the magnetic component** — that is the Wigner-Eckart theorem
    and it is exact, not an approximation. So the projection is: average each block over the
    magnetic components of its multiplet, and zero everything else. ``groups`` is any iterable
    of ``(n_radial, multiplicity)`` index arrays, one per symmetry class, as
    :func:`angular_channel_groups` builds for a real-harmonic scalar basis (pass its
    ``.values()``) and :func:`kuiva.amf.pyscf_dhf.spinor_symmetry_groups` for the ``(l, j)``
    classes of a j-adapted spinor basis.

    ``blocks`` divides the matrix into that many equal super-blocks whose functions carry the
    *same* labels — ``2`` for a four-component matrix ``[[LL, LS], [SL, SS]]``, whose small
    component is kinetically balanced and therefore shares the large component's ``(l, j,
    m_j)`` labels. Every super-block, on and off the diagonal, is projected.

    ⚠ **This is a constraint on the state, not a convergence aid, and the difference is
    load-bearing.** The spherical solution is an **unstable** fixed point of an
    average-of-configuration SCF: measured on Ti(+1) ``s7 p12 d2``, the quadrupole anisotropy
    of the density grows by about one order of magnitude per cycle from roundoff — 2e-12 to
    4e-4 over nine cycles — while the total energy falls by 1.2e-5 Eh. A fractionally occupied
    Hartree-Fock functional has broken-symmetry solutions *below* the spherical one, so the
    iteration slides into one whenever numerical noise gives it a direction, and damping,
    level shifts or a tighter threshold cannot recover what the functional does not want. The
    spherical ensemble is the state the average of configuration *names*, so it is imposed.

    ⚠ At a spherical fixed point the projection is the **identity**, which is why applying it
    does not move a solution that was already clean: it changes the trajectory, not the answer.

    The index arrays must be an **exact cover** of one super-block. A function left out of
    every group would have its rows and columns zeroed — a basis function silently deleted
    from the calculation — so the cover is checked here rather than trusted.
    """
    if dimension % max(int(blocks), 1):
        raise ValueError("a {}-dimensional matrix does not divide into {} equal blocks"
                         .format(dimension, blocks))
    block = int(dimension) // int(blocks)
    groups = [np.asarray(g, dtype=int) for g in groups]
    covered = (np.concatenate([g.ravel() for g in groups])
               if groups else np.zeros(0, dtype=int))
    if covered.size != block or sorted(covered.tolist()) != list(range(block)):
        raise ValueError(
            "the symmetry classes cover {} of the {} functions of a block, and not exactly "
            "once each; a function in no class would be projected away entirely".format(
                covered.size, block))

    offsets = tuple(range(0, int(dimension), block))
    plans = []
    for index in groups:
        for a in offsets:
            for b in offsets:
                plans.append(((index + a).T[:, :, None], (index + b).T[:, None, :]))

    def project(a):
        """Project one matrix. Allocates its own output; the input is not touched."""
        a = np.asarray(a)
        out = np.zeros_like(a)
        for rows, columns in plans:
            # (multiplicity, n_radial, n_radial) gathered, averaged over m, written back to
            # every m: one fancy-index pair per class and super-block, nothing per element.
            out[rows, columns] = a[rows, columns].mean(axis=0)
        return out

    return project


def average_occupations(configuration: AtomicConfiguration, energies: Sequence[float],
                        angular_momenta: Sequence[int], spatial: bool = False):
    """Average-of-configuration occupations for a set of orbitals, as a plain array.

    This is the whole of the AOC rule, isolated from any SCF: given each orbital's energy and
    its angular momentum, fill each ``l`` channel in energy order — ``N_l // (4l+2)`` shells
    fully, then ``q = N_l mod (4l+2)`` electrons spread **equally over all ``4l+2`` spinors**
    of the frontier shell.

    ``spatial=True`` states the same rule for a **spin-restricted scalar** orbital set, where
    one orbital of angular momentum ``l`` holds two electrons and a shell is ``2l+1`` orbitals
    rather than ``4l+2`` spinors: a full shell is occupation 2.0 and the frontier one carries
    ``q / (2l+1)`` on each of its orbitals. It is the *same state* — the same ``q`` electrons
    spread equally over the same shell, so the electron count, the radial density and the
    open-shell coupling coefficient of :attr:`OpenShell.coupling` (which counts **electrons in
    spinors**, ``n = 4l+2``, whichever basis the orbitals are written in) are unchanged. Only
    the container differs. Generalizing the rule in place rather than beside it is what keeps
    the spinor and scalar average-of-configuration SCFs occupying one configuration one way.

    It is shared deliberately. The four-component backend
    (:mod:`kuiva.amf.pyscf_dhf`) uses it to occupy Dirac spinors, and the two-component
    validation SCF that checks the resulting correction against four-component theory must
    occupy *the same configuration the same way* or the comparison is between two different
    states. Having one implementation of the rule is what makes that true by construction
    rather than by two functions agreeing; it also makes the rule testable with no integrals
    at all.

    Parameters
    ----------
    configuration : AtomicConfiguration
    energies : sequence of float
        Orbital energies. Only their **order within an ``l`` channel** is used.
    angular_momenta : sequence of int
        The ``l`` each orbital belongs to, same length as ``energies``. Orbitals of an ``l``
        the configuration does not occupy are simply left empty, so the caller may pass the
        whole electronic branch without filtering it.
    spatial : bool
        ``False`` (the default) for spinors — ``4l+2`` per shell, a filled one at occupation
        1. ``True`` for spin-restricted scalar orbitals — ``2l+1`` per shell, a filled one at
        occupation 2.

    Returns
    -------
    ndarray of occupations, summing to ``configuration.n_electrons`` either way.
    """
    import numpy as np

    e = np.asarray(energies, dtype=float)
    l_of = np.asarray(angular_momenta, dtype=int)
    if e.shape != l_of.shape:
        raise ValueError("got {} energies and {} angular momenta".format(e.size, l_of.size))
    occ = np.zeros(e.size)
    for l, n_electrons in enumerate(configuration.occupations):
        if not n_electrons:
            continue
        # ``degeneracy`` counts the *electrons* a shell holds and is what the configuration is
        # divided by; ``per_shell`` counts the orbitals they are written on, and ``filled`` is
        # what one of those orbitals carries when the shell is full. The two representations
        # differ only in these three numbers.
        degeneracy = 4 * l + 2
        per_shell = (2 * l + 1) if spatial else degeneracy
        filled = 2.0 if spatial else 1.0
        index = np.where(l_of == l)[0]
        index = index[np.argsort(e[index], kind="stable")]
        full, remainder = divmod(n_electrons, degeneracy)
        needed = full * per_shell + (per_shell if remainder else 0)
        if index.size < needed:
            raise RuntimeError(
                "the {} channel offers {} orbitals but the configuration {} needs {} of them "
                "({} electrons, and a partially filled shell needs all {} orbitals of it "
                "present, not only the occupied part). Use a larger basis for this "
                "element.".format(SHELL_LETTERS[l], index.size, configuration.canonical,
                                  needed, n_electrons, per_shell))
        occ[index[:full * per_shell]] = filled
        if remainder:
            # ⚠ ``q / per_shell``, and the ``filled`` factor deliberately does **not** appear:
            # the electrons are shared out over the orbitals of the shell, so the sum over a
            # frontier shell is ``q`` in either representation. Multiplying by 2 here gives a
            # spherical density with the wrong number of electrons in it.
            occ[index[full * per_shell:needed]] = remainder / per_shell
    return occ


#: One open shell, as ``get_occ`` resolved it: the indices of its spinors in the current
#: spectrum, and the ``(q, n)`` that fix its coupling coefficient.
class OpenShell(object):
    __slots__ = ("l", "q", "n", "index")

    def __init__(self, l, q, n, index):
        self.l, self.q, self.n, self.index = int(l), int(q), int(n), index

    @property
    def occupation(self) -> float:
        return self.q / float(self.n)

    @property
    def coupling(self) -> float:
        """``alpha = n(q-1) / (q(n-1))``: the open-open two-electron coupling of a true
        configuration average, relative to what a fractional density gives.

        The average of a determinantal energy over all ``C(n,q)`` ways of putting ``q``
        electrons in ``n`` degenerate spinors needs the **pair** average
        ``<n_i n_j> = q(q-1)/(n(n-1))``, not the product of one-particle averages ``(q/n)^2``
        that a fractional density supplies. Only the open-open block differs: closed-closed
        pairs have ``n_i n_j = 1`` identically and closed-open pairs average to ``q/n``, which
        is exactly what a fractional density gives.

        ``alpha = 1`` for a closed shell (``q = n``) — which is why the closed-shell path is
        bitwise unchanged — and ``alpha = 0`` at ``q = 1``, where a true average gives the
        open shell no two-electron energy at all and a fractional density still charges the
        single electron ``1/n^2`` of its own repulsion.
        """
        if self.q <= 0 or self.n <= 1:
            return 1.0
        return (self.n * (self.q - 1)) / float(self.q * (self.n - 1))


def install_configuration_average(mf, mol, state) -> None:
    """Make ``mf`` minimize the **average-of-configuration** energy, not the fractional one.

    ⚠ **This is the fix for a real defect** — the first implementation was fractional-occupation
    Hartree-Fock, which is 0.3-0.5 Eh and up to 15% out on an open-shell splitting — and the
    distinction it turns on is
    one coefficient. Occupying a frontier shell fractionally is right; evaluating the
    two-electron energy over the resulting *density* is not, because it factorizes the pair
    average ``<n_i n_j>`` into ``<n_i><n_j>``. See :attr:`OpenShell.coupling`.

    Two consequences, and the second is what makes this more than a scale factor:

    **The energy gains one term per open shell.** ``E_AOC = E_FON + sum_s (a_s - 1)/2 *
    Tr[D_s G[D_s]]``, exact and linear in the mean field. Measured on the *unrelaxed*
    fractional-occupation orbitals it already recovers 93-98% of the gap to four-component
    DIRAC (C 92.9%, O 97.7%, Ti(3+) 93.6%), which is what established that this coefficient —
    and not some other missing term — is the whole story.

    **The shells stop sharing a Fock operator.** ``F_c = h + G[D]`` for the closed shells and
    ``F_s = F_c + (a_s - 1) G[D_s]`` for open shell ``s``, so this becomes an ROHF-shaped
    problem and a single diagonalization has to reproduce three (or more) stationarity
    conditions at once. Differentiating the energy with respect to an orbital rotation mixing
    two shells ``X`` and ``Y`` of occupations ``f_X, f_Y`` gives one rule that covers every
    pair::

        f_X F_X[X,Y] - f_Y F_Y[X,Y] = 0

    with ``f = 1`` for the closed shells and ``f = 0`` for the virtuals (whose Fock is then
    irrelevant, as it must be). The effective Fock is therefore assembled block by block in
    the current MO basis as::

        F_eff[X,Y] = (f_X F_X - f_Y F_Y) / (f_X - f_Y)      X != Y
        F_eff[X,X] = F_X

    The normalization keeps every block on the scale of a Fock matrix rather than of a
    gradient, and the diagonal choice makes each shell's orbital energies eigenvalues of *its
    own* Fock — the quantity a four-component code reports for an open shell, and therefore
    the one the j-splitting comparison needs.

    ⚠ **A closed shell reduces to the identity, exactly.** With no open shells there is one
    ``F_c`` and the rule gives ``F_eff = F_c`` in every block, so the closed-shell path is
    bitwise what it was and every validated X2CAMF number stands untouched. A test asserts that rather
    than trusting it.

    ⚠ **Shared, deliberately, with the two-component validation SCF.** It touches nothing but
    ``mf.get_veff`` / ``get_fock`` / ``energy_elec`` and a state dict, so the four-component
    backend and ``tests/test_amf_open_shell.average_of_configuration_ghf`` install the *same*
    function. That is why it lives here beside :func:`average_occupations` rather than in the
    PySCF backend: the coupling coefficient is now part of "occupying the same configuration
    the same way", and two implementations of it that agreed today would be a comparison
    waiting to become meaningless.

    ⚠ **It is convention-agnostic, and that is a property worth stating rather than
    rediscovering.** The same function drives a four-component spinor SCF, the two-component
    validation GHF and the spin-restricted **scalar** average-of-configuration SCF of the
    front-end, whose orbitals hold *two* electrons each. Three things make that work and each
    is exact rather than approximate:

    * ``alpha`` is a ratio of pair averages over **spin orbitals** and does not know how the
      orbitals are written, so ``n = 4l+2`` and ``q`` electrons give one number in every
      representation.
    * ``G[D_s]`` is whatever ``mf.get_veff`` is — ``J - K`` on a spin-orbital density, ``J -
      K/2`` on a spin-restricted total density — and ``1/2 Tr[D_s G[D_s]]`` is the open-open
      two-electron energy either way.
    * the block rule is invariant under scaling **every** occupation by a common factor, so
      the fractional fillings (1 for closed, ``q/(4l+2)`` for open) may be used in place of
      electron counts. What is *not* invariant is deciding which orbitals are closed by
      comparing an occupation against 1 — see the comment where the partition is built.

    ⚠ **Scope (user decision):** this changes the *SCF* — the orbitals, the density
    and ``e_tot``. It deliberately does **not** change the mean fields the correction is built
    from: :mod:`kuiva.amf.decouple` keeps plain ``G[D]`` on both sides of
    ``dG = PC[G[D]] - G_nr[D~]``, because the subtraction's whole logic is one operator
    transformed minus the same operator untransformed, and the molecular Hamiltonian carries
    the plain two-electron operator with no configuration averaging in it. The consequence to
    respect: **the SCF's Fock and the stored ``veff`` are no longer the same object**, and
    nothing downstream may conflate them.
    """
    # ⚠ ``state`` is **empty at install time**: ``get_occ`` fills it with the shell partition
    # *and the current orbitals* on every cycle, and it is held by reference. Both hooks below
    # fall back to the plain Fock and the plain energy while it is empty, which is exactly
    # right for the first iteration — there are no orbitals yet, so there is no shell
    # partition to respect and the fixed point is unaffected by where the iteration starts.
    base_get_fock = mf.get_fock
    base_energy_elec = mf.energy_elec
    # ``G[D_s]`` is needed by both the Fock and the energy in the same cycle, and a
    # four-component Fock build is the expensive thing here (a lanthanide solve is
    # already 15-60 minutes). Memoized on two independent scalar invariants of the density
    # rather than on ``id(dm)``, which a garbage collector can recycle.
    cache = {"key": None, "veff": None}

    def open_mean_fields(dm):
        """``[(D_s, G[D_s], alpha_s), ...]`` for the current orbitals, or ``[]``."""
        shells = state.get("shells") or []
        mo, occ = state.get("mo_coeff"), state.get("mo_occ")
        if mo is None or occ is None or not shells:
            return []
        dm = np.asarray(dm)
        key = (dm.shape, complex(np.trace(dm)), float(np.linalg.norm(dm)))
        if cache["key"] == key:
            return cache["veff"]
        mo, occ = np.asarray(mo), np.asarray(occ)
        built = []
        for shell in shells:
            index = shell.index
            d = (mo[:, index] * occ[index]) @ mo[:, index].conj().T
            built.append((d, np.asarray(mf.get_veff(mol, d)), shell.coupling))
        cache["key"], cache["veff"] = key, built
        return built

    def get_fock(h1e=None, s1e=None, vhf=None, dm=None, *args, **kwargs):
        if h1e is None:
            h1e = mf.get_hcore(mol)
        if s1e is None:
            s1e = mf.get_ovlp(mol)
        if dm is None:
            dm = mf.make_rdm1()
        fields = open_mean_fields(dm)
        mo = state.get("mo_coeff")
        if not fields or mo is None:
            # First cycle: no orbitals yet, so no shell partition. The plain Fock is a
            # perfectly good starting point and the fixed point is unaffected.
            return base_get_fock(h1e, s1e, vhf, dm, *args, **kwargs)

        mo = np.asarray(mo)
        occ = np.asarray(state["mo_occ"])
        shells = state["shells"]
        f_closed = np.asarray(h1e) + np.asarray(vhf if vhf is not None
                                                else mf.get_veff(mol, dm))
        # (indices, occupation, Fock for the off-diagonal rule, Fock for the diagonal block)
        # per shell: closed first, then each open shell, then the virtuals. The virtuals carry
        # occupation 0, so the rule below never reads their Fock.
        #
        # ⚠ **The diagonal blocks are a reporting convention, and it is not the obvious one.**
        # They cannot change the density or the energy — every orbital within a shell has the
        # same occupation, so any rotation among them leaves both invariant — but they fix
        # what the orbital *energies* are, and for an open shell that is worth **+/-15%** on a
        # j-splitting. Measured against DIRAC over the whole occupied spectrum:
        #
        #   closed F_c, open F_o (each shell's own):   MAE 1.0e-01 Eh
        #   everything from (F_c + F_o)/2:             MAE 1.0e-01 Eh
        #   closed F_c, open (F_c + F_o)/2:            MAE 2.1e-08 Eh   <- this one
        #
        # on C (alpha = 0.6), O (0.9) and Ti(3+) (0.0), i.e. seven orders of discrimination
        # across the full range of the coupling. That is Roothaan's canonical convention, and
        # matching it is what makes a cross-code j-splitting comparison like-for-like.
        # ⚠ Note it is *not* Janak's theorem, which gives dE/dn_t = F_o for the open shell;
        # both are defensible and the comparison merely has to be consistent. For a closed
        # shell the two coincide identically, which is why this never arose for a closed shell.
        #
        # ⚠ **A closed shell is "occupied and in no open shell", never "occupation above a
        # threshold".** A spinor carries at most 1 and a spin-restricted *spatial* orbital
        # carries 2, so a threshold written for one convention silently reclassifies the other:
        # an f^9 shell holds 9/7 = 1.29 electrons per spatial orbital, which is above 1, and
        # the open shell would then be treated as closed *and* as open at once. Reading the
        # partition off ``shells`` is convention-free, and it gives the identical index set in
        # the spinor case, where it was a threshold.
        open_index = ([s.index for s in shells] or [np.zeros(0, dtype=int)])
        closed = np.setdiff1d(np.where(occ > 1e-12)[0], np.concatenate(open_index))
        groups = [(closed, 1.0, f_closed, f_closed)]
        for shell, (_, g_s, alpha) in zip(shells, fields):
            f_shell = f_closed + (alpha - 1.0) * g_s
            groups.append((shell.index, shell.occupation, f_shell,
                           0.5 * (f_closed + f_shell)))
        assigned = np.concatenate([g[0] for g in groups]) if groups else np.array([], int)
        virtual = np.setdiff1d(np.arange(mo.shape[1]), assigned)
        groups.append((virtual, 0.0, f_closed, f_closed))

        # Assemble in the MO basis, then push back with F_ao = (S C) F_mo (S C)^dag, which
        # inverts C^dag F_ao C = F_mo because C^dag S C = 1.
        sc = np.asarray(s1e) @ mo
        # ⚠ The dtype follows the inputs rather than being complex unconditionally: a
        # spin-restricted *scalar* SCF has a real symmetric Fock, and handing PySCF a complex
        # one whose imaginary part happens to be zero makes every orbital it returns complex —
        # with an arbitrary phase per column, which no consumer of a real MO set expects. The
        # spinor paths are complex either way and are bitwise unchanged.
        f_mo = np.zeros((mo.shape[1], mo.shape[1]), dtype=np.result_type(mo, f_closed))
        transformed = [mo.conj().T @ f @ mo for _, _, f, _ in groups]
        diagonal = [mo.conj().T @ f @ mo for _, _, _, f in groups]
        for a, (ia, fa, _, _) in enumerate(groups):
            if ia.size == 0:
                continue
            for b, (ib, fb, _, _) in enumerate(groups):
                if ib.size == 0:
                    continue
                if a == b:
                    block = diagonal[a][np.ix_(ia, ib)]
                else:
                    if abs(fa - fb) < 1e-12:
                        # Two shells of equal occupation: the rotation between them is
                        # redundant and any Hermitian block will do.
                        block = 0.5 * (transformed[a][np.ix_(ia, ib)]
                                       + transformed[b][np.ix_(ia, ib)])
                    else:
                        block = (fa * transformed[a][np.ix_(ia, ib)]
                                 - fb * transformed[b][np.ix_(ia, ib)]) / (fa - fb)
                f_mo[np.ix_(ia, ib)] = block
        f_mo = 0.5 * (f_mo + f_mo.conj().T)
        f_eff = sc @ f_mo @ sc.conj().T
        return base_get_fock(h1e, s1e, f_eff - np.asarray(h1e), dm, *args, **kwargs)

    def energy_elec(dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = mf.make_rdm1()
        # ⚠ PySCF's ``energy_elec`` returns ``(e1 + e_coul, e_coul)`` — the **total**
        # electronic energy first, not the one-electron part — and ``energy_tot`` reads
        # element ``[0]``. Shifting only the second element leaves the reported energy
        # untouched while the orbitals change underneath it, which converges cleanly to a
        # number that is neither functional's answer. Both elements carry the shift.
        electronic, e_coul = base_energy_elec(dm, h1e, vhf)
        shift = 0.0
        for d, g, alpha in open_mean_fields(dm):
            shift += 0.5 * (alpha - 1.0) * float(np.real(np.einsum("ij,ji->", d, g)))
        return electronic + shift, e_coul + shift

    mf.get_fock = get_fock
    mf.energy_elec = energy_elec



__all__ = ["AtomicConfiguration", "F_BLOCKS", "NOBLE_GASES", "OpenShell", "SHELL_LETTERS",
           "angular_channel_groups", "average_occupations", "default_configuration",
           "install_configuration_average", "is_f_block", "parse_shell_terms",
           "spherical_projector"]
