"""The curated common-oxidation-state table, and the reference-configuration resolver.

**Why a curated table (user decision).** An oxidation state alone is ambiguous input — which
electrons a cation loses is chemistry, not arithmetic — so the program keeps one table of
the *most common* oxidation states per element and, for each, exactly **one canonical
configuration** (what an oxidation-state input produces) plus a set of **accepted**
configurations (explicitly written configurations that raise no warning, several where the
literature genuinely admits more than one, e.g. the near-degenerate d/s occupations of the
late transition metals). Everything else still runs — these are *warnings*, because an
unusual reference is a legitimate research choice — but it runs announced.

The two tiers:

* **Refused** (hard errors): an electron count below 1; a channel filled beyond what shells
  up to the element's period + 1 can hold (no ``8s`` electron on iron); an anion whose
  electron count does not close a noble-gas shell when derived from an oxidation state
  (write the configuration explicitly if that is really what is meant).
* **Warned**: an oxidation state outside the element's common set; an explicit
  configuration that matches no accepted configuration of any common state of that element
  (an excited or unusual reference); any anion reference (a free anion is at best weakly
  bound and the finite basis acts as confinement).

**Canonical configurations are derived, not stored.** The mechanical rules are exact for
every common state: neutral atoms take the tabulated ground configuration, f-block cations
the ``[core] f^n`` rule, other cations the strip-highest-``n`` rule (which yields the
``d^k s^0`` occupation every transition-metal cation has), and table anions the noble-gas
closure. The table stores only *which* states are common and the few genuinely ambiguous
accepted alternates — data that would otherwise be wrong silently is data this module does
not keep.

References
----------
* Common oxidation states: N. N. Greenwood, A. Earnshaw, "Chemistry of the Elements",
  2nd ed., Butterworth-Heinemann (1997); the selection kept here is the "most common"
  (boldface) states of the usual periodic-table compilations, deliberately short.
* Ground-state configurations: the aufbau table of :mod:`kuiva.amf.configuration`
  (PySCF's, following NIST ASD).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from .configuration import (AtomicConfiguration, F_BLOCKS, NOBLE_GASES,
                            _neutral_occupations, default_configuration, is_f_block)
from ..util.logging import get_logger

log = get_logger(__name__)

__all__ = ["COMMON_STATES", "canonical_configuration", "accepted_configurations",
           "resolve_reference_configuration", "common_states"]

#: Most common oxidation states per element, H(1)-Lr(103). 0 (the neutral atom) is common
#: for every element and is not listed. Kept deliberately short: this is "the states a
#: chemist writes without comment", not "every state ever isolated".
COMMON_STATES: Dict[str, Tuple[int, ...]] = {
    "H": (-1, 1), "He": (),
    "Li": (1,), "Be": (2,), "B": (3,), "C": (-4, 2, 4), "N": (-3, 3, 5), "O": (-2,),
    "F": (-1,), "Ne": (),
    "Na": (1,), "Mg": (2,), "Al": (3,), "Si": (-4, 4), "P": (-3, 3, 5), "S": (-2, 4, 6),
    "Cl": (-1, 1, 3, 5, 7), "Ar": (),
    "K": (1,), "Ca": (2,),
    "Sc": (3,), "Ti": (2, 3, 4), "V": (2, 3, 4, 5), "Cr": (2, 3, 6),
    "Mn": (2, 3, 4, 6, 7), "Fe": (2, 3), "Co": (2, 3), "Ni": (2,), "Cu": (1, 2),
    "Zn": (2,),
    "Ga": (3,), "Ge": (2, 4), "As": (-3, 3, 5), "Se": (-2, 4, 6), "Br": (-1, 1, 3, 5),
    "Kr": (),
    "Rb": (1,), "Sr": (2,),
    "Y": (3,), "Zr": (4,), "Nb": (3, 5), "Mo": (3, 4, 6), "Tc": (4, 7),
    "Ru": (2, 3, 4), "Rh": (3,), "Pd": (2, 4), "Ag": (1,), "Cd": (2,),
    "In": (3,), "Sn": (2, 4), "Sb": (3, 5), "Te": (-2, 4, 6), "I": (-1, 1, 5, 7),
    "Xe": (2, 4, 6),
    "Cs": (1,), "Ba": (2,),
    "La": (3,), "Ce": (3, 4), "Pr": (3,), "Nd": (3,), "Pm": (3,), "Sm": (2, 3),
    "Eu": (2, 3), "Gd": (3,), "Tb": (3, 4), "Dy": (3,), "Ho": (3,), "Er": (3,),
    "Tm": (3,), "Yb": (2, 3), "Lu": (3,),
    "Hf": (4,), "Ta": (5,), "W": (4, 6), "Re": (3, 4, 7), "Os": (3, 4), "Ir": (3, 4),
    "Pt": (2, 4), "Au": (1, 3), "Hg": (1, 2),
    "Tl": (1, 3), "Pb": (2, 4), "Bi": (3,), "Po": (2, 4), "At": (-1, 1), "Rn": (),
    "Fr": (1,), "Ra": (2,),
    "Ac": (3,), "Th": (4,), "Pa": (4, 5), "U": (3, 4, 5, 6), "Np": (3, 4, 5),
    "Pu": (3, 4), "Am": (3,), "Cm": (3,), "Bk": (3, 4), "Cf": (3,), "Es": (3,),
    "Fm": (3,), "Md": (2, 3), "No": (2, 3), "Lr": (3,),
}

#: Explicitly accepted alternates beyond what the mechanical rules generate: the genuinely
#: near-degenerate occupations of the late platinum-group neutrals. (The other famous
#: anomalies — Cr, Cu, Nb, Mo, La, Ce, Gd, Th, U... — are covered mechanically: both the
#: tabulated ground configuration and the regular Madelung filling are accepted for every
#: neutral atom.)
_ACCEPTED_EXPLICIT: Dict[Tuple[str, int], Tuple[str, ...]] = {
    ("Ni", 0): ("[Ar]3d9 4s1",),
    ("Pd", 0): ("[Kr]4d9 5s1",),
    ("Pt", 0): ("[Xe]4f14 5d10",),
}

_PERIOD_EDGES = (0, 2, 10, 18, 36, 54, 86, 118)
_MADELUNG = None    # lazily built (n+l, n)-ordered shell list


def _z_of(element: Union[str, int]) -> int:
    from pyscf import gto
    return int(gto.charge(element)) if isinstance(element, str) else int(element)


def _symbol_of(element: Union[str, int]) -> str:
    if isinstance(element, str):
        return element.capitalize()
    from pyscf import gto
    inv = {int(gto.charge(s)): s for s in COMMON_STATES}
    return inv.get(int(element), str(element))


def _period(z: int) -> int:
    for p, edge in enumerate(_PERIOD_EDGES[1:], start=1):
        if z <= edge:
            return p
    return len(_PERIOD_EDGES) - 1


def _madelung_occupations(n_elec: int) -> Tuple[int, ...]:
    """Per-l totals of the regular (n+l, n)-ordered aufbau filling of ``n_elec``."""
    global _MADELUNG
    if _MADELUNG is None:
        shells = [(n + l, n, l) for n in range(1, 9) for l in range(0, min(n, 4))]
        _MADELUNG = sorted(shells)
    occ = [0, 0, 0, 0]
    left = int(n_elec)
    for _, _n, l in _MADELUNG:
        if left <= 0:
            break
        take = min(left, 2 * (2 * l + 1))
        occ[l] += take
        left -= take
    return tuple(occ)


def common_states(element: Union[str, int]) -> Tuple[int, ...]:
    """The curated common oxidation states of ``element`` (0 is always common)."""
    return COMMON_STATES.get(_symbol_of(element), ())


def canonical_configuration(element: Union[str, int], q: int) -> AtomicConfiguration:
    """THE configuration an oxidation-state input produces (user decision: exactly one).

    Neutral: the tabulated ground configuration. f block: ``[core] f^n``, any ``q``. Other
    cations: the strip-highest-``n`` rule, which gives the ``d^k`` occupation of every
    transition-metal cation. Anions: the noble-gas closure, refused when ``Z - q`` closes no
    shell (an anion that is not a closed shell must be written out explicitly).
    """
    z = _z_of(element)
    sym = _symbol_of(element)
    q = int(q)
    if q == 0:
        return AtomicConfiguration.ground(element)
    if q > 0 or is_f_block(z):
        return AtomicConfiguration.for_oxidation_state(element, q)
    n_elec = z - q
    if n_elec not in NOBLE_GASES.values():
        raise ValueError(
            "{}({:+d}) has {} electrons, which closes no noble-gas shell; a non-closed-"
            "shell anion reference must be written as an explicit configuration.".format(
                sym, q, n_elec))
    return AtomicConfiguration(_neutral_occupations(n_elec),
                               label="{}({:+d}) closed shell".format(sym, q))


def accepted_configurations(element: Union[str, int], q: int) -> List[AtomicConfiguration]:
    """Every configuration of ``element`` in state ``q`` that raises no warning.

    The canonical one; for a neutral atom also the regular Madelung filling (which absorbs
    the Cr/Cu/Nb/Mo/lanthanide/actinide ground-state anomalies from either direction); for
    a d-block cation also the near-degenerate ``d^(k-1) s^1`` occupation; plus the explicit
    alternates of the table.
    """
    z = _z_of(element)
    sym = _symbol_of(element)
    out = [canonical_configuration(element, q)]
    if q == 0:
        alt = AtomicConfiguration(_madelung_occupations(z), label="Madelung filling")
        if alt != out[0]:
            out.append(alt)
    d_block = (21 <= z <= 30) or (39 <= z <= 48) or (72 <= z <= 80)
    if q > 0 and d_block:
        occ = list(out[0].occupations)
        if len(occ) > 2 and occ[2] > 0:
            ds = list(occ)
            ds[2] -= 1
            ds[0] += 1
            out.append(AtomicConfiguration(ds, label="d^(k-1) s^1 alternate"))
    for text in _ACCEPTED_EXPLICIT.get((sym, q), ()):
        alt = AtomicConfiguration.parse(text)
        if all(alt != a for a in out):
            out.append(alt)
    return out


def _check_capacity(element: Union[str, int], config: AtomicConfiguration) -> None:
    """Refuse a channel filled beyond what shells up to the element's period + 1 hold."""
    z = _z_of(element)
    n_max = _period(z) + 1
    letters = "spdfghi"
    for l, n in enumerate(config.occupations):
        shells = max(0, n_max - l)
        cap = 2 * (2 * l + 1) * shells
        if n > cap:
            raise ValueError(
                "the configuration {} puts {} electrons in the {} channel of {}, which "
                "cannot hold more than {} with shells up to n = {} (period + 1). This is "
                "not a reasonable atomic reference.".format(
                    config.canonical, n, letters[l], _symbol_of(element), cap, n_max))


def resolve_reference_configuration(element: Union[str, int], spec,
                                    *, warn: bool = True
                                    ) -> Tuple[AtomicConfiguration, bool]:
    """The single entry point every consumer resolves a user reference state through.

    Returns ``(configuration, is_default)``. ``spec`` is anything
    :meth:`AtomicConfiguration.coerce` accepts; the difference is that *this* routing runs
    the curated-table checks of the module docstring, exactly once, on user input only —
    the program's own defaults never warn (they are recorded decisions, not user choices).
    """
    import re

    sym = _symbol_of(element)
    default = default_configuration(element)
    if spec is None:
        return default, True
    if hasattr(spec, "to_atomic"):
        spec = spec.to_atomic()

    q: Optional[int] = None
    config: Optional[AtomicConfiguration] = None
    if isinstance(spec, bool):
        raise TypeError("a bool is not a configuration")
    if isinstance(spec, int):
        q = int(spec)
    elif isinstance(spec, str) and re.match(r"^\s*[+-]?\d+\s*$|^\s*\d+\s*[+-]\s*$",
                                            spec):
        text = spec.strip().replace(" ", "")
        q = -int(text[:-1]) if text.endswith("-") else \
            int(text[:-1]) if text.endswith("+") else int(text)
    elif isinstance(spec, AtomicConfiguration):
        config = spec
    else:
        config = AtomicConfiguration.coerce(spec, element=sym)

    if q is not None:                      # an oxidation state: the table picks THE config
        config = canonical_configuration(element, q)
        if warn and q != 0 and q not in common_states(element):
            log.warning("%+d is not a common oxidation state of %s (common: %s); the "
                        "reference %s was derived by the standard rule and should be "
                        "checked.", q, sym,
                        ", ".join("%+d" % s for s in common_states(element)) or "none",
                        config.canonical)
    else:                                  # an explicit configuration: check acceptance
        _check_capacity(element, config)
        z = _z_of(element)
        q_implied = z - config.n_electrons
        accepted = []
        if q_implied == 0 or q_implied in common_states(element):
            try:
                accepted = accepted_configurations(element, q_implied)
            except ValueError:
                accepted = []
        if warn and all(config != a for a in accepted):
            if q_implied != 0 and q_implied not in common_states(element):
                log.warning("the reference %s for %s implies the oxidation state %+d, "
                            "which is not a common one (common: %s).", config.canonical,
                            sym, q_implied,
                            ", ".join("%+d" % s for s in common_states(element)) or "none")
            else:
                log.warning("the reference %s differs from the known ground "
                            "configuration(s) of %s(%+d) — an excited or unusual "
                            "reference. It will be used as given.", config.canonical, sym,
                            q_implied)
    if warn and config.n_electrons > _z_of(element):
        log.warning("the reference for %s is an anion (%d electrons, Z = %d): a free "
                    "anion is at best weakly bound and the finite basis acts as its "
                    "confinement. The charges and mean fields built on it depend on the "
                    "basis more than usual.", sym, config.n_electrons, _z_of(element))
    return config, config == default
