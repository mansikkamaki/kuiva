"""Per-element free-atom reference orbitals — the boundary object behind the robust charges.

One entry per *element* of the molecule: the orbitals and occupations of a spherically
averaged free-atom scalar-relativistic SCF **in that element's own basis**, computed in the
front end (the only place integrals can be evaluated) and carried downstream as plain arrays,
so the analysis layer needs no integral library. The construction and its measured robustness
live with the consumer, :mod:`kuiva.props.population`; what is fixed here is the contract:

* ``c`` is ``(nbf, nbf)`` over the element's AO block in the molecule's own AO ordering —
  the *same* basis entry produces the same shell ordering for the atom alone as for that atom
  inside the molecule, which is what lets a block-diagonal placement reproduce atomic
  character exactly.
* ``occ`` are the average-of-configuration occupations (fractional on open shells), which
  double as the orthogonalization weights downstream.
* ``configuration`` is the label of the atomic reference state, and ``is_default`` records
  whether it is Kuiva's per-element default. ⚠ Charges computed against *non-default*
  references are not comparable with default-reference charges, and the consumer warns —
  which is why the flag travels with the data instead of being re-derived.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

__all__ = ["AtomicReferenceEntry", "AtomicReferenceSet"]


@dataclass(frozen=True)
class AtomicReferenceEntry:
    """The free-atom reference for one element: orbitals, occupations, provenance."""
    element: str
    c: np.ndarray            #: (nbf, nbf) atomic MO coefficients over the element's AO block
    occ: np.ndarray          #: (nbf,) average-of-configuration occupations
    configuration: str       #: label of the atomic reference state (provenance)
    is_default: bool         #: True when it is Kuiva's per-element default reference
    converged: bool = True

    def __post_init__(self) -> None:
        c = np.asarray(self.c, dtype=float)
        occ = np.asarray(self.occ, dtype=float)
        if c.ndim != 2 or c.shape[0] != c.shape[1] or occ.shape != (c.shape[1],):
            raise ValueError("reference orbitals for {} have shape {} against occupations "
                             "{}".format(self.element, c.shape, occ.shape))
        object.__setattr__(self, "c", np.ascontiguousarray(c))
        object.__setattr__(self, "occ", np.ascontiguousarray(occ))


@dataclass
class AtomicReferenceSet:
    """One entry per label group of a molecule, plus the per-atom key list.

    A key is the atom's front-end label: the plain element symbol, or the decorated
    ``"Ti2"`` (1-based) of an atom carrying its own basis or reference state.
    ``atom_keys[i]`` maps atom ``i`` (0-based internally) to its entry; a set built by hand
    without ``atom_keys`` falls back to element-symbol lookup, which is exact whenever
    nothing per-atom was requested.
    """
    entries: Dict[str, AtomicReferenceEntry] = field(default_factory=dict)
    atom_keys: list = field(default_factory=list)
    basis_label: str = ""

    def __contains__(self, key: str) -> bool:
        return key.capitalize() in self.entries

    def __getitem__(self, key: str) -> AtomicReferenceEntry:
        return self.entries[key.capitalize()]

    def entry_for_atom(self, ia: int, element: str) -> AtomicReferenceEntry:
        """The reference for atom ``ia`` (0-based); ``element`` is the fallback lookup."""
        if self.atom_keys:
            return self.entries[self.atom_keys[ia]]
        return self.entries[element.capitalize()]

    @property
    def any_non_default(self) -> bool:
        return any(not e.is_default for e in self.entries.values())
