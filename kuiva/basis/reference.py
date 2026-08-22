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
    """All elements of one molecule, keyed by capitalized symbol."""
    entries: Dict[str, AtomicReferenceEntry] = field(default_factory=dict)
    basis_label: str = ""

    def __contains__(self, element: str) -> bool:
        return element.capitalize() in self.entries

    def __getitem__(self, element: str) -> AtomicReferenceEntry:
        return self.entries[element.capitalize()]

    @property
    def any_non_default(self) -> bool:
        return any(not e.is_default for e in self.entries.values())
