"""Ghost atoms: a basis set at a point in space, with no nucleus and no electrons.

A ghost carries basis functions and nothing else. It exists so that a calculation can be run
in *another* structure's basis without that structure's charge or electrons being present,
which is what the counterpoise correction and every basis-set-superposition-error diagnostic
are built from — and, for this program in particular, what makes a basis-set projection's
"did the target basis have room for these orbitals?" question answerable.

**The spelling is a label, not a flag** (user decision): an atom is written
``("ghost-Cl", (x, y, z))``, and that label is what the per-atom basis map addresses, what
the output prints and what every per-atom consumer sees. A flag on the side would leave the
label saying "Cl" while the atom is nothing of the kind. Two ghosts of one element take
decorated labels (``ghost-Cl1``, ``ghost-Cl2``) exactly as two real atoms of one element do,
so they can carry different bases.

``X-Cl`` and ``ghost:Cl`` are accepted on input, because both are spellings the integral
library understands and a user arriving from elsewhere will try them; they are **normalized
to one form** here so that nothing downstream has to know there were ever three.

⚠ **What a ghost is not.** It has no nucleus, so it contributes no nuclear attraction, no
nuclear repulsion, no mass and no electrons; and it therefore has **no atomic mean field**
(:mod:`kuiva.amf`), no free-atom reference orbitals, and no oxidation state. Every one of
those consumers skips ghosts by an explicit test rather than by an accident of naming — a
ghost's element symbol is *not* recoverable by the integral library's own
``atom_pure_symbol``, which returns the ghost label whole, so code that assumed otherwise
would ask for the chemistry of an element called ``GHOST-Cl``.

⚠ **A ghost still breaks symmetry, and still costs.** It carries AO functions, so it enters
the orthonormal working basis, the integral factorization, the memory plan and the symmetry
detection exactly as a real atom does. What it does not do is put anything in the
Hamiltonian's nuclear potential.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

#: The canonical prefix. One spelling downstream, whatever was written on input.
GHOST_PREFIX = "ghost-"

#: Accepted input spellings of the prefix, all normalized to :data:`GHOST_PREFIX`.
_PREFIXES = ("ghost-", "ghost:", "ghost", "x-")

_LABELLED = re.compile(r"^([A-Za-z]{1,2})([0-9]*)$")


def is_ghost(symbol: str) -> bool:
    """Whether ``symbol`` names a ghost, in any accepted spelling."""
    return _prefix_length(str(symbol)) > 0


def _prefix_length(symbol: str) -> int:
    low = symbol.strip().lower()
    for p in _PREFIXES:
        if low.startswith(p):
            rest = low[len(p):]
            # ``x-`` is a ghost prefix; ``Xe`` is an element. The remainder has to look like
            # an element for this to be a prefix at all.
            if rest and rest[0].isalpha():
                return len(p)
    return 0


def ghost_element(symbol: str) -> str:
    """The element a ghost carries the basis of; the symbol itself for a real atom.

    ``"ghost-Cl2" -> "Cl"``, ``"Cl2" -> "Cl"``, ``"Cl" -> "Cl"``. This is what a basis-set
    registry lookup and a coverage check must use: a ghost chlorine needs chlorine's
    functions, and the registry knows nothing about ghosts.
    """
    text = str(symbol).strip()
    text = text[_prefix_length(text):]
    m = _LABELLED.match(text)
    return m.group(1).capitalize() if m else text.capitalize()


def normalize_symbol(symbol: str) -> str:
    """One canonical spelling of an atom symbol, ghost or not.

    ``"GHOST-cl" -> "ghost-Cl"``, ``"x-Cl2" -> "ghost-Cl2"``, ``"cl" -> "Cl"``. Applied at
    every boundary where a symbol arrives from a user, so the rest of the program compares
    strings that were written the same way.
    """
    text = str(symbol).strip()
    n = _prefix_length(text)
    if not n:
        return text.capitalize()
    rest = text[n:]
    m = _LABELLED.match(rest)
    body = (m.group(1).capitalize() + m.group(2)) if m else rest.capitalize()
    return GHOST_PREFIX + body


def split_label(symbol: str) -> Tuple[str, Optional[int]]:
    """``("ghost-Cl", 2)`` for ``"ghost-Cl2"``; ``("Cl", None)`` for ``"Cl"``.

    The label number is the 1-based atom number the front end decorates with when two atoms
    of one element must be told apart; splitting it off is how an element-level lookup finds
    its family again.
    """
    text = normalize_symbol(symbol)
    n = _prefix_length(text)
    prefix, rest = text[:n], text[n:]
    m = _LABELLED.match(rest)
    if m and m.group(2):
        return prefix + m.group(1).capitalize(), int(m.group(2))
    return text, None


def label_for(element: str, ghost: bool) -> str:
    """The canonical label for an element carried as a real atom or as a ghost."""
    element = str(element).strip().capitalize()
    return (GHOST_PREFIX + element) if ghost else element


__all__ = ["GHOST_PREFIX", "ghost_element", "is_ghost", "label_for", "normalize_symbol",
           "split_label"]
