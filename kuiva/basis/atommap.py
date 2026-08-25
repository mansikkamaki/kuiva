"""Per-atom assignment maps: one addressing scheme for "which atoms get this value".

Both per-atom features — reference configurations and basis sets — accept the same mapping
syntax, resolved here so the two cannot drift apart. A mapping key is one of

* an **element symbol** (``"Ti"``) — every atom of that element;
* an **atom label** (``"Ti2"``) — atom number 2 of the molecule's atom list, which must be a
  titanium: a label naming the wrong element is refused, never reinterpreted;
* a **plain integer** (``3``, or the string ``"3"``) — atom number 3 whatever its element.

A **ghost** (:mod:`kuiva.basis.ghosts`) is addressed by its own label — ``"ghost-Cl"`` for
every ghost chlorine, ``"ghost-Cl2"`` for one of them — and never by the element it carries
the basis of. ⚠ That is the point rather than a detail: ``basis={"Cl": ...}`` must not reach
a ghost chlorine, because a ghost and a real atom of one element are two different things to
every consumer downstream, and a key that covered both would be a way to state one basis and
get two.

⚠ **Numbering is 1-based in all input and output** (user decision: the quantum-chemistry
convention). Internally everything is 0-based NumPy indexing; this module is the boundary
where the two meet, so no other module converts.

Precedence is most-specific-first: an atom label or index beats an element entry, which
beats the default. A key naming an atom or element the molecule does not contain is refused
— a silently ignored entry would mean a calculation ran with a different assignment from the
one that was asked for (the same rule the atomic mean field has always applied).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from .ghosts import normalize_symbol

__all__ = ["parse_atom_key", "resolve_atom_assignments"]

_LABEL = re.compile(r"^((?:ghost-)?[A-Za-z]{1,2})([0-9]+)$")


def parse_atom_key(key, symbols: Sequence[str]):
    """Classify one mapping key against the molecule's atoms.

    Returns ``("element", symbol)`` or ``("atom", index0)`` with a 0-based index. Raises on
    anything that names no atom of this molecule.
    """
    n = len(symbols)
    caps = [normalize_symbol(s) for s in symbols]
    if isinstance(key, int) and not isinstance(key, bool):
        if not 1 <= key <= n:
            raise ValueError(
                "atom number {} is out of range: this molecule has atoms 1..{} "
                "(numbering is 1-based)".format(key, n))
        return "atom", key - 1
    if isinstance(key, str):
        text = key.strip()
        if text.isdigit():
            return parse_atom_key(int(text), symbols)
        text = normalize_symbol(text) if not text.isdigit() else text
        m = _LABEL.match(text)
        if m and normalize_symbol(m.group(1)) in caps:
            sym, num = normalize_symbol(m.group(1)), int(m.group(2))
            if not 1 <= num <= n:
                raise ValueError(
                    "atom label {!r} is out of range: this molecule has atoms 1..{} "
                    "(numbering is 1-based)".format(text, n))
            if caps[num - 1] != sym:
                raise ValueError(
                    "atom label {!r} names a {} but atom {} of this molecule is {} — a "
                    "label is refused rather than reinterpreted".format(
                        text, sym, num, caps[num - 1]))
            return "atom", num - 1
        if text in caps:
            return "element", text
        raise ValueError(
            "the key {!r} names no atom of this molecule (elements: {}; atoms 1..{}). "
            "Use an element symbol, a label like {}2, or a 1-based atom number.".format(
                key, ", ".join(sorted(set(caps))), n, caps[0]))
    raise TypeError("an atom-assignment key must be an element symbol, an atom label or a "
                    "1-based atom number, not {!r}".format(key))


def resolve_atom_assignments(spec, symbols: Sequence[str], *, what: str,
                             default=None, allow_scalar: bool = True
                             ) -> Tuple[List[object], List[bool]]:
    """Resolve a per-atom assignment spec into one value per atom.

    ``spec`` is ``None`` (every atom takes ``default``), a scalar (every atom takes it —
    refused for a multi-element molecule unless ``allow_scalar`` says otherwise, mirroring
    the atomic mean field's rule that an oxidation state stated once must not strip
    electrons off the ligands), or a mapping in the module syntax. A mapping may carry a
    ``"default"`` key, which fills every atom no more specific entry covers.

    Returns ``(values, is_specific)`` with one entry per atom; ``is_specific[i]`` says an
    explicit (non-default) entry reached atom ``i`` — what a caller needs to decide which
    atoms must keep their own label downstream.
    """
    n = len(symbols)
    caps = [normalize_symbol(s) for s in symbols]
    if spec is None:
        return [default] * n, [False] * n

    if not isinstance(spec, dict):
        if len(set(caps)) > 1 and not allow_scalar:
            raise ValueError(
                "a single {} ({!r}) cannot be applied to every atom of a molecule "
                "containing {}; pass a mapping.".format(what, spec, ", ".join(sorted(set(caps)))))
        return [spec] * n, [True] * n

    values: List[object] = [default] * n
    tier = [0] * n                     # 0 default, 1 element, 2 atom
    fallback = None
    has_fallback = False
    for key, value in spec.items():
        if isinstance(key, str) and key.strip().lower() == "default":
            fallback, has_fallback = value, True
            continue
        kind, target = parse_atom_key(key, symbols)
        if kind == "element":
            for i in range(n):
                if caps[i] == target and tier[i] < 1:
                    values[i], tier[i] = value, 1
        else:
            if tier[target] == 2:
                raise ValueError(
                    "atom {} is assigned twice ({} entries {!r} and another)".format(
                        target + 1, what, key))
            values[target], tier[target] = value, 2
    if has_fallback:
        for i in range(n):
            if tier[i] == 0:
                values[i] = fallback
    return values, [t == 2 for t in tier]
