"""A basis set the registry does not know: the escape hatch, with the registry's discipline.

The fifty registered families exist so that a calculation cannot silently be run with a basis
that does not suit it — coverage, contraction type, conditioning and, above all, the
**relativistic treatment** are attached to a name and checked before ingestion. That is worth
keeping, and it is also a wall: a newly published set, a locally modified one, or a set from a
paper being reproduced has no name here.

:class:`CustomBasis` is the way through, and it keeps every check that can still be made:

* **Contraction type is measured, never declared.** It comes from
  :func:`kuiva.basis.registry.classify_contraction` on the *parsed* shells, exactly as it does
  for a registered family's data. A user's opinion about their own basis is not evidence.
* **Coverage is measured**: the elements the data actually defines are the elements it covers.
* ⚠ **The relativistic treatment cannot be measured and is therefore REQUIRED** (user
  decision). Nothing in a list of exponents says whether the set was recontracted for X2C, for
  DKH, or for a non-relativistic Hamiltonian; the difference is invisible in every number until
  it reaches a heavy element, where a non-relativistic set under an X2C Hamiltonian gives
  plausible, wrong splittings. A `CustomBasis` without ``relativistic_treatment=`` is refused
  rather than defaulted, and what is declared takes part in the same cross-atom compatibility
  check every registered family goes through.
* **Conditioning is unknown and says so**, which routes the two-electron integrals to Cholesky
  — the choice whose error is a threshold the user sets rather than an unbounded fitting error.

**Two input forms**, because both are things a user already has: an **NWChem-format string**
(what Basis Set Exchange emits and what most published sets are distributed as) and a
**parsed** specification (the nested lists the integral library itself works in, i.e. what a
previous calculation or another program's Python interface hands over). Both are per element,
and either may be mixed with registered families atom by atom through the ordinary per-atom
basis map.

⚠ **A custom basis is not exempt from the front end's own rules.** It is decontracted for the
four-component atomic solve like any other, its atomic mean field is cached on the *content* of
the parsed data rather than on its name (so two different sets sharing a name cannot alias),
and Cartesian shells are refused wherever two-component dimensions are involved.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Tuple, Union

from .registry import Conditioning, Contraction, FitRoute, RelTreatment, classify_contraction
from .registry import symbol_of, z_of

#: Accepted spellings of the relativistic-treatment declaration, mapped onto the registry's
#: own vocabulary so a custom set and a registered one are compared in one language.
_TREATMENTS = {
    "x2c": RelTreatment.X2C_2C, "x2c-2c": RelTreatment.X2C_2C, "x2c_2c": RelTreatment.X2C_2C,
    "2c": RelTreatment.X2C_2C,
    "x2c-1c": RelTreatment.X2C_1C, "x2c_1c": RelTreatment.X2C_1C, "1c": RelTreatment.X2C_1C,
    "scalar": RelTreatment.X2C_1C,
    "dkh": RelTreatment.DKH, "dk": RelTreatment.DKH,
    "nonrel": RelTreatment.NONREL, "non-relativistic": RelTreatment.NONREL,
    "nr": RelTreatment.NONREL,
}


def _parse(data) -> Dict[str, object]:
    """``{element symbol: parsed shells}`` from whatever form the user supplied.

    An NWChem-format string is parsed by the integral library — the same call the registry
    makes for a Basis Set Exchange family, so a set that round-trips through this path is the
    set the registry would have produced.
    """
    from pyscf import gto

    if isinstance(data, str):
        parsed = gto.basis.parse(data)
        # ``parse`` returns one element's shells for a single-element block, and a dict when
        # the string defines several. Normalize to the dict form.
        if isinstance(parsed, dict):
            return {str(k).capitalize(): v for k, v in parsed.items()}
        elements = _elements_named_in(data)
        if len(elements) != 1:
            raise ValueError(
                "this NWChem basis string defines {} elements ({}) but parsed to a single "
                "shell list; supply one element per CustomBasis, or a mapping".format(
                    len(elements), ", ".join(elements) or "none"))
        return {elements[0]: parsed}
    if isinstance(data, dict):
        out = {}
        for key, value in data.items():
            sym = symbol_of(z_of(key)) if not isinstance(key, str) or key.isdigit() \
                else str(key).capitalize()
            out[sym] = gto.basis.parse(value) if isinstance(value, str) else value
        return out
    raise TypeError(
        "a custom basis is an NWChem-format string or a mapping of element to shells; "
        "got {!r}".format(type(data).__name__))


def _elements_named_in(text: str) -> Tuple[str, ...]:
    """Element symbols appearing in the leading column of an NWChem block."""
    seen = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isalpha() and len(parts[1]) <= 2:
            sym = parts[0].capitalize()
            try:
                z_of(sym)
            except KeyError:
                continue
            if sym not in seen:
                seen.append(sym)
    return tuple(seen)


@dataclass(frozen=True, eq=False)          # identity equality: ``data`` may be an unhashable
class CustomBasis:                          # mapping, and two sets are compared by digest()
    """A user-supplied basis, usable anywhere a registry name is.

    Parameters
    ----------
    data : str or mapping
        An NWChem-format string, or ``{element: shells}`` where the shells are either an
        NWChem string for that element or an already-parsed shell list.
    relativistic_treatment : str
        ⚠ **Required.** ``"x2c-2c"`` (a two-component recontraction, what this program's
        default Hamiltonian wants), ``"x2c-1c"``, ``"dkh"`` or ``"nonrel"``. It cannot be
        measured from the data and it is the one property whose absence produces plausible
        wrong numbers on a heavy element, so it is asked for rather than guessed.
    name : str
        A label for the output, the provenance and the basis-consistency messages. Two
        different sets may share a name without aliasing anywhere it matters — the atomic
        mean-field cache keys on the parsed content, not on this.
    notes : str
        Free text carried into the provenance: where the set came from, what was modified.
    """

    data: object
    relativistic_treatment: str = ""
    name: str = "custom"
    notes: str = ""
    _parsed: Dict[str, object] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.relativistic_treatment:
            raise ValueError(
                "a custom basis must declare relativistic_treatment= ('x2c-2c', 'x2c-1c', "
                "'dkh' or 'nonrel'). It cannot be measured from the shells, and an "
                "undeclared one is how a non-relativistic set ends up under a relativistic "
                "Hamiltonian: every number stays plausible and the heavy-element splittings "
                "are wrong.")
        key = str(self.relativistic_treatment).strip().lower()
        if key not in _TREATMENTS:
            raise ValueError(
                "unknown relativistic_treatment {!r}; expected one of {}".format(
                    self.relativistic_treatment,
                    ", ".join(sorted({t.value for t in _TREATMENTS.values()}))))
        object.__setattr__(self, "_parsed", _parse(self.data))
        if not self._parsed:
            raise ValueError("this custom basis defines no elements")

    # -- the family-shaped surface the front end and the registry consume ----------------
    @property
    def rel_treatment(self) -> RelTreatment:
        return _TREATMENTS[str(self.relativistic_treatment).strip().lower()]

    @property
    def conditioning(self) -> Conditioning:
        """⚠ Unknown, and stated as the worst case: a custom set gets the Cholesky route,
        whose error is a threshold the user sets rather than an unbounded fitting error."""
        return Conditioning.ILL

    @property
    def recommended_aux(self) -> Optional[str]:
        return None

    @property
    def role(self) -> str:
        return "user-supplied"

    @property
    def references(self) -> Tuple[str, ...]:
        return ()

    def reference_objs(self) -> Tuple:
        return ()

    def fit_route(self) -> FitRoute:
        return FitRoute.CHOLESKY

    def covers(self, element) -> bool:
        return self.element_symbol(element) in self._parsed

    def covered_elements(self) -> FrozenSet[int]:
        return frozenset(z_of(s) for s in self._parsed)

    @staticmethod
    def element_symbol(element) -> str:
        return symbol_of(z_of(element))

    def shells_for(self, element) -> object:
        """The parsed shells this set defines for ``element``."""
        sym = self.element_symbol(element)
        if sym not in self._parsed:
            raise ValueError(
                "the custom basis {!r} defines no functions for {} (it defines {})".format(
                    self.name, sym, ", ".join(sorted(self._parsed))))
        return self._parsed[sym]

    def contraction_of(self, element) -> Contraction:
        """The contraction type **measured** from this element's parsed shells."""
        return classify_contraction(self.shells_for(element))

    def digest(self) -> str:
        """A stable hash of the parsed data — what two files are compared through."""
        try:
            text = json.dumps({k: v for k, v in sorted(self._parsed.items())},
                              sort_keys=True, default=repr)
        except (TypeError, ValueError):                                 # pragma: no cover
            text = repr(sorted(self._parsed.items()))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def label(self, element=None) -> str:
        """The output/provenance string, with the measured contraction where an element is
        named."""
        contraction = self.contraction_of(element).value if element is not None else "?"
        return "custom:{} [{}, {}, fit=cholesky, conditioning=unknown]".format(
            self.name, self.rel_treatment.value, contraction)

    def __str__(self) -> str:
        return "custom:{} ({}, {} elements)".format(
            self.name, self.rel_treatment.value, len(self._parsed))


def is_custom(basis) -> bool:
    """Whether a per-atom basis value is a user-supplied set rather than a registry name."""
    return isinstance(basis, CustomBasis)


#: How a custom set names itself wherever a *name* is what is carried — the output line, the
#: provenance, the per-label basis map stashed on a built molecule.
CUSTOM_PREFIX = "custom:"


def is_custom_name(name) -> bool:
    """Whether a basis *name* refers to a user-supplied set."""
    return isinstance(name, str) and name.startswith(CUSTOM_PREFIX)


def stash(basis: CustomBasis) -> Dict[str, object]:
    """A JSON-safe form of a custom set, for stashing on a built ``Mole``.

    ⚠ **Plain data only, and this is not tidiness**: the integral library's SCF checkpoint
    JSON-serializes the molecule's ``__dict__``, so an object parked there fails the whole
    calculation at the first checkpoint write — with a traceback from inside the JSON encoder
    that says nothing about basis sets. The same rule, for the same reason, governs the
    stashed reference configurations.
    """
    return {"name": basis.name, "relativistic_treatment": basis.relativistic_treatment,
            "notes": basis.notes,
            "shells": {k: v for k, v in sorted(basis._parsed.items())}}


def unstash(data: Dict[str, object]) -> CustomBasis:
    """Rebuild a :class:`CustomBasis` from :func:`stash`'s plain form."""
    return CustomBasis(data=dict(data["shells"]),
                       relativistic_treatment=str(data["relativistic_treatment"]),
                       name=str(data["name"]), notes=str(data.get("notes", "")))


__all__ = ["CUSTOM_PREFIX", "CustomBasis", "is_custom", "is_custom_name", "stash", "unstash"]
