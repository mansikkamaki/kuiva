"""The X2C method surface: named Hamiltonians, the two axes behind them, and provenance.

A user picks a Hamiltonian with **one** name; internally that name is a point
on two independent axes, and both are recorded so that a stored result is never ambiguous
about which of several Hamiltonians produced it.

The two axes
------------
``decoupling`` — how the one-electron X2C transformation is obtained:

``"1e"``
    Exact molecular one-electron X2C, via PySCF. **The default**, and the route every
    committed reference number in this project was produced with.
``"1e-dlu"``
    The local (atom-blocked) DLU approximation of :mod:`kuiva.x2c.local`. The bottom rung of
    the cost ladder, for when the exact decoupling is prohibitive.
``"atom1e"``
    PySCF's block-diagonal ``X`` with a **molecular** ``R``. ⚠ Not DLU, and not cheaper in
    scaling — the ``O(nao^3)`` work stays in ``R``. It exists because it is what makes the
    one- and two-electron decouplings consistent (see :mod:`kuiva.amf.correction`).

``screening`` — the two-electron picture change: ``"none"``, ``"x2camf"`` (the default,
the default), ``"x2camf-external"``, and ``"mmf"``. The first three are owned by
:mod:`kuiva.amf` and named here only as strings, which is what keeps this module free of any
dependency on that package.

⚠ **X2CAMF and X2C-mmf are the same subtraction, atomically and molecularly**, so they belong on
one axis and cannot be combined — a design that removes a double-counting failure mode rather
than guarding against it. ``"mmf"`` is a **benchmark tool**: `production=False`, never a
default, and it warns at the point of selection.

⚠ **Why there is no separate name for "Kuiva's own exact decoupling"**
----------------------------------------------------------------------
There are two implementations of the exact transformation — PySCF's, which ``"1e"`` uses, and
Kuiva's own, which the DLU path is built from — and they **differ by up to 2.4e-07 relative on
a heavy element**, because Kuiva projects the four-component metric and PySCF does not.
Comparing ``"1e"`` against ``"1e-dlu"`` therefore measures the DLU approximation *plus* that
difference, which would be a confound sitting right in the range being measured.

The fix is that ``"1e-dlu"`` with ``partition="single"`` **is** Kuiva's exact decoupling — one
fragment covering the whole basis, through identical code. So the like-for-like reference needs
no new method name, and the DLU error is isolated by construction rather than by argument. That
is the comparison any accuracy claim about DLU must be built on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

#: How the one-electron decoupling is obtained. See the module docstring.
DECOUPLINGS = ("1e", "atom1e", "1e-dlu")

#: Two-electron picture-change options. ``"none"``, ``"x2camf"`` and ``"x2camf-external"`` are
#: owned by :mod:`kuiva.amf` and named here as strings; ``"mmf"`` is the molecular mean field of
#::func:`kuiva.interface.pyscf_bridge.molecular_mean_field`.
#:
#: ⚠ **They are mutually exclusive by construction, and that is not an accident of the API.**
#: X2CAMF and X2C-mmf are the *same* subtraction — one on atoms, one on the molecule — so
#: applying both would double-count the two-electron picture change exactly. Making mmf a value
#: on this axis rather than a separate knob means there is no combination to refuse.
SCREENINGS = ("none", "x2camf", "x2camf-external", "mmf")

#: Partitions the local decoupling may use. ``"atoms"`` is DLU proper; ``"single"`` is the
#: exact transformation expressed through the same code (see the module docstring).
PARTITIONS = ("atoms", "single")


@dataclass(frozen=True)
class HamiltonianMethod:
    """A named Hamiltonian: a point on the two axes, plus how it may be used.

    ``production`` is not decoration. A method that is a benchmark tool must say so at the
    point of selection, because the whole failure mode is a result produced with it being
    mistaken later for a standard one.
    """

    name: str
    decoupling: str
    screening: str
    production: bool = True
    note: str = ""

    def __str__(self) -> str:
        return self.name


#: The canonical named methods. Anything reachable by the axes but not named here is a valid
#: combination that simply has no established name — :func:`resolve` synthesizes one rather
#: than refusing, so that provenance is never empty.
METHODS: Tuple[HamiltonianMethod, ...] = (
    HamiltonianMethod(
        "X2C-1e", "1e", "none",
        note="one-electron X2C only; atomic j-splittings come out 5-30% too large"),
    HamiltonianMethod(
        "X2C-AMF", "1e", "x2camf",
        note="exact one-electron decoupling plus the atomic mean-field two-electron "
             "picture change"),
    HamiltonianMethod(
        "X2C-1e-DLU", "1e-dlu", "none",
        note="local decoupling, no two-electron picture change; doubly approximate"),
    HamiltonianMethod(
        "X2C-AMF-DLU", "1e-dlu", "x2camf",
        note="the cheap end of the ladder: local decoupling with the atomic mean field"),
    HamiltonianMethod(
        "X2C-mmf", "1e", "mmf", production=False,
        note="EXPERIMENTAL BENCHMARK ONLY: the two-electron picture change from a full "
             "four-component SCF on the whole molecule, with no atomic approximation. Costs "
             "what X2CAMF exists to avoid and grows as the fourth power of the basis"),
)

#: ⚠ **The default, and it is a user decision.** X2CAMF is intended to be used
#: everywhere; DLU is the escape hatch for when the exact decoupling is prohibitive, not a
#: cheaper default to drift into.
DEFAULT_METHOD = "X2C-AMF"

_BY_NAME: Dict[str, HamiltonianMethod] = {m.name.lower(): m for m in METHODS}


def known_methods() -> Tuple[str, ...]:
    """The canonical method names, in cost order."""
    return tuple(m.name for m in METHODS)


def resolve(method: Optional[str] = None, *, decoupling: Optional[str] = None,
            screening: Optional[str] = None) -> HamiltonianMethod:
    """Resolve a method name and/or explicit axes into one :class:`HamiltonianMethod`.

    ``method`` is the user-facing knob; ``decoupling`` and ``screening`` address the axes
    directly. Supplying both is allowed **only if they agree** — a name and an axis that
    contradict each other is a request whose intent cannot be guessed, and silently letting
    one win would mean a calculation ran with a Hamiltonian nobody asked for.

    Raises ``ValueError`` on an unknown name, an unknown axis value, or a contradiction.
    """
    if method is not None:
        key = str(method).strip().lower()
        if key not in _BY_NAME:
            raise ValueError(
                "unknown Hamiltonian method {!r}; expected one of {}. To reach a combination "
                "with no canonical name, set decoupling= and screening= directly."
                .format(method, ", ".join(known_methods())))
        resolved = _BY_NAME[key]
        for axis, value, chosen in (("decoupling", decoupling, resolved.decoupling),
                                    ("screening", screening, resolved.screening)):
            if value is not None and str(value) != chosen:
                raise ValueError(
                    "method={!r} means {}={!r}, but {}={!r} was also given. Pass one or the "
                    "other: a name and an axis that contradict each other cannot be resolved "
                    "into the Hamiltonian that was intended.".format(
                        method, axis, chosen, axis, value))
        return resolved

    decoupling = "1e" if decoupling is None else str(decoupling)
    screening = "x2camf" if screening is None else str(screening)
    if decoupling not in DECOUPLINGS:
        raise ValueError("unknown decoupling {!r}; expected one of {}".format(
            decoupling, ", ".join(DECOUPLINGS)))
    if screening not in SCREENINGS:
        raise ValueError("unknown screening {!r}; expected one of {}".format(
            screening, ", ".join(SCREENINGS)))

    for candidate in METHODS:
        if candidate.decoupling == decoupling and candidate.screening == screening:
            return candidate
    # A valid combination with no established name. Synthesized rather than refused, so that
    # provenance always carries something a reader can act on.
    return HamiltonianMethod(
        "X2C[{}/{}]".format(decoupling, screening), decoupling, screening,
        note="combination without a canonical name")


@dataclass(frozen=True)
class DecouplingRecord:
    """What one-electron decoupling produced a Hamiltonian — the decoupling half of provenance.

    ⚠ Like :class:`kuiva.amf.correction.ScreeningRecord`, this is a **contract with stored
    data**: it goes into the property dump's header, so that a stored property matrix is
    never ambiguous about which of several Hamiltonians produced it. Add fields; do not rename
    or repurpose them.

    ``DecouplingRecord()`` — all defaults — describes the exact molecular one-electron X2C
    that every committed reference in this project was produced with.
    """

    decoupling: str = "1e"
    implementation: str = "pyscf"
    partition: str = ""
    source: str = ""
    fragments: int = 0
    #: ``max |X|`` per fragment, the conditioning diagnostic. Empty for the exact path,
    #: where it is a single global number with no fragment to attribute it to.
    block_scales: Dict[str, float] = field(default_factory=dict)

    @property
    def local(self) -> bool:
        """Whether the decoupling was approximated block-diagonally (DLU)."""
        return self.decoupling == "1e-dlu" and self.partition != "single"

    @property
    def max_block_scale(self) -> float:
        return max(self.block_scales.values()) if self.block_scales else 0.0

    def as_dict(self) -> Dict[str, object]:
        """JSON-serializable form, for the property dump header and reference records."""
        return {
            "decoupling": self.decoupling,
            "implementation": self.implementation,
            "partition": self.partition,
            "source": self.source,
            "fragments": int(self.fragments),
            "block_scales": {str(k): float(v) for k, v in sorted(self.block_scales.items())},
        }

    def report(self, logger=None) -> None:
        """The provenance output block for the one-electron decoupling."""
        from ..util import output as out
        from ..util.logging import get_logger

        logger = logger or get_logger(__name__)
        if not self.local:
            detail = ("exact molecular decoupling"
                      if self.decoupling != "atom1e"
                      else "atom-blocked X, molecular R (not DLU)")
            out.entry(logger, "one-electron decoupling", self.decoupling,
                      note="{}, {}".format(detail, self.implementation))
            return
        out.entries(logger, [
            ("one-electron decoupling", self.decoupling),
            ("local partition", "{} ({} fragments)".format(self.partition, self.fragments)),
            ("local problem source", self.source),
            ("largest local max|X|", self.max_block_scale, "", "", "{:.3f}"),
        ])
        logger.warning("this Hamiltonian uses the LOCAL (DLU) decoupling, an approximation to "
                       "the exact X2C transformation, intended for systems where the exact "
                       "decoupling is prohibitive. Measured at the state level (SA-CASSCF, "
                       "x2c-SVPall-2c, d1 and f1 ligand fields): splittings move by <= 0.6 "
                       "cm^-1 and <= 0.1%, principal g values by <= 2e-4 relative. The "
                       "TRANSVERSE g of a strongly axial doublet is NOT protected - measured "
                       "+6e-4 absolute (+6% of itself) on a g_perp of 0.01 - so check that "
                       "number against decoupling_options={'partition': 'single'} before "
                       "quoting it.")


__all__ = ["DECOUPLINGS", "DEFAULT_METHOD", "METHODS", "PARTITIONS", "SCREENINGS",
           "DecouplingRecord", "HamiltonianMethod", "known_methods", "resolve"]
