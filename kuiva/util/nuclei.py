"""THE nuclear-moment table: spin, ``g`` factor and quadrupole moment per isotope.

One definition, because a nuclear moment is data and two copies of data drift. Everything that
needs to know what a nucleus *is* — the hyperfine field operator's isotope resolution, the
reported hyperfine coupling in MHz, the quadrupole term, the header of every stored property
file — reads this module. It is **plain data**: no PySCF, no integral library, no molecule, so
it sits below every layer that consumes it.

What is here and what is not
----------------------------
* ``I`` is stored as :attr:`NuclearMoment.twice_spin`, an **integer**, because half-integer
  spins are the common case and a float ``3.5`` invites a comparison that fails on the last
  bit. :attr:`NuclearMoment.spin` is the physical ``I``.
* ``g`` is the nuclear ``g`` factor in the convention ``mu = g_N mu_N I`` — dimensionless,
  signed, and **the sign matters**: it is the sign of every hyperfine splitting computed from
  it. It is *not* the gyromagnetic ratio and not the moment itself; the moment in nuclear
  magnetons is ``g * I``.
* ``Q`` is the electric quadrupole moment in **barn**, and it is ``None`` wherever this table
  does not carry a value whose **sign** has been verified. ⚠ That is deliberate and a
  consumer must refuse rather than substitute a magnitude: the sign of ``Q`` is the sign of
  every quadrupole splitting, and the published compilations disagree on the sign of several
  nuclei whose magnitude they agree on.
* Coverage is **curated, not exhaustive** — the elements this program targets (3d/4d/5d
  transition metals, lanthanides, the early actinides) and the nuclei that appear as their
  ligands. An element or an isotope the table does not carry is **refused**, naming
  :class:`NuclearMoment` as the way to state one explicitly. A table that guessed would put a
  wrong ``g`` into a file that outlives the session.
* An isotope with ``I = 0`` is listed where it is the element's most abundant nuclide, so that
  naming it is refused *by name* ("16O has no nuclear moment") rather than as an unknown
  isotope. It is never a default.

⚠ **The isotope chosen here does not change the Gaussian nuclear exponent.** The finite-nucleus
model (:mod:`kuiva.x2c.nuclear`) is built from the integral library's own main-isotope masses
and offers no per-atom override on the path molecules are built on, so a calculation on
``"163Dy"`` uses the charge distribution of the *main* Dy isotope. The difference is far below
the finite-nucleus effect itself; it is **recorded** in the provenance rather than reconciled,
because reconciling it silently would make two of the three nuclear inputs disagree with the
third.

References
----------
* N. J. Stone, "Table of Recommended Nuclear Magnetic Dipole Moments", IAEA INDC(NDS)-0794
  (2019), doi:10.61092/iaea.yjpc-cns6 — the magnetic dipole moments the ``g`` factors are
  derived from; and INDC(NDS)-0833 (2021) for the electric quadrupole moments.
* P. Pyykkö, "Year-2017 nuclear quadrupole moments", Mol. Phys. 116, 1328 (2018),
  doi:10.1080/00268976.2018.1426131 — the quadrupole moments and their signs.
* Natural abundances: J. Meija et al., "Isotopic compositions of the elements 2013",
  Pure Appl. Chem. 88, 293 (2016), doi:10.1515/pac-2015-0503.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .logging import get_logger

log = get_logger(__name__)

#: ⚠ **The constants this table is used with are NOT here** — the nuclear magneton, the
#: Hartree-to-MHz factor and the barn are CODATA values and :mod:`kuiva.util.units` is the one
#: table that holds them. Re-exporting them would give each a second import path, which is
#: precisely the sideways import that motivated a shared table in the first place.
__all__ = ["NuclearMoment", "ISOTOPES", "MOMENT_SOURCE", "QUADRUPOLE_SOURCE", "CUSTOM_SOURCE",
           "isotopes", "default_isotope", "resolve_isotope", "parse_isotope_label",
           "covered_elements"]

#: Provenance strings stored with a resolved moment, so a property file says where its nuclear
#: data came from without the reader having to trust the program's version number.
MOMENT_SOURCE = "Stone, IAEA INDC(NDS)-0794 (2019)"
QUADRUPOLE_SOURCE = "Pyykko, Mol. Phys. 116, 1328 (2018)"
CUSTOM_SOURCE = "user-supplied"


@dataclass(frozen=True)
class NuclearMoment:
    """One nucleus: its spin, ``g`` factor and quadrupole moment.

    Constructed either from the table (:func:`resolve_isotope`) or **by the user**, which is
    the escape hatch for a nuclear isomer, a revised moment, or an element this table does not
    cover::

        hyperfine={"Tb1": NuclearMoment(spin=1.5, g=1.343, quadrupole_barn=1.432)}

    Parameters
    ----------
    spin : float
        ``I``. Must be a non-negative multiple of one half; stored as
        :attr:`twice_spin` so no comparison depends on a float.
    g : float
        The nuclear ``g`` factor, ``mu = g mu_N I``. Signed.
    quadrupole_barn : float, optional
        ``Q`` in barn, **signed**. ``None`` means "not known here", and a consumer refuses
        rather than guessing (see the module docstring).
    symbol, mass_number : optional
        What this nucleus is, for labelling. Absent on a user-supplied moment, and the label
        then says so rather than inventing an isotope.
    abundance : float
        Natural abundance in per cent; ``0.0`` for a synthetic nuclide and for a
        user-supplied moment. Used only to choose a **default** isotope.
    source : str
        Where the numbers came from, carried into every stored file's header.
    """

    spin: float
    g: float
    quadrupole_barn: Optional[float] = None
    symbol: str = ""
    mass_number: Optional[int] = None
    abundance: float = 0.0
    source: str = CUSTOM_SOURCE

    def __post_init__(self) -> None:
        two_i = 2.0 * float(self.spin)
        if two_i < 0.0 or abs(two_i - round(two_i)) > 1e-9:
            raise ValueError(
                "a nuclear spin must be a non-negative multiple of one half; got I = {!r}"
                .format(self.spin))
        # ⚠ Normalised to the exact half-integer so `spin` never carries a representation
        # error into a (2I+1) dimension count or a comparison against a tabulated value.
        object.__setattr__(self, "spin", round(two_i) / 2.0)
        if self.mass_number is not None:
            object.__setattr__(self, "mass_number", int(self.mass_number))
        object.__setattr__(self, "g", float(self.g))
        if self.quadrupole_barn is not None:
            object.__setattr__(self, "quadrupole_barn", float(self.quadrupole_barn))
        object.__setattr__(self, "abundance", float(self.abundance))

    @property
    def twice_spin(self) -> int:
        """``2I`` as an integer — the form every dimension count and comparison uses."""
        return int(round(2.0 * self.spin))

    @property
    def multiplicity(self) -> int:
        """``2I + 1``: the dimension of this nucleus's pseudospin site."""
        return self.twice_spin + 1

    @property
    def magnetic(self) -> bool:
        """Whether this nucleus has a magnetic moment at all (``I > 0``)."""
        return self.twice_spin > 0

    @property
    def has_quadrupole(self) -> bool:
        """Whether a *signed* ``Q`` is available **and** ``I`` admits one (``I >= 1``)."""
        return self.twice_spin >= 2 and self.quadrupole_barn is not None

    @property
    def label(self) -> str:
        """``"159Tb"``, or a self-describing label for a user-supplied moment.

        ⚠ **Contains no whitespace, in either form.** This string is written into a column of
        the ``[NUCLEI]`` table of every stored property file, which is parsed by splitting on
        whitespace, so a space inside it would silently shift every column after it.
        """
        if self.symbol and self.mass_number is not None:
            return "{}{}".format(self.mass_number, self.symbol)
        return "custom(I={:g},g={:.6g})".format(self.spin, self.g)

    @property
    def moment_nuclear_magnetons(self) -> float:
        """``mu / mu_N = g I`` — the quantity the compilations actually tabulate."""
        return self.g * self.spin

    def as_dict(self) -> Dict[str, object]:
        """The provenance record: plain JSON-able data, for a stored file's header."""
        return {
            "label": self.label,
            "element": self.symbol or None,
            "mass_number": self.mass_number,
            "twice_spin": self.twice_spin,
            "g": self.g,
            "quadrupole_barn": self.quadrupole_barn,
            "abundance_percent": self.abundance,
            "source": self.source,
        }

    def __repr__(self) -> str:
        q = "None" if self.quadrupole_barn is None else "{:.5g} b".format(self.quadrupole_barn)
        return "NuclearMoment({}, I={:g}, g={:.6g}, Q={})".format(
            self.label, self.spin, self.g, q)


# --- the table -------------------------------------------------------------------------------
#
# ``(mass number, 2I, g, Q [barn] or None, natural abundance [%])``, ascending in mass number.
# ⚠ A ``Q`` of ``None`` means "no verified signed value here", not "zero" — see the module
# docstring. An entry with ``2I = 0`` is present only so that naming it is refused informatively.
_TABLE: Dict[str, Tuple[Tuple[int, int, float, Optional[float], float], ...]] = {
    "H":  ((1, 1, 5.58569468, None, 99.9885), (2, 2, 0.8574382, 0.002862, 0.0115)),
    "B":  ((10, 6, 0.6001, 0.08459, 19.9), (11, 3, 1.7924326, 0.040591, 80.1)),
    "C":  ((12, 0, 0.0, None, 98.93), (13, 1, 1.4048236, None, 1.07)),
    "N":  ((14, 2, 0.403761, 0.020443, 99.636), (15, 1, -0.5663784, None, 0.364)),
    "O":  ((16, 0, 0.0, None, 99.757), (17, 5, -0.757516, -0.02558, 0.038)),
    "F":  ((19, 1, 5.257736, None, 100.0),),
    "Na": ((23, 3, 1.478348, 0.1041, 100.0),),
    "Al": ((27, 5, 1.4566028, 0.1466, 100.0),),
    "Si": ((28, 0, 0.0, None, 92.223), (29, 1, -1.11058, None, 4.685)),
    "P":  ((31, 1, 2.2632, None, 100.0),),
    "S":  ((32, 0, 0.0, None, 94.99), (33, 3, 0.429214, -0.0678, 0.75)),
    "Cl": ((35, 3, 0.5479162, -0.08165, 75.76), (37, 3, 0.4560824, -0.06435, 24.24)),
    "K":  ((39, 3, 0.26098, None, 93.258),),
    "Ca": ((40, 0, 0.0, None, 96.941), (43, 7, -0.37637, None, 0.135)),
    "Sc": ((45, 7, 1.35899, None, 100.0),),
    "Ti": ((47, 5, -0.31539, None, 7.44), (49, 7, -0.315477, None, 5.41)),
    "V":  ((51, 7, 1.47106, None, 99.75),),
    "Cr": ((53, 3, -0.31636, None, 9.501),),
    "Mn": ((55, 5, 1.3813, 0.330, 100.0),),
    "Fe": ((56, 0, 0.0, None, 91.754), (57, 1, 0.1809, None, 2.119)),
    "Co": ((59, 7, 1.322, 0.42, 100.0),),
    "Ni": ((61, 3, -0.50001, None, 1.1399),),
    "Cu": ((63, 3, 1.4824, -0.220, 69.15), (65, 3, 1.5878, -0.204, 30.85)),
    "Zn": ((67, 5, 0.350192, None, 4.10),),
    "Ga": ((69, 3, 1.34439, 0.171, 60.108), (71, 3, 1.70818, 0.107, 39.892)),
    "As": ((75, 3, 0.95965, 0.314, 100.0),),
    "Se": ((77, 1, 1.07008, None, 7.63),),
    "Br": ((79, 3, 1.404267, 0.313, 50.69), (81, 3, 1.513708, 0.262, 49.31)),
    "Y":  ((89, 1, -0.2748308, None, 100.0),),
    "Zr": ((91, 5, -0.521448, None, 11.22),),
    "Nb": ((93, 9, 1.3712, None, 100.0),),
    "Mo": ((95, 5, -0.3657, None, 15.92),),
    "Ru": ((101, 5, -0.288, None, 17.06),),
    "Rh": ((103, 1, -0.1768, None, 100.0),),
    "Pd": ((105, 5, -0.257, None, 22.33),),
    "Ag": ((107, 1, -0.22714, None, 51.839), (109, 1, -0.26115, None, 48.161)),
    "Cd": ((111, 1, -1.18977, None, 12.80), (113, 1, -1.244602, None, 12.22)),
    "In": ((113, 9, 1.2286, None, 4.29), (115, 9, 1.2313, None, 95.71)),
    "Sn": ((117, 1, -2.00208, None, 7.68), (119, 1, -2.09456, None, 8.59)),
    "Sb": ((121, 5, 1.3454, None, 57.21), (123, 7, 0.72851, None, 42.79)),
    "Te": ((125, 1, -1.7770102, None, 7.07),),
    "I":  ((127, 5, 1.12531, -0.696, 100.0),),
    "Cs": ((133, 7, 0.7377214, -0.00343, 100.0),),
    "Ba": ((135, 3, 0.55863, None, 6.592), (137, 3, 0.62491, 0.245, 11.232)),
    "La": ((139, 7, 0.795156, None, 99.911),),
    "Ce": ((140, 0, 0.0, None, 88.450),),
    "Pr": ((141, 5, 1.7102, None, 100.0),),
    "Nd": ((143, 7, -0.3043, None, 12.2),),
    "Sm": ((147, 7, -0.232, None, 14.99),),
    "Eu": ((151, 5, 1.3887, 0.903, 47.81), (153, 5, 0.6134, 2.412, 52.19)),
    "Gd": ((155, 3, -0.1715, 1.27, 14.80), (157, 3, -0.2265, 1.35, 15.65)),
    "Tb": ((159, 3, 1.343, 1.432, 100.0),),
    "Dy": ((161, 5, -0.192, 2.507, 18.91), (163, 5, 0.2691, 2.648, 24.90)),
    "Ho": ((165, 7, 1.668, 3.58, 100.0),),
    "Er": ((167, 7, -0.1611, 3.57, 22.93),),
    "Tm": ((169, 1, -0.462, None, 100.0),),
    "Yb": ((171, 1, 0.98734, None, 14.28), (173, 5, -0.2592, 2.80, 16.13)),
    "Lu": ((175, 7, 0.6378, 3.49, 97.401),),
    "Hf": ((177, 7, 0.2267, None, 18.60),),
    "Ta": ((181, 7, 0.67729, None, 99.988),),
    "W":  ((183, 1, 0.2355695, None, 14.31),),
    "Re": ((185, 5, 1.2748, None, 37.40), (187, 5, 1.2879, None, 62.60)),
    "Os": ((187, 1, 0.1293038, None, 1.96),),
    "Ir": ((191, 3, 0.1005, None, 37.3), (193, 3, 0.1091, None, 62.7)),
    "Pt": ((195, 1, 1.2190, None, 33.78),),
    "Au": ((197, 3, 0.097164, 0.547, 100.0),),
    "Hg": ((199, 1, 1.011771, None, 16.87), (201, 3, -0.373484, None, 13.18)),
    "Tl": ((203, 1, 3.24451, None, 29.52), (205, 1, 3.2764292, None, 70.48)),
    "Pb": ((207, 1, 1.18512, None, 22.1), (208, 0, 0.0, None, 52.4)),
    "Bi": ((209, 9, 0.9134, -0.516, 100.0),),
    "Th": ((229, 5, 0.18, None, 0.0), (232, 0, 0.0, None, 100.0)),
    "U":  ((235, 7, -0.109, None, 0.7204), (238, 0, 0.0, None, 99.2742)),
    "Np": ((237, 5, 1.256, None, 0.0),),
    "Pu": ((239, 1, 0.406, None, 0.0),),
    "Am": ((243, 5, 0.6, None, 0.0),),
}


def _build() -> Dict[str, Tuple[NuclearMoment, ...]]:
    table: Dict[str, Tuple[NuclearMoment, ...]] = {}
    for symbol, rows in _TABLE.items():
        table[symbol] = tuple(
            NuclearMoment(spin=two_i / 2.0, g=g, quadrupole_barn=q, symbol=symbol,
                          mass_number=a, abundance=ab,
                          source=(MOMENT_SOURCE if q is None else
                                  MOMENT_SOURCE + "; Q: " + QUADRUPOLE_SOURCE))
            for a, two_i, g, q, ab in rows)
    return table


#: ``element symbol -> tuple of NuclearMoment``, ascending in mass number.
ISOTOPES: Dict[str, Tuple[NuclearMoment, ...]] = _build()

_LABEL = re.compile(r"^(?:([0-9]{1,3})\s*-?\s*([A-Za-z]{1,2})|([A-Za-z]{1,2})\s*-?\s*([0-9]{1,3}))$")


def covered_elements() -> Tuple[str, ...]:
    """The elements this table carries, alphabetically — what a refusal quotes."""
    return tuple(sorted(ISOTOPES))


def parse_isotope_label(text: str) -> Tuple[Optional[str], int]:
    """``"159Tb"`` / ``"Tb159"`` / ``"Tb-159"`` / ``"159"`` -> ``(symbol or None, mass number)``.

    The symbol is returned so a caller can refuse a label naming a *different* element from
    the atom it was attached to, rather than silently resolving the mass number against the
    atom's own element — ``{"Tb1": "163Dy"}`` is a mistake, not a request.
    """
    s = str(text).strip()
    if s.isdigit():
        return None, int(s)
    m = _LABEL.match(s)
    if m is None:
        raise ValueError(
            "cannot read {!r} as an isotope: write the mass number and the element in either "
            "order ('159Tb', 'Tb159', 'Tb-159'), or the mass number alone ('159')".format(text))
    if m.group(1) is not None:
        return m.group(2).capitalize(), int(m.group(1))
    return m.group(3).capitalize(), int(m.group(4))


def isotopes(symbol: str) -> Tuple[NuclearMoment, ...]:
    """Every tabulated isotope of ``symbol``, or a refusal naming the escape hatch."""
    key = str(symbol).strip().capitalize()
    try:
        return ISOTOPES[key]
    except KeyError:
        raise ValueError(
            "no nuclear moments are tabulated for {}. The table is curated rather than "
            "exhaustive ({} elements: {}); state the nucleus explicitly with "
            "NuclearMoment(spin=..., g=..., quadrupole_barn=...) instead — a guessed g factor "
            "would be written into a property file that outlives this session."
            .format(key, len(ISOTOPES), ", ".join(covered_elements()))) from None


def default_isotope(symbol: str) -> NuclearMoment:
    """The most abundant isotope of ``symbol`` with ``I > 0`` — what ``hyperfine=True`` means.

    ⚠ It is "most abundant **among those with a moment**", not "most abundant": the most
    abundant nuclide of carbon, oxygen, silicon and half the transition metals has ``I = 0``
    and no hyperfine interaction at all, so choosing it would make the request vacuous.

    Refuses when the element has no tabulated magnetic isotope, and **warns** when the one it
    picks does not occur naturally — a synthetic nuclide is a legitimate choice and a
    surprising default.
    """
    entries = [m for m in isotopes(symbol) if m.magnetic]
    if not entries:
        listed = ", ".join(m.label for m in isotopes(symbol))
        raise ValueError(
            "{} has no tabulated isotope with a nuclear moment (this table lists {}), so "
            "there is nothing for a hyperfine interaction to couple to. If you mean a "
            "specific nuclide, name it or pass a NuclearMoment."
            .format(str(symbol).capitalize(), listed))
    best = max(entries, key=lambda m: (m.abundance, -(m.mass_number or 0)))
    if best.abundance <= 0.0:
        log.warning(
            "the default hyperfine isotope for %s is %s, which does not occur naturally "
            "(no tabulated %s isotope with I > 0 does). That is a real nuclide and a real "
            "calculation, but it is not the isotope a natural-abundance sample contains -- "
            "name the isotope explicitly if you meant another one.",
            str(symbol).capitalize(), best.label, str(symbol).capitalize())
    return best


def resolve_isotope(symbol: str, spec: object) -> NuclearMoment:
    """Resolve one atom's hyperfine request against its element.

    ``spec`` is ``True`` (the :func:`default_isotope`), a mass number (``163``, ``"163"``), an
    isotope label (``"163Dy"``, ``"Dy-163"``), or a :class:`NuclearMoment` used as given.

    Three refusals, each because the silent alternative produces a plausible wrong number:
    an isotope this table does not carry (rather than the nearest one); a label naming a
    different element from the atom it was attached to (rather than reading the mass number
    against the atom's own element); and an isotope with ``I = 0`` (rather than a hyperfine
    operator that is identically zero and says nothing about why).
    """
    element = str(symbol).strip().capitalize()
    if isinstance(spec, NuclearMoment):
        return spec
    if spec is True:
        return default_isotope(element)
    if spec is False or spec is None:
        raise ValueError(
            "a hyperfine request for {} is True, a mass number, an isotope label or a "
            "NuclearMoment; {!r} selects nothing. Leave the atom out of the mapping instead."
            .format(element, spec))
    # ``__index__`` rather than ``isinstance(spec, int)``: a mass number arriving from an array
    # or a parsed table is a NumPy integer, which is not a Python ``int`` and would otherwise be
    # refused as "not a mass number" while printing as one.
    if hasattr(spec, "__index__") and not isinstance(spec, bool):
        named, mass = None, int(spec)
    elif isinstance(spec, str):
        named, mass = parse_isotope_label(spec)
    else:
        raise TypeError(
            "a hyperfine request for {} must be True, a mass number, an isotope label like "
            "'163Dy', or a NuclearMoment; got {!r}".format(element, spec))
    if named is not None and named != element:
        raise ValueError(
            "the isotope {!r} names {} but the atom it is attached to is {}; a label is "
            "refused rather than reinterpreted as mass number {} of {}."
            .format(spec, named, element, mass, element))
    table = isotopes(element)
    for moment in table:
        if moment.mass_number == mass:
            if not moment.magnetic:
                raise ValueError(
                    "{} has I = 0: it has no nuclear magnetic moment, so there is no "
                    "hyperfine interaction to compute for it. The {} isotopes with a moment "
                    "in this table are {}."
                    .format(moment.label, element,
                            ", ".join(m.label for m in table if m.magnetic) or "none"))
            return moment
    raise ValueError(
        "no nuclear moment is tabulated for mass number {} of {}; this table carries {}. "
        "State it explicitly with NuclearMoment(spin=..., g=..., quadrupole_barn=...) if you "
        "need that nuclide.".format(mass, element,
                                    ", ".join(m.label for m in table)))
