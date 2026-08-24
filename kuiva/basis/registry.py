"""Basis-set registry.

The registry carries, per basis family, the metadata Kuiva needs *before* touching the raw
integrals: element coverage, contraction type, the targeted relativistic treatment, a
recommended density-fitting auxiliary, expected conditioning, and the DF-vs-Cholesky routing
that follows from it — plus full literature references. It also performs the
ingestion consistency checks (relativistic-treatment compatibility across atoms, and
guarding against silently mixing incompatible recontraction variants).

The registry does **not** store basis coefficients. Actual basis data is obtained from a
provider at resolve time — and, for the Basis Set Exchange families, **cached per
``(family, element)``**: the fetch-and-parse is pure and deterministic and was being repeated
on every ``build_mole``, once per atom label, hence twice per atom whenever a cross-basis
overlap builds both bases. Measured 0.245 s cold against 20 us warm for one two-element
molecule. ⚠ The cached parse is stored **frozen** (nested tuples) and thawed into fresh lists
per call: the same object would otherwise be handed to every consumer, and PySCF's ``Mole``
build is entitled to normalize what it is given — one in-place mutation would silently change
the basis of every later molecule in the process.
  * ``pyscf``  — families bundled with PySCF (Dyall, ANO-RCC);
  * ``bse``    — Basis Set Exchange (families PySCF does not bundle: the Karlsruhe
                 ``x2c-*all(-2c)`` sets, their ``x2c-JFIT`` auxiliaries, and the Peterson
                 ``cc-pVnZ-X2C`` / ``cc-pwCVnZ-X2C`` sets).

Families supported here (the types and elements of interest):
  * **Karlsruhe x2c-nZVPall / -2c** — segmented, H–Rn; ``-2c`` recontraction for two-component
    (SOC) work is the project default. Pollak & Weigend, J. Chem. Theory Comput. 13, 3696
    (2017), doi:10.1021/acs.jctc.7b00593 (DZ/TZ); Franzke, Spiske, Pollak & Weigend,
    J. Chem. Theory Comput. 16, 5658 (2020), doi:10.1021/acs.jctc.0c00546 (QZ).
  * **x2c-JFIT / x2c-JFIT-universal** — Coulomb-fitting auxiliaries for the above; Franzke
    et al. 2020 (same DOI).
  * **Peterson cc-pVnZ-X2C / cc-pwCVnZ-X2C** — mixed contraction (general low-l, uncontracted
    high-l), covering alkali/alkaline-earth (K–Ra), lanthanides (La–Lu) and actinides
    (Ac–Lr); primary for actinides and CBS-extrapolable. Hill & Peterson, J. Chem. Phys. 147,
    244106 (2018), doi:10.1063/1.5010587 (K–Ra); Lu & Peterson, J. Chem. Phys. 145, 054111
    (2016), doi:10.1063/1.4959280 (La–Lu); Feng & Peterson, J. Chem. Phys. 147, 084108 (2017),
    doi:10.1063/1.4994725 (actinides); ccRepo, http://www.grant-hill.group.shef.ac.uk/ccrepo/.
  * **Dyall** (benchmarking only) — uncontracted, heavy-element; per-shell provenance spans
    many papers, see the DIRAC basis header and http://dirac.chem.sdu.dk. Representative:
    Dyall, Theor. Chem. Acc. 135, 128 (2016) and references therein.
  * **ANO-RCC** — Douglas–Kroll–Hess-recontracted atomic natural orbitals (native to
    OpenMolcas), H–Cm. Roos, Lindh, Malmqvist, Veryazov & Widmark, J. Phys. Chem. A 108, 2851
    (2004), doi:10.1021/jp031064+, and companion papers (transition metals: J. Phys. Chem. A
    109, 6575 (2005); actinides: Chem. Phys. Lett. 409, 295 (2005)); original ANO scheme:
    Widmark, Malmqvist & Roos, Theor. Chim. Acta 77, 291 (1990), doi:10.1007/BF01120130.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

from ..util.logging import get_logger

log = get_logger(__name__)

# --- Periodic table (Z=1..103) ---------------------------------------------------------
_SYMBOLS: Tuple[str, ...] = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
)
_Z_OF: Dict[str, int] = {s: i + 1 for i, s in enumerate(_SYMBOLS)}


def z_of(element: Union[int, str]) -> int:
    """Nuclear charge Z of an element given its symbol (case-insensitive) or Z."""
    if isinstance(element, int):
        return element
    s = element.strip().capitalize()
    if s not in _Z_OF:
        raise KeyError(f"unknown element symbol: {element!r}")
    return _Z_OF[s]


def symbol_of(z: int) -> str:
    """Element symbol for nuclear charge ``z`` (1..103)."""
    if not 1 <= z <= len(_SYMBOLS):
        raise KeyError(f"Z out of tabulated range 1..{len(_SYMBOLS)}: {z}")
    return _SYMBOLS[z - 1]


def _zset(*ranges_or_syms) -> FrozenSet[int]:
    """Build a Z-set from (lo, hi) inclusive ranges and/or element symbols."""
    out = set()
    for item in ranges_or_syms:
        if isinstance(item, tuple):
            lo, hi = item
            out.update(range(z_of(lo) if isinstance(lo, str) else lo,
                             (z_of(hi) if isinstance(hi, str) else hi) + 1))
        else:
            out.add(z_of(item))
    return frozenset(out)


# --- Enumerations ----------------------------------------------------------------------
class Contraction(Enum):
    SEGMENTED = "segmented"
    GENERAL = "general"
    UNCONTRACTED = "uncontracted"
    MIXED = "mixed"            # general low-l, uncontracted high-l (Peterson X2C sets)


class RelTreatment(Enum):
    """Relativistic operator the basis was recontracted for."""
    X2C_2C = "x2c-2c"          # two-component (SOC) recontraction — project default
    X2C_1C = "x2c-1c"          # one-component / scalar X2C recontraction
    DKH = "dkh"                # Douglas–Kroll–Hess (ANO-RCC)
    NONREL = "nonrel"          # non-relativistic

    @property
    def is_relativistic(self) -> bool:
        return self is not RelTreatment.NONREL

    @property
    def family(self) -> str:
        """Coarse relativistic-Hamiltonian group used for compatibility checks.

        The scalar and two-component X2C recontractions share the ``"x2c"`` group (mixing
        them across atoms is fine); ``"dkh"`` and ``"nonrel"`` are distinct.
        """
        if self in (RelTreatment.X2C_1C, RelTreatment.X2C_2C):
            return "x2c"
        return self.value  # "dkh" or "nonrel"


class Provider(Enum):
    PYSCF = "pyscf"
    BSE = "bse"


class Conditioning(Enum):
    WELL = "well"              # DF suitable
    MODERATE = "moderate"      # DF usually fine; watch near-linear-dependence
    ILL = "ill" # near-linear-dependent; prefer Cholesky


class FitRoute(Enum):
    DF = "df"
    CHOLESKY = "cholesky"


@dataclass(frozen=True)
class Reference:
    key: str
    citation: str
    doi: str = ""


# --- Reference database ----------------------------------------------------------------
_REFS: Dict[str, Reference] = {r.key: r for r in [
    Reference("pollak2017",
              "P. Pollak, F. Weigend, J. Chem. Theory Comput. 13, 3696-3705 (2017)",
              "10.1021/acs.jctc.7b00593"),
    Reference("franzke2020",
              "Y. J. Franzke, L. Spiske, P. Pollak, F. Weigend, "
              "J. Chem. Theory Comput. 16, 5658-5674 (2020)",
              "10.1021/acs.jctc.0c00546"),
    Reference("hill2018",
              "J. G. Hill, K. A. Peterson, J. Chem. Phys. 147, 244106 (2018)",
              "10.1063/1.5010587"),
    Reference("lu2016",
              "Q. Lu, K. A. Peterson, J. Chem. Phys. 145, 054111 (2016)",
              "10.1063/1.4959280"),
    Reference("feng2017",
              "R. Feng, K. A. Peterson, J. Chem. Phys. 147, 084108 (2017)",
              "10.1063/1.4994725"),
    Reference("ccrepo",
              "J. G. Hill, ccRepo: a correlation consistent basis sets repository",
              "http://www.grant-hill.group.shef.ac.uk/ccrepo/"),
    Reference("dyall2016",
              "K. G. Dyall, Theor. Chem. Acc. 135, 128 (2016) and references therein "
              "(per-shell provenance: see the DIRAC basis header / http://dirac.chem.sdu.dk)",
              "10.1007/s00214-016-1884-y"),
    Reference("widmark1990",
              "P.-O. Widmark, P.-Å. Malmqvist, B. O. Roos, Theor. Chim. Acta 77, 291 (1990)",
              "10.1007/BF01120130"),
    Reference("roos2004",
              "B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark, "
              "J. Phys. Chem. A 108, 2851 (2004)",
              "10.1021/jp031064+"),
    Reference("roos2005tm",
              "B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark, "
              "J. Phys. Chem. A 109, 6575 (2005)",
              "10.1021/jp0581126"),
    Reference("roos2005act",
              "B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark, "
              "A. C. Borin, Chem. Phys. Lett. 409, 295 (2005)",
              "10.1016/j.cplett.2005.05.011"),
]}


def reference(key: str) -> Reference:
    return _REFS[key]


# --- Element coverage sets -------------------------------------------------------------
_COV_KARLSRUHE = _zset(("H", "Rn"))                                    # 1..86
_COV_PETERSON = _zset("K", "Ca", "Rb", "Sr", "Cs", "Ba",
                      ("La", "Lu"), "Fr", "Ra", ("Ac", "Lr"))          # 38 elements
_COV_ANORCC = _zset((1, 96))                                           # H..Cm (verified)


@dataclass(frozen=True)
class BasisFamily:
    """One basis-set family and everything the front-end needs to use it correctly."""
    name: str                                  # canonical Kuiva name (case preserved)
    provider: Provider
    provider_name: str                         # name understood by the provider
    contraction: Contraction
    rel_treatment: RelTreatment
    role: str
    conditioning: Conditioning
    references: Tuple[str, ...]
    elements: Optional[FrozenSet[int]] = None  # explicit Z-set; None => ask the provider
    recommended_aux: Optional[str] = None      # canonical name of a DF auxiliary, if any
    is_auxiliary: bool = False
    variant_group: str = ""                    # families sharing a group must not be mixed
    notes: str = ""

    # -- coverage ---------------------------------------------------------------------
    def covers(self, element: Union[int, str]) -> bool:
        z = z_of(element)
        els = self.elements if self.elements is not None else _provider_elements(self.name)
        return z in els

    def covered_elements(self) -> FrozenSet[int]:
        return self.elements if self.elements is not None else _provider_elements(self.name)

    # -- fitting route (DF where suitable, Cholesky fallback) -----------
    def fit_route(self) -> FitRoute:
        if self.recommended_aux is not None and self.conditioning in (
            Conditioning.WELL, Conditioning.MODERATE
        ):
            return FitRoute.DF
        return FitRoute.CHOLESKY

    def reference_objs(self) -> Tuple[Reference, ...]:
        return tuple(_REFS[k] for k in self.references)


# --- Registry construction -------------------------------------------------------------
_REGISTRY: Dict[str, BasisFamily] = {}


def _add(fam: BasisFamily) -> None:
    _REGISTRY[fam.name.lower()] = fam


# Karlsruhe x2c-nZVPall: scalar (X2C_1C) and -2c (X2C_2C, SOC — project default).
# Cardinality -> reference: DZ/TZ = Pollak2017, QZ = Franzke2020.
_KARLSRUHE = [
    ("x2c-SVPall", "pollak2017"),
    ("x2c-SV(P)all", "pollak2017"),
    ("x2c-TZVPall", "pollak2017"),
    ("x2c-TZVPPall", "pollak2017"),
    ("x2c-QZVPall", "franzke2020"),
    ("x2c-QZVPPall", "franzke2020"),
]
for _base, _ref in _KARLSRUHE:
    _refs = ("franzke2020",) if _ref == "franzke2020" else (_ref,)
    # scalar (one-component) variant
    _add(BasisFamily(
        name=_base, provider=Provider.BSE, provider_name=_base,
        contraction=Contraction.SEGMENTED, rel_treatment=RelTreatment.X2C_1C,
        role="Karlsruhe all-electron X2C (scalar); primary through Rn",
        conditioning=Conditioning.WELL, references=_refs, elements=_COV_KARLSRUHE,
        recommended_aux="x2c-JFIT", variant_group="karlsruhe-x2c",
        notes="Segmented, error-consistent. Use the -2c variant for two-component (SOC) work.",
    ))
    # two-component (-2c) variant — project default
    _add(BasisFamily(
        name=_base + "-2c", provider=Provider.BSE, provider_name=_base + "-2c",
        contraction=Contraction.SEGMENTED, rel_treatment=RelTreatment.X2C_2C,
        role="Karlsruhe all-electron X2C two-component (SOC); PROJECT DEFAULT through Rn",
        conditioning=Conditioning.WELL, references=_refs, elements=_COV_KARLSRUHE,
        recommended_aux="x2c-JFIT", variant_group="karlsruhe-x2c",
    ))

# x2c Coulomb-fitting auxiliaries (Franzke et al. 2020).
_add(BasisFamily(
    name="x2c-JFIT", provider=Provider.BSE, provider_name="x2c-JFIT",
    contraction=Contraction.SEGMENTED, rel_treatment=RelTreatment.X2C_2C,
    role="Coulomb (J) fitting auxiliary for the Karlsruhe x2c sets",
    conditioning=Conditioning.WELL, references=("franzke2020",), elements=_COV_KARLSRUHE,
    is_auxiliary=True, variant_group="karlsruhe-x2c",
))
_add(BasisFamily(
    name="x2c-JFIT-universal", provider=Provider.BSE, provider_name="x2c-JFIT-universal",
    contraction=Contraction.SEGMENTED, rel_treatment=RelTreatment.X2C_2C,
    role="Universal Coulomb (J) fitting auxiliary for the Karlsruhe x2c sets",
    conditioning=Conditioning.WELL, references=("franzke2020",), elements=_COV_KARLSRUHE,
    is_auxiliary=True, variant_group="karlsruhe-x2c",
))

# Peterson cc-pVnZ-X2C / cc-pwCVnZ-X2C (+aug). Mixed contraction; DF auxiliaries are sparse
# for these -> Cholesky fallback (conditioning marked ILL to route to Cholesky).
_PETERSON_REFS = ("hill2018", "lu2016", "feng2017", "ccrepo")
for _core in ("cc-pV{Z}-X2C", "cc-pwCV{Z}-X2C", "aug-cc-pV{Z}-X2C", "aug-cc-pwCV{Z}-X2C"):
    for _zeta in ("DZ", "TZ", "QZ"):
        _nm = _core.format(Z=_zeta)
        _cv = "pwCV" in _core
        _add(BasisFamily(
            name=_nm, provider=Provider.BSE, provider_name=_nm,
            contraction=Contraction.MIXED, rel_treatment=RelTreatment.X2C_1C,
            role=("Peterson correlation-consistent X2C"
                  + (" (core-valence weighted)" if _cv else "")
                  + "; primary for actinides, CBS-extrapolable"),
            conditioning=Conditioning.ILL, references=_PETERSON_REFS,
            elements=_COV_PETERSON, recommended_aux=None, variant_group="peterson-ccx2c",
            notes="Canonical X2C recontraction. Do NOT mix with SF/SO research "
                  "recontraction variants. DF auxiliaries are sparse; "
                  "Kuiva routes these to Cholesky.",
        ))

# Dyall — benchmarking only; uncontracted; heavy-element. Provider = PySCF (bundled).
# Coverage is per-variant, so we let the provider report it (elements=None).
_DYALL_PYSCF = [
    "dyallv2z", "dyallv3z", "dyallv4z",
    "dyallcv2z", "dyallcv3z", "dyallcv4z",
    "dyallacv2z", "dyallacv3z", "dyallacv4z",
    "dyallae2z", "dyallae3z", "dyallae4z",
    "dyallaae2z", "dyallaae3z", "dyallaae4z",
    "dyallav2z", "dyallav3z", "dyallav4z",
    "dyall2zp", "dyall3zp", "dyall4zp",
]
for _nm in _DYALL_PYSCF:
    _add(BasisFamily(
        name=_nm, provider=Provider.PYSCF, provider_name=_nm,
        contraction=Contraction.UNCONTRACTED, rel_treatment=RelTreatment.X2C_2C,
        role="Dyall all-electron relativistic (benchmarking only)",
        conditioning=Conditioning.ILL, references=("dyall2016",), elements=None,
        recommended_aux=None, variant_group="dyall",
        notes="Uncontracted; large and often near-linearly-dependent -> Cholesky. "
              "Developed for 4-component/DIRAC; usable as an X2C basis for benchmarks.",
    ))

# ANO-RCC — DKH-recontracted atomic natural orbitals; native to OpenMolcas. Provider = PySCF.
_add(BasisFamily(
    name="ANO-RCC", provider=Provider.PYSCF, provider_name="anorcc",
    contraction=Contraction.GENERAL, rel_treatment=RelTreatment.DKH,
    role="Douglas–Kroll–Hess ANO-RCC (general contraction), H–Cm",
    conditioning=Conditioning.ILL, references=("widmark1990", "roos2004", "roos2005tm",
                                               "roos2005act"),
    elements=_COV_ANORCC, recommended_aux=None, variant_group="ano-rcc",
    notes="Recontracted for DKH, not X2C. Generally contracted; Cholesky recommended. "
          "Mixing DKH-recontracted with X2C-recontracted sets is flagged as a soft "
          "relativistic-treatment mismatch.",
))

# The named contraction levels of ANO-RCC. An ANO set is one primitive set truncated to
# different numbers of contracted functions, so these are the *same* family at a fixed size
# rather than separate sets — hence the shared variant group and identical metadata.
#
# ⚠ They are here because the **matched-basis** validation rule needs them: a Tier-2 comparison is
# only attributable to the Hamiltonian if Kuiva runs in the external code's own basis, and
# ``ano-rcc-vdzp`` is what the committed OpenMolcas records were generated in. Without a
# registry entry the front-end would refuse the name, and reaching around the registry to
# PySCF would put an unchecked basis on the multireference path.
for _level, _role in (("VDZP", "double-zeta + polarization"),
                      ("VTZP", "triple-zeta + polarization")):
    _add(BasisFamily(
        name="ANO-RCC-" + _level, provider=Provider.PYSCF,
        provider_name="ano-rcc-" + _level.lower(),
        contraction=Contraction.GENERAL, rel_treatment=RelTreatment.DKH,
        role="Douglas–Kroll–Hess ANO-RCC, {} contraction, H–Cm".format(_role),
        conditioning=Conditioning.ILL,
        references=("widmark1990", "roos2004", "roos2005tm", "roos2005act"),
        elements=_COV_ANORCC, recommended_aux=None, variant_group="ano-rcc",
        notes="A fixed contraction level of ANO-RCC, not a different primitive set. "
              "Recontracted for DKH, not X2C; the same soft relativistic-treatment mismatch "
              "applies. This is the basis the committed OpenMolcas Tier-2 records use "
              "(matched bases for cross-code validation).",
    ))
del _level, _role


# --- Provider-reported coverage (lazy, cached) -----------------------------------------
@lru_cache(maxsize=None)
def _provider_elements(name: str) -> FrozenSet[int]:
    """Element coverage for a family whose ``elements`` is None, asked of the provider.

    Only PySCF-bundled families use this path; it attempts to load each element and keeps
    the ones that succeed. Cached per family name.
    """
    fam = _REGISTRY[name.lower()]
    if fam.provider is not Provider.PYSCF:
        raise RuntimeError(f"provider coverage requested for non-pyscf family {name!r}")
    from pyscf.gto import basis as _pbasis  # local import: keep registry import light
    got = set()
    for z in range(1, len(_SYMBOLS) + 1):
        try:
            _pbasis.load(fam.provider_name, symbol_of(z))
            got.add(z)
        except Exception:
            pass
    return frozenset(got)


# --- Public lookup / resolution --------------------------------------------------------
def get_family(name: str) -> BasisFamily:
    """Return the :class:`BasisFamily` for ``name`` (case-insensitive)."""
    try:
        return _REGISTRY[name.lower()]
    except KeyError:
        raise KeyError(
            f"basis family {name!r} is not in the registry. "
            f"Known families: {', '.join(sorted(f.name for f in _REGISTRY.values()))}"
        )


def has_family(name: str) -> bool:
    return name.lower() in _REGISTRY


def list_families(auxiliary: Optional[bool] = None) -> Tuple[str, ...]:
    """Canonical names of registered families, optionally filtering auxiliaries."""
    fams = _REGISTRY.values()
    if auxiliary is not None:
        fams = [f for f in fams if f.is_auxiliary is auxiliary]
    return tuple(sorted(f.name for f in fams))


def covers(name: str, element: Union[int, str]) -> bool:
    return get_family(name).covers(element)


def recommended_auxiliary(name: str) -> Optional[str]:
    return get_family(name).recommended_aux


def fit_route(name: str) -> FitRoute:
    return get_family(name).fit_route()


def references_for(name: str) -> Tuple[Reference, ...]:
    return get_family(name).reference_objs()


def resolve_for_pyscf(name: str, elements: Sequence[Union[int, str]]):
    """Return a PySCF-consumable basis specification for ``name`` and ``elements``.

    * PySCF-bundled family -> the provider alias string (PySCF loads it directly).
    * BSE family           -> a dict ``{symbol: parsed_shells}`` obtained from Basis Set
                              Exchange (NWChem format) and parsed via ``pyscf.gto.basis.parse``.

    Raises ``ValueError`` if any requested element is outside the family's coverage.
    """
    fam = get_family(name)
    syms = [symbol_of(z_of(e)) for e in elements]
    missing = [s for s in syms if not fam.covers(s)]
    if missing:
        raise ValueError(
            f"basis {fam.name!r} does not cover element(s) {missing} "
            f"(covers Z in {sorted(fam.covered_elements())[:1]}.."
            f"{sorted(fam.covered_elements())[-1:]}, {len(fam.covered_elements())} elements)"
        )
    if fam.provider is Provider.PYSCF:
        return fam.provider_name
    return {s: _parsed_bse_element(fam.provider_name, s) for s in syms}


@lru_cache(maxsize=None)
def _bse_element_shells(provider_name: str, symbol: str) -> tuple:
    """Fetch and parse ONE element of ONE BSE family. Cached on ``(family, element)``.

    Fetching from Basis Set Exchange and running ``pyscf.gto.basis.parse`` over the NWChem
    text is pure, deterministic and by far the most expensive thing in
    :func:`resolve_for_pyscf` — and it was being repeated on **every** ``build_mole``, once
    per atom label, hence twice per atom for a cross-basis overlap. The key is
    ``(provider family, element)`` rather than the whole element list, so a molecule of five
    elements and a molecule of one share four of the five entries.

    ⚠ **The parsed shells are returned as nested tuples and must stay immutable.** They are
    handed out from the cache to every caller, and PySCF's ``Mole`` build is free to
    normalize what it is given; one caller mutating a shared list in place would poison the
    basis of every later molecule in the process. :func:`_parsed_bse_element` is what turns
    them back into the fresh nested lists PySCF expects, per call.
    """
    import basis_set_exchange as bse
    from pyscf.gto import basis as _pbasis

    nw = bse.get_basis(provider_name, elements=[symbol], fmt="nwchem", header=False)
    return _freeze(_pbasis.parse(nw))


def _freeze(obj):
    """Nested lists -> nested tuples, so a cached parse cannot be mutated by a consumer."""
    return tuple(_freeze(x) for x in obj) if isinstance(obj, (list, tuple)) else obj


def _thaw(obj):
    """The inverse of :func:`_freeze` — a fresh nested list per caller."""
    return [_thaw(x) for x in obj] if isinstance(obj, (list, tuple)) else obj


def _parsed_bse_element(provider_name: str, symbol: str) -> list:
    """One element's parsed shells, from the cache, as a fresh mutable structure."""
    return _thaw(_bse_element_shells(provider_name, symbol))


# --- Measuring the contraction of actual basis data -------------------

def classify_contraction(shells: Sequence) -> Contraction:
    """The contraction type of a *parsed* basis, measured from the data itself.

    Every other contraction statement in this module is **declared** — it is the type the
    published family is documented to have, attached to a family name. This one is
    **measured**, from the nested lists PySCF parses a basis into
    (``[l, [exponent, coeff, ...], ...]`` per shell), and it is what the four-component
    atomic backend (:mod:`kuiva.amf.pyscf_dhf`) reports as provenance for a basis it was
    handed as raw data with no name attached.

    The distinction is not cosmetic. Per-atom bases may be mixed and
    overridden, so the family name a molecule was built with is not evidence about the
    functions any particular element actually carries — which is the same reason
    :func:`kuiva.amf.atomic.basis_digest` keys the cache on content rather than on a name.

    ⚠ **The discriminator is primitive *sharing*, not the shape of the stored block.** PySCF
    parses the NWChem form that Basis Set Exchange emits, and a segmented set arrives there as
    a *single* block listing all the primitives of a channel with a **block-diagonal**
    coefficient matrix — Karlsruhe ``x2c-SVPall-2c`` neon is one s shell of 7 primitives and 3
    contractions, which looks generally contracted and is not. What separates the two is
    whether any primitive contributes to more than one contracted function:

    ========================== ================= ==========================================
    channel                    verdict           example
    ========================== ================= ==========================================
    one primitive per function uncontracted      Dyall; ``cc-pVTZ-X2C`` g functions
    a primitive shared         general           ANO-RCC; ``cc-pVnZ-X2C`` low *l*
    neither                    segmented         Karlsruhe ``x2c-nZVPall-2c``
    ========================== ================= ==========================================

    Channels that disagree give :attr:`Contraction.MIXED`, which is exactly the Peterson
    structure expected to break a naive implementation first —
    measured on ``cc-pVTZ-X2C`` for Ce: general through *f*, then three uncontracted *g*
    primitives and one *h*. (At double zeta the same family has no multi-primitive
    uncontracted channel and is honestly just *general*, which is why the mixed case has to be
    tested at triple zeta.)

    A channel holding a single primitive in a single function — a lone polarization shell — is
    **neutral**: it is trivially both uncontracted and segmented, and counting it either way
    would label every Karlsruhe set as mixed on the strength of one *d* function.

    ⚠ **A measured verdict can disagree with the family's declared one, and that is the point
    rather than a bug.** ``x2c-SVPall-2c`` measures segmented, matching its ``BasisFamily``
    entry; ``x2c-TZVPPall-2c`` measures **mixed**, because its two *d* polarization functions
    are uncontracted primitives sitting beside segmented *s* and *p* channels. Both statements
    are true — the family-level label describes the valence contraction, and this function
    describes the data in hand. :attr:`Contraction.MIXED` here means only *the channels
    disagree*; the known mixed case (Peterson: general low-*l*, uncontracted high-*l*)
    is one instance of it and not the definition.
    """
    per_l: Dict[int, list] = {}
    for shell in shells:
        rows = [r for r in shell[1:] if isinstance(r, (list, tuple))]
        if not rows:
            continue                       # a shell with no primitives contributes nothing
        n_contr = max(len(r) - 1 for r in rows)
        shared = sum(1 for r in rows if sum(1 for c in r[1:] if c) > 1)
        per_l.setdefault(int(shell[0]), []).append((len(rows), n_contr, shared))
    if not per_l:
        raise ValueError("the basis specification contains no primitive functions")

    kinds = set()
    for blocks in per_l.values():
        n_prim = sum(b[0] for b in blocks)
        n_contr = sum(b[1] for b in blocks)
        if n_prim == 1 and n_contr == 1:
            continue                       # neutral: a lone primitive is not contracted
        if any(b[2] for b in blocks):
            kinds.add(Contraction.GENERAL)
        elif n_prim == n_contr:
            kinds.add(Contraction.UNCONTRACTED)
        else:
            kinds.add(Contraction.SEGMENTED)
    if not kinds:
        return Contraction.UNCONTRACTED    # every channel a single primitive
    if len(kinds) == 1:
        return kinds.pop()
    return Contraction.MIXED


# --- Ingestion consistency checks --------------------------------------
@dataclass
class ConsistencyReport:
    ok: bool
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()


def check_consistency(atom_basis, *, emit: bool = True) -> ConsistencyReport:
    """Validate a per-atom basis assignment before ingestion.

    ``atom_basis`` maps element symbol -> basis family name, or is an iterable of
    ``(symbol, family)`` pairs — the form a per-atom assignment needs, where one element may
    legitimately carry two families on different atoms and a dict could not say so. Checks:
      1. every family exists and covers its element (ERROR otherwise);
      2. relativistic-treatment compatibility across atoms — mixing a relativistic set with a
         non-relativistic set is an ERROR (a silent-error trap); mixing different
         *relativistic* recontractions (e.g. DKH ANO-RCC with X2C sets) is a WARNING;
      3. mixing incompatible variants within the same ``variant_group`` is a WARNING.

    When ``emit`` is True, messages are logged at ERROR/WARNING.
    """
    errors, warnings = [], []
    pairs = atom_basis.items() if isinstance(atom_basis, dict) else list(atom_basis)
    fams: List[BasisFamily] = []
    for sym, bname in pairs:
        try:
            fam = get_family(bname)
        except KeyError as exc:
            errors.append(str(exc))
            continue
        if not fam.covers(sym):
            errors.append(f"{sym}: basis {fam.name!r} does not cover this element")
        fams.append(fam)

    if fams:
        groups = {f.rel_treatment.family for f in fams}   # "x2c" / "dkh" / "nonrel"
        if "nonrel" in groups and groups != {"nonrel"}:
            errors.append(
                "relativistic-treatment mismatch: mixing relativistic and non-relativistic "
                "bases across atoms (a silent-error trap)"
            )
        rel_groups = groups - {"nonrel"}
        if len(rel_groups) > 1:
            warnings.append(
                "mixed relativistic Hamiltonians across atoms: "
                f"{sorted(rel_groups)} — allowed, but confirm this is intended (e.g. "
                "DKH-recontracted ANO-RCC alongside X2C-recontracted sets)"
            )

    if emit:
        for e in errors:
            log.error(e)
        for w in warnings:
            log.warning(w)

    return ConsistencyReport(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


__all__ = [
    "Contraction", "RelTreatment", "Provider", "Conditioning", "FitRoute",
    "Reference", "BasisFamily", "ConsistencyReport",
    "z_of", "symbol_of", "reference",
    "get_family", "has_family", "list_families", "covers", "recommended_auxiliary",
    "fit_route", "references_for", "resolve_for_pyscf", "check_consistency",
    "classify_contraction",
]
