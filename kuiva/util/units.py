"""Energy units and the atomic-unit constants: the one CODATA table every layer converts through.

Why a module in ``util/``
-------------------------
Several layers turn a Hartree difference into the unit a user reads it in — the CI drivers
print splittings in cm^-1, the property layer groups a spectrum in cm^-1, the perturbation
theory reports its corrections in cm^-1, and a state-selection window is *stated* in cm^-1 (or
in K, meV, eV, kJ/mol ...) and converted once. A conversion factor that lives inside one of
those consumers is imported "sideways" by the others, and the dependency rules of the
package forbid most of those imports (``kuiva.ci`` and ``kuiva.dmrg`` may not import
``kuiva.mcscf``; nothing below ``props/`` should import ``props/``). ``util/`` sits below
every solver, so this is where the table belongs. ``kuiva.props.multiplet.HARTREE_TO_CM`` is
an *import* of :data:`HARTREE_TO_CM`, bitwise the same literal it always was.

Conventions
-----------
* Every factor here is ``1 Eh`` expressed in the target unit, so a conversion *from* Hartree
  multiplies and a conversion *to* Hartree divides — one table, two directions, no second
  set of reciprocals to drift.
* Unit names are the **ASCII** spellings the output stream uses (``"cm^-1"``, ``"K"``,
  ``"meV"``, ``"eV"``, ``"Eh"``, ``"kJ/mol"``, ``"kcal/mol"``), matched case-insensitively
  together with a few aliases (``"cm-1"``, ``"wavenumber"``, ``"kelvin"``, ``"hartree"``,
  ...). An unknown unit is refused naming the accepted ones; nothing is guessed.
* ``kcal/mol`` uses the thermochemical calorie, ``1 cal = 4.184 J`` exactly.

Besides the energy table this module carries the handful of **atomic-unit constants** whose
value is a property of nature rather than of a method: the electron ``g`` factor, the nuclear
magneton, and the barn. They live here for the reason the energy factors do — every one of
them has more than one consumer, and a constant defined inside its first consumer is imported
sideways by the next one across a dependency boundary the package forbids — the electron ``g``
factor used to live in :mod:`kuiva.props.multiplet`, where the hyperfine field operator in
:mod:`kuiva.interface` could not reach it. :data:`kuiva.props.multiplet.G_ELECTRON` is now an
*import* of the value below, bitwise the literal it always was. ⚠ **Nothing else re-exports
these**: a second import path for one constant is the problem this table solves.

⚠ **The speed of light is deliberately NOT here.** It belongs to whatever produced the
integrals it is combined with (see :mod:`kuiva.x2c.decouple`), so a Hamiltonian reading it from
a shared table is exactly the mismatch that rule exists to prevent.

References
----------
* CODATA 2018 recommended values: E. Tiesinga, P. J. Mohr, D. B. Newell, B. N. Taylor,
  Rev. Mod. Phys. 93, 025010 (2021), doi:10.1103/RevModPhys.93.025010 — the Hartree energy in
  eV, K and J (with the exact 2019 SI value of the Avogadro constant for the molar units), the
  Hartree-to-wavenumber factor ``E_h / (h c)``, the Hartree-to-frequency factor ``E_h / h``,
  the electron ``g`` factor, the proton-electron mass ratio and the Bohr radius.
"""
from __future__ import annotations

from typing import Dict, Tuple

#: Hartree -> wavenumber, ``E_h / (h c)`` [cm^-1] (CODATA 2018). ⚠ The literal every
#: committed reference splitting was converted with; it may not move.
HARTREE_TO_CM = 219474.6313632
#: Hartree -> electronvolt [eV] (CODATA 2018).
HARTREE_TO_EV = 27.211386245988
#: Hartree -> millielectronvolt [meV].
HARTREE_TO_MEV = 1000.0 * HARTREE_TO_EV
#: Hartree -> kelvin, ``E_h / k_B`` [K] (CODATA 2018).
HARTREE_TO_K = 315775.02480407
#: Hartree -> kilojoule per mole, ``E_h N_A / 1000`` [kJ/mol]: ``E_h = 4.3597447222071e-18 J``
#: (CODATA 2018) and the exact ``N_A = 6.02214076e23 /mol``.
HARTREE_TO_KJ_MOL = 2625.4996394799
#: Hartree -> kilocalorie per mole [kcal/mol], thermochemical calorie (``4.184 J`` exactly).
HARTREE_TO_KCAL_MOL = HARTREE_TO_KJ_MOL / 4.184
#: Hartree -> megahertz, ``E_h / h`` [MHz] (CODATA 2018, ``E_h/h = 6.579683920502e15 Hz``).
#: ⚠ It is a **frequency**, not an energy in disguise: hyperfine couplings are quoted in MHz
#: throughout the EPR literature, and every ``A`` value this program reports is converted with
#: this factor and no other. It is deliberately not in :data:`FACTORS` — the energy table is
#: for spectra and state selection, where a unit named ``MHz`` would invite a state window
#: stated in a linewidth.
HARTREE_TO_MHZ = 6.579683920502e9

#: The free-electron ``g`` factor (CODATA 2018). ⚠ Dirac theory gives exactly ``2``; the
#: difference is the QED anomaly, and *which of the two a given operator carries* is a
#: statement each property operator has to make for itself — the four-component magnetic
#: interaction ``c alpha.A`` carries Dirac's ``2`` and the anomaly is a separate term.
G_ELECTRON = 2.00231930436256

#: Proton-to-electron mass ratio (CODATA 2018), the only place the proton mass enters.
PROTON_ELECTRON_MASS_RATIO = 1836.15267343

#: The nuclear magneton in **atomic units**, ``mu_N = mu_B m_e / m_p = 1 / (2 m_p/m_e)``.
#: ⚠ In Hartree atomic units ``mu_B = 1/2``, not ``1``; dropping the one half is a factor-of-two
#: error in every hyperfine coupling and it looks entirely plausible.
NUCLEAR_MAGNETON = 1.0 / (2.0 * PROTON_ELECTRON_MASS_RATIO)

#: The barn in bohr squared, ``1e-28 m^2 / a_0^2`` with ``a_0 = 5.29177210903e-11 m`` (CODATA
#: 2018) — for nuclear quadrupole moments, which every compilation tabulates in barn.
BARN_TO_BOHR2 = 1e-28 / 5.29177210903e-11 ** 2

#: Canonical unit name -> ``1 Eh`` in that unit. The canonical names are the ASCII spellings
#: used in the output stream.
FACTORS: Dict[str, float] = {
    "Eh": 1.0,
    "cm^-1": HARTREE_TO_CM,
    "eV": HARTREE_TO_EV,
    "meV": HARTREE_TO_MEV,
    "K": HARTREE_TO_K,
    "kJ/mol": HARTREE_TO_KJ_MOL,
    "kcal/mol": HARTREE_TO_KCAL_MOL,
}

#: Accepted spellings (lower-cased) -> canonical name. Every canonical name is its own alias.
ALIASES: Dict[str, str] = {
    "eh": "Eh", "hartree": "Eh", "hartrees": "Eh", "au": "Eh", "a.u.": "Eh",
    "cm^-1": "cm^-1", "cm-1": "cm^-1", "cm**-1": "cm^-1", "1/cm": "cm^-1",
    "wavenumber": "cm^-1", "wavenumbers": "cm^-1",
    "ev": "eV", "electronvolt": "eV",
    "mev": "meV",
    "k": "K", "kelvin": "K",
    "kj/mol": "kJ/mol", "kjmol": "kJ/mol", "kj mol^-1": "kJ/mol",
    "kcal/mol": "kcal/mol", "kcalmol": "kcal/mol", "kcal mol^-1": "kcal/mol",
}

#: The canonical names, in the order they are listed in a refusal.
UNITS: Tuple[str, ...] = tuple(FACTORS)


def canonical_unit(unit: str) -> str:
    """The canonical spelling of ``unit``, or a ``ValueError`` naming the accepted ones."""
    key = str(unit).strip().lower()
    try:
        return ALIASES[key]
    except KeyError:
        raise ValueError("unknown energy unit {!r}; accepted units are {} (case-insensitive, "
                         "with the aliases {})".format(
                             unit, ", ".join(UNITS),
                             ", ".join(sorted(a for a in ALIASES if a not in
                                              {u.lower() for u in UNITS})))) from None


def factor(unit: str) -> float:
    """``1 Eh`` expressed in ``unit``."""
    return FACTORS[canonical_unit(unit)]


def to_hartree(value: float, unit: str) -> float:
    """``value`` [unit] -> [Eh]."""
    return float(value) / factor(unit)


def from_hartree(value: float, unit: str) -> float:
    """``value`` [Eh] -> [unit]."""
    return float(value) * factor(unit)


__all__ = ["HARTREE_TO_CM", "HARTREE_TO_EV", "HARTREE_TO_MEV", "HARTREE_TO_K",
           "HARTREE_TO_KJ_MOL", "HARTREE_TO_KCAL_MOL", "HARTREE_TO_MHZ", "G_ELECTRON",
           "PROTON_ELECTRON_MASS_RATIO", "NUCLEAR_MAGNETON", "BARN_TO_BOHR2",
           "FACTORS", "ALIASES", "UNITS",
           "canonical_unit", "factor", "to_hartree", "from_hartree"]
