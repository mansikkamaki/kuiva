"""Energy units: the one CODATA table every layer converts through.

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

References
----------
* CODATA 2018 recommended values: E. Tiesinga, P. J. Mohr, D. B. Newell, B. N. Taylor,
  Rev. Mod. Phys. 93, 025010 (2021), doi:10.1103/RevModPhys.93.025010 — the Hartree energy in
  eV, K and J (with the exact 2019 SI value of the Avogadro constant for the molar units) and
  the Hartree-to-wavenumber factor ``E_h / (h c)``.
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
           "HARTREE_TO_KJ_MOL", "HARTREE_TO_KCAL_MOL", "FACTORS", "ALIASES", "UNITS",
           "canonical_unit", "factor", "to_hartree", "from_hartree"]
