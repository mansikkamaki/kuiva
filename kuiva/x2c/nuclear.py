"""The nuclear charge model: the third axis of a Hamiltonian, and its provenance record.

A nucleus is either a **point charge** or a **finite (Gaussian) distribution**. Nothing else
in Kuiva chooses between them; this module holds the vocabulary, the record that travels with
a stored result, and the one mapping onto the integral library's own flag.

**It is an axis, not a method name.** :mod:`kuiva.x2c.methods` resolves a Hamiltonian *name*
(``"X2C-AMF"``, ...) onto the ``decoupling`` and ``screening`` axes; the nuclear model is
carried alongside and is deliberately absent from every name, because it is a property of the
**potential the integrals were evaluated over** rather than of the decoupling or the screening
applied afterwards. Both of those axes, and the property operators, inherit whatever the
integrals were made with.

That is the same rule the speed of light obeys in :mod:`kuiva.x2c.decouple`, and for the same
reason: a value chosen in one place and a set of integrals made in another is how two halves of
one Hamiltonian end up describing different physics with nothing in the output saying so. The
practical consequence is stated where it binds — an atomic mean-field solve
(:mod:`kuiva.amf`) and the molecular integrals it corrects must use the **same** model, so
:func:`kuiva.amf.correction.amf_correction` reads it off the molecule it is correcting rather
than taking it as an argument a caller could fail to pass.

The finite model
----------------
The Gaussian charge distribution is

    rho(r) = Z * N * exp(-zeta r^2),    zeta = 3 / (2 <r^2>),

with the root-mean-square radius taken from the mass number of the element's most abundant
isotope. Kuiva does not implement this: it asks the integral library for it, and the record
below says which parametrization answered. PySCF's is

    L. Visscher and K. Dyall, *Atomic Data and Nuclear Data Tables* **67**, 207 (1997),

implemented as ``pyscf.gto.mole.dyall_nuc_mod`` with the main-isotope masses of
``pyscf.data.elements.ISOTOPE_MAIN``.

⚠ **The effect grows steeply with Z and is invisible for light elements.** It is the first
mismatch against a four-component program whose own default is Gaussian (DIRAC's is), so a
comparison that does not state the model on both sides is comparing two Hamiltonians.

⚠ **The integral library's flag is not a boolean and 1 does not mean "point".** PySCF's
``_parse_nuc_mod`` treats *any* non-zero, non-callable value as a request for a Gaussian
nucleus, so ``nucmod=1`` — the obvious spelling of "model number one", and the value of the
``NUC_POINT`` constant — silently gives a finite nucleus. :func:`pyscf_nucmod` is the only
place in Kuiva that mapping is written down, and every caller goes through it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

#: The nuclear charge models. ``"point"`` is the default everywhere and is what every
#: committed reference number in this project was produced with.
NUCLEAR_MODELS = ("point", "gaussian")

#: Spellings accepted for each model. ``"finite"`` and ``"gauss"`` are common in the
#: four-component literature and in other programs' input; they name the same thing.
_ALIASES = {
    "point": "point", "pointcharge": "point", "point-charge": "point", "pnc": "point",
    "gaussian": "gaussian", "gauss": "gaussian", "finite": "gaussian",
    "gauss_nuc": "gaussian", "fnc": "gaussian",
}

#: The parametrization the finite model is taken from, for :attr:`NuclearRecord.parametrization`.
DYALL_PARAMETRIZATION = "visscher-dyall-1997"

#: Which isotope masses fix the rms radius. PySCF uses the most abundant isotope of each
#: element and offers no per-atom override through the path Kuiva builds molecules on, so this
#: is a constant rather than an option — but it is *recorded*, because a different choice of
#: isotope is a different nucleus.
MAIN_ISOTOPES = "main"


def resolve_nuclear_model(value: Optional[object]) -> str:
    """Normalize a user's nuclear-model statement to one of :data:`NUCLEAR_MODELS`.

    ``None`` means the default, ``"point"``. Unknown names are refused rather than mapped to
    the default: a typo that silently selects a point nucleus produces numbers that are right
    for the wrong Hamiltonian.

    ⚠ **A number is refused**, however plausible it looks. The integral library's flag is not
    the user surface (see this module's docstring), and ``1`` means opposite things in the two
    vocabularies.
    """
    if value is None:
        return "point"
    if isinstance(value, bool) or not isinstance(value, str):
        raise TypeError(
            "nuclear_model must be one of {}; got {!r}. ⚠ It is deliberately not a number or "
            "a flag: PySCF's nucmod reads any non-zero value as a request for a Gaussian "
            "nucleus, so nuclear_model=1 would mean the opposite of what it looks like."
            .format(" or ".join(repr(m) for m in NUCLEAR_MODELS), value))
    key = value.strip().lower().replace(" ", "")
    if key not in _ALIASES:
        raise ValueError(
            "unknown nuclear_model {!r}; expected one of {}".format(
                value, ", ".join(repr(m) for m in NUCLEAR_MODELS)))
    return _ALIASES[key]


def pyscf_nucmod(model: str) -> Union[int, str]:
    """The value to hand PySCF as ``Mole.nucmod`` for ``model``.

    ⚠ **The only place the mapping is written.** ``0`` is a point nucleus and the *string*
    ``"gauss"`` is the Gaussian one; every other integer, ``1`` included, is read by PySCF as
    Gaussian (``pyscf.gto.mole._parse_nuc_mod``). Writing the flag at a call site is how a
    calculation ends up with a finite nucleus nobody asked for, in a run whose output says
    nothing about it.
    """
    model = resolve_nuclear_model(model)
    return "gauss" if model == "gaussian" else 0


@dataclass(frozen=True)
class NuclearRecord:
    """Which nuclear charge model produced a Hamiltonian — the third part of provenance.

    ⚠ Like :class:`~kuiva.x2c.methods.DecouplingRecord` and
    :class:`kuiva.amf.correction.ScreeningRecord`, this is a **contract with stored data**: it
    goes into the property dump's header, so a stored property matrix is never ambiguous about
    which of several Hamiltonians produced it. Add fields; do not rename or repurpose them.

    It is a sibling of those two rather than a field on either, because the nucleus is
    upstream of both: the same model has to be stated for a Hamiltonian with no two-electron
    screening at all, and the atomic mean field and the one-electron decoupling must agree
    about it.

    ``NuclearRecord()`` — all defaults — describes the point nucleus that every committed
    reference in this project was produced with.
    """

    model: str = "point"
    #: Where the finite distribution's parameters come from. Empty for a point nucleus.
    parametrization: str = ""
    #: Which isotope masses fix the rms radius. Empty for a point nucleus.
    isotopes: str = ""
    #: What evaluated the integrals over this potential. Provenance only.
    implementation: str = ""

    @property
    def finite(self) -> bool:
        """Whether the nucleus has a spatial extent."""
        return self.model != "point"

    def as_dict(self) -> Dict[str, object]:
        """JSON-serializable form, for the property dump header and reference records."""
        return {
            "model": str(self.model),
            "parametrization": str(self.parametrization),
            "isotopes": str(self.isotopes),
            "implementation": str(self.implementation),
        }

    def label(self) -> str:
        """One ASCII line naming the model, for the output stream and file headers."""
        if not self.finite:
            return "point"
        parts = [p for p in (self.parametrization, self.isotopes and
                             "{} isotopes".format(self.isotopes)) if p]
        return "gaussian ({})".format(", ".join(parts)) if parts else "gaussian"

    def report(self, logger=None) -> None:
        """The provenance line, printed wherever the Hamiltonian is described."""
        from ..util import output as out
        from ..util.logging import get_logger

        out.entry(logger or get_logger(__name__), "nuclear charge model", self.label())


def nuclear_record(model: Optional[object], *, implementation: str = "pyscf") -> NuclearRecord:
    """The record for ``model``, with the parametrization that goes with it."""
    resolved = resolve_nuclear_model(model)
    if resolved == "point":
        return NuclearRecord(model="point", implementation=implementation)
    return NuclearRecord(model="gaussian", parametrization=DYALL_PARAMETRIZATION,
                         isotopes=MAIN_ISOTOPES, implementation=implementation)


__all__ = ["DYALL_PARAMETRIZATION", "MAIN_ISOTOPES", "NUCLEAR_MODELS", "NuclearRecord",
           "nuclear_record", "pyscf_nucmod", "resolve_nuclear_model"]
