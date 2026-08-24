"""Thin PySCF front-end / bridge.

PySCF is used for two things and no more: a **scalar X2C** (``sfx2c1e``) SCF that produces
real scalar-relativistic MOs, and the raw integrals — including the **two-component X2C
one-electron Hamiltonian** from which spin-orbit coupling enters the multireference step.
This module ingests them into a self-contained :class:`ScalarX2CData` container. **After
ingestion, nothing downstream calls PySCF** — the multireference layer consumes only this
container.

What is extracted: scalar MO coefficients (one set for RHF/ROHF, two for UHF), AO
overlap ``S``, the spin-free X2C one-electron Hamiltonian ``h_x2c``, the two-component X2C
Hamiltonian as a :class:`SpinOrbitX2C` decomposition, and the two-electron integrals
(conventional 8-fold-packed ERIs, or density-fitting factors).

Why the SCF and the correlated Hamiltonian use *different* one-electron operators
---------------------------------------------------------------------------------
The SCF runs with ``sfx2c1e`` (spin-free), because the design puts SOC at the CASSCF/CI level and a
scalar SCF is what converges reliably. The multireference step then uses the **full**
two-component X2C Hamiltonian, whose spin-free part is *not* identical to the ``sfx2c1e`` one:
the two are obtained from different decoupling transformations (spin-free ``X`` versus full
``X``), and they differ by a picture-change effect that reaches several Eh on heavy-element
core matrix elements. This is not an inconsistency to be fixed. Orbitals are only a basis —
the CASSCF re-optimizes them — so the scalar set is a *guess*, while the operator whose
expectation value is the energy is one well-defined Hamiltonian throughout. The size of the
difference is measured at ingestion and reported (:attr:`SpinOrbitX2C.picture_change_shift`)
so it is never a silent surprise.

.. warning::
   **Two-electron spin-orbit coupling is ON by default** (``screening="x2camf"``). The ingested operator carries the atomic mean-field two-electron picture change of
   :mod:`kuiva.amf`, which reproduces four-component j-splittings in the same basis to 0.03%
   closed-shell and 0.7% open-shell.

   ``screening="none"`` ingests the one-electron X2C spin-orbit operator alone. That is a
   supported choice and the right one wherever spin-orbit coupling is not the subject, since
   the correction changes no scalar quantity and costs a four-component atomic solve — but it
   makes SOC splittings **too large by 5-30%**, which is larger than the Tier-2 cross-code
   cross-code tolerance of 15%. The molecular validation suite therefore cannot tell a working
   correction from a broken one, and the atomic four-component atomic references
   exist for that reason.

   ⚠ **The cost of the default is one four-component atomic SCF per unique element** — under a
   second for a light atom, ~35 minutes for a lanthanide — paid once ever per
   ``(element, basis, configuration, interaction)`` and cached both in the process and on disk
   (:mod:`kuiva.amf.cache`), because the correction depends on no geometry.

   ⚠ **Adding a correction to a Hamiltonian that already has one double-counts it.**
   :attr:`SpinOrbitX2C.screening` says what the operator **contains**, not what was requested,
   and now that screening is the default, the pattern to watch for is
   ``ingest_spin_orbit(mol).hamiltonian() + correction.hamiltonian()`` — which halves a
   splitting and looks entirely plausible. Ask for ``screening="none"`` whenever the correction
   is being supplied separately.

   The empirical alternatives (SNSO/Boettger, Breit-Pauli AMFI) are *rejected*, not
   unimplemented; see :mod:`kuiva.amf.correction`.

References:
- Scalar/one-electron X2C (``sfx2c1e``): W. Liu, D. Peng, J. Chem. Phys. 131, 031104 (2009),
  doi:10.1063/1.3159445; T. Nakajima, K. Hirao, Chem. Rev. 112, 385 (2012); and the review
  by D. Peng, M. Reiher, Theor. Chem. Acc. 131, 1081 (2012), doi:10.1007/s00214-011-1081-y.
- Two-component X2C and the exact decoupling: W. Kutzelnigg, W. Liu, J. Chem. Phys. 123,
  241102 (2005), doi:10.1063/1.2137315; W. Liu, D. Peng, J. Chem. Phys. 125, 044102 (2006),
  doi:10.1063/1.2222365; M. Ilias, T. Saue, J. Chem. Phys. 126, 064102 (2007),
  doi:10.1063/1.2436882; D. Peng, M. Reiher, Theor. Chem. Acc. 131, 1081 (2012),
  doi:10.1007/s00214-011-1081-y.
- Two-electron SOC screening, as implemented (``screening="x2camf"``): J. Liu, L. Cheng,
  J. Chem. Phys. 148, 144108 (2018), doi:10.1063/1.5023750; B. A. Hess, C. M. Marian,
  U. Wahlgren, O. Gropen, Chem. Phys. Lett. 251, 365 (1996) for the atomic mean-field idea.
  See :mod:`kuiva.amf` for the full reference list.
- Two-electron SOC screening, rejected alternatives (recorded so they are not reintroduced):
  J. C. Boettger, Phys. Rev. B 57, 8743 (1998), doi:10.1103/PhysRevB.57.8743 (SNSO);
  M. Filatov, W. Zou, D. Cremer, J. Chem. Phys. 139, 014106 (2013), doi:10.1063/1.4811776;
  B. de Souza, G. Farias, F. Neese, R. Izsak, J. Chem. Theory Comput. 15, 1896 (2019),
  doi:10.1021/acs.jctc.8b00841 (AMFI/RI-SOMF).
- PySCF: Q. Sun et al., J. Chem. Phys. 153, 024109 (2020), doi:10.1063/5.0006074.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..amf.correction import ScreeningRecord
from ..basis import registry as reg
from ..basis.layout import AOLayout, Shell, build_layout
# One threshold, defined where the decomposition is. The integral layer imports nothing from
# here, so this direction is the safe one and it keeps the front end from carrying a second
# copy of a number that is a physics decision.
from ..integrals.transform import CHOLESKY_VECTORS_PER_AO, DEFAULT_CHOLESKY_TOL
from ..util import output as out
from ..util import resources as res
from ..util.logging import get_logger
from ..util.timing import timer
from ..x2c.methods import DecouplingRecord

log = get_logger(__name__)


def eri_memory_gb(nao: int) -> float:
    """Size [GB] of the conventional 8-fold-packed AO ERI array (exact sizing function).

    ``npair*(npair+1)/2`` doubles with ``npair = nao(nao+1)/2``, i.e. ``O(nao^4/8)``: 0.6 GB
    at nao = 200 and 10 GB at nao = 400. This is the single largest array the default
    (Cholesky) route allocates, it is allocated by PySCF inside ``mol.intor``, and it is
    knowable from the AO count alone — which makes it the earliest useful check in the whole
    program (estimate at the first point where an estimate is possible).

    ⚠ Until an integral-direct Cholesky decomposition exists (``pivoted_cholesky`` already
    takes a column callback so that it can be added without touching anything else), this
    array bounds the system size of the default route.
    """
    npair = nao * (nao + 1) // 2
    return res.array_gb((npair * (npair + 1) // 2,), np.float64)


@dataclass(frozen=True)
class SpinOrbitX2C:
    """The two-component X2C one-electron Hamiltonian, decomposed into Kuiva's conventions.

    PySCF's ``SpinOrbitalX2CHelper.get_hcore()`` returns the full 2c X2C Hamiltonian in the
    spin-blocked ``[alpha; beta]`` AO basis — the same row convention as
    :mod:`kuiva.spinor.expand` — so no basis reordering is needed. It is stored here in the
    decomposed form ``H = A (x) 1_2 + sigma . W`` with ``W_k = i w_k``:

    ==================  ==========================================================
    ``h_sf``  ``A``     ``(H_aa + H_bb) / 2``           real symmetric
    ``w[2]``  ``w_z``   ``Im(H_aa - H_bb) / 2``         real antisymmetric
    ``w[0]``  ``w_x``   ``Im(H_ab + H_ba) / 2``         real antisymmetric
    ``w[1]``  ``w_y``   ``Re(H_ab - H_ba) / 2``         real antisymmetric
    ==================  ==========================================================

    Three reasons to store the decomposition rather than the assembled matrix:

    1. **It is smaller**: ``4 nao^2`` reals against ``8 nao^2``, and each ``w_k`` transforms
       like an ordinary one-electron operator under a change of scalar basis, so the
       working-basis transformation is four ordinary congruences instead of a 2c one.
    2. **It restores exact time-reversal symmetry.** Taking ``A`` real symmetric and ``w_k``
       real antisymmetric *is* the projection onto the time-reversal-even part of ``H``. The
       X2C decoupling involves a matrix square root whose rounding error breaks that symmetry
       slightly — measured at 3.5e-6 Eh for Bi/x2c-SVPall-2c against matrix elements of order
       1e4 Eh — which would otherwise appear downstream as a spurious Kramers splitting of the
       same size, right in the 1e-8..1e-6 Eh band reserved for genuine numerical
       splitting. The discarded part is measured and reported, not silently dropped.
    3. It is the exact input :func:`kuiva.spinor.expand.two_component_operator` expects, so
       assembling the Hamiltonian downstream is one call with no convention to re-derive.

    Attributes
    ----------
    h_sf : ndarray (nao, nao)
        Spin-free part of the **full 2c** Hamiltonian. Not the same as ``sfx2c1e``'s
        ``h_x2c`` — see this module's docstring and ``picture_change_shift``.
    w : ndarray (3, nao, nao)
        Real antisymmetric spin-orbit factors, ``W_k = i w_k``.
    approx : str
        The one-electron decoupling axis (``"1e"``, ``"1e-dlu"``, ``"atom1e"``). Kept as a bare
        string for the callers that have always passed it positionally; :attr:`decoupling` is
        the record that describes what was actually done.
    method : str
        The resolved Hamiltonian name (``"X2C-AMF"``, ...), or ``""`` for an object built
        before the method surface existed. Provenance, not behaviour.
    decoupling : DecouplingRecord
        Which one-electron decoupling produced this operator — the partition
        and per-fragment ``max |X|`` when it was local. ⚠ Like :attr:`screening`, a **contract
        with stored data**: it goes into the property dump header so that a stored property matrix is
        never ambiguous about which of several Hamiltonians produced it.
    screening : ScreeningRecord
        **Which two-electron picture change is already contained in** ``h_sf`` and ``w``, in
        full: method, interaction, four-component backend and version, the per-element
        reference configurations, and the magnitudes of the two halves of the correction. The
        default record describes an uncorrected one-electron Hamiltonian and its ``method`` is
        ``"none"``.

        ⚠ It is a record of what has **already been added**, not a request. Adding
        ``amf_correction(...)`` to a Hamiltonian whose record is not ``"none"`` double-counts
        the correction. The one place it is applied is :func:`ingest_spin_orbit`.
    tr_residual, tr_residual_rel : float
        Absolute [Eh] and relative size of the discarded time-reversal-odd part.
    picture_change_shift : float
        ``max |h_sf - h_sfx2c1e|`` [Eh] — how far the operator the multireference step uses has
        moved from the one the SCF ran with. Measured on ``h_sf`` **as returned**, so with
        ``screening`` on it includes the correction's spin-free half; with it off it is bitwise
        the number it always was. For a light element the two contributions are comparable
        (HF: both 4.5e-3 Eh), for a heavy one the picture change dominates (Bi: 3.4 Eh).
    """

    h_sf: np.ndarray
    w: np.ndarray
    approx: str = "1e"
    screening: ScreeningRecord = field(default_factory=ScreeningRecord)
    decoupling: DecouplingRecord = field(default_factory=DecouplingRecord)
    method: str = ""

    tr_residual: float = 0.0
    tr_residual_rel: float = 0.0
    picture_change_shift: float = 0.0

    @property
    def nao(self) -> int:
        return int(self.h_sf.shape[0])

    @property
    def soc_strength(self) -> float:
        """``max |w|`` [Eh] — the scale of the spin-orbit operator in this basis."""
        return float(np.max(np.abs(self.w))) if self.w.size else 0.0

    def hamiltonian(self) -> np.ndarray:
        """Assemble the ``(2*nao, 2*nao)`` two-component Hamiltonian in the AO basis."""
        from ..spinor.expand import two_component_operator
        return two_component_operator(self.h_sf, self.w)

    def transform(self, x: np.ndarray) -> "SpinOrbitX2C":
        """Re-express in another scalar basis (``X^dag A X`` on each component), e.g. the
        orthonormal working basis. All four components transform identically, which is
        the practical payoff of storing the decomposition."""
        return SpinOrbitX2C(
            h_sf=x.T @ self.h_sf @ x,
            w=np.stack([x.T @ wk @ x for wk in self.w]),
            approx=self.approx, screening=self.screening,
            decoupling=self.decoupling, method=self.method,
            tr_residual=self.tr_residual, tr_residual_rel=self.tr_residual_rel,
            picture_change_shift=self.picture_change_shift)

    def report(self, logger=None) -> None:
        logger = logger or log
        out.entry(logger, "two-component Hamiltonian", self.method or "X2C")
        self.decoupling.report(logger)
        out.entry(logger, "spin-orbit operator scale, max |w|", self.soc_strength, "Eh",
                  fmt="{:.6f}")
        out.entry(logger, "picture-change shift vs sfx2c1e", self.picture_change_shift, "Eh",
                  fmt="{:.3e}", note="guess orbitals only")
        out.entry(logger, "discarded time-reversal-odd part", self.tr_residual, "Eh",
                  fmt="{:.3e}", note="{:.1e} relative".format(self.tr_residual_rel))
        self.screening.report(logger)

    def provenance(self) -> Dict[str, object]:
        """A JSON-serializable description of **which Hamiltonian this is**.

        Everything an external consumer needs to know what produced a stored matrix: the
        decoupling approximation, the size of the projected time-reversal-odd part, and the
        full two-electron screening record. This is the dict the property dump writes into
        its header and a Tier-2 reference record stores alongside its energies — a stored
        property matrix that does not say whether it was screened is not interpretable, and
        the difference is 15-30% on every splitting in it.
        """
        return {
            "method": self.method,
            "one_electron": "X2C (approx={})".format(self.approx),
            "decoupling": self.decoupling.as_dict(),
            "soc_strength": float(self.soc_strength),
            "picture_change_shift": float(self.picture_change_shift),
            "tr_residual": float(self.tr_residual),
            "tr_residual_rel": float(self.tr_residual_rel),
            "screening": self.screening.as_dict(),
        }

    def __repr__(self) -> str:
        return "SpinOrbitX2C(nao={}, method={}, approx={}, screening={}, " \
               "max|w|={:.4f} Eh)".format(self.nao, self.method or "?", self.approx,
                                          self.screening.method, self.soc_strength)


@dataclass(frozen=True)
class PropertyIntegrals:
    """One-electron property integrals for the magnetic-moment dump.

    The magnetic moment is ``mu = -(L + g_e S) mu_B``. ``S`` needs nothing but the overlap
    and the spin-blocked row layout (:func:`kuiva.spinor.expand.spin_operator`), so the only thing
    that has to cross the ingestion boundary is the **orbital** angular momentum, which is
    what this carries.

    Attributes
    ----------
    irxp : ndarray (3, nao, nao), real
        ``<mu| (r - R_G) x nabla |nu>`` — PySCF's ``int1e_cg_irxp`` about the gauge origin,
        real and **antisymmetric**. The physical operator is
        :meth:`angular_momentum`, ``L = -i (r - R_G) x nabla`` in units of hbar; the factor
        of ``-i`` is kept out of storage so the array stays real, exactly as
        :class:`SpinOrbitX2C` keeps ``W_k = i w_k`` decomposed.
    gauge_origin : ndarray (3,)
        The gauge origin ``R_G`` in **bohr**.
    origin_label : str
        How it was chosen (``"centre of mass"``, ``"centre of nuclear charge"``,
        ``"explicit"``). Provenance for the dump header, since ``L`` — and hence every
        moment matrix built from it — depends on this choice.

    Notes
    -----
    ⚠ **No picture change is applied** (an explicit standing decision): these are the bare
    non-relativistic AO operators, used unchanged in the two-component basis. That matches
    what OpenMolcas RASSI does, so the Tier-2 comparison is like-for-like, and it is an
    approximation whose size nobody here has measured. :func:`kuiva.props.dump.write_dump`
    warns about it at the point the file is written and records it in the header.

    References
    ----------
    * Picture change of property operators under X2C, i.e. what is being skipped: D. Peng,
      M. Reiher, J. Chem. Phys. 136, 244108 (2012), doi:10.1063/1.4729788.
    * Magnetic moments from spin-orbit eigenstates: L. F. Chibotaru, L. Ungur, J. Chem.
      Phys. 137, 064112 (2012), doi:10.1063/1.4739763.
    """

    irxp: np.ndarray
    gauge_origin: np.ndarray
    origin_label: str = "explicit"
    picture_change: Optional["PictureChangedMoment"] = None

    @property
    def nao(self) -> int:
        return int(self.irxp.shape[-1])

    def moment_operator(self) -> Optional[np.ndarray]:
        """``(L + 2 S)`` picture-changed, or ``None`` when the bare operators are in use.

        ⚠ A consumer that gets a non-``None`` here must build ``mu`` from it **instead of**
        from ``L + g_e S``, adding only the anomaly ``(g_e - 2) S`` — the picture change does
        not separate into an ``L`` part and an ``S`` part (see :class:`PictureChangedMoment`).
        """
        return None if self.picture_change is None else self.picture_change.moment

    def anomaly_spin(self) -> Optional[np.ndarray]:
        """The picture-changed spin operator for the ``g_e - 2`` term, if one was built."""
        return None if self.picture_change is None else self.picture_change.spin

    def angular_momentum(self) -> np.ndarray:
        """``L_k`` as ``(3, nao, nao)`` **complex Hermitian**, in units of hbar."""
        return -1j * np.asarray(self.irxp)

    def two_component(self) -> np.ndarray:
        """``L_k (x) 1_2`` as ``(3, 2*nao, 2*nao)`` — the operator in the spin-blocked row layout.

        Orbital angular momentum is spin-free, so it lifts by
        :func:`kuiva.spinor.expand.spin_block_diagonal` and by nothing else.
        """
        from ..spinor.expand import spin_block_diagonal
        return np.stack([spin_block_diagonal(lk) for lk in self.angular_momentum()])

    def provenance(self) -> Dict[str, object]:
        pc = "none (bare AO operators, used unchanged in the 2c basis)"
        if self.picture_change is not None:
            pc = self.picture_change.label()
        return {
            "gauge_origin_bohr": [float(x) for x in np.asarray(self.gauge_origin).ravel()],
            "gauge_origin_choice": self.origin_label,
            "picture_change": pc,
        }

    def __repr__(self) -> str:
        r = np.asarray(self.gauge_origin).ravel()
        return "PropertyIntegrals(nao={}, gauge origin {} = ({:.4f}, {:.4f}, {:.4f}) bohr)" \
            .format(self.nao, self.origin_label, r[0], r[1], r[2])


#: Bohr per Angstrom, for the ``("angstrom", x, y, z)`` gauge-origin form. PySCF's own
#: constant, so a coordinate given here and the same coordinate given in the geometry land in
#: exactly the same place — a gauge origin that misses an atom by a rounding difference is a
#: silent shift of every ``L`` matrix in the file.
def _bohr_per_angstrom() -> float:
    from pyscf.data.nist import BOHR
    return 1.0 / float(BOHR)


def gauge_origin_for(mol, origin=None) -> Tuple[np.ndarray, str]:
    """Resolve a gauge origin to ``(coordinates [bohr], label)``.

    Five forms, and the units are the whole point of three of them:

    ``None`` / ``"mass"``
        The **centre of mass** — the default.
    ``"charge"`` / ``"origin"``
        The centre of nuclear charge, or the coordinate origin.
    ``("atom", k)``
        **On an atom.** ``k`` is a 1-based atom number, an element symbol, or an atom label
        (``"Ti2"``) — the same addressing per-atom bases and reference configurations use
        (:func:`kuiva.basis.atommap.parse_atom_key`), so there is one way to name an atom in
        this program. An element symbol is accepted only where the molecule has exactly one
        such atom; otherwise it names several places at once and is refused rather than
        resolved to the first.
    ``("bohr", x, y, z)`` / ``("angstrom", x, y, z)``
        An explicit point, **saying which unit it is in**.
    ``(x, y, z)``
        The same, in **bohr**. ⚠ This is the historical form and it keeps its meaning, so no
        stored number moves — but the geometry is in *Angstrom* by default, and a coordinate
        copied from there into here lands 1.89x too far out with no error and no clue. It is
        a shift of the origin ``L`` is defined about, so every orbital moment in the file is
        wrong and every one of them looks perfectly reasonable. Hence: it **warns**, once,
        naming the two united forms. Passing ``("bohr", ...)`` says the same thing silently.
    """
    coords = np.asarray(mol.atom_coords(), dtype=float)
    if origin is None or (isinstance(origin, str) and origin.lower() in ("mass", "com")):
        w = np.asarray(mol.atom_mass_list(), dtype=float)
        return coords.T @ w / w.sum(), "centre of mass"
    if isinstance(origin, str):
        key = origin.lower()
        if key in ("charge", "conc"):
            z = np.asarray([mol.atom_charge(i) for i in range(mol.natm)], dtype=float)
            return coords.T @ z / z.sum(), "centre of nuclear charge"
        if key in ("origin", "zero"):
            return np.zeros(3), "coordinate origin"
        raise ValueError(
            "unknown gauge origin {!r}; expected 'mass', 'charge', 'origin', "
            "('atom', k), ('bohr', x, y, z), ('angstrom', x, y, z), or three coordinates "
            "in bohr".format(origin))

    # -- the tagged forms, which are the ones that say what they mean -----------------------
    if (isinstance(origin, (tuple, list)) and origin
            and isinstance(origin[0], str)):
        tag, rest = origin[0].strip().lower(), list(origin[1:])
        if tag == "atom":
            if len(rest) != 1:
                raise ValueError("('atom', k) takes one atom: a 1-based number, an element "
                                 "symbol, or a label like 'Ti2'; got {!r}".format(origin))
            return _atom_gauge_origin(mol, coords, rest[0])
        if tag in ("bohr", "angstrom", "ang"):
            r = np.asarray(rest, dtype=float).ravel()
            if r.size != 3:
                raise ValueError("({!r}, x, y, z) takes three coordinates, got {}"
                                 .format(tag, r.size))
            if tag == "bohr":
                return r, "explicit (bohr)"
            return r * _bohr_per_angstrom(), "explicit (angstrom)"
        raise ValueError(
            "unknown gauge-origin form {!r}; the tagged forms are ('atom', k), "
            "('bohr', x, y, z) and ('angstrom', x, y, z)".format(origin))

    r = np.asarray(origin, dtype=float).ravel()
    if r.size != 3:
        raise ValueError(
            "an explicit gauge origin is three coordinates, got {}. Give ('bohr', x, y, z) "
            "or ('angstrom', x, y, z) to say which unit you mean, or ('atom', k) to put it "
            "on an atom".format(np.shape(origin)))
    log.warning(
        "the gauge origin (%.6f, %.6f, %.6f) was given as a bare tuple and is read as "
        "BOHR, which is what it has always meant -- but this molecule's geometry is in "
        "Angstrom, and a coordinate copied from there lands 1.89x too far out with no "
        "error. Every L matrix in the property dump is defined about this point. Pass "
        "('bohr', x, y, z) or ('angstrom', x, y, z) to say which you mean, or "
        "('atom', k) to put it on an atom", r[0], r[1], r[2])
    return r, "explicit (bohr, untagged)"


def _atom_gauge_origin(mol, coords: np.ndarray, key) -> Tuple[np.ndarray, str]:
    """``("atom", k)`` -> that atom's position [bohr] and a label naming it.

    ⚠ An element symbol that names **several** atoms is refused rather than resolved to the
    first: "put the origin on the chlorine" is not a statement about a molecule with three of
    them, and picking one silently would put the gauge origin somewhere the user did not
    choose — which no output would reveal.
    """
    from ..basis.atommap import parse_atom_key

    # ⚠ `atom_pure_symbol`, never `atom_symbol`. The latter returns PySCF's *decorated* symbol
    # for an atom that carries a per-atom basis or reference state ("Ti2"), and this molecule
    # may well have some. `parse_atom_key` is given plain element symbols in molecule order
    # everywhere else in the program and reads a label like "Ti2" as "atom 2, which is a Ti";
    # feeding it decorated symbols would make it compare a label against a label and answer a
    # different question for exactly the molecules the labels exist for.
    symbols = [str(mol.atom_pure_symbol(i)) for i in range(mol.natm)]
    kind, value = parse_atom_key(key, symbols)
    if kind == "element":
        matches = [i for i, s in enumerate(symbols)
                   if s.capitalize() == str(value).capitalize()]
        if len(matches) != 1:
            raise ValueError(
                "gauge_origin=('atom', {!r}) names {} atoms of this molecule ({}), so it "
                "does not name a point. Give the 1-based atom number or a label like "
                "{}{}".format(key, len(matches), [m + 1 for m in matches],
                              str(value).capitalize(), matches[0] + 1 if matches else 1))
        index = matches[0]
    else:
        index = int(value)
    return coords[index], "atom {} ({})".format(index + 1, symbols[index])


def ingest_property_integrals(mol, gauge_origin=None, *, picture_change: bool = False,
                              approx: str = "1e",
                              decoupling_options: Optional[Dict[str, object]] = None,
                              anomaly_picture_change: bool = False) -> PropertyIntegrals:
    """Orbital angular momentum about a gauge origin, as plain arrays.

    ⚠ **The gauge origin is a real choice and it changes the answer.** ``L`` is defined
    relative to it, so for a charged system every orbital moment matrix moves with it. The
    default is the centre of mass; it is recorded in :class:`PropertyIntegrals` and written
    into the dump header, because a stored moment matrix that does not say where its origin
    was is not interpretable.

    ``picture_change`` additionally builds the picture-changed moment operator
    (:func:`picture_changed_moment`). ⚠ **It is off by default and it is not free**: it costs
    a second four-component one-electron problem and its decoupling, and it changes the meaning
    of every moment matrix built from the result.
    """
    r_g, label = gauge_origin_for(mol, gauge_origin)
    with timer("angular-momentum integrals"):
        mol.set_common_orig(r_g)
        irxp = np.asarray(mol.intor("int1e_cg_irxp", comp=3), dtype=float)
    # <mu| r x nabla |nu> is exactly antisymmetric for real Gaussians, and L = -i(r x nabla)
    # is Hermitian only because of it. Cheap, structural, and it fails loudly if the integral
    # convention ever changes under us.
    asym = float(np.max(np.abs(irxp + irxp.transpose(0, 2, 1)))) if irxp.size else 0.0
    scale = float(np.max(np.abs(irxp))) or 1.0
    if asym > 1e-12 * scale:
        raise RuntimeError(
            "the angular-momentum integrals are not antisymmetric (max |A + A^T| = {:.2e}, "
            "{:.1e} relative); L = -i (r x nabla) would not be Hermitian and every moment "
            "matrix built from it would be meaningless".format(asym, asym / scale))
    if anomaly_picture_change and not picture_change:
        raise ValueError(
            "anomaly_picture_change=True asks for the g_e-2 term to be picture-changed while "
            "picture_change=False leaves the moment operator itself bare. That combination is "
            "a 2e-06 correction applied on top of an uncorrected operator, which is not a "
            "meaningful Hamiltonian; set picture_change=True as well or neither.")
    pc = None
    if picture_change:
        pc = picture_changed_moment(mol, r_g, approx=approx,
                                    decoupling_options=decoupling_options,
                                    anomaly_picture_change=anomaly_picture_change)
    return PropertyIntegrals(irxp=np.ascontiguousarray(irxp),
                             gauge_origin=np.ascontiguousarray(r_g), origin_label=label,
                             picture_change=pc)


# --- the picture change of the magnetic moment operator ------------------------------------

#: Relative tolerance on the non-relativistic-limit identity below. ⚠ It is a **refusal**, not
#: a warning: the identity is exact algebra and holds to ~1e-16 in practice, so a violation
#: means the spin structure or the gauge origin is wrong, and every moment matrix built from
#: the result would be Hermitian, plausible and wrong.
MOMENT_IDENTITY_TOL = 1e-10


def moment_memory_gb(nao: int) -> float:
    """Resident cost of a picture-changed ``(3, 2*nao, 2*nao)`` complex operator."""
    return res.array_gb((3, 2 * nao, 2 * nao), np.complex128)


@dataclass(frozen=True)
class PictureChangedMoment:
    """The magnetic moment operator with the X2C picture change applied to it.

    ⚠ **The picture-changed moment does not separate into an ``L`` part and an ``S`` part.**
    In the four-component theory the magnetic interaction is the *odd* operator ``c alpha.A``,
    so what is transformed is the whole moment; ``L`` and ``S`` remain perfectly good operators
    but their sum is no longer what ``mu`` is built from. Consumers therefore use
    :attr:`moment` in place of ``L + 2 S``, and add the ``g_e - 2`` anomaly separately.

    Attributes
    ----------
    moment : ndarray (3, 2*nao, 2*nao) complex
        ``(L + 2 S)`` picture-changed, in units of hbar and in the molecule's own AO basis.
        The ``2`` is Dirac's g factor, **not** ``g_e``: the QED anomaly is not part of
        ``c alpha.A`` and is added by the consumer as ``(g_e - 2) * S``.
    spin : ndarray (3, 2*nao, 2*nao) complex or None
        The picture-changed spin operator, when it was asked for. ⚠ Used for the anomaly term
        it changes the moment by ``O((g_e - 2)/c^2)``; ``None`` means the bare ``S`` is used
        there, which is the default and the measured-negligible choice.
    decoupling : str
        Which decoupling produced it (``"1e"``, ``"1e-dlu"``), for the stored provenance.
    identity_residual : float
        Relative residual of the non-relativistic-limit identity, measured on every build.
    """

    moment: np.ndarray
    spin: Optional[np.ndarray]
    decoupling: str
    identity_residual: float

    @property
    def nao(self) -> int:
        return int(self.moment.shape[-1] // 2)

    def label(self) -> str:
        anomaly = "picture-changed" if self.spin is not None else "bare S"
        return ("Peng-Reiher X2C picture change on the magnetic moment operator "
                "(decoupling={}, g_e-2 anomaly on {})".format(self.decoupling, anomaly))


def _odd_moment_blocks(xmol, r_g) -> np.ndarray:
    """``LS_k = <chi| c (r_g x sigma)_k |chi^S> = 1/2 <chi| (r_g x sigma)_k (sigma.p) |chi>``.

    The four-component magnetic moment operator is the derivative of the Dirac minimal-coupling
    term ``c alpha.A`` with ``A = 1/2 B x r_g``, i.e. ``mu = -c (r_g x alpha) mu_B``, which is
    **purely odd**: its ``LL`` and ``SS`` blocks vanish and only ``LS``/``SL`` survive.

    ⚠ **No factor of ``c`` appears here.** In the restricted-kinetic-balance normalization
    :func:`four_component_one_electron` fixes, the small-component basis carries ``1/(2c)``,
    which cancels the operator's explicit ``c`` exactly. A stray ``c`` or ``1/2c`` is the
    easiest way to produce a plausible wrong answer in this whole construction, and the
    identity checked by :func:`picture_changed_moment` is what catches it.

    ⚠ **The ``_spinor`` form of the integral is used deliberately.** ``libcint`` returns
    ``int1e_cg_sa10sp`` in the spherical basis as twelve components that are *not* in the
    ``int1e_spnucsp`` convention of three spin factors plus a spin-free part, so
    :func:`kuiva.spinor.expand.two_component_operator` cannot assemble it: an exhaustive search
    over both groupings, all four choices of spin-free component, all six permutations and both
    signs gets no closer than 1.08 relative. The spinor form is documented by the integral
    library as ``.5 rc cross sigma | sigma dot p`` — so the factor of one half is already inside
    it and no further scaling is applied — and it is mapped into this project's spin-blocked
    ``[alpha; beta]`` row layout by the unitary ``sph2spinor_coeff``. That mapping is
    *validated*, not assumed: it is exactly what the non-relativistic-limit identity tests.
    """
    xmol.set_common_orig(np.asarray(r_g, dtype=float).ravel())
    ua, ub = xmol.sph2spinor_coeff()
    u = np.vstack([np.asarray(ua), np.asarray(ub)])          # (2 nao, n2c), unitary
    raw = xmol.intor("int1e_cg_sa10sp_spinor", comp=3)
    return np.stack([u @ np.asarray(a) @ u.conj().T for a in raw])


def _anomaly_small_block(xmol) -> np.ndarray:
    """The ``SS`` block of the anomalous moment operator ``beta Sigma_k``, without the
    ``-1/(4 c^2)`` prefactor: ``<chi| (sigma.p) sigma_k (sigma.p) |chi>``.

    There is no integral for this one, so it is reduced to integrals that exist by the operator
    identity (``p_a`` commute with each other, and ``sigma_a sigma_k sigma_b`` is expanded with
    ``sigma_a sigma_b = delta_ab + i eps_abc sigma_c``)::

        (sigma.p) sigma_k (sigma.p) = 2 (sigma.p) p_k - sigma_k p^2

    with ``<p_a p_k> = <grad_a chi | grad_k chi>`` (one integration by parts) and
    ``<p^2> = 2 T``. ⚠ The ``W`` this expands over is **real and not antisymmetric**, unlike
    every spin-orbit integral in this front end, which is why it is assembled through
    :func:`kuiva.spinor.expand.sigma_dot` rather than ``two_component_operator`` — the latter
    would project it onto the antisymmetric part and silently return something smaller.
    """
    from ..spinor.expand import sigma_dot, spin_operator

    nao = xmol.nao
    # <grad_a chi | grad_k chi>, (3, 3, nao, nao) -- equals <p_a p_k> after one by-parts.
    d = np.asarray(xmol.intor("int1e_ipovlpip", comp=9)).reshape(3, 3, nao, nao)
    t = np.asarray(xmol.intor_symmetric("int1e_kin"))
    # sigma_k (x) p^2 = sigma_k (x) 2T = 4 * spin_operator(T), since spin_operator is sigma/2.
    sig_p2 = 4.0 * spin_operator(t)
    return np.stack([2.0 * sigma_dot(d[:, k]) - sig_p2[k] for k in range(3)])


def picture_changed_moment(mol, gauge_origin=None, *, approx: str = "1e",
                           decoupling_options: Optional[Dict[str, object]] = None,
                           light_speed: Optional[float] = None,
                           anomaly_picture_change: bool = False) -> PictureChangedMoment:
    """Apply the X2C picture change to the magnetic moment operator.

    The property operators this project normally uses are the **bare** non-relativistic ``L``
    and ``S``, used unchanged in the two-component basis — the same choice OpenMolcas RASSI
    makes, which is what keeps a cross-code comparison like-for-like. This function is the
    alternative: it transforms the four-component moment operator with the same ``X`` and ``R``
    that decouple the Hamiltonian, which is the picture change that choice omits.

    ⚠ **It is not the default and it changes the meaning of a stored moment matrix.** A file
    whose ``mu`` was built this way is not comparable element-for-element with one that was
    not, and its header must say so.

    The construction, and the test that it is right
    -----------------------------------------------
    In units of ``-mu_B``, the four-component operator has ``LL = SS = 0`` and
    ``LS_k = 1/2 <chi| (r_g x sigma)_k (sigma.p) |chi>`` (see :func:`_odd_moment_blocks`), and
    the transformation is the *same* ``R^dag (A_LL + A_LS X + X^dag A_SL + X^dag A_SS X) R``
    the Hamiltonian goes through. The non-relativistic limit is then exact algebra::

        (r_g x sigma)_k (sigma.p) + (sigma.p) (r_g x sigma)_k = 2 L_k + 2 sigma_k = 2 (L + 2S)_k

    so ``LS + LS^dag = L + 2 S`` identically, and since ``X -> 1`` and ``R -> 1`` as ``c -> inf``
    the transformed operator returns the bare one. **That identity is checked on every build and
    a violation raises**, because it is the one thing that fails loudly if the spin mapping, the
    gauge origin or the normalization is wrong.

    Parameters
    ----------
    approx : str
        The decoupling, which must be the one the Hamiltonian uses: ``"1e"`` (exact molecular)
        or ``"1e-dlu"`` (local). ⚠ **No statement about the accuracy of a moment operator
        transformed through a DLU decoupling exists**, so a ``"1e-dlu"`` moment is a research
        quantity and nothing computed from it may be quoted as a spectroscopic accuracy.
    anomaly_picture_change : bool
        Also return the picture-changed **spin** operator, for the ``g_e - 2`` anomaly term.
        Off by default: the anomaly is an even operator whose small-component block enters at
        ``O((g_e - 2)/c^2)``, and using the bare ``S`` there is the measured-negligible choice.

    References
    ----------
    * Picture change of property operators under X2C: D. Peng, M. Reiher, J. Chem. Phys. 136,
      244108 (2012), doi:10.1063/1.4729788.
    * The restricted-kinetic-balance four-component setup and the decoupling: W. Kutzelnigg,
      W. Liu, J. Chem. Phys. 123, 241102 (2005), doi:10.1063/1.2137315.
    """
    from pyscf.x2c import x2c

    from ..spinor.expand import spin_block_diagonal, spin_operator
    from ..x2c.decouple import FourComponentBlocks, decoupling_matrices, picture_change

    if approx not in ("1e", "1e-dlu"):
        raise NotImplementedError(
            "the property picture change is implemented for the exact molecular decoupling "
            "(approx='1e') and the local one (approx='1e-dlu'), not for {!r}. The decoupling "
            "used for the moment operator must be the one the Hamiltonian uses, so this is a "
            "refusal rather than a silent substitution.".format(approx))

    r_g, _ = gauge_origin_for(mol, gauge_origin)
    fc = four_component_one_electron(mol, uncontract=True, light_speed=light_speed)

    helper = x2c.SpinOrbitalX2CHelper(mol)
    helper.xuncontract = True
    xmol, _ = helper.get_xmol(mol)
    if int(xmol.nao) != int(fc.nao):
        raise RuntimeError(
            "the working basis of the four-component blocks ({} functions) and of the property "
            "integrals ({}) differ; they must be the same decontracted basis or the picture "
            "change would transform one operator in another's basis"
            .format(fc.nao, xmol.nao))

    res.require("picture-changed moment operator", 2.0 * moment_memory_gb(int(fc.nao)),
                note="3 x (2 nao)^2 complex, working basis nao = {}".format(fc.nao))

    with timer("property picture change"):
        ls = _odd_moment_blocks(xmol, r_g)

        # ⚠ The load-bearing check: the non-relativistic-limit identity. It fails loudly on a
        # wrong spin mapping, a gauge origin applied to the wrong Mole, or a stray factor of c.
        irxp = np.asarray(xmol.intor("int1e_cg_irxp", comp=3), dtype=float)
        bare = np.stack([spin_block_diagonal(-1j * lk) for lk in irxp]) \
            + 2.0 * spin_operator(np.asarray(xmol.intor_symmetric("int1e_ovlp")))
        resid = max(float(np.max(np.abs(ls[k] + ls[k].conj().T - bare[k])))
                    / (float(np.max(np.abs(bare[k]))) or 1.0) for k in range(3))
        if resid > MOMENT_IDENTITY_TOL:
            raise RuntimeError(
                "the four-component moment operator fails its non-relativistic-limit identity "
                "by {:.2e} relative (tolerance {:.0e}): <(r x sigma)(sigma.p)> + h.c. must "
                "equal L + 2S exactly. The spin mapping, the gauge origin or the "
                "small-component normalization is wrong, and every moment matrix built from "
                "this operator would be Hermitian, plausible and wrong."
                .format(resid, MOMENT_IDENTITY_TOL))

        if approx == "1e-dlu":
            from ..x2c.local import local_decoupling_matrices
            opts = dict(decoupling_options or {})
            opts.pop("report", None)
            x, r = local_decoupling_matrices(fc.hcore, fc.overlap, fc.light_speed,
                                             molecular_partition(mol, fc), **opts)
        else:
            if decoupling_options:
                raise ValueError(
                    "decoupling_options={} apply only to the local (DLU) decoupling, and this "
                    "moment operator is being built with approx={!r}"
                    .format(sorted(decoupling_options), approx))
            x, r = decoupling_matrices(fc.hcore, fc.overlap, fc.light_speed)

        zero = np.zeros_like(ls[0])
        moment = np.stack([
            fc.contract(picture_change(
                FourComponentBlocks(ll=zero, ls=ls[k], sl=ls[k].conj().T, ss=zero), x, r))
            for k in range(3)])

        spin = None
        if anomaly_picture_change:
            c = float(fc.light_speed)
            small = _anomaly_small_block(xmol)
            s_big = spin_operator(np.asarray(xmol.intor_symmetric("int1e_ovlp")))
            # beta Sigma_k has LL = sigma_k and SS = -sigma_k; in units of S = sigma/2 the
            # blocks below carry the factor of one half, so the result is a spin operator.
            spin = np.stack([
                fc.contract(picture_change(
                    FourComponentBlocks(ll=2.0 * s_big[k], ls=zero, sl=zero,
                                        ss=-small[k] * (0.25 / c ** 2)), x, r)) * 0.5
                for k in range(3)])

    worst = max(float(np.max(np.abs(m - m.conj().T))) for m in moment)
    if worst > 1e-10 * max(float(np.max(np.abs(moment))), 1.0):
        log.warning("the picture-changed moment operator is not Hermitian (max |A - A^dag| = "
                    "%.2e); the decoupling matrices are suspect", worst)
    return PictureChangedMoment(moment=np.ascontiguousarray(moment), spin=spin,
                                decoupling=approx, identity_residual=float(resid))


@dataclass(frozen=True)
class MoleculeSpec:
    """The input needed to rebuild this calculation's ``Mole`` — geometry, basis, charge.

    ⚠ **This is plain data, not a ``Mole``**: no ``Mole`` crosses the ingestion boundary and
    nothing downstream gains a PySCF dependency by carrying this. It exists for the one
    operation that genuinely needs the *basis functions* back after ingestion: the cross-basis
    overlap ``<AO(target)|AO(source)>`` behind :func:`cross_overlap`, which is what lets a
    converged orbital set be projected onto a different basis set.

    ⚠ It is rebuilt through :func:`build_mole`, from the same registry names the run was
    given, rather than from the primitives :class:`kuiva.basis.layout.AOLayout` carries: the
    layout stores the integral library's *internal* contraction coefficients (primitive
    normalisation already folded in), and feeding those back through the basis-input path
    normalises them a second time. The result is a basis that looks right, integrates to
    plausible numbers, and is not the one the calculation ran in.

    It duck-types as a :class:`kuiva.interface.api.Molecule` for :func:`build_mole`, which is
    what makes the rebuilt basis the original one rather than a reconstruction of it.
    """

    atoms: Tuple[Tuple[str, Tuple[float, float, float]], ...]
    basis: object
    charge: int = 0
    spin: int = 0
    unit: str = "Angstrom"
    #: The per-atom reference-configuration spec, carried so the rebuilt ``Mole`` gets the
    #: same decorated atom labels (``"Ti2"``) and therefore the same per-atom basis routing.
    configuration: object = None

    @classmethod
    def from_molecule(cls, molecule, configuration=None) -> "MoleculeSpec":
        """Capture a (duck-typed) Kuiva ``Molecule`` and the run's ``configuration``."""
        return cls(atoms=tuple((str(sym), tuple(float(x) for x in xyz))
                               for sym, xyz in molecule.atoms),
                   basis=molecule.basis, charge=int(molecule.charge),
                   spin=int(molecule.spin), unit=str(getattr(molecule, "unit", "Angstrom")),
                   configuration=configuration)

    @property
    def elements(self) -> Tuple[str, ...]:
        return tuple(sym.capitalize() for sym, _ in self.atoms)

    def __repr__(self) -> str:
        return "MoleculeSpec({} atoms: {}, basis={!r})".format(
            len(self.atoms), " ".join(self.elements), self.basis)


def cross_overlap(source: MoleculeSpec, target: MoleculeSpec, *,
                  verbose: int = 0) -> np.ndarray:
    """``<AO(target) | AO(source)>``, shape ``(nao_target, nao_source)``.

    The one integral a basis-set projection needs (:mod:`kuiva.orth.project`): the target
    basis' representation of every source basis function. Both molecules are rebuilt through
    :func:`build_mole`, so each side is exactly the basis its own calculation ran in.

    ⚠ **The two must be the same molecule.** Elements and their order are checked and a
    mismatch is refused; the nuclear coordinates are checked and a mismatch **warns** rather
    than refusing, because carrying an orbital set from one geometry to the next along a scan
    is a legitimate and useful thing to do — but it is not what a basis-set projection is, and
    doing it by accident produces a guess that is merely mediocre rather than obviously wrong.
    """
    from pyscf import gto

    mol_s = build_mole(source, verbose=verbose, configuration=source.configuration)
    mol_t = build_mole(target, verbose=verbose, configuration=target.configuration)
    sym_s = [mol_s.atom_pure_symbol(i) for i in range(mol_s.natm)]
    sym_t = [mol_t.atom_pure_symbol(i) for i in range(mol_t.natm)]
    if sym_s != sym_t:
        raise ValueError(
            "a basis-set projection needs the same molecule on both sides; the source has "
            "atoms {} and the target has {}".format(sym_s, sym_t))
    dr = (float(np.max(np.abs(np.asarray(mol_s.atom_coords())
                              - np.asarray(mol_t.atom_coords())))) if sym_s else 0.0)
    if dr > 1e-8:
        log.warning("the two bases sit on geometries differing by up to %.3e bohr; the "
                    "projection is still well defined, but it is now a projection across "
                    "geometries as well as across bases", dr)

    s_cross = np.asarray(gto.intor_cross("int1e_ovlp", mol_t, mol_s), dtype=float)
    if s_cross.shape != (mol_t.nao, mol_s.nao):
        raise RuntimeError("cross overlap has shape {} for {} x {} basis functions"
                           .format(s_cross.shape, mol_t.nao, mol_s.nao))
    return np.ascontiguousarray(s_cross)


@dataclass(frozen=True)
class ScalarX2CData:
    """Self-contained scalar-relativistic X2C reference (the ingestion boundary).

    The scalar quantities are real (the scalar guess carries no SOC); spin-orbit coupling
    arrives separately in :attr:`soc` and is applied at the multireference level.
    ``eri``, ``df_cderi`` and ``factors`` are mutually exclusive: exactly one is populated
    per ``fit_route`` (``"conventional"``, ``"df"``, ``"direct"``).

    **Restricted and unrestricted references.** RHF/ROHF give one set of MOs and
    ``mo_coeff`` has shape ``(nao, nmo)``; UHF gives two and it has shape ``(2, nao, nmo)``
    with ``unrestricted = True``. Use :meth:`mo_sets` rather than branching on the shape.

    ⚠ **``reference = "aoc"`` (:func:`run_scalar_aoc`) is a restricted set with FRACTIONAL
    occupations**, and two fields then say less than they appear to. ``mo_occ`` runs over
    ``[0, 2]`` continuously rather than taking the values 0 and 2, so a consumer that counts
    occupied orbitals with ``mo_occ > 0`` counts a whole partly filled shell. And ``nelec`` is
    the formal ``(ceil(N/2), floor(N/2))`` split: a spin-restricted average of configuration
    shares every open-shell electron over both spins, so there is no per-spin count to report
    and only ``nelec_total`` means anything. ⚠ **Feeding an AOC container into the pipeline
    stages is not validated and is not what the function is for** — it exists so that atomic
    *shell* quantities have a spherical, single-radial-function reference to come from.
    """
    # dimensions / bookkeeping
    nao: int
    nmo: int
    nelec: Tuple[int, int]                 # (n_alpha, n_beta)
    e_scf: float                           # scalar X2C SCF total energy [Eh]
    converged: bool
    # one-electron quantities (AO basis)
    s_ao: np.ndarray                       # (nao, nao) overlap
    h_x2c: np.ndarray                      # (nao, nao) SPIN-FREE X2C one-electron Hamiltonian
    # molecular orbitals: (nao, nmo), or (2, nao, nmo) when unrestricted
    mo_coeff: np.ndarray
    mo_energy: np.ndarray                  # (nmo,) or (2, nmo)
    mo_occ: np.ndarray                     # (nmo,) or (2, nmo)
    # two-electron integrals: one of the following, per fit_route
    fit_route: str                         # "conventional" | "direct" | "df"
    eri: Optional[np.ndarray] = None       # 8-fold packed (nao_pair_pair,) if conventional
    df_cderi: Optional[np.ndarray] = None  # (naux, nao_pair) DF factors if df
    aux_name: Optional[str] = None
    #: Finished three-index factors, on the **integral-direct** route only. That route
    #: decomposes in the front end, while the integrals can still be evaluated and without
    #: ever storing them, so what it can hand on is the factorization rather than its input.
    #: ⚠ The Cholesky threshold and the pivot choice are therefore arguments of the front-end
    #: call on that route; :meth:`kuiva.integrals.transform.ThreeIndexAO.from_scalar_data`
    #: hands these back unchanged and says so if it is asked for a different threshold.
    factors: Optional[object] = None
    # spin-orbit coupling (None if ingestion was disabled)
    soc: Optional[SpinOrbitX2C] = None
    # reference type
    reference: str = "rhf"                 # "rhf" | "rohf" | "uhf"
    unrestricted: bool = False
    #: <S^2> minus its exact value; unrestricted references only, None otherwise.
    s2_deviation: Optional[float] = None
    # provenance
    basis_meta: Dict[str, str] = field(default_factory=dict)
    e_nuc: float = 0.0
    #: AO basis layout — geometry, shells and per-AO labels. Mostly *analysis* metadata
    #: (Loewdin populations, the molden dump), ⚠ **but no longer only that**: its ``ao_shell``
    #: and ``ao_atom`` columns build the symmetry-orbit labelling the Cholesky decomposition
    #: pivots on, which is what makes the factorization's spherical symmetry
    #: exact rather than threshold-dependent. It stays optional — a container built by hand in
    #: a test is still valid — but the factorization then warns and falls back to plain column
    #: pivoting. Everything the front-end builds carries it.
    ao_layout: Optional[AOLayout] = None
    #: Angular-momentum integrals about the gauge origin, for the property dump. Like
    #: :attr:`ao_layout` this is *output* metadata, read by no part of the calculation, and
    #: optional for the same reason.
    properties: Optional[PropertyIntegrals] = None
    #: Abelian double-group symmetry of this reference (:class:`kuiva.symm.MolecularSymmetry`)
    #: — the group actually used, what was asked for, what the geometry has, and one integer
    #: label per scalar MO. ``None`` means **no symmetry**, and every consumer then takes the
    #: path it took before labels existed. ⚠ When it is not ``None``, :attr:`mo_coeff` is the
    #: *symmetry-adapted* orbital set: degenerate blocks may have been rotated inside
    #: themselves so that each orbital is an eigenvector of every group operation. That
    #: changes no density, no energy and no observable, and it is the only way a label can
    #: exist for an orbital the SCF was free to return as an arbitrary mixture.
    symmetry: Optional[object] = None
    #: Per-element free-atom reference orbitals (:class:`kuiva.basis.reference.
    #: AtomicReferenceSet`), behind the atomic-reference charges of
    #: :mod:`kuiva.props.population`. Opt-in (``atomic_reference=True``) because it costs one
    #: small atomic SCF per unique element — sub-second for a light element, ~10 s for a
    #: lanthanide — and it must be computed *here*: the analysis layer has no integral
    #: library, and the reference must be in the molecule's own basis. Analysis metadata
    #: only; no part of the calculation reads it.
    atomic_reference: Optional[object] = None
    #: What this calculation was asked to run on (:class:`MoleculeSpec`), kept so the basis
    #: can be rebuilt for the one operation that needs it after ingestion: the cross-basis
    #: overlap behind :func:`cross_overlap`, i.e. projecting an orbital set onto a different
    #: basis set. Plain data; optional, so a container built by hand in a test is still valid
    #: — a projection then refuses and says which side is missing it.
    molecule: Optional[MoleculeSpec] = None

    @property
    def nelec_total(self) -> int:
        return self.nelec[0] + self.nelec[1]

    @property
    def has_soc(self) -> bool:
        return self.soc is not None

    def mo_sets(self) -> Tuple[np.ndarray, ...]:
        """The MO coefficient sets: ``(C,)`` if restricted, ``(C_alpha, C_beta)`` if not.

        Iterating this is the safe way to consume the orbitals — it is the same code for both
        reference types, and it cannot silently treat a ``(2, nao, nmo)`` array as ``(nao,
        nmo)`` of a differently sized basis.
        """
        return tuple(self.mo_coeff) if self.unrestricted else (self.mo_coeff,)

    def spin_contamination(self) -> Optional[float]:
        """``<S^2>`` deviation from its exact value (unrestricted references only).

        Zero for RHF/ROHF by construction, which is why it is ``None`` there rather than
        ``0.0``: "not applicable" and "measured to be zero" are different statements.
        """
        return self.s2_deviation

    def __repr__(self) -> str:
        two_e = f"eri{self.eri.shape}" if self.eri is not None else (
            f"df_cderi{self.df_cderi.shape}" if self.df_cderi is not None else (
                f"factors(naux={self.factors.naux})" if self.factors is not None else "none"))
        return (f"ScalarX2CData(nao={self.nao}, nmo={self.nmo}, nelec={self.nelec}, "
                f"E={self.e_scf:.8f} Eh, conv={self.converged}, ref={self.reference}, "
                f"route={self.fit_route}, {two_e}, soc={self.has_soc})")


def _resolve_basis(atoms, basis) -> Tuple[list, list]:
    """Resolve a basis spec into one family per atom. Returns ``(families, is_specific)``.

    A string applies one family to every atom. A dict assigns per element, per atom label
    (``"O3"``), or per 1-based atom number, with an optional ``"default"`` entry filling
    every atom no more specific key covers (:mod:`kuiva.basis.atommap` — the same addressing
    the reference configurations use). Without a ``"default"`` the assignment must be
    complete, exactly as before. Registry coverage and compatibility run over the whole
    per-atom assignment, so two families on one element are checked as the pair they are.
    """
    from ..basis.atommap import resolve_atom_assignments

    symbols = [a[0].capitalize() for a in atoms]
    if isinstance(basis, str):
        families, specific = [basis] * len(symbols), [False] * len(symbols)
    else:
        families, specific = resolve_atom_assignments(basis, symbols, what="basis")
        missing = sorted({s for s, f in zip(symbols, families) if f is None})
        if missing:
            raise ValueError("no basis assigned for atom(s) {} — add element entries or a "
                             "\"default\" entry".format(missing))

    report = reg.check_consistency(sorted({(s, f) for s, f in zip(symbols, families)}))
    if not report.ok:
        raise ValueError("basis consistency check failed:\n  " + "\n  ".join(report.errors))
    return families, specific


def build_mole(molecule, verbose: int = 0, configuration=None):
    """Build a PySCF ``Mole`` from a Kuiva ``Molecule`` (duck-typed), via the registry.

    ``molecule`` must expose ``atoms`` (list of ``(symbol, (x, y, z))``), ``charge``,
    ``spin`` (2S), ``basis`` and ``unit``. ``configuration`` is the per-atom reference-state
    spec (element / label / 1-based number keys, values an oxidation state or a
    configuration), resolved here through the curated-table checks because the *labelling*
    of the molecule depends on it: ⚠ **an atom whose basis or reference state differs from
    another atom of the same element gets a decorated PySCF symbol** (``"Ti2"``, 1-based),
    which is what lets every per-label consumer downstream — the atomic mean field's
    assembly first among them — treat the two atoms differently. Atoms with nothing
    atom-specific keep their plain element symbol, so outputs, caches and provenance are
    unchanged wherever the feature is not used.
    """
    from pyscf import gto

    from ..amf.oxidation import resolve_reference_configuration
    from ..basis.atommap import resolve_atom_assignments

    symbols = [a[0].capitalize() for a in molecule.atoms]
    families, _ = _resolve_basis(molecule.atoms, molecule.basis)

    spec_values, _ = resolve_atom_assignments(configuration, symbols,
                                              what="reference configuration",
                                              allow_scalar=False)
    resolved: Dict[tuple, tuple] = {}
    atom_configs = []
    for sym, value in zip(symbols, spec_values):
        key = (sym, repr(value))
        if key not in resolved:          # one resolution (and one warning) per unique spec
            resolved[key] = resolve_reference_configuration(sym, value)
        atom_configs.append(resolved[key])

    # Decoration: every atom of an element whose atoms are not all alike gets its own label.
    varies = {s for s in set(symbols)
              if len({(f, c[0]) for s2, f, c in zip(symbols, families, atom_configs)
                      if s2 == s}) > 1}
    labels = [("{}{}".format(s, i + 1) if s in varies else s)
              for i, s in enumerate(symbols)]

    pyscf_basis: Dict[str, object] = {}
    meta: Dict[str, str] = {}
    atom_basis: Dict[str, str] = {}
    for label, sym, fam_name in zip(labels, symbols, families):
        if label in pyscf_basis:
            continue
        fam = reg.get_family(fam_name)
        pyscf_basis[label] = reg.resolve_for_pyscf(fam_name, [sym])[sym] \
            if fam.provider is reg.Provider.BSE else fam.provider_name
        meta[label] = f"{fam.name} [{fam.rel_treatment.value}, {fam.contraction.value}, " \
                      f"fit={fam.fit_route().value}]"
        atom_basis[label] = fam_name

    atom_str = [(label, tuple(a[1])) for label, a in zip(labels, molecule.atoms)]
    mol = gto.M(atom=atom_str, basis=pyscf_basis, charge=molecule.charge,
                spin=molecule.spin, unit=molecule.unit, verbose=verbose)
    mol.__dict__["_kuiva_basis_meta"] = meta
    mol.__dict__["_kuiva_atom_basis"] = atom_basis
    mol.__dict__["_kuiva_atom_labels"] = labels
    mol.__dict__["_kuiva_atom_families"] = list(families)
    # ⚠ Plain data only: PySCF's SCF checkpoint JSON-serializes ``mol.__dict__``, so the
    # resolved configurations are stashed as (occupations, label, is_default) and rebuilt
    # by _stashed_configs() — equality (and therefore every cache key) is by occupations.
    mol.__dict__["_kuiva_atom_configs"] = [
        (list(cfg.occupations), cfg.label, bool(d)) for cfg, d in atom_configs]
    mol.__dict__["_kuiva_config_given"] = configuration is not None
    return mol


def _stashed_configs(mol):
    """Rebuild the per-atom ``(AtomicConfiguration, is_default)`` list off the JSON-safe
    stash."""
    from ..amf.configuration import AtomicConfiguration

    return [(AtomicConfiguration(occ, label=label), bool(d))
            for occ, label, d in mol.__dict__["_kuiva_atom_configs"]]


def _choose_fit(atom_basis: Dict[str, str], fitting: Optional[str],
                auxbasis: Optional[object] = None) -> Tuple[str, Optional[object]]:
    """Decide the two-electron route and auxiliary. Returns ``(fit_route, aux)``.

    **Cholesky is the default in every case** (a user decision: a Cholesky threshold is an
    error bound, a fitting error is not bounded at all). The
    bridge ingests conventional ERIs and :mod:`kuiva.integrals.transform` decomposes them;
    the resulting error is bounded by a threshold the user sets, which a density fit's error
    is not. Density fitting is never selected automatically — not even where the registry
    recommends an auxiliary — because the recommended sets are Coulomb-fitting sets, and a
    J-fit reproduces individual transformed integrals to only ~1e-3 Eh (measured).

    Density fitting is used **only when the user asks for it**, either by ``fitting="df"`` or
    by supplying ``auxbasis``. That is deliberate: with a user-chosen auxiliary the accuracy
    of the fit is the user's decision, and the code's job is to apply it faithfully and say
    what it did — not to make that choice on their behalf.

    ``fitting="cholesky-direct"`` is the same Cholesky factorization with the integrals
    evaluated as the decomposition asks for them (:class:`DirectERIMatrix`) instead of stored
    first. Same threshold, same error bound, same object downstream; it removes the
    ``O(nao^4/8)`` array, which is what bounds the size of system that fits. It is a value on
    *this* axis and not a separate switch, so it cannot be combined with density fitting and
    there is no contradiction to refuse.

    ``fitting=None`` (and its explicit spelling ``"auto"``) returns the sentinel ``"auto"``:
    **which Cholesky route serves the default is decided by the memory plan**, in
    :func:`_auto_fit_route`, once the AO count and the configured limit are both known. The
    stored route wherever its plan fits — it is never slower than the direct one by more
    than a few per cent and is up to ~40% faster on a small system — and the integral-direct
    route where the stored plan exceeds the limit, which is the one regime where the direct
    route is *more* efficient overall (a calculation that runs, against one that is refused).
    There is no fixed size cutoff: the measured CPU costs of the two routes stay within a
    few per cent of each other from ~160 AOs up, so the array that only the stored route
    needs is the entire decision.
    """
    if auxbasis is not None and fitting in (None, "df"):
        return "df", auxbasis
    if fitting == "df":
        # No auxiliary given: fall back on the registry's recommendation, which is a Coulomb
        # fitting set. from_df() warns about exactly this when the factors are used.
        auxes = {reg.recommended_auxiliary(b) for b in atom_basis.values()}
        auxes.discard(None)
        aux = auxes.pop() if len(auxes) == 1 else "def2-universal-jkfit"
        log.warning("density fitting requested without an auxiliary basis; falling back on "
                    "the registry recommendation %r. Supply auxbasis= explicitly for "
                    "correlated work - a Coulomb-fitting set is not accurate enough for "
                    "individual transformed integrals.", aux)
        return "df", aux
    if fitting in ("cholesky-direct", "direct"):
        return "direct", None
    if fitting in (None, "auto"):
        return "auto", None
    if fitting in ("conventional", "cholesky"):
        return "conventional", None
    raise ValueError("unknown fitting route {!r}; expected 'auto', 'conventional', "
                     "'cholesky', 'cholesky-direct', 'df' or None".format(fitting))


def _auto_fit_route(nao: int, *, nelec: int, n_active: Optional[int] = None,
                    screening: bool = False) -> Tuple[str, str]:
    """Resolve ``fit_route="auto"`` against the memory plan. Returns ``(route, note)``.

    The stored (conventional) Cholesky route whenever its own plan fits the configured
    limit, the integral-direct route when it does not. The decision is the *plan*, not a
    size constant, because the crossover is a property of the machine: on the two-electron
    phase the two routes' CPU costs sit within a few per cent of each other from ~160 AOs
    up (stored 8.7 vs direct 11.2 CPU s at nao = 105, 121.8 vs 119.7 at nao = 234, 433 vs
    412–633 at nao = 324 depending on the batch-cache budget; 4 threads, Haswell), so CPU
    cost never separates them — the ``O(nao^4/8)`` array only the stored route must hold is
    the whole difference, and whether it fits is exactly what the plan already computes.

    ⚠ The direct route is *not* asymptotically cheaper on CPU, and the entry predicting a
    cost crossover near nao ≈ 350 was wrong: its batch evaluator loses the bra↔ket shell
    symmetry (``aosym="s2ij"`` against the stored fill's ``s8``) and re-evaluates batches
    its cache cannot hold, which together outweigh evaluating fewer columns at every size
    measured. What it buys is memory, and the note returned says so in the output file.

    ``note`` is non-empty only when the resolution *changed* something a user would see —
    i.e. when the direct route was chosen — so the output of every calculation the stored
    route serves is unchanged, byte for byte, by the existence of this mechanism.
    """
    budget = res.BUDGET
    limits = budget.limits
    limit = limits.memory_gb if limits is not None else float("inf")
    peak = res.plan_peak_gb(memory_plan(nao, conventional=True, n_active=n_active,
                                        nelec=nelec, screening=screening), budget=budget)
    if peak <= limit:
        return "conventional", ""
    return "direct", ("chosen automatically: the stored-integral route plans "
                      "{:.2f} GB against the {:.2f} GB limit".format(peak, limit))


# The vectors-per-AO estimate is defined where the decomposition is (it seeds that array's
# capacity as well as this plan) and re-exported here, because this is where the pre-flight
# reads it and where every caller has always imported it from.
CHOLESKY_VECTORS_PER_AO = CHOLESKY_VECTORS_PER_AO


def _nevpt2_block_split(n: int, n_occ: int, n_active: int) -> Tuple[int, int, int]:
    """``(inactive, active, virtual)`` spinor counts a memory plan must assume.

    The perturbation holds four three-index blocks at once, totalling
    ``naux * (n_virtual + n_active) * (n_inactive + n_active)``, and with ``n_virtual =
    n - n_inactive - n_active`` the only unknown left at pre-flight is the inactive count:
    it is the electron count minus however many electrons the active space takes, and the
    active electron count is not part of the plan's input.

    That product is **concave** in ``n_inactive``, with its maximum at ``(n - n_active)/2``.
    The plan therefore assumes the admissible split closest to that maximum — ``n_inactive``
    can never exceed the number of occupied spinors, which is the electron count — so the
    estimate bounds every split the calculation could turn out to have, without the safety
    factor that a padded sizing function would be. On any real system the occupied count is
    far below ``n/2`` and the bound is simply "every occupied spinor is inactive".
    """
    hi = max(0, min(int(n_occ), n - int(n_active)))
    n_inactive = min(hi, max(0, (n - int(n_active)) // 2))
    return n_inactive, int(n_active), n - n_inactive - int(n_active)


def memory_plan(nao: int, *, conventional: bool = True, direct: bool = False,
                shell_ao_max: Optional[int] = None, n_shells: Optional[int] = None,
                naux: Optional[int] = None,
                nspinor: Optional[int] = None, n_active: Optional[int] = None,
                nelec: Optional[int] = None,
                nevpt2: bool = False, screening: bool = False) -> list:
    """Phase-by-phase memory estimate for a calculation on ``nao`` AO functions.

    Everything here is a function of dimensions only — no array exists yet — which is what
    lets the whole plan be printed and judged before the SCF starts. Returns a list of
    :class:`kuiva.util.resources.PhaseEstimate` for :func:`kuiva.util.resources.preflight`.

    ``n_active`` adds the multireference phases when the active space is already known;
    ``nevpt2`` adds the 4-RDM, which is ``n_active^8`` and is by a wide margin the largest
    array in the program (direct contraction was chosen over a cumulant approximation, so
    there is no cheaper route to it). ``screening`` adds the two-electron picture change
    (one four-component atomic solve per unique element). ``nelec`` is the electron count,
    which is what bounds the transformed three-index blocks (see below) and is known as soon
    as the molecule is built.

    ``conventional`` is whether the ``O(nao^4/8)`` AO integral array is materialized;
    ``direct`` is the integral-direct Cholesky route, which does not materialize it and holds
    shell-pair batches of integrals instead, sized from ``shell_ao_max`` (AO functions of the
    largest shell) and ``n_shells`` (how many batches there are to cache). The two are
    alternatives, and both are false for density fitting.

    ⚠ **The three-index MO phase is planned as the orbital *blocks* the pipeline builds, never
    as the square** ``B^P_{pq}``. Nothing on any production path transforms every spinor
    against every spinor: the orbital optimizer and the CI drivers hold ``B^P_{p,active}``,
    and the perturbation holds four blocks over its space pairs. Budgeting the square refuses
    calculations that would have run — and a refusal on an array nobody allocates teaches the
    user to raise the limit blindly, which is the one response that also defeats the next
    refusal, the real one. The bound is stated in the plan's own notes so that a user who does
    hit it knows which block grew.
    """
    from ..amf.correction import correction_memory_gb
    from ..integrals.transform import (factor_memory_gb, mo_block_memory_gb,
                                       transform_buffer_gb)
    from ..mcscf.orbopt import cas_integrals_memory_gb, hessian_response_memory_gb

    estimated_naux = naux is None
    naux = int(naux if naux is not None else CHOLESKY_VECTORS_PER_AO * nao)
    n = int(nspinor if nspinor is not None else 2 * nao)
    # Without an electron count nothing bounds the occupied space but the whole spinor set,
    # which is what the block bound below then falls back on. Both callers in this file supply
    # it; the fallback exists so that ``memory_plan`` remains usable as a bare estimate.
    n_occ = int(nelec) if nelec is not None else n
    phases = [res.PhaseEstimate(
        name="scalar X2C SCF", governed=False,
        external_note="allocated by PySCF, not accounted for here; it is given "
                      "mol.max_memory = the rest of the limit")]

    ints = res.PhaseEstimate(name="two-electron integrals", advice=[
        "fitting=\"cholesky-direct\" evaluates the integrals as the decomposition asks for "
        "them and never forms the conventional array, at the same threshold and the same "
        "error bound",
        "density fitting with an explicit auxiliary basis does not form it either, at the "
        "cost of an error the auxiliary sets rather than one the user does"])
    if conventional:
        ints.allocations.append(res.PlannedAllocation(
            "conventional AO ERI array", eri_memory_gb(nao),
            note="nao = {}; grows as nao^4".format(nao)))
    ints.allocations.append(res.PlannedAllocation(
        "three-index AO factors", factor_memory_gb(nao, naux),
        note="naux {} {}".format("~" if estimated_naux else "=", naux)))
    if direct:
        # ⚠ What replaces the array above, and the whole of what the direct route holds at
        # once: shell-pair batches of integrals. Transient — each is re-used across the
        # columns of a symmetry orbit and then dropped — and stated here as the same
        # ``min(what it wants, what it is allowed)`` the kernel itself computes, so the plan
        # is what the evaluator holds rather than what it could get away with. Sizing it from
        # the free budget alone, unstated, was measured holding gigabytes of batches under a
        # plan that claimed one.
        batch = direct_block_memory_gb(nao, shell_ao_max if shell_ao_max else 1)
        want = (direct_cache_memory_gb(nao, shell_ao_max, n_shells)
                if shell_ao_max and n_shells else batch)
        ints.allocations.append(res.PlannedAllocation(
            "integral-direct shell-pair batches",
            min(want, res.BUDGET.transient_gb()), resident=False,
            note="{:.3f} GB a batch; a smaller share only costs re-evaluations".format(
                batch)))
    phases.append(ints)

    if screening:
        # ⚠ Only the assembled molecular correction is planned from ``nao``. The
        # four-component atomic arrays are per element and depend on that element's
        # *primitive* count, which is not knowable from the molecular AO total — so they are
        # checked where they are allocated instead (``kuiva.amf.pyscf_dhf``, one ``require``
        # per element). Inventing a bound here would mean guessing the heaviest element's
        # decontracted size, and the resource budget refuses on estimates, so a guess would refuse runs.
        phases.append(res.PhaseEstimate(
            name="two-electron SOC picture change", allocations=[
                res.PlannedAllocation("assembled AMF correction",
                                      correction_memory_gb(nao),
                                      note="h_sf and the three w_k over nao = {}".format(nao)),
            ],
            advice=["screening=\"none\" removes this phase, and the four-component atomic "
                    "solve behind it, at the cost of j-splittings 5-30% too large "
                    ""]))

    mo_advice = ["reduce the auxiliary/Cholesky dimension, which this block is linear in",
                 "use a smaller basis set"]
    if n_active:
        # Exactly what ``CASIntegrals`` owns, and what every CI driver is handed.
        mo_label = "three-index MO block B^P_pt"
        mo_ket = int(n_active)
        mo_gb = cas_integrals_memory_gb(naux, n, mo_ket)
        mo_note = "{} spinors x {} active".format(n, mo_ket)
        mo_advice.insert(0, "reduce the active space: this block carries one index over it")
    else:
        # No active space yet, so the plan bounds it by the largest block anything downstream
        # asks for: every spinor against the occupied ones. An active space wider than the
        # occupied space would exceed it, which is why the advice below says to pass
        # ``n_active`` rather than leaving the planner to guess.
        mo_label = "three-index MO block B^P_p,occ"
        mo_ket = n_occ
        mo_gb = mo_block_memory_gb(naux, n, mo_ket)
        mo_note = "{} spinors x {} occupied{}".format(
            n, mo_ket, "" if nelec is not None else " (no electron count given)")
        mo_advice.insert(0, "state the active space (n_active=): the plan then budgets the "
                            "block the multireference stage really holds, which is normally "
                            "far narrower than this bound")
    phases.append(res.PhaseEstimate(name="spinor MO transform", allocations=[
        res.PlannedAllocation(mo_label, mo_gb, note=mo_note),
        # What the kernel would use unblocked, capped by what it is allowed: blocking means
        # it never needs more than either. Planning for the cap alone is how a memory plan
        # becomes pessimistic enough to refuse calculations that would have run.
        res.PlannedAllocation("transform buffers",
                              min(transform_buffer_gb(nao, mo_ket, naux),
                                  res.BUDGET.transient_gb()), resident=False),
    ], advice=mo_advice))

    if n_active:
        phases.append(res.PhaseEstimate(name="active-space integrals", allocations=[
            res.PlannedAllocation("active four-index integrals",
                                  res.array_gb((n_active,) * 4),
                                  note="{} active spinors".format(n_active)),
            res.PlannedAllocation("2-RDM", res.rdm_gb(n_active, 2)),
            # ⚠ The second-order step's peak, and it is three times the block above rather
            # than a fraction of it. "auto" escalates to that step on the gradient
            # trajectory, so this is planned whenever an active space is, not only when
            # second order was asked for by name.
            res.PlannedAllocation("second-order Hessian response blocks",
                                  hessian_response_memory_gb(naux, n, int(n_active)),
                                  resident=False,
                                  note="3 x B^P_pt, live together in one Hessian-vector "
                                       "product"),
        ], advice=["reduce the active space",
                   "mode=\"quasi-newton\" never forms the Hessian response blocks, at the "
                   "cost of the robustness the second-order step exists for"]))
    if nevpt2 and n_active:
        n_i, n_a, n_v = _nevpt2_block_split(n, n_occ, int(n_active))
        phases.append(res.PhaseEstimate(name="SC-NEVPT2", allocations=[
            res.PlannedAllocation("4-RDM", res.rdm_gb(n_active, 4),
                                  note="n_active^8; 6.4 GB at 12 spinors, 382 GB at 20"),
            res.PlannedAllocation("3-RDM", res.rdm_gb(n_active, 3)),
            # The space-pair blocks the perturbation caches, all live at once and all live
            # beside the RDMs. ⚠ Summed by the identity
            # ``sum over the four pairs = naux (n_v + n_a) (n_i + n_a)`` rather than by asking
            # the perturbation for the list, because **nothing on the calculation path may
            # import the perturbation layer** and the front-end is on it. The pair list stays
            # the perturbation's own; the suite cross-checks this forecast against it, so a
            # class that starts requesting a fifth pair fails a test rather than quietly
            # leaving the plan describing a different calculation.
            res.PlannedAllocation("three-index MO blocks B^P_(bra|ket)",
                                  mo_block_memory_gb(naux, n_v + n_a, n_i + n_a),
                                  note="{} inactive / {} active / {} virtual".format(
                                      n_i, n_a, n_v)),
        ], advice=["reduce the active space: the 4-RDM grows as its eighth power",
                   "freeze core spinors or delete high virtuals, which shortens the label "
                   "ranges of the three-index blocks"]))
    return phases


def _set_pyscf_memory(mol) -> None:
    """Hand PySCF whatever is left of the memory limit (the SCF is *not* governed by the budget).

    The SCF's allocation pattern is dynamic and belongs to PySCF, so Kuiva cannot predict or
    police it. What it can do is stop PySCF from using its own 4 GB default on a machine
    where the user asked for something else — ``mol.max_memory`` is best effort on PySCF's
    side too, and this is the whole of the coverage the front-end gets. The pre-flight table
    says so explicitly rather than implying the phase is checked.
    """
    if not res.BUDGET.configured:
        return
    available_mb = res.BUDGET.available_gb() * 1024.0
    if available_mb < 1.0:
        return
    mol.max_memory = available_mb
    log.debug("PySCF max_memory set to %.0f MB (remaining Kuiva budget)", available_mb)


def _reserve_eri_memory(nao: int):
    """Refuse to materialize an ERI array that will not fit in the memory limit.

    Declared resident because :class:`ScalarX2CData` keeps the array: the Cholesky
    decomposition of :mod:`kuiva.integrals.transform` reads it, and both are live at once.
    """
    return res.reserve(
        "conventional AO ERI array (8-fold packed)", eri_memory_gb(nao),
        note="nao = {}, O(nao^4/8)".format(nao),
        advice=["supply an auxiliary basis (auxbasis=...) to use density fitting, which "
                "never forms this array — but beware: a Coulomb-fitting "
                "auxiliary is not accurate enough for correlated work",
                "use a smaller basis set; this array grows as the fourth power of nao",
                "an integral-direct Cholesky decomposition would remove the array entirely "
                "and is not implemented yet "])


@dataclass(frozen=True)
class MolecularFourComponent:
    """The molecular four-component **one-electron** problem, as plain arrays.

    Everything an X2C decoupling of a molecule needs, and nothing else: the core Hamiltonian
    and metric in the ``LL/LS/SL/SS`` blocking of :mod:`kuiva.x2c.decouple`, the matrix that
    contracts a result back to the molecule's own AO basis, and the per-atom AO ranges.

    ⚠ **No ``Mole`` crosses this boundary**. This container is the whole
    interface between PySCF's integrals and Kuiva's decoupling: :mod:`kuiva.x2c` never learns
    that a molecule exists, and the atom ranges are carried here because this is the only
    place that can compute them — an atom-local decoupling that had to re-derive them from a
    ``Mole`` would put basis-set machinery back downstream of the front-end.

    Attributes
    ----------
    hcore, overlap : FourComponentBlocks
        In the **working** basis, which is the decontracted one unless ``uncontract=False``.
    light_speed : float
        The ``c`` the blocks were built at, carried so that nothing downstream reads a global.
    contraction : ndarray (nao_work, nao_target) or None
        Scalar contraction back to the molecule's AO basis; ``None`` when the working basis
        already **is** it. Applied by :meth:`contract`.
    atom_ranges : tuple of (int, int)
        Half-open ``[p0, p1)`` **scalar** AO ranges, one per atom, in the working basis.
        ⚠ These are scalar ranges: an atom occupies *two* row ranges of a block, ``[p0, p1)``
        and ``[nao + p0, nao + p1)``, because the fixed conventions order rows spin-blocked ``[alpha; beta]``
        rather than interleaved. :meth:`spin_blocked_indices` is the only sanctioned way to
        turn one into the other.
    """

    hcore: "FourComponentBlocks"
    overlap: "FourComponentBlocks"
    light_speed: float
    contraction: Optional[np.ndarray]
    atom_ranges: Tuple[Tuple[int, int], ...]

    @property
    def nao(self) -> int:
        """Scalar basis size of the **working** basis."""
        return int(self.hcore.nao)

    @property
    def nao_target(self) -> int:
        """Scalar basis size of the molecule's own AO basis, which results contract back to."""
        return self.nao if self.contraction is None else int(self.contraction.shape[1])

    @property
    def decontracted(self) -> bool:
        return self.contraction is not None

    def spin_blocked_indices(self, atom: int) -> np.ndarray:
        """Row/column indices of ``atom`` in a ``(2*nao, 2*nao)`` block."""
        p0, p1 = self.atom_ranges[atom]
        n = self.nao
        return np.concatenate([np.arange(p0, p1), np.arange(n + p0, n + p1)])

    def contract(self, a: np.ndarray) -> np.ndarray:
        """Contract a ``(2*nao, 2*nao)`` two-component matrix back to the target AO basis.

        The scalar contraction is applied to both spin blocks, which is what makes it valid:
        contraction mixes primitives within one atom and one angular momentum and does nothing
        to spin, so it commutes with the spin-blocked row layout.
        """
        a = np.asarray(a)
        if self.contraction is None:
            return a
        from ..spinor.expand import spin_block_diagonal
        c = spin_block_diagonal(self.contraction)
        return np.ascontiguousarray(c.conj().T @ a @ c)


def _shell_ao_max(mol) -> int:
    """AO functions of the largest **libcint** shell — what sizes an integral-direct batch."""
    return int(np.diff(np.asarray(mol.ao_loc_nr())).max()) if mol.nbas else 1


def _npair_of(nao: int) -> int:
    """Packed AO pair count. Local, so the front end needs no import from the integral layer
    for a formula that is one line and is the same everywhere it appears."""
    return nao * (nao + 1) // 2


def direct_cache_memory_gb(nao: int, shell_ao_max: int, n_shells: int) -> float:
    """Size [GB] :class:`DirectERIMatrix`'s batch cache would like (exact sizing function).

    One batch per shell pair in the basis, i.e. every batch the decomposition could ever
    re-use. ⚠ **It is a want, not a need**, and the caller caps it with what the transient
    budget allows: the cache is what turns re-evaluations into re-use, and both ends of that
    trade are measured. Ti2Cl6 (351 shell pairs, 118 of them ever selected, 37 MB a batch)
    costs 731 integral batch evaluations with 8 cached, 380 with 16, 146 with 32 and 118 —
    the floor — with 64 or more; the CPU cost runs 243 s down to 102 s across that range.

    So on a machine with room the cache spends memory to go fast, and on the machine this
    route exists for — where the transient share is small because the limit is what binds —
    it shrinks and the decomposition pays up to ~6x more integral evaluations while still
    being the only route that runs at all. That self-adjustment is the point: this is a
    transient buffer, and the array it replaces was resident.

    ⚠ **Never more than the whole integral array would have cost.** Caching more integrals
    than storing all of them takes is absurd on a route whose reason to exist is not storing
    them — and it is not academic: without this cap a small molecule *planned worse* on the
    direct route than on the conventional one, i.e. the route could be refused exactly where
    the one it replaces would run.
    """
    pairs = int(n_shells) * (int(n_shells) + 1) // 2
    return min(pairs * direct_block_memory_gb(nao, shell_ao_max), eri_memory_gb(nao))


def direct_block_memory_gb(nao: int, shell_ao_max: int) -> float:
    """Size [GB] of one shell-pair batch of :class:`DirectERIMatrix` (exact sizing function).

    The evaluator's working set: ``(npair, d_K, d_L)`` doubles, where ``d`` is the number of
    AO functions of a shell. ⚠ It is **the integral library's granularity, not a blocking
    choice** — one call evaluates every ket component of a shell pair whether or not the
    caller wants them, so this array cannot be made smaller by asking for less. Sized with the
    largest shell in the basis on both sides, which is the worst pair that can occur.
    """
    return res.array_gb((_npair_of(nao), int(shell_ao_max), int(shell_ao_max)), np.float64)


class DirectERIMatrix:
    """The two-electron matrix ``(mu nu | la ka)``, evaluated on demand and never stored.

    The integral-direct half of the Cholesky route. It presents exactly the interface
    :meth:`kuiva.integrals.transform.ThreeIndexAO.from_matrix` asks for — a ``diagonal`` and a
    ``column(q)`` over the packed AO pair index — so the decomposition itself is unchanged and
    cannot tell the two routes apart. What changes is the memory: the conventional route must
    materialize ``O(nao^4/8)`` doubles first, and this one holds one shell-pair batch at a
    time.

    **Batching.** A column is one AO pair ``(la, ka)`` against every AO pair ``(mu, nu)``, and
    the integral library's smallest unit is a *shell* quartet — so one evaluation yields every
    ket component of the shell pair that ``(la, ka)`` belongs to, ``(npair, d_K, d_L)`` of
    them. Those are exactly the columns of one symmetry orbit, which is what the orbit-complete
    path asks for next, so the batch is cached and an orbit costs **one** evaluation. The cache
    is a bounded LRU: the plain path hops between shell pairs by descending diagonal and would
    otherwise re-evaluate the same batch several times (measured 2.3 evaluations per shell pair
    with a single-entry cache).

    ⚠ **Bit-level agreement is a property of this class, not a hope.** Two rules hold it up.
    (1) The diagonal is taken from the shell quartet ``(K,L|K,L)``, which is the same library
    call, with the same arguments, that the column batch for ``(K, L)`` makes for its own
    bra pair — so ``diagonal()[q]`` equals ``column(q)[q]`` bit for bit, which the plain path
    depends on. (2) Columns are packed by the library itself (``s2ij``), in the same
    lower-triangular order the rest of this module uses, so no repacking step exists to get
    wrong. Fed the same matrix elements, this route and the in-memory one produce **bitwise
    identical** factors.

    ⚠ **What it does not agree with bitwise is the 8-fold packed array** the conventional
    route ingests. That array is filled by a different traversal of the same integrals, and the
    library's two fills differ in the last bits for a sixth of the elements — a property of the
    integral library, not of the decomposition. The consequence is measured rather than
    assumed, and it is nothing a result is read from: the reconstructed integrals agree to
    3e-15 and a spin-orbit spectrum to 0.0 cm^-1.

    References
    ----------
    Evaluating only the columns the pivoting selects, rather than the whole matrix, is the
    formulation of H. Koch, A. Sanchez de Meras, T. B. Pedersen, "Reduced scaling in electronic
    structure calculations using Cholesky decompositions", J. Chem. Phys. 118, 9481 (2003),
    doi:10.1063/1.1578621; see also F. Aquilante et al., "Cholesky Decomposition Techniques in
    Electronic Structure Theory", in "Linear-Scaling Techniques in Computational Chemistry and
    Physics", Springer (2011), pp. 301-343, doi:10.1007/978-90-481-2853-2_13, for the
    integral-direct practice this follows.
    """

    def __init__(self, mol, *, cache_gb: Optional[float] = None) -> None:
        self.mol = mol
        self.nao = int(mol.nao)
        self.nbas = int(mol.nbas)
        self.npair = _npair_of(self.nao)
        self._ao_loc = np.asarray(mol.ao_loc_nr(), dtype=np.int64)
        # ⚠ libcint shells, NOT AOLayout shells: a general contraction is one shell to the
        # integral library and several to the layout, and shls_slice speaks the library's
        # language. Deriving this from the layout would slice the wrong ranges on exactly the
        # bases the actinide work uses.
        self._shell_of = np.repeat(np.arange(self.nbas, dtype=np.int64),
                                   np.diff(self._ao_loc))
        self._rows, self._cols = np.tril_indices(self.nao)
        # One budget question, asked once and outside every loop — and the cache is bounded by
        # a **batch count** as well, so that what it holds is the number the memory plan
        # carries rather than however much of the limit happens to be free. Sizing it from the
        # budget alone was measured holding gigabytes of integral batches for no gain.
        limit = (min(direct_cache_memory_gb(self.nao, self.shell_ao_max, self.nbas),
                     res.transient_gb())
                 if cache_gb is None else float(cache_gb))
        self._cache_limit_bytes = limit * res.BYTES_PER_GB
        self._cache: "OrderedDict[Tuple[int, int], np.ndarray]" = OrderedDict()
        self._cache_bytes = 0
        #: Shell-pair batches evaluated, and columns served from them: the cost of the route.
        self.n_batches = 0
        self.n_columns = 0

    @property
    def shell_ao_max(self) -> int:
        """AO functions of the largest shell — what sizes one batch."""
        return int(np.diff(self._ao_loc).max()) if self.nbas else 1

    def batch_memory_gb(self) -> float:
        """Size [GB] of the largest shell-pair batch this basis can produce."""
        return direct_block_memory_gb(self.nao, self.shell_ao_max)

    def diagonal(self) -> np.ndarray:
        """``(mu nu | mu nu)`` for every packed AO pair, from one quartet per shell pair."""
        diag = np.zeros(self.npair, dtype=np.float64)
        loc = self._ao_loc
        with timer("direct ERI diagonal"):
            for k in range(self.nbas):
                for l in range(k + 1):
                    quartet = self.mol.intor(
                        "int2e", shls_slice=(k, k + 1, l, l + 1, k, k + 1, l, l + 1))
                    block = np.einsum("abab->ab", quartet)      # (d_K, d_L), tiny
                    a = np.arange(loc[k], loc[k + 1])[:, None]
                    b = np.arange(loc[l], loc[l + 1])[None, :]
                    keep = a >= b
                    diag[(a * (a + 1) // 2 + b)[keep]] = block[keep]
        return diag

    def column(self, q: int) -> np.ndarray:
        """Column ``q`` of the matrix: ``(mu nu | la ka)`` with ``(la, ka)`` the pair ``q``."""
        a = int(self._rows[q])
        b = int(self._cols[q])
        k = int(self._shell_of[a])
        l = int(self._shell_of[b])
        self.n_columns += 1
        return self._batch(k, l)[:, a - self._ao_loc[k], b - self._ao_loc[l]]

    def _batch(self, k: int, l: int) -> np.ndarray:
        """The ``(npair, d_K, d_L)`` batch for shell pair ``(k, l)``, evaluated or cached."""
        key = (k, l)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        block = self.mol.intor(
            "int2e", shls_slice=(0, self.nbas, 0, self.nbas, k, k + 1, l, l + 1),
            aosym="s2ij")
        self.n_batches += 1
        nbytes = block.nbytes
        # Evict oldest first; a batch larger than the whole budget is used and not kept
        # rather than refused, because the library cannot produce a smaller one.
        while self._cache and self._cache_bytes + nbytes > self._cache_limit_bytes:
            _, dropped = self._cache.popitem(last=False)
            self._cache_bytes -= dropped.nbytes
        if nbytes <= self._cache_limit_bytes:
            self._cache[key] = block
            self._cache_bytes += nbytes
        return block

    def report(self, logger=None) -> None:
        logger = logger or log
        out.entry(logger or log, "integral batches evaluated", self.n_batches,
                  note="{} shell pairs in the basis; a pair is re-evaluated when the "
                       "cache cannot hold it".format(self.nbas * (self.nbas + 1) // 2))
        out.entry(logger or log, "columns served", self.n_columns)
        out.entry(logger or log, "largest batch", self.batch_memory_gb(), "GB", fmt="{:.3f}")

    def __repr__(self) -> str:
        return "DirectERIMatrix(nao={}, batches={}, columns={})".format(
            self.nao, self.n_batches, self.n_columns)


def _direct_cholesky(mol, *, tol: float, orbit_pivots: bool = True,
                     one_centre: bool = True, report: bool = True):
    """Cholesky-decompose the two-electron integrals without ever storing them.

    The front-end half of the integral-direct route: it pairs :class:`DirectERIMatrix` with
    the decomposition every other route runs, and returns the same
    :class:`~kuiva.integrals.transform.ThreeIndexAO` those produce. Pivoting on complete
    symmetry orbits is the default here exactly as it is elsewhere, and the labelling comes
    from the same AO layout, so the two routes select the same orbits in the same order.
    """
    from ..integrals.transform import ThreeIndexAO, shell_pair_orbits

    orbits = None
    if orbit_pivots:
        layout = ao_layout(mol)
        orbits = shell_pair_orbits(layout.ao_shell, layout.ao_atom, one_centre=one_centre)
    source = DirectERIMatrix(mol)
    with timer("integral-direct two-electron integrals"):
        factors = ThreeIndexAO.from_matrix(
            source.diagonal(), source.column, int(mol.nao), tol, orbits=orbits,
            report=report, note="evaluated on demand (integral-direct)")
    if report:
        source.report()
    return factors


def ao_layout(mol) -> AOLayout:
    """Extract the AO basis layout: which atom and shell every basis function belongs to.

    This is the metadata the *analysis* stages need back after the front-end quarantine has made
    contraction invisible to everything else — Loewdin population analysis
    (:mod:`kuiva.props.population`) and the molden dump (:mod:`kuiva.props.molden`). It is
    plain arrays and strings: no ``Mole`` crosses the boundary, so carrying it downstream does
    not put PySCF back on the multireference path.

    A ``libcint`` shell with ``nctr > 1`` (a general contraction) becomes ``nctr`` separate
    :class:`~kuiva.basis.layout.Shell` entries. That is the granularity molden's ``[GTO]``
    section works at, and it preserves the AO order exactly, because the library lays a
    general contraction out contraction-major and m-minor.

    Raises on a Cartesian basis: those are refused wherever two-component dimensions are
    involved, and every ``2l+1`` here assumes spherical functions.
    """
    if getattr(mol, "cart", False):
        raise NotImplementedError(
            "Cartesian bases are not supported; the AO layout, the molden "
            "dump and the population analysis all assume 2l+1 spherical functions per shell")

    shells: List[Shell] = []
    for ib in range(mol.nbas):
        l = int(mol.bas_angular(ib))
        exps = np.asarray(mol.bas_exp(ib), dtype=float)
        ctr = np.asarray(mol.bas_ctr_coeff(ib), dtype=float).reshape(exps.size, -1)
        atom = int(mol.bas_atom(ib))
        for ic in range(ctr.shape[1]):
            shells.append(Shell(atom=atom, l=l, exponents=exps.copy(),
                                coefficients=np.ascontiguousarray(ctr[:, ic])))

    # "0 Ce 5g+2  " -> "5g+2"; the atom is carried by ao_atom, which is unambiguous.
    labels = [lbl.split()[2] for lbl in mol.ao_labels()]
    layout = build_layout(
        atom_symbols=[mol.atom_pure_symbol(ia) for ia in range(mol.natm)],
        atom_charges=[float(mol.atom_charge(ia)) for ia in range(mol.natm)],
        coords_bohr=np.asarray(mol.atom_coords()),
        shells=shells, ao_labels=labels)
    if layout.nao != mol.nao:
        raise RuntimeError("AO layout has {} functions against the basis' {}; the shell "
                           "expansion is wrong".format(layout.nao, mol.nao))
    return layout


def four_component_one_electron(mol, *, uncontract: bool = True,
                                light_speed: Optional[float] = None
                                ) -> MolecularFourComponent:
    """Build the molecular four-component one-electron blocks.

    In the restricted kinetically balanced normalization :mod:`kuiva.x2c.decouple` fixes::

        hcore.ll = V                       overlap.ll = S
        hcore.ls = hcore.sl = T            overlap.ls = overlap.sl = 0
        hcore.ss = W / (4 c^2) - T         overlap.ss = T / (2 c^2)

    with ``W = <sigma.p V sigma.p>`` the spin-dependent term the spin-orbit coupling comes
    from. Feeding the result to :func:`kuiva.x2c.decouple.decoupling_matrices` and
    :func:`kuiva.x2c.decouple.picture_change` reproduces the exact molecular X2C-1e
    Hamiltonian; feeding it an atom-blocked decoupling gives the DLU approximation.

    ⚠ **The spin structure is assembled with Kuiva's own spin-structure helpers, not PySCF's.**
    ``two_component_operator(w[3], w[0:3])`` reproduces ``pyscf.x2c.x2c._sigma_dot`` and
    ``spin_block_diagonal`` reproduces ``_block_diag`` — both **bitwise**, asserted in
    ``tests/test_x2c_decouple.py``. That test is the point: the conventions are
    defined in :mod:`kuiva.spinor.expand` and nowhere else, so this function must not reach
    for a second definition of them, and the equivalence to PySCF's is a *measured result*
    rather than an assumption. ``int1e_spnucsp`` returns ``(4, nao, nao)``: three components
    whose Kuiva-convention ``w_k`` they are exactly, and the spin-free part last.

    Parameters
    ----------
    uncontract : bool
        Decouple in the fully decontracted basis and contract the result back (
        the decoupling belongs in the primitive basis). The default, and what PySCF's own X2C
        helper does — so it is also what makes the two comparable.
    light_speed : float, optional
        Override ``c``. Only for the ``c -> inf`` limit tests; a real calculation never sets it.

    ⚠ **The default ``c`` is read from PySCF, not from**
    :data:`kuiva.x2c.decouple.LIGHT_SPEED`. The two are *not* the same number — PySCF ships
    ``137.03599967994`` against Kuiva's CODATA-2018 ``137.035999084``, a relative difference
    of 4.3e-09 — and the integrals in these blocks are PySCF's. Building them at Kuiva's
    constant instead measurably degrades agreement with PySCF's own X2C Hamiltonian, from
    **6e-13 to 1e-11 relative**, for no reason other than the mismatch. Whatever produced the
    integrals has to define the ``c`` they are combined with.
    """
    from pyscf import lib
    from pyscf.x2c import x2c

    from ..spinor.expand import spin_block_diagonal, two_component_operator

    if mol.has_ecp():
        raise NotImplementedError(
            "the four-component one-electron problem requires an all-electron basis; this "
            "molecule uses an ECP, and X2C has no meaning with a pseudopotential core. The "
            "supported basis families are all-electron by design.")
    if getattr(mol, "cart", False):
        # Same structural refusal as the atomic mean field makes (kuiva.amf.correction): the
        # small-component/spinor machinery is built on solid harmonics, so a Cartesian basis
        # would express the result over different functions from the ones it is added to. For
        # l <= 1 the two coincide, which is exactly why refusing must not depend on the basis.
        raise NotImplementedError(
            "the four-component one-electron problem requires a spherical (solid-harmonic) AO "
            "basis; this molecule was built with cart=True.")

    from ..x2c.decouple import FourComponentBlocks

    helper = x2c.SpinOrbitalX2CHelper(mol)
    helper.xuncontract = bool(uncontract)
    xmol, contraction = helper.get_xmol(mol)
    c = float(lib.param.LIGHT_SPEED if light_speed is None else light_speed)

    with timer("molecular four-component one-electron integrals"):
        t = spin_block_diagonal(xmol.intor_symmetric("int1e_kin"))
        v = spin_block_diagonal(xmol.intor_symmetric("int1e_nuc"))
        s = spin_block_diagonal(xmol.intor_symmetric("int1e_ovlp"))
        raw = xmol.intor("int1e_spnucsp")
        w = two_component_operator(raw[3], raw[0:3])

    zero = np.zeros_like(s)
    hcore = FourComponentBlocks(ll=v, ls=t, sl=t.copy(), ss=w * (0.25 / c**2) - t)
    overlap = FourComponentBlocks(ll=s, ls=zero, sl=zero.copy(), ss=t * (0.5 / c**2))

    slices = xmol.aoslice_by_atom()
    ranges = tuple((int(slices[ia][2]), int(slices[ia][3])) for ia in range(xmol.natm))
    return MolecularFourComponent(
        hcore=hcore, overlap=overlap, light_speed=c,
        contraction=None if contraction is None else np.asarray(contraction),
        atom_ranges=ranges)


def molecular_partition(mol, fc: MolecularFourComponent) -> "Partition":
    """The atom partition of ``fc``'s working basis, for the DLU decoupling.

    Labelled by ``mol.atom_symbol(ia)`` — the *labelled* symbol, so ``Ti1`` and ``Ti2`` stay
    separate — which is the same key :func:`kuiva.amf.atomic.elements_by_label` uses, so an
    isolated-fragment mapping and an atomic mean-field correction are keyed alike.
    """
    from ..x2c.local import Partition

    if len(fc.atom_ranges) != mol.natm:
        raise ValueError("these blocks describe {} atoms, this molecule has {}".format(
            len(fc.atom_ranges), mol.natm))
    return Partition(
        indices=tuple(fc.spin_blocked_indices(ia) for ia in range(mol.natm)),
        labels=tuple(mol.atom_symbol(ia) for ia in range(mol.natm)))


def isolated_fragment_blocks(mol, *, uncontract: bool = True,
                             light_speed: Optional[float] = None) -> Dict[str, object]:
    """``{atom label: (hcore_A, overlap_A)}`` for the ``source="isolated"`` DLU decoupling.

    One four-component one-electron problem per **unique atom label**, each on an isolated
    atom carrying only its own nucleus, in exactly the basis the molecule gives that atom.
    Solved once per label rather than once per atom, which is what makes the decoupling
    transferable across a potential-energy surface.

    ⚠ **The blocks are only usable if their AO ordering matches the molecular diagonal block,
    and that is not checked here** — it is checked by
    :func:`kuiva.x2c.local.check_local_blocks`, against the *overlap*, which depends on the
    basis alone. A permuted block would be Hermitian, of the right magnitude, and wrong.
    """
    from pyscf import gto

    from ..amf.atomic import elements_by_label

    blocks = {}
    for label, (symbol, basis) in elements_by_label(mol).items():
        atom = gto.M(atom=[(symbol, (0.0, 0.0, 0.0))], basis={symbol: basis},
                     spin=None, charge=0, verbose=0, unit="Bohr")
        fc = four_component_one_electron(atom, uncontract=uncontract,
                                         light_speed=light_speed)
        blocks[label] = (fc.hcore, fc.overlap)
    return blocks


def local_x2c_hamiltonian(mol, *, partition: str = "atoms", source: str = "diagonal",
                          uncontract: bool = True, light_speed: Optional[float] = None
                          ) -> Tuple[np.ndarray, DecouplingRecord]:
    """The two-component X2C Hamiltonian from a **local (DLU)** decoupling.

    Returns ``(h2c, record)`` with ``h2c`` in the molecule's own AO basis, in the spin-blocked
    spin-blocked layout — the same object PySCF's ``SpinOrbitalX2CHelper.get_hcore()`` returns,
    so :func:`ingest_spin_orbit` decomposes it identically whichever route produced it.

    ``partition="single"`` puts every basis function in one fragment, which makes this the
    **exact** transformation expressed through the local code path. That is not a curiosity:
    it is the only like-for-like reference for the DLU error, because Kuiva's exact decoupling
    and PySCF's differ by up to 2.4e-07 relative on a heavy element — a confound sitting
    in the same range as the effect being measured.
    """
    from ..x2c.local import (Partition, local_block_scales, local_decoupling_matrices)
    from ..x2c.decouple import picture_change
    from ..x2c.methods import PARTITIONS

    if partition not in PARTITIONS:
        raise ValueError("unknown partition {!r}; expected one of {}".format(
            partition, ", ".join(PARTITIONS)))

    fc = four_component_one_electron(mol, uncontract=uncontract, light_speed=light_speed)
    if partition == "single":
        parts = Partition.single(fc.hcore.n2c)
    else:
        parts = molecular_partition(mol, fc)

    local = None
    if source == "isolated":
        local = isolated_fragment_blocks(mol, uncontract=uncontract, light_speed=light_speed)
        if partition == "single":
            raise ValueError(
                "source='isolated' has no meaning with partition='single': the single "
                "fragment is the whole molecule, and an 'isolated molecule' is the molecule. "
                "Use source='diagonal', which for one fragment is the exact decoupling.")

    with timer("local (DLU) X2C decoupling"):
        x, r = local_decoupling_matrices(fc.hcore, fc.overlap, fc.light_speed, parts,
                                         source=source, local=local, report=True)
        h2c = fc.contract(picture_change(fc.hcore, x, r))

    record = DecouplingRecord(
        decoupling="1e-dlu", implementation="kuiva", partition=partition, source=source,
        fragments=len(parts), block_scales=local_block_scales(x, parts))
    return np.ascontiguousarray(h2c), record


def molecular_mean_field(mol, *, interaction: str = "coulomb", uncontract: bool = True,
                         light_speed: Optional[float] = None, conv_tol: float = 1e-10,
                         max_cycle: int = 100) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """X2C-mmf: the two-electron picture change from a **molecular** four-component SCF.

    Returns ``(delta h_sf, delta w, info)`` in the molecule's own AO basis, in the spin-blocked
    conventions — the same shape as :func:`kuiva.amf.correction.amf_correction` produces, and
    added to the one-electron X2C Hamiltonian in exactly the same place.

    ⚠ **Experimental benchmark method. Never a default, never for production.** It requires a
    full four-component Dirac-Hartree-Fock on the whole molecule including ``(SS|SS)``
    integrals, which is precisely the cost X2CAMF exists to avoid: HF in the project default
    basis takes ~5 CPU s, HCl ~43 CPU s, and it grows as the fourth power of the basis. Use it
    to check X2CAMF on a small system, not to run one.

    ⚠ **This is the *same subtraction* X2CAMF performs**, on a molecular four-component solve
    instead of an atomic one — :func:`kuiva.x2c.mean_field.mean_field_picture_change` is shared
    between them, so the committed atomic reference numbers validate this path too. What mmf
    removes is the *atomic* approximation: no per-element solve, no atom-diagonal assembly, and
    the off-atom blocks of the two-electron picture change are present rather than zero.

    ⚠ **Never combine with** ``screening="x2camf"``: both supply the two-electron picture
    change, and applying them together double-counts it. :func:`ingest_spin_orbit` refuses.
    """
    from pyscf import gto, scf
    from pyscf.x2c import x2c

    from ..amf.backend import INTERACTIONS
    from ..amf.pyscf_dhf import eigh_canonical, spinor_to_spin_orbital, light_speed as _c
    from ..spinor.expand import decompose_two_component, spin_block_diagonal
    from ..x2c.decouple import METRIC_LINDEP_THRESHOLD, FourComponentBlocks
    from ..x2c.mean_field import mean_field_picture_change

    if interaction not in INTERACTIONS:
        raise ValueError("unknown two-electron interaction {!r}; expected one of {}".format(
            interaction, INTERACTIONS))
    if mol.has_ecp() or getattr(mol, "cart", False):
        raise NotImplementedError(
            "the molecular mean field requires an all-electron, spherical (solid-harmonic) "
            "basis, for the same structural reasons the atomic one does.")

    helper = x2c.SpinOrbitalX2CHelper(mol)
    helper.xuncontract = bool(uncontract)
    xmol, contraction = helper.get_xmol(mol)

    log.warning("X2C-mmf is an EXPERIMENTAL BENCHMARK method and is not intended for "
                "production calculations. It runs a full four-component SCF on the whole "
                "molecule (%d four-component functions here), which is the cost X2CAMF exists "
                "to avoid. Use screening='x2camf' for real work.", 4 * xmol.nao)

    with _c(light_speed):
        from ..amf.pyscf_dhf import current_light_speed
        c = current_light_speed()
        with timer("molecular 4c DHF"):
            mf = scf.DHF(xmol)
            mf.verbose = 0
            mf.conv_tol = conv_tol
            mf.max_cycle = max_cycle
            mf.with_ssss = True
            mf.with_gaunt = interaction in ("gaunt", "breit")
            mf.with_breit = interaction == "breit"
            # ⚠ The same metric projection the atomic solve uses, for the same reason: a
            # decontracted heavy-element basis is numerically singular, and the decoupling
            # must not represent directions the SCF projected away.
            mf._eigh = eigh_canonical(METRIC_LINDEP_THRESHOLD)
            e_tot = float(mf.kernel())
        if not mf.converged:
            log.error("the molecular four-component SCF did not converge in %d cycles "
                      "(E = %.8f Eh). Everything the mmf correction says rests on these "
                      "orbitals.", max_cycle, e_tot)

        u = spinor_to_spin_orbital(xmol)

        def to_spin_orbital(a) -> FourComponentBlocks:
            b = FourComponentBlocks.from_matrix(np.asarray(a))
            return FourComponentBlocks(
                ll=np.ascontiguousarray(u @ b.ll @ u.conj().T),
                ls=np.ascontiguousarray(u @ b.ls @ u.conj().T),
                sl=np.ascontiguousarray(u @ b.sl @ u.conj().T),
                ss=np.ascontiguousarray(u @ b.ss @ u.conj().T))

        dm = mf.make_rdm1()
        hcore = to_spin_orbital(mf.get_hcore(xmol))
        overlap = to_spin_orbital(mf.get_ovlp(xmol))
        density = to_spin_orbital(dm)
        veff = to_spin_orbital(mf.get_veff(xmol, dm))

        def coulomb_mean_field(d_2c):
            # ⚠ Over **exactly** the basis the four-component solve used; a separately built
            # basis would make the subtraction a difference between two different things.
            vj, vk = scf.GHF(xmol).get_jk(xmol, np.ascontiguousarray(d_2c, dtype=np.complex128),
                                          hermi=1)
            return np.asarray(vj) - np.asarray(vk)

        change = mean_field_picture_change(hcore, overlap, veff, density.ll, c,
                                           coulomb_mean_field,
                                           label=xmol.atom_symbol(0) + "... (molecular)")

    dg = change.dg
    if contraction is not None:
        cc = spin_block_diagonal(np.asarray(contraction))
        dg = cc.conj().T @ dg @ cc
    h_sf, w = decompose_two_component(dg)
    info = {"method": "x2c-mmf", "interaction": interaction, "e_4c_tot": e_tot,
            "converged": bool(mf.converged), "n_four_component": int(4 * xmol.nao),
            "uncontracted": bool(uncontract), "light_speed": c,
            "scale": change.scale, "cancellation": change.cancellation,
            "tr_residual": change.tr_residual, "tr_residual_rel": change.tr_residual_rel,
            "compensation_scale": change.compensation_scale}
    return np.ascontiguousarray(h_sf), np.ascontiguousarray(w), info


def ingest_spin_orbit(mol, h_sfx2c1e: Optional[np.ndarray] = None,
                      approx: str = "1e", *, screening: str = "x2camf",
                      decoupling_options: Optional[Dict[str, object]] = None,
                      **screening_kwargs) -> SpinOrbitX2C:
    """Extract the two-component X2C Hamiltonian and decompose it (see :class:`SpinOrbitX2C`).

    PySCF returns it in the spin-blocked ``[alpha; beta]`` AO basis, which is already Kuiva's
    row convention — verified here structurally rather than assumed: the decomposition
    below only reproduces the matrix if the layout and the ``W_k = i w_k`` convention both
    match, and the residual is measured and reported.

    Parameters
    ----------
    screening : str
        Two-electron picture change to add: ``"x2camf"`` (**the default** — the atomic mean
        field of :mod:`kuiva.amf`), ``"none"`` (one-electron X2C only, which
        leaves j-splittings 5-30% too large), or ``"x2camf-external"`` (the reference
        implementation, a bisection tool and never a default).

        ⚠ **The default costs one four-component atomic solve per unique element**, cached in
        the process and on disk (:mod:`kuiva.amf.cache`) and independent of geometry. That is
        under a second for a light element and ~35 minutes for a lanthanide, paid once ever
        per (element, basis, configuration). ``screening="none"`` is the escape hatch, and it
        is a statement about cost, not about correctness.
    **screening_kwargs
        Passed to :func:`kuiva.amf.correction.amf_correction`: ``interaction``, ``backend``,
        ``configuration`` (a mapping for a heteronuclear molecule), ``uncontract``,
        ``light_speed``.

    Notes
    -----
    ⚠ **The correction is added here, in the AO basis, before any change of basis.** That is
    what keeps :meth:`SpinOrbitX2C.transform` working untouched: ``Delta h_sf`` and
    ``Delta w`` transform under a change of scalar basis exactly as their one-electron
    counterparts do, so once they are summed in there is nothing left that has to be
    transformed separately, and no way for a caller to transform one and forget the other.
    The result records what it contains in :attr:`SpinOrbitX2C.screening`.
    """
    from pyscf.x2c import x2c

    from ..x2c.methods import DECOUPLINGS, resolve

    if mol.has_ecp():
        raise NotImplementedError(
            "X2C spin-orbit ingestion requires an all-electron basis; this molecule uses an "
            "ECP. The supported basis families are all-electron by design.")
    if approx not in DECOUPLINGS:
        raise ValueError("unknown one-electron decoupling {!r}; expected one of {}".format(
            approx, ", ".join(DECOUPLINGS)))
    resolved = resolve(decoupling=approx, screening=screening)

    if approx == "1e-dlu":
        # The local path is Kuiva's own; see local_x2c_hamiltonian for why the exact route
        # deliberately stays on PySCF's implementation rather than switching to this one.
        h2c, decoupling_record = local_x2c_hamiltonian(mol, **(decoupling_options or {}))
    else:
        if decoupling_options:
            raise ValueError(
                "decoupling_options={} were given, but they only apply to the local (DLU) "
                "decoupling and this calculation uses approx={!r}. Silently ignoring them "
                "would mean running a different Hamiltonian from the one requested."
                .format(sorted(decoupling_options), approx))
        with timer("X2C spin-orbit integrals"):
            helper = x2c.SpinOrbitalX2CHelper(mol)
            helper.approx = approx
            h2c = np.asarray(helper.get_hcore())
        decoupling_record = DecouplingRecord(decoupling=approx, implementation="pyscf")

    # Projection onto the time-reversal-even part, in the fixed spinor conventions. The
    # decomposition itself is defined once, in kuiva.spinor.expand, so that the atomic
    # mean-field correction decomposes its two-electron picture change in exactly the
    # same convention as the one-electron operator it corrects.
    from ..spinor.expand import decompose_two_component, time_reversal_residual
    h_sf, w = decompose_two_component(h2c)
    residual, rel = time_reversal_residual(h2c)
    # The X2C decoupling uses a matrix square root; its rounding error is time-reversal odd
    # and would show up downstream as a spurious Kramers splitting if kept.
    if rel > 1e-8:
        log.warning("the two-component X2C Hamiltonian deviates from time-reversal symmetry "
                    "by %.2e Eh (%.1e relative). The odd part has been projected out so "
                    "Kramers degeneracy is exact; a large value here means the X2C decoupling "
                    "is poorly conditioned in this basis.", residual, rel)

    # The two-electron picture change, added in the AO basis — see the
    # docstring for why here and nowhere later. ``method="none"`` returns exact zeros, so the
    # uncorrected Hamiltonian stays bitwise what it was before any of this existed.
    from ..amf.correction import AMFCorrection, ScreeningRecord, amf_correction
    if screening == "mmf":
        # X2C-mmf: the same subtraction on a *molecular* four-component solve. It is a
        # value on this axis rather than a separate knob precisely so that it cannot be
        # combined with X2CAMF — both supply the two-electron picture change.
        d_h_sf, d_w, mmf_info = molecular_mean_field(mol, **screening_kwargs)
        # ⚠ ``backend_version`` is a ScreeningRecord contract field and is not a place to park
        # the four-component dimension, however convenient — that goes to the log. Repurposing
        # a stored-data field is how a record stops meaning what its name says.
        from pyscf import __version__ as _pyscf_version
        correction = AMFCorrection(
            h_sf=d_h_sf, w=d_w, method="mmf", interaction=mmf_info["interaction"],
            backend="pyscf-dhf", backend_version=str(_pyscf_version),
            elements=tuple(sorted({mol.atom_symbol(ia) for ia in range(mol.natm)})),
            light_speed=None,
            spin_free_scale=float(np.max(np.abs(d_h_sf))) if d_h_sf.size else 0.0,
            spin_orbit_scale=float(np.max(np.abs(d_w))) if d_w.size else 0.0,
            tr_residual=float(mmf_info["tr_residual"]),
            tr_residual_rel=float(mmf_info["tr_residual_rel"]))
    else:
        correction = amf_correction(mol, method=screening, **screening_kwargs)
    if not correction.is_zero:
        log.debug("two-electron picture change added: max |dh_sf| = %.3e Eh, max |dw| = "
                  "%.3e Eh, over %s", correction.spin_free_scale,
                  correction.spin_orbit_scale, ", ".join(correction.elements))
        h_sf = h_sf + correction.h_sf
        w = w + correction.w

    # Measured on the operator that is actually returned, so that it describes the difference
    # between the Hamiltonian the SCF ran with and the one the multireference step will use —
    # correction included. With screening off this is bitwise the number it always was.
    shift = 0.0
    if h_sfx2c1e is not None:
        shift = float(np.max(np.abs(h_sf - np.asarray(h_sfx2c1e))))

    return SpinOrbitX2C(h_sf=np.ascontiguousarray(h_sf), w=np.ascontiguousarray(w),
                        approx=approx, screening=correction.provenance(),
                        decoupling=decoupling_record, method=resolved.name,
                        tr_residual=residual, tr_residual_rel=rel,
                        picture_change_shift=shift)


#: Smallest Mulliken weight of an occupied orbital on the ``l`` channel it was assigned to
#: for :func:`run_scalar_aoc` to consider the assignment unambiguous. An atomic orbital has a
#: definite ``l``, so a clean solution gives 1 to within the SCF's own noise; a value below
#: this means two channels are genuinely mixed in one orbital, which no filling rule keyed on
#: ``l`` can resolve. It is advisory (a ``WARNING``, never a refusal): the run may still be the
#: one the user wanted, and stopping it would be a guess of the opposite kind.
AOC_ASSIGNMENT_TOLERANCE = 0.9


def _aoc_occupation(mol, configuration, ao_l: np.ndarray, state: Dict[str, object]):
    """A scalar ``get_occ`` implementing **average-of-configuration** occupation.

    The spin-restricted counterpart of
    :func:`kuiva.amf.pyscf_dhf._average_of_configuration_occupation`, and deliberately the same
    construction: assign every MO to an ``l`` channel by the **Mulliken weight** of its AO
    coefficients, then fill each channel in energy order through the shared filling rule
    :func:`kuiva.amf.configuration.average_occupations` — here with ``spatial=True``, so a
    shell is ``2l+1`` orbitals, a full one carries 2 electrons and a frontier one carries
    ``q / (2l+1)`` each.

    ⚠ **Assignment by ``l`` rather than by index is what makes a cold start reproducible**, and
    it is the reason this is not simply an aufbau fill: 4f, 5d and 6s lie within an eV of each
    other in a lanthanide and change order during the SCF, while within one ``l`` the ordering
    is never in doubt. It is also what lets the *configuration* rather than the current
    spectrum decide which shells are occupied — a run that would have collapsed 4f into 5d
    keeps its f electrons in f orbitals and reports a poor assignment weight instead.

    ``state`` is filled on every cycle with the current orbitals, occupations and open-shell
    partition, because :func:`kuiva.amf.configuration.install_configuration_average` needs all
    three and ``get_occ`` is the only hook PySCF's SCF loop calls with the coefficients (it
    keeps them in a local until convergence). It also records the assignment purity, which is
    *reported after* the SCF rather than logged here: a diagnostic inside the loop would print
    once per cycle.
    """
    from ..amf.configuration import SHELL_LETTERS, OpenShell, average_occupations

    ao_l = np.asarray(ao_l, dtype=int)
    channels = sorted(set(int(l) for l in ao_l))
    overlap = np.asarray(mol.intor("int1e_ovlp"))
    missing = [l for l, n in enumerate(configuration.occupations) if n and l not in channels]
    if missing:
        raise ValueError(
            "the configuration {} needs {} functions, which this basis does not have".format(
                configuration.canonical, "/".join(SHELL_LETTERS[l] for l in missing)))

    def get_occ(mo_energy=None, mo_coeff=None):
        if mo_coeff is None:
            raise RuntimeError(
                "average-of-configuration occupation needs the orbital coefficients to "
                "resolve the angular momentum of each orbital; get_occ was called with "
                "energies only")
        e = np.asarray(mo_energy, dtype=float)
        c = np.asarray(mo_coeff)
        population = np.real(np.conj(c) * (overlap @ c))       # Mulliken weight per AO
        weights = np.stack([population[ao_l == l].sum(axis=0) for l in channels])
        assigned = np.asarray(channels)[np.argmax(weights, axis=0)]
        occ = average_occupations(configuration, e, assigned, spatial=True)

        state["mo_coeff"], state["mo_occ"] = c, occ
        occupied = occ > 1e-12
        state["assignment_weight"] = (float(np.min(weights.max(axis=0)[occupied]))
                                      if occupied.any() else 1.0)
        # ⚠ Resolved here and nowhere else: which orbitals form each open shell. Grouping by
        # occupation *value* instead would merge two open shells that happen to share a
        # fraction — possible, and silent when it happens.
        shells = []
        for l, n_electrons in enumerate(configuration.occupations):
            degeneracy = 4 * l + 2
            q = n_electrons % degeneracy
            if not q:
                continue
            index = np.where((assigned == l) & occupied & (occ < 2.0 - 1e-12))[0]
            if index.size != 2 * l + 1:
                raise RuntimeError(
                    "the {} open shell came back on {} orbitals where {} were filled "
                    "fractionally".format(SHELL_LETTERS[l], index.size, 2 * l + 1))
            # ``n`` is the number of SPINORS the shell holds, in every representation: the
            # coupling coefficient is a pair average over spin orbitals and does not know
            # that these orbitals are spatial ones.
            shells.append(OpenShell(l, q, degeneracy, index))
        state["shells"] = shells
        return occ

    return get_occ


@dataclass(frozen=True)
class _SingleAtom:
    """The minimal ``Molecule``-shaped object :func:`build_mole` duck-types on.

    An atomic calculation has no geometry to speak of, so building one of these locally is
    cheaper than asking the caller for a container — and it keeps the basis registry, the
    consistency checks and the provenance metadata on exactly the path a molecule takes.
    """
    atoms: List[Tuple[str, Tuple[float, float, float]]]
    charge: int
    spin: int
    basis: object
    unit: str = "Bohr"


def _build_scf(mol, reference: str):
    """Instantiate the scalar-X2C SCF object for ``reference``. Returns ``(mf, name)``."""
    from pyscf import scf

    ref = reference.lower()
    if ref == "auto":
        ref = "rhf" if mol.spin == 0 else "rohf"
    if ref == "rhf" and mol.spin != 0:
        raise ValueError("RHF requires a closed shell; this molecule has spin (2S) = {}. "
                         "Use 'rohf' or 'uhf'.".format(mol.spin))
    builders = {"rhf": scf.RHF, "rohf": scf.ROHF, "uhf": scf.UHF}
    if ref not in builders:
        raise ValueError("unknown reference {!r}; expected 'auto', 'rhf', 'rohf' or 'uhf'"
                         .format(reference))
    return builders[ref](mol).sfx2c1e(), ref


def run_scalar_x2c(molecule, *, reference: str = "auto", fitting: Optional[str] = None,
                   auxbasis: Optional[object] = None, with_soc: bool = True,
                   method: Optional[str] = None,
                   x2c_approx: Optional[str] = None, screening: Optional[str] = None,
                   screening_options: Optional[Dict[str, object]] = None,
                   configuration=None,
                   decoupling_options: Optional[Dict[str, object]] = None,
                   conv_tol: float = 1e-10, max_cycle: int = 200,
                   memory_gb: Optional[float] = None, n_active: Optional[int] = None,
                   cholesky_tol: float = DEFAULT_CHOLESKY_TOL, orbit_pivots: bool = True,
                   one_centre: bool = True,
                   gauge_origin=None, property_picture_change: bool = False,
                   anomaly_picture_change: bool = False,
                   atomic_reference: bool = False,
                   point_group: Optional[str] = None,
                   classification="auto",
                   verbose: int = 0) -> ScalarX2CData:
    """Run a scalar X2C (``sfx2c1e``) SCF and ingest it into :class:`ScalarX2CData`.

    Parameters
    ----------
    reference : str
        ``"auto"`` (RHF if closed shell, else ROHF), ``"rhf"``, ``"rohf"`` or ``"uhf"``.
        The restricted references give one MO set, UHF gives two; see
        :meth:`ScalarX2CData.mo_sets`. UHF costs spin purity (``<S^2>`` contamination is
        measured and reported) but can describe spin polarization the restricted references
        cannot, which matters for the antiferromagnetically coupled polynuclear systems of
        the polynuclear targets — there ROHF often will not even converge.
    fitting : str, optional
        ``None`` / ``"auto"`` (**the default**) — Cholesky decomposition, with the route the
        integrals take to it decided by the memory plan: conventional stored ERIs wherever
        their plan fits the configured limit, and the integral-direct evaluation where it
        does not (in which case the output states so on its "two-electron route" line). The
        two routes cost the same CPU to within a few per cent on anything larger than ~160
        AOs and produce factors that agree to every digit a result is read from, so the
        array is the entire decision. ``"conventional"`` / ``"cholesky"`` — always ingest
        conventional ERIs, which :mod:`kuiva.integrals.transform` Cholesky-decomposes; a
        system whose array does not fit is then refused rather than rerouted.
        ``"cholesky-direct"`` — always evaluate the integrals as the decomposition asks for
        them and never store them, which removes the ``O(nao^4/8)`` array that otherwise
        bounds the size of system that fits; the factors come back on
        :attr:`ScalarX2CData.factors`. ``"df"`` — density fitting; see ``auxbasis``.
    cholesky_tol, orbit_pivots, one_centre :
        The decomposition's own settings, used **only** when the integral-direct route runs
        (``fitting="cholesky-direct"``, or ``"auto"`` resolving to it),
        because that route decomposes here rather than downstream. They mean exactly what
        they mean in :meth:`kuiva.integrals.transform.ThreeIndexAO.from_scalar_data`, which
        is where the other routes take them from.
    auxbasis : optional
        Auxiliary basis for density fitting: a registry family name, a PySCF basis name, or
        any PySCF ``auxbasis`` specification. Supplying it selects the DF route. **The
        accuracy of the fit is then the user's responsibility** — Kuiva applies it faithfully
        and reports what it used, but a poor auxiliary shows up as an error in every
        two-electron integral, unbounded by any threshold.
    with_soc : bool
        Ingest the two-component X2C Hamiltonian (:class:`SpinOrbitX2C`). Without it the
        calculation has no spin-orbit coupling at all.
    method : str, optional
        The Hamiltonian, by name: ``"X2C-AMF"`` (**the default**), ``"X2C-1e"``,
        ``"X2C-AMF-DLU"``, ``"X2C-1e-DLU"``. Resolves to the ``x2c_approx`` and ``screening``
        axes below, which may be set directly instead. ⚠ Setting both a name and an axis that
        **contradicts** it raises rather than picking a winner, so no
        calculation runs with a Hamiltonian nobody asked for. See :mod:`kuiva.x2c.methods`.
    point_group : str, optional
        Turn on abelian double-group symmetry and label every orbital by its irrep.
        ``"auto"`` uses whatever operations the geometry has **in the frame it was given
        in**; a named group of the D2h chain is verified rather than assumed and refused if
        the molecule does not have it. ⚠ The molecule is never reoriented — that would move
        the gauge origin and every property operator fixed with it — so a symmetry axis that
        is not ``z`` is reported and not used, and the fix is to orient the input geometry.
        ⚠ Groups whose **double** group is non-abelian (``C2v``, ``D2``, ``D2h``) carry
        two-dimensional fermion irreps, which one integer cannot label; those reduce to the
        largest subgroup that does have one-dimensional fermion irreps, and the reduction is
        reported at the point of selection. Labels alone change nothing: they enable
        per-irrep state selection and the symmetry-preserving orbital optimizer, both of
        which are requested separately.
    classification : str or bool, optional
        The **non-abelian** classification layer, which needs ``point_group=`` to be on.
        ``"auto"`` (the default) detects the molecule's full point double group and activates
        the layer only when that group is genuinely larger than the abelian label group — the
        case in which a per-irrep count can cut a physically degenerate manifold with every
        abelian check still passing. A named group is verified rather than assumed;
        ``False`` switches it off. ⚠ It **classifies and never adapts**: converged states get
        the name of the multiplet they are, and the mathematics of every stage still runs in
        the abelian subgroup. It changes no number.
    property_picture_change : bool
        Apply the X2C picture change to the magnetic moment operator
        (:func:`picture_changed_moment`) instead of using the bare non-relativistic ``L`` and
        ``S``. ⚠ **Off by default, deliberately**: the bare operators are what OpenMolcas RASSI
        uses, which is what makes a cross-code comparison of a property dump like-for-like, and
        turning this on changes the meaning of every moment matrix and of any file they are
        written to.
    anomaly_picture_change : bool
        Also picture-change the spin operator used for the ``g_e - 2`` anomaly. Requires
        ``property_picture_change``; the effect is ``O((g_e - 2)/c^2)``.
    x2c_approx : str, optional
        The one-electron decoupling axis: ``"1e"`` (exact molecular, the default),
        ``"1e-dlu"`` (the local DLU approximation) or ``"atom1e"`` (PySCF's
        block-diagonal ``X`` with a molecular ``R`` — *not* DLU).
    decoupling_options : dict, optional
        Passed to :func:`local_x2c_hamiltonian` for ``x2c_approx="1e-dlu"``: ``partition``
        (``"atoms"``, or ``"single"`` for the exact transformation through the same code) and
        ``source`` (``"diagonal"`` or ``"isolated"``). ⚠ Supplying these on a non-local route
        raises rather than being ignored.
    screening : str, optional
        Two-electron spin-orbit picture change: ``"x2camf"`` (**default**),
        ``"none"`` or ``"x2camf-external"``. With ``"none"`` atomic j-splittings come out
        5-30% too large. The default costs one four-component atomic SCF **per unique
        element** — seconds for a light element, ~35 minutes for a lanthanide — and the result
        is geometry-independent and cached both in the process and on disk
        (:mod:`kuiva.amf.cache`), so a potential-energy surface pays it once ever.
    screening_options : dict, optional
        Extra arguments for the correction — ``interaction`` (``"coulomb"`` / ``"gaunt"`` /
        ``"breit"``), ``backend``, ``uncontract``. See
        :func:`kuiva.amf.correction.amf_correction`. Its ``configuration`` entry still
        works but ``configuration=`` below is the surface; giving both warns and the
        explicit argument wins.
    configuration : optional
        **One reference-state statement per atom**, feeding both the atomic mean field and
        the atomic-reference charges. Keys of a mapping are an element symbol (``"Ti"``),
        an atom label (``"Ti2"``) or a 1-based atom number (``3``), most specific wins; a
        scalar is allowed for a single-element molecule only. Each value is an oxidation
        state (``"+3"``, ``2``) — resolved to **the one canonical configuration of the
        curated common-states table**, warning outside it — or an explicit configuration
        (``"[Xe]4f1"``), checked against the table's accepted set and warned about when it
        is an excited or unusual reference (several configurations are accepted where the
        literature genuinely admits more than one, e.g. the d/s occupations of the late
        transition metals). ⚠ Atoms of one element with *different* reference states (or
        bases) get decorated symbols (``"Ti2"``) throughout output and provenance, and the
        charge report warns that non-default references are not comparable with default
        ones.
    memory_gb : float, optional
        Working-memory limit for this calculation, overriding the configured default.
        Without a default *and* without this, the calculation refuses to start.
    n_active : int, optional
        Active-space size, if it is already known. Only used to sharpen the memory pre-flight
        — the earlier a size is known, the earlier a calculation that cannot fit is stopped.
        It adds the multireference phases, and it makes the three-index MO block exact rather
        than bounded by the occupied space, so it is worth passing whenever the active space
        is narrower than the occupied one, which is the usual case.
    gauge_origin : optional
        Gauge origin for the orbital angular momentum of the property dump: ``"mass"``
        (the default centre of mass), ``"charge"``, ``"origin"``, or three coordinates in
        bohr. It is ingested here because the multireference layer never calls PySCF again
, and it is recorded in the dump header because ``L`` is defined relative to it.
    atomic_reference : bool
        Also compute the per-element free-atom reference orbitals behind the
        atomic-reference charges (:func:`kuiva.props.population.atomic_reference_charges`).
        **Off by default** because it costs one spherically constrained atomic SCF per
        unique element — sub-second for a light element, ~10 s for a lanthanide — cached in
        the process, and it must run *here*: the analysis layer has no integral library. The
        reference state per element is the atomic mean field's default (neutral atom;
        trivalent ion on the f block), overridden by the same
        ``screening_options["configuration"]`` mapping the mean field takes, so one element
        has one reference across the program; a non-default reference is recorded and the
        charge report warns that its charges are not comparable with default-reference ones.
    """
    from ..x2c.methods import DEFAULT_METHOD, resolve

    # ⚠ Resolved once, here, and everything downstream reads the resolved pair. A method name
    # and an explicit axis that contradict each other are refused rather than reconciled
    #, so no calculation can run with a Hamiltonian nobody asked for.
    if method is None and x2c_approx is None and screening is None:
        method = DEFAULT_METHOD
    chosen = resolve(method, decoupling=x2c_approx, screening=screening)

    # One reference-state statement per atom, feeding BOTH consumers (the atomic mean field
    # and the atomic-reference charges). The screening_options back door keeps working; when
    # both are given the explicit argument wins, announced.
    legacy_cfg = (screening_options or {}).get("configuration")
    if configuration is not None and legacy_cfg is not None:
        log.warning("both configuration= and screening_options['configuration'] were "
                    "given; the explicit configuration= argument is used and the "
                    "screening_options entry is ignored.")
    elif configuration is None:
        configuration = legacy_cfg

    mol = build_mole(molecule, verbose=verbose, configuration=configuration)
    atom_basis = mol.__dict__["_kuiva_atom_basis"]
    meta = mol.__dict__["_kuiva_basis_meta"]
    fit_route, aux = _choose_fit(atom_basis, fitting, auxbasis)

    # The AO count is the first thing that is known and it fixes every large array of
    # the front-end, so the whole plan is printed and judged here, before the SCF — and it
    # is also what resolves fitting="auto", which must happen first so the pre-flight judges
    # the plan of the route that will actually run.
    res.ensure_configured(memory_gb)
    fit_note = ""
    if fit_route == "auto":
        fit_route, fit_note = _auto_fit_route(
            mol.nao, nelec=mol.nelectron, n_active=n_active,
            screening=with_soc and chosen.screening != "none")
    res.preflight(memory_plan(mol.nao, conventional=fit_route == "conventional",
                              direct=fit_route == "direct",
                              shell_ao_max=_shell_ao_max(mol), n_shells=int(mol.nbas),
                              n_active=n_active, nelec=mol.nelectron,
                              screening=with_soc and chosen.screening != "none"))
    if fit_route == "conventional":
        _reserve_eri_memory(mol.nao)
    _set_pyscf_memory(mol)

    mf, ref_name = _build_scf(mol, reference)
    if fit_route == "df":
        # Resolve a registry/BSE auxiliary explicitly (don't rely on PySCF's implicit BSE
        # fallback); pass PySCF-bundled auxiliaries and raw specifications through unchanged.
        aux_spec = aux
        if isinstance(aux, str) and reg.has_family(aux) and \
                reg.get_family(aux).provider is reg.Provider.BSE:
            aux_spec = reg.resolve_for_pyscf(aux, list(atom_basis.keys()))
        mf = mf.density_fit(auxbasis=aux_spec)
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    with timer("scalar X2C SCF") as t_scf:
        e_scf = mf.kernel()
    if not mf.converged:
        log.error("scalar X2C SCF did not converge in %d cycles (E = %.8f Eh); everything "
                  "downstream is built on these orbitals", max_cycle, e_scf)

    mo_coeff = np.asarray(mf.mo_coeff)
    unrestricted = mo_coeff.ndim == 3
    nmo = int(mo_coeff.shape[-1])

    s2_dev = None
    if unrestricted:
        s2, mult = mf.spin_square()
        s_exact = 0.5 * mol.spin
        s2_dev = float(s2 - s_exact * (s_exact + 1.0))
        if abs(s2_dev) > 0.1:
            log.warning("unrestricted reference is spin contaminated: <S^2> = %.4f against "
                        "the exact %.4f (deviation %.4f). The orbitals are a guess for the "
                        "multireference step, so this is not fatal, but a strongly "
                        "contaminated set is a poor starting point.",
                        s2, s_exact * (s_exact + 1.0), s2_dev)

    with timer("AO integrals") as t_ints:
        s_ao = mol.intor("int1e_ovlp")
        h_x2c = mf.get_hcore()           # spin-free X2C one-electron Hamiltonian (AO)
        if unrestricted:
            h_x2c = np.asarray(h_x2c)
            if h_x2c.ndim == 3:          # UHF may hand back one copy per spin; they are equal
                h_x2c = h_x2c[0]

        # the dump's property operators. One cheap intor, ingested here because nothing downstream
        # may call PySCF — and the gauge origin has to be fixed before the Mole is gone. The
        # optional picture change is built here for the same reason and no other: it needs the
        # four-component problem, the gauge origin and the Mole all at once, and this is the
        # only place all three exist.
        props = ingest_property_integrals(
            mol, gauge_origin, picture_change=bool(property_picture_change),
            approx=chosen.decoupling, decoupling_options=decoupling_options,
            anomaly_picture_change=bool(anomaly_picture_change))

        eri = None
        df_cderi = None
        if fit_route == "df":
            # DF factors, packed over the lower-triangular AO pair index.
            df_cderi = np.asarray(mf.with_df._cderi)
        elif fit_route == "conventional":
            eri = mol.intor("int2e", aosym="s8")

    # ⚠ **The integral-direct decomposition runs here, and it has to.** It is the one step
    # that needs the integrals without needing them stored, so it belongs where the evaluator
    # is still alive — nothing downstream may call PySCF, and no Mole crosses this boundary.
    # The container then carries finished factors instead of an array, and everything after
    # this point sees the same object it would have seen on either other route.
    factors = None
    if fit_route == "direct":
        factors = _direct_cholesky(mol, tol=cholesky_tol, orbit_pivots=orbit_pivots,
                                   one_centre=one_centre)

    # The mean field takes the SAME resolved per-label reference states the charges do —
    # one statement per atom, everywhere — so the raw user spec never reaches it twice.
    screen_opts = dict(screening_options or {})
    if mol.__dict__["_kuiva_config_given"]:
        screen_opts["configuration"] = {
            lab: cfg for lab, (cfg, _d) in zip(mol.__dict__["_kuiva_atom_labels"],
                                               _stashed_configs(mol))}
    soc = (ingest_spin_orbit(mol, h_x2c, chosen.decoupling, screening=chosen.screening,
                             decoupling_options=decoupling_options,
                             **screen_opts) if with_soc else None)

    # ⚠ Must run here for the same reason the direct decomposition does: the reference is
    # per (element, basis) and needs the integral library, which nothing downstream has.
    atomic_ref = _atomic_reference_set(mol) if atomic_reference else None

    data_layout = ao_layout(mol)

    # Standard output block: the front-end's contribution to the output file.
    rows = [
        ("SCF reference", ref_name.upper() + (" (unrestricted)" if unrestricted else "")),
        ("SCF one-electron Hamiltonian", "X2C (sfx2c1e, spin-free)"),
        ("AO basis functions", int(mol.nao)),
        ("molecular orbitals", nmo, "", "2 spin sets" if unrestricted else ""),
        ("electrons (alpha, beta)", "{}, {}".format(*mol.nelec)),
        ("two-electron route", fit_route + (" [{}]".format(aux) if aux else ""), "",
         fit_note),
        ("gauge origin (property operators)", "({:.4f}, {:.4f}, {:.4f}) bohr".format(
            *np.asarray(props.gauge_origin).ravel()), "", props.origin_label),
        ("nuclear repulsion", float(mol.energy_nuc()), "Eh", "", out.E_FMT),
        ("scalar X2C SCF energy", float(e_scf), "Eh", "", out.E_FMT),
        ("SCF converged", bool(mf.converged)),
    ]
    if s2_dev is not None:
        rows.append(("<S^2> deviation from exact", s2_dev, "", "", "{:.4f}"))
    rows += [
        ("SCF time", t_scf.wall, "s wall", "{:.2f} s cpu".format(t_scf.cpu), out.TIME_FMT),
        ("AO integral time", t_ints.wall, "s wall",
         "{:.2f} s cpu".format(t_ints.cpu), out.TIME_FMT),
    ]
    out.entries(log, rows)
    if soc is not None:
        out.subsection(log, "Two-component X2C spin-orbit Hamiltonian")
        soc.report()
    else:
        log.warning("spin-orbit coupling was not ingested (with_soc=False): the calculation "
                    "will be scalar-relativistic only")

    # Symmetry last, because it may *replace* the orbital set with a symmetry-adapted one and
    # everything above reads mo_coeff. The adaptation is a rotation inside a degenerate block:
    # the density, the energy and every observable are untouched, and it is what makes a label
    # exist for an orbital the SCF was entitled to return as an arbitrary mixture of partners.
    symmetry = None
    if point_group is not None:
        from ..symm import analyze as _analyze_symmetry
        from ..symm import report as _report_symmetry
        with timer("symmetry labelling"):
            symmetry, adapted = _analyze_symmetry(
                data_layout, tuple(np.asarray(c) for c in
                                   (mo_coeff if unrestricted else (mo_coeff,))),
                s_ao, point_group=point_group, mo_energy=mf.mo_energy,
                classification=classification)
        mo_coeff = np.asarray(adapted) if unrestricted else adapted[0]
        _report_symmetry(symmetry, log, spinor_labels=symmetry.spinor_labels())

    na, nb = mol.nelec
    data = ScalarX2CData(
        nao=mol.nao, nmo=nmo, nelec=(int(na), int(nb)),
        e_scf=float(e_scf), converged=bool(mf.converged),
        s_ao=np.ascontiguousarray(s_ao), h_x2c=np.ascontiguousarray(h_x2c),
        mo_coeff=np.ascontiguousarray(mo_coeff),
        mo_energy=np.ascontiguousarray(mf.mo_energy),
        mo_occ=np.ascontiguousarray(mf.mo_occ),
        fit_route=fit_route, eri=eri, df_cderi=df_cderi, factors=factors,
        aux_name=aux if isinstance(aux, str) else ("custom" if aux is not None else None),
        soc=soc, reference=ref_name, unrestricted=unrestricted, s2_deviation=s2_dev,
        basis_meta=meta, e_nuc=float(mol.energy_nuc()), ao_layout=data_layout,
        properties=props, atomic_reference=atomic_ref, symmetry=symmetry,
        molecule=MoleculeSpec.from_molecule(molecule, configuration),
    )
    return data


def _run_aoc_scf(mf, mol, config, layout, element: str, *, spherical: bool = True):
    """Install the average-of-configuration machinery on ``mf`` and run it.

    The one assembly of the AOC SCF, shared by :func:`run_scalar_aoc` and the atomic
    reference builder (:func:`_atomic_reference_entry`), so the two cannot drift apart on
    which constraints an atomic reference carries. Returns ``(e_scf, state)``.
    """
    from ..amf.configuration import (angular_channel_groups,
                                     install_configuration_average, spherical_projector)

    state: Dict[str, object] = {}
    mf.get_occ = _aoc_occupation(mol, config, layout.ao_l, state)
    if not config.is_closed_shell:
        # ⚠ **The occupations alone are not average of configuration.** Without this the
        # two-electron energy is evaluated over the fractional *density*, whose open-open pair
        # average factorizes into a product of one-particle averages — 0.3-0.5 Eh and up to
        # 15% on a splitting. The same function drives the four-component backend; see its
        # docstring for why one implementation covers both conventions.
        install_configuration_average(mf, mol, state)
    if spherical:
        # ⚠ **And the occupations are not sphericity.** Filling a whole ``l`` shell equally
        # makes the density spherical *given* spherical orbitals; it does nothing to stop the
        # iteration from sliding into the lower, symmetry-broken solutions a fractionally
        # occupied Hartree-Fock functional has, which it does — the anisotropy grows about an
        # order of magnitude per cycle from roundoff. Projecting the Fock onto its rank-zero
        # part each cycle imposes the symmetry that *defines* the average of configuration.
        # Installed after the effective Fock above, so the projection is the last thing before
        # the eigensolver.
        project = spherical_projector(
            angular_channel_groups(layout.ao_l, layout.ao_m, layout.ao_shell).values(),
            int(mol.nao))
        inner_get_fock = mf.get_fock
        mf.get_fock = lambda *a, **k: project(inner_get_fock(*a, **k))
    else:
        log.warning(
            "the average-of-configuration SCF for %s is running WITHOUT the spherical "
            "constraint. An open-shell atom then converges to a symmetry-broken solution "
            "whose shells no longer share one radial function; this setting exists to "
            "measure that and is never a production one.", element)
    e_scf = mf.kernel()
    if not mf.converged:
        log.error("scalar average-of-configuration SCF did not converge in %d cycles "
                  "(E = %.8f Eh); everything downstream is built on these orbitals. An atom "
                  "with several shells close in energy often needs level_shift= or damp=.",
                  mf.max_cycle, e_scf)
    return e_scf, state


#: In-process cache of free-atom reference orbitals, keyed ``(element, basis spec,
#: configuration)``. Geometry-independent by construction, so one entry serves every
#: molecule and every geometry in the process; cheap enough (an atomic scalar SCF) that a
#: persistent on-disk cache is not worth a formula version.
_ATOMIC_REFERENCE_CACHE: Dict[tuple, object] = {}


def _atomic_reference_entry(element: str, family, config, is_default: bool):
    """The free-atom reference orbitals of one element, in one basis family (cached).

    ``config`` is an already-resolved :class:`~kuiva.amf.configuration.AtomicConfiguration`
    — resolution, the curated-table checks and their warnings happened exactly once, in
    :func:`kuiva.amf.oxidation.resolve_reference_configuration`. The default is the atomic
    mean field's (neutral atom, trivalent ion on the f block; a user decision so each
    element has one default reference across the program), and a non-default entry is what
    makes the charge report downstream warn about comparability.

    The SCF is the same spherically constrained average-of-configuration assembly the AOC
    driver runs (:func:`_run_aoc_scf`), in the *molecule's own basis for this element* — the
    same basis entry produces the same AO ordering for the free atom as for the atom inside
    the molecule, which is what makes the downstream block-diagonal placement exact. An
    anion reference (more electrons than ``Z``) is allowed — the resolver has already warned
    that the finite basis acts as its confinement.
    """
    from pyscf import gto
    from pyscf.scf import hf as pyscf_hf

    from ..basis.reference import AtomicReferenceEntry

    key = (element.capitalize(), str(family), config)
    hit = _ATOMIC_REFERENCE_CACHE.get(key)
    if hit is not None:
        return hit

    z = int(gto.charge(element))
    n_elec = config.n_electrons
    if n_elec < 1:
        raise ValueError(
            "the reference configuration {} has {} electrons; an atomic reference needs at "
            "least one".format(config.canonical, n_elec))
    mol = build_mole(_SingleAtom(atoms=[(element, (0.0, 0.0, 0.0))], charge=z - n_elec,
                                 spin=n_elec % 2, basis={element: family}), verbose=0)
    _set_pyscf_memory(mol)
    layout = ao_layout(mol)
    mf = pyscf_hf.RHF(mol).sfx2c1e()
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    with timer("atomic reference orbitals"):
        _run_aoc_scf(mf, mol, config, layout, element, spherical=True)
    entry = AtomicReferenceEntry(
        element=element.capitalize(), c=np.asarray(mf.mo_coeff),
        occ=np.asarray(mf.mo_occ), configuration=config.label or config.canonical,
        is_default=bool(is_default), converged=bool(mf.converged))
    _ATOMIC_REFERENCE_CACHE[key] = entry
    return entry


def _atomic_reference_set(mol):
    """Free-atom reference orbitals for every label group of a built molecule.

    One entry per unique atom label (plain element symbol, or the decorated ``"Ti2"`` of an
    atom with its own basis or reference state — :func:`build_mole` decides which), plus the
    per-atom key list the charge partition maps atoms through. Same element, same family,
    same configuration share one cached atomic solve however many labels point at it.
    """
    from ..basis.reference import AtomicReferenceSet

    labels = mol.__dict__["_kuiva_atom_labels"]
    families = mol.__dict__["_kuiva_atom_families"]
    configs = _stashed_configs(mol)
    entries = {}
    for ia, label in enumerate(labels):
        if label in entries:
            continue
        cfg, is_default = configs[ia]
        entries[label] = _atomic_reference_entry(
            mol.atom_pure_symbol(ia), families[ia], cfg, is_default)
    return AtomicReferenceSet(
        entries=entries, atom_keys=list(labels),
        basis_label=", ".join("{}: {}".format(lab, mol.__dict__["_kuiva_atom_basis"][lab])
                              for lab in sorted(entries)))


def run_scalar_aoc(element: str, configuration=None, *, basis,
                   fitting: Optional[str] = None, auxbasis: Optional[object] = None,
                   with_soc: bool = True, method: Optional[str] = None,
                   x2c_approx: Optional[str] = None, screening: Optional[str] = None,
                   screening_options: Optional[Dict[str, object]] = None,
                   decoupling_options: Optional[Dict[str, object]] = None,
                   conv_tol: float = 1e-10, max_cycle: int = 200,
                   level_shift: float = 0.0, damp: float = 0.0,
                   init_guess: Optional[str] = None,
                   spherical: bool = True,
                   memory_gb: Optional[float] = None,
                   cholesky_tol: float = DEFAULT_CHOLESKY_TOL, orbit_pivots: bool = True,
                   one_centre: bool = True, gauge_origin=None,
                   verbose: int = 0) -> ScalarX2CData:
    """Scalar X2C SCF on **one atom or ion, averaged over a configuration**.

    The average-of-configuration counterpart of :func:`run_scalar_x2c`, returning the same
    :class:`ScalarX2CData`. Where an ordinary SCF picks one determinant out of a partly filled
    shell — breaking the spherical symmetry of an atom and baking an arbitrary spatial
    orientation into the orbitals — this occupies the frontier shell of each ``l`` channel
    **equally** and minimizes the true configuration-average energy, so the solution is
    spherical and every orbital of a shell has one radial function. That is the reference an
    atomic *shell* quantity has to come from.

    ⚠ **It is spin-restricted and there is no spin-polarized variant**, by decision: the
    average is over *all* microstates of the configuration (the grand average), which is the
    only one that is exactly spherical and the one atomic-parameter conventions are defined
    against. It is therefore not the ground state of the atom, and its total energy is above
    the true one by the term energy the average discards — a well-defined choice, not an
    approximation to a Hartree-Fock ground state.

    ⚠ **What comes back is a valid ``ScalarX2CData`` whose occupations are fractional.** The
    downstream pipeline stages have not been validated on one and are not the purpose of this
    function; see :class:`ScalarX2CData`.

    Parameters
    ----------
    element : str
        Chemical symbol. The atom sits at the origin; the **charge is derived** from the
        configuration's electron count, so the two can never disagree.
    configuration
        The reference configuration: a string (``"[Xe] 4f9 5d1 6s1"``), an
        :class:`kuiva.amf.configuration.AtomicConfiguration`, an oxidation state, any object
        with a ``to_atomic()`` method (a shell-resolved configuration), or ``None`` for
        :func:`kuiva.amf.configuration.default_configuration`.
    basis : str or dict
        As :func:`run_scalar_x2c`, through the same registry.
    level_shift, damp, init_guess
        Convergence aids, passed to PySCF. ⚠ **An open-shell atom with several shells within
        an eV of each other is where an SCF actually fails to converge** — the lanthanides
        this function exists for are exactly that case — so these are exposed rather than
        left to be rediscovered.
    spherical : bool
        Constrain the SCF to spherically symmetric solutions by projecting the Fock operator
        onto its rank-zero part each cycle
        (:func:`kuiva.amf.configuration.spherical_projector`). ⚠ **On by default, and it is a
        statement about which state is being solved rather than a convergence aid**: the
        spherical solution is an *unstable* fixed point for an open shell, so without it the
        anisotropy grows from roundoff until the diagnostics below fire — or until the run
        stops just under them, which is worse. ``False`` exists to measure that and warns.
    with_soc, method, x2c_approx, screening, screening_options, decoupling_options, fitting,
    auxbasis, conv_tol, max_cycle, memory_gb, gauge_origin, verbose
        As :func:`run_scalar_x2c`.

        ⚠ **The atomic mean field defaults to *this* configuration**, not to the element's
        default reference, whenever screening is on and ``screening_options`` names none: a
        parameter extracted from an AOC reference and screened by the mean field of a
        different ion would be a mixture of two states. It costs one four-component atomic
        solve per ``(element, basis, configuration)`` — tens of minutes for a lanthanide, once
        ever — so pass ``screening="none"`` wherever spin-orbit coupling is not wanted.

    Notes
    -----
    Two diagnostics run afterwards and both are advisory:

    * the **density anisotropy** (:func:`kuiva.amf.pyscf_dhf.density_anisotropy`), which under
      average of configuration is the assertion that the averaging *worked* — a solution that
      comes back anisotropic occupied something that is not a shell;
    * the **assignment purity**, the smallest Mulliken weight any occupied orbital has on the
      channel it was assigned to. Below :data:`AOC_ASSIGNMENT_TOLERANCE` two channels are
      genuinely mixed in one orbital and the per-``l`` filling is then not the rule the user
      thinks it is.
    """
    from pyscf import gto
    from pyscf.scf import hf as pyscf_hf

    from ..amf.configuration import SHELL_LETTERS, AtomicConfiguration
    from ..amf.pyscf_dhf import SPHERICAL_DENSITY_TOLERANCE, density_anisotropy
    from ..x2c.methods import DEFAULT_METHOD, resolve

    if method is None and x2c_approx is None and screening is None:
        method = DEFAULT_METHOD
    chosen = resolve(method, decoupling=x2c_approx, screening=screening)

    # ⚠ Through the curated-table resolver, not bare coerce: an oxidation state produces
    # the table's one canonical configuration, and an unusual state or an excited explicit
    # configuration warns here exactly as it does on the molecular path. (A shell-resolved
    # configuration is duck-typed via ``to_atomic()`` inside the resolver — the per-l form
    # is what an SCF and an atomic mean field are defined by.)
    from ..amf.oxidation import resolve_reference_configuration
    config, _config_is_default = resolve_reference_configuration(element, configuration)
    z = int(gto.charge(element))
    n_elec = config.n_electrons
    if not 1 <= n_elec <= z:
        raise ValueError(
            "the configuration {} has {} electrons, which is not a bound state of {} "
            "(Z = {})".format(config.canonical, n_elec, element, z))
    charge = z - n_elec

    mol = build_mole(_SingleAtom(atoms=[(element, (0.0, 0.0, 0.0))], charge=charge,
                                 spin=n_elec % 2, basis=basis), verbose=verbose)
    atom_basis = mol.__dict__["_kuiva_atom_basis"]
    meta = mol.__dict__["_kuiva_basis_meta"]
    fit_route, aux = _choose_fit(atom_basis, fitting, auxbasis)

    res.ensure_configured(memory_gb)
    fit_note = ""
    if fit_route == "auto":
        fit_route, fit_note = _auto_fit_route(
            mol.nao, nelec=mol.nelectron,
            screening=with_soc and chosen.screening != "none")
    res.preflight(memory_plan(mol.nao, conventional=fit_route == "conventional",
                              direct=fit_route == "direct",
                              shell_ao_max=_shell_ao_max(mol), n_shells=int(mol.nbas),
                              nelec=mol.nelectron,
                              screening=with_soc and chosen.screening != "none"))
    if fit_route == "conventional":
        _reserve_eri_memory(mol.nao)
    _set_pyscf_memory(mol)

    layout = ao_layout(mol)
    # ⚠ ``pyscf.scf.hf.RHF``, never ``pyscf.scf.RHF``: the latter is a dispatcher that hands
    # back an ROHF for a molecule with spin > 0, and an odd-electron average of configuration
    # would then silently run a different (aufbau, symmetry-broken) method. ``mol.spin`` here
    # is a parity setting only — the occupations are supplied below and overrule it.
    mf = pyscf_hf.RHF(mol).sfx2c1e()
    if fit_route == "df":
        aux_spec = aux
        if isinstance(aux, str) and reg.has_family(aux) and \
                reg.get_family(aux).provider is reg.Provider.BSE:
            aux_spec = reg.resolve_for_pyscf(aux, list(atom_basis.keys()))
        mf = mf.density_fit(auxbasis=aux_spec)
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    mf.level_shift = level_shift
    mf.damp = damp
    if init_guess is not None:
        mf.init_guess = init_guess
    with timer("scalar AOC X2C SCF") as t_scf:
        e_scf, state = _run_aoc_scf(mf, mol, config, layout, element, spherical=spherical)

    mo_coeff = np.ascontiguousarray(np.asarray(mf.mo_coeff))
    mo_occ = np.ascontiguousarray(np.asarray(mf.mo_occ))
    nmo = int(mo_coeff.shape[-1])
    dm = np.asarray(mf.make_rdm1())

    anisotropy = density_anisotropy(mol, dm)
    if anisotropy > SPHERICAL_DENSITY_TOLERANCE:
        log.warning("the converged average-of-configuration density is not spherical "
                    "(anisotropy %.2e against a tolerance of %.0e). An atom is spherically "
                    "symmetric, so the averaging did not do what it claims: the orbitals a "
                    "channel was filled with are probably not one shell.",
                    anisotropy, SPHERICAL_DENSITY_TOLERANCE)
    purity = float(state.get("assignment_weight", 1.0))
    if purity < AOC_ASSIGNMENT_TOLERANCE:
        log.warning("an occupied orbital carries only %.2f of its Mulliken weight on the "
                    "angular-momentum channel it was assigned to. The configuration was "
                    "filled per channel, so a genuinely mixed orbital means the occupied "
                    "shells are not the ones %s names.", purity, config.canonical)
    if abs(float(mo_occ.sum()) - n_elec) > 1e-8:
        raise RuntimeError(
            "the converged occupations hold {:.6f} electrons where the configuration {} has "
            "{}".format(float(mo_occ.sum()), config.canonical, n_elec))

    with timer("AO integrals") as t_ints:
        s_ao = mol.intor("int1e_ovlp")
        h_x2c = np.asarray(mf.get_hcore())
        props = ingest_property_integrals(mol, gauge_origin)
        eri = None
        df_cderi = None
        if fit_route == "df":
            df_cderi = np.asarray(mf.with_df._cderi)
        elif fit_route == "conventional":
            eri = mol.intor("int2e", aosym="s8")

    # Same reason as in the molecular driver: the direct route decomposes here, where the
    # integrals can still be evaluated, and hands on the factors rather than their input.
    factors = (_direct_cholesky(mol, tol=cholesky_tol, orbit_pivots=orbit_pivots,
                                one_centre=one_centre) if fit_route == "direct" else None)

    screen_opts = dict(screening_options or {})
    screen_opts.setdefault("configuration", config)
    soc = (ingest_spin_orbit(mol, h_x2c, chosen.decoupling, screening=chosen.screening,
                             decoupling_options=decoupling_options,
                             **screen_opts) if with_soc else None)

    open_shells = ", ".join("{}^{}".format(SHELL_LETTERS[l], q)
                            for l, q in config.open_shells()) or "none (closed shell)"
    rows = [
        ("SCF reference", "average of configuration (spin-restricted)"),
        ("atom", "{} ({:+d})".format(element, charge) if charge else element),
        ("configuration", config.label, "", config.canonical),
        ("open shells", open_shells),
        ("electrons", n_elec),
        ("SCF one-electron Hamiltonian", "X2C (sfx2c1e, spin-free)"),
        ("AO basis functions", int(mol.nao)),
        ("molecular orbitals", nmo),
        ("two-electron route", fit_route + (" [{}]".format(aux) if aux else ""), "",
         fit_note),
        ("nuclear repulsion", float(mol.energy_nuc()), "Eh", "", out.E_FMT),
        ("scalar AOC X2C SCF energy", float(e_scf), "Eh", "", out.E_FMT),
        ("SCF converged", bool(mf.converged)),
        ("density anisotropy", anisotropy, "", "spherical below {:.0e}".format(
            SPHERICAL_DENSITY_TOLERANCE), out.SCI_FMT),
        ("angular-momentum assignment", purity, "", "min Mulliken weight, occupied",
         "{:.4f}"),
        ("SCF time", t_scf.wall, "s wall", "{:.2f} s cpu".format(t_scf.cpu), out.TIME_FMT),
        ("AO integral time", t_ints.wall, "s wall",
         "{:.2f} s cpu".format(t_ints.cpu), out.TIME_FMT),
    ]
    out.entries(log, rows)
    shells = state.get("shells") or []
    if shells:
        tab = out.Table(log, [out.Column("shell", "{:s}", 6),
                              out.col_count("electrons", 10),
                              out.col_count("orbitals", 9),
                              out.Column("occupation", "{:.6f}", 11),
                              out.Column("coupling a", "{:.6f}", 11)])
        tab.start("open shells, average of configuration")
        for shell in shells:
            tab.row(SHELL_LETTERS[shell.l], shell.q, int(shell.index.size),
                    float(shell.q) / (2 * shell.l + 1), shell.coupling)
        tab.end()
    if soc is not None:
        out.subsection(log, "Two-component X2C spin-orbit Hamiltonian")
        soc.report()

    # ⚠ A spin-restricted average of configuration has **no meaningful per-spin split**: the
    # electrons are shared over whole shells and both spins carry half of each occupation.
    # The pair is the formal ``(ceil, floor)`` division so that ``nelec_total`` is right, and
    # anything reading the two halves separately is reading something that is not there.
    na = (n_elec + 1) // 2
    return ScalarX2CData(
        nao=mol.nao, nmo=nmo, nelec=(int(na), int(n_elec - na)),
        e_scf=float(e_scf), converged=bool(mf.converged),
        s_ao=np.ascontiguousarray(s_ao), h_x2c=np.ascontiguousarray(h_x2c),
        mo_coeff=mo_coeff, mo_energy=np.ascontiguousarray(np.asarray(mf.mo_energy)),
        mo_occ=mo_occ, fit_route=fit_route, eri=eri, df_cderi=df_cderi, factors=factors,
        aux_name=aux if isinstance(aux, str) else ("custom" if aux is not None else None),
        soc=soc, reference="aoc", unrestricted=False, s2_deviation=None,
        basis_meta=meta, e_nuc=float(mol.energy_nuc()), ao_layout=layout,
        properties=props,
    )


__all__ = ["ScalarX2CData", "SpinOrbitX2C", "PropertyIntegrals", "MoleculeSpec",
           "build_mole", "cross_overlap",
           "run_scalar_x2c", "run_scalar_aoc", "ingest_spin_orbit",
           "ingest_property_integrals", "gauge_origin_for", "eri_memory_gb", "ao_layout"]
