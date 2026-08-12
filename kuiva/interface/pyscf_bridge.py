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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..amf.correction import ScreeningRecord
from ..basis import registry as reg
from ..basis.layout import AOLayout, Shell, build_layout
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

    @property
    def nao(self) -> int:
        return int(self.irxp.shape[-1])

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
        return {
            "gauge_origin_bohr": [float(x) for x in np.asarray(self.gauge_origin).ravel()],
            "gauge_origin_choice": self.origin_label,
            "picture_change": "none (bare AO operators, used unchanged in the 2c basis)",
        }

    def __repr__(self) -> str:
        r = np.asarray(self.gauge_origin).ravel()
        return "PropertyIntegrals(nao={}, gauge origin {} = ({:.4f}, {:.4f}, {:.4f}) bohr)" \
            .format(self.nao, self.origin_label, r[0], r[1], r[2])


def gauge_origin_for(mol, origin=None) -> Tuple[np.ndarray, str]:
    """Resolve a gauge origin: an explicit ``(x, y, z)`` in bohr, or a named choice.

    ``None`` and ``"mass"`` give the **centre of mass** (the default); ``"charge"`` the
    centre of nuclear charge; ``"origin"`` the coordinate origin. Returns
    ``(coordinates [bohr], label)``.
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
        raise ValueError("unknown gauge origin {!r}; expected 'mass', 'charge', 'origin' or "
                         "three coordinates in bohr".format(origin))
    r = np.asarray(origin, dtype=float).ravel()
    if r.size != 3:
        raise ValueError("an explicit gauge origin is three coordinates in bohr, got {}"
                         .format(np.shape(origin)))
    return r, "explicit"


def ingest_property_integrals(mol, gauge_origin=None) -> PropertyIntegrals:
    """Orbital angular momentum about a gauge origin, as plain arrays.

    ⚠ **The gauge origin is a real choice and it changes the answer.** ``L`` is defined
    relative to it, so for a charged system every orbital moment matrix moves with it. The
    default is the centre of mass; it is recorded in :class:`PropertyIntegrals` and written
    into the dump header, because a stored moment matrix that does not say where its origin
    was is not interpretable.
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
    return PropertyIntegrals(irxp=np.ascontiguousarray(irxp),
                             gauge_origin=np.ascontiguousarray(r_g), origin_label=label)


@dataclass(frozen=True)
class ScalarX2CData:
    """Self-contained scalar-relativistic X2C reference (the ingestion boundary).

    The scalar quantities are real (the scalar guess carries no SOC); spin-orbit coupling
    arrives separately in :attr:`soc` and is applied at the multireference level.
    ``eri`` and ``df_cderi`` are mutually exclusive: exactly one is populated per
    ``fit_route``.

    **Restricted and unrestricted references.** RHF/ROHF give one set of MOs and
    ``mo_coeff`` has shape ``(nao, nmo)``; UHF gives two and it has shape ``(2, nao, nmo)``
    with ``unrestricted = True``. Use :meth:`mo_sets` rather than branching on the shape.
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
    fit_route: str                         # "conventional" | "df"
    eri: Optional[np.ndarray] = None       # 8-fold packed (nao_pair_pair,) if conventional
    df_cderi: Optional[np.ndarray] = None  # (naux, nao_pair) DF factors if df
    aux_name: Optional[str] = None
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
            f"df_cderi{self.df_cderi.shape}" if self.df_cderi is not None else "none")
        return (f"ScalarX2CData(nao={self.nao}, nmo={self.nmo}, nelec={self.nelec}, "
                f"E={self.e_scf:.8f} Eh, conv={self.converged}, ref={self.reference}, "
                f"route={self.fit_route}, {two_e}, soc={self.has_soc})")


def _resolve_basis(atoms, basis) -> Tuple[object, Dict[str, str]]:
    """Turn a basis spec (str, or {symbol: family}) into a PySCF ``basis`` and metadata.

    A single family name is applied to every atom; a dict assigns per-atom families. Both go
    through the registry so coverage and (for BSE families) data fetching are handled there.
    """
    symbols = sorted({a[0].capitalize() for a in atoms})
    if isinstance(basis, str):
        atom_basis = {s: basis for s in symbols}
    else:
        atom_basis = {s.capitalize(): b for s, b in basis.items()}
        missing = [s for s in symbols if s not in atom_basis]
        if missing:
            raise ValueError(f"no basis assigned for atom(s) {missing}")

    report = reg.check_consistency(atom_basis)
    if not report.ok:
        raise ValueError("basis consistency check failed:\n  " + "\n  ".join(report.errors))

    pyscf_basis: Dict[str, object] = {}
    meta: Dict[str, str] = {}
    for sym, fam_name in atom_basis.items():
        fam = reg.get_family(fam_name)
        pyscf_basis[sym] = reg.resolve_for_pyscf(fam_name, [sym])[sym] \
            if fam.provider is reg.Provider.BSE else fam.provider_name
        meta[sym] = f"{fam.name} [{fam.rel_treatment.value}, {fam.contraction.value}, " \
                    f"fit={fam.fit_route().value}]"
    return pyscf_basis, (meta, atom_basis)


def build_mole(molecule, verbose: int = 0):
    """Build a PySCF ``Mole`` from a Kuiva ``Molecule`` (duck-typed), via the registry.

    ``molecule`` must expose ``atoms`` (list of ``(symbol, (x, y, z))``), ``charge``,
    ``spin`` (2S), ``basis`` and ``unit``.
    """
    from pyscf import gto
    pyscf_basis, (meta, atom_basis) = _resolve_basis(molecule.atoms, molecule.basis)
    atom_str = [(a[0].capitalize(), tuple(a[1])) for a in molecule.atoms]
    mol = gto.M(atom=atom_str, basis=pyscf_basis, charge=molecule.charge,
                spin=molecule.spin, unit=molecule.unit, verbose=verbose)
    mol.__dict__["_kuiva_basis_meta"] = meta
    mol.__dict__["_kuiva_atom_basis"] = atom_basis
    return mol


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
    if fitting in (None, "conventional", "cholesky"):
        return "conventional", None
    raise ValueError("unknown fitting route {!r}; expected 'conventional', 'cholesky', 'df' "
                     "or None".format(fitting))


#: Cholesky vectors per AO function, for **pre-flight estimation only**. Measured at
#: the default 1e-6 threshold: 5.6 for Ne and 7.4 for TiCl3, both in x2c-SVPall-2c. Set to 8
#: because this is one of the few numbers that must err *high*: it multiplies the three-index
#: MO array, and a pre-flight that under-estimates lets a calculation start that cannot
#: finish. It is replaced by the true count as soon as the decomposition has run, so the
#: pessimism lasts only until then.
CHOLESKY_VECTORS_PER_AO = 8.0


def memory_plan(nao: int, *, conventional: bool = True, naux: Optional[int] = None,
                nspinor: Optional[int] = None, n_active: Optional[int] = None,
                nevpt2: bool = False, screening: bool = False) -> list:
    """Phase-by-phase memory estimate for a calculation on ``nao`` AO functions.

    Everything here is a function of dimensions only — no array exists yet — which is what
    lets the whole plan be printed and judged before the SCF starts. Returns a list of
    :class:`kuiva.util.resources.PhaseEstimate` for :func:`kuiva.util.resources.preflight`.

    ``n_active`` adds the multireference phases when the active space is already known;
    ``nevpt2`` adds the 4-RDM, which is ``n_active^8`` and is by a wide margin the largest
    array in the program (direct contraction was chosen over a cumulant approximation, so
    there is no cheaper route to it). ``screening`` adds the two-electron picture change
    (one four-component atomic solve per unique element).
    """
    from ..amf.correction import correction_memory_gb
    from ..integrals.transform import (factor_memory_gb, mo_block_memory_gb,
                                       transform_buffer_gb)

    estimated_naux = naux is None
    naux = int(naux if naux is not None else CHOLESKY_VECTORS_PER_AO * nao)
    n = int(nspinor if nspinor is not None else 2 * nao)
    phases = [res.PhaseEstimate(
        name="scalar X2C SCF", governed=False,
        external_note="allocated by PySCF, not accounted for here; it is given "
                      "mol.max_memory = the rest of the limit")]

    ints = res.PhaseEstimate(name="two-electron integrals", advice=[
        "the Cholesky route needs the conventional ERI array; density fitting with an "
        "explicit auxiliary basis does not "])
    if conventional:
        ints.allocations.append(res.PlannedAllocation(
            "conventional AO ERI array", eri_memory_gb(nao),
            note="nao = {}; grows as nao^4".format(nao)))
    ints.allocations.append(res.PlannedAllocation(
        "three-index AO factors", factor_memory_gb(nao, naux),
        note="naux {} {}".format("~" if estimated_naux else "=", naux)))
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

    phases.append(res.PhaseEstimate(name="spinor MO transform", allocations=[
        res.PlannedAllocation("three-index MO integrals B^P_pq",
                              mo_block_memory_gb(naux, n, n),
                              note="{} spinors".format(n)),
        # What the kernel would use unblocked, capped by what it is allowed: blocking means
        # it never needs more than either. Planning for the cap alone is how a memory plan
        # becomes pessimistic enough to refuse calculations that would have run.
        res.PlannedAllocation("transform buffers",
                              min(transform_buffer_gb(nao, n, naux),
                                  res.BUDGET.transient_gb()), resident=False),
    ], advice=["transform only the orbital blocks that are needed rather than all spinors"]))

    if n_active:
        phases.append(res.PhaseEstimate(name="active-space integrals", allocations=[
            res.PlannedAllocation("active four-index integrals",
                                  res.array_gb((n_active,) * 4),
                                  note="{} active spinors".format(n_active)),
            res.PlannedAllocation("2-RDM", res.rdm_gb(n_active, 2)),
        ], advice=["reduce the active space"]))
    if nevpt2 and n_active:
        phases.append(res.PhaseEstimate(name="SC-NEVPT2", allocations=[
            res.PlannedAllocation("4-RDM", res.rdm_gb(n_active, 4),
                                  note="n_active^8; 6.4 GB at 12 spinors, 382 GB at 20"),
            res.PlannedAllocation("3-RDM", res.rdm_gb(n_active, 3)),
        ], advice=["reduce the active space: the 4-RDM grows as its eighth power"]))
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
                   decoupling_options: Optional[Dict[str, object]] = None,
                   conv_tol: float = 1e-10, max_cycle: int = 200,
                   memory_gb: Optional[float] = None, n_active: Optional[int] = None,
                   gauge_origin=None, verbose: int = 0) -> ScalarX2CData:
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
        ``None`` / ``"conventional"`` / ``"cholesky"`` — ingest conventional ERIs, which
        :mod:`kuiva.integrals.transform` Cholesky-decomposes. **This is the default in every
        case.** ``"df"`` — density fitting; see ``auxbasis``.
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
        ``"breit"``), ``configuration`` (a ``{symbol: reference}`` mapping for a heteronuclear
        molecule), ``backend``, ``uncontract``. See
        :func:`kuiva.amf.correction.amf_correction`.
    memory_gb : float, optional
        Working-memory limit for this calculation, overriding the configured default.
        Without a default *and* without this, the calculation refuses to start.
    n_active : int, optional
        Active-space size, if it is already known. Only used to extend the memory pre-flight
        to the multireference phases — the earlier a size is known, the earlier a calculation
        that cannot fit is stopped.
    gauge_origin : optional
        Gauge origin for the orbital angular momentum of the property dump: ``"mass"``
        (the default centre of mass), ``"charge"``, ``"origin"``, or three coordinates in
        bohr. It is ingested here because the multireference layer never calls PySCF again
, and it is recorded in the dump header because ``L`` is defined relative to it.
    """
    from ..x2c.methods import DEFAULT_METHOD, resolve

    # ⚠ Resolved once, here, and everything downstream reads the resolved pair. A method name
    # and an explicit axis that contradict each other are refused rather than reconciled
    #, so no calculation can run with a Hamiltonian nobody asked for.
    if method is None and x2c_approx is None and screening is None:
        method = DEFAULT_METHOD
    chosen = resolve(method, decoupling=x2c_approx, screening=screening)

    mol = build_mole(molecule, verbose=verbose)
    atom_basis = mol.__dict__["_kuiva_atom_basis"]
    meta = mol.__dict__["_kuiva_basis_meta"]
    fit_route, aux = _choose_fit(atom_basis, fitting, auxbasis)

    # The AO count is the first thing that is known and it fixes every large array of
    # the front-end, so the whole plan is printed and judged here, before the SCF.
    res.ensure_configured(memory_gb)
    res.preflight(memory_plan(mol.nao, conventional=fit_route == "conventional",
                              n_active=n_active,
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
        # may call PySCF — and the gauge origin has to be fixed before the Mole is gone.
        props = ingest_property_integrals(mol, gauge_origin)

        eri = None
        df_cderi = None
        if fit_route == "df":
            # DF factors, packed over the lower-triangular AO pair index.
            df_cderi = np.asarray(mf.with_df._cderi)
        else:
            eri = mol.intor("int2e", aosym="s8")

    soc = (ingest_spin_orbit(mol, h_x2c, chosen.decoupling, screening=chosen.screening,
                             decoupling_options=decoupling_options,
                             **(screening_options or {})) if with_soc else None)

    # Standard output block: the front-end's contribution to the output file.
    rows = [
        ("SCF reference", ref_name.upper() + (" (unrestricted)" if unrestricted else "")),
        ("SCF one-electron Hamiltonian", "X2C (sfx2c1e, spin-free)"),
        ("AO basis functions", int(mol.nao)),
        ("molecular orbitals", nmo, "", "2 spin sets" if unrestricted else ""),
        ("electrons (alpha, beta)", "{}, {}".format(*mol.nelec)),
        ("two-electron route", fit_route + (" [{}]".format(aux) if aux else "")),
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

    na, nb = mol.nelec
    data = ScalarX2CData(
        nao=mol.nao, nmo=nmo, nelec=(int(na), int(nb)),
        e_scf=float(e_scf), converged=bool(mf.converged),
        s_ao=np.ascontiguousarray(s_ao), h_x2c=np.ascontiguousarray(h_x2c),
        mo_coeff=np.ascontiguousarray(mo_coeff),
        mo_energy=np.ascontiguousarray(mf.mo_energy),
        mo_occ=np.ascontiguousarray(mf.mo_occ),
        fit_route=fit_route, eri=eri, df_cderi=df_cderi,
        aux_name=aux if isinstance(aux, str) else ("custom" if aux is not None else None),
        soc=soc, reference=ref_name, unrestricted=unrestricted, s2_deviation=s2_dev,
        basis_meta=meta, e_nuc=float(mol.energy_nuc()), ao_layout=ao_layout(mol),
        properties=props,
    )
    return data


__all__ = ["ScalarX2CData", "SpinOrbitX2C", "PropertyIntegrals", "build_mole",
           "run_scalar_x2c", "ingest_spin_orbit", "ingest_property_integrals",
           "gauge_origin_for", "eri_memory_gb", "ao_layout"]
