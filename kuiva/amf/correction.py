"""Top-level entry point for the atomic mean-field correction.

:func:`amf_correction` is the seam between the front-end and everything four-component. A
caller passes a molecule and a method name and gets back ``(delta h_sf, delta w)`` in the
molecular AO basis, in the conventions of kuiva/spinor/expand.py — ready to be added to what
``pyscf_bridge.ingest_spin_orbit`` already produces. Nothing above this line knows that a
four-component atomic calculation happened.

Scope note — X2CAMF is the atomic approximation to X2C-mmf
----------------------------------------------------------
X2CAMF does the mean field **atomically**, at negligible cost (0.3 s for Ne, 2.5 s for Ar, once
per element per run and cacheable across an entire potential-energy surface). **X2C-mmf** does
the same subtraction on the whole molecule, at the cost of a full four-component molecular SCF,
and lives in :func:`kuiva.interface.pyscf_bridge.molecular_mean_field`. It is
an **experimental benchmark method only** — never a default, never for production, and it warns
at the point of selection; it exists to say what the atomic approximation here is worth.

⚠ **The two share one implementation of the subtraction**
(:func:`kuiva.x2c.mean_field.mean_field_picture_change`) and are **mutually exclusive values of
the same option**, because applying both would double-count the two-electron picture change
exactly. On a closed-shell atom they solve the identical four-component problem and agree to
1e-13 Eh, which is what lets this module's committed reference numbers validate that path too.

Methods
-------
``"x2camf"``
    The in-house atomic mean field of :mod:`kuiva.amf.decouple`. **The default**, because the error it removes — j-splittings 5-30% too large — is larger than the
    cross-code tolerance band that would have to catch it. It costs one four-component atomic SCF per unique
    element, cached in-process and on disk and independent of geometry.
``"none"``
    Exact zeros. The Hamiltonian is bitwise the untouched one-electron X2C operator, and the
    provenance record says so. The right choice wherever spin-orbit coupling is not the
    subject, since the correction changes no scalar quantity and is pure cost there.
``"x2camf-external"``
    The same correction from the external ``x2camf`` plugin — the authors' own implementation
    (:mod:`kuiva.amf.x2camf_plugin`). Import-gated and **never default**. It
    exists so a disagreement between Kuiva and DIRAC can be bisected against a third
    implementation of the *same* method rather than argued about between two, and it is not
    a drop-in equivalent: it always takes the **neutral** atom as its
    reference, and it cannot decouple in a contracted basis. All three are refused explicitly
    rather than silently absorbed; see that module for the measured consequences of each.

⚠ **What is deliberately absent, and what would bring it back.** Three alternatives were
considered and *rejected*, not left unimplemented; reversing any of them needs a new decision,
not a judgement call.

* **SNSO / Boettger screening factors** — empirical, with parameters fitted to neutral atoms and
  then applied to ions in ligand fields, which is precisely the single-molecule-magnet regime
  this program targets, and no systematic route to improving them.
* **An in-house molecular SOMF** — its unique contribution is multi-centre two-electron spin-orbit
  coupling, out of regime for 5d/4f/5f targets where the metal-centred term dominates, and its
  integral classes are used by nothing else.
* **An in-house Breit-Pauli AMFI** — redundant: the light-atom check is better served by the NIST
  fine-structure series, cross-code comparability by the installed OpenMolcas and DIRAC, and the
  atomic-vs-in-situ density question *within* X2CAMF by varying the reference configuration.
  ⚠ Three conditions would make it worth revisiting, named so the decision does not become drift:
  (1) a cross-code discrepancy appears that OpenMolcas/DIRAC comparison cannot attribute, so an
  internal like-for-like with exactly one variable changed becomes necessary; (2) a target system
  falls outside the atomic approximation's regime (light-element delocalized spin actually
  carrying relevant spin-orbit coupling); (3) average-of-configuration Dirac-Hartree-Fock proves
  unreliable for a class of ions that are needed.

The asymmetry in ``X``, stated rather than hidden
-------------------------------------------------------------
Kuiva's one-electron path uses PySCF's exact **molecular** one-electron X2C
(``x2c_approx="1e"``), while the AMF correction is atomic by construction — its ``X`` and
``R`` come from an isolated-atom one-electron problem. This is standard for X2CAMF and is not
a bug: the atomic approximation is applied to the *two-electron* picture change, which is the
term that would otherwise be missing entirely, while the one-electron part stays exact. It is
recorded in the provenance so that a comparison against another program is never ambiguous
about it, and ``x2c_approx="atom1e"`` makes the two consistent for anyone who wants to measure
the difference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..spinor.expand import is_time_reversal_even, two_component_operator
from ..util import output as out
from ..util import resources as res
from ..util.logging import get_logger
from ..util.timing import timer
from .atomic import atomic_correction, elements_by_label, nuclear_model_of
from .backend import INTERACTIONS

log = get_logger(__name__)

#: Absolute tolerance [dimensionless] on the per-atom AO-ordering check of
#: :func:`_check_atom_ordering`. Overlap matrix elements are O(1) and the two matrices are
#: built from the *same* parsed basis by the same integral engine, so the measured difference
#: is **exactly** zero on every supported basis family tried; the tolerance exists only so that a future
#: normalization convention change fails with a diagnosis rather than at the 17th digit.
AO_ORDERING_TOLERANCE = 1e-12


def correction_memory_gb(nao: int) -> float:
    """Size [GB] of an assembled molecular correction (exact sizing function).

    ``h_sf`` is ``nao^2`` reals and ``w`` is ``3 nao^2``, so four real ``nao x nao`` matrices —
    the same footprint as the :class:`~kuiva.interface.pyscf_bridge.SpinOrbitX2C` it is added
    to, and half what the assembled two-component operator would cost. Small next to the
    integrals, but the resource-accounting rule has no exceptions.
    """
    return res.array_gb((4, nao, nao), np.float64)


#: The correction methods :func:`amf_correction` accepts. ``"x2camf-external"`` is the
#: optional plugin path: import-gated, never default, and never selected on
#: Kuiva's behalf — it is a bisection tool, not a fallback.
METHODS = ("none", "x2camf", "x2camf-external")


@dataclass(frozen=True)
class AMFCorrection:
    """A two-electron picture-change correction, with the provenance to interpret it.

    ``h_sf`` and ``w`` add directly to the corresponding members of
    :class:`kuiva.interface.pyscf_bridge.SpinOrbitX2C`, in the same basis and the same spin-blocked
    conventions.

    Attributes
    ----------
    h_sf : ndarray (nao, nao)
        Spin-free part of the correction, real symmetric.
    w : ndarray (3, nao, nao)
        Spin-orbit part, real antisymmetric, ``W_k = i w_k``.
    method : str
        One of :data:`METHODS`.
    interaction : str
        Two-electron interaction of the atomic reference: ``"coulomb"``, ``"gaunt"`` or
        ``"breit"``. Meaningless for ``method="none"`` and recorded as ``"none"`` there,
        because "Coulomb" would imply something was computed.
    backend, backend_version : str
        Which four-component implementation produced the atomic references.
    configurations : dict
        ``{atom label: canonical configuration}`` — the atomic states the mean field was taken
        over, in the canonical per-``l`` form (``"s6 p12 d1"``). Keyed by the *labelled* symbol
        (``"Ti1"``), because that is the granularity at which they can differ. For an open
        shell this is a real choice that changes the correction, so it is recorded and not
        summarized.
    elements : tuple of str
        The distinct atom labels the correction covers, in the order the molecule lists them.
        ⚠ Its length is **not** the number of four-component solves once hydrogen is in the
        molecule — a one-electron reference contributes an exactly zero block and runs no SCF.
        A caching test should read ``atomic.cache_statistics()["solves"]``, which is a call
        count rather than a timing.
    light_speed : float or None
        Non-``None`` only when the correction was built at a modified speed of light, i.e. by
        the non-relativistic-limit test. A record that must never be lost: a correction
        computed at ``c = 1e6`` is numerically fine and physically meaningless.
    """

    h_sf: np.ndarray
    w: np.ndarray
    method: str = "none"
    interaction: str = "none"
    backend: str = ""
    backend_version: str = ""
    configurations: Dict[str, str] = field(default_factory=dict)
    elements: Tuple[str, ...] = ()
    light_speed: Optional[float] = None
    #: ``max |dh_sf|`` and ``max |dw|`` [Eh]. Reported separately, never summed: the
    #: two-electron *scalar* picture change is a distinguishing feature of this method that
    #: Breit-Pauli AMFI and SNSO do not capture at all, and it is typically an
    #: order of magnitude larger than the spin-orbit part.
    spin_free_scale: float = 0.0
    spin_orbit_scale: float = 0.0
    #: Absolute and relative size of the time-reversal-odd part projected out of the assembled
    #: correction.
    tr_residual: float = 0.0
    tr_residual_rel: float = 0.0

    @property
    def nao(self) -> int:
        return int(self.h_sf.shape[0])

    @property
    def is_zero(self) -> bool:
        """True when the correction is exactly zero — not merely small."""
        return not (self.h_sf.any() or self.w.any())

    def hamiltonian(self) -> np.ndarray:
        """The assembled ``(2*nao, 2*nao)`` two-component correction."""
        return two_component_operator(self.h_sf, self.w)

    def double_counting(self, dm: np.ndarray) -> float:
        """``1/2 Tr(D dG)`` [Eh] — what a **total** energy over-counts if ``dG`` went into
        ``hcore``.

        ⚠ **This is the standard frozen-mean-field trap and it is a factor of two in the
        total.** ``dG`` is a two-electron mean field, so it enters the **Fock operator whole**
        and the **energy with a half**. Putting it into ``hcore`` — which is the method as
        published, and what Kuiva's front-end does — therefore gives every relative state
        energy and every spin-orbit splitting correctly while inflating the absolute total by
        this number. Subtract it before comparing a total against a four-component one.

        Kuiva itself never reports such a total: the scalar SCF runs on ``sfx2c1e`` and
        is untouched by this correction, and the multireference energies downstream are
        expectation values of the corrected Hamiltonian, i.e. exactly the quantity that is
        right. The helper exists so that the accounting is written down in the library rather
        than rediscovered in a test file each time somebody wants an absolute number.

        Parameters
        ----------
        dm : ndarray ``(2*nao, 2*nao)``
            The two-component density matrix the energy was evaluated at, in the
            spin-blocked basis.
        """
        return mean_field_double_counting(self.hamiltonian(), dm)

    def provenance(self) -> "ScreeningRecord":
        """The array-free provenance record (see :class:`ScreeningRecord`)."""
        return ScreeningRecord(
            method=self.method, interaction=self.interaction, backend=self.backend,
            backend_version=self.backend_version,
            configurations=dict(self.configurations), elements=tuple(self.elements),
            light_speed=self.light_speed, spin_free_scale=self.spin_free_scale,
            spin_orbit_scale=self.spin_orbit_scale, tr_residual=self.tr_residual,
            tr_residual_rel=self.tr_residual_rel)

    def relative_to(self, soc) -> Tuple[float, float]:
        """``(spin-free, spin-orbit)`` size of the correction relative to a
        :class:`~kuiva.interface.pyscf_bridge.SpinOrbitX2C`'s one-electron parts.

        The spin-orbit ratio is the one that matters for the cross-code validation: a one-electron
        operator overestimates atomic j-splittings by 15-30% (measured: Ne +30.0%, Ar +18.7%
        against four-component Dirac-Coulomb), so a correction of a few percent of ``max |w|``
        in the valence region is the expected magnitude, and one of a few *per mille* or of
        tens of percent both mean something is wrong.
        """
        sf = float(np.max(np.abs(soc.h_sf))) or 1.0
        so = float(np.max(np.abs(soc.w))) or 1.0
        return self.spin_free_scale / sf, self.spin_orbit_scale / so

    def report(self, logger=None) -> None:
        """The provenance output block. Printed wherever the Hamiltonian is described, so a
        stored property matrix is never ambiguous about which Hamiltonian produced it.

        One implementation, in :meth:`ScreeningRecord.report`, so that the block printed by
        the front-end (which has only the record) and the one printed here cannot drift.
        """
        self.provenance().report(logger)

    def __repr__(self) -> str:
        return "AMFCorrection(nao={}, method={}, interaction={}, max|dh_sf|={:.3e}, " \
               "max|dw|={:.3e} Eh)".format(self.nao, self.method, self.interaction,
                                           self.spin_free_scale, self.spin_orbit_scale)


def mean_field_double_counting(dg_2c: np.ndarray, dm: np.ndarray) -> float:
    """``1/2 Tr(dG D)`` [Eh] — see :meth:`AMFCorrection.double_counting` for what it is for.

    Takes the raw ``(2*nao, 2*nao)`` correction rather than an :class:`AMFCorrection`, so the
    same accounting serves the validation path (which builds ``dG`` directly from
    :mod:`kuiva.amf.decouple` and never assembles a molecular correction) and the production
    one. One implementation, because two would eventually differ by the factor of two this
    function exists to get right.
    """
    dg_2c = np.asarray(dg_2c)
    dm = np.asarray(dm)
    if dm.shape != dg_2c.shape:
        raise ValueError("expected a {} two-component density, got {}".format(
            dg_2c.shape, dm.shape))
    return 0.5 * float(np.real(np.einsum("ij,ji->", dg_2c, dm)))


@dataclass(frozen=True)
class ScreeningRecord:
    """What two-electron spin-orbit treatment produced a Hamiltonian — without the arrays.

    :class:`AMFCorrection` carries the same fields *and* the correction itself; this carries
    only the record, and it is what lives on
    :attr:`kuiva.interface.pyscf_bridge.SpinOrbitX2C.screening` once the correction has been
    added in. Separating the two is deliberate: keeping the whole ``AMFCorrection`` there
    would hold ``4 nao^2`` doubles alive for the life of the calculation to describe arrays
    that have already been summed into ``h_sf`` and ``w``, and would invite a caller to add
    the correction a second time.

    ``ScreeningRecord()`` — all defaults — is the honest description of an uncorrected
    one-electron X2C Hamiltonian, and :attr:`applied` is ``False`` for it.

    ⚠ This record is a **contract with stored data**: it goes
    into the property dump's header and into any committed reference record, so that a
    stored property matrix is never ambiguous about which Hamiltonian produced it. Add fields;
    do not rename or repurpose them.
    """

    method: str = "none"
    interaction: str = "none"
    backend: str = ""
    backend_version: str = ""
    configurations: Dict[str, str] = field(default_factory=dict)
    elements: Tuple[str, ...] = ()
    light_speed: Optional[float] = None
    spin_free_scale: float = 0.0
    spin_orbit_scale: float = 0.0
    tr_residual: float = 0.0
    tr_residual_rel: float = 0.0

    @property
    def applied(self) -> bool:
        """Whether a two-electron picture change was actually applied."""
        return self.method != "none"

    def as_dict(self) -> Dict[str, object]:
        """A JSON-serializable form, for the property dump header and reference records.

        Plain builtins only — no NumPy scalars, no tuples-as-keys — so it round-trips through
        ``json`` unchanged and a stored record can be compared field by field.
        """
        return {
            "method": self.method,
            "interaction": self.interaction,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "configurations": {str(k): str(v) for k, v in sorted(self.configurations.items())},
            "elements": list(self.elements),
            "light_speed": None if self.light_speed is None else float(self.light_speed),
            "spin_free_scale": float(self.spin_free_scale),
            "spin_orbit_scale": float(self.spin_orbit_scale),
            "tr_residual": float(self.tr_residual),
            "tr_residual_rel": float(self.tr_residual_rel),
        }

    def report(self, logger=None) -> None:
        """The provenance output block, printed wherever the Hamiltonian is described."""
        logger = logger or log
        if not self.applied:
            out.entry(logger, "two-electron SOC picture change", "none",
                      note="one-electron X2C only; splittings 5-30% too large")
            return
        out.entries(logger, [
            ("two-electron SOC picture change", self.method),
            ("atomic reference interaction", self.interaction),
            ("atomic reference backend", "{} {}".format(self.backend,
                                                        self.backend_version).strip()),
            ("elements corrected", ", ".join(self.elements) or "-"),
            ("reference configurations", ", ".join(
                "{}: {}".format(k, v) for k, v in sorted(self.configurations.items())) or "-"),
            ("spin-free correction, max |dh_sf|", self.spin_free_scale, "Eh", "", "{:.6e}"),
            ("spin-orbit correction, max |dw|", self.spin_orbit_scale, "Eh", "", "{:.6e}"),
            ("discarded time-reversal-odd part", self.tr_residual, "Eh",
             "{:.1e} relative".format(self.tr_residual_rel), "{:.3e}"),
        ])
        if self.light_speed is not None:
            logger.warning("this Hamiltonian was corrected at a NON-PHYSICAL speed of light "
                           "(c = %.6e a.u.). It is a numerical experiment, not a "
                           "calculation.", self.light_speed)

    def __str__(self) -> str:
        return self.method


def zero_correction(nao: int) -> AMFCorrection:
    """An exactly-zero correction over ``nao`` scalar basis functions.

    Exact zeros, not small numbers: ``method="none"`` must leave the Hamiltonian **bitwise**
    what it was before this module existed, and a test asserts that.
    """
    return AMFCorrection(h_sf=np.zeros((nao, nao)), w=np.zeros((3, nao, nao)))


def validate_correction(h_sf: np.ndarray, w: np.ndarray, *, tol: float = 1e-12,
                        what: str = "atomic mean-field correction") -> None:
    """Assert the structural invariants of the two-component conventions **on the correction itself**.

    Real symmetric ``h_sf``, real antisymmetric ``w``, and a time-reversal-even assembled
    operator. These are asserted on the correction and not only on the final Hamiltonian,
    for a reason worth stating: a correction that broke time-reversal symmetry would be
    invisible in the total, because the one-electron part is large and its own residual is
    projected out at ingestion — the corrupted Kramers splitting would appear later, in the
    CI, as a physical-looking near-degeneracy that nothing points back to here.

    Raises ``ValueError`` on failure. This is a programming-error check, not a numerical
    tolerance to tune: every one of these properties holds by construction of
    :func:`kuiva.spinor.expand.decompose_two_component`, so a failure means an array was built
    or transformed somewhere that bypassed it.
    """
    h_sf = np.asarray(h_sf)
    w = np.asarray(w)
    if w.shape != (3,) + h_sf.shape:
        raise ValueError("{}: w must have shape (3, {}, {}) to match h_sf, got {}".format(
            what, h_sf.shape[0], h_sf.shape[0], w.shape))
    if np.iscomplexobj(h_sf) and float(np.max(np.abs(h_sf.imag))) > tol:
        raise ValueError("{}: the spin-free part is not real (max |Im| = {:.2e})".format(
            what, float(np.max(np.abs(h_sf.imag)))))
    if np.iscomplexobj(w) and float(np.max(np.abs(w.imag))) > tol:
        raise ValueError("{}: the spin-orbit factors are not real (max |Im| = {:.2e})".format(
            what, float(np.max(np.abs(w.imag)))))
    asym = float(np.max(np.abs(h_sf - h_sf.T))) if h_sf.size else 0.0
    if asym > tol:
        raise ValueError("{}: the spin-free part is not symmetric (max |A - A^T| = "
                         "{:.2e})".format(what, asym))
    sym = float(np.max(np.abs(w + np.transpose(w, (0, 2, 1))))) if w.size else 0.0
    if sym > tol:
        raise ValueError("{}: the spin-orbit factors are not antisymmetric "
                         "(max |w + w^T| = {:.2e})".format(what, sym))
    if not is_time_reversal_even(two_component_operator(h_sf, w), tol=max(tol, 1e-12)):
        raise ValueError("{}: the assembled correction is not time-reversal even. This "
                         "cannot happen for real symmetric h_sf and real antisymmetric w, so "
                         "one of the arrays was not produced by "
                         "spinor.expand.decompose_two_component.".format(what))


def amf_correction(mol, *, method: str = "x2camf", interaction: str = "coulomb",
                   backend: str = "pyscf", configuration: Optional[str] = None,
                   light_speed: Optional[float] = None, uncontract: bool = True,
                   report: bool = False, **solver_kwargs) -> AMFCorrection:
    """The two-electron picture-change correction for a molecule.

    Parameters
    ----------
    mol : PySCF ``Mole``
        The molecule whose AO basis the correction must be expressed in. Same argument as
        :func:`kuiva.interface.pyscf_bridge.ingest_spin_orbit` takes, deliberately.
    method : str
        ``"x2camf"`` (**the default**) or ``"none"`` (exact zeros, and a
        Hamiltonian left bitwise as it was). ``"x2camf-external"`` routes to the reference
        implementation and is a bisection tool, never a default.
    interaction : str
        Two-electron interaction of the four-component atomic reference: ``"coulomb"``,
        ``"gaunt"`` (adds spin-other-orbit) or ``"breit"``.
    backend : str
        Four-component backend name; see :func:`kuiva.amf.backend.available_backends`.
    configuration : mapping, AtomicConfiguration, str, int or None
        The atomic reference the mean field is taken over. Accepts a configuration
        (``"[Ar]3d1"``, ``"[Xe]4f9"``), an **oxidation state** (``"+4"``, ``"6+"``, ``4``), or
        the object itself. ``None`` takes :func:`kuiva.amf.configuration.default_configuration`
        for each element — the **trivalent ion** for the f block, the neutral atom elsewhere.

        For a molecule with more than one element it must be a **mapping**,
        ``{symbol: configuration}``, keyed by the atom label (``"Ti1"``) or the element
        (``"Ti"``); elements it omits take their default. A single value applied to every
        element of a heteronuclear molecule is refused rather than obeyed, because
        ``configuration="+3"`` almost always means "the metal is trivalent" and would
        otherwise silently strip three electrons off every ligand atom as well.

        ⚠ The +3 f-block default is a default, not a claim: thorium is usually tetravalent,
        uranium and plutonium high-valent, and the late actinides such as nobelium more stable
        divalent, so ``configuration="+4"`` and friends exist precisely to override it per
        element. An open shell is occupied fractionally over the whole frontier ``l`` shell
        (average of configuration), so the atomic mean field stays spherical whichever
        reference is chosen.

        ⚠ ``mol.charge`` is deliberately **not** consulted. The atomic mean field is a
        property of an element, and a molecular charge belongs to no single atom of it.
    light_speed : float, optional
        Override the speed of light throughout. **Only** for the non-relativistic-limit test
        of the subtraction; a correction built this way is marked as such and warns when reported.
    uncontract : bool
        Do the atomic four-component solve and the X2C decoupling in the fully decontracted
        basis, then contract the correction back (decoupling belongs in the
        primitive basis). The default and the physically correct choice.

    ⚠ **There is deliberately no nuclear-model parameter.** The atomic references are solved
    over whichever nucleus ``mol`` was built with, read off the molecule itself
    (:func:`kuiva.amf.atomic.nuclear_model_of`): a mean field over a different nucleus from
    the integrals it corrects would be Hermitian, of plausible magnitude and wrong, and an
    argument is something a caller can fail to pass. A ``mol`` that mixes models is refused.

    Returns
    -------
    AMFCorrection over ``mol.nao`` scalar basis functions.

    Notes
    -----
    **Molecular assembly** is placing each element's block on the diagonal of
    the molecular AO matrix: the correction is atom-diagonal by construction, so
    off-atom blocks are never computed, only never written — and that is asserted on the
    result rather than assumed. One four-component solve happens per **unique element**, not
    per atom, which is what makes the correction affordable for a dimer or a cluster and what
    ``kuiva.amf.atomic.cache_statistics()["solves"]`` is there to verify.

    The two things assembly can silently get wrong are both checked. That each atomic block
    lands on the right AO range is checked against the **isolated-atom overlap matrix**, not
    against a shape: two bases can agree on how many functions an atom has and disagree on
    their order, and that difference is invisible to every norm-based test while being fatal
    (the failure mode :mod:`kuiva.amf.x2camf_plugin` guards the same way). That the off-atom
    blocks are zero is checked by counting non-zeros, exactly, not to a tolerance.
    """
    if method not in METHODS:
        raise ValueError("unknown two-electron picture-change method {!r}; expected one of "
                         "{}".format(method, METHODS))
    if interaction not in INTERACTIONS:
        raise ValueError("unknown two-electron interaction {!r}; expected one of {}".format(
            interaction, INTERACTIONS))

    nao = int(mol.nao)
    if method == "none":
        return zero_correction(nao)

    if getattr(mol, "cart", False):
        # ⚠ Structural, not a limitation that could be lifted by more code. PySCF's
        # four-component solver works in the j-adapted 2-spinor basis, which is built on
        # *spherical* harmonics — ``Mole.nao_2c()`` of a Cartesian molecule is 2x its
        # spherical nao, not its Cartesian one. The correction would then be computed over a
        # different set of functions from the one it has to be added to. For l <= 1 the two
        # bases coincide, so the shape check further down would not catch it; refusing here
        # is what makes the failure independent of which angular momenta happen to be present.
        raise NotImplementedError(
            "the atomic mean-field correction requires a spherical (solid-harmonic) AO "
            "basis; this molecule was built with cart=True. The four-component spinor basis "
            "is spherical by construction, so the correction would be expressed over "
            "different functions from the Hamiltonian it corrects.")
    if mol.has_ecp():
        raise NotImplementedError(
            "the atomic mean-field correction requires an all-electron basis; this molecule "
            "uses an ECP, and X2C has no meaning with a pseudopotential core. The supported "
            "basis families are all-electron by design.")

    if mol.nelectron == 1:
        # ⚠ A one-electron atom has no two-electron mean field, so it has no picture change of
        # one either, and the correction is exactly zero *by definition* — not by numerical
        # accident. This is the standard Hartree-Fock convention for a one-electron system
        # (PySCF's own ``scf.DHF`` dispatches to ``HF1e``, which sets ``vhf = 0``), and it is
        # applied to **both** halves of the subtraction, which is the part that
        # matters. Computing it instead would picture-change a Hartree-Fock self-interaction —
        # an artefact of the method rather than a physical screening — and produce a small
        # correction where the right answer is none.
        #
        # This is a definition, so it cannot on its own test the subtraction. The test that
        # does is the structural one: a solution with a vanishing mean field must give an
        # exactly vanishing correction, and that is asserted separately on the machinery
        # itself (tests/test_amf_decouple.py) rather than routed around here.
        log.debug("one-electron system: the atomic mean-field correction is zero by "
                  "definition (no second electron to screen)")
        return AMFCorrection(
            h_sf=np.zeros((nao, nao)), w=np.zeros((3, nao, nao)),
            method=method, interaction=interaction, backend=backend,
            elements=tuple(sorted({mol.atom_symbol(ia) for ia in range(mol.natm)})),
            light_speed=light_speed)

    if method == "x2camf-external":
        return _external_correction(mol, interaction=interaction,
                                    configuration=configuration, light_speed=light_speed,
                                    uncontract=uncontract, report=report, **solver_kwargs)

    # ⚠ The molecule's charge is deliberately **not** passed through as the atomic reference's
    # charge. The atomic mean field is a property of an element, and a molecular charge
    # belongs to no single atom of it — so the reference is the element's default unless a
    # configuration says otherwise. For an isolated ion that is a
    # real choice with a measurable effect, which is why it is a named parameter and not an
    # inference from ``mol.charge``.
    labels = elements_by_label(mol)
    configurations = _configuration_map(configuration, labels)
    # ⚠ **Taken from the molecule, never from an argument.** The atomic mean field must be
    # solved over the same nucleus the molecular integrals were evaluated over, and a
    # parameter is something a caller can fail to pass: the two would then differ by a
    # Hermitian, plausible, wrong term concentrated at the nucleus, which is exactly where
    # this correction lives. Reading it off ``mol`` makes the agreement structural.
    nuclear_model = nuclear_model_of(mol)

    blocks = {}
    with timer("atomic mean-field correction"):
        for label, (symbol, basis) in labels.items():
            block = atomic_correction(symbol, basis, configuration=configurations[label],
                                      interaction=interaction, backend=backend,
                                      light_speed=light_speed, uncontract=uncontract,
                                      nuclear_model=nuclear_model,
                                      report=report, **solver_kwargs)
            validate_correction(block.h_sf, block.w,
                                what="X2CAMF correction for {}".format(label))
            blocks[label] = block
        h_sf, w = _assemble(mol, labels, blocks)

    # Validate the **assembled** correction and not only each block: the structural
    # invariants are precisely what a block-placement bug would break, and this is cheap.
    validate_correction(h_sf, w, what="assembled X2CAMF correction")

    from .backend import get_backend
    impl = get_backend(backend)
    correction = AMFCorrection(
        h_sf=h_sf, w=w, method=method, interaction=interaction,
        backend=backend, backend_version=impl.version,
        configurations={label: blocks[label].configuration.canonical for label in labels},
        elements=tuple(labels), light_speed=light_speed,
        spin_free_scale=float(np.max(np.abs(h_sf))) if h_sf.size else 0.0,
        spin_orbit_scale=float(np.max(np.abs(w))) if w.size else 0.0,
        # The residual of the *worst* block, not their sum: it is a diagnostic of the
        # conditioning of one atomic decoupling, and averaging it over a molecule would let a
        # broken heavy element hide behind a dozen healthy ligand atoms.
        tr_residual=max((blocks[label].tr_residual for label in labels), default=0.0),
        tr_residual_rel=max((blocks[label].tr_residual_rel for label in labels), default=0.0))
    if report:
        correction.report()
    return correction


def _configuration_map(configuration, labels: Dict[str, Tuple[str, object]]
                       ) -> Dict[str, object]:
    """``{atom label: configuration}`` from whatever the caller supplied.

    A mapping is looked up by label first and then by element symbol, so ``{"Ti": "+3"}``
    covers ``Ti1`` and ``Ti2`` while ``{"Ti1": ...}`` can still separate them. Anything else
    is a single reference; see :func:`amf_correction` for why that is refused when the
    molecule has more than one element.
    """
    try:
        from collections.abc import Mapping
    except ImportError:                                                   # pragma: no cover
        from collections import Mapping                                   # type: ignore

    if isinstance(configuration, Mapping):
        wanted = {str(k).capitalize(): v for k, v in configuration.items()}
        known = {name.capitalize() for name in labels}
        known |= {symbol.capitalize() for symbol, _ in labels.values()}
        unknown = sorted(set(wanted) - known)
        if unknown:
            raise ValueError(
                "the reference configurations name {} which this molecule does not contain; "
                "it has {}. A silently ignored entry here would mean a calculation ran with "
                "a different atomic reference from the one that was asked for.".format(
                    ", ".join(unknown), ", ".join(sorted(labels))))
        # ``in`` rather than ``get``'s default, so that an explicit ``{"Ti1": None}`` means
        # "Ti1 takes its default" instead of silently falling through to a ``{"Ti": ...}``
        # entry that was meant for its partner.
        resolved = {}
        for label, (symbol, _) in labels.items():
            for key in (label.capitalize(), symbol.capitalize()):
                if key in wanted:
                    resolved[label] = wanted[key]
                    break
            else:
                resolved[label] = None
        return resolved

    elements = {symbol for symbol, _ in labels.values()}
    if configuration is not None and len(elements) > 1:
        raise ValueError(
            "a single reference configuration ({!r}) cannot be applied to a molecule "
            "containing {}. Pass a mapping, e.g. configuration={{'{}': {!r}}} — an oxidation "
            "state stated once would otherwise be applied to every element, stripping "
            "electrons off the ligands as well as the metal.".format(
                configuration, ", ".join(sorted(elements)), sorted(elements)[0],
                configuration))
    return {label: configuration for label in labels}


def _assemble(mol, labels: Dict[str, Tuple[str, object]], blocks
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Place each element's atom-diagonal block into the molecular AO matrix."""
    nao = int(mol.nao)
    res.reserve("assembled atomic mean-field correction", correction_memory_gb(nao),
                note="nao = {}; h_sf and the three w_k".format(nao),
                advice=["use a smaller basis set",
                        "the correction is the same size as the one-electron spin-orbit "
                        "operator it is added to, so a limit that refuses it would refuse "
                        "the Hamiltonian as well"])
    h_sf = np.zeros((nao, nao))
    w = np.zeros((3, nao, nao))

    slices = mol.aoslice_by_atom()
    s_mol = mol.intor("int1e_ovlp")
    verified = set()
    ranges = []
    for ia in range(mol.natm):
        label = mol.atom_symbol(ia)
        if label not in blocks:
            # A ghost: no nucleus, so no two-electron picture change of one. Its diagonal
            # block stays exactly zero, and it is deliberately absent from ``ranges`` so the
            # off-atom check below reads it as what it is.
            continue
        block = blocks[label]
        p0, p1 = int(slices[ia][2]), int(slices[ia][3])
        if block.h_sf.shape[0] != p1 - p0:
            raise RuntimeError(
                "the atomic correction for {} spans {} basis functions where atom {} of the "
                "molecule has {}. The contraction back from the primitive basis did not land "
                "in the molecular AO basis.".format(label, block.h_sf.shape[0], ia, p1 - p0))
        if label not in verified:
            _check_atom_ordering(labels[label], s_mol[p0:p1, p0:p1], label)
            verified.add(label)
        h_sf[p0:p1, p0:p1] = block.h_sf
        w[:, p0:p1, p0:p1] = block.w
        ranges.append((p0, p1))

    _check_off_atom_blocks_are_zero(h_sf, w, ranges)
    return h_sf, w


def _check_atom_ordering(element_and_basis, s_block: np.ndarray, label: str) -> None:
    """Assert that an atom's AO block in the molecule is the isolated atom's own AO basis.

    ⚠ **A shape check is not enough, and this is the same trap
    :func:`kuiva.amf.x2camf_plugin._decontracted` guards.** The atomic correction is computed
    over a ``Mole`` built from one atom and the parsed basis; the molecule builds its shells
    from the same data, so the two orderings *do* agree — but if they ever stopped agreeing,
    a permuted block would be Hermitian, time-reversal even, of exactly the right magnitude
    and completely wrong. The overlap matrix of an atom with itself is geometry-independent,
    so comparing it against the molecule's own diagonal block costs one tiny integral
    evaluation per element and detects a permutation, a normalization difference and a
    silently different basis alike.

    Measured on every supported basis family tried: **exactly 0.0**, not merely below the tolerance.
    """
    from pyscf import gto

    symbol, basis = element_and_basis
    probe = gto.M(atom=[(symbol, (0.0, 0.0, 0.0))], basis={symbol: basis},
                  spin=int(gto.charge(symbol)) % 2, verbose=0)
    diff = float(np.max(np.abs(np.asarray(s_block) - probe.intor("int1e_ovlp"))))
    if diff > AO_ORDERING_TOLERANCE:
        raise RuntimeError(
            "the AO basis of atom {} in the molecule differs from the isolated atom's "
            "(max |dS| = {:.2e} over {} functions) — an ordering or normalization mismatch. "
            "The correction would be placed over different functions from the ones it "
            "corrects.".format(label, diff, probe.nao))


def _check_off_atom_blocks_are_zero(h_sf: np.ndarray, w: np.ndarray, ranges) -> None:
    """Assert that nothing was written outside the atom-diagonal blocks.

    Counted, not compared to a tolerance — the off-atom blocks are never *computed*, only
    never written, so anything but exact zero means a block landed in the wrong place. Done
    by counting non-zeros inside the blocks and over the whole array rather than by masking,
    so no second ``nao^2`` array is allocated to check the first.
    """
    inside = sum(int(np.count_nonzero(h_sf[p0:p1, p0:p1]))
                 + int(np.count_nonzero(w[:, p0:p1, p0:p1])) for p0, p1 in ranges)
    total = int(np.count_nonzero(h_sf)) + int(np.count_nonzero(w))
    if total != inside:
        raise RuntimeError(
            "the assembled atomic mean-field correction has {} non-zero elements outside the "
            "atom-diagonal blocks, where it is atom-diagonal by construction. A block was "
            "placed at the wrong AO offset.".format(total - inside))


def _external_correction(mol, *, interaction: str, configuration, light_speed,
                         uncontract: bool, report: bool,
                         variant: str = "pcc") -> AMFCorrection:
    """``method="x2camf-external"``: the same correction from the plugin.

    Kept in its own function so the import of :mod:`kuiva.amf.x2camf_plugin` — and through it
    of the optional dependency — happens only when the method is actually asked for.

    ⚠ ``variant`` reaches here through ``**solver_kwargs`` and defaults to the *full*
    two-electron picture change. The plugin's own default entry point returns only the
    spin-dependent half, which is a different quantity and not a Hamiltonian correction; see
    :data:`kuiva.amf.x2camf_plugin.VARIANTS`.

    **Molecules are allowed here**. The plugin performs its own
    per-element assembly, so this path is a second implementation of Kuiva's block placement
    rather than an atoms-only bisection tool — which is worth strictly more than refusing, at
    no cost. It stays never-default all the same.
    """
    # ⚠ The plugin runs its own four-component atomic solves and Kuiva has no way to tell it
    # which nucleus to use, so a finite-nucleus molecule is **refused** rather than corrected
    # with a point-nucleus mean field. It exists to bisect a disagreement against a third
    # implementation, and a bisection tool that silently answers a different question is worse
    # than one that is unavailable.
    model = nuclear_model_of(mol)
    if model != "point":
        raise NotImplementedError(
            "screening='x2camf-external' cannot be used with a {} nuclear model: the plugin "
            "solves its own atomic references over a point nucleus, so the correction would "
            "not describe the molecule it is added to. Use the built-in screening='x2camf', "
            "or compare against the plugin with nuclear_model='point'.".format(model))

    import contextlib

    from ..spinor.expand import decompose_two_component, time_reversal_residual
    from .configuration import AtomicConfiguration
    from .pyscf_dhf import light_speed as _light_speed
    from .x2camf_plugin import plugin_correction_matrix, version as plugin_version

    symbols = tuple(sorted({mol.atom_pure_symbol(ia) for ia in range(mol.natm)}))
    context = (_light_speed(light_speed) if light_speed is not None
               else contextlib.nullcontext())
    with timer("atomic mean-field correction (x2camf plugin)"), context:
        dg = plugin_correction_matrix(mol, interaction=interaction, variant=variant,
                                      configuration=configuration, uncontract=uncontract)
    # Measured here rather than taken on trust: the plugin's matrix comes back
    # time-reversal even to 2e-19 for Ne, but "the external code got it right" is exactly
    # the kind of claim this project measures, and the contraction is a second
    # operation that could in principle break it.
    residual, _ = time_reversal_residual(dg)
    scale = float(np.max(np.abs(dg))) if dg.size else 0.0
    h_sf, w = decompose_two_component(dg)
    validate_correction(h_sf, w, what="x2camf plugin correction for {}".format(
        ", ".join(symbols)))
    correction = AMFCorrection(
        h_sf=h_sf, w=w, method="x2camf-external", interaction=interaction,
        backend="x2camf-plugin", backend_version=plugin_version(),
        # The plugin takes no configuration input at all, so the record says what it used
        # rather than what was asked for (kuiva.amf.x2camf_plugin, point 3).
        configurations={s: AtomicConfiguration.ground(s).canonical for s in symbols},
        elements=symbols, light_speed=light_speed,
        spin_free_scale=float(np.max(np.abs(h_sf))) if h_sf.size else 0.0,
        spin_orbit_scale=float(np.max(np.abs(w))) if w.size else 0.0,
        tr_residual=residual, tr_residual_rel=residual / (scale or 1.0))
    if report:
        correction.report()
    return correction


__all__ = ["AMFCorrection", "METHODS", "ScreeningRecord", "amf_correction",
           "correction_memory_gb", "mean_field_double_counting", "validate_correction",
           "zero_correction"]
