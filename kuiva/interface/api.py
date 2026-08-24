"""Primary Python API (input is a Python API, not a text format).

A user constructs a :class:`Molecule` (geometry + per-atom basis by registry name) and calls
:func:`scalar_x2c_reference` to obtain the ingested scalar-relativistic X2C reference that the
multireference layer consumes. This is the single public entry point for the front-end.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..basis import registry as reg
from ..integrals.transform import DEFAULT_CHOLESKY_TOL, ThreeIndexAO
from ..orth.canonical import DEFAULT_THRESHOLD, OrthonormalBasis
from ..orth.project import (DEFAULT_CARRY as _DEFAULT_CARRY,
                            DEFAULT_SCHEME as _DEFAULT_PROJECTION_SCHEME)
from ..spinor.expand import SpinorBasis
from ..util import output as out
from ..util import resources as res
from ..util.logging import get_logger
from .pyscf_bridge import (ScalarX2CData, build_mole, cross_overlap, memory_plan,
                           run_scalar_x2c)

log = get_logger(__name__)

Atom = Tuple[str, Tuple[float, float, float]]
BasisSpec = Union[str, Dict[str, str]]


@dataclass
class Molecule:
    """A molecule: geometry, charge/spin, and a basis assignment by registry name.

    Parameters
    ----------
    atoms : list of ``(symbol, (x, y, z))``
    basis : str or dict
        One registry family name applied to all atoms, or a mapping whose keys are element
        symbols (``"O"``), atom labels (``"O3"``), or 1-based atom numbers (``3``), most
        specific wins; an optional ``"default"`` entry fills every atom no other key
        covers. Without a ``"default"`` the assignment must cover every element.
    charge : int
    spin : int
        ``2S`` (number of unpaired electrons), PySCF convention.
    unit : str
        ``"Angstrom"`` (default) or ``"Bohr"``.
    point_group : str, optional
        Abelian double-group symmetry: ``"auto"``, or a name of the D2h chain. ⚠ The
        operations are tested in the frame the geometry is given in and the molecule is never
        reoriented, so orient the input so the symmetry axis is ``z``. See
        :func:`kuiva.interface.pyscf_bridge.run_scalar_x2c`.
    classification : str or bool, optional
        The non-abelian classification layer (needs ``point_group``): ``"auto"`` detects the
        full point double group and labels converged states by its irreps, activating only
        where the abelian group is not the whole story. ⚠ Classification, never adaptation —
        it changes no number.
    """
    atoms: List[Atom]
    basis: BasisSpec
    charge: int = 0
    spin: int = 0
    unit: str = "Angstrom"
    point_group: Optional[str] = None
    classification: object = "auto"

    def __post_init__(self) -> None:
        # Normalise symbols and validate the basis assignment eagerly (fail fast). The
        # resolution is the front end's (one family per atom, same addressing as the
        # reference configurations); this only runs it early so a typo fails here.
        self.atoms = [(s.capitalize(), tuple(float(x) for x in xyz)) for s, xyz in self.atoms]
        from .pyscf_bridge import _resolve_basis
        _resolve_basis(self.atoms, self.basis)

    @classmethod
    def from_xyz_string(cls, xyz: str, basis: BasisSpec, **kw) -> "Molecule":
        """Build from a simple ``"El x y z"`` per-line string (Angstrom)."""
        atoms: List[Atom] = []
        for line in xyz.strip().splitlines():
            parts = line.split()
            if not parts:
                continue
            s, x, y, z = parts[0], *map(float, parts[1:4])
            atoms.append((s, (x, y, z)))
        return cls(atoms=atoms, basis=basis, **kw)

    @property
    def elements(self) -> Tuple[str, ...]:
        return tuple(sorted({s for s, _ in self.atoms}))


def scalar_x2c_reference(molecule: Molecule, *, reference: str = "auto", fitting=None,
                         auxbasis=None, with_soc: bool = True,
                         method: Optional[str] = None,
                         x2c_approx: Optional[str] = None,
                         screening: Optional[str] = None,
                         screening_options: Optional[Dict[str, object]] = None,
                         configuration=None,
                         decoupling_options: Optional[Dict[str, object]] = None,
                         conv_tol: float = 1e-10, max_cycle: int = 200,
                         memory_gb: Optional[float] = None, n_active: Optional[int] = None,
                         cholesky_tol: float = DEFAULT_CHOLESKY_TOL,
                         orbit_pivots: bool = True, one_centre: bool = True,
                         gauge_origin=None, property_picture_change: bool = False,
                         anomaly_picture_change: bool = False,
                         atomic_reference: bool = False,
                         point_group: Optional[str] = None,
                         classification=None,
                         verbose: int = 0) -> ScalarX2CData:
    """Run the scalar-X2C front-end for ``molecule`` and return the ingested reference.

    This is the front-end boundary: the returned :class:`ScalarX2CData` is PySCF-free and
    is what the CASSCF/DMRG/NEVPT2 layers build on. It carries the scalar orbitals (one set
    for RHF/ROHF, two for UHF) *and* the two-component X2C Hamiltonian that spin-orbit
    coupling enters through. See :func:`kuiva.interface.pyscf_bridge.run_scalar_x2c` for the
    arguments.

    ``method`` picks the Hamiltonian by name — ``"X2C-AMF"`` (**the default**), ``"X2C-1e"``,
    ``"X2C-AMF-DLU"``, ``"X2C-1e-DLU"`` — and resolves to the two axes ``x2c_approx`` and
    ``screening``, which may also be set directly. Setting both a name and an axis that
    contradict it is refused. See :mod:`kuiva.x2c.methods`.

    ``screening`` selects the two-electron spin-orbit picture change. ``"x2camf"`` is
    the **default**: it adds the atomic mean field, at one four-component atomic SCF per
    unique element — seconds for a light element, ~35 minutes for a lanthanide, cached on disk
    and reusable across every geometry and every job. ``"none"`` skips it and leaves atomic
    j-splittings 5-30% too large.

    ``memory_gb`` sets the working-memory limit for this calculation and overrides the
    configured default; with neither, the calculation refuses to start rather than guess.
    """
    out.section(log, "Scalar-relativistic X2C reference")
    out.entries(log, [
        ("atoms", len(molecule.atoms), "", " ".join(molecule.elements)),
        ("charge", molecule.charge),
        ("spin (2S)", molecule.spin),
    ])
    if molecule.point_group is not None and point_group is not None \
            and str(molecule.point_group) != str(point_group):
        raise ValueError(
            "the molecule declares point_group={!r} and this call asks for {!r}; the symmetry "
            "a calculation runs in is one statement, not two".format(molecule.point_group,
                                                                     point_group))
    return run_scalar_x2c(molecule, reference=reference, fitting=fitting, auxbasis=auxbasis,
                          with_soc=with_soc, method=method, x2c_approx=x2c_approx,
                          screening=screening, screening_options=screening_options,
                          configuration=configuration,
                          decoupling_options=decoupling_options, conv_tol=conv_tol,
                          max_cycle=max_cycle, memory_gb=memory_gb, n_active=n_active,
                          cholesky_tol=cholesky_tol, orbit_pivots=orbit_pivots,
                          one_centre=one_centre, gauge_origin=gauge_origin,
                          property_picture_change=property_picture_change,
                          anomaly_picture_change=anomaly_picture_change,
                          atomic_reference=atomic_reference,
                          point_group=(molecule.point_group if point_group is None
                                       else point_group),
                          classification=(molecule.classification if classification is None
                                          else classification),
                          verbose=verbose)


@dataclass
class SpinorReference:
    """Everything the multireference layer starts from, in one PySCF-free container.

    This is the assembled output of the ingestion pipeline up to (not including) the CI step:
    the ingested scalar reference, the orthonormal working basis built from its overlap, the
    Kramers-paired spinor guess in that basis, and the three-index two-electron factors in
    the AO basis. Downstream code takes this and nothing else.
    """

    data: ScalarX2CData
    orth: OrthonormalBasis
    spinors: SpinorBasis
    factors: ThreeIndexAO

    @property
    def nspinor(self) -> int:
        return self.spinors.nspinor

    def spinors_in_ao(self, columns=None) -> np.ndarray:
        """Spinor coefficients in the AO basis, as the integral transform expects them."""
        sb = self.spinors.transform_scalar_basis(self.orth.x, basis="ao")
        return sb.c if columns is None else sb.take(columns)

    @property
    def symmetry(self):
        """The scalar reference's :class:`kuiva.symm.MolecularSymmetry`, or ``None``.

        ``None`` means the front end was not asked for symmetry, and every consumer then
        behaves exactly as it did before labels existed.
        """
        return self.data.symmetry

    @property
    def spinor_labels(self):
        """Per-spinor irrep labels (:class:`kuiva.symm.OrbitalLabels`), or ``None``.

        The labels are of the **guess** spinors, which are the columns every active-space
        selection indexes into. ⚠ They stay exact only while the orbitals stay symmetry-pure:
        a CASSCF that is not told to preserve the symmetry may rotate out of it, which the
        CI measures rather than assumes.
        """
        if self.data.symmetry is None:
            return None
        return self.data.symmetry.spinor_labels()

    @property
    def ao_layout(self):
        """The AO basis layout, for population analysis and the molden dump."""
        if self.data.ao_layout is None:
            raise ValueError("this reference carries no AO layout; it was not built by the "
                             "front-end (see pyscf_bridge.ao_layout)")
        return self.data.ao_layout

    def population_analysis(self, *, coeff: Optional[np.ndarray] = None, **kwargs):
        """Loewdin population analysis of a spinor set.

        Defaults to the reference's own guess spinors and their occupations; pass ``coeff``
        (AO basis) together with ``dm=`` or ``occupation=`` to analyse an optimized set. See
        :func:`kuiva.props.population.lowdin_analysis` for the ``level``, ``group``,
        ``tolerance`` and ``spin_vector`` options.
        """
        from ..props.population import lowdin_analysis
        if coeff is None:
            coeff = self.spinors_in_ao()
            kwargs.setdefault("occupation", self.spinors.occ)
            kwargs.setdefault("energy", self.spinors.energy)
        return lowdin_analysis(coeff, self.data.s_ao, self.ao_layout, **kwargs)

    def atomic_reference_charges(self, *, coeff: Optional[np.ndarray] = None,
                                 report: bool = True, **kwargs):
        """Atomic charges in the free-atom reference partition — the robust ones.

        Defaults to the reference's own guess spinors and occupations; pass ``coeff`` (AO
        basis) with ``dm=`` or ``occupation=`` to analyse an optimized or correlated
        density. Needs the front end to have been run with ``atomic_reference=True``
        (the per-element free-atom orbitals cannot be computed downstream); the error
        message says so if it was not. See
        :func:`kuiva.props.population.atomic_reference_charges`.
        """
        from ..props.population import atomic_reference_charges
        if coeff is None:
            coeff = self.spinors_in_ao()
            kwargs.setdefault("occupation", self.spinors.occ)
        return atomic_reference_charges(coeff, self.data.s_ao, self.ao_layout,
                                        self.data.atomic_reference, report=report,
                                        **kwargs)

    def write_molden(self, path, *, coeff: Optional[np.ndarray] = None, **kwargs):
        """Write spinor densities to a molden file.

        ⚠ The file holds the **exact real components of the spinor densities**, not orbitals;
        :mod:`kuiva.props.molden` states what that means and the file's own header repeats it.
        The Hamiltonian provenance is written into the header automatically.
        """
        from ..props.molden import write_spinor_molden
        if coeff is None:
            coeff = self.spinors_in_ao()
            kwargs.setdefault("occupation", self.spinors.occ)
            kwargs.setdefault("energy", self.spinors.energy)
        provenance = list(kwargs.pop("provenance", ()))
        if self.data.soc is not None:
            provenance.append("Hamiltonian: {}".format(
                json.dumps(self.data.soc.provenance(), sort_keys=True)))
        return write_spinor_molden(path, self.ao_layout, coeff, self.data.s_ao,
                                   provenance=provenance, **kwargs)

    def h_one_electron(self) -> np.ndarray:
        """The ``(2*nao, 2*nao)`` one-electron Hamiltonian in the AO basis.

        With spin-orbit coupling ingested this is the **full two-component X2C** Hamiltonian
        — the operator the multireference energy is an expectation value of. Without it,
        the spin-free Hamiltonian lifted to two components, i.e. a calculation with no SOC.
        Pass it straight to :func:`kuiva.integrals.transform.transform_1e`.
        """
        from ..spinor.expand import spin_block_diagonal
        if self.data.soc is not None:
            return self.data.soc.hamiltonian()
        return spin_block_diagonal(self.data.h_x2c)


def spinor_reference(molecule_or_data, *, threshold: float = DEFAULT_THRESHOLD,
                     scheme: str = "canonical", cholesky_tol: float = DEFAULT_CHOLESKY_TOL,
                     memory_gb: Optional[float] = None, orbit_pivots: bool = True,
                     **scf_kwargs) -> SpinorReference:
    """Front-end -> orthonormal working basis -> spinor guess -> factorized integrals.

    Accepts either a :class:`Molecule` (runs the scalar-X2C SCF first) or an already ingested
    :class:`ScalarX2CData`. Each stage logs its own standard output block, so this function is also
    the reference example of what the output looks like.

    ``memory_gb`` sets the working-memory limit. Given an already-ingested
    :class:`ScalarX2CData` the front-end's own pre-flight has already happened, so the limit
    is merely installed here; the remaining stages check against it as they allocate.

    ``orbit_pivots`` (default on) makes the Cholesky decomposition pivot on complete symmetry
    orbits, so the atomic spherical symmetry of the factorization is exact by construction
    rather than accurate to ``cholesky_tol``. Turning it off restores plain column
    pivoting, which splits free-ion degeneracies at the size of the threshold; it exists for
    measuring that, not for production.

    ⚠ ``cholesky_tol`` and ``orbit_pivots`` are **passed on to the front end** when this is
    given a :class:`Molecule`, because ``fitting="cholesky-direct"`` (or the default
    ``"auto"`` resolving to it when the stored plan exceeds the memory limit) decomposes
    there — while
    the integrals can still be evaluated and without ever storing them. Given an
    already-ingested container that came off that route, the decomposition has happened and
    these two can no longer be applied; a threshold that disagrees with the one it ran at is
    reported rather than silently ignored.
    """
    from ..integrals.transform import ThreeIndexAO
    from ..orth.canonical import orthogonalize, project_orbitals
    from ..spinor.expand import expand_scalar_mos, expand_unrestricted_mos

    if isinstance(molecule_or_data, ScalarX2CData):
        res.ensure_configured(memory_gb)
        data = molecule_or_data
    else:
        data = scalar_x2c_reference(molecule_or_data, memory_gb=memory_gb,
                                    cholesky_tol=cholesky_tol, orbit_pivots=orbit_pivots,
                                    **scf_kwargs)

    out.section(log, "Orthonormal working basis")
    orth = orthogonalize(data.s_ao, scheme, threshold, report=True)
    mo_work = tuple(project_orbitals(orth, c, data.s_ao) for c in data.mo_sets())

    out.section(log, "Spinor expansion")
    if data.unrestricted:
        spinors = expand_unrestricted_mos(mo_work[0], mo_work[1], data.mo_energy,
                                          data.mo_occ, basis="working", report=True)
    else:
        spinors = expand_scalar_mos(mo_work[0], data.mo_energy, data.mo_occ,
                                    basis="working", report=True)

    out.section(log, "Two-electron integrals")
    factors = ThreeIndexAO.from_scalar_data(data, cholesky_tol, orbit_pivots=orbit_pivots)
    return SpinorReference(data=data, orth=orth, spinors=spinors, factors=factors)


def project_to_basis(source: SpinorReference, target: SpinorReference,
                     coeff: Optional[np.ndarray] = None, *, space=None,
                     carry: str = _DEFAULT_CARRY,
                     scheme: str = _DEFAULT_PROJECTION_SCHEME,
                     repair_pairing="auto", report: bool = True):
    """Carry an orbital set from one basis-set calculation onto another.

    The production route to a large-basis CASSCF: converge it in a small basis, where the
    active orbitals are cheap to find and easy to identify, then project that result into the
    production basis and start there. The reverse direction (large onto small) is the same
    call and is supported; it discards a variational space rather than reproducing one, which
    the diagnostics say out loud.

    Parameters
    ----------
    source, target : :class:`SpinorReference`
        The two ingested references. They must be the same molecule in different bases; the
        elements are checked and a geometry difference warns. Both must have come from the
        front end (a container assembled by hand carries no basis to rebuild).
    coeff : ndarray ``(2*nao_source, n_source)``, optional
        The orbitals to carry over, in the **source AO basis** — normally a converged
        ``CASSCFOutcome.coeff``. Defaults to the source reference's own guess spinors.
    space : :class:`~kuiva.mcscf.casci.ActiveSpace` or
        :class:`~kuiva.mcscf.orbopt.OrbitalSpaces`, optional
        The source partition. Given, the inactive / active / virtual split is carried across
        exactly and each space is reported on separately; omitted, the set is one block.
    carry : str
        ``"active"`` (**the default**) carries the active orbitals and takes the inactive and
        virtual ones from the target's own SCF; ``"all"`` carries every orbital. ⚠ The
        default deliberately carries *less*: the active space is what the small-basis
        calculation was for, while the source's inactive orbitals are not eigenvectors of
        anything in the target basis and start the optimization with a large core-virtual
        gradient. :mod:`kuiva.orth.project` states the measurement.
    scheme : str
        The orthonormalization the projection is repaired with — ``"blocked"`` (default),
        ``"symmetric"`` or ``"gram-schmidt"``. See :mod:`kuiva.orth.project`, which is where
        the choice is documented and where the numbers behind the default live.
    repair_pairing : bool or ``"auto"``
        Rebuild the carried orbitals as explicit Kramers pairs. ⚠ On by default, because a
        converged general-complex CASSCF is entitled to leave its *active* orbitals far from
        pair-aligned — active-active rotations are redundant — while everything downstream
        that reads the pairing convention needs pairs. ``"auto"`` degrades to a warning for
        an unrestricted reference, whose spinors are not Kramers pairs at all.

    Returns
    -------
    :class:`~kuiva.orth.project.BasisProjection`
        ``.coeff`` is in the **target AO basis**, ready to pass as ``coeff=`` to
        :func:`casscf` or :func:`casci`; ``.plan`` holds the target orbital partition, and
        the rest is the evidence that the projection is worth using.
    """
    from ..orth.project import project_spinors

    spaces = getattr(space, "spaces", space)
    for name, ref in (("source", source), ("target", target)):
        if ref.data.molecule is None:
            raise ValueError(
                "the {} reference carries no molecule specification, so its basis cannot be "
                "rebuilt for the cross-basis overlap; a projection needs both sides to have "
                "come from the front end (kuiva.interface.api.scalar_x2c_reference)"
                .format(name))
    if source.data.nelec_total != target.data.nelec_total:
        raise ValueError(
            "the source reference holds {} electrons and the target {}; a projection carries "
            "one calculation's orbitals into another basis, not into another molecule"
            .format(source.data.nelec_total, target.data.nelec_total))
    c_source = (source.spinors_in_ao() if coeff is None
                else np.ascontiguousarray(coeff, dtype=np.complex128))

    if report:
        out.section(log, "Basis-set projection of the orbitals")
        out.entries(log, [
            ("source basis", ", ".join(sorted(set(source.data.basis_meta.values())))),
            ("target basis", ", ".join(sorted(set(target.data.basis_meta.values())))),
            ("AO functions", "{} -> {}".format(source.data.nao, target.data.nao)),
        ])
    s_cross = cross_overlap(source.data.molecule, target.data.molecule)
    kw = {}
    if spaces is not None:
        kw = dict(inactive=spaces.inactive, active=spaces.active, virtual=spaces.virtual)
    return project_spinors(c_source, s_cross, target.orth,
                           complete_with=target.spinors_in_ao(),
                           complete_energy=target.spinors.energy,
                           carry=carry, scheme=scheme, repair_pairing=repair_pairing,
                           report=report, **kw)


def projected_active_space(plan, target: SpinorReference, n_active_elec: int,
                           description: str = ""):
    """The target-basis :class:`~kuiva.mcscf.casci.ActiveSpace` a projection lands on.

    ``plan`` is a :class:`~kuiva.orth.project.ColumnPlan` or anything carrying one (a
    finished :class:`~kuiva.orth.project.BasisProjection` does).

    ⚠ **The active space follows the orbitals.** A projection carries the source partition
    across column for column, so the target space is the projected index set with the
    source's own electron count — never a fresh selection against the target's guess
    orbitals, which would be a different calculation wearing the same name. It is also why
    this needs no ``character=``: the physical statement was made once, where the space was
    chosen. The target reference's irrep labels are attached here, because they are the
    target's own.
    """
    from ..mcscf.casci import ActiveSpace
    from ..mcscf.orbopt import OrbitalSpaces

    plan = getattr(plan, "plan", plan)
    spaces = OrbitalSpaces(inactive=plan.inactive, active=plan.active,
                           virtual=plan.virtual, n_orb=plan.n_target)
    return _with_labels(ActiveSpace(spaces=spaces, n_elec=int(n_active_elec),
                                    description=description), target)


# --- The multireference entry points ------------------------------------------

def active_space_for(reference: SpinorReference, *, active=None, character=None,
                     n_active: Optional[int] = None,
                     n_active_elec: Optional[int] = None,
                     threshold: Optional[float] = None):
    """Resolve a user's active-space request into an
    :class:`~kuiva.mcscf.casci.ActiveSpace`.

    Exactly one of the two routes, because they answer the same question and a run that
    silently preferred one would be unreproducible:

    ``active=[...]``
        Explicit spinor indices. Exact, and meaningless to anyone without these orbitals.
        An already-resolved :class:`~kuiva.mcscf.casci.ActiveSpace` is passed through
        unchanged (the class layer resolves eagerly and hands the result down).
    ``character=(atom, l), n_active=N``
        The lowest ``N/2`` Kramers pairs of that character — *"the ten lowest spinors of d
        character on Ti"*. ⚠ reproducibility requires this form for any calculation that is going to be
        a reference, because it is the only one an independent implementation can reproduce.
        Needs the AO layout, which the front-end carries on the reference. ``atom`` may be a
        sequence of centres, whose populations are pooled — the right form for equivalent
        centres whose canonical orbitals delocalize.
    ``character=[(atom, l, n_spinors), ...]``
        A **list** means a union of per-fragment selections — *"6 spinors of d character on
        the Ti plus 14 of f character on the Dy"* — refused rather than shared when two
        fragments claim the same pair (:func:`kuiva.mcscf.casci.active_space_by_characters`).
        A fragment may omit its count (``(atom, l)``) when ``n_active`` fixes the remainder.

    ``n_active_elec`` is optional in all forms: without it the count follows from aufbau. See
    :func:`kuiva.mcscf.casci.active_space` for the electron-count and Kramers-pair traps this
    enforces.
    """
    from ..mcscf.casci import (ActiveSpace, active_space, active_space_by_character,
                               active_space_by_characters)

    if isinstance(active, ActiveSpace):
        if character is not None:
            raise ValueError("give the resolved ActiveSpace or a character selection, "
                             "not both")
        return _with_labels(active, reference)
    if (active is None) == (character is None):
        raise ValueError(
            "give exactly one of active=[spinor indices] or character=(atom, l) with "
            "n_active=; an active space stated only as an index window is not a definition "
            "another program can reproduce ")
    n_orb = reference.nspinor
    n_elec_total = reference.data.nelec_total
    if active is not None:
        return _with_labels(
            active_space(active, n_orb, n_elec_total, n_active_elec=n_active_elec,
                         kramers_paired=reference.spinors.kramers_paired), reference)
    kwargs = {} if threshold is None else {"threshold": float(threshold)}

    if isinstance(character, list):
        entries: List[List[object]] = []
        unspecified: List[int] = []
        n_explicit = 0
        for entry in character:
            entry = tuple(entry)
            if len(entry) == 2:
                unspecified.append(len(entries))
                entries.append([entry[0], entry[1], None])
            elif len(entry) == 3:
                entries.append([entry[0], entry[1], int(entry[2])])
                n_explicit += int(entry[2])
            else:
                raise ValueError("a fragment is (atom, l, n_spinors) or (atom, l); got {!r}"
                                 .format(entry))
        if unspecified:
            if n_active is None:
                raise ValueError(
                    "fragment(s) {} carry no spinor count and no n_active was given to "
                    "supply one".format([tuple(entries[i][:2]) for i in unspecified]))
            share, leftover = divmod(int(n_active) - n_explicit, len(unspecified))
            if share <= 0 or leftover != 0 or share % 2 != 0:
                raise ValueError(
                    "n_active = {} leaves {} spinors for {} unspecified fragment(s), which "
                    "does not divide into equal whole Kramers pairs; state the counts as "
                    "(atom, l, n_spinors) triples".format(
                        n_active, int(n_active) - n_explicit, len(unspecified)))
            for i in unspecified:
                entries[i][2] = share
        elif n_active is not None and n_explicit != int(n_active):
            raise ValueError("the fragment counts sum to {} spinors but n_active = {}; drop "
                             "n_active or make them agree".format(n_explicit, n_active))
        return _with_labels(active_space_by_characters(
            reference.spinors_in_ao(), reference.data.s_ao, reference.ao_layout,
            n_elec_total, fragments=[tuple(e) for e in entries],
            n_active_elec=n_active_elec, occupation=reference.spinors.occ, **kwargs), reference)

    if n_active is None or int(n_active) % 2 != 0:
        raise ValueError("character selection needs an even n_active (whole Kramers pairs, "
                         "whole Kramers pairs only); got {!r}".format(n_active))
    atom, l = character
    return _with_labels(active_space_by_character(
        reference.spinors_in_ao(), reference.data.s_ao, reference.ao_layout, n_elec_total,
        atom=atom, l=l, n_pairs=int(n_active) // 2, n_active_elec=n_active_elec,
        occupation=reference.spinors.occ, **kwargs), reference)


def _with_labels(space, reference: SpinorReference):
    """Attach the reference's spinor labels, sliced to the active columns.

    ⚠ **A label-open active space makes every sector count a lie.** The labels are attached
    only after checking that the space is closed under conjugation — that whenever a spinor is
    active, the partner carrying the conjugate label is too. A Kramers-paired selection is
    closed by construction (a pair's two members carry conjugate labels), so this fires only
    where the pairing was already broken, and it warns rather than refusing because the
    calculation itself is still perfectly well defined without labels.

    ⚠ Closure under the *abelian* group is not closure under the molecule's real group. Where
    the two differ — every atom, and every molecule reduced from a group whose double group
    is non-abelian — a per-irrep count can still cut a physically degenerate manifold, which
    is what the state-average gate and the boundary diagnostic are there for.
    """
    labels = reference.spinor_labels
    if labels is None:
        return space
    if len(labels) != reference.nspinor:
        log.warning("the reference carries %d spinor labels for %d spinors, so the active "
                    "space is left unlabelled; per-irrep selection is unavailable for this "
                    "run", len(labels), reference.nspinor)
        return space
    active = labels.take(space.spaces.active)
    group = active.group
    counts = active.counts()
    unbalanced = [group.irrep_name(t) for t in group.labels(fermion=True)
                  if counts.get(t, 0) != counts.get(group.conjugate(t), 0)]
    if unbalanced:
        log.warning("the active space holds unequal numbers of spinors in the conjugate "
                    "irrep pair(s) %s, so it is not closed under time reversal; sector counts "
                    "computed from it describe the truncation as much as the physics",
                    ", ".join(sorted(unbalanced)))
    return replace(space, labels=active)


def _classifier_for(reference: SpinorReference, space, orbitals, solver_space,
                    classify=True):
    """A :class:`kuiva.symm.StateClassifier` for this reference, or ``None``.

    ⚠ Everything here degrades to ``None`` rather than raising: the non-abelian layer is a
    *labelling* of converged states and no calculation may fail because a label could not be
    attached. The one thing that does refuse is downstream — a state count that cuts a
    multiplet whose dimension theory fixes.
    """
    if not classify:
        return None
    symmetry = reference.symmetry
    if symmetry is None or getattr(symmetry, "full_group", None) is None:
        return None
    from ..symm.classify import StateClassifier
    try:
        return StateClassifier(symmetry.full_group, reference.ao_layout, orbitals,
                               np.asarray(reference.data.s_ao), space.spaces, solver_space)
    except Exception as exc:                       # noqa: BLE001 - advisory by design
        log.warning("the non-abelian classification layer is off for this active space (%s); "
                    "the abelian labels and every number are unaffected", exc)
        return None


def _check_restart_state_average(resumed, solver, path) -> None:
    """Refuse a restart whose state average differs from the one in the file.

    The active space is checked the same way and for the same reason a few lines above: a
    restart **continues the calculation that was interrupted**, and a different state average
    is a different calculation, not a different chart of this one. The energy functional
    itself changes, so the converged answer does — and the trajectory that came out would be
    an average of two averages, with a plausible number at the end of it and nothing in the
    output saying so.

    ⚠ Refusing, rather than clearing curvature the way a genuine chart change does, is the
    decision here. Continuing from converged orbitals into a *new* state average is a real
    thing to want, and it is what ``coeff=`` is for; the message says so.

    A file written before the state average was recorded carries no entry. That is read as
    "cannot be compared" and warned about — never as "matches", which would be the same
    silent pass this check exists to remove.
    """
    from ..io.checkpoint import STATE_AVERAGE_KEY, state_average_key

    stored = resumed.metadata.get(STATE_AVERAGE_KEY)
    current = state_average_key(solver)
    if stored is None:
        log.warning("the checkpoint at %s predates the recorded state average, so this "
                    "restart cannot be checked against it; confirm that n_states and weights "
                    "match the interrupted run, because restoring curvature across a changed "
                    "state average converges to a plausible wrong answer", path)
        return
    if current is not None and stored != current:
        raise ValueError(
            "the checkpoint at {} was written for a state average of {} and this call asks "
            "for {}. A restart continues the calculation that was interrupted; a different "
            "state average is a different calculation, whose orbitals and energy differ. "
            "Leave n_states/weights out so they come from the file, make them match — or, if "
            "starting a NEW state average from these converged orbitals is what you meant, "
            "read the orbitals and pass them as coeff= instead of restarting."
            .format(path, stored.replace(";", ", "), current.replace(";", ", ")))


def casci(reference: SpinorReference, *, active=None, character=None,
          n_active: Optional[int] = None, n_active_elec: Optional[int] = None,
          n_states: int = 1, weights=None, coeff: Optional[np.ndarray] = None,
          report: bool = True, classify: bool = True, **solver_kwargs):
    """A full CI at fixed orbitals over the chosen active space.

    The spectrum this returns **is** the spin-orbit spectrum: the CI is already
    two-component, so its roots are the SOC eigenstates and there is no separate spin-orbit
    mixing step (unlike a RASSI-style two-step treatment). See
    :class:`~kuiva.mcscf.casci.CASCIResult`.

    ``coeff`` runs the CASCI on an orbital set other than the reference's guess — converged
    CASSCF orbitals, or a set read from a checkpoint.
    """
    from ..mcscf.casci import casci as _casci

    space = active_space_for(reference, active=active, character=character,
                             n_active=n_active, n_active_elec=n_active_elec)
    if report:
        out.section(log, "CASCI")
        space.report(log)
    orbitals = reference.spinors_in_ao() if coeff is None else np.ascontiguousarray(coeff)
    if space.labels is not None:
        solver_kwargs.setdefault("symmetry", space.labels)
    from ..ci.strings import CASSpace
    classifier = _classifier_for(reference, space, orbitals,
                                 CASSpace(space.spaces.n_active, space.n_elec),
                                 classify=classify)
    result = _casci(reference.factors, reference.h_one_electron(), orbitals, space.spaces,
                    space.n_elec, n_states=n_states, e_nuc=reference.data.e_nuc,
                    weights=weights, report=report, classifier=classifier, **solver_kwargs)
    result.description = space.description
    return result


def casscf(reference: SpinorReference, *, active=None, character=None,
           n_active: Optional[int] = None, n_active_elec: Optional[int] = None,
           n_states=1, weights=None, coeff: Optional[np.ndarray] = None,
           checkpoint=None, restart=None, checkpoint_options: Optional[Dict] = None,
           solver_options: Optional[Dict] = None, callback=None,
           preserve_symmetry: bool = False, report: bool = True, classify: bool = True,
           **optimizer_kwargs):
    """State-averaged two-component CASSCF — the calculation this program exists for.

    Front-end to :func:`kuiva.mcscf.casscf`: resolves the active space (see
    :func:`active_space_for`), builds the full-CI solver, hands it to the shared orbital
    optimizer, and returns a :class:`~kuiva.mcscf.casci.CASSCFOutcome` carrying both
    the orbitals and the states.

    Checkpointing and restart
    ------------------------------
    ``checkpoint=path`` writes a schema-versioned HDF5 restart point every macro-iteration the
    adaptive budget allows (:class:`kuiva.io.checkpoint.CheckpointPolicy`); the converged one
    is always written. ``restart=path`` resumes from one: the orbitals, the orbital-rotation
    state, the iteration count and the Davidson warm start all come back, and the integrals
    are **regenerated** from the orbitals rather than stored.

    ⚠ A restart takes its active space from the checkpoint, so ``active``/``character`` may be
    omitted — and if they are given and disagree with the file, it is refused rather than
    reconciled. ⚠ ``max_iter`` counts total macro-iterations across the restart, so an
    interrupted run costs what an uninterrupted one would.

    ``optimizer_kwargs`` pass through to :func:`kuiva.mcscf.orbopt.optimize_orbitals`
    (``mode``, ``max_iter``, ``conv_grad``, ``conv_energy``, ``max_step``, ...).

    Symmetry
    --------
    ``n_states={irrep: n}`` selects states **per irrep** instead of "lowest n" (the front end
    must have been run with ``point_group=``). ``preserve_symmetry=True`` additionally
    restricts the orbital rotation to within each irrep, so the labels still mean something at
    convergence rather than only at the start. ⚠ That is a **constraint**: what it converges
    to is the lowest *symmetric* solution, which is not the global one wherever the symmetry
    is spontaneously broken. Without it a per-irrep selection still works, and the CI measures
    how far the orbitals have drifted out of the symmetry rather than assuming they have not.

    Is the state average complete?
    -------------------------------------
    ⚠ A state-averaged CASSCF is exactly as symmetric as the set it averages over, and a
    ``n_states`` that ends *inside* a near-degenerate manifold breaks that symmetry
    **self-consistently** while every number it produces stays plausible. This therefore solves
    ``boundary_check`` extra roots at fixed orbitals, reports how far the average's last root is
    from the first one it leaves out, and warns when that gap is too small to be unambiguous.
    The extra roots are discarded, never averaged over.

    It runs **twice**, and the two are different statements:
    :attr:`~kuiva.mcscf.casci.CASSCFOutcome.boundary_initial` at the orbitals the optimization
    started from and :attr:`~kuiva.mcscf.casci.CASSCFOutcome.boundary` at the converged ones. ⚠
    The **initial** one is what says whether the trajectory was safe — an incomplete average
    breaks the density on the way, so the converged check is already too late, and if the
    Kramers gate refuses part way through it is the only one that exists. Seeding the first CI
    solve from the pre-flight's own vectors keeps its cost to about the extra roots; the
    converged one costs an integral build plus a Davidson solve, i.e. roughly one
    macro-iteration. ``boundary_check=0`` switches both off. ⚠ Either may come back ``None``:
    the extra roots are the hardest in the solve, and a check that cannot converge warns rather
    than killing the calculation it was only advising on.
    """
    from ..io.checkpoint import CheckpointPolicy, read_checkpoint
    from ..mcscf.casci import ActiveSpace, FullCISolver, casscf as _casscf

    resumed = None
    if restart is not None:
        # ⚠ Read failure on an explicit restart is an ERROR that propagates:
        # the user asked to resume, and silently starting over wastes what the file protects.
        resumed = read_checkpoint(restart)
        space = ActiveSpace(spaces=resumed.spaces, n_elec=resumed.n_active_elec,
                            description="restored from {}".format(restart))
        if active is not None or character is not None:
            requested = active_space_for(reference, active=active, character=character,
                                         n_active=n_active, n_active_elec=n_active_elec)
            if (not np.array_equal(requested.spaces.active, space.spaces.active)
                    or requested.n_elec != space.n_elec):
                raise ValueError(
                    "the checkpoint at {} holds CAS({}, {}) on spinors {}..{} and the "
                    "arguments ask for CAS({}, {}); a restart continues the calculation that "
                    "was interrupted, so leave the active space out or make it match"
                    .format(restart, space.n_elec, space.spaces.n_active,
                            int(space.spaces.active[0]), int(space.spaces.active[-1]),
                            requested.n_elec, requested.spaces.n_active))
        orbitals = np.ascontiguousarray(resumed.coeff)
    else:
        space = active_space_for(reference, active=active, character=character,
                                 n_active=n_active, n_active_elec=n_active_elec)
        orbitals = reference.spinors_in_ao() if coeff is None else np.ascontiguousarray(coeff)

    if report:
        out.section(log, "CASSCF")
        space.report(log)
        if resumed is not None:
            resumed.report(log)

    solver_options = dict(solver_options or {})
    if space.labels is not None:
        solver_options.setdefault("symmetry", space.labels)
    solver = FullCISolver(space.spaces.n_active, space.n_elec, n_states=n_states,
                          weights=weights, **solver_options)
    if resumed is not None:
        _check_restart_state_average(resumed, solver, restart)
        solver.set_guess(resumed.ci_vectors)
        # ⚠ The LIVE solver's key, never the file's own. Chart-scoping compares the key
        # recorded inside optimizer_state against the key of the solver about to run; handing
        # the checkpoint its own key back compares the file with itself, so the comparison
        # always passes and curvature is restored across a chart change without a word.
        optimizer_kwargs.update(resumed.optimizer_kwargs(space_key=solver.space_key()))

    hook = callback
    policy = None
    if checkpoint is not None:
        metadata = {"active_space": space.description}
        if reference.data.soc is not None:
            metadata["hamiltonian"] = json.dumps(reference.data.soc.provenance(),
                                                 sort_keys=True)
        policy = CheckpointPolicy(checkpoint, solver=solver, metadata=metadata,
                                  n_active_elec=space.n_elec, chain=callback,
                                  **(checkpoint_options or {}))
        hook = policy.callback

    if preserve_symmetry:
        labels = reference.spinor_labels
        if labels is None:
            raise ValueError(
                "preserve_symmetry=True needs irrep labels for the orbitals, and this "
                "reference carries none; run the front end with point_group=")
        if coeff is not None or resumed is not None:
            log.warning("preserve_symmetry=True masks the orbital rotation by the labels of "
                        "the reference's OWN spinors, and this run starts from a different "
                        "orbital set; the mask is only meaningful if that set is the "
                        "symmetry-adapted one the labels were read off")
        optimizer_kwargs.setdefault("labels", labels.labels)

    optimizer_kwargs.setdefault("report", report)
    # the two-component spin matrices let the converged boundary report state whether the
    # averaged density is spin-rotation invariant — the front-end has the overlap, the
    # mcscf layer deliberately does not
    from ..spinor.expand import spin_operator
    optimizer_kwargs.setdefault("spin_ao_2c", spin_operator(np.asarray(reference.data.s_ao)))
    classifier = _classifier_for(reference, space, orbitals, solver.space, classify=classify)
    outcome = _casscf(reference.factors, reference.h_one_electron(), orbitals, space.spaces,
                      space.n_elec, n_states=n_states, e_nuc=reference.data.e_nuc,
                      solver=solver, active=space, callback=hook, classifier=classifier,
                      **optimizer_kwargs)
    if policy is not None:
        outcome.checkpoint_path = str(policy.path)
        if report:
            policy.report(log)
    return outcome


def property_matrices(reference: SpinorReference, source, *, comments=(),
                      inactive_tol: Optional[float] = None):
    """``H`` and the three magnetic-moment matrices in the SOC eigenstate basis.

    ``source`` is what a calculation returned: a
    :class:`~kuiva.mcscf.casci.CASSCFOutcome` (from :func:`casscf`) or a
    :class:`~kuiva.mcscf.casci.CASCIResult` (from :func:`casci`). Either carries the states,
    the orbitals they were solved at and the solver that owns the excitation map, which is
    everything the moment matrices need.

    ⚠ **The gauge origin is fixed at ingestion**, not here: ``L`` is defined relative to it
    and the multireference layer never calls PySCF again. Pass
    ``gauge_origin=`` to :func:`scalar_x2c_reference` to change it.

    Returns a :class:`kuiva.props.dump.PropertyMatrices`. ⚠ Its phases are arbitrary and
    degenerate states mix arbitrarily — compare only through
:meth:`~kuiva.props.dump.PropertyMatrices.analyse`.
    """
    from ..mcscf.casci import CASSCFOutcome
    from ..props.dump import property_matrices as _matrices

    if isinstance(source, CASSCFOutcome):
        ci, coeff, spaces = source.ci, source.coeff, source.active.spaces
        description = source.active.description or ci.description
    else:
        ci, coeff, spaces, description = source, source.coeff, source.spaces, source.description
    if coeff is None or spaces is None:
        raise ValueError(
            "this CASCI result does not know which orbitals or which orbital-space partition "
            "it belongs to, so the property matrices cannot be built from it. Use "
            "api.casci/api.casscf, which record both — a moment matrix built from a "
            "mismatched orbital and state set is Hermitian, plausible and wrong")
    if reference.data.properties is None:
        raise ValueError("this reference carries no property integrals; it was not built by "
                         "the front-end (see pyscf_bridge.ingest_property_integrals)")

    provenance: Dict[str, object] = {
        "active_space": description or "unspecified",
        "n_active_spinors": int(spaces.n_active),
        # Tr(gamma) is the active electron count by construction, so this is a statement the
        # states themselves make rather than one copied from the request.
        "n_active_electrons": int(round(float(np.real(np.trace(ci.gamma))))),
        "n_states": int(np.size(ci.energies)),
        "state_averaging_weights": [float(w) for w in np.asarray(ci.weights).ravel()],
    }
    if reference.data.soc is not None:
        provenance["hamiltonian"] = reference.data.soc.provenance()
    provenance["basis"] = dict(reference.data.basis_meta)
    kwargs = {} if inactive_tol is None else {"inactive_tol": float(inactive_tol)}
    return _matrices(coeff, spaces, ci.transition_densities(), ci.total_energies,
                     reference.data.properties, reference.data.s_ao,
                     provenance=provenance, active_space=description,
                     comments=comments, **kwargs)


def property_dump(reference: SpinorReference, source, path, *, title: str = "",
                  include_l_s: bool = True, report: bool = True, **kwargs):
    """Write the property-matrix file — the program's product — and return the matrices.

    A thin composition of :func:`property_matrices` and
    :func:`kuiva.props.dump.write_dump`. The header carries the full Hamiltonian provenance
, the gauge origin and the active space; ⚠ a ``WARNING`` is emitted about the
    missing picture change on ``L`` and ``S``, every time, and recorded in the file.
    """
    matrices = property_matrices(reference, source, **kwargs)
    if report:
        out.section(log, "Property matrices")
        matrices.report(log)
    matrices.write(path, title=title, include_l_s=include_l_s)
    return matrices


__all__ = ["Molecule", "ScalarX2CData", "SpinorReference", "scalar_x2c_reference",
           "spinor_reference", "build_mole", "memory_plan",
           "project_to_basis", "projected_active_space",
           "active_space_for", "casci", "casscf", "property_matrices", "property_dump"]
