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
from .environment import Environment
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
        An entry written ``("ghost-Cl", pos)`` is a **ghost**: chlorine's basis functions with
        no nucleus, no electrons and no mass (:mod:`kuiva.basis.ghosts`). It is addressed by
        that label everywhere — in ``basis``, in the output, in every per-atom map — and never
        by the element it carries the basis of.
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
    environment : Environment, optional
        What surrounds the molecule (:mod:`kuiva.interface.environment`): today a field of
        classical **point charges**, which is what makes a crystal-embedded calculation
        different from a gas-phase one. ⚠ The field's coordinates are in the molecule's own
        ``unit`` unless the environment states its own, it does not move the gauge origin, and
        it **does** take part in symmetry detection — a field of lower symmetry than the nuclei
        is a real symmetry breaking, not a labelling detail.
    nuclear_model : str, optional
        The nuclear charge distribution: ``"point"`` (**the default**, and what every
        reference number shipped with this program was produced with) or ``"gaussian"``, the
        finite nucleus of Visscher and Dyall (:mod:`kuiva.x2c.nuclear`). One statement for the
        whole molecule, inherited by every consumer — the molecular integrals, the atomic
        four-component solves behind the two-electron spin-orbit screening, and the free-atom
        reference orbitals.

        ⚠ **It is part of the Hamiltonian, so it is not comparable across settings**: the
        finite nucleus lowers j-splittings by an amount that grows steeply with Z and is
        negligible for light elements. It is also the first thing to check against a
        four-component program, several of which default to a Gaussian nucleus where Kuiva
        defaults to a point one.
    """
    atoms: List[Atom]
    basis: BasisSpec
    charge: int = 0
    spin: int = 0
    unit: str = "Angstrom"
    point_group: Optional[str] = None
    classification: object = "auto"
    nuclear_model: str = "point"
    environment: object = None

    def __post_init__(self) -> None:
        # Normalise symbols and validate the basis assignment eagerly (fail fast). The
        # resolution is the front end's (one family per atom, same addressing as the
        # reference configurations); this only runs it early so a typo fails here.
        from ..basis.ghosts import normalize_symbol
        self.atoms = [(normalize_symbol(s), tuple(float(x) for x in xyz))
                      for s, xyz in self.atoms]
        from ..x2c.nuclear import resolve_nuclear_model
        from .pyscf_bridge import _resolve_basis
        _resolve_basis(self.atoms, self.basis)
        # Normalized here so ``"gauss"`` and ``"gaussian"`` are one model everywhere
        # downstream, and a misspelling fails at construction rather than after the SCF.
        self.nuclear_model = resolve_nuclear_model(self.nuclear_model)
        # Coerced eagerly, so a malformed charge field fails at construction rather than
        # after the memory pre-flight and a four-component atomic solve.
        if self.environment is not None and not isinstance(self.environment, Environment):
            self.environment = Environment(point_charges=self.environment)

    @classmethod
    def from_xyz_string(cls, xyz: str, basis: BasisSpec, **kw) -> "Molecule":
        """Build from a simple ``"El x y z"`` per-line string (Angstrom).

        ⚠ **Those lines only.** A real ``.xyz`` *file* opens with an atom count and a comment
        line, and neither is skipped here — use :meth:`from_xyz_file`, which knows about them.
        """
        atoms: List[Atom] = []
        for line in xyz.strip().splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                s, x, y, z = parts[0], *map(float, parts[1:4])
            except ValueError:
                raise ValueError(
                    "cannot read {!r} as an 'El x y z' line. ⚠ If this is the header of an "
                    "XMol .xyz file (an atom count, then a comment), use "
                    "Molecule.from_xyz_file, which skips it".format(line.strip()[:60]))
            atoms.append((s, (x, y, z)))
        return cls(atoms=atoms, basis=basis, **kw)

    @classmethod
    def from_xyz_file(cls, path, basis: BasisSpec, **kw) -> "Molecule":
        """Build from an XMol ``.xyz`` file — count line, comment line, then the atoms.

        The two-line header is what distinguishes this from :meth:`from_xyz_string`, and it
        is the reason this exists: pointing the string form at a real file fails on the count
        line, which is a confusing way to learn about a format.

        ⚠ **The count is checked, not trusted.** A file whose header disagrees with the atoms
        it contains is truncated or concatenated, and reading the first ``n`` of a longer file
        would silently compute a different molecule. Headerless files are accepted too — some
        tools emit them — so what is refused is a *disagreement*, never the absence of a
        header.

        Coordinates are Angstrom, which is the format's convention and this class's default.
        """
        from pathlib import Path as _Path

        text = _Path(path).read_text()
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("{}: the file is empty".format(path))
        declared = None
        if len(lines[0].split()) == 1 and lines[0].strip().isdigit():
            declared = int(lines[0].strip())
            lines = lines[2:] if len(lines) > 1 else []       # count, then the comment line
        mol = cls.from_xyz_string("\n".join(lines), basis, **kw)
        if declared is not None and declared != len(mol.atoms):
            raise ValueError(
                "{}: the header declares {} atoms and the file carries {}. A truncated or "
                "concatenated .xyz is not read as its first {} atoms, because that is a "
                "different molecule and nothing downstream would say so"
                .format(path, declared, len(mol.atoms), declared))
        return mol

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
                         level_shift: float = 0.0, damp: float = 0.0,
                         init_guess: Optional[str] = None, diis=None,
                         diis_space: Optional[int] = None,
                         diis_start_cycle: Optional[int] = None,
                         second_order: bool = False, stability: Optional[str] = None,
                         guess_from=None, broken_symmetry=None,
                         bs_min_population: Optional[float] = None,
                         allow_unconverged_scf: bool = False,
                         memory_gb: Optional[float] = None, n_active: Optional[int] = None,
                         n_active_elec: Optional[int] = None,
                         n_states: Optional[int] = None,
                         nevpt2: bool = False,
                         factors: Optional[str] = None,
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

    The SCF's convergence controls — ``level_shift``, ``damp``, ``init_guess``, ``diis``
    (``"cdiis"``/``"adiis"``/``"ediis"``), ``diis_space``, ``diis_start_cycle``,
    ``second_order`` and ``stability`` — are documented on
    :func:`~kuiva.interface.pyscf_bridge.run_scalar_x2c`. ``guess_from`` starts the SCF from a
    previous calculation's scalar orbitals, projecting them if the basis differs.

    ⚠ **An SCF that does not converge refuses**, because everything downstream is built on its
    orbitals; ``allow_unconverged_scf=True`` continues on them deliberately.

    ``memory_gb`` sets the working-memory limit for this calculation and overrides the
    configured default; with neither, the calculation refuses to start rather than guess.
    ``n_active`` — and, one step further, ``n_active_elec`` with ``n_states``, and
    ``nevpt2=True`` for a run that will end in the perturbation — sharpen the memory
    pre-flight when the multireference stage is already decided: with the space fully
    stated the plan carries the conventional-CI residency (and the perturbation's
    shifted-space workspace and vector sets) too, so a request those stages cannot hold is
    refused here, before the SCF is paid for. Planning only; the CASSCF still states its
    own space.
    ``factors`` is where the three-index factor rows live: ``"in-core"``, ``"scratch"``
    (spilled to a scratch file after the decomposition and streamed back in sequential
    blocks — bitwise identical), ``"streamed"`` (the decomposition itself runs out of core,
    writing each vector to scratch as it is produced, so the factor array is never allocated;
    ⚠ **not** bitwise, see :func:`kuiva.integrals.transform.streamed_cholesky`), or
    ``"auto"`` (the default — each rung taken only where it lowers the planned peak below the
    memory limit, announced on its own output line).
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
                          max_cycle=max_cycle, level_shift=level_shift, damp=damp,
                          init_guess=init_guess, diis=diis, diis_space=diis_space,
                          diis_start_cycle=diis_start_cycle, second_order=second_order,
                          stability=stability, guess_from=guess_from,
                          broken_symmetry=broken_symmetry,
                          bs_min_population=bs_min_population,
                          allow_unconverged_scf=allow_unconverged_scf,
                          memory_gb=memory_gb, n_active=n_active,
                          n_active_elec=n_active_elec, n_states=n_states, nevpt2=nevpt2,
                          factors=factors,
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



def localize_active_space(reference: SpinorReference, space, sites, *,
                          coeff: Optional[np.ndarray] = None,
                          counts: Optional[Sequence[int]] = None,
                          min_population: Optional[float] = None,
                          repair_pairing: bool = True, report: bool = True):
    """Rotate an active space into orbitals that each belong to one centre.

    The user surface of :mod:`kuiva.mcscf.localize`. Selection by character says *which
    orbitals* are active — "the ten lowest spinors of d character on the two titaniums" —
    and for two equivalent centres that is as far as it can go: the canonical orbitals are
    the symmetric and antisymmetric combinations, each half on each metal. This says **which
    centre**, by rotating inside the space that was selected.

    ⚠ **It changes no number.** The rotation is active-active, so the CASCI/CASSCF energy is
    invariant to machine precision — asserted in the suite rather than tolerated. What it
    changes is what the orbitals *mean*, which is what a broken-symmetry guess
    (:func:`kuiva.interface.pyscf_bridge.run_scalar_x2c`'s ``broken_symmetry=``) and a
    site-blocked mode ordering both need.

    Parameters
    ----------
    reference, space
        A finished :class:`SpinorReference` and the :class:`~kuiva.mcscf.casci.ActiveSpace`
        to localize (from :func:`active_space_for`).
    sites : sequence
        One entry per centre; the addressing is ``character=``'s (atom index, unique element
        symbol, or a sequence of either for a multi-atom fragment).
    coeff : ndarray ``(2*nao, n)``, optional
        Localize *these* orbitals (AO basis) rather than the reference's guess — a converged
        ``CASSCFOutcome.coeff`` is the case this exists for, since the orbitals a multi-site
        export or a tensor network is handed are the optimized ones.
    counts : sequence of int, optional
        Orbitals per site; the default equal split is refused when it does not divide.
    min_population : float, optional
        Own-site Löwdin population every localized orbital must reach
        (:data:`~kuiva.mcscf.localize.DEFAULT_SITE_POPULATION_MIN`). The refusal prints the
        table, because "these orbitals are not site orbitals" is the useful answer.
    repair_pairing : bool
        Rebuild each site's block as explicit Kramers pairs afterwards (default). ⚠ The
        localizing rotation mixes the members of a pair like any other active-active
        rotation, and the pair convention is what an active space is addressed in; each
        site's span is time-reversal closed, so the repair is a rotation *inside* a site and
        moves no population. Set ``False`` only to inspect the raw localization.

    Returns
    -------
    :class:`kuiva.mcscf.localize.FragmentLocalization`
        Whose ``coeff`` is in the **AO basis**, ready to pass as ``coeff=`` to
        :func:`casscf` / :func:`casci` or to the ``CASSCF`` stage.
    """
    from ..mcscf.localize import DEFAULT_SITE_POPULATION_MIN, fragment_populations, localize
    from ..spinor.expand import nearest_kramers_paired

    active = np.asarray(space.spaces.active, dtype=int)
    floor = (DEFAULT_SITE_POPULATION_MIN if min_population is None else float(min_population))
    c_ao = (reference.spinors_in_ao() if coeff is None
            else np.ascontiguousarray(coeff, dtype=np.complex128))
    result = localize(c_ao, reference.data.s_ao, reference.ao_layout,
                      active, sites, counts=counts, min_population=floor)
    if repair_pairing:
        # ⚠ In the **working** basis, which is where the pairing repair is defined (an
        # orthonormal real scalar basis; the AO metric is not the identity). The rotation is
        # a right multiplication, so the same unitary moves either representation, and the
        # repaired columns come back through the same transform the reference uses.
        blocks = [result.site_columns(i) for i in range(result.n_sites)]
        if any(b.size % 2 for b in blocks):
            raise ValueError(
                "repair_pairing needs an even number of orbitals on every site (a Kramers "
                "pair belongs to one centre, and half a pair belongs to nothing); got {}"
                .format([int(b.size) for b in blocks]))
        # AO -> working for a caller-supplied set (X^T S per spin block); the reference's
        # own spinors are already there.
        if coeff is None:
            c_work = np.array(reference.spinors.c, copy=True)
        else:
            nao = int(reference.data.s_ao.shape[0])
            c_work = np.vstack([reference.orth.to_working(c_ao[:nao]),
                                reference.orth.to_working(c_ao[nao:])])
        c_work[:, active] = c_work[:, active] @ result.rotation
        c_work = nearest_kramers_paired(c_work, blocks)
        n_col = c_work.shape[1]
        sb = SpinorBasis(c_work, np.zeros(n_col), np.zeros(n_col),
                         kramers_paired=reference.spinors.kramers_paired)
        coeff = sb.transform_scalar_basis(reference.orth.x, basis="ao").c
        pops = fragment_populations(coeff, reference.data.s_ao, reference.ao_layout, sites,
                                    active)
        result = replace(result, coeff=np.ascontiguousarray(coeff),
                                     populations=pops)
        if result.weakest < floor:              # the repair moved a span it should not have
            raise AssertionError(
                "rebuilding the Kramers pairs moved the localization below its floor "
                "({:.3f} < {:.2f}); a site span that is time-reversal closed cannot do that"
                .format(result.weakest, floor))
    if report:
        out.subsection(log, "Fragment localization of the active space")
        result.report()
    return result


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
        A fragment may omit its count (``(atom, l)``) when ``n_active`` fixes the remainder,
        or carry a fourth element ``skip_pairs`` — the ordinal window that names a **double
        shell**: ``[("Ti", "d", 10), ("Ti", "d", 10, 5)]`` is the five lowest d pairs plus
        the next five, two same-``l`` fragments over disjoint pairs.

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
                entries.append([entry[0], entry[1], None, 0])
            elif len(entry) in (3, 4):
                entries.append([entry[0], entry[1], int(entry[2]),
                                int(entry[3]) if len(entry) == 4 else 0])
                n_explicit += int(entry[2])
            else:
                raise ValueError("a fragment is (atom, l, n_spinors), (atom, l), or "
                                 "(atom, l, n_spinors, skip_pairs); got {!r}".format(entry))
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
           solver_options: Optional[Dict] = None, callback=None, deadline=None, signals=None,
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

    Stopping in time
    ----------------
    ``deadline=`` makes the run stop itself while there is still time to write a checkpoint:
    ``"6h"`` or ``21600`` for a budget of your own, ``"slurm"`` for this batch allocation's
    own limit, ``"auto"`` for that limit where there is one and no deadline where there is
    not. ⚠ **The default is no deadline at all** — a cluster with no time limit is an
    ordinary place to run — and an explicitly named source that cannot be read refuses
    rather than leaving the run unprotected. See :mod:`kuiva.util.deadline`; with
    ``checkpoint=`` given, the final write happens *before* the stop.

    ``signals=`` is the other half: a kill that was never announced (``scancel``, a
    preemption, ``SIGTERM`` at the wall) stops the run at the next macro-iteration boundary
    with its checkpoint written, instead of ending it mid-iteration. ``signals=True`` catches
    ``SIGTERM``/``SIGUSR1``/``SIGUSR2``; a sequence names them. ⚠ **Off by default and
    installed only for the duration of this call**, because a library that installs signal
    handlers behind your back breaks embedding, test runners and notebooks. See
    :mod:`kuiva.util.signals`.

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
    from ..util.deadline import Deadline
    from ..util.signals import SignalStop, raise_if_pending, stop_context

    deadline = Deadline.resolve(deadline)
    stopper = SignalStop.resolve(signals)

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
        if deadline is not None:
            deadline.report(log)
        if stopper is not None:
            stopper.report(log)
    # ⚠ Both refusals are here, before anything expensive: a stage that cannot finish is
    # better not started than started and killed. A stop already requested by signal is the
    # sharper of the two -- it means this process is on its way out.
    raise_if_pending("this CASSCF")
    if deadline is not None:
        deadline.assert_room("this CASSCF")

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
                                  deadline=deadline, signals=stopper,
                                  **(checkpoint_options or {}))
        hook = policy.callback
    elif deadline is not None or stopper is not None:
        # ⚠ Without a checkpoint there is nothing to write before the stop, and stopping is
        # worth doing anyway: a run that stops itself exits cleanly and its output file is
        # complete, where one killed at the wall ends mid-line. The warning it raises says
        # that nothing was saved. The signal is the outer of the two, being the one that is
        # already on its way.
        hook = callback
        if deadline is not None:
            hook = deadline.as_callback(chain=hook)
        if stopper is not None:
            hook = stopper.as_callback(chain=hook)

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
    # ⚠ The handlers live exactly as long as the optimization and the previous dispositions
    # come back afterwards, exception or not -- which is what makes an opt-in signal handler
    # something a library may install at all.
    with stop_context(stopper):
        outcome = _casscf(reference.factors, reference.h_one_electron(), orbitals,
                          space.spaces, space.n_elec, n_states=n_states,
                          e_nuc=reference.data.e_nuc, solver=solver, active=space,
                          callback=hook, classifier=classifier, **optimizer_kwargs)
    if policy is not None:
        outcome.checkpoint_path = str(policy.path)
        if report:
            policy.report(log)
    return outcome


def property_matrices(reference: SpinorReference, source, *, comments=(),
                      inactive_tol: Optional[float] = None):
    """``H``, the three magnetic-moment matrices and the three electric-dipole matrices.

    ``source`` is what a calculation returned: a
    :class:`~kuiva.mcscf.casci.CASSCFOutcome` (from :func:`casscf`) or a
    :class:`~kuiva.mcscf.casci.CASCIResult` (from :func:`casci`). Either carries the states,
    the orbitals they were solved at and the solver that owns the excitation map, which is
    everything the moment matrices need.

    Also carries the three electric dipole components ``d`` when the reference was built with
    dipole integrals, which every front-end route does.

    ⚠ **The gauge origin is fixed at ingestion**, not here: ``L`` and ``r`` are both defined
    relative to it and the multireference layer never calls PySCF again. Pass
    ``gauge_origin=`` to :func:`scalar_x2c_reference` to change it.

    Returns a :class:`kuiva.props.dump.PropertyMatrices`. ⚠ Its phases are arbitrary and
    degenerate states mix arbitrarily — compare only through
:meth:`~kuiva.props.dump.PropertyMatrices.analyse`.
    """
    from ..props.dump import property_matrices as _matrices

    ci, coeff, spaces, description = _states_of(source)
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
                  include_l_s: bool = True, include_dipole: bool = True,
                  report: bool = True, **kwargs):
    """Write the property-matrix file — the program's product — and return the matrices.

    A thin composition of :func:`property_matrices` and
    :func:`kuiva.props.dump.write_dump`. The header carries the full Hamiltonian provenance
, the gauge origin and the active space; ⚠ a ``WARNING`` is emitted about the
    treatment of the property operators, every time, and recorded in the file.

    ``include_dipole`` (default ``True``) writes ``d_x, d_y, d_z`` beside ``mu``. ⚠ For a
    **charged** molecule the dipole's diagonal depends on the gauge origin; that warns too and
    the header says so.
    """
    matrices = property_matrices(reference, source, **kwargs)
    if report:
        out.section(log, "Property matrices")
        matrices.report(log)
    matrices.write(path, title=title, include_l_s=include_l_s,
                   include_dipole=include_dipole)
    return matrices


def _states_of(source):
    """``(ci, coeff, spaces, description)`` from a CASSCF outcome or a bare CASCI result.

    One resolution shared by :func:`property_matrices` and :func:`spin_analysis`, since both
    need the same trio and the trap they must not fall into is the same: a quantity built
    from one orbital set and a state set solved at another is Hermitian, plausible and wrong.
    """
    from ..mcscf.casci import CASSCFOutcome

    if isinstance(source, CASSCFOutcome):
        return (source.ci, source.coeff, source.active.spaces,
                source.active.description or source.ci.description)
    return source, source.coeff, source.spaces, source.description


def spin_analysis(reference: SpinorReference, source, *, tol_cm: float = 1.0,
                  energies=None, report: bool = False):
    """``<S^2>`` of a converged spectrum, per degenerate block.

    ``source`` is a :class:`~kuiva.mcscf.casci.CASSCFOutcome` or
    :class:`~kuiva.mcscf.casci.CASCIResult`, as :func:`property_matrices` takes.

    With spin-orbit coupling **off** the block value is ``S(S+1)`` and ``2S+1`` is the term
    multiplicity; with it **on** ``S`` is not conserved and the same number reads as spin
    purity. ⚠ Per *block*, never per state — inside a degenerate block the individual value
    depends on the eigensolver's arbitrary basis. See :mod:`kuiva.props.spin`.

    ⚠ ``energies=`` overrides the spectrum the **blocking** is taken from, and exists for one
    case: a NEVPT2-corrected file, whose ``H`` is the perturbed spectrum while its states are
    still the CASSCF ones. The ``<S^2>`` values belong to those states either way; what would
    otherwise go wrong is that this and :func:`kuiva.props.multiplet.analyse_spectrum` would
    group *different* spectra into blocks and the two could not be paired at all. Passing the
    same array to both makes them the same blocking by construction rather than by luck.
    """
    from ..props.spin import spin_analysis as _spin

    ci, coeff, spaces, _ = _states_of(source)
    if coeff is None or spaces is None:
        raise ValueError(
            "this CASCI result does not know which orbitals or which orbital-space partition "
            "it belongs to, so <S^2> cannot be built from it. Use api.casci/api.casscf, "
            "which record both")
    result = _spin(ci.solver, coeff, spaces, reference.data.s_ao,
                   ci.total_energies if energies is None else energies,
                   vectors=ci.vectors, tol_cm=tol_cm, has_soc=reference.data.has_soc)
    if report:
        result.report(log)
    return result


def assign_states(reference: SpinorReference, source, *, matrices=None,
                  tol_cm: float = 1.0, report: bool = True):
    """Offer a ``^{2S+1}L_J`` (or ``^{2S+1}L``) label for each degenerate block.

    Assembles the evidence — the degeneracy pattern and g values
    (:func:`kuiva.props.multiplet.analyse_spectrum`), the block ``<S^2>``
    (:func:`spin_analysis`) and the symmetry labels where the run carries them — and hands
    them to :func:`kuiva.props.assign.assign_terms`.

    ⚠ **Every label it returns is an inference**, printed with the measurements behind it and
    a fit residual, and withheld as ``"?"`` where they do not add up. It is deliberately its
    own report and never a column of the state table.

    ``matrices`` is a finished :class:`~kuiva.props.dump.PropertyMatrices`; without one, the
    moment matrices are built here when the Hamiltonian carries spin-orbit coupling (they are
    what the Landé inversion needs) and skipped when it does not.
    """
    from ..props.assign import assign_terms
    from ..props.multiplet import analyse_spectrum

    ci, _, _, _ = _states_of(source)
    if matrices is None and reference.data.has_soc:
        matrices = property_matrices(reference, source)
    # ⚠ One spectrum, blocked once. With a NEVPT2-corrected `matrices` its `H` is the
    # perturbed spectrum while the states are still the CASSCF ones, so taking the blocking
    # from the CASSCF energies here would group a different set of levels than
    # `matrices.analyse()` does and the two could not be paired at all.
    energies = None if matrices is None else matrices.energies
    spin = spin_analysis(reference, source, tol_cm=tol_cm, energies=energies)
    multiplets = (matrices.analyse(tol_cm=tol_cm) if matrices is not None
                  else analyse_spectrum(ci.total_energies, tol_cm=tol_cm))
    result = assign_terms(multiplets, spin, irreps=ci.multiplets or ci.irreps)
    if report:
        spin.report(log)
        result.report(log)
    return result


def avas_active_space(reference: SpinorReference, *, atom, l, coeff=None,
                      occupation=None, report: bool = True, **kwargs):
    """An active space (and the rotated orbitals) from an AVAS projection.

    The route to an active space when the target orbitals are **covalent mixtures** and no
    canonical orbital carries enough ``(atom, l)`` character to be selected by
    :func:`active_space_for`. Projects onto the free-atom orbitals the front end computed
    with ``atomic_reference=True`` and rotates *within* the occupied and virtual spaces, so
    the reference density does not move. ``n_shells=2`` asks for the **double shell**.

    Defaults to the reference's own guess spinors and occupations; pass ``coeff`` (AO basis)
    with ``occupation=`` to project an already-optimized set. Returns a
    :class:`~kuiva.mcscf.avas.AVASResult`, whose ``coeff`` — not the input — is what a
    subsequent CASSCF must start from. See :mod:`kuiva.mcscf.avas` for every option and for
    what this implementation does differently from the published method.

    ⚠ **An AVAS space carries no symmetry labels**, deliberately. The labels belong to the
    *guess* spinors and AVAS has rotated them; carrying them across would attach a label to
    an orbital it no longer describes, and per-irrep state selection built on that would be
    counting the wrong thing. Per-irrep ``n_states`` is therefore unavailable after AVAS —
    select the space by character if the symmetry labels are what the run needs.
    """
    from ..mcscf.avas import avas as _avas

    if coeff is None:
        coeff = reference.spinors_in_ao()
        if occupation is None:
            occupation = reference.spinors.occ
    if occupation is None:
        raise ValueError("give occupation= alongside coeff=: AVAS rotates within groups of "
                         "equal occupation and cannot infer them from the orbitals")
    result = _avas(coeff, reference.data.s_ao, reference.ao_layout,
                   reference.data.atomic_reference, reference.data.nelec_total,
                   atom=atom, l=l, occupation=occupation, **kwargs)
    if report:
        result.report(log)
    return result


__all__ = ["Molecule", "ScalarX2CData", "SpinorReference", "scalar_x2c_reference",
           "spinor_reference", "build_mole", "memory_plan",
           "project_to_basis", "projected_active_space",
           "active_space_for", "avas_active_space", "localize_active_space",
           "casci", "casscf",
           "property_matrices", "property_dump", "spin_analysis", "assign_states"]
