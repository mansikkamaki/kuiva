"""Primary Python API (input is a Python API, not a text format).

A user constructs a :class:`Molecule` (geometry + per-atom basis by registry name) and calls
:func:`scalar_x2c_reference` to obtain the ingested scalar-relativistic X2C reference that the
multireference layer consumes. This is the single public entry point for the front-end.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..basis import registry as reg
from ..integrals.transform import DEFAULT_CHOLESKY_TOL, ThreeIndexAO
from ..orth.canonical import DEFAULT_THRESHOLD, OrthonormalBasis
from ..spinor.expand import SpinorBasis
from ..util import output as out
from ..util import resources as res
from ..util.logging import get_logger
from .pyscf_bridge import ScalarX2CData, build_mole, memory_plan, run_scalar_x2c

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
        Either one registry family name applied to all atoms, or a ``{symbol: family}`` map.
    charge : int
    spin : int
        ``2S`` (number of unpaired electrons), PySCF convention.
    unit : str
        ``"Angstrom"`` (default) or ``"Bohr"``.
    """
    atoms: List[Atom]
    basis: BasisSpec
    charge: int = 0
    spin: int = 0
    unit: str = "Angstrom"

    def __post_init__(self) -> None:
        # Normalise symbols and validate the basis assignment eagerly (fail fast).
        self.atoms = [(s.capitalize(), tuple(float(x) for x in xyz)) for s, xyz in self.atoms]
        atom_basis = self._atom_basis()
        report = reg.check_consistency(atom_basis, emit=True)
        if not report.ok:
            raise ValueError("invalid basis assignment:\n  " + "\n  ".join(report.errors))

    def _atom_basis(self) -> Dict[str, str]:
        syms = sorted({s for s, _ in self.atoms})
        if isinstance(self.basis, str):
            return {s: self.basis for s in syms}
        ab = {s.capitalize(): b for s, b in self.basis.items()}
        missing = [s for s in syms if s not in ab]
        if missing:
            raise ValueError(f"no basis assigned for atom(s) {missing}")
        return ab

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
                         decoupling_options: Optional[Dict[str, object]] = None,
                         conv_tol: float = 1e-10, max_cycle: int = 200,
                         memory_gb: Optional[float] = None, n_active: Optional[int] = None,
                         cholesky_tol: float = DEFAULT_CHOLESKY_TOL,
                         orbit_pivots: bool = True, one_centre: bool = True,
                         gauge_origin=None, property_picture_change: bool = False,
                         anomaly_picture_change: bool = False,
                         atomic_reference: bool = False,
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
    return run_scalar_x2c(molecule, reference=reference, fitting=fitting, auxbasis=auxbasis,
                          with_soc=with_soc, method=method, x2c_approx=x2c_approx,
                          screening=screening, screening_options=screening_options,
                          decoupling_options=decoupling_options, conv_tol=conv_tol,
                          max_cycle=max_cycle, memory_gb=memory_gb, n_active=n_active,
                          cholesky_tol=cholesky_tol, orbit_pivots=orbit_pivots,
                          one_centre=one_centre, gauge_origin=gauge_origin,
                          property_picture_change=property_picture_change,
                          anomaly_picture_change=anomaly_picture_change,
                          atomic_reference=atomic_reference, verbose=verbose)


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
        return active
    if (active is None) == (character is None):
        raise ValueError(
            "give exactly one of active=[spinor indices] or character=(atom, l) with "
            "n_active=; an active space stated only as an index window is not a definition "
            "another program can reproduce ")
    n_orb = reference.nspinor
    n_elec_total = reference.data.nelec_total
    if active is not None:
        return active_space(active, n_orb, n_elec_total, n_active_elec=n_active_elec,
                            kramers_paired=reference.spinors.kramers_paired)
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
        return active_space_by_characters(
            reference.spinors_in_ao(), reference.data.s_ao, reference.ao_layout,
            n_elec_total, fragments=[tuple(e) for e in entries],
            n_active_elec=n_active_elec, occupation=reference.spinors.occ, **kwargs)

    if n_active is None or int(n_active) % 2 != 0:
        raise ValueError("character selection needs an even n_active (whole Kramers pairs, "
                         "whole Kramers pairs only); got {!r}".format(n_active))
    atom, l = character
    return active_space_by_character(
        reference.spinors_in_ao(), reference.data.s_ao, reference.ao_layout, n_elec_total,
        atom=atom, l=l, n_pairs=int(n_active) // 2, n_active_elec=n_active_elec,
        occupation=reference.spinors.occ, **kwargs)


def casci(reference: SpinorReference, *, active=None, character=None,
          n_active: Optional[int] = None, n_active_elec: Optional[int] = None,
          n_states: int = 1, weights=None, coeff: Optional[np.ndarray] = None,
          report: bool = True, **solver_kwargs):
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
    result = _casci(reference.factors, reference.h_one_electron(), orbitals, space.spaces,
                    space.n_elec, n_states=n_states, e_nuc=reference.data.e_nuc,
                    weights=weights, report=report, **solver_kwargs)
    result.description = space.description
    return result


def casscf(reference: SpinorReference, *, active=None, character=None,
           n_active: Optional[int] = None, n_active_elec: Optional[int] = None,
           n_states: int = 1, weights=None, coeff: Optional[np.ndarray] = None,
           checkpoint=None, restart=None, checkpoint_options: Optional[Dict] = None,
           solver_options: Optional[Dict] = None, callback=None, report: bool = True,
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

    solver = FullCISolver(space.spaces.n_active, space.n_elec, n_states=n_states,
                          weights=weights, **(solver_options or {}))
    if resumed is not None:
        solver.set_guess(resumed.ci_vectors)
        optimizer_kwargs.update(resumed.optimizer_kwargs())

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

    optimizer_kwargs.setdefault("report", report)
    outcome = _casscf(reference.factors, reference.h_one_electron(), orbitals, space.spaces,
                      space.n_elec, n_states=n_states, e_nuc=reference.data.e_nuc,
                      solver=solver, active=space, callback=hook, **optimizer_kwargs)
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
           "active_space_for", "casci", "casscf", "property_matrices", "property_dump"]
