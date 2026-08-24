"""The high-level class API: a production calculation as a short script of stage objects.

::

    ScalarSCF -> Reference -> (CheapCI) -> CASSCF -> (NEVPT2) -> PropertyDump
                                                             \\-> PseudospinExport

**The uniform contract, which every class here obeys** — learn one, guess the rest:

* the **constructor takes the finished upstream stage** plus keyword options, and validates
  everything it can immediately: a misspelled option, an impossible active space or a missing
  prerequisite fails at construction, not an hour into the run. An upstream stage that has
  not been ``run()`` is refused, so scripts read linearly, one finished stage per line;
* :meth:`~_Stage.run` is the **only expensive call**. It executes the stage, stores results
  as plain attributes, and returns ``self`` — so ``cas = CASSCF(ref, ...).run()`` — and a
  second call returns the same object without recomputing;
* :meth:`~_Stage.summary` returns a short plain-text block of the headline results;
* results are plain attributes, and the underlying low-level objects stay reachable
  (``.data``, ``.reference``, ``.outcome``, ``.result``): this module is a thin layer **over**
  :mod:`kuiva.interface.api` and the module drivers, never a restructuring of them, and the
  low-level API remains available and unchanged underneath.

Each stage keeps a pointer to its upstream stage, so a downstream stage finds what it needs
through the chain — ``PropertyDump(cas, "file.props")`` needs nothing else.

The three shapes this module was designed on (each about a dozen lines)::

    # a lanthanide free ion
    ion = kuiva.Molecule([("Er", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", charge=3, spin=3)
    scf = kuiva.ScalarSCF(ion, memory_gb=16.0).run()
    ref = kuiva.Reference(scf).run()
    cas = kuiva.CASSCF(ref, character=("Er", "f"), n_active=14, n_active_elec=11,
                       n_states=16, mode="second-order", checkpoint="er.h5").run()
    kuiva.PropertyDump(cas, "er_ion.props", title="Er3+ free ion").run()

    # a transition-metal complex, CASSCF + NEVPT2
    pre = kuiva.CheapCI(ref, character=("Ti", "d"), n_active=10, n_active_elec=1).run()
    cas = kuiva.CASSCF(pre, n_states=10).run()      # space and orbitals inherited from pre
    pt  = kuiva.NEVPT2(cas, frozen_core=-10.0).run()
    kuiva.PropertyDump(pt, "ticl3.props").run()     # corrected H; protocol in the header

    # a DMRG-CASSCF and the pseudospin export
    cas = kuiva.CASSCF(pre, solver="dmrg", n_states=4, graph="mutual-information",
                       solver_options=dict(max_bond=128, adaptive=True)).run()
    kuiva.PseudospinExport(cas, "dimer.psd", rule="dimension", dims=2).run()
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util.logging import get_logger
from .api import (Molecule, SpinorReference as _SpinorData, active_space_for,
                  project_to_basis as _project_to_basis, projected_active_space,
                  property_matrices as _property_matrices, scalar_x2c_reference,
                  spinor_reference)

log = get_logger(__name__)

__all__ = ["ScalarSCF", "Reference", "CheapCI", "CASSCF", "NEVPT2", "PropertyDump",
           "PseudospinExport"]


def _allowed_options(*funcs, exclude: Sequence[str] = ()) -> set:
    """The union of the keyword parameters of ``funcs``, minus ``exclude``.

    Used for eager validation: a stage forwards its ``**options`` to these functions at
    ``run()`` time, so any key none of them accepts is a construction-time ``TypeError``
    rather than a failure an expensive stage into the run.
    """
    allowed = set()
    for func in funcs:
        for name, p in inspect.signature(func).parameters.items():
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                allowed.add(name)
    return allowed - set(exclude)


def _check_options(options: Dict[str, Any], allowed: set, where: str) -> None:
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise TypeError("{} got unexpected option(s): {}. Valid options: {}"
                        .format(where, ", ".join(unknown), ", ".join(sorted(allowed))))


class _Stage:
    """Base of every stage: the run-once / eager-validation / summary contract."""

    def __init__(self) -> None:
        self._ran = False

    def run(self) -> "_Stage":
        """Execute the stage (the only expensive call) and return ``self``.

        Idempotent: a stage records one calculation, so a second call returns the same
        finished object without recomputing.
        """
        if not self._ran:
            self._execute()
            self._ran = True
        return self

    def _execute(self) -> None:                              # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def ran(self) -> bool:
        return self._ran

    def _check_ran(self) -> None:
        if not self._ran:
            raise RuntimeError("{0}.run() has not been called yet; results exist only on a "
                               "finished stage".format(type(self).__name__))

    @staticmethod
    def _finished(upstream, kinds, what: str):
        """Validate that ``upstream`` is a finished stage of one of ``kinds``."""
        if not isinstance(upstream, kinds):
            names = " or ".join(k.__name__ for k in
                                (kinds if isinstance(kinds, tuple) else (kinds,)))
            raise TypeError("{} takes a finished {} stage; got {!r}"
                            .format(what, names, type(upstream).__name__))
        if not upstream.ran:
            raise ValueError("the upstream {} stage has not been run; call .run() on every "
                             "stage before handing it downstream (each constructor validates "
                             "against the upstream *results*)"
                             .format(type(upstream).__name__))
        return upstream

    def _summary_entries(self) -> List[Tuple[str, str]]:     # pragma: no cover - abstract
        raise NotImplementedError

    def summary(self) -> str:
        """A short plain-text block of the stage's headline results."""
        self._check_ran()
        entries = self._summary_entries()
        width = max(len(name) for name, _ in entries)
        lines = [type(self).__name__] + ["  {:<{w}} : {}".format(name, value, w=width)
                                         for name, value in entries]
        return "\n".join(lines)


# --- 1. the scalar-relativistic SCF ---------------------------------------------------------

class ScalarSCF(_Stage):
    """The scalar-X2C SCF front end: a :class:`~kuiva.interface.api.Molecule` in, an ingested
    :class:`~kuiva.interface.pyscf_bridge.ScalarX2CData` out.

    Options are those of :func:`kuiva.interface.api.scalar_x2c_reference` — ``method``
    (``"X2C-AMF"`` is the default Hamiltonian), ``screening``, ``reference``, ``memory_gb``,
    ``gauge_origin``, ``conv_tol``, ... — validated eagerly by name.

    After :meth:`run`: :attr:`data`, :attr:`energy` [Eh], :attr:`converged`.
    """

    _EXCLUDE = ("molecule",)

    def __init__(self, molecule: Molecule, **options) -> None:
        super().__init__()
        if not isinstance(molecule, Molecule):
            raise TypeError("ScalarSCF takes a kuiva Molecule; got {!r}"
                            .format(type(molecule).__name__))
        _check_options(options, _allowed_options(scalar_x2c_reference,
                                                 exclude=self._EXCLUDE), "ScalarSCF")
        reference = options.get("reference", "auto")
        if reference not in ("auto", "rhf", "rohf", "uhf"):
            raise ValueError("reference must be 'auto', 'rhf', 'rohf' or 'uhf'; got {!r}"
                             .format(reference))
        self.molecule = molecule
        self.options = dict(options)

    def _execute(self) -> None:
        self.data = scalar_x2c_reference(self.molecule, **self.options)

    @property
    def energy(self) -> float:
        """Scalar X2C SCF total energy [Eh]."""
        self._check_ran()
        return float(self.data.e_scf)

    @property
    def converged(self) -> bool:
        self._check_ran()
        return bool(self.data.converged)

    def _summary_entries(self):
        soc = self.data.soc
        return [
            ("E(SCF) [Eh]", out.E_FMT.format(self.energy)),
            ("converged", str(self.converged)),
            ("reference", self.data.reference),
            ("hamiltonian", "spin-free" if soc is None else soc.provenance()["method"]),
        ]


# --- 2. the multireference starting point ----------------------------------------------------

class Reference(_Stage):
    """Orthonormal working basis, Kramers-paired spinor guess and factorized integrals.

    Wraps :func:`kuiva.interface.api.spinor_reference` on a finished :class:`ScalarSCF`; the
    result — everything the multireference layer starts from — is :attr:`reference`, a
    :class:`kuiva.interface.api.SpinorReference` container. (The stage is named ``Reference``
    precisely so the two are not confused: the stage runs the step, the container holds the
    data.)

    After :meth:`run`: :attr:`reference`, :attr:`nspinor`, and the inspection helpers
    :meth:`population_analysis` / :meth:`write_molden`.
    """

    _EXCLUDE = ("molecule_or_data", "memory_gb")

    def __init__(self, scf: ScalarSCF, **options) -> None:
        super().__init__()
        self.scf = self._finished(scf, ScalarSCF, "Reference")
        _check_options(options, _allowed_options(spinor_reference, exclude=self._EXCLUDE),
                       "Reference")
        self.options = dict(options)

    def _execute(self) -> None:
        self.reference = spinor_reference(self.scf.data, **self.options)

    @property
    def nspinor(self) -> int:
        self._check_ran()
        return self.reference.nspinor

    def population_analysis(self, **kwargs):
        """Loewdin populations of a spinor set; see
        :meth:`kuiva.interface.api.SpinorReference.population_analysis`."""
        self._check_ran()
        return self.reference.population_analysis(**kwargs)

    def atomic_reference_charges(self, **kwargs):
        """Atomic charges in the free-atom reference partition (needs the scalar SCF stage
        to have run with ``atomic_reference=True``); see
        :meth:`kuiva.interface.api.SpinorReference.atomic_reference_charges`."""
        self._check_ran()
        return self.reference.atomic_reference_charges(**kwargs)

    def write_molden(self, path, **kwargs):
        """Spinor densities to a molden file; see
        :meth:`kuiva.interface.api.SpinorReference.write_molden`."""
        self._check_ran()
        return self.reference.write_molden(path, **kwargs)

    def _summary_entries(self):
        r = self.reference
        return [
            ("spinors", str(r.nspinor)),
            ("working-basis columns dropped",
             str(int(2 * r.orth.x.shape[0] - r.nspinor) // 2)),
            ("two-electron factorization",
             "{} ({} vectors)".format(r.factors.origin, int(r.factors.naux))),
        ]


def _resolve_space(reference: Reference, *, active, character, n_active, n_active_elec,
                   threshold, what: str):
    """Resolve a stage's active-space request against a finished :class:`Reference`."""
    if active is None and character is None:
        raise ValueError(
            "{} needs an active space: give active=[spinor indices] or "
            "character=(atom, l) with n_active= (or a list of (atom, l, n_spinors) "
            "fragments); an active space is a physical statement, so there is no default"
            .format(what))
    return active_space_for(reference.reference, active=active, character=character,
                            n_active=n_active, n_active_elec=n_active_elec,
                            threshold=threshold)


# --- 3. the cheap pre-optimization -----------------------------------------------------------

class CheapCI(_Stage):
    """The cheap-CI pre-optimization: raw spinor guess in, physical active orbitals out.

    Wraps :func:`kuiva.mcscf.preopt.preoptimize`. Its two products feed the stages after it:
    the rotated orbitals (natural in the active space) start the CASSCF, and the
    entanglement data seeds the tensor-network topology. A :class:`CASSCF` built on this
    stage inherits both, plus the active space stated here, unless told otherwise.

    ⚠ The pre-optimizer's total energy means nothing and is deliberately not an attribute;
    what it claims is that the *occupations* are converged enough to select orbitals by.

    Exact Kramers pairing is restored on the orbitals before they leave this stage — the
    truncated cheap CI drifts off it and the state-averaging gate downstream assumes it — so
    chaining a pre-optimization into a CASSCF through this stage needs no repair of its own.

    After :meth:`run`: :attr:`orbitals`, :attr:`occupations`, :attr:`natural_occupation`,
    :attr:`entropy`, :attr:`mutual_information`, :meth:`suggested_active`,
    :meth:`dmrg_ordering`, and the full :class:`~kuiva.mcscf.preopt.PreoptResult` as
    :attr:`result`.
    """

    _EXCLUDE = ("factors", "h_ao", "c_spinor", "spaces", "n_active_elec", "e_nuc",
                "h_eff", "eri", "n_elec")

    def __init__(self, reference: Reference, *, active=None, character=None,
                 n_active: Optional[int] = None, n_active_elec: Optional[int] = None,
                 threshold: Optional[float] = None, **options) -> None:
        super().__init__()
        from ..mcscf.preopt import cheap_ci, preoptimize
        self.reference_stage = self._finished(reference, Reference, "CheapCI")
        _check_options(options, _allowed_options(preoptimize, cheap_ci,
                                                 exclude=self._EXCLUDE), "CheapCI")
        self.space = _resolve_space(reference, active=active, character=character,
                                    n_active=n_active, n_active_elec=n_active_elec,
                                    threshold=threshold, what="CheapCI")
        self.options = dict(options)

    def _execute(self) -> None:
        from ..mcscf.preopt import preoptimize
        from ..spinor.expand import nearest_kramers_paired, spin_block_diagonal, time_reverse
        ref = self.reference_stage.reference
        self.result = preoptimize(ref.factors, ref.h_one_electron(), ref.spinors_in_ao(),
                                  self.space.spaces, self.space.n_elec,
                                  e_nuc=ref.data.e_nuc, **self.options)
        # ⚠ Restore exact Kramers pairing before anything downstream consumes the orbitals.
        # The cheap CI's truncated determinant space is not closed under time reversal, so
        # the orbitals it optimizes drift off pairing — legitimately, it is a *cheap* stage —
        # while every consumer of this stage (the state-averaging gate, a contiguous-pair
        # active space) assumes the pairing convention exactly.
        x2 = spin_block_diagonal(ref.orth.x)
        c_work = x2.conj().T @ spin_block_diagonal(ref.data.s_ao) @ self.result.coeff
        deviation = float(np.max(np.abs(
            1.0 - np.abs(np.sum(np.conj(c_work[:, 1::2]) * time_reverse(c_work[:, ::2]),
                                axis=0)))))
        spaces = self.space.spaces
        c_work = nearest_kramers_paired(c_work,
                                        (spaces.inactive, spaces.active, spaces.virtual))
        log.debug("Kramers pairing restored on the pre-optimized orbitals (worst partner "
                  "deviation before repair: %.2e)", deviation)
        self.orbitals = np.ascontiguousarray(x2 @ c_work)
        self.occupations = self.result.orbital_occupation
        self.natural_occupation = self.result.natural_occupation
        self.entropy = self.result.entropy
        self.mutual_information = self.result.mutual_information

    def suggested_active(self, **kwargs) -> np.ndarray:
        """Fractionally occupied active spinors — ⚠ a **lower bound** on the active space,
        to be combined with orbital character and near-degeneracy; see
        :meth:`kuiva.mcscf.preopt.PreoptResult.suggest_active_space`."""
        self._check_ran()
        return self.result.suggest_active_space(**kwargs)

    def dmrg_ordering(self) -> np.ndarray:
        """Fiedler ordering of the active spinors for a path network."""
        self._check_ran()
        return self.result.dmrg_ordering()

    def _summary_entries(self):
        occ = ", ".join("{:.3f}".format(x) for x in self.occupations)
        return [
            ("active space", "CAS({}, {})  {}".format(self.space.n_elec,
                                                      self.space.n_active,
                                                      self.space.description)),
            ("converged (occupations)", str(self.result.converged)),
            ("orbital occupations", occ),
            ("suggested active spinors", str(self.suggested_active().tolist())),
        ]


# --- 4. the CASSCF ---------------------------------------------------------------------------

class CASSCF(_Stage):
    """State-averaged two-component CASSCF — the calculation this program exists for.

    ``upstream`` is a finished :class:`Reference` or :class:`CheapCI`. Built on a
    :class:`CheapCI`, the stage starts from its rotated orbitals and — when no space is
    requested here — inherits its active space unchanged.

    ``solver`` picks the CI method behind the **same** orbital optimizer:

    ``"ci"`` (default)
        Conventional complex determinant CI through :func:`kuiva.interface.api.casscf`,
        including ``checkpoint=``/``restart=`` and the state-average boundary diagnostic at
        both ends. ``solver_options`` go to :class:`~kuiva.mcscf.casci.FullCISolver`.
    ``"dmrg"``
        The in-house tree tensor network solver (:class:`~kuiva.dmrg.DMRGSolver`);
        ``solver_options`` must carry ``max_bond`` and may carry ``adaptive=True``, which
        routes the optimization through the event-gated driver
        (:func:`~kuiva.mcscf.events.optimize_orbitals_events`) so network-topology changes
        are adopted only when they lower the energy at fixed integrals. ``graph=`` seeds the
        topology: a :class:`~kuiva.dmrg.NetworkGraph`, or ``"mutual-information"`` /
        ``"fiedler"`` to build one from a :class:`CheapCI` upstream. The converged orbitals
        are finished with one warm solve carrying the boundary diagnostic.
        ⚠ Checkpoint/restart of the network state is not wired into this layer; ask for it
        or drive :mod:`kuiva.dmrg` directly.

    Remaining keyword options go to the orbital optimizer (``mode``, ``max_iter``,
    ``conv_grad``, ``conv_energy``, ``max_step``, and for the event-gated driver ``tau``,
    ``event_interval``, ...). ``mode="second-order"`` is the right explicit choice for a
    heavy element or a large state average.

    Starting from a smaller (or larger) basis
    -----------------------------------------
    ``project_from=`` takes a **finished stage of the same molecule in a different basis** —
    a :class:`CASSCF`, a :class:`CheapCI` or a plain :class:`Reference` — and starts this run
    from that stage's orbitals projected onto this basis
    (:func:`kuiva.interface.api.project_to_basis`). That is the production route to a
    large-basis CASSCF: converge it in a small basis, where the active orbitals are cheap to
    optimize and easy to identify, and continue here. The reverse (large onto small) is the
    same call.

    ⚠ **The active space comes across with the orbitals and may not be restated here.** It
    was chosen once, against the orbitals being carried; re-selecting it against this
    reference's guess orbitals would silently define a different calculation. A projection
    from a bare :class:`Reference`, which has no active space, is the one case where the
    space *is* stated here — and it is then resolved against the **source** reference, for
    the same reason.

    ⚠ A projection **replaces** the pre-optimization rather than following it: what it hands
    over is already optimized active orbitals. ``project_from=`` therefore needs a
    :class:`Reference` upstream, and does not combine with ``restart=`` (which brings its own
    orbitals and its own space). ``projection=dict(...)`` passes options through to
    :func:`~kuiva.interface.api.project_to_basis` — ``carry`` (``"active"``, the default, or
    ``"all"``), ``scheme`` (``"blocked"``, ``"symmetric"``, ``"gram-schmidt"``) and
    ``repair_pairing``; that function's docstring is where each is explained and
    :mod:`kuiva.orth.project` is where the defaults are argued.

    With point-group symmetry on (``point_group=`` at the front end), ``n_states`` may be a
    mapping ``{irrep: n}`` instead of a count — each irrep is then solved in its own sector of
    the determinant space, which is a request "lowest n" cannot express — and
    ``preserve_symmetry=True`` masks inter-irrep orbital rotations so the labels are still
    exact at convergence. ⚠ The mask is a **constraint**: it converges to the lowest
    *symmetric* solution, which is not the global one where the symmetry is spontaneously
    broken.

    After :meth:`run`: :attr:`energy`, :attr:`energies` (total state energies [Eh]),
    :attr:`coeff`, :attr:`converged`, :attr:`active`, :attr:`solver`, plus per-solver
    results (``"ci"``: :attr:`outcome`, :attr:`boundary`, :attr:`boundary_initial`;
    ``"dmrg"``: :attr:`orbital`, :attr:`events`, :attr:`boundary_gap_cm`, :attr:`graph`) and,
    with ``project_from=``, :attr:`projection` — the
    :class:`~kuiva.orth.project.BasisProjection` carrying the orbitals it started from and the
    invariants that say whether the projection was worth using.
    """

    _GRAPH_CHOICES = ("mutual-information", "fiedler")

    def __init__(self, upstream, *, active=None, character=None,
                 n_active: Optional[int] = None, n_active_elec: Optional[int] = None,
                 threshold: Optional[float] = None, n_states=1, weights=None,
                 solver: str = "ci", solver_options: Optional[Dict[str, Any]] = None,
                 graph=None, checkpoint=None, restart=None,
                 checkpoint_options: Optional[Dict[str, Any]] = None,
                 callback: Optional[Callable[[dict], Optional[bool]]] = None,
                 preserve_symmetry: bool = False, project_from=None,
                 projection: Optional[Dict[str, Any]] = None,
                 report: bool = True, **optimizer_options) -> None:
        super().__init__()
        self._finished(upstream, (Reference, CheapCI), "CASSCF")
        self.upstream = upstream
        self.reference_stage = (upstream if isinstance(upstream, Reference)
                                else upstream.reference_stage)
        if solver not in ("ci", "dmrg"):
            raise ValueError("solver must be 'ci' or 'dmrg'; got {!r}".format(solver))
        self.solver_kind = solver
        #: ``n_states`` is either a count or, with point-group labels present, a per-irrep
        #: mapping ``{irrep: n}``. The two are forms of one argument; the mapping's total is
        #: what :attr:`n_states` reports so every consumer of the count still works.
        self.state_request = dict(n_states) if isinstance(n_states, dict) else None
        self.n_states = (sum(int(v) for v in n_states.values())
                         if self.state_request is not None else int(n_states))
        self.preserve_symmetry = bool(preserve_symmetry)
        if self.preserve_symmetry and solver != "ci":
            raise ValueError("preserve_symmetry= constrains the shared orbital optimizer and "
                             "is wired for solver='ci'; drive kuiva.mcscf directly with "
                             "labels= for a network solver")
        self.weights = weights
        self.solver_options = dict(solver_options or {})
        self.checkpoint, self.restart = checkpoint, restart
        self.checkpoint_options = checkpoint_options
        self.callback, self.report = callback, bool(report)
        self.optimizer_options = dict(optimizer_options)

        self.project_from = project_from
        self.projection_options = dict(projection or {})
        if project_from is None and self.projection_options:
            raise ValueError("projection= configures project_from=, which was not given")
        if project_from is not None:
            _check_options(self.projection_options,
                           _allowed_options(_project_to_basis,
                                            exclude=("source", "target", "coeff", "space",
                                                     "report")),
                           "CASSCF projection")

        # -- the active space and the starting orbitals, resolved now (fail fast) ----------
        requested = active is not None or character is not None
        if project_from is not None:
            self._setup_projection(upstream, requested, active=active, character=character,
                                   n_active=n_active, n_active_elec=n_active_elec,
                                   threshold=threshold, restart=restart)
        elif restart is not None:
            if solver == "dmrg":
                raise ValueError("restart= is the conventional-CI checkpoint route; the "
                                 "network state is not checkpointed by this layer")
            if not Path(restart).exists():
                raise ValueError("restart checkpoint {!r} does not exist".format(restart))
        if project_from is None:
            self.space = (_resolve_space(self.reference_stage, active=active,
                                         character=character, n_active=n_active,
                                         n_active_elec=n_active_elec, threshold=threshold,
                                         what="CASSCF")
                          if requested else None)
            if self.space is None and restart is None:
                if isinstance(upstream, CheapCI):
                    self.space = upstream.space
                else:
                    _resolve_space(self.reference_stage, active=None, character=None,
                                   n_active=None, n_active_elec=None, threshold=None,
                                   what="CASSCF")            # raises with the guidance
            self._orbitals = upstream.orbitals if isinstance(upstream, CheapCI) else None
        else:
            self._orbitals = None                            # built by run(), from the plan

        # -- per-solver eager validation ----------------------------------------------------
        from ..mcscf.orbopt import optimize_orbitals
        if solver == "ci":
            if graph is not None:
                raise ValueError("graph= is a tensor-network option; it needs solver='dmrg'")
            from ..mcscf.casci import FullCISolver, casscf as _mcscf_casscf
            _check_options(self.solver_options,
                           _allowed_options(FullCISolver.__init__,
                                            exclude=("self", "n_spinor", "n_elec",
                                                     "n_states", "weights")),
                           "CASSCF solver_options (FullCISolver)")
            allowed = _allowed_options(
                optimize_orbitals, _mcscf_casscf,
                exclude=("factors", "h_ao", "c_spinor", "spaces", "ci_solver", "n_elec",
                         "e_nuc", "n_states", "weights", "solver", "active",
                         "solver_options", "callback", "report", "optimizer_state",
                         "start_iteration", "space_key", "history"))
            _check_options(self.optimizer_options, allowed, "CASSCF (orbital optimizer)")
        else:
            from ..dmrg import DMRGSolver
            from ..mcscf.events import optimize_orbitals_events
            if "max_bond" not in self.solver_options:
                raise ValueError(
                    "solver='dmrg' needs solver_options=dict(max_bond=...): an uncapped "
                    "tree state allocates charge-sector-maximal bond dimensions")
            if checkpoint is not None:
                raise ValueError("checkpoint= stores CI vectors and is the conventional-CI "
                                 "route; the network state is not checkpointed by this "
                                 "layer")
            _check_options(self.solver_options,
                           _allowed_options(DMRGSolver.__init__,
                                            exclude=("self", "n_elec", "n_roots", "weights",
                                                     "graph", "initial_state")),
                           "CASSCF solver_options (DMRGSolver; the topology is the "
                           "stage-level graph= option)")
            self._adaptive = bool(self.solver_options.get("adaptive", False))
            driver = optimize_orbitals_events if self._adaptive else optimize_orbitals
            allowed = _allowed_options(
                driver, exclude=("factors", "h_ao", "c_spinor", "spaces", "ci_solver",
                                 "e_nuc", "callback", "report", "optimizer_state",
                                 "start_iteration", "space_key", "history"))
            _check_options(self.optimizer_options, allowed, "CASSCF (orbital optimizer)")
            if isinstance(graph, str):
                if graph not in self._GRAPH_CHOICES:
                    raise ValueError("graph= must be a NetworkGraph or one of {}; got {!r}"
                                     .format(self._GRAPH_CHOICES, graph))
                if not isinstance(upstream, CheapCI):
                    raise ValueError("graph={!r} builds the topology from the cheap CI's "
                                     "entanglement, so it needs a CheapCI upstream"
                                     .format(graph))
            self.graph_request = graph

    # -- starting from another basis ----------------------------------------------------

    def _setup_projection(self, upstream, requested, *, active, character, n_active,
                          n_active_elec, threshold, restart) -> None:
        """Resolve everything a ``project_from=`` run needs, without doing the projection.

        The projection itself costs a one-electron integral over two bases and an
        ``O(nao^3)`` orthonormalization, so it belongs in ``run()`` like every other
        expensive thing. What has to happen *here* is the part that can be wrong: which
        stage is being projected from, whether the two are really the same molecule in two
        bases, and where the active space lands in the target's numbering — which is pure
        integer bookkeeping (:func:`kuiva.orth.project.plan_columns`) and needs no integrals.
        """
        from ..orth.project import plan_columns

        self._finished(self.project_from, (Reference, CheapCI, CASSCF), "CASSCF project_from")
        if restart is not None:
            raise ValueError(
                "project_from= and restart= are two different ways to supply the starting "
                "orbitals and the active space; a restart continues an interrupted run in "
                "its own basis, so give one or the other")
        if not isinstance(upstream, Reference):
            raise ValueError(
                "project_from= needs a Reference upstream: what it hands over is already "
                "optimized active orbitals, so it replaces the cheap pre-optimization "
                "rather than following it. Build this stage on the target Reference.")
        source_stage = (self.project_from if isinstance(self.project_from, Reference)
                        else self.project_from.reference_stage)
        if source_stage is upstream:
            raise ValueError(
                "project_from= names a stage on this same Reference, so there is no basis "
                "to project between; give the stage from the other basis' calculation")
        self.projection_source = source_stage.reference

        # The space is a statement about the orbitals being carried, so it comes from the
        # source and is resolved in the source's numbering.
        source_space = getattr(self.project_from, "active", None) \
            or getattr(self.project_from, "space", None)
        if source_space is not None:
            if requested:
                raise ValueError(
                    "the active space comes across with the projected orbitals ({}); "
                    "restating it here would resolve it against this reference's guess "
                    "orbitals instead, which is a different calculation. Drop active=/"
                    "character=, or project from the Reference and state it once."
                    .format(source_space.description or "CAS({}, {})".format(
                        source_space.n_elec, source_space.n_active)))
        else:
            source_space = _resolve_space(source_stage, active=active, character=character,
                                          n_active=n_active, n_active_elec=n_active_elec,
                                          threshold=threshold,
                                          what="CASSCF with project_from=Reference")
        self.source_space = source_space

        spaces = source_space.spaces
        plan = plan_columns(spaces.inactive, spaces.active, spaces.virtual,
                            upstream.reference.nspinor)
        self.space = projected_active_space(
            plan, upstream.reference, source_space.n_elec,
            description="{} (projected from {})".format(
                source_space.description or "explicit spinor indices",
                ", ".join(sorted(set(self.projection_source.data.basis_meta.values())))))
        self._plan = plan

    def _run_projection(self) -> None:
        """Do the projection and install its orbitals as this run's starting guess."""
        source = self.projection_source
        target = self.reference_stage.reference
        coeff = getattr(self.project_from, "coeff", None)
        if coeff is None:
            coeff = getattr(self.project_from, "orbitals", None)
        self.projection = _project_to_basis(source, target, coeff, space=self.source_space,
                                            report=self.report, **self.projection_options)
        plan = self.projection.plan
        if not (np.array_equal(plan.active, self._plan.active)
                and np.array_equal(plan.inactive, self._plan.inactive)):
            raise RuntimeError("the projection landed on a different orbital partition than "
                               "the one validated at construction; this is a bug")
        self._orbitals = self.projection.coeff

    # -- execution --------------------------------------------------------------------------

    def _execute(self) -> None:
        if self.project_from is not None:
            self._run_projection()
        if self.solver_kind == "ci":
            self._execute_ci()
        else:
            self._execute_dmrg()

    def _execute_ci(self) -> None:
        from .api import casscf as _api_casscf
        outcome = _api_casscf(self.reference_stage.reference, active=self.space,
                              n_states=(self.state_request if self.state_request is not None
                                        else self.n_states),
                              preserve_symmetry=self.preserve_symmetry,
                              weights=self.weights,
                              coeff=self._orbitals, checkpoint=self.checkpoint,
                              restart=self.restart,
                              checkpoint_options=self.checkpoint_options,
                              solver_options=self.solver_options, callback=self.callback,
                              report=self.report, **self.optimizer_options)
        self.outcome = outcome
        self.active = outcome.active
        self.solver = outcome.solver
        self.orbital = outcome.orbital
        self.ci = outcome.ci
        self.boundary = outcome.boundary
        self.boundary_initial = outcome.boundary_initial
        self.checkpoint_path = outcome.checkpoint_path
        self._energies = np.asarray(outcome.ci.total_energies, dtype=float)

    def _execute_dmrg(self) -> None:
        from ..dmrg import DMRGSolver
        from ..mcscf.casci import BOUNDARY_MARGIN
        from ..mcscf.events import optimize_orbitals_events
        from ..mcscf.orbopt import CASIntegrals, optimize_orbitals

        ref = self.reference_stage.reference
        h_ao = ref.h_one_electron()
        orbitals = self._orbitals if self._orbitals is not None else ref.spinors_in_ao()
        options = dict(self.solver_options)
        solver = DMRGSolver(self.space.n_elec, n_roots=self.n_states,
                            weights=self.weights, graph=self._resolve_graph(), **options)
        driver = optimize_orbitals_events if self._adaptive else optimize_orbitals
        if self.report:
            out.section(log, "CASSCF (DMRG solver)")
            self.space.report(log)
        result = driver(ref.factors, h_ao, orbitals, self.space.spaces, solver,
                        e_nuc=ref.data.e_nuc, callback=self.callback, report=self.report,
                        **self.optimizer_options)

        # The optimizer's last solve may sit at a rejected trial step; the states this stage
        # reports must belong to the returned orbitals. One warm solve pins them there and
        # carries the state-average boundary diagnostic the in-loop solves skip.
        ints = CASIntegrals.build(ref.factors, h_ao, result.coeff, self.space.spaces,
                                  e_nuc=ref.data.e_nuc)
        solver.boundary_check = BOUNDARY_MARGIN
        solver.solve(ints)

        self.active = self.space
        self.solver = solver
        self.orbital = result
        self.events = getattr(result, "events", [])
        self.boundary_gap_cm = solver.last.boundary_gap_cm
        self.graph = solver.graph
        self._energies = np.asarray(solver.last.energies, dtype=float) + ints.e_core
        if self.report:
            out.entries(log, [
                ("state energies [Eh]", ", ".join(out.E_FMT.format(e)
                                                  for e in self._energies)),
                ("largest bond dimension", solver.last.max_bond_dim),
                ("state-average boundary gap",
                 "complete" if self.boundary_gap_cm is None
                 else "{:.2f} cm^-1".format(self.boundary_gap_cm)),
            ])

    def _resolve_graph(self):
        request = self.graph_request
        if request is None or not isinstance(request, str):
            return request
        from ..dmrg import NetworkGraph, topology_from_mutual_information
        info = self.upstream.mutual_information
        if request == "mutual-information":
            return topology_from_mutual_information(info).graph
        order = self.upstream.dmrg_ordering()
        n = int(order.size)
        return NetworkGraph(n, [(i, i + 1) for i in range(n - 1)],
                            contents=[(int(m),) for m in order])

    # -- results ------------------------------------------------------------------------------

    @property
    def energy(self) -> float:
        """State-averaged total energy [Eh]."""
        self._check_ran()
        return float(self.orbital.energy)

    @property
    def energies(self) -> np.ndarray:
        """Total state energies [Eh], ascending — the spin-orbit spectrum."""
        self._check_ran()
        return self._energies

    @property
    def coeff(self) -> np.ndarray:
        """The converged spinor coefficients (AO basis)."""
        self._check_ran()
        return self.orbital.coeff

    @property
    def converged(self) -> bool:
        self._check_ran()
        return bool(self.orbital.converged)

    def _summary_entries(self):
        entries = [
            ("active space", "CAS({}, {})  {}".format(self.active.n_elec,
                                                      self.active.n_active,
                                                      self.active.description)),
            ("solver", self.solver_kind),
            ("E(CASSCF) [Eh]", out.E_FMT.format(self.energy)),
            ("converged", str(self.converged)),
            ("macro-iterations", str(self.orbital.n_iterations)),
            ("|grad|", "{:.2e}".format(self.orbital.grad_norm)),
            ("states", str(self._energies.size)),
        ]
        if self.solver_kind == "dmrg":
            entries.append(("largest bond dimension", str(self.solver.last.max_bond_dim)))
        if self.project_from is not None:
            entries.append(("projected active-space overlap",
                            "{:.6f}".format(self.projection.fidelity)))
        return entries


# --- 5. SC-NEVPT2 ---------------------------------------------------------------------------

class NEVPT2(_Stage):
    """SC-NEVPT2 dynamic correlation on a converged CASSCF — post-processing, per state.

    Wraps :func:`kuiva.pt.nevpt2.sc_nevpt2`; options (``frozen_core``, ``deleted_virtual``,
    ``shift``, ``imaginary_shift``, ``fock``, ``classes``, ...) are its keywords, validated
    eagerly. Needs a ``solver="ci"`` CASSCF: a tensor-network reference has no stored CI
    vectors, and the network-backed contraction it would need is not built.

    After :meth:`run`: :attr:`e2` [Eh, per state], :attr:`total_energies`,
    :attr:`class_energies`, :meth:`multiplets`, and the full
    :class:`~kuiva.pt.nevpt2.NEVPT2Result` as :attr:`result`.
    """

    _EXCLUDE = ("factors", "h_ao", "c_spinor", "spaces", "civecs", "n_elec", "energies",
                "weights", "e_nuc")

    def __init__(self, casscf: CASSCF, **options) -> None:
        super().__init__()
        from ..pt.nevpt2 import sc_nevpt2
        self.casscf = self._finished(casscf, CASSCF, "NEVPT2")
        if casscf.solver_kind != "ci":
            raise ValueError(
                "NEVPT2 needs the conventional-CI CASSCF (its per-state RDMs come from the "
                "stored CI vectors); a tensor-network reference would need a network-backed "
                "contraction provider, which is not built. Re-run the CASSCF with "
                "solver='ci' if the active space allows it")
        self.reference_stage = casscf.reference_stage
        _check_options(options, _allowed_options(sc_nevpt2, exclude=self._EXCLUDE),
                       "NEVPT2")
        if "classes" in options and options["classes"] is not None:
            from ..pt.classes import excitation_class
            for name in options["classes"]:
                excitation_class(name)                       # refuse an unknown name now
        self.options = dict(options)

    def _execute(self) -> None:
        from ..pt.nevpt2 import sc_nevpt2
        ref = self.reference_stage.reference
        oc = self.casscf.outcome
        self.result = sc_nevpt2(ref.factors, ref.h_one_electron(), oc.coeff,
                                oc.active.spaces, oc.ci.vectors, oc.active.n_elec,
                                energies=oc.ci.energies, e_nuc=ref.data.e_nuc,
                                **self.options)
        self.e2 = self.result.e2
        self.total_energies = self.result.total_energies
        self.class_energies = self.result.class_energies

    def multiplets(self, *args, **kwargs):
        """Degenerate-manifold view of the corrected spectrum (barycentres beside members)."""
        self._check_ran()
        return self.result.multiplets(*args, **kwargs)

    def _summary_entries(self):
        return [
            ("E2, state 0 [Eh]", out.E_FMT.format(float(self.e2[0]))),
            ("total, state 0 [Eh]", out.E_FMT.format(float(self.total_energies[0]))),
            ("complete (all eight classes)", str(self.result.complete)),
            ("frozen / deleted spinors", "{} / {}".format(self.result.n_frozen,
                                                          self.result.n_deleted)),
        ]


# --- 6. the two formatted products -----------------------------------------------------------

class PropertyDump(_Stage):
    """The property-matrix file: ``H`` and ``mu_x, mu_y, mu_z`` in the SOC eigenstate basis.

    ``source`` is a finished ``solver="ci"`` :class:`CASSCF` — or a finished :class:`NEVPT2`,
    in which case the corrected energies replace the diagonal **and the header records the
    hybrid protocol** (``H`` from perturbation theory, ``mu`` from the CASSCF states); that
    substitution is available only through this argument, never as a flag, so the file and
    its provenance cannot be separated.

    After :meth:`run`: :attr:`matrices` (compare only through its phase-invariant
    :meth:`~kuiva.props.dump.PropertyMatrices.analyse`) and :attr:`path`.
    """

    def __init__(self, source, path, *, title: str = "", include_l_s: bool = True,
                 comments: Sequence[str] = (), inactive_tol: Optional[float] = None,
                 report: bool = True) -> None:
        super().__init__()
        self._finished(source, (CASSCF, NEVPT2), "PropertyDump")
        self.source = source
        self.casscf = source if isinstance(source, CASSCF) else source.casscf
        if self.casscf.solver_kind != "ci":
            raise ValueError(
                "the property dump needs the transition densities of the conventional-CI "
                "CASSCF; the tensor-network route to properties is the pseudospin export "
                "(PseudospinExport)")
        data = self.casscf.reference_stage.reference.data
        if data.properties is None:
            raise ValueError("this reference carries no angular-momentum integrals; it was "
                             "not built by the front-end")
        self.path = Path(path)
        self.title, self.include_l_s = str(title), bool(include_l_s)
        self.comments, self.inactive_tol = tuple(comments), inactive_tol
        self.report = bool(report)

    def _execute(self) -> None:
        kwargs = {} if self.inactive_tol is None else {"inactive_tol": self.inactive_tol}
        matrices = _property_matrices(self.casscf.reference_stage.reference,
                                      self.casscf.outcome, comments=self.comments, **kwargs)
        if isinstance(self.source, NEVPT2):
            from ..pt.nevpt2 import corrected_property_matrices
            matrices = corrected_property_matrices(matrices, self.source.result)
        if self.report:
            out.section(log, "Property matrices")
            matrices.report(log)
        self.matrices = matrices
        matrices.write(self.path, title=self.title, include_l_s=self.include_l_s)

    def _summary_entries(self):
        return [
            ("file", str(self.path)),
            ("states", str(self.matrices.n_states)),
            ("energies", "NEVPT2-corrected (hybrid protocol, recorded)"
             if isinstance(self.source, NEVPT2) else "CASSCF"),
        ]


class PseudospinExport(_Stage):
    """The pseudospin export: local multiplets, ``H_eff`` and moments, the OuluSpin file.

    The tensor-network property route (a sibling of :class:`PropertyDump`, which needs the
    conventional CI): at the converged CASSCF orbitals the active-space Hamiltonian and the
    three magnetic-moment components are contracted onto a local-multiplet model space by
    the ensemble loop of :func:`kuiva.dmrg.manifold.solve_manifold`, the model is assigned
    pseudospin labels, and the formatted file is written. Works from either CASSCF solver —
    the manifold loop re-solves the network from the integrals either way (warm topology
    from a DMRG run when available).

    ``sites`` partitions the **active spinors** (by position in the active list) into the
    local multiplet sites; ``None`` uses the structure discovered from the converged state's
    entanglement. ``rule``/``dims`` choose each site's multiplet space (``rule="dimension"``
    with ``dims=2`` is "the ground Kramers doublet per site"). Further loop knobs
    (``n_roots``, ``max_roots``, ``max_outer``, ...) pass through ``manifold_options``.

    After :meth:`run`: :attr:`model` (:class:`~kuiva.props.pseudospin.PseudospinModel`),
    :attr:`g_values` (per site), :attr:`path`. Validation of the file goes through
    phase-invariant reductions only.
    """

    def __init__(self, casscf: CASSCF, path, *, sites=None, rule: str = "gap", dims=None,
                 max_bond: Optional[int] = None, axes=None, common_axis=None,
                 rotate_frame: bool = False, g_electron: Optional[float] = None,
                 title: str = "", comments: Sequence[str] = (), seed: int = 0,
                 manifold_options: Optional[Dict[str, Any]] = None,
                 report: bool = True) -> None:
        super().__init__()
        from ..dmrg.manifold import solve_manifold
        self.casscf = self._finished(casscf, CASSCF, "PseudospinExport")
        data = casscf.reference_stage.reference.data
        if data.properties is None:
            raise ValueError("this reference carries no angular-momentum integrals; it was "
                             "not built by the front-end")
        n_active = casscf.active.n_active
        self.sites = None
        if sites is not None:
            groups = [tuple(int(m) for m in group) for group in sites]
            flat = [m for group in groups for m in group]
            if sorted(flat) != list(range(n_active)):
                raise ValueError(
                    "sites must partition the {} active spinor positions 0..{} exactly; "
                    "got {}".format(n_active, n_active - 1, groups))
            self.sites = groups
        self.manifold_options = dict(manifold_options or {})
        _check_options(self.manifold_options,
                       _allowed_options(solve_manifold,
                                        exclude=("terms", "graph", "n_elec", "bases",
                                                 "sites", "rule", "dims", "operators",
                                                 "max_bond", "rng")),
                       "PseudospinExport manifold_options")
        self.rule, self.dims, self.max_bond = rule, dims, max_bond
        self.axes, self.common_axis, self.rotate_frame = axes, common_axis, rotate_frame
        self.g_electron = g_electron
        self.path = Path(path)
        self.title, self.comments = str(title), tuple(comments)
        self.seed, self.report = int(seed), bool(report)

    def _execute(self) -> None:
        from ..dmrg import NetworkGraph, hamiltonian_product_terms
        from ..dmrg.manifold import solve_manifold
        from ..dmrg.ttno import one_electron_product_terms
        from ..mcscf.orbopt import CASIntegrals
        from ..props.dump import inactive_moment, spinor_operator, spinor_operators
        from ..props.multiplet import G_ELECTRON
        from ..props.pseudospin import pseudospin_from_model, write_pseudospin
        from ..spinor.expand import spin_operator

        cas = self.casscf
        ref = cas.reference_stage.reference
        spaces = cas.active.spaces
        g_e = G_ELECTRON if self.g_electron is None else float(self.g_electron)

        ints = CASIntegrals.build(ref.factors, ref.h_one_electron(), cas.coeff, spaces,
                                  e_nuc=ref.data.e_nuc)
        terms = hamiltonian_product_terms(ints.h_active_effective(), ints.active_eri())

        # mu = -(L + g_e S) over the active spinors; the inactive trace — exactly zero for a
        # Kramers-paired inactive set, warned about otherwise — is added to the total
        # operators after the contraction (a scalar cannot be attributed to one site).
        l_mo, s_mo = spinor_operators(cas.coeff, ref.data.properties.two_component(),
                                      spin_operator(ref.data.s_ao))
        ix = np.ix_(spaces.active, spaces.active)
        names = ("mu_x", "mu_y", "mu_z")
        moment_ao = ref.data.properties.moment_operator()
        if moment_ao is None:
            mu_act = -(np.stack([lk[ix] for lk in l_mo])
                       + g_e * np.stack([sk[ix] for sk in s_mo]))
            mu_inactive = -(inactive_moment(l_mo, spaces.inactive, name="L")
                            + g_e * inactive_moment(s_mo, spaces.inactive, name="S"))
        else:
            # ⚠ The picture-changed moment does not separate into an L part and an S part, so
            # mu is built from the transformed (L + 2S) plus the (g_e - 2) anomaly. This branch
            # must stay in step with the one in kuiva.props.dump.property_matrices — the two
            # are held together by a test that runs both routes on one reference, not by
            # structure, because the branch above is kept byte-identical on purpose.
            m_mo = spinor_operator(cas.coeff, moment_ao)
            anomaly_ao = ref.data.properties.anomaly_spin()
            a_mo = s_mo if anomaly_ao is None else spinor_operator(cas.coeff, anomaly_ao)
            mu_act = -(np.stack([mk[ix] for mk in m_mo])
                       + (g_e - 2.0) * np.stack([ak[ix] for ak in a_mo]))
            mu_inactive = -(inactive_moment(m_mo, spaces.inactive, name="L+2S")
                            + (g_e - 2.0) * inactive_moment(a_mo, spaces.inactive, name="S"))
        operators = {name: one_electron_product_terms(np.ascontiguousarray(mu_act[k]))
                     for k, name in enumerate(names)}

        graph = getattr(cas, "graph", None)
        if graph is None:
            graph = NetworkGraph.path(cas.active.n_active)
        node_sites = (None if self.sites is None
                      else [_nodes_of_modes(graph, group) for group in self.sites])
        if cas.solver_kind == "dmrg":
            self.manifold_options.setdefault("max_sweeps", cas.solver.max_sweeps)
        max_bond = self.max_bond
        if max_bond is None and cas.solver_kind == "dmrg":
            max_bond = cas.solver.max_bond
        manifold = solve_manifold(terms, graph, cas.active.n_elec, sites=node_sites,
                                  rule=self.rule, dims=self.dims, operators=operators,
                                  max_bond=max_bond, rng=np.random.default_rng(self.seed),
                                  **self.manifold_options)
        model = manifold.model
        eye = np.eye(model.model_dim)
        for k, name in enumerate(names):
            model.operators[name] = model.operators[name] + mu_inactive[k] * eye

        provenance: Dict[str, object] = {
            "active_space": cas.active.description or "explicit spinor indices",
            "n_active_spinors": int(cas.active.n_active),
            "n_active_electrons": int(cas.active.n_elec),
            "casscf_solver": cas.solver_kind,
            "properties": ref.data.properties.provenance(),
            "basis": dict(ref.data.basis_meta),
        }
        if ref.data.soc is not None:
            provenance["hamiltonian"] = ref.data.soc.provenance()

        self.model = pseudospin_from_model(model, axes=self.axes,
                                           common_axis=self.common_axis,
                                           rotate_frame=self.rotate_frame,
                                           energy_shift=float(ints.e_core),
                                           provenance=provenance, comments=self.comments)
        if self.report:
            self.model.report(log)
        self.g_values = tuple(site.g_values for site in self.model.sites)
        write_pseudospin(self.path, self.model, title=self.title)

    def _summary_entries(self):
        return [
            ("file", str(self.path)),
            ("sites", " x ".join(str(d) for d in self.model.dims)),
            ("g values", "; ".join(
                " ".join("{:.4f}".format(g) for g in site) for site in self.g_values)),
        ]


def _nodes_of_modes(graph, modes: Sequence[int]) -> Tuple[int, ...]:
    """The network nodes carrying exactly ``modes``; a straddling node is refused."""
    wanted = set(int(m) for m in modes)
    nodes = [u for u, held in enumerate(graph.contents) if wanted & set(held)]
    covered = set()
    for u in nodes:
        held = set(graph.contents[u])
        if not held <= wanted:
            raise ValueError(
                "network node {} carries modes {} and the requested site only {}: a site "
                "boundary cannot cut through a node. Regroup the site, or give a graph "
                "whose nodes respect it".format(u, sorted(held), sorted(wanted)))
        covered |= held
    if covered != wanted:
        raise ValueError("modes {} are on no network node".format(sorted(wanted - covered)))
    return tuple(nodes)
