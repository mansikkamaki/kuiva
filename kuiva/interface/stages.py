"""The high-level class API: a production calculation as a short script of stage objects.

::

    ScalarSCF -> Reference -> (CheapCI) -> CASSCF -> (NEVPT2) -> PropertyDump
                                                             \\-> PseudospinExport

:class:`CASCI` is the fixed-orbital sibling of :class:`CASSCF`: it takes any stage that
carries orbitals — a :class:`Reference`, a :class:`CheapCI` or a finished :class:`CASSCF` —
and feeds :class:`NEVPT2` and :class:`PropertyDump` exactly as a :class:`CASSCF` does. It is
where a scan at fixed orbitals is written: a state count, a symmetry mode or an active space
varied without paying for a second orbital optimization.

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
from .pyscf_bridge import ScalarX2CData, validate_scf_controls
from .api import (Molecule, SpinorReference as _SpinorData, active_space_for,
                  project_to_basis as _project_to_basis, projected_active_space,
                  property_matrices as _property_matrices, scalar_x2c_reference,
                  spinor_reference)

log = get_logger(__name__)

__all__ = ["ScalarSCF", "Reference", "CheapCI", "CASSCF", "CASCI", "NEVPT2",
           "PropertyDump", "PseudospinExport"]


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

    **When the SCF will not converge**, which on a real open-shell metal complex is where a
    calculation first stops: ``level_shift=``, ``damp=``, ``diis="adiis"`` and
    ``second_order=True`` are the levers, ``stability="check"`` says whether the solution that
    came back is a minimum at all, and ``guess_from=`` starts from another finished
    ``ScalarSCF`` (projecting its orbitals if the basis differs). ⚠ An SCF that does not
    converge **refuses**; ``allow_unconverged_scf=True`` continues on it deliberately.

    ⚠ **For antiferromagnetically coupled centres there is a fourth lever, and it is not a
    convergence one**: an unrestricted SCF started the ordinary way *converges* perfectly well
    — to the symmetric solution, because the closed-shell density is a stationary point.
    ``broken_symmetry={"Fe1": +5, "Fe2": -5}`` (with ``reference="uhf"``) builds the polarized
    density instead, from the high-spin solution's localized magnetic orbitals, and reports the
    two things that say whether it held: ``<S^2>`` between the low-spin and high-spin values,
    and spin populations carrying the signs that were asked for.

    After :meth:`run`: :attr:`data`, :attr:`energy` [Eh], :attr:`converged`.
    """

    _EXCLUDE = ("molecule",)

    #: Options this stage validates itself, beyond the by-name check: the SCF convergence
    #: controls, checked through the same implementation the driver applies them with, so a
    #: misspelled DIIS variant or an out-of-range damping fails at construction rather than
    #: after the memory pre-flight and a four-component atomic solve.
    _CONTROL_OPTIONS = ("level_shift", "damp", "init_guess", "diis", "diis_space",
                        "diis_start_cycle", "second_order", "stability")

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
        validate_scf_controls(**{k: options[k] for k in self._CONTROL_OPTIONS
                                 if k in options})
        options = dict(options)
        guess = options.get("guess_from")
        if guess is not None:
            # A stage is unwrapped here and nowhere else: api/ and the bridge take data, not
            # stages, and a stage that has not run has no orbitals to give.
            if isinstance(guess, ScalarSCF):
                options["guess_from"] = self._finished(guess, ScalarSCF, "ScalarSCF").data
            elif not isinstance(guess, ScalarX2CData):
                raise TypeError(
                    "guess_from takes a finished ScalarSCF stage or the ScalarX2CData it "
                    "produced; got {!r}".format(type(guess).__name__))
        self.molecule = molecule
        self.options = options

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

    @property
    def stable(self) -> Optional[bool]:
        """Internal stability of the converged solution, or ``None`` if it was not measured.

        ⚠ ``None`` is not ``True``: it means ``stability=`` was not asked for, and a check
        written as ``if not scf.stable`` reads a run that never measured as unstable while
        ``if scf.stable`` reads it as stable. Compare against ``True``/``False`` explicitly.
        """
        self._check_ran()
        return self.data.scf_stable

    def _summary_entries(self):
        soc = self.data.soc
        entries = [
            ("E(SCF) [Eh]", out.E_FMT.format(self.energy)),
            ("converged", str(self.converged)),
            ("reference", self.data.reference),
            ("hamiltonian", "spin-free" if soc is None else soc.provenance()["method"]),
        ]
        if self.stable is not None:
            entries.append(("internal stability",
                            "stable" if self.stable else "UNSTABLE (saddle point)"))
        return entries


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

    ⚠ **On the stored route this stage releases the SCF's two-electron integral array**, the
    moment the factors that replace it exist: nothing downstream reads it again and it is the
    largest thing the container holds (``O(nao^4/8)``). A script that wants the array
    afterwards — an exactness check, a second factorization at another threshold — takes its
    own reference to ``scf.data.eri`` first, or factorizes through
    :meth:`kuiva.integrals.transform.ThreeIndexAO.from_scalar_data` with ``release_eri=False``.
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
            "{} needs an active space: give active=[spinor indices], character=(atom, l) "
            "with n_active= (or a list of (atom, l, n_spinors) fragments), or "
            "avas=dict(atom=..., l=...) where the target orbitals are covalent mixtures no "
            "single orbital carries; an active space is a physical statement, so there is "
            "no default".format(what))
    return active_space_for(reference.reference, active=active, character=character,
                            n_active=n_active, n_active_elec=n_active_elec,
                            threshold=threshold)


def _resolve_avas(reference: Reference, upstream, avas, *, active, character,
                  n_active, threshold, what: str):
    """Run an ``avas=`` request eagerly, returning ``(AVASResult, rotated orbitals)``.

    ⚠ **AVAS runs against the reference's own SCF orbitals and their integer occupations**,
    which is why it refuses a :class:`CheapCI` upstream. The rotation is only density-
    preserving inside groups of *equal* occupation; the cheap CI's natural occupations are
    all distinct, so every group would hold one pair and the "rotation" would be the
    identity — an AVAS that silently did nothing. Put ``avas=`` on the :class:`CheapCI`
    instead and let the CASSCF inherit the space.
    """
    from .api import avas_active_space

    if active is not None or character is not None:
        raise ValueError("give exactly one of active=, character= and avas=: they are three "
                         "ways of answering the same question, and a run that silently "
                         "preferred one would not be reproducible")
    if not isinstance(upstream, Reference):
        raise ValueError(
            "{} cannot run AVAS on a {} upstream: AVAS rotates within groups of equal "
            "occupation and the cheap CI's natural occupations are all distinct, so the "
            "rotation would be the identity. Put avas= on the CheapCI stage instead"
            .format(what, type(upstream).__name__))
    if not isinstance(avas, dict):
        raise ValueError("avas= takes a dict of options for "
                         "api.avas_active_space, e.g. avas=dict(atom='Ti', l='d'); got {!r}"
                         .format(avas))
    options = dict(avas)
    if threshold is not None and "threshold" not in options:
        raise ValueError("threshold= is the character-selection Loewdin cut and does not "
                         "apply to AVAS; put AVAS's projection cut inside avas= as "
                         "avas=dict(..., threshold=...)")
    if n_active is not None:
        raise ValueError("AVAS chooses the number of orbitals from the projection spectrum, "
                         "so n_active= does not apply; bound it with "
                         "avas=dict(..., max_pairs=...) if a size limit is wanted")
    from ..mcscf.avas import avas as _avas_fn
    _check_options(options,
                   _allowed_options(avas_active_space, _avas_fn,
                                    exclude=("reference", "coeff", "occupation", "report",
                                             "coeff_ao", "s_ao", "layout", "n_elec_total")),
                   "{} avas".format(what))
    result = avas_active_space(reference.reference, report=False, **options)
    return result, result.coeff


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

    The active space is stated as ``active=``, ``character=`` or ``avas=`` — exactly one, as
    on :class:`CASSCF`, and ``avas=`` additionally rotates the orbitals this stage starts
    from onto the atomic valence set (:func:`kuiva.interface.api.avas_active_space`). ⚠ This
    is the stage AVAS belongs on when a pre-optimization is wanted: it works from the
    reference's integer occupations, which the cheap CI's natural occupations are not.

    After :meth:`run`: :attr:`orbitals`, :attr:`occupations`, :attr:`natural_occupation`,
    :attr:`entropy`, :attr:`mutual_information`, :meth:`suggested_active`,
    :meth:`dmrg_ordering`, the full :class:`~kuiva.mcscf.preopt.PreoptResult` as
    :attr:`result`, and (with ``avas=``) the :class:`~kuiva.mcscf.avas.AVASResult` as
    :attr:`avas`.
    """

    _EXCLUDE = ("factors", "h_ao", "c_spinor", "spaces", "n_active_elec", "e_nuc",
                "h_eff", "eri", "n_elec")

    def __init__(self, reference: Reference, *, active=None, character=None,
                 n_active: Optional[int] = None, n_active_elec: Optional[int] = None,
                 threshold: Optional[float] = None, avas=None, **options) -> None:
        super().__init__()
        from ..mcscf.preopt import cheap_ci, preoptimize
        self.reference_stage = self._finished(reference, Reference, "CheapCI")
        _check_options(options, _allowed_options(preoptimize, cheap_ci,
                                                 exclude=self._EXCLUDE), "CheapCI")
        self.avas, self._orbitals = None, None
        if avas is not None:
            if n_active_elec is not None:
                avas = dict(avas, n_active_elec=n_active_elec)
            self.avas, self._orbitals = _resolve_avas(
                reference, reference, avas, active=active, character=character,
                n_active=n_active, threshold=threshold, what="CheapCI")
            self.space = self.avas.space
        else:
            self.space = _resolve_space(reference, active=active, character=character,
                                        n_active=n_active, n_active_elec=n_active_elec,
                                        threshold=threshold, what="CheapCI")
        self.options = dict(options)

    def _execute(self) -> None:
        from ..mcscf.preopt import preoptimize
        from ..spinor.expand import nearest_kramers_paired, spin_block_diagonal, time_reverse
        ref = self.reference_stage.reference
        start = self._orbitals if self._orbitals is not None else ref.spinors_in_ao()
        if self.avas is not None:
            self.avas.report(log)
        self.result = preoptimize(ref.factors, ref.h_one_electron(), start,
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
        ``checkpoint=``/``restart=`` work on this route too and write **two** files: the
        ordinary CASSCF checkpoint carries the trajectory — orbitals, RDMs,
        orbital-optimizer state — and a sibling ``*.network.h5`` file carries the network
        state itself, rolling, at the end of each completed sweep
        (:mod:`kuiva.dmrg.checkpoint`). A restart resumes the trajectory and warm-starts
        the network from that sibling; ⚠ an absent or unfitting network file warns and
        starts the network cold, which costs time and not correctness. ``restart=`` needs
        the frozen-chart driver — the event-gated one (``adaptive=True``, or a
        ``bond_steps=`` ladder) does not resume an optimizer state.

        Production controls, all through ``solver_options``: ``bond_schedule=`` ramps
        the cap per sweep inside the first solve and ``expansion=`` perturbs its
        truncations (the deterministic subspace expansion — see
        :func:`kuiva.dmrg.sweep.solve_ttn`); ``bond_steps=[64, 128, 256]`` is the
        per-macro-iteration cap ladder, a sequence of chart changes offered through the
        propose/adopt seam — giving it selects the event-gated driver automatically, and
        each rung is adopted only when it lowers the energy at fixed integrals. The
        ``E(w_disc -> 0)`` extrapolation is a separate driver over a converged problem:
        :func:`kuiva.dmrg.bond_series`.

    Remaining keyword options go to the orbital optimizer (``mode``, ``max_iter``,
    ``conv_grad``, ``conv_energy``, ``max_step``, and for the event-gated driver ``tau``,
    ``event_interval``, ...). ``mode="second-order"`` is the right explicit choice for a
    heavy element or a large state average.

    Stopping before the allocation does
    -----------------------------------
    ``deadline=`` makes the run stop *itself*, in time to write a checkpoint, instead of
    being killed at a queue's wall limit with only the last checkpoint to show for it:

    ``None`` (the default)
        No deadline. Nothing is read, nothing is printed, nothing ever stops the run early
        — which is what a cluster with no time limit needs, and why it is the default.
    ``"6h"``, ``"90m"``, ``"24:00:00"``, ``21600``
        A budget of your own. ⚠ It starts when the **stage is constructed**, not when it is
        run — build the stage next to its ``run()``, or use the queue's limit, which is an
        absolute instant and cannot drift this way. ⚠ A bare numeric *string* is refused,
        because ``"60"`` is sixty minutes to Slurm and sixty seconds here.
    ``"slurm"`` / ``"queue"``
        This batch allocation's own time limit, read once from ``$SLURM_JOB_END_TIME`` and
        then from ``scontrol``. ⚠ **Refuses** if it cannot be read: an explicit request
        that silently produced no deadline is the one outcome worse than either.
    ``"auto"``
        The queue's limit where there is one, and no deadline where there is not, stated
        either way. The portable spelling for a script that runs on a laptop, on an
        unlimited cluster and inside a queue without being edited.

    ⚠ **The granularity is one macro-iteration.** The run stops between them and nowhere
    else, so the decision is predictive: it stops when the time left is less than the
    longest recent iteration plus the estimated checkpoint write plus a stated margin. One
    CI solve, and on the DMRG route one whole network solve, cannot be interrupted.
    ⚠ With ``checkpoint=``, the final write is **forced** past the cadence rules and happens
    before the stop; without one, the run still exits cleanly but keeps nothing, and says so.

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

    Choosing the active space
    -------------------------
    ``active=`` (spinor indices), ``character=`` (the lowest pairs of an ``(atom, l)``
    character — the form a reference calculation must use) and ``avas=`` are three ways of
    answering one question, and exactly one may be given.

    ``avas=dict(atom="Ti", l="d")`` runs an AVAS projection
    (:func:`kuiva.interface.api.avas_active_space`): it rotates the reference's orbitals onto
    the free-atom valence orbitals and takes the combinations that carry the character. Use
    it where the target orbitals are **covalent mixtures** that no single canonical orbital
    carries, which is where a character threshold fails. ``avas=dict(..., n_shells=2)`` is
    the double shell. ⚠ It needs ``atomic_reference=True`` on the front end, needs a
    :class:`Reference` upstream (not a :class:`CheapCI` — put ``avas=`` on that stage
    instead), and its space carries no symmetry labels.

    After a run, :meth:`spin_analysis` gives ``<S^2>`` per degenerate block and
    :meth:`assign` offers a term label per block with the evidence behind it. ⚠ The
    assignment is an inference and prints as its own report, never as a column of the state
    table.

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
    ``"dmrg"``: :attr:`orbital`, :attr:`events`, :attr:`boundary_gap_cm`, :attr:`graph`,
    :attr:`max_discarded` -- the largest ensemble truncation weight of the final sweep, the
    network's primary quality number and the one a truncated result must be quoted with) and,
    with ``project_from=``, :attr:`projection` — the
    :class:`~kuiva.orth.project.BasisProjection` carrying the orbitals it started from and the
    invariants that say whether the projection was worth using.
    """

    _GRAPH_CHOICES = ("mutual-information", "fiedler")

    def __init__(self, upstream, *, active=None, character=None,
                 n_active: Optional[int] = None, n_active_elec: Optional[int] = None,
                 threshold: Optional[float] = None, avas=None, n_states=1, weights=None,
                 solver: str = "ci", solver_options: Optional[Dict[str, Any]] = None,
                 graph=None, checkpoint=None, restart=None,
                 checkpoint_options: Optional[Dict[str, Any]] = None,
                 callback: Optional[Callable[[dict], Optional[bool]]] = None,
                 deadline=None,
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
        # ⚠ Resolved at construction, like every other option here: deadline="slurm" outside
        # a Slurm job is a mistake that must surface now, not after the first hour.
        from ..util.deadline import Deadline
        self.deadline = Deadline.resolve(deadline)
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
        self.avas = None
        if avas is not None:
            if project_from is not None or restart is not None:
                raise ValueError("avas= chooses an active space and rotates the orbitals it "
                                 "is chosen from; project_from= and restart= each bring "
                                 "their own orbitals and their own space")
            if n_active_elec is not None:
                avas = dict(avas, n_active_elec=n_active_elec)
            self.avas, self._orbitals = _resolve_avas(
                self.reference_stage, upstream, avas, active=active, character=character,
                n_active=n_active, threshold=threshold, what="CASSCF")
            self.space = self.avas.space
        requested = active is not None or character is not None
        if avas is None and project_from is not None:
            self._setup_projection(upstream, requested, active=active, character=character,
                                   n_active=n_active, n_active_elec=n_active_elec,
                                   threshold=threshold, restart=restart)
        elif avas is None and restart is not None:
            if not Path(restart).exists():
                raise ValueError("restart checkpoint {!r} does not exist".format(restart))
        if avas is not None:
            pass                                     # space and orbitals both set above
        elif project_from is None:
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
                         "start_iteration", "space_key", "history", "extra_columns"))
            # api.casscf's own, and named rather than swept in with the rest of that
            # function's keywords: those are this stage's explicit arguments and letting
            # them through here would let one be given twice.
            allowed.add("classify")
            _check_options(self.optimizer_options, allowed, "CASSCF (orbital optimizer)")
        else:
            from ..dmrg import DMRGSolver
            from ..mcscf.events import optimize_orbitals_events
            if "max_bond" not in self.solver_options:
                raise ValueError(
                    "solver='dmrg' needs solver_options=dict(max_bond=...): an uncapped "
                    "tree state allocates charge-sector-maximal bond dimensions")
            _check_options(self.solver_options,
                           _allowed_options(DMRGSolver.__init__,
                                            exclude=("self", "n_elec", "n_roots", "weights",
                                                     "graph", "initial_state",
                                                     "checkpoint", "restart")),
                           "CASSCF solver_options (DMRGSolver; the topology is the "
                           "stage-level graph= option, and checkpoint=/restart= are the "
                           "stage-level arguments — the network-state file is derived "
                           "from them)")
            # bond_steps is a ladder of chart changes, and chart changes are events: it
            # routes the optimization through the event-gated driver exactly as
            # adaptive=True does, so the two share every consequence below.
            self._adaptive = bool(self.solver_options.get("adaptive", False)) \
                or self.solver_options.get("bond_steps") is not None
            if restart is not None and self._adaptive:
                raise ValueError(
                    "restart= on the DMRG route needs the frozen-chart driver: the "
                    "event-gated one (adaptive=True, or a bond_steps= ladder) re-derives "
                    "its space by proposals and does not resume an optimizer state. "
                    "Restart without them, or drive kuiva.mcscf directly")
            driver = optimize_orbitals_events if self._adaptive else optimize_orbitals
            allowed = _allowed_options(
                driver, exclude=("factors", "h_ao", "c_spinor", "spaces", "ci_solver",
                                 "e_nuc", "callback", "report", "optimizer_state",
                                 "start_iteration", "space_key", "history",
                                 "extra_columns"))
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
        if self.avas is not None and self.report:
            self.avas.report(log)
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
                              deadline=self.deadline,
                              report=self.report, **self.optimizer_options)
        self.outcome = outcome
        #: What the property stages consume, under the name :class:`CASCI` uses for it too,
        #: so a consumer of either stage asks one question. The network route has no such
        #: object -- its states live in the solver -- and deliberately does not set it.
        self.states = outcome
        self.active = outcome.active
        self.solver = outcome.solver
        self.orbital = outcome.orbital
        self.ci = outcome.ci
        self.boundary = outcome.boundary
        self.boundary_initial = outcome.boundary_initial
        self.checkpoint_path = outcome.checkpoint_path
        self._energies = np.asarray(outcome.ci.total_energies, dtype=float)

    def _execute_dmrg(self) -> None:
        import json

        from ..dmrg import DMRGSolver
        from ..mcscf.casci import BOUNDARY_MARGIN, ActiveSpace
        from ..mcscf.events import optimize_orbitals_events
        from ..mcscf.orbopt import CASIntegrals, optimize_orbitals

        ref = self.reference_stage.reference
        h_ao = ref.h_one_electron()
        optimizer_options = dict(self.optimizer_options)
        resumed = None
        if self.restart is not None:
            # ⚠ Read failure on an explicit restart is an ERROR that propagates, exactly as
            # on the conventional-CI route: the user asked to resume, and silently starting
            # over wastes what the file protects.
            from ..io.checkpoint import read_checkpoint
            resumed = read_checkpoint(self.restart)
            restored = ActiveSpace(spaces=resumed.spaces, n_elec=resumed.n_active_elec,
                                   description="restored from {}".format(self.restart))
            if self.space is not None and (
                    not np.array_equal(self.space.spaces.active, restored.spaces.active)
                    or self.space.n_elec != restored.n_elec):
                raise ValueError(
                    "the checkpoint at {} holds CAS({}, {}) and the arguments ask for "
                    "CAS({}, {}); a restart continues the calculation that was "
                    "interrupted, so leave the active space out or make it match"
                    .format(self.restart, restored.n_elec, restored.n_active,
                            self.space.n_elec, self.space.n_active))
            self.space = restored
            orbitals = np.ascontiguousarray(resumed.coeff)
        else:
            orbitals = self._orbitals if self._orbitals is not None else ref.spinors_in_ao()
        options = dict(self.solver_options)
        from ..dmrg.checkpoint import NetworkCheckpointPolicy, network_state_path
        if self.checkpoint is not None:
            # The cadence knobs of checkpoint_options= govern both files: they are
            # statements about the machine (budget, disk, interval), not about which
            # object is being protected.
            shared = {k: v for k, v in (self.checkpoint_options or {}).items()
                      if k in ("budget_gb", "min_interval", "cost_fraction")}
            options["checkpoint"] = NetworkCheckpointPolicy(
                network_state_path(self.checkpoint), **shared)
        if resumed is not None:
            network_file = network_state_path(self.restart)
            if network_file.is_file():
                options["restart"] = network_file
            else:
                log.warning("no network-state file at %s beside the checkpoint; the "
                            "restart resumes the orbital trajectory and the first solve "
                            "rebuilds the network from scratch (time, not correctness)",
                            network_file)
        solver = DMRGSolver(self.space.n_elec, n_roots=self.n_states,
                            weights=self.weights, graph=self._resolve_graph(), **options)
        # Resolve the default topology now, so the solver's space_key names a real chart:
        # a restart compared against "dmrg:unset" would clear curvature that belongs to
        # exactly this surface.
        solver._ensure_chart(self.space.n_active)
        if resumed is not None:
            from .api import _check_restart_state_average
            _check_restart_state_average(resumed, solver, self.restart)
            # ⚠ The LIVE solver's key, never the file's own (see api.casscf): handing the
            # checkpoint its own key back would compare the file with itself and restore
            # curvature across any chart change without a word.
            optimizer_options.update(resumed.optimizer_kwargs(
                space_key=solver.space_key()))
        hook = self.callback
        policy = None
        if self.checkpoint is not None:
            from ..io.checkpoint import CheckpointPolicy
            metadata = {"active_space": self.space.description}
            if ref.data.soc is not None:
                metadata["hamiltonian"] = json.dumps(ref.data.soc.provenance(),
                                                     sort_keys=True)
            policy = CheckpointPolicy(self.checkpoint, solver=solver, metadata=metadata,
                                      n_active_elec=self.space.n_elec, chain=self.callback,
                                      deadline=self.deadline,
                                      **(self.checkpoint_options or {}))
            hook = policy.callback
        elif self.deadline is not None:
            hook = self.deadline.as_callback(chain=self.callback)
        driver = optimize_orbitals_events if self._adaptive else optimize_orbitals
        if self.report:
            out.section(log, "CASSCF (DMRG solver)")
            self.space.report(log)
            if resumed is not None:
                resumed.report(log)
            if self.deadline is not None:
                self.deadline.report(log)
        if self.deadline is not None:
            # ⚠ The granularity is a macro-iteration, and on this route one of those holds a
            # whole DMRG solve: the deadline can refuse to start another, never interrupt one.
            self.deadline.assert_room("this DMRG-CASSCF")
        # ⚠ The truncation weight is the tensor network's primary quality number, and without
        # this it appeared nowhere at INFO: the sweep table is at DEBUG (one table per sweep
        # times many macro-iterations is noise in a file that IS the output), so a production
        # DMRG output never said how much of the state was thrown away. It rides the
        # optimizer's additive extra_columns keyword, so the shared driver stays ignorant of
        # what a bond dimension is. The trend matters as much as the final value -- truncation
        # growing as the orbitals move is the signal that max_bond is too small.
        w_disc = ((out.col_sci("w_disc"),
                   lambda: (float("nan") if solver.last is None
                            else float(solver.last.max_discarded))),)
        result = driver(ref.factors, h_ao, orbitals, self.space.spaces, solver,
                        e_nuc=ref.data.e_nuc, callback=hook, report=self.report,
                        extra_columns=w_disc, **optimizer_options)

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
        self.max_discarded = float(solver.last.max_discarded)
        self.checkpoint_path = str(policy.path) if policy is not None else None
        self.network_checkpoint_path = (str(solver.checkpoint.path)
                                        if solver.checkpoint is not None else None)
        if policy is not None and self.report:
            policy.report(log)
            if solver.checkpoint is not None:
                solver.checkpoint.report(log)
        if policy is not None and solver.checkpoint is not None \
                and solver.checkpoint.n_written == 0:
            # ⚠ Stated plainly, not buried: the trajectory file protects the calculation,
            # and losing the network state costs a restart's first solve its warm start —
            # time, not correctness.
            out.note(log, "no network state was written (see the policy lines above): a "
                          "restart resumes the orbital trajectory and rebuilds the "
                          "network from scratch on its first solve")
        if self.report:
            out.entries(log, [
                ("state energies [Eh]", ", ".join(out.E_FMT.format(e)
                                                  for e in self._energies)),
                ("largest bond dimension", solver.last.max_bond_dim),
                # ⚠ The number that says whether any of the above is converged with respect
                # to the network, and the one a truncated result has to be quoted with.
                ("largest discarded weight", self.max_discarded, "",
                 "ensemble truncation, final sweep", "{:.3e}"),
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

    def spin_analysis(self, *, tol_cm: float = 1.0, report: bool = False):
        """``<S^2>`` per degenerate block — the multiplicity, or the spin-purity diagnostic.

        Spin-orbit coupling off: ``2S+1`` is the term multiplicity, read straight off. On:
        ``S`` is not conserved and the number measures how pure the spin still is. ⚠ Per
        block and never per state (:mod:`kuiva.props.spin`). Available on **both** solver
        routes through one implementation: the CI solver applies ``S`` to the CI vectors
        through the excitation map, the network solver contracts the same quantities
        through each root's own densities, and ``props`` duck-types the difference away.
        The out-of-active-space correction is a property of the *orbitals* and is the
        same code on both routes.
        """
        self._check_ran()
        if self.solver_kind != "ci":
            from ..props.spin import spin_analysis as _spin_states
            ref = self.reference_stage.reference
            result = _spin_states(self.solver, self.coeff, self.active.spaces,
                                  ref.data.s_ao, self._energies, tol_cm=tol_cm,
                                  has_soc=ref.data.has_soc)
            if report:
                result.report(log)
            return result
        from .api import spin_analysis as _spin
        return _spin(self.reference_stage.reference, self.outcome, tol_cm=tol_cm,
                     report=report)

    def assign(self, *, matrices=None, tol_cm: float = 1.0, report: bool = True):
        """Offer a term/level label per degenerate block, with the evidence behind it.

        ⚠ **Inference, not a computed quantity**, which is why it is its own report and
        never a column of the state table. See :func:`kuiva.interface.api.assign_states`.
        Building the moment matrices it needs on the spin-orbit route costs one transition-
        density pass (on either solver route — the network solver contracts its transition
        densities through the applied-string Gram); pass ``matrices=`` (from a finished
        :class:`PropertyDump`) to reuse one already built.
        """
        self._check_ran()
        from .api import assign_states
        if self.solver_kind != "ci":
            return self._assign_network(matrices=matrices, tol_cm=tol_cm, report=report)
        return assign_states(self.reference_stage.reference, self.outcome,
                             matrices=matrices, tol_cm=tol_cm, report=report)

    def _assign_network(self, *, matrices, tol_cm: float, report: bool):
        """The assignment on the network route: same evidence, network contractions.

        Mirrors :func:`kuiva.interface.api.assign_states` — one spectrum, blocked once —
        with the moment matrices built from the solver's own transition densities. The
        run carries no non-abelian classification on this route, so no irrep column is
        offered.
        """
        from ..props.assign import assign_terms
        from ..props.multiplet import analyse_spectrum

        ref = self.reference_stage.reference
        if matrices is None and ref.data.has_soc:
            matrices = self._network_property_matrices()
        # One spectrum, blocked once: the blocking energies come from the matrices when
        # there are any, so the multiplet analysis and the <S^2> blocks pair up by
        # construction rather than by luck.
        energies = self._energies if matrices is None else matrices.energies
        from ..props.spin import spin_analysis as _spin_states
        spin = _spin_states(self.solver, self.coeff, self.active.spaces,
                            ref.data.s_ao, energies, tol_cm=tol_cm,
                            has_soc=ref.data.has_soc)
        multiplets = (matrices.analyse(tol_cm=tol_cm) if matrices is not None
                      else analyse_spectrum(self._energies, tol_cm=tol_cm))
        result = assign_terms(multiplets, spin, irreps=None)
        if report:
            spin.report(log)
            result.report(log)
        return result

    def _network_property_matrices(self, comments: Sequence[str] = ()):
        """``H`` and the moment matrices from the network's transition densities.

        The same assembly as :func:`kuiva.interface.api.property_matrices`, with the
        transition densities contracted through the network instead of the excitation
        map. ⚠ For the assignment's evidence only — the formatted products keep their
        routes: the property dump is the conventional-CI product, the pseudospin export
        the network one.
        """
        from ..props.dump import property_matrices as _matrices

        ref = self.reference_stage.reference
        if ref.data.properties is None:
            raise ValueError("this reference carries no angular-momentum integrals; it "
                             "was not built by the front-end")
        tdm = self.solver.transition_densities()
        provenance: Dict[str, object] = {
            "active_space": self.active.description or "explicit spinor indices",
            "n_active_spinors": int(self.active.n_active),
            "n_active_electrons": int(self.active.n_elec),
            "n_states": int(self._energies.size),
            "casscf_solver": self.solver_kind,
        }
        if ref.data.soc is not None:
            provenance["hamiltonian"] = ref.data.soc.provenance()
        provenance["basis"] = dict(ref.data.basis_meta)
        return _matrices(self.coeff, self.active.spaces, tdm, self._energies,
                         ref.data.properties, ref.data.s_ao, provenance=provenance,
                         active_space=self.active.description, comments=comments)

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
            # ⚠ Beside the bond dimension, never instead of it: the cap says what was
            # allowed and this says what it cost. An energy from this solver is not quotable
            # without it.
            entries.append(("largest discarded weight",
                            "{:.3e}".format(self.max_discarded)))
        if self.deadline is not None and self.deadline.fired:
            # ⚠ Beside "converged: False", not instead of it: an unconverged result and the
            # reason it is unconverged are two different things a reader needs.
            entries.append(("stopped by", "the deadline ({})".format(self.deadline.source)))
        if self.project_from is not None:
            entries.append(("projected active-space overlap",
                            "{:.6f}".format(self.projection.fidelity)))
        if self.avas is not None:
            # ⚠ Beside the space, not instead of it: the gap says whether the projection
            # spectrum or the threshold is what chose these orbitals.
            entries.append(("AVAS eigenvalue gap at the cut",
                            "{:.3f}".format(self.avas.gap)))
        return entries


# --- 5. the fixed-orbital CI -----------------------------------------------------------------

class CASCI(_Stage):
    """A full CI at **fixed orbitals** — the scan primitive of this API.

    ``upstream`` is any finished stage that carries orbitals: a :class:`CASSCF` (the usual
    one — spend one converged orbital set on a second spectrum without paying for a second
    optimization), a :class:`CheapCI`, or a plain :class:`Reference` (the CI at the SCF
    guess, which is what a CASSCF starts from). The orbitals come from that stage and so
    does the active space; what is varied here is everything else — the number of states, a
    per-irrep request, the CI symmetry mode, a Davidson tolerance, or the active space
    restated against those same orbitals::

        cas      = kuiva.CASSCF(ref, character=("Ti", "d"), n_active=10,
                                n_active_elec=1, n_states=2).run()   # orbitals for the doublet
        spectrum = kuiva.CASCI(cas, n_states=10).run()       # all ten, at those orbitals

    ⚠ **Two rules, and they are one rule stated twice: a statement about orbitals belongs
    to the orbitals it was made against.**

    * ``character=`` and ``avas=`` read atomic populations off the *reference's own* SCF
      orbitals, so they may be stated only on a :class:`Reference` upstream — where those
      are also the orbitals the CI runs at. On a :class:`CheapCI` or :class:`CASSCF`
      upstream the space is **inherited**, and an active space varied at fixed orbitals is
      stated as ``active=`` (spinor indices into the orbital set at hand, or an already
      resolved ``ActiveSpace``), which is a statement about exactly those orbitals. The
      orbitals have moved, so re-running a character selection against them may
      legitimately return a different set — and the spectrum would then not be the one
      belonging to these orbitals, with nothing in the output saying so.
    * ``coeff=`` is accepted only where there is nothing to inherit: on a
      :class:`Reference` upstream, for orbitals that came from somewhere else (a
      checkpoint, another program), and then with ``active=`` for the same reason.
      Elsewhere the chain already answers "which orbitals", and two answers to that is how
      a state set and an orbital set stop matching.

    ``n_states`` is a count, or a per-irrep mapping ``{irrep: n}`` wherever :class:`CASSCF`
    accepts one. ``solver_options`` are :class:`~kuiva.mcscf.casci.FullCISolver`'s —
    ``kramers="restricted"``, ``conv_tol``, ``enforce_kramers``, ``degeneracy_tol``, ... —
    and ``classify=False`` switches off the full-double-group labelling of the converged
    blocks.

    ⚠ **The state-averaging gate applies here exactly as it does to a CASSCF**: the weights
    are equalized inside a degenerate block and a count that splits one is refused. What
    does *not* run is the state-average boundary diagnostic, which is a statement about an
    orbital *trajectory* and there is none here.

    Feeds :class:`NEVPT2` and :class:`PropertyDump` exactly as a :class:`CASSCF` does — but
    not :class:`PseudospinExport`, which consumes converged orbitals rather than states and
    therefore belongs on the stage that produced them.

    A ``solver="dmrg"`` CASSCF is a legal upstream as well — an exact CI at network-converged
    orbitals is a real check on a truncated result — but this is the conventional CI, so its
    determinant ceiling applies and past it the memory ledger refuses before it allocates.

    After :meth:`run`: :attr:`energy` (state-averaged), :attr:`energies` (total state
    energies [Eh]), :attr:`coeff`, :attr:`active`, the
    :class:`~kuiva.mcscf.casci.CASCIResult` as :attr:`result`, and
    :meth:`spin_analysis` / :meth:`assign` as on :class:`CASSCF`.
    """

    #: The arguments that configure an active-space *selection*, for the refusal below. A
    #: stage that inherits its space must not silently ignore one of these.
    _SELECTION = ("n_active", "n_active_elec", "threshold")

    def __init__(self, upstream, *, active=None, character=None,
                 n_active: Optional[int] = None, n_active_elec: Optional[int] = None,
                 threshold: Optional[float] = None, avas=None, n_states=1, weights=None,
                 coeff: Optional[np.ndarray] = None,
                 solver_options: Optional[Dict[str, Any]] = None,
                 classify: bool = True, report: bool = True) -> None:
        super().__init__()
        from ..mcscf.casci import FullCISolver
        self._finished(upstream, (Reference, CheapCI, CASSCF), "CASCI")
        self.upstream = upstream
        self.reference_stage = (upstream if isinstance(upstream, Reference)
                                else upstream.reference_stage)
        #: The CI method behind this stage, in the vocabulary :class:`CASSCF` uses for it, so
        #: that a consumer of either stage asks one question. A fixed-orbital CI is the
        #: conventional-CI route by construction.
        self.solver_kind = "ci"
        self.state_request = dict(n_states) if isinstance(n_states, dict) else None
        self.n_states = (sum(int(v) for v in n_states.values())
                         if self.state_request is not None else int(n_states))
        self.weights = weights
        self.classify, self.report = bool(classify), bool(report)
        self.solver_options = dict(solver_options or {})
        _check_options(self.solver_options,
                       _allowed_options(FullCISolver.__init__,
                                        exclude=("self", "n_spinor", "n_elec", "n_states",
                                                 "weights")),
                       "CASCI solver_options (FullCISolver)")

        inherits = not isinstance(upstream, Reference)
        what = type(upstream).__name__
        if inherits and (character is not None or avas is not None):
            raise ValueError(
                "character= and avas= select against the orbitals the reference's own SCF "
                "produced, and this CASCI runs at the {}'s orbitals instead: those have "
                "moved, so the selection could legitimately return a different set and the "
                "spectrum would not be the one belonging to them. The space is inherited "
                "from the {} (leave it out), or state it as active=[spinor indices], which "
                "is a statement about the orbitals at hand".format(what, what))
        if coeff is not None:
            if inherits:
                raise ValueError(
                    "coeff= and a {} upstream are two answers to 'which orbitals', and two "
                    "answers is how a state set and an orbital set stop matching. Build this "
                    "stage on the Reference to run at orbitals of your own, or on the {} to "
                    "run at its".format(what, what))
            if character is not None or avas is not None:
                raise ValueError(
                    "character= and avas= select against the reference's own SCF orbitals "
                    "and coeff= replaces them, so the selection would not describe the "
                    "orbitals the CI runs at; state the space as active=[spinor indices]")
        stated = [name + "=" for name, value in zip(self._SELECTION,
                                                    (n_active, n_active_elec, threshold))
                  if value is not None]
        if stated and inherits and active is None and character is None and avas is None:
            # Silently ignoring these would leave a run whose active space is not the one
            # its arguments describe -- on a Reference upstream they fall through to the
            # "no active space was stated" guidance instead, which is the real problem there.
            raise ValueError(
                "{} tune an active-space selection and none was requested here: this stage "
                "inherits its active space from the {}. Drop them, or state the space with "
                "active=[spinor indices]".format(", ".join(stated), what))

        self.avas = None
        if avas is not None:
            if n_active_elec is not None:
                avas = dict(avas, n_active_elec=n_active_elec)
            self.avas, self._orbitals = _resolve_avas(
                self.reference_stage, upstream, avas, active=active, character=character,
                n_active=n_active, threshold=threshold, what="CASCI")
            self.space = self.avas.space
            return
        if coeff is not None:
            self._orbitals = np.ascontiguousarray(coeff)
        elif isinstance(upstream, CASSCF):
            self._orbitals = upstream.coeff
        elif isinstance(upstream, CheapCI):
            self._orbitals = upstream.orbitals
        else:
            self._orbitals = None                # the reference's own guess spinors
        if active is not None or character is not None:
            self.space = _resolve_space(self.reference_stage, active=active,
                                        character=character, n_active=n_active,
                                        n_active_elec=n_active_elec, threshold=threshold,
                                        what="CASCI")
        elif inherits:
            self.space = (upstream.active if isinstance(upstream, CASSCF)
                          else upstream.space)
        else:
            _resolve_space(self.reference_stage, active=None, character=None, n_active=None,
                           n_active_elec=None, threshold=None,
                           what="CASCI")                     # raises with the guidance

    def _execute(self) -> None:
        from .api import casci as _api_casci
        if self.avas is not None and self.report:
            self.avas.report(log)
        self.result = _api_casci(
            self.reference_stage.reference, active=self.space,
            n_states=(self.state_request if self.state_request is not None
                      else self.n_states),
            weights=self.weights, coeff=self._orbitals, report=self.report,
            classify=self.classify, **self.solver_options)
        #: The same object under the two names the rest of the layer knows it by: `ci` is
        #: what a CASSCF calls its spectrum, `states` is what the property stages consume.
        self.ci = self.states = self.result
        self.active = self.space
        self.solver = self.result.solver
        self._energies = np.asarray(self.result.total_energies, dtype=float)

    # -- results ------------------------------------------------------------------------------

    @property
    def energy(self) -> float:
        """State-averaged total energy [Eh] — ⚠ at these orbitals, which are not
        stationary for this average unless the upstream CASSCF optimized exactly it."""
        self._check_ran()
        return float(self.result.energy)

    @property
    def energies(self) -> np.ndarray:
        """Total state energies [Eh], ascending — the spin-orbit spectrum."""
        self._check_ran()
        return self._energies

    @property
    def coeff(self) -> np.ndarray:
        """The spinor coefficients (AO basis) the CI was solved at."""
        self._check_ran()
        return self.result.coeff

    def spin_analysis(self, *, tol_cm: float = 1.0, report: bool = False):
        """``<S^2>`` per degenerate block; see :meth:`CASSCF.spin_analysis`."""
        self._check_ran()
        from .api import spin_analysis as _spin
        return _spin(self.reference_stage.reference, self.result, tol_cm=tol_cm,
                     report=report)

    def assign(self, *, matrices=None, tol_cm: float = 1.0, report: bool = True):
        """Offer a term label per degenerate block; ⚠ inference, see :meth:`CASSCF.assign`."""
        self._check_ran()
        from .api import assign_states
        return assign_states(self.reference_stage.reference, self.result,
                             matrices=matrices, tol_cm=tol_cm, report=report)

    def _summary_entries(self):
        if self.avas is not None:
            orbitals = "the AVAS-rotated reference orbitals"
        elif isinstance(self.upstream, CASSCF):
            orbitals = "the converged CASSCF orbitals"
        elif isinstance(self.upstream, CheapCI):
            orbitals = "the pre-optimized orbitals"
        elif self._orbitals is not None:
            orbitals = "given as coeff="
        else:
            orbitals = "the reference's guess spinors"
        return [
            ("active space", "CAS({}, {})  {}".format(self.active.n_elec,
                                                      self.active.n_active,
                                                      self.active.description)),
            ("orbitals", orbitals),
            ("<E> [Eh]", out.E_FMT.format(self.energy)),
            ("E(state 0) [Eh]", out.E_FMT.format(float(self._energies[0]))),
            ("states", str(self._energies.size)),
            ("determinants", str(self.solver.ndet)),
            ("applications of H", str(self.result.n_apply)),
        ]


# --- 6. SC-NEVPT2 ---------------------------------------------------------------------------

class NEVPT2(_Stage):
    """SC-NEVPT2 dynamic correlation on a converged reference — post-processing, per state.

    ``source`` is a finished :class:`CASSCF` or :class:`CASCI`: the correction is a
    post-processing step over converged orbitals and their states, and whether those
    orbitals were optimized for this state average is the caller's statement, not this
    stage's. ⚠ On a :class:`CASCI` the reference is a CASCI wavefunction and the total is
    ``E(CASCI) + E2`` — a different reference from ``E(CASSCF) + E2`` and not comparable
    with it.

    Wraps :func:`kuiva.pt.nevpt2.sc_nevpt2`; options (``frozen_core``, ``deleted_virtual``,
    ``shift``, ``imaginary_shift``, ``fock``, ``classes``, ...) are its keywords, validated
    eagerly. Works on **both** solver routes through the driver's engine seam: the
    conventional-CI route supplies its stored CI vectors, a ``solver="dmrg"`` CASSCF
    supplies its converged network through the network-backed contraction provider
    (:mod:`kuiva.pt.network`).

    ⚠ **On the network route the correction is a PARTIAL E2 — six of the eight classes.**
    The primed single-external classes (``Sr (-1')``, ``Si (+1')``) are not served by the
    network provider yet (the scalable route is a per-label perturber network, a separate
    piece of work); they are skipped with a warning, :attr:`result` ``.complete`` is
    ``False``, the report prints the total as PARTIAL, and a partial correction is not
    comparable with another program's NEVPT2. The refusal machinery is the driver's
    standing one, not a special case.

    After :meth:`run`: :attr:`e2` [Eh, per state], :attr:`total_energies`,
    :attr:`class_energies`, :meth:`multiplets`, and the full
    :class:`~kuiva.pt.nevpt2.NEVPT2Result` as :attr:`result`.
    """

    _EXCLUDE = ("factors", "h_ao", "c_spinor", "spaces", "civecs", "solver", "n_elec",
                "energies", "weights", "e_nuc")

    def __init__(self, source, **options) -> None:
        super().__init__()
        from ..pt.network import sc_nevpt2_dmrg
        from ..pt.nevpt2 import sc_nevpt2
        #: The stage whose orbitals and states are being corrected -- a CASSCF or a CASCI.
        self.states_stage = self._finished(source, (CASSCF, CASCI), "NEVPT2")
        self.reference_stage = source.reference_stage
        _check_options(options, _allowed_options(sc_nevpt2, sc_nevpt2_dmrg,
                                                 exclude=self._EXCLUDE),
                       "NEVPT2")
        if "classes" in options and options["classes"] is not None:
            from ..pt.classes import excitation_class
            for name in options["classes"]:
                excitation_class(name)                       # refuse an unknown name now
        self.options = dict(options)

    def _execute(self) -> None:
        ref = self.reference_stage.reference
        cas = self.states_stage
        if cas.solver_kind == "dmrg":
            from ..pt.network import sc_nevpt2_dmrg
            # energies default to the finishing solve's spectrum at the converged
            # orbitals, which is exactly what the driver's state-averaging gate needs.
            self.result = sc_nevpt2_dmrg(ref.factors, ref.h_one_electron(), cas.coeff,
                                         cas.active.spaces, cas.solver,
                                         cas.active.n_elec, e_nuc=ref.data.e_nuc,
                                         **self.options)
        else:
            # One path for both conventional-CI stages: a CASSCF and a CASCI differ in where
            # their orbitals came from, and the perturbation consumes only the orbitals, the
            # active space and the CI vectors -- which they carry under the same names.
            from ..pt.nevpt2 import sc_nevpt2
            self.result = sc_nevpt2(ref.factors, ref.h_one_electron(), cas.coeff,
                                    cas.active.spaces, cas.ci.vectors, cas.active.n_elec,
                                    energies=cas.ci.energies, e_nuc=ref.data.e_nuc,
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


# --- 7. the two formatted products -----------------------------------------------------------

class PropertyDump(_Stage):
    """The property-matrix file: ``H``, ``mu_x, mu_y, mu_z`` and ``d_x, d_y, d_z`` in the SOC
    eigenstate basis.

    The electric dipole is written by default (``include_dipole=False`` turns it off). It is
    the **total** dipole — electronic plus, on the diagonal, nuclear — so its diagonal is each
    state's dipole moment and its off-diagonal elements are transition dipoles. ⚠ Kuiva writes
    the operator and its invariants; oscillator strengths and radiative rates are the external
    property code's job, as the crystal-field analysis is.

    ``source`` is a finished ``solver="ci"`` :class:`CASSCF` or :class:`CASCI` — or a
    finished :class:`NEVPT2` on either, in which case the corrected energies replace the
    diagonal **and the header records the hybrid protocol** (``H`` from perturbation theory,
    ``mu`` from the CASSCF states); that substitution is available only through this
    argument, never as a flag, so the file and its provenance cannot be separated.

    After :meth:`run`: :attr:`matrices` (compare only through its phase-invariant
    :meth:`~kuiva.props.dump.PropertyMatrices.analyse`) and :attr:`path`.
    """

    def __init__(self, source, path, *, title: str = "", include_l_s: bool = True,
                 include_dipole: bool = True, comments: Sequence[str] = (),
                 inactive_tol: Optional[float] = None, report: bool = True) -> None:
        super().__init__()
        self._finished(source, (CASSCF, CASCI, NEVPT2), "PropertyDump")
        self.source = source
        #: The stage that produced the states -- a CASSCF or a CASCI, reached through the
        #: NEVPT2 when the energies are corrected ones.
        self.states_stage = (source.states_stage if isinstance(source, NEVPT2) else source)
        if self.states_stage.solver_kind != "ci":
            raise ValueError(
                "the property dump needs the transition densities of the conventional-CI "
                "solver; the tensor-network route to properties is the pseudospin export "
                "(PseudospinExport)")
        data = self.states_stage.reference_stage.reference.data
        if data.properties is None:
            raise ValueError("this reference carries no angular-momentum integrals; it was "
                             "not built by the front-end")
        self.path = Path(path)
        self.title, self.include_l_s = str(title), bool(include_l_s)
        self.include_dipole = bool(include_dipole)
        self.comments, self.inactive_tol = tuple(comments), inactive_tol
        self.report = bool(report)

    def _execute(self) -> None:
        kwargs = {} if self.inactive_tol is None else {"inactive_tol": self.inactive_tol}
        matrices = _property_matrices(self.states_stage.reference_stage.reference,
                                      self.states_stage.states, comments=self.comments,
                                      **kwargs)
        if isinstance(self.source, NEVPT2):
            from ..pt.nevpt2 import corrected_property_matrices
            matrices = corrected_property_matrices(matrices, self.source.result)
        if self.report:
            out.section(log, "Property matrices")
            matrices.report(log)
        self.matrices = matrices
        matrices.write(self.path, title=self.title, include_l_s=self.include_l_s,
                       include_dipole=self.include_dipole)

    def assign(self, *, tol_cm: float = 1.0, report: bool = True):
        """The term assignment of these states, reusing the moment matrices already built.

        ⚠ Inference; see :meth:`CASSCF.assign`, which this forwards to with ``matrices=``
        already filled in — so it costs one ``<S^2>`` pass and no second transition-density
        pass.
        """
        self._check_ran()
        return self.states_stage.assign(matrices=self.matrices, tol_cm=tol_cm, report=report)

    def _summary_entries(self):
        return [
            ("file", str(self.path)),
            ("states", str(self.matrices.n_states)),
            ("energies", "NEVPT2-corrected (hybrid protocol, recorded)"
             if isinstance(self.source, NEVPT2)
             else type(self.states_stage).__name__),
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

    ⚠ **This one takes a CASSCF and not a :class:`CASCI`**, unlike :class:`PropertyDump`:
    what it consumes is the converged *orbitals*, not the states — the model space is
    re-solved from the integrals those orbitals define — so it belongs on the stage that
    optimized them, and a CASCI's orbitals are always some other stage's.

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
