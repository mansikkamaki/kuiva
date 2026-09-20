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
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util.logging import get_logger
from .pyscf_bridge import ScalarX2CData, validate_scf_controls
from ..util.signals import raise_if_pending, stop_context
from .api import (Molecule, SpinorReference as _SpinorData, active_space_for,
                  localize_active_space,
                  project_to_basis as _project_to_basis, projected_active_space,
                  property_matrices as _property_matrices, scalar_x2c_reference,
                  spinor_reference)

log = get_logger(__name__)

__all__ = ["ScalarSCF", "Reference", "AutoCAS", "CheapCI", "CASSCF", "CASCI", "NEVPT2",
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

    ``n_states`` is a count or an :class:`~kuiva.util.window.EnergyWindow`. With a window the
    count is resolved on the **reference space's** spectrum at the first solve and held for
    the orbital loop (one round: this stage is qualitative and its count is an estimate),
    then re-resolved at the pre-optimized orbitals for the report. ⚠ What that count is for
    is the *handoff*: a :class:`CASSCF` built on this stage and given its own window takes
    the ladder's first **rung** from :attr:`spectrum_cm`, never a verdict — a truncated CI
    in a truncated space says roughly how many roots lie inside a cutoff and no more.

    After :meth:`run`: :attr:`orbitals`, :attr:`occupations`, :attr:`natural_occupation`,
    :attr:`entropy`, :attr:`mutual_information`, :attr:`n_states`, :attr:`spectrum_cm`,
    :attr:`window`, :meth:`suggested_active`, :meth:`dmrg_ordering`, the full
    :class:`~kuiva.mcscf.preopt.PreoptResult` as :attr:`result`, and (with ``avas=``) the
    :class:`~kuiva.mcscf.avas.AVASResult` as :attr:`avas`.
    """

    _EXCLUDE = ("factors", "h_ao", "c_spinor", "spaces", "n_active_elec", "e_nuc",
                "h_eff", "eri", "n_elec")

    def __init__(self, reference, *, active=None, character=None,
                 n_active: Optional[int] = None, n_active_elec: Optional[int] = None,
                 threshold: Optional[float] = None, avas=None, n_states=1,
                 **options) -> None:
        super().__init__()
        from ..mcscf.preopt import cheap_ci, preoptimize
        from ..util.window import EnergyWindow, is_window_request
        self._finished(reference, (Reference, AutoCAS), "CheapCI")
        self.upstream = reference
        self.reference_stage = (reference if isinstance(reference, Reference)
                                else reference.reference_stage)
        if isinstance(reference, AutoCAS) and character is not None:
            raise ValueError(
                "character= selects against the orbitals the reference's own SCF produced, "
                "and this cheap CI starts from the AutoCAS's instead: the AVAS projection "
                "rotated them, so the selection could legitimately return a different set. "
                "The space is inherited from the AutoCAS (leave it out), or stated as "
                "active=[spinor indices]")
        _check_options(options, _allowed_options(preoptimize, cheap_ci,
                                                 exclude=self._EXCLUDE + ("n_states",)),
                       "CheapCI")
        #: The energy window the count is resolved from, or ``None`` for a stated count.
        self.window_request: Optional[EnergyWindow] = None
        #: The resolution at the pre-optimized orbitals after :meth:`run`.
        self.window = None
        if is_window_request(n_states):
            self.window_request = n_states
            #: ``None`` until :meth:`run` with a window; the resolved count after it.
            self.n_states: Optional[int] = None
        else:
            self.n_states = int(n_states)
        self._n_states_arg = n_states
        self.avas, self._orbitals = None, None
        if avas is not None:
            if n_active_elec is not None:
                avas = dict(avas, n_active_elec=n_active_elec)
            self.avas, self._orbitals = _resolve_avas(
                self.reference_stage, reference, avas, active=active, character=character,
                n_active=n_active, threshold=threshold, what="CheapCI")
            self.space = self.avas.space
        elif isinstance(reference, AutoCAS) and active is None:
            # The space and the orbitals travel together, exactly as they do from a CheapCI
            # into a CASSCF: what the upstream selected is a statement about the orbitals it
            # is handing over.
            self.space, self._orbitals = reference.space, reference.orbitals
        else:
            self.space = _resolve_space(self.reference_stage, active=active,
                                        character=character, n_active=n_active,
                                        n_active_elec=n_active_elec, threshold=threshold,
                                        what="CheapCI")
            if isinstance(reference, AutoCAS):
                self._orbitals = reference.orbitals
        self.options = dict(options)

    def _execute(self) -> None:
        from ..mcscf.preopt import preoptimize, repair_kramers_pairing
        ref = self.reference_stage.reference
        start = self._orbitals if self._orbitals is not None else ref.spinors_in_ao()
        if self.avas is not None:
            self.avas.report(log)
        self.result = preoptimize(ref.factors, ref.h_one_electron(), start,
                                  self.space.spaces, self.space.n_elec,
                                  e_nuc=ref.data.e_nuc, n_states=self._n_states_arg,
                                  **self.options)
        # ⚠ Restore exact Kramers pairing before anything downstream consumes the orbitals.
        # The cheap CI's truncated determinant space is not closed under time reversal, so
        # the orbitals it optimizes drift off pairing — legitimately, it is a *cheap* stage —
        # while every consumer of this stage (the state-averaging gate, a contiguous-pair
        # active space) assumes the pairing convention exactly. One implementation, in
        # kuiva.mcscf.preopt beside the warning that says a consumer must do it.
        self.orbitals, deviation = repair_kramers_pairing(
            self.result.coeff, ref.orth.x, ref.data.s_ao, self.space.spaces)
        log.debug("Kramers pairing restored on the pre-optimized orbitals (worst partner "
                  "deviation before repair: %.2e)", deviation)
        self.occupations = self.result.orbital_occupation
        self.natural_occupation = self.result.natural_occupation
        self.entropy = self.result.entropy
        self.mutual_information = self.result.mutual_information
        #: The state spectrum of the analysis solve, relative to the lowest [cm^-1]. ⚠ The
        #: read side of the handoff, and **qualitative**: a downstream energy window takes
        #: its ladder's first *rung* from it, never a verdict.
        self.spectrum_cm = self.result.ci.relative_cm
        self.window = self.result.ci.window
        self.n_states = int(np.asarray(self.result.ci.energies).size)

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
        entries = [
            ("active space", "CAS({}, {})  {}".format(self.space.n_elec,
                                                      self.space.n_active,
                                                      self.space.description)),
            ("converged (occupations)", str(self.result.converged)),
            ("orbital occupations", occ),
            ("suggested active spinors", str(self.suggested_active().tolist())),
        ]
        if self.window_request is not None:
            entries.insert(1, ("states", "{} (resolved from {})".format(
                self.n_states, self.window_request)))
        return entries



class _Unstated:
    """The value of ``n_states`` when the caller did not state one.

    ⚠ It exists so that "unstated" and "one state" stay different things. The proposal of an
    :class:`AutoCAS` upstream is taken **only** where nothing was said -- an explicit
    ``n_states=1`` is a request for one state and stays one -- and a plain default of ``1``
    could not tell the two apart.
    """

    def __repr__(self) -> str:                              # pragma: no cover - debugging
        return "<unstated>"


_UNSTATED = _Unstated()


def _states_from_upstream(upstream, n_states, *, what: str):
    """``(n_states, source)``: the proposal of an :class:`AutoCAS` upstream, or the default.

    ⚠ **This is the one place a stage default changes on its upstream's account, and it is
    announced** in the stage's own output. The count is taken where the proposal has one; a
    proposal whose boundary lies above the cap has *no* count -- proposing a capped one would
    be proposing a cut inside a manifold -- so there the window is taken instead, which
    refuses rather than rounds.
    """
    if n_states is not _UNSTATED:
        return n_states, ""
    if not isinstance(upstream, AutoCAS):
        return 1, ""
    if upstream.n_states is not None:
        return upstream.n_states, "proposed by AutoCAS"
    return upstream.window, ("proposed by AutoCAS as a window: its manifold boundary is "
                             "above the state cap, so there is no count to propose")


# --- 4. the automatic active space -----------------------------------------------------------

class AutoCAS(_Stage):
    """Choose the active space automatically from **stated targets**, and propose a count.

    A stage between :class:`Reference` and the production one
    (:class:`CASSCF` / :class:`CASCI` / :class:`CheapCI`), which accept it wherever they
    accept a :class:`CheapCI`. It assembles an active space from *physical statements* of what
    the calculation has to describe, decides each feature class by probing the cheap CI, and
    hands on what a :class:`CheapCI` hands on -- orbitals, a stated space, a spectrum and an
    ordering -- plus a proposed number of states::

        auto = kuiva.AutoCAS(ref).run()                       # the detected d/f shells
        cas  = kuiva.CASSCF(auto).run()                        # space, orbitals and count

    ``targets`` is one statement or a list of them; ``None`` means "the valence shell of every
    open d/f centre the reference shows", which is the level-0 default::

        "shells"                                  every detected centre's valence shell
        ("shell", "Dy")                           that element's centres
        ("shell", ("Ti1", "Ti2"), "d")            these atoms, this shell
        ("frontier", ("N1", "N2"))                a radical fragment's singly occupied pairs
        ("frontier", ("N1", "N2"), 1, 1)          ... plus one occupied and one empty pair
        ("bridge", ("Dy1", "Dy2"))                ligand orbitals between the two sites
        ("bridge", ("Dy1", "Dy2"), ("O5",), 2)    ... on named atoms, at most two pairs
        ("bonding", "Fe")                         the shell's metal-ligand bonding partners
        ("double", "Ce")                          the correlating shell

    Atoms are addressed as everywhere else (an element symbol, a label ``"Dy2"``, a 1-based
    atom number, or a sequence of those for a fragment), and an element symbol **pools** every
    atom of that element.

    How a class is decided
    ----------------------
    ⚠ **The cheap CI probes, the spectrum decides, and entropy only prunes.** The shells are
    the core; each further class is offered in a fixed priority order, its candidates pruned
    by relative single-orbital entropy, and the class kept only if the *target manifold* --
    the ground manifold's relative energies and the gap above it -- moved by more than
    ``max(spectrum_tol x the manifold width, spectrum_tol_cm)``. A change under the tolerance
    but above the measured noise floor is **"inconclusive, kept"**: a larger space is the safe
    error, and the budget is what bounds it. Every verdict is printed with the number it was
    taken on.

    ⚠ **Every verdict is taken at fixed orbitals** -- the cheap CI at the candidate
    construction's orbitals, each trial's determinants nested in the accepted space's -- and
    the accepted space is pre-optimized once, at the end. Comparing two pre-optimizations
    read the optimizer's trajectory: pairs that describe nothing moved the spectrum by up to
    100 cm^-1, where at fixed orbitals they move it by a fraction of one. ⚠ And the probe's
    determinant budget (``probe={"max_determinants": ...}``) must hold the product of the
    sites' Hund configurations -- 32 768 for three d^5 ions -- or the stage refuses: below it a
    selected CI cannot represent a coupled system's ground manifold at all.

    ⚠ **A target shell is never pruned and never cut.** Its empty members *are* the
    ligand-field spectrum, and a shell with pairs missing is a different physical statement
    wearing the shell's name -- so over budget whole classes are dropped, lowest priority
    first, and the core alone over budget **refuses** with the two ways out named.

    ⚠ **The proposal is a proposal.** The count is read off a qualitative probe as the first
    manifold boundary at or above the Hund ground-manifold floor; what makes it a state count
    is the production stage's own machinery -- the state-averaging gate, the boundary
    diagnostic at both ends, the window's ladder. :attr:`window` is the equivalent
    :class:`~kuiva.util.window.EnergyWindow`, whose cutoff sits in the middle of the gap the
    count was read at, so a production ladder re-resolves the same boundary against its own
    spectrum.

    Requirements and limits
    -----------------------
    ⚠ Needs ``atomic_reference=True`` on the :class:`ScalarSCF` (every shell is an AVAS
    projection onto the free-atom orbitals) and a restricted or ROHF reference; both are
    refused at construction, naming the knob. ⚠ Its space carries **no symmetry labels**, as
    any AVAS space does -- the labels belong to the guess spinors and the projection has
    rotated them -- so a per-irrep ``n_states`` is unavailable downstream. ⚠ Shells of two
    different ``l`` (a heteronuclear 3d/4f pair) are **one** AVAS projection onto the union of
    their reference shells, each pair attributed to the shell it projects onto most -- never
    one projection per ``l``, whose second rotation re-mixes the first one's selection.

    ``max_spinors`` / ``max_determinants`` bound the size; unstated, the budget is resolved
    from the configured memory limit at the floor root count for ``solver="ci"`` and is a
    provisional default for ``solver="dmrg"``. ``require=`` and ``exclude=`` pin or ban
    orbitals by a character statement -- ``("character", atom, l, n_spinors[, skip_pairs])``
    -- for what the probe cannot see; ``require=`` also takes ``("avas", atom, l,
    n_spinors)``, pairs of that reference shell selected by the core's own union projection.

    ⚠ **The cost is printed per round.** One fixed-orbital CI per round, one more for every
    prune that removed something, and one pre-optimization of the accepted space.

    After :meth:`run`: :attr:`space`, :attr:`orbitals`, :attr:`n_states`, :attr:`window`,
    :attr:`floor`, :attr:`product_floor`, :attr:`spectrum_cm`, :attr:`rounds`,
    :attr:`mutual_information`, :attr:`entropy`, :attr:`occupations`, :attr:`sites`,
    :attr:`assembly` (the whole :class:`~kuiva.autocas.protocol.Assembly`), :attr:`result`
    (the final :class:`~kuiva.mcscf.preopt.PreoptResult`), :meth:`dmrg_ordering`,
    :meth:`fiedler_ordering`.
    """

    def __init__(self, reference: Reference, targets=None, *, solver: str = "ci",
                 max_spinors: Optional[int] = None,
                 max_determinants: Optional[int] = None,
                 max_states: Optional[int] = None,
                 spectrum_tol: Optional[float] = None,
                 spectrum_tol_cm: Optional[float] = None,
                 probe_noise_cm: Optional[float] = None,
                 prune_rel: Optional[float] = None,
                 manifold_gap_cm: Optional[float] = None,
                 require=(), exclude=(), probe: Optional[Dict[str, Any]] = None,
                 localize: bool = True, report: bool = True) -> None:
        super().__init__()
        from ..autocas import centres as _centres
        from ..autocas import protocol as _protocol
        from ..autocas.probe import ProbeBudget
        from ..autocas.targets import parse_targets

        self.reference_stage = self._finished(reference, Reference, "AutoCAS")
        # Eager, and in this order: the two prerequisites of the method, then the statement
        # itself. A misspelled target must fail here and not after the first probe.
        _centres.check_reference(reference.reference)
        self.targets = parse_targets(targets)
        if solver not in ("ci", "dmrg"):
            raise ValueError("solver must be 'ci' or 'dmrg'; got {!r}".format(solver))
        self.solver_kind = solver
        _check_options(dict(probe or {}),
                       _allowed_options(ProbeBudget.__init__, exclude=("self",)),
                       "AutoCAS probe")
        self.probe_budget = ProbeBudget(**dict(probe or {}))
        self.localize = bool(localize)
        self.report = bool(report)
        #: Everything that is not a target, forwarded to
        #: :func:`kuiva.autocas.protocol.assemble` with the module's own defaults where
        #: nothing was stated -- so the defaults live in one place and this stage never
        #: restates a number.
        self.options = {name: value for name, value in (
            ("solver", solver), ("max_spinors", max_spinors),
            ("max_determinants", max_determinants), ("max_states", max_states),
            ("spectrum_tol", spectrum_tol), ("spectrum_tol_cm", spectrum_tol_cm),
            ("probe_noise_cm", probe_noise_cm), ("prune_rel", prune_rel),
            ("manifold_gap_cm", manifold_gap_cm)) if value is not None}
        self.options["require"] = tuple(require)
        self.options["exclude"] = tuple(exclude)
        # A stated size is validated now, against nothing expensive: two statements of one
        # size, or a determinant bound on a route that has no determinants.
        _protocol.resolve_budget(solver=solver, n_elec=2, n_roots=2, max_spinors=max_spinors,
                                 max_determinants=max_determinants, available_gb=1.0e9)

    def _execute(self) -> None:
        from ..autocas.protocol import assemble

        reference = self.reference_stage.reference
        self.assembly = assemble(reference, self.targets, probe_budget=self.probe_budget,
                                 report=self.report, **self.options)
        #: The assembled :class:`~kuiva.mcscf.casci.ActiveSpace`; its description is the
        #: physical statement, never an index list.
        self.space = self.assembly.space
        #: ``(2*nao, n)`` AO-basis spinors the space's columns index -- the final probe's
        #: pre-optimized, Kramers-repaired set.
        self.orbitals = self.assembly.coeff
        self.result = self.assembly.probe.result
        self.entropy = self.assembly.probe.entropy
        self.mutual_information = self.assembly.probe.mutual_information
        self.occupations = self.assembly.probe.occupations
        #: The final probe's relative state energies [cm^-1]. ⚠ Qualitative, exactly as a
        #: :class:`CheapCI`'s is: a downstream energy window takes its ladder's first *rung*
        #: from it and never a verdict.
        self.spectrum_cm = self.assembly.probe.spectrum_cm
        self.proposal = self.assembly.proposal
        #: The proposed count, or ``None`` where the boundary is above the cap -- there the
        #: window is the proposal and a count would be a cut inside the manifold.
        self.n_states = self.proposal.n_states
        #: The equivalent :class:`~kuiva.util.window.EnergyWindow`.
        self.window = self.proposal.window
        self.floor = self.assembly.floor
        self.product_floor = self.assembly.product_floor
        self.rounds = self.assembly.rounds
        self.budget = self.assembly.budget
        self.sites = None
        self._localize()

    def _localize(self) -> None:
        """Localize the shells per centre, so the space has sites as well as orbitals.

        ⚠ **A warning here, not the refusal the localizer raises.** A localization that did
        not reach its population floor means the site *labels* are not trustworthy; the active
        space is exactly as valid as it was, and what falls back is the tensor-network
        ordering. It goes through :func:`kuiva.interface.api.localize_active_space` -- the
        project's one site partition -- rather than a second implementation of "which centre
        is this orbital on".
        """
        if not self.localize or len(self.assembly.site_atoms) < 2:
            return
        from ..autocas.protocol import shell_columns_after_probe

        reference = self.reference_stage.reference
        columns, note = shell_columns_after_probe(reference, self.assembly)
        if columns is None:
            log.warning("%s", note)
            return
        sites = [list(atoms) for atoms in self.assembly.site_atoms]
        # Shell spinors per site from the centres, never an equal split: a 3d centre beside a
        # 4f one is ten spinors and fourteen.
        counts = (list(self.assembly.site_counts)
                  or [int(columns.size // len(sites))] * len(sites))
        try:
            localization = localize_active_space(
                reference, self.space, sites, coeff=self.orbitals, columns=columns,
                counts=counts, report=self.report)
        except ValueError as exc:
            log.warning("the shells did not localize onto the individual centres, so no site "
                        "partition is claimed and a tensor-network ordering falls back to the "
                        "Fiedler order: %s", exc)
            return
        self.orbitals = localization.coeff
        self.sites = tuple(tuple(int(c) for c in localization.site_columns(i))
                           for i in range(localization.n_sites))
        self.assembly.sites = self.sites

    def dmrg_ordering(self) -> np.ndarray:
        """Mode order for a tensor network: **site-blocked** where the sites are known.

        ⚠ Unlike :meth:`CheapCI.dmrg_ordering`, which is always the Fiedler order of the
        mutual information, this prefers the site partition -- because the topology of a
        polynuclear space comes from which centre an orbital is on and never from
        entanglement: a bridging orbital shares at most ``ln 2`` with either ion, so an
        entanglement-ordered chain puts the pathway wherever the noise of a qualitative probe
        happens to rank it. :meth:`fiedler_ordering` is the other one, by name.
        """
        self._check_ran()
        return self.assembly.dmrg_ordering()

    def fiedler_ordering(self) -> np.ndarray:
        """The Fiedler order of the probe's mutual information -- what ``graph="fiedler"``
        asks for, named so that it cannot be confused with the site-blocked one."""
        self._check_ran()
        return self.result.dmrg_ordering()

    def summary(self) -> str:
        text = super().summary()
        rows = ["  rounds:"]
        for record in self.rounds:
            rows.append("    {:>2}  {:<9} {:<22} {:>3} spinors".format(
                record.index, record.cls, record.verdict, record.n_spinors))
        return "\n".join([text] + rows)

    def _summary_entries(self):
        entries = list(self.assembly.summary_entries())
        entries.append(("orbitals", "pre-optimized, Kramers repaired"
                        + (", shells localized per site" if self.sites else "")))
        return entries

# --- 5. the CASSCF ---------------------------------------------------------------------------

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

    ⚠ **``restart=`` resumes a RUNNING optimization; :meth:`from_checkpoint` materializes a
    FINISHED one.** Both exist because re-entering the optimizer with nothing left to do ends
    its table ``NOT converged in 0 macro-iterations`` on a calculation that converged hours
    earlier. The two also default differently, and that is the distinction rather than a
    convenience: a restart continues a run whose settings the caller restates and the file
    only checks, while a materialization configures nothing and therefore takes
    ``n_states``/``weights`` from the file.

    ⚠ **The granularity is one macro-iteration.** The run stops between them and nowhere
    else, so the decision is predictive: it stops when the time left is less than the
    longest recent iteration plus the estimated checkpoint write plus a stated margin. One
    CI solve, and on the DMRG route one whole network solve, cannot be interrupted.
    ⚠ With ``checkpoint=``, the final write is **forced** past the cadence rules and happens
    before the stop; without one, the run still exits cleanly but keeps nothing, and says so.

    ``signals=`` is the other half of the same problem: the kill nobody announced —
    ``scancel``, a preemption, ``SIGTERM`` at the wall. ``signals=True`` catches
    ``SIGTERM``/``SIGUSR1``/``SIGUSR2`` (Slurm's ``--signal=B:USR1@<seconds>`` sends the
    second at a lead time you choose), and a sequence names them instead. The run then stops
    at the next macro-iteration boundary with its checkpoint written, exactly as the deadline
    does.

    ⚠ **Off by default, and installed only while this stage runs**: a library that installs
    signal handlers behind your back breaks embedding, test runners and notebooks, so the
    previous dispositions are restored when the stage ends, exception or not. ⚠ **A second
    signal is not waited for** — the handlers step aside and it acts at once. ⚠ **The
    request outlives this stage**: a :class:`NEVPT2`, :class:`CASCI` or
    :class:`PseudospinExport` started afterwards refuses with
    :class:`~kuiva.util.signals.StopRequested` rather than beginning work the process will
    not live to finish, which is what makes a signalled run *exit* rather than merely pause.

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

    Choosing the states by an energy cutoff
    ---------------------------------------
    ``n_states=kuiva.EnergyWindow(1000)`` averages over every state within 1000 cm^-1 of the
    lowest, extended to the top of any manifold the cutoff falls inside
    (:class:`~kuiva.util.window.EnergyWindow`; ``{irrep: EnergyWindow}`` beside fixed counts
    is its per-irrep form). :attr:`n_states` is then ``None`` until :meth:`run` and the
    resolved count after it, :attr:`window` carries the
    :class:`~kuiva.util.window.WindowResolution` and :attr:`rounds` the rounds it took;
    :attr:`window_initial` is the resolution at the *starting* orbitals, where the ladder's
    first rung says where it came from.

    ⚠ **The count is resolved at fixed orbitals and held for a whole orbital optimization;
    it may change only between rounds** — the optimizer never sees a window, and a run whose
    first round resolves to ``n`` and stays there is the fixed-count run at ``n``. ``max_iter``
    is therefore the budget **across** rounds, and a window that never settles keeps the last
    converged result and says so rather than picking a side (:mod:`kuiva.mcscf.rounds`).
    ⚠ ``weights=`` is refused with a window. A :class:`CheapCI` upstream supplies the
    ladder's first rung from its own spectrum; on the network route, where a rung is a whole
    sweep campaign, a short **pilot** at a small bond dimension supplies it instead when
    there is no upstream estimate, and the witness roots the verdict is read against are
    converged network roots (:mod:`kuiva.dmrg.window`). The per-irrep form of a window is a
    conventional-CI request and is refused with ``solver="dmrg"``.

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

    _GRAPH_CHOICES = ("mutual-information", "fiedler", "site-blocked")

    def __init__(self, upstream, *, active=None, character=None,
                 n_active: Optional[int] = None, n_active_elec: Optional[int] = None,
                 threshold: Optional[float] = None, avas=None, n_states=_UNSTATED,
                 weights=None,
                 solver: str = "ci", solver_options: Optional[Dict[str, Any]] = None,
                 graph=None, checkpoint=None, restart=None,
                 checkpoint_options: Optional[Dict[str, Any]] = None,
                 callback: Optional[Callable[[dict], Optional[bool]]] = None,
                 deadline=None, signals=None,
                 preserve_symmetry: bool = False, project_from=None,
                 projection: Optional[Dict[str, Any]] = None,
                 report: bool = True, **optimizer_options) -> None:
        super().__init__()
        self._finished(upstream, (Reference, CheapCI, AutoCAS), "CASSCF")
        self.upstream = upstream
        self.reference_stage = (upstream if isinstance(upstream, Reference)
                                else upstream.reference_stage)
        #: Where ``n_states`` came from when the caller did not state it, for the output.
        n_states, self.n_states_source = _states_from_upstream(upstream, n_states,
                                                               what="CASSCF")
        if not isinstance(upstream, Reference) and character is not None:
            # ⚠ A selection is resolved against the orbitals it was stated on, and this run
            # starts from the upstream's instead. Re-resolving here would read populations off
            # the reference's own SCF orbitals and then run at different ones -- the same
            # refusal CASCI and project_from= make, for the same reason.
            raise ValueError(
                "character= selects against the orbitals the reference's own SCF produced, "
                "and this CASSCF starts from the {0}'s instead: those have moved (the {0} "
                "rotated them), so the selection could legitimately return a different set "
                "and the calculation would not be the one the statement describes. The space "
                "is inherited from the {0} (leave it out), or state it as active=[spinor "
                "indices], which is a statement about the orbitals at hand"
                .format(type(upstream).__name__))
        if solver not in ("ci", "dmrg"):
            raise ValueError("solver must be 'ci' or 'dmrg'; got {!r}".format(solver))
        self.solver_kind = solver
        from ..util.window import EnergyWindow, is_window_request, shared_window
        #: The energy window the count is resolved from, or ``None`` for a stated count.
        self.window_request: Optional[EnergyWindow] = None
        #: The :class:`~kuiva.util.window.WindowResolution` at the converged orbitals after
        #: :meth:`run`; ``None`` before it and for a stated count.
        self.window = None
        #: The resolution at the **starting** orbitals — the window's counterpart of
        #: :attr:`boundary_initial`, and ``None`` for a stated count or a restart.
        self.window_initial = None
        #: The rounds a windowed run took; empty for a stated count.
        self.rounds: List[Any] = []
        #: ``n_states`` is a count, a per-irrep mapping ``{irrep: n}`` with point-group
        #: labels present, or an energy window. The three are forms of one argument; a
        #: mapping's total and a window's resolved count are what :attr:`n_states` reports,
        #: so every consumer of the count still works.
        self.state_request = dict(n_states) if isinstance(n_states, dict) else None
        if is_window_request(n_states):
            if solver != "ci" and not isinstance(n_states, EnergyWindow):
                raise ValueError(
                    "a per-irrep energy window selects states per determinant sector, which "
                    "is a conventional-CI request; a network solve targets one sector and "
                    "takes a plain n_states=kuiva.EnergyWindow(...)")
            if weights is not None:
                raise ValueError("weights= cannot be combined with an energy window: a "
                                 "window's weights are equal by construction and the "
                                 "state-averaging gate equalizes them inside degenerate "
                                 "blocks. Drop weights=, or state a count")
            self.window_request = (n_states if isinstance(n_states, EnergyWindow)
                                   else shared_window(n_states))
            self.n_states: Optional[int] = None
        else:
            self.n_states = (sum(int(v) for v in n_states.values())
                             if self.state_request is not None else int(n_states))
        self._n_states_arg = n_states
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
        from ..util.signals import SignalStop
        self.deadline = Deadline.resolve(deadline)
        # Same rule: signals= off the main thread cannot be honoured and is refused here
        # rather than at the first hour of the run.
        self.signals = SignalStop.resolve(signals)
        self.callback, self.report = callback, bool(report)
        self.optimizer_options = dict(optimizer_options)
        #: Set by :meth:`from_checkpoint`; ``None`` on an ordinary stage. Its presence is what
        #: makes :meth:`run` materialize a finished calculation instead of optimizing one.
        self._materialize: Optional[Dict[str, Any]] = None

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
                if isinstance(upstream, (CheapCI, AutoCAS)):
                    self.space = upstream.space
                else:
                    _resolve_space(self.reference_stage, active=None, character=None,
                                   n_active=None, n_active_elec=None, threshold=None,
                                   what="CASSCF")            # raises with the guidance
            self._orbitals = (upstream.orbitals
                              if isinstance(upstream, (CheapCI, AutoCAS)) else None)
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
                         "n_active_elec",
                         "e_nuc", "n_states", "weights", "solver", "active",
                         "solver_options", "callback", "report", "optimizer_state",
                         "start_iteration", "space_key", "history", "extra_columns",
                         "window_estimate", "on_round"))
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
                                 "n_active_elec", "extra_columns", "repair_orbitals"))
            _check_options(self.optimizer_options, allowed, "CASSCF (orbital optimizer)")
            if isinstance(graph, str):
                if graph not in self._GRAPH_CHOICES:
                    raise ValueError("graph= must be a NetworkGraph or one of {}; got {!r}"
                                     .format(self._GRAPH_CHOICES, graph))
                if graph == "site-blocked":
                    # ⚠ Refused rather than degraded: a site-blocked ordering is a statement
                    # about which centre each orbital is on, and a stage with no site
                    # partition has no such statement to make. Silently handing back an
                    # entanglement order would answer a different question -- and for a
                    # polynuclear space that is the wrong answer, since a bridging orbital
                    # shares at most ln 2 with either ion.
                    if not isinstance(upstream, AutoCAS) or upstream.sites is None:
                        raise ValueError(
                            "graph='site-blocked' orders the modes by which centre each "
                            "orbital sits on, which needs an AutoCAS upstream whose shells "
                            "localized onto their centres (one centre, or a localization "
                            "that did not reach its population floor, leaves no partition to "
                            "order by). Use graph='mutual-information' or 'fiedler'")
                elif not isinstance(upstream, (CheapCI, AutoCAS)):
                    raise ValueError("graph={!r} builds the topology from the cheap CI's "
                                     "entanglement, so it needs a CheapCI or AutoCAS "
                                     "upstream".format(graph))
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

        self._finished(self.project_from, (Reference, CheapCI, AutoCAS, CASSCF),
                       "CASSCF project_from")
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
        self._announce_states()
        if self.project_from is not None:
            self._run_projection()
        if self.solver_kind == "ci":
            self._execute_ci()
        else:
            self._execute_dmrg()

    def _announce_states(self) -> None:
        """Say so when the state count came from the upstream rather than from the caller.

        ⚠ A default that changes on the upstream's account has to appear in the output file,
        or a reader cannot tell a proposed average from a requested one -- and the two are
        different calculations.
        """
        if self.report and self.n_states_source:
            out.entry(log, "n_states", self.n_states if self.n_states is not None
                      else repr(self.window_request), "", self.n_states_source)

    @classmethod
    def from_checkpoint(cls, path, reference, *, n_states=None, weights=None,
                        solver_options: Optional[Dict[str, Any]] = None,
                        boundary_check: Optional[int] = None,
                        require_converged: bool = True,
                        report: bool = True, classify: bool = True) -> "CASSCF":
        """A **finished** CASSCF stage read back from its checkpoint — nothing is optimized.

        ``restart=`` resumes a running optimization; this materializes one that already
        ended::

            ref = kuiva.Reference(scf).run()
            cas = kuiva.CASSCF.from_checkpoint("run.h5", ref)      # no macro-iterations
            kuiva.PropertyDump(cas, "props.out").run()             # the dump nobody asked for

        The result is an ordinary :class:`CASSCF`, so :class:`PropertyDump`,
        :class:`NEVPT2`, :class:`CASCI` and :class:`PseudospinExport` all take it unchanged.
        See :func:`kuiva.interface.api.casscf_from_checkpoint` for what is checked, what is
        re-solved and why: in short, the file's system fingerprint must match ``reference``,
        the states are re-solved at the stored orbitals seeded by the stored CI vectors, and
        ``n_states``/``weights`` default from the file because there is no optimization left
        to configure.

        ⚠ **Only the converged-orbital boundary diagnostic comes back.** The starting-orbital
        one is a statement about a trajectory and belongs to the run that wrote the file.

        ⚠ ``n_states`` may not be an :class:`~kuiva.util.window.EnergyWindow` here. A window
        is resolved against a trajectory's orbitals and can change the count between rounds;
        there is no trajectory left, so a file written by a windowed run supplies its
        **resolved count** and keeps the window as provenance, which is the only reading that
        can be right.
        """
        from ..util.window import is_window_request
        if is_window_request(n_states):
            raise ValueError(
                "n_states= cannot be an energy window here: a materialization configures "
                "nothing and resolves nothing, so the count comes from the file (with the "
                "window it was resolved from kept as provenance). Leave n_states out, or "
                "state a count to re-solve a different number of states at these orbitals")
        stage = cls(reference, restart=path,
                    n_states=1 if n_states is None else n_states, weights=weights,
                    solver_options=solver_options, report=report, classify=classify)
        stage._materialize = {"n_states": n_states, "weights": weights,
                              "solver_options": solver_options,
                              "boundary_check": boundary_check,
                              "require_converged": require_converged,
                              "classify": classify}
        return stage

    def _execute_ci(self) -> None:
        from .api import casscf as _api_casscf
        if self._materialize is not None:
            from .api import casscf_from_checkpoint
            self._adopt_ci(casscf_from_checkpoint(self.reference_stage.reference,
                                                  self.restart, report=self.report,
                                                  **self._materialize))
            return
        options = dict(self.optimizer_options)
        if self.window_request is not None:
            options["window_estimate"] = self._window_first_rung()
        outcome = _api_casscf(self.reference_stage.reference, active=self.space,
                              n_states=self._n_states_arg,
                              preserve_symmetry=self.preserve_symmetry,
                              weights=self.weights,
                              coeff=self._orbitals, checkpoint=self.checkpoint,
                              restart=self.restart,
                              checkpoint_options=self.checkpoint_options,
                              solver_options=self.solver_options, callback=self.callback,
                              deadline=self.deadline, signals=self.signals,
                              report=self.report, **options)
        self._adopt_ci(outcome)

    def _window_first_rung(self) -> Optional[int]:
        """The ladder's first rung, from an upstream :class:`CheapCI`'s own spectrum.

        ⚠ **A rung, never a verdict.** The cheap CI's energies are qualitative by
        construction — a truncated CI in a truncated space — so what its spectrum is good for
        is saying roughly how many roots lie inside the cutoff, which is exactly what the
        first rung is. The rule then runs on the full CI's spectrum from there, and the
        output states where the rung came from. With no upstream spectrum (a ``Reference``
        upstream, or a cheap CI run at one root) the ladder takes its own default, and
        ``EnergyWindow(initial=)`` overrides both.
        """
        spectrum = getattr(self.upstream, "spectrum_cm", None)
        if spectrum is None:
            return None
        from ..util.units import HARTREE_TO_CM
        from ..util.window import resolve_window
        energies = np.asarray(spectrum, dtype=float).ravel() / HARTREE_TO_CM
        if energies.size < 2:
            return None
        return max(1, int(resolve_window(energies, self.window_request).count))

    def _adopt_ci(self, outcome) -> None:
        """Publish a conventional-CI outcome as this stage's results.

        One place, so an optimized stage and one materialized from a checkpoint present
        exactly the same attributes to everything downstream — which is the whole point of
        :meth:`from_checkpoint` returning a :class:`CASSCF` rather than something new.
        """
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
        #: The window resolution (``None`` for a stated count). A window resolves to a count
        #: at run time and the stage then reports that count exactly as a stated one, so
        #: every consumer of :attr:`n_states` reads a number.
        self.window = outcome.window
        #: The resolution at the **starting** orbitals (``None`` for a stated count, and on
        #: a materialized or restarted stage): the window's counterpart of
        #: :attr:`boundary_initial`, and where the ladder's first rung says where it came
        #: from — the handoff from an upstream :class:`CheapCI`, or the network's pilot.
        self.window_initial = outcome.window_initial
        self.rounds = outcome.rounds
        self.checkpoint_path = outcome.checkpoint_path
        self._energies = np.asarray(outcome.ci.total_energies, dtype=float)
        # ⚠ A materialized stage was constructed before the file was read, so its declared
        # space and state count were placeholders. They are the file's now, and every
        # consumer that asks the stage rather than the outcome sees the same answer.
        self.space = outcome.active
        self.n_states = int(np.asarray(outcome.ci.energies).size)

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
            from ..io.checkpoint import (SYSTEM_KEY, check_system, read_checkpoint,
                                         system_fingerprint)
            resumed = read_checkpoint(self.restart)
            # ⚠ Before the active space, exactly as on the conventional-CI route: a matching
            # active space says nothing about whether these are the same molecule's orbitals.
            check_system(resumed.metadata.get(SYSTEM_KEY), system_fingerprint(ref),
                         self.restart, what="the checkpoint")
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
        solver = DMRGSolver(self.space.n_elec,
                            n_roots=(self.window_request if self.window_request is not None
                                     else self.n_states),
                            weights=self.weights, graph=self._resolve_graph(), **options)
        # Resolve the default topology now, so the solver's space_key names a real chart:
        # a restart compared against "dmrg:unset" would clear curvature that belongs to
        # exactly this surface.
        solver._ensure_chart(self.space.n_active)
        if resumed is not None:
            from .api import _check_restart_state_average, _window_from_checkpoint
            if self.window_request is not None:
                # ⚠ The count comes from the FILE and the window is what is compared: a
                # restart continues the calculation that was interrupted, and that
                # calculation ran at one count. Re-resolving here would start the round loop
                # from a ladder at the restored orbitals and could pick another one.
                solver = _window_from_checkpoint(resumed, solver, self.restart)
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
            from ..io.checkpoint import SYSTEM_KEY, system_fingerprint
            metadata = {"active_space": self.space.description}
            fingerprint = system_fingerprint(ref)
            if fingerprint is not None:
                metadata[SYSTEM_KEY] = fingerprint
            if ref.data.soc is not None:
                metadata["hamiltonian"] = json.dumps(ref.data.soc.provenance(),
                                                     sort_keys=True)
            policy = CheckpointPolicy(self.checkpoint, solver=solver, metadata=metadata,
                                      n_active_elec=self.space.n_elec, chain=self.callback,
                                      deadline=self.deadline, signals=self.signals,
                                      **(self.checkpoint_options or {}))
            hook = policy.callback
        elif self.deadline is not None or self.signals is not None:
            hook = self.callback
            if self.deadline is not None:
                hook = self.deadline.as_callback(chain=hook)
            if self.signals is not None:
                hook = self.signals.as_callback(chain=hook)
        driver = optimize_orbitals_events if self._adaptive else optimize_orbitals
        if self.report:
            out.section(log, "CASSCF (DMRG solver)")
            self.space.report(log)
            if resumed is not None:
                resumed.report(log)
            if self.deadline is not None:
                self.deadline.report(log)
            if self.signals is not None:
                self.signals.report(log)
        raise_if_pending("this DMRG-CASSCF")
        if self.deadline is not None:
            # ⚠ The granularity is a macro-iteration, and on this route one of those holds a
            # whole DMRG solve: neither stop cause can interrupt one, only refuse the next.
            self.deadline.assert_room("this DMRG-CASSCF")
        # The ladder's first rung from an upstream CheapCI's own spectrum, when there is
        # one: a rung, never a verdict (see :meth:`_window_first_rung`). Without it the
        # network's own pilot campaign supplies the estimate.
        estimate = (self._window_first_rung() if self.window_request is not None else None)

        def _round_hook(chain, index, current):
            """The optimizer's ``callback``, extended **additively** with the round and the
            count: a callback that ignores the two new keys is unaffected."""
            if self.window_request is None:
                return chain

            def hook(info):
                info["window_round"] = int(index)
                info["n_states"] = int(current.n_roots)
                return None if chain is None else chain(info)
            return hook

        # ⚠ The truncation weight is the tensor network's primary quality number, and without
        # this it appeared nowhere at INFO: the sweep table is at DEBUG (one table per sweep
        # times many macro-iterations is noise in a file that IS the output), so a production
        # DMRG output never said how much of the state was thrown away. It rides the
        # optimizer's additive extra_columns keyword, so the shared driver stays ignorant of
        # what a bond dimension is. The trend matters as much as the final value -- truncation
        # growing as the orbitals move is the signal that max_bond is too small.
        # ⚠ It reads the *current* solver, because a windowed run optimizes each round on a
        # sibling built at that round's count: a column bound to the solver this line was
        # written beside would report the first round's truncation for ever.
        live = [solver]
        w_disc = ((out.col_sci("w_disc"),
                   lambda: (float("nan") if live[0].last is None
                            else float(live[0].last.max_discarded))),)

        # ⚠ The event-gated driver has no ``start_iteration``: it is the sibling of the
        # smooth one, not a mode of it, and its signature says so. A window's rounds still
        # need ``max_iter`` to be a budget ACROSS rounds, so where the keyword is absent the
        # budget is handed over as what is left and the result's counter is shifted back —
        # the rounds loop reads ``n_iterations`` as a running total either way.
        counts_from_start = "start_iteration" in inspect.signature(driver).parameters

        def _optimize(coeff, current, *, round_index=1, start_iteration=None,
                      max_iter=None):
            # ⚠ ``start_iteration=None`` means "whatever the options already say" — a restart
            # brings its own, and a fixed-count run must not have it reset to zero here. Only
            # the round loop states one, because only it knows how much of the budget the
            # rounds before it spent.
            kwargs = dict(optimizer_options)
            spent = int(kwargs.get("start_iteration", 0) or 0) if start_iteration is None \
                else int(start_iteration)
            if max_iter is not None:
                kwargs["max_iter"] = (int(max_iter) if counts_from_start
                                      else max(1, int(max_iter) - spent))
            if counts_from_start and start_iteration is not None:
                kwargs["start_iteration"] = int(start_iteration)
            if round_index > 1:
                # ⚠ Curvature is chart-scoped: a new count is a new energy functional, so
                # the previous round's L-BFGS pairs are a memory of another surface.
                kwargs.pop("optimizer_state", None)
                kwargs.pop("history", None)
            live[0] = current
            if policy is not None:
                # The checkpoint reads its state average off the solver, and each round runs
                # a sibling; without this the file would record a solver that never solved.
                policy.rebind(current)
            kwargs["callback"] = _round_hook(hook, round_index, current)
            out_ = driver(ref.factors, h_ao, np.ascontiguousarray(coeff),
                          self.space.spaces, current, e_nuc=ref.data.e_nuc,
                          report=self.report, n_active_elec=self.space.n_elec,
                          extra_columns=w_disc, **kwargs)
            if not counts_from_start and spent:
                out_ = replace(out_, n_iterations=int(out_.n_iterations) + spent)
            return out_

        def _resolve(coeff, current, *, n_prev, where):
            ints_at = CASIntegrals.build(ref.factors, h_ao, np.ascontiguousarray(coeff),
                                         self.space.spaces, e_nuc=ref.data.e_nuc)
            return current.resolve_window(ints_at, estimate=estimate, n_prev=n_prev,
                                          where=where, report=self.report)

        # ⚠ The handlers live exactly as long as the optimization; the previous dispositions
        # come back afterwards, exception or not.
        rounds = None
        with stop_context(self.signals):
            if self.window_request is None:
                result = _optimize(orbitals, solver)
            else:
                from ..mcscf.rounds import optimize_rounds
                rounds = optimize_rounds(
                    self.window_request, solver, orbitals, resolve=_resolve,
                    optimize=_optimize,
                    max_iter=int(optimizer_options.get(
                        "max_iter",
                        inspect.signature(driver).parameters["max_iter"].default)),
                    start_iteration=int(optimizer_options.get("start_iteration", 0) or 0),
                    resolved=None if solver.n_roots is None else (None, solver),
                    report=self.report)
                solver, result = rounds.solver, rounds.orbital
                if rounds.n_rounds > 1:
                    # One trajectory, reported as one: the per-round counters are of the
                    # same run, exactly as on the conventional-CI route.
                    result = replace(
                        result, history=list(rounds.history),
                        n_hessian_matvec=sum(r.orbital.n_hessian_matvec
                                             for r in rounds.rounds),
                        n_second_order_steps=sum(r.orbital.n_second_order_steps
                                                 for r in rounds.rounds),
                        n_rejected=sum(r.orbital.n_rejected for r in rounds.rounds),
                        n_solver_failures=sum(r.orbital.n_solver_failures
                                              for r in rounds.rounds))

        # The optimizer's last solve may sit at a rejected trial step; the states this stage
        # reports must belong to the returned orbitals. One warm solve pins them there and
        # carries the state-average boundary diagnostic the in-loop solves skip.
        ints = CASIntegrals.build(ref.factors, h_ao, result.coeff, self.space.spaces,
                                  e_nuc=ref.data.e_nuc)
        # ⚠ With a window the ladder has already solved the roots the average does not use
        # and read the gap to the first of them, so the resolution REPLACES this diagnostic
        # rather than being added to it — a second boundary sweep would be a second
        # definition of the witness, and on this route it costs a whole extra sweep.
        solver.boundary_check = 0 if self.window_request is not None else BOUNDARY_MARGIN
        solver.solve(ints)

        self.active = self.space
        self.solver = solver
        self.orbital = result
        self.events = getattr(result, "events", [])
        #: The rounds a windowed run took (:class:`kuiva.mcscf.rounds.Round`); empty for a
        #: stated count, so every consumer asks one question of either route.
        self.rounds = [] if rounds is None else list(rounds.rounds)
        self.window_initial = None if rounds is None else rounds.initial
        if rounds is not None:
            self.window = rounds.final
            self.n_states = int(solver.n_roots)
        # ⚠ With a window this is the ladder's own witness gap, which is what the resolution
        # measured; without one it is the boundary sweep's local, one-sided gap. The
        # resolution's ``witness`` field is what says which, and it is printed with it.
        self.boundary_gap_cm = (self.window.boundary_gap_cm if rounds is not None
                                else solver.last.boundary_gap_cm)
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
        # ⚠ Each name asks for exactly one ordering and gets it. An AutoCAS prefers its site
        # partition in `dmrg_ordering()`, so "fiedler" goes to the entanglement order by name
        # -- otherwise a user asking for one ordering would silently receive the other.
        if request == "fiedler" and hasattr(self.upstream, "fiedler_ordering"):
            order = self.upstream.fiedler_ordering()
        else:
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
        if self.window is not None:
            gap = self.window.boundary_gap_cm
            entries.append(("state window", "{}; {} round(s), witness gap {}{}".format(
                self.window.window.describe(), len(self.rounds),
                "none (whole space)" if gap is None else "{:.2f} cm^-1".format(gap),
                "; AMBIGUOUS, the verdict was still moving" if self.window.ambiguous
                else "")))
        if self.solver_kind == "dmrg":
            entries.append(("largest bond dimension", str(self.solver.last.max_bond_dim)))
            # ⚠ Beside the bond dimension, never instead of it: the cap says what was
            # allowed and this says what it cost. An energy from this solver is not quotable
            # without it.
            entries.append(("largest discarded weight",
                            "{:.3e}".format(self.max_discarded)))
        # ⚠ Beside "converged: False", not instead of it: an unconverged result and the
        # reason it is unconverged are two different things a reader needs.
        if self.signals is not None and self.signals.fired:
            request = self.signals.requested
            entries.append(("stopped by", "a signal ({})".format(
                request.describe() if request is not None else "requested")))
        elif self.deadline is not None and self.deadline.fired:
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

    ``n_states`` is a count, a per-irrep mapping ``{irrep: n}`` wherever :class:`CASSCF`
    accepts one, or an **energy window** — ``n_states=kuiva.EnergyWindow(1000)`` selects
    every state within 1000 cm^-1 of the lowest, extended to the top of any manifold the
    cutoff falls inside (:class:`~kuiva.util.window.EnergyWindow`; ``{irrep: EnergyWindow}``
    with fixed counts beside it is its per-irrep form). With a window :attr:`n_states` is
    ``None`` until :meth:`run` and the resolved count after it, and :attr:`window` carries
    the :class:`~kuiva.util.window.WindowResolution` — the rungs it took, the witness gap
    and the edge. ``weights=`` is refused with a window. ``solver_options`` are
    :class:`~kuiva.mcscf.casci.FullCISolver`'s — ``kramers="restricted"``, ``conv_tol``,
    ``enforce_kramers``, ``degeneracy_tol``, ... — and ``classify=False`` switches off the
    full-double-group labelling of the converged blocks.

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
                 threshold: Optional[float] = None, avas=None, n_states=_UNSTATED,
                 weights=None,
                 coeff: Optional[np.ndarray] = None,
                 solver_options: Optional[Dict[str, Any]] = None,
                 classify: bool = True, report: bool = True) -> None:
        super().__init__()
        from ..mcscf.casci import FullCISolver
        self._finished(upstream, (Reference, CheapCI, AutoCAS, CASSCF), "CASCI")
        self.upstream = upstream
        self.reference_stage = (upstream if isinstance(upstream, Reference)
                                else upstream.reference_stage)
        #: Where ``n_states`` came from when the caller did not state it, for the output.
        n_states, self.n_states_source = _states_from_upstream(upstream, n_states,
                                                               what="CASCI")
        #: The CI method behind this stage, in the vocabulary :class:`CASSCF` uses for it, so
        #: that a consumer of either stage asks one question. A fixed-orbital CI is the
        #: conventional-CI route by construction.
        self.solver_kind = "ci"
        from ..util.window import EnergyWindow, is_window_request, shared_window
        #: The energy window the count is resolved from, or ``None`` for a stated count.
        self.window_request: Optional[EnergyWindow] = None
        #: The :class:`~kuiva.util.window.WindowResolution` after :meth:`run`; ``None``
        #: before it and for a stated count.
        self.window = None
        self._n_states_arg = n_states
        if is_window_request(n_states):
            if weights is not None:
                raise ValueError("weights= cannot be combined with an energy window: a "
                                 "window's weights are equal by construction and the "
                                 "state-averaging gate equalizes them inside degenerate "
                                 "blocks. Drop weights=, or state a count")
            self.window_request = (n_states if isinstance(n_states, EnergyWindow)
                                   else shared_window(n_states))
            self.state_request = dict(n_states) if isinstance(n_states, dict) else None
            self.n_states: Optional[int] = None
        else:
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
        elif isinstance(upstream, (CheapCI, AutoCAS)):
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
        raise_if_pending("this CASCI")
        if self.avas is not None and self.report:
            self.avas.report(log)
        if self.report and self.n_states_source:
            out.entry(log, "n_states", self.n_states if self.n_states is not None
                      else repr(self.window_request), "", self.n_states_source)
        self.result = _api_casci(
            self.reference_stage.reference, active=self.space, n_states=self._n_states_arg,
            weights=self.weights, coeff=self._orbitals, report=self.report,
            classify=self.classify, **self.solver_options)
        #: The same object under the two names the rest of the layer knows it by: `ci` is
        #: what a CASSCF calls its spectrum, `states` is what the property stages consume.
        self.ci = self.states = self.result
        self.active = self.space
        self.solver = self.result.solver
        self._energies = np.asarray(self.result.total_energies, dtype=float)
        # A window resolves to a count at run time; the stage then reports that count exactly
        # as a stated one, so every consumer of `n_states` reads a number.
        self.window = self.result.window
        if self.window_request is not None:
            self.n_states = int(self._energies.size)

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
        elif isinstance(self.upstream, AutoCAS):
            orbitals = "the automatically selected, pre-optimized orbitals"
        elif isinstance(self.upstream, CheapCI):
            orbitals = "the pre-optimized orbitals"
        elif self._orbitals is not None:
            orbitals = "given as coeff="
        else:
            orbitals = "the reference's guess spinors"
        rows = [
            ("active space", "CAS({}, {})  {}".format(self.active.n_elec,
                                                      self.active.n_active,
                                                      self.active.description)),
            ("orbitals", orbitals),
            ("<E> [Eh]", out.E_FMT.format(self.energy)),
            ("E(state 0) [Eh]", out.E_FMT.format(float(self._energies[0]))),
            ("states", str(self._energies.size)),
        ]
        if self.window is not None:
            gap = self.window.boundary_gap_cm
            rows.append(("state window", "{}; {} rung(s), witness gap {}".format(
                self.window.window.describe(), len(self.window.rungs),
                "none (whole space)" if gap is None else "{:.2f} cm^-1".format(gap))))
        rows += [
            ("determinants", str(self.solver.ndet)),
            ("applications of H", str(self.result.n_apply)),
        ]
        return rows


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

    Checkpointing and restart
    -------------------------
    ``checkpoint=path`` writes the per-``(state, class)`` table as it fills and ``restart=``
    resumes from one, skipping the classes the file already holds
    (:mod:`kuiva.pt.checkpoint`). ⚠ They are separate arguments here as on :class:`CASSCF`,
    and a resumed run that is to keep protecting itself passes **both**, usually the same
    path: ``restart=`` alone reads the file and does not write it. ⚠ **The file is kilobytes
    whatever the active space** — a class result is eight scalars — so there is no size
    question here and no byte budget; the only cadence knob is
    ``checkpoint_options=dict(min_interval=...)``, which rations filesystem calls rather
    than bytes.

    ⚠ **It stores no reference.** The orbitals and the CI vectors belong to the CASSCF
    checkpoint, and what is kept here is a *digest* of them. A restart against a different
    reference — different orbitals, different active space, different state energies or
    different CI vectors — is **refused**, because inside a degenerate manifold the CI's
    basis is arbitrary and resuming across a re-solved reference would compute some members
    of a manifold in one basis and some in another. The practical consequence:
    :meth:`CASSCF.from_checkpoint` reproduces the vectors exactly when they survived the
    CASSCF file's thinning, and only then do the two restarts compose.

    ``deadline=`` and ``signals=`` stop the correction **between classes**, which is the only
    boundary this stage has, with the finished ones written first. ⚠ The stop raises
    :class:`~kuiva.util.signals.StopRequested` rather than returning a partial ``E2``: a
    correction that stopped is not a smaller correction, and the file is what finishes it.

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
        #: Set by :meth:`from_checkpoint`; ``None`` on an ordinary stage. Popped before the
        #: option check, because it is this class's own marker and not one of the driver's.
        self._materialize = options.pop("_materialize", None)
        _check_options(options, _allowed_options(sc_nevpt2, sc_nevpt2_dmrg,
                                                 exclude=self._EXCLUDE),
                       "NEVPT2")
        if "classes" in options and options["classes"] is not None:
            from ..pt.classes import excitation_class
            for name in options["classes"]:
                excitation_class(name)                       # refuse an unknown name now
        # ⚠ Resolved at construction, as on every other stage: deadline="slurm" outside a
        # Slurm job, or signals= off the main thread, is a mistake that must surface now and
        # not after the first class.
        from ..util.deadline import Deadline
        from ..util.signals import SignalStop
        if "deadline" in options:
            options["deadline"] = Deadline.resolve(options["deadline"])
        if "signals" in options:
            options["signals"] = SignalStop.resolve(options["signals"])
        if options.get("restart") is not None and not Path(options["restart"]).exists():
            raise ValueError("restart checkpoint {!r} does not exist"
                             .format(str(options["restart"])))
        self.options = dict(options)

    @classmethod
    def from_checkpoint(cls, path, source, *, report: bool = True) -> "NEVPT2":
        """A **finished** NEVPT2 stage read back from a complete checkpoint — no computation.

        The sibling of :meth:`CASSCF.from_checkpoint`, and cheaper: everything the assembly
        reads is already in the table, so this is a file read rather than a re-solve. It is
        what lets a correction that finished hours ago reach a :class:`PropertyDump` that was
        never asked for::

            cas = kuiva.CASSCF.from_checkpoint("run.h5", ref)
            pt2 = kuiva.NEVPT2.from_checkpoint("e2.h5", cas)
            kuiva.PropertyDump(pt2, "props.out").run()

        ``source`` is the stage the correction belongs to, and it is what the corrected
        energies are attached to downstream. ⚠ **The file's reference digest is not checked
        against it** — the digest identifies the run that wrote the table, and a stage
        materialized from a *thinned* CASSCF checkpoint legitimately differs from it inside a
        degenerate manifold. What that means in practice is that the ``H`` of the resulting
        dump comes from the file and its ``mu`` from ``source``; pairing a table with the
        wrong source is the one mistake this cannot see, so pass the stage the correction was
        computed on.

        ⚠ An **incomplete** table is refused with the state it stopped at: pass it to
        ``restart=`` and finish it instead.
        """
        stage = cls(source, _materialize={"path": path, "report": bool(report)})
        return stage

    def _execute(self) -> None:
        if self._materialize is not None:
            from ..pt.nevpt2 import assemble_from_checkpoint
            self.result = assemble_from_checkpoint(self._materialize["path"],
                                                   report=self._materialize["report"])
            self._publish()
            return
        # ⚠ Hours of work: starting one after a stop has been requested spends the process's
        # last seconds on something nobody will see. The stage now has a checkpoint of its
        # own, so a stop that arrives *during* it is handled between classes instead.
        raise_if_pending("this NEVPT2")
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
        self._publish()

    def _publish(self) -> None:
        """The attributes a finished stage carries, in one place so a computed correction and
        one assembled from a checkpoint present the same surface."""
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

    ⚠ **The hyperfine field operators are written whenever the reference ingested them**
    (``hyperfine=`` on :class:`ScalarSCF`), with a ``[NUCLEI]`` table, and there is no second
    switch here: naming the nuclei is the request. The same boundary holds — what the file
    carries is the isotope-independent operator ``T_{k,u}``, and the A tensor, the spin
    Hamiltonian and the nuclear-spin algebra belong to the external code.

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
        entries = [
            ("file", str(self.path)),
            ("states", str(self.matrices.n_states)),
            ("energies", "NEVPT2-corrected (hybrid protocol, recorded)"
             if isinstance(self.source, NEVPT2)
             else type(self.states_stage).__name__),
        ]
        if self.matrices.has_hyperfine:
            entries.append(("hyperfine nuclei",
                            ", ".join(self.matrices.hyperfine_labels)))
        return entries


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

        raise_if_pending("this pseudospin export")
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
