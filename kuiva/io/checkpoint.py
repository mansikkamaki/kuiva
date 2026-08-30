"""Schema-versioned HDF5 checkpoints for the CASSCF driver, and restart from them.

What is written, and what is deliberately not
---------------------------------------------
The checkpoint policy divides the state of a run into three:

* **always, every macro-iteration** (cheap, precious) — the orbitals, the orbital partition,
  the 1-RDM, the converged state energies, the run metadata and the Hamiltonian provenance.
  The orbitals are the calculation; everything here is small beside them or is what says
  which calculation they belong to;
* **by policy** (large) — the 2-RDM, the CI vectors and the orbital curvature, dropped in
  that order when the adaptive budget bites (:data:`THINNING_LADDER`). ⚠ **The order is by
  what it costs to get the thing back, not by what it costs to store**, and the three answers
  are one RDM contraction, one CI solve, and a few extra macro-iterations respectively;
* **never** — the three-index integral factors, four-index integrals, 3- and 4-RDMs, and
  ⚠ **transition densities**. All of them regenerate deterministically from the orbitals, and
  a restart does regenerate them. The transition densities are the one that has to be said
  out loud, because storing them looks attractive — they are what a property dump needs and
  they are only ``n_states**2 * n_act**2`` — but they are *derived* from the CI vectors that
  are already here, they go stale against a source change that the vectors do not, and they
  serve no other consumer (:mod:`kuiva.pt` needs the vectors themselves). Two answers to
  "what is this wavefunction" is one too many.
  ⚠ The expensive part of that regeneration is the four-component atomic solve, ~35
  minutes for a lanthanide, and it is **not** duplicated here: it depends on the element and
  the basis, not on the geometry or the iterate, and its own persistent cache is what makes a
  restart cheap. Copying it into every checkpoint would be storing the same megabyte per
  iteration for no benefit.

⚠ Failure semantics, and they are the opposite of a cache's
------------------------------------------------------------
The rule that "a cache degrades to a miss on every failure" governs the X2CAMF correction
cache. It does **not** govern this:

* a **write** failure — full disk, unwritable directory, a broken HDF5 — is a ``WARNING`` and
  the run continues. Losing a restart point is not losing the calculation, and a calculation
  that aborts at the end of a good macro-iteration because a disk was full has thrown away
  everything for nothing.
* a **read** failure on an explicitly requested restart is an ``ERROR`` and it propagates. The
  user asked to resume; silently starting over wastes the hours the file existed to protect,
  and does it invisibly.

The schema version **hard-refuses** on mismatch, for the same reason: a file written by a
different layout is not a file to guess at. The code fingerprint only warns — it says the
sources moved, which is usually a comment change and occasionally a reason not to trust the
numbers, and only the person reading the log can tell which.

The adaptive budget
-------------------------
"Estimate write size/time from array dimensions and measured disk bandwidth; skip or thin a
checkpoint when writing it would cost more than a set fraction (~5%) of the compute elapsed
since the last one." That is :class:`CheckpointPolicy`, and all four inputs — the byte count,
the measured bandwidth (:func:`kuiva.util.resources.disk_write_bandwidth_gb_s`), the elapsed
compute and the fraction — are real rather than assumed. **Thinning comes before skipping**:
dropping the CI vectors usually brings a checkpoint an order of magnitude under the budget,
and a thinned checkpoint still restarts the calculation where a skipped one does not.

⚠ A checkpoint at a **converged** iteration is written unconditionally. The cadence policy is
about how often to insure against a crash; the last one is the result.

⚠ **The converged write inverts the ladder: it stores no curvature at all.** A converged run
is never resumed, so its L-BFGS pairs and augmented-Hessian guess cannot be used by anything
— and on a large-basis run they are the *majority* of the file, because they scale with the
rotation-parameter count while everything else scales with the active space. What that file
does need is the CI vectors, which are what lets :func:`kuiva.interface.api.casscf_from_checkpoint`
materialize the finished calculation without re-solving it from cold. ⚠ The inversion keys on
**convergence, not on ``force``**: a deadline or a signal also forces a write, and that run
*will* be resumed, so its curvature is precious.

⚠ What a restart is checked against, and the two ways it silently was not
-------------------------------------------------------------------------
A restart continues **the calculation that was interrupted**. Two things define that
calculation beyond the orbitals, and both were being taken on trust:

* **the state average.** :func:`state_average_key` records ``n_states`` and the *requested*
  weights, read from the solver by :meth:`CheckpointPolicy._metadata` rather than accepted
  from a caller, so no driver can leave a restart uncheckable by forgetting to describe
  itself. A mismatch is **refused**, on the same grounds a restated active space that
  disagrees is: a different state average changes the energy functional, so it is a different
  calculation and not a different chart of this one. Continuing from converged orbitals into a
  *new* state average is what ``coeff=`` is for. A file predating the record cannot be
  compared and says so — which is a weaker statement than "matches", and is worded as one.
* **the chart, for curvature.** :meth:`CASSCFCheckpoint.optimizer_kwargs` takes the **live**
  solver's ``space_key`` as a *required* argument. It used to take none and hand back
  ``self.space_key``, which compared the file with itself: ``same_chart`` was then
  unconditionally true, and L-BFGS pairs, the augmented-Hessian warm start and the trust
  radius were restored across any chart change without a word. Nothing observable failed —
  the run converged, to a number indistinguishable from a good one.

Both are the same lesson twice: a check whose two sides come from the same place is not a
check, and the mechanism, not the observable, is what the tests hold.

References
----------
* HDF5: The HDF Group, "Hierarchical Data Format, version 5" (1997-2024), https://www.hdfgroup.org/HDF5/.
* Checkpoint/restart cost models — the "how often" question this policy answers empirically
  rather than analytically: J. T. Daly, Future Gener. Comput. Syst. 22, 303 (2006),
  doi:10.1016/j.future.2004.11.016; J. W. Young, Commun. ACM 17, 530 (1974),
  doi:10.1145/361147.361115.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..mcscf.orbopt import OrbitalSpaces
from ..util import output as out
from ..util import resources as res
from ..util.logging import get_logger

log = get_logger(__name__)

#: Layout version of the file. Bumped whenever the **structure** changes in a way a reader of
#: the previous version would get wrong; a mismatch is refused, never guessed at.
#:
#: ``2`` (0.34.0) made two payload groups **optional** — the 2-RDM and the orbital curvature
#: can now legitimately be absent (:data:`THINNING_LADDER`). A version-1 reader has no way to
#: tell an absent ``gamma2`` from an empty one, which is a wrong 2-RDM rather than a missing
#: file, so version-1 files are refused rather than read.
SCHEMA_VERSION = 2

#: Optimizer-state entries that are **pure warm start**: L-BFGS curvature, the previous
#: gradient and step, and the augmented-Hessian guess. Dropping them costs a restart the
#: convergence rate it had accumulated and nothing else — exactly what chart-scoping already
#: discards without ceremony whenever the CI surface changes
#: (:meth:`kuiva.mcscf.orbopt.OrbitalOptimizer.load_state_dict`). ⚠ The trust radius, the
#: stall count and the gradient history are **not** here: they are scalars, they cost nothing
#: to keep, and the trust radius is worth real iterations.
CURVATURE_KEYS = ("lbfgs_s", "lbfgs_y", "prev_grad", "prev_step", "ah_guess")

#: What a checkpoint may drop, cheapest-to-regenerate first, when the adaptive budget bites.
#: ⚠ **The order is by regeneration cost, not by size**, which is the whole of the rule:
#:
#: * ``gamma2`` — one RDM contraction away from the CI vectors, and strictly implied by them.
#:   It is also what grows as ``n_act**4``, so it is both the cheapest to lose and the term
#:   that makes a large active space expensive to store.
#: * ``ci_vectors`` — one CI solve at the stored orbitals. A Davidson warm start, worth
#:   roughly an order of magnitude in applications of ``H`` on the next solve and nothing
#:   after that.
#: * ``curvature`` — not regenerable at all, only *discardable*: a cold start at the same
#:   orbitals converges, and pays a few extra macro-iterations for it. Each one is a CI solve
#:   plus an integral transform, so this is the most expensive rung and it is last.
#:
#: ⚠ **Dropping ``gamma2`` is offered only while the CI vectors are still there.** Without
#: them it is not an implied quantity but the only record of the converged density.
THINNING_LADDER = ("gamma2", "ci_vectors", "curvature")

#: Metadata key under which :func:`system_fingerprint` is stored.
SYSTEM_KEY = "system"

#: Modules whose source is fingerprinted into the file. A restart is a claim that this code
#: produces the same trajectory from the same state, and these are the modules that carry it.
FINGERPRINTED_MODULES = ("kuiva.mcscf.orbopt", "kuiva.mcscf.casci", "kuiva.ci.sigma",
                         "kuiva.ci.davidson", "kuiva.rdm.rdm")


def _kuiva_version() -> str:
    """The running code's version, recorded in every checkpoint."""
    from kuiva import __version__
    return str(__version__)


class CheckpointError(RuntimeError):
    """A checkpoint could not be read. ⚠ Raised only on the **read** path (see the module
    docstring): a write failure warns and the run continues."""


def _h5py():
    """``h5py``, or ``None``. Imported lazily so checkpointing is optional at import time."""
    try:
        import h5py
    except ImportError:                                        # pragma: no cover - optional
        return None
    return h5py


def code_fingerprint(modules: Sequence[str] = FINGERPRINTED_MODULES) -> str:
    """A 128-bit digest of the sources a restart's reproducibility depends on.

    Recorded as metadata and **warned** about on mismatch, never refused: most source changes
    (a docstring, a log line) cannot move a number, and only the reader knows whether this one
    could. ⚠ It cannot see NumPy, MKL, the basis-set data or the machine — the same limit
    the test-suite stage checkpoints state, and for the same reason: a checkpoint is a
    replay, never a claim of portability.
    """
    import importlib

    digest = hashlib.blake2b(digest_size=16)
    for name in sorted(modules):
        try:
            path = importlib.import_module(name).__file__
            with open(path, "rb") as handle:
                digest.update(handle.read())
        except (ImportError, OSError, TypeError):              # pragma: no cover - defensive
            digest.update(name.encode())
    return digest.hexdigest()


def system_fingerprint(reference) -> Optional[str]:
    """A digest of **which calculation** a checkpoint belongs to, or ``None``.

    The state average says whether a restart is the same *calculation*; this says whether it
    is even the same *system*. Until it existed the file carried nothing that could tell one
    molecule from another: pointing a restart at the wrong checkpoint of the right shape gave
    a converging run and a plausible number, because orbitals are just an orthonormal set of
    the right dimension and the optimizer re-optimizes them.

    What goes in is everything that defines the integrals and cannot be inferred from the
    arrays: atom labels **as given** (a ghost label is not its element and must not compare
    equal to one), coordinates and their unit, charge and spin, the per-atom basis
    assignment, the reference configurations, the nuclear model, the embedding, the
    ``ScreeningRecord``/``DecouplingRecord`` provenance, the gauge origin — ``L`` and ``r``
    are defined about it and a moved origin is a different property file — and the AO/orbital
    dimensions.

    ⚠ It is a **digest, not a description**: it says two runs differ, never how. Rebuilding
    the comparison from the parts would mean carrying the parts, and the parts are exactly
    what the ingestion boundary keeps out of the multireference layer.

    Returns ``None`` for a reference that carries no :class:`MoleculeSpec` — a container built
    by hand in a test — which is read as "cannot be compared", never as "matches".
    """
    data = getattr(reference, "data", reference)
    spec = getattr(data, "molecule", None)
    if spec is None:
        return None
    parts: Dict[str, Any] = {
        "atoms": [[str(sym), [float(x) for x in xyz]] for sym, xyz in spec.atoms],
        "unit": str(spec.unit), "charge": int(spec.charge), "spin": int(spec.spin),
        "basis": repr(spec.basis), "configuration": repr(spec.configuration),
        "nuclear_model": str(spec.nuclear_model), "environment": repr(spec.environment),
        "nao": int(getattr(data, "nao", 0)), "nmo": int(getattr(data, "nmo", 0)),
        "reference": str(getattr(data, "reference", "")),
    }
    soc = getattr(data, "soc", None)
    if soc is not None:
        parts["hamiltonian"] = soc.provenance()
    properties = getattr(data, "properties", None)
    if properties is not None:
        parts["gauge_origin"] = [float(x) for x in
                                 np.asarray(properties.gauge_origin).ravel()]
    payload = json.dumps(parts, sort_keys=True, default=repr).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def check_system(stored: Optional[str], current: Optional[str], path, *, what: str) -> None:
    """Refuse a restart whose system digest differs from the file's.

    ⚠ **Refused, not warned about**: unlike a source change, which usually cannot move a
    number, this one always does — it is a different molecule, basis, Hamiltonian or gauge
    origin. An unrecorded digest on either side means "cannot be compared" and warns, which is
    the weaker statement and is worded as one.
    """
    if stored is None or current is None:
        log.warning("%s at %s carries no system fingerprint on one side, so it cannot be "
                    "checked against this reference; confirm by hand that the molecule, "
                    "basis, Hamiltonian and gauge origin are the ones it was written for",
                    what, path)
        return
    if stored != current:
        raise ValueError(
            "{} at {} was written for a different system (fingerprint {} against {}). The "
            "molecule, the basis, the per-atom configuration, the Hamiltonian axes, the "
            "nuclear model, the embedding or the gauge origin differs, so its orbitals do "
            "not belong to this reference -- and an orthonormal set of the right dimension "
            "optimizes to a plausible number whatever it came from"
            .format(what, path, stored[:12], current[:12]))


def _has_tier(checkpoint: "CASSCFCheckpoint", tier: str) -> bool:
    """Is there anything of this :data:`THINNING_LADDER` rung left to give up?"""
    if tier == "curvature":
        return any(checkpoint.optimizer_state.get(key) is not None for key in CURVATURE_KEYS)
    return getattr(checkpoint, tier) is not None


def _kernel_backend_token() -> str:
    """The active kernel backend as provenance (``numpy`` or ``native:<build id>``)."""
    from ..util import native

    return native.fingerprint_token()


# --- The record ----------------------------------------------------------------------------

@dataclass
class CASSCFCheckpoint:
    """One restart point of a state-averaged CASSCF.

    Plain arrays and scalars throughout, deliberately: this is what crosses a process
    boundary, and a container holding live objects (an optimizer, a solver, a
    :class:`~kuiva.integrals.transform.ThreeIndexAO`) could not.
    """

    iteration: int
    energy: float
    grad_norm: float
    converged: bool
    coeff: np.ndarray                                   # (2*nao, n_orb) spinors, AO basis
    inactive: np.ndarray
    active: np.ndarray
    virtual: np.ndarray
    n_orb: int
    n_active_elec: int
    gamma: np.ndarray
    state_energies: np.ndarray
    #: ``(n_act,)*4`` — the state-averaged 2-RDM, or ``None`` if it was thinned away. ⚠ It is
    #: one contraction away from :attr:`ci_vectors` and it is the term that grows as
    #: ``n_act**4``, which is why it is the first rung of :data:`THINNING_LADDER`.
    gamma2: Optional[np.ndarray] = None
    optimizer_state: Dict[str, Any] = field(default_factory=dict)
    #: ``(n_states, ndet)`` — the Davidson warm start, or ``None`` if it was thinned away.
    ci_vectors: Optional[np.ndarray] = None
    space_key: Optional[str] = None
    history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    metadata: Dict[str, str] = field(default_factory=dict)
    #: Which rungs of :data:`THINNING_LADDER` are absent from this object. ⚠ Recorded rather
    #: than inferred: an absent array and an empty one are the same bytes on disk, and a
    #: reader that guesses between them reads a dropped 2-RDM as a zero one.
    dropped: Tuple[str, ...] = ()

    @property
    def spaces(self) -> OrbitalSpaces:
        """The orbital partition, rebuilt. Its own validation re-runs on the way out, so a
        corrupted index array is caught here rather than three macro-iterations later."""
        return OrbitalSpaces(inactive=self.inactive, active=self.active,
                             virtual=self.virtual, n_orb=int(self.n_orb))

    @property
    def n_states(self) -> int:
        return int(self.state_energies.size)

    def payload(self) -> List[np.ndarray]:
        """Every array that goes to disk, in one place, so the sizing function cannot drift
        from what is actually written."""
        arrays = [self.coeff, self.gamma, self.state_energies,
                  np.asarray(self.history), self.inactive, self.active, self.virtual]
        arrays += [np.asarray(v) for v in self.optimizer_state.values()
                   if isinstance(v, np.ndarray)]
        for optional in (self.gamma2, self.ci_vectors):
            if optional is not None:
                arrays.append(optional)
        return arrays

    def size_gb(self) -> float:
        """Exact payload size [GB], excluding HDF5's own few kB of structure."""
        return sum(float(a.nbytes) for a in self.payload()) / res.BYTES_PER_GB

    def reduced(self, *tiers: str) -> "CASSCFCheckpoint":
        """The same checkpoint with the named :data:`THINNING_LADDER` tiers removed.

        Still a complete restart in every combination: what each rung costs is stated on the
        ladder itself, and none of it is correctness. ⚠ Dropping ``"gamma2"`` from a
        checkpoint whose CI vectors are **already** gone is refused rather than silently
        allowed — the point of the ordering is that ``gamma2`` is cheap to lose *because* the
        vectors imply it, and without them it is the only record of the converged density.
        Asking for both in one call is a different thing and is allowed: it is the explicit
        "reduce this to the orbitals" the policy makes as its last resort before skipping.
        """
        import dataclasses

        unknown = [t for t in tiers if t not in THINNING_LADDER]
        if unknown:
            raise ValueError("not a thinning tier: {}; the ladder is {}"
                             .format(unknown, list(THINNING_LADDER)))
        changes = {}
        if "gamma2" in tiers:
            if self.ci_vectors is None and "ci_vectors" not in tiers:
                raise ValueError(
                    "the 2-RDM may only be dropped while the CI vectors are stored; without "
                    "them it is not an implied quantity but the only record of the density")
            changes["gamma2"] = None
        if "ci_vectors" in tiers:
            changes["ci_vectors"] = None
        if "curvature" in tiers:
            # ⚠ **Removed, not set to ``None``.** ``load_state_dict`` restores the L-BFGS
            # pairs as ``state.get("lbfgs_s", <empty>)``, so a key that is *present* and
            # ``None`` reaches ``np.asarray(None, dtype=complex128)`` and raises — a restart
            # that dies on the file it was given. An absent key takes the default, which is
            # exactly "no curvature"; ``dropped`` is what records that it was deliberate.
            changes["optimizer_state"] = {key: value
                                          for key, value in self.optimizer_state.items()
                                          if key not in CURVATURE_KEYS}
        present = set(self.dropped) | set(tiers)
        changes["dropped"] = tuple(t for t in THINNING_LADDER if t in present)
        return dataclasses.replace(self, **changes)

    def thinned(self) -> "CASSCFCheckpoint":
        """The same checkpoint without the CI vectors. ⚠ One rung of :meth:`reduced`, kept
        under its old name because it is the rung callers outside the policy ask for."""
        return self.reduced("ci_vectors")

    def optimizer_kwargs(self, *, space_key: Optional[str]) -> Dict[str, Any]:
        """The keyword arguments that resume :func:`kuiva.mcscf.orbopt.optimize_orbitals`.

        Kept here rather than at the call site so that a restart cannot silently drop one of
        them — forgetting ``start_iteration`` would restart the *budget*.

        ⚠ **``space_key`` is the LIVE solver's key and is a required argument, never
        ``self.space_key``.** Chart-scoping works by comparing the key recorded *inside*
        ``optimizer_state`` against the key of the solver that is about to run: handing this
        object's own key back would compare the file with itself, make ``same_chart``
        unconditionally true, and restore curvature — L-BFGS pairs, the augmented-Hessian
        warm start, the trust radius — belonging to a surface that no longer exists. That is
        precisely the bug the mechanism exists to prevent, and it is invisible: the run
        converges, to a number nobody can tell apart from a good one. It was a no-argument
        method that did exactly this; the argument is required so no caller can restore the
        mistake by omission.
        """
        return {"optimizer_state": self.optimizer_state,
                "start_iteration": int(self.iteration),
                "space_key": space_key,
                "history": np.asarray(self.history).tolist()}

    def report(self, logger=None) -> None:
        out.entries(logger or log, [
            ("checkpoint iteration", self.iteration),
            ("energy", self.energy, "Eh", "", out.E_FMT),
            ("gradient norm", self.grad_norm, "", "", out.SCI_FMT),
            ("converged", self.converged),
            ("states", self.n_states),
            ("CI vectors stored", "yes" if self.ci_vectors is not None else
             "no (thinned; the Davidson warm start is lost, not the restart)"),
            ("thinned away", ", ".join(self.dropped) if self.dropped else "nothing"),
            ("payload", self.size_gb(), "GB", "", "{:.4f}"),
        ])

    def __repr__(self) -> str:
        return ("CASSCFCheckpoint(iteration={}, E={:.10f} Eh, |g|={:.2e}, converged={}, "
                "{:.3f} GB)".format(self.iteration, self.energy, self.grad_norm,
                                    self.converged, self.size_gb()))


def checkpoint_size_gb(checkpoint: CASSCFCheckpoint) -> float:
    """Size [GB] a checkpoint will occupy (exact sizing function).

    Exact and unpadded: the sum of the ``nbytes`` of every array written. It excludes HDF5's
    own structural metadata, which is a few kB and does not scale with anything — stated
    rather than absorbed into a fudge factor, per the rule that sizing functions never pad.
    """
    return checkpoint.size_gb()


# --- Writing and reading -------------------------------------------------------------------

def _write_optional(group, name: str, value) -> None:
    """Store an optional array. HDF5 has no null dataset, so absence is an empty one — and
    :func:`kuiva.mcscf.orbopt._as_optional_vector` turns it back into ``None`` on the way in,
    which is what keeps a restored warm start from being a zero-length guess."""
    if value is None:
        group.create_dataset(name, data=np.zeros(0, dtype=np.complex128))
    else:
        group.create_dataset(name, data=np.ascontiguousarray(value))


def write_checkpoint(path, checkpoint: CASSCFCheckpoint, *,
                     compression: Optional[str] = None) -> float:
    """Write ``checkpoint`` to ``path``, atomically. Returns the file size [GB].

    Written to a temporary file in the same directory and renamed, so a run killed mid-write
    leaves the **previous** checkpoint intact rather than a truncated file that reads as a
    restart point and is not one. ``os.replace`` is atomic within a filesystem.

    ⚠ Raises on failure. Callers that are protecting a running calculation go through
    :class:`CheckpointPolicy`, which turns that into a ``WARNING`` (module docstring).
    """
    h5py = _h5py()
    if h5py is None:
        raise CheckpointError("h5py is not installed, so checkpoints cannot be written")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Free space on the checkpoint's **own** filesystem, re-read now: a limit set at start-up
    # is not evidence that the space is still there. Deliberately not `require_scratch`, which
    # asks about the configured scratch directory and this file need not live there.
    free = res.scratch_free_gb(path.parent)
    if free is not None and checkpoint.size_gb() > free:
        raise CheckpointError(
            "the checkpoint needs {:.3f} GB and {} has {:.3f} GB free".format(
                checkpoint.size_gb(), path.parent, free))
    tmp = path.with_name(path.name + ".writing-{}".format(os.getpid()))
    kwargs = {} if compression is None else {"compression": compression}
    try:
        with h5py.File(str(tmp), "w") as handle:
            handle.attrs["schema_version"] = SCHEMA_VERSION
            # ⚠ The code version is not the schema version and neither implies the other:
            # `schema_version` decides whether the file can be read at all, `kuiva_version`
            # says which release wrote it, and `code_fingerprint` is what actually detects a
            # numerically relevant source change. Recorded, never checked.
            handle.attrs["kuiva_version"] = _kuiva_version()
            handle.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            handle.attrs["code_fingerprint"] = code_fingerprint()
            # ⚠ The kernel backend beside the source fingerprint: a source hash cannot see
            # a compiled .so, and a threaded reduction is only 1e-13-equal, so a restart
            # across backends is a warned-about event exactly like a source change.
            handle.attrs["kernel_backend"] = _kernel_backend_token()
            handle.attrs["iteration"] = int(checkpoint.iteration)
            handle.attrs["energy"] = float(checkpoint.energy)
            handle.attrs["grad_norm"] = float(checkpoint.grad_norm)
            handle.attrs["converged"] = bool(checkpoint.converged)
            handle.attrs["n_orb"] = int(checkpoint.n_orb)
            handle.attrs["n_active_elec"] = int(checkpoint.n_active_elec)
            handle.attrs["space_key"] = "" if checkpoint.space_key is None \
                else str(checkpoint.space_key)
            # ⚠ Recorded, not inferred: on disk an absent array and an empty one are the
            # same bytes, and a reader that guesses reads a dropped 2-RDM as a zero one.
            handle.attrs["dropped"] = json.dumps(list(checkpoint.dropped))
            handle.attrs["metadata"] = json.dumps(checkpoint.metadata, sort_keys=True)

            orbitals = handle.create_group("orbitals")
            orbitals.create_dataset("coeff", data=checkpoint.coeff, **kwargs)
            orbitals.create_dataset("inactive", data=np.asarray(checkpoint.inactive,
                                                                dtype=np.int64))
            orbitals.create_dataset("active", data=np.asarray(checkpoint.active,
                                                              dtype=np.int64))
            orbitals.create_dataset("virtual", data=np.asarray(checkpoint.virtual,
                                                               dtype=np.int64))

            rdm = handle.create_group("rdm")
            rdm.create_dataset("gamma", data=checkpoint.gamma, **kwargs)
            _write_optional(rdm, "gamma2", checkpoint.gamma2)

            states = handle.create_group("states")
            states.create_dataset("energies", data=np.asarray(checkpoint.state_energies,
                                                              dtype=np.float64))
            states.create_dataset("history", data=np.asarray(checkpoint.history,
                                                             dtype=np.float64))
            _write_optional(states, "ci_vectors", checkpoint.ci_vectors)

            optimizer = handle.create_group("optimizer")
            for key, value in sorted(checkpoint.optimizer_state.items()):
                if value is None:
                    _write_optional(optimizer, key, None)
                    optimizer[key].attrs["is_none"] = True
                elif isinstance(value, np.ndarray):
                    optimizer.create_dataset(key, data=np.ascontiguousarray(value))
                elif isinstance(value, str):
                    optimizer.attrs[key] = value
                elif isinstance(value, (bool, np.bool_)):
                    optimizer.attrs[key] = bool(value)
                elif isinstance(value, (int, np.integer)):
                    optimizer.attrs[key] = int(value)
                else:
                    optimizer.attrs[key] = float(value)
        os.replace(str(tmp), str(path))
    except Exception:
        try:
            os.unlink(str(tmp))
        except OSError:
            pass
        raise
    size = Path(path).stat().st_size / res.BYTES_PER_GB
    log.debug("checkpoint written: %s, iteration %d, %.4f GB", path, checkpoint.iteration,
              size)
    return size


def read_checkpoint(path, *, check_fingerprint: bool = True) -> CASSCFCheckpoint:
    """Read a checkpoint. ⚠ **Raises** :class:`CheckpointError` on any failure.

    A restart is explicitly requested, so a missing, corrupt or wrong-schema file is an
    ``ERROR``: quietly starting from scratch would waste exactly the compute the file exists
    to protect, and would do it without saying so.
    """
    h5py = _h5py()
    if h5py is None:
        raise CheckpointError("h5py is not installed, so the checkpoint at {} cannot be read"
                              .format(path))
    path = Path(path)
    if not path.is_file():
        message = "no checkpoint at {}".format(path)
        log.error("%s", message)
        raise CheckpointError(message)
    try:
        with h5py.File(str(path), "r") as handle:
            version = int(handle.attrs.get("schema_version", -1))
            if version != SCHEMA_VERSION:
                message = (
                    "checkpoint {} carries schema version {} and this Kuiva reads version {}. "
                    "The layout changed, so the file cannot be interpreted; rerun from the "
                    "start, or read it with the Kuiva that wrote it".format(
                        path, version, SCHEMA_VERSION))
                if version == 1:
                    message += (
                        ". Version 1 stored the 2-RDM and the orbital curvature unconditionally "
                        "and version 2 may drop either; there is no way to tell an absent array "
                        "from an empty one in a version-1 file, so it is refused rather than "
                        "read as a zero density")
                log.error("%s", message)
                raise CheckpointError(message)
            metadata = json.loads(handle.attrs.get("metadata", "{}"))
            stored_fingerprint = str(handle.attrs.get("code_fingerprint", ""))
            stored_backend = str(handle.attrs.get("kernel_backend", ""))
            stored_version = str(handle.attrs.get("kuiva_version", "unrecorded"))

            optimizer_state: Dict[str, Any] = dict(handle["optimizer"].attrs)
            for key, dataset in handle["optimizer"].items():
                if dataset.attrs.get("is_none", False):
                    optimizer_state[key] = None
                else:
                    optimizer_state[key] = np.ascontiguousarray(dataset[()])
            ci = np.ascontiguousarray(handle["states/ci_vectors"][()])
            gamma2 = np.ascontiguousarray(handle["rdm/gamma2"][()])
            checkpoint = CASSCFCheckpoint(
                iteration=int(handle.attrs["iteration"]),
                energy=float(handle.attrs["energy"]),
                grad_norm=float(handle.attrs["grad_norm"]),
                converged=bool(handle.attrs["converged"]),
                coeff=np.ascontiguousarray(handle["orbitals/coeff"][()]),
                inactive=np.ascontiguousarray(handle["orbitals/inactive"][()]),
                active=np.ascontiguousarray(handle["orbitals/active"][()]),
                virtual=np.ascontiguousarray(handle["orbitals/virtual"][()]),
                n_orb=int(handle.attrs["n_orb"]),
                n_active_elec=int(handle.attrs["n_active_elec"]),
                gamma=np.ascontiguousarray(handle["rdm/gamma"][()]),
                gamma2=None if gamma2.size == 0 else gamma2,
                state_energies=np.ascontiguousarray(handle["states/energies"][()]),
                optimizer_state=optimizer_state,
                ci_vectors=None if ci.size == 0 else ci,
                space_key=str(handle.attrs.get("space_key", "")) or None,
                history=np.ascontiguousarray(handle["states/history"][()]),
                metadata=metadata,
                dropped=tuple(json.loads(handle.attrs.get("dropped", "[]"))))
    except CheckpointError:
        raise
    except Exception as exc:                       # h5py raises OSError, KeyError, ValueError
        message = "checkpoint {} could not be read: {}: {}".format(
            path, type(exc).__name__, exc)
        log.error("%s", message)
        raise CheckpointError(message)

    if check_fingerprint and stored_fingerprint and stored_fingerprint != code_fingerprint():
        log.warning("checkpoint %s was written by a different build of Kuiva (source "
                    "fingerprint %s against %s). The restart proceeds -- most source changes "
                    "cannot move a number -- but if the CASSCF, CI or RDM code changed "
                    "numerically, this trajectory is not the one that was interrupted",
                    path, stored_fingerprint[:12], code_fingerprint()[:12])
    if check_fingerprint and stored_backend and stored_backend != _kernel_backend_token():
        log.warning("checkpoint %s was written under kernel backend %s and this run uses "
                    "%s. The restart proceeds -- serial kernel parity is bitwise -- but a "
                    "threaded reduction is only 1e-13-equal, so the resumed trajectory can "
                    "differ from the interrupted one at that level",
                    path, stored_backend, _kernel_backend_token())
    log.debug("checkpoint read: %s, iteration %d, written by Kuiva %s",
              path, checkpoint.iteration, stored_version)
    return checkpoint


# --- The state average, as an identity ------------------------------------------------------

#: Metadata key under which :func:`state_average_key` is stored. Metadata is a free-form
#: ``{str: str}`` map already written into every checkpoint, so recording the state average
#: there costs no schema change and an older file simply carries no entry — which the restart
#: check reads as "cannot be compared", never as "matches".
STATE_AVERAGE_KEY = "state_average"


def state_average_key(solver) -> Optional[str]:
    """A canonical string identifying the state average a solver was built for.

    ⚠ **This is deliberately NOT part of** :meth:`~kuiva.mcscf.casci.FullCISolver.space_key`.
    That key is the identity of the CI *surface* and is compared to decide whether restored
    L-BFGS curvature is still meaningful; widening it would move the key for every
    calculation ever checkpointed and silently downgrade each of their warm starts to cold
    ones. The state average is a different question — not "is this the same surface?" but "is
    this the same calculation?" — so it is checked separately, against the file's metadata,
    and answered by a refusal rather than by discarding curvature.

    The *requested* weights are what is recorded, not the equalized ones the averaging gate
    produces: the request is what a user restates and what a mismatch is about, and the
    equalization is a deterministic function of it and of the spectrum.

    ⚠ **A per-irrep request is recorded as itself, not as its total.** Two different
    ``{irrep: n}`` mappings can sum to the same count while selecting different states, so a
    key holding only the total would let one restart as the other — the exact silent pass this
    check exists to remove.

    Returns ``None`` for a solver that declares no state average, which is not comparable to
    anything and is treated as such.
    """
    n_states = getattr(solver, "n_states", None)
    if n_states is None:
        n_states = getattr(solver, "n_roots", None)      # the tensor-network spelling
    if n_states is None:
        return None
    weights = getattr(solver, "requested_weights", None)
    if weights is None:
        rendered = "equal"
    else:
        rendered = ",".join("{:.12g}".format(float(w)) for w in np.asarray(weights).ravel())
    requested = getattr(solver, "requested_states", None)
    request = ""
    if isinstance(requested, dict):
        request = ";request={}".format(json.dumps({str(k): int(v)
                                                   for k, v in requested.items()},
                                                  sort_keys=True))
    return "n_states={}{};weights={}".format(int(n_states), request, rendered)


def parse_state_average_key(key: Optional[str]):
    """``(n_states, weights)`` back out of :func:`state_average_key`, or ``(None, None)``.

    ⚠ For **defaulting**, never for checking: a restart restates its state average and the key
    refuses a mismatch, and only a materialization — which runs no optimization and so has
    nothing to configure — takes the file's word for it. ``n_states`` comes back as the
    ``{irrep: n}`` mapping where the run used one, so the count that is defaulted is the
    request and not its total.
    """
    if not key:
        return None, None
    fields: Dict[str, str] = {}
    for chunk in str(key).split(";"):
        name, _, value = chunk.partition("=")
        fields[name.strip()] = value.strip()
    if "request" in fields:
        n_states: Any = {str(k): int(v) for k, v in json.loads(fields["request"]).items()}
    elif "n_states" in fields:
        n_states = int(fields["n_states"])
    else:
        return None, None
    rendered = fields.get("weights", "equal")
    weights = (None if rendered in ("", "equal")
               else np.array([float(w) for w in rendered.split(",")]))
    return n_states, weights


# --- The policy ------------------------------------------------------------------------------

@dataclass
class CheckpointStats:
    """What the policy did, so a run can say it rather than the user having to infer it."""
    n_written: int = 0
    n_thinned: int = 0
    n_skipped: int = 0
    gb_written: float = 0.0
    seconds_writing: float = 0.0


class CheckpointPolicy:
    """Adaptive checkpointing for :func:`kuiva.mcscf.orbopt.optimize_orbitals`.

    Construct one and pass :meth:`callback` as the optimizer's ``callback``; it writes a
    restart point after every macro-iteration it judges worth the time. All the numbers come
    from :mod:`kuiva.util.resources` — the budget, the minimum interval, the cost fraction and
    the **measured** disk bandwidth — so the policy adapts to the machine instead of encoding
    a rule about it.

    The decision, in order:

    1. a **converged** iteration is always written (that one is the result, not insurance);
    2. below ``checkpoint_min_interval_seconds`` since the last write, skip quietly;
    3. over ``checkpoint_budget_gb``, or costing more than ``checkpoint_cost_fraction`` of the
       compute elapsed since the last checkpoint, **thin** — drop the rungs of
       :data:`THINNING_LADDER` in order until it fits;
    4. still over either, **skip with a ``WARNING``** (proceeds, but the user should
       know, because a run that is silently not checkpointing looks exactly like one that is).

    ⚠ **A converged iteration drops its curvature before any of that**, because nothing will
    ever resume it and on a large-basis run the L-BFGS pairs are most of the file. See the
    module docstring; the test is convergence and never ``force``, since a deadline or a
    signal forces a write for a run that *will* be resumed.

    ⚠ A user-supplied ``callback`` is chained rather than replaced, and its return value is
    passed through — that is how a wall-clock budget and checkpointing coexist on the
    optimizer's single hook.

    The deadline
    ------------
    ``deadline`` is a :class:`kuiva.util.deadline.Deadline`, and it lives **here** rather
    than beside the policy for one reason: the write and the stop are one decision. At the
    end of each macro-iteration the policy hands the deadline that iteration's wall time and
    its own estimate of what writing *this* checkpoint would cost, and if the answer is that
    another iteration cannot finish and be saved, the checkpoint is written **forced** — past
    the minimum interval and the cost rule, exactly as a converged one is, because this one
    is the result rather than insurance — and the callback returns ``False`` to stop the run.

    ⚠ The write estimate uses the **unthinned** size, so it can only over-estimate: thinning
    happens inside :meth:`write` and makes the write cheaper than the reserve allowed for.
    ⚠ Nothing here polls a clock inside a loop; the deadline is consulted once per
    macro-iteration, at the point where a checkpoint is already being considered.

    ``signals`` is a :class:`kuiva.util.signals.SignalStop` and is the *other* stop cause —
    a kill that was never announced, where nothing can be predicted. The two are handled by
    one piece of code here because from this side they are the same question ("does this
    iteration end the run?") with the same consequence (force the write, then stop); the
    signal is asked first, being free, and it names itself in the warning it prints.
    """

    def __init__(self, path, *, solver=None, budget_gb: Optional[float] = None,
                 min_interval: Optional[float] = None,
                 cost_fraction: Optional[float] = None,
                 ci_vectors: bool = True, metadata: Optional[Dict[str, str]] = None,
                 n_active_elec: Optional[int] = None, enabled: bool = True,
                 deadline=None, signals=None, chain=None) -> None:
        self.path = Path(path)
        self.solver = solver
        self.budget_gb = (res.checkpoint_budget_gb() if budget_gb is None
                          else float(budget_gb))
        self.min_interval = (res.checkpoint_min_interval_seconds() if min_interval is None
                             else float(min_interval))
        self.cost_fraction = (res.checkpoint_cost_fraction() if cost_fraction is None
                              else float(cost_fraction))
        self.store_ci_vectors = bool(ci_vectors)
        self.metadata = dict(metadata or {})
        self.n_active_elec = n_active_elec
        self.enabled = bool(enabled)
        self.deadline = deadline
        self.signals = signals
        self.chain = chain
        self.stats = CheckpointStats()
        self._last_write = time.time()
        self._bandwidth: Optional[float] = None

    # -- the optimizer hook ---------------------------------------------------------------
    def callback(self, info: dict):
        """The ``callback`` :func:`~kuiva.mcscf.orbopt.optimize_orbitals` calls.

        ⚠ **Building the checkpoint is inside the guarded region, not beside it.** The
        module's failure semantics say a *write* failure is a WARNING and the run continues;
        an exception raised while assembling the object would have escaped into the
        optimizer's loop and killed the calculation instead, which is the same promise broken
        one step earlier. Everything from ``from_info`` onwards is therefore guarded here, and
        :meth:`write` keeps its own guard around the file itself.

        Returns ``False`` — the optimizer's stop signal — when the deadline says another
        macro-iteration cannot finish and be saved, or when a signal has asked the run to
        stop; the checkpoint that makes that stop worth making has been written by then.
        """
        converged = bool(info.get("converged"))
        if self.deadline is not None:
            self.deadline.observe(info.get("wall", 0.0))
        # The signal is asked first and costs nothing: an unannounced kill is already on its
        # way, so there is no arithmetic to do and no point pricing a write against a clock
        # that has stopped mattering.
        cause = None
        if self.signals is not None and not converged and self.signals.should_stop():
            cause = self.signals
        write_seconds = 0.0
        if self.enabled:
            try:
                checkpoint = self.from_info(info)
            except Exception as exc:
                log.warning("could not assemble the checkpoint at iteration %s (%s: %s); the "
                            "calculation continues, but there is no restart point for it",
                            info.get("iteration"), type(exc).__name__, exc)
                self.stats.n_skipped += 1
                if cause is None and self.deadline is not None and not converged:
                    cause = self.deadline if self.deadline.should_stop() else None
            else:
                if cause is None and self.deadline is not None and not converged:
                    write_seconds = checkpoint.size_gb() / self._disk_bandwidth()
                    if self.deadline.should_stop(write_seconds):
                        cause = self.deadline
                self.write(checkpoint, force=converged or cause is not None)
        elif cause is None and self.deadline is not None and not converged:
            cause = self.deadline if self.deadline.should_stop() else None
        chained = None if self.chain is None else self.chain(info)
        if cause is not None:
            cause.announce(info, write_seconds=write_seconds,
                           wrote=str(self.path) if self.stats.n_written else None)
            return False
        return chained

    def from_info(self, info: dict) -> CASSCFCheckpoint:
        """Build a checkpoint from the optimizer's iteration info dict.

        The dict was extended additively for exactly this: the orbitals, the RDMs and the
        optimizer are already in hand at the end of a macro-iteration, so nothing has to be
        recomputed to checkpoint. The *states* come from the solver, which the optimizer does
        not know about — it never knew there were states.

        ⚠ **What the solver's ``last`` carries is duck-typed, never assumed.** A
        ``FullCISolver`` leaves a ``CASCIResult`` there, with CI vectors and *total* state
        energies; a tensor-network solver leaves a ``SweepResult``, which has neither — its
        ``energies`` exclude ``e_core`` and are not the same quantity, so they are **not**
        substituted. A solver that cannot supply one of them contributes nothing for it
        rather than raising: the orbitals, the RDMs and the optimizer state are what a restart
        actually needs, and they are present either way.
        """
        spaces: OrbitalSpaces = info["spaces"]
        optimizer = info["optimizer"]
        last = getattr(self.solver, "last", None)
        space_key = None
        if self.solver is not None and hasattr(self.solver, "space_key"):
            space_key = self.solver.space_key()
        vectors = None
        if self.store_ci_vectors and getattr(last, "vectors", None) is not None:
            vectors = np.ascontiguousarray(last.vectors)
        totals = getattr(last, "total_energies", None)
        energies = np.zeros(0) if totals is None else np.asarray(totals, dtype=float)
        if last is not None and totals is None:
            log.debug("the solver's last result (%s) carries no total state energies; the "
                      "checkpoint records none rather than storing a different quantity "
                      "under that name", type(last).__name__)
        n_elec = self.n_active_elec
        if n_elec is None:
            n_elec = int(getattr(self.solver, "n_elec", 0))
        return CASSCFCheckpoint(
            iteration=int(info["iteration"]), energy=float(info["energy"]),
            grad_norm=float(info["grad_norm"]), converged=bool(info.get("converged")),
            coeff=np.ascontiguousarray(info["coeff"]),
            inactive=spaces.inactive, active=spaces.active, virtual=spaces.virtual,
            n_orb=spaces.n_orb, n_active_elec=n_elec,
            gamma=np.ascontiguousarray(info["gamma"]),
            gamma2=np.ascontiguousarray(info["gamma2"]),
            state_energies=energies,
            optimizer_state=optimizer.state_dict(space_key=space_key),
            ci_vectors=vectors, space_key=space_key,
            history=np.asarray(info.get("history", []), dtype=float),
            metadata=self._metadata())

    def _metadata(self) -> Dict[str, str]:
        """``self.metadata`` plus the state average, taken from the solver.

        ⚠ Read from the solver here rather than accepted from the caller, so that a restart
        can never fail to be checkable because a driver forgot to describe its own state
        average. The caller's own entries win nothing and lose nothing: this key is reserved.
        """
        metadata = dict(self.metadata)
        key = state_average_key(self.solver)
        if key is not None:
            metadata[STATE_AVERAGE_KEY] = key
        return metadata

    # -- the decision ---------------------------------------------------------------------
    def write(self, checkpoint: CASSCFCheckpoint, *, force: bool = False) -> bool:
        """Apply the policy and write if it says so. Returns whether anything was written."""
        now = time.time()
        elapsed = max(now - self._last_write, 0.0)
        if not force and elapsed < self.min_interval:
            log.debug("checkpoint at iteration %d skipped: %.1f s since the last one, "
                      "minimum interval is %.1f s", checkpoint.iteration, elapsed,
                      self.min_interval)
            self.stats.n_skipped += 1
            return False

        candidate, thinned = self._fit(checkpoint, elapsed, force)
        if candidate is None:
            self.stats.n_skipped += 1
            return False
        try:
            tic = time.time()
            size = write_checkpoint(self.path, candidate)
            self.stats.seconds_writing += time.time() - tic
        except Exception as exc:
            # ⚠ A write failure is a WARNING and the run continues (module docstring).
            log.warning("could not write the checkpoint to %s (%s: %s); the calculation "
                        "continues, but there is no restart point for iteration %d",
                        self.path, type(exc).__name__, exc, checkpoint.iteration)
            self.stats.n_skipped += 1
            return False
        self.stats.n_written += 1
        self.stats.n_thinned += int(thinned)
        self.stats.gb_written += size
        self._last_write = time.time()
        return True

    def _fit(self, checkpoint: CASSCFCheckpoint, elapsed: float, force: bool):
        """``(checkpoint_to_write_or_None, was_thinned)`` under the budget and the cost rule.

        Thinning is tried **before** skipping, one rung of :data:`THINNING_LADDER` at a time:
        a thinned checkpoint restarts the calculation where a skipped one does not, and the
        ladder gives up the cheapest thing to regenerate first.
        """
        affordable = self.cost_fraction * elapsed
        bandwidth = self._disk_bandwidth()

        def fits(candidate: CASSCFCheckpoint) -> bool:
            size = candidate.size_gb()
            if size > self.budget_gb:
                return False
            return force or (size / bandwidth) <= affordable

        full = checkpoint.size_gb()
        # ⚠ On the way out of a converged run the curvature is not a rung of the ladder but
        # dead weight: nothing resumes a converged optimization, and on a large basis the
        # L-BFGS pairs scale with the rotation-parameter count while the rest of the file
        # scales with the active space. The CI vectors are what this file is *for*.
        if checkpoint.converged and any(checkpoint.optimizer_state.get(k) is not None
                                        for k in CURVATURE_KEYS):
            checkpoint = checkpoint.reduced("curvature")
            log.debug("the converged checkpoint drops the orbital curvature (%.3f GB -> "
                      "%.3f GB): a converged run is never resumed, and the CI vectors are "
                      "what this file exists to carry", full, checkpoint.size_gb())

        if fits(checkpoint):
            return checkpoint, bool(checkpoint.dropped)
        candidate = checkpoint
        for tier in THINNING_LADDER:
            # A rung already gone frees nothing, and marking it dropped would say the policy
            # gave up something this run never had.
            if tier in candidate.dropped or not _has_tier(candidate, tier):
                continue
            if tier == "gamma2" and candidate.ci_vectors is None:
                continue                # implied by the vectors, and only while they are here
            candidate = candidate.reduced(tier)
            if fits(candidate):
                log.debug("checkpoint at iteration %d thinned to drop %s: %.3f GB and %.2f s "
                          "against a %.3f GB budget and %.2f s affordable, from %.3f GB "
                          "whole", checkpoint.iteration, ", ".join(candidate.dropped),
                          candidate.size_gb(), candidate.size_gb() / bandwidth,
                          self.budget_gb, affordable, full)
                return candidate, True

        thin = candidate
        size = thin.size_gb()
        if force:
            # A converged iteration is the result, not insurance: the cadence does not apply
            # to it and only a hard size refusal can stop it.
            log.warning("the final checkpoint is %.3f GB even reduced to the orbitals alone "
                        "(%s dropped), over the %.3f GB budget; NOTHING has been written and "
                        "the converged orbitals exist only in memory. Raise "
                        "checkpoint_budget_gb", size, ", ".join(thin.dropped) or "nothing",
                        self.budget_gb)
            return None, False
        log.warning("checkpoint at iteration %d SKIPPED: %.3f GB would take %.2f s to write "
                    "at %.2f GB/s, against %.2f s affordable (%.0f%% of the %.1f s since the "
                    "last checkpoint) and a %.3f GB budget. The run is not currently "
                    "protected against a crash", checkpoint.iteration, size,
                    size / bandwidth, bandwidth, affordable,
                    100.0 * self.cost_fraction, elapsed, self.budget_gb)
        return None, False

    def _disk_bandwidth(self) -> float:
        if self._bandwidth is None:
            self._bandwidth = res.disk_write_bandwidth_gb_s(self.path.parent)
        return max(self._bandwidth, 1e-9)

    def report(self, logger=None) -> None:
        out.entries(logger or log, [
            ("checkpoint file", str(self.path)),
            ("checkpoints written", self.stats.n_written, "",
             "{} thinned, {} skipped".format(self.stats.n_thinned, self.stats.n_skipped)),
            ("checkpoint volume", self.stats.gb_written, "GB", "", "{:.4f}"),
            ("time spent checkpointing", self.stats.seconds_writing, "s", "", "{:.2f}"),
        ])

    def __repr__(self) -> str:
        return "CheckpointPolicy({}, written={}, skipped={})".format(
            self.path, self.stats.n_written, self.stats.n_skipped)


__all__ = ["CASSCFCheckpoint", "CheckpointError", "CheckpointPolicy", "CheckpointStats",
           "SCHEMA_VERSION", "FINGERPRINTED_MODULES", "CURVATURE_KEYS", "THINNING_LADDER",
           "STATE_AVERAGE_KEY", "SYSTEM_KEY", "checkpoint_size_gb", "check_system",
           "code_fingerprint", "parse_state_average_key", "read_checkpoint",
           "state_average_key", "system_fingerprint", "write_checkpoint"]
