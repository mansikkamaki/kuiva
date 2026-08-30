"""Per-class checkpoints for SC-NEVPT2, and restart from them.

Why this file is kilobytes and the CASSCF one is not
----------------------------------------------------
The driver is a loop over states and, inside it, over the eight excitation classes, and one
``(state, class)`` pair produces a :class:`~kuiva.pt.classes.ClassResult` — eight scalars. A
complete restart table for sixteen states is therefore a few kilobytes, and the adaptive byte
budget that governs :mod:`kuiva.io.checkpoint` has nothing to weigh here: what a write costs is
the ``open``/``rename``, not the payload. The only cadence knob that means anything is a
minimum interval, and it exists so a fast class loop does not hammer a parallel filesystem.

That is the whole argument for doing this at all. An ``E2`` over eight classes on a large
reference is hours; the classes are independent; a job that dies in the last one loses every
one before it. The insurance is essentially free, so the question is only whether the restart
is *the same calculation*, and that is what most of this module is about.

⚠ What is deliberately NOT here
--------------------------------
**The reference.** The orbitals and the CI vectors have exactly one owner and it is the CASSCF
checkpoint (:mod:`kuiva.io.checkpoint`). A restart is handed a live reference — including one
that :func:`kuiva.interface.api.casscf_from_checkpoint` materialized from that file — and what
is stored here is a *digest* of it, not a copy. Two files holding the same orbitals is two
answers to "which orbitals", and the wrong one is chosen silently.

**Everything the driver rebuilds.** The pseudo-canonical orbitals, the transformed integral
blocks and the sigma workspace are the entire memory footprint of the stage and every one of
them is a deterministic function of the orbitals. They are regenerated for the first
incomplete state, which is one Fock build and one block transform — far below the class
evaluations they serve.

⚠ Identity: what a restart is refused for, and why the CI vectors are in it
----------------------------------------------------------------------------
:func:`reference_digest` covers the orbitals, the orbital partition, the active electron
count, the state energies **and the CI vectors**; :func:`options_digest` covers every argument
that changes a number. Both are refusals, not warnings, on the same grounds a restated active
space that disagrees is refused: a different one is a different calculation.

The CI vectors are the interesting one. Inside a degenerate manifold the CI's basis is
arbitrary and the per-state ``E2`` carry an arbitrary share of the manifold's internal spread
— the manifold's *barycentre* is what means something. Resuming against a re-solved reference
would therefore compute half the members in one arbitrary basis and half in another, and the
barycentre of that mixture belongs to neither run. There is no way to detect it afterwards: the
numbers are all plausible. So the vectors are part of the identity, and a mismatch names the
fix — restart from a CASSCF checkpoint whose vectors survived the thinning, or recompute the
correction.

⚠ Failure semantics are :mod:`kuiva.io.checkpoint`'s, unchanged: a **write** failure is a
``WARNING`` and the run continues, a **read** failure on an explicitly requested restart is an
``ERROR`` that propagates, and a schema mismatch hard-refuses.

References
----------
* HDF5: The HDF Group, "Hierarchical Data Format, version 5" (1997-2024),
  https://www.hdfgroup.org/HDF5/.
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

from ..io.checkpoint import CheckpointError, code_fingerprint
from ..util import output as out
from ..util.logging import get_logger

log = get_logger(__name__)

#: Layout version of the file. A mismatch is refused, never guessed at.
SCHEMA_VERSION = 1

#: Modules whose source a resumed correction depends on. ⚠ Warned about on mismatch, never
#: refused — the same reading :mod:`kuiva.io.checkpoint` gives it, and for the same reason:
#: most source changes cannot move a number and only the reader knows whether this one could.
FINGERPRINTED_MODULES = ("kuiva.pt.nevpt2", "kuiva.pt.classes", "kuiva.pt.contractions",
                         "kuiva.pt.blocks", "kuiva.pt.network")

#: Seconds between writes below which one is skipped. ⚠ Not a byte budget: at kilobytes the
#: cost of a checkpoint here is the filesystem call, so what has to be rationed is their
#: number. A forced write (the last class of a state, a stop, the end) ignores it.
DEFAULT_MIN_INTERVAL = 5.0

#: The fields of a :class:`~kuiva.pt.classes.ClassResult`, in the order they are stored. ⚠ The
#: table is written as one rectangular array of these columns rather than as a group per
#: entry: a few thousand HDF5 objects is a slow file, and this stays one dataset however many
#: states and classes there are.
RESULT_FIELDS = ("norm", "energy", "n_perturbers", "n_dropped", "min_denominator",
                 "min_signed_denominator", "max_imaginary")


def _h5py():
    """``h5py``, or ``None``. Imported lazily so checkpointing is optional at import time —
    the same three lines :mod:`kuiva.dmrg.checkpoint` carries, rather than a private import
    across packages."""
    try:
        import h5py
    except ImportError:                                        # pragma: no cover - optional
        return None
    return h5py


def _digest(payload: Dict[str, Any]) -> str:
    return hashlib.blake2b(json.dumps(payload, sort_keys=True, default=repr).encode("utf-8"),
                           digest_size=16).hexdigest()


def _array_digest(array) -> str:
    """A digest of an array's exact bytes, dtype and shape.

    ⚠ Exact, not rounded to a tolerance. What this answers is "are these the same numbers?",
    and a tolerance would turn that into "are these close?", which is a different question with
    no defensible threshold: two CASSCF runs that agree to 1e-10 have genuinely different CI
    vectors inside a degenerate manifold, and that is exactly the case this must catch.
    """
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.blake2b(digest_size=16)
    digest.update("{}|{}|".format(contiguous.dtype, contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def reference_digest(c_spinor, spaces, n_elec: int, energies=None, **extra) -> str:
    """Identity of the reference a correction was computed on.

    The orbitals, the orbital partition, the active electron count and the state energies,
    plus whatever the driver's engine adds through ``extra`` — the CI vectors on the
    conventional route, the root count on the network one, where nothing bitwise
    reproducible exists to hash. See the module docstring for why the vectors are in it
    rather than beside it, and :meth:`kuiva.pt.network._NetworkEngine.reference_arrays` for
    what the network route can and cannot promise as a result.
    """
    payload: Dict[str, Any] = {
        "coeff": _array_digest(c_spinor),
        "inactive": [int(i) for i in np.asarray(spaces.inactive).ravel()],
        "active": [int(i) for i in np.asarray(spaces.active).ravel()],
        "virtual_n": int(np.asarray(spaces.virtual).size),
        "n_orb": int(spaces.n_orb), "n_elec": int(n_elec),
    }
    if energies is not None:
        payload["energies"] = _array_digest(np.asarray(energies, dtype=float))
    for name, value in extra.items():
        payload[name] = (_array_digest(np.atleast_2d(np.asarray(value)))
                         if isinstance(value, np.ndarray) else value)
    return _digest(payload)


def options_digest(**options) -> str:
    """Identity of the *settings* a correction was computed under.

    ⚠ Every argument of :func:`kuiva.pt.nevpt2.sc_nevpt2` that can move a number goes in,
    including the class list: the eight are a partition of the first-order interacting space,
    so a resumed run that quietly widened or narrowed it would report a sum over one set under
    the identity of another.
    """
    return _digest({k: (list(v) if isinstance(v, (tuple, list)) else v)
                    for k, v in options.items()})


@dataclass
class NEVPT2Checkpoint:
    """The accumulated ``(state, class)`` table of one SC-NEVPT2 run.

    Plain scalars and small arrays, like every checkpoint: this is what crosses a process
    boundary. ⚠ It is enough to **reassemble the finished result** as well as to resume an
    unfinished one — :func:`kuiva.pt.nevpt2.assemble_from_checkpoint` — because everything
    :func:`kuiva.pt.nevpt2._assemble` reads is here: the class tables, the reference energies,
    the pseudo-canonical ``eps`` and the frozen/deleted counts.
    """

    reference_key: str
    options_key: str
    n_states: int
    class_names: Tuple[str, ...]
    #: ``(state, class name) -> ClassResult``. Absent pairs are the ones still to do.
    entries: Dict[Tuple[int, str], Any] = field(default_factory=dict)
    #: The classes this calculation actually produces, recorded once the first state has
    #: finished. ⚠ Not the same tuple as :attr:`class_names`: a class whose status is not
    #: ``"energy"``, and one the contraction provider does not serve (the network route's
    #: primed single-external pair), is never evaluated and never appears. Without this a
    #: state could only be called finished by inferring the served set from its own contents,
    #: and a partly-written state would then read as a complete one.
    served_classes: Tuple[str, ...] = ()
    #: ``(n_states,)`` reference total energies; ``nan`` for a state not yet reached.
    e_casscf: np.ndarray = field(default_factory=lambda: np.zeros(0))
    eps_inactive: Optional[np.ndarray] = None
    eps_virtual: Optional[np.ndarray] = None
    n_frozen: int = 0
    n_deleted: int = 0
    fock: str = "state-averaged"
    shift: float = 0.0
    imaginary_shift: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)

    def done(self, state: int, name: str) -> bool:
        return (int(state), str(name)) in self.entries

    @property
    def n_done(self) -> int:
        return len(self.entries)

    @property
    def n_pairs(self) -> int:
        return int(self.n_states) * len(self.class_names)

    def state_finished(self, state: int) -> bool:
        """Has every class this calculation produces been computed for ``state``?

        ⚠ ``False`` until :attr:`served_classes` is known, which is deliberate: before the
        first state has finished there is no authority on which classes count, and guessing
        would let a partly-written state be skipped.
        """
        return bool(self.served_classes) and all(self.done(state, name)
                                                 for name in self.served_classes)

    def first_incomplete_state(self) -> int:
        """The state a resumed run starts at — the lowest not finished."""
        for state in range(int(self.n_states)):
            if not self.state_finished(state):
                return state
        return int(self.n_states)

    def per_state(self) -> List[Dict[str, Any]]:
        """The driver's ``per_state`` list of dictionaries, rebuilt."""
        return [{name: self.entries[(state, name)] for name in self.class_names
                 if self.done(state, name)} for state in range(int(self.n_states))]

    @property
    def complete(self) -> bool:
        """Is every state finished — i.e. is this file a whole correction rather than a
        restart point? ⚠ It says the *table* is whole, never that ``E2`` is complete: a class
        the provider does not serve is absent from :attr:`served_classes` and the assembled
        result reports the sum as partial, which is a different statement."""
        return all(self.state_finished(s) for s in range(int(self.n_states)))

    def report(self, logger=None) -> None:
        out.entries(logger or log, [
            ("NEVPT2 checkpoint", "{}/{} (state, class) pairs".format(self.n_done,
                                                                     self.n_pairs)),
            ("states finished", self.first_incomplete_state()),
            ("classes this run produces", ", ".join(self.served_classes) or "not yet known"),
            ("classes", ", ".join(self.class_names)),
            ("zeroth-order Fock", self.fock),
        ])

    def __repr__(self) -> str:
        return "NEVPT2Checkpoint({}/{} pairs, {} states)".format(self.n_done, self.n_pairs,
                                                                 self.n_states)


def write_nevpt2_checkpoint(path, checkpoint: NEVPT2Checkpoint) -> float:
    """Write ``checkpoint`` to ``path``, atomically. Returns the file size [GB].

    ⚠ Raises on failure. Callers protecting a running calculation go through
    :class:`NEVPT2CheckpointPolicy`, which turns that into a ``WARNING``.
    """
    from ..util import resources as res

    h5py = _h5py()
    if h5py is None:
        raise CheckpointError("h5py is not installed, so checkpoints cannot be written")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(checkpoint.entries)
    table = np.full((len(keys), len(RESULT_FIELDS)), np.nan)
    for row, key in enumerate(keys):
        entry = checkpoint.entries[key]
        for column, name in enumerate(RESULT_FIELDS):
            value = getattr(entry, name)
            table[row, column] = np.nan if value is None else float(value)
    tmp = path.with_name(path.name + ".writing-{}".format(os.getpid()))
    try:
        with h5py.File(str(tmp), "w") as handle:
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["kuiva_version"] = _kuiva_version()
            handle.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            handle.attrs["code_fingerprint"] = code_fingerprint(FINGERPRINTED_MODULES)
            handle.attrs["reference_key"] = checkpoint.reference_key
            handle.attrs["options_key"] = checkpoint.options_key
            handle.attrs["n_states"] = int(checkpoint.n_states)
            handle.attrs["class_names"] = json.dumps(list(checkpoint.class_names))
            handle.attrs["served_classes"] = json.dumps(list(checkpoint.served_classes))
            handle.attrs["n_frozen"] = int(checkpoint.n_frozen)
            handle.attrs["n_deleted"] = int(checkpoint.n_deleted)
            handle.attrs["fock"] = str(checkpoint.fock)
            handle.attrs["shift"] = float(checkpoint.shift)
            handle.attrs["imaginary_shift"] = bool(checkpoint.imaginary_shift)
            handle.attrs["metadata"] = json.dumps(checkpoint.metadata, sort_keys=True)
            handle.attrs["fields"] = json.dumps(list(RESULT_FIELDS))

            table_group = handle.create_group("table")
            table_group.create_dataset("state", data=np.array([k[0] for k in keys],
                                                              dtype=np.int64))
            table_group.create_dataset("class_name",
                                       data=json.dumps([k[1] for k in keys]))
            table_group.create_dataset("values", data=table)

            reference = handle.create_group("reference")
            reference.create_dataset("e_casscf", data=np.asarray(checkpoint.e_casscf,
                                                                 dtype=float))
            for name in ("eps_inactive", "eps_virtual"):
                value = getattr(checkpoint, name)
                reference.create_dataset(name, data=np.zeros(0) if value is None
                                         else np.asarray(value, dtype=float))
        os.replace(str(tmp), str(path))
    except Exception:
        try:
            os.unlink(str(tmp))
        except OSError:
            pass
        raise
    size = Path(path).stat().st_size / res.BYTES_PER_GB
    log.debug("NEVPT2 checkpoint written: %s, %d/%d pairs, %.6f GB",
              path, checkpoint.n_done, checkpoint.n_pairs, size)
    return size


def read_nevpt2_checkpoint(path, *, check_fingerprint: bool = True) -> NEVPT2Checkpoint:
    """Read a checkpoint. ⚠ **Raises** :class:`CheckpointError` on any failure — a restart is
    explicitly requested, and quietly starting over wastes what the file exists to protect."""
    from .classes import ClassResult

    h5py = _h5py()
    if h5py is None:
        raise CheckpointError("h5py is not installed, so the NEVPT2 checkpoint at {} cannot "
                              "be read".format(path))
    path = Path(path)
    if not path.is_file():
        message = "no NEVPT2 checkpoint at {}".format(path)
        log.error("%s", message)
        raise CheckpointError(message)
    try:
        with h5py.File(str(path), "r") as handle:
            version = int(handle.attrs.get("schema_version", -1))
            if version != SCHEMA_VERSION:
                message = ("NEVPT2 checkpoint {} carries schema version {} and this Kuiva "
                           "reads version {}; the layout changed, so the file cannot be "
                           "interpreted".format(path, version, SCHEMA_VERSION))
                log.error("%s", message)
                raise CheckpointError(message)
            fields = tuple(json.loads(handle.attrs.get("fields", "[]")))
            if fields != RESULT_FIELDS:
                raise CheckpointError(
                    "NEVPT2 checkpoint {} stores the columns {} and this Kuiva expects {}"
                    .format(path, list(fields), list(RESULT_FIELDS)))
            states = np.asarray(handle["table/state"][()], dtype=int)
            names = json.loads(_as_text(handle["table/class_name"][()]))
            values = np.asarray(handle["table/values"][()], dtype=float)
            entries = {}
            for row, (state, name) in enumerate(zip(states, names)):
                kwargs = {}
                for column, field_name in enumerate(RESULT_FIELDS):
                    number = values[row, column]
                    if field_name in ("n_perturbers", "n_dropped"):
                        kwargs[field_name] = int(0 if np.isnan(number) else number)
                    elif field_name in ("norm", "max_imaginary"):
                        kwargs[field_name] = float(number)
                    else:
                        kwargs[field_name] = None if np.isnan(number) else float(number)
                entries[(int(state), str(name))] = ClassResult(name=str(name), **kwargs)
            eps = {}
            for name in ("eps_inactive", "eps_virtual"):
                array = np.asarray(handle["reference/" + name][()], dtype=float)
                eps[name] = None if array.size == 0 else array
            checkpoint = NEVPT2Checkpoint(
                reference_key=str(handle.attrs["reference_key"]),
                options_key=str(handle.attrs["options_key"]),
                n_states=int(handle.attrs["n_states"]),
                class_names=tuple(json.loads(handle.attrs["class_names"])),
                served_classes=tuple(json.loads(handle.attrs.get("served_classes", "[]"))),
                entries=entries,
                e_casscf=np.asarray(handle["reference/e_casscf"][()], dtype=float),
                n_frozen=int(handle.attrs.get("n_frozen", 0)),
                n_deleted=int(handle.attrs.get("n_deleted", 0)),
                fock=str(handle.attrs.get("fock", "state-averaged")),
                shift=float(handle.attrs.get("shift", 0.0)),
                imaginary_shift=bool(handle.attrs.get("imaginary_shift", False)),
                metadata=json.loads(handle.attrs.get("metadata", "{}")),
                **eps)
            stored_fingerprint = str(handle.attrs.get("code_fingerprint", ""))
            stored_version = str(handle.attrs.get("kuiva_version", "unrecorded"))
    except CheckpointError:
        raise
    except Exception as exc:
        message = "NEVPT2 checkpoint {} could not be read: {}: {}".format(
            path, type(exc).__name__, exc)
        log.error("%s", message)
        raise CheckpointError(message)

    current = code_fingerprint(FINGERPRINTED_MODULES)
    if check_fingerprint and stored_fingerprint and stored_fingerprint != current:
        log.warning("NEVPT2 checkpoint %s was written by a different build of Kuiva "
                    "(perturbation source fingerprint %s against %s). The restart proceeds "
                    "-- most source changes cannot move a number -- but the classes already "
                    "in the file were computed by the other one, and a resumed E2 is then a "
                    "sum over two", path, stored_fingerprint[:12], current[:12])
    log.debug("NEVPT2 checkpoint read: %s, %d/%d pairs, written by Kuiva %s",
              path, checkpoint.n_done, checkpoint.n_pairs, stored_version)
    return checkpoint


def _as_text(value) -> str:
    """HDF5 hands a stored string back as ``bytes`` or ``str`` depending on the writer."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _kuiva_version() -> str:
    from kuiva import __version__
    return str(__version__)


def check_resumable(stored: NEVPT2Checkpoint, *, reference_key: str, options_key: str,
                    n_states: int, class_names: Sequence[str], path) -> None:
    """Refuse a restart that is not a continuation of the run in the file.

    ⚠ Each of these is a **refusal**, on the same grounds a restated active space that
    disagrees is refused: the file holds finished class energies, and resuming a *different*
    calculation into them produces one number out of two calculations with nothing in the
    output saying so. The reference message names the one fix that is not "start again",
    because the common way to reach it is a thinned CASSCF checkpoint.
    """
    if stored.reference_key != reference_key:
        raise ValueError(
            "the NEVPT2 checkpoint at {} was computed on a different reference (digest {} "
            "against {}): the orbitals, the active space, the state energies or the CI "
            "vectors differ. Inside a degenerate manifold the CI's basis is arbitrary, so "
            "resuming across a re-solved reference would compute some members in one basis "
            "and some in another and the manifold's barycentre would belong to neither run. "
            "Materialize the reference from a CASSCF checkpoint whose CI vectors survived "
            "the thinning, or recompute the correction"
            .format(path, stored.reference_key[:12], reference_key[:12]))
    if stored.options_key != options_key:
        raise ValueError(
            "the NEVPT2 checkpoint at {} was computed with different settings (digest {} "
            "against {}). The Fock mode, a frozen-core or deleted-virtual threshold, a level "
            "shift, a cutoff or the class list changes every number in the file; a restart "
            "continues the calculation that was interrupted"
            .format(path, stored.options_key[:12], options_key[:12]))
    if int(stored.n_states) != int(n_states):
        raise ValueError(
            "the NEVPT2 checkpoint at {} holds {} states and this call asks for {}"
            .format(path, stored.n_states, n_states))
    if tuple(stored.class_names) != tuple(class_names):
        raise ValueError(
            "the NEVPT2 checkpoint at {} holds classes {} and this call asks for {}"
            .format(path, list(stored.class_names), list(class_names)))


class NEVPT2CheckpointPolicy:
    """Accumulates the ``(state, class)`` table and writes it as it fills.

    The cadence is a minimum interval and nothing else (see :data:`DEFAULT_MIN_INTERVAL`), with
    the last class of each state and the end of the run forced past it. ⚠ **A write failure is
    a ``WARNING`` and the run continues** — losing the insurance is not losing the correction,
    and a perturbation that aborts after four hours because a disk filled has thrown away
    everything for nothing.

    ``deadline`` and ``signals`` are :class:`kuiva.util.deadline.Deadline` and
    :class:`kuiva.util.signals.SignalStop`, duck-typed exactly as
    :class:`kuiva.io.checkpoint.CheckpointPolicy` takes them, and they are asked **between
    classes** — which is the granularity that exists here, the perturbation having no
    macro-iteration. A stop forces the write and then raises, so the process exits with the
    finished classes on disk rather than walking into the next hours of doomed work.
    """

    def __init__(self, path, checkpoint: NEVPT2Checkpoint, *,
                 min_interval: Optional[float] = None, enabled: bool = True,
                 deadline=None, signals=None) -> None:
        #: ``None`` where the policy exists only to carry a stop cause — a deadline or a
        #: signal with no ``checkpoint=`` given. The run still stops cleanly; it keeps nothing.
        self.path = None if path is None else Path(path)
        self.checkpoint = checkpoint
        self.min_interval = (DEFAULT_MIN_INTERVAL if min_interval is None
                             else float(min_interval))
        self.enabled = bool(enabled)
        self.deadline = deadline
        self.signals = signals
        self.n_written = 0
        self.n_skipped = 0
        self.seconds_writing = 0.0
        self._last_write = time.time()

    def record(self, state: int, name: str, entry, *, wall: float = 0.0,
               force: bool = False) -> None:
        """Add one finished ``(state, class)`` result and write if the cadence allows."""
        self.checkpoint.entries[(int(state), str(name))] = entry
        if self.deadline is not None:
            self.deadline.observe(float(wall))
        self.write(force=force)

    def write(self, *, force: bool = False) -> bool:
        """Write the table. Returns whether anything was written."""
        if not self.enabled or self.path is None:
            return False
        elapsed = max(time.time() - self._last_write, 0.0)
        if not force and elapsed < self.min_interval:
            self.n_skipped += 1
            return False
        try:
            tic = time.time()
            write_nevpt2_checkpoint(self.path, self.checkpoint)
            self.seconds_writing += time.time() - tic
        except Exception as exc:
            log.warning("could not write the NEVPT2 checkpoint to %s (%s: %s); the "
                        "correction continues, but the classes finished so far are not "
                        "protected", self.path, type(exc).__name__, exc)
            self.n_skipped += 1
            return False
        self.n_written += 1
        self._last_write = time.time()
        return True

    def stop_cause(self):
        """The stop cause asking for one, or ``None``. The signal is asked first, being free
        and being the one already on its way."""
        for cause in (self.signals, self.deadline):
            if cause is not None and cause.should_stop():
                return cause
        return None

    def report(self, logger=None) -> None:
        out.entries(logger or log, [
            ("NEVPT2 checkpoint file", "none" if self.path is None else str(self.path)),
            ("checkpoints written", self.n_written, "", "{} skipped by the interval"
             .format(self.n_skipped)),
            ("time spent checkpointing", self.seconds_writing, "s", "", "{:.2f}"),
        ])

    def __repr__(self) -> str:
        return "NEVPT2CheckpointPolicy({}, written={})".format(self.path, self.n_written)


__all__ = ["DEFAULT_MIN_INTERVAL", "FINGERPRINTED_MODULES", "NEVPT2Checkpoint",
           "NEVPT2CheckpointPolicy", "RESULT_FIELDS", "SCHEMA_VERSION", "check_resumable",
           "options_digest", "read_nevpt2_checkpoint", "reference_digest",
           "write_nevpt2_checkpoint"]
