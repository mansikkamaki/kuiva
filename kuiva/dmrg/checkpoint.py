"""Rolling checkpoints of the tensor-network state, and restart from them.

What is stored
--------------
One :class:`~kuiva.dmrg.sweep.TTNState`, whole: the topology (edges and node contents),
the canonical center, the total quantum number with its moduli, and every block tensor —
per leg the sector quantum numbers and dimensions, the leg signs, the sector table, and
the payload as one flat complex array per tensor. The file is self-contained: a restart
rebuilds the state with no other input, through the **validating** ``BlockTensor``
constructor, so a corrupted payload is caught at read rather than three sweeps later.

The metadata carries the solver's ``space_key`` (topology digest, cap, root count), the
active electron count, the sweep index and per-root energies at the write, and a source
fingerprint over the network modules — recorded and warned about on mismatch, never
refused, exactly like the CASSCF checkpoint's.

Cadence and failure semantics (the checkpoint rules, restated where they bind)
------------------------------------------------------------------------------
The state is written at the end of **each completed sweep, rolling** — one file, replaced
atomically (a unique temporary in the same directory, then ``os.replace``), so a run
killed mid-write leaves the previous sweep's state intact. The adaptive budget applies:
below the minimum interval the write is skipped quietly; a write that would cost more
than its fraction of the compute elapsed since the last one is skipped with a DEBUG line
(there is nothing to thin — the state is one object); a **converged** state is written
unconditionally. A **write** failure is a ``WARNING`` and the sweep continues; a **read**
failure on an explicitly requested restart raises, because silently starting cold wastes
exactly the minutes the file exists to protect.

⚠ What a restart of the network state is, and is not
-----------------------------------------------------
It is a **warm start whose loss costs time, not correctness**: the sweep re-derives its
fixed point from the integrals whatever it starts from, so an absent or stale file
degrades the restart, never the answer. The calculation's *identity* — active space,
state average, orbitals — is the ordinary CASSCF checkpoint's job and is checked there;
this file only refuses what would be arithmetic nonsense (a different electron count) and
**warns and starts cold** on anything that merely spoils the warm start (a different
topology or root count). The orbital optimizer's curvature is chart-scoped by
``space_key`` in the ordinary checkpoint, not here.

Schema
------
``NETWORK_SCHEMA_VERSION`` is bumped whenever the layout changes in a way a reader of the
previous version would get wrong; a mismatch is refused, never guessed at. It is
independent of the CASSCF checkpoint's ``SCHEMA_VERSION`` and of the code version.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util import resources as res
from ..util.logging import get_logger
from .block import BlockTensor, QuantumNumber, Space
from .graph import NetworkGraph
from .sweep import TTNState

log = get_logger(__name__)

#: Layout version of the network-state file (module docstring). Independent of the CASSCF
#: checkpoint's ``SCHEMA_VERSION``.
NETWORK_SCHEMA_VERSION = 1

#: Modules whose source is fingerprinted into the file — the ones a replayed sweep's
#: trajectory depends on. Warned about on mismatch, never refused.
NETWORK_FINGERPRINTED_MODULES = ("kuiva.dmrg.block", "kuiva.dmrg.graph",
                                 "kuiva.dmrg.sweep", "kuiva.dmrg.ttno",
                                 "kuiva.dmrg.sparse")


class NetworkCheckpointError(RuntimeError):
    """A network-state checkpoint could not be read. ⚠ Raised only on the read path: a
    write failure warns and the sweep continues (module docstring)."""


def _h5py():
    try:
        import h5py
    except ImportError:                                        # pragma: no cover - optional
        return None
    return h5py


def _fingerprint(modules: Sequence[str] = NETWORK_FINGERPRINTED_MODULES) -> str:
    """A 128-bit digest of the network sources — same contract as the CASSCF checkpoint's
    ``code_fingerprint``, over this layer's own modules (``kuiva.dmrg`` may not import
    ``kuiva.io``, which reaches ``mcscf`` at module scope)."""
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


def network_checkpoint_gb(state: TTNState) -> float:
    """Exact payload size [GB] of a network-state checkpoint: the block payloads and
    sector tables of every tensor (``TTNState.nbytes``). Excludes the per-leg sector
    metadata and HDF5's own structure, which are ``O(nblocks)`` bytes and scale with
    nothing — stated rather than absorbed into a fudge factor (sizing never pads)."""
    return state.nbytes / res.BYTES_PER_GB


def network_state_path(checkpoint_path) -> Path:
    """The network-state file that lives beside an ordinary CASSCF checkpoint.

    One deterministic rule, shared by the writer and the restart, so the two cannot
    disagree about where the state is: ``run.h5`` -> ``run.network.h5``.
    """
    p = Path(checkpoint_path)
    return p.with_name(p.stem + ".network" + (p.suffix or ".h5"))


# --- serialization ---------------------------------------------------------------------------

def _moduli_arrays(qn: QuantumNumber) -> Tuple[np.ndarray, bool]:
    """``(moduli, none)``: per-component moduli with 0 for an unbounded component."""
    if qn.moduli is None:
        return np.zeros(len(qn), dtype=np.int64), True
    return np.array([0 if m is None else int(m) for m in qn.moduli], dtype=np.int64), False


def _moduli_from(arr: np.ndarray, none: bool):
    if none:
        return None
    return tuple(None if int(m) == 0 else int(m) for m in arr)


def _write_tensor(group, tensor: BlockTensor) -> None:
    group.attrs["ndim"] = int(tensor.ndim)
    group.attrs["signs"] = np.asarray(tensor.signs, dtype=np.int64)
    group.attrs["charge"] = np.asarray(tuple(tensor.charge), dtype=np.int64)
    for i, space in enumerate(tensor.spaces):
        group.create_dataset("qns_{}".format(i),
                             data=np.asarray([tuple(q) for q in space.qns], dtype=np.int64))
        group.create_dataset("dims_{}".format(i),
                             data=np.asarray(space.dims, dtype=np.int64))
    group.create_dataset("sectors", data=tensor.sectors)
    payload = (np.concatenate([b.ravel() for b in tensor.blocks])
               if tensor.blocks else np.zeros(0, dtype=np.complex128))
    group.create_dataset("payload", data=payload)


def _read_tensor(group, moduli, moduli_none: bool) -> BlockTensor:
    ndim = int(group.attrs["ndim"])
    signs = tuple(int(s) for s in group.attrs["signs"])
    mod = _moduli_from(moduli, moduli_none)
    charge = QuantumNumber(*(int(x) for x in group.attrs["charge"]), moduli=mod)
    spaces = []
    for i in range(ndim):
        qns = np.asarray(group["qns_{}".format(i)][()], dtype=np.int64)
        dims = np.asarray(group["dims_{}".format(i)][()], dtype=np.int64)
        spaces.append(Space([(QuantumNumber(*(int(x) for x in row), moduli=mod), int(d))
                             for row, d in zip(qns, dims)]))
    sectors = np.asarray(group["sectors"][()], dtype=np.int64)
    payload = np.ascontiguousarray(group["payload"][()], dtype=np.complex128)
    blocks: List[np.ndarray] = []
    pos = 0
    for row in sectors:
        shape = tuple(int(sp.dims[int(i)]) for sp, i in zip(spaces, row))
        size = int(np.prod(shape)) if shape else 1
        blocks.append(np.ascontiguousarray(payload[pos:pos + size].reshape(shape)))
        pos += size
    if pos != payload.size:
        raise NetworkCheckpointError(
            "tensor payload holds {} elements but the sector table accounts for {}"
            .format(payload.size, pos))
    # The validating constructor, deliberately: this is the read path of a file that
    # outlived the process that wrote it, and a corrupted block is caught here.
    return BlockTensor(tuple(spaces), signs, charge, sectors, blocks)


def write_network_state(path, state: TTNState, *, space_key: Optional[str] = None,
                        sweep: Optional[int] = None,
                        energies: Optional[Sequence[float]] = None,
                        converged: bool = False) -> float:
    """Write ``state`` to ``path``, atomically. Returns the file size [GB].

    ⚠ Raises on failure; a caller protecting a running sweep goes through
    :class:`NetworkCheckpointPolicy`, which turns that into a ``WARNING``.
    """
    h5py = _h5py()
    if h5py is None:
        raise NetworkCheckpointError(
            "h5py is not installed, so network-state checkpoints cannot be written")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    free = res.scratch_free_gb(path.parent)
    if free is not None and network_checkpoint_gb(state) > free:
        raise NetworkCheckpointError(
            "the network state needs {:.3f} GB and {} has {:.3f} GB free".format(
                network_checkpoint_gb(state), path.parent, free))
    graph = state.graph
    moduli, moduli_none = _moduli_arrays(state.charge)
    tmp = path.with_name(path.name + ".writing-{}".format(os.getpid()))
    try:
        with h5py.File(str(tmp), "w") as handle:
            handle.attrs["network_schema_version"] = NETWORK_SCHEMA_VERSION
            from kuiva import __version__
            handle.attrs["kuiva_version"] = str(__version__)
            handle.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            handle.attrs["code_fingerprint"] = _fingerprint()
            handle.attrs["center"] = int(state.center)
            handle.attrs["n_roots"] = int(state.n_roots)
            handle.attrs["n_elec"] = int(state.charge.n)
            handle.attrs["charge"] = np.asarray(tuple(state.charge), dtype=np.int64)
            handle.attrs["moduli"] = moduli
            handle.attrs["moduli_none"] = bool(moduli_none)
            handle.attrs["space_key"] = "" if space_key is None else str(space_key)
            handle.attrs["sweep"] = -1 if sweep is None else int(sweep)
            handle.attrs["converged"] = bool(converged)
            handle.create_dataset("energies",
                                  data=np.zeros(0) if energies is None
                                  else np.asarray(energies, dtype=np.float64))
            g = handle.create_group("graph")
            g.attrs["n_nodes"] = int(graph.n_nodes)
            g.create_dataset("edges", data=np.asarray(
                [(int(u), int(v)) for u, v in graph.edges], dtype=np.int64).reshape(-1, 2))
            flat, offsets = [], [0]
            for content in graph.contents:
                flat.extend(int(m) for m in content)
                offsets.append(len(flat))
            g.create_dataset("contents", data=np.asarray(flat, dtype=np.int64))
            g.create_dataset("content_offsets", data=np.asarray(offsets, dtype=np.int64))
            tensors = handle.create_group("tensors")
            for u, tensor in enumerate(state.tensors):
                if tensor is not None:
                    _write_tensor(tensors.create_group("node_{}".format(u)), tensor)
            roots = handle.create_group("roots")
            for r, center in enumerate(state.centers):
                _write_tensor(roots.create_group("root_{}".format(r)), center)
        os.replace(str(tmp), str(path))
    except Exception:
        try:
            os.unlink(str(tmp))
        except OSError:
            pass
        raise
    size = Path(path).stat().st_size / res.BYTES_PER_GB
    log.debug("network state written: %s, sweep %s, %.4f GB", path, sweep, size)
    return size


def read_network_state(path, *, check_fingerprint: bool = True
                       ) -> Tuple[TTNState, Dict[str, object]]:
    """Read a network state. ⚠ **Raises** :class:`NetworkCheckpointError` on any failure.

    Returns ``(state, metadata)`` with ``metadata`` carrying ``space_key``, ``n_elec``,
    ``n_roots``, ``sweep``, ``energies``, ``converged`` and ``kuiva_version``.
    """
    h5py = _h5py()
    if h5py is None:
        raise NetworkCheckpointError(
            "h5py is not installed, so the network state at {} cannot be read".format(path))
    path = Path(path)
    if not path.is_file():
        message = "no network-state checkpoint at {}".format(path)
        log.error("%s", message)
        raise NetworkCheckpointError(message)
    try:
        with h5py.File(str(path), "r") as handle:
            version = int(handle.attrs.get("network_schema_version", -1))
            if version != NETWORK_SCHEMA_VERSION:
                message = ("network state {} carries schema version {} and this Kuiva "
                           "reads version {}; the layout changed, so the file cannot be "
                           "interpreted".format(path, version, NETWORK_SCHEMA_VERSION))
                log.error("%s", message)
                raise NetworkCheckpointError(message)
            moduli = np.asarray(handle.attrs["moduli"], dtype=np.int64)
            moduli_none = bool(handle.attrs["moduli_none"])
            mod = _moduli_from(moduli, moduli_none)
            charge = QuantumNumber(*(int(x) for x in handle.attrs["charge"]), moduli=mod)
            g = handle["graph"]
            edges = [(int(u), int(v)) for u, v in np.asarray(g["edges"][()],
                                                             dtype=np.int64).reshape(-1, 2)]
            flat = np.asarray(g["contents"][()], dtype=np.int64)
            offs = np.asarray(g["content_offsets"][()], dtype=np.int64)
            contents = [tuple(int(m) for m in flat[offs[i]:offs[i + 1]])
                        for i in range(len(offs) - 1)]
            graph = NetworkGraph(int(g.attrs["n_nodes"]), edges, contents=contents)
            center = int(handle.attrs["center"])
            tensors: List[Optional[BlockTensor]] = [None] * graph.n_nodes
            for name, group in handle["tensors"].items():
                tensors[int(name.split("_")[1])] = _read_tensor(group, moduli, moduli_none)
            roots = [(_read_tensor(handle["roots"]["root_{}".format(r)], moduli,
                                   moduli_none))
                     for r in range(int(handle.attrs["n_roots"]))]
            metadata = {
                "space_key": str(handle.attrs.get("space_key", "")) or None,
                "n_elec": int(handle.attrs["n_elec"]),
                "n_roots": int(handle.attrs["n_roots"]),
                "sweep": int(handle.attrs.get("sweep", -1)),
                "converged": bool(handle.attrs.get("converged", False)),
                "energies": np.asarray(handle["energies"][()], dtype=np.float64),
                "kuiva_version": str(handle.attrs.get("kuiva_version", "unrecorded")),
            }
            stored_fingerprint = str(handle.attrs.get("code_fingerprint", ""))
    except NetworkCheckpointError:
        raise
    except Exception as exc:                       # h5py raises OSError, KeyError, ValueError
        message = "network state {} could not be read: {}: {}".format(
            path, type(exc).__name__, exc)
        log.error("%s", message)
        raise NetworkCheckpointError(message)
    if tensors[center] is not None:
        raise NetworkCheckpointError(
            "network state {} carries an isometry at its own center node {}"
            .format(path, center))
    if check_fingerprint and stored_fingerprint and stored_fingerprint != _fingerprint():
        log.warning("network state %s was written by a different build of the network "
                    "layer (fingerprint %s against %s). The warm start proceeds -- the "
                    "sweep re-derives its fixed point from the integrals -- but the "
                    "resumed trajectory is not a replay of the interrupted one",
                    path, stored_fingerprint[:12], _fingerprint()[:12])
    state = TTNState(graph=graph, center=center, tensors=tensors, centers=roots,
                     charge=charge)
    log.debug("network state read: %s, sweep %d, %d roots", path, metadata["sweep"],
              metadata["n_roots"])
    return state, metadata


# --- the policy ------------------------------------------------------------------------------

class NetworkCheckpointPolicy:
    """Rolling per-sweep checkpointing for :func:`kuiva.dmrg.sweep.solve_ttn`.

    Construct one (or hand a path to ``DMRGSolver(checkpoint=...)``) and pass it as
    ``solve_ttn``'s ``checkpoint=``; it is called at the end of every completed sweep and
    applies the checkpoint cadence rules (module docstring). All the numbers come from
    :mod:`kuiva.util.resources` unless overridden. ``space_key`` is stamped into every
    write; the solver refreshes it before each solve, so an adopted topology is recorded
    under its own key.
    """

    def __init__(self, path, *, space_key: Optional[str] = None,
                 budget_gb: Optional[float] = None,
                 min_interval: Optional[float] = None,
                 cost_fraction: Optional[float] = None, enabled: bool = True) -> None:
        self.path = Path(path)
        self.space_key = space_key
        self.budget_gb = (res.checkpoint_budget_gb() if budget_gb is None
                          else float(budget_gb))
        self.min_interval = (res.checkpoint_min_interval_seconds() if min_interval is None
                             else float(min_interval))
        self.cost_fraction = (res.checkpoint_cost_fraction() if cost_fraction is None
                              else float(cost_fraction))
        self.enabled = bool(enabled)
        self.n_written = 0
        self.n_skipped = 0
        self.gb_written = 0.0
        self.seconds_writing = 0.0
        self._last_write = time.time()
        self._bandwidth: Optional[float] = None
        self._warned = False

    def __call__(self, state: TTNState, *, sweep: int,
                 energies: Optional[Sequence[float]] = None,
                 converged: bool = False) -> bool:
        """Apply the cadence and write if it says so. Returns whether anything was written."""
        if not self.enabled:
            return False
        now = time.time()
        elapsed = max(now - self._last_write, 0.0)
        size = network_checkpoint_gb(state)
        if not converged:
            if elapsed < self.min_interval:
                self.n_skipped += 1
                return False
            affordable = self.cost_fraction * elapsed
            if size > self.budget_gb or (size / self._disk_bandwidth()) > affordable:
                # There is nothing to thin -- the state is one object -- so the cadence
                # skip is quiet; the trajectory file still protects the calculation.
                log.debug("network state at sweep %d skipped: %.3f GB against a %.3f GB "
                          "budget and %.2f s affordable", sweep, size, self.budget_gb,
                          affordable)
                self.n_skipped += 1
                return False
        elif size > self.budget_gb:
            log.warning("the converged network state is %.3f GB, over the %.3f GB "
                        "checkpoint budget; NOTHING has been written and the state exists "
                        "only in memory. Raise checkpoint_budget_gb", size, self.budget_gb)
            self.n_skipped += 1
            return False
        try:
            tic = time.time()
            written = write_network_state(self.path, state, space_key=self.space_key,
                                          sweep=sweep, energies=energies,
                                          converged=converged)
            self.seconds_writing += time.time() - tic
        except Exception as exc:
            # ⚠ A write failure is a WARNING and the sweep continues (module docstring);
            # warned once per policy, because a full disk would otherwise say so once per
            # sweep for the rest of the run.
            if not self._warned:
                log.warning("could not write the network state to %s (%s: %s); the sweep "
                            "continues, but a restart will rebuild the network from "
                            "scratch", self.path, type(exc).__name__, exc)
                self._warned = True
            self.n_skipped += 1
            return False
        self.n_written += 1
        self.gb_written += written
        self._last_write = time.time()
        return True

    def _disk_bandwidth(self) -> float:
        if self._bandwidth is None:
            self._bandwidth = res.disk_write_bandwidth_gb_s(self.path.parent)
        return max(self._bandwidth, 1e-9)

    def report(self, logger=None) -> None:
        out.entries(logger or log, [
            ("network state file", str(self.path)),
            ("network states written", self.n_written, "",
             "{} skipped; rolling, one file".format(self.n_skipped)),
            ("network state volume", self.gb_written, "GB", "", "{:.4f}"),
            ("time spent writing network states", self.seconds_writing, "s", "", "{:.2f}"),
        ])

    def __repr__(self) -> str:
        return "NetworkCheckpointPolicy({}, written={}, skipped={})".format(
            self.path, self.n_written, self.n_skipped)


__all__ = ["NETWORK_SCHEMA_VERSION", "NETWORK_FINGERPRINTED_MODULES",
           "NetworkCheckpointError", "NetworkCheckpointPolicy", "network_checkpoint_gb",
           "network_state_path", "read_network_state", "write_network_state"]
