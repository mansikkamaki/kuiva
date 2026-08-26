"""Scratch-backed storage for cold DMRG environments.

Why this exists, in one measurement: the environment cache — one ``D^2 x D_op`` block
tensor per directed bond, ``2(n-1)`` of them — is the dominant DMRG object at large bond
dimension, while a two-site window ever touches only ``(deg u - 1) + (deg v - 1)`` of them
at once. The hot set is ``O(max degree)``; the cold set is ``O(n)`` and, on the Euler-tour
sweep schedule, an environment written on one leg of the tour is not read again until the
next sweep comes back around. Everything about that shape says "page it": recomputing a cold
environment instead would recurse over its entire subtree (the cache's ``_build`` calls
``get`` on every child), at a flop/byte ratio of order ``D x d`` against one sequential
read-back.

What is stored, and what is not
-------------------------------
Only the **payload** goes to disk: the block arrays, concatenated. The metadata — spaces,
signs, charge, the sector table and its sort keys — stays resident, because it is
``O(nblocks)`` bytes, it is what the adaptive driver's survival test reads, and it is what
lets a page-in be a single contiguous ``readinto`` into one flat buffer whose block-shaped
**views** are handed to :meth:`~kuiva.dmrg.block.BlockTensor._trusted` — no copy, no
re-validation (the tensor is bit-for-bit the one that was written, which the round-trip
test asserts).

Storage is a :class:`kuiva.util.scratch.ExtentFile`: environments are replaced every sweep
(``refresh``), so an append-only file would grow by one full generation per sweep; freed
extents are coalesced and reused instead, and the file's high-water mark tracks the live
set. The scratch *directory* is :func:`kuiva.util.resources.require_scratch`'s decision —
no built-in default, refusal when unconfigured — checked when the file is created and again
whenever its high-water mark doubles, because a scratch filesystem is shared and yesterday's
free space is not a promise.

Failure semantics are the factor spill's, not a cache's: by the time a page-in is asked for,
the RAM copy is gone, and the only recovery would be a full subtree rebuild — so an IO error
propagates rather than degrades.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..util import resources as res
from ..util.logging import get_logger
from ..util.scratch import ExtentFile
from .block import BlockTensor

log = get_logger(__name__)


class _PagedEnv(object):
    """Resident metadata of one paged environment: everything but the payload."""

    __slots__ = ("spaces", "signs", "charge", "sectors", "keys", "shapes", "offset",
                 "nbytes")

    def __init__(self, env: BlockTensor, offset: int, nbytes: int) -> None:
        self.spaces = env.spaces
        self.signs = env.signs
        self.charge = env.charge
        self.sectors = env.sectors
        self.keys = env._keys
        self.shapes = [b.shape for b in env.blocks]
        self.offset = int(offset)
        self.nbytes = int(nbytes)


class EnvironmentPager(object):
    """Store / load / discard cold environments on scratch, keyed as the cache keys them.

    The file is created lazily, at the first :meth:`store` — a solve whose environments all
    fit never touches the scratch configuration at all, which is what keeps the no-default
    scratch rule from reaching calculations that never needed it.
    """

    def __init__(self) -> None:
        self._file: Optional[ExtentFile] = None
        self._paged: Dict[tuple, _PagedEnv] = {}
        self._checked_gb = 0.0
        self.n_stored = 0
        self.n_loaded = 0

    def __len__(self) -> int:
        return len(self._paged)

    def has(self, key: tuple) -> bool:
        return key in self._paged

    def _ensure_file(self, first_gb: float) -> ExtentFile:
        if self._file is None:
            directory = res.require_scratch("DMRG environment paging", first_gb)
            path = os.path.join(str(directory),
                                "kuiva-environments-{}-{:x}.bin".format(os.getpid(),
                                                                        id(self)))
            self._file = ExtentFile(path)
            self._checked_gb = max(first_gb, 1e-9)
            log.debug("environment pager opened %s", path)
        return self._file

    def store(self, key: tuple, env: BlockTensor) -> None:
        """Page ``env`` out. The caller drops its own reference and ledger reservation."""
        nbytes = int(sum(b.nbytes for b in env.blocks))
        f = self._ensure_file(nbytes / 1024.0 ** 3)
        offset = f.allocate(nbytes)
        # Re-check the shared scratch filesystem when the live set has doubled since the
        # last look — per-store checks would be a statvfs inside the sweep loop for a
        # number that moves slowly.
        if f.size_gb > 2.0 * self._checked_gb:
            res.require_scratch("DMRG environment paging", f.size_gb)
            self._checked_gb = f.size_gb
        f.write_at(offset, env.blocks)
        self._paged[key] = _PagedEnv(env, offset, nbytes)
        self.n_stored += 1

    def load(self, key: tuple) -> BlockTensor:
        """Rebuild the environment from its resident metadata and one contiguous read."""
        meta = self._paged.pop(key)
        assert self._file is not None
        flat = np.empty(meta.nbytes // 16, dtype=np.complex128)
        self._file.read_at(meta.offset, flat)
        self._file.free(meta.offset, meta.nbytes)
        blocks: List[np.ndarray] = []
        pos = 0
        for shape in meta.shapes:
            size = int(np.prod(shape)) if shape else 1
            blocks.append(flat[pos:pos + size].reshape(shape))
            pos += size
        self.n_loaded += 1
        # Bit-for-bit the tensor that was written; _trusted skips the validation that
        # cannot fail here (measured at half a sweep when it ran on every construction).
        return BlockTensor._trusted(meta.spaces, meta.signs, meta.charge, meta.sectors,
                                    meta.keys, blocks)

    def discard(self, key: tuple) -> None:
        """Drop a paged copy that is now stale (its environment was refreshed)."""
        meta = self._paged.pop(key, None)
        if meta is not None and self._file is not None:
            self._file.free(meta.offset, meta.nbytes)

    def drop_all(self) -> None:
        """Forget every paged entry (a topology change re-keys the world; see rebind)."""
        if self._file is not None:
            for meta in self._paged.values():
                self._file.free(meta.offset, meta.nbytes)
        self._paged.clear()

    def close(self) -> None:
        if self._file is not None:
            log.debug("environment pager closed: %d stores, %d loads, %.3f GB high water",
                      self.n_stored, self.n_loaded, self._file.size_gb)
            self._file.close()
            self._file = None
        self._paged.clear()


__all__ = ["EnvironmentPager"]
