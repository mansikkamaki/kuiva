"""Raw scratch-file primitives shared by every out-of-core store.

Two loops and one allocator, defined once so that no out-of-core layer re-derives them:
a raw (unbuffered) file's ``readinto`` and ``write`` may both return short, and a partial
array silently zero-padded is a numerical error with no message. The scratch *policy* —
where the directory comes from, and that its absence refuses — is
:func:`kuiva.util.resources.require_scratch`'s; this module only moves bytes.

All IO here is deliberately **buffered by the page cache, never O_DIRECT**: the configured
memory limit governs Kuiva's own arrays, so whatever RAM the machine has beyond it is
exactly what the OS page cache uses. A store re-read while the machine has spare RAM is
served at memory speed, and one on a tight machine degrades gracefully to device speed —
an adaptive cache nobody had to size.
"""
from __future__ import annotations

import os
import weakref
from typing import List, Optional, Tuple

import numpy as np

from .logging import get_logger

log = get_logger(__name__)


def readinto_full(fileobj, array: np.ndarray) -> None:
    """Fill ``array`` completely from ``fileobj`` at its current offset, or raise."""
    view = memoryview(array).cast("B")
    got = 0
    while got < len(view):
        n = fileobj.readinto(view[got:])
        if not n:
            raise OSError("scratch file ended {} bytes early; the file was truncated or "
                          "the scratch filesystem lost it".format(len(view) - got))
        got += n


def write_full(fileobj, array: np.ndarray) -> None:
    """Write all of ``array``; a raw file's ``write`` may be short, exactly like its read."""
    view = memoryview(array).cast("B")
    put = 0
    while put < len(view):
        n = fileobj.write(view[put:])
        if n is None or n <= 0:
            raise OSError("short write to a scratch file ({} of {} bytes)"
                          .format(put, len(view)))
        put += n


def _cleanup(fileobj, path: str) -> None:
    """Finalizer for a scratch file: close and delete. Never raises."""
    try:
        fileobj.close()
    except Exception:
        pass
    try:
        os.unlink(path)
    except OSError:
        pass


class ExtentFile:
    """A scratch file managed as reusable extents — for stores whose entries come and go.

    The factor spill writes its rows once and reads them forever, so a fixed layout serves
    it; a store whose entries are replaced every sweep (the DMRG environment pager is the
    case this exists for) would instead grow by one full generation per sweep on an
    append-only file. This allocator hands out byte extents, takes them back, coalesces
    adjacent free ones, and reuses them best-fit — so the file's high-water mark tracks the
    *live* set, not the history.

    Single-threaded by design (one consumer per store, exactly like the ledger); a store
    that wants overlap adds its own worker, as the factor spill's reader does.
    """

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._file = open(self.path, "w+b", buffering=0)
        self._free: List[Tuple[int, int]] = []     # (offset, nbytes), sorted by offset
        self._end = 0                              # high-water mark [bytes]
        self._finalizer = weakref.finalize(self, _cleanup, self._file, self.path)

    @property
    def size_gb(self) -> float:
        """High-water mark [GB] — what the file occupies on the scratch filesystem."""
        return self._end / 1024.0 ** 3

    def allocate(self, nbytes: int) -> int:
        """An extent of ``nbytes``: the best-fitting free hole, else the end of the file."""
        nbytes = int(nbytes)
        best = None
        for i, (off, size) in enumerate(self._free):
            if size >= nbytes and (best is None or size < self._free[best][1]):
                best = i
        if best is not None:
            off, size = self._free.pop(best)
            if size > nbytes:                      # return the tail of the hole
                self._free.insert(best, (off + nbytes, size - nbytes))
                self._free.sort()
            return off
        off = self._end
        self._end += nbytes
        return off

    def free(self, offset: int, nbytes: int) -> None:
        """Return an extent, coalescing with free neighbours (and with the file end)."""
        self._free.append((int(offset), int(nbytes)))
        self._free.sort()
        merged: List[Tuple[int, int]] = []
        for off, size in self._free:
            if merged and merged[-1][0] + merged[-1][1] == off:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((off, size))
        if merged and merged[-1][0] + merged[-1][1] == self._end:
            self._end = merged.pop()[0]
        self._free = merged

    def write_at(self, offset: int, arrays) -> None:
        """Write ``arrays`` back to back starting at ``offset``."""
        self._file.seek(int(offset))
        for a in arrays:
            write_full(self._file, a)

    def read_at(self, offset: int, array: np.ndarray) -> None:
        """Fill ``array`` from ``offset``."""
        self._file.seek(int(offset))
        readinto_full(self._file, array)

    def close(self) -> None:
        self._finalizer()


__all__ = ["ExtentFile", "readinto_full", "write_full"]
