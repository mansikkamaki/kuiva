"""Memory and scratch-disk budgeting.

Why this module exists
----------------------
A production run of this code allocates arrays whose size is a steep power of the system
dimensions: the conventional ERI array is ``O(nao^4/8)``, the three-index MO factors are
``naux * n^2``, and the NEVPT2 4-RDM is ``n_act^8`` — 6.4 GB at 12 active spinors,
22 GB at 14, **381 GB at 20**. Every one of those numbers is knowable from the dimensions
*before* a single integral is computed, and discovering them instead as an ``MemoryError``
after an hour of CASSCF is a pure waste of the user's time.

So: the size of every large array is estimated at the earliest point where an estimate is
possible, checked against a limit the user (or the site administrator) has set, and a run
that clearly cannot fit is refused **before it starts**, with a full statement of how the
memory was going to be used.

What is governed, and what is not
---------------------------------
This is deliberately not a memory allocator, and it does not track individual allocations.
It distinguishes two categories, and treats them completely differently:

**Resident arrays** — sized by problem dimensions and known before allocation (the ERI array,
the Cholesky factors, ``B^P_{pq}``, CI vectors, RDMs, MPS tensors). These are *declared* with
:meth:`MemoryBudget.reserve` and checked. There are order 10--100 such calls in a whole run.

**Transient buffers** — blocked kernel temporaries. These are never checked; the kernel simply
asks :meth:`MemoryBudget.transient_gb` once, *outside* its loop, for a number to block
against. This costs one function call per kernel invocation and it makes the code *faster*
than the fixed constants it replaces, because the block size finally scales with the machine.

Nothing here wraps ``np.empty``, hooks the allocator, or runs ``tracemalloc``; no check is
ever made on a hot path (efficiency first, memory management second). The price is that
coverage is partial and honest about it: the SCF is PySCF's, its allocation pattern is
dynamic, and all Kuiva can do is hand PySCF the same budget through ``mol.max_memory``. Such
phases are listed in the pre-flight table as **not governed** rather than silently omitted.

⚠ A reservation lives as long as the process, so a loop needs a scope
---------------------------------------------------------------------
Nothing here observes the Python object it accounts for, so nothing can notice that the array
has been freed: a reservation stays on the ledger until it is given back. That is right for
the one-calculation-per-process case this design assumes, and it fails in a way worth knowing
about for a script that loops over systems — the reservations accumulate, and the *n*-th
calculation is refused against a limit its predecessors filled, with a message that reads as
though the machine were too small. It was met in production on a thirteen-element atomic
series at ``memory_gb = 6``: the sixth element was refused after paying for its
four-component atomic solve, and every element after it was refused in zero seconds.

Two public entry points, and the first is the one to reach for:

* ``with resources.calculation():`` around one calculation — everything reserved inside is
  released at the end, so a loop over systems accounts each one on its own.
* :func:`clear` forgets every reservation while keeping the limits, for a driver whose
  structure does not suit a scope.

⚠ Neither weakens the check itself: a calculation is still refused *before* it allocates, and
the accounted total is still a lower bound on RSS. What is scoped is the **lifetime of a
reservation**, not whether it is checked.

To keep the failure from being silent when neither is used, the ledger stamps each reservation
with the calculation that made it (:meth:`MemoryBudget.begin_calculation`, which
:func:`preflight` calls). A calculation that starts on leftovers **which put it at risk** says
so at the pre-flight rather than several phases later; a refusal itemizes them, sums them, and
where releasing them would be enough it ⚠ **withholds the suggestion to raise the limit** —
a limit raised on a wrong diagnosis swallows the next refusal too, the real one.

⚠ The estimate is a lower bound on RSS
--------------------------------------
MKL allocates working buffers Kuiva never sees, and glibc does not promptly return freed
arenas to the OS, so resident-set size is always somewhat above the sum of array sizes. That
is what ``warn_fraction`` leaves room for. :func:`summary` compares the planned peak against
the true peak RSS (``VmHWM``) at the end of every run, which is the feedback loop that keeps
the sizing functions honest as the code changes — an estimate nobody checks is decoration.

The limit is configured, not guessed (user decision)
----------------------------------------------------------
There is **no built-in default**: a calculation refuses to start until a limit has been set
once. It is read from a configuration file so that a site administrator can choose a sensible
value for the machine and no user has to supply it per run, and any explicit user input beats
the file. See :func:`config_search_path` for the order.

Being wrong in the pessimistic direction is a real failure mode
---------------------------------------------------------------
A hard error is only defensible if the estimate is accurate, so:

* sizing functions compute exact array sizes from shapes and itemsizes; **they never pad**.
  The single safety margin in the code is ``warn_fraction``, and it only warns.
* refusals report the machine's actual free memory alongside the limit, so "raise the limit"
  and "restructure the calculation" are distinguishable at a glance.
* ``allow_overcommit`` downgrades every refusal to a warning, for the case where the user
  knows better than the estimate.
* ``tests/test_resources.py`` pins each sizing function against the ``nbytes`` of the array it
  predicts — bounded on *both* sides, so a function that grows a pessimism factor fails.
"""
from __future__ import annotations

import configparser
import os
import shutil
import sys
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import output as out
from .logging import get_logger

log = get_logger(__name__)

#: Bytes per GB. Binary throughout — the number is compared against ``/proc/meminfo``.
BYTES_PER_GB = 1024.0 ** 3

#: Name of the configuration file searched for along :func:`config_search_path`.
CONFIG_FILENAME = "defaults.conf"

#: Environment overrides. ``KUIVA_MEMORY_GB`` is the one a batch script sets per job.
ENV_MEMORY = "KUIVA_MEMORY_GB"
ENV_SCRATCH_GB = "KUIVA_SCRATCH_GB"
ENV_SCRATCH_DIR = "KUIVA_SCRATCH"
ENV_CONFIG = "KUIVA_CONFIG"
ENV_CHECKPOINT_GB = "KUIVA_CHECKPOINT_GB"

#: Largest checkpoint Kuiva will write [GB] when nothing is configured. ⚠ Unlike the memory
#: limit this **does** have a default, and the asymmetry is deliberate: exceeding the memory
#: limit ends the calculation, whereas exceeding the checkpoint budget only makes a
#: checkpoint smaller. A default that refuses to start is a safety property; a default that
#: writes a smaller restart file is a policy, and the user is not the right person to have to
#: supply one before their first run.
DEFAULT_CHECKPOINT_BUDGET_GB = 2.0

#: Minimum wall time [s] between two checkpoints. Zero means "every macro-iteration", which is
#: right for the expensive iterations this code is built for and wrong for a toy system where
#: an iteration is milliseconds — hence the fraction below, which adapts to both without a
#: number having to be chosen.
DEFAULT_CHECKPOINT_MIN_INTERVAL_S = 0.0

#: Fraction of the compute elapsed since the last checkpoint that writing the next one may
#: cost before it is thinned or skipped ("a set fraction (~5%)"). Applied to a *measured*
#: disk bandwidth, so the policy adapts to the hardware instead of hard-coding a rule.
DEFAULT_CHECKPOINT_COST_FRACTION = 0.05

#: Fraction of the limit above which an allocation warns while still proceeding (a warning:
#: "proceeds but the user should know").
DEFAULT_WARN_FRACTION = 0.7

#: Fraction of the *unreserved* budget a blocked kernel may spend on temporaries. Well below
#: one on purpose: several kernels can be live at once, and the remainder absorbs the
#: allocations no sizing function knows about (BLAS buffers, NumPy intermediates).
DEFAULT_TRANSIENT_FRACTION = 0.25

#: Transient buffer [GB] used when no limit is configured. Kernels never refuse to run for
#: want of configuration — only the drivers do (see :func:`ensure_configured`).
FALLBACK_BUFFER_GB = 0.5


class ConfigurationError(RuntimeError):
    """No memory limit has been configured and a driver needs one."""


class MemoryLimitError(MemoryError):
    """A planned allocation does not fit in the configured memory limit.

    Subclasses :class:`MemoryError` so that code which already catches out-of-memory catches
    this too; the difference is that this one is raised *before* the allocation, and its
    message states the whole plan.
    """


class ScratchLimitError(RuntimeError):
    """A planned file does not fit in the configured scratch limit or the free space."""


# --- Sizing helpers ----------------------------------------------------------------------


def array_gb(shape: Sequence[int], dtype: Any = np.complex128) -> float:
    """Size [GB] of a dense array of ``shape`` and ``dtype``.

    Exact, and computed in floating point so that an ``n_act^8`` RDM does not overflow the
    integer product on the way to being rejected.
    """
    size = 1.0
    for n in shape:
        size *= float(n)
    return size * float(np.dtype(dtype).itemsize) / BYTES_PER_GB


def rdm_gb(n_active: int, rank: int, dtype: Any = np.complex128) -> float:
    """Size [GB] of a dense ``rank``-particle RDM over ``n_active`` spinors.

    ``rank=4`` is the one that decides whether a NEVPT2 is possible at all: the 4-RDM
    is ``n_active^8`` complex numbers, so 12 active spinors need 6.4 GB and 20 need 381 GB.
    """
    return array_gb((n_active,) * (2 * rank), dtype)


# --- Machine and process interrogation ---------------------------------------------------


def _read_meminfo_kb(key: str) -> Optional[float]:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith(key + ":"):
                    return float(line.split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return None


def _cgroup_limit_gb() -> Optional[float]:
    """Memory limit imposed by a cgroup (container, SLURM cpuset), if any.

    Worth reading: inside a container ``MemAvailable`` reports the *host's* memory, which
    would make the "you could just raise the limit" hint in a refusal message wrong.
    """
    for path, unlimited in (("/sys/fs/cgroup/memory.max", "max"),
                            ("/sys/fs/cgroup/memory/memory.limit_in_bytes", None)):
        try:
            with open(path) as fh:
                text = fh.read().strip()
        except OSError:
            continue
        if unlimited is not None and text == unlimited:
            return None
        try:
            value = float(text) / BYTES_PER_GB
        except ValueError:
            continue
        # cgroup v1 reports a nonsense-large sentinel when unlimited.
        if value > 0.0 and value < 1.0e6:
            return value
    return None


def machine_available_gb() -> Optional[float]:
    """Memory [GB] the machine reports as available right now, or ``None`` if unknown.

    ``MemAvailable`` (not ``MemFree``): the kernel's own estimate of what a new allocation
    could obtain, which counts reclaimable page cache. Capped by any cgroup limit.
    """
    kb = _read_meminfo_kb("MemAvailable")
    avail = kb * 1024.0 / BYTES_PER_GB if kb is not None else None
    cgroup = _cgroup_limit_gb()
    if avail is None:
        return cgroup
    return avail if cgroup is None else min(avail, cgroup)


def machine_total_gb() -> Optional[float]:
    kb = _read_meminfo_kb("MemTotal")
    return kb * 1024.0 / BYTES_PER_GB if kb is not None else None


def peak_rss_gb() -> Optional[float]:
    """Peak resident-set size [GB] of this process (``VmHWM``), or ``None`` if unknown.

    One small file read; used once per run in :func:`summary` and optionally once per
    macro-iteration. Never call it inside a kernel.
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) * 1024.0 / BYTES_PER_GB
    except (OSError, IndexError, ValueError):
        return None
    return None


# --- Configuration -----------------------------------------------------------------------


def config_search_path() -> List[Path]:
    """Configuration files, most specific first; the first one that parses wins.

    ==================================================  =================================
    ``$KUIVA_CONFIG``                                   an explicit file, for batch scripts
    ``~/.config/kuiva/defaults.conf``                   per user
    ``/etc/kuiva/defaults.conf``                        per machine (administrator)
    ``<sys.prefix>/etc/kuiva/defaults.conf``            per installation (administrator)
    ``<repo root>/defaults.conf``                       the committed source-tree default
    ==================================================  =================================

    The installed-tree entry is where an installer should write the site default; the
    repo-root entry is what makes a source checkout runnable without one.
    """
    paths: List[Path] = []
    explicit = os.environ.get(ENV_CONFIG)
    if explicit:
        paths.append(Path(explicit))
    home = os.environ.get("XDG_CONFIG_HOME")
    paths.append((Path(home) if home else Path.home() / ".config") / "kuiva" / CONFIG_FILENAME)
    paths.append(Path("/etc/kuiva") / CONFIG_FILENAME)
    paths.append(Path(sys.prefix) / "etc" / "kuiva" / CONFIG_FILENAME)
    paths.append(Path(__file__).resolve().parents[2] / CONFIG_FILENAME)
    return paths


def read_config(path: Optional[Path] = None) -> Tuple[Dict[str, str], Optional[Path]]:
    """Read the first configuration file found. Returns ``(values, path)``.

    Sections are flattened (``[memory] memory_gb`` -> ``memory_gb``): the sections exist for
    the reader's benefit, not as namespaces. A malformed file is a hard error — silently
    falling back on no configuration would defeat the point of having one.
    """
    candidates = [path] if path is not None else config_search_path()
    for candidate in candidates:
        if candidate is None or not Path(candidate).is_file():
            continue
        parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        try:
            parser.read(str(candidate))
        except configparser.Error as exc:
            raise ConfigurationError("cannot parse the Kuiva configuration file {}: {}"
                                     .format(candidate, exc))
        values: Dict[str, str] = {}
        for section in parser.sections():
            values.update({k: v for k, v in parser.items(section)})
        return values, Path(candidate)
    return {}, None


@dataclass(frozen=True)
class ResourceLimits:
    """The resource limits in force, and where each number came from.

    ``source`` is carried so the pre-flight table can say *why* the limit is what it is. A
    limit whose provenance is invisible is a limit the user will not trust when it refuses
    their calculation.
    """

    memory_gb: float
    source: str = "explicit"
    scratch_gb: Optional[float] = None
    scratch_dir: Optional[str] = None
    warn_fraction: float = DEFAULT_WARN_FRACTION
    allow_overcommit: bool = False
    #: Checkpoint policy. One number for the size and one for the cadence, and they live
    #: here rather than in :mod:`kuiva.io.checkpoint` because the resource limits
    #: are defined in one place — "one number, not two".
    checkpoint_budget_gb: float = DEFAULT_CHECKPOINT_BUDGET_GB
    checkpoint_min_interval_seconds: float = DEFAULT_CHECKPOINT_MIN_INTERVAL_S
    checkpoint_cost_fraction: float = DEFAULT_CHECKPOINT_COST_FRACTION

    def __post_init__(self) -> None:
        if not (self.memory_gb > 0.0):
            raise ConfigurationError("memory_gb must be positive, got {!r}"
                                     .format(self.memory_gb))
        if not (0.0 < self.warn_fraction <= 1.0):
            raise ConfigurationError("warn_fraction must lie in (0, 1], got {!r}"
                                     .format(self.warn_fraction))
        if self.checkpoint_budget_gb < 0.0:
            raise ConfigurationError("checkpoint_budget_gb must not be negative, got {!r}"
                                     .format(self.checkpoint_budget_gb))
        if not (0.0 < self.checkpoint_cost_fraction <= 1.0):
            raise ConfigurationError("checkpoint_cost_fraction must lie in (0, 1], got {!r}"
                                     .format(self.checkpoint_cost_fraction))

    @classmethod
    def from_config(cls, memory_gb: Optional[float] = None, *,
                    scratch_gb: Optional[float] = None,
                    scratch_dir: Optional[str] = None,
                    warn_fraction: Optional[float] = None,
                    allow_overcommit: Optional[bool] = None,
                    checkpoint_budget_gb: Optional[float] = None,
                    checkpoint_min_interval_seconds: Optional[float] = None,
                    checkpoint_cost_fraction: Optional[float] = None,
                    path: Optional[Path] = None) -> "ResourceLimits":
        """Resolve the limits: explicit argument > environment > configuration file.

        Raises :class:`ConfigurationError` when no memory limit can be found anywhere — the
        deliberate refusal, so that a limit is chosen once by someone who knows the
        machine rather than guessed every run by the code.
        """
        values, cfg_path = read_config(path)
        origin: Dict[str, str] = {}

        def resolve(name: str, explicit, env: Optional[str], cast):
            if explicit is not None:
                origin[name] = "set for this calculation"
                return cast(explicit)
            if env and os.environ.get(env):
                origin[name] = "$" + env
                return cast(os.environ[env])
            if name in values:
                origin[name] = str(cfg_path)
                return cast(values[name])
            return None

        def as_bool(v):
            return v if isinstance(v, bool) else str(v).strip().lower() in (
                "1", "true", "yes", "on")

        mem = resolve("memory_gb", memory_gb, ENV_MEMORY, float)
        if mem is None:
            raise ConfigurationError(_no_limit_message(cfg_path))
        scr_gb = resolve("scratch_gb", scratch_gb, ENV_SCRATCH_GB, float)
        scr_dir = resolve("scratch_dir", scratch_dir, ENV_SCRATCH_DIR, str)
        warn = resolve("warn_fraction", warn_fraction, None, float)
        over = resolve("allow_overcommit", allow_overcommit, None, as_bool)
        ckpt_gb = resolve("checkpoint_budget_gb", checkpoint_budget_gb, ENV_CHECKPOINT_GB,
                          float)
        ckpt_interval = resolve("checkpoint_min_interval_seconds",
                                checkpoint_min_interval_seconds, None, float)
        ckpt_fraction = resolve("checkpoint_cost_fraction", checkpoint_cost_fraction, None,
                                float)
        # The provenance reported is that of memory_gb: it is the number a refusal is about,
        # and a compound string listing every field's origin is unreadable in an error.
        return cls(memory_gb=mem,
                   source=origin["memory_gb"],
                   scratch_gb=scr_gb,
                   scratch_dir=scr_dir,
                   warn_fraction=DEFAULT_WARN_FRACTION if warn is None else warn,
                   allow_overcommit=bool(over),
                   checkpoint_budget_gb=(DEFAULT_CHECKPOINT_BUDGET_GB if ckpt_gb is None
                                         else ckpt_gb),
                   checkpoint_min_interval_seconds=(DEFAULT_CHECKPOINT_MIN_INTERVAL_S
                                                    if ckpt_interval is None
                                                    else ckpt_interval),
                   checkpoint_cost_fraction=(DEFAULT_CHECKPOINT_COST_FRACTION
                                             if ckpt_fraction is None else ckpt_fraction))

    def resolved_scratch_dir(self) -> Path:
        """Directory scratch files go to: configured, else ``$TMPDIR``, else the cwd."""
        if self.scratch_dir:
            return Path(self.scratch_dir)
        return Path(os.environ.get("TMPDIR", ".")).resolve()


def _no_limit_message(cfg_path: Optional[Path]) -> str:
    seen = "read {} but it sets no memory_gb".format(cfg_path) if cfg_path else \
        "no configuration file was found on the search path"
    total = machine_total_gb()
    suggestion = ""
    if total is not None:
        # Half the machine is a conservative starting point for a shared workstation, not a
        # recommendation: only the administrator knows what else the node has to do.
        suggestion = ("\n  This machine has {:.1f} GB of RAM in total. Leave room for the "
                      "operating\n  system and anything else that has to run alongside."
                      .format(total))
    return (
        "no memory limit is configured, so Kuiva will not start.\n"
        "  ({})\n".format(seen) +
        "  Set one once, in either of these ways:\n"
        "    - write a configuration file, e.g. ~/.config/kuiva/{}:\n".format(CONFIG_FILENAME) +
        "        [memory]\n"
        "        memory_gb = 8.0\n"
        "    - or export {}=8.0 for this job,\n".format(ENV_MEMORY) +
        "    - or pass memory_gb=... to the calculation.\n"
        "  Search path (first match wins):\n    " +
        "\n    ".join(str(p) for p in config_search_path()) + suggestion)


# --- The budget --------------------------------------------------------------------------


@dataclass
class Allocation:
    """One declared array: what it is, how big, and which phase owns it.

    ``generation`` is the calculation that reserved it (:meth:`MemoryBudget.begin_calculation`
    advances the counter). It is what lets a refusal distinguish "this calculation is too
    big" from "the previous one never gave its memory back" — two situations with the same
    arithmetic and opposite remedies.
    """
    label: str
    gb: float
    phase: str
    note: str = ""
    generation: int = 0

    def __str__(self) -> str:
        return "{} ({:.3f} GB)".format(self.label, self.gb)


@dataclass
class PlannedAllocation:
    """An entry in a pre-flight plan (:func:`preflight`), before anything is allocated."""
    label: str
    gb: float
    resident: bool = True
    note: str = ""


@dataclass
class PhaseEstimate:
    """The memory a phase of the calculation is expected to need.

    ``governed=False`` marks a phase whose allocation pattern Kuiva does not control (the
    PySCF SCF). It is still listed, with ``external_note`` saying what *is* done about it —
    hiding it would misrepresent the coverage of the whole mechanism.
    """
    name: str
    allocations: List[PlannedAllocation] = field(default_factory=list)
    governed: bool = True
    external_note: str = ""
    #: What the user could change to make this phase fit; quoted in a refusal.
    advice: List[str] = field(default_factory=list)

    def resident_gb(self) -> float:
        return sum(a.gb for a in self.allocations if a.resident)

    def transient_gb(self) -> float:
        return sum(a.gb for a in self.allocations if not a.resident)


class MemoryBudget:
    """Tracks declared resident arrays against a limit, and refuses what cannot fit.

    Not thread-safe and not meant to be: like :class:`~kuiva.util.timing.TimingRegistry` it
    is touched only from the serial Python orchestration layer.
    """

    def __init__(self, limits: Optional[ResourceLimits] = None) -> None:
        self._limits = limits
        self._resident: List[Allocation] = []
        self._phases: List[str] = []
        self._peak_gb = 0.0
        self._peak_where = ""
        self._plan_peak_gb = 0.0
        self._generation = 0

    # -- configuration --
    @property
    def limits(self) -> Optional[ResourceLimits]:
        return self._limits

    @property
    def configured(self) -> bool:
        return self._limits is not None

    def configure(self, limits: ResourceLimits) -> ResourceLimits:
        self._limits = limits
        return limits

    def clear(self) -> None:
        """Forget all reservations (tests; successive calculations in one process)."""
        self._resident.clear()
        self._phases.clear()
        self._peak_gb = 0.0
        self._peak_where = ""
        self._plan_peak_gb = 0.0
        self._generation = 0

    def reset(self) -> None:
        """:meth:`clear`, and forget the limits too."""
        self.clear()
        self._limits = None

    # -- accounting --
    @property
    def limit_gb(self) -> float:
        return self._limits.memory_gb if self._limits is not None else float("inf")

    def resident_gb(self) -> float:
        return sum(a.gb for a in self._resident)

    def available_gb(self) -> float:
        return max(0.0, self.limit_gb - self.resident_gb())

    @property
    def peak_gb(self) -> float:
        """Largest total Kuiva has committed to (resident plus the transient of the moment)."""
        return self._peak_gb

    @property
    def plan_peak_gb(self) -> float:
        """Peak the last :func:`preflight` predicted, before anything was allocated.

        Kept separate from :attr:`peak_gb` so the end-of-run summary can compare the
        prediction against what was actually declared — the two drifting apart is the first
        symptom of a stale sizing function.
        """
        return self._plan_peak_gb

    @property
    def phase(self) -> str:
        return self._phases[-1] if self._phases else ""

    @contextmanager
    def in_phase(self, name: str) -> Iterator["MemoryBudget"]:
        """Scope reservations to a phase; everything reserved inside is released on exit.

        This is what keeps the accounting from being pessimistic: phases are sequential, so
        the peak is *resident carried forward plus this phase's own arrays*, never the sum
        over all phases.
        """
        self._phases.append(name)
        keep = list(self._resident)
        try:
            yield self
        finally:
            self._phases.pop()
            # Keep exactly what was live on entry, minus anything released meanwhile:
            # everything reserved inside the phase goes out of scope with it.
            self._resident = [a for a in self._resident if a in keep]

    # -- calculation scope --
    def stale_allocations(self) -> List[Allocation]:
        """Reservations made before the current calculation began.

        They are not necessarily wrong — a caller may legitimately hold one calculation's
        arrays while setting up the next — but they are what a refusal has to point at
        before the user concludes that the machine is too small.
        """
        return [a for a in self._resident if a.generation < self._generation]

    def begin_calculation(self, name: str = "", *, warn: bool = True) -> List[Allocation]:
        """Mark the start of a calculation, and report what the last one left behind.

        Called by :func:`preflight`, which is the earliest point of every calculation, and by
        :meth:`calculation`. It only advances the generation counter and reports; releasing
        is the caller's decision, because this cannot see whether the arrays are still live.

        ⚠ A script that holds one small calculation's arrays while starting the next is doing
        nothing wrong, and a warning that fires on every second calculation is one the user
        learns to skip — including the time it is the reason the run stops. So the warning is
        gated on the leftovers being large enough to threaten this calculation: here on what
        can be seen without a plan (they are already past ``warn_fraction`` of the limit on
        their own), and in :func:`preflight`, which passes ``warn=False`` and applies the
        sharper test it can make once the planned peak is known.
        """
        self._generation += 1
        stale = self.stale_allocations()
        if stale and warn and self._limits is not None and \
                sum(a.gb for a in stale) > self._limits.warn_fraction * self._limits.memory_gb:
            self._warn_stale(stale, name)
        return stale

    def _warn_stale(self, stale: Sequence[Allocation], name: str = "") -> None:
        """The one wording of the leftover-ledger warning, wherever it is decided to say it."""
        log.warning(
            "the memory budget still holds %d reservation(s) totalling %.3f GB made before "
            "this calculation%s began, and they count against it. If those arrays are still "
            "live this is the right accounting; if the calculation that made them has "
            "finished, release them with kuiva.util.resources.clear(), or scope each "
            "calculation with 'with kuiva.util.resources.calculation():'.",
            len(stale), sum(a.gb for a in stale), " ({})".format(name) if name else "")

    @contextmanager
    def calculation(self, name: str = "") -> Iterator["MemoryBudget"]:
        """Scope one whole calculation: everything reserved inside is released at its end.

        The remedy for a driver that runs several calculations in one interpreter. Unlike
        :meth:`in_phase` — which scopes one phase *within* a calculation, and whose peak
        model depends on phases being sequential — this makes no statement about the peak; it
        says only that the calculation is over and its arrays are gone.

        Reservations that were already live on entry are kept, so a nested or overlapping
        scope cannot release someone else's array. Identity, not equality, decides: two
        arrays of the same size with the same label are two arrays.
        """
        self.begin_calculation(name)
        held = {id(a) for a in self._resident}
        try:
            yield self
        finally:
            self._resident = [a for a in self._resident if id(a) in held]

    # -- the two verbs --
    def reserve(self, label: str, gb: float, *, note: str = "",
                advice: Sequence[str] = ()) -> Allocation:
        """Declare a resident array of ``gb``; raise :class:`MemoryLimitError` if it will not fit.

        ``advice`` are caller-supplied lines saying what the user could change to make it fit
        — the caller is the only one that knows its own knobs, and an error the user cannot
        act on is only half an error.
        """
        self._check(label, gb, note=note, advice=advice, kind="resident")
        alloc = Allocation(label=label, gb=gb, phase=self.phase, note=note,
                           generation=self._generation)
        if self.configured:
            self._resident.append(alloc)
        return alloc

    def release(self, alloc: Optional[Allocation]) -> None:
        """Give back a reservation (an array that has gone out of scope)."""
        if alloc is not None and alloc in self._resident:
            self._resident.remove(alloc)

    def require(self, label: str, gb: float, *, note: str = "",
                advice: Sequence[str] = ()) -> None:
        """Check that a *transient* ``gb`` fits alongside what is already resident.

        Nothing is recorded: the array is expected to be gone before the next check.
        """
        self._check(label, gb, note=note, advice=advice, kind="transient")

    def transient_gb(self, *, fraction: float = DEFAULT_TRANSIENT_FRACTION,
                     minimum: float = 0.03, maximum: float = 8.0) -> float:
        """Buffer size [GB] a blocked kernel may use for temporaries.

        Called once per kernel invocation, outside the loop. With no limit configured this
        returns :data:`FALLBACK_BUFFER_GB`, so kernels stay callable in isolation (unit
        tests, interactive use) — refusing to run for want of configuration is a driver's
        job, not a kernel's.
        """
        if not self.configured:
            return FALLBACK_BUFFER_GB
        return float(min(maximum, max(minimum, fraction * self.available_gb())))

    # -- the check itself --
    def _check(self, label: str, gb: float, *, note: str, advice: Sequence[str],
               kind: str) -> None:
        if not self.configured:
            log.debug("no memory limit configured; not checking %s (%.3f GB)", label, gb)
            return
        assert self._limits is not None
        total = self.resident_gb() + gb
        if total > self._peak_gb:
            self._peak_gb, self._peak_where = total, label
        if total > self._limits.memory_gb:
            message = self._shortfall_report(label, gb, note, advice, kind)
            if self._limits.allow_overcommit:
                log.warning("%s\n  proceeding anyway: allow_overcommit is set.", message)
                return
            log.error("%s", message)
            raise MemoryLimitError(message)
        if total > self._limits.warn_fraction * self._limits.memory_gb:
            log.warning("%s needs %.3f GB, bringing the total to %.3f GB of the %.3f GB "
                        "limit (%.0f%%). %s", label, gb, total, self._limits.memory_gb,
                        100.0 * total / self._limits.memory_gb,
                        "The estimate excludes BLAS work buffers, so the true resident set "
                        "will be somewhat larger.")
        log.debug("%s %s: %.3f GB, total %.3f / %.3f GB", kind, label, gb, total,
                  self._limits.memory_gb)

    def _shortfall_report(self, label: str, gb: float, note: str, advice: Sequence[str],
                          kind: str) -> str:
        """The message a refusal carries: the whole plan, the shortfall, and what to do.

        Deliberately long. This is the one moment where the user has to decide between
        "give the job more memory" and "set up the calculation differently", and they can
        only do that if they can see where the memory was going.
        """
        assert self._limits is not None
        lim = self._limits.memory_gb
        resident = self.resident_gb()
        lines = [
            "not enough memory for {} ({}).".format(label, kind),
            "",
            "  memory limit                {:12.3f} GB   ({})".format(lim, self._limits.source),
            "  already committed           {:12.3f} GB".format(resident),
            "  needed now                  {:12.3f} GB{}".format(
                gb, "   ({})".format(note) if note else ""),
            "  shortfall                   {:12.3f} GB".format(resident + gb - lim),
        ]
        stale = self.stale_allocations()
        if self._resident:
            lines += ["", "  committed so far:"]
            width = max(len(a.label) for a in self._resident)
            for a in self._resident:
                marks = "[{}]".format(a.phase) if a.phase else ""
                if a in stale:
                    marks = (marks + "  " if marks else "") + "<- earlier calculation"
                lines.append("    {:<{w}}  {:10.3f} GB  {}".format(
                    a.label, a.gb, marks, w=width))
        # ⚠ The two situations have identical arithmetic and opposite remedies: "this
        # calculation is too big for the limit" and "the previous one never gave its memory
        # back". Saying only the first is what makes a user raise the limit blindly — and a
        # limit raised on a wrong diagnosis then swallows the next refusal too.
        stale_gb = sum(a.gb for a in stale)
        stale_suffices = bool(stale) and (resident - stale_gb + gb <= lim)
        if stale:
            lines += [
                "",
                "  {} of these reservations ({:.3f} GB) were made before this calculation "
                "began.".format(len(stale), stale_gb),
                "  A reservation lives as long as the process, so a script running several "
                "calculations",
                "  in one interpreter accumulates them until one is refused against a limit "
                "its",
                "  predecessors filled. If those arrays are no longer live, release them "
                "with",
                "  kuiva.util.resources.clear(), or scope each calculation with",
                "  'with kuiva.util.resources.calculation():'.",
            ]
            if stale_suffices:
                lines.append("  Without them this calculation needs {:.3f} GB and fits in the "
                             "limit as it stands.".format(resident - stale_gb + gb))
        avail = machine_available_gb()
        lines.append("")
        if avail is not None:
            need = resident + gb
            if stale_suffices:
                # Deliberately *not* a suggestion to raise the limit: the ledger, not the
                # machine and not the limit, is what stopped this one.
                lines.append("  This machine reports {:.1f} GB available; releasing the "
                             "reservations above is".format(avail))
                lines.append("  what this needs, not a larger limit.")
            elif avail > need:
                lines.append("  This machine reports {:.1f} GB available: memory_gb = {:.1f} "
                             "would run this".format(avail, _round_up(need)))
                lines.append("  calculation as set up. The limit, not the machine, is what "
                             "stopped it.")
            else:
                lines.append("  This machine reports only {:.1f} GB available, so raising the "
                             "limit will not".format(avail))
                lines.append("  help; the calculation has to be set up differently.")
        if advice:
            lines.append("")
            lines.append("  ways to reduce it:")
            lines += _wrap_bullets(advice)
        lines += [
            "",
            "  Set the limit with memory_gb=..., ${}, or the [memory] section of a".format(
                ENV_MEMORY),
            "  configuration file ({}). allow_overcommit = true".format(CONFIG_FILENAME),
            "  downgrades this error to a warning if you believe the estimate is pessimistic.",
        ]
        return "\n".join(lines)

    # -- reporting --
    def report(self, logger=None, *, title: str = "Memory") -> None:
        """Log the current commitments as a standard output block."""
        logger = logger or log
        if not self.configured:
            out.entry(logger, "memory limit", "not configured")
            return
        assert self._limits is not None
        out.entry(logger, "memory limit", self._limits.memory_gb, "GB",
                  note=self._limits.source, fmt="{:.3f}")
        out.entry(logger, "committed", self.resident_gb(), "GB", fmt="{:.3f}")
        if self._resident:
            for a in self._resident:
                out.entry(logger, "  " + a.label, a.gb, "GB", note=a.note, fmt="{:.3f}")


#: The process-global budget. Modules check against this unless handed another one.
BUDGET = MemoryBudget()


def _wrap_bullets(items: Iterable[str], indent: str = "    ") -> List[str]:
    """Bullet list wrapped to the standard output width, so a refusal is readable in a log file."""
    lines: List[str] = []
    for item in items:
        wrapped = textwrap.wrap(item, width=out.WIDTH - len(indent) - 2) or [item]
        lines.append(indent + "- " + wrapped[0])
        lines += [indent + "  " + w for w in wrapped[1:]]
    return lines


def _round_up(gb: float) -> float:
    """Round a GB figure up to something a human would type into a configuration file."""
    step = 0.5 if gb < 16.0 else (1.0 if gb < 128.0 else 8.0)
    return step * float(int(gb / step) + 1)


# --- Module-level convenience (the API most callers use) ---------------------------------


def configure(memory_gb: Optional[float] = None, *, budget: MemoryBudget = BUDGET,
              **kwargs) -> ResourceLimits:
    """Resolve and install the process-wide limits. See :meth:`ResourceLimits.from_config`."""
    return budget.configure(ResourceLimits.from_config(memory_gb, **kwargs))


def ensure_configured(memory_gb: Optional[float] = None, *, budget: MemoryBudget = BUDGET,
                      **kwargs) -> ResourceLimits:
    """Limits for a driver about to start a calculation, configuring them if necessary.

    Raises :class:`ConfigurationError` when nothing is set anywhere. **This is the refusal
    point**: kernels never raise for want of configuration, drivers do.
    """
    if memory_gb is None and not kwargs and budget.configured:
        assert budget.limits is not None
        return budget.limits
    return configure(memory_gb, budget=budget, **kwargs)


def limits(budget: MemoryBudget = BUDGET) -> Optional[ResourceLimits]:
    return budget.limits


def reset(budget: MemoryBudget = BUDGET) -> None:
    """Clear reservations *and* limits (tests)."""
    budget.reset()


def clear(budget: MemoryBudget = BUDGET) -> None:
    """Forget every reservation, keeping the limits — one finished calculation's arrays.

    For a driver that runs several independent calculations in one interpreter: a
    reservation lives as long as the process, so without this the *n*-th calculation is
    refused against a limit its predecessors filled. Prefer :func:`calculation`, which
    cannot be forgotten halfway through a loop.
    """
    budget.clear()


def calculation(name: str = "", *, budget: MemoryBudget = BUDGET):
    """Scope one calculation's reservations; they are released when the block ends.

    The supported way to run a series of calculations in one process::

        for system in systems:
            with resources.calculation(system.label):
                run(system)

    See :meth:`MemoryBudget.calculation`.
    """
    return budget.calculation(name)


def reserve(label: str, gb: float, **kwargs) -> Allocation:
    return BUDGET.reserve(label, gb, **kwargs)


def require(label: str, gb: float, **kwargs) -> None:
    BUDGET.require(label, gb, **kwargs)


def transient_gb(**kwargs) -> float:
    return BUDGET.transient_gb(**kwargs)


def in_phase(name: str):
    return BUDGET.in_phase(name)


# --- Scratch disk ------------------------------------------------------------------------


def scratch_free_gb(path: Optional[Path] = None) -> Optional[float]:
    """Free space [GB] on the filesystem holding ``path``, or ``None`` if it cannot be read."""
    target = Path(path) if path is not None else Path(".")
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        return shutil.disk_usage(str(target)).free / BYTES_PER_GB
    except OSError:
        return None


def require_scratch(label: str, gb: float, *, budget: MemoryBudget = BUDGET,
                    advice: Sequence[str] = ()) -> Path:
    """Check that a scratch file of ``gb`` may be written, and return the directory.

    Checks the configured ``scratch_gb`` **and** the filesystem's actual free space, using
    whichever is smaller: a limit set at start-up is not evidence that the space is still
    there, since scratch filesystems are shared. There is no reservation bookkeeping — free
    space is re-read on every call, which is cheap and is the only number that is true.
    """
    lims = budget.limits
    directory = lims.resolved_scratch_dir() if lims is not None else Path(".")
    free = scratch_free_gb(directory)
    cap = lims.scratch_gb if lims is not None else None
    bound = min([v for v in (cap, free) if v is not None], default=None)
    if bound is None:
        log.debug("no scratch limit and no free-space reading for %s; not checking", directory)
        return directory
    if gb > bound:
        which = "the configured scratch_gb limit" if cap is not None and bound == cap else \
            "the free space on {}".format(directory)
        lines = ["not enough scratch disk for {}.".format(label), "",
                 "  scratch directory           {}".format(directory),
                 "  needed                      {:12.3f} GB".format(gb)]
        if cap is not None:
            lines.append("  scratch_gb limit            {:12.3f} GB".format(cap))
        if free is not None:
            lines.append("  free on the filesystem      {:12.3f} GB".format(free))
        lines += ["  binding constraint          {}".format(which), ""]
        if advice:
            lines.append("  ways to reduce it:")
            lines += ["    - " + a for a in advice]
        lines.append("  Set scratch_dir / scratch_gb in the [scratch] section of a "
                     "configuration file")
        lines.append("  or through ${} / ${}.".format(ENV_SCRATCH_DIR, ENV_SCRATCH_GB))
        message = "\n".join(lines)
        if lims is not None and lims.allow_overcommit:
            log.warning("%s\n  proceeding anyway: allow_overcommit is set.", message)
            return directory
        log.error("%s", message)
        raise ScratchLimitError(message)
    if free is not None and gb > 0.7 * free:
        log.warning("%s needs %.2f GB of the %.2f GB free on %s; a scratch filesystem is "
                    "shared, so this may not still be true when the file is written.",
                    label, gb, free, directory)
    return directory


# --- Checkpoint budget ---------------------------------------------------------------

#: Probe size and the per-process cache of the measured bandwidth.
_BANDWIDTH_PROBE_MB = 8.0
_BANDWIDTH_CACHE: Dict[str, float] = {}

#: Bandwidth [GB/s] assumed when the probe cannot run. Deliberately **pessimistic** — a
#: spinning disk rather than an NVMe — because the number is used to decide whether a write
#: is affordable, and the failure mode of guessing too high is a checkpoint that costs more
#: than the compute it protects.
FALLBACK_DISK_BANDWIDTH_GB_S = 0.1


def disk_write_bandwidth_gb_s(directory: Optional[Path] = None) -> float:
    """Measured sequential write bandwidth [GB/s] of the filesystem holding ``directory``.

    Probes **once per process per filesystem** by writing and fsyncing a few MB, and caches
    the result: the adaptive checkpoint policy needs a bandwidth to turn a byte count
    into a time, and hard-coding one would defeat the point of the policy being adaptive
    ("this adapts to hardware rather than hard-coding a rule").

    ⚠ Never raises. A probe that cannot run — read-only directory, full filesystem, an
    exotic mount — returns :data:`FALLBACK_DISK_BANDWIDTH_GB_S`, because a failure to
    *measure* the disk must not be able to fail a calculation that would otherwise have run.
    """
    target = Path(directory) if directory is not None else Path(".")
    while not target.is_dir() and target != target.parent:
        target = target.parent
    key = str(target.resolve())
    if key in _BANDWIDTH_CACHE:
        return _BANDWIDTH_CACHE[key]
    import time as _time

    payload = b"\0" * int(_BANDWIDTH_PROBE_MB * 1024 * 1024)
    bandwidth = FALLBACK_DISK_BANDWIDTH_GB_S
    try:
        import tempfile
        handle, name = tempfile.mkstemp(dir=str(target), prefix=".kuiva-probe-")
        try:
            tic = _time.time()
            os.write(handle, payload)
            os.fsync(handle)
            elapsed = _time.time() - tic
        finally:
            os.close(handle)
            os.unlink(name)
        if elapsed > 0.0:
            bandwidth = (len(payload) / BYTES_PER_GB) / elapsed
    except OSError as exc:
        log.debug("disk bandwidth probe on %s failed (%s); assuming %.2f GB/s",
                  target, exc, bandwidth)
    _BANDWIDTH_CACHE[key] = bandwidth
    log.debug("measured disk write bandwidth on %s: %.3f GB/s", target, bandwidth)
    return bandwidth


def checkpoint_budget_gb(budget: MemoryBudget = BUDGET) -> float:
    """Largest checkpoint that may be written [GB]."""
    lims = budget.limits
    return (DEFAULT_CHECKPOINT_BUDGET_GB if lims is None else lims.checkpoint_budget_gb)


def checkpoint_min_interval_seconds(budget: MemoryBudget = BUDGET) -> float:
    """Minimum wall time between two checkpoints [s]."""
    lims = budget.limits
    return (DEFAULT_CHECKPOINT_MIN_INTERVAL_S if lims is None
            else lims.checkpoint_min_interval_seconds)


def checkpoint_cost_fraction(budget: MemoryBudget = BUDGET) -> float:
    """Fraction of elapsed compute a checkpoint write may cost before it is thinned."""
    lims = budget.limits
    return (DEFAULT_CHECKPOINT_COST_FRACTION if lims is None
            else lims.checkpoint_cost_fraction)


# --- Pre-flight --------------------------------------------------------------------------


def _phase_walk(phases: Sequence[PhaseEstimate], carried: float):
    """Walk the phases under the peak model: resident carried forward plus the current
    phase's transients. Returns ``(rows, peak, worst)`` with one ``(phase, carried,
    phase_peak)`` row per phase (``None`` values for an ungoverned one) — the single
    implementation of the model, so :func:`plan_peak_gb` and :func:`preflight` cannot
    disagree about what a plan peaks at."""
    rows = []
    peak = carried
    worst: Optional[PhaseEstimate] = None
    for phase in phases:
        if not phase.governed:
            rows.append((phase, None, None))
            continue
        carried += phase.resident_gb()
        phase_peak = carried + phase.transient_gb()
        # The phase that *peaks* is the one to report, not the first that happens to cross
        # the limit: suggesting a limit that only gets the run as far as the next phase is
        # worse than useless.
        if phase_peak >= peak:
            peak, worst = phase_peak, phase
        rows.append((phase, carried, phase_peak))
    return rows, peak, worst


def plan_peak_gb(phases: Sequence[PhaseEstimate], *, budget: MemoryBudget = BUDGET) -> float:
    """Planned peak [GB] of ``phases``, with no printing, no refusal and no ledger effect.

    Exactly the model :func:`preflight` prints and judges — they share one implementation —
    so a caller can compare alternative plans *before* committing to one and handing it to
    the pre-flight. This is what the automatic two-electron route selection reads.
    """
    _, peak, _ = _phase_walk(phases, budget.resident_gb())
    return peak


def preflight(phases: Sequence[PhaseEstimate], *, budget: MemoryBudget = BUDGET,
              logger=None, title: str = "Memory pre-flight") -> float:
    """Print the phase-by-phase memory plan and refuse a calculation that cannot fit.

    Returns the planned peak [GB]. The peak model is *resident carried forward plus the
    current phase's transients*, because phases are sequential; summing every phase would be
    pessimistic and pessimism is what makes a hard limit unusable.

    The table is printed **before** any verdict is raised, so a refusal always comes with the
    full plan above it in the output file.

    This is also where a calculation *begins* as far as the ledger is concerned
    (:meth:`MemoryBudget.begin_calculation`): it is the earliest point of every calculation
    and nothing has been reserved for this one yet, so anything already committed came from
    an earlier one and is warned about here rather than in a refusal several phases later.
    """
    logger = logger or log
    lims = budget.limits
    limit = lims.memory_gb if lims is not None else float("inf")

    out.subsection(logger, title)
    table = out.Table(logger, [
        out.Column("phase", "{}", 24, align="<"),
        out.Column("resident [GB]", "{:.3f}", 13),
        out.Column("transient [GB]", "{:.3f}", 14),
        out.Column("peak [GB]", "{:.3f}", 10),
        out.Column("verdict", "{}", 9, align="<"),
    ])
    table.start()
    rows, peak, worst = _phase_walk(phases, budget.resident_gb())
    for phase, carried, phase_peak in rows:
        if carried is None:
            table.row(phase.name, None, None, None, "external")
            continue
        if phase_peak > limit:
            verdict = "ERROR"
        elif lims is not None and phase_peak > lims.warn_fraction * limit:
            verdict = "warn"
        else:
            verdict = "ok"
        table.row(phase.name, carried, phase.transient_gb(), phase_peak, verdict)
    table.end()

    for phase in phases:
        for a in phase.allocations:
            if a.note:
                out.entry(logger, "  " + a.label, a.gb, "GB", note=a.note, fmt="{:.3f}")
        if not phase.governed and phase.external_note:
            for line in textwrap.wrap("{}: {}".format(phase.name, phase.external_note),
                                      width=out.WIDTH - len(out.INDENT) - 2,
                                      subsequent_indent="  "):
                out.note(logger, line)
    # After the table, so a warning about a stale ledger sits under the numbers that show it:
    # 'carried' in the first row is exactly what an earlier calculation left behind.
    #
    # ⚠ It warns only when those leftovers are what puts this calculation at risk. Holding
    # one finished calculation's arrays while starting the next is common and harmless when
    # both are small, and a warning that fires on every second calculation in a script is one
    # the user learns to skip — including the time it is the reason the run stops.
    stale = budget.begin_calculation(warn=False)
    if stale:
        if lims is not None and peak > lims.warn_fraction * limit:
            budget._warn_stale(stale)
        else:
            log.debug("%d reservation(s) totalling %.3f GB carried in from an earlier "
                      "calculation", len(stale), sum(a.gb for a in stale))
    budget._plan_peak_gb = max(budget._plan_peak_gb, peak)
    if lims is not None:
        out.entry(logger, "memory limit", limit, "GB", note=lims.source, fmt="{:.3f}")
    out.entry(logger, "planned peak", peak, "GB", fmt="{:.3f}")
    avail = machine_available_gb()
    if avail is not None:
        out.entry(logger, "machine available", avail, "GB", fmt="{:.3f}")

    if lims is not None and peak > limit and worst is not None:
        biggest = max(worst.allocations, key=lambda a: a.gb) if worst.allocations else None
        advice: List[str] = []
        if biggest is not None:
            advice.append("'{}' alone needs {:.3f} GB{}".format(
                biggest.label, biggest.gb,
                " ({})".format(biggest.note) if biggest.note else ""))
        advice += list(worst.advice)
        budget.require("the calculation as set up", peak - budget.resident_gb(),
                       note="peak of the plan above, reached in phase '{}'".format(worst.name),
                       advice=advice)
    return peak


def summary(logger=None, *, budget: MemoryBudget = BUDGET) -> None:
    """Log planned peak against the true peak RSS — the honesty check on the estimates.

    ⚠ ``VmHWM`` is the whole process, so it includes PySCF's SCF and every BLAS buffer, none
    of which Kuiva accounts for. It is therefore expected to sit *above* the planned peak;
    what this catches is the two failure modes that matter: a plan far below reality (the
    sizing functions have gone stale and a hard limit is no longer protecting anyone) and a
    plan far above it (the estimates have grown pessimistic and will refuse runs that would
    have fitted).
    """
    logger = logger or log
    planned = budget.peak_gb
    actual = peak_rss_gb()
    if planned <= 0.0 and actual is None:
        return
    out.subsection(logger, "Memory")
    if budget.limits is not None:
        out.entry(logger, "limit", budget.limits.memory_gb, "GB",
                  note=budget.limits.source, fmt="{:.3f}")
    if budget.plan_peak_gb > 0.0:
        out.entry(logger, "pre-flight estimate", budget.plan_peak_gb, "GB", fmt="{:.3f}")
    out.entry(logger, "declared peak (Kuiva arrays)", planned, "GB", fmt="{:.3f}")
    if actual is not None:
        out.entry(logger, "peak resident set (VmHWM)", actual, "GB", fmt="{:.3f}")
        # Only judge the comparison once the numbers are large enough to mean something: on a
        # small run the resident set is mostly the interpreter, NumPy, PySCF and MKL, none of
        # which Kuiva's sizing functions describe or claim to.
        if planned > 0.5 and actual > 0.5:
            if planned > 2.0 * actual:
                log.warning("the memory plan (%.2f GB) was more than twice the process peak "
                            "(%.2f GB). Estimates this pessimistic refuse calculations that "
                            "would have run; please report the case.", planned, actual)
            elif actual > 4.0 * planned:
                log.warning("the process peaked at %.2f GB against a plan of %.2f GB: most "
                            "of the memory used is not accounted for by Kuiva's sizing "
                            "functions, so the limit is protecting less than it appears to.",
                            actual, planned)


__all__ = [
    "BYTES_PER_GB", "CONFIG_FILENAME", "ENV_MEMORY", "ENV_SCRATCH_DIR", "ENV_SCRATCH_GB",
    "ENV_CONFIG", "ENV_CHECKPOINT_GB",
    "DEFAULT_WARN_FRACTION", "DEFAULT_TRANSIENT_FRACTION", "FALLBACK_BUFFER_GB",
    "DEFAULT_CHECKPOINT_BUDGET_GB", "DEFAULT_CHECKPOINT_MIN_INTERVAL_S",
    "DEFAULT_CHECKPOINT_COST_FRACTION", "FALLBACK_DISK_BANDWIDTH_GB_S",
    "disk_write_bandwidth_gb_s", "checkpoint_budget_gb", "checkpoint_min_interval_seconds",
    "checkpoint_cost_fraction",
    "ConfigurationError", "MemoryLimitError", "ScratchLimitError",
    "array_gb", "rdm_gb", "machine_available_gb", "machine_total_gb", "peak_rss_gb",
    "config_search_path", "read_config", "ResourceLimits",
    "Allocation", "PlannedAllocation", "PhaseEstimate", "MemoryBudget", "BUDGET",
    "configure", "ensure_configured", "limits", "reset", "clear", "calculation",
    "reserve", "require",
    "transient_gb", "in_phase", "scratch_free_gb", "require_scratch", "preflight",
    "plan_peak_gb", "summary",
]
