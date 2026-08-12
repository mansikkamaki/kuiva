"""Throttle-aware run accounting for the reference generators.

Why this exists
---------------
The development machine runs ``thermald`` in adaptive mode. Under sustained load it clamps
the CPU by injecting idle cycles (the ``idle_inject/N`` kernel threads, driven by
``intel_powerclamp``), so **wall-clock time on this box is not a measure of computational
cost**. A reference calculation can take three times longer than its arithmetic warrants
purely because the package got hot.

This already produced one wrong conclusion: the scalar Ce2Cl6 generation was recorded as
"did not converge in 50 minutes, reduce the root count" when the real cause was thermal
clamping, not the size of the calculation. Attributing a thermal stall to the physics is
exactly the kind of mistake that gets written into a comment and believed for years.

So every generator records **both** wall and CPU time, plus how much idle time the kernel
injected while it ran. The rule of thumb:

* ``cpu_seconds ~= wall_seconds * n_threads``  -> the run was compute-bound; a long wall time
  means the calculation really is expensive.
* ``cpu_seconds << wall_seconds * n_threads`` with a large ``idle_injected_seconds``
  -> the machine was being cooled, and the wall time says nothing about the method.

Nothing here changes a calculation; it only makes the timings interpretable, and warns (loudly,
WARNING: "proceeds but the user should know") when a recorded timing is thermally distorted.

Reading these numbers on other machines: ``idle_injected_seconds`` is summed over all CPUs, so
on an 8-core box a 60 s fully-clamped interval contributes up to 480 s. Only the *ratio*
matters, which is what :attr:`RunResources.throttle_fraction` reports.
"""
from __future__ import annotations

import os
import re
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_CLOCK_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_THERMAL_ROOT = Path("/sys/class/thermal")

#: Fraction of the available CPU-seconds lost to injected idle above which a run is reported
#: as thermally distorted. 5% is comfortably above the few-percent background thermald keeps
#: up even on an idle machine, and well below the level that actually stretches a run.
THROTTLE_WARN_FRACTION = 0.05


def _idle_inject_cpu_seconds() -> Optional[float]:
    """Total CPU time consumed by the ``idle_inject/N`` kernel threads, summed over CPUs.

    These threads "burn" the time the CPU is being forced to stay idle, so the increase over
    an interval measures how much compute capacity thermal management took away. Returns
    ``None`` where the mechanism is absent (non-Linux, or no powerclamp driver).
    """
    total = 0.0
    found = False
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except (OSError, ValueError):
            continue                                  # process vanished; normal
        # comm is parenthesised and may contain spaces/slashes: take the last ')'.
        close = stat.rfind(")")
        open_ = stat.find("(")
        if close < 0 or open_ < 0:
            continue
        if not stat[open_ + 1:close].startswith("idle_inject"):
            continue
        fields = stat[close + 2:].split()
        try:                                          # utime, stime are fields 14, 15 (1-based)
            total += (int(fields[11]) + int(fields[12])) / _CLOCK_TICKS
            found = True
        except (IndexError, ValueError):
            continue
    return total if found else None


def _powerclamp_state() -> Optional[int]:
    """Current ``intel_powerclamp`` cooling state (0-100, percent of idle injected)."""
    for dev in sorted(_THERMAL_ROOT.glob("cooling_device*")):
        try:
            if (dev / "type").read_text().strip() == "intel_powerclamp":
                return int((dev / "cur_state").read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def package_temperature_c() -> Optional[float]:
    """CPU package temperature in degrees Celsius, if exposed."""
    for zone in sorted(_THERMAL_ROOT.glob("thermal_zone*")):
        try:
            if (zone / "type").read_text().strip() == "x86_pkg_temp":
                return int((zone / "temp").read_text().strip()) / 1000.0
        except (OSError, ValueError):
            continue
    return None


def _child_cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def _self_cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


@dataclass
class RunResources:
    """Wall time, CPU time and thermal context for one reference calculation."""
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    idle_injected_seconds: Optional[float] = None
    temp_start_c: Optional[float] = None
    temp_end_c: Optional[float] = None
    powerclamp_end: Optional[int] = None
    n_cpus: int = field(default_factory=lambda: os.cpu_count() or 1)

    @property
    def throttle_fraction(self) -> Optional[float]:
        """Injected idle as a fraction of the machine's total CPU-seconds over the interval.

        ``None`` when the platform does not expose idle injection.
        """
        if self.idle_injected_seconds is None or self.wall_seconds <= 0:
            return None
        available = self.wall_seconds * self.n_cpus
        return self.idle_injected_seconds / available if available > 0 else None

    @property
    def throttled(self) -> bool:
        frac = self.throttle_fraction
        return frac is not None and frac > THROTTLE_WARN_FRACTION

    @property
    def parallel_efficiency(self) -> float:
        """``cpu_seconds / wall_seconds`` - effective cores used. Low *and* throttled means
        the wall time is thermal, not computational."""
        return self.cpu_seconds / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def as_dict(self) -> Dict:
        out: Dict = {"wall_seconds": round(self.wall_seconds, 1),
                     "cpu_seconds": round(self.cpu_seconds, 1),
                     "effective_cores": round(self.parallel_efficiency, 2),
                     "n_cpus": self.n_cpus}
        if self.idle_injected_seconds is not None:
            out["idle_injected_seconds"] = round(self.idle_injected_seconds, 1)
            frac = self.throttle_fraction
            out["throttle_fraction"] = round(frac, 4) if frac is not None else None
            out["thermally_throttled"] = self.throttled
        if self.temp_start_c is not None:
            out["temp_start_c"] = round(self.temp_start_c, 1)
        if self.temp_end_c is not None:
            out["temp_end_c"] = round(self.temp_end_c, 1)
        if self.powerclamp_end is not None:
            out["powerclamp_end"] = self.powerclamp_end
        return out

    def summary(self) -> str:
        """One-line human summary, suitable for a generator's progress output."""
        s = f"{self.wall_seconds:.0f} s wall / {self.cpu_seconds:.0f} s cpu"
        frac = self.throttle_fraction
        if frac is not None and self.throttled:
            s += f", THROTTLED {100 * frac:.0f}%"
        if self.temp_end_c is not None:
            s += f", {self.temp_end_c:.0f} C"
        return s


class track_resources:
    """Context manager recording wall/CPU time and thermal clamping around a calculation.

    Counts CPU time of this process *and* of reaped children, so it covers both the in-process
    PySCF runs and the OpenMolcas/DIRAC subprocesses.

    >>> with track_resources() as res:      # doctest: +SKIP
    ...     run_the_calculation()
    >>> res.summary()                       # doctest: +SKIP
    '1017 s wall / 940 s cpu, 62 C'
    """

    def __init__(self) -> None:
        self.result = RunResources()

    def __enter__(self) -> RunResources:
        self._t0 = time.time()
        self._cpu0 = _self_cpu_seconds() + _child_cpu_seconds()
        self._idle0 = _idle_inject_cpu_seconds()
        self.result.temp_start_c = package_temperature_c()
        return self.result

    def __exit__(self, *exc) -> bool:
        self.result.wall_seconds = time.time() - self._t0
        self.result.cpu_seconds = (_self_cpu_seconds() + _child_cpu_seconds()) - self._cpu0
        idle1 = _idle_inject_cpu_seconds()
        if self._idle0 is not None and idle1 is not None:
            self.result.idle_injected_seconds = max(0.0, idle1 - self._idle0)
        self.result.temp_end_c = package_temperature_c()
        self.result.powerclamp_end = _powerclamp_state()
        return False


def describe_environment() -> Dict:
    """Thermal-management context to store once per reference file.

    Records whether an idle-injection mechanism is present at all, so a future reader can tell
    a machine whose timings are trustworthy from one whose timings are not.
    """
    return {"n_cpus": os.cpu_count(),
            "idle_injection_available": _idle_inject_cpu_seconds() is not None,
            "powerclamp_state": _powerclamp_state(),
            "package_temp_c": package_temperature_c()}


__all__ = ["RunResources", "THROTTLE_WARN_FRACTION", "describe_environment",
           "package_temperature_c", "track_resources"]
