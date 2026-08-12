"""Atomic mean-field two-electron spin-orbit coupling (X2CAMF).

The working
equations and the local traps are in the module docstrings below, ``decouple.py`` first.
The public entry point is
:func:`kuiva.amf.correction.amf_correction`, and the correction is **on by default** at the
front-end seam (``screening="x2camf"``, the default).

⚠ :mod:`kuiva.amf.x2camf_plugin` is **deliberately not imported here**. It reaches an
optional external dependency, and this package is imported by the front-end — an eager import
would turn a reference-only tool into a runtime requirement. It is reached only through
``amf_correction(method="x2camf-external")`` or by importing it by name.
"""
from . import cache
from .backend import (AtomicDiracBackend, AtomicDiracSolution, FourComponentBlocks,
                      available_backends, get_backend, register_backend)
from .configuration import AtomicConfiguration
from .correction import (AMFCorrection, ScreeningRecord, amf_correction,
                         correction_memory_gb, validate_correction)

__all__ = ["AMFCorrection", "AtomicConfiguration", "ScreeningRecord", "amf_correction", "cache",
           "correction_memory_gb", "validate_correction",
           "AtomicDiracBackend", "AtomicDiracSolution", "FourComponentBlocks",
           "available_backends", "get_backend", "register_backend"]
