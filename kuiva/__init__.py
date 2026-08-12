"""Kuiva — relativistic multireference quantum chemistry.

Two-component (X2C) relativity with spin–orbit coupling introduced at the CASSCF/CI
level; conventional complex CI or in-house DMRG; SC-NEVPT2 for dynamic correlation.

A calculation is a short script of stage objects, re-exported here and documented in
``kuiva.interface.stages``::

    ScalarSCF -> Reference -> (CheapCI) -> CASSCF -> (NEVPT2) -> PropertyDump
                                                             \\-> PseudospinExport

Each constructor takes the finished stage before it, ``.run()`` is the only expensive call,
and results are plain attributes. The function API underneath (``kuiva.interface.api`` and
the module drivers) is public and unchanged; ``README.md`` is the manual.
"""

#: The project version, ``MAJOR.MINOR.PATCH`` (MAJOR.MINOR.PATCH; every commit bumps it in the same change).
#:
#: ⚠ **This assignment is the single source of truth.** ``pyproject.toml`` reads it through
#: ``[tool.setuptools.dynamic]``, the run banner prints it, every stored product records it,
#: and the documents quoting it are held in step by test. Keep it a plain literal: setuptools parses this file
#: statically and a computed version would not be readable without importing the package.
__version__ = "0.3.7"

#: The class API (kuiva/interface/stages.py) and the Molecule container, re-exported at the
#: top level so a user script reads ``kuiva.CASSCF(...)``. Resolved lazily (PEP 562): the
#: interface package imports PySCF-adjacent machinery, and ``import kuiva`` must stay
#: side-effect-free and dependency-light for everything that never touches the front-end.
_TOP_LEVEL = {
    "Molecule": "kuiva.interface.api",
    "ScalarSCF": "kuiva.interface.stages",
    "Reference": "kuiva.interface.stages",
    "CheapCI": "kuiva.interface.stages",
    "CASSCF": "kuiva.interface.stages",
    "NEVPT2": "kuiva.interface.stages",
    "PropertyDump": "kuiva.interface.stages",
    "PseudospinExport": "kuiva.interface.stages",
}

__all__ = ["__version__"] + sorted(_TOP_LEVEL)


def __getattr__(name):
    if name in _TOP_LEVEL:
        import importlib
        return getattr(importlib.import_module(_TOP_LEVEL[name]), name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    return sorted(list(globals()) + list(_TOP_LEVEL))
