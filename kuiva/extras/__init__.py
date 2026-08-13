"""Special-purpose methods that are shipped and usable but are **not core functionality**.

Kuiva's core is the relativistic multireference pipeline — X2C, CASSCF/DMRG, SC-NEVPT2, the
property export. This package is where a self-contained method that is *useful beside* that
pipeline lives without becoming part of it: something a user can run and rely on, described
briefly in ``README.md`` and never at the level of a pipeline stage.

Three rules bind everything added here, and they are what keep a special-purpose method from
becoming a hidden dependency of the calculation path:

* **The dependency runs one way, asserted from the sources.** ``kuiva.extras`` imports from
  the core packages; **nothing in the calculation path imports** ``kuiva.extras``. The same
  discipline as :mod:`kuiva.qc` and for the same reason — a back-edge, even a convenience
  re-export, would put a side feature between a user and their calculation.
* **A module here is self-contained.** It reaches core machinery through the same public
  interfaces any other caller would use, so removing the package would break nothing else.
* **``import kuiva.extras`` stays cheap**: the names below resolve lazily (PEP 562), exactly
  as the top-level package does, and no module here imports PySCF at module scope.

Contents
--------
:mod:`kuiva.extras.shells`
    Shell-resolved atomic configurations, ``(n, l, q)`` per shell — what a method needs when
    the principal quantum number matters and the per-``l`` electron count of
    :class:`kuiva.amf.configuration.AtomicConfiguration` is not enough — and the extraction
    of a converged solution's radial functions and ``m``-aligned shell orbitals.
:mod:`kuiva.extras.angular`
    The angular algebra of atomic two-electron integrals: exact Wigner 3j symbols, Gaunt and
    Condon-Shortley coefficients, the real-harmonic ``A^k`` tensors, and ``l . s``. Pure
    angular momentum, no integrals and no basis set.
:mod:`kuiva.extras.slater_condon`
    The Slater-Condon radial parameters ``F^k``, ``G^k`` and ``R^k`` of an atom or ion,
    obtained by inverting the angular expansion of the two-electron integrals over the shell
    orbitals above; the driver that runs the whole thing, and the file it writes.
:mod:`kuiva.extras.spin_orbit`
    The one-electron spin-orbit constants ``zeta_nl`` of the same shells, fitted to the
    two-component one-electron Hamiltonian.
"""

#: Lazily resolved public names, ``name -> module``. Resolution is PEP 562 for the reason the
#: top-level package uses it: a special feature must not cost anything to a run that never
#: touches it.
_LAZY = {
    "AtomicShells": "kuiva.extras.shells",
    "RadialParameters": "kuiva.extras.slater_condon",
    "ShellConfiguration": "kuiva.extras.shells",
    "ShellOrbitals": "kuiva.extras.shells",
    "SlaterCondonResult": "kuiva.extras.slater_condon",
    "SlaterParameter": "kuiva.extras.slater_condon",
    "SpinOrbitConstant": "kuiva.extras.spin_orbit",
    "SpinOrbitConstants": "kuiva.extras.spin_orbit",
    "extract_parameters": "kuiva.extras.slater_condon",
    "extract_shells": "kuiva.extras.shells",
    "extract_spin_orbit": "kuiva.extras.spin_orbit",
    "read_parameters": "kuiva.extras.slater_condon",
    "slater_condon_parameters": "kuiva.extras.slater_condon",
    "write_parameters": "kuiva.extras.slater_condon",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    import importlib

    value = getattr(importlib.import_module(_LAZY[name]), name)
    globals()[name] = value                      # resolved once; later reads are plain lookups
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
