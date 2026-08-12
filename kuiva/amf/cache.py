"""Persistent on-disk cache for atomic mean-field corrections.

Why this exists, in one number
------------------------------
The correction is a property of ``(element, basis, charge, configuration, interaction,
backend, c, uncontract)`` and of **nothing else** — not the geometry, not the other atoms, not
anything the molecular SCF does. :mod:`kuiva.amf.atomic` already exploits that with a
module-level dictionary, which removes the cost from a molecule with repeated elements and
from a test session. It does **not** remove it from the case that matters most: a
potential-energy surface run as N separate jobs pays the atomic solve N times, and for a
lanthanide that solve is ~35 minutes of four-component average-of-configuration SCF (CeCl3:
45 minutes of wall time before the molecular work starts, essentially all of it Ce(3+)).
A 40-point scan therefore pays about 24 hours for a quantity that has one value.

This module makes that a one-time payment. It is the on-disk cache
:func:`kuiva.amf.atomic.cache_key` was written for.

What is stored, and what deliberately is not
--------------------------------------------
**The correction, not the four-component solution.** The solution is the expensive thing, so
storing it is the tempting choice, and it is the wrong one: it is three orders of magnitude
larger (four ``(2*nao)^2`` complex blocks each for ``hcore``, ``overlap``, ``density`` and
``veff`` — 258 MB for a decontracted lanthanide against ~1 MB for the correction), and
everything downstream wants the correction. The checkpoint rule is to keep what is *cheap
and precious*; an intermediate that is neither is regenerated. A solution cache would be a
different feature with a different cost profile, and it is not this one.

⚠ Three ways an on-disk cache silently returns a wrong answer, and what stops each
-----------------------------------------------------------------------------------
An in-process cache can only be wrong about the key. A persistent one can also be wrong about
*time* — it outlives the code that wrote it — and that is a new failure mode for this project,
so each way is closed explicitly rather than by care:

1. ⚠ **The physics changes and the stale entries do not.** This is the dangerous one: a change of
   convention once moved which ``X`` the decoupling uses
   (:func:`kuiva.amf.decouple.x2c_decoupling`), which
   moves ``dh_sf`` by 10% for the *same* request. A cache written the day before would have
   gone on serving the old matrices forever, with nothing anywhere reading wrong.
   :data:`FORMULA_VERSION` is stored in every entry and checked on every read; **it must be
   bumped whenever the numerical content of an** :class:`~kuiva.amf.decouple.AtomicAMF`
   **changes for an unchanged request.** An entry from a different version is a miss, not an
   error — old entries simply stop being used, and a rerun overwrites them.
2. **The key is incomplete.** Every field of the :class:`~kuiva.amf.atomic.AtomicRequest` is
   written into the file and compared on read, not just hashed into the filename. So a
   filename collision, a truncated digest or a future field that someone forgets to put in
   :func:`kuiva.amf.atomic.cache_key` presents as a miss rather than as another element's
   correction. ⚠ The comparison is against the *stored* request, so adding a field to
   ``AtomicRequest`` without adding it here is caught by
   ``tests/test_amf_cache.py::test_every_request_field_is_stored_and_checked``.
3. **The file is corrupt or half-written.** A job killed mid-write, a full filesystem, two
   processes racing on the same entry. Writes go to a unique temporary name in the same
   directory and are renamed into place, which is atomic on POSIX; a file that fails to open
   or to validate is a miss with a ``WARNING``, never an exception. **A cache is an
   optimization and must never be able to fail a calculation** — that is why every operation
   here is wrapped and degrades to "compute it again".

Configuration
-------------
``[amf] cache_dir`` in the ``defaults.conf`` configuration file, ``$KUIVA_AMF_CACHE`` overriding it, and
``~/.cache/kuiva/amf`` (or ``$XDG_CACHE_HOME/kuiva/amf``) as the default. Setting either to
``off`` (or ``none``/``disabled``/an empty value) turns the cache off entirely.

⚠ **The default is *on*, unlike the configured memory limit, which refuses to guess.** The reasons
are not the same: a memory limit is a property of the machine that only the user knows, while
a cache directory has a correct conventional answer, and a wrong guess costs disk rather than
a failed run. What the two do share is that the choice is *reported* — the directory appears
in the correction's output block the first time it is used.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..util import resources
from ..util.logging import get_logger

log = get_logger(__name__)

#: File-format version. Bump when the *layout* changes (a renamed dataset, a new required
#: attribute) — not when the physics does; that is :data:`FORMULA_VERSION`.
SCHEMA = 1

#: ⚠ **Version of the correction's numerical content.** Bump this whenever a change to
#: :mod:`kuiva.amf.decouple`, :mod:`kuiva.amf.pyscf_dhf` or anything else they depend on would
#: give a different :class:`~kuiva.amf.decouple.AtomicAMF` for an unchanged
#: :class:`~kuiva.amf.atomic.AtomicRequest`. Entries carrying a different value are ignored.
#:
#: ==  =========================================================================
#: 1   One-electron ``X``, no compensating term. (Never released with
#:     a persistent cache; recorded so the numbering means something.)
#: 2   ``X`` from the converged four-component Fock, plus
#:     ``h1e(X_2e) - h1e(X_1e)``. See :func:`kuiva.amf.decouple.x2c_decoupling`.
#: ==  =========================================================================
FORMULA_VERSION = 2

#: Environment override for the cache directory. ``off`` disables the cache.
ENV_CACHE_DIR = "KUIVA_AMF_CACHE"

#: Configuration-file key, in the ``[amf]`` section (sections are flattened).
CONFIG_KEY = "amf_cache_dir"

#: Values of either that mean "no persistent cache".
DISABLED = ("", "off", "none", "no", "false", "disabled", "0")

#: Attributes that must match between a stored entry and the request being served. Derived
#: from :class:`kuiva.amf.atomic.AtomicRequest` at call time rather than listed here, so a new
#: field cannot be forgotten — see point 2 of the module docstring.
_LIGHT_SPEED_NONE = "physical"

_ANNOUNCED = {"dir": False}


def _default_directory() -> Path:
    home = os.environ.get("XDG_CACHE_HOME")
    return (Path(home) if home else Path.home() / ".cache") / "kuiva" / "amf"


def cache_directory() -> Optional[Path]:
    """Where corrections are stored, or ``None`` when the cache is off.

    Resolution order, most specific first: ``$KUIVA_AMF_CACHE``, then ``cache_dir`` in the
    ``[amf]`` section of the ``defaults.conf`` configuration file, then ``~/.cache/kuiva/amf``.
    """
    value = os.environ.get(ENV_CACHE_DIR)
    if value is None:
        try:
            values, _ = resources.read_config()
        except resources.ConfigurationError:
            # ⚠ A malformed configuration file is a hard error for the memory limit, which
            # cannot proceed without one. The cache can, and refusing to run a calculation
            # because a *cache* setting could not be parsed would be the wrong trade.
            log.warning("cannot read the Kuiva configuration file; the persistent X2CAMF "
                        "cache is off for this run")
            return None
        value = values.get(CONFIG_KEY)
    if value is None:
        return _default_directory()
    if value.strip().lower() in DISABLED:
        return None
    return Path(os.path.expanduser(value.strip()))


def available() -> bool:
    """Whether the cache is configured on. Says nothing about whether it is writable."""
    return cache_directory() is not None


def _h5py():
    """``h5py``, or ``None``. Imported lazily so the cache is optional at import time."""
    try:
        import h5py
    except ImportError:                                                     # pragma: no cover
        log.debug("h5py is not installed; the persistent X2CAMF cache is off")
        return None
    return h5py


def _request_attributes(request) -> Dict[str, object]:
    """The request as plain attributes, one per field of ``AtomicRequest``.

    Built from ``dataclasses.fields`` so that a field added to the request appears here
    automatically and takes part in the validation of point 2 in the module docstring.
    """
    import dataclasses

    attrs: Dict[str, object] = {}
    for f in dataclasses.fields(request):
        value = getattr(request, f.name)
        if f.name == "configuration":
            value = value.canonical
        elif value is None:
            value = _LIGHT_SPEED_NONE
        elif isinstance(value, bool):
            value = int(value)
        attrs[f.name] = value
    return attrs


def entry_path(directory: Path, key: str) -> Path:
    return Path(directory) / (key + ".h5")


def load(request, key: str):
    """The stored :class:`~kuiva.amf.decouple.AtomicAMF` for ``request``, or ``None``.

    Never raises: every failure — no cache, no ``h5py``, no file, a stale
    :data:`FORMULA_VERSION`, a mismatched request, a corrupt file — is a miss.
    """
    from .decouple import AtomicAMF

    directory = cache_directory()
    if directory is None:
        return None
    h5py = _h5py()
    if h5py is None:
        return None
    path = entry_path(directory, key)
    if not path.is_file():
        return None

    try:
        with h5py.File(str(path), "r") as f:
            stored_schema = int(f.attrs.get("schema", -1))
            stored_formula = int(f.attrs.get("formula_version", -1))
            if stored_schema != SCHEMA or stored_formula != FORMULA_VERSION:
                log.debug("ignoring cache entry %s: schema %d/formula %d, expected %d/%d",
                          path.name, stored_schema, stored_formula, SCHEMA, FORMULA_VERSION)
                return None
            expected = _request_attributes(request)
            group = f["request"]
            for name, want in expected.items():
                got = group.attrs.get(name)
                if isinstance(got, bytes):
                    got = got.decode("utf-8")
                if isinstance(want, str):
                    matches = str(got) == want
                else:
                    matches = got is not None and float(got) == float(want)
                if not matches:
                    # ⚠ A warning, not a debug line. Two different requests reaching one
                    # filename means cache_key has stopped separating them, which is a defect
                    # in the key and not a normal miss.
                    log.warning("X2CAMF cache entry %s does not describe the request it was "
                                "found for (%s: stored %r, wanted %r). Recomputing, but this "
                                "means kuiva.amf.atomic.cache_key is not separating two "
                                "different atomic references.", path.name, name, got, want)
                    return None
            h_sf = np.array(f["correction/h_sf"])
            w = np.array(f["correction/w"])
            meta = dict(f["correction"].attrs)
    except Exception as exc:                                                # noqa: BLE001
        log.warning("cannot read the X2CAMF cache entry %s (%s: %s); recomputing",
                    path, type(exc).__name__, exc)
        return None

    return AtomicAMF(
        h_sf=h_sf, w=w, configuration=request.configuration,
        scale=float(meta.get("scale", 0.0)),
        tr_residual=float(meta.get("tr_residual", 0.0)),
        tr_residual_rel=float(meta.get("tr_residual_rel", 0.0)),
        transformed_scale=float(meta.get("transformed_scale", 0.0)),
        subtracted_scale=float(meta.get("subtracted_scale", 0.0)),
        compensation_scale=float(meta.get("compensation_scale", 0.0)))


def store(request, key: str, correction) -> bool:
    """Write ``correction`` to the cache. Returns whether it was written.

    Never raises, for the reason in point 3 of the module docstring: a cache that can fail a
    calculation is worse than no cache.
    """
    directory = cache_directory()
    if directory is None:
        return False
    h5py = _h5py()
    if h5py is None:
        return False

    gb = (correction.h_sf.nbytes + correction.w.nbytes) / resources.BYTES_PER_GB
    path = entry_path(directory, key)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        free = resources.scratch_free_gb(directory)
        if free is not None and gb > free:
            log.warning("not writing the X2CAMF cache entry for %s: it needs %.3f GB and %s "
                        "has %.3f GB free", request.element, gb, directory, free)
            return False
        # Atomic: a unique temporary in the same directory, then a rename. A job killed
        # between the two leaves the temporary behind and the cache intact.
        fd, tmp = tempfile.mkstemp(prefix=key + ".", suffix=".tmp", dir=str(directory))
        os.close(fd)
        try:
            with h5py.File(tmp, "w") as f:
                f.attrs["schema"] = SCHEMA
                f.attrs["formula_version"] = FORMULA_VERSION
                f.attrs["key"] = key
                f.attrs["writer"] = "kuiva {}".format(_kuiva_version())
                group = f.create_group("request")
                for name, value in _request_attributes(request).items():
                    group.attrs[name] = value
                correction_group = f.create_group("correction")
                correction_group.create_dataset("h_sf", data=np.asarray(correction.h_sf))
                correction_group.create_dataset("w", data=np.asarray(correction.w))
                for name in ("scale", "tr_residual", "tr_residual_rel", "transformed_scale",
                             "subtracted_scale", "compensation_scale"):
                    correction_group.attrs[name] = float(getattr(correction, name))
            os.replace(tmp, str(path))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except Exception as exc:                                                # noqa: BLE001
        log.warning("cannot write the X2CAMF cache entry %s (%s: %s); continuing without it",
                    path, type(exc).__name__, exc)
        return False

    log.debug("wrote X2CAMF cache entry %s (%.1f MB)", path.name, gb * 1024.0)
    return True


def announce() -> None:
    """Log the cache directory once per process, at DEBUG.

    ⚠ Not at INFO: INFO is the output file by design, and a cache path is a diagnostic, not a
    result. Where the correction *is* the subject — the atomic correction's own output block —
    :func:`kuiva.amf.atomic.atomic_correction` reports it through the output grammar instead.
    """
    if _ANNOUNCED["dir"]:
        return
    _ANNOUNCED["dir"] = True
    directory = cache_directory()
    if directory is None:
        log.debug("the persistent X2CAMF cache is off ($%s or [amf] %s)",
                  ENV_CACHE_DIR, CONFIG_KEY)
    else:
        log.debug("persistent X2CAMF cache: %s", directory)


def _kuiva_version() -> str:
    try:
        from .. import __version__
    except ImportError:                                                     # pragma: no cover
        return "unknown"
    return str(__version__)


def describe() -> str:
    """One line for a report: where entries go, or why there are none."""
    directory = cache_directory()
    if directory is None:
        return "off"
    if _h5py() is None:
        return "off (h5py is not installed)"
    return str(directory)


def purge(directory: Optional[Path] = None) -> int:
    """Delete every entry, returning how many were removed.

    Provided because :data:`FORMULA_VERSION` makes stale entries *inert* rather than absent —
    they stop being read but keep occupying disk — so there has to be a supported way to
    reclaim it that is not ``rm -rf`` on a path the user has to reconstruct.
    """
    target = Path(directory) if directory is not None else cache_directory()
    if target is None or not target.is_dir():
        return 0
    removed = 0
    for path in sorted(target.glob("*.h5")):
        try:
            path.unlink()
            removed += 1
        except OSError as exc:                                              # pragma: no cover
            log.warning("cannot remove %s: %s", path, exc)
    return removed


__all__ = ["SCHEMA", "FORMULA_VERSION", "ENV_CACHE_DIR", "CONFIG_KEY", "available",
           "cache_directory", "describe", "entry_path", "load", "purge", "store"]
