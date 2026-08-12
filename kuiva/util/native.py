"""The import gate for the compiled kernel backend.

**Exactly one Python module touches ``import kuiva._native``, and this is it** — the same
one-gate rule :mod:`kuiva.qc.gate` established for quantum-computing frameworks, and for
the same reason: an eager import anywhere else turns an optional compiled artifact into a
runtime requirement the first time someone adds a convenience re-export. Everything the
extension exports is registered here under backend name ``"native"`` in the
:mod:`kuiva.ci.kernels` registry; no caller and no test changes when it appears, which is
the whole point of the registry.

The switch: ``KUIVA_KERNELS``
-----------------------------
One environment variable, read once, call-site override via
:func:`kuiva.ci.kernels.set_preferred_backend`:

``auto`` (default)
    Use ``"native"`` where registered, ``"numpy"`` otherwise. A missing build logs one
    DEBUG line and is not an event: the default must run on a laptop with no compiler.
``numpy``
    Pure NumPy, even if the build exists — the reproducibility / debugging setting, and
    the reference implementation staying alive as a first-class way to run.
``native``
    **Refuse** (raise, naming the build script) if the extension is missing or its
    ``API_VERSION`` does not match. An explicit request is refused, never silently
    degraded (the project's gate philosophy).

Loading is lazy: :func:`kuiva.ci.kernels.spec` triggers :func:`activate` on first
resolution, so ``import kuiva`` stays instant and side-effect-free with or without the
build. A refused ``native`` request re-raises on *every* resolution — a run must never
proceed on a backend nobody asked for.

Provenance
----------
The extension exports a ``BUILD_ID`` (source hash + compiler version, baked in by the
``cpp/`` build) because a source-hash fingerprint cannot see a
``.so``. :func:`fingerprint_token` is what the test-stage fingerprints and the
checkpoint metadata record, and :func:`banner_entry` is the banner's one-line
statement of which backend ran — printed only when the state differs from the pure-NumPy
default, so a default run's output (and every committed example reference) is unchanged
by the backend's existence: an unmarked output *is* a pure-NumPy output.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ..ci import kernels
from .logging import get_logger

log = get_logger(__name__)

#: How to build the extension, named in every refusal. Two routes to one build: ``cpp/`` is
#: the build itself and works anywhere oneAPI is installed, while the bootstrap script is the
#: development sandbox's wrapper around it (it adds the pinned venv and pybind11). A refusal
#: names both, because the reader may be either kind of user.
BUILD_COMMAND = "cd cpp && ./configure && make"
BOOTSTRAP_SCRIPT = "scripts/bootstrap/95_native.sh"

#: The interface version this gate understands. Must equal ``kuiva._native.API_VERSION``;
#: a mismatch means the .so predates a kernel-signature change and may not register.
#: History: 1 = Stage 1 (probe, connections_scan, block_pair_gemm); 2 = + sparse_pair_dot.
API_VERSION = 2

VALID_MODES = ("auto", "numpy", "native")

_STATE = {"activated": False, "module": None, "error": None, "mode": None}


def mode() -> str:
    """The requested backend mode: ``KUIVA_KERNELS``, validated, default ``auto``."""
    value = os.environ.get("KUIVA_KERNELS", "auto").strip().lower()
    if value not in VALID_MODES:
        raise ValueError("KUIVA_KERNELS={!r} is not one of {}".format(value, VALID_MODES))
    return value


def _import_extension():
    import importlib

    module = importlib.import_module("kuiva._native")
    version = int(getattr(module, "API_VERSION", -1))
    if version != API_VERSION:
        raise ImportError(
            "kuiva._native has API_VERSION {} but this source tree expects {}: the "
            "extension is stale. Rebuild it with `{}` (in the development sandbox, "
            "`bash {}`).".format(version, API_VERSION, BUILD_COMMAND, BOOTSTRAP_SCRIPT))
    return module


def activate() -> None:
    """Resolve ``KUIVA_KERNELS`` once and register the native backend accordingly.

    Idempotent; called lazily by :func:`kuiva.ci.kernels.spec` on first resolution. A
    failed explicit request (``native``) is remembered and re-raised on every call.
    """
    if _STATE["activated"]:
        if _STATE["error"] is not None:
            raise _STATE["error"]
        return
    requested = mode()
    _STATE["mode"] = requested
    if requested == "numpy":
        _STATE["activated"] = True
        log.debug("KUIVA_KERNELS=numpy: compiled backend not loaded by request")
        return
    try:
        module = _import_extension()
    except Exception as exc:                                                # noqa: BLE001
        if requested == "native":
            error = ImportError(
                "KUIVA_KERNELS=native, but the compiled backend could not be loaded "
                "({}: {}). Build it with `{}` (in the development sandbox, `bash {}`); "
                "or unset KUIVA_KERNELS to run the pure-NumPy reference path."
                .format(type(exc).__name__, exc, BUILD_COMMAND, BOOTSTRAP_SCRIPT))
            _STATE["activated"] = True
            _STATE["error"] = error
            raise error
        _STATE["activated"] = True
        log.debug("compiled kernel backend absent (%s); running pure NumPy. Build it "
                  "with `%s` (sandbox: `bash %s`) if wanted.", exc, BUILD_COMMAND,
                  BOOTSTRAP_SCRIPT)
        return
    _register(module)
    _STATE["module"] = module
    _STATE["activated"] = True
    kernels.set_preferred_backend("native")
    log.debug("native kernel backend registered (build %s)", module.BUILD_ID)


def available() -> bool:
    """Whether the native backend is loaded and registered (never raises)."""
    return _STATE["module"] is not None


def build_id() -> Optional[str]:
    """The loaded extension's build id, or ``None`` without one."""
    module = _STATE["module"]
    return None if module is None else str(module.BUILD_ID)


def fingerprint_token() -> str:
    """The backend provenance tuple, rendered for fingerprints and checkpoint metadata.

    ``numpy`` for the pure-NumPy path, ``native:<build id>`` for the compiled one. This
    must be recorded wherever sources are fingerprinted (tests/stages.py, io/checkpoint):
    a source hash cannot see a ``.so``, and a threaded reduction is only 1e-13-equal —
    replaying a NumPy-epoch artifact into a native-epoch run could mask a parity defect.
    Never raises: an unloadable explicit request renders as ``native:unavailable`` (the
    run itself will have refused long before any fingerprint is compared).
    """
    try:
        activate()
    except Exception:                                                       # noqa: BLE001
        return "native:unavailable"
    if kernels.preferred_backend() == "native" and available():
        return "native:{}".format(build_id())
    return "numpy"


def banner_entry() -> Optional[str]:
    """The run banner's backend line, or ``None`` when the pure-NumPy default ran.

    ``None`` in the default state is a decision, not an omission: every committed example
    reference and every build-less clone prints exactly what it printed before this
    backend existed, so an unmarked output is *defined* to be a pure-NumPy output. A
    non-default state — the compiled backend active, or NumPy forced explicitly — must
    announce itself at the point of use, the same announce-at-selection rule applied to non-production
    Hamiltonians. A refused explicit ``native`` request raises here exactly as it would
    at the first kernel resolution.
    """
    activate()
    if kernels.preferred_backend() == "native" and available():
        return "compiled kernels: native (build {})".format(build_id())
    if _STATE["mode"] == "numpy":
        return "compiled kernels: disabled (KUIVA_KERNELS=numpy)"
    return None


def _reset_for_testing() -> None:
    """Forget the activation state (testing hook; pairs with a fresh KUIVA_KERNELS).

    Also rewinds the registry's gate flag and preferred backend to the import-time state.
    Registered native specs stay registered (the registry refuses only a *different* impl
    under the same name, and re-activation re-registers the same wrapper objects).
    """
    _STATE.update({"activated": False, "module": None, "error": None, "mode": None})
    kernels._GATE["done"] = False
    kernels.set_preferred_backend(kernels.DEFAULT_BACKEND)


# --- The registered wrappers ----------------------------------------------------------
#
# pybind11 functions carry no introspectable Python signature, and the kernel-portability
# contract (tests/test_kernel_contracts.py) is enforced through `inspect.signature` — so
# each native kernel is registered through a thin annotated wrapper. The wrappers do no
# checking (the C++ boundary asserts dtype, layout and aliasing with the same messages as
# the NumPy reference) and add one Python call per kernel invocation, far below every
# measured per-call cost the port exists to remove.

def _native_probe(x: np.ndarray, out: np.ndarray) -> np.ndarray:
    """``out[i] = x[i] + 1`` — the trivial kernel that proves the gate/registry machinery.

    Reduction order (B10): none — elementwise, no reduction of any kind.
    """
    return _STATE["module"].native_probe(x, out)


def _connections_scan(masks: np.ndarray, row_start: int, row_stop: int,
                      s_i: np.ndarray, s_j: np.ndarray, s_from: np.ndarray,
                      s_to: np.ndarray, s_phase: np.ndarray,
                      d_i: np.ndarray, d_j: np.ndarray, d_from: np.ndarray,
                      d_to: np.ndarray, d_phase: np.ndarray, n_threads: int):
    """Compiled determinant connection scan; contract as the NumPy kernel of this name.

    Reduction order (B10): none — the outputs are integer index arrays and +-1 phases,
    pure per-pair functions with no floating-point reduction anywhere. The threaded scan
    partitions rows into contiguous ascending chunks and concatenates per-thread buffers
    in chunk order, so it is **bitwise** identical to the serial scan at any thread count
    (anything else here is a defect, not a tolerance case).
    """
    return _STATE["module"].connections_scan(masks, row_start, row_stop, s_i, s_j,
                                             s_from, s_to, s_phase, d_i, d_j, d_from,
                                             d_to, d_phase, n_threads)


def _block_pair_gemm(a_data: np.ndarray, a_offset: np.ndarray, b_data: np.ndarray,
                     b_offset: np.ndarray, pairs: np.ndarray, dims: np.ndarray,
                     out_data: np.ndarray, out_offset: np.ndarray,
                     n_threads: int) -> np.ndarray:
    """Compiled block-pair GEMM driver; contract as the NumPy kernel of this name.

    Reduction order (B10): the accumulation into an output block runs in pair-table
    order, and several pairs may target the same block. This backend preserves that order
    at every thread count by **owner-computes** — each output block belongs to exactly one
    thread, which processes its pairs in table order — and dispatches the same
    ``cblas_zgemm`` per pair (a beta=1 pair is a beta=0 GEMM into scratch plus an
    elementwise add, exactly as the NumPy kernel's ``om += am.dot(bm)`` does it). ⚠ The
    one reduction this port cannot pin is **MKL's own, inside a single zgemm**: its order
    depends on the BLAS thread width, and inside the parallel region MKL is clamped to
    one thread while the NumPy reference runs it ambient. Measured consequence: at any
    *fixed* MKL width every thread count here is bitwise vs NumPy; across widths, blocks
    large enough for MKL to partition (k of order 32 up) differ by ~2e-16 relative —
    within the fixed 1e-13 band, and a difference pure NumPy exhibits between the same two
    MKL widths, i.e. not introduced by the port. A work-split *within* one output block
    would be a genuine reorder and is deliberately not done.
    """
    return _STATE["module"].block_pair_gemm(a_data, a_offset, b_data, b_offset, pairs,
                                            dims, out_data, out_offset, n_threads)


def _sparse_pair_dot(am_data: np.ndarray, am_offset: np.ndarray, csr_values: np.ndarray,
                     csr_indices: np.ndarray, csr_indptr: np.ndarray,
                     csr_meta: np.ndarray, pairs: np.ndarray, dims: np.ndarray,
                     out_data: np.ndarray, out_offset: np.ndarray,
                     n_threads: int) -> np.ndarray:
    """Compiled sparse-W application; contract as the NumPy kernel of this name.

    Reduction order (B10): for one output element the sum runs pairs in table order and,
    within a pair, the CSR row's entries in ascending stored order into a zeroed per-pair
    scratch, added to the output once per pair — the order the NumPy kernel's
    ``csr_matvecs`` engine defines. This backend preserves it at every thread count by
    **owner-computes** over output blocks (block ``io`` belongs to thread ``io % T``,
    pairs walked in table order) and reproduces the scratch and the naive complex product
    with FP contraction off, so parity is **bitwise**, serial and threaded — asserted
    against the running SciPy build by ``tests/test_native_backend.py``. No BLAS runs
    inside, so there is no MKL-width term (unlike ``block_pair_gemm``).
    """
    return _STATE["module"].sparse_pair_dot(am_data, am_offset, csr_values, csr_indices,
                                            csr_indptr, csr_meta, pairs, dims, out_data,
                                            out_offset, n_threads)


def _register(module) -> None:
    for name, impl in (("native_probe", _native_probe),
                       ("connections_scan", _connections_scan),
                       ("block_pair_gemm", _block_pair_gemm),
                       ("sparse_pair_dot", _sparse_pair_dot)):
        if not hasattr(module, name):                          # pragma: no cover - defensive
            raise ImportError("kuiva._native (build {}) does not export {!r}; rebuild "
                              "with `{}` (sandbox: `bash {}`)"
                              .format(module.BUILD_ID, name, BUILD_COMMAND,
                                      BOOTSTRAP_SCRIPT))
        kernels.register(name, "native", impl)


# The NumPy reference implementation of the probe, registered at import so the gate and
# parity machinery are exercisable with or without a build.
@kernels.kernel("native_probe")
def native_probe_numpy(x: np.ndarray, out: np.ndarray) -> np.ndarray:
    """``out[i] = x[i] + 1`` — NumPy reference for the probe kernel.

    Reduction order (B10): none — elementwise, no reduction of any kind.
    """
    if x.dtype != np.float64 or out.dtype != np.float64:
        raise TypeError("x and out must be float64")
    if not (x.flags.c_contiguous and out.flags.c_contiguous):
        raise ValueError("x and out must be C-contiguous")
    if np.shares_memory(out, x):
        raise ValueError("the output buffer may not alias an operand")
    np.add(x, 1.0, out=out)
    return out


__all__ = ["BUILD_COMMAND", "BOOTSTRAP_SCRIPT", "API_VERSION", "VALID_MODES", "mode", "activate",
           "available", "build_id", "fingerprint_token", "banner_entry"]
