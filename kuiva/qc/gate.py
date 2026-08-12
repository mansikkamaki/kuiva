"""The import gate for every quantum-computing framework Kuiva ever talks to.

One module, so the gate is a single behaviour rather than a convention repeated in each
adapter. Its job is small and entirely about failure: to answer *is this framework here?*
honestly, and, when it is not, to raise an error that says what to run.

Why it is a hard rule rather than tidiness
------------------------------------------
:mod:`kuiva.qc` is pure NumPy/SciPy — the qubit mapping, the circuit representation, the
ansatz builders and the sampled-subspace driver all are, by design. A
framework appears only inside an adapter under :mod:`kuiva.qc.backends`, imported **inside the
function that needs it**, never at module or package import time. That is what makes
``import kuiva.qc`` succeed on a machine with nothing installed, which in turn is what keeps
the default ``pytest`` run independent of ``external/venv_qc``.

The precedent is :mod:`kuiva.amf.x2camf_plugin`, and the reason is the same one the X2CAMF plugin gate gives: an eager import turns an optional research tool into a runtime
requirement, silently, the first time someone adds a convenience re-export.

⚠ The gate is deliberately not a *fallback*. It never emulates a missing framework or
degrades to something that still returns a number — the project's refusal culture
applies here as everywhere: an unavailable backend is refused with a message naming both the
package and the script that builds it. Kuiva's own stub backend is a **declared**
implementation of the backend protocol, not what happens when Qiskit is missing.
"""
from __future__ import annotations

from typing import Optional

#: The bootstrap script that provisions every framework this gate guards.
BOOTSTRAP_SCRIPT = "scripts/bootstrap/90_qiskit.sh"

#: Where it installs them. ⚠ A **different interpreter** from ``external/venv``: Qiskit
#: dropped Python 3.9 and IQM's provider requires >= 3.10, so this venv is not the pinned
#: baseline one and cannot be put on the baseline interpreter's ``PYTHONPATH``. Run the
#: qc-marked tests *under* it, with the repository root on ``PYTHONPATH``.
QC_VENV = "external/venv_qc"


def available(package: str) -> bool:
    """Whether ``package`` can be imported. The only honest way to ask.

    Catches ``Exception``, not ``ImportError``: a half-built extension module raises all
    sorts of things on import, and every one of them means the same thing here.
    """
    import importlib

    try:
        importlib.import_module(package)
    except Exception:                                                       # noqa: BLE001
        return False
    return True


def require(package: str, purpose: Optional[str] = None):
    """Import ``package`` or raise an :class:`ImportError` that says what to run.

    Parameters
    ----------
    package : str
        The top-level module name, e.g. ``"qiskit"`` or ``"qiskit_aer"``.
    purpose : str, optional
        What the caller wanted it for, in a few words ("the Aer backend adapter"). It is put
        in the message because the useful question at that moment is not *which package is
        missing* but *which part of the calculation stopped*.
    """
    import importlib

    try:
        return importlib.import_module(package)
    except Exception as exc:                                                # noqa: BLE001
        raise ImportError(
            "{!r} is not installed{}. It is an optional, research-only dependency of "
            "kuiva.qc ({}: {}); build it with `bash {}`, which creates {} — then run under "
            "that interpreter with the repository root on PYTHONPATH. Nothing else in Kuiva "
            "needs it.".format(package, "" if purpose is None else " (needed for " + purpose + ")",
                               type(exc).__name__, exc, BOOTSTRAP_SCRIPT, QC_VENV))


__all__ = ["BOOTSTRAP_SCRIPT", "QC_VENV", "available", "require"]
