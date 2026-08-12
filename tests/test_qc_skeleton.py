"""The boundary :mod:`kuiva.qc` is defined by (asserted from the sources).

Stage 0 adds no algorithm, so there is no number to check. What there *is* to check is the
thing that would be expensive to discover later: that the package imports on a machine with no
quantum framework installed, and that neither direction of the dependency rule has been
violated. Both are asserted **from the sources**, because both fail silently — an eager
framework import at the top of an adapter works perfectly on the machine that added it, and a
back-edge into ``ci/`` or ``mcscf/`` is invisible until the day someone tries to run Kuiva
without ``external/venv_qc``.

These tests run in the default suite and must never need the qc stack. The ones that do carry
the ``qc`` marker and arrive with Stage 2.
"""
import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
QC_DIR = REPO / "kuiva" / "qc"

#: Everything the no-framework rule keeps out of the algorithm layer. Adding an adapter for a
#: new stack means adding its top-level name here, not making an exception to the rule.
FRAMEWORKS = ("qiskit", "qiskit_aer", "qiskit_ibm_runtime", "iqm", "cirq", "pennylane",
              "openfermion", "ffsim", "braket", "pytket")


def _module_files(package_dir):
    """Every ``.py`` in the package, adapters included."""
    return sorted(p for p in package_dir.rglob("*.py") if "__pycache__" not in p.parts)


def _top_level_imports(tree):
    """Module names imported at **module scope** — i.e. at import time, not inside a call."""
    names = []
    for node in tree.body:                     # body only: nested = inside a def/class/try
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
        elif isinstance(node, ast.Try):        # a top-level try/except ImportError is still
            for sub in node.body:              # an import at import time
                if isinstance(sub, ast.Import):
                    names += [alias.name for alias in sub.names]
                elif isinstance(sub, ast.ImportFrom) and sub.level == 0 and sub.module:
                    names.append(sub.module)
    return names


def test_kuiva_qc_imports_with_nothing_installed():
    """The Stage-0 exit criterion. If this fails the default suite has grown a dependency on
    ``external/venv_qc``, which is exactly what the package layout exists to prevent."""
    import kuiva.qc

    assert kuiva.qc.BOOTSTRAP_SCRIPT == "scripts/bootstrap/90_qiskit.sh"
    assert (REPO / kuiva.qc.BOOTSTRAP_SCRIPT).is_file(), "the gate names a script that is gone"


def test_no_framework_is_imported_at_import_time():
    """⚠ The no-framework-import rule, asserted where it can actually fail: at **module scope**.

    A framework import inside the function that needs it is the contract; the same import one
    indentation level out passes every test on a developer's machine and breaks ``import
    kuiva`` on everyone else's.
    """
    offenders = []
    for path in _module_files(QC_DIR):
        for name in _top_level_imports(ast.parse(path.read_text())):
            if name.split(".")[0] in FRAMEWORKS:
                offenders.append((path.relative_to(REPO).as_posix(), name))
    assert offenders == [], (
        "kuiva.qc must import no quantum framework at module scope; move these inside the "
        "function that needs them, behind kuiva.qc.gate.require: {}".format(offenders))


def test_the_dependency_runs_one_way():
    """⚠ The one-way dependency rule: ``kuiva.qc`` may be imported by nothing in the calculation path.

    The mirror of ``test_x2c_decouple.py::test_the_dependency_runs_one_way``. A back-edge —
    even a convenience re-export in a ``__init__`` — would make a quantum framework reachable
    from the classical CI path and put the whole of the package's isolation at the mercy of an import
    ordering.
    """
    offenders = []
    for package in ("ci", "mcscf", "rdm", "x2c", "dmrg", "amf", "integrals", "interface",
                    "props", "spinor", "orth", "basis", "io", "util"):
        for path in _module_files(REPO / "kuiva" / package):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[:2] == ["kuiva", "qc"]:
                            offenders.append((path.relative_to(REPO).as_posix(), alias.name))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    absolute = node.level == 0 and module.startswith("kuiva.qc")
                    relative = node.level > 0 and module.split(".")[0] == "qc"
                    if absolute or relative:
                        offenders.append((path.relative_to(REPO).as_posix(), module))
    assert offenders == [], "nothing in the calculation path may import kuiva.qc: {}".format(
        offenders)


def test_require_refuses_and_says_what_to_run():
    """⚠ A missing framework is refused, never emulated (the refuse-never-reconcile culture of the method surface).

    The message is the whole feature: the useful error here names the package, what it was
    wanted for, the script that builds it and the venv it lands in — because the person who
    hits it is, by definition, on a machine where none of that is set up.
    """
    from kuiva.qc import gate

    assert gate.available("numpy") is True
    assert gate.available("kuiva_no_such_framework") is False

    with pytest.raises(ImportError) as excinfo:
        gate.require("kuiva_no_such_framework", purpose="the Aer backend adapter")
    message = str(excinfo.value)
    for expected in ("kuiva_no_such_framework", "the Aer backend adapter",
                     gate.BOOTSTRAP_SCRIPT, gate.QC_VENV):
        assert expected in message, (expected, message)


def test_the_qc_marker_is_registered():
    """Stage 2's tests are opt-in by marker, so the marker has to exist before they do — an
    unregistered one is a warning, not an error, and the tests would silently run in the
    default suite against a venv that need not be there."""
    import re

    text = (REPO / "pyproject.toml").read_text()
    assert re.search(r'^\s*"qc:', text, re.MULTILINE), \
        "pyproject.toml [tool.pytest.ini_options] markers must register 'qc'"
