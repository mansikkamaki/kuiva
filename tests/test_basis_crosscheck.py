"""Basis cross-check tests.

Fast test: the committed reference (``tests/reference/basis_crosscheck.json``, produced by
``tests/generate/crosscheck_external_basis.py``) records that PySCF's Dyall and ANO-RCC
primitive exponents match DIRAC and OpenMolcas exactly. We assert that recorded evidence and
re-verify a couple of exponents live from PySCF so the check is not purely trusting the file.

External test (``requires_external``): regenerate the comparison against the actual
DIRAC/OpenMolcas installs. Skipped automatically when those are absent (Tier-2 only).
"""
import json
import sys
from pathlib import Path

import pytest

from conftest import requires_external

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "tests/reference/basis_crosscheck.json"


def test_committed_crosscheck_all_ok():
    assert REF.is_file(), "run tests/generate/crosscheck_external_basis.py to create it"
    data = json.loads(REF.read_text())
    assert data["all_ok"] is True
    # structure: both providers present with per-case results
    assert data["dyall"] and data["anorcc"]
    for case in {**data["dyall"], **data["anorcc"]}.values():
        assert case.get("ok") is True


def test_pyscf_exponents_are_stable():
    # Independent, install-free sanity: known ANO-RCC H s-exponents (OpenMolcas values).
    sys.path.insert(0, str(REPO / "tests/generate"))
    from crosscheck_external_basis import pyscf_exponents
    h_s = pyscf_exponents("anorcc", "H")[0]
    assert len(h_s) == 8
    assert abs(max(h_s) - 188.61445) < 1e-3       # largest H s-exponent (ANO-RCC)
    assert abs(min(h_s) - 0.027962) < 1e-5        # smallest


@requires_external
def test_regenerate_crosscheck_matches():
    sys.path.insert(0, str(REPO / "tests/generate"))
    import importlib
    import crosscheck_external_basis as X
    importlib.reload(X)
    assert X.main() == 0                           # 0 == all comparisons OK
