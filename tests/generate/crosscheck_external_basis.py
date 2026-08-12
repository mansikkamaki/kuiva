"""Cross-check Kuiva's basis data against DIRAC (Dyall) and OpenMolcas (ANO-RCC).

The Karlsruhe x2c and Peterson cc-pVnZ-X2C families are absent from OpenMolcas/DIRAC, so the
meaningful "did we get it right?" comparison is on the families they *do* ship natively:
  * **Dyall** — native to DIRAC. We compare PySCF's ``dyall*`` primitive exponents against
    DIRAC's ``external/install/dirac/share/dirac/basis/dyall.*`` files.
  * **ANO-RCC** — native to OpenMolcas. We compare PySCF's ``anorcc`` primitive exponents
    against OpenMolcas's ``external/install/openmolcas/basis_library/ANO-RCC``.

Both are all-electron relativistic sets, so agreement of the *primitive exponents* (per
angular momentum) confirms Kuiva/PySCF is using the same published basis. This is a Tier-2
generation step: it needs the external installs, runs once, and writes a
committed reference file (``tests/reference/basis_crosscheck.json``) that the fast test suite
checks against without needing DIRAC/OpenMolcas.

Run:  python tests/generate/crosscheck_external_basis.py
(with ``external/env.sh`` sourced so DIRAC/OpenMolcas paths and PySCF are available).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
DIRAC_BASIS = REPO / "external/install/dirac/share/dirac/basis"
MOLCAS_ANORCC = REPO / "external/install/openmolcas/basis_library/ANO-RCC"
REF_OUT = REPO / "tests/reference/basis_crosscheck.json"

_LMAP = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}
RELTOL = 1e-6           # exponents are printed to ~8-9 sig figs in both sources


# --- parsers ---------------------------------------------------------------------------
def parse_dirac_dyall(path: Path, symbol: str) -> Dict[int, List[float]]:
    """Extract {l: [exponents]} for one element from a DIRAC Dyall basis file."""
    lines = path.read_text().splitlines()
    # locate the element block: a comment line '$ <Sym>' then 'a <Z>' ... until next 'a <Z>'.
    start = None
    for i, ln in enumerate(lines):
        if re.match(rf"^\$\s+{re.escape(symbol)}\s*$", ln):
            start = i
            break
    if start is None:
        raise KeyError(f"{symbol} not found in {path.name}")
    exps: Dict[int, List[float]] = {}
    i = start + 1
    while i < len(lines):
        ln = lines[i]
        if re.match(r"^\$\s+[A-Z][a-z]?\s*$", ln) and i > start + 1:
            break  # next element
        m = re.match(r"^\$\s+([spdfghi])\s+functions", ln)
        if m:
            l = _LMAP[m.group(1)]
            hdr = lines[i + 1].split()
            nprim = int(hdr[0])
            vals = [float(lines[i + 2 + k]) for k in range(nprim)]
            exps.setdefault(l, []).extend(vals)
            i += 2 + nprim
            continue
        i += 1
    return {l: sorted(v) for l, v in exps.items()}


def parse_molcas_anorcc(path: Path, symbol: str) -> Dict[int, List[float]]:
    """Extract {l: [primitive exponents]} for one element from OpenMolcas ANO-RCC."""
    text = path.read_text()
    # element block: '/<Sym>.ANO-rcc...' until the next '/' header line.
    blocks = re.split(r"(?m)^/", text)
    block = None
    for b in blocks:
        if re.match(rf"^{re.escape(symbol)}\.ANO-rcc", b):
            block = b
            break
    if block is None:
        raise KeyError(f"{symbol} not found in {path.name}")
    lines = block.splitlines()
    exps: Dict[int, List[float]] = {}
    i = 0
    while i < len(lines):
        m = re.match(r"^\*\s*([spdfghi])-type functions", lines[i], re.IGNORECASE)
        if m:
            l = _LMAP[m.group(1).lower()]
            # next non-comment line is 'nprim ncontr'
            j = i + 1
            while lines[j].lstrip().startswith("*"):
                j += 1
            nprim = int(lines[j].split()[0])
            vals = [float(lines[j + 1 + k].split()[0]) for k in range(nprim)]
            exps[l] = sorted(vals)
            i = j + 1 + nprim
            continue
        i += 1
    return exps


def pyscf_exponents(basis_name: str, symbol: str) -> Dict[int, List[float]]:
    """{l: [primitive exponents]} for a PySCF-loadable basis (dedup within l)."""
    from pyscf import gto
    shells = gto.basis.load(basis_name, symbol)
    exps: Dict[int, set] = {}
    for sh in shells:
        l = sh[0]
        for prim in sh[1:]:
            # Relativistic shells may carry a leading integer 'kappa' label; skip it.
            if not isinstance(prim, (list, tuple)):
                continue
            exps.setdefault(l, set()).add(round(float(prim[0]), 10))
    return {l: sorted(v) for l, v in exps.items()}


# --- comparison ------------------------------------------------------------------------
def compare(ref: Dict[int, List[float]], got: Dict[int, List[float]]) -> Dict:
    ok = True
    detail = {}
    for l in sorted(set(ref) | set(got)):
        a, b = ref.get(l, []), got.get(l, [])
        matched = len(a) == len(b) and all(
            abs(x - y) <= RELTOL * max(abs(x), abs(y)) for x, y in zip(a, b))
        detail[l] = {"n_ref": len(a), "n_got": len(b), "match": matched}
        ok = ok and matched
    return {"ok": ok, "per_l": detail}


def main() -> int:
    results = {"reltol": RELTOL, "dyall": {}, "anorcc": {}, "all_ok": True}

    # Dyall vs DIRAC (map pyscf name <-> DIRAC file name).
    # ⚠ ``dyallv2z`` carries the four-component atomic references of
    # ``tests/generate/x2camf_dirac.py``. A cross-code number is only worth something if the
    # two programs used the *same functions*, and "dyall.v2z" (DIRAC) vs "dyallv2z" (PySCF) is
    # exactly the name-vs-content distinction this file exists to settle — so the atoms of
    # that reference set are checked here before anything is compared.
    # ⚠ The open-shell ions are in this list for the same reason
    # the closed-shell atoms are: a cross-code number is worth nothing unless the two programs
    # used the *same functions*, and that is checked on the primitive exponents, never on the
    # basis-set name.
    dyall_cases = [("dyallv2z", "dyall.v2z",
                    ["C", "O", "Ne", "Ar", "Kr", "Xe", "Ti", "Ce", "Dy", "Yb", "Bi"]),
                   ("dyallv3z", "dyall.v3z", ["Kr", "Xe", "Rn"]),
                   ("dyallcv3z", "dyall.cv3z", ["Xe"])]
    for pyname, fname, syms in dyall_cases:
        f = DIRAC_BASIS / fname
        for s in syms:
            try:
                ref = parse_dirac_dyall(f, s)
                got = pyscf_exponents(pyname, s)
                cmp = compare(ref, got)
            except Exception as e:  # noqa: BLE001
                cmp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            results["dyall"][f"{pyname}/{s}"] = cmp
            results["all_ok"] &= cmp["ok"]
            print(f"[Dyall ] {pyname:10s} {s:3s} vs {fname:12s}: "
                  f"{'OK' if cmp['ok'] else 'MISMATCH'}")

    # ANO-RCC vs OpenMolcas.
    for s in ["H", "C", "O", "Fe", "Zn"]:
        try:
            ref = parse_molcas_anorcc(MOLCAS_ANORCC, s)
            got = pyscf_exponents("anorcc", s)
            cmp = compare(ref, got)
        except Exception as e:  # noqa: BLE001
            cmp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        results["anorcc"][f"anorcc/{s}"] = cmp
        results["all_ok"] &= cmp["ok"]
        print(f"[ANO-RCC] anorcc     {s:3s} vs OpenMolcas ANO-RCC: "
              f"{'OK' if cmp['ok'] else 'MISMATCH'}")

    REF_OUT.parent.mkdir(parents=True, exist_ok=True)
    REF_OUT.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nall_ok = {results['all_ok']}  ->  wrote {REF_OUT.relative_to(REPO)}")
    return 0 if results["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
