"""Sizing arithmetic and overlap spectra: which array stops a calculation, and where a
linear-dependence cut falls.

⚠ **A study generator, not a reference generator.** Nothing here produces a committed
reference file; it produces the numbers behind a design decision, which live in the owning
package's local validation notes. Written to the bounded-run discipline: every part is bounded by a hard wall budget, writes its records
incrementally, and terminates on a fixed work plan rather than on convergence.

Two parts, selected with ``--part``:

``sizes``
    Pure arithmetic over the exact sizing functions — no SCF, no integrals, nothing built.
    Answers three questions at once: the AO count at which the conventional ERI array is
    refused against a configured limit and which *other* array binds first (the
    integral-direct Cholesky question), the spinor count at which the resident sigma
    intermediates are refused (the conventional-CI ceiling), and the rank-4 RDM size at the
    active spaces the Tier-3 systems would need (the multi-site NEVPT2 question).

``overlap``
    The overlap-eigenvalue spectrum near the linear-dependence threshold, for the two places
    the code cuts one: :mod:`kuiva.orth.canonical` on the molecule's **contracted** AO
    overlap, and :func:`kuiva.x2c.decouple.canonical_orth` on the **decontracted**
    four-component metric. The reported quantity is not the count of dropped functions but
    whether the cut falls *inside* a numerically degenerate group — which is the failure
    the degenerate-group rule forbids and neither cut currently prevents.

Usage::

    python tests/generate/spec_gaps_study.py --part sizes
    python tests/generate/spec_gaps_study.py --part overlap [--budget 540]

Records: ``temp/spec_gaps_<part>.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TEMP = Path(__file__).resolve().parents[2] / "temp"


def _write(path: Path, payload) -> None:
    """Incremental write: every record survives a kill."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=float)


# =========================================================================================
# Part 1 — sizing arithmetic (items 4, 5, 7)
# =========================================================================================

def _refusal_dimension(size_fn, limit_gb: float, hi: int = 4096) -> int:
    """Largest ``n`` with ``size_fn(n) <= limit_gb`` (monotone ``size_fn``)."""
    lo = 1
    if size_fn(lo) > limit_gb:
        return 0
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if size_fn(mid) <= limit_gb:
            lo = mid
        else:
            hi = mid
    return lo


def part_sizes(limits: Sequence[float]) -> dict:
    from kuiva.interface.pyscf_bridge import CHOLESKY_VECTORS_PER_AO, eri_memory_gb
    from kuiva.integrals.transform import factor_memory_gb, mo_block_memory_gb
    from kuiva.util import resources as res
    from kuiva.ci.sigma import sigma_workspace_gb

    import systems as sysmod
    import tier3_systems as t3

    def naux_of(nao):
        return int(CHOLESKY_VECTORS_PER_AO * nao)

    def factors_ao(nao):
        return factor_memory_gb(nao, naux_of(nao))

    def factors_mo(nao):
        return mo_block_memory_gb(naux_of(nao), 2 * nao, 2 * nao)

    rec: dict = {"limits_gb": list(limits)}

    # -- item 4: which array binds, and at what nao ---------------------------------------
    rec["nao_ladder"] = [
        {"nao": nao,
         "eri_gb": eri_memory_gb(nao),
         "ao_factors_gb": factors_ao(nao),
         "mo_factors_gb": factors_mo(nao),
         "naux_estimated": naux_of(nao)}
        for nao in (100, 150, 200, 250, 300, 350, 400, 500, 600, 800, 1000)
    ]
    rec["refusal_nao"] = [
        {"limit_gb": lim,
         "eri": _refusal_dimension(eri_memory_gb, lim),
         "ao_factors": _refusal_dimension(factors_ao, lim),
         "mo_factors": _refusal_dimension(factors_mo, lim),
         "eri_plus_ao_factors": _refusal_dimension(
             lambda n: eri_memory_gb(n) + factors_ao(n), lim)}
        for lim in limits
    ]

    # -- item 5: the conventional-CI ceiling, from the sizing function ---------------------
    ceiling = []
    for n in range(12, 29, 2):
        k = n // 2
        row = {"n_spinors": n, "n_elec": k, "ndet": math.comb(n, k),
               "workspace_gb": sigma_workspace_gb(n, k)}
        row["vectors_gb"] = res.array_gb((row["ndet"],))
        ceiling.append(row)
    rec["ci_ceiling"] = ceiling
    rec["ci_ceiling_at_limit"] = [
        {"limit_gb": lim,
         "largest_n_resident": max([r["n_spinors"] for r in ceiling
                                    if r["workspace_gb"] <= lim] or [0]),
         "largest_n_vectors_only": max([r["n_spinors"] for r in ceiling
                                        if 12.0 * r["vectors_gb"] <= lim] or [0])}
        for lim in limits
    ]

    # ⚠ The batched kernel moves the ceiling from the intermediates onto the *Davidson
    # vectors*, and those scale with the ROOT COUNT — which is the axis a state-averaged
    # heavy-element calculation lives on. This table is the honest form of "the
    # scattering kernel reaches n = 28".
    from kuiva.ci.davidson import SUBSPACE_FACTOR, davidson_workspace_gb
    rec["davidson_roots"] = {"subspace_factor": SUBSPACE_FACTOR, "rows": [
        {"n_spinors": r["n_spinors"], "ndet": r["ndet"],
         "roots_at_limit": {
             "{:g}".format(lim): max(
                 [nr for nr in (1, 2, 4, 8, 16, 32, 64, 128, 256)
                  if davidson_workspace_gb(r["ndet"], SUBSPACE_FACTOR * nr) <= lim] or [0])
             for lim in limits}}
        for r in ceiling]}

    # -- item 7: the rank-4 RDM at Tier-3 active spaces ------------------------------------
    rec["tier3_rdm"] = [
        {"key": s.key, "topology": s.topology,
         "cas_electrons": s.cas_electrons, "cas_orbitals": s.cas_orbitals,
         "cas_spinors": s.cas_spinors, "cas_determinants": s.cas_determinants,
         "rdm2_gb": res.rdm_gb(s.cas_spinors, 2),
         "rdm3_gb": res.rdm_gb(s.cas_spinors, 3),
         "rdm4_gb": res.rdm_gb(s.cas_spinors, 4)}
        for s in t3.SYSTEMS
    ]
    rec["rdm4_ladder"] = [{"n_active": n, "rdm4_gb": res.rdm_gb(n, 4)}
                          for n in (8, 10, 12, 14, 16, 18, 20, 24, 30, 40)]

    # -- context: the Tier-1/2 suite's own AO counts, measured not guessed -----------------
    rec["suite_nao"] = _suite_nao(sysmod)
    return rec


def _basis_nao(basis: str, elements: Sequence[str]) -> Optional[int]:
    """Spherical AO count per element for ``basis``, or ``None`` if it does not cover one.

    Geometry-free: the count is a sum over atoms of a per-element property, so no molecule is
    built and nothing is evaluated.
    """
    from pyscf import gto
    from kuiva.basis import registry as reg
    # "A|B" = family A wherever it covers the element, family B otherwise — the mixed
    # assignment the registry explicitly supports (Peterson covers no light atom, so an actinide
    # complex in cc-pVnZ-X2C is *necessarily* a mixed one).
    names = basis.split("|")
    total = 0
    for sym in elements:
        try:
            spec = None
            for name in names:
                try:
                    spec = reg.resolve_for_pyscf(name, [sym])
                    break
                except ValueError:
                    continue
            if spec is None:
                return None
            shells = (spec[sym] if isinstance(spec, dict)
                      else gto.basis.load(spec, sym))
        except Exception:
            return None
        n = 0
        for shell in shells:
            ell = shell[0]
            # PySCF shells are [l, (exp, c...), ...] or [l, kappa, (exp, c...), ...]
            prims = shell[2:] if isinstance(shell[1], int) else shell[1:]
            ncontr = len(prims[0]) - 1
            n += (2 * ell + 1) * ncontr
        total += n
    return total


def _suite_nao(sysmod) -> List[dict]:
    rows = []
    for s in sysmod.SYSTEMS:
        elements = [sym for sym, _ in s.atoms]
        row = {"key": s.key, "n_atoms": len(elements)}
        for basis in ("x2c-SVPall-2c", "x2c-TZVPall-2c", "x2c-QZVPall-2c",
                      "ANO-RCC-VTZP", "cc-pVTZ-X2C", "dyallv3z"):
            row[basis] = _basis_nao(basis, elements)
        rows.append(row)
    return rows


def part_targets(bases: Sequence[str]) -> dict:
    """AO counts for the target system classes at defensible bases (item 4).

    Composition only — the systems are named by their formula, and the count is a sum over
    elements. No geometry is needed and none is asserted: what decides the item is the
    number of basis functions, and that is a property of the composition and the basis.
    """
    targets = [
        # (label, class, [(element, count), ...])
        ("[ReCl6]2-", "5d complex", [("Re", 1), ("Cl", 6)]),
        ("[Os(bpy-model)3]2+ core: Os + 6N + 12C + 8H", "5d complex",
         [("Os", 1), ("N", 6), ("C", 12), ("H", 8)]),
        ("[Dy(H2O)8]3+", "lanthanide, aquo", [("Dy", 1), ("O", 8), ("H", 16)]),
        ("[Dy(acac)3(H2O)2]", "lanthanide SMM",
         [("Dy", 1), ("O", 8), ("C", 15), ("H", 25)]),
        ("[Dy(Cp*)2]+ (dysprosocenium model)", "lanthanide SMM",
         [("Dy", 1), ("C", 20), ("H", 30)]),
        ("[Dy(Cp^ttt)2]+ (the real SMM)", "lanthanide SMM",
         [("Dy", 1), ("C", 34), ("H", 54)]),
        ("Dy2 dimer, [Dy2(OH)2(H2O)8]4+", "multi-site lanthanide",
         [("Dy", 2), ("O", 10), ("H", 18)]),
        ("UO2(H2O)5 2+", "actinide", [("U", 1), ("O", 7), ("H", 10)]),
        ("[U(BH4)4]", "actinide", [("U", 1), ("B", 4), ("H", 16)]),
        ("[UCl6]2-", "actinide", [("U", 1), ("Cl", 6)]),
    ]
    from kuiva.interface.pyscf_bridge import CHOLESKY_VECTORS_PER_AO, eri_memory_gb
    from kuiva.integrals.transform import factor_memory_gb, mo_block_memory_gb
    from kuiva.basis.registry import z_of as _z_of

    rows = []
    for label, klass, comp in targets:
        elements = []
        for sym, count in comp:
            elements.extend([sym] * count)
        row = {"label": label, "class": klass,
               "composition": {sym: count for sym, count in comp}}
        for basis in bases:
            nao = _basis_nao(basis, elements)
            if nao is None:
                row[basis] = None
                continue
            naux = int(CHOLESKY_VECTORS_PER_AO * nao)
            eri = eri_memory_gb(nao)
            ao = factor_memory_gb(nao, naux)
            mo = mo_block_memory_gb(naux, 2 * nao, 2 * nao)
            # The two phases that can peak, as the pre-flight plans them: the ERI array and
            # the AO factors are live together (the decomposition reads the first while
            # filling the second), then the AO factors and the MO block. "direct" is the
            # same plan with the ERI array removed — i.e. exactly what an integral-direct
            # Cholesky decomposition would buy, and nothing else.
            # ⚠ The MO term is a *soft* bound and the ERI term is a hard one: the pre-flight
            # already advises "transform only the orbital blocks that are needed", and the
            # cheapest useful block is ``B^P_{p,occ}`` (every spinor against the occupied
            # ones). That column is what the ordering of the two future work items turns on.
            nelec = sum(_z_of(sym) * count for sym, count in comp)
            mo_occ = mo_block_memory_gb(naux, 2 * nao, nelec)
            row[basis] = {
                "nao": nao, "n_electrons": nelec,
                "eri_gb": eri, "ao_factors_gb": ao, "mo_factors_gb": mo,
                "mo_factors_occ_blocked_gb": mo_occ,
                "peak_current_gb": max(eri + ao, ao + mo),
                "peak_direct_gb": ao + mo,
                "peak_blocked_gb": max(eri + ao, ao + mo_occ),
                "peak_blocked_direct_gb": ao + mo_occ,
                "binding_array": "ERI" if eri + ao > ao + mo else "MO factors",
                "binding_array_blocked": ("ERI" if eri + ao > ao + mo_occ
                                          else "MO factors"),
            }
        rows.append(row)
    return {"targets": rows, "bases": list(bases)}


# =========================================================================================
# Part 2 — overlap spectra near the linear-dependence threshold (item 6)
# =========================================================================================

#: Relative tolerance used to call two eigenvalues numerically degenerate. The same value the
#: Cholesky orbit grouping uses (``ORBIT_DEGENERACY_RTOL``), and the measured "no universal valley"
#: finding applies here too — which is why the sweep reports the answer at three tolerances.
GROUP_RTOLS = (1e-4, 1e-6, 1e-8)

#: Thresholds a user could plausibly set. The default is 1e-7; the question is not only
#: whether *that* cut splits a group but whether the neighbourhood is clean.
THRESHOLD_SCAN = tuple(10.0 ** (-e) for e in np.arange(5.0, 8.01, 0.25))


def _groups(evals_desc: np.ndarray, rtol: float) -> List[Tuple[int, int]]:
    """Consecutive-run grouping of descending eigenvalues by *relative* gap.

    Returns ``[(start, stop), ...]`` half-open index ranges. A group is a maximal run whose
    neighbours differ by less than ``rtol`` relatively — the same construction the Cholesky orbit
    grouping uses, applied here to the whole spectrum rather than to one block.
    """
    out = []
    start = 0
    for i in range(1, evals_desc.size):
        a, b = evals_desc[i - 1], evals_desc[i]
        scale = max(abs(a), abs(b), 1e-300)
        if (a - b) / scale > rtol:
            out.append((start, i))
            start = i
    out.append((start, evals_desc.size))
    return out


def _cut_report(evals_desc: np.ndarray, threshold: float, rtol: float) -> dict:
    """Does the cut at ``threshold`` fall inside a numerically degenerate group?"""
    keep = int(np.count_nonzero(evals_desc > threshold))
    rep = {"threshold": threshold, "rtol": rtol, "n_kept": keep,
           "n_dropped": int(evals_desc.size - keep)}
    if keep == 0 or keep == evals_desc.size:
        rep["splits_group"] = False
        rep["straddle_ratio"] = None
        return rep
    last_kept = float(evals_desc[keep - 1])
    first_dropped = float(evals_desc[keep])
    scale = max(abs(last_kept), abs(first_dropped), 1e-300)
    rel_gap = (last_kept - first_dropped) / scale
    rep["last_kept"] = last_kept
    rep["first_dropped"] = first_dropped
    rep["relative_gap_at_cut"] = rel_gap
    rep["splits_group"] = bool(rel_gap <= rtol)
    if rep["splits_group"]:
        for start, stop in _groups(evals_desc, rtol):
            if start < keep < stop:
                rep["split_group_size"] = stop - start
                rep["split_group_kept"] = keep - start
                rep["split_group_value"] = float(evals_desc[start])
                break
    return rep


def _spectrum_record(name: str, evals_desc: np.ndarray, threshold: float) -> dict:
    ev = np.asarray(evals_desc, dtype=float)
    rec = {"what": name, "n": int(ev.size),
           "largest": float(ev[0]), "smallest": float(ev[-1]),
           "condition": float(ev[0] / ev[-1]) if ev[-1] > 0 else float("inf")}
    # decades of margin: how far the nearest eigenvalue sits from the default cut
    near = float(np.min(np.abs(np.log10(np.clip(ev, 1e-300, None) / threshold))))
    rec["decades_to_nearest_eigenvalue"] = near
    rec["cuts"] = {}
    for rtol in GROUP_RTOLS:
        rec["cuts"]["rtol_{:g}".format(rtol)] = {
            "default": _cut_report(ev, threshold, rtol),
            "scan_splitting": [t for t in THRESHOLD_SCAN
                               if _cut_report(ev, t, rtol)["splits_group"]],
        }
    # the eigenvalues within a decade of the cut, verbatim — small and worth having raw
    window = ev[(ev < threshold * 10.0) & (ev > threshold * 0.1)]
    rec["window_x0.1_to_x10"] = [float(v) for v in window[:200]]
    return rec


def _mole(atoms, basis: str, charge: int, spin: int):
    from pyscf import gto
    from kuiva.basis import registry as reg
    mol = gto.Mole()
    mol.atom = [(sym, tuple(xyz)) for sym, xyz in atoms]
    names = basis.split("|")
    if len(names) == 1:
        mol.basis = reg.resolve_for_pyscf(basis, sorted({s for s, _ in atoms}))
    else:
        # "A|B": family A wherever it covers the element, B otherwise (see _basis_nao).
        spec = {}
        for sym in sorted({s for s, _ in atoms}):
            for name in names:
                try:
                    one = reg.resolve_for_pyscf(name, [sym])
                except ValueError:
                    continue
                spec[sym] = one[sym] if isinstance(one, dict) else one
                break
        mol.basis = spec
    mol.charge = charge
    mol.spin = spin
    mol.cart = False
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()
    return mol


def _four_component_metric_evals(mol) -> np.ndarray:
    """Eigenvalues of the diagonal-normalized 4c metric that ``canonical_orth`` cuts.

    Reproduces :func:`kuiva.x2c.decouple.canonical_orth`'s own preconditioning exactly (it
    normalizes by the metric diagonal before the eigendecomposition), on the **decontracted**
    basis the X2C helper builds — which is the metric the front-end actually cuts.
    """
    from pyscf.x2c import x2c
    helper = x2c.SpinOrbitalX2CHelper(mol)
    helper.xuncontract = True
    xmol, _ = helper.get_xmol(mol)
    s = xmol.intor_symmetric("int1e_ovlp")
    t = xmol.intor_symmetric("int1e_kin")
    from kuiva.x2c.decouple import LIGHT_SPEED
    m = np.zeros((2 * s.shape[0],) * 2)
    m[:s.shape[0], :s.shape[0]] = s
    m[s.shape[0]:, s.shape[0]:] = t / (2.0 * LIGHT_SPEED ** 2)
    d = np.diag(m).copy()
    norm = 1.0 / np.sqrt(np.where(d > 0.0, d, 1.0))
    mn = norm[:, None] * m * norm[None, :]
    return np.linalg.eigvalsh(mn)[::-1], xmol.nao


class _AtomProbe:
    """The three fields :func:`part_overlap` needs from a ``systems.System``."""

    def __init__(self, atoms, charge, spin):
        self.atoms, self.charge, self.spin = atoms, charge, spin


def _atom_probe(key: str) -> "_AtomProbe":
    from kuiva.basis import registry as reg
    _, sym, charge = key.split(":")
    nelec = reg.z_of(sym) - int(charge)
    return _AtomProbe([(sym, (0.0, 0.0, 0.0))], int(charge), nelec % 2)


def part_overlap(keys: Sequence[str], bases: Sequence[str], budget_s: float,
                 out_path: Path, do_4c: bool = True) -> dict:
    import systems as sysmod
    from kuiva.orth.canonical import DEFAULT_THRESHOLD
    from kuiva.x2c.decouple import METRIC_LINDEP_THRESHOLD

    t0 = time.time()
    records: List[dict] = []
    payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "orth_threshold": DEFAULT_THRESHOLD,
               "metric_threshold": METRIC_LINDEP_THRESHOLD,
               "group_rtols": list(GROUP_RTOLS),
               "records": records}

    for key in keys:
        # "atom:U:3" is an ad-hoc single-atom probe — the Peterson sets are primary
        # for the actinides and the suite has no actinide, so the family that matters most
        # for this question would otherwise go unmeasured.
        s = _atom_probe(key) if key.startswith("atom:") else sysmod.get(key)
        elements = sorted({sym for sym, _ in s.atoms})
        for basis in bases:
            if time.time() - t0 > budget_s:
                payload["stopped_early"] = "wall budget {:.0f}s exhausted".format(budget_s)
                _write(out_path, payload)
                return payload
            if any(_basis_nao(basis, [e]) is None for e in elements):
                continue
            row = {"system": key, "basis": basis, "elements": elements}
            try:
                mol = _mole(s.atoms, basis, s.charge, s.spin)
                row["nao"] = int(mol.nao)
                sao = mol.intor_symmetric("int1e_ovlp")
                ev = np.linalg.eigvalsh(sao)[::-1]
                row["contracted_ao_overlap"] = _spectrum_record(
                    "orth.canonical on the molecular AO overlap", ev, DEFAULT_THRESHOLD)
                if do_4c:
                    ev4, nao_unc = _four_component_metric_evals(mol)
                    row["decontracted_4c_metric"] = _spectrum_record(
                        "x2c.decouple.canonical_orth on the decontracted 4c metric",
                        ev4, METRIC_LINDEP_THRESHOLD)
                    row["nao_decontracted"] = int(nao_unc)
            except Exception as exc:                      # a missing basis is data, not a crash
                row["error"] = "{}: {}".format(type(exc).__name__, exc)
            row["elapsed_s"] = round(time.time() - t0, 1)
            records.append(row)
            _write(out_path, payload) # incremental
            print("[{:6.1f}s] {:12s} {:16s} nao={} {}".format(
                row["elapsed_s"], key, basis, row.get("nao", "-"),
                "SPLITS" if _any_split(row) else "clean"), flush=True)
    payload["elapsed_s"] = round(time.time() - t0, 1)
    _write(out_path, payload)
    return payload


def _any_split(row: dict) -> bool:
    for part in ("contracted_ao_overlap", "decontracted_4c_metric"):
        rec = row.get(part)
        if not rec:
            continue
        for cuts in rec["cuts"].values():
            if cuts["default"]["splits_group"]:
                return True
    return False


# =========================================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=("sizes", "targets", "overlap"), required=True)
    ap.add_argument("--budget", type=float, default=540.0,
                    help="hard wall budget in seconds (the ten-minute rule)")
    ap.add_argument("--systems", default="ne,ticl3,bi,tlh,ce3p,dy3p,yb3p,cecl3,hi,zn2p,tif3")
    ap.add_argument("--bases", default="x2c-SVPall-2c,x2c-TZVPall-2c,ANO-RCC-VTZP,"
                                       "cc-pVTZ-X2C,dyallv3z")
    ap.add_argument("--no-4c", action="store_true")
    ap.add_argument("--limits", default="8,16,64,256")
    args = ap.parse_args(argv)

    os.environ.setdefault("KUIVA_MEMORY_GB", "8")
    limits = [float(x) for x in args.limits.split(",")]
    bases = [b for b in args.bases.split(",") if b]

    if args.part == "sizes":
        rec = part_sizes(limits)
        _write(TEMP / "spec_gaps_sizes.json", rec)
        _print_sizes(rec)
    elif args.part == "targets":
        rec = part_targets(bases)
        _write(TEMP / "spec_gaps_targets.json", rec)
        _print_targets(rec)
    else:
        keys = [k for k in args.systems.split(",") if k]
        part_overlap(keys, bases, args.budget, TEMP / "spec_gaps_overlap.json",
                     do_4c=not args.no_4c)
    return 0


def _print_sizes(rec: dict) -> None:
    print("\n-- nao ladder (GB) --")
    print("%6s %12s %14s %14s" % ("nao", "ERI", "AO factors", "MO factors"))
    for r in rec["nao_ladder"]:
        print("%6d %12.2f %14.2f %14.2f"
              % (r["nao"], r["eri_gb"], r["ao_factors_gb"], r["mo_factors_gb"]))
    print("\n-- refusal nao by limit --")
    print("%10s %8s %12s %12s %14s" % ("limit GB", "ERI", "AO factors", "MO factors",
                                       "ERI+AO"))
    for r in rec["refusal_nao"]:
        print("%10.0f %8d %12d %12d %14d" % (r["limit_gb"], r["eri"], r["ao_factors"],
                                             r["mo_factors"], r["eri_plus_ao_factors"]))
    print("\n-- conventional CI ceiling --")
    print("%8s %8s %14s %12s %12s" % ("n", "k", "ndet", "workspace", "one vector"))
    for r in rec["ci_ceiling"]:
        print("%8d %8d %14d %12.2f %12.4f"
              % (r["n_spinors"], r["n_elec"], r["ndet"], r["workspace_gb"],
                 r["vectors_gb"]))
    for r in rec["ci_ceiling_at_limit"]:
        print("  limit %6.0f GB: resident kernel reaches n = %d; "
              "vectors-only (12 stacks) reaches n = %d"
              % (r["limit_gb"], r["largest_n_resident"], r["largest_n_vectors_only"]))
    print("\n-- Tier-3 rank-4 RDM --")
    print("%-14s %8s %14s %16s" % ("key", "spinors", "dets", "4-RDM GB"))
    for r in rec["tier3_rdm"]:
        print("%-14s %8d %14.3e %16.3e"
              % (r["key"], r["cas_spinors"], r["cas_determinants"], r["rdm4_gb"]))
    print("\n-- 4-RDM ladder --")
    for r in rec["rdm4_ladder"]:
        print("  n_active %3d -> %14.3f GB" % (r["n_active"], r["rdm4_gb"]))


def _print_targets(rec: dict) -> None:
    for b in rec["bases"]:
        print("\n== %s ==" % b)
        print("%-34s %6s %8s %8s %8s %8s %9s %9s %9s %-11s"
              % ("target", "nao", "ERI", "AOfact", "MOfull", "MOocc", "peak now",
                 "pk direct", "pk blk+dir", "binds(blk)"))
        for row in rec["targets"]:
            v = row.get(b)
            if not v:
                print("%-34s %6s" % (row["label"][:34], "-"))
                continue
            print("%-34s %6d %8.1f %8.1f %8.1f %8.1f %9.1f %9.1f %9.1f %-11s"
                  % (row["label"][:34], v["nao"], v["eri_gb"], v["ao_factors_gb"],
                     v["mo_factors_gb"], v["mo_factors_occ_blocked_gb"],
                     v["peak_current_gb"], v["peak_direct_gb"],
                     v["peak_blocked_direct_gb"], v["binding_array_blocked"]))


if __name__ == "__main__":
    raise SystemExit(main())
