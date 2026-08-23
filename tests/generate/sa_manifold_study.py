"""When is a sub-manifold state average safe? The instability ladder.

The open question this measures: an ensemble that passes the
degeneracy gate AND the boundary check — a complete ``2J+1`` manifold, say — can still make
the symmetric point a *saddle* of the SA-CASSCF functional, so the run slides off it by
however much rounding pushes. Characterized here on a ladder of (system, ensemble) pairs:

* free-ion sub-term averages (b: j=1/2 of ^2P; ce3p: ^2F5/2 of ^2F; fe2p: J=4 of ^5D,
  the multi-electron d shell) — the suspected saddles;
* ligand-field sub-averages (ticl3, cecl3: ground Kramers doublet) — the practically
  important cases;
* full-space controls (every root of the CAS), which are invariant by construction;
* optionally dy3p's ^6H15/2-only average (16 of 2002) — the standard lanthanide protocol.

Per case, every probe starts at the **symmetric point** — the converged orbitals of the
system's invariant reference ensemble (``reference_n``, converged once from the guess and
cached), because the guess-orbital statics measure the scalar ROHF's own broken field as
much as the ensemble:

* **static** detectors at the symmetric point and at the unperturbed probe's converged
  orbitals (dense CI over the tiny active space): boundary gap; time-odd fraction, spin
  non-invariance ``max_k ||[gamma, S_k]|| / ||gamma||`` and occupation spectrum of the SA
  density; ``||gamma_ens - gamma_block||`` against every enclosing block boundary of the
  spectrum head, with that boundary's gap. Detector costs are recorded — "what would a
  cheap invariance check cost" is one of the three questions.
* **dynamic** probes: SA-CASSCF from the unperturbed symmetric point (is the sub-ensemble
  *stationary* there?) and from two deterministically seeded two-scale displacements
  ``c0 @ expm(K)`` (is it *stable*?): a Kramers-preserving even part (PERTURB_EVEN) forces
  the optimizer to do real work without tripping the degeneracy gate, and a tiny time-odd
  passenger (PERTURB_ODD) rides along; whether the passenger decays or is amplified over
  those iterations is the stability measurement. Per-iteration series (pairing defect,
  time-odd gamma, spin non-invariance); converged runs record within-manifold spreads,
  Kramers splittings, the phase-invariant level reduction, and the boundary reports.
  ⚠ A mid-run state-averaging-gate refusal is a recorded *outcome* of the probe (the
  instability reaching 1e-6 Eh), never an error of the study.

Verdict thresholds, fixed before any number existed:
stable = every expected-exact block spread < 0.1 cm^-1 AND Kramers splitting < 1e-10 Eh AND
cross-seed principal-g spread < 1e-3. Unstable = any violated, or a gate refusal.

Run discipline: every leg is minutes (the ten-minute rule); ``--budget`` is checked between
legs, records are rewritten after every completed case, and a heartbeat ticks per leg.

Usage::

    python tests/generate/sa_manifold_study.py --only b          # smoke, ~2 min
    python tests/generate/sa_manifold_study.py --merge           # the ladder
    python tests/generate/sa_manifold_study.py --only dy3p --budget 1800
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import systems as sysdef                                                    # noqa: E402
import thermal                                                              # noqa: E402
from progress import Heartbeat                                              # noqa: E402

OUT = REPO / "temp/sa_manifold.json"
SCHEMA = 1
MEMORY_GB = 8.0
#: The Kramers-preserving displacement: big enough that the optimizer must do real work
#: (|g| well above conv_grad), gate-safe by construction (it cannot split a pair).
PERTURB_EVEN = 3e-3
#: ⚠ The symmetry-breaking passenger: small enough that its own Kramers splitting (~linear
#: in eps) stays well under the 1e-6 Eh degeneracy tolerance — at 1e-5 the very first CI
#: solve is gate-refused, which measures the gate, not the instability. 1e-8 is still 5-7
#: orders above machine noise.
PERTURB_ODD = 1e-8
SEEDS = (11, 23)

#: Verdict thresholds — fixed in the plan before any number existed.
STABLE_SPREAD_CM = 0.1          # within an expected-exact manifold (the free-ion bar)
STABLE_KRAMERS_EH = 1e-10       # the class-API bar for a Kramers doublet
SEED_G_SPREAD = 1e-3            # cross-run principal-g reproducibility (suite band is 2e-3)

#: The ladder. ``ensembles`` maps n_states -> the degeneracy pattern that is *exact physics*
#: for that converged ensemble (free ion: 2J+1 manifolds; ligand field: Kramers doublets).
#: ``reference_n`` is the invariant ensemble converged ONCE from the scalar guess; its
#: converged orbitals are the symmetric point every probe starts from (cached on disk so a
#: sweep can be resumed one ten-minute invocation at a time).
CASES: Dict[str, Dict] = {
    "b": {"ensembles": {2: [2], 6: [2, 4]}, "label": {2: "j=1/2 of ^2P", 6: "whole ^2P"},
          "reference_n": 6, "from_guess": True},
    "ce3p": {"ensembles": {6: [6], 14: [6, 8]},
             "label": {6: "^2F5/2 alone", 14: "whole ^2F"}, "reference_n": 14,
             "from_guess": True},
    "ticl3": {"ensembles": {2: [2], 10: [2] * 5},
              "label": {2: "ground LF doublet", 10: "full d^1 space"}, "reference_n": 10},
    # two legs are deliberately absent here: the full-space control (the reference leg IS
    # an SA-14 run, and the control physics is demonstrated on b/ce3p/ticl3), and the
    # 6-of-14 j=5/2-parentage group — same leaning class as the ground doublet at a 16x
    # LARGER backing gap (3533 vs 226 cm^-1), so the doublet leg already answers the harder
    # question, and at 350 spinors a near-convergence second-order step outlives a bounded
    # invocation on this box (zero iterations per ten-minute window, measured)
    "cecl3": {"ensembles": {2: [2]},
              "label": {2: "ground LF doublet"},
              "reference_n": 14,
              # 350 spinors on a thermally clamped box: the last decade of |g| is a
              # second-order step that outlives a bounded invocation, and the question
              # needs a symmetric point, not a 1e-4 gradient. Recorded per run.
              "conv_grad": 5e-4},
    # the originally recorded instance: J=4 of ^5D converged 725 cm^-1 split at a 252 cm^-1
    # boundary gap — the multi-electron d shell, whose gap sits almost exactly at cecl3's
    # 226 cm^-1, so it decides whether any static threshold separates the two regimes
    # ⚠ from_guess probes reproduce the *trajectory* question on the free ions: a locally
    # stable symmetric point can still be missed by the guess-started run (a different
    # basin), which is what the recorded J=4-of-^5D 725 cm^-1 split turned out to be
    "fe2p": {"ensembles": {9: [9], 25: [9, 7, 5, 3, 1]},
             "label": {9: "J=4 of ^5D alone", 25: "whole ^5D"}, "reference_n": 25,
             "from_guess": True},
    # dy3p's reference ensemble is the committed Tier-2 boundary (134), not the full space
    # (2002 roots is not a calculation); the probe is the standard "ground multiplet only"
    "dy3p": {"ensembles": {16: [16]}, "label": {16: "^6H15/2 alone (16 of 2002)"},
             "reference_n": 134, "seeds": (11,)},
}
ORDER = ("b", "ce3p", "ticl3", "cecl3", "fe2p", "dy3p")

#: fe2p is a study-only system (the committed registry stays the tier suite's):
#: Fe(2+) 3d^6, ^5D term, inverted multiplet (J=4 ground). No tier records exist for it.
EXTRA_SYSTEMS = {
    "fe2p": sysdef.System(
        key="fe2p", label="Fe(2+)", atoms=[("Fe", (0.0, 0.0, 0.0))], charge=2, spin=4,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=5, nelecas=6, active_l="d", nroots={5: 5}, tier1=False),
}
PROBE_MAX_ITER = 40
REFERENCE_MAX_ITER = 100


# --- measures -------------------------------------------------------------------------------

def time_odd_fraction(a: np.ndarray) -> float:
    """``max |J conj(A) J^T - A| / max |A|`` in the interleaved Kramers pair order."""
    n = a.shape[0]
    j = np.zeros((n, n))
    j[1::2, 0::2] = np.eye(n // 2)
    j[0::2, 1::2] = -np.eye(n // 2)
    scale = float(np.max(np.abs(a))) or 1.0
    return float(np.max(np.abs(j @ np.conj(a) @ j.T - a))) / scale


def pairing_defect(c_act: np.ndarray) -> float:
    """``max |c_{2p+1} - T c_{2p}|`` over active columns — the seed the instability amplifies."""
    from kuiva.spinor.expand import time_reverse
    partner = time_reverse(c_act[:, 0::2])
    return float(np.max(np.abs(c_act[:, 1::2] - partner)))


def spin_operator_mo(reference, coeff: np.ndarray, active: np.ndarray) -> np.ndarray:
    """``S_k`` (3, n_act, n_act) over the active spinors."""
    from kuiva.spinor.expand import spin_operator
    s2c = spin_operator(np.asarray(reference.data.s_ao))
    c_act = coeff[:, active]
    return np.stack([c_act.conj().T @ s2c[k] @ c_act for k in range(3)])


def spin_noninvariance(gamma: np.ndarray, s_mo: np.ndarray) -> float:
    """``max_k ||[gamma, S_k]||_F / ||gamma||_F`` — 0 iff the ensemble density is invariant
    under spin rotations, i.e. iff the average does not lean on the SOC structure."""
    scale = float(np.linalg.norm(gamma)) or 1.0
    return max(float(np.linalg.norm(gamma @ s_mo[k] - s_mo[k] @ gamma)) / scale
               for k in range(3))


def perturbed(c0: np.ndarray, eps_even: float, eps_odd: float, seed: int,
              active: np.ndarray) -> np.ndarray:
    """``c0 @ expm(K)``: a two-scale deterministic displacement from the symmetric point.

    ``K = eps_even K_even + eps_odd K_odd`` with ``K_even`` commuting with time reversal
    (a Kramers-preserving rotation: it forces the optimizer to do real work but cannot split
    a pair, so the first CI solve cannot be gate-refused) and ``K_odd`` the tiny
    symmetry-breaking passenger whose growth or decay under those iterations is the actual
    stability measurement. Both are scaled to unit max element before the factors apply.

    ⚠ Both live in a window around the active shell (active +- a few pairs), not over all
    orbitals: a random rotation touching deep core orbitals starts the probe at |g| ~ 1e2
    and the optimizer spends 40 macro-iterations crawling core levels back — pure cost, and
    physics the question is not about.
    """
    from scipy.linalg import expm
    n = c0.shape[1]
    lo = max(0, int(active[0]) - 6)
    hi = min(n, int(active[-1]) + 11)
    lo -= lo % 2                                     # whole Kramers pairs
    hi += hi % 2
    m = hi - lo
    j = np.zeros((m, m))
    j[1::2, 0::2] = np.eye(m // 2)
    j[0::2, 1::2] = -np.eye(m // 2)
    rng = np.random.default_rng(seed)

    def anti():
        a = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
        return 0.5 * (a - a.conj().T)

    a, b = anti(), anti()
    k_even = 0.5 * (a + j @ np.conj(a) @ j.T)
    k_odd = 0.5 * (b - j @ np.conj(b) @ j.T)
    k_win = (eps_even * k_even / max(float(np.max(np.abs(k_even))), 1e-300)
             + eps_odd * k_odd / max(float(np.max(np.abs(k_odd))), 1e-300))
    k = np.zeros((n, n), dtype=np.complex128)
    k[lo:hi, lo:hi] = k_win
    return np.ascontiguousarray(c0 @ expm(k))


def block_spreads(energies: np.ndarray, pattern: Sequence[int]) -> List[float]:
    """Spread [cm^-1] within each expected-exact block of the converged ensemble."""
    from kuiva.props.multiplet import HARTREE_TO_CM
    spreads, at = [], 0
    for size in pattern:
        block = energies[at:at + size]
        spreads.append(float(np.ptp(block)) * HARTREE_TO_CM if block.size else float("nan"))
        at += size
    return spreads


def level_g(reference, outcome) -> List[Dict]:
    """``(size, g)`` per degenerate level — the phase-invariant reduction, whole blocks only.

    ⚠ Never a slice of a block: half a quartet's "g" depends on the eigensolver's arbitrary
    intra-block rotation. The level *pattern* is itself an observable here — an unstable run
    splits blocks a stable one keeps whole.
    """
    from kuiva.interface import api
    matrices = api.property_matrices(reference, outcome)
    return [{"size": int(level.size),
             "rel_cm": round(float(level.energy_cm), 4),
             "g": [round(float(g), 8) for g in level.g_values]}
            for level in matrices.analyse()]


# --- the legs -------------------------------------------------------------------------------

def static_diagnostics(reference, space, coeff: np.ndarray, n_states: int) -> Dict:
    """Dense CI at fixed orbitals: the boundary spectrum and every candidate detector."""
    from kuiva.mcscf.casci import FullCISolver
    from kuiva.mcscf.orbopt import CASIntegrals
    from kuiva.props.multiplet import HARTREE_TO_CM
    from kuiva.rdm.rdm import RDMBuilder, state_average_weights

    ints = CASIntegrals.build(reference.factors, reference.h_one_electron(), coeff,
                              space.spaces, e_nuc=reference.data.e_nuc)
    n_act = space.spaces.n_active
    ndet_cap = 2048                       # dy3p head: enough spectrum, bounded cost
    solver = FullCISolver(n_act, space.n_elec, n_states=1)
    ndet = solver.ndet
    n_all = min(ndet, ndet_cap)
    # every root and vector at these integrals (tiny spaces; dy3p's 2002 is seconds dense)
    h = np.ascontiguousarray(ints.h_active_effective())
    eri = ints.active_eri()
    if ndet <= ndet_cap:
        from kuiva.ci.strings import Determinants, hamiltonian_matrix
        dets = Determinants(solver.space.masks, n_spinor=n_act, n_elec=space.n_elec)
        ham = hamiltonian_matrix(dets, h, eri)
        ham = ham.toarray() if hasattr(ham, "toarray") else np.asarray(ham)
        evals, evecs = np.linalg.eigh(ham)
        vectors = np.ascontiguousarray(evecs.T)
    else:                                                     # pragma: no cover - not hit
        raise ValueError("static leg expects a dense-solvable space, got {}".format(ndet))
    rel_cm = (evals - evals[0]) * HARTREE_TO_CM

    builder = RDMBuilder(solver.space)
    s_mo = spin_operator_mo(reference, coeff, np.asarray(space.spaces.active))

    def ensemble_gamma(m: int) -> np.ndarray:
        w = state_average_weights(evals[:m], space.n_elec, on_split="warn")
        gamma, _ = builder(vectors[:m], w, energies=evals[:m], on_split="warn")
        return gamma

    t0 = time.process_time()
    gamma = ensemble_gamma(n_states)
    t_gamma = time.process_time() - t0
    t0 = time.process_time()
    s_ni = spin_noninvariance(gamma, s_mo)
    t_detector = time.process_time() - t0

    occ = np.sort(np.linalg.eigvalsh(gamma))[::-1]
    rec = {
        "ndet": int(ndet),
        "boundary_gap_cm": (None if n_states >= n_all
                            else round(float(rel_cm[n_states] - rel_cm[n_states - 1]), 4)),
        "spectrum_head_cm": [round(float(x), 4) for x in rel_cm[:min(n_all, 40)]],
        "time_odd": float(time_odd_fraction(gamma)),
        "spin_noninvariance": round(s_ni, 6),
        "occupations": [round(float(x), 6) for x in occ[:12]],
        "detector_cpu_s": {"gamma": round(t_gamma, 4), "spin_check": round(t_detector, 6)},
    }
    # gamma against every enclosing block boundary of the spectrum head — the raw data a
    # cluster-based detector would be built from: how far the ensemble density is from each
    # larger invariant candidate, and how strongly that boundary is backed by its own gap
    from kuiva.rdm.rdm import degenerate_blocks
    head = min(n_all, 40)
    blocks = degenerate_blocks(evals[:head])                  # fine (1e-6 Eh) blocks
    enclosing = []
    for _, stop in blocks:
        if stop <= n_states:
            continue
        gap = float(rel_cm[stop] - rel_cm[stop - 1]) if stop < n_all else None
        d = np.linalg.norm(ensemble_gamma(stop) - gamma) / max(np.linalg.norm(gamma), 1e-300)
        enclosing.append({"m": int(stop), "gap_cm": None if gap is None else round(gap, 4),
                          "dgamma_rel": round(float(d), 6)})
        if len(enclosing) >= 6:
            break
    rec["enclosing"] = enclosing
    return rec


def dynamic_probe(reference, space, coeff: np.ndarray, n_states: int, pattern, *,
                  tag: str, max_iter: int = 120, key: str = "",
                  conv_grad: float = 1e-4) -> Tuple[Dict, Optional[np.ndarray]]:
    """One SA-CASSCF from these starting orbitals, instrumented per macro-iteration.

    Resumable: the trajectory checkpoints every macro-iteration and the per-iteration
    series appends to a side file, so a probe a bounded invocation cannot finish continues
    in the next one instead of restarting — a killed probe otherwise loses exactly the
    early iterations the growth measurement lives in.
    """
    from kuiva.interface import api
    from kuiva.props.multiplet import HARTREE_TO_CM

    act = np.asarray(space.spaces.active)
    stem = "sa_manifold_probe_{}_{}_{}".format(key, n_states, tag)
    ckpt = REPO / "temp/{}.h5".format(stem)
    series_path = REPO / "temp/{}.series.jsonl".format(stem)

    def watch(info) -> None:
        c = info["coeff"]
        s_mo = spin_operator_mo(reference, c, act)
        with series_path.open("a") as fh:
            fh.write(json.dumps({
                "iteration": int(info["iteration"]),
                "grad_norm": float(info["grad_norm"]),
                "pairing": pairing_defect(c[:, act]),
                "time_odd": float(time_odd_fraction(info["gamma"])),
                "spin_ni": spin_noninvariance(info["gamma"], s_mo),
            }) + "\n")

    def stored_series() -> List[Dict]:
        if not series_path.is_file():
            return []
        return [json.loads(line) for line in series_path.read_text().splitlines() if line]

    t0 = time.process_time()
    rec: Dict = {"tag": tag, "n_states": int(n_states), "conv_grad": conv_grad}
    kw = dict(n_states=n_states, callback=watch, report=False, max_iter=max_iter,
              checkpoint=str(ckpt), checkpoint_options=dict(min_interval=0.0),
              conv_grad=conv_grad)
    try:
        if ckpt.is_file():
            outcome = api.casscf(reference, restart=str(ckpt), **kw)
        else:
            outcome = api.casscf(reference, active=act.tolist(),
                                 n_active_elec=space.n_elec, coeff=coeff, **kw)
    except ValueError as exc:
        # the state-averaging gate refusing mid-run IS the instability reaching 1e-6 Eh
        series = stored_series()
        rec.update(status="gate-refused", error=str(exc)[:300],
                   iterations=len(series), series=series,
                   cpu_seconds=round(time.process_time() - t0, 2))
        for path in (ckpt, series_path):
            if path.is_file():
                path.unlink()
        return rec, None
    series = stored_series()
    for path in (ckpt, series_path):
        if path.is_file():
            path.unlink()

    energies = np.asarray(outcome.ci.energies, dtype=float)
    spreads = block_spreads(energies, pattern)
    kramers = [float(energies[i + 1] - energies[i])
               for i in range(0, energies.size - 1, 2)]
    rec.update(
        status="ok", converged=bool(outcome.converged),
        iterations=int(outcome.orbital.n_iterations),
        grad_norm=float(outcome.orbital.grad_norm),
        e_avg=float(outcome.energy),
        rel_cm=[round(float(x), 6)
                for x in (energies - energies[0]) * HARTREE_TO_CM],
        block_spreads_cm=[round(s, 6) for s in spreads],
        max_spread_cm=round(max(spreads), 6),
        kramers_max_eh=float(np.max(np.abs(kramers))) if kramers else 0.0,
        boundary_initial_cm=(None if outcome.boundary_initial is None
                             else outcome.boundary_initial.gap_cm),
        boundary_final_cm=(None if outcome.boundary is None else outcome.boundary.gap_cm),
        series=series,
        cpu_seconds=round(time.process_time() - t0, 2),
    )
    if series:
        rec["final_spin_ni"] = series[-1]["spin_ni"]
        rec["final_time_odd"] = series[-1]["time_odd"]
    try:
        rec["levels"] = level_g(reference, outcome)
    except Exception as exc:                                  # noqa: BLE001 - record, go on
        rec["levels_error"] = "{}: {}".format(type(exc).__name__, exc)
    return rec, np.ascontiguousarray(outcome.coeff)


def reference_orbitals(reference, space, plan: Dict, key: str, rec: Dict, tick) -> np.ndarray:
    """The symmetric point: the invariant reference ensemble converged once from the guess.

    Cached in ``temp/`` so a sweep resumes one bounded invocation at a time; the cache is
    keyed by nothing but the system, which is fine for a *study* (never for a test, and the
    file lives outside the repo's committed data).
    """
    from kuiva.interface import api

    orb_path = REPO / "temp/sa_manifold_orbs_{}.npy".format(key)
    n_ref = int(plan["reference_n"])
    conv_grad = float(plan.get("conv_grad", 1e-4))
    if orb_path.is_file():
        rec["reference_leg"] = {"status": "cached", "n_states": n_ref,
                                "path": str(orb_path)}
        return np.ascontiguousarray(np.load(orb_path))
    tick("reference-casscf-{}".format(n_ref))
    ckpt = REPO / "temp/sa_manifold_ref_{}.h5".format(key)
    if ckpt.is_file():
        # a budget-interrupted reference whose gradient is already under the plan's
        # tolerance is the symmetric point; finishing the last decade of |g| would be
        # measuring the box (one near-convergence second-order step outlives a bounded
        # invocation at 350 spinors)
        from kuiva.io.checkpoint import read_checkpoint
        resumed = read_checkpoint(str(ckpt))
        if float(resumed.grad_norm) < conv_grad:
            coeff = np.ascontiguousarray(resumed.coeff)
            rec["reference_leg"] = {"status": "accepted-from-checkpoint",
                                    "n_states": n_ref,
                                    "grad_norm": float(resumed.grad_norm),
                                    "conv_grad": conv_grad,
                                    "iterations": int(resumed.iteration)}
            np.save(orb_path, coeff)
            return coeff
    t0 = time.process_time()
    kw = dict(n_states=n_ref, report=False, max_iter=REFERENCE_MAX_ITER,
              checkpoint=str(ckpt), conv_grad=conv_grad)
    if ckpt.is_file():                       # resume a budget-interrupted reference leg
        outcome = api.casscf(reference, restart=str(ckpt), **kw)
    else:
        outcome = api.casscf(reference, active=np.asarray(space.spaces.active).tolist(),
                             n_active_elec=space.n_elec, **kw)
    rec["reference_leg"] = {
        "status": "ok", "n_states": n_ref, "converged": bool(outcome.converged),
        "iterations": int(outcome.orbital.n_iterations),
        "grad_norm": float(outcome.orbital.grad_norm),
        "e_avg": float(outcome.energy),
        "cpu_seconds": round(time.process_time() - t0, 2),
    }
    if not outcome.converged:
        raise RuntimeError("the reference-ensemble CASSCF did not converge; the symmetric "
                           "point does not exist yet, so no probe from it means anything")
    coeff = np.ascontiguousarray(outcome.coeff)
    orb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(orb_path, coeff)
    return coeff


def run_case(system: sysdef.System, plan: Dict, *, heartbeat, deadline,
             legs: Sequence[int] = (), previous: Optional[Dict] = None,
             save=None) -> Dict:
    from kuiva.interface import api
    from kuiva.util import resources as res

    res.clear()
    key = system.key
    t_sys = time.time()
    rec: Dict = {"key": key, "basis": system.basis, "legs": {}}
    if previous and previous.get("status") in ("ok", "partial"):
        # one bounded invocation per ensemble: keep only legs that finished all their
        # probes (the verdict is computed exactly then); partial legs contribute their
        # finished probes through prev_runs below
        rec["legs"] = {n: leg for n, leg in previous.get("legs", {}).items()
                       if "verdict" in leg
                       and not any(r.get("status") == "skipped"
                                   for r in leg.get("runs", ()))}

    def tick(stage):
        heartbeat.tick(int(time.time() - t_sys), system=key, stage=stage)

    tick("reference")
    molecule = api.Molecule(atoms=system.atoms, basis=system.basis,
                            charge=system.charge, spin=system.spin)
    # screening="none": the 2e-SOC picture change changes no conclusion here and costs a
    # 4c atomic solve per element; the 1e SOC the instability needs stays on
    reference = api.spinor_reference(molecule, memory_gb=MEMORY_GB, screening="none")
    rec["e_scf"] = float(reference.data.e_scf)

    space = api.active_space_for(reference, character=(0, system.active_l),
                                 n_active=2 * system.ncas, n_active_elec=system.nelecas)
    rec["active_space"] = space.description
    c_ref = reference_orbitals(reference, space, plan, key, rec, tick)

    seeds = plan.get("seeds", SEEDS)
    for n_states, pattern in plan["ensembles"].items():
        # from-guess probes only for strict sub-ensembles: for the reference ensemble the
        # from-guess run is the reference leg itself
        from_guess = bool(plan.get("from_guess")) and n_states < int(plan["reference_n"])
        expected_tags = ["unperturbed"] + ["seed-{}".format(s) for s in seeds]
        if from_guess:
            expected_tags.append("from-guess")
        kept = rec["legs"].get(str(n_states))
        if kept is not None:
            have = {r.get("tag") for r in kept.get("runs", ())}
            if set(expected_tags) <= have:
                continue
            del rec["legs"][str(n_states)]      # protocol widened: redo with prev_runs reuse
        if legs and n_states not in legs:
            rec["legs"].setdefault(str(n_states), {"status": "skipped",
                                                   "reason": "not in --legs"})
            continue
        if time.time() > deadline:
            rec["legs"][str(n_states)] = {"status": "skipped", "reason": "wall budget spent"}
            continue
        tick("static-{}".format(n_states))
        # a partial leg from an earlier bounded invocation contributes its finished probes;
        # only the missing ones are recomputed (a probe is minutes and the ladder is many)
        prev_leg = (previous or {}).get("legs", {}).get(str(n_states), {})
        prev_runs = {r.get("tag"): r for r in prev_leg.get("runs", ())
                     if r.get("status") in ("ok", "gate-refused")}
        # every probe starts at the symmetric point (the reference ensemble's converged
        # orbitals): the unperturbed probe asks whether the sub-ensemble is *stationary*
        # there, the seeded probes whether it is *stable*
        leg: Dict = {"label": plan["label"][n_states],
                     "static_at_reference": (prev_leg.get("static_at_reference")
                                             or static_diagnostics(reference, space, c_ref,
                                                                   n_states)),
                     "runs": []}
        if prev_leg.get("static_converged"):
            leg["static_converged"] = prev_leg["static_converged"]
        starts = [("unperturbed", c_ref)] + [
            ("seed-{}".format(s), perturbed(c_ref, PERTURB_EVEN, PERTURB_ODD, s,
                                np.asarray(space.spaces.active)))
            for s in seeds]
        if from_guess:
            starts.append(("from-guess", np.ascontiguousarray(reference.spinors_in_ao())))
        for tag, c_start in starts:
            if tag in prev_runs:
                leg["runs"].append(prev_runs[tag])
                continue
            if time.time() > deadline:
                leg["runs"].append({"tag": tag, "status": "skipped",
                                    "reason": "wall budget spent"})
                continue
            tick("casscf-{}-{}".format(n_states, tag))
            run, c_conv = dynamic_probe(reference, space, c_start, n_states, pattern,
                                        tag=tag, max_iter=PROBE_MAX_ITER, key=key,
                                        conv_grad=float(plan.get("conv_grad", 1e-4)))
            leg["runs"].append(run)
            if tag == "unperturbed" and c_conv is not None:
                tick("static-conv-{}".format(n_states))
                leg["static_converged"] = static_diagnostics(reference, space, c_conv,
                                                             n_states)
            if save is not None:                    # a killed invocation keeps its probes
                partial = dict(rec, status="partial")
                partial["legs"] = dict(rec["legs"], **{str(n_states): leg})
                save(partial)
        # the verdict, by the pre-registered thresholds
        ok_runs = [r for r in leg["runs"] if r.get("status") == "ok"]
        refused = any(r.get("status") == "gate-refused" for r in leg["runs"])
        spread_bad = any(r["max_spread_cm"] > STABLE_SPREAD_CM for r in ok_runs)
        # Kramers' theorem needs an odd electron count; for even N the adjacent-pair gap
        # is real physics, not a degeneracy defect, and asserting on it is a misfire
        kramers_bad = (space.n_elec % 2 == 1 and
                       any(r["kramers_max_eh"] > STABLE_KRAMERS_EH for r in ok_runs))
        # cross-seed reproducibility of the phase-invariant reduction: compare whole levels,
        # and treat a level *pattern* that changes with the seed as instability itself
        g_sets = [[(lv["size"], g) for lv in r["levels"] for g in lv["g"]]
                  for r in ok_runs if r.get("levels")]
        patterns = [tuple(s for s, _ in gs) for gs in g_sets]
        pattern_bad = len(set(patterns)) > 1 if len(patterns) > 1 else False
        g_spread = None
        if len(g_sets) > 1 and not pattern_bad:
            g_spread = max(abs(a[1] - b[1]) for ga, gb in zip(g_sets, g_sets[1:])
                           for a, b in zip(ga, gb))
        leg["g_cross_seed_spread"] = None if g_spread is None else round(g_spread, 8)
        leg["level_pattern_seed_dependent"] = pattern_bad
        g_bad = pattern_bad or (g_spread is not None and g_spread > SEED_G_SPREAD)
        leg["verdict"] = ("unstable" if (refused or spread_bad or kramers_bad or g_bad)
                          else ("stable" if ok_runs else "no-data"))
        leg["verdict_why"] = {"gate_refused": refused, "spread": spread_bad,
                              "kramers": kramers_bad, "g_seed_dependent": g_bad}
        rec["legs"][str(n_states)] = leg
        print("  [{}] n={} ({}): {}".format(key, n_states, leg["label"], leg["verdict"]),
              flush=True)

    rec["status"] = "ok"
    rec["seconds"] = round(time.time() - t_sys, 2)
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="sub-manifold state-average instability ladder")
    ap.add_argument("--only", default="", help="comma-separated system keys")
    ap.add_argument("--legs", default="",
                    help="comma-separated n_states to run this invocation (others kept "
                         "from the merged record or marked skipped)")
    ap.add_argument("--budget", type=float, default=3600.0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    import kuiva
    out_path = Path(args.out)
    out: Dict = {"schema": SCHEMA, "generator": "tests/generate/sa_manifold_study.py",
                 "kuiva_version": kuiva.__version__,
                 "perturb_even": PERTURB_EVEN, "perturb_odd": PERTURB_ODD,
                 "thresholds": {"spread_cm": STABLE_SPREAD_CM,
                                "kramers_eh": STABLE_KRAMERS_EH,
                                "g_seed": SEED_G_SPREAD},
                 "environment": thermal.describe_environment(), "records": {}}
    if args.merge and out_path.is_file():
        out = json.loads(out_path.read_text())
        out.setdefault("records", {})

    keys = [k for k in args.only.split(",") if k]
    todo = [k for k in ORDER if (not keys or k in keys)]
    heartbeat = Heartbeat("sa_manifold_study", budget_seconds=args.budget,
                          meta={"systems": todo})
    deadline = time.time() + args.budget
    def complete(key: str, record: Dict) -> bool:
        if record.get("status") != "ok":
            return False
        plan = CASES[key]
        seeds = plan.get("seeds", SEEDS)
        for n_states in plan["ensembles"]:
            leg = record.get("legs", {}).get(str(n_states))
            if leg is None or leg.get("status") == "skipped" or "verdict" not in leg:
                return False
            expected = {"unperturbed"} | {"seed-{}".format(x) for x in seeds}
            if plan.get("from_guess") and n_states < int(plan["reference_n"]):
                expected.add("from-guess")
            have = {r.get("tag") for r in leg.get("runs", ())
                    if r.get("status") in ("ok", "gate-refused")}
            if not expected <= have:
                return False
        return True

    for index, key in enumerate(todo):
        if args.merge and complete(key, out["records"].get(key, {})):
            print("[sa-manifold] {:8s} already present, skipping".format(key), flush=True)
            continue
        if time.time() > deadline:
            print("[sa-manifold] budget spent; stopping before {}".format(key), flush=True)
            break
        print("[sa-manifold] {:8s} starting".format(key), flush=True)
        with thermal.track_resources() as tres:
            try:
                def save(partial, _key=key):
                    out["records"][_key] = partial
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))

                rec = run_case(EXTRA_SYSTEMS.get(key) or sysdef.SYSTEMS_BY_KEY[key],
                               CASES[key],
                               heartbeat=heartbeat, deadline=deadline,
                               legs=tuple(int(x) for x in args.legs.split(",") if x),
                               previous=out["records"].get(key), save=save)
            except Exception as exc:                                    # noqa: BLE001
                rec = {"key": key, "status": "error",
                       "error": "{}: {}".format(type(exc).__name__, exc)}
        rec["resources"] = tres.as_dict()
        out["records"][key] = rec
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
        print("[sa-manifold] {:8s} {} ({})".format(key, rec.get("status"), tres.summary()),
              flush=True)
        if rec.get("status") == "error":
            print("  [error] {}".format(rec["error"]), flush=True)
        heartbeat.tick(index + 1, system=key, status=str(rec.get("status")))

    n_err = sum(1 for r in out["records"].values() if r.get("status") != "ok")
    print("\nwrote {} ({} records, {} not ok)".format(out_path, len(out["records"]), n_err),
          flush=True)
    heartbeat.finish(records=len(out["records"]), errors=n_err)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
