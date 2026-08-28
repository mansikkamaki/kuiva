"""The ab initio ladder: local-multiplet models on real integrals.

What this run establishes (the leg deferred from the stage-6 Tier-0 validation)
-------------------------------------------------------------------------------
The manifold machinery of ``kuiva/dmrg/manifold.py`` was validated on Tier-0/model
oracles; this generator runs it on **ab initio integrals** for the reference-system ladder —
``ticl3``/``tif3`` (single d¹ site), ``ti2cl6_far`` (exact product), ``ti2cl6`` (coupled),
``ti3f9_far`` (the d¹ trimer, 1000-state manifold) — with the **same-integral conventional
CI as the oracle** (energy agreement validates the network solver at these
integrals). Per system it records:

* the conventional CI manifold (dense diagonalization of the CAS Hamiltonian — 190
  determinants for a dimer, 4060 for the trimer);
* the DMRG local-multiplet model spectrum against it: the *ground* product model
  (``dims=2`` per site) and the *full* one-electron-per-site model (``dims=10``),
  with the product-truncation error **measured** on the coupled dimer;
* additivity for the far systems (CI spectrum vs sums of the site sub-Hamiltonian
  spectra — exact at 25 Å);
* structure discovery on ``ti2cl6_far`` from a deliberately wrong seed tree (the stage-5
  validation leg (a) that moved here);
* pseudospin/Ouluspin exports with real Hamiltonian provenance (stage 7 on real data).

Orbitals, and the localization step this run needed
---------------------------------------------------
``ticl3``/``tif3`` use **converged SA-CASSCF orbitals** (minutes, ties the run to the
committed tier2 record). The dimers and the trimer run at the **scalar-guess orbitals**,
deliberately: the machinery oracle is the same-integral CI, so CASSCF-converged orbitals
would add 43 min – 4.7 h per system and change nothing about the comparison (the committed
tier2 protocol already covers "does the CASSCF reproduce its record").

⚠ **Multi-site actives are localized before the network sees them.** Canonical guess
orbitals delocalize over equivalent centres (symmetric/antisymmetric pairs at 25 Å), and
site structure *as a mode partition* does not exist in that basis — the concern recorded
in the stage-6 plan entry. The fix here: within the active space, diagonalize the Löwdin
atomic projector of each metal centre and assign whole orbitals to centres (sequential
deflation, site-blocked mode order). This is an active-active rotation, so the CI/DMRG
spectrum is exactly invariant — asserted by comparing the CI at localized vs
canonical orbitals — while the per-site Löwdin populations of the localized set are
recorded as the diagnostic.

Run discipline (bounded, incremental)
----------------
Hours, not minutes, and designed accordingly: records are rewritten after every completed
system, a hard ``--budget`` is checked between legs, a ``progress`` heartbeat ticks per
leg, systems run cheapest-first, and ``resources.BUDGET`` is cleared between systems (the
tier2_kuiva lesson). A leg that fails records its error and the sweep continues.

Usage::

    python tests/generate/manifold_ladder.py --only ticl3            # smoke, ~5 min
    python tests/generate/manifold_ladder.py --merge --budget 21600  # the sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import systems as sysdef                                                    # noqa: E402
import thermal                                                              # noqa: E402
from progress import Heartbeat                                              # noqa: E402

REF_OUT = REPO / "tests/reference/manifold_ladder.json"
DUMP_DIR = REPO / "temp/stage6_dumps"
#: Exemplar Kuiva pseudospin files for the OuluSpin interface work (writing under
#: ouluspin/temp is sanctioned; nothing outside it is ever touched).
OULUSPIN_TEMP = Path("/home/akseli/Programs/oulupy/ouluspin/temp/kuiva_examples")

SCHEMA = 1
MEMORY_GB = 11.5
DEGENERACY_TOL_CM = 1.0

#: Cheapest first: a structural failure surfaces in the first minutes.
ORDER = ("tif3", "ticl3", "ti3_far", "ti2cl6_far", "ti2cl6", "ti3f9_far")

#: Per-system plan. ``ground_dims`` is the per-site ground-multiplet dimension (2 for a
#: Kramers doublet in a ligand field, 4 for a free ion's j = 3/2 quartet — a dims=2 cut
#: there would be refused, correctly, for splitting the quartet). ``full_roots`` fixes
#: the full-model ensemble: for a *far* system nothing entangles the sites, so the
#: ensemble must cover the manifold and the honest root count IS the manifold count
#: (blind doubling overshoots into charge-transfer states, whose ensemble weight then
#: dwarfs the upper local levels — measured on the first sweep); a *coupled* system
#: starts small and lets entanglement resolution + growth do the work, which is the
#: production mechanism. ``full=None`` skips the full model with the reason recorded:
#: the far trimer's would need the 1000-root ensemble by the same far-limit argument
#: the dimer already measures.
PLAN = {
    "tif3": dict(n_sites=1, casscf=True, discovery=False, ground_dims=2,
                 full_roots=(8, 10)),
    "ticl3": dict(n_sites=1, casscf=True, discovery=False, ground_dims=2,
                  full_roots=(8, 10)),
    "ti3_far": dict(n_sites=3, casscf=False, discovery=False, ground_dims=4,
                    modes_per_node=2, max_bond=64, full_roots=None,
                    manifold=False,
                    manifold_skip="blocked by TTNO memory at 30 spinors: the compiled "
                                  "operator's W tensors store dense sector blocks that "
                                  "scale as the operator bond dimension squared "
                                  "(~1100^2 here), ~13 GB on a 15 GB machine — killed "
                                  "by the kernel OOM twice, at 5-mode and at 2-mode "
                                  "nodes. Sparse W storage is the recorded stage-8/10 "
                                  "obligation; the CI manifold and additivity legs "
                                  "below stand on their own, and the 3-site manifold "
                                  "machinery is covered Tier-0 "
                                  "(tests/test_dmrg_manifold.py three-fragment oracle)",
                    full_skip="see manifold_skip"),
    "ti2cl6_far": dict(n_sites=2, casscf=False, discovery=True, ground_dims=2,
                       full_roots=(100, 100)),
    "ti2cl6": dict(n_sites=2, casscf=True, discovery=False, ground_dims=2,
                   full_roots=(8, 128)),
    "ti3f9_far": dict(n_sites=3, casscf=False, discovery=False, ground_dims=2,
                      full_roots=None, full_skip="unreachable: the memory budget refuses the "
                      "front-end at 14.8 GB on this machine (see the record)"),
}


def _rel_cm(e: np.ndarray, n: Optional[int] = None) -> List[float]:
    from kuiva.props.multiplet import HARTREE_TO_CM
    e = np.sort(np.asarray(e, dtype=float))
    if n is not None:
        e = e[:n]
    return [round(float(x), 4) for x in (e - e[0]) * HARTREE_TO_CM]


def _pattern(rel_cm: Sequence[float], tol: float = DEGENERACY_TOL_CM) -> List[int]:
    from kuiva.props.multiplet import degenerate_blocks
    return [b for _, b in degenerate_blocks(list(rel_cm), tol_cm=tol)]


def dense_ci(h_act: np.ndarray, eri_act: np.ndarray, n_elec: int) -> np.ndarray:
    """The oracle: every eigenvalue of the CAS Hamiltonian, densely .

    Independent Slater–Condon machinery (``ci/strings.py``) — the same oracle every
    Tier-0 DMRG test uses, now on ab initio integrals. Excludes ``e_core`` like the
    network does; every comparison below is on relative energies anyway.
    """
    from kuiva.ci.strings import Determinants, hamiltonian_matrix
    n = h_act.shape[0]
    masks = np.sort(np.array([sum(1 << i for i in c)
                              for c in combinations(range(n), n_elec)],
                             dtype=np.uint64))
    dets = Determinants(masks, n_spinor=n, n_elec=n_elec)
    ham = hamiltonian_matrix(dets, h_act, eri_act).toarray()
    return np.sort(np.linalg.eigvalsh(ham))


def localize_active(reference, space, centres: Sequence[int],
                    coeff: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """Localize the active spinors onto the metal centres, one site per centre.

    ⚠ **A thin call into the package**, deliberately: this used to be a copy of the
    projection localization living in the test tree, and "which centre an orbital belongs to"
    is exactly the kind of definition that must have one implementation. The site-blocked
    column order the site-local operators need (the reconnection JW lesson) is what
    :func:`kuiva.interface.api.localize_active_space` returns, the pairs are rebuilt per
    site, and the rotation is active-active so the CI is exactly invariant.
    """
    result = api.localize_active_space(reference, space, list(centres), coeff=coeff,
                                       report=False)
    diag = {"per_site_population_min":
            [round(float(result.populations[result.site == i, i].min()), 4)
             for i in range(result.n_sites)],
            "per_site_population_mean":
            [round(float(result.populations[result.site == i, i].mean()), 4)
             for i in range(result.n_sites)]}
    return result.coeff, diag


def active_moments(reference, coeff: np.ndarray, space) -> np.ndarray:
    """``mu = -(L + g_e S)`` over the active spinors [mu_B] — what the model lifts."""
    from kuiva.props.dump import spinor_operators
    from kuiva.props.multiplet import G_ELECTRON
    from kuiva.spinor.expand import spin_operator
    l_mo, s_mo = spinor_operators(coeff, reference.data.properties.two_component(),
                                  spin_operator(reference.data.s_ao))
    act = np.asarray(space.spaces.active, dtype=int)
    ix = np.ix_(act, act)
    return np.stack([-(l_mo[k][ix] + G_ELECTRON * s_mo[k][ix]) for k in range(3)])


def site_graph(n_modes: int, n_sites: int, modes_per_node: int = 5) -> Tuple:
    """A path of multi-mode nodes with sites = contiguous node groups per centre.

    Fat nodes keep every two-site space large enough to hold a big shared-basis root set
    (an end bond of a *single-mode* path caps the ensemble at its own two-site dimension —
    measured on the far dimer: dimension 6, which no 100-root average fits). ⚠ But node
    fatness is paid in the TTNO: every label transition materializes a dense
    ``(2^m, 2^m)`` local block, and at 30 modes with 5-mode nodes that is several GB of
    *unaccounted* operator tensors — the first sweep died to the kernel's OOM killer
    there, silently (TTNO sizing functions are a recorded stage-10 obligation). The
    trimer therefore runs on 2-mode nodes.
    """
    from kuiva.dmrg import NetworkGraph
    m = int(modes_per_node)
    n_nodes = n_modes // m
    contents = [tuple(range(m * i, m * i + m)) for i in range(n_nodes)]
    graph = NetworkGraph(n_nodes, [(i, i + 1) for i in range(n_nodes - 1)], contents)
    per_site = n_nodes // n_sites
    sites = [tuple(range(per_site * k, per_site * (k + 1))) for k in range(n_sites)]
    return graph, sites


def run_manifold(terms, graph, sites, n_elec, dims, ops, *, n_roots, max_roots,
                 label, max_bond=None, rng_seed=11) -> Dict:
    """One ``solve_manifold`` leg, recorded with its cost and its honest outcome.

    ⚠ A refusal (a cut through a degenerate group) and under-resolution are both
    *recorded*, never allowed to clobber the system's other legs — the first sweep lost
    ``ti2cl6_far``'s completed CI/additivity/discovery legs to exactly that.
    """
    from kuiva.dmrg import UnderResolved, solve_manifold
    t0 = time.process_time()
    rec: Dict = {"dims": dims, "n_roots_start": n_roots, "max_roots": max_roots}
    try:
        result = solve_manifold(terms, graph, n_elec, sites=sites, rule="dimension",
                                dims=dims, operators=ops, n_roots=n_roots,
                                max_roots=max_roots, max_outer=9, outer_tol=1e-7,
                                max_bond=max_bond,
                                rng=np.random.default_rng(rng_seed))
    except UnderResolved as exc:
        rec.update(status="under-resolved", error=str(exc),
                   cpu_seconds=round(time.process_time() - t0, 2))
        return rec, None
    except ValueError as exc:
        rec.update(status="refused", error=str(exc),
                   cpu_seconds=round(time.process_time() - t0, 2))
        return rec, None
    spec = result.model.spectrum()
    rec.update(status="ok", converged=bool(result.converged),
               n_roots_final=int(result.n_roots), n_outer=int(result.n_outer),
               model_dim=int(result.model.model_dim),
               sector_dim=int(spec.size),
               roots_history=[int(s["n_roots"]) for s in result.history],
               rel_cm=_rel_cm(spec), pattern=_pattern(_rel_cm(spec)),
               site_gap_ratios=[None if not np.isfinite(sp.gap_ratio)
                                else round(float(sp.gap_ratio), 2)
                                for sp in result.model.sites],
               cpu_seconds=round(time.process_time() - t0, 2))
    print("  [{}] model {} states from {} roots ({} outer, {:.0f} CPU s)".format(
        label, spec.size, result.n_roots, result.n_outer, rec["cpu_seconds"]),
        flush=True)
    return rec, result


def compare_to_ci(rec: Dict, model_spec: np.ndarray, e_ci: np.ndarray) -> None:
    """Model vs the same-integral CI: interlacing (exact) and per-state deviation."""
    from kuiva.props.multiplet import HARTREE_TO_CM
    m = np.sort(np.asarray(model_spec, dtype=float))
    ref = np.sort(np.asarray(e_ci, dtype=float))[:m.size]
    rec["interlacing_ok"] = bool(np.all(m >= ref - 1e-8))
    dev = (m - m[0]) - (ref - ref[0])
    rec["max_dev_cm"] = round(float(np.max(np.abs(dev))) * HARTREE_TO_CM, 4)
    rec["rms_dev_cm"] = round(float(np.sqrt(np.mean(dev ** 2))) * HARTREE_TO_CM, 4)


def export_pseudospin(model, reference, space, key: str, ints, rec: Dict) -> None:
    """The stage-7 file, with real Hamiltonian provenance, for OuluSpin."""
    from kuiva.props.pseudospin import pseudospin_from_model
    provenance: Dict = {"system": key, "code": "kuiva",
                        "active_space": space.description,
                        "orbitals": rec.get("orbitals", "scalar guess"),
                        "generator": "tests/generate/manifold_ladder.py"}
    if reference.data.soc is not None:
        provenance["hamiltonian"] = reference.data.soc.provenance()
    ps = pseudospin_from_model(model, energy_shift=float(ints.e_core),
                               provenance=provenance,
                               comments=("ab initio d1 ladder run (stage 6)",))
    rec["pseudospin"] = {
        "twice_s": [s.twice_s for s in ps.sites],
        "site_g": [[round(float(g), 6) for g in s.g_values] for s in ps.sites],
        "unitarity_error": float(ps.unitarity_error()),
    }
    # the OuluSpin-native variant: one quantization axis for every site, components in
    # its principal triad (z = axis) — what a single-axis ITO consumer wants
    ps_z = pseudospin_from_model(model, energy_shift=float(ints.e_core),
                                 common_axis="ground-doublet", rotate_frame=True,
                                 provenance=provenance,
                                 comments=("ab initio d1 ladder run (stage 6); "
                                           "quantization-axis frame",))
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    path = DUMP_DIR / "{}_ground.psd".format(key)
    ps.write(path, title="{} ground-multiplet product model (stage-6 ladder)".format(key))
    ps_z.write(DUMP_DIR / "{}_ground_zframe.psd".format(key),
               title="{} ground model, quantization-axis frame".format(key))
    rec["pseudospin"]["file"] = str(path)
    try:
        OULUSPIN_TEMP.mkdir(parents=True, exist_ok=True)
        ps.write(OULUSPIN_TEMP / "{}_ground.psd".format(key),
                 title="{} ground-multiplet product model (Kuiva stage-6 ladder)"
                 .format(key))
        ps_z.write(OULUSPIN_TEMP / "{}_ground_zframe.psd".format(key),
                   title="{} ground model, quantization-axis frame (Kuiva stage-6 "
                         "ladder)".format(key))
    except OSError as exc:                                        # pragma: no cover
        print("  [warn] could not write OuluSpin exemplar: {}".format(exc), flush=True)


def run_system(system: sysdef.System, *, memory_gb: float, heartbeat, deadline) -> Dict:
    from kuiva.interface import api
    from kuiva.mcscf.orbopt import CASIntegrals
    from kuiva.util import resources as res
    from kuiva.dmrg import (NetworkGraph, ReconnectionPolicy, expansion_to_ttn,
                            compile_ttno, hamiltonian_product_terms,
                            one_electron_product_terms, solve_adaptive)
    from kuiva.mcscf.preopt import cheap_ci

    res.clear()
    plan = PLAN[system.key]
    key = system.key
    t_sys = time.time()
    rec: Dict = {"key": key, "basis": system.basis, "charge": system.charge,
                 "spin": system.spin, "ncas": system.ncas, "nelecas": system.nelecas,
                 "n_sites": plan["n_sites"], "legs": {}}

    def tick(stage):
        heartbeat.tick(int(time.time() - t_sys), system=key, stage=stage)

    def over_budget():
        return time.time() > deadline

    # --- reference and active space -------------------------------------------------------
    tick("reference")
    molecule = api.Molecule(atoms=system.atoms, basis=system.basis,
                            charge=system.charge, spin=system.spin)
    reference = api.spinor_reference(molecule, memory_gb=memory_gb)
    rec["nao"] = int(reference.data.nao)
    rec["e_scf"] = float(reference.data.e_scf)
    rec["scf_converged"] = bool(reference.data.converged)
    if reference.data.soc is not None:
        rec["hamiltonian"] = reference.data.soc.provenance()

    element = system.atoms[0][0]
    centres = [i for i, (sym, _) in enumerate(system.atoms) if sym == element]
    n_act = 2 * system.ncas
    n_elec = system.nelecas

    # --- orbitals: CASSCF where the plan says so, localized guess otherwise; multi-site
    # actives are Loewdin-localized per centre either way (an active-active rotation:
    # the CI is exactly invariant, and the network needs site-local mode labels) --------
    if plan["casscf"]:
        tick("casscf")
        # checkpointed, so a budget-interrupted sweep never repays the CASSCF —
        # the first ti2cl6 pass spent 4.8 h converging and then lost the orbitals to the
        # wall budget. mode="second-order" for the multi-site averages per the optimizer's own
        # guidance (a large state average is the recorded case for choosing it).
        DUMP_DIR.mkdir(parents=True, exist_ok=True)
        ckpt = DUMP_DIR / "{}_casscf.h5".format(key)
        kw = dict(n_states=system.soc_states, max_iter=60, conv_grad=1e-4,
                  report=False, checkpoint=str(ckpt))
        if plan["n_sites"] > 1:
            kw["mode"] = "second-order"
        if ckpt.is_file():
            outcome = api.casscf(reference, restart=str(ckpt), **kw)
        else:
            outcome = api.casscf(reference, character=(centres, "d"), n_active=n_act,
                                 n_active_elec=n_elec, **kw)
        rec["legs"]["casscf"] = {
            "converged": bool(outcome.converged),
            "iterations": int(outcome.orbital.n_iterations),
            "e_avg": float(outcome.energy),
            "ci_rel_cm": _rel_cm(np.asarray(outcome.ci.total_energies)),
        }
        coeff = np.ascontiguousarray(outcome.coeff)
        space = outcome.active
        rec["orbitals"] = "SA-CASSCF ({} roots)".format(system.soc_states)
    else:
        space = api.active_space_for(reference, character=(centres, "d"),
                                     n_active=n_act, n_active_elec=n_elec)
        coeff = None
        rec["orbitals"] = "scalar guess"
    if plan["n_sites"] > 1:
        coeff, diag = localize_active(reference, space, centres, coeff=coeff)
        rec["legs"]["localization"] = diag
        rec["orbitals"] += ", active Loewdin-localized per centre"
    elif coeff is None:
        coeff = reference.spinors_in_ao()
    coeff = np.ascontiguousarray(coeff)
    rec["active_space"] = space.description

    # --- integrals and the conventional oracle --------------------------------------------
    tick("integrals")
    ints = CASIntegrals.build(reference.factors, reference.h_one_electron(), coeff,
                              space.spaces, e_nuc=reference.data.e_nuc)
    h_act = ints.h_active_effective()
    eri_act = ints.active_eri()
    mu_act = active_moments(reference, coeff, space)
    ops = {name: one_electron_product_terms(mu_act[k])
           for k, name in enumerate(("mu_x", "mu_y", "mu_z"))}

    tick("dense-ci")
    t0 = time.process_time()
    e_ci = dense_ci(h_act, eri_act, n_elec)
    n_manifold = 10 ** plan["n_sites"]
    ci_leg = {"ndet": int(e_ci.size), "n_manifold": n_manifold,
              "rel_cm": _rel_cm(e_ci, n_manifold),
              "cpu_seconds": round(time.process_time() - t0, 2)}
    ci_leg["pattern"] = _pattern(ci_leg["rel_cm"])
    if not plan["casscf"]:
        # the localization is an active-active rotation: the CI must be exactly invariant
        c_can = reference.spinors_in_ao()
        ints_can = CASIntegrals.build(reference.factors, reference.h_one_electron(),
                                      np.ascontiguousarray(c_can), space.spaces,
                                      e_nuc=reference.data.e_nuc)
        e_can = dense_ci(ints_can.h_active_effective(), ints_can.active_eri(), n_elec)
        ci_leg["localization_invariance_eh"] = float(np.max(np.abs(e_ci - e_can)))
    rec["legs"]["ci"] = ci_leg
    print("  [{}] CI manifold: {} of {} states, pattern head {}".format(
        key, n_manifold, e_ci.size, ci_leg["pattern"][:8]), flush=True)

    # --- additivity for the far systems ---------------------------------------------------
    if "far" in key:
        n_sites = plan["n_sites"]
        per = n_act // n_sites
        site_levels = [np.linalg.eigvalsh(h_act[per * k:per * (k + 1),
                                                per * k:per * (k + 1)])
                       for k in range(n_sites)]
        sums = site_levels[0]
        for lv in site_levels[1:]:
            sums = (sums[:, None] + lv[None, :]).ravel()
        sums = np.sort(sums)
        dev = (np.sort(e_ci)[:sums.size] - np.sort(e_ci)[0]) - (sums - sums[0])
        from kuiva.props.multiplet import HARTREE_TO_CM
        rec["legs"]["additivity"] = {
            "max_dev_cm": round(float(np.max(np.abs(dev))) * HARTREE_TO_CM, 6),
            "site_span_cm": [_rel_cm(lv)[-1] for lv in site_levels],
        }

    if not plan.get("manifold", True):
        rec["legs"]["manifold_ground"] = {"status": "skipped",
                                          "reason": plan.get("manifold_skip", "")}
        rec["legs"]["manifold_full"] = {"status": "skipped",
                                        "reason": plan.get("manifold_skip", "")}
        rec["status"] = "ok"
        rec["seconds"] = round(time.time() - t_sys, 2)
        return rec

    terms = hamiltonian_product_terms(h_act, eri_act)

    # --- structure discovery from a wrong tree (far dimer; stage-5 leg (a)) --------------
    if plan["discovery"] and not over_budget():
        tick("discovery")
        t0 = time.process_time()
        per = n_act // 2
        interleaved = [m for pair in zip(range(per), range(per, n_act)) for m in pair]
        wrong = NetworkGraph(n_act, [(i, i + 1) for i in range(n_act - 1)],
                             contents=[(m,) for m in interleaved])
        ci_seed = cheap_ci(np.ascontiguousarray(h_act), np.ascontiguousarray(eri_act),
                           n_elec, n_states=4, with_2rdm=False)
        op_wrong = compile_ttno(wrong, terms)
        state = expansion_to_ttn(op_wrong, ci_seed.dets.masks, ci_seed.civecs[:, :4])
        adaptive = solve_adaptive(terms, wrong, n_elec, state=state, max_bond=24,
                                  policy=ReconnectionPolicy(rule="entropy"),
                                  boundary_check=0)
        sites_found = sorted(s.orbitals for s in adaptive.structure.sites)
        weak = adaptive.structure.weak_edges
        rec["legs"]["discovery"] = {
            "moves": len(adaptive.moves),
            "sites": [list(s) for s in sites_found],
            "sites_are_centres": sites_found == [tuple(range(per)),
                                                 tuple(range(per, n_act))],
            "inter_site_eff_dim": [int(adaptive.structure.bonds[e].eff_dim)
                                   for e in weak],
            "e_vs_ci_eh": float(np.max(np.abs(np.sort(adaptive.energies)
                                              - np.sort(e_ci)[:4]))),
            "converged": bool(adaptive.converged),
            "cpu_seconds": round(time.process_time() - t0, 2),
        }
        print("  [{}] discovery: {} moves, sites-as-centres {}".format(
            key, len(adaptive.moves),
            rec["legs"]["discovery"]["sites_are_centres"]), flush=True)

    # --- the manifold models --------------------------------------------------------------
    graph, sites = site_graph(n_act, plan["n_sites"],
                              modes_per_node=plan.get("modes_per_node", 5))
    n_sites = plan["n_sites"]

    if over_budget():
        # absence is not a record (learned when the wall budget silently ate the first
        # ti2cl6 manifold legs): a budget skip says so
        rec["legs"].setdefault("manifold_ground",
                               {"status": "skipped", "reason": "wall budget spent"})
        rec["legs"].setdefault("manifold_full",
                               {"status": "skipped", "reason": "wall budget spent"})
    if not over_budget():
        tick("manifold-ground")
        gd = int(plan["ground_dims"])
        ground_rec, ground = run_manifold(
            terms, graph, sites, n_elec, gd, ops,
            n_roots=2 ** n_sites, max_roots=min(4 * gd ** n_sites, int(e_ci.size)),
            label=key + "/ground", max_bond=plan.get("max_bond"), rng_seed=21)
        if ground is not None:
            compare_to_ci(ground_rec, ground.model.spectrum(), e_ci)
            export_pseudospin(ground.model, reference, space, key, ints, ground_rec)
        rec["legs"]["manifold_ground"] = ground_rec

    if plan["full_roots"] is None:
        rec["legs"]["manifold_full"] = {"status": "skipped",
                                        "reason": plan.get("full_skip", "")}
    elif over_budget():
        rec["legs"].setdefault("manifold_full",
                               {"status": "skipped", "reason": "wall budget spent"})
    elif True:
        tick("manifold-full")
        start, cap = plan["full_roots"]
        cap = min(int(cap), int(e_ci.size))
        full_rec, full = run_manifold(
            terms, graph, sites, n_elec, 10, ops,
            n_roots=min(int(start), cap), max_roots=cap,
            label=key + "/full", max_bond=plan.get("max_bond"), rng_seed=22)
        if full is not None:
            compare_to_ci(full_rec, full.model.spectrum(), e_ci)
        rec["legs"]["manifold_full"] = full_rec

    rec["status"] = "ok"
    rec["seconds"] = round(time.time() - t_sys, 2)
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="stage-6 ab initio manifold ladder")
    ap.add_argument("--only", default="", help="comma-separated system keys")
    ap.add_argument("--budget", type=float, default=6 * 3600.0,
                    help="hard wall budget [s], checked between legs ")
    ap.add_argument("--memory-gb", type=float, default=MEMORY_GB)
    ap.add_argument("--merge", action="store_true",
                    help="keep existing ok records and skip them")
    ap.add_argument("--force", default="",
                    help="comma-separated keys to re-run even when --merge would skip")
    ap.add_argument("--out", default=str(REF_OUT))
    args = ap.parse_args(argv)

    import kuiva
    out_path = Path(args.out)
    out: Dict = {"schema": SCHEMA, "generator": "tests/generate/manifold_ladder.py",
                 "kuiva_version": kuiva.__version__,
                 "degeneracy_tol_cm": DEGENERACY_TOL_CM,
                 "environment": thermal.describe_environment(), "records": {}}
    if args.merge and out_path.is_file():
        out = json.loads(out_path.read_text())
        out.setdefault("records", {})
        out["environment"] = thermal.describe_environment()

    keys = [k for k in args.only.split(",") if k]
    todo = [sysdef.SYSTEMS_BY_KEY[k] for k in ORDER
            if (not keys or k in keys)]
    heartbeat = Heartbeat("manifold_ladder", budget_seconds=args.budget,
                          meta={"systems": [s.key for s in todo]})
    deadline = time.time() + args.budget
    forced = {k for k in args.force.split(",") if k}
    for index, system in enumerate(todo):
        if (args.merge and system.key not in forced
                and out["records"].get(system.key, {}).get("status") == "ok"):
            print("[stage6] {:12s} already present, skipping".format(system.key),
                  flush=True)
            continue
        if time.time() > deadline:
            print("[stage6] wall budget spent; stopping before {}".format(system.key),
                  flush=True)
            break
        print("[stage6] {:12s} starting".format(system.key), flush=True)
        with thermal.track_resources() as tres:
            try:
                rec = run_system(system, memory_gb=args.memory_gb,
                                 heartbeat=heartbeat, deadline=deadline)
            except Exception as exc:                                    # noqa: BLE001
                rec = {"key": system.key, "status": "error",
                       "error": "{}: {}".format(type(exc).__name__, exc)}
        rec["resources"] = tres.as_dict()
        out["records"][system.key] = rec
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
        print("[stage6] {:12s} {} ({})".format(system.key, rec.get("status"),
                                               tres.summary()), flush=True)
        if rec.get("status") == "error":
            print("  [error] {}".format(rec["error"]), flush=True)
        if tres.throttled:
            print("  [warn] CPU thermally clamped for {:.0f}% of the run ("
                  "14.3)".format(100 * (tres.throttle_fraction or 0)), flush=True)
        heartbeat.tick(index + 1, system=system.key, status=str(rec.get("status")),
                       stage="record")

    n_err = sum(1 for r in out["records"].values() if r.get("status") != "ok")
    print("\nwrote {} ({} records, {} not ok)".format(out_path, len(out["records"]),
                                                      n_err), flush=True)
    heartbeat.finish(records=len(out["records"]), errors=n_err)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
