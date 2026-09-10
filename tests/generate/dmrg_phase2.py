"""Phase 2 of the DMRG cost/reliability campaign: the road to Tier 3.

⚠ **A study support module, not a reference generator.** Everything it writes lands in
``temp/``; nothing here is committed reference data and nothing in the default suite reads a
record it produced. Two stages share it because both put the network solver on a
*Tier-3-shaped* problem — many sites, few roots — which is the regime the single-site
ladders of Phase 1 could not reach. Both are driven by :mod:`dmrg_cost_ladder`
(``--stage s2.1`` / ``--stage s2.2``), which owns the record, the log and the heartbeat.

S2.1 — spin-model DMRG on the actual Tier-3 exchange graphs
-----------------------------------------------------------
The Tier-3 systems have no reference calculation and never will (:mod:`tier3_systems`);
what they have is an exchange graph, local spins and a set of theorems. The network
layer's operator compiler accepts a generic operator-sum input — the seam the layer's
docstring reserves for model Hamiltonians with *known* structure — so the effective
Heisenberg model of every Tier-3 graph is solved by the same two-site sweep the ab initio
path uses, one mode per paramagnetic centre, and graded against what is known exactly:

* dense ED of the same model where the coupled space is small (six of the eight systems);
* a sparse Lanczos solve of the same model inside the target ``M`` sector on the two
  eight-site rings (65 536 states — past the suite's dense cap, trivial for Lanczos);
* the Lieb–Mattis ground spin, and the experimental one, on the bipartite systems;
* the four-fold chirality × spin degeneracy of the frustrated triangle, and the analytic
  complete-graph spectrum of the cubane.

``fe4_star`` runs on both topologies: it is the minimal system where a tree should
measurably beat a chain, since every bond of the star tree is a leaf cut of rank at most
six while a chain must carry a two-site cut of rank up to 36.

⚠ **What is graded is the variational energy of the committed state, never the sweep's
local eigenvalue.** A two-site problem on a three- or four-site network spans most or all
of the system, so its Davidson eigenvalue is exact at *any* cap while the state that is
kept has been truncated behind it. The committed state is contracted to a dense vector
(every Tier-3 model admits it) and its energy and ``<S^2>`` are what the record carries;
the local eigenvalue is stored beside them so the gap between the two is visible. The
total spin is *read* from ``<S^2>``, never assumed from the sector.

S2.2 — the Tier-3-shaped ab initio TTNO
---------------------------------------
The measurement the tensor-network layer has carried as an open item since its first
multi-site attempt: whether the 30-spinor, three-site ab initio operator fits a memory
limit at all, what its compile costs, and what one sweep costs — using ``ti3f9_far``'s
real integrals, which *are* that shape. The front end is built once in the parent
process; each node partition is then measured in **its own child process** under a hard
wall budget, because a compile that does not fit ends either in a memory refusal (an
answer) or, past the plan's reach, in an allocator residue that would poison the next
partition's numbers (the residue-plus-next-compile failure is a recorded one). A child
writes its record after every step, so a budget kill still leaves the steps it completed.

S2.3 — the multi-site bridge, CI-checkable, production approximations ON
--------------------------------------------------------------------------
The two systems whose shape is Tier 3's — several localized centres, a local-multiplet
product manifold — and whose exact CI still exists: the coupled dimer ``ti2cl6``
(CAS(2, 20), 190 determinants, the 10 x 10 product) and the far trimer ``ti3f9_far``
(CAS(3, 30), 4060 determinants, the 10^3 product). Phase 1 measured truncation on
single-site multiplets; this measures it on a *product* structure, with every Tier-3
ingredient switched on rather than at saturating settings:

* **localized site orbitals** through the one site-partition implementation (an
  active-active rotation; the CI is invariant and that is asserted), and a
  **site-blocked chain** whose nodes never straddle a site — the S2.1 finding that the
  topology comes from the site partition, never from mutual information;
* **few roots, several lanes**: the ground product block (4 on the dimer, 8 on the trimer),
  the two-multiplet product (16), and the whole manifold (100) — because the two-site
  floor is what a shared-basis ensemble pays, and the lanes show where a manifold stops
  being an ensemble a network can hold;
* **truncating caps**, graded against the exact CI on the same integrals through the same
  phase-invariant reductions as Phase 1, plus the two product-structure questions Tier 3
  actually asks: does the **local-multiplet model** built from the truncated state (site
  RDM spaces, the effective Hamiltonian, the **site pseudospin g values**) survive the cap,
  and does the far trimer's exactly degenerate ground block stay one block;
* **checkpoint and restart across blocks** on every point: the first block is stopped
  after a fixed number of sweeps, the second is a *new* solver built from the file, and
  at one control cap per lane an uninterrupted solve says whether the split changed the
  fixed point;
* **environment paging**, forced at the control cap by a resident cap far below the
  environment set, bitwise against the unpaged solve;
* the **weight acceptance rule** under a binding cap, at the control cap, through the
  adaptive-topology driver from the site-blocked start — recording whether the rule moved
  a mode across a site boundary;
* **extrapolation** through the campaign's own ``E(w_disc -> 0)`` fit over the recorded
  points (``dmrg_cost_ladder.py --stage extrapolate``).

S2.4a — the first Tier-3 calculation: front end and feasibility
---------------------------------------------------------------
``mn3_linear`` — three high-spin Mn(II) d^5 sites on a path, CAS(15, 30 spinors), 1.6e8
determinants, so no exact CI exists — through the same protocol the bridges rehearsed:
the high-spin ROHF reference, the fifteen 3d orbitals by character (trap-checked
against the reference's own occupations), localized per centre, the site-blocked chain
of the finest partition that holds the six-root S = 5/2 ensemble. Measurement-first, in
two child processes so the front end's memory is never resident beside the sweep: the
front end writes the active integrals (plus the spin-operator blocks and the moment
operators the analyses need) to a file, and the ladder child compiles the operator once
and runs **bounded** sweeps per cap, recording the memory plan, the per-sweep CPU, the
six energies with their Kramers pairing, ``<S^2>`` of every root, and the local-multiplet
model at six states per site. Nothing here is a converged Tier-3 result: the deliverable
is the per-sweep cost and the projected total on which S2.4b is decided.

Run discipline (all stages): one JSON record per point written as it completes, a hard
wall budget checked between points, a heartbeat line per point, refusals and
non-convergence recorded as outcomes rather than raised.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dmrg_campaign as camp                                            # noqa: E402
import tier3_systems as t3                                              # noqa: E402

#: The bond-cap ladder of the spin-model stage, cheapest first; a lane stops at the cap
#: where the truncation no longer bites (``w_disc = 0`` below the cap), since every
#: higher rung would solve the same problem again.
SPIN_CAPS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)

#: Sweep budget and convergence for a spin-model point. The models are tiny, so the budget
#: is generous; a lane that does not converge is a finding, not a cost to be managed.
SPIN_MAX_SWEEPS = 40
SPIN_CONV_TOL = 1e-10
SPIN_DAVIDSON_TOL = 1e-9

#: Dense-vector cap for the committed-state contraction: the rings are 4^8 = 65 536.
DENSE_MAX = 70000

#: Degeneracy tolerance on state energies (J = 1 units; the models are exact rationals).
DEGENERACY_TOL = 1e-6


# ==============================================================================================
# S2.1 — spin models
# ==============================================================================================

@dataclass
class SpinCase:
    """One Tier-3 system as a network problem plus its oracle."""

    system: t3.Tier3System
    terms: list
    bases: Dict[int, object]
    n_sector: int
    twice_m: int
    oracle: Dict
    ham: object                       # scipy.sparse, the same H the network compiles
    s2: object                        # scipy.sparse S_tot^2

    @property
    def key(self) -> str:
        return self.system.key

    @property
    def dims(self) -> Tuple[int, ...]:
        return self.system.local_dims

    @property
    def roots(self) -> int:
        """The ensemble the network solves: the oracle's ground level, whole."""
        return int(self.oracle["ground_degeneracy"])


def spin_case(key: str, n_states: int = 6) -> SpinCase:
    """Everything a spin-model lane needs, built once per system."""
    system = t3.get(key)
    n_sector, twice_m = t3.target_sector(system)
    oracle = t3.sector_spectrum(system, n_sector, n_states=n_states)
    return SpinCase(system=system, terms=t3.heisenberg_product_terms(system),
                    bases=t3.spin_model_bases(system), n_sector=n_sector,
                    twice_m=twice_m, oracle=oracle, ham=t3.sparse_heisenberg(system),
                    s2=t3.sparse_total_spin_squared(system))


def dfs_order(system: t3.Tier3System) -> List[int]:
    """A chain order for the sites: depth-first from a lowest-degree site.

    For a path it is the path; for the star it puts the centre second (leaf, centre,
    leaf, leaf), so one long-range bond remains whatever the order — which is the point of
    the star; for the ring it is the ring opened at one bond.
    """
    adj = t3.adjacency(system)
    start = min(range(system.n_sites), key=lambda i: (len(adj[i]), i))
    order: List[int] = []
    seen = set()

    def visit(u: int) -> None:
        seen.add(u)
        order.append(u)
        for v in sorted(adj[u], key=lambda x: (len(adj[x]), x)):
            if v not in seen:
                visit(v)

    visit(start)
    for u in range(system.n_sites):                    # disconnected: never, guarded anyway
        if u not in seen:
            visit(u)
    return order


def spanning_tree(system: t3.Tier3System) -> List[Tuple[int, int]]:
    """A breadth-first spanning tree from the highest-degree site.

    On a tree it is the tree itself; on a graph with cycles it is one choice of which
    edges become long-range terms — stated, never searched, because topology discovery is
    not load-bearing here.
    """
    adj = t3.adjacency(system)
    root = max(range(system.n_sites), key=lambda i: (len(adj[i]), -i))
    edges: List[Tuple[int, int]] = []
    seen = {root}
    frontier = [root]
    while frontier:
        nxt = []
        for u in frontier:
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v)
                    edges.append((u, v))
                    nxt.append(v)
        frontier = nxt
    return edges


def lane_graphs(system: t3.Tier3System) -> Dict[str, object]:
    """The stated topologies for one system: a chain always; a tree where the graph has a
    branching site and few enough sites that the tree lane is affordable."""
    from kuiva.dmrg import NetworkGraph

    n = system.n_sites
    order = dfs_order(system)
    lanes: Dict[str, object] = {
        "chain": NetworkGraph(n, [(i, i + 1) for i in range(n - 1)],
                              [(order[i],) for i in range(n)])}
    tree = spanning_tree(system)
    degree = [0] * n
    for u, v in tree:
        degree[u] += 1
        degree[v] += 1
    if max(degree) >= 3 and n <= 4:
        lanes["tree"] = NetworkGraph(n, tree, [(i,) for i in range(n)])
    return lanes


def graph_record(graph) -> Dict:
    return {"edges": [list(map(int, e)) for e in graph.edges],
            "contents": [list(map(int, c)) for c in graph.contents]}


def _dense_roots(result, ttno, n_roots: int) -> List[np.ndarray]:
    """The committed roots as normalized dense vectors in the site kron basis."""
    from kuiva.dmrg import state_to_dense

    out = []
    for root in range(n_roots):
        vec = state_to_dense(result.state, ttno, root=root, max_dim=DENSE_MAX)
        norm = float(np.linalg.norm(vec))
        out.append((vec / norm, norm) if norm > 0.0 else (vec, 0.0))
    return out


def network_point(case: SpinCase, graph, cap: int, *, n_roots: Optional[int] = None,
                  max_sweeps: int = SPIN_MAX_SWEEPS, conv_tol: float = SPIN_CONV_TOL,
                  seed: int = 0, expansion: float = 0.0,
                  keep_vectors: bool = False) -> Dict:
    """One spin-model point: a fresh random start, the sweep, the committed state graded.

    ⚠ Three outcomes, all records: ``ok``; ``refused`` (the group-complete truncation
    rule declined to cut through a degenerate Schmidt group — which is what a leaf cut of
    a spin multiplet looks like); ``unconverged``. ``on_split="warn"`` because the
    state-averaging gate reads the sector label as an electron count
    (:func:`tier3_systems.target_sector` explains, and chooses the label parity so the
    warning is rare).
    """
    from kuiva.dmrg import compile_ttno, random_state, solve_ttn

    roots = case.roots if n_roots is None else int(n_roots)
    c0 = time.process_time()
    ttno = compile_ttno(graph, case.terms, bases=case.bases)
    compile_cpu = time.process_time() - c0
    state = random_state(ttno, case.n_sector, int(cap), n_roots=roots,
                         rng=np.random.default_rng(seed))
    point: Dict = {"key": case.key, "cap": int(cap), "roots": roots,
                   "n_sector": case.n_sector, "twice_m": case.twice_m,
                   "compile_cpu_s": round(compile_cpu, 3), "seed": int(seed),
                   "expansion": float(expansion)}
    t0, w0 = time.process_time(), time.time()
    try:
        result = solve_ttn(ttno, state, max_sweeps=max_sweeps, conv_tol=conv_tol,
                           davidson_tol=SPIN_DAVIDSON_TOL, max_bond=int(cap),
                           n_elec=case.n_sector, boundary_check=0, on_split="warn",
                           expansion=expansion, report=False, memory_plan=False)
    except ValueError as exc:
        point.update(status="refused", error="{}".format(exc),
                     cpu_s=round(time.process_time() - t0, 3),
                     wall_s=round(time.time() - w0, 3))
        return point
    point.update(cpu_s=round(time.process_time() - t0, 3),
                 wall_s=round(time.time() - w0, 3),
                 status="ok" if result.converged else "unconverged",
                 n_sweeps=int(result.n_sweeps), w_disc=float(result.max_discarded),
                 bond_used=int(result.max_bond_dim),
                 saturating=bool(int(result.max_bond_dim) < int(cap)
                                 and float(result.max_discarded) == 0.0),
                 e_local=[float(e) for e in result.energies],
                 bond_dimensions={"{}-{}".format(u, v): int(d)
                                  for (u, v), d in result.state.bond_dimensions().items()})
    roots_dense = _dense_roots(result, ttno, roots)
    e_state, s2_state, twice_s, norm2 = [], [], [], []
    for vec, norm in roots_dense:
        norm2.append(norm ** 2)
        if norm == 0.0:
            # ⚠ A root whose committed weight is entirely discarded: at a cap below the
            # ensemble's rank a member of a degenerate level can be truncated to nothing.
            # It has no energy and no spin, and the grade below reads that as unacceptable.
            e_state.append(None)
            s2_state.append(None)
            twice_s.append(None)
            continue
        e_state.append(float(np.real(np.vdot(vec, case.ham @ vec))))
        val, two_s = t3.total_spin_of(vec, case.system, case.s2)
        s2_state.append(val)
        twice_s.append(two_s)
    # ⚠ The ensemble's spins are the S^2 spectrum on its span, not the per-root values:
    # inside a degenerate level the roots are an arbitrary rotation of its members, and
    # a level may mix total spins (tier3_systems.spin_spectrum_of says where it does).
    live = [v for v, norm in roots_dense if norm > 0.0]
    s2_spec, twice_spec = t3.spin_spectrum_of(live, case.system, case.s2) if live \
        else ([], [])
    point.update(e_state=e_state, s2_state=s2_state, twice_s_state=twice_s,
                 kept_weight=norm2, s2_spectrum=s2_spec,
                 twice_s_spectrum=sorted(twice_spec), n_lost_roots=roots - len(live))
    if keep_vectors:
        point["_vectors"] = [vec for vec, _ in roots_dense]
    return point


def grade_point(point: Dict, case: SpinCase, bands=camp.BANDS) -> Dict:
    """The three-level grade, in the campaign's vocabulary, on a J = 1 model.

    ``unacceptable`` is any qualitative change — a wrong total spin of the committed
    ground state, a degeneracy the oracle has that the ensemble does not — and otherwise
    the tier is the *relative* energy error of the committed state against the oracle's
    ground energy (there is no absolute floor in J = 1 units; the campaign's percentage
    bands apply as they stand). A refusal has no grade.
    """
    if point.get("status") == "refused":
        return {"overall": None, "reason": "refused"}
    e0 = float(case.oracle["ground_energy"])
    twice_s0 = sorted(int(x) for x in case.oracle["ground_twice_total_spins"])
    if point.get("n_lost_roots", 0) or any(e is None for e in point["e_state"]):
        return {"overall": "unacceptable", "reason": "a root was truncated to nothing",
                "energy_rel_error": None, "energy_abs_error": None,
                "ground_spin_ok": False, "s2_deviation": None, "degeneracy_ok": False,
                "level_spread": None, "e_local_minus_e0": float(min(point["e_local"]) - e0)}
    e_state = np.asarray(point["e_state"], dtype=float)
    rel = float(np.max(np.abs(e_state - e0)) / max(abs(e0), 1e-300))
    spread = float(np.max(e_state) - np.min(e_state)) if e_state.size > 1 else 0.0
    spin_ok = sorted(int(s) for s in point["twice_s_spectrum"]) == twice_s0
    s2_exact = np.asarray([0.25 * t * (t + 2) for t in twice_s0], dtype=float)
    s2_got = np.sort(np.asarray(point["s2_spectrum"], dtype=float))
    s2_dev = float(np.max(np.abs(s2_got - np.sort(s2_exact)))) \
        if s2_got.size == s2_exact.size else float("nan")
    # ⚠ The degeneracy statement is judged at the *qualitative* energy resolution: two
    # members of a level split by less than the truncation error are the same level seen
    # through a truncated state, not a broken degeneracy.
    degeneracy_ok = spread <= max(bands.energy_qual_rel * abs(e0), DEGENERACY_TOL)
    if not spin_ok or not degeneracy_ok or point.get("status") != "ok":
        tier = "unacceptable"
    elif rel <= bands.energy_quant_rel:
        tier = "quantitative"
    elif rel <= bands.energy_qual_rel:
        tier = "qualitative"
    else:
        tier = "unacceptable"
    return {"overall": tier, "energy_rel_error": rel,
            "energy_abs_error": float(np.max(np.abs(e_state - e0))),
            "ground_spin_ok": bool(spin_ok), "s2_deviation": s2_dev,
            "degeneracy_ok": bool(degeneracy_ok), "level_spread": spread,
            "e_local_minus_e0": float(np.min(point["e_local"]) - e0)}


# --- the entanglement-driven ordering against the known graph ----------------------------

def _reduced_density(vec: np.ndarray, dims: Sequence[int], keep: Sequence[int]) -> np.ndarray:
    """Reduced density matrix of the sites ``keep`` from a dense site-kron vector."""
    psi = np.asarray(vec).reshape(tuple(dims))
    keep = list(keep)
    other = [i for i in range(len(dims)) if i not in keep]
    a = np.transpose(psi, keep + other)
    dk = int(np.prod([dims[i] for i in keep]))
    a = a.reshape(dk, -1)
    return a @ a.conj().T


def _von_neumann(rho: np.ndarray) -> float:
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-14]
    return float(-np.sum(w * np.log(w)))


def site_mutual_information(vectors: Sequence[np.ndarray],
                            dims: Sequence[int]) -> np.ndarray:
    """``I_ij = S_i + S_j - S_ij`` of the ensemble's averaged density, site by site.

    Sites rather than orbitals: the network nodes here *are* the paramagnetic centres, so
    this is the matrix the layer's topology guess would be grown from, with the ensemble
    density in place of the fermionic single- and two-orbital densities of
    :mod:`kuiva.rdm.entropy`.
    """
    n = len(dims)
    w = 1.0 / len(vectors)
    single = []
    for i in range(n):
        rho = sum(w * _reduced_density(v, dims, [i]) for v in vectors)
        single.append(_von_neumann(rho))
    info = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho = sum(w * _reduced_density(v, dims, [i, j]) for v in vectors)
            info[i, j] = info[j, i] = max(0.0, single[i] + single[j] - _von_neumann(rho))
    return info


def ordering_report(case: SpinCase, vectors: Sequence[np.ndarray]) -> Dict:
    """What the entanglement heuristics make of the converged state, against the graph.

    The Fiedler order (:func:`kuiva.rdm.entropy.fiedler_order`) should walk the exchange
    graph — every consecutive pair an edge — on a path, and the mutual-information tree
    (:func:`kuiva.dmrg.guess.topology_from_mutual_information`) should *be* the exchange
    graph on a tree. Both statements are checked against the definition, never against
    the network's own output.
    """
    from kuiva.dmrg import topology_from_mutual_information
    from kuiva.rdm.entropy import fiedler_order

    system = case.system
    info = site_mutual_information(vectors, case.dims)
    order = [int(x) for x in fiedler_order(info)]
    edges = {(min(a, b), max(a, b)) for a, b in system.edges}
    walk_is_path = all((min(a, b), max(a, b)) in edges for a, b in zip(order, order[1:]))
    guess = topology_from_mutual_information(info)
    mi_edges = {(min(a, b), max(a, b)) for a, b in guess.graph.edges}
    # ⚠ Two readings of the same matrix. The default guess clusters modes into *sites*
    # and chains the modes inside a site, so on a strongly entangled few-site model every
    # centre lands in one cluster and the "tree" is the Fiedler chain by construction —
    # the reading an orbital-level guess gives. With the centres declared as sites (the
    # ab initio route gets them from the localization, never from this clustering) the
    # guess builds only the inter-site tree, which is the reading a Tier-3 topology
    # question asks for.
    per_site = topology_from_mutual_information(info, site_split=2.0)
    site_edges = {(min(a, b), max(a, b)) for a, b in per_site.graph.edges}
    entropies = [float(_von_neumann(sum(_reduced_density(v, case.dims, [i])
                                        for v in vectors) / len(vectors)))
                 for i in range(system.n_sites)]
    return {"mutual_information": info.round(6).tolist(),
            "site_entropies": [round(x, 6) for x in entropies],
            "fiedler_order": order, "fiedler_walks_the_graph": bool(walk_is_path),
            "mi_tree_edges": sorted(list(e) for e in mi_edges),
            "mi_tree_is_the_graph": bool(mi_edges == edges),
            "mi_tree_within_the_graph": bool(mi_edges <= edges),
            "mi_site_tree_edges": sorted(list(e) for e in site_edges),
            "mi_site_tree_is_the_graph": bool(site_edges == edges),
            "mi_site_tree_within_the_graph": bool(site_edges <= edges),
            "exchange_edges": sorted(list(e) for e in edges),
            "strongest_pair": [int(x) for x in np.unravel_index(int(np.argmax(info)),
                                                                 info.shape)]}


def stage_spin_models(record, heartbeat, *, deadline: float,
                      keys: Optional[Sequence[str]] = None,
                      caps: Optional[Sequence[int]] = None,
                      max_sweeps: int = SPIN_MAX_SWEEPS) -> None:
    """S2.1: every Tier-3 exchange graph through the network solver, graded."""
    from kuiva.util import resources as res

    res.ensure_configured()
    ladder = list(SPIN_CAPS if caps is None else caps)
    systems = [s.key for s in t3.SYSTEMS] if keys is None else list(keys)
    already = record.done_points()
    n_point = 0
    for key in systems:
        if time.time() > deadline:
            record.data.setdefault("stopped_early", []).append(
                "{}: stage wall budget exhausted before it started".format(key))
            record.flush()
            break
        t_job = time.time()
        case = spin_case(key)
        system = case.system
        lanes = lane_graphs(system)
        lm = t3.lieb_mattis_twice_spin(system)
        ed = t3.heisenberg_ground_state(system, structural=True)
        job = {"key": key, "stage_kind": "spin-model", "label": system.label,
               "topology": system.topology, "network_target": system.network_target,
               "local_dims": list(system.local_dims), "hilbert_dim": system.hilbert_dim,
               "edges": [list(e) for e in system.edges],
               "structural_model": bool(not system.isotropic_exchange
                                        or any(s.kind == "ion_soc" for s in system.sites)),
               "sector": {"n": case.n_sector, "twice_m": case.twice_m,
                          "dim": case.oracle["sector_dim"]},
               "oracle": {k: v for k, v in case.oracle.items() if k != "energies"},
               "oracle_energies": case.oracle["energies"],
               "lieb_mattis_twice_spin": lm,
               "experimental_twice_spin": system.experimental_twice_spin,
               "dense_ed": ed, "roots": case.roots,
               "lanes": {name: graph_record(g) for name, g in lanes.items()},
               "sweep_budget": int(max_sweeps), "conv_tol": SPIN_CONV_TOL}
        if not record.has_job(key):
            record.add_job(job)
        print("  [{}] {} sites {}, sector N={} (2M={}) dim {}, oracle E0={:.8f} "
              "deg {} 2S={} ({}); LM 2S={}".format(
                  key, system.n_sites, list(system.local_dims), case.n_sector,
                  case.twice_m, case.oracle["sector_dim"], case.oracle["ground_energy"],
                  case.oracle["ground_degeneracy"], case.oracle["ground_twice_total_spins"],
                  case.oracle["method"], lm), flush=True)
        for name, graph in lanes.items():
            best_vectors = None
            for cap in ladder:
                if (key, name, int(cap)) in already:
                    print("  [{}/{}] D={:4d}  already measured; kept".format(key, name, cap),
                          flush=True)
                    continue
                if time.time() > deadline:
                    record.add_point({"key": key, "topology": name, "cap": int(cap),
                                      "status": "skipped",
                                      "reason": "stage wall budget exhausted"})
                    break
                point = network_point(case, graph, cap, max_sweeps=max_sweeps,
                                      keep_vectors=True)
                vectors = point.pop("_vectors", None)
                point["topology"] = name
                point["grade"] = grade_point(point, case)
                point["elapsed_s"] = round(time.time() - t_job, 1)
                record.add_point(point)
                n_point += 1
                heartbeat.tick(n_point, system=key, topology=name, cap=int(cap),
                               status=point["status"],
                               grade=point["grade"].get("overall") or "-")
                _print_spin_point(key, name, cap, point, case)
                if point["status"] == "ok":
                    best_vectors = vectors
                if point["status"] == "ok" and point["saturating"]:
                    record.add_point({"key": key, "topology": name, "cap": None,
                                      "status": "ladder-complete",
                                      "reason": "exact at D = {} (bond used {})".format(
                                          cap, point["bond_used"])})
                    break
            if best_vectors is not None and (key, name, "ordering") not in already:
                rep = ordering_report(case, best_vectors)
                rep.update(key=key, topology=name, cap="ordering", status="ok")
                record.add_point(rep)
                print("  [{}/{}] ordering: Fiedler {} walks graph={}, MI tree = graph {} "
                      "(within {}); sites-as-centres tree = graph {} (within {}); site "
                      "entropies {}".format(
                          key, name, rep["fiedler_order"], rep["fiedler_walks_the_graph"],
                          rep["mi_tree_is_the_graph"], rep["mi_tree_within_the_graph"],
                          rep["mi_site_tree_is_the_graph"],
                          rep["mi_site_tree_within_the_graph"],
                          [round(x, 3) for x in rep["site_entropies"]]), flush=True)


def _print_spin_point(key: str, name: str, cap: int, point: Dict, case: SpinCase) -> None:
    if point["status"] == "refused":
        print("  [{}/{}] D={:4d}  REFUSED: {}".format(key, name, cap,
                                                     point["error"][:90]), flush=True)
        return
    g = point["grade"]
    if g.get("energy_rel_error") is None:
        print("  [{}/{}] D={:4d}  {} roots lost to the truncation; {} -> {}".format(
            key, name, cap, point.get("n_lost_roots"), point["status"], g["overall"]),
            flush=True)
        return
    print("  [{}/{}] D={:4d}  E_state={:+.8f} (dE {:+.2e}, rel {:.1e})  E_loc-E0={:+.1e}  "
          "2S={}  w={:.1e} used={} sweeps={} {}  {:.1f} CPU s  -> {}".format(
              key, name, cap, min(point["e_state"]), min(point["e_state"])
              - case.oracle["ground_energy"], g["energy_rel_error"],
              g["e_local_minus_e0"], point["twice_s_spectrum"], point["w_disc"],
              point["bond_used"], point["n_sweeps"], point["status"], point["cpu_s"],
              g["overall"]), flush=True)


def summarize_spin_models(record_path: Path) -> List[Dict]:
    """Per lane: the smallest cap at each tier, the exact cap, and the ordering verdicts."""
    data = json.loads(Path(record_path).read_text())
    jobs = {j["key"]: j for j in data.get("jobs", [])}
    lanes: Dict[Tuple[str, str], List[Dict]] = {}
    ordering: Dict[Tuple[str, str], Dict] = {}
    for p in data.get("points", []):
        if p.get("cap") == "ordering":
            ordering[(p["key"], p["topology"])] = p
        elif p.get("cap") is not None and p.get("status") != "skipped":
            lanes.setdefault((p["key"], p["topology"]), []).append(p)
    rows = []
    for (key, topo), pts in sorted(lanes.items()):
        pts = sorted(pts, key=lambda p: p["cap"])
        job = jobs.get(key, {})

        def first(tier):
            for p in pts:
                if p.get("grade", {}).get("overall") == tier:
                    return p["cap"], p["cpu_s"]
            return None, None

        exact = next((p["cap"] for p in pts if p.get("status") == "ok" and p["saturating"]),
                     None)
        ok = [p for p in pts if p.get("status") == "ok"]
        rows.append({"key": key, "topology": topo,
                     "sites": job.get("hilbert_dim"),
                     "roots": job.get("roots"),
                     "refused": [p["cap"] for p in pts if p.get("status") == "refused"],
                     "unconverged": [p["cap"] for p in pts
                                     if p.get("status") == "unconverged"],
                     "wrong_spin": [p["cap"] for p in ok
                                    if not p["grade"]["ground_spin_ok"]],
                     "quantitative": first("quantitative"),
                     "qualitative": first("qualitative"),
                     "exact_at": exact,
                     "ordering": {k: v for k, v in ordering.get((key, topo), {}).items()
                                  if k in ("fiedler_order", "fiedler_walks_the_graph",
                                           "mi_tree_is_the_graph",
                                           "mi_tree_within_the_graph",
                                           "mi_site_tree_is_the_graph",
                                           "mi_site_tree_within_the_graph")}})
    return rows


def print_spin_summary(rows: Sequence[Dict]) -> None:
    head = "{:<12s} {:<6s} {:>5s} {:>7s}  {:<14s} {:<14s} {:<8s} {:<10s} {:<14s} {}".format(
        "system", "topo", "roots", "exact@", "quantitative", "qualitative", "refused",
        "wrong-spin", "unconverged", "ordering")
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        def fmt(pair):
            return "-" if pair[0] is None else "D={} ({:.1f}s)".format(pair[0], pair[1])
        o = r["ordering"]
        otxt = "Fiedler {}{}{}".format(
            "walks" if o.get("fiedler_walks_the_graph") else "NO",
            ", MI tree = graph" if o.get("mi_tree_is_the_graph")
            else (", MI tree in graph" if o.get("mi_tree_within_the_graph") else ""),
            ", site tree = graph" if o.get("mi_site_tree_is_the_graph")
            else (", site tree in graph" if o.get("mi_site_tree_within_the_graph")
                  else ", site tree NOT in graph")) if o else "-"
        print("{:<12s} {:<6s} {:>5s} {:>7s}  {:<14s} {:<14s} {:<8s} {:<10s} {:<14s} {}".format(
            r["key"], r["topology"], str(r["roots"]), str(r["exact_at"]),
            fmt(r["quantitative"]), fmt(r["qualitative"]),
            ",".join(map(str, r["refused"])) or "-",
            ",".join(map(str, r["wrong_spin"])) or "-",
            ",".join(map(str, r["unconverged"])) or "-", otxt))


# ==============================================================================================
# S2.2 — the 30-spinor ab initio TTNO
# ==============================================================================================

#: The campaign system whose integrals are the smallest Tier-3 shape.
FEASIBILITY_SYSTEM = "ti3f9_far"

#: Modes per node of the site-blocked chains measured, finest first: 15 x 2 (what the
#: manifold ladder ran the trimer at when the dense W killed it), 10 x 3, 6 x 5 (its
#: first attempt), 3 x 10 (one node per site — the local-multiplet shape, local
#: dimension 1024).
FEASIBILITY_PARTITIONS = (2, 3, 5, 10)

#: Bond caps and roots of the bounded sweeps. Modest on purpose: this measures the cost
#: *shape*, not a converged energy. ⚠ The roots are the trimer's ground product manifold,
#: whole — three Kramers doublets, 2 x 2 x 2 = 8 states at one energy — because an
#: ensemble cut inside a degenerate level is neither solvable (a Krylov solver cannot
#: separate two roots of an eight-fold level; the two-root oracle stalled at a residual of
#: 7e-5) nor the Tier-3 shape, whose "few roots" are a site-product manifold.
FEASIBILITY_CAPS = (8, 16)
FEASIBILITY_ROOTS = 8
FEASIBILITY_SWEEPS = 2


def feasibility_integrals_path() -> Path:
    return camp.WORK / "{}_active_integrals.npz".format(FEASIBILITY_SYSTEM)


def build_feasibility_integrals(record, heartbeat) -> Dict:
    """Front end, per-centre localized active space, the active integrals to disk, and the
    exact CI of the low manifold — once, in the parent process."""
    from kuiva.interface import api
    from kuiva.util import resources as res

    system = camp.get(FEASIBILITY_SYSTEM)
    res.clear()
    t0, c0 = time.time(), time.process_time()
    reference = camp.build_reference(system)
    front = {"nao": int(reference.data.nao), "e_scf": float(reference.data.e_scf),
             "scf_converged": bool(reference.data.converged),
             "front_end_wall_s": round(time.time() - t0, 1),
             "front_end_cpu_s": round(time.process_time() - c0, 1)}
    if reference.data.soc is not None:
        front["hamiltonian"] = reference.data.soc.provenance()
    heartbeat.tick(0, system=FEASIBILITY_SYSTEM, stage="front-end")
    space = api.active_space_for(reference, **system.selection())
    centres = [i for i, (sym, _) in enumerate(system.atoms) if sym == system.element]
    # ⚠ Site-blocked orbitals, through the one site-partition implementation: the
    # Tier-3 shape is "one node per centre", and it is only that shape if the modes a
    # node holds are one centre's orbitals. An active-active rotation, so the exact CI
    # below is invariant to it (asserted, not assumed).
    loc = api.localize_active_space(reference, space, centres, report=False)
    coeff = np.ascontiguousarray(loc.coeff)
    front["localization"] = {
        "sites": [int(x) for x in loc.site],
        "per_site_population_min": [round(float(loc.populations[loc.site == i, i].min()), 4)
                                    for i in range(loc.n_sites)]}
    ints = camp.cas_integrals(reference, coeff, space)
    h = np.ascontiguousarray(ints.h_active_effective())
    eri = np.ascontiguousarray(ints.active_eri())
    heartbeat.tick(0, system=FEASIBILITY_SYSTEM, stage="integrals")
    ladder = camp.replace(system, ladder_states=FEASIBILITY_ROOTS)
    from kuiva.util.errors import SolverFailure
    try:
        e_ci, _, ci_cost = camp.exact_ci(ints, ladder)
        ints_can = camp.cas_integrals(reference,
                                      np.ascontiguousarray(reference.spinors_in_ao()), space)
        e_can, _, _ = camp.exact_ci(ints_can, ladder)
        front["exact_ci"] = {"energies": [float(e) for e in e_ci], **ci_cost,
                             "localization_invariance_eh":
                                 float(np.max(np.abs(e_ci - e_can)))}
    except SolverFailure as exc:
        # ⚠ The oracle is a courtesy here, not the measurement: the stage measures the
        # operator and the sweep, and a CI that did not converge is recorded, not fatal.
        e_ci = np.full(FEASIBILITY_ROOTS, np.nan)
        front["exact_ci"] = {"status": "unconverged", "error": "{}".format(exc),
                             "ndet": int(system.n_det) or None}
    front["active_space"] = space.description
    path = feasibility_integrals_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, h=h, eri=eri, e_core=float(ints.e_core), n_elec=int(system.n_active_elec),
             sites=np.asarray(loc.site, dtype=np.int64), e_ci=np.asarray(e_ci))
    front["integrals"] = str(path)
    return front


class _ActiveIntegrals:
    """The duck-typed ``CASIntegrals`` surface the network layer reads, from the file."""

    def __init__(self, path: Path) -> None:
        data = np.load(path)
        self._h = np.ascontiguousarray(data["h"])
        self._eri = np.ascontiguousarray(data["eri"])
        self.e_core = float(data["e_core"])
        self.n_elec = int(data["n_elec"])
        self.sites = [int(x) for x in data["sites"]]
        self.e_ci = np.asarray(data["e_ci"], dtype=float)

    def h_active_effective(self):
        return self._h

    def active_eri(self):
        return self._eri


def partition_graph(n: int, modes_per_node: int):
    """A site-blocked chain of ``n / modes_per_node`` nodes over ascending modes."""
    from kuiva.dmrg import NetworkGraph

    m = int(modes_per_node)
    if n % m:
        raise ValueError("{} modes do not split into nodes of {}".format(n, m))
    k = n // m
    return NetworkGraph(k, [(i, i + 1) for i in range(k - 1)],
                        [tuple(range(m * i, m * i + m)) for i in range(k)])


def feasibility_child(ints_path: Path, modes_per_node: int, out_path: Path, *,
                      caps: Sequence[int], n_roots: int, max_sweeps: int,
                      budget: float) -> int:
    """One partition, measured in this process and written to ``out_path`` step by step."""
    from progress import Heartbeat
    from dmrg_memory_plan import PhaseSampler, instrument, rss_gb
    from kuiva.dmrg import TTNOTemplate, random_state, solve_ttn
    from kuiva.dmrg.plan import network_memory_plan
    from kuiva.dmrg.sweep import environment_gb, state_gb
    from kuiva.util import resources as res

    lims = res.ensure_configured()
    deadline = time.time() + float(budget)
    ints = _ActiveIntegrals(ints_path)
    n = ints._h.shape[0]
    graph = partition_graph(n, modes_per_node)
    rec: Dict = {"modes_per_node": int(modes_per_node), "n_nodes": graph.n_nodes,
                 "graph": graph_record(graph), "n_modes": n, "n_elec": ints.n_elec,
                 "memory_limit_gb": float(lims.memory_gb), "budget_s": float(budget),
                 "pid": os.getpid(), "status": "running", "steps": {}}
    heartbeat = Heartbeat("dmrg_cost_ladder_s2.2_m{}".format(modes_per_node),
                          budget_seconds=budget,
                          meta={"modes_per_node": modes_per_node, "pid": os.getpid()})

    def flush(status: Optional[str] = None) -> None:
        if status is not None:
            rec["status"] = status
        rec["elapsed_s"] = round(time.time() - t_start, 1)
        tmp = out_path.with_suffix(".json.part")
        with open(tmp, "w") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True, default=_jsonable)
        os.replace(tmp, out_path)

    sampler = PhaseSampler(res.BUDGET)
    instrument(sampler)
    sampler.start()
    t_start = time.time()
    rec["rss_baseline_gb"] = round(sampler.baseline, 4)
    flush()

    # --- the compile -----------------------------------------------------------------------
    t0, c0 = time.time(), time.process_time()
    try:
        template = TTNOTemplate(graph)
    except res.MemoryLimitError as exc:
        rec["steps"]["compile"] = {"status": "refused-memory",
                                   "error": "{}".format(exc).splitlines()[0],
                                   "wall_s": round(time.time() - t0, 1),
                                   "cpu_s": round(time.process_time() - c0, 1),
                                   "phases": sampler.table()}
        flush("refused-memory")
        heartbeat.finish(status="refused-memory")
        return 0
    op = template.ttno
    rec["steps"]["compile"] = {
        "status": "ok", "wall_s": round(time.time() - t0, 1),
        "cpu_s": round(time.process_time() - c0, 1),
        "n_terms": int(template.n_terms),
        "stored_gb": round(op.nbytes / 1024.0 ** 3, 4),
        "dense_equivalent_gb": round(op.dense_nbytes / 1024.0 ** 3, 4),
        "nnz": int(op.nnz),
        "operator_bond_dimensions": {"{}-{}".format(u, v): int(d)
                                     for (u, v), d in op.bond_dimensions().items()},
        "max_operator_bond": int(max(op.bond_dimensions().values())),
        "ledger_resident_gb": round(res.BUDGET.resident_gb(), 4),
        "rss_after_gb": round(rss_gb(), 4),
        "phases": sampler.table()}
    heartbeat.tick(1, stage="compiled", cpu=rec["steps"]["compile"]["cpu_s"])
    flush()
    print("    m={} compile: {:.0f} CPU s, stored {:.3f} GB (dense {:.1f} GB), max op bond "
          "{}, RSS {:.2f} GB, ledger {:.2f} GB".format(
              modes_per_node, rec["steps"]["compile"]["cpu_s"],
              rec["steps"]["compile"]["stored_gb"],
              rec["steps"]["compile"]["dense_equivalent_gb"],
              rec["steps"]["compile"]["max_operator_bond"],
              rec["steps"]["compile"]["rss_after_gb"],
              rec["steps"]["compile"]["ledger_resident_gb"]), flush=True)

    # --- the refill ------------------------------------------------------------------------
    t0, c0 = time.time(), time.process_time()
    try:
        ttno = template.fill(ints.h_active_effective(), ints.active_eri())
    except res.MemoryLimitError as exc:
        rec["steps"]["fill"] = {"status": "refused-memory",
                                "error": "{}".format(exc).splitlines()[0]}
        flush("refused-memory")
        heartbeat.finish(status="refused-memory")
        return 0
    rec["steps"]["fill"] = {"status": "ok", "wall_s": round(time.time() - t0, 2),
                            "cpu_s": round(time.process_time() - c0, 2),
                            "rss_after_gb": round(rss_gb(), 4)}
    flush()

    # --- per cap: the state, the plan, the bounded sweeps ------------------------------------
    rec["points"] = []
    for cap in caps:
        if time.time() > deadline:
            rec["points"].append({"cap": int(cap), "status": "skipped",
                                  "reason": "partition wall budget exhausted"})
            flush()
            break
        point: Dict = {"cap": int(cap), "roots": int(n_roots)}
        sampler.reset()
        t0, c0 = time.time(), time.process_time()
        try:
            state = random_state(ttno, ints.n_elec, int(cap), n_roots=int(n_roots),
                                 rng=np.random.default_rng(0))
        except res.MemoryLimitError as exc:
            point.update(status="refused-memory", where="state",
                         error="{}".format(exc).splitlines()[0])
            rec["points"].append(point)
            flush()
            continue
        point["state_gb"] = round(state_gb(state), 5)
        point["state_bond_dimensions"] = {"{}-{}".format(u, v): int(d) for (u, v), d
                                          in state.bond_dimensions().items()}
        # The plan, as numbers, before the sweep is asked to honour it: what the two-site
        # solves and the environments will take, with and without the RDM contraction a
        # CASSCF would add — so the record says what limit each would need.
        try:
            phases = network_memory_plan(ttno, state, n_roots=int(n_roots),
                                         max_bond=int(cap), rdm=False)
            plan_sweep = float(res.plan_peak_gb(phases))
            phases_rdm = network_memory_plan(ttno, state, n_roots=int(n_roots),
                                             max_bond=int(cap), rdm=True)
            plan_rdm = float(res.plan_peak_gb(phases_rdm))
            point["plan"] = {
                "sweep_peak_gb": round(plan_sweep, 4),
                "with_rdm_peak_gb": round(plan_rdm, 4),
                "environments_gb": round(environment_gb(ttno, state), 4),
                "phases": [{"phase": ph.name,
                            "allocations": [{"label": a.label, "gb": round(a.gb, 4),
                                             "resident": bool(a.resident)}
                                            for a in ph.allocations]}
                           for ph in phases_rdm],
                "fits_limit": bool(plan_sweep <= lims.memory_gb),
                "fits_limit_with_rdm": bool(plan_rdm <= lims.memory_gb),
                "plan_cpu_s": round(time.process_time() - c0, 2)}
        except Exception as exc:                          # the plan itself may not fit
            point["plan"] = {"status": "failed", "error": "{}: {}".format(
                type(exc).__name__, exc)[:300]}
        flush()
        heartbeat.tick(2, stage="planned", cap=int(cap))
        print("    m={} D={}: plan peak {} GB (with RDMs {} GB) against {:.1f} GB".format(
            modes_per_node, cap, point["plan"].get("sweep_peak_gb", "?"),
            point["plan"].get("with_rdm_peak_gb", "?"), lims.memory_gb), flush=True)
        sweep_times: List[Dict] = []
        t_sw = [time.time()]
        c_sw = [time.process_time()]

        def on_sweep(state, sweep, energies, converged):
            now, cpu = time.time(), time.process_time()
            sweep_times.append({"sweep": int(sweep), "wall_s": round(now - t_sw[-1], 1),
                                "cpu_s": round(cpu - c_sw[-1], 1),
                                "energies": [float(e) + ints.e_core for e in energies],
                                "rss_gb": round(rss_gb(), 3)})
            t_sw.append(now)
            c_sw.append(cpu)

        t0, c0 = time.time(), time.process_time()
        try:
            result = solve_ttn(ttno, state, max_sweeps=int(max_sweeps), conv_tol=1e-12,
                               max_bond=int(cap), n_elec=ints.n_elec, boundary_check=0,
                               on_split="warn", checkpoint=on_sweep, memory_plan=True,
                               plan_rdms=False, report=True)
        except res.MemoryLimitError as exc:
            point.update(status="refused-memory", where="sweep",
                         error="{}".format(exc).splitlines()[0],
                         wall_s=round(time.time() - t0, 1),
                         cpu_s=round(time.process_time() - c0, 1),
                         sweeps=sweep_times, phases=sampler.table())
            rec["points"].append(point)
            flush()
            print("    m={} D={}: REFUSED by the memory plan: {}".format(
                modes_per_node, cap, point["error"][:100]), flush=True)
            continue
        point.update(status="ok" if result.converged else "bounded",
                     wall_s=round(time.time() - t0, 1),
                     cpu_s=round(time.process_time() - c0, 1),
                     n_sweeps=int(result.n_sweeps), w_disc=float(result.max_discarded),
                     bond_used=int(result.max_bond_dim),
                     energies=[float(e) + ints.e_core for e in result.energies],
                     e_ci=[float(e) for e in ints.e_ci[:int(n_roots)]],
                     sweeps=sweep_times, phases=sampler.table(),
                     rss_peak_gb=round(max(r["peak_rss_gb"] for r in sampler.table()), 3),
                     ledger_resident_gb=round(res.BUDGET.resident_gb(), 4))
        rec["points"].append(point)
        flush()
        heartbeat.tick(3, stage="swept", cap=int(cap), cpu=point["cpu_s"])
        print("    m={} D={}: {} sweeps, {:.0f} CPU s ({:.0f} per sweep), peak RSS {:.2f} "
              "GB, E0 {:+.6f} vs CI {:+.6f}".format(
                  modes_per_node, cap, point["n_sweeps"], point["cpu_s"],
                  point["cpu_s"] / max(1, point["n_sweeps"]), point["rss_peak_gb"],
                  point["energies"][0], point["e_ci"][0]), flush=True)
    sampler.stop()
    flush("done")
    heartbeat.finish(status="done")
    return 0


def stage_feasibility(record, heartbeat, *, deadline: float,
                      partitions: Optional[Sequence[int]] = None,
                      caps: Optional[Sequence[int]] = None,
                      n_roots: int = FEASIBILITY_ROOTS,
                      max_sweeps: int = FEASIBILITY_SWEEPS) -> None:
    """S2.2: the front end once, then one child process per node partition."""
    parts = list(FEASIBILITY_PARTITIONS if partitions is None else partitions)
    ladder = list(FEASIBILITY_CAPS if caps is None else caps)
    ints_path = feasibility_integrals_path()
    if record.has_job(FEASIBILITY_SYSTEM) and ints_path.is_file():
        print("  [{}] front end already built; integrals at {}".format(
            FEASIBILITY_SYSTEM, ints_path), flush=True)
    else:
        front = build_feasibility_integrals(record, heartbeat)
        system = camp.get(FEASIBILITY_SYSTEM)
        record.add_job({"key": FEASIBILITY_SYSTEM, "stage_kind": "feasibility",
                        "label": system.label, "n_active": system.n_active,
                        "n_active_elec": system.n_active_elec,
                        "protocol_note": system.protocol_note, "partitions": parts,
                        "caps": ladder, "roots": int(n_roots), "sweeps": int(max_sweeps),
                        **front})
        ci = front["exact_ci"]
        print("  [{}] front end: {} AOs, SCF {} E = {:.6f}; localized populations {}; "
              "CI {}".format(
                  FEASIBILITY_SYSTEM, front["nao"],
                  "converged" if front["scf_converged"] else "NOT CONVERGED",
                  front["e_scf"], front["localization"]["per_site_population_min"],
                  ("{} dets, {} in {:.1f} CPU s; invariance {:.1e}".format(
                      ci["ndet"], ["{:.6f}".format(e) for e in ci["energies"]],
                      ci["cpu_s"], ci["localization_invariance_eh"])
                   if "energies" in ci else "UNCONVERGED: {}".format(ci["error"][:120]))),
              flush=True)
    out_dir = camp.RECORDS / "s2.2"
    out_dir.mkdir(parents=True, exist_ok=True)
    already = record.done_points()
    n_point = 0
    for m in parts:
        if (FEASIBILITY_SYSTEM, "m{}".format(m), int(m)) in already:
            print("  [m={}] already measured; kept".format(m), flush=True)
            continue
        remaining = deadline - time.time()
        if remaining < 60.0:
            record.add_point({"key": FEASIBILITY_SYSTEM, "topology": "m{}".format(m),
                              "cap": int(m), "status": "skipped",
                              "reason": "stage wall budget exhausted"})
            break
        # ⚠ Each partition gets an equal share of what is left, and the share is the
        # child's own deadline as well as the parent's kill: a compile the child cannot
        # interrupt is ended from outside, and the record it wrote up to then stands.
        share = remaining / max(1, len([p for p in parts if
                                        (FEASIBILITY_SYSTEM, "m{}".format(p), int(p))
                                        not in already and p >= m]))
        out_path = out_dir / "m{}.json".format(m)
        # ⚠ A child record that already ended is a completed measurement and is adopted,
        # never re-run: a partition measured on its own (after the parent was killed for
        # the machine's memory with the front end still resident beside the child) is
        # the same measurement this loop would make, minus the parent's 2.7 GB.
        if out_path.is_file():
            try:
                child = json.loads(out_path.read_text())
            except ValueError:
                child = {}
            if child.get("status") not in (None, "running"):
                point = {"key": FEASIBILITY_SYSTEM, "topology": "m{}".format(m),
                         "cap": int(m), "status": child["status"], "child_exit": None,
                         "wall_s": child.get("elapsed_s"), "record": str(out_path),
                         "child": child, "adopted": True}
                record.add_point(point)
                print("  [m={}] adopted the existing child record ({})".format(
                    m, child["status"]), flush=True)
                continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "--feasibility-child",
               str(ints_path), str(m), str(out_path), "--caps",
               ",".join(str(c) for c in ladder), "--roots", str(n_roots),
               "--max-sweeps", str(max_sweeps), "--budget", "{:.0f}".format(share)]
        print("  [m={}] child: {:.0f} s share -> {}".format(m, share, out_path), flush=True)
        t0 = time.time()
        killed = None
        try:
            proc = subprocess.run(cmd, timeout=share + 120.0, cwd=str(REPO))
            exit_code = int(proc.returncode)
        except subprocess.TimeoutExpired:
            exit_code = None
            killed = "parent timeout after {:.0f} s".format(time.time() - t0)
        child: Dict = {}
        if out_path.is_file():
            try:
                child = json.loads(out_path.read_text())
            except ValueError:
                child = {}
        status = child.get("status", "no record")
        if killed is not None:
            status = "killed: {}".format(killed)
        elif exit_code is not None and exit_code < 0:
            status = "killed by signal {}".format(-exit_code)
        elif exit_code not in (0, None):
            status = "child exit {}".format(exit_code)
        point = {"key": FEASIBILITY_SYSTEM, "topology": "m{}".format(m), "cap": int(m),
                 "status": status, "child_exit": exit_code,
                 "wall_s": round(time.time() - t0, 1), "record": str(out_path),
                 "child": child}
        record.add_point(point)
        n_point += 1
        heartbeat.tick(n_point, partition=int(m), status=status)
        print("  [m={}] {} in {:.0f} s".format(m, status, point["wall_s"]), flush=True)


def summarize_feasibility(record_path: Path) -> List[Dict]:
    data = json.loads(Path(record_path).read_text())
    rows = []
    for p in data.get("points", []):
        child = p.get("child") or {}
        comp = child.get("steps", {}).get("compile", {})
        row = {"modes_per_node": p.get("cap"), "status": p.get("status"),
               "compile_status": comp.get("status"), "compile_cpu_s": comp.get("cpu_s"),
               "stored_gb": comp.get("stored_gb"),
               "dense_equivalent_gb": comp.get("dense_equivalent_gb"),
               "max_operator_bond": comp.get("max_operator_bond"),
               "compile_rss_gb": comp.get("rss_after_gb"), "points": []}
        for q in child.get("points", []):
            plan = q.get("plan", {})
            per_sweep = None
            if q.get("sweeps"):
                per_sweep = [s["cpu_s"] for s in q["sweeps"]]
            row["points"].append({"cap": q.get("cap"), "status": q.get("status"),
                                  "plan_gb": plan.get("sweep_peak_gb"),
                                  "plan_rdm_gb": plan.get("with_rdm_peak_gb"),
                                  "fits": plan.get("fits_limit"),
                                  "cpu_per_sweep_s": per_sweep,
                                  "rss_peak_gb": q.get("rss_peak_gb")})
        rows.append(row)
    return rows


def print_feasibility_summary(rows: Sequence[Dict]) -> None:
    head = "{:>5s} {:<22s} {:>10s} {:>9s} {:>9s} {:>7s} {:>8s}  {}".format(
        "m", "status", "compile s", "stored", "dense", "op-D", "RSS", "per cap")
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        caps = "; ".join("D={} {} plan {} GB (+RDM {}) sweeps {} peak {}".format(
            q["cap"], q["status"], q["plan_gb"], q["plan_rdm_gb"], q["cpu_per_sweep_s"],
            q["rss_peak_gb"]) for q in r["points"]) or "-"
        print("{:>5s} {:<22s} {:>10s} {:>9s} {:>9s} {:>7s} {:>8s}  {}".format(
            str(r["modes_per_node"]), str(r["status"])[:22],
            "-" if r["compile_cpu_s"] is None else "{:.0f}".format(r["compile_cpu_s"]),
            "-" if r["stored_gb"] is None else "{:.3f}".format(r["stored_gb"]),
            "-" if r["dense_equivalent_gb"] is None
            else "{:.1f}".format(r["dense_equivalent_gb"]),
            str(r["max_operator_bond"] or "-"),
            "-" if r["compile_rss_gb"] is None else "{:.2f}".format(r["compile_rss_gb"]),
            caps))


# ==============================================================================================
# S2.3 — the multi-site bridge
# ==============================================================================================

#: Root lanes per bridge system: ``(roots, per-site multiplet dimension)``. The dimension is
#: what the local-multiplet model is asked for (``rule="dimension"``) — the product of the
#: site dimensions is the model space, and it equals the root count by construction:
#: 2 x 2 = 4 (ground Kramers doublets), 4 x 4 = 16 (two doublets per site), 10 x 10 = 100
#: (the whole one-electron-per-site manifold) on the dimer; 2^3 = 8 on the trimer, whose
#: 1000-state manifold no shared-basis ensemble can hold (the floor is ``r x`` the
#: single-root rank — the S2.1 finding).
BRIDGE_LANES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "ti2cl6": ((4, 2), (16, 4), (100, 10)),
    "ti3f9_far": ((8, 2),),
}
BRIDGE_CAPS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
#: The sweep budget of a bridge point, and where the checkpoint/restart split is placed:
#: block 1 runs ``BRIDGE_SPLIT_SWEEPS`` sweeps and stops, block 2 is a new solver built
#: from the file. A point that converges before the split has nothing to restart and
#: says so.
BRIDGE_MAX_SWEEPS = 30
BRIDGE_SPLIT_SWEEPS = 2
#: ⚠ Looser than Phase 1's 1e-8, by measurement: on the dimer's four-root lane the sweep
#: at D = 4 and 8 stalls at |dE| of 2e-7 to 9e-6 Eh for the whole 30-sweep budget, and
#: 1e-7 Eh is 0.02 cm^-1 — sixteen times under the quantitative band's 0.35 cm^-1 floor.
#: The achieved dE is recorded per point either way.
BRIDGE_CONV_TOL = 1.0e-7
#: The solver's own default. ⚠ The first bridge run loosened this to 1e-6 because the
#: local Davidson on the dimer's 135-dimensional four-root problem stalled at
#: max|r| = 2.4e-7 for 300 iterations at every cap from 12 up (a triplet degenerate to
#: 0.4 cm^-1 among the roots); two-site problems that small are now diagonalized exactly
#: (`kuiva.dmrg.sweep.DENSE_LOCAL_MAX_DIM`), and the lane converges at the default.
BRIDGE_DAVIDSON_TOL = 1.0e-8
#: The control cap per lane (the extra legs — uninterrupted, paged, adaptive — run there):
#: the first cap of the ladder at or above this that the lane reaches.
BRIDGE_CONTROL_CAP = 8
#: A lane's root count must land on a manifold boundary: below this gap to the first root
#: left out (the state-averaging diagnostic's own figure) the lane is skipped, never run.
BRIDGE_BOUNDARY_MIN_CM = 50.0
#: Resident-environment cap that FORCES paging on the control leg: far below any
#: environment set, so every cold entry goes to scratch. A measurement of the escape
#: hatch, never a production setting.
BRIDGE_PAGING_RESIDENT_GB = 1.0e-5
#: Most nodes a site may be split into. The local-multiplet extraction merges a site's
#: nodes into one tensor and contracts the operator through it, one leg per node on each
#: side, and the block layer stops at sixteen legs: ten single-mode nodes per site
#: (measured on the dimer's four-root lane) merged but could not be contracted. Five
#: nodes of two modes is the shape the 30-spinor feasibility stage measured as well.
BRIDGE_MAX_NODES_PER_SITE = 5
#: Wall budget of the dimer's reference SA-CASSCF inside the stage, and the margin kept
#: for the ladder behind it. Checkpointed, so a stopped optimization resumes next time.
BRIDGE_ORBITAL_BUDGET_S = 5400.0


def bridge_partition(site_sizes: Sequence[int], n_elec: int, n_roots: int,
                     max_nodes_per_site: int = BRIDGE_MAX_NODES_PER_SITE) -> List[int]:
    """The finest site-blocked node partition whose every bond carries ``n_roots``.

    Like :func:`dmrg_campaign.node_partition`, but **no node straddles a site**: the
    local-multiplet extraction merges a site's nodes, and a node holding modes of two
    centres is a site of neither. Uniform chunks of at most ``m`` modes per site, ``m``
    grown from one until the two-site floor holds on every adjacent pair — including the
    pair across each site boundary, which is the tight one — and no site holds more than
    ``max_nodes_per_site`` nodes.
    """
    n = int(sum(site_sizes))
    for m in range(1, max(site_sizes) + 1):
        sizes: List[int] = []
        if any(-(-int(ns) // m) > int(max_nodes_per_site) for ns in site_sizes):
            continue
        for ns in site_sizes:
            k = -(-int(ns) // m)
            base, extra = divmod(int(ns), k)
            sizes.extend(camp._arrange([base + 1] * extra + [base] * (k - extra)))
        worst = min(a + b for a, b in zip(sizes, sizes[1:])) if len(sizes) > 1 else n
        if camp.two_site_floor(n, n_elec, worst) >= n_roots:
            return sizes
    return [int(x) for x in site_sizes]


def bridge_graph(site_sizes: Sequence[int], sizes: Sequence[int]):
    """The site-blocked chain over ascending modes, and the node groups per site."""
    graph = camp.path_graph(range(int(sum(site_sizes))), sizes)
    sites: List[Tuple[int, ...]] = []
    at, node = 0, 0
    for ns in site_sizes:
        group: List[int] = []
        while at < ns:
            group.append(node)
            at += sizes[node]
            node += 1
        if at != ns:
            raise ValueError("node partition {} straddles a site boundary at {}"
                             .format(list(sizes), ns))
        at = 0
        sites.append(tuple(group))
    return graph, sites


#: How each bridge system's active space is stated. ``"character"`` is the campaign's
#: ordinary form; ``"avas"`` projects onto the free-atom d orbitals of every centre
#: (``atomic_reference=True`` on the system's front end), for the case the character
#: selection lands on an incomplete set of delocalized combinations — the dimer's
#: ``protocol_note`` records the measurement that forced it.
BRIDGE_SELECTION = {"ti2cl6": "avas", "ti3f9_far": "character"}
#: The AVAS threshold, stated rather than defaulted: the dimer's projection spectrum has
#: ten pairs at 0.64 and above and the eleventh at 0.36 (measured on the first attempt,
#: which the 0.20 default carried to fourteen pairs), so the cut sits in that gap.
BRIDGE_AVAS_THRESHOLD = 0.5


def bridge_orbitals(system, reference, space, start: np.ndarray, *,
                    deadline: float, freeze: bool = False) -> Tuple[np.ndarray, object, Dict]:
    """The reference SA-CASSCF started from the stated (AVAS) orbitals, checkpointed.

    ⚠ At the scalar guess the dimer is not a weakly coupled d^1-d^1 pair at all: the
    closed-shell SCF pairs the two electrons in a Ti-Ti sigma bond and the CI at those
    orbitals puts the triplet 6 569 cm^-1 above the singlet (measured), where the committed
    record's optimized orbitals put the whole 2 x 2 ground product inside 27 cm^-1. The
    bridge exists to test a *product* manifold under truncation, so the orbitals are
    optimized for one — a state average over ``system.n_states`` roots — with a wall
    deadline the optimizer honours from the inside, resumed from its own checkpoint on a
    later run. A budget-stopped set is recorded as such and used: every quantity the ladder
    grades is a same-integral difference, so what an unconverged set costs is the
    physical meaning of the manifold, never the comparison.
    """
    from kuiva.interface import api

    camp.WORK.mkdir(parents=True, exist_ok=True)
    ckpt = camp.orbital_checkpoint(system)
    kw = dict(n_states=system.n_states, mode=system.casscf_mode, max_iter=system.max_iter,
              conv_grad=system.conv_grad, checkpoint=str(ckpt), report=False)
    if deadline > 0.0:
        kw["deadline"] = float(deadline)
    t0, c0 = time.time(), time.process_time()
    if ckpt.is_file() and freeze:
        # ⚠ A resumed record keeps the orbitals its earlier points were measured at: the
        # restart is capped at the checkpoint's own iteration count, so the optimizer
        # returns the stored set unchanged rather than moving it for another budget.
        import h5py
        with h5py.File(str(ckpt), "r") as handle:
            kw["max_iter"] = int(handle.attrs["iteration"])
        kw.pop("deadline", None)
    if ckpt.is_file():
        outcome = api.casscf(reference, restart=str(ckpt), **kw)
    else:
        outcome = api.casscf(reference, active=space, coeff=np.ascontiguousarray(start),
                             **kw)
    rec = {"source": "SA-CASSCF ({} roots) from the AVAS orbitals".format(system.n_states),
           "frozen_on_resume": bool(ckpt.is_file() and freeze),
           "converged": bool(outcome.converged),
           "iterations": int(outcome.orbital.n_iterations),
           "grad_norm": float(outcome.orbital.grad_norm),
           "e_avg": float(outcome.energy), "mode": system.casscf_mode,
           "checkpoint": str(ckpt),
           "wall_s": round(time.time() - t0, 1),
           "cpu_s": round(time.process_time() - c0, 1),
           "state_rel_cm": camp.rel_cm(np.asarray(outcome.ci.total_energies)),
           "boundary": camp._boundary(outcome.boundary),
           "boundary_initial": camp._boundary(outcome.boundary_initial)}
    return np.ascontiguousarray(outcome.coeff), outcome.active, rec


def localized_front_end(system, *, orbital_deadline: float = 0.0, freeze: bool = False
                        ) -> Tuple[object, np.ndarray, object, Dict, List[int]]:
    """Reference, per-centre localized active orbitals, the space, the diagnostics."""
    from kuiva.interface import api

    reference = camp.build_reference(system)
    centres = [i for i, (sym, _) in enumerate(system.atoms) if sym == system.element]
    route = BRIDGE_SELECTION.get(system.key, "character")
    selection: Dict = {"route": route}
    if route == "avas":
        avas = api.avas_active_space(reference, atom=centres, l="d",
                                     n_active_elec=system.n_active_elec,
                                     threshold=BRIDGE_AVAS_THRESHOLD,
                                     max_pairs=system.n_active // 2, report=False)
        if int(avas.space.n_active) != int(system.n_active):
            raise RuntimeError(
                "{}: AVAS selected {} active spinors where the system states {}; a "
                "different space is a different calculation".format(
                    system.key, avas.space.n_active, system.n_active))
        space, start = avas.space, np.ascontiguousarray(avas.coeff)
        selection.update(eigenvalues=[round(float(x), 4) for x in
                                      np.sort(avas.eigenvalues)[::-1][:system.n_active]],
                         gap=float(avas.gap), fold_residual=float(avas.fold_residual),
                         reference=avas.reference)
    else:
        space, start = api.active_space_for(reference, **system.selection()), None
    orbitals: Dict = {"source": "scalar guess", "converged": True}
    if system.orbitals == "casscf":
        start, space, orbitals = bridge_orbitals(
            system, reference, space,
            reference.spinors_in_ao() if start is None else start,
            deadline=orbital_deadline, freeze=freeze)
    loc = api.localize_active_space(reference, space, centres, coeff=start, report=False)
    coeff = np.ascontiguousarray(loc.coeff)
    #: the orbitals the localization started from, for the invariance assertion
    before = np.ascontiguousarray(reference.spinors_in_ao() if start is None else start)
    site = np.asarray(loc.site, dtype=int)
    if any(np.any(np.diff(np.nonzero(site == k)[0]) != 1) for k in range(loc.n_sites)):
        raise RuntimeError("the localized active orbitals are not site-blocked")
    diag = {"selection": selection, "orbitals": orbitals,
            "sites": [int(x) for x in site],
            "site_sizes": [int(np.sum(site == k)) for k in range(loc.n_sites)],
            "per_site_population_min": [round(float(loc.populations[site == k, k].min()), 4)
                                        for k in range(loc.n_sites)],
            "per_site_population_mean": [round(float(loc.populations[site == k, k].mean()),
                                               4) for k in range(loc.n_sites)]}
    diag["unlocalized_coeff"] = before
    return reference, coeff, space, diag, centres


def _checkpoint_policy(path: Path):
    """A per-sweep policy that always writes: the split below is the measurement."""
    from kuiva.dmrg.checkpoint import NetworkCheckpointPolicy
    return NetworkCheckpointPolicy(path, min_interval=0.0, cost_fraction=1.0e9,
                                   budget_gb=64.0)


def _bridge_solver(system, graph, template, cap: int, roots: int, **kw):
    from kuiva.dmrg import DMRGSolver

    solver = DMRGSolver(system.n_active_elec, max_bond=int(cap), n_roots=int(roots),
                        graph=graph, adaptive=False, conv_tol=BRIDGE_CONV_TOL,
                        davidson_tol=BRIDGE_DAVIDSON_TOL, on_split="warn", rdms=False,
                        **kw)
    solver._templates[graph] = template
    return solver


def _solve_outcome(solver, ints) -> Tuple[Optional[str], Optional[str]]:
    """Run one solve; ``(status, error)`` with ``status`` None on success."""
    from kuiva.dmrg.solver import SolverFailure
    from kuiva.util import resources as res

    try:
        solver.solve(ints)
    except ValueError as exc:
        return "refused", "{}".format(exc)
    except SolverFailure as exc:
        return "unconverged", "{}".format(exc)
    except res.MemoryLimitError as exc:
        return "refused-memory", "{}".format(exc).splitlines()[0]
    return None, None


def _split_solve(system, graph, template, ints, cap: int, roots: int, ckpt: Path,
                 max_sweeps: int) -> Tuple[Optional[object], Dict]:
    """Block 1 to the split, then a NEW solver restarted from the file. ``(solver, rec)``."""
    from kuiva.dmrg.checkpoint import read_network_state

    rec: Dict = {"split_sweeps": int(BRIDGE_SPLIT_SWEEPS)}
    if ckpt.exists():
        ckpt.unlink()
    t0, c0 = time.time(), time.process_time()
    first = _bridge_solver(system, graph, template, cap, roots,
                           max_sweeps=int(BRIDGE_SPLIT_SWEEPS),
                           checkpoint=_checkpoint_policy(ckpt))
    status, error = _solve_outcome(first, ints)
    rec["block1"] = {"status": status or "converged", "wall_s": round(time.time() - t0, 3),
                     "cpu_s": round(time.process_time() - c0, 3),
                     "checkpoint_written": bool(ckpt.is_file()),
                     "error": error}
    if status not in (None, "unconverged"):
        rec.update(status=status, error=error)
        return None, rec
    if status is None:
        # converged inside the first block: nothing to restart, and the record says so
        rec["block1"]["n_sweeps"] = int(first.last.n_sweeps)
        rec.update(status="ok", restarted=False, n_sweeps=int(first.last.n_sweeps),
                   wall_s=rec["block1"]["wall_s"], cpu_s=rec["block1"]["cpu_s"])
        return first, rec
    if not ckpt.is_file():
        # ⚠ No state after an unconverged block means the sweep never completed a sweep:
        # the solver raised INSIDE one (a local eigensolve that did not converge is the
        # case), which is the same "unconverged" outcome a plain ladder point records,
        # with its error text — not a failure of the split.
        rec.update(status="unconverged",
                   error="block 1 raised before completing a sweep (no network state "
                         "written): {}".format(error))
        return None, rec
    _, meta = read_network_state(ckpt, check_fingerprint=False)
    rec["block1"]["n_sweeps"] = int(meta.get("sweep", BRIDGE_SPLIT_SWEEPS))
    rec["block1"]["energies_at_split"] = [float(e) + float(ints.e_core)
                                          for e in np.asarray(meta.get("energies", []))]
    t1, c1 = time.time(), time.process_time()
    second = _bridge_solver(system, graph, template, cap, roots,
                            max_sweeps=int(max_sweeps), restart=ckpt,
                            checkpoint=_checkpoint_policy(ckpt))
    status, error = _solve_outcome(second, ints)
    rec["block2"] = {"status": status or "converged", "wall_s": round(time.time() - t1, 3),
                     "cpu_s": round(time.process_time() - c1, 3)}
    if status is not None:
        rec.update(status=status, error=error,
                   wall_s=round(time.time() - t0, 3),
                   cpu_s=round(time.process_time() - c0, 3))
        return None, rec
    r = second.last
    rec["block2"]["n_sweeps"] = int(r.n_sweeps)
    # the warm start's first sweep against the split's last: a restart that re-derives
    # the same energies in its first sweep started where the file said it had stopped
    e_first = float(r.history[0]) + float(ints.e_core) if r.history else None
    e_split = (float(np.mean(rec["block1"]["energies_at_split"]))
               if rec["block1"]["energies_at_split"] else None)
    rec["restart_warm_de_eh"] = (None if e_first is None or e_split is None
                                 else float(e_first - e_split))
    rec.update(status="ok", restarted=True,
               n_sweeps=int(rec["block1"]["n_sweeps"]) + int(r.n_sweeps),
               wall_s=round(time.time() - t0, 3), cpu_s=round(time.process_time() - c0, 3))
    return second, rec


def _plain_solve(system, graph, template, ints, cap: int, roots: int, max_sweeps: int,
                 **kw) -> Tuple[Optional[object], Dict]:
    """One uninterrupted solve from the same seed. ``(solver, rec)``."""
    t0, c0 = time.time(), time.process_time()
    solver = _bridge_solver(system, graph, template, cap, roots, max_sweeps=int(max_sweeps),
                            **kw)
    status, error = _solve_outcome(solver, ints)
    rec: Dict = {"wall_s": round(time.time() - t0, 3),
                 "cpu_s": round(time.process_time() - c0, 3)}
    if status is not None:
        rec.update(status=status, error=error)
        return None, rec
    r = solver.last
    rec.update(status="ok", n_sweeps=int(r.n_sweeps), w_disc=float(r.max_discarded),
               bond_used=int(r.max_bond_dim),
               energies=[float(e) + float(ints.e_core) for e in r.energies],
               n_paged_out=int(getattr(r, "n_paged_out", 0)),
               n_paged_in=int(getattr(r, "n_paged_in", 0)))
    return solver, rec


def _product_model(template, ints, state, weights, sites, dim: int, ops: Dict,
                   n_elec: int, e_ci: Optional[np.ndarray]) -> Dict:
    """The local-multiplet model of a (truncated) state, and what it says.

    ⚠ Re-gauges ``state`` in place; called after every other reading of the state.
    ``e_ci=None`` is the Tier-3 case — no oracle — and then the model's spectrum is
    recorded without a deviation or an interlacing verdict, never with an invented one.
    """
    from kuiva.dmrg import UnderResolved, effective_model
    from kuiva.props.multiplet import HARTREE_TO_CM
    from kuiva.props.pseudospin import pseudospin_from_model
    from kuiva.util import resources as res

    t0 = time.process_time()
    ttno = template.fill(ints.h_active_effective(), ints.active_eri())
    try:
        model = effective_model(ttno, state, sites, weights=weights, rule="dimension",
                                dims=int(dim), operators=ops, n_elec=int(n_elec),
                                report=False)
    except UnderResolved as exc:
        return {"status": "under-resolved", "error": "{}".format(exc)[:300],
                "cpu_s": round(time.process_time() - t0, 3)}
    except ValueError as exc:
        return {"status": "refused", "error": "{}".format(exc)[:300],
                "cpu_s": round(time.process_time() - t0, 3)}
    except (MemoryError, res.MemoryLimitError) as exc:
        # A memory refusal of the extraction (every array of the quotient-tree
        # contraction is required before it exists) is a result about the extraction,
        # recorded on the point rather than allowed to end the lane.
        return {"status": "refused-memory", "error": "{}".format(exc).splitlines()[0][:300],
                "cpu_s": round(time.process_time() - t0, 3)}
    spec = np.sort(model.spectrum()) + float(ints.e_core)
    rec: Dict = {"status": "ok", "model_dim": int(model.model_dim),
                 "sector_dim": int(spec.size),
                 "site_gap_ratios": [None if not np.isfinite(sp.gap_ratio)
                                     else round(float(sp.gap_ratio), 3)
                                     for sp in model.sites],
                 "site_n_electrons": [sp.n_electrons for sp in model.sites],
                 "rel_cm": camp.rel_cm(spec)}
    if e_ci is not None:
        ref = np.sort(np.asarray(e_ci, dtype=float))[:spec.size]
        dev = (spec - spec[0]) - (ref - ref[0])
        rec["interlacing_ok"] = bool(np.all(spec >= ref - 1e-8))
        rec["max_dev_cm"] = round(float(np.max(np.abs(dev))) * HARTREE_TO_CM, 4)
    else:
        rec["interlacing_ok"] = None
        rec["max_dev_cm"] = None
    try:
        ps = pseudospin_from_model(model, energy_shift=float(ints.e_core))
        rec["site_g"] = [[round(float(g), 6) for g in sp.g_values] for sp in ps.sites]
        rec["twice_s"] = [int(sp.twice_s) for sp in ps.sites]
        rec["unitarity_error"] = float(ps.unitarity_error())
    except ValueError as exc:
        rec["pseudospin"] = {"status": "refused", "error": "{}".format(exc)[:300]}
    rec["cpu_s"] = round(time.process_time() - t0, 3)
    return rec


def _adaptive_leg(system, graph, ints, cap: int, roots: int, sites_modes: Sequence[set],
                  e_ci: np.ndarray, max_sweeps: int) -> Dict:
    """The weight rule under a binding cap, from the site-blocked start."""
    from kuiva.dmrg import ReconnectionPolicy, hamiltonian_product_terms, solve_adaptive
    from kuiva.dmrg.solver import SolverFailure
    from kuiva.props.multiplet import HARTREE_TO_CM
    from kuiva.util import resources as res

    t0, c0 = time.time(), time.process_time()
    terms = hamiltonian_product_terms(np.ascontiguousarray(ints.h_active_effective()),
                                      np.ascontiguousarray(ints.active_eri()))
    try:
        result = solve_adaptive(terms, graph, system.n_active_elec, n_roots=int(roots),
                                max_bond=int(cap), policy=ReconnectionPolicy(rule="weight"),
                                max_sweeps=int(max_sweeps), conv_tol=BRIDGE_CONV_TOL,
                                davidson_tol=BRIDGE_DAVIDSON_TOL, boundary_check=0,
                                on_split="warn", rng=np.random.default_rng(0),
                                report_structure=False, report=False)
    except (ValueError, SolverFailure) as exc:
        return {"status": "refused" if isinstance(exc, ValueError) else "unconverged",
                "error": "{}".format(exc)[:300], "wall_s": round(time.time() - t0, 1),
                "cpu_s": round(time.process_time() - c0, 1)}
    except res.MemoryLimitError as exc:
        return {"status": "refused-memory", "error": "{}".format(exc).splitlines()[0],
                "wall_s": round(time.time() - t0, 1),
                "cpu_s": round(time.process_time() - c0, 1)}
    e = np.sort(np.asarray(result.final.energies, dtype=float)) + float(ints.e_core)
    ref = np.sort(np.asarray(e_ci, dtype=float))[:e.size]
    # did any adopted move put modes of two centres into one node?
    mixed = [u for u in range(result.graph.n_nodes)
             if sum(1 for s in sites_modes if set(result.graph.contents[u]) & s) > 1]
    return {"status": "ok", "converged": bool(result.converged),
            "n_moves": int(len(result.moves)),
            "moves": [{"sweep": int(m.sweep), "bond": [int(x) for x in m.bond],
                       "phys_swapped": bool(m.phys_swapped),
                       "metric_before": float(m.metric_before),
                       "metric_after": float(m.metric_after)} for m in result.moves],
            "final_edges": [[int(a), int(b)] for a, b in result.graph.edges],
            "final_contents": [[int(x) for x in c] for c in result.graph.contents],
            "nodes_mixing_sites": [int(u) for u in mixed],
            "topology_changed": bool(result.graph != graph),
            "w_disc": float(result.final.max_discarded),
            "bond_used": int(result.final.max_bond_dim),
            "e_sa_error_cm": round(float(np.mean(e - ref)) * HARTREE_TO_CM, 5),
            "max_level_dev_cm": round(float(np.max(np.abs((e - e[0]) - (ref - ref[0]))))
                                      * HARTREE_TO_CM, 5),
            "wall_s": round(time.time() - t0, 1),
            "cpu_s": round(time.process_time() - c0, 1)}


def stage_bridge(record, heartbeat, *, deadline: float, keys: Sequence[str],
                 caps: Optional[Sequence[int]] = None,
                 max_sweeps: int = BRIDGE_MAX_SWEEPS) -> None:
    """S2.3: the CI-checkable multi-site bridge with the Tier-3 protocol on."""
    from manifold_ladder import active_moments
    from kuiva.dmrg import one_electron_product_terms
    from kuiva.props.multiplet import HARTREE_TO_CM
    from kuiva.util import resources as res

    ladder = list(BRIDGE_CAPS if caps is None else caps)
    for key in keys:
        if time.time() > deadline:
            break
        system = camp.get(key)
        res.clear()
        camp.clear_templates()
        t_job, c_job = time.time(), time.process_time()
        # ⚠ The reference orbitals are frozen whenever their checkpoint exists, not only
        # on a resumed record: a fresh record after a code change re-measures the ladder
        # on the SAME orbitals, which is what makes its points comparable with the
        # earlier ones (a different orbital set is a different ladder).
        reference, coeff, space, loc, centres = localized_front_end(
            system, orbital_deadline=max(0.0, min(BRIDGE_ORBITAL_BUDGET_S,
                                                  deadline - time.time() - 1800.0)),
            freeze=record.has_job(key) or camp.orbital_checkpoint(system).is_file())
        orbitals = loc.pop("orbitals")
        ints = camp.cas_integrals(reference, coeff, space)
        n = int(space.n_active)
        site_sizes = loc["site_sizes"]
        sites_modes = [set(np.nonzero(np.asarray(loc["sites"]) == k)[0].tolist())
                       for k in range(len(site_sizes))]
        mu_act = active_moments(reference, coeff, space)
        ops = {name: one_electron_product_terms(mu_act[k])
               for k, name in enumerate(("mu_x", "mu_y", "mu_z"))}
        front = {"nao": int(reference.data.nao), "e_scf": float(reference.data.e_scf),
                 "scf_converged": bool(reference.data.converged),
                 "front_end_wall_s": round(time.time() - t_job, 1),
                 "front_end_cpu_s": round(time.process_time() - c_job, 1),
                 "localization": loc, "active_space": space.description,
                 "orbitals": dict(orbitals, note="active localized per centre")}
        if reference.data.soc is not None:
            front["hamiltonian"] = reference.data.soc.provenance()
        heartbeat.tick(0, system=key, stage="front-end")
        print("  [{}] front end: {} AOs, SCF {} E = {:.6f}; orbitals {}{}; site "
              "populations min {}".format(
                  key, front["nao"],
                  "converged" if front["scf_converged"] else "NOT CONVERGED",
                  front["e_scf"], orbitals["source"],
                  "" if orbitals["converged"] else " (NOT converged: |g| = {:.1e} after {} "
                  "iterations)".format(orbitals["grad_norm"], orbitals["iterations"]),
                  loc["per_site_population_min"]), flush=True)
        # the localization invariance, asserted on the largest lane's CI
        lanes = list(BRIDGE_LANES[key])
        widest = camp.replace(system, ladder_states=max(r for r, _ in lanes))
        e_loc, _, _ = camp.exact_ci(ints, widest)
        ints_can = camp.cas_integrals(reference, loc.pop("unlocalized_coeff"), space)
        e_can, _, _ = camp.exact_ci(ints_can, widest)
        front["localization_invariance_eh"] = float(np.max(np.abs(e_loc - e_can)))
        front["ci_rel_cm"] = camp.rel_cm(e_loc)
        del ints_can
        n_point = 0
        for roots, dim in lanes:
            lane = "r{}".format(roots)
            if time.time() > deadline:
                record.add_point({"key": key, "topology": lane, "cap": None,
                                  "status": "skipped",
                                  "reason": "stage wall budget exhausted"})
                break
            lane_sys = camp.replace(system, ladder_states=int(roots))
            sizes = bridge_partition(site_sizes, system.n_active_elec, roots)
            graph, site_nodes = bridge_graph(site_sizes, sizes)
            worst = min(a + b for a, b in zip(sizes, sizes[1:])) if len(sizes) > 1 else n
            e_ci, tdm_ci, ci_cost = camp.exact_ci(ints, lane_sys)
            m_ci = camp.property_matrices(reference, coeff, space, tdm_ci, e_ci, lane_sys)
            blocks = camp.oracle_blocks(m_ci, lane_sys)
            ref_reduction = camp.reduce_at(m_ci, blocks)
            e_sa_ci = float(np.mean(e_ci))
            boundary_cm = None
            if e_ci.size < e_loc.size:
                boundary_cm = round(float(e_loc[e_ci.size] - e_ci[-1]) * HARTREE_TO_CM, 3)
            if boundary_cm is not None and boundary_cm < BRIDGE_BOUNDARY_MIN_CM:
                # ⚠ A root count that is not a manifold boundary at these orbitals is not
                # a lane: the ensemble would cut a degenerate block, and every grade on
                # it would be a statement about an arbitrary rotation inside that block
                # (measured on the dimer: the 16-state count sits 0.03 cm^-1 below state
                # 17). Recorded and skipped, with the spectrum around the cut.
                record.add_point({"key": key, "topology": lane, "cap": None,
                                  "status": "skipped",
                                  "reason": "the {}-root count is not a manifold boundary "
                                            "at these orbitals (gap {} cm^-1 to the next "
                                            "root)".format(roots, boundary_cm),
                                  "levels_cm": camp.rel_cm(e_loc, roots + 4)})
                print("  [{}/{}] SKIPPED: {} roots cut a degenerate block (gap {} cm^-1); "
                      "levels {}".format(key, lane, roots, boundary_cm,
                                         camp.rel_cm(e_loc, roots + 4)[-6:]), flush=True)
                continue
            try:
                template, compile_cost = camp.compiled_template(graph)
            except res.MemoryLimitError as exc:
                # ⚠ A lane the machine cannot compile is a result about the ensemble, not
                # a failure of the stage: the whole 10 x 10 manifold as one shared-basis
                # ensemble forces two ten-mode nodes (the two-site floor), and a ten-mode
                # node's operator transition tables are 186 GB (measured). Recorded with
                # the node sizes that forced it; the other lanes go on.
                record.add_point({"key": key, "topology": lane, "cap": None,
                                  "status": "refused-memory",
                                  "reason": "{}".format(exc).splitlines()[0],
                                  "node_sizes": sizes, "roots": int(roots)})
                print("  [{}/{}] REFUSED by the memory plan at the compile (nodes {}): {}"
                      .format(key, lane, sizes, "{}".format(exc).splitlines()[0][:120]),
                      flush=True)
                continue
            job = {"key": key, "stage_kind": "bridge", "label": system.label,
                   "topology": lane, "roots": int(roots), "site_dim": int(dim),
                   "n_active": n, "n_active_elec": system.n_active_elec,
                   "n_det": ci_cost["ndet"], "protocol_note": system.protocol_note,
                   "sweep_budget": int(max_sweeps), "split_sweeps": BRIDGE_SPLIT_SWEEPS,
                   "conv_tol": BRIDGE_CONV_TOL,
                   "oracle": {"cost": ci_cost, "e_sa_eh": e_sa_ci,
                              "reduction": ref_reduction, "refused": None,
                              "blocks": [[int(a), int(b)] for a, b in blocks],
                              "boundary_gap_cm": boundary_cm},
                   "partition": {"node_sizes": sizes, "n_nodes": len(sizes),
                                 "two_site_floor": camp.two_site_floor(
                                     n, system.n_active_elec, worst),
                                 "sites_as_nodes": [list(g) for g in site_nodes]},
                   "topologies": {lane: {"edges": [list(map(int, e)) for e in graph.edges],
                                         "contents": [list(map(int, c))
                                                      for c in graph.contents],
                                         **compile_cost}},
                   **front}
            job["key_lane"] = "{}/{}".format(key, lane)
            previous = next((j for j in record.data["jobs"]
                             if j.get("key_lane") == job["key_lane"]), None)
            if previous is None:
                record.add_job(job)
            elif abs(float(previous["orbitals"].get("e_avg", 0.0))
                     - float(orbitals.get("e_avg", 0.0))) > 1e-9:
                # ⚠ A resumed record must not mix points at two orbital sets: the reference
                # CASSCF resumes from its checkpoint and may have moved since the earlier
                # points were measured, and a ladder across two orbital sets is not one
                # ladder. Refused with the knob named.
                raise RuntimeError(
                    "{}: the record's earlier points were measured at orbitals with E_avg "
                    "= {:.10f} and this run's are at {:.10f}; re-run with --fresh"
                    .format(job["key_lane"], previous["orbitals"].get("e_avg", 0.0),
                            orbitals.get("e_avg", 0.0)))
            print("  [{}/{}] oracle: {} dets, {} roots, {} blocks, boundary gap {} cm^-1, "
                  "nodes {} (floor {} vs {} roots), compile {:.1f} CPU s".format(
                      key, lane, ci_cost["ndet"], roots, len(blocks), boundary_cm, sizes,
                      job["partition"]["two_site_floor"], roots,
                      compile_cost["compile_cpu_s"]), flush=True)
            already = record.done_points()
            ckpt = camp.WORK / "bridge_{}_{}.network.h5".format(key, lane)
            # once per lane across the whole record, not per invocation: a resumed run
            # re-running the control legs at its first cap would repeat the adaptive leg,
            # whose reconnected graph carries the lane's largest two-site problems
            control_done = any("control" in q for q in record.data["points"]
                               if q.get("key") == key and q.get("topology") == lane)
            for cap in ladder:
                if (key, lane, int(cap)) in already:
                    print("  [{}/{}] D={:4d}  already measured; kept".format(key, lane, cap),
                          flush=True)
                    continue
                if time.time() > deadline:
                    record.add_point({"key": key, "topology": lane, "cap": int(cap),
                                      "status": "skipped",
                                      "reason": "stage wall budget exhausted"})
                    break
                point: Dict = {"key": key, "topology": lane, "cap": int(cap),
                               "roots": int(roots), "protocol": "tier3-bridge"}
                solver, rec = _split_solve(system, graph, template, ints, cap, roots, ckpt,
                                           max_sweeps)
                point.update(rec)
                if solver is None:
                    record.add_point(point)
                    n_point += 1
                    heartbeat.tick(n_point, system=key, lane=lane, cap=int(cap),
                                   status=point["status"])
                    print("  [{}/{}] D={:4d}  {:11s}  {}".format(
                        key, lane, cap, point["status"].upper(),
                        point.get("error", "")[:100]), flush=True)
                    continue
                r = solver.last
                energies = np.asarray(r.energies, dtype=float) + float(ints.e_core)
                point.update(bond_used=int(r.max_bond_dim), w_disc=float(r.max_discarded),
                             saturating=bool(int(r.max_bond_dim) < int(cap)),
                             energies=[float(e) for e in energies],
                             n_paged_out=int(getattr(r, "n_paged_out", 0)),
                             n_paged_in=int(getattr(r, "n_paged_in", 0)))
                tdm = solver.transition_densities()
                m_net = camp.property_matrices(reference, coeff, space, tdm, energies,
                                               lane_sys)
                trial = camp.reduce_at(m_net, blocks)
                point["reduction"] = trial
                point["e_sa_eh"] = float(np.mean(energies))
                e_err = point["e_sa_eh"] - e_sa_ci
                point["e_sa_error_eh"] = e_err
                point["e_sa_error_cm"] = round(e_err * HARTREE_TO_CM, 6)
                point["grade"] = camp.grade(ref_reduction, trial,
                                            e_sa_error_cm=e_err * HARTREE_TO_CM)
                # the product structure: the local-multiplet model of THIS truncated state
                point["product_model"] = _product_model(
                    template, ints, solver._state, r.weights, site_nodes, dim, ops,
                    system.n_active_elec, e_ci)
                # the point is on disk BEFORE its control legs: the adaptive leg is the
                # lane's largest and longest computation, and a kill inside it must not
                # take the measured point with it
                point["elapsed_s"] = round(time.time() - t_job, 1)
                record.add_point(point)
                # --- the control legs, once per lane -----------------------------------
                if not control_done and int(cap) >= BRIDGE_CONTROL_CAP \
                        and time.time() < deadline:
                    control_done = True
                    point["control"] = {}
                    record.flush()
                    _, plain = _plain_solve(system, graph, template, ints, cap, roots,
                                            max_sweeps)
                    if plain["status"] == "ok":
                        plain["de_vs_split_eh"] = float(np.max(np.abs(
                            np.asarray(plain["energies"]) - energies)))
                    point["control"]["uninterrupted"] = plain
                    _, paged = _plain_solve(system, graph, template, ints, cap, roots,
                                            max_sweeps,
                                            environment_resident_gb=BRIDGE_PAGING_RESIDENT_GB)
                    if paged["status"] == "ok" and plain["status"] == "ok":
                        paged["de_vs_unpaged_eh"] = float(np.max(np.abs(
                            np.asarray(paged["energies"]) - np.asarray(plain["energies"]))))
                        paged["bitwise"] = bool(paged["de_vs_unpaged_eh"] == 0.0)
                    point["control"]["paged"] = paged
                    point["control"]["adaptive_weight_rule"] = _adaptive_leg(
                        system, graph, ints, cap, roots, sites_modes, e_ci, max_sweeps)
                point["elapsed_s"] = round(time.time() - t_job, 1)
                record.flush()
                n_point += 1
                heartbeat.tick(n_point, system=key, lane=lane, cap=int(cap),
                               status=point["status"], grade=point["grade"]["overall"])
                _print_bridge_point(key, lane, cap, point)
                if point["saturating"] and point["grade"]["overall"] == "quantitative":
                    record.add_point({"key": key, "topology": lane, "cap": None,
                                      "status": "ladder-complete",
                                      "reason": "saturated at D = {} (bond used {})".format(
                                          cap, point["bond_used"])})
                    print("  [{}/{}] saturated at D={} (bond used {})".format(
                        key, lane, cap, point["bond_used"]), flush=True)
                    break
            if ckpt.exists():
                ckpt.unlink()


def _print_bridge_point(key: str, lane: str, cap: int, point: Dict) -> None:
    g = point["grade"]
    pm = point.get("product_model", {})
    model = ("model {}".format(pm["status"]) if pm.get("status") != "ok"
             else "model dev {:.2f} cm^-1 g {}".format(
                 pm["max_dev_cm"], "/".join("{:.3f}".format(x) for x in pm["site_g"][0])
                 if pm.get("site_g") else "-"))
    ctrl = ""
    if "control" in point:
        c = point["control"]
        ctrl = "  [restart dE {:.1e}, paged {} bitwise, adaptive {} moves]".format(
            c["uninterrupted"].get("de_vs_split_eh", float("nan")),
            c["paged"].get("bitwise", "?"),
            c["adaptive_weight_rule"].get("n_moves", "?"))
    print("  [{}/{}] D={:4d}  used {:3d}  {:2d} sweeps{}  w_disc {:.2e}  dE_SA {:+.3e} Eh  "
          "dE_max {:8.3f} cm^-1  {:14s}  {}  {:.1f} CPU s{}".format(
              key, lane, cap, point["bond_used"], point["n_sweeps"],
              "*" if point.get("restarted") else " ", point["w_disc"],
              point["e_sa_error_eh"],
              g["max_energy_dev_cm"] if g["max_energy_dev_cm"] is not None else float("nan"),
              g["overall"], model, point["cpu_s"], ctrl), flush=True)


def summarize_bridge(record_path: Path) -> List[Dict]:
    """Per lane: the tier floors, the product model across caps, the control legs."""
    data = json.loads(record_path.read_text())
    rows: List[Dict] = []
    jobs = {j["key_lane"]: j for j in data.get("jobs", []) if "key_lane" in j}
    for key_lane, job in sorted(jobs.items()):
        key, lane = key_lane.split("/")
        pts = [p for p in data.get("points", [])
               if p.get("key") == key and p.get("topology") == lane
               and p.get("cap") is not None]
        pts.sort(key=lambda p: int(p["cap"]))
        ok = [p for p in pts if p.get("status") == "ok"]
        top = ok[-1] if ok else None
        row: Dict = {"key": key, "lane": lane, "roots": job["roots"],
                     "n_nodes": job["partition"]["n_nodes"],
                     "oracle_cpu_s": job["oracle"]["cost"]["cpu_s"],
                     "boundary_gap_cm": job["oracle"].get("boundary_gap_cm"),
                     "refused": [int(p["cap"]) for p in pts if p.get("status") == "refused"],
                     "unconverged": [int(p["cap"]) for p in pts
                                     if p.get("status") == "unconverged"],
                     "caps": [int(p["cap"]) for p in ok]}
        for tier in ("qualitative", "quantitative"):
            hit = next((p for p in ok if camp.TIERS.index(p["grade"]["overall"])
                        <= camp.TIERS.index(tier)), None) \
                if hasattr(camp, "TIERS") else None
            row[tier] = None if hit is None else {"cap": int(hit["cap"]),
                                                  "cpu_s": hit["cpu_s"],
                                                  "n_sweeps": hit["n_sweeps"]}
        row["model"] = [{"cap": int(p["cap"]), "status": p["product_model"]["status"],
                         "max_dev_cm": p["product_model"].get("max_dev_cm"),
                         "site_g": p["product_model"].get("site_g"),
                         "gap_ratios": p["product_model"].get("site_gap_ratios")}
                        for p in ok]
        row["top_site_g"] = (top["product_model"].get("site_g") if top else None)
        row["controls"] = next((p["control"] for p in ok if "control" in p), None)
        row["restart_warm_de_eh"] = [p.get("restart_warm_de_eh") for p in ok
                                     if p.get("restarted")]
        rows.append(row)
    return rows


def print_bridge_summary(rows: Sequence[Dict]) -> None:
    head = "{:<11s} {:<6s} {:>5s} {:>5s} {:>9s}  {:<22s} {:<22s}  {}".format(
        "system", "lane", "roots", "nodes", "CI CPU s", "qualitative", "quantitative",
        "controls (restart dE / paged bitwise / adaptive moves)")
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        def tier(t):
            return "-" if t is None else "D={} {:.0f}s {}sw".format(t["cap"], t["cpu_s"],
                                                                 t["n_sweeps"])
        c = r["controls"] or {}
        ctrl = "-" if not c else "{:.1e} / {} / {}".format(
            c["uninterrupted"].get("de_vs_split_eh", float("nan")),
            c["paged"].get("bitwise", "?"), c["adaptive_weight_rule"].get("n_moves", "?"))
        print("{:<11s} {:<6s} {:>5d} {:>5d} {:>9.1f}  {:<22s} {:<22s}  {}".format(
            r["key"], r["lane"], r["roots"], r["n_nodes"], r["oracle_cpu_s"],
            tier(r["qualitative"]), tier(r["quantitative"]), ctrl))
        for m in r["model"]:
            print("      D={:<4d} model {:<14s} dev {:>10s} cm^-1  site g {}".format(
                m["cap"], m["status"],
                "-" if m["max_dev_cm"] is None else "{:.3f}".format(m["max_dev_cm"]),
                "-" if not m["site_g"] else "; ".join(
                    "/".join("{:.4f}".format(x) for x in g) for g in m["site_g"])))


# ==============================================================================================
# S2.4 — the first Tier-3 calculation: the front end and the feasibility ladder
# ==============================================================================================

#: The Tier-3 system of the first attempt (a campaign entry; its Tier-3 definition is
#: ``tier3_systems.get("mn3_linear")``, and the two are checked against each other).
TIER3_SYSTEM = "mn3_linear"
#: The feasibility ladder, cheapest first. ⚠ Bounded sweeps at every cap: what is
#: measured is the cost SHAPE — per-sweep CPU, the memory plan, whether the cap is honoured
#: — never a converged energy. The cap a converged Tier-3 run needs is S2.4b's question and
#: is decided on these numbers.
TIER3_CAPS = (8, 16, 32, 64, 128)
TIER3_SWEEPS = 2
#: The ensemble is the WHOLE S = 5/2 ground multiplet — six states, three Kramers doublets
#: (the system's ``n_states``). Not a cut inside it: Mn(II)'s zero-field splitting separates
#: the doublets by well under a wavenumber, and a count inside a near-degenerate manifold is
#: the defect the state-averaging rule exists for. Not a wider one either: the next
#: multiplet (S = 3/2) sits a few |J| above, and whether ten roots is a boundary at these
#: orbitals is S2.4b's first decision.
TIER3_SITE_DIM = 6
#: ⚠ The local Davidson tolerance of the bounded sweeps, looser than the solver's 1e-8 by
#: measurement: at D = 32 the two-site problem on bond (8, 7) — 4204 determinants, six
#: roots among which the Kramers pairs are exactly degenerate — stalled for 300
#: iterations at max|r| = 4.5e-6, the residual floor the dimer bridge met on its
#: 135-dimensional problem before that size got a dense route. 1e-6 bounds the local
#: energy error at 1e-12 Eh, five orders under anything a bounded sweep resolves; the
#: value is recorded on the child's record. A converged Tier-3 run (S2.4b) has to decide
#: this for itself, and the stall is a finding it inherits.
TIER3_DAVIDSON_TOL = 1.0e-6
#: Wall budget of the front-end child: the SCF (~27 s per cycle on 352 AOs, direct
#: integrals; ~25 cycles with the measured recipe), the Mn four-component solve (84 s,
#: then cached), the direct Cholesky decomposition and the localization.
TIER3_FRONT_END_BUDGET_S = 3600.0


def tier3_integrals_path(key: str) -> Path:
    return camp.WORK / "{}_tier3_integrals.npz".format(key)


def _spin_blocks(reference, coeff: np.ndarray, space) -> Dict[str, np.ndarray]:
    """The spin operator over the orbital spaces — what ``<S^2>`` needs on the network side.

    Exactly the four pieces :func:`kuiva.props.spin.spin_analysis` builds (the active block,
    the inactive trace, and the three out-of-CAS families), computed here in the front-end
    process because the analysis layer has no integrals and the ladder child has no
    reference: it reads them off the integrals file.
    """
    from kuiva.props.dump import inactive_moment, spinor_operator
    from kuiva.spinor.expand import spin_operator

    s_mo = spinor_operator(np.ascontiguousarray(coeff), spin_operator(reference.data.s_ao))
    spaces = space.spaces
    act = np.asarray(spaces.active, dtype=int)
    inact = np.asarray(spaces.inactive, dtype=int)
    virt = np.asarray(spaces.virtual, dtype=int)
    return {"s_active": np.stack([sk[np.ix_(act, act)] for sk in s_mo]),
            "s_inactive_trace": np.asarray(inactive_moment(s_mo, inact, name="S"),
                                           dtype=float),
            "s_b": np.stack([sk[np.ix_(act, inact)] for sk in s_mo]),
            "s_c": np.stack([sk[np.ix_(virt, act)] for sk in s_mo]),
            "s_d": np.stack([sk[np.ix_(virt, inact)] for sk in s_mo])}


def tier3_front_end_child(key: str, out_path: Path, *, budget: float) -> int:
    """The Tier-3 front end, in its own process: SCF, selection, localization, integrals.

    Writes the active integrals, the spin-operator blocks and the active moment operators
    to one ``.npz`` and a JSON record of every diagnostic beside it. ⚠ Two refusals are
    recorded as outcomes and end the stage: an unconverged SCF (everything downstream is
    built on those orbitals), and a character selection that did not land on the fifteen
    singly occupied 3d orbitals (a wrong shell is a different calculation, and no
    observable of a Tier-3 run can see it).
    """
    from manifold_ladder import active_moments
    from progress import Heartbeat
    from dmrg_memory_plan import rss_gb
    from kuiva.interface import api
    from kuiva.util import resources as res

    import tier3_systems as t3

    system = camp.get(key)
    tier3 = t3.get(key)
    heartbeat = Heartbeat("dmrg_cost_ladder_s2.4a_front", budget_seconds=budget,
                          meta={"key": key, "pid": os.getpid()})
    rec: Dict = {"key": key, "label": system.label, "pid": os.getpid(), "status": "running",
                 "basis": system.basis, "charge": system.charge, "spin": system.spin,
                 "atoms": [[s, list(xyz)] for s, xyz in system.atoms],
                 "geometry_note": system.geom_note, "physics_note": system.physics_note,
                 "protocol_note": system.protocol_note,
                 "tier3_definition": {"formula": tier3.formula, "topology": tier3.topology,
                                      "edges": [list(e) for e in tier3.edges],
                                      "local_dims": list(tier3.local_dims),
                                      "cas_electrons": tier3.cas_electrons,
                                      "cas_spinors": tier3.cas_spinors,
                                      "cas_determinants": tier3.cas_determinants,
                                      "lieb_mattis_twice_spin": t3.lieb_mattis_twice_spin(
                                          tier3)}}
    t_start = time.time()

    def flush(status: Optional[str] = None) -> None:
        if status is not None:
            rec["status"] = status
        rec["elapsed_s"] = round(time.time() - t_start, 1)
        tmp = out_path.with_suffix(".json.part")
        with open(tmp, "w") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True, default=_jsonable)
        os.replace(tmp, out_path)

    # the campaign entry and the Tier-3 definition must describe the same calculation
    if (2 * system.n_active_elec != 2 * tier3.cas_electrons
            or system.n_active != tier3.cas_spinors
            or system.n_det != tier3.cas_determinants):
        rec["error"] = ("the campaign entry (CAS({}, {}), {} dets) disagrees with the "
                        "Tier-3 definition (CAS({}, {}), {} dets)".format(
                            system.n_active_elec, system.n_active, system.n_det,
                            tier3.cas_electrons, tier3.cas_spinors, tier3.cas_determinants))
        flush("definition-mismatch")
        return 1
    flush()
    res.clear()
    t0, c0 = time.time(), time.process_time()
    reference = camp.build_reference(system)
    rec["front_end"] = {"nao": int(reference.data.nao), "nspinor": int(reference.nspinor),
                        "e_scf": float(reference.data.e_scf),
                        "scf_converged": bool(reference.data.converged),
                        "n_cholesky": int(reference.factors.naux),
                        "wall_s": round(time.time() - t0, 1),
                        "cpu_s": round(time.process_time() - c0, 1),
                        "rss_gb": round(rss_gb(), 3)}
    if reference.data.soc is not None:
        rec["front_end"]["hamiltonian"] = reference.data.soc.provenance()
    heartbeat.tick(1, stage="scf", converged=int(reference.data.converged))
    flush()
    print("    front end: {} AOs, {} Cholesky vectors, SCF {} E = {:.8f} Eh, {:.0f} CPU s, "
          "RSS {:.2f} GB".format(rec["front_end"]["nao"], rec["front_end"]["n_cholesky"],
                                 "converged" if reference.data.converged else "NOT CONVERGED",
                                 rec["front_end"]["e_scf"], rec["front_end"]["cpu_s"],
                                 rec["front_end"]["rss_gb"]), flush=True)
    if not reference.data.converged:
        rec["error"] = "the scalar SCF did not converge; fix the guess before anything runs"
        flush("scf-unconverged")
        heartbeat.finish(status="scf-unconverged")
        return 1

    # --- the active space, trap-checked against the reference's own occupations -------------
    space = api.active_space_for(reference, **system.selection())
    active = np.asarray(space.spaces.active, dtype=int)
    occ = np.asarray(reference.data.mo_occ, dtype=float)
    occ_active = occ[active // 2] if occ.ndim == 1 else None
    rec["active_space"] = {"description": space.description,
                           "n_active": int(space.n_active), "n_elec": int(space.n_elec),
                           "spinor_indices": [int(i) for i in active],
                           "reference_occupations": (None if occ_active is None
                                                     else [float(x) for x in occ_active])}
    if occ_active is None or not np.allclose(occ_active, 1.0):
        rec["error"] = ("the {} lowest d-character pairs are not the singly occupied 3d "
                        "shell of the high-spin reference: occupations {}".format(
                            space.n_active // 2, None if occ_active is None
                            else [round(float(x), 3) for x in occ_active]))
        flush("selection-off-shell")
        heartbeat.finish(status="selection-off-shell")
        return 1
    centres = [i for i, (sym, _) in enumerate(system.atoms) if sym == system.element]
    loc = api.localize_active_space(reference, space, centres, report=False)
    coeff = np.ascontiguousarray(loc.coeff)
    site = np.asarray(loc.site, dtype=int)
    if any(np.any(np.diff(np.nonzero(site == k)[0]) != 1) for k in range(loc.n_sites)):
        rec["error"] = "the localized active orbitals are not site-blocked"
        flush("localization-failed")
        heartbeat.finish(status="localization-failed")
        return 1
    # the localization is an active-active rotation: the active SPAN is invariant, and
    # with no CI to assert it on that is what is asserted — the overlap of the localized
    # active block with the canonical one is unitary
    s_ao = np.asarray(reference.data.s_ao)
    nao = s_ao.shape[0]
    c_can = np.ascontiguousarray(reference.spinors_in_ao())[:, active]
    c_loc = coeff[:, active]
    sc = np.concatenate([s_ao @ c_can[:nao], s_ao @ c_can[nao:]], axis=0)
    m = c_loc.conj().T @ sc
    rec["localization"] = {
        "sites": [int(x) for x in site],
        "site_sizes": [int(np.sum(site == k)) for k in range(loc.n_sites)],
        "per_site_population_min": [round(float(loc.populations[site == k, k].min()), 4)
                                    for k in range(loc.n_sites)],
        "per_site_population_mean": [round(float(loc.populations[site == k, k].mean()), 4)
                                     for k in range(loc.n_sites)],
        "span_invariance": float(np.max(np.abs(m.conj().T @ m - np.eye(m.shape[1]))))}
    heartbeat.tick(2, stage="localized")
    flush()
    print("    active space: {}; reference occupations {}; localized per-site populations "
          "min {} (span invariance {:.1e})".format(
              space.description, [round(float(x), 2) for x in occ_active],
              rec["localization"]["per_site_population_min"],
              rec["localization"]["span_invariance"]), flush=True)

    # --- the integrals, the spin blocks, the moments: everything the ladder child reads -----
    t0, c0 = time.time(), time.process_time()
    ints = camp.cas_integrals(reference, coeff, space)
    h = np.ascontiguousarray(ints.h_active_effective())
    eri = np.ascontiguousarray(ints.active_eri())
    blocks = _spin_blocks(reference, coeff, space)
    mu = active_moments(reference, coeff, space)
    path = tier3_integrals_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, h=h, eri=eri, e_core=float(ints.e_core),
             n_elec=int(system.n_active_elec), sites=site.astype(np.int64), mu=mu,
             e_ci=np.full(system.n_states, np.nan), **blocks)
    rec["integrals"] = {"path": str(path), "e_core": float(ints.e_core),
                        "wall_s": round(time.time() - t0, 1),
                        "cpu_s": round(time.process_time() - c0, 1),
                        "h_hermiticity": float(np.max(np.abs(h - h.conj().T))),
                        "s_active_hermiticity": float(np.max(np.abs(
                            blocks["s_active"] - blocks["s_active"].conj().transpose(0, 2, 1)))),
                        "s_inactive_trace": [float(x) for x in blocks["s_inactive_trace"]],
                        "s_leakage_max": max(float(np.max(np.abs(blocks[k]))) if blocks[k].size
                                             else 0.0 for k in ("s_b", "s_c", "s_d"))}
    rec["rss_peak_gb"] = round(rss_gb(), 3)
    heartbeat.tick(3, stage="integrals")
    flush("ok")
    heartbeat.finish(status="ok")
    return 0


class _Tier3Integrals(_ActiveIntegrals):
    """The integrals file plus the spin blocks and the moments the analysis reads."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        data = np.load(path)
        self.mu = np.ascontiguousarray(data["mu"])
        self.s_active = np.ascontiguousarray(data["s_active"])
        self.s_inactive_trace = np.asarray(data["s_inactive_trace"], dtype=float)
        self.s_leak = tuple(np.ascontiguousarray(data[k]) for k in ("s_b", "s_c", "s_d"))


def _spin_of_roots(ints: _Tier3Integrals, template, graph, state, cap: int, n_roots: int
                   ) -> Dict:
    """``<S^2>`` of every stored root, through the same contraction a Tier-3 run reports.

    The analysis contract is the solver's (``one_body_moments``), so the state is handed
    to a solver shell on the same template — the props layer duck-types it and never sees
    the network. Each root's own 1- and 2-RDM are contracted for it; a memory refusal of
    that extraction is recorded on the point, not raised.
    """
    from kuiva.dmrg import DMRGSolver
    from kuiva.props.spin import spin_from_s_squared, spin_squared_states
    from kuiva.util import resources as res

    t0 = time.process_time()
    shell = DMRGSolver(ints.n_elec, max_bond=int(cap), n_roots=int(n_roots), graph=graph,
                       rdms=False, on_split="warn")
    shell._templates[graph] = template
    shell._state = state
    try:
        s2, leak = spin_squared_states(shell, ints.s_active,
                                       inactive_trace=ints.s_inactive_trace,
                                       leak_blocks=ints.s_leak)
    except (MemoryError, res.MemoryLimitError) as exc:
        return {"status": "refused-memory", "error": "{}".format(exc).splitlines()[0][:300],
                "cpu_s": round(time.process_time() - t0, 3)}
    return {"status": "ok", "s_squared": [round(float(x), 6) for x in s2],
            "spin": [round(float(spin_from_s_squared(x)), 4) for x in s2],
            "leak": [round(float(x), 8) for x in leak],
            "cpu_s": round(time.process_time() - t0, 3)}


def tier3_ladder_child(key: str, ints_path: Path, out_path: Path, *, caps: Sequence[int],
                       n_roots: int, max_sweeps: int, budget: float) -> int:
    """The feasibility ladder of one Tier-3 system, in its own process.

    The site-blocked chain of the finest partition that holds the ensemble, compiled once;
    then, per cap and cheapest first: the memory plan (with and without the RDM
    contraction a CASSCF would add), ``max_sweeps`` bounded sweeps from a seeded random
    start with the per-sweep wall and CPU recorded, the six energies and their Kramers
    pairing, ``<S^2>`` of every root, and the local-multiplet model at six states per site
    with its site pseudospins. Every step is on disk before the next starts; a budget kill
    leaves what was measured.
    """
    from progress import Heartbeat
    from dmrg_memory_plan import PhaseSampler, instrument, rss_gb
    from kuiva.dmrg import TTNOTemplate, one_electron_product_terms, random_state, solve_ttn
    from kuiva.dmrg.plan import network_memory_plan
    from kuiva.util.errors import SolverFailure
    from kuiva.dmrg.sweep import environment_gb, state_gb
    from kuiva.props.multiplet import HARTREE_TO_CM
    from kuiva.util import resources as res

    lims = res.ensure_configured()
    deadline = time.time() + float(budget)
    ints = _Tier3Integrals(ints_path)
    n = ints._h.shape[0]
    n_sites = int(max(ints.sites)) + 1
    site_sizes = [sum(1 for s in ints.sites if s == k) for k in range(n_sites)]
    sizes = bridge_partition(site_sizes, ints.n_elec, int(n_roots))
    graph, site_nodes = bridge_graph(site_sizes, sizes)
    worst = min(a + b for a, b in zip(sizes, sizes[1:])) if len(sizes) > 1 else n
    rec: Dict = {"key": key, "roots": int(n_roots), "caps": [int(c) for c in caps],
                 "sweeps": int(max_sweeps), "conv_tol": BRIDGE_CONV_TOL,
                 "davidson_tol": TIER3_DAVIDSON_TOL,
                 "n_modes": n, "n_elec": ints.n_elec, "site_sizes": site_sizes,
                 "partition": {"node_sizes": sizes, "n_nodes": len(sizes),
                               "two_site_floor": camp.two_site_floor(n, ints.n_elec, worst),
                               "sites_as_nodes": [list(g) for g in site_nodes]},
                 "graph": graph_record(graph),
                 "memory_limit_gb": float(lims.memory_gb), "budget_s": float(budget),
                 "pid": os.getpid(), "status": "running", "steps": {}, "points": []}
    # ⚠ A cap already measured by an earlier invocation is kept, never repeated: the
    # record on disk is the authority, and a child that died at one cap (a solver
    # failure, a budget kill) resumes at the next.
    kept: List[Dict] = []
    if out_path.is_file():
        try:
            previous = json.loads(out_path.read_text())
        except ValueError:
            previous = {}
        kept = [p for p in previous.get("points", [])
                if p.get("status") not in (None, "running", "skipped")]
        if kept:
            rec["resumed_from"] = previous.get("elapsed_s")
            rec["points"] = kept
    done = {int(p["cap"]) for p in kept}
    heartbeat = Heartbeat("dmrg_cost_ladder_s2.4a_ladder", budget_seconds=budget,
                          meta={"key": key, "pid": os.getpid()})
    t_start = time.time()

    def flush(status: Optional[str] = None) -> None:
        if status is not None:
            rec["status"] = status
        rec["elapsed_s"] = round(time.time() - t_start, 1)
        tmp = out_path.with_suffix(".json.part")
        with open(tmp, "w") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True, default=_jsonable)
        os.replace(tmp, out_path)

    sampler = PhaseSampler(res.BUDGET)
    instrument(sampler)
    sampler.start()
    rec["rss_baseline_gb"] = round(sampler.baseline, 4)
    flush()

    # --- the compile, once --------------------------------------------------------------------
    t0, c0 = time.time(), time.process_time()
    try:
        template = TTNOTemplate(graph)
    except res.MemoryLimitError as exc:
        rec["steps"]["compile"] = {"status": "refused-memory",
                                   "error": "{}".format(exc).splitlines()[0],
                                   "wall_s": round(time.time() - t0, 1),
                                   "cpu_s": round(time.process_time() - c0, 1)}
        flush("refused-memory")
        heartbeat.finish(status="refused-memory")
        return 0
    op = template.ttno
    rec["steps"]["compile"] = {
        "status": "ok", "wall_s": round(time.time() - t0, 1),
        "cpu_s": round(time.process_time() - c0, 1), "n_terms": int(template.n_terms),
        "stored_gb": round(op.nbytes / 1024.0 ** 3, 4),
        "max_operator_bond": int(max(op.bond_dimensions().values())),
        "ledger_resident_gb": round(res.BUDGET.resident_gb(), 4),
        "rss_after_gb": round(rss_gb(), 4)}
    heartbeat.tick(1, stage="compiled", cpu=rec["steps"]["compile"]["cpu_s"])
    flush()
    print("    compile: {:.0f} CPU s, stored {:.3f} GB, max operator bond {}, RSS {:.2f} GB"
          .format(rec["steps"]["compile"]["cpu_s"], rec["steps"]["compile"]["stored_gb"],
                  rec["steps"]["compile"]["max_operator_bond"],
                  rec["steps"]["compile"]["rss_after_gb"]), flush=True)
    t0, c0 = time.time(), time.process_time()
    ttno = template.fill(ints.h_active_effective(), ints.active_eri())
    rec["steps"]["fill"] = {"status": "ok", "wall_s": round(time.time() - t0, 2),
                            "cpu_s": round(time.process_time() - c0, 2)}
    ops = {name: one_electron_product_terms(ints.mu[k])
           for k, name in enumerate(("mu_x", "mu_y", "mu_z"))}
    flush()

    # --- per cap ------------------------------------------------------------------------------
    for cap in caps:
        if int(cap) in done:
            print("    D={}: already measured; kept".format(cap), flush=True)
            continue
        if time.time() > deadline:
            rec["points"].append({"cap": int(cap), "status": "skipped",
                                  "reason": "ladder wall budget exhausted"})
            flush()
            break
        point: Dict = {"cap": int(cap), "roots": int(n_roots), "protocol": "tier3"}
        sampler.reset()
        t0, c0 = time.time(), time.process_time()
        try:
            state = random_state(ttno, ints.n_elec, int(cap), n_roots=int(n_roots),
                                 rng=np.random.default_rng(0))
        except res.MemoryLimitError as exc:
            point.update(status="refused-memory", where="state",
                         error="{}".format(exc).splitlines()[0])
            rec["points"].append(point)
            flush()
            continue
        point["state_gb"] = round(state_gb(state), 5)
        try:
            phases = network_memory_plan(ttno, state, n_roots=int(n_roots),
                                         max_bond=int(cap), rdm=False)
            plan_sweep = float(res.plan_peak_gb(phases))
            phases_rdm = network_memory_plan(ttno, state, n_roots=int(n_roots),
                                             max_bond=int(cap), rdm=True)
            plan_rdm = float(res.plan_peak_gb(phases_rdm))
            point["plan"] = {"sweep_peak_gb": round(plan_sweep, 4),
                             "with_rdm_peak_gb": round(plan_rdm, 4),
                             "environments_gb": round(environment_gb(ttno, state), 4),
                             "fits_limit": bool(plan_sweep <= lims.memory_gb),
                             "fits_limit_with_rdm": bool(plan_rdm <= lims.memory_gb)}
        except Exception as exc:                          # the plan itself may not fit
            point["plan"] = {"status": "failed",
                             "error": "{}: {}".format(type(exc).__name__, exc)[:300]}
        flush()
        print("    D={}: plan peak {} GB (with RDMs {} GB, environments {} GB) against "
              "{:.1f} GB".format(cap, point["plan"].get("sweep_peak_gb", "?"),
                                 point["plan"].get("with_rdm_peak_gb", "?"),
                                 point["plan"].get("environments_gb", "?"), lims.memory_gb),
              flush=True)
        sweep_times: List[Dict] = []
        t_sw = [time.time()]
        c_sw = [time.process_time()]
        # ⚠ The point is on disk from its first sweep, as "running": a cap whose second
        # sweep the budget kills still leaves the first sweep's cost, which at the top of
        # the ladder is the number the projection needs most.
        point.update(status="running", sweeps=sweep_times)
        rec["points"].append(point)

        def on_sweep(state, sweep, energies, converged):
            now, cpu = time.time(), time.process_time()
            sweep_times.append({"sweep": int(sweep), "wall_s": round(now - t_sw[-1], 1),
                                "cpu_s": round(cpu - c_sw[-1], 1),
                                "energies": [float(e) + ints.e_core for e in energies],
                                "rss_gb": round(rss_gb(), 3)})
            t_sw.append(now)
            c_sw.append(cpu)
            flush()
            heartbeat.tick(2, stage="sweep", cap=int(cap), sweep=int(sweep),
                           cpu=sweep_times[-1]["cpu_s"])

        t0, c0 = time.time(), time.process_time()
        try:
            result = solve_ttn(ttno, state, max_sweeps=int(max_sweeps),
                               conv_tol=BRIDGE_CONV_TOL, davidson_tol=TIER3_DAVIDSON_TOL,
                               max_bond=int(cap), n_elec=ints.n_elec, boundary_check=0,
                               on_split="warn", checkpoint=on_sweep, memory_plan=True,
                               plan_rdms=False, report=True)
        except SolverFailure as exc:
            # a local eigensolve that did not converge inside a sweep: the outcome a
            # plain ladder point records as unconverged, with the sweeps it completed
            point.update(status="unconverged", where="sweep", error="{}".format(exc)[:300],
                         wall_s=round(time.time() - t0, 1),
                         cpu_s=round(time.process_time() - c0, 1),
                         sweeps=sweep_times, phases=sampler.table())
            flush()
            print("    D={}: UNCONVERGED: {}".format(cap, point["error"][:140]), flush=True)
            continue
        except res.MemoryLimitError as exc:
            point.update(status="refused-memory", where="sweep",
                         error="{}".format(exc).splitlines()[0],
                         wall_s=round(time.time() - t0, 1),
                         cpu_s=round(time.process_time() - c0, 1),
                         sweeps=sweep_times, phases=sampler.table())
            flush()
            print("    D={}: REFUSED by the memory plan: {}".format(
                cap, point["error"][:120]), flush=True)
            continue
        except ValueError as exc:
            # the group-complete truncation rule: a cap that cuts a degenerate Schmidt
            # group is refused, and which caps are refused is part of the answer
            point.update(status="refused", where="sweep", error="{}".format(exc)[:300],
                         wall_s=round(time.time() - t0, 1),
                         cpu_s=round(time.process_time() - c0, 1),
                         sweeps=sweep_times, phases=sampler.table())
            flush()
            print("    D={}: REFUSED: {}".format(cap, point["error"][:120]), flush=True)
            continue
        energies = np.asarray(result.energies, dtype=float) + float(ints.e_core)
        e_cm = (energies - energies[0]) * HARTREE_TO_CM
        n_pair = int(energies.size) // 2
        point.update(status="ok" if result.converged else "bounded",
                     converged=bool(result.converged),
                     wall_s=round(time.time() - t0, 1),
                     cpu_s=round(time.process_time() - c0, 1),
                     n_sweeps=int(result.n_sweeps), w_disc=float(result.max_discarded),
                     bond_used=int(result.max_bond_dim),
                     saturating=bool(int(result.max_bond_dim) < int(cap)),
                     energies=[float(e) for e in energies],
                     rel_cm=[round(float(x), 4) for x in e_cm],
                     history=[float(e) + ints.e_core for e in result.history],
                     kramers_pair_spread_cm=round(float(np.max(
                         e_cm[1:2 * n_pair:2] - e_cm[0:2 * n_pair:2])), 6) if n_pair else None,
                     manifold_spread_cm=round(float(e_cm[-1]), 4),
                     n_paged_out=int(result.n_paged_out), n_paged_in=int(result.n_paged_in),
                     sweeps=sweep_times, phases=sampler.table(),
                     rss_peak_gb=round(max(r["peak_rss_gb"] for r in sampler.table()), 3),
                     ledger_resident_gb=round(res.BUDGET.resident_gb(), 4))
        flush()
        heartbeat.tick(3, stage="swept", cap=int(cap), cpu=point["cpu_s"])
        per_sweep = [s["cpu_s"] for s in sweep_times]
        print("    D={}: {} sweeps, {:.0f} CPU s (per sweep {}), w_disc {:.2e}, bond {}, "
              "peak RSS {:.2f} GB; E0 {:.8f}, manifold spread {:.3f} cm^-1, Kramers pair "
              "spread {} cm^-1".format(
                  cap, point["n_sweeps"], point["cpu_s"], per_sweep, point["w_disc"],
                  point["bond_used"], point["rss_peak_gb"], energies[0],
                  point["manifold_spread_cm"], point["kramers_pair_spread_cm"]), flush=True)
        # --- the analyses, each on disk before the next ------------------------------------
        if time.time() < deadline:
            point["spin"] = _spin_of_roots(ints, template, graph, result.state, cap, n_roots)
            flush()
            if point["spin"]["status"] == "ok":
                print("    D={}: <S^2> {} (S {}), {:.0f} CPU s".format(
                    cap, point["spin"]["s_squared"], point["spin"]["spin"],
                    point["spin"]["cpu_s"]), flush=True)
        if time.time() < deadline:
            point["product_model"] = _product_model(
                template, ints, result.state, result.weights, site_nodes, TIER3_SITE_DIM,
                ops, ints.n_elec, None)
            flush()
            pm = point["product_model"]
            print("    D={}: model {}{}, {:.0f} CPU s".format(
                cap, pm["status"],
                "" if pm.get("status") != "ok" else ": dim {}, spectrum {} cm^-1, site g {}"
                .format(pm["model_dim"], pm["rel_cm"][:8],
                        [["{:.4f}".format(x) for x in g] for g in pm.get("site_g", [])]),
                pm["cpu_s"]), flush=True)
        heartbeat.tick(4, stage="analysed", cap=int(cap))
    sampler.stop()
    flush("done")
    heartbeat.finish(status="done")
    return 0


def _run_child(cmd: Sequence[str], out_path: Path, share: float) -> Tuple[Dict, str, Optional[int]]:
    """Run one child to completion; ``(record, status, exit_code)`` from its own file."""
    t0 = time.time()
    killed = None
    try:
        proc = subprocess.run(list(cmd), timeout=share + 120.0, cwd=str(REPO))
        exit_code: Optional[int] = int(proc.returncode)
    except subprocess.TimeoutExpired:
        exit_code = None
        killed = "parent timeout after {:.0f} s".format(time.time() - t0)
    child: Dict = {}
    if out_path.is_file():
        try:
            child = json.loads(out_path.read_text())
        except ValueError:
            child = {}
    status = child.get("status", "no record")
    if killed is not None:
        status = "killed: {}".format(killed)
    elif exit_code is not None and exit_code < 0:
        status = "killed by signal {}".format(-exit_code)
    elif exit_code not in (0, None) and status in ("running", "no record"):
        status = "child exit {}".format(exit_code)
    return child, status, exit_code


def stage_tier3(record, heartbeat, *, deadline: float, keys: Sequence[str],
                caps: Optional[Sequence[int]] = None, max_sweeps: int = TIER3_SWEEPS) -> None:
    """S2.4a: the Tier-3 front end, then the feasibility ladder — two child processes.

    The parent holds nothing but the record, so the memory the front end leaves behind
    (the factors, the SCF) is not resident beside the sweep — the residue-plus-next-phase
    kill is a recorded failure mode of this campaign. ⚠ A finished child record is
    adopted, never re-run: delete ``temp/dmrg_cost_ladder/s2.4a/<key>_*.json`` to
    re-measure a step.
    """
    ladder = list(TIER3_CAPS if caps is None else caps)
    out_dir = camp.RECORDS / "s2.4a"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        system = camp.get(key)
        front_path = out_dir / "{}_front.json".format(key)
        ints_path = tier3_integrals_path(key)
        front: Dict = {}
        if front_path.is_file():
            try:
                front = json.loads(front_path.read_text())
            except ValueError:
                front = {}
        if front.get("status") == "ok" and ints_path.is_file():
            print("  [{}] front end adopted from {}".format(key, front_path), flush=True)
        else:
            share = min(TIER3_FRONT_END_BUDGET_S, max(60.0, deadline - time.time() - 60.0))
            cmd = [sys.executable, str(Path(__file__).resolve()), "--tier3-front-end", key,
                   str(front_path), "--budget", "{:.0f}".format(share)]
            print("  [{}] front-end child: {:.0f} s share -> {}".format(key, share, front_path),
                  flush=True)
            front, status, exit_code = _run_child(cmd, front_path, share)
            front["child_exit"] = exit_code
            if status != "ok":
                front["status"] = status
        job = {"key": key, "stage_kind": "tier3", "label": system.label,
               "roots": int(system.n_states), "n_active": system.n_active,
               "n_active_elec": system.n_active_elec, "n_det": system.n_det,
               "protocol_note": system.protocol_note, "caps": ladder,
               "sweeps": int(max_sweeps), "front": front}
        # a resumed record replaces its job for this key: the front end may have failed
        # last time and been re-run now, and the record must say what stands
        record.data["jobs"] = [j for j in record.data["jobs"] if j.get("key") != key]
        record.add_job(job)
        heartbeat.tick(0, system=key, stage="front-end", status=front.get("status"))
        if front.get("status") != "ok":
            record.add_point({"key": key, "topology": "r{}".format(system.n_states),
                              "cap": None, "status": "front-end-failed",
                              "reason": "{}: {}".format(front.get("status"),
                                                        front.get("error", ""))[:300]})
            print("  [{}] FRONT END FAILED: {}: {}".format(
                key, front.get("status"), front.get("error", "")[:200]), flush=True)
            continue
        fe = front["front_end"]
        print("  [{}] front end: {} AOs, {} Cholesky vectors, SCF E = {:.8f} Eh ({:.0f} CPU s); "
              "{}; localized site populations min {}".format(
                  key, fe["nao"], fe["n_cholesky"], fe["e_scf"], fe["cpu_s"],
                  front["active_space"]["description"],
                  front["localization"]["per_site_population_min"]), flush=True)
        lane = "r{}".format(system.n_states)
        ladder_path = out_dir / "{}_ladder.json".format(key)
        child: Dict = {}
        if ladder_path.is_file():
            try:
                child = json.loads(ladder_path.read_text())
            except ValueError:
                child = {}
        if child.get("status") not in (None, "running"):
            status = child["status"]
            print("  [{}/{}] ladder adopted from {} ({})".format(key, lane, ladder_path,
                                                                 status), flush=True)
        else:
            share = max(60.0, deadline - time.time() - 60.0)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--tier3-ladder", key,
                   str(ints_path), str(ladder_path), "--caps",
                   ",".join(str(c) for c in ladder), "--roots", str(system.n_states),
                   "--max-sweeps", str(max_sweeps), "--budget", "{:.0f}".format(share)]
            print("  [{}/{}] ladder child: {:.0f} s share, caps {} -> {}".format(
                key, lane, share, ladder, ladder_path), flush=True)
            child, status, _ = _run_child(cmd, ladder_path, share)
        for job_ in record.data["jobs"]:
            if job_.get("key") == key:
                job_["partition"] = child.get("partition")
                job_["compile"] = child.get("steps", {}).get("compile")
                job_["ladder_status"] = status
        for p in child.get("points", []):
            if p.get("status") == "running":
                # the sweep the budget killed: its completed sweeps are the record
                p = dict(p, status="killed", reason=status)
            record.add_point(dict(p, key=key, topology=lane))
        record.flush()
        heartbeat.tick(1, system=key, stage="ladder", status=status)
        print("  [{}/{}] ladder {}: {} point(s)".format(key, lane, status,
                                                        len(child.get("points", []))),
              flush=True)


def summarize_tier3(record_path: Path) -> List[Dict]:
    """Per cap: the cost shape and what the bounded state already says."""
    data = json.loads(record_path.read_text())
    rows: List[Dict] = []
    for job in data.get("jobs", []):
        key = job["key"]
        for p in data.get("points", []):
            if p.get("key") != key or p.get("cap") is None:
                continue
            sweeps = p.get("sweeps", [])
            spin = p.get("spin", {})
            pm = p.get("product_model", {})
            rows.append({"key": key, "cap": int(p["cap"]), "status": p.get("status"),
                         "bond_used": p.get("bond_used"), "w_disc": p.get("w_disc"),
                         "n_sweeps": p.get("n_sweeps"),
                         "cpu_per_sweep": [s["cpu_s"] for s in sweeps],
                         "wall_per_sweep": [s["wall_s"] for s in sweeps],
                         "plan_peak_gb": p.get("plan", {}).get("sweep_peak_gb"),
                         "plan_rdm_gb": p.get("plan", {}).get("with_rdm_peak_gb"),
                         "rss_peak_gb": p.get("rss_peak_gb"),
                         "paged": [p.get("n_paged_out"), p.get("n_paged_in")],
                         "e0": (p.get("energies") or [None])[0],
                         "manifold_spread_cm": p.get("manifold_spread_cm"),
                         "kramers_pair_spread_cm": p.get("kramers_pair_spread_cm"),
                         "spin": spin.get("spin"), "spin_status": spin.get("status"),
                         "model_status": pm.get("status"),
                         "site_g": pm.get("site_g"), "twice_s": pm.get("twice_s"),
                         "error": p.get("error")})
    return rows


def print_tier3_summary(rows: Sequence[Dict]) -> None:
    head = "{:<11s} {:>5s} {:<9s} {:>5s} {:>9s} {:>16s} {:>7s} {:>7s} {:>13s} {:>12s}  {}".format(
        "system", "D", "status", "bond", "w_disc", "CPU s / sweep", "plan", "RSS",
        "E0 [Eh]", "spread cm^-1", "S per root / model")
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        print("{:<11s} {:>5d} {:<9s} {:>5s} {:>9s} {:>16s} {:>7s} {:>7s} {:>13s} {:>12s}  {}".format(
            r["key"], r["cap"], (r["status"] or "-")[:9], str(r["bond_used"] or "-"),
            "-" if r["w_disc"] is None else "{:.2e}".format(r["w_disc"]),
            "/".join("{:.0f}".format(x) for x in r["cpu_per_sweep"]) or "-",
            "-" if r["plan_peak_gb"] is None else "{:.2f}".format(r["plan_peak_gb"]),
            "-" if r["rss_peak_gb"] is None else "{:.2f}".format(r["rss_peak_gb"]),
            "-" if r["e0"] is None else "{:.6f}".format(r["e0"]),
            "-" if r["manifold_spread_cm"] is None else "{:.3f}".format(r["manifold_spread_cm"]),
            ("{} / {}".format(r["spin"], r["model_status"]) if r["spin"] else
             (r["error"] or "-")[:60])))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """The child entry point of S2.2 (the stages themselves run under
    :mod:`dmrg_cost_ladder`)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feasibility-child", nargs=3, metavar=("INTS", "MODES", "OUT"))
    ap.add_argument("--tier3-front-end", nargs=2, metavar=("KEY", "OUT"))
    ap.add_argument("--tier3-ladder", nargs=3, metavar=("KEY", "INTS", "OUT"))
    ap.add_argument("--caps", default=",".join(str(c) for c in FEASIBILITY_CAPS))
    ap.add_argument("--roots", type=int, default=FEASIBILITY_ROOTS)
    ap.add_argument("--max-sweeps", type=int, default=FEASIBILITY_SWEEPS)
    ap.add_argument("--budget", type=float, default=1800.0)
    args = ap.parse_args(argv)
    if (args.feasibility_child is None and args.tier3_front_end is None
            and args.tier3_ladder is None):
        ap.error("this module is driven by dmrg_cost_ladder.py --stage s2.1 / s2.2 / "
                 "s2.4a; the direct entry points are the child processes "
                 "(--feasibility-child, --tier3-front-end, --tier3-ladder)")
    out = (args.feasibility_child or args.tier3_front_end or args.tier3_ladder)[-1]
    from kuiva.util import logging as klog
    import logging
    klog.set_verbosity("INFO")
    log_path = Path(out).with_suffix(".log")
    klog.add_file_handler(log_path, level=logging.INFO)
    for handler in logging.getLogger("kuiva").handlers:
        if isinstance(handler, logging.StreamHandler) \
                and not isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.WARNING)
    if args.tier3_front_end is not None:
        key, out = args.tier3_front_end
        return tier3_front_end_child(key, Path(out), budget=float(args.budget))
    if args.tier3_ladder is not None:
        key, ints, out = args.tier3_ladder
        return tier3_ladder_child(key, Path(ints), Path(out),
                                  caps=[int(x) for x in args.caps.split(",")],
                                  n_roots=int(args.roots), max_sweeps=int(args.max_sweeps),
                                  budget=float(args.budget))
    ints, modes, out = args.feasibility_child
    return feasibility_child(Path(ints), int(modes), Path(out),
                             caps=[int(x) for x in args.caps.split(",")],
                             n_roots=int(args.roots), max_sweeps=int(args.max_sweeps),
                             budget=float(args.budget))


if __name__ == "__main__":
    raise SystemExit(main())
