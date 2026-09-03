"""What a tensor-network solve will hold, stated before it holds any of it.

Why this module exists
----------------------
The resource budget's promise is that an over-large calculation is refused *before* it
allocates, with a diagnosis naming the knob to turn. The dense stages keep it: their plan is
built in the front end from the AO and active-space dimensions, and the pre-flight table is
printed before the SCF starts.

The network layer could not keep it, for a reason that is structural rather than an
oversight. Its largest array is not a declared object but a **contraction intermediate**
inside one application of the effective Hamiltonian — built, used and freed inside a single
Davidson matrix-vector product, hundreds of times per bond. Nothing on the ledger ever saw
it, so nothing could refuse it: measured on a 20-spinor double-shell network, the process
oscillated between 4 and 11 GB against a ledger that declared 0.13 GB, and the run ended as
a kernel OOM kill with no message at all. On a shared machine that takes the neighbours with
it; on any machine it means the only way to find the reachable bond dimension is to be
killed by it.

What is planned here
--------------------
Everything a sweep adds to what the compiled operator and the state already hold, sized from
structure alone (:class:`kuiva.dmrg.block.BlockShape`) and therefore exactly, with nothing
allocated and no contraction performed:

* the **environment cache**, which grows to every directed bond over one sweep and is
  released only when the solve ends;
* the **two-site solves**, whose resident part is the merged ensemble, the packed guess, the
  Davidson stacks, the roots and the pre-contracted operator halves a branching node may
  take, and whose transient part is the effective-Hamiltonian application above — the term
  that was missing;
* the **RDM contraction**, which is a second environment set plus the ``n^4`` two-particle
  array, and which runs after the sweep on the same state.

⚠ **The bond that peaks is found, not assumed.** The tour visits every bond, the local
dimensions differ by a factor of many between a leaf bond and an interior one, and the
largest is what has to fit — so every bond of the sweep schedule is sized and the maximum
reported, with the bond named in the plan's note so a refusal points somewhere.

⚠ **A plan, not a bound.** Bond dimensions are re-derived inside the cap by every truncation,
so a sweep can arrive at a distribution the initial state did not have. The plan is built
from the state the sweep starts on; what makes the guarantee hard is the per-bond
``require`` in :func:`kuiva.dmrg.sweep._solve_local`, which is exact for the problem in front
of it and refuses at the first bond that does not fit. This module is what lets a user see it
coming.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..util import resources as res
from .block import BlockShape
from .sweep import ShapeEnvironments, TTNState, environment_gb
from .ttno import TTNO


def _recentred(ttno: TTNO, state: TTNState, center: int) -> TTNState:
    """The state's structure as it will be once the sweep has moved the center to ``center``.

    ⚠ A two-site problem is solved at every bond of the tour and they are not the same size
    — a leaf bond and an interior one differ by orders of magnitude here — so the plan has
    to look at all of them, and every one but the first belongs to a *different* centring.
    The node tensors are rebuilt through :func:`kuiva.dmrg.sweep.node_layouts`, the one
    statement of the leg-and-sign convention, from the bond spaces the state currently has.
    """
    from .sweep import node_layouts

    graph = state.graph
    bond_spaces = {(min(a, b), max(a, b)): state.bond_space(a, b) for a, b in graph.edges}
    layouts = node_layouts(graph, bond_spaces, ttno.phys_space, int(center), state.charge)
    tensors = [None if u == center else BlockShape.allowed(*layouts[u])
               for u in range(graph.n_nodes)]
    centre = BlockShape.allowed(*layouts[int(center)])
    return TTNState(graph=graph, center=int(center), tensors=tensors,
                    centers=[centre] * state.n_roots, charge=state.charge)


def two_site_peak(ttno: TTNO, state: TTNState, *, n_roots: int,
                  extra_roots: int = 0) -> Tuple[float, float, Optional[Tuple[int, int]]]:
    """``(workspace_gb, apply_gb, bond)`` at the bond of the sweep whose solve is largest.

    Both terms come from the same :class:`~kuiva.dmrg.sweep._LocalProblem` the sweep builds,
    run over structure instead of data — so there is no second description of the two-site
    problem to drift from the one that is solved.

    ⚠ Every bond carries the *full* allowed sector set at its centring, which is what a
    freshly canonicalized state has and an upper bound on what a truncated one keeps. That
    is the one place this forecast is not exact, and it errs toward reporting more rather
    than less — the opposite direction from the failure it exists to prevent.
    """
    from .sweep import _LocalProblem                      # local: _LocalProblem is private

    best = (0.0, 0.0, None)
    for u, v in state.graph.sweep_schedule(state.center):
        shapes = _recentred(ttno, state, u)
        problem = _LocalProblem(ttno, shapes, ShapeEnvironments(ttno, shapes), u, v)
        n_solve = min(int(n_roots) + int(extra_roots), problem.dim)
        workspace = problem.solve_workspace_gb(int(n_roots), n_solve)
        apply_gb = problem.apply_peak_gb()
        if workspace + apply_gb > best[0] + best[1]:
            best = (workspace, apply_gb, (u, v))
    return best


def network_memory_plan(ttno: TTNO, state: TTNState, *, n_roots: int,
                        extra_roots: int = 0, max_bond: Optional[int] = None,
                        rdm: bool = True) -> List[res.PhaseEstimate]:
    """The phases a two-site sweep adds, for :func:`kuiva.util.resources.preflight`.

    ⚠ The compiled operator and the state itself are **deliberately absent**: both are
    reserved on the ledger before this is called (at the TTNO compile and in
    :func:`kuiva.dmrg.sweep.random_state`), and the pre-flight's peak model already carries
    everything resident forward. Listing them again would double-count them and turn a plan
    into a refusal of calculations that fit.
    """
    n = sum(len(c) for c in state.graph.contents)
    envs = environment_gb(ttno, state)
    workspace, apply_gb, bond = two_site_peak(ttno, state, n_roots=n_roots,
                                              extra_roots=extra_roots)
    cap_note = "" if max_bond is None else "; max_bond = {}".format(max_bond)
    phases = [
        res.PhaseEstimate(name="network environments", allocations=[
            res.PlannedAllocation(
                "renormalized operator blocks", envs,
                note="{} directed bonds, held until the solve ends{}".format(
                    2 * len(state.graph.edges), cap_note)),
        ], advice=["reduce max_bond: an environment scales as D^2 times the operator "
                   "bond dimension",
                   "environment_paging=True pages the coldest entries to scratch"]),
        res.PhaseEstimate(name="two-site solves", allocations=[
            res.PlannedAllocation(
                "two-site workspace", workspace,
                note="largest bond of the tour{}: merged ensemble, guess, Davidson "
                     "stacks, roots and operator halves over {} root{}".format(
                         "" if bond is None else " {}".format(bond), n_roots,
                         "" if n_roots == 1 else "s")),
            res.PlannedAllocation(
                "H_eff application", apply_gb, resident=False,
                note="one contraction intermediate, built and freed once per "
                     "matrix-vector product"),
        ], advice=[
            "reduce max_bond: the two-site dimension scales as D^2 d^2 and the "
            "application's intermediate carries one operator bond dimension on top of "
            "that",
            "reduce n_states: the Davidson stacks scale with the subspace cap, which is a "
            "multiple of the root count",
            "a finer node partition (more nodes, fewer modes each) shrinks the "
            "application's intermediate quadratically in the local dimension"]),
    ]
    if rdm:
        phases.append(res.PhaseEstimate(name="network RDMs", allocations=[
            res.PlannedAllocation("state-averaged 2-RDM", res.rdm_gb(n, 2),
                                  note="{} spinors".format(n)),
            res.PlannedAllocation("per-node environments", envs, resident=False,
                                  note="a second environment set, released with the "
                                       "contraction"),
        ], advice=["the 2-RDM is n^4 in the active space and no setting moves it"]))
    return phases


__all__ = ["network_memory_plan", "two_site_peak"]
