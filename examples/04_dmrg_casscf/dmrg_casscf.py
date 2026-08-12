"""Example 4 -- the tensor-network solver: a cheap CI, a topology, and DMRG-CASSCF.

    source setup.sh          # once per shell
    python dmrg_casscf.py

Runs in three to four minutes on TiF3 and writes ``output/dmrg_casscf.out``.

WHAT THIS SHOWS
---------------
The conventional CI of example 3 stores one amplitude per determinant, which caps the
active space at around twenty spinors at half filling. Beyond that the CI vector cannot be
held at all and a tensor network takes over. Kuiva's in-house solver is tree-native -- a
matrix product state is the path special case -- and it plugs into the *same* orbital
optimizer through the *same* contract as the exact CI: RDMs in, orbital rotation out.

Two stages appear here for the first time:

    CheapCI     a cheap pre-optimization that rotates the raw spinor guess toward physical
                active orbitals. Its total energy means nothing and is deliberately not
                exposed; what it claims -- and what it is for -- is that the occupations
                converge long before the energy does. Its two products feed the stages
                after it: the rotated orbitals start the CASSCF, and the entanglement it
                measures seeds the network topology.

    CASSCF(solver="dmrg")
                the same CASSCF stage, with the CI step done by the network. ``max_bond``
                caps the bond dimension; ``graph="mutual-information"`` builds the tree
                from the cheap CI's entanglement instead of assuming a chain.

The honest framing: this example checks the network solver against the exact CI on a space
the exact CI can still handle, because that is the only way to check it. Past the ceiling
there is nothing to compare against, which is exactly why the agreement below has to be
established here.

WHAT TO LOOK FOR IN THE OUTPUT
------------------------------
* the cheap CI's orbital occupations -- two spinors at 0.5 and the rest empty, which is
  what one electron shared over a Kramers pair looks like -- and the single-orbital
  entropies and mutual information built from them;
* the topology the mutual information produced: ten nodes and nine edges, a *tree* rather
  than a chain, with no sweep, operator or environment code aware of the difference;
* the two CASSCF energies agreeing to well below the convergence threshold, and the
  largest bond dimension the network actually needed staying far under the cap.

A NOTE ON THE ORBITAL OPTIMIZER
-------------------------------
Both CASSCFs below ask for ``mode="quasi-newton"`` rather than the default ``"auto"``. The
default escalates to a second-order step when the gradient trajectory stalls, which is the
right behaviour for a hard orbital problem but spends Hessian-vector products. Pinning both
runs to the same step engine is what makes the comparison a comparison between the two CI
methods rather than between two optimizer trajectories.

Worth knowing while choosing whether to use ``CheapCI`` at all: for a single d^1 centre
like this one, the canonical SCF orbitals are already an excellent starting point and the
pre-optimization costs more iterations than it saves. It earns its keep where a single
determinant is qualitatively wrong -- several open shells, antiferromagnetically coupled
centres -- which is also where the network solver is needed.
"""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import List

import numpy as np

import kuiva
from kuiva.util import output as out
from kuiva.util import resources as res
from kuiva.util import timing
from kuiva.util.logging import add_file_handler, get_logger

NAME = "dmrg_casscf"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

log = get_logger("examples." + NAME)

#: Planar D3h TiF3, Ti-F [Angstrom]. The same d^1 physics as example 3 in a smaller basis,
#: so that two full CASSCF optimizations fit inside the cost budget for one example.
R_TIF = 1.780

#: The 3d shell again: five Kramers pairs, one electron, the ground Kramers doublet.
N_ACTIVE, N_ACTIVE_ELEC, N_STATES = 10, 1, 2

#: The bond-dimension cap. An uncapped tree state allocates charge-sector-maximal bonds,
#: so the solver requires this rather than defaulting it.
MAX_BOND = 16

#: Budgets, explicit so termination never depends on convergence.
MAX_ITER, CONV_GRAD = 60, 1.0e-4

#: The two solvers see the same integrals and the same orbitals, so their state-averaged
#: energies must agree far below the convergence threshold, not merely near it.
SOLVER_AGREEMENT_EH = 1.0e-8


def planar_mx3(metal: str, ligand: str, r: float) -> List[tuple]:
    """Planar D3h MX3, metal at the origin, ligands in the xy plane."""
    atoms = [(metal, (0.0, 0.0, 0.0))]
    for k in range(3):
        theta = 2.0 * math.pi * k / 3.0
        atoms.append((ligand, (r * math.cos(theta), r * math.sin(theta), 0.0)))
    return atoms


def prepare_output() -> Path:
    """Write this run to output/<name>.out, with a scratch spin-orbit cache of its own."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT / "amf-cache"
    shutil.rmtree(cache, ignore_errors=True)
    os.environ["KUIVA_AMF_CACHE"] = str(cache)
    path = OUTPUT / (NAME + ".out")
    add_file_handler(path)
    return path


def entanglement_table(pre) -> None:
    """Report what the cheap CI measured about the active orbitals."""
    out.subsection(log, "What the cheap CI measured")
    table = out.Table(log, [
        out.col_count("active spinor", 14),
        out.Column("occupation", "{:.4f}", 12),
        out.Column("entropy", "{:.4f}", 10),
        out.Column("max mutual info", "{:.4e}", 17),
    ])
    table.start()
    info = np.asarray(pre.mutual_information, dtype=float)
    for k, (occ, ent) in enumerate(zip(pre.occupations, pre.entropy)):
        row = np.delete(info[k], k)
        table.row(k, float(occ), float(ent), float(np.max(row)) if row.size else 0.0)
    table.end("single-orbital entropy is zero for an empty or full spinor by definition")

    # A lower bound, and it says so: occupation-based selection cannot flag an orbital that
    # is empty here but would be populated by a better treatment. Combine it with orbital
    # character and near-degeneracy, as any active-space choice must be.
    out.entry(log, "fractionally occupied spinors (a LOWER bound on the active space)",
              str(pre.suggested_active().tolist()))


def main() -> int:
    outfile = prepare_output()
    out.banner(log, kuiva.__version__, "example 4: DMRG-CASSCF on TiF3")
    out.entry(log, "output file", str(outfile.relative_to(HERE)))

    # ----------------------------------------------------------------------------------
    # 1. Front end.
    # ----------------------------------------------------------------------------------
    molecule = kuiva.Molecule(atoms=planar_mx3("Ti", "F", R_TIF), basis="x2c-SVPall-2c",
                              charge=0, spin=1)
    scf = kuiva.ScalarSCF(molecule, memory_gb=6.0).run()
    reference = kuiva.Reference(scf).run()

    out.section(log, "Problem")
    out.entries(log, [
        ("system", "TiF3", "", "planar D3h, Ti-F = {:.3f} A".format(R_TIF)),
        ("physics", "Ti(III) d^1 in a ligand field"),
        ("spinors", reference.nspinor),
        ("active space", "CAS({}, {})".format(N_ACTIVE_ELEC, N_ACTIVE), "",
         "the Ti 3d shell, selected by orbital character"),
        ("states", N_STATES, "", "the ground Kramers doublet"),
        ("bond-dimension cap", MAX_BOND),
    ])

    # ----------------------------------------------------------------------------------
    # 2. The cheap pre-optimization.
    # ----------------------------------------------------------------------------------
    # The reference space is a small CAS around the Fermi level rather than a single
    # determinant, because for coupled open-shell centres a single determinant is
    # qualitatively wrong.
    #
    # The stage also restores exact Kramers pairing of the orbitals it produces. The
    # truncated cheap CI is not closed under time reversal, so its optimized orbitals drift
    # off the pairing convention -- legitimately, it is a cheap stage -- while everything
    # downstream, the state-averaging gate included, assumes that convention exactly.
    pre = kuiva.CheapCI(reference, character=("Ti", "d"), n_active=N_ACTIVE,
                        n_active_elec=N_ACTIVE_ELEC, n_states=N_STATES).run()

    out.section(log, "The finished CheapCI stage")
    log.info("%s", pre.summary())
    entanglement_table(pre)

    # ----------------------------------------------------------------------------------
    # 3. The same CASSCF twice: exact CI, then the tensor network.
    # ----------------------------------------------------------------------------------
    out.section(log, "CASSCF with the conventional CI")
    exact = kuiva.CASSCF(pre, n_states=N_STATES, mode="quasi-newton", max_iter=MAX_ITER,
                         conv_grad=CONV_GRAD).run()
    log.info("%s", exact.summary())

    out.section(log, "CASSCF with the tensor-network solver")
    # graph="mutual-information" builds the topology from the cheap CI's entanglement;
    # "fiedler" would order the spinors along a chain instead, and a NetworkGraph object
    # can be passed directly. solver_options go to the network solver, optimizer options
    # (mode, max_iter, conv_grad) to the shared orbital optimizer -- the same one the exact
    # CI just used, unchanged.
    network = kuiva.CASSCF(pre, n_states=N_STATES, solver="dmrg",
                           solver_options=dict(max_bond=MAX_BOND),
                           graph="mutual-information", mode="quasi-newton",
                           max_iter=MAX_ITER, conv_grad=CONV_GRAD).run()
    log.info("%s", network.summary())

    graph = network.graph
    out.subsection(log, "The network")
    out.entries(log, [
        ("nodes", graph.n_nodes, "", "one active spinor each"),
        ("edges", len(graph.edges), "",
         "a tree: {} nodes need {} edges".format(graph.n_nodes, graph.n_nodes - 1)),
        ("topology", " ".join("{}-{}".format(u, v) for u, v in graph.edges)),
        ("largest bond dimension used", network.solver.last.max_bond_dim, "",
         "cap was {}".format(MAX_BOND)),
    ])
    out.note(log, "the solver is tree-native and nothing in the sweep, the operators or")
    out.note(log, "the environments knows 'left' from 'right'; a chain is the special case")
    out.note(log, "where the tree happens to be a path.")

    # ----------------------------------------------------------------------------------
    # 4. Side by side.
    # ----------------------------------------------------------------------------------
    out.section(log, "Exact CI against the network")
    table = out.Table(log, [
        out.Column("solver", "{}", 26, align="<"),
        out.col_iter("iter"),
        out.col_energy("E(CASSCF) [Eh]"),
        out.col_resid("|grad|"),
        out.Column("conv", "{}", 6),
    ])
    table.start()
    for label, cas in (("conventional CI", exact), ("tree tensor network", network)):
        table.row(label, cas.orbital.n_iterations, cas.energy, cas.orbital.grad_norm,
                  "yes" if cas.converged else "NO")
    table.end()

    difference = abs(exact.energy - network.energy)
    splitting = float(abs(network.energies[1] - network.energies[0]))
    out.entries(log, [
        ("|E(network) - E(CI)|", difference, "Eh", "", out.SCI_FMT),
        ("Kramers splitting of the network doublet", splitting, "Eh", "", out.SCI_FMT),
        ("state-average boundary", "complete" if network.boundary_gap_cm is None
         else "{:.1f} cm^-1 to the next root".format(network.boundary_gap_cm)),
        ("network states [Eh]", ", ".join(out.E_FMT.format(e)
                                          for e in network.energies)),
    ])

    # ----------------------------------------------------------------------------------
    # 5. Assert.
    # ----------------------------------------------------------------------------------
    checks = {
        "the conventional-CI CASSCF converged": bool(exact.converged),
        "the tensor-network CASSCF converged": bool(network.converged),
        "the two solvers agree to {:.0e} Eh".format(SOLVER_AGREEMENT_EH):
            difference < SOLVER_AGREEMENT_EH,
        "the network doublet is Kramers degenerate": splitting < 1.0e-6,
        "the topology is a tree over the active spinors":
            graph.n_nodes == N_ACTIVE and len(graph.edges) == N_ACTIVE - 1,
        "the bond dimension stayed inside its cap":
            int(network.solver.last.max_bond_dim) <= MAX_BOND,
        "the cheap CI found the expected fractional occupations":
            int(np.sum(np.asarray(pre.occupations) > 0.1)) == 2,
    }
    failures = report(checks)

    timing.summary(log)
    res.summary(log)
    return 1 if failures else 0


def report(checks) -> int:
    """Print the check table and return the number of failures."""
    out.section(log, "Result")
    table = out.Table(log, [out.Column("check", "{}", 62, align="<"),
                            out.Column("verdict", "{}", 10, align="<")])
    table.start()
    for label, ok in checks.items():
        table.row(label, "ok" if ok else "FAILED")
    table.end()
    failed = [label for label, ok in checks.items() if not ok]
    for label in failed:
        log.error("check failed: %s", label)
    out.entry(log, "checks", "FAILED ({})".format(len(failed)) if failed
              else "all {} passed".format(len(checks)))
    return len(failed)


if __name__ == "__main__":
    raise SystemExit(main())
