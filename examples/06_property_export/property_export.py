"""Example 6 -- the two formatted products: the property dump and the pseudospin export.

    source setup.sh          # once per shell
    python property_export.py

Runs in three to four minutes on TiCl3 and writes ``output/property_export.out`` plus the
two data files it is about.

WHAT THIS SHOWS
---------------
Kuiva does not evaluate magnetic properties, crystal-field parameters or Stevens operators.
It writes the matrices an external code needs in order to do that, and this example is
about those files.

    PropertyDump        the effective Hamiltonian H and the three magnetic-moment
                        components mu_x, mu_y, mu_z, in the basis of the spin-orbit
                        eigenstates, as plain self-describing text with a versioned header.
                        For an ITO / crystal-field code.

    PseudospinExport    the same physics contracted onto a local-multiplet model space:
                        H_eff, the moment operators, the ordered pseudospin product basis
                        and the unitary that maps the ab initio states onto it. For the
                        external OuluSpin code, in its conventions and its storage order.

Both are contracts with programs outside this one, so both carry the full provenance of the
Hamiltonian that produced them -- in particular which spin-orbit screening the Hamiltonian
already contained, because that is worth 5 to 30 per cent on every splitting in the file
and a file that does not say is not interpretable.

THREE THINGS THE FILES SAY IN THEIR HEADERS, AND WHY
----------------------------------------------------
* **H is diagonal in the property dump.** Kuiva's CI is already two-component, so its roots
  *are* the spin-orbit eigenstates -- there is no separate spin-orbit mixing step. A reader
  arriving from a two-step (spin-free CI, then spin-orbit mixing) workflow will expect
  otherwise, so the header says so. In the pseudospin file H is *not* diagonal, because
  there the basis is the pseudospin product basis rather than the eigenbasis.

* **No picture-change correction is applied to L and S.** They are the bare
  non-relativistic AO operators, used unchanged in the two-component basis. This matches
  what OpenMolcas RASSI does, which keeps a cross-code comparison like for like, but the
  size of the approximation has not been measured. Every run emits that warning and records
  it in the file; it is not configurable, because the file outlives the session.

* **Phases are arbitrary and are never canonicalized.** Degenerate spin-orbit states mix
  freely, so an element-by-element comparison of a moment matrix -- against another program,
  or against this one run twice -- is meaningless. What can be compared are degeneracy
  patterns, relative energies and the invariant Tr_block(mu_i mu_j), whose principal values
  are the g factors. The example uses that reduction and nothing else.

WHAT TO LOOK FOR IN THE OUTPUT
------------------------------
* the ten states resolving into five Kramers doublets, each with its principal g values;
* the inactive contribution to L coming out at zero -- exact for a Kramers-paired inactive
  set, and computed rather than assumed, because a nonzero value would be a statement about
  the orbitals;
* the ground doublet's g tensor being *anisotropic*: a D3h ligand field defines an axis, so
  it must be. (A free ion is the opposite check -- example 2 -- and having both is what
  shows the machinery responds to real anisotropy instead of returning a plausible
  constant.)
* the same g values coming out of the pseudospin export, which reaches them by contracting
  the network onto a model space rather than through CI transition densities: two
  independent routes to one invariant.
"""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import List

import numpy as np

import kuiva
from kuiva.props.dump import read_dump
from kuiva.props.pseudospin import read_pseudospin
from kuiva.util import output as out
from kuiva.util import timing
from kuiva.util.logging import add_file_handler, get_logger

NAME = "property_export"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

log = get_logger("examples." + NAME)

#: Planar D3h TiCl3, Ti-Cl [Angstrom]: Ti(III) d^1 in a ligand field, as in example 3.
R_TICL = 2.25

#: The 3d shell and every root of it. Ten spinors holding one electron give exactly ten
#: determinants, so averaging over all ten states makes the state average complete by
#: construction -- there is no boundary that could fall inside a degenerate manifold -- and
#: puts all five ligand-field doublets on an equal footing, which is what a property file
#: for a crystal-field analysis wants.
N_ACTIVE, N_ACTIVE_ELEC, N_STATES = 10, 1, 10

#: Budgets, explicit so termination never depends on convergence.
MAX_ITER, CONV_GRAD = 100, 1.0e-4

#: The two property routes share the states but not the arithmetic: one contracts CI
#: transition densities, the other contracts a tensor network onto a model space. They must
#: agree far inside anything physical.
G_AGREEMENT = 1.0e-6


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


def main() -> int:
    outfile = prepare_output()
    out.banner(log, kuiva.__version__, "example 6: property and pseudospin export")
    out.entry(log, "output file", str(outfile.relative_to(HERE)))

    # ----------------------------------------------------------------------------------
    # 1. The calculation whose product these files are.
    # ----------------------------------------------------------------------------------
    molecule = kuiva.Molecule(atoms=planar_mx3("Ti", "Cl", R_TICL),
                              basis="x2c-SVPall-2c", charge=0, spin=1)
    scf = kuiva.ScalarSCF(molecule, memory_gb=6.0).run()
    reference = kuiva.Reference(scf).run()
    cas = kuiva.CASSCF(reference, character=("Ti", "d"), n_active=N_ACTIVE,
                       n_active_elec=N_ACTIVE_ELEC, n_states=N_STATES,
                       max_iter=MAX_ITER, conv_grad=CONV_GRAD).run()

    out.section(log, "The reference calculation")
    log.info("%s", cas.summary())
    out.entry(log, "state average",
              "complete" if cas.boundary.gap_cm is None
              else "{:.1f} cm^-1 to the next root".format(cas.boundary.gap_cm),
              "", "all {} determinants of the 3d shell".format(N_STATES))

    # ----------------------------------------------------------------------------------
    # 2. The property-matrix file.
    # ----------------------------------------------------------------------------------
    # The gauge origin that L is defined relative to was fixed at ingestion (centre of mass
    # by default), not here: the multireference layer never calls the front end again, so
    # the choice has to be made where the integrals are produced.
    dump_path = OUTPUT / "ticl3.props"
    dump = kuiva.PropertyDump(
        cas, dump_path,
        title="TiCl3 d^1, CAS({}, {}), SA-CASSCF({})".format(N_ACTIVE_ELEC, N_ACTIVE,
                                                             N_STATES),
        comments=["planar D3h, Ti-Cl = {:.2f} A".format(R_TICL)]).run()

    out.section(log, "The property-matrix file")
    log.info("%s", dump.summary())

    # Read it back with Kuiva's own reader, which is what an external consumer's parser has
    # to reproduce. format_version in the header exists so a consumer can refuse rather than
    # misinterpret; it is bumped when the MEANING of a stored field changes, not when a
    # field is added.
    stored = read_dump(dump_path)
    hamiltonian = stored["provenance"]["hamiltonian"]
    h_matrix = stored["matrices"]["H"]
    off_diagonal = float(np.abs(h_matrix - np.diag(np.diag(h_matrix))).max())
    round_trip = all(np.array_equal(stored["matrices"]["mu_" + axis], dump.matrices.mu[k])
                     for k, axis in enumerate("xyz"))

    out.entries(log, [
        ("file", str(dump_path.relative_to(HERE))),
        ("size", dump_path.stat().st_size / 1024.0, "kB", "", "{:.1f}"),
        ("matrices", len(stored["matrices"]), "",
         " ".join(sorted(stored["matrices"]))),
        ("format version", stored["header"]["format_version"]),
        ("gauge origin", stored["header"]["gauge_origin_choice"]),
        ("Hamiltonian in the header", hamiltonian["method"], "",
         "screening = " + hamiltonian["screening"]["method"]),
        ("H off-diagonal magnitude", off_diagonal, "Eh",
         "the CI roots ARE the spin-orbit eigenstates", out.SCI_FMT),
        ("moment matrices read back element for element", "exact" if round_trip
         else "MISMATCH"),
        ("inactive contribution to L", float(np.abs(dump.matrices.inactive_l).max()),
         "hbar", "exactly zero for a Kramers-paired inactive set", out.SCI_FMT),
    ])

    # ----------------------------------------------------------------------------------
    # 3. The physics, through the phase-invariant reduction and nothing else.
    # ----------------------------------------------------------------------------------
    out.subsection(log, "Spin-orbit multiplets")
    doublets = dump.matrices.analyse()
    table = out.Table(log, [
        out.col_count("doublet", 9),
        out.col_count("states", 8),
        out.Column("rel [cm^-1]", out.CM_FMT, 14),
        out.Column("spread [cm^-1]", out.SCI_FMT, 16),
        out.Column("g principal values", "{}", 28, align="<"),
    ])
    table.start()
    for k, block in enumerate(doublets):
        table.row(k, block.size, block.energy_cm, block.spread_cm,
                  "  ".join("{:.4f}".format(g) for g in block.g_values))
    table.end("D3h splits the d shell into a1' + e'' + e': five Kramers doublets")

    ground = doublets[0]
    anisotropy = max(ground.g_values) - min(ground.g_values)
    out.entry(log, "g anisotropy of the ground doublet", anisotropy, "",
              "a D3h ligand field defines an axis, so this must be nonzero", "{:.4f}")

    # ----------------------------------------------------------------------------------
    # 4. The pseudospin export.
    # ----------------------------------------------------------------------------------
    # `sites` partitions the ACTIVE SPINORS into the local multiplet sites, by position in
    # the active list. TiCl3 has one magnetic centre, so there is one site and it holds the
    # whole 3d shell; a polynuclear complex would list one group per centre. Every site must
    # sit in a definite particle-number sector -- a pseudospin |S, M> labels a multiplet, and
    # a charge-mixed space is not one -- which is why a single delocalized electron cannot
    # be split across two sites.
    #
    # rule="dimension" with dims=2 asks for "the lowest two-dimensional multiplet on the
    # site", i.e. the ground Kramers doublet as an effective spin 1/2. rule="gap" would
    # instead cut wherever the local spectrum leaves a gap.
    pseudospin_path = OUTPUT / "ticl3.psd"
    export = kuiva.PseudospinExport(
        cas, pseudospin_path, sites=[tuple(range(N_ACTIVE))], rule="dimension", dims=2,
        title="TiCl3 ground Kramers doublet as an effective spin 1/2",
        comments=["planar D3h, Ti-Cl = {:.2f} A".format(R_TICL)]).run()

    out.section(log, "The pseudospin file")
    log.info("%s", export.summary())

    back = read_pseudospin(pseudospin_path)
    (site_g,) = export.g_values
    out.entries(log, [
        ("file", str(pseudospin_path.relative_to(HERE))),
        ("size", pseudospin_path.stat().st_size / 1024.0, "kB", "", "{:.1f}"),
        ("model space", " x ".join(str(d) for d in export.model.dims), "",
         "one site: an effective spin 1/2"),
        ("product basis (2M labels)", str([tuple(row) for row in back["basis"]])),
        ("Hamiltonian provenance carried over",
         "hamiltonian" in export.model.provenance),
        ("g values from the model space",
         "  ".join("{:.4f}".format(g) for g in site_g)),
        ("g values from the property dump",
         "  ".join("{:.4f}".format(g) for g in ground.g_values)),
        ("largest difference between the two routes",
         max(abs(a - b) for a, b in zip(sorted(site_g), sorted(ground.g_values))), "",
         "", out.SCI_FMT),
    ])
    out.note(log, "the two routes share the converged orbitals and the states, and nothing")
    out.note(log, "else: one contracts CI transition densities, the other contracts a")
    out.note(log, "tensor network onto the model space. Agreement at this level is a")
    out.note(log, "statement about both of them.")

    # ----------------------------------------------------------------------------------
    # 5. Assert. Structure and invariants only -- never a matrix element, which the file
    #    format leaves undefined up to a phase.
    # ----------------------------------------------------------------------------------
    checks = {
        "the CASSCF converged": bool(cas.converged),
        "the state average uses the whole determinant space": cas.boundary.gap_cm is None,
        "the dump analyses into five Kramers doublets":
            [block.size for block in doublets] == [2, 2, 2, 2, 2],
        "H in the dump is diagonal": off_diagonal == 0.0,
        "the dump reads back element for element": bool(round_trip),
        "the dump header carries the Hamiltonian provenance":
            "screening" in hamiltonian,
        "the moment matrices are Hermitian":
            dump.matrices.hermiticity_error() < 1e-10,
        "the Kramers-paired inactive space carries no moment":
            float(np.abs(dump.matrices.inactive_l).max()) < 1e-8,
        "the ground doublet is anisotropic, as D3h requires": anisotropy > 0.1,
        "the pseudospin file is one effective spin 1/2": tuple(export.model.dims) == (2,),
        "the pseudospin basis is OuluSpin's 2M ordering":
            [tuple(row) for row in back["basis"]] == [(-1,), (1,)],
        "the two property routes agree to {:.0e}".format(G_AGREEMENT):
            max(abs(a - b) for a, b in zip(sorted(site_g),
                                           sorted(ground.g_values))) < G_AGREEMENT,
    }
    failures = report(checks)

    timing.summary(log)
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
