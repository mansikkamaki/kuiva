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

    PropertyDump        the effective Hamiltonian H, the three magnetic-moment components
                        mu_x, mu_y, mu_z and the three electric-dipole components
                        d_x, d_y, d_z, in the basis of the spin-orbit eigenstates, as plain
                        self-describing text with a versioned header. For an ITO /
                        crystal-field code.

    PseudospinExport    the same physics contracted onto a local-multiplet model space:
                        H_eff, the moment operators, the ordered pseudospin product basis
                        and the unitary that maps the ab initio states onto it. For the
                        external OuluSpin code, in its conventions and its storage order.

Both of them also carry the HYPERFINE FIELD operator of every nucleus named at ingestion
(here 47Ti), under the same block names and beside the same [NUCLEI] table. What is written
is the isotope-independent operator T, in Hartree per nuclear magneton, so that

    H_hf = sum_k g_N(k) sum_u T_<k>_u (x) I_<k>_u        [Eh]

on the product of the electronic space with the nuclear spins. Kuiva never forms that product
space -- it writes the electronic matrices and a nuclear table separately, which is exactly
what lets a consumer change the isotope, or drop a nucleus, without re-running any of this.
There is no A tensor and no hyperfine spin Hamiltonian anywhere in these files: those are the
external code's, as the crystal field is.

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

* **No picture-change correction is applied to L, S or r.** They are the bare
  non-relativistic AO operators, used unchanged in the two-component basis. This matches
  what OpenMolcas RASSI does, which keeps a cross-code comparison like for like, and the
  size of the approximation has been measured and is small. Every run emits that warning
  and records it in the file; it is not configurable, because the file outlives the session.
  (property_picture_change=True on the reference turns it on for both the magnetic and the
  electric operator -- never for only one of them.)

* **Phases are arbitrary and are never canonicalized.** Degenerate spin-orbit states mix
  freely, so an element-by-element comparison of a moment matrix -- against another program,
  or against this one run twice -- is meaningless. What can be compared are degeneracy
  patterns, relative energies and the invariant Tr_block(mu_i mu_j), whose principal values
  are the g factors. The example uses that reduction and nothing else.

* **The hyperfine operators are the exception to the first of those, and it is recorded.**
  They ALWAYS carry the picture change, whatever the flag above says, because a bare
  hyperfine operator is wrong by a factor of four to ten wherever s character carries spin
  density. What replaces the "one flag governs both" guarantee is that each file states the
  treatment of each operator family separately, in its own header fields.

WHAT THIS EXAMPLE CANNOT SHOW, AND SAYS SO
------------------------------------------
The isotropic (contact) part of the hyperfine coupling comes from spin polarization of the
core s shells, and a VALENCE active space -- ten 3d spinors here -- carries none of it. For a
4f ion that is a minor error, the orbital mechanism dominating; for a 3d ion like this one it
is qualitatively wrong, by twenty-five per cent and worse. So the |A| values below are a
demonstration of the machinery and of the two routes agreeing with each other, and NOT a
prediction of an experiment. Every run says this in a warning and both files record the
active space so a reader can judge. The remedy is core s shells in the active space, which
needs a tensor-network active space, not a correction term.

WHAT TO LOOK FOR IN THE OUTPUT
------------------------------
* the ten states resolving into five Kramers doublets, each with its principal g values;
* the permanent electric dipole coming out at ZERO in all ten states, as planar D3h requires,
  and the d-d line strengths out of the ground doublet coming out small but NOT zero -- the
  Laporte rule is exact only in a centrosymmetric field, and D3h has no inversion centre;
* the inactive contribution to L coming out at zero -- exact for a Kramers-paired inactive
  set, and computed rather than assumed, because a nonzero value would be a statement about
  the orbitals;
* the ground doublet's g tensor being *anisotropic*: a D3h ligand field defines an axis, so
  it must be. (A free ion is the opposite check -- example 2 -- and having both is what
  shows the machinery responds to real anisotropy instead of returning a plausible
  constant.)
* the same g values coming out of the pseudospin export, which reaches them by contracting
  the network onto a model space rather than through CI transition densities: two
  independent routes to one invariant -- and the same again for the hyperfine field, whose
  |A| values must agree between the routes far more tightly than they mean anything;
* the [NUCLEI] table carrying a NEGATIVE g_N for 47Ti and 'none' for its quadrupole moment.
  Both are deliberate: |A| is quadratic and cannot hold that sign, which is why the mixed
  invariant Tr_block(mu.T) is printed beside it, and a quadrupole moment whose SIGN is not
  established is written as absent rather than as a magnitude, because that sign is the sign
  of every quadrupole splitting a consumer would compute from it.
"""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import List

import numpy as np

import kuiva
from kuiva.props.dump import DEFAULT_INACTIVE_TOL, read_dump
from kuiva.props.multiplet import block_collinearity, multiplet_hyperfine_values
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

#: The same claim for the hyperfine field, as a RELATIVE band because |A| is a coupling in
#: MHz rather than a dimensionless factor. Nothing here is an accuracy claim: see the
#: core-polarization warning in the header comment.
A_AGREEMENT = 1.0e-6

#: The nucleus whose hyperfine field both files carry. 47Ti has I = 5/2, a negative g_N, and
#: no tabulated signed quadrupole moment -- all three of which the [NUCLEI] table has to say.
HYPERFINE = {"Ti": 47}


def planar_mx3(metal: str, ligand: str, r: float) -> List[tuple]:
    """Planar D3h MX3, metal at the origin, ligands in the xy plane."""
    atoms = [(metal, (0.0, 0.0, 0.0))]
    for k in range(3):
        theta = 2.0 * math.pi * k / 3.0
        atoms.append((ligand, (r * math.cos(theta), r * math.sin(theta), 0.0)))
    return atoms


def energy_sorted(matrices, operator) -> np.ndarray:
    """An operator re-indexed into the energy-ordered basis the multiplets are indexed in.

    The blocks `analyse()` returns are slices of the *sorted* spectrum, and the matrices
    arrive in the solver's order; a block invariant taken without this line is a trace over
    whichever states happened to sit at those indices.
    """
    order = np.argsort(np.asarray(matrices.energies, dtype=float))
    return np.asarray(operator)[:, order, :][:, :, order]


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
    # hyperfine=: naming the nuclei HERE is the whole request. Neither formatted product has
    # a switch of its own -- a file that had computed the coupling and then not written it is
    # the one thing a reader cannot recover from.
    scf = kuiva.ScalarSCF(molecule, memory_gb=6.0, hyperfine=HYPERFINE).run()
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

    # The other half of the hyperfine contract, and it is the half that is not a matrix: the
    # operators name no nucleus and state no I, and the table states no coupling. A consumer
    # that parses no JSON at all still gets this, because it is a plain table and not part of
    # the provenance block.
    nucleus = stored["nuclei"][0]
    out.subsection(log, "The nuclear table")
    out.entries(log, [
        ("nuclei", len(stored["nuclei"]), "",
         ", ".join("{} ({})".format(n["atom_label"], n["label"])
                   for n in stored["nuclei"])),
        ("spin I", 0.5 * nucleus["twice_spin"], "hbar",
         "a pseudospin site of dimension 2I+1 = {} for the consumer".format(
             nucleus["twice_spin"] + 1), "{:.1f}"),
        ("g_N", nucleus["g"], "",
         "NEGATIVE for 47Ti; |A| is quadratic and cannot carry that sign", "{:+.6f}"),
        ("quadrupole moment", "none (no signed value tabulated)", "",
         "refused rather than replaced by a magnitude: the sign of Q is the sign of "
         "every quadrupole splitting"),
        ("nuclear data source", nucleus["source"]),
        ("hyperfine operator unit", stored["header"]["hyperfine_unit"], "",
         stored["header"]["hyperfine_operator"]),
        ("picture change on the hyperfine operators",
         stored["header"]["hyperfine_picture_change"], "",
         "independently of what the header says about mu and d"),
        # ⚠ Printed against max|T|, not on its own: T is of order 1e-06 Eh/mu_N here, so an
        # absolute residual of 1e-16 says nothing until it is read relative to the operator.
        ("inactive contribution to T", float(np.abs(
            dump.matrices.hyperfine_inactive[nucleus["atom_label"]]).max())
         / float(np.abs(dump.matrices.hyperfine[nucleus["atom_label"]]).max()),
         "relative to max|T|",
         "zero for a Kramers-paired core: T is time odd, like L and S", out.SCI_FMT),
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

    # The hyperfine field goes through exactly the same discipline, and the stage's own report
    # above already printed it per doublet. What is tabulated here is the ground doublet alone,
    # because that is the block the pseudospin export will model and therefore the only place
    # the two routes can be compared at all.
    #
    # |A| is a REDUCTION of the stored operator, not a fitted tensor: the principal values of
    # 3 g_N^2 Tr_b(T_i T_j) / [J(J+1)(2J+1)] at the isotope the table names -- the A.A^T
    # construction of the published implementations, in the same sense the principal g values
    # are a reduction. A.g is the MIXED invariant Tr_b(mu.T) normalized, and it is the only
    # quantity here that carries the relative sign and orientation of A against g: |A| is
    # quadratic and throws both away, while a comparison of matrix elements is forbidden by
    # the arbitrary phases.
    label = nucleus["atom_label"]
    t_sorted = energy_sorted(dump.matrices, dump.matrices.hyperfine[label])
    mu_sorted = energy_sorted(dump.matrices, dump.matrices.mu)
    a_dump = multiplet_hyperfine_values(ground.hyperfine[label], ground.size,
                                        float(nucleus["g"]))
    collinearity_dump = block_collinearity(mu_sorted, t_sorted, ground.start, ground.size)
    out.entries(log, [
        ("|A| of the ground doublet", "  ".join("{:.2f}".format(a) for a in a_dump), "MHz",
         "for {}; the stored operator is isotope independent".format(nucleus["label"])),
        ("A.g on the ground doublet", collinearity_dump, "",
         "+-1 would mean T is proportional to mu there; a ligand field need not make it so",
         "{:+.4f}"),
    ])

    # ----------------------------------------------------------------------------------
    # 3b. The electric dipole, which is in the same file and is read the same way.
    # ----------------------------------------------------------------------------------
    # d is the TOTAL dipole: the electronic operator over all electrons, plus -- on the
    # DIAGONAL only -- the nuclear sum_A Z_A (R_A - R_G). So a diagonal element is that
    # state's dipole moment and an off-diagonal one is a transition dipole.
    #
    # Planar D3h has no dipole, so every diagonal element here must be zero. Note what that
    # does and does not pin: with the gauge origin at the centre of mass -- which for this
    # molecule is the Ti nucleus -- the nuclear and inactive terms are separately zero by
    # the same symmetry, so what this checks is the ACTIVE term's sign and completeness. The
    # three-term cancellation itself is exercised on a polar molecule in the test suite,
    # where a wrong sign or a missing term cannot hide behind a symmetry that zeroes it.
    permanent = np.array([np.real(np.diag(dump.matrices.d[k])) for k in range(3)])
    largest_permanent = float(np.abs(permanent).max())
    inactive_dipole = float(np.abs(dump.matrices.inactive_d).max())
    nuclear_dipole = float(np.abs(dump.matrices.nuclear_dipole).max())

    # Transitions are compared through the line strength and nothing else: within a Kramers
    # doublet the two states mix arbitrarily, so an individual |d_IJ| is a phase, while the
    # double sum over two whole doublets is invariant. d-d transitions are Laporte forbidden
    # in a centrosymmetric field; D3h has no inversion centre and the "3d" orbitals carry
    # ligand character, so these are small but not zero.
    strengths = dump.matrices.line_strengths(multiplets=doublets)

    out.subsection(log, "Electric dipole")
    out.entries(log, [
        ("molecular charge", dump.matrices.molecular_charge, "e",
         "neutral, so every dipole here is independent of the gauge origin"),
        ("nuclear term", nuclear_dipole, "e*a0",
         "diagonal only; zero here because D3h puts the origin on Ti", out.SCI_FMT),
        ("inactive electrons' term", inactive_dipole, "e*a0",
         "generally NOT zero -- r is time even, unlike L and S -- but zero by symmetry here",
         out.SCI_FMT),
        ("largest permanent moment over all ten states", largest_permanent, "e*a0",
         "planar D3h has no dipole, so the three terms must cancel", out.SCI_FMT),
        ("strongest d-d line strength out of the ground doublet",
         float(strengths[0, 1:].max()), "(e*a0)^2",
         "Laporte forbidden in a centrosymmetric field; D3h is not one", out.SCI_FMT),
    ])
    out.note(log, "a line strength is NOT an oscillator strength and NOT a rate: this")
    out.note(log, "program writes operators and their invariants, and turning one into an")
    out.note(log, "intensity is the external property code's job, as the crystal field is.")

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

    # ...and the same claim for the hyperfine field, which is the sharper half of it: T took
    # the network route here and the transition-density route there, and the only comparable
    # quantities are the invariants. The [NUCLEI] table is written by the same code into both
    # files, so it is the matrices that are being compared and not two different nuclei.
    (site_block,) = export.model.analyse()
    a_export = multiplet_hyperfine_values(site_block.hyperfine[label], site_block.size,
                                          float(nucleus["g"]))
    collinearity_export = block_collinearity(
        export.model.mu_in_eigenbasis(), export.model.hyperfine_in_eigenbasis()[label], 0, 2)
    a_difference = max(abs(a - b) for a, b in zip(sorted(a_export), sorted(a_dump)))
    out.subsection(log, "The hyperfine field, on both routes")
    out.entries(log, [
        ("nucleus", "{} ({})".format(label, nucleus["label"]), "",
         "2I+1 = {} nuclear states; the electron-nuclear product space is {}".format(
             nucleus["twice_spin"] + 1, export.model.product_dim())),
        ("|A| from the model space", "  ".join("{:.2f}".format(a) for a in a_export), "MHz"),
        ("|A| from the property dump", "  ".join("{:.2f}".format(a) for a in a_dump), "MHz"),
        ("largest difference between the two routes", a_difference, "MHz",
         "{:.1e} relative".format(a_difference / max(a_dump)), out.SCI_FMT),
        ("A.g, model space / property dump",
         "{:+.6f} / {:+.6f}".format(collinearity_export, collinearity_dump)),
    ])
    out.note(log, "these |A| are NOT a prediction: a valence 3d active space carries no core-s")
    out.note(log, "spin polarization, so the isotropic part of a transition-metal hyperfine")
    out.note(log, "coupling is qualitatively wrong. What the agreement tests is the machinery.")
    out.note(log, "Kuiva never builds the electron-nuclear product space named above -- it")
    out.note(log, "writes the electronic matrices and the nuclear table, and the consumer")
    out.note(log, "chooses the isotope. The dimension is reported for that reason only.")

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
        "the dipole matrices are Hermitian":
            float(np.abs(dump.matrices.d - dump.matrices.d.conj().transpose(0, 2, 1)).max())
            < 1e-12,
        "planar D3h has no permanent dipole, in any of the ten states":
            largest_permanent < 1e-6,
        "d-d transitions are weak but not zero: D3h has no inversion":
            0.0 < float(strengths[0, 1:].max()) < 1.0,
        "the ground doublet is anisotropic, as D3h requires": anisotropy > 0.1,
        "the pseudospin file is one effective spin 1/2": tuple(export.model.dims) == (2,),
        "the pseudospin basis is OuluSpin's 2M ordering":
            [tuple(row) for row in back["basis"]] == [(-1,), (1,)],
        "the two property routes agree to {:.0e}".format(G_AGREEMENT):
            max(abs(a - b) for a, b in zip(sorted(site_g),
                                           sorted(ground.g_values))) < G_AGREEMENT,
        "both files carry the hyperfine operators and the nuclear table":
            dump.matrices.has_hyperfine and export.model.has_hyperfine
            and len(stored["nuclei"]) == len(back["nuclei"]) == 1,
        "the nuclear table survives the round trip through both files":
            back["nuclei"][0]["label"] == nucleus["label"] == "47Ti"
            and back["nuclei"][0]["quadrupole_barn"] is None,
        # ⚠ A RELATIVE band, at the library's own inactive tolerance. T is of order 1e-06
        # Eh/mu_N here, four orders below the absolute tolerance that guards L and S, so an
        # absolute reading of it would be a check that cannot fail on exactly the quantity it
        # exists to protect.
        "the Kramers-paired inactive space carries no hyperfine field either":
            float(np.abs(dump.matrices.hyperfine_inactive[label]).max())
            < DEFAULT_INACTIVE_TOL * float(np.abs(dump.matrices.hyperfine[label]).max()),
        "the hyperfine field is a real coupling, not a rounding artefact":
            min(a_dump) > 1.0,
        "the two routes agree on |A| to {:.0e} relative".format(A_AGREEMENT):
            a_difference < A_AGREEMENT * max(a_dump),
        "...and on the mixed invariant that carries its sign against g":
            abs(collinearity_export - collinearity_dump) < 1e-6,
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
