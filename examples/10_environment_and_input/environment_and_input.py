"""Example 10 -- saying what surrounds the molecule, and what basis it is in.

    source setup.sh          # once per shell
    python environment_and_input.py

Runs in a few seconds on a neon atom and writes ``output/environment_and_input.out``.

WHAT THIS SHOWS
---------------
Three ways of stating an input that the first nine examples never needed, because they all
compute an isolated molecule in a basis this program already knows:

    Environment   a field of classical point charges around the molecule. A single-molecule
                  magnet is measured in a crystal, and a bare gas-phase ion has a
                  qualitatively different ligand field from the one in the lattice.

    ghost atoms   basis functions with no nucleus, no electrons and no mass. The same
                  molecule, in a bigger basis, with the same number of electrons -- which is
                  what a counterpoise correction is made of.

    CustomBasis   a basis set the registry does not have, supplied as an NWChem string or as
                  parsed shells, and still checked for everything that *can* be checked.

WHAT TO LOOK FOR IN THE OUTPUT
------------------------------
* the environment block in the Hamiltonian section: how many charges, the net charge, how
  close the nearest one comes to a nucleus, and whether the potential was picture-changed;
* the charge-nucleus interaction on its **own** line, deliberately not folded into the
  nuclear repulsion -- an embedded total stays separable into the part that is chemistry and
  the part that is the field it sits in;
* what the field does to the symmetry: a single charge on the z axis leaves C2(z) and
  destroys inversion, and the labelling says so rather than using a group the calculation is
  not in;
* the basis-set superposition error, as one number, from a pair of runs differing only in
  whether the second molecule's functions are present;
* the custom basis reproducing the registered family it was copied from to machine precision,
  with its contraction type **measured** from the shells rather than taken on trust.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

import kuiva
from kuiva.util import output as out
from kuiva.util import resources as res
from kuiva.util import timing
from kuiva.util.logging import add_file_handler, get_logger

NAME = "environment_and_input"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
BASIS = "x2c-SVPall-2c"

log = get_logger("examples." + NAME)


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
    out.banner(log, kuiva.__version__, "example 10: environment, ghosts and custom bases")
    out.entry(log, "output file", str(outfile.relative_to(HERE)))

    # ----------------------------------------------------------------------------------
    # 1. The molecule on its own, as every other example computes one.
    # ----------------------------------------------------------------------------------
    # screening="none" throughout: none of what this example shows is about spin-orbit
    # coupling, and the two-electron picture change would cost a four-component atomic solve
    # per element to change no number here.
    vacuum = kuiva.ScalarSCF(
        kuiva.Molecule(atoms=[("Ne", (0.0, 0.0, 0.0))], basis=BASIS, unit="Bohr"),
        memory_gb=2.0, screening="none").run()

    # ----------------------------------------------------------------------------------
    # 2. The same molecule in a field of point charges.
    # ----------------------------------------------------------------------------------
    # An Environment states what surrounds the molecule. Its coordinates are in the
    # molecule's own unit unless it says otherwise -- a charge field copied out of a
    # crystallographic file is in Angstrom, and reading it as bohr puts the lattice 1.9x too
    # far away and produces a perfectly plausible ligand field for the wrong crystal.
    #
    # What an embedding changes is exactly two things: the one-electron Hamiltonian gains the
    # charges' potential, and the classical charge-nucleus energy joins the total. Nothing
    # after the front end learns that the charges exist, so every invariant downstream means
    # what it meant before.
    field = kuiva.Environment(
        point_charges=[(+0.7, (0.0, 0.0, 5.0)), (-0.3, (0.0, 2.0, -4.0))],
        label="two charges, illustrative")
    embedded = kuiva.ScalarSCF(
        kuiva.Molecule(atoms=[("Ne", (0.0, 0.0, 0.0))], basis=BASIS, unit="Bohr",
                       environment=field),
        memory_gb=2.0, screening="none").run()

    out.section(log, "A molecule in a field of point charges")
    out.entries(log, [
        ("vacuum SCF energy", vacuum.energy, "Eh", "", out.E_FMT),
        ("embedded SCF energy", embedded.energy, "Eh", "", out.E_FMT),
        ("difference", embedded.energy - vacuum.energy, "Eh",
         "the field's effect on the electrons and the nuclei together", out.SCI_FMT),
        ("charge-nucleus interaction", embedded.data.e_embedding, "Eh",
         "reported separately, never folded into the nuclear repulsion", out.E_FMT),
        ("field digest", embedded.data.embedding.digest[:16], "",
         "identifies the field in every stored product's header"),
    ])
    out.note(log, "The charges' interaction with each other is a constant of the lattice and")
    out.note(log, "not of this calculation: it is neither computed nor reported here.")

    # The field also decides what symmetry the calculation actually has. A single charge on
    # the z axis leaves C2(z) and destroys inversion; labelling the states from the nuclei
    # alone would give them irreps of a group this calculation is not in.
    lopsided = kuiva.ScalarSCF(
        kuiva.Molecule(atoms=[("Ne", (0.0, 0.0, 0.0))], basis=BASIS, unit="Bohr",
                       point_group="auto",
                       environment=kuiva.Environment(
                           point_charges=[(0.4, (0.0, 0.0, 5.0))])),
        memory_gb=2.0, screening="none").run()
    symmetric = kuiva.ScalarSCF(
        kuiva.Molecule(atoms=[("Ne", (0.0, 0.0, 0.0))], basis=BASIS, unit="Bohr",
                       point_group="auto",
                       environment=kuiva.Environment(
                           point_charges=[(0.4, (0.0, 0.0, 5.0)),
                                          (0.4, (0.0, 0.0, -5.0))])),
        memory_gb=2.0, screening="none").run()

    out.subsection(log, "What the field does to the symmetry")
    out.entries(log, [
        ("one charge on +z", ", ".join(lopsided.data.symmetry.detected)),
        ("two charges, +z and -z", ", ".join(symmetric.data.symmetry.detected)),
    ])
    out.note(log, "The nuclei alone have all three operations in both runs. The molecule is")
    out.note(log, "not the whole system, and the labels say what the whole system has.")

    # ----------------------------------------------------------------------------------
    # 3. Ghost atoms: the same electrons, in a bigger basis.
    # ----------------------------------------------------------------------------------
    # ("ghost-Ar", pos) is argon's basis functions with no nucleus. The label is the address:
    # basis={"ghost-Ar": ...} reaches it and basis={"Ar": ...} does not, because a ghost and
    # a real atom of one element are different things to everything downstream.
    ghosted = kuiva.ScalarSCF(
        kuiva.Molecule(atoms=[("Ne", (0.0, 0.0, 0.0)), ("ghost-Ar", (0.0, 0.0, 4.0))],
                       basis={"Ne": BASIS, "ghost-Ar": BASIS}, unit="Bohr"),
        memory_gb=2.0, screening="none", atomic_reference=True).run()
    bsse = ghosted.energy - vacuum.energy

    out.section(log, "Ghost atoms and the superposition error they measure")
    out.entries(log, [
        ("neon alone, AO functions", vacuum.data.nao),
        ("neon + ghost argon, AO functions", ghosted.data.nao),
        ("electrons, both runs", "{}, {}".format(*ghosted.data.nelec), "",
         "a ghost brings none"),
        ("nuclear repulsion, both runs", ghosted.data.e_nuc, "Eh", "a ghost has no nucleus",
         out.E_FMT),
        ("basis-set superposition error", bsse, "Eh",
         "negative by construction: more functions, same electrons", out.SCI_FMT),
    ])

    # The charge partition resolves it per centre: the ghost's "charge" is minus the density
    # that leaked onto functions carrying no electrons of their own.
    from kuiva.props.population import atomic_reference_charges
    charges = atomic_reference_charges(
        ghosted.data.mo_coeff, ghosted.data.s_ao, ghosted.data.ao_layout,
        reference=ghosted.data.atomic_reference, occupation=ghosted.data.mo_occ)
    out.entries(log, [
        ("density on the ghost's functions", -charges.charge[1], "e", "", out.SCI_FMT),
    ])

    # ----------------------------------------------------------------------------------
    # 4. A basis the registry does not have.
    # ----------------------------------------------------------------------------------
    # The escape hatch keeps every check that can still be made. What it cannot measure is
    # the relativistic treatment -- nothing in a list of exponents says whether a set was
    # recontracted for X2C -- so that one is required, and a set that declares itself
    # non-relativistic is still refused alongside a relativistic one.
    from kuiva.interface.pyscf_bridge import build_mole
    registered = build_mole(kuiva.Molecule(atoms=[("Ne", (0.0, 0.0, 0.0))], basis=BASIS))
    shells = registered._basis[registered.atom_symbol(0)]
    custom = kuiva.CustomBasis(data={"Ne": shells}, relativistic_treatment="x2c-2c",
                               name="svpall-copy", notes="the registered set, by hand")
    copied = kuiva.ScalarSCF(
        kuiva.Molecule(atoms=[("Ne", (0.0, 0.0, 0.0))], basis=custom, unit="Bohr"),
        memory_gb=2.0, screening="none").run()

    refused = ""
    try:
        kuiva.CustomBasis(data={"Ne": shells})
    except ValueError as exc:
        refused = str(exc).split(".")[0].strip()

    out.section(log, "A user-supplied basis set")
    out.entries(log, [
        ("registry family", BASIS),
        ("the same shells, supplied by hand", custom.label("Ne"), "",
         "contraction measured from the data"),
        ("registered SCF energy", vacuum.energy, "Eh", "", out.E_FMT),
        ("custom SCF energy", copied.energy, "Eh", "", out.E_FMT),
        ("difference", abs(copied.energy - vacuum.energy), "Eh", "", out.SCI_FMT),
        ("without relativistic_treatment", "refused"),
        ("message", refused),
    ])

    # ----------------------------------------------------------------------------------
    # 5. Assert. An example that only prints numbers cannot fail, and one that cannot fail
    #    demonstrates nothing.
    # ----------------------------------------------------------------------------------
    checks = {
        "the field changes the energy": abs(embedded.energy - vacuum.energy) > 1e-6,
        "the charge-nucleus term is reported separately":
            embedded.data.e_embedding != 0.0 and abs(embedded.data.e_nuc) < 1e-12,
        "the embedding is recorded in the Hamiltonian's provenance":
            embedded.data.soc.provenance()["embedding"]["embedded"] is True,
        "a vacuum run records no environment":
            vacuum.data.soc.provenance()["embedding"]["embedded"] is False,
        "a lopsided field removes inversion from the labelling":
            tuple(lopsided.data.symmetry.detected) == ("C2(z)",),
        "a symmetric field keeps all three operations":
            len(symmetric.data.symmetry.detected) == 3,
        "a ghost adds functions": ghosted.data.nao > vacuum.data.nao,
        "a ghost adds no electrons": ghosted.data.nelec == vacuum.data.nelec,
        # Not `== 0.0`: the integral library multiplies its way to a denormal (~1e-199)
        # rather than to a hard zero when one charge of a pair is zero.
        "a ghost adds no nuclear repulsion": abs(ghosted.data.e_nuc) < 1e-12,
        "the superposition error is negative and small": -1e-2 < bsse < 0.0,
        "the leaked density is attributed to the ghost": charges.charge[1] < 0.0,
        "a custom basis reproduces the family it copies":
            abs(copied.energy - vacuum.energy) < 1e-10,
        "the contraction type is measured": "segmented" in custom.label("Ne"),
        "an undeclared relativistic treatment is refused": bool(refused),
    }
    failures = report(checks)

    timing.summary(log)
    res.summary(log)
    return 1 if failures else 0


def report(checks) -> int:
    """One table of pass/fail, and the count that decides the exit status."""
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
