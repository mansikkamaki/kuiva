"""Example 3 -- a state-averaged two-component CASSCF on a real molecule.

    source setup.sh          # once per shell
    python casscf_ligand_field.py

Runs in two to three minutes on TiCl3 and writes ``output/casscf_ligand_field.out``.

WHAT THIS SHOWS
---------------
This is the calculation the program exists to do. Planar TiCl3 is Ti(III), one 3d electron
in a ligand field -- the smallest honest version of the systems Kuiva is aimed at, and
cheap enough to run while you read the file.

    ScalarSCF -> Reference -> CASSCF -> the spin-orbit spectrum

Three things are worth watching, and each of them is a decision you will have to make in
any real calculation:

1. **The active space is stated as orbital character, not as indices.** "The five lowest
   Kramers pairs of d character on the titanium" is a definition another program can
   reproduce; "spinors 68 to 77" is not, and it silently follows the basis set around.
   Kuiva refuses a principal quantum number here on purpose: PySCF's principal-quantum
   labels count shells *within the basis*, which is not the same thing.

2. **Which states the orbitals are optimized for.** The CASSCF here averages over the
   ground Kramers doublet only -- that is the state whose orbitals are wanted -- and the
   full spectrum is then a CASCI at those fixed orbitals. Averaging over all ten from the
   start would optimize the orbitals for an average nobody asked about.

3. **Where the state average stops.** A state-averaged CASSCF is exactly as symmetric as
   the set it averages over. A count that stops *inside* a near-degenerate manifold makes
   the averaged density non-invariant; the Fock operator built from it splits the shell,
   the orbitals optimize on the broken density, and the result is entirely plausible and
   wrong. The only evidence is a root the average does *not* use, so Kuiva solves a few
   extra, discards them, and reports the gap -- at the starting orbitals as well as the
   converged ones, because the starting one is what says whether the trajectory was safe.

THE ANSWER YOU SHOULD SEE
-------------------------
D3h splits the d shell into a1' + e'' + e', so the ten spinors of the 3d shell give **five
Kramers doublets**:

    0, 0 | 15442, 15442 | 15521, 15521 | 23882, 23882 | 23958, 23958   cm^-1

Five pairs, each exactly degenerate. That pattern is fixed by symmetry and by Kramers'
theorem, not by any parameter of this program. The absolute splittings are a small-basis
CASSCF without dynamic correlation, so treat them as the ligand-field pattern rather than
as spectroscopy; example 5 is where the correlation correction comes in.

A note on the size of the Kramers splitting. Kuiva's general two-component CI realizes
Kramers degeneracy numerically rather than by construction, and 1e-8 to 1e-6 Eh is the band
to expect once the degeneracy has to emerge from a many-electron CI. It does not have to
here: a one-electron active space makes the CI matrix the one-particle matrix, whose
time-reversal symmetry is exact, so the splitting comes out at the bottom of that band or
below it. The example asserts the pairs are degenerate and reports what was measured,
rather than demanding a number it has no right to expect.
"""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import List

import numpy as np

import kuiva
from kuiva.interface.api import casci
from kuiva.props.multiplet import HARTREE_TO_CM
from kuiva.props.population import orbital_populations
from kuiva.util import output as out
from kuiva.util import resources as res
from kuiva.util import timing
from kuiva.util.logging import add_file_handler, get_logger

NAME = "casscf_ligand_field"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

log = get_logger("examples." + NAME)

#: Ti-Cl bond length [Angstrom] of the planar D3h gas-phase molecule.
R_TICL = 2.25

#: The 3d shell: five Kramers pairs on the titanium, holding the single d electron.
N_ACTIVE, N_ACTIVE_ELEC = 10, 1

#: The CASSCF averages over the ground Kramers doublet; the spectrum is the ten roots of
#: the same active space at those orbitals.
N_STATES_SCF, N_STATES_CI = 2, 10

#: Budgets, explicit so that termination never depends on convergence.
MAX_ITER, CONV_GRAD = 40, 1.0e-4

#: The band reserved for numerical Kramers splitting in the general two-component CI. See
#: the module docstring on why this is an upper bound rather than an expected value.
KRAMERS_SPLIT_MAX = 1.0e-6

#: The state average must stop far from a near-degeneracy. Below 50 cm^-1 Kuiva warns; here
#: the next root is four orders of magnitude away, so the requirement is stated as such.
BOUNDARY_MIN_CM = 1.0e4


def planar_mx3(metal: str, ligand: str, r: float) -> List[tuple]:
    """Planar D3h MX3, metal at the origin, ligands in the xy plane."""
    atoms = [(metal, (0.0, 0.0, 0.0))]
    for k in range(3):
        theta = 2.0 * math.pi * k / 3.0
        atoms.append((ligand, (r * math.cos(theta), r * math.sin(theta), 0.0)))
    return atoms


def prepare_output() -> Path:
    """Write this run to output/<name>.out, with a scratch spin-orbit cache of its own.

    The two-electron spin-orbit correction below is one four-component atomic calculation
    per element -- about a minute for titanium in this basis, seconds for chlorine -- and it
    is cached on disk between jobs, keyed on the element, basis and configuration rather
    than on the geometry. An example must neither write into the cache you rely on nor let
    its own timings depend on what happens to be in it.
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT / "amf-cache"
    shutil.rmtree(cache, ignore_errors=True)
    os.environ["KUIVA_AMF_CACHE"] = str(cache)
    path = OUTPUT / (NAME + ".out")
    add_file_handler(path)
    return path


def spectrum_table(title: str, energies: np.ndarray) -> List[float]:
    """Log the spectrum in Kramers pairs and return the pair splittings [Eh]."""
    out.subsection(log, title)
    table = out.Table(log, [
        out.col_count("state", 7),
        out.col_energy("E [Eh]"),
        out.Column("rel [cm^-1]", out.CM_FMT, 14),
        out.Column("pair split [Eh]", out.SCI_FMT, 17),
    ])
    table.start()
    splittings: List[float] = []
    for i, energy in enumerate(energies):
        split = ""
        if i % 2 == 1:
            split = float(energies[i] - energies[i - 1])
            splittings.append(split)
        table.row(i, energy, (energies[i] - energies[0]) * HARTREE_TO_CM, split)
    table.end("{} Kramers doublets; the split column is the numerical degeneracy"
              .format(len(splittings)))
    return splittings


def is_titanium_d(reference, coeff, active) -> float:
    """Fraction of the converged active spinors' density sitting on Ti d.

    Not a tautology: the active space was selected on the *starting* orbitals, and an
    optimization that had rotated it onto the chlorines would give a perfectly plausible
    energy and the wrong calculation. Reported per Kramers pair, because a single spinor's
    populations are basis dependent inside a degenerate manifold while the pair sum is not.
    """
    layout = reference.ao_layout
    populations = orbital_populations(coeff, reference.data.s_ao, layout, columns=active,
                                      group="kramers")
    on_ti_d = (np.asarray(layout.ao_atom) == 0) & (np.asarray(layout.ao_l) == 2)
    fraction = populations.normalized()[on_ti_d].sum(axis=0)
    out.entry(log, "Ti d character of the converged active pairs",
              "{:.1%} - {:.1%}".format(float(fraction.min()), float(fraction.max())))
    return float(fraction.min())


def main() -> int:
    outfile = prepare_output()
    out.banner(log, kuiva.__version__, "example 3: two-component CASSCF on TiCl3")
    out.entry(log, "output file", str(outfile.relative_to(HERE)))

    # ----------------------------------------------------------------------------------
    # 1. The front end. Everything below starts from these orbitals and integrals.
    # ----------------------------------------------------------------------------------
    molecule = kuiva.Molecule(atoms=planar_mx3("Ti", "Cl", R_TICL),
                              basis="x2c-SVPall-2c", charge=0, spin=1)
    # atomic_reference=True also computes each element's free-atom reference orbitals here,
    # in this basis (cached per element) -- they are what the robust atomic-reference
    # charges below are measured against, and only the front end can build them.
    scf = kuiva.ScalarSCF(molecule, memory_gb=6.0, atomic_reference=True).run()
    reference = kuiva.Reference(scf).run()

    out.section(log, "Problem")
    out.entries(log, [
        ("system", "TiCl3", "", "planar D3h, Ti-Cl = {:.2f} A".format(R_TICL)),
        ("physics", "Ti(III) d^1 in a ligand field"),
        ("basis", "x2c-SVPall-2c"),
        ("spinors", reference.nspinor),
        ("active space", "CAS({}, {})".format(N_ACTIVE_ELEC, N_ACTIVE), "",
         "the Ti 3d shell, selected by orbital character"),
        ("CASSCF roots", N_STATES_SCF, "", "the ground Kramers doublet"),
        ("CASCI roots at the converged orbitals", N_STATES_CI, "", "5 Kramers doublets"),
    ])

    # ----------------------------------------------------------------------------------
    # 2. The CASSCF.
    # ----------------------------------------------------------------------------------
    # The orbital optimizer is shared by every CI method in the program: RDMs in, orbital
    # rotation out, with the CI entering only as a callback. mode="auto" is the robust
    # default rather than the cheapest one, and escalates on the gradient trajectory rather
    # than on the energy. For a heavy element or a large state average, ask for
    # mode="second-order" explicitly -- it is the orbital problem that decides, not the CI.
    #
    # max_iter and conv_grad are stated rather than left to chance, so the run terminates
    # on a budget instead of on hope.
    cas = kuiva.CASSCF(reference, character=("Ti", "d"), n_active=N_ACTIVE,
                       n_active_elec=N_ACTIVE_ELEC, n_states=N_STATES_SCF,
                       max_iter=MAX_ITER, conv_grad=CONV_GRAD).run()

    out.section(log, "The finished CASSCF stage")
    log.info("%s", cas.summary())

    # The two ends of the state-average boundary check. Both are printed by the stage
    # itself; they are repeated here because they are the number to read first when a
    # spectrum looks odd. "Complete" would mean the average used the whole determinant
    # space, so there is no boundary at all.
    out.entries(log, [
        ("states averaged over", "{} of {}".format(cas.boundary.n_states,
                                                   cas.boundary.ndet)),
        ("gap at the starting orbitals", cas.boundary_initial.gap_cm, "cm^-1", "",
         out.CM_FMT),
        ("gap at the converged orbitals", cas.boundary.gap_cm, "cm^-1", "", out.CM_FMT),
        ("boundary unambiguous", cas.boundary.is_clean and cas.boundary_initial.is_clean),
    ])

    # ----------------------------------------------------------------------------------
    # 3. The spectrum, as a CASCI at the converged orbitals.
    # ----------------------------------------------------------------------------------
    # The stage classes are a thin layer over the module drivers, and the drivers stay
    # public: a CASCI at fixed orbitals is one call to interface.api.casci.
    #
    # The active space is given as the spinor INDICES the CASSCF actually used, not as
    # character again. The orbitals have moved, so re-running a character selection could
    # legitimately return a different set and the spectrum would not be the one belonging
    # to these orbitals.
    out.section(log, "The spin-orbit spectrum at the converged orbitals")
    spectrum = casci(reference.reference, n_states=N_STATES_CI, coeff=cas.coeff,
                     active=cas.active.spaces.active, n_active_elec=N_ACTIVE_ELEC)
    splittings = spectrum_table("TiCl3 d^1 in D3h", spectrum.total_energies)

    out.entries(log, [
        ("Kramers doublets", len(splittings)),
        ("largest pair splitting", max(abs(s) for s in splittings), "Eh", "", out.SCI_FMT),
        ("ligand-field span", (float(spectrum.total_energies[-1])
                               - float(spectrum.total_energies[0])) * HARTREE_TO_CM,
         "cm^-1", "", out.CM_FMT),
    ])
    out.note(log, "the CI is already two-component, so these roots ARE the spin-orbit")
    out.note(log, "eigenstates. There is no separate spin-orbit mixing step and no")
    out.note(log, "spin-free intermediate to interpret.")

    ti_d_fraction = is_titanium_d(reference.reference, cas.coeff,
                                  cas.active.spaces.active)

    # ----------------------------------------------------------------------------------
    # 3b. Atomic charges, the robust kind. Populations in the free-atom reference basis
    #     the front end ingested (atomic_reference=True above): stable in sign and to
    #     ~0.1 e across basis sets where a plain Loewdin charge flips sign on this very
    #     molecule. Computed twice -- from the SCF guess density and from the converged
    #     state-averaged CASSCF density -- to show a correlated density moves a charge
    #     smoothly, not qualitatively.
    # ----------------------------------------------------------------------------------
    q_scf = reference.atomic_reference_charges(report=False)
    spaces = cas.active.spaces
    n_sp = cas.coeff.shape[1]
    gamma = np.zeros((n_sp, n_sp), dtype=complex)
    inact = np.asarray(spaces.inactive)
    gamma[inact[:, None], inact] = np.eye(inact.size)
    act = np.asarray(spaces.active)
    gamma[act[:, None], act] = cas.ci.gamma
    q_cas = reference.atomic_reference_charges(coeff=cas.coeff, dm=gamma, report=False)
    out.subsection(log, "Atomic-reference charges (SCF guess vs converged CASSCF)")
    out.entries(log, [
        ("q(Ti), SCF guess density", float(q_scf.charge[0]), "", "", "{:+.4f}"),
        ("q(Ti), CASSCF state-averaged density", float(q_cas.charge[0]), "", "", "{:+.4f}"),
        ("q(Cl), CASSCF", float(q_cas.charge[1]), "", "", "{:+.4f}"),
    ])
    out.note(log, "the titanium is unambiguously positive, as Ti(III) must be -- the")
    out.note(log, "number a plain Loewdin partition gets qualitatively wrong here.")

    # ----------------------------------------------------------------------------------
    # 3c. Per-atom control, demonstrated on the front end only. One extra cheap SCF with
    #     (a) a bigger basis on ONE chlorine -- keys are element symbols, atom labels
    #     ("Cl2") or 1-based atom numbers, most specific wins -- and (b) an explicit
    #     Ti(III) reference state, which feeds the mean field and the charges alike. Atoms
    #     of one element that differ get decorated labels, and charges against non-default
    #     references announce that they are not comparable with default-reference ones
    #     (the WARNING below is the feature working, not a problem with the run).
    # ----------------------------------------------------------------------------------
    out.section(log, "Per-atom basis and reference state (front end only)")
    molecule_pa = kuiva.Molecule(atoms=planar_mx3("Ti", "Cl", R_TICL),
                                 basis={"default": "x2c-SVPall-2c",
                                        "Cl2": "x2c-TZVPall-2c"},
                                 charge=0, spin=1)
    scf_pa = kuiva.ScalarSCF(molecule_pa, memory_gb=6.0, screening="none",
                             atomic_reference=True,
                             configuration={"Ti": "+3"}).run()
    ref_pa = kuiva.Reference(scf_pa).run()
    q_pa = ref_pa.atomic_reference_charges(report=True)
    out.entries(log, [
        ("AO functions (uniform SVP was {})".format(scf.data.nao), scf_pa.data.nao),
        ("q(Ti) against the Ti(3+) reference", float(q_pa.charge[0]), "", "", "{:+.4f}"),
    ])

    # ----------------------------------------------------------------------------------
    # 4. Assert.
    # ----------------------------------------------------------------------------------
    checks = {
        "the CASSCF converged": bool(cas.converged),
        "five Kramers doublets in the spectrum": len(splittings) == 5,
        "the pairs are degenerate to better than {:.0e} Eh".format(KRAMERS_SPLIT_MAX):
            max(abs(s) for s in splittings) < KRAMERS_SPLIT_MAX,
        "the CASCI at converged orbitals reproduces the CASSCF doublet":
            abs(float(np.mean(spectrum.total_energies[:2])) - cas.energy) < 1e-8,
        "the state average is unambiguous at both ends":
            bool(cas.boundary.is_clean and cas.boundary_initial.is_clean),
        "the boundary gap exceeds {:.0e} cm^-1 at both ends".format(BOUNDARY_MIN_CM):
            min(cas.boundary.gap_cm, cas.boundary_initial.gap_cm) > BOUNDARY_MIN_CM,
        "the converged active space really is Ti d": ti_d_fraction > 0.5,
        "the atomic-reference charge is Ti(III)-like positive": q_scf.charge[0] > 1.5,
        "the correlated density moves the charge smoothly, not qualitatively":
            abs(float(q_cas.charge[0] - q_scf.charge[0])) < 0.3,
        "the per-atom basis upgraded exactly one chlorine":
            scf_pa.data.nao - scf.data.nao == 19,   # TZVP - SVP for one Cl
        "per-atom labels are decorated where atoms differ":
            list(q_pa.configurations) != ["Cl", "Ti"],
        "the overridden reference is announced as non-default": q_pa.any_non_default,
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
