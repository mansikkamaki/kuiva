"""Example 3 -- a state-averaged two-component CASSCF on a real molecule.

    source setup.sh          # once per shell
    python casscf_ligand_field.py

Runs in about three minutes on TiCl3 and writes ``output/casscf_ligand_field.out``.

WHAT THIS SHOWS
---------------
This is the calculation the program exists to do. Planar TiCl3 is Ti(III), one 3d electron
in a ligand field -- the smallest honest version of the systems Kuiva is aimed at, and
cheap enough to run while you read the file.

    ScalarSCF -> Reference -> CASSCF -> the spin-orbit spectrum

Eight things are worth watching, and each of them is a decision you will have to make in
any real calculation:

1. **The active space is stated as orbital character, not as indices.** "The five lowest
   Kramers pairs of d character on the titanium" is a definition another program can
   reproduce; "spinors 68 to 77" is not, and it silently follows the basis set around.
   Kuiva refuses a principal quantum number here on purpose: PySCF's principal-quantum
   labels count shells *within the basis*, which is not the same thing.

   The space is then constructed a second way, by an **AVAS** projection onto the free-atom
   Ti 3d orbitals, and the comparison is more interesting than agreement. A selection can
   only pick orbitals that already *are* the target shell; AVAS rotates several partly-d
   orbitals into it, which is the only route when the metal-ligand bond is covalent. Run
   twice here:

   * at the default threshold 0.20 it takes **seven** Kramers pairs -- the five 3d ones and
     the two Cl sigma combinations that carry 27 per cent of themselves in the Ti 3d span,
     giving CAS(5, 14);
   * above that group (0.40) it reproduces the character selection's CAS(1, 10) exactly.

   Neither is wrong. One is the covalency-aware active space and the other the pure
   ligand-field one, and the eigenvalue table is what that choice is made from -- which is
   why the threshold is documented as a *selection knob* and never as a tolerance, and why
   the gap at the cut is printed beside the answer. ⚠ The rotation stays inside the occupied
   space and inside the virtual space, so the SCF density does not move; that is checked
   here rather than assumed, because a rotation that mixed occupations would change the
   reference silently. The double shell (3d plus its diffuse 4d correlating partner) is the
   same call with `n_shells=2`, and it is the case no character threshold can find.

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

4. **Which CI symmetry mode.** The spectrum is solved twice: once on the general complex
   path, which is the default and the reference path, and once with the
   time-reversal-adapted (Kramers-restricted) one, which reaches the same ten states from
   five Kramers pairs and half the applications of H. It is an odd-electron-only cost
   option, it is worth nothing below three averaged pairs, and it needs the orbitals to be
   Kramers paired -- all three of which the run states rather than leaves you to discover.

5. **What point-group symmetry buys, and what it does not.** The molecule is run with
   `point_group="auto"`, so every orbital carries an irrep label and the run prints the
   character table of the group it is actually using, with every operation named by its
   lab-frame geometry. Two things follow. States can then be asked for **per irrep**
   (`n_states={"1E1/2": 5, "2E1/2": 5}`) instead of "lowest n", which is a request the
   general path cannot express at all. And the orbital rotation can be masked to within
   each irrep, so the labels still mean something at convergence rather than only at the
   start.

   The honest limits are printed too. The real group of planar TiCl3 is D3h; the labels
   come from the largest subgroup whose *double* group is abelian, which in this frame is
   Cs(xy) -- so the two members of every Kramers doublet carry the two conjugate fermion
   labels, and a per-irrep count of one in each is one whole doublet. Because the abelian
   group is smaller than the real one, a per-irrep count can still cut a physically
   degenerate manifold, which is exactly why the state-average boundary check of point 3
   stays load-bearing with symmetry on.

6. **What the multiplets actually are.** The abelian labels say the same thing about all
   five doublets -- every one of them is "1E1/2 + 2E1/2", because a Kramers pair always
   spans both conjugate sectors of a group this small. The converged states are therefore
   also *classified* by the irreps of the molecule's full double group, D3h, which
   separates the same five doublets into three different multiplets. That is a labelling
   and nothing else: no symmetry-adapted basis is built, the mathematics of every stage
   still runs in the abelian subgroup, and the energies are bit-for-bit the ones the
   unclassified run produces. The run prints three tables so the labels can be read: the
   abelian character table used in the math, the D3h double-group table used in the
   labelling, and the computed correspondence between them.

   ⚠ The correspondence table is the row worth reading. Every D3h fermion irrep subduces
   to *both* abelian sectors, which is the precise statement of why a per-irrep count in
   the abelian group cannot protect a multiplet -- and why, with the multiplets named, a
   state count that cuts one is refused outright.

7. **Both symmetries at once.** ``kramers="restricted"`` and ``n_states={irrep: n}``
   combine, and the combination is stated over **conjugate pairs of irreps**: time reversal
   conjugates a label, so a sector is not time-reversal-closed by itself and only the union
   of a conjugate pair is. Asking for five states of 1E1/2 in the restricted mode therefore
   returns ten -- the five and their time-reversed partners in 2E1/2 -- which is the same
   spectrum again, reached with both symmetries imposed.

8. **Trying to NAME the levels, and being told they have no free-ion name.** ``<S^2>`` is
   0.7511 on every doublet, and the excess over 3/4 is the interesting part: one active
   electron cannot mix spins, so spin-pure orbitals would give 3/4 exactly (boron does, in
   example 2). These orbitals are not spin-pure -- a converged general-complex CASSCF mixes
   spin, which is what spin-orbit coupling is -- so ``S`` reaches out of the active space,
   and that contribution is computed and included rather than assumed away. It goes as the
   square of the leakage, and the run prints both so the excess can be checked against its
   own cause. The term assignment then **refuses**: a
   free-ion level is a 2J+1 manifold whose g is the Lande factor, and a crystal-field
   doublet is neither, so every label comes back "?" with the evidence saying why. That is
   the correct answer, and it is the half of this feature worth seeing -- example 2 runs
   the same call on a free atom, where the inference succeeds. ⚠ Every label is an
   inference either way, which is why it prints in its own report with the evidence and a
   fit residual beside it, and never as a column of the state table.

THE ANSWER YOU SHOULD SEE
-------------------------
D3h splits the d shell into a1' + e'' + e', so the ten spinors of the 3d shell give **five
Kramers doublets**:

    0, 0 | 15442, 15442 | 15521, 15521 | 23882, 23882 | 23958, 23958   cm^-1

Five pairs, each exactly degenerate, and with symmetry on each pair carries one state of
1E1/2 and one of 2E1/2 -- the two conjugate fermion irreps of Cs(xy), which is what a
Kramers pair looks like once the abelian group is smaller than the real one. Classified in
the full group the same five doublets read **1E1/2, 1E1/2, E3/2, E3/2, 2E1/2**: three
distinct D3h multiplets where the abelian labels saw one. That pattern
is fixed by symmetry and by Kramers' theorem, not by any parameter of this program. The absolute splittings are a small-basis
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
from kuiva.interface.api import (assign_states, avas_active_space, casci,
                                 spin_analysis)
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

#: AVAS projection cut that isolates the Ti 3d shell alone. ⚠ A *selection* knob, not a
#: tolerance: the default 0.20 also takes the two Cl sigma combinations that carry 27% of
#: themselves in the Ti 3d span, and 0.40 sits in the gap between those two groups. The
#: eigenvalue table printed beside it is what a value like this is chosen from.
AVAS_TIGHT = 0.40

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
    # point_group="auto" tests the operations in the frame the geometry is GIVEN in -- the
    # molecule is never reoriented, because that would move the gauge origin and every
    # property operator fixed with it. Planar TiCl3 built as above has its C3 axis along z
    # and its ligands in the xy plane, so what is found here is the reflection sigma(xy).
    molecule = kuiva.Molecule(atoms=planar_mx3("Ti", "Cl", R_TICL),
                              basis="x2c-SVPall-2c", charge=0, spin=1,
                              point_group="auto")
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
    # 1b. The second route to the same active space: AVAS.
    # ----------------------------------------------------------------------------------
    # The character selection above takes the five lowest Kramers pairs whose Loewdin
    # population is at least half on Ti d. That works here because TiCl3 is ionic enough
    # for five canonical orbitals to *be* the d shell. Where the metal-ligand bond is
    # covalent, no single orbital carries enough d character to clear any threshold and
    # the useful active space is a rotation of several -- which a selection cannot produce.
    #
    # AVAS builds that rotation: project every orbital onto the free-atom Ti 3d orbitals
    # (the ones atomic_reference=True already computed, at the same per-element reference
    # state the atomic mean field uses), diagonalize the projector, and take the
    # combinations of large eigenvalue. The rotation stays inside the occupied space and
    # inside the virtual space, so the SCF density -- and the reference everything below is
    # built on -- does not move at all.
    #
    # The eigenvalues are the real output: each says how much of that rotated pair lies in
    # the Ti 3d span, and the gap at the cut says whether the electronic structure or the
    # threshold is what chose the space. It is run here at TWO thresholds, because on this
    # molecule the two answers are different and the difference is the whole lesson.
    #
    # In production it would be written as CASSCF(reference, avas=dict(atom="Ti", l="d")),
    # which also hands the CASSCF the rotated orbitals; avas=dict(..., n_shells=2) asks for
    # the double shell, the 3d plus its correlating 4d partner, which is the case no
    # character threshold finds because the correlating shell is diffuse and covalent.
    out.section(log, "The active space, chosen a second way (AVAS)")
    avas = avas_active_space(reference.reference, atom="Ti", l="d")
    avas_tight = avas_active_space(reference.reference, atom="Ti", l="d",
                                   threshold=AVAS_TIGHT, report=False)
    character_active = list(kuiva.CASSCF(reference, character=("Ti", "d"),
                                         n_active=N_ACTIVE,
                                         n_active_elec=N_ACTIVE_ELEC).space.spaces.active)
    avas_active = list(avas.space.spaces.active)
    tight_active = list(avas_tight.space.spaces.active)
    # ⚠ The property the occupation grouping exists for, checked rather than assumed: a
    # rotation that mixed orbitals of different occupation would move the SCF density that
    # every stage below is built on, and nothing else in this file would notice.
    guess = reference.reference.spinors_in_ao()
    occ = reference.reference.spinors.occ
    avas_density_shift = float(np.abs((guess * occ) @ guess.conj().T
                                      - (avas.coeff * occ) @ avas.coeff.conj().T).max())
    out.entries(log, [
        ("character selection picks", str(character_active)),
        ("AVAS at the default threshold 0.20 picks", str(avas_active)),
        ("AVAS at threshold {:.2f} picks".format(AVAS_TIGHT), str(tight_active)),
        ("the tighter AVAS and the character selection agree",
         tight_active == character_active),
        ("eigenvalue gap at the cut, 0.20 / {:.2f}".format(AVAS_TIGHT),
         "{:.3f} / {:.3f}".format(avas.gap, avas_tight.gap)),
        ("change in the SCF density", avas_density_shift, "", "the rotation is inside the "
         "occupied and virtual spaces", out.SCI_FMT),
    ])
    out.note(log, "the default threshold takes two MORE pairs than the character selection:")
    out.note(log, "the two Cl sigma combinations that carry 27% of themselves in the Ti 3d")
    out.note(log, "span. Neither answer is wrong -- CAS(5,14) is the covalency-aware space")
    out.note(log, "and CAS(1,10) the pure ligand-field one -- and the eigenvalue table above")
    out.note(log, "is what the choice is made from. That is what the threshold means, and")
    out.note(log, "why the gap is printed beside it rather than left to be assumed.")

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

    # ----------------------------------------------------------------------------------
    # 3b. Trying to NAME the levels, and being told they have no free-ion name.
    # ----------------------------------------------------------------------------------
    # Two measurements per degenerate block: <S^2>, from applying S to the CI vectors, and
    # the isotropic g.
    #
    # <S^2> comes out at 0.7511 rather than exactly 3/4, and the small excess is the point
    # rather than an error. One active electron cannot mix spins, so a doublet built on
    # spin-PURE orbitals would give 3/4 exactly -- as the boron atom of example 2 does. These
    # orbitals are not spin-pure: a converged general-complex CASSCF mixes spin, which is
    # what spin-orbit coupling IS, so the spin operator connects the active space to the
    # inactive and virtual ones (measured here at 9e-03) and S|I> leaves the CAS space. That
    # contribution is COMPUTED AND INCLUDED, never assumed away, and it goes as the square of
    # the leakage: 9e-03 here gives 1.7e-03, and boron's 8e-05 gives 4e-08. Both are printed
    # so the excess can be checked against its own cause instead of being taken on trust.
    #
    # The assignment is the interesting half, because of what it REFUSES. A free-ion level
    # is a 2J+1 manifold and its g is the Lande factor, which can be inverted for L; a
    # crystal-field doublet of a complex is neither. So the labels come back "?" with the
    # evidence saying why, and that is the correct answer -- a labeller that produced
    # "^2D_5/2" here would be reading a ligand-field state as something it is not. Example 2
    # runs the same call on a free atom, where the inference succeeds and the residual is
    # the picture-change error of the bare L and S operators.
    #
    # ⚠ Every label is an inference either way, which is why it lives in its own report with
    # the evidence and a fit residual beside it, and never as a column of the state table.
    out.section(log, "Naming the levels, and what a ligand field does to the question")
    assignment = assign_states(reference.reference, spectrum)
    spin = spin_analysis(reference.reference, spectrum)

    # ----------------------------------------------------------------------------------
    # 3a. The same spectrum, solved with the time-reversal-adapted (Kramers-restricted) CI.
    # ----------------------------------------------------------------------------------
    # A non-default symmetry mode, for ODD electron counts only, and it is kept for cost --
    # not for the degeneracy, which the general path already delivers below. The
    # eigensolver keeps its expansion subspace closed under time reversal, so one stored
    # direction spans a whole Kramers pair and half as many applications of H deliver the
    # same spectrum. Everything it returns is in the general convention: the same energies
    # in the same order, the same density matrices, the same state-averaging gate.
    #
    # ⚠ Two honest caveats, both worth knowing before you switch it on:
    #
    #   * the saving arrives above TWO Kramers pairs. At two, a block Davidson adds one
    #     expansion direction per iteration where the general path adds two, needs twice
    #     the iterations, and comes out slightly slower. The CASSCF above averages over
    #     exactly one doublet, which is why the mode is used here on the ten-root spectrum
    #     and not on the orbital optimization.
    #   * it needs the orbitals to be Kramers paired, which is a property of the orbital
    #     set and not of the method. That is checked at every solve rather than assumed,
    #     and an unrestricted reference is refused rather than silently mishandled.
    out.subsection(log, "The same spectrum, Kramers-restricted")
    restricted = casci(reference.reference, n_states=N_STATES_CI, coeff=cas.coeff,
                       active=cas.active.spaces.active, n_active_elec=N_ACTIVE_ELEC,
                       kramers="restricted", report=False)
    mode_shift = float(np.max(np.abs(restricted.total_energies
                                     - spectrum.total_energies))) * HARTREE_TO_CM
    out.entries(log, [
        ("applications of H, general", spectrum.n_apply),
        ("applications of H, Kramers-restricted", restricted.n_apply, "",
         "the same {} states from {} time-reversal pairs".format(N_STATES_CI,
                                                                 N_STATES_CI // 2)),
        ("largest shift in the spectrum", mode_shift, "cm^-1", "", out.SCI_FMT),
    ])
    out.note(log, "the two modes are two routes to one spectrum. The general path is the")
    out.note(log, "default and the reference path; this one is a cost option and says so")
    out.note(log, "when it is selected.")

    # ----------------------------------------------------------------------------------
    # 3c. The same spectrum again, asked for PER IRREP.
    # ----------------------------------------------------------------------------------
    # With labels on, n_states can be a mapping instead of a count. Each irrep is solved in
    # its own sector of the determinant space, so "the five lowest states of 1E1/2" is a
    # request rather than a filter applied afterwards -- and the states come back
    # sector-pure, which the general path's degenerate blocks are not (inside a degenerate
    # block the eigensolver may return any rotation, and here the two conjugate sectors meet
    # at every level, so half of a general solve's states are 50/50 mixtures. That is the
    # eigensolver's freedom, not impurity, and the run reports it as a block).
    #
    # ⚠ The active space here is a Kramers-paired one and the state count is even, so this
    # asks for exactly the same ten states by a different route. That is the point: it is a
    # comparison, not a shortcut.
    out.subsection(log, "The same spectrum, selected per irrep")
    fermion = [reference.reference.spinor_labels.group.irrep_name(t)
               for t in reference.reference.spinor_labels.group.labels(fermion=True)]
    per_irrep = casci(reference.reference,
                      n_states={name: N_STATES_CI // len(fermion) for name in fermion},
                      coeff=cas.coeff, active=cas.active.spaces.active,
                      n_active_elec=N_ACTIVE_ELEC, enforce_kramers=False, report=False)
    irrep_shift = float(np.max(np.abs(np.sort(per_irrep.total_energies)
                                      - spectrum.total_energies))) * HARTREE_TO_CM
    irrep_counts = {name: sum(1 for x in per_irrep.irreps if x == name) for name in fermion}
    out.entries(log, [
        ("fermion irreps of the label group", ", ".join(fermion), "",
         "conjugate pair: one member of every Kramers doublet in each"),
        ("states per irrep", ", ".join("{}: {}".format(k, v)
                                       for k, v in irrep_counts.items())),
        ("largest shift in the spectrum", irrep_shift, "cm^-1", "", out.SCI_FMT),
        ("weight outside a state's own irrep", per_irrep.sector_leakage, "", "",
         out.SCI_FMT),
    ])
    out.note(log, "the two selections are two routes to one spectrum. Per-irrep selection")
    out.note(log, "changes what n_states MEANS, so it is never applied because a group was")
    out.note(log, "detected -- only because it was asked for.")

    # ----------------------------------------------------------------------------------
    # 3c-bis. The multiplets, named by the FULL double group.
    # ----------------------------------------------------------------------------------
    # The abelian labels above are as far as one integer per spinor can go: every Kramers
    # doublet spans both conjugate sectors, so all five read "1E1/2 + 2E1/2" and the labels
    # cannot tell one doublet from another. The classification layer applies the operators
    # of the molecule's full double group (D3h here) to the converged CI states and projects
    # the block traces onto its characters, which separates the same five doublets into three
    # multiplets.
    #
    # ⚠ It CLASSIFIES and never adapts. No symmetry-adapted many-particle basis is built,
    # nothing in the CI or the orbital optimizer changes, and the energies are the same
    # numbers -- which is asserted below rather than asserted in prose. What it buys is that
    # a multiplet's dimension is then fixed by theory, so a state count that cuts one is
    # refused instead of producing a plausible average over a fragment.
    out.subsection(log, "The multiplets, in the full double group")
    unclassified = casci(reference.reference, n_states=N_STATES_CI, coeff=cas.coeff,
                         active=cas.active.spaces.active, n_active_elec=N_ACTIVE_ELEC,
                         classify=False, report=False)
    full = reference.reference.symmetry.full_group
    multiplets = list(spectrum.multiplets or ())
    distinct = sorted(set(multiplets))
    subduction = full.subduction(reference.reference.symmetry.group)
    spread = {full.irrep_names[r]: sorted(reference.reference.symmetry.group.irrep_name(t)
                                          for t in d)
              for r, d in subduction.items() if full.is_fermion(r)}
    out.entries(log, [
        ("classification group", full.name, "",
         "labels on converged states; the math ran in {}".format(
             reference.reference.symmetry.group.name)),
        ("multiplets of the ten states", ", ".join(multiplets)),
        ("distinct multiplets", len(distinct), "", ", ".join(distinct)),
        ("worst projection residual", spectrum.classification.max_residual, "", "",
         out.SCI_FMT),
    ])
    for name in sorted(spread):
        out.entry(log, "{} subduces to".format(name), " + ".join(spread[name]), "",
                  "both abelian sectors, which is why an abelian count cannot protect it")
    out.note(log, "the abelian labels call all five doublets the same thing; the full group")
    out.note(log, "separates them. The classification changes no energy and no density.")

    # ----------------------------------------------------------------------------------
    # 3c-ter. Both symmetries at once.
    # ----------------------------------------------------------------------------------
    # ⚠ Time reversal CONJUGATES an irrep label, so a sector is time-reversal-closed only
    # when it is self-conjugate and otherwise only the union of a conjugate pair is. The
    # unit of a Kramers-restricted per-irrep selection is therefore that pair: five states
    # of 1E1/2 come back as ten, the five and their partners in 2E1/2.
    out.subsection(log, "Both symmetries at once")
    combined = casci(reference.reference, n_states={fermion[0]: N_STATES_CI // 2},
                     coeff=cas.coeff, active=cas.active.spaces.active,
                     n_active_elec=N_ACTIVE_ELEC, kramers="restricted", report=False)
    combined_shift = float(np.max(np.abs(np.sort(combined.total_energies)
                                         - spectrum.total_energies))) * HARTREE_TO_CM
    out.entries(log, [
        ("requested", "{}: {}".format(fermion[0], N_STATES_CI // 2), "",
         "one irrep of the conjugate pair"),
        ("states returned", int(combined.energies.size), "",
         "the partners in {} come with them".format(fermion[1])),
        ("largest shift in the spectrum", combined_shift, "cm^-1", "", out.SCI_FMT),
    ])

    # ----------------------------------------------------------------------------------
    # 3d. A symmetry-preserving CASSCF, for comparison.
    # ----------------------------------------------------------------------------------
    # preserve_symmetry=True masks the orbital rotation to within each irrep, so exp(kappa)
    # cannot mix them and the labels are exact at convergence rather than only at the start.
    # ⚠ It is a CONSTRAINT: what it converges to is the lowest *symmetric* solution, which
    # is not the global one wherever the symmetry is spontaneously broken. Here it is not,
    # so the two agree -- and that agreement is the check worth making.
    out.subsection(log, "A symmetry-preserving CASSCF")
    cas_sym = kuiva.CASSCF(reference, character=("Ti", "d"), n_active=N_ACTIVE,
                           n_active_elec=N_ACTIVE_ELEC, n_states=N_STATES_SCF,
                           max_iter=MAX_ITER, conv_grad=CONV_GRAD,
                           preserve_symmetry=True, report=False).run()
    out.entries(log, [
        ("unconstrained CASSCF energy", cas.energy, "Eh", "", out.E_FMT),
        ("irrep-blocked CASSCF energy", cas_sym.energy, "Eh", "", out.E_FMT),
        ("difference", (cas_sym.energy - cas.energy) * HARTREE_TO_CM, "cm^-1", "",
         out.SCI_FMT),
    ])

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
        "the Kramers-restricted CI gives the same spectrum": mode_shift < 1e-4,
        "per-irrep selection gives the same spectrum": irrep_shift < 1e-4,
        "and returns sector-pure states": per_irrep.sector_leakage < 1e-8,
        "each Kramers doublet has one member in each conjugate irrep":
            set(irrep_counts.values()) == {N_STATES_CI // 2},
        "the full group separates the five doublets into three multiplets":
            len(distinct) == 3 and all(x != "?" for x in multiplets),
        "every multiplet is a whole irrep of D3h":
            spectrum.classification.max_residual < 1e-8,
        "and every one of them spreads over both abelian sectors":
            all(len(v) == 2 for v in spread.values()),
        "classifying changed no energy": np.array_equal(spectrum.energies,
                                                        unclassified.energies),
        "both symmetries at once give the same spectrum": combined_shift < 1e-4,
        "and a per-irrep request under Kramers restriction returns whole conjugate pairs":
            combined.energies.size == N_STATES_CI,
        "the symmetry-preserving CASSCF converged to the same energy":
            bool(cas_sym.converged) and abs(cas_sym.energy - cas.energy) < 1e-6,
        "and reaches it with half the applications of H":
            2 * restricted.n_apply == spectrum.n_apply,
        "the state average is unambiguous at both ends":
            bool(cas.boundary.is_clean and cas.boundary_initial.is_clean),
        "the boundary gap exceeds {:.0e} cm^-1 at both ends".format(BOUNDARY_MIN_CM):
            min(cas.boundary.gap_cm, cas.boundary_initial.gap_cm) > BOUNDARY_MIN_CM,
        "the converged active space really is Ti d": ti_d_fraction > 0.5,
        # Two independent constructions of one active space. Once AVAS is asked for the
        # d shell ALONE, they must agree element for element; if they did not, one of them
        # would be wrong, which is the only reason to run both.
        "AVAS above the covalent group picks the character selection exactly":
            tight_active == character_active,
        # And the default threshold's larger space must CONTAIN it: the extra pairs are the
        # ligand sigma combinations, not a different d shell.
        "the default AVAS space contains it, plus the two ligand sigma pairs":
            set(character_active) < set(avas_active) and len(avas_active) == 14,
        # The projector's trace is the number of reference orbitals it projects onto, so
        # this is a check on the projection, the Kramers-pair fold and the per-occupation
        # diagonalization together -- not on the selection.
        "the AVAS projections account for the whole 3d reference shell":
            abs(float(avas.eigenvalues.sum()) - 5.0) < 1e-8,
        # ⚠ Contiguity is what the per-occupation ordering exists for: the active block has
        # to straddle the occupied/virtual boundary, not sit below the core.
        "the AVAS space is one contiguous block of spinors":
            avas_active == list(range(min(avas_active), max(avas_active) + 1)),
        "a decisive eigenvalue gap at the tighter cut": avas_tight.gap > 0.3,
        "the AVAS rotation left the SCF density where it was": avas_density_shift < 1e-12,
        # ⚠ Not 3/4 to machine precision, and it should not be: these orbitals are not
        # spin-pure (see above), so the doublet carries a little spin contamination. The bar
        # is set at 5e-3 because the excess is bounded by the measured out-of-CAS-space
        # contribution, which the second check states directly -- a residual accounted for by
        # its own cause is a different thing from one that is merely small.
        "every ligand-field doublet is a doublet to 5e-3 in <S^2>":
            float(np.max(np.abs(np.asarray(spin.block_s_squared) - 0.75))) < 5.0e-3,
        "and the excess over 3/4 is accounted for by the measured spin leakage":
            float(np.max(np.abs(np.asarray(spin.block_s_squared) - 0.75)))
            <= float(np.max(np.abs(np.asarray(spin.block_leak)))) + 1.0e-9,
        # ⚠ The refusal is the point. A crystal-field doublet is not a 2J+1 manifold and its
        # g is not a Lande factor, so no free-ion level is offered -- and a labeller that
        # offered one here would be reading a ligand-field state as something it is not.
        "no free-ion term is offered for a crystal-field level":
            set(assignment.labels()) == {"?"},
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
