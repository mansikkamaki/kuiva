"""Example 11 -- letting the program choose the active space, and checking that it did.

    source setup.sh          # once per shell
    python automatic_active_space.py

Runs in about six minutes on TiCl3 and writes ``output/automatic_active_space.out``.

WHAT THIS SHOWS
---------------
Every other example in this directory states its active space by hand, which is the honest
way to run a production calculation and a slow way to start one. This example asks the
program instead:

    ScalarSCF -> Reference -> AutoCAS -> CASSCF

``AutoCAS`` is not a black box and is not meant to be read as one. It turns *physical
statements* into orbitals, measures each of them with the cheap CI, prints every keep and drop
with the number it was decided on, and proposes how many states to average over. What comes
out is the same kind of object a hand-written selection produces -- a stated active space and
a stated count -- and every check downstream runs on it unchanged.

Planar TiCl3 is Ti(III), one 3d electron in a ligand field: the same system as example 3, so
the automatic answer can be compared against a space that has been validated against an
external code.

Six things are worth watching.

1. **Level 0 takes no input at all beyond the molecule.** ``kuiva.AutoCAS(ref)`` detects the
   open d/f centres and makes each one's valence shell the target. Detection is *measured*,
   never inferred from an element symbol: an atom is a centre for l when some frontier
   Kramers pair carries most of its population on (atom, l) **and** the atom's own free-atom
   reference has an OPEN shell of that l. Both conditions are load-bearing. A Zn(2+)'s filled
   3d occupies five frontier pairs at population 1.000 and is the most convincing centre in
   any population table; a CeCl3 cerium clears the population cut on its 5d as convincingly
   as on its 4f, and only the reference state's twenty d electrons say which shell the
   physics is in. Here the titanium is a d centre at 0.940 and the chlorines are not centres
   at all -- p is not a centre channel, because an open p shell is a main-group radical and
   is asked for by name.

2. **The space that comes out is the one the external reference is defined by.** This example
   asserts it element for element against ``character=("Ti", "d"), n_active=10`` -- the
   statement the committed OpenMolcas cross-check uses. That agreement is not automatic and
   is not guaranteed in general: the shell is built by an AVAS projection onto the free-atom
   3d orbitals, which *rotates*, so what has to agree in general is the **span**, and on a
   covalent system the two constructions legitimately differ by a few per cent. On this
   molecule they agree exactly, which is why it is the example.

3. **A feature class is kept or dropped by what it does to the SPECTRUM.** The second run
   here asks for the metal-ligand bonding partners as well -- ``("bonding", "Ti")`` -- and the
   rounds table shows the class being measured, pruned and then judged on how far it moved the
   target manifold. Entanglement only *prunes*, inside one class, and is never asked whether
   the class matters: a single-orbital entropy is blind to a correlating shell and to the
   empty members of a d manifold, and a bridging orbital shares at most ln 2 nats with either
   metal however real the pathway. ⚠ Watch the middle of the verdict column: a change under
   the tolerance but above the measured noise floor comes back **"inconclusive, kept"** rather
   than decided. A larger space is the safe error and the budget bounds it.

4. **Every verdict is taken at FIXED orbitals, and the numbers are qualitative.** Each number
   in the rounds table is a cheap CI -- a truncated CI in a truncated space, at a fixed small
   budget -- at the orbitals the candidates were constructed in, with the trial space's
   determinants starting from the accepted space's. Only the space that comes out is
   pre-optimized, once, at the end. ⚠ Deciding on two pre-optimizations instead would read
   the optimizer's path rather than the class: a pre-optimization stopped on its budget
   rotates into whatever it is given, and pairs that describe nothing once moved the spectrum
   by up to 100 cm^-1 that way. Two measurements at the same orbitals and budget are
   comparable with each other and with nothing else -- never with a CASSCF -- which is also why
   the state count that comes out is a proposal rather than an answer.

5. **The proposed count is a manifold boundary at or above a theoretical floor.** For a d1
   ion the floor is the spin multiplicity, 2 -- on the d block the ligand field decides the
   orbital part and only the probe can see it, so 2J+1 is not a manifold of the molecule at
   all. The probe then chains its spectrum at the manifold gap and proposes the first clean
   boundary at or above 2. ⚠ A count that ended *inside* a near-degenerate manifold would
   make the averaged density non-invariant, the Fock operator built from it would split the
   shell, and the result would be entirely plausible and wrong -- so the boundary is extended
   outward and never truncated inward, and a spectrum with no gap proposes the floor marked
   as a lower bound rather than a number invented at the edge.

6. **The handoff adds a statement and not a calculation.** ``kuiva.CASSCF(auto)`` inherits
   the space, the orbitals *and* the proposed count, announcing the last of these in the
   output. The example asserts that this is **bitwise** the same calculation as spelling all
   three out by hand -- shown on two macro-iterations, because what is being tested is what
   the handoff carries and not how far the optimizer gets. It also asserts the refusal in the
   other direction: a restated ``character=`` on top of an AutoCAS is refused, because a
   character selection reads its populations off the reference's own SCF orbitals and these
   have been rotated.

   ⚠ **The orbitals are a by-product, not the point.** What this stage is for is the space
   and the proposal. The orbitals it hands on are a pre-optimized guess at a deliberately
   small budget, and preoptimized orbitals are a better starting *density* and a
   worse-conditioned starting *point* for the orbital problem -- the same trade-off a
   `CheapCI` upstream has. On an easy ionic system like this one the production CASSCF started
   from them therefore costs *more* than one started from the plain SCF guess, the optimizer's
   second-order escalation paying for a pre-optimization far from stationary. Worth knowing
   before reaching for a bigger probe budget on a system where the space was never in doubt.

THE ANSWER YOU SHOULD SEE
-------------------------
Level 0 reproduces CAS(1, 10) -- the five Kramers pairs of the Ti 3d shell, one electron --
and proposes an even state count at a manifold boundary with a gap of thousands of cm^-1
above it. That count is even because this is an odd-electron system and Kramers' theorem
makes every level at least two-fold; the gap is what makes the boundary unambiguous.

The absolute splittings in the rounds table are a cheap CI's and are not spectroscopy. The
CASSCF spectrum at the end of this file is, to the extent example 3's is: five Kramers
doublets from the D3h splitting of the d shell.
"""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import List

import numpy as np

import kuiva
from kuiva.interface.api import active_space_for
from kuiva.props.multiplet import HARTREE_TO_CM
from kuiva.util import output as out
from kuiva.util import resources as res
from kuiva.util import timing
from kuiva.util.logging import add_file_handler, get_logger

NAME = "automatic_active_space"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

log = get_logger("examples." + NAME)

#: Ti-Cl bond length [Angstrom] of the planar D3h gas-phase molecule -- example 3's geometry,
#: so the automatic answer is comparable with a space validated against an external code.
R_TICL = 2.25

#: What the committed cross-check states by hand: the Ti 3d shell, one electron.
N_ACTIVE, N_ACTIVE_ELEC = 10, 1

#: Cheap-CI budget for this example, smaller than the default so the run stays short. ⚠ It is
#: the same budget for every round -- two measurements are comparable only at one budget -- and
#: it is stated rather than adapted, because a budget that moved with the space would make the
#: rounds table a measurement of itself.
PROBE = dict(max_iter=6, max_determinants=4000)

#: Budgets for the production CASSCF, explicit so termination never depends on convergence.
#: ⚠ ``conv_grad`` is 1e-3 rather than the 1e-4 a production run would ask for, and the reason
#: is the one section 6 of the header names: the orbitals an AutoCAS hands over are a
#: worse-conditioned starting point than the SCF guess, the optimizer escalates to second-order
#: steps that cost a minute each here, and the last decade of the gradient would put this
#: example over its ten-minute budget for a spectrum that does not move. (Measured: |g| reaches
#: 1.4e-3 at macro-iteration 10 with the energy already settled to 8e-9 Eh.)
MAX_ITER, CONV_GRAD = 20, 1.0e-3

#: Macro-iterations of the two runs whose energies must agree **bitwise**. Two is enough: what
#: is being tested is that the handoff carries the same space, orbitals and count, and two
#: iterations of an identical trajectory show that as conclusively as forty do.
HANDOFF_ITER = 2

#: The proposal must land on a boundary this wide at least. Below 50 cm^-1 Kuiva warns that a
#: boundary is ambiguous; the ligand-field gap here is three orders of magnitude above that.
BOUNDARY_MIN_CM = 1.0e3


def planar_mx3(metal: str, ligand: str, r: float) -> List[tuple]:
    """Planar D3h MX3, metal at the origin, ligands in the xy plane."""
    atoms = [(metal, (0.0, 0.0, 0.0))]
    for k in range(3):
        theta = 2.0 * math.pi * k / 3.0
        atoms.append((ligand, (r * math.cos(theta), r * math.sin(theta), 0.0)))
    return atoms


def prepare_output() -> Path:
    """Write this run to output/<name>.out, with a scratch spin-orbit cache of its own.

    An example whose result depends on what is already in the developer's cache is not a
    demonstration of anything, so the cache is redirected here and cleared first.
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
    ])
    table.start()
    for i, energy in enumerate(energies):
        table.row(i, energy, (energies[i] - energies[0]) * HARTREE_TO_CM)
    table.end("{} states".format(len(energies)))
    return [float(energies[i] - energies[i - 1]) for i in range(1, len(energies), 2)]


def main() -> int:
    outfile = prepare_output()
    out.banner(log, kuiva.__version__, "example 11: the active space, chosen automatically")
    out.entry(log, "output file", str(outfile.relative_to(HERE)))

    # ----------------------------------------------------------------------------------
    # 1. The front end. atomic_reference=True is a REQUIREMENT here, not an option.
    # ----------------------------------------------------------------------------------
    # Every shell AutoCAS builds is an AVAS projection onto the free-atom reference orbitals,
    # and whether an atom has an OPEN shell of a given l is read off those same orbitals.
    # Only the front end can compute them (the selection layer has no integral library), so
    # the stage refuses without them and names the knob.
    molecule = kuiva.Molecule(atoms=planar_mx3("Ti", "Cl", R_TICL),
                              basis="x2c-SVPall-2c", charge=0, spin=1)
    scf = kuiva.ScalarSCF(molecule, memory_gb=6.0, atomic_reference=True).run()
    reference = kuiva.Reference(scf).run()

    out.section(log, "Problem")
    out.entries(log, [
        ("system", "TiCl3", "", "planar D3h, Ti-Cl = {:.2f} A".format(R_TICL)),
        ("physics", "Ti(III) d^1 in a ligand field"),
        ("basis", "x2c-SVPall-2c"),
        ("spinors", reference.nspinor),
        ("what is being demonstrated", "the active space chosen from stated targets"),
    ])

    # ----------------------------------------------------------------------------------
    # 2. Level 0: no input beyond the molecule.
    # ----------------------------------------------------------------------------------
    # targets=None means "the valence shell of every open d/f centre the reference shows".
    # The block this prints is the whole argument: which centres were detected and on what
    # evidence, the AVAS projections and the gap at the cut, the rounds table, the size
    # budget it was allowed, and the proposal with the floor beside it.
    auto = kuiva.AutoCAS(reference, probe=PROBE).run()
    log.info("%s", auto.summary())

    # The comparison that matters: the space a validated reference calculation states by
    # hand. ⚠ Element for element is the STRONG form of this check and holds here because
    # TiCl3's 3d shell is ionic enough for five canonical orbitals to be it. On a covalent
    # system AVAS rotates ligand admixture in, and what has to agree is the span.
    stated = active_space_for(reference.reference, character=("Ti", "d"),
                              n_active=N_ACTIVE, n_active_elec=N_ACTIVE_ELEC)
    out.section(log, "Against the hand-written space")
    out.entries(log, [
        ("automatic", "CAS({}, {})".format(auto.space.n_elec, auto.space.n_active), "",
         auto.space.description),
        ("hand-written", "CAS({}, {})".format(stated.n_elec, stated.n_active), "",
         stated.description),
        ("same spinor columns",
         list(auto.space.spaces.active) == list(stated.spaces.active)),
    ])

    # ----------------------------------------------------------------------------------
    # 3. Level 2: a named feature class, and the rounds table that decides it.
    # ----------------------------------------------------------------------------------
    # The bonding class offers the metal-ligand combinations just below the shell's
    # projection cut -- on TiCl3 the two Cl sigma pairs that carry 27% of themselves in the
    # Ti 3d span. Whether they belong in the active space is not a question a threshold can
    # answer, so the probe answers it: does adding them move the ground manifold?
    #
    # Read the verdict column rather than the size. "kept" means the class moved the target
    # manifold by more than the tolerance; "dropped" means it moved it by less than the
    # probe can resolve; "kept (inconclusive)" means in between -- which is a report about
    # the probe, not about the physics, and it keeps the class because a larger space is the
    # safe error here.
    auto_bonding = kuiva.AutoCAS(reference, targets=["shells", ("bonding", "Ti", 2)],
                                 probe=PROBE).run()
    out.section(log, "The rounds")
    out.entry(log, "space with the bonding class offered",
              "CAS({}, {})".format(auto_bonding.space.n_elec, auto_bonding.space.n_active))
    out.entry(log, "verdicts", ", ".join("{}: {}".format(r.cls, r.verdict)
                                          for r in auto_bonding.rounds))

    # ----------------------------------------------------------------------------------
    # 4. The handoff: a statement, not a calculation.
    # ----------------------------------------------------------------------------------
    # CASSCF(auto) inherits three things -- the space, the orbitals and the proposed state
    # count -- and says in the output that the count was proposed rather than requested. The
    # explicit spelling below produces the same numbers bit for bit, which is the claim this
    # stage makes about itself.
    out.section(log, "The handoff, twice")
    # The claim is that inheriting the space, the orbitals and the count is the SAME
    # calculation as spelling all three out -- so the comparison is bitwise, and two
    # macro-iterations of an identical trajectory show it as conclusively as forty would
    # while costing seconds. ⚠ The explicit run is built on the SAME upstream, because the
    # orbitals travel with the space: restating the selection against the reference's own
    # guess spinors would be a different calculation, and Kuiva refuses the form of that
    # mistake which is easy to make (`character=`) while `active=` -- a statement about the
    # orbitals at hand -- is allowed.
    short_inherited = kuiva.CASSCF(auto, max_iter=HANDOFF_ITER, report=False).run()
    short_explicit = kuiva.CASSCF(auto, active=list(auto.space.spaces.active),
                                  n_active_elec=auto.space.n_elec, n_states=auto.n_states,
                                  max_iter=HANDOFF_ITER, report=False).run()
    out.entries(log, [
        ("inherited E after {} macro-iterations".format(HANDOFF_ITER),
         float(short_inherited.energy), "Eh", "", out.E_FMT),
        ("explicit  E after {} macro-iterations".format(HANDOFF_ITER),
         float(short_explicit.energy), "Eh", "", out.E_FMT),
        ("bitwise identical",
         float(short_inherited.energy) == float(short_explicit.energy)),
    ])

    out.section(log, "The production CASSCF")
    inherited = kuiva.CASSCF(auto, max_iter=MAX_ITER, conv_grad=CONV_GRAD).run()
    splittings = spectrum_table("the automatically selected CASSCF",
                                np.asarray(inherited.energies, dtype=float))

    # ----------------------------------------------------------------------------------
    # 5. The refusal in the other direction.
    # ----------------------------------------------------------------------------------
    restated_refused = False
    try:
        kuiva.CASSCF(auto, character=("Ti", "d"), n_active=N_ACTIVE)
    except ValueError:
        restated_refused = True

    # ----------------------------------------------------------------------------------
    # 6. Checks.
    # ----------------------------------------------------------------------------------
    proposal = auto.proposal
    checks = {
        "one d centre was detected, and it is the titanium":
            len(auto.assembly.centres) == 1 and auto.assembly.centres[0].l == 2,
        "the chlorines are not centres (p is not a centre channel)":
            all(c.atoms == (0,) for c in auto.assembly.centres),
        "level 0 reproduces the hand-written active space, element for element":
            list(auto.space.spaces.active) == list(stated.spaces.active),
        "the electron count is measured off the selected orbitals":
            auto.space.n_elec == N_ACTIVE_ELEC,
        "the space is whole Kramers pairs":
            auto.space.n_active % 2 == 0 and auto.space.n_active == N_ACTIVE,
        "the statement carries no index list": "[" not in auto.space.description,
        "the proposed count is a manifold boundary": proposal.boundary.found,
        "the proposed count is at or above the theoretical floor":
            auto.n_states >= auto.floor,
        "the proposed count is even (an odd-electron system has no non-degenerate level)":
            auto.n_states % 2 == 0,
        "the boundary is unambiguous": proposal.boundary.gap_cm > BOUNDARY_MIN_CM,
        "the equivalent window sits inside the gap it was read at":
            0.0 < proposal.window.cutoff_cm < proposal.boundary.gap_cm
            + float(auto.spectrum_cm[proposal.boundary.count - 1]),
        "the CASSCF was told where its state count came from":
            "AutoCAS" in inherited.n_states_source,
        "the handoff is bitwise the same calculation as spelling it out":
            float(short_inherited.energy) == float(short_explicit.energy),
        "a restated character= on an AutoCAS upstream is refused": restated_refused,
        # A verdict rather than a number: what the example claims is that the class was
        # *decided*, and "the manifold changed structure" is a decision too (it reports no
        # distance, because a change of size is not one).
        "the bonding round was probed and given a verdict":
            any(r.cls == "bonding"
                and r.verdict in ("kept", "kept (inconclusive)", "dropped")
                for r in auto_bonding.rounds),
        "the shell round is the core and is never pruned":
            auto_bonding.rounds[0].verdict == "core"
            and auto_bonding.rounds[0].n_pruned == 0,
        "the CASSCF converged": bool(inherited.converged),
        "the spectrum comes in Kramers doublets":
            all(abs(s) < 1.0e-6 for s in splittings),
    }
    failures = report(checks)
    timing.summary(log)
    res.summary(log)
    return 1 if failures else 0


def report(checks) -> int:
    """Print the check table and return the number of failures."""
    out.section(log, "Result")
    table = out.Table(log, [out.Column("check", "{}", 68, align="<"),
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
