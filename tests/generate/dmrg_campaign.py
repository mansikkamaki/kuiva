"""Systems, grading bands and shared machinery for the DMRG cost/reliability campaign.

**A study support module, not a reference generator.** It exists to answer two questions the
saturated-bond-dimension validation record could not: what a *truncating* tensor network
costs on real heavy-element multiplet structure, and how little of that cost still buys an
answer a chemist would act on. The driver is :mod:`dmrg_cost_ladder`; every number this
module produces lands in ``temp/`` and nothing here is committed reference data.

What it holds
-------------
* the campaign's systems — the reused ones taken from :mod:`systems` so the geometry and the
  active-space statement keep exactly one source, and two new ones (``dycl3``, ``uf3``)
  stated inline with their provenance until the campaign decides whether they are worth
  committing;
* the **grading bands** (:data:`BANDS`, user-confirmed 2026-08-29) and the reduction that
  applies them;
* the shared plumbing: reference, converged orbitals, active integrals, the exact-CI oracle,
  one network ladder point, and the phase-invariant comparison between them;
* the **protocol-B** primitives (:func:`network_solver`, :func:`relax_orbitals`): the same
  orbital optimization driven by two different CI solvers, which is the only way to ask
  whether orbital relaxation absorbs a truncation error or amplifies it.

The comparison discipline, which is the whole design
----------------------------------------------------
⚠ **Same integrals, same orbitals, same downstream code.** A ladder point differs from its
oracle in one thing only: where the CI vectors came from. Both sides go through the *same*
property-matrix assembly and the *same* phase-invariant reduction, because a moment matrix
carries arbitrary phases and degenerate states mix arbitrarily — element-by-element
comparison of the matrices is meaningless, and the only sound statements are the degeneracy
pattern, the relative energies and the block invariant ``Tr_block(mu_i mu_j)`` with its
principal g values.

⚠ **The blocks are the ORACLE's blocks.** Truncation splits a Kramers pair by roughly twice
the energy error (the local validation notes measure the relation), so blocking the two
spectra independently would compare different partitions and read a truncation error as a
changed degeneracy pattern. The reduction therefore evaluates the network's spectrum and
moments at the *oracle's* block boundaries and reports the within-block spread separately —
which is also the quantity the degeneracy grading needs.

⚠ **A refusal is a data point, not a crash.** The group-complete truncation rule declines to
cut through a degenerate Schmidt group rather than round it, and near free-ion degeneracy it
refuses most low caps outright. "This cap is not reachable on this system" is part of the
answer, so every refusal is recorded with the cap that produced it.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
#: Orbital checkpoints and per-system scratch. Git-ignored; nothing here is a reference.
WORK = REPO / "temp/dmrg_campaign"
#: One JSON file per stage, rewritten after every completed ladder point.
RECORDS = REPO / "temp/dmrg_cost_ladder"

#: Degeneracy tolerance used to block the ORACLE spectrum [cm^-1]. Loose enough that a
#: physically degenerate pair is one block, tight enough that two crystal-field levels are
#: not: a Dy(III) ligand field puts its Kramers doublets tens to hundreds of cm^-1 apart.
BLOCK_TOL_CM = 1.0


# --- the grading bands (user decision, 2026-08-29) ------------------------------------------
@dataclass(frozen=True)
class Bands:
    """What "comparable to CASSCF" means, on the three-level scale.

    ⚠ **A tier is a statement about the answer to a physical question, not about digits.**
    *Quantitative* means the truncated result is interchangeable with the exact one;
    *qualitative* means the numbers differ visibly but the physics does not, by an amount
    comparable to what is already being neglected outside the active space (dynamic
    correlation moves a ligand-field splitting by 10-15 % routinely); *unacceptable* means
    the answer itself changed — a reordered level, a collapsed or invented gap, an
    easy-axis/easy-plane flip, a wrong multiplet dimension.

    Two guards on the grading, both of which exist because the naive form flunks correct
    results. **Near-degenerate levels get an ordering exemption**: a swap of two levels
    closer together than the truncation error is precision, not physics, and is graded on
    the pair's positions rather than on their order. And **the percentage is judged per
    splitting with an absolute floor**, so an exchange-scale gap of a few thousandths of a
    wavenumber is not flunked for a sub-cm^-1 absolute error no observable could see.
    """

    #: Relative energies of the multiplets [cm^-1] and [relative].
    energy_quant_cm: float = 0.35
    energy_quant_rel: float = 0.005
    energy_qual_cm: float = 10.0
    energy_qual_rel: float = 0.15
    #: Principal g values per block [relative].
    g_quant_rel: float = 2e-3
    g_qual_rel: float = 0.10
    #: ⚠ **The moment floor: below it a block HAS no moment, and is graded on that rather
    #: than on a percentage.** A relative deviation needs something non-zero to be relative
    #: to, and a block can legitimately carry no moment at all — an ``|Omega| = 0`` level
    #: of an axial ion comes out at g ~ 0.005, which is zero by symmetry with rounding on
    #: top. Two wrong answers were tried before this one. Grading such a block against
    #: ITSELF turns a 6e-4 absolute difference into a 12% error and flunks a result whose
    #: every physically meaningful g agrees to six figures (measured on FeCl2 at a
    #: saturating bond dimension). Grading it against a floor is no better: it holds a
    #: no-moment block to a TIGHTER absolute tolerance than a real one, which is backwards.
    #: So the question asked of a block below the floor is the only one that means
    #: anything — is it still below the floor? A moment appearing where symmetry forbids
    #: one is a qualitative failure; 0.005 against 0.004 is not a number anyone measures.
    #: 0.1 is two orders below a free-electron g and below anything an experiment
    #: distinguishes from zero.
    g_floor: float = 0.1
    #: A level pair closer than this is exempt from the ordering rule (see the class note).
    ordering_exempt_cm: float = 1.0


BANDS = Bands()

#: The three tiers, best first. Ordering is used by :func:`worst_of`.
TIERS = ("quantitative", "qualitative", "unacceptable")


def worst_of(tiers: Sequence[str]) -> str:
    """The worst tier present — a system's grade is the worst of its metrics."""
    idx = max((TIERS.index(t) for t in tiers if t in TIERS), default=0)
    return TIERS[idx]


# --- geometries for the new systems ---------------------------------------------------------
def _planar_mx3(metal: str, ligand: str, r: float) -> List[Tuple[str, Tuple[float, ...]]]:
    """Planar D3h MX3: metal at the origin, ligands in the xy plane, C3 along z.

    The same construction the committed ``cecl3``/``ticl3``/``tif3`` systems use, so a
    lanthanide trihalide added here is the ligand-field analogue of one already validated
    rather than a differently posed problem. ⚠ The C3 axis is z on purpose: spin is
    quantized along z, and a symmetry axis that is not z is reported and not used.
    """
    atoms = [(metal, (0.0, 0.0, 0.0))]
    for k in range(3):
        th = 2.0 * math.pi * k / 3.0
        atoms.append((ligand, (r * math.cos(th), r * math.sin(th), 0.0)))
    return atoms


def _pyramidal_mx3(metal: str, ligand: str, r: float,
                   angle_deg: float) -> List[Tuple[str, Tuple[float, ...]]]:
    """Pyramidal C3v MX3 with bond length ``r`` and X-M-X angle ``angle_deg``, C3 along z.

    The three ligands sit on a cone of half-angle ``beta`` about -z, where
    ``cos(theta) = 1.5 cos^2(beta) - 0.5`` relates the X-M-X angle to it. Planar is the
    ``theta = 120`` limit, so the same helper states both and the choice is a number rather
    than a second construction.
    """
    cos_t = math.cos(math.radians(angle_deg))
    cos_b2 = (cos_t + 0.5) / 1.5
    if not 0.0 <= cos_b2 <= 1.0:
        raise ValueError("X-M-X angle {} deg is not realisable for a C3v MX3"
                         .format(angle_deg))
    cos_b, sin_b = math.sqrt(cos_b2), math.sqrt(max(0.0, 1.0 - cos_b2))
    atoms = [(metal, (0.0, 0.0, 0.0))]
    for k in range(3):
        th = 2.0 * math.pi * k / 3.0
        atoms.append((ligand, (r * sin_b * math.cos(th), r * sin_b * math.sin(th),
                               -r * cos_b)))
    return atoms


# --- the campaign record --------------------------------------------------------------------
@dataclass(frozen=True)
class Campaign:
    """One campaign system and the protocol applied to it.

    ``character`` is the active-space statement in physical terms, exactly as
    ``api.casscf`` takes it — never an orbital-index window, because a reference is only
    valid if an independent implementation can define the same calculation.

    ``n_states`` is both the state-averaged CASSCF ensemble and the root count of the ladder
    task, unless ``ladder_states`` says otherwise: the two differ only where the orbital
    protocol is inherited from a committed record whose ensemble is larger than the
    manifold under test.
    """

    key: str
    label: str
    atoms: List[Tuple[str, Tuple[float, ...]]]
    charge: int
    spin: int
    basis: object                      # str, or {element/label/number: str} per atom
    n_active: int                      # spinors
    n_active_elec: int
    character: object = None           # (atom, l) or the fragment-list ordinal-window form
    #: The FULL keyword form of the selection, when it is not simply ``character`` +
    #: ``n_active`` + ``n_active_elec``. ⚠ Carried whole rather than rebuilt at the call
    #: site: the fragment-list form encodes its own spinor count and its ordinal window,
    #: and re-deriving either is how a selection lands on a filled shell instead of the
    #: valence one — silently, because no observable this campaign grades can see it.
    selection_kw: Optional[Dict[str, object]] = None
    n_states: int = 1
    caps: Tuple[int, ...] = ()
    n_det: int = 0
    ladder_states: Optional[int] = None
    #: Whether an exact CI oracle exists for this system at all. ⚠ **Set ``False`` only
    #: where the memory pre-flight has REFUSED it**, and record the refusal: past the
    #: conventional-CI ceiling there is no exact answer to grade against, and the ladder is
    #: then run in the mode Tier 3 will have to use — reduced at the manifold structure the
    #: rungs below establish, and graded for internal convergence in D. It is not a way to
    #: skip an oracle that could have been computed, which would make every number below it
    #: a claim with nothing behind it.
    exact_oracle: bool = True
    #: Where the ladder's fixed orbitals come from: ``"casscf"`` (the converged
    #: state-averaged reference) or ``"guess"`` (the scalar reference's own spinors).
    #: ⚠ **Both are legitimate for this measurement and they are not the same claim.**
    #: Every quantity the ladder grades is a difference between a truncated network CASCI
    #: and an exact CASCI *at the same orbitals and the same integrals*, so orbital quality
    #: cancels out of the comparison — orbital feedback is a separate protocol. What
    #: changes with the orbitals is the entanglement of the active space, hence how hard
    #: the truncation has to work, so a guess-orbital ladder measures a slightly different
    #: system and says so. The precedent is this project's own ab initio manifold ladder,
    #: which ran its multi-site systems at guess orbitals for exactly this reason.
    orbitals: str = "casscf"
    casscf_mode: str = "auto"
    max_iter: int = 60
    conv_grad: float = 1e-4
    #: SCF convergence controls for this system, passed to the front end unchanged.
    #: ⚠ Not a formality for an open f shell: a 4f^9 ROHF has a dense manifold of nearly
    #: degenerate solutions and plain DIIS oscillates between them forever. What is NOT
    #: allowed here is ``allow_unconverged_scf`` — everything downstream is built on these
    #: orbitals and "the CASSCF re-optimizes them" is a hope, not a property, so the fix is
    #: the guess and the damping, never a flag that proceeds anyway.
    scf_options: Dict[str, object] = field(default_factory=dict)
    #: A first, deliberately UNCONVERGED SCF whose orbitals seed the real one through
    #: ``guess_from=``. ⚠ **Not a convergence trick — a choice of solution.** An open 4f
    #: shell has many stationary points, and the second-order solver converges to the one
    #: nearest its guess: on DyCl3 it lands, stably and with every check clean, 0.66 Eh
    #: ABOVE the ground SCF state from the default guess. What picks the right basin is
    #: where the first stage leaves the orbitals, so the prelude is part of the definition
    #: of the calculation and is recorded with it.
    scf_prelude: Optional[Dict[str, object]] = None
    #: Opt-in non-Kramers pairing [cm^-1]. Set ONLY where the ground block is genuinely two
    #: tunnelling-split singlets rather than a Kramers doublet — an integer-spin ion. It is a
    #: physical claim, never inferred, and a wrong grouping makes a plausible g out of two
    #: unrelated states.
    pseudo_doublet_tol_cm: Optional[float] = None
    geom_note: str = ""
    physics_note: str = ""
    protocol_note: str = ""

    @property
    def roots(self) -> int:
        """Roots the ladder solves for — the manifold under test."""
        return int(self.n_states if self.ladder_states is None else self.ladder_states)

    @property
    def element(self) -> str:
        return self.atoms[0][0]

    def selection(self) -> Dict[str, object]:
        """The active-space keyword arguments, one place so no call site rebuilds them."""
        if self.selection_kw is not None:
            return dict(self.selection_kw)
        return dict(character=self.character, n_active=self.n_active,
                    n_active_elec=self.n_active_elec)


def _from_suite(key: str, *, caps: Sequence[int], ladder_states: Optional[int] = None,
                pseudo_doublet_tol_cm: Optional[float] = None,
                            casscf_mode: str = "auto", protocol_note: str = "") -> Campaign:
    """A campaign system built from a committed suite entry — one source of truth.

    ⚠ Geometry, basis, charge, spin, the active-space statement (its ordinal window
    included) and the state count all come from the suite record. Restating any of them
    here would be a second definition of a system that already has one.
    """
    import systems as sysdef

    s = sysdef.get(key)
    return Campaign(
        key=key, label=s.label, atoms=list(s.atoms), charge=s.charge, spin=s.spin,
        basis=s.basis, n_active=2 * s.ncas, n_active_elec=s.nelecas,
        selection_kw=sysdef.character_selection(s),
        n_states=s.soc_states, ladder_states=ladder_states, caps=tuple(caps),
        casscf_mode=casscf_mode, pseudo_doublet_tol_cm=pseudo_doublet_tol_cm,
        geom_note=s.geom_note, physics_note=s.physics_note, protocol_note=protocol_note)


#: Bond-dimension ladders. Cheapest first, so a budget kill leaves the *interesting* end —
#: the low caps, where the approximation is real — measured rather than the saturating end,
#: which the existing record already covers.
CAPS_SMALL = (2, 3, 4, 6, 8, 12, 16, 24, 32)
CAPS_F_SHELL = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)

#: The ordinal window naming uranium's VALENCE 5f shell, measured off the ``uf3``
#: reference's own character-ordered populations rather than counted off a periodic table.
#: ⚠ Uranium has a filled **4f** shell below the 5f (and filled 3d/4d/5d below the 6d), so
#: a valence shell here is not the lowest of its character and a bare ``(U, l)`` selection
#: takes core orbitals — silently, because an f^3 shell gives the same 4I9/2 pattern and the
#: same Lande g whichever f shell holds it.
U_F_SKIP_PAIRS = 7

#: Ordinal windows for the DyCl3 rungs, measured on that reference the same way (the
#: populations are recorded with the campaign's results). ⚠ Chlorine's filled **2p** is
#: nine Kramers
#: pairs across the three ligands and lies below the valence 3p; dysprosium's filled **3d**
#: and **4d** are ten pairs below the 5d. Neither is visible in an occupation — a filled
#: core shell and a filled valence shell both read 2.0.
CL_2P_SKIP_PAIRS = 9
DY_4D_SKIP_PAIRS = 10


def systems() -> Dict[str, Campaign]:
    """Every campaign system, keyed. Built lazily so importing this module is cheap."""
    out: Dict[str, Campaign] = {}
    for camp in (
        # --- reused, committed geometry and active space -------------------------------
        _from_suite("tif3", caps=CAPS_SMALL,
                    protocol_note="committed suite protocol unchanged; 10 determinants, so "
                                  "the ladder saturates to exact CI"),
        _from_suite(
            "fecl2", caps=CAPS_SMALL,
            protocol_note="committed suite protocol unchanged. THE integer-spin system and "
                          "the only one: its ground block is a NON-Kramers pseudo-doublet "
                          "with an analytic g_z = 12.009 and transverse g values that are "
                          "zero by symmetry rather than small, so it is the only place a "
                          "truncation can be tested against a non-Kramers structure. "
                          "⚠ pseudo_doublet_tol_cm is deliberately LEFT UNSET: with the "
                          "molecular axis along z the pair is degenerate by the axial "
                          "field, so the ordinary blocking already groups it and its "
                          "largest principal g IS the analytic g_z. Setting a tolerance "
                          "would only matter if the pair came out split, and pairing two "
                          "singlets that are not a tunnelling-split doublet makes a "
                          "plausible g out of two unrelated states — a physical claim to "
                          "be made on a measured gap, never in advance"),
        _from_suite(
            "dy3p", caps=CAPS_F_SHELL, ladder_states=16,
            protocol_note="ORBITALS from the committed 134-root protocol, LADDER on the "
                          "16-fold ground multiplet. The orbital protocol is inherited "
                          "unchanged because it is the one the committed record "
                          "establishes: a 126-root average cuts that spectrum's last "
                          "manifold in half and splits the ground multiplet by 44.85 "
                          "cm^-1, and re-deciding the ensemble here would put a "
                          "state-averaging question inside a truncation measurement. The "
                          "ladder solves the same 16-root task as dycl3, which is what "
                          "makes the free-ion control a control"),
        # --- new: the named target and its actinide sibling ----------------------------
        Campaign(
            key="dycl3", label="DyCl3",
            atoms=_planar_mx3("Dy", "Cl", 2.45), charge=0, spin=5,
            basis="x2c-SVPall-2c", n_active=14, n_active_elec=9,
            character=("Dy", "f"), n_states=16, caps=CAPS_F_SHELL, n_det=2002,
            casscf_mode="second-order", max_iter=250,
            scf_prelude=dict(diis="adiis", max_cycle=200),
            scf_options=dict(second_order=True, stability="follow", max_cycle=150),
            geom_note="planar D3h model, r(Dy-Cl) = 2.45 A, C3 along z. A MODEL "
                      "structure in the tradition of the committed cecl3/ticl3/tif3 "
                      "entries, not an optimised or measured one: the bond length is the "
                      "gas-phase value reported for the lanthanide trichloride series and "
                      "is stated so an independent implementation can define the same "
                      "calculation, which is all a validation system needs of it",
            physics_note="THE named cost target. Dy(3+) 4f^9, ground 6H15/2, split by the "
                         "D3h ligand field into 8 Kramers doublets. 2002 determinants, so "
                         "an exact CI oracle is cheap at every ladder point; the free-ion "
                         "analytic Lande g = 4/3 survives as the barycentre statement while "
                         "the per-doublet g values are the ligand field's own. "
                         "4f is the lowest f shell, so the selection needs no ordinal "
                         "window",
            protocol_note="⚠ THE SCF IS A TWO-STAGE RECIPE AND THE SECOND STAGE ALONE IS "
                          "WRONG. Every first-order recipe tried (plain/ADIIS DIIS, level "
                          "shift, damping, atom guess, a 2.0 Eh shift) oscillates without "
                          "converging in 200-400 cycles; the second-order CIAH solver "
                          "converges from the default guess, reports stable=True, and "
                          "lands on a solution 0.665 Eh ABOVE the ground SCF state. Three "
                          "independent routes agree on that wrong answer. Seeding CIAH "
                          "with 200 cycles of ADIIS instead reaches -13535.40442 Eh, "
                          "stable, in 39 s. An open 4f shell has many stationary points, "
                          "CIAH converges to the one nearest its guess, and no check in "
                          "the run says which one that was. "
                          "SA-16 over the complete 6H15/2 manifold. The boundary to 6H13/2 "
                          "is thousands of cm^-1, and BOTH boundary diagnostics plus the "
                          "spin-invariance figure are recorded: a complete 2J+1 manifold "
                          "is a clean boundary but not an ensemble the term's symmetry "
                          "leaves invariant, so the average leans and is protected by its "
                          "gap rather than by construction"),
        Campaign(
            key="uf3", label="UF3",
            atoms=_pyramidal_mx3("U", "F", 2.06, 108.0), charge=0, spin=3,
            basis={"U": "cc-pVDZ-X2C", "F": "x2c-SVPall-2c"},
            n_active=14, n_active_elec=3,
            selection_kw=dict(character=[([0], "f", 14, U_F_SKIP_PAIRS)],
                              n_active=14, n_active_elec=3),
            n_states=10, caps=CAPS_F_SHELL, n_det=364,
            orbitals="guess", casscf_mode="second-order", max_iter=250,
            scf_prelude=dict(level_shift=2.0, damp=0.7, diis="adiis", max_cycle=200),
            scf_options=dict(second_order=True, stability="follow", max_cycle=150),
            geom_note="pyramidal C3v model, r(U-F) = 2.06 A, F-U-F = 108 deg, C3 along z. "
                      "A MODEL structure stated for reproducibility, as the trihalide "
                      "entries above are; the pyramidal form is the one reported for the "
                      "gas-phase actinide trifluorides",
            physics_note="the actinide leg: U(3+) 5f^3, ground 4I9/2 split by the ligand "
                         "field into 5 Kramers doublets. 364 determinants — the cheapest "
                         "f-block oracle in the campaign. ⚠ THE ORDINAL WINDOW IS "
                         "REQUIRED HERE AND ITS ABSENCE WAS A REAL DEFECT: uranium has a "
                         "FILLED 4f shell below the 5f, so the bare (U, f) form this entry "
                         "carried until 2026-09-02 selected the 4f core — fourteen "
                         "electrons' worth of doubly occupied orbitals, opened with three "
                         "electrons in them while the singly occupied 5f pairs were frozen "
                         "as inactive. Measured on this reference: f-character pairs 26-32 "
                         "are 4f (population 1.000 on U f, occupation 2.0) and pairs 8-14 "
                         "of the character ordering are the 5f (0.44-0.99, occupation 1.0 "
                         "on the lowest two). Nothing in the result looked wrong — an f^3 "
                         "shell gives the same 4I9/2 pattern and the same Lande g whichever "
                         "f shell holds it, which is exactly the silent failure the ordinal "
                         "window exists to prevent. Every uf3 record written before that "
                         "date is on the 4f space and says so in its own active_space "
                         "field",
            protocol_note="⚠ THE LADDER RUNS AT SCALAR-GUESS ORBITALS, and it is a "
                          "recorded deviation rather than a shortcut: the reference "
                          "SA-10 CASSCF DOES NOT CONVERGE HERE. Four step engines were "
                          "tried (second-order, auto, and each with a shortened maximum "
                          "step, up to 52 minutes per attempt) and every one ends the "
                          "same way — a TRIAL orbital step produces active-space "
                          "integrals that are no longer time-reversal closed, every "
                          "Kramers pair of this odd-electron spectrum splits by more "
                          "than the 1e-6 Eh degeneracy tolerance, and the "
                          "state-averaging gate refuses inside the trial evaluation, "
                          "which ends the optimization instead of rejecting the step. "
                          "The refusal is CORRECT and is not worked around: a "
                          "state-averaged density built on a split pair depends on an "
                          "arbitrary rotation inside it and nothing downstream recovers "
                          "that. What the ladder needs is a fixed, reproducible, "
                          "well-posed set of orbitals, and the guess is one: its CASCI "
                          "gives the 5 Kramers doublets of 4I9/2 at 0, 888.76, 2495.08, "
                          "2792.70 and 3189.74 cm^-1, pair splittings below 8e-7 cm^-1, "
                          "and a 48307 cm^-1 gap to the next manifold. Every quantity "
                          "graded here is a difference at those same orbitals, so the "
                          "truncation measurement stands; what it does not carry is a "
                          "statement about CASSCF-quality ligand-field splittings for "
                          "this system. The unconverged CASSCF is a finding in its own "
                          "right and is not closed by this choice. "
                          "⚠ Two-stage SCF like dycl3 but with a DIFFERENT PRELUDE, and "
                          "the difference is measured rather than preferred: here the "
                          "first-order iteration is not oscillating but CREEPING — it "
                          "reaches the second-order solver's answer to 5.6e-9 Eh and "
                          "cannot close the last decade to a 1e-10 tolerance in 200 "
                          "cycles — so an ADIIS seed lands in the same basin CIAH finds "
                          "on its own, while a heavily shifted seed reaches a solution "
                          "1.6e-4 Eh (35 cm^-1) BELOW it. Both are stable; the lower one "
                          "is used. 35 cm^-1 is small against a total energy and not "
                          "small against a ligand-field manifold a few hundred cm^-1 "
                          "wide, which is why the search was run rather than assumed. "
                          "BASIS IS PER ATOM and it has to be: the Peterson X2C "
                          "correlation-consistent sets cover the heavy elements only, so U "
                          "takes cc-pVDZ-X2C (the Karlsruhe x2c-...all-2c series stops at "
                          "Rn) and F takes the project default. Both are X2C "
                          "recontractions, which is the compatibility the front end "
                          "checks. SA-10 over the complete 4I9/2 manifold"),
    ):
        out[camp.key] = camp
    for camp in _beyond_minimal(out):
        out[camp.key] = camp
    return out


def _beyond_minimal(base: Dict[str, Campaign]) -> List[Campaign]:
    """The "beyond the minimal active space" leg: same molecules, a larger CAS.

    ⚠ **``mode="second-order"`` on both, and it is the orbital problem that decides it**
    (never the CI cost): a correlating shell is nearly empty in every state of the average,
    so the rotations that determine it are the weakly curved ones a quasi-Newton step is
    worst at. ``max_iter`` is doubled with it, because an escalation the budget does not
    survive returns an unconverged iterate rather than a slow answer.

    ⚠ **Derived from the minimal entries rather than restated**, so the geometry, the basis,
    the charge, the spin and the SCF recipe keep exactly one definition and only the active
    space differs — which is the whole point of the comparison. What each one adds is the
    *correlating* shell, and naming it is the trap this leg exists to walk into carefully:
    a second shell of the same ``l`` cannot be selected by character alone, because the
    lowest pairs of that character are the shell already in the space (and, on an actinide,
    three filled core shells below it). Both spaces therefore state an **ordinal window**
    over the character-ordered list, which is reproducible from what it prints.
    """
    out: List[Campaign] = []
    ti = base["tif3"]
    out.append(replace(
        ti, key="tif3_dd", label="TiF3 (3d + 4d' double shell)",
        n_active=20, n_det=20, caps=CAPS_SMALL, ladder_states=ti.roots,
        casscf_mode="second-order", max_iter=120,
        selection_kw=dict(character=[([0], "d", 10), ([0], "d", 10, 5)],
                          n_active=20, n_active_elec=ti.n_active_elec),
        physics_note="the double-shell extension of the committed d^1 system: the five "
                     "lowest d pairs plus the next five. 20 determinants, so the exact CI "
                     "oracle is free at every cap, and the manifold under test is the same "
                     "10-state one the minimal ladder graded - which is what makes the two "
                     "ladders comparable at all",
        protocol_note="committed suite geometry, basis and SCF; ACTIVE SPACE EXTENDED. "
                      "The correlating shell is named as an ordinal window "
                      "[(Ti,d,10), (Ti,d,10,skip=5)] because two (Ti,d) fragments without "
                      "one claim the same pairs and are refused. Measured on this "
                      "reference: the window takes d-character pairs 6-10, whose "
                      "populations are 0.70-0.98 on Ti d"))
    fe = base["fecl2"]
    out.append(replace(
        fe, key="fecl2_dd", label="FeCl2 (3d + 4d' double shell)",
        n_active=20, n_det=38760, caps=CAPS_SMALL, ladder_states=fe.roots,
        casscf_mode="second-order", max_iter=120,
        selection_kw=dict(character=[([0], "d", 10), ([0], "d", 10, 5)],
                          n_active=20, n_active_elec=fe.n_active_elec),
        physics_note="the flagship beyond-minimal space of this stage: CAS(6, 20 spinors), "
                     "38 760 determinants over a 25-state manifold — well past the "
                     "comfort zone of the 10-spinor ladders and still an exact oracle. It "
                     "is also the only beyond-minimal space in the campaign on an EVEN "
                     "electron count, so its protocol-B leg is immune to the "
                     "state-averaging gate by construction — the gate is a theorem about "
                     "ODD counts — rather than by the measurement an odd-count system "
                     "needs",
        protocol_note="committed suite geometry, basis and SCF; ACTIVE SPACE EXTENDED by "
                      "the same ordinal-window form the titanium double shell uses. "
                      "⚠ It replaced the planned uranium 5f+6d extension, which is BLOCKED "
                      "BY MEASUREMENT rather than by cost: on the uf3 reference the 6d and "
                      "5f fragments are not disjoint — pair 68 is 40 % U d and 48 % U f "
                      "and clears both thresholds — so the union form refuses, correctly, "
                      "and no window separates them. Naming that space needs AVAS (a "
                      "projection onto free-atom reference orbitals), which is a different "
                      "statement requiring atomic_reference=True at ingestion and is not "
                      "smuggled into a stage as an afterthought"))
    dy = base["dycl3"]
    out.append(replace(
        dy, key="dycl3_a", label="DyCl3 rung A (4f + the Cl sigma-donor set)",
        n_active=20, n_active_elec=15, n_det=15504, caps=CAPS_F_SHELL,
        ladder_states=dy.roots,
        selection_kw=dict(character=[([0], "f", 14),
                                     ([1, 2, 3], "p", 6, CL_2P_SKIP_PAIRS)],
                          n_active=20, n_active_elec=15),
        physics_note="the covalent rung: the 4f^9 shell plus the chlorine sigma-donor "
                     "set, CAS(15, 20 spinors), 15 504 determinants — an exact oracle that "
                     "is still affordable, on the campaign's named target and at its own "
                     "16-state manifold",
        protocol_note="⚠ THREE Cl PAIRS, NOT TWO, AND THE DIFFERENCE IS A DEGENERACY. The "
                      "plan asked for two sigma-donor pairs; measured on this reference the "
                      "three lowest valence p pairs on the chlorines are one a1' orbital "
                      "(population 0.842 on Cl p) and an e' PAIR (0.829 each, degenerate), "
                      "so a selection of two splits that pair — a symmetry-broken active "
                      "space, and the rule that any truncation takes whole degenerate "
                      "groups applies to an active-space selection as much as to a Schmidt "
                      "spectrum. Three pairs is the complete D3h donor set. "
                      "⚠ THE WINDOW IS REQUIRED: chlorine's filled 2p lies below its "
                      "valence 3p — measured, pairs 24-32 are 2p (population 1.000 on Cl p, "
                      "occupation 2.0) and pairs 47-55 are the 3p — so a bare (Cl, p) "
                      "selection takes nine core orbitals, and the occupations cannot tell "
                      "them apart because both shells are full. Orbitals and SCF recipe are "
                      "the minimal system's, unchanged"))
    out.append(replace(
        dy, key="dycl3_b", label="DyCl3 rung B (4f + 5d)",
        n_active=24, n_active_elec=9, n_det=1307504, caps=CAPS_F_SHELL,
        ladder_states=dy.roots,
        selection_kw=dict(character=[([0], "f", 14), ([0], "d", 10, DY_4D_SKIP_PAIRS)],
                          n_active=24, n_active_elec=9),
        exact_oracle=False, orbitals="guess",
        physics_note="the double-shell rung: 4f + 5d, CAS(9, 24 spinors). ⚠ 1 307 504 "
                     "determinants — the exact CI oracle is MARGINAL here by design, and "
                     "whether it exists at this machine's memory limit is decided by the "
                     "memory pre-flight for free. Where it refuses, the rung is validated "
                     "by "
                     "internal convergence in D against the rungs below it, which is the "
                     "Tier-3 methodology rehearsed one step from an oracle",
        protocol_note="⚠ THE WINDOW IS REQUIRED AND MEASURED: dysprosium's filled 3d "
                      "(pairs 12-16) and 4d (pairs 33-37) lie below the 5d, all at "
                      "population 1.000 on Dy d and occupation 2.0, so ten pairs are "
                      "skipped. What the window then takes is NOT a clean 5d shell and the "
                      "record says so: pairs 61-66 carry 0.38-0.88 of their population on "
                      "Dy d with the rest on the chlorines, because in a trihalide the 5d "
                      "is mixed into the ligand field. It is reproducible, which is what a "
                      "reference selection has to be. The SCF recipe is the minimal "
                      "system's, unchanged. "
                      "⚠ THE LADDER RUNS AT SCALAR-GUESS ORBITALS, and here that is forced "
                      "rather than chosen: a reference CASSCF for this space would be a "
                      "CONVENTIONAL CI over 1 307 504 determinants, which is the 11.3 GB "
                      "workspace the pre-flight refuses — the very solve this rung is "
                      "declared oracle-free to avoid. Running it under the overcommit that "
                      "lets the reference build would trade a diagnosed refusal for an OOM "
                      "kill. Every quantity graded here is a difference at the same fixed "
                      "orbitals, so the truncation measurement stands; what it does not "
                      "carry is a statement about CASSCF-quality splittings for this rung"))
    return out


def get(key: str) -> Campaign:
    all_ = systems()
    if key not in all_:
        raise KeyError("unknown campaign system {!r}; known: {}"
                       .format(key, sorted(all_)))
    return all_[key]


# --- the pipeline, one step per function ----------------------------------------------------
def build_reference(camp: Campaign):
    """Front end -> working basis -> spinor guess -> factorized integrals.

    ⚠ Two-electron spin-orbit screening is left at its default (the atomic mean field),
    because the multiplet structures this campaign grades have to be the physically
    meaningful ones. The per-element four-component solves are paid once into the
    persistent cache and are geometry-independent, so a whole ladder pays them zero times
    after the first system that needs them.
    """
    from kuiva.interface import api

    molecule = api.Molecule(atoms=camp.atoms, basis=camp.basis, charge=camp.charge,
                            spin=camp.spin)
    kw = dict(camp.scf_options)
    if camp.scf_prelude is not None:
        # ⚠ The prelude is allowed to stop unconverged BY DESIGN: its job is to leave the
        # orbitals in the right basin, not to solve anything. It runs without the
        # two-electron spin-orbit screening because a guess is orbitals and the screening
        # is applied after the SCF, so paying a four-component atomic solve for it would
        # be pure cost.
        warm = api.scalar_x2c_reference(molecule, screening="none",
                                        allow_unconverged_scf=True,
                                        **dict(camp.scf_prelude))
        kw["guess_from"] = warm
    if not camp.exact_oracle:
        # ⚠ An oracle-free rung: the front end's pre-flight plans the conventional CI for
        # ``n_states`` and refuses it (23 GB at 24 spinors), but that CI is exactly the
        # solve this rung is declared oracle-free to avoid and it never runs — the ladder
        # runs at guess orbitals. So the refusal is downgraded for the REFERENCE BUILD
        # ONLY, and the honest limit is restored before any network phase, which is where
        # the memory plan has to refuse rather than warn.
        from dataclasses import replace as _replace
        from kuiva.util import resources as res

        lims = res.BUDGET.limits
        res.BUDGET.configure(_replace(lims, allow_overcommit=True))
        try:
            return api.spinor_reference(molecule, n_active=camp.n_active,
                                        n_active_elec=camp.n_active_elec,
                                        n_states=camp.n_states, **kw)
        finally:
            res.BUDGET.configure(lims)
    return api.spinor_reference(molecule, n_active=camp.n_active,
                                n_active_elec=camp.n_active_elec,
                                n_states=camp.n_states, **kw)


def orbital_checkpoint(camp: Campaign) -> Path:
    return WORK / "{}_casscf.h5".format(camp.key)


def converged_orbitals(camp: Campaign, reference, *, report: bool = False,
                       max_iter: Optional[int] = None,
                       deadline: Optional[float] = None) -> Tuple[np.ndarray, object, Dict]:
    """The reference SA-CASSCF: ``(coeff, active_space, record)``.

    Checkpointed and restarted from that checkpoint, so a budget-interrupted campaign never
    repays an orbital optimization it has already done — the orbitals are the expensive part
    and every ladder point in the stage below reuses exactly these.

    ⚠ ``deadline`` (seconds of wall budget left) is handed to the optimizer rather than
    enforced from outside, because the stop has to be **predictive**: the run must end while
    there is still room to write its checkpoint, not when its time is already spent. An
    externally killed optimization leaves nothing behind and the whole thing is repaid.
    """
    from kuiva.interface import api

    WORK.mkdir(parents=True, exist_ok=True)
    ckpt = orbital_checkpoint(camp)
    kw = dict(n_states=camp.n_states, mode=camp.casscf_mode,
              max_iter=camp.max_iter if max_iter is None else int(max_iter),
              conv_grad=camp.conv_grad, checkpoint=str(ckpt), report=report)
    if deadline is not None and float(deadline) > 0.0:
        kw["deadline"] = float(deadline)
    t0, c0 = time.time(), time.process_time()
    if ckpt.is_file():
        outcome = api.casscf(reference, restart=str(ckpt), **kw)
    else:
        outcome = api.casscf(reference, **camp.selection(), **kw)
    rec = {"converged": bool(outcome.converged),
           "iterations": int(outcome.orbital.n_iterations),
           "grad_norm": float(outcome.orbital.grad_norm),
           "e_avg": float(outcome.energy),
           "active_space": outcome.active.description,
           "wall_s": round(time.time() - t0, 1),
           "cpu_s": round(time.process_time() - c0, 1),
           "state_rel_cm": rel_cm(np.asarray(outcome.ci.total_energies)),
           "boundary": _boundary(outcome.boundary),
           "boundary_initial": _boundary(outcome.boundary_initial),
           "n_hessian_matvec": int(outcome.orbital.n_hessian_matvec),
           "n_rejected": int(outcome.orbital.n_rejected)}
    return np.ascontiguousarray(outcome.coeff), outcome.active, rec


def _boundary(report) -> Optional[Dict]:
    """Both state-average diagnostics, recorded whatever they say.

    ⚠ A failure to measure is recorded as such and never as a clean result: "could not be
    measured" is a weaker statement than "is clean", and the check may never kill a run.
    """
    if report is None:
        return None
    return {"gap_cm": (None if report.gap_cm is None else round(float(report.gap_cm), 3)),
            "spans_full_ci": bool(report.spans_full_ci),
            "is_clean": bool(report.is_clean),
            "where": report.where,
            "spin_noninvariance": (None if report.spin_noninvariance is None
                                   else float(report.spin_noninvariance)),
            "leaning": report.leaning}


def cas_integrals(reference, coeff: np.ndarray, space):
    from kuiva.mcscf.orbopt import CASIntegrals
    return CASIntegrals.build(reference.factors, reference.h_one_electron(),
                              np.ascontiguousarray(coeff), space.spaces,
                              e_nuc=reference.data.e_nuc)


def rel_cm(e: Sequence[float], n: Optional[int] = None) -> List[float]:
    from kuiva.props.multiplet import HARTREE_TO_CM
    a = np.sort(np.asarray(e, dtype=float))
    if n is not None:
        a = a[:n]
    return [round(float(x), 5) for x in (a - a[0]) * HARTREE_TO_CM]


# --- the two solvers, behind one shape ------------------------------------------------------
def exact_ci(ints, camp: Campaign) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """The oracle: exact CI on these integrals. ``(energies, tdm, cost)``."""
    from kuiva.mcscf.casci import FullCISolver

    t0, c0 = time.time(), time.process_time()
    solver = FullCISolver(ints.spaces.n_active, camp.n_active_elec, n_states=camp.roots)
    result = solver.casci(ints)
    energies = np.asarray(result.total_energies, dtype=float)
    tdm = result.transition_densities()
    return energies, tdm, {"ndet": int(result.vectors.shape[1]),
                           "wall_s": round(time.time() - t0, 3),
                           "cpu_s": round(time.process_time() - c0, 3)}


# --- node partitions: the root count constrains the topology ---------------------------------
def two_site_floor(n: int, n_elec: int, n_modes: int) -> int:
    """Smallest dimension a two-site problem spanning ``n_modes`` modes can have.

    ⚠ **The constraint nobody expects, and it decides the topology before the physics
    does.** A two-site update solves for the whole ensemble inside the block of modes it
    covers, so a bond whose two-site space is smaller than the root count cannot represent
    the ensemble at all — and the sweep refuses rather than silently averaging over a
    smaller one. That space is bounded by the charge sectors the block can reach, which
    depends on the block's mode count and the electron number and **not on the bond
    dimension**: raising the cap does not open it.

    The floor is the state's first (smallest) allocation, one dimension per charge sector,
    so this is ``sum_q C(n_modes, q)`` over the electron numbers ``q`` the block may hold
    given that the remaining ``n - n_modes`` modes must hold the rest. Reproduced exactly by
    the solver's own refusals on two systems, which is why it is used here instead of a
    trial solve.
    """
    lo = max(0, int(n_elec) - (int(n) - int(n_modes)))
    hi = min(int(n_modes), int(n_elec))
    return sum(math.comb(int(n_modes), q) for q in range(lo, hi + 1)) if hi >= lo else 0


def _arrange(sizes: Sequence[int]) -> List[int]:
    """Order node sizes so the SMALLEST adjacent pair is as large as possible.

    Two small nodes side by side make the tightest bond in the network, so the small ones
    go to the ends and the large ones to the middle: filling positions from the outside in,
    smallest first, does exactly that. It is the difference between a partition that works
    and one that refuses at the same node count.
    """
    asc = sorted(int(x) for x in sizes)
    k = len(asc)
    out = [0] * k
    lo, hi = 0, k - 1
    for i, size in enumerate(asc):
        if i % 2 == 0:
            out[lo] = size
            lo += 1
        else:
            out[hi] = size
            hi -= 1
    return out


def node_partition(n: int, n_elec: int, n_roots: int) -> List[int]:
    """The FINEST node partition of ``n`` modes whose every bond can carry ``n_roots``.

    Finest, because a coarse network is a dense diagonalization wearing a tensor network's
    clothes: the whole point of the ladder is that the bond dimension binds, and it cannot
    bind on a two-node graph whose single bond spans the full space. Searched from
    single-mode nodes upward and the first partition that satisfies
    :func:`two_site_floor` on every adjacent pair wins.

    ⚠ It can come back as two nodes, and that is an answer rather than a failure: a
    one-electron active space averaged over its *entire* determinant space has nowhere to
    put an ensemble except a bond that spans everything.
    """
    for k in range(n // 2, 1, -1):                    # finest first
        base, extra = divmod(n, k)
        if base < 1:
            continue
        sizes = _arrange([base + 1] * extra + [base] * (k - extra))
        worst = min(a + b for a, b in zip(sizes, sizes[1:]))
        if two_site_floor(n, n_elec, worst) >= n_roots:
            return sizes
    return _arrange([n - n // 2, n // 2])


def _contents(order: Sequence[int], sizes: Sequence[int]) -> List[Tuple[int, ...]]:
    """Consecutive slices of ``order`` of the given sizes — one tuple of modes per node.

    ⚠ A node's modes need not be consecutive in the global mode index and it does not
    matter that they are not: the Jordan-Wigner order is the global ascending mode index
    everywhere in this program, so the operator carries the phases whatever the nodes hold.
    """
    o = [int(x) for x in order]
    blocks, at = [], 0
    for size in sizes:
        blocks.append(tuple(o[at:at + size]))
        at += size
    return blocks


def path_graph(order: Sequence[int], sizes: Sequence[int]):
    from kuiva.dmrg import NetworkGraph
    contents = _contents(order, sizes)
    k = len(contents)
    return NetworkGraph(k, [(i, i + 1) for i in range(k - 1)], contents)


def mi_tree_graph(info: np.ndarray, order: Sequence[int], sizes: Sequence[int]):
    """A mutual-information tree over the SAME node blocks the chain uses.

    The tree is built on the coarsened information matrix — block-to-block sums of the
    orbital mutual information — so the two topologies differ in their *edges* and in
    nothing else. Comparing a fine tree against a coarse chain would compare node fatness,
    which is a different experiment.
    """
    from kuiva.dmrg import NetworkGraph, topology_from_mutual_information

    contents = _contents(order, sizes)
    k = len(contents)
    coarse = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            coarse[a, b] = float(np.asarray(info)[np.ix_(list(contents[a]),
                                                         list(contents[b]))].sum())
    guess = topology_from_mutual_information(coarse)
    return NetworkGraph(k, [tuple(int(x) for x in e) for e in guess.graph.edges], contents)


def topologies(ints, camp: "Campaign", which: Sequence[str]) -> Tuple[Dict[str, object],
                                                                     Dict]:
    """The stated topologies, built from a cheap CI's entanglement — the production route.

    ⚠ Topology *discovery* is not load-bearing anywhere in this campaign: both topologies
    are stated before the ladder runs and neither moves during it. What the cheap CI
    supplies is the mode ordering (a Fiedler chain) and the mutual information the tree is
    grown from, which is what the class API hands a production tensor-network CASSCF.
    """
    from kuiva.mcscf.preopt import cheap_ci
    from kuiva.rdm.entropy import fiedler_order

    n = int(ints.spaces.n_active)
    sizes = node_partition(n, camp.n_active_elec, camp.roots)
    meta = {"node_sizes": sizes, "n_nodes": len(sizes),
            "two_site_floor": two_site_floor(n, camp.n_active_elec,
                                             min(a + b for a, b in zip(sizes, sizes[1:]))),
            "roots": camp.roots}
    made: Dict[str, object] = {}
    if not which:
        return made, meta
    ci = cheap_ci(np.ascontiguousarray(ints.h_active_effective()),
                  np.ascontiguousarray(ints.active_eri()), camp.n_active_elec,
                  n_states=min(camp.roots, 8), with_2rdm=False)
    info = ci.entanglement()[1]
    order = [int(x) for x in fiedler_order(info)]
    meta["fiedler_order"] = order
    for name in which:
        if name == "path":
            made[name] = path_graph(order, sizes)
        elif name == "tree":
            made[name] = mi_tree_graph(info, order, sizes)
        elif name == "path-ascending":
            made[name] = path_graph(range(n), sizes)
        else:
            raise ValueError("unknown topology {!r}".format(name))
    return made, meta


#: Compiled TTNO templates by topology, shared across every cap of a ladder (and every
#: variant of a control run) on that topology. ⚠ Two reasons, both measured. Cost: the
#: compile depends on the graph alone, and paying it per point put ~100 of `tif3_dd`'s
#: 105 CPU s per point into the compile a production run pays once — the per-point CPU
#: figure is taken AFTER the template exists, and the compile's own cost is recorded
#: beside it (``compile_cpu_s``). Memory: on a tree with a degree-4 node the compile's
#: transition tables are a ~4.5 GB transient that the allocator keeps, so two compiles
#: back to back — the residue of one point plus the next point's — exceeded the machine
#: while every plan said 1.3 GB. Cleared with the ledger at the start of every job.
_TEMPLATE_CACHE: Dict[object, Tuple[object, Dict]] = {}


def compiled_template(graph) -> Tuple[object, Dict]:
    """``(template, compile_cost)`` for ``graph``, compiled once per job and reused."""
    from kuiva.dmrg.ttno import TTNOTemplate

    hit = _TEMPLATE_CACHE.get(graph)
    if hit is not None:
        return hit
    t0, c0 = time.time(), time.process_time()
    template = TTNOTemplate(graph)
    cost = {"compile_wall_s": round(time.time() - t0, 3),
            "compile_cpu_s": round(time.process_time() - c0, 3)}
    _TEMPLATE_CACHE[graph] = (template, cost)
    return template, cost


def clear_templates() -> None:
    """Forget every compiled template — with ``resources.clear()``, at a job boundary."""
    _TEMPLATE_CACHE.clear()


def network_ci(ints, camp: Campaign, *, max_bond: int, graph,
               max_sweeps: int = 30, conv_tol: float = 1e-9,
               davidson_tol: float = 1e-8, adaptive: bool = False,
               rule: Optional[str] = None,
               **solver_kwargs) -> Tuple[Optional[np.ndarray],
                                         Optional[np.ndarray], Dict]:
    """One ladder point: a network CASCI at fixed orbitals. ``(energies, tdm, meta)``.

    ⚠ Both failure modes come back as *records*, not exceptions. A ``ValueError`` from the
    group-complete truncation rule is a **refusal** — the cap cannot be honoured without
    cutting through a degenerate Schmidt group — and which caps are refused is itself part
    of the campaign's answer. A ``SolverFailure`` is a sweep that did not converge in the
    fixed budget, which is a cost statement rather than a correctness one.

    ⚠ **``on_split="warn"`` here, and ONLY here.** The state-averaging gate refuses when a
    truncated spectrum splits a Kramers pair by more than its degeneracy tolerance, because
    a state-averaged RDM built across a split pair depends on an arbitrary rotation inside
    it. That is exactly right where the averaged RDM is *used* — in a CASSCF, where it
    moves the orbitals — and it stays ``"raise"`` there. This protocol never uses it: it
    grades per-root energies and transition densities at FIXED orbitals, reduced to block
    invariants summed over whole degenerate blocks of the oracle, which is invariant to
    precisely the rotation the gate protects against. Leaving it at ``"raise"`` would refuse
    every truncating cap of an odd-electron system — truncation error and Kramers splitting
    being the same quantity, roughly a factor of two apart — and so refuse to measure the
    approximation regime this whole campaign exists to measure. The refusals that remain
    (the group-complete truncation rule) are untouched and still recorded.
    """
    from kuiva.dmrg import DMRGSolver, ReconnectionPolicy
    from kuiva.dmrg.solver import SolverFailure
    from kuiva.util import resources as res

    # ⚠ ``solver_kwargs`` are protocol C's variants (a per-sweep bond ramp, deterministic
    # subspace expansion). They are iteration strategies INSIDE the cap, never ways to
    # exceed it — the solver refuses a schedule whose last rung is not the cap — so a
    # variant point is comparable with the plain one at the same D by construction.
    kw: Dict[str, object] = dict(solver_kwargs)
    if rule is not None:
        kw["policy"] = ReconnectionPolicy(rule=rule)
    # ⚠ ``rdms=False``: a ladder point is a fixed-orbital CASCI whose products are the
    # energies and the transition densities below; the state-averaged (gamma, Gamma) an
    # optimizer would need are never read here, and their extraction forms every node's
    # dE/dW — dense in the local dimension squared, one operator leg per neighbour — which
    # the memory plan refuses on every tree and on the double shells while the sweep fits.
    solver = DMRGSolver(camp.n_active_elec, max_bond=int(max_bond), n_roots=camp.roots,
                        graph=graph, adaptive=bool(adaptive), max_sweeps=max_sweeps,
                        conv_tol=conv_tol, davidson_tol=davidson_tol,
                        on_split="warn", rdms=False, **kw)
    # ⚠ The template is compiled once per topology and injected (see _TEMPLATE_CACHE);
    # the point's clock starts after it exists, so ``cpu_s`` is the solve and nothing
    # a production run would pay once. An adaptive point may still compile for the
    # topologies it proposes; that cost stays in its figure, as it should.
    template, compile_cost = compiled_template(graph)
    solver._templates[graph] = template
    t0, c0 = time.time(), time.process_time()
    try:
        solver.solve(ints)
    except ValueError as exc:
        return None, None, {"status": "refused", "error": "{}".format(exc),
                            "wall_s": round(time.time() - t0, 3),
                            "cpu_s": round(time.process_time() - c0, 3)}
    except SolverFailure as exc:
        return None, None, {"status": "unconverged", "error": "{}".format(exc),
                            "wall_s": round(time.time() - t0, 3),
                            "cpu_s": round(time.process_time() - c0, 3)}
    except res.MemoryLimitError as exc:
        # ⚠ The third outcome, and a result rather than a crash: the memory plan refused
        # the cap before allocating (which is what the plan exists for — before it, this
        # was an OOM kill with no record). Which caps fit the machine is campaign data,
        # and a refusal must not abandon the remaining caps and topologies of the job.
        return None, None, {"status": "refused-memory",
                            "error": "{}".format(exc).splitlines()[0],
                            "wall_s": round(time.time() - t0, 3),
                            "cpu_s": round(time.process_time() - c0, 3)}
    r = solver.last
    energies = np.asarray(r.energies, dtype=float) + float(ints.e_core)
    # ⚠ The Kramers spread is REPORTED, never gated (see the note above): for an odd-electron
    # system it is the truncation error wearing a different hat, and the degeneracy metric
    # tests it against that relation rather than against zero.
    #
    # ⚠ It is a statement about an ODD electron count and about nothing else — Kramers'
    # theorem is what makes consecutive roots a pair — and an even count reports **None**
    # rather than a number. The first draft paired consecutive roots unconditionally and
    # broadcast a 13-element slice against a 12-element one the first time it met an odd
    # ROOT count (FeCl2, 25 states), which is a different quantity entirely and had nothing
    # to do with pairing.
    n_pair = int(energies.size) // 2
    pair_spread = (float(np.max(energies[1:2 * n_pair:2] - energies[0:2 * n_pair:2]))
                   if (camp.n_active_elec % 2 == 1 and n_pair) else None)
    tdm = solver.transition_densities()
    meta = {"status": "ok", "n_sweeps": int(r.n_sweeps),
            "kramers_pair_spread_eh": pair_spread, **compile_cost,
            "bond_used": int(r.max_bond_dim),
            "saturating": bool(int(r.max_bond_dim) < int(max_bond)),
            "w_disc": float(r.max_discarded),
            "wall_s": round(time.time() - t0, 3),
            "cpu_s": round(time.process_time() - c0, 3)}
    return energies, tdm, meta


# --- the phase-invariant reduction ----------------------------------------------------------
def property_matrices(reference, coeff: np.ndarray, space, tdm: np.ndarray,
                      energies: np.ndarray, camp: Campaign):
    """``H`` and ``mu`` in the state basis — the SAME assembly on both sides.

    That it is the same call for the exact CI and for the network is the point: the two
    differ in where the CI vectors came from and in nothing else downstream.
    """
    from kuiva.props.dump import property_matrices as _matrices

    provenance: Dict[str, object] = {
        "active_space": space.description,
        "n_active_spinors": int(space.n_active),
        "n_active_electrons": int(camp.n_active_elec),
        "n_states": int(np.size(energies)),
        "campaign": "dmrg cost ladder",
    }
    if reference.data.soc is not None:
        provenance["hamiltonian"] = reference.data.soc.provenance()
    return _matrices(np.ascontiguousarray(coeff), space.spaces, tdm, energies,
                     reference.data.properties, reference.data.s_ao,
                     provenance=provenance, active_space=space.description)


def oracle_blocks(matrices, camp: Campaign) -> List[Tuple[int, int]]:
    """``(start, size)`` of every degenerate block of the ORACLE spectrum.

    ⚠ These boundaries are then imposed on the truncated spectra too (module docstring).
    For the integer-spin system the ground block is two tunnelling-split singlets rather
    than a Kramers doublet, and pairing them is the explicit physical claim its campaign
    record carries — never an inference from their energies.
    """
    multiplets = matrices.analyse(tol_cm=BLOCK_TOL_CM,
                                  pseudo_doublet_tol_cm=camp.pseudo_doublet_tol_cm)
    return [(int(m.start), int(m.size)) for m in multiplets]


def own_pattern(matrices) -> List[int]:
    """The block sizes the trial spectrum produces on its OWN, with no boundary imposed.

    The degeneracy-pattern metric needs this and the imposed-boundary reduction cannot
    supply it: imposing the oracle's blocks makes every size agree by construction. ⚠ A
    disagreement here is not automatically a defect — a truncated sweep splits a Kramers
    pair by roughly twice the energy error it is already accepting, and the grading tests
    that relation before calling a split pattern a changed one.
    """
    from kuiva.props.multiplet import degenerate_blocks, HARTREE_TO_CM

    e = np.sort(np.asarray(matrices.energies, dtype=float))
    e_cm = (e - e[0]) * HARTREE_TO_CM
    return [int(n) for _, n in degenerate_blocks(e_cm, tol_cm=BLOCK_TOL_CM)]


def reduce_at(matrices, blocks: Sequence[Tuple[int, int]]) -> Dict:
    """The phase-invariant description of one spectrum at GIVEN block boundaries.

    Per block: the mean relative energy, the internal spread (which is what a truncation
    inflates), and the principal g values of the block invariant ``Tr_block(mu_i mu_j)``.
    ⚠ A block whose invariant carries no moment reports **nothing, never zero** — a single
    state has no magnetic moment, and reporting a zero would make an integer-spin singlet
    look like an isotropic doublet.
    """
    from kuiva.props.multiplet import (HARTREE_TO_CM, block_moment_tensor,
                                       multiplet_g_values)

    e = np.asarray(matrices.energies, dtype=float)
    order = np.argsort(e)
    e_cm = (e[order] - e[order][0]) * HARTREE_TO_CM
    mu = np.asarray(matrices.mu)[:, order, :][:, :, order]
    rows: List[Dict] = []
    for start, size in blocks:
        blk = e_cm[start:start + size]
        m = block_moment_tensor(mu, start, size)
        g = multiplet_g_values(m, size)
        rows.append({"start": int(start), "size": int(size),
                     "energy_cm": round(float(np.mean(blk)), 5),
                     "spread_cm": round(float(blk.max() - blk.min()), 6),
                     "g": ([round(float(x), 6) for x in g] if g else None),
                     "character": (axial_character(g) if g else None)})
    return {"levels_cm": [round(float(x), 5) for x in e_cm], "blocks": rows,
            "own_pattern": own_pattern(matrices)}


#: How far past the truncation-spread relation a split degenerate block may sit before the
#: pattern counts as *changed* rather than as the truncation error announcing itself. The
#: local validation record measures spread/dE_SA between 0.67 and 2.0 over five orders of
#: magnitude and on two systems; this is an order of magnitude of headroom on top.
SPREAD_RELATION_SLACK = 5.0


def grade(ref: Dict, trial: Dict, *, e_sa_error_cm: Optional[float] = None,
          bands: Bands = BANDS) -> Dict:
    """Grade a truncated reduction against the oracle's, metric by metric.

    ``e_sa_error_cm`` is the state-averaged energy error against the exact CI on the same
    integrals, and it is what makes the degeneracy-pattern metric honest: a truncated sweep
    splits a degenerate block by roughly twice that, so a split pattern is graded against
    the relation rather than against zero.

    Returns the per-metric tier, the deviations behind it, and the overall grade — which is
    the **worst** metric, because a tier is a claim about the whole answer.
    """
    rb, tb = ref["blocks"], trial["blocks"]
    n = min(len(rb), len(tb))
    energy_tiers: List[str] = []
    g_tiers: List[str] = []
    energy_dev: List[float] = []
    g_dev: List[float] = []
    notes: List[str] = []

    # --- multiplet relative energies, per splitting, percentage with an absolute floor ---
    for k in range(n):
        e_ref, e_tri = rb[k]["energy_cm"], tb[k]["energy_cm"]
        d = abs(e_tri - e_ref)
        energy_dev.append(round(d, 5))
        scale = abs(e_ref)
        if d <= max(bands.energy_quant_cm, bands.energy_quant_rel * scale):
            energy_tiers.append("quantitative")
        elif d <= max(bands.energy_qual_cm, bands.energy_qual_rel * scale):
            energy_tiers.append("qualitative")
        else:
            energy_tiers.append("unacceptable")

    # --- level ordering, with the near-degeneracy exemption -----------------------------
    ordering_ok = True
    for k in range(n - 1):
        gap = rb[k + 1]["energy_cm"] - rb[k]["energy_cm"]
        if gap <= bands.ordering_exempt_cm:
            continue                     # a swap here is precision, graded on the positions
        if tb[k + 1]["energy_cm"] < tb[k]["energy_cm"]:
            ordering_ok = False
            notes.append("multiplets {} and {} reordered across a {:.2f} cm^-1 gap"
                         .format(k, k + 1, gap))

    # --- principal g values per block ----------------------------------------------------
    # ⚠ Normalized by the block's LARGEST reference g, never component by component: the
    # transverse g of a strongly axial doublet is zero by symmetry, and a component-wise
    # relative deviation on a number that should be zero is a division by noise. This is
    # also the reading that makes the metric right for a non-Kramers pseudo-doublet, whose
    # g_z is the largest principal value and whose other two ARE the residual.
    for k in range(n):
        gr, gt = rb[k]["g"], tb[k]["g"]
        if gr is None or gt is None:
            if (gr is None) != (gt is None):
                g_tiers.append("unacceptable")
                notes.append("block {} carries a moment on one side and none on the other"
                             .format(k))
            continue
        if len(gr) != len(gt):
            g_tiers.append("unacceptable")
            notes.append("block {} returns {} g values against {}"
                         .format(k, len(gt), len(gr)))
            continue
        g_max_ref = max(abs(x) for x in gr)
        if g_max_ref < bands.g_floor:
            # A block with no moment: the graded question is whether it still has none.
            if max(abs(x) for x in gt) < bands.g_floor:
                g_tiers.append("quantitative")
            else:
                g_tiers.append("unacceptable")
                notes.append("block {} acquires a moment (g_max {:.3g}) where the "
                             "reference has none ({:.3g})".format(
                                 k, max(abs(x) for x in gt), g_max_ref))
            continue
        rel = max(abs(b - a) for a, b in zip(gr, gt)) / g_max_ref
        g_dev.append(round(float(rel), 6))
        if rel <= bands.g_quant_rel:
            g_tiers.append("quantitative")
        elif rel <= bands.g_qual_rel:
            g_tiers.append("qualitative")
        else:
            g_tiers.append("unacceptable")
        # ⚠ Anisotropy character is only a statement about a block that HAS a moment.
        # Below the floor the principal values are noise about zero and their ordering
        # carries no physics, so "easy-axis became easy-plane" there would be a verdict on
        # rounding.
        if rb[k]["character"] != tb[k]["character"]:
            g_tiers.append("unacceptable")
            notes.append("block {} changes anisotropy character ({} -> {})".format(
                k, rb[k]["character"], tb[k]["character"]))

    # --- degeneracy pattern, tested against the truncation-spread relation ---------------
    pattern_same = ref["own_pattern"] == trial["own_pattern"]
    spread = max((b["spread_cm"] for b in tb), default=0.0)
    explained = None
    if e_sa_error_cm is not None:
        floor = 2.0 * abs(float(e_sa_error_cm))
        explained = bool(spread <= SPREAD_RELATION_SLACK * max(floor, 1e-12))
    pattern_ok = pattern_same or bool(explained)
    if not pattern_same:
        notes.append("degeneracy pattern differs: {} against {}{}".format(
            trial["own_pattern"], ref["own_pattern"],
            "" if explained is None
            else ("; explained by the truncation-spread relation"
                  if explained else "; NOT explained by the truncation-spread relation")))

    tiers = list(energy_tiers) + list(g_tiers)
    if not ordering_ok or not pattern_ok:
        tiers.append("unacceptable")
    overall = worst_of(tiers) if tiers else "unacceptable"
    return {"overall": overall,
            "energy": worst_of(energy_tiers) if energy_tiers else None,
            "g": worst_of(g_tiers) if g_tiers else None,
            "ordering_preserved": bool(ordering_ok),
            "pattern_same": bool(pattern_same),
            "pattern_split_explained": explained,
            "max_energy_dev_cm": (max(energy_dev) if energy_dev else None),
            "max_g_rel_dev": (max(g_dev) if g_dev else None),
            "max_block_spread_cm": round(float(spread), 6),
            "notes": notes}


#: Blocks closer to isotropic than this are called isotropic rather than being sorted into
#: easy-axis or easy-plane by the ordering of a difference that is noise.
ISOTROPY_RTOL = 0.10


def axial_character(g: Sequence[float]) -> str:
    """``easy-axis`` / ``easy-plane`` / ``isotropic`` from the principal g values.

    The principal values come back ascending, so the question is whether the odd one out is
    the largest or the smallest. A crude label on purpose: it exists to catch the
    qualitative flip the grading calls unacceptable, not to describe the anisotropy.
    """
    a = sorted(float(x) for x in g)
    if len(a) < 3:
        return "undefined"
    lo, mid, hi = a[0], a[1], a[-1]
    if (hi - lo) <= ISOTROPY_RTOL * max(abs(hi), 1e-9):
        return "isotropic"
    return "easy-axis" if (hi - mid) > (mid - lo) else "easy-plane"


__all__ = ["BANDS", "BLOCK_TOL_CM", "Bands", "Campaign", "RECORDS", "TIERS", "WORK",
           "build_reference", "cas_integrals", "converged_orbitals", "exact_ci", "get",
           "grade", "network_ci", "oracle_blocks", "orbital_checkpoint", "own_pattern",
           "mi_tree_graph", "node_partition", "path_graph", "property_matrices",
           "axial_character", "reduce_at", "rel_cm", "relax_orbitals", "network_solver", "systems",
           "topologies", "two_site_floor", "worst_of"]


# --- protocol B: the same orbital optimization, driven by two different CI solvers ----------
#: What a **stopped** optimization looks like from the outside, as against a slow one: a
#: collapsed trust radius, a flat energy, and a gradient still an order of magnitude outside
#: its target, for this many consecutive macro-iterations. All three, because any one of them
#: alone has a healthy reading — a quasi-Newton phase before the second-order escalation is
#: flat-energy and far from convergence for several iterations by design, and cutting it
#: there would truncate a legitimate optimization. Study-generator settings, and they do not
#: change a result: the orbitals at the stop are the ones the full iteration budget returns.
STALL_TRUST = 1.0e-4
STALL_DE = 1.0e-7
STALL_GRAD_FACTOR = 10.0
STALL_ITERATIONS = 8
def network_solver(camp: Campaign, *, max_bond: int, graph, max_sweeps: int = 30,
                   conv_tol: float = 1e-8, davidson_tol: float = 1e-8):
    """The DMRG solver as an orbital optimizer's ``ci_solver``, at a stated cap.

    ⚠ **``on_split`` is left at its default ``"raise"`` here, and that is the whole point of
    the protocol.** The fixed-orbital ladder overrides it because it never builds a
    state-averaged density that moves anything (module docstring); this protocol builds
    exactly that density and hands it to the optimizer, so the gate protecting it is the one
    that belongs in the calculation. A refusal at a truncating cap is therefore a *result* —
    it says the truncated ensemble is not one an orbital optimization may be built on.
    """
    from kuiva.dmrg import DMRGSolver

    return DMRGSolver(camp.n_active_elec, max_bond=int(max_bond), n_roots=camp.roots,
                      graph=graph, adaptive=False, max_sweeps=int(max_sweeps),
                      conv_tol=float(conv_tol), davidson_tol=float(davidson_tol))


def relax_orbitals(reference, coeff0: np.ndarray, space, ci_solver, *, mode: str = "auto",
                   max_iter: int = 60, conv_grad: float = 1e-4,
                   wall_budget: Optional[float] = None, driver: str = "trust-region",
                   report: bool = True) -> Tuple[Optional[object], Dict]:
    """One orbital optimization with a stated CI solver. ``(result, record)``.

    The **only** difference between protocol B's two legs is the ``ci_solver`` handed in —
    same driver, same starting orbitals, same step engine, same convergence criterion — which
    is what makes the comparison a statement about CI methods rather than about optimizer
    trajectories.

    ⚠ Three outcomes, and two of them are data rather than failures. A
    :class:`~kuiva.util.errors.StateAverageSplit` at the *first* solve is recorded as
    ``"state-average-split"``: the truncated spectrum's degenerate blocks come apart, and an
    optimizer has nothing to reject because the starting point is already refused. A
    ``ValueError`` from the group-complete truncation rule is ``"refused"`` — the cap cannot
    be honoured on this network at all. Both are recorded with the cap that produced them.

    ``wall_budget`` stops the optimization **from the inside**, through the driver's own
    callback, so a stopped run still returns everything it had reached; an externally killed
    one leaves nothing.

    ``driver="events"`` swaps the outer control for the event-gated one, which exists for a
    solver whose internal space moves with the orbitals — a truncating network being exactly
    that, since a warm-started sweep at a binding cap is only locally optimal inside its
    manifold and the energy it returns is therefore not a function of the rotation alone.
    Everything else about the leg is unchanged, so the two drivers are comparable.
    """
    from kuiva.mcscf.orbopt import optimize_orbitals
    from kuiva.util.errors import SolverFailure, StateAverageSplit

    if driver == "events":
        from kuiva.mcscf.events import optimize_orbitals_events as _drive
    elif driver == "trust-region":
        _drive = optimize_orbitals
    else:
        raise ValueError("unknown driver {!r}".format(driver))

    t0, c0 = time.time(), time.process_time()
    stopped = {"budget": False, "stall": 0}

    def callback(info):
        if wall_budget is not None and (time.time() - t0) > float(wall_budget):
            stopped["budget"] = True
            return False
        # ⚠ **A flat energy at a large gradient is a stopped optimization, and paying out
        # the iteration budget on it measures nothing.** At a binding cap every trial point
        # is rejected — the energy the truncated solver returns is not a function of the
        # rotation alone, so the quadratic model never predicts it — and the run then spends
        # a full CI solve per iteration to reproduce the same number. Measured on FeCl2 at
        # D = 4: every step 0.0000 from macro-iteration 8, the gradient frozen at 6.062, and
        # fifty more iterations of it. The stop is recorded with the iteration count, and
        # the returned orbitals are the ones the full budget would have returned.
        if (float(info.get("trust", 1.0)) < STALL_TRUST
                and abs(float(info["de"])) < STALL_DE
                and float(info["grad_norm"]) > STALL_GRAD_FACTOR * float(conv_grad)):
            stopped["stall"] += 1
            if stopped["stall"] >= STALL_ITERATIONS:
                return False
        else:
            stopped["stall"] = 0
        return None

    def cost() -> Dict:
        return {"wall_s": round(time.time() - t0, 1),
                "cpu_s": round(time.process_time() - c0, 1)}

    from kuiva.util import resources as res

    try:
        result = _drive(reference.factors, reference.h_one_electron(),
                        np.ascontiguousarray(coeff0), space.spaces, ci_solver,
                        e_nuc=reference.data.e_nuc, mode=mode,
                        max_iter=int(max_iter), conv_grad=float(conv_grad),
                        n_active_elec=space.n_elec, callback=callback, report=report)
    except StateAverageSplit as exc:
        rec = {"status": "state-average-split", "error": "{}".format(exc)}
        rec.update(cost())
        return None, rec
    except (ValueError, SolverFailure) as exc:
        rec = {"status": "refused", "error": "{}: {}".format(type(exc).__name__, exc)}
        rec.update(cost())
        return None, rec
    except res.MemoryLimitError as exc:
        # the memory plan refused a solve of this optimization (a diagnosed outcome)
        rec = {"status": "refused-memory", "error": "{}".format(exc).splitlines()[0]}
        rec.update(cost())
        return None, rec
    rec = {"status": "ok", "driver": driver, "converged": bool(result.converged),
           "stopped_on_wall_budget": bool(stopped["budget"]),
           "stopped_on_stall": bool(not result.converged and not stopped["budget"]
                                    and stopped["stall"] >= STALL_ITERATIONS),
           "iterations": int(result.n_iterations),
           "grad_norm": float(result.grad_norm),
           "e_avg": float(result.energy),
           "n_hessian_matvec": int(result.n_hessian_matvec),
           "n_rejected": int(result.n_rejected),
           "n_solver_failures": int(result.n_solver_failures),
           "history": [round(float(e), 10) for e in result.history]}
    for extra in ("event_stable", "n_adoptions", "n_proposals"):
        if hasattr(result, extra):
            rec[extra] = getattr(result, extra)
    rec.update(cost())
    return result, rec
