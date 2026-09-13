"""The cheap CI used as a measuring instrument: one probe of one candidate space.

What a probe is
---------------
:func:`probe` runs the pre-optimizer (:func:`kuiva.mcscf.preopt.preoptimize`) in a candidate
active space with **bounded** budgets and returns what the round loop reads: a spectrum, the
single-orbital entropies, the mutual information, the pre-optimized orbitals and what it
cost. Nothing about the pre-optimizer's method changes here -- this module is orchestration
and says so: fixed budgets, the Kramers repair, a root count chosen from the theoretical
floor, and the cost recorded.

⚠ **A probe's energies are qualitative and its spectrum is comparable only with another
probe's at the same budget.** A truncated CI in a truncated determinant space is not a
variational statement about anything; what it reproduces is the *structure* of the low
spectrum, and only when the two things compared were computed the same way. Never compare a
probe's spectrum with a CASSCF's, never across budgets, and never quote one.

⚠ **Two blindnesses are structural and no budget lifts them**, which is why the round loop
never prunes a target shell and never accepts a correlating shell on what it carries: the
empty members of a d manifold come back at occupations of order 1e-4, and a double shell is
empty at this level altogether. They are the ligand-field spectrum and the correlation the
space exists for, respectively, and an occupation or entropy criterion that drops them is
measuring the probe.

The root count
--------------
A probe solves the **floor plus a margin** roots (:func:`probe_roots`) -- enough that the
manifold boundary the proposal reads is inside the solved spectrum, rounded up to an even
count for an odd electron number (Kramers), and clamped to the determinant count, where
"the space is exhausted" is a complete answer rather than a reason to pad. The margin exists
because a boundary is only visible from the state *above* it. ⚠ The protocol's one
pre-optimization extends that count to a manifold boundary of the fixed-orbital spectrum
(:func:`kuiva.autocas.protocol._probe_count_on_a_boundary`), because an average that ends
inside a manifold splits it.

⚠ **And the probe AVERAGES over all of them, which is a decision about the handoff and was
measured rather than assumed.** The natural-looking alternative is to average over the floor
alone and treat the margin as witnesses -- which is what a witness root is downstream, one
the average does not use -- and it is measurably **worse**, because the orbitals a probe
leaves are a *starting guess* and a broad average is the robust one. Measured on TiCl3, whose
CAS(1, 10) the cheap CI diagonalizes exactly, at six macro-iterations:

* averaging the ten roots: the following two-state CASSCF starts 2.5 mEh above the plain SCF
  guess and converges;
* averaging the floor's two: the pre-optimization ends at ``|g| = 2.4`` and the same CASSCF
  starts **0.44 Eh** above, because a two-state average of a ``d^1`` shell pulls the orbitals
  hard onto one d orbital and a half-converged optimization leaves them somewhere bad.

A probe is not a production calculation, and what it owes the stage after it is orbitals that
are a good place to start rather than orbitals optimal for an average that has not been
proposed yet.

What it costs
-------------
One probe is one pre-optimization: a few macro-iterations of the shared orbital optimizer
with a selected-CI solver inside it. The budgets here are small on purpose (the ten-minute
rule for exploratory work binds the defaults of anything that runs several of these in a
row), they are explicit arguments rather than adaptive, and every probe's CPU seconds go
into the rounds table -- a selection protocol whose cost is invisible is one nobody can
bound.

References
----------
* The selected CI behind the probe is cited where it is implemented
  (:mod:`kuiva.mcscf.preopt`): B. Huron, J. P. Malrieu, P. Rancurel, J. Chem. Phys. 58, 5745
  (1973), doi:10.1063/1.1679199 and N. M. Tubman et al., J. Chem. Phys. 145, 044112 (2016),
  doi:10.1063/1.4955109.
* J. Rissler, R. M. Noack, S. R. White, Chem. Phys. 323, 519 (2006),
  doi:10.1016/j.chemphys.2005.10.018 -- the mutual information a bridge round ranks by,
  implemented in :mod:`kuiva.rdm.entropy`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..util import output as out
from ..util.logging import get_logger
from ..util.timing import timer

log = get_logger(__name__)

__all__ = ["DEFAULT_PROBE_DETERMINANTS", "DEFAULT_PROBE_MAX_ITER", "DEFAULT_ROOT_MARGIN",
           "ProbeBudget", "ProbeResult", "embed_determinants", "measure", "probe", "probe_roots",
           "report_probe"]

#: Macro-iterations one probe may spend. Small on purpose: a probe is asked whether a class
#: of orbitals *changes the spectrum*, and the occupations that answer converge far faster
#: than the energy does.
DEFAULT_PROBE_MAX_ITER = 8

#: Determinants one probe's selected CI may hold. The cost of a probe is set by this number
#: and not by the spinor count, so it is the knob a bigger candidate space is paid for with.
DEFAULT_PROBE_DETERMINANTS = 6000

#: Roots solved **above** the theoretical floor. A manifold boundary is only visible from the
#: state above it, and a floor that is not the boundary needs room to chain outward.
DEFAULT_ROOT_MARGIN = 8


@dataclass(frozen=True)
class ProbeBudget:
    """What one probe is allowed to spend -- fixed in advance, printed, never adaptive.

    ⚠ **Two probes are comparable only at the same budget**, so this object is what the round
    loop holds constant across a whole protocol. A round that needed a larger budget (a
    lanthanide bridge, where the exchange splitting is below what the default resolves) gets
    it as a *stated* change, and the rounds table shows the budget it was measured at.
    """

    max_iter: int = DEFAULT_PROBE_MAX_ITER
    max_determinants: int = DEFAULT_PROBE_DETERMINANTS
    space_policy: str = "event"
    mode: str = "quasi-newton"
    conv_grad: float = 1e-3
    margin: int = DEFAULT_ROOT_MARGIN

    def describe(self) -> str:
        return ("{} macro-iterations, {} determinants, space policy {!r}, {} roots above the "
                "floor".format(self.max_iter, self.max_determinants, self.space_policy,
                               self.margin))


@dataclass
class ProbeResult:
    """One probe: the spectrum it measured, the orbitals it left, and what it cost."""

    space: object                          #: the :class:`~kuiva.mcscf.casci.ActiveSpace` probed
    coeff: np.ndarray                      #: (2*nao, n) AO-basis spinors, **Kramers repaired**
    result: object                         #: the :class:`~kuiva.mcscf.preopt.PreoptResult`
    #: State energies relative to the lowest [cm^-1], ascending. ⚠ Qualitative: comparable
    #: with another probe at the same budget and with nothing else.
    spectrum_cm: np.ndarray
    entropy: np.ndarray                    #: (n_active,) single-orbital entropies
    mutual_information: np.ndarray         #: (n_active, n_active)
    occupations: np.ndarray                #: (n_active,) occupation OF each returned orbital
    n_roots: int
    n_determinants: int
    converged: bool
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    budget: ProbeBudget = field(default_factory=ProbeBudget)
    #: Worst Kramers partner deviation before the repair -- how far the cheap stage drifted.
    pairing_deviation: float = 0.0
    #: ``True`` for :func:`measure` -- the cheap CI at the orbitals it was given, with no
    #: pre-optimization. :attr:`result` is then the :class:`~kuiva.mcscf.preopt.CheapCIResult`
    #: rather than a :class:`~kuiva.mcscf.preopt.PreoptResult`.
    fixed_orbitals: bool = False

    @property
    def n_active(self) -> int:
        return int(np.size(self.space.spaces.active))

    @property
    def columns(self) -> np.ndarray:
        """Active spinor columns of :attr:`coeff` -- what an entropy index refers to."""
        return np.asarray(self.space.spaces.active, dtype=int)

    def pair_entropy(self) -> np.ndarray:
        """Single-orbital entropy per **Kramers pair** of the active space, in column order.

        ⚠ Pruning is a decision about pairs, never about spinors: a space is defined on
        spatial orbitals and a pair may not be split across a boundary. The two members of a
        pair carry the same entropy up to the probe's own noise (time reversal maps one onto
        the other), so the pair's value is their mean and the *spread* is a diagnostic of how
        far the truncated solve drifted off pairing rather than a second number to select on.
        """
        s1 = np.asarray(self.entropy, dtype=float).ravel()
        if s1.size % 2:
            raise ValueError("the active space has an odd number of spinors ({}), so it is "
                             "not Kramers paired".format(s1.size))
        return 0.5 * (s1[0::2] + s1[1::2])

    def pair_information(self) -> np.ndarray:
        """Mutual information summed over Kramers partners: ``(n_pair, n_pair)``.

        The quantity a bridge candidate is ranked by (how much it shares with a *shell*, which
        is a set of pairs), so the fold happens once here rather than at each consumer.
        """
        info = np.asarray(self.mutual_information, dtype=float)
        npair = info.shape[0] // 2
        blocks = info.reshape(npair, 2, npair, 2).sum(axis=(1, 3))
        return blocks

    def __repr__(self) -> str:
        return ("ProbeResult(CAS({}, {}), {} roots, {} determinants, {:.1f} s cpu)"
                .format(self.space.n_elec, self.n_active, self.n_roots, self.n_determinants,
                        self.cpu_seconds))


def probe_roots(floor: int, *, n_elec: int, n_active: int,
                margin: int = DEFAULT_ROOT_MARGIN) -> int:
    """How many roots a probe solves: ``floor + margin``, Kramers-even, clamped to the space.

    ⚠ **Clamped and not padded.** Asking a CI for more roots than the determinant space holds
    gives duplicated energies and zero vectors; "the space is exhausted" is a complete answer
    about a spectrum, so the count stops at the determinant count and the verdict that reads
    it says the average spans the space.
    """
    from ..ci.strings import cas_dimension

    n = int(floor) + int(margin)
    if int(n_elec) % 2 and n % 2:
        n += 1                              # an odd count cannot be a manifold boundary
    return int(max(1, min(n, cas_dimension(int(n_active), int(n_elec)))))



def probe(reference, coeff: np.ndarray, space, *, n_roots: int,
          budget: ProbeBudget = ProbeBudget(), report: bool = False) -> ProbeResult:
    """Pre-optimize ``space`` from ``coeff`` under ``budget`` and measure its spectrum.

    Parameters
    ----------
    reference
        A finished spinor reference (duck-typed: ``factors``, ``h_one_electron()``, ``data``,
        ``orth``) -- the same five things every other consumer of one asks for.
    coeff
        ``(2*nao, nspinor)`` AO-basis spinors the candidate columns index into. ⚠ It is the
        **rotated** set a candidate construction returned, never the reference's own guess: a
        selection read against the wrong orbital set is a different active space wearing the
        same description.
    space
        The :class:`~kuiva.mcscf.casci.ActiveSpace` to probe.
    n_roots
        Roots solved, and averaged over (:func:`probe_roots`). ⚠ The module docstring
        measures why those are one number here and two downstream.

    The returned orbitals are **Kramers repaired**
    (:func:`kuiva.mcscf.preopt.repair_kramers_pairing`) before they leave, because the next
    round's candidate construction folds them onto pairs and the state-averaging gate
    downstream assumes the convention exactly.

    ⚠ A :class:`~kuiva.mcscf.adaptive.SolverFailure` propagates: a probe that could not be
    measured is *not* a probe that measured nothing, and the round loop keeps the class out
    with the failure printed rather than reading a spectrum that does not exist.
    """
    from ..mcscf.preopt import preoptimize, repair_kramers_pairing

    spaces = space.spaces
    n_roots = int(max(1, n_roots))
    with timer("autocas/probe", log=log) as clock:
        result = preoptimize(
            reference.factors, reference.h_one_electron(), np.asarray(coeff),
            spaces, space.n_elec, e_nuc=reference.data.e_nuc, n_states=n_roots,
            max_iter=budget.max_iter, mode=budget.mode, conv_grad=budget.conv_grad,
            space_policy=budget.space_policy, max_determinants=budget.max_determinants,
            report=report)
        repaired, deviation = repair_kramers_pairing(result.coeff, reference.orth.x,
                                                     reference.data.s_ao, spaces)
    spectrum = np.asarray(result.ci.relative_cm, dtype=float)
    probe_result = ProbeResult(
        space=space, coeff=repaired, result=result, spectrum_cm=spectrum,
        entropy=np.asarray(result.entropy, dtype=float),
        mutual_information=np.asarray(result.mutual_information, dtype=float),
        occupations=np.asarray(result.orbital_occupation, dtype=float),
        n_roots=int(spectrum.size), n_determinants=int(result.ci.n_determinants),
        converged=bool(result.converged), cpu_seconds=float(clock.cpu),
        wall_seconds=float(clock.wall), budget=budget, pairing_deviation=float(deviation))
    log.debug("probe of CAS(%d, %d): %d roots over %d determinants, %.1f s cpu",
              space.n_elec, probe_result.n_active, probe_result.n_roots,
              probe_result.n_determinants, probe_result.cpu_seconds)
    if report:
        report_probe(probe_result)
    return probe_result


def embed_determinants(dets, space, trial, occupation) -> object:
    """``dets`` of ``space`` re-expressed over the larger ``trial`` space's active spinors.

    Every active spinor of ``space`` must be active in ``trial``; the spinors ``trial`` adds
    carry their reference occupation -- a doubly occupied pair both bits, an empty pair none,
    and a singly occupied pair **either** bit, so each seed determinant appears once per
    choice and the embedded list stays closed under time reversal as far as the seed was.
    ⚠ The electron count must come out as ``trial``'s: a space whose electrons were *stated*
    for a class rather than read off the occupations cannot be embedded, and the refusal says
    so rather than seeding with a list of the wrong particle number.
    """
    from itertools import product

    from ..ci.strings import Determinants

    small = np.asarray(space.spaces.active, dtype=int)
    large = np.asarray(trial.spaces.active, dtype=int)
    position = {int(c): i for i, c in enumerate(large)}
    missing = [int(c) for c in small if int(c) not in position]
    if missing:
        raise ValueError("spinor(s) {} of the accepted space are not active in the trial "
                         "space, so its determinants cannot be embedded".format(missing))
    added = [int(c) for c in large if int(c) not in set(small.tolist())]
    occ = np.asarray(occupation, dtype=float)
    base = np.zeros(dets.ndet, dtype=np.uint64)
    masks = np.asarray(dets.masks, dtype=np.uint64)
    for i, c in enumerate(small):
        bit = (masks >> np.uint64(i)) & np.uint64(1)
        base |= bit << np.uint64(position[int(c)])
    fixed = np.uint64(0)
    choices = []
    electrons = 0
    for c in added[0::2]:
        pair = float(occ[c] + occ[c + 1])
        lo, hi = np.uint64(1) << np.uint64(position[c]), np.uint64(1) << np.uint64(position[c + 1])
        if pair > 1.5:
            fixed |= lo | hi
            electrons += 2
        elif pair > 0.5:
            choices.append((lo, hi))
            electrons += 1
    if int(trial.n_elec) != int(space.n_elec) + electrons:
        raise ValueError(
            "the trial space holds {} electrons and the accepted space {} plus {} from the "
            "reference occupations of the added pairs: the class's electron count is not the "
            "one its orbitals carry, so the accepted determinants do not embed"
            .format(trial.n_elec, space.n_elec, electrons))
    parts = [base | fixed | np.uint64(sum(int(b) for b in bits))
             for bits in product(*choices)] if choices else [base | fixed]
    return Determinants(masks=np.concatenate(parts), n_spinor=int(large.size),
                        n_elec=int(trial.n_elec))


def measure(reference, coeff: np.ndarray, space, *, n_roots: int,
            budget: ProbeBudget = ProbeBudget(), seed=None) -> ProbeResult:
    """The cheap CI of ``space`` at the orbitals ``coeff``, **without** pre-optimizing them.

    What a keep/drop verdict is taken on. The returned object has the probe's shape (spectrum,
    entropies, mutual information) so the pruning and the bridge ranking read it unchanged;
    :attr:`ProbeResult.coeff` is ``coeff`` itself and :attr:`ProbeResult.fixed_orbitals` is set.

    ⚠ **Why the decision is not taken on two probes.** A probe pre-optimizes, and an
    optimization stopped on its iteration budget rotates into whatever it is given: adding
    four or seven Kramers pairs that describe nothing (the highest virtuals, the deepest core)
    moved a probe's target spectrum by 58-102 cm^-1, above the tolerance, and even a rotation
    *inside* the inactive and virtual blocks -- which changes no energy -- moved a TiCl3 probe's
    gap from 3945 to 2312 cm^-1. At fixed orbitals the same additions move the spectrum by at
    most 0.2 cm^-1 on TiCl3 and FeCl2 (seven pairs of either kind, drawn from orbital-energy
    eigenvectors so the control is unique), and the measurement is deterministic: no
    trajectory, and one selected CI whose inputs are the integrals alone.

    ⚠ **What it therefore measures is what a class does to the CI at the reference orbitals.**
    That is correlation plus the *state-specific* relaxation a larger space allows (a
    correlating shell on a ``d^1`` ion is all relaxation) -- both real reasons for a class, and
    neither something a state-averaged calculation on the smaller space would recover. What it
    no longer sees is how the optimizer would move the orbitals for the average, which is the
    part that did not reproduce.

    The orbitals are the candidate construction's (AVAS-rotated reference orbitals), which are
    exactly Kramers paired, so there is nothing to repair.

    ``seed`` is the accepted space's result (a :class:`ProbeResult` of this function) when
    ``space`` extends it: its determinants are embedded (:func:`embed_determinants`) and kept
    whole, and ``budget.max_determinants`` bounds what this measurement adds to them. ⚠ Without
    the nesting a selection started afresh in the larger space can drop determinants the
    accepted one held, and the distance between the two spectra reads the truncation.
    """
    from ..mcscf.orbopt import CASIntegrals
    from ..mcscf.preopt import cheap_ci

    n_roots = int(max(1, n_roots))
    coeff = np.asarray(coeff)
    with timer("autocas/measure", log=log) as clock:
        ints = CASIntegrals.build(reference.factors, reference.h_one_electron(), coeff,
                                  space.spaces, e_nuc=reference.data.e_nuc)
        seeded = None
        if seed is not None:
            seeded = embed_determinants(seed.result.dets, seed.space, space,
                                        reference.spinors.occ)
        ci = cheap_ci(ints.h_active_effective(), ints.active_eri(), int(space.n_elec),
                      n_states=n_roots, max_determinants=budget.max_determinants,
                      with_2rdm=False, seed=seeded)
        entropy, information = ci.entanglement()
    spectrum = np.asarray(ci.relative_cm, dtype=float)
    result = ProbeResult(
        space=space, coeff=coeff, result=ci, spectrum_cm=spectrum,
        entropy=np.asarray(entropy, dtype=float),
        mutual_information=np.asarray(information, dtype=float),
        occupations=np.clip(np.real(np.diag(ci.gamma)), 0.0, 1.0),
        n_roots=int(spectrum.size), n_determinants=int(ci.n_determinants), converged=True,
        cpu_seconds=float(clock.cpu), wall_seconds=float(clock.wall), budget=budget,
        fixed_orbitals=True)
    log.debug("fixed-orbital CI of CAS(%d, %d): %d roots over %d determinants, %.1f s cpu",
              space.n_elec, result.n_active, result.n_roots, result.n_determinants,
              result.cpu_seconds)
    return result


def report_probe(result: ProbeResult, logger=None) -> None:
    """The probe's own block: what was solved, at what budget, and what it cost."""
    logger = logger or log
    out.entries(logger, [
        ("probe space", "CAS({}, {})".format(result.space.n_elec, result.n_active), "",
         result.space.description),
        ("probe budget", result.budget.describe()),
        ("probe roots / determinants", "{} / {}".format(result.n_roots,
                                                        result.n_determinants)),
        ("probe cost", result.cpu_seconds, "s cpu", "{:.1f} s wall".format(
            result.wall_seconds), out.TIME_FMT),
    ])
