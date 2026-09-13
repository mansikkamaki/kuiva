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
because a boundary is only visible from the state *above* it.

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
           "ProbeBudget", "ProbeResult", "probe", "probe_roots", "report_probe"]

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
