"""The round loop: which feature classes stay, how large the space may be, and why.

One paragraph of design
-----------------------
Round 0 measures the **core** -- every target shell, plus anything ``require=`` pinned -- with
the cheap CI at the candidate construction's orbitals. Each later round offers one feature
class, prunes its candidates by relative single-orbital entropy, measures again at the same
orbitals, and keeps the class only if the *target spectrum* moved by more than a stated
tolerance. Over budget, whole classes are dropped in a fixed priority order. The accepted
space is then pre-optimized **once** (the probe), for the orbitals handed downstream and the
spectrum the root proposal reads. Every keep, drop and number is printed in one table, so the
space that comes out is attributable class by class.

⚠ **Verdicts at fixed orbitals, the pre-optimization once at the end** (measured, 2026-09-13).
The rounds used to compare two *probes*, each pre-optimized, and a pre-optimization stopped on
its iteration budget rotates into whatever it is given: adding seven Kramers pairs that
describe nothing moved the target spectrum by up to 102 cm^-1, above the tolerance, so a
"kept" on a multi-pair class did not say the class mattered. At fixed orbitals the same
additions move it by at most 0.2 cm^-1. See :func:`kuiva.autocas.probe.measure` for what
that measurement does and does not include.

⚠ **The probe's determinant budget must hold the sites' Hund configurations**, or the
protocol refuses before its first measurement (:func:`_check_probe_budget`): below that
product a selected CI cannot represent a coupled system's ground manifold, and the verdicts
were once read off truncation artefacts without anything noticing.

⚠ **The spectrum decides and entropy only prunes** (user decision, 2026-09-12), and the two
halves of that rule are here for measured reasons rather than taste:

* an entropy criterion is **blind to a low-dimensional bridge** -- a radical or a bridging
  ligand shares at most ``ln 2`` nats with either ion, so in *absolute* terms it ranks below
  every metal orbital, and a threshold that admits it admits everything;
* an entropy criterion is **blind to a correlating shell** and to the empty members of a d
  manifold, which at the probe's level of correlation carry nothing at all;
* but entropy is a perfectly good *relative* ranking inside one class of candidates, where
  all the members answer the same question, and that is what it is used for.

What "the target spectrum" is
-----------------------------
The relative energies inside the theoretical ground manifold, plus the gap to the next one
(:func:`kuiva.autocas.roots.target_spectrum`). For coupled shells that is the exchange
splitting, for a single ion the ligand-field pattern, for a radical bridge the radical-ion
coupling -- the quantity the active space is being chosen to describe. A class is kept when it
moves one of those numbers by more than ``max(spectrum_tol x the manifold's width,
spectrum_tol_cm)``: a percentage **with an absolute floor**, because a ligand-field band
stated as a percentage alone is meaningless where the splitting is small.

⚠ **Three verdicts, not two.** Above the tolerance is *kept*; below the measured probe noise
is *dropped*; in between is **"inconclusive, kept"**. A larger space is the safe error and
the budget bounds it, whereas dropping a class the probe could not resolve is a silent claim
that it does not matter. Lanthanide exchange is the case this exists for: 256 states split by
cm^-1 is below anything a cheap CI resolves, and the honest outcome there is that the bridge
orbitals ride on a *requested* class.

⚠ **Order matters and the priority is not a knob.** Classes are tested against the accepted
space of the previous rounds -- a bonding partner is tested in the presence of the bridge,
never the other way round. That is the physics (the mechanism first, the correlation
refinement after) and it is what makes a drop attributable: a class that changes nothing
*given what is already there* is a class this calculation does not need.

⚠ **A shell is never pruned and never cut**, and the core alone over budget is a refusal
rather than a smaller shell. A shell with pairs missing is a different physical statement
wearing the shell's name, and the two ways out -- the tensor-network solver, or fewer/pooled
centres -- are named in the refusal.

References
----------
* C. J. Stein, M. Reiher, J. Chem. Theory Comput. 12, 1760 (2016),
  doi:10.1021/acs.jctc.6b00156; Chimia 71, 170 (2017), doi:10.2533/chimia.2017.170 -- the
  relative single-orbital-entropy criterion used here to prune (and, as the module docstring
  says, *only* to prune).
* J. Rissler, R. M. Noack, S. R. White, Chem. Phys. 323, 519 (2006),
  doi:10.1016/j.chemphys.2005.10.018 -- the mutual information the bridge ranking reads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util.logging import get_logger
from ..util.window import DEFAULT_MANIFOLD_GAP_CM, DEFAULT_MAX_STATES
from . import candidates as cand
from . import centres as ctr
from . import multiplets as mult
from . import targets as tg
from .probe import ProbeBudget, ProbeResult, measure, probe, probe_roots
from .roots import RootProposal, propose_roots, target_spectrum

log = get_logger(__name__)

__all__ = ["Assembly", "Budget", "DEFAULT_DMRG_SPINORS", "DEFAULT_PROBE_NOISE_CM",
           "DEFAULT_PRUNE_REL", "DEFAULT_SPECTRUM_TOL", "DEFAULT_SPECTRUM_TOL_CM",
           "RoundRecord", "assemble", "ci_residency_gb", "prune_candidates",
           "resolve_budget", "spectrum_distance"]

#: Relative part of the keep/drop tolerance: a class has to move the target manifold by this
#: fraction of the manifold's own width.
DEFAULT_SPECTRUM_TOL = 0.05

#: Absolute floor of the same tolerance [cm^-1]. ⚠ A percentage alone is meaningless where the
#: manifold is narrow, which is the rule every ligand-field band in this project obeys.
#:
#: **Deliberately the manifold-gap threshold** (:data:`~kuiva.util.window.DEFAULT_MANIFOLD_GAP_CM`,
#: which is also the state-average boundary diagnostic's): a class stays when it moves the
#: ground manifold by more than the amount that makes a boundary *unambiguous* in the first
#: place. It sits five times above the measured noise floor below.
DEFAULT_SPECTRUM_TOL_CM = 50.0

#: The probe's noise floor [cm^-1]: below this a difference between two probes is not a
#: measurement at all, and the class is **dropped**. Between it and the tolerance the verdict
#: is "inconclusive, kept".
#:
#: ⚠ **Measured against a control, because repeats cannot see it.** The measurement is
#: deterministic, so what the round loop reads -- the distance between the core and the core
#: plus a class -- can only be checked for a component that is not the class's physics by
#: adding pairs that describe nothing the ground manifold is made of: the deepest inactive and
#: the highest virtual pairs, taken as eigenvectors of the pair-folded inactive Fock so the
#: control is unique. At the fixed orbitals the verdicts are taken at, one to seven such pairs
#: moved the target spectrum by at most 0.2 cm^-1 on TiCl3 and FeCl2, so 10 is far above the
#: noise at every class size.
#:
#: ⚠ **Size-independent only because the orbitals are fixed.** When the verdict compared two
#: pre-optimized probes the same one-pair control gave 1.7-7.7 cm^-1 and four to seven pairs
#: gave 58-102, above :data:`DEFAULT_SPECTRUM_TOL_CM`: a pre-optimization stopped on its budget
#: rotates into whatever it is given. Reintroducing an optimization before a verdict
#: reintroduces that.
DEFAULT_PROBE_NOISE_CM = 10.0

#: Relative single-orbital-entropy cut for pruning a class's candidates: keep a pair whose
#: entropy is at least this fraction of the largest among the candidates of the same class.
DEFAULT_PRUNE_REL = 0.1

#: Provisional spinor cap for ``solver="dmrg"``, printed as provisional. A network solve's
#: cost is set by bond dimension and topology rather than by the spinor count, so this is a
#: statement about what has been *measured*, not a ceiling of the method.
DEFAULT_DMRG_SPINORS = 40


# --- the size budget --------------------------------------------------------------------

def ci_residency_gb(n_spinor: int, n_elec: int, n_roots: int) -> float:
    """Resident memory [GB] a conventional-CI solve of ``CAS(n_elec, n_spinor)`` needs.

    The terms the conventional-CI ceiling is made of, each from its own exact sizing function
    and none of them padded here: the sigma workspace (``C(n,k) n^2``, which is what actually
    binds), the Davidson stacks at this root count, the excitation map, the CI vectors, the
    reshaped active integrals and the state-averaged 2-RDM.

    ⚠ **The root count co-decides**, which is why this takes one: past 20 spinors at half
    filling a few-root solve runs where a real state average does not, so a budget quoted
    without the number of roots it was computed at is not a budget.
    """
    from ..ci.davidson import davidson_workspace_gb, subspace_cap
    from ..ci.sigma import eri_matrix_gb, sigma_workspace_gb
    from ..ci.strings import cas_dimension, cas_vector_gb, excitation_map_gb
    from ..util.resources import rdm_gb

    n_spinor, n_elec, n_roots = int(n_spinor), int(n_elec), int(max(1, n_roots))
    ndet = cas_dimension(n_spinor, n_elec)
    return (sigma_workspace_gb(n_spinor, n_elec)
            + davidson_workspace_gb(ndet, subspace_cap(n_roots, ndet))
            + excitation_map_gb(n_spinor, n_elec)
            + cas_vector_gb(n_spinor, n_elec, n_roots)
            + eri_matrix_gb(n_spinor)
            + rdm_gb(n_spinor, 2))


@dataclass(frozen=True)
class Budget:
    """How large the active space may be, and where the number came from.

    ⚠ **A budget is a statement about a solver at a root count**, never about spinors alone:
    the conventional CI's ceiling moves with both. :attr:`source` is what the output prints,
    so a user reading "34 spinors at 16 roots under 6.2 GB (ci)" can see every input to it.
    """

    solver: str
    max_spinors: int
    n_roots: int
    source: str
    max_determinants: Optional[int] = None
    #: The memory the budget was resolved against, when it was resolved rather than stated.
    #: ⚠ Its presence changes what :meth:`fits` *means*: a memory budget is re-evaluated at
    #: the filling it is asked about, because the conventional-CI ceiling is a bound on the
    #: **determinant count** and not on the spinor count -- a dilute space of 30 spinors runs
    #: where a half-filled one of 24 refuses. :attr:`max_spinors` is then the headline number
    #: at the core's own filling, for the report.
    limit_gb: Optional[float] = None
    provisional: bool = False

    def fits(self, n_spinor: int, n_elec: int) -> bool:
        from ..ci.strings import cas_dimension

        if self.max_determinants is not None and self.solver == "ci":
            return cas_dimension(int(n_spinor), int(n_elec)) <= int(self.max_determinants)
        if self.limit_gb is not None and self.solver == "ci":
            return ci_residency_gb(int(n_spinor), int(n_elec),
                                   self.n_roots) <= float(self.limit_gb)
        return int(n_spinor) <= int(self.max_spinors)

    def describe(self) -> str:
        text = "{} spinors at {} roots ({})".format(self.max_spinors, self.n_roots,
                                                    self.solver)
        if self.max_determinants is not None:
            text += ", at most {} determinants".format(self.max_determinants)
        return text + " -- " + self.source


def resolve_budget(*, solver: str = "ci", n_elec: int, n_roots: int,
                   max_spinors: Optional[int] = None,
                   max_determinants: Optional[int] = None,
                   available_gb: Optional[float] = None) -> Budget:
    """The size budget, stated or resolved from the memory limit at ``n_roots`` roots.

    With neither ``max_spinors`` nor ``max_determinants`` the conventional-CI budget is the
    largest spinor count whose :func:`ci_residency_gb` fits what the configured memory limit
    still has free, and the tensor-network one is :data:`DEFAULT_DMRG_SPINORS`, flagged
    **provisional** in the output because it is a measured-in-waiting default rather than a
    property of the method.

    ⚠ There is a loop here and it is resolved by measuring twice: the budget depends on the
    root count and the proposed root count depends on the final probe. The protocol resolves
    at the **floor** count, assembles, and then re-checks at the proposal, reporting both --
    a space that fits at the floor and not at the proposal is a real outcome and is stated
    rather than silently rounded away.
    """
    if solver not in ("ci", "dmrg"):
        raise ValueError("solver is 'ci' or 'dmrg'; got {!r}".format(solver))
    n_roots = int(max(1, n_roots))
    if max_spinors is not None and max_determinants is not None:
        raise ValueError("give max_spinors= or max_determinants=, not both: they are two "
                         "statements of one size, and a run that silently preferred one "
                         "would not be reproducible from its report")
    if max_determinants is not None and solver != "ci":
        raise ValueError("max_determinants= bounds a determinant space and therefore the "
                         "conventional CI; a network solve's size is bounded in spinors "
                         "(max_spinors=)")
    if max_spinors is not None:
        return Budget(solver=solver, max_spinors=int(max_spinors), n_roots=n_roots,
                      source="stated as max_spinors={}".format(int(max_spinors)))
    if max_determinants is not None:
        from ..ci.strings import cas_dimension

        n = int(n_elec)
        while cas_dimension(n + 2, int(n_elec)) <= int(max_determinants):
            n += 2                                   # whole Kramers pairs, always
        return Budget(solver=solver, max_spinors=n, n_roots=n_roots,
                      max_determinants=int(max_determinants),
                      source="stated as max_determinants={}".format(int(max_determinants)))
    if solver == "dmrg":
        return Budget(solver=solver, max_spinors=DEFAULT_DMRG_SPINORS, n_roots=n_roots,
                      source="the provisional tensor-network default", provisional=True)

    from ..util.resources import BUDGET

    free = float(available_gb if available_gb is not None else BUDGET.available_gb())
    if not np.isfinite(free):
        raise ValueError(
            "no memory limit is configured, so the size of an automatic active space cannot "
            "be bounded: the conventional-CI ceiling IS a memory bound. Set the limit once "
            "(memory_gb= on the front end, or the configuration file), or state the size "
            "here with max_spinors= / max_determinants=")
    n = max(int(n_elec), 2)
    while ci_residency_gb(n + 2, int(n_elec), n_roots) <= free:
        n += 2
    return Budget(solver=solver, max_spinors=int(n), n_roots=n_roots, limit_gb=free,
                  source="{:.1f} GB free of the configured limit".format(free))


# --- the rounds -------------------------------------------------------------------------

@dataclass
class RoundRecord:
    """One row of the rounds table: what was offered, what survived, and on what number."""

    index: int
    cls: str
    n_candidates: int
    n_pruned: int
    distance_cm: Optional[float]
    tolerance_cm: Optional[float]
    verdict: str
    n_spinors: int
    cpu_seconds: float
    note: str = ""

    @property
    def kept(self) -> bool:
        return self.verdict.startswith("core") or self.verdict.startswith("kept")


@dataclass
class Assembly:
    """What the protocol assembled: a space, the orbitals it lives in, and the evidence."""

    space: Any                              #: :class:`~kuiva.mcscf.casci.ActiveSpace`
    coeff: np.ndarray                       #: (2*nao, n) AO-basis spinors the columns index
    sets: Tuple[cand.CandidateSet, ...]     #: the accepted classes, in priority order
    rounds: List[RoundRecord]
    probe: ProbeResult                      #: the final probe, at the assembled space
    centres: Tuple[ctr.Centre, ...]
    floor: int
    product_floor: int
    budget: Budget
    proposal: RootProposal
    #: The atoms of each site, one entry per magnetic centre (per **atom** where a centre is
    #: pooled over equivalent ones). What a site partition is computed *from*; empty for a
    #: single-atom centre, where there is no partition to make.
    site_atoms: Tuple[Tuple[int, ...], ...] = ()
    #: Per-site active spinor columns, once the shells have been localized. ⚠ Filled in by
    #: whoever runs the localization -- it goes through the project's one site partition
    #: (:func:`kuiva.interface.api.localize_active_space`) rather than a second one here --
    #: and stays ``None`` where there is one centre or the localization did not reach its
    #: population floor.
    sites: Optional[Tuple[Tuple[int, ...], ...]] = None
    notes: Tuple[str, ...] = ()
    #: The fixed-orbital measurement of the accepted space the last verdict was taken on
    #: (:func:`~kuiva.autocas.probe.measure`), beside :attr:`probe`, the pre-optimization of
    #: the same space. ⚠ Two spectra of one space that differ by the orbitals only; the rounds
    #: table's numbers are this one's.
    decided: Optional[ProbeResult] = None

    @property
    def n_active(self) -> int:
        return int(self.space.n_active)

    def dmrg_ordering(self) -> np.ndarray:
        """Mode order for a tensor network: **site-blocked** where sites are known.

        ⚠ The topology of a polynuclear space comes from the site partition and never from
        mutual information: a bridging orbital shares at most ``ln 2`` with either ion, so an
        entanglement-ordered chain puts the pathway wherever the numerical noise of a
        qualitative probe happens to rank it. With one centre -- or where the localization
        could not separate the sites -- there is no site structure to use and this is the
        Fiedler order of the probe's own mutual information, as the cheap CI's is.
        """
        active = list(np.asarray(self.space.spaces.active, dtype=int))
        if self.sites is None or len(self.sites) < 2:
            return self.probe.result.dmrg_ordering()
        position = {int(c): i for i, c in enumerate(active)}
        placed: List[int] = []
        site_columns = [list(s) for s in self.sites]
        ligand = [c for c in active if not any(int(c) in set(s) for s in site_columns)]
        for i, site in enumerate(site_columns):
            placed.extend(int(c) for c in site)
            if i == 0:
                placed.extend(int(c) for c in ligand)   # the pathway between the first two
        return np.asarray([position[c] for c in placed], dtype=int)

    def summary_entries(self) -> List[Tuple[str, str]]:
        return [
            ("active space", "CAS({}, {})".format(self.space.n_elec, self.n_active)),
            ("statement", self.space.description),
            ("budget", self.budget.describe()),
            ("floor", "{} (product over centres {})".format(self.floor, self.product_floor)),
            ("proposal", self.proposal.describe()),
        ]


def spectrum_distance(reference_cm: Sequence[float],
                      trial_cm: Sequence[float]) -> Tuple[float, str]:
    """How far two target spectra are apart [cm^-1], and what to say about it.

    The largest change over the manifold's relative energies and its gap. ⚠ **A change in the
    *number* of those energies is not a distance at all** -- the class changed the manifold
    structure, which is a bigger statement than any tolerance -- so it comes back as
    ``inf`` with the two sizes named, and the class is kept.
    """
    a = np.asarray(reference_cm, dtype=float).ravel()
    b = np.asarray(trial_cm, dtype=float).ravel()
    if a.size != b.size:
        return (float("inf"),
                "the target manifold changed size ({} -> {} states): the class changed the "
                "structure of the ground manifold, not only its splittings"
                .format(a.size, b.size))
    if a.size == 0:
        return 0.0, ""
    return float(np.max(np.abs(a - b))), ""


def prune_candidates(candidate: cand.CandidateSet, result: ProbeResult, *,
                     prune_rel: float = DEFAULT_PRUNE_REL,
                     ranking: Optional[np.ndarray] = None,
                     rtol: float = cand.DEFAULT_RANKING_RTOL) -> Tuple[np.ndarray, str]:
    """Candidate columns that survive the relative single-orbital-entropy cut.

    Keeps a candidate Kramers pair whose entropy is at least ``prune_rel`` times the largest
    among the candidates **of the same class** -- a relative criterion inside one class,
    which is the only form in which an entanglement measure says anything about a
    low-dimensional bridge.

    ⚠ Three rules bind it, and each has its own failure mode:

    * **whole Kramers pairs**, because a space is defined on spatial orbitals;
    * **whole degenerate groups**: a cut that would split a tie in the entropy is rounded
      outward, so the selection cannot break a degeneracy the molecule has exactly;
    * **the core is never touched** -- this function is only ever given one class's
      candidates, and a fixed class (a shell, a correlating shell) never reaches it at all.

    ``ranking`` is an optional secondary order used where entropies tie (the bridge's mutual
    information with the two sites); it changes which of two equally entangled pairs is kept
    and never how many.
    """
    columns = np.asarray(candidate.columns, dtype=int)
    if columns.size == 0:
        return columns, ""
    pairs = columns[0::2] // 2
    active = np.asarray(result.space.spaces.active, dtype=int)
    position = {int(c): i for i, c in enumerate(active)}
    missing = [int(c) for c in columns if int(c) not in position]
    if missing:
        raise ValueError("candidate spinor(s) {} are not in the probed active space: a prune "
                         "reads the entropies of the space it was measured in"
                         .format(missing))
    s1 = np.asarray(result.entropy, dtype=float)
    values = np.asarray([0.5 * (s1[position[int(2 * p)]] + s1[position[int(2 * p) + 1]])
                         for p in pairs], dtype=float)
    secondary = (np.zeros_like(values) if ranking is None
                 else np.asarray(ranking, dtype=float).ravel())
    if secondary.size != values.size:
        raise ValueError("the secondary ranking has {} values for {} candidate pairs"
                         .format(secondary.size, values.size))
    order = np.lexsort((-secondary, -values))       # entropy first, the ranking breaks ties
    ordered, sorted_values = pairs[order], values[order]
    cut = float(prune_rel) * float(np.max(values))
    n_keep = int(np.count_nonzero(sorted_values >= cut))
    n_keep, note = cand._round_out(sorted_values, n_keep, rtol=rtol)
    kept = np.sort(ordered[:n_keep])
    text = ""
    if n_keep < pairs.size:
        text = ("pruned {} of {} candidate pair(s) below {:.0%} of the largest single-orbital "
                "entropy in the class (entropies {:.4f}..{:.4f}, cut {:.4f})"
                .format(pairs.size - n_keep, pairs.size, prune_rel,
                        float(np.max(values)), float(np.min(values)), cut))
    if note:
        text = (text + "; " if text else "") + note
    return cand._pair_columns(kept), text


# --- assembling ---------------------------------------------------------------------------

@dataclass
class _Projection:
    """The AVAS spectrum carried forward through a rotation that preserved it.

    ⚠ ``coeff`` is the **current** orbital set and need not be the one the projection was
    computed in: a fragment rotation (the frontier class) is confined to blocks of pairs whose
    projection is degenerate, so every eigenvalue survives it unchanged while the orbitals
    move. Handing the bridge class the original coefficients instead would rank it on
    populations of orbitals that no longer exist.
    """

    eigenvalues: np.ndarray
    occupations: np.ndarray
    coeff: np.ndarray
    selected: np.ndarray


def _require_columns(reference, coeff, occupation, statements, *, what: str):
    """``require=``/``exclude=`` statements -> ``(columns, descriptions)``.

    One spelling, ``("character", atom, l, n_spinors[, skip_pairs])``, resolved through
    :func:`kuiva.mcscf.casci._character_columns` -- the project's one character selection,
    imported rather than re-derived, with its own refusals inherited.

    ⚠ **The atoms are addressed the way every other statement in this package addresses
    them** (:func:`kuiva.autocas.targets.resolve_atoms`: 1-based numbers, labels, or an
    element symbol that pools every atom of it), and resolved here before the selection sees
    them. Two numbering conventions inside one stage is how a pin lands on the wrong atom and
    still produces an active space.

    ⚠ **An AVAS statement is refused here rather than served.** A second projector's rotation
    re-mixes the pairs the shell's projection selected (they are degenerate at zero
    eigenvalue in it), so an AVAS-stated pin would silently redefine the shell it was meant to
    sit beside. State it as a character selection, or state the whole space by hand.
    """
    from ..mcscf.casci import DEFAULT_CHARACTER_THRESHOLD, _character_columns

    columns: List[int] = []
    described: List[str] = []
    for statement in (statements or ()):
        if not isinstance(statement, (tuple, list)):
            raise ValueError(
                "{}= takes character statements ('character', atom, l, n_spinors[, skip]); "
                "got {!r}".format(what, statement))
        parts = tuple(statement)
        if not parts or str(parts[0]).lower() != "character":
            raise ValueError(
                "{}= takes character statements ('character', atom, l, n_spinors[, skip]); "
                "an AVAS statement is deliberately not accepted, because a second projection "
                "rotates the pairs the shell's projection already selected. Got {!r}"
                .format(what, statement))
        if not 4 <= len(parts) <= 5:
            raise ValueError("('character', atom, l, n_spinors[, skip_pairs]); got {!r}"
                             .format(statement))
        n_spinors = int(parts[3])
        if n_spinors % 2:
            raise ValueError("{}= names {} spinors, which is not a whole number of Kramers "
                             "pairs".format(what, n_spinors))
        atoms = tg.resolve_atoms(reference.ao_layout, parts[1])
        cols, text = _character_columns(
            np.asarray(coeff), np.asarray(reference.data.s_ao), reference.ao_layout,
            atom=list(atoms), l=parts[2], n_pairs=n_spinors // 2,
            threshold=DEFAULT_CHARACTER_THRESHOLD, occupation=np.asarray(occupation),
            skip_pairs=int(parts[4]) if len(parts) == 5 else 0)
        columns.extend(int(c) for c in np.asarray(cols, dtype=int))
        described.append(text)
    return np.unique(np.asarray(columns, dtype=int)), tuple(described)


def _space_of(reference, sets: Sequence[cand.CandidateSet], *, extra_columns=(),
              extra_description: Sequence[str] = ()):
    """The :class:`~kuiva.mcscf.casci.ActiveSpace` of a set of accepted classes.

    The electron count is the **sum of what the classes carry** -- measured off the reference
    occupations of the selected pairs, except for a shell that is empty or full in the
    reference, where the ion's reference state supplies it (loudly, where it is a statement
    about the ion, and not at all where it is only the neutral-atom default).
    """
    from ..mcscf.casci import active_space

    parts = [np.asarray(s.columns, dtype=int) for s in sets if not s.empty]
    parts.append(np.asarray(extra_columns, dtype=int).ravel())
    columns = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=int)
    electrons = sum(s.electrons for s in sets if not s.empty)
    electrons += _extra_electrons(reference, extra_columns)
    fragments = tuple(s.fragments for s in sets if not s.empty and s.fragments)
    flat = tuple(f for group in fragments for f in group)
    description = "; ".join([s.description for s in sets if not s.empty]
                            + list(extra_description))
    space = active_space(columns, int(reference.nspinor), int(reference.data.nelec_total),
                         n_active_elec=int(round(electrons)), description=description)
    if flat:
        from dataclasses import replace as _replace
        space = _replace(space, fragments=flat)
    return space


def _extra_electrons(reference, columns) -> float:
    columns = np.asarray(columns, dtype=int).ravel()
    if columns.size == 0:
        return 0.0
    occ = np.asarray(reference.spinors.occ, dtype=float)
    return float(np.sum(occ[columns]))


def _floors(centres: Sequence[ctr.Centre], shell: cand.CandidateSet) -> Tuple[int, int, str]:
    """``(floor, product, term statement)`` from the centres' measured shell occupation.

    The Hund ground manifold per centre -- ``2J+1`` on the f block, the spin multiplicity on
    the d block, because spin-orbit coupling and the ligand field dominate in opposite orders
    there -- and the **product** over the individual centres, which is the dimension of the
    exchange manifold. Both are reported: the product is what a whole-manifold average of a
    coupled system would be (two Dy(3+) is 256 states) and it is printed even where no cap
    can reach it, since quietly proposing a fraction of it would be proposing a cut inside an
    exchange manifold.

    ⚠ The electron count per atom is the shell's **measured** total divided over the atoms
    carrying it: one pooled centre of two equivalent metals holds both shells, and its floor
    is one atom's manifold, not the pair's.
    """
    if not centres:
        return 1, 1, "no centre: no theoretical floor"
    n_atoms = sum(len(c.atoms) for c in centres)
    per_atom = float(shell.electrons) / max(n_atoms, 1)
    note = ""
    if abs(per_atom - round(per_atom)) > 1e-6:
        note = " (the shell's {:.1f} electrons do not divide over {} atoms; rounded)".format(
            float(shell.electrons), n_atoms)
    floors: List[int] = []
    terms: List[str] = []
    for centre in centres:
        n = int(max(0, min(round(per_atom), 4 * centre.l + 2)))
        term = mult.hund_ground_term(centre.l, n)
        floor = mult.shell_floor(centre.l, n)
        floors.extend([int(floor)] * len(centre.atoms))
        terms.append("{}{} ({})".format(
            "{} x ".format(len(centre.atoms)) if len(centre.atoms) > 1 else "",
            term.symbol, centre.where))
    return int(max(floors)), int(mult.coupled_floor(floors)), ", ".join(terms) + note


def _hund_product(centres: Sequence[ctr.Centre], shell: cand.CandidateSet) -> Tuple[int, str]:
    """``(dimension, statement)``: determinants of the product of every site's Hund configurations.

    Per **atom**, with the shell's measured electrons divided over the atoms carrying it, as
    :func:`_floors` does: a pooled centre of three d^5 manganese is three sites of 32
    determinants each, 32 768 together.
    """
    n_atoms = sum(len(c.atoms) for c in centres)
    per_atom = int(round(float(shell.electrons) / max(n_atoms, 1)))
    product, parts = 1, []
    for centre in centres:
        n = int(max(0, min(per_atom, 4 * centre.l + 2)))
        dim = mult.hund_configuration_dimension(centre.l, n)
        product *= dim ** len(centre.atoms)
        parts.append("{}{}^{} ({} determinants{})".format(
            "{} x ".format(len(centre.atoms)) if len(centre.atoms) > 1 else "",
            tg.angular_momentum_letter(centre.l), n, dim,
            " each" if len(centre.atoms) > 1 else ""))
    return int(product), ", ".join(parts)


def _check_probe_budget(budget: ProbeBudget, space, centres, shell) -> None:
    """Refuse a determinant budget that cannot hold the sites' Hund configurations.

    ⚠ **A selected CI capped below that product cannot represent a coupled system's exchange
    manifold at all**, and nothing downstream notices: measured on ``mn3_linear`` (three d^5
    sites, 32 768 determinants) at the 6000-determinant default, three probes of nearly the
    same space returned incompatible spectra -- the core's fifth state at 50 000 cm^-1, the
    bridged space's at 4 400 -- and the round loop "kept" a bridge at 60 538 cm^-1 on them.
    Where the core's whole determinant space fits the budget there is nothing to truncate and
    nothing to check. The product is a lower bound, never a sufficient budget: the
    charge-transfer determinants that mediate the coupling come on top of it.
    """
    from ..ci.strings import cas_dimension

    if cas_dimension(space.n_active, space.n_elec) <= int(budget.max_determinants):
        return
    product, statement = _hund_product(centres, shell)
    if product > int(budget.max_determinants):
        raise ValueError(
            "the probe's determinant budget ({}) cannot hold the {} determinants of the "
            "centres' Hund configurations ({}), so the selected CI cannot represent the "
            "ground manifold the classes are judged on, and every verdict would be read off a "
            "truncation artefact. State a budget of at least that size -- "
            "probe={{'max_determinants': {}}} on the stage, probe_budget=ProbeBudget("
            "max_determinants={}) here -- and more than it where ligand classes are offered, "
            "since the charge-transfer determinants come on top"
            .format(int(budget.max_determinants), product, statement, product, product))


def _coupled_and_truncated(space, centres, budget: ProbeBudget) -> str:
    """The reason no verdict can be measured on this core, or ``""`` where one can.

    ⚠ **A truncated core of coupled centres is not a space the cheap CI can measure an
    exchange manifold in, at any budget this was measured at.** On ``mn3_linear``'s CAS(15, 30)
    (three high-spin d^5 sites): selected at 6000 and at 40 000 determinants -- the latter above
    the 32 768 of the sites' Hund configurations -- the roots are not spin eigenstates at all
    (``<S^2>`` 9.69, 6.66, ... and 10.37, 8.83, ...); seeded with the whole Hund product space
    and 8000 selected on top, the ground level is S = 15/2 (the D = 8 network at comparable
    orbitals agrees) where Lieb-Mattis requires S = 5/2, and the selection splits that
    multiplet's components by 57-116 cm^-1 -- more than the exchange splittings a bridge class
    is judged on. A verdict read off that is a verdict on the selection. So where the shells
    belong to more than one atom and their determinant space exceeds the budget, the ligand
    classes are **kept unmeasured**, because they were asked for, and the output says so.
    Where the core is complete (``ti2cl6``'s CAS(2, 20)) the measurement is exact in the core
    and this does not apply.
    """
    from ..ci.strings import cas_dimension

    n_sites = sum(len(c.atoms) for c in centres)
    ndet = cas_dimension(space.n_active, space.n_elec)
    if n_sites < 2 or ndet <= int(budget.max_determinants):
        return ""
    return ("the core is the shells of {} coupled centres in {} determinants, over the "
            "{}-determinant budget: a truncated cheap CI of coupled centres does not "
            "represent their exchange manifold (measured: its roots are not spin eigenstates, "
            "or a spin multiplet split by more than the exchange), so no class can be decided "
            "on it. Every requested class that fits the size budget is KEPT UNMEASURED -- the "
            "pathway is in the space because it was asked for -- and the proposed count is "
            "read off a spectrum that has the same defect".format(n_sites, ndet,
                                                                   int(budget.max_determinants)))


def _site_split(reference, coeff, centres, shell_columns):
    """Shell columns per centre, by Loewdin population -- or ``None`` where it is ambiguous.

    ⚠ For *equivalent* centres the canonical (and AVAS-rotated) orbitals are the symmetric and
    antisymmetric combinations, each about half on each centre, and no population split of
    them is a statement about anything. That case returns ``None`` and the consumer says so;
    separating the sites is a localization, and it happens once, at the end.
    """
    atoms = [list(c.atoms) for c in centres]
    if len(atoms) < 2:
        if len(centres) == 1 and len(centres[0].atoms) > 1:
            atoms = [[a] for a in centres[0].atoms]
        else:
            return None
    columns = np.asarray(shell_columns, dtype=int)
    pairs = columns[0::2] // 2
    populations = np.stack([cand._fragment_pair_populations(reference, coeff, a)[pairs]
                            for a in atoms])
    if float(np.max(populations)) < 0.75:
        return None                        # delocalized combinations: no honest split
    owner = np.argmax(populations, axis=0)
    return tuple(tuple(int(c) for c in cand._pair_columns(pairs[owner == i]))
                 for i in range(len(atoms)))


def _resolve_centres(reference, shell_targets, *, report: bool) -> Tuple[ctr.Centre, ...]:
    """The centres a set of :class:`~kuiva.autocas.targets.Shell` targets names.

    ``Shell()`` -- the default target -- is "whatever the detection finds"; a named one is
    resolved atom by atom, with the detection's own refusals (a closed shell, two open
    channels, a ghost). ⚠ Centres of two different ``l`` are refused rather than composed:
    one AVAS projector per ``l``, and a second projector re-mixes the first one's selection.
    """
    found: List[ctr.Centre] = []
    for target in shell_targets:
        if getattr(target, "detected", False):
            detection = ctr.detect_centres(reference, report=report)
            if not detection.centres:
                raise ValueError(
                    "no magnetic centre was detected: no frontier Kramers pair carries more "
                    "than {:.0%} of its population on an atom whose free-atom reference has "
                    "an OPEN d or f shell. This molecule has no open d/f shell to build a "
                    "default active space around -- name what the space is for instead "
                    "(targets=[('frontier', atoms)] for a radical, ('shell', atoms, l) for a "
                    "shell the detection did not reach)"
                    .format(ctr.DEFAULT_CENTRE_THRESHOLD))
            found.extend(detection.centres)
        else:
            found.append(ctr.shell_centre(reference, target.atoms, target.l))
    seen: Dict[Tuple[Tuple[int, ...], int], ctr.Centre] = {}
    for centre in found:
        seen.setdefault((centre.atoms, centre.l), centre)
    centres = tuple(sorted(seen.values(), key=lambda c: (c.atoms[0], c.l)))
    ells = sorted({c.l for c in centres})
    if len(ells) > 1:
        raise ValueError(
            "the targets name shells of more than one angular momentum ({}), which needs one "
            "AVAS projection each -- and a second projection's rotation re-mixes the pairs "
            "the first one selected, because they are degenerate at zero projection in it. "
            "Select one shell at a time, or state the active space explicitly"
            .format(", ".join(tg.angular_momentum_letter(e) for e in ells)))
    return centres


def _centre_atoms(centres: Sequence[ctr.Centre]) -> Tuple[Tuple[int, ...], ...]:
    """One site per **atom** of every centre -- what a site partition is computed from."""
    atoms = tuple((int(a),) for c in centres for a in c.atoms)
    return atoms if len(atoms) > 1 else ()


def _check_named(centres: Sequence[ctr.Centre], target, layout, what: str) -> None:
    """Refuse a class that names atoms which are not centres of the shell construction."""
    named = set(tg.resolve_atoms(layout, target.atoms))
    known = set(int(a) for c in centres for a in c.atoms)
    stray = sorted(named - known)
    if stray:
        raise ValueError(
            "the {} target names {}, which is not a centre of this selection ({}): the class "
            "is read off the shell's own projection, so it exists only where a shell does. "
            "Add a shell target for it, or drop the class"
            .format(what, ", ".join(layout.atom_label(a) for a in stray),
                    ", ".join(c.where for c in centres)))


def _drop_excluded(candidate: cand.CandidateSet, excluded) -> cand.CandidateSet:
    """Remove banned columns from a class, keeping whole Kramers pairs."""
    from dataclasses import replace as _replace

    banned = set(int(c) // 2 for c in np.asarray(excluded, dtype=int).ravel())
    if not banned or candidate.empty:
        return candidate
    columns = np.asarray(candidate.columns, dtype=int)
    pairs = columns[0::2] // 2
    keep = np.asarray([int(p) not in banned for p in pairs], dtype=bool)
    if bool(np.all(keep)):
        return candidate
    if candidate.fixed:
        raise ValueError(
            "exclude= names Kramers pair(s) inside the {} class, which is taken whole or not "
            "at all ('{}'). A shell with pairs missing is a different physical statement "
            "wearing the shell's name; drop the whole class instead"
            .format(candidate.cls, candidate.description))
    note = "exclude= removed {} of {} candidate pair(s)".format(
        int(np.count_nonzero(~keep)), int(keep.size))
    # ⚠ One value per Kramers pair in ``occupations`` and ``ranking``, two columns per pair
    # in ``columns``: the mask is the pair one.
    return _replace(candidate, columns=cand._pair_columns(pairs[keep]),
                    occupations=np.asarray(candidate.occupations)[keep],
                    ranking=np.asarray(candidate.ranking)[keep],
                    fragments=(tuple(int(c) for c in cand._pair_columns(pairs[keep])),),
                    notes=tuple(candidate.notes) + (note,))


def _bridge_ranking(reference, coeff, candidate, centres, shell_columns, *, probe_result):
    """``min_i I(candidate pair, shell of site i)`` -- or the total, where the sites do not split.

    ⚠ **A bridge orbital is one that talks to BOTH sites**, so the ranking is the *smaller* of
    its two mutual informations and never their sum: an orbital strongly entangled with one
    metal and orthogonal to the other is that metal's ligand, not a pathway. And it is a
    ranking rather than a verdict, because a low-dimensional bridge shares at most ``ln 2``
    with either ion -- small in absolute terms however real the pathway.

    Where the shell's orbitals are delocalized over equivalent centres (the symmetric and
    antisymmetric combinations of a pooled centre), no population split of them says which
    site is which; the ranking then falls back to the mutual information with the **whole**
    shell and says so. That degrades the order inside the class and cannot change the class's
    verdict, which is the spectrum's.
    """
    info = probe_result.pair_information()
    active = np.asarray(probe_result.space.spaces.active, dtype=int)
    pair_of = {int(c) // 2: i for i, c in enumerate(active[0::2])}
    columns = np.asarray(candidate.columns, dtype=int)
    rows = [pair_of[int(c) // 2] for c in columns[0::2]]
    split = _site_split(reference, coeff, centres, shell_columns)
    if split is None or len(split) < 2:
        shell_rows = [pair_of[int(c) // 2] for c in np.asarray(shell_columns, dtype=int)[0::2]]
        return np.asarray([float(np.sum(info[r, shell_rows])) for r in rows]), True
    per_site = []
    for site in split:
        site_rows = [pair_of[int(c) // 2] for c in np.asarray(site, dtype=int)[0::2]]
        per_site.append([float(np.sum(info[r, site_rows])) for r in rows])
    return np.min(np.asarray(per_site), axis=0), False


def assemble(reference, targets=None, *, solver: str = "ci",
             max_spinors: Optional[int] = None, max_determinants: Optional[int] = None,
             max_states: int = DEFAULT_MAX_STATES,
             spectrum_tol: float = DEFAULT_SPECTRUM_TOL,
             spectrum_tol_cm: float = DEFAULT_SPECTRUM_TOL_CM,
             probe_noise_cm: float = DEFAULT_PROBE_NOISE_CM,
             prune_rel: float = DEFAULT_PRUNE_REL,
             manifold_gap_cm: float = DEFAULT_MANIFOLD_GAP_CM,
             require=(), exclude=(), probe_budget: Optional[ProbeBudget] = None,
             report: bool = True) -> Assembly:
    """Assemble an active space from stated targets, deciding each class by the probe.

    The round loop of the module docstring. Returns an :class:`Assembly`: the space, the
    orbitals its columns index, the accepted classes, the rounds table, the final probe and
    the root proposal.

    ⚠ **This is the expensive call and it is several pre-optimizations.** One probe per round
    plus one per pruning that removed something; the rounds table records the CPU seconds of
    each, because a selection protocol whose cost is invisible is one nobody can bound.
    """
    from ..mcscf.adaptive import SolverFailure

    budget_probe = probe_budget or ProbeBudget()
    parsed = tg.parse_targets(targets)
    if not hasattr(reference, "ao_layout") and hasattr(reference, "reference"):
        # The stage and the container are easy to confuse and the duck-typed failure is an
        # AttributeError four frames down. (Measured: it cost a three-minute front end.)
        raise TypeError(
            "assemble() takes the finished SpinorReference container, not the Reference "
            "stage: pass stage.reference -- or use the kuiva.AutoCAS stage, which is the "
            "user surface and does this for you")
    layout = reference.ao_layout
    shell_targets = [t for t in parsed if t.cls == "shell"]
    if not shell_targets:
        raise ValueError(
            "every target set needs at least one shell: the ligand classes are defined "
            "relative to a shell's projection (a bonding partner is 'below the shell's cut', "
            "a bridge is 'mixed with the shell'), so there is nothing for them to be relative "
            "to. Add 'shells', or state the active space by hand")
    centres = _resolve_centres(reference, shell_targets, report=report)

    double_targets = [t for t in parsed if t.cls == "double"]
    bonding_targets = [t for t in parsed if t.cls == "bonding"]
    for target in double_targets:
        _check_named(centres, target, layout, "double-shell")
    for target in bonding_targets:
        _check_named(centres, target, layout, "bonding")
    bonding_pairs = max([int(t.n_pairs) for t in bonding_targets], default=0)

    coeff = reference.spinors_in_ao()
    occupation = np.asarray(reference.spinors.occ, dtype=float)
    construction = cand.shell_candidates(
        reference, centres, coeff=coeff, occupation=occupation,
        double=bool(double_targets), bonding=bonding_pairs, report=report)
    coeff = construction.coeff
    projection = _Projection(eigenvalues=np.asarray(construction.avas.eigenvalues, float),
                             occupations=np.asarray(construction.avas.occupations, float),
                             coeff=coeff,
                             selected=np.asarray(construction.avas.selected, dtype=int))
    shell = construction.shell

    # The banned columns' own descriptions are not carried into the space's statement: what
    # a space says is what it CONTAINS, and "everything except a Cl 3p pair" describes a
    # selection process rather than an active space. The exclusion is printed in the
    # candidate sets' notes, where it says which pairs it actually removed.
    banned, _ = _require_columns(reference, coeff, occupation, exclude, what="exclude")
    pinned, pinned_text = _require_columns(reference, coeff, occupation, require,
                                           what="require")
    claimed = set(int(c) for c in shell.columns) | set(int(c) for c in pinned)
    for extra in (construction.double, construction.bonding):
        if extra is not None and not extra.empty:
            claimed.update(int(c) for c in extra.columns)

    # ⚠ Construction order is priority order because the frontier class **rotates** the
    # orbitals: its rotation is confined to blocks of pairs whose AVAS projection is
    # degenerate, so the shell's selection and the projection spectrum survive it exactly,
    # while the bridge class -- which reads populations of the orbitals as they now are --
    # must be constructed after it, against the rotated set.
    ligand_sets: List[cand.CandidateSet] = []
    for target in [t for t in parsed if t.cls == "frontier"]:
        built = cand.frontier_candidates(reference, target, coeff=coeff,
                                         occupation=occupation, exclude=sorted(claimed),
                                         protect=projection.eigenvalues, report=report)
        coeff = built.coeff
        projection = _Projection(eigenvalues=projection.eigenvalues,
                                 occupations=projection.occupations, coeff=coeff,
                                 selected=projection.selected)
        claimed.update(int(c) for c in built.candidates.columns)
        ligand_sets.append(built.candidates)
    for target in [t for t in parsed if t.cls == "bridge"]:
        built = cand.bridge_candidates(reference, target, projection,
                                       exclude=sorted(claimed), report=report,
                                       barriers=sorted({int(a) for c in centres
                                                        for a in c.atoms}))
        claimed.update(int(c) for c in built.columns)
        ligand_sets.append(built)

    offered = [_drop_excluded(s, banned) for s in ligand_sets]
    if construction.bonding is not None:
        offered.append(_drop_excluded(construction.bonding, banned))
    if construction.double is not None:
        offered.append(_drop_excluded(construction.double, banned))
    offered.sort(key=lambda s: (s.priority, ))
    cand.check_disjoint([shell] + offered)
    if np.intersect1d(np.asarray(shell.columns, dtype=int), banned).size:
        raise ValueError("exclude= names Kramers pair(s) of the target shell, which is never "
                         "cut: '{}'".format(shell.description))

    floor, product_floor, terms = _floors(centres, shell)
    floor_target = int(max(1, min(product_floor, int(max_states))))
    budget = resolve_budget(solver=solver, n_elec=int(round(shell.electrons)),
                            n_roots=floor_target, max_spinors=max_spinors,
                            max_determinants=max_determinants)

    accepted: List[cand.CandidateSet] = [shell]
    space = _space_of(reference, accepted, extra_columns=pinned,
                      extra_description=pinned_text)
    if not budget.fits(space.n_active, space.n_elec):
        raise ValueError(
            "the core targets alone are {} spinors and the budget is {}: a shell is never "
            "cut to fit, because a shell with pairs missing is a different physical statement "
            "wearing the shell's name. The two ways out are the tensor-network solver "
            "(solver='dmrg'), whose ceiling is not the determinant count, and fewer or pooled "
            "centres. The core is: {}"
            .format(space.n_active, budget.describe(), space.description))

    _check_probe_budget(budget_probe, space, centres, shell)
    unmeasurable = _coupled_and_truncated(space, centres, budget_probe)

    # The floor plus a margin: a manifold boundary is only visible from the state above it,
    # and the probe averages over all of them (measured -- see the probe module).
    n_roots = probe_roots(floor_target, n_elec=space.n_elec, n_active=space.n_active,
                          margin=budget_probe.margin)
    rounds: List[RoundRecord] = []
    # ⚠ **Every verdict is taken at FIXED orbitals** -- the candidate construction's, one set
    # for the whole protocol -- and only the accepted space is pre-optimized, once, at the
    # end. Two pre-optimizations stopped on their budgets differ by what the addition did to
    # the trajectory: seven pairs that describe nothing moved a probe's target spectrum by up
    # to 102 cm^-1 and by at most 0.2 cm^-1 at fixed orbitals (see :func:`measure`).
    #
    # ⚠ And every trial is **nested** in the accepted space: its selected CI starts from the
    # accepted space's whole determinant list, and the budget bounds what the class adds.
    # A selection started afresh in the larger space dropped determinants the accepted one
    # held, and deep core pairs then "moved" a Ti2Cl6 target manifold by 3149 cm^-1.
    notes: List[str] = []
    current = None
    reference_spectrum = None
    if unmeasurable:
        notes.append(unmeasurable)
        log.warning("%s", unmeasurable)
    else:
        current = measure(reference, coeff, space, n_roots=n_roots, budget=budget_probe)
        reference_spectrum = target_spectrum(current.spectrum_cm, floor_target,
                                             gap_cm=manifold_gap_cm)
    rounds.append(RoundRecord(index=0, cls="shell", n_candidates=shell.n_pairs, n_pruned=0,
                              distance_cm=None, tolerance_cm=None, verdict="core",
                              n_spinors=space.n_active,
                              cpu_seconds=current.cpu_seconds if current is not None else 0.0,
                              note=terms))

    for index, candidate in enumerate(offered, start=1):
        if candidate.empty:
            rounds.append(RoundRecord(index=index, cls=candidate.cls, n_candidates=0,
                                      n_pruned=0, distance_cm=None, tolerance_cm=None,
                                      verdict="empty class", n_spinors=space.n_active,
                                      cpu_seconds=0.0, note=candidate.description))
            continue
        trial_sets = accepted + [candidate]
        trial = _space_of(reference, trial_sets, extra_columns=pinned,
                          extra_description=pinned_text)
        if not budget.fits(trial.n_active, trial.n_elec):
            rounds.append(RoundRecord(
                index=index, cls=candidate.cls, n_candidates=candidate.n_pairs, n_pruned=0,
                distance_cm=None, tolerance_cm=None, verdict="dropped (over budget)",
                n_spinors=space.n_active, cpu_seconds=0.0,
                note="the space would be {} spinors against a budget of {}".format(
                    trial.n_active, budget.max_spinors)))
            continue
        if unmeasurable:
            # ⚠ Kept because it was asked for, and said so: no CI is solved for a verdict
            # the measurement cannot give (see `_coupled_and_truncated`).
            accepted.append(candidate)
            space = trial
            rounds.append(RoundRecord(
                index=index, cls=candidate.cls, n_candidates=candidate.n_pairs, n_pruned=0,
                distance_cm=None, tolerance_cm=None, verdict="kept (not measurable)",
                n_spinors=space.n_active, cpu_seconds=0.0,
                note="requested and kept unmeasured: the core is a truncated space of "
                     "coupled centres"))
            continue
        try:
            trial_probe = measure(reference, coeff, trial, n_roots=n_roots,
                                  budget=budget_probe, seed=current)
        except SolverFailure as exc:
            # ⚠ An untested addition is not an accepted one. The round is marked and the run
            # continues on the last accepted space, with the failure printed.
            rounds.append(RoundRecord(
                index=index, cls=candidate.cls, n_candidates=candidate.n_pairs, n_pruned=0,
                distance_cm=None, tolerance_cm=None, verdict="dropped (not measured)",
                n_spinors=space.n_active, cpu_seconds=0.0,
                note="the probe failed to solve this space: {}".format(exc)))
            log.warning("the %s round could not be measured (%s); the class is kept OUT and "
                        "the run continues on the last accepted space", candidate.cls, exc)
            continue
        cpu = trial_probe.cpu_seconds
        kept_candidate, prune_note = candidate, ""
        n_pruned = 0
        if not candidate.fixed:
            ranking, pooled = (None, True)
            if candidate.cls == "bridge":
                ranking, pooled = _bridge_ranking(reference, coeff, candidate, centres,
                                                  shell.columns, probe_result=trial_probe)
                if pooled:
                    notes.append(
                        "the bridge candidates were ranked by their mutual information with "
                        "the WHOLE shell: the shell's orbitals are delocalized over the "
                        "equivalent centres, so no population split of them says which site "
                        "is which. The ranking only orders the class; the verdict is the "
                        "spectrum's")
            kept_columns, prune_note = prune_candidates(candidate, trial_probe,
                                                        prune_rel=prune_rel, ranking=ranking)
            n_pruned = candidate.n_pairs - int(np.size(kept_columns)) // 2
            if n_pruned:
                from dataclasses import replace as _replace

                # ⚠ ``occupations`` and ``ranking`` carry one value per **Kramers pair**
                # while ``columns`` carries two, so the mask is built on the pairs. Masking
                # the pair arrays with a column mask silently halves them where it does not
                # simply raise.
                keep = np.isin(np.asarray(candidate.columns, dtype=int)[0::2] // 2,
                               np.asarray(kept_columns, dtype=int)[0::2] // 2)
                kept_candidate = _replace(
                    candidate, columns=np.asarray(kept_columns, dtype=int),
                    occupations=np.asarray(candidate.occupations)[keep],
                    ranking=np.asarray(candidate.ranking)[keep],
                    fragments=(tuple(int(c) for c in kept_columns),),
                    notes=tuple(candidate.notes) + ((prune_note,) if prune_note else ()))
        if kept_candidate.empty:
            rounds.append(RoundRecord(
                index=index, cls=candidate.cls, n_candidates=candidate.n_pairs,
                n_pruned=n_pruned, distance_cm=None, tolerance_cm=None,
                verdict="dropped (all pruned)", n_spinors=space.n_active, cpu_seconds=cpu,
                note=prune_note))
            continue
        if n_pruned:
            trial_sets = accepted + [kept_candidate]
            trial = _space_of(reference, trial_sets, extra_columns=pinned,
                              extra_description=pinned_text)
            trial_probe = measure(reference, coeff, trial, n_roots=n_roots,
                                  budget=budget_probe, seed=current)
            cpu += trial_probe.cpu_seconds
        trial_spectrum = target_spectrum(trial_probe.spectrum_cm, floor_target,
                                         gap_cm=manifold_gap_cm)
        distance, structure_note = spectrum_distance(reference_spectrum, trial_spectrum)
        width = (float(reference_spectrum[:-1].max()) if reference_spectrum.size > 1
                 else 0.0)
        tolerance = max(float(spectrum_tol) * width, float(spectrum_tol_cm))
        if distance > tolerance:
            verdict = "kept"
        elif distance > float(probe_noise_cm):
            verdict = "kept (inconclusive)"
        else:
            verdict = "dropped"
        note = structure_note or prune_note
        if verdict.startswith("kept"):
            accepted.append(kept_candidate)
            space, current = trial, trial_probe
            reference_spectrum = trial_spectrum
        rounds.append(RoundRecord(
            index=index, cls=candidate.cls, n_candidates=candidate.n_pairs,
            n_pruned=n_pruned, distance_cm=None if not np.isfinite(distance) else distance,
            tolerance_cm=tolerance, verdict=verdict, n_spinors=space.n_active,
            cpu_seconds=cpu, note=note))
        if verdict == "kept (inconclusive)":
            log.warning("the %s class moved the target manifold by %.2f cm^-1, under the "
                        "tolerance of %.2f but above the probe's noise floor of %.2f: it is "
                        "KEPT, because a larger space is the safe error and the budget bounds "
                        "it. This is the honest outcome where the physics is below what a "
                        "cheap CI resolves", candidate.cls, distance, tolerance,
                        probe_noise_cm)

    # The one pre-optimization: the orbitals handed downstream and the spectrum the proposal
    # reads. ⚠ A failure here propagates -- there is no accepted space left to fall back to,
    # and a stage that returned construction orbitals labelled as probed ones would be lying
    # about what a CASSCF starts from.
    decided = current
    current = probe(reference, coeff, space, n_roots=n_roots, budget=budget_probe)
    proposal = propose_roots(current.spectrum_cm, floor, product_floor=product_floor,
                             gap_cm=manifold_gap_cm, max_states=max_states)
    # ⚠ The loop the budget has with the proposal, closed by measuring twice: the budget was
    # resolved at the FLOOR root count (the proposal did not exist yet) and the conventional-CI
    # ceiling moves with the root count, so a space that fits at the floor need not fit at the
    # proposal. Both numbers are reported rather than one of them quietly winning.
    if proposal.n_states is not None:
        recheck = resolve_budget(solver=solver, n_elec=space.n_elec,
                                 n_roots=int(proposal.n_states), max_spinors=max_spinors,
                                 max_determinants=max_determinants)
        if not recheck.fits(space.n_active, space.n_elec):
            notes.append(
                "the space fits the budget at the floor of {} roots and NOT at the proposed "
                "{} ({}): the root count co-decides the conventional-CI ceiling. Lower the "
                "count, or run the tensor-network solver"
                .format(floor_target, proposal.n_states, recheck.describe()))
            log.warning("%s", notes[-1])

    assembly = Assembly(space=space, coeff=current.coeff, sets=tuple(accepted),
                        rounds=rounds, probe=current, centres=centres, floor=floor,
                        product_floor=product_floor, budget=budget, proposal=proposal,
                        site_atoms=_centre_atoms(centres), notes=tuple(notes),
                        decided=decided)
    if report:
        report_assembly(assembly)
    return assembly


def report_assembly(assembly: Assembly, logger=None) -> None:
    """The ``[automatic active space]`` block: the core, the rounds, the budget, the proposal.

    Through the output grammar only, and every keep and drop carries the number it was decided
    on: a protocol whose verdicts are not printed with their evidence is one nobody can check.
    """
    logger = logger or log
    out.section(logger, "Automatic active space")
    out.entry(logger, "centres", ", ".join(
        "{} ({}, {:.0f} e)".format(c.where, tg.angular_momentum_letter(c.l), c.open_electrons)
        for c in assembly.centres) or "none")
    out.entry(logger, "core", "{} pair(s)".format(assembly.sets[0].n_pairs), "",
              assembly.sets[0].description)
    table = out.Table(logger, [
        out.col_count("round", 6), out.Column("class", "{}", 10, "<"),
        out.col_count("cand", 6), out.col_count("pruned", 7),
        out.Column("d [cm^-1]", out.CM_FMT, 10), out.Column("tol", out.CM_FMT, 8),
        out.Column("verdict", "{}", 22, "<"), out.col_count("spinors", 8),
        out.Column("cpu [s]", out.TIME_FMT, 8)]).start()
    for record in assembly.rounds:
        table.row(record.index, record.cls, record.n_candidates, record.n_pruned,
                  record.distance_cm, record.tolerance_cm, record.verdict,
                  record.n_spinors, record.cpu_seconds)
    table.end("d is the largest change of the target manifold's relative energies and its "
              "gap, both measured at the candidate construction's orbitals; a class is kept "
              "above the tolerance, kept as inconclusive above the noise floor, and dropped "
              "below it")
    for record in assembly.rounds:
        if record.note:
            out.note(logger, "round {}: {}".format(record.index, record.note))
    out.entry(logger, "budget", assembly.budget.describe(), "",
              "provisional" if assembly.budget.provisional else "")
    out.entry(logger, "final probe", assembly.probe.cpu_seconds, "s cpu",
              "pre-optimized once, on the accepted space ({} determinants)".format(
                  assembly.probe.n_determinants), out.TIME_FMT)
    assembly.proposal.report(logger)
    out.entry(logger, "active space", "CAS({}, {})".format(assembly.space.n_elec,
                                                           assembly.n_active))
    out.note(logger, "statement: {}".format(assembly.space.description))
    for note in assembly.notes:
        out.note(logger, note)


def shell_columns_after_probe(reference, assembly: Assembly, *,
                              threshold: float = cand.DEFAULT_FRAGMENT_THRESHOLD):
    """The assembled space's **shell-like** active columns, in the probe's own orbitals.

    ⚠ The shell's columns before the probe are not the shell's columns after it: the
    pre-optimization rotates inside the active space and returns natural spinors, so the
    selection's index list means nothing there. What survives is the *character*, so the
    shell-like columns are measured again -- the active pairs carrying at least ``threshold``
    of their Loewdin population on the centres' atoms.

    Returns ``(columns, note)``; ``columns`` is ``None`` when the count does not come out as
    the shell's own size, which is a statement about the orbitals (the pre-optimization mixed
    the shell with something else) and not an error.
    """
    atoms = sorted({int(a) for c in assembly.centres for a in c.atoms})
    active = np.asarray(assembly.space.spaces.active, dtype=int)
    population = cand._fragment_pair_populations(reference, assembly.coeff, atoms)
    pairs = active[0::2] // 2
    keep = pairs[population[pairs] >= float(threshold)]
    wanted = sum(c.n_pairs for c in assembly.centres)
    if keep.size != wanted:
        return None, ("{} active Kramers pair(s) carry {:.0%} of their population on the "
                      "centres after the pre-optimization, against a shell of {}: the "
                      "orbitals do not separate into a shell and the rest, so no site "
                      "partition is claimed".format(int(keep.size), threshold, wanted))
    return cand._pair_columns(keep), ""
