"""Automatic active-space selection on the cheap CI.

What this package does
----------------------
It assembles an active space from **stated targets** -- physical statements of what the
calculation has to describe -- resolves each one against the reference and its free-atom
orbitals, probes the candidate spaces with the cheap CI, and hands the production stage a
stated active space and a proposed state count. The user surface is the ``AutoCAS`` stage;
everything here is the machinery behind it.

Modules
-------
:mod:`kuiva.autocas.targets`
    The vocabulary: the five target classes, their spellings, their fixed priority, and the
    one atom-addressing boundary between 1-based statements and internal indices.
:mod:`kuiva.autocas.multiplets`
    Hund ground terms of an ``l^n`` shell and the theoretical **floor** of a state count --
    the level ``2J+1`` on the f block, the spin multiplicity ``2S+1`` on the d block, and
    the product for coupled centres.
:mod:`kuiva.autocas.centres`
    Which atoms are the magnetic centres, **measured**: a frontier population *and* an open
    shell in the atom's own free-atom reference, with equivalent centres pooled.
:mod:`kuiva.autocas.candidates`
    One constructor per target class, each producing whole Kramers pairs, whole degenerate
    groups and a description another program could reproduce.
:mod:`kuiva.autocas.probe`
    The cheap CI as a measuring instrument: one bounded pre-optimization of one candidate
    space, and the qualitative spectrum it leaves behind.
:mod:`kuiva.autocas.protocol`
    The round loop, the entropy pruning, the spectrum test, the size budget and the rounds
    table -- everything that decides which classes stay.
:mod:`kuiva.autocas.roots`
    Where a ground manifold ends, and the state count and energy window proposed from it.

Rules that bind everything here
-------------------------------
* ⚠ **The dependency runs one way, asserted from the sources.** ``kuiva.autocas`` imports
  the core packages; **nothing in the calculation path imports** ``kuiva.autocas``. The
  stage reaches it lazily, exactly as ``CheapCI`` reaches the pre-optimizer. Selection is a
  step *before* a calculation, and a back-edge would put it inside one.
* ⚠ **No index list ever leaves a description.** A target resolves to columns of one
  particular orbital set; what is *stated* is the physical selection rule, because that is
  the only form another implementation can reproduce and the only form a stored product can
  carry.
* ⚠ **Every shell-like candidate is an AVAS projection onto the free-atom reference**, so
  ``atomic_reference=True`` is required. Character selection is not offered as a fallback:
  the same character statement has been measured selecting a 1s core pair in one basis and
  not in another, while the AVAS selection was basis-stable -- and two constructions of "the
  shell" would be two definitions of it.
* ⚠ **A target shell is never pruned and never cut.** Its empty members are the ligand-field
  spectrum; an occupation or entropy criterion that drops them is measuring the probe.
* ``import kuiva.autocas`` stays cheap: the names below resolve lazily and no module here
  imports the front end at module scope.

Designs decided against (recorded so they are not re-proposed)
---------------------------------------------------------------
* **Selecting by entropy thresholds alone**, the published automated route: blind to a
  correlating shell and to the empty members of a d manifold, and to a low-dimensional bridge
  in absolute terms. Entropy prunes inside one class; the probe's spectrum decides.
* **Growing the whole candidate pool at once and pruning down.** One probe instead of several,
  but nothing then says *which* class moved the spectrum, and a size budget would have to drop
  by entropy across classes -- a number from a qualitative probe deciding between a mechanism
  and a refinement. The rounds attribute every change.
* **Inferring a shell's electron count from the configuration statement.** It defaults to the
  neutral atom outside the f block; the count is measured off the reference occupations and
  the statement only cross-checked -- except for a shell entirely empty or full in the
  reference, where an ion-level statement (stated, or the f-block M(3+) default) supplies it.
* **Re-selecting the space inside the CASSCF.** A change of active space is a change of
  calculation, not of chart, and a variational adoption rule cannot govern it (a larger space
  is lower by construction). Selection happens once, before the production stage.
* **Cutting a shell to fit the budget**, or a count-only / window-only proposal: the first is
  a different physical statement wearing the shell's name; the second is decided by the
  consumers -- the count is what the state-averaging gate takes, the window what survives the
  production solver's own spectrum moving -- so both are proposed.
* **The Fiedler order as the tensor-network ordering of a polynuclear space.** The site
  partition orders; mutual information only ranks inside a class.
* **Occupancy restrictions (RAS-like) on bridge orbitals**: outside the current scope; a
  bridge is ordinary CAS orbitals here.

References
----------
* E. R. Sayfutyarova, Q. Sun, G. K.-L. Chan, G. Knizia, J. Chem. Theory Comput. 13, 4063
  (2017), doi:10.1021/acs.jctc.7b00128 -- AVAS, the shell construction's projector (cited at
  the point of implementation in :mod:`kuiva.mcscf.avas`, whose count-stated mode this
  package needs).
* C. J. Stein, M. Reiher, J. Chem. Theory Comput. 12, 1760 (2016),
  doi:10.1021/acs.jctc.6b00156; C. J. Stein, M. Reiher, Chimia 71, 170 (2017),
  doi:10.2533/chimia.2017.170 -- the entanglement-based automated selection protocol whose
  relative single-orbital-entropy criterion is used here to **prune** candidates.
* F. Hund, Z. Phys. 33, 345 (1925), doi:10.1007/BF01328319 -- the ground-term rules behind
  the state-count floor.
"""

#: Lazily resolved public names, ``name -> module`` (PEP 562), so importing this package
#: costs nothing to a run that never asks for an automatic active space. ⚠ A name here may
#: not be one of the submodule names above: importing a submodule binds it on the package, so
#: an entry like ``probe`` would resolve to the *module* and never reach this map. The probe
#: entry point is therefore reached as ``kuiva.autocas.probe.probe``.
_LAZY = {
    "Bonding": "kuiva.autocas.targets",
    "Bridge": "kuiva.autocas.targets",
    "CandidateSet": "kuiva.autocas.candidates",
    "Centre": "kuiva.autocas.centres",
    "CentreDetection": "kuiva.autocas.centres",
    "DoubleShell": "kuiva.autocas.targets",
    "Frontier": "kuiva.autocas.targets",
    "HundTerm": "kuiva.autocas.multiplets",
    "LigandConstruction": "kuiva.autocas.candidates",
    "ManifoldBoundary": "kuiva.autocas.roots",
    "ProbeBudget": "kuiva.autocas.probe",
    "ProbeResult": "kuiva.autocas.probe",
    "RootProposal": "kuiva.autocas.roots",
    "RoundRecord": "kuiva.autocas.protocol",
    "Shell": "kuiva.autocas.targets",
    "ShellConstruction": "kuiva.autocas.candidates",
    "assemble": "kuiva.autocas.protocol",
    "bonding_candidates": "kuiva.autocas.candidates",
    "bridge_candidates": "kuiva.autocas.candidates",
    "check_disjoint": "kuiva.autocas.candidates",
    "coupled_floor": "kuiva.autocas.multiplets",
    "detect_centres": "kuiva.autocas.centres",
    "fragment_rotation": "kuiva.autocas.candidates",
    "frontier_candidates": "kuiva.autocas.candidates",
    "hund_ground_term": "kuiva.autocas.multiplets",
    "manifold_boundary": "kuiva.autocas.roots",
    "parse_targets": "kuiva.autocas.targets",
    "propose_roots": "kuiva.autocas.roots",
    "resolve_budget": "kuiva.autocas.protocol",
    "shell_candidates": "kuiva.autocas.candidates",
    "shell_centre": "kuiva.autocas.centres",
    "shell_floor": "kuiva.autocas.multiplets",
    "target_spectrum": "kuiva.autocas.roots",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    import importlib

    value = getattr(importlib.import_module(_LAZY[name]), name)
    globals()[name] = value                      # resolved once; later reads are plain lookups
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
