"""Multiconfigurational SCF: the shared orbital optimizer and the cheap pre-optimizer.

``orbopt`` is the **single** state-averaged orbital optimizer — RDMs in,
orbital rotation out — consumed identically by the cheap CI here, by conventional-CI CASSCF
and by DMRG-CASSCF. ``preopt`` is the cheap pre-optimization CI.

``adaptive`` and ``events`` extend that contract to solvers whose *internal space* adapts (a
selected CI, later a DMRG): ``adaptive`` defines the four-method protocol, ``events`` the
outer controller that owns the space changes. The step engines in ``orbopt`` are unchanged and
are shared by both drivers.

``casci`` is the full CI presented as that same callback, plus the CASCI/CASSCF
drivers and the reproducible active-space selection. It lives here rather than in ``ci/`` because
``preopt`` imports ``ci``, so the dependency may not run the other way.
"""
from .adaptive import (AdaptiveCISolver, Proposal, SolverFailure, StaticSolver,
                       as_adaptive_solver)
from .casci import (ActiveSpace, CASCIResult, CASSCFOutcome, FullCISolver, active_space,
                    active_space_by_character, casci, casscf)
from .events import EventCASSCFResult, EventRecord, optimize_orbitals_events
from .orbopt import (AHResult, CASIntegrals, OrbitalHessian, OrbitalOptimizer, OrbitalSpaces,
                     augmented_hessian_step, averaged_fock, cas_energy, fock_diagonal,
                     generalized_fock,
                     optimize_orbitals,
                     unitary_from_antihermitian)
from .preopt import (CheapCIResult, CheapCISolver, PreoptResult, SPACE_POLICIES, cheap_ci,
                     preoptimize)

__all__ = ["OrbitalSpaces", "CASIntegrals", "OrbitalOptimizer", "optimize_orbitals",
           "OrbitalHessian", "augmented_hessian_step", "AHResult",
           "cas_energy", "generalized_fock", "averaged_fock", "fock_diagonal",
           "unitary_from_antihermitian",
           "cheap_ci", "preoptimize", "CheapCIResult", "PreoptResult", "SPACE_POLICIES",
           "AdaptiveCISolver", "Proposal", "SolverFailure", "StaticSolver",
           "as_adaptive_solver", "CheapCISolver",
           "optimize_orbitals_events", "EventCASSCFResult", "EventRecord",
           "FullCISolver", "CASCIResult", "CASSCFOutcome", "ActiveSpace", "active_space",
           "active_space_by_character", "casci", "casscf"]
