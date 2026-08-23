"""Determinant CI machinery.

Two determinant spaces, both addressed in ``ci/strings.py`` and nowhere else:

* an **arbitrary list** (:class:`~kuiva.ci.strings.Determinants`) with a hash lookup, which is
  what a truncated or selected CI needs (:mod:`kuiva.mcscf.preopt`), together with the
  ``O(N^2)`` connection search, the sparse Hamiltonian and the RDMs built on it;
* a **complete CAS space** (:class:`~kuiva.ci.strings.CASSpace`), addressed by combinatorial
  rank and carrying the two rectangular single-excitation tables the full CI runs on.

:mod:`kuiva.ci.sigma` is the matrix-free full-CI sigma vector over the second of those, and
:mod:`kuiva.ci.kernels` is the dispatch shim through which every hot kernel is reached, so a
compiled backend can be registered later without touching a caller or a test.

:mod:`kuiva.ci.davidson` is the block complex-Hermitian eigensolver that drives it. The 1- and
2-RDMs built from the same intermediate live in :mod:`kuiva.rdm.rdm`, one layer up, because
they are consumed by the orbital optimizer rather than by the CI.

**Two symmetry modes.** The general complex (Kramers-unrestricted) path above is the reference
path and is the default everywhere, forever: every committed reference in this project is a
general-complex result. Beside it, for an **odd** electron count,
:func:`~kuiva.ci.davidson.davidson_kramers` is the time-reversal-adapted path — the same
eigensolver over a subspace kept closed under ``T`` (:class:`~kuiva.ci.strings.KramersMap`),
which halves the stored subspace and the applications of ``H`` for the same spectrum and
returns pair-expanded states in the general convention. It is selected explicitly, with
``FullCISolver(kramers="restricted")``, and announces itself.

**And a third restriction, orthogonal to those two**: with abelian double-group labels on the
orbitals (:mod:`kuiva.symm`) the Hamiltonian is block diagonal over irrep sectors, and
:func:`~kuiva.ci.davidson.davidson_sector` solves one block. ⚠ It restricts the
**eigenproblem**, not the operator: the sigma vector stays a full-space contraction and every
registered kernel is untouched, so an application of ``H`` costs the full-space price whatever
the sector's size. The consequence a caller cannot see and must not re-derive is that the
dense-solve fallback is decided on the **full** determinant count, and that is done here.
"""
from . import kernels
from . import sigma
from .davidson import (DavidsonResult, davidson, davidson_kramers,
                       davidson_kramers_sector, davidson_sector,
                       davidson_workspace_gb)
from .sigma import (SigmaOperator, assert_time_reversal, sigma_vector, sigma_workspace_gb,
                    time_reversal_violation)
from .strings import (CASSpace, Determinants, DEFAULT_MAX_SPINORS, KramersMap, apply_ladder,
                      binomial_table, cas_dimension, cas_vector_gb, connections,
                      diagonal_energies, excitation_map_gb, hamiltonian_matrix,
                      kramers_map_gb, kramers_partner, kramers_representative, kramers_sign,
                      ladder_map, occupation_matrix, rdm12, single_excitation_operator)

__all__ = ["CASSpace", "Determinants", "DavidsonResult", "KramersMap", "SigmaOperator",
           "apply_ladder", "assert_time_reversal", "ladder_map",
           "binomial_table", "cas_dimension", "davidson", "davidson_kramers",
           "davidson_kramers_sector",
           "davidson_sector", "davidson_workspace_gb",
           "cas_vector_gb", "connections", "diagonal_energies", "excitation_map_gb",
           "hamiltonian_matrix", "kernels", "kramers_map_gb", "kramers_partner",
           "kramers_representative", "kramers_sign",
           "occupation_matrix", "rdm12", "sigma",
           "sigma_vector", "sigma_workspace_gb", "single_excitation_operator",
           "time_reversal_violation", "DEFAULT_MAX_SPINORS"]
