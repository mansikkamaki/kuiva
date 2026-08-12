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

Still to come: the Kramers-restricted (time-reversal adapted) path, which is validated
*against* the general-complex one and therefore cannot come first.
"""
from . import kernels
from . import sigma
from .davidson import DavidsonResult, davidson, davidson_workspace_gb
from .sigma import SigmaOperator, sigma_vector, sigma_workspace_gb
from .strings import (CASSpace, Determinants, DEFAULT_MAX_SPINORS, apply_ladder,
                      binomial_table, cas_dimension, cas_vector_gb, connections,
                      diagonal_energies, excitation_map_gb, hamiltonian_matrix, ladder_map,
                      occupation_matrix, rdm12, single_excitation_operator)

__all__ = ["CASSpace", "Determinants", "DavidsonResult", "SigmaOperator",
           "apply_ladder", "ladder_map",
           "binomial_table", "cas_dimension", "davidson", "davidson_workspace_gb",
           "cas_vector_gb", "connections", "diagonal_energies", "excitation_map_gb",
           "hamiltonian_matrix", "kernels", "occupation_matrix", "rdm12", "sigma",
           "sigma_vector", "sigma_workspace_gb", "single_excitation_operator",
           "DEFAULT_MAX_SPINORS"]
