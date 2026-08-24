"""Orthonormal working basis: canonical orthogonalization and linear-dependence removal.

This subpackage owns the load-bearing boundary of the architecture — everything downstream of
here works in an **orthonormal** basis, and the segmented/general/uncontracted contraction
distinction does not survive past it.

It also owns the *other* change of basis a calculation can make: carrying a converged orbital
set from one basis set onto another (:mod:`kuiva.orth.project`), which is what lets a CASSCF
be converged cheaply in a small basis and continued in the production one.
"""
from .canonical import (OrthonormalBasis, canonical_orthogonalization, orthogonalize,
                        DEFAULT_THRESHOLD)
from .project import (BasisProjection, SCHEMES, project_scalar_orbitals,
                      project_spinors)

__all__ = ["OrthonormalBasis", "canonical_orthogonalization", "orthogonalize",
           "DEFAULT_THRESHOLD", "BasisProjection", "project_spinors",
           "project_scalar_orbitals", "SCHEMES"]
