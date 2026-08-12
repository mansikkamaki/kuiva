"""Orthonormal working basis: canonical orthogonalization and linear-dependence removal.

This subpackage owns the load-bearing boundary of the architecture — everything downstream of
here works in an **orthonormal** basis, and the segmented/general/uncontracted contraction
distinction does not survive past it.
"""
from .canonical import (OrthonormalBasis, canonical_orthogonalization, orthogonalize,
                        DEFAULT_THRESHOLD)

__all__ = ["OrthonormalBasis", "canonical_orthogonalization", "orthogonalize",
           "DEFAULT_THRESHOLD"]
