"""Integral factorization and the AO -> spinor-MO transformation.

Density fitting and Cholesky decomposition are exposed through **one** interface
(:class:`~kuiva.integrals.transform.ThreeIndexAO`), because downstream code only ever needs
"a set of real three-index factors whose square reproduces the ERIs".
"""
from .transform import (SpinorMOIntegrals, ThreeIndexAO, assemble_4c, coulomb_exchange,
                        pivoted_cholesky, shell_pair_orbits, transform_1e, transform_3c)

__all__ = ["ThreeIndexAO", "SpinorMOIntegrals", "transform_3c", "transform_1e",
           "assemble_4c", "coulomb_exchange", "pivoted_cholesky", "shell_pair_orbits"]
