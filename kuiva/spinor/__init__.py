"""Scalar MO -> two-component spinor expansion.

Owns the spinor ordering and time-reversal conventions, which propagate into CI addressing
 and must therefore be fixed in exactly one place. See :mod:`kuiva.spinor.expand`.
"""
from .density import (DensityComponents, decompose_density, decompose_spinor_density,
                      kramers_pair_density)
from .expand import (SpinorBasis, barred, expand_scalar_mos, expand_unrestricted_mos,
                     is_barred, is_time_reversal_even, kramers_block_permutation,
                     spatial_index, spin_block_diagonal, spin_operator, spinor_indices,
                     time_reverse, two_component_operator, unbarred)

__all__ = ["SpinorBasis", "DensityComponents", "decompose_density",
           "decompose_spinor_density", "kramers_pair_density",
           "expand_scalar_mos", "expand_unrestricted_mos",
           "time_reverse", "spin_block_diagonal", "spin_operator",
           "two_component_operator", "is_time_reversal_even", "kramers_block_permutation",
           "spinor_indices", "spatial_index", "barred", "unbarred", "is_barred"]
