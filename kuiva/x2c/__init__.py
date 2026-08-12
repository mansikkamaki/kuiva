"""Exact two-component (X2C) decoupling: the primitives every X2C path shares.

This package holds the linear algebra of the X2C transformation itself, independent of what
is being transformed and of where the four-component blocks came from. Two callers use it:

*:mod:`kuiva.amf` — the atomic mean-field two-electron picture change,
  which applies the transformation to a converged four-component *atomic* mean field;
* the molecular one-electron path of :mod:`kuiva.interface.pyscf_bridge`.

**The dependency runs one way: ``kuiva.amf`` and ``kuiva.interface`` import from here, and
nothing here imports from either.** That is the whole reason the package exists — the
decoupling is not an atomic-mean-field concept, and a shared primitive living inside the
package of its first consumer makes every later consumer look like a special case of it.
"""
from .decouple import (FourComponentBlocks, LIGHT_SPEED, METRIC_LINDEP_THRESHOLD,
                       blocks_memory_gb, canonical_orth, decoupling_matrices,
                       decoupling_memory_gb, exact_decoupling_workspace_gb, picture_change,
                       renormalization, two_component_density)
from .local import (LOCAL_SOURCES, Partition, check_local_blocks, local_block_scales,
                    local_decoupling_matrices, off_block_weight, sub_blocks)

__all__ = ["FourComponentBlocks", "LIGHT_SPEED", "LOCAL_SOURCES", "METRIC_LINDEP_THRESHOLD",
           "Partition", "blocks_memory_gb", "canonical_orth", "check_local_blocks",
           "decoupling_matrices", "decoupling_memory_gb", "exact_decoupling_workspace_gb",
           "local_block_scales", "local_decoupling_matrices", "off_block_weight",
           "picture_change", "renormalization", "sub_blocks", "two_component_density"]
