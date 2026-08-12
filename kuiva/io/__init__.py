"""Persistence: HDF5 checkpoints and restart.

One module so far. ``checkpoint`` is the schema-versioned restart file the CASSCF driver
writes every macro-iteration and reads back to resume — orbitals, orbital-rotation state,
RDMs, state energies, provenance, and the CI vectors when they are small enough to be worth
keeping.

⚠ **This is not a cache, and its failure semantics are the opposite of one** (one rule governs
the X2CAMF correction cache, not this). A **write** failure is a ``WARNING`` and the run
continues, because losing a restart point is not losing the calculation. A **read** failure on
an explicitly requested restart is an ``ERROR`` that propagates, because the user asked for it
and silently starting over wastes the hours the checkpoint existed to protect.
"""
from .checkpoint import (CASSCFCheckpoint, CheckpointPolicy, SCHEMA_VERSION,
                         checkpoint_size_gb, read_checkpoint, write_checkpoint)

__all__ = ["CASSCFCheckpoint", "CheckpointPolicy", "SCHEMA_VERSION", "checkpoint_size_gb",
           "read_checkpoint", "write_checkpoint"]
