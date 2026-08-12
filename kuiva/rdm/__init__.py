"""Reduced density matrices and quantities derived from them.

``rdm`` builds the state-averaged 1- and 2-particle density matrices of a **complete CAS
space** from the sigma vector's own intermediate (one excitation map,
one intermediate, three consumers), and is where state averaging and Kramers-block
completeness are imposed. ``ci.strings.rdm12`` remains the independent implementation
for an arbitrary determinant list — the selected CI — and is what this one is
validated against. 3- and 4-RDMs belong to SC-NEVPT2 and do not exist yet.

``entropy`` holds the orbital-entanglement measures that drive DMRG orbital ordering
and active-space selection.
"""
from .entropy import (fiedler_order, mutual_information, single_orbital_entropy,
                      two_orbital_entropy)
from .rdm import (RDMBuilder, active_space_energy, cas_rdms, degenerate_blocks,
                  rdm_workspace_gb, state_average_weights)

__all__ = ["single_orbital_entropy", "two_orbital_entropy", "mutual_information",
           "fiedler_order", "RDMBuilder", "cas_rdms", "active_space_energy",
           "state_average_weights", "degenerate_blocks", "rdm_workspace_gb"]
