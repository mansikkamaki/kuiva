"""Second-order perturbation theory on the two-component multireference reference.

``nevpt2`` is the driver and the Dyall zeroth-order Hamiltonian; ``classes`` holds the eight
excitation classes behind a name registry and their spinor working equations; ``contractions``
serves the active-space density primitives; ``blocks`` the integral blocks, on demand and never
as a stored four-index array.

⚠ **The dependency runs one way**: this package imports ``ci/``, ``mcscf/``, ``rdm/``,
``integrals/`` and ``props/``, and nothing in the calculation path imports it. Corrected
energies reach the property dump only because ``corrected_property_matrices`` hands them to it.

All eight excitation classes produce an energy on the conventional-CI route, so its ``E2``
is the complete strongly contracted correction; frozen-core and deleted-virtual selections
are options and are stated as orbital energies, never as counts. ``network`` is the
tensor-network reference's provider and driver (``sc_nevpt2_dmrg``) — ⚠ six of the eight
classes today, so its ``E2`` is PARTIAL and says so (see that module's docstring for the
recorded scope and the missing piece).

``checkpoint`` is the per-class restart point: the driver's granularity is one
``(state, class)`` pair and a :class:`~kuiva.pt.classes.ClassResult` is eight scalars, so the
whole table is kilobytes whatever the active space. It stores no reference — the orbitals and
the CI vectors belong to the CASSCF checkpoint — only a digest of one, and a restart that is
not a continuation of the run in the file is refused rather than merged into it.
"""
from .blocks import IntegralBlocks, batch_slices
from .classes import (ClassContext, ClassResult, ExcitationClass, available_classes,
                      excitation_class, implemented_classes, register_class)
from .contractions import (CIContractionProvider, ShiftedSpace, ShiftedSpaces,
                           hole_pair_matrix, hole_rdm1, koopmans_annihilation,
                           koopmans_creation, pair_matrix)
from .network import NetworkContractionProvider, sc_nevpt2_dmrg
from .checkpoint import (NEVPT2Checkpoint, NEVPT2CheckpointPolicy, read_nevpt2_checkpoint,
                         write_nevpt2_checkpoint)
from .nevpt2 import (DEFAULT_MULTIPLET_TOL_CM, MULTIPLET_TOL_CM, CanonicalOrbitals,
                     CorrelatedSpaces, MultipletCorrection, NEVPT2Result,
                     assemble_from_checkpoint, corrected_property_matrices,
                     pseudo_canonicalize, sc_nevpt2, select_correlated)

__all__ = ["sc_nevpt2", "sc_nevpt2_dmrg", "NetworkContractionProvider", "NEVPT2Result",
           "NEVPT2Checkpoint", "NEVPT2CheckpointPolicy", "assemble_from_checkpoint",
           "read_nevpt2_checkpoint", "write_nevpt2_checkpoint",
           "CanonicalOrbitals", "pseudo_canonicalize",
           "CorrelatedSpaces", "select_correlated", "corrected_property_matrices",
           "MultipletCorrection", "MULTIPLET_TOL_CM", "DEFAULT_MULTIPLET_TOL_CM",
           "IntegralBlocks", "batch_slices",
           "ClassContext", "ClassResult", "ExcitationClass", "available_classes",
           "excitation_class", "implemented_classes", "register_class",
           "CIContractionProvider", "ShiftedSpace", "ShiftedSpaces",
           "hole_rdm1", "hole_pair_matrix", "pair_matrix",
           "koopmans_annihilation", "koopmans_creation"]
