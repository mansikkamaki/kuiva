"""Second-order perturbation theory on the two-component multireference reference.

``nevpt2`` is the driver and the Dyall zeroth-order Hamiltonian; ``classes`` holds the eight
excitation classes behind a name registry and their spinor working equations; ``contractions``
serves the active-space density primitives; ``blocks`` the integral blocks, on demand and never
as a stored four-index array.

⚠ **The dependency runs one way**: this package imports ``ci/``, ``mcscf/``, ``rdm/``,
``integrals/`` and ``props/``, and nothing in the calculation path imports it. Corrected
energies reach the property dump only because ``corrected_property_matrices`` hands them to it.

All eight excitation classes produce an energy, so ``E2`` is the complete strongly contracted
correction; frozen-core and deleted-virtual selections are options and are stated as orbital
energies, never as counts.
"""
from .blocks import IntegralBlocks, batch_slices
from .classes import (ClassContext, ClassResult, ExcitationClass, available_classes,
                      excitation_class, implemented_classes, register_class)
from .contractions import (CIContractionProvider, ShiftedSpace, ShiftedSpaces,
                           hole_pair_matrix, hole_rdm1, koopmans_annihilation,
                           koopmans_creation, pair_matrix)
from .nevpt2 import (DEFAULT_MULTIPLET_TOL_CM, MULTIPLET_TOL_CM, CanonicalOrbitals,
                     CorrelatedSpaces, MultipletCorrection, NEVPT2Result,
                     corrected_property_matrices, pseudo_canonicalize, sc_nevpt2,
                     select_correlated)

__all__ = ["sc_nevpt2", "NEVPT2Result", "CanonicalOrbitals", "pseudo_canonicalize",
           "CorrelatedSpaces", "select_correlated", "corrected_property_matrices",
           "MultipletCorrection", "MULTIPLET_TOL_CM", "DEFAULT_MULTIPLET_TOL_CM",
           "IntegralBlocks", "batch_slices",
           "ClassContext", "ClassResult", "ExcitationClass", "available_classes",
           "excitation_class", "implemented_classes", "register_class",
           "CIContractionProvider", "ShiftedSpace", "ShiftedSpaces",
           "hole_rdm1", "hole_pair_matrix", "pair_matrix",
           "koopmans_annihilation", "koopmans_creation"]
