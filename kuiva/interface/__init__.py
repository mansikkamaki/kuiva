"""Front-end / PySCF bridge and the public Python API.

Two layers, both public: the **class API** of :mod:`kuiva.interface.stages` (re-exported at
the top level, ``kuiva.CASSCF`` etc.) is the primary user surface; the functions of
:mod:`kuiva.interface.api` are the layer under it and remain available unchanged.
"""
from .api import (Molecule, SpinorReference, active_space_for, casci, casscf,
                  project_to_basis, projected_active_space, scalar_x2c_reference,
                  spinor_reference)
from .pyscf_bridge import ScalarX2CData, build_mole, run_scalar_x2c
from .stages import (CASSCF, CheapCI, NEVPT2, PropertyDump, PseudospinExport, Reference,
                     ScalarSCF)

__all__ = [
    "Molecule", "SpinorReference", "scalar_x2c_reference", "spinor_reference",
    "active_space_for", "casci", "casscf", "project_to_basis", "projected_active_space",
    "ScalarX2CData", "build_mole", "run_scalar_x2c",
    "ScalarSCF", "Reference", "CheapCI", "CASSCF", "NEVPT2", "PropertyDump",
    "PseudospinExport",
]
