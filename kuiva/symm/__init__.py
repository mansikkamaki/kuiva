"""Abelian double-group symmetry: the label vocabulary and everything that produces labels.

One module owns the descriptor — components, moduli, composition, conjugation — and every
other layer consumes it: the front end assigns labels to orbitals, the CI blocks its
determinants by them, the orbital optimizer masks rotations between them, and the
tensor-network layer widens its ``QuantumNumber`` with them. Nothing here imports from any
consumer, and nothing here needs an integral library or a ``Mole``: it takes the AO layout,
plain coefficient arrays and an overlap matrix.

⚠ **What abelian symmetry can and cannot promise.** Abelian fermion irreps are
one-dimensional, so a spinor carries an integer label, a determinant's label is the group sum
of its occupied spinors' labels, and the Hamiltonian is exactly block diagonal over the
sectors whenever the orbitals are symmetry-pure. But for a molecule whose true group is larger
than the abelian group being used — every atom, and every molecule reduced from ``D2h``,
``D2`` or ``C2v`` because their double groups are not abelian — **physically degenerate
partners can carry different labels**, so a per-irrep state count can cut a degenerate
manifold in half. Group completeness and the state-average boundary diagnostics stay
load-bearing *with* symmetry on; "safe by construction" is only ever claimed where the
abelian group is the whole story.

⚠ **Which is why there is a second layer, and why it CLASSIFIES rather than adapts.**
:mod:`~kuiva.symm.double` builds the molecule's full point double group and
:mod:`~kuiva.symm.classify` names each converged degenerate block by its irreps, so a
multiplet the abelian labels cannot see gets a name and a theory-fixed dimension. That runs
**after** a solve and changes no number: no symmetry-adapted many-particle basis is built, no
double-group coupling coefficients exist anywhere here, and the mathematics of every stage
still runs in the abelian subgroup. Non-abelian symmetry *adaptation* is out of scope and
this is not a first instalment of it.
"""
from __future__ import annotations

from .assign import (MolecularSymmetry, OrbitalLabels, analyze, group_from_operations,
                     resolve_classification)
from .classify import (Classification, StateClassifier, apply_orbital_rotation,
                       assert_multiplet_boundary)
from .double import DoubleGroup, detect_point_group, double_group
from .groups import GROUPS, REDUCTION, Generator, Group, resolve_group
from .operators import (AOOperation, ao_operation, ao_signs, atom_permutation,
                        detect_operations, group_operations, label_scalar_orbitals)
from .report import (character_table, correspondence_table, double_character_table,
                     report, sector_table)
from .sectors import (SectorTable, assert_sector_symmetry, determinant_labels,
                      mode_bases, resolve_state_request, sector_charge,
                      sector_violation)

__all__ = ["AOOperation", "Classification", "DoubleGroup", "GROUPS", "Generator",
           "Group", "MolecularSymmetry", "StateClassifier",
           "OrbitalLabels", "REDUCTION", "SectorTable", "analyze", "apply_orbital_rotation",
           "assert_multiplet_boundary", "assert_sector_symmetry", "ao_operation", "ao_signs",
           "atom_permutation", "character_table", "correspondence_table", "detect_operations",
           "determinant_labels", "detect_point_group", "double_character_table", "double_group",
           "group_from_operations", "group_operations", "label_scalar_orbitals",
           "mode_bases", "report", "sector_charge",
           "resolve_group", "resolve_state_request",
           "sector_table", "sector_violation"]
