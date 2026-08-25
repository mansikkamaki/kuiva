"""Property output and analysis.

* ``dump.py`` — the plain-text contract with the external ITO/crystal-field code: ``H``, the
  three magnetic-moment components and the three electric-dipole components, in the basis of
  the spin-orbit eigenstates. This is what the whole program is for. ⚠ Operators and their
  invariants only: no oscillator strengths, no rates.
* ``multiplet.py`` — the phase-invariant reductions that make SOC spectra and magnetic-moment
  matrices comparable at all, given that the dump fixes no phase convention. Any validation of a
  dump goes through here.
* ``population.py`` — Loewdin population analysis for two-component spinors: atomic charges
  and spin, and the reduced AO populations that identify an active spinor *without* a picture.
* ``molden.py`` — spinor densities as exact real components, in a format standard
  visualization software reads.
* ``spin.py`` — ``<S^2>`` over converged CI states, per degenerate block: the term
  multiplicity with spin-orbit coupling off, the spin-purity diagnostic with it on.
* ``assign.py`` — a ``^{2S+1}L_J`` label per block, offered with the evidence behind it and
  withheld where the evidence does not add up. ⚠ Inference, never a computed quantity: it has
  its own report and never enters a state table or a stored file.
* ``pseudospin.py`` — the multi-site counterpart of ``dump.py``: pseudospin assignment of
  the local-multiplet model and the formatted file the external Ouluspin code
  reads. Validation goes through ``multiplet.py``'s invariants, exactly like the dump.

``population.py`` and ``molden.py`` are the two halves of "is this active space the one I
meant?", and they share the invariance discipline every module here follows: an individual
spinor -- or an individual state -- inside a degenerate manifold is not a well-defined object,
so everything defaults to summing over the block.
"""
from .assign import Assignment, TermAssignment, assign_terms, term_letter
from .dump import (FORMAT_VERSION, PropertyMatrices, inactive_moment, property_matrices,
                   read_dump, spinor_operators, state_operator_matrices, write_dump)
from .molden import MoldenOrbital, SpinorMoldenReport, write_molden, write_spinor_molden
from .multiplet import (
    AXIS_DEFINED_RTOL, G_ELECTRON, HARTREE_TO_CM, PSEUDO_DOUBLET_HINT_CM, Multiplet,
    analyse_spectrum,
    axis_is_defined, block_dipole_tensor, block_line_strengths, block_moment_tensor,
    block_operator_tensor, degeneracy_pattern, degenerate_blocks,
    g_determinant_sign, lande_g, magnetic_moment_matrices, multiplet_g_axes,
    multiplet_g_values, spectrum_line_strengths,
)
from .population import (
    AtomicPopulations, AtomicReferenceCharges, OrbitalPopulations, atomic_populations,
    atomic_reference_charges, lowdin_analysis, orbital_populations,
)
from .pseudospin import (PseudospinModel, PseudospinSite, assign_pseudospin,
                         pseudospin_from_model, read_pseudospin, write_pseudospin)
from .spin import SpinAnalysis, spin_analysis, spin_from_s_squared, spin_squared_states

__all__ = [
    "FORMAT_VERSION", "PropertyMatrices", "property_matrices", "spinor_operators",
    "inactive_moment", "state_operator_matrices", "write_dump", "read_dump",
    "G_ELECTRON", "HARTREE_TO_CM", "Multiplet", "analyse_spectrum", "block_moment_tensor",
    "block_operator_tensor", "block_dipole_tensor", "block_line_strengths",
    "spectrum_line_strengths",
    "degeneracy_pattern", "degenerate_blocks", "lande_g", "magnetic_moment_matrices",
    "multiplet_g_values",
    "AXIS_DEFINED_RTOL", "PSEUDO_DOUBLET_HINT_CM", "axis_is_defined", "g_determinant_sign", "multiplet_g_axes",
    "AtomicPopulations", "AtomicReferenceCharges", "OrbitalPopulations",
    "atomic_populations", "atomic_reference_charges", "lowdin_analysis",
    "orbital_populations",
    "MoldenOrbital", "SpinorMoldenReport", "write_molden", "write_spinor_molden",
    "PseudospinModel", "PseudospinSite", "assign_pseudospin", "pseudospin_from_model",
    "read_pseudospin", "write_pseudospin",
    "SpinAnalysis", "spin_analysis", "spin_from_s_squared", "spin_squared_states",
    "Assignment", "TermAssignment", "assign_terms", "term_letter",
]
