"""Quantum-computing CI solvers for the CASSCF branch.

**Status: everything reachable without a quantum computer is implemented.** What is here is
the import gate (:mod:`kuiva.qc.gate`), the fermion-to-qubit mapping
(:mod:`kuiva.qc.mapping`), the Kuiva-owned circuit representation (:mod:`kuiva.qc.circuits`),
the operator-to-circuit compiler (:mod:`kuiva.qc.fermionic`), the backend protocol and registry
(:mod:`kuiva.qc.backend`) with Kuiva's own exact statevector simulator and a Qiskit Aer adapter
behind it, the circuit strategies (:mod:`kuiva.qc.ansatz`) from a bare reference state up to
complex-spinor UCC and cluster-Jastrow ansaetze and Krylov time evolution, configuration
recovery (:mod:`kuiva.qc.recovery`) including the time-reversal-closed policy, the
sampled-subspace solver (:mod:`kuiva.qc.sqd`), the variational eigensolver
(:mod:`kuiva.qc.vqe`) and the algorithm registry (:mod:`kuiva.qc.algorithms`). What is left is
hardware, and it is an access question rather than an engineering one.

⚠ **Implemented and exact is not the same as validated as good.** The Stage-C ansaetze
generalize the published families past the spin separability Kuiva's Hamiltonian does not
have, and each generator's exponential is compiled exactly rather than approximated. Whether
such a circuit concentrates its measurements where they matter, on a system large enough for
the question to be interesting, is **open** — it needs hardware — so no number from this
package is evidence about the *method*, and no docstring here may read as though it were.
A working plan document exists locally during development and is deleted when the
hardware stage finishes.

⚠ **This module deliberately re-exports nothing but the gate.** A convenience import of
:mod:`kuiva.qc.backend` here would pull the backend registry — and one day, through a
carelessly eager factory, a framework — into ``import kuiva.qc``, which is the one thing the
package layout exists to prevent. Import the submodule you need.

What this package is
--------------------
A **second CI solver**, plugged into the seam ``mcscf/preopt.py``'s cheap CI and
``dmrg/solver.py`` already share: the shared ``ci_solver(ints) -> (energy, gamma, Gamma)``
contract, or the four-method ``AdaptiveCISolver`` protocol of the adaptive layer for the re-selecting
kind — which a sampling-based solver is, by construction. Everything else in the program (the
SCF, X2C, the orbital optimizer, the RDM machinery, NEVPT2, the property dump) is unchanged
and never learns that a quantum computer was involved.

Two structural facts make the fit unusually close, and neither was arranged for it:

* ``ci/strings.py`` addresses the CI space with a **single occupation-string bitmask over
  spinors** — chosen years earlier because SOC forbids the usual alpha/beta factorization
   — which *is* the Jordan-Wigner computational basis, mode ``p`` to qubit ``p``. No
  index reshuffling is needed at the determinant level.
* ``CASIntegrals`` (``mcscf/orbopt.py``) already hands out exactly the complex one- and
  two-electron active-space integrals a qubit Hamiltonian is built from, in the same basis
  ``ci/strings.hamiltonian_matrix`` consumes. This package is a second consumer of an
  existing interface, not a new seam into the orbital optimizer.

⚠ What does NOT transfer from the literature, stated here because it is the whole risk:
every published SQD/VQE quantum-chemistry demonstration assumes a **spin-separable, real,
non-relativistic** Hamiltonian. Kuiva's is none of those. The qubit mapping generalizes
mechanically (Jordan-Wigner of a complex Hermitian second-quantized operator is ordinary
linear algebra); the *ansatz families* do not, because they encode separability structurally.

⚠ **The sharp, checkable form of that, because it turned out smaller and worse than expected:**
with a *real* amplitude an excitation generator maps to 2 (singles) or 8 (doubles) Pauli
strings and **they all commute**, which is precisely why every published UCC circuit is a plain
product of Pauli exponentials — and with a **complex** amplitude the count doubles and the
halves do **not** commute, so the same product silently becomes a Trotter approximation of
something the literature calls exact. :func:`kuiva.qc.fermionic.excitation_circuit` compiles it
exactly instead, by conjugating a real generator with a phase gate. The same mechanism decides
whether a Trotterized ``exp(-iHt)`` conserves particle number, and it is why a variational
one-body rotation is parameterized by Givens angles rather than by an anti-Hermitian ``kappa``.
One finding, three consequences — and none of it is visible in an energy.

The boundary rules (they bind every module added here)
----------------------------------------------------------------------
1. **The dependency runs one way.** Nothing in ``ci/``, ``mcscf/``, ``rdm/`` or ``x2c/`` may
   import :mod:`kuiva.qc`; this package imports *them*. Same one-way rule as
   :mod:`kuiva.x2c`, and it is asserted from the sources rather than by habit.
2. **No vendor stack anywhere but an adapter.** Framework imports live inside the function
   that needs them, in ``kuiva/qc/backends/``, behind :func:`kuiva.qc.gate.require`. Plain
   NumPy arrays and metadata cross every boundary, in both directions — circuits included, so
   ansatz research stays portable and an out-of-process adapter is never foreclosed.
3. **Two orthogonal axes: algorithm and backend.** Each is a name-to-factory registry; an
   algorithm declares the primitives it needs and an incapable pairing is refused at
   construction, naming both sides. The same discipline the method surface imposes on decoupling x
   screening, for the same reason.
4. **Provenance is part of the result.** Backend name and version, device, shots, seed, noise
   model — carried with the energy, as the ``ScreeningRecord``/``DecouplingRecord`` are
. An energy whose record does not say which device and how many shots produced
   it is not interpretable, and the record outlives the session.

References
----------------------------
* Sample-based quantum diagonalization: IBM Quantum, "Chemistry beyond the scale of exact
  diagonalization on a quantum-centric supercomputer", *Sci. Adv.* (2025),
  doi:10.1126/sciadv.adu9991.
* Jordan-Wigner: P. Jordan, E. Wigner, *Z. Phys.* **47**, 631 (1928), doi:10.1007/BF01331938.
* VQE: A. Peruzzo et al., *Nat. Commun.* **5**, 4213 (2014), doi:10.1038/ncomms5213.
* The simulator stack the first adapter targets: A. Javadi-Abhari et al., "Quantum computing
  with Qiskit", arXiv:2405.08810 (2024) — an optional install, never a
  runtime dependency.
"""
from .gate import INSTALL_HINT, available, require

__all__ = ["INSTALL_HINT", "available", "require"]
