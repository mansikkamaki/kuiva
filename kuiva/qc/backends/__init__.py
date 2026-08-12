"""Backend adapters: one module per stack, each import-gated.

⚠ **This package imports nothing.** Not the stub, not an adapter, not a framework. Every
backend is reached through ``kuiva.qc.backend``'s name-to-factory registry, whose factories
import the module they need at call time — so ``import kuiva.qc`` succeeds on a machine with
nothing installed, and the default ``pytest`` run never depends on ``external/venv_qc``. A
convenience re-export here would undo all of that silently, which is why
``tests/test_qc_skeleton.py`` asserts the absence from the sources rather than trusting the
convention.

Present: :mod:`kuiva.qc.backends.stub` (Kuiva's own exact statevector simulator, the declared
second implementation the boundary is validated against) and
:mod:`kuiva.qc.backends.qiskit_aer` (Qiskit Aer, built by ``scripts/bootstrap/90_qiskit.sh``).
A Cirq, cuQuantum, vendor-REST or IQM adapter is another entry in the same table.
"""
