#!/usr/bin/env bash
# 90_qiskit.sh — build external/venv_qc: the quantum-computing simulator stack
# The research-only quantum-computing simulator stack.
#
# ROLE: research/testing only. Qiskit is the *first backend adapter* of kuiva/qc/, never its
# substrate — the qubit mapping, the circuit representation, the ansatz builders and the
# sampled-subspace driver are pure NumPy/SciPy, and every framework import is gated
# (kuiva/qc/gate.py). Nothing in kuiva imports Qiskit at package import time, the default
# pytest run never needs this venv, and a `pip install kuiva` never pulls it. Same status as
# OpenMolcas and DIRAC: a development dependency, not a runtime one.
#
# ⚠ A SEPARATE VENV, and the reason is structural rather than tidy. x2camf (80_x2camf.sh)
# shares external/venv and is guarded with `pip install --no-deps`, because it is one small
# pybind11 extension against a controlled dependency. Qiskit is not: it brings a large,
# independently fast-moving tree (rustworkx, symengine, its own numpy/scipy floors) with real
# potential to drag the shared venv off the pinned NumPy-1.x / Python-3.9 baseline that the
# *entire rest of the project* — PySCF, Intel's system-site MKL SciPy — depends on. Isolating
# it removes that risk instead of pinning around it.
#
# ⚠ A DIFFERENT INTERPRETER, and this is a pinned decision (versions.env::QC_PYTHON_BIN).
# Qiskit dropped Python 3.9 at 2.3.0 (2026-01) and qiskit-iqm — the documented route to VTT
# Q50, Stage 6 — requires >=3.10,<3.13. Building venv_qc from intelpython 3.9 would therefore
# either freeze Qiskit at 2.2.3 forever or foreclose the hardware path. The pinned baseline binds
# external/venv and the dev toolchain; kuiva is written to 3.9 *syntax* and runs unmodified on
# newer Pythons, and venv_qc's isolation is exactly what keeps this choice from propagating
# back. Verified below, not assumed: the check at the end imports kuiva's classical CI/RDM
# machinery under this interpreter, because that is what the qc-marked tests will do.
#
# ⚠ pip is bootstrapped with get-pip.py rather than ensurepip. Ubuntu's python3.10 ships
# without ensurepip (`python3-venv` is an apt package), and the sandbox forbids sudo. `--without-pip`
# plus get-pip.py is the sudo-free route and needs the network this script already needs.
#
# NOT INSTALLED HERE: qiskit-iqm (hardware access; optional dependencies are built only when
# actually needed — see versions.env for the pin it will take) and qiskit-addon-sqd (IBM's SQD
# reference implementation; it assumes the spin-separable, real, non-relativistic Hamiltonian
# kuiva's SOC-coupled active space is not, so it is something to *read* before writing
# kuiva/qc/sqd.py, not something to depend on).
#
# REFERENCES:
#   * Qiskit: A. Javadi-Abhari et al., "Quantum computing with Qiskit", arXiv:2405.08810
#     (2024), doi:10.48550/arXiv.2405.08810; https://github.com/Qiskit/qiskit.
#   * Qiskit Aer (high-performance simulator backends): https://github.com/Qiskit/qiskit-aer.
#   * IQM's Qiskit provider, the bridge to VTT Q50: https://github.com/iqm-finland/qiskit-on-iqm.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "QC stack: qiskit ${QISKIT_VERSION} + qiskit-aer ${QISKIT_AER_VERSION}"
log "interpreter: ${QC_PYTHON_BIN} (NOT ${PYTHON_BIN} — see the header and versions.env)"

[[ -x "${QC_PYTHON_BIN}" ]] || die "QC interpreter not found at ${QC_PYTHON_BIN}; set QC_PYTHON_BIN in versions.env"

# Refuse an interpreter the pinned packages cannot run on, here rather than three pip
# resolutions later. The floor is Qiskit's (>=3.10); the ceiling is qiskit-iqm's (<3.13),
# enforced now so Stage 6 does not discover it against a venv everything else was built in.
"${QC_PYTHON_BIN}" - <<'PY' || die "QC interpreter is outside the >=3.10,<3.13 window (qiskit / qiskit-iqm)"
import sys
v = sys.version_info
print("  interpreter {}.{}.{}".format(v.major, v.minor, v.micro))
sys.exit(0 if (3, 10) <= (v.major, v.minor) < (3, 13) else 1)
PY

# --- create the venv, sudo-free --------------------------------------------
# ⚠ --system-site-packages is deliberately NOT used here (10_python_env.sh does use it, to
# reuse Intel's MKL NumPy/SciPy). This venv must be self-contained: mixing a 3.9 MKL NumPy's
# site-packages into a 3.10 interpreter is exactly the ABI accident the isolation exists to
# prevent, and Qiskit's own numpy/scipy floors are what should win in here.
if [[ ! -x "${VENV_QC_DIR}/bin/python" ]]; then
  log "creating ${VENV_QC_DIR} (isolated; no system site-packages)"
  "${QC_PYTHON_BIN}" -m venv --without-pip "${VENV_QC_DIR}"
fi

if ! "${VENV_QC_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
  log "bootstrapping pip with get-pip.py (no ensurepip on this interpreter)"
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "${SRC_DIR}/get-pip.py" \
    || die "could not fetch get-pip.py (network?)"
  "${VENV_QC_DIR}/bin/python" "${SRC_DIR}/get-pip.py" -q || die "pip bootstrap failed"
fi
"${VENV_QC_DIR}/bin/python" -m pip install --quiet --upgrade pip

# --- install, pinned -------------------------------------------------------
# pytest so the qc-marked tests can be run under this interpreter; nothing else. In
# particular kuiva itself is NOT installed here — the tests put the repository root on
# PYTHONPATH (as setup.sh does), so they exercise the working tree and no
# second, stale copy of kuiva can exist.
log "installing pinned packages into ${VENV_QC_DIR}"
"${VENV_QC_DIR}/bin/python" -m pip install \
  "qiskit==${QISKIT_VERSION}" "qiskit-aer==${QISKIT_AER_VERSION}" pytest \
  2>&1 | tee "${LOG_DIR}/qiskit_install.log" | tail -3

# --- verify ----------------------------------------------------------------
# Not "does it import" but "does it produce the right numbers", on both primitives
# kuiva/qc/backend.py will declare: an exact statevector and a shot
# distribution. A silently mis-built Aer would import perfectly and sample nonsense.
log "verifying the two backend primitives against an analytically known circuit"
"${VENV_QC_DIR}/bin/python" - <<'PY'
import numpy as np
import qiskit, qiskit_aer
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

# A Bell pair: statevector exactly (|00> + |11>)/sqrt(2), samples exactly two outcomes.
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

sv = np.asarray(Statevector(qc).data)
ref = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2.0)
err = np.abs(sv - ref).max()
assert err < 1e-12, sv

meas = qc.copy()
meas.measure_all()
sim = AerSimulator(seed_simulator=20260807)
counts = sim.run(transpile(meas, sim), shots=4096).result().get_counts()
assert set(counts) == {"00", "11"}, counts          # particle-number-like structure survives
assert abs(counts["00"] - counts["11"]) < 400, counts

print("  OK  qiskit {} / aer {} / numpy {}".format(
    qiskit.__version__, qiskit_aer.__version__, np.__version__))
print("  OK  statevector exact to {:.2e}; 4096 shots -> {}".format(err, counts))
PY

# The venv is only useful if kuiva's own classical machinery runs in it: the qc-marked tests
# compare a simulated solver against kuiva's FullCISolver, in this interpreter. This asserts
# the baseline-crossing claim in the header instead of leaving it as a hope.
log "verifying kuiva's classical CI machinery under this interpreter"
PYTHONPATH="${ROOT_DIR}" "${VENV_QC_DIR}/bin/python" - <<'PY'
import itertools
import numpy as np
import kuiva.qc                       # must import with no framework installed at all
from kuiva.ci.strings import Determinants, cas_dimension, hamiltonian_matrix

# Run kernels, not just imports: a NumPy 2.x casting or dtype change would surface in the
# uint64 bitmask arithmetic long before it surfaced anywhere a bare import could see.
n, k = 4, 2
dets = Determinants.from_occupations(itertools.combinations(range(n), k), n)
assert len(dets) == cas_dimension(n, k) == 6, len(dets)
rng = np.random.default_rng(20260807)
h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
ham = hamiltonian_matrix(dets, h + h.conj().T, np.zeros((n,) * 4, dtype=complex))
assert np.abs(ham - ham.conj().T).max() == 0.0, "complex CI Hamiltonian is not Hermitian"
print("  OK  kuiva imports on numpy {} (CAS(2,4): {} determinants, Hermitian H)".format(
    np.__version__, len(dets)))
print("  OK  kuiva.qc imports with no quantum framework required")
PY

log "QC simulator stack ready: ${VENV_QC_DIR}"
log "run the qc-marked tests with:  PYTHONPATH=${ROOT_DIR} ${VENV_QC_DIR}/bin/python -m pytest -m qc"
