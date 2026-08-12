#!/usr/bin/env bash
# 95_native.sh — build the OPTIONAL compiled kernel backend in the developer sandbox.
#
# ⚠ A WRAPPER, NOT A BUILD. Everything about the build — toolchain detection, flags, the
# link-time assertions, the build id, the parity check — lives in cpp/configure and
# cpp/Makefile, so there is exactly one build path and a user without this sandbox uses the
# same one. What this file adds is the sandbox's two pins: the venv the extension is built
# against, and the pybind11 version.
#
# ROLE: a per-kernel C++ replacement behind the ci/kernels.py registry. The pure-NumPy path
# stays a first-class way to run: a clone that never runs this script runs everything with
# NumPy alone, and KUIVA_KERNELS selects the backend at run time (auto | numpy | native).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
print_toolchain

[[ -x "${VENV_DIR}/bin/python" ]] || die "python venv not found; run 10_python_env.sh first"

PY="${VENV_DIR}/bin/python"
CPP_DIR="${ROOT_DIR}/cpp"

# --- pybind11, pinned, headers only ----------------------------------------
# --no-deps for the same reason as 80_x2camf.sh: an unpinned transitive numpy would resolve
# to 2.x and silently upgrade the venv out of the baseline the rest of the project needs.
have="$("${PY}" -c 'import pybind11; print(pybind11.__version__)' 2>/dev/null || true)"
if [[ "${have}" != "${PYBIND11_VERSION}" ]]; then
  log "installing pybind11 ${PYBIND11_VERSION} (found: ${have:-none})"
  "${PY}" -m pip install --no-deps "pybind11==${PYBIND11_VERSION}" \
    2>&1 | tee "${LOG_DIR}/native_pybind11.log" | tail -2
else
  log "pybind11 ${PYBIND11_VERSION} already installed"
fi

# --- configure, build, verify ----------------------------------------------
# OPTFLAGS comes from common.sh (-O3 -xCORE-AVX2: this box is Haswell, AVX2/FMA and no
# AVX-512), and configure takes it from the environment. cpp/configure's own default is
# -xHost, which would be the same thing here and the wrong thing on a shared cluster build.
cd "${CPP_DIR}"
./configure --python "${PY}" --cxx "${CXX}" --optflags "${OPTFLAGS}" \
  2>&1 | tee "${LOG_DIR}/native_configure.log"
make 2>&1 | tee "${LOG_DIR}/native_build.log"
make check 2>&1 | tee "${LOG_DIR}/native_check.log"

log "native backend ready: $(make -s print-config | awk '/^TARGET/ {print $2}')"
log "run-time switch: KUIVA_KERNELS=auto (default) | numpy | native"
