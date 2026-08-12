#!/usr/bin/env bash
# 10_python_env.sh — create the project Python venv from oneAPI intelpython,
# reusing Intel's MKL-accelerated NumPy/SciPy, and install pinned PySCF.
#
# Pin PySCF and test against it.  The interpreter choice — and why the
# *development* interpreter and the *syntax floor* are two different numbers — is in
# versions.env.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
print_toolchain

[[ -x "${PYTHON_BIN}" ]] || die "intelpython not found at ${PYTHON_BIN}"

# intelpython venvs need PYTHONHOME set, or python emits '<prefix>' warnings.
export PYTHONHOME="${INTELPYTHON_ROOT}"

# ⚠ An existing venv is checked for a VERSION MATCH, not just for existence, and this is not
# hypothetical hardening: a oneAPI update repointed intelpython/latest from 3.9 to 3.12, and
# because a venv's bin/python is a symlink *through* that path, external/venv silently became
# a 3.12 interpreter sitting on a lib/python3.9/site-packages tree. `import pyscf` failed and
# every C extension in it was the wrong ABI, while `[[ -x bin/python ]]` was perfectly true.
# versions.env now pins the versioned directory so this cannot recur; this check is what
# would catch the next variant of it.
_want="$("${PYTHON_BIN}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ -x "${VENV_DIR}/bin/python" ]]; then
  _have="$("${VENV_DIR}/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "?")"
  _site="$(ls -d "${VENV_DIR}"/lib/python*/ 2>/dev/null | head -1)"
  _site="$(basename "${_site:-python?}")"
  if [[ "${_have}" != "${_want}" || "${_site}" != "python${_want}" ]]; then
    warn "existing venv is inconsistent with ${PYTHON_BIN}"
    warn "  interpreter reports ${_have}, layout is ${_site}, wanted ${_want}"
    log "moving it to ${VENV_DIR}.stale and rebuilding"
    rm -rf "${VENV_DIR}.stale"
    mv "${VENV_DIR}" "${VENV_DIR}.stale"
  fi
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  log "creating venv from python ${_want} (system-site-packages -> reuse Intel MKL numpy/scipy)"
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip >/dev/null
log "installing pyscf==${PYSCF_VERSION} + basis_set_exchange + h5py + pytest"
# basis_set_exchange supplies the basis families PySCF does not bundle (Karlsruhe x2c,
# Peterson cc-pVnZ-X2C) used by kuiva.basis.registry.
"${VENV_DIR}/bin/python" -m pip install "pyscf==${PYSCF_VERSION}" basis_set_exchange h5py pytest
# Install the project itself (editable) so `import kuiva` works and deps resolve.
if [[ -f "${ROOT_DIR}/pyproject.toml" ]]; then
  "${VENV_DIR}/bin/python" -m pip install -e "${ROOT_DIR}" >/dev/null 2>&1 || \
    warn "editable install of kuiva failed (ok if the package is not ready yet)"
fi

log "verifying scalar-X2C (sfx2c1e) front-end path"
"${VENV_DIR}/bin/python" - <<'PY'
import sys
import numpy, scipy
from pyscf import gto, scf

# NumPy 1.x is a project-wide assumption (see every numpy<2 pin). intelpython
# still ships 1.x on 3.12, so this is a guard rather than a constraint — but it is the thing
# that would change silently under the next interpreter bump.
assert numpy.__version__.startswith("1."), \
    "numpy {} breaks the project-wide NumPy-1.x assumption".format(numpy.__version__)
mol = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).sfx2c1e(); e = mf.kernel()
assert mf.converged, "sfx2c1e SCF did not converge"
print("  OK  python {} / numpy {} / scipy {}".format(
    ".".join(map(str, sys.version_info[:3])), numpy.__version__, scipy.__version__))
print("  OK  sfx2c1e E(Ne/cc-pvdz) = {:.8f} Eh".format(e))
PY
log "python env ready: ${VENV_DIR}"
[[ -d "${VENV_DIR}.stale" ]] && log "the previous venv is at ${VENV_DIR}.stale — delete it once you are happy"
exit 0
