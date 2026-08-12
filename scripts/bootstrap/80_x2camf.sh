#!/usr/bin/env bash
# 80_x2camf.sh — build the `x2camf` plugin (X2CAMF reference generator).
#
# ROLE: external cross-check only, and an *optional* runtime path. The plugin is a second,
# independent implementation of the same method kuiva/amf/ implements, by the group that
# published it (J. Liu, L. Cheng, J. Chem. Phys. 148, 144108 (2018)). It is what makes the
# term-by-term comparison possible: DIRAC can say that Kuiva's
# *result* is right (tests/reference/x2camf_dirac.json), but only a same-method
# implementation can say which **term** a disagreement lives in.
#
# NOT a required dependency of kuiva. Every import of it is gated
# (kuiva/amf/x2camf_plugin.py, tests/test_x2camf_plugin.py), the correction method
# "x2camf-external" it enables is never the default, and the committed reference file
# tests/reference/x2camf_plugin.json means the comparison still runs without it.
#
# SOURCE: https://github.com/Warlocat/x2camf — pinned by commit in versions.env, because the
# repository publishes no tags or releases. It vendors pybind11 and Eigen as git submodules,
# so the pin is a superproject commit and the submodule SHAs come with it.
#
# COMPILER: the C++ toolchain is taken from common.sh (icpx) like every other dependency,
# with X2CAMF_CXX to override. ⚠ Nothing in kuiva links against this library — it is loaded
# by the reference generator through pybind11 — so if a future compiler update breaks the
# Eigen/pybind11 build, `X2CAMF_CXX=g++ bash scripts/bootstrap/80_x2camf.sh` is a legitimate
# answer rather than a workaround. Both were built and compared at the time of writing: the
# Ne four-component atomic energy agrees to 2.2e-11 Eh (icpx -128.589917919887, g++
# ...919909) and max |dG| to every printed digit, i.e. the difference is floating-point
# summation order and not the build.
#
# PYTHON: the plugin declares `python_requires>=3.7` and builds against pybind11, so the
# pinned Python 3.9 baseline is enough and no reference-only interpreter is needed. This was
# checked before the plugin was adopted as a cross-check.
#
# STACK SIZE: the plugin's own README warns that its atomic solver can exhaust the default
# stack for large basis sets. `ulimit -s unlimited` before a heavy-element run; the reference
# generator does not need it for the atoms it uses.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
print_toolchain

[[ -x "${VENV_DIR}/bin/python" ]] || die "python venv not found; run 10_python_env.sh first"

SRC="${SRC_DIR}/x2camf"
CXX_FOR_X2CAMF="${X2CAMF_CXX:-${CXX}}"
command -v cmake >/dev/null 2>&1 || die "cmake not found (the plugin builds with cmake >= 3.9)"

# --- fetch, pinned ---------------------------------------------------------
if [[ ! -d "${SRC}/.git" ]]; then
  log "cloning ${X2CAMF_REPO}"
  git clone --quiet "${X2CAMF_REPO}" "${SRC}"
fi
cd "${SRC}"
git fetch --quiet origin
git checkout --quiet "${X2CAMF_COMMIT}" || die "commit ${X2CAMF_COMMIT} not found upstream"
log "x2camf pinned at $(git rev-parse --short HEAD) ($(git log -1 --format=%ci))"
# pybind11 and Eigen are vendored as submodules; their SHAs travel with the pin above.
git submodule update --init --recursive --quiet
git submodule status | while read -r line; do log "  submodule ${line}"; done

# --- build + install into the project venv ---------------------------------
# --no-build-isolation: build against the venv's own setuptools/pybind11 rather than letting
# pip provision a second, unpinned build environment (every dependency pinned).
#
# ⚠ --no-deps is NOT optional, and this was found the hard way. The plugin declares an
# unpinned `install_requires=["numpy"]`, which pip resolves to NumPy 2.x — silently
# **upgrading the venv out of the pinned-version baseline** (NumPy 1.x, because Python 3.9 cannot have
# 2.x's minimum). The result imports and then dies inside PySCF with "numpy.dtype size
# changed": Intel's system-site SciPy 1.7.3 is compiled against the 1.x ABI. The plugin needs
# nothing NumPy 1.x does not provide, so the right fix is to install no dependencies at all
# rather than to pin a second NumPy.
log "building with CXX=${CXX_FOR_X2CAMF} and installing into ${VENV_DIR}"
export CXX="${CXX_FOR_X2CAMF}"
"${VENV_DIR}/bin/python" -m pip install --no-build-isolation --no-deps --force-reinstall . \
  2>&1 | tee "${LOG_DIR}/x2camf_build.log" | tail -5

# Guard the baseline rather than trusting the flag above: a future pip or a changed
# install_requires must fail here, not three modules downstream inside PySCF.
"${VENV_DIR}/bin/python" - <<'PY' || die "numpy was upgraded out of the pinned baseline"
import sys, numpy
major = int(numpy.__version__.split('.')[0])
print("  numpy {} (baseline: 1.x, pinned)".format(numpy.__version__))
sys.exit(0 if major == 1 else 1)
PY

# --- verify ----------------------------------------------------------------
# Not "does it import" but "does it produce the right number". The plugin runs its own
# four-component atomic SCF, so its Ne total energy is directly comparable with PySCF's — and
# a silently mis-built Eigen kernel would import perfectly and return nonsense.
log "verifying against an independent four-component atomic energy"
"${VENV_DIR}/bin/python" - <<'PY'
import numpy, x2camf
from pyscf import gto
from pyscf.gto import mole
from pyscf.x2c import x2c

mol = gto.M(atom='Ne 0 0 0',
            basis={'Ne': mole.uncontracted_basis(gto.basis.load('x2c-SVPall-2c', 'Ne'))},
            verbose=0)
amf = numpy.asarray(x2camf.amfi(x2c.X2C(mol), with_gaunt=False, with_gauge=False))
assert amf.shape == (mol.nao_2c(), mol.nao_2c()), amf.shape
assert numpy.max(numpy.abs(amf - amf.conj().T)) < 1e-12
# The plugin prints its own 4c DHF total energy; PySCF's dhf gives -128.5899179199 for the
# same atom, basis and speed of light (agreement measured at 9e-11 Eh).
print("  OK  x2camf.amfi(Ne) -> {} matrix, max |dG| = {:.6e} Eh".format(amf.shape,
                                                                       numpy.abs(amf).max()))
PY
log "x2camf ready (import-gated; not a runtime dependency of kuiva)"
