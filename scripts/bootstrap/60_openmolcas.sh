#!/usr/bin/env bash
# 60_openmolcas.sh — build OpenMolcas (Tier-2 reference generator).
#
# ROLE: external cross-check only. RASSI provides SOC state energies + magnetic-moment
# matrices comparable to kuiva's output. Built MPI-parallel on Intel MPI + Global Arrays
# (project default), with MKL (ILP64) and our source-built HDF5.
#
# Compiler: Intel ifx/icx. (This line used to offer ifort as a fallback; oneAPI removed the
# classic compilers after 2024, so there is no fallback any more.)
#
# Reference: F. Aquilante et al., "Modern quantum chemistry with [Open]Molcas",
# J. Chem. Phys. 152 (2020) 214117. DOI:10.1063/5.0004835. And the OpenMolcas project,
# https://gitlab.com/Molcas/OpenMolcas.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
print_toolchain

TB="OpenMolcas-${OPENMOLCAS_VERSION}.tar.gz"
TOP="OpenMolcas-${OPENMOLCAS_VERSION}"
PREFIX="${INSTALL_DIR}/openmolcas"
GAROOT="${INSTALL_DIR}/globalarrays"
HDF5_ROOT="${INSTALL_DIR}/hdf5"

[[ -d "${GAROOT}" ]]   || die "Global Arrays not installed (run 50_globalarrays.sh first)"
[[ -d "${HDF5_ROOT}" ]] || die "HDF5 not installed (run 40_hdf5.sh first)"

extract_to_build "${TB}" "${TOP}"
BD="${BUILD_DIR}/${TOP}/build"
rm -rf "${BD}"; mkdir -p "${BD}"

# ⚠ Give this source tree a throwaway git repository, purely as a place for OpenMolcas's
# hook installer to land.
#
# OpenMolcas's CMake unconditionally runs sbin/install_hooks.sh, which resolves its target as
#   gitdir=$(cd $MOLCAS ; (cd $(git rev-parse --git-dir) ; pwd))
# and copies its own pre-commit hook there. Both failure modes have actually happened here:
# with no `.git` in the extracted tarball, git's upward search found **Kuiva's** repository
# and the hook ran on every Kuiva commit; and before Kuiva was a repository at all — or now,
# with common.sh's GIT_CEILING_DIRECTORIES stopping the search — `git rev-parse` returns
# nothing, `cd` with no argument falls back to $HOME, and it creates **~/hooks/pre-commit**
# (dated 2026-07-24, from the first build here).
#
# The ceiling alone therefore only moves the problem. An empty repo in the source tree gives
# the resolution a well-defined answer *inside* external/, which is git-ignored, so the hook
# lands somewhere harmless and neither Kuiva nor $HOME is touched.
if [[ ! -d "${BUILD_DIR}/${TOP}/.git" ]]; then
  log "creating a throwaway git repo in the source tree (hook-installer containment)"
  git init -q "${BUILD_DIR}/${TOP}"
fi

# pymolcas driver needs a few pure-Python packages; put them in the project venv.
if [[ -x "${VENV_DIR}/bin/pip" ]]; then
  "${VENV_DIR}/bin/pip" install --quiet pyparsing six >/dev/null 2>&1 || \
    warn "could not install pymolcas python deps into venv"
fi

# OpenMolcas locates GA via the GAROOT *environment variable* (find_path PATHS ENV GAROOT),
# not the CMake cache var — export it.
export GAROOT

# MPI-parallel build on Intel MPI + GA. ⚠ The wrappers are mpiicx/mpiicpx/mpiifx from
# common.sh, not mpiicc/mpiicpc/mpiifort — those wrap the classic icc/icpc/ifort, which oneAPI
# removed after 2024, so they exist and fail. `ifort` is no longer an available fallback.
OMOLCAS_FC="${FC}"   # ifx
# ⚠ Point CMake at the PROJECT VENV's interpreter, not at whatever `setvars.sh` put first on
# PATH. OpenMolcas's FindPython otherwise picks bare intelpython, and the pymolcas driver is
# then configured against an interpreter this script never provisioned: it needs `pyparsing`,
# which intelpython **3.9 happened to bundle and 3.12 does not**, so the build silently
# depended on a conda package for as long as the baseline was 3.9 and failed the moment it
# moved ("Some Python modules are not available: pyparsing" -> "Failed to configure the
# pymolcas driver"). The venv above is where this script installs those modules, so it is the
# interpreter that is actually guaranteed to have them.
cmake -S "${BUILD_DIR}/${TOP}" -B "${BD}" \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DPython_EXECUTABLE="${VENV_DIR}/bin/python" \
  -DCMAKE_C_COMPILER="${MPICC}" \
  -DCMAKE_CXX_COMPILER="${MPICXX}" \
  -DCMAKE_Fortran_COMPILER="${MPIFC}" \
  -DMPI=ON \
  -DGA=ON \
  -DGAROOT="${GAROOT}" \
  -DLINALG=MKL \
  -DMKLROOT="${MKLROOT}" \
  -DHDF5=ON \
  -DHDF5_ROOT="${HDF5_ROOT}" \
  -DCMAKE_BUILD_TYPE=Release \
  2>&1 | tee "${LOG_DIR}/openmolcas_configure.log"

cmake --build "${BD}" -j "${NPROC}" 2>&1 | tee "${LOG_DIR}/openmolcas_make.log"
cmake --install "${BD}" 2>&1 | tee "${LOG_DIR}/openmolcas_install.log"
log "OpenMolcas installed: ${PREFIX}"
log "NOTE: set MOLCAS=${PREFIX} and use pymolcas driver for Tier-2 reference runs."
