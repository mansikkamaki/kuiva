#!/usr/bin/env bash
# common.sh — shared setup + helpers for all bootstrap scripts.
# Sourced (not executed) by each NN_*.sh build script.
#
# ⚠ These scripts are developer and reference tooling, not an installation route: they
# reproduce the development sandbox and build the external programs that generate reference
# data. Running Kuiva requires none of it — see setup.sh at the repository root.
#
# Establishes the Intel oneAPI toolchain (icx / icpx / ifx) with Intel MPI as the
# DEFAULT parallel backend (per project decision: Intel MPI is default everywhere;
# OpenMPI is built only as a portability reference).

set -euo pipefail

# --- Resolve repo layout ---------------------------------------------------
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${COMMON_DIR}/../.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external" # git-ignored build sandbox
SRC_DIR="${EXTERNAL_DIR}/src"             # staged/downloaded upstream tarballs (git-ignored)
BUILD_DIR="${EXTERNAL_DIR}/build"
INSTALL_DIR="${EXTERNAL_DIR}/install"
LOG_DIR="${EXTERNAL_DIR}/logs"
VENV_DIR="${EXTERNAL_DIR}/venv"
VENV_QC_DIR="${EXTERNAL_DIR}/venv_qc"     # quantum-computing simulator stack, deliberately
                                          # separate from VENV_DIR
mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}" "${LOG_DIR}"

# ⚠ A DEPENDENCY'S BUILD MUST NOT BE ABLE TO REACH KUIVA'S GIT REPOSITORY.
#
# The sandbox lives *inside* the working tree, and the dependencies here are extracted
# tarballs with no `.git` of their own — so git's upward search for a repository walks out of
# external/ and finds Kuiva's. Measured from external/build/OpenMolcas-v26.06:
# `git rev-parse --git-dir` answered /home/akseli/Programs/kuiva/.git.
#
# That is not hypothetical. OpenMolcas's CMake unconditionally runs sbin/install_hooks.sh,
# which resolves the git dir exactly that way and copied its own pre-commit hook — the one
# calling its sbin/copyright and sbin/check_style — into Kuiva's .git/hooks/, where it then
# ran on every Kuiva commit. Installing a hook is the *mild* version; a build script running
# `git clean`, `git checkout` or `git config` would reach real history.
#
# GIT_CEILING_DIRECTORIES stops the upward search at external/, so any git command run inside
# a dependency's tree simply finds no repository (verified: "fatal: not a git repository").
export GIT_CEILING_DIRECTORIES="${EXTERNAL_DIR}${GIT_CEILING_DIRECTORIES:+:${GIT_CEILING_DIRECTORIES}}"

# shellcheck source=versions.env
source "${COMMON_DIR}/versions.env"

# --- Intel oneAPI toolchain ------------------------------------------------
if [[ -z "${SETVARS_COMPLETED:-}" ]]; then
  # setvars.sh is not `set -u` clean; relax nounset while sourcing it.
  set +u
  source "${ONEAPI_ROOT}/setvars.sh" >/dev/null 2>&1 || true
  set -u
fi

# ⚠ Fortran is NOT in the C/C++ compiler tree any more, and setvars.sh does not expose it.
# From oneAPI 2026.x, compiler/<ver>/bin holds icx/icpx only; `ifx` ships as its own component
# under a *different*, older version directory (2025.3 here), which setvars leaves off PATH.
# So find it rather than assume it — and fail loudly if it is absent, because the alternative
# is a cmake run that "successfully" configures a Fortran-less build hours before it matters.
#
# ⚠ **Its lib/ must go on LD_LIBRARY_PATH too, and that half is the one that bites.** A pure
# Fortran program links the Intel Fortran runtime *statically*, so `ifx hello.f && ./a.out`
# works and proves nothing; a mixed program linked by the **C** driver gets `-lifport
# -lifcoremt` *dynamically*, and then fails at load with "libifport.so.5: cannot open shared
# object file" because setvars only exported the 2026.1 lib directory. That killed every
# mixed-language `configure` run-probe in Global Arrays — reported as "could not determine C
# type matching Fortran INTEGER", which names neither Fortran nor a missing library — and it
# would equally break an OpenMolcas or DIRAC binary at run time, long after the build.
# ⚠ `|| true` on both: this file runs under `set -e`, and a failing command substitution in an
# assignment aborts the script — silently, with no output at all, since the failure is the
# lookup rather than anything that prints.
_ifx="$(command -v ifx 2>/dev/null || true)"
[[ -n "${_ifx}" ]] || _ifx="$(ls -d "${ONEAPI_ROOT}"/compiler/*/bin/ifx 2>/dev/null | sort -V | tail -1 || true)"
if [[ -n "${_ifx}" ]]; then
  _ifx_root="$(cd "$(dirname "${_ifx}")/.." && pwd)"
  case ":${PATH}:" in *":${_ifx_root}/bin:"*) ;; *) export PATH="${_ifx_root}/bin:${PATH}" ;; esac
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${_ifx_root}/lib:"*) ;;
    *) export LD_LIBRARY_PATH="${_ifx_root}/lib:${LD_LIBRARY_PATH:-}" ;;
  esac
  unset _ifx_root
fi
unset _ifx

# Compilers: Intel LLVM-based C/C++ (icx/icpx) + Intel Fortran (ifx).
export CC="${CC:-icx}"
export CXX="${CXX:-icpx}"
export FC="${FC:-ifx}"
export F77="${F77:-ifx}"

# Intel MPI compiler wrappers pointed at the Intel LLVM/Fortran compilers.
# ⚠ mpiifort wraps `ifort`, which oneAPI REMOVED after 2024 — the wrapper still exists and
# still fails. mpiicx/mpiicpx/mpiifx are the LLVM-compiler wrappers and are what to use;
# I_MPI_* remain as a second belt for build systems that call the generic mpicc/mpicxx/mpif90.
export I_MPI_CC="${I_MPI_CC:-icx}"
export I_MPI_CXX="${I_MPI_CXX:-icpx}"
export I_MPI_F90="${I_MPI_F90:-ifx}"
export I_MPI_FC="${I_MPI_FC:-ifx}"
export MPICC="${MPICC:-mpiicx}"
export MPICXX="${MPICXX:-mpiicpx}"
export MPIFC="${MPIFC:-mpiifx}"

# Optimization flags. Dev box is Haswell (AVX2/FMA, NO AVX-512): -xCORE-AVX2 is the
# correct/safe ISA target here. On an AVX-512 cluster, retune OPTFLAGS.
export OPTFLAGS="${OPTFLAGS:--O3 -xCORE-AVX2}"

NPROC="$(nproc)"

# --- Helpers ---------------------------------------------------------------
log()  { printf '\033[1;34m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bootstrap WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[bootstrap ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

require_tarball() {
  local tb="$1"
  [[ -f "${SRC_DIR}/${tb}" ]] || die "missing tarball: ${SRC_DIR}/${tb}"
}

extract_to_build() {
  # extract_to_build <tarball> <expected-top-dir>
  local tb="$1" top="$2"
  require_tarball "${tb}"
  if [[ ! -d "${BUILD_DIR}/${top}" ]]; then
    log "extracting ${tb} -> build/${top}"
    tar -xf "${SRC_DIR}/${tb}" -C "${BUILD_DIR}"
  else
    log "already extracted: build/${top}"
  fi
}

print_toolchain() {
  log "CC=${CC} ($(command -v ${CC} 2>/dev/null || echo missing))"
  log "CXX=${CXX} ($(command -v ${CXX} 2>/dev/null || echo missing))"
  log "FC=${FC} ($(command -v ${FC} 2>/dev/null || echo missing))"
  log "MPIFC=${MPIFC} ($(command -v ${MPIFC} 2>/dev/null || echo missing))"
  log "MKLROOT=${MKLROOT:-unset}"
  log "Intel MPI mpirun=$(command -v mpirun 2>/dev/null || echo missing)"
  log "OPTFLAGS=${OPTFLAGS}  NPROC=${NPROC}"
}
