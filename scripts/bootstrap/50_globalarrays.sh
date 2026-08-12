#!/usr/bin/env bash
# 50_globalarrays.sh — build Global Arrays (GA) on Intel MPI + Intel MKL (ILP64).
#
# WHY: MPI-parallel OpenMolcas requires Global Arrays. Built on Intel MPI (the project
# default). 64-bit integers (--enable-i8) + MKL ILP64 to match OpenMolcas's int64 build.
# Runtime: two-sided MPI (--with-mpi-ts) — the most portable/robust ARMCI backend on a
# single fat node (no special fabric/progress-rank requirements).
#
# Reference: J. Nieplocha, R.J. Harrison, R.J. Littlefield, "Global Arrays: A portable
# shared-memory programming model for distributed memory computers", Supercomputing '94.
# Also: J. Nieplocha et al., Int. J. High Perform. Comput. Appl. 20 (2006) 203.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
print_toolchain

PREFIX="${INSTALL_DIR}/globalarrays"
TOP="ga-${GA_VERSION}"
TB="${TOP}.tar.gz"

if [[ ! -f "${SRC_DIR}/${TB}" ]]; then
  URL="https://github.com/GlobalArrays/ga/releases/download/v${GA_VERSION}/${TB}"
  log "downloading Global Arrays ${GA_VERSION}"
  wget -q -O "${SRC_DIR}/${TB}" "${URL}" || die "GA download failed: ${URL}"
fi

extract_to_build "${TB}" "${TOP}"
BD="${BUILD_DIR}/${TOP}"
cd "${BD}"

# ⚠ MKL moved its shared libraries from lib/intel64 to lib/ in 2025.x+ (and bumped the soname
# .so.2 -> .so.3 in 2026.x, which is what invalidated the previous build of this package).
# Take whichever directory this MKL actually has rather than the historical one.
MKL_LIB="${MKLROOT}/lib/intel64"; [[ -d "${MKL_LIB}" ]] || MKL_LIB="${MKLROOT}/lib"
MKL_ILP64_BLAS="-L${MKL_LIB} -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"

# ⚠ The wrappers come from common.sh and are mpiicx/mpiicpx/mpiifx, NOT mpiicc/mpiicpc/
# mpiifort: those wrap the classic icc/icpc/ifort, which oneAPI removed after 2024. They still
# exist as scripts and still fail.
# ⚠ GA 5.8.2's Fortran-string-convention probe FAILS under ifx (2025.3): it builds and links a
# mixed C/Fortran executable, runs it, gets no usable output, and aborts with "f2c string
# convention is neither after args nor after string". The probe is broken, not the compiler —
# ifx passes hidden string lengths **after all args**, which is the ordinary Linux convention
# and was verified directly before seeding it here (a two-file icx/ifx program printing the
# lengths: `call csub('hello','worldly')` arrives as a_len=5, b_len=7 after both pointers).
# Seeding the cache variable skips the probe and keeps the answer it would have given.
# ⚠ Re-verify this if the Fortran compiler is ever changed; a wrong convention here would
# corrupt every string GA passes across the language boundary, silently.
./configure \
  --prefix="${PREFIX}" \
  ga_cv_f2c_string_after_args=yes \
  MPICC="${MPICC}" MPICXX="${MPICXX}" MPIF77="${MPIFC}" MPIFC="${MPIFC}" \
  --with-mpi-ts \
  --enable-i8 \
  --enable-cxx \
  --with-blas8="${MKL_ILP64_BLAS}" \
  2>&1 | tee "${LOG_DIR}/ga_configure.log"

make -j "${NPROC}" 2>&1 | tee "${LOG_DIR}/ga_make.log"
make install 2>&1 | tee "${LOG_DIR}/ga_install.log"
log "Global Arrays installed: ${PREFIX}"
ls -1 "${PREFIX}/lib"*/libga.* 2>/dev/null | head || warn "libga not found under ${PREFIX}"
