#!/usr/bin/env bash
# 30_openmpi.sh — build OpenMPI with icx/icpx/ifx.
#
# ROLE: portability REFERENCE ONLY. Intel MPI (oneAPI) is the project's default MPI
# for both dependencies and kuiva itself. OpenMPI is built so we can periodically
# confirm kuiva also compiles/links/runs against OpenMPI (avoids Intel-MPI lock-in).
#
# Reference: E. Gabriel et al., "Open MPI: Goals, Concept, and Design of a Next
# Generation MPI Implementation", EuroPVM/MPI 2004, LNCS 3241, pp. 97-104.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
print_toolchain

TB="openmpi-${OPENMPI_VERSION}.tar.gz"
TOP="openmpi-${OPENMPI_VERSION}"
PREFIX="${INSTALL_DIR}/openmpi"
extract_to_build "${TB}" "${TOP}"

BD="${BUILD_DIR}/${TOP}"
cd "${BD}"

# Build with Intel compilers directly (NOT via any MPI wrapper).
./configure \
  --prefix="${PREFIX}" \
  CC="${CC}" CXX="${CXX}" FC="${FC}" \
  CFLAGS="${OPTFLAGS}" CXXFLAGS="${OPTFLAGS}" FCFLAGS="${OPTFLAGS}" \
  --enable-mpi-fortran=yes \
  --disable-sphinx \
  2>&1 | tee "${LOG_DIR}/openmpi_configure.log"

make -j "${NPROC}" 2>&1 | tee "${LOG_DIR}/openmpi_make.log"
make install 2>&1 | tee "${LOG_DIR}/openmpi_install.log"

log "OpenMPI installed: ${PREFIX}"
"${PREFIX}/bin/mpicc" --version 2>/dev/null | head -1 || warn "OpenMPI mpicc not runnable"
