#!/usr/bin/env bash
# 40_hdf5.sh — build HDF5 (C, serial) with icx.
#
# WHY FROM SOURCE: the system has HDF5 runtime libs only (no -dev headers), and the no-sudo sandbox rule
# forbids sudo/system installs. OpenMolcas needs the HDF5 C library + headers.
# A serial C build is sufficient (OpenMolcas uses the HDF5 C API; its parallelism
# comes from Global Arrays / MPI, not from parallel HDF5).
#
# Reference: The HDF Group, Hierarchical Data Format version 5. https://www.hdfgroup.org/HDF5/
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
print_toolchain

PREFIX="${INSTALL_DIR}/hdf5"
TOP="hdf5-${HDF5_VERSION}"
TB="${TOP}.tar.gz"

# Fetch if not staged in external/src (HDF5 is a network dependency, not a pre-staged tarball).
if [[ ! -f "${SRC_DIR}/${TB}" ]]; then
  MAJMIN="${HDF5_VERSION%.*}"
  URL="https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${MAJMIN}/hdf5-${HDF5_VERSION}/src/${TB}"
  log "downloading HDF5 ${HDF5_VERSION}"
  wget -q -O "${SRC_DIR}/${TB}" "${URL}" || die "HDF5 download failed: ${URL}"
fi

extract_to_build "${TB}" "${TOP}"
BD="${BUILD_DIR}/${TOP}"
cd "${BD}"

CC="${CC}" CFLAGS="${OPTFLAGS}" ./configure \
  --prefix="${PREFIX}" \
  --enable-build-mode=production \
  --disable-parallel \
  --disable-fortran \
  --disable-cxx \
  2>&1 | tee "${LOG_DIR}/hdf5_configure.log"

make -j "${NPROC}" 2>&1 | tee "${LOG_DIR}/hdf5_make.log"
make install 2>&1 | tee "${LOG_DIR}/hdf5_install.log"
log "HDF5 installed: ${PREFIX}"
"${PREFIX}/bin/h5cc" -showconfig 2>/dev/null | grep -iE "version|compiler" | head -3 || warn "h5cc not runnable"
