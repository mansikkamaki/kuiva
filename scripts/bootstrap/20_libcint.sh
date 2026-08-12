#!/usr/bin/env bash
# 20_libcint.sh — build libcint (Sun Qiming) with icx + Intel MKL.
#
# libcint provides the raw Gaussian integrals. PySCF bundles its own copy for the
# front-end; this STANDALONE build exists so kuiva's future compiled (C++/pybind11)
# integral kernels can link libcint directly (sandbox install/libcint).
#
# Reference: Q. Sun, "Libcint: An efficient general integral library for Gaussian
# basis functions", J. Comput. Chem. 36 (2015) 1664. DOI:10.1002/jcc.23981.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
print_toolchain

TB="libcint-${LIBCINT_VERSION}.tar.gz"
TOP="libcint-${LIBCINT_VERSION}"
PREFIX="${INSTALL_DIR}/libcint"
extract_to_build "${TB}" "${TOP}"

BD="${BUILD_DIR}/${TOP}/build"
rm -rf "${BD}"; mkdir -p "${BD}"

# Link BLAS against sequential MKL (libcint is threaded at the shell/OpenMP level; a
# sequential BLAS avoids nested oversubscription).
cmake -S "${BUILD_DIR}/${TOP}" -B "${BD}" \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_C_FLAGS="${OPTFLAGS}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBLA_VENDOR=Intel10_64lp_seq \
  -DWITH_FORTRAN=OFF \
  -DWITH_CINT2_INTERFACE=OFF \
  -DWITH_RANGE_COULOMB=ON \
  -DENABLE_EXAMPLE=OFF \
  -DENABLE_TEST=OFF \
  -DBUILD_SHARED_LIBS=ON

cmake --build "${BD}" -j "${NPROC}"
cmake --install "${BD}"
log "libcint installed: ${PREFIX}"
ls -1 "${PREFIX}/lib"*/libcint.so* 2>/dev/null || warn "libcint.so not found under ${PREFIX}"
