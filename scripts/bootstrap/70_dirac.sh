#!/usr/bin/env bash
# 70_dirac.sh — build DIRAC (Tier-2 reference generator, alongside OpenMolcas).
#
# ROLE: external cross-check only. DIRAC provides 4c/2c relativistic SCF, KRCI/GASCI CI,
# and magnetic/SOC property matrices directly comparable to kuiva output. Built
# MPI-parallel on Intel MPI + MKL. NOT a runtime dependency of kuiva.
#
# INTEGER SIZE: DIRAC is built with 32-bit integers (DIRAC's well-tested default), NOT the
# --int64/ILP64 used for the GA/OpenMolcas chain. Reason: with --int64, DIRAC's `use mpi`
# calls become 8-byte while Intel MPI's standard Fortran interface is 4-byte, and DIRAC's
# runtime MPI self-test aborts ("MPI self test failed") — Intel MPI's ILP64 interface
# (-lmpi_ilp64) is not wired up by DIRAC's setup. 32-bit integers keep Intel MPI parallelism
# working and are ample for Tier-2 reference calculations on small systems. DIRAC is a
# standalone reference tool and need not match the project's int64 dependency chain.
#
# Compiler: DIRAC 26.1 ships explicit ifx support (-DVAR_IFX in cmake/custom/compiler_flags/
# Intel.Fortran.cmake), so we use ifx/icx/icpx via the Intel MPI wrappers.
#
# ExaCorr's tensor backends (ExaTensor, TBLIS) are DISABLED: they are heavy/fragile to build
# and only power relativistic coupled cluster, which our SOC-energy / magnetic-moment
# cross-checks do not need. Re-enable with --exatensor=ON --tblis=ON if ExaCorr is required.
# PElib (polarizable embedding) is DISABLED: not needed for our SOC/property references.
#
# Reference: DIRAC, a relativistic ab initio electronic structure program, Release DIRAC26
# (2026), written by H. J. Aa. Jensen, R. Bast, A. S. P. Gomes, T. Saue, L. Visscher, and
# others (see https://doi.org/10.5281/zenodo.and the DIRAC author list; https://diracprogram.org).
# T. Saue et al., "The DIRAC code for relativistic molecular calculations",
# J. Chem. Phys. 152 (2020) 204104. DOI:10.1063/5.0004844.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
print_toolchain

TB="DIRAC-${DIRAC_VERSION}-Source.tar.gz"
TOP="DIRAC-${DIRAC_VERSION}-Source"
PREFIX="${INSTALL_DIR}/dirac"
extract_to_build "${TB}" "${TOP}"

SRCROOT="${BUILD_DIR}/${TOP}"
BD="${SRCROOT}/build"
rm -rf "${BD}"

# DIRAC has a few very large Fortran translation units; cap parallelism to avoid OOM on
# memory-limited nodes (this dev box has 15 GB). Override with DIRAC_NPROC.
DIRAC_NPROC="${DIRAC_NPROC:-$(( NPROC < 4 ? NPROC : 4 ))}"

# Steer the Intel MPI wrappers to ifx/icx/icpx (also set in common.sh; repeated for clarity).
# ⚠ The wrappers themselves come from common.sh as mpiicx/mpiicpx/mpiifx: mpiicc/mpiicpc/
# mpiifort wrap the classic icc/icpc/ifort, removed from oneAPI after 2024.
export I_MPI_F90=ifx I_MPI_CC=icx I_MPI_CXX=icpx

cd "${SRCROOT}"
log "configuring DIRAC (Intel MPI + MKL, 32-bit int, ExaTensor/TBLIS/PElib off)"
python3 ./setup \
  --fc="${MPIFC}" --cc="${MPICC}" --cxx="${MPICXX}" \
  --mpi \
  --qmkl=sequential \
  --pelib=OFF \
  --exatensor=OFF \
  --tblis=OFF \
  --type=release \
  --prefix="${PREFIX}" \
  "${BD}" 2>&1 | tee "${LOG_DIR}/dirac_setup.log"

cd "${BD}"
make -j "${DIRAC_NPROC}" 2>&1 | tee "${LOG_DIR}/dirac_make.log"
make install 2>&1 | tee "${LOG_DIR}/dirac_install.log"

log "DIRAC installed: ${PREFIX}"
# Locate the pam run driver (install layout varies: prefix root, bin/, or share/).
PAM="$(find "${PREFIX}" -maxdepth 3 -name pam -type f 2>/dev/null | head -1)"
[[ -n "${PAM}" ]] && log "pam driver: ${PAM}" || warn "pam driver not found under ${PREFIX}"
