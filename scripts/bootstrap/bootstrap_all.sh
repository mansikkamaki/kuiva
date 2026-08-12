#!/usr/bin/env bash
# bootstrap_all.sh — build the full external toolchain, in dependency order.
# Idempotent-ish: each step rebuilds its own tree. Logs land in external/logs/.
#
# ⚠ DEVELOPER AND REFERENCE TOOLING — NOT AN INSTALLATION ROUTE. Running Kuiva needs a
# Python environment and nothing else: `source setup.sh` checks for one and tells you what
# is missing. What is built here is the machinery for *developing* Kuiva and for generating
# the external reference data it is validated against — OpenMolcas and DIRAC above all,
# which take hours to build and which no user of the program ever needs. The optional
# compiled kernel backend has its own, much smaller build (cpp/configure), and even that is
# optional: KUIVA_KERNELS=numpy is a fully supported way to run.
#
# Usage: scripts/bootstrap/bootstrap_all.sh [step ...]
#   with no args, runs every step below in order.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STEPS=(
  10_python_env.sh     # intelpython venv + pinned PySCF
  20_libcint.sh        # libcint (icx + MKL)
  30_openmpi.sh        # OpenMPI reference build (Intel MPI is the default)
  40_hdf5.sh           # HDF5 C serial (OpenMolcas dep)
  50_globalarrays.sh   # Global Arrays on Intel MPI + MKL ILP64 (OpenMolcas dep)
  60_openmolcas.sh     # OpenMolcas Tier-2 reference generator
  70_dirac.sh          # DIRAC Tier-2 reference generator
)

to_run=("$@"); [[ ${#to_run[@]} -eq 0 ]] && to_run=("${STEPS[@]}")
for s in "${to_run[@]}"; do
  echo "==================== ${s} ===================="
  bash "${HERE}/${s}"
done
echo "All requested bootstrap steps completed."
