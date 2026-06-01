#!/bin/bash
#SBATCH --job-name=dz_isaac_build
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/isaac_build_%j.log
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --account=ls_polle
# Builds isaac-sim.sif from euler/isaac.def with --fakeroot on a compute node.
set -euo pipefail
module load eth_proxy   # apptainer docker:// pull honors HTTPS_PROXY

export APPTAINER_CACHEDIR=/cluster/scratch/rjiang/.apptainer/cache
export APPTAINER_TMPDIR=/cluster/scratch/rjiang/.apptainer/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

DEF=/cluster/scratch/rjiang/dreamzero/euler/isaac.def
SIF=/cluster/scratch/rjiang/dreamzero/isaac-sim.sif

echo "Building $SIF from $DEF ..."
apptainer build --fakeroot --force "$SIF" "$DEF"
echo "ISAAC_SIF_BUILD_COMPLETE"
ls -lh "$SIF"
