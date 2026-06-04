#!/bin/bash
#SBATCH --job-name=dz_preproc_baggroceries
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=11G
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/preproc.%j.log
# Full preprocessing: all 300 bag_groceries episodes -> LeRobot v2 + GEAR metadata
# for franka_orca_bimanual. Output: data/franka_orca_lerobot. Exits non-zero on
# failure so the 100h training (afterok) won't start on a broken dataset.
set -eo pipefail
module load eth_proxy 2>/dev/null || true
REPO=/cluster/scratch/rjiang/dreamzero
export SRC=/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries
export DST=$REPO/data/franka_orca_lerobot
export NW=14 FPS=50 TASK="bag the groceries" TARGET_RES=640x480
bash "$REPO/scripts/data/preprocess_bag_groceries.sh"
echo "===== validating full dataset ====="
"$REPO/.venv/bin/python" "$REPO/scripts/data/validate_lerobot_dataset.py" "$DST"
echo "==> full preprocessing PASSED"
