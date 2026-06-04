#!/bin/bash
#SBATCH --job-name=dz_validate_mini
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=12G
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/validate_mini.%j.log
# Mini end-to-end data check: convert 3 bag_groceries episodes -> LeRobot+GEAR,
# then validate structure + decord video readability. Exits non-zero on failure
# so the GPU smoke test (afterok) only runs on a sane dataset.
set -eo pipefail
module load eth_proxy 2>/dev/null || true
REPO=/cluster/scratch/rjiang/dreamzero
export SRC=/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries
export DST=$REPO/data/franka_orca_lerobot_mini
export NW=3 FPS=50 TASK="bag the groceries" TARGET_RES=640x480 MAX_EPISODES=3
bash "$REPO/scripts/data/preprocess_bag_groceries.sh"
echo "===== running dataset validation ====="
"$REPO/.venv/bin/python" "$REPO/scripts/data/validate_lerobot_dataset.py" "$DST"
echo "==> validate_mini PASSED"
