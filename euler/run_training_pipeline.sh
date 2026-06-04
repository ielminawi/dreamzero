#!/bin/bash
# One command to launch the whole chain on Euler, with SLURM dependencies:
#   preprocess (array) --afterok--> finalize+GEAR --afterok--> 100h training
#
# Run this ONLY after eyeballing the background-removal gallery
# (output/bg_gallery/) and deciding BG=segformer (default) or BG=none.
#
# Usage:
#   bash euler/run_training_pipeline.sh                 # bg removal ON (segformer)
#   BG=none bash euler/run_training_pipeline.sh         # baseline, no bg removal
set -euo pipefail
cd /cluster/scratch/rjiang/dreamzero

export RAW=${RAW:-/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries}
export DATA_ROOT=${DATA_ROOT:-/cluster/scratch/rjiang/dreamzero/data/franka_orca_lerobot}
export NUM_SHARDS=${NUM_SHARDS:-10}
export BG=${BG:-mask2former}

echo "RAW=$RAW"
echo "DATA_ROOT=$DATA_ROOT  NUM_SHARDS=$NUM_SHARDS  BG=$BG"

# 1) preprocess array (exports RAW/DATA_ROOT/NUM_SHARDS/BG via --export)
PREP=$(sbatch --parsable --export=ALL,RAW,DATA_ROOT,NUM_SHARDS,BG \
        euler/preprocess_bag_groceries.sbatch)
echo "submitted preprocess array: $PREP"

# 2) finalize + GEAR + validation, after the whole array succeeds
GEAR=$(sbatch --parsable --dependency=afterok:$PREP --export=ALL,RAW,DATA_ROOT \
        euler/finalize_gear.sbatch)
echo "submitted finalize/GEAR: $GEAR (afterok:$PREP)"

# 3) 100h training, after the dataset is finalized + validated
TRAIN=$(sbatch --parsable --dependency=afterok:$GEAR --export=ALL,DATA_ROOT \
        euler/train_franka_orca_100h.sbatch)
echo "submitted training: $TRAIN (afterok:$GEAR)"

echo
echo "chain: preprocess $PREP -> gear $GEAR -> train $TRAIN"
echo "watch:  squeue -u $USER ; tail -f euler/logs/prep.${PREP}_*.log"
