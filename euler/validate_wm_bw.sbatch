#!/bin/bash
#SBATCH --job-name=dz_wmval
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:90g
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/wmval.%j.log

# World-model video-prediction validation vs ground-truth LeRobot episodes.
# Runs the validated offline inference path (GrootSimPolicy.lazy_joint_forward_causal)
# on an A100-80GB node, then aggregates metrics + a horizon-degradation curve.
set -uo pipefail

DZ_REPO=/cluster/scratch/rjiang/dreamzero
VENV="$DZ_REPO/.venv"
CKPT=/cluster/scratch/rjiang/checkpoints/dreamzero_franka_orca_lora/checkpoint-20000
DATA="$DZ_REPO/data/franka_orca_lerobot"
OUT=/cluster/scratch/rjiang/wm_validation/checkpoint-20000

# Shared inference env + NVMe staging helpers (dz_server_env, dz_local_rundir).
source "$DZ_REPO/euler/scripts/_env.sh"
dz_server_env                       # DS_ACCELERATOR=cpu, HF_HUB_OFFLINE, TOKENIZER_DIR, ...
export ATTENTION_BACKEND=torch   # SDPA (Blackwell sm_120 safe)
export ENABLE_TENSORRT=True         # eager, skip torch.compile (fast startup, no recompiles)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation
export MASTER_ADDR=localhost MASTER_PORT=12357

# Stage the 43GB DreamZero-AgiBot base to node-local NVMe (conf.yaml loads it via a
# cwd-relative path); avoids the ~8MB/s Lustre mmap crawl seen on the first attempt.
RUN="$(dz_local_rundir)"
echo "==> node: $(hostname)  run-dir: $RUN  gpu:"; nvidia-smi -L 2>&1 | head
echo "==> ATTENTION_BACKEND=$ATTENTION_BACKEND  starting validation $(date)"

cd "$RUN"
"$VENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$DZ_REPO/eval_utils/validate_wm.py" \
  --model-path "$CKPT" \
  --dataset-dir "$DATA" \
  --embodiment franka_orca_bimanual \
  --episodes "2,5,32,73,147" \
  --starts "50,800,1600,2400" \
  --num-steps 8 \
  --frames-per-chunk 4 \
  --instruction "bag the groceries" \
  --out-dir "$OUT" \
  --fps-out 10 \
  --max-align-off 6
RC=$?
echo "==> validate_wm exit code: $RC ($(date))"

"$VENV/bin/python" "$DZ_REPO/eval_utils/aggregate_wm.py" --out-dir "$OUT" --checkpoint checkpoint-20000
echo "==> aggregate exit code: $? ($(date))"
echo "==> outputs in $OUT"
