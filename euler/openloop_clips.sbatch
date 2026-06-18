#!/bin/bash
#SBATCH --job-name=dz_openloop_clips
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:90g
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/openloop_clips.%j.log
set -uo pipefail
DZ_REPO=/cluster/scratch/rjiang/dreamzero
CKPT=/cluster/scratch/rjiang/checkpoints/dreamzero_franka_orca_lora/checkpoint-20000
OUT=/cluster/scratch/rjiang/wm_validation/openloop_clips
# high-motion clips (ep:start) with >=10s of GT runway, from state-velocity scan
CLIPS="239:2464,179:2011,149:1248"
# horizons in FRAMES @ 50fps = 1s,3s,5s,10s
HZ="50,150,250,500"
mkdir -p "$OUT"
source "$DZ_REPO/euler/scripts/_env.sh"; dz_server_env
export ATTENTION_BACKEND=torch ENABLE_TENSORRT=True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=localhost MASTER_PORT=12359
RUN="$(dz_local_rundir)"
echo "==> node $(hostname)  run $RUN  clips $CLIPS"; nvidia-smi -L | head
cd "$RUN"
"$DZ_REPO/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$DZ_REPO/eval_utils/openloop.py" \
  --model-path "$CKPT" --dataset-dir "$DZ_REPO/data/franka_orca_lerobot" \
  --clip "$CLIPS" --modes "gtsim,dream,polact" --num-steps 60 --frames-per-chunk 4 \
  --horizons "$HZ" --instruction "bag the groceries" --out-dir "$OUT"
echo "==> rollout exit $? ($(date))"
# pretty figures (CPU, no model)
DS_ACCELERATOR=cpu "$DZ_REPO/.venv/bin/python" "$DZ_REPO/eval_utils/replot_openloop.py" \
  --roll-dir "$OUT" --dataset-dir "$DZ_REPO/data/franka_orca_lerobot" \
  --clip "$CLIPS" --modes "gtsim,dream,polact" --horizons "$HZ" --out-dir "$OUT"
echo "==> done ($(date)); outputs in $OUT"
