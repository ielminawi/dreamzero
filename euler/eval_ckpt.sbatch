#!/bin/bash
#SBATCH --job-name=dz_evalckpt
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:80g
#SBATCH --tmp=140G
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/evalckpt.%j.log
# Evaluate ONE checkpoint: action-prediction accuracy (vs stay-still) + wrong-action
# ablation (does the video obey actions?). Pass CKPT=<checkpoint dir> OUT=<out dir>.
set -uo pipefail
DZ_REPO=/cluster/scratch/rjiang/dreamzero
DATA="$DZ_REPO/data/franka_orca_lerobot"
: "${CKPT:?set CKPT}"; : "${OUT:?set OUT}"
source "$DZ_REPO/euler/scripts/_env.sh"; dz_server_env
export ATTENTION_BACKEND=FA2 ENABLE_TENSORRT=True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=localhost MASTER_PORT=12377
RUN="$(dz_local_rundir)"; echo "==> node $(hostname) ckpt=$CKPT out=$OUT"; nvidia-smi -L|head
# Stage the Wan-14B base to node-local NVMe too (sequential cp ~1GB/s) and repoint the
# RUN symlink, so the model loads via FAST local mmap instead of ~8MB/s Lustre mmap.
if [ -n "${TMPDIR:-}" ] && [ ! -s "$TMPDIR/wanstage/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" ]; then
  echo "staging Wan base -> $TMPDIR/wanstage (NVMe) ..."; mkdir -p "$TMPDIR/wanstage"
  cp -r "$DZ_REPO/checkpoints/Wan2.1-I2V-14B-480P" "$TMPDIR/wanstage/" \
    && ln -sfn "$TMPDIR/wanstage/Wan2.1-I2V-14B-480P" "$RUN/checkpoints/Wan2.1-I2V-14B-480P" \
    && echo "Wan base staged + symlink repointed." || echo "WARN: Wan staging failed, using Lustre"
fi
mkdir -p "$OUT"; cd "$RUN"
echo "=== [1/2] action accuracy ==="
"$DZ_REPO/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$DZ_REPO/eval_utils/action_eval.py" --model-path "$CKPT" --dataset-dir "$DATA" \
  --episodes "2,5,32" --per-episode 6 --out-dir "$OUT/action_accuracy" ; echo "action_eval rc=$?"
if [ "${RUN_ABLATION:-1}" = "1" ]; then
echo "=== [2/2] wrong-action ablation ==="
"$DZ_REPO/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$DZ_REPO/eval_utils/openloop.py" --model-path "$CKPT" --dataset-dir "$DATA" \
  --clip "32:1600" --other-ep 73 --modes "gtsim,gtsim_zero,dream" \
  --num-steps 10 --horizons "5,10,15,20,25" --out-dir "$OUT/ablation" ; echo "ablation rc=$?"
else echo "=== ablation skipped (RUN_ABLATION=0) ==="; fi
echo "==> eval done $(date); outputs in $OUT"
