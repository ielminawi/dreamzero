#!/bin/bash
#SBATCH --job-name=dz_evalsweep
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=14G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:90g
#SBATCH --tmp=120G
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/evalsweep.%j.log
# Open-loop (NON-SIM) sweep: action_eval.py over all varnorm checkpoints, enriched with
# per-chunk motion-capture ratio + direction cosine + all_traces.npz. One GPU, AgiBot staged once.
set -uo pipefail
DZ_REPO=/cluster/scratch/rjiang/dreamzero
DATA="${DATA:-$DZ_REPO/data/franka_orca_lerobot_joints_clean}"
CKPT_DIR="$DZ_REPO/checkpoints/dreamzero_franka_orca_lora_joints_varnorm"
STEPS="${STEPS:-1000 2000 3000 4000 5000}"
EPISODES="${EPISODES:-2,5,32,73,147}"; PER_EP="${PER_EP:-24}"
source "$DZ_REPO/euler/scripts/_env.sh"; dz_server_env
export ATTENTION_BACKEND=torch ENABLE_TENSORRT=True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN="$(dz_local_rundir)"; echo "==> node $(hostname) run $RUN"; nvidia-smi -L | head
cd "$RUN"
for s in $STEPS; do
  CKPT="$CKPT_DIR/checkpoint-$s"
  [ -f "$CKPT/model.safetensors" ] || { echo "== skip step $s (no model.safetensors)"; continue; }
  OUT="$DZ_REPO/output/action_eval_varnorm_${s}_full"; mkdir -p "$OUT"
  cf="$CKPT/experiment_cfg/conf.yaml"
  [ -f "$cf" ] && sed -i 's|^pretrained_model_path:.*|pretrained_model_path: ./checkpoints/DreamZero-AgiBot|' "$cf"
  echo "==================== EVAL checkpoint-$s -> $OUT ($(date)) ===================="
  "$DZ_REPO/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 --master-port 12365 \
    "$DZ_REPO/eval_utils/action_eval.py" --model-path "$CKPT" --dataset-dir "$DATA" \
    --episodes "$EPISODES" --per-episode "$PER_EP" --out-dir "$OUT" \
    && echo "== done step $s" || echo "== FAILED step $s"
done
echo "==> SWEEP DONE $(date)"
