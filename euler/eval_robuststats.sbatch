#!/bin/bash
#SBATCH --job-name=dz_evalrs
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=14G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:90g
#SBATCH --tmp=120G
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/evalrs.%j.log
# Decisive test of the robust-stats FIX: action-prediction MAE of the latest robust-stats
# checkpoint vs the stay-still floor, on the robust-stats dataset. The fix works if the
# policy now BEATS stay-still (overall stay MAE ~0.054 rad; old run was 0.111, lost).
set -uo pipefail
DZ_REPO=/cluster/scratch/rjiang/dreamzero
DATA="${DATA:-$DZ_REPO/data/franka_orca_lerobot_joints_robuststats}"
CKPT_DIR="${CKPT_DIR:-$DZ_REPO/checkpoints/dreamzero_franka_orca_lora_joints_robuststats}"
CKPT="${CKPT:-$(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)}"
OUT="${OUT:-$DZ_REPO/output/action_eval_robuststats}"
EPISODES="${EPISODES:-2,5,32,73,147}"; PER_EP="${PER_EP:-8}"
echo "==> ckpt=$CKPT  data=$DATA"
[ -d "$CKPT" ] || { echo "ERROR: no robust-stats checkpoint found in $CKPT_DIR"; exit 1; }
source "$DZ_REPO/euler/scripts/_env.sh"; dz_server_env
export ATTENTION_BACKEND=torch ENABLE_TENSORRT=True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# conf base-path fix (dead TMPDIR AgiBot path -> relative, resolves to NVMe in the rundir)
cf="$CKPT/experiment_cfg/conf.yaml"
[ -f "$cf" ] && sed -i 's|^pretrained_model_path:.*|pretrained_model_path: ./checkpoints/DreamZero-AgiBot|' "$cf" && echo "patched $cf"
RUN="$(dz_local_rundir)"; echo "==> node $(hostname) run $RUN"; nvidia-smi -L | head
cd "$RUN"; mkdir -p "$OUT"
"$DZ_REPO/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 --master-port 12364 \
  "$DZ_REPO/eval_utils/action_eval.py" --model-path "$CKPT" --dataset-dir "$DATA" \
  --episodes "$EPISODES" --per-episode "$PER_EP" --out-dir "$OUT"
echo "==> robust-stats eval done $(date); see $OUT/action_accuracy.json (.png). Compare policy_mae vs stay_mae."
