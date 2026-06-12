#!/bin/bash
#SBATCH --job-name=dz_actacc
#SBATCH --time=00:25:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:80g
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/actacc.%j.log
set -uo pipefail
DZ_REPO=/cluster/scratch/rjiang/dreamzero
CKPT=/cluster/scratch/rjiang/checkpoints/dreamzero_franka_orca_lora/checkpoint-20000
OUT=/cluster/scratch/rjiang/wm_validation/action_accuracy
source "$DZ_REPO/euler/scripts/_env.sh"; dz_server_env
export ATTENTION_BACKEND=FA2 ENABLE_TENSORRT=True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=localhost MASTER_PORT=12361
RUN="$(dz_local_rundir)"; echo "==> node $(hostname) run $RUN"; nvidia-smi -L|head
cd "$RUN"
"$DZ_REPO/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$DZ_REPO/eval_utils/action_eval.py" \
  --model-path "$CKPT" --dataset-dir "$DZ_REPO/data/franka_orca_lerobot" \
  --episodes "2,5,32,73,147" --per-episode 8 --out-dir "$OUT"
echo "==> action_eval exit $? ($(date)); outputs in $OUT"
