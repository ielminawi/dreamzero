#!/bin/bash
#SBATCH --job-name=dz_openloop_bw
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:90g
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/openloop.%j.log
set -uo pipefail
DZ_REPO=/cluster/scratch/rjiang/dreamzero
CKPT=/cluster/scratch/rjiang/checkpoints/dreamzero_franka_orca_lora/checkpoint-20000
OUT=/cluster/scratch/rjiang/wm_validation/openloop_v3
source "$DZ_REPO/euler/scripts/_env.sh"; dz_server_env
export ATTENTION_BACKEND=torch ENABLE_TENSORRT=True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=localhost MASTER_PORT=12359
RUN="$(dz_local_rundir)"
echo "==> node $(hostname)  run $RUN"; nvidia-smi -L | head
cd "$RUN"
"$DZ_REPO/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$DZ_REPO/eval_utils/openloop.py" \
  --model-path "$CKPT" --dataset-dir "$DZ_REPO/data/franka_orca_lerobot" \
  --clip "32:1600" --modes "gtsim,dream,polact" --num-steps 10 --frames-per-chunk 4 \
  --horizons "5,10,15,20,25" --instruction "bag the groceries" --out-dir "$OUT"
echo "==> openloop exit $? ($(date)); outputs in $OUT"
