#!/bin/bash
#SBATCH --job-name=dz_jsmoke
#SBATCH --time=03:30:00
#SBATCH --partition=cuda13pr.4h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/jsmoke.%j.log

# ============================================================================
# JOINT-SPACE closed-loop Isaac smoke test for DreamZero.
# Single 1-GPU job (respects the ls_polle 2-GPU cap: training uses the other GPU).
#
# This runs the FULL closed-loop pipeline (policy server + Isaac Sim) in
# JOINT-SPACE mode (DZ_JOINT_ACTIONS=1 -> --joint-actions): the policy's arm
# dims [0:14] are Panda JOINT ANGLES, so we bypass FK on obs and IK on actions.
#
# Usage (smoke, defaults to checkpoint-4000, 1 episode, short horizon):
#   sbatch euler/e2e_joints_smoke.sbatch
# Later, on a mature checkpoint:
#   CKPT=/cluster/scratch/rjiang/dreamzero/checkpoints/dreamzero_franka_orca_lora_joints_clean/checkpoint-14000 \
#     DZ_EPISODES=2 DZ_HORIZON=24 sbatch euler/e2e_joints_smoke.sbatch
# ============================================================================
set -uo pipefail
module load eth_proxy 2>/dev/null || true
cd /cluster/scratch/rjiang/dreamzero

# ---- joint-space checkpoint under test ----
export CKPT="${CKPT:-/cluster/scratch/rjiang/dreamzero/checkpoints/dreamzero_franka_orca_lora_joints_clean/checkpoint-4000}"
export DZ_LORA="$CKPT"
export DZ_JOINT_ACTIONS=1                       # -> run_sim_eval_bimanual.py --joint-actions
export DZ_EPISODES="${DZ_EPISODES:-1}"
export DZ_HORIZON="${DZ_HORIZON:-16}"           # short chunk for a quick smoke
export DZ_INSTRUCTION="${DZ_INSTRUCTION:-bag the groceries}"

echo "==> JOINT-SPACE smoke: CKPT=$CKPT  episodes=$DZ_EPISODES horizon=$DZ_HORIZON"
[ -f "$CKPT/model.safetensors" ] || { echo "ERROR: $CKPT/model.safetensors missing"; exit 1; }

# Repair the dead training-time pretrained_model_path so load_lora rebuilds the base
# from the local DreamZero-AgiBot (relative path), NOT vanilla Wan2.1. Idempotent.
CONF="$CKPT/experiment_cfg/conf.yaml"
if grep -q "^pretrained_model_path: /scratch/tmp\." "$CONF" 2>/dev/null; then
  cp "$CONF" "$CONF.bak.$(date +%s)"
  sed -i 's#^pretrained_model_path: /scratch/tmp\.[0-9]*\.rjiang/dz_agibot#pretrained_model_path: ./checkpoints/DreamZero-AgiBot#' "$CONF"
  echo "==> patched pretrained_model_path -> ./checkpoints/DreamZero-AgiBot"
fi
grep -n "^pretrained_model_path:" "$CONF" || true

bash euler/run_e2e.sh; rc=$?
echo "==> run_e2e exit: $rc"

vid="$(ls -t /cluster/scratch/rjiang/dreamzero/output/sim/*/*/*.mp4 2>/dev/null | head -1)"
[ -n "$vid" ] && echo "==> Rollout video: $vid" || echo "==> No video produced (check log above)."
echo "==> joint smoke done (rc=$rc)"
exit $rc
