#!/bin/bash
# ============================================================================
# DreamZero END-TO-END inference test (policy server + Isaac Sim, dual Franka+Orca)
# Run this from inside your GPU interactive session (srun ... --pty bash).
#
#   bash /cluster/scratch/rjiang/dreamzero/euler/run_e2e.sh
#
# Optional overrides:
#   DZ_INSTRUCTION="pick up the cube"  DZ_EPISODES=1  DZ_HORIZON=24  bash .../run_e2e.sh
#
# It will: (1) start the policy server, (2) wait until healthy, (3) run the Isaac
# closed-loop eval against it, (4) print the output rollout video path, (5) stop the server.
# ============================================================================
set -uo pipefail
source "$(dirname "$0")/scripts/_env.sh"

echo "########## DreamZero end-to-end eval ##########"
if ! dz_on_gpu_node; then
  echo "ERROR: no GPU visible. Run this inside the GPU srun session:"
  echo "  srun --time=4:00:00 --ntasks=1 --cpus-per-task=16 --mem-per-cpu=16G \\"
  echo "       --gpus-per-node=1 --gres=gpumem:96g --account=ls_polle --pty bash"
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/GPU: /'

# Preconditions
miss=0
[ -x "$DZ_VENV/bin/python" ] || { echo "MISSING: venv ($DZ_VENV)"; miss=1; }
[ -f "$DZ_SIF" ]            || { echo "MISSING: Isaac SIF ($DZ_SIF)"; miss=1; }
[ -d "$DZ_REPO/checkpoints/Wan2.1-I2V-14B-480P" ] || { echo "MISSING: Wan2.1 weights"; miss=1; }
[ -d "$DZ_REPO/checkpoints/DreamZero-AgiBot" ]    || { echo "MISSING: DreamZero-AgiBot weights"; miss=1; }
[ -f "$DZ_LORA/model.safetensors" ] || { echo "MISSING: LoRA checkpoint ($DZ_LORA)"; miss=1; }
ls "$DZ_REPO"/sim_envs/assets/orcahand_description/* >/dev/null 2>&1 || { echo "MISSING: orcahand_description submodule meshes"; miss=1; }
[ "$miss" -eq 0 ] || { echo "Preconditions not met (run: bash euler/preflight.sh). Aborting."; exit 1; }

trap 'echo; echo "Cleaning up server..."; dz_stop_server' EXIT

dz_start_server_bg
dz_wait_health 1800 || exit 1

echo; echo "########## Launching Isaac Sim closed-loop eval ##########"
dz_run_isaac_eval "$DZ_EPISODES" "$DZ_INSTRUCTION" "$DZ_HORIZON"
rc=$?

echo; echo "########## Done (eval exit=$rc) ##########"
vid="$(dz_latest_video)"
if [ -n "$vid" ]; then echo "Rollout video: $vid"; else echo "No video found under $DZ_OUTPUT (check Isaac output above)."; fi
exit $rc
