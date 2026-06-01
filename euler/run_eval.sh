#!/bin/bash
# Run ONLY the Isaac Sim closed-loop eval against an already-running policy server.
# (Start the server first with: bash euler/start_server.sh)
#   bash /cluster/scratch/rjiang/dreamzero/euler/run_eval.sh [EPISODES] [INSTRUCTION] [HORIZON]
# e.g.
#   bash euler/run_eval.sh 1 "pick up the cube" 24
set -uo pipefail
source "$(dirname "$0")/scripts/_env.sh"

dz_on_gpu_node || { echo "ERROR: no GPU visible — start a GPU srun session first."; exit 1; }
[ -f "$DZ_SIF" ] || { echo "ERROR: Isaac SIF not found ($DZ_SIF). Build still running?"; exit 1; }
if ! dz_server_running; then
  echo "ERROR: policy server not healthy on :${DZ_PORT}. Start it first: bash euler/start_server.sh"
  exit 1
fi

EP="${1:-$DZ_EPISODES}"; INSTR="${2:-$DZ_INSTRUCTION}"; HOR="${3:-$DZ_HORIZON}"
dz_run_isaac_eval "$EP" "$INSTR" "$HOR"
rc=$?
vid="$(dz_latest_video)"
[ -n "$vid" ] && echo "Rollout video: $vid"
exit $rc
