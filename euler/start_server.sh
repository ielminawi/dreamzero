#!/bin/bash
# Start ONLY the DreamZero policy server (background) and wait until healthy.
# Use this when you want to keep the server up and run several evals / smoketests.
#   bash /cluster/scratch/rjiang/dreamzero/euler/start_server.sh
# Then in the same session: bash euler/run_eval.sh   or   bash euler/smoketest.sh
# Stop it later with: bash euler/stop_server.sh
set -uo pipefail
source "$(dirname "$0")/scripts/_env.sh"

dz_on_gpu_node || { echo "ERROR: no GPU visible — start a GPU srun session first."; exit 1; }
dz_start_server_bg
dz_wait_health 1800 || { echo "Server failed to come up."; exit 1; }
echo "Server is up on :${DZ_PORT}. Leave this session open; run evals from another shell on the same node,"
echo "or just run:  bash $DZ_REPO/euler/run_eval.sh"
