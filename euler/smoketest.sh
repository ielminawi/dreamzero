#!/bin/bash
# FAST sanity check of the policy server WITHOUT Isaac Sim: starts the server (if needed),
# waits until healthy, then fires the 48-dim franka_orca client for a few inference chunks.
# Good first test after starting a GPU session — confirms the model + LoRA load and infer.
#   bash /cluster/scratch/rjiang/dreamzero/euler/smoketest.sh
set -uo pipefail
source "$(dirname "$0")/scripts/_env.sh"

dz_on_gpu_node || { echo "ERROR: no GPU visible — start a GPU srun session first."; exit 1; }

started_here=0
if ! dz_server_running; then dz_start_server_bg; started_here=1; fi
dz_wait_health 1800 || exit 1

echo "==> Running 48-dim smoke client (3 chunks)..."
( cd "$DZ_REPO" && NO_PROXY=localhost,127.0.0.1 "$DZ_VENV/bin/python" test_client_franka_orca.py \
    --host localhost --port "$DZ_PORT" --num-chunks 3 --prompt "$DZ_INSTRUCTION" )
rc=$?
echo "==> Smoke client exit=$rc"
[ "$started_here" -eq 1 ] && echo "(server left running; stop with: bash euler/stop_server.sh)"
exit $rc
