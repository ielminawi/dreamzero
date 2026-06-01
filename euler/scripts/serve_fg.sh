#!/bin/bash
# Run the policy server in the FOREGROUND (for the persistent-allocation debug pattern:
# srun --jobid=<dev> --overlap --ntasks=1 bash euler/scripts/serve_fg.sh > euler/logs/server.log 2>&1 )
set -uo pipefail
source "$(dirname "$0")/_env.sh"
dz_server_env
RUN="$(dz_local_rundir)"
cd "$RUN"
echo "[serve_fg] host=$(hostname) port=$DZ_PORT model=$DZ_LORA cwd=$RUN"
echo "[serve_fg] ENABLE_TENSORRT=$ENABLE_TENSORRT ATTENTION_BACKEND=$ATTENTION_BACKEND CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
exec "$DZ_VENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 \
  socket_test_optimized_AR.py --port "$DZ_PORT" --enable-dit-cache \
  --embodiment franka_orca_bimanual --model-path "$DZ_LORA"
