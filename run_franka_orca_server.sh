#!/usr/bin/env bash
# Launch the DreamZero inference server for the two-Franka + Orca-hand setup
# (franka_orca_bimanual embodiment) on a single H100.
#
# Usage:  ./run_franka_orca_server.sh [PORT]
# Then test with:  .venv/bin/python test_client_franka_orca.py --port <PORT>
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-5000}"
LORA_CKPT="/home/ubuntu/checkpoints/dreamzero_franka_orca_lora/checkpoint-20000"

# ENABLE_TENSORRT=True -> run in EAGER mode (disables torch.compile).
#   torch.compile (reduce-overhead/CUDA-graphs) crashes on session reset with an
#   "FX symbolic trace of a dynamo-optimized function" error. Eager mode is stable
#   across sessions and still fast (~5 s/inference with dit-cache on H100).
# ATTENTION_BACKEND falls back to FA2 automatically (Transformer Engine not installed).
# TOKENIZER_DIR points the umt5-xxl tokenizer at a local copy (cluster path in the
#   checkpoint config is dead). The base weights come from ./checkpoints/DreamZero-AgiBot
#   (resolved via experiment_cfg/conf.yaml::pretrained_model_path).

CUDA_VISIBLE_DEVICES=0 \
NO_ALBUMENTATIONS_UPDATE=1 \
TOKENIZERS_PARALLELISM=false \
ENABLE_TENSORRT=True \
TOKENIZER_DIR=/home/ubuntu/dreamzero/checkpoints/umt5-xxl \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  socket_test_optimized_AR.py \
  --port "$PORT" \
  --enable-dit-cache \
  --model-path "$LORA_CKPT" \
  --embodiment franka_orca_bimanual
