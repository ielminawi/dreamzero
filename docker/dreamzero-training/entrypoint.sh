#!/bin/bash
set -euo pipefail

echo "============================================"
echo "  DreamZero Training Container"
echo "============================================"

# ---- GPU Detection & Attention Backend ----
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
echo "Detected GPU: $GPU_NAME"

if [ -z "${ATTENTION_BACKEND:-}" ]; then
    if echo "$GPU_NAME" | grep -qi "GH200\|Grace Hopper"; then
        export ATTENTION_BACKEND=torch
        echo "GH200 detected -> ATTENTION_BACKEND=torch"
    elif python -c "import flash_attn" 2>/dev/null; then
        export ATTENTION_BACKEND=flash
        echo "Flash Attention available -> ATTENTION_BACKEND=flash"
    else
        export ATTENTION_BACKEND=torch
        echo "No Flash Attention found -> ATTENTION_BACKEND=torch"
    fi
else
    echo "ATTENTION_BACKEND already set to: $ATTENTION_BACKEND"
fi

# ---- W&B Auto-Login ----
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY found, logging in..."
    wandb login "$WANDB_API_KEY" 2>/dev/null || echo "WARNING: wandb login failed"
    export REPORT_TO=wandb
    echo "W&B enabled (report_to=wandb)"
else
    export REPORT_TO=none
    echo "No WANDB_API_KEY set, logging disabled (report_to=none)"
fi

# ---- Auto-Download Weights ----
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/checkpoints/umt5-xxl"}

if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Downloading Wan2.1-I2V-14B-480P to $WAN_CKPT_DIR..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "Downloading umt5-xxl to $TOKENIZER_DIR..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

# ---- Validate Mounts ----
FRANKA_ORCA_DATA_ROOT=${FRANKA_ORCA_DATA_ROOT:-"/data"}
PRETRAINED_PATH=${PRETRAINED_PATH:-"/checkpoints/DreamZero-AgiBot"}

if [ ! -d "$FRANKA_ORCA_DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $FRANKA_ORCA_DATA_ROOT"
    echo "Mount your dataset directory to /data"
    exit 1
fi

if [ ! -d "$PRETRAINED_PATH" ]; then
    echo "ERROR: DreamZero-AgiBot checkpoint not found at $PRETRAINED_PATH"
    echo "Mount or download it: git clone https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  GPU:               $GPU_NAME"
echo "  ATTENTION_BACKEND: $ATTENTION_BACKEND"
echo "  W&B:               $REPORT_TO"
echo "  Dataset:           $FRANKA_ORCA_DATA_ROOT"
echo "  Wan2.1 Weights:    $WAN_CKPT_DIR"
echo "  Tokenizer:         $TOKENIZER_DIR"
echo "  Pretrained:        $PRETRAINED_PATH"
echo "  Output:            ${OUTPUT_DIR:-/output}"
echo "============================================"
echo ""

# ---- Launch Training ----
exec bash scripts/train/franka_orca_training.sh
