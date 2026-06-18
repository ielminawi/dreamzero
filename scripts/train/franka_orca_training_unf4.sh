#!/bin/bash
# DreamZero Franka+Orca — WARM-START + UNFREEZE-TOP-4 variant.
#
# Fixes the under-conditioned action head: a frozen-DiT LoRA (any rank) cannot route
# action<->vision attention (rank4 ~= rank16 in eval). This run additionally FULL-finetunes
# the top-4 DiT blocks, warm-started from the rank-16 checkpoint so the franka adaptation
# carries over. Keeps rank/alpha=16 (alpha bump would 2x the warm-started LoRA scaling).
#
# Env knobs (with defaults):
#   OUTPUT_DIR            output checkpoint dir
#   WARMSTART_LORA_PATH   prior LoRA ckpt to warm-start delta+MLPs from (post-injection load)
#   UNFREEZE_TOP_K        number of top DiT blocks to full-finetune (default 4)
#   MAX_STEPS / SAVE_STEPS
set -eo pipefail
export HYDRA_FULL_ERROR=1
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-torch}

FRANKA_ORCA_DATA_ROOT=${FRANKA_ORCA_DATA_ROOT:-"./data/franka_orca_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_franka_orca_lora_r16unf4"}
WARMSTART_LORA_PATH=${WARMSTART_LORA_PATH:-"./checkpoints/dreamzero_franka_orca_lora_r16/checkpoint-6000"}
UNFREEZE_TOP_K=${UNFREEZE_TOP_K:-4}

if [ -z "${NUM_GPUS}" ]; then NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l); fi
NUM_GPUS=${NUM_GPUS:-1}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}

if [ ! -d "$FRANKA_ORCA_DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $FRANKA_ORCA_DATA_ROOT"; exit 1
fi

echo "==> unf4 run: OUTPUT_DIR=$OUTPUT_DIR  WARMSTART=$WARMSTART_LORA_PATH  UNFREEZE_TOP_K=$UNFREEZE_TOP_K  MAX_STEPS=${MAX_STEPS:-14000}"

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=${REPORT_TO:-none} \
    data=dreamzero/franka_orca_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=2 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=5e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=${SAVE_STEPS:-2000} \
    training_args.warmup_ratio=0.10 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=2 \
    max_steps=${MAX_STEPS:-14000} \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=4 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    max_state_dim=48 \
    max_action_dim=48 \
    save_strategy=steps \
    franka_orca_data_root=$FRANKA_ORCA_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=${PRETRAINED_PATH:-./checkpoints/DreamZero-AgiBot} \
    ++warmstart_lora_path=$WARMSTART_LORA_PATH \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    ++action_head_cfg.config.lora_rank=16 \
    ++action_head_cfg.config.lora_alpha=16 \
    ++action_head_cfg.config.unfreeze_top_k_blocks=$UNFREEZE_TOP_K
