#!/bin/bash
# DreamZero Franka+Orca — JOINT-SPACE actions variant (warm-start + unfreeze-top-K).
#
# Trains on data/franka_orca_lerobot_joints: the arm dims of state/action are IK'd
# Panda JOINT ANGLES instead of EE poses (convert_h5_to_lerobot.py --arm-ee-to-joints
# --deglitch-arms). Same warm-start + unfreeze-top-K recipe as r16unf4, plus the
# per-segment WEIGHTED NORMALIZED action loss: arm_l/arm_r/hand_l/hand_r contribute
# equally instead of the flat per-dim mean (which lets the 34 hand dims dominate 71/29).
# CAVEATS vs r16unf4: differs in BOTH action space and loss weighting (not a
# single-variable A/B); the warm-start ckpt was trained on EE actions (video-side
# adaptation carries over, the action head re-adapts).
#
# Env knobs (with defaults):
#   OUTPUT_DIR            output checkpoint dir
#   WARMSTART_LORA_PATH   prior LoRA ckpt to warm-start delta+MLPs from (post-injection load)
#   UNFREEZE_TOP_K        number of top DiT blocks to full-finetune (default 3; 4 OOMs 96GB)
#   MAX_STEPS / SAVE_STEPS
set -eo pipefail
export HYDRA_FULL_ERROR=1
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-torch}

FRANKA_ORCA_DATA_ROOT=${FRANKA_ORCA_DATA_ROOT:-"./data/franka_orca_lerobot_joints"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_franka_orca_lora_joints"}
WARMSTART_LORA_PATH=${WARMSTART_LORA_PATH:-"./checkpoints/dreamzero_franka_orca_lora_r16/checkpoint-6000"}
UNFREEZE_TOP_K=${UNFREEZE_TOP_K:-3}

if [ -z "${NUM_GPUS}" ]; then NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l); fi
NUM_GPUS=${NUM_GPUS:-1}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}

if [ ! -d "$FRANKA_ORCA_DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $FRANKA_ORCA_DATA_ROOT"; exit 1
fi

echo "==> joints run: DATA=$FRANKA_ORCA_DATA_ROOT  OUTPUT_DIR=$OUTPUT_DIR  WARMSTART=$WARMSTART_LORA_PATH  UNFREEZE_TOP_K=$UNFREEZE_TOP_K  MAX_STEPS=${MAX_STEPS:-14000}"

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
    save_total_limit=${SAVE_TOTAL_LIMIT:-5} \
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
    ++action_head_cfg.config.unfreeze_top_k_blocks=$UNFREEZE_TOP_K \
    '++action_head_cfg.config.action_loss_segment_slices=[[0,7],[7,14],[14,31],[31,48]]' \
    '++action_head_cfg.config.action_loss_segment_weights=[1.0,1.0,1.0,1.0]'
