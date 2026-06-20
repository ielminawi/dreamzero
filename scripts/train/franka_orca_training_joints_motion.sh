#!/bin/bash
# DreamZero Franka+Orca — JOINT-SPACE, VARIANCE-NORMALIZED loss + WITHIN-CHUNK MOTION loss.
#
# Warm-starts the run's OWN joints_varnorm/checkpoint-5000 (KEEPS the trained joint head; does
# NOT skip the action-head MLPs) and adds an explicit first-difference (along horizon) motion
# loss (action_loss_motion_weight) on top of the existing variance-normalized action loss. The
# anchor offset is constant across the horizon, so diff_T cancels it and the motion term's
# gradient is purely within-chunk motion shape — the diagnosed-unlearned quantity. Goal: lift
# within-chunk motion-shape correlation (open-loop dir_cos) off ~0.
#
# Env knobs: OUTPUT_DIR, WARMSTART_LORA_PATH, UNFREEZE_TOP_K, MAX_STEPS/SAVE_STEPS/SAVE_TOTAL_LIMIT,
#            MOTION_W (motion-loss weight, default 1.0).
set -eo pipefail
export HYDRA_FULL_ERROR=1
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-torch}

FRANKA_ORCA_DATA_ROOT=${FRANKA_ORCA_DATA_ROOT:-"./data/franka_orca_lerobot_joints_clean"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_franka_orca_lora_joints_motion"}
WARMSTART_LORA_PATH=${WARMSTART_LORA_PATH:-"./checkpoints/dreamzero_franka_orca_lora_joints_varnorm/checkpoint-5000"}
UNFREEZE_TOP_K=${UNFREEZE_TOP_K:-3}
MOTION_W=${MOTION_W:-1.0}

if [ -z "${NUM_GPUS}" ]; then NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l); fi
NUM_GPUS=${NUM_GPUS:-1}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}

if [ ! -d "$FRANKA_ORCA_DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $FRANKA_ORCA_DATA_ROOT"; exit 1
fi

# Per-dim normalized-target variance (length 48), kept identical to the varnorm run (the motion
# term is complementary; DIM_VAR recompute is deliberately held constant for clean attribution).
DIM_VAR='[0.0105,0.01291,0.00387,0.01547,0.01091,0.01233,0.00254,0.02585,0.01791,0.02651,0.05795,0.04219,0.05233,0.03839,0.08411,0.10285,0.08838,0.09723,0.10106,0.11494,0.07817,0.10771,0.14395,0.08068,0.08038,0.17142,0.06752,0.08373,0.12656,0.07429,0.08189,0.07992,0.04505,0.08879,0.08888,0.12674,0.08231,0.04963,0.05409,0.11839,0.04505,0.05806,0.10695,0.05264,0.06675,0.12811,0.06797,0.07031]'

echo "==> joints MOTION run: DATA=$FRANKA_ORCA_DATA_ROOT  OUTPUT_DIR=$OUTPUT_DIR  WARMSTART=$WARMSTART_LORA_PATH  UNFREEZE_TOP_K=$UNFREEZE_TOP_K  MOTION_W=$MOTION_W  MAX_STEPS=${MAX_STEPS:-2000}"

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
    save_steps=${SAVE_STEPS:-500} \
    training_args.warmup_ratio=0.10 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=2 \
    max_steps=${MAX_STEPS:-2000} \
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
    '++action_head_cfg.config.action_loss_segment_weights=[1.0,1.0,1.0,1.0]' \
    ++action_head_cfg.config.action_loss_variance_normalize=true \
    ++action_head_cfg.config.action_loss_variance_floor=0.02 \
    ++action_head_cfg.config.action_loss_motion_weight=$MOTION_W \
    "++action_head_cfg.config.action_loss_dim_variances=$DIM_VAR"
