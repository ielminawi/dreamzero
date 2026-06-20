#!/bin/bash
# DreamZero Franka+Orca — JOINT-SPACE, MOTION fix v2: push the STUCK HANDS.
# vs motion v1: (1) CORRECTED action_loss_dim_variances = TRUE within-chunk motion variance
# (v1/varnorm used the offset-dominated across-sample span, which down-weighted the high-motion
# hand dims); (2) lower variance floor 0.02 -> 0.002; (3) up-weight the hand segments 2.5x via
# segment_weights [1,1,2.5,2.5] (hands "ignore input"); (4) stronger motion loss MOTION_W=2.0.
# Warm-start the run's own joints_varnorm/checkpoint-5000 (keep the trained joint head).
set -eo pipefail
export HYDRA_FULL_ERROR=1
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-torch}

FRANKA_ORCA_DATA_ROOT=${FRANKA_ORCA_DATA_ROOT:-"./data/franka_orca_lerobot_joints_clean"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_franka_orca_lora_joints_motion2"}
WARMSTART_LORA_PATH=${WARMSTART_LORA_PATH:-"./checkpoints/dreamzero_franka_orca_lora_joints_varnorm/checkpoint-5000"}
UNFREEZE_TOP_K=${UNFREEZE_TOP_K:-3}
MOTION_W=${MOTION_W:-2.0}

if [ -z "${NUM_GPUS}" ]; then NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l); fi
NUM_GPUS=${NUM_GPUS:-1}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}

if [ ! -d "$FRANKA_ORCA_DATA_ROOT" ]; then echo "ERROR: Dataset not found at $FRANKA_ORCA_DATA_ROOT"; exit 1; fi

# CORRECTED per-dim variance = mean within-chunk variance of the relative target (var over the 24
# chunk steps, averaged over chunks) -- from output/corrected_dim_var.json (1244 chunks).
DIM_VAR='[0.00012,0.00034,0.00010,0.00060,0.00047,0.00077,0.00035,0.00036,0.00080,0.00061,0.00141,0.00071,0.00087,0.00072,0.00038,0.00217,0.00286,0.00208,0.00204,0.00117,0.00249,0.00123,0.00057,0.00171,0.00432,0.00028,0.00229,0.00479,0.00067,0.00302,0.00374,0.00056,0.00065,0.00097,0.00088,0.00011,0.00078,0.00440,0.00159,0.00033,0.00774,0.00591,0.00038,0.00559,0.00332,0.00032,0.00349,0.00244]'

echo "==> joints MOTION2 (hands push): OUT=$OUTPUT_DIR WARMSTART=$WARMSTART_LORA_PATH MOTION_W=$MOTION_W seg_w=[1,1,2.5,2.5] floor=0.002 MAX_STEPS=${MAX_STEPS:-2000}"

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
    '++action_head_cfg.config.action_loss_segment_weights=[1.0,1.0,2.5,2.5]' \
    ++action_head_cfg.config.action_loss_variance_normalize=true \
    ++action_head_cfg.config.action_loss_variance_floor=0.002 \
    ++action_head_cfg.config.action_loss_motion_weight=$MOTION_W \
    "++action_head_cfg.config.action_loss_dim_variances=$DIM_VAR"
