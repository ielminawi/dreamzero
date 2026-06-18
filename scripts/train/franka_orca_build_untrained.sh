#!/bin/bash
# DreamZero Franka+Orca — build a NO-TRAINING ("untrained") JOINT-SPACE baseline checkpoint.
#
# Identical model build to scripts/train/franka_orca_training_joints.sh EXCEPT:
#   * NO warm-start (no ++warmstart_lora_path): the LoRA + robot MLPs start at their
#     random init, not a prior fine-tune.
#   * UNFREEZE_TOP_K defaults to 0 (no top-K DiT blocks unfrozen) -> at eval, those
#     blocks come from the AgiBot base (i.e. untrained), and the checkpoint stays small.
#   * MAX_STEPS=1, SAVE_STEPS=1 -> save checkpoint-1 immediately. One optimizer step at
#     warmup LR moves params negligibly vs random init, so this is "effectively untrained".
#
# Purpose: a baseline for action-prediction error. The trained policy must beat THIS
# (a random-head policy) to show the training actually learned the obs->action mapping,
# not just the stay-still floor. See eval_utils/action_eval.py.
set -eo pipefail
export HYDRA_FULL_ERROR=1
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-torch}

FRANKA_ORCA_DATA_ROOT=${FRANKA_ORCA_DATA_ROOT:-"./data/franka_orca_lerobot_joints"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_franka_orca_lora_joints_untrained"}
UNFREEZE_TOP_K=${UNFREEZE_TOP_K:-0}

if [ -z "${NUM_GPUS}" ]; then NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l); fi
NUM_GPUS=${NUM_GPUS:-1}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}

if [ ! -d "$FRANKA_ORCA_DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $FRANKA_ORCA_DATA_ROOT"; exit 1
fi

echo "==> UNTRAINED build: DATA=$FRANKA_ORCA_DATA_ROOT  OUTPUT_DIR=$OUTPUT_DIR  (NO warm-start)  UNFREEZE_TOP_K=$UNFREEZE_TOP_K  MAX_STEPS=${MAX_STEPS:-1}"

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
    save_steps=${SAVE_STEPS:-1} \
    training_args.warmup_ratio=0.10 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=2 \
    max_steps=${MAX_STEPS:-1} \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=2 \
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
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    ++action_head_cfg.config.lora_rank=16 \
    ++action_head_cfg.config.lora_alpha=16 \
    ++action_head_cfg.config.unfreeze_top_k_blocks=$UNFREEZE_TOP_K \
    '++action_head_cfg.config.action_loss_segment_slices=[[0,7],[7,14],[14,31],[31,48]]' \
    '++action_head_cfg.config.action_loss_segment_weights=[1.0,1.0,1.0,1.0]'
