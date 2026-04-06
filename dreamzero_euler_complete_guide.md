### Euler GPU Hardware

| GPU | VRAM | Fits 14B? | Notes |
|-----|------|-----------|-------|
| A100-80GB | 80 GB | ✅ Yes | Best option |
| RTX PRO 6000 | 96 GB | ✅ Yes | Newest, may have queue |
| A100-40GB | 40 GB | ⚠️ Maybe | Tight with LoRA |
| RTX 4090 | 24 GB | ❌ No | Too small |
| RTX 3090 | 24 GB | ❌ No | Too small |
| RTX 2080 Ti | 11 GB | ❌ No | Way too small |

---

## Step 1: SSH into Euler

```bash
ssh rjiang@euler.ethz.ch
```

If off-campus, connect to ETH VPN first.

---

## Step 2: Set Up Environment (One-Time)

This is already done. For reference:

```bash
# Create workspace
mkdir -p $SCRATCH/dreamzero/{checkpoints,data,output,logs}

# Load modules
module load stack/2024-06 gcc/12.2.0
module load eth_proxy

# Create venv
python -m venv $SCRATCH/dreamzero/venv
source $SCRATCH/dreamzero/venv/bin/activate

# Clone and install DreamZero or scp our code 
cd $SCRATCH/dreamzero
git clone https://github.com/dreamzero0/dreamzero.git // SCP
cd dreamzero
pip install -e . 

# Install build deps
pip install packaging setuptools wheel ninja psutil numpy einops deepspeed peft wandb

# get flash-attn (no need to compile ourselves)
cd /tmp
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install --no-deps /tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

### Status: ✅ Environment complete

- [x] PyTorch 2.11.0+cu129
- [x] flash-attn 2.8.3
- [x] DreamZero installed
- [x] DeepSpeed installed

---

## Step 3: Download Checkpoints (One-Time)

Start an interactive session for downloads (CPU-only is fine):

```bash
srun --ntasks=1 --cpus-per-task=4 --mem-per-cpu=8G --time=4:00:00 --account=ls_polle --pty bash
module load eth_proxy
source $SCRATCH/dreamzero/venv/bin/activate

export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
pip install "huggingface_hub[cli]"
```

### 3a. DreamZero-AgiBot checkpoint (~45 GB)

```bash
hf download GEAR-Dreams/DreamZero-AgiBot \
  --repo-type model \
  --local-dir $SCRATCH/dreamzero/checkpoints/DreamZero-AgiBot
```

### 3b. Wan2.1 backbone (~28 GB)

```bash
hf download Wan-AI/Wan2.1-I2V-14B-480P \
  --local-dir $SCRATCH/dreamzero/checkpoints/Wan2.1-I2V-14B-480P
```

### 3c. umt5-xxl tokenizer

```bash
hf download google/umt5-xxl \
  --local-dir $SCRATCH/dreamzero/checkpoints/umt5-xxl
```

### Verify downloads

```bash
ls $SCRATCH/dreamzero/checkpoints/DreamZero-AgiBot/
ls $SCRATCH/dreamzero/checkpoints/Wan2.1-I2V-14B-480P/
ls $SCRATCH/dreamzero/checkpoints/umt5-xxl/
```

Exit the interactive session when done:

---

## Step 4: Verify Your Dataset

```bash
ls /cluster/scratch/rjiang/dreamzero/data/franka_orca_lerobot/
```

This should contain your Franka+Orca bimanual data in LeRobot format with GEAR metadata. If it's not there yet, you need to run:

python scripts/data/convert_h5_to_lerobot.py \
    --input-dir /cluster/work/cvg/data/Egoverse/raw_timesynced_h5 \
    --output-dir /cluster/scratch/rjiang/dreamzero/data/franka_orca_lerobot \
    --fps 50 \
    --task "bimanual manipulation" \
    --num-workers 1

python scripts/data/convert_lerobot_to_gear.py \
    --dataset-path /cluster/scratch/rjiang/dreamzero/data/franka_orca_lerobot \
    --embodiment-tag franka_orca_bimanual \
    --state-keys '{"left_arm_joint_pos": [0, 7], "right_arm_joint_pos": [7, 14], "left_hand_joint_pos": [14, 31], "right_hand_joint_pos": [31, 48]}' \
    --action-keys '{"left_arm_joint_pos": [0, 7], "right_arm_joint_pos": [7, 14], "left_hand_joint_pos": [14, 31], "right_hand_joint_pos": [31, 48]}' \
    --relative-action-keys left_arm_joint_pos right_arm_joint_pos left_hand_joint_pos right_hand_joint_pos \
    --action-horizon 48

## Step 6: Create the SLURM Batch Script

Run this on Euler to run the slurm task:

/cluster/scratch/rjiang/dreamzero$ sbatch train_franca_orca.sh 

## Step 7: Submit the Job

```bash
cd $SCRATCH/dreamzero
sbatch train_franka_orca.sh
```

You'll see: `Submitted batch job <JOBID>`

---

## Step 8: Monitor the Job

### Check job status

```bash
squeue --user=$USER
```

States:
- `PD` = pending (waiting for resources)
- `R` = running
- `CG` = completing

### Lambda

```
  # Create workspace
  mkdir -p ~/Dreamzero/{checkpoints,data,output,logs}

  # Install miniconda if not already present
  # (Lambda instances often have it pre-installed — check with `conda --version`)
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
  bash Miniconda3-latest-Linux-aarch64.sh -b -p $HOME/miniconda3
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

  # Create environment
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  conda create -n dreamzero python=3.11 -y
  conda activate dreamzero

  # Deps
  cd ~/Dreamzero
  git clone <YOUR_REPO_URL> repo   # or scp your local copy
  cd repo

  pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
  MAX_JOBS=8 pip install --no-build-isolation flash-attn
  pip install --no-build-isolation transformer_engine[pytorch]
  pip install packaging setuptools wheel ninja psutil

  # Models
  pip install "huggingface_hub[cli]"
  hf download GEAR-Dreams/DreamZero-AgiBot \
      --local-dir ~/Dreamzero/repo/checkpoints/DreamZero-AgiBot
  
  hf download Wan-AI/Wan2.1-I2V-14B-480P \
      --local-dir ~/dreamzero/repo/checkpoints/Wan2.1-I2V-14B-480P

  hf download google/umt5-xxl \
      --local-dir ~/dreamzero/repo/checkpoints/umt5-xxl

  # Data

  rsync -av --progress -e "ssh -c aes128-gcm@openssh.com -o Compression=no" \
    20250826_111157.h5 ubuntu@192.222.56.227:~/Dreamzero/data

  cd ~/Dreamzero/repo
  conda activate dreamzero

  # Step 5a: HDF5 → LeRobot format
  python scripts/data/convert_h5_to_lerobot.py \
      --input-dir ../data/raw_h5 \
      --output-dir ../data/franka_orca_lerobot \
      --fps 50 \
      --task "bimanual manipulation" \
      --target-resolution 640x480

  # Step 5b: Generate GEAR metadata
  python3 scripts/data/convert_lerobot_to_gear.py \
      --dataset-path ~/Dreamzero/data/franka_orca_lerobot \
      --embodiment-tag franka_orca_bimanual \
      --state-keys '{"left_arm_joint_pos": [0, 7], "right_arm_joint_pos": [7, 14], "left_hand_joint_pos": [14, 31],
  "right_hand_joint_pos": [31, 48]}' \
      --action-keys '{"left_arm_joint_pos": [0, 7], "right_arm_joint_pos": [7, 14], "left_hand_joint_pos": [14, 31],
  "right_hand_joint_pos": [31, 48]}' \
      --relative-action-keys left_arm_joint_pos right_arm_joint_pos left_hand_joint_pos right_hand_joint_pos \
      --task-key annotation.task --action-horizon 48 --force

  Verify the conversion:
  ls ~/Dreamzero/data/franka_orca_lerobot/meta/
  # Should contain: modality.json, embodiment.json, stats.json,
  # relative_stats_dreamzero.json, tasks.jsonl, episodes.jsonl, info.json

  # Training
  cd ~/Dreamzero/repo

  # Set paths
  export FRANKA_ORCA_DATA_ROOT=~/Dreamzero/data/franka_orca_lerobot
  export OUTPUT_DIR=~/Dreamzero/output/dreamzero_franka_orca_lora
  export NUM_GPUS=1

  # Set NCCL/performance env vars
  export PYTHONUNBUFFERED=1
  export NCCL_DEBUG=WARN
  export OMP_NUM_THREADS=16

  # Launch training
  bash scripts/train/franka_orca_training.sh
```