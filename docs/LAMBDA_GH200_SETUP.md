# DreamZero + Isaac Sim on Lambda GH200 — Setup Guide

Step-by-step guide to run closed-loop DreamZero inference with Isaac Sim on a Lambda Cloud GH200 instance, with remote Omniverse Streaming.

## Architecture

```
┌──────────────────────── Lambda GH200 ────────────────────────┐
│                                                               │
│  ┌─────────────────────┐      ┌───────────────────────────┐  │
│  │ dreamzero-inference  │      │       isaac-sim            │  │
│  │                      │      │                            │  │
│  │ WebSocket :8000      │◄─────│ Eval loop client           │  │
│  │ 14B model (bf16)     │      │                            │  │
│  │ ~55-65GB VRAM        │      │ Omniverse Streaming :8211  │──┼──► Your laptop
│  └─────────────────────┘      └───────────────────────────┘  │
│                                                               │
│  GPU: 1x H100 96GB HBM3  │  CPU: 72 Arm Neoverse cores      │
│  RAM: 480GB unified       │  ~$3.49/hr                       │
└───────────────────────────────────────────────────────────────┘
```

---

## Step 1: Launch the GH200 Instance

1. Go to [Lambda Cloud](https://cloud.lambdalabs.com/instances)
2. Launch a **1x GH200** instance (Grace Hopper, 96GB HBM3)
3. Select your SSH key
4. Note the public IP address

```bash
# SSH into the instance
ssh ubuntu@<LAMBDA_IP>
```

---

## Step 2: Clone the Repo

```bash
cd ~
git clone --recurse-submodules git@github.com:<your-org>/dreamzero.git
cd dreamzero

# If you already cloned without --recurse-submodules:
git submodule update --init --recursive
```

Verify the Orca hand submodule is present:
```bash
ls sim_envs/assets/orcahand_description/models/urdf/
# Should show: orcahand.urdf.xacro  orcahand_right.urdf  README.md
```

---

## Step 3: Download Model Checkpoints

You need four sets of weights. Create the checkpoints directory and download:

```bash
mkdir -p checkpoints

# 1. Wan2.1-I2V-14B-480P backbone (~28GB)
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
    --local-dir checkpoints/Wan2.1-I2V-14B-480P

# 2. umt5-xxl tokenizer (~5GB)
huggingface-cli download google/umt5-xxl \
    --local-dir checkpoints/umt5-xxl

# 3. DreamZero-AgiBot base model (~45GB)
git lfs install
git clone https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot \
    checkpoints/DreamZero-AgiBot

# 4. Your trained Franka Orca LoRA checkpoint
#    Copy from wherever you trained it:
#    scp -r <training-machine>:path/to/checkpoint checkpoints/dreamzero_franka_orca_lora
#
#    Or if it's on HuggingFace:
#    huggingface-cli download <your-org>/dreamzero-franka-orca \
#        --local-dir checkpoints/dreamzero_franka_orca_lora
```

Verify the checkpoint structure:
```bash
ls checkpoints/
# Should show: DreamZero-AgiBot  dreamzero_franka_orca_lora  umt5-xxl  Wan2.1-I2V-14B-480P

ls checkpoints/dreamzero_franka_orca_lora/
# Should show: experiment_cfg/  config.json  pytorch_model.bin  adapter_config.json (or similar)
```

**Disk space needed**: ~100GB total for all checkpoints.

---

## Step 4: Install Docker + NVIDIA Container Toolkit

Lambda instances usually have Docker pre-installed. Verify:

```bash
docker --version
nvidia-smi
docker compose version
```

If Docker is not installed:
```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Step 5: Configure the Environment

Edit the Docker .env file to match your setup:

```bash
cd ~/dreamzero
nano docker/.env
```

Key variables to verify:
```bash
# Path to your trained LoRA checkpoint (inside the container)
MODEL_PATH=/checkpoints/dreamzero_franka_orca_lora

# Host paths (relative to docker/ directory)
HOST_CHECKPOINT_DIR=../checkpoints
HOST_OUTPUT_DIR=../output

# For GH200, use Transformer Engine attention backend
ATTENTION_BACKEND=TE

# Task instruction for evaluation
EVAL_INSTRUCTION="pick up the cube"
```

---

## Step 6: Build the Docker Images

```bash
cd ~/dreamzero/docker

# Build both containers (this takes 15-30 minutes the first time)
docker compose build

# Or build individually to debug issues:
# docker compose build dreamzero-inference
# docker compose build isaac-sim
```

**Note for GH200 (ARM64)**: The launch script auto-detects the architecture. If you see base image issues, manually set:
```bash
export BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3-igpu
docker compose build --build-arg BASE_IMAGE=$BASE_IMAGE dreamzero-inference
```

**Isaac Sim container**: The `nvcr.io/nvidia/isaac-sim:4.5.0` image is large (~50GB). The first pull will take time. If the ARM64 variant is not available, use the latest available version:
```bash
# Check available tags:
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim/tags
docker compose build --build-arg ISAAC_SIM_VERSION=4.5.0 isaac-sim
```

---

## Step 7: Start the Inference Server

Start the inference server first (it takes a few minutes to load the model):

```bash
cd ~/dreamzero/docker

# Start just the inference server
docker compose up -d dreamzero-inference

# Watch the logs — wait until you see "Server started" or the healthcheck passes
docker compose logs -f dreamzero-inference
```

You should see:
```
Rank 0/1 (PID: ...) setting device to 0
Loading model from /checkpoints/dreamzero_franka_orca_lora...
...
Server started on 0.0.0.0:8000
```

Verify the healthcheck:
```bash
curl http://localhost:8000/healthz
# Should return: OK
```

**Expected startup time**: 3-5 minutes (model loading + torch compilation warmup).

---

## Step 8: Set Up SSH Tunnel for Omniverse Streaming

On your **local machine** (not the Lambda instance), open a terminal:

```bash
ssh -L 8211:localhost:8211 -L 49100:localhost:49100 ubuntu@<LAMBDA_IP>
```

Keep this terminal open. The tunnel forwards Isaac Sim's streaming ports to your local machine.

---

## Step 9: Start Isaac Sim

Back on the Lambda instance:

```bash
cd ~/dreamzero/docker

# Start Isaac Sim (waits for inference server healthcheck to pass first)
docker compose up -d isaac-sim

# Watch the logs
docker compose logs -f isaac-sim
```

You should see:
```
[Info] Loading FrankaOrcaBimanual-v0 environment...
[Info] Omniverse Streaming enabled on port 8211
[Info] Connecting to inference server at dreamzero-inference:8000...
Episode 1/10: ...
```

---

## Step 10: Connect Omniverse Streaming Client

1. Download and install [Omniverse Streaming Client](https://docs.omniverse.nvidia.com/streaming-client/latest/user-guide/installation.html) on your local machine
2. Open the Streaming Client
3. Connect to: `localhost:8211` (the SSH tunnel forwards this to the Lambda instance)
4. You should see the Isaac Sim viewport with the dual Franka arms + Orca hands

---

## Step 11: Monitor and Collect Results

```bash
# View live logs from both services
docker compose logs -f

# Check GPU utilization
nvidia-smi -l 1

# Evaluation videos are saved to:
ls ~/dreamzero/output/sim/
```

---

## Useful Commands

```bash
# Stop everything
cd ~/dreamzero/docker
docker compose down

# Restart just Isaac Sim (e.g., to change the instruction)
docker compose restart isaac-sim

# Change eval instruction and restart
EVAL_INSTRUCTION="put the banana in the bowl" docker compose up -d isaac-sim

# Start with a full rebuild
docker compose up -d --build

# Shell into the inference container for debugging
docker compose exec dreamzero-inference bash

# Shell into Isaac Sim container
docker compose exec isaac-sim bash

# View resource usage
docker stats
```

---

## Troubleshooting

### "CUDA out of memory"
The 14B model + Isaac Sim rendering is ~70GB on a 96GB GPU. If OOM:
- Ensure no other processes use the GPU: `nvidia-smi`
- Try reducing `num_inference_timesteps` in the model config (4 → 2)
- Enable tiling in the action head config for memory-efficient VAE decoding

### Isaac Sim won't start
- Check EULA acceptance: `ACCEPT_EULA=Y` must be in the environment
- Check NVIDIA driver: `nvidia-smi` should work inside the container
- For ARM64 compatibility issues, check the [Isaac Sim container catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim/tags)

### Omniverse Streaming Client won't connect
- Verify the SSH tunnel is active (Step 8)
- Check ports are exposed: `docker compose port isaac-sim 8211`
- Check Isaac Sim logs for streaming errors: `docker compose logs isaac-sim | grep -i stream`
- Try connecting to `<LAMBDA_IP>:8211` directly (without tunnel) if firewall allows

### Inference is slow (>5s per step)
- Verify `ATTENTION_BACKEND=TE` is set (Transformer Engine, critical for GH200)
- Verify `ENABLE_DIT_CACHE=true`
- First few inference calls are slow due to torch compilation warmup — this is expected
- Steady-state on GH200 should be ~0.6s per step

### Mesh/URDF loading errors in Isaac Sim
- The URDF mesh paths are relative. Verify the Orca hand submodule was cloned:
  ```bash
  docker compose exec isaac-sim ls /app/sim_envs/assets/orcahand_description/assets/urdf/visual/
  ```
- If missing, rebuild: `docker compose build isaac-sim`

### Hand mount looks wrong in simulation
The fixed joint between `panda_link8` and the Orca hand root may need adjustment.
Edit `sim_envs/assets/generate_combined_urdf.py`, change `mount_xyz` and `mount_rpy`,
then regenerate:
```bash
python3 sim_envs/assets/generate_combined_urdf.py
docker compose build isaac-sim
docker compose up -d isaac-sim
```

---

## File Reference

| File | Purpose |
|------|---------|
| `docker/docker-compose.yml` | Service orchestration |
| `docker/.env` | Configuration variables |
| `docker/dreamzero-inference/Dockerfile` | Inference server container |
| `docker/isaac-sim/Dockerfile` | Isaac Sim + IsaacLab container |
| `docker/scripts/launch.sh` | One-command launcher |
| `sim_envs/assets/franka_orca_{left,right}.urdf` | Combined 24-DOF arm+hand URDFs |
| `sim_envs/assets/generate_combined_urdf.py` | URDF generator script |
| `sim_envs/franka_orca_bimanual_cfg.py` | IsaacLab scene config |
| `sim_envs/franka_orca_bimanual_env.py` | IsaacLab environment |
| `eval_utils/run_sim_eval_bimanual.py` | Closed-loop eval client |
| `socket_test_optimized_AR.py` | Inference server (supports --embodiment franka_orca_bimanual) |
