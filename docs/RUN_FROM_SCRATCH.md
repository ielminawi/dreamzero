# DreamZero — Franka + Orca Bimanual: Run From Scratch (H100 / Lambda)

End-to-end runbook to stand up **DreamZero inference** and the **closed-loop Isaac Sim
evaluation** for the `franka_orca_bimanual` embodiment on a fresh single-GPU box
(validated on a Lambda **H100 PCIe 80 GB**, Ubuntu 22.04, driver `580.105.08`).

This captures the exact setup that was validated on 2026-06-01, including the
non-obvious fixes (dead cluster paths, private submodule, Vulkan libs, Docker GPU
flags). Commands assume the repo lives at `~/dreamzero`.

> **TL;DR of what runs at the end**
> - A **policy server** (`socket_test_optimized_AR.py`) loads the 16.5 B Wan2.1 video
>   world-model + LoRA action head and serves actions over WebSocket on `:5000`.
> - The **Isaac Sim** container drives the two-Franka + Orca-hand scene, querying the
>   server every 24 steps, and writes a rollout video.

---

## 0. Hardware / prerequisites

| Requirement | Validated value | Notes |
|---|---|---|
| GPU | 1× NVIDIA H100 80 GB | Needs ~45 GB (server, FA2) + ~10–15 GB (Isaac). 80 GB fits both. |
| Driver | `580.105.08` (Lambda) | A **`-server`/datacenter** driver — lacks Vulkan/GL by default (see §6). |
| OS | Ubuntu 22.04 | |
| Python | **3.11** | pyproject requires `~=3.11,<3.13`. System 3.10 will not work. |
| Disk | ~120 GB free | Wan2.1 ~66 GB + AgiBot ~45 GB + Isaac images ~50 GB. |
| Docker | 24+ with compose v2 | Only needed for Isaac Sim (and optional container inference). |

You will need:
- A **trained LoRA checkpoint** for `franka_orca_bimanual` (here:
  `~/checkpoints/dreamzero_franka_orca_lora/checkpoint-20000/`). This is your own
  fine-tune; it is **not** on HuggingFace.
- Network access to HuggingFace (Wan2.1, DreamZero-AgiBot) and GitHub (public repos).

---

## 1. Clone the repo + submodule (HTTPS fix)

```bash
cd ~
git clone <your dreamzero remote> dreamzero    # or copy the repo onto the box
cd ~/dreamzero
```

The Orca-hand meshes are a **git submodule** that the sim URDFs need (62 collision
meshes). `.gitmodules` pins an **SSH** URL to a private-looking repo, but the repo is
**public over HTTPS** — override the URL so no deploy key is needed:

```bash
git config submodule.sim_envs/assets/orcahand_description.url \
  https://github.com/srl-ethz/orcahand_description_team3.git
git submodule update --init sim_envs/assets/orcahand_description
# verify: should print 237 files and 0 missing
find sim_envs/assets/orcahand_description -type f | wc -l
```

> If you skip this, Isaac Sim fails to import the hand URDFs (missing `.stl` files).

---

## 2. Python env + package install

```bash
cd ~/dreamzero
/usr/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
```

This pulls the pinned stack (torch 2.8.0+cu129, transformers 4.51.3, diffusers 0.30.2,
peft, deepspeed, gear, tianshou, openpi-client, websockets, etc.).

**Flash-Attention** (the server uses it as the attention backend on H100):

```bash
MAX_JOBS=8 pip install --no-build-isolation flash-attn   # validated: flash_attn 2.8.3
```

> **Do NOT install `transformer_engine` on H100.** The server hardcodes
> `ATTENTION_BACKEND=TE`, but the code **auto-falls back to FA2** when TE is absent
> (`groot/.../wan2_1_attention.py`). TE is only for GB200/GH200.

---

## 3. Download base model weights (~111 GB)

The LoRA checkpoint is tiny (~207 MB) and **useless without the base weights**. Two
artifacts are required and resolved by the checkpoint's `experiment_cfg/conf.yaml`:

```bash
source ~/dreamzero/.venv/bin/activate
mkdir -p ~/dreamzero/checkpoints

# 1) Wan2.1 backbone — provides DiT init, VAE, T5 text encoder, CLIP image encoder (~66 GB)
hf download Wan-AI/Wan2.1-I2V-14B-480P --repo-type model \
  --local-dir ~/dreamzero/checkpoints/Wan2.1-I2V-14B-480P

# 2) DreamZero-AgiBot — the fine-tuned DiT base the LoRA is rebuilt on top of (~45 GB)
hf download GEAR-Dreams/DreamZero-AgiBot --repo-type model \
  --local-dir ~/dreamzero/checkpoints/DreamZero-AgiBot

# 3) umt5-xxl tokenizer (small). Either download google/umt5-xxl, or reuse the copy
#    bundled inside the Wan2.1 dir:
ln -s ~/dreamzero/checkpoints/Wan2.1-I2V-14B-480P/google/umt5-xxl \
      ~/dreamzero/checkpoints/umt5-xxl   # if present; else: hf download google/umt5-xxl --local-dir .../umt5-xxl
```

> At load time the log shows: `Rebuilding DiT base from training base model:
> .../DreamZero-AgiBot` while VAE/text/image encoders come from the Wan2.1 dir.

Place your **LoRA checkpoint** at `~/checkpoints/dreamzero_franka_orca_lora/checkpoint-20000/`
(must contain `config.json`, `model.safetensors`, `experiment_cfg/{conf.yaml,metadata.json}`).

---

## 4. Fix the checkpoint's dead cluster paths ⚠️

The released `conf.yaml` hardcodes a training-cluster path
(`/cluster/scratch/rjiang/...`) for every weight + tokenizer. **The Python does NOT
honor the `WAN_CKPT_DIR`/`*_PRETRAINED_PATH` env vars** (only `TOKENIZER_DIR`), so you
must rewrite the paths in the file:

```bash
CONF=~/checkpoints/dreamzero_franka_orca_lora/checkpoint-20000/experiment_cfg/conf.yaml
cp "$CONF" "$CONF.bak"
sed -i 's#/cluster/scratch/rjiang/dreamzero/checkpoints/Wan2.1-I2V-14B-480P#'"$HOME"'/dreamzero/checkpoints/Wan2.1-I2V-14B-480P#g' "$CONF"
sed -i 's#/cluster/scratch/rjiang/dreamzero/checkpoints/umt5-xxl#'"$HOME"'/dreamzero/checkpoints/umt5-xxl#g' "$CONF"
# verify only training-only output_dir/data_root remain (harmless for inference):
grep -n "/cluster/scratch" "$CONF"
```

> Also set `TOKENIZER_DIR=$HOME/dreamzero/checkpoints/umt5-xxl` when launching (belt-and-suspenders).

---

## 5. Run the inference server (bare-metal, lean FA2)

```bash
cd ~/dreamzero
CUDA_VISIBLE_DEVICES=0 \
TOKENIZER_DIR=$HOME/dreamzero/checkpoints/umt5-xxl \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  socket_test_optimized_AR.py \
    --port 5000 --enable-dit-cache --embodiment franka_orca_bimanual \
    --model-path $HOME/checkpoints/dreamzero_franka_orca_lora/checkpoint-20000 \
  > logs/server_fa2.log 2>&1 &

# wait until healthy (first load: a few minutes), then:
curl -s http://localhost:5000/healthz      # -> OK
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # ~44.5 GB (FA2)
```

Smoke-test the server with the **48-dim** client (the stock `test_client_AR.py` is
DROID/8-dim and will assert-fail):

```bash
.venv/bin/python test_client_franka_orca.py --host localhost --port 5000 --num-chunks 3
```

### Eager vs FA2 (important tradeoff)
`run_franka_orca_server.sh` launches with `ENABLE_TENSORRT=True` → **eager** mode
(torch SDPA attention). It is **stable across session resets** but uses **~73 GB**,
leaving no room for Isaac on an 80 GB GPU. Dropping `ENABLE_TENSORRT` (as above) gives
**FA2, ~44.5 GB** — required to fit Isaac alongside — but enables `torch.compile`,
which the script comments warn can crash on a *session reset*. For a single-episode (or
few-episode) eval this is fine; it was validated end-to-end. For many-episode batches on
a ≥96 GB GPU, prefer the eager server.

---

## 6. Isaac Sim — Vulkan graphics libraries ⚠️ (the big gotcha)

Lambda's **datacenter (`-server`) driver ships compute/CUDA only** — no Vulkan/GL.
Isaac Sim's RTX renderer needs Vulkan **even headless**, or you get
`ERROR_INCOMPATIBLE_DRIVER / Vulkan 1.1 is not supported`. Install the **version-matched**
graphics userspace (userspace-only — **no driver reload / reboot**):

```bash
# match the EXACT running driver version (here 580.105.08); pinning avoids CUDA breakage
sudo apt-get install -y libnvidia-gl-580-server=580.105.08-0lambda0.22.04.1
sudo ldconfig          # CRITICAL: refresh ldcache so nvidia-container-toolkit can find the libs
# verify:
ls /usr/share/vulkan/icd.d/nvidia_icd.json
ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.580.105.08
sudo nvidia-container-cli list | grep -E "GLX_nvidia|glvkspirv|EGL_nvidia"   # must list all 3
```

> Find the exact version with `cat /proc/driver/nvidia/version`. If your driver is the
> non-server variant, use `libnvidia-gl-<NNN>` instead. The repo's
> `scripts/setup_lambda_gpu.sh` does the same idea but uses the *non-server* package and
> an unpinned version — prefer the pinned `-server` command above for this box.

---

## 7. Docker: NVIDIA runtime + build Isaac image

```bash
sudo usermod -aG docker $USER          # then re-login, or prefix docker with sudo
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo docker info | grep -i runtimes    # must include "nvidia"

cd ~/dreamzero/docker
sudo docker compose build isaac-sim    # ~20 min; base nvcr.io/nvidia/isaac-sim:4.5.0 (no NGC login needed)
```

---

## 8. Run the closed-loop evaluation

With the **FA2 server from §5 already running** on `:5000`, launch the Isaac container
pointed at the host server. **Use `--runtime nvidia` with explicit
`NVIDIA_DRIVER_CAPABILITIES` — do NOT use `--gpus all`** (it strips the `graphics`
capability and Vulkan breaks again):

```bash
cd ~/dreamzero
sudo docker run --rm --runtime nvidia --network host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility,display \
  -e OMNI_KIT_ALLOW_ROOT=1 -e ACCEPT_EULA=Y -e LIVESTREAM_ENABLED=false \
  -v $HOME/dreamzero/output/sim:/output \
  -v $HOME/dreamzero/sim_envs:/app/sim_envs:ro \
  -v $HOME/dreamzero/eval_utils:/app/eval_utils:ro \
  -v $HOME/dreamzero/configs:/app/configs:ro \
  docker-isaac-sim:latest \
  eval_utils/run_sim_eval_bimanual.py --headless --enable_cameras \
    --host localhost --port 5000 \
    --instruction "pick up the cube" --episodes 1 --open-loop-horizon 24
```

Expected: Isaac loads in ~12 s, `Action Manager ... shape: 48`, `Connecting to inference
server at localhost:5000...`, then `Episode 1/1: ... 1500/1500` and
`Episode 1 saved to /output/<date>/<time>/episode_0.mp4`. A 1500-step episode (~62
inference chunks) takes ~10 min. GPU peaks ~60 GB.

Output video: `~/dreamzero/output/sim/<date>/<time>/episode_0.mp4` (two camera views,
640×240, 30 s).

### Optional: remote 3D viewer
Add `--livestream 2` to the eval command and publish ports `8211`/`49100`
(`--network host` already does), then from your laptop:
`ssh -L 8211:localhost:8211 -L 49100:localhost:49100 ubuntu@<box>` and connect the
Omniverse Streaming Client to `localhost:8211`.

---

## 9. Troubleshooting cheat-sheet

| Symptom | Cause | Fix |
|---|---|---|
| `No such file .../cluster/scratch/...` on server load | dead training paths in `conf.yaml` | §4 sed rewrite |
| `Transformer Engine not available` (warning only) | TE not installed | expected on H100 — FA2 fallback, ignore |
| Server uses ~73 GB, Isaac OOMs | eager server (`ENABLE_TENSORRT=True`) | run FA2 server (§5, drop `ENABLE_TENSORRT`) |
| Missing `.stl` / URDF import fails | submodule not initialized | §1 HTTPS submodule init |
| `ERROR_INCOMPATIBLE_DRIVER`, `Vulkan 1.1 not supported` | `-server` driver lacks Vulkan/GL | §6 install `libnvidia-gl-…-server` + `ldconfig` |
| `libGLX_nvidia` missing **inside** container | `--gpus all` strips graphics capability | §8: use `--runtime nvidia` + `NVIDIA_DRIVER_CAPABILITIES=graphics,...` |
| `unknown or invalid runtime name: nvidia` | runtime not registered | §7 `nvidia-ctk runtime configure` + restart docker |
| `test_client_AR.py` asserts `(N,8)` | that client is DROID-only | use `test_client_franka_orca.py` (48-dim) |
| First inference takes minutes | torch.compile / autoregressive warm-up | normal; steady-state ~2–3 it/s |

---

## 10. What the parallel setup session had already done

The box was partly provisioned by an earlier session. For reference, it had:
- Created `.venv` (`python3.11 -m venv`), run `pip install -e .`, installed flash-attn 2.8.3.
- Downloaded `Wan2.1-I2V-14B-480P`, `DreamZero-AgiBot`, `umt5-xxl` into `./checkpoints/`.
- Added `run_franka_orca_server.sh` (eager launcher) and `test_client_franka_orca.py`
  (48-dim smoke client).
- Added `scripts/setup_lambda_gpu.sh` (Vulkan setup — but unrun and using the non-server
  GL package; §6 is the corrected version).
- Run standalone video-gen tests (`logs/test_gen*.log`, `output/test_video_gen/`).

This validation added on top: the HTTPS submodule fix (§1), the `conf.yaml` path rewrite
(§4), the lean FA2 server (§5), the version-matched Vulkan install (§6), the docker
nvidia-runtime + Isaac image build (§7), and the corrected container GPU flags (§8) —
then ran the full closed-loop episode (§8) to completion.

Validated frames: `docs/images/rollout_first.png`, `rollout_strip.png`, `rollout_montage.png`.
