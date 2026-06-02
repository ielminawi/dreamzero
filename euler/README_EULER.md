# DreamZero end-to-end inference on Euler (dual Franka + Orca, Isaac Sim)

This directory sets up the **closed-loop Isaac Sim evaluation** of your `franka_orca_bimanual`
LoRA checkpoint on Euler, adapted from `docs/RUN_FROM_SCRATCH.md` (validated on Lambda H100)
to **Euler / Apptainer / SLURM** and your **RTX PRO 6000 (Blackwell, 96 GB)** GPU.

Two pieces run on a single GPU node and talk over `localhost:5000`:
- **Policy server** — bare-metal venv, loads the Wan2.1 video world-model + your LoRA, serves
  48-dim actions over WebSocket. Runs in **eager / torch-SDPA** mode (Blackwell-safe).
- **Isaac Sim** — Apptainer container (Isaac Sim 4.5.0 + IsaacLab 2.1.0), drives the two-Franka
  + Orca-hand scene, queries the server every `--open-loop-horizon` steps, writes a rollout video.

> ## ✅ Validated end-to-end on Euler — 2026-06-01
> Ran the full closed loop on node **eu-g7-010** (RTX PRO 6000 Blackwell, 96 GB, driver 580.159.03,
> CUDA 13). Server loaded in ~6 min (44.6 GB VRAM), inference produced 24×48 action chunks
> (~5 s/chunk steady-state), and the **Isaac dual-Franka+Orca sim ran a full 1500-step (30 s) episode
> in closed loop** (~8.5 min, ~86 GB total VRAM with rendering). Rollout video written to
> `output/sim/2026-06-01/22-19-18/episode_0.mp4`. All the Blackwell/Euler-specific fixes below are
> already baked into the scripts, so `run_e2e.sh` does this for you.

> ## ✅ Sim scene + joint setup validated — 2026-06-02
> Debugged the sim vs. training contract and the manipulation scene (details + how to reproduce in
> [`../docs/SIM_VALIDATION_AND_SCENE.md`](../docs/SIM_VALIDATION_AND_SCENE.md)):
> - **Joint setup** — action-term order, `use_default_offset`, and the Isaac hand-joint reorder were
>   fixed so the 48-dim absolute action maps to the right joints (the server already returns absolute
>   targets). Round-trip "hold current pose" drift ~0.03–0.08 rad.
> - **Scene objects** — added `bag_groceries` stand-ins (paper bag + container + tube + round item).
> - **Physics fidelity** — gave the Orca fingertip links valid inertia (0 PhysX warnings) and raised
>   hand actuator gains so fingers track grasps.
> - **Collisions** — the table is now a solid collider (was a visual-only ghost → objects/hands fell
>   through), shifted forward so it doesn't interpenetrate the arm bases.
> - **Camera** — `focal_length` 15→20 to frame the workspace like the training videos.
> Quick check (any 24 GB+ GPU, ~5 min): `sbatch euler/inspect_joints.sbatch` → results in
> `output/sim/joint_inspection.txt` + a rendered frame `output/sim/sim_oakd_frame.png`.

---

## TL;DR — run the test

1) Grab a GPU node (your command):
```bash
srun --time=4:00:00 --ntasks=1 --cpus-per-task=16 --mem-per-cpu=16G \
     --gpus-per-node=1 --gres=gpumem:96g --account=ls_polle --pty bash
```

2) (optional) check everything is in place:
```bash
bash /cluster/scratch/rjiang/dreamzero/euler/preflight.sh
```

3) Run the whole thing (server + Isaac + video):
```bash
bash /cluster/scratch/rjiang/dreamzero/euler/run_e2e.sh
```
First model load takes a few minutes; one 1500-step episode then runs (~10 min). The rollout
video path is printed at the end:
`/cluster/scratch/rjiang/dreamzero/output/sim/<date>/<time>/episode_0.mp4`

Override the task / length:
```bash
DZ_INSTRUCTION="pick up the cube" DZ_EPISODES=1 DZ_HORIZON=24 \
  bash /cluster/scratch/rjiang/dreamzero/euler/run_e2e.sh
```

**Batch (non-interactive)** — same thing as a SLURM job, pinned to the Blackwell RTX PRO 6000:
```bash
sbatch euler/run_e2e.sbatch          # #SBATCH --gpus=nvidia_rtx_pro_6000:1 ; log: euler/logs/e2e.<jobid>.log
```
`gpumem:96g` alone can also land on a Hopper card; `--gpus=nvidia_rtx_pro_6000:1` forces Blackwell
(nodes `eu-g7-*`). On Hopper/Ampere you can instead set `DZ_SIM_DEVICE=cuda` and drop the model pin.

---

## Step-by-step (if you'd rather drive it yourself)

```bash
# 1. fast check the server alone (no Isaac): loads model + LoRA, runs 3 inference chunks
bash euler/smoketest.sh

# 2. or: start the server, keep it up, run evals against it
bash euler/start_server.sh           # waits until /healthz is OK
bash euler/run_eval.sh 1 "pick up the cube" 24
bash euler/stop_server.sh            # when done
```

Manual server (what start_server.sh runs under the hood), from the repo root:
```bash
cd /cluster/scratch/rjiang/dreamzero
CUDA_VISIBLE_DEVICES=0 ENABLE_TENSORRT=True ATTENTION_BACKEND=torch \
TOKENIZER_DIR=$PWD/checkpoints/umt5-xxl HF_HUB_OFFLINE=1 \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  socket_test_optimized_AR.py --port 5000 --enable-dit-cache \
  --embodiment franka_orca_bimanual \
  --model-path /cluster/scratch/rjiang/checkpoints/dreamzero_franka_orca_lora/checkpoint-20000
# health: curl http://localhost:5000/healthz   -> OK
```

---

## What I pre-built for you (no GPU needed)

| Item | Location |
|---|---|
| Policy-server venv (Py3.11, torch2.8+cu129, flash-attn, groot `-e .`) | `/.venv` |
| Base weights Wan2.1-I2V-14B-480P / DreamZero-AgiBot / umt5-xxl (~111 GB) | `/checkpoints/` |
| Isaac Sim 4.5.0 + IsaacLab 2.1.0 container | `/isaac-sim.sif` |
| Launcher scripts | `euler/*.sh` |

The checkpoint's `conf.yaml` already points at `/cluster/scratch/rjiang/dreamzero/checkpoints/...`
(you trained it on this account), so **no path-rewriting is needed** — the weights are downloaded
to exactly those paths and the server runs from the repo root (so `./checkpoints/DreamZero-AgiBot`
resolves).

---

## Orca-hand submodule — already done ✅

The dual-arm URDFs need the **`orcahand_description`** meshes (a git submodule). This has been
**initialized** (237 files, all 78 URDF mesh references resolve). If you ever need to re-init it
(fresh clone), run on the login node:
```bash
cd /cluster/scratch/rjiang/dreamzero
git config submodule.sim_envs/assets/orcahand_description.url \
    https://github.com/srl-ethz/orcahand_description_team3.git
git submodule update --init sim_envs/assets/orcahand_description
```

---

## Euler/Blackwell-specific fixes (all automatic in the scripts)

These are the non-obvious things that make it work here; you don't have to do anything, but here's
what `run_e2e.sh` handles under the hood (and why):

| Fix | Why |
|---|---|
| **Removed `deepspeed`** from the venv | training-only dep; on a GPU node it demands a CUDA toolkit (`CUDA_HOME`) the venv lacks and crashes transformers at import. Nothing in inference uses it. |
| **`ENABLE_TENSORRT=True ATTENTION_BACKEND=torch`** (server) | Blackwell-safe: torch SDPA, no flash-attn kernel calls, no `torch.compile`. Loaded at **44.6 GB** VRAM. |
| **Stage `DreamZero-AgiBot` → node-local NVMe `$TMPDIR`** before loading | the loader **mmaps** the AgiBot safetensors, and mmap on Lustre is ~8 MB/s (a load that never finishes). Local NVMe mmap is ~1.7 GB/s → full server load in ~6 min. (`serve_fg.sh` / `dz_local_rundir`.) |
| **Isaac writable overlay** (`euler/isaac_overlay.img`, 8 GB) | Isaac writes to `/isaac-sim/kit/{data,cache}` which are read-only in the SIF; `--writable-tmpfs` is capped at 64 MB here. Auto-created on first run; the RTX shader cache persists in it (later runs start faster). |
| **`--env TERM=xterm`** into the container | `isaaclab.sh` dies with `'ansi+tabs': unknown terminal type` otherwise. |
| **`--device cpu`** for Isaac | Isaac Sim 4.5 bundles **torch 2.5.1+cu118** (max sm_90) which can't run on Blackwell sm_120 ("no kernel image"). Physics/state run on CPU (1 env — trivial); RTX **rendering stays on the GPU** via Vulkan + warp (warp 1.14 is cu12.9/NVRTC, supports Blackwell). Override with `DZ_SIM_DEVICE=cuda` on Hopper/Ampere nodes. |
| **ETH proxy passed into the container** | Isaac fetches default assets (ground plane, materials) from its online S3 asset server; the compute node only reaches the internet via `proxy.service.consul:3128`. Assets cache into the overlay after the first run. |
| **`--home euler/ovhome`** (scratch) | Isaac/Omniverse caches go to scratch, not your 50 GB home quota. |

## Notes
- **VRAM**: server 44.6 GB + Isaac RTX rendering ~41 GB ≈ **86 GB** on the 96 GB card. Fits.
- **Timing**: first server load ~6 min (then `/healthz` OK); first Isaac run compiles RTX shaders
  (slower, esp. on a busy node) — subsequent runs reuse the overlay cache. A 1500-step episode is
  ~8–10 min.
- **Output / logs**: videos in `output/sim/<date>/<time>/episode_N.mp4`; logs in `euler/logs/`
  (`server.log`, `isaac_eval.log`).
- **Ports**: server on `5000` (`DZ_PORT` to change); `localhost` is in `no_proxy`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `preflight` shows MISS for weights / SIF | a build job is still running — `squeue --me`; logs in `euler/logs/` |
| Server never healthy | `tail -f euler/logs/server.log`; cold load reads weights off Lustre — give it time (the 1800 s health timeout covers it) |
| Server OOM (CPU) | only happens if you launch it inside a sub-allocation with little RAM; the interactive session has the full 256 GB, so just run `run_e2e.sh` directly |
| Isaac `no kernel image is available` | Isaac torch is cu118; this is why we use `--device cpu` (already the default). Don't set `DZ_SIM_DEVICE=cuda` on Blackwell. |
| Isaac ground plane / asset `NoneType ... GetPath` | proxy not reaching S3 — check `module load eth_proxy`; the launcher passes `proxy.service.consul:3128` into the container |
| Isaac `Read-only file system /isaac-sim/kit/...` | overlay missing — `apptainer overlay create --size 8192 euler/isaac_overlay.img` (run_eval auto-creates it) |
| Isaac missing `.stl` / URDF import fails | Orca-hand submodule (already initialized; see above) |
| `torch.compile` error | already disabled via `ENABLE_TENSORRT=True`; as a belt-and-suspenders add `TORCHDYNAMO_DISABLE=1` |
