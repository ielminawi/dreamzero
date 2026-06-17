#!/bin/bash
# Shared config + helpers for DreamZero end-to-end inference on Euler.
# Sourced by the euler/*.sh launchers. Safe to `source` from any cwd.

# ---- Paths (everything lives on scratch) ----
export DZ_REPO=/cluster/scratch/rjiang/dreamzero
export DZ_VENV="$DZ_REPO/.venv"
export DZ_LORA="${DZ_LORA:-/cluster/scratch/rjiang/checkpoints/dreamzero_franka_orca_lora/checkpoint-20000}"
export DZ_SIF="$DZ_REPO/isaac-sim.sif"
export DZ_TOKENIZER_DIR="$DZ_REPO/checkpoints/umt5-xxl"
export DZ_OUTPUT="$DZ_REPO/output/sim"
export DZ_LOGDIR="$DZ_REPO/euler/logs"
export DZ_OVHOME="$DZ_REPO/euler/ovhome"        # scratch HOME for Isaac/Omniverse caches (protects 50GB home quota)
export DZ_OVERLAY="$DZ_REPO/euler/isaac_overlay.img"  # optional writable overlay for Isaac (used if present)
export DZ_PORT="${DZ_PORT:-5000}"

# ---- Eval defaults (override by exporting before calling) ----
export DZ_INSTRUCTION="${DZ_INSTRUCTION:-pick up the cube}"
export DZ_EPISODES="${DZ_EPISODES:-1}"
export DZ_HORIZON="${DZ_HORIZON:-24}"

mkdir -p "$DZ_LOGDIR" "$DZ_OUTPUT" "$DZ_OVHOME" 2>/dev/null

# ---- Server launch env ----
# Blackwell (RTX PRO 6000 / sm_120) safe: ENABLE_TENSORRT=True forces torch SDPA attention
# AND disables torch.compile (no flash-attn kernel calls, no compile-on-reset crashes).
dz_server_env() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export NO_ALBUMENTATIONS_UPDATE=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1
  export TOKENIZER_DIR="$DZ_TOKENIZER_DIR"
  export ENABLE_TENSORRT="${ENABLE_TENSORRT:-True}"      # eager + SDPA
  export ATTENTION_BACKEND="${ATTENTION_BACKEND:-torch}" # explicit SDPA
  export DS_ACCELERATOR=cpu   # deepspeed is a training-only dep; on a GPU node its CUDA accelerator
                              # demands a CUDA toolkit (CUDA_HOME) we don't have. Forcing CPU avoids
                              # transformers' deepspeed integration crashing at import. (We also uninstall it.)
  export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
}

dz_on_gpu_node()      { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }
dz_server_running()   { curl -sf "http://localhost:${DZ_PORT}/healthz" >/dev/null 2>&1; }

dz_wait_health() {
  local timeout="${1:-1800}" waited=0   # cold load reads ~120GB of weights off Lustre; can take 15-25 min
  echo "Waiting for policy server /healthz on :${DZ_PORT} (timeout ${timeout}s, tail $DZ_LOGDIR/server.log)..."
  while ! dz_server_running; do
    if [ -f "$DZ_LOGDIR/server.pid" ] && ! kill -0 "$(cat "$DZ_LOGDIR/server.pid" 2>/dev/null)" 2>/dev/null; then
      echo "ERROR: server process died. Last 40 log lines:"; tail -40 "$DZ_LOGDIR/server.log"; return 1
    fi
    sleep 5; waited=$((waited+5))
    [ $((waited % 30)) -eq 0 ] && echo "  ...${waited}s still loading"
    [ "$waited" -ge "$timeout" ] && { echo "ERROR: not healthy after ${timeout}s"; return 1; }
  done
  echo "==> Policy server HEALTHY on :${DZ_PORT}"
}

# The loader mmaps the DreamZero-AgiBot *.safetensors, and mmap page-faults on Lustre are
# ~8 MB/s (vs 1.7 GB/s on the node-local NVMe). So stage AgiBot (43 GB) to $TMPDIR (local NVMe,
# ~30 s) and run the server from a mirror dir whose ./checkpoints/DreamZero-AgiBot points at the
# local copy (the conf.yaml uses a relative path for AgiBot; Wan2.1/umt5 are absolute and load
# fast via torch.load). Echoes the dir to run the server from. Set DZ_STAGE_LOCAL=0 to disable.
dz_local_rundir() {
  if [ "${DZ_STAGE_LOCAL:-1}" != "1" ] || [ -z "${TMPDIR:-}" ]; then echo "$DZ_REPO"; return 0; fi
  local stage="$TMPDIR/dz_agibot" run="$TMPDIR/dzrun" e name
  if [ ! -s "$stage/model-00010-of-00010.safetensors" ]; then
    echo "Staging AgiBot -> local NVMe ($stage), one-time per node (~30s)..." >&2
    mkdir -p "$stage" && cp -r "$DZ_REPO/checkpoints/DreamZero-AgiBot/." "$stage/" >&2 || { echo "$DZ_REPO"; return 0; }
  fi
  rm -rf "$run"; mkdir -p "$run/checkpoints"
  for e in "$DZ_REPO"/*; do name=$(basename "$e"); [ "$name" = checkpoints ] && continue; ln -sfn "$e" "$run/$name"; done
  ln -sfn "$stage" "$run/checkpoints/DreamZero-AgiBot"
  for e in "$DZ_REPO"/checkpoints/*; do name=$(basename "$e"); [ "$name" = DreamZero-AgiBot ] && continue; ln -sfn "$e" "$run/checkpoints/$name"; done
  echo "$run"
}

dz_start_server_bg() {
  if dz_server_running; then echo "Policy server already healthy on :${DZ_PORT}."; return 0; fi
  dz_server_env
  local run; run="$(dz_local_rundir)"
  echo "Starting policy server (eager/SDPA, model=$DZ_LORA) on :${DZ_PORT}  [cwd=$run]"
  echo "  log -> $DZ_LOGDIR/server.log"
  ( cd "$run" && setsid "$DZ_VENV/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 \
      socket_test_optimized_AR.py \
      --port "$DZ_PORT" --enable-dit-cache --embodiment franka_orca_bimanual \
      --model-path "$DZ_LORA" >"$DZ_LOGDIR/server.log" 2>&1 & echo $! >"$DZ_LOGDIR/server.pid" )
  echo "  server PID $(cat "$DZ_LOGDIR/server.pid" 2>/dev/null)"
}

dz_stop_server() {
  pkill -f "socket_test_optimized_AR.py" 2>/dev/null && echo "Stopped policy server." || echo "No server process found."
  rm -f "$DZ_LOGDIR/server.pid"
}

# Build the apptainer argv for the Isaac eval and run it.
dz_run_isaac_eval() {
  local episodes="${1:-$DZ_EPISODES}" instr="${2:-$DZ_INSTRUCTION}" horizon="${3:-$DZ_HORIZON}"
  mkdir -p "$DZ_OUTPUT" "$DZ_OVHOME"
  # Isaac writes to /isaac-sim/kit/{data,cache,logs}, read-only in the SIF -> needs a writable
  # overlay (--writable-tmpfs is capped at 64MB here, too small). Create an 8GB overlay once.
  if [ ! -f "$DZ_OVERLAY" ]; then
    echo "Creating Isaac writable overlay ($DZ_OVERLAY, 8GB)..." >&2
    apptainer overlay create --size 8192 "$DZ_OVERLAY" >&2 2>&1 || echo "WARN: overlay create failed" >&2
  fi
  # Isaac fetches default assets (ground plane, materials) from its online S3 asset server; the
  # compute node only reaches the internet via the ETH proxy. Pass it in (no_proxy keeps the policy
  # WebSocket on localhost direct). Assets get cached into the overlay after the first run.
  local proxy="${DZ_PROXY:-http://proxy.service.consul:3128}"
  local args=( run --nv --env TERM=xterm
               --env "http_proxy=$proxy" --env "https_proxy=$proxy"
               --env "HTTP_PROXY=$proxy" --env "HTTPS_PROXY=$proxy"
               --env "no_proxy=localhost,127.0.0.1" --env "NO_PROXY=localhost,127.0.0.1"
               --home "$DZ_OVHOME:$DZ_OVHOME"
               --bind "$DZ_REPO:/app" --bind "$DZ_OUTPUT:/output" --pwd /app )
  [ -f "$DZ_OVERLAY" ] && args+=( --overlay "$DZ_OVERLAY" )
  # Isaac Sim 4.5 ships torch 2.5.1+cu118 (max sm_90) which cannot run on Blackwell sm_120
  # ("no kernel image"). Run PhysX/state tensors on CPU (1 env, trivial); RTX rendering stays on
  # the GPU via Vulkan + warp (warp 1.14 is cu12.9/NVRTC, supports Blackwell). Override w/ DZ_SIM_DEVICE.
  args+=( "$DZ_SIF" eval_utils/run_sim_eval_bimanual.py --headless --enable_cameras
          --device "${DZ_SIM_DEVICE:-cpu}"
          --host localhost --port "$DZ_PORT"
          --instruction "$instr" --episodes "$episodes" --open-loop-horizon "$horizon" )
  # JOINT-SPACE policy: arm dims [0:14] are Panda joint angles, not EE poses -> bypass
  # FK on obs and IK on actions. Enable via DZ_JOINT_ACTIONS=1.
  [ "${DZ_JOINT_ACTIONS:-0}" = "1" ] && args+=( --joint-actions )
  echo "==> apptainer ${args[*]}"
  apptainer "${args[@]}"
}

dz_latest_video() { ls -t "$DZ_OUTPUT"/*/*/*.mp4 2>/dev/null | head -1; }
