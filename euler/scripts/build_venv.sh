#!/bin/bash
# Build the bare-metal policy-server venv (Python 3.11 + torch2.8/cu129 + groot).
# Run on the LOGIN node (direct internet). One-time, ~15-20 min.
set -euo pipefail

export XDG_DATA_HOME=/cluster/scratch/rjiang/.local/share
export XDG_CACHE_HOME=/cluster/scratch/rjiang/.cache
export UV_PYTHON_INSTALL_DIR=/cluster/scratch/rjiang/.local/share/uv/python
export UV_NO_CONFIG=1            # don't walk parent dirs for uv.toml (/cluster/scratch/uv.toml is unreadable)
UV=/cluster/scratch/rjiang/.local/bin/uv

cd /cluster/scratch/rjiang/dreamzero

echo "== [1/4] Create Python 3.11 venv (uv, seeded with pip) =="
"$UV" venv --no-config --python 3.11 --seed --clear .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q -U pip wheel setuptools

echo "== [2/4] Install dreamzero (-e .) + pinned deps (torch 2.8.0+cu129) =="
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129

echo "== [3/4] Install flash-attn prebuilt wheel (cp311 / torch2.8 / cu12) =="
# Not on the critical path for Blackwell (we default the server to SDPA/eager), but
# matches the validated H100 stack and is harmless to import. Falls back to SDPA if it fails.
pip install --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl" \
  || echo "WARN: flash-attn wheel failed; server will use torch SDPA (eager) — fine on Blackwell."

# deepspeed is a training-only dep pulled in by pyproject; for inference it crashes at import on a
# GPU node ("CUDA_HOME does not exist") because the venv has no CUDA toolkit. Nothing in the inference
# path imports it directly, so remove it. (Reinstall `deepspeed==0.19.1` if you later train here.)
pip uninstall -y deepspeed >/dev/null 2>&1 && echo "removed deepspeed (training-only)" || true

echo "== [4/4] Sanity imports =="
python - <<'PY'
import torch, transformers, diffusers, peft, tianshou, websockets
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
print("transformers", transformers.__version__, "| diffusers", diffusers.__version__, "| peft", peft.__version__)
try:
    import flash_attn; print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn not importable (ok):", type(e).__name__)
PY

echo "VENV_BUILD_COMPLETE"
