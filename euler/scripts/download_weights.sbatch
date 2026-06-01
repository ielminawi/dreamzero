#!/bin/bash
#SBATCH --job-name=dz_weights
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/download_%j.log
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --account=ls_polle
# Downloads the ~111 GB of base weights the LoRA checkpoint needs:
#   Wan2.1-I2V-14B-480P (~66 GB), DreamZero-AgiBot (~45 GB), umt5-xxl (small)
# into the EXACT paths the checkpoint's conf.yaml expects:
#   /cluster/scratch/rjiang/dreamzero/checkpoints/{...}
set -euo pipefail
module load eth_proxy   # compute-node internet via proxy.service.consul:3128

export HF_HOME=/cluster/scratch/rjiang/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0   # proxy-friendly (Python http stack honors HTTPS_PROXY)
CKPT=/cluster/scratch/rjiang/dreamzero/checkpoints
mkdir -p "$CKPT"

# Lightweight downloader venv via uv (compute-node system python lacks ensurepip).
export UV_NO_CONFIG=1
export XDG_DATA_HOME=/cluster/scratch/rjiang/.local/share
export XDG_CACHE_HOME=/cluster/scratch/rjiang/.cache
export UV_PYTHON_INSTALL_DIR=/cluster/scratch/rjiang/.local/share/uv/python
UV=/cluster/scratch/rjiang/.local/bin/uv
HFVENV=/cluster/scratch/rjiang/dreamzero/euler/hfvenv
if ! "$HFVENV/bin/python" -c "import huggingface_hub" >/dev/null 2>&1; then
  rm -rf "$HFVENV"
  "$UV" venv --no-config --python 3.11 --seed "$HFVENV"
  "$HFVENV/bin/pip" install -q "huggingface_hub>=0.24"
fi

CKPT="$CKPT" "$HFVENV/bin/python" - <<'PY'
import os, sys
from huggingface_hub import snapshot_download
ck = os.environ["CKPT"]
jobs = [
    ("Wan-AI/Wan2.1-I2V-14B-480P", "Wan2.1-I2V-14B-480P"),
    ("GEAR-Dreams/DreamZero-AgiBot", "DreamZero-AgiBot"),
    ("google/umt5-xxl", "umt5-xxl"),
]
for repo, sub in jobs:
    dst = os.path.join(ck, sub)
    print(f"==> downloading {repo} -> {dst}", flush=True)
    snapshot_download(repo_id=repo, repo_type="model", local_dir=dst,
                      max_workers=8, tqdm_class=None)
    print(f"==> DONE {repo}", flush=True)
print("ALL_DOWNLOADS_COMPLETE", flush=True)
PY

echo "Disk usage of checkpoints:"
du -sh "$CKPT"/* 2>/dev/null || true
echo "ALL_DOWNLOADS_COMPLETE"
