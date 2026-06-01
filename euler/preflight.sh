#!/bin/bash
# Readiness check for the DreamZero end-to-end test. Safe to run anywhere (login or GPU node).
#   bash /cluster/scratch/rjiang/dreamzero/euler/preflight.sh
set -uo pipefail
source "$(dirname "$0")/scripts/_env.sh"

ok(){ printf "  [ OK ] %s\n" "$1"; }
no(){ printf "  [MISS] %s\n" "$1"; FAIL=1; }
FAIL=0

echo "=== DreamZero preflight ==="

echo "- Policy-server venv"
{ [ -x "$DZ_VENV/bin/python" ] && "$DZ_VENV/bin/python" -c "import torch,transformers,diffusers,peft,tianshou,websockets" 2>/dev/null; } \
  && ok "venv + core imports ($("$DZ_VENV/bin/python" -c 'import torch;print("torch",torch.__version__)' 2>/dev/null))" \
  || no "venv not ready ($DZ_VENV) — see euler/logs/venv_build.log"

echo "- Base weights"
[ -d "$DZ_REPO/checkpoints/Wan2.1-I2V-14B-480P" ] && [ -f "$DZ_REPO/checkpoints/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" ] \
  && ok "Wan2.1-I2V-14B-480P ($(du -sh "$DZ_REPO/checkpoints/Wan2.1-I2V-14B-480P" 2>/dev/null | cut -f1))" || no "Wan2.1-I2V-14B-480P"
[ -d "$DZ_REPO/checkpoints/DreamZero-AgiBot" ] \
  && ok "DreamZero-AgiBot ($(du -sh "$DZ_REPO/checkpoints/DreamZero-AgiBot" 2>/dev/null | cut -f1))" || no "DreamZero-AgiBot"
[ -d "$DZ_REPO/checkpoints/umt5-xxl" ] && ok "umt5-xxl tokenizer" || no "umt5-xxl tokenizer"

echo "- LoRA checkpoint"
[ -f "$DZ_LORA/model.safetensors" ] && [ -f "$DZ_LORA/experiment_cfg/conf.yaml" ] \
  && ok "LoRA checkpoint-20000" || no "LoRA checkpoint ($DZ_LORA)"

echo "- Isaac Sim SIF"
[ -f "$DZ_SIF" ] && ok "isaac-sim.sif ($(du -sh "$DZ_SIF" 2>/dev/null | cut -f1))" || no "isaac-sim.sif — see euler/logs/isaac_build_*.log"

echo "- Orca-hand submodule meshes (needed for dual-arm URDF)"
n=$(find "$DZ_REPO/sim_envs/assets/orcahand_description" -type f 2>/dev/null | wc -l)
[ "$n" -gt 50 ] && ok "orcahand_description ($n files)" || no "orcahand_description submodule not initialized ($n files) — see README step 'Orca-hand submodule'"

echo "- GPU"
if dz_on_gpu_node; then ok "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"; else echo "  [info] no GPU here (fine on login node; required for run_e2e.sh)"; fi

echo "==========================="
[ "$FAIL" -eq 0 ] && echo "ALL READY ✅  -> bash euler/run_e2e.sh" || echo "Some items missing ❌ (see above)."
exit $FAIL
