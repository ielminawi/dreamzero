#!/bin/bash
#SBATCH --job-name=dz_evaljt
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=14G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:90g
#SBATCH --tmp=120G
#SBATCH --account=ls_polle
#SBATCH --output=/cluster/scratch/rjiang/dreamzero/euler/logs/evaljt.%j.log
# Action-prediction accuracy on the JOINT-SPACE run: trained checkpoint vs a NO-TRAINING
# baseline checkpoint, both vs the demonstration actions (and the stay-still floor that
# action_eval.py already computes). Answers: did training actually beat an untrained policy?
set -uo pipefail
DZ_REPO=/cluster/scratch/rjiang/dreamzero
DATA="${DATA:-$DZ_REPO/data/franka_orca_lerobot_joints_clean}"
CKPT_TRAINED="${CKPT_TRAINED:-$DZ_REPO/checkpoints/dreamzero_franka_orca_lora_joints/checkpoint-5000}"
UNTR_DIR="${UNTR_DIR:-$DZ_REPO/checkpoints/dreamzero_franka_orca_lora_joints_untrained}"
CKPT_UNTRAINED="${CKPT_UNTRAINED:-$(ls -d "$UNTR_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)}"
OUT="${OUT:-$DZ_REPO/output/action_eval_joints}"
EPISODES="${EPISODES:-2,5,32,73,147}"
PER_EP="${PER_EP:-8}"

echo "==> trained=$CKPT_TRAINED"
echo "==> untrained=$CKPT_UNTRAINED"
[ -d "$CKPT_TRAINED" ] || { echo "ERROR: trained ckpt missing"; exit 1; }
[ -d "$CKPT_UNTRAINED" ] || { echo "ERROR: untrained ckpt missing (run make_untrained_joints.sbatch first)"; exit 1; }

source "$DZ_REPO/euler/scripts/_env.sh"; dz_server_env
export ATTENTION_BACKEND=torch ENABLE_TENSORRT=True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- conf fix: these checkpoints recorded a now-dead node-local AgiBot path
# (pretrained_model_path=/scratch/tmp.<jobid>/dz_agibot). At eval, _resolve_base_model_path
# can't resolve it and load_lora SILENTLY falls back to vanilla Wan2.1 (wrong frozen base).
# Repoint to the relative AgiBot dir so it resolves to the fast NVMe symlink in the rundir. ---
for ck in "$CKPT_TRAINED" "$CKPT_UNTRAINED"; do
  cf="$ck/experiment_cfg/conf.yaml"
  if [ -f "$cf" ]; then
    sed -i 's|^pretrained_model_path:.*|pretrained_model_path: ./checkpoints/DreamZero-AgiBot|' "$cf"
    echo "patched base path in $cf -> $(grep '^pretrained_model_path:' "$cf")"
  fi
done

RUN="$(dz_local_rundir)"; echo "==> node $(hostname) run $RUN"; nvidia-smi -L | head
cd "$RUN"
mkdir -p "$OUT/trained" "$OUT/untrained"

run_eval () {  # $1=label $2=ckpt $3=outdir $4=master_port
  echo "=== [$1] action_eval: $2 ==="
  "$DZ_REPO/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 \
    --master-port "$4" \
    "$DZ_REPO/eval_utils/action_eval.py" \
    --model-path "$2" --dataset-dir "$DATA" \
    --episodes "$EPISODES" --per-episode "$PER_EP" --out-dir "$3"
  echo "[$1] action_eval rc=$?"
}

run_eval trained   "$CKPT_TRAINED"   "$OUT/trained"   12362
run_eval untrained "$CKPT_UNTRAINED" "$OUT/untrained" 12363

echo "=== COMPARISON: trained vs untrained vs stay-still (MAE vs demo action) ==="
"$DZ_REPO/.venv/bin/python" - "$OUT/trained/action_accuracy.json" "$OUT/untrained/action_accuracy.json" <<'PY'
import json, sys
tr = json.load(open(sys.argv[1]))["summary"]
un = json.load(open(sys.argv[2]))["summary"]
segs = list(tr.keys())
print(f"{'segment':24s} {'trained':>9s} {'untrained':>9s} {'stay':>9s} {'tr<un?':>7s} {'tr<stay?':>9s} {'tr_corr':>8s}")
def row(name, t, u, s, c):
    print(f"{name:24s} {t:9.4f} {u:9.4f} {s:9.4f} {('YES' if t<u else 'no'):>7s} {('YES' if t<s else 'no'):>9s} {c:8.2f}")
import numpy as np
tt=uu=ss=0.0; n=0
for k in segs:
    t=tr[k]["policy_mae"]; u=un[k]["policy_mae"]; s=tr[k]["stay_mae"]; c=tr[k].get("corr",float('nan'))
    row(k.replace('_joint_pos',''), t, u, s, c)
    tt+=t; uu+=u; ss+=s; n+=1
print("-"*72)
row("OVERALL(mean of segs)", tt/n, uu/n, ss/n, float('nan'))
print()
print("Read: trained should be < untrained (training beat a random-head policy) AND")
print("      < stay-still (it learned real motion, not just 'do nothing').")
PY
echo "==> eval done $(date); outputs in $OUT  (per-ckpt action_accuracy.json + .png)"
