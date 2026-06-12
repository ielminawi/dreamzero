#!/bin/bash
# Continuous watcher for the r16 training run: confirm healthy, then for each new
# checkpoint auto-submit the eval battery (action accuracy + wrong-action ablation) in
# the free GPU slot; exit (to re-invoke the caller) whenever an eval completes, or the
# training run crashes/ends. Idempotent via the STATE file -> safe to relaunch.
NEW=2376454
REPO=/cluster/scratch/rjiang/dreamzero
CKPTDIR=$REPO/checkpoints/dreamzero_franka_orca_lora_r16
LOG=$REPO/euler/logs/train_r16.$NEW.log
STATE=/cluster/scratch/rjiang/wm_validation/r16_eval_state.txt
EVALROOT=/cluster/scratch/rjiang/wm_validation/r16_eval
mkdir -p "$EVALROOT"; touch "$STATE"
jobstate(){ squeue -j "$1" -h -o "%T" 2>/dev/null; }

# ---- Phase 1: confirm healthy start (re-entrant) ----
if ! grep -q "action_loss_avg" "$LOG" 2>/dev/null; then
  while :; do
    s=$(jobstate $NEW)
    if [ "$s" = RUNNING ]; then
      grep -q "action_loss_avg" "$LOG" 2>/dev/null && break
      grep -qiE "Traceback|CUDA out of memory|ChildFailedError|HydraException|ConfigKeyError|omegaconf" "$LOG" 2>/dev/null && { echo "[watch] r16 FAILED at startup $(date)"; tail -30 "$LOG"; exit 0; }
    fi
    [ -z "$s" ] && { echo "[watch] r16 left queue before healthy ($(sacct -j $NEW --format=State -P -n 2>/dev/null|head -1)) $(date)"; tail -20 "$LOG" 2>/dev/null; exit 0; }
    sleep 30
  done
  echo "[watch] r16 HEALTHY $(date)"; grep -E "out_features=16|Trainable param|action_loss_avg|dynamics_loss_avg" "$LOG" 2>/dev/null | head -6
fi

# ---- Phase 2: watch checkpoints + evals ----
while :; do
  ts=$(jobstate $NEW)
  for d in "$CKPTDIR"/checkpoint-*/; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    [ -f "$d/model.safetensors" ] && [ -d "$d/experiment_cfg" ] || continue
    grep -q "^$n " "$STATE" 2>/dev/null && continue
    eid=$(CKPT="$d" OUT="$EVALROOT/$n" sbatch --parsable "$REPO/euler/eval_ckpt.sbatch" 2>/dev/null)
    echo "$n ${eid:-NONE} SUBMITTED" >> "$STATE"
    echo "[watch] submitted eval for $n -> job ${eid:-FAILED} $(date)"
  done
  while read -r n eid status; do
    [ "$status" = SUBMITTED ] || continue
    [ -n "$eid" ] && [ "$eid" != NONE ] || continue
    if ! jobstate "$eid" | grep -q .; then
      sed -i "s|^$n $eid SUBMITTED|$n $eid REPORTED|" "$STATE"
      echo "=========================================="
      echo "[watch] EVAL COMPLETE for $n (job $eid) $(date)"
      "$REPO/.venv/bin/python" - "$EVALROOT/$n" <<'PY' 2>/dev/null || true
import sys, json, os
d=sys.argv[1]; aj=os.path.join(d,"action_accuracy","action_accuracy.json")
if os.path.isfile(aj):
    s=json.load(open(aj)).get("summary",{})
    print("ACTION ACCURACY (policy MAE vs stay-still, rad):")
    for k,v in s.items():
        beat="BEATS stay" if v['policy_mae']<v['stay_mae'] else "NOT better than stay"
        print(f"  {k:22s} policy={v['policy_mae']:.4f} stay={v['stay_mae']:.4f} corr={v['corr']:.2f} -> {beat}")
else:
    print("(no action_accuracy.json -- check eval log)")
print("ablation curves:", os.path.join(d,"ablation"))
PY
      echo "outputs: $EVALROOT/$n"
      exit 0
    fi
  done < "$STATE"
  if [ -z "$ts" ]; then
    echo "[watch] training $NEW ENDED ($(sacct -j $NEW --format=State -P -n 2>/dev/null|head -1)) $(date)"
    ls -d "$CKPTDIR"/checkpoint-*/ 2>/dev/null
    exit 0
  fi
  sleep 600
done
