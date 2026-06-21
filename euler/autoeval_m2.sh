#!/bin/bash
cd /cluster/scratch/rjiang/dreamzero
TJ=4095119   # motion2 train job id
CK=checkpoints/dreamzero_franka_orca_lora_joints_motion2
RLOG=euler/logs/autoeval_m2_results.log
for STEP in 500 1000 1500 2000; do
  SF=$CK/checkpoint-$STEP/model.safetensors
  OUT=output/action_eval_motion2_${STEP}_full; RES=$OUT/action_accuracy.json
  [ -f "$RES" ] && { echo "[m2] $STEP done already" >>"$RLOG"; continue; }
  # wait for checkpoint complete (or train job gone)
  ok=0
  for i in $(seq 1 800); do
    if [ -f "$SF" ]; then s=$(stat -c %s "$SF" 2>/dev/null); sleep 20; s2=$(stat -c %s "$SF" 2>/dev/null)
      [ "$s" = "$s2" ] && [ "${s:-0}" -gt 2000000000 ] && { ok=1; break; }; fi
    squeue -h -j "$TJ" 2>/dev/null | grep -q "$TJ" || { [ -f "$SF" ] || { echo "[m2] train $TJ gone before $STEP" >>"$RLOG"; break; } ; }
    sleep 60
  done
  [ "$ok" = 1 ] || { echo "[m2] skip $STEP (no ckpt)" >>"$RLOG"; continue; }
  JID=$(CKPT=$PWD/$CK/checkpoint-$STEP DATA=$PWD/data/franka_orca_lerobot_joints_clean OUT=$PWD/$OUT PER_EP=24 \
        sbatch --parsable --job-name=evm2_$STEP euler/eval_robuststats.sbatch)
  echo "[m2] eval $JID for step $STEP $(date)" >>"$RLOG"
  for i in $(seq 1 90); do
    [ -f "$RES" ] && break
    squeue -h -j "$JID" 2>/dev/null | grep -q "$JID" || break
    sleep 60
  done
  echo "==== [m2] REPORT step $STEP $(date) ====" >>"$RLOG"
  .venv/bin/python eval_utils/openloop_report.py "$OUT" 2>&1 | grep -E "ckpt|_arm|_hand" >>"$RLOG"
done
echo "[m2] DONE $(date)" >>"$RLOG"
