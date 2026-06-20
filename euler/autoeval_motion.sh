#!/bin/bash
cd /cluster/scratch/rjiang/dreamzero
TRAIN_JID=3979078
CKDIR=checkpoints/dreamzero_franka_orca_lora_joints_motion
for STEP in 500 1000 1500 2000; do
  SAFET=$CKDIR/checkpoint-$STEP/model.safetensors
  RES=$PWD/output/action_eval_motion_${STEP}_full/action_accuracy.json
  [ -f "$RES" ] && { echo "[autoeval-motion] step $STEP already done"; continue; }
  echo "[autoeval-motion] waiting for $SAFET ..."
  ok=0
  for i in $(seq 1 700); do
    if [ -f "$SAFET" ]; then sz=$(stat -c %s "$SAFET" 2>/dev/null); sleep 20; sz2=$(stat -c %s "$SAFET" 2>/dev/null)
      [ "$sz" = "$sz2" ] && [ "${sz:-0}" -gt 2000000000 ] && { ok=1; break; }
    fi
    squeue -h -j "$TRAIN_JID" 2>/dev/null | grep -q "$TRAIN_JID" || { [ -f "$SAFET" ] || { echo "[autoeval-motion] train job $TRAIN_JID gone before step $STEP"; break; } ; }
    sleep 60
  done
  [ "$ok" = 1 ] || { echo "[autoeval-motion] giving up on step $STEP"; continue; }
  echo "[autoeval-motion] ckpt-$STEP complete; submitting eval"
  JID=$(CKPT=$PWD/$CKDIR/checkpoint-$STEP DATA=$PWD/data/franka_orca_lerobot_joints_clean \
        OUT=$PWD/output/action_eval_motion_${STEP}_full PER_EP=24 \
        sbatch --parsable euler/eval_robuststats.sbatch)
  echo "[autoeval-motion] eval job $JID for step $STEP"
  for i in $(seq 1 90); do
    [ -f "$RES" ] && { echo "[autoeval-motion] RESULT step $STEP ready"; break; }
    squeue -h -j "$JID" 2>/dev/null | grep -q "$JID" || { [ -f "$RES" ] || echo "[autoeval-motion] eval $JID ended w/o result (step $STEP)"; break; }
    sleep 60
  done
  echo "[autoeval-motion] === REPORT step $STEP ==="
  .venv/bin/python eval_utils/openloop_report.py output/action_eval_motion_${STEP}_full 2>&1 | head -30
done
echo "[autoeval-motion] DONE all steps"
