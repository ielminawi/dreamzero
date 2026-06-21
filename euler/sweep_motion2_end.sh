#!/bin/bash
cd /cluster/scratch/rjiang/dreamzero
TJ=4095119; RLOG=euler/logs/m2_sweep_results.log
echo "[sweep] waiting for motion2 $TJ to finish $(date)" >>"$RLOG"
for i in $(seq 1 480); do squeue -h -j "$TJ" 2>/dev/null | grep -q "$TJ" || break; sleep 90; done
echo "[sweep] motion2 ended $(date); evaluating checkpoints" >>"$RLOG"
CK=checkpoints/dreamzero_franka_orca_lora_joints_motion2
for STEP in 500 1000 1500; do
  SF=$CK/checkpoint-$STEP/model.safetensors; OUT=output/action_eval_motion2_${STEP}_full
  [ -f "$SF" ] || { echo "[sweep] no ckpt-$STEP" >>"$RLOG"; continue; }
  [ -f "$OUT/action_accuracy.json" ] && { echo "[sweep] $STEP already evaled" >>"$RLOG"; continue; }
  # ensure conf.yaml present (copy from root if missing)
  [ -f "$CK/checkpoint-$STEP/experiment_cfg/conf.yaml" ] || cp -n "$CK/experiment_cfg/conf.yaml" "$CK/checkpoint-$STEP/experiment_cfg/" 2>/dev/null
  rm -rf "$OUT"
  JID=$(CKPT=$PWD/$CK/checkpoint-$STEP DATA=$PWD/data/franka_orca_lerobot_joints_clean OUT=$PWD/$OUT PER_EP=24 \
        sbatch --parsable --job-name=sw_$STEP euler/eval_robuststats.sbatch)
  echo "[sweep] eval $JID step $STEP $(date)" >>"$RLOG"
  for j in $(seq 1 90); do [ -f "$OUT/action_accuracy.json" ] && break; squeue -h -j "$JID" 2>/dev/null | grep -q "$JID" || break; sleep 60; done
  echo "==== [sweep] REPORT step $STEP $(date) ====" >>"$RLOG"
  .venv/bin/python eval_utils/openloop_report.py "$OUT" 2>&1 | grep -E "ckpt|_arm|_hand" >>"$RLOG"
done
echo "[sweep] DONE $(date)" >>"$RLOG"
