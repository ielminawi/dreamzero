#!/bin/bash
cd /cluster/scratch/rjiang/dreamzero
RLOG=euler/logs/autoeval_results.log
echo "[daemon] start $(date)" >> "$RLOG"
declare -A TAG=( [dreamzero_franka_orca_lora_joints_motion]=motion [dreamzero_franka_orca_lora_joints_motion2]=motion2 )
for round in $(seq 1 400); do
  pending=0
  for d in dreamzero_franka_orca_lora_joints_motion dreamzero_franka_orca_lora_joints_motion2; do
    t=${TAG[$d]}
    for ck in checkpoints/$d/checkpoint-*; do
      [ -d "$ck" ] || continue
      step=$(basename "$ck" | sed 's/checkpoint-//')
      sf="$ck/model.safetensors"; out="output/action_eval_${t}_${step}_full"
      [ -f "$sf" ] || continue
      [ -f "$out/action_accuracy.json" ] && continue   # already evaled
      # ensure ckpt is complete + stable
      sz=$(stat -c %s "$sf" 2>/dev/null); sleep 5; sz2=$(stat -c %s "$sf" 2>/dev/null)
      [ "$sz" = "$sz2" ] && [ "${sz:-0}" -gt 2000000000 ] || { pending=1; continue; }
      # already submitted & still running? skip resubmit
      if squeue -h -u rjiang -o "%j" 2>/dev/null | grep -q "ev_${t}_${step}"; then pending=1; continue; fi
      echo "[daemon] submitting eval ${t}-${step} $(date)" | tee -a "$RLOG"
      CKPT=$PWD/$ck DATA=$PWD/data/franka_orca_lerobot_joints_clean OUT=$PWD/$out PER_EP=24 \
        sbatch --job-name=ev_${t}_${step} euler/eval_robuststats.sbatch >/dev/null 2>&1
      pending=1
    done
  done
  # report any newly-finished evals
  for out in output/action_eval_motion_*_full output/action_eval_motion2_*_full; do
    [ -f "$out/action_accuracy.json" ] || continue
    [ -f "$out/.reported" ] && continue
    echo "==== REPORT $out $(date) ====" | tee -a "$RLOG"
    .venv/bin/python eval_utils/openloop_report.py "$out" 2>&1 | grep -E "ckpt|_arm|_hand" | tee -a "$RLOG"
    touch "$out/.reported"
  done
  # exit when no training jobs AND nothing pending
  trains=$(squeue -h -j 3979078,4036818 2>/dev/null | wc -l)
  [ "$trains" = "0" ] && [ "$pending" = "0" ] && { echo "[daemon] all done $(date)" | tee -a "$RLOG"; exit 0; }
  sleep 120
done
echo "[daemon] max rounds $(date)" | tee -a "$RLOG"
