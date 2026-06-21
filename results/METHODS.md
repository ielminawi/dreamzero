# Methods & Reproducibility

## Model & task
- Base: DreamZero VLA, video-world-model + flow-matching action head, LoRA (r16) on a frozen DiT backbone with
  the top-3 DiT blocks unfrozen for vision↔action coupling. Base weights warm-started from `DreamZero-AgiBot`.
- Embodiment: dual Franka arm + Orca hand. Action/state = 48-D: `[left_arm 7, right_arm 7, left_hand 17, right_hand 17]`.
- Action chunk horizon H = 24; 2 camera views (aria_rgb_cam, oakd_front_view); task instruction "bag the groceries".
- Action target is **relative** (anchor-subtracted: `action[t+k] − state[t_chunk_start]`), q99-normalized to [−1,1].

## Dataset
- `data/franka_orca_lerobot_joints_clean` — deglitched joint-space LeRobot dataset with robust (q025/q975)
  relative-action normalization stats. (Earlier defect: tail-dominated q99 stats inflated the normalization
  span ×15–42, fixed here.)
- Note (relevant caveat): in this dataset the **hand action equals the observation state exactly** at every
  step (action−state = 0 for both hands across all episodes) — the hand "target" is the hand's own future
  trajectory relative to the chunk anchor; it is structured/learnable (constant-velocity ceiling dir_cos
  0.66–0.75) but the model does not learn it.

## Training recipes compared (all warm-started, LoRA r16, unfreeze-top-3, lr 5e-5, bs 1 × grad-accum 2)
| label | key config | purpose |
|---|---|---|
| `clean` (baseline) | robust stats, plain action loss | plateaued reference |
| `varnorm` | + per-dim variance-normalized action loss (`action_loss_variance_normalize`, `action_loss_dim_variances`, floor 0.02) + fresh joint head | equalize per-dim gradient |
| `motion_v1` | varnorm + explicit within-chunk **first-difference motion loss** (`action_loss_motion_weight=1.0`), warm-start varnorm-5000 keeping the joint head | inject gradient on within-chunk motion |
| `motion_v2_handspush` | motion_v1 + corrected `DIM_VAR` (true within-chunk motion variance) + floor 0.002 + **hand segment weights ×2.5** + `motion_weight=2.0` | target the stuck hands |
Recipes: `scripts/train/franka_orca_training_joints_{varnorm,motion,motion2}.sh`; sbatches in `euler/`.
Code: the motion loss + variance-normalize live in
`groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`; warm-start-skip-head flag in
`groot/vla/experiment/base.py`.

## Open-loop evaluation (primary, NON-SIM)
- `eval_utils/action_eval.py` runs the policy on held-out demo frames (episodes 2,5,32,73,147 × 24 timesteps
  each = 120 chunks), denormalizes predictions to absolute joints, and writes `action_accuracy.json` (summary
  + per-sample rows) + `all_traces.npz` (all pred/gt/state chunk traces) per checkpoint.
- `eval_utils/openloop_report.py` computes the per-segment motion-capture ratio + direction cosine on the
  high-motion subset (top third by true motion).
- Metric definitions: see RESULTS_SUMMARY.md §3. The **decisive metric is `hi_dir_cos`** (within-chunk motion
  direction on high-motion chunks); `hi_capture_ratio` measures magnitude; `disp_mae` is offset-free shape.
- IMPORTANT eval gotcha: checkpoints written via TMPDIR base-staging record a dead `pretrained_model_path` in
  `experiment_cfg/conf.yaml`; the eval sbatch `sed`-patches it to `./checkpoints/DreamZero-AgiBot`. Verify the
  log says "Rebuilding DiT base from … DreamZero-AgiBot" (not "falling back to … Wan2.1").

## Closed-loop evaluation (Isaac Sim)
- `eval_utils/run_sim_eval_bimanual.py` (joint-space mode, `--joint-actions`) + `euler/e2e_joints_smoke.sbatch`.
- **Re-anchor fix** (`DZ_REANCHOR=1`, default): apply predicted *motion* from the measured state
  (`q_target = q_now + (pred − pred[0])`) instead of absolute predictions — fixes the per-chunk anchor-offset
  integration drift that previously ratcheted joints to their limits. Optional `DZ_CLAMP_JOINTS=1`.
- **Object-pose logging** (added): records grocery_bag / white_container / blue_tube / round_item world poses
  each step → `episode_N_objects.npz`, with a per-object "moved Xcm / max_lift Ycm" task-success proxy.
  This is what makes closed-loop success objectively measurable (vs guessing from video).

## Reproducing the results package
```
.venv/bin/python results/consolidate_metrics.py   # -> results/data/*.csv,*.json
.venv/bin/python results/make_figures.py          # -> results/figures/*.png
```
Both read the eval outputs in `output/action_eval_*` and `output/sim/*/episode_*_objects.npz`.

## Infrastructure notes (Euler HPC) — affected runs, documented for completeness
- Account `ls_polle`: hard 2-GPU cap; Blackwell `cuda13pr` (96 GB), request `--gres=gpumem:90g`; venv
  `.venv/bin/python`; training `DS_ACCELERATOR=cuda`, inference `DS_ACCELERATOR=cpu`.
- Scratch auto-purges files ~15 days (atime); a purge mid-project wiped the venv → rebuilt from
  `RECOVERED_requirements.txt`. A later venv rename broke `torchrun` shebangs (exit 126) → fixed.
- Disk: ~2.5 TB quota; checkpoints ~100 GB each (DeepSpeed `global_step*` optimizer state dominates). Optimizer
  state was pruned to free space — every `model.safetensors` (the LoRA weights needed for eval/warm-start) is kept.
- These did not affect the scientific results (only caused reruns); listed so the timeline/checkpoint gaps are
  explained.
