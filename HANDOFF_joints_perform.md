# DreamZero franka_orca JOINT-SPACE policy — make-it-perform handoff
_Last updated: 2026-06-17 ~17:20 CEST. Repo: `/cluster/scratch/rjiang/dreamzero` (branch `dev-rj`)._

## GOAL
Make the joint-space DreamZero VLA actually **perform the bag-groceries task closed-loop** (Isaac sim).
Open-loop action-MAE is only a proxy; the real verdict is closed-loop task behavior.

## BOTTOM LINE (read this first)
The long-standing "trained policy is worse than doing nothing" was investigated exhaustively (multiple
fan-out + consensus panels). **It was never model capacity.** Three real causes were found; two are fixed,
the third (durable retrain) is RUNNING:

1. **Normalization span inflated ×15–42** (robust-quantile defect). FIXED: deglitched dataset + robust
   q025/q975 stats. Over-shoot dropped 2.05× → 1.33×.
2. **Closed-loop integrated a per-chunk ANCHOR OFFSET** (policy predicts `pred[0] != measured state` by a
   ~0.05 rad same-signed offset; the harness applied absolute joints with NO re-anchoring → the offset
   integrated over ~62 chunks → joints ratcheted to their limits → arm thrashed). FIXED at DEPLOY via
   **re-anchoring** (apply predicted *motion* from measured state). VALIDATED: limit-violations went from
   runaway (~150) to flat (~23); arm now reaches sensibly in-workspace (filmstrip).
3. **Within-chunk MOTION shape is under-learned** (loses to stay-still even on high-motion chunks). Cause:
   per-dim action-loss imbalance — a few near-constant large-offset arm dims dominate the velocity-MSE
   (E[tgt²] spans 37×), starving the task-relevant (hand/motion) dims of gradient. DURABLE FIX = per-dim
   **variance-normalized action loss** + a **truly-fresh joint head**. RETRAIN RUNNING (see below).
4. (Secondary) **Sim renders are severe OOD** vs training video (matte primitives/white hands/flat
   backdrop/squished oakd vs real textured scene/dark hands). Fix IN PROGRESS (camera + scene).

## INFRA INCIDENT (2026-06-18 ~08:13 CEST) — scratch purge, RECOVERED
A scratch auto-purge (15-day atime policy, no backup) wiped the `.venv` (62% of files) + the base
uv-python stdlib. Training job 3742969 survived (interpreter in memory). All checkpoints/AgiBot/data
SURVIVED. RECOVERED: reinstalled uv + python 3.11.15, rebuilt `.venv` from `RECOVERED_requirements.txt`
(reconstructed from surviving dist-info; KEEP that file), installed flash_attn wheel + `-e .`, swapped
`.venv_rebuild`→`.venv` (old broken one at `.venv_purged_broken/` — deletable). `touch -h`'d all of
scratch to reset the 15-day clock (buys ONE cycle — re-touch or move env off scratch by ~2026-07-03).
Full recipe: memory `euler-scratch-purge-venv-recovery`. Decisive eval (ckpt-2000) RESUBMITTED as job 3813513.

## WHAT IS RUNNING / STATE (2026-06-17 ~18:40 — updated this session)
- **Durable varnorm retrain**: job `3742969` (+ requeue chain `3742972`, `3742974`), RUNNING on cuda13pr
  (eu-g7-003), step ~76/14000, ~15 s/step actual (the 29 s/it shown includes slow warmup). VERIFIED engaged:
  `skip_action_head_mlps=True dropped 14 robot-MLP keys (fresh joint head)`; `800 keys -> landed 800, 0
  unexpected`; `variance-normalize ON: dim weights mean=1.000 min=0.283 max=2.429 (floor=0.02)`; base staging
  DreamZero-AgiBot (not Wan). action_seg losses differentiate (seg0=2.57 → seg3=0.80). checkpoint-1000 ETA
  ~3.5 h from 18:40 (≈22:00 CEST).
- **AUTO-EVAL WATCHER**: `euler/watch_varnorm_ckpt.sh` running DETACHED (nohup, PID 2090417, log
  `euler/logs/watch_varnorm.log`). Polls for `checkpoint-1000/model.safetensors` (size-stable >1GB) then
  auto-submits `CKPT=...varnorm/checkpoint-1000 DATA=...joints_clean OUT=output/action_eval_varnorm_1000
  sbatch euler/eval_robuststats.sbatch`. Survives session end. (Re-launch if the node reboots.)
- **SIM CAMERA-MATCH: DONE/GOOD.** `output/sim_vs_train_oakd.png` confirms the matched OAK-D view
  (x00_z45_p26, dir=(0.901,-0.05,-0.432), ~26° pitch, focal 9.6/~95°) reproduces the training FRAMING (right
  hand prominent right, object+table left). Remaining gap is APPEARANCE only (white sim hands vs dark, matte
  primitives, grid floor, flat backdrop) — diminishing returns; defer until varnorm is shown to help motion.
  (`sim_vs_train_final.png` right panel is a *topdown diagnostic*, NOT the matched cam — ignore for fidelity.)
- (superseded) original line: job `3742969` step ~63, ~30 s/step. Recipe: warm-start EE backbone
  `dreamzero_franka_orca_lora_r16/checkpoint-6000` + `warmstart_skip_action_head_mlps=true` (FRESH joint
  head) + data `franka_orca_lerobot_joints_clean` + `action_loss_variance_normalize=true`, unfreeze-top-3,
  lora r16, MAX_STEPS 14000, SAVE_STEPS 1000, SAVE_TOTAL_LIMIT 5. (action_loss values are on the
  variance-normalized scale ~1.x — NOT comparable to the old ~0.04–0.1; only the trend matters.)
- **Background watcher `baan7g7kz`**: fires when the first varnorm checkpoint (checkpoint-1000, ~8h) appears
  → trigger the decisive eval.
- **Sim-fidelity agent `a1ead17b`**: PAUSED waiting on its render — **resume it** (SendMessage) to finish.
  It already edited `sim_envs/franka_orca_bimanual_cfg.py` (camera/scene) and made
  `run_sim_eval_bimanual.py::_resize_image` aspect-preserving (no 960×540→640×480 squish). Render job
  3743851 COMPLETED; the sim-vs-train comparison still needs assessing.
- Old runs cancelled/superseded: the plateaued "clean" run (3507285) and chain were cancelled.

## CONFIRMED FINDINGS — do NOT re-investigate
- **Capacity is NOT the bottleneck**: ckpt weights healthy (0 dead/exploded LoRA adapters), conditioning
  demonstrably works (per-sample pred varies with obs), metrics flat across checkpoints (plateau by step 3000).
- **Training is mechanically correct**: loss / normalization / temporal alignment all audited clean.
- **The displacement (motion-shape) metric beats stay**, but absolute & high-motion lose → it's the
  *within-chunk motion* that's under-learned (→ fix #3), not the direction.
- **The data has a real setpoint** (`action[i] != state[i]`, ~0.03 rad) — do NOT "pin pred[0]=state"; the
  re-anchor (apply *motion* from measured state) is the correct deploy handling.
- **The old "clean" run silently inherited the EE-trained head MLPs** (14 shape-compatible keys) — it was
  NOT a fresh head. The varnorm run fixes this (`warmstart_skip_action_head_mlps=true`).

## FIXES APPLIED (code/data — all on disk, branch dev-rj, NOT committed)
- `eval_utils/run_sim_eval_bimanual.py`: joint-space mode (`--joint-actions` / `JOINT_ACTIONS=1`) bypassing
  FK/IK; **RE-ANCHOR** `q_target = q_now + (pred - pred[0])` gated by `DZ_REANCHOR` (default "1"); optional
  `DZ_CLAMP_JOINTS=1` clips to `ee_ik.LIMITS`; aspect-preserving `_resize_image`.
- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`: `action_loss_variance_normalize`,
  `action_loss_dim_variances`, `action_loss_variance_floor` (default OFF; loss multiply at ~L817; helper
  `_get_action_dim_loss_weights` ~L405).
- `groot/vla/experiment/base.py`: `warmstart_skip_action_head_mlps` flag (~L759) — drops EE head MLPs from warm-start.
- `eval_utils/action_eval.py`: added fair metrics — `policy_reanchored_mae`, `policy_disp_mae`,
  `highmotion_policy_mae/stay_mae`, per-joint `offset_consistency`. (Do NOT judge on raw chunk-MAE-vs-stay alone.)
- Datasets (videos symlinked, only stats/parquets changed): `data/franka_orca_lerobot_joints_clean`
  (deglitched parquets + robust q025/q975 relative stats) — USE THIS; also `_robuststats` (robust stats only).
- New scripts: `euler/train_franka_orca_joints_varnorm.sbatch` + `scripts/train/franka_orca_training_joints_varnorm.sh`;
  `euler/e2e_joints_smoke.sbatch` (closed-loop joint, re-anchor); `euler/eval_robuststats.sbatch` (open-loop,
  CKPT/DATA/OUT env-overridable); `euler/make_untrained_joints.sbatch`; `scripts/data/deglitch_joints_dataset.py`;
  `scripts/check_variance_loss_balance.py`.

## RESULTS @ varnorm ckpt-2000/3000 (2026-06-18) — verified by adversarial workflow `wf_fcccc631-72d`
**Open-loop (ckpt-3000, n=120, fair metrics):** still does NOT decisively beat stay-still — reanchored
beats stay on high-motion in only **1/4** groups (right_arm +6%; others −3…−14%). disp_mae 0.03–0.05,
corr 0.99 (but corr is dominated by the static pose baseline, NOT evidence of motion capture).
**Three corrected findings (high-confidence, do NOT relitigate):**
- There is **no constant error floor** — reanchored-error scales ~proportionally with motion (slope 0.8–0.95,
  R²0.93–0.99). Also `reanchored_mae` is NOT offset-free on arms (carries state-vs-action[0] gap); use
  `disp_mae` as the offset-free shape metric.
- **Delta target is ALREADY in use** — the run trains on relative (action−chunk-anchor) targets
  (`relative_action: true`, `lerobot_sharded.py:_convert_to_relative_action`) AND per-dim variance-normalize.
  So action representation is NOT the untried lever. Old TODO "switch to delta" is moot; only a *pure per-step
  velocity* (action[k]−action[k-1]) variant remains untried (low expected upside).
- **The varnorm recipe PLATEAUED by step ~1500–2000** (action_loss flat ~0.13; tail slope 110× shallower).
  Training to 5000–14000 will NOT lower the open-loop floor. (Confirm: ckpt-1000 eval job 3850158 vs ckpt-3000.)
**Closed-loop (ckpt-3000, 2 ep, re-anchor+clamp, job 3846754, exit 0):** big qualitative leap from
thrashing → **stable, coordinated reach+grasp** (EE path 1.9–2.2m, finger range ~1.97rad = active grasps,
0 limit-viol, 0 nan, joint tracking 3.6–4.4° mean). BUT motion is **too large/imprecise** (workspace extent
1.1–1.6m for a tabletop task = over-shoot, consistent with the 2.5× over-shoot open-loop trace), and **task
success is UNCONFIRMED** (npz logs only robot q/ee/hand — NO object poses; filmstrips
`output/sim_filmstrip_varnorm3000_ep{0,1}*.png` show reaching+grasping but no clear pick-lift-place).

## OBJECTIVE TASK-SUCCESS (2026-06-18, ckpt-3000, object-pose logging now in run_sim_eval_bimanual.py)
DONE: added object-pose logging (writes `episode_N_objects.npz` + "[objlog] moved Xcm/max_lift Ycm" proxy;
objects: grocery_bag, white_container, blue_tube, round_item). Re-ran closed-loop (job 3851764). VERDICT:
**TASK NOT PERFORMED — no pick/place.** Nothing lifted (max_lift ≤2.5cm = tipping/slide). grocery_bag SHOVED
~13–14cm in −X (dragged toward robot), never lifted; white_container/blue_tube barely touched (sub-cm);
round_item LAUNCHED 6.5m off the table in ep0 (real physics, no NaN) by the over-shooting arm (static ep1 →
stochastic collision). Failure mode = large coordinated "reach-and-SWEEP" that COLLIDES with the scene rather
than manipulating it — over-shoot/imprecision + acting on matte-primitive renders (severe OOD vs real-video
training). Tested shorter replan DZ_HORIZON=8 (job 3852987): did NOT help — bag-shoving got WORSE (13–14→18–23cm),
round_item still launched the IDENTICAL 672.6cm in ep0, lift still ≤3.3cm. So over-shoot is in the PER-CHUNK
predictions, not the replan cadence; deployment tuning won't fix it. Remaining levers for task success:
in-distribution visual input (sim fidelity / OOD) and prediction-magnitude precision — NOT horizon, NOT more
training (plateaued), NOT delta (already used). NEXT DECISIVE/CHEAP diagnostic before big sim investment:
feed held-out REAL demo video frames to the closed-loop server (in-distribution input) and see if the policy
manipulates better than on matte-primitive sim renders — isolates "OOD is the blocker" from "policy is weak".

## OPEN-LOOP DEEP-DIVE (2026-06-19, NON-SIM, ckpts 1000–5000, PER_EP=24) — the real bottleneck
Enriched `action_eval.py` (per-chunk motion-capture ratio + direction cosine + `all_traces.npz`); sweep
`eval_varnorm_sweep.sbatch` → `output/action_eval_varnorm_{N}_full/`; analysis `eval_utils/openloop_report.py`.
DECISIVE FINDINGS (high-motion top-third subset; hands move MOST in GT so this is not low-motion noise):
- **The policy only learned the RIGHT ARM.** ckpt-5000 high-motion: R_arm capture 0.81, dir_cos 0.42,
  50% of chunks correctly-headed (cos>0.5), improving over training (dir_cos 0.17→0.42, 1000→5000).
- **LEFT ARM + BOTH HANDS are essentially UNLEARNED**: dir_cos ~0.1, **40–47% of chunks move the WRONG
  direction** (cos<0); L_hand captures only **14%** of a large 0.45-rad motion (3% chunks correct). No
  improvement across checkpoints. The "reanch beats stay" wins are HOLLOW — achieved by barely moving
  (under-prediction ≈ stay), not by correct motion.
- **Policy UNDER-predicts** motion magnitude everywhere (capture <1); the lone 2.5× over-shoot trace was an
  outlier. **No left/right swap** (cross-segment dir-agreement diagonal wins). corr=0.99 was a static-pose
  artifact, NOT motion learning — earlier "conditioning works" was too generous (pred varies with obs but
  mostly INCORRECTLY for 3/4 limbs).
- **Plateau confirmed** except the slow right-arm gain. ⇒ Bottleneck is a LEARNING/conditioning failure on
  left-arm + hands (not normalization/anchor/gross-capacity). Hypotheses to chase (non-sim): right-arm-dominant
  task vs hard-to-predict left-arm/hands from single view; hand-DOF normalization/temporal-alignment; per-arm
  conditioning asymmetry. Right arm IS learnable here → the pipeline can learn; the others need investigation.

## ROOT-CAUSE FANOUT+CONSENSUS (2026-06-19, NON-SIM, workflow wf_59c2d280-ecc, 4 lenses+adjudicator)
REFUTED (high-conf, quantified) all four "obvious" causes: (A) observability — left arm is the MOST visible
limb (pixel-corr +0.70 vs right +0.35), not occluded; (B) data — constant-velocity ceiling dir_cos L 0.799 ≈
R 0.807 while policy hits 0.051 (16× MODEL gap, data is clean/learnable, left even smoother); (C) loss-weighting
— explicit segment weights are uniform [1,1,1,1]; the "2.43/1.48/0.52/0.70" are EMERGENT variance-normalize
weights; left arm gets the LARGEST gradient share (54.8%) yet fails ⇒ re-weighting cannot create an L/R
asymmetry; (D) architecture — shared encoder/decoder (no per-segment heads), L_hand pose is NOT a constant
(corr 0.968 w/ gt, matched rank) ⇒ not a marginal-mean/low-rank collapse, hand motion is LOW-freq (representable).
PRIMARY CAUSE (high-conf, I re-verified): **the within-chunk MOTION profile is unlearned for ALL FOUR limbs**
(within-chunk velocity-shape corr L_arm −0.01, R_arm 0.07, L_hand 0.01, R_hand 0.04). The head reproduces the
held anchor POSE (corr(pred0,state) 0.95–1.0) + gross net-direction (right arm only) but no velocity profile.
Driven by an OFFSET-DOMINATED target/loss: offset energy is 65–1000× the within-chunk motion energy, and the
variance-normalize DIM_VAR was computed from the offset-dominated across-sample span (floors all 14 arm dims at
0.02, down-weights hands) so gradient barely touches motion. SECONDARY (separate, left-specific, mechanism
open): net reach-DIRECTION R_arm 0.45 vs L_arm 0.11 (R climbs 0.12→0.46 over training, L flat) — likely
vision/attention coupling to the left workspace; diagnose AFTER the motion fix.
CAVEAT: analysis is on ABSOLUTE denormalized traces; model trains on RELATIVE (anchor-subtracted ≈ step-0
delta). Lens C confirmed across-sample variance ≫ within-chunk motion var even in normalized space, so the
offset-domination holds — but the falsifier (below) settles it.
RECOMMENDED NEXT EXPERIMENT (non-sim, consensus): retrain / LoRA-finetune the action head with (1) a PER-STEP
VELOCITY target (action_t−action_{t-1}) or an explicit within-chunk first-difference motion loss term, AND
(2) recompute action_loss_dim_variances from TRUE within-chunk motion variance + lower floor 0.02→~0.002.
Then re-run eval_utils/openloop_report.py and check within-chunk motion-shape corr lifts >~0.1 (most for hands,
L_hand std-ratio 0.36→~1.0). FALSIFIER: if motion-shape stays <0.05 after the velocity target, it's a
capacity/temporal limit of the shared 24-token head, not target-encoding. NB "delta" overlaps the existing
relative target; the NEW levers are per-step velocity, the DIM_VAR recompute, and the motion-loss term.

## MOTION-LOSS EXPERIMENT RESULT (2026-06-20, NON-SIM) — falsifier HIT
Added a flag-gated within-chunk first-difference motion loss (`action_loss_motion_weight`, code in
wan_flow_matching_action_tf.py ~L847; scripts/train/franka_orca_training_joints_motion.sh + euler/
train_franka_orca_joints_motion.sbatch). Warm-started varnorm ckpt-5000 (kept joint head, 910 keys landed),
MOTION_W=1.0, job 3979078. RESULT @ ckpt-500 (auto-eval euler/autoeval_motion.sh): **NO lift in within-chunk
motion** — dir_cos motion-500 vs varnorm-5000 baseline: L_arm 0.01 vs 0.05, R_arm 0.35 vs 0.38, L_hand 0.10 vs
0.09, R_hand 0.15 vs 0.12 (all within noise). `action_motion_loss` plateaued at ~0.018 from step ~10 (not
descending); action_loss flat ~0.10. ⇒ This HITS the consensus FALSIFIER: minimizing the motion term does NOT
improve open-loop motion-shape, so the bottleneck is NOT target-encoding/gradient-masking but a deeper
CAPACITY / CONDITIONING / temporal-representation limit of the shared 24-token action head. Loss-shape levers
(varnorm, motion-loss) are exhausted. Remaining levers are bigger (action-head capacity/per-segment heads,
stronger vision->action conditioning/attention, temporal resolution, or more/better data) — none cheaply
validated. NB venv-rename had broken torchrun (exit 126) — FIXED all 103 bin/ shebangs (memory
[[euler-scratch-purge-venv-recovery]]). Varnorm train walled at step 5820 (TIMEOUT) and was NOT resumed
(plateaued; motion run supersedes it). Auto-eval still confirming ckpt-1000/1500/2000.

## TO-DO (ordered) — revised after the above
1. **[done] Instrument task success** — object-pose logging added; see verdict above.
2. **Sim-fidelity / OOD**: sim is still matte primitives vs textured training scene; camera matched but scene
   gap is large. The policy acts on OOD inputs → plausibly explains imprecision. (cfg already edited; needs a
   realistic/textured scene.) Real task success needs BOTH motion AND reduced OOD.
3. **Over-shoot/precision**: investigate whether varnorm UP-weighting (dim weights→2.43) over-amplifies motion
   magnitude (hypothesis, unverified); try `DZ_HORIZON=4–8` (shorter replan) for closed-loop precision; maybe
   a magnitude calibration. NOT more training (plateaued), NOT delta (already used).
4. **DISK**: ~1.7 TB of 2.5 TB. varnorm checkpoints ~100GB each. `.venv_purged_broken/` (~9GB, dead) is
   deletable. Superseded plateaued-clean checkpoints deletable (keep latest). (Need explicit user OK to delete.)

## KEY ENV / GOTCHAS (Euler)
- Account `ls_polle`: **HARD 2-GPU cap** (sbatch rejects >2 concurrent). Blackwell `cuda13pr` (96 GB,
  `ATTENTION_BACKEND=torch`) — request `--gres=gpumem:90g` (90g routes to Blackwell instantly; 80g routes to
  busy A100). venv `.venv/bin/python`. Training: `module load cuda/13.0.2`, `DS_ACCELERATOR=cuda`. Inference:
  `DS_ACCELERATOR=cpu` (else deepspeed import crash).
- Host RAM: training needs **256 GB** (16 cpu × 16 G); 192 G OOMs at base load.
- `save_total_limit` **must be ≥ 5** (code assertion). Checkpoints ≈ 100 GB each (DeepSpeed `global_step*`
  optimizer state dominates; `model.safetensors` is only ~2.8 GB). Watch the 2.5 TB quota — a prior run
  crashed on "Disk quota exceeded".
- **Resume**: resubmit the same sbatch → auto-resumes from latest checkpoint in OUTPUT_DIR
  (`get_checkpoint_path`); a top-level `config.json` in OUTPUT_DIR means "done → skip".
- **Eval base-resolution BUG**: checkpoints written with the TMPDIR-staging trick record a dead
  `pretrained_model_path` in `experiment_cfg/conf.yaml` → `load_lora` silently rebuilds the DiT from vanilla
  Wan2.1 instead of AgiBot. The eval sbatches `sed`-patch it to `./checkpoints/DreamZero-AgiBot`. ALWAYS
  verify the eval/server log says `Rebuilding DiT base from training base model: .../DreamZero-AgiBot`
  (NOT "falling back to ... Wan2.1").
- **Closed-loop Isaac**: `--device cpu` for PhysX on Blackwell (Isaac torch is cu118, can't run sm_120;
  RTX render via Vulkan/warp is fine), writable overlay, ETH proxy for asset fetch. `e2e_joints_smoke.sbatch`
  handles env + conf-patch; re-anchor default-on.
- The shell cwd can reset between Bash calls — use ABSOLUTE paths.

## MEMORY FILES (persistent, in ~/.claude/.../memory/)
`dreamzero-joints-closedloop-drift` (this whole closed-loop story + roadmap), `dreamzero-joints-action-eval`
(normalization fix + fair metrics + conf bug), `dreamzero-ik-bridge-eval`, `dreamzero-action-conditioning-fix`,
`euler-dreamzero-training-env`, `dreamzero-oakd-camera-matching` (for the sim camera fix).

## KEY OUTPUTS / EVIDENCE
- Open-loop fair-metric evals: `output/action_eval_clean_{2000,4000,5000}/action_accuracy.json`,
  `output/action_eval_clean_3000_reanchor/action_accuracy.json`.
- Closed-loop RAW (drift) traj+log: `output/sim/2026-06-17/03-06-28/episode_*_traj.npz`, `euler/logs/jsmoke.3650548.log`.
- Closed-loop RE-ANCHORED (fixed) traj+log: `output/sim/2026-06-17/15-30-08/`, `euler/logs/jsmoke.3732549.log`.
- Filmstrips: `output/sim_filmstrip_clean5000.png` (RAW thrashing) vs `output/sim_filmstrip_reanchor5000.png`
  (re-anchored, coherent reaching).
