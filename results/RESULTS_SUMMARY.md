# Results Summary — Joint-space DreamZero VLA on bag-groceries (Franka + Orca hands)

_Self-contained results package. All numbers below are reproducible from `results/data/*.csv` and the
scripts `results/consolidate_metrics.py` / `results/make_figures.py`. Figures in `results/figures/`._

## 1. Task & setup
Goal: make a joint-space DreamZero vision-language-action (VLA) policy **perform the bimanual
"bag the groceries" task** (dual Franka arm + Orca hand, 48-D action = 7+7 arm joints + 17+17 hand joints,
24-step action chunks, two camera views). Evaluation is primarily **open-loop action prediction** on held-out
demo frames (no simulator dependence), complemented by **closed-loop** rollouts in Isaac Sim.

## 2. Headline result (one sentence)
**The policy reliably reproduces the held arm/hand pose and learns the right arm's gross reach direction,
but it does NOT learn the within-chunk motion of the left arm or either hand — the hands effectively ignore
the visual input — so closed-loop it reaches and grasps but cannot complete the pick-and-place.**

## 3. Open-loop metrics (definitions)
For each 24-step chunk we compare the predicted action trajectory to ground truth, per limb-segment:
- **stay_mae** — error of a do-nothing baseline (hold the current state). The bar to beat.
- **disp_mae** — error of the predicted *displacement* shape (offset-free).
- **hi_capture_ratio** — ‖predicted motion‖ / ‖true motion‖ on the **high-motion** chunks (top third). 1.0 = right magnitude, <1 = under-prediction, >1 = over-shoot.
- **hi_dir_cos** — cosine between predicted and true within-chunk motion **direction** on high-motion chunks. 1 = perfect heading, 0 = random.
Data: `results/data/openloop_metrics.csv` (13 checkpoints × 4 segments). The decisive quantity is **hi_dir_cos**.

## 4. Key quantitative findings

### 4.1 Within-chunk motion is learned ONLY for the right arm  (Fig 1)
`hi_dir_cos` vs training step (high-motion chunks):

| segment | varnorm (1k→5k) | motion_v1 (0.5k→1.5k) | motion_v2 hands-push (0.5k) |
|---|---|---|---|
| **right_arm** | 0.12 → **0.38** | 0.35 → **0.43** | 0.39 |
| left_arm | 0.02–0.05 (noisy) | 0.01–0.20 (noisy) | 0.06 |
| left_hand | 0.06–0.12 | 0.07–0.11 | 0.08 |
| right_hand | 0.06–0.15 | 0.13–0.15 | 0.16 |

Only the **right arm** shows a clear, monotone increase with training. The **left arm and both hands stay
near-random** (cos ≈ 0.1) across every recipe and every checkpoint.

### 4.2 The hands under-predict motion and ignore the input  (Fig 2)
At the strongest checkpoint (motion_v1, step 1500), high-motion `capture_ratio` / `dir_cos`:
- right_arm 0.76 / 0.43  · left_arm 0.36 / 0.07 · **left_hand 0.12 / 0.09** · right_hand 0.63 / 0.14.

The **left hand captures only ~12–17% of its (large, ~0.45 rad) true motion** and moves in a near-random
direction. A separate conditioning probe (variance of the predicted motion across different observations ÷
variance of the true motion) gives **0.08 for the left hand** — i.e. its prediction barely changes with the
input: it emits a near-constant grasp regardless of the scene. (right_arm probe ≈ 0.84 = properly conditioned.)

### 4.3 The "right arm learns" is only gross direction, not the velocity profile
Decomposing the right-arm success: NET chunk reach-direction cosine = **0.45**, but the WITHIN-chunk
velocity-shape correlation (first-difference of the trajectory) ≈ **0.07** — and is ≈0 for all four limbs
(L_arm −0.01, R_arm 0.07, L_hand 0.01, R_hand 0.04). So even the right arm only gets the overall direction
right; the fine motion profile is unlearned everywhere.

### 4.4 Two loss-side fixes do not move the hands
- **varnorm** (per-dim variance-normalized action loss): hands flat.
- **motion_v1** (explicit within-chunk first-difference motion loss): arms improve, hands flat.
- **motion_v2** (corrected motion-variance weights + hand-segment loss ×2.5 + 2× motion loss): hands still
  flat (`dir_cos` 0.08 / 0.16 at step 500, indistinguishable from baseline).
The motion-loss term is minimized (≈0.03→0.014) **without** improving open-loop motion — the pre-registered
falsifier (see ANALYSIS.md §3). Conclusion: the hand failure is a **capacity/conditioning limit, not a
loss-weighting problem.**

## 5. Closed-loop (Isaac Sim) — task behavior  (Fig 3, `results/closed_loop/`)
After fixing a closed-loop drift bug (per-chunk anchor-offset integration → re-anchor at deploy; limit
violations dropped from runaway ~150 to 0), the policy runs **stably**: coordinated bimanual reaching
(EE path 1.9–2.2 m), active finger open/close (range ~1.97 rad), **0 limit violations, 0 NaN, no thrashing.**
BUT object-pose logging shows the **task is not completed**:
- grocery_bag **shoved ~13–22 cm**, **never lifted** (max lift ≤ 3.3 cm = sliding/tipping, not a grasp).
- target items (container, tube) barely touched (<1 cm).
- in one episode the over-shooting arm **knocked an object 6.5 m off the table**.
The policy executes a coarse "reach-and-sweep" that collides with the scene rather than manipulating it —
consistent with the open-loop finding (it lacks precise, conditioned hand/left-arm motion). The sim is also
severe **visual OOD** vs the real training video (matte primitives vs textured scene; see
`results/closed_loop/sim_vs_train_ood.png`), which compounds imprecision.
_Data: `results/data/closedloop_task.csv`. A fresh closed-loop run on the best motion checkpoint is included
when complete (see README "live status")._

## 6. What works vs what doesn't (for the report)
**Positive / contributions:**
1. A reproducible **non-sim open-loop diagnostic** (motion-capture + direction metrics) that isolates *which*
   limbs learn motion — revealing a strong, previously-hidden **right-arm-only** asymmetry.
2. The **right arm genuinely learns** reach direction and improves with training (dir_cos 0.12→0.43).
3. Closed-loop **stability fix** (re-anchoring) turning thrashing into coherent reaching.
4. A **rigorous root-cause diagnosis** (two adversarial fan-out+consensus studies) that *refutes* the obvious
   explanations and localizes the failure to conditioning/capacity (ANALYSIS.md).

**Negative / open:**
1. The **left arm and both hands do not learn within-chunk motion**; the hands ignore the visual input.
2. **Loss-side interventions are exhausted** (variance-normalize, motion-loss, hand up-weighting all fail).
3. Closed-loop **task success = 0** (no successful lift/place); behavior is reach-and-shove.

## 7. Recommended framing for the results section
Lead with the **diagnostic methodology + the asymmetric learning finding** (right arm learns, hands ignore
input) as the contribution, supported by Fig 1–2 and the refutation table (ANALYSIS.md §2). Present the
closed-loop (Fig 3) as confirming the open-loop diagnosis at the task level. Frame the hand/conditioning limit
as the identified bottleneck and future work (architectural conditioning capacity), explicitly noting that
representation/normalization/observability/data were ruled out.
