# Results package — for writing the report's Results section

This folder is a self-contained handoff for an agent (or person) writing the **Results section** of the report
on the joint-space DreamZero VLA for the bimanual "bag the groceries" task. Everything needed — the narrative,
the numbers, the figures, and the methods — is here.

## Start here
1. **`RESULTS_SUMMARY.md`** — the findings, with all key numbers and tables. This is the spine of the results section.
2. **`ANALYSIS.md`** — the root-cause diagnosis: the refuted-hypotheses table + the pre-registered falsifier.
   Use for the analysis/discussion paragraphs.
3. **`METHODS.md`** — model, dataset, training recipes, metric definitions, eval pipeline, reproducibility.
   Use for the methods/setup text and metric definitions.

## Figures (`results/figures/`)
- **`fig1_dir_cos_trend.png`** — within-chunk motion direction (`dir_cos`) vs training step, per limb, across
  recipes. THE headline figure: right arm climbs to 0.43; left arm + both hands stay near-random (~0.1).
- **`fig2_capture_direction_bestckpt.png`** — per-limb motion-capture ratio + direction at the best checkpoint;
  shows hands under-predict and move randomly.
- **`fig3_closedloop_task.png`** — closed-loop object displacement vs lift: objects are shoved/knocked, never
  lifted (task not completed).

## Visual artifacts (`results/closed_loop/`)
- `filmstrip_reach_grasp_ep0.png` — closed-loop rollout filmstrip (coordinated reach + grasp, but no pick-place).
- `sim_vs_train_ood.png` — sim (matte primitives) vs real training video; illustrates the visual OOD gap.

## Data tables (`results/data/`)
- **`openloop_metrics.csv`** — per (run, step, segment): stay_mae, disp_mae, reanchored_mae, corr,
  hi_capture_ratio, hi_dir_cos, hi_motion_rad. The primary quantitative source. (`run` ∈ {clean, varnorm,
  motion_v1, motion_v2_handspush}; `segment` ∈ {left/right}_{arm/hand}_joint_pos.)
- **`openloop_metrics.json`** — same, nested by run→step→segment.
- **`closedloop_task.csv`** — per (sim_run, episode, object): moved_cm, max_lift_cm (task-success proxy).

## The one-paragraph result (for an abstract/intro)
We evaluate a joint-space DreamZero VLA on a bimanual bag-groceries task with a non-sim open-loop diagnostic
that measures, per limb, how well the policy predicts within-chunk motion magnitude and direction. The policy
reproduces the held pose and learns the **right arm's reach direction** (direction cosine 0.12→0.43 over
training), but the **left arm and both hands never learn within-chunk motion** (direction cosine ≈0.1,
unchanged across recipes); a conditioning probe shows the **hands essentially ignore the visual input**.
We rule out — quantitatively — observability, data quality, loss re-weighting, action representation, and a
marginal-mean/architecture collapse, and a pre-registered falsifier (an explicit motion loss that is minimized
without improving open-loop motion) localizes the bottleneck to **conditioning/capacity**. Closed-loop, the
policy reaches and grasps stably (after a re-anchoring fix that eliminates drift) but cannot complete the
pick-and-place: objects are shoved (~13–22 cm) and never lifted (<3.5 cm).

## Honest framing guidance
- **Contributions to emphasize:** the diagnostic methodology + the right-arm-only asymmetry finding; the
  rigorous refutation of alternative explanations; the closed-loop stability (re-anchor) fix.
- **Negative results to state plainly:** left-arm/hand motion + task success are not achieved; loss-side fixes
  are exhausted; the bottleneck is conditioning/capacity (future work).
- Do not overclaim closed-loop success — the objective object-pose metric shows 0 successful lifts/places.

## Live status (as of packaging)
Two jobs may still be finishing and will *add* confirmatory data (they do not change the conclusions):
- `motion_v2_handspush` training (more checkpoints 1000/1500) — an end-of-run sweep evaluates them into
  `output/action_eval_motion2_*_full`.
- A fresh closed-loop rollout on the best motion checkpoint (`output/sim/<today>/…`).
To refresh the package after they land: rerun `consolidate_metrics.py` then `make_figures.py`.
All current conclusions hold with the data already in this folder.
```
```
