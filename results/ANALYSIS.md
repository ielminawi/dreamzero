# Analysis & Root-Cause Diagnosis

This documents *why* the policy fails to learn within-chunk motion, derived from two adversarial
fan-out + consensus studies (multiple independent agents each trying to *refute* the finding) plus direct
recomputation from the prediction traces. It is the evidentiary backbone for the discussion/analysis.

## 1. The two-level failure (resolving an apparent contradiction)
Open-loop prediction has two distinct quantities that were initially conflated:
- **Pose** (the held configuration over the chunk): learned well for all limbs — across-chunk pose correlation
  pred-vs-gt = 0.86–0.97; `corr(pred_step0, measured_state)` ≈ 0.95–1.0. The head reproduces the held pose.
- **Within-chunk motion** (the velocity profile): essentially **unlearned for all four limbs** —
  velocity-shape correlation ≈ 0 (L_arm −0.01, R_arm 0.07, L_hand 0.01, R_hand 0.04).
The right arm additionally gets the **net reach direction** right (cos 0.45, improving with training); no other
limb does. So "right arm learns" = gross direction only; the detailed motion is unlearned universally.
The high reported `corr` (~0.99) in earlier evals was a **static-pose artifact** (correlation dominated by the
large constant pose), NOT evidence of motion learning.

## 2. Refuted hypotheses (each ruled out quantitatively — do not re-investigate)
| Hypothesis (why hands/left-arm fail) | Verdict | Decisive evidence |
|---|---|---|
| **Observability** — limbs not visible to the cameras | REFUTED (high conf) | Pixel-motion↔joint-speed correlation is **equal-or-higher for the left arm** (+0.70/+0.71) than the right (+0.35/+0.51); left hand shows +53% pixel motion in its high-motion windows; all limbs in-frame, unoccluded in every eval episode. Visibility is symmetric/left-favoring while learning is right-favoring → opposite directions. |
| **Data quality / labels** — targets noisy or mislabeled | REFUTED (high conf) | A trivial constant-velocity baseline reaches `dir_cos` **0.80 for BOTH arms** (and 0.66/0.75 hands) on the identical high-motion chunks, while the policy reaches only 0.05 → a **~16× model gap, not a data gap**. Left arm is the *smoothest* / least jittery segment; no NaN/dead joints; URDF joint limits match the data; no left/right convention defect. |
| **Loss re-weighting** — gradient share starves hands | REFUTED as the asymmetry cause | Explicit segment weights are uniform `[1,1,1,1]`. Under variance-normalize the **left arm receives the LARGEST gradient share (~55%) yet is unlearned**; a per-dim weight cannot manufacture a left/right asymmetry from near-identical L/R target statistics. (Weighting *is* a minor contributor to hand magnitude under-prediction — see §4 — but not the asymmetry.) |
| **Architecture: marginal-mean collapse / per-segment head bug** | REFUTED (high conf) | The action head is a single shared encoder/decoder (no per-segment heads), and the left-hand chunk **pose is NOT a constant** (across-chunk pose corr 0.968, matched effective rank vs GT). Hand motion is *low*-frequency (representable in 24 steps). So it is not a low-rank/marginal-mean collapse or a missing-head wiring bug. |
| **Left/right action swap** | REFUTED | Cross-segment direction-agreement matrix: the diagonal (pred-L↔gt-L, pred-R↔gt-R) is the row-max in every case; cross-arm agreement is ~0/negative. |

## 3. Pre-registered falsifier (and that it fired)
The consensus predicted: *if the within-chunk motion stays near-random after we directly inject gradient on it
(an explicit first-difference motion loss), then the bottleneck is a capacity/conditioning/temporal limit, not
a target-encoding/gradient-masking artifact.* We ran exactly that experiment (motion_v1, motion_v2):
- the added `action_motion_loss` is **minimized** (≈0.030 → 0.014) yet the open-loop `dir_cos` does **not**
  improve for the hands (or left arm). **Falsifier fired** → the failure is a **capacity / conditioning limit**.
- Direct probe corroborates: the left-hand prediction varies only **0.08×** as much as it should across
  different observations — it is **under-conditioned** (ignores the input), not mis-weighted.

## 4. Best current mechanistic picture
1. **Primary:** the action head does not learn to *condition* fine within-chunk motion on the visual input
   for the left arm and hands (the right arm partially does, for gross direction only). The shared 24-token
   chunk head + frozen-backbone-LoRA conditioning appears to lack the capacity/coupling to map vision → precise
   per-limb motion, especially the 34 hand DOF.
2. **Secondary / contributing:** the relative-action target is offset(pose)-dominated (held-pose energy ≫
   within-chunk motion energy), so the plain velocity-MSE gradient is weak on the motion component; and the
   original variance-normalize weights were computed from the offset-dominated spread, down-weighting the
   high-motion hand dims. Correcting these (motion_v2) helps the magnitude balance but does **not** fix the
   conditioning failure → confirms (1) is primary.

## 5. Implication for future work (not yet tried / out of time)
The remaining untried lever targets conditioning capacity, not loss shape: **unfreeze more DiT blocks**
(stronger vision→action coupling) and/or **add action-head/per-limb capacity**, and reduce the visual OOD for
closed-loop. These require (near-)from-scratch retrains and were out of scope for the deadline. Loss-side,
normalization, anchor, representation (delta is already used), observability, and data levers are exhausted/ruled out.
