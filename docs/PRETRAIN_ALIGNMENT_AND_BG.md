# Pre-retrain alignment audit + background removal + launch

Scope of this round: pick ONE dataset under `raw_timesynced_h5/`, audit every train↔eval
alignment point, implement background removal (with before/after for review), and stage the
100h training so it launches with one command. Nothing in the heavy path is launched until the
background-removal gallery is reviewed (per request).

## Dataset choice: `bag_groceries` (not `object_in_bowl_processed_50hz`)

`bag_groceries` is **bimanual, 2-camera** and matches the entire existing `franka_orca_bimanual`
pipeline exactly; `object_in_bowl` is **single-arm, 1-camera** and would need a brand-new
embodiment/modality/sim config (i.e. "mixing things up"). Raw schema (300 episodes, T≈1.6k–5.4k):

| h5 key | shape | → LeRobot |
|---|---|---|
| `observations/qpos_arm_left` / `_right` | (T,7) | `observation.state[0:7]` / `[7:14]` |
| `observations/qpos_hand_left` / `_right` | (T,17) | `observation.state[14:31]` / `[31:48]` |
| `actions_arm_left`/`_right`, `actions_hand_left`/`_right` | (T,7)/(T,17) | `action[...]` same layout |
| `observations/images/aria_rgb_cam/color` | (T,480,640,3) | `observation.images.aria_rgb_cam` (video) |
| `observations/images/oakd_front_view/color` | (T,540,960,3) | `observation.images.oakd_front_view` (video) |

> The pre-converted `lerobot_egoverse/*` datasets are a **different embodiment** (`robot_type:
> aria_bimanual`, state/action dim **12**, `dtype: image`, 0 videos, fps 30) and are **not usable**
> for `franka_orca_bimanual` training — confirming preprocessing must be re-run from the raw h5.

## Alignment audit (data ↔ preprocessing ↔ training config ↔ sim inference)

| Item | Raw / Preprocessing | Training config (`franka_orca_relative` + base) | Sim inference | Verdict |
|---|---|---|---|---|
| State/action dim | 7+7+17+17 = **48** | `max_state_dim/max_action_dim=48` (CLI) | server fallback_dim 48 | ✅ match |
| State/action key names + order | `left_arm,right_arm,left_hand,right_hand`_joint_pos | identical `state.*`/`action.*` keys | same order | ✅ match |
| Camera names + order | `[aria_rgb_cam, oakd_front_view]` | `video.aria_rgb_cam, video.oakd_front_view` | `exterior_0←aria, exterior_1←oakd` | ✅ match |
| Raw resolution | aria 640×480, oakd 960×540 | resized in transform | both resized to 640×480 then sent | ✅ match |
| Convert resolution | `--target-resolution 640x480` (both) | `VideoResize` to 320×176 | client resize 640×480 | ✅ (see flag 1) |
| fps | converter `--fps 50` | `fps[franka_orca_bimanual]=50` | sim 50 Hz (`update_period=0.02`) | ✅ match |
| num_frames / action_horizon | n/a | 33 / 24 | open_loop_horizon 24 | ✅ match |
| Relative-action stats horizon | GEAR `--action-horizon 24` | `relative_action_per_horizon=false` → uses `relative_stats_dreamzero.json` | — | ✅ match |
| Embodiment tag | GEAR `franka_orca_bimanual` | `franka_orca_bimanual` | `_embodiment==franka_orca_bimanual` | ✅ match |
| Arm joint convention | raw real convention (kept as-is) | trained in real convention | eval remaps sim↔real (`real_arm_to_sim`) | ✅ consistent (see flag 2) |

### Flags (none are blockers)
1. **Eval declares 180-high, training uses 176-high.** Training passes `image_resolution_height=176`;
   the inference server config declares `image_resolution=(180,320)`. 4 px (~2%) height skew, eval-side
   and pre-existing. The model 2×2-grids/downsamples views, so it's within tolerance — but worth making
   identical eventually (set the server to 176).
2. **Arm joint convention is handled at eval, not in preprocessing.** Data is left in its recorded
   ("real") convention; the eval loop maps sim→real before sending state and real→sim before applying
   actions, so the policy lives entirely in real convention — consistent. The doc's earlier idea of
   normalizing joints *in preprocessing* would ALSO require removing the eval remap (else double remap).
   We did **not** do it; do both-or-neither if you change it later.
3. **oak-d 16:9 → 4:3 squash** happens identically in preprocessing (960×540→640×480) and eval, so it's
   consistent; intrinsics/extrinsics are hand-tuned (no calibration), see SIM_VALIDATION_AND_SCENE.md.
4. **Background removal must be applied identically** in training data and at sim inference — see below.

## Background removal

Goal (pre-retrain checklist item 1): remove the *room* behind the table so the policy is
background-invariant, closing the biggest real↔sim visual gap. The scene is a real Franka+Orca
**robot** tabletop (aria ego = dark Orca hands; oak-d top-down = white Franka arms; bag + objects on a
wood table; room behind). So human-matting / person-seg are inapplicable.

**Method (`bg_removal.py`, shared by preprocessing AND eval):** ADE20K semantic segmentation
(SegFormer-b2, pure PyTorch — fits the existing cu12.9 torch env, no onnxruntime). ADE20K has explicit
`wall/floor/ceiling/window/...` classes → treated as background and filled with constant gray (128);
everything else (table, and the OOD robot arms/objects, which ADE20K mislabels but never as floor/wall)
is kept. Raw seg is noisy on the robot arms, so masks are cleaned at segmentation resolution
(fill-holes, morphological close, drop tiny speckle, dilate, feather). **Component-pruning to the table
was disabled** because it could delete an arm not connected to the detected table blob — manipulation
content must never be removed; residual misclassified floor is the lesser evil.

- **Consistency contract:** the converter applies `BackgroundRemover.apply` to each 640×480 frame
  before encoding; the eval client applies the SAME `BackgroundRemover` (env `DZ_BG_REMOVAL=segformer`,
  `DZ_BG_FILL=128`) to each 640×480 frame before sending. Enable eval bg **iff** the training data was
  converted with bg.
- **Quality:** suppresses walls / obvious room reliably; imperfect on the OOD robot arms and tiled
  floor (partial, sometimes leaves floor patches, never deletes arms after the pruning fix). Review the
  gallery before committing: `output/bg_gallery/contact.png` and `clip_{aria_rgb_cam,oakd_front_view}.mp4`.
- **Switch off** for a baseline: `BG=none bash euler/run_training_pipeline.sh`.

## Environment change

`deepspeed==0.16.5` was installed into `.venv` (it had been deliberately removed for inference;
HF Trainer's ZeRO-2 path needs it). Training has **never run on Euler** before, so the first hours are
also a stack shakedown.

## Launch (after reviewing the gallery)

```bash
# bg removal ON (default):           or baseline:  BG=none bash euler/run_training_pipeline.sh
bash euler/run_training_pipeline.sh
```
Chain (SLURM `afterok` dependencies): `preprocess (array 0-9, 1 GPU each, bg removal)` →
`finalize+GEAR+validation (CPU)` → `100h training (8× A100-80GB, gpupr.120h, DeepSpeed ZeRO-2)`.
Dataset lands at `data/franka_orca_bag_groceries_lerobot`; checkpoints at
`checkpoints/dreamzero_franka_orca_lora_euler` (LoRA-only, `SAVE_STEPS=2000`).
