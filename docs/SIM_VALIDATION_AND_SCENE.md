# Franka + Orca sim ↔ training validation, scene objects, and fidelity fixes

This documents the debugging done to make the Isaac Sim `franka_orca_bimanual` closed-loop
eval faithfully match the conditions the DreamZero policy was trained on, the manipulable
objects added to the scene, and the physics-fidelity fixes. It also documents the reusable
validation tooling.

Embodiment: `franka_orca_bimanual` (dual Franka arm + Orca hand, 48-dim).
Training data: `bag_groceries` (bimanual) under `/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/`.

---

## TL;DR — what was wrong and what changed

| Area | Status | Change |
|---|---|---|
| Image resolution | ✅ already correct | client sends 640×480 (dataset res); server resizes to 176×320 internally. **Do not** set the client to 320×176 (would fail `VideoCrop`). |
| Relative vs absolute actions | ✅ fixed | server already returns **absolute** targets (`GrootSimPolicy.unapply` adds obs state back). Sim must apply them as absolute. |
| Live action path | ✅ identified | `ManagerBasedEnv` ActionManager (cfg `ActionsCfg`) — `env._apply_action` is dead code. |
| Action term order | ✅ fixed | was `[L_arm, L_hand, R_arm, R_hand]` → now `[L_arm, R_arm, L_hand, R_hand]` to match the server. |
| `use_default_offset` | ✅ fixed | was `True` (added home pose) → `False`; absolute targets applied directly. |
| Hand joint order | ✅ fixed | Isaac reorders the branched Orca hand on import; now gathered **by name** in training order (`preserve_order=True` for actions, `find_joints` for obs). |
| Scene objects | ➕ added | bag + container + tube + round item, matching the `bag_groceries` videos. |
| Fingertip inertia | 🔧 fixed | Orca fingertip/root links were mass=0 (PhysX substituted invalid inertia) → small valid mass+inertia. |
| Hand actuator gains | 🔧 fixed | stiffness 2→20, damping 0.1→1.0, effort 1→5 so fingers track commanded grasps. |

Full closed-loop e2e (server + Isaac) was re-validated on the **Blackwell RTX PRO 6000** (exit 0,
1500-step rollout, coherent arms). See `euler/run_e2e.sbatch`.

---

## The joint-setup contract (sim must match training)

- **48-dim layout**: `[left_arm(7), right_arm(7), left_hand(17), right_hand(17)]` → indices
  `0:7, 7:14, 14:31, 31:48`.
- **Orca hand order (17)**: `wrist, thumb_{mcp,abd,pip,dip}, index_{abd,mcp,pip},
  middle_{abd,mcp,pip}, ring_{abd,mcp,pip}, pinky_{abd,mcp,pip}` (matches the h5 `qpos_hand_*`
  and the URDF). **IsaacLab/PhysX reorders these** (groups by tree depth) — so we must address
  joints by name, never by raw column index.
- **Actions are ABSOLUTE joint-position targets** at inference. Training used relative deltas,
  but `groot/.../n1_5/sim_policy.py::unapply` reconstructs absolute targets by adding the
  chunk-start observation state (because the checkpoint has `relative_action: true`).
- **Image path**: LeRobot videos are stored at 640×480; the groot transform crops 0.95 then
  resizes to 176×320 (W×H = 320×176). The client must send 640×480; the server does the resize.

## Files changed

- `sim_envs/franka_orca_bimanual_cfg.py`
  - `ActionsCfg`: term order `[left_arm, right_arm, left_hand, right_hand]`; each term uses an
    explicit ordered `joint_names` list + `preserve_order=True`, `use_default_offset=False`,
    `scale=1.0`.
  - Observation getters gather joints **by name** in canonical training order (`_joint_cols`).
  - Added manipulable objects (`grocery_bag`, `white_container`, `blue_tube`, `round_item`).
  - Hand actuators: `stiffness=20, damping=1.0, effort_limit=5.0` (was `2/0.1/1`).
- `sim_envs/franka_orca_bimanual_env.py`
  - `_relative_actions=False` (note: the live path is the ManagerBasedEnv managers; the
    DirectRLEnv-style methods here are vestigial — documented in the class docstring).
- `sim_envs/assets/franka_orca_{left,right}.urdf`
  - Fingertip + hand-root links given a small valid mass (0.01 kg) + inertia (1e-6).
- `sim_envs/assets/generate_combined_urdf.py`
  - Added `sanitize_inertials()` so regeneration keeps the inertia fix.

## Scene objects (bag_groceries)

The `bag_groceries` videos consistently show, on a wooden table: a **white paper bag**, a
**white container/box**, a **blue tube**, and a **small round item**. The h5 files record **no
object poses** (objects appear only in the camera images), so the sim objects are primitive
rigid-body stand-ins placed around the camera workspace target (x≈0.35, y∈[−0.2, 0.2], table
top z=0.05). Camera-right = −y_world, so +y appears on the image left (bag on the left, matching
the videos).

Refinement ideas (not yet done): model the bag as an open-top container; tune placement to the
camera projection; swap primitives for real meshes.

### Hand grasping (fixed)

The hand couldn't grip — **no friction was set**, so fingers/objects used PhysX's low default and
anything "grasped" slipped out. Fix: a high-friction material `_GRIP_MAT`
(`RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.2, friction_combine_mode="max")`) on
the objects + table. The `max` combine makes the object's friction govern the finger contact even
though the URDF-imported fingers keep the default material (so no need to set a material on the
articulation). Verified with `eval_utils/grasp_test.py` (+ `euler/grasp_test.sbatch`): the hand
closes on the sphere, holds it under gravity, and **lifts it** (object rises ~18 cm with the hand,
staying ~1.7 cm from the hand) → `GRASP WORKS`. (Fingertip links still lack collision — power grasps
work via the phalanges; add fingertip colliders if precise pinch grasps are needed.)

### Table collision (fixed)

The table `CuboidCfg` originally had **no `collision_props`** — it was a visual-only ghost, so
objects (and the hands) fell straight through it onto the ground plane ("bodies sinking into the
table"). Fix: added `collision_props=CollisionPropertiesCfg()`. But a collidable table at its
original position (`x∈[-0.1,1.1]`) **overlapped the Franka base spheres** at x=0 (base r=0.06 at
z=0 spans z∈[−0.06,0.06] ⟂ table top z=0.05) → PhysX **hung** on the deep interpenetration. So the
table was also **shifted forward** to `pos=(0.6,0,0.025), size=(0.9,0.8,0.05)` (back edge x=0.15),
clear of the arm bases. Validated: all four objects rest ON the table (z = top + half-extent), arms
stable, no hang. Trade-off: a thin strip of Isaac ground grid shows behind the table (the arms are
mounted at z=0, so the table can't extend back under them without re-mounting the arms higher —
a follow-up if a seamless table is wanted).

## Camera framing (partially improved) + remaining domain gap

Comparing rendered sim `oakd_front_view` frames to the training video
(`output/sim_vs_train_oakd*.png`) showed the original sim view was framed **much wider** than the
real camera (it included the Isaac ground grid and the two Franka base spheres; objects were
tiny). **Fixed (partially):** raised `focal_length` 15→**20** for both cameras (HFOV ~70°→~55°),
tuned by matching the rendered frame to the training frame. Now the workspace + the grocery
objects fill the frame, both arms stay in view, and the floor grid / base links are cropped. (24
over‑zoomed to a single hand; 20 is the balance.)

**Camera viewpoints (matched to the training frames):**
- **oak-d** (front external): the training view looks **down** at the table (~50°, object tops
  visible). Gave oak-d its own top-down pose `_OAKD_EYE=(-0.05,0,0.60) → _OAKD_TARGET=(0.40,0,0.06)`
  and `focal_length=20` (~55° HFOV). Validated vs `output/sim_vs_train_oakd*.png`.
- **aria** (egocentric Aria glasses): wide FOV — `focal_length=8` (~105°) so the whole table + both
  hands fill the view like `output/bg_aria_ref.png` (the shared focal=20 was far too narrow).
- **Backdrop:** visual-only warm floor + beige back wall replace the black-grid void (no
  `collision_props` so they can't hang PhysX on the arm bases).

**Why not use the calibration extrinsics directly?** `configs/franka_orca_calibration.json` has
camera→base extrinsics (`left_cam`/`right_cam`) but **no intrinsics**, and deriving world poses from
them is inconclusive: under the obvious conventions the cameras end up looking at the *sides*
(y≈−0.55 / +0.69), not the centered workspace the training frames show — the base-reference frame
and the camera↔sim mapping are ambiguous (the Aria stereo "left/right_cam" likely aren't the
sim RGB/oak-d cameras). So hand-tuning to the observed training frames is the reliable approach.

## Pre-retrain checklist (before retraining with background removed)

Plan: remove the background from the training videos in preprocessing + retrain so the policy is
background-invariant. Things to get right *before* spending the retrain (the LeRobot dataset is not
on disk, so conversion will be re-run — apply all of this there). Dataset = `bag_groceries`,
**300 episodes**, both cameras present, T≈1.6k–5.4k.

1. **Background removal must be applied IDENTICALLY at sim inference.** Whatever you mask/replace
   the background with in training (e.g. fill with a constant black/gray), the sim must feed the
   policy the *same* thing — either render the sim with that constant background, or mask the sim
   camera frames before sending to the server. Otherwise the train↔inference gap just moves from
   "real room vs sim room" to "masked vs unmasked." Decide the foreground/background **boundary**
   (does "foreground" include the table, or only arms+objects?) and apply it the same way in both.
   Note: sim segmentation is *perfect* (renderer ground-truth masks) while real segmentation has
   artifacts — consider matching that (feather/�උdegrade sim masks) so mask quality isn't itself OOD.

2. **Normalize the arm joint convention in preprocessing (removes the inference remap).** The data's
   `joint4` is ~+0.9 (not the standard Franka `[-3.07,-0.07]`) and `joint6` is offset; inference
   currently remaps `j4-=π, j6+=π` (`real_arm_to_sim`/`sim_arm_to_real`, verified). Cleaner: convert
   the recorded arm joints to the sim/standard Franka convention *in preprocessing* so data, sim
   state, and policy all share ONE convention (no inference remap, cleaner proprioception).
   **Watch the outliers:** ~4 episodes have `j4` dipping to −0.65…−0.70 and several have low-positive
   mins — these don't fit a clean ±π offset (would push `j4` past the Franka limit), so verify the
   per-joint transform across the dataset and **unwrap/clip to the physical Franka range** rather
   than assuming a constant offset.

3. **Foreground appearance gap remains.** Removing the background fixes the *biggest* visual
   difference, but the foreground is still OOD: sim Franka+Orca render (white/gray arms, gray hands)
   vs real arms/black Orca hands, and **primitive box/sphere objects vs real textured groceries**.
   Background removal helps a lot but won't fully close it — set expectations, and consider whether
   to also standardize the foreground or improve object/arm appearance.

4. **Keep the action/camera pipeline consistent on the re-conversion:**
   - `convert_h5_to_lerobot.py --target-resolution 640x480` (cameras must share one resolution for
     `VideoCrop`).
   - `convert_lerobot_to_gear.py --action-horizon 24` (match training `action_horizon`; it sets the
     relative-stats window) — regenerate `relative_stats_dreamzero.json` with horizon 24.
   - **Camera order** `[aria_rgb_cam, oakd_front_view]` must match between training and inference
     (the README warns view-order misalignment kills transfer); the 2-view → 2×2 grid (2 black
     quadrants) in `DreamTransform` is fine as long as train==eval.
   - Reconcile `max_state_dim` (YAML default 64 vs launch-script override 48 → set 48).

5. **Idle-frame filtering / data hygiene.** Episodes are long (40–110 s); the DROID pipeline filters
   idle frames but `convert_h5_to_lerobot.py` keeps all frames — idle start/end segments dilute
   training. Consider idle filtering. Also exclude `object_in_bowl` (mono-arm, different embodiment).

6. **Camera viewpoint/FOV** still matters even with background removed (it sets where arms/objects
   land in-frame): sim uses top-down oak-d + wide aria approximations; the real intrinsics aren't in
   the calibration file. If you can get the real oak-d/Aria intrinsics+extrinsics, set them exactly.

## Status & the remaining gap (why the policy still doesn't do the task)

After all of the above, a logged closed-loop e2e shows the policy now gets **in-distribution
proprioception** (`state nonzero=48/48`, arm convention remapped; state range ~[−1.2,+1.7] vs the
pre-remap −3.07) and an **in-distribution viewpoint** (top-down oak-d), and the **arms reach toward
the objects** — but it still **drifts** and doesn't complete the bagging task.

The remaining wall is the **appearance domain gap**, which is asset-limited:
- Objects are primitive boxes/cylinders/sphere vs a real **textured paper bag + grocery items**.
- The table is a flat color vs **wood**; the room is plain beige vs the real **brick** room.
- No real camera **intrinsics**, so FOV/poses are best-effort approximations.

A video-world-model trained on real RGB will not zero-shot transfer into an abstract sim no matter
how good the *geometry* is. To actually make the policy perform, you need one of:
1. **Real assets / a digital twin** — USD meshes + textures for the bag and grocery items, the real
   room, and the real oak-d/Aria intrinsics + a clear extrinsics→world mapping.
2. **Policy adaptation** — fine-tune (or LoRA) the policy on sim renders, or train with domain
   randomization, so it tolerates the sim appearance.

Everything mechanical (joint setup, conventions, collisions, fidelity) is fixed and verified — those
were the in-scope, data-grounded bugs. The appearance gap is a separate, resource-dependent effort.

## Policy closed-loop debugging (2026-06-03) — why the policy doesn't do the task

After the mechanics were validated, the closed-loop policy still produced slow/aimless motion. The
investigation (open-loop replay + 2 analysis agents + server action logging + a logged e2e):

**Mechanics are correct — the failure is the policy seeing out-of-distribution input.**
- **Open-loop replay** of the recorded `bag_groceries` actions (`eval_utils/replay_h5.py`)
  reproduces the demonstrated *hand* motion; the hand tracks commanded joints to **<0.1 rad**.
- **`left_wrist` was inverted** (the URDF mirror negated its axis: `−1 0 0`) → **fixed** to `1 0 0`
  in the left URDF + `generate_combined_urdf.py` (axis‑x negation removed; it only affected the
  wrist since finger axes are in the Y/Z plane). **Hands confirmed NOT twisted** — fingers curl
  inward to grasp and left/right are mirror‑symmetric (`eval_utils/show_hands.py` →
  `output/hands_zoom.png`).
- **Server action logging** (added in `socket_test_optimized_AR.py` `infer`, prints `[ACTION]` to
  `euler/logs/server.log`): policy receives `state nonzero=48/48`, sane magnitudes, but per‑chunk
  arm motion is tiny (`horizon_spread` ~0.03–0.06) with a joint pinned at a limit and the left hand
  clenched → **out‑of‑distribution behavior**, not a code/scale bug.

### ⚠️ Open critical issue: arm joint-convention mismatch (sim ≠ training)

The training/real `qpos_arm` uses a **different Franka joint convention** than the sim URDF:

| joint | real/training range | sim Franka limit | inferred map `sim = f(real)` |
|---|---|---|---|
| j1,j2,j3,j5,j7 | within limits | wide | identity |
| **j4** | +0.81…+1.00 | [−3.07, −0.07] | **sim = real − π** |
| **j6** | −0.43…+0.14 | [−0.02, +3.75] | **sim = real + π** |

So the sim hands the policy proprioceptive **state in the wrong convention** (the policy never saw
`joint4` negative in training) **and** can't reach the demo arm poses (`joint4` clamps). This is
likely a bigger driver of the failure than the visual gap. The ±π offsets on j4/j6 are a known
Franka URDF/DH convention difference and are self‑consistent across 14k frames + the sim home pose
(home j4=−2.81↔real +0.33, j6=3.04≈π↔real −0.10, next to the demo distribution). Inferred from data
ranges; **must be verified by replaying the remapped actions** (arms should reach demo poses, not
clamp).

**Fix (apply the remap):** send the policy `state = f⁻¹(sim_qpos)` (j4 += π, j6 −= π) and apply the
returned action via `sim = f(action)` (j4 −= π, j6 += π) for the arm joints — in the eval client
(`run_sim_eval_bimanual.py`) or the env. Then re-run the e2e.

## Validation tooling (reusable)

- `eval_utils/inspect_joints.py` + `euler/inspect_joints.sbatch` — builds the env in Isaac and
  reports: live action path, action-term order, per-term offset/scale, articulation joint order
  vs training, a round-trip "command current state" drift test (arm should hold ≈0; hand should
  track), object resting check, and saves a sim camera frame to `output/sim/sim_oakd_frame.png`.
  Writes results to `output/sim/joint_inspection.txt` (Isaac captures stdout, so it writes to a
  bound file).
- `eval_utils/replay_h5.py` + `euler/replay_h5.sbatch` — open-loop replay of recorded h5 actions in
  the sim (no policy server). Logs commanded-vs-achieved joint tracking + writes a rollout video.
  The definitive mechanics test (isolates joints/fingers from the policy + domain gap).
- `eval_utils/show_hands.py` + `euler/show_hands.sbatch` — render the hands open vs grasp from the
  home pose to visually confirm fingers/wrist aren't twisted (`output/hands_zoom.png`).
- `euler/run_e2e.sbatch` — full closed-loop e2e on a Blackwell RTX PRO 6000
  (`--gpus=nvidia_rtx_pro_6000:1`). Wraps `euler/run_e2e.sh`. The server prints `[ACTION]` policy
  output stats to `euler/logs/server.log` (see them with `grep '\[ACTION\]'`).

### Reproduce

```bash
cd /cluster/scratch/rjiang/dreamzero
sbatch euler/inspect_joints.sbatch        # quick joint/scene/fidelity check (any 24GB+ GPU)
sbatch euler/run_e2e.sbatch               # full closed-loop rollout (Blackwell)
# results: output/sim/joint_inspection.txt, output/sim/<date>/<time>/episode_0.mp4
```
