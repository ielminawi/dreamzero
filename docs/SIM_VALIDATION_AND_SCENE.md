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

Remaining gap (not fixed — needs data we don't have):
- The sim camera **pose is hand‑tuned**, not derived from calibration: `_OAKD_EYE = _ARIA_EYE =
  (-0.25, 0, 0.45)`, `_WORKSPACE_TARGET = (0.35, 0, 0.15)`. `configs/franka_orca_calibration.json`
  has camera→base **extrinsics** only (no intrinsics), and the cfg doesn't use them. The sim
  perspective is more level than the training top‑down view.
- Background is the Isaac grid/void vs the training room (brick wall, floor).

To fully close it: derive each camera's world pose from the calibration extrinsics (compose with
the arm‑base world transforms; mind the optical‑frame convention), obtain the real oak‑d / Aria
intrinsics to set the exact HFOV, and add a simple room backdrop.

## Validation tooling (reusable)

- `eval_utils/inspect_joints.py` + `euler/inspect_joints.sbatch` — builds the env in Isaac and
  reports: live action path, action-term order, per-term offset/scale, articulation joint order
  vs training, a round-trip "command current state" drift test (arm should hold ≈0; hand should
  track), object resting check, and saves a sim camera frame to `output/sim/sim_oakd_frame.png`.
  Writes results to `output/sim/joint_inspection.txt` (Isaac captures stdout, so it writes to a
  bound file).
- `euler/run_e2e.sbatch` — full closed-loop e2e on a Blackwell RTX PRO 6000
  (`--gpus=nvidia_rtx_pro_6000:1`). Wraps `euler/run_e2e.sh`.

### Reproduce

```bash
cd /cluster/scratch/rjiang/dreamzero
sbatch euler/inspect_joints.sbatch        # quick joint/scene/fidelity check (any 24GB+ GPU)
sbatch euler/run_e2e.sbatch               # full closed-loop rollout (Blackwell)
# results: output/sim/joint_inspection.txt, output/sim/<date>/<time>/episode_0.mp4
```
