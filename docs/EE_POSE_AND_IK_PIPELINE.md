# Franka-Orca dataset decode + IK pipeline (2026-06-11)

This documents what the `franka_orca_bimanual` h5 "arm" fields actually are, the IK needed
to drive the sim/robot from them, and the URDF fixes — so the training/eval pipeline can be
updated. Reference dataset: `20250826_111157.h5`.

## TL;DR — the one thing that changes everything

The 7-dim `observations/qpos_arm_{left,right}` and `actions_arm_{left,right}` fields are
**end-effector POSES, NOT joint angles.** The modality is mislabelled (`*_arm_joint_pos`).

Layout of each 7-vector (per arm):

| idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|-----|---|---|---|---|---|---|---|
| meaning | x | y | z | qx | qy | qz | qw |

- `[0:3]` = **position of the Franka flange (`panda_link8`) in that arm's BASE frame**, meters.
  Base axes are world-aligned (identity base rotation); +X points forward into the workspace.
- `[3:7]` = **orientation quaternion, scalar-LAST (XYZW)**, i.e. `(qx,qy,qz,qw)`. It is the
  absolute flange orientation in the base frame. Unit norm to machine precision.
  - IMPORTANT: it is NOT scalar-first (wxyz), and NOT the conjugate. Reading it as wxyz or
    inverting it gives a physically-unreachable / wrong pose (verified across the episode).

The 17-dim `qpos_hand_{left,right}` ARE real joint angles (radians), applied directly, in this
canonical order: `[wrist, thumb_mcp, thumb_abd, thumb_pip, thumb_dip, index_abd, index_mcp,
index_pip, middle_abd, middle_mcp, middle_pip, ring_abd, ring_mcp, ring_pip, pinky_abd,
pinky_mcp, pinky_pip]`. `actions_hand_*` are byte-identical to `qpos_hand_*`.

Base positions (world): left arm base at `(0, +0.30635, 0)`, right at `(0, -0.30635, 0)`
(`ARM_SEPARATION_Y/2`). The h5 "left" arm is the +Y one. See `sim_envs/franka_orca_bimanual_cfg.py`.

## Consequences for the training / eval pipeline

1. **Driving the robot (actions):** the action `actions_arm_*` is an EE pose. To command the
   articulation you must run **inverse kinematics** (EE pose -> 7 Panda joint targets), then
   send joint targets. You cannot send the 7-vector as joint positions.
2. **Observations:** if the policy consumes `qpos_arm_*`, it is consuming an EE pose. For
   closed-loop serving the obs builder must emit the **FK-computed flange pose** in THIS exact
   convention (position in base frame; quaternion XYZW; flange = panda_link8), or the policy
   sees a different distribution than it trained on.
3. The old per-joint remap `ARM_SIM_FROM_REAL` / `real_arm_to_sim()` in the cfg is a fit of
   joint angles to pose components and is **meaningless** — do not use it. (Left in the cfg
   only because other legacy scripts import the symbol; it is bypassed by the IK path.)

## The IK solver (authoritative implementation)

Host-side, no Isaac needed. Full implementation: `eval_utils/export_xyzw_full.py`
(single frame / hypothesis variants: `export_xyzw_compare.py`, `export_orientation_hypotheses.py`).
It is warm-started least-squares IK against an analytic Franka FK, joint-limit bounded.

Franka Panda FK (standard, matches `sim_envs/assets/generate_combined_urdf.py`):

```python
PANDA = [  # (xyz, rpy) of each joint frame in its parent
    ([0,0,0.333],       [0,0,0]),
    ([0,0,0],           [-pi/2,0,0]),
    ([0,-0.316,0],      [ pi/2,0,0]),
    ([0.0825,0,0],      [ pi/2,0,0]),
    ([-0.0825,0.384,0], [-pi/2,0,0]),
    ([0,0,0],           [ pi/2,0,0]),
    ([0.088,0,0],       [ pi/2,0,0]),
]
FLANGE_T = [0,0,0.107]   # panda_link7 -> panda_link8 (flange)
# flange pose = compose(joint i: parent_tf @ Rz(q_i)) then translate by FLANGE_T

LIMITS = [[-2.8973,2.8973],[-1.7628,1.7628],[-2.8973,2.8973],[-3.0718,-0.0698],
          [-2.8973,2.8973],[-1.0,3.7525],[-2.8973,2.8973]]   # j6 lower widened -0.0175 -> -1.0
Q_HOME = [0.0,-0.569,0.0,-2.81,0.0,3.037,0.741]
```

Per frame, target flange pose = `(pos = field[0:3], R = quat_xyzw(field[3:7]))`. Solve:

```python
def resid(q):
    R, p = fk(q)
    return concat([ p - p_target,
                    rotvec(R @ R_target.T),          # orientation error (axis-angle)
                    1e-3 * (q - Q_HOME) ])           # redundancy resolution (7-DOF null space)
q_t = least_squares(resid, q_prev, bounds=LIMITS).x  # warm-start from previous frame
```

On this episode the fit is exact: **pos residual max 0.02 mm, ori max 0.00°, joint speeds
≤1.7 rad/s** (smooth, within real Panda limits) for both arms. Output npz keys:
`ik_left (T,7)`, `ik_right (T,7)`, `pos_err_*`, `ori_err_*`.

To regenerate the full-episode joint trajectory:
```
~/.venvs/dz_h5/bin/python eval_utils/export_xyzw_full.py --out ik_xyzw_full.npz
```
(Use `--source actions` for the teleop targets instead of measured `qpos`.)

## URDF fixes (needed for correct sim hands)

`sim_envs/assets/generate_combined_urdf.py` builds `franka_orca_{left,right}.urdf`
(24 DOF each: 7 arm + 17 hand). Two left-hand bugs were fixed:

1. **Left hand chirality.** The left hand was previously a hand-rolled mirror of the right
   URDF that reflected frames but did NOT flip joint axes (an invalid reflection) -> thumb on
   the wrong side, erratic fingers. Now the left hand is the proper chirality hand from the
   upstream xacro. Regenerate it (the `xacro` CLI is a pip package, installed in the dz_h5 venv):
   ```
   xacro sim_envs/assets/orcahand_description/models/urdf/orcahand.urdf.xacro \
       prefix:=left_ chirality:=left \
       -o sim_envs/assets/orcahand_left.urdf
   ```
   `generate_combined_urdf.py` parses `sim_envs/assets/orcahand_left.urdf` (`ORCA_LEFT_URDF_PATH`)
   instead of mirroring. That file is checked into the main repo, so regeneration is only needed
   if the upstream hand model changes. The already-baked `sim_envs/assets/franka_orca_left.urdf`
   is committed and self-contained (except for the submodule meshes it references).

2. **Left Gavin-mount orientation.** The flange->hand mount transform for the left arm:
   `xyz = right xyz with y negated`, `rpy = y-mirror of right rpy (roll & yaw negated) THEN
   +180° about the hand-root local Y axis`. The extra 180° is required because the chirality
   left hand's root frame is reflected about the hand's local x; without it the left hand
   closes 180° the wrong way. Final committed values:
   ```
   DEFAULT_MOUNT_XYZ["left"] = "0.006220897 -0.03045047 0.08067889"
   DEFAULT_MOUNT_RPY["left"] = "1.403854 -0.047154 3.052323"
   DEFAULT_MOUNT_YAW["left"] = "-1.5707963"
   ```
   Right arm: `xyz "0.006220897 0.03045047 0.08067889"`, `rpy "1.737739 0.047154 -0.089270"`,
   `yaw "1.5707963"`.

Also baked in earlier: the -45° flange yaw and the 90° Gavin bracket; j6 lower limit -1.0.

## Tooling map (eval_utils/; on Euler everything Isaac runs via apptainer isaac-sim.sif, see euler/*.sbatch)

- `ee_ik.py` — the SHARED FK/IK module (analytic Panda flange FK, `fk_pose7`, warm-started
  bounded least-squares `solve_ik`/`solve_chunk`, dataset-hemisphere `canonical_quat_xyzw`).
  Importable host-side (.venv) and inside isaac-sim.sif (scipy 1.10 bundled). ~25 ms/frame
  warm-started, <0.05 mm residual on recorded data.
- `export_xyzw_full.py` — full-episode EE-pose -> joints IK (standalone; same math).
  NOTE: pass `--h5` explicitly (the default repo-root h5 copy is gone).
- `replay_h5.py --joints-npz <ik npz>` — Isaac GT replay driving the ARMS from the IK'd
  joint trajectory (hands from the h5); logs joint tracking + FK-vs-recorded EE error.
  Launch: `DZ_JOINTS_NPZ=... DZ_MAXSTEPS=... sbatch euler/replay_h5.sbatch`. Without
  `--joints-npz` it falls back to the legacy (known-wrong) poses-as-joints + remap path.
- `run_sim_eval_bimanual.py` — CLOSED-LOOP policy eval with the IK bridge built in:
  obs arm proprioception = FK(sim joints) -> EE pose (dataset convention), and each
  received action chunk's arm EE poses are IK'd to joint targets (warm-started from the
  current sim joints). Dumps a per-episode `episode_*_traj.npz` (ee/q cmd vs meas, IK
  residuals). Launch: `sbatch euler/run_e2e.sbatch` (policy server + Isaac on one node).
- `scripts/data/convert_h5_to_lerobot.py --arm-ee-to-joints [--skip-videos]` /
  `ARM_TO_JOINTS=1 bash scripts/data/preprocess_bag_groceries.sh` — JOINT-SPACE dataset
  conversion: IKs the arm EE-pose blocks of state+action during preprocessing (output
  `data/franka_orca_lerobot_joints`, videos symlinked from the EE dataset). Downstream
  GEAR stats/relative-stats are recomputed from the converted data, so joint actions are
  q99-normalized correctly with no further changes.
- `project_ee_pose_check.py` — host-side: project recorded EE position into the calibrated
  cameras (no Isaac), sanity-check it lands on the hand.
- `render_left_hand_closeup.py` — 8 cameras auto-aimed at `{side}_palm` for hand inspection.
- `export_orientation_hypotheses.py` / `render_orientation_hypotheses.py`,
  `export_xyzw_compare.py` / `render_xyzw_compare.py` — the convention-disambiguation harness
  (how wxyz-vs-xyzw-vs-conj and flange-vs-hand were tested).

## Still open / not perfect

- Flange-vs-hand reference point: the pose is treated as the flange (`panda_link8`); the hand
  attaches via the fixed Gavin mount. This reproduces the videos but a residual hand-frame
  offset has not been fully ruled out.
- The on-screen EE error in `replay_h5_ik` videos (~16-20 mm mean) is the one-step physics
  settle from teleporting exact joints, not a convention error (offline IK is 0.02 mm).
- Sim is not yet pixel-faithful to the real videos; camera/mount/scene refinement ongoing.
