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
| meaning | x | y | z | qw | qx | qy | qz |

- `[0:3]` = **position of the Franka flange (`panda_link8`) in that arm's BASE frame**, meters.
  Base axes are world-aligned (identity base rotation); +X points forward into the workspace.
- `[3:7]` = **orientation quaternion, scalar-FIRST (w, x, y, z)**. Column 3 (the first
  quaternion element) is always near 1.0 — the textbook signature of a scalar-first unit
  quaternion near identity. The flange orientation is then:
  ```
  R_flange = A @ R_wxyz,   A = 135° about (-1,1,1)/sqrt3  (wxyz: 0.38268,-0.53340,0.53340,0.53340)
  ```
  i.e. read the quaternion scalar-first and **pre-multiply** the constant world-frame
  correction `A`.
  - CRITICAL (corrected 2026-06-11): an earlier pass mis-read this as scalar-LAST (xyzw,
    absolute). That is a ~142° wrist error that twists both hands inward and makes them
    interpenetrate at convergence frames (e.g. t=1960). It is reachable by IK (so it hides in
    the residual) but it is WRONG. The `A @ R_wxyz` reading is the only one that keeps the
    hands non-colliding (min hand-hand 229 mm vs 8.6 mm) AND pointing down at the table (100%
    of grasp frames vs 39%).

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
   convention (position in base frame; quaternion scalar-first w,x,y,z with `R_flange = A @ R_wxyz`;
   flange = panda_link8), or the policy sees a different distribution than it trained on.
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

Per frame, target flange pose = `(pos = field[0:3], R = A @ R_wxyz(field[3:7]))` where the
quaternion is read **scalar-first** and `A = 135° about (-1,1,1)/sqrt3`:

```python
A = Rot.from_quat([-0.53340, 0.53340, 0.53340, 0.38268])   # scipy xyzw of wxyz(0.38268,-0.5334,0.5334,0.5334)
def flange_R(q4):
    w,x,y,z = q4                                # stored scalar-first
    return (A * Rot.from_quat([x,y,z,w])).as_matrix()

def resid(q):
    R, p = fk(q)
    return concat([ p - p_target,
                    rotvec(R @ R_target.T),          # orientation error (axis-angle)
                    1e-3 * (q - Q_HOME) ])           # redundancy resolution (7-DOF null space)
q_t = least_squares(resid, q_prev, bounds=LIMITS).x  # warm-start from previous frame
```

The LEFT arm fits exactly (0 mm, joint speeds ≤2.1 rad/s). The RIGHT arm is reachable too
(verified: any frame solves to 0 mm from fresh seeds), but the warm-started solver gets
trapped in a high-residual branch through one segment (frames ~1670–1905); the export adds a
small continuity-preserving re-seed but that segment can still be slightly off — see
"Still open" below. Output npz keys: `ik_left (T,7)`, `ik_right (T,7)`, `pos_err_*`, `ori_err_*`.

To regenerate the full-episode joint trajectory:
```
~/.venvs/dz_h5/bin/python eval_utils/export_xyzw_full.py --out ik_Awxyz_full.npz
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

2. **Left Gavin-mount orientation.** Final committed values:
   ```
   DEFAULT_MOUNT_XYZ["left"] = "0.006220897 -0.03045047 0.08067889"   # right xyz, y negated
   DEFAULT_MOUNT_RPY["left"] = "1.832292 0.0 3.141593"                # original hand-tuned value
   DEFAULT_MOUNT_YAW["left"] = "-1.5707963"
   ```
   Right arm: `xyz "0.006220897 0.03045047 0.08067889"`, `rpy "1.737739 0.047154 -0.089270"`,
   `yaw "1.5707963"`. Under the CORRECT orientation this rpy points the left hand down 100% of
   the time and is the best available mirror of the right hand (~22° approach-axis error, the
   residual being genuine task asymmetry). NOTE: a mid-session "option Y" value
   (`"1.403854 -0.047154 3.052323"`) was tuned to fix the left hand's apparent *closing*
   direction, but that symptom was caused by the xyzw orientation misread, not the mount — with
   the orientation fixed, option Y is wrong (it double-flips the already-mirrored chirality hand).

Also baked in earlier: the -45° flange yaw and the 90° Gavin bracket; j6 lower limit -1.0.

## Tooling map (eval_utils/, run via docker/scripts/run_*.sh in the dreamzero/isaac-sim:4.5.0 image)

- `export_xyzw_full.py` — full-episode EE-pose -> joints IK (the solver above).
- `replay_h5_ik.py` — Isaac replay from a joints npz (`--joints-npz`) or online diff-IK;
  emits real|sim videos (aria / oak-d / side-by-side / perspective) + EE-error curves.
- `project_ee_pose_check.py` — host-side: project recorded EE position into the calibrated
  cameras (no Isaac), sanity-check it lands on the hand.
- `render_left_hand_closeup.py` — 8 cameras auto-aimed at `{side}_palm` for hand inspection.
- `export_orientation_hypotheses.py` / `render_orientation_hypotheses.py`,
  `export_xyzw_compare.py` / `render_xyzw_compare.py` — the convention-disambiguation harness
  (how wxyz-vs-xyzw-vs-conj and flange-vs-hand were tested).

## Still open / not perfect

- **Right-arm IK tracking, frames ~1670-1905.** Under the correct `A @ R_wxyz` orientation the
  right-arm poses there are reachable (each frame solves to 0 mm from fresh seeds), but the
  warm-started solver gets trapped in a high-residual branch through that contiguous segment.
  The left arm is clean throughout. A proper global/branch-tracking trajectory IK (or a
  slightly refined per-arm A) would close this; the current export keeps continuity and leaves
  that ~5 s segment slightly off. Everything else in the episode is correct.
- The constant `A` was taken as the clean 135° about (-1,1,1)/sqrt3. A per-arm fit of `A` to
  the reachability/down/non-collision constraints might be marginally better (esp. for the
  right arm) and is worth a pass.
- Flange-vs-hand reference point: the pose is treated as the flange (`panda_link8`); the hand
  attaches via the fixed Gavin mount. This reproduces the videos but a residual hand-frame
  offset has not been fully ruled out.
- The on-screen EE error in `replay_h5_ik` videos (~16-20 mm mean) is the one-step physics
  settle from teleporting exact joints, not a convention error (offline IK is 0.02 mm).
- Sim is not yet pixel-faithful to the real videos; camera/mount/scene refinement ongoing.
