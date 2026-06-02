# TODO — Post GH200 Training Bringup

Items identified from the 2026-04-05 commit review after getting training running on GH200.

## Done — sim closed-loop validation (2026-06-02)

Resolved while validating the `franka_orca_bimanual` Isaac eval (details:
[`SIM_VALIDATION_AND_SCENE.md`](SIM_VALIDATION_AND_SCENE.md)):
- ✅ Sim joint setup matches training: action-term order `[L_arm,R_arm,L_hand,R_hand]`,
  `use_default_offset=False` (server returns absolute targets), hand joints gathered by name in
  training order (Isaac reorders the branched hand). Image resolution confirmed (client 640×480 →
  server resizes to 320×176).
- ✅ Scene objects added (`bag_groceries` stand-ins); physics fidelity (fingertip inertia + hand
  gains); table collision (was a visual-only ghost → objects sank through); camera framing.
- ✅ Full closed-loop e2e re-validated on Blackwell (`euler/run_e2e.sbatch`).

## Must Fix

### Re-enable W&B logging
- **File:** `scripts/train/franka_orca_training.sh`
- `report_to=none` was set to get training running without W&B configured. Change back to `report_to=wandb`.
- Training metrics are not being tracked right now.

### SDPA fallback ignores variable-length sequences
- **File:** `attention.py`
- **Commit:** `1670ccc`
- The PyTorch SDPA fallback in `flash_attention()` ignores `q_lens`/`k_lens`, `softmax_scale`, and `window_size`. If batches ever contain padded sequences, attention will be computed over padding tokens producing silently wrong results.
- Options: add proper masking, or at minimum add a runtime warning when `q_lens`/`k_lens` indicate variable lengths.
- Note: the existing TensorRT SDPA path has the same limitation.

## Should Fix

### Reconcile `max_state_dim` between config and launch script
- **Config:** `franka_orca_relative.yaml` has `max_state_dim: 64`
- **Launch script:** `franka_orca_training.sh` overrides with `max_state_dim=48` (the override wins,
  so the effective value is already correct — this is cosmetic; update the YAML default to 48 for clarity).

### Deduplicate `filter_shape_mismatches`
- Same shape-mismatch filter exists in two places:
  - `groot/vla/model/dreamzero/base_vla.py` (from_pretrained path)
  - `groot/vla/experiment/base.py` (checkpoint resume path)
- Extract into a shared utility.

## Nice to Have

### Scale sequence length back up (revisit only if needed)
- `franka_orca_training.sh` uses `num_frames=33`, `action_horizon=24`, `num_action_per_block=24`.
- This is now the **validated** configuration the checkpoint + closed-loop eval were verified with
  (not a temporary memory hack). Only revisit if you intend to retrain at 49/48 on more GPUs.

### Install Flash Attention on GH200 (GH200-only)
- On **GH200** training uses `ATTENTION_BACKEND=torch` (SDPA) because FA2/3 lacks ARM+Hopper binaries.
- Note: the validated **Euler Blackwell** inference path also runs torch-SDPA *by design*
  (`ENABLE_TENSORRT=True ATTENTION_BACKEND=torch`; flash-attn kernels aren't sm_120-safe there), so
  this is not a blocker for the closed-loop eval — only a GH200 training-throughput nicety.
