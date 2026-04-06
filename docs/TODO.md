# TODO — Post GH200 Training Bringup

Items identified from the 2026-04-05 commit review after getting training running on GH200.

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
- **Launch script:** `franka_orca_training.sh` overrides with `max_state_dim=48`
- 48 is correct for the Franka Orca. Update the YAML to match.

### Deduplicate `filter_shape_mismatches`
- Same shape-mismatch filter exists in two places:
  - `groot/vla/model/dreamzero/base_vla.py` (from_pretrained path)
  - `groot/vla/experiment/base.py` (checkpoint resume path)
- Extract into a shared utility.

## Nice to Have

### Scale sequence length back up
- `franka_orca_training.sh` was reduced to `num_frames=33`, `action_horizon=24`, `num_action_per_block=24` (from 49/48/48) to fit in GH200 memory.
- Revisit if moving to multi-GPU or if Flash Attention becomes available on GH200.

### Install Flash Attention on GH200
- Currently using `ATTENTION_BACKEND=torch` (PyTorch SDPA) because Flash Attention 2/3 doesn't have prebuilt binaries for the GH200 ARM+Hopper architecture.
- SDPA is slower and uses more memory than Flash Attention. Try compiling FA from source for GH200 if performance becomes a bottleneck.
