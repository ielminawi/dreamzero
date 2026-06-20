"""CPU-only sanity check for the per-dim variance-normalized action loss.

Does NOT instantiate the 14B model or touch a GPU. Replicates the loss math from
WANPolicyHead.forward (variance-normalize branch) on a SYNTHETIC batch whose per-dim
target variance matches the real franka_orca_joints_clean imbalance, and asserts:
  (1) the variance-normalized loss is FINITE,
  (2) per-dim loss CONTRIBUTIONS are far more balanced than the un-normalized loss.
Run: .venv/bin/python scripts/check_variance_loss_balance.py
"""
import torch

# Canonical per-dim normalized-target variances (from a few episodes of
# franka_orca_lerobot_joints_clean; see /tmp/canon_var.py in the prep notes).
DIM_VAR = [0.01050,0.01291,0.00387,0.01547,0.01091,0.01233,0.00254,0.02585,0.01791,0.02651,
           0.05795,0.04219,0.05233,0.03839,0.08411,0.10285,0.08838,0.09723,0.10106,0.11494,
           0.07817,0.10771,0.14395,0.08068,0.08038,0.17142,0.06752,0.08373,0.12656,0.07429,
           0.08189,0.07992,0.04505,0.08879,0.08888,0.12674,0.08231,0.04963,0.05409,0.11839,
           0.04505,0.05806,0.10695,0.05264,0.06675,0.12811,0.06797,0.07031]
FLOOR = 0.02

def dim_weights(action_dim):
    var = torch.tensor(DIM_VAR[:action_dim], dtype=torch.float32)
    inv = 1.0 / torch.clamp(var, min=max(FLOOR, 1e-8))
    return inv / inv.mean()  # mean == 1

def main():
    torch.manual_seed(0)
    B, T, D = 256, 24, 48  # large enough that empirical per-dim contribution is stable
    var = torch.tensor(DIM_VAR, dtype=torch.float32)

    # Synthetic flow-matching: clean target ~ per-dim variance, noise unit, pred = noise (so the
    # residual the head must learn is the action-dependent target part), err^2 ~ per-dim target^2.
    target = torch.randn(B, T, D) * var.sqrt()[None, None, :]      # clean (per-dim scaled)
    noise = torch.randn(B, T, D)
    fm_target = noise - target                                     # flow-matching velocity target
    pred = noise                                                   # naive predictor
    sq_err = (pred - fm_target) ** 2                               # = target^2 here  [B,T,D]

    # --- un-normalized per-dim contribution (current behavior) ---
    contrib_raw = sq_err.mean(dim=(0, 1))                          # [D]
    raw_imb = (contrib_raw.max() / contrib_raw.min()).item()

    # --- variance-normalized ---
    w = dim_weights(D)
    assert torch.isfinite(w).all(), "weights not finite"
    assert abs(w.mean().item() - 1.0) < 1e-5, f"weights not mean-1: {w.mean()}"
    sq_err_w = sq_err * w[None, None, :]
    contrib_w = sq_err_w.mean(dim=(0, 1))                          # [D]
    vn_imb = (contrib_w.max() / contrib_w.min()).item()

    loss_raw = sq_err.mean()
    loss_vn = sq_err_w.mean()
    assert torch.isfinite(loss_vn), "variance-normalized loss is not finite"

    print(f"action_dim={D}  floor={FLOOR}")
    print(f"weights: mean={w.mean():.4f} min={w.min():.4f} max={w.max():.4f} (finite={bool(torch.isfinite(w).all())})")
    print(f"per-dim contribution imbalance  RAW = {raw_imb:6.1f}x   VAR-NORM = {vn_imb:6.1f}x")
    print(f"total action loss  RAW = {loss_raw.item():.4f}   VAR-NORM = {loss_vn.item():.4f}  (finite={bool(torch.isfinite(loss_vn))})")

    # Among dims at/above the variance floor, contributions should be ~flat after normalization
    # (floored tiny-motion dims stay intentionally lower -- the floor is the safety tradeoff).
    above = var >= FLOOR
    imb_above_raw = (contrib_raw[above].max() / contrib_raw[above].min()).item()
    imb_above_vn = (contrib_w[above].max() / contrib_w[above].min()).item()
    print(f"among dims with var>=floor ({int(above.sum())} dims): "
          f"imbalance RAW={imb_above_raw:.1f}x -> VAR-NORM={imb_above_vn:.1f}x")

    assert vn_imb < raw_imb, "variance-normalization did not improve overall balance"
    assert imb_above_vn < 1.5, f"motion dims not balanced after norm: {imb_above_vn}x"
    # A near-constant low-variance arm dim (6) must NOT dominate a high-motion dim (25):
    # its weighted contribution stays at/below the balanced motion level.
    assert contrib_w[6] <= contrib_w[25] * 1.1, "near-constant dim 6 still dominates motion dim 25"
    print("\nPASS: variance-normalized action loss is finite; motion dims are balanced "
          "and near-constant dims no longer dominate.")

if __name__ == "__main__":
    main()
