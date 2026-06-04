"""Log a sim rollout (video + metadata) to wandb (offline by default; sync later)."""
from __future__ import annotations
import argparse, os, glob
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--video", default="")
    ap.add_argument("--instruction", default=os.environ.get("DZ_INSTRUCTION", ""))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "dreamzero"))
    args = ap.parse_args()

    video = args.video
    if not video:
        cands = sorted(glob.glob("/cluster/scratch/rjiang/dreamzero/output/sim/**/*.mp4", recursive=True),
                       key=os.path.getmtime)
        video = cands[-1] if cands else ""
    step = "".join(c for c in Path(args.ckpt).name if c.isdigit()) or "0"

    import wandb
    os.environ.setdefault("WANDB_MODE", "offline")
    run = wandb.init(project=args.project, group="rollouts", name=f"rollout-{Path(args.ckpt).name}",
                     config={"checkpoint": args.ckpt, "instruction": args.instruction})
    log = {"checkpoint_step": int(step)}
    if video and os.path.exists(video):
        log["rollout_video"] = wandb.Video(video, fps=10, format="mp4")
        print(f"[wandb] logging rollout video {video}")
    else:
        print(f"[wandb] no video found (ckpt={args.ckpt}); logging metadata only")
    wandb.log(log)
    wandb.finish()


if __name__ == "__main__":
    main()
