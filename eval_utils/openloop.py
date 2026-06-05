#!/usr/bin/env python3
"""TRUE open-loop world-model rollout + horizon plot.

Seed the model with ONE real frame, then roll forward autoregressively feeding the
model's OWN decoded predictions back as the next observation (no GT re-grounding).
Produce a plot comparing the open-loop prediction to ground truth at horizons
t = 5, 10, 15, 20, 25 frames, for both camera views (2x2 canvas: TL=aria, BL=oakd).
"""
import argparse, os, sys
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
from validate_wm import (CAMERA_KEYS, CONCAT_ORDER, STATE_KEYS_DIMS, LANG_KEY,
                         read_gt_window, decode_latents, resize_to)

import logging
log = logging.getLogger("openloop")


def seed_obs(frame_by_cam, instruction):
    """Step-0 obs: 1 real seed frame per camera."""
    obs = {ck: frame_by_cam[ck][None] for ck in CAMERA_KEYS}   # (1,H,W,3)
    for k, d in STATE_KEYS_DIMS.items():
        obs[k] = np.zeros((1, d), dtype=np.float64)
    obs[LANG_KEY] = instruction
    return obs


def pred_obs(pred_buf, fpc, instruction):
    """Step-k obs (k>=1): last `fpc` PREDICTED frames per camera (open loop)."""
    obs = {}
    for ck in CAMERA_KEYS:
        buf = pred_buf[ck]
        sel = np.stack(buf[-fpc:]) if len(buf) >= fpc else np.stack(
            buf + [buf[-1]] * (fpc - len(buf)))
        obs[ck] = sel
    for k, d in STATE_KEYS_DIMS.items():
        obs[k] = np.zeros((1, d), dtype=np.float64)
    obs[LANG_KEY] = instruction
    return obs


def split_quads(canvas_frames):
    """(T,2h,2w,3) -> dict aria=TL, oakd=BL  (each (T,h,w,3))."""
    H, W = canvas_frames.shape[1], canvas_frames.shape[2]
    hh, ww = H // 2, W // 2
    return {CONCAT_ORDER[0]: canvas_frames[:, :hh, :ww],
            CONCAT_ORDER[1]: canvas_frames[:, hh:hh * 2, :ww]}, hh, ww


def rollout(policy, Batch, dsdir, ep, start, num_steps, fpc, instruction):
    from tianshou.data import Batch as B
    seed = read_gt_window(dsdir, ep, CAMERA_KEYS, start, 1)
    seed1 = {ck: seed[ck][0] for ck in CAMERA_KEYS}
    pred_buf = {ck: [] for ck in CAMERA_KEYS}
    all_lat, latent_video = [], None
    for step in range(num_steps):
        obs = seed_obs(seed1, instruction) if step == 0 else pred_obs(pred_buf, fpc, instruction)
        with torch.no_grad():
            _, vp = policy.lazy_joint_forward_causal(B(obs=obs), latent_video=latent_video)
        latent_video = vp
        all_lat.append(vp)
        step_pix = decode_latents(policy, [vp])           # decode THIS chunk to feed back
        q, hh, ww = split_quads(step_pix)
        for ck, cam in zip(CAMERA_KEYS, CONCAT_ORDER):
            pred_buf[ck].extend(list(q[cam]))
    full = decode_latents(policy, all_lat)                 # full predicted canvas
    quads, hh, ww = split_quads(full)
    return quads, hh, ww


def best_off(gt, pr, maxoff=6):
    from skimage.metrics import structural_similarity as ssim
    P, G = pr.shape[0], gt.shape[0]
    bo, bs = 0, -1e9
    for o in range(maxoff + 1):
        n = min(P, G - o)
        if n <= 6: continue
        s = np.mean([ssim(gt[o + k], pr[k], data_range=255, channel_axis=2)
                     for k in range(0, min(n, 12))])
        if s > bs: bs, bo = s, o
    return bo


def make_plot(quads, hh, ww, dsdir, ep, start, horizons, out_png, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from skimage.metrics import structural_similarity as ssim
    T = quads[CONCAT_ORDER[0]].shape[0]
    gt = read_gt_window(dsdir, ep, CAMERA_KEYS, start, T + max(horizons) + 8)
    gt_v = {cam: np.stack([resize_to(f, hh, ww) for f in gt[ck]])
            for ck, cam in zip(CAMERA_KEYS, CONCAT_ORDER)}
    off = best_off(gt_v[CONCAT_ORDER[0]], quads[CONCAT_ORDER[0]])

    cams = CONCAT_ORDER
    rows = 2 * len(cams)
    cols = len(horizons)
    fig, ax = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    for ci, t in enumerate(horizons):
        ax[0, ci].set_title(f"t = {t}", fontsize=13)
    for vi, cam in enumerate(cams):
        for ci, t in enumerate(horizons):
            pt = min(t, T - 1)
            gtf = gt_v[cam][min(off + t, gt_v[cam].shape[0] - 1)]
            prf = quads[cam][pt]
            ssv = ssim(gtf, prf, data_range=255, channel_axis=2)
            a_gt = ax[2 * vi, ci]; a_pr = ax[2 * vi + 1, ci]
            a_gt.imshow(gtf); a_pr.imshow(prf)
            a_pr.set_xlabel(f"SSIM {ssv:.2f}", fontsize=9)
            for a in (a_gt, a_pr): a.set_xticks([]); a.set_yticks([])
        ax[2 * vi, 0].set_ylabel(f"{cam}\nGROUND TRUTH", fontsize=10)
        ax[2 * vi + 1, 0].set_ylabel(f"{cam}\nOPEN-LOOP PRED", fontsize=10)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=120)
    log.info(f"wrote {out_png}  (T={T} pred frames, align off={off})")
    return T, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--embodiment", default="franka_orca_bimanual")
    ap.add_argument("--clips", default="32:1600,5:50", help="ep:start,ep:start")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--frames-per-chunk", type=int, default=4)
    ap.add_argument("--horizons", default="5,10,15,20,25")
    ap.add_argument("--instruction", default="bag the groceries")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    os.environ.setdefault("TOKENIZER_DIR", "/cluster/scratch/rjiang/dreamzero/checkpoints/umt5-xxl")
    os.environ.setdefault("ATTENTION_BACKEND", "FA2")
    torch._dynamo.config.recompile_limit = 800
    for k, v in dict(MASTER_ADDR="localhost", MASTER_PORT="12359", RANK="0", WORLD_SIZE="1").items():
        os.environ.setdefault(k, v)
    torch.cuda.set_device(0)
    if not dist.is_initialized(): dist.init_process_group("nccl")

    from torch.distributed.device_mesh import init_device_mesh
    from tianshou.data import Batch
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
    log.info("loading model ...")
    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag(a.embodiment), model_path=a.model_path,
                            device=torch.device("cuda:0"),
                            device_mesh=init_device_mesh("cuda", (1,), mesh_dim_names=("ip",)))
    log.info("model loaded.")

    os.makedirs(a.out_dir, exist_ok=True)
    horizons = [int(x) for x in a.horizons.split(",")]
    import imageio.v2 as iio
    for spec in a.clips.split(","):
        ep, start = (int(x) for x in spec.split(":"))
        log.info(f"open-loop rollout ep{ep} start{start} ...")
        quads, hh, ww = rollout(policy, Batch, a.dataset_dir, ep, start,
                                a.num_steps, a.frames_per_chunk, a.instruction)
        tag = f"ep{ep:06d}_s{start:05d}"
        # save the open-loop predicted videos (per view) + a combined canvas video
        for cam in CONCAT_ORDER:
            iio.mimsave(f"{a.out_dir}/{tag}_{cam}_openloop.mp4", list(quads[cam]), fps=10, codec="libx264")
        make_plot(quads, hh, ww, a.dataset_dir, ep, start, horizons,
                  f"{a.out_dir}/openloop_horizons_{tag}.png",
                  f"Open-loop WM prediction vs GT  |  ckpt-20000  |  {tag}")
    log.info("done.")


if __name__ == "__main__":
    main()
