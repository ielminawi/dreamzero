#!/usr/bin/env python3
"""Open-loop world-model rollouts + horizon comparison plot.

Two modes (both roll the VIDEO open-loop: seed 1 real frame, then feed the model's
own decoded predictions back; never re-ground on GT frames):

  mode "gtsim"  -- World model AS SIMULATOR: at each step inject the RECORDED
                   ground-truth ACTION (normalized) via cond_action, so the video is
                   driven by the real actions. Tests action->video dynamics fidelity.
  mode "dream"  -- Closed-loop policy<->WM: the joint model predicts its OWN action and
                   the video consistent with it, autoregressively (self-rollout).

Produces, per camera, a 3-row plot [GT | gtsim | dream] at horizons t=5,10,15,20,25,
plus the rollout mp4s.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.distributed as dist
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from validate_wm import (CAMERA_KEYS, CONCAT_ORDER, STATE_KEYS_DIMS, LANG_KEY,
                         read_gt_window, decode_latents, resize_to)
from action_norm import load_stats, normalize_action_chunk

import logging
log = logging.getLogger("openloop")


def seed_obs(frame_by_cam, instruction):
    obs = {ck: frame_by_cam[ck][None] for ck in CAMERA_KEYS}
    for k, d in STATE_KEYS_DIMS.items():
        obs[k] = np.zeros((1, d), dtype=np.float64)
    obs[LANG_KEY] = instruction
    return obs


def pred_obs(pred_buf, fpc, instruction, raw_hw=(480, 640)):
    obs = {}
    for ck in CAMERA_KEYS:
        buf = pred_buf[ck]
        sel = buf[-fpc:] if len(buf) >= fpc else buf + [buf[-1]] * (fpc - len(buf))
        obs[ck] = np.stack([resize_to(f, raw_hw[0], raw_hw[1]) for f in sel])
    for k, d in STATE_KEYS_DIMS.items():
        obs[k] = np.zeros((1, d), dtype=np.float64)
    obs[LANG_KEY] = instruction
    return obs


def split_quads(canvas):
    H, W = canvas.shape[1], canvas.shape[2]
    hh, ww = H // 2, W // 2
    return {CONCAT_ORDER[0]: canvas[:, :hh, :ww],
            CONCAT_ORDER[1]: canvas[:, hh:hh * 2, :ww]}, hh, ww


def rollout(policy, Batch, dsdir, ep, start, num_steps, fpc, instruction, mode,
            act=None, st=None, stats=None, horizon=24):
    seed = read_gt_window(dsdir, ep, CAMERA_KEYS, start, 1)
    seed1 = {ck: seed[ck][0] for ck in CAMERA_KEYS}
    pred_buf = {ck: [] for ck in CAMERA_KEYS}
    all_lat, latent_video, produced = [], None, 0
    for step in range(num_steps):
        cond_action = None
        if mode == "gtsim":
            anchor = min(start + produced, len(act) - 1)             # real-time frame index
            chunk = act[anchor:anchor + horizon]
            if len(chunk) < horizon:                                  # pad tail
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], horizon - len(chunk), 0)], 0)
            ca = normalize_action_chunk(chunk, st[anchor], stats)     # (24,48) in [-1,1]
            cond_action = torch.tensor(ca, dtype=torch.bfloat16, device="cuda")
        obs = seed_obs(seed1, instruction) if step == 0 else pred_obs(pred_buf, fpc, instruction)
        with torch.no_grad():
            _, vp = policy.lazy_joint_forward_causal(Batch(obs=obs), latent_video=latent_video,
                                                     cond_action=cond_action)
        latent_video = vp
        all_lat.append(vp)
        step_pix = decode_latents(policy, [vp])
        q, hh, ww = split_quads(step_pix)
        for ck, cam in zip(CAMERA_KEYS, CONCAT_ORDER):
            pred_buf[ck].extend(list(q[cam]))
        produced += step_pix.shape[0]
    full = decode_latents(policy, all_lat)
    quads, hh, ww = split_quads(full)
    return quads, hh, ww


def best_off(gt, pr, maxoff=6):
    from skimage.metrics import structural_similarity as ssim
    P, G = pr.shape[0], gt.shape[0]
    bo, bs = 0, -1e9
    for o in range(maxoff + 1):
        n = min(P, G - o)
        if n <= 6: continue
        s = np.mean([ssim(gt[o + k], pr[k], data_range=255, channel_axis=2) for k in range(0, min(n, 12))])
        if s > bs: bs, bo = s, o
    return bo


def plot_compare(results, hh, ww, dsdir, ep, start, horizons, out_png):
    """results: dict mode-> quads(dict cam->frames). Build per-camera [GT|modeA|modeB] rows."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from skimage.metrics import structural_similarity as ssim
    modes = list(results.keys())
    Tmax = max(results[m][CONCAT_ORDER[0]].shape[0] for m in modes)
    gt = read_gt_window(dsdir, ep, CAMERA_KEYS, start, Tmax + max(horizons) + 8)
    gt_v = {cam: np.stack([resize_to(f, hh, ww) for f in gt[ck]])
            for ck, cam in zip(CAMERA_KEYS, CONCAT_ORDER)}
    offs = {m: best_off(gt_v[CONCAT_ORDER[0]], results[m][CONCAT_ORDER[0]]) for m in modes}
    LBL = {"gtsim": "WM-sim (GT actions)", "dream": "closed-loop (own actions)"}

    for cam in CONCAT_ORDER:
        rows = 1 + len(modes)
        fig, ax = plt.subplots(rows, len(horizons), figsize=(2.3 * len(horizons), 2.3 * rows))
        for ci, t in enumerate(horizons):
            ax[0, ci].set_title(f"t = {t}", fontsize=13)
            ax[0, ci].imshow(gt_v[cam][min(t, gt_v[cam].shape[0] - 1)])
            ax[0, ci].set_xticks([]); ax[0, ci].set_yticks([])
        ax[0, 0].set_ylabel("GROUND TRUTH", fontsize=10)
        for mi, m in enumerate(modes):
            q = results[m][cam]; off = offs[m]; T = q.shape[0]
            for ci, t in enumerate(horizons):
                pt = min(t, T - 1)
                gtf = gt_v[cam][min(off + t, gt_v[cam].shape[0] - 1)]
                prf = q[pt]
                sv = ssim(gtf, prf, data_range=255, channel_axis=2)
                a = ax[mi + 1, ci]; a.imshow(prf); a.set_xticks([]); a.set_yticks([])
                a.set_xlabel(f"SSIM {sv:.2f}", fontsize=9)
            ax[mi + 1, 0].set_ylabel(LBL.get(m, m), fontsize=10)
        fig.suptitle(f"Open-loop WM vs GT  |  ckpt-20000  |  ep{ep} s{start}  |  {cam}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        p = out_png.replace(".png", f"_{cam}.png")
        fig.savefig(p, dpi=120); plt.close(fig)
        log.info(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--embodiment", default="franka_orca_bimanual")
    ap.add_argument("--clip", default="32:1600", help="ep:start")
    ap.add_argument("--modes", default="gtsim,dream")
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
    import imageio.v2 as iio
    horizons = [int(x) for x in a.horizons.split(",")]
    ep, start = (int(x) for x in a.clip.split(":"))
    modes = a.modes.split(",")

    stats = load_stats(os.path.join(a.dataset_dir, "meta", "relative_stats_dreamzero.json"))
    df = pd.read_parquet(os.path.join(a.dataset_dir, "data", "chunk-000", f"episode_{ep:06d}.parquet"))
    act = np.stack(df["action"].values)
    st = np.stack(df["observation.state"].values)

    results, hh, ww = {}, None, None
    for mode in modes:
        log.info(f"rollout mode={mode} ep{ep} s{start} ...")
        quads, hh, ww = rollout(policy, Batch, a.dataset_dir, ep, start, a.num_steps,
                                a.frames_per_chunk, a.instruction, mode, act=act, st=st, stats=stats)
        results[mode] = quads
        tag = f"ep{ep:06d}_s{start:05d}_{mode}"
        for cam in CONCAT_ORDER:
            iio.mimsave(f"{a.out_dir}/{tag}_{cam}.mp4", list(quads[cam]), fps=10, codec="libx264")
    plot_compare(results, hh, ww, a.dataset_dir, ep, start, horizons,
                 f"{a.out_dir}/openloop_compare_ep{ep:06d}_s{start:05d}.png")
    log.info("done.")


if __name__ == "__main__":
    main()
