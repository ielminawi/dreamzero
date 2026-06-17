#!/usr/bin/env python3
"""Test A: POLICY action-prediction accuracy vs the demonstrations.

For many sampled timesteps, feed ONE real observation (frame + REAL proprio state +
task) and read the policy's predicted 24-step action chunk (absolute, via unapply).
Compare to the demonstration action[t:t+24]. Crucially also compare to a STAY-STILL
baseline (predict current state for all 24 steps) -- because the arms barely move
(~0.02 rad / 24 steps), "do nothing" already scores low error, so the policy must
BEAT stay-still to have actually learned the motion.
"""
import argparse, os, sys, json
import numpy as np
import torch
import torch.distributed as dist
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from validate_wm import CAMERA_KEYS, LANG_KEY
from action_norm import SEGMENTS, load_stats, denormalize_action_chunk  # [(name, a, b), ...]

import logging
log = logging.getLogger("action_eval")

DCAM = {"video.aria_rgb_cam": "observation.images.aria_rgb_cam",
        "video.oakd_front_view": "observation.images.oakd_front_view"}
ACT_KEYS = [f"action.{n}" for n, _, _ in SEGMENTS]


def build_obs(frames_t, state_t, instruction):
    obs = {ck: frames_t[ck][None] for ck in CAMERA_KEYS}              # (1,H,W,3) uint8
    for n, a, b in SEGMENTS:
        obs[f"state.{n}"] = state_t[a:b][None].astype(np.float64)    # (1, seg)
    obs[LANG_KEY] = instruction
    return obs


def pred_to_chunk(act):
    """policy batch.act -> (24,48) absolute, robust to dict or array."""
    if isinstance(act, dict):
        parts = [np.asarray(act[k], dtype=np.float64) for k in ACT_KEYS]
        arr = np.concatenate([p.reshape(p.shape[-2], p.shape[-1]) if p.ndim >= 2 else p[None]
                              for p in parts], axis=-1)
    else:
        arr = np.asarray(act, dtype=np.float64)
        arr = arr.reshape(-1, 48)
    return arr[:24]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--embodiment", default="franka_orca_bimanual")
    ap.add_argument("--episodes", default="2,5,32,73,147")
    ap.add_argument("--per-episode", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--instruction", default="bag the groceries")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    os.environ.setdefault("TOKENIZER_DIR", "/cluster/scratch/rjiang/dreamzero/checkpoints/umt5-xxl")
    os.environ.setdefault("ATTENTION_BACKEND", "FA2")
    torch._dynamo.config.recompile_limit = 800
    for k, v in dict(MASTER_ADDR="localhost", MASTER_PORT="12361", RANK="0", WORLD_SIZE="1").items():
        os.environ.setdefault(k, v)
    torch.cuda.set_device(0)
    if not dist.is_initialized(): dist.init_process_group("nccl")

    from torch.distributed.device_mesh import init_device_mesh
    from tianshou.data import Batch
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
    import decord
    log.info("loading model ...")
    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag(a.embodiment), model_path=a.model_path,
                            device=torch.device("cuda:0"),
                            device_mesh=init_device_mesh("cuda", (1,), mesh_dim_names=("ip",)))
    log.info("model loaded.")
    os.makedirs(a.out_dir, exist_ok=True)

    stats = load_stats(os.path.join(a.dataset_dir, "meta", "relative_stats_dreamzero.json"))
    episodes = [int(x) for x in a.episodes.split(",")]
    H = a.horizon
    # accumulators: per-segment abs-error lists for policy and stay baseline
    seg_names = [n for n, _, _ in SEGMENTS]
    P = {n: [] for n in seg_names}     # policy |pred-gt| (per joint, flattened)
    S = {n: [] for n in seg_names}     # stay   |state-gt|
    corr = {n: [] for n in seg_names}  # correlation pred vs gt (per chunk)
    P_ra = {n: [] for n in seg_names}  # re-anchored: pred shifted so pred[0]==state[t] (removes anchor offset)
    Pd = {n: [] for n in seg_names}    # displacement-only: (pred[h]-pred[0]) vs (gt[h]-gt[0])
    offs = {n: [] for n in seg_names}  # per-sample per-joint constant offset mean_h(pred-gt) -> consistency check
    samples = []                        # (ep,t, per-seg mae) for logging
    traces = None                       # save one chunk for trajectory plot

    for ep in episodes:
        df = pd.read_parquet(os.path.join(a.dataset_dir, "data", "chunk-000", f"episode_{ep:06d}.parquet"))
        act = np.stack(df["action"].values).astype(np.float64)
        st = np.stack(df["observation.state"].values).astype(np.float64)
        N = len(df)
        vrs = {ck: decord.VideoReader(os.path.join(a.dataset_dir, "videos", "chunk-000",
                                                    DCAM[ck], f"episode_{ep:06d}.mp4"))
               for ck in CAMERA_KEYS}
        ts = np.linspace(5, N - H - 5, a.per_episode).astype(int)
        for t in ts:
            frames_t = {ck: vrs[ck][t].asnumpy() for ck in CAMERA_KEYS}
            obs = build_obs(frames_t, st[t], a.instruction)
            with torch.no_grad():
                b, _ = policy.lazy_joint_forward_causal(Batch(obs=obs))
            # use the predicted NORMALIZED action (clean tensor) and denormalize with the
            # real state as reference -> absolute (same as the model's internal unapply).
            norm = b.normalized_action.detach().float().cpu().numpy().reshape(-1, 48)[:H]
            pred = denormalize_action_chunk(norm, st[t], stats)   # (24,48) absolute
            gt = act[t:t + H]                      # (24,48)
            stay = np.repeat(st[t][None], H, 0)    # (24,48) stay-still baseline
            rec = {"ep": int(ep), "t": int(t)}
            pred_ra = pred - pred[0:1] + st[t][None]     # re-anchor so pred_ra[0] == state[t]
            for n, aa, bb in SEGMENTS:
                pe = np.abs(pred[:, aa:bb] - gt[:, aa:bb])
                se = np.abs(stay[:, aa:bb] - gt[:, aa:bb])
                P[n].extend(pe.ravel().tolist()); S[n].extend(se.ravel().tolist())
                pv, gv = pred[:, aa:bb].ravel(), gt[:, aa:bb].ravel()
                if pv.std() > 1e-8 and gv.std() > 1e-8:
                    corr[n].append(float(np.corrcoef(pv, gv)[0, 1]))
                ra = np.abs(pred_ra[:, aa:bb] - gt[:, aa:bb]); P_ra[n].extend(ra.ravel().tolist())
                pdisp = pred[:, aa:bb] - pred[0:1, aa:bb]; gdisp = gt[:, aa:bb] - gt[0:1, aa:bb]
                Pd[n].extend(np.abs(pdisp - gdisp).ravel().tolist())
                offs[n].append((pred[:, aa:bb] - gt[:, aa:bb]).mean(axis=0).tolist())
                rec[f"{n}_mae"] = float(pe.mean()); rec[f"{n}_stay_mae"] = float(se.mean())
                rec[f"{n}_reanchored_mae"] = float(ra.mean())
            samples.append(rec)
            if traces is None:
                traces = dict(ep=int(ep), t=int(t), pred=pred.tolist(), gt=gt.tolist(), state=st[t].tolist())
        log.info(f"ep{ep}: done {len(ts)} timesteps")

    # summary
    summary = {}
    for n in seg_names:
        pol = float(np.mean(P[n])); sta = float(np.mean(S[n]))
        hi = [r for r in samples if r.get(f"{n}_stay_mae", 0.0) >= 0.02]   # high-motion chunks only
        hi_pol = float(np.mean([r[f"{n}_mae"] for r in hi])) if hi else float("nan")
        hi_sta = float(np.mean([r[f"{n}_stay_mae"] for r in hi])) if hi else float("nan")
        off_arr = np.array(offs[n])                       # (n_samples, segdim) per-joint offset per sample
        off_consistency = float(np.abs(off_arr.mean(axis=0)).mean() / (np.abs(off_arr).mean() + 1e-9))
        summary[n] = dict(policy_mae=pol, stay_mae=sta,
                          improvement=float((sta - pol) / sta) if sta > 0 else 0.0,
                          corr=float(np.mean(corr[n])) if corr[n] else float("nan"),
                          policy_reanchored_mae=float(np.mean(P_ra[n])),
                          policy_disp_mae=float(np.mean(Pd[n])),
                          highmotion_policy_mae=hi_pol, highmotion_stay_mae=hi_sta, n_highmotion=len(hi),
                          offset_mean_abs=float(np.abs(off_arr.mean(axis=0)).mean()),
                          offset_consistency=off_consistency)   # ~1 = same bias every sample (real anchor bug); ~0 = random
    with open(os.path.join(a.out_dir, "action_accuracy.json"), "w") as f:
        json.dump(dict(summary=summary, samples=samples, traces=traces), f, indent=2)

    log.info("=== ACTION-PREDICTION ACCURACY (MAE, radians) ===")
    log.info(f"{'segment':24s} {'policy':>8s} {'stay':>8s} {'policy<stay?':>12s} {'corr':>6s}")
    for n in seg_names:
        s = summary[n]
        log.info(f"{n:24s} {s['policy_mae']:8.4f} {s['stay_mae']:8.4f} "
                 f"{('YES +'+format(100*s['improvement'],'.0f')+'%') if s['policy_mae']<s['stay_mae'] else 'NO':>12s} "
                 f"{s['corr']:6.2f}")

    log.info("=== RE-ANCHORED / DISPLACEMENT / HIGH-MOTION (rad) -- isolate the anchor-bias from real tracking ===")
    log.info(f"{'segment':24s} {'reanch':>8s} {'disp':>8s} {'stay':>8s} {'hi_pol':>8s} {'hi_stay':>8s} {'offset':>7s} {'consist':>7s}")
    for n in seg_names:
        s = summary[n]
        log.info(f"{n:24s} {s['policy_reanchored_mae']:8.4f} {s['policy_disp_mae']:8.4f} {s['stay_mae']:8.4f} "
                 f"{s['highmotion_policy_mae']:8.4f} {s['highmotion_stay_mae']:8.4f} {s['offset_mean_abs']:7.4f} {s['offset_consistency']:7.2f}")

    # plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        # bar: policy vs stay MAE per segment
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        x = np.arange(len(seg_names)); w = 0.38
        ax[0].bar(x - w/2, [summary[n]["policy_mae"] for n in seg_names], w, label="policy")
        ax[0].bar(x + w/2, [summary[n]["stay_mae"] for n in seg_names], w, label="stay-still baseline")
        ax[0].set_xticks(x); ax[0].set_xticklabels([n.replace("_joint_pos", "") for n in seg_names], rotation=20)
        ax[0].set_ylabel("MAE vs demo action (rad)"); ax[0].legend()
        ax[0].set_title("Policy vs stay-still (lower=better; policy<stay => learned motion)")
        # trajectory overlay for a few joints from the saved chunk
        tr = traces; pr = np.array(tr["pred"]); gt = np.array(tr["gt"]); state = np.array(tr["state"])
        for j, jn in [(0, "L-arm j0"), (7, "R-arm j0"), (14, "L-hand j0"), (31, "R-hand j0")]:
            ax[1].plot(gt[:, j], "-", label=f"GT {jn}")
            ax[1].plot(pr[:, j], "--", label=f"pred {jn}")
        ax[1].axhline(0, color="k", lw=0.3)
        ax[1].set_title(f"Predicted vs demo action trace (ep{tr['ep']} t{tr['t']})")
        ax[1].set_xlabel("horizon step"); ax[1].set_ylabel("joint pos (rad)"); ax[1].legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "action_accuracy.png"), dpi=120)
        log.info(f"wrote {a.out_dir}/action_accuracy.png")
    except Exception as e:
        log.info(f"plot skipped: {e}")
    log.info("done.")


if __name__ == "__main__":
    main()
