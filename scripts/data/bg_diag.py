"""
Diagnostic: what does Mask2Former label the wood table vs the red tiles? And does a
"keep table, remove only room-walls + red-coloured floor" rule keep the FULL table?
Output: output/bg_diag/contact.png  rows=frames; cols = original | seg-colorized | candidate-keep(black)
Also prints, per frame, the classes present (name, %px, mean RGB).
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import cv2, h5py, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bg_removal  # noqa: E402

RAW = "/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries"
OUT = Path("/cluster/scratch/rjiang/dreamzero/output/bg_diag"); OUT.mkdir(parents=True, exist_ok=True)
TARGET = (640, 480)
ROOM = ("wall", "ceiling", "windowpane", "window ", "door", "curtain", "blind",
        "building", "house", "sky", "skyscraper", "fence", "railing")
FLOORISH = ("floor", "rug", "carpet", "earth", "road", "sidewalk", "path", "sand", "field")


def grab():
    files = sorted(glob.glob(f"{RAW}/*.h5"))
    picks = [(files[0], "aria_rgb_cam", 0.45), (files[0], "oakd_front_view", 0.45),
             (files[len(files)//2], "aria_rgb_cam", 0.4), (files[len(files)//2], "oakd_front_view", 0.55)]
    out = []
    for fp, cam, fr in picks:
        with h5py.File(fp, "r") as f:
            ds = f[f"observations/images/{cam}/color"]; T = ds.shape[0]
            out.append(cv2.resize(ds[int(T*fr)], TARGET, interpolation=cv2.INTER_AREA))
    return np.stack(out)


def main():
    frames = grab()
    br = bg_removal.BackgroundRemover(backend="mask2former", fill=0)
    id2label = br.model.config.id2label
    room_ids = {i for i, n in id2label.items() if any(s in n.lower() for s in ROOM)}
    floor_ids = {i for i, n in id2label.items() if any(s in n.lower() for s in FLOORISH)}
    rng = np.random.RandomState(0)
    palette = rng.randint(0, 255, (256, 3), np.uint8)

    rows = []
    for i in range(len(frames)):
        seg = br._seg_lowres(frames[i:i+1])[0]  # (h,w) low-res
        H, W = frames[i].shape[:2]
        seg_full = cv2.resize(seg.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
        colorized = palette[seg_full % 256]
        # candidate: keep everything except room-walls and RED floor pixels
        rgb = frames[i].astype(np.int32)
        red = (rgb[..., 0] > rgb[..., 1] + 12) & (rgb[..., 0] > rgb[..., 2] + 25) & (rgb[..., 0] > 70)
        is_room = np.isin(seg_full, list(room_ids))
        is_floor = np.isin(seg_full, list(floor_ids))
        remove = is_room | (is_floor & red)
        keep = ~remove
        from scipy import ndimage as ndi
        keep = ndi.binary_fill_holes(keep)
        cand = (frames[i] * keep[..., None]).astype(np.uint8)
        trio = [cv2.resize(x, (320, 240)) for x in (frames[i], colorized, cand)]
        rows.append(np.concatenate(trio, axis=1))
        # legend
        ids, cnts = np.unique(seg_full, return_counts=True)
        order = np.argsort(-cnts)
        print(f"--- frame {i} ---")
        for k in order[:8]:
            cid = ids[k]; m = seg_full == cid; mc = frames[i][m].mean(0)
            tag = "ROOM" if cid in room_ids else ("FLOOR" if cid in floor_ids else "keep")
            print(f"  {id2label[cid]:<16} {100*cnts[k]/(H*W):5.1f}%  meanRGB=({mc[0]:.0f},{mc[1]:.0f},{mc[2]:.0f}) [{tag}]")
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(OUT / "contact.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("wrote", OUT / "contact.png", "| cols: original | seg-colorized | candidate(keep table, drop walls+red floor)")


if __name__ == "__main__":
    main()
