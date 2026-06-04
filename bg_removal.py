"""
Background removal for DreamZero franka_orca_bimanual (tabletop dual-arm scene).

Goal: make the policy background-invariant. KEEP the manipulation scene (table + arms +
hands + objects), REMOVE the room behind/around it (red-tile floor, walls, clutter) and
replace it with a constant **black** fill. This closes the biggest train<->sim visual gap
("real room vs sim room"); see the pre-retrain checklist in docs/SIM_VALIDATION_AND_SCENE.md.

CRITICAL: the *same* transform must be applied to the sim camera frames at inference
(run_sim_eval_bimanual) so train==eval. Both the offline converter and the online eval import
`BackgroundRemover` from here so behaviour is identical.

Requirements (per user): background simply black; keep the table; remove the red floor tiles;
**the full arms must remain visible** (masking must never eat the arms).

Method: ADE20K semantic segmentation (pure PyTorch -> fits the existing cu12.9 torch env, no
onnxruntime). ADE20K has explicit wall/floor/ceiling/window classes -> background; the table
('table'/'desk') and everything else (incl. the OOD robot arms/objects, which ADE20K mislabels
but never as floor/wall) are kept. Two backends:
  - 'mask2former' (facebook/mask2former-swin-large-ade-semantic): most accurate boundaries; best
    at separating the red-tile floor from the wood table. Slower.
  - 'segformer'   (nvidia/segformer-b4/b2-finetuned-ade-512-512): faster, lower quality.
Masks are cleaned at low res (fill holes, close, drop speckle, DILATE so full arms survive). No
component pruning to the table (that could delete an arm not touching the table blob).
"""
from __future__ import annotations

import numpy as np

# ADE20K class *names* (substring match on id2label) that are background room surfaces.
# Robot arms/hands/objects are not ADE20K classes; they get mislabelled as e.g.
# chair/airplane/person/base but never floor/wall, so they survive as foreground.
ADE20K_REMOVE_SUBSTRINGS = (
    "wall", "floor", "ceiling", "windowpane", "window ", "door", "road",
    "sidewalk", "earth", "field", "rug", "carpet", "stairway", "stairs", "step",
    "runway", "path", "grass", "sky", "hill", "sand", "skyscraper",
    "building", "house", "fence", "railing", "blind", "curtain",
)
# Trusted foreground "anchor" — the workspace the arms operate over.
ADE20K_ANCHOR_SUBSTRINGS = ("table", "desk", "countertop", "counter", "pool table", "coffee table")

DEFAULT_SEGFORMER = "nvidia/segformer-b4-finetuned-ade-512-512"
DEFAULT_MASK2FORMER = "facebook/mask2former-swin-large-ade-semantic"
DEFAULT_BACKEND = "mask2former"
DEFAULT_MODEL = None       # resolved from backend if None
DEFAULT_FILL = 0           # black


class BackgroundRemover:
    def __init__(
        self,
        backend: str = DEFAULT_BACKEND,
        model_name: str | None = DEFAULT_MODEL,
        device: str | None = None,
        fill: int = DEFAULT_FILL,
        batch_size: int | None = None,
        close_radius: int = 4,
        dilate_radius: int = 4,
        min_island_frac: float = 0.003,
        anchor_dilate: int = 10,
        feather: float = 1.0,
        prune_to_anchor: bool = False,
    ):
        import torch

        self.torch = torch
        self.backend = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model_name is None:
            model_name = DEFAULT_MASK2FORMER if backend == "mask2former" else DEFAULT_SEGFORMER
        self.model_name = model_name

        if backend == "mask2former":
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
            self.proc = AutoImageProcessor.from_pretrained(model_name)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).eval().to(self.device)
            default_bs = 6
        elif backend == "segformer":
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
            self.proc = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).eval().to(self.device)
            default_bs = 16
        else:
            raise ValueError(f"unknown backend {backend!r}")
        if self.device == "cuda":
            self.model = self.model.half()

        id2label = self.model.config.id2label
        self.remove_ids = np.array(
            sorted(i for i, n in id2label.items() if any(s in n.lower() for s in ADE20K_REMOVE_SUBSTRINGS)),
            dtype=np.int64,
        )
        self.anchor_ids = np.array(
            sorted(i for i, n in id2label.items() if any(s in n.lower() for s in ADE20K_ANCHOR_SUBSTRINGS)),
            dtype=np.int64,
        )
        self.fill = int(fill)
        self.batch_size = int(batch_size or default_bs)
        self.close_radius = close_radius
        self.dilate_radius = dilate_radius
        self.min_island_frac = min_island_frac
        self.anchor_dilate = anchor_dilate
        self.feather = feather
        self.prune_to_anchor = prune_to_anchor

    # -- segmentation -> low-res label maps (N, h, w) -----------------------
    def _seg_lowres(self, frames: np.ndarray) -> np.ndarray:
        torch = self.torch
        H, W = frames.shape[1], frames.shape[2]
        h, w = max(8, H // 4), max(8, W // 4)
        inp = self.proc(images=list(frames), return_tensors="pt")
        inp = {k: (v.half() if (self.device == "cuda" and torch.is_floating_point(v)) else v).to(self.device)
               for k, v in inp.items()}
        with torch.no_grad():
            out = self.model(**inp)
        if self.backend == "mask2former":
            n = frames.shape[0]
            maps = self.proc.post_process_semantic_segmentation(out, target_sizes=[(h, w)] * n)
            return np.stack([m.to("cpu").numpy().astype(np.int64) for m in maps])
        logits = out.logits  # (N,C,h0,w0)
        up = torch.nn.functional.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False)
        return up.argmax(1).to("cpu").numpy().astype(np.int64)

    # -- mask cleanup (runs at low res; cheap) ------------------------------
    @staticmethod
    def _disk(r: int) -> np.ndarray:
        y, x = np.ogrid[-r : r + 1, -r : r + 1]
        return (x * x + y * y) <= r * r

    def _clean_lowres(self, seg: np.ndarray) -> np.ndarray:
        from scipy import ndimage as ndi

        h, w = seg.shape
        fg = ~np.isin(seg, self.remove_ids)
        fg = ndi.binary_fill_holes(fg)
        if self.close_radius > 0:
            fg = ndi.binary_closing(fg, structure=self._disk(self.close_radius))
            fg = ndi.binary_fill_holes(fg)

        # drop tiny speckle (keeps big regions incl. arms). NO anchor pruning by
        # default: never delete an arm that isn't connected to the table blob.
        lbl, n = ndi.label(fg)
        if n > 0:
            sizes = np.bincount(lbl.ravel())
            small = np.where(sizes < self.min_island_frac * h * w)[0]
            if small.size:
                fg &= ~np.isin(lbl, small)

        if self.prune_to_anchor:
            anchor = np.isin(seg, self.anchor_ids)
            if anchor.sum() < max(8, 0.002 * h * w):
                anchor = np.zeros((h, w), bool)
                anchor[h // 3 : 2 * h // 3 + 1, w // 4 : 3 * w // 4 + 1] = True
            anchor = ndi.binary_dilation(anchor, structure=self._disk(self.anchor_dilate))
            lbl, n = ndi.label(fg)
            touch = set(np.unique(lbl[anchor & (lbl > 0)]))
            if touch:
                fg = np.isin(lbl, list(touch))

        fg = ndi.binary_fill_holes(fg)
        if self.dilate_radius > 0:  # grow so the FULL arm survives (no eaten edges)
            fg = ndi.binary_dilation(fg, structure=self._disk(self.dilate_radius))
        return fg

    # -- public API ---------------------------------------------------------
    def foreground_alpha(self, frames: np.ndarray) -> np.ndarray:
        import cv2

        T, H, W, _ = frames.shape
        alpha = np.empty((T, H, W), dtype=np.float32)
        for s in range(0, T, self.batch_size):
            seg = self._seg_lowres(frames[s : s + self.batch_size])
            for j in range(seg.shape[0]):
                m = self._clean_lowres(seg[j]).astype(np.float32)
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
                if self.feather > 0:
                    k = max(1, int(self.feather * 4) | 1)
                    m = cv2.GaussianBlur(m, (k, k), self.feather)
                alpha[s + j] = np.clip(m, 0.0, 1.0)
        return alpha

    def apply(self, frames: np.ndarray) -> np.ndarray:
        alpha = self.foreground_alpha(frames)[..., None]
        out = frames.astype(np.float32) * alpha + self.fill * (1.0 - alpha)
        return out.astype(np.uint8)

    def apply_one(self, frame: np.ndarray) -> np.ndarray:
        return self.apply(frame[None])[0]
