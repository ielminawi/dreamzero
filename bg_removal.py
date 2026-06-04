"""
Background removal for DreamZero franka_orca_bimanual (tabletop dual-arm scene).

Goal: make the policy background-invariant by removing the *room* (floor / walls /
clutter behind the table) from the camera frames and replacing it with a constant
fill, while KEEPING the manipulation scene (table + arms + objects). This closes the
biggest train<->sim visual gap ("real room vs sim room"); see the pre-retrain
checklist in docs/SIM_VALIDATION_AND_SCENE.md.

CRITICAL: the *same* transform must be applied to the sim camera frames at inference
(run_sim_eval_bimanual) so train==eval. Both the offline converter and the online
eval import `BackgroundRemover` from this module so the behaviour is identical.

Method: ADE20K semantic segmentation (SegFormer, pure PyTorch -> works with the
existing torch/cu12.9 env, no onnxruntime). ADE20K has explicit wall/floor/ceiling/
window classes; everything else (table, and the robot arms/objects which ADE20K
mislabels but does NOT call floor/wall) is treated as foreground. Raw seg is noisy on
the (out-of-distribution) robot arms, so we clean the mask at segmentation resolution:
fill holes, morphological close, keep only components connected to the table / image
centre (drops far misclassified floor islands), dilate, then upsample + feather.
"""
from __future__ import annotations

import numpy as np

# ADE20K class *names* (substring match on id2label) that are background room surfaces.
# Robot arms/hands/objects are NOT in ADE20K; they get mislabelled as e.g.
# chair/airplane/person/base but never as floor/wall, so they survive as foreground.
ADE20K_REMOVE_SUBSTRINGS = (
    "wall", "floor", "ceiling", "windowpane", "window ", "door", "road",
    "sidewalk", "earth", "field", "rug", "stairway", "stairs", "step",
    "runway", "path", "grass", "sky", "hill", "sand", "skyscraper",
    "building", "house", "fence", "railing", "blind",
)
# Trusted foreground "anchor" — the workspace the arms operate over.
ADE20K_ANCHOR_SUBSTRINGS = ("table", "desk", "countertop", "counter", "pool table")

DEFAULT_MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"
DEFAULT_FILL = 128  # mid-gray; matches a neutral sim backdrop


class BackgroundRemover:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        fill: int = DEFAULT_FILL,
        batch_size: int = 16,
        close_radius: int = 4,
        dilate_radius: int = 2,
        min_island_frac: float = 0.004,
        anchor_dilate: int = 10,
        feather: float = 1.5,
        prune_to_anchor: bool = False,
    ):
        import torch
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = SegformerImageProcessor.from_pretrained(model_name)
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(model_name)
            .eval()
            .to(self.device)
        )
        if self.device == "cuda":
            self.model = self.model.half()
        id2label = self.model.config.id2label
        self.remove_ids = np.array(
            sorted(
                i
                for i, n in id2label.items()
                if any(s in n.lower() for s in ADE20K_REMOVE_SUBSTRINGS)
            ),
            dtype=np.int64,
        )
        self.anchor_ids = np.array(
            sorted(
                i
                for i, n in id2label.items()
                if any(s in n.lower() for s in ADE20K_ANCHOR_SUBSTRINGS)
            ),
            dtype=np.int64,
        )
        self.fill = int(fill)
        self.batch_size = int(batch_size)
        self.close_radius = close_radius
        self.dilate_radius = dilate_radius
        self.min_island_frac = min_island_frac
        self.anchor_dilate = anchor_dilate
        self.feather = feather
        self.prune_to_anchor = prune_to_anchor

    # -- segmentation -------------------------------------------------------
    def _seg_lowres(self, frames: np.ndarray) -> np.ndarray:
        """frames (N,H,W,3) uint8 RGB -> seg label map at logits resolution (N,h,w)."""
        torch = self.torch
        inp = self.proc(images=list(frames), return_tensors="pt")
        pix = inp["pixel_values"].to(self.device)
        if self.device == "cuda":
            pix = pix.half()
        with torch.no_grad():
            logits = self.model(pixel_values=pix).logits  # (N,C,h,w)
        return logits.argmax(1).to("cpu").numpy().astype(np.int64)

    # -- mask cleanup (runs at low res; cheap) ------------------------------
    @staticmethod
    def _disk(r: int) -> np.ndarray:
        y, x = np.ogrid[-r : r + 1, -r : r + 1]
        return (x * x + y * y) <= r * r

    def _clean_lowres(self, seg: np.ndarray) -> np.ndarray:
        from scipy import ndimage as ndi

        h, w = seg.shape
        remove = np.isin(seg, self.remove_ids)
        fg = ~remove
        fg = ndi.binary_fill_holes(fg)
        if self.close_radius > 0:
            fg = ndi.binary_closing(fg, structure=self._disk(self.close_radius))
            fg = ndi.binary_fill_holes(fg)

        # Drop tiny speckle always. Component pruning to the table anchor is OFF by
        # default: it can delete an arm that isn't connected to the detected table
        # blob (manipulation content must never be removed). Residual misclassified
        # floor is the lesser evil.
        lbl, n = ndi.label(fg)
        if n > 0:
            sizes = np.bincount(lbl.ravel())
            min_size = self.min_island_frac * h * w
            small = np.where(sizes < min_size)[0]
            if small.size:
                fg &= ~np.isin(lbl, small)

        if self.prune_to_anchor:
            anchor = np.isin(seg, self.anchor_ids)
            if anchor.sum() < max(8, 0.002 * h * w):
                anchor = np.zeros((h, w), bool)
                anchor[h // 3 : 2 * h // 3 + 1, w // 4 : 3 * w // 4 + 1] = True
            if self.anchor_dilate > 0:
                anchor = ndi.binary_dilation(anchor, structure=self._disk(self.anchor_dilate))
            lbl, n = ndi.label(fg)
            touch = set(np.unique(lbl[anchor & (lbl > 0)]))
            if touch:
                fg = np.isin(lbl, list(touch))

        fg = ndi.binary_fill_holes(fg)
        if self.dilate_radius > 0:
            fg = ndi.binary_dilation(fg, structure=self._disk(self.dilate_radius))
        return fg

    # -- public API ---------------------------------------------------------
    def foreground_alpha(self, frames: np.ndarray) -> np.ndarray:
        """frames (T,H,W,3) uint8 -> alpha (T,H,W) float32 in [0,1] (1=foreground)."""
        import cv2

        T, H, W, _ = frames.shape
        alpha = np.empty((T, H, W), dtype=np.float32)
        for s in range(0, T, self.batch_size):
            batch = frames[s : s + self.batch_size]
            seg = self._seg_lowres(batch)  # (n,h,w)
            for j in range(seg.shape[0]):
                m = self._clean_lowres(seg[j]).astype(np.float32)
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
                if self.feather > 0:
                    k = max(1, int(self.feather * 4) | 1)
                    m = cv2.GaussianBlur(m, (k, k), self.feather)
                alpha[s + j] = np.clip(m, 0.0, 1.0)
        return alpha

    def apply(self, frames: np.ndarray) -> np.ndarray:
        """frames (T,H,W,3) uint8 -> background-filled frames (T,H,W,3) uint8."""
        alpha = self.foreground_alpha(frames)[..., None]
        out = frames.astype(np.float32) * alpha + self.fill * (1.0 - alpha)
        return out.astype(np.uint8)

    def apply_one(self, frame: np.ndarray) -> np.ndarray:
        """Single (H,W,3) uint8 frame -> background-filled frame (for inference)."""
        return self.apply(frame[None])[0]
