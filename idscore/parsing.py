"""Human parsing into face / exposed skin / hair / clothing.

Model: mattmdjaga/segformer_b2_clothes (SegFormer-B2 fine-tuned on ATR,
MIT licence, open weights, ~110 MB, runs on CPU).

The region definition encodes the project's stated priorities:
  - face            -> scored
  - exposed skin    -> scored (arms, legs as labelled "skin" by the parser)
  - hair            -> reported separately, not part of the headline score
  - clothing, shoes, bags, accessories -> never scored
"""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

MODEL_ID = "mattmdjaga/segformer_b2_clothes"
ATR_LABELS = [
    "background", "hat", "hair", "sunglasses", "upper_clothes", "skirt", "pants",
    "dress", "belt", "left_shoe", "right_shoe", "face", "left_leg", "right_leg",
    "left_arm", "right_arm", "bag", "scarf",
]
L = {n: i for i, n in enumerate(ATR_LABELS)}

FACE = [L["face"]]
SKIN_LIMBS = [L["left_arm"], L["right_arm"], L["left_leg"], L["right_leg"]]
HAIR = [L["hair"]]
CLOTHING = [L[n] for n in ("hat", "sunglasses", "upper_clothes", "skirt", "pants", "dress",
                           "belt", "left_shoe", "right_shoe", "bag", "scarf")]


class HumanParser:
    def __init__(self, cache_dir: str, target_height: int = 768):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        os.makedirs(cache_dir, exist_ok=True)
        try:  # use the local copy without touching the network (works offline)
            self.proc = SegformerImageProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir, local_files_only=True)
            self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID, cache_dir=cache_dir, local_files_only=True).eval()
        except Exception:  # first use: download once
            self.proc = SegformerImageProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
            self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID, cache_dir=cache_dir).eval()
        self.target_height = target_height

    @torch.no_grad()
    def parse_crop(self, rgb_crop: np.ndarray, flip_tta: bool = True) -> np.ndarray:
        """uint8 HxWx3 person crop -> class probabilities (18,H,W) float32."""
        h, w = rgb_crop.shape[:2]
        scale = self.target_height / h
        im = Image.fromarray(rgb_crop).resize((max(32, round(w * scale)), self.target_height), Image.BICUBIC)

        def run(pil):
            inp = self.proc(images=pil, return_tensors="pt", do_resize=False)
            logits = self.model(**inp).logits
            logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            return logits.softmax(1)[0]

        p = run(im)
        if flip_tta:
            pf = run(im.transpose(Image.FLIP_LEFT_RIGHT)).flip(-1)
            # left/right labels swap under a horizontal flip
            swap = list(range(len(ATR_LABELS)))
            for a, b in (("left_shoe", "right_shoe"), ("left_leg", "right_leg"), ("left_arm", "right_arm")):
                swap[L[a]], swap[L[b]] = L[b], L[a]
            p = (p + pf[swap]) / 2.0
        return p.numpy().astype(np.float32)


def expand_box(box, img_wh, margin=0.15):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    m = margin * max(w, h)
    W, H = img_wh
    return (
        int(max(0, np.floor(x0 - m))), int(max(0, np.floor(y0 - m))),
        int(min(W, np.ceil(x1 + m))), int(min(H, np.ceil(y1 + m))),
    )


def region_masks(prob: np.ndarray, min_conf: float = 0.5) -> dict:
    """Class probabilities -> boolean region masks (confident pixels only)."""
    lab = prob.argmax(0)
    conf = prob.max(0)
    ok = conf >= min_conf

    def m(ids):
        return np.isin(lab, ids) & ok

    face, skin, hair, cloth = m(FACE), m(SKIN_LIMBS), m(HAIR), m(CLOTHING)
    return {
        "face": face,
        "skin_limbs": skin,
        "hair": hair,
        "clothing": cloth,
        "identity": face | skin,  # headline region: face + exposed skin
        "person": lab != L["background"],
        "label": lab.astype(np.uint8),
    }
