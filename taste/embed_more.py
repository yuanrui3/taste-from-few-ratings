#!/usr/bin/env python3
"""Embed the owner photos with larger open image models (GPU).

Models (open weights): OpenCLIP ViT-H-14 laion2b_s32b_b79k (MIT); SigLIP ViT-SO400M-14-SigLIP-384
webli (Apache-2.0). Variants: nocloth and face (the main model's inputs) and cut.
Writes features_more.npz with user, code and "<model>_<variant>" arrays (L2-normalised), in the
same photo order as features.npz.  `--prefetch` only downloads the weights (login node).
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

MODELS = {"vith": ("ViT-H-14", "laion2b_s32b_b79k"), "siglip": ("ViT-SO400M-14-SigLIP-384", "webli")}
VARIANTS = ["nocloth", "face", "cut"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    import open_clip
    dev = "cuda" if torch.cuda.is_available() and not a.prefetch else "cpu"
    loaded = {}
    for k, (arch, pre) in MODELS.items():
        if k not in a.models.split(","):
            continue
        m, _, tf = open_clip.create_model_and_transforms(arch, pretrained=pre)
        loaded[k] = (m.eval().to(dev).half() if dev == "cuda" else m.eval(), tf)
        print("ready", k, flush=True)
    if a.prefetch:
        return
    base = np.load(os.path.join(a.root, "features.npz"), allow_pickle=True)
    users, codes = base["user"], base["code"]
    out = {"user": users, "code": codes}
    for k, (m, tf) in loaded.items():
        for v in VARIANTS:
            embs = []
            for i in range(0, len(users), 32):
                ims = [tf(Image.open(os.path.join(a.root, "cut", str(u), "%s_%s.jpg" % (c, v))).convert("RGB"))
                       for u, c in zip(users[i:i + 32], codes[i:i + 32])]
                with torch.no_grad():
                    e = m.encode_image(torch.stack(ims).to(dev).half()).float()
                embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
            out["%s_%s" % (k, v)] = np.concatenate(embs)
            print(k, v, out["%s_%s" % (k, v)].shape, flush=True)
    np.savez_compressed(os.path.join(a.root, "features_more_%s.npz" % "_".join(loaded)), **out)
    print("saved")


if __name__ == "__main__":
    main()
