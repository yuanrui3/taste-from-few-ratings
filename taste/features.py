#!/usr/bin/env python3
"""Embed every owner photo with open-source image models (GPU). No ratings are used.

Models (all open weights):
  DINOv2 ViT-L/14   facebook/dinov2-large (Apache-2.0), self-supervised; CLS token
  OpenCLIP ViT-L/14 laion2b_s32b_b82k (MIT); image embedding, and text prompts for zero-shot cues
  LAION aesthetic predictor (Apache-2.0): MLP on OpenAI CLIP ViT-L/14 -> the crowd-taste baseline

Variants embedded (made by owner_cutouts.py): cut, nocloth, face, bg, cloth.
Zero-shot cues (OpenCLIP text prompts, per photo): build (slim / athletic / curvy / heavier),
woman vs man, long vs short hair.

Writes OUT.npz with one row per owner photo: user, code, and for each model+variant an
L2-normalised embedding, plus aesthetic score and zero-shot probabilities.

`--prefetch` only downloads the weights (run on the login node; compute nodes may lack internet).
"""
import argparse
import json
import os
import urllib.request

import numpy as np
import torch
from PIL import Image

VARIANTS = ["cut", "nocloth", "face", "bg", "cloth"]
AES_URL = ("https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/"
           "sac+logos+ava1-l14-linearMSE.pth")
PROMPTS = {
    "build": ["a photo of a slim person", "a photo of a lean athletic person with visible muscles",
              "a photo of a curvy person", "a photo of a heavier person"],
    "sex": ["a photo of a woman", "a photo of a man"],
    "hair": ["a photo of a person with long hair", "a photo of a person with short hair"],
}


class AesMLP(torch.nn.Module):  # architecture of the LAION improved aesthetic predictor
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2), torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64), torch.nn.Dropout(0.1), torch.nn.Linear(64, 16), torch.nn.Linear(16, 1))

    def forward(self, x):
        return self.layers(x)


def load_models(models_dir, dev):
    import open_clip
    from transformers import AutoImageProcessor, AutoModel
    dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    dino = AutoModel.from_pretrained("facebook/dinov2-large").eval().to(dev).half()
    clip, _, clip_pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    clip = clip.eval().to(dev).half()
    tok = open_clip.get_tokenizer("ViT-L-14")
    # OpenAI's weights were trained with QuickGELU; the plain ViT-L-14 config would mis-embed
    oai, _, oai_pre = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
    oai = oai.eval().to(dev).half()
    aes_path = os.path.join(models_dir, "aesthetic", "sac+logos+ava1-l14-linearMSE.pth")
    if not os.path.exists(aes_path):
        os.makedirs(os.path.dirname(aes_path), exist_ok=True)
        urllib.request.urlretrieve(AES_URL, aes_path)
    aes = AesMLP()
    aes.load_state_dict(torch.load(aes_path, map_location="cpu", weights_only=True))
    aes = aes.eval().to(dev)
    with torch.no_grad():
        text = {k: torch.nn.functional.normalize(clip.encode_text(tok(v).to(dev)).float(), dim=-1)
                for k, v in PROMPTS.items()}
    return dict(dino=dino, dino_proc=dino_proc, clip=clip, clip_pre=clip_pre, oai=oai, oai_pre=oai_pre,
                aes=aes, text=text)


@torch.no_grad()
def embed_batch(M, ims, dev, variant):
    out = {}
    x = M["dino_proc"](images=ims, return_tensors="pt")["pixel_values"].to(dev).half()
    out["dino"] = torch.nn.functional.normalize(M["dino"](pixel_values=x).pooler_output.float(), dim=-1)
    x = torch.stack([M["clip_pre"](im) for im in ims]).to(dev).half()
    c = torch.nn.functional.normalize(M["clip"].encode_image(x).float(), dim=-1)
    out["clip"] = c
    if variant in ("cut", "face"):
        for k, t in M["text"].items():
            out["zs_" + k] = (100.0 * c @ t.T).softmax(-1)
    if variant == "cut":
        x = torch.stack([M["oai_pre"](im) for im in ims]).to(dev).half()
        o = torch.nn.functional.normalize(M["oai"].encode_image(x).float(), dim=-1)
        out["aesthetic"] = M["aes"](o).squeeze(-1)
    return {k: v.cpu().numpy() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", help="owner_cutouts.py output folder")
    ap.add_argument("--out")
    ap.add_argument("--models", required=True)
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--batch", type=int, default=48)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    M = load_models(a.models, dev if not a.prefetch else "cpu")
    if a.prefetch:
        print("weights ready")
        return
    rows = []
    for u in sorted(os.listdir(a.cut)):
        p = os.path.join(a.cut, u, "owner.json")
        if os.path.exists(p):
            rec = json.load(open(p))
            if rec.get("n_owner_photos", 0) >= 4:
                rows += [(u, ph["code"]) for ph in rec["photos"]]
    res = {"user": np.array([r[0] for r in rows]), "code": np.array([r[1] for r in rows])}
    for v in VARIANTS:
        acc = {}
        for i in range(0, len(rows), a.batch):
            ims = [Image.open(os.path.join(a.cut, u, "%s_%s.jpg" % (c, v))).convert("RGB") for u, c in rows[i:i + a.batch]]
            for k, arr in embed_batch(M, ims, dev, v).items():
                acc.setdefault(k, []).append(arr)
        for k, parts in acc.items():
            res["%s_%s" % (k, v)] = np.concatenate(parts)
        print("variant %s: %d photos" % (v, len(rows)), flush=True)
    res["zs_labels"] = np.array(json.dumps(PROMPTS))
    np.savez_compressed(a.out, **res)
    print("saved", a.out, {k: v.shape for k, v in res.items() if hasattr(v, "shape")})


if __name__ == "__main__":
    main()
