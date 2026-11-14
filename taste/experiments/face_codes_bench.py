#!/usr/bin/env python3
"""Which face codes rank groups the rater rarely rated? (PROTOCOL.md, "No more ratings: face codes ...").

  face_codes_bench.py embed ROOT                    (GPU) ViT-H, DINOv2 and ArcFace codes of both portrait
                                                    batches -> mebatch*/codes_more.npz
  face_codes_bench.py eval ROOT RATINGS1 RATINGS2   variants C0-C7 -> results/face_codes_bench.json
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]
BATCHES = [("mebatch_set.json", "mebatch"), ("mebatch2_set.json", "mebatch2")]
VARIANTS = {"C0": ["clip", "siglip", "sface"], "C1": ["clip", "siglip"], "C2": ["clip", "siglip", "sface", "vith"],
            "C3": ["clip", "siglip", "sface", "dino"], "C4": ["clip", "siglip", "arc"], "C5": ["clip", "siglip", "sface", "arc"],
            "C6": ["siglip", "sface"], "C7": ["clip", "siglip", "sface", "vith", "dino", "arc"]}
ALPHAS = np.logspace(-1, 5, 25)


def l2(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def embed(root):
    import open_clip
    import torch
    import onnxruntime as ort
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    from idscore.faceid import FaceID
    from owner_arcface import aligned
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vh, _, vtf = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k"); vh = vh.eval().to(dev).half()
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-large"); dino = AutoModel.from_pretrained("facebook/dinov2-large").eval().to(dev).half()
    sess = ort.InferenceSession(os.path.join(root, "models/insightface/buffalo_l/w600k_r50.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    fid = FaceID(os.path.join(root, "models", "opencv_zoo"))
    for set_name, d in BATCHES:
        items = json.load(open(os.path.join(root, "results", set_name)))
        ims = [Image.open(os.path.join(root, d, it["id"] + "_face.jpg")).convert("RGB") for it in items]
        with torch.no_grad():
            v = torch.nn.functional.normalize(vh.encode_image(torch.stack([vtf(x) for x in ims]).to(dev).half()).float(), dim=-1).cpu().numpy()
            x = dproc(images=ims, return_tensors="pt")["pixel_values"].to(dev).half()
            dn = torch.nn.functional.normalize(dino(pixel_values=x).pooler_output.float(), dim=-1).cpu().numpy()
        arc = np.full((len(ims), 512), np.nan, np.float32)
        for j, im in enumerate(ims):
            rgb = np.array(im); row = fid.detect(rgb)
            if row is not None:
                a = ((aligned(rgb, row).astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
                e = sess.run(None, {sess.get_inputs()[0].name: a})[0][0]; arc[j] = e / np.linalg.norm(e)
        np.savez_compressed(os.path.join(root, d, "codes_more.npz"), id=np.array([it["id"] for it in items]), vith_face=v, dino_face=dn, arc=arc)
        print(d, "ViT-H", v.shape, "DINOv2", dn.shape, "ArcFace found", int((~np.isnan(arc).any(1)).sum()), "of", len(ims), flush=True)


def pool_arc(root, people):
    out = []
    for u in people:
        a = np.load(os.path.join(root, "arc", u + ".npz")); rec = json.load(open(os.path.join(root, "cut", u, "owner.json"))); E = []
        for ph in rec["photos"]:
            m = np.nonzero(a["code"] == ph["code"])[0]
            if len(m):
                box = np.array(ph["face_box"]); j = m[np.argmin(np.abs(a["row"][m, :4] - box).sum(1))]
                if np.abs(a["row"][j, :4] - box).sum() < 4:
                    E.append(a["emb"][j])
        mvec = np.mean(E, 0) if E else np.zeros(512); out.append(mvec / (np.linalg.norm(mvec) + 1e-12))
    return np.stack(out)


def evaluate(root, r1, r2):
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from mebatch2 import batch_xy
    from sface_input import sface_means
    from train_eval import folds, load, mean_vec, photo_index
    F, train, test, first, R, group, split, same = load(root); idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True)); Fx = dict(F)
    Fx["siglip_face"] = M["siglip_face"]; Fx["vith_face"] = M["vith_face"]
    people = sorted(u for u in set(split["train"]) | set(split["test"]) if u in first); yp = np.array([first[u] for u in people], float)
    P = {"clip": mean_vec(Fx, idx, people, "clip_face"), "siglip": mean_vec(Fx, idx, people, "siglip_face"), "sface": sface_means(root, people),
         "vith": mean_vec(Fx, idx, people, "vith_face"), "dino": mean_vec(Fx, idx, people, "dino_face"), "arc": pool_arc(root, people)}
    tr_pos = np.array([people.index(u) for u in train]); ytr = yp[tr_pos]
    batches = []
    for (set_name, d), rp in zip(BATCHES, (r1, r2)):
        items, X3, y = batch_xy(root, set_name, d, rp)
        Mo = np.load(os.path.join(root, d, "codes_more.npz"), allow_pickle=True); assert list(Mo["id"]) == [it["id"] for it in items]
        arc = Mo["arc"].astype(float); arc = np.where(np.isnan(arc), np.nanmean(arc, 0), arc)
        B = {"clip": X3[:, :768], "siglip": X3[:, 768:768 + 1152], "sface": X3[:, -128:], "vith": l2(Mo["vith_face"].astype(float)),
             "dino": l2(Mo["dino_face"].astype(float)), "arc": l2(arc)}
        ok = ~np.isnan(y)
        sel = {"all": ok, "Black": ok & np.array([it["ethnicity"] == "black" for it in items]), "men": ok & np.array([it["gender"] == "male" for it in items])}
        batches.append((B, y, sel))
    out = {}
    print("%-4s %-34s %8s | %-26s | %-26s" % ("", "codes", "pool CV", "batch 1: all / Black / men", "batch 2: all / Black / men"))
    for v, blocks in VARIANTS.items():
        Xp = np.concatenate([P[b] for b in blocks], 1); rh = []
        for r in range(3):
            fo = folds(train, group, seed=r); p = np.zeros(len(train)); Xt = Xp[tr_pos]
            for k in range(5):
                a, b = fo != k, fo == k; sc = StandardScaler().fit(Xt[a])
                p[b] = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xt[a]), ytr[a]).predict(sc.transform(Xt[b]))
            rh.append(spearmanr(p, ytr)[0])
        sc = StandardScaler().fit(Xp); m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xp), yp)
        res = {"pool_cv": float(np.mean(rh))}
        for bi, (B, y, sel) in enumerate(batches):
            pr = m.predict(sc.transform(np.concatenate([B[b] for b in blocks], 1)))
            res["batch%d" % (bi + 1)] = {k: float(spearmanr(pr[s], y[s])[0]) for k, s in sel.items()}
        out[v] = res
        print("%-4s %-34s %8.3f | %6.3f %6.3f %6.3f        | %6.3f %6.3f %6.3f" % (
            v, "+".join(blocks), res["pool_cv"], *[res["batch1"][k] for k in ("all", "Black", "men")], *[res["batch2"][k] for k in ("all", "Black", "men")]))
    elig = [v for v in VARIANTS if out[v]["pool_cv"] >= out["C0"]["pool_cv"] - 0.01]
    choice = max(elig, key=lambda v: out[v]["batch1"]["Black"])
    b2, b0 = out[choice]["batch2"], out["C0"]["batch2"]
    passes = choice != "C0" and b2["Black"] - b0["Black"] >= 0.05 and b2["all"] - b0["all"] >= -0.02 and b2["men"] - b0["men"] >= -0.02
    out["choice"] = {"variant": choice, "eligible": elig, "batch2_test_passes": bool(passes)}
    print("chosen on batch 1 (pool CV within 0.01 of C0): %s; batch 2 test: Black %+.3f, all %+.3f, men %+.3f -> %s" % (
        choice, b2["Black"] - b0["Black"], b2["all"] - b0["all"], b2["men"] - b0["men"],
        "passes; full-model check next" if passes else ("current codes stay" if choice != "C0" else "C0 is best: current codes stay")))
    json.dump(out, open(os.path.join(root, "results/face_codes_bench.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        embed(sys.argv[2])
    else:
        evaluate(sys.argv[2], sys.argv[3], sys.argv[4])
