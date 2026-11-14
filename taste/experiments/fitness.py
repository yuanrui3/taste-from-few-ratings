#!/usr/bin/env python3
"""Fitness score from body-part crops (GPU). See PROTOCOL.md "Fitness score".

For every owner photo with body keypoints: crops of the lower torso (abs), upper arms and thighs
from the original photo; OpenCLIP judges whether the part is bare and how defined it looks.
Per person: weighted mean definition per region, standardized, averaged over the regions shown.

Outputs: results/fitness.json (per-person scores, coverage, validation numbers),
results/fitness_top.jpg / fitness_bottom.jpg (12 highest / lowest: best abs-or-arm crop + 2 cut-outs).
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from owner_cutouts import load as load_photo  # noqa: E402  (same resizing as the owner step)
from probe_search import winner_features  # noqa: E402
from train_eval import fit_predict, folds, load, photo_index  # noqa: E402

PROMPTS = {
    "abs": (("a bare stomach", "a stomach covered by a shirt"),
            ("a lean toned stomach with visible abs", "a soft stomach with some belly fat")),
    "arms": (("bare arms", "arms covered by sleeves"),
             ("toned muscular arms", "soft untoned arms")),
    "thighs": (("bare legs", "legs covered by trousers"),
               ("toned athletic thighs", "soft untoned thighs")),
}
VIS = 2.0  # keypoint score above which a joint counts as visible


def seg_box(a, b, W, H, widen=0.45):
    (x0, y0), (x1, y1) = a, b
    L = max(1.0, np.hypot(x1 - x0, y1 - y0)); m = widen * L
    bx0, by0, bx1, by1 = min(x0, x1) - m, min(y0, y1) - m * 0.4, max(x0, x1) + m, max(y0, y1) + m * 0.4
    return [int(max(0, bx0)), int(max(0, by0)), int(min(W, bx1)), int(min(H, by1))]


def crops_for(rgb, kp):
    H, W = rgb.shape[:2]; k = np.array(kp); v = lambda i: k[i][2] > VIS; p = lambda i: k[i][:2]
    out = []
    if all(v(i) for i in (5, 6, 11, 12)):
        sy = (p(5)[1] + p(6)[1]) / 2; hy = (p(11)[1] + p(12)[1]) / 2; h = hy - sy
        if h > 30:
            xs = [p(i)[0] for i in (5, 6, 11, 12)]; w = max(xs) - min(xs)
            out.append(("abs", [int(max(0, min(xs) - 0.12 * w)), int(max(0, sy + 0.4 * h)),
                                int(min(W, max(xs) + 0.12 * w)), int(min(H, hy + 0.1 * h))]))
    for a, b in ((5, 7), (6, 8)):
        if v(a) and v(b):
            out.append(("arms", seg_box(p(a), p(b), W, H)))
    for a, b in ((11, 13), (12, 14)):
        if v(a) and v(b):
            out.append(("thighs", seg_box(p(a), p(b), W, H)))
    return [(r, b) for r, b in out if b[2] - b[0] >= 40 and b[3] - b[1] >= 40]


def main(root):
    import open_clip
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    users = sorted(set(split["train"]) | set(split["test"]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    model = model.eval().to(dev).half(); tok = open_clip.get_tokenizer("ViT-L-14")
    with torch.no_grad():
        T = {}
        for r, (gate, defn) in PROMPTS.items():
            t = model.encode_text(tok(["a photo of " + s for s in gate + defn]).to(dev)).float()
            T[r] = torch.nn.functional.normalize(t, dim=-1)

    recs, imgs = [], []
    def flush():
        if not imgs:
            return
        with torch.no_grad():
            e = torch.nn.functional.normalize(model.encode_image(torch.stack(imgs).to(dev).half()).float(), dim=-1)
        for (u, code, r, box), z in zip(recs[-len(imgs):], e):
            t = T[r]
            gate = torch.softmax(100 * z @ t[:2].T, -1)[0].item()
            defn = (z @ t[2] - z @ t[3]).item()
            store.append({"user": u, "code": code, "region": r, "bare": gate, "def": defn, "box": box})
        imgs.clear()
    store = []
    for u in users:
        rec = json.load(open(os.path.join(root, "cut", u, "owner.json")))
        files = {os.path.splitext(f)[0]: f for f in os.listdir(os.path.join(root, "photos", u))}
        for ph in rec["photos"]:
            if not ph.get("keypoints") or ph["code"] not in files:
                continue
            rgb = load_photo(os.path.join(root, "photos", u, files[ph["code"]]))
            if rgb is None or list(rgb.shape[1::-1]) != ph["size"]:
                continue
            for r, b in crops_for(rgb, ph["keypoints"]):
                recs.append((u, ph["code"], r, b)); imgs.append(pre(Image.fromarray(rgb[b[1]:b[3], b[0]:b[2]])))
                if len(imgs) == 64:
                    flush()
    flush()
    print("crops scored:", len(store), {r: sum(s["region"] == r for s in store) for r in PROMPTS}, flush=True)

    # per person
    per = {u: {} for u in users}
    for r in PROMPTS:
        for u in users:
            ss = [s for s in store if s["user"] == u and s["region"] == r and s["bare"] > 0.5]
            if ss:
                w = np.array([s["bare"] for s in ss]); d = np.array([s["def"] for s in ss])
                per[u][r] = {"score": float((w * d).sum() / w.sum()), "n": len(ss)}
    z = {}
    for r in PROMPTS:
        vals = [per[u][r]["score"] for u in users if r in per[u]]
        mu, sd = np.mean(vals), np.std(vals) + 1e-9
        for u in users:
            if r in per[u]:
                z.setdefault(u, []).append((per[u][r]["score"] - mu) / sd)
    fit = {u: float(np.mean(v)) for u, v in z.items()}
    cov = {u: len(per[u]) for u in users}
    print("people with a fitness score: %d of %d (2+ regions: %d)" % (len(fit), len(users), sum(c >= 2 for c in cov.values())))

    # sheets: 12 highest / lowest among people with 2+ regions
    ranked = sorted([u for u in fit if cov[u] >= 2], key=lambda u: -fit[u])
    def sheet(us, path, cell=150, gap=3):
        im = Image.new("RGB", (4 * (3 * (cell + gap) + 14), 3 * (cell + gap) + gap), (245, 245, 242))
        for i, u in enumerate(us):
            ss = sorted([s for s in store if s["user"] == u and s["region"] in ("abs", "arms") and s["bare"] > 0.5],
                        key=lambda s: -s["bare"])
            tiles = []
            if ss:
                s = ss[0]; files = {os.path.splitext(f)[0]: f for f in os.listdir(os.path.join(root, "photos", u))}
                rgb = load_photo(os.path.join(root, "photos", u, files[s["code"]])); b = s["box"]
                tiles.append(Image.fromarray(rgb[b[1]:b[3], b[0]:b[2]]))
            rec = json.load(open(os.path.join(root, "cut", u, "owner.json")))
            for ph in rec["photos"][:3 - len(tiles)]:
                tiles.append(Image.open(os.path.join(root, "cut", u, ph["code"] + "_thumb.jpg")).convert("RGB"))
            x0 = (i % 4) * (3 * (cell + gap) + 14); y0 = (i // 4) * (cell + gap) + gap
            for j, t in enumerate(tiles):
                t = t.copy(); t.thumbnail((cell, cell))
                im.paste(t, (x0 + j * (cell + gap) + (cell - t.width) // 2, y0 + (cell - t.height) // 2))
        im.save(path, quality=85)
    sheet(ranked[:12], os.path.join(root, "results/fitness_top.jpg"))
    sheet(ranked[::-1][:12], os.path.join(root, "results/fitness_bottom.jpg"))

    # against the ratings: training accounts only
    follows = {l.split("\t")[0] for l in open(os.path.join(root, "links/following.tsv")) if l.strip() and not l.startswith(("#", "username"))}
    tu = [u for u in train if u in fit]
    y = np.array([first[u] for u in tu]); f = np.array([fit[u] for u in tu]); s_ = np.array([u not in follows for u in tu])
    out = {"n_scored_train": len(tu), "rho_all": float(spearmanr(f, y).correlation),
           "rho_strangers": float(spearmanr(f[s_], y[s_]).correlation)}
    for r in PROMPTS:
        ur = [u for u in train if r in per[u]]
        out["rho_" + r] = float(spearmanr([per[u][r]["score"] for u in ur], [first[u] for u in ur]).correlation)
        out["n_" + r] = len(ur)
    # does it add to the main model? (all training accounts; missing score -> mean, plus a coverage flag)
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(winner_features(F, idx, train)); yy = np.array([first[u] for u in train], float)
    fv = np.array([fit.get(u, np.nan) for u in train]); has = ~np.isnan(fv); fv[~has] = np.nanmean(fv)
    extra = np.c_[(fv - fv.mean()) / fv.std(), has.astype(float)] * 5
    ref, add = [], []
    for rep in range(3):
        fo = folds(train, group, seed=rep); p0 = np.zeros(len(yy)); p1 = np.zeros(len(yy))
        for k in range(5):
            a, b = fo != k, fo == k
            p0[b] = fit_predict(X[a], yy[a], X[b], 3000)
            p1[b] = fit_predict(np.c_[X[a], extra[a]], yy[a], np.c_[X[b], extra[b]], 3000)
        ref.append(spearmanr(p0, yy).correlation); add.append(spearmanr(p1, yy).correlation)
    out["cv_main_model"] = float(np.mean(ref)); out["cv_main_plus_fitness"] = float(np.mean(add))
    print(json.dumps(out, indent=1))
    json.dump({"validation": out, "fitness": fit, "coverage": cov, "per_region": per,
               "sheet_top": ranked[:12], "sheet_bottom": ranked[::-1][:12]},
              open(os.path.join(root, "results/fitness.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
