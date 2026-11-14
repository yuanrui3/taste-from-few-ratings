#!/usr/bin/env python3
"""A second crowd through face-recognition codes: MEBeauty on SFace (PROTOCOL.md, afternoon E). CPU only.
  mebeauty_sface.py embed I N   -> datasets/mebeauty/sface_part_I.npz (chunk I of N)
  mebeauty_sface.py eval        -> results/mebeauty_sface.json
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]  # experiments/, pipeline, code/ (idscore)

NS = [5, 10, 20, 40, 80]
A = np.logspace(-1, 5, 25)


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def embed(root, i, n):
    from idscore.faceid import FaceID
    me = np.load(os.path.join(root, "datasets/mebeauty/me.npz"), allow_pickle=True)
    paths = [str(p) for p in me["paths"]]; mine = list(range(i, len(paths), n))
    fid = FaceID(os.path.join(root, "models", "opencv_zoo")); out = np.full((len(mine), 128), np.nan)
    for j, k in enumerate(mine):
        p = paths[k] if os.path.isabs(paths[k]) else os.path.join(root, paths[k])
        rgb = np.array(Image.open(p).convert("RGB")); row = fid.detect(rgb)
        e, _ = fid.embed(rgb, row) if row is not None else fid.embed(rgb, None, (0, 0, rgb.shape[1], rgb.shape[0]))
        if e is not None:
            out[j] = e
    np.savez_compressed(os.path.join(root, "datasets/mebeauty/sface_part_%d.npz" % i), idx=np.array(mine), emb=out)
    print("chunk %d done: %d faces" % (i, len(mine)), flush=True)


def evaluate(root):
    from probe_search import herding, winner_features
    from sface_input import sface_means
    from taste_onboard import crowd_score
    from train_eval import load, photo_index
    me = np.load(os.path.join(root, "datasets/mebeauty/me.npz"), allow_pickle=True); E = np.full((len(me["paths"]), 128), np.nan)
    for f in glob.glob(os.path.join(root, "datasets/mebeauty/sface_part_*.npz")):
        d = np.load(f); E[d["idx"]] = d["emb"]
    ok = ~np.isnan(E).any(1); y_me = me["mean"].astype(float)
    print("MEBeauty faces with an SFace code: %d / %d" % (ok.sum(), len(ok)), flush=True)
    F, train, test, first, R, group, split, same = load(root); idx = photo_index(F)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    sc = StandardScaler().fit(E[ok]); zm = zs(RidgeCV(alphas=A).fit(sc.transform(E[ok]), y_me[ok]).predict(sc.transform(sface_means(root, allu))))
    z2 = crowd_score(root, F, idx, allu); z3 = zs(z2 * np.sqrt(2) + zm)
    tr = np.array([pos[u] for u in train]); te = np.array([pos[u] for u in test])
    ytr = np.array([first[u] for u in train], float); yte = np.array([first[u] for u in test], float)
    out = {"zero": {}}
    for nm, z in (("CLIP + SFace (adopted)", z2), ("MEBeauty on SFace alone", zm), ("adopted + MEBeauty SFace", z3)):
        out["zero"][nm] = {"train": float(spearmanr(z[tr], ytr).correlation), "test": float(spearmanr(z[te], yte).correlation)}
        print("%-26s zero-rating: training %.3f, test %.3f" % (nm, out["zero"][nm]["train"], out["zero"][nm]["test"]), flush=True)
    XL = StandardScaler().fit_transform(winner_features(F, idx, allu)); Z = PCA(n_components=32, random_state=0).fit_transform(XL)[tr]; X = XL[tr]
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid)); res = {m: {n: [] for n in NS} for m in ("adopted", "with MEBeauty")}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Z[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]; w = n / (n + 20); pr = rankdata(RidgeCV(alphas=A).fit(X[pk], ytr[pk]).predict(X[Ei])) / len(Ei)
            for m, zz in (("adopted", z2[tr]), ("with MEBeauty", z3[tr])):
                res[m][n].append(spearmanr((1 - w) * rankdata(zz[Ei]) / len(Ei) + w * pr, ytr[Ei]).correlation)
    out["few"] = {m: {n: float(np.mean(v)) for n, v in d.items()} for m, d in res.items()}
    gtr = out["zero"]["adopted + MEBeauty SFace"]["train"] - out["zero"]["CLIP + SFace (adopted)"]["train"]
    gte = out["zero"]["adopted + MEBeauty SFace"]["test"] - out["zero"]["CLIP + SFace (adopted)"]["test"]
    wins = int(np.sum(np.array(res["with MEBeauty"][20]) > np.array(res["adopted"][20])))
    out["verdict"] = {"zero_gain_train": gtr, "zero_gain_test": gte, "wins_n20": wins, "adopt": bool(gtr >= 0.03 and wins >= 15 and gte > 0)}
    for m in res:
        print("%-14s %s" % (m, "  ".join("n=%d %.3f" % (n, out["few"][m][n]) for n in NS)))
    print("MEBeauty on SFace: zero-rating %+.3f (test %+.3f), wins %d/20 -> %s" % (gtr, gte, wins, "adopt" if out["verdict"]["adopt"] else "no"))
    json.dump(out, open(os.path.join(root, "results/mebeauty_sface.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        embed(sys.argv[4] if len(sys.argv) > 4 else ".", int(sys.argv[2]), int(sys.argv[3]))
    else:
        evaluate(sys.argv[2] if len(sys.argv) > 2 else ".")
