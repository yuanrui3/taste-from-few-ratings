#!/usr/bin/env python3
"""Crowd model on face-recognition (SFace) codes (PROTOCOL.md, second exploration window). CPU only."""
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
from probe_search import herding, winner_features  # noqa: E402
from sface_input import sface_means  # noqa: E402
from train_eval import load, mean_vec, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
A = np.logspace(-1, 5, 25)


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def scut_sface(root, names):
    path = os.path.join(root, "datasets/scut/scut_sface.npz")
    if os.path.exists(path):
        return np.load(path)["emb"]
    from idscore.faceid import FaceID
    fid = FaceID(os.path.join(root, "models", "opencv_zoo"))
    base = os.path.join(root, "datasets/scut/SCUT-FBP5500_v2/Images"); out = np.full((len(names), 128), np.nan)
    for i, n in enumerate(names):
        rgb = np.array(Image.open(os.path.join(base, n)).convert("RGB")); row = fid.detect(rgb)
        e, _ = fid.embed(rgb, row) if row is not None else fid.embed(rgb, None, (0, 0, rgb.shape[1], rgb.shape[0]))
        if e is not None:
            out[i] = e
        if i % 1000 == 0:
            print("SCUT SFace", i, flush=True)
    np.savez_compressed(path, emb=out); return out


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    E = scut_sface(root, [str(n) for n in S["names"]]); ok = ~np.isnan(E).any(1)
    print("SCUT faces with an SFace code: %d / %d" % (ok.sum(), len(ok)), flush=True)
    sc = StandardScaler().fit(E[ok]); m = RidgeCV(alphas=A).fit(sc.transform(E[ok]), S["mean"][ok])
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    zsf = zs(m.predict(sc.transform(sface_means(root, allu))))
    scs = StandardScaler().fit(S["emb"]); crowd = RidgeCV(alphas=A).fit(scs.transform(S["emb"]), S["mean"])
    zcl = zs(crowd.predict(scs.transform(mean_vec(F, idx, allu, "clip_face"))))
    pri = {"current (CLIP)": zcl, "SFace crowd": zsf, "SFace + current": zs(zsf + zcl)}
    tr = np.array([pos[u] for u in train]); te = np.array([pos[u] for u in test])
    ytr = np.array([first[u] for u in train], float); yte = np.array([first[u] for u in test], float)
    out = {"zero_train": {k: float(spearmanr(v[tr], ytr).correlation) for k, v in pri.items()},
           "zero_test": {k: float(spearmanr(v[te], yte).correlation) for k, v in pri.items()}}
    for k in pri:
        print("%-16s zero-rating: training %.3f, test %.3f" % (k, out["zero_train"][k], out["zero_test"][k]), flush=True)
    best = max(("SFace crowd", "SFace + current"), key=lambda k: out["zero_train"][k])
    Xall = StandardScaler().fit_transform(winner_features(F, idx, allu)); Z32 = PCA(n_components=32, random_state=0).fit_transform(Xall)
    X, Z = Xall[tr], Z32[tr]; P2 = {"current (CLIP)": zcl[tr], best: pri[best][tr]}
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid)); res = {k: {n: [] for n in NS} for k in P2}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Z[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]; w = n / (n + 20); pr = RidgeCV(alphas=A).fit(X[pk], ytr[pk]).predict(X[Ei])
            for k, zc in P2.items():
                res[k][n].append(spearmanr((1 - w) * rankdata(zc[Ei]) / len(Ei) + w * rankdata(pr) / len(pr), ytr[Ei]).correlation)
    out["few"] = {k: {n: float(np.mean(v)) for n, v in d.items()} for k, d in res.items()}
    for k in P2:
        print("%-16s %s" % (k, "  ".join("n=%d %.3f" % (n, out["few"][k][n]) for n in NS)))
    g_tr = out["zero_train"][best] - out["zero_train"]["current (CLIP)"]; g_te = out["zero_test"][best] - out["zero_test"]["current (CLIP)"]
    wins = int(np.sum(np.array(res[best][20]) > np.array(res["current (CLIP)"][20])))
    out["verdict"] = {"best": best, "zero_gain_train": g_tr, "zero_gain_test": g_te, "wins_n20": wins,
                      "adopt": bool(g_tr >= 0.03 and wins >= 15 and g_te > 0)}
    print("%s: training %+.3f, test %+.3f, wins %d/20 -> %s" % (best, g_tr, g_te, wins, "adopt" if out["verdict"]["adopt"] else "keep current"))
    json.dump(out, open(os.path.join(root, "results/sface_crowd.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
