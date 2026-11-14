#!/usr/bin/env python3
"""Typicality as a free prior (PROTOCOL.md). CPU only."""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from probe_search import herding, winner_features  # noqa: E402
from train_eval import load, mean_vec, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
A = np.logspace(-1, 5, 25)


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    tr = np.array([pos[u] for u in train]); y = np.array([first[u] for u in train], float)
    typ = {}
    for key in ("clip_face", "clip_nocloth"):
        V = mean_vec(F, idx, allu, key); c = V.mean(0); c /= np.linalg.norm(c); typ[key] = zs(V @ c)
    typ["both"] = zs(typ["clip_face"] + typ["clip_nocloth"])
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    scs = StandardScaler().fit(S["emb"]); crowd = RidgeCV(alphas=A).fit(scs.transform(S["emb"]), S["mean"])
    zsc = zs(crowd.predict(scs.transform(mean_vec(F, idx, allu, "clip_face"))))
    out = {"zero_rating": {"SCUT": float(spearmanr(zsc[tr], y).correlation)}}
    for k, v in typ.items():
        out["zero_rating"]["typicality " + k] = float(spearmanr(v[tr], y).correlation)
    bestk = max(typ, key=lambda k: out["zero_rating"]["typicality " + k])
    comb = zs(zsc + typ[bestk]); out["zero_rating"]["SCUT + typicality " + bestk] = float(spearmanr(comb[tr], y).correlation)
    out["corr_scut_typicality"] = float(spearmanr(zsc, typ[bestk]).correlation)
    for k, v in out["zero_rating"].items():
        print("%-34s zero-rating %.3f" % (k, v), flush=True)
    print("typicality vs SCUT score correlation: %.3f" % out["corr_scut_typicality"])
    Xall = StandardScaler().fit_transform(winner_features(F, idx, allu)); Z32 = PCA(n_components=32, random_state=0).fit_transform(Xall)
    X, Z = Xall[tr], Z32[tr]; pri = {"SCUT": zsc[tr], "SCUT + typicality": comb[tr]}
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid)); res = {m: {n: [] for n in NS} for m in pri}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Z[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]; w = n / (n + 20); pr = RidgeCV(alphas=A).fit(X[pk], y[pk]).predict(X[Ei])
            for m, zc in pri.items():
                res[m][n].append(spearmanr((1 - w) * rankdata(zc[Ei]) / len(Ei) + w * rankdata(pr) / len(pr), y[Ei]).correlation)
    out["few"] = {m: {n: float(np.mean(v)) for n, v in d.items()} for m, d in res.items()}
    gain = out["zero_rating"]["SCUT + typicality " + bestk] - out["zero_rating"]["SCUT"]
    wins = int(np.sum(np.array(res["SCUT + typicality"][20]) > np.array(res["SCUT"][20])))
    out["verdict"] = {"best": bestk, "zero_gain": gain, "wins_n20": wins, "adopt": bool(gain >= 0.03 and wins >= 15)}
    for m in pri:
        print("%-20s %s" % (m, "  ".join("n=%d %.3f" % (n, out["few"][m][n]) for n in NS)))
    print("typicality (%s): zero-rating gain %+.3f, wins %d/20 -> %s" % (bestk, gain, wins, "adopt" if out["verdict"]["adopt"] else "keep SCUT"))
    json.dump(out, open(os.path.join(root, "results/prior_typical.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
