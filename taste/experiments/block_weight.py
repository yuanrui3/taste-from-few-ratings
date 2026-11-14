#!/usr/bin/env python3
"""Face / body weighting in the few-rating model (PROTOCOL.md). CPU only."""
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
VARIANTS = {"face only": (0, 1), "face x2": (1, 2), "equal": (1, 1), "body x2": (2, 1), "body only": (1, 0)}


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    Xall = StandardScaler().fit_transform(winner_features(F, idx, allu)); Z32 = PCA(n_components=32, random_state=0).fit_transform(Xall)
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    scs = StandardScaler().fit(S["emb"]); crowd = RidgeCV(alphas=A).fit(scs.transform(S["emb"]), S["mean"])
    tr = np.array([pos[u] for u in train]); y = np.array([first[u] for u in train], float)
    X, Z, zc = Xall[tr], Z32[tr], crowd.predict(scs.transform(mean_vec(F, idx, allu, "clip_face")))[tr]
    d = X.shape[1] // 2
    feats = {}
    for nm, (wb, wf) in VARIANTS.items():
        cols = ([X[:, :d] * wb] if wb else []) + ([X[:, d:] * wf] if wf else []); feats[nm] = np.concatenate(cols, 1)
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid)); res = {m: {n: [] for n in NS} for m in feats}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Z[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]; w = n / (n + 20); pc = rankdata(zc[Ei]) / len(Ei)
            for m, Xm in feats.items():
                pr = RidgeCV(alphas=A).fit(Xm[pk], y[pk]).predict(Xm[Ei])
                res[m][n].append(spearmanr((1 - w) * pc + w * rankdata(pr) / len(pr), y[Ei]).correlation)
    out = {"few": {m: {n: float(np.mean(v)) for n, v in dd.items()} for m, dd in res.items()}}
    for m in feats:
        print("%-10s %s" % (m, "  ".join("n=%d %.3f" % (n, out["few"][m][n]) for n in NS)))
    alts = [m for m in feats if m != "equal"]
    gains = {m: float(np.mean([out["few"][m][n] - out["few"]["equal"][n] for n in NS[:4]])) for m in alts}
    best = max(gains, key=gains.get); wins = int(np.sum(np.array(res[best][20]) > np.array(res["equal"][20])))
    out["verdict"] = {"best": best, "gain_5_40": gains[best], "wins_n20": wins, "replace": bool(gains[best] >= 0.02 and wins >= 15), "gains": gains}
    print("best %s: gain %+.3f, wins %d/20 -> %s" % (best, gains[best], wins, "replace" if out["verdict"]["replace"] else "keep equal"))
    json.dump(out, open(os.path.join(root, "results/block_weight.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
