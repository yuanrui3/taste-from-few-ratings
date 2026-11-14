#!/usr/bin/env python3
"""Large-model switch at 40+ ratings (PROTOCOL.md, post-hoc follow-up to round 1). CPU only."""
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

NS = [40, 80, 120]
A = np.logspace(-1, 5, 25)


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True))
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    XL = StandardScaler().fit_transform(winner_features(F, idx, allu))
    XS = StandardScaler().fit_transform(np.c_[winner_features(F, idx, allu), mean_vec(M, idx, allu, "siglip_nocloth"), mean_vec(M, idx, allu, "siglip_face")])
    Z32 = PCA(n_components=32, random_state=0).fit_transform(XL)
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    sc = StandardScaler().fit(S["emb"]); crowd = RidgeCV(alphas=A).fit(sc.transform(S["emb"]), S["mean"])
    zc = crowd.predict(sc.transform(mean_vec(F, idx, allu, "clip_face")))
    tr = np.array([pos[u] for u in train]); y = np.array([first[u] for u in train], float)
    XL, XS, Z32, zc = XL[tr], XS[tr], Z32[tr], zc[tr]
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid))
    res = {m: {n: [] for n in NS} for m in ("baseline", "switch")}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); P, E = np.where(isP)[0], np.where(~isP)[0]
        order = herding(Z32[P], max(NS))
        for n in NS:
            pk = P[order[:n]]; w = n / (n + 20); pc = rankdata(zc[E]) / len(E)
            for m, X in (("baseline", XL), ("switch", XS)):
                pr = RidgeCV(alphas=A).fit(X[pk], y[pk]).predict(X[E])
                res[m][n].append(spearmanr((1 - w) * pc + w * rankdata(pr) / len(pr), y[E]).correlation)
    out = {m: {n: float(np.mean(v)) for n, v in d.items()} for m, d in res.items()}
    out["wins"] = {n: int(np.sum(np.array(res["switch"][n]) > np.array(res["baseline"][n]))) for n in NS}
    out["adopt"] = bool(out["wins"][40] >= 15 and out["wins"][80] >= 15)
    for m in ("baseline", "switch"):
        print("%-9s %s" % (m, "  ".join("n=%d %.3f" % (n, out[m][n]) for n in NS)))
    print("switch wins:", out["wins"], "->", "adopt for n >= 40" if out["adopt"] else "keep baseline")
    json.dump(out, open(os.path.join(root, "results/switch_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
