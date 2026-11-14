#!/usr/bin/env python3
"""Coarser answers: like / pass, or low / mid / high (PROTOCOL.md). CPU only."""
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

NS = [5, 10, 20, 40, 80, 120]
A = np.logspace(-1, 5, 25)


def coarse(v, kind):
    if kind == "0-10":
        return v
    if kind == "like / pass":
        return (v > np.median(v)).astype(float)
    q = np.quantile(v, [1 / 3, 2 / 3]); return np.digitize(v, q, right=True).astype(float)


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    Xall = StandardScaler().fit_transform(winner_features(F, idx, allu)); Z32 = PCA(n_components=32, random_state=0).fit_transform(Xall)
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    scs = StandardScaler().fit(S["emb"]); crowd = RidgeCV(alphas=A).fit(scs.transform(S["emb"]), S["mean"])
    tr = np.array([pos[u] for u in train]); y = np.array([first[u] for u in train], float)
    X, Z, zc = Xall[tr], Z32[tr], crowd.predict(scs.transform(mean_vec(F, idx, allu, "clip_face")))[tr]
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid))
    kinds = ["0-10", "low / mid / high", "like / pass"]; res = {k: {n: [] for n in NS} for k in kinds}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Z[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]; w = n / (n + 20); pc = rankdata(zc[Ei]) / len(Ei)
            for k in kinds:
                yk = coarse(y[pk], k)
                pr = RidgeCV(alphas=A).fit(X[pk], yk).predict(X[Ei]) if len(set(yk)) > 1 else np.zeros(len(Ei))
                res[k][n].append(spearmanr((1 - w) * pc + w * rankdata(pr) / len(pr), y[Ei]).correlation)
    out = {"few": {k: {n: float(np.nanmean(v)) for n, v in d.items()} for k, d in res.items()}}
    for k in kinds:
        print("%-17s %s" % (k, "  ".join("n=%d %.3f" % (n, out["few"][k][n]) for n in NS)))
    out["verdict"] = {}
    for k in kinds[1:]:
        gap = float(np.mean([out["few"]["0-10"][n] - out["few"][k][n] for n in NS[:4]]))
        out["verdict"][k] = {"mean_gap_5_40": gap, "good_enough": bool(gap <= 0.02)}
        print("%-17s gap to 0-10 over n=5-40: %.3f -> %s" % (k, gap, "good enough" if gap <= 0.02 else "loses information"))
    target = out["few"]["0-10"][20]
    match = next((n for n in NS if out["few"]["like / pass"][n] >= target), None)
    out["binary_n_to_match_0_10_at_20"] = match
    print("like / pass reaches the 0-10 score at 20 ratings (%.3f) at n = %s" % (target, match))
    json.dump(out, open(os.path.join(root, "results/coarse_ratings.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
