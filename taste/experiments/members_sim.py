#!/usr/bin/env python3
"""How many members, rating how many people? SCUT simulation (PROTOCOL.md, afternoon D). CPU only."""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from multi_rater_scut import loo_mse  # noqa: E402
from probe_prior import herding_fast  # noqa: E402

A = np.logspace(-1, 5, 25)
MS, RS, NS, KS = [5, 10, 20], [40, 80, 160], [10, 20, 40], [1, 2, 5]


def zs(v, ref):
    return (v - ref.mean()) / (ref.std() + 1e-12)


def main(root):
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    X = StandardScaler().fit_transform(S["emb"]); Z = PCA(n_components=32, random_state=0).fit_transform(X); Rm = S["R"]; nr = len(Rm)
    res = {(M, R_): {n: np.zeros((nr, 5)) for n in NS} for M in MS for R_ in RS}; cur = {n: np.zeros((nr, 5)) for n in NS}
    for s in range(5):
        perm = np.random.default_rng(500 + s).permutation(Rm.shape[1]); P, E = perm[:int(0.7 * len(perm))], perm[int(0.7 * len(perm)):]
        hord = np.array(herding_fast(Z[P], max(RS))); mu = X[P].mean(0); Xc, Ec = X[P] - mu, X[E] - mu
        a = RidgeCV(alphas=A).fit(Xc, np.nanmean(Rm, 0)[P]).alpha_
        rng = np.random.default_rng(900 + s)
        for r in range(nr):
            others = [q for q in range(nr) if q != r]
            crowd = Ridge(alpha=a).fit(Xc, np.nanmean(Rm[others], 0)[P]); zP, zE = crowd.predict(Xc), crowd.predict(Ec)
            y = Rm[r]
            for n in NS:
                pk = hord[:n]; w = n / (n + 20); pr = RidgeCV(alphas=A).fit(X[P][pk], y[P][pk]).predict(X[E])
                cur[n][r, s] = spearmanr((1 - w) * rankdata(zE) / len(E) + w * rankdata(pr) / len(pr), y[E]).correlation
            for M in MS:
                mem = rng.choice(others, M, replace=False)
                for R_ in RS:
                    rows = hord[:R_]
                    W = np.stack([Ridge(alpha=a).fit(Xc[rows], Rm[q][P][rows] - Rm[q][P][rows].mean()).coef_ for q in mem])
                    pcs = PCA(n_components=min(max(KS), M - 1), random_state=0).fit(W - W.mean(0)).components_
                    for n in NS:
                        pk = hord[:n]; yy = y[P][pk]
                        ys = (yy - yy.mean()) / yy.std() if yy.std() > 0 else yy * 0
                        dev = ys - zs(zP[pk], zP); best = None
                        for k in KS:
                            if k > len(pcs) or k >= n:
                                continue
                            e, m = loo_mse(Xc[pk] @ pcs[:k].T, dev)
                            if best is None or e < best[0]:
                                best = (e, k, m)
                        pe = zs(zE, zP) + (best[2].predict(Ec @ pcs[:best[1]].T) if best else 0)
                        res[(M, R_)][n][r, s] = spearmanr(pe, y[E]).correlation
        print("split %d done" % s, flush=True)
    out = {"current": {n: float(np.nanmean(v)) for n, v in cur.items()}, "grid": {}}
    curr = {n: np.nanmean(v, 1) for n, v in cur.items()}
    print("current blend: %s" % "  ".join("n=%d %.3f" % (n, out["current"][n]) for n in NS))
    best = None
    for M in MS:
        for R_ in RS:
            d = {n: float(np.nanmean(v)) for n, v in res[(M, R_)].items()}
            wins = int(np.sum(np.nanmean(res[(M, R_)][20], 1) > curr[20]))
            out["grid"]["M=%d R=%d" % (M, R_)] = {"few": d, "wins_n20": wins}
            print("members %2d x %3d ratings: %s   wins at n=20: %d/60" % (M, R_, "  ".join("n=%d %.3f" % (n, d[n]) for n in NS), wins), flush=True)
            if wins >= 45 and (best is None or M * R_ < best[0]):
                best = (M * R_, M, R_)
    out["recommended"] = None if best is None else {"members": best[1], "ratings_each": best[2]}
    print("smallest members x ratings that helps 45+/60 raters at n=20:", out["recommended"])
    json.dump(out, open(os.path.join(root, "results/members_sim.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
