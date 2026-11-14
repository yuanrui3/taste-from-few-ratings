#!/usr/bin/env python3
"""Semi-supervised few-rating model: Laplacian-regularised least squares on the pool's neighbour
graph (PROTOCOL.md). CPU only."""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from probe_search import herding, winner_features  # noqa: E402
from train_eval import load, mean_vec, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
A = np.logspace(-1, 5, 25)
AL = np.logspace(-3, 3, 13)
GS = [0.0, 0.01, 0.1, 1.0]


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    Xall = StandardScaler().fit_transform(winner_features(F, idx, allu)); Z32 = PCA(n_components=32, random_state=0).fit_transform(Xall)
    N = len(allu); K = Xall @ Xall.T / Xall.shape[1]
    W = kneighbors_graph(Z32, 10, mode="connectivity", include_self=False).toarray(); W = np.maximum(W, W.T)
    L = np.diag(W.sum(1)) - W; L = L / np.trace(L) * N  # scale so that g is comparable across graphs
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    scs = StandardScaler().fit(S["emb"]); crowd = RidgeCV(alphas=A).fit(scs.transform(S["emb"]), S["mean"])
    zc_all = crowd.predict(scs.transform(mean_vec(F, idx, allu, "clip_face")))
    tr = np.array([pos[u] for u in train]); y_all = np.full(N, np.nan); y_all[tr] = [first[u] for u in train]
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid))
    res = {m: {n: [] for n in NS} for m in ("baseline", "LapRLS")}; chosen_g = {n: [] for n in NS}
    for s in range(20):
        g0 = np.random.default_rng(1000 + s); gs = ug[:]; g0.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = tr[isP], tr[~isP]
        o = herding(Z32[Pi], max(NS))
        for n in NS:
            lab = Pi[o[:n]]; y = y_all[lab]; ym = y.mean(); w = n / (n + 20); pc = rankdata(zc_all[Ei]) / len(Ei)
            pr = RidgeCV(alphas=A).fit(Xall[lab], y).predict(Xall[Ei])
            res["baseline"][n].append(spearmanr((1 - w) * pc + w * rankdata(pr) / len(pr), y_all[Ei]).correlation)
            J = np.zeros(N); J[lab] = 1; JK = J[:, None] * K; Y = np.zeros(N); Y[lab] = y - ym
            best = None
            for g in GS:
                LK = g * (L @ K)
                for a in AL:
                    Minv = np.linalg.inv(JK + a * np.eye(N) + LK)
                    H = K[lab] @ Minv[:, lab]; r = (y - ym) - H @ (y - ym); e = np.mean((r / (1 - np.diag(H) + 1e-12)) ** 2)
                    if best is None or e < best[0]:
                        best = (e, g, a, Minv)
            c = best[3] @ Y; f = K @ c + ym
            chosen_g[n].append(best[1])
            res["LapRLS"][n].append(spearmanr((1 - w) * pc + w * rankdata(f[Ei]) / len(Ei), y_all[Ei]).correlation)
        print("split %d done" % s, flush=True)
    out = {"few": {m: {n: float(np.nanmean(v)) for n, v in d.items()} for m, d in res.items()},
           "chosen_g": {n: [float(x) for x in v] for n, v in chosen_g.items()}}
    gain = float(np.mean([out["few"]["LapRLS"][n] - out["few"]["baseline"][n] for n in NS[:4]]))
    wins = int(np.sum(np.array(res["LapRLS"][20]) > np.array(res["baseline"][20])))
    out["verdict"] = {"gain_5_40": gain, "wins_n20": wins, "replace": bool(gain >= 0.02 and wins >= 15)}
    for m in res:
        print("%-9s %s" % (m, "  ".join("n=%d %.3f" % (n, out["few"][m][n]) for n in NS)))
    print("graph weight chosen (share > 0):", {n: float(np.mean(np.array(v) > 0)) for n, v in chosen_g.items()})
    print("LapRLS: gain %+.3f, wins %d/20 -> %s" % (gain, wins, "replace" if out["verdict"]["replace"] else "keep"))
    json.dump(out, open(os.path.join(root, "results/laprls.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
