#!/usr/bin/env python3
"""Do SCUT's shared taste directions carry over to this rater? (PROTOCOL.md, --user test). CPU only."""
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
from multi_rater_scut import loo_mse  # noqa: E402
from probe_search import herding, winner_features  # noqa: E402
from train_eval import load, mean_vec, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
A = np.logspace(-1, 5, 25)


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    scs = StandardScaler().fit(S["emb"]); Xs = scs.transform(S["emb"]); mu = Xs.mean(0); Xc = Xs - mu
    a = RidgeCV(alphas=A).fit(Xc, S["mean"]).alpha_
    U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = np.nan_to_num(S["R"] - np.nanmean(S["R"], 1, keepdims=True))
    W = (Vt.T * (sv / (sv ** 2 + a))) @ (U.T @ Y.T)  # 768 x 60
    wm = W.mean(1); pcs = PCA(n_components=10, random_state=0).fit((W - wm[:, None]).T).components_
    face = scs.transform(mean_vec(F, idx, allu, "clip_face")) - mu
    tr = np.array([pos[u] for u in train]); y = np.array([first[u] for u in train], float)
    z = face @ wm; z = (z - z.mean()) / z.std(); Bf = face @ pcs.T
    Xall = StandardScaler().fit_transform(winner_features(F, idx, allu)); Z32 = PCA(n_components=32, random_state=0).fit_transform(Xall)
    X, Zp, z, Bf = Xall[tr], Z32[tr], z[tr], Bf[tr]
    print("anchor alone (SCUT mean direction) zero-rating: %.3f" % spearmanr(z, y).correlation, flush=True)
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid)); res = {m: {n: [] for n in NS} for m in ("current", "anchored SCUT directions")}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Zp[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]; yy = y[pk]; w = n / (n + 20)
            pr = RidgeCV(alphas=A).fit(X[pk], yy).predict(X[Ei])
            res["current"][n].append(spearmanr((1 - w) * rankdata(z[Ei]) / len(Ei) + w * rankdata(pr) / len(pr), y[Ei]).correlation)
            ys = (yy - yy.mean()) / (yy.std() + 1e-12) if yy.std() > 0 else yy * 0
            dev = ys - z[pk]; best = None
            for k in (2, 5, 10):
                if k >= n:
                    continue
                e, m = loo_mse(Bf[pk][:, :k], dev)
                if best is None or e < best[0]:
                    best = (e, k, m)
            pe = z[Ei] if best is None else z[Ei] + best[2].predict(Bf[Ei][:, :best[1]])
            res["anchored SCUT directions"][n].append(spearmanr(pe, y[Ei]).correlation)
    out = {m: {n: float(np.nanmean(v)) for n, v in d.items()} for m, d in res.items()}
    gain = float(np.mean([out["anchored SCUT directions"][n] - out["current"][n] for n in NS[:4]]))
    wins = int(np.sum(np.array(res["anchored SCUT directions"][20]) > np.array(res["current"][20])))
    out["verdict"] = {"gain_5_40": gain, "wins_n20": wins, "carries_over": bool(gain >= 0.02 and wins >= 15)}
    for m in res:
        print("%-26s %s" % (m, "  ".join("n=%d %.3f" % (n, out[m][n]) for n in NS)))
    print("SCUT directions on this rater: gain %+.3f, wins %d/20 -> %s" % (gain, wins, "carry over" if out["verdict"]["carries_over"] else "do not carry over"))
    json.dump(out, open(os.path.join(root, "results/multi_rater_user.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
