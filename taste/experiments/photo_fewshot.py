#!/usr/bin/env python3
"""Few-rating model trained on single photos (PROTOCOL.md). CPU only."""
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


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    Xall = StandardScaler().fit_transform(winner_features(F, idx, allu)); Z32 = PCA(n_components=32, random_state=0).fit_transform(Xall)
    Pall = np.c_[F["clip_nocloth"], F["clip_face"]]; psc = StandardScaler().fit(Pall); Pall = psc.transform(Pall)
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    scs = StandardScaler().fit(S["emb"]); crowd = RidgeCV(alphas=A).fit(scs.transform(S["emb"]), S["mean"])
    tr = np.array([pos[u] for u in train]); y = np.array([first[u] for u in train], float)
    X, Z, zc = Xall[tr], Z32[tr], crowd.predict(scs.transform(mean_vec(F, idx, allu, "clip_face")))[tr]
    rows = [np.array(idx[u]) for u in train]
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid))
    res = {m: {n: [] for n in NS} for m in ("baseline", "photo-level")}
    def kridge(Xtr, ytr):
        """Ridge with intercept in kernel form; returns (predict(Xte, alpha) function) after one eigh."""
        xm, ym = Xtr.mean(0), ytr.mean(); Xc = Xtr - xm
        l, Q = np.linalg.eigh(Xc @ Xc.T); qy = Q.T @ (ytr - ym)
        def pred(Xte, a):
            c = Q @ (qy / (l + a)); return ym + (Xte - xm) @ (Xc.T @ c)
        return pred
    def photo_model(pk, Ei):
        Xp = np.concatenate([Pall[rows[i]] for i in pk]); yp = np.concatenate([[y[i]] * len(rows[i]) for i in pk])
        who = np.concatenate([[j] * len(rows[i]) for j, i in enumerate(pk)])
        err = np.zeros(len(A))
        for j in range(len(pk)):  # leave-one-person-out error of the person-mean prediction
            f = kridge(Xp[who != j], yp[who != j]); Xj = Xp[who == j]
            err += np.array([(f(Xj, a).mean() - y[pk[j]]) ** 2 for a in A])
        f = kridge(Xp, yp); a = A[int(np.argmin(err))]
        return np.array([f(Pall[rows[i]], a).mean() for i in Ei])
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Z[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]; w = n / (n + 20); pc = rankdata(zc[Ei]) / len(Ei)
            prs = {"baseline": RidgeCV(alphas=A).fit(X[pk], y[pk]).predict(X[Ei]), "photo-level": photo_model(pk, Ei)}
            for m, pr in prs.items():
                res[m][n].append(spearmanr((1 - w) * pc + w * rankdata(pr) / len(pr), y[Ei]).correlation)
        print("split %d done" % s, flush=True)
    out = {m: {n: float(np.nanmean(v)) for n, v in d.items()} for m, d in res.items()}
    gain = float(np.mean([out["photo-level"][n] - out["baseline"][n] for n in NS[:4]]))
    wins = int(np.sum(np.array(res["photo-level"][20]) > np.array(res["baseline"][20])))
    out["verdict"] = {"gain_5_40": gain, "wins_n20": wins, "replace": bool(gain >= 0.02 and wins >= 15)}
    for m in ("baseline", "photo-level"):
        print("%-12s %s" % (m, "  ".join("n=%d %.3f" % (n, out[m][n]) for n in NS)))
    print("photo-level: gain %+.3f, wins %d/20 -> %s" % (gain, wins, "replace" if out["verdict"]["replace"] else "keep"))
    json.dump(out, open(os.path.join(root, "results/photo_fewshot.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
