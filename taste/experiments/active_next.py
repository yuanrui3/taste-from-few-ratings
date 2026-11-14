#!/usr/bin/env python3
"""Who to rate next, after the 20 onboarding people (PROTOCOL.md "Who to rate next"). CPU only."""
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

A = np.logspace(-1, 5, 25)
BASE, STEP, TARGETS = 20, 5, (40, 60)
STRATS = ["herding", "random", "uncertainty", "disagreement", "top-predicted"]


def blend(zc, pr, n):
    w = n / (n + 20); return (1 - w) * rankdata(zc) / len(zc) + w * rankdata(pr) / len(pr)


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
    res = {m: {t: [] for t in TARGETS} for m in STRATS}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); P, E = np.where(isP)[0], np.where(~isP)[0]
        horder = [int(P[i]) for i in herding(Z[P], max(TARGETS))]
        for m in STRATS:
            rated = horder[:BASE]; rng = np.random.default_rng(77 + s)
            while len(rated) < max(TARGETS):
                cand = np.array([i for i in P if i not in set(rated)])
                if m == "herding":
                    new = [i for i in horder if i not in set(rated)][:STEP]
                elif m == "random":
                    new = list(rng.choice(cand, STEP, replace=False))
                else:
                    fit = RidgeCV(alphas=A).fit(X[rated], y[rated])
                    if m == "uncertainty":
                        new = []; cur = list(rated); a = fit.alpha_
                        for _ in range(STEP):
                            Xs = X[cur]; Kinv = np.linalg.inv(Xs @ Xs.T + a * np.eye(len(cur)))
                            kc = X[cand] @ Xs.T
                            var = (X[cand] ** 2).sum(1) - np.einsum("ij,jk,ik->i", kc, Kinv, kc)
                            for j, c in enumerate(cand):
                                if c in cur:
                                    var[j] = -np.inf
                            pick = int(cand[int(np.argmax(var))]); new.append(pick); cur.append(pick)
                    else:
                        pr = fit.predict(X[cand]); n = len(rated)
                        if m == "disagreement":
                            gap = np.abs(rankdata(zc[cand]) - rankdata(pr)); new = list(cand[np.argsort(-gap)[:STEP]])
                        else:
                            sc = blend(zc[cand], pr, n); new = list(cand[np.argsort(-sc)[:STEP]])
                rated = rated + [int(i) for i in new]
                if len(rated) in TARGETS:
                    pr = RidgeCV(alphas=A).fit(X[rated], y[rated]).predict(X[E])
                    res[m][len(rated)].append(spearmanr(blend(zc[E], pr, len(rated)), y[E]).correlation)
    out = {m: {t: float(np.mean(v)) for t, v in d.items()} for m, d in res.items()}
    out["wins_vs_herding"] = {m: {t: int(np.sum(np.array(res[m][t]) > np.array(res["herding"][t]))) for t in TARGETS} for m in STRATS[1:]}
    out["replace"] = {m: bool(all(out["wins_vs_herding"][m][t] >= 15 for t in TARGETS)) for m in STRATS[1:]}
    for m in STRATS:
        print("%-14s %s %s" % (m, "  ".join("n=%d %.3f" % (t, out[m][t]) for t in TARGETS),
                               "" if m == "herding" else "wins %s -> %s" % (out["wins_vs_herding"][m], "replace" if out["replace"][m] else "keep herding")))
    json.dump(out, open(os.path.join(root, "results/active_next.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
