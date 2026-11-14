#!/usr/bin/env python3
"""Which people to rate first when a crowd prior exists (see PROTOCOL.md). CPU only."""
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
from train_eval import load, photo_index  # noqa: E402

NS = [5, 10, 20, 40]
ALPHAS = np.logspace(-1, 5, 25)


def herding_fast(Z, n):
    """Same kernel herding as probe_search.herding, without the n x n x d intermediate."""
    sq = (Z ** 2).sum(1); D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0)
    s2 = np.median(D2[D2 > 0]); K = np.exp(-D2 / s2); mu = K.mean(1)
    acc = np.zeros(len(Z)); order = []
    for t in range(n):
        score = mu - acc / (t + 1); score[order] = -np.inf
        i = int(np.argmax(score)); order.append(i); acc += K[:, i]
    return order


def picks(method, Z, zc, n, seed, hord=None):
    """hord: herding order for this pool, computed once (greedy, so its first n are herding(Z, n))."""
    if method == "random":
        return list(np.random.default_rng(seed).choice(len(Z), n, replace=False))
    if method == "herding":
        return list(hord[:n]) if hord is not None else herding(Z, n)
    if method == "crowd-stratified":
        q = np.argsort(zc); bands = np.array_split(q, n); out = []
        for b in bands:
            c = Z[b].mean(0); out.append(int(b[np.argmin(np.linalg.norm(Z[b] - c, axis=1))]))
        return out
    if method == "crowd extremes":
        q = np.argsort(zc); return list(q[:n // 2]) + list(q[-(n - n // 2):])
    raise ValueError(method)


def blend(XS, yS, XE, zE, k=20):
    pc = rankdata(zE) / len(zE)
    if len(set(yS)) < 2:
        return pc
    pr = RidgeCV(alphas=ALPHAS).fit(XS, yS).predict(XE)
    w = len(yS) / (len(yS) + k)
    return (1 - w) * pc + w * rankdata(pr) / len(pr)


METHODS = ["random", "herding", "crowd-stratified", "crowd extremes"]


def scut(root):
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    X = StandardScaler().fit_transform(S["emb"]); Z = PCA(n_components=32, random_state=0).fit_transform(X)
    R = S["R"]; nr, ni = R.shape
    res = {m: {n: np.zeros((nr, 5)) for n in NS} for m in METHODS}
    for s in range(5):
        perm = np.random.default_rng(500 + s).permutation(ni); P, E = perm[:int(0.7 * ni)], perm[int(0.7 * ni):]
        hord = herding_fast(Z[P], max(NS))
        for r in range(nr):
            y = R[r]; others = np.nanmean(np.delete(R, r, 0), 0)
            pri = RidgeCV(alphas=ALPHAS).fit(X[P], others[P]); zP, zE = pri.predict(X[P]), pri.predict(X[E])
            for m in METHODS:
                for n in NS:
                    pk = picks(m, Z[P], zP, n, s * 1000 + r, hord)
                    res[m][n][r, s] = spearmanr(blend(X[P][pk], y[P][pk], X[E], zE), y[E]).correlation
        print("SCUT split %d done" % s, flush=True)
    return {m: {n: a.mean(1) for n, a in d.items()} for m, d in res.items()}


def user(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    users = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(users)}
    Xall = StandardScaler().fit_transform(winner_features(F, idx, users)); Zall = PCA(n_components=32, random_state=0).fit_transform(Xall)
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    sc = StandardScaler().fit(S["emb"]); crowd = RidgeCV(alphas=ALPHAS).fit(sc.transform(S["emb"]), S["mean"])
    face = np.stack([F["clip_face"][idx[u]].mean(0) for u in users]); face /= np.linalg.norm(face, axis=1, keepdims=True)
    zc_all = crowd.predict(sc.transform(face))
    tr = np.array([pos[u] for u in train]); y = np.array([first[u] for u in train], float)
    X, Z, zc = Xall[tr], Zall[tr], zc_all[tr]
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid))
    res = {m: {n: [] for n in NS} for m in METHODS}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); P, E = np.where(isP)[0], np.where(~isP)[0]
        hord = herding(Z[P], max(NS))
        for m in METHODS:
            for n in NS:
                vals = []
                for rep in (range(5) if m == "random" else [0]):
                    pk = picks(m, Z[P], zc[P], n, s * 100 + rep, hord)
                    vals.append(spearmanr(blend(X[P][pk], y[P][pk], X[E], zc[E]), y[E]).correlation)
                res[m][n].append(float(np.mean(vals)))
    return {m: {n: np.array(v) for n, v in d.items()} for m, d in res.items()}


def main(root):
    A = scut(root); B = user(root)
    out = {"scut": {m: {n: float(v.mean()) for n, v in d.items()} for m, d in A.items()},
           "user": {m: {n: float(v.mean()) for n, v in d.items()} for m, d in B.items()}}
    print("\n%-18s %s   |   %s" % ("method", "SCUT n=" + "/".join(map(str, NS)), "user n=" + "/".join(map(str, NS))))
    for m in METHODS:
        print("%-18s %s   |   %s" % (m, " ".join("%.3f" % out["scut"][m][n] for n in NS), " ".join("%.3f" % out["user"][m][n] for n in NS)))
    verdict = {}
    for m in METHODS:
        if m == "herding":
            continue
        ws = int(np.sum(A[m][20] > A["herding"][20])); wu = int(np.sum(B[m][20] > B["herding"][20]))
        verdict[m] = {"scut_wins_n20": ws, "user_wins_n20": wu, "replaces_herding": bool(ws >= 45 and wu >= 12)}
        print("%-18s beats herding at n=20: SCUT %d/60, user %d/20 -> %s" % (m, ws, wu, "replace" if verdict[m]["replaces_herding"] else "keep herding"))
    out["verdict"] = verdict
    json.dump(out, open(os.path.join(root, "results/probe_prior.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
