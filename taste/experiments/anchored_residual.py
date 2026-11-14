#!/usr/bin/env python3
"""Crowd-anchored residual model (PROTOCOL.md, second exploration window). CPU only."""
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
from probe_prior import herding_fast  # noqa: E402
from probe_search import herding, winner_features  # noqa: E402
from train_eval import load, mean_vec, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
A = np.logspace(-1, 5, 25)


def zs(v, ref=None):
    ref = v if ref is None else ref; return (v - ref.mean()) / (ref.std() + 1e-12)


def both(Xk, yk, zk, XE, zE, n):
    """current rank blend and the anchored residual model; returns two prediction vectors for E."""
    w = n / (n + 20)
    pr = RidgeCV(alphas=A).fit(Xk, yk).predict(XE)
    cur = (1 - w) * rankdata(zE) / len(zE) + w * rankdata(pr) / len(pr)
    if yk.std() == 0:
        return cur, zE
    dev = zs(yk) - zk
    anc = zE + RidgeCV(alphas=A).fit(Xk, dev).predict(XE)
    return cur, anc


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    Xall = StandardScaler().fit_transform(winner_features(F, idx, allu)); Z32 = PCA(n_components=32, random_state=0).fit_transform(Xall)
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    scs = StandardScaler().fit(S["emb"]); crowd = RidgeCV(alphas=A).fit(scs.transform(S["emb"]), S["mean"])
    zc = zs(crowd.predict(scs.transform(mean_vec(F, idx, allu, "clip_face"))))
    tr = np.array([pos[u] for u in train]); y = np.array([first[u] for u in train], float)
    X, Zp, zc = Xall[tr], Z32[tr], zc[tr]
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid))
    U = {m: {n: [] for n in NS} for m in ("current", "anchored")}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Zp[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]
            zk = zs(zc[pk], zc[Pi]); zE = zs(zc[Ei], zc[Pi])
            cur, anc = both(X[pk], y[pk], zk, X[Ei], zE, n)
            U["current"][n].append(spearmanr(cur, y[Ei]).correlation); U["anchored"][n].append(spearmanr(anc, y[Ei]).correlation)
    out = {"user": {m: {n: float(np.nanmean(v)) for n, v in d.items()} for m, d in U.items()}}
    gain = float(np.mean([out["user"]["anchored"][n] - out["user"]["current"][n] for n in NS[:4]]))
    wins = int(np.sum(np.array(U["anchored"][20]) > np.array(U["current"][20])))
    out["verdict"] = {"gain_5_40": gain, "wins_n20": wins, "replace": bool(gain >= 0.02 and wins >= 15)}
    for m in U:
        print("user %-9s %s" % (m, "  ".join("n=%d %.3f" % (n, out["user"][m][n]) for n in NS)), flush=True)
    print("user: anchored gain %+.3f, wins %d/20 -> %s" % (gain, wins, "replace" if out["verdict"]["replace"] else "keep blend"), flush=True)
    # SCUT reference
    Xs = StandardScaler().fit_transform(S["emb"]); Zs = PCA(n_components=32, random_state=0).fit_transform(Xs); Rm = S["R"]
    NS2 = [5, 10, 20, 40]; Sres = {m: np.zeros((len(Rm), 5, len(NS2))) for m in ("current", "anchored")}
    for s in range(5):
        perm = np.random.default_rng(500 + s).permutation(Rm.shape[1]); P, E = perm[:int(0.7 * len(perm))], perm[int(0.7 * len(perm)):]
        Xc = Xs[P] - Xs[P].mean(0); Ue, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
        hord = herding_fast(Zs[P], max(NS2))
        for r in range(len(Rm)):
            others = np.nanmean(np.delete(Rm, r, 0), 0)
            a = 1000.0 if (r or s) else RidgeCV(alphas=A).fit(Xc, others[P] - others[P].mean()).alpha_
            if r == 0 and s == 0:
                a0 = a
            wv = Vt.T @ ((sv / (sv ** 2 + a0)) * (Ue.T @ (others[P] - others[P].mean())))
            zP_ = Xc @ wv; zE_ = (Xs[E] - Xs[P].mean(0)) @ wv
            for j, n in enumerate(NS2):
                pk = hord[:n]; yk = Rm[r][P][pk]
                cur, anc = both(Xs[P][pk], yk, zs(zP_[pk], zP_), Xs[E], zs(zE_, zP_), n)
                Sres["current"][r, s, j] = spearmanr(cur, Rm[r][E]).correlation; Sres["anchored"][r, s, j] = spearmanr(anc, Rm[r][E]).correlation
        print("SCUT split %d done" % s, flush=True)
    out["scut"] = {m: {n: float(np.nanmean(v[:, :, j])) for j, n in enumerate(NS2)} for m, v in Sres.items()}
    pr = {m: np.nanmean(v, 1) for m, v in Sres.items()}
    out["scut_wins_n20"] = int(np.sum(pr["anchored"][:, 2] > pr["current"][:, 2]))
    for m in Sres:
        print("SCUT %-9s %s" % (m, "  ".join("n=%d %.3f" % (n, out["scut"][m][n]) for n in NS2)))
    print("SCUT: anchored wins at n=20 for %d/60 raters" % out["scut_wins_n20"])
    json.dump(out, open(os.path.join(root, "results/anchored_residual.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
