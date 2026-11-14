#!/usr/bin/env python3
"""Shared taste dimensions from other raters, simulated on SCUT's 60 raters (PROTOCOL.md). CPU only."""
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

NS = [5, 10, 20, 40]
A = np.logspace(-1, 5, 25)
KS = [0, 2, 5, 10]


def loo_mse(B, y):
    m = RidgeCV(alphas=np.logspace(-3, 3, 13), store_cv_results=True).fit(B, y)
    return float(np.min(m.cv_results_.mean(0))), m


def main(root):
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    X = StandardScaler().fit_transform(S["emb"]); Z = PCA(n_components=32, random_state=0).fit_transform(X)
    R = S["R"]; nr, ni = R.shape
    ANCH = "--anchored" in sys.argv
    names = ("current", "shared dimensions") + (("anchored",) if ANCH else ())
    res = {m: np.zeros((nr, 5, len(NS))) for m in names}; kchosen = []
    for s in range(5):
        perm = np.random.default_rng(500 + s).permutation(ni); P, E = perm[:int(0.7 * ni)], perm[int(0.7 * ni):]
        Xp = X[P]; mu = Xp.mean(0); Xc = Xp - mu; Ec = X[E] - mu
        a = RidgeCV(alphas=A).fit(Xc, np.nanmean(R, 0)[P]).alpha_
        U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
        Yp = np.nan_to_num(R[:, P] - np.nanmean(R[:, P], 1, keepdims=True))
        Wall = (Vt.T * (sv / (sv ** 2 + a))) @ (U.T @ Yp.T)  # 768 x 60: every rater's ridge weights
        hord = herding_fast(Z[P], max(NS))
        for r in range(nr):
            Wo = np.delete(Wall, r, 1); wm = Wo.mean(1)
            pcs = PCA(n_components=max(KS), random_state=0).fit((Wo - wm[:, None]).T).components_  # k x 768
            basis = np.c_[wm, pcs.T]  # 768 x (1 + 10)
            y = R[r]; zP, zE = Xc @ wm, Ec @ wm
            for j, n in enumerate(NS):
                pk = hord[:n]; yy = y[P][pk]
                # current method
                pr = RidgeCV(alphas=A).fit(X[P][pk], yy).predict(X[E]); w = n / (n + 20)
                cur = (1 - w) * rankdata(zE) / len(E) + w * rankdata(pr) / len(pr)
                res["current"][r, s, j] = spearmanr(cur, y[E]).correlation
                # shared dimensions: k chosen by leave-one-out
                best = None
                for k in KS:
                    if k + 1 >= n:
                        continue
                    Bk = Xc[pk] @ basis[:, :k + 1]; e, m = loo_mse(Bk, yy)
                    if best is None or e < best[0]:
                        best = (e, k, m)
                _, k, m = best; kchosen.append(k)
                res["shared dimensions"][r, s, j] = spearmanr(m.predict(Ec @ basis[:, :k + 1]), y[E]).correlation
                if ANCH:
                    zs_p = (zP - zP.mean()) / (zP.std() + 1e-12); zs_e = (zE - zP.mean()) / (zP.std() + 1e-12)
                    ys = (yy - yy.mean()) / (yy.std() + 1e-12) if yy.std() > 0 else yy * 0
                    dev = ys - zs_p[pk]; bestA = None
                    for k2 in (2, 5, 10):
                        if k2 >= n:
                            continue
                        Bk = Xc[pk] @ pcs[:k2].T; e, m2 = loo_mse(Bk, dev)
                        if bestA is None or e < bestA[0]:
                            bestA = (e, k2, m2)
                    if bestA is None:
                        pe = zs_e
                    else:
                        _, k2, m2 = bestA; pe = zs_e + m2.predict(Ec @ pcs[:k2].T)
                    res["anchored"][r, s, j] = spearmanr(pe, y[E]).correlation
        print("SCUT split %d done" % s, flush=True)
    out = {m: {n: float(np.nanmean(v[:, :, j])) for j, n in enumerate(NS)} for m, v in res.items()}
    per_rater = {m: np.nanmean(v, 1) for m, v in res.items()}
    cand = "anchored" if ANCH else "shared dimensions"
    wins = int(np.sum(per_rater[cand][:, NS.index(20)] > per_rater["current"][:, NS.index(20)]))
    out["wins_n20"] = wins; out["better"] = bool(wins >= 45); out["candidate"] = cand
    out["k_chosen_share"] = {k: float(np.mean(np.array(kchosen) == k)) for k in KS}
    for m in names:
        print("%-18s %s" % (m, "  ".join("n=%d %.3f" % (n, out[m][n]) for n in NS)))
    print("k chosen:", out["k_chosen_share"])
    print("%s beats the current method at n=20 for %d/60 raters -> %s" % (cand, wins, "better" if out["better"] else "not better"))
    json.dump(out, open(os.path.join(root, "results/multi_rater_scut%s.json" % ("_anchored" if ANCH else "")), "w"), indent=1)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else ".")
