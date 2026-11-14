#!/usr/bin/env python3
"""Per-person crowd weight (see PROTOCOL.md "Per-person crowd weight"). CPU only."""
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
from probe_prior import herding_fast  # noqa: E402
from probe_search import herding, winner_features  # noqa: E402
from train_eval import load, photo_index  # noqa: E402

NS = [10, 20, 40]
ALPHAS = np.logspace(-1, 5, 25)
WS = np.round(np.arange(0, 1.01, 0.1), 2)
HYBRID = "--hybrid" in sys.argv
METHODS = ["rank k=20", "z fixed k=20", "z adaptive"] + (["z hybrid"] if HYBRID else [])


def z(v, ref):
    return (v - ref.mean()) / (ref.std() + 1e-12)


def run(XS, yS, XE, priorS, priorE, priorPool):
    """Returns predictions on E for each method, and the chosen w."""
    n = len(yS); out = {}
    pc = rankdata(priorE) / len(priorE)
    if len(set(yS)) < 2:
        return {m: pc for m in METHODS}, None
    m = RidgeCV(alphas=ALPHAS).fit(XS, yS); a = m.alpha_
    pE = m.predict(XE); ref = np.r_[pE, m.predict(XS)]
    w0 = n / (n + 20)
    out["rank k=20"] = (1 - w0) * pc + w0 * rankdata(pE) / len(pE)
    zpE, zpS = z(priorE, priorPool), z(priorS, priorPool)
    out["z fixed k=20"] = (1 - w0) * zpE + w0 * z(pE, ref)
    loo = np.array([Ridge(alpha=a).fit(np.delete(XS, i, 0), np.delete(yS, i)).predict(XS[i:i + 1])[0] for i in range(n)])
    zl = z(loo, ref)
    best_w, best_r = 0.0, -np.inf
    for w in WS:
        r = spearmanr((1 - w) * zpS + w * zl, yS).correlation
        r = -np.inf if np.isnan(r) else r
        if r >= best_r - 1e-12:
            best_w, best_r = w, r
    out["z adaptive"] = (1 - best_w) * zpE + best_w * z(pE, ref)
    if HYBRID:
        wh = (best_w + w0) / 2
        out["z hybrid"] = (1 - wh) * zpE + wh * z(pE, ref)
    return out, float(best_w)


def scut(root):
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    X = StandardScaler().fit_transform(S["emb"]); Z = PCA(n_components=32, random_state=0).fit_transform(X)
    R = S["R"]; nr, ni = R.shape
    res = {m: {n: np.zeros((nr, 5)) for n in NS} for m in METHODS}; ws = {n: [] for n in NS}
    for s in range(5):
        perm = np.random.default_rng(500 + s).permutation(ni); P, E = perm[:int(0.7 * ni)], perm[int(0.7 * ni):]
        hord = herding_fast(Z[P], max(NS))
        for r in range(nr):
            y = R[r]; others = np.nanmean(np.delete(R, r, 0), 0)
            pri = RidgeCV(alphas=ALPHAS).fit(X[P], others[P]); zP, zE = pri.predict(X[P]), pri.predict(X[E])
            for n in NS:
                pk = hord[:n]
                out, w = run(X[P][pk], y[P][pk], X[E], zP[pk], zE, zP)
                for m in METHODS:
                    res[m][n][r, s] = spearmanr(out[m], y[E]).correlation
                ws[n].append(w)
        print("SCUT split %d done" % s, flush=True)
    return res, ws


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
    res = {m: {n: [] for n in NS} for m in METHODS}; ws = {n: [] for n in NS}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); P, E = np.where(isP)[0], np.where(~isP)[0]
        hord = herding(Z[P], max(NS))
        for n in NS:
            pk = hord[:n]
            out, w = run(X[P][pk], y[P][pk], X[E], zc[P][pk], zc[E], zc[P])
            for m in METHODS:
                res[m][n].append(spearmanr(out[m], y[E]).correlation)
            ws[n].append(w)
    return {m: {n: np.array(v) for n, v in d.items()} for m, d in res.items()}, ws


def main(root):
    A, wa = scut(root); B, wb = user(root)
    out = {"scut": {m: {n: float(np.nanmean(v)) for n, v in d.items()} for m, d in A.items()},
           "user": {m: {n: float(np.nanmean(v)) for n, v in d.items()} for m, d in B.items()},
           "w_scut": {n: float(np.mean([w for w in v if w is not None])) for n, v in wa.items()},
           "w_user": {n: float(np.mean([w for w in v if w is not None])) for n, v in wb.items()}}
    print("\n%-14s %s   |   %s" % ("method", "SCUT n=" + "/".join(map(str, NS)), "user n=" + "/".join(map(str, NS))))
    for m in METHODS:
        print("%-14s %s   |   %s" % (m, " ".join("%.3f" % out["scut"][m][n] for n in NS), " ".join("%.3f" % out["user"][m][n] for n in NS)))
    print("mean chosen w: SCUT %s | user %s" % (out["w_scut"], out["w_user"]))
    cand = "z hybrid" if HYBRID else "z adaptive"
    ws_ = int(np.sum(np.nanmean(A[cand][20], 1) > np.nanmean(A["rank k=20"][20], 1)))
    du = out["user"][cand][20] - out["user"]["rank k=20"][20]
    out["verdict"] = {"scut_raters_won_n20": ws_, "user_mean_diff_n20": du, "replace": bool(ws_ >= 45 and du >= -0.01)}
    print("%s vs rank k=20 at n=20: SCUT raters won %d/60, user mean diff %+.3f -> %s" % (cand, ws_, du, "replace" if out["verdict"]["replace"] else "keep k=20"))
    json.dump(out, open(os.path.join(root, "results/adaptive_blend%s.json" % ("_hybrid" if HYBRID else "")), "w"), indent=1)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else ".")
