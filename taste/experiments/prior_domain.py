#!/usr/bin/env python3
"""Aligning the crowd model to Instagram faces (PROTOCOL.md). CPU only."""
import json
import os
import sys

import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
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
    tr = np.array([pos[u] for u in train]); y = np.array([first[u] for u in train], float)
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True)); Es = S["emb"].astype(np.float64); t = S["mean"]
    rows = np.concatenate([idx[u] for u in allu]); Ep = F["clip_face"][rows].astype(np.float64)
    face = mean_vec(F, idx, allu, "clip_face").astype(np.float64)
    ms, ss = Es.mean(0), Es.std(0) + 1e-9; mp, sp = Ep.mean(0), Ep.std(0) + 1e-9
    lin = RidgeCV(alphas=A).fit((Es - ms) / ss, t)
    zs = {"SCUT as now": lin.predict((face - ms) / ss),
          "(a) centre each domain": lin.predict((face - mp) / ss),
          "(b) standardize each domain": lin.predict((face - mp) / sp)}
    Cs = np.cov(Es.T); Cp = np.cov(Ep.T); Cp += 0.1 * np.trace(Cp) / len(Cp) * np.eye(len(Cp)); Cs += 1e-6 * np.eye(len(Cs))
    T = np.real(np.linalg.inv(sqrtm(Cp)) @ sqrtm(Cs))
    zs["(c) CORAL"] = lin.predict(((face - mp) @ T + ms - ms) / ss)
    Xs = (Es - ms) / ss; sq = (Xs ** 2).sum(1); sub = np.random.default_rng(0).choice(len(Xs), 1000, replace=False)
    D2 = np.maximum(sq[sub][:, None] + sq[None, sub] - 2 * Xs[sub] @ Xs[sub].T, 0); g = 1.0 / np.median(D2[D2 > 0])
    best_a, best_e = None, np.inf
    for a in (0.1, 0.3, 1, 3, 10):
        e = 0.0
        for trn, val in KFold(5, shuffle=True, random_state=0).split(Xs):
            m = KernelRidge(alpha=a, kernel="rbf", gamma=g).fit(Xs[trn], t[trn] - t[trn].mean()); e += np.mean((m.predict(Xs[val]) + t[trn].mean() - t[val]) ** 2)
        if e < best_e:
            best_a, best_e = a, e
    krr = KernelRidge(alpha=best_a, kernel="rbf", gamma=g).fit(Xs, t - t.mean())
    zs["(d) kernel ridge"] = krr.predict((face - ms) / ss)
    if "--combo" in sys.argv:
        zs["(b+d) kernel ridge, per-domain"] = krr.predict((face - mp) / sp)
    out = {"zero_rating": {k: float(spearmanr(v[tr], y).correlation) for k, v in zs.items()}, "krr_alpha": best_a}
    for k, v in out["zero_rating"].items():
        print("%-30s zero-rating %.3f" % (k, v), flush=True)
    best = "(b+d) kernel ridge, per-domain" if "--combo" in sys.argv else max((k for k in zs if k != "SCUT as now"), key=lambda k: out["zero_rating"][k])
    Xall = StandardScaler().fit_transform(winner_features(F, idx, allu)); Z32 = PCA(n_components=32, random_state=0).fit_transform(Xall)
    X, Z = Xall[tr], Z32[tr]; pri = {"SCUT as now": zs["SCUT as now"][tr], best: zs[best][tr]}
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid)); res = {m: {n: [] for n in NS} for m in pri}
    for s in range(20):
        gg = np.random.default_rng(1000 + s); gs = ug[:]; gg.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Z[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]; w = n / (n + 20); pr = RidgeCV(alphas=A).fit(X[pk], y[pk]).predict(X[Ei])
            for m, zc in pri.items():
                res[m][n].append(spearmanr((1 - w) * rankdata(zc[Ei]) / len(Ei) + w * rankdata(pr) / len(pr), y[Ei]).correlation)
    out["few"] = {m: {n: float(np.mean(v)) for n, v in d.items()} for m, d in res.items()}
    for m in pri:
        print("%-30s %s" % (m, "  ".join("n=%d %.3f" % (n, out["few"][m][n]) for n in NS)))
    gain = out["zero_rating"][best] - out["zero_rating"]["SCUT as now"]
    wins = int(np.sum(np.array(res[best][20]) > np.array(res["SCUT as now"][20])))
    if "--combo" in sys.argv:
        fg = float(np.mean([out["few"][best][n] - out["few"]["SCUT as now"][n] for n in NS[:4]]))
        out["verdict"] = {"best": best, "few_gain_5_40": fg, "wins_n20": wins, "pass": bool(fg >= 0.02 and wins >= 15)}
        print("%s: few gain %+.3f, wins %d/20 -> %s" % (best, fg, wins, "passes (post-hoc lead)" if out["verdict"]["pass"] else "no"))
    else:
        out["verdict"] = {"best": best, "zero_gain": gain, "wins_n20": wins, "adopt": bool(gain >= 0.03 and wins >= 15)}
        print("best %s: zero-rating gain %+.3f, wins %d/20 -> %s" % (best, gain, wins, "adopt" if out["verdict"]["adopt"] else "keep"))
    json.dump(out, open(os.path.join(root, "results/prior_domain%s.json" % ("_combo" if "--combo" in sys.argv else "")), "w"), indent=1)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else ".")
