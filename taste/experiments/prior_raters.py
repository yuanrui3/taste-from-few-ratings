#!/usr/bin/env python3
"""SCUT raters as stand-in members: a prior weighted by agreement with the user's first ratings
(PROTOCOL.md). CPU only."""
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
from probe_search import herding, winner_features  # noqa: E402
from train_eval import load, mean_vec, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
A = np.logspace(-1, 5, 25)


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    tr = np.array([pos[u] for u in train]); y = np.array([first[u] for u in train], float)
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    scs = StandardScaler().fit(S["emb"]); E = scs.transform(S["emb"]); face = scs.transform(mean_vec(F, idx, allu, "clip_face"))
    a = RidgeCV(alphas=A).fit(E, S["mean"]).alpha_
    Rm = S["R"]; Z = []
    for r in range(len(Rm)):
        ok = ~np.isnan(Rm[r]); Z.append(zs(Ridge(alpha=a).fit(E[ok], Rm[r][ok]).predict(face)))
    Z = np.stack(Z)[:, tr]; zmean = zs(Z.mean(0))
    rr = [spearmanr(Z[r], y).correlation for r in range(len(Z))]
    print("single SCUT raters vs user, zero-rating Spearman: median %.3f, best %.3f, worst %.3f; mean prior %.3f"
          % (np.median(rr), np.max(rr), np.min(rr), spearmanr(zmean, y).correlation), flush=True)
    Xall = StandardScaler().fit_transform(winner_features(F, idx, allu)); Z32 = PCA(n_components=32, random_state=0).fit_transform(Xall)
    X, Zp = Xall[tr], Z32[tr]
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid)); SHARP = "--sharp" in sys.argv
    names = ("mean prior", "weighted prior") + (("top-5", "softmax") if SHARP else ())
    res = {m: {n: [] for n in NS} for m in names}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Zp[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]; w = n / (n + 20); pr = RidgeCV(alphas=A).fit(X[pk], y[pk]).predict(X[Ei])
            rho = np.array([spearmanr(Z[r][pk], y[pk]).correlation for r in range(len(Z))]); rho = np.nan_to_num(rho)
            wt = np.maximum(rho, 0) ** 2
            zw = zmean if wt.sum() == 0 else (wt[:, None] * Z).sum(0) / wt.sum()
            cands = [("mean prior", zmean), ("weighted prior", zw)]
            if SHARP:
                top = np.argsort(-rho)[:5]; cands.append(("top-5", Z[top].mean(0)))
                sw = np.exp(20 * (rho - rho.max())); cands.append(("softmax", (sw[:, None] * Z).sum(0) / sw.sum()))
            for m, zc in cands:
                res[m][n].append(spearmanr((1 - w) * rankdata(zc[Ei]) / len(Ei) + w * rankdata(pr) / len(pr), y[Ei]).correlation)
    out = {"single_raters": {"median": float(np.median(rr)), "best": float(np.max(rr)), "worst": float(np.min(rr))},
           "few": {m: {n: float(np.mean(v)) for n, v in d.items()} for m, d in res.items()}}
    out["verdict"] = {}
    for m in names[1:]:
        gain = float(np.mean([out["few"][m][n] - out["few"]["mean prior"][n] for n in NS[:4]]))
        wins = int(np.sum(np.array(res[m][20]) > np.array(res["mean prior"][20])))
        out["verdict"][m] = {"gain_5_40": gain, "wins_n20": wins, "pass": bool(gain >= 0.02 and wins >= 15)}
    for m in res:
        print("%-15s %s" % (m, "  ".join("n=%d %.3f" % (n, out["few"][m][n]) for n in NS)))
    for m, v in out["verdict"].items():
        print("%-15s gain %+.3f, wins %d/20 -> %s" % (m, v["gain_5_40"], v["wins_n20"], "passes" if v["pass"] else "keep mean"))
    json.dump(out, open(os.path.join(root, "results/prior_raters%s.json" % ("_sharp" if SHARP else "")), "w"), indent=1)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else ".")
