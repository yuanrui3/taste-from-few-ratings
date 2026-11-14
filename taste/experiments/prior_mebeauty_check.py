#!/usr/bin/env python3
"""Does aligning the SCUT crowd model help on an independent face set (MEBeauty)? (PROTOCOL.md)"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV

A = np.logspace(-1, 5, 25)


def main(root):
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True)); Es = S["emb"].astype(np.float64); t = S["mean"]
    Me = dict(np.load(os.path.join(root, "datasets/mebeauty/me.npz"), allow_pickle=True)); Em = Me["emb"].astype(np.float64); ym = Me["mean"].astype(float)
    ms, ss = Es.mean(0), Es.std(0) + 1e-9; mm, sm = Em.mean(0), Em.std(0) + 1e-9
    a = json.load(open(os.path.join(root, "results/prior_domain_combo.json")))["krr_alpha"]
    Xs = (Es - ms) / ss; sq = (Xs ** 2).sum(1); sub = np.random.default_rng(0).choice(len(Xs), 1000, replace=False)
    D2 = np.maximum(sq[sub][:, None] + sq[None, sub] - 2 * Xs[sub] @ Xs[sub].T, 0); g = 1.0 / np.median(D2[D2 > 0])
    old = RidgeCV(alphas=A).fit(Xs, t).predict((Em - ms) / ss)
    krr = KernelRidge(alpha=a, kernel="rbf", gamma=g).fit(Xs, t - t.mean())
    new = krr.predict((Em - mm) / sm); krr_only = krr.predict((Em - ms) / ss); std_only = RidgeCV(alphas=A).fit(Xs, t).predict((Em - mm) / sm)
    r = {k: float(spearmanr(v, ym).correlation) for k, v in (("current", old), ("standardize only", std_only), ("kernel ridge only", krr_only), ("aligned kernel ridge", new))}
    rng = np.random.default_rng(20261113); d = []
    for _ in range(2000):
        i = rng.choice(len(ym), len(ym)); d.append(spearmanr(new[i], ym[i]).correlation - spearmanr(old[i], ym[i]).correlation)
    ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    out = {"n": int(len(ym)), "spearman": r, "gain": r["aligned kernel ridge"] - r["current"], "gain_ci95": ci,
           "supported": bool(r["aligned kernel ridge"] - r["current"] >= 0.03 and ci[0] > 0)}
    for k, v in r.items():
        print("%-22s Spearman with MEBeauty means %.3f" % (k, v))
    print("gain %+.3f [%.3f, %.3f] -> %s" % (out["gain"], ci[0], ci[1], "supported" if out["supported"] else "not supported"))
    json.dump(out, open(os.path.join(root, "results/prior_mebeauty_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
