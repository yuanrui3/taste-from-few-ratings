#!/usr/bin/env python3
"""Confirm the aligned kernel-ridge crowd model on the held-out accounts (PROTOCOL.md). CPU only."""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from train_eval import load, mean_vec, photo_index  # noqa: E402

A = np.logspace(-1, 5, 25)


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    allu = sorted(set(split["train"]) | set(split["test"]))
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True)); Es = S["emb"].astype(np.float64); t = S["mean"]
    Ep = F["clip_face"][np.concatenate([idx[u] for u in allu])].astype(np.float64)
    ms, ss = Es.mean(0), Es.std(0) + 1e-9; mp, sp = Ep.mean(0), Ep.std(0) + 1e-9
    face = mean_vec(F, idx, test, "clip_face").astype(np.float64); y = np.array([first[u] for u in test], float)
    a = json.load(open(os.path.join(root, "results/prior_domain_combo.json")))["krr_alpha"]
    Xs = (Es - ms) / ss; sq = (Xs ** 2).sum(1); sub = np.random.default_rng(0).choice(len(Xs), 1000, replace=False)
    D2 = np.maximum(sq[sub][:, None] + sq[None, sub] - 2 * Xs[sub] @ Xs[sub].T, 0); g = 1.0 / np.median(D2[D2 > 0])
    old = RidgeCV(alphas=A).fit(Xs, t).predict((face - ms) / ss)
    new = KernelRidge(alpha=a, kernel="rbf", gamma=g).fit(Xs, t - t.mean()).predict((face - mp) / sp)
    r_old, r_new = spearmanr(old, y).correlation, spearmanr(new, y).correlation
    rng = np.random.default_rng(20261113); d = []
    for _ in range(4000):
        i = rng.choice(len(y), len(y)); d.append(spearmanr(new[i], y[i]).correlation - spearmanr(old[i], y[i]).correlation)
    ci = [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
    out = {"n_test": len(y), "current": float(r_old), "aligned_kernel_ridge": float(r_new), "gain": float(r_new - r_old), "gain_ci95": ci,
           "confirmed": bool(r_new - r_old >= 0.03 and ci[0] > 0)}
    print("test accounts n=%d: current %.3f, aligned kernel ridge %.3f, gain %+.3f [%.3f, %.3f] -> %s" % (
        len(y), r_old, r_new, r_new - r_old, ci[0], ci[1], "confirmed" if out["confirmed"] else "not confirmed"))
    json.dump(out, open(os.path.join(root, "results/prior_confirm.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
