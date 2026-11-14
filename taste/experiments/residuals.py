#!/usr/bin/env python3
"""Where the full model goes wrong (exploratory, training accounts only, no decision rule).
Cross-validated predictions of the full model (L-14 + SigLIP, ridge), then the size of each
person's rank error against properties that need no ratings. CPU only."""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from train_eval import folds, load, mean_vec, photo_index  # noqa: E402


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True))
    y = np.array([first[u] for u in train], float)
    X = np.c_[mean_vec(F, idx, train, "clip_nocloth"), mean_vec(F, idx, train, "clip_face"),
              mean_vec(M, idx, train, "siglip_nocloth"), mean_vec(M, idx, train, "siglip_face")]
    preds = []
    for r in range(3):
        fo = folds(train, group, seed=r); p = np.zeros(len(y))
        for k in range(5):
            trm, te = fo != k, fo == k; sc = StandardScaler().fit(X[trm])
            p[te] = Ridge(alpha=3000).fit(sc.transform(X[trm]), y[trm]).predict(sc.transform(X[te]))
        preds.append(p)
    p = np.mean(preds, 0)
    err = np.abs(rankdata(p) / len(p) - rankdata(y) / len(y))  # rank error, 0..1
    props = {
        "owner photos": np.array([len(idx[u]) for u in train], float),
        "woman (zero-shot)": np.array([F["zs_sex_face"][idx[u], 0].mean() for u in train]),
        "photo consistency": np.array([np.mean((F["clip_nocloth"][idx[u]] @ F["clip_nocloth"][idx[u]].T)[np.triu_indices(len(idx[u]), 1)]) for u in train]),
        "face consistency": np.array([np.mean((F["clip_face"][idx[u]] @ F["clip_face"][idx[u]].T)[np.triu_indices(len(idx[u]), 1)]) for u in train]),
        "photo aesthetic (LAION)": np.array([F["aesthetic_cut"][idx[u]].mean() for u in train]),
        "rating extremeness": np.abs(y - np.median(y)),
    }
    out = {"cv_spearman": float(spearmanr(p, y).correlation), "error_vs": {}}
    print("CV Spearman of these predictions: %.3f" % out["cv_spearman"])
    for k, v in props.items():
        rho = float(spearmanr(v, err).correlation); out["error_vs"][k] = rho
        lo, hi = v <= np.median(v), v > np.median(v)
        sub = (float(spearmanr(p[lo], y[lo]).correlation), float(spearmanr(p[hi], y[hi]).correlation))
        out["error_vs"][k + " | spearman low/high half"] = sub
        print("%-24s error correlation %+.3f | model Spearman in low half %.3f, high half %.3f" % (k, rho, sub[0], sub[1]))
    w = props["woman (zero-shot)"] > 0.5
    out["women_men"] = {"n_women": int(w.sum()), "n_men": int((~w).sum()), "women": float(spearmanr(p[w], y[w]).correlation),
                        "men": float(spearmanr(p[~w], y[~w]).correlation) if (~w).sum() > 5 else None}
    print("women n=%d %.3f; men n=%d %s" % (w.sum(), out["women_men"]["women"], (~w).sum(), out["women_men"]["men"]))
    json.dump(out, open(os.path.join(root, "results/residuals.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
