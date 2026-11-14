#!/usr/bin/env python3
"""Cross-batch check of portrait ratings in the face model (PROTOCOL.md, "MEBeauty batch 2", follow-up):
train on the pool + one batch (weight 0, 0.5, 1, 2, 5), test on the other batch; both directions.
Usage: mebatch_weights.py RATINGS1 RATINGS2 ROOT -> results/mebatch_weights.json"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from mebatch2 import batch_xy  # noqa: E402
from sface_input import sface_means  # noqa: E402
from train_eval import load, mean_vec, photo_index  # noqa: E402

WEIGHTS = [0, 0.5, 1, 2, 5]


def main(r1, r2, root):
    F, train, test, first, R, group, split, same = load(root); idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True)); Fm = dict(F); Fm["siglip_face"] = M["siglip_face"]
    people = sorted(u for u in set(split["train"]) | set(split["test"]) if u in first); yp = np.array([first[u] for u in people], float)
    Xp = np.c_[mean_vec(F, idx, people, "clip_face"), mean_vec(Fm, idx, people, "siglip_face"), sface_means(root, people)]
    a = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(StandardScaler().fit_transform(Xp), yp).alpha_
    b1 = batch_xy(root, "mebatch_set.json", "mebatch", r1); b2 = batch_xy(root, "mebatch2_set.json", "mebatch2", r2)
    out = {}
    for name, (itr, Xtr, ytr), (ite, Xte, yte) in (("batch 1 -> batch 2", b1, b2), ("batch 2 -> batch 1", b2, b1)):
        ok = ~np.isnan(ytr); okt = ~np.isnan(yte)
        black = okt & np.array([it["ethnicity"] == "black" for it in ite]); men = okt & np.array([it["gender"] == "male" for it in ite])
        out[name] = {}
        for w in WEIGHTS:
            X = np.r_[Xp, Xtr[ok]] if w else Xp; y = np.r_[yp, ytr[ok]] if w else yp
            sw = np.r_[np.ones(len(yp)), w * np.ones(ok.sum())] if w else np.ones(len(yp))
            sc = StandardScaler().fit(X); p = Ridge(alpha=a).fit(sc.transform(X), y, sample_weight=sw).predict(sc.transform(Xte))
            out[name][w] = {k: float(spearmanr(p[m], yte[m])[0]) for k, m in (("all", okt), ("Black", black), ("men", men))}
            print("%-19s weight %-4s all %.3f  Black %.3f  men %.3f" % (name, w, *[out[name][w][k] for k in ("all", "Black", "men")]))
    best = max(WEIGHTS[1:], key=lambda w: np.mean([out[n][w]["Black"] for n in out]))
    ok = all(out[n][best]["Black"] > out[n][0]["Black"] for n in out)
    out["verdict"] = {"best_weight": best, "beats_pool_only_both_ways": bool(ok)}
    print("best weight within Black faces: %s; beats the pool-only model in both directions: %s" % (best, ok))
    json.dump(out, open(os.path.join(root, "results/mebatch_weights.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ".")
