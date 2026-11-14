#!/usr/bin/env python3
"""Follow-up to the MEBeauty batch (PROTOCOL.md): (a) the face-only taste model's CV on the pool, as the
reference for the batch result; (b) can the batch ratings teach it? Leave-one-out over the 60 batch photos:
face model trained on the 407 rated people plus the other 59 batch photos (batch rows weighted x5),
predicting the held-out photo. Usage: mebatch_learn.py RATINGS.jsonl ROOT -> results/mebatch_learn.json"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from mebeauty_batch import l2  # noqa: E402
from sface_input import sface_means  # noqa: E402
from train_eval import folds, load, mean_vec, photo_index  # noqa: E402

ALPHAS = np.logspace(-1, 5, 25)


def main(ratings_path, root):
    F, train, test, first, R, group, split, same = load(root); idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True)); Fm = dict(F); Fm["siglip_face"] = M["siglip_face"]
    def feats(people):
        return np.c_[mean_vec(F, idx, people, "clip_face"), mean_vec(Fm, idx, people, "siglip_face"), sface_means(root, people)]
    # (a) face-only model, grouped 5-fold x 3 CV on the training accounts
    Xt = feats(train); yt = np.array([first[u] for u in train], float); rh = []
    for r in range(3):
        fo = folds(train, group, seed=r); p = np.zeros(len(yt))
        for k in range(5):
            tr, te = fo != k, fo == k; sc = StandardScaler().fit(Xt[tr])
            p[te] = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xt[tr]), yt[tr]).predict(sc.transform(Xt[te]))
        rh.append(spearmanr(p, yt)[0])
    print("face-only model on the pool (CV, training accounts): %.3f" % np.mean(rh))
    # (b) leave-one-out over the batch
    items = json.load(open(os.path.join(root, "results/mebatch_set.json"))); B = np.load(os.path.join(root, "mebatch/codes.npz"), allow_pickle=True)
    Xb = np.c_[l2(B["clip_face"].astype(float)), l2(B["siglip_face"].astype(float)), l2(np.array([it["sface"] for it in items]))]
    rat = {}
    for line in open(ratings_path):
        r = json.loads(line)
        if r.get("pass", 1) == 1:
            rat[r["id"]] = r.get("rating")
    yb = np.array([rat.get(it["id"]) for it in items], dtype=float); ok = ~np.isnan(yb)
    people = sorted(u for u in set(split["train"]) | set(split["test"]) if u in first)
    Xp = feats(people); yp = np.array([first[u] for u in people], float)
    a = RidgeCV(alphas=ALPHAS).fit(StandardScaler().fit_transform(Xp), yp).alpha_
    loo = np.full(len(items), np.nan)
    for i in np.where(ok)[0]:
        m = ok.copy(); m[i] = False
        X = np.r_[Xp, Xb[m]]; y = np.r_[yp, yb[m]]; w = np.r_[np.ones(len(yp)), 5 * np.ones(m.sum())]
        sc = StandardScaler().fit(X); loo[i] = Ridge(alpha=a).fit(sc.transform(X), y, sample_weight=w).predict(sc.transform(Xb[i:i + 1]))[0]
    pred = json.load(open(os.path.join(root, "results/mebatch_pred.json"))); frozen = np.array([pred["your_face_model"][pred["ids"].index(it["id"])] for it in items])
    sel = {"all": ok, "Black women and men": ok & np.array([it["ethnicity"] == "black" for it in items]),
           "men": ok & np.array([it["gender"] == "male" for it in items])}
    out = {"pool_face_only_cv": float(np.mean(rh))}
    for k, m in sel.items():
        out[k] = {"frozen": float(spearmanr(frozen[m], yb[m])[0]), "with_batch_loo": float(spearmanr(loo[m], yb[m])[0]), "n": int(m.sum())}
        print("%-22s frozen %.3f -> with the other batch ratings %.3f (n=%d)" % (k, out[k]["frozen"], out[k]["with_batch_loo"], m.sum()))
    json.dump(out, open(os.path.join(root, "results/mebatch_learn.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
