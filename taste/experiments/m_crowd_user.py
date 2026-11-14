#!/usr/bin/env python3
"""G3 follow-up: does crowd M (SCUT + MEBeauty v2 dating crowd) keep this rater's zero-rating start?
PROTOCOL.md G3: M reaches the pool only if zero-rating Spearman on training accounts drops by 0.01 or less.
Usage: m_crowd_user.py ROOT -> results/m_crowd_user.json"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]  # experiments/, pipeline, code/ (idscore)

NS = [5, 10, 20, 40, 80]
A = np.logspace(-1, 5, 25)


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def evaluate(root):
    from probe_search import herding, winner_features
    from sface_input import sface_means
    from taste_onboard import crowd_score
    from train_eval import load, photo_index
    from mebeauty_v2 import read_rows
    rows = read_rows(root, ["image_id", "ratings"]); C = np.load(os.path.join(root, "datasets/mebeauty/v2/codes.npz"), allow_pickle=True)
    pos_v2 = {str(k): i for i, k in enumerate(C["image_id"])}; order = [pos_v2[r["image_id"]] for r in rows]
    Cc = C["clip"][order].astype(np.float64); E = C["sface"][order]; ok = ~np.isnan(E).any(1)
    nr = np.array([len(r["ratings"]) for r in rows]); y_me = np.array([np.mean([x["score"] for x in r["ratings"]]) if r["ratings"] else np.nan for r in rows])
    has = nr >= 3; ok = ok & has; Cc, y_c = Cc[has], y_me[has]  # faces with 3+ dating ratings
    print("MEBeauty v2 dating crowd: %d faces with 3+ ratings, mean %.1f raters each" % (has.sum(), nr[has].mean()), flush=True)
    F, train, test, first, R, group, split, same = load(root); idx = photo_index(F)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    sc = StandardScaler().fit(E[ok]); zs_ = zs(RidgeCV(alphas=A).fit(sc.transform(E[ok]), y_me[ok]).predict(sc.transform(sface_means(root, allu))))
    face = np.stack([F["clip_face"][idx[u]].mean(0) for u in allu]); face /= np.linalg.norm(face, axis=1, keepdims=True)
    sc2 = StandardScaler().fit(Cc); zc = zs(RidgeCV(alphas=A).fit(sc2.transform(Cc), y_c).predict(sc2.transform(face)))
    zm = zs(zc + zs_)  # MEBeauty dating crowd (CLIP + SFace), as crowd M in G3
    z2 = crowd_score(root, F, idx, allu); z3 = zs(z2 + zm)
    tr = np.array([pos[u] for u in train]); te = np.array([pos[u] for u in test])
    ytr = np.array([first[u] for u in train], float); yte = np.array([first[u] for u in test], float)
    out = {"zero": {}}
    for nm, z in (("CLIP + SFace (adopted)", z2), ("MEBeauty dating crowd alone", zm), ("M: SCUT + MEBeauty dating", z3)):
        out["zero"][nm] = {"train": float(spearmanr(z[tr], ytr).correlation), "test": float(spearmanr(z[te], yte).correlation)}
        print("%-26s zero-rating: training %.3f, test %.3f" % (nm, out["zero"][nm]["train"], out["zero"][nm]["test"]), flush=True)
    XL = StandardScaler().fit_transform(winner_features(F, idx, allu)); Z = PCA(n_components=32, random_state=0).fit_transform(XL)[tr]; X = XL[tr]
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid)); res = {m: {n: [] for n in NS} for m in ("adopted", "with MEBeauty")}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Z[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]; w = n / (n + 20); pr = rankdata(RidgeCV(alphas=A).fit(X[pk], ytr[pk]).predict(X[Ei])) / len(Ei)
            for m, zz in (("adopted", z2[tr]), ("with MEBeauty", z3[tr])):
                res[m][n].append(spearmanr((1 - w) * rankdata(zz[Ei]) / len(Ei) + w * pr, ytr[Ei]).correlation)
    out["few"] = {m: {n: float(np.mean(v)) for n, v in d.items()} for m, d in res.items()}
    gtr = out["zero"]["M: SCUT + MEBeauty dating"]["train"] - out["zero"]["CLIP + SFace (adopted)"]["train"]
    gte = out["zero"]["M: SCUT + MEBeauty dating"]["test"] - out["zero"]["CLIP + SFace (adopted)"]["test"]
    wins = int(np.sum(np.array(res["with MEBeauty"][20]) > np.array(res["adopted"][20])))
    out["verdict"] = {"zero_gain_train": gtr, "zero_gain_test": gte, "wins_n20": wins, "passes_user_check": bool(gtr >= -0.01)}
    for m in res:
        print("%-14s %s" % (m, "  ".join("n=%d %.3f" % (n, out["few"][m][n]) for n in NS)))
    print("M on this rater: zero-rating %+.3f (test %+.3f), wins %d/20 -> %s" % (gtr, gte, wins, "passes the user check" if out["verdict"]["passes_user_check"] else "no"))
    json.dump(out, open(os.path.join(root, "results/m_crowd_user.json"), "w"), indent=1)


if __name__ == "__main__":
    evaluate(sys.argv[1] if len(sys.argv) > 1 else ".")
