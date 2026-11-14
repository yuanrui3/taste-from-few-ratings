#!/usr/bin/env python3
"""Afternoon tests A (full model: single photos + SFace) and B (picks covering SFace space). CPU only."""
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
from improve_r2 import kfit  # noqa: E402
from probe_search import herding, winner_features  # noqa: E402
from push_down import n_star  # noqa: E402
from sface_input import sface_means  # noqa: E402
from taste_onboard import crowd_score  # noqa: E402
from train_eval import ALPHAS, folds, load, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
A = np.logspace(-1, 5, 25)


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True))
    y = np.array([first[u] for u in train], float)
    # ---- A ----
    SFt = sface_means(root, train)
    rows = np.concatenate([idx[u] for u in train]); owner = np.concatenate([[i] * len(idx[u]) for i, u in enumerate(train)])
    P = np.c_[F["clip_nocloth"][rows], F["clip_face"][rows], M["siglip_nocloth"][rows], M["siglip_face"][rows], SFt[owner]].astype(np.float64)
    pa = [a * 10 for a in ALPHAS]; rhos = {a: [] for a in pa}
    for r in range(3):
        fo = folds(train, group, seed=r); pred = {a: np.zeros(len(y)) for a in pa}
        for k in range(5):
            tr, te = np.where(fo != k)[0], np.where(fo == k)[0]
            trp, tep = np.isin(owner, tr), np.isin(owner, te); f = kfit(P[trp], y[owner[trp]]); o = owner[tep]
            for a in pa:
                pp = f(P[tep], a); pred[a][te] = [pp[o == i].mean() for i in te]
        for a in pa:
            rhos[a].append(spearmanr(pred[a], y).correlation)
    cvA = {a: float(np.mean(v)) for a, v in rhos.items()}; aA = max(cvA, key=cvA.get)
    out = {"A": {"cv": cvA[aA], "alpha": aA, "baseline": 0.636, "adopt": bool(cvA[aA] >= 0.656)}}
    print("A photo-level L+SigLIP+SFace: CV %.3f (alpha %g) -> %s" % (cvA[aA], aA, "adopt" if out["A"]["adopt"] else "keep the full model"), flush=True)
    # ---- B ----
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}; tri = np.array([pos[u] for u in train])
    XL = StandardScaler().fit_transform(winner_features(F, idx, allu)); SFa = StandardScaler().fit_transform(sface_means(root, allu))
    ZL = PCA(n_components=32, random_state=0).fit_transform(XL)
    ZB = PCA(n_components=32, random_state=0).fit_transform(np.c_[XL, SFa * np.sqrt(XL.shape[1] / SFa.shape[1])])
    z = crowd_score(root, F, idx, allu)[tri]; X = XL[tri]
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid)); res = {m: {n: [] for n in NS} for m in ("L-14 picks", "L-14 + SFace picks")}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        for m, Zs in (("L-14 picks", ZL[tri]), ("L-14 + SFace picks", ZB[tri])):
            o = herding(Zs[Pi], max(NS))
            for n in NS:
                pk = Pi[o[:n]]; w = n / (n + 20); pr = RidgeCV(alphas=A).fit(X[pk], y[pk]).predict(X[Ei])
                res[m][n].append(spearmanr((1 - w) * rankdata(z[Ei]) / len(Ei) + w * rankdata(pr) / len(pr), y[Ei]).correlation)
    few = {m: {n: float(np.mean(v)) for n, v in d.items()} for m, d in res.items()}
    gain = float(np.mean([few["L-14 + SFace picks"][n] - few["L-14 picks"][n] for n in NS[:4]]))
    wins = int(np.sum(np.array(res["L-14 + SFace picks"][20]) > np.array(res["L-14 picks"][20])))
    out["B"] = {"few": few, "n_star": {m: n_star(c) for m, c in few.items()}, "gain_5_40": gain, "wins_n20": wins, "adopt": bool(gain >= 0.02 and wins >= 15)}
    for m in few:
        print("B %-20s %s   n* %s" % (m, "  ".join("%.3f" % few[m][n] for n in NS), out["B"]["n_star"][m]))
    print("B: gain %+.3f, wins %d/20 -> %s" % (gain, wins, "adopt" if out["B"]["adopt"] else "keep L-14 picks"))
    json.dump(out, open(os.path.join(root, "results/afternoon_ab.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
