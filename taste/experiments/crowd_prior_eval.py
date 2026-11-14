#!/usr/bin/env python3
"""Crowd face-beauty prior refinements (see PROTOCOL.md "Crowd prior refinements"). Training only."""
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
from probe_search import herding, winner_features  # noqa: E402
from train_eval import fit_predict, folds, load, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
KS = [10, 20, 40, 80]


def main(root):
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    names = np.array([str(n) for n in S["names"]])
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    follows = {l.split("\t")[0] for l in open(os.path.join(root, "links/following.tsv")) if l.strip() and not l.startswith(("#", "username"))}
    face = np.stack([F["clip_face"][idx[u]].mean(0) for u in train]); face /= np.linalg.norm(face, axis=1, keepdims=True)
    y = np.array([first[u] for u in train], float); st = np.array([u not in follows for u in train])
    variants = {"all": np.ones(len(names), bool), "women": np.array([n[:2] in ("AF", "CF") for n in names]),
                "caucasian women": np.array([n[:2] == "CF" for n in names]), "asian women": np.array([n[:2] == "AF" for n in names])}
    zs = {}
    out = {"variants": {}}
    for v, m in variants.items():
        sc = StandardScaler().fit(S["emb"][m])
        crowd = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(sc.transform(S["emb"][m]), S["mean"][m])
        z = crowd.predict(sc.transform(face)); zs[v] = z
        out["variants"][v] = {"n_faces": int(m.sum()), "rho_all": float(spearmanr(z, y).correlation),
                              "rho_strangers": float(spearmanr(z[st], y[st]).correlation)}
        print("%-16s faces %4d  zero-rating Spearman all %.3f, strangers %.3f" % (v, m.sum(), out["variants"][v]["rho_all"], out["variants"][v]["rho_strangers"]), flush=True)
    best = max(out["variants"], key=lambda v: out["variants"][v]["rho_all"]); out["chosen"] = best
    z = zs[best]
    users = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(users)}
    Xall = StandardScaler().fit_transform(winner_features(F, idx, users)); tr = np.array([pos[u] for u in train]); X = Xall[tr]
    zz = ((z - z.mean()) / z.std())[:, None] * 5
    ref, add = [], []
    for rep in range(3):
        fo = folds(train, group, seed=rep); p0 = np.zeros(len(y)); p1 = np.zeros(len(y))
        for k in range(5):
            a, b = fo != k, fo == k
            p0[b] = fit_predict(X[a], y[a], X[b], 3000); p1[b] = fit_predict(np.c_[X[a], zz[a]], y[a], np.c_[X[b], zz[b]], 3000)
        ref.append(spearmanr(p0, y).correlation); add.append(spearmanr(p1, y).correlation)
    out["full_cv_main"] = float(np.mean(ref)); out["full_cv_main_plus_crowd"] = float(np.mean(add))
    out["adopt_into_main"] = bool(out["full_cv_main_plus_crowd"] - out["full_cv_main"] >= 0.02)
    print("full-data CV: main %.3f, main + crowd (%s) %.3f -> %s" % (out["full_cv_main"], best, out["full_cv_main_plus_crowd"],
          "adopt" if out["adopt_into_main"] else "keep main"), flush=True)
    Z32 = PCA(n_components=32, random_state=0).fit_transform(Xall)
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid))
    res = {k: {n: [] for n in NS} for k in KS}; only = {n: [] for n in NS}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); P, E = np.where(isP)[0], np.where(~isP)[0]
        order = herding(Z32[tr[P]], max(NS)); rz = rankdata(z[E]) / len(E)
        for n in NS:
            pk = order[:n]
            pr = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(X[P][pk], y[P][pk]).predict(X[E]); ro = rankdata(pr) / len(pr)
            only[n].append(spearmanr(pr, y[E]).correlation)
            for k in KS:
                w = n / (n + k); res[k][n].append(spearmanr((1 - w) * rz + w * ro, y[E]).correlation)
    out["few"] = {"ratings only": {n: float(np.nanmean(v)) for n, v in only.items()}}
    for k in KS:
        out["few"]["blend k=%d" % k] = {n: float(np.nanmean(v)) for n, v in res[k].items()}
    avg = {key: float(np.mean([d[n] for n in (5, 10, 20, 40)])) for key, d in out["few"].items()}
    out["few_avg_5_40"] = avg; out["few_best"] = max((k for k in avg if k.startswith("blend")), key=avg.get)
    for key, d in out["few"].items():
        print("  %-14s %s   avg %.3f" % (key, "  ".join("%.3f" % d[n] for n in NS), avg[key]))
    print("best blend:", out["few_best"])
    json.dump(out, open(os.path.join(root, "results/crowd_prior_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
