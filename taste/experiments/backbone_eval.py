#!/usr/bin/env python3
"""Compare image models for full-data accuracy and few-rating learning (see PROTOCOL.md "Larger
image models"). Training accounts only."""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from probe_search import herding  # noqa: E402
from train_eval import ALPHAS, cv_score, load, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]


def means(E, idx, users):
    out = np.stack([E[idx[u]].mean(0) for u in users]); return out / np.linalg.norm(out, axis=1, keepdims=True)


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    M = {}
    for f in sorted(os.listdir(root)):
        if f.startswith("features_more_") and f.endswith(".npz"):
            d = np.load(os.path.join(root, f), allow_pickle=True)
            assert (d["code"] == F["code"]).all()
            M.update({k: d[k] for k in d.files if k not in ("user", "code")})
    idx = photo_index(F)
    users = sorted(set(split["train"]) | set(split["test"]))
    blocks = {"clipL": ["clip_nocloth", "clip_face"], "vitH": ["vith_nocloth", "vith_face"],
              "siglip": ["siglip_nocloth", "siglip_face"]}
    src = lambda k: F[k] if k in F.files else M[k]
    blocks = {b: ks for b, ks in blocks.items() if all(k in F.files or k in M for k in ks)}
    print("models available:", list(blocks), flush=True)
    base = {b: np.concatenate([means(src(k), idx, users) for k in ks], 1) for b, ks in blocks.items()}
    names = list(blocks)
    combos = {b: [b] for b in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            combos[names[i] + "+" + names[j]] = [names[i], names[j]]
    if len(names) == 3:
        combos["all3"] = names
    feats = {c: StandardScaler().fit_transform(np.concatenate([base[b] for b in bs], 1)) for c, bs in combos.items()}
    pos = {u: i for i, u in enumerate(users)}
    tr = np.array([pos[u] for u in train]); y = np.array([first[u] for u in train], float)
    out = {"full_cv": {}, "few": {}}
    for c, X in feats.items():
        best = max(((a, cv_score(X[tr], y, train, group, a)) for a in ALPHAS), key=lambda t: t[1])
        out["full_cv"][c] = {"alpha": best[0], "cv": float(best[1])}
        print("%-14s full-data CV %.3f (alpha %g)" % (c, best[1], best[0]), flush=True)
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid))
    per = {c: {n: [] for n in NS} for c in feats}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); P, E = np.where(isP)[0], np.where(~isP)[0]
        for c, X in feats.items():
            Z = PCA(n_components=32, random_state=0).fit_transform(X)
            order = herding(Z[tr[P]], max(NS))
            for n in NS:
                pk = order[:n]
                m = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(X[tr[P]][pk], y[P][pk])
                per[c][n].append(float(spearmanr(m.predict(X[tr[E]]), y[E]).correlation))
        print("split %d done" % s, flush=True)
    for c in feats:
        out["few"][c] = {n: float(np.mean(v)) for n, v in per[c].items()}
    avg = {c: float(np.mean([out["few"][c][n] for n in (5, 10, 20, 40)])) for c in feats}
    best_few = max(avg, key=avg.get)
    wins = int(np.sum(np.array(per[best_few][20]) > np.array(per["clipL"][20])))
    best_full = max(out["full_cv"], key=lambda c: out["full_cv"][c]["cv"])
    gain = out["full_cv"][best_full]["cv"] - out["full_cv"]["clipL"]["cv"]
    print("\nfew-rating (herding picks), mean evaluation Spearman")
    print("%-14s %s  avg(5-40)" % ("model", "  ".join("n=%-4d" % n for n in NS)))
    for c in sorted(avg, key=avg.get, reverse=True):
        print("%-14s %s  %.3f" % (c, "  ".join("%.3f " % out["few"][c][n] for n in NS), avg[c]))
    print("\nBEST full-data: %s (+%.3f over clipL) -> %s" % (best_full, gain, "replace" if gain >= 0.02 else "keep clipL"))
    print("BEST few-rating: %s, beats clipL at n=20 in %d/20 splits -> %s" % (best_few, wins, "better" if wins >= 15 and best_few != "clipL" else "not better"))
    out.update({"avg_5_40": avg, "best_few": best_few, "wins_n20": wins, "best_full": best_full, "full_gain": gain})
    json.dump(out, open(os.path.join(root, "results/backbone_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
