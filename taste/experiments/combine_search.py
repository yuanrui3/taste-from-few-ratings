#!/usr/bin/env python3
"""Combine the follow-list prior with a few ratings (see PROTOCOL.md "Combining the follow-list
prior with few ratings"). Training accounts only, the same 20 pool/evaluation splits as
probe_search.py; then one second look at the used test set for the rule's pick.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from probe_search import herding, winner_features  # noqa: E402
from train_eval import load, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
ALPHAS = np.logspace(-1, 5, 25)
C_GRID = [0, 0.25, 0.5, 0.75, 1.0, 1.5]
R0 = [0.3, 0.5]
KS = [10, 20, 40, 80]


def ridge(X, y):
    return RidgeCV(alphas=ALPHAS).fit(X, y)


def methods(XS, yS, XE, zS, zE):
    """All candidate predictions for the evaluation rows, from n rated rows (XS, yS)."""
    out = {}
    m = ridge(XS, yS)
    only = m.predict(XE)
    out["ratings only"] = only
    best = None
    for c in C_GRID:
        mm = ridge(XS, yS - c * zS)
        if best is None or mm.best_score_ > best[0]:
            best = (mm.best_score_, c, mm)
    out["transfer ridge"] = best[2].predict(XE) + best[1] * zE
    sd = yS.std() if yS.std() > 0 else 1.0
    for r0 in R0:
        c = r0 * sd
        out["fixed offset r0=%.1f" % r0] = ridge(XS, yS - c * zS).predict(XE) + c * zE
    n = len(yS)
    rp, ro = rankdata(zE) / len(zE), rankdata(only) / len(only)
    for k in KS:
        w = n / (n + k)
        out["rank blend k=%d" % k] = (1 - w) * rp + w * ro
    return out


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    follows = {l.split("\t")[0] for l in open(os.path.join(root, "links/following.tsv"))
               if l.strip() and not l.startswith(("#", "username"))}
    all_users = sorted(set(split["train"]) | set(split["test"]))
    W = winner_features(F, idx, all_users)
    X = StandardScaler().fit_transform(W)
    Z32 = PCA(n_components=32, random_state=0).fit_transform(X)
    pos = {u: i for i, u in enumerate(all_users)}
    tr = np.array([pos[u] for u in train]); y = np.array([first[u] for u in train], float)
    fol = np.array([u in follows for u in all_users])
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid))

    def prior(rows_fit, *rows_score):
        clf = LogisticRegressionCV(Cs=8, cv=5, max_iter=2000).fit(X[rows_fit], fol[rows_fit])
        s = [clf.decision_function(X[r]) for r in rows_score]
        mu, sd = s[0].mean(), s[0].std() + 1e-9
        return [(v - mu) / sd for v in s]

    res, prior_alone = {}, []
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); P, E = np.where(isP)[0], np.where(~isP)[0]
        zP, zE = prior(tr[P], tr[P], tr[E])
        prior_alone.append(spearmanr(zE, y[E]).correlation)
        order = herding(Z32[tr[P]], max(NS))
        for n in NS:
            sels = {"herding": [order[:n]], "random": [list(np.random.default_rng(s * 100 + r).choice(len(P), n, replace=False)) for r in range(5)]}
            for sel, picklists in sels.items():
                acc = {}
                for pk in picklists:
                    pk = list(pk)
                    preds = methods(X[tr[P]][pk], y[P][pk], X[tr[E]], zP[pk], zE)
                    for name, p in preds.items():
                        acc.setdefault(name, []).append(spearmanr(p, y[E]).correlation)
                for name, v in acc.items():
                    res.setdefault("%s | %s" % (sel, name), {}).setdefault(n, []).append(float(np.nanmean(v)))
        print("split %d done" % s, flush=True)

    pa = np.array(prior_alone)
    summary = {k: {n: float(np.mean(v)) for n, v in d.items()} for k, d in res.items()}
    avg = {k: float(np.mean([summary[k][n] for n in (5, 10, 20, 40)])) for k in summary}
    ranked = sorted(avg, key=avg.get, reverse=True)
    best = ranked[0]
    wins = int(np.sum(np.array(res[best][20]) > pa))
    print("\nprior alone (0 ratings): %.3f" % pa.mean())
    print("%-36s %s   avg(5,10,20,40)" % ("selection | combination", "  ".join("n=%-4d" % n for n in NS)))
    for k in ranked:
        print("%-36s %s   %.3f" % (k, "  ".join("%.3f " % summary[k][n] for n in NS), avg[k]))
    print("\nBEST:", best, "| beats the prior alone at n=20 in %d/20 splits" % wins,
          "->", "fixes the problem" if wins >= 15 else "does NOT fix the problem")

    # second look at the used test set, rule's pick only
    sel, name = best.split(" | ")
    te = np.array([pos[u] for u in test]); yte = np.array([first[u] for u in test], float)
    zT, zTE = prior(tr, tr, te)
    order = herding(Z32[tr], max(NS))
    second = {"prior alone": float(spearmanr(zTE, yte).correlation)}
    for n in NS:
        picklists = [order[:n]] if sel == "herding" else [list(np.random.default_rng(r).choice(len(tr), n, replace=False)) for r in range(50)]
        vals = []
        for pk in picklists:
            pk = list(pk)
            vals.append(spearmanr(methods(X[tr][pk], y[pk], X[te], zT[pk], zTE)[name], yte).correlation)
        second[n] = float(np.mean(vals))
    print("second look at the used test set (%s): %s" % (best, ", ".join("%s %.3f" % (k, v) for k, v in second.items())))
    json.dump({"summary": summary, "avg_5_10_20_40": avg, "prior_alone": float(pa.mean()), "best": best,
               "wins_vs_prior_n20": wins, "second_look_test": second},
              open(os.path.join(root, "results/combine_search.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
