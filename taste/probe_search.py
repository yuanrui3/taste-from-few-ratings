#!/usr/bin/env python3
"""Search for a probe set that needs few ratings (training accounts only; see PROTOCOL.md
"Probe-set search").

20 splits of the 305 rated training accounts into pool (70%) / evaluation (30%), same-person
accounts together. A selection method picks n pool accounts without ratings; a ridge model
(strength by leave-one-out on those n ratings) is fitted on them; Spearman on the evaluation part.

Then, for the method the fixed rule picks, one second look at the (already used) test set.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_eval import load, mean_vec, photo_index  # noqa: E402

NS = [5, 10, 20, 30, 40, 60, 80]
ALPHAS = np.logspace(-1, 5, 25)
SPLITS = 20


def winner_features(F, idx, users):
    return np.concatenate([mean_vec(F, idx, users, "clip_nocloth"), mean_vec(F, idx, users, "clip_face")], 1)


def farthest_first(Z, n):
    c = Z.mean(0)
    order = [int(np.argmin(np.linalg.norm(Z - c, axis=1)))]
    d = np.linalg.norm(Z - Z[order[0]], axis=1)
    while len(order) < n:
        i = int(np.argmax(d)); order.append(i); d = np.minimum(d, np.linalg.norm(Z - Z[i], axis=1))
    return order


def kmeans_medoids(Z, n, seed=0):
    km = KMeans(n_clusters=n, n_init=10, random_state=seed).fit(Z)
    picks = []
    for c in km.cluster_centers_:
        d = np.linalg.norm(Z - c, axis=1)
        d[picks] = np.inf
        picks.append(int(np.argmin(d)))
    return picks


def herding(Z, n):
    D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    s2 = np.median(D2[D2 > 0])
    K = np.exp(-D2 / s2)
    mu = K.mean(1)
    acc = np.zeros(len(Z)); order = []
    for t in range(n):
        score = mu - acc / (t + 1)
        score[order] = -np.inf
        i = int(np.argmax(score)); order.append(i); acc += K[:, i]
    return order


def fit_pred(Xs, y, Xe):
    m = RidgeCV(alphas=ALPHAS).fit(Xs, y)
    return m.predict(Xe)


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    follows = {l.split("\t")[0] for l in open(os.path.join(root, "links/following.tsv"))
               if l.strip() and not l.startswith(("#", "username"))}
    all_users = sorted(set(split["train"]) | set(split["test"]))
    W_all = winner_features(F, idx, all_users)
    sc = StandardScaler().fit(W_all)           # unsupervised: features of all 412 accounts, no ratings
    pca = PCA(n_components=32, random_state=0).fit(sc.transform(W_all))
    pos = {u: i for i, u in enumerate(all_users)}
    full = sc.transform(W_all)
    reps = {"full": full, "pca8": pca.transform(full)[:, :8], "pca16": pca.transform(full)[:, :16],
            "pca32": pca.transform(full)}
    sel_space = reps["pca32"]

    tr = np.array([pos[u] for u in train]); y = np.array([first[u] for u in train], float)
    fol = np.array([u in follows for u in train])
    gid = np.array([group.get(u, u) for u in train])
    ugroups = sorted(set(gid))

    def prior_scores(pool_rows, *targets):
        clf = LogisticRegressionCV(Cs=8, cv=5, max_iter=2000).fit(full[pool_rows], fol_lookup[pool_rows])
        return [clf.decision_function(full[t]) for t in targets]

    fol_lookup = np.zeros(len(all_users), bool)
    for u in all_users:
        fol_lookup[pos[u]] = u in follows

    res = {}  # key -> n -> list of per-split scores
    prior_only = []

    def add(key, n, v):
        res.setdefault(key, {}).setdefault(n, []).append(v)

    for s in range(SPLITS):
        g = np.random.default_rng(1000 + s); gs = ugroups[:]; g.shuffle(gs)
        pool_g = set(gs[:int(0.7 * len(gs))])
        is_pool = np.array([x in pool_g for x in gid])
        P, E = np.where(is_pool)[0], np.where(~is_pool)[0]
        Zp = sel_space[tr[P]]
        sp_prior_pool, sp_prior_eval = prior_scores(tr[P], tr[P], tr[E])
        prior_only.append(spearmanr(sp_prior_eval, y[E]).correlation)
        orders = {"farthest": farthest_first(Zp, max(NS)), "herding": herding(Zp, max(NS))}
        for n in NS:
            picks = {"farthest": orders["farthest"][:n], "herding": orders["herding"][:n],
                     "kmeans": kmeans_medoids(Zp, n, seed=s)}
            for r in range(5):
                picks["random%d" % r] = list(np.random.default_rng(s * 100 + r).choice(len(P), n, replace=False))
            for rep, X in reps.items():
                Xp, Xe = X[tr[P]], X[tr[E]]
                for prior in (False, True):
                    if prior:
                        z = sp_prior_pool / (sp_prior_pool.std() + 1e-9) * 10
                        ze = sp_prior_eval / (sp_prior_pool.std() + 1e-9) * 10
                        Xp2, Xe2 = np.c_[Xp, z], np.c_[Xe, ze]
                    else:
                        Xp2, Xe2 = Xp, Xe
                    rand = []
                    for name, pk in picks.items():
                        pk = list(pk)
                        sc_ = spearmanr(fit_pred(Xp2[pk], y[P][pk], Xe2), y[E]).correlation
                        if name.startswith("random"):
                            rand.append(sc_)
                        else:
                            add("%s|%s|%s" % (name, rep, "prior" if prior else "-"), n, sc_)
                    add("random|%s|%s" % (rep, "prior" if prior else "-"), n, float(np.nanmean(rand)))
        print("split %d done; prior-only %.3f" % (s, prior_only[-1]), flush=True)

    summary = {}
    for key, d in res.items():
        summary[key] = {n: [float(np.nanmean(v)), float(np.nanstd(v))] for n, v in d.items()}
    avg = {k: np.mean([summary[k][n][0] for n in (10, 20, 40)]) for k in summary}
    ranked = sorted(avg, key=avg.get, reverse=True)
    best = ranked[0]
    ref = "random|full|-"
    wins = int(np.sum(np.array(res[best][20]) > np.array(res[ref][20])))
    same_model_random = "random|" + "|".join(best.split("|")[1:])
    wins_same = int(np.sum(np.array(res[best][20]) > np.array(res[same_model_random][20])))
    print("\nzero ratings (follow-list prior only): %.3f +- %.3f" % (np.nanmean(prior_only), np.nanstd(prior_only)))
    print("\n%-34s %s   avg(10,20,40)" % ("method | features | prior", "  ".join("n=%-4d" % n for n in NS)))
    for k in ranked[:12] + [k for k in ranked if k.startswith("random|full|-")]:
        print("%-34s %s   %.3f" % (k, "  ".join("%.3f " % summary[k][n][0] for n in NS), avg[k]))
    print("\nBEST by the rule:", best, "| beats plain random at n=20 in %d/20 splits" % wins,
          "| beats random with the same model in %d/20" % wins_same)

    # second look at the used test set, for the chosen method only
    sel, rep, prior = best.split("|")
    X = reps[rep]
    trows, terows = tr, np.array([pos[u] for u in test]); yte = np.array([first[u] for u in test], float)
    Zall = sel_space[trows]
    if prior == "prior":
        clf = LogisticRegressionCV(Cs=8, cv=5, max_iter=2000).fit(full[trows], fol_lookup[trows])
        zt, zte = clf.decision_function(full[trows]), clf.decision_function(full[terows])
        s_ = zt.std() + 1e-9
        Xtr2, Xte2 = np.c_[X[trows], zt / s_ * 10], np.c_[X[terows], zte / s_ * 10]
        prior_test = float(spearmanr(zte, yte).correlation)
    else:
        Xtr2, Xte2 = X[trows], X[terows]; prior_test = None
    clf0 = LogisticRegressionCV(Cs=8, cv=5, max_iter=2000).fit(full[trows], fol_lookup[trows])
    zero = float(spearmanr(clf0.decision_function(full[terows]), yte).correlation)
    second = {}
    for n in NS:
        pk = {"farthest": lambda: farthest_first(Zall, n), "herding": lambda: herding(Zall, n),
              "kmeans": lambda: kmeans_medoids(Zall, n), "random": None}[sel]
        if pk is None:
            second[n] = float(np.mean([spearmanr(fit_pred(Xtr2[list(p)], y[list(p)], Xte2), yte).correlation
                                       for p in (np.random.default_rng(r).choice(len(trows), n, replace=False) for r in range(50))]))
        else:
            p = pk(); second[n] = float(spearmanr(fit_pred(Xtr2[p], y[p], Xte2), yte).correlation)
    print("\nsecond look at the used test set (%s): zero ratings %.3f; " % (best, zero) +
          ", ".join("n=%d %.3f" % (n, v) for n, v in second.items()))
    out = {"summary": summary, "avg_10_20_40": avg, "best": best, "wins_vs_plain_random_n20": wins,
           "wins_vs_same_model_random_n20": wins_same, "zero_rating_prior_cv": [float(np.nanmean(prior_only)), float(np.nanstd(prior_only))],
           "second_look_test": {"method": best, "curve": second, "zero_rating_prior": zero}}
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(root, "results/probe_search.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
