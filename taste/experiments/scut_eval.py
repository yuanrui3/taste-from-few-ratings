#!/usr/bin/env python3
"""Few-rating learning across the 60 SCUT-FBP5500 raters, and transfer of crowd face beauty to the
user (see PROTOCOL.md "Many raters: SCUT-FBP5500")."""
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
from train_eval import load, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
ALPHAS = np.logspace(-1, 5, 25)
C_GRID = [0, 0.25, 0.5, 0.75, 1.0, 1.5]


def ridge(X, y):
    return RidgeCV(alphas=ALPHAS).fit(X, y)


def few_models(XS, yS, XE, zS, zE):
    out = {}
    only = ridge(XS, yS).predict(XE); out["ratings only"] = only
    out["crowd prior only"] = zE
    w = len(yS) / (len(yS) + 40)
    out["prior + ratings (rank blend)"] = (1 - w) * rankdata(zE) / len(zE) + w * rankdata(only) / len(only)
    best = None
    for c in C_GRID:
        m = ridge(XS, yS - c * zS)
        if best is None or m.best_score_ > best[0]:
            best = (m.best_score_, c, m)
    out["prior + ratings (transfer ridge)"] = best[2].predict(XE) + best[1] * zE
    return out


def part_a(S):
    X = StandardScaler().fit_transform(S["emb"]); Z = PCA(n_components=32, random_state=0).fit_transform(X)
    R = S["R"]; nr, ni = R.shape
    res = {}  # (sel, model, target) -> n -> list over raters of mean over splits
    agree = []
    for s in range(5):
        rng = np.random.default_rng(500 + s); perm = rng.permutation(ni); P, E = perm[:int(0.7 * ni)], perm[int(0.7 * ni):]
        order = herding(Z[P], max(NS))
        for r in range(nr):
            y = R[r]; others = np.nanmean(np.delete(R, r, 0), 0)
            if s == 0:
                agree.append(spearmanr(y, others, nan_policy="omit").correlation)
            prior = ridge(X[P], others[P]); zP, zE = prior.predict(X[P]), prior.predict(X[E])
            mu, sd = zP.mean(), zP.std() + 1e-9; zP, zE = (zP - mu) / sd, (zE - mu) / sd
            dev_E = y[E] - others[E]
            for n in NS:
                picks = {"herding": [order[:n]],
                         "random": [list(np.random.default_rng(s * 1000 + r * 10 + k).choice(len(P), n, replace=False)) for k in range(3)]}
                for sel, lists in picks.items():
                    acc = {}
                    for pk in lists:
                        pk = list(pk)
                        for name, p in few_models(X[P][pk], y[P][pk], X[E], zP[pk], zE).items():
                            acc.setdefault((name, "own"), []).append(spearmanr(p, y[E]).correlation)
                            acc.setdefault((name, "personal part"), []).append(spearmanr(p, dev_E).correlation)
                    for (name, tgt), v in acc.items():
                        res.setdefault((sel, name, tgt), {}).setdefault(n, {}).setdefault(r, []).append(float(np.nanmean(v)))
        print("A: split %d done" % s, flush=True)
    per_rater = {k: {n: np.array([np.mean(v[n][r]) for r in range(nr)]) for n in v} for k, v in res.items()}
    summary = {" | ".join(k): {n: float(a.mean()) for n, a in d.items()} for k, d in per_rater.items()}
    hw = int(np.sum(per_rater[("herding", "ratings only", "own")][20] > per_rater[("random", "ratings only", "own")][20]))
    pw = int(np.sum(per_rater[("herding", "prior + ratings (rank blend)", "own")][20] > per_rater[("herding", "ratings only", "own")][20]))
    pw2 = int(np.sum(per_rater[("herding", "prior + ratings (transfer ridge)", "own")][20] > per_rater[("herding", "ratings only", "own")][20]))
    print("\nA. %d raters; each rater vs the others' mean (how shared taste is): median Spearman %.3f" % (nr, np.median(agree)))
    print("%-62s %s" % ("selection | model | target", "  ".join("n=%-4d" % n for n in NS)))
    for k in sorted(summary):
        print("%-62s %s" % (k, "  ".join("%.3f " % summary[k][n] for n in NS)))
    print("herding beats random (ratings only, own, n=20) for %d/%d raters -> %s" % (hw, nr, "better" if hw >= 45 else "not better"))
    print("prior + ratings beats ratings only (herding, own, n=20): rank blend %d/%d, transfer ridge %d/%d -> %s"
          % (pw, nr, pw2, nr, "prior helps" if max(pw, pw2) >= 45 else "prior does not help"))
    return {"summary": summary, "agreement_median": float(np.median(agree)), "herding_wins_n20": hw,
            "prior_blend_wins_n20": pw, "prior_transfer_wins_n20": pw2}


def part_b(root, S):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    follows = {l.split("\t")[0] for l in open(os.path.join(root, "links/following.tsv")) if l.strip() and not l.startswith(("#", "username"))}
    sc = StandardScaler().fit(S["emb"])
    crowd = ridge(sc.transform(S["emb"]), S["mean"])
    face = np.stack([F["clip_face"][idx[u]].mean(0) for u in train]); face /= np.linalg.norm(face, axis=1, keepdims=True)
    z = crowd.predict(sc.transform(face)); y = np.array([first[u] for u in train], float)
    st = np.array([u not in follows for u in train])
    out = {"crowd_vs_user_all": float(spearmanr(z, y).correlation), "crowd_vs_user_strangers": float(spearmanr(z[st], y[st]).correlation)}
    print("\nB. crowd face beauty vs your ratings: all %.3f, strangers %.3f" % (out["crowd_vs_user_all"], out["crowd_vs_user_strangers"]))
    if out["crowd_vs_user_all"] >= 0.20:
        users = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(users)}
        X = StandardScaler().fit_transform(winner_features(F, idx, users)); Z = PCA(n_components=32, random_state=0).fit_transform(X)
        tr = np.array([pos[u] for u in train]); gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid))
        res = {}
        for s in range(20):
            g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
            isP = np.array([x in pool for x in gid]); P, E = np.where(isP)[0], np.where(~isP)[0]
            zP, zE = z[P], z[E]; mu, sd = zP.mean(), zP.std() + 1e-9; zP, zE = (zP - mu) / sd, (zE - mu) / sd
            order = herding(Z[tr[P]], max(NS))
            for n in NS:
                pk = order[:n]
                for name, p in few_models(X[tr[P]][pk], y[P][pk], X[tr[E]], zP[pk], zE).items():
                    res.setdefault(name, {}).setdefault(n, []).append(float(spearmanr(p, y[E]).correlation))
        out["user_few"] = {k: {n: float(np.mean(v)) for n, v in d.items()} for k, d in res.items()}
        wins = max(int(np.sum(np.array(res[k][20]) > np.array(res["ratings only"][20]))) for k in res if k.startswith("prior +"))
        out["user_prior_wins_n20"] = wins
        for k, d in out["user_few"].items():
            print("  %-36s %s" % (k, "  ".join("%.3f" % d[n] for n in NS)))
        print("  prior + ratings beats ratings only at n=20 in %d/20 splits -> %s" % (wins, "helps" if wins >= 15 else "does not help"))
    return out


def main(root):
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    out = {"A": part_a(S), "B": part_b(root, S)}
    json.dump(out, open(os.path.join(root, "results/scut_eval.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
