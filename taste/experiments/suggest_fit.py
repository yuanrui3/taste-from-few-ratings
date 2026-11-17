#!/usr/bin/env python3
"""Fit and test the per-photo sexual implication / sexual attractiveness models (docs/ADDENDUM-sexual-attractiveness.md, "Sexual
implication and sexual attractiveness, per photo"; candidates and pass rule fixed before rating, details
in "Change made before fitting"). CPU only, about a minute.

  suggest_fit.py RATINGS [--root R]   -> R/results/suggest_eval.json (no photo ids or accounts in it)

Candidates (ridge regression, one per scale, picked with its strength by grouped 5-fold CV over the
training people, mean Spearman over fold seeds 0-2):
  a  OpenAI CLIP L-14 code of the shown crop (suggest_scores.npz "oai")
  b  a + the pipeline's L-14 (LAION-2B) and SigLIP codes of the owner cut out with clothing ("cut")
  c  b + the five stand-in scores (percentile ranks over the whole pool)
Each block is standardised on the training rows and scaled to equal total weight.
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import binomtest, pearsonr, rankdata, spearmanr

SCALES = ("implication", "attractive")
STANDIN = ["laion_nsfw", "p_sexyporn", "zs_suggest", "bare", "nn_area"]
ALPHAS = np.logspace(-3, 3, 25)
SEEDS = (0, 1, 2)
N_BOOT, BOOT_SEED = 2000, 20261116


def pct_rank(x):
    return (rankdata(x) - 0.5) / len(x)


def read_ratings(path):
    eff = {}
    for line in open(path):
        r = json.loads(line)
        k = (r["id"], r.get("pass", 1))
        if r.get("undo"):
            eff.pop(k, None)
        else:
            eff[k] = r
    return eff


def folds(people, group, k=5, seed=0):  # as train_eval.folds: accounts of one person share a fold
    g = np.random.default_rng(seed)
    gids = sorted({group.get(u, u) for u in people})
    g.shuffle(gids)
    f = {gid: i % k for i, gid in enumerate(gids)}
    return np.array([f[group.get(u, u)] for u in people])


def prep(blocks, tr, te):
    A, B = [], []
    for X in blocks:
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1
        s = np.sqrt(X.shape[1])
        A.append((X[tr] - mu) / sd / s)
        B.append((X[te] - mu) / sd / s)
    return np.hstack(A), np.hstack(B)


def ridge_all(A, y, B, alphas):
    """Dual ridge with intercept; one prediction vector per strength."""
    ym = y.mean()
    w, V = np.linalg.eigh(A @ A.T)
    Vy = V.T @ (y - ym)
    BA = B @ A.T
    return [BA @ (V @ (Vy / (w + a))) + ym for a in alphas]


def rho(a, b):
    return float(spearmanr(a, b)[0])


def boot_people(people_of_row, fn, seed=BOOT_SEED, n=N_BOOT):
    ppl = np.unique(people_of_row)
    rows = {p: np.where(people_of_row == p)[0] for p in ppl}
    g = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = np.concatenate([rows[p] for p in g.choice(ppl, len(ppl), replace=True)])
        out.append(fn(idx))
    return np.array(out)


def ci(x):
    return [round(float(v), 4) for v in np.percentile(x, [2.5, 97.5])]


def pair_order(people_of_row, y, score):
    """Among people whose two photos got different ratings: does `score` put the higher-rated one higher?"""
    k = n = 0
    for p in np.unique(people_of_row):
        i = np.where(people_of_row == p)[0]
        if len(i) != 2 or y[i[0]] == y[i[1]]:
            continue
        hi, lo = (i[0], i[1]) if y[i[0]] > y[i[1]] else (i[1], i[0])
        n += 1
        k += int(score[hi] > score[lo])
    return {"right": k, "n": n, "share": round(k / n, 4) if n else None,
            "p_one_sided": float(binomtest(k, n, 0.5, alternative="greater").pvalue) if n else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ratings")
    ap.add_argument("--root", default=os.getcwd())
    a = ap.parse_args()
    R = a.root
    S = json.load(open(os.path.join(R, "split/suggest_set.json")))
    same = json.load(open(os.path.join(R, "split/same_person.json")))
    appearance = json.load(open(os.path.join(R, "split/ratings_effective.json")))["first"]
    group = {u: g[0] for g in same["groups"] for u in g}
    test_ok = set(same["test_after"])

    eff = read_ratings(a.ratings)
    photos = S["photos"]
    first = {k[0]: v for k, v in eff.items() if k[1] == 1 and not v.get("skip")}
    second = {k[0]: v for k, v in eff.items() if k[1] == 2 and not v.get("skip")}
    photos = [p for p in photos if p["id"] in first]

    # inputs
    d = dict(np.load(os.path.join(R, "results/suggest_scores.npz"), allow_pickle=False))
    pool = {}
    for X in ("laion_nsfw", "zs_suggest", "bare", "nn_area"):
        pool[X] = pct_rank(d[X].astype(float))
    pool["p_sexyporn"] = pct_rank(d["p_sexy"].astype(float) + d["p_porn"].astype(float))
    si = {(str(u), str(c)): i for i, (u, c) in enumerate(zip(d["user"], d["code"]))}
    F = np.load(os.path.join(R, "features.npz"), allow_pickle=True)
    M = np.load(os.path.join(R, "features_more_vith_siglip.npz"), allow_pickle=True)
    fi = {(str(u), str(c)): i for i, (u, c) in enumerate(zip(F["user"], F["code"]))}
    mi = {(str(u), str(c)): i for i, (u, c) in enumerate(zip(M["user"], M["code"]))}
    key = [(p["user"], p["code"]) for p in photos]
    r_s = np.array([si[k] for k in key])
    oai = d["oai"][r_s].astype(np.float64)
    clip_cut = F["clip_cut"][[fi[k] for k in key]].astype(np.float64)
    sig_cut = M["siglip_cut"][[mi[k] for k in key]].astype(np.float64)
    stand5 = np.c_[[pool[k][r_s] for k in STANDIN]].T
    standin = stand5.mean(1)
    assert np.allclose(standin, [p["standin"] for p in photos], atol=1e-3), "stand-in score does not match the pack"
    cands = {"a": [oai], "b": [oai, clip_cut, sig_cut], "c": [oai, clip_cut, sig_cut, stand5]}

    user = np.array([p["user"] for p in photos])
    is_tr = np.array([p["split"] == "train" for p in photos])
    is_te = np.array([p["split"] == "test" and p["user"] in test_ok for p in photos])
    Y = {s: np.array([first[p["id"]][s] for p in photos], float) for s in SCALES}
    male_p = {u: float(np.nanmedian(d["male"][d["user"] == u].astype(float))) for u in np.unique(user)}
    sex = np.array(["man" if male_p[u] >= 0.5 else "woman" for u in user])
    out = {"photos_rated": len(photos), "train": {"people": int(len(np.unique(user[is_tr]))), "photos": int(is_tr.sum())},
           "test": {"people": int(len(np.unique(user[is_te]))), "photos": int(is_te.sum()),
                    "dropped_same_person": len({u for u, t, p in zip(user, is_te, photos) if p["split"] == "test" and not t})}}

    # cross-validation over training people
    tr_idx = np.where(is_tr)[0]
    tr_people = sorted(set(user[tr_idx]))
    cv = {s: {} for s in SCALES}
    for c, blocks in cands.items():
        oof = {s: np.zeros((len(SEEDS), len(ALPHAS), len(tr_idx))) for s in SCALES}
        for si_, seed in enumerate(SEEDS):
            fp = dict(zip(tr_people, folds(tr_people, group, seed=seed)))
            fo = np.array([fp[u] for u in user[tr_idx]])
            for k in range(5):
                a_, b_ = tr_idx[fo != k], tr_idx[fo == k]
                A, B = prep(blocks, a_, b_)
                for s in SCALES:
                    for j, pr in enumerate(ridge_all(A, Y[s][a_], B, ALPHAS)):
                        oof[s][si_, j, fo == k] = pr
        for s in SCALES:
            r = np.array([[rho(oof[s][i, j], Y[s][tr_idx]) for j in range(len(ALPHAS))] for i in range(len(SEEDS))]).mean(0)
            j = int(np.nanargmax(r))
            cv[s][c] = {"cv_spearman": round(float(r[j]), 4), "alpha": float(ALPHAS[j])}
    out["cv"] = cv
    out["cv_standin"] = {s: round(rho(standin[tr_idx], Y[s][tr_idx]), 4) for s in SCALES}

    te_idx = np.where(is_te)[0]
    yt = {s: Y[s][te_idx] for s in SCALES}
    pt = user[te_idx]
    out["result"] = {}
    for s in SCALES:
        c = max(cv[s], key=lambda k: cv[s][k]["cv_spearman"])
        alpha = cv[s][c]["alpha"]
        A, B = prep(cands[c], tr_idx, te_idx)
        pred = ridge_all(A, Y[s][tr_idx], B, [alpha])[0]
        st = standin[te_idx]
        r_m, r_b = rho(pred, yt[s]), rho(st, yt[s])
        bm = boot_people(pt, lambda i: rho(pred[i], yt[s][i]))
        bb = boot_people(pt, lambda i: rho(st[i], yt[s][i]))
        pairs = pair_order(pt, yt[s], pred)
        res = {"candidate": c, "alpha": alpha, "spearman": round(r_m, 4), "ci95": ci(bm),
               "standin_spearman": round(r_b, 4), "standin_ci95": ci(bb),
               "difference": round(r_m - r_b, 4), "difference_ci95": ci(bm - bb),
               "pairs_model": pairs, "pairs_standin": pair_order(pt, yt[s], st)}
        res["rule"] = {"1_spearman": bool(r_m >= 0.5 and res["ci95"][0] >= 0.3),
                       "2_beats_standin": bool(res["difference_ci95"][0] > 0),
                       "3_pairs": bool(pairs["n"] and pairs["p_one_sided"] < 0.05)}
        res["pass"] = all(res["rule"].values())
        # also reported
        res["single_standin_spearman"] = {k: round(rho(stand5[te_idx, j], yt[s]), 4) for j, k in enumerate(STANDIN)}
        res["by_sex"] = {g: {"photos": int((sex[te_idx] == g).sum()),
                             "spearman_model": round(rho(pred[sex[te_idx] == g], yt[s][sex[te_idx] == g]), 4) if (sex[te_idx] == g).sum() > 5 else None,
                             "spearman_standin": round(rho(st[sex[te_idx] == g], yt[s][sex[te_idx] == g]), 4) if (sex[te_idx] == g).sum() > 5 else None}
                         for g in ("woman", "man")}
        app = np.array([appearance[u] for u in pt], float)
        res["earlier_appearance_rating_spearman"] = round(rho(app, yt[s]), 4)
        res["earlier_appearance_rating_ci95"] = ci(boot_people(pt, lambda i: rho(app[i], yt[s][i])))
        res["difference_vs_appearance_ci95"] = ci(bm - boot_people(pt, lambda i: rho(app[i], yt[s][i])))
        res["test_answers"] = {str(v): int((yt[s] == v).sum()) for v in range(11) if (yt[s] == v).any()}
        out["result"][s] = res

    # consistency on the repeats, both scales; the two scales against each other
    rep = [p for p in photos if p["id"] in second]
    out["repeats"] = {}
    for s in SCALES:
        x1 = np.array([first[p["id"]][s] for p in rep], float)
        x2 = np.array([second[p["id"]][s] for p in rep], float)
        out["repeats"][s] = {"n": len(rep), "spearman": round(rho(x1, x2), 4), "pearson": round(float(pearsonr(x1, x2)[0]), 4),
                             "same": int((x1 == x2).sum()), "within_1": int((abs(x1 - x2) <= 1).sum()),
                             "mean_abs_diff": round(float(abs(x1 - x2).mean()), 3)}
    out["scales_spearman_all"] = round(rho(Y["implication"], Y["attractive"]), 4)
    out["scales_spearman_test"] = round(rho(yt["implication"], yt["attractive"]), 4)
    # stand-in's own pair order over everyone (it picked each person's low and high photo)
    out["pairs_standin_all"] = {s: pair_order(user, Y[s], standin) for s in SCALES}
    out["answers_all"] = {s: {str(v): int((Y[s] == v).sum()) for v in range(11) if (Y[s] == v).any()} for s in SCALES}
    path = os.path.join(R, "results/suggest_eval.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
