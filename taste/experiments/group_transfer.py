#!/usr/bin/env python3
"""Does the model rank people from groups the rater has barely rated? (PROTOCOL.md, "Generalizing to
people the rater has rarely rated", G1 and G2.)

G1: MEBeauty raters, trained only on their ratings of White and East Asian women, scored on Black
faces, men and other women. G2: SCUT raters, trained on 400 women, scored on men. Methods: the
current crowd/ratings blend; E = erase the directions separating group means (learned from
MEBeauty's folder labels, no ratings); U = lean on the crowd for people whose codes lie outside what
the ratings model learned (no labels); E + U. Group labels come only from the datasets' folders.

Usage: group_transfer.py ROOT   -> results/group_transfer.json
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xlsx_lite import table  # noqa: E402

ALPHAS = [1e2, 1e3, 1e4, 1e5]
K_BLEND = 20
CROWD_ALPHAS = np.logspace(-1, 5, 25)
METHODS = ["current", "E", "U", "E+U", "crowd only", "ratings only"]


def l2(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def std_cols(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def group_basis(E, groups):
    """Orthonormal directions spanned by the group means' differences from the grand mean."""
    m = E.mean(0)
    D = np.stack([E[groups == g].mean(0) - m for g in sorted(set(groups))])
    _, S, Vt = np.linalg.svd(D, full_matrices=False)
    return m, Vt[: int((S > 1e-8 * S[0]).sum())].T


def erase(E, basis):
    m, Q = basis
    return E - ((E - m) @ Q) @ Q.T


def kr_fit(Xtr, y):
    """Kernel ridge with intercept, strength by leave-one-out; returns predict and share functions."""
    mu = Xtr.mean(0); A = Xtr - mu; ym = y.mean(); yc = y - ym
    l, Q = np.linalg.eigh(A @ A.T); l = np.clip(l, 0, None); Qy = Q.T @ yc
    best = None
    for a in ALPHAS:
        h = (Q ** 2) @ (l / (l + a)); fit = Q @ (Qy * l / (l + a))
        loo = np.mean(((yc - fit) / (1 - h + 1e-12)) ** 2)
        if best is None or loo < best[0]:
            best = (loo, a)
    a = best[1]; w = A.T @ (Q @ (Qy / (l + a)))

    def predict(X):
        return ym + (X - mu) @ w

    def share(X):  # k'(K + aI)^-1 k / |x|^2: the part of x the ratings model has learned about
        Xc = X - mu; t = Q.T @ (A @ Xc.T)
        return np.sum(t ** 2 / (l + a)[:, None], 0) / (np.sum(Xc ** 2, 1) + 1e-12)
    return predict, share


def run_rater(Xs, crowd, y, fam, unf, seed=0):
    """Xs: {False: codes, True: erased codes}. fam / unf: index arrays. Returns scores per method on fam+unf."""
    fold = np.random.default_rng(seed).permutation(len(fam)) % 5
    mixed = np.r_[fam, unf]; N = len(mixed)
    rc = rankdata(crowd[mixed]) / N
    n = 0.8 * len(fam); wb = n / (n + K_BLEND)
    out = {}
    for er in (False, True):
        X = Xs[er]; pf = np.zeros(len(fam)); qf = np.zeros(len(fam)); pu = np.zeros(len(unf)); qu = np.zeros(len(unf))
        for k in range(5):
            tr, te = fam[fold != k], fam[fold == k]
            f, sh = kr_fit(X[tr], y[tr])
            pf[fold == k] = f(X[te]); qf[fold == k] = sh(X[te])
            pu += f(X[unf]) / 5; qu += sh(X[unf]) / 5
        rp = rankdata(np.r_[pf, pu]) / N
        s = np.minimum(1.0, np.r_[qf, qu] / np.median(qf)); w = s * wb
        out["E" if er else "current"] = (1 - wb) * rc + wb * rp
        out["E+U" if er else "U"] = (1 - w) * rc + w * rp
        if not er:
            out["ratings only"] = rp
            out["share_familiar"] = float(np.median(qf)); out["share_unfamiliar"] = float(np.median(qu))
    out["crowd only"] = rc
    return mixed, out


def measures(score, ytrue, labels, within, place):
    """Spearman on all, within named label sets; placement error over the 'place' labels."""
    r = {"mixed": spearmanr(score, ytrue)[0]}
    for name, labs in within.items():
        m = np.isin(labels, labs)
        r[name] = spearmanr(score[m], ytrue[m])[0] if m.sum() >= 10 else np.nan
    ps, pt = rankdata(score) / len(score), rankdata(ytrue) / len(ytrue); errs = []
    for g in place:
        m = labels == g
        if m.sum() >= 10:
            errs.append(abs(ps[m].mean() - pt[m].mean()))
    r["placement"] = float(np.mean(errs)) if errs else np.nan
    return r


def summarize(name, rows, within_names):
    """rows: list of {method: measures}. Prints means, wins vs current, verdicts."""
    keys = ["mixed"] + within_names + ["placement"]
    res = {}
    print("%s: %d raters" % (name, len(rows)))
    print("  %-13s" % "method" + "".join("%11s" % k for k in keys) + "   wins")
    for m in METHODS:
        mean = {k: float(np.nanmean([r[m][k] for r in rows])) for k in keys}
        wins = int(sum(r[m]["mixed"] > r["current"]["mixed"] for r in rows))
        res[m] = dict(mean, wins=wins)
        print("  %-13s" % m + "".join("%11.3f" % mean[k] for k in keys) + "   %d/%d" % (wins, len(rows)))
    cur = res["current"]
    for m in ["E", "U", "E+U"]:
        g = res[m]["mixed"] - cur["mixed"]; fam = res[m]["familiar"] - cur["familiar"]
        ok = g >= 0.02 and res[m]["wins"] >= 2 * len(rows) / 3 and fam >= -0.01
        res[m]["verdict"] = "build for the pool" if ok else "no"
        print("  %s: mixed %+.3f, wins %d/%d, familiar %+.3f -> %s" % (m, g, res[m]["wins"], len(rows), fam, res[m]["verdict"]))
    return res


def scut_crowd_models(root):
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    E = np.load(os.path.join(root, "datasets/scut/scut_sface.npz"))["emb"]; ok = ~np.isnan(E).any(1)
    sc = StandardScaler().fit(S["emb"]); clip = RidgeCV(alphas=CROWD_ALPHAS).fit(sc.transform(S["emb"]), S["mean"])
    sc2 = StandardScaler().fit(E[ok]); sf = RidgeCV(alphas=CROWD_ALPHAS).fit(sc2.transform(E[ok]), S["mean"][ok])
    return lambda C, F: zs(zs(clip.predict(sc.transform(C))) + zs(sf.predict(sc2.transform(F))))


def load_mebeauty(root):
    me = np.load(os.path.join(root, "datasets/mebeauty/me.npz"), allow_pickle=True)
    paths = [str(p) for p in me["paths"]]
    key = ["/".join(p.split("/")[-3:]) for p in paths]
    sex = np.array([p.split("/")[-3] for p in paths]); eth = np.array([p.split("/")[-2] for p in paths])
    clip = l2(me["emb"].astype(np.float64)); sf = np.full((len(paths), 128), np.nan)
    for f in glob.glob(os.path.join(root, "datasets/mebeauty/sface_part_*.npz")):
        d = np.load(f); sf[d["idx"]] = d["emb"]
    return key, sex, eth, clip, l2(sf)


def g1(root, crowd_fn, bases, me):
    key, sex, eth, clip, sf = me
    X0 = std_cols(np.c_[clip, sf]); Xe = std_cols(np.c_[erase(clip, bases[0]), erase(sf, bases[1])])
    crowd = crowd_fn(clip, sf)
    lab = np.where((sex == "female") & np.isin(eth, ["caucasian", "asian"]), "familiar",
                   np.where(eth == "black", np.where(sex == "male", "black men", "black women"),
                            np.where(sex == "male", "other men", "other women")))
    T = table(os.path.join(root, "datasets/mebeauty/MEBeauty-database-main/scores/generic_scores_all.xlsx"))
    pos = {k: i for i, k in enumerate(key)}
    img = [pos.get("/".join(p.split("/")[-3:])) if p else None for p in T["path"]]
    rows = []; shares = []
    for rater in [c for c in T if c not in ("mean", "image", "path", None)]:
        y = np.full(len(key), np.nan)
        for i, v in zip(img, T[rater]):
            if i is not None and v not in (None, "") and np.isnan(y[i]):
                y[i] = float(v)
        rated = ~np.isnan(y)
        fam = np.where(rated & (lab == "familiar"))[0]; unf = np.where(rated & (lab != "familiar"))[0]
        nb = int((rated & (eth == "black")).sum()); nm = int((rated & (sex == "male")).sum())
        if len(fam) < 100 or nb < 40 or nm < 100:
            continue
        mixed, sc = run_rater({False: X0, True: Xe}, crowd, y, fam, unf)
        shares.append((sc.pop("share_familiar"), sc.pop("share_unfamiliar")))
        L = lab[mixed]; yt = y[mixed]
        within = {"familiar": ["familiar"], "black": ["black women", "black men"],
                  "men": ["black men", "other men"], "other women": ["other women"]}
        rows.append({m: measures(sc[m], yt, L, within, ["black women", "black men", "other men", "other women"]) for m in METHODS})
    res = summarize("G1 MEBeauty", rows, ["familiar", "black", "men", "other women"])
    sh = np.array(shares); print("  share of codes learned (median): familiar %.3f, unfamiliar %.3f" % tuple(sh.mean(0)))
    return res


def g2(root, bases):
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    sfr = np.load(os.path.join(root, "datasets/scut/scut_sface.npz"))["emb"]; ok = np.where(~np.isnan(sfr).any(1))[0]
    names = [str(n) for n in S["names"]][:]; clip = l2(S["emb"].astype(np.float64))[ok]; sf = l2(sfr[ok]); R = S["R"][:, ok]
    names = [names[i] for i in ok]
    lab = np.array([("asian " if n[0] == "A" else "white ") + ("men" if n[1] == "M" else "women") for n in names])
    X0 = std_cols(np.c_[clip, sf]); Xe = std_cols(np.c_[erase(clip, bases[0]), erase(sf, bases[1])])
    nr = R.shape[0]; others = (R.sum(0)[None, :] - R) / (nr - 1)  # each rater's crowd: the other raters' mean
    rows = []
    for s in range(5):
        rng = np.random.default_rng(100 + s); perm = rng.permutation(len(names)); A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
        fam = rng.choice(B[np.char.endswith(lab[B], "women")], 400, replace=False)
        unf = rng.choice(B[np.char.endswith(lab[B], " men")], 1000, replace=False)
        sc1 = StandardScaler().fit(clip[A]); m1 = RidgeCV(alphas=CROWD_ALPHAS, alpha_per_target=True).fit(sc1.transform(clip[A]), others[:, A].T)
        sc2 = StandardScaler().fit(sf[A]); m2 = RidgeCV(alphas=CROWD_ALPHAS, alpha_per_target=True).fit(sc2.transform(sf[A]), others[:, A].T)
        idx = np.r_[fam, unf]; P1 = m1.predict(sc1.transform(clip[idx])); P2 = m2.predict(sc2.transform(sf[idx]))
        for r in range(nr):
            crowd = np.zeros(len(names)); crowd[idx] = zs(zs(P1[:, r]) + zs(P2[:, r]))
            mixed, sc = run_rater({False: X0, True: Xe}, crowd, R[r], fam, unf, seed=s)
            sc.pop("share_familiar"); sc.pop("share_unfamiliar")
            within = {"familiar": ["asian women", "white women"], "men": ["asian men", "white men"]}
            rows.append({m: measures(sc[m], R[r][mixed], lab[mixed], within, ["asian men", "white men"]) for m in METHODS})
        print("  G2 split %d done" % s, flush=True)
    return summarize("G2 SCUT (rows = rater x split)", rows, ["familiar", "men"])


def main(root):
    me = load_mebeauty(root); key, sex, eth, clip, sf = me
    groups = np.array([a + "/" + b for a, b in zip(sex, eth)])
    bases = (group_basis(clip, groups), group_basis(sf, groups))
    print("group directions removed: CLIP %d, SFace %d" % (bases[0][1].shape[1], bases[1][1].shape[1]))
    crowd_fn = scut_crowd_models(root)
    out = {"G1": g1(root, crowd_fn, bases, me)}
    out["G2"] = g2(root, bases)
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(root, "results/group_transfer.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
