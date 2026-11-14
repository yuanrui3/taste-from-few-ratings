#!/usr/bin/env python3
"""What the latent space may miss (PROTOCOL.md, "What the latent space may miss"): five tests on the existing
ratings. T1 best-photo aggregation, T2 style back in, T3 several types, T4 measurements on the residuals,
T5 outside knowledge (followed vs not followed). Usage: latent_gaps.py ROOT -> results/latent_gaps.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from improve_r2 import kfit  # noqa: E402
from sface_input import sface_means  # noqa: E402
from train_eval import folds, load, mean_vec, photo_index  # noqa: E402

ALPHA = 10000.0
SEEDS = [0, 1, 2]
RULE = 0.689
ALPHAS = np.logspace(-1, 5, 25)


def oof(P, owner, y, train, group):
    """Photo-level out-of-fold predictions of the full model, one array per fold seed."""
    out = []
    for r in SEEDS:
        fo = folds(train, group, seed=r); pp = np.zeros(len(owner))
        for k in range(5):
            tr, te = np.where(fo != k)[0], np.where(fo == k)[0]
            trp, tep = np.isin(owner, tr), np.isin(owner, te)
            pp[tep] = kfit(P[trp], y[owner[trp]])(P[tep], ALPHA)
        out.append(pp)
    return out


def aggregate(pp, owner, n, how):
    res = np.zeros(n)
    for i in range(n):
        v = np.sort(pp[owner == i])[::-1]
        res[i] = {"mean": v.mean(), "max": v[0], "top3": v[:3].mean(), "median": np.median(v)}[how]
    return res


def cv(preds, y):
    return float(np.mean([spearmanr(p, y)[0] for p in preds]))


def knn_predict(Xtr, ytr, Xte, k):
    S = Xte @ Xtr.T
    idx = np.argsort(-S, 1)[:, :k]
    return ytr[idx].mean(1)


def knn_fold(Xtr, ytr, Xte):
    """k by leave-one-out inside the training fold, then predict the test fold."""
    S = Xtr @ Xtr.T; np.fill_diagonal(S, -np.inf); order = np.argsort(-S, 1); best = None
    for k in (5, 10, 20, 40):
        r = spearmanr(ytr[order[:, :k]].mean(1), ytr)[0]
        if best is None or r > best[0]:
            best = (r, k)
    return knn_predict(Xtr, ytr, Xte, best[1]), best[1]


def unit_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def measurements(root, people, sf_means):
    """Per person, median over photos. Face geometry from the five YuNet landmarks of the owner's face;
    body ratios from Keypoint R-CNN where visible; hair share; face typicality. No skin colour."""
    names = ["face asymmetry", "nose below eye line", "mouth below eye line", "mouth width", "head roll",
             "shoulder / hip width", "leg / torso length", "hair share", "face typicality"]
    pool_mean = unit_rows(sf_means).mean(0); pool_mean /= np.linalg.norm(pool_mean)
    out = np.full((len(people), len(names)), np.nan)
    for i, u in enumerate(people):
        rec = json.load(open(os.path.join(root, "cut", u, "owner.json")))
        a = np.load(os.path.join(root, "arc", u + ".npz"))
        face, body, hair = [], [], []
        for ph in rec["photos"]:
            hair.append(ph["frac"]["hair"])
            m = np.nonzero(a["code"] == ph["code"])[0]
            if len(m):
                box = np.array(ph["face_box"]); j = m[np.argmin(np.abs(a["row"][m, :4] - box).sum(1))]
                if np.abs(a["row"][j, :4] - box).sum() < 4:
                    r = a["row"][j]; re, le, no, rm, lm = r[4:6], r[6:8], r[8:10], r[10:12], r[12:14]
                    v = le - re; d = np.linalg.norm(v)
                    if d > 0:
                        ux = v / d; uy = np.array([-ux[1], ux[0]]); mid = (re + le) / 2; mm = (rm + lm) / 2
                        nx, ny = (no - mid) @ ux / d, (no - mid) @ uy / d
                        mx, my = (mm - mid) @ ux / d, (mm - mid) @ uy / d
                        tilt = (lm - rm) @ uy / d
                        face.append([np.mean([abs(nx), abs(mx), abs(tilt)]), abs(ny), abs(my), np.linalg.norm(lm - rm) / d,
                                     abs(np.degrees(np.arctan2(v[1], v[0])))])
            kp = ph.get("keypoints")
            if kp is not None:
                k = np.array(kp); vis = k[:, 2] > 2
                sw = np.linalg.norm(k[5, :2] - k[6, :2]) if vis[5] and vis[6] else np.nan
                hw = np.linalg.norm(k[11, :2] - k[12, :2]) if vis[11] and vis[12] else np.nan
                sm, hm = (k[5, :2] + k[6, :2]) / 2, (k[11, :2] + k[12, :2]) / 2
                torso = np.linalg.norm(sm - hm) if vis[5] and vis[6] and vis[11] and vis[12] else np.nan
                leg = np.linalg.norm(hm - (k[15, :2] + k[16, :2]) / 2) if vis[11] and vis[12] and vis[15] and vis[16] else np.nan
                body.append([sw / hw if hw > 0 else np.nan, leg / torso if torso > 0 else np.nan])
        if face:
            out[i, :5] = np.nanmedian(np.array(face), 0)
        if body:
            b = np.array(body)
            for c in range(2):
                if np.isfinite(b[:, c]).any():
                    out[i, 5 + c] = np.nanmedian(b[:, c])
        out[i, 7] = np.median(hair)
        out[i, 8] = 1 - float(unit_rows(sf_means[i:i + 1])[0] @ pool_mean)
    return names, out


def holm(p):
    order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def main(root):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    F, train, test, first, R, group, split, same = load(root); idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True))
    y = np.array([first[u] for u in train], float); n = len(train); SFt = sface_means(root, train)
    rows = np.concatenate([idx[u] for u in train]); owner = np.concatenate([[i] * len(idx[u]) for i, u in enumerate(train)])
    base_blocks = [F["clip_nocloth"][rows], F["clip_face"][rows], M["siglip_nocloth"][rows], M["siglip_face"][rows], SFt[owner]]
    P = np.concatenate(base_blocks, 1).astype(np.float64)
    out = {}
    base_pp = oof(P, owner, y, train, group)
    base = [aggregate(pp, owner, n, "mean") for pp in base_pp]
    out["base"] = cv(base, y); print("base full model (mean of photo predictions): CV %.4f" % out["base"], flush=True)
    # T1
    out["T1"] = {}
    for how in ("max", "top3", "median"):
        c = cv([aggregate(pp, owner, n, how) for pp in base_pp], y); out["T1"][how] = c
        print("T1 %-7s CV %.4f (%+.4f) -> %s" % (how, c, c - out["base"], "adopt" if c >= RULE else "no"), flush=True)
    # T2
    out["T2"] = {}
    for name, extra in (("with-clothing cut-out codes", [F["clip_cut"][rows], M["siglip_cut"][rows]]), ("clothing-only codes", [F["clip_cloth"][rows]])):
        pp2 = oof(np.concatenate(base_blocks + extra, 1).astype(np.float64), owner, y, train, group)
        c = cv([aggregate(pp, owner, n, "mean") for pp in pp2], y); out["T2"][name] = c
        print("T2 %-28s CV %.4f (%+.4f) -> %s" % (name, c, c - out["base"], "adopt" if c >= RULE else "no"), flush=True)
    # T3
    X = np.c_[mean_vec(F, idx, train, "clip_nocloth"), mean_vec(F, idx, train, "clip_face"),
              mean_vec(M, idx, train, "siglip_nocloth"), mean_vec(M, idx, train, "siglip_face"), SFt]
    knn_p, mix_p, clu_p, ks = [], [], [], []
    for si, r in enumerate(SEEDS):
        fo = folds(train, group, seed=r); pk = np.zeros(n); pc = np.zeros(n)
        for k in range(5):
            tr, te = np.where(fo != k)[0], np.where(fo == k)[0]
            sc = StandardScaler().fit(X[tr]); Xtr, Xte = unit_rows(sc.transform(X[tr])), unit_rows(sc.transform(X[te]))
            pk[te], kk = knn_fold(Xtr, y[tr], Xte); ks.append(kk)
            Z = PCA(n_components=32, random_state=0).fit(sc.transform(X[tr])); Ztr, Zte = Z.transform(sc.transform(X[tr])), Z.transform(sc.transform(X[te]))
            km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Ztr); ctr, cte = km.labels_, km.predict(Zte)
            for c in range(3):
                a, b = tr[ctr == c], te[cte == c]
                if len(b) and len(a) >= 10:
                    s2 = StandardScaler().fit(X[a]); pc[b] = RidgeCV(alphas=ALPHAS).fit(s2.transform(X[a]), y[a]).predict(s2.transform(X[b]))
                elif len(b):
                    pc[b] = y[tr].mean()
        knn_p.append(pk); mix_p.append(rankdata(base[si]) + rankdata(pk)); clu_p.append(rankdata(base[si]) + rankdata(pc))
    out["T3"] = {"kNN": cv(knn_p, y), "full model + kNN": cv(mix_p, y), "full model + per-cluster ridge": cv(clu_p, y),
                 "k chosen": sorted(set(ks))}
    for name in ("kNN", "full model + kNN", "full model + per-cluster ridge"):
        c = out["T3"][name]; print("T3 %-31s CV %.4f (%+.4f) -> %s" % (name, c, c - out["base"], "adopt" if c >= RULE else "no"), flush=True)
    # T4
    names, Mx = measurements(root, train, SFt)
    pred = np.mean(base, 0); resid = (rankdata(y) - rankdata(pred)) / n
    rho, pv, avail = [], [], []
    for j in range(len(names)):
        ok = np.isfinite(Mx[:, j]); r_, p_ = spearmanr(Mx[ok, j], resid[ok]); rho.append(r_); pv.append(p_); avail.append(int(ok.sum()))
    adj = holm(np.array(pv)); out["T4"] = {"measures": {}}
    print("T4 measurement vs out-of-fold residual (Spearman, n, Holm p):", flush=True)
    for j, nm in enumerate(names):
        out["T4"]["measures"][nm] = {"rho": float(rho[j]), "n": avail[j], "holm_p": float(adj[j])}
        print("   %-22s %+.3f  n=%3d  p=%.3f%s" % (nm, rho[j], avail[j], adj[j], "  <- missed" if adj[j] < 0.05 else ""), flush=True)
    Mf = np.where(np.isfinite(Mx), Mx, np.nanmedian(Mx, 0))
    st2, b2 = [], []
    for si, r in enumerate(SEEDS):
        fo = folds(train, group, seed=r); p2 = np.zeros(n)
        for k in range(5):
            tr, te = np.where(fo != k)[0], np.where(fo == k)[0]
            sub = [train[i] for i in tr]; fi = folds(sub, group, seed=100 + r); inner = np.zeros(len(tr))
            for kk in range(5):
                a, b = tr[fi != kk], tr[fi == kk]; ap, bp = np.isin(owner, a), np.isin(owner, b)
                ppb = kfit(P[ap], y[owner[ap]])(P[bp], ALPHA); ob = owner[bp]
                inner[fi == kk] = [ppb[ob == i].mean() for i in b]
            tep = np.isin(owner, te); trp = np.isin(owner, tr)
            ppt = kfit(P[trp], y[owner[trp]])(P[tep], ALPHA); ot = owner[tep]; outer = np.array([ppt[ot == i].mean() for i in te])
            S2 = StandardScaler().fit(np.c_[inner, Mf[tr]])
            p2[te] = RidgeCV(alphas=ALPHAS).fit(S2.transform(np.c_[inner, Mf[tr]]), y[tr]).predict(S2.transform(np.c_[outer, Mf[te]]))
        st2.append(p2)
    out["T4"]["stacked_cv"] = cv(st2, y)
    print("T4 full model + measurements (cross-fitted second stage): CV %.4f (%+.4f) -> %s" % (
        out["T4"]["stacked_cv"], out["T4"]["stacked_cv"] - out["base"], "adopt" if out["T4"]["stacked_cv"] - out["base"] >= 0.02 else "no"), flush=True)
    # T5
    fol = set(split.get("followed", [])); isf = np.array([u in fol for u in train]); rng = np.random.default_rng(0)
    def within(m):
        return spearmanr(pred[m], y[m])[0]
    d0 = within(isf) - within(~isf); bs = []; rb = []
    fi_, ni_ = np.where(isf)[0], np.where(~isf)[0]
    for _ in range(2000):
        a = rng.choice(fi_, len(fi_)); b = rng.choice(ni_, len(ni_))
        bs.append(spearmanr(pred[a], y[a])[0] - spearmanr(pred[b], y[b])[0]); rb.append(resid[a].mean() - resid[b].mean())
    out["T5"] = {"n_followed": int(isf.sum()), "n_not": int((~isf).sum()), "rho_followed": float(within(isf)), "rho_not": float(within(~isf)),
                 "rho_diff": [float(d0), *map(float, np.percentile(bs, [2.5, 97.5]))],
                 "resid_diff": [float(resid[isf].mean() - resid[~isf].mean()), *map(float, np.percentile(rb, [2.5, 97.5]))]}
    t = out["T5"]
    print("T5 followed (n=%d) %.3f vs not followed (n=%d) %.3f; difference %+.3f [%+.3f, %+.3f]" % (
        t["n_followed"], t["rho_followed"], t["n_not"], t["rho_not"], *t["rho_diff"]), flush=True)
    print("   mean rank residual, followed minus not: %+.3f [%+.3f, %+.3f] (positive: rated higher than the photos predict)" % tuple(t["resid_diff"]), flush=True)
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(root, "results/latent_gaps.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
