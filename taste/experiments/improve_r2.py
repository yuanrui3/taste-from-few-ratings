#!/usr/bin/env python3
"""Full model, round 2 (PROTOCOL.md): photo-level training with three image models, and an
account-level + photo-level ensemble. CPU only; ridge in kernel form for speed."""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from train_eval import ALPHAS, folds, load, mean_vec, photo_index  # noqa: E402


def kfit(Xtr, ytr):
    """Standardize on the training rows, ridge with intercept in kernel form; returns predict(Xte, a)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; A = (Xtr - mu) / sd; ym = ytr.mean()
    l, Q = np.linalg.eigh(A @ A.T); qy = Q.T @ (ytr - ym)
    def pred(Xte, a):
        return ym + ((Xte - mu) / sd) @ (A.T @ (Q @ (qy / (l + a))))
    return pred


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True))
    y = np.array([first[u] for u in train], float)
    rows = np.concatenate([idx[u] for u in train]); owner = np.concatenate([[i] * len(idx[u]) for i, u in enumerate(train)])
    PLS = np.c_[F["clip_nocloth"][rows], F["clip_face"][rows], M["siglip_nocloth"][rows], M["siglip_face"][rows]].astype(np.float64)
    PLSH = np.c_[PLS, M["vith_nocloth"][rows], M["vith_face"][rows]].astype(np.float64)
    XA = np.c_[mean_vec(F, idx, train, "clip_nocloth"), mean_vec(F, idx, train, "clip_face"),
               mean_vec(M, idx, train, "siglip_nocloth"), mean_vec(M, idx, train, "siglip_face")].astype(np.float64)
    def cv(fn):
        rhos = {}
        for r in range(3):
            fo = folds(train, group, seed=r); preds = {}
            for k in range(5):
                tr, te = np.where(fo != k)[0], np.where(fo == k)[0]
                for key, p in fn(tr, te).items():
                    preds.setdefault(key, np.zeros(len(y)))[te] = p
            for key, p in preds.items():
                rhos.setdefault(key, []).append(spearmanr(p, y).correlation)
        return {k: float(np.mean(v)) for k, v in rhos.items()}
    def photo(P, tr, te, alphas):
        trp, tep = np.isin(owner, tr), np.isin(owner, te); f = kfit(P[trp], y[owner[trp]]); o = owner[tep]
        out = {}
        for a in alphas:
            pp = f(P[tep], a); out[a] = np.array([pp[o == i].mean() for i in te])
        return out
    pa = [a * 10 for a in ALPHAS]
    g1 = cv(lambda tr, te: photo(PLSH, tr, te, pa))
    a1 = max(g1, key=g1.get)
    def ens(tr, te):
        fa = kfit(XA[tr], y[tr]); acc = {a: fa(XA[te], a) for a in ALPHAS}; acc_tr = {a: fa(XA[tr], a) for a in ALPHAS}
        ph = photo(PLS, tr, te, pa); ph_tr = photo(PLS, tr, tr, pa)
        out = {}
        for a in ALPHAS:
            for b in pa:
                z1 = (acc[a] - acc_tr[a].mean()) / (acc_tr[a].std() + 1e-12); z2 = (ph[b] - ph_tr[b].mean()) / (ph_tr[b].std() + 1e-12)
                out[(a, b)] = (z1 + z2) / 2
        return out
    g2 = cv(ens); a2 = max(g2, key=g2.get)
    out = {"G1 photo-level L+SigLIP+ViT-H": {"cv": g1[a1], "alpha": a1},
           "G2 account + photo ensemble": {"cv": g2[a2], "alphas": list(a2)}, "baseline": 0.636, "round1 F2": 0.652}
    for k in ("G1 photo-level L+SigLIP+ViT-H", "G2 account + photo ensemble"):
        print("%-32s CV %.3f" % (k, out[k]["cv"]), flush=True)
    best = max(("G1 photo-level L+SigLIP+ViT-H", "G2 account + photo ensemble"), key=lambda k: out[k]["cv"])
    out["verdict"] = {"best": best, "passes": bool(out[best]["cv"] >= 0.656)}
    print("best %s %.3f -> %s" % (best, out[best]["cv"], "passes (post-hoc; confirm with new ratings)" if out["verdict"]["passes"] else "keep the full model"))
    json.dump(out, open(os.path.join(root, "results/improve_r2.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
