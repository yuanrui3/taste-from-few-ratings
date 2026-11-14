#!/usr/bin/env python3
"""Photo-style confound check (see PROTOCOL.md "Photo-style confound"). Training accounts only."""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from taste_explain import text_embed  # noqa: E402
from train_eval import fit_predict, folds, load, photo_index  # noqa: E402

STYLE = [("a professional studio photoshoot", "a casual amateur phone photo"),
         ("a professionally retouched photo", "an unedited snapshot"),
         ("a fashion magazine photo", "a selfie")]


def account_feats(F, idx, users, nocloth, face):
    def m(E):
        out = np.stack([E[idx[u]].mean(0) for u in users]); return out / np.linalg.norm(out, axis=1, keepdims=True)
    return np.c_[m(nocloth), m(face)]


def remove_dirs(E, D):
    Q, _ = np.linalg.qr(D.T)  # orthonormal basis of the style directions
    E2 = E - (E @ Q) @ Q.T
    return E2 / np.linalg.norm(E2, axis=1, keepdims=True)


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def res(a, b):
        b1 = np.c_[np.ones(len(b)), b]; return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    return float(np.corrcoef(res(rx, rz), res(ry, rz))[0, 1])


def cv_pred(X, y, train, group, rep):
    fo = folds(train, group, seed=rep); p = np.zeros(len(y))
    for k in range(5):
        a, b = fo != k, fo == k
        p[b] = fit_predict(X[a], y[a], X[b], 3000)
    return p


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    follows = {l.split("\t")[0] for l in open(os.path.join(root, "links/following.tsv")) if l.strip() and not l.startswith(("#", "username"))}
    T = text_embed([p for pair in STYLE for p in pair])
    D = np.stack([T[2 * k] - T[2 * k + 1] for k in range(len(STYLE))])
    # per-photo style on the no-clothing cut-out (what the model sees), standardized per probe, averaged
    S = F["clip_nocloth"] @ D.T
    S = ((S - S.mean(0)) / S.std(0)).mean(1)
    y = np.array([first[u] for u in train], float)
    stranger = np.array([u not in follows for u in train])
    style_acc = np.array([S[idx[u]].mean() for u in train])
    out = {"style_vs_rating_all": float(spearmanr(style_acc, y).correlation),
           "style_vs_rating_strangers": float(spearmanr(style_acc[stranger], y[stranger]).correlation)}

    Xm = StandardScaler().fit_transform(account_feats(F, idx, train, F["clip_nocloth"], F["clip_face"]))
    Xd = StandardScaler().fit_transform(account_feats(F, idx, train, remove_dirs(F["clip_nocloth"], D), remove_dirs(F["clip_face"], D)))
    main_r, des_r, part = [], [], []
    for rep in range(3):
        pm = cv_pred(Xm, y, train, group, rep); pd = cv_pred(Xd, y, train, group, rep)
        main_r.append(spearmanr(pm, y).correlation); des_r.append(spearmanr(pd, y).correlation)
        part.append(partial_spearman(pm, y, style_acc))
    out.update({"cv_main": float(np.mean(main_r)), "cv_destyled": float(np.mean(des_r)),
                "partial_main_given_style": float(np.mean(part))})

    # within-person leak: account-level model applied to single photos
    sc = StandardScaler().fit(account_feats(F, idx, train, F["clip_nocloth"], F["clip_face"]))
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=3000).fit(Xm, y)
    leaks = []
    for u in train:
        ii = idx[u]
        if len(ii) < 6:
            continue
        P = np.c_[F["clip_nocloth"][ii], F["clip_face"][ii]]
        pp = m.predict(sc.transform(P))
        leaks.append(spearmanr(pp, S[ii]).correlation)
    out["within_person_leak"] = float(np.nanmean(leaks)); out["within_person_n"] = len(leaks)
    material = (out["cv_main"] - out["cv_destyled"] >= 0.05) or (out["within_person_leak"] >= 0.20)
    out["material"] = bool(material)
    out["recommend_destyled"] = bool(material and out["cv_main"] - out["cv_destyled"] < 0.03)
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(root, "results/style_confound.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
