#!/usr/bin/env python3
"""Check 2 of the bottleneck review (PROTOCOL.md, "Three checks from the bottleneck review"): learn a few
personal weights over a bank of crowd scorers instead of (S1) or as well as (S2) the ridge on L-14 codes,
in the few-rating harness of push_down.py. CPU, a few minutes. No new ratings; training accounts only.

Bank (one number per person, each trained on other people's labels): SCUT crowd on CLIP face codes, SCUT
crowd on SFace codes, MEBeauty dating crowd (CLIP + SFace), LAION aesthetic predictor on the cut-out.

Usage: scorer_bank.py ROOT -> ROOT/results/scorer_bank.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from probe_search import herding, winner_features  # noqa: E402
from sface_input import sface_means  # noqa: E402
from taste_onboard import crowd_score  # noqa: E402
from train_eval import load, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
A = np.logspace(-1, 5, 25)


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def bank(root, F, idx, allu):
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    face = np.stack([F["clip_face"][idx[u]].mean(0) for u in allu])
    face /= np.linalg.norm(face, axis=1, keepdims=True)
    sc = StandardScaler().fit(S["emb"])
    z_clip = zs(RidgeCV(alphas=A).fit(sc.transform(S["emb"]), S["mean"]).predict(sc.transform(face)))
    E = np.load(os.path.join(root, "datasets/scut/scut_sface.npz"))["emb"]
    ok = ~np.isnan(E).any(1)
    SF = sface_means(root, allu)
    sc2 = StandardScaler().fit(E[ok])
    z_sface = zs(RidgeCV(alphas=A).fit(sc2.transform(E[ok]), S["mean"][ok]).predict(sc2.transform(SF)))
    # MEBeauty v2 dating crowd, as in m_crowd_user.py
    from mebeauty_v2 import read_rows
    rows = read_rows(root, ["image_id", "ratings"])
    C = np.load(os.path.join(root, "datasets/mebeauty/v2/codes.npz"), allow_pickle=True)
    pos = {str(k): i for i, k in enumerate(C["image_id"])}
    order = [pos[r["image_id"]] for r in rows]
    Cc, Es = C["clip"][order].astype(np.float64), C["sface"][order]
    nr = np.array([len(r["ratings"]) for r in rows])
    y_me = np.array([np.mean([x["score"] for x in r["ratings"]]) if r["ratings"] else np.nan for r in rows])
    has = nr >= 3
    ok2 = ~np.isnan(Es).any(1) & has
    s3 = StandardScaler().fit(Es[ok2])
    zm_s = zs(RidgeCV(alphas=A).fit(s3.transform(Es[ok2]), y_me[ok2]).predict(s3.transform(SF)))
    s4 = StandardScaler().fit(Cc[has])
    zm_c = zs(RidgeCV(alphas=A).fit(s4.transform(Cc[has]), y_me[has]).predict(s4.transform(face)))
    z_me = zs(zm_c + zm_s)
    z_aes = zs(np.array([np.nanmean(np.asarray(F["aesthetic_cut"][idx[u]], float)) for u in allu]))
    return {"SCUT crowd, CLIP face": z_clip, "SCUT crowd, SFace": z_sface, "MEBeauty dating crowd": z_me,
            "LAION aesthetic, cut-out": z_aes}


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    allu = sorted(set(split["train"]) | set(split["test"]))
    pos = {u: i for i, u in enumerate(allu)}
    tr = np.array([pos[u] for u in train])
    ytr = np.array([first[u] for u in train], float)
    z2 = crowd_score(root, F, idx, allu)  # adopted prior
    Bd = bank(root, F, idx, allu)
    out = {"zero_rating_train": {k: round(float(spearmanr(v[tr], ytr)[0]), 4) for k, v in Bd.items()}}
    out["zero_rating_train"]["adopted prior"] = round(float(spearmanr(z2[tr], ytr)[0]), 4)
    for k, v in out["zero_rating_train"].items():
        print("%-26s zero-rating Spearman on training people %.3f" % (k, v), flush=True)
    B = StandardScaler().fit_transform(np.stack(list(Bd.values()), 1))[tr]
    XL = StandardScaler().fit_transform(winner_features(F, idx, allu))
    Z = PCA(n_components=32, random_state=0).fit_transform(XL)[tr]
    X = XL[tr]
    XB = np.c_[X, B * np.sqrt(X.shape[1] / B.shape[1])]  # bank block scaled to the codes' total weight
    variants = {"baseline (L-14 codes)": X, "S1 bank only": B, "S2 L-14 codes + bank": XB}
    gid = np.array([group.get(u, u) for u in train])
    ug = sorted(set(gid))
    res = {m: {n: [] for n in NS} for m in variants}
    for s in range(20):
        g = np.random.default_rng(1000 + s)
        gs = ug[:]
        g.shuffle(gs)
        pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid])
        Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Z[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]
            w = n / (n + 20)
            zz = rankdata(z2[tr][Ei]) / len(Ei)
            for m, Xm in variants.items():
                pr = rankdata(RidgeCV(alphas=A).fit(Xm[pk], ytr[pk]).predict(Xm[Ei])) / len(Ei)
                res[m][n].append(float(spearmanr((1 - w) * zz + w * pr, ytr[Ei])[0]))
    out["few"] = {m: {str(n): round(float(np.mean(v)), 4) for n, v in d.items()} for m, d in res.items()}
    print("\n%-24s %s" % ("variant", "  ".join("n=%d" % n for n in NS)))
    for m in variants:
        print("%-24s %s" % (m, "  ".join("%.3f" % out["few"][m][str(n)] for n in NS)), flush=True)
    base = res["baseline (L-14 codes)"]
    out["verdict"] = {}
    for m in variants:
        if m.startswith("baseline"):
            continue
        gain = float(np.mean([np.mean(res[m][n]) - np.mean(base[n]) for n in NS[:4]]))
        wins = int(np.sum(np.array(res[m][20]) > np.array(base[20])))
        out["verdict"][m] = {"gain_5_40": round(gain, 4), "wins_n20": wins, "adopt": bool(gain >= 0.02 and wins >= 15)}
        print("%-24s gain %+.4f over n = 5-40, wins %d/20 at n = 20 -> %s" % (m, gain, wins, "adopt" if out["verdict"][m]["adopt"] else "no"))
    json.dump(out, open(os.path.join(root, "results/scorer_bank.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
