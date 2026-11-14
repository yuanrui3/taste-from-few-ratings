#!/usr/bin/env python3
"""Identity space: self-supervised directions from same-person photos (PROTOCOL.md). CPU only."""
import json
import os
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from probe_search import herding, winner_features  # noqa: E402
from train_eval import ALPHAS, best_alpha, load, mean_vec, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
A = np.logspace(-1, 5, 25)


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    P = np.c_[F["clip_nocloth"], F["clip_face"]].astype(np.float64)
    owner = {}
    for u in allu:
        for i in idx[u]:
            owner[i] = u
    rows = np.array(sorted(owner)); lab = np.array([pos[owner[i]] for i in rows])
    Pm = np.stack([P[rows[lab == j]].mean(0) for j in range(len(allu))])
    W = P[rows] - Pm[lab]
    Sw = W.T @ W / len(W); Sb = np.cov(Pm.T, bias=True)
    ev = np.linalg.eigvalsh(Sw); Sw += 0.01 * ev.mean() * np.eye(len(Sw))
    lam, V = eigh(Sb, Sw); order = np.argsort(lam)[::-1]; lam, V = lam[order], V[:, order]
    k_rule = int(np.sum(lam > 1))
    print("directions with between/within ratio > 1: %d of %d (top ratios %s)" % (k_rule, len(lam), np.round(lam[:5], 2)), flush=True)
    def ident(k):
        Z = Pm @ V[:, :k]; return StandardScaler().fit_transform(Z)
    out = {"k_rule": k_rule, "top_ratios": [float(x) for x in lam[:10]], "full": {}, "few": {}}
    tr = np.array([pos[u] for u in train]); y = np.array([first[u] for u in train], float)
    gid = np.array([group.get(u, u) for u in train])
    # full data
    Xb = winner_features(F, idx, train)
    a, s0 = best_alpha(Xb, y, train, group); out["full"]["baseline L-14"] = s0
    variants = {"identity (rule)": k_rule}
    for k in (32, 64, 128, 256):
        if k != k_rule:
            variants["identity top %d" % k] = k
    for nm, k in variants.items():
        a, sc = best_alpha(ident(k)[tr], y, train, group); out["full"][nm] = sc
    for nm, v in out["full"].items():
        print("full %-22s CV %.3f" % (nm, v), flush=True)
    # few-rating
    Xall = StandardScaler().fit_transform(winner_features(F, idx, allu)); Z32 = PCA(n_components=32, random_state=0).fit_transform(Xall)
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    scs = StandardScaler().fit(S["emb"]); crowd = RidgeCV(alphas=A).fit(scs.transform(S["emb"]), S["mean"])
    zc = crowd.predict(scs.transform(mean_vec(F, idx, allu, "clip_face")))[tr]
    feats = {"baseline": Xall[tr]}
    for nm, k in variants.items():
        feats[nm] = ident(k)[tr]
    ug = sorted(set(gid)); res = {m: {n: [] for n in NS} for m in feats}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Z32[tr][Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]; w = n / (n + 20); pc = rankdata(zc[Ei]) / len(Ei)
            for m, X in feats.items():
                pr = RidgeCV(alphas=A).fit(X[pk], y[pk]).predict(X[Ei])
                res[m][n].append(spearmanr((1 - w) * pc + w * rankdata(pr) / len(pr), y[Ei]).correlation)
    out["few"] = {m: {n: float(np.nanmean(v)) for n, v in d.items()} for m, d in res.items()}
    print("\n%-22s %s" % ("few-rating", "  ".join("n=%d" % n for n in NS)))
    for m in feats:
        print("%-22s %s" % (m, "  ".join("%.3f" % out["few"][m][n] for n in NS)))
    m = "identity (rule)"
    gain = float(np.mean([out["few"][m][n] - out["few"]["baseline"][n] for n in NS[:4]]))
    wins = int(np.sum(np.array(res[m][20]) > np.array(res["baseline"][20])))
    fgain = out["full"][m] - s0
    out["verdict"] = {"few_gain_5_40": gain, "few_wins_n20": wins, "few_replace": bool(gain >= 0.02 and wins >= 15),
                      "full_gain": fgain, "full_replace": bool(fgain >= 0.02)}
    print("identity (rule, k=%d): few gain %+.3f, wins %d/20 -> %s; full gain %+.3f -> %s" % (
        k_rule, gain, wins, "replace" if out["verdict"]["few_replace"] else "keep", fgain, "replace" if out["verdict"]["full_replace"] else "keep"))
    json.dump(out, open(os.path.join(root, "results/identity_space.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
