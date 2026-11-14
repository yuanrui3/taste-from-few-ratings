#!/usr/bin/env python3
"""Face-recognition codes (SFace) as an extra input (PROTOCOL.md, second exploration window). CPU only."""
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
from train_eval import best_alpha, load, mean_vec, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
A = np.logspace(-1, 5, 25)


def sface_means(root, users):
    out = []
    for u in users:
        e = np.load(os.path.join(root, "cut", u, "sface.npz"))["emb"].astype(np.float64)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12); m = e.mean(0); out.append(m / (np.linalg.norm(m) + 1e-12))
    return np.stack(out)


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True))
    y = np.array([first[u] for u in train], float)
    SF = sface_means(root, train)
    XLS = np.c_[winner_features(F, idx, train), mean_vec(M, idx, train, "siglip_nocloth"), mean_vec(M, idx, train, "siglip_face")]
    full = {}
    for nm, X in (("SFace alone", SF), ("L-14 + SigLIP", XLS), ("L-14 + SigLIP + SFace", np.c_[XLS, SF])):
        full[nm] = best_alpha(X, y, train, group)[1]; print("full %-24s CV %.3f" % (nm, full[nm]), flush=True)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}; tr = np.array([pos[u] for u in train])
    XL = StandardScaler().fit_transform(winner_features(F, idx, allu)); Z32 = PCA(n_components=32, random_state=0).fit_transform(XL)
    XLF = StandardScaler().fit_transform(np.c_[winner_features(F, idx, allu), sface_means(root, allu)])
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    scs = StandardScaler().fit(S["emb"]); crowd = RidgeCV(alphas=A).fit(scs.transform(S["emb"]), S["mean"])
    zc = crowd.predict(scs.transform(mean_vec(F, idx, allu, "clip_face")))[tr]
    feats = {"L-14": XL[tr], "L-14 + SFace": XLF[tr]}; Z = Z32[tr]
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid)); res = {m: {n: [] for n in NS} for m in feats}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Z[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]; w = n / (n + 20); pc = rankdata(zc[Ei]) / len(Ei)
            for m, X in feats.items():
                pr = RidgeCV(alphas=A).fit(X[pk], y[pk]).predict(X[Ei])
                res[m][n].append(spearmanr((1 - w) * pc + w * rankdata(pr) / len(pr), y[Ei]).correlation)
    out = {"full": full, "few": {m: {n: float(np.mean(v)) for n, v in d.items()} for m, d in res.items()}}
    gain = float(np.mean([out["few"]["L-14 + SFace"][n] - out["few"]["L-14"][n] for n in NS[:4]]))
    wins = int(np.sum(np.array(res["L-14 + SFace"][20]) > np.array(res["L-14"][20])))
    fg = full["L-14 + SigLIP + SFace"] - full["L-14 + SigLIP"]
    out["verdict"] = {"full_gain": fg, "few_gain_5_40": gain, "few_wins_n20": wins, "full_adopt": bool(fg >= 0.02), "few_adopt": bool(gain >= 0.02 and wins >= 15)}
    for m in feats:
        print("%-14s %s" % (m, "  ".join("n=%d %.3f" % (n, out["few"][m][n]) for n in NS)))
    print("SFace: full %+.3f -> %s; few %+.3f, wins %d/20 -> %s" % (fg, "adopt" if fg >= 0.02 else "keep", gain, wins, "adopt" if out["verdict"]["few_adopt"] else "keep"))
    json.dump(out, open(os.path.join(root, "results/sface_input.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
