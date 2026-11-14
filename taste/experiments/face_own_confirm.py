#!/usr/bin/env python3
"""Owner-only face crops in the full model, confirmed on 10 fresh fold seeds (PROTOCOL.md, "Owner-only face
crops in the full model: confirmation"). Usage: face_own_confirm.py ROOT -> results/face_own_confirm.json"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from improve_r2 import kfit  # noqa: E402
from sface_input import sface_means  # noqa: E402
from train_eval import folds, load, photo_index  # noqa: E402

SEEDS = range(100, 110)
ALPHA = 10000.0


def main(root):
    F, train, test, first, R, group, split, same = load(root); idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True))
    O = dict(np.load(os.path.join(root, "features_faceown.npz"), allow_pickle=True))
    y = np.array([first[u] for u in train], float); SFt = sface_means(root, train)
    rows = np.concatenate([idx[u] for u in train]); owner = np.concatenate([[i] * len(idx[u]) for i, u in enumerate(train)])
    P = {name: np.c_[F["clip_nocloth"][rows], cf[rows], M["siglip_nocloth"][rows], gf[rows], SFt[owner]].astype(np.float64)
         for name, cf, gf in (("current", F["clip_face"], M["siglip_face"]), ("owner-only", O["clip_face"], O["siglip_face"]))}
    res = {k: [] for k in P}
    for s in SEEDS:
        fo = folds(train, group, seed=s)
        for name, X in P.items():
            pred = np.zeros(len(y))
            for k in range(5):
                tr, te = np.where(fo != k)[0], np.where(fo == k)[0]
                trp, tep = np.isin(owner, tr), np.isin(owner, te); pp = kfit(X[trp], y[owner[trp]])(X[tep], ALPHA); o = owner[tep]
                pred[te] = [pp[o == i].mean() for i in te]
            res[name].append(float(spearmanr(pred, y)[0]))
        print("seed %d: current %.4f, owner-only %.4f" % (s, res["current"][-1], res["owner-only"][-1]), flush=True)
    gain = float(np.mean(res["owner-only"]) - np.mean(res["current"])); wins = int(sum(a > b for a, b in zip(res["owner-only"], res["current"])))
    ok = gain >= 0.005 and wins >= 7
    print("mean: current %.4f, owner-only %.4f (%+.4f), better on %d/10 seeds -> %s" % (
        np.mean(res["current"]), np.mean(res["owner-only"]), gain, wins, "adopt for the full model" if ok else "no"))
    json.dump({"per_seed": res, "gain": gain, "wins": wins, "adopt": bool(ok)}, open(os.path.join(root, "results/face_own_confirm.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
