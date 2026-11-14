#!/usr/bin/env python3
"""Labelled look at the test set for the adopted full model (PROTOCOL.md). CPU only."""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from improve_r2 import kfit  # noqa: E402
from sface_input import sface_means  # noqa: E402
from train_eval import boot_ci, load, photo_index  # noqa: E402


def photo_rows(F, M, idx, users, SF):
    rows = np.concatenate([idx[u] for u in users]); owner = np.concatenate([[i] * len(idx[u]) for i, u in enumerate(users)])
    return np.c_[F["clip_nocloth"][rows], F["clip_face"][rows], M["siglip_nocloth"][rows], M["siglip_face"][rows], SF[owner]].astype(np.float64), owner


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F); M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True))
    Ptr, otr = photo_rows(F, M, idx, train, sface_means(root, train)); Pte, ote = photo_rows(F, M, idx, test, sface_means(root, test))
    ytr = np.array([first[u] for u in train], float); yte = np.array([first[u] for u in test], float)
    pp = kfit(Ptr, ytr[otr])(Pte, 10000.0); p = np.array([pp[ote == i].mean() for i in range(len(test))])
    r = float(spearmanr(p, yte).correlation); ci = boot_ci(p, yte)
    out = {"n_test": len(test), "spearman": r, "ci95": ci, "note": "third, labelled look at an already-used test set; clean result 0.668 (original model)"}
    print("adopted full model on the 99 test accounts (labelled look): %.3f [%.3f, %.3f]" % (r, ci[0], ci[1]))
    json.dump(out, open(os.path.join(root, "results/full_test_look.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
