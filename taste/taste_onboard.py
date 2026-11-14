#!/usr/bin/env python3
"""Few-rating onboarding: predict a new rater's taste from a handful of ratings (images only).

What it does, all without Instagram metadata:
  1. Starting point: two crowd face-beauty models trained on SCUT-FBP5500's mean scores, one on
     OpenCLIP ViT-L-14 face codes and one on SFace face-recognition codes (which ignore lighting
     and make-up, so they carry over from studio faces better). Each scores every person from
     their face crops; the crowd score is the mean of the two, standardized. Without
     datasets/scut/scut_sface.npz (experiments/sface_crowd.py makes it) only the CLIP one is used.
  2. Probe order: kernel herding on PCA-32 of the people's image codes (no ratings) -> the people
     to rate first, typical of the whole pool.
  3. With n ratings: a ridge on the image codes (strength by leave-one-out) fitted to them, blended
     with the crowd score by rank: w = n / (n + 20) on the ratings model.
Evidence (PROTOCOL.md): on the user's training accounts this gives about 0.39 / 0.46 / 0.48 / 0.55
Spearman at 0 / 10 / 20 / 40 ratings, against 0.28 / 0.37 / 0.48 for ratings alone at 10 / 20 / 40.

Usage (project folder on the cluster):
  taste_onboard.py probes --n 40            # the first n people to rate (ids from rate_ids.json)
  taste_onboard.py predict RATINGS.json     # {"<user>": rating, ...} -> ranked predictions
  taste_onboard.py export OUT.json          # data for the local onboarding page (opaque ids only)
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_search import herding, winner_features  # noqa: E402
from train_eval import load, photo_index  # noqa: E402

K_BLEND = 20


def setup(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    users = sorted(set(split["train"]) | set(split["test"]))
    X = StandardScaler().fit_transform(winner_features(F, idx, users))
    z = crowd_score(root, F, idx, users)
    order = herding(PCA(n_components=32, random_state=0).fit_transform(X), 80)
    return users, X, z, order


def _std(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def crowd_score(root, F, idx, users):
    """Mean of the standardized SCUT crowd scores on CLIP face codes and on SFace codes."""
    alphas = np.logspace(-1, 5, 25)
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    sc = StandardScaler().fit(S["emb"])
    clip = RidgeCV(alphas=alphas).fit(sc.transform(S["emb"]), S["mean"])
    face = np.stack([F["clip_face"][idx[u]].mean(0) for u in users]); face /= np.linalg.norm(face, axis=1, keepdims=True)
    z = _std(clip.predict(sc.transform(face)))
    path = os.path.join(root, "datasets/scut/scut_sface.npz")
    if not os.path.exists(path):
        print("note: no SCUT SFace codes; crowd score from CLIP face codes only", file=sys.stderr)
        return z
    E = np.load(path)["emb"]; ok = ~np.isnan(E).any(1)
    sc2 = StandardScaler().fit(E[ok]); sf = RidgeCV(alphas=alphas).fit(sc2.transform(E[ok]), S["mean"][ok])
    pool = []
    for u in users:  # each person's owner face codes, stored by owner_cutouts.py
        e = np.load(os.path.join(root, "cut", u, "sface.npz"))["emb"].astype(np.float64)
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12; m = e.mean(0); pool.append(m / (np.linalg.norm(m) + 1e-12))
    return _std(z + _std(sf.predict(sc2.transform(np.stack(pool)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probes", "predict", "export"])
    ap.add_argument("ratings", nargs="?")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    users, X, z, order = setup(a.root)
    if a.cmd == "export":
        # Gram matrix of the standardized image codes: enough for ridge in kernel form in the page.
        ids = json.load(open(os.path.join(a.root, "split/rate_ids.json")))
        R = json.load(open(os.path.join(a.root, "split/ratings_effective.json")))["first"]
        keep = [i for i, u in enumerate(users) if u in ids]
        G = X[keep] @ X[keep].T / X.shape[1]
        out = {"ids": [ids[users[i]] for i in keep], "gram": [[round(float(v), 4) for v in row] for row in G],
               "crowd": [round(float(z[i]), 4) for i in keep], "k_blend": K_BLEND, "dim": int(X.shape[1]),
               "probes": [ids[users[j]] for j in order if users[j] in ids],
               "earlier": {ids[users[i]]: R[users[i]] for i in keep if users[i] in R}}
        json.dump(out, open(a.ratings, "w"), separators=(",", ":"))
        print("exported", len(keep), "people")
        return
    if a.cmd == "probes":
        ids = json.load(open(os.path.join(a.root, "split/rate_ids.json")))
        print(json.dumps([{"rank": i + 1, "user": users[j], "id": ids.get(users[j])} for i, j in enumerate(order[:a.n])], indent=1))
        return
    rated = json.load(open(a.ratings)); pos = {u: i for i, u in enumerate(users)}
    rows = [pos[u] for u in rated if u in pos]; y = np.array([rated[users[r]] for r in rows], float)
    rest = [i for i in range(len(users)) if i not in set(rows)]
    pc = rankdata(z[rest]) / len(rest)
    if len(rows) >= 3:
        pr = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(X[rows], y).predict(X[rest])
        w = len(rows) / (len(rows) + K_BLEND); score = (1 - w) * pc + w * rankdata(pr) / len(pr)
    else:
        score = pc
    ranked = sorted(zip(rest, score), key=lambda t: -t[1])
    print(json.dumps([{"user": users[i], "score": round(float(s), 3)} for i, s in ranked], indent=1))


if __name__ == "__main__":
    main()
