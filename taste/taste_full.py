#!/usr/bin/env python3
"""The full model: one rater's taste from all their ratings, applied to people they have not rated.

Adopted 2026-11-13 (PROTOCOL.md, "Afternoon exploration", A): ridge regression trained on single
photos, each photo described by OpenCLIP ViT-L-14 and SigLIP codes of the no-clothing cut-out and
the face crop, plus the person's mean SFace face-recognition code; a person's score is the mean
over their photos. Cross-validated Spearman 0.669 on the first rater's training accounts (0.636
for the previous account-level model); labelled look at the held-out accounts 0.723.

Inputs (cluster folder): features.npz, features_more_vith_siglip.npz, cut/<person>/sface.npz,
split/ratings_effective.json, split/rate_ids.json. Every person in features.npz is scored;
rated people are scored by a model that did not see their own rating (5-fold), the others by
the model trained on all ratings.

Usage: taste_full.py [--root .] [--out results/full_scores.json]
Output: [{"id": opaque id, "score": standardized score, "rated": bool}, ...], highest first.
"""
import argparse
import json
import os

import numpy as np

ALPHA = 10000.0


def kfit(Xtr, ytr):
    """Ridge with intercept in kernel form, after standardizing on the training rows."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = (Xtr - mu) / sd; ym = ytr.mean()
    l, Q = np.linalg.eigh(A @ A.T); c = Q @ ((Q.T @ (ytr - ym)) / (l + ALPHA))
    w = A.T @ c
    return lambda X: ym + ((X - mu) / sd) @ w


def sface_mean(root, person):
    e = np.load(os.path.join(root, "cut", person, "sface.npz"))["emb"].astype(np.float64)
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12; m = e.mean(0)
    return m / (np.linalg.norm(m) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results/full_scores.json")
    a = ap.parse_args()
    F = np.load(os.path.join(a.root, "features.npz"), allow_pickle=True)
    M = np.load(os.path.join(a.root, "features_more_vith_siglip.npz"), allow_pickle=True)
    users = np.array([str(u) for u in F["user"]])
    people = sorted(set(users))
    sf = {p: sface_mean(a.root, p) for p in people}
    P = np.c_[F["clip_nocloth"], F["clip_face"], M["siglip_nocloth"], M["siglip_face"],
              np.stack([sf[u] for u in users])].astype(np.float64)
    rated = json.load(open(os.path.join(a.root, "split/ratings_effective.json")))["first"]
    ids = json.load(open(os.path.join(a.root, "split/rate_ids.json")))
    lab = [p for p in people if p in rated]
    fold = {p: i % 5 for i, p in enumerate(np.random.default_rng(0).permutation(lab))}
    rows = np.isin(users, lab); y_rows = np.array([rated.get(u, np.nan) for u in users], float)
    score = {}
    for k in range(5):  # rated people: scored by a model that never saw their rating
        tr = rows & np.array([fold.get(u, -1) != k for u in users]); f = kfit(P[tr], y_rows[tr])
        for p in lab:
            if fold[p] == k:
                score[p] = float(f(P[users == p]).mean())
    f = kfit(P[rows], y_rows[rows])  # everyone else: the model trained on all ratings
    for p in people:
        if p not in score:
            score[p] = float(f(P[users == p]).mean())
    v = np.array(list(score.values())); m, s = v.mean(), v.std() + 1e-12
    out = sorted(({"id": ids.get(p, "?"), "score": round((score[p] - m) / s, 3), "rated": p in rated} for p in people),
                 key=lambda r: -r["score"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("scored %d people (%d rated, %d not rated) -> %s" % (len(out), len(lab), len(out) - len(lab), a.out))


if __name__ == "__main__":
    main()
