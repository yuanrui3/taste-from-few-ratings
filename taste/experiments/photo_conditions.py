#!/usr/bin/env python3
"""Do photo conditions move the full model's score of the same person? (PROTOCOL.md, "Photo conditions").

Photo-level out-of-fold predictions of the adopted full model (taste_full.py: L-14 + SigLIP photo codes
plus the person's SFace code, strength 10,000, same 5 random person folds). Conditions per photo, from
owner.json and the scene-only image: log resolution, log eye distance (face size), share of the crop that
is the person, 2+ people, scene brightness (mean luminance of the photo with every person blanked out;
skin is never measured). Within-person regression of the score on the conditions gives the effect of
conditions with the person held fixed; the adjusted person score removes the person's average conditions'
effect. Usage: photo_conditions.py ROOT -> results/photo_conditions.json
"""
import json
import os
import sys

import cv2
import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from taste_full import kfit, sface_mean  # noqa: E402

COND = ["log resolution", "log face size", "person share of crop", "2+ people", "scene brightness"]


def conditions(root, user, code, ph):
    W, H = ph["size"]
    bg = cv2.imread(os.path.join(root, "cut", user, code + "_bg.jpg"), cv2.IMREAD_GRAYSCALE)
    scene = np.nan
    if bg is not None:
        m = np.abs(bg.astype(int) - 128) > 3  # people are blanked to grey 128
        if m.mean() > 0.1:
            scene = float(bg[m].mean())
    return [np.log(W * H), np.log(max(ph["eye_px"], 1.0)), ph["frac"]["person_of_crop"], float(ph["n_people"] >= 2), scene]


def main(root):
    F = np.load(os.path.join(root, "features.npz"), allow_pickle=True)
    M = np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True)
    users = np.array([str(u) for u in F["user"]]); codes = np.array([str(c) for c in F["code"]])
    rated = json.load(open(os.path.join(root, "split/ratings_effective.json")))["first"]
    lab = sorted(p for p in set(users) if p in rated)
    sf = {p: sface_mean(root, p) for p in lab}
    rows = np.isin(users, lab)
    P = np.c_[F["clip_nocloth"], F["clip_face"], M["siglip_nocloth"], M["siglip_face"],
              np.stack([sf.get(u, np.zeros(128)) for u in users])].astype(np.float64)
    y = np.array([rated.get(u, np.nan) for u in users], float)
    fold = {p: i % 5 for i, p in enumerate(np.random.default_rng(0).permutation(lab))}
    pred = np.full(len(users), np.nan)
    for k in range(5):
        tr = rows & np.array([fold.get(u, -1) != k for u in users]); te = rows & np.array([fold.get(u, -1) == k for u in users])
        pred[te] = kfit(P[tr], y[tr])(P[te])
    recs = {}
    for p in lab:
        for ph in json.load(open(os.path.join(root, "cut", p, "owner.json")))["photos"]:
            recs[(p, ph["code"])] = ph
    idx = np.where(rows)[0]
    C = np.array([conditions(root, users[i], codes[i], recs[(users[i], codes[i])]) for i in idx], float)
    C[:, 4] = np.where(np.isnan(C[:, 4]), np.nanmean(C[:, 4]), C[:, 4])
    Cs = (C - C.mean(0)) / (C.std(0) + 1e-12); pu = users[idx]; pp = pred[idx]
    people = sorted(set(pu)); gi = np.array([people.index(u) for u in pu])
    cm = np.stack([Cs[gi == g].mean(0) for g in range(len(people))]); pm = np.array([pp[gi == g].mean() for g in range(len(people))])
    Ct = Cs - cm[gi]; pt = pp - pm[gi]  # within-person deviations
    beta, *_ = np.linalg.lstsq(Ct, pt, rcond=None)
    r2 = 1 - np.sum((pt - Ct @ beta) ** 2) / np.sum(pt ** 2)
    yp = np.array([rated[u] for u in people], float)
    adj = pm - cm @ beta  # remove the effect of the person's average conditions (pool average = 0)
    base_r = spearmanr(pm, yp)[0]; adj_r = spearmanr(adj, yp)[0]
    move = np.abs(rankdata(adj) - rankdata(pm)) / len(people)
    print("rated people %d, photos %d" % (len(people), len(idx)))
    print("within-person effect on the score (per 1 SD of the condition, in rating points):")
    for name, b in zip(COND, beta):
        print("  %-22s %+.3f" % (name, b))
    print("share of within-person score variance explained by conditions: %.1f%%" % (100 * r2))
    print("between-person spread of the average-conditions effect: SD %.3f rating points (score SD %.3f)" % ((cm @ beta).std(), pm.std()))
    print("Spearman with your ratings: unadjusted %.4f, adjusted %.4f (%+.4f)" % (base_r, adj_r, adj_r - base_r))
    print("people moving 10+ percentile points when adjusted: %d" % int((move >= 0.10).sum()))
    ok = r2 >= 0.05 and adj_r - base_r >= -0.005
    print("rule (conditions explain 5%%+ and Spearman drops by 0.005 or less): %s" % ("adopt the adjustment" if ok else "no"))
    json.dump({"beta": dict(zip(COND, map(float, beta))), "within_r2": float(r2), "spearman": [float(base_r), float(adj_r)],
               "moved_10pct": int((move >= 0.10).sum()), "verdict": "adopt" if ok else "no"},
              open(os.path.join(root, "results/photo_conditions.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
