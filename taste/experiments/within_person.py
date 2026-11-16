#!/usr/bin/env python3
"""How much do the models' scores vary across one person's photos? (PROTOCOL.md, "Within-person photo
variation"). No new ratings; training accounts only.
Usage: within_person.py ROOT -> results/within_person.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from improve_r2 import kfit  # noqa: E402
from overview import framing  # noqa: E402
from sface_input import sface_means  # noqa: E402
from train_eval import folds, load, photo_index  # noqa: E402

ALPHA = 10000.0
FRAMES = ["face or close-up", "head and shoulders", "to the hips", "to the knees", "full body"]


def oof(P, owner, y, train, group, seed):
    fo = folds(train, group, seed=seed); pp = np.zeros(len(owner))
    for k in range(5):
        tr, te = np.where(fo != k)[0], np.where(fo == k)[0]
        trp, tep = np.isin(owner, tr), np.isin(owner, te)
        pp[tep] = kfit(P[trp], y[owner[trp]])(P[tep], ALPHA)
    return pp


def icc1(x, g):
    """One-way random-effects ICC(1) with unequal group sizes."""
    groups = [x[g == i] for i in np.unique(g)]; k = len(groups); N = len(x); m = x.mean()
    ssb = sum(len(v) * (v.mean() - m) ** 2 for v in groups); ssw = sum(((v - v.mean()) ** 2).sum() for v in groups)
    msb, msw = ssb / (k - 1), ssw / (N - k); n0 = (N - sum(len(v) ** 2 for v in groups) / N) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def within_rel(a, b, g):
    rs = [spearmanr(a[g == i], b[g == i])[0] for i in np.unique(g) if (g == i).sum() >= 3]
    rs = [r for r in rs if np.isfinite(r)]
    return float(np.mean(rs)), len(rs)


def main(root):
    from group_transfer import scut_crowd_models
    F, train, test, first, R, group, split, same = load(root); idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True))
    O = dict(np.load(os.path.join(root, "features_faceown.npz"), allow_pickle=True))
    y = np.array([first[u] for u in train], float); SFt = sface_means(root, train)
    rows = np.concatenate([idx[u] for u in train]); owner = np.concatenate([[i] * len(idx[u]) for i, u in enumerate(train)])
    users = np.array([str(u) for u in F["user"]])[rows]; codes = np.array([str(c) for c in F["code"]])[rows]
    full = np.c_[F["clip_nocloth"][rows], O["clip_face"][rows], M["siglip_nocloth"][rows], O["siglip_face"][rows], SFt[owner]].astype(np.float64)
    clip_only = np.c_[F["clip_nocloth"][rows], O["clip_face"][rows]].astype(np.float64)
    sig_only = np.c_[M["siglip_nocloth"][rows], O["siglip_face"][rows]].astype(np.float64)
    s = {"full seed 0": oof(full, owner, y, train, group, 0), "full seed 1": oof(full, owner, y, train, group, 1),
         "full seed 2": oof(full, owner, y, train, group, 2)}
    print("full model fitted", flush=True)
    s["CLIP only"] = oof(clip_only, owner, y, train, group, 0); s["SigLIP only"] = oof(sig_only, owner, y, train, group, 0)
    full_mean = (s["full seed 0"] + s["full seed 1"] + s["full seed 2"]) / 3
    # crowd score per photo (raw-photo face crops, as the adopted crowd model reads them)
    sf_photo = np.full((len(rows), 128), np.nan)
    for u in set(users):
        d = np.load(os.path.join(root, "cut", u, "sface.npz")); pos = {str(c): i for i, c in enumerate(d["codes"])}
        for j in np.where(users == u)[0]:
            if codes[j] in pos:
                sf_photo[j] = d["emb"][pos[codes[j]]]
    ok = ~np.isnan(sf_photo).any(1)
    clipf = F["clip_face"][rows].astype(np.float64); clipf /= np.linalg.norm(clipf, axis=1, keepdims=True)
    sfp = sf_photo[ok] / np.linalg.norm(sf_photo[ok], axis=1, keepdims=True)
    crowd = np.full(len(rows), np.nan); crowd[ok] = scut_crowd_models(root)(clipf[ok], sfp)
    out = {"photos": int(len(rows)), "people": int(len(train)),
           "icc": {"full model": icc1(full_mean, owner), "crowd (per photo)": icc1(crowd[ok], owner[ok])}}
    print("photos %d of %d people" % (len(rows), len(train)))
    for k, v in out["icc"].items():
        print("ICC(1) %-18s %.3f  (share of photo-score variance between people; human judges about 0.5 or lower)" % (k, v))
    out["within_reliability"] = {}
    for name, a, b in (("full: seed 0 vs seed 1", s["full seed 0"], s["full seed 1"]), ("CLIP-only vs SigLIP-only", s["CLIP only"], s["SigLIP only"])):
        r, n = within_rel(a, b, owner); out["within_reliability"][name] = {"mean_spearman": r, "people": n}
        print("within-person ordering, %-26s mean Spearman %.3f over %d people" % (name, r, n))
    recs = {}
    for u in set(users):
        for ph in json.load(open(os.path.join(root, "cut", u, "owner.json")))["photos"]:
            recs[(u, ph["code"])] = framing(ph.get("keypoints"))
    fr = np.array([recs.get((u, c), "face or close-up") for u, c in zip(users, codes)])
    D = np.stack([(fr == f).astype(float) for f in FRAMES], 1)
    dev = full_mean - np.array([full_mean[owner == i].mean() for i in owner])
    Dc = D - np.stack([D[owner == i].mean(0) for i in owner])
    beta, *_ = np.linalg.lstsq(Dc, dev, rcond=None); r2 = 1 - ((dev - Dc @ beta) ** 2).sum() / (dev ** 2).sum()
    out["framing_r2_within"] = float(r2); out["framing_effect"] = {f: float(b - beta.mean()) for f, b in zip(FRAMES, beta)}
    out["framing_share_of_photos"] = {f: float((fr == f).mean()) for f in FRAMES}
    print("share of within-person score variance explained by framing: %.1f%%" % (100 * r2))
    print("framing effect on the same person's photo score (rating points, relative): " + ", ".join("%s %+.3f" % (f, out["framing_effect"][f]) for f in FRAMES))
    sd_w = float(np.sqrt(np.mean([full_mean[owner == i].var() for i in range(len(train))]))); sd_b = float(np.std([full_mean[owner == i].mean() for i in range(len(train))]))
    out["sd_within"], out["sd_between"] = sd_w, sd_b
    print("typical spread of one person's photo scores %.3f vs spread between people's means %.3f (rating points)" % (sd_w, sd_b))
    json.dump(out, open(os.path.join(root, "results/within_person.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
