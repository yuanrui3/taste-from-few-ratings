#!/usr/bin/env python3
"""How often does a person's photo set mix in someone else? (user, 2026-11-13: "some of your data shows
multiple people under one person"). Reads cut/<person>/owner.json and sface.npz; prints counts only.

Flags per kept photo:
  face outlier   the owner face agrees poorly with the person's other photos (median SFace cosine
                 to the other kept photos below 0.30, SFace's same-person line being 0.363)
  whole photo    no body was found for the face, so the whole photo was kept, with 2+ faces in it
  crowded        2+ people detected (the body chosen for the face may be someone else's)
  body elsewhere the chosen body's own nose and eyes (Keypoint R-CNN) lie outside the owner's face box
                 grown by half its size: the cut-out body probably belongs to someone else
Per person: number of face clusters with 2+ photos (average-linkage at the same-person line).
Usage: owner_audit.py ROOT  -> results/owner_audit.json (opaque ids only)
"""
import json
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def main(root):
    ids = json.load(open(os.path.join(root, "split/rate_ids.json")))
    people = sorted(p for p in os.listdir(os.path.join(root, "cut")) if os.path.exists(os.path.join(root, "cut", p, "sface.npz")))
    tot = {"people": 0, "photos": 0, "face outlier": 0, "whole photo": 0, "crowded": 0, "body elsewhere": 0}
    per = []
    for p in people:
        rec = json.load(open(os.path.join(root, "cut", p, "owner.json")))
        d = np.load(os.path.join(root, "cut", p, "sface.npz")); codes = [str(c) for c in d["codes"]]
        E = d["emb"].astype(np.float64); E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
        S = E @ E.T; n = len(codes)
        med = np.array([np.median(np.delete(S[i], i)) for i in range(n)]) if n > 1 else np.ones(n)
        out = {c for c, m in zip(codes, med) if m < 0.30}
        whole = {ph["code"] for ph in rec["photos"] if not ph["instance"] and ph["n_faces"] >= 2}
        crowd = {ph["code"] for ph in rec["photos"] if ph["n_people"] >= 2}
        body = set()
        for ph in rec["photos"]:
            kp = ph.get("keypoints")
            if kp is None:
                continue
            x, y = np.median(np.array(kp)[:3, :2], 0); fx, fy, fw, fh = ph["face_box"]
            if not (fx - fw / 2 <= x <= fx + 1.5 * fw and fy - fh / 2 <= y <= fy + 1.5 * fh):
                body.add(ph["code"])
        clusters = 1
        if n >= 3:
            lab = fcluster(linkage(1 - S[np.triu_indices(n, 1)], "average"), t=1 - 0.363, criterion="distance")
            clusters = int(sum(np.bincount(lab)[1:] >= 2))
        tot["people"] += 1; tot["photos"] += n
        tot["face outlier"] += len(out); tot["whole photo"] += len(whole); tot["crowded"] += len(crowd); tot["body elsewhere"] += len(body)
        per.append({"id": ids.get(p, "?"), "photos": n, "face_outliers": sorted(out), "whole_photo": sorted(whole),
                    "crowded": len(crowd), "body_elsewhere": sorted(body), "face_clusters": clusters, "median_pair_cos": round(float(np.median(S[np.triu_indices(n, 1)])), 3) if n > 1 else None})
    print("people %d, kept photos %d" % (tot["people"], tot["photos"]))
    field = {"face outlier": "face_outliers", "whole photo": "whole_photo", "body elsewhere": "body_elsewhere"}
    for k in ("face outlier", "whole photo", "crowded", "body elsewhere"):
        anyp = sum(1 for r in per if (len(r[field[k]]) if k in field else r["crowded"]) > 0)
        print("  %-14s photos %4d (%.1f%%), people with any %d" % (k, tot[k], 100 * tot[k] / tot["photos"], anyp))
    bad = [r for r in per if r["face_outliers"] or r["body_elsewhere"] or r["face_clusters"] >= 2]
    print("  people with any face outlier, body elsewhere or 2+ face clusters: %d" % len(bad))
    print("  people with 2+ face clusters of 2+ photos: %d" % sum(r["face_clusters"] >= 2 for r in per))
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    json.dump({"totals": tot, "people": per}, open(os.path.join(root, "results/owner_audit.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
