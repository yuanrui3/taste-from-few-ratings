#!/usr/bin/env python3
"""Do the images the models see show more than one person? Counts faces (YuNet, as in owner_cutouts.py)
in each kept photo's cut-out (owner on grey: rating page and appearance model) and face crop (CLIP /
SigLIP face codes). Usage: cutout_faces.py ROOT -> results/cutout_faces.json (opaque ids only)."""
import json
import os
import sys

import cv2
import numpy as np


def faces(det, bgr):
    H, W = bgr.shape[:2]; f = max(1.0, 960 / max(H, W))
    img = cv2.resize(bgr, (round(W * f), round(H * f)), interpolation=cv2.INTER_CUBIC) if f > 1 else bgr
    det.setInputSize((img.shape[1], img.shape[0])); _, fs = det.detect(img)
    return [r for r in (fs if fs is not None else []) if r[14] >= 0.8 and np.linalg.norm(r[4:6] - r[6:8]) / f >= 6]


def main(root):
    det = cv2.FaceDetectorYN.create(os.path.join(root, "models/opencv_zoo/face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.5, 0.3, 50)
    ids = json.load(open(os.path.join(root, "split/rate_ids.json")))
    people = sorted(p for p in os.listdir(os.path.join(root, "cut")) if os.path.exists(os.path.join(root, "cut", p, "owner.json")))
    n = {"photos": 0, "cut 2+": 0, "face crop 2+": 0}; per = []
    for p in people:
        rec = json.load(open(os.path.join(root, "cut", p, "owner.json"))); mc, mf = [], []
        for ph in rec["photos"]:
            c = ph["code"]; n["photos"] += 1
            if len(faces(det, cv2.imread(os.path.join(root, "cut", p, c + "_cut.jpg")))) >= 2:
                mc.append(c)
            if len(faces(det, cv2.imread(os.path.join(root, "cut", p, c + "_face.jpg")))) >= 2:
                mf.append(c)
        n["cut 2+"] += len(mc); n["face crop 2+"] += len(mf)
        per.append({"id": ids.get(p, "?"), "photos": len(rec["photos"]), "cut_multi": mc, "face_multi": mf})
    print("kept photos %d" % n["photos"])
    for k, f in (("cut 2+", "cut_multi"), ("face crop 2+", "face_multi")):
        print("  %-13s %4d photos (%.1f%%), people with any %d, people with 3+ such photos %d" % (
            k, n[k], 100 * n[k] / n["photos"], sum(bool(r[f]) for r in per), sum(len(r[f]) >= 3 for r in per)))
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    json.dump({"totals": n, "people": per}, open(os.path.join(root, "results/cutout_faces.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
