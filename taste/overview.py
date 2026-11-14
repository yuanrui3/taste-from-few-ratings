#!/usr/bin/env python3
"""A rough look at the collected dataset (runs on the cluster after owner_cutouts.py).

Writes OUT/overview.json (counts and distributions) and sample sheets as JPEG:
  sheet_people.jpg   random accounts with an owner, 4 cut-outs each, no handles
  sheet_dropped.jpg  random accounts dropped at the owner step, 4 original photos each
Usage: overview.py PHOTOS CUT QUEUE_TXT FOLLOWING_TSV OUT
"""
import json
import os
import random
import sys

from PIL import Image, ImageDraw

KP_HIP, KP_KNEE, KP_ANKLE, KP_SHOULDER = (11, 12), (13, 14), (15, 16), (5, 6)


def framing(kp):
    """face / upper body / to the knees / full body, from Keypoint R-CNN scores (>2 = visible)."""
    if not kp:
        return "face or close-up"
    vis = lambda idx: any(kp[i][2] > 2 for i in idx)
    if vis(KP_ANKLE):
        return "full body"
    if vis(KP_KNEE):
        return "to the knees"
    if vis(KP_HIP):
        return "to the hips"
    if vis(KP_SHOULDER):
        return "head and shoulders"
    return "face or close-up"


def sheet(rows, path, cell=170, gap=4, label_rows=None):
    w = len(rows[0]) * (cell + gap) + gap
    h = len(rows) * (cell + gap) + gap
    im = Image.new("RGB", (w, h), (245, 245, 242))
    for r, row in enumerate(rows):
        for c, p in enumerate(row):
            if p is None:
                continue
            t = Image.open(p).convert("RGB")
            t.thumbnail((cell, cell))
            im.paste(t, (gap + c * (cell + gap) + (cell - t.width) // 2, gap + r * (cell + gap) + (cell - t.height) // 2))
    im.save(path, quality=85)


def main(photos, cut, queue_txt, following_tsv, out):
    random.seed(7)
    queue = [l.strip() for l in open(queue_txt) if l.strip()]
    follows = {l.split("\t")[0] for l in open(following_tsv) if l and not l.startswith(("#", "username"))}
    owners = json.load(open(os.path.join(cut, "owners.json")))
    have = set(os.listdir(photos))
    kept, dropped = [], []
    for u in queue:
        s = owners.get(u)
        if s and s.get("owner") and (s.get("n_owner_photos") or 0) >= 4:
            kept.append(u)
        else:
            dropped.append(u)
    per_photo_frame, owner_counts, people_counts = {}, [], []
    for u in kept:
        rec = json.load(open(os.path.join(cut, u, "owner.json")))
        owner_counts.append(rec["n_owner_photos"])
        for ph in rec["photos"]:
            f = framing(ph.get("keypoints"))
            per_photo_frame[f] = per_photo_frame.get(f, 0) + 1
            people_counts.append(ph["n_people"])
    n_photos = sum(len([f for f in os.listdir(os.path.join(photos, u)) if f.endswith((".jpg", ".webp", ".png"))])
                   for u in have if os.path.isdir(os.path.join(photos, u)))
    stats = {
        "queue": len(queue), "queue_followed": sum(u in follows for u in queue),
        "accounts_with_photos": len(have & set(queue)), "photos": n_photos,
        "kept": len(kept), "kept_followed": sum(u in follows for u in kept),
        "dropped": len(dropped), "dropped_followed": sum(u in follows for u in dropped),
        "owner_photos_per_account": {str(k): owner_counts.count(k) for k in sorted(set(owner_counts))},
        "owner_photos_total": sum(owner_counts),
        "framing": per_photo_frame,
        "photos_with_other_people": sum(1 for n in people_counts if n > 1),
        "dropped_reasons": {u: (owners.get(u) or {}).get("reason", "no photos") for u in dropped},
    }
    json.dump(stats, open(os.path.join(out, "overview.json"), "w"), indent=1)
    # sample sheet of kept accounts: 24 accounts x 4 cut-outs
    rows = []
    for u in random.sample(kept, min(24, len(kept))):
        rec = json.load(open(os.path.join(cut, u, "owner.json")))
        codes = [p["code"] for p in rec["photos"]]
        pick = random.sample(codes, min(4, len(codes)))
        rows.append([os.path.join(cut, u, c + "_thumb.jpg") for c in pick] + [None] * (4 - len(pick)))
    # two accounts per sheet row
    paired = [rows[i] + (rows[i + 1] if i + 1 < len(rows) else [None] * 4) for i in range(0, len(rows), 2)]
    sheet(paired, os.path.join(out, "sheet_people.jpg"))
    drows = []
    for u in random.sample(dropped, min(8, len(dropped))):
        d = os.path.join(photos, u)
        fs = sorted(f for f in os.listdir(d) if f.endswith((".jpg", ".webp", ".png")))[:4] if os.path.isdir(d) else []
        drows.append([os.path.join(d, f) for f in fs] + [None] * (4 - len(fs)))
    if drows:
        paired = [drows[i] + (drows[i + 1] if i + 1 < len(drows) else [None] * 4) for i in range(0, len(drows), 2)]
        sheet(paired, os.path.join(out, "sheet_dropped.jpg"), cell=130)
    print(json.dumps({k: v for k, v in stats.items() if k != "dropped_reasons"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:6])
