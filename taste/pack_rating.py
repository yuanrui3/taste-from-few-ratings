#!/usr/bin/env python3
"""Build the rating page's files (runs on the cluster after freeze_split.py).

SITE/thumbs/<id>/<k>.jpg  up to 12 cut-out thumbnails of the account's owner (gray background)
SITE/items.json           the frozen rating sequence as [{"id", "pass", "n"}], no handles
Usage: pack_rating.py CUT SPLIT_DIR SITE
"""
import json
import os
import shutil
import sys


def main(cut, split_dir, site):
    seq = json.load(open(os.path.join(split_dir, "rate_order.json")))
    ids = json.load(open(os.path.join(split_dir, "rate_ids.json")))
    n_thumbs = {}
    for u, i in ids.items():
        rec = json.load(open(os.path.join(cut, u, "owner.json")))
        codes = [p["code"] for p in rec["photos"]][:12]
        d = os.path.join(site, "thumbs", i)
        os.makedirs(d, exist_ok=True)
        for k, c in enumerate(codes):
            shutil.copyfile(os.path.join(cut, u, c + "_thumb.jpg"), os.path.join(d, "%d.jpg" % k))
        n_thumbs[u] = len(codes)
    items = [{"id": ids[s["user"]], "pass": s["pass"], "n": n_thumbs[s["user"]]} for s in seq]
    json.dump(items, open(os.path.join(site, "items.json"), "w"))
    print("items %d, accounts %d, thumbnails %d" % (len(items), len(ids), sum(n_thumbs.values())))


if __name__ == "__main__":
    main(*sys.argv[1:4])
