#!/usr/bin/env python3
"""Turn the rating page's log into the ratings the models read. Standard library only.

Input: RATINGS.jsonl from rate_server.py (one line per click: {"id", "pass", "rating"} or
{"skip": true} or {"undo": true}) and SPLIT_DIR/rate_ids.json (person -> opaque id).
The last entry per (id, pass) counts, after undos. Pass 1 is the rating used for training and
testing; pass 2 (the repeats) only measures how consistent the rater is.
Output: SPLIT_DIR/ratings_effective.json = {"first": {person: rating}, "second": {...},
"skipped": [person, ...]}.
Usage: ratings_effective.py RATINGS.jsonl SPLIT_DIR
"""
import json
import os
import sys


def main(ratings_path, split_dir):
    who = {v: k for k, v in json.load(open(os.path.join(split_dir, "rate_ids.json"))).items()}
    last = {}
    for line in open(ratings_path):
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["id"], int(r.get("pass", 1)))
        if r.get("undo"):
            last.pop(key, None)
        else:
            last[key] = r
    out = {"first": {}, "second": {}, "skipped": []}
    for (i, p), r in sorted(last.items()):
        person = who.get(i)
        if person is None:
            continue
        if r.get("skip") or r.get("rating") is None:
            if p == 1:
                out["skipped"].append(person)
        else:
            out["first" if p == 1 else "second"][person] = r["rating"]
    json.dump(out, open(os.path.join(split_dir, "ratings_effective.json"), "w"), indent=1, sort_keys=True)
    print("first pass %d, repeats %d, skipped %d" % (len(out["first"]), len(out["second"]), len(out["skipped"])))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
