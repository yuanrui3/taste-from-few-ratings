#!/usr/bin/env python3
"""Freeze the train/test split and the rating order, before any rating exists.

- Accounts kept by the owner step (owner on >= 4 photos) are split 75% train / 25% test at
  random, stratified by followed / not followed (seed below).
- Rating order: all kept accounts in random order; about 10% of them (drawn from both splits)
  come back a second time, at least 40 accounts after their first showing, to measure how
  consistent the ratings are.
- Every account gets an opaque id for the rating page; the id -> account map stays private.

Writes OUT/split.json, OUT/rate_order.json, OUT/rate_ids.json and prints the SHA-256 of
split.json (recorded in PROTOCOL.md).
"""
import hashlib
import json
import random
import sys

SEED = 20261113


def main(owners_json, following_tsv, out):
    owners = json.load(open(owners_json))
    # following_tsv: people the rater follows (first column), used only to stratify; "-" for none
    follows = set() if following_tsv == "-" else {l.split("\t")[0] for l in open(following_tsv) if l.strip() and not l.startswith(("#", "username"))}
    kept = sorted(u for u, s in owners.items() if s.get("owner") and (s.get("n_owner_photos") or 0) >= 4)
    rng = random.Random(SEED)
    train, test = [], []
    for group in (sorted(u for u in kept if u in follows), sorted(u for u in kept if u not in follows)):
        g = group[:]
        rng.shuffle(g)
        n_test = round(0.25 * len(g))
        test += g[:n_test]
        train += g[n_test:]
    split = {"seed": SEED, "rule": "75/25 stratified by followed", "train": sorted(train), "test": sorted(test),
             "followed": sorted(u for u in kept if u in follows)}
    blob = json.dumps(split, sort_keys=True, indent=1).encode()
    open(f"{out}/split.json", "wb").write(blob)

    order = kept[:]
    rng.shuffle(order)
    repeats = rng.sample(kept, max(1, round(0.10 * len(kept))))
    seq = [{"user": u, "pass": 1} for u in order]
    for u in repeats:  # insert the second showing at least 40 places after the first
        first = next(i for i, x in enumerate(seq) if x["user"] == u and x["pass"] == 1)
        lo = min(first + 41, len(seq))
        seq.insert(rng.randint(lo, len(seq)), {"user": u, "pass": 2})
    ids = {}
    for u in kept:
        ids[u] = "%08x" % rng.getrandbits(32)
    assert len(set(ids.values())) == len(ids)
    json.dump(seq, open(f"{out}/rate_order.json", "w"), indent=0)
    json.dump(ids, open(f"{out}/rate_ids.json", "w"), indent=0)
    print("kept %d: train %d, test %d (followed in test %d); rating sequence %d (repeats %d)"
          % (len(kept), len(train), len(test), sum(u in follows for u in test), len(seq), len(repeats)))
    print("split.json sha256", hashlib.sha256(blob).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:4])
