#!/usr/bin/env python3
"""Pick the photos to rate for sexual implication / sexual attractiveness and build the page's files
(GPU cluster). See docs/ADDENDUM-sexual-attractiveness.md "Sexual implication and sexual attractiveness, per photo". No ratings used.

1. second apparent-age screen: zero-shot "child or teenager" vs "adult" with the OpenAI CLIP text
   encoder on the crop codes saved by suggest_range.py (the first screen is the InsightFace estimate)
2. photos with one person in them; per person their lowest and highest photo by the stand-in score;
   all test people of the frozen split, N_TRAIN training people at random (seeded)
3. N_REPEAT photos come back later; random order
4. writes SITE/img/<id>.jpg (owner's crop from the original photo), SITE/items.json ([{"id","pass","g"}]
   with opaque photo and person ids, nothing else), and split/suggest_set.json (the private
   id -> account/photo map and the record)

Usage: suggest_pack.py [--root R] [--site R/suggest_site]
"""
import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from owner_cutouts import load as load_photo  # noqa: E402
from suggest_range import COMBINED, pct_rank  # noqa: E402

SEED = 20261116
N_TRAIN, N_REPEAT = 130, 40
AGE_MIN, YOUNG_PERSON, YOUNG_PHOTO = 21.0, 0.5, 0.7
YOUNG = ["a photo of a child", "a photo of a teenager", "a photo of a high school student"]
ADULT = ["a photo of an adult woman", "a photo of an adult man", "a photo of a grown-up person"]
MAX_SIDE = 1100


def young_prob(codes):
    import open_clip
    import torch
    m, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
    m = m.eval()
    tok = open_clip.get_tokenizer("ViT-L-14-quickgelu")
    with torch.no_grad():
        T = torch.stack([torch.nn.functional.normalize(
            torch.nn.functional.normalize(m.encode_text(tok(p)).float(), dim=-1).mean(0), dim=-1)
            for p in (YOUNG, ADULT)])
        return (100.0 * torch.from_numpy(codes.astype(np.float32)) @ T.T).softmax(-1)[:, 0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--site", default=None)
    a = ap.parse_args()
    R = a.root
    site = a.site or os.path.join(R, "suggest_site")
    d = dict(np.load(os.path.join(R, "results/suggest_scores.npz"), allow_pickle=False))
    u, code = d["user"], d["code"]
    X = {k: d[k].astype(float) for k in ("laion_nsfw", "zs_suggest", "bare", "nn_area")}
    X["p_sexyporn"] = d["p_sexy"].astype(float) + d["p_porn"].astype(float)
    comb = np.mean([pct_rank(X[k]) for k in COMBINED], 0)

    py = young_prob(d["oai"])
    people = np.unique(u)
    med_age = {k: float(np.median(d["age"][u == k])) for k in people}
    med_young = {k: float(np.median(py[u == k])) for k in people}
    aside = sorted(k for k in people if med_age[k] < AGE_MIN or med_young[k] >= YOUNG_PERSON)
    print("zero-shot young: photo quantiles", np.percentile(py, [50, 90, 95, 99, 100]).round(3).tolist(),
          "| person-median quantiles", np.percentile(list(med_young.values()), [50, 90, 95, 99, 100]).round(3).tolist())
    print("people set aside: %d (age estimate under %g: %d; zero-shot median >= %g: %d); photos dropped at >= %g: %d"
          % (len(aside), AGE_MIN, sum(med_age[k] < AGE_MIN for k in people), YOUNG_PERSON,
             sum(med_young[k] >= YOUNG_PERSON for k in people), YOUNG_PHOTO, int((py >= YOUNG_PHOTO).sum())))

    elig = (d["n_people"] <= 1) & (py < YOUNG_PHOTO) & ~np.isin(u, aside)
    split = json.load(open(os.path.join(R, "split/split.json")))
    test, train = set(split["test"]), set(split["train"])
    ok = [k for k in people if (elig & (u == k)).sum() >= 2]
    rng = np.random.default_rng(SEED)
    tr = sorted(k for k in ok if k in train)
    te = sorted(k for k in ok if k in test)
    chosen = te + sorted(rng.choice(tr, size=min(N_TRAIN, len(tr)), replace=False).tolist())
    reserve = sorted(set(tr) - set(chosen))
    rows = []
    for k in chosen:
        idx = np.where(elig & (u == k))[0]
        o = idx[np.argsort(comb[idx], kind="stable")]
        for which, i in (("low", o[0]), ("high", o[-1])):
            pid = hashlib.sha256(("%d:%s:%s" % (SEED, k, code[i])).encode()).hexdigest()[:12]
            rows.append({"id": pid, "user": str(k), "code": str(code[i]), "split": "test" if k in test else "train",
                         "own": which, "standin": round(float(comb[i]), 4)})
    assert len({r["id"] for r in rows}) == len(rows)
    first = [int(i) for i in rng.permutation(len(rows))]
    # "g" is an opaque person id: the page uses it to pass over a person answered "may be under 18"
    grp = {r["id"]: hashlib.sha256(("%d:person:%s" % (SEED, r["user"])).encode()).hexdigest()[:8] for r in rows}
    seq = [{"id": rows[i]["id"], "pass": 1, "g": grp[rows[i]["id"]]} for i in first]
    pos = {s["id"]: p for p, s in enumerate(seq)}
    n = len(seq)
    early = [i for i in range(len(rows)) if pos[rows[i]["id"]] <= n - 61]  # a repeat comes 60+ screens later
    rep = sorted(int(i) for i in rng.choice(early, size=min(N_REPEAT, len(early)), replace=False))
    ins = sorted(((int(rng.integers(max(pos[rows[i]["id"]] + 60, n // 2), n + 1)), rows[i]["id"]) for i in rep),
                 reverse=True)
    for p, pid in ins:
        seq.insert(p, {"id": pid, "pass": 2, "g": grp[pid]})
    os.makedirs(os.path.join(site, "img"), exist_ok=True)
    for r in rows:
        rec = {p["code"]: p for p in json.load(open(os.path.join(R, "cut", r["user"], "owner.json")))["photos"]}[r["code"]]
        pdir = os.path.join(R, "photos", r["user"])
        fn = [x for x in os.listdir(pdir) if os.path.splitext(x)[0] == r["code"]][0]
        rgb = load_photo(os.path.join(pdir, fn))
        x0, y0, x1, y1 = rec["crop"]
        c = rgb[y0:y1, x0:x1]
        f = min(1.0, MAX_SIDE / max(c.shape[:2]))
        if f < 1:
            c = cv2.resize(c, (round(c.shape[1] * f), round(c.shape[0] * f)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(site, "img", r["id"] + ".jpg"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump(seq, open(os.path.join(site, "items.json"), "w"))
    rec = {"seed": SEED, "rule": "one-person photos; per person lowest and highest by the stand-in score; "
           "all test people, %d training people at random; %d repeats" % (N_TRAIN, N_REPEAT),
           "photos": rows, "sequence": seq, "set_aside": aside, "reserve_training_people": reserve}
    path = os.path.join(R, "split/suggest_set.json")
    json.dump(rec, open(path, "w"), indent=0, sort_keys=True)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    lv = np.histogram([r["standin"] for r in rows], bins=np.percentile(comb, [0, 20, 40, 60, 80, 100]))[0]
    print("photos %d (test %d, train %d) of %d people; screens %d; per fifth of the stand-in score %s; reserve people %d"
          % (len(rows), sum(r["split"] == "test" for r in rows), sum(r["split"] == "train" for r in rows),
             len(chosen), len(seq), lv.tolist(), len(reserve)))
    print("suggest_set.json sha256", sha)


if __name__ == "__main__":
    main()
