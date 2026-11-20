#!/usr/bin/env python3
"""Photo sets and later-session repeats (PROTOCOL.md, "Photo sets and later-session repeats"; planned before picking).

  photosets.py pick ROOT NEW1 NEW2   (GPU job: zero-shot age check per photo)
      40 repeat people from ROOT's pool, 80 new people from the discovery folders with 4 photos each;
      (public copy: NEW1/NEW2 hold Instagram similar-account suggestions collected for a separate,
      unpublished study; its helper `discover.py` is left out with the other Instagram tools, so `pick`
      does not run from this repository; `eval` does)
      writes ROOT/site_photosets/{index.html, items.json, thumbs/}, the frozen predictions
      ROOT/results/photosets_pred.json (opaque ids; SHA-256 printed) and the private map
      ROOT/results/photosets_map.json
  photosets.py eval ROOT RATINGS     -> ROOT/results/photosets_eval.json
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

SEED = 20261119
N_NEW, N_REP, K, N_WITHIN, GAP = 80, 40, 4, 8, 30
YOUNG_PHOTO = 0.7
ALPHA = 1e4


def opaque(prefix, *parts):
    return prefix + hashlib.sha1(("|".join(str(p) for p in (SEED,) + parts)).encode()).hexdigest()[:9]


def shown_since(R, N2):
    """Pool people shown on any page after the first rating (2026-11-12)."""
    ids = json.load(open(os.path.join(R, "split/rate_ids.json")))
    inv = {v: k for k, v in ids.items()}
    s = {inv[i] for i in json.load(open(os.path.join(R, "results/onboard_data.json")))["probes"] if i in inv}
    s |= {p["user"] for p in json.load(open(os.path.join(R, "results/photo_pairs_set.json")))}
    for f in ("split/suggest_set.json", "split/suggest_set2.json"):
        s |= {p["user"] for p in json.load(open(os.path.join(R, f)))["photos"]}
    s |= {v["user"] for v in json.load(open(os.path.join(N2, "results/rec17_map.json"))).values() if v["folder"] == "."}
    return s


def rated_discovery(N1, N2):
    s = set()
    for f in (os.path.join(N1, "results/discover_check_map.json"), os.path.join(N1, "results/discover_check3_map.json")):
        if os.path.exists(f):
            s |= {v if isinstance(v, str) else v["user"] for v in json.load(open(f)).values()}
    s |= {v["user"] for v in json.load(open(os.path.join(N2, "results/rec17_map.json"))).values() if v["folder"] != "."}
    return s


def pick(a):
    import open_clip
    import torch
    from PIL import Image
    from bag_loss import BagFit
    from discover import AGE_MIN, YOUNG_PERSON, design, kept_people, same_share
    from owner_cutouts import load as load_photo
    from suggest_pack import ADULT, YOUNG
    R, N1, N2 = (os.path.abspath(x) for x in (a.root, a.new1, a.new2))
    rng = np.random.default_rng(SEED)
    rated = json.load(open(os.path.join(R, "split/ratings_effective.json")))["first"]

    # repeats
    seen = shown_since(R, N2)
    rep_pool = sorted(u for u in kept_people(R) if u in rated and u not in seen)
    reps = sorted(rng.choice(rep_pool, N_REP, replace=False).tolist())
    print("repeat pool: %d rated people not shown since; %d drawn" % (len(rep_pool), len(reps)), flush=True)

    # pool face codes for the same-person rule
    def emb(folder, u):
        f = os.path.join(folder, "cut", u, "sface.npz")
        return np.load(f)["emb"] if os.path.exists(f) else None
    pool_E = {u: emb(R, u) for u in kept_people(R)}
    pool_E = {u: e for u, e in pool_E.items() if e is not None and len(e) >= 2}
    pool_W = {u: same_share(e) for u, e in pool_E.items()}

    # new people: unrated discovery candidates who pass the age screen, in a seeded order
    done = rated_discovery(N1, N2)
    cands = []
    for tag, folder in (("discover", N1), ("discover2", N2)):
        age = json.load(open(os.path.join(folder, "results/age.json")))
        for u in kept_people(folder):
            g = age.get(u, {})
            if u in done or not g.get("n") or g["age_median"] < AGE_MIN or g["young_median"] >= YOUNG_PERSON:
                continue
            cands.append((tag, folder, u))
    order = rng.permutation(len(cands))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, _, pre = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
    m = m.eval().to(dev)
    tok = open_clip.get_tokenizer("ViT-L-14-quickgelu")
    with torch.no_grad():
        T = torch.stack([torch.nn.functional.normalize(
            torch.nn.functional.normalize(m.encode_text(tok(p).to(dev)).float(), dim=-1).mean(0), dim=-1) for p in (YOUNG, ADULT)])
    feats = {}
    for tag, folder in (("discover", N1), ("discover2", N2)):
        Fn = np.load(os.path.join(folder, "features.npz"), allow_pickle=True)
        feats[tag] = {(str(u), str(c)) for u, c in zip(Fn["user"], Fn["code"])}
    chosen, why = [], {"same person as a pool account": 0, "fewer than 4 eligible photos": 0}
    for j in order:
        if len(chosen) >= N_NEW:
            break
        tag, folder, u = cands[j]
        e = emb(folder, u)
        if e is not None and len(e) >= 2:
            w = same_share(e)
            if any(same_share(e, pe) >= 0.9 * min(w, pool_W[v]) and min(w, pool_W[v]) > 0 for v, pe in pool_E.items()):
                why["same person as a pool account"] += 1
                continue
        rec = json.load(open(os.path.join(folder, "cut", u, "owner.json")))
        pdir = os.path.join(folder, "photos", u)
        files = {os.path.splitext(x)[0]: x for x in os.listdir(pdir)}
        ok, ims = [], []
        for ph in rec["photos"]:
            c = ph["code"]
            if (ph.get("n_people") or 1) > 1 or c not in files or (u, c) not in feats[tag] \
                    or not os.path.exists(os.path.join(folder, "cut", u, c + "_thumb.jpg")):
                continue
            rgb = load_photo(os.path.join(pdir, files[c]))
            if rgb is None:
                continue
            x0, y0, x1, y1 = ph["crop"]
            ok.append(c)
            ims.append(Image.fromarray(np.ascontiguousarray(rgb[y0:y1, x0:x1])))
        if ims:
            with torch.no_grad():
                o = torch.nn.functional.normalize(m.encode_image(torch.stack([pre(i) for i in ims]).to(dev)).float(), dim=-1)
                young = (100.0 * o @ T.T).softmax(-1)[:, 0].cpu().numpy()
            ok = [c for c, y in zip(ok, young) if y < YOUNG_PHOTO]
        if len(ok) < K:
            why["fewer than 4 eligible photos"] += 1
            continue
        shown = sorted(rng.choice(ok, K, replace=False).tolist())
        chosen.append({"tag": tag, "folder": folder, "user": u, "codes": shown})
    assert len(chosen) == N_NEW, "only %d new people qualified" % len(chosen)
    print("new people: %d candidates; %d chosen after passing over %s" % (len(cands), len(chosen), why), flush=True)

    # frozen predictions: full model trained on all rated pool people, rho = 1 and rho = 0
    P, users = design(R)
    lab = sorted(p for p in set(users) if p in rated)
    pos = {p: i for i, p in enumerate(lab)}
    rows = np.isin(users, lab)
    owner = np.array([pos[u] for u in users[rows]])
    y = np.array([rated[p] for p in lab], float)
    f1, f0 = BagFit(P[rows], owner, y, 1.0), BagFit(P[rows], owner, y, 0.0)
    des = {}
    for tag, folder in (("discover", N1), ("discover2", N2)):
        Pn, un = design(folder)
        cn = np.array([str(c) for c in np.load(os.path.join(folder, "features.npz"), allow_pickle=True)["code"]])
        des[tag] = (Pn, un, cn)

    site = os.path.join(R, "site_photosets")
    shutil.rmtree(site, ignore_errors=True)
    os.makedirs(site)
    shutil.copyfile(os.path.join(os.path.dirname(HERE), "photoset.html"), os.path.join(site, "index.html"))
    items, pred, idmap = [], {"seed": SEED, "repeats": [], "sets": []}, {}
    for u in reps:
        i = opaque("r", ".", u)
        rec = json.load(open(os.path.join(R, "cut", u, "owner.json")))
        codes = [p["code"] for p in rec["photos"]][:12]  # as on the first rating page (pack_rating.py)
        os.makedirs(os.path.join(site, "thumbs", i))
        for k, c in enumerate(codes):
            shutil.copyfile(os.path.join(R, "cut", u, c + "_thumb.jpg"), os.path.join(site, "thumbs", i, "%d.jpg" % k))
        items.append({"id": i, "pass": 1, "kind": "rate", "n": len(codes), "g": opaque("g", ".", u)})
        pred["repeats"].append({"id": i, "first": rated[u]})
        idmap[i] = {"user": u, "folder": "."}
    for p in chosen:
        i = opaque("s", p["tag"], p["user"])
        Pn, un, cn = des[p["tag"]]
        rr = [int(np.where((un == p["user"]) & (cn == c))[0][0]) for c in p["codes"]]
        s1, s0 = f1.predict(Pn[rr], ALPHA), f0.predict(Pn[rr], ALPHA)
        os.makedirs(os.path.join(site, "thumbs", i))
        for k, c in enumerate(p["codes"]):
            shutil.copyfile(os.path.join(p["folder"], "cut", p["user"], c + "_thumb.jpg"), os.path.join(site, "thumbs", i, "%d.jpg" % k))
        items.append({"id": i, "pass": 1, "kind": "set", "n": K, "order": rng.permutation(K).tolist(), "g": opaque("g", p["tag"], p["user"])})
        pred["sets"].append({"id": i, "s1": [round(float(v), 5) for v in s1], "s0": [round(float(v), 5) for v in s0],
                             "person_pred": round(float(s1.mean()), 5)})
        idmap[i] = {"user": p["user"], "folder": p["tag"], "codes": p["codes"]}
    seq = [items[j] for j in rng.permutation(len(items))]
    at = {it["id"]: j for j, it in enumerate(seq)}
    n = len(seq)
    early = [it["id"] for it in seq if it["kind"] == "set" and at[it["id"]] <= n - GAP - 1]
    again = sorted(rng.choice(early, N_WITHIN, replace=False).tolist())
    ins = sorted(((int(rng.integers(max(at[i] + GAP, n // 2), n + 1)), i) for i in again), reverse=True)
    byid = {it["id"]: it for it in items}
    for q, i in ins:
        o = byid[i]["order"]
        o2 = o
        while o2 == o:
            o2 = rng.permutation(K).tolist()
        seq.insert(q, dict(byid[i], **{"pass": 2, "order": o2}))
    json.dump(seq, open(os.path.join(site, "items.json"), "w"))
    path = os.path.join(R, "results/photosets_pred.json")
    json.dump(pred, open(path, "w"), indent=0, sort_keys=True)
    json.dump(idmap, open(os.path.join(R, "results/photosets_map.json"), "w"), indent=1)
    agree = np.mean([np.argmax(p["s1"]) == np.argmax(p["s0"]) for p in pred["sets"]])
    print("screens %d (%d repeats of pool people, %d photo sets, %d sets shown again); rho 0 and rho 1 agree on the top photo for %.0f%% of sets"
          % (len(seq), len(reps), len(chosen), N_WITHIN, 100 * agree))
    print("photosets_pred.json sha256", hashlib.sha256(open(path, "rb").read()).hexdigest())


def evaluate(a):
    from scipy.stats import binomtest, pearsonr, spearmanr
    R = a.root
    P = json.load(open(os.path.join(R, "results/photosets_pred.json")))
    items = json.load(open(os.path.join(R, "site_photosets/items.json")))
    g_of = {it["id"]: it["g"] for it in items}
    eff = {}
    for line in open(a.ratings):
        r = json.loads(line)
        k = (r["id"], r.get("pass", 1))
        if r.get("undo"):
            eff.pop(k, None)
        else:
            eff[k] = r
    gone = {g_of[k[0]] for k, v in eff.items() if v.get("reason") == "under18"}
    first = {k[0]: v for k, v in eff.items() if k[1] == 1 and not v.get("skip") and g_of[k[0]] not in gone}
    second = {k[0]: v for k, v in eff.items() if k[1] == 2 and not v.get("skip") and g_of[k[0]] not in gone}
    g = np.random.default_rng(SEED)
    out = {"removed_under18": len(gone), "skipped": sum(1 for k, v in eff.items() if v.get("skip") and v.get("reason") != "under18")}

    def boot(x, y, fn, n=2000):
        idx = np.arange(len(x))
        v = [fn(x[b], y[b]) for b in (g.choice(idx, len(idx)) for _ in range(n))]
        return [round(float(q), 4) for q in np.nanpercentile(v, [2.5, 97.5])]

    rep = [p for p in P["repeats"] if p["id"] in first]
    x0 = np.array([p["first"] for p in rep], float)
    x1 = np.array([first[p["id"]]["rating"] for p in rep], float)
    rs = float(spearmanr(x0, x1)[0])
    out["repeats"] = {"n": len(rep), "spearman": round(rs, 4), "spearman_ci95": boot(x0, x1, lambda a_, b_: spearmanr(a_, b_)[0]),
                      "pearson": round(float(pearsonr(x0, x1)[0]), 4), "mean_change": round(float((x1 - x0).mean()), 3),
                      "within_1": int((abs(x1 - x0) <= 1).sum()), "same": int((x1 == x0).sum()),
                      "ceiling_sqrt_r": round(float(np.sqrt(max(rs, 0))), 4)}

    sets = [p for p in P["sets"] if p["id"] in first]
    out["sets_answered"] = len(sets)
    res = {}
    for mname in ("s1", "s0"):
        right, conc = [], []
        for p in sets:
            a_ = first[p["id"]]
            s, b, w = p[mname], a_["best"], a_["worst"]
            right.append(s[b] > s[w])
            mids = [k for k in range(K) if k not in (b, w)]
            pairs = [(b, k) for k in range(K) if k != b] + [(k, w) for k in mids]
            conc.append(np.mean([s[hi] > s[lo] for hi, lo in pairs]))
        right = np.array(right)
        res[mname] = right
        out[mname] = {"right": int(right.sum()), "n": len(right), "share": round(float(right.mean()), 4),
                      "p_one_sided": float(binomtest(int(right.sum()), len(right), 0.5, alternative="greater").pvalue),
                      "five_orderings_mean": round(float(np.mean(conc)), 4)}
    dis = res["s1"] != res["s0"]
    k0 = int(res["s0"][dis].sum())
    out["disagree"] = {"n": int(dis.sum()), "rho0_right": k0,
                       "p_one_sided": float(binomtest(k0, int(dis.sum()), 0.5, alternative="greater").pvalue) if dis.any() else None}
    out["rule"] = {"P1": bool(out["s1"]["p_one_sided"] < 0.05),
                   "P2": bool(out["disagree"]["p_one_sided"] is not None and out["disagree"]["p_one_sided"] < 0.05
                              and out["s0"]["p_one_sided"] < 0.05)}
    pp = np.array([p["person_pred"] for p in sets])
    yy = np.array([first[p["id"]]["rating"] for p in sets], float)
    out["person_level"] = {"n": len(sets), "spearman": round(float(spearmanr(pp, yy)[0]), 4),
                           "ci95": boot(pp, yy, lambda a_, b_: spearmanr(a_, b_)[0]),
                           "mean_rating": round(float(yy.mean()), 3), "rated_7plus": int((yy >= 7).sum())}
    again = [p for p in sets if p["id"] in second]
    out["sets_shown_again"] = {"n": len(again),
                               "same_most_liked": int(sum(first[p["id"]]["best"] == second[p["id"]]["best"] for p in again)),
                               "same_least_liked": int(sum(first[p["id"]]["worst"] == second[p["id"]]["worst"] for p in again)),
                               "rating_within_1": int(sum(abs(first[p["id"]]["rating"] - second[p["id"]]["rating"]) <= 1 for p in again))}
    json.dump(out, open(os.path.join(R, "results/photosets_eval.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pick", "eval"])
    ap.add_argument("root")
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()
    if a.mode == "pick":
        a.new1, a.new2 = a.rest
        pick(a)
    else:
        a.ratings = a.rest[0]
        evaluate(a)
