#!/usr/bin/env python3
"""Photo pairs: does the model pick the photo of a person that the rater prefers? (PROTOCOL.md, "Photo pairs").

  photo_pairs.py pick ROOT            60 pairs + frozen predictions -> results/photo_pairs_set.json (private:
                                      accounts and codes), results/photo_pairs_pred.json (opaque ids; SHA-256),
                                      pairs_site/thumbs/<pair id>/{0,1}.jpg
  photo_pairs.py eval RATINGS ROOT    -> results/photo_pairs_eval.json
Ratings: 1 = left photo, 2 = right photo (rate_server.py), skips ignored.
"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
SEED = 20261115
ALPHA = 10000.0


def pick(root):
    from improve_r2 import kfit
    from group_transfer import scut_crowd_models
    from sface_input import sface_means
    from train_eval import folds, load, photo_index
    F, train, test, first, R, group, split, same = load(root); idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True))
    O = dict(np.load(os.path.join(root, "features_faceown.npz"), allow_pickle=True))
    y = np.array([first[u] for u in train], float); SFt = sface_means(root, train)
    rows = np.concatenate([idx[u] for u in train]); owner = np.concatenate([[i] * len(idx[u]) for i, u in enumerate(train)])
    codes = np.array([str(c) for c in F["code"]])[rows]
    P = np.c_[F["clip_nocloth"][rows], O["clip_face"][rows], M["siglip_nocloth"][rows], O["siglip_face"][rows], SFt[owner]].astype(np.float64)
    score = np.zeros(len(rows))
    for seed in (0, 1, 2):
        fo = folds(train, group, seed=seed)
        for k in range(5):
            tr, te = np.where(fo != k)[0], np.where(fo == k)[0]; trp, tep = np.isin(owner, tr), np.isin(owner, te)
            score[tep] += kfit(P[trp], y[owner[trp]])(P[tep], ALPHA) / 3
    # crowd per photo (raw-photo face crops, as the adopted crowd model reads them)
    crowd_fn = scut_crowd_models(root); crowd = np.full(len(rows), np.nan)
    clipf = F["clip_face"][rows].astype(np.float64); clipf /= np.linalg.norm(clipf, axis=1, keepdims=True)
    have = np.zeros(len(rows), bool); sf = np.zeros((len(rows), 128))
    for i, u in enumerate(train):
        d = np.load(os.path.join(root, "cut", u, "sface.npz")); pos = {str(c): j for j, c in enumerate(d["codes"])}
        for j in np.where(owner == i)[0]:
            if codes[j] in pos:
                sf[j] = d["emb"][pos[codes[j]]]; have[j] = True
    sf[have] /= np.linalg.norm(sf[have], axis=1, keepdims=True)
    crowd[have] = crowd_fn(clipf[have], sf[have])  # one call: standardized over all photos
    rng = np.random.default_rng(SEED)
    eligible = [i for i in range(len(train)) if (owner == i).sum() >= 4]
    people = rng.choice(eligible, 60, replace=False)
    out_dir = os.path.join(root, "pairs_site", "thumbs"); os.makedirs(out_dir, exist_ok=True)
    pairs = []
    for n, i in enumerate(people):
        js = np.where(owner == i)[0]
        if n < 40:
            a, b = js[np.argmax(score[js])], js[np.argmin(score[js])]; kind = "top vs bottom"
        else:
            a, b = rng.choice(js, 2, replace=False); kind = "random"
        if rng.random() < 0.5:
            a, b = b, a  # a = left, b = right
        u = train[i]; pid = "q" + hashlib.sha1(("%s|%s|%s" % (u, codes[a], codes[b])).encode()).hexdigest()[:7]
        os.makedirs(os.path.join(out_dir, pid), exist_ok=True)
        for k, j in enumerate((a, b)):
            shutil.copy(os.path.join(root, "cut", u, codes[j] + "_thumb.jpg"), os.path.join(out_dir, pid, "%d.jpg" % k))
        cp = None if np.isnan(crowd[a]) or np.isnan(crowd[b]) else (1 if crowd[a] > crowd[b] else 2)
        pairs.append({"id": pid, "kind": kind, "user": u, "left": codes[a], "right": codes[b],
                      "model_pick": 1 if score[a] > score[b] else 2, "model_diff": round(float(abs(score[a] - score[b])), 4), "crowd_pick": cp})
    json.dump(pairs, open(os.path.join(root, "results/photo_pairs_set.json"), "w"), indent=1)
    pred = [{k: p[k] for k in ("id", "kind", "model_pick", "model_diff", "crowd_pick")} for p in pairs]
    path = os.path.join(root, "results/photo_pairs_pred.json"); json.dump(pred, open(path, "w"), indent=1)
    print("60 pairs (40 top vs bottom, 20 random); frozen predictions SHA-256 %s" % hashlib.sha256(open(path, "rb").read()).hexdigest())


def evaluate(ratings, root):
    from scipy.stats import binomtest, spearmanr
    pred = {p["id"]: p for p in json.load(open(os.path.join(root, "results/photo_pairs_pred.json")))}
    eff = {}
    for line in open(ratings):
        r = json.loads(line); key = (r["id"], r.get("pass", 1))
        if r.get("undo"):
            eff.pop(key, None)
        else:
            eff[key] = r
    first = {k[0]: v["rating"] for k, v in eff.items() if k[1] == 1 and not v.get("skip") and v.get("rating") in (1, 2)}
    rep = {k[0]: v["rating"] for k, v in eff.items() if k[1] == 2 and not v.get("skip") and v.get("rating") in (1, 2)}
    out = {"answered": len(first), "skipped": sum(1 for k, v in eff.items() if k[1] == 1 and v.get("skip"))}
    same = [first[i] == (3 - rep[i]) for i in rep if i in first]  # repeats show the sides swapped
    if same:
        out["repeat_agreement"] = [int(sum(same)), len(same)]; print("repeats (sides swapped): same choice %d of %d" % (sum(same), len(same)))
    for who in ("model_pick", "crowd_pick"):
        out[who] = {}
        for kind in ("all", "top vs bottom", "random"):
            ids = [i for i in first if pred[i][who] is not None and (kind == "all" or pred[i]["kind"] == kind)]
            k = sum(first[i] == pred[i][who] for i in ids); n = len(ids)
            p = binomtest(k, n, 0.5, alternative="greater").pvalue if n else float("nan")
            out[who][kind] = {"agree": k, "n": n, "share": k / n if n else None, "p_one_sided": p}
            print("%-10s %-14s agrees on %2d of %2d (%.0f%%), one-sided p = %.4f" % (who.split("_")[0], kind, k, n, 100 * k / max(n, 1), p))
    ids = [i for i in first if pred[i]["kind"] == "top vs bottom"]
    if ids:
        hit = np.array([first[i] == pred[i]["model_pick"] for i in ids], float); d = np.array([pred[i]["model_diff"] for i in ids])
        out["agree_vs_model_gap"] = float(spearmanr(d, hit)[0]); print("larger model gap -> more agreement? Spearman %.3f" % out["agree_vs_model_gap"])
    ok = out["model_pick"]["all"]["p_one_sided"] < 0.05 and out["model_pick"]["all"]["share"] > 0.5
    out["verdict"] = "the photo-level signal predicts image-level preference" if ok else "not shown"
    print("verdict:", out["verdict"])
    json.dump(out, open(os.path.join(root, "results/photo_pairs_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "pick":
        pick(sys.argv[2] if len(sys.argv) > 2 else ".")
    else:
        evaluate(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ".")
