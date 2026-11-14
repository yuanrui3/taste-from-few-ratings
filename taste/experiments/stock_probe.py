#!/usr/bin/env python3
"""Stock faces as the probe set (see PROTOCOL.md "Stock faces as the probe set"). CPU only.

  stock_probe.py pick [ROOT]                 -> results/stock_probe_set.json + stock_faces/<id>.jpg
  stock_probe.py eval STOCK_RATINGS.jsonl [ROOT] -> results/stock_probe.json
"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from train_eval import load, photo_index  # noqa: E402

ALPHAS = np.logspace(-1, 5, 25)
N_WOMEN, N_MEN = 30, 10


def herding(Z, n):
    sq = (Z ** 2).sum(1); D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0)
    s2 = np.median(D2[D2 > 0]); K = np.exp(-D2 / s2); mu = K.mean(1)
    acc = np.zeros(len(Z)); order = []
    for t in range(n):
        score = mu - acc / (t + 1); score[order] = -np.inf
        i = int(np.argmax(score)); order.append(i); acc += K[:, i]
    return order


def scut(root):
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    S["names"] = np.array([str(n) for n in S["names"]]); return S


def pick(root):
    S = scut(root); names = S["names"]
    Z = PCA(n_components=32, random_state=0).fit_transform(StandardScaler().fit_transform(S["emb"]))
    women = np.where(np.char.startswith(names, "AF") | np.char.startswith(names, "CF"))[0]
    men = np.where(np.char.startswith(names, "AM") | np.char.startswith(names, "CM"))[0]
    chosen = [int(women[i]) for i in herding(Z[women], N_WOMEN)] + [int(men[i]) for i in herding(Z[men], N_MEN)]
    out_dir = os.path.join(root, "stock_faces"); os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(root, "datasets/scut/SCUT-FBP5500_v2/Images")
    order = sorted(chosen, key=lambda j: hashlib.sha1(names[j].encode()).hexdigest())  # mix women and men
    items = []
    for j in order:
        sid = "s" + hashlib.sha1(names[j].encode()).hexdigest()[:7]
        shutil.copy(os.path.join(base, names[j]), os.path.join(out_dir, sid + ".jpg"))
        items.append({"id": sid, "name": names[j], "crowd_mean": float(S["mean"][j])})
    json.dump(items, open(os.path.join(root, "results/stock_probe_set.json"), "w"), indent=1)
    print("picked %d stock faces (%d women, %d men)" % (len(items), N_WOMEN, N_MEN))


def boot_gain(p, base, y, n=2000, seed=0):
    g = np.random.default_rng(seed); idx = np.arange(len(y)); v = []
    for _ in range(n):
        s = g.choice(idx, len(idx))
        v.append(spearmanr(p[s], y[s]).correlation - spearmanr(base[s], y[s]).correlation)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def evaluate(ratings_path, root):
    S = scut(root); pos = {n: i for i, n in enumerate(S["names"])}
    items = {it["id"]: it for it in json.load(open(os.path.join(root, "results/stock_probe_set.json")))}
    last = {}
    for line in open(ratings_path):
        r = json.loads(line)
        if r.get("undo"):
            last.pop(r["id"], None)
        elif r["id"] in items and r.get("rating") is not None and not r.get("skip"):
            last[r["id"]] = float(r["rating"])
    rows = [pos[items[i]["name"]] for i in last]; ys = np.array([last[i] for i in last])
    print("stock ratings used:", len(ys))
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F); users = train + test
    face = np.stack([F["clip_face"][idx[u]].mean(0) for u in users]); face /= np.linalg.norm(face, axis=1, keepdims=True)
    y = np.array([first[u] for u in users], float)
    sc = StandardScaler().fit(S["emb"]); E = sc.transform(S["emb"]); Fz = sc.transform(face)
    crowd = RidgeCV(alphas=ALPHAS).fit(E, S["mean"])
    za = crowd.predict(Fz); za = (za - za.mean()) / za.std()
    rb = RidgeCV(alphas=ALPHAS).fit(E[rows], ys); zb = rb.predict(Fz); zb = (zb - zb.mean()) / zb.std()
    zc = (rankdata(za) + rankdata(zb)) / 2
    yc = (ys - ys.mean()) / ys.std(); cm = (S["mean"][rows] - S["mean"][rows].mean()) / S["mean"][rows].std()
    rd = RidgeCV(alphas=ALPHAS).fit(E[rows], yc - cm); dd = rd.predict(Fz)
    zd = za + dd  # both in crowd-score standard units
    preds = {"a crowd prior": za, "b stock ratings": zb, "c rank average": zc, "d crowd + own deviation": zd}
    out = {"n_stock": int(len(ys)), "stock_vs_crowd_on_stock": float(spearmanr(ys, S["mean"][rows]).correlation), "methods": {}}
    for k, p in preds.items():
        out["methods"][k] = {"spearman": float(spearmanr(p, y).correlation),
                             "gain_ci95": boot_gain(p, za, y) if k[0] != "a" else None}
        print("%-26s %.3f %s" % (k, out["methods"][k]["spearman"], out["methods"][k]["gain_ci95"] or ""))
    best = max([k for k in preds if k[0] != "a"], key=lambda k: out["methods"][k]["spearman"])
    gain = out["methods"][best]["spearman"] - out["methods"]["a crowd prior"]["spearman"]
    out["best"] = best; out["helps"] = bool(gain >= 0.05 and out["methods"][best]["gain_ci95"][0] > 0)
    print("user's stock ratings vs crowd mean on the same faces: %.3f" % out["stock_vs_crowd_on_stock"])
    print("best %s, gain %+.3f -> %s" % (best, gain, "stock faces help" if out["helps"] else "no clear help"))
    json.dump(out, open(os.path.join(root, "results/stock_probe.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "pick":
        pick(sys.argv[2] if len(sys.argv) > 2 else ".")
    else:
        evaluate(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ".")
