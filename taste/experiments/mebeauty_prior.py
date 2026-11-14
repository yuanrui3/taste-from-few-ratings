#!/usr/bin/env python3
"""Second crowd prior from MEBeauty (see PROTOCOL.md "Second crowd: MEBeauty"). GPU for embedding."""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from probe_search import herding, winner_features  # noqa: E402
from train_eval import load, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]


def main(root):
    base = os.path.join(root, "datasets/mebeauty/MEBeauty-database-main")
    items = {}
    for f in ("train_2022.txt", "val_2022.txt", "test_2022.txt"):
        for line in open(os.path.join(base, "scores", f)):
            parts = line.strip().rsplit(" ", 1)
            if len(parts) == 2:
                p = os.path.normpath(os.path.join(base, parts[0]))
                if os.path.exists(p):
                    items[p] = float(parts[1])
    paths = sorted(items); scores = np.array([items[p] for p in paths])
    print("MEBeauty faces with scores:", len(paths), flush=True)
    import open_clip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, _, tf = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k"); m = m.eval().to(dev).half()
    embs = []
    for i in range(0, len(paths), 64):
        with torch.no_grad():
            e = m.encode_image(torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 64]]).to(dev).half()).float()
        embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    me = np.concatenate(embs)

    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    follows = {l.split("\t")[0] for l in open(os.path.join(root, "links/following.tsv")) if l.strip() and not l.startswith(("#", "username"))}
    face = np.stack([F["clip_face"][idx[u]].mean(0) for u in train]); face /= np.linalg.norm(face, axis=1, keepdims=True)
    y = np.array([first[u] for u in train], float); st = np.array([u not in follows for u in train])
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    def prior(E, t):
        sc = StandardScaler().fit(E); r = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(sc.transform(E), t)
        z = r.predict(sc.transform(face)); return (z - z.mean()) / z.std()
    zs = {"SCUT": prior(S["emb"], S["mean"]), "MEBeauty": prior(me, scores)}
    zs["SCUT + MEBeauty"] = (zs["SCUT"] + zs["MEBeauty"]) / 2
    out = {"zero_rating": {k: {"all": float(spearmanr(z, y).correlation), "strangers": float(spearmanr(z[st], y[st]).correlation)} for k, z in zs.items()}}
    for k, v in out["zero_rating"].items():
        print("%-18s zero-rating Spearman all %.3f, strangers %.3f" % (k, v["all"], v["strangers"]), flush=True)
    best = max(out["zero_rating"], key=lambda k: out["zero_rating"][k]["all"]); out["chosen"] = best
    users = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(users)}
    Xall = StandardScaler().fit_transform(winner_features(F, idx, users)); tr = np.array([pos[u] for u in train]); X = Xall[tr]
    Z32 = PCA(n_components=32, random_state=0).fit_transform(Xall)
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid))
    res = {k: {n: [] for n in NS} for k in ("SCUT", best)}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); P, E = np.where(isP)[0], np.where(~isP)[0]
        order = herding(Z32[tr[P]], max(NS))
        for n in NS:
            pk = order[:n]
            pr = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(X[P][pk], y[P][pk]).predict(X[E]); ro = rankdata(pr) / len(pr)
            w = n / (n + 20)
            for k in res:
                res[k][n].append(spearmanr((1 - w) * rankdata(zs[k][E]) / len(E) + w * ro, y[E]).correlation)
    out["few_blend_k20"] = {k: {n: float(np.nanmean(v)) for n, v in d.items()} for k, d in res.items()}
    for k, d in out["few_blend_k20"].items():
        print("  blend k=20 with %-18s %s" % (k, "  ".join("%.3f" % d[n] for n in NS)))
    json.dump(out, open(os.path.join(root, "results/mebeauty_prior.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(root, "datasets/mebeauty/me.npz"), paths=np.array(paths), emb=me, mean=scores)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
