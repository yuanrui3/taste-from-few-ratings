#!/usr/bin/env python3
"""The latent space itself: one point per person (OpenCLIP cut-out without clothing + face,
averaged over the owner's photos), mapped to 2D and split into types.

- results/latent_map.png: t-SNE map of all 412 people, colour = the user's rating (gray = not
  rated), ring = followed; arrow = the taste direction (ridge weights of the main model).
- results/latent_types.jpg + explain_types.json: k-means types in the latent space (no ratings
  used to form them), each with the mean rating, size, share followed, and 4 people nearest the
  type's centre.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from probe_search import winner_features  # noqa: E402
from train_eval import load, photo_index  # noqa: E402

K_TYPES = 8


def colour(r):  # rating 2..10 -> blue..red
    t = min(1.0, max(0.0, (r - 3) / 6))
    return (int(40 + 200 * t), int(90 + 40 * (1 - abs(t - 0.5) * 2)), int(220 - 190 * t))


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    follows = {l.split("\t")[0] for l in open(os.path.join(root, "links/following.tsv"))
               if l.strip() and not l.startswith(("#", "username"))}
    users = sorted(set(split["train"]) | set(split["test"]))
    X = StandardScaler().fit_transform(winner_features(F, idx, users))
    P = PCA(n_components=32, random_state=0).fit_transform(X)
    xy = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(P)
    km = KMeans(n_clusters=K_TYPES, n_init=20, random_state=0).fit(P)
    lab = km.labels_
    # taste direction from the main model (training accounts only), drawn in the 2D map
    tr = [i for i, u in enumerate(users) if u in set(train)]
    m = Ridge(alpha=3000).fit(X[tr], [first[users[i]] for i in tr])
    score = X @ m.coef_
    g = np.c_[xy, np.ones(len(xy))]
    a, *_ = np.linalg.lstsq(g, score, rcond=None)  # best linear fit of the taste score on the map
    W, H, pad = 1100, 900, 60
    lo, hi = xy.min(0), xy.max(0)
    to = lambda p: (pad + (p[0] - lo[0]) / (hi[0] - lo[0]) * (W - 2 * pad), pad + (p[1] - lo[1]) / (hi[1] - lo[1]) * (H - 2 * pad))
    im = Image.new("RGB", (W, H), (248, 248, 245)); d = ImageDraw.Draw(im)
    for i, u in enumerate(users):
        x, y_ = to(xy[i]); r = first.get(u)
        c = colour(r) if r is not None else (190, 190, 190)
        d.ellipse([x - 6, y_ - 6, x + 6, y_ + 6], fill=c, outline=(30, 30, 30) if u in follows else None, width=2)
    for k in range(K_TYPES):  # type labels at their centres on the map
        cx, cy = to(xy[lab == k].mean(0)); d.text((cx + 8, cy - 8), "T%d" % (k + 1), fill=(0, 0, 0))
    v = a[:2] / (np.linalg.norm(a[:2]) + 1e-9); c0 = np.array([W / 2, H / 2])
    d.line([tuple(c0 - v * 180), tuple(c0 + v * 180)], fill=(0, 0, 0), width=3)
    tip = c0 + v * 180; d.ellipse([tip[0] - 7, tip[1] - 7, tip[0] + 7, tip[1] + 7], fill=(0, 0, 0))
    d.text((12, 12), "Latent map of 412 people (t-SNE of OpenCLIP cut-out + face). Colour: your rating, blue low -> red high; gray = not rated.", fill=(0, 0, 0))
    d.text((12, 28), "Dark ring: you follow them. Line with dot: direction in which the model's predicted rating rises. T1..T8: latent types.", fill=(0, 0, 0))
    im.save(os.path.join(root, "results/latent_map.png"))

    types = []
    cell, gap = 140, 3
    sheet = Image.new("RGB", (4 * (cell + gap) + 240, K_TYPES * (cell + gap) + gap), (245, 245, 242))
    ds = ImageDraw.Draw(sheet)
    for k in range(K_TYPES):
        members = [i for i in range(len(users)) if lab[i] == k]
        rs = [first[users[i]] for i in members if users[i] in first]
        near = sorted(members, key=lambda i: np.linalg.norm(P[i] - km.cluster_centers_[k]))[:4]
        types.append({"type": "T%d" % (k + 1), "n": len(members), "mean_rating": float(np.mean(rs)) if rs else None,
                      "share_followed": float(np.mean([users[i] in follows for i in members])),
                      "model_score_mean": float(score[members].mean())})
        y0 = k * (cell + gap) + gap
        ds.text((8, y0 + 10), "T%d  n=%d" % (k + 1, len(members)), fill=(0, 0, 0))
        ds.text((8, y0 + 30), "mean rating %.1f" % types[-1]["mean_rating"], fill=(0, 0, 0))
        ds.text((8, y0 + 50), "followed %d%%" % round(100 * types[-1]["share_followed"]), fill=(0, 0, 0))
        for j, i in enumerate(near):
            rec = json.load(open(os.path.join(root, "cut", users[i], "owner.json")))
            t = Image.open(os.path.join(root, "cut", users[i], rec["photos"][0]["code"] + "_thumb.jpg")).convert("RGB")
            t.thumbnail((cell, cell)); sheet.paste(t, (240 + j * (cell + gap) + (cell - t.width) // 2, y0 + (cell - t.height) // 2))
    sheet.save(os.path.join(root, "results/latent_types.jpg"), quality=85)
    json.dump(types, open(os.path.join(root, "results/explain_types.json"), "w"), indent=1)
    for t in sorted(types, key=lambda t: -(t["mean_rating"] or 0)):
        print("%s  n=%3d  mean rating %.2f  followed %3d%%" % (t["type"], t["n"], t["mean_rating"], round(100 * t["share_followed"])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
