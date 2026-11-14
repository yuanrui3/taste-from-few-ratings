#!/usr/bin/env python3
"""Mirror averaging of the L-14 codes (PROTOCOL.md). GPU for the embedding, the rest CPU."""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from probe_search import herding  # noqa: E402
from train_eval import best_alpha, load, mean_vec, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
A = np.logspace(-1, 5, 25)


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    path = os.path.join(root, "features_flip.npz")
    if not os.path.exists(path):
        import open_clip
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m, _, tf = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k"); m = m.eval().to(dev).half()
        out = {}
        for v in ("nocloth", "face"):
            embs = []
            for i in range(0, len(F["user"]), 48):
                ims = [tf(ImageOps.mirror(Image.open(os.path.join(root, "cut", str(u), "%s_%s.jpg" % (c, v))).convert("RGB")))
                       for u, c in zip(F["user"][i:i + 48], F["code"][i:i + 48])]
                with torch.no_grad():
                    e = m.encode_image(torch.stack(ims).to(dev).half()).float()
                embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
            out["clip_%s" % v] = np.concatenate(embs); print("mirrored", v, flush=True)
        np.savez_compressed(path, **out)
    Fl = dict(np.load(path))
    G = {}
    for v in ("clip_nocloth", "clip_face"):
        a = F[v] + Fl[v]; G[v] = a / np.linalg.norm(a, axis=1, keepdims=True)
    print("photo agreement original vs mirrored (mean cosine): nocloth %.3f, face %.3f" % (
        np.mean((F["clip_nocloth"] * Fl["clip_nocloth"]).sum(1)), np.mean((F["clip_face"] * Fl["clip_face"]).sum(1))), flush=True)
    feats = lambda src, users: np.c_[mean_vec(src, idx, users, "clip_nocloth"), mean_vec(src, idx, users, "clip_face")]
    y = np.array([first[u] for u in train], float)
    _, s0 = best_alpha(feats(F, train), y, train, group); _, s1 = best_alpha(feats(G, train), y, train, group)
    print("full CV: original %.3f, mirror-averaged %.3f" % (s0, s1), flush=True)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}; tr = np.array([pos[u] for u in train])
    Xo = StandardScaler().fit_transform(feats(F, allu))[tr]; Xm = StandardScaler().fit_transform(feats(G, allu))[tr]
    Z = PCA(n_components=32, random_state=0).fit_transform(StandardScaler().fit_transform(feats(F, allu)))[tr]
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    scs = StandardScaler().fit(S["emb"]); crowd = RidgeCV(alphas=A).fit(scs.transform(S["emb"]), S["mean"])
    zc = crowd.predict(scs.transform(mean_vec(F, idx, allu, "clip_face")))[tr]
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid)); res = {m: {n: [] for n in NS} for m in ("original", "mirror-averaged")}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Z[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]; w = n / (n + 20); pc = rankdata(zc[Ei]) / len(Ei)
            for m, X in (("original", Xo), ("mirror-averaged", Xm)):
                pr = RidgeCV(alphas=A).fit(X[pk], y[pk]).predict(X[Ei])
                res[m][n].append(spearmanr((1 - w) * pc + w * rankdata(pr) / len(pr), y[Ei]).correlation)
    out = {"full": {"original": s0, "mirror-averaged": s1}, "few": {m: {n: float(np.mean(v)) for n, v in d.items()} for m, d in res.items()}}
    for m in res:
        print("%-16s %s" % (m, "  ".join("n=%d %.3f" % (n, out["few"][m][n]) for n in NS)))
    gain = float(np.mean([out["few"]["mirror-averaged"][n] - out["few"]["original"][n] for n in NS[:4]]))
    wins = int(np.sum(np.array(res["mirror-averaged"][20]) > np.array(res["original"][20])))
    out["verdict"] = {"full_gain": s1 - s0, "few_gain_5_40": gain, "few_wins_n20": wins,
                      "adopt": bool(s1 - s0 >= 0.02 or (gain >= 0.02 and wins >= 15))}
    print("mirror: full %+.3f, few %+.3f, wins %d/20 -> %s" % (s1 - s0, gain, wins, "adopt" if out["verdict"]["adopt"] else "keep"))
    json.dump(out, open(os.path.join(root, "results/flip_tta.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
