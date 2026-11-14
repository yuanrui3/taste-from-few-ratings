#!/usr/bin/env python3
"""Pushing the onboarding below 40 ratings (PROTOCOL.md). GPU for the FaceNet embedding."""
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
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]  # experiments/, pipeline, code/ (idscore)
from probe_search import herding, winner_features  # noqa: E402
from sface_input import sface_means  # noqa: E402
from taste_onboard import crowd_score  # noqa: E402
from train_eval import load, mean_vec, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
A = np.logspace(-1, 5, 25)
TARGET = 0.547


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def n_star(curve):
    """Ratings needed to reach TARGET, linear in log n between measured points."""
    xs = [n for n in NS]; ys = [curve[n] for n in NS]
    if ys[0] >= TARGET:
        return float(NS[0])
    for (n0, y0), (n1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y1 >= TARGET > y0:
            t = (TARGET - y0) / (y1 - y0); return float(np.exp(np.log(n0) + t * (np.log(n1) - np.log(n0))))
    return None


def facenet_codes(root, F, idx, allu, S):
    path = os.path.join(root, "datasets/facenet_codes.npz")
    if os.path.exists(path):
        d = np.load(path); return d["scut"], d["pool"]
    from idscore.facenet import FaceNetEmbedder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emb = FaceNetEmbedder(os.path.join(root, "models", "facenet", "20180402-114759-vggface2.pt")); emb.net = emb.net.to(dev)
    def run(paths):
        out = []
        for i in range(0, len(paths), 64):
            xs = []
            for p in paths[i:i + 64]:
                x = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)).permute(2, 0, 1)[None]
                xs.append(torch.nn.functional.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False))
            with torch.no_grad():
                e = emb.net(((torch.cat(xs) - 127.5) / 128.0).to(dev)).cpu().numpy()
            out.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12))
        return np.concatenate(out)
    base = os.path.join(root, "datasets/scut/SCUT-FBP5500_v2/Images")
    scut = run([os.path.join(base, str(n)) for n in S["names"]]); print("FaceNet: SCUT done", flush=True)
    rows = [(u, str(c)) for u in allu for c in F["code"][idx[u]]]
    ph = run([os.path.join(root, "cut", u, "%s_face.jpg" % c) for u, c in rows]); print("FaceNet: pool done", flush=True)
    pool, k = [], 0
    for u in allu:
        m = ph[k:k + len(idx[u])].mean(0); pool.append(m / (np.linalg.norm(m) + 1e-12)); k += len(idx[u])
    pool = np.stack(pool); np.savez_compressed(path, scut=scut, pool=pool); return scut, pool


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    tr = np.array([pos[u] for u in train]); te = np.array([pos[u] for u in test])
    ytr = np.array([first[u] for u in train], float); yte = np.array([first[u] for u in test], float)
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    z2 = crowd_score(root, F, idx, allu)  # adopted prior: CLIP + SFace
    fs, fp = facenet_codes(root, F, idx, allu, S)
    sc = StandardScaler().fit(fs); zf = zs(RidgeCV(alphas=A).fit(sc.transform(fs), S["mean"]).predict(sc.transform(fp)))
    z3 = zs(z2 * np.sqrt(2) + zf)  # mean of three standardized scores (z2 already holds two)
    out = {"zero": {}}
    for nm, z in (("CLIP + SFace (adopted)", z2), ("FaceNet alone", zf), ("CLIP + SFace + FaceNet", z3)):
        out["zero"][nm] = {"train": float(spearmanr(z[tr], ytr).correlation), "test": float(spearmanr(z[te], yte).correlation)}
        print("%-24s zero-rating: training %.3f, test %.3f" % (nm, out["zero"][nm]["train"], out["zero"][nm]["test"]), flush=True)
    XL = StandardScaler().fit_transform(winner_features(F, idx, allu)); Z32 = PCA(n_components=32, random_state=0).fit_transform(XL)
    XLS = StandardScaler().fit_transform(np.c_[winner_features(F, idx, allu), sface_means(root, allu)])
    X, XS, Z = XL[tr], XLS[tr], Z32[tr]
    variants = {"baseline (k=20)": (X, z2, 20), "T1 k=10": (X, z2, 10), "T1 k=40": (X, z2, 40), "T1 k=80": (X, z2, 80),
                "T2 L-14 + SFace input": (XS, z2, 20), "T3 three-code prior": (X, z3, 20)}
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid)); res = {m: {n: [] for n in NS} for m in variants}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Z[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]; cache = {}
            for m, (Xm, zp, k) in variants.items():
                key = id(Xm)
                if key not in cache:
                    pr = RidgeCV(alphas=A).fit(Xm[pk], ytr[pk]).predict(Xm[Ei]); cache[key] = rankdata(pr) / len(pr)
                w = n / (n + k); zz = zp[tr]
                res[m][n].append(spearmanr((1 - w) * rankdata(zz[Ei]) / len(Ei) + w * cache[key], ytr[Ei]).correlation)
    out["few"] = {m: {n: float(np.mean(v)) for n, v in d.items()} for m, d in res.items()}
    out["n_star"] = {m: n_star(c) for m, c in out["few"].items()}
    print("\n%-24s %s   n* (ratings to reach %.3f)" % ("variant", "  ".join("n=%d" % n for n in NS), TARGET))
    for m in variants:
        ns = out["n_star"][m]
        print("%-24s %s   %s" % (m, "  ".join("%.3f" % out["few"][m][n] for n in NS), "%.0f" % ns if ns else "> 80"))
    base = out["few"]["baseline (k=20)"]; out["verdict"] = {}
    for m in variants:
        if m.startswith("baseline"):
            continue
        gain = float(np.mean([out["few"][m][n] - base[n] for n in NS[:4]]))
        wins = int(np.sum(np.array(res[m][20]) > np.array(res["baseline (k=20)"][20])))
        if m.startswith("T3"):
            gtr = out["zero"]["CLIP + SFace + FaceNet"]["train"] - out["zero"]["CLIP + SFace (adopted)"]["train"]
            gte = out["zero"]["CLIP + SFace + FaceNet"]["test"] - out["zero"]["CLIP + SFace (adopted)"]["test"]
            ok = gtr >= 0.03 and wins >= 15 and gte > 0
            out["verdict"][m] = {"zero_gain_train": gtr, "zero_gain_test": gte, "wins_n20": wins, "gain_5_40": gain, "adopt": bool(ok)}
            print("%-24s zero-rating %+.3f (test %+.3f), wins %d/20 -> %s" % (m, gtr, gte, wins, "adopt" if ok else "no"))
        else:
            ok = gain >= 0.02 and wins >= 15
            out["verdict"][m] = {"gain_5_40": gain, "wins_n20": wins, "adopt": bool(ok)}
            print("%-24s gain %+.3f, wins %d/20 -> %s" % (m, gain, wins, "adopt" if ok else "no"))
    json.dump(out, open(os.path.join(root, "results/push_down.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
