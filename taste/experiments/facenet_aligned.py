#!/usr/bin/env python3
"""FaceNet on eye-aligned faces as a third crowd code (PROTOCOL.md, afternoon C). GPU for FaceNet."""
import glob
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
from taste_onboard import crowd_score  # noqa: E402
from train_eval import load, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
A = np.logspace(-1, 5, 25)
OUT = 160


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def aligned(img, e1, e2):
    le, re = (np.array(e1, float), np.array(e2, float)) if e1[0] <= e2[0] else (np.array(e2, float), np.array(e1, float))
    d = re - le; iod = np.linalg.norm(d)
    if iod < 4:
        return None
    ex = d / iod; ey = np.array([-ex[1], ex[0]]); c = (le + re) / 2 + 0.35 * iod * ey; side = 3.0 * iod
    tl = c - side / 2 * ex - side / 2 * ey; k = side / OUT
    return img.transform((OUT, OUT), Image.AFFINE, (k * ex[0], k * ey[0], tl[0], k * ex[1], k * ey[1], tl[1]), resample=Image.BILINEAR)


def read_pts(path):
    b = open(path, "rb").read(); n = int(np.frombuffer(b[:4], np.int32)[0]); return np.frombuffer(b[4:4 + 8 * n], np.float32).reshape(n, 2)


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True)); names = [str(n) for n in S["names"]]
    path = os.path.join(root, "datasets/facenet_aligned.npz")
    if not os.path.exists(path):
        from idscore.facenet import FaceNetEmbedder
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = FaceNetEmbedder(os.path.join(root, "models", "facenet", "20180402-114759-vggface2.pt")).net.to(dev)
        def emb(ims):
            x = torch.stack([torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1) for im in ims])
            with torch.no_grad():
                e = net(((x - 127.5) / 128.0).to(dev)).cpu().numpy()
            return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        base = os.path.join(root, "datasets/scut/SCUT-FBP5500_v2")
        scut, batch = [], []
        for n in names:
            p = read_pts(os.path.join(base, "facial landmark", n.rsplit(".", 1)[0] + ".pts"))
            e1, e2 = (p[58], p[59]) if len(p) > 59 else ((133.0, 159.0), (224.0, 158.0))  # one file is empty: SCUT's mean pupils
            batch.append(aligned(Image.open(os.path.join(base, "Images", n)).convert("RGB"), e1, e2))
            if len(batch) == 64:
                scut.append(emb(batch)); batch = []
        if batch:
            scut.append(emb(batch))
        scut = np.concatenate(scut); print("SCUT aligned and embedded:", scut.shape, flush=True)
        pool, used = [], []
        for u in allu:
            rec = json.load(open(os.path.join(root, "cut", u, "owner.json"))); ims = []
            for ph in rec["photos"]:
                kp = ph.get("keypoints")
                if not kp or len(kp) < 3:
                    continue
                f = (glob.glob(os.path.join(root, "photos", u, ph["code"] + ".*")) or [None])[0]
                if f is None:
                    continue
                im = aligned(Image.open(f).convert("RGB"), kp[1][:2], kp[2][:2])
                if im is not None:
                    ims.append(im)
            used.append(len(ims))
            if ims:
                e = emb(ims).mean(0); pool.append(e / (np.linalg.norm(e) + 1e-12))
            else:
                pool.append(np.full(512, np.nan))
        pool = np.stack(pool); print("pool: aligned faces per person, median %d; people with none: %d" % (np.median(used), int(np.sum(np.array(used) == 0))), flush=True)
        np.savez_compressed(path, scut=scut, pool=pool)
    d = np.load(path); fs, fp = d["scut"], d["pool"]
    fp = np.where(np.isnan(fp), np.nanmean(fp, 0), fp)
    sc = StandardScaler().fit(fs); zf = zs(RidgeCV(alphas=A).fit(sc.transform(fs), S["mean"]).predict(sc.transform(fp)))
    z2 = crowd_score(root, F, idx, allu); z3 = zs(z2 * np.sqrt(2) + zf)
    tr = np.array([pos[u] for u in train]); te = np.array([pos[u] for u in test])
    ytr = np.array([first[u] for u in train], float); yte = np.array([first[u] for u in test], float)
    out = {"zero": {}}
    for nm, z in (("CLIP + SFace (adopted)", z2), ("FaceNet aligned alone", zf), ("CLIP + SFace + FaceNet", z3)):
        out["zero"][nm] = {"train": float(spearmanr(z[tr], ytr).correlation), "test": float(spearmanr(z[te], yte).correlation)}
        print("%-24s zero-rating: training %.3f, test %.3f" % (nm, out["zero"][nm]["train"], out["zero"][nm]["test"]), flush=True)
    XL = StandardScaler().fit_transform(winner_features(F, idx, allu)); Z = PCA(n_components=32, random_state=0).fit_transform(XL)[tr]; X = XL[tr]
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid)); res = {m: {n: [] for n in NS} for m in ("adopted", "with FaceNet")}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Z[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]; w = n / (n + 20); pr = rankdata(RidgeCV(alphas=A).fit(X[pk], ytr[pk]).predict(X[Ei])) / len(Ei)
            for m, zz in (("adopted", z2[tr]), ("with FaceNet", z3[tr])):
                res[m][n].append(spearmanr((1 - w) * rankdata(zz[Ei]) / len(Ei) + w * pr, ytr[Ei]).correlation)
    out["few"] = {m: {n: float(np.mean(v)) for n, v in dd.items()} for m, dd in res.items()}
    gtr = out["zero"]["CLIP + SFace + FaceNet"]["train"] - out["zero"]["CLIP + SFace (adopted)"]["train"]
    gte = out["zero"]["CLIP + SFace + FaceNet"]["test"] - out["zero"]["CLIP + SFace (adopted)"]["test"]
    wins = int(np.sum(np.array(res["with FaceNet"][20]) > np.array(res["adopted"][20])))
    out["verdict"] = {"zero_gain_train": gtr, "zero_gain_test": gte, "wins_n20": wins, "adopt": bool(gtr >= 0.03 and wins >= 15 and gte > 0)}
    for m in res:
        print("%-14s %s" % (m, "  ".join("n=%d %.3f" % (n, out["few"][m][n]) for n in NS)))
    print("FaceNet aligned: zero-rating %+.3f (test %+.3f), wins %d/20 -> %s" % (gtr, gte, wins, "adopt" if out["verdict"]["adopt"] else "no"))
    json.dump(out, open(os.path.join(root, "results/facenet_aligned.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
