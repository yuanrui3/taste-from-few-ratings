#!/usr/bin/env python3
"""Owner-only face crops (PROTOCOL.md, "Owner-only face crops").

The face crop of owner_cutouts.py is cut from the raw photo, so in group photos it shows neighbours' faces
(3.9% of kept photos, `cutout_faces.py`). Here the same face square is cut from the owner-only cut-out
(everyone else already grey), padded with grey to a square.

  face_own.py crop ROOT    -> cut/<person>/<photo>_faceown.jpg (CPU)
  face_own.py embed ROOT   -> features_faceown.npz {user, code, clip_face, siglip_face} (GPU; same models as
                              features.py and embed_more.py)
  face_own.py eval ROOT    full-model CV and the few-rating curve with the new face codes -> results/face_own.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
NS = [5, 10, 20, 40, 80]
GRAY = 128


def crop(root):
    import cv2
    n = 0
    for u in sorted(os.listdir(os.path.join(root, "cut"))):
        p = os.path.join(root, "cut", u, "owner.json")
        if not os.path.exists(p):
            continue
        for ph in json.load(open(p))["photos"]:
            cut = cv2.imread(os.path.join(root, "cut", u, ph["code"] + "_cut.jpg"))
            x0, y0, x1, y1 = ph["crop"]; f = min(1.0, 448 / max(x1 - x0, y1 - y0))
            fx, fy, fw, fh = ph["face_box"]; cx, cy = (fx + fw / 2 - x0) * f, (fy + fh / 2 - y0) * f
            s = int(round(1.1 * max(fw, fh) * f)); S = 2 * s
            canvas = np.full((S, S, 3), GRAY, np.uint8)
            a0, b0 = int(round(cx)) - s, int(round(cy)) - s
            H, W = cut.shape[:2]
            xs, ys = max(0, a0), max(0, b0); xe, ye = min(W, a0 + S), min(H, b0 + S)
            if xe > xs and ye > ys:
                canvas[ys - b0:ye - b0, xs - a0:xe - a0] = cut[ys:ye, xs:xe]
            out = cv2.resize(canvas, (224, 224), interpolation=cv2.INTER_AREA if S > 224 else cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(root, "cut", u, ph["code"] + "_faceown.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
            n += 1
    print("owner-only face crops written: %d" % n, flush=True)


def embed(root):
    import open_clip
    import torch
    from PIL import Image
    F = np.load(os.path.join(root, "features.npz"), allow_pickle=True)
    users, codes = F["user"], F["code"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"user": users, "code": codes}
    for key, arch, pre in (("clip_face", "ViT-L-14", "laion2b_s32b_b82k"), ("siglip_face", "ViT-SO400M-14-SigLIP-384", "webli")):
        m, _, tf = open_clip.create_model_and_transforms(arch, pretrained=pre); m = m.eval().to(dev).half(); embs = []
        for i in range(0, len(users), 32):
            ims = [tf(Image.open(os.path.join(root, "cut", str(u), "%s_faceown.jpg" % c)).convert("RGB")) for u, c in zip(users[i:i + 32], codes[i:i + 32])]
            with torch.no_grad():
                e = m.encode_image(torch.stack(ims).to(dev).half()).float()
            embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        out[key] = np.concatenate(embs); print(key, out[key].shape, flush=True)
        del m; torch.cuda.empty_cache()
    np.savez_compressed(os.path.join(root, "features_faceown.npz"), **out)
    print("saved", flush=True)


def evaluate(root):
    from scipy.stats import rankdata, spearmanr
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from improve_r2 import kfit
    from probe_search import herding, winner_features
    from sface_input import sface_means
    from taste_onboard import crowd_score
    from train_eval import ALPHAS, folds, load, photo_index
    A = np.logspace(-1, 5, 25)
    F, train, test, first, R, group, split, same = load(root); idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True))
    O = dict(np.load(os.path.join(root, "features_faceown.npz"), allow_pickle=True))
    assert (O["user"] == F["user"]).all() and (O["code"] == F["code"]).all()
    Fo = dict(F); Fo["clip_face"] = O["clip_face"]
    y = np.array([first[u] for u in train], float); SFt = sface_means(root, train)
    rows = np.concatenate([idx[u] for u in train]); owner = np.concatenate([[i] * len(idx[u]) for i, u in enumerate(train)])
    out = {}
    for name, cf, gf in (("current", F["clip_face"], M["siglip_face"]), ("owner-only faces", O["clip_face"], O["siglip_face"])):
        P = np.c_[F["clip_nocloth"][rows], cf[rows], M["siglip_nocloth"][rows], gf[rows], SFt[owner]].astype(np.float64)
        pa = [a * 10 for a in ALPHAS]; rhos = {a: [] for a in pa}
        for r in range(3):
            fo = folds(train, group, seed=r); pred = {a: np.zeros(len(y)) for a in pa}
            for k in range(5):
                tr, te = np.where(fo != k)[0], np.where(fo == k)[0]
                trp, tep = np.isin(owner, tr), np.isin(owner, te); f = kfit(P[trp], y[owner[trp]]); o = owner[tep]
                for a in pa:
                    pp = f(P[tep], a); pred[a][te] = [pp[o == i].mean() for i in te]
            for a in pa:
                rhos[a].append(spearmanr(pred[a], y).correlation)
        cv = {a: float(np.mean(v)) for a, v in rhos.items()}; best = max(cv, key=cv.get)
        out[name] = {"full_cv": cv[best], "alpha": best}
        print("%-17s full model CV %.4f (strength %g)" % (name, cv[best], best), flush=True)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    tri = np.array([pos[u] for u in train]); tei = np.array([pos[u] for u in test]); yte = np.array([first[u] for u in test], float)
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid)); curves = {}
    for name, FF in (("current", F), ("owner-only faces", Fo)):
        z = crowd_score(root, FF, idx, allu)
        out[name]["zero_train"] = float(spearmanr(z[tri], y).correlation); out[name]["zero_test"] = float(spearmanr(z[tei], yte).correlation)
        XL = StandardScaler().fit_transform(winner_features(FF, idx, allu)); Z = PCA(n_components=32, random_state=0).fit_transform(XL)[tri]; X = XL[tri]; zt = z[tri]
        res = {n: [] for n in NS}
        for s in range(20):
            g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
            isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]; o = herding(Z[Pi], max(NS))
            for n in NS:
                pk = Pi[o[:n]]; w = n / (n + 20); pr = rankdata(RidgeCV(alphas=A).fit(X[pk], y[pk]).predict(X[Ei])) / len(Ei)
                res[n].append(spearmanr((1 - w) * rankdata(zt[Ei]) / len(Ei) + w * pr, y[Ei]).correlation)
        curves[name] = res; out[name]["few"] = {n: float(np.mean(v)) for n, v in res.items()}
        print("%-17s zero-rating %.3f (test %.3f)  " % (name, out[name]["zero_train"], out[name]["zero_test"]) +
              "  ".join("n=%d %.3f" % (n, out[name]["few"][n]) for n in NS), flush=True)
    dcv = out["owner-only faces"]["full_cv"] - out["current"]["full_cv"]
    dfew = np.mean([out["owner-only faces"]["few"][n] - out["current"]["few"][n] for n in NS[:4]])
    wins = int(np.sum(np.array(curves["owner-only faces"][20]) > np.array(curves["current"][20])))
    ok = dcv >= -0.005 and dfew >= -0.005
    out["verdict"] = {"d_full_cv": dcv, "d_few_mean_5_40": float(dfew), "wins_n20": wins, "adopt": bool(ok)}
    print("owner-only faces: full CV %+.4f, few-rating mean (n=5-40) %+.4f, wins at n=20 %d/20 -> %s" % (
        dcv, dfew, wins, "adopt (correctness fix)" if ok else "keep the current crops"), flush=True)
    json.dump(out, open(os.path.join(root, "results/face_own.json"), "w"), indent=1)


if __name__ == "__main__":
    {"crop": crop, "embed": embed, "eval": evaluate}[sys.argv[1]](sys.argv[2] if len(sys.argv) > 2 else ".")
