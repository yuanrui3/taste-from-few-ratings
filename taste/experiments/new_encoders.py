#!/usr/bin/env python3
"""Check 3 of the bottleneck review (PROTOCOL.md, "Three checks from the bottleneck review"): newer image
models in the full model and in the few-rating curve. No new ratings; training accounts only.

  new_encoders.py embed ROOT [LIBS]  (GPU) -> ROOT/features_newenc.npz, rows in features.npz order:
        pe_nocloth, pe_face      Perception Encoder PE-Core-L14-336 (Apache-2.0), output embedding
        sig2_nocloth, sig2_face  SigLIP 2 SO400M/14 at 378 px (Apache-2.0), output embedding
        sap_nocloth, sap_face    Sapiens 0.6B pretrained on images of people (CC BY-NC 4.0), mean of the
                                 last layer's tokens
     inputs: cut/<person>/<photo>_nocloth.jpg and <photo>_faceown.jpg, as the full model uses.
     LIBS: folder with open_clip >= 3.0 and timm >= 1.0.17 (needed for PE and SigLIP 2), put first on the path.
  new_encoders.py eval ROOT  -> ROOT/results/new_encoders.json
     (A) person-level CV of the adopted full model with each encoder's two codes added (and alone);
     (B) the push_down.py few-rating harness with each encoder's person-mean codes in place of L-14.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

ENCODERS = {"pe": "hf-hub:timm/PE-Core-L-14-336", "sig2": "hf-hub:timm/ViT-SO400M-14-SigLIP2-378"}
SAPIENS = ("facebook/sapiens-pretrain-0.6b-torchscript", "sapiens_0.6b_epoch_1600_torchscript.pt2")
VARIANTS = {"nocloth": "%s_nocloth.jpg", "face": "%s_faceown.jpg"}
NS = [5, 10, 20, 40, 80]
CV_ALPHAS = [3e3, 1e4, 3e4]


def embed(root, libs=None):
    if libs:
        sys.path.insert(0, os.path.abspath(libs))
    import torch
    from PIL import Image
    dev = "cuda"
    F = np.load(os.path.join(root, "features.npz"), allow_pickle=True)
    users, codes = [str(u) for u in F["user"]], [str(c) for c in F["code"]]
    out = {"user": F["user"], "code": F["code"]}

    def run(fn, tf, key, bs, half=True):
        for v, pat in VARIANTS.items():
            res = []
            for i in range(0, len(users), bs):
                ims = [Image.open(os.path.join(root, "cut", u, pat % c)).convert("RGB") for u, c in zip(users[i:i + bs], codes[i:i + bs])]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=half):
                    e = fn(torch.stack([tf(im) for im in ims]).to(dev)).float()
                res.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy().astype(np.float16))
            out["%s_%s" % (key, v)] = np.concatenate(res)
            print("%s %s: %d photos, %d dims" % (key, v, len(users), out["%s_%s" % (key, v)].shape[1]), flush=True)

    import open_clip
    for key, name in ENCODERS.items():
        m, _, pre = open_clip.create_model_and_transforms(name)
        m = m.eval().to(dev)
        run(m.encode_image, pre, key, 64)
        del m
        torch.cuda.empty_cache()

    from huggingface_hub import hf_hub_download
    sap = torch.jit.load(hf_hub_download(*SAPIENS), map_location="cpu").eval().to(dev)  # TorchScript traced at 1024 x 1024
    mean = torch.tensor([123.5, 116.5, 103.5]).view(3, 1, 1)
    std = torch.tensor([58.5, 57.0, 57.5]).view(3, 1, 1)

    def sap_tf(im):  # letterbox on grey to the 1024 x 1024 input the released TorchScript was traced at
        w, h = im.size
        s = min(1024 / w, 1024 / h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        canvas = Image.new("RGB", (1024, 1024), (128, 128, 128))
        canvas.paste(im, ((1024 - im.size[0]) // 2, (1024 - im.size[1]) // 2))
        x = torch.from_numpy(np.asarray(canvas, dtype=np.float32)).permute(2, 0, 1)
        return (x - mean) / std

    def sap_fn(x):
        f = sap(x)
        f = f[0] if isinstance(f, (tuple, list)) else f
        return f.mean(dim=(2, 3)) if f.dim() == 4 else f.mean(dim=1)

    run(sap_fn, sap_tf, "sap", 8, half=False)  # an exported graph keeps its own dtypes
    np.savez_compressed(os.path.join(root, "features_newenc.npz"), **out)
    print("wrote features_newenc.npz", flush=True)


def evaluate(root):
    from scipy.stats import rankdata, spearmanr
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from improve_r2 import kfit
    from probe_search import herding, winner_features
    from sface_input import sface_means
    from taste_onboard import crowd_score
    from train_eval import folds, load, mean_vec, photo_index
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True))
    O = dict(np.load(os.path.join(root, "features_faceown.npz"), allow_pickle=True))
    N = dict(np.load(os.path.join(root, "features_newenc.npz"), allow_pickle=True))
    assert (N["user"] == F["user"]).all() and (N["code"] == F["code"]).all()
    y = np.array([first[u] for u in train], float)
    rows = np.concatenate([idx[u] for u in train])
    owner = np.concatenate([[i] * len(idx[u]) for i, u in enumerate(train)])
    base = np.c_[F["clip_nocloth"][rows], O["clip_face"][rows], M["siglip_nocloth"][rows], O["siglip_face"][rows],
                 sface_means(root, train)[owner]].astype(np.float64)
    enc = {k: np.c_[N["%s_nocloth" % k][rows], N["%s_face" % k][rows]].astype(np.float64) for k in ("pe", "sig2", "sap")}

    def cv(P):
        r = np.zeros((3, len(CV_ALPHAS)))
        for s in range(3):
            fo = folds(train, group, seed=s)
            pp = np.zeros((len(CV_ALPHAS), len(train)))
            for k in range(5):
                trp = np.isin(owner, np.where(fo != k)[0])
                tep = np.where(np.isin(owner, np.where(fo == k)[0]))[0]
                f = kfit(P[trp], y[owner[trp]])
                for j, a in enumerate(CV_ALPHAS):
                    ph = f(P[tep], a)
                    for i in np.unique(owner[tep]):
                        pp[j, i] = ph[owner[tep] == i].mean()
            r[s] = [spearmanr(pp[j], y)[0] for j in range(len(CV_ALPHAS))]
        m = r.mean(0)
        j = int(np.argmax(m))
        return {"cv_spearman": round(float(m[j]), 4), "alpha": CV_ALPHAS[j], "by_alpha": [round(float(v), 4) for v in m]}

    out = {"A_full_model": {"current": cv(base)}}
    print("current full model: %s" % out["A_full_model"]["current"], flush=True)
    for k, E in enc.items():
        out["A_full_model"]["current + %s" % k] = cv(np.c_[base, E])
        out["A_full_model"]["%s alone" % k] = cv(E)
        print("current + %s: %s | %s alone: %s" % (k, out["A_full_model"]["current + %s" % k], k, out["A_full_model"]["%s alone" % k]), flush=True)
    b0 = out["A_full_model"]["current"]["cv_spearman"]
    out["A_verdict"] = {k: {"gain": round(out["A_full_model"]["current + %s" % k]["cv_spearman"] - b0, 4),
                            "adopt": bool(out["A_full_model"]["current + %s" % k]["cv_spearman"] >= b0 + 0.02)} for k in enc}

    # (B) few ratings: the push_down.py harness, ridge on person-mean codes, blended with the crowd prior (k = 20)
    A = np.logspace(-1, 5, 25)
    allu = sorted(set(split["train"]) | set(split["test"]))
    pos = {u: i for i, u in enumerate(allu)}
    tr = np.array([pos[u] for u in train])
    ytr = y
    z2 = crowd_score(root, F, idx, allu)
    XL = StandardScaler().fit_transform(winner_features(F, idx, allu))
    Z = PCA(n_components=32, random_state=0).fit_transform(XL)[tr]
    Xs = {"baseline (L-14)": XL[tr]}
    for k in enc:
        Xs[k] = StandardScaler().fit_transform(np.concatenate([mean_vec(N, idx, allu, "%s_nocloth" % k),
                                                               mean_vec(N, idx, allu, "%s_face" % k)], 1))[tr]
    gid = np.array([group.get(u, u) for u in train])
    ug = sorted(set(gid))
    res = {m: {n: [] for n in NS} for m in Xs}
    for s in range(20):
        g = np.random.default_rng(1000 + s)
        gs = ug[:]
        g.shuffle(gs)
        pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid])
        Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        o = herding(Z[Pi], max(NS))
        for n in NS:
            pk = Pi[o[:n]]
            w = n / (n + 20)
            zz = rankdata(z2[tr][Ei]) / len(Ei)
            for m, X in Xs.items():
                pr = rankdata(RidgeCV(alphas=A).fit(X[pk], ytr[pk]).predict(X[Ei])) / len(Ei)
                res[m][n].append(float(spearmanr((1 - w) * zz + w * pr, ytr[Ei])[0]))
    out["B_few"] = {m: {str(n): round(float(np.mean(v)), 4) for n, v in d.items()} for m, d in res.items()}
    bb = res["baseline (L-14)"]
    out["B_verdict"] = {}
    for m in enc:
        gain = float(np.mean([np.mean(res[m][n]) - np.mean(bb[n]) for n in NS[:4]]))
        wins = int(np.sum(np.array(res[m][20]) > np.array(bb[20])))
        out["B_verdict"][m] = {"gain_5_40": round(gain, 4), "wins_n20": wins, "adopt": bool(gain >= 0.02 and wins >= 15)}
    for m in Xs:
        print("%-16s %s" % (m, "  ".join("n=%d %.3f" % (n, out["B_few"][m][str(n)]) for n in NS)), flush=True)
    print("A verdict:", out["A_verdict"], "\nB verdict:", out["B_verdict"], flush=True)
    json.dump(out, open(os.path.join(root, "results/new_encoders.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        embed(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        evaluate(sys.argv[2])
