#!/usr/bin/env python3
"""Follow-ups after the model comparison (see PROTOCOL.md "Larger image models" results):
(1) second look at the used test set for OpenCLIP L-14 + SigLIP;
(2) crowd face prior built with SigLIP / ViT-H embeddings of the SCUT faces (GPU for embedding).
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from probe_search import herding, winner_features  # noqa: E402
from train_eval import boot_ci, load, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]


def means(E, idx, users):
    out = np.stack([E[idx[u]].mean(0) for u in users]); return out / np.linalg.norm(out, axis=1, keepdims=True)


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True))
    follows = {l.split("\t")[0] for l in open(os.path.join(root, "links/following.tsv")) if l.strip() and not l.startswith(("#", "username"))}
    out = {}
    # (1) second look
    users = train + test
    X = np.c_[means(F["clip_nocloth"], idx, users), means(F["clip_face"], idx, users),
              means(M["siglip_nocloth"], idx, users), means(M["siglip_face"], idx, users)]
    sc = StandardScaler().fit(X[:len(train)]); Xtr, Xte = sc.transform(X[:len(train)]), sc.transform(X[len(train):])
    ytr = np.array([first[u] for u in train], float); yte = np.array([first[u] for u in test], float)
    p = Ridge(alpha=3000).fit(Xtr, ytr).predict(Xte)
    out["second_look_L_plus_siglip"] = {"spearman": float(spearmanr(p, yte).correlation), "ci95": boot_ci(p, yte),
                                        "note": "second look at an already-used test set; the clean result is 0.668 (original model)"}
    print("second look, test set, L-14 + SigLIP: %.3f %s" % (out["second_look_L_plus_siglip"]["spearman"], out["second_look_L_plus_siglip"]["ci95"]), flush=True)
    # (2) crowd prior with other embeddings
    import open_clip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    base = os.path.join(root, "datasets/scut/SCUT-FBP5500_v2/Images")
    names = [str(n) for n in S["names"]]
    emb = {"clipL": S["emb"]}
    for key, (arch, pre) in {"vith": ("ViT-H-14", "laion2b_s32b_b79k"), "siglip": ("ViT-SO400M-14-SigLIP-384", "webli")}.items():
        m, _, tf = open_clip.create_model_and_transforms(arch, pretrained=pre); m = m.eval().to(dev).half()
        es = []
        for i in range(0, len(names), 64):
            with torch.no_grad():
                e = m.encode_image(torch.stack([tf(Image.open(os.path.join(base, n)).convert("RGB")) for n in names[i:i + 64]]).to(dev).half()).float()
            es.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        emb[key] = np.concatenate(es); del m; torch.cuda.empty_cache()
        print("SCUT embedded with", key, flush=True)
    np.savez_compressed(os.path.join(root, "datasets/scut/scut_more.npz"), **{k: v for k, v in emb.items() if k != "clipL"})
    face_keys = {"clipL": F["clip_face"], "vith": M["vith_face"], "siglip": M["siglip_face"]}
    y = ytr; st = np.array([u not in follows for u in train])
    zs = {}
    for k in emb:
        s2 = StandardScaler().fit(emb[k]); r = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(s2.transform(emb[k]), S["mean"])
        z = r.predict(s2.transform(means(face_keys[k], idx, train))); zs[k] = (z - z.mean()) / z.std()
    zs["clipL + siglip"] = (zs["clipL"] + zs["siglip"]) / 2
    zs["all three"] = (zs["clipL"] + zs["vith"] + zs["siglip"]) / 3
    out["crowd_prior"] = {k: {"all": float(spearmanr(z, y).correlation), "strangers": float(spearmanr(z[st], y[st]).correlation)} for k, z in zs.items()}
    for k, v in out["crowd_prior"].items():
        print("crowd prior %-16s zero-rating all %.3f, strangers %.3f" % (k, v["all"], v["strangers"]), flush=True)
    best = max(out["crowd_prior"], key=lambda k: out["crowd_prior"][k]["all"]); out["best_prior"] = best
    # few-rating blend k=20 with the best prior (ratings model: L-14, herding), same 20 splits
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    Xall = StandardScaler().fit_transform(winner_features(F, idx, allu)); tr = np.array([pos[u] for u in train])
    Z32 = PCA(n_components=32, random_state=0).fit_transform(Xall); XL = Xall[tr]
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid))
    res = {k: {n: [] for n in NS} for k in ("clipL", best)}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); P, E = np.where(isP)[0], np.where(~isP)[0]
        order = herding(Z32[tr[P]], max(NS))
        for n in NS:
            pk = order[:n]; pr = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(XL[P][pk], y[P][pk]).predict(XL[E]); ro = rankdata(pr) / len(pr)
            w = n / (n + 20)
            for k in res:
                res[k][n].append(spearmanr((1 - w) * rankdata(zs[k][E]) / len(E) + w * ro, y[E]).correlation)
    out["few_blend_k20"] = {k: {n: float(np.nanmean(v)) for n, v in d.items()} for k, d in res.items()}
    wins = int(np.sum(np.array(res[best][20]) > np.array(res["clipL"][20]))) if best != "clipL" else 0
    out["best_prior_wins_n20"] = wins
    for k, d in out["few_blend_k20"].items():
        print("  blend k=20, prior %-16s %s" % (k, "  ".join("%.3f" % d[n] for n in NS)))
    print("best prior %s beats the L-14 prior at n=20 in %d/20 splits" % (best, wins))
    json.dump(out, open(os.path.join(root, "results/followups.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
