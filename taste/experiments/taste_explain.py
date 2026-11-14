#!/usr/bin/env python3
"""What drives the user's taste: images and Instagram network (training accounts only).

1. Sheets of the top- and bottom-rated training accounts (3 cut-outs each) to look at.
2. Describable attributes: OpenCLIP text probes (positive vs negative wording) scored on every
   owner photo, averaged per account, correlated (Spearman) with the ratings.
3. Network: features from the user's follow list and Instagram's similar-accounts lists, alone
   and stacked on the image model, by grouped cross-validation on the training accounts.
Exploratory; the test set is not used. Output: results/explain.json, results/sheet_top.jpg,
results/sheet_bottom.jpg.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from probe_search import winner_features  # noqa: E402
from train_eval import fit_predict, folds, load, photo_index  # noqa: E402

BODY = [  # scored on the cut-out (clip_cut)
    ("blonde hair", "dark hair"), ("red hair", "dark hair"), ("long hair", "short hair"),
    ("curly hair", "straight hair"), ("a slim figure", "a curvy figure"),
    ("an athletic toned body", "a soft untoned body"), ("visible abs", "a soft stomach"),
    ("a muscular build", "a slender build"), ("tanned skin", "pale skin"), ("dark skin", "light skin"),
    ("visible tattoos", "no tattoos"), ("long legs", "short legs"), ("a petite frame", "a tall frame"),
    ("an elegant pose", "a casual pose"), ("a confident posture", "a shy posture"),
    ("a large chest", "a small chest"), ("wide hips", "narrow hips"),
]
FACE = [  # scored on the face crop (clip_face)
    ("heavy makeup", "no makeup"), ("a big smile", "a serious expression"), ("freckles", "clear skin"),
    ("full lips", "thin lips"), ("high cheekbones", "round cheeks"), ("large eyes", "small eyes"),
    ("a sharp jawline", "a soft jawline"), ("a youthful face", "a mature face"),
    ("lip fillers", "natural lips"), ("a natural look", "a glamorous look"), ("glasses", "no glasses"),
    ("light colored eyes", "dark eyes"), ("thick eyebrows", "thin eyebrows"), ("a small nose", "a large nose"),
]


def text_embed(prompts):
    import open_clip
    import torch
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    tok = open_clip.get_tokenizer("ViT-L-14")
    with torch.no_grad():
        t = model.encode_text(tok(prompts)).float()
    return torch.nn.functional.normalize(t, dim=-1).numpy()


def sheet(root, users, path, per=3, cell=150, gap=3):
    rows = []
    for u in users:
        rec = json.load(open(os.path.join(root, "cut", u, "owner.json")))
        codes = [p["code"] for p in rec["photos"]][:per]
        rows.append([os.path.join(root, "cut", u, c + "_thumb.jpg") for c in codes])
    cols = 4  # accounts per sheet row
    W = cols * (per * (cell + gap) + 12)
    H = ((len(rows) + cols - 1) // cols) * (cell + gap) + gap
    im = Image.new("RGB", (W, H), (245, 245, 242))
    for i, r in enumerate(rows):
        x0 = (i % cols) * (per * (cell + gap) + 12); y0 = (i // cols) * (cell + gap) + gap
        for j, p in enumerate(r):
            t = Image.open(p).convert("RGB"); t.thumbnail((cell, cell))
            im.paste(t, (x0 + j * (cell + gap) + (cell - t.width) // 2, y0 + (cell - t.height) // 2))
    im.save(path, quality=85)


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    follows = {}
    for l in open(os.path.join(root, "links/following.tsv")):
        if l.strip() and not l.startswith(("#", "username")):
            p = l.rstrip("\n").split("\t"); follows[p[0]] = p[2] == "1"
    chain = json.load(open(os.path.join(root, "links/chain.json")))["chain"]
    y = np.array([first[u] for u in train], float)
    fol = np.array([u in follows for u in train])
    out = {}

    # 1. sheets
    order = np.argsort(-y)
    top = [train[i] for i in order[:16]]; bottom = [train[i] for i in order[::-1][:16]]
    sheet(root, top, os.path.join(root, "results/sheet_top.jpg"))
    sheet(root, bottom, os.path.join(root, "results/sheet_bottom.jpg"))
    out["sheet_top_ratings"] = [first[u] for u in top]; out["sheet_bottom_ratings"] = [first[u] for u in bottom]

    # 2. attributes
    prompts = [("a photo of a person with " + a, "a photo of a person with " + b) for a, b in BODY + FACE]
    T = text_embed([p for pair in prompts for p in pair])
    attrs = []
    for k, (a, b) in enumerate(BODY + FACE):
        key = "clip_cut" if k < len(BODY) else "clip_face"
        d = T[2 * k] - T[2 * k + 1]
        s = F[key] @ d
        acc = np.array([s[idx[u]].mean() for u in train])
        r_all = spearmanr(acc, y).correlation
        r_sug = spearmanr(acc[~fol], y[~fol]).correlation
        attrs.append({"attribute": "%s (vs %s)" % (a, b), "rho_all": float(r_all), "rho_suggested_only": float(r_sug)})
    attrs.sort(key=lambda r: -abs(r["rho_all"]))
    out["attributes"] = attrs
    print("\nattribute (positive vs negative wording)                 rho all   rho suggested-only")
    for r in attrs:
        print("%-58s %+.3f    %+.3f" % (r["attribute"], r["rho_all"], r["rho_suggested_only"]))

    # 3. network features
    kept = sorted(set(split["train"]) | set(split["test"]))
    lists = {f: {e[1] for e in v} for f, v in chain.items()}
    verified = {u: bool(v) for u, v in follows.items()}
    for v in chain.values():
        for e in v:
            verified.setdefault(e[1], bool(e[3]))
    fidx = {f: i for i, f in enumerate(sorted(chain))}
    nbrs = {u: set() for u in kept}
    for f, L in lists.items():
        for s in L:
            if s in nbrs and f in nbrs:
                nbrs[f].add(s); nbrs[s].add(f)
    def net_feats(users, rated):
        """rated: dict user->rating usable as neighbour labels (training-fold only)."""
        mu = np.mean(list(rated.values()))
        rows, srcs = [], []
        for u in users:
            chain_in = sum(u in L for L in lists.values())
            nr = [rated[v] for v in nbrs.get(u, ()) if v in rated and v != u]
            rows.append([float(u in follows), np.log1p(chain_in), float(verified.get(u, False)),
                         np.mean(nr) if nr else mu, np.log1p(len(nr))])
            sv = np.zeros(len(fidx))
            for f, L in lists.items():
                if u in L: sv[fidx[f]] = 1
            if u in fidx: sv[fidx[u]] = 1
            srcs.append(sv)
        return np.array(rows), np.array(srcs)

    Ximg = StandardScaler().fit_transform(winner_features(F, idx, train))
    res = {k: [] for k in ("image", "follow status", "network", "network + source accounts",
                           "image + network", "image + network + source accounts")}
    sub = {k: {"suggested": [], "followed": []} for k in res}
    for rep in range(3):
        fo = folds(train, group, seed=rep)
        preds = {k: np.zeros(len(y)) for k in res}
        for k in range(5):
            trm, vam = fo != k, fo == k
            tr_u = [u for u, m in zip(train, trm) if m]; va_u = [u for u, m in zip(train, vam) if m]
            rated = {u: first[u] for u in tr_u}
            img_va = fit_predict(Ximg[trm], y[trm], Ximg[vam], 3000)
            # out-of-fold image predictions inside the training part, for stacking
            inner = folds(tr_u, group, seed=100 + rep)
            img_tr = np.zeros(trm.sum()); Xt, yt = Ximg[trm], y[trm]
            for j in range(5):
                a, b = inner != j, inner == j
                img_tr[b] = fit_predict(Xt[a], yt[a], Xt[b], 3000)
            # neighbour labels: for training rows use other training rows only (leave-self-out by construction)
            Ntr, Str = net_feats(tr_u, rated); Nva, Sva = net_feats(va_u, rated)
            preds["image"][vam] = img_va
            preds["follow status"][vam] = Nva[:, 0]
            def ridge(A, B):
                sc = StandardScaler().fit(A); m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(A), y[trm])
                return m.predict(sc.transform(B))
            preds["network"][vam] = ridge(Ntr, Nva)
            preds["network + source accounts"][vam] = ridge(np.c_[Ntr, Str], np.c_[Nva, Sva])
            preds["image + network"][vam] = ridge(np.c_[img_tr, Ntr], np.c_[img_va, Nva])
            preds["image + network + source accounts"][vam] = ridge(np.c_[img_tr, Ntr, Str], np.c_[img_va, Nva, Sva])
        for name, p in preds.items():
            res[name].append(spearmanr(p, y).correlation)
            sub[name]["suggested"].append(spearmanr(p[~fol], y[~fol]).correlation)
            sub[name]["followed"].append(spearmanr(p[fol], y[fol]).correlation)
    print("\nmodel (CV on training accounts)             all     suggested-only   followed-only")
    out["network_cv"] = {}
    for name in res:
        a, s_, f_ = np.nanmean(res[name]), np.nanmean(sub[name]["suggested"]), np.nanmean(sub[name]["followed"])
        out["network_cv"][name] = {"all": float(a), "suggested_only": float(s_), "followed_only": float(f_)}
        print("%-42s %.3f    %.3f            %.3f" % (name, a, s_, f_))
    json.dump(out, open(os.path.join(root, "results/explain.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
