#!/usr/bin/env python3
"""Second MEBeauty batch: 100 new photos, mostly Black women and men (PROTOCOL.md, "MEBeauty batch 2").

  mebatch2.py pick ROOT     35 Black women, 35 Black men, 6 men each of five other groups, none from batch 1,
                            kernel herding that counts batch 1 as already covered -> mebatch2/, results/mebatch2_set.json
  mebatch2.py embed ROOT    (GPU) codes of the face crops -> mebatch2/codes.npz
  mebatch2.py freeze ROOT RATINGS1.jsonl   predictions before rating: face model on the pool alone, and on the
                            pool + batch 1 ratings (weight x5) -> results/mebatch2_pred.json
  mebatch2.py eval RATINGS2.jsonl ROOT     -> results/mebatch2_eval.json
"""
import hashlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]
STRATA = [("female", "black", 35), ("male", "black", 35), ("male", "caucasian", 6), ("male", "asian", 6),
          ("male", "hispanic", 6), ("male", "indian", 6), ("male", "mideastern", 6)]
ALPHAS = np.logspace(-1, 5, 25)
OUT = "mebatch2"


def herding_after(Z, prior, cand):
    """Kernel herding over Z[cand], with the rows `prior` counted as already chosen; returns the order of cand."""
    sq = (Z ** 2).sum(1); D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0)
    s2 = np.median(D2[D2 > 0]); K = np.exp(-D2 / s2); mu = K.mean(1)
    acc = K[:, prior].sum(1) if len(prior) else np.zeros(len(Z)); t0 = len(prior); order = []
    for t in range(len(cand)):
        score = mu[cand] - acc[cand] / (t0 + t + 1); score[order] = -np.inf
        j = int(np.argmax(score)); order.append(j); acc += K[:, cand[j]]
    return order


def pick(root):
    import cv2
    from PIL import Image
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from idscore.faceid import FaceID
    from mebeauty_v2 import read_rows
    rows = read_rows(root, ["image_id", "gender", "ethnicity", "image_native"])
    C = np.load(os.path.join(root, "datasets/mebeauty/v2/codes.npz"), allow_pickle=True)
    pos = {str(k): i for i, k in enumerate(C["image_id"])}; X = C["clip"][[pos[r["image_id"]] for r in rows]]
    Z = PCA(n_components=32, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    sex = np.array([r["gender"] for r in rows]); eth = np.array([r["ethnicity"] for r in rows])
    done = {it["image_id"] for it in json.load(open(os.path.join(root, "results/mebatch_set.json")))}
    used = np.array([r["image_id"] in done for r in rows])
    fid = FaceID(os.path.join(root, "models", "opencv_zoo")); out = os.path.join(root, OUT); os.makedirs(out, exist_ok=True)
    chosen = []
    for s, e, n in STRATA:
        grp = (sex == s) & (eth == e); cand = np.where(grp & ~used)[0]; prior = np.where(grp & used)[0]; got = 0
        for j in herding_after(Z, prior, cand):
            i = int(cand[j]); im = Image.open(io.BytesIO(rows[i]["image_native"]["bytes"])).convert("RGB")
            if max(im.size) > 1600:
                g = 1600 / max(im.size); im = im.resize((round(im.size[0] * g), round(im.size[1] * g)), Image.LANCZOS)
            rgb = np.array(im); row = fid.detect(rgb)
            if row is None:
                continue
            chosen.append((i, rgb, row)); got += 1
            if got == n:
                break
    items = []
    for i, rgb, row in sorted(chosen, key=lambda t: hashlib.sha1(rows[t[0]]["image_id"].encode()).hexdigest()):
        mid = "n" + hashlib.sha1(rows[i]["image_id"].encode()).hexdigest()[:7]
        H, W = rgb.shape[:2]; f = min(1.0, 900 / max(H, W))
        show = cv2.resize(rgb, (round(W * f), round(H * f)), interpolation=cv2.INTER_AREA) if f < 1 else rgb
        cv2.imwrite(os.path.join(out, mid + ".jpg"), cv2.cvtColor(show, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        fx, fy, fw, fh = [float(v) for v in row[:4]]; cx, cy = fx + fw / 2, fy + fh / 2; side = 1.1 * max(fw, fh)
        face = rgb[int(max(0, cy - side)):int(min(H, cy + side)), int(max(0, cx - side)):int(min(W, cx + side))]
        g = min(1.0, 224 / max(face.shape[:2]))
        if g < 1:
            face = cv2.resize(face, (round(face.shape[1] * g), round(face.shape[0] * g)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out, mid + "_face.jpg"), cv2.cvtColor(face, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        e, _ = fid.embed(rgb, row)
        items.append({"id": mid, "image_id": rows[i]["image_id"], "gender": rows[i]["gender"], "ethnicity": rows[i]["ethnicity"],
                      "sface": [float(v) for v in e]})
    json.dump(items, open(os.path.join(root, "results/mebatch2_set.json"), "w"), indent=1)
    print("picked %d new photos: %s" % (len(items), {"%s %s" % (s, e): n for s, e, n in STRATA}))


def embed(root):
    import open_clip
    import torch
    from PIL import Image
    items = json.load(open(os.path.join(root, "results/mebatch2_set.json"))); dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"id": np.array([it["id"] for it in items])}
    for key, arch, pre in (("clip_face", "ViT-L-14", "laion2b_s32b_b82k"), ("siglip_face", "ViT-SO400M-14-SigLIP-384", "webli")):
        m, _, tf = open_clip.create_model_and_transforms(arch, pretrained=pre); m = m.eval().to(dev).half()
        x = torch.stack([tf(Image.open(os.path.join(root, OUT, it["id"] + "_face.jpg")).convert("RGB")) for it in items]).to(dev).half()
        with torch.no_grad():
            out[key] = torch.nn.functional.normalize(m.encode_image(x).float(), dim=-1).cpu().numpy()
        print(key, out[key].shape, flush=True); del m
    np.savez_compressed(os.path.join(root, OUT, "codes.npz"), **out); print("saved", flush=True)


def batch_xy(root, set_name, codes_dir, ratings_path=None):
    from mebeauty_batch import l2
    items = json.load(open(os.path.join(root, "results", set_name))); B = np.load(os.path.join(root, codes_dir, "codes.npz"), allow_pickle=True)
    assert list(B["id"]) == [it["id"] for it in items]
    X = np.c_[l2(B["clip_face"].astype(float)), l2(B["siglip_face"].astype(float)), l2(np.array([it["sface"] for it in items]))]
    if ratings_path is None:
        return items, X, None
    rat = {}
    for line in open(ratings_path):
        r = json.loads(line)
        if r.get("pass", 1) == 1:
            rat[r["id"]] = r.get("rating")
    return items, X, np.array([np.nan if rat.get(it["id"]) is None else float(rat[it["id"]]) for it in items])


def freeze(root, ratings1):
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sface_input import sface_means
    from train_eval import load, mean_vec, photo_index
    F, train, test, first, R, group, split, same = load(root); idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True)); Fm = dict(F); Fm["siglip_face"] = M["siglip_face"]
    people = sorted(u for u in set(split["train"]) | set(split["test"]) if u in first); yp = np.array([first[u] for u in people], float)
    Xp = np.c_[mean_vec(F, idx, people, "clip_face"), mean_vec(Fm, idx, people, "siglip_face"), sface_means(root, people)]
    _, X1, y1 = batch_xy(root, "mebatch_set.json", "mebatch", ratings1); ok1 = ~np.isnan(y1)
    items2, X2, _ = batch_xy(root, "mebatch2_set.json", OUT)
    sc = StandardScaler().fit(Xp); m0 = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xp), yp)
    X = np.r_[Xp, X1[ok1]]; y = np.r_[yp, y1[ok1]]; w = np.r_[np.ones(len(yp)), 5 * np.ones(ok1.sum())]
    sc1 = StandardScaler().fit(X); m1 = Ridge(alpha=m0.alpha_).fit(sc1.transform(X), y, sample_weight=w)
    pred = {"ids": [it["id"] for it in items2], "pool_only": m0.predict(sc.transform(X2)).round(4).tolist(),
            "pool_plus_batch1": m1.predict(sc1.transform(X2)).round(4).tolist(), "batch1_rated": int(ok1.sum())}
    path = os.path.join(root, "results/mebatch2_pred.json"); json.dump(pred, open(path, "w"), indent=1)
    print("frozen predictions for %d photos; SHA-256 %s" % (len(items2), hashlib.sha256(open(path, "rb").read()).hexdigest()))


def evaluate(ratings2, root):
    from scipy.stats import spearmanr
    pred = json.load(open(os.path.join(root, "results/mebatch2_pred.json")))
    items, X2, y = batch_xy(root, "mebatch2_set.json", OUT, ratings2); ok = ~np.isnan(y)
    second = {}
    for line in open(ratings2):
        r = json.loads(line)
        if r.get("pass", 1) == 2 and r.get("rating") is not None:
            second[r["id"]] = float(r["rating"])
    out = {"n_rated": int(ok.sum())}
    first = {it["id"]: v for it, v in zip(items, y) if not np.isnan(v)}; rep = [i for i in second if i in first]
    if rep:
        out["consistency"] = float(spearmanr([first[i] for i in rep], [second[i] for i in rep])[0])
        print("consistency on %d repeats: %.3f" % (len(rep), out["consistency"]))
    sel = {"all": ok, "Black women and men": ok & np.array([it["ethnicity"] == "black" for it in items]),
           "Black women": ok & np.array([it["ethnicity"] == "black" and it["gender"] == "female" for it in items]),
           "Black men": ok & np.array([it["ethnicity"] == "black" and it["gender"] == "male" for it in items]),
           "men": ok & np.array([it["gender"] == "male" for it in items])}
    rng = np.random.default_rng(0)
    for name in ("pool_only", "pool_plus_batch1"):
        p = np.array(pred[name]); out[name] = {}; line = "%-18s" % name
        for k, m in sel.items():
            r = spearmanr(p[m], y[m])[0]
            bs = [spearmanr(p[m][b], y[m][b])[0] for b in (rng.integers(0, m.sum(), m.sum()) for _ in range(2000))]
            lo, hi = np.nanpercentile(bs, [2.5, 97.5]); out[name][k] = [float(r), float(lo), float(hi), int(m.sum())]
            line += "  %s %.2f [%.2f, %.2f]" % (k, r, lo, hi)
        print(line)
    json.dump(out, open(os.path.join(root, "results/mebatch2_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "eval":
        evaluate(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ".")
    elif cmd == "freeze":
        freeze(sys.argv[2], sys.argv[3])
    else:
        {"pick": pick, "embed": embed}[cmd](sys.argv[2] if len(sys.argv) > 2 else ".")
