#!/usr/bin/env python3
"""A rated batch of MEBeauty faces: Black women and men, and men of other groups (PROTOCOL.md,
"MEBeauty batch"). Group labels are the dataset's own; no model labels anyone.

  mebeauty_batch.py pick ROOT     60 faces by kernel herding within strata -> mebatch/<id>.jpg (photo shown),
                                  mebatch/<id>_face.jpg (face crop as in owner_cutouts.py), results/mebatch_set.json
  mebeauty_batch.py embed ROOT    CLIP L-14, SigLIP and SFace codes of the face crops -> mebatch/codes.npz
  mebeauty_batch.py freeze ROOT   predictions before any rating -> results/mebatch_pred.json (SHA-256 printed)
  mebeauty_batch.py eval RATINGS.jsonl ROOT   -> results/mebatch_eval.json
"""
import hashlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]
STRATA = [("female", ["black"], 15), ("male", ["black"], 15), ("male", ["caucasian"], 6), ("male", ["asian"], 6),
          ("male", ["hispanic"], 6), ("male", ["indian"], 6), ("male", ["mideastern"], 6)]
ALPHAS = np.logspace(-1, 5, 25)


def pick(root):
    import cv2
    from PIL import Image
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from idscore.faceid import FaceID
    from mebeauty_v2 import read_rows
    from stock_probe import herding
    rows = read_rows(root, ["image_id", "gender", "ethnicity", "image_native"])
    C = np.load(os.path.join(root, "datasets/mebeauty/v2/codes.npz"), allow_pickle=True)
    pos = {str(k): i for i, k in enumerate(C["image_id"])}; X = C["clip"][[pos[r["image_id"]] for r in rows]]
    Z = PCA(n_components=32, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    sex = np.array([r["gender"] for r in rows]); eth = np.array([r["ethnicity"] for r in rows])
    fid = FaceID(os.path.join(root, "models", "opencv_zoo")); out = os.path.join(root, "mebatch"); os.makedirs(out, exist_ok=True)
    chosen = []
    for s, es, n in STRATA:
        cand = np.where((sex == s) & np.isin(eth, es))[0]; got = 0
        for j in herding(Z[cand], len(cand)):  # herding order; skip photos with no detectable face
            i = int(cand[j]); im = Image.open(io.BytesIO(rows[i]["image_native"]["bytes"])).convert("RGB")
            if max(im.size) > 1600:  # as owner_cutouts.load: photos are processed at most 1600 px
                g = 1600 / max(im.size); im = im.resize((round(im.size[0] * g), round(im.size[1] * g)), Image.LANCZOS)
            rgb = np.array(im); row = fid.detect(rgb)
            if row is None:
                continue
            chosen.append((i, rgb, row)); got += 1
            if got == n:
                break
    items = []
    for i, rgb, row in sorted(chosen, key=lambda t: hashlib.sha1(rows[t[0]]["image_id"].encode()).hexdigest()):
        mid = "m" + hashlib.sha1(rows[i]["image_id"].encode()).hexdigest()[:7]
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
    json.dump(items, open(os.path.join(root, "results/mebatch_set.json"), "w"), indent=1)
    print("picked %d faces: %s" % (len(items), {"%s %s" % (s, "/".join(es)): n for s, es, n in STRATA}))


def embed(root):
    import open_clip
    import torch
    from PIL import Image
    items = json.load(open(os.path.join(root, "results/mebatch_set.json"))); dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"id": np.array([it["id"] for it in items])}
    for key, arch, pre in (("clip_face", "ViT-L-14", "laion2b_s32b_b82k"), ("siglip_face", "ViT-SO400M-14-SigLIP-384", "webli")):
        m, _, tf = open_clip.create_model_and_transforms(arch, pretrained=pre); m = m.eval().to(dev).half()
        x = torch.stack([tf(Image.open(os.path.join(root, "mebatch", it["id"] + "_face.jpg")).convert("RGB")) for it in items]).to(dev).half()
        with torch.no_grad():
            out[key] = torch.nn.functional.normalize(m.encode_image(x).float(), dim=-1).cpu().numpy()
        print(key, out[key].shape, flush=True); del m
    np.savez_compressed(os.path.join(root, "mebatch/codes.npz"), **out); print("saved", flush=True)


def l2(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def freeze(root):
    from scipy.stats import rankdata
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from group_transfer import scut_crowd_models
    from mebeauty_v2 import read_rows
    from sface_input import sface_means
    from train_eval import load, mean_vec, photo_index
    items = json.load(open(os.path.join(root, "results/mebatch_set.json")))
    B = np.load(os.path.join(root, "mebatch/codes.npz"), allow_pickle=True); assert list(B["id"]) == [it["id"] for it in items]
    bc, bg, bs = l2(B["clip_face"].astype(float)), l2(B["siglip_face"].astype(float)), l2(np.array([it["sface"] for it in items]))
    # 1. crowd (SCUT, adopted)
    crowd = scut_crowd_models(root)(bc, bs)
    # 2. crowd M: SCUT + MEBeauty dating crowd trained WITHOUT these 60 faces, read on the v2 crops it was trained on
    rows = read_rows(root, ["image_id", "ratings"]); C = np.load(os.path.join(root, "datasets/mebeauty/v2/codes.npz"), allow_pickle=True)
    pos = {str(k): i for i, k in enumerate(C["image_id"])}; batch = {it["image_id"] for it in items}
    tr = [r for r in rows if r["image_id"] not in batch and len(r["ratings"]) >= 3]
    Xc = C["clip"][[pos[r["image_id"]] for r in tr]].astype(float); Xs = C["sface"][[pos[r["image_id"]] for r in tr]]
    ym = np.array([np.mean([x["score"] for x in r["ratings"]]) for r in tr]); ok = ~np.isnan(Xs).any(1)
    s1 = StandardScaler().fit(Xc); m1 = RidgeCV(alphas=ALPHAS).fit(s1.transform(Xc), ym)
    s2 = StandardScaler().fit(Xs[ok]); m2 = RidgeCV(alphas=ALPHAS).fit(s2.transform(Xs[ok]), ym[ok])
    bi = [pos[it["image_id"]] for it in items]
    me = zs(zs(m1.predict(s1.transform(C["clip"][bi].astype(float)))) + zs(m2.predict(s2.transform(np.nan_to_num(C["sface"][bi])))))
    crowd_m = zs(crowd + me)
    # 3. your taste from faces: ridge on all 407 rated people's face codes (CLIP face, SigLIP face, SFace)
    F, train, test, first, R, group, split, same = load(root); idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True)); Fm = dict(F); Fm["siglip_face"] = M["siglip_face"]
    people = sorted(u for u in set(split["train"]) | set(split["test"]) if u in first); y = np.array([first[u] for u in people], float)
    Xp = np.c_[mean_vec(F, idx, people, "clip_face"), mean_vec(Fm, idx, people, "siglip_face"), sface_means(root, people)]
    sc = StandardScaler().fit(Xp); face_model = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xp), y)
    yours = face_model.predict(sc.transform(np.c_[bc, bg, bs]))
    w = len(people) / (len(people) + 20); n = len(items)
    blend = (1 - w) * rankdata(crowd) / n + w * rankdata(yours) / n
    pred = {"ids": [it["id"] for it in items], "crowd": crowd.round(4).tolist(), "crowd_M": crowd_m.round(4).tolist(),
            "your_face_model": yours.round(4).tolist(), "blend": blend.round(4).tolist(),
            "trained_on": len(people), "face_model_alpha": float(face_model.alpha_)}
    path = os.path.join(root, "results/mebatch_pred.json"); json.dump(pred, open(path, "w"), indent=1)
    print("frozen predictions for %d faces (face model on %d rated people); SHA-256 %s" % (
        n, len(people), hashlib.sha256(open(path, "rb").read()).hexdigest()))


def evaluate(ratings_path, root):
    from scipy.stats import spearmanr
    items = {it["id"]: it for it in json.load(open(os.path.join(root, "results/mebatch_set.json")))}
    pred = json.load(open(os.path.join(root, "results/mebatch_pred.json")))
    first, second = {}, {}
    for line in open(ratings_path):
        r = json.loads(line)
        if r.get("rating") is None:
            continue
        (first if r.get("pass", 1) == 1 else second)[r["id"]] = float(r["rating"])
    ids = [i for i in pred["ids"] if i in first]; y = np.array([first[i] for i in ids])
    sel = {"all": np.ones(len(ids), bool),
           "Black women and men": np.array([items[i]["ethnicity"] == "black" for i in ids]),
           "Black women": np.array([items[i]["ethnicity"] == "black" and items[i]["gender"] == "female" for i in ids]),
           "men": np.array([items[i]["gender"] == "male" for i in ids])}
    rng = np.random.default_rng(0); out = {"n_rated": len(ids)}
    rep = [i for i in second if i in first]
    if rep:
        out["consistency"] = float(spearmanr([first[i] for i in rep], [second[i] for i in rep])[0]); out["n_repeats"] = len(rep)
        print("consistency on %d repeats: %.3f" % (len(rep), out["consistency"]))
    print("%-22s" % "" + "".join("%26s" % k for k in sel))
    for name in ("your_face_model", "blend", "crowd", "crowd_M"):
        p = np.array([pred[name][pred["ids"].index(i)] for i in ids]); out[name] = {}
        line = "%-22s" % name
        for k, m in sel.items():
            r = spearmanr(p[m], y[m])[0]; bs = []
            for _ in range(2000):
                b = rng.integers(0, m.sum(), m.sum()); bs.append(spearmanr(p[m][b], y[m][b])[0])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5]); out[name][k] = [float(r), float(lo), float(hi)]
            line += "%26s" % ("%.2f [%.2f, %.2f] n=%d" % (r, lo, hi, m.sum()))
        print(line)
    json.dump(out, open(os.path.join(root, "results/mebatch_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "eval":
        evaluate(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ".")
    else:
        {"pick": pick, "embed": embed, "freeze": freeze}[cmd](sys.argv[2] if len(sys.argv) > 2 else ".")
