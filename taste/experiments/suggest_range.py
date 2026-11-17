#!/usr/bin/env python3
"""How much range in sexual suggestiveness do the saved photos already have? (GPU cluster)

A look before any rating or new collection. No ratings of this kind exist, nothing here is a test,
and no rule is decided by it. "Suggestiveness" means the viewer's judgement of how much a photo has
sexual implications; the scorers below are crowd or objective stand-ins for that judgement, used
only to see how much spread the saved photos have and which people have no photo at the high end.

Pool: split/suggest_pool.json (accounts that are verified, or were suggested from a verified
account's similar-accounts list; unverified follows and their suggestions are held back). People
whose median apparent age is under --age-min are set aside (a screen, not proof of age).

Scored on the owner's crop from the original photo (crop box written by owner_cutouts.py):
  laion_nsfw  LAION CLIP-based NSFW detector (MIT): small MLP on OpenAI CLIP ViT-L/14 codes, 0..1
  p_sexy ...  LukeJacob2023/nsfw-image-detector (Apache-2.0): ViT-B with classes drawings / hentai /
              neutral / porn / sexy (the label scheme of GantMan's nsfw_model; that one needs TensorFlow)
  nn_*        NudeNet 3.4 detector (MIT): boxes of exposed and covered body parts
  zs_suggest  zero-shot text probe with the same OpenAI CLIP model
  bare        skin / (skin + clothing) from the clothes parser already run by owner_cutouts.py
  age         InsightFace buffalo_l genderage on the owner's face (for the screen only)

Outputs: results/suggest_scores.npz (per photo), results/suggest_range.json (summary),
results/suggest_sheet.jpg (photos at six levels of the combined score, to check it by eye).
`--prefetch` downloads the weights (login node); `--summary-only` redoes the summary from the npz.
"""
import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from owner_cutouts import load as load_photo  # noqa: E402  (same resizing as the owner step)

LAION_URL = ("https://github.com/LAION-AI/CLIP-based-NSFW-Detector/files/10250461/"
             "clip_autokeras_binary_nsfw.zip")
SEXY5 = "LukeJacob2023/nsfw-image-detector"
ZS_POS = ["a sexually suggestive photo of a person", "a seductive, provocative photo of a person",
          "a sexy photo of a person in revealing clothing"]
ZS_NEG = ["an ordinary photo of a person", "a casual everyday photo of a person in regular clothes",
          "a plain, non-sexual portrait of a person"]
NN_EXPOSED = ["BELLY_EXPOSED", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED", "MALE_BREAST_EXPOSED",
              "ARMPITS_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_EXPOSED"]
NN_EXPLICIT = ["FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
               "ANUS_EXPOSED", "BUTTOCKS_EXPOSED"]
NN_MIN = 0.3   # detection score kept
COMBINED = ["laion_nsfw", "p_sexyporn", "zs_suggest", "bare", "nn_area"]  # averaged as percentile ranks
SHEET_BANDS = [(0, 5), (20, 25), (45, 50), (70, 75), (90, 95), (99, 100)]


def laion_mlp():
    import torch
    from torch import nn

    class Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.zeros([768]))
            self.register_buffer("variance", torch.ones([768]))

        def forward(self, x):
            return (x - self.mean) / self.variance.sqrt()

    class M(nn.Module):  # layout of the detector's published PyTorch weights
        def __init__(self):
            super().__init__()
            self.norm = Norm()
            self.linear_1, self.linear_2 = nn.Linear(768, 64), nn.Linear(64, 512)
            self.linear_3, self.linear_4 = nn.Linear(512, 256), nn.Linear(256, 1)

        def forward(self, x):
            x = torch.relu(self.linear_1(self.norm(x)))
            x = torch.relu(self.linear_3(torch.relu(self.linear_2(x))))
            return torch.sigmoid(self.linear_4(x)).squeeze(-1)

    return M()


def laion_path(models):
    return os.path.join(models, "laion_nsfw", "clip_autokeras_binary_nsfw.pth")


def prefetch(models):
    import open_clip
    from huggingface_hub import snapshot_download
    p = laion_path(models)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(LAION_URL, timeout=60).read()))
        name = [n for n in z.namelist() if n.endswith(".pth")][0]
        with open(p, "wb") as f:
            f.write(z.read(name))
    import torch
    m = laion_mlp()
    m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    print("laion detector ok", os.path.getsize(p), "bytes")
    print("sexy5 at", snapshot_download(SEXY5, allow_patterns=["config.json", "preprocessor_config.json",
                                                              "model.safetensors"]))
    open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
    import nudenet
    print("nudenet", os.path.dirname(nudenet.__file__), [f for f in os.listdir(os.path.dirname(nudenet.__file__))
                                                         if f.endswith(".onnx")])
    print("weights ready")


def nude_detector(prov):
    """NudeNet's own constructor opens its session without providers, which onnxruntime-gpu refuses."""
    import nudenet
    import onnxruntime as ort
    det = nudenet.NudeDetector.__new__(nudenet.NudeDetector)
    det.onnx_session = ort.InferenceSession(os.path.join(os.path.dirname(nudenet.__file__), "320n.onnx"),
                                            providers=prov)
    det.input_width = det.input_height = 320
    det.input_name = det.onnx_session.get_inputs()[0].name
    return det


def age_of(sess, rgb, fb):
    """buffalo_l genderage as InsightFace feeds it: square of 1.5 x the face box, 96 px, RGB 0..255."""
    fx, fy, fw, fh = fb
    sc = 96.0 / (max(fw, fh) * 1.5)
    M = np.array([[sc, 0, 48 - (fx + fw / 2) * sc], [0, sc, 48 - (fy + fh / 2) * sc]], np.float32)
    a = cv2.warpAffine(rgb, M, (96, 96), borderValue=0.0)
    out = sess.run(None, {sess.get_inputs()[0].name: a.astype(np.float32).transpose(2, 0, 1)[None]})[0][0]
    return float(out[2] * 100.0), int(np.argmax(out[:2]))


def score(a):
    import onnxruntime as ort
    import open_clip
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R = a.root
    pool = json.load(open(os.path.join(R, "split/suggest_pool.json")))
    oai, _, pre = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
    oai = oai.eval().to(dev)
    tok = open_clip.get_tokenizer("ViT-L-14-quickgelu")
    mlp = laion_mlp()
    mlp.load_state_dict(torch.load(laion_path(a.models), map_location="cpu", weights_only=True))
    mlp = mlp.eval().to(dev)
    proc = AutoImageProcessor.from_pretrained(SEXY5)
    vit = AutoModelForImageClassification.from_pretrained(SEXY5).eval().to(dev)
    labels = [vit.config.id2label[i] for i in range(vit.config.num_labels)]
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if dev == "cuda" else ["CPUExecutionProvider"]
    det = nude_detector(prov)
    ages = ort.InferenceSession(os.path.join(a.models, "insightface/buffalo_l/genderage.onnx"), providers=prov)
    with torch.no_grad():
        T = torch.stack([torch.nn.functional.normalize(
            torch.nn.functional.normalize(oai.encode_text(tok(p).to(dev)).float(), dim=-1).mean(0), dim=-1)
            for p in (ZS_POS, ZS_NEG)])
    out = {k: [] for k in ("user", "code", "tier", "laion_nsfw", "zs_suggest", "bare", "skin", "clothing",
                           "person_of_crop", "age", "male", "nn_area", "nn_n", "nn_explicit", "nn_json",
                           "oai", "n_people", "instance")}
    out.update({"p_" + l: [] for l in labels})
    users = sorted(pool)[:a.limit] if a.limit else sorted(pool)
    for ui, u in enumerate(users):
        f = os.path.join(R, "cut", u, "owner.json")
        if not os.path.exists(f):
            continue
        rec = json.load(open(f))
        if not rec.get("owner") or (rec.get("n_owner_photos") or 0) < 4:
            continue
        pdir = os.path.join(R, "photos", u)
        files = {os.path.splitext(x)[0]: x for x in os.listdir(pdir) if x.endswith((".jpg", ".webp", ".png"))}
        ims, keep = [], []
        for ph in rec["photos"]:
            rgb = load_photo(os.path.join(pdir, files[ph["code"]])) if ph["code"] in files else None
            if rgb is None:
                continue
            x0, y0, x1, y1 = ph["crop"]
            crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
            age, male = age_of(ages, rgb, ph["face_box"])
            dets = [d for d in det.detect(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)) if d["score"] >= NN_MIN]
            area = float(crop.shape[0] * crop.shape[1])
            ex = [d for d in dets if d["class"] in NN_EXPOSED]
            fr = ph["frac"]
            out["user"].append(u); out["code"].append(ph["code"]); out["tier"].append(pool[u])
            out["bare"].append(fr["skin"] / max(1e-6, fr["skin"] + fr["clothing"]))
            out["skin"].append(fr["skin"]); out["clothing"].append(fr["clothing"])
            out["person_of_crop"].append(fr["person_of_crop"])
            out["age"].append(age); out["male"].append(male)
            out["nn_area"].append(sum(d["box"][2] * d["box"][3] for d in ex) / area)
            out["nn_n"].append(len(ex))
            out["nn_explicit"].append(max([d["score"] for d in dets if d["class"] in NN_EXPLICIT] or [0.0]))
            out["nn_json"].append(json.dumps([[d["class"], round(float(d["score"]), 2)] for d in dets]))
            out["n_people"].append(ph["n_people"]); out["instance"].append(bool(ph["instance"]))
            ims.append(Image.fromarray(crop)); keep.append(ph["code"])
        if not ims:
            continue
        with torch.no_grad():
            o = torch.nn.functional.normalize(
                oai.encode_image(torch.stack([pre(im) for im in ims]).to(dev)).float(), dim=-1)
            out["laion_nsfw"] += mlp(o).cpu().tolist()
            out["zs_suggest"] += (100.0 * o @ T.T).softmax(-1)[:, 0].cpu().tolist()
            out["oai"].append(o.cpu().numpy().astype(np.float16))
            pr = vit(**{k: v.to(dev) for k, v in proc(images=ims, return_tensors="pt").items()}).logits.softmax(-1)
            for j, l in enumerate(labels):
                out["p_" + l] += pr[:, j].cpu().tolist()
        if ui % 25 == 0:
            print("%d/%d accounts, %d photos" % (ui + 1, len(users), len(out["user"])), flush=True)
    res = {k: (np.concatenate(v) if k == "oai" else np.array(v)) for k, v in out.items()}
    np.savez_compressed(os.path.join(R, "results/suggest_scores.npz"), **res)
    print("saved %d photos of %d accounts" % (len(res["user"]), len(set(res["user"].tolist()))))


def pct_rank(x):
    from scipy.stats import rankdata
    return (rankdata(x) - 0.5) / len(x)


def icc1(x, groups):
    """Share of a score's variance that lies between people (one-way ANOVA ICC)."""
    g = {}
    for v, k in zip(x, groups):
        g.setdefault(k, []).append(v)
    ks = [np.array(v) for v in g.values() if len(v) > 1]
    n = sum(len(v) for v in ks); m = np.concatenate(ks).mean()
    msb = sum(len(v) * (v.mean() - m) ** 2 for v in ks) / (len(ks) - 1)
    msw = sum(((v - v.mean()) ** 2).sum() for v in ks) / (n - len(ks))
    k0 = (n - sum(len(v) ** 2 for v in ks) / n) / (len(ks) - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def summary(a):
    from scipy.stats import spearmanr
    R = a.root
    d = dict(np.load(os.path.join(R, "results/suggest_scores.npz"), allow_pickle=False))
    users = d["user"]
    med_age = {u: float(np.median(d["age"][users == u])) for u in np.unique(users)}
    young = sorted(u for u, v in med_age.items() if v < a.age_min)
    ok = ~np.isin(users, young)
    S = {"n_accounts_scored": int(len(med_age)), "n_photos_scored": int(len(users)),
         "age_min": a.age_min, "n_accounts_set_aside_young": len(young),
         "median_apparent_age_quantiles": {str(q): round(float(np.percentile(list(med_age.values()), q)), 1)
                                           for q in (0, 5, 25, 50, 75, 95, 100)},
         "accounts_under": {str(t): int(sum(v < t for v in med_age.values())) for t in (18, 21, 23, 25)}}
    u = users[ok]
    X = {k: d[k][ok].astype(float) for k in ("laion_nsfw", "zs_suggest", "bare", "nn_area", "nn_n", "nn_explicit",
                                             "p_sexy", "p_porn", "p_neutral")}
    X["p_sexyporn"] = X["p_sexy"] + X["p_porn"]
    comb = np.mean([pct_rank(X[k]) for k in COMBINED], 0)
    X["combined"] = comb
    S["n_accounts"], S["n_photos"] = int(len(set(u.tolist()))), int(len(u))
    S["women_share_of_photos"] = round(float(1 - d["male"][ok].mean()), 3)
    S["scores"] = {}
    for k, x in X.items():
        S["scores"][k] = {"quantiles": {str(q): round(float(np.percentile(x, q)), 4) for q in (5, 25, 50, 75, 90, 95, 99, 100)},
                          "share_over_0.5": round(float((x > 0.5).mean()), 4),
                          "share_exactly_0": round(float((x == 0).mean()), 4),
                          "between_person_share": round(icc1(x, u), 3)}
    keys = COMBINED + ["nn_explicit"]
    S["spearman"] = {k1: {k2: round(float(spearmanr(X[k1], X[k2])[0]), 3) for k2 in keys} for k1 in keys}
    # per person: how high does their most suggestive saved photo reach, and how wide is their own spread
    hi = {q: float(np.percentile(comb, q)) for q in (75, 90)}
    per = {}
    for k in np.unique(u):
        c = comb[u == k]
        per[k] = {"n": int(len(c)), "max": float(c.max()), "min": float(c.min()), "spread": float(c.max() - c.min())}
    S["people"] = {
        "n": len(per),
        "no_photo_in_top_quarter": int(sum(v["max"] < hi[75] for v in per.values())),
        "no_photo_in_top_tenth": int(sum(v["max"] < hi[90] for v in per.values())),
        "all_photos_in_top_quarter": int(sum(v["min"] >= hi[75] for v in per.values())),
        "own_spread_quantiles": {str(q): round(float(np.percentile([v["spread"] for v in per.values()], q)), 3)
                                 for q in (10, 25, 50, 75, 90)},
        "photos_in_top_tenth_come_from_n_people": int(len(set(u[comb >= hi[90]].tolist()))),
    }
    S["explicit_flags"] = {"nudenet_explicit_part_over_0.5": int((X["nn_explicit"] > 0.5).sum()),
                           "p_porn_over_0.5": int((X["p_porn"] > 0.5).sum()),
                           "laion_over_0.9": int((X["laion_nsfw"] > 0.9).sum())}
    json.dump(S, open(os.path.join(R, "results/suggest_range.json"), "w"), indent=1)
    json.dump({"young": young, "per_person": per}, open(os.path.join(R, "results/suggest_people.json"), "w"))
    print(json.dumps(S, indent=1))
    # sheet: eight photos (different people) at six levels of the combined score
    rng = np.random.default_rng(20261116)
    codes = d["code"][ok]
    rows = []
    for lo, hi_ in SHEET_BANDS:
        idx = np.where((comb >= np.percentile(comb, lo)) & (comb <= np.percentile(comb, hi_)))[0]
        rng.shuffle(idx)
        seen, pick = set(), []
        for i in idx:
            if u[i] not in seen:
                seen.add(u[i]); pick.append(i)
            if len(pick) == 8:
                break
        tiles = []
        for i in pick:
            rec = {p["code"]: p for p in json.load(open(os.path.join(R, "cut", u[i], "owner.json")))["photos"]}[codes[i]]
            pdir = os.path.join(R, "photos", u[i])
            fn = [x for x in os.listdir(pdir) if os.path.splitext(x)[0] == codes[i]][0]
            rgb = load_photo(os.path.join(pdir, fn))
            x0, y0, x1, y1 = rec["crop"]
            c = rgb[y0:y1, x0:x1]
            h = 200; w = max(1, round(c.shape[1] * h / c.shape[0]))
            t = cv2.resize(c, (min(w, 180), h), interpolation=cv2.INTER_AREA)
            pad = np.full((h, 184, 3), 128, np.uint8); pad[:, :t.shape[1]] = t
            tiles.append(pad)
        while len(tiles) < 8:
            tiles.append(np.full((200, 184, 3), 128, np.uint8))
        row = np.concatenate(tiles, 1)
        bar = np.full((22, row.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, "combined score, percentile %d-%d" % (lo, hi_), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        rows += [bar, row]
    cv2.imwrite(os.path.join(R, "results/suggest_sheet.jpg"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--models", default=None)
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--age-min", type=float, default=21.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N accounts, no summary (a quick check)")
    a = ap.parse_args()
    a.models = a.models or os.path.join(a.root, "models")
    if a.prefetch:
        return prefetch(a.models)
    if not a.summary_only:
        score(a)
    if not a.limit:
        summary(a)


if __name__ == "__main__":
    main()
