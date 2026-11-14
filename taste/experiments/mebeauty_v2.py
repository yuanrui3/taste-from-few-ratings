#!/usr/bin/env python3
"""MEBeauty v2.1 dating-preference raters (PROTOCOL.md, "Generalizing ...", G3).

  mebeauty_v2.py embed ROOT   CLIP L-14 (laion2b_s32b_b82k, as me.npz) and SFace (YuNet + SFace, as
                              mebeauty_sface.py) codes of the 2,462 faces -> datasets/mebeauty/v2/codes.npz
  mebeauty_v2.py eval ROOT    G3 -> results/mebeauty_v2_g3.json

G3 is G1 with the dating-preference raters (631 raters, ratings of both sexes and six ethnicities), on
one random half of the faces (B). The other half (A) trains "M": a crowd that has seen Black faces, the
mean of the standardized SCUT crowd and a MEBeauty crowd (ridge on A to the mean of the OTHER raters).
"""
import glob
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]
V2 = "datasets/mebeauty/v2/personalized_date"


def read_rows(root, cols):
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(os.path.join(root, V2, "*.parquet"))):
        rows += pq.read_table(f, columns=cols).to_pylist()
    return rows


def embed(root):
    import open_clip
    import torch
    from PIL import Image
    from idscore.faceid import FaceID
    rows = read_rows(root, ["image_id", "image"])
    ims = [Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB") for r in rows]
    print("faces", len(ims), "first size", ims[0].size, flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, _, tf = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k"); m = m.eval().to(dev).half()
    clip = []
    for i in range(0, len(ims), 64):
        with torch.no_grad():
            e = m.encode_image(torch.stack([tf(x) for x in ims[i:i + 64]]).to(dev).half()).float()
        clip.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    fid = FaceID(os.path.join(root, "models", "opencv_zoo")); sf = np.full((len(ims), 128), np.nan); found = 0
    for j, im in enumerate(ims):
        rgb = np.array(im); row = fid.detect(rgb); found += row is not None
        e, _ = fid.embed(rgb, row) if row is not None else fid.embed(rgb, None, (0, 0, rgb.shape[1], rgb.shape[0]))
        if e is not None:
            sf[j] = e
    np.savez_compressed(os.path.join(root, "datasets/mebeauty/v2/codes.npz"), image_id=np.array([r["image_id"] for r in rows]),
                        clip=np.concatenate(clip), sface=sf)
    print("saved; SFace face found in %d of %d" % (found, len(ims)), flush=True)


def evaluate(root):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from group_transfer import (CROWD_ALPHAS, erase, group_basis, l2, measures, run_rater, scut_crowd_models,
                                std_cols, zs)
    rows = read_rows(root, ["image_id", "gender", "ethnicity", "ratings"])
    C = np.load(os.path.join(root, "datasets/mebeauty/v2/codes.npz"), allow_pickle=True)
    pos = {str(k): i for i, k in enumerate(C["image_id"])}; order = [pos[r["image_id"]] for r in rows]
    clip = l2(C["clip"][order].astype(np.float64)); sfr = C["sface"][order]
    ok = ~np.isnan(sfr).any(1); sf = l2(np.where(ok[:, None], sfr, np.nanmean(sfr, 0)))
    sex = np.array([r["gender"] for r in rows]); eth = np.array([r["ethnicity"] for r in rows]); n = len(rows)
    lab = np.where((sex == "female") & np.isin(eth, ["caucasian", "asian"]), "familiar",
                   np.where(eth == "black", np.where(sex == "male", "black men", "black women"),
                            np.where(sex == "male", "other men", "other women")))
    groups = np.array([a + "/" + b for a, b in zip(sex, eth)])
    bases = (group_basis(clip, groups), group_basis(sf, groups))
    X0 = std_cols(np.c_[clip, sf]); Xe = std_cols(np.c_[erase(clip, bases[0]), erase(sf, bases[1])])
    scut = scut_crowd_models(root)(clip, sf)
    raters = {}
    for i, r in enumerate(rows):
        for x in r["ratings"]:
            raters.setdefault(x["rater_id"], {})[i] = float(x["score"])
    tot = np.zeros(n); cnt = np.zeros(n)
    for v in raters.values():
        for i, s in v.items():
            tot[i] += s; cnt[i] += 1
    rng = np.random.default_rng(7); inA = np.zeros(n, bool); inA[rng.permutation(n)[: n // 2]] = True
    A = np.where(inA)[0]; B = np.where(~inA)[0]
    keep = []
    for rid, v in raters.items():
        idx = np.array([i for i in v if not inA[i]])
        if len(idx) == 0:
            continue
        L = lab[idx]
        if (L == "familiar").sum() >= 40 and np.isin(L, ["black women", "black men"]).sum() >= 15 and np.isin(L, ["black men", "other men"]).sum() >= 40:
            keep.append(rid)
    print("G3 raters kept: %d of %d" % (len(keep), len(raters)), flush=True)
    # MEBeauty crowd per kept rater: ridge on A to the mean of the other raters, applied to B (one multi-target fit)
    T = np.zeros((len(A), len(keep)))
    for k, rid in enumerate(keep):
        own = np.array([raters[rid].get(i, np.nan) for i in A]); has = ~np.isnan(own)
        T[:, k] = (tot[A] - np.where(has, own, 0)) / np.maximum(1, cnt[A] - has)
    s1 = StandardScaler().fit(clip[A]); m1 = RidgeCV(alphas=CROWD_ALPHAS, alpha_per_target=True).fit(s1.transform(clip[A]), T)
    s2 = StandardScaler().fit(sf[A]); m2 = RidgeCV(alphas=CROWD_ALPHAS, alpha_per_target=True).fit(s2.transform(sf[A]), T)
    P1 = m1.predict(s1.transform(clip[B])); P2 = m2.predict(s2.transform(sf[B]))
    within = {"familiar": ["familiar"], "black": ["black women", "black men"], "black women": ["black women"],
              "men": ["black men", "other men"], "other women": ["other women"]}
    place = ["black women", "black men", "other men", "other women"]
    names = ["current", "E", "U", "E+U", "M", "M+U", "crowd only", "crowd M only", "ratings only"]
    res_rows = []
    for k, rid in enumerate(keep):
        v = raters[rid]; y = np.full(n, np.nan)
        for i, s in v.items():
            y[i] = s
        rated_B = np.array([i for i in B if not np.isnan(y[i])])
        fam = rated_B[lab[rated_B] == "familiar"]; unf = rated_B[lab[rated_B] != "familiar"]
        cM = np.zeros(n); cM[B] = zs(zs(scut[B]) + zs(zs(P1[:, k]) + zs(P2[:, k])))
        mixed, s0 = run_rater({False: X0, True: Xe}, scut, y, fam, unf, seed=k)
        _, sM = run_rater({False: X0, True: X0}, cM, y, fam, unf, seed=k)
        sc = {m: s0[m] for m in ["current", "E", "U", "E+U", "crowd only", "ratings only"]}
        sc.update({"M": sM["current"], "M+U": sM["U"], "crowd M only": sM["crowd only"]})
        res_rows.append({m: measures(sc[m], y[mixed], lab[mixed], within, place) for m in names})
    keys = ["mixed", "familiar", "black", "black women", "men", "other women", "placement"]
    out = {}
    print("G3 MEBeauty dating raters: %d" % len(res_rows))
    print("  %-13s" % "method" + "".join("%12s" % k for k in keys) + "   wins")
    for m in names:
        mean = {k: float(np.nanmean([r[m][k] for r in res_rows])) for k in keys}
        wins = int(sum(r[m]["mixed"] > r["current"]["mixed"] for r in res_rows)); out[m] = dict(mean, wins=wins)
        print("  %-13s" % m + "".join("%12.3f" % mean[k] for k in keys) + "   %d/%d" % (wins, len(res_rows)))
    for m in ["E", "U", "E+U", "M", "M+U"]:
        g = out[m]["mixed"] - out["current"]["mixed"]; fd = out[m]["familiar"] - out["current"]["familiar"]
        ok = g >= 0.02 and out[m]["wins"] >= 2 * len(res_rows) / 3 and fd >= -0.01
        out[m]["verdict"] = "passes the simulation rule" if ok else "no"
        print("  %s: mixed %+.4f, wins %d/%d, familiar %+.3f -> %s" % (m, g, out[m]["wins"], len(res_rows), fd, out[m]["verdict"]))
    json.dump(out, open(os.path.join(root, "results/mebeauty_v2_g3.json"), "w"), indent=1)


if __name__ == "__main__":
    (embed if sys.argv[1] == "embed" else evaluate)(sys.argv[2] if len(sys.argv) > 2 else ".")
