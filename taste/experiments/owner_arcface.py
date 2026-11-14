#!/usr/bin/env python3
"""Owner identity with ArcFace R50 (InsightFace buffalo_l) instead of SFace (PROTOCOL.md, "Owner identity
with a stronger face model").

  owner_arcface.py embed ROOT [I N]  every YuNet face (same detection as owner_cutouts.py) in every photo
                                     of photos/<user>/ -> arc/<user>.npz {code, row, emb}; chunk I of N
  owner_arcface.py eval ROOT         same-person line from cross-account owner faces (99.9th percentile),
                                     owner step rerun on ArcFace codes, comparison with the SFace owner
                                     step -> results/owner_arcface.json, arc/owners_arcface.json

ArcFace input: 5-point similarity alignment to the standard 112 x 112 template, RGB, (x - 127.5) / 127.5.
The owner rule is the one in owner_cutouts.find_owner with ArcFace's line.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

MAX_SIDE = 1600
MIN_PHOTOS = 4
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def load(path):  # as owner_cutouts.load, so face rows match owner.json
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]; f = min(1.0, MAX_SIDE / max(h, w))
    if f < 1:
        bgr = cv2.resize(bgr, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def all_faces(det, rgb):  # as owner_cutouts.all_faces
    H, W = rgb.shape[:2]; f = max(1.0, 960 / max(H, W))
    img = rgb if f == 1 else cv2.resize(rgb, (round(W * f), round(H * f)), interpolation=cv2.INTER_CUBIC)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    det.setInputSize((bgr.shape[1], bgr.shape[0])); _, faces = det.detect(bgr)
    out = []
    for r in (faces if faces is not None else []):
        r = r.copy(); r[:14] /= f
        if r[14] >= 0.8 and float(np.linalg.norm(r[4:6] - r[6:8])) >= 12:
            out.append(r)
    return out


def aligned(rgb, row):
    src = row[4:14].reshape(5, 2).astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, TEMPLATE, method=cv2.LMEDS)
    return cv2.warpAffine(rgb, M, (112, 112), borderValue=0.0)


def embed(root, i, n):
    try:
        import torch  # noqa: F401  loads the CUDA / cuDNN libraries onnxruntime-gpu needs
    except ImportError:
        pass
    import onnxruntime as ort
    det = cv2.FaceDetectorYN.create(os.path.join(root, "models/opencv_zoo/face_detection_yunet_2023mar.onnx"),
                                    "", (320, 320), 0.5, 0.3, 50)
    sess = ort.InferenceSession(os.path.join(root, "models/insightface/buffalo_l/w600k_r50.onnx"),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("onnxruntime providers:", sess.get_providers(), flush=True)
    name = sess.get_inputs()[0].name
    users = sorted(u for u in os.listdir(os.path.join(root, "photos")) if os.path.isdir(os.path.join(root, "photos", u)))
    os.makedirs(os.path.join(root, "arc"), exist_ok=True)
    for k, u in enumerate(users):
        if k % n != i:
            continue
        out = os.path.join(root, "arc", u + ".npz")
        if os.path.exists(out):
            continue
        codes, rows, crops = [], [], []
        pdir = os.path.join(root, "photos", u)
        for f in sorted(os.listdir(pdir)):
            if not f.endswith((".jpg", ".webp", ".png")):
                continue
            rgb = load(os.path.join(pdir, f))
            if rgb is None:
                continue
            for r in all_faces(det, rgb):
                codes.append(os.path.splitext(f)[0]); rows.append(r[:15]); crops.append(aligned(rgb, r))
        emb = np.zeros((0, 512), np.float32)
        if crops:
            x = ((np.stack(crops).astype(np.float32) - 127.5) / 127.5).transpose(0, 3, 1, 2)
            emb = np.concatenate([sess.run(None, {name: x[j:j + 64]})[0] for j in range(0, len(x), 64)])
            emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        np.savez(out, code=np.array(codes), row=np.array(rows, np.float32).reshape(-1, 15), emb=emb)
        if k % 40 == i:
            print("%d/%d accounts" % (k + 1, len(users)), flush=True)
    print("chunk %d of %d done" % (i, n), flush=True)


def find_owner(codes, E, line):
    """owner_cutouts.find_owner on ArcFace codes. Returns template or None, number of agreeing photos."""
    if len(E) == 0:
        return None, 0
    S = E @ E.T; photo = np.array(codes)
    agree = [len({photo[b] for b in np.nonzero(S[a] >= line)[0] if photo[b] != photo[a]}) for a in range(len(E))]
    seed = int(np.argmax(agree)); members = {}
    for b in np.nonzero(S[seed] >= line)[0]:
        if photo[b] not in members or S[seed, b] > S[seed, members[photo[b]]]:
            members[photo[b]] = b
    t = E[list(members.values())].mean(0)
    return t / np.linalg.norm(t), len(members)


def evaluate(root):
    ids = json.load(open(os.path.join(root, "split/rate_ids.json")))
    sp = os.path.join(root, "split/same_person.json")
    pairs = {frozenset(p[:2]) for p in json.load(open(sp))["pairs"]} if os.path.exists(sp) else set()
    A = {os.path.basename(f)[:-4]: dict(np.load(f)) for f in glob.glob(os.path.join(root, "arc", "*.npz"))}
    sface_kept, owner_face = {}, {}
    for u in A:
        p = os.path.join(root, "cut", u, "owner.json")
        if not os.path.exists(p):
            continue
        rec = json.load(open(p)); a = A[u]; kept = {}
        for ph in rec["photos"]:  # the ArcFace code of the face SFace chose: same detector, match by box
            m = np.nonzero(a["code"] == ph["code"])[0]
            if len(m):
                box = np.array(ph["face_box"]); j = m[np.argmin(np.abs(a["row"][m, :4] - box).sum(1))]
                if np.abs(a["row"][j, :4] - box).sum() < 4:
                    kept[ph["code"]] = j
        sface_kept[u] = kept
        if kept:
            t = a["emb"][list(kept.values())].mean(0); owner_face[u] = t / np.linalg.norm(t)
    # same-person line: cross-account cosines of SFace-kept owner faces, known same-person pairs left out
    users = sorted(sface_kept); rng = np.random.default_rng(0); cross = []
    for x in range(len(users)):
        Ex = A[users[x]]["emb"][list(sface_kept[users[x]].values())]
        for y in range(x + 1, len(users)):
            if frozenset((users[x], users[y])) in pairs or not len(Ex):
                continue
            Ey = A[users[y]]["emb"][list(sface_kept[users[y]].values())]
            if len(Ey):
                c = (Ex @ Ey.T).ravel(); cross.append(rng.choice(c, min(len(c), 20), replace=False))
    cross = np.concatenate(cross); line = float(np.quantile(cross, 0.999))
    within = np.concatenate([(lambda E: (E @ E.T)[np.triu_indices(len(E), 1)])(A[u]["emb"][list(k.values())]) for u, k in sface_kept.items() if len(k) > 1])
    print("ArcFace same-person line (99.9th pct of %d cross-account pairs): %.3f" % (len(cross), line))
    print("  SFace-kept pairs within a person: median %.3f, share below the line %.1f%%" % (np.median(within), 100 * (within < line).mean()))
    res = []; keeps = {}; tot = dict(people=0, owner_changed=0, dropped=0, rejected=0, added=0, kept_both=0)
    for u in sorted(A):
        a = A[u]; t, nag = find_owner(list(a["code"]), a["emb"], line)
        keep = {}
        if t is not None and nag >= MIN_PHOTOS:
            for c in sorted(set(a["code"])):
                m = np.nonzero(a["code"] == c)[0]; s = a["emb"][m] @ t; j = m[int(np.argmax(s))]
                if s.max() >= line:
                    keep[c] = int(j)
        keeps[u] = keep if len(keep) >= MIN_PHOTOS else {}
        sk = sface_kept.get(u)
        r = {"id": ids.get(u, "?"), "arc_owner": len(keep) >= MIN_PHOTOS, "arc_photos": len(keep)}
        if sk is not None:
            tot["people"] += 1
            changed = bool(keep) and float(owner_face[u] @ t) < line if u in owner_face and t is not None else False
            rej = [c for c in sk if c not in keep or keep[c] != sk[c]]; add = [c for c in keep if c not in sk]
            r.update(sface_photos=len(sk), owner_changed=changed, rejected=rej, added=add, dropped=len(keep) < MIN_PHOTOS)
            tot["owner_changed"] += changed; tot["dropped"] += len(keep) < MIN_PHOTOS
            tot["rejected"] += len(rej); tot["added"] += len(add); tot["kept_both"] += len(sk) - len(rej)
        res.append(r)
    S = [r for r in res if "sface_photos" in r]
    print("people kept by the SFace owner step: %d" % tot["people"])
    print("  owner is a different person under ArcFace: %d" % tot["owner_changed"])
    print("  fall below %d photos: %d" % (MIN_PHOTOS, tot["dropped"]))
    print("  photos kept by both %d; SFace-kept photos ArcFace rejects %d (people %d); photos ArcFace adds %d (people %d)" % (
        tot["kept_both"], tot["rejected"], sum(bool(r["rejected"]) for r in S), tot["added"], sum(bool(r["added"]) for r in S)))
    print("accounts the SFace step dropped that ArcFace keeps: %d" % sum(r["arc_owner"] for r in res if "sface_photos" not in r))
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    json.dump({"line": line, "totals": tot, "people": res}, open(os.path.join(root, "results/owner_arcface.json"), "w"), indent=1)
    json.dump({"line": line, "keep": keeps}, open(os.path.join(root, "arc", "owners_arcface.json"), "w"))  # face index per kept photo


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        embed(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0, int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    else:
        evaluate(sys.argv[2])
