#!/usr/bin/env python3
"""Find each account's owner and cut them out of their photos (GPU).

For every account folder PHOTOS/<user>/ written by fetch_photos.py:
 1. faces: YuNet + SFace (idscore.faceid, OpenCV model zoo) on every photo, all faces
 2. owner: the face that most photos agree on (SFace cosine >= 0.363, the agreement idea of
    tools/select_frames.py); at least MIN_PHOTOS photos must show it, else the account is dropped
 3. person: Mask R-CNN v2 instance masks and Keypoint R-CNN keypoints (torchvision weights, BSD);
    the owner's instance is the one that holds the owner's face
 4. parts: the SegFormer clothes parser of idscore.parsing on the owner's crop
 5. writes OUT/<user>/<code>_{cut,nocloth,bg,cloth,face,thumb}.jpg, OUT/<user>/owner.json and
    OUT/<user>/sface.npz (owner face embeddings)

Variants (what each later model sees):
  cut      owner only, background flat gray (128)          -> the appearance model, rating page
  nocloth  as cut, clothing flat dark gray (80)             -> appearance with fashion removed
  face     square crop around the owner's face
  bg       whole photo with every person blanked             -> scenery-only control
  cloth    only the owner's clothing pixels                  -> clothing-only control
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # code/ holds idscore
from idscore.faceid import FaceID  # noqa: E402
from idscore.parsing import ATR_LABELS, CLOTHING, FACE, HAIR, MODEL_ID, SKIN_LIMBS  # noqa: E402

ACCEPT = 0.363      # SFace "same person"
MIN_PHOTOS = 4      # photos that must show the owner's face
MAX_SIDE = 1600     # photos are processed at most this large
OUT_SIDE = 448      # saved variants
THUMB_SIDE = 360    # rating-page thumbnails
GRAY, DARK = 128, 80


def load(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    f = min(1.0, MAX_SIDE / max(h, w))
    if f < 1:
        bgr = cv2.resize(bgr, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save(path, rgb, side):
    h, w = rgb.shape[:2]
    f = min(1.0, side / max(h, w))
    if f < 1:
        rgb = cv2.resize(rgb, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def all_faces(fid, rgb):
    """Every face YuNet finds (rows: x, y, w, h, 5 landmarks, score), on an image upscaled so
    small faces in group photos still count."""
    H, W = rgb.shape[:2]
    f = max(1.0, 960 / max(H, W))
    img = rgb if f == 1 else cv2.resize(rgb, (round(W * f), round(H * f)), interpolation=cv2.INTER_CUBIC)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    fid.det.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, faces = fid.det.detect(bgr)
    if faces is None:
        return []
    out = []
    for r in faces:
        r = r.copy()
        r[:14] /= f
        eye = float(np.linalg.norm(r[4:6] - r[6:8]))
        if r[14] >= 0.8 and eye >= 12:
            out.append(r)
    return out


class Parser:
    """SegFormer clothes parser on the GPU (same model and labels as idscore.parsing)."""

    def __init__(self, cache_dir, dev):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        self.proc = SegformerImageProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir, local_files_only=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_ID, cache_dir=cache_dir, local_files_only=True).eval().to(dev)
        self.dev = dev

    @torch.no_grad()
    def labels(self, rgb, target_h=768):
        h, w = rgb.shape[:2]
        s = target_h / h
        im = cv2.resize(rgb, (max(32, round(w * s)), target_h), interpolation=cv2.INTER_CUBIC)
        inp = self.proc(images=im, return_tensors="pt", do_resize=False)
        logits = self.model(pixel_values=inp["pixel_values"].to(self.dev)).logits
        logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


class People:
    """Mask R-CNN v2 person masks + Keypoint R-CNN keypoints."""

    def __init__(self, wdir, dev):
        tv = torchvision.models.detection
        self.mask = tv.maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        self.mask.load_state_dict(torch.load(os.path.join(wdir, "maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"),
                                             map_location="cpu", weights_only=True))
        self.kp = tv.keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        self.kp.load_state_dict(torch.load(os.path.join(wdir, "keypointrcnn_resnet50_fpn_coco-fc266e95.pth"),
                                           map_location="cpu", weights_only=True))
        self.mask.eval().to(dev)
        self.kp.eval().to(dev)
        self.dev = dev

    @torch.no_grad()
    def __call__(self, rgb):
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.dev)
        m = self.mask([x])[0]
        keep = (m["labels"] == 1) & (m["scores"] >= 0.7)
        masks = (m["masks"][keep, 0] > 0.5).cpu().numpy()
        boxes = m["boxes"][keep].cpu().numpy()
        k = self.kp([x])[0]
        kk = k["scores"] >= 0.7
        kboxes = k["boxes"][kk].cpu().numpy()
        kps = torch.cat([k["keypoints"][kk][..., :2], k["keypoints_scores"][kk][..., None]], -1).cpu().numpy()
        return masks, boxes, kboxes, kps


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def find_owner(per_photo):
    """per_photo: list of lists of (face_row, emb). Returns (template or None, n_agreeing)."""
    faces = [(i, j, e) for i, fs in enumerate(per_photo) for j, (_, e) in enumerate(fs)]
    if not faces:
        return None, 0
    E = np.stack([e for _, _, e in faces])
    S = E @ E.T
    photo = np.array([i for i, _, _ in faces])
    agree = []
    for a in range(len(faces)):
        others = {photo[b] for b in np.nonzero(S[a] >= ACCEPT)[0] if photo[b] != photo[a]}
        agree.append(len(others))
    seed = int(np.argmax(agree))
    members = {}
    for b in np.nonzero(S[seed] >= ACCEPT)[0]:  # best agreeing face per photo
        if photo[b] not in members or S[seed, b] > S[seed, members[photo[b]]]:
            members[photo[b]] = b
    t = E[list(members.values())].mean(0)
    return t / np.linalg.norm(t), len(members)


def process_account(user, pdir, odir, fid, people, parser):
    files = {os.path.splitext(f)[0]: f for f in os.listdir(pdir) if f.endswith((".jpg", ".webp", ".png"))}
    codes = sorted(files)
    imgs, per_photo = {}, []
    for c in codes:
        rgb = load(os.path.join(pdir, files[c]))
        imgs[c] = rgb
        fs = []
        if rgb is not None:
            for r in all_faces(fid, rgb):
                e, _ = fid.embed(rgb, face_row=r)
                if e is not None:
                    fs.append((r, e))
        per_photo.append(fs)
    template, n_agree = find_owner(per_photo)
    rec = {"user": user, "n_photos": len(codes), "n_agree": n_agree, "photos": []}
    if template is None or n_agree < MIN_PHOTOS:
        rec["owner"] = False
        rec["reason"] = "fewer than %d photos agree on one face" % MIN_PHOTOS
        return rec
    rec["owner"] = True
    os.makedirs(odir, exist_ok=True)
    embs = {}
    for c, fs in zip(codes, per_photo):
        rgb = imgs[c]
        if rgb is None or not fs:
            continue
        sims = [float(e @ template) for _, e in fs]
        j = int(np.argmax(sims))
        if sims[j] < ACCEPT:
            continue
        row, emb = fs[j]
        embs[c] = emb
        H, W = rgb.shape[:2]
        fx, fy, fw, fh = [float(v) for v in row[:4]]
        fc = (fx + fw / 2, fy + fh / 2)
        masks, boxes, kboxes, kps = people(rgb)
        # the owner's instance: holds the face centre; ties -> largest
        cand = [i for i in range(len(boxes))
                if 0 <= int(fc[1]) < H and 0 <= int(fc[0]) < W and masks[i][int(fc[1]), int(fc[0])]]
        if not cand:
            cand = [i for i in range(len(boxes)) if boxes[i][0] <= fc[0] <= boxes[i][2] and boxes[i][1] <= fc[1] <= boxes[i][3]]
        inst = max(cand, key=lambda i: masks[i].sum()) if cand else None
        if inst is not None:
            box = boxes[inst]
            m = 0.06 * max(box[2] - box[0], box[3] - box[1])
            x0, y0 = int(max(0, box[0] - m)), int(max(0, box[1] - m))
            x1, y1 = int(min(W, box[2] + m)), int(min(H, box[3] + m))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            own = cv2.dilate(masks[inst].astype(np.uint8), k).astype(bool)
            kp = None
            if len(kboxes):
                ki = int(np.argmax([iou(box, kb) for kb in kboxes]))
                if iou(box, kboxes[ki]) > 0.5:
                    kp = kps[ki].round(1).tolist()
        else:  # no person instance (tight close-up): the whole photo
            x0, y0, x1, y1 = 0, 0, W, H
            own = np.ones((H, W), bool)
            kp = None
        crop = rgb[y0:y1, x0:x1]
        lab = parser.labels(crop)
        own_c = own[y0:y1, x0:x1]
        person = (lab != 0) & own_c
        cloth = person & np.isin(lab, CLOTHING)
        cut = np.where(person[..., None], crop, GRAY).astype(np.uint8)
        nocloth = cut.copy()
        nocloth[cloth] = DARK
        cl = np.where(cloth[..., None], crop, GRAY).astype(np.uint8)
        anyone = own.copy()
        for i in range(len(masks)):
            anyone |= masks[i]
        bg = np.where(anyone[..., None], GRAY, rgb).astype(np.uint8)
        side = 1.1 * max(fw, fh)
        cx, cy = fc
        fx0, fy0 = int(max(0, cx - side)), int(max(0, cy - side))
        fx1, fy1 = int(min(W, cx + side)), int(min(H, cy + side))
        face = rgb[fy0:fy1, fx0:fx1]
        save(os.path.join(odir, c + "_cut.jpg"), cut, OUT_SIDE)
        save(os.path.join(odir, c + "_nocloth.jpg"), nocloth, OUT_SIDE)
        save(os.path.join(odir, c + "_cloth.jpg"), cl, OUT_SIDE)
        save(os.path.join(odir, c + "_bg.jpg"), bg, OUT_SIDE)
        save(os.path.join(odir, c + "_face.jpg"), face, 224)
        save(os.path.join(odir, c + "_thumb.jpg"), cut, THUMB_SIDE)

        def lab_median(mask):
            if mask.sum() < 200:
                return None
            labc = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)[mask].astype(np.float32)
            return [round(float(v), 1) for v in np.median(labc, 0)]

        n = max(1, int(person.sum()))
        rec["photos"].append({
            "code": c, "size": [W, H], "owner_cos": round(sims[j], 3), "n_faces": len(fs),
            "n_people": int(len(boxes)), "face_box": [round(v, 1) for v in (fx, fy, fw, fh)],
            "eye_px": round(float(np.linalg.norm(row[4:6] - row[6:8])), 1),
            "crop": [x0, y0, x1, y1], "instance": inst is not None, "keypoints": kp,
            "frac": {"face": round(float(np.isin(lab, FACE)[person].sum() / n), 3),
                     "hair": round(float(np.isin(lab, HAIR)[person].sum() / n), 3),
                     "skin": round(float(np.isin(lab, SKIN_LIMBS)[person].sum() / n), 3),
                     "clothing": round(float(cloth.sum() / n), 3),
                     "person_of_crop": round(float(person.mean()), 3)},
            "hair_lab": lab_median(person & np.isin(lab, HAIR)),
            "skin_lab": lab_median(person & np.isin(lab, FACE + SKIN_LIMBS)),
        })
    rec["n_owner_photos"] = len(rec["photos"])
    if embs:
        np.savez(os.path.join(odir, "sface.npz"), codes=np.array(list(embs)), emb=np.stack(list(embs.values())))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fid = FaceID(os.path.join(a.models, "opencv_zoo"))
    people = People(os.path.join(a.models, "torchvision"), dev)
    parser = Parser(os.path.join(a.models, "hf"), dev)
    users = sorted(u for u in os.listdir(a.photos) if os.path.isdir(os.path.join(a.photos, u)))
    summary_path = os.path.join(a.out, "owners.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) and not a.force else {}
    t0 = time.time()
    for i, u in enumerate(users):
        if u in summary:
            continue
        rec = process_account(u, os.path.join(a.photos, u), os.path.join(a.out, u), fid, people, parser)
        if rec["owner"]:
            json.dump(rec, open(os.path.join(a.out, u, "owner.json"), "w"))
        summary[u] = {k: rec.get(k) for k in ("owner", "n_photos", "n_agree", "n_owner_photos", "reason")}
        json.dump(summary, open(summary_path, "w"), indent=0)
        print("%d/%d %s owner=%s photos=%s (%.0fs)" % (i + 1, len(users), u, rec["owner"], rec.get("n_owner_photos"),
                                                     time.time() - t0), flush=True)
    kept = sum(1 for v in summary.values() if v["owner"] and v["n_owner_photos"] >= MIN_PHOTOS)
    print("accounts %d, with an owner on >= %d photos: %d" % (len(summary), MIN_PHOTOS, kept))


if __name__ == "__main__":
    main()
