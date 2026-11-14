#!/usr/bin/env python3
"""Prepare SCUT-FBP5500 (GPU): unzip, read every rater's scores, embed the faces with the
main model's OpenCLIP ViT-L-14 (laion2b). Writes datasets/scut/scut.npz:
names (5500), emb (5500 x 768), R (60 raters x 5500 images, NaN if missing), mean (5500).
The archive is treated as data only: nothing in it is executed.
"""
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_xlsx(path):
    """Minimal .xlsx reader (first sheet) with the standard library: returns rows of strings/numbers."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
    sheet = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))[0]
    rows = []
    for row in ET.fromstring(z.read(sheet)).iter("{%s}row" % NS["m"]):
        vals = {}
        for c in row.findall("m:c", NS):
            col = re.match(r"[A-Z]+", c.get("r")).group(0)
            v = c.find("m:v", NS)
            if v is None:
                continue
            vals[col] = shared[int(v.text)] if c.get("t") == "s" else float(v.text)
        rows.append(vals)
    return rows


def main(root):
    d = os.path.join(root, "datasets/scut")
    base = os.path.join(d, "SCUT-FBP5500_v2")
    if not os.path.isdir(base):
        zipfile.ZipFile(os.path.join(d, "scut.zip")).extractall(d)
    rows = read_xlsx(os.path.join(base, "All_Ratings.xlsx"))
    header = rows[0]; print("columns:", header, "rows:", len(rows) - 1, flush=True)
    cols = {v: k for k, v in header.items()}
    rcol = next(cols[k] for k in cols if str(k).lower().startswith("rater"))
    fcol = next(cols[k] for k in cols if str(k).lower().startswith("filename"))
    scol = next(cols[k] for k in cols if str(k).lower().startswith("rating"))
    names = sorted({r[fcol] for r in rows[1:] if fcol in r})
    raters = sorted({int(r[rcol]) for r in rows[1:] if rcol in r})
    ni = {n: i for i, n in enumerate(names)}; ri = {r: i for i, r in enumerate(raters)}
    R = np.full((len(raters), len(names)), np.nan)
    for r in rows[1:]:
        if rcol in r and fcol in r and scol in r:
            R[ri[int(r[rcol])], ni[r[fcol]]] = r[scol]
    print("raters %d, images %d, filled %.3f" % (len(raters), len(names), np.mean(~np.isnan(R))), flush=True)
    import open_clip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, _, tf = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    m = m.eval().to(dev).half()
    embs = []
    for i in range(0, len(names), 64):
        ims = [tf(Image.open(os.path.join(base, "Images", n)).convert("RGB")) for n in names[i:i + 64]]
        with torch.no_grad():
            e = m.encode_image(torch.stack(ims).to(dev).half()).float()
        embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    emb = np.concatenate(embs)
    np.savez_compressed(os.path.join(d, "scut.npz"), names=np.array(names), emb=emb, R=R, raters=np.array(raters),
                        mean=np.nanmean(R, 0))
    print("saved", emb.shape)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
