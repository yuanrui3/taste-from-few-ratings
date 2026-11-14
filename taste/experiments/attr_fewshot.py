#!/usr/bin/env python3
"""Vision-only few-rating model on an attribute basis (see PROTOCOL.md "Vision-only few-rating:
attribute basis"). Training accounts only, the same 20 splits as probe_search.py; then one second
look at the used test set for the rule's pick.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from probe_search import herding, winner_features  # noqa: E402
from taste_explain import BODY, FACE, text_embed  # noqa: E402
from train_eval import load, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
ALPHAS = np.logspace(-1, 5, 25)
NOT_FOR_USER = {"large eyes", "blonde hair", "red hair", "freckles"}  # user: eye size, hair colour, freckles


def r(X, y):
    return RidgeCV(alphas=ALPHAS).fit(X, y)


def candidates(full_S, att_S, y, full_E, att_E):
    out = {}
    mf = r(full_S, y); pf = mf.predict(full_E); out["full codes only (reference)"] = pf
    ma = r(att_S, y); pa = ma.predict(att_E); out["attributes"] = pa
    res = y - ma.predict(att_S)
    out["attributes, then full codes on the rest"] = pa + r(full_S, res).predict(full_E)
    out["attributes x5 + full codes"] = r(np.c_[att_S * 5, full_S], y).predict(np.c_[att_E * 5, full_E])
    w = len(y) / (len(y) + 40)
    out["rank blend attributes/full"] = (1 - w) * rankdata(pa) / len(pa) + w * rankdata(pf) / len(pf)
    return out


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    users = sorted(set(split["train"]) | set(split["test"]))
    pos = {u: i for i, u in enumerate(users)}
    full = StandardScaler().fit_transform(winner_features(F, idx, users))
    Z32 = PCA(n_components=32, random_state=0).fit_transform(full)
    pairs = BODY + FACE
    T = text_embed(["a photo of a person with " + p for pair in pairs for p in pair])
    cols = []
    for k, (a, b) in enumerate(pairs):
        key = "clip_cut" if k < len(BODY) else "clip_face"
        s = F[key] @ (T[2 * k] - T[2 * k + 1])
        cols.append([s[idx[u]].mean() for u in users])
    A = StandardScaler().fit_transform(np.array(cols).T)
    keep_personal = [k for k, (a, b) in enumerate(pairs) if a not in NOT_FOR_USER]
    bases = {"main (31)": A, "personal (%d)" % len(keep_personal): A[:, keep_personal]}

    tr = np.array([pos[u] for u in train]); y = np.array([first[u] for u in train], float)
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid))
    res = {}
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); P, E = np.where(isP)[0], np.where(~isP)[0]
        order = herding(Z32[tr[P]], max(NS))
        for n in NS + ["all"]:
            nn = len(P) if n == "all" else n
            sels = {"herding": [order[:nn]] if n != "all" else [list(range(len(P)))],
                    "random": [list(np.random.default_rng(s * 100 + k).choice(len(P), nn, replace=False)) for k in range(5 if n != "all" else 1)]}
            for sel, lists in sels.items():
                for bname, B in bases.items():
                    acc = {}
                    for pk in lists:
                        pk = list(pk)
                        preds = candidates(full[tr[P]][pk], B[tr[P]][pk], y[P][pk], full[tr[E]], B[tr[E]])
                        for name, p in preds.items():
                            acc.setdefault(name, []).append(spearmanr(p, y[E]).correlation)
                    for name, v in acc.items():
                        key = "%s | %s | %s" % (sel, bname, name)
                        res.setdefault(key, {}).setdefault(str(n), []).append(float(np.nanmean(v)))
        print("split %d done" % s, flush=True)

    summary = {k: {n: float(np.mean(v)) for n, v in d.items()} for k, d in res.items()}
    avg = {k: float(np.mean([summary[k][str(n)] for n in (5, 10, 20, 40)])) for k in summary}
    main_keys = [k for k in summary if "main (31)" in k]
    best = max(main_keys, key=avg.get)
    ref = "herding | main (31) | full codes only (reference)"
    wins = int(np.sum(np.array(res[best]["20"]) > np.array(res[ref]["20"])))
    print("\n%-72s %s   avg(5-40)" % ("selection | basis | model", "  ".join("n=%-4s" % n for n in NS + ["all"])))
    for k in sorted(summary, key=avg.get, reverse=True):
        if "random" in k and "reference" not in k and "attributes" not in k:
            continue
        print("%-72s %s   %.3f" % (k, "  ".join("%.3f " % summary[k][str(n)] for n in NS + ["all"]), avg[k]))
    print("\nBEST (main basis):", best, "| beats the reference at n=20 in %d/20 splits ->" % wins,
          "improves" if wins >= 15 else "does NOT improve")

    sel, bname, name = best.split(" | ")
    B = bases[bname]
    te = np.array([pos[u] for u in test]); yte = np.array([first[u] for u in test], float)
    order = herding(Z32[tr], max(NS))
    second = {}
    for n in NS:
        lists = [order[:n]] if sel == "herding" else [list(np.random.default_rng(k).choice(len(tr), n, replace=False)) for k in range(50)]
        v = [spearmanr(candidates(full[tr][list(pk)], B[tr][list(pk)], y[list(pk)], full[te], B[te])[name], yte).correlation for pk in lists]
        second[n] = float(np.mean(v))
    print("second look at the used test set (%s): %s" % (best, ", ".join("n=%d %.3f" % (n, v) for n, v in second.items())))
    json.dump({"summary": summary, "avg_5_40": avg, "best": best, "wins_vs_reference_n20": wins,
               "second_look_test": second, "excluded_personal": sorted(NOT_FOR_USER)},
              open(os.path.join(root, "results/attr_fewshot.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
