#!/usr/bin/env python3
"""Sexual attractiveness, round 2: the person's earlier appearance rating plus the photo, tested on people
never shown in round 1 (docs/ADDENDUM-sexual-attractiveness.md, "Sexual attractiveness, round 2"; planned before picking).

  suggest_fresh.py pick ROUND1_RATINGS [--root R] [--site R/suggest_site2]   (GPU job: zero-shot age screen)
      fresh people and photos -> SITE/img/<id>.jpg, SITE/items.json, split/suggest_set2.json (private map);
      fits the models on round 1 and freezes predictions -> results/suggest_fresh_pred.json (opaque ids; SHA-256)
  suggest_fresh.py eval ROUND2_RATINGS [--root R]  -> results/suggest_fresh_eval.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from suggest_fit import (ALPHAS, SEEDS, STANDIN, boot_people, ci, folds, pct_rank, prep,  # noqa: E402
                         read_ratings, rho, ridge_all)

SEED = 202611162
PER_PERSON, N_REPEAT, GAP = 4, 12, 30
YOUNG_PHOTO = 0.7
MAX_SIDE = 1100


def inputs(R):
    d = dict(np.load(os.path.join(R, "results/suggest_scores.npz"), allow_pickle=False))
    pool = {k: pct_rank(d[k].astype(float)) for k in ("laion_nsfw", "zs_suggest", "bare", "nn_area")}
    pool["p_sexyporn"] = pct_rank(d["p_sexy"].astype(float) + d["p_porn"].astype(float))
    stand5 = np.c_[[pool[k] for k in STANDIN]].T
    F = np.load(os.path.join(R, "features.npz"), allow_pickle=True)
    M = np.load(os.path.join(R, "features_more_vith_siglip.npz"), allow_pickle=True)
    fi = {(str(u), str(c)): i for i, (u, c) in enumerate(zip(F["user"], F["code"]))}
    mi = {(str(u), str(c)): i for i, (u, c) in enumerate(zip(M["user"], M["code"]))}
    key = [(str(u), str(c)) for u, c in zip(d["user"], d["code"])]
    clip_cut = np.zeros((len(key), F["clip_cut"].shape[1]), np.float32)
    sig_cut = np.zeros((len(key), M["siglip_cut"].shape[1]), np.float32)
    have = np.array([k in fi and k in mi for k in key])
    for j in np.where(have)[0]:
        clip_cut[j] = F["clip_cut"][fi[key[j]]]
        sig_cut[j] = M["siglip_cut"][mi[key[j]]]
    return d, key, {"oai": d["oai"].astype(np.float64), "clip_cut": clip_cut.astype(np.float64),
                    "sig_cut": sig_cut.astype(np.float64), "stand5": stand5, "standin": stand5.mean(1), "have": have}


def fit_predict(blocks_tr, y, blocks_te, alpha):
    n_tr = len(y)
    allb = [np.vstack([a, b]) for a, b in zip(blocks_tr, blocks_te)]
    A, B = prep(allb, np.arange(n_tr), np.arange(n_tr, n_tr + len(blocks_te[0])))
    return ridge_all(A, y, B, [alpha])[0]


def ols_predict(Xtr, y, Xte):
    A = np.c_[np.ones(len(Xtr)), Xtr]
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    return np.c_[np.ones(len(Xte)), Xte] @ w, w


def pick(a):
    from suggest_pack import young_prob
    R = a.root
    site = a.site or os.path.join(R, "suggest_site2")
    d, key, X = inputs(R)
    u = np.array([k[0] for k in key])
    code = np.array([k[1] for k in key])
    S1 = json.load(open(os.path.join(R, "split/suggest_set.json")))
    same = json.load(open(os.path.join(R, "split/same_person.json")))
    appearance = json.load(open(os.path.join(R, "split/ratings_effective.json")))["first"]
    group = {x: g[0] for g in same["groups"] for x in g}
    rated1 = {p["user"] for p in S1["photos"]}
    groups1 = {group.get(x, x) for x in rated1}
    py = young_prob(d["oai"])
    elig = (d["n_people"] <= 1) & (py < YOUNG_PHOTO) & X["have"]
    fresh = sorted(k for k in set(u) if k not in rated1 and k not in S1["set_aside"] and group.get(k, k) not in groups1
                   and k in appearance and (elig & (u == k)).any())
    rng = np.random.default_rng(SEED)
    rows = []
    for k in fresh:
        idx = np.where(elig & (u == k))[0]
        o = idx[np.argsort(X["standin"][idx], kind="stable")]
        take = [("low", o[0])] + ([("high", o[-1])] if len(o) > 1 else [])
        mid = list(o[1:-1])
        if mid:
            take += [("mid", int(i)) for i in sorted(rng.choice(mid, size=min(PER_PERSON - 2, len(mid)), replace=False))]
        for which, i in take:
            pid = hashlib.sha256(("%d:%s:%s" % (SEED, k, code[i])).encode()).hexdigest()[:12]
            rows.append({"id": pid, "user": str(k), "code": str(code[i]), "own": which, "row": int(i),
                         "standin": round(float(X["standin"][i]), 4)})
    assert len({r["id"] for r in rows}) == len(rows)
    grp = {r["id"]: hashlib.sha256(("%d:person:%s" % (SEED, r["user"])).encode()).hexdigest()[:8] for r in rows}
    seq = [{"id": rows[i]["id"], "pass": 1, "g": grp[rows[i]["id"]]} for i in rng.permutation(len(rows))]
    pos = {s["id"]: p for p, s in enumerate(seq)}
    n = len(seq)
    early = [r["id"] for r in rows if pos[r["id"]] <= n - GAP - 1]
    rep = sorted(rng.choice(early, size=min(N_REPEAT, len(early)), replace=False).tolist())
    ins = sorted(((int(rng.integers(max(pos[i] + GAP, n // 2), n + 1)), i) for i in rep), reverse=True)
    for p, pid in ins:
        seq.insert(p, {"id": pid, "pass": 2, "g": grp[pid]})

    # models fit on round 1 (all 380 photos, first showings)
    eff = read_ratings(a.ratings)
    first1 = {k[0]: v for k, v in eff.items() if k[1] == 1 and not v.get("skip")}
    ph1 = [p for p in S1["photos"] if p["id"] in first1]
    ki = {kk: j for j, kk in enumerate(key)}
    r1 = np.array([ki[(p["user"], p["code"])] for p in ph1])
    u1 = np.array([p["user"] for p in ph1])
    y_att = np.array([first1[p["id"]]["attractive"] for p in ph1], float)
    y_imp = np.array([first1[p["id"]]["implication"] for p in ph1], float)
    app1 = np.array([[appearance[x]] for x in u1], float)
    r2 = np.array([r["row"] for r in rows])
    app2 = np.array([[appearance[r["user"]]] for r in rows], float)
    codes = ("oai", "clip_cut", "sig_cut")
    blocks = lambda rr, app: [app] + [X[c][rr] for c in codes]  # noqa: E731

    people1 = sorted(set(u1))
    cvB = np.zeros(len(SEEDS))
    cvC = np.zeros((len(SEEDS), len(ALPHAS)))
    for s_, seed in enumerate(SEEDS):
        fp = dict(zip(people1, folds(people1, group, seed=seed)))
        fo = np.array([fp[x] for x in u1])
        oofB = np.zeros(len(u1))
        oofC = np.zeros((len(ALPHAS), len(u1)))
        for k in range(5):
            tr, te = np.where(fo != k)[0], np.where(fo == k)[0]
            XB = np.c_[app1, X["standin"][r1]]
            oofB[te] = ols_predict(XB[tr], y_att[tr], XB[te])[0]
            A, B = prep(blocks(r1, app1), tr, te)
            for j, pr in enumerate(ridge_all(A, y_att[tr], B, ALPHAS)):
                oofC[j, te] = pr
        cvB[s_] = rho(oofB, y_att)
        cvC[s_] = [rho(oofC[j], y_att) for j in range(len(ALPHAS))]
    cvC = cvC.mean(0)
    jC = int(np.nanargmax(cvC))
    cv = {"B": round(float(cvB.mean()), 4), "C": round(float(cvC[jC]), 4), "C_alpha": float(ALPHAS[jC]),
          "appearance_alone": round(rho(app1[:, 0], y_att), 4), "standin_alone": round(rho(X["standin"][r1], y_att), 4)}
    chosen = "B" if cv["B"] >= cv["C"] else "C"
    predB, wB = ols_predict(np.c_[app1, X["standin"][r1]], y_att, np.c_[app2, X["standin"][r2]])
    predC = fit_predict(blocks(r1, app1), y_att, blocks(r2, app2), float(ALPHAS[jC]))
    # also reported: round 1's adopted models refit on all 380 photos with their round-1 strengths
    img_att = fit_predict([X[c][r1] for c in codes], y_att, [X[c][r2] for c in codes], 10.0)
    imp = fit_predict([X[c][r1] for c in codes] + [X["stand5"][r1]], y_imp,
                      [X[c][r2] for c in codes] + [X["stand5"][r2]], 56.23413251903491)
    model = predB if chosen == "B" else predC
    pred = {"rule": "PROTOCOL.md 'Sexual attractiveness, round 2'", "cv_round1": cv, "chosen": chosen,
            "B_weights": [round(float(w), 5) for w in wB],
            "photos": [{"id": r["id"], "g": grp[r["id"]], "own": r["own"], "model": round(float(m), 5),
                        "appearance": float(app2[i, 0]), "standin": r["standin"], "image_only": round(float(ia), 5),
                        "implication_model": round(float(im), 5)}
                       for i, (r, m, ia, im) in enumerate(zip(rows, model, img_att, imp))]}
    os.makedirs(os.path.join(R, "results"), exist_ok=True)
    path = os.path.join(R, "results/suggest_fresh_pred.json")
    json.dump(pred, open(path, "w"), indent=0, sort_keys=True)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()

    import cv2
    from owner_cutouts import load as load_photo
    os.makedirs(os.path.join(site, "img"), exist_ok=True)
    for r in rows:
        rec = {p["code"]: p for p in json.load(open(os.path.join(R, "cut", r["user"], "owner.json")))["photos"]}[r["code"]]
        pdir = os.path.join(R, "photos", r["user"])
        fn = [x for x in os.listdir(pdir) if os.path.splitext(x)[0] == r["code"]][0]
        rgb = load_photo(os.path.join(pdir, fn))
        x0, y0, x1, y1 = rec["crop"]
        c = rgb[y0:y1, x0:x1]
        f = min(1.0, MAX_SIDE / max(c.shape[:2]))
        if f < 1:
            c = cv2.resize(c, (round(c.shape[1] * f), round(c.shape[0] * f)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(site, "img", r["id"] + ".jpg"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump(seq, open(os.path.join(site, "items.json"), "w"))
    rec = {"seed": SEED, "photos": rows, "sequence": seq, "people": fresh}
    json.dump(rec, open(os.path.join(R, "split/suggest_set2.json"), "w"), indent=0, sort_keys=True)
    male = {k: float(np.nanmedian(d["male"][u == k].astype(float))) for k in fresh}
    print("fresh people %d (%d look male), photos %d (%s), screens %d" % (
        len(fresh), sum(v >= 0.5 for v in male.values()), len(rows),
        {w: sum(r["own"] == w for r in rows) for w in ("low", "high", "mid")}, len(seq)))
    print("round-1 CV:", cv, "-> chosen", chosen, "| B weights (intercept, appearance, stand-in)", pred["B_weights"])
    print("suggest_fresh_pred.json sha256", sha)


def evaluate(a):
    R = a.root
    P = json.load(open(os.path.join(R, "results/suggest_fresh_pred.json")))
    ph = P["photos"]
    eff = read_ratings(a.ratings)
    first = {k[0]: v for k, v in eff.items() if k[1] == 1 and not v.get("skip")}
    second = {k[0]: v for k, v in eff.items() if k[1] == 2 and not v.get("skip")}
    flagged = {v["id"] for v in eff.values() if v.get("skip") and v.get("reason") == "under18"}
    gone = {p["g"] for p in ph if p["id"] in flagged}
    ph = [p for p in ph if p["id"] in first and p["g"] not in gone]
    g = np.array([p["g"] for p in ph])
    y = np.array([first[p["id"]]["attractive"] for p in ph], float)
    yi = np.array([first[p["id"]]["implication"] for p in ph], float)
    sc = {k: np.array([p[k] for p in ph], float) for k in ("model", "appearance", "standin", "image_only", "implication_model")}
    out = {"photos": len(ph), "people": int(len(np.unique(g))), "people_removed_under18": len(gone),
           "skipped": sum(1 for k, v in eff.items() if k[1] == 1 and v.get("skip")), "chosen": P["chosen"]}
    bm = boot_people(g, lambda i: rho(sc["model"][i], y[i]))
    ba = boot_people(g, lambda i: rho(sc["appearance"][i], y[i]))
    r_m, r_a = rho(sc["model"], y), rho(sc["appearance"], y)

    def within(score, yy):
        right = wrong = 0
        for p in np.unique(g):
            i = np.where(g == p)[0]
            ok = n = 0
            for x in range(len(i)):
                for z in range(x + 1, len(i)):
                    if yy[i[x]] == yy[i[z]]:
                        continue
                    n += 1
                    ok += int((score[i[x]] - score[i[z]]) * (yy[i[x]] - yy[i[z]]) > 0)
            if n and ok / n > 0.5:
                right += 1
            elif n and ok / n < 0.5:
                wrong += 1
        m = right + wrong
        return {"right": right, "wrong": wrong, "p_one_sided": float(binomtest(right, m, 0.5, alternative="greater").pvalue) if m else None}

    res = {"spearman": round(r_m, 4), "ci95": ci(bm), "appearance_spearman": round(r_a, 4), "appearance_ci95": ci(ba),
           "difference": round(r_m - r_a, 4), "difference_ci95": ci(bm - ba), "within_person": within(sc["model"], y)}
    res["rule"] = {"1_spearman": bool(r_m >= 0.5 and res["ci95"][0] >= 0.3), "2_beats_appearance": bool(res["difference_ci95"][0] > 0),
                   "3_within_person": bool(res["within_person"]["p_one_sided"] is not None and res["within_person"]["p_one_sided"] < 0.05)}
    res["pass"] = all(res["rule"].values())
    out["attractiveness"] = res
    out["also"] = {
        "standin_attractiveness": {"spearman": round(rho(sc["standin"], y), 4), "ci95": ci(boot_people(g, lambda i: rho(sc["standin"][i], y[i]))),
                                   "within_person": within(sc["standin"], y)},
        "image_only_attractiveness": {"spearman": round(rho(sc["image_only"], y), 4), "ci95": ci(boot_people(g, lambda i: rho(sc["image_only"][i], y[i]))),
                                      "within_person": within(sc["image_only"], y)},
        "implication_model": {"spearman": round(rho(sc["implication_model"], yi), 4),
                              "ci95": ci(boot_people(g, lambda i: rho(sc["implication_model"][i], yi[i]))),
                              "standin_spearman": round(rho(sc["standin"], yi), 4), "within_person": within(sc["implication_model"], yi)},
        "scales_spearman": round(rho(y, yi), 4),
        "answers": {s: {str(v): int((arr == v).sum()) for v in range(11) if (arr == v).any()} for s, arr in (("attractive", y), ("implication", yi))},
    }
    rep = [p for p in ph if p["id"] in second]
    out["repeats"] = {}
    for s in ("attractive", "implication"):
        x1 = np.array([first[p["id"]][s] for p in rep], float)
        x2 = np.array([second[p["id"]][s] for p in rep], float)
        out["repeats"][s] = {"n": len(rep), "spearman": round(rho(x1, x2), 4) if len(rep) > 2 else None,
                             "pearson": round(float(pearsonr(x1, x2)[0]), 4) if len(rep) > 2 else None,
                             "same": int((x1 == x2).sum()), "within_1": int((abs(x1 - x2) <= 1).sum())}
    path = os.path.join(R, "results/suggest_fresh_eval.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pick", "eval"])
    ap.add_argument("ratings")
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--site", default=None)
    a = ap.parse_args()
    pick(a) if a.mode == "pick" else evaluate(a)
