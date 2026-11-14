#!/usr/bin/env python3
"""Train the appearance taste model on the training accounts and evaluate it once on the test set.

Stage 1 (--stage cv): model choice by cross-validation on training accounts only (5 folds x 3
repeats, same-person accounts kept in one fold). Prints the table; never reads test ratings.
Stage 2 (--stage test): refits the CV winner on all training accounts, scores the frozen test set
once, applies the pass rule from PROTOCOL.md, and runs the few-rating learning curve.

Inputs (project folder): features.npz, cut/<user>/owner.json, split/split.json,
split/same_person.json, split/ratings_effective.json. Output: results/<stage>.json
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ALPHAS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
RNG = np.random.default_rng(20261114)


# ---------- data ----------
def load(root):
    F = np.load(os.path.join(root, "features.npz"), allow_pickle=True)
    split = json.load(open(os.path.join(root, "split/split.json")))
    sp = os.path.join(root, "split/same_person.json")  # optional: accounts of the same person
    same = json.load(open(sp)) if os.path.exists(sp) else {"groups": [], "test_after": split["test"]}
    R = json.load(open(os.path.join(root, "split/ratings_effective.json")))
    first = {u: v for u, v in R["first"].items()}
    test = [u for u in same["test_after"] if u in first]
    train = [u for u in split["train"] if u in first]
    group = {}
    for g in same["groups"]:
        for u in g:
            group[u] = g[0]
    return F, train, test, first, R, group, split, same


def photo_index(F):
    idx = {}
    for i, u in enumerate(F["user"]):
        idx.setdefault(str(u), []).append(i)
    return idx


def mean_vec(F, idx, users, key):
    X = F[key]
    out = np.stack([X[idx[u]].mean(0) for u in users])
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n > 0, n, 1)


def explicit(root, F, idx, users):
    """Per-account explicit features: hair and skin colour, region shares, body proportions,
    zero-shot cues (build, woman, long hair)."""
    rows = []
    for u in users:
        rec = json.load(open(os.path.join(root, "cut", u, "owner.json")))
        ph = rec["photos"]
        def m(vals):
            vals = [v for v in vals if v is not None]
            return np.mean(vals, 0) if vals else None
        hair = m([p.get("hair_lab") for p in ph])
        skin = m([p.get("skin_lab") for p in ph])
        fr = np.mean([[p["frac"][k] for k in ("face", "hair", "skin", "clothing")] for p in ph], 0)
        props = []
        for p in ph:
            k = p.get("keypoints")
            if not k:
                continue
            k = np.array(k)
            ok = lambda *i: all(k[j][2] > 2 for j in i)
            if ok(5, 6, 11, 12):
                sh = np.linalg.norm(k[5, :2] - k[6, :2]); hp = np.linalg.norm(k[11, :2] - k[12, :2])
                torso = np.linalg.norm((k[5, :2] + k[6, :2]) / 2 - (k[11, :2] + k[12, :2]) / 2)
                if torso > 1:
                    leg = None
                    if ok(15, 16):
                        leg = np.linalg.norm((k[11, :2] + k[12, :2]) / 2 - (k[15, :2] + k[16, :2]) / 2) / torso
                    props.append([sh / torso, hp / torso, sh / max(hp, 1), leg])
        pr = []
        for j in range(4):
            vals = [q[j] for q in props if q[j] is not None]
            pr.append(np.median(vals) if vals else np.nan)
        zs = [F["zs_build_cut"][idx[u]].mean(0), F["zs_sex_face"][idx[u]][:, :1].mean(0),
              F["zs_hair_cut"][idx[u]][:, :1].mean(0)]
        rows.append(np.concatenate([hair if hair is not None else [np.nan] * 3,
                                    skin if skin is not None else [np.nan] * 3, fr, pr, *zs]))
    X = np.array(rows, float)
    col_mean = np.nanmean(X, 0)
    X = np.where(np.isnan(X), col_mean, X)
    return X


# ---------- outfit- and scene-invariant projection (no ratings) ----------
def train_projection(F, idx, users_all, keys, group, dim=128, steps=1500, seed=0):
    torch.manual_seed(seed)
    X = np.concatenate([F[k] for k in keys], 1).astype(np.float32)
    X = torch.from_numpy(X)
    label = {}
    for u in users_all:
        label.setdefault(group.get(u, u), len(label))
    photo_users = [str(u) for u in F["user"]]
    keep = [i for i, u in enumerate(photo_users) if u in set(users_all)]
    y = torch.tensor([label[group.get(photo_users[i], photo_users[i])] for i in keep])
    X = X[keep]
    by_label = {}
    for i, l in enumerate(y.tolist()):
        by_label.setdefault(l, []).append(i)
    labels = [l for l, v in by_label.items() if len(v) >= 2]
    W = torch.nn.Linear(X.shape[1], dim, bias=False)
    opt = torch.optim.Adam(W.parameters(), lr=1e-3, weight_decay=1e-4)
    g = np.random.default_rng(seed)
    for _ in range(steps):
        ls = g.choice(labels, size=min(64, len(labels)), replace=False)
        b = []
        for l in ls:
            b += list(g.choice(by_label[l], size=min(4, len(by_label[l])), replace=False))
        xb, yb = X[b], y[b]
        z = torch.nn.functional.normalize(W(xb), dim=1)
        sim = z @ z.T / 0.1
        eye = torch.eye(len(b), dtype=torch.bool)
        sim = sim.masked_fill(eye, -1e9)
        pos = (yb[:, None] == yb[None, :]) & ~eye
        logp = sim - torch.logsumexp(sim, 1, keepdim=True)
        loss = -(logp * pos).sum(1) / pos.sum(1).clamp(min=1)
        opt.zero_grad(); loss.mean().backward(); opt.step()
    with torch.no_grad():
        Z = torch.nn.functional.normalize(W(torch.from_numpy(np.concatenate([F[k] for k in keys], 1).astype(np.float32))), dim=1).numpy()
    return Z


def mean_rows(Z, idx, users):
    out = np.stack([Z[idx[u]].mean(0) for u in users])
    return out / np.linalg.norm(out, axis=1, keepdims=True)


# ---------- model + CV ----------
def fit_predict(Xtr, ytr, Xte, alpha):
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def folds(users, group, k=5, seed=0):
    g = np.random.default_rng(seed)
    gids = sorted({group.get(u, u) for u in users})
    g.shuffle(gids)
    f = {gid: i % k for i, gid in enumerate(gids)}
    return np.array([f[group.get(u, u)] for u in users])


def cv_score(X, y, users, group, alpha, repeats=3):
    rhos = []
    for r in range(repeats):
        fo = folds(users, group, seed=r)
        pred = np.zeros(len(y))
        for k in range(5):
            tr, te = fo != k, fo == k
            pred[te] = fit_predict(X[tr], y[tr], X[te], alpha)
        rhos.append(spearmanr(pred, y).correlation)
    return float(np.mean(rhos))


def best_alpha(X, y, users, group):
    scores = {a: cv_score(X, y, users, group, a) for a in ALPHAS}
    a = max(scores, key=scores.get)
    return a, scores[a]


# ---------- metrics ----------
def boot_ci(pred, y, n=2000, other=None):
    idx = np.arange(len(y)); vals = []
    for _ in range(n):
        s = RNG.choice(idx, len(idx))
        r = spearmanr(pred[s], y[s]).correlation
        if other is not None:
            r -= spearmanr(other[s], y[s]).correlation
        vals.append(r)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def pairwise_acc(pred, y, gap=2):
    ok = tot = 0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if abs(y[i] - y[j]) >= gap:
                tot += 1; ok += (pred[i] - pred[j]) * (y[i] - y[j]) > 0
    return ok / tot if tot else float("nan")


def kcenter_order(X, n, seed=0):
    """Greedy k-center (farthest-first) on standardized features: a rating-free probe order."""
    Xs = StandardScaler().fit_transform(X)
    c = Xs.mean(0)
    first = int(np.argmin(np.linalg.norm(Xs - c, axis=1)))  # most typical account first
    order = [first]
    d = np.linalg.norm(Xs - Xs[first], axis=1)
    while len(order) < n:
        i = int(np.argmax(d)); order.append(i)
        d = np.minimum(d, np.linalg.norm(Xs - Xs[i], axis=1))
    return order


def dopt_order(X, n, alpha):
    """Sequential design: next = account the current ridge model is least sure about (max x'(X'X+aI)^-1 x)."""
    Xs = StandardScaler().fit_transform(X)
    A = alpha * np.eye(Xs.shape[1]); Ainv = np.linalg.inv(A); order = []
    for _ in range(n):
        v = ((Xs @ Ainv) * Xs).sum(1)
        v[order] = -1
        i = int(np.argmax(v)); order.append(i)
        x = Xs[i:i + 1]
        Ainv = Ainv - (Ainv @ x.T @ x @ Ainv) / (1 + x @ Ainv @ x.T)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--stage", choices=["cv", "test"], required=True)
    a = ap.parse_args()
    F, train, test, first, R, group, split, same = load(a.root)
    idx = photo_index(F)
    all_users = sorted(set(split["train"]) | set(split["test"]))
    ytr = np.array([first[u] for u in train], float)

    proj_cache = {}
    def proj(keys):
        k = tuple(keys)
        if k not in proj_cache:
            proj_cache[k] = train_projection(F, idx, all_users, keys, group)
        return proj_cache[k]

    def feats(name, users):
        parts = name.split("+")
        out = []
        for p in parts:
            if p == "explicit":
                out.append(explicit(a.root, F, idx, users))
            elif p.startswith("proj("):
                keys = p[5:-1].split(",")
                out.append(mean_rows(proj(keys), idx, users))
            else:
                out.append(mean_vec(F, idx, users, p))
        return np.concatenate(out, 1)

    candidates = ["clip_cut", "dino_cut", "clip_cut+clip_face", "clip_nocloth+clip_face",
                  "dino_cut+clip_cut+clip_face", "proj(clip_cut,clip_face)", "proj(dino_cut,clip_cut,clip_face)",
                  "explicit"]
    table = {}
    for c in candidates:
        X = feats(c, train)
        al, s = best_alpha(X, ytr, train, group)
        table[c] = {"alpha": al, "cv_spearman": s, "dims": int(X.shape[1])}
        print("%-40s dims %5d  alpha %6g  CV Spearman %.3f" % (c, X.shape[1], al, s), flush=True)
    best = max((c for c in candidates if c != "explicit"), key=lambda c: table[c]["cv_spearman"])
    with_exp = best + "+explicit"
    X = feats(with_exp, train)
    al, s = best_alpha(X, ytr, train, group)
    table[with_exp] = {"alpha": al, "cv_spearman": s, "dims": int(X.shape[1])}
    print("%-40s dims %5d  alpha %6g  CV Spearman %.3f" % (with_exp, X.shape[1], al, s), flush=True)
    winner = with_exp if table[with_exp]["cv_spearman"] > table[best]["cv_spearman"] else best
    controls = {"scene-only": "dino_bg+clip_bg", "clothing-only": "dino_cloth+clip_cloth"}
    for k, c in controls.items():
        Xc = feats(c, train)
        al, s = best_alpha(Xc, ytr, train, group)
        table[c] = {"alpha": al, "cv_spearman": s, "dims": int(Xc.shape[1])}
        print("%-40s dims %5d  alpha %6g  CV Spearman %.3f  (%s control)" % (c, Xc.shape[1], al, s, k), flush=True)
    aes_tr = np.array([F["aesthetic_cut"][idx[u]].mean() for u in train])
    table["crowd aesthetic (no training)"] = {"cv_spearman": float(spearmanr(aes_tr, ytr).correlation)}
    print("crowd aesthetic on training accounts: Spearman %.3f" % table["crowd aesthetic (no training)"]["cv_spearman"])
    print("WINNER:", winner)
    os.makedirs(os.path.join(a.root, "results"), exist_ok=True)
    out = {"n_train": len(train), "n_test": len(test), "cv": table, "winner": winner}
    if a.stage == "cv":
        json.dump(out, open(os.path.join(a.root, "results/cv.json"), "w"), indent=1)
        return

    # ----- stage 2: the one test -----
    yte = np.array([first[u] for u in test], float)
    Xw_tr, Xw_te = feats(winner, train), feats(winner, test)
    pred = fit_predict(Xw_tr, ytr, Xw_te, table[winner]["alpha"])
    aes = np.array([F["aesthetic_cut"][idx[u]].mean() for u in test])
    ctrl_pred = {}
    for k, c in controls.items():
        ctrl_pred[k] = fit_predict(feats(c, train), ytr, feats(c, test), table[c]["alpha"])
    rho = float(spearmanr(pred, yte).correlation)
    ci = boot_ci(pred, yte)
    d_aes = boot_ci(pred, yte, other=aes)
    d_ctrl = {k: boot_ci(pred, yte, other=v) for k, v in ctrl_pred.items()}
    top = np.argsort(-pred)[:max(1, len(pred) // 10)]
    woman = np.array([F["zs_sex_face"][idx[u]][:, 0].mean() > 0.5 for u in test])
    res = {
        "spearman": rho, "spearman_ci95": ci,
        "crowd_aesthetic_spearman": float(spearmanr(aes, yte).correlation), "diff_vs_crowd_ci95": d_aes,
        "controls": {k: {"spearman": float(spearmanr(v, yte).correlation), "diff_ci95": d_ctrl[k]} for k, v in ctrl_pred.items()},
        "pairwise_acc_gap2": pairwise_acc(pred, yte), "mae": float(np.mean(np.abs(pred - yte))),
        "top10pct_rated_8plus": float(np.mean(yte[top] >= 8)), "base_rate_8plus": float(np.mean(yte >= 8)),
        "women": {"n": int(woman.sum()), "spearman": float(spearmanr(pred[woman], yte[woman]).correlation)},
        "men": {"n": int((~woman).sum()), "spearman": float(spearmanr(pred[~woman], yte[~woman]).correlation) if (~woman).sum() > 3 else None},
    }
    rule = [rho >= 0.50 and ci[0] >= 0.30, d_aes[0] > 0, all(v[0] > 0 for v in d_ctrl.values())]
    res["pass_rule"] = {"1_spearman": rule[0], "2_beats_crowd": rule[1], "3_beats_controls": rule[2], "PASS": all(rule)}
    # few-rating curve on the test set
    ns = [5, 10, 15, 20, 30, 40, 60, 80, 120, 160, 200, len(train)]
    al = table[winner]["alpha"]
    curve = {"probe_kcenter": {}, "active_dopt": {}, "random_mean": {}, "random_p10": {}, "random_p90": {}}
    order_k = kcenter_order(Xw_tr, len(train))
    order_d = dopt_order(Xw_tr, min(200, len(train)), al)
    for n in ns:
        for name, order in (("probe_kcenter", order_k), ("active_dopt", order_d)):
            if n <= len(order):
                s = order[:n]
                curve[name][n] = float(spearmanr(fit_predict(Xw_tr[s], ytr[s], Xw_te, al), yte).correlation)
        rs = []
        for r in range(50):
            s = np.random.default_rng(r).choice(len(train), n, replace=False)
            rs.append(spearmanr(fit_predict(Xw_tr[s], ytr[s], Xw_te, al), yte).correlation)
        curve["random_mean"][n] = float(np.nanmean(rs)); curve["random_p10"][n] = float(np.nanpercentile(rs, 10))
        curve["random_p90"][n] = float(np.nanpercentile(rs, 90))
    res["few_rating_curve"] = curve
    out["test"] = res
    out["test_predictions"] = {u: float(p) for u, p in zip(test, pred)}
    json.dump(out, open(os.path.join(a.root, "results/test.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "few_rating_curve"}, indent=1))
    print("few-rating curve:")
    for n in ns:
        print("  n=%4d  probe %.3f  active %s  random %.3f [%.3f, %.3f]" % (
            n, curve["probe_kcenter"].get(n, float("nan")),
            "%.3f" % curve["active_dopt"][n] if n in curve["active_dopt"] else "  -  ",
            curve["random_mean"][n], curve["random_p10"][n], curve["random_p90"][n]))


if __name__ == "__main__":
    main()
