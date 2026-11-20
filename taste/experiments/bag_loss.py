#!/usr/bin/env python3
"""Check 1 of the bottleneck review (PROTOCOL.md, "Three checks from the bottleneck review"): train the full
model on per-person averages instead of copying each person's rating onto every photo. CPU, a few minutes.

Loss, with the adopted full model's inputs (L-14 and SigLIP codes of the no-clothing cut-out and the
owner-only face crop, plus the person's mean SFace code):
    sum_p n_p (y_p - w.xbar_p)^2 + rho * sum_i (w.(x_i - xbar_p))^2 + alpha |w|^2
rho = 1 is the current model (identical to fitting every photo to its person's rating); rho = 0 is the pure
per-person-average loss. Solved in kernel form on rows [sqrt(n_p) xbar_p ; sqrt(rho) (x_i - xbar_p)].

Usage: bag_loss.py ROOT PAIR_ANSWERS SUGGEST1 SUGGEST2  -> ROOT/results/bag_loss.json
  PAIR_ANSWERS  photo-pair answers (1 = left, 2 = right), opaque pair ids as in results/photo_pairs_set.json
  SUGGEST1/2    the addendum's per-photo ratings (ids as in split/suggest_set.json and suggest_set2.json)
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from sface_input import sface_means  # noqa: E402
from train_eval import folds, load, photo_index  # noqa: E402
from within_person import icc1  # noqa: E402

RHOS = [1.0, 0.5, 0.25, 0.0]
ALPHAS = [1e3, 3e3, 1e4, 3e4, 1e5]
SEEDS = (0, 1, 2)


def features(root, F, M, O, idx, users):
    rows = np.concatenate([idx[u] for u in users])
    owner = np.concatenate([[i] * len(idx[u]) for i, u in enumerate(users)])
    SF = sface_means(root, users)
    P = np.c_[F["clip_nocloth"][rows], O["clip_face"][rows], M["siglip_nocloth"][rows], O["siglip_face"][rows],
              SF[owner]].astype(np.float64)
    return rows, owner, P


class BagFit:
    """Kernel-form ridge for the rho-weighted loss; one eigendecomposition, any alpha."""

    def __init__(self, P, owner, y, rho):
        self.mu, self.sd = P.mean(0), P.std(0) + 1e-9
        Z = (P - self.mu) / self.sd
        ppl = np.unique(owner)
        n = np.array([(owner == p).sum() for p in ppl], float)
        zbar = np.stack([Z[owner == p].mean(0) for p in ppl])
        yp = y[ppl]
        self.ym = float((n * yp).sum() / n.sum())
        A = [np.sqrt(n)[:, None] * zbar]
        t = [np.sqrt(n) * (yp - self.ym)]
        if rho > 0:
            pos = {p: k for k, p in enumerate(ppl)}
            A.append(np.sqrt(rho) * (Z - zbar[[pos[o] for o in owner]]))
            t.append(np.zeros(len(Z)))
        self.A, t = np.vstack(A), np.concatenate(t)
        self.l, self.Q = np.linalg.eigh(self.A @ self.A.T)
        self.qt = self.Q.T @ t

    def predict(self, P, alpha):
        w = self.A.T @ (self.Q @ (self.qt / (self.l + alpha)))
        return self.ym + ((P - self.mu) / self.sd) @ w


def read(path):
    eff = {}
    for line in open(path):
        r = json.loads(line)
        k = (r["id"], r.get("pass", 1))
        if r.get("undo"):
            eff.pop(k, None)
        else:
            eff[k] = r
    return {k[0]: v for k, v in eff.items() if k[1] == 1 and not v.get("skip")}


def main(root, pair_answers, s1, s2):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True))
    O = dict(np.load(os.path.join(root, "features_faceown.npz"), allow_pickle=True))
    users_all = np.array([str(u) for u in F["user"]])
    codes_all = np.array([str(c) for c in F["code"]])
    y = np.array([first[u] for u in train], float)
    rows, owner, P = features(root, F, M, O, idx, train)
    print("training: %d people, %d photos, %d inputs" % (len(train), len(rows), P.shape[1]), flush=True)

    # person-level CV and out-of-fold photo scores, for every rho and alpha
    oof = {r: np.zeros((len(SEEDS), len(ALPHAS), len(rows))) for r in RHOS}
    for si, seed in enumerate(SEEDS):
        fo = folds(train, group, seed=seed)
        for k in range(5):
            trp = np.isin(owner, np.where(fo != k)[0])
            tep = np.isin(owner, np.where(fo == k)[0])
            for r in RHOS:
                f = BagFit(P[trp], owner[trp], y, r)
                for j, a in enumerate(ALPHAS):
                    oof[r][si, j, tep] = f.predict(P[tep], a)
        print("seed %d done" % seed, flush=True)

    def person_scores(photo):
        return np.array([photo[owner == i].mean() for i in range(len(train))])

    out = {"people": len(train), "photos": int(len(rows)), "rho": {}}
    chosen = {}
    for r in RHOS:
        cv = [float(np.mean([spearmanr(person_scores(oof[r][si, j]), y)[0] for si in range(len(SEEDS))]))
              for j in range(len(ALPHAS))]
        j = int(np.argmax(cv))
        chosen[r] = j
        ph = oof[r][:, j].mean(0)
        out["rho"][str(r)] = {"alpha": ALPHAS[j], "cv_spearman": round(cv[j], 4), "cv_by_alpha": [round(c, 4) for c in cv],
                              "icc_photo_scores": round(icc1(ph, owner), 4)}
        print("rho %.2f: alpha %g, person-level CV %.4f, ICC of photo scores %.3f" % (r, ALPHAS[j], cv[j], out["rho"][str(r)]["icc_photo_scores"]), flush=True)

    # photo scores for anyone: out-of-fold for training people, all-training fit for the rest
    key_oof = {(users_all[rr], codes_all[rr]): i for i, rr in enumerate(rows)}
    full = {r: BagFit(P, owner, y, r) for r in RHOS}

    def photo_score(r, u, c):
        if (u, c) in key_oof:
            return float(oof[r][:, chosen[r], key_oof[(u, c)]].mean())
        rr = np.where((users_all == u) & (codes_all == c))[0]
        assert len(rr) == 1, (u, c)
        sf = sface_means(root, [u])
        x = np.c_[F["clip_nocloth"][rr], O["clip_face"][rr], M["siglip_nocloth"][rr], O["siglip_face"][rr], sf].astype(np.float64)
        return float(full[r].predict(x, ALPHAS[chosen[r]])[0])

    # (a) the 40 answered photo pairs (training people, out-of-fold scores)
    pairs = {p["id"]: p for p in json.load(open(os.path.join(root, "results/photo_pairs_set.json")))}
    ans = {k: v["rating"] for k, v in read(pair_answers).items() if v.get("rating") in (1, 2)}
    picks = {}
    for r in RHOS:
        picks[r] = {}
        for pid, a in ans.items():
            p = pairs[pid]
            sl, sr = photo_score(r, p["user"], p["left"]), photo_score(r, p["user"], p["right"])
            picks[r][pid] = (1 if sl > sr else 2) == a
    out["photo_pairs"] = {str(r): {"right": int(sum(picks[r].values())), "n": len(ans)} for r in RHOS}
    dis = [pid for pid in ans if picks[0.0][pid] != picks[1.0][pid]]
    k0 = sum(picks[0.0][pid] for pid in dis)
    out["photo_pairs_disagree"] = {"n": len(dis), "rho0_right": int(k0),
                                   "p_one_sided": float(binomtest(k0, len(dis), 0.5, alternative="greater").pvalue) if dis else None}
    for r in RHOS:
        print("rho %.2f: photo pairs right %d of %d" % (r, out["photo_pairs"][str(r)]["right"], len(ans)), flush=True)
    print("pairs where rho 0 and rho 1 pick differently: %d, rho 0 right on %d" % (len(dis), k0), flush=True)

    # (b) secondary: the addendum's per-photo sexual-attractiveness ratings, within a person
    rated = []
    for sset, ratings in (("split/suggest_set.json", s1), ("split/suggest_set2.json", s2)):
        mp = {p["id"]: p for p in json.load(open(os.path.join(root, sset)))["photos"]}
        for pid, v in read(ratings).items():
            if pid in mp:
                rated.append((mp[pid]["user"], mp[pid]["code"], float(v["attractive"])))
    people = sorted({u for u, _, _ in rated})
    out["sexual_attractiveness_within"] = {}
    for r in RHOS:
        sc = {(u, c): photo_score(r, u, c) for u, c, _ in rated}
        conc = tot = right = wrong = 0
        for u in people:
            ph = [(c, a) for uu, c, a in rated if uu == u]
            ok = n = 0
            for x in range(len(ph)):
                for z in range(x + 1, len(ph)):
                    if ph[x][1] == ph[z][1]:
                        continue
                    n += 1
                    ok += int((sc[(u, ph[x][0])] - sc[(u, ph[z][0])]) * (ph[x][1] - ph[z][1]) > 0)
            conc += ok
            tot += n
            if n and ok / n > 0.5:
                right += 1
            elif n and ok / n < 0.5:
                wrong += 1
        out["sexual_attractiveness_within"][str(r)] = {
            "pairs_right": conc, "pairs": tot, "share": round(conc / tot, 4) if tot else None,
            "people_right": right, "people_wrong": wrong,
            "p_one_sided": float(binomtest(right, right + wrong, 0.5, alternative="greater").pvalue) if right + wrong else None}
        print("rho %.2f: sexual attractiveness within a person, pairs right %d of %d; people right %d, wrong %d" % (r, conc, tot, right, wrong), flush=True)

    base = out["rho"]["1.0"]["cv_spearman"]
    a_ok = out["photo_pairs_disagree"]["p_one_sided"] is not None and out["photo_pairs_disagree"]["p_one_sided"] < 0.05 \
        and out["photo_pairs"]["0.0"]["right"] > out["photo_pairs"]["1.0"]["right"]
    b_ok = out["rho"]["0.0"]["cv_spearman"] >= base - 0.01
    out["rule"] = {"photo_scores_switch_to_rho0": bool(a_ok and b_ok), "a_pairs": bool(a_ok), "b_cv_within_0.01": bool(b_ok),
                   "person_scores_change": [r for r in RHOS if r != 1.0 and out["rho"][str(r)]["cv_spearman"] >= base + 0.02]}
    print("rule:", out["rule"], flush=True)
    json.dump(out, open(os.path.join(root, "results/bag_loss.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:5])
