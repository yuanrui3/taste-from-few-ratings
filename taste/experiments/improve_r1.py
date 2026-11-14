#!/usr/bin/env python3
"""Round 1 of improvements (see PROTOCOL.md "Round 1 of improvements"). Training accounts only.
GPU only for the zero-shot text prompts; everything else is CPU.
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]  # experiments/ and the pipeline folder
from probe_search import herding, winner_features  # noqa: E402
from train_eval import ALPHAS, folds, load, mean_vec, photo_index  # noqa: E402

NS = [5, 10, 20, 40, 80]
RALPHAS = np.logspace(-1, 5, 25)
POS = ["a photo of a very attractive person", "a photo of a beautiful woman", "a photo of a handsome man", "a photo of a gorgeous model"]
NEG = ["a photo of an unattractive person", "a photo of a plain-looking woman", "a photo of a plain-looking man", "a photo of an ordinary-looking person"]


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def cv_pred(make_pred, y, users, group, repeats=3):
    """make_pred(tr_mask, te_mask) -> predictions for te; returns mean Spearman over repeats."""
    rhos = []
    for r in range(repeats):
        fo = folds(users, group, seed=r); pred = np.zeros(len(y))
        for k in range(5):
            tr, te = fo != k, fo == k
            pred[te] = make_pred(tr, te)
        rhos.append(spearmanr(pred, y).correlation)
    return float(np.mean(rhos))


def ridge_fp(X, y, a):
    def f(tr, te):
        sc = StandardScaler().fit(X[tr]); return Ridge(alpha=a).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return f


def best(score_fn, grid):
    s = {g: score_fn(g) for g in grid}; g = max(s, key=s.get); return g, s[g]


def text_scores(F, M, dev):
    import open_clip
    out = {}
    for name, arch, pre, keys in [("clipL", "ViT-L-14", "laion2b_s32b_b82k", (F["clip_nocloth"], F["clip_face"])),
                                  ("siglip", "ViT-SO400M-14-SigLIP-384", "webli", (M["siglip_nocloth"], M["siglip_face"]))]:
        m, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pre); m = m.eval().to(dev)
        tok = open_clip.get_tokenizer(arch)
        with torch.no_grad():
            t = torch.nn.functional.normalize(m.encode_text(tok(POS + NEG).to(dev)).float(), dim=-1).cpu().numpy()
        del m; torch.cuda.empty_cache()
        p, n = t[:len(POS)], t[len(POS):]
        out[name] = [E @ p.T.mean(1) - E @ n.T.mean(1) for E in keys]  # per photo: nocloth, face
        print("text prompts embedded with", name, flush=True)
    return out


def main(root):
    F, train, test, first, R, group, split, same = load(root)
    idx = photo_index(F)
    M = dict(np.load(os.path.join(root, "features_more_vith_siglip.npz"), allow_pickle=True))
    y = np.array([first[u] for u in train], float)
    out = {"full": {}, "prior": {}, "few": {}}
    skip_full = "--skip-full" in sys.argv  # rerun after a crash: reuse results/improve_r1_full.json

    # ---------------- full-data ----------------
    if not skip_full:
        def acc(key, src=F, users=train):
            return mean_vec(src, idx, users, key)
        XL = np.c_[acc("clip_nocloth"), acc("clip_face")]
        XS = np.c_[acc("siglip_nocloth", M), acc("siglip_face", M)]
        XH = np.c_[acc("vith_nocloth", M), acc("vith_face", M)]
        XLS = np.c_[XL, XS]
        a0, s0 = best(lambda a: cv_pred(ridge_fp(XLS, y, a), y, train, group), ALPHAS)
        out["full"]["baseline L+SigLIP"] = {"cv": s0, "alpha": a0}
        print("full baseline L+SigLIP %.3f (alpha %g)" % (s0, a0), flush=True)
        # F1 ensemble
        al = {nm: best(lambda a: cv_pred(ridge_fp(X, y, a), y, train, group), ALPHAS)[0] for nm, X in (("L", XL), ("S", XS), ("H", XH))}
        def ens(tr, te):
            ps = []
            for nm, X in (("L", XL), ("S", XS), ("H", XH)):
                sc = StandardScaler().fit(X[tr]); m = Ridge(alpha=al[nm]).fit(sc.transform(X[tr]), y[tr])
                ptr, pte = m.predict(sc.transform(X[tr])), m.predict(sc.transform(X[te]))
                ps.append((pte - ptr.mean()) / (ptr.std() + 1e-12))
            return np.mean(ps, 0)
        out["full"]["F1 ensemble"] = {"cv": cv_pred(ens, y, train, group), "alphas": al}
        # F2 photo-level
        rows = np.concatenate([idx[u] for u in train]); owner = np.concatenate([[i] * len(idx[u]) for i, u in enumerate(train)])
        P = np.c_[F["clip_nocloth"][rows], F["clip_face"][rows], M["siglip_nocloth"][rows], M["siglip_face"][rows]]
        def photo_fp(a):
            def f(tr, te):
                trp, tep = np.isin(owner, np.where(tr)[0]), np.isin(owner, np.where(te)[0])
                sc = StandardScaler().fit(P[trp]); m = Ridge(alpha=a).fit(sc.transform(P[trp]), y[owner[trp]])
                pp = m.predict(sc.transform(P[tep])); o = owner[tep]
                return np.array([pp[o == i].mean() for i in np.where(te)[0]])
            return f
        a2, s2 = best(lambda a: cv_pred(photo_fp(a), y, train, group), [a * 10 for a in ALPHAS])
        out["full"]["F2 photo-level"] = {"cv": s2, "alpha": a2}
        # F3 robust account mean
        def robust(key, src):
            res = []
            for u in train:
                E = src[key][idx[u]]
                if len(E) >= 4:
                    ref = F["clip_nocloth"][idx[u]]; med = np.median(ref, 0); sim = ref @ med
                    keep = sim >= np.quantile(sim, 0.25); E = E[keep]
                v = E.mean(0); res.append(v / (np.linalg.norm(v) + 1e-12))
            return np.stack(res)
        XR = np.c_[robust("clip_nocloth", F), robust("clip_face", F), robust("siglip_nocloth", M), robust("siglip_face", M)]
        a3, s3 = best(lambda a: cv_pred(ridge_fp(XR, y, a), y, train, group), ALPHAS)
        out["full"]["F3 robust mean"] = {"cv": s3, "alpha": a3}
        # F4 kernel ridge RBF
        Xs = StandardScaler().fit_transform(XLS); sq = (Xs ** 2).sum(1); D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Xs @ Xs.T, 0); med = np.median(D2[D2 > 0])
        def krr_fp(params):
            bw, a = params
            def f(tr, te):
                sc = StandardScaler().fit(XLS[tr]); A, B = sc.transform(XLS[tr]), sc.transform(XLS[te]); ym = y[tr].mean()
                return KernelRidge(alpha=a, kernel="rbf", gamma=1.0 / (bw * med)).fit(A, y[tr] - ym).predict(B) + ym
            return f
        grid = [(bw, a) for bw in (0.5, 1, 2) for a in (0.01, 0.03, 0.1, 0.3, 1, 3)]
        p4, s4 = best(lambda g: cv_pred(krr_fp(g), y, train, group), grid)
        out["full"]["F4 kernel ridge"] = {"cv": s4, "bandwidth_x": p4[0], "alpha": p4[1]}
        for k, v in out["full"].items():
            print("full %-22s CV %.3f" % (k, v["cv"]), flush=True)
        bestF = max((k for k in out["full"] if k[0] == "F"), key=lambda k: out["full"][k]["cv"])
        json.dump(out["full"], open(os.path.join(root, "results/improve_r1_full.json"), "w"), indent=1)
    else:
        out["full"] = json.load(open(os.path.join(root, "results/improve_r1_full.json")))
        s0 = out["full"]["baseline L+SigLIP"]["cv"]
        bestF = max((k for k in out["full"] if k[0] == "F"), key=lambda k: out["full"][k]["cv"])
    out["full_verdict"] = {"best": bestF, "gain": out["full"][bestF]["cv"] - s0, "replace": bool(out["full"][bestF]["cv"] - s0 >= 0.02)}

    # ---------------- zero-rating prior ----------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ts = text_scores(F, M, dev)
    allu = sorted(set(split["train"]) | set(split["test"])); pos = {u: i for i, u in enumerate(allu)}
    def acc_scalar(v, users):
        return np.array([v[idx[u]].mean() for u in users])
    zparts = [zs(acc_scalar(v, allu)) for nm in ts for v in ts[nm]]
    Zshot = zs(np.mean(zparts, 0))
    S = dict(np.load(os.path.join(root, "datasets/scut/scut.npz"), allow_pickle=True))
    sc = StandardScaler().fit(S["emb"]); crowd = RidgeCV(alphas=RALPHAS).fit(sc.transform(S["emb"]), S["mean"])
    face = mean_vec(F, idx, allu, "clip_face"); Zscut = zs(crowd.predict(sc.transform(face)))
    Zcomb = zs(Zscut + Zshot)
    tr = np.array([pos[u] for u in train])
    for nm, zz in (("SCUT", Zscut), ("zero-shot", Zshot), ("SCUT + zero-shot", Zcomb)):
        out["prior"][nm] = float(spearmanr(zz[tr], y).correlation)
    for nm, zz in zip([f"{m} {p}" for m in ts for p in ("nocloth", "face")], zparts):
        out["prior"]["part " + nm] = float(spearmanr(zz[tr], y).correlation)
    for k, v in out["prior"].items():
        print("prior %-24s zero-rating %.3f" % (k, v), flush=True)

    # ---------------- few-rating ----------------
    Xall = StandardScaler().fit_transform(winner_features(F, idx, allu))
    XLSall = StandardScaler().fit_transform(np.c_[winner_features(F, idx, allu), mean_vec(M, idx, allu, "siglip_nocloth"), mean_vec(M, idx, allu, "siglip_face")])
    Z32 = PCA(n_components=32, random_state=0).fit_transform(Xall)
    PC = PCA(n_components=64, random_state=0).fit_transform(Xall)
    X, XX, Zp, Pc = Xall[tr], XLSall[tr], Z32[tr], PC[tr]
    zc_s, zc_c = Zscut[tr], Zcomb[tr]
    gid = np.array([group.get(u, u) for u in train]); ug = sorted(set(gid))
    meths = ["baseline", "prior SCUT+zero-shot", "R2 transfer ridge", "R3 PCA-k", "R4 switch"]
    res = {m: {n: [] for n in NS} for m in meths}
    def blend(prior, pr, n):
        w = n / (n + 20); return (1 - w) * rankdata(prior) / len(prior) + w * rankdata(pr) / len(pr)
    for s in range(20):
        g = np.random.default_rng(1000 + s); gs = ug[:]; g.shuffle(gs); pool = set(gs[:int(0.7 * len(gs))])
        isP = np.array([x in pool for x in gid]); Pi, Ei = np.where(isP)[0], np.where(~isP)[0]
        order = herding(Zp[Pi], max(NS))
        for n in NS:
            pk = Pi[order[:n]]; yy = y[pk]
            pr = RidgeCV(alphas=RALPHAS).fit(X[pk], yy).predict(X[Ei])
            res["baseline"][n].append(spearmanr(blend(zc_s[Ei], pr, n), y[Ei]).correlation)
            res["prior SCUT+zero-shot"][n].append(spearmanr(blend(zc_c[Ei], pr, n), y[Ei]).correlation)
            A = np.c_[X, 30 * zc_s]
            res["R2 transfer ridge"][n].append(spearmanr(RidgeCV(alphas=RALPHAS).fit(A[pk], yy).predict(A[Ei]), y[Ei]).correlation)
            bestk, bests = 64, np.inf
            for k in (4, 8, 16, 32, 64):
                if k >= n: continue
                m = RidgeCV(alphas=RALPHAS, store_cv_results=True).fit(Pc[pk][:, :k], yy)
                e = float(np.min(m.cv_results_.mean(0)))
                if e < bests: bests, bestk = e, k
            prk = RidgeCV(alphas=RALPHAS).fit(Pc[pk][:, :bestk], yy).predict(Pc[Ei][:, :bestk])
            res["R3 PCA-k"][n].append(spearmanr(blend(zc_s[Ei], prk, n), y[Ei]).correlation)
            prs = RidgeCV(alphas=RALPHAS).fit(XX[pk], yy).predict(XX[Ei]) if n >= 40 else pr
            res["R4 switch"][n].append(spearmanr(blend(zc_s[Ei], prs, n), y[Ei]).correlation)
    out["few"] = {m: {n: float(np.nanmean(v)) for n, v in d.items()} for m, d in res.items()}
    out["few_verdict"] = {}
    for m in meths[1:]:
        gain = np.mean([out["few"][m][n] - out["few"]["baseline"][n] for n in NS[:4]])
        wins = int(np.sum(np.array(res[m][20]) > np.array(res["baseline"][20])))
        out["few_verdict"][m] = {"mean_gain_5_40": float(gain), "wins_n20": wins, "replace": bool(gain >= 0.02 and wins >= 15)}
    print("\n%-22s %s" % ("few-rating", "  ".join("n=%d" % n for n in NS)))
    for m in meths:
        print("%-22s %s" % (m, "  ".join("%.3f" % out["few"][m][n] for n in NS)))
    for m, v in out["few_verdict"].items():
        print("%-22s gain %+.3f, wins %d/20 -> %s" % (m, v["mean_gain_5_40"], v["wins_n20"], "replace" if v["replace"] else "keep baseline"))
    zgain = out["prior"]["SCUT + zero-shot"] - out["prior"]["SCUT"]
    out["prior_verdict"] = {"zero_rating_gain": zgain, "wins_n20": out["few_verdict"]["prior SCUT+zero-shot"]["wins_n20"],
                            "adopt": bool(zgain >= 0.03 and out["few_verdict"]["prior SCUT+zero-shot"]["wins_n20"] >= 15)}
    print("full: best %s, gain %+.3f -> %s" % (bestF, out["full_verdict"]["gain"], "replace" if out["full_verdict"]["replace"] else "keep"))
    print("prior: zero-rating gain %+.3f, wins %d/20 -> %s" % (zgain, out["prior_verdict"]["wins_n20"], "adopt" if out["prior_verdict"]["adopt"] else "keep SCUT"))
    json.dump(out, open(os.path.join(root, "results/improve_r1.json"), "w"), indent=1)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else ".")
