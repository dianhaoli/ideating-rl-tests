"""AUDIT (independent shortcut auditor): offline, CPU-only upper bounds for cheap policies a clever agent (or an RL
policy trained on many instances) could learn. Nothing here runs on the GPU: every activation an attack "measures" is
read from the precomputed max-activation tables (cache/acts_L*_{C,names}.npy), which hold exactly the numbers the
`latent_activations` tool would return (max over non-BOS tokens; NOTES: batched vs precomputed corr 0.999).

Attacks (all are trained on in-memory instances from seeds 600000+, disjoint from the pool (5000+/105000+/205000+),
the fingerprint check (500000+) and prior.json (900000+); evaluated on held-out in-memory instances (610000+) AND on
the real 72-instance pool, graded with grader.grade):

  learned_prior   model-free. What an RL policy could learn from the menus alone over thousands of episodes:
                  a per-(label, layer) answer prior + taxonomy structure features (gradient boosting) to pick an option,
                  and the fingerprint features (fingerprint_check.feats) + the option scores to decide "nothing found".
                  The taxonomy (DBPedia path) is used as a proxy for what an LLM can infer from the labels: upper bound.
  latent_stats    learned_prior PLUS cheap, concept-agnostic latent statistics: the slot latent's activation on a FIXED
                  generic text set (64 texts, same for every slot; a stand-in for "run 64 random texts") and on the 20
                  bare option names (3 surface variants; the token-identity probe). Tests whether planted vs null
                  (esp. off-universe nulls, and the name filter that is applied to planted slots only) leaks through
                  latent statistics, and whether the name filter itself is an exploitable anti-signal.

Policy: per slot answer argmax{ P(null), (1-P(null)) * max_i P(option i | planted) }. pass = all slots right.
Run: $PY -m tasks.featurematch.audit.offline_attacks [--n-train 1200 --n-test 400] --out runs/featurematch/<ts>_audit_offline
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from tasks.featurematch import concepts as C
from tasks.featurematch.fingerprint_check import feats as menu_feats
from tasks.featurematch.fm_core import LAYERS
from tasks.featurematch.generate import TIERS, Tables, make_instance
from tasks.featurematch.grader import grade

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_GENERIC = 64
# ablation for latent_stats: "all" (default), "generic" (only the generic-text statistics), "names" (only the
# option-name probes). Set with FM_AUDIT_FEATS.
FEATS = os.environ.get("FM_AUDIT_FEATS", "all")


class Acts:
    def __init__(self, T):
        meta = json.load(open(os.path.join(C.CACHE, "precompute_meta.json")))
        self.C = {L: np.load(os.path.join(C.CACHE, f"acts_L{L}_C.npy"), mmap_mode="r") for L in LAYERS}
        rng = np.random.default_rng(424242)
        self.generic_rows = np.sort(rng.choice(len(meta["rowsC"]), N_GENERIC, replace=False))
        self.G = {L: np.asarray(self.C[L][self.generic_rows], dtype=np.float32) for L in LAYERS}   # [64, 16384]
        self.names = T.names                                                                    # [3*232, 16384]


def option_rows(T, lab2c, layer, opts, prior, latent=None, A=None):
    idx = [lab2c[o] for o in opts]
    cl = T.close[np.ix_(idx, idx)].astype(float)
    np.fill_diagonal(cl, np.nan)
    rm = np.nanmean(cl, 1)
    rows = []
    nm = None
    if latent is not None and FEATS in ("all", "names"):
        nm = np.array([[A.names[layer][3 * c + v, latent] for v in range(3)] for c in idx], dtype=np.float32)
        am = nm.argmax(0)
        anyp = nm.max(0) > 0
    for k, c in enumerate(idx):
        r = [layer, float(T.paths[c][0] == "Language"), prior.get((c, layer), 0.0), prior.get((c, None), 0.0),
             rm[k], float(np.nanmax(cl[k])), float((cl[k] >= len(T.paths[c]) - 1).sum()), float(rm[k] - np.nanmedian(rm))]
        if nm is not None:
            r += list(nm[k]) + [float(nm[k].max()), float(sum((am[v] == k) and anyp[v] for v in range(3)))]
        rows.append(r)
    return rows


def slot_feats(T, lab2c, layer, opts, opt_scores, latent=None, A=None):
    f = list(menu_feats(T, layer, opts, lab2c))
    s = np.sort(opt_scores)[::-1]
    f += [s[0], s[0] - s[1], float(np.std(opt_scores))]
    if latent is not None and FEATS in ("all", "generic"):
        g = A.G[layer][:, latent]
        f += [float((g > 0).mean()), float(g.mean()), float(g.max()), float(np.quantile(g, 0.9))]
    if latent is not None and FEATS in ("all", "names"):
        idx = [lab2c[o] for o in opts]
        nm = np.array([[A.names[layer][3 * c + v, latent] for v in range(3)] for c in idx], dtype=np.float32)
        f += [float((nm > 0).mean()), float(nm.max()), float(nm.mean())]
    return f


def gen(T, seeds, tier):
    out = []
    for s in seeds:
        _, inst, pub = make_instance(T, s, tier)
        out.append((inst, pub))
    return out


def build_prior(T, lab2c, data):
    """P(label is the answer | label in a planted menu at layer L), smoothed; also pooled over layers."""
    num, den = Counter(), Counter()
    for inst, pub in data:
        for s, a in zip(pub["slots"], inst["answer"]["slots"]):
            if not a["planted"]:
                continue
            for i, o in enumerate(s["options"], 1):
                c = lab2c[o]
                for key in ((c, s["layer"]), (c, None)):
                    den[key] += 1
                    num[key] += int(a["choice"] == i)
    return {k: (num[k] + 0.05) / (den[k] + 1.0) for k in den}


def fit(T, lab2c, data, prior, A, use_latent):
    Xo, yo = [], []
    for inst, pub in data:
        for s, a, x in zip(pub["slots"], inst["answer"]["slots"], inst["extra"]["slots"]):
            if not a["planted"]:
                continue
            lat = x["real_latent"] if use_latent else None
            rows = option_rows(T, lab2c, s["layer"], s["options"], prior, lat, A)
            Xo += rows
            yo += [int(a["choice"] == i) for i in range(1, len(rows) + 1)]
    mo = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05).fit(np.array(Xo), np.array(yo))
    Xs, ys = [], []
    for inst, pub in data:
        for s, a, x in zip(pub["slots"], inst["answer"]["slots"], inst["extra"]["slots"]):
            lat = x["real_latent"] if use_latent else None
            sc = option_scores(mo, T, lab2c, s, prior, lat, A)
            Xs.append(slot_feats(T, lab2c, s["layer"], s["options"], sc, lat, A))
            ys.append(int(not a["planted"]))
    ms = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05).fit(np.array(Xs), np.array(ys))
    return mo, ms


def option_scores(mo, T, lab2c, s, prior, lat, A):
    p = mo.predict_proba(np.array(option_rows(T, lab2c, s["layer"], s["options"], prior, lat, A)))[:, 1]
    return p / p.sum()


def policy(mo, ms, T, lab2c, pub, prior, inst, A, use_latent):
    answers, pnull, conf = [], [], []
    for s, x in zip(pub["slots"], inst["extra"]["slots"]):
        lat = x["real_latent"] if use_latent else None
        q = option_scores(mo, T, lab2c, s, prior, lat, A)
        pn = ms.predict_proba(np.array([slot_feats(T, lab2c, s["layer"], s["options"], q, lat, A)]))[0, 1]
        best = int(np.argmax(q))
        if pn >= (1 - pn) * q[best]:
            answers.append({"slot": s["slot"], "choice": "nothing found"})
        else:
            answers.append({"slot": s["slot"], "choice": best + 1})
        pnull.append(pn)
    return {"answers": answers}, pnull


def evaluate(name, mo, ms, T, lab2c, prior, A, use_latent, episodes, grade_fn):
    tot = defaultdict(lambda: Counter())
    yn, pn_all = [], []
    for tier, inst, pub, gref in episodes:
        sub, pn = policy(mo, ms, T, lab2c, pub, prior, inst, A, use_latent)
        g = grade_fn(gref, sub)
        d = g["details"]["slots"]
        t = tot[tier]
        t["n"] += 1
        t["pass"] += int(g["pass"])
        t["score_sum_x1000"] += int(round(1000 * g["score"]))
        t["planted"] += sum(x["planted"] for x in d)
        t["planted_ok"] += sum(x["planted"] and x["correct"] for x in d)
        t["null"] += sum(not x["planted"] for x in d)
        t["null_ok"] += sum((not x["planted"]) and x["correct"] for x in d)
        for x, p in zip(d, pn):
            yn.append(int(not x["planted"]))
            pn_all.append(p)
    out = {}
    for tier, t in sorted(tot.items()):
        out[tier] = {"episodes": t["n"], "pass": t["pass"], "pass_rate": round(t["pass"] / t["n"], 4),
                     "mean_score": round(t["score_sum_x1000"] / 1000 / t["n"], 4),
                     "planted_slot_acc": round(t["planted_ok"] / max(1, t["planted"]), 4),
                     "null_slot_acc": round(t["null_ok"] / max(1, t["null"]), 4)}
    out["null_auroc"] = round(float(roc_auc_score(yn, pn_all)), 4) if len(set(yn)) > 1 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-model", default="", help="write the model-free learned_prior policy (pickle) here")
    ap.add_argument("--only-latent", action="store_true", help="skip the model-free learned_prior (for ablations)")
    ap.add_argument("--tiers", default=",".join(TIERS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T = Tables()
    A = Acts(T)
    lab2c = {c["label"]: i for i, c in enumerate(T.cs)}
    pool = []
    for d in sorted(glob.glob(os.path.join(HERE, "instances", "fm-*"))):
        inst = json.load(open(os.path.join(d, "instance.json")))
        pub = json.load(open(os.path.join(d, "public.json")))
        pool.append((inst["tier"], inst, pub, d))
    results = {"config": dict(vars(a), feats=FEATS), "n_pool": len(pool)}
    for tier in a.tiers.split(","):
        ti = sorted(TIERS).index(tier)
        tr = gen(T, range(600000 + 100000 * ti, 600000 + 100000 * ti + a.n_train), tier)
        te = gen(T, range(610000 + 100000 * ti, 610000 + 100000 * ti + a.n_test), tier)
        prior = build_prior(T, lab2c, tr)
        te_eps = [(tier, inst, pub, inst) for inst, pub in te]
        pool_eps = [e for e in pool if e[0] == tier]

        def g_mem(inst, sub):
            return grade_mem(inst, sub)
        for name, use_lat in ((("learned_prior", False),) if not a.only_latent else ()) + (("latent_stats", True),):
            mo, ms = fit(T, lab2c, tr, prior, A, use_lat)
            r_te = evaluate(name, mo, ms, T, lab2c, prior, A, use_lat, te_eps, g_mem)
            r_pool = evaluate(name, mo, ms, T, lab2c, prior, A, use_lat, pool_eps, grade)
            results.setdefault(name, {})[tier] = {"heldout_inmemory": r_te, "pool72": r_pool}
            print(name, tier, json.dumps(results[name][tier]), flush=True)
            if a.save_model and name == "learned_prior":
                import pickle
                os.makedirs(a.save_model, exist_ok=True)
                pickle.dump({"mo": mo, "ms": ms, "prior": prior}, open(os.path.join(a.save_model, f"lp_{tier}.pkl"), "wb"))
    json.dump(results, open(os.path.join(a.out, "offline_attacks.json"), "w"), indent=1)


def grade_mem(inst, sub):
    """grader.grade on an in-memory instance (same logic; writes a temp instance.json)."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        json.dump(inst, open(os.path.join(td, "instance.json"), "w"))
        return grade(td, sub)


if __name__ == "__main__":
    main()
