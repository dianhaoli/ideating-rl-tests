"""Offline (CPU, no tools) simulation of the reference solver and the recipes, using the precomputed split-C and
name-probe activations. Used only to TUNE thresholds and budgets quickly; the gate numbers that count come from
run_gates.py, which goes through the real tools (Env) with caps charged.

Run: $PY -m tasks.featurematch.simulate [--k 6 --bg 32 --tau 0.8 --gap 0.05]
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict

import numpy as np

from tasks.featurematch import concepts as C
from tasks.featurematch.fm_core import LAYERS
from tasks.featurematch.grader import grade
from tasks.featurematch.reference_solver import solve_slot

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    meta = json.load(open(os.path.join(C.CACHE, "precompute_meta.json")))
    cs = C.load()
    rowsB = defaultdict(list)          # (named B for history; holds the reference corpus split C)
    for r, c in enumerate(meta["rowsC"]):
        rowsB[c].append(r)
    accB = {L: np.load(os.path.join(C.CACHE, f"acts_L{L}_C.npy")) for L in LAYERS}
    names = {L: np.load(os.path.join(C.CACHE, f"acts_L{L}_names.npy")) for L in LAYERS}
    lab2c = {c["label"]: i for i, c in enumerate(cs)}
    return cs, rowsB, accB, names, lab2c


def sim_ref(inst, pub, D, kw, seed):
    cs, rowsB, accB, names, lab2c = D
    rng = random.Random(seed)
    corpus = {c["label"]: list(range(len(rowsB[i]))) for i, c in enumerate(cs) if c["C"]}   # text ids stand in for texts
    answers, used = [], 0
    for s, priv in zip(pub["slots"], inst["extra"]["slots"]):
        L, j = s["layer"], priv["real_latent"]
        lab_of = {}
        corpus_t = {}
        for lab, ids in corpus.items():
            c = lab2c[lab]
            corpus_t[lab] = [(c, t) for t in ids]
        def measure(texts):
            nonlocal used
            used += len(texts)
            return [float(accB[L][rowsB[c][t], j]) for c, t in texts]
        ch, _ = solve_slot(measure, s["options"], corpus_t, rng, **kw)
        answers.append({"slot": s["slot"], "choice": ch})
    return {"answers": answers}, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k1", type=int, default=5)
    ap.add_argument("--k2", type=int, default=15)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--bg1", type=int, default=32)
    ap.add_argument("--bg2", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.78)
    ap.add_argument("--gap", type=float, default=0.0)
    ap.add_argument("--bg_only", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    D = load()
    kw = {k: v for k, v in vars(a).items() if k in ('k1', 'k2', 'top', 'bg1', 'bg2', 'tau', 'gap', 'bg_only')}
    res = defaultdict(lambda: defaultdict(list))
    slot_acc = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(HERE, "instances", "*"))):
        inst = json.load(open(os.path.join(d, "instance.json")))
        pub = json.load(open(os.path.join(d, "public.json")))
        t = inst["tier"]
        passes = []
        for sd in range(a.seeds):
            sub, used = sim_ref(inst, pub, D, kw, sd)
            g = grade(d, sub)
            passes.append(g["pass"])
            for sl, pr in zip(g["details"]["slots"], inst["extra"]["slots"]):
                slot_acc[(t, pr["kind"])].append(sl["correct"])
        res[t]["ref_one_shot"].append(passes[0])
        res[t]["ref_best_of"].append(any(passes))
        res[t]["ref_mean"].append(np.mean(passes))
        res[t]["forward_used"].append(used)
        n = len(pub["slots"])
        for name, sub in {
            "nothing": {"answers": [{"slot": i, "choice": "nothing found"} for i in range(n)]},
            "always_claim": {"answers": [{"slot": i, "choice": 1} for i in range(n)]},
        }.items():
            res[t][name].append(grade(d, sub)["pass"])
        # naive name-probe recipe, three surface variants, with and without the zero threshold
        for v in range(3):
            for thr in (False, True):
                ans = []
                for s, pr in zip(pub["slots"], inst["extra"]["slots"]):
                    L, j = s["layer"], pr["real_latent"]
                    sc = [float(D[3][L][3 * D[4][o] + v, j]) for o in s["options"]]
                    b = int(np.argmax(sc))
                    ans.append({"slot": s["slot"], "choice": "nothing found" if thr and sc[b] <= 0 else b + 1})
                g = grade(d, {"answers": ans})
                res[t][f"name_v{v}{'_thr' if thr else ''}"].append(g["pass"])
                for sl in g["details"]["slots"]:
                    if sl["planted"]:
                        res[t][f"name_v{v}{'_thr' if thr else ''}_planted_slot"].append(sl["correct"])
    out = {t: {k: round(float(np.mean(v)), 3) for k, v in r.items()} for t, r in res.items()}
    out["slot_acc_ref"] = {f"{t}/{k}": round(float(np.mean(v)), 3) for (t, k), v in slot_acc.items()}
    out["n"] = {t: len(r["nothing"]) for t, r in res.items()}
    out["params"] = vars(a)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
