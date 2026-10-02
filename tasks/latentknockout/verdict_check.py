"""CPU re-check of the pre-registered decision rule for FEASIBILITY_VERDICT.md (no GPU, no model).

Reads the stored per-item margins of the feasibility sweep (runs/latentknockout/20261002T0642_sweep/cells, seed 0)
and recomputes, with its own few lines of metric code (no import of analyze.py or the skeptics' scripts):
  - the pre-registered quantities p_pair, p_cell, median R_ref(k<=10) at the best layer, q_b for each cheap recipe,
    and the per-family feasible-group counts (ADJUST check (c));
  - the same quantities under a stricter Effect: a held-out target counts as knocked out only if its accepted answer
    now trails the new top-1 by >= delta logits (delta = 0 is the pre-registered hard flip);
  - the same with near-tie held-out targets dropped (clean lead < 1.5 logits), the skeptic's suggested item filter;
  - the clean-margin distribution of held-out new-style targets.
Note: the reference sets were chosen for the lenient (delta = 0) objective, so the strict rows UNDER-state what a
reference built for the strict objective would reach (the skeptic's strict oracle shows 0.72-0.77 on 3 cells).

usage: python -m tasks.latentknockout.verdict_check <cells_dir> <out.json>
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

KAPPA, TAU = 0.1, 0.5
CHEAP = ["mostact_k5", "rand_k5_0", "randact_k5_0", "cos_k5", "naive_k5"]  # pre-registered cheap list
EXTRA = ["contr_k5"]  # reference's first stage; rule (f) only


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def R(c, name, delta, min_clean=None):
    roles = np.array([it[0] for it in c["items"]])
    m = np.array(c["margins"][name], np.float32)
    clean = np.array(c["clean_margin"], np.float32)
    kl = float(np.mean(c["kl"][name]))
    t = roles == "tS"
    if min_clean is not None:
        t = t & (clean >= min_clean)
    s = roles == "sS"
    if t.sum() == 0:
        return None
    return float((m[t] < -delta).mean() * (m[s] > 0).mean() * max(0.0, 1 - kl / KAPPA))


def summarise(cells, delta, min_clean=None):
    rows = []
    for c in cells:
        r = {"pair": (c["family"], c["group"]), "layer": c["layer"]}
        for n in ["ref_k5", "ref_k10"] + CHEAP + EXTRA:
            r[n] = R(c, n, delta, min_clean)
        rows.append(r)
    feas = [r for r in rows if r["ref_k5"] >= TAU]
    pairs = defaultdict(list)
    for r in rows:
        pairs[r["pair"]].append(r)
    feas_pairs = [p for p, rs in pairs.items() if any(x["ref_k5"] >= TAU for x in rs)]
    fam_groups = defaultdict(int)
    for p in feas_pairs:
        fam_groups[p[0]] += 1
    best10 = [max(x["ref_k10"] for x in rs) for rs in pairs.values()]
    q = {}
    for b in CHEAP + EXTRA:
        k = sum(r[b] >= 0.5 * r["ref_k5"] for r in feas)
        q[b] = dict(q=round(k / len(feas), 3) if feas else None, k=k, n=len(feas), ci=wilson(k, len(feas)))
    q80 = sum(r["contr_k5"] >= 0.8 * r["ref_k5"] for r in feas)
    by_layer = {}
    for L in sorted({r["layer"] for r in rows}):
        rl = [r for r in rows if r["layer"] == L]
        by_layer[L] = dict(n=len(rl), feasible=sum(r["ref_k5"] >= TAU for r in rl),
                           median_ref_k5=round(float(np.median([r["ref_k5"] for r in rl])), 3))
    any_method = sum(max(r[n] for n in ["ref_k5", "cos_k5", "naive_k5", "contr_k5"]) >= TAU for r in rows)
    return dict(
        delta=delta, min_clean=min_clean, n_cells=len(rows), n_pairs=len(pairs),
        p_cell=round(len(feas) / len(rows), 3), p_cell_ci=wilson(len(feas), len(rows)), n_feasible_cells=len(feas),
        p_pair=round(len(feas_pairs) / len(pairs), 3), p_pair_ci=wilson(len(feas_pairs), len(pairs)),
        n_feasible_pairs=len(feas_pairs),
        median_ref_k10_best_layer=round(float(np.median(best10)), 3),
        families_with_ge3_feasible_groups=sorted(f for f, n in fam_groups.items() if n >= 3),
        feasible_groups_by_family=dict(fam_groups),
        q=q, contr_ge_0_8_ref_share=round(q80 / len(feas), 3) if feas else None,
        cells_any_of_ref_cos_naive_contr_ge_tau=any_method, by_layer=by_layer)


def main():
    cdir, out = sys.argv[1:3]
    cells = []
    for f in sorted(glob.glob(os.path.join(cdir, "*__s0.json"))):
        c = json.load(open(f))
        if "margins" not in c or c["family"] == "country_capital":
            continue
        cells.append(c)
    res = dict(n_cells_loaded=len(cells), rules={})
    for d in (0.0, 0.5, 1.0, 2.0):
        res["rules"][f"delta{d}"] = summarise(cells, d)
    res["rules"]["delta0.0_drop_clean_lt_1.5"] = summarise(cells, 0.0, 1.5)
    res["rules"]["delta1.0_drop_clean_lt_1.5"] = summarise(cells, 1.0, 1.5)
    cm = []
    for c in cells:
        if c["layer"] not in (12, 18):
            continue
        roles = [it[0] for it in c["items"]]
        cm += [m for m, r in zip(c["clean_margin"], roles) if r == "tS"]
    cm = np.array(cm, np.float32)
    res["clean_margin_tS_L12_L18"] = dict(n=int(len(cm)), median=round(float(np.median(cm)), 2),
                                          share_lt_1=round(float((cm < 1).mean()), 3),
                                          share_lt_1_5=round(float((cm < 1.5).mean()), 3),
                                          share_ge_3=round(float((cm >= 3).mean()), 3))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, default=str)
    for k, v in res["rules"].items():
        print(k, "p_cell", v["p_cell"], v["p_cell_ci"], "p_pair", v["p_pair"], v["p_pair_ci"],
              "medR10", v["median_ref_k10_best_layer"], "fam>=3", v["families_with_ge3_feasible_groups"],
              "| q:", {b: (x["q"], x["ci"]) for b, x in v["q"].items()}, "| contr>=0.8:", v["contr_ge_0_8_ref_share"],
              "| any>=tau:", v["cells_any_of_ref_cos_naive_contr_ge_tau"], "| by layer:", v["by_layer"])
    print("clean margin", res["clean_margin_tS_L12_L18"])


if __name__ == "__main__":
    main()
