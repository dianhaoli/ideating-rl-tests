"""CPU re-analysis of the ORIGINAL sweep cells with my own metric code (no import of analyze.py).
 1. Stricter effect: a target counts as knocked out only if its answer is beaten by >= delta logits (delta 0, 1, 2).
    Recompute the feasible share (reference R_S >= 0.5, k <= 5) and the recipe share q (baseline >= 0.5 x reference).
 2. Item vs entity-cluster bootstrap CI width of the reference's R_S, and how many feasibility labels are fragile
    (entity-cluster 95% CI straddles 0.5).
 3. Example-objective vs held-out gap for the reference (the 'looks solved on the examples' cells).
usage: python -m tasks.latentknockout.skeptic_reproduce.orig_robustness <cells_dir> <out.json>
"""
import glob
import json
import os
import sys

import numpy as np

KAPPA = 0.1


def R(m, roles, kl, delta, tr="tS", sr="sS"):
    return float((m[roles == tr] < -delta).mean() * (m[roles == sr] > 0).mean() * max(0.0, 1 - kl.mean() / KAPPA))


def boot(m, roles, ents, kl, cluster, B=600, seed=0):
    rng = np.random.default_rng(seed)
    it, isb = np.flatnonzero(roles == "tS"), np.flatnonzero(roles == "sS")
    if cluster:
        gt, gs = {}, {}
        for i in it:
            gt.setdefault(ents[i], []).append(i)
        for i in isb:
            gs.setdefault(ents[i], []).append(i)
        gt, gs = list(gt.values()), list(gs.values())
    out = []
    for _ in range(B):
        if cluster:
            tt = np.concatenate([gt[j] for j in rng.integers(0, len(gt), len(gt))])
            ss = np.concatenate([gs[j] for j in rng.integers(0, len(gs), len(gs))])
        else:
            tt, ss = it[rng.integers(0, len(it), len(it))], isb[rng.integers(0, len(isb), len(isb))]
        kk = kl[rng.integers(0, len(kl), len(kl))]
        out.append((m[tt] < 0).mean() * (m[ss] > 0).mean() * max(0.0, 1 - kk.mean() / KAPPA))
    return np.percentile(out, [2.5, 97.5])


def main():
    cdir, out = sys.argv[1:3]
    rows = []
    for f in sorted(glob.glob(os.path.join(cdir, "*__s0.json"))):
        c = json.load(open(f))
        if "margins" not in c or c["family"] == "country_capital" or c["layer"] not in (12, 18):
            continue
        roles = np.array([it[0] for it in c["items"]])
        ents = [f"{it[1]}|{it[2]}" for it in c["items"]]
        r = dict(cell=f"{c['family']}:{c['group']}:L{c['layer']}")
        for name in ("ref_k5", "cos_k5", "naive_k5", "contr_k5"):
            m = np.array(c["margins"][name], np.float32)
            kl = np.array(c["kl"][name], np.float32)
            for d in (0, 1, 2):
                r[f"{name}_d{d}"] = R(m, roles, kl, d)
        m = np.array(c["margins"]["ref_k5"], np.float32)
        kl = np.array(c["kl"]["ref_k5"], np.float32)
        ci_i, ci_e = boot(m, roles, ents, kl, False), boot(m, roles, ents, kl, True)
        r["ci_item"], r["ci_entity"] = [round(float(x), 3) for x in ci_i], [round(float(x), 3) for x in ci_e]
        tr = c.get("greedy", [])
        # example objective of the chosen k<=5 prefix (the sweep picks the prefix with the best example J)
        r["J_ex_best5"] = max([t["J"] for t in tr[:5]], default=0.0)
        r["n_tS_entities"] = len({e for e, ro in zip(ents, roles) if ro == "tS"})
        rows.append(r)
    res = dict(n_cells=len(rows))
    for d in (0, 1, 2):
        feas = [r for r in rows if r[f"ref_k5_d{d}"] >= 0.5]
        q = {}
        for b in ("cos_k5", "naive_k5", "contr_k5"):
            q[b] = round(float(np.mean([r[f"{b}_d{d}"] >= 0.5 * r[f"ref_k5_d{d}"] for r in feas])), 3) if feas else None
        best_any = sum(max(r[f"{n}_d{d}"] for n in ("ref_k5", "cos_k5", "naive_k5", "contr_k5")) >= 0.5 for r in rows)
        res[f"delta{d}"] = dict(n_feasible_ref=len(feas), share_feasible_ref=round(len(feas) / len(rows), 3),
                                n_cells_any_method_ge_0_5=int(best_any), recipe_q=q,
                                median_ref_R=round(float(np.median([r[f"ref_k5_d{d}"] for r in rows])), 3))
    wi = [r["ci_item"][1] - r["ci_item"][0] for r in rows]
    we = [r["ci_entity"][1] - r["ci_entity"][0] for r in rows]
    res["ci_width_median"] = dict(item=round(float(np.median(wi)), 3), entity_cluster=round(float(np.median(we)), 3))
    res["fragile_labels_entity_ci_straddles_0_5"] = sum(r["ci_entity"][0] < 0.5 <= r["ci_entity"][1] for r in rows)
    res["fragile_labels_item_ci_straddles_0_5"] = sum(r["ci_item"][0] < 0.5 <= r["ci_item"][1] for r in rows)
    res["median_tS_entities"] = float(np.median([r["n_tS_entities"] for r in rows]))
    res["looks_solved_on_examples_but_fails"] = [
        (r["cell"], round(r["J_ex_best5"], 2), round(r["ref_k5_d0"], 2)) for r in rows if r["J_ex_best5"] >= 0.6 and r["ref_k5_d0"] < 0.5]
    res["rows"] = rows
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
