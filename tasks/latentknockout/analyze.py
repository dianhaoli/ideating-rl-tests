"""CPU aggregation of sweep cells -> tables for FEASIBILITY.md (and the pre-registered decision rule).

usage: python -m tasks.latentknockout.analyze <sweep_dir> [<sweep_dir> ...] --out <tables.json> [--md tables.md]
Metrics (held-out only):
  Effect_X   = share of held-out target items (style X: T = same styles as examples, S = new styles) whose accepted
               answer is no longer top-1 after the ablation;
  Preserve_X = share of held-out sibling items (other groups, same family, style X) still answered correctly;
  KLfac      = 1 - min(1, KL/kappa), KL = mean next-token KL(clean||ablated) on 24 held-out wikitext paragraphs;
  R_X        = Effect_X x Preserve_X x KLfac.  Primary = R_S (pre-registered).  R_all pools T and S items.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

KS = [1, 2, 3, 5, 8, 10]


def load(dirs):
    cells = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            c = json.load(open(f))
            if "skipped" in c or "margins" not in c:
                continue
            c["_file"] = f
            cells.append(c)
    return cells


def metrics(c, name, kappa=0.1, idx=None):
    roles = np.array([it[0] for it in c["items"]])
    m = np.array(c["margins"][name], np.float32)
    kl = np.array(c["kl"][name], np.float32)
    if idx is not None:
        roles, m = roles[idx], m[idx]
    out = {}
    klm = float(kl.mean())
    fac = max(0.0, 1 - klm / kappa)
    for st in ("T", "S"):
        t, s = m[roles == "t" + st], m[roles == "s" + st]
        E = float((t < 0).mean()) if len(t) else np.nan
        P = float((s > 0).mean()) if len(s) else np.nan
        out["E_" + st], out["P_" + st], out["R_" + st] = E, P, E * P * fac
    t = m[np.isin(roles, ["tT", "tS"])]
    s = m[np.isin(roles, ["sT", "sS"])]
    out["E_all"], out["P_all"] = float((t < 0).mean()), float((s > 0).mean())
    out["R_all"] = out["E_all"] * out["P_all"] * fac
    k = m[roles == "ks"]
    if len(k):
        out["P_ks"] = float((k > 0).mean())
        out["R_S_ks"] = out["R_S"] * out["P_ks"]
    out["KL"], out["KLfac"] = klm, fac
    return out


def boot_R(c, name, B=400, seed=0, style="S", kappa=0.1):
    rng = np.random.default_rng(seed)
    roles = np.array([it[0] for it in c["items"]])
    m = np.array(c["margins"][name], np.float32)
    kl = np.array(c["kl"][name], np.float32)
    t, s = m[roles == "t" + style], m[roles == "s" + style]
    rs = []
    for _ in range(B):
        tt = t[rng.integers(0, len(t), len(t))]
        ss = s[rng.integers(0, len(s), len(s))]
        kk = kl[rng.integers(0, len(kl), len(kl))]
        rs.append((tt < 0).mean() * (ss > 0).mean() * max(0, 1 - kk.mean() / kappa))
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def med(x):
    x = [v for v in x if v is not None and not np.isnan(v)]
    return round(float(np.median(x)), 3) if x else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kappa", type=float, default=0.1)
    ap.add_argument("--seeds", nargs="*", default=[], help="dirs with seed 1,2 reference-only cells (memorisation)")
    a = ap.parse_args()
    cells = load(a.dirs)
    kap = a.kappa
    T = {}
    rows = []
    for c in cells:
        meas = c["n"]["tS"] >= 20 and c["n"]["sS"] >= 40
        r = dict(fam=c["family"], group=c["group"], L=c["layer"], seed=c["seed"], measurable=meas, n=c["n"])
        for name in c["set_names"]:
            r[name] = metrics(c, name, kap)
        r["clean_check"] = r["clean"]["E_all"]
        rows.append(r)
    grp = [r for r in rows if r["fam"] != "country_capital" and r["seed"] == 0]
    cc = [r for r in rows if r["fam"] == "country_capital" and r["seed"] == 0]
    meas = [r for r in grp if r["measurable"]]
    T["n_cells"] = dict(group_level=len(grp), measurable=len(meas), country_capital=len(cc),
                        unmeasurable=[f"{r['fam']}:{r['group']}:L{r['L']}" for r in grp if not r["measurable"]])
    T["clean_flip_max"] = max(r["clean_check"] for r in rows) if rows else None
    layers = sorted({r["L"] for r in grp})
    fams = sorted({r["fam"] for r in grp})

    # ---------------- best R by layer and k (median over pairs), R_S primary; also R_T, R_all
    for key in ("R_S", "R_T", "R_all"):
        T[f"median_{key}_by_layer_k"] = {L: {k: med([r[f"ref_k{k}"][key] for r in meas if r["L"] == L]) for k in KS}
                                         for L in layers}
    T["components_k5_by_layer"] = {L: {comp: med([r["ref_k5"][comp] for r in meas if r["L"] == L])
                                       for comp in ("E_S", "P_S", "KL", "KLfac", "E_T", "P_T")} for L in layers}
    # ---------------- feasibility shares
    def share(rs, key, thr, k=5):
        return round(float(np.mean([r[f"ref_k{k}"][key] >= thr for r in rs])), 3) if rs else None
    T["share_cells_R_ge"] = {f"L{L}": {thr: share([r for r in meas if r["L"] == L], "R_S", thr) for thr in (0.5, 0.6, 0.7)}
                             for L in layers}
    T["share_cells_R_ge"]["all"] = {thr: share(meas, "R_S", thr) for thr in (0.5, 0.6, 0.7)}
    T["share_cells_R_ge_by_family"] = {f: {f"L{L}": share([r for r in meas if r["fam"] == f and r["L"] == L], "R_S", 0.5)
                                           for L in layers} for f in fams}
    pairs = defaultdict(list)
    for r in meas:
        pairs[(r["fam"], r["group"])].append(r)
    best = {p: max(v, key=lambda r: r["ref_k5"]["R_S"]) for p, v in pairs.items()}
    T["pairs_best_layer"] = {f"{p[0]}:{p[1]}": dict(L=b["L"], R_S=round(b["ref_k5"]["R_S"], 3),
                                                     R_T=round(b["ref_k5"]["R_T"], 3),
                                                     R10=round(b["ref_k10"]["R_S"], 3)) for p, b in best.items()}
    p_pair = float(np.mean([b["ref_k5"]["R_S"] >= 0.5 for b in best.values()])) if best else np.nan
    p_cell = float(np.mean([r["ref_k5"]["R_S"] >= 0.5 for r in meas])) if meas else np.nan
    T["p_pair"], T["p_cell"] = round(p_pair, 3), round(p_cell, 3)
    T["p_pair_ci"] = wilson(sum(b["ref_k5"]["R_S"] >= 0.5 for b in best.values()), len(best))
    T["p_pair_R07"] = round(float(np.mean([b["ref_k5"]["R_S"] >= 0.7 for b in best.values()])), 3) if best else None
    T["median_R10_best_layer"] = med([max(r["ref_k10"]["R_S"] for r in v) for v in pairs.values()])
    T["median_R5_best_layer"] = med([b["ref_k5"]["R_S"] for b in best.values()])
    feas_by_fam = defaultdict(int)
    for p, b in best.items():
        feas_by_fam[p[0]] += b["ref_k5"]["R_S"] >= 0.5
    T["feasible_groups_by_family"] = dict(feas_by_fam)
    T["pairs_by_family"] = {f: sum(1 for p in best if p[0] == f) for f in fams}
    # ---------------- baselines as share of reference (feasible cells, k=5)
    feas = [r for r in meas if r["ref_k5"]["R_S"] >= 0.5]
    T["n_feasible_cells"] = len(feas)
    base = ["clean", "rand_k5_0", "rand_k5_1", "randact_k5_0", "randact_k5_1", "mostact_k5", "cos_k5", "naive_k5",
            "contr_k5", "naive_k10", "contr_k10", "steer_tuned", "steer_a1", "proj", "contr_k50"]
    bt = {}
    for b in base:
        rs = [r for r in feas if b in r]
        if not rs:
            continue
        ratio = [r[b]["R_S"] / r["ref_k5"]["R_S"] for r in rs]
        q = sum(x >= 0.5 for x in ratio)
        bt[b] = dict(median_ratio=med(ratio), q_ge_half=round(q / len(rs), 3), q_ci=wilson(q, len(rs)),
                     median_R=med([r[b]["R_S"] for r in rs]), median_E=med([r[b]["E_S"] for r in rs]),
                     median_P=med([r[b]["P_S"] for r in rs]), median_KL=med([r[b]["KL"] for r in rs]),
                     q_ge_08=round(sum(x >= 0.8 for x in ratio) / len(rs), 3), n=len(rs))
    T["baselines_on_feasible"] = bt
    # baselines on ALL measurable cells (absolute R), so ~0 baselines are visible even where ref fails
    T["baselines_all_cells_median_R"] = {b: med([r[b]["R_S"] for r in meas if b in r]) for b in base + ["ref_k5"]}
    T["baselines_all_cells_share_R_ge_05"] = {b: round(float(np.mean([r[b]["R_S"] >= 0.5 for r in meas if b in r])), 3)
                                              for b in base + ["ref_k5"] if any(b in r for r in meas)}
    # ---------------- style: R_S vs R_T on feasible cells
    rat = [r["ref_k5"]["R_S"] / r["ref_k5"]["R_T"] for r in feas if r["ref_k5"]["R_T"] > 0]
    T["style"] = dict(median_ratio_S_over_T=med(rat), share_ratio_lt_07=round(float(np.mean([x < 0.7 for x in rat])), 3) if rat else None,
                      median_abs_drop=med([r["ref_k5"]["R_T"] - r["ref_k5"]["R_S"] for r in feas]),
                      by_layer={L: med([r["ref_k5"]["R_S"] / r["ref_k5"]["R_T"] for r in feas if r["L"] == L and r["ref_k5"]["R_T"] > 0]) for L in layers},
                      naive_ratio=med([r["naive_k5"]["R_S"] / r["naive_k5"]["R_T"] for r in feas if "naive_k5" in r and r["naive_k5"]["R_T"] > 0]),
                      all_cells_median_R_T_k5=med([r["ref_k5"]["R_T"] for r in meas]),
                      all_cells_median_R_S_k5=med([r["ref_k5"]["R_S"] for r in meas]))
    # ---------------- keep-state (city_capital)
    ks = [r for r in meas if r["fam"] == "city_capital" and "refks_k5" in r]
    if ks:
        T["keep_state"] = dict(
            n=len(ks),
            ref_R_S=med([r["ref_k5"]["R_S"] for r in ks]), ref_P_ks=med([r["ref_k5"].get("P_ks") for r in ks]),
            ref_R_S_ks=med([r["ref_k5"].get("R_S_ks") for r in ks]),
            refks_R_S_ks=med([r["refks_k5"].get("R_S_ks") for r in ks]), refks_P_ks=med([r["refks_k5"].get("P_ks") for r in ks]),
            naive_R_S_ks=med([r["naive_k5"].get("R_S_ks") for r in ks if "naive_k5" in r]),
            naive_P_ks=med([r["naive_k5"].get("P_ks") for r in ks if "naive_k5" in r]),
            share_refks_ge05=round(float(np.mean([r["refks_k5"]["R_S_ks"] >= 0.5 for r in ks])), 3),
            pairs_feasible_ks=round(float(np.mean([max(r["refks_k5"]["R_S_ks"] for r in ks if r["group"] == g) >= 0.5
                                                   for g in {r["group"] for r in ks}])), 3),
            naive_ratio_ks=med([r["naive_k5"]["R_S_ks"] / r["refks_k5"]["R_S_ks"] for r in ks
                                if "naive_k5" in r and r["refks_k5"]["R_S_ks"] >= 0.5]),
            naive_q_ks=round(float(np.mean([r["naive_k5"]["R_S_ks"] >= 0.5 * r["refks_k5"]["R_S_ks"] for r in ks
                                            if "naive_k5" in r and r["refks_k5"]["R_S_ks"] >= 0.5])), 3)
            if any(r["refks_k5"]["R_S_ks"] >= 0.5 for r in ks) else None)
    # ---------------- country_capital (reported separately)
    if cc:
        T["country_capital"] = {f"L{L}": dict(n=len([r for r in cc if r["L"] == L]),
                                              median_R_S=med([r["ref_k5"]["R_S"] for r in cc if r["L"] == L]),
                                              share_ge05=share([r for r in cc if r["L"] == L], "R_S", 0.5),
                                              share_ge07=share([r for r in cc if r["L"] == L], "R_S", 0.7))
                                for L in sorted({r["L"] for r in cc})}
    # ---------------- per family table at each layer (median R_S k=5)
    T["median_R_S_k5_family_layer"] = {f: {f"L{L}": med([r["ref_k5"]["R_S"] for r in meas if r["fam"] == f and r["L"] == L])
                                           for L in layers} for f in fams}
    # ---------------- error term: share of attribution, top-50 ablation effect
    cl = [json.load(open(c["_file"])) for c in cells if c["seed"] == 0 and c["family"] != "country_capital"]
    T["error_term"] = {L: dict(
        err_share_t=med([c["attr_err_t"] / (c["attr_err_t"] + c["attr_lat_t"]) for c in cl if c["layer"] == L
                         and (c["attr_err_t"] + c["attr_lat_t"]) != 0]),
        top50_E_S=med([metrics(c, "contr_k50", kap)["E_S"] for c in cl if c["layer"] == L and "contr_k50" in c["margins"]]),
        top50_P_S=med([metrics(c, "contr_k50", kap)["P_S"] for c in cl if c["layer"] == L and "contr_k50" in c["margins"]]),
        top50_KL=med([metrics(c, "contr_k50", kap)["KL"] for c in cl if c["layer"] == L and "contr_k50" in c["margins"]]))
        for L in layers}
    # ---------------- bootstrap CIs for ref_k5 R_S on measurable cells
    ci = {}
    for c in cl:
        if c["n"]["tS"] >= 20 and c["n"]["sS"] >= 40:
            lo, hi = boot_R(c, "ref_k5", kappa=kap)
            ci[f"{c['family']}:{c['group']}:L{c['layer']}"] = [round(metrics(c, "ref_k5", kap)["R_S"], 3), round(lo, 3), round(hi, 3)]
    T["ref_k5_R_S_bootstrap"] = ci
    T["median_ci_width"] = med([v[2] - v[1] for v in ci.values()])
    # ---------------- generalisation: example Effect vs held-out Effect (reference, k=5 prefix)
    gen = []
    for c in cl:
        s5 = c["sets"]["ref_k5"]
        if not s5 or not c["greedy"]:
            continue
        e_ex = c["greedy"][len(s5) - 1]["E_ex"]
        if e_ex > 0:
            gen.append(metrics(c, "ref_k5", kap)["E_S"] / e_ex)
    T["heldout_over_example_effect"] = med(gen)
    # ---------------- feature splitting: entity coverage of chosen latents
    fs = defaultdict(list)
    for c in cl:
        ls = c.get("latent_stats", {})
        for nm in ("ref_k5", "naive_k5"):
            for i in c["sets"].get(nm, []):
                d = ls.get(str(i))
                if not d:
                    continue
                cov = d["ent_cov"][0] / max(1, d["ent_cov"][1])
                fs[nm].append(dict(cov=cov, few=d["ent_cov"][0] <= 2, wiki=d["wiki_tok_freq"],
                                   sib=d.get("sS", [0, 0])[0], tgt=d.get("tS", [0, 0])[0], last=d.get("tS", [0, 0])[1],
                                   fam=c["family"], L=c["layer"]))
    T["latent_stats"] = {nm: dict(n=len(v), share_fire_le2_entities=round(float(np.mean([x["few"] for x in v])), 3),
                                  median_entity_coverage=med([x["cov"] for x in v]),
                                  median_target_fire=med([x["tgt"] for x in v]), median_sibling_fire=med([x["sib"] for x in v]),
                                  share_wiki_freq_gt_1pct=round(float(np.mean([x["wiki"] > 0.01 for x in v])), 3),
                                  share_sibling_fire_gt_target=round(float(np.mean([x["sib"] >= x["tgt"] for x in v])), 3))
                         for nm, v in fs.items()}
    # ---------------- memorisation: does the reference pick the same latents for the same cell under new splits?
    if a.seeds:
        sc = load(a.seeds) + [c for c in cells if c["seed"] == 0]
        byc = defaultdict(dict)
        for c in sc:
            byc[(c["family"], c["group"], c["layer"])][c["seed"]] = c
        jac, top1same, cross, rs = [], [], [], []
        for key, d in byc.items():
            if len(d) < 2:
                continue
            ss = [set(d[k]["sets"]["ref_k5"]) for k in sorted(d)]
            firsts = [d[k]["greedy"][0]["latent"] if d[k]["greedy"] else None for k in sorted(d)]
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    if ss[i] and ss[j]:
                        jac.append(len(ss[i] & ss[j]) / len(ss[i] | ss[j]))
                        top1same.append(firsts[i] == firsts[j])
            rs.append([round(metrics(d[k], "ref_k5", kap)["R_S"], 3) for k in sorted(d)])
        # overlap between DIFFERENT groups of the same family and layer (seed 0): are the latents group-specific?
        s0 = [c for c in cells if c["seed"] == 0 and c["family"] != "country_capital"]
        for i, c1 in enumerate(s0):
            for c2 in s0[i + 1:]:
                if c1["family"] == c2["family"] and c1["layer"] == c2["layer"]:
                    a1, a2 = set(c1["sets"]["ref_k5"]), set(c2["sets"]["ref_k5"])
                    if a1 and a2:
                        cross.append(len(a1 & a2) / len(a1 | a2))
        T["memorisation"] = dict(n_cells_with_2plus_seeds=sum(1 for d in byc.values() if len(d) >= 2),
                                 median_jaccard_same_cell=med(jac), mean_jaccard_same_cell=round(float(np.mean(jac)), 3) if jac else None,
                                 share_same_first_latent=round(float(np.mean(top1same)), 3) if top1same else None,
                                 mean_jaccard_different_groups=round(float(np.mean(cross)), 3) if cross else None,
                                 share_pairs_any_overlap_diff_groups=round(float(np.mean([x > 0 for x in cross])), 3) if cross else None,
                                 R_S_across_seeds_examples=rs[:12],
                                 median_sd_R_across_seeds=med([float(np.std(r)) for r in rs if len(r) >= 2]))
    json.dump(dict(T=T, rows=rows), open(a.out, "w"), default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    print(json.dumps(T, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
