"""Bank-F validity audit, part 2: unblinded analysis (CPU; caches memory-mapped, only needed columns sliced).

Reads the blind ratings (ratings_bankF.jsonl, ratings_splitA.jsonl), the unblind key, key_check.jsonl, the bank-F
activation cache (diagnosis/cache/bank_acts_L*_F.npy) and the split-A cache of the v2 generator (read-only).
Writes results.json and per_concept.csv in this directory. See PROTOCOL.md for the bar and the analyses.

Run (from ~/wt/fmdiag, after sourcing common/env.sh):
  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=2G -- \
      $PY tasks/featurematch/diagnosis/verify_banks/analyze.py
"""
import csv
import json
import math
import os
import random
import sys

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
SRC = "/home/ec2-user/wt/featurematch/tasks/featurematch/cache"
sys.path.insert(0, HERE)
import terms  # noqa: E402

SEED = 20261002
NBOOT = 10_000
LAYERS = (6, 12, 18)
GROUPS = ("KEPT", "DROPLOW")


# ------------------------------------------------------------------------------------------------------- stats
def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def cluster_boot(per_concept, nboot=NBOOT):
    """per_concept: list of (successes, trials). Ratio estimator sum(s)/sum(n), resampling concepts."""
    rng = random.Random(SEED)
    k = len(per_concept)
    vals = []
    for _ in range(nboot):
        idx = rng.choices(range(k), k=k)
        s = sum(per_concept[i][0] for i in idx)
        n = sum(per_concept[i][1] for i in idx)
        vals.append(s / n if n else float("nan"))
    vals = np.array(vals)
    return [round(float(np.nanpercentile(vals, 2.5)), 4), round(float(np.nanpercentile(vals, 97.5)), 4)]


def cluster_boot_diff(a, b, nboot=NBOOT):
    """Difference of ratio estimators b - a, resampling concepts within each group."""
    rng = random.Random(SEED)
    vals = []
    for _ in range(nboot):
        ia = rng.choices(range(len(a)), k=len(a))
        ib = rng.choices(range(len(b)), k=len(b))
        ra = sum(a[i][0] for i in ia) / max(1, sum(a[i][1] for i in ia))
        rb = sum(b[i][0] for i in ib) / max(1, sum(b[i][1] for i in ib))
        vals.append(rb - ra)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def rate_block(per_concept):
    s = sum(x[0] for x in per_concept)
    n = sum(x[1] for x in per_concept)
    return {"rate": round(s / n, 4) if n else None, "k": s, "n": n, "n_concepts": len(per_concept),
            "ci95_cluster_boot": cluster_boot(per_concept) if n else None, "ci95_wilson_ignores_clustering": wilson(s, n)}


def boot_median(xs, nboot=NBOOT):
    rng = random.Random(SEED)
    xs = list(xs)
    v = [float(np.median(rng.choices(xs, k=len(xs)))) for _ in range(nboot)]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def auroc_from_ranks(ranks_pos, n_pos, n_neg):
    """Mann-Whitney AUROC from the summed ranks (average ranks for ties) of the positives."""
    return (ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ------------------------------------------------------------------------------------------------------- load
def load():
    sample = json.load(open(os.path.join(HERE, "sample_blinded.json")))
    key = json.load(open(os.path.join(HERE, "unblind_key.json")))
    rF = {r["bid"]: r for r in map(json.loads, open(os.path.join(HERE, "ratings_bankF.jsonl")))}
    rA = {r["bid"]: r for r in map(json.loads, open(os.path.join(HERE, "ratings_splitA.jsonl")))}
    meta = json.load(open(os.path.join(DIAG, "cache", "bank_meta.json")))
    kc = [json.loads(l) for l in open(os.path.join(DIAG, "key_check.jsonl"))]
    concepts = {c["cid"]: c for c in json.load(open(os.path.join(SRC, "concepts.json")))}
    pmeta = json.load(open(os.path.join(SRC, "precompute_meta.json")))
    assert pmeta["cids"] == meta["cids"]
    return sample, key, rF, rA, meta, kc, concepts, pmeta


def splitA_pick():
    """The 5 split-A indices per concept that were rated (same draw as the rating viewer)."""
    sample = json.load(open(os.path.join(HERE, "sample_blinded.json")))
    rng = random.Random(SEED)
    return {c["bid"]: sorted(rng.sample(range(40), 5)) for c in sample["concepts"]}


# ------------------------------------------------------------------------------------------------------- main
def main():
    sample, key, rF, rA, meta, kc, concepts, pmeta = load()
    cids = meta["cids"]
    cidx = {c: i for i, c in enumerate(cids)}
    rowsF = np.asarray(meta["banks"]["F"]["rows"])
    stylesF = meta["banks"]["F"]["styles"]
    rowsA = np.asarray(pmeta["rowsA"])
    pickA = splitA_pick()
    S = sample["concepts"]
    group = {c["bid"]: key[c["bid"]] for c in S}
    out = {"n_concepts": {g: sum(v == g for v in group.values()) for g in GROUPS}}

    # ---------------- per text table (bank F) with manual codes and scripted flags
    texts = []   # dicts
    for c in S:
        codes = rF[c["bid"]]["codes"]
        for i, (t, code) in enumerate(zip(c["bankF"], codes)):
            texts.append({"bid": c["bid"], "cid": c["cid"], "group": group[c["bid"]], "i": i, "style": t["style"],
                          "about": code[0], "inst": code[1] == "1", "famous": code[2] == "y",
                          "term": terms.has_term(c["cid"], t["text"]), "enc": terms.enc_lead(t["text"]),
                          "len": len(t["text"])})
    for t in texts:
        t["named"] = t["inst"] or t["term"]
        t["clear_named"] = t["about"] == "c" and t["named"]

    def per_concept(pred, rows, grp=None):
        res = []
        for c in S:
            if grp and group[c["bid"]] != grp:
                continue
            rr = [r for r in rows if r["bid"] == c["bid"]]
            res.append((sum(bool(pred(r)) for r in rr), len(rr)))
        return res

    measures = {
        "clear_or_loose": lambda r: r["about"] in "cl",
        "clear": lambda r: r["about"] == "c",
        "loose": lambda r: r["about"] == "l",
        "off": lambda r: r["about"] == "o",
        "instance_named": lambda r: r["inst"],
        "instance_famous_real": lambda r: r["famous"],
        "term_named (scripted)": lambda r: r["term"],
        "term_or_instance_named": lambda r: r["named"],
        "clear_and_named": lambda r: r["clear_named"],
        "encyclopedic_lead (scripted)": lambda r: r["enc"],
    }
    bf = {}
    for m, f in measures.items():
        bf[m] = {"all": rate_block(per_concept(f, texts))}
        for g in GROUPS:
            bf[m][g] = rate_block(per_concept(f, texts, g))
        bf[m]["DROPLOW_minus_KEPT"] = {
            "diff": round(bf[m]["DROPLOW"]["rate"] - bf[m]["KEPT"]["rate"], 4),
            "ci95_cluster_boot": cluster_boot_diff(per_concept(f, texts, "KEPT"), per_concept(f, texts, "DROPLOW"))}
    out["bankF_ratings"] = bf
    # concepts with >= 18/20 clear-or-loose, and minimum per concept
    pc = per_concept(measures["clear_or_loose"], texts)
    out["bankF_ratings"]["per_concept_clear_or_loose_min"] = min(s / n for s, n in pc)
    pcc = {c["bid"]: s / n for c, (s, n) in zip(S, per_concept(measures["clear"], texts))}
    out["bankF_ratings"]["per_concept_clear_min"] = {"min": round(min(pcc.values()), 3),
                                                     "concepts_below_0.75": sorted(
                                                         f"{b} {group[b]} {next(c['label'] for c in S if c['bid'] == b)} {v:.2f}"
                                                         for b, v in pcc.items() if v < 0.75)}

    # ---------------- bar (PROTOCOL.md)
    col = {g: bf["clear_or_loose"][g]["rate"] for g in GROUPS}
    bar = {"clear_or_loose_KEPT": col["KEPT"], "clear_or_loose_DROPLOW": col["DROPLOW"],
           "each_group_ge_0.85": all(v >= 0.85 for v in col.values()),
           "DROPLOW_not_lower_than_KEPT_by_more_than_0.10": col["DROPLOW"] - col["KEPT"] >= -0.10,
           "clear_ge_0.60_each_group": all(bf["clear"][g]["rate"] >= 0.60 for g in GROUPS)}
    bar["passed"] = bar["each_group_ge_0.85"] and bar["DROPLOW_not_lower_than_KEPT_by_more_than_0.10"]
    out["bar"] = bar

    # ---------------- split A: manual (5 per concept) and scripted (all 40 per concept)
    rowsAm = []
    for c in S:
        for i, code in zip(pickA[c["bid"]], rA[c["bid"]]["codes"]):
            t = concepts[c["cid"]]["A"][i]
            rowsAm.append({"bid": c["bid"], "group": group[c["bid"]], "about": code[0], "inst": code[1] == "1",
                           "term": terms.has_term(c["cid"], t), "enc": terms.enc_lead(t)})
    rowsAs = []
    for c in S:
        for t in concepts[c["cid"]]["A"]:
            rowsAs.append({"bid": c["bid"], "group": group[c["bid"]], "term": terms.has_term(c["cid"], t),
                           "enc": terms.enc_lead(t), "len": len(t)})
    sa = {}
    for m in ("clear_or_loose", "clear", "off", "instance_named"):
        sa[m + " (manual, 5/concept)"] = {"all": rate_block(per_concept(measures[m], rowsAm)),
                                          **{g: rate_block(per_concept(measures[m], rowsAm, g)) for g in GROUPS}}
    for m in ("term_named (scripted)", "encyclopedic_lead (scripted)"):
        sa[m.replace("(scripted)", "(scripted, all 40/concept)")] = {
            "all": rate_block(per_concept(measures[m], rowsAs)),
            **{g: rate_block(per_concept(measures[m], rowsAs, g)) for g in GROUPS}}
    sa["median_chars"] = {"splitA": int(np.median([r["len"] for r in rowsAs])),
                          "bankF": int(np.median([len(t["text"]) for c in S for t in c["bankF"]]))}
    out["splitA"] = sa

    # ---------------- latent firing vs rating (pooled latents of the sampled concepts)
    pooled = {c["cid"]: [r for r in kc if r["c_star"] == c["cid"]] for c in S}
    lat_rows = []          # one row per pooled latent of a sampled concept
    fire_on = {}           # (layer, latent) -> bool[20] on the anchor's bank-F texts
    actsA_needed = {L: set() for L in LAYERS}
    for L in LAYERS:
        X = np.load(os.path.join(DIAG, "cache", f"bank_acts_L{L}_F.npy"), mmap_mode="r")
        cols = sorted({r["real_latent"] for c in S for r in pooled[c["cid"]] if r["layer"] == L})
        if not cols:
            continue
        sub = np.asarray(X[:, cols], dtype=np.float32)
        pos = {j: k for k, j in enumerate(cols)}
        for c in S:
            m = rowsF == cidx[c["cid"]]
            for r in pooled[c["cid"]]:
                if r["layer"] != L:
                    continue
                a = sub[m, pos[r["real_latent"]]]
                fire_on[(L, r["real_latent"], c["cid"])] = a > 0
                actsA_needed[L].add(r["real_latent"])
        del sub, X
    # fire rate on split A (anchor's 40 texts) for the same latents
    fireA = {}
    for L in LAYERS:
        cols = sorted(actsA_needed[L])
        if not cols:
            continue
        XA = np.load(os.path.join(SRC, f"acts_L{L}_A.npy"), mmap_mode="r")
        sub = np.asarray(XA[:, cols], dtype=np.float32)
        pos = {j: k for k, j in enumerate(cols)}
        for c in S:
            m = rowsA == cidx[c["cid"]]
            for r in pooled[c["cid"]]:
                if r["layer"] == L:
                    fireA[(L, r["real_latent"], c["cid"])] = float((sub[m, pos[r["real_latent"]]] > 0).mean())
        del sub, XA

    for c in S:
        tt = [t for t in texts if t["bid"] == c["bid"]]
        for r in pooled[c["cid"]]:
            f = fire_on[(r["layer"], r["real_latent"], c["cid"])]
            kind = "kept" if r["kept"] else ("dropped_lowF" if f.mean() <= 0.2 else "dropped_other")
            lat_rows.append({"bid": c["bid"], "group": group[c["bid"]], "cid": c["cid"], "layer": r["layer"],
                             "latent": r["real_latent"], "kind": kind, "fire_F": float(f.mean()),
                             "fire_A": fireA[(r["layer"], r["real_latent"], c["cid"])],
                             "auroc_F2": r["auroc_F2_cstar"], "auroc_C": r["auroc_C_cstar"],
                             "f": f, "tt": tt})

    def fire_by(pred, kind, grp=None):
        """Per concept: (fires, trials) summed over latents of `kind` and texts matching pred."""
        res = []
        for c in S:
            if grp and group[c["bid"]] != grp:
                continue
            s = n = 0
            for lr in lat_rows:
                if lr["bid"] != c["bid"] or lr["kind"] != kind:
                    continue
                for fi, t in zip(lr["f"], lr["tt"]):
                    if pred(t):
                        s += int(fi)
                        n += 1
            if n:
                res.append((s, n))
        return res

    lf = {"n_latents": {k: sum(lr["kind"] == k for lr in lat_rows) for k in ("kept", "dropped_lowF", "dropped_other")},
          "n_latents_by_group": {g: {k: sum(lr["kind"] == k and lr["group"] == g for lr in lat_rows)
                                     for k in ("kept", "dropped_lowF", "dropped_other")} for g in GROUPS}}
    cats = {"clear_and_named": lambda t: t["clear_named"],
            "clear_and_famous_real_instance": lambda t: t["about"] == "c" and t["famous"],
            "clear_unnamed": lambda t: t["about"] == "c" and not t["named"],
            "loose": lambda t: t["about"] == "l",
            "encyclopedia-style sentence (bank style)": lambda t: t["style"] == "encyclopedia-style sentence",
            "encyclopedic_lead (scripted)": lambda t: t["enc"],
            "term_named (scripted)": lambda t: t["term"],
            "no term (scripted)": lambda t: not t["term"],
            "length < 100 chars": lambda t: t["len"] < 100,
            "length 100-149 chars": lambda t: 100 <= t["len"] < 150,
            "length >= 150 chars": lambda t: t["len"] >= 150,
            "encyclopedia-style sentence AND term named": lambda t: t["style"] == "encyclopedia-style sentence" and t["term"],
            "all bank-F texts": lambda t: True}
    for kind in ("dropped_lowF", "kept", "dropped_other"):
        lf[kind] = {}
        for nm, p in cats.items():
            pc_ = fire_by(p, kind)
            lf[kind][nm] = rate_block(pc_) if pc_ else None
        # same latents on split A
        fa = [lr["fire_A"] for lr in lat_rows if lr["kind"] == kind]
        lf[kind]["fire_on_anchor_splitA_median"] = round(float(np.median(fa)), 4) if fa else None
    # per style, dropped_lowF and kept
    styles = sorted(set(stylesF))
    lf["per_style"] = {}
    for kind in ("dropped_lowF", "kept"):
        lf["per_style"][kind] = {st: rate_block(fire_by(lambda t, st=st: t["style"] == st, kind))["rate"]
                                 for st in styles}
    # within-concept contrast: concepts (either group) that have BOTH kept and dropped_lowF latents: same 20 texts
    both = [c["bid"] for c in S
            if any(lr["bid"] == c["bid"] and lr["kind"] == "kept" for lr in lat_rows)
            and any(lr["bid"] == c["bid"] and lr["kind"] == "dropped_lowF" for lr in lat_rows)]

    def fire_on_bids(kind, bids):
        res = []
        for b in bids:
            s_ = n_ = 0
            for lr in lat_rows:
                if lr["bid"] == b and lr["kind"] == kind:
                    s_ += int(lr["f"].sum())
                    n_ += len(lr["f"])
            res.append((s_, n_))
        return res
    lf["within_concept"] = {
        "concepts_with_both_kept_and_dropped_lowF_latents": len(both),
        "kept_latents_fire_rate_on_these_texts": rate_block(fire_on_bids("kept", both)) if both else None,
        "dropped_lowF_latents_fire_rate_on_the_same_texts": rate_block(fire_on_bids("dropped_lowF", both)) if both else None}
    pre = lf["dropped_lowF"]["clear_and_named"]["rate"]
    lf["prereg_secondary_2"] = {"dropped_lowF_fire_on_clear_and_named": pre, "below_0.30": pre < 0.30}
    out["latent_firing"] = lf

    # ---------------- model-side check: best of all 16384 latents, selected on F1, tested on F2 and split A
    f1 = np.zeros(len(rowsF), bool)
    seen = set()
    for k, (r, st) in enumerate(zip(rowsF, stylesF)):
        if (r, st) not in seen:
            f1[k] = True
            seen.add((r, st))
    f2 = ~f1
    ms = {}
    best = {c["cid"]: {} for c in S}
    for L in LAYERS:
        X = np.load(os.path.join(DIAG, "cache", f"bank_acts_L{L}_F.npy"), mmap_mode="r")
        for nm, mask in (("F1", f1), ("F2", f2)):
            sub = np.asarray(X[mask], dtype=np.float32)
            R = rankdata(sub, axis=0).astype(np.float32)   # [2320, 16384]
            rr = rowsF[mask]
            for c in S:
                pm = rr == cidx[c["cid"]]
                npos, nneg = int(pm.sum()), int((~pm).sum())
                au = auroc_from_ranks(R[pm].sum(0, dtype=np.float64), npos, nneg)
                best[c["cid"]][(L, nm)] = au
            del sub, R
        del X
    XA = {L: np.load(os.path.join(SRC, f"acts_L{L}_A.npy"), mmap_mode="r") for L in LAYERS}
    per_c = []
    for c in S:
        rec = {"bid": c["bid"], "group": group[c["bid"]]}
        cand = []
        for L in LAYERS:
            au1 = best[c["cid"]][(L, "F1")]
            j = int(np.argmax(au1))
            cand.append((float(au1[j]), L, j, float(best[c["cid"]][(L, "F2")][j])))
        au1, L, j, au2 = max(cand)
        # split-A AUROC of the selected latent (anchor's 40 texts vs all other concepts' split-A texts)
        col = np.asarray(XA[L][:, j], dtype=np.float32)
        rk = rankdata(col)
        pm = rowsA == cidx[c["cid"]]
        auA = float(auroc_from_ranks(rk[pm].sum(), int(pm.sum()), int((~pm).sum())))
        rec.update({"best_layer": L, "best_latent": j, "auroc_F1_selected": round(au1, 4),
                    "auroc_F2_heldout": round(au2, 4), "auroc_A_of_that_latent": round(auA, 4),
                    "pooled_best_auroc_F2": round(max([r["auroc_F2_cstar"] for r in pooled[c["cid"]]] or [float("nan")]), 4),
                    "is_pooled": any(r["layer"] == L and r["real_latent"] == j for r in pooled[c["cid"]]),
                    "kept_in_pool": any(r["layer"] == L and r["real_latent"] == j and r["kept"] for r in pooled[c["cid"]])})
        per_c.append(rec)
    for g in GROUPS:
        rows = [r for r in per_c if r["group"] == g]
        x2 = [r["auroc_F2_heldout"] for r in rows]
        xA = [r["auroc_A_of_that_latent"] for r in rows]
        xp = [r["pooled_best_auroc_F2"] for r in rows]
        ms[g] = {"n_concepts": len(rows),
                 "heldout_F2_auroc_of_F1_selected_latent_median": round(float(np.median(x2)), 4),
                 "median_ci95_boot": boot_median(x2),
                 "share_heldout_F2_ge_0.85": rate_block([(int(v >= 0.85), 1) for v in x2]),
                 "splitA_auroc_of_that_latent_median": round(float(np.median(xA)), 4),
                 "share_splitA_ge_0.85": rate_block([(int(v >= 0.85), 1) for v in xA]),
                 "best_pooled_latent_auroc_F2_median": round(float(np.median(xp)), 4),
                 "selected_latent_is_in_v2_pool": sum(r["is_pooled"] for r in rows),
                 "selected_latent_kept_by_A1.1": sum(r["kept_in_pool"] for r in rows)}
    # why is the bank-robust latent not in the v2 pool? apply generate.Tables.pools' criteria (read-only tables)
    P_THR, P_THR_B, MARGIN, FIRE_MIN, MAX_SPECIFIC = 0.90, 0.85, 0.10, 0.5, 3     # generate.py v2 constants
    eligible = np.array([bool(concepts[c]["C"]) for c in cids])
    why = []
    for rec, c in zip(per_c, S):
        L, j = rec["best_layer"], rec["best_latent"]
        aA = np.asarray(np.load(os.path.join(SRC, f"auroc_L{L}_A.npy"), mmap_mode="r")[j], dtype=np.float32)
        aB = np.asarray(np.load(os.path.join(SRC, f"auroc_L{L}_B.npy"), mmap_mode="r")[j], dtype=np.float32)
        fA = np.asarray(np.load(os.path.join(SRC, f"fire_L{L}_A.npy"), mmap_mode="r")[j], dtype=np.float32)
        nm = np.asarray(np.load(os.path.join(SRC, f"acts_L{L}_names.npy"), mmap_mode="r")[:, j], dtype=np.float32)
        dens = float(np.load(os.path.join(SRC, f"firerate_L{L}_A.npy"), mmap_mode="r")[j])
        b = int(aA.argmax())
        fails = []
        if b != cidx[c["cid"]]:
            fails.append("best_concept_on_A_is_other:" + cids[b])
        if aA[b] < P_THR:
            fails.append("auroc_A<0.90")
        if aB[b] < P_THR_B:
            fails.append("auroc_B<0.85")
        if fA[b] < FIRE_MIN:
            fails.append("fire_A<0.5")
        if int((aA >= aA[b] - MARGIN).sum()) > MAX_SPECIFIC:
            fails.append("more_than_3_concepts_within_0.10(generic)")
        if not eligible[b]:
            fails.append("not_eligible")
        own = nm.reshape(len(cids), 3)[b]
        if own.max() > 0:
            fails.append("fires_on_own_concept_name")
        rec.update({"pool_fail_reasons": fails, "names_act_max": round(float(own.max()), 3),
                    "density_A": round(dens, 4), "n_concepts_within_0.10": int((aA >= aA[b] - MARGIN).sum())})
        why.append(fails)
    from collections import Counter
    for g in GROUPS:
        rows = [r for r in per_c if r["group"] == g]
        cnt = Counter(f.split(":")[0] for r in rows for f in r["pool_fail_reasons"])
        ms[g]["why_not_in_pool (reasons, overlapping)"] = dict(cnt)
        ms[g]["fails_only_name_filter"] = sum(r["pool_fail_reasons"] == ["fires_on_own_concept_name"] for r in rows)
        ms[g]["passes_all_pool_criteria"] = sum(not r["pool_fail_reasons"] for r in rows)
        ms[g]["density_A_median_of_selected"] = round(float(np.median([r["density_A"] for r in rows])), 4)
    out["model_side_check"] = ms

    # ---------------- per-concept table
    with open(os.path.join(HERE, "per_concept.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bid", "group", "cid", "label", "bankF_clear", "bankF_loose", "bankF_off", "bankF_instance_named",
                    "bankF_famous_real", "bankF_term_named", "splitA_term_named", "splitA_manual_clear_of5",
                    "n_kept", "n_dropped_lowF", "n_dropped_other", "dropped_lowF_fire_F_mean",
                    "dropped_lowF_fire_A_mean", "kept_fire_F_mean", "best_any_latent_layer",
                    "best_any_latent_auroc_F1", "best_any_latent_auroc_F2_heldout", "best_any_latent_auroc_A",
                    "best_pooled_auroc_F2"])
        for c, pcr in zip(S, per_c):
            tt = [t for t in texts if t["bid"] == c["bid"]]
            am = [r for r in rowsAm if r["bid"] == c["bid"]]
            As = [r for r in rowsAs if r["bid"] == c["bid"]]
            L_ = [lr for lr in lat_rows if lr["bid"] == c["bid"]]
            dl = [lr for lr in L_ if lr["kind"] == "dropped_lowF"]
            kp = [lr for lr in L_ if lr["kind"] == "kept"]
            w.writerow([c["bid"], group[c["bid"]], c["cid"], c["label"],
                        sum(t["about"] == "c" for t in tt), sum(t["about"] == "l" for t in tt),
                        sum(t["about"] == "o" for t in tt), sum(t["inst"] for t in tt), sum(t["famous"] for t in tt),
                        sum(t["term"] for t in tt), round(np.mean([r["term"] for r in As]), 3),
                        sum(r["about"] == "c" for r in am), len(kp), len(dl),
                        sum(lr["kind"] == "dropped_other" for lr in L_),
                        round(float(np.mean([lr["fire_F"] for lr in dl])), 3) if dl else "",
                        round(float(np.mean([lr["fire_A"] for lr in dl])), 3) if dl else "",
                        round(float(np.mean([lr["fire_F"] for lr in kp])), 3) if kp else "",
                        pcr["best_layer"], pcr["auroc_F1_selected"], pcr["auroc_F2_heldout"],
                        pcr["auroc_A_of_that_latent"], pcr["pooled_best_auroc_F2"]])
    out["per_concept_model_side"] = per_c
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"bar": out["bar"], "clear_or_loose": {g: bf["clear_or_loose"][g] for g in GROUPS},
                      "latent_prereg_2": lf["prereg_secondary_2"]}, indent=1))


if __name__ == "__main__":
    main()
