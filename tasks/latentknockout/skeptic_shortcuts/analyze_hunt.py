"""Shortcut hunt (CPU): score cheap recipes and scripted policies on the recommended LatentKnockout-Verify cells.

Inputs: hunt.py cells (held-out margins + self-test margins for every recipe set) and the feasibility sweep cells
(reference ref_k5 R_S and the greedy trajectory). Outputs a JSON of tables and a markdown summary.

Slot rule (FEASIBILITY.md section 14): a submitted set passes if held-out R_S >= 0.5; "cannot" passes only on a null
slot. Episode = n in {3,4,5} slots, each slot null with probability p ~ U(0.3, 0.5), slots drawn independently from the
feasible / null pools; pass = every slot right. With per-slot accuracies a_f (feasible) and a_n (null) the episode
pass rate is E_{n,p}[(p a_n + (1-p) a_f)^n] (computed exactly on a grid). CIs: 95% bootstrap over (family, group) pairs
(both layers of a pair resampled together), thresholds re-chosen inside each resample where a policy has one.
Thresholded policies choose t by leave-one-pair-out CV (t never sees the held-out pair).

usage: python -m tasks.latentknockout.skeptic_shortcuts.analyze_hunt <hunt_cells_dir> <sweep_dir> --out <json> --md <md>
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

from tasks.latentknockout.analyze import metrics, wilson

TAU = 0.5
KAPPA = 0.1
P_GRID = np.linspace(0.3, 0.5, 21)
NS = (3, 4, 5)


def ci(t):
    return "[" + ", ".join("-" if x is None else f"{float(x):.2f}" for x in t) + "]"


def ep_pass(a_f, a_n):
    return float(np.mean([np.mean([(p * a_n + (1 - p) * a_f) ** n for p in P_GRID]) for n in NS]))


def ep_pass_n(a_f, a_n, n):
    return float(np.mean([(p * a_n + (1 - p) * a_f) ** n for p in P_GRID]))


def st_metrics(c, name):
    roles = np.array([it[0] for it in c["st_items"]])
    m = np.array(c["st_margins"][name], np.float32)
    fac = max(0.0, 1 - c["st_klproxy"][name] / KAPPA)
    out = {}
    for s in ("A", "B"):
        t, sb = m[roles == s + "t"], m[roles == s + "s"]
        E = float((t < 0).mean()) if len(t) else 0.0
        P = float((sb > 0).mean()) if len(sb) else 1.0
        out["R_" + s] = E * P * fac
    # a budgeted self-test: only the first 10 target and 10 sibling prompts of B (20 forward passes per candidate)
    t, sb = m[roles == "Bt"][:10], m[roles == "Bs"][:10]
    out["R_Bsmall"] = (float((t < 0).mean()) if len(t) else 0.0) * (float((sb > 0).mean()) if len(sb) else 1.0) * fac
    t = m[np.isin(roles, ["At", "Bt"])]
    sb = m[np.isin(roles, ["As", "Bs"])]
    out["R_AB"] = float((t < 0).mean()) * float((sb > 0).mean()) * fac
    return out


def load(hdir, sdir):
    cells = []
    for f in sorted(glob.glob(os.path.join(hdir, "*.json"))):
        c = json.load(open(f))
        o = json.load(open(os.path.join(sdir, "cells", f"{c['family']}__{c['group'].replace(' ', '_')}__L{c['layer']}__s0.json")))
        c["ref_R"] = metrics(o, "ref_k5")["R_S"]
        c["orig_sets"] = o["sets"]
        c["R"] = {n: metrics(c, n)["R_S"] for n in c["set_names"]}
        c["comp"] = {n: metrics(c, n) for n in c["set_names"]}
        c["ST"] = {n: st_metrics(c, n) for n in c["set_names"] if n in c["st_margins"]}
        c["pair"] = (c["family"], c["group"])
        cells.append(c)
    return cells


# ------------------------------------------------------------------------------------------------ policies
# A policy maps a cell (and, for thresholded ones, a threshold) to the slot decision: (submitted set name or None).
def slot_right(c, choice):
    if choice is None:
        return c["label"] == "null"
    return c["R"].get(choice, 0.0) >= TAU


def exJ(c):
    """Best example objective J of the reference's greedy prefixes (k <= 5) and the prefix name."""
    g = c["greedy"][:5]
    if not g:
        return 0.0, None
    j = int(np.argmax([t["J"] for t in g]))
    return g[j]["J"], f"pref_k{j + 1}"


def make_selftest(cands, which):
    def score(c):
        best, bn = -1.0, None
        for n in cands:
            if n in c["ST"]:
                v = c["ST"][n][which]
                if v > best:
                    best, bn = v, n
        return best, bn
    return score


def thresholded(cells, scorer, ts=np.linspace(0, 1, 101), w_null=0.4):
    """Leave-one-pair-out CV: for each pair, choose t on the other pairs (max weighted slot accuracy), apply."""
    sc = [scorer(c) for c in cells]
    s = np.array([v for v, _ in sc])
    r_sub = np.array([slot_right(c, n) for c, (_, n) in zip(cells, sc)], float)
    r_can = np.array([c["label"] == "null" for c in cells], float)
    isn = r_can.astype(bool)
    pr = np.array([hash(c["pair"]) for c in cells])
    sub = s[None, :] >= ts[:, None]                               # [n_t, n_cells]
    right = np.where(sub, r_sub[None, :], r_can[None, :])
    out = {}
    for p in np.unique(pr):
        tr = pr != p
        f, n = tr & ~isn, tr & isn
        acc = (1 - w_null) * (right[:, f].mean(1) if f.any() else 0) + w_null * (right[:, n].mean(1) if n.any() else 0)
        acc = acc - 1e-6 * np.abs(ts - 0.5)                        # ties -> closest to 0.5
        t = ts[int(np.argmax(acc))]
        for k in np.flatnonzero(pr == p):
            out[id(cells[k])] = sc[k][1] if s[k] >= t else None
    return out


FAMS = ["athlete_sport", "city_capital", "city_state", "country_lang", "langid"]


def lr_policy(cells, feat_fn, choose_fn):
    """Leave-one-pair-out logistic regression: predict whether submitting choose_fn(c) is right (its held-out R >= tau);
    submit it if p >= 0.5, else answer 'cannot'. Labels come from the training pairs only."""
    from sklearn.linear_model import LogisticRegression
    X = np.array([feat_fn(c) for c in cells], np.float64)
    ch = [choose_fn(c) for c in cells]
    y = np.array([c["R"].get(n, 0.0) >= TAU for c, n in zip(cells, ch)], int)
    pairs = sorted({c["pair"] for c in cells})
    out = {}
    for p in pairs:
        tr = np.array([c["pair"] != p for c in cells])
        te = ~tr
        if len(set(y[tr])) < 2:
            pr = np.full(te.sum(), float(y[tr].mean()))
        else:
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            m = LogisticRegression(C=1.0, max_iter=500).fit((X[tr] - mu) / sd, y[tr])
            pr = m.predict_proba((X[te] - mu) / sd)[:, 1]
        for c, n, q in zip([c for c, t in zip(cells, te) if t], [n for n, t in zip(ch, te) if t], pr):
            out[id(c)] = n if q >= 0.5 else None
    return out


def rates(cells, decisions):
    f = [slot_right(c, decisions[id(c)]) for c in cells if c["label"] == "feasible"]
    n = [slot_right(c, decisions[id(c)]) for c in cells if c["label"] == "null"]
    return float(np.mean(f)), float(np.mean(n)), len(f), len(n), sum(f), sum(n)


def boot_policy(cells, make_decisions, B=200, seed=0):
    """Pair-cluster bootstrap of the episode pass rate; decisions recomputed inside each resample."""
    rng = np.random.default_rng(seed)
    pairs = sorted({c["pair"] for c in cells})
    by = defaultdict(list)
    for c in cells:
        by[c["pair"]].append(c)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(pairs), len(pairs))
        cs = []
        for j, i in enumerate(pick):
            for c in by[pairs[i]]:
                cc = dict(c)
                # copies keep their pair key, so leave-one-pair-out CV drops every copy of the held-out pair
                cs.append(cc)
        if not any(c["label"] == "feasible" for c in cs) or not any(c["label"] == "null" for c in cs):
            continue
        d = make_decisions(cs)
        af, an = rates(cs, d)[:2]
        vals.append(ep_pass(af, an))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hunt")
    ap.add_argument("sweep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--fame", default="")
    a = ap.parse_args()
    cells = load(a.hunt, a.sweep)
    feas = [c for c in cells if c["label"] == "feasible"]
    null = [c for c in cells if c["label"] == "null"]
    T = dict(n_feasible=len(feas), n_null=len(null), n_pairs=len({c["pair"] for c in cells}))
    # ---------------------------------------------------------------- 1. recipe sets: share of reference, pass rate
    names = sorted({n for c in cells for n in c["set_names"]} - {"clean"})
    rec = {}
    for n in names:
        fs = [c for c in feas if n in c["R"]]
        ns = [c for c in null if n in c["R"]]
        if not fs:
            continue
        ratio = [c["R"][n] / c["ref_R"] for c in fs]
        q = sum(r >= 0.5 for r in ratio)
        pf = sum(c["R"][n] >= TAU for c in fs)
        pn = sum(c["R"][n] >= TAU for c in ns)
        rec[n] = dict(n_feas=len(fs), median_share_of_ref=round(float(np.median(ratio)), 3),
                      q_half_ref=round(q / len(fs), 3), q_ci=wilson(q, len(fs)),
                      pass_feas=round(pf / len(fs), 3), pass_feas_ci=wilson(pf, len(fs)),
                      n_null=len(ns), null_R_ge_tau=pn,
                      by_layer={L: dict(q=round(float(np.mean([c["R"][n] / c["ref_R"] >= 0.5 for c in fs if c["layer"] == L])), 3)
                                        if any(c["layer"] == L for c in fs) else None,
                                        pass_=round(float(np.mean([c["R"][n] >= TAU for c in fs if c["layer"] == L])), 3)
                                        if any(c["layer"] == L for c in fs) else None) for L in (12, 18)},
                      has_ref_first=round(float(np.mean([bool(c["greedy"]) and c["greedy"][0]["latent"] in c["sets"][n]
                                                         for c in fs])), 3),
                      median_E=round(float(np.median([c["comp"][n]["E_S"] for c in fs])), 3),
                      median_P=round(float(np.median([c["comp"][n]["P_S"] for c in fs])), 3),
                      median_KL=round(float(np.median([c["comp"][n]["KL"] for c in fs])), 4))
    T["recipes"] = rec
    T["ref_pass_feas"] = 1.0
    # oracle over cheap no-search recipes (choose on held-out: an upper bound for "one of the recipes works")
    cheap = [n for n in names if not n.startswith(("pref_", "mem_"))]
    orc = [max(c["R"][n] for n in cheap if n in c["R"]) for c in feas]
    T["cheap_oracle_pass_feas"] = round(float(np.mean([o >= TAU for o in orc])), 3)
    T["cheap_oracle_null_R_ge_tau"] = int(sum(max(c["R"][n] for n in cheap if n in c["R"]) >= TAU for c in null))
    T["null_cells_where_some_cheap_set_passes"] = [f"{c['family']}:{c['group']}:L{c['layer']}:" +
                                                   max(cheap, key=lambda n: c["R"].get(n, -1)) + f"={max(c['R'].get(n, -1) for n in cheap):.2f}"
                                                   for c in null if max(c["R"].get(n, -1) for n in cheap) >= TAU]
    # ---------------------------------------------------------------- 2. slot policies and episode pass rates
    pols = {}

    def add(name, make, desc):
        d = make(cells)
        af, an, nf, nn, kf, kn = rates(cells, d)
        lo, hi = boot_policy(cells, make, B=a.boot) if a.boot else (None, None)
        # the design's dial: enrich "deceptive" slots (nulls that look solved on the examples, J >= 0.6)
        dn = [slot_right(c, d[id(c)]) for c in cells if c["label"] == "null" and exJ(c)[0] >= 0.6]
        ep_dec = round(ep_pass(af, float(np.mean(dn))), 3) if dn else None
        pols[name] = dict(desc=desc, slot_feas=round(af, 3), slot_feas_ci=wilson(kf, nf), slot_null=round(an, 3),
                          slot_null_deceptive=f"{sum(dn)}/{len(dn)}", episode_pass_deceptive_nulls_only=ep_dec,
                          slot_null_ci=wilson(kn, nn), episode_pass=round(ep_pass(af, an), 3),
                          episode_pass_ci=[round(lo, 3) if lo is not None else None, round(hi, 3) if hi is not None else None],
                          episode_pass_4slot=round(ep_pass_n(af, an, 4), 3))

    def const(choice):
        return lambda cs: {id(c): choice for c in cs}
    add("always_cannot", const(None), "answer 'cannot' on every slot")
    for n in ("contr_k5", "naive_k5", "cos_k5", "actdiff_k5", "ll_contr_k5", "dla_k5"):
        add(f"always_{n}", const(n), f"submit {n} on every slot")
    # best fixed recipe per layer (cosine at 18, contrastive at 12): a 'layer rule'
    add("always_layer_rule", lambda cs: {id(c): ("cos_k5" if c["layer"] == 18 else "contr_k5") for c in cs},
        "submit cos_k5 at layer 18 and contr_k5 at layer 12")
    # the report's own verifier: trust the example objective of the reference search
    add("exJ_verifier", lambda cs: thresholded(cs, exJ),
        "reference greedy; submit its best prefix if the example objective J >= t, else 'cannot' (t by CV)")
    # cheap verifiers: no search at all, only rank -> test a few sets on self-made prompts -> threshold
    rank_only = [f"{r}_k{k}" for r in ("contr", "naive", "cos") for k in range(1, 6)]
    all_c = rank_only + [f"pref_k{k}" for k in range(1, 6)] + ["actdiff_k5", "actdiff_last_k5", "ent_actdiff_k5",
                                                                "ll_contr_k5", "ll_active_k5", "dla_k5"]
    for which in ("R_A", "R_B", "R_AB"):
        add(f"selftest_rank_{which}", lambda cs, w=which: thresholded(cs, make_selftest(rank_only, w)),
            f"15 ranked sets (contr/naive/cos top-1..5), pick best on self-test {which[2:]}, submit if >= t else 'cannot'")
        add(f"selftest_all_{which}", lambda cs, w=which: thresholded(cs, make_selftest(all_c, w)),
            f"ranked sets + greedy prefixes + activation/logit-lens sets, best on self-test {which[2:]}, threshold t")
    add("selftest_budget60_Bsmall", lambda cs: thresholded(cs, make_selftest(["contr_k5", "cos_k5", "naive_k5"], "R_Bsmall")),
        "3 ranked sets (contr/cos/naive top-5) each tested on 10 new-entity target + 10 sibling prompts "
        "(60 forward passes per slot); submit the best if >= t else 'cannot'")
    # contrastive top-5 tested on self-made prompts (single candidate, one threshold)
    add("selftest_contr5_AB", lambda cs: thresholded(cs, make_selftest(["contr_k5"], "R_AB")),
        "submit contr_k5 if its self-test R (A+B) >= t, else 'cannot'")
    # priors from generic statistics: family, layer, how famous the group name is (wikitext count), answer token id
    fame = json.load(open(a.fame)) if a.fame else None
    if fame:
        import math as _m
        fams_ = __import__("tasks.latentknockout.lk_data", fromlist=["x"]).families()

        def gen_feats(c):
            answ = fams_[c["family"]]["answers"][c["group"]][0]
            return ([float(c["family"] == f) for f in FAMS] + [float(c["layer"] == 18),
                    _m.log1p(fame["count"][c["group"]]), _m.log(fame["token_id"][answ])])
        layer_rule = lambda c: "cos_k5" if c["layer"] == 18 else "contr_k5"  # noqa: E731
        add("prior_textstats_layer_rule", lambda cs: lr_policy(cs, gen_feats, layer_rule),
            "logistic prior on family, layer, group-name frequency, answer token id (no model run) decides "
            "'cannot'; otherwise submit cos_k5 (L18) / contr_k5 (L12)")
        add("prior_family_layer_only", lambda cs: lr_policy(cs, lambda c: gen_feats(c)[:6], layer_rule),
            "same with family and layer only")
        st_rank = make_selftest(rank_only, "R_AB")
        add("prior_plus_selftest_rank", lambda cs: lr_policy(
            cs, lambda c: gen_feats(c) + [st_rank(c)[0]], lambda c: st_rank(c)[1]),
            "logistic on text statistics + best self-test R of the 15 ranked sets; submit that set or 'cannot'")
        st_all = make_selftest(all_c, "R_AB")
        add("prior_plus_selftest_all_plus_exJ", lambda cs: lr_policy(
            cs, lambda c: gen_feats(c) + [st_all(c)[0], exJ(c)[0]], lambda c: st_all(c)[1]),
            "logistic on text statistics + self-test R (all candidate sets) + example objective J")
    # in-distribution memorisers (lesson 3): the null label of a (family, group, layer) cell is fixed, so a policy that
    # has seen the cell before knows it; IDs are permuted per episode, so on feasible cells it must still find a set
    add("memo_label_layer_rule", lambda cs: {id(c): (None if c["label"] == "null" else
                                                     ("cos_k5" if c["layer"] == 18 else "contr_k5")) for c in cs},
        "remember which cells are null (seen before, labels only); else cos_k5 (L18) / contr_k5 (L12)")
    add("memo_label_best_cheap_recipe", lambda cs: {id(c): (None if c["label"] == "null" else
                                                            max(cheap, key=lambda n: c["R"].get(n, -1))) for c in cs},
        "remember null cells AND which cheap ranking worked for this cell (reward only, ID-free)")
    T["policies"] = pols
    # ---------------------------------------------------------------- 3. priors from generic statistics (no model)
    fam_null = {f: round(float(np.mean([c["label"] == "null" for c in cells if c["family"] == f])), 3)
                for f in sorted({c["family"] for c in cells})}
    lay_null = {L: round(float(np.mean([c["label"] == "null" for c in cells if c["layer"] == L])), 3) for L in (12, 18)}
    T["null_share_by_family"] = fam_null
    T["null_share_by_layer"] = lay_null
    # deceptive cells: null with high example objective, feasible with low
    T["deceptive"] = dict(
        null_exJ_ge_06=[f"{c['family']}:{c['group']}:L{c['layer']} J={exJ(c)[0]:.2f}" for c in null if exJ(c)[0] >= 0.6],
        feas_exJ_lt_05=[f"{c['family']}:{c['group']}:L{c['layer']} J={exJ(c)[0]:.2f} R={c['ref_R']:.2f}" for c in feas if exJ(c)[0] < 0.5])
    # per-cell table of the main shortcuts
    T["cells"] = [dict(cell=f"{c['family']}:{c['group']}:L{c['layer']}", label=c["label"], ref=round(c["ref_R"], 2),
                       exJ=round(exJ(c)[0], 2),
                       **{n: round(c["R"].get(n, float('nan')), 2) for n in ("contr_k5", "cos_k5", "naive_k5", "actdiff_k5",
                                                                           "actdiff_last_k5", "ent_actdiff_k5", "ent_mostact_k5",
                                                                           "last_mostact_k5", "fmt_k5", "ll_raw_k5",
                                                                           "ll_contr_k5", "ll_active_k5", "dla_k5",
                                                                           "mem_seed1", "mem_xfam")},
                       st_best_rank_AB=round(make_selftest(rank_only, "R_AB")(c)[0], 2)) for c in cells]
    json.dump(T, open(a.out, "w"), indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o))
    # ---------------------------------------------------------------- markdown
    L = [f"# Shortcut hunt tables ({len(feas)} feasible + {len(null)} null cells, {T['n_pairs']} pairs)\n",
         "## Recipes (feasible cells; pass = held-out R_S >= 0.5)\n",
         "| Recipe | share of ref R (median) | q = share >= half ref [CI] | pass rate [CI] | L12 pass | L18 pass | E / P / KL | contains ref's 1st latent | nulls with R >= 0.5 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, r in sorted(rec.items(), key=lambda kv: -kv[1]["pass_feas"]):
        L.append(f"| {n} | {r['median_share_of_ref']} | {r['q_half_ref']} {ci(r['q_ci'])} | {r['pass_feas']} {ci(r['pass_feas_ci'])} "
                 f"| {r['by_layer'][12]['pass_']} | {r['by_layer'][18]['pass_']} | {r['median_E']} / {r['median_P']} / {r['median_KL']} "
                 f"| {r['has_ref_first']} | {r['null_R_ge_tau']}/{r['n_null']} |")
    L += ["", f"Cheap-recipe oracle (best no-search recipe per cell, chosen on held-out): passes {T['cheap_oracle_pass_feas']} of "
          f"feasible cells; reaches 0.5 on {T['cheap_oracle_null_R_ge_tau']} null cells: {T['null_cells_where_some_cheap_set_passes']}", "",
          "## Scripted policies (slot accuracy; episode = 3-5 slots, 30-50% null, pass = all slots right)\n",
          "| Policy | feasible slots right [CI] | null slots right [CI] | episode pass [95% pair-bootstrap CI] | 4-slot pass | deceptive nulls right | episode pass, deceptive nulls only |",
          "|---|---|---|---|---|---|---|"]
    for n, p in sorted(pols.items(), key=lambda kv: -kv[1]["episode_pass"]):
        L.append(f"| {n}: {p['desc']} | {p['slot_feas']} {ci(p['slot_feas_ci'])} | {p['slot_null']} {ci(p['slot_null_ci'])} "
                 f"| **{p['episode_pass']}** {ci(p['episode_pass_ci'])} | {p['episode_pass_4slot']} "
                 f"| {p['slot_null_deceptive']} | {p['episode_pass_deceptive_nulls_only']} |")
    L += ["", f"Null share by family: {fam_null}; by layer: {lay_null}", "",
          f"Deceptive nulls (example J >= 0.6): {T['deceptive']['null_exJ_ge_06']}",
          f"Feasible cells with example J < 0.5: {T['deceptive']['feas_exJ_lt_05']}"]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
