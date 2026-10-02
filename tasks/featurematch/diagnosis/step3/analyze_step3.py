"""Step 3 analysis (CPU, offline): metrics, gate check and stop rule for every baseline on the filtered pool.

Inputs (all produced on tasks/featurematch/instances_v2f, whose files are checked against the step-2 manifest):
  --inproc DIR   episodes_private.jsonl of inproc_step3.py (reference, black-box, self_probe, template_probe and the
                 zero-effort recipes; label "in-process")
  --sr DIR       sr_recipe.py output (summary.json, answers.jsonl): SR-max / SR-thr (offline from the bank-R cache)
Outputs:
  baselines_public.json   aggregates only (no per-instance or per-slot data)
  tables_autogen.md       the tables BASELINES.md quotes

Metrics per baseline, per tier (T1, T2, T3, all) and per anchor family (pooled, topic, language; PREREG A3.2):
  pass rate (Wilson 95%); planted accuracy (Wilson 95% and instance-clustered bootstrap 95%: 10,000 resamples of the
  instances with replacement, indices from a fresh random.Random(20261002).choices, percentile interval; exactly
  sr_recipe.cluster_bootstrap, A2.5); planted "nothing found" rate; null false-claim rate (both with Wilson); n.
  Family subsets: the slots whose anchor concept is a topic ("topic:...") or a language ("lang:..."). An episode
  enters a family subset if it has at least one slot of that family, and it counts as passed there if every slot of
  that family is correct (an episode-level pass is not defined for one family otherwise). The bootstrap clusters are
  those episodes.
Reference: one-shot = the seed-7 episode of every instance; best-of-5 = seeds 7, 1007, ... until the first pass (as
  inproc_gates.py); its slot metrics use the selected episode (the first passing one, else the last one run).
Gate (PREREG step 3): every non-reference recipe except SR has planted accuracy <= 0.15; the reference (one-shot)
  has planted accuracy >= 0.95. Point estimates decide; "robust" says whether the Wilson CI is on the same side.
Stop rule (PREREG section 4 + A2 + A3): read from the SR summary (primary SR-max / SR-thr, all tiers pooled), with
  the topic-only and language-only numbers alongside (A3.2) and their own text-selection spread.

Run: systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=2G -- /usr/bin/time -v \
       $PY -m tasks.featurematch.diagnosis.step3.analyze_step3 --inproc <run dir> --sr <sr out dir>
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from tasks.featurematch.diagnosis import sr_recipe as SR
from tasks.featurematch.diagnosis.step3 import inproc_step3 as IS

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
TASK = os.path.dirname(DIAG)
POOL = os.path.join(TASK, "instances_v2f")
TIERS = ("all", "T1", "T2", "T3")
FAMS = ("pooled", "topic", "language")
GATE_MAX, REF_MIN = 0.15, 0.95
SR_VARIANTS = ["SR-max", "SR-thr", "SR-max-capped", "SR-thr-capped", "SR-max-all20", "SR-thr-all20"]
NAMES = {  # inproc solver -> display name
    "blackbox": "black-box control",
    "recipe_self_probe": "self_probe",
    "recipe_template_probe": "template_probe",
    "recipe_nothing": "nothing",
    "recipe_always_claim": "always_claim",
    "recipe_prior": "prior [prior.json, unfiltered v2]",
    "recipe_prior_or_none": "prior_or_none [prior.json, unfiltered v2]",
    "recipe_prior@v2f": "prior [prior_v2f.json, filtered draw]",
    "recipe_prior_or_none@v2f": "prior_or_none [prior_v2f.json, filtered draw]",
    "recipe_random": "random",
    "recipe_name_probe": "name_probe",
    "recipe_name_probe_thr": "name_probe_thr",
    "recipe_vocab_match": "vocab_match",
}
ORDER = (["reference (one-shot)", "reference (best-of-5)"] + SR_VARIANTS + [NAMES[k] for k in NAMES])
GATED = [NAMES[k] for k in NAMES]           # every non-reference recipe except SR


def fam_of(cid):
    p = cid.split(":")[0]
    return {"lang": "language", "topic": "topic"}[p]


def load_pool(pool=POOL):
    out = {}
    for p in sorted(glob.glob(os.path.join(pool, "*", "instance.json"))):
        inst = json.load(open(p))
        out[inst["instance_id"]] = {"tier": inst["tier"],
                                    "fam": [fam_of(s["anchor"]) for s in inst["extra"]["slots"]],
                                    "planted": [a["planted"] for a in inst["answer"]["slots"]]}
    return out


def attach(pool, iid, slots):
    fam = pool[iid]["fam"]
    if len(fam) != len(slots) or [s["planted"] for s in slots] != pool[iid]["planted"]:
        raise ValueError(f"{iid}: graded slots do not match the instance")
    return [dict(s, fam=f) for s, f in zip(slots, fam)]


def wil(k, n):
    return SR.wilson(k, n)


def metrics(eps, tier="all", fam="pooled"):
    """eps: list of {"iid", "tier", "slots": [{planted, correct, claimed, fam}], "forward"}."""
    sub = []
    for e in eps:
        if tier != "all" and e["tier"] != tier:
            continue
        sl = e["slots"] if fam == "pooled" else [s for s in e["slots"] if s["fam"] == fam]
        if sl:
            sub.append((e, sl))
    n = len(sub)
    k_pass = sum(all(s["correct"] for s in sl) for _, sl in sub)
    pl = [s for _, sl in sub for s in sl if s["planted"]]
    nu = [s for _, sl in sub for s in sl if not s["planted"]]
    k_pl = sum(s["correct"] for s in pl)
    k_nf = sum(not s["claimed"] for s in pl)
    k_fc = sum(s["claimed"] for s in nu)
    k_inst = [sum(s["correct"] for s in sl if s["planted"]) for _, sl in sub]
    n_inst = [sum(s["planted"] for s in sl) for _, sl in sub]
    fw = [e.get("forward") for e, _ in sub if e.get("forward") is not None]
    return {"n_instances": n, "pass": k_pass, "pass_rate": SR.rate(k_pass, n), "pass_wilson95": wil(k_pass, n),
            "n_planted": len(pl), "planted_correct": k_pl, "planted_acc": SR.rate(k_pl, len(pl)),
            "planted_acc_wilson95": wil(k_pl, len(pl)),
            "planted_acc_cluster_boot95": SR.cluster_bootstrap(k_inst, n_inst) if pl else [None, None],
            "planted_nothing_found": k_nf, "planted_nothing_found_rate": SR.rate(k_nf, len(pl)),
            "planted_nothing_found_wilson95": wil(k_nf, len(pl)),
            "n_null": len(nu), "null_false_claims": k_fc, "null_false_claim_rate": SR.rate(k_fc, len(nu)),
            "null_false_claim_wilson95": wil(k_fc, len(nu)),
            "forward_units_mean": round(float(np.mean(fw)), 1) if fw else None}


def block(eps):
    return {t: {f: metrics(eps, t, f) for f in FAMS} for t in TIERS}


def load_inproc(d, pool):
    rows = [json.loads(line) for line in open(os.path.join(d, "episodes_private.jsonl"))]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["solver"]][r["instance_id"]].append(r)
    eps, info = {}, {}
    for solver, per in by.items():
        if set(per) != set(pool):
            raise ValueError(f"{solver}: {len(per)} instances run, pool has {len(pool)}")
        if solver == "reference":
            one, best = [], []
            n_tries = []
            for iid, rs in per.items():
                rs = sorted(rs, key=lambda r: r["seed"])
                assert rs[0]["seed"] == 7
                one.append(rs[0])
                sel = next((r for r in rs if r["pass"]), rs[-1])
                best.append(sel)
                n_tries.append(len(rs))
            groups = {"reference (one-shot)": one, "reference (best-of-5)": best}
            info["reference"] = {"episodes_run": sum(n_tries), "tries_hist": {str(k): n_tries.count(k)
                                                                              for k in sorted(set(n_tries))}}
        else:
            for iid, rs in per.items():
                assert len(rs) == 1 and rs[0]["seed"] == 7, (solver, iid)
            groups = {NAMES[solver]: [rs[0] for rs in per.values()]}
        for name, rs in groups.items():
            eps[name] = [{"iid": r["instance_id"], "tier": r["tier"], "pass": r["pass"],
                          "slots": attach(pool, r["instance_id"], r["slots"]),
                          "forward": r["used"].get("forward")} for r in rs]
            for e, r in zip(eps[name], rs):
                assert e["pass"] == all(s["correct"] for s in e["slots"])
        allr = [r for rs in per.values() for r in rs]
        info[solver] = dict(info.get(solver, {}), n_episodes=len(allr),
                            n_crashed=sum(r["error"] is not None for r in allr),
                            n_unsubmitted=sum(not r["submitted"] for r in allr),
                            crash_examples=sorted({r["error"][:120] for r in allr if r["error"]})[:3],
                            seconds_mean=round(float(np.mean([r["seconds"] for r in allr])), 2))
    return eps, info


def load_sr(d, pool):
    eps = defaultdict(list)
    for line in open(os.path.join(d, "answers.jsonl")):
        r = json.loads(line)
        eps[r["variant"]].append({"iid": r["instance"], "tier": r["tier"], "pass": r["grade"]["pass"],
                                  "slots": attach(pool, r["instance"], r["grade"]["details"]["slots"]),
                                  "forward": r["forward"]})
    for v, es in eps.items():
        if {e["iid"] for e in es} != set(pool):
            raise ValueError(f"SR {v}: instance set differs from the pool")
    return dict(eps), json.load(open(os.path.join(d, "summary.json")))


def check_sr_reproduced(res, summ):
    """My metrics must equal sr_recipe's own pooled and per-tier numbers (same pool, same functions)."""
    bad = []
    for v in SR_VARIANTS:
        for t in TIERS:
            a, b = res[v][t]["pooled"], summ["metrics"][v][t]
            for k in ("pass", "n_instances", "n_planted", "planted_acc", "planted_acc_wilson95",
                      "planted_acc_cluster_boot95", "planted_nothing_found_rate", "null_false_claim_rate", "n_null"):
                if a[k] != b[k]:
                    bad.append((v, t, k, a[k], b[k]))
    return bad


def family_sensitivity(sr_dir_summary, fam):
    """A2.2 text-selection spread of SR-max / SR-thr planted accuracy on one family's slots only."""
    meta = json.load(open(os.path.join(SR.DIAG_CACHE, "bank_meta.json")))
    cids = meta["cids"]
    cidx = {c: i for i, c in enumerate(cids)}
    R = meta["banks"]["R"]
    insts = []
    for d, inst, pub in SR.iter_instances(POOL):
        specs = [s for s in SR.slot_specs(inst, pub, cidx) if fam_of(s["anchor"]) == fam]
        if specs:
            insts.append((d, inst, specs))
    needed = defaultdict(set)
    for _, _, specs in insts:
        for s in specs:
            needed[s["layer"]].add(s["latent"])
    acts, col, _ = SR.load_acts(SR.DIAG_CACHE, needed, verify=False)
    sens = SR.selection_sensitivity(insts, acts, col, R["rows"], R["styles"], len(cids))
    return {"planted_acc": sens["planted_acc"], "straddles_0.50": sens["straddles_0.50"],
            "primary": [o for o in sens["runs"] if o["primary"]][0]}


def paired_ratio(eps_a, eps_b):
    """A1.5 stop-rule case: pa(a) / pa(b) on the whole pool, with a paired instance bootstrap (10,000 resamples,
    random.Random(20261002); same index draw as sr_recipe.boot_index)."""
    ia = {e["iid"]: e for e in eps_a}
    ids = sorted(ia)
    ib = {e["iid"]: e for e in eps_b}
    ka = np.array([sum(s["correct"] for s in ia[i]["slots"] if s["planted"]) for i in ids], float)
    kb = np.array([sum(s["correct"] for s in ib[i]["slots"] if s["planted"]) for i in ids], float)
    n = np.array([sum(s["planted"] for s in ia[i]["slots"]) for i in ids], float)
    idx = SR.boot_index(len(ids))
    A, B = ka[idx].sum(1), kb[idx].sum(1)
    r = A[B > 0] / B[B > 0]
    lo, hi = np.percentile(r, [2.5, 97.5])
    return {"ratio": round(float(ka.sum() / kb.sum()), 4), "paired_boot95": [round(float(lo), 4), round(float(hi), 4)],
            "pa_num": round(float(ka.sum() / n.sum()), 4), "pa_den": round(float(kb.sum() / n.sum()), 4)}


def fmt_ci(ci):
    return "[–]" if ci[0] is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def fmt_rate(m, k, kk=None):
    v = m[k]
    return "–" if v is None else f"{v:.3f}"


def table(res, names, tier="all", fam="pooled"):
    lines = ["| Baseline | Pass (Wilson 95%) | Planted acc | Wilson 95% | Clustered 95% | Planted NF (Wilson) | "
             "Null false claim (Wilson) | Fwd mean |",
             "|---|---|---|---|---|---|---|---|"]
    for b in names:
        m = res[b][tier][fam]
        lines.append(
            f"| {b} | {m['pass']}/{m['n_instances']} = {fmt_rate(m, 'pass_rate')} {fmt_ci(m['pass_wilson95'])} | "
            f"{m['planted_correct']}/{m['n_planted']} = {fmt_rate(m, 'planted_acc')} | "
            f"{fmt_ci(m['planted_acc_wilson95'])} | {fmt_ci(m['planted_acc_cluster_boot95'])} | "
            f"{fmt_rate(m, 'planted_nothing_found_rate')} {fmt_ci(m['planted_nothing_found_wilson95'])} | "
            f"{m['null_false_claims']}/{m['n_null']} = {fmt_rate(m, 'null_false_claim_rate')} "
            f"{fmt_ci(m['null_false_claim_wilson95'])} | {m['forward_units_mean']} |")
    return "\n".join(lines)


def compact_table(res, names, key="planted_acc"):
    """planted accuracy (or pass) by tier x family, compact."""
    cols = [(t, f) for f in FAMS for t in TIERS]
    head = "| Baseline | " + " | ".join(f"{f} {t}" for t, f in cols) + " |"
    lines = [head, "|---" * (len(cols) + 1) + "|"]
    for b in names:
        cells = []
        for t, f in cols:
            m = res[b][t][f]
            if key == "planted_acc":
                cells.append("–" if m["planted_acc"] is None else f"{m['planted_acc']:.2f} (n={m['n_planted']})")
            else:
                cells.append("–" if m["pass_rate"] is None else f"{m['pass_rate']:.2f} (n={m['n_instances']})")
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inproc", required=True)
    ap.add_argument("--sr", default=os.path.join(HERE, "sr_recipe_out"))
    ap.add_argument("--out", default=os.path.join(HERE, "baselines_public.json"))
    ap.add_argument("--tables", default=os.path.join(HERE, "tables_autogen.md"))
    a = ap.parse_args(argv)
    ident = IS.verify_pool()
    pool = load_pool()
    eps_in, info = load_inproc(a.inproc, pool)
    eps_sr, sr_summ = load_sr(a.sr, pool)
    if sr_summ["pool_sha256"] != ident["pool_sha256"]:
        raise SystemExit("SR summary was computed on a different pool")
    run_meta = json.load(open(os.path.join(a.inproc, "run_meta.json")))
    if run_meta["pool"]["pool_sha256"] != ident["pool_sha256"]:
        raise SystemExit("in-process run was on a different pool")
    eps = dict(eps_in, **eps_sr)
    names = [b for b in ORDER if b in eps]
    res = {b: block(eps[b]) for b in names}
    bad = check_sr_reproduced(res, sr_summ)
    if bad:
        raise SystemExit(f"SR metrics not reproduced: {bad[:5]}")

    # gate (PREREG step 3)
    gate = {"rule": f"every non-reference recipe except SR: planted accuracy <= {GATE_MAX}; reference (one-shot) "
                    f"planted accuracy >= {REF_MIN}; point estimates decide, 'robust' = the Wilson 95% CI is on the "
                    f"same side of the bar", "recipes": {}, "reference": {}}
    for b in GATED:
        if b not in res:
            continue
        m = res[b]["all"]["pooled"]
        gate["recipes"][b] = {"planted_acc": m["planted_acc"], "wilson95": m["planted_acc_wilson95"],
                              "cluster_boot95": m["planted_acc_cluster_boot95"], "n_planted": m["n_planted"],
                              "passes_gate": m["planted_acc"] <= GATE_MAX,
                              "robust": m["planted_acc_wilson95"][1] <= GATE_MAX
                              if m["planted_acc"] <= GATE_MAX else m["planted_acc_wilson95"][0] > GATE_MAX}
    for b in ("reference (one-shot)", "reference (best-of-5)"):
        if b not in res:
            continue
        m = res[b]["all"]["pooled"]
        gate["reference"][b] = {"planted_acc": m["planted_acc"], "wilson95": m["planted_acc_wilson95"],
                                "cluster_boot95": m["planted_acc_cluster_boot95"], "n_planted": m["n_planted"],
                                "pass_rate": m["pass_rate"], "pass_wilson95": m["pass_wilson95"],
                                "meets_bar": m["planted_acc"] >= REF_MIN,
                                "robust": (m["planted_acc_wilson95"][0] >= REF_MIN) if m["planted_acc"] >= REF_MIN
                                else (m["planted_acc_wilson95"][1] < REF_MIN)}
    gate["all_recipes_pass"] = all(v["passes_gate"] for v in gate["recipes"].values())
    gate["failing_recipes"] = [b for b, v in gate["recipes"].items() if not v["passes_gate"]]
    gate["reference_meets_bar"] = gate["reference"].get("reference (one-shot)", {}).get("meets_bar")

    # stop rule (from the SR summary) + family numbers (A3.2) + family text-selection spread
    stop = dict(sr_summ["stop_rule"])
    fam_stop = {}
    for f in ("topic", "language"):
        fs = family_sensitivity(sr_summ, f)
        per = {}
        for v in SR.PRIMARY:
            m = res[v]["all"][f]
            per[v] = {"planted_acc": m["planted_acc"], "n_planted": m["n_planted"],
                      "wilson95": m["planted_acc_wilson95"], "cluster_boot95": m["planted_acc_cluster_boot95"],
                      "above_0.50": m["planted_acc"] > SR.STOP_THR,
                      "wilson_includes_0.50": m["planted_acc_wilson95"][0] <= SR.STOP_THR
                      <= m["planted_acc_wilson95"][1]}
            assert fs["primary"]["sr_max_planted_acc" if v == "SR-max" else "sr_thr_planted_acc"] == m["planted_acc"]
        firing = [v for v in SR.PRIMARY if per[v]["above_0.50"] and per[v]["n_planted"] >= SR.STOP_MIN_PLANTED]
        fam_stop[f] = {"per_variant": per, "would_fire_on_this_family_alone": bool(firing),
                       "firing_variants": firing,
                       "borderline": bool(firing) and all(per[v]["wilson_includes_0.50"] for v in firing),
                       "text_selection_spread": fs["planted_acc"], "straddles_0.50": fs["straddles_0.50"]}
    stop["by_family"] = fam_stop
    stop["fires_pooled_but_not_topic_only"] = bool(stop["fires"] and not fam_stop["topic"][
        "would_fire_on_this_family_alone"])
    stop["P6_note"] = ("The filtered pool failed the PREREG P6 fingerprint check narrowly (menu-feature CV AUROC 0.616 "
                       "in the close tiers, bar 0.60; style_filter_out/v2f_summary.json). The fingerprint is a "
                       "statistic of the menus that only a policy trained on menu statistics could exploit; the SR "
                       "recipe never looks at menu statistics, so it cannot have raised SR's accuracy (A3.1).")
    stop["primary_selection_sensitivity"] = {k: sr_summ["text_selection_sensitivity"][k] for k in
                                             ("planted_acc", "all20_planted_acc", "straddles_0.50",
                                              "text_selection_sensitive")}

    # A1.5 stop-rule case and criterion-3 inputs
    a15 = {}
    if "reference (one-shot)" in eps:
        for v in ("SR-max", "SR-thr"):
            a15[f"pa({v})/pa(ref one-shot)"] = paired_ratio(eps[v], eps["reference (one-shot)"])
    crit3 = {}
    if all(b in res for b in ("reference (one-shot)", "self_probe", "template_probe")):
        r_ref = res["reference (one-shot)"]["all"]["pooled"]["planted_nothing_found_rate"]
        r_sp = res["self_probe"]["all"]["pooled"]["planted_nothing_found_rate"]
        r_tp = res["template_probe"]["all"]["pooled"]["planted_nothing_found_rate"]
        crit3 = {"r_ref": r_ref, "r_rec": round((r_sp + r_tp) / 2, 4), "r_self_probe": r_sp,
                 "r_template_probe": r_tp, "note": "PREREG section 4 criterion 3 inputs, measured on the filtered pool"}

    pub = {"what": "FeatureMatch diagnosis step 3: baselines on the filtered pool (instances_v2f)",
           "pool": ident, "labels": {"in-process": "reference, black-box, self_probe, template_probe and the zero-"
                                                   "effort recipes: inproc_step3.py (inproc_gates path, one gpuq job)",
                                     "offline": "SR variants: sr_recipe.py from the bank-R activation cache"},
           "inproc_run": {"dir": os.path.relpath(os.path.abspath(a.inproc), TASK), **{k: run_meta[k] for k in (
               "label", "solvers", "ref_tries", "seeds", "model_service_socket", "generate_memo", "n_episodes",
               "n_crashed", "elapsed_s", "git_sha", "prior_v2f_sha256", "prior_v1_sha256")}},
           "solver_info": info,
           "sr_run": {"dir": os.path.relpath(os.path.abspath(a.sr), TASK), "git_sha": sr_summ["git_sha"],
                      "rule": sr_summ["rule"], "bank_check": sr_summ["bank_check"],
                      "cache_files": sr_summ["cache_files"]},
           "family_definition": "anchor concept id prefix: 'topic:' -> topic, 'lang:' -> language; family pass = "
                                "every slot of that family in the episode is correct (episodes with >= 1 such slot)",
           "ci": {"wilson": "Wilson score 95% (z=1.959964)", "cluster_boot": "instance-clustered bootstrap, 10,000 "
                  "resamples, random.Random(20261002).choices, percentile 2.5/97.5"},
           "gate": gate, "stop_rule": stop, "A1.5_stop_rule_case": a15, "criterion3_inputs": crit3,
           "metrics": res}
    json.dump(pub, open(a.out, "w"), indent=1)

    md = ["# Step 3 tables (auto-generated by analyze_step3.py; do not edit by hand)", "",
          f"Pool sha256 {ident['pool_sha256']} (manifest sha256 {ident['manifest_sha256']}).", ""]
    for f in FAMS:
        md += [f"## All tiers, {f} slots", "", table(res, names, "all", f), ""]
    for t in ("T1", "T2", "T3"):
        md += [f"## {t}, pooled", "", table(res, names, t, "pooled"), ""]
    md += ["## Planted accuracy by tier and family", "", compact_table(res, names, "planted_acc"), "",
           "## Pass rate by tier and family", "", compact_table(res, names, "pass"), ""]
    open(a.tables, "w").write("\n".join(md))
    print(json.dumps({"gate": gate, "stop_rule": {k: stop[k] for k in ("labels", "fires", "borderline",
                                                                     "text_selection_sensitive",
                                                                     "fires_pooled_but_not_topic_only")},
                      "by_family": {f: {v: fam_stop[f]["per_variant"][v]["planted_acc"] for v in SR.PRIMARY}
                                    for f in fam_stop},
                      "A1.5": a15, "crit3": crit3}, indent=1))


if __name__ == "__main__":
    main()
