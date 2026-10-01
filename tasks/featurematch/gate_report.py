"""Gate report for FeatureMatch: combines the HARNESS gate run dirs and the IN-PROCESS complement.

Sources (kept apart in the output, never pooled):
  harness     runs/featurematch/<ts>_gate_{reference,blackbox,recipes}/episodes/*/grade.json (common.sandbox
              run-scripted). Scripted solvers always submit unless they crash, so an episode with NO submission is
              an infrastructure failure (e.g. a solver killed while waiting for the shared GPU queue). It is counted
              under infrastructure_failures_excluded and left out of the rates. INVALID episodes are counted too.
  in-process  runs/featurematch/<ts>_inproc/episodes_private.jsonl (inproc_gates.py; same solvers, Env, caps and
              out-of-process grader, but no broker). Used because the shared GPU queue made harness GPU episodes
              take 15-40 min each (NOTES).
Also reports per-slot agreement between the two sources on the same (solver, instance) as a check of the in-process
path (deterministic solvers must agree exactly).

Gate rules: reference best-of-5 >= 95% on kept instances (one-shot also reported), blackbox <= 10%, every recipe
<= 10%, with Wilson 95% intervals (common.sandbox.wilson).

Writes:
  runs/featurematch/<out-prefix>_gates_public.json  aggregates only (no per-instance data; safe to commit, D6)
  tasks/featurematch/instances_manifest.json        `kept` flags (reference best-of-5 passed in either source)
  tasks/featurematch/smoke_plan.json                (--smoke) 3 kept instances per tier (random, seeded), not
                                                    already seen by any LLM probe (--exclude), together holding >= 1
                                                    null and >= 1 planted slot per tier
Run: $PY -m tasks.featurematch.gate_report --harness R1 R2 R3 --inproc D --out-prefix <ts> [--smoke --exclude ids]
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict

from common.sandbox import wilson

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "runs", "featurematch")
GATE_MAX, REF_MIN = 0.10, 0.95


def ci(k, n):
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": [round(lo, 4), round(hi, 4)]}


def slot_rates(slot_lists):
    null = [s for sl in slot_lists for s in sl if not s["planted"]]
    pl = [s for sl in slot_lists for s in sl if s["planted"]]
    return {"null_slot_false_claim_rate": round(sum(s["claimed"] for s in null) / len(null), 4) if null else None,
            "planted_slot_accuracy": round(sum(s["correct"] for s in pl) / len(pl), 4) if pl else None}


def load_harness(dirs):
    eps, invalid, infra = [], defaultdict(int), defaultdict(int)
    for rd in dirs:
        for p in glob.glob(os.path.join(rd, "episodes", "*", "grade.json")):
            g = json.load(open(p))
            h = g["harness"]
            ep = json.load(open(os.path.join(os.path.dirname(p), "episode.json")))
            lab = h.get("solver_label")
            if not h["valid"]:
                invalid[lab] += 1
            elif not h.get("submitted"):
                infra[lab] += 1
            else:
                eps.append({"solver": lab, "instance_id": h["instance_id"], "tier": h["tier"], "pass": bool(g["pass"]),
                            "score": float(g["score"]), "slots": g["details"]["slots"],
                            "t": ep.get("created_at") or 0})
    return sorted(eps, key=lambda e: e["t"]), dict(invalid), dict(infra)


def load_inproc(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "episodes_private.jsonl"))]
    return [dict(r, t=i) for i, r in enumerate(rows)]


def block(eps):
    """Per-tier and overall stats per solver for one source."""
    out = defaultdict(dict)
    by = defaultdict(lambda: defaultdict(list))
    for e in eps:
        by[e["solver"]][e["tier"]].append(e)
    for s, tiers in sorted(by.items()):
        allr = [e for es in tiers.values() for e in es]
        for t, es in sorted(tiers.items()) + [("all", allr)]:
            if s == "reference":
                first, anyp, n_try = {}, defaultdict(bool), defaultdict(int)
                for e in sorted(es, key=lambda e: e["t"]):
                    if n_try[e["instance_id"]] >= 5:
                        continue
                    n_try[e["instance_id"]] += 1
                    first.setdefault(e["instance_id"], e)
                    anyp[e["instance_id"]] |= e["pass"]
                out[s][t] = dict(one_shot=ci(sum(e["pass"] for e in first.values()), len(first)),
                                 best_of_5=ci(sum(anyp.values()), len(anyp)),
                                 **slot_rates([e["slots"] for e in first.values()]))
            else:
                k = sum(e["pass"] for e in es)
                out[s][t] = dict(ci(k, len(es)), **slot_rates([e["slots"] for e in es]),
                                 mean_score=round(sum(e.get("score", 0.0) for e in es) / len(es), 4) if es else None,
                                 gate_ok=(k / len(es) <= GATE_MAX) if es else None)
    return dict(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", nargs="*", default=[])
    ap.add_argument("--inproc", default=None)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=[], help="instance ids already seen by an LLM probe")
    a = ap.parse_args()

    h_eps, h_invalid, h_infra = load_harness(a.harness)
    i_eps = load_inproc(a.inproc) if a.inproc else []
    pub = {"generated_by": "tasks/featurematch/gate_report.py",
           "harness_runs": [os.path.basename(os.path.normpath(r)) for r in a.harness],
           "inproc_run": os.path.basename(os.path.normpath(a.inproc)) if a.inproc else None,
           "gate_rules": {"reference_best_of_5_min": REF_MIN, "blackbox_max": GATE_MAX, "recipe_max": GATE_MAX},
           "harness": {"invalid_episodes": h_invalid, "infrastructure_failures_excluded": h_infra,
                       "results": block(h_eps)},
           "inproc": {"results": block(i_eps)}}

    # agreement between sources on the same (solver, instance): first episode of each
    first_h, first_i = {}, {}
    for e in h_eps:
        first_h.setdefault((e["solver"], e["instance_id"]), e)
    for e in i_eps:
        first_i.setdefault((e["solver"], e["instance_id"]), e)
    agree = defaultdict(lambda: [0, 0])
    for k in set(first_h) & set(first_i):
        same = [s["correct"] for s in first_h[k]["slots"]] == [s["correct"] for s in first_i[k]["slots"]]
        agree[k[0]][0] += same
        agree[k[0]][1] += 1
    pub["harness_vs_inproc_slot_agreement"] = {s: {"same": v[0], "n": v[1]} for s, v in sorted(agree.items())}

    # kept flags: reference best-of-5 passed in either source
    ref_pass, ref_tries, first_ref = defaultdict(bool), defaultdict(int), {}
    for e in i_eps + h_eps:
        if e["solver"] == "reference":
            ref_pass[e["instance_id"]] |= e["pass"]
            ref_tries[e["instance_id"]] += 1
    for e in sorted([e for e in i_eps if e["solver"] == "reference"], key=lambda e: e["t"]):
        first_ref.setdefault(e["instance_id"], e)
    man_path = os.path.join(HERE, "instances_manifest.json")
    man = json.load(open(man_path))
    dropped = []
    for m in man["instances"]:
        iid = m["instance_id"]
        if iid in ref_pass:
            m["kept"] = bool(ref_pass[iid]) or ref_tries[iid] < 5
            m["gate_runs"] = [x for x in [pub["inproc_run"]] + pub["harness_runs"] if x]
            if not m["kept"]:
                dropped.append(iid)
    json.dump(man, open(man_path, "w"), indent=1)
    pub["pool"] = {"n_instances": len(man["instances"]), "n_dropped": len(dropped), "dropped": dropped}
    out_pub = os.path.join(RUNS, f"{a.out_prefix}_gates_public.json")
    json.dump(pub, open(out_pub, "w"), indent=1)

    if a.smoke:
        rng = random.Random(20261001)
        tier_of = {m["instance_id"]: m["tier"] for m in man["instances"]}
        plan = {"tiers": {}, "profile": "full",
                "note": "3 instances per tier drawn at random (seeded) from ALL kept instances (reference best-of-5 "
                        "passed), excluding instances already used by LLM probes; together they hold at least one "
                        "null and one planted slot per tier. Not restricted to the reference's first-try passes, so "
                        "the smoke set is not easier than the pool (audit MINOR 8). Chosen by gate_report.py."}
        for t in sorted(set(tier_of.values())):
            cands = sorted(i for i, e in first_ref.items()
                           if tier_of[i] == t and ref_pass[i] and i not in set(a.exclude) and i not in dropped)
            for _ in range(1000):
                pick = sorted(rng.sample(cands, 3))
                sl = [s for i in pick for s in first_ref[i]["slots"]]
                if any(not s["planted"] for s in sl) and any(s["planted"] for s in sl):
                    break
            plan["tiers"][t] = [os.path.join("tasks", "featurematch", "instances", i) for i in pick]
        json.dump(plan, open(os.path.join(HERE, "smoke_plan.json"), "w"), indent=1)
    print(out_pub)


if __name__ == "__main__":
    main()
