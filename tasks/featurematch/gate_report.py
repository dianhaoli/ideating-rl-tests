"""Gate report for FeatureMatch from harness run dirs (common.sandbox run-scripted output).

Reads every episodes/*/grade.json under the given run dirs and computes, per tier, on KEPT instances:
  reference one-shot  = the reference's FIRST episode on each instance passed
  reference best-of-5 = any of the reference's first 5 episodes passed (retries are only run after a failure)
  every other solver label (blackbox, recipe_<variant>): pass rate over valid episodes
with Wilson 95% intervals (common.sandbox.wilson). Scripted solvers always submit unless they crash, so an episode
with NO submission is an infrastructure failure (e.g. the solver gave up waiting for the shared GPU): it is counted
under infrastructure_failures_excluded and left out of the rates instead of being scored as a fail. An instance is DROPPED when the reference failed all 5 attempts
(builder guide rule 8); one that failed one-shot and has < 5 attempts is PENDING (run more repeats on it).

Writes:
  runs/featurematch/<prefix>_gates_public.json   aggregates only (no per-instance or per-slot data; safe to commit
                                                  while the pool is still used for agent episodes, D6)
  <reference run dir>/gate_private.json          per-instance detail (gitignored until the smoke run is done)
  tasks/featurematch/instances_manifest.json     `kept` flags + gate run name (no answers)
  tasks/featurematch/smoke_plan.json             (--smoke) 3 kept instances per tier the reference passed one-shot,
                                                  together holding >= 1 null and >= 1 planted slot per tier

Run: $PY -m tasks.featurematch.gate_report --reference R --others B C ... --out-prefix <ts> [--smoke] [--pending]
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
GATE_MAX = 0.10
REF_MIN = 0.95


def episodes(run_dir):
    out = []
    for p in glob.glob(os.path.join(run_dir, "episodes", "*", "grade.json")):
        g = json.load(open(p))
        ep = json.load(open(os.path.join(os.path.dirname(p), "episode.json")))
        g["_created"] = ep.get("created_at") or 0
        g["_dir"] = os.path.dirname(p)
        out.append(g)
    return sorted(out, key=lambda g: g["_created"])


def ci(k, n):
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": [round(lo, 4), round(hi, 4)]}


def slot_stats(eps):
    null = [s for g in eps for s in g["details"].get("slots", []) if not s["planted"]]
    pl = [s for g in eps for s in g["details"].get("slots", []) if s["planted"]]
    return {"null_slot_false_claim_rate": round(sum(s["claimed"] for s in null) / len(null), 4) if null else None,
            "planted_slot_accuracy": round(sum(s["correct"] for s in pl) / len(pl), 4) if pl else None,
            "n_null_slots": len(null), "n_planted_slots": len(pl)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pending", action="store_true", help="only print instance dirs that need more reference tries")
    a = ap.parse_args()

    ref = defaultdict(list)
    invalid = defaultdict(int)
    infra = defaultdict(int)
    tier_of, inst_dir = {}, {}
    for g in episodes(a.reference):
        h = g["harness"]
        if h.get("solver_label") != "reference":
            continue
        if not h["valid"]:
            invalid["reference"] += 1
            continue
        if not h.get("submitted"):          # scripted solvers always submit unless they crashed (infrastructure)
            infra["reference"] += 1
            continue
        ref[h["instance_id"]].append(g)
        tier_of[h["instance_id"]] = h["tier"]
    man_path = os.path.join(HERE, "instances_manifest.json")
    man = json.load(open(man_path))
    for m in man["instances"]:
        inst_dir[m["instance_id"]] = os.path.join("tasks", "featurematch", "instances", m["instance_id"])
        tier_of.setdefault(m["instance_id"], m["tier"])

    status = {}
    for iid, gs in ref.items():
        tries = [bool(g["pass"]) for g in gs[:5]]
        if tries[0] or any(tries):
            status[iid] = "kept"
        elif len(tries) >= 5:
            status[iid] = "dropped"
        else:
            status[iid] = "pending"
    if a.pending:      # "<instance dir> <attempts still needed>": failed one-shot (up to 5) or no graded attempt yet
        for iid, s in sorted(status.items()):
            if s == "pending":
                print(inst_dir[iid], 5 - len(ref[iid]))
        for iid in sorted(set(inst_dir) - set(ref)):
            print(inst_dir[iid], 1)
        return

    kept = {i for i, s in status.items() if s == "kept"}
    tiers = sorted({tier_of[i] for i in ref})
    pub = {"generated_by": "tasks/featurematch/gate_report.py", "reference_run": os.path.basename(a.reference),
           "other_runs": [os.path.basename(o) for o in a.others],
           "pool": {"n_instances": len(ref), "n_kept": len(kept),
                    "n_dropped": sum(s == "dropped" for s in status.values()),
                    "n_pending": sum(s == "pending" for s in status.values()),
                    "by_tier": {t: {"n": sum(tier_of[i] == t for i in ref),
                                    "kept": sum(tier_of[i] == t for i in kept)} for t in tiers}},
           "gate_rules": {"reference_best_of_5_min": REF_MIN, "blackbox_max": GATE_MAX, "recipe_max": GATE_MAX},
           "invalid_episodes": {}, "tiers": {}, "all_tiers": {}}
    priv = {"status": status, "reference": {i: [bool(g["pass"]) for g in gs] for i, gs in ref.items()}, "others": {}}

    others = defaultdict(lambda: defaultdict(list))
    for rd in a.others:
        for g in episodes(rd):
            h = g["harness"]
            lab = h.get("solver_label")
            if not h["valid"]:
                invalid[lab] += 1
                continue
            if not h.get("submitted"):
                infra[lab] += 1
                continue
            if h["instance_id"] in kept:
                others[lab][h["tier"]].append(g)
                priv["others"].setdefault(lab, {})[h["instance_id"]] = bool(g["pass"])
    pub["invalid_episodes"] = dict(invalid)
    pub["infrastructure_failures_excluded"] = dict(infra)

    def block(tier_filter):
        ks = [i for i in kept if tier_filter(tier_of[i])]
        one = sum(bool(ref[i][0]["pass"]) for i in ks)
        b5 = sum(any(bool(g["pass"]) for g in ref[i][:5]) for i in ks)
        allref = [i for i in ref if tier_filter(tier_of[i]) and status[i] != "pending"]
        b5_all = sum(any(bool(g["pass"]) for g in ref[i][:5]) for i in allref)
        out = {"reference_one_shot": ci(one, len(ks)),
               "reference_best_of_5_kept": ci(b5, len(ks)),
               "reference_best_of_5_all_generated": ci(b5_all, len(allref)),
               "reference_slots_first_try": slot_stats([ref[i][0] for i in ks])}
        for lab in sorted(others):
            eps = [g for t, gs in others[lab].items() if tier_filter(t) for g in gs]
            k = sum(bool(g["pass"]) for g in eps)
            out[lab] = dict(ci(k, len(eps)), **slot_stats(eps),
                            gate_ok=(k / len(eps) <= GATE_MAX) if eps else None)
        return out

    for t in tiers:
        pub["tiers"][t] = block(lambda x, t=t: x == t)
    pub["all_tiers"] = block(lambda x: True)
    worst = {}
    for t, b in list(pub["tiers"].items()) + [("all", pub["all_tiers"])]:
        rec = [(lab, v["rate"]) for lab, v in b.items() if lab.startswith("recipe_") and v["rate"] is not None]
        worst[t] = max(rec, key=lambda x: x[1]) if rec else None
    pub["worst_recipe"] = worst
    out_pub = os.path.join(RUNS, f"{a.out_prefix}_gates_public.json")
    json.dump(pub, open(out_pub, "w"), indent=1)
    json.dump(priv, open(os.path.join(a.reference, "gate_private.json"), "w"), indent=1)

    for m in man["instances"]:
        if m["instance_id"] in status:
            m["kept"] = status[m["instance_id"]] != "dropped"
            m["gate_run"] = os.path.basename(a.reference)
    json.dump(man, open(man_path, "w"), indent=1)

    if a.smoke:
        rng = random.Random(20261001)
        plan = {"tiers": {}, "profile": "full",
                "note": "3 instances per tier the reference solver passed on its first try; together they hold at "
                        "least one null and one planted slot per tier. Chosen by gate_report.py (seeded)."}
        for t in tiers:
            cands = sorted(i for i in kept if tier_of[i] == t and ref[i][0]["pass"])
            for _ in range(1000):
                pick = sorted(rng.sample(cands, 3))
                sl = [s for i in pick for s in ref[i][0]["details"]["slots"]]
                if any(not s["planted"] for s in sl) and any(s["planted"] for s in sl):
                    break
            plan["tiers"][t] = [inst_dir[i] for i in pick]
        json.dump(plan, open(os.path.join(HERE, "smoke_plan.json"), "w"), indent=1)
    print(json.dumps({"pool": pub["pool"], "worst_recipe": worst,
                      "tiers": {t: {k: (v.get("rate"), v.get("n")) for k, v in b.items()}
                                for t, b in pub["tiers"].items()}}, indent=1))


if __name__ == "__main__":
    main()
