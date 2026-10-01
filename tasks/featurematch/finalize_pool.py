"""Apply the "drop instances the reference cannot solve" rule to a gate run, and record the result.

Reads runs/featurematch/<run>/episodes.jsonl (from run_gates.py), marks every instance whose reference best-of-5
failed as dropped, writes `kept` flags into instances_manifest.json (no answers), and writes smoke_plan.json
(3 kept instances per tier, each containing at least one null slot; the first also has at least one planted slot).
Also recomputes every gate on the KEPT instances only and writes kept_summary.json into the run dir.

Run: $PY -m tasks.featurematch.finalize_pool runs/featurematch/<run> [more run dirs...]
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    recs = []
    for rd in sys.argv[1:]:
        recs += [dict(json.loads(l), _run=rd) for l in open(os.path.join(rd, "episodes.jsonl"))]
    ref = {r["instance"]: r for r in recs if r["solver"] == "reference"}
    dropped = sorted(i for i, r in ref.items() if not r["best_of_5"])
    mpath = os.path.join(HERE, "instances_manifest.json")
    man = json.load(open(mpath))
    for m in man["instances"]:
        if m["instance_id"] in ref:
            m["kept"] = m["instance_id"] not in dropped
            m["gate_run"] = os.path.basename(ref[m["instance_id"]]["_run"])
    json.dump(man, open(mpath, "w"), indent=1)
    kept = {i for i in ref if i not in dropped}
    st = defaultdict(lambda: defaultdict(list))
    slot = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r["instance"] not in kept:
            continue
        t = r["tier"]
        if r["solver"] == "reference":
            st[t]["reference_one_shot"].append(r["one_shot"])
            st[t]["reference_best_of_5"].append(r["best_of_5"])
            sl = r["tries"][0]["slots"]
        else:
            st[t][r["solver"]].append(r["pass"])
            sl = r["slots"]
        for s, k in zip(sl, r["kinds"]):
            slot[t][f"{r['solver']}/{k}"].append(s["correct"])
            if not s["planted"]:
                slot[t][f"{r['solver']}/null_fp"].append(s["claimed"])
    out = {"dropped": dropped, "n_ref_instances": len(ref), "n_kept": len(kept),
           "by_tier": {t: {k: {"k": int(sum(v)), "n": len(v), "rate": round(sum(v) / len(v), 3)} for k, v in s.items()}
                       for t, s in sorted(st.items())},
           "slot_rates": {t: {k: round(sum(v) / len(v), 3) for k, v in sorted(s.items())} for t, s in sorted(slot.items())}}
    for rd in sys.argv[1:]:
        json.dump(out, open(os.path.join(rd, "kept_summary.json"), "w"), indent=1)
    # smoke plan
    plan = {"tiers": {}, "profile": "full"}
    for t in sorted(st):
        cands = []
        for i in sorted(kept):
            r = ref[i]
            if r["tier"] != t:
                continue
            kinds = r["kinds"]
            if any(k != "planted" for k in kinds):
                cands.append((any(k == "planted" for k in kinds), i))
        cands.sort(key=lambda x: (not x[0], x[1]))
        plan["tiers"][t] = [os.path.join("tasks", "featurematch", "instances", i) for _, i in cands[:3]]
    json.dump(plan, open(os.path.join(HERE, "smoke_plan.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("dropped", "n_ref_instances", "n_kept", "by_tier")}, indent=1))


if __name__ == "__main__":
    main()
