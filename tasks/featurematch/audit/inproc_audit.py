"""AUDIT: in-process complement for the GPU attack solvers (same pattern and caveats as tasks/featurematch/inproc_gates.py:
same Env, profiles and per-instance caps via common.toolserver.make_local_call, `budget`/`submit` built-ins, grader.py
run as a separate process; no broker, leak scan or wall clock). One normally queued GPU job instead of one 7 GB
admission per episode. The harness runs of the same solvers (runs/featurematch/<ts>_audit_*) cross-check it.

Run: $PY -m common.gpuq run --gb 7 --label featurematch-audit -- $PY -m tasks.featurematch.audit.inproc_audit \
        --instances-file F --variants template_probe,self_probe,density_null --out runs/featurematch/<ts>_audit_inproc
Writes <out>/episodes_private.jsonl (per-slot detail; gitignored by the **/episodes_private.jsonl rule, D6) and
<out>/audit_public.json (aggregates).
"""
import argparse
import json
import os
import time
from collections import defaultdict

from tasks.featurematch.inproc_gates import ci, grade_out_of_process, make_call


def run_one(instance_dir, variant, seed):
    from common.broker import merged_caps
    from tasks.featurematch.audit import attack_solvers
    from tasks.featurematch.tools import Env
    inst = json.load(open(os.path.join(instance_dir, "instance.json")))
    caps = {k: v for k, v in merged_caps(inst.get("caps")).items() if k not in ("wall_clock_s", "call_timeout_s")}
    env = Env(instance_dir=instance_dir, sandbox_dir=None, profile="full")
    env.load(instance_dir)
    call, state, raw = make_call(env, caps)
    t0, err = time.time(), None
    try:
        attack_solvers.solve(call, variant, seed=seed)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    g = grade_out_of_process(instance_dir, state["submission"])
    return {"instance_id": inst["instance_id"], "tier": inst["tier"], "solver": "audit_" + variant, "seed": seed,
            "pass": bool(g["pass"]), "score": g["score"], "slots": g["details"]["slots"],
            "submitted": state["submission"] is not None, "error": err, "used": dict(raw.used),
            "seconds": round(time.time() - t0, 1)}


def summarize(rows):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["solver"]][r["tier"]].append(r)
        by[r["solver"]]["all"].append(r)
    out = {}
    for s, tiers in sorted(by.items()):
        out[s] = {}
        for t, rs in sorted(tiers.items()):
            null = [x for r in rs for x in r["slots"] if not x["planted"]]
            pl = [x for r in rs for x in r["slots"] if x["planted"]]
            out[s][t] = dict(ci(sum(r["pass"] for r in rs), len(rs)),
                             mean_score=round(sum(r["score"] for r in rs) / len(rs), 4),
                             planted_slot_accuracy=round(sum(x["correct"] for x in pl) / max(1, len(pl)), 4),
                             planted_slot_nothing_rate=round(sum(not x["claimed"] for x in pl) / max(1, len(pl)), 4),
                             null_slot_false_claim_rate=round(sum(x["claimed"] for x in null) / max(1, len(null)), 4),
                             n_crashed=sum(r["error"] is not None for r in rs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances-file", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from common import gpuq
    import torch  # noqa: F401
    gpuq.apply_caps()
    insts = [l.strip() for l in open(a.instances_file) if l.strip()]
    os.makedirs(a.out, exist_ok=True)
    priv = os.path.join(a.out, "episodes_private.jsonl")
    rows, done = [], set()
    if os.path.exists(priv):
        for l in open(priv):
            r = json.loads(l)
            rows.append(r)
            done.add((r["instance_id"], r["solver"]))
    with open(priv, "a") as f:
        for v in a.variants.split(","):
            for d in insts:
                iid = os.path.basename(d.rstrip("/"))
                if (iid, "audit_" + v) in done:
                    continue
                r = run_one(os.path.abspath(d), v, 7)
                f.write(json.dumps(r) + "\n")
                f.flush()
                rows.append(r)
                print(json.dumps({k: r[k] for k in ("instance_id", "solver", "pass", "score", "error", "seconds")}),
                      flush=True)
    pub = {"mode": "in-process complement (NOT through the broker); see tasks/featurematch/inproc_gates.py",
           "instances_file": a.instances_file, "n_instances": len(insts), "results": summarize(rows)}
    json.dump(pub, open(os.path.join(a.out, "audit_public.json"), "w"), indent=1)
    print(json.dumps(pub["results"], indent=1))


if __name__ == "__main__":
    main()
