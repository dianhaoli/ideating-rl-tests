"""In-process COMPLEMENT to the harness gates (compute-forced; see NOTES 2026-10-01 GPU-queue starvation).

WHY this exists: every harness episode starts its own 7 GB tool server, and on the shared L4 each admission waited
15-40 min behind other tasks' smaller jobs, so ~300 GPU gate episodes through `common.sandbox run-scripted` would take
more than a day. This script runs the SAME solver code against the SAME Env class, profiles and per-instance caps, as
ONE normally queued GPU job (`common.gpuq run --gb 7 ...`, model loaded once), so the GPU-using gates can be measured
on >= 20 instances per tier. Results are labelled "in-process" everywhere and never mixed with harness numbers.

What is the same as the harness: Env.load / tools / profiles (blackbox cannot call SAE tools), cap accounting
(common.toolserver.make_local_call: tool_calls + forward/generate charges, instance caps merged with the harness
defaults), the `budget` and `submit` built-ins (budget in the broker's shape; submit runs Env.validate_submission and
accepts exactly one valid submission), and grading (grader.py run as a SEPARATE PROCESS on the stored submission).
What is not: no broker, no leak scan, no sandbox or transcript audit (scripted solvers have no transcript anyway),
no wall-clock or per-call timeout.

Run (from the worktree root):
  $PY -m common.gpuq run --gb 7 --label fm-gates-inproc -- $PY -m tasks.featurematch.inproc_gates \
      --instances-file F --solvers reference,blackbox,recipe_name_probe_thr,... --out runs/featurematch/<ts>_inproc
Writes <out>/episodes_private.jsonl (per episode incl. per-slot detail; gitignored, D6) and <out>/gates_public.json
(aggregates with Wilson 95% CIs).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def make_call(env, caps):
    from common.toolserver import ToolError, make_local_call
    call = make_local_call(env, caps)
    state = {"submission": None}

    def wrapped(name, **args):
        if name == "budget":
            used = call.used
            return {k: {"used": used.get(k, 0), "cap": caps[k], "remaining": caps[k] - used.get(k, 0)}
                    for k in ("tool_calls", "forward", "generate", "gradient") if k in caps}
        if name == "submit":
            if state["submission"] is not None:
                raise ToolError("episode finished")
            err = env.validate_submission(args)
            if err:
                raise ToolError(f"invalid submission: {err}")
            state["submission"] = args
            return {"accepted": True}
        return call(name, **args)
    return wrapped, state, call


def grade_out_of_process(instance_dir, submission):
    with tempfile.TemporaryDirectory() as td:
        sp, gp = os.path.join(td, "sub.json"), os.path.join(td, "grade.json")
        with open(sp, "w") as f:
            json.dump(submission if submission is not None else {}, f)
        subprocess.run([PY, os.path.join(HERE, "grader.py"), "--instance-dir", instance_dir, "--submission", sp,
                        "--out", gp], check=True, stdout=subprocess.DEVNULL)
        return json.load(open(gp))


def run_one(instance_dir, solver, seed):
    from common.broker import merged_caps
    from tasks.featurematch import blackbox_control, recipe_baseline, reference_solver
    from tasks.featurematch.tools import Env
    inst = json.load(open(os.path.join(instance_dir, "instance.json")))
    caps = {k: v for k, v in merged_caps(inst.get("caps")).items() if k not in ("wall_clock_s", "call_timeout_s")}
    profile = "blackbox" if solver == "blackbox" else "full"
    env = Env(instance_dir=instance_dir, sandbox_dir=None, profile=profile)
    env.load(instance_dir)
    call, state, raw = make_call(env, caps)
    t0 = time.time()
    err = None
    try:
        if solver == "reference":
            reference_solver.solve(call, seed=seed)
        elif solver == "blackbox":
            blackbox_control.solve(call, seed=seed)
        elif solver.startswith("recipe_"):
            recipe_baseline.solve(call, solver[len("recipe_"):], seed=seed)
        else:
            raise SystemExit(f"unknown solver {solver}")
    except Exception as e:          # a solver crash is recorded; the episode is graded on whatever was submitted
        err = f"{type(e).__name__}: {e}"
    g = grade_out_of_process(instance_dir, state["submission"])
    return {"instance_id": inst["instance_id"], "tier": inst["tier"], "solver": solver, "seed": seed,
            "pass": bool(g["pass"]), "score": g["score"], "slots": g["details"]["slots"],
            "submitted": state["submission"] is not None, "error": err, "used": dict(raw.used),
            "seconds": round(time.time() - t0, 1)}


def ci(k, n):
    from common.sandbox import wilson
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": [round(lo, 4), round(hi, 4)]}


def summarize(rows):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["solver"]][r["tier"]].append(r)
    out = {}
    for s, tiers in sorted(by.items()):
        out[s] = {}
        for t, rs in sorted(list(tiers.items()) + [("all", [r for rr in tiers.values() for r in rr])]):
            if s == "reference":
                first = {}
                anyp = defaultdict(bool)
                for r in rs:
                    first.setdefault(r["instance_id"], r["pass"])
                    anyp[r["instance_id"]] |= r["pass"]
                out[s][t] = {"one_shot": ci(sum(first.values()), len(first)),
                             "best_of_5": ci(sum(anyp.values()), len(anyp))}
            else:
                null = [x for r in rs for x in r["slots"] if not x["planted"]]
                out[s][t] = dict(ci(sum(r["pass"] for r in rs), len(rs)),
                                 null_slot_false_claim_rate=round(sum(x["claimed"] for x in null) / len(null), 4)
                                 if null else None,
                                 n_crashed=sum(r["error"] is not None for r in rs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances-file", required=True)
    ap.add_argument("--solvers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-tries", type=int, default=5, help="reference: up to this many seeds, stop at first pass")
    a = ap.parse_args()
    from common import gpuq
    import torch  # noqa: F401  (loaded before the caps are applied, as the builder guide asks)
    gpuq.apply_caps()
    insts = [l.strip() for l in open(a.instances_file) if l.strip()]
    os.makedirs(a.out, exist_ok=True)
    priv = os.path.join(a.out, "episodes_private.jsonl")
    done = set()
    rows = []
    if os.path.exists(priv):            # resumable
        for l in open(priv):
            r = json.loads(l)
            rows.append(r)
            done.add((r["instance_id"], r["solver"], r["seed"]))
    with open(priv, "a") as f:
        for solver in a.solvers.split(","):
            for d in insts:
                iid = os.path.basename(d.rstrip("/"))
                tries = a.ref_tries if solver == "reference" else 1
                for k in range(tries):
                    seed = 1000 * k + 7
                    if (iid, solver, seed) in done:
                        r = next(x for x in rows if (x["instance_id"], x["solver"], x["seed"]) == (iid, solver, seed))
                    else:
                        r = run_one(os.path.abspath(d), solver, seed)
                        f.write(json.dumps(r) + "\n")
                        f.flush()
                        rows.append(r)
                        print(json.dumps({k2: r[k2] for k2 in ("instance_id", "solver", "seed", "pass", "error",
                                                               "seconds")}), flush=True)
                    if r["pass"]:
                        break
    pub = {"mode": "in-process complement (NOT through the broker); see docstring", "instances_file": a.instances_file,
           "n_instances": len(insts), "results": summarize(rows)}
    json.dump(pub, open(os.path.join(a.out, "gates_public.json"), "w"), indent=1)
    print(json.dumps(pub["results"], indent=1))


if __name__ == "__main__":
    main()
