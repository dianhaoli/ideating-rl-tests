"""Preliminary gates, IN-PROCESS (the shared harness is not READY yet).

For every instance it builds a fresh Env, wraps it in local_shim.LocalEpisode (same caps, one tool_calls unit per
call, forward/generate charged by the tools, one submission), runs a solver through `call(...)` only, and grades
with grader.grade (the real grader function, called here in-process for speed; the harness will call its CLI).

Solvers: reference (one-shot seed 0; on failure re-run seeds 1-4 => best-of-5), blackbox (blackbox profile),
and every recipe_baseline variant.

Writes runs/featurematch/<YYYYmmdd-HHMMSS>_<label>/{config.json, episodes.jsonl, summary.json, summary.md}.
Run: $PY -m common.gpuq run --gb 8 --heavy --label fm-gates -- $PY -m tasks.featurematch.run_gates --label prelim
"""
import argparse
import glob
import json
import math
import os
import platform
import subprocess
import time
from collections import defaultdict

from tasks.featurematch import blackbox_control, recipe_baseline, reference_solver
from tasks.featurematch.grader import grade
from tasks.featurematch.local_shim import LocalEpisode, ToolError
from tasks.featurematch.tools import Env

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 3), round((c + h) / d, 3))


def redact(rec):
    """Drop everything slot-level (which slots are null/planted is part of the answer key)."""
    keep = {k: v for k, v in rec.items() if k not in ("kinds", "slots", "tries", "submission")}
    if "tries" in rec:
        keep["tries"] = [{k: v for k, v in t.items() if k != "slots"} for t in rec["tries"]]
    return keep


def episode(d, profile, fn):
    ep = LocalEpisode(Env(), d, profile)
    t0 = time.time()
    err = None
    try:
        fn(ep.call)
    except ToolError as e:
        err = f"ToolError: {e}"
    except Exception as e:     # a solver crash is recorded, never hidden
        err = f"{type(e).__name__}: {e}"
    g = grade(d, ep.submission)
    return {"grade": g, "used": ep.used, "secs": round(time.time() - t0, 2), "error": err,
            "submission": ep.submission}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="prelim")
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--solvers", default="reference,blackbox," + ",".join(recipe_baseline.VARIANTS))
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--tiers", default="", help="comma list, e.g. T3 (default: all)")
    a = ap.parse_args()
    dirs = a.instances or sorted(glob.glob(os.path.join(HERE, "instances", "*")))
    if a.tiers:
        dirs = [d for d in dirs if os.path.basename(d).split("-")[1].upper() in a.tiers.split(",")]
    if a.max:
        dirs = dirs[:a.max]
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    rd = os.path.join(REPO, "runs", "featurematch", f"{ts}_{a.label}")
    os.makedirs(rd, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    import torch
    import transformers
    json.dump({"git_sha": sha, "task": "featurematch", "mode": "in-process (local_shim), harness not READY",
               "solvers": a.solvers.split(","), "instance_ids": [os.path.basename(x) for x in dirs],
               "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               "model": "google/gemma-2-2b (bf16 local copy)", "sae": "google/gemma-scope-2b-pt-res width_16k L6/12/18"},
              open(os.path.join(rd, "config.json"), "w"), indent=1)
    # episodes_private.jsonl has per-slot planted/null detail (a partial answer key; gitignored, D6).
    # episodes.jsonl is the redacted, committable version (episode-level outcomes only).
    out = open(os.path.join(rd, "episodes_private.jsonl"), "w")
    pub_out = open(os.path.join(rd, "episodes.jsonl"), "w")
    stats = defaultdict(lambda: defaultdict(list))
    slot_stats = defaultdict(lambda: defaultdict(list))
    for d in dirs:
        inst = json.load(open(os.path.join(d, "instance.json")))
        tier = inst["tier"]
        kinds = [s["kind"] for s in inst["extra"]["slots"]]
        for solver in a.solvers.split(","):
            if solver == "reference":
                runs = []
                for seed in range(5):
                    r = episode(d, "full", lambda call, s=seed: reference_solver.solve(call, seed=s))
                    runs.append(r)
                    if r["grade"]["pass"]:
                        break
                one, best = runs[0]["grade"]["pass"], any(r["grade"]["pass"] for r in runs)
                stats[tier]["reference_one_shot"].append(one)
                stats[tier]["reference_best_of_5"].append(best)
                stats[tier]["reference_forward_used"].append(runs[0]["used"]["forward"])
                for sl, k in zip(runs[0]["grade"]["details"]["slots"], kinds):
                    slot_stats[tier][f"reference/{k}"].append(sl["correct"])
                rec = {"instance": os.path.basename(d), "tier": tier, "solver": "reference", "kinds": kinds,
                       "one_shot": one, "best_of_5": best, "n_tries": len(runs),
                       "tries": [{"pass": r["grade"]["pass"], "score": r["grade"]["score"], "used": r["used"],
                                  "secs": r["secs"], "error": r["error"],
                                  "slots": r["grade"]["details"]["slots"]} for r in runs]}
            else:
                if solver == "blackbox":
                    r = episode(d, "blackbox", lambda call: blackbox_control.solve(call))
                else:
                    r = episode(d, "full", lambda call, v=solver: recipe_baseline.solve(call, v))
                stats[tier][solver].append(r["grade"]["pass"])
                for sl, k in zip(r["grade"]["details"]["slots"], kinds):
                    slot_stats[tier][f"{solver}/{k}"].append(sl["correct"])
                rec = {"instance": os.path.basename(d), "tier": tier, "solver": solver, "kinds": kinds,
                       "pass": r["grade"]["pass"], "score": r["grade"]["score"], "used": r["used"], "secs": r["secs"],
                       "error": r["error"], "slots": r["grade"]["details"]["slots"]}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            pub_out.write(json.dumps(redact(rec)) + "\n")
            pub_out.flush()
        print(os.path.basename(d), {s: (v[-1] if v else None) for s, v in stats[tier].items()}, flush=True)
    summ = {}
    for tier, st in stats.items():
        summ[tier] = {}
        for k, v in st.items():
            if k.endswith("forward_used"):
                summ[tier][k] = {"mean": round(sum(v) / len(v), 1), "max": max(v)}
            else:
                summ[tier][k] = {"rate": round(sum(v) / len(v), 3), "k": int(sum(v)), "n": len(v),
                                 "wilson95": wilson(int(sum(v)), len(v))}
        summ[tier]["slot_accuracy"] = {k: round(sum(v) / len(v), 3) for k, v in slot_stats[tier].items()}
    json.dump(summ, open(os.path.join(rd, "summary.json"), "w"), indent=1)
    lines = [f"# FeatureMatch preliminary gates ({a.label}, in-process)", "", f"run dir: {rd}", ""]
    for tier, st in summ.items():
        lines.append(f"## {tier}")
        lines.append("| solver | pass rate | k/n | Wilson 95% |")
        lines.append("|---|---|---|---|")
        for k, v in st.items():
            if isinstance(v, dict) and "rate" in v:
                lines.append(f"| {k} | {v['rate']} | {v['k']}/{v['n']} | {v['wilson95']} |")
        lines.append("")
    open(os.path.join(rd, "summary.md"), "w").write("\n".join(lines))
    print(json.dumps(summ, indent=1))
    print("RUN_DIR", rd)


if __name__ == "__main__":
    main()
