"""Thin CLI over common.sandbox for FeatureMatch episodes (HARNESS_API.md sections 5 and 9).

Run from the worktree root after `source ~/ideating-rl-tests/common/env.sh`.

  # one LLM test-agent episode: prepare prints the episode id, then the exact prompt for a FRESH subagent
  $PY -m tasks.featurematch.run_agent prepare --instance fm-t2-xxxxxxxxxx --label smoke --solver-label opus
  $PY -m tasks.featurematch.run_agent finish --episode epXXXXXXXXXX --agent-model claude-opus-5-5 [--transcript P]

  # the whole smoke plan (3 instances per tier, smoke_plan.json): prepares every episode and prints the prompts
  $PY -m tasks.featurematch.run_agent smoke --label smoke --solver-label opus

  # scripted gates (all through `common.sandbox run-scripted`, one run dir per solver)
  $PY -m tasks.featurematch.run_agent gates --tiers T1,T2,T3 --label gates [--only reference,blackbox,recipes]

  $PY -m tasks.featurematch.run_agent summarize --run-dir runs/featurematch/<ts>_<label>

Nothing here reads answers; grading happens in common.sandbox.finish, which runs grader.py out of process.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TASK = "featurematch"
PY = os.environ.get("PY", sys.executable)
RECIPES = ["nothing", "always_claim", "prior", "prior_or_none", "random", "name_probe", "name_probe_thr",
           "vocab_match"]


def ts():
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def run_dir(label):
    return os.path.join(ROOT, "runs", TASK, f"{ts()}_{label}")


def inst_path(x):
    if os.path.isdir(x):
        return os.path.abspath(x)
    return os.path.join(HERE, "instances", x)


def kept_instances(tiers):
    """Instance dirs of the current pool in the given tiers, minus any marked kept=false in the manifest."""
    man = json.load(open(os.path.join(HERE, "instances_manifest.json")))["instances"]
    return [os.path.join(HERE, "instances", m["instance_id"]) for m in man
            if m["tier"] in tiers and m.get("kept", True)]


def sandbox(*args, env=None):
    cmd = [PY, "-m", "common.sandbox", *args]
    print("+", " ".join(cmd[:8]), "..." if len(cmd) > 8 else "", flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, **(env or {}))).returncode


def cmd_prepare(a):
    rd = a.run_dir or run_dir(a.label)
    return sandbox("prepare", "--task", TASK, "--instance-dir", inst_path(a.instance), "--profile", a.profile,
                   "--run-dir", rd, "--solver-label", a.solver_label)


def cmd_smoke(a):
    plan = json.load(open(os.path.join(HERE, "smoke_plan.json")))
    rd = a.run_dir or run_dir(a.label)
    for tier, insts in sorted(plan["tiers"].items()):
        for d in insts:
            sandbox("prepare", "--task", TASK, "--instance-dir", os.path.join(ROOT, d), "--profile",
                    plan.get("profile", "full"), "--run-dir", rd, "--solver-label", a.solver_label)
    print(f"run dir: {rd}")


def cmd_finish(a):
    args = ["finish", "--episode", a.episode]
    if a.agent_model:
        args += ["--agent-model", a.agent_model]
    if a.transcript:
        args += ["--transcript", *a.transcript]
    return sandbox(*args)


def cmd_gates(a):
    tiers = a.tiers.split(",")
    insts = kept_instances(tiers)
    only = set(a.only.split(",")) if a.only else {"reference", "blackbox", "recipes"}
    out = {}
    if "reference" in only:
        rd = a.run_dir_prefix + "reference" if a.run_dir_prefix else run_dir(f"{a.label}_reference")
        sandbox("run-scripted", "--task", TASK, "--solver", os.path.join(HERE, "reference_solver.py"),
                "--instances", *insts, "--profile", "full", "--run-dir", rd, "--repeats", str(a.ref_repeats),
                "--solver-label", "reference")
        out["reference"] = rd
    if "blackbox" in only:
        rd = run_dir(f"{a.label}_blackbox")
        sandbox("run-scripted", "--task", TASK, "--solver", os.path.join(HERE, "blackbox_control.py"),
                "--instances", *insts, "--profile", "blackbox", "--run-dir", rd, "--solver-label", "blackbox")
        out["blackbox"] = rd
    if "recipes" in only or any(r in only for r in RECIPES):
        variants = RECIPES if "recipes" in only else [r for r in RECIPES if r in only]
        rd = run_dir(f"{a.label}_recipes")
        for v in variants:
            sandbox("run-scripted", "--task", TASK, "--solver", os.path.join(HERE, "recipe_baseline.py"),
                    "--instances", *insts, "--profile", "full", "--run-dir", rd, "--solver-label", f"recipe_{v}",
                    env={"RL_RECIPE": v})
        out["recipes"] = rd
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(prog="python -m tasks.featurematch.run_agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--instance", required=True, help="instance id (in instances/) or instance dir")
    p.add_argument("--profile", default="full")
    p.add_argument("--label", default="agent")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--solver-label", default="opus")
    s = sub.add_parser("smoke")
    s.add_argument("--label", default="smoke")
    s.add_argument("--run-dir", default=None)
    s.add_argument("--solver-label", default="opus")
    f = sub.add_parser("finish")
    f.add_argument("--episode", required=True)
    f.add_argument("--agent-model", default=None)
    f.add_argument("--transcript", nargs="*", default=None)
    g = sub.add_parser("gates")
    g.add_argument("--tiers", default="T1,T2,T3")
    g.add_argument("--label", default="gates")
    g.add_argument("--only", default=None, help="comma list of reference, blackbox, recipes or recipe variant names")
    g.add_argument("--ref-repeats", type=int, default=1)
    g.add_argument("--run-dir-prefix", default=None)
    m = sub.add_parser("summarize")
    m.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "prepare":
        sys.exit(cmd_prepare(a))
    if a.cmd == "smoke":
        cmd_smoke(a)
    elif a.cmd == "finish":
        sys.exit(cmd_finish(a))
    elif a.cmd == "gates":
        cmd_gates(a)
    elif a.cmd == "summarize":
        sys.exit(sandbox("summarize", "--run-dir", a.run_dir))


if __name__ == "__main__":
    main()
