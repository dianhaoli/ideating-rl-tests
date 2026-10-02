"""Step 3 baselines on the filtered pool, IN-PROCESS: a thin adapter over tasks/featurematch/inproc_gates.py.

PREREG step 3 runs the reference solver, the black-box control, self_probe, template_probe and the zero-effort recipes
on all 180 filtered instances (tasks/featurematch/instances_v2f). The harness path takes 15-40 min per GPU episode on
the shared queue, so this runs everything as ONE gpuq job with the model loaded once, exactly as inproc_gates.py does:
the same solver code, the same Env class, profiles and per-instance caps (common.toolserver.make_local_call), the same
`budget` / `submit` built-ins (inproc_gates.make_call) and the out-of-process grader (inproc_gates.grade_out_of_process).
Every number from this script is labelled "in-process".

What this adapter adds (task files are not edited):
  1. Reference corpus. reference_solver.load_corpus() reads tasks/featurematch/cache/concepts.json, which this worktree
     does not have (gitignored caches). The adapter reads the same file from the fix-stage worktree
     (~/wt/featurematch/tasks/featurematch/cache, read-only), checks its sha256 against the concepts_sha256 recorded
     in diagnosis/instances_v2f_manifest.json, and passes it as `corpus=`.
  2. Two priors. `recipe_prior` / `recipe_prior_or_none` read tasks/featurematch/prior.json (unfiltered v2 generator,
     seeds 900000+). `recipe_prior@v2f` / `recipe_prior_or_none@v2f` run the same recipe code with
     recipe_baseline.load_prior replaced by diagnosis/step3/prior_v2f.json (filtered generator, disjoint seeds; see
     build_prior_v2f.py).
  3. generate memo (speed only). self_probe calls `generate` (greedy, one prompt per call, max_new_tokens=48) with
     prompts that depend only on the option label, so the same prompt recurs across instances. The adapter memoises the
     model computation behind `generate` by (prompt, max_new_tokens). The tool still validates and CHARGES every call
     (the memo sits behind Env.compute, after the charge). The first N_VERIFY memo hits are recomputed and compared;
     mismatches are counted in run_meta.json.
  4. Reference best-of-5 as inproc_gates: seeds 7, 1007, ... stop at the first pass. One-shot = seed 7 on every
     instance. All solvers other than the reference run once (seed 7).
  5. Per episode it also stores the submission (private file, gitignored) so the slots can be re-graded offline.

Run (from the worktree root):
  FM_MODEL_SERVICE=off $PY -m common.gpuq run --gb 7 --heavy --label fm-diag-step3-inproc -- \
      $PY -m tasks.featurematch.diagnosis.step3.inproc_step3 --out tasks/featurematch/diagnosis/step3/runs/<ts>_inproc
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

from tasks.featurematch import inproc_gates as IG

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
TASK = os.path.dirname(DIAG)
POOL = os.path.join(TASK, "instances_v2f")
MANIFEST = os.path.join(DIAG, "instances_v2f_manifest.json")
FM_MAIN_CACHE = os.path.expanduser("~/wt/featurematch/tasks/featurematch/cache")
PRIOR_V2F = os.path.join(HERE, "prior_v2f.json")
N_VERIFY = 25

# cheap first, reference next, the slow generate-based recipe last (the run is resumable)
SOLVERS = ["recipe_nothing", "recipe_always_claim", "recipe_prior", "recipe_prior_or_none", "recipe_prior@v2f",
           "recipe_prior_or_none@v2f", "recipe_random", "recipe_name_probe", "recipe_name_probe_thr",
           "recipe_vocab_match", "blackbox", "recipe_template_probe", "reference", "recipe_self_probe"]

_CORPUS = None
_PRIOR_V2F = None
GEN_MEMO = {}
MEMO = {"misses": 0, "hits": 0, "verified": 0, "mismatches": 0}


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def verify_pool(pool=POOL, manifest=MANIFEST):
    """Every instance file must match the step-2 manifest; returns the pool's identity."""
    m = json.load(open(manifest))
    ids = set()
    for e in m["instances"]:
        d = os.path.join(pool, e["instance_id"])
        ids.add(e["instance_id"])
        for f, h in e["files"].items():
            if sha256_file(os.path.join(d, f)) != h:
                raise SystemExit(f"pool file {d}/{f} does not match {manifest}")
    found = {os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(pool, "*", "instance.json"))}
    if found != ids:
        raise SystemExit(f"pool dirs differ from manifest: extra {sorted(found - ids)[:3]}, missing "
                         f"{sorted(ids - found)[:3]}")
    h = hashlib.sha256()                         # same digest as sr_recipe.pool_digest
    for p in sorted(glob.glob(os.path.join(pool, "*", "instance.json"))):
        h.update(os.path.basename(os.path.dirname(p)).encode())
        h.update(open(p, "rb").read())
    return {"n_instances": len(ids), "manifest_sha256": sha256_file(manifest), "pool_sha256": h.hexdigest(),
            "concepts_sha256": m.get("concepts_sha256"), "key_check_sha256": m.get("key_check_sha256")}


def corpus(expected_sha):
    global _CORPUS
    if _CORPUS is None:
        p = os.path.join(FM_MAIN_CACHE, "concepts.json")
        blob = open(p, "rb").read()
        cs = json.loads(blob)
        got = hashlib.sha256(blob).hexdigest()          # concepts.py writes exactly the hashed blob
        if not expected_sha or got != expected_sha:
            raise SystemExit(f"concepts.json sha256 {got} != manifest concepts_sha256 {expected_sha}")
        _CORPUS = {c["label"]: c["C"] for c in cs if c["C"]}       # = reference_solver.load_corpus()
    return _CORPUS


def prior_v2f():
    global _PRIOR_V2F
    if _PRIOR_V2F is None:
        _PRIOR_V2F = json.load(open(PRIOR_V2F))
    return _PRIOR_V2F


def memoise_generate(env):
    orig = env.compute

    def compute(op, **args):
        if op != "generate":
            return orig(op, **args)
        key = (args["prompt"], int(args["max_new_tokens"]))
        if key in GEN_MEMO:
            MEMO["hits"] += 1
            if MEMO["verified"] < N_VERIFY:
                MEMO["verified"] += 1
                MEMO["mismatches"] += orig(op, **args)["completion"] != GEN_MEMO[key]["completion"]
            return dict(GEN_MEMO[key])
        r = orig(op, **args)
        GEN_MEMO[key] = dict(r)
        MEMO["misses"] += 1
        return r
    env.compute = compute          # instance attribute shadows Env.compute; the tools call self.compute


def run_one(instance_dir, solver, seed, ident):
    """inproc_gates.run_one plus the corpus / prior / memo additions above."""
    from common.broker import merged_caps
    from tasks.featurematch import blackbox_control, recipe_baseline, reference_solver
    from tasks.featurematch.tools import Env
    inst = json.load(open(os.path.join(instance_dir, "instance.json")))
    caps = {k: v for k, v in merged_caps(inst.get("caps")).items() if k not in ("wall_clock_s", "call_timeout_s")}
    profile = "blackbox" if solver == "blackbox" else "full"
    env = Env(instance_dir=instance_dir, sandbox_dir=None, profile=profile)
    env.load(instance_dir)
    memoise_generate(env)
    call, state, raw = IG.make_call(env, caps)
    t0 = time.time()
    err = None
    try:
        if solver == "reference":
            reference_solver.solve(call, seed=seed, corpus=corpus(ident["concepts_sha256"]))
        elif solver == "blackbox":
            blackbox_control.solve(call, seed=seed)
        elif solver.startswith("recipe_") and solver.endswith("@v2f"):
            variant = solver[len("recipe_"):-len("@v2f")]
            assert variant in ("prior", "prior_or_none")
            orig = recipe_baseline.load_prior
            recipe_baseline.load_prior = lambda: dict(prior_v2f())
            try:
                recipe_baseline.solve(call, variant, seed=seed)
            finally:
                recipe_baseline.load_prior = orig
        elif solver.startswith("recipe_"):
            recipe_baseline.solve(call, solver[len("recipe_"):], seed=seed)
        else:
            raise SystemExit(f"unknown solver {solver}")
    except Exception as e:          # a solver crash is recorded; the episode is graded on whatever was submitted
        err = f"{type(e).__name__}: {e}"
    g = IG.grade_out_of_process(instance_dir, state["submission"])
    return {"instance_id": inst["instance_id"], "tier": inst["tier"], "solver": solver, "seed": seed,
            "pass": bool(g["pass"]), "score": g["score"], "slots": g["details"]["slots"],
            "submitted": state["submission"] is not None, "submission": state["submission"], "error": err,
            "used": dict(raw.used), "caps": {k: caps[k] for k in ("tool_calls", "forward", "generate") if k in caps},
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--solvers", default=",".join(SOLVERS))
    ap.add_argument("--ref-tries", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="first N instances only (smoke test)")
    a = ap.parse_args()
    if os.path.exists("TASK.md"):
        raise SystemExit("a TASK.md in the cwd would change the model-free recipes' slot source; run from the root")
    ident = verify_pool()
    insts = sorted(glob.glob(os.path.join(POOL, "*", "")))
    if a.limit:
        insts = insts[:a.limit]
    from common import gpuq
    import torch  # noqa: F401  (loaded before the caps are applied, as inproc_gates does)
    gpuq.apply_caps()
    from tasks.featurematch import tools
    os.makedirs(a.out, exist_ok=True)
    priv = os.path.join(a.out, "episodes_private.jsonl")
    rows, done = [], set()
    if os.path.exists(priv):            # resumable
        for line in open(priv):
            r = json.loads(line)
            rows.append(r)
            done.add((r["instance_id"], r["solver"], r["seed"]))
    t_start = time.time()
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
                        r = run_one(os.path.abspath(d), solver, seed, ident)
                        f.write(json.dumps(r) + "\n")
                        f.flush()
                        rows.append(r)
                        print(json.dumps({k2: r[k2] for k2 in ("solver", "seed", "error", "seconds")}), flush=True)
                    if r["pass"]:
                        break
            print(f"[step3] solver {solver} done at {round(time.time() - t_start)} s; memo {MEMO}", flush=True)
    meta = {"label": "in-process (inproc_gates.py path via diagnosis/step3/inproc_step3.py; NOT through the broker)",
            "pool": ident, "n_instances_run": len(insts), "solvers": a.solvers.split(","), "ref_tries": a.ref_tries,
            "seeds": "1000*k+7 (k=0 for every solver; reference k=0..4, stop at first pass)",
            "model_service_socket": tools.SERVICE_SOCK,
            "corpus": os.path.join(FM_MAIN_CACHE, "concepts.json") + " (sha256 checked vs manifest concepts_sha256)",
            "prior_v2f_sha256": sha256_file(PRIOR_V2F) if os.path.exists(PRIOR_V2F) else None,
            "prior_v1_sha256": sha256_file(os.path.join(TASK, "prior.json")),
            "generate_memo": dict(MEMO, distinct_prompts=len(GEN_MEMO), n_verify=N_VERIFY),
            "n_episodes": len(rows), "n_crashed": sum(r["error"] is not None for r in rows),
            "elapsed_s": round(time.time() - t_start, 1), "git_sha": IG.subprocess.run(
                ["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    json.dump(meta, open(os.path.join(a.out, "run_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    sys.exit(0)


if __name__ == "__main__":
    main()
