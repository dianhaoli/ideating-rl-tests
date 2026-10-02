"""Check that the shared model service gives the same tool outputs as the in-process model.

Two passes over the SAME instances and the SAME fixed list of tool calls (every FeatureMatch tool, all three layers,
the instances' own slot latents plus random latents, truncation and unicode texts, a few argument errors):

  1. record (in-process model; one normal GPU job):
     FM_MODEL_SERVICE=off $PY -m common.gpuq run --gb 8 --label fm-service-check-inproc -- \
         $PY -m tasks.featurematch.diagnosis.compare_service record --instances D1 D2 D3 --out /tmp/fm_ref.json
  2. check (CPU only; needs the service running, see diagnosis/MODEL_SERVICE.md):
     $PY -m tasks.featurematch.diagnosis.compare_service check --instances D1 D2 D3 --ref /tmp/fm_ref.json \
         --report tasks/featurematch/diagnosis/model_service_check.json

The report counts exact matches and the largest numeric difference per tool. Pass rule: every string (tokens,
completions, decoded vocab tokens), every latent index and every error message identical; every number within
1e-3 + 1e-2 * |value| (bf16 tolerance). The recorded outputs themselves stay in /tmp (not committed).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

TEXTS = [
    "The striker scored twice in the second half as the visitors held on to win the cup final.",
    "Il pleut depuis trois jours et la rivière a débordé dans le village.",
    "def add(a, b):\n    return a + b  # simple helper",
    "東京の天気は明日晴れるでしょう。",
    "lol did u see that game last night?? absolutely wild ending",
    " ".join(["The committee reviewed the annual budget and approved several new projects for the region."] * 8),
]
PROMPTS = ["The capital of France is", "Once upon a time, in a small village,", "2 + 2 =",
           "The most important thing about writing code is"]


def calls_for(public, rng):
    """The fixed call list for one instance: (tool, args)."""
    out = [("task_info", {})]
    slot_lat = {}
    for s in public["slots"]:
        slot_lat.setdefault(int(s["layer"]), []).append(int(s["latent"]))
    for L in (6, 12, 18):
        lats = (slot_lat.get(L, []) + [int(x) for x in rng.integers(0, 16384, 8)])[:8]
        out.append(("latent_activations", {"texts": TEXTS, "layer": L, "latents": lats}))
        out.append(("latent_activations", {"texts": TEXTS[:3], "layer": L, "latents": lats[:2], "per_token": False}))
        out.append(("top_latents", {"texts": TEXTS, "layer": L, "k": 20}))
        for lat in lats[:3]:
            out.append(("vocab_projection", {"layer": L, "latent": lat, "k": 25}))
    for p in PROMPTS[:2]:
        out.append(("generate", {"prompt": p, "max_new_tokens": 32}))
    out.append(("next_token_logits", {"prompts": PROMPTS, "top_k": 20}))
    # argument errors must be identical too
    out.append(("latent_activations", {"texts": TEXTS[:1], "layer": 7, "latents": [1]}))
    out.append(("top_latents", {"texts": [], "layer": 6}))
    out.append(("vocab_projection", {"layer": 6, "latent": 99999}))
    return out


def run_all(instances):
    from common.toolserver import ToolError, make_local_call
    from tasks.featurematch import tools
    from tasks.featurematch.tools import Env
    mode = "service" if tools.SERVICE_SOCK else "in-process"
    res = {"mode": mode, "gpu_gb": Env.GPU_GB, "instances": {}}
    for d in instances:
        public = json.load(open(os.path.join(d, "public.json")))
        iid = os.path.basename(d.rstrip("/"))
        env = Env(instance_dir=d, sandbox_dir=None, profile="full")
        env.load(d)
        call = make_local_call(env, {})
        rows = []
        t0 = time.time()
        for tool, args in calls_for(public, np.random.default_rng(1234)):
            try:
                rows.append({"tool": tool, "args": args, "ok": True, "result": call(tool, **args)})
            except ToolError as e:
                rows.append({"tool": tool, "args": args, "ok": False, "error": str(e)})
        res["instances"][iid] = {"rows": rows, "seconds": round(time.time() - t0, 1)}
        print(f"{mode}: {iid}: {len(rows)} calls in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    return res


def compare(a, b, path, st):
    """a = reference (in-process), b = service. Updates st (counters + max diffs)."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            st["mismatch"].append(f"{path}: keys differ")
            return
        for k in a:
            compare(a[k], b[k], f"{path}.{k}", st)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            st["mismatch"].append(f"{path}: list length differs")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", st)
    elif isinstance(a, bool) or isinstance(a, str) or a is None or (isinstance(a, int) and ".latent" in path):
        st["n_exact_fields"] += 1
        if a != b:
            st["mismatch"].append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, (int, float)):
        st["n_numbers"] += 1
        d = abs(float(a) - float(b))
        if d == 0:
            st["n_numbers_identical"] += 1
        st["max_abs_diff"] = max(st["max_abs_diff"], d)
        if d > 1e-3 + 1e-2 * abs(float(a)):
            st["mismatch"].append(f"{path}: {a} vs {b} (diff {d:.3g})")
    else:
        st["mismatch"].append(f"{path}: unexpected type {type(a).__name__}")


def check(ref, cur):
    per_tool = {}
    for iid, r in ref["instances"].items():
        c = cur["instances"].get(iid)
        if c is None:
            per_tool.setdefault("_missing", {"mismatch": []})["mismatch"].append(iid)
            continue
        for i, (x, y) in enumerate(zip(r["rows"], c["rows"])):
            st = per_tool.setdefault(x["tool"], {"n_calls": 0, "n_exact_fields": 0, "n_numbers": 0,
                                                 "n_numbers_identical": 0, "max_abs_diff": 0.0, "mismatch": []})
            st["n_calls"] += 1
            assert x["args"] == y["args"]
            compare({k: x.get(k) for k in ("ok", "result", "error")}, {k: y.get(k) for k in ("ok", "result", "error")},
                    f"{iid}#{i}:{x['tool']}", st)
    for st in per_tool.values():
        st["n_mismatch"] = len(st["mismatch"])
        st["mismatch"] = st["mismatch"][:20]
    return per_tool


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--instances", nargs="+", required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--instances", nargs="+", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--report", required=True)
    a = ap.parse_args()
    if a.cmd == "record":
        res = run_all(a.instances)
        if res["mode"] != "in-process":
            raise SystemExit("record must run with the in-process model (set FM_MODEL_SERVICE=off)")
        json.dump(res, open(a.out, "w"))
        return
    ref = json.load(open(a.ref))
    t0 = time.time()
    cur = run_all(a.instances)
    if cur["mode"] != "service":
        raise SystemExit("check must run against a live model service")
    per_tool = check(ref, cur)
    total_mis = sum(v.get("n_mismatch", 0) for v in per_tool.values())
    rep = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "instances": [os.path.basename(x.rstrip("/")) for x in a.instances],
           "n_calls": sum(v.get("n_calls", 0) for v in per_tool.values()),
           "service_gpu_gb_per_episode_server": cur["gpu_gb"], "reference_gpu_gb": ref["gpu_gb"],
           "tolerance": "strings/latent ids/errors exact; numbers |d| <= 1e-3 + 1e-2*|ref|",
           "pass": total_mis == 0, "n_mismatch": total_mis,
           "seconds_service": {k: v["seconds"] for k, v in cur["instances"].items()},
           "seconds_inprocess": {k: v["seconds"] for k, v in ref["instances"].items()},
           "per_tool": per_tool, "check_wall_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("pass", "n_calls", "n_mismatch")}))
    for t, v in per_tool.items():
        print(f"  {t}: calls={v.get('n_calls')} numbers={v.get('n_numbers')} identical={v.get('n_numbers_identical')} "
              f"max_abs_diff={v.get('max_abs_diff')} mismatches={v.get('n_mismatch')}")


if __name__ == "__main__":
    main()
