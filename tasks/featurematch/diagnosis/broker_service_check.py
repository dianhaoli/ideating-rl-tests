"""End-to-end check through the REAL broker while the shared model service runs: prepare several FeatureMatch
episodes, call every tool on all of them at the same time (threads), and verify that
  (a) no per-episode tool server entered the GPU queue (they are CPU-only),
  (b) the responses equal the in-process reference recorded by compare_service.py (same args, same tolerance),
  (c) the service's request counter went up.
Episodes are finished as scripted (no transcript) under a throwaway run dir.

    $PY -m tasks.featurematch.diagnosis.broker_service_check --instances D1 D2 D3 --ref /tmp/fm_svc_ref.json \
        --run-dir /tmp/fm_svc_broker --report tasks/featurematch/diagnosis/model_service_broker_check.json
"""
import argparse
import json
import os
import sys
import threading
import time

from common import gpuq, sandbox
from common.toolclient import Client, ToolCallError
from tasks.featurematch import model_service as ms
from tasks.featurematch.diagnosis.compare_service import compare

TASK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    m = ms.read_marker()
    before = ms.ping(m["socket"]) if m else None
    if before is None:
        raise SystemExit("model service is not running")
    ref = json.load(open(a.ref))
    eps = {}
    for d in a.instances:
        iid = os.path.basename(d.rstrip("/"))
        ep = sandbox.prepare("featurematch", d, "full", a.run_dir, solver_label="service-check", tasks_root=TASK_ROOT)
        eps[iid] = ep["episode"]
    results, errors = {}, []
    ledger_labels = set()
    stop = threading.Event()

    def watch_ledger():
        while not stop.is_set():
            with gpuq._locked() as led:
                for j in led["jobs"].values():
                    ledger_labels.add(j.get("label"))
                for j in led.get("waiting", {}).values():
                    ledger_labels.add(j.get("label"))
            time.sleep(1)

    def run(iid, eid):
        c = Client(eid)
        rows = []
        for r in ref["instances"][iid]["rows"]:
            try:
                rows.append({"ok": True, "result": c.call(r["tool"], **r["args"]), "error": None})
            except ToolCallError as e:
                rows.append({"ok": False, "result": None, "error": str(e)})
        results[iid] = rows

    w = threading.Thread(target=watch_ledger, daemon=True)
    w.start()
    t0 = time.time()
    ths = [threading.Thread(target=run, args=(i, e)) for i, e in eps.items()]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    stop.set()
    w.join()
    st = {"n_exact_fields": 0, "n_numbers": 0, "n_numbers_identical": 0, "max_abs_diff": 0.0, "mismatch": []}
    n = 0
    for iid, rows in results.items():
        for i, (x, y) in enumerate(zip(ref["instances"][iid]["rows"], rows)):
            n += 1
            # the broker prefixes nothing; ToolCallError carries the server's message
            compare({"ok": x["ok"], "result": x.get("result"), "error": x.get("error")},
                    {"ok": y["ok"], "result": y["result"], "error": y["error"] if not x["ok"] else None},
                    f"{iid}#{i}:{x['tool']}", st)
    graded = {}
    for iid, eid in eps.items():
        g = sandbox.finish(eid)
        rec = json.load(open(os.path.join(a.run_dir, "episodes", eid, "episode.json")))
        graded[iid] = {"episode": eid, "valid": g["harness"]["valid"], "gpu_gb_declared": rec.get("gpu_gb"),
                       "counters": g["harness"]["counters"]}
    after = ms.ping(m["socket"])
    fm_server_jobs = sorted(l for l in ledger_labels if l and l.startswith("toolserver:featurematch"))
    rep = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "episodes": graded,
           "n_calls": n, "concurrent_episodes": len(eps), "wall_s": round(wall, 1),
           "service_requests_served": after["served"] - before["served"],
           "featurematch_toolserver_gpu_jobs_seen": fm_server_jobs,
           "n_mismatch": len(st["mismatch"]), "mismatch": st["mismatch"][:20], "max_abs_diff": st["max_abs_diff"],
           "n_numbers": st["n_numbers"], "n_numbers_identical": st["n_numbers_identical"],
           "pass": not st["mismatch"] and not fm_server_jobs and after["served"] > before["served"]}
    json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("pass", "n_calls", "n_mismatch", "wall_s", "service_requests_served",
                                          "featurematch_toolserver_gpu_jobs_seen")}))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
