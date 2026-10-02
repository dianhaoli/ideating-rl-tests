"""Replay the finish-time sandbox leak scan (common.sandbox._scan_sandbox) before and after commit 5d0f6fe1.

    python runs/_harness/audit_replay_2026-10-02/replay_sandbox_scan.py

The commit changed how the task codename is treated in files the agent wrote (counted instead of a leak, unless in
a repo-path/canary form). This runs both versions on every privileged episode record whose sandbox still exists
(read-only) and compares leak verdicts and reasons. Writes only sandbox_scan_replay.json next to this script
(counts and episode ids; no file contents, no private strings).
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

WORKER = r'''
import glob, json, os, sys
sys.path.insert(0, sys.argv[1])
from common import sandbox
assert sandbox.__file__.startswith(sys.argv[1]), sandbox.__file__
res = {}
for rp in sorted(glob.glob(os.path.join(sys.argv[3], "ep*.json"))):
    try:
        rec = json.load(open(rp))
    except ValueError:
        continue
    eid = rec.get("episode") or os.path.basename(rp)[:-5]
    if not rec.get("sandbox") or not os.path.isdir(rec["sandbox"]):
        res[eid] = {"missing": True}
        continue
    try:
        r = sandbox._scan_sandbox(rec)
    except Exception as e:
        res[eid] = {"error": repr(e)[:200]}
        continue
    res[eid] = {"leak": r["leak"], "reasons": r["reasons"], "n_files": r["n_files"],
                "n_agent_files_with_codename": len(r.get("agent_files_with_codename") or [])}
json.dump(res, open(sys.argv[2], "w"))
'''


def main():
    tmp = tempfile.mkdtemp(prefix="scan_replay_")
    outs = {}
    procs = []
    for tag, rev in (("old", "5d0f6fe1~1"), ("new", "5d0f6fe1")):
        d = os.path.join(tmp, tag)
        os.makedirs(d)
        subprocess.run(f"git -C {REPO} archive {rev} common | tar -x -C {d}", shell=True, check=True)
        outs[tag] = os.path.join(tmp, tag + ".json")
        procs.append(subprocess.Popen([sys.executable, "-c", WORKER, d, outs[tag],
                                       os.path.join(REPO, "runs", ".episodes")]))
    for p in procs:
        assert p.wait() == 0
    old, new = (json.load(open(outs[t])) for t in ("old", "new"))
    diff = sorted(k for k in old if (old[k].get("leak"), old[k].get("reasons"), old[k].get("missing")) !=
                  (new.get(k, {}).get("leak"), new.get(k, {}).get("reasons"), new.get(k, {}).get("missing")))
    summ = {"records": len(old), "sandbox_missing": sum(1 for v in new.values() if v.get("missing")),
            "errors_old": sum(1 for v in old.values() if "error" in v),
            "errors_new": sum(1 for v in new.values() if "error" in v),
            "files_scanned": sum(v.get("n_files", 0) for v in new.values()),
            "leak_old": sum(1 for v in old.values() if v.get("leak")),
            "leak_new": sum(1 for v in new.values() if v.get("leak")),
            "episodes_with_codename_in_agent_files_new": sum(1 for v in new.values()
                                                              if v.get("n_agent_files_with_codename")),
            "verdict_or_reason_changes": diff}
    with open(os.path.join(HERE, "sandbox_scan_replay.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
