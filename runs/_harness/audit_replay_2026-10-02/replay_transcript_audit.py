"""Replay the transcript audit before and after commit 5d0f6fe1 on every recorded LLM-agent transcript.

    python runs/_harness/audit_replay_2026-10-02/replay_transcript_audit.py [--old REV] [--new REV]

Read-only on every run dir (main repo runs/ and ~/wt/*/runs/). Writes only transcript_audit_replay.csv and
transcript_audit_replay.json next to this script. Both hold verdicts and rule names only (no commands, no
submissions, no answer material).
"""
import argparse
import collections
import csv
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def load_rev(rev, tmp):
    src = subprocess.run(["git", "-C", REPO, "show", f"{rev}:common/transcript_audit.py"], check=True,
                         capture_output=True, text=True).stdout
    p = os.path.join(tmp, f"ta_{rev.replace('~', '_').replace('/', '_')}.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(os.path.basename(p)[:-3], p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="5d0f6fe1~1")
    ap.add_argument("--new", default="5d0f6fe1")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="audit_replay_")
    old, new = load_rev(a.old, tmp), load_rev(a.new, tmp)
    roots = [os.path.join(REPO, "runs")] + sorted(glob.glob(os.path.expanduser("~/wt/*/runs")))
    dirs = []
    for r in roots:
        for e in sorted(glob.glob(r + "/**/episodes/*", recursive=True)):
            files = os.listdir(e)
            ts = [f for f in ("api_transcript.jsonl", "transcript.jsonl") if f in files]
            if not ts:
                continue
            ep = json.load(open(os.path.join(e, "episode.json")))
            eid = ep["episode"]
            sbx = ep.get("sandbox") or os.path.expanduser("~/rlsbx/" + eid)
            hashes = {f: hashlib.sha256(open(os.path.join(e, f), "rb").read()).hexdigest() for f in ts}
            tpaths = [os.path.join(e, ts[0])] if len(set(hashes.values())) == 1 else [os.path.join(e, f) for f in ts]
            o, n = old.audit(eid, sbx, tpaths), new.audit(eid, sbx, tpaths)
            key = lambda v: (v["i"], v["rule"], v["detail"])
            ov, nv = set(map(key, o["violations"])), set(map(key, n["violations"]))
            try:
                stored = json.load(open(os.path.join(e, "audit.json"))).get("valid")
            except (OSError, ValueError):
                stored = None
            dirs.append({"episode": eid, "task": ep.get("task"), "agent_model": ep.get("agent_model"),
                         "run": os.path.relpath(e, os.path.expanduser("~")).split("/episodes/")[0],
                         "transcript_sha256_16": hashes[ts[0]][:16], "n_tool_calls": n["n_tool_calls"],
                         "stored_valid": stored, "old_valid": o["valid"], "new_valid": n["valid"],
                         "removed": sorted(ov - nv), "added": sorted(nv - ov), "kept": sorted(ov & nv)})
    uniq = collections.OrderedDict()
    for d in dirs:
        u = uniq.setdefault(d["episode"], dict(d, copies=0))
        u["copies"] += 1
        assert u["transcript_sha256_16"] == d["transcript_sha256_16"] and u["old_valid"] == d["old_valid"] \
            and u["new_valid"] == d["new_valid"], d["episode"]
    rows = []
    for u in uniq.values():
        rc = lambda vs: ";".join(f"{k}x{c}" for k, c in sorted(collections.Counter(v[1] for v in vs).items()))
        rows.append({"episode": u["episode"], "task": u["task"], "agent_model": u["agent_model"], "run": u["run"],
                     "copies": u["copies"], "n_tool_calls": u["n_tool_calls"], "stored_valid": u["stored_valid"],
                     "old_valid": u["old_valid"], "new_valid": u["new_valid"],
                     "change": ("flagged->clean" if not u["old_valid"] and u["new_valid"] else
                                "clean->flagged" if u["old_valid"] and not u["new_valid"] else
                                "violations changed" if (u["removed"] or u["added"]) else "none"),
                     "calls_removed": len({v[0] for v in u["removed"]}), "rules_removed": rc(u["removed"]),
                     "rules_added": rc(u["added"]), "rules_still_firing": rc(u["kept"]),
                     "removed_details": " | ".join(sorted({v[2][:80] for v in u["removed"]}))})
    with open(os.path.join(HERE, "transcript_audit_replay.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = {"old": a.old, "new": a.new, "episode_dirs": len(dirs), "unique_episodes": len(uniq),
            "dirs_old_to_new": {f"{k[0]}->{k[1]}": v for k, v in
                                collections.Counter((d["old_valid"], d["new_valid"]) for d in dirs).items()},
            "unique_old_to_new": {f"{k[0]}->{k[1]}": v for k, v in
                                  collections.Counter((u["old_valid"], u["new_valid"]) for u in uniq.values()).items()},
            "call_level_removed": sum(len(u["removed"]) for u in uniq.values()),
            "call_level_added": sum(len(u["added"]) for u in uniq.values())}
    with open(os.path.join(HERE, "transcript_audit_replay.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
