"""Replay the transcript audit, OLD vs NEW, over every recorded transcript under ~/wt/*/runs/** (v3 harness fixes,
2026-10-04).

    $PY runs/_harness/audit_replay_v3fixes/replay_v3.py [--old REV] [--detail]

OLD = common/transcript_audit.py at REV (default HEAD of this checkout, i.e. before the uncommitted change; pass the
pre-fix commit after committing), loaded in memory; NEW = the working tree. Read-only over every input.

Units: every episode directory under ~/wt/*/runs/** that holds a recorded LLM transcript. The transcripts audited are
the ones `finish` used and copied (transcript.jsonl, transcript_<k>.jsonl, in order), else api_transcript.jsonl.
Units are deduplicated by (episode id, transcript bytes): the _demo episodes are tracked in git and appear in every
checkout. The sandbox comes from the unit's episode.json.

Writes replay_v3.json next to this script: aggregates plus one row per unit whose verdict or violation set changed
(episode id, task, run, old/new verdict, call indices and rule names of removed/added violations). No commands,
submissions or answer material. --detail prints a short context around each change (operator review only).
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
from common import transcript_audit as new  # noqa: E402


def load_rev(rev):
    src = subprocess.run(["git", "-C", REPO, "show", f"{rev}:common/transcript_audit.py"], check=True,
                         capture_output=True, text=True).stdout
    m = types.ModuleType("ta_old")
    m.__file__ = os.path.join(REPO, "common", "transcript_audit.py")
    exec(compile(src, f"{rev}:common/transcript_audit.py", "exec"), m.__dict__)
    return m


def units(also_main=False):
    seen, out = set(), []
    pats = [os.path.expanduser("~/wt/*/runs/**/episodes/ep*")]
    if also_main:
        from common.paths import MAIN_REPO
        pats.append(os.path.join(MAIN_REPO, "runs", "**", "episodes", "ep*"))
    eds = sorted({e for p in pats for e in glob.glob(p, recursive=True)})
    for ed in eds:
        if not os.path.isdir(ed) or "/_scratch" in ed:
            continue
        ts = sorted(glob.glob(os.path.join(ed, "transcript_*.jsonl")),
                    key=lambda p: int(os.path.basename(p)[11:-6]) if os.path.basename(p)[11:-6].isdigit() else 1 << 30)
        ts = ([os.path.join(ed, "transcript.jsonl")] if os.path.exists(os.path.join(ed, "transcript.jsonl")) else []) + ts
        if not ts and os.path.exists(os.path.join(ed, "api_transcript.jsonl")):
            ts = [os.path.join(ed, "api_transcript.jsonl")]
        if not ts:
            continue
        h = hashlib.sha256(b"".join(open(t, "rb").read() for t in ts)).hexdigest()[:16]
        eid = os.path.basename(ed)
        if (eid, h) in seen:
            continue
        seen.add((eid, h))
        try:
            sb = json.load(open(os.path.join(ed, "episode.json"))).get("sandbox")
        except (OSError, ValueError):
            sb = None
        out.append({"episode": eid, "edir": ed, "transcripts": ts, "sandbox": sb or os.path.expanduser("~/rlsbx/" + eid),
                    "sha16": h})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="HEAD")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "replay_v3.json"))
    ap.add_argument("--also-main", action="store_true", help="also replay the main checkout's runs/")
    a = ap.parse_args()
    old = load_rev(a.old)
    key = lambda v: (v["i"], v["rule"], v["detail"])
    rows, n_calls, verdicts = [], 0, collections.Counter()
    us = units(a.also_main)
    for u in us:
        o = old.audit(u["episode"], u["sandbox"], u["transcripts"])
        n = new.audit(u["episode"], u["sandbox"], u["transcripts"])
        n_calls += n["n_tool_calls"]
        verdicts[("clean" if o["valid"] else "flagged") + "->" + ("clean" if n["valid"] else "flagged")] += 1
        ov, nv = collections.Counter(map(key, o["violations"])), collections.Counter(map(key, n["violations"]))
        removed, added = sorted((ov - nv).elements()), sorted((nv - ov).elements())
        if not (removed or added or o["valid"] != n["valid"]):
            continue
        parts = os.path.relpath(u["edir"], os.path.expanduser("~/wt")).split(os.sep)
        if parts[0] == "..":
            parts = ["<main>"] + parts
        rows.append({"episode": u["episode"], "worktree": parts[0], "run": parts[-3], "transcripts_sha16": u["sha16"],
                     "old_valid": o["valid"], "new_valid": n["valid"],
                     "removed": [[i, r, d[:80]] for i, r, d in removed], "added": [[i, r, d[:80]] for i, r, d in added]})
        if a.detail:
            calls = list(new.iter_tool_calls(u["transcripts"][0]))
            for i, r, d in removed + added:
                if i is not None and i < len(calls):
                    c = calls[i][1].get("command") or ""
                    k = c.find("'/'")
                    print(f"  [{u['episode']} c{i} {r}]", repr(c[max(0, k - 80):k + 40]))
    summary = {"old_rev": a.old, "also_main": a.also_main, "new": "working tree", "python": sys.version.split()[0], "n_units": len(us),
               "n_tool_calls": n_calls, "verdict_change": dict(verdicts),
               "n_units_with_any_change": len(rows),
               "violations_removed": sum(len(r["removed"]) for r in rows),
               "violations_added": sum(len(r["added"]) for r in rows),
               "removed_by_rule": dict(collections.Counter(f"{x[1]}: {x[2]}" for r in rows for x in r["removed"])),
               "added_by_rule": dict(collections.Counter(f"{x[1]}: {x[2]}" for r in rows for x in r["added"]))}
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "changed_units": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))
    for r in rows:
        print(r["episode"], r["worktree"], r["run"], "old", r["old_valid"], "new", r["new_valid"],
              "removed", r["removed"], "added", r["added"])


if __name__ == "__main__":
    main()
