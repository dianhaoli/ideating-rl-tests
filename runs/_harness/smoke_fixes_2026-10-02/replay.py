"""Replay the transcript audit and the transcript discovery before and after the 2026-10-02 smoke fixes.

    python runs/_harness/smoke_fixes_2026-10-02/replay.py [--old REV]

OLD = common/transcript_audit.py at REV (default 190d6a64, the commit before the fixes); NEW = the working tree.
Read-only on every run dir (main repo runs/ and ~/wt/*/runs/) and on ~/.claude/projects. Inputs:
  - every episode dir with a recorded LLM transcript (api_transcript.jsonl or transcript.jsonl), deduplicated by
    episode id (the _demo episodes are tracked in git and appear in every checkout);
  - the 24 test-agent transcripts of the smoke workflow wf_6c6e6341-0ee (Claude Code subagents), whose run-dir copies
    were made by finish operators (some include an operator's transcript), so the originals are used.
Writes replay.json next to this script: verdicts, rule names, call indices and episode ids only (no commands,
submissions or answer material).
"""
import argparse
import collections
import glob
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
from common import transcript_audit as new  # noqa: E402

SMOKE_WF = os.path.expanduser("~/.claude/projects/-home-ec2-user-ideating-rl-tests/64965452-3c8b-4462-8cab-82771922f494"
                              "/subagents/workflows/wf_6c6e6341-0ee")


def load_rev(rev):
    src = subprocess.run(["git", "-C", REPO, "show", f"{rev}:common/transcript_audit.py"], check=True,
                         capture_output=True, text=True).stdout
    p = os.path.join(tempfile.mkdtemp(prefix="smokefix_replay_"), "ta_old.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("ta_old", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def episodes():
    """{episode: {"transcript", "sandbox", "prompt_file", "source"}}"""
    out = collections.OrderedDict()
    for m in sorted(glob.glob(SMOKE_WF + "/*.meta.json")):
        d = json.load(open(m))
        if not d.get("description", "").startswith("agent:"):
            continue
        j = m[:-len(".meta.json")] + ".jsonl"
        ep = re.search(r"rlsbx/(ep[0-9a-f]{10})", new.first_user_message(j) or "").group(1)
        task = d["description"].split(":")[1]
        edir = os.path.expanduser(f"~/wt/{task}/runs/{task}/20261002-smoke1_opus55/episodes/{ep}")
        out[ep] = {"transcript": j, "sandbox": os.path.expanduser(f"~/rlsbx/{ep}"), "task": task,
                   "prompt_file": os.path.join(edir, "agent_prompt.txt"), "source": "smoke_subagent"}
    roots = [os.path.join(REPO, "runs")] + sorted(glob.glob(os.path.expanduser("~/wt/*/runs")))
    for r in roots:
        for e in sorted(glob.glob(r + "/**/episodes/*", recursive=True)):
            ts = [f for f in ("api_transcript.jsonl", "transcript.jsonl") if os.path.exists(os.path.join(e, f))]
            if not ts or not os.path.exists(os.path.join(e, "episode.json")):
                continue
            ep = json.load(open(os.path.join(e, "episode.json")))
            eid = ep["episode"]
            if eid in out:
                continue
            out[eid] = {"transcript": os.path.join(e, ts[0]), "task": ep.get("task"),
                        "sandbox": ep.get("sandbox") or os.path.expanduser("~/rlsbx/" + eid),
                        "prompt_file": os.path.join(e, "agent_prompt.txt"), "source": "run_dir:" + ts[0]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="190d6a64")
    a = ap.parse_args()
    old = load_rev(a.old)
    rows = []
    key = lambda v: (v["i"], v["rule"], v["detail"])
    for eid, x in episodes().items():
        o = old.audit(eid, x["sandbox"], [x["transcript"]])
        n = new.audit(eid, x["sandbox"], [x["transcript"]])
        ov, nv = set(map(key, o["violations"])), set(map(key, n["violations"]))
        prompt = open(x["prompt_file"]).read() if os.path.exists(x["prompt_file"]) else None
        pm = new.prompt_match(new.first_user_message(x["transcript"]), prompt) if prompt else "no_prompt_file"
        rows.append({"episode": eid, "task": x["task"], "source": x["source"],
                     "transcript_sha256_16": hashlib.sha256(open(x["transcript"], "rb").read()).hexdigest()[:16],
                     "n_tool_calls": n["n_tool_calls"], "old_valid": o["valid"], "new_valid": n["valid"],
                     "removed": sorted({(i, r) for i, r, _ in ov - nv}), "added": sorted({(i, r) for i, r, _ in nv - ov}),
                     "kept": sorted({(i, r) for i, r, _ in ov & nv}), "prompt_match": pm or "mismatch"})
    # discovery: the 24 smoke episodes, searched over the whole ~/.claude/projects store
    disc = []
    eps = episodes()
    for r in rows:
        if r["source"] != "smoke_subagent":
            continue
        x = eps[r["episode"]]
        prompt = open(x["prompt_file"]).read()
        oldf = old.find_transcripts(r["episode"])
        paths, r0, info = new.locate(r["episode"], prompt)
        disc.append({"episode": r["episode"], "old_found": len(oldf),
                     "old_found_agent": x["transcript"] in oldf,
                     "new_found": len(paths), "new_is_agent": paths == [x["transcript"]],
                     "new_r0": [k for k, _ in r0], "prompt_match": [c["prompt_match"] for c in info["prompt_check"]]})
    change = collections.Counter(("clean" if r["old_valid"] else "flagged") + "->" + ("clean" if r["new_valid"]
                                                                                     else "flagged") for r in rows)
    summary = {"old_rev": a.old, "n_episodes": len(rows), "by_source": dict(collections.Counter(
        r["source"].split(":")[0] for r in rows)), "verdict_change": dict(change),
        "n_calls_removed": sum(len(r["removed"]) for r in rows), "n_calls_added": sum(len(r["added"]) for r in rows),
        "prompt_match": dict(collections.Counter(r["prompt_match"] for r in rows)),
        "discovery_all_exactly_agent": all(d["new_is_agent"] and not d["new_r0"] for d in disc),
        "discovery_old_found_counts": dict(collections.Counter(d["old_found"] for d in disc))}
    with open(os.path.join(HERE, "replay.json"), "w") as f:
        json.dump({"summary": summary, "episodes": rows, "discovery": disc}, f, indent=1)
    print(json.dumps(summary, indent=1))
    for r in rows:
        if r["removed"] or r["added"]:
            print(r["episode"], r["task"], r["source"][:14], "old", r["old_valid"], "new", r["new_valid"],
                  "removed", r["removed"], "added", r["added"])


if __name__ == "__main__":
    main()
