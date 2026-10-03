"""Replay the transcript audit before and after the E1 fix (ShiftHunt scaled run T1 false positives, 2026-10-03).

    $PY runs/_harness/audit_replay_t1scale/replay_t1scale.py [--old REV] [--detail]

OLD = common/transcript_audit.py at REV (default 5b965373, the last commit before the fix), loaded in memory (no temp
file); NEW = the working tree. Every input is read-only: run dirs (main repo runs/ and ~/wt/*/runs) and the Claude Code
store (~/.claude/projects).

Units audited (one per episode, the test agent's own transcript only):
  1. Claude Code subagent test agents: every *.jsonl in the store whose first user message IS some run-dir episode's
     agent_prompt.txt (common.transcript_audit.prompt_match: exact or the documented workflow wrapper). This finds the
     smoke agents (wf_6c6e6341-0ee), the T1 scaled-run agents (wf_96d109e5-27e) and any other, and never an
     operator transcript. The run-dir copy (transcript.jsonl) of each is compared byte for byte with the original.
  2. Every other run-dir episode with a recorded LLM transcript (api_transcript.jsonl or transcript.jsonl: API and
     OpenAI agents), deduplicated by episode id (the _demo episodes are tracked in git and appear in every checkout).

Each call violation removed by NEW is attributed to the E1 exemption that removed it by switching the four exemptions
off one at a time ((a) print separator, (b) sandbox join, (c) f-string range, (d) assignment target).

Writes replay_t1scale.json next to this script: aggregates, plus per-episode rows (episode id, verdicts, rule names, call
indices) for units whose episodes are already public in earlier replays. Units of a run in REDACT_RUNS (a scaled run
whose instances are not retired; D6) appear in aggregates only. No commands, submissions or answer material.
--detail prints a short context around each removed hit to the terminal (operator review only; never written).
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
from common import transcript_audit as new  # noqa: E402

STORE = os.path.expanduser("~/.claude/projects")
REDACT_RUNS = ("20261003-scale1_T1_opus55",)
EXEMPTIONS = {"a_print_separator": ("_print_sep_slash_offsets", set()), "b_sandbox_join": ("_sandbox_join_spans", []),
              "c_fstring_range": ("_fstring_dotdot_offsets", set()), "d_assign_target": ("_py_assign_name_offsets", set())}


def load_rev(rev):
    src = subprocess.run(["git", "-C", REPO, "show", f"{rev}:common/transcript_audit.py"], check=True,
                         capture_output=True, text=True).stdout
    m = types.ModuleType("ta_old")
    m.__file__ = os.path.join(REPO, "common", "transcript_audit.py")
    exec(compile(src, f"{rev}:common/transcript_audit.py", "exec"), m.__dict__)
    return m


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def episode_dirs():
    """{episode id: [episode dir, ...]} over the main repo and every worktree."""
    out = collections.defaultdict(list)
    roots = [os.path.join(REPO, "runs")] + sorted(glob.glob(os.path.expanduser("~/wt/*/runs")))
    for r in roots:
        for e in sorted(glob.glob(r + "/**/episodes/ep*", recursive=True)):
            if os.path.isdir(e) and "/_scratch" not in e:          # operator scratch dirs (e.g. pytest basetemp)
                out[os.path.basename(e)].append(e)
    return out


def units():
    eds = episode_dirs()
    out = collections.OrderedDict()
    # 1. Claude Code subagents whose first user message is an episode's agent prompt
    n_store = 0
    for j in sorted(glob.glob(STORE + "/**/*.jsonl", recursive=True)):
        n_store += 1
        try:
            msg = new.first_user_message(j)
        except OSError:
            continue
        if not msg:
            continue
        for eid in sorted(set(re.findall(r"rlsbx/(ep[0-9a-f]{10})", msg))):
            for ed in eds.get(eid, []):
                pf = os.path.join(ed, "agent_prompt.txt")
                if not os.path.exists(pf):
                    continue
                pm = new.prompt_match(msg, open(pf).read())
                if pm and eid not in out:
                    copy = os.path.join(ed, "transcript.jsonl")
                    out[eid] = {"transcript": j, "edir": ed, "source": "claude_code_subagent", "prompt_match": pm,
                                "run_dir_copy_identical": (os.path.exists(copy) and sha16(copy) == sha16(j))}
                elif pm and eid in out and out[eid]["transcript"] != j:
                    out[eid].setdefault("other_matches", []).append(j)
    # 2. recorded LLM transcripts in run dirs (API / OpenAI agents)
    for eid, dirs in eds.items():
        if eid in out:
            continue
        for ed in dirs:
            ts = [f for f in ("api_transcript.jsonl", "transcript.jsonl") if os.path.exists(os.path.join(ed, f))]
            if ts and os.path.exists(os.path.join(ed, "episode.json")):
                pf = os.path.join(ed, "agent_prompt.txt")
                t = os.path.join(ed, ts[0])
                pm = new.prompt_match(new.first_user_message(t), open(pf).read()) if os.path.exists(pf) else None
                out[eid] = {"transcript": t, "edir": ed, "source": "run_dir:" + ts[0], "prompt_match": pm or "mismatch"}
                break
    return out, n_store


def sandbox_of(eid, ed):
    try:
        sb = json.load(open(os.path.join(ed, "episode.json"))).get("sandbox")
    except (OSError, ValueError):
        sb = None
    return sb or os.path.expanduser("~/rlsbx/" + eid)


def attribute(calls, i, rule, detail, sandbox, eid):
    """Which E1 exemption removed violation (i, rule, detail): switch each off and see whether it comes back."""
    name, inp, cwd = calls[i]
    hits = []
    for label, (fn, empty) in EXEMPTIONS.items():
        orig = getattr(new, fn)
        setattr(new, fn, lambda *a, **k: type(empty)())
        try:
            back = (rule, detail[:300]) in {(r, d[:300]) for r, d in new.check_call(name, inp, cwd, sandbox, eid)}
        finally:
            setattr(new, fn, orig)
        if back:
            hits.append(label)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="5b965373")
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    old = load_rev(a.old)
    us, n_store = units()
    rows, attrib, redacted = [], collections.Counter(), collections.Counter()
    key = lambda v: (v["i"], v["rule"], v["detail"])
    for eid, x in us.items():
        sandbox = sandbox_of(eid, x["edir"])
        o = old.audit(eid, sandbox, [x["transcript"]])
        n = new.audit(eid, sandbox, [x["transcript"]])
        ov, nv = set(map(key, o["violations"])), set(map(key, n["violations"]))
        removed, added = sorted(ov - nv), sorted(nv - ov)
        calls = list(new.iter_tool_calls(x["transcript"])) if removed else []
        rem_attr = []
        for i, rule, detail in removed:
            lab = attribute(calls, i, rule, detail, new._norm_sandbox(sandbox), eid)
            attrib["+".join(lab) or "unattributed"] += 1
            rem_attr.append((i, rule, "+".join(lab) or "unattributed"))
            if a.detail:
                cmd = calls[i][1].get("command") or calls[i][1].get("content") or ""
                pat = {"a_print_separator": r"""['"]/['"]""", "b_sandbox_join": r"""\w\s*\+\s*['"]/""",
                       "c_fstring_range": r"\}\.\.\{", "d_assign_target": r"(?im)^\s*(?:top|ps|pgrep|pkill|lsof|htop|"
                       r"killall)\s*=|;\s*(?:top|ps)\s*="}
                for lb in lab or ["?"]:
                    for m in list(re.finditer(pat.get(lb, r"$^"), cmd))[:3]:
                        print(f"  [{eid} c{i} {rule} {lb}]", repr(cmd[max(0, m.start() - 70):m.end() + 50]))
        run = os.path.basename(os.path.dirname(os.path.dirname(x["edir"])))
        row = {"episode": eid, "task": os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(x["edir"])))),
               "run": run, "source": x["source"], "prompt_match": x["prompt_match"],
               "run_dir_copy_identical": x.get("run_dir_copy_identical"), "other_matches": len(x.get("other_matches", [])),
               "transcript_sha256_16": sha16(x["transcript"]), "n_tool_calls": n["n_tool_calls"],
               "old_valid": o["valid"], "new_valid": n["valid"],
               "removed": [list(t) for t in rem_attr], "added": sorted({(i, r) for i, r, _ in added}),
               "kept": sorted({(i, r) for i, r, _ in ov & nv})}
        rows.append(row)
    change = lambda rs: dict(collections.Counter(("clean" if r["old_valid"] else "flagged") + "->"
                                                 + ("clean" if r["new_valid"] else "flagged") for r in rs))
    by_group = collections.defaultdict(list)
    for r in rows:
        g = "T1 scaled run (redacted)" if r["run"] in REDACT_RUNS else (
            "claude_code_subagent" if r["source"] == "claude_code_subagent" else "run_dir_api")
        by_group[g].append(r)
    summary = {
        "old_rev": a.old, "new": "working tree", "python": sys.version.split()[0], "store_jsonl_scanned": n_store,
        "n_units": len(rows),
        "groups": {g: {"n": len(rs), "verdict_change": change(rs), "n_calls": sum(r["n_tool_calls"] for r in rs),
                       "violations_removed": sum(len(r["removed"]) for r in rs),
                       "violations_added": sum(len(r["added"]) for r in rs),
                       "prompt_match": dict(collections.Counter(r["prompt_match"] for r in rs)),
                       "run_dir_copy_identical": dict(collections.Counter(str(r["run_dir_copy_identical"])
                                                                          for r in rs)),
                       "units_with_other_matching_transcripts": sum(1 for r in rs if r["other_matches"])}
                   for g, rs in sorted(by_group.items())},
        "verdict_change": change(rows),
        "n_calls_removed": sum(len(r["removed"]) for r in rows), "n_calls_added": sum(len(r["added"]) for r in rows),
        "removed_by_exemption": dict(attrib),
        "redacted_runs": list(REDACT_RUNS),
        "redacted_removed_by_rule_and_exemption": dict(collections.Counter(
            f"{t[1]}:{t[2]}" for r in rows if r["run"] in REDACT_RUNS for t in r["removed"])),
        "redacted_episodes_with_removals": sum(1 for r in rows if r["run"] in REDACT_RUNS and r["removed"]),
    }
    public = [r for r in rows if r["run"] not in REDACT_RUNS]
    with open(os.path.join(HERE, "replay_t1scale.json"), "w") as f:
        json.dump({"summary": summary, "episodes": public}, f, indent=1)
    print(json.dumps(summary, indent=1))
    for r in public:
        if r["removed"] or r["added"]:
            print(r["episode"], r["task"], r["source"][:20], "old", r["old_valid"], "new", r["new_valid"],
                  "removed", r["removed"], "added", r["added"])


if __name__ == "__main__":
    main()
