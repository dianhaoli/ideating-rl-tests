"""Independent replay of the 2026-10-02 smoke fixes (02514eb4) against the audit the smoke ran under (190d6a64).

    $PY runs/_harness/audit_replay_smokefix/replay_smokefix.py --scratch DIR

What it does (every input is opened read-only; nothing in a run dir or in ~/.claude/projects is written):
  1. Extracts common/ at OLD and NEW with `git archive` into DIR (pinned code, not the working tree).
  2. Enumerates every recorded LLM test-agent transcript:
       - the 24 test agents of smoke workflow wf_6c6e6341-0ee (meta.json description "agent:<task>:<instance>"),
         mapped to their episode through the privileged episode record (task + instance_id) and cross-checked
         against the sandbox path in the first user message; the ORIGINAL subagent JSONL is used, and its run-dir
         copy is compared byte for byte;
       - every api_transcript.jsonl in the main repo runs/ and every ~/wt/*/runs (Anthropic and OpenAI API agents),
         deduplicated by (episode, sha256); transcript.jsonl copies are compared with it byte for byte;
       - coverage checks: every episode id in the API ledgers, and every transcript in ~/.claude/projects whose first
         user message is SOME episode's agent_prompt.txt (exact or inside the workflow wrapper).
  3. Runs the OLD and the NEW audit CLI (`python -m common.transcript_audit`) on each with `--transcript <the test
     agent's transcript only>` (NEW also gets `--prompt-file <agent_prompt.txt>`), each in its own process.
  4. Discovery: for each smoke episode, NEW `locate()` with the prompt `sandbox.episode_prompt(rec)` gives finish
     (default root ~/.claude/projects), the NEW CLI without --transcript, and OLD `find_transcripts()`; every found
     file is classified by its workflow meta.json. Negative controls: each smoke episode's operator transcripts
     (prep/finish), and the stale run-dir transcript_N copies, passed explicitly to the NEW CLI must be R0.
Writes into DIR: full audit outputs (they contain commands: scratch only). Writes next to this script: aggregates only
(verdicts, rule names, call indices, episode ids, hashes, counts) -- no commands, submissions or answer material.
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/ec2-user/ideating-rl-tests"
HOME = os.path.expanduser("~")
STORE = os.path.join(HOME, ".claude", "projects")
SMOKE_WF = os.path.join(STORE, "-home-ec2-user-ideating-rl-tests", "64965452-3c8b-4462-8cab-82771922f494",
                        "subagents", "workflows", "wf_6c6e6341-0ee")
SMOKE_RUN = "20261002-smoke1_opus55"
SMOKE_TASKS = ("boolintermediates", "shifthunt", "latentdiff")
EPISODES = os.path.join(REPO, "runs", ".episodes")
PY = os.environ.get("PY") or "/opt/pytorch/bin/python"


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def extract(rev, dest):
    if not os.path.isdir(os.path.join(dest, "common")):
        os.makedirs(dest, exist_ok=True)
        a = subprocess.run(["git", "-C", REPO, "archive", rev, "common"], check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", dest], input=a, check=True)
    return dest


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def record(ep):
    p = os.path.join(EPISODES, ep + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def meta_desc(jsonl):
    m = jsonl[:-len(".jsonl")] + ".meta.json"
    if os.path.exists(m):
        try:
            return json.load(open(m)).get("description", "")
        except ValueError:
            return "?"
    return None


def smoke_agents(tn):
    """{episode: info} for the 24 smoke test agents, mapped by (task, instance_id) through the episode record."""
    by_inst = {}
    for task in SMOKE_TASKS:
        for edir in sorted(glob.glob(f"{HOME}/wt/{task}/runs/{task}/{SMOKE_RUN}/episodes/ep*")):
            ep = os.path.basename(edir)
            rec = record(ep)
            by_inst[(task, rec["instance_id"])] = (ep, edir, rec)
    out, problems = collections.OrderedDict(), []
    for m in sorted(glob.glob(SMOKE_WF + "/*.meta.json")):
        desc = json.load(open(m)).get("description", "")
        if not desc.startswith("agent:"):
            continue
        _, task, inst = desc.split(":", 2)
        ep, edir, rec = by_inst[(task, inst)]
        j = m[:-len(".meta.json")] + ".jsonl"
        first = tn.first_user_message(j) or ""
        ids = set(re.findall(r"rlsbx/(ep[0-9a-f]{10})", first))
        if ids != {ep}:
            problems.append(f"{ep}: first user message names {sorted(ids)}")
        out[ep] = {"episode": ep, "task": task, "instance": inst, "transcript": j, "edir": edir,
                   "prompt_file": os.path.join(edir, "agent_prompt.txt"), "sandbox": rec["sandbox"],
                   "source": "smoke_subagent", "rec": rec}
    if len(out) != len(by_inst):
        problems.append(f"{len(out)} agent transcripts for {len(by_inst)} smoke episodes")
    return out, problems


def run_dir_episodes():
    """All episode dirs (main runs/ and ~/wt/*/runs) holding any transcript file."""
    roots = [os.path.join(REPO, "runs")] + sorted(glob.glob(f"{HOME}/wt/*/runs"))
    dirs = []
    for r in roots:
        for e in glob.glob(r + "/**/episodes/ep*", recursive=True):
            if not os.path.isdir(e):
                continue
            fs = sorted(f for f in os.listdir(e) if f.endswith(".jsonl") and "transcript" in f)
            if fs:
                dirs.append((e, fs))
    return sorted(dirs)


def ledger_episodes():
    eps = collections.defaultdict(set)
    files = glob.glob(f"{HOME}/.rl_api/*.jsonl") + glob.glob(f"{REPO}/runs/api_budget/*.jsonl") + \
        glob.glob(f"{HOME}/wt/*/runs/api_budget/*.jsonl")
    for f in files:
        for line in open(f, errors="replace"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            e = d.get("episode")
            if isinstance(e, str) and re.fullmatch(r"ep[0-9a-f]{10}", e):
                eps[e].add("openai" if "openai" in os.path.basename(f) else "anthropic")
    return eps


def run_cli(code_dir, args, out):
    env = dict(os.environ, PYTHONPATH=code_dir, RL_EPISODES_DIR=EPISODES, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([PY, "-m", "common.transcript_audit"] + args + ["--out", out], cwd=code_dir, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"audit CLI failed ({code_dir} {args}): {r.stderr[-2000:]}")
    return json.load(open(out))


def vkey(v):
    return (v["i"], v["rule"], v["detail"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--old", default="190d6a64")
    ap.add_argument("--new", default="02514eb4")
    ap.add_argument("--out", default=os.path.join(HERE, "replay_smokefix.json"))
    a = ap.parse_args()
    S = os.path.abspath(a.scratch)
    old_dir, new_dir = extract(a.old, os.path.join(S, "old")), extract(a.new, os.path.join(S, "new"))
    outd = os.path.join(S, "audits_py" + sys.version.split()[0])
    os.makedirs(outd, exist_ok=True)
    sys.path.insert(0, new_dir)
    from common import transcript_audit as tn  # NEW, pinned copy
    from common import sandbox as sn
    assert os.path.dirname(tn.__file__) == os.path.join(new_dir, "common"), tn.__file__
    to = load_module(os.path.join(old_dir, "common", "transcript_audit.py"), "ta_old")
    problems = []

    # ---- 1. enumerate
    smoke, p = smoke_agents(tn)
    problems += p
    units = collections.OrderedDict()   # (episode, sha) -> unit
    copies = []                         # run-dir copy checks
    for ep, x in smoke.items():
        h = sha(x["transcript"])
        units[(ep, h)] = dict(x, sha=h)
    for edir, fs in run_dir_episodes():
        ep = os.path.basename(edir)
        rec = record(ep)
        epj = os.path.join(edir, "episode.json")
        epd = json.load(open(epj)) if os.path.exists(epj) else {}
        task = epd.get("task") or (rec or {}).get("task")
        sandbox = epd.get("sandbox") or (rec or {}).get("sandbox") or os.path.join(HOME, "rlsbx", ep)
        prompt_file = os.path.join(edir, "agent_prompt.txt")
        hs = {f: sha(os.path.join(edir, f)) for f in fs}
        if ep in smoke:
            ref = units[(ep, sha(smoke[ep]["transcript"]))]["sha"]
            for f, h in hs.items():
                if f == "transcript.jsonl":
                    kind = "identical_to_agent" if h == ref else "DIFFERS_from_agent"
                else:
                    kind = "stale_extra_copy"
                copies.append({"episode": ep, "dir": edir.replace(HOME, "~"), "file": f, "sha16": h[:16],
                               "kind": kind})
            continue
        if "api_transcript.jsonl" not in hs:
            problems.append(f"{edir}: transcript files {fs} but no api_transcript.jsonl and not a smoke episode")
            for f in fs:
                h = hs[f]
                units.setdefault((ep, h), {"episode": ep, "task": task, "transcript": os.path.join(edir, f),
                                           "prompt_file": prompt_file, "sandbox": sandbox, "sha": h,
                                           "source": "run_dir:" + f, "dirs": []})
                units[(ep, h)]["dirs"].append(edir)
            continue
        api = hs["api_transcript.jsonl"]
        for f, h in hs.items():
            if f != "api_transcript.jsonl":
                copies.append({"episode": ep, "dir": edir.replace(HOME, "~"), "file": f, "sha16": h[:16],
                               "kind": "identical_to_api_transcript" if h == api else "DIFFERS_from_api_transcript"})
        u = units.setdefault((ep, api), {"episode": ep, "task": task, "transcript": os.path.join(edir,
                                         "api_transcript.jsonl"), "prompt_file": prompt_file, "sandbox": sandbox,
                                         "sha": api, "source": "api_transcript", "dirs": []})
        u["dirs"].append(edir)
        if os.path.exists(prompt_file) and os.path.exists(u["prompt_file"]) and sha(prompt_file) != sha(u["prompt_file"]):
            problems.append(f"{ep}: agent_prompt.txt differs between checkouts")
    per_ep = collections.Counter(e for e, _ in units)
    multi = sorted(e for e, c in per_ep.items() if c > 1)
    if multi:
        problems.append(f"episodes with more than one distinct transcript: {multi}")

    # coverage: ledgers
    led = ledger_episodes()
    in_set = {e for e, _ in units}
    ledger_missing = sorted(e for e in led if e not in in_set)

    # coverage: every store transcript whose first user message is some episode's agent prompt
    prompt_index = collections.defaultdict(set)
    for r in [os.path.join(REPO, "runs")] + sorted(glob.glob(f"{HOME}/wt/*/runs")):
        for pf in glob.glob(r + "/**/episodes/ep*/agent_prompt.txt", recursive=True):
            t = open(pf, errors="replace").read()
            prompt_index[t[:-1] if t.endswith("\n") else t].add(os.path.basename(os.path.dirname(pf)))
    store_matches, store_n = [], 0
    for dp, _, fns in os.walk(STORE):
        for fn in fns:
            if not fn.endswith(".jsonl"):
                continue
            store_n += 1
            pth = os.path.join(dp, fn)
            msg = tn.first_user_message(pth)
            if msg is None:
                continue
            inner = tn.unwrap_workflow(msg)
            for cand in (msg, inner):
                if cand is None:
                    continue
                c = cand[:-1] if cand.endswith("\n") else cand
                if c in prompt_index:
                    store_matches.append({"transcript": pth, "episodes": sorted(prompt_index[c]),
                                          "meta": meta_desc(pth), "wrapped": cand is inner})
                    break
    store_unexplained = [m for m in store_matches if not any((e, sha(m["transcript"])) in units for e in m["episodes"])]

    # ---- 2. audits
    rows = []
    for (ep, h), u in units.items():
        tag = f"{ep}_{h[:12]}"
        base = ["--episode", ep, "--sandbox", u["sandbox"], "--transcript", u["transcript"]]
        o = run_cli(old_dir, base, os.path.join(outd, tag + ".old.json"))
        pf = ["--prompt-file", u["prompt_file"]] if os.path.exists(u["prompt_file"]) else []
        n = run_cli(new_dir, base + pf, os.path.join(outd, tag + ".new.json"))
        ov, nv = set(map(vkey, o["violations"])), set(map(vkey, n["violations"]))
        rec_h = {}
        if u["source"] == "smoke_subagent" and os.path.exists(os.path.join(u["edir"], "grade.json")):
            g = json.load(open(os.path.join(u["edir"], "grade.json"))).get("harness", {})
            rec_h = {"recorded_valid": g.get("valid"), "recorded_invalid_reasons": g.get("invalid_reasons"),
                     "recorded_n_audit_violations": g.get("n_audit_violations")}
        rows.append(dict(rec_h, **{
            "episode": ep, "task": u["task"], "source": u["source"], "transcript_sha16": h[:16],
            "n_dirs": len(u.get("dirs", [])) or 1, "n_tool_calls_old": o["n_tool_calls"],
            "n_tool_calls_new": n["n_tool_calls"], "old_valid": o["valid"], "new_valid": n["valid"],
            "prompt_file": bool(pf), "prompt_match": [c["prompt_match"] for c in n.get("prompt_check", [])],
            "old_rules": sorted(collections.Counter(v["rule"] for v in o["violations"]).items()),
            "new_rules": sorted(collections.Counter(v["rule"] for v in n["violations"]).items()),
            "removed": sorted({(i if i is not None else -1, r) for i, r, _ in ov - nv}),
            "added": sorted({(i if i is not None else -1, r) for i, r, _ in nv - ov}),
            "kept": sorted({(i if i is not None else -1, r) for i, r, _ in ov & nv}),
            "n_removed": len(ov - nv), "n_added": len(nv - ov), "n_kept": len(ov & nv)}))

    # ---- 3. discovery on the smoke run dirs
    disc = []
    for ep, x in smoke.items():
        rec = x["rec"]
        prompt = sn.episode_prompt(rec)           # what finish passes
        paths, r0, info = tn.locate(ep, prompt)   # finish's call with transcripts=None, search_root=None
        cli = run_cli(new_dir, ["--episode", ep, "--sandbox", rec["sandbox"]], os.path.join(outd, ep + ".disc.json"))
        oldf = to.find_transcripts(ep)
        cls = lambda ps: [("AGENT" if p == x["transcript"] else (meta_desc(p) or "no-meta").split(":")[0] or "?")
                          for p in ps]
        neg = []
        for m in sorted(glob.glob(SMOKE_WF + "/*.meta.json")):
            d = json.load(open(m)).get("description", "")
            if d.split(":")[0] in ("prep", "finish") and d.split(":")[-1] == x["instance"]:
                j = m[:-len(".meta.json")] + ".jsonl"
                r = run_cli(new_dir, ["--episode", ep, "--sandbox", rec["sandbox"], "--transcript", j,
                                      "--prompt-file", x["prompt_file"]],
                            os.path.join(outd, f"{ep}.neg_{d.split(':')[0]}.json"))
                neg.append({"operator": d.split(":")[0], "valid": r["valid"],
                            "r0": sorted({v["rule"] for v in r["violations"] if v["rule"].startswith("R0")})})
        for c in copies:
            if c["episode"] == ep and c["kind"] == "stale_extra_copy":
                pth = os.path.join(x["edir"], c["file"])
                r = run_cli(new_dir, ["--episode", ep, "--sandbox", rec["sandbox"], "--transcript", pth,
                                      "--prompt-file", x["prompt_file"]],
                            os.path.join(outd, f"{ep}.neg_{c['file']}.json"))
                # the stale copy was taken while its source was still being written: match it as a byte prefix
                b = open(pth, "rb").read()
                src = [m for m in glob.glob(SMOKE_WF + "/*.jsonl") if open(m, "rb").read().startswith(b)]
                neg.append({"operator": "run-dir " + c["file"] + " (byte prefix of: " +
                            ", ".join(meta_desc(m) or "?" for m in src) + ")" if src else
                            "run-dir " + c["file"] + " (not a prefix of any workflow transcript)",
                            "valid": r["valid"],
                            "r0": sorted({v["rule"] for v in r["violations"] if v["rule"].startswith("R0")})})
        disc.append({
            "episode": ep, "task": x["task"],
            "new_locate_found": len(paths), "new_locate_classes": cls(paths),
            "new_locate_exactly_agent": paths == [x["transcript"]] and not r0,
            "new_locate_r0": [k for k, _ in r0], "new_prompt_match": [c["prompt_match"] for c in info["prompt_check"]],
            "new_cli_found": cli["transcripts"] == [x["transcript"]] and cli["valid"] == any(
                rr["episode"] == ep and rr["new_valid"] for rr in rows),
            "new_cli_n": len(cli["transcripts"]),
            "old_found": len(oldf), "old_classes": cls(oldf),
            "negative_controls": neg,
            "negative_controls_all_r0": all(not n_["valid"] and "R0-prompt-mismatch" in n_["r0"] for n_ in neg)})

    # cross-check against the fixer's own replay (runs/_harness/smoke_fixes_2026-10-02/replay.json)
    fx_p = os.path.join(REPO, "runs", "_harness", "smoke_fixes_2026-10-02", "replay.json")
    fixer_disagree = []
    if os.path.exists(fx_p):
        fx = {r["episode"]: r for r in json.load(open(fx_p))["episodes"]}
        for r in rows:
            f = fx.get(r["episode"])
            mine = (r["old_valid"], r["new_valid"], [list(x) for x in r["removed"]], [list(x) for x in r["added"]])
            if f is None or mine != (f["old_valid"], f["new_valid"], [list(x) for x in f["removed"]],
                                     [list(x) for x in f["added"]]) or f["transcript_sha256_16"] != r["transcript_sha16"]:
                fixer_disagree.append(r["episode"])
        fixer_disagree += sorted(set(fx) - {r["episode"] for r in rows})
    change = collections.Counter(("clean" if r["old_valid"] else "flagged") + "->" +
                                 ("clean" if r["new_valid"] else "flagged") for r in rows)
    summary = {
        "old_rev": a.old, "new_rev": a.new, "python": sys.version.split()[0],
        "n_units": len(rows), "n_episodes": len({r["episode"] for r in rows}),
        "by_source": dict(collections.Counter(r["source"] for r in rows)),
        "verdict_change": dict(change),
        "n_violations_removed": sum(r["n_removed"] for r in rows),
        "n_violations_added": sum(r["n_added"] for r in rows),
        "episodes_with_added": [r["episode"] for r in rows if r["n_added"]],
        "episodes_with_removed": [r["episode"] for r in rows if r["n_removed"]],
        "tool_call_count_mismatch": [r["episode"] for r in rows if r["n_tool_calls_old"] != r["n_tool_calls_new"]],
        "prompt_match": dict(collections.Counter(",".join(r["prompt_match"]) or "none" for r in rows)),
        "copies": dict(collections.Counter(c["kind"] for c in copies)),
        "ledger_episodes": len(led), "ledger_episodes_missing_from_replay": ledger_missing,
        "store_transcripts_scanned": store_n, "store_prompt_matches": len(store_matches),
        "store_prompt_matches_not_in_replay": [(m["transcript"].replace(HOME, "~"), m["episodes"], m["meta"])
                                               for m in store_unexplained],
        "store_prompt_matches_non_agent_meta": [(m["transcript"].replace(HOME, "~"), m["meta"]) for m in store_matches
                                                if m["meta"] is not None and not m["meta"].startswith("agent:")],
        "discovery_new_exactly_agent": sum(d["new_locate_exactly_agent"] for d in disc),
        "discovery_new_cli_exactly_agent": sum(d["new_cli_found"] for d in disc),
        "discovery_old_found_counts": dict(collections.Counter(d["old_found"] for d in disc)),
        "discovery_negative_controls": sum(len(d["negative_controls"]) for d in disc),
        "discovery_negative_controls_all_r0": all(d["negative_controls_all_r0"] for d in disc),
        "recorded_grade_valid_smoke": sum(bool(r.get("recorded_valid")) for r in rows if r["source"] == "smoke_subagent"),
        "new_agent_only_valid_smoke": sum(r["new_valid"] for r in rows if r["source"] == "smoke_subagent"),
        "fixer_replay_disagreements": fixer_disagree if os.path.exists(fx_p) else "fixer replay.json not found",
        "problems": problems}
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "episodes": rows, "discovery": disc,
                   "copies": copies}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
