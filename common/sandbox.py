"""Sandbox launcher: prepares episodes, finishes (grades + audits) them, runs scripted solvers, summarizes.

    python -m common.sandbox setup        # once per machine: analysis venv for ./py (numpy, scipy, sklearn; NO torch)
    python -m common.sandbox prepare --task T --instance-dir D --profile full|blackbox \
           --run-dir runs/T/<YYYYmmdd-HHMMSS>_<label> [--solver-label opus] [--json]
    python -m common.sandbox finish --episode E [--transcript PATH ...] [--agent-model MODEL]
    python -m common.sandbox run-scripted --task T --solver tasks/T/reference_solver.py \
           --instances D1 D2 ... --profile full --run-dir R [--repeats 5] [--solver-label reference]
    python -m common.sandbox summarize --run-dir R

An episode sandbox ~/rlsbx/<E>/ holds ONLY: TASK.md, tool (client), py (-> analysis venv python),
out/ (arrays written by tools, large responses), scratch/ (agent's own files) and .episode.
Everything privileged (instance dir, answer key, canary, leak strings, counters, tool log) lives in
the episode record <main repo>/runs/.episodes/<E>.json, which only the broker and this module read.
See docs/HARNESS_API.md for the contract.
"""
import argparse
import fcntl
import glob
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import broker, leakscan, paths, transcript_audit  # noqa: E402
from common.toolserver import BUILTIN_DOCS  # noqa: E402

HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT_SRC = os.path.join(HARNESS_ROOT, "common", "agent_client", "tool")
PY = sys.executable

AGENT_PROMPT = """You are being given a self-contained research task to attempt on your own.

Your working directory is {sandbox}/ . Only read, write and run things inside that directory. Do not read, list or search any other location on this machine (no other directories, not your home directory, not /tmp, no ".." paths). Do not create symlinks, read environment variables, or use encoded commands (base64, eval, exec).

Start every shell command with:  cd {sandbox} &&
Then read TASK.md there (cat TASK.md). It explains the task, the tools and the exact answer format.

- Use ./tool to call the task's tools (./tool help lists them; ./tool budget shows what you have left).
- Use ./py (a Python with numpy, scipy and scikit-learn) for analysis, for example of the .npy files that tools save under out/. Keep your own files in scratch/.
- Do not use web search or web fetch, and do not install or download anything.
- Do not spawn sub-agents or helper tasks; work alone.
- The first tool call may print "waiting for compute..." while the system is loaded. If a command times out, simply run it again.
- Finish by calling ./tool submit with your answer in the format TASK.md describes. Only one submission is accepted. A malformed submission is rejected with an error message, and you can fix it and resubmit.

After you have submitted, reply with a short summary of what you found and how."""


# ---------------------------------------------------------------------------------------------- helpers
def _git_sha(root):
    try:
        sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True).stdout.strip())
        return sha, dirty
    except Exception:
        return None, None


def _versions():
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for pkg in ("numpy", "torch", "transformers", "peft"):
        try:
            out[pkg] = metadata.version(pkg)
        except Exception:
            pass
    try:
        out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        out["gpu"] = None
    return out


def _load_json(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _update_config(run_dir, task, task_root, **add):
    """Create/merge <run-dir>/config.json (git sha, versions, and sets of profiles/labels/instances)."""
    os.makedirs(run_dir, exist_ok=True)
    p = os.path.join(run_dir, "config.json")
    with open(os.path.join(run_dir, ".config.lock"), "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        cfg = _load_json(p, None)
        if cfg is None:
            sha, dirty = _git_sha(task_root)
            cfg = {"task": task, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "git_sha": sha, "git_dirty": dirty, "task_root": task_root, "versions": _versions(),
                   "profiles": [], "solver_labels": [], "agent_models": [], "instances": {}}
        for k in ("profiles", "solver_labels", "agent_models"):
            v = add.get(k)
            if v and v not in cfg[k]:
                cfg[k].append(v)
        if add.get("instance_id"):
            cfg["instances"][add["instance_id"]] = {"seed": add.get("seed"), "tier": add.get("tier"),
                                                   "dial": add.get("dial"), "instance_dir": add.get("instance_dir")}
        broker.write_json_atomic(p, cfg)


def _describe(task_root, task, profile):
    r = subprocess.run([PY, "-m", "common.toolserver", "describe", "--task-root", task_root, "--task", task,
                        "--profile", profile], cwd=task_root, env=broker.server_env(task_root),
                       capture_output=True, text=True, timeout=600)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    if r.returncode != 0 or not lines:
        raise RuntimeError(f"could not load tasks/{task}/tools.py:\n{r.stderr[-3000:]}")
    return json.loads(lines[-1])


def render_tool_docs(tool_docs):
    lines = ["How to call tools (from your working directory):",
             "    ./tool <name> '<json object of args>'",
             "    ./tool <name> key=value key2='[1,2]'      (values are parsed as JSON when possible)",
             'Every response is JSON: {"ok": true, "result": ...} or {"ok": false, "error": "..."}. '
             "Responses over 30 KB are saved to out/resp_<n>.json and only a preview is printed.",
             "", "Tools:"]
    lines += [f"- `{t['name']}`: {t['doc']}" for t in tool_docs]
    lines += ["", "Built-in:"]
    lines += [f"- `{k}`: {v}" for k, v in BUILTIN_DOCS.items()]
    return "\n".join(lines)


def render_caps(caps):
    lines = []
    for k in broker.counter_kinds(caps):
        note = " (each task tool call uses one; help and budget are free)" if k == "tool_calls" else " units"
        lines.append(f"- {k}: {caps[k]}{note}")
    lines.append(f"- time: {caps['wall_clock_s'] / 60:.0f} minutes from your first tool call "
                 f"(time spent waiting for compute is not counted); each tool call may run at most "
                 f"{caps['call_timeout_s']} s")
    return "\n".join(lines)


def render_task_md(template, public, tool_docs_md, caps_md):
    def sub_public(m):
        key = m.group(1)
        if key not in public:
            raise KeyError(f"agent_prompt.md uses {{public.{key}}} but public.json has no '{key}'")
        v = public[key]
        return v if isinstance(v, str) else json.dumps(v)

    out = re.sub(r"\{public\.([A-Za-z0-9_]+)\}", sub_public, template)
    if "{tool_docs}" in out:
        out = out.replace("{tool_docs}", tool_docs_md)
    else:
        out += "\n\n## Tools\n\n" + tool_docs_md + "\n"
    if "{caps}" in out:
        out = out.replace("{caps}", caps_md)
    else:
        out += "\n\n## Budget\n\n" + caps_md + "\n"
    return out


def codename_variants(task):
    base = task.strip("_-").lower()
    return {v for v in (base, base.replace("_", ""), base.replace("-", ""), base.replace("_", " ")) if len(v) >= 3}


def private_strings_for(task, task_root, instance_dir, inst, public):
    """Strings no agent-visible text may contain, ever (leak scan `private_strings`): where the answer
    key lives (instance dir, repo paths, episode registry), the instance id, and the task codename.
    The instance id is skipped if the task deliberately publishes it in public.json; codename
    variants shorter than 6 characters are skipped (too generic, e.g. "demo")."""
    out = [os.path.abspath(instance_dir), os.path.abspath(task_root), HARNESS_ROOT, paths.episodes_dir(),
           paths.MAIN_REPO]
    iid = str(inst.get("instance_id") or "")
    if len(leakscan.norm(iid)) >= 6 and leakscan.norm(iid) not in leakscan.norm(json.dumps(public)):
        out.append(iid)
    out += sorted(v for v in codename_variants(task) if len(v) >= 6)
    return list(dict.fromkeys(out))


def check_task_md(text, task, canary, leak_strings, private_strings=()):
    problems = []
    low = text.lower()
    for v in codename_variants(task):
        if v in low:
            problems.append(f"task codename '{v}' appears in TASK.md")
    hit = leakscan.scan_text(text, canary, leak_strings, private_strings)
    if hit["canary"]:
        problems.append("canary appears in TASK.md")
    if hit["leak_strings"]:
        problems.append(f"leak_strings {hit['leak_strings']} appear in TASK.md")
    if hit["private"]:
        problems.append(f"private strings (instance path/id, repo path) {hit['private']} appear in TASK.md")
    return problems


# ---------------------------------------------------------------------------------------------- setup
def setup():
    venv = paths.venv_dir()
    if not os.path.exists(os.path.join(venv, "bin", "python")):
        subprocess.run(["/usr/bin/python3", "-m", "venv", venv], check=True)
    vpy = os.path.join(venv, "bin", "python")
    subprocess.run([vpy, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    subprocess.run([vpy, "-m", "pip", "install", "-q", "numpy", "scipy", "scikit-learn"], check=True)
    r = subprocess.run([vpy, "-c", "import torch"], capture_output=True)
    if r.returncode == 0:
        raise SystemExit(f"{venv} can import torch; the analysis venv must not have torch (agents could load models)")
    print(f"analysis venv ready: {vpy}")


# ---------------------------------------------------------------------------------------------- prepare
SNAPSHOT_COPY_MAX_BYTES = 300 * 1024 * 1024


def _snapshot_instance(src):
    """Copy (or hard-link, if large) an instance dir to <episodes dir>/snap/<random>/<basename>; return the copy."""
    if not os.path.isfile(os.path.join(src, "instance.json")):
        return src  # let the caller raise its usual error
    total = 0
    for root, _dirs, files in os.walk(src):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    dst = os.path.join(paths.episodes_dir(), "snap", secrets.token_hex(6), os.path.basename(src.rstrip("/")))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if total <= SNAPSHOT_COPY_MAX_BYTES:
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copytree(src, dst, symlinks=True, copy_function=os.link)
    return dst


def prepare(task, instance_dir, profile, run_dir, solver_label=None, tasks_root=None, agent_model=None,
            start_broker=True):
    task_root = os.path.abspath(tasks_root or HARNESS_ROOT)
    instance_dir = os.path.abspath(instance_dir)
    run_dir = os.path.abspath(run_dir)
    tdir = os.path.join(task_root, "tasks", task)
    # Snapshot the instance into a private per-episode copy (2026-10-01): a builder regenerated an instance with the
    # same id while an episode was running on it, so the tool server and the grader saw different files. Everything
    # below (tool server, grader) uses the snapshot. Small instances are copied; large ones are hard-linked, which
    # still protects against the usual atomic-replace rewrites.
    instance_src = instance_dir
    instance_dir = _snapshot_instance(instance_src)
    inst = _load_json(os.path.join(instance_dir, "instance.json"))
    if inst is None:
        raise FileNotFoundError(f"{instance_dir}/instance.json missing or unreadable")
    public = _load_json(os.path.join(instance_dir, "public.json"), {})
    canary = inst.get("canary")
    if not canary or "RLCANARY" not in canary:
        raise ValueError("instance.json must have a canary of the form RLCANARY-<task>-<hex>")
    leak_strings = list(inst.get("leak_strings") or [])
    bad = leakscan.check_leak_strings(leak_strings)
    if bad:
        raise ValueError("bad leak_strings: " + "; ".join(bad))
    caps = broker.merged_caps(inst.get("caps"))
    desc = _describe(task_root, task, profile)
    with open(os.path.join(tdir, "agent_prompt.md")) as f:
        template = f.read()
    task_md = render_task_md(template, public, render_tool_docs(desc["tool_docs"]), render_caps(caps))
    private = private_strings_for(task, task_root, instance_dir, inst, public)
    problems = check_task_md(task_md, task, canary, leak_strings, private)
    if problems:
        raise ValueError("TASK.md failed checks: " + "; ".join(problems))
    vpy = os.path.join(paths.venv_dir(), "bin", "python")
    if not os.path.exists(vpy):
        raise FileNotFoundError(f"analysis venv missing ({vpy}); run: python -m common.sandbox setup")

    eid = "ep" + secrets.token_hex(5)
    sbx = os.path.join(paths.sandbox_root(), eid)
    os.makedirs(os.path.join(sbx, "out"))
    os.makedirs(os.path.join(sbx, "scratch"))
    with open(os.path.join(sbx, "TASK.md"), "w") as f:
        f.write(task_md)
    shutil.copyfile(CLIENT_SRC, os.path.join(sbx, "tool"))
    os.chmod(os.path.join(sbx, "tool"), 0o755)
    # ./py is a 2-line exec wrapper, not a symlink: Python 3.9 finds its venv (pyvenv.cfg) relative to the
    # path it was invoked by, so a symlink in the sandbox would run the bare system python without numpy.
    with open(os.path.join(sbx, "py"), "w") as f:
        f.write(f'#!/bin/sh\nexec "{vpy}" "$@"\n')
    os.chmod(os.path.join(sbx, "py"), 0o755)
    with open(os.path.join(sbx, ".episode"), "w") as f:
        f.write(eid + "\n")

    sha, dirty = _git_sha(task_root)
    now = time.time()
    rec = {"episode": eid, "task": task, "task_root": task_root, "instance_dir": instance_dir,
           "instance_dir_source": instance_src,
           "instance_id": inst.get("instance_id"), "tier": inst.get("tier"), "profile": profile,
           "sandbox": sbx, "run_dir": run_dir, "solver_label": solver_label, "agent_model": agent_model,
           "caps": caps, "counters": {k: 0 for k in broker.counter_kinds(caps)},
           "canary": canary, "leak_strings": leak_strings, "private_strings": private,
           "model_output_fields": desc["model_output_fields"],
           "tools": [t["name"] for t in desc["tool_docs"]], "tool_docs": desc["tool_docs"],
           "builtin_docs": BUILTIN_DOCS, "gpu_gb": desc["gpu_gb"], "grader_gpu_gb": desc.get("grader_gpu_gb", 0),
           "status": "open", "created_at": now, "started_at": None, "wait_s": 0.0, "submission": None,
           "leak_detected": False, "leak_events": [], "behavioral_exposure": 0, "git_sha": sha, "git_dirty": dirty}
    broker.write_json_atomic(broker.record_path(eid), rec)

    edir = os.path.join(run_dir, "episodes", eid)
    os.makedirs(edir, exist_ok=True)
    prompt = AGENT_PROMPT.format(sandbox=sbx)
    with open(os.path.join(edir, "agent_prompt.txt"), "w") as f:
        f.write(prompt + "\n")
    shutil.copyfile(os.path.join(sbx, "TASK.md"), os.path.join(edir, "TASK.md"))
    broker.write_json_atomic(os.path.join(edir, "episode.json"), _public_record(rec))
    _update_config(run_dir, task, task_root, profiles=profile, solver_labels=solver_label, agent_models=agent_model,
                   instance_id=inst.get("instance_id"), seed=inst.get("seed"), tier=inst.get("tier"),
                   dial=inst.get("dial"), instance_dir=instance_dir)
    if start_broker:
        broker.start(quiet=True)
    return {"episode": eid, "sandbox": sbx, "prompt": prompt, "episode_dir": edir}


def _public_record(rec):
    """The run-dir copy of the episode record: no canary, no leak strings, no submission internals."""
    keep = ("episode", "task", "instance_id", "tier", "profile", "solver_label", "agent_model", "sandbox",
            "instance_dir", "caps", "counters", "tools", "gpu_gb", "status", "created_at", "started_at",
            "submitted_at", "closed_at", "wait_s", "leak_detected", "behavioral_exposure", "git_sha", "git_dirty")
    out = {k: rec.get(k) for k in keep}
    out["n_leak_events"] = len(rec.get("leak_events") or [])
    out["cross_episode_access"] = bool(rec.get("cross_episode_access"))
    return out


# ---------------------------------------------------------------------------------------------- finish
GRADER_TIMEOUT_S = 3600


def _check_grade(g):
    """Return an error string if the grader output is not {"score": finite float in [0,1], "pass": bool, ...}."""
    if not isinstance(g, dict):
        return f"grader output is not a JSON object ({type(g).__name__})"
    sc = g.get("score")
    if isinstance(sc, bool) or not isinstance(sc, (int, float)) or not math.isfinite(sc) or not 0 <= sc <= 1:
        return f"grader score must be a number in [0, 1], got {sc!r}"
    if not isinstance(g.get("pass"), bool):
        return f"grader pass must be true/false, got {g.get('pass')!r}"
    if "details" in g and not isinstance(g["details"], dict):
        return "grader details must be a JSON object"
    return None


def _run_grader(rec, sub_path, grade_path):
    """Run tasks/T/grader.py in a separate process. Any failure (crash, non-zero exit, timeout, missing or
    malformed output) gives score 0, pass False and a non-empty grader_error; finish() then marks the
    episode INVALID (grader_error) instead of crashing the harness or trusting a broken grade."""
    grader = os.path.join(rec["task_root"], "tasks", rec["task"], "grader.py")
    cmd = [PY, grader, "--instance-dir", rec["instance_dir"], "--submission", sub_path, "--out", grade_path]
    gb = float(rec.get("grader_gpu_gb") or 0)
    if gb > 0:   # graders that run the model on held-out data queue for the GPU like everything else
        cmd = [PY, "-m", "common.gpuq", "run", "--gb", str(gb), "--label", f"grader:{rec['task']}", "--"] + cmd
    timeout = float(os.environ.get("RL_GRADER_TIMEOUT_S", GRADER_TIMEOUT_S))
    try:
        os.unlink(grade_path)            # never read a stale grade from an earlier finish
    except OSError:
        pass
    fail = {"score": 0.0, "pass": False, "details": {}}
    proc = subprocess.Popen(cmd, cwd=rec["task_root"], env=broker.server_env(rec["task_root"]),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        _, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except OSError:
            pass
        proc.communicate()
        return dict(fail, grader_error=f"grader timed out after {timeout:.0f} s")
    if proc.returncode != 0:
        return dict(fail, grader_error=f"grader exited with code {proc.returncode}: {(err or '')[-2000:]}")
    g = _load_json(grade_path)
    if g is None:
        return dict(fail, grader_error="grader wrote no (valid JSON) output file")
    bad = _check_grade(g)
    if bad:
        return dict(fail, grader_error=bad)
    g = dict(g, score=float(g["score"]))
    g.setdefault("details", {})
    return g


def _scan_sandbox(rec):
    """Scan the sandbox at finish (file contents AND file names):
    - canary or a private string (instance path/id, repo path, codename) anywhere => leak;
    - leak_strings in harness-written files (TASK.md as written, tool, py, .episode) or in the names of
      tool-written arrays (out/*.npy) => leak;
    - leak_strings in files the agent wrote are expected when it found the answer, so they are only
      counted (agent_files_with_leak_strings), never flagged. A TASK.md the agent edited counts as its file;
    - a symlink that points outside the sandbox => symlinks_outside (finish marks the episode INVALID:
      reading through it would leave the sandbox without naming an outside path)."""
    sbx = os.path.realpath(rec["sandbox"])
    canary, leaks, private = rec.get("canary"), rec.get("leak_strings"), rec.get("private_strings", [])
    pristine = {"tool": CLIENT_SRC}
    edir = os.path.join(rec.get("run_dir") or "", "episodes", rec.get("episode") or "")
    if rec.get("run_dir") and os.path.exists(os.path.join(edir, "TASK.md")):
        pristine["TASK.md"] = os.path.join(edir, "TASK.md")

    def same_as_written(rel, p):
        if rel in (".episode", "py"):
            return True
        src = pristine.get(rel)
        if src is None:
            return rel == "TASK.md"         # no copy to compare against: treat as harness-written
        try:
            with open(src, "rb") as a, open(p, "rb") as b:
                return a.read() == b.read()
        except OSError:
            return False

    res = {"leak": False, "reasons": [], "agent_files_with_leak_strings": 0, "n_files": 0, "symlinks_outside": []}
    for dp, dns, fns in os.walk(sbx):
        for fn in sorted(fns + [d for d in dns if os.path.islink(os.path.join(dp, d))]):
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, sbx)
            name_hit = leakscan.scan_text(rel, canary, leaks, private)
            if name_hit["canary"] or name_hit["private"]:
                res["leak"] = True
                res["reasons"].append("canary or private string in a file name")
            if name_hit["leak_strings"]:
                if rel.startswith("out" + os.sep) and rel.endswith(".npy"):
                    res["leak"] = True
                    res["reasons"].append("leak_string in the name of a tool-written array")
                else:
                    res["agent_files_with_leak_strings"] += 1
            if os.path.islink(p):
                tgt = os.path.realpath(p)
                if not (tgt == sbx or tgt.startswith(sbx + os.sep)):
                    res["symlinks_outside"].append(rel)
                continue
            res["n_files"] += 1
            try:
                with open(p, "rb") as f:
                    text = f.read(50_000_000).decode("utf-8", errors="replace")
            except OSError:
                continue
            hit = leakscan.scan_text(text, canary, leaks, private)
            if hit["canary"]:
                res["leak"] = True
                res["reasons"].append(f"canary in sandbox file {rel}")
            if hit["private"]:
                res["leak"] = True
                res["reasons"].append(f"private string (instance path/id, repo path or codename) in sandbox file {rel}")
            if hit["leak_strings"]:
                if rel in ("TASK.md", "tool", "py", ".episode") and same_as_written(rel, p):
                    res["leak"] = True
                    res["reasons"].append(f"leak_string in harness file {rel}")
                else:
                    res["agent_files_with_leak_strings"] += 1
    return res


def _scan_tool_log(rec, log_path):
    """Re-scan every served response in the tool log (a safety net behind the live scan)."""
    res = {"leak": False, "reasons": []}
    if not os.path.exists(log_path):
        return res
    with open(log_path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = e.get("response")
            if resp is None:
                continue
            try:
                obj = json.loads(resp)
                # If the logged args were truncated, the agent-text exemption cannot be recomputed; the live
                # scan (which saw the full args) is authoritative for leak strings, so re-check only the
                # canary and the private strings here.
                ls = [] if e.get("args_truncated") else rec.get("leak_strings")
                s = leakscan.scan(obj, rec.get("canary"), ls, rec.get("model_output_fields"),
                                  agent_text=e.get("args"), private_strings=rec.get("private_strings", []))
                bad = s["leak"]
            except json.JSONDecodeError:      # truncated entry: canary / private check only
                h = leakscan.scan_text(resp, rec.get("canary"), (), rec.get("private_strings", []))
                bad = h["canary"] or bool(h["private"])
            if bad:
                res["leak"] = True
                res["reasons"].append(f"tool log entry n={e.get('n')} tool={e.get('tool')}")
    return res


def finish(episode, transcripts=None, agent_model=None, search_root=None):
    rp = broker.record_path(episode)
    rec = _load_json(rp)
    if rec is None:
        raise FileNotFoundError(f"no episode record {rp}")
    if broker.ping():
        try:
            broker.admin("close", episode=episode)
        except Exception as e:          # never let a busy/old broker crash grading; the record is closed below
            print(f"warning: broker close failed ({type(e).__name__}); closing the record directly", file=sys.stderr)
    rec = _load_json(rp)
    if rec.get("status") == "open":
        rec["status"] = "closed"
        rec["closed_at"] = time.time()
    if agent_model:
        rec["agent_model"] = agent_model
    broker.write_json_atomic(rp, rec)

    edir = os.path.join(rec["run_dir"], "episodes", episode)
    os.makedirs(edir, exist_ok=True)
    log_src = broker.tool_log_path(episode)
    log_dst = os.path.join(edir, "tool_log.jsonl")
    if os.path.exists(log_src):
        shutil.copyfile(log_src, log_dst)
    else:
        open(log_dst, "w").close()
    sub_path = os.path.join(edir, "submission.json")
    with open(sub_path, "w") as f:
        json.dump(rec.get("submission"), f, indent=1)

    grade = _run_grader(rec, sub_path, os.path.join(edir, ".grader_out.json"))
    sbx_scan = _scan_sandbox(rec)
    log_scan = _scan_tool_log(rec, log_dst)

    is_llm = bool(agent_model or rec.get("agent_model") or transcripts)
    if is_llm:
        ts = list(transcripts or []) or transcript_audit.find_transcripts(episode, search_root)
        audit = transcript_audit.audit(episode, rec["sandbox"], ts)
        for i, t in enumerate(ts):
            shutil.copyfile(t, os.path.join(edir, "transcript.jsonl" if i == 0 else f"transcript_{i}.jsonl"))
    else:
        audit = {"episode": episode, "valid": True, "skipped": "scripted solver (no LLM transcript)",
                 "violations": [], "n_tool_calls": 0, "commands": [], "transcripts": []}
    broker.write_json_atomic(os.path.join(edir, "audit.json"), audit)

    leak = bool(rec.get("leak_detected") or sbx_scan["leak"] or log_scan["leak"])
    invalid = []
    if leak:
        invalid.append("leak_detected")
    if not audit["valid"]:
        invalid.append("transcript_audit")
    if grade.get("grader_error") is not None:
        invalid.append("grader_error")
    if rec.get("cross_episode_access"):
        invalid.append("cross_episode_access")
    if sbx_scan["symlinks_outside"]:
        invalid.append("sandbox_symlink_outside")
    end = rec.get("submitted_at") or rec.get("closed_at") or time.time()
    grade["harness"] = {
        "episode": episode, "task": rec["task"], "instance_id": rec.get("instance_id"), "tier": rec.get("tier"),
        "profile": rec["profile"], "solver_label": rec.get("solver_label"), "agent_model": rec.get("agent_model"),
        "submitted": rec.get("submission") is not None, "status": rec["status"], "counters": rec["counters"],
        "caps": rec["caps"], "behavioral_exposure": rec.get("behavioral_exposure", 0),
        "leak_detected": leak, "leak_reasons": [r for e in rec.get("leak_events", []) for r in e["reasons"]]
        + sbx_scan["reasons"] + log_scan["reasons"],
        "agent_files_with_leak_strings": sbx_scan["agent_files_with_leak_strings"],
        "symlinks_outside": sbx_scan["symlinks_outside"],
        "cross_episode_access": bool(rec.get("cross_episode_access")),
        "audit_valid": audit["valid"], "audit_skipped": audit.get("skipped"),
        "n_audit_violations": len(audit["violations"]),
        "valid": not invalid, "invalid_reasons": invalid,
        "elapsed_s": round(end - rec["started_at"], 1) if rec.get("started_at") else None,
        "wait_s": round(rec.get("wait_s", 0.0), 1),
    }
    broker.write_json_atomic(os.path.join(edir, "grade.json"), grade)
    try:
        os.unlink(os.path.join(edir, ".grader_out.json"))
    except OSError:
        pass
    rec["finished_at"] = time.time()
    rec["leak_detected"] = leak
    broker.write_json_atomic(rp, rec)
    pub = _public_record(rec)
    pub["finished_at"] = rec["finished_at"]
    broker.write_json_atomic(os.path.join(edir, "episode.json"), pub)
    return grade


# ---------------------------------------------------------------------------------------------- run-scripted
def run_scripted(task, solver, instances, profile, run_dir, repeats=1, solver_label=None, tasks_root=None,
                 extra_args=()):
    solver = os.path.abspath(solver)
    label = solver_label or os.path.splitext(os.path.basename(solver))[0]
    results = []
    for inst in instances:
        for r in range(repeats):
            ep = prepare(task, inst, profile, run_dir, solver_label=label, tasks_root=tasks_root)
            eid = ep["episode"]
            rec = _load_json(broker.record_path(eid))
            env = dict(os.environ, RL_EPISODE=eid,
                       PYTHONPATH=os.pathsep.join(dict.fromkeys([rec["task_root"], HARNESS_ROOT])))
            # The solver gets ONLY the episode id (argv + RL_EPISODE): no instance path, no record path.
            t0 = time.time()
            with open(os.path.join(ep["episode_dir"], "solver.log"), "w") as log:
                try:
                    rc = subprocess.run([PY, solver, "--episode", eid, *extra_args], cwd=ep["sandbox"], env=env,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        timeout=rec["caps"]["wall_clock_s"] + 1800).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            g = finish(eid)
            h = g["harness"]
            row = {"episode": eid, "instance_id": h["instance_id"], "repeat": r, "solver_rc": rc,
                   "pass": bool(g.get("pass")), "score": g.get("score"), "valid": h["valid"],
                   "seconds": round(time.time() - t0, 1)}
            print(json.dumps(row), flush=True)
            results.append(row)
    summarize(run_dir)
    return results


# ---------------------------------------------------------------------------------------------- summarize
def wilson(k, n, z=1.96):
    """Wilson score 95% interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


HIST_BINS = ["0", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1)", "1"]


def _bin(r):
    if r == 0:
        return "0"
    if r == 1:
        return "1"
    if r <= 0.25:
        return "(0,0.25]"
    if r <= 0.5:
        return "(0.25,0.5]"
    if r <= 0.75:
        return "(0.5,0.75]"
    return "(0.75,1)"


def _dict(x):
    return x if isinstance(x, dict) else {}


def _group_stats(rows):
    valid = [g for g in rows if g["harness"]["valid"]]
    n, k = len(valid), sum(1 for g in valid if g.get("pass"))
    lo, hi = wilson(k, n)
    per_inst = {}
    for g in valid:
        a = per_inst.setdefault(g["harness"]["instance_id"], [0, 0])
        a[0] += bool(g.get("pass"))
        a[1] += 1
    hist = {b: 0 for b in HIST_BINS}
    for kk, nn in per_inst.values():
        hist[_bin(kk / nn)] += 1
    multi = [kk / nn for kk, nn in per_inst.values() if nn >= 2]
    slots = [s for g in valid for s in (_dict(g.get("details")).get("slots") or []) if isinstance(s, dict)]
    null_slots = [s for s in slots if s.get("planted") is False]
    planted_slots = [s for s in slots if s.get("planted") is True]
    fc = sum(1 for s in null_slots if s.get("claimed"))
    invalid_reasons = {}
    for g in rows:
        for r in g["harness"]["invalid_reasons"]:
            invalid_reasons[r] = invalid_reasons.get(r, 0) + 1
    return {
        "n_episodes": len(rows), "n_valid": n, "n_invalid": len(rows) - n, "invalid_reasons": invalid_reasons,
        "n_pass": k, "pass_rate": (k / n) if n else None, "wilson95": [round(lo, 4), round(hi, 4)],
        "mean_score": (sum(float(g.get("score") or 0) for g in valid) / n) if n else None,
        "n_submitted": sum(1 for g in valid if g["harness"]["submitted"]),
        "per_instance": {i: {"pass": kk, "n": nn} for i, (kk, nn) in sorted(per_inst.items())},
        "per_instance_hist": hist,
        "frac_instances_at_0_or_1": (sum(1 for x in multi if x in (0, 1)) / len(multi)) if multi else None,
        "n_slots": len(slots), "n_null_slots": len(null_slots),
        "null_slot_false_claim_rate": (fc / len(null_slots)) if null_slots else None,
        "planted_slot_accuracy": (sum(1 for s in planted_slots if s.get("correct")) / len(planted_slots))
        if planted_slots else None,
        "behavioral_exposure_total": sum(g["harness"].get("behavioral_exposure", 0) for g in rows),
    }


def summarize(run_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(run_dir, "episodes", "*", "grade.json"))):
        g = _load_json(p)
        if isinstance(g, dict) and "harness" in g:
            rows.append(g)
    groups = {}
    for g in rows:
        h = g["harness"]
        key = f"solver={h.get('solver_label')} model={h.get('agent_model')} profile={h['profile']} tier={h.get('tier')}"
        groups.setdefault(key, []).append(g)
    summary = {"run_dir": os.path.abspath(run_dir), "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "overall": _group_stats(rows) if rows else None,
               "groups": {k: _group_stats(v) for k, v in sorted(groups.items())}}
    broker.write_json_atomic(os.path.join(run_dir, "summary.json"), summary)
    md = [f"# Summary: {os.path.basename(os.path.abspath(run_dir))}", "",
          f"Generated {summary['generated_at']} by `python -m common.sandbox summarize`. "
          "Pass rate is over VALID episodes only (no leak, transcript audit passed, grader ran). "
          "CI = Wilson 95%. These numbers describe this run directory only.", "",
          "| group | valid/total | pass | pass rate | Wilson 95% | mean score | null-slot false-claim |",
          "|---|---|---|---|---|---|---|"]

    def fmt(x):
        return "-" if x is None else f"{x:.3f}"

    for k, s in summary["groups"].items():
        md.append(f"| {k} | {s['n_valid']}/{s['n_episodes']} | {s['n_pass']} | {fmt(s['pass_rate'])} | "
                  f"[{s['wilson95'][0]:.3f}, {s['wilson95'][1]:.3f}] | {fmt(s['mean_score'])} | "
                  f"{fmt(s['null_slot_false_claim_rate'])} |")
    for k, s in summary["groups"].items():
        md += ["", f"## {k}", "", "Per-instance pass-rate histogram (instances per bin):", "",
               "| " + " | ".join(HIST_BINS) + " |", "|" + "---|" * len(HIST_BINS),
               "| " + " | ".join(str(s["per_instance_hist"][b]) for b in HIST_BINS) + " |", ""]
        if s["invalid_reasons"]:
            md.append(f"Invalid episodes: {s['invalid_reasons']}")
        md.append(f"Per instance (pass/n): " + ", ".join(f"{i}: {v['pass']}/{v['n']}" for i, v in s["per_instance"].items()))
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    return summary


# ---------------------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(prog="python -m common.sandbox")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    p = sub.add_parser("prepare")
    p.add_argument("--task", required=True)
    p.add_argument("--instance-dir", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--solver-label", default=None)
    p.add_argument("--agent-model", default=None)
    p.add_argument("--tasks-root", default=None, help="checkout holding tasks/<T>/ (default: this checkout)")
    p.add_argument("--json", action="store_true")
    f = sub.add_parser("finish")
    f.add_argument("--episode", required=True)
    f.add_argument("--transcript", nargs="*", default=None)
    f.add_argument("--agent-model", default=None)
    f.add_argument("--search-root", default=None)
    r = sub.add_parser("run-scripted")
    r.add_argument("--task", required=True)
    r.add_argument("--solver", required=True)
    r.add_argument("--instances", nargs="+", required=True)
    r.add_argument("--profile", required=True)
    r.add_argument("--run-dir", required=True)
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--solver-label", default=None)
    r.add_argument("--tasks-root", default=None)
    s = sub.add_parser("summarize")
    s.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "setup":
        setup()
    elif a.cmd == "prepare":
        ep = prepare(a.task, a.instance_dir, a.profile, a.run_dir, a.solver_label, a.tasks_root, a.agent_model)
        if a.json:
            print(json.dumps({k: ep[k] for k in ("episode", "sandbox", "prompt")}))
        else:
            print(ep["episode"])                      # first line = bare id, so E=$(prepare ... | head -1) works
            print(f"SANDBOX {ep['sandbox']}")
            print("----- TEST-AGENT PROMPT (give this, and nothing else, to a fresh subagent) -----")
            print(ep["prompt"])
            print("----- END PROMPT -----")
    elif a.cmd == "finish":
        g = finish(a.episode, a.transcript, a.agent_model, a.search_root)
        h = g["harness"]
        print(json.dumps({"episode": a.episode, "pass": g.get("pass"), "score": g.get("score"), "valid": h["valid"],
                          "invalid_reasons": h["invalid_reasons"], "submitted": h["submitted"]}))
    elif a.cmd == "run-scripted":
        run_scripted(a.task, a.solver, a.instances, a.profile, a.run_dir, a.repeats, a.solver_label, a.tasks_root)
    elif a.cmd == "summarize":
        s = summarize(a.run_dir)
        print(json.dumps(s["overall"], indent=1) if s["overall"] else "no graded episodes")


if __name__ == "__main__":
    main()
