"""Sandbox launcher: prepares episodes, finishes (grades + audits) them, runs scripted solvers, summarizes.

    python -m common.sandbox setup        # once per machine: analysis venv for ./py (numpy, scipy, sklearn; NO torch)
    python -m common.sandbox prepare --task T --instance-dir D --profile full|blackbox \
           --run-dir runs/T/<YYYYmmdd-HHMMSS>_<label> [--solver-label opus] [--json] \
           [--prompt-template FILE] [--min-submit-frac forward=0.6] [--extra-file SRC:DST ...]
    python -m common.sandbox finish --episode E [--transcript PATH ...] [--agent-model MODEL]
    python -m common.sandbox run-scripted --task T --solver tasks/T/reference_solver.py \
           --instances D1 D2 ... --profile full --run-dir R [--repeats 5] [--solver-label reference]
    python -m common.sandbox summarize --run-dir R

An episode sandbox ~/rlsbx/<E>/ holds ONLY: TASK.md, tool (client), py (-> analysis venv python),
out/ (arrays written by tools, large responses, over-long command outputs saved by the API runners), scratch/ (agent's
own files), tmp/ (the agent's temp dir: TMPDIR in the runners; v3 fix 3) and .episode.
Everything privileged (instance dir, answer key, canary, leak strings, counters, tool log) lives in
the episode record <main repo>/runs/.episodes/<E>.json, which only the broker and this module read.
See docs/HARNESS_API.md for the contract.
"""
import argparse
import collections
import fcntl
import glob
import hashlib
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

Your working directory is {sandbox}/ . Only read, write and run things inside that directory. Do not read, list or search any other location on this machine (no other directories, not your home directory, not /tmp, no ".." paths). Do not create symlinks, read environment variables, or run encoded or obfuscated commands (no base64 or hex decoding of commands, no eval, no exec of generated or transformed code). Running your own files from that directory is fine, for example ./py scratch/a.py or, inside Python, exec(open('scratch/a.py').read()); so is running one piece of such a file cut at a fixed text marker, for example exec(open('scratch/a.py').read().split('if __name__')[0]).

Start every shell command with:  cd {sandbox} &&
Then read TASK.md there (cat TASK.md). It explains the task, the tools and the exact answer format.

- Use ./tool to call the task's tools (./tool help lists them; ./tool budget shows what you have left).
- Use ./py (a Python with numpy, scipy and scikit-learn) for analysis, for example of the .npy files that tools save under out/. Keep your own files in scratch/.
- For temporary files use tmp/ in your working directory, never /tmp. (Python's tempfile module already writes there.)
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
        for k in ("profiles", "solver_labels", "agent_models", "prompt_templates", "min_submit_fracs",
                  "extra_files"):
            v = add.get(k)
            if v and v not in cfg.setdefault(k, []):
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
    lines.append(f"- absolute time limit: {broker.hard_cap(caps) / 60:.0f} minutes from your first tool call, "
                 f"including any waiting; after it no submission is accepted")
    return "\n".join(lines)


def fill_public(text, public, where="agent_prompt.md"):
    """Replace every {public.<key>} in `text` with public.json's value (strings as is, anything else as JSON). A key
    that public.json lacks raises KeyError, so a template can never reach an agent with a placeholder left in it."""
    def sub_public(m):
        key = m.group(1)
        if key not in public:
            raise KeyError(f"{where} uses {{public.{key}}} but public.json has no '{key}'")
        v = public[key]
        return v if isinstance(v, str) else json.dumps(v)

    return re.sub(r"\{public\.([A-Za-z0-9_]+)\}", sub_public, text)


def fill_tool_docs(tool_docs, public):
    """Tool docs may use {public.<key>} placeholders too (2026-10-02, ShiftHunt smoke F7): a task whose tools take
    different arguments per instance (e.g. an access mode) documents exactly the arguments this instance accepts.
    Filled at prepare, so TASK.md and `./tool help` (served from the episode record) show the same text."""
    return [dict(t, doc=fill_public(t["doc"], public, where=f"tool doc of {t['name']}")) for t in tool_docs]


def render_task_md(template, public, tool_docs_md, caps_md):
    out = fill_public(template, public)
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


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- prepare options: --min-submit-frac (persistence requirement enforced by the sandbox's ./tool client)
CLIENT_POLICY_LINE = "MIN_SUBMIT_FRAC = {}  # set by common.sandbox prepare --min-submit-frac"
BUDGET_KEYS_EXTRA = ("wall_clock_s",)


def parse_min_submit_frac(spec, caps):
    """'forward=0.6' or 'forward=0.6,generate=0.3' (or a list of such strings) -> {"forward": 0.6, ...}.
    Keys must be budget names this episode has (counter kinds such as forward/generate/tool_calls, or
    wall_clock_s); fractions must be in (0, 1]."""
    if not spec:
        return {}
    items = [spec] if isinstance(spec, str) else list(spec)
    out = {}
    allowed = list(broker.counter_kinds(caps)) + list(BUDGET_KEYS_EXTRA)
    for item in items:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"--min-submit-frac: expected key=fraction, got {part!r}")
            k, v = (x.strip() for x in part.split("=", 1))
            if k not in allowed:
                raise ValueError(f"--min-submit-frac: unknown budget {k!r} (this episode has: {', '.join(allowed)})")
            try:
                fv = float(v)
            except ValueError:
                raise ValueError(f"--min-submit-frac: {k}={v!r} is not a number")
            if not (0 < fv <= 1) or not math.isfinite(fv):
                raise ValueError(f"--min-submit-frac: {k}={fv} must be a fraction in (0, 1]")
            out[k] = fv
    return out


def _budget_label(k):
    return {"wall_clock_s": "time", "tool_calls": "tool-call"}.get(k, k)


def render_min_submit_note(policy):
    """The one-line TASK.md note that tells the agent about the submission requirement."""
    parts = [f"{int(round(v * 100))}% of your {_budget_label(k)} budget" for k, v in sorted(policy.items())]
    return ("Note: `./tool submit` is accepted only after you have used at least " + " and ".join(parts)
            + " (or once your time or one of your budgets has run out). `./tool budget` shows what you have used.")


def render_client(policy=None):
    """The ./tool client as written into a sandbox: the stdlib client with the episode's submit policy baked in."""
    with open(CLIENT_SRC) as f:
        src = f.read()
    if CLIENT_POLICY_LINE not in src:
        raise RuntimeError("agent client is missing its MIN_SUBMIT_FRAC line")
    if policy:
        src = src.replace(CLIENT_POLICY_LINE,
                          f"MIN_SUBMIT_FRAC = {json.dumps(policy, sort_keys=True)}  "
                          "# set by common.sandbox prepare --min-submit-frac", 1)
    return src


def min_submit_check(rec):
    """Was the submit policy satisfied when the episode was submitted? (The client enforces it; this catches a
    submission that bypassed the client.) Exempt when the time cap or any counter cap had run out."""
    policy = rec.get("min_submit_frac") or {}
    if not policy:
        return None
    caps, counters = rec.get("caps") or {}, rec.get("counters") or {}
    used = {}
    for k in policy:
        if k == "wall_clock_s":
            if rec.get("started_at") and rec.get("submitted_at"):
                el = rec["submitted_at"] - rec["started_at"] - rec.get("wait_s", 0.0)
            else:
                el = 0.0
            used[k] = el / caps[k] if caps.get(k) else 1.0
        else:
            used[k] = counters.get(k, 0) / caps[k] if caps.get(k) else 1.0
    exhausted = [k for k in broker.counter_kinds(caps)
                 if (caps.get(k) or 0) > 0 and counters.get(k, 0) >= caps[k]]
    if rec.get("started_at") and rec.get("submitted_at"):
        if rec["submitted_at"] - rec["started_at"] - rec.get("wait_s", 0.0) >= caps.get("wall_clock_s", float("inf")):
            exhausted.append("wall_clock_s")
    met = all(used[k] >= v - 1e-9 for k, v in policy.items())
    return {"policy": policy, "used_frac": {k: round(v, 4) for k, v in used.items()}, "met": met,
            "exempt_exhausted": exhausted, "submitted": rec.get("submission") is not None,
            "bypassed": bool(rec.get("submission") is not None and not met and not exhausted)}


# ---- prepare options: --extra-file SRC:DST (copied into the sandbox after a leak scan)
SANDBOX_TMP = "tmp"          # the episode's own temp dir (TMPDIR in the runners); see prepare
RESERVED_SANDBOX_NAMES = ("TASK.md", "tool", "py", ".episode", "out", SANDBOX_TMP)


def parse_extra_files(specs):
    """['SRC:DST', ...] -> [(abs src, normalised relative dst)]. DST must stay inside the sandbox and must not
    replace a harness file (TASK.md, tool, py, .episode) or go under out/ (tool-written files)."""
    out = []
    for spec in specs or ():
        if ":" not in spec:
            raise ValueError(f"--extra-file: expected SRC:DST, got {spec!r}")
        src, dst = spec.rsplit(":", 1)
        src = os.path.abspath(os.path.expanduser(src))
        if not os.path.isfile(src):
            raise FileNotFoundError(f"--extra-file: source {src} is not a file")
        if not dst or os.path.isabs(dst) or dst.startswith("~"):
            raise ValueError(f"--extra-file: DST must be a relative path inside the sandbox, got {dst!r}")
        nd = os.path.normpath(dst)
        if nd == "." or nd.startswith("..") or nd.split(os.sep)[0] in RESERVED_SANDBOX_NAMES or nd.endswith(os.sep):
            raise ValueError(f"--extra-file: DST {dst!r} is outside the sandbox or replaces a harness file/dir")
        if nd in [d for _, d in out]:
            raise ValueError(f"--extra-file: DST {dst!r} given twice")
        out.append((src, nd))
    return out


def check_extra_file(src, dst, task, canary, leak_strings, private_strings):
    """Leak-scan an extra file (contents and destination name) before it goes into the sandbox."""
    with open(src, "rb") as f:
        text = f.read().decode("utf-8", errors="replace")
    problems = [p.replace("TASK.md", f"extra file {dst}")
                for p in check_task_md(text, task, canary, leak_strings, private_strings)]
    name = leakscan.scan_text(dst, canary, leak_strings, private_strings)
    if name["canary"] or name["leak_strings"] or name["private"]:
        problems.append(f"destination name {dst!r} contains a protected string")
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
            start_broker=True, prompt_template=None, min_submit_frac=None, extra_files=None):
    """Create one episode. Options (all recorded in the episode record and the run's config.json):
    prompt_template: render TASK.md from this file instead of tasks/<task>/agent_prompt.md (same checks).
    min_submit_frac: {"forward": 0.6} or "forward=0.6": ./tool refuses submit until that fraction of the cap is
        used (or a budget/time has run out); a one-line note is appended to TASK.md.
    extra_files: ["SRC:DST", ...]: copy SRC to <sandbox>/DST after leak-scanning it."""
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
    caps.setdefault("wall_clock_hard_s", int(round(broker.hard_cap(caps))))   # explicit in the record (v3 fix 1)
    policy = parse_min_submit_frac(min_submit_frac, caps)
    extras = parse_extra_files(extra_files)
    desc = _describe(task_root, task, profile)
    desc["tool_docs"] = fill_tool_docs(desc["tool_docs"], public)
    template_path = os.path.abspath(prompt_template) if prompt_template else os.path.join(tdir, "agent_prompt.md")
    with open(template_path) as f:
        template = f.read()
    template_info = {"path": template_path, "sha256": _sha256_file(template_path),
                     "default": not prompt_template}
    task_md = render_task_md(template, public, render_tool_docs(desc["tool_docs"]), render_caps(caps))
    if policy:
        task_md = task_md.rstrip("\n") + "\n\n" + render_min_submit_note(policy) + "\n"
    private = private_strings_for(task, task_root, instance_dir, inst, public)
    problems = check_task_md(task_md, task, canary, leak_strings, private)
    if problems:
        raise ValueError("TASK.md failed checks: " + "; ".join(problems))
    extra_info = []
    for src, dst in extras:
        bad = check_extra_file(src, dst, task, canary, leak_strings, private)
        if bad:
            raise ValueError(f"extra file {src} refused: " + "; ".join(bad))
        extra_info.append({"src": src, "dst": dst, "sha256": _sha256_file(src), "bytes": os.path.getsize(src)})
    vpy = os.path.join(paths.venv_dir(), "bin", "python")
    if not os.path.exists(vpy):
        raise FileNotFoundError(f"analysis venv missing ({vpy}); run: python -m common.sandbox setup")

    eid = "ep" + secrets.token_hex(5)
    sbx = os.path.join(paths.sandbox_root(), eid)
    os.makedirs(os.path.join(sbx, "out"))
    os.makedirs(os.path.join(sbx, "scratch"))
    # Sandbox-local temp dir (v3 fix 3). WHY: 5 Haiku episodes (ShiftHunt scale1 T1) were INVALID for writing tool
    # output to /tmp (`./tool ... > /tmp/x.json`), and Python's tempfile defaults there too. The runners set
    # TMPDIR/TMP/TEMP to this directory and the agent prompt names it; /tmp itself stays an audit violation.
    os.makedirs(os.path.join(sbx, SANDBOX_TMP))
    with open(os.path.join(sbx, "TASK.md"), "w") as f:
        f.write(task_md)
    with open(os.path.join(sbx, "tool"), "w") as f:
        f.write(render_client(policy))
    os.chmod(os.path.join(sbx, "tool"), 0o755)
    for e in extra_info:
        dp = os.path.join(sbx, e["dst"])
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        shutil.copyfile(e["src"], dp)
        os.chmod(dp, 0o644)
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
           "leak_detected": False, "leak_events": [], "behavioral_exposure": 0, "git_sha": sha, "git_dirty": dirty,
           "prompt_template": template_info, "min_submit_frac": policy, "extra_files": extra_info}
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
                   dial=inst.get("dial"), instance_dir=instance_dir,
                   prompt_templates={"path": template_path, "sha256": template_info["sha256"]},
                   min_submit_fracs=policy or None,
                   extra_files=[{"dst": e["dst"], "src": e["src"], "sha256": e["sha256"]} for e in extra_info] or None)
    if start_broker:
        broker.start(quiet=True)
    return {"episode": eid, "sandbox": sbx, "prompt": prompt, "episode_dir": edir}


def _public_record(rec):
    """The run-dir copy of the episode record: no canary, no leak strings, no submission internals."""
    keep = ("episode", "task", "instance_id", "tier", "profile", "solver_label", "agent_model", "sandbox",
            "instance_dir", "caps", "counters", "tools", "gpu_gb", "status", "created_at", "started_at",
            "submitted_at", "closed_at", "wait_s", "leak_detected", "behavioral_exposure", "git_sha", "git_dirty",
            "prompt_template", "min_submit_frac", "extra_files")
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


# ---- task codename in the agent's OWN files (2026-10-02, FeatureMatch audit MINOR 8a).
# The codename is a private string so that private task material copied into the sandbox is caught. But the codenames
# are ordinary English compounds of the task's subject ("featurematch", "latentdiff", "freqhunt"), so an agent can
# coin one by itself, e.g. by naming a script featurematching.py. That one word is not evidence of a leak, and it
# made the episode INVALID. Rule now:
# - harness-written files (TASK.md as written, tool, py, .episode, unedited extra files) and the names of tool-written
#   arrays (out/*.npy): the bare codename is still a leak (harness text must never contain it);
# - files the agent wrote (contents and names): the codename is a leak only in a form that names the privileged side,
#   which an agent cannot coin: a repo path or module path (tasks/<codename>, runs/<codename>, wt/<codename>,
#   tasks.<codename>) or the canary prefix (RLCANARY-<codename>). The bare word is only counted
#   (grade.json harness.agent_files_with_codename) so a reviewer can look.
# Every other private string (instance dir and id, repo paths, episode registry) stays a leak anywhere, and the
# canary and its hex tail stay leaks anywhere. Reading the repo is caught by the transcript audit (R2/R3) in any case.
def _codename_set(task):
    return {v for v in codename_variants(task or "") if len(v) >= 6}


def _codename_privileged_rx(codenames):
    alts = "|".join(re.escape(c) for c in sorted(codenames, key=len, reverse=True))
    return re.compile(r"(?<![a-z0-9_])(?:(?:tasks|runs|wt)[/\\.](?:" + alts + r")(?=$|[/\\.\s'\"`,;:)\]}])"
                      r"|rlcanary-(?:" + alts + r"))")


def _split_private_hits(hit_idx, private, codenames):
    """Private-string hit indices -> (non-codename hits, codename hits)."""
    other = [i for i in hit_idx if private[i] not in codenames]
    return other, [i for i in hit_idx if private[i] in codenames]


def _scan_sandbox(rec):
    """Scan the sandbox at finish (file contents AND file names):
    - canary or a private string (instance path/id, repo path) anywhere => leak; the task codename => leak in
      harness files and tool-written array names, but in the agent's own files only in a repo-path/canary form
      (the bare word is counted in agent_files_with_codename; see the comment above _codename_set);
    - leak_strings in harness-written files (TASK.md as written, tool, py, .episode) or in the names of
      tool-written arrays (out/*.npy) => leak;
    - leak_strings in files the agent wrote are expected when it found the answer, so they are only
      counted (agent_files_with_leak_strings), never flagged. A TASK.md the agent edited counts as its file;
    - a symlink that points outside the sandbox => symlinks_outside (finish marks the episode INVALID:
      reading through it would leave the sandbox without naming an outside path)."""
    sbx = os.path.realpath(rec["sandbox"])
    canary, leaks, private = rec.get("canary"), rec.get("leak_strings"), rec.get("private_strings", [])
    pristine = {"tool": render_client(rec.get("min_submit_frac")).encode()}
    edir = os.path.join(rec.get("run_dir") or "", "episodes", rec.get("episode") or "")
    if rec.get("run_dir") and os.path.exists(os.path.join(edir, "TASK.md")):
        with open(os.path.join(edir, "TASK.md"), "rb") as f:
            pristine["TASK.md"] = f.read()
    extra_sha = {e["dst"]: e["sha256"] for e in rec.get("extra_files") or []}

    def same_as_written(rel, p):
        if rel in (".episode", "py"):
            return True
        if rel in extra_sha:
            try:
                return _sha256_file(p) == extra_sha[rel]
            except OSError:
                return False
        src = pristine.get(rel)
        if src is None:
            return rel == "TASK.md"         # no copy to compare against: treat as harness-written
        try:
            with open(p, "rb") as b:
                return b.read() == src
        except OSError:
            return False

    codenames = _codename_set(rec.get("task")) & set(private or [])
    cn_rx = _codename_privileged_rx(codenames) if codenames else None
    res = {"leak": False, "reasons": [], "agent_files_with_leak_strings": 0, "n_files": 0, "symlinks_outside": [],
           "agent_files_with_codename": []}

    def codename_is_leak(text):
        return bool(cn_rx and cn_rx.search(leakscan.norm_keep_ws(text)))

    for dp, dns, fns in os.walk(sbx):
        for fn in sorted(fns + [d for d in dns if os.path.islink(os.path.join(dp, d))]):
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, sbx)
            name_hit = leakscan.scan_text(rel, canary, leaks, private)
            tool_array = rel.startswith("out" + os.sep) and rel.endswith(".npy")
            n_other, n_cn = _split_private_hits(name_hit["private"], private, codenames)
            if name_hit["canary"] or n_other or (n_cn and (tool_array or codename_is_leak(rel))):
                res["leak"] = True
                res["reasons"].append("canary or private string in a file name")
            elif n_cn and rel not in res["agent_files_with_codename"]:
                res["agent_files_with_codename"].append(rel)
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
            harness_file = (rel in ("TASK.md", "tool", "py", ".episode") or rel in extra_sha) and same_as_written(rel, p)
            if hit["canary"]:
                res["leak"] = True
                res["reasons"].append(f"canary in sandbox file {rel}")
            h_other, h_cn = _split_private_hits(hit["private"], private, codenames)
            if h_other:
                res["leak"] = True
                res["reasons"].append(f"private string (instance path/id, repo path or codename) in sandbox file {rel}")
            elif h_cn and (harness_file or codename_is_leak(text)):
                res["leak"] = True
                res["reasons"].append(f"task codename in a harness file or in a repo-path/canary form in sandbox file {rel}")
            elif h_cn and rel not in res["agent_files_with_codename"]:
                res["agent_files_with_codename"].append(rel)
            if hit["leak_strings"]:
                if harness_file:
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


# ---- infrastructure failures and broken scripted solvers (2026-10-02; LatentDiff audit MAJOR-2, FeatureMatch audit
# MINOR 8b). WHY: an episode that never got compute, or whose tool server died and never came back, and that then
# ended without a submission, was graded valid:true with score 0. Every reader (summaries, gates, RL rewards) then
# counted an infrastructure problem as the solver failing the task, which biases every "must fail" gate toward
# passing and every pass rate down. Such episodes are now INVALID with reason `infra_failure`.
# Infrastructure-failure entries in the tool log:
#   - a response "tool server failed to start; try again later" (GPU-queue admission timeout, or the model load crashed);
#   - the event server_stopped with reason "server died" or "found dead" (the tool-server process died: OOM, crash,
#     RAM-cap kill; the in-flight call returned "tool failed (internal error)");
#   - the event client_disconnected_while_waiting (the client gave up, or was killed, while queued for compute) --
#     but this one is SERVER-side only for scripted solvers; see below.
# The episode is `infra_failure` only if it has no submission AND no successful task-tool call came after the last
# COUNTED entry (it never recovered). An episode that recovered and then ran out of time is an ordinary fail.
#
# WHY client disconnect is not infra for LLM episodes (audit_replay finding F2): the broker logs
# `client_disconnected_while_waiting` whenever the client process dies while queued -- including when the agent's OWN
# `timeout N ./tool <gpu tool>` expires (observed in residualrecall probe 3: `timeout 100`/`timeout 110` produced the
# event at 100.01 s / 110.01 s). A failing LLM agent could make its last call `timeout 5 ./tool <gpu tool>` while a
# queue exists and then not submit, turning a scored fail into INVALID `infra_failure` and skewing "must fail" gates
# and RL rewards -- the exact bias this reason was meant to remove. The tool log cannot tell an agent-issued timeout
# from the agent runner's own ~900 s command limit (api_agent.py), and an elapsed-time threshold is just as forgeable
# (the agent can pick `timeout 910`). So for LLM episodes we count only SERVER-side failures, which the agent cannot
# manufacture at will: `server_failed_to_start` and `server died`/`found dead`. Scripted solvers are trusted builder
# code (not the adversarial side) and also get `no_successful_call`/`solver_failed`, so their client disconnects --
# the run-scripted driver's 1800 s wait limit giving up under real contention -- still count.
# Residual limitation: a server OOM/`server died` that an LLM agent deliberately induced and then abandons would
# still read as infra; inducing it is far harder than a `timeout` and is self-limited by the per-call budget and the
# server RAM cap, so it is left as a documented gap rather than guessed at from the log.
INFRA_ERROR_PREFIXES = ("tool server failed to start",)
INFRA_STOP_REASONS = ("server died", "found dead")
CLIENT_DISCONNECT_EVENT = "client_disconnected_while_waiting"
BUILTIN_TOOLS = ("help", "budget", "submit")


def tool_log_health(log_path, client_disconnect_is_infra=True):
    """{"infra_failures": [{"n", "kind"}], "n_ok_task_calls": int, "recovered": bool} from a tool log.
    client_disconnect_is_infra=False (LLM episodes) ignores `client_disconnected_while_waiting`, which the agent can
    trigger itself with `timeout N ./tool ...` (finding F2); only server-side failures then count."""
    fails, n_ok, recovered = [], 0, True
    try:
        f = open(log_path)
    except OSError:
        return {"infra_failures": [], "n_ok_task_calls": 0, "recovered": True}
    with f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = None
            if e.get("event") == CLIENT_DISCONNECT_EVENT and client_disconnect_is_infra:
                kind = e["event"]
            elif e.get("event") == "server_stopped" and e.get("reason") in INFRA_STOP_REASONS:
                kind = "server_" + e["reason"].replace(" ", "_")
            elif e.get("event") is None and e.get("ok") is False:
                try:
                    err = str((json.loads(e.get("response") or "{}") or {}).get("error") or "")
                except (json.JSONDecodeError, AttributeError):
                    err = ""
                if err.startswith(INFRA_ERROR_PREFIXES):
                    kind = "server_failed_to_start"
            if kind:
                fails.append({"n": e.get("n"), "kind": kind})
                recovered = False
            elif e.get("event") is None and e.get("ok") is True and e.get("tool") not in BUILTIN_TOOLS:
                n_ok += 1
                recovered = True
    return {"infra_failures": fails, "n_ok_task_calls": n_ok, "recovered": recovered}


def episode_prompt(rec):
    """The exact test-agent prompt prepare printed for this episode: <run_dir>/episodes/<E>/agent_prompt.txt without
    the one newline prepare appends (falls back to re-rendering AGENT_PROMPT if the file is missing)."""
    try:
        with open(os.path.join(rec["run_dir"], "episodes", rec["episode"], "agent_prompt.txt")) as f:
            txt = f.read()
        return txt[:-1] if txt.endswith("\n") else txt
    except OSError:
        return AGENT_PROMPT.format(sandbox=rec["sandbox"])


def _last_activity(log_path):
    """End time of the last agent call in a tool log (t + elapsed_s of entries that name a tool), or None. Server
    events are skipped: finish's own close logs a server_stopped event at finish time."""
    last = None
    try:
        with open(log_path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = e.get("t")
                if isinstance(t, (int, float)) and e.get("tool"):
                    last = max(last or 0.0, t + float(e.get("elapsed_s") or 0.0))
    except OSError:
        pass
    return last


def episode_time_report(rec, log_path, edir):
    """Both episode times for grade.json (v3 fix 1): agent time (since the first tool call, minus compute wait; capped
    by wall_clock_s) and total time (wait included; capped by the absolute ceiling broker.hard_cap). The end is the
    submission; for an episode without one, the hard-limit close if the broker closed it, else the last tool-log
    activity (NOT the finish time, which can be hours later). If the agent runner wrote api_meta.json next to the
    transcript, its own clock (runner_total_s: from runner start to stop, i.e. including model time after the last
    tool call) and stop reason are added. over_time: None, "nominal" (past wall_clock_s) or "hard" (past the ceiling;
    by either clock)."""
    r = dict(rec)
    if r.get("submitted_at") is None:
        r["closed_at"] = (rec.get("closed_at") if rec.get("close_reason") == "wall_clock_hard"
                          else _last_activity(log_path) or rec.get("started_at"))
    t = broker.episode_times(r)
    out = {"agent_s": t["agent_s"], "total_s": t["total_s"], "wait_s": t["wait_s"], "cap_s": t["cap_s"],
           "hard_cap_s": t["hard_cap_s"], "close_reason": rec.get("close_reason")}
    meta = _load_json(os.path.join(edir, "api_meta.json"))
    if isinstance(meta, dict) and meta.get("episode") == rec.get("episode"):
        out["runner_total_s"] = meta.get("total_time_s", meta.get("wall_s"))
        out["runner_stop"] = meta.get("stop")
    over_hard = t["over_hard"] or (isinstance(out.get("runner_total_s"), (int, float))
                                   and out["runner_total_s"] > t["hard_cap_s"])
    out["over_time"] = "hard" if over_hard else ("nominal" if t["over_nominal"] else None)
    return out


def finish(episode, transcripts=None, agent_model=None, search_root=None, solver_rc=None):
    """Grade, scan and audit one episode. solver_rc: the scripted solver's exit code (run-scripted passes it; an int,
    or "timeout"). With it, the episode is INVALID if rc != 0 (`solver_failed`) or if the solver made no successful
    task-tool call and did not submit (`no_successful_call`)."""
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
    if agent_model:
        # 2026-10-02 ShiftHunt smoke F4: a model given only at finish (the usual case: prepare runs before the agent
        # is launched) was recorded in grade.json but not in the run's config.json (agent_models stayed []).
        _update_config(rec["run_dir"], rec["task"], rec.get("task_root") or HARNESS_ROOT, agent_models=agent_model)

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
        # 2026-10-02 smoke F1: the transcript must start with THIS episode's agent prompt (exactly, or inside the
        # documented workflow wrapper); search finds only such transcripts and refuses to pick one of several (R0).
        ts, r0, info = transcript_audit.locate(episode, episode_prompt(rec), transcripts, search_root)
        audit = transcript_audit.audit(episode, rec["sandbox"], ts, r0=r0, info=info)
        # The run dir holds exactly the transcripts this audit used. WHY: a re-finish used to leave older copies
        # (transcript_1.jsonl from a first finish that had also picked up an operator's transcript) next to the new
        # ones (smoke ep3bd6b501bc). Sources are read first: one may be a copy in this very directory.
        data = []
        for t in ts:
            with open(t, "rb") as f:
                data.append(f.read())
        for old in glob.glob(os.path.join(edir, "transcript.jsonl")) + glob.glob(os.path.join(edir, "transcript_*.jsonl")):
            os.unlink(old)
        for i, b in enumerate(data):
            with open(os.path.join(edir, "transcript.jsonl" if i == 0 else f"transcript_{i}.jsonl"), "wb") as f:
                f.write(b)
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
    msc = min_submit_check(rec)
    if msc and msc["bypassed"]:
        invalid.append("min_submit_bypassed")
    submitted = rec.get("submission") is not None
    # F2: for LLM episodes a client disconnect is not infra (the agent can cause it with `timeout N ./tool`); only
    # server-side failures count. Scripted solvers are trusted and also gated by no_successful_call/solver_failed.
    health = tool_log_health(log_dst, client_disconnect_is_infra=not is_llm)
    if not submitted and health["infra_failures"] and not health["recovered"]:
        invalid.append("infra_failure")
    if solver_rc is not None:
        if solver_rc != 0:
            invalid.append("solver_failed")
        if not submitted and health["n_ok_task_calls"] == 0:
            invalid.append("no_successful_call")
    end = rec.get("submitted_at") or rec.get("closed_at") or time.time()
    times = episode_time_report(rec, log_dst, edir)
    grade["harness"] = {
        "episode": episode, "task": rec["task"], "instance_id": rec.get("instance_id"), "tier": rec.get("tier"),
        "profile": rec["profile"], "solver_label": rec.get("solver_label"), "agent_model": rec.get("agent_model"),
        "submitted": rec.get("submission") is not None, "status": rec["status"], "counters": rec["counters"],
        "caps": rec["caps"], "behavioral_exposure": rec.get("behavioral_exposure", 0),
        "leak_detected": leak, "leak_reasons": [r for e in rec.get("leak_events", []) for r in e["reasons"]]
        + sbx_scan["reasons"] + log_scan["reasons"],
        "agent_files_with_leak_strings": sbx_scan["agent_files_with_leak_strings"],
        "agent_files_with_codename": sbx_scan["agent_files_with_codename"],
        "symlinks_outside": sbx_scan["symlinks_outside"],
        "solver_rc": solver_rc, "infra_failures": health["infra_failures"],
        "n_ok_task_calls": health["n_ok_task_calls"],
        "cross_episode_access": bool(rec.get("cross_episode_access")),
        "audit_valid": audit["valid"], "audit_skipped": audit.get("skipped"),
        "n_audit_violations": len(audit["violations"]),
        "transcript_discovery": audit.get("discovery"),
        "prompt_match": [c["prompt_match"] for c in audit.get("prompt_check", [])],
        "valid": not invalid, "invalid_reasons": invalid,
        "elapsed_s": round(end - rec["started_at"], 1) if rec.get("started_at") else None,
        "wait_s": round(rec.get("wait_s", 0.0), 1),
        "time": times, "over_time": times["over_time"],
        "prompt_template": rec.get("prompt_template"), "min_submit": msc,
        "extra_files": [{"dst": e["dst"], "sha256": e["sha256"]} for e in rec.get("extra_files") or []],
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
    _prune_sandbox(rec, edir)
    return grade


PRUNE_MIN_BYTES = int(os.environ.get("RL_PRUNE_MIN_BYTES", "100000"))


def _prune_sandbox(rec, edir):
    """After grading (2026-10-02, disk hit 99%: ~9,500 finished sandboxes held 33 GB of tool-output arrays): delete files
    larger than PRUNE_MIN_BYTES from the sandbox's out/ and scratch/. They were leak-scanned above, and their tool calls
    are in tool_log.jsonl. Small files (agent scripts, notes) are kept for transcript analysis. RL_KEEP_SANDBOX=1 disables."""
    if os.environ.get("RL_KEEP_SANDBOX") == "1":
        return
    removed, freed = [], 0
    for sub in ("out", "scratch", SANDBOX_TMP):
        for root, _dirs, files in os.walk(os.path.join(rec["sandbox"], sub)):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    sz = os.path.getsize(fp)
                    if sz > PRUNE_MIN_BYTES and not os.path.islink(fp):
                        os.remove(fp)
                        removed.append({"path": os.path.relpath(fp, rec["sandbox"]), "bytes": sz})
                        freed += sz
                except OSError:
                    pass
    if removed:
        try:
            broker.write_json_atomic(os.path.join(edir, "sandbox_pruned.json"), {"freed_bytes": freed, "files": removed})
        except OSError:
            pass


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
            g = finish(eid, solver_rc=rc)
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
        # v3 fix 1: time-limit exceedance, over ALL episodes (valid or not). Graded before 2026-10-04 => "unknown".
        "over_time": dict(sorted(collections.Counter(
            str(g["harness"].get("over_time")) if "over_time" in g["harness"] else "unknown" for g in rows).items())),
        "max_total_time_s": max([_dict(g["harness"].get("time")).get("total_s") or 0 for g in rows] or [0]),
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
          "| group | valid/total | pass | pass rate | Wilson 95% | mean score | null-slot false-claim | over time (nominal/hard) |",
          "|---|---|---|---|---|---|---|---|"]

    def fmt(x):
        return "-" if x is None else f"{x:.3f}"

    for k, s in summary["groups"].items():
        md.append(f"| {k} | {s['n_valid']}/{s['n_episodes']} | {s['n_pass']} | {fmt(s['pass_rate'])} | "
                  f"[{s['wilson95'][0]:.3f}, {s['wilson95'][1]:.3f}] | {fmt(s['mean_score'])} | "
                  f"{fmt(s['null_slot_false_claim_rate'])} | "
                  f"{s['over_time'].get('nominal', 0)}/{s['over_time'].get('hard', 0)} |")
    for k, s in summary["groups"].items():
        md += ["", f"## {k}", "", "Per-instance pass-rate histogram (instances per bin):", "",
               "| " + " | ".join(HIST_BINS) + " |", "|" + "---|" * len(HIST_BINS),
               "| " + " | ".join(str(s["per_instance_hist"][b]) for b in HIST_BINS) + " |", ""]
        if s["invalid_reasons"]:
            md.append(f"Invalid episodes: {s['invalid_reasons']}")
        if s["over_time"].get("nominal") or s["over_time"].get("hard"):
            md.append(f"TIME LIMIT EXCEEDED: {s['over_time'].get('nominal', 0)} episode(s) past the nominal cap, "
                      f"{s['over_time'].get('hard', 0)} past the absolute ceiling (max total {s['max_total_time_s']} s)")
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
    p.add_argument("--prompt-template", default=None,
                   help="render TASK.md from this file instead of tasks/<T>/agent_prompt.md (same placeholders/checks)")
    p.add_argument("--min-submit-frac", action="append", default=None, metavar="CAP=FRAC[,CAP=FRAC]",
                   help="./tool refuses submit until this fraction of the cap is used, e.g. forward=0.6")
    p.add_argument("--extra-file", action="append", default=None, metavar="SRC:DST",
                   help="copy SRC into the sandbox at relative path DST after a leak scan (repeatable)")
    f = sub.add_parser("finish")
    f.add_argument("--episode", required=True)
    f.add_argument("--transcript", nargs="*", default=None,
                   help="the test agent's transcript(s); each must start with this episode's agent prompt. Default: search "
                        "~/.claude/projects for the one transcript that does (none or several => INVALID, R0)")
    f.add_argument("--agent-model", default=None)
    f.add_argument("--search-root", default=None)
    f.add_argument("--solver-rc", default=None,
                   help="scripted solver's exit code (or 'timeout'): rc != 0 or no successful call and no submit => INVALID")
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
        ep = prepare(a.task, a.instance_dir, a.profile, a.run_dir, a.solver_label, a.tasks_root, a.agent_model,
                     prompt_template=a.prompt_template, min_submit_frac=a.min_submit_frac, extra_files=a.extra_file)
        if a.json:
            print(json.dumps({k: ep[k] for k in ("episode", "sandbox", "prompt")}))
        else:
            print(ep["episode"])                      # first line = bare id, so E=$(prepare ... | head -1) works
            print(f"SANDBOX {ep['sandbox']}")
            print("----- TEST-AGENT PROMPT (give this, and nothing else, to a fresh subagent) -----")
            print(ep["prompt"])
            print("----- END PROMPT -----")
    elif a.cmd == "finish":
        rc = a.solver_rc
        if rc is not None and re.fullmatch(r"-?\d+", rc):
            rc = int(rc)
        g = finish(a.episode, a.transcript, a.agent_model, a.search_root, solver_rc=rc)
        h = g["harness"]
        print(json.dumps({"episode": a.episode, "pass": g.get("pass"), "score": g.get("score"), "valid": h["valid"],
                          "invalid_reasons": h["invalid_reasons"], "submitted": h["submitted"],
                          "prompt_match": h.get("prompt_match")}))
    elif a.cmd == "run-scripted":
        run_scripted(a.task, a.solver, a.instances, a.profile, a.run_dir, a.repeats, a.solver_label, a.tasks_root)
    elif a.cmd == "summarize":
        s = summarize(a.run_dir)
        print(json.dumps(s["overall"], indent=1) if s["overall"] else "no graded episodes")


if __name__ == "__main__":
    main()
