"""Budget-capped Claude API test agent for one harness episode (fast difficulty probes).

Why this exists: Dan granted a TOTAL Anthropic API budget of $17 (2026-10-01) to
measure task difficulty early, with fast feedback loops, before a lot gets built.
Fresh Claude Code subagents stay the free, high-volume option. The API agent adds
exact model control, a clean context (no Claude Code system prompt) and tighter
containment. The model can only run shell commands *inside its sandbox*, with a
scrubbed environment (no HF token, no API key, HOME = sandbox).

Budget safety (hard rules):
  * Every request's cost is computed from response.usage at list prices and
    appended to the machine-wide ledger ~/.rl_api/ledger.jsonl (flock-guarded,
    outside every checkout, so all worktrees share one budget). `spent --snapshot`
    copies the deduplicated ledger to runs/api_budget/ledger_snapshot.jsonl in the
    main repo for git, so every dollar traces to an episode.
  * GLOBAL_CAP_USD (default 16.0, leaving $1 of slack under Dan's $17) is checked
    before every request, using a pessimistic estimate of the next request's cost.
    If the estimate would cross the cap, the episode stops without calling the API.
  * Each episode also has --max-usd (default 1.25) and --max-turns caps.

Usage (after `python -m common.sandbox prepare ...` has printed episode E):
    python -m common.api_agent run --episode E --prompt-file <run>/episodes/E/agent_prompt.txt \
        --model claude-opus-5-5 --effort medium --out <run>/episodes/E
    python -m common.sandbox finish --episode E --transcript <run>/episodes/E/api_transcript.jsonl \
        --agent-model api:claude-opus-5-5:medium
    python -m common.api_agent spent        # ledger total and per-model / per-task breakdown

The transcript is written in a Claude-Code-like JSONL shape: assistant entries whose
message.content holds tool_use blocks named "Bash" with input.command. The standard
transcript audit (common/transcript_audit.py) therefore applies unchanged.
"""
import argparse
import atexit
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
try:
    from common.paths import MAIN_REPO
except Exception:  # pragma: no cover
    MAIN_REPO = REPO
from common import broker as _broker, paths as _paths  # noqa: E402
# The canonical ledger lives OUTSIDE every checkout so all worktrees share one budget. Code before 2026-10-01 16:30
# wrote per-worktree ledgers at <checkout>/runs/api_budget/ledger.jsonl; those are still read (deduplicated) so the
# global total counts every dollar wherever it was spent.
LEDGER_DIR = os.path.expanduser(os.environ.get("RL_API_LEDGER_DIR", "~/.rl_api"))
LEDGER = os.path.join(LEDGER_DIR, "ledger.jsonl")
LOCK = os.path.join(LEDGER_DIR, ".lock")
CAPS_FILE = os.path.join(MAIN_REPO, "runs", "api_budget", "task_caps.json")
SNAPSHOT = os.path.join(MAIN_REPO, "runs", "api_budget", "ledger_snapshot.jsonl")
GLOBAL_CAP_USD = float(os.environ.get("API_GLOBAL_CAP_USD", "16.0"))
KEY_FILE = os.path.expanduser("~/.anthropic_env")
SBX_ROOT = _paths.sandbox_root()       # RL_SANDBOX_ROOT aware (tests run an isolated broker)

# $ per 1M tokens: input, output, cache read. Cache writes (5-minute TTL) cost 1.25x input.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-haiku-4-5": (1.0, 5.0, 0.10),
}

SYSTEM = (
    "You are an autonomous research agent working alone on a self-contained technical task. "
    "You have one tool, bash, which runs a shell command inside your working directory (your sandbox). "
    "Work only inside that directory. Read TASK.md first. Use ./tool to call the task's tools and ./py to run "
    "Python (numpy/scipy/scikit-learn available). There is no network access and no human to ask. "
    "Think about what evidence would actually distinguish the possible answers, gather it efficiently within "
    "the budget, and finish by calling ./tool submit with your answer."
)

BASH_TOOL = {
    "name": "bash",
    "description": (
        "Run a bash command in your sandbox working directory and return stdout+stderr (outputs over 12k chars show "
        "the first and last 6k; the full output is saved to a file under out/ whose path is given). "
        "Each call is a fresh shell starting in the sandbox; files you write persist. Timeout 900 s."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "the bash command to run"}},
        "required": ["command"],
        "additionalProperties": False,
    },
    "strict": True,
}

# Literal references that are refused before execution (defence in depth; the transcript audit is the real check).
_BLOCK = re.compile(
    r"(ideating-rl-tests|/wt/|\.claude|hf_home|\.hf_env|anthropic_env|/opt/pytorch|RLCANARY|rlsbx|/home/|\.\./|\.\.$|"
    r"\bsudo\b|\bsu\b|\bcurl\b|\bwget\b|pip3?\s+install|/proc/|\bdocker\b|\$HOME/\.\.)"
)


def _load_key():
    with open(KEY_FILE) as f:
        for line in f:
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("no ANTHROPIC_API_KEY in ~/.anthropic_env")


def _cost(model, usage):
    pin, pout, pread = PRICES[model]
    inp = getattr(usage, "input_tokens", 0) or 0
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    return (inp * pin + cw * pin * 1.25 + cr * pread + out * pout) / 1e6, dict(input=inp, cache_write=cw, cache_read=cr, output=out)


def _ledger_files():
    import glob
    files = [LEDGER, os.path.join(MAIN_REPO, "runs", "api_budget", "ledger.jsonl")]
    files += sorted(glob.glob(os.path.expanduser("~/wt/*/runs/api_budget/ledger.jsonl")))
    return [f for f in files if os.path.exists(f)]


def _records():
    """Every ledger record from every known ledger file, deduplicated (worktrees copied main's committed lines)."""
    seen, out = set(), []
    for fpath in _ledger_files():
        with open(fpath) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                k = (r.get("t"), r.get("episode"), r.get("turn"), r.get("model"))
                if k not in seen:
                    seen.add(k)
                    out.append(r)
    return out


def _ledger_total():
    return sum(r.get("usd", 0.0) for r in _records())


def _ledger_append(rec):
    os.makedirs(LEDGER_DIR, exist_ok=True)
    with open(LOCK, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with open(LEDGER, "a") as f:
                f.write(json.dumps(rec) + "\n")
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _task_total(task):
    return sum(r.get("usd", 0.0) for r in _records() if r.get("task") == task)


def _task_cap(task):
    """Per-task cap from <main repo>/runs/api_budget/task_caps.json ({"default": x, "<task>": y}); orchestrator-owned."""
    try:
        caps = json.load(open(CAPS_FILE))
    except Exception:
        caps = {}
    return float(caps.get(task, caps.get("default", 1.2)))


def _est_next(model, ctx_tokens, out_guess):
    """Realistic-high next-request cost: prior context read from cache, ~4k new tokens written, out_guess output.
    (The first version charged the whole context as a cache write plus max_tokens of output, ~15x the real turn cost,
    and it stopped a BoolIntermediates probe at $0.28 of a $0.60 cap before the agent could submit.)"""
    pin, pout, pread = PRICES[model]
    return (ctx_tokens * pread + 4000 * pin * 1.25 + out_guess * pout) / 1e6


def _episode_record(episode):
    """The harness's privileged episode record (None if unreadable). The runner reads it, never writes it."""
    try:
        with open(os.path.join(_paths.episodes_dir(), episode + ".json")) as f:
            return json.load(f)
    except Exception:
        return None


def _episode_submitted(episode):
    """Ask the harness's privileged episode record whether a submission was accepted. (The first version grepped the
    command output for '"ok": true', which misfired when an agent chained a REJECTED submit with another tool call.)
    v3 fix 2 (2026-10-04): the runners call this after EVERY command, not only after commands that match
    `./tool submit`: gpt-6.1-sol submitted through `subprocess.check_output(['./tool','submit',...])` inside ./py
    (ShiftHunt openai_T1_sol ep8f9ba96cef, epa024656248), the regex missed it, the runner nudged a finished agent and
    reported stop=ended_without_submit while the grader graded the accepted submission."""
    rec = _episode_record(episode)
    if rec is None:
        return False
    return rec.get("submission") is not None or rec.get("status") in ("submitted", "finished")


def final_stop(episode, loop_stop):
    """The stop reason the runner reports: "submitted" whenever the broker's episode record holds an accepted
    submission (authoritative), whatever the loop saw; otherwise the loop's own reason. A loop that believed it saw a
    submission the record does not hold reports "submit_not_recorded" (should not happen)."""
    if _episode_submitted(episode):
        return "submitted"
    return "submit_not_recorded" if loop_stop == "submitted" else loop_stop


# ---- wall clock (v3 fix 1, 2026-10-04). WHY: ShiftHunt diag_T1_luna_high ep40a2ecda0d ran 6366 s against
# wall_clock_s 3600 with 15 s of compute wait. Neither runner had a clock: the broker refused task tools after the cap,
# but three API requests took 1169 s, 1906 s and 1422 s (600 s SDK request timeouts plus internal retries), and the
# agent submitted 46 minutes late. Now: (a) once the agent time (broker definition: since the first tool call, minus
# compute wait) reaches wall_clock_s, the agent gets TIME_WARN and at most TIME_GRACE_TURNS more turns; (b) the
# runner's agent time (since the runner started, which is before the first tool call, minus the broker's compute wait,
# including a wait still in progress) may not pass the hard ceiling broker.hard_cap(caps): every API request gets a
# timeout that ends at the ceiling, a bash command is stopped when it is reached (re-checked while it runs, so a GPU
# wait during the command extends it), SDK retries are off (_api_call retries itself within the ceiling), and the loop
# stops with "wall_clock_hard". Audit r0 (2026-10-04): the first version capped total time, wait included, which
# would have closed GPU-queued but well-behaved episodes (see the broker docstring); compute wait no longer counts.
TIME_WARN = ("[time notice from the environment] Your time limit for this episode has been reached. Submit your best "
             "answer now with ./tool submit (use the exact answer format in TASK.md); you have at most two more turns.")
TIME_GRACE_TURNS = 2
REQUEST_TIMEOUT_S = 600.0
BASH_TIMEOUT_S = 900
HARD_MARGIN_S = 5.0           # stop this close to the hard ceiling (no request or command could finish in time)


class EpisodeClock:
    """Agent time and runner total time for one episode, with the episode's caps from its record."""

    def __init__(self, episode, t0=None):
        self.episode, self.t0 = episode, time.time() if t0 is None else t0
        rec = _episode_record(episode) or {}
        caps = rec.get("caps") or {}
        self.cap_s = float(caps.get("wall_clock_s", _broker.DEFAULT_CAPS["wall_clock_s"]))
        self.hard_s = _broker.hard_cap(caps)

    def now(self):
        return time.time()

    def total_s(self):
        return self.now() - self.t0

    def agent_s(self):
        rec = _episode_record(self.episode) or {}
        if rec.get("started_at") is None:
            return self.total_s()
        return self.now() - rec["started_at"] - self.wait_s()

    def wait_s(self):
        """The broker's compute wait so far, including a wait still in progress (record field waiting_since)."""
        rec = _episode_record(self.episode) or {}
        w = float(rec.get("wait_s") or 0.0)
        if rec.get("waiting_since"):
            w += max(0.0, self.now() - float(rec["waiting_since"]))
        return w

    def runner_agent_s(self):
        """The runner's view of agent time: runner total minus compute wait. It starts at the runner start, before
        the broker's started_at, so it is never below the broker's agent time (the runner stops first)."""
        return self.total_s() - self.wait_s()

    def hard_left(self):
        return self.hard_s - self.runner_agent_s()

    def expired(self):
        return self.hard_left() <= HARD_MARGIN_S

    def report(self):
        return {"agent_time_s": round(self.agent_s(), 1), "total_time_s": round(self.total_s(), 1),
                "broker_wait_s": round(self.wait_s(), 1), "runner_agent_s": round(self.runner_agent_s(), 1),
                "wall_clock_cap_s": self.cap_s, "wall_clock_hard_s": self.hard_s}


class DeadlineReached(Exception):
    pass


def _api_call(make, clock, classify, max_transient=4):
    """make(timeout) -> response. Retries rate limits (30 s pause) and transient errors (classify(e) == "rate" /
    "transient"; exponential pause) while the hard ceiling allows; raises DeadlineReached when it does not, and
    re-raises anything else (or a transient error after max_transient attempts)."""
    n = 0
    while True:
        left = clock.hard_left()
        if left <= HARD_MARGIN_S:
            raise DeadlineReached()
        try:
            return make(min(REQUEST_TIMEOUT_S, left))
        except Exception as e:
            kind = classify(e)
            if kind is None:
                raise
            n += kind == "transient"
            if n >= max_transient:
                raise
            pause = 30.0 if kind == "rate" else float(2 ** n)
            if clock.hard_left() - pause <= HARD_MARGIN_S:
                raise DeadlineReached()
            time.sleep(pause)


MAX_OUT_CHARS = 12000
SANDBOX_TMP = "tmp"           # = common.sandbox.SANDBOX_TMP (not imported: sandbox imports the whole harness)


PROC_MARK = "RL_AGENT_PROC"   # in the environment of every agent command (and inherited by its children)


def bash_env(sbx, episode=None):
    """The scrubbed environment of an agent command. TMPDIR/TMP/TEMP point at the sandbox's own tmp/ (v3 fix 3):
    Python's tempfile, mktemp and sort write there instead of the shared /tmp, which the audit (rightly) flags.
    PROC_MARK=<episode> lets the runner find and kill the command's processes, even ones that left its process group
    (setsid), when a command times out and when the runner stops (kill_episode_processes)."""
    tmp = os.path.join(sbx, SANDBOX_TMP)
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": sbx, "LANG": "C.UTF-8", "TERM": "dumb",
           "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": tmp, "TMP": tmp, "TEMP": tmp}
    if episode:
        env[PROC_MARK] = episode
    return env


def _marked_pids(episode):
    """PIDs (of this user) whose environment carries PROC_MARK=<episode>. A process that cleared its environment
    (`env -i`) is not found; that is deliberate evasion, not something an agent does by accident."""
    me, needle = os.getpid(), f"{PROC_MARK}={episode}".encode()
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            with open(f"/proc/{d}/environ", "rb") as f:
                if needle in f.read().split(b"\0"):
                    out.append(int(d))
        except OSError:
            pass
    return out


def kill_episode_processes(episode, rounds=5):
    """SIGKILL every process an agent command of this episode left behind (audit r0, 2026-10-04: a timed-out
    command's children, and background jobs, kept running after the runner moved on or stopped, using CPU/RAM on the
    shared box and able to call ./tool late). Repeats while new ones appear (a fork racing the kill). Returns the
    number killed."""
    n = 0
    for _ in range(rounds):
        pids = _marked_pids(episode)
        if not pids:
            break
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
                n += 1
            except OSError:
                pass
        time.sleep(0.05)
    return n


def stop_episode(episode, sync_timeout=30.0):
    """At runner exit: kill the episode's left-over processes, then wait until the broker has finished any call it is
    still processing for the episode (admin "sync" takes the episode lock), so that final_stop reads the record a
    late `setsid ./tool submit &` may have written. A broker without "sync" (started before this fix) is skipped."""
    killed = kill_episode_processes(episode)
    try:
        if _broker.ping():
            _broker.admin("sync", episode=episode, _timeout=sync_timeout)
    except Exception:
        pass
    return killed


def _inside(path, root):
    rp, rr = os.path.realpath(path), os.path.realpath(root)
    return rp == rr or rp.startswith(rr + os.sep)


def _save_full_output(sbx, out):
    """Save an over-long command output to out/cmd_output_<n>.txt in the sandbox; return the relative path or None.
    Never writes outside the sandbox (audit r0: an agent-made symlink out/ -> elsewhere, or a planted
    cmd_output_<n>.txt symlink): out/ must resolve inside the sandbox and the file is created O_EXCL|O_NOFOLLOW."""
    d = os.path.join(sbx, "out")
    try:
        os.makedirs(d, exist_ok=True)
        if os.path.islink(d) or not _inside(d, sbx):
            return None
        n = 1 + sum(1 for f in os.listdir(d) if f.startswith("cmd_output_"))
        while os.path.lexists(os.path.join(d, f"cmd_output_{n}.txt")):
            n += 1
        rel = f"out/cmd_output_{n}.txt"
        fd = os.open(os.path.join(sbx, rel), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(out)
        return rel
    except OSError:
        return None


def truncate_output(out, sbx, max_chars=MAX_OUT_CHARS):
    """Head + tail of an over-long output with an explicit note (v3 fix 4). WHY: the runners silently cut outputs at
    12k chars with only "...[N chars truncated]..." in the middle, so an agent could not tell that a listing or table
    was incomplete, nor get the rest. The full text is now saved in the sandbox and the note names the file."""
    if len(out) <= max_chars:
        return out
    half = max_chars // 2
    rel = _save_full_output(sbx, out)
    where = (f"The full output ({len(out)} chars) is saved in {rel}; read parts of it with head, tail, sed -n or grep."
             if rel else "The full output could not be saved.")
    return (out[:half] + f"\n\n[OUTPUT TRUNCATED by the environment: {len(out) - 2 * half} of {len(out)} chars "
            f"omitted here; showing the first {half} and the last {half}. {where}]\n\n" + out[-half:])


_EXIT_KILL = set()


def _kill_at_exit(episode):
    """Backstop for a runner that dies of an exception before stop_episode: kill the episode's processes at exit."""
    if episode and episode not in _EXIT_KILL:
        _EXIT_KILL.add(episode)
        atexit.register(kill_episode_processes, episode)


def _kill_group(p):
    try:
        os.killpg(p.pid, signal.SIGKILL)      # start_new_session: the group id is bash's pid
    except OSError:
        pass


def _run_bash(cmd, sbx, episode, timeout=BASH_TIMEOUT_S, clock=None, poll_s=0.5):
    """Run one agent command. It is stopped after `timeout` s, or earlier when clock.expired() (the hard ceiling;
    re-checked every poll_s, so compute wait that accrues during the command moves the ceiling). Stopping kills the
    whole process group and every process carrying this episode's PROC_MARK (audit r0: subprocess.run(timeout=)
    killed only `bash -c`; its children kept running)."""
    # The only allowed absolute reference into /home is this episode's own sandbox.
    stripped = cmd.replace(sbx + "/", "./").replace(sbx, ".")
    if _BLOCK.search(stripped):
        return "blocked: this command references something outside your sandbox or a disallowed operation.", True
    os.makedirs(os.path.join(sbx, SANDBOX_TMP), exist_ok=True)     # episodes prepared before the v3 fix have none
    _kill_at_exit(episode)
    t0 = time.time()
    p = subprocess.Popen(["bash", "-c", cmd], cwd=sbx, env=bash_env(sbx, episode), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, errors="replace", start_new_session=True)
    why = None
    while True:
        try:
            so, se = p.communicate(timeout=poll_s)
            break
        except subprocess.TimeoutExpired:
            if time.time() - t0 >= timeout:
                why = f"[timed out after {int(round(time.time() - t0))} s]"
            elif clock is not None and clock.expired():
                why = (f"[timed out after {int(round(time.time() - t0))} s: the episode's hard time limit "
                       f"was reached]")
            if why:
                _kill_group(p)
                kill_episode_processes(episode)
                try:
                    p.communicate(timeout=10)       # reap; output of a killed command is dropped
                except subprocess.TimeoutExpired:   # a grandchild outside the group still holds the pipe
                    pass
                break
    if why:
        out = why
    else:
        out = (so or "") + (("\n[stderr]\n" + se) if se else "")
        if p.returncode:
            out += f"\n[exit code {p.returncode}]"
    return truncate_output(out, sbx) or "[no output]", False


def _anthropic_classify(anthropic):
    def classify(e):
        if isinstance(e, anthropic.RateLimitError):
            return "rate"
        if isinstance(e, anthropic.APIConnectionError):          # includes APITimeoutError
            return "transient"
        if isinstance(e, anthropic.APIStatusError) and (e.status_code >= 500 or e.status_code in (408, 409, 529)):
            return "transient"
        return None
    return classify


def run(args, client=None):
    """One episode. `client`: an Anthropic client (tests pass a fake); default: a real one with SDK retries off."""
    import anthropic

    model = args.model
    if not args.task:
        raise SystemExit("--task is required (per-task API caps are enforced)")
    if model not in PRICES:
        raise SystemExit(f"unknown model {model}; known: {list(PRICES)}")
    try:
        allowed = json.load(open(CAPS_FILE)).get("allowed_models")
    except Exception:
        allowed = None
    if allowed and model not in allowed and not args.allow_any_model:
        raise SystemExit(f"model {model} is not allowed for probes (Dan, 2026-10-01: Opus is too expensive). "
                         f"Use one of {allowed}, e.g. --model claude-sonnet-5-5 (default) or claude-haiku-4-5.")
    sbx = os.path.join(_paths.sandbox_root(), args.episode)     # read now: RL_SANDBOX_ROOT may be set late
    if not os.path.isdir(sbx):
        raise SystemExit(f"sandbox {sbx} not found (run common.sandbox prepare first)")
    prompt = open(args.prompt_file).read()
    os.makedirs(args.out, exist_ok=True)
    tpath = os.path.join(args.out, "api_transcript.jsonl")
    mpath = os.path.join(args.out, "api_meta.json")
    if client is None:
        client = anthropic.Anthropic(api_key=_load_key(), max_retries=0)    # _api_call retries within the ceiling
    classify = _anthropic_classify(anthropic)

    messages = [{"role": "user", "content": prompt}]
    tfile = open(tpath, "w")

    def tlog(entry):
        entry["episode"] = args.episode
        entry.setdefault("t", round(time.time(), 3))
        tfile.write(json.dumps(entry, default=str) + "\n")
        tfile.flush()

    tlog({"type": "user", "message": {"role": "user", "content": prompt}})
    spent, turns, stop, ctx_tokens, blocked = 0.0, 0, "max_turns", len(prompt) // 3, 0
    usage_tot = dict(input=0, cache_write=0, cache_read=0, output=0)
    clock = EpisodeClock(args.episode)
    submitted = False
    nudged = False
    warned = False
    time_warned_at = None          # turn count when TIME_WARN was sent
    max_out_seen = 0
    WARN = ("[budget notice from the environment] You are close to this episode's compute limit. Submit your best "
            "answer now with ./tool submit (use the exact answer format in TASK.md); you have at most two more turns.")
    while turns < args.max_turns + (2 if warned else 0):
        if clock.expired():
            stop = "wall_clock_hard"
            break
        if time_warned_at is not None and turns >= time_warned_at + TIME_GRACE_TURNS:
            stop = "wall_clock"
            break
        out_guess = min(args.max_tokens, max(3000, int(1.5 * max_out_seen)))
        est = _est_next(model, ctx_tokens, out_guess)
        if _ledger_total() + est > GLOBAL_CAP_USD:
            stop = "global_budget"
            break
        if _task_total(args.task) + est > _task_cap(args.task):
            stop = "task_budget"
            break
        if spent + est > args.max_usd * (1.15 if warned else 1.0):
            stop = "episode_budget"
            break
        kw = dict(model=model, max_tokens=args.max_tokens, system=SYSTEM, tools=[BASH_TOOL], messages=messages,
                  cache_control={"type": "ephemeral"})
        if model.startswith("claude-haiku"):
            pass  # Haiku 4.5: no effort parameter; run without extended thinking.
        else:
            kw["output_config"] = {"effort": args.effort}
        try:
            resp = _api_call(lambda timeout: client.messages.create(timeout=timeout, **kw), clock, classify)
        except DeadlineReached:
            stop = "wall_clock_hard"
            break
        except anthropic.RateLimitError as e:
            stop = "api_rate_limited"
            tlog({"type": "error", "error": str(e)[:500]})
            break
        except anthropic.APIStatusError as e:
            stop = f"api_error_{e.status_code}"
            tlog({"type": "error", "error": str(e)[:500]})
            break
        except anthropic.APIConnectionError as e:
            stop = "api_connection_error"
            tlog({"type": "error", "error": str(e)[:500]})
            break
        turns += 1
        usd, u = _cost(model, resp.usage)
        spent += usd
        for k in usage_tot:
            usage_tot[k] += u[k]
        ctx_tokens = u["input"] + u["cache_write"] + u["cache_read"] + u["output"]
        max_out_seen = max(max_out_seen, u["output"])
        _ledger_append({"t": time.time(), "episode": args.episode, "task": args.task, "model": model,
                        "effort": args.effort, "turn": turns, "usd": round(usd, 6), **u})
        content = [b.model_dump() for b in resp.content]
        tlog({"type": "assistant", "cwd": sbx, "message": {"role": "assistant", "model": model, "content": content},
              "usage": u, "usd": usd})
        if resp.stop_reason == "refusal":
            stop = "refusal"
            tlog({"type": "refusal", "stop_details": getattr(resp, "stop_details", None)})
            break
        messages.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            if submitted or _episode_submitted(args.episode):
                stop = "submitted"
                break
            if nudged:
                stop = "ended_without_submit"
                break
            nudged = True
            nudge = "You have not submitted yet. Continue working, or call ./tool submit with your final answer."
            messages.append({"role": "user", "content": nudge})
            tlog({"type": "user", "message": {"role": "user", "content": nudge}})
            continue
        results = []
        for b in uses:
            cmd = (b.input or {}).get("command", "")
            if clock.expired():
                out, was_blocked = "[not run: the episode's hard time limit has been reached]", False
            else:
                out, was_blocked = _run_bash(cmd, sbx, args.episode, timeout=BASH_TIMEOUT_S, clock=clock)
            blocked += int(was_blocked)
            if _episode_submitted(args.episode):      # any route: ./tool submit, ./py + subprocess, ... (fix 2)
                submitted = True
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            tlog({"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": b.id, "content": out}]}, "blocked": was_blocked})
        if not submitted and not warned and (spent + 3 * est > args.max_usd or turns >= args.max_turns - 2):
            warned = True
            results.append({"type": "text", "text": WARN})
            tlog({"type": "user", "message": {"role": "user", "content": WARN}, "budget_warning": True})
        if not submitted and time_warned_at is None and clock.agent_s() >= clock.cap_s:
            time_warned_at = turns
            results.append({"type": "text", "text": TIME_WARN})
            tlog({"type": "user", "message": {"role": "user", "content": TIME_WARN}, "time_warning": True})
        messages.append({"role": "user", "content": results})
        if submitted:
            stop = "submitted"
            break
    tfile.close()
    killed = stop_episode(args.episode)       # before final_stop: a detached late submit has landed or is dead
    meta = {"episode": args.episode, "task": args.task, "model": model, "effort": args.effort, "turns": turns,
            "stop": final_stop(args.episode, stop), "runner_stop": stop, "budget_warned": warned,
            "time_warned": time_warned_at is not None, "usd": round(spent, 4), "usage": usage_tot,
            "blocked_commands": blocked, "wall_s": round(clock.total_s(), 1), **clock.report(),
            "killed_processes": killed, "ledger_total_usd": round(_ledger_total(), 4),
            "global_cap_usd": GLOBAL_CAP_USD, "anthropic_sdk": anthropic.__version__}
    json.dump(meta, open(mpath, "w"), indent=1)
    print(json.dumps(meta))
    return meta


def spent(args):
    recs = sorted(_records(), key=lambda r: r.get("t", 0))
    by = {}
    for r in recs:
        for k in (("model", r.get("model")), ("task", r.get("task") or "?")):
            by[k] = by.get(k, 0.0) + r.get("usd", 0.0)
    caps = {}
    try:
        caps = json.load(open(CAPS_FILE))
    except Exception:
        pass
    if getattr(args, "snapshot", False):
        with open(SNAPSHOT, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    print(json.dumps({"total_usd": round(sum(r.get("usd", 0.0) for r in recs), 4), "cap_usd": GLOBAL_CAP_USD,
                      "task_caps": caps, "n_requests": len(recs), "ledger_files": _ledger_files(),
                      "breakdown": {f"{a}={b}": round(v, 4) for (a, b), v in sorted(by.items(), key=str)}}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--episode", required=True)
    r.add_argument("--prompt-file", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--task", default="")
    r.add_argument("--model", default="claude-sonnet-5-5")
    r.add_argument("--effort", default="medium")
    r.add_argument("--allow-any-model", action="store_true", help="orchestrator only: bypass allowed_models")
    r.add_argument("--max-turns", type=int, default=40)
    r.add_argument("--max-tokens", type=int, default=12000)
    r.add_argument("--max-usd", type=float, default=1.25)
    sp = sub.add_parser("spent")
    sp.add_argument("--snapshot", action="store_true", help="also write the deduplicated ledger into the main repo")
    a = ap.parse_args()
    {"run": run, "spent": spent}[a.cmd](a)


if __name__ == "__main__":
    main()
