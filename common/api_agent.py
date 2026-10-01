"""Budget-capped Claude API test agent for one harness episode (fast difficulty probes).

Why this exists: Dan granted a TOTAL Anthropic API budget of $17 (2026-10-01) to
measure task difficulty early, with fast feedback loops, before a lot gets built.
Fresh Claude Code subagents stay the free, high-volume option. The API agent adds
exact model control, a clean context (no Claude Code system prompt) and tighter
containment. The model can only run shell commands *inside its sandbox*, with a
scrubbed environment (no HF token, no API key, HOME = sandbox).

Budget safety (hard rules):
  * Every request's cost is computed from response.usage at list prices and
    appended to runs/api_budget/ledger.jsonl (flock-guarded). Committed to git,
    so every dollar traces to an episode.
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
import fcntl
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER_DIR = os.path.join(REPO, "runs", "api_budget")
LEDGER = os.path.join(LEDGER_DIR, "ledger.jsonl")
LOCK = os.path.join(LEDGER_DIR, ".lock")
GLOBAL_CAP_USD = float(os.environ.get("API_GLOBAL_CAP_USD", "16.0"))
KEY_FILE = os.path.expanduser("~/.anthropic_env")
SBX_ROOT = os.path.expanduser("~/rlsbx")

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
        "Run a bash command in your sandbox working directory and return stdout+stderr (truncated to ~12k chars). "
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


def _ledger_total():
    tot = 0.0
    if os.path.exists(LEDGER):
        with open(LEDGER) as f:
            for line in f:
                try:
                    tot += json.loads(line)["usd"]
                except Exception:
                    pass
    return tot


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
    tot = 0.0
    if os.path.exists(LEDGER):
        with open(LEDGER) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("task") == task:
                    tot += r["usd"]
    return tot


def _task_cap(task):
    """Per-task cap from runs/api_budget/task_caps.json ({"default": x, "<task>": y}); orchestrator-owned."""
    try:
        caps = json.load(open(os.path.join(LEDGER_DIR, "task_caps.json")))
    except Exception:
        caps = {}
    return float(caps.get(task, caps.get("default", 1.2)))


def _est_next(model, ctx_tokens, max_out):
    """Pessimistic next-request cost: whole context as a cache write, plus max_out output tokens."""
    pin, pout, _ = PRICES[model]
    return (ctx_tokens * pin * 1.25 + max_out * pout) / 1e6


def _run_bash(cmd, sbx, episode):
    # The only allowed absolute reference into /home is this episode's own sandbox.
    stripped = cmd.replace(sbx + "/", "./").replace(sbx, ".")
    if _BLOCK.search(stripped):
        return "blocked: this command references something outside your sandbox or a disallowed operation.", True
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": sbx, "LANG": "C.UTF-8", "TERM": "dumb",
           "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        p = subprocess.run(["bash", "-c", cmd], cwd=sbx, env=env, capture_output=True, text=True, timeout=900)
        out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
        if p.returncode:
            out += f"\n[exit code {p.returncode}]"
    except subprocess.TimeoutExpired:
        out = "[timed out after 900 s]"
    if len(out) > 12000:
        out = out[:6000] + f"\n...[{len(out) - 12000} chars truncated]...\n" + out[-6000:]
    return out or "[no output]", False


def run(args):
    import anthropic

    model = args.model
    if not args.task:
        raise SystemExit("--task is required (per-task API caps are enforced)")
    if model not in PRICES:
        raise SystemExit(f"unknown model {model}; known: {list(PRICES)}")
    sbx = os.path.join(SBX_ROOT, args.episode)
    if not os.path.isdir(sbx):
        raise SystemExit(f"sandbox {sbx} not found (run common.sandbox prepare first)")
    prompt = open(args.prompt_file).read()
    os.makedirs(args.out, exist_ok=True)
    tpath = os.path.join(args.out, "api_transcript.jsonl")
    mpath = os.path.join(args.out, "api_meta.json")
    client = anthropic.Anthropic(api_key=_load_key(), max_retries=3)

    messages = [{"role": "user", "content": prompt}]
    tfile = open(tpath, "w")

    def tlog(entry):
        entry["episode"] = args.episode
        tfile.write(json.dumps(entry, default=str) + "\n")
        tfile.flush()

    tlog({"type": "user", "message": {"role": "user", "content": prompt}})
    spent, turns, stop, ctx_tokens, blocked = 0.0, 0, "max_turns", len(prompt) // 3, 0
    usage_tot = dict(input=0, cache_write=0, cache_read=0, output=0)
    t0 = time.time()
    submitted = False
    nudged = False
    while turns < args.max_turns:
        est = _est_next(model, ctx_tokens + 2000, args.max_tokens)
        if _ledger_total() + est > GLOBAL_CAP_USD:
            stop = "global_budget"
            break
        if _task_total(args.task) + est > _task_cap(args.task):
            stop = "task_budget"
            break
        if spent + est > args.max_usd:
            stop = "episode_budget"
            break
        kw = dict(model=model, max_tokens=args.max_tokens, system=SYSTEM, tools=[BASH_TOOL], messages=messages,
                  cache_control={"type": "ephemeral"})
        if model.startswith("claude-haiku"):
            pass  # Haiku 4.5: no effort parameter; run without extended thinking.
        else:
            kw["output_config"] = {"effort": args.effort}
        try:
            resp = client.messages.create(**kw)
        except anthropic.RateLimitError:
            time.sleep(30)
            continue
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
            if submitted:
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
            out, was_blocked = _run_bash(cmd, sbx, args.episode)
            blocked += int(was_blocked)
            if re.search(r"\./tool\s+submit\b", cmd) and '"ok": true' in out.replace("'", '"'):
                submitted = True
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            tlog({"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": b.id, "content": out}]}, "blocked": was_blocked})
        messages.append({"role": "user", "content": results})
        if submitted:
            stop = "submitted"
            break
    tfile.close()
    meta = {"episode": args.episode, "task": args.task, "model": model, "effort": args.effort, "turns": turns,
            "stop": stop, "usd": round(spent, 4), "usage": usage_tot, "blocked_commands": blocked,
            "wall_s": round(time.time() - t0, 1), "ledger_total_usd": round(_ledger_total(), 4),
            "global_cap_usd": GLOBAL_CAP_USD, "anthropic_sdk": anthropic.__version__}
    json.dump(meta, open(mpath, "w"), indent=1)
    print(json.dumps(meta))


def spent(_args):
    by = {}
    if os.path.exists(LEDGER):
        for line in open(LEDGER):
            r = json.loads(line)
            for k in (("model", r["model"]), ("task", r.get("task") or "?")):
                by.setdefault(k, 0.0)
                by[k] += r["usd"]
    caps = {}
    try:
        caps = json.load(open(os.path.join(LEDGER_DIR, "task_caps.json")))
    except Exception:
        pass
    print(json.dumps({"total_usd": round(_ledger_total(), 4), "cap_usd": GLOBAL_CAP_USD, "task_caps": caps,
                      "breakdown": {f"{a}={b}": round(v, 4) for (a, b), v in sorted(by.items())}}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--episode", required=True)
    r.add_argument("--prompt-file", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--task", default="")
    r.add_argument("--model", default="claude-opus-5-5")
    r.add_argument("--effort", default="medium")
    r.add_argument("--max-turns", type=int, default=40)
    r.add_argument("--max-tokens", type=int, default=12000)
    r.add_argument("--max-usd", type=float, default=1.25)
    sub.add_parser("spent")
    a = ap.parse_args()
    {"run": run, "spent": spent}[a.cmd](a)


if __name__ == "__main__":
    main()
