"""Budget-capped OpenAI test agent for one harness episode (twin of common/api_agent.py).

Why: Dan added an OpenAI key on 2026-10-01 with a $60 budget ("use when necessary, no need to ration").
A second model family makes difficulty measurements less specific to Claude, and gpt-6-luna is a very cheap
"small model" for the capability-separation check. Same containment as api_agent: one `bash` tool that runs
only inside the episode sandbox with a scrubbed environment (HOME = sandbox, no keys), the same command guard,
and a Claude-Code-shaped transcript, so common/transcript_audit.py applies unchanged.

Budget (hard rules): every response's list-price cost goes to ~/.rl_api/openai_ledger.jsonl (outside every
checkout). There is a global stop at OPENAI_GLOBAL_CAP_USD (default 57.0 of Dan's $60), a per-task cap
(runs/api_budget/task_caps.json -> "openai": {"default": x, "<task>": y}), and per-episode --max-usd / --max-turns
caps. The agent gets a "submit now" notice before any budget stop.

Uses the Responses API: function tools plus reasoning effort are rejected on /v1/chat/completions for these models.
Turns are chained with previous_response_id, so only new tool outputs are sent each turn.

Usage:
    python -m common.openai_agent run --episode E --task T --prompt-file <run>/episodes/E/agent_prompt.txt \
        --out <run>/episodes/E --model gpt-6.1-sol --effort medium
    python -m common.sandbox finish --episode E --transcript <run>/episodes/E/api_transcript.jsonl \
        --agent-model openai:gpt-6.1-sol:medium
    python -m common.openai_agent spent [--snapshot]
"""
import argparse
import fcntl
import json
import os
import time

from common.api_agent import (SYSTEM, _episode_submitted, _run_bash, MAIN_REPO, CAPS_FILE, TIME_WARN,
                              TIME_GRACE_TURNS, BASH_TIMEOUT_S, EpisodeClock, DeadlineReached, _api_call, final_stop,
                              _paths)

LEDGER_DIR = os.path.expanduser(os.environ.get("RL_API_LEDGER_DIR", "~/.rl_api"))
LEDGER = os.path.join(LEDGER_DIR, "openai_ledger.jsonl")
LOCK = os.path.join(LEDGER_DIR, ".openai_lock")
SNAPSHOT = os.path.join(MAIN_REPO, "runs", "api_budget", "openai_ledger_snapshot.jsonl")
GLOBAL_CAP_USD = float(os.environ.get("OPENAI_GLOBAL_CAP_USD", "57.0"))
KEY_FILE = os.path.expanduser("~/.openai_env")

# $ per 1M tokens (short context <= 272K input): input, cached input, output.
# Source: https://developers.openai.com/api/docs/pricing, fetched 2026-10-01.
PRICES = {
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6-sol": (2.00, 0.20, 10.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
}

BASH_TOOL = {
    "type": "function",
    "name": "bash",
    "description": ("Run a bash command in your sandbox working directory and return stdout+stderr (outputs over "
                    "12k chars show the first and last 6k; the full output is saved to a file under out/ whose path "
                    "is given). Each call is a fresh shell starting in the sandbox; files you write persist. "
                    "Timeout 900 s."),
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "the bash command"}},
                   "required": ["command"], "additionalProperties": False},
    "strict": True,
}

WARN = ("[budget notice from the environment] You are close to this episode's compute limit. Submit your best "
        "answer now with ./tool submit (use the exact answer format in TASK.md); you have at most two more turns.")


def _load_key():
    with open(KEY_FILE) as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("no OPENAI_API_KEY in ~/.openai_env")


def _cost(model, usage):
    pin, pcache, pout = PRICES[model]
    inp = getattr(usage, "input_tokens", 0) or 0
    det = getattr(usage, "input_tokens_details", None)
    cached = (getattr(det, "cached_tokens", 0) or 0) if det is not None else 0
    out = getattr(usage, "output_tokens", 0) or 0
    odet = getattr(usage, "output_tokens_details", None)
    reasoning = (getattr(odet, "reasoning_tokens", 0) or 0) if odet is not None else 0
    usd = ((inp - cached) * pin + cached * pcache + out * pout) / 1e6
    return usd, dict(input=inp, cached=cached, output=out, reasoning=reasoning)


def _records():
    out = []
    if os.path.exists(LEDGER):
        with open(LEDGER) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def _total(task=None):
    return sum(r.get("usd", 0.0) for r in _records() if task is None or r.get("task") == task)


def _append(rec):
    os.makedirs(LEDGER_DIR, exist_ok=True)
    with open(LOCK, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with open(LEDGER, "a") as f:
                f.write(json.dumps(rec) + "\n")
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _task_cap(task):
    try:
        caps = json.load(open(CAPS_FILE)).get("openai", {})
    except Exception:
        caps = {}
    return float(caps.get(task, caps.get("default", 4.0)))


def _est_next(model, ctx_tokens, out_guess):
    pin, pcache, pout = PRICES[model]
    return (ctx_tokens * pcache + 4000 * pin + out_guess * pout) / 1e6


def _openai_classify(openai):
    def classify(e):
        if isinstance(e, openai.RateLimitError):
            return "rate"
        if isinstance(e, openai.APIConnectionError):             # includes APITimeoutError
            return "transient"
        if isinstance(e, openai.APIStatusError) and (e.status_code >= 500 or e.status_code in (408, 409)):
            return "transient"
        return None
    return classify


def run(args, client=None):
    """One episode. `client`: an OpenAI client (tests pass a fake); default: a real one with SDK retries off (the
    600 s request timeout times 4 SDK attempts is how ep40a2ecda0d spent 1906 s on one turn; _api_call retries
    within the episode's absolute time limit instead)."""
    import openai

    model = args.model
    if model not in PRICES:
        raise SystemExit(f"unknown model {model}; known: {list(PRICES)}")
    if not args.task:
        raise SystemExit("--task is required")
    sbx = os.path.join(_paths.sandbox_root(), args.episode)     # read now: RL_SANDBOX_ROOT may be set late
    if not os.path.isdir(sbx):
        raise SystemExit(f"sandbox {sbx} not found (run common.sandbox prepare first)")
    prompt = open(args.prompt_file).read()
    os.makedirs(args.out, exist_ok=True)
    tpath = os.path.join(args.out, "api_transcript.jsonl")
    if client is None:
        client = openai.OpenAI(api_key=_load_key(), max_retries=0)
    classify = _openai_classify(openai)
    tfile = open(tpath, "w")

    def tlog(entry):
        entry["episode"] = args.episode
        entry.setdefault("t", round(time.time(), 3))
        tfile.write(json.dumps(entry, default=str) + "\n")
        tfile.flush()

    tlog({"type": "user", "message": {"role": "user", "content": prompt}})
    next_input = [{"role": "user", "content": prompt}]
    prev_id = None
    spent, turns, stop, ctx, blocked, max_out = 0.0, 0, "max_turns", len(prompt) // 3, 0, 0
    usage_tot = dict(input=0, cached=0, output=0, reasoning=0)
    submitted = nudged = warned = False
    time_warned_at = None
    clock = EpisodeClock(args.episode)
    while turns < args.max_turns + (2 if warned else 0):
        if clock.expired():
            stop = "wall_clock_hard"
            break
        if time_warned_at is not None and turns >= time_warned_at + TIME_GRACE_TURNS:
            stop = "wall_clock"
            break
        est = _est_next(model, ctx, min(args.max_tokens, max(3000, int(1.5 * max_out))))
        if _total() + est > GLOBAL_CAP_USD:
            stop = "global_budget"
            break
        if _total(args.task) + est > _task_cap(args.task):
            stop = "task_budget"
            break
        if spent + est > args.max_usd * (1.15 if warned else 1.0):
            stop = "episode_budget"
            break
        kw = dict(model=model, instructions=SYSTEM, input=next_input, tools=[BASH_TOOL],
                  max_output_tokens=args.max_tokens, reasoning={"effort": args.effort})
        if prev_id:
            kw["previous_response_id"] = prev_id
        try:
            r = _api_call(lambda timeout: client.responses.create(timeout=timeout, **kw), clock, classify)
        except DeadlineReached:
            stop = "wall_clock_hard"
            break
        except openai.RateLimitError as e:
            stop = "api_rate_limited"
            tlog({"type": "error", "error": str(e)[:500]})
            break
        except openai.APIStatusError as e:
            stop = f"api_error_{e.status_code}"
            tlog({"type": "error", "error": str(e)[:500]})
            break
        except openai.APIConnectionError as e:
            stop = "api_connection_error"
            tlog({"type": "error", "error": str(e)[:500]})
            break
        turns += 1
        prev_id = r.id
        usd, u = _cost(model, r.usage)
        spent += usd
        for k in usage_tot:
            usage_tot[k] += u[k]
        ctx = u["input"] + u["output"]
        max_out = max(max_out, u["output"])
        _append({"t": time.time(), "episode": args.episode, "task": args.task, "model": model, "effort": args.effort,
                 "turn": turns, "usd": round(usd, 6), **u})
        calls = [o for o in r.output if o.type == "function_call"]
        text = r.output_text or ""
        content = ([{"type": "text", "text": text}] if text.strip() else []) + [
            {"type": "tool_use", "id": c.call_id, "name": "Bash",
             "input": json.loads(c.arguments) if c.arguments else {}} for c in calls]
        tlog({"type": "assistant", "cwd": sbx, "message": {"role": "assistant", "model": model, "content": content},
              "usage": u, "usd": usd, "status": r.status})
        if not calls:
            if submitted or _episode_submitted(args.episode):
                stop = "submitted"
                break
            if nudged:
                stop = "ended_without_submit"
                break
            nudged = True
            nudge = "You have not submitted yet. Continue working, or call ./tool submit with your final answer."
            next_input = [{"role": "user", "content": nudge}]
            tlog({"type": "user", "message": {"role": "user", "content": nudge}})
            continue
        next_input = []
        for c in calls:
            try:
                cmd = json.loads(c.arguments).get("command", "")
            except Exception:
                cmd = ""
            left = clock.hard_left()
            if clock.expired():
                out, was_blocked = "[not run: the episode's absolute time limit has been reached]", False
            else:
                out, was_blocked = _run_bash(cmd, sbx, args.episode, timeout=min(BASH_TIMEOUT_S, left))
            blocked += int(was_blocked)
            if _episode_submitted(args.episode):      # any route: ./tool submit, ./py + subprocess, ... (fix 2)
                submitted = True
            next_input.append({"type": "function_call_output", "call_id": c.call_id, "output": out})
            tlog({"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": c.call_id, "content": out}]}, "blocked": was_blocked})
        if submitted:
            stop = "submitted"
            break
        if not warned and (spent + 3 * est > args.max_usd or turns >= args.max_turns - 2):
            warned = True
            next_input.append({"role": "user", "content": WARN})
            tlog({"type": "user", "message": {"role": "user", "content": WARN}, "budget_warning": True})
        if time_warned_at is None and clock.agent_s() >= clock.cap_s:
            time_warned_at = turns
            next_input.append({"role": "user", "content": TIME_WARN})
            tlog({"type": "user", "message": {"role": "user", "content": TIME_WARN}, "time_warning": True})
    tfile.close()
    meta = {"episode": args.episode, "task": args.task, "provider": "openai", "model": model, "effort": args.effort,
            "turns": turns, "stop": final_stop(args.episode, stop), "runner_stop": stop, "budget_warned": warned,
            "time_warned": time_warned_at is not None, "usd": round(spent, 4), "usage": usage_tot,
            "blocked_commands": blocked, "wall_s": round(clock.total_s(), 1), **clock.report(),
            "openai_ledger_total_usd": round(_total(), 4), "global_cap_usd": GLOBAL_CAP_USD,
            "openai_sdk": openai.__version__}
    json.dump(meta, open(os.path.join(args.out, "api_meta.json"), "w"), indent=1)
    print(json.dumps(meta))
    return meta


def spent(args):
    recs = sorted(_records(), key=lambda r: r.get("t", 0))
    by = {}
    for r in recs:
        for k in (("model", r.get("model")), ("task", r.get("task"))):
            by[k] = by.get(k, 0.0) + r.get("usd", 0.0)
    if args.snapshot:
        with open(SNAPSHOT, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    print(json.dumps({"total_usd": round(_total(), 4), "cap_usd": GLOBAL_CAP_USD, "n_requests": len(recs),
                      "breakdown": {f"{a}={b}": round(v, 4) for (a, b), v in sorted(by.items(), key=str)}}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--episode", required=True)
    r.add_argument("--prompt-file", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--task", required=True)
    r.add_argument("--model", default="gpt-6.1-sol")
    r.add_argument("--effort", default="medium")
    r.add_argument("--max-turns", type=int, default=40)
    r.add_argument("--max-tokens", type=int, default=16000)
    r.add_argument("--max-usd", type=float, default=0.5)
    sp = sub.add_parser("spent")
    sp.add_argument("--snapshot", action="store_true")
    a = ap.parse_args()
    {"run": run, "spent": spent}[a.cmd](a)


if __name__ == "__main__":
    main()
