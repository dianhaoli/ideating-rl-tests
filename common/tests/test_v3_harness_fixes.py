"""v3 harness fix-firsts (2026-10-04), from the ShiftHunt v2 evidence (docs/forensics/SHIFTHUNT_FORENSICS.md,
docs/results/shifthunt/*). One section per fix:
 1. wall clock: a nominal cap on agent time plus an absolute ceiling on total time, enforced by the broker AND both
    runners, recorded in the episode record / grade.json / summaries (luna-high ep40a2ecda0d ran 6366 s vs 3600 s);
 2. the runners' stop reason comes from the broker's episode record (sol submitted through subprocess and the runner
    said ended_without_submit);
 3. a sandbox-local TMPDIR (5 Haiku episodes INVALID for `./tool ... > /tmp/x.json`); /tmp itself stays flagged;
 4. over-long command output is saved in the sandbox and shown head + tail with an explicit note;
 5. audit false positive `print(s.replace('\\n', '/'))` (openai_T2_sol ep2c6190e0f5), with twins that stay flagged.
The runner tests drive the real run loops with fake API clients against the isolated broker (no network, no spend).
"""
import json
import os
import time
import types

import pytest

from common import api_agent, broker, openai_agent, sandbox
from common import transcript_audit as ta
from common.tests.conftest import edited_instance
from common.tests.test_verify_audit import C, SBX, audit, bash
from common.toolclient import Client, ToolCallError


def _record(eid):
    with open(broker.record_path(eid)) as f:
        return json.load(f)


def _prep(hx, inst, label="v3"):
    ep = sandbox.prepare("_demo", inst, "full", hx.run_dir, solver_label=label)
    return ep, Client(ep["episode"])


# =============================================================================== fix 1: wall clock (broker side)
def test_hard_cap_default_and_explicit():
    assert broker.hard_cap({"wall_clock_s": 3600}) == 5400
    assert broker.hard_cap({"wall_clock_s": 10}) == 70                 # at least 60 s of grace to submit
    assert broker.hard_cap({"wall_clock_s": 3600, "wall_clock_hard_s": 4000}) == 4000
    assert broker.hard_cap({}) == 5400                                  # records without caps: the defaults
    rec = {"caps": {"wall_clock_s": 100}, "started_at": 1000.0, "wait_s": 30.0, "submitted_at": 1150.0}
    t = broker.episode_times(rec)
    assert (t["agent_s"], t["total_s"], t["over_nominal"], t["over_hard"]) == (120.0, 150.0, True, False)
    assert broker.episode_times({"caps": {}, "started_at": None})["total_s"] == 0.0


def test_prepare_records_hard_cap_and_task_md_states_it(hx):
    ep, _ = _prep(hx, hx.insts["null"][0])
    rec = _record(ep["episode"])
    assert rec["caps"]["wall_clock_hard_s"] == broker.hard_cap(rec["caps"])
    task_md = open(os.path.join(ep["sandbox"], "TASK.md")).read()
    assert "absolute time limit" in task_md and "including any waiting" in task_md
    assert "wall_clock_hard_s" not in task_md                           # not a counter line


def test_submit_after_nominal_cap_is_accepted_and_marked(hx):
    ep, c = _prep(hx, edited_instance(hx.insts["null"][0], hx.root, wall_clock_s=1, wall_clock_hard_s=30))
    c.call("weights")
    time.sleep(1.3)
    with pytest.raises(ToolCallError, match="time limit"):
        c.call("weights")
    b = c.call("budget")
    assert b["wall_clock_s"]["remaining"] == 0 and b["wall_clock_hard_s"]["cap"] == 30
    assert c.submit({"nothing_found": True})["ok"]
    rec = _record(ep["episode"])
    assert rec["time"]["over_nominal"] and not rec["time"]["over_hard"] and rec["time"]["agent_s"] >= 1.0
    g = sandbox.finish(ep["episode"], solver_rc=0)
    h = g["harness"]
    assert h["over_time"] == "nominal" and h["time"]["agent_s"] >= 1.0 and h["time"]["hard_cap_s"] == 30
    assert h["valid"]                                                   # visible, not invalid


def test_hard_ceiling_refuses_tools_and_submit_and_closes(hx):
    ep, c = _prep(hx, edited_instance(hx.insts["null"][0], hx.root, wall_clock_s=1, wall_clock_hard_s=2))
    c.call("weights")
    time.sleep(2.3)
    with pytest.raises(ToolCallError, match="hard time limit"):
        c.call("weights")
    r = c.submit({"nothing_found": True})
    assert not r["ok"]
    rec = _record(ep["episode"])
    assert rec["status"] == "closed" and rec["close_reason"] == "wall_clock_hard" and rec["submission"] is None
    assert rec["time"]["over_hard"] and rec["time"]["total_s"] >= 2.0
    closed_at = rec["closed_at"]
    g = sandbox.finish(ep["episode"], solver_rc=0)
    assert _record(ep["episode"])["closed_at"] == closed_at          # finish keeps the time the ceiling closed it
    h = g["harness"]
    assert h["over_time"] == "hard" and h["time"]["close_reason"] == "wall_clock_hard" and not h["submitted"]
    log = [json.loads(l) for l in open(os.path.join(hx.run_dir, "episodes", ep["episode"], "tool_log.jsonl"))]
    assert any(e.get("event") == "wall_clock_hard" for e in log)


def test_unsubmitted_time_ends_at_last_activity_not_at_finish(hx):
    """An episode finished long after the agent stopped must not read as over time."""
    ep, c = _prep(hx, edited_instance(hx.insts["null"][0], hx.root, wall_clock_s=1, wall_clock_hard_s=2))
    c.call("weights")
    time.sleep(2.3)                        # finish runs after the ceiling, but the agent was last seen at ~0 s
    h = sandbox.finish(ep["episode"], solver_rc=0)["harness"]
    assert h["over_time"] is None and h["time"]["total_s"] < 1.0


def test_summary_shows_time_exceedance(hx):
    run_dir = os.path.join(hx.root, "runs_v3_time")
    ep = sandbox.prepare("_demo", edited_instance(hx.insts["null"][0], hx.root, wall_clock_s=1, wall_clock_hard_s=30),
                         "full", run_dir, solver_label="t")
    c = Client(ep["episode"])
    c.call("weights")
    time.sleep(1.3)
    c.submit({"nothing_found": True})
    sandbox.finish(ep["episode"], solver_rc=0)
    ep2 = sandbox.prepare("_demo", hx.insts["null"][0], "full", run_dir, solver_label="t")
    Client(ep2["episode"]).submit({"nothing_found": True})
    sandbox.finish(ep2["episode"], solver_rc=0)
    s = sandbox.summarize(run_dir)
    assert s["overall"]["over_time"] == {"None": 1, "nominal": 1}
    md = open(os.path.join(run_dir, "summary.md")).read()
    assert "over time (nominal/hard)" in md and "TIME LIMIT EXCEEDED: 1 episode(s) past the nominal cap" in md


# =============================================================================== runner fakes (fixes 1, 2, 4)
def _args(ep, out, **kw):
    a = dict(episode=ep["episode"], task="_demo", prompt_file=os.path.join(out, "prompt.txt"), out=out,
             model="gpt-6-luna", effort="low", max_turns=10, max_tokens=1000, max_usd=1.0, allow_any_model=True)
    a.update(kw)
    os.makedirs(out, exist_ok=True)
    with open(a["prompt_file"], "w") as f:
        f.write(ep["prompt"])
    return types.SimpleNamespace(**a)


@pytest.fixture
def no_ledger(monkeypatch):
    """Runner budget bookkeeping off: no reads of the real ledgers, no writes."""
    monkeypatch.setattr(openai_agent, "_total", lambda task=None: 0.0)
    monkeypatch.setattr(openai_agent, "_append", lambda rec: None)
    monkeypatch.setattr(api_agent, "_ledger_total", lambda: 0.0)
    monkeypatch.setattr(api_agent, "_task_total", lambda task: 0.0)
    monkeypatch.setattr(api_agent, "_ledger_append", lambda rec: None)


class FakeOpenAI:
    """client.responses.create(**kw) plays a script: each step is a list of bash commands (a turn with function
    calls), a string (a text-only turn), or a callable(timeout) (e.g. a request that hangs)."""

    def __init__(self, script):
        self.script, self.n, self.inputs = list(script), 0, []
        self.responses = self

    def create(self, timeout=None, **kw):
        self.inputs.append(kw.get("input"))
        step = self.script.pop(0) if self.script else "done"
        if callable(step):
            return step(timeout)
        self.n += 1
        usage = types.SimpleNamespace(input_tokens=100, output_tokens=10,
                                      input_tokens_details=types.SimpleNamespace(cached_tokens=0),
                                      output_tokens_details=types.SimpleNamespace(reasoning_tokens=0))
        if isinstance(step, str):
            return types.SimpleNamespace(id=f"r{self.n}", usage=usage, output=[], output_text=step, status="completed")
        calls = [types.SimpleNamespace(type="function_call", call_id=f"c{self.n}_{k}",
                                       arguments=json.dumps({"command": cmd})) for k, cmd in enumerate(step)]
        return types.SimpleNamespace(id=f"r{self.n}", usage=usage, output=calls, output_text="", status="completed")


class FakeAnthropic:
    def __init__(self, script):
        self.script, self.n = list(script), 0
        self.messages = self

    def create(self, timeout=None, **kw):
        step = self.script.pop(0) if self.script else "done"
        if callable(step):
            return step(timeout)
        self.n += 1
        usage = types.SimpleNamespace(input_tokens=100, cache_creation_input_tokens=0, cache_read_input_tokens=0,
                                      output_tokens=10)

        def block(**b):
            return types.SimpleNamespace(model_dump=lambda: dict(b), **b)

        if isinstance(step, str):
            content = [block(type="text", text=step)]
        else:
            content = [block(type="tool_use", id=f"t{self.n}_{k}", name="bash", input={"command": cmd})
                       for k, cmd in enumerate(step)]
        return types.SimpleNamespace(content=content, usage=usage, stop_reason="end_turn")


def _subprocess_submit(sbx):
    # the shape gpt-6.1-sol used: submit from inside a Python program, so no `./tool submit` in the command text
    return (f"cd {sbx} && python3 -c \"import subprocess,json; "
            f"print(subprocess.check_output(['./tool','sub'+'mit',json.dumps({{'nothing_found': True}})]).decode())\"")


@pytest.mark.parametrize("runner", ["openai", "anthropic"])
def test_runner_stop_comes_from_episode_record(hx, tmp_path, no_ledger, runner):
    ep, _ = _prep(hx, hx.insts["null"][0])
    script = [[f"cd {ep['sandbox']} && cat TASK.md | head -3"], [_subprocess_submit(ep["sandbox"])],
              "I submitted.", "Still done."]
    if runner == "openai":
        meta = openai_agent.run(_args(ep, str(tmp_path / "o")), client=FakeOpenAI(script))
    else:
        meta = api_agent.run(_args(ep, str(tmp_path / "a"), model="claude-haiku-4-5"), client=FakeAnthropic(script))
    assert _record(ep["episode"])["submission"] == {"nothing_found": True}
    assert meta["stop"] == "submitted" and meta["runner_stop"] == "submitted"
    assert meta["turns"] == 2                          # stopped right after the submitting command (no nudge)
    for k in ("agent_time_s", "total_time_s", "broker_wait_s", "wall_clock_cap_s", "wall_clock_hard_s"):
        assert k in meta


def test_final_stop_prefers_record(hx):
    ep, c = _prep(hx, hx.insts["null"][0])
    assert api_agent.final_stop(ep["episode"], "ended_without_submit") == "ended_without_submit"
    assert api_agent.final_stop(ep["episode"], "submitted") == "submit_not_recorded"
    c.submit({"nothing_found": True})
    for loop in ("ended_without_submit", "max_turns", "episode_budget", "wall_clock"):
        assert api_agent.final_stop(ep["episode"], loop) == "submitted"


def test_runner_without_submit_reports_loop_reason(hx, tmp_path, no_ledger):
    ep, _ = _prep(hx, hx.insts["null"][0])
    meta = openai_agent.run(_args(ep, str(tmp_path / "o")), client=FakeOpenAI(["thinking", "still thinking"]))
    assert meta["stop"] == "ended_without_submit" and _record(ep["episode"])["submission"] is None


@pytest.mark.parametrize("runner", ["openai", "anthropic"])
def test_runner_time_warning_then_stop(hx, tmp_path, no_ledger, runner):
    """Agent time past the nominal cap: TIME_WARN, at most TIME_GRACE_TURNS more turns, then stop "wall_clock"."""
    ep, _ = _prep(hx, edited_instance(hx.insts["null"][0], hx.root, wall_clock_s=1, wall_clock_hard_s=60))
    sbx = ep["sandbox"]
    script = [[f"cd {sbx} && ./tool weights"], [f"cd {sbx} && sleep 1.2"]] + [[f"cd {sbx} && true"]] * 8
    if runner == "openai":
        meta = openai_agent.run(_args(ep, str(tmp_path / "o")), client=FakeOpenAI(script))
    else:
        meta = api_agent.run(_args(ep, str(tmp_path / "a"), model="claude-haiku-4-5"), client=FakeAnthropic(script))
    assert meta["time_warned"] and meta["stop"] == "wall_clock" and meta["turns"] == 2 + api_agent.TIME_GRACE_TURNS
    tr = open(os.path.join(str(tmp_path / ("o" if runner == "openai" else "a")), "api_transcript.jsonl")).read()
    assert "time notice from the environment" in tr


def test_runner_hard_ceiling_bounds_a_hanging_request(hx, tmp_path, no_ledger, monkeypatch):
    """ep40a2ecda0d: single API turns took up to 1906 s. A request may now only run until the ceiling."""
    import openai
    monkeypatch.setattr(api_agent, "HARD_MARGIN_S", 0.2)
    ep, _ = _prep(hx, edited_instance(hx.insts["null"][0], hx.root, wall_clock_s=1, wall_clock_hard_s=3))
    seen = []

    def hang(timeout):
        seen.append(timeout)
        time.sleep(timeout)
        raise openai.APITimeoutError(request=None)

    t0 = time.time()
    meta = openai_agent.run(_args(ep, str(tmp_path / "o")), client=FakeOpenAI([hang] * 50))
    assert meta["stop"] == "wall_clock_hard" and time.time() - t0 < 3 + 1.5
    assert seen and all(t <= 3 for t in seen)
    assert meta["total_time_s"] <= 3 + 1.0


def test_runner_bash_timeout_bounded_by_ceiling(hx, tmp_path, no_ledger, monkeypatch):
    monkeypatch.setattr(api_agent, "HARD_MARGIN_S", 0.2)
    ep, _ = _prep(hx, edited_instance(hx.insts["null"][0], hx.root, wall_clock_s=1, wall_clock_hard_s=3))
    t0 = time.time()
    meta = openai_agent.run(_args(ep, str(tmp_path / "o")),
                            client=FakeOpenAI([[f"cd {ep['sandbox']} && sleep 30"]] * 5 + ["x", "y"]))
    assert time.time() - t0 < 3 + 1.5 and meta["stop"] == "wall_clock_hard"
    assert "timed out after" in open(os.path.join(str(tmp_path / "o"), "api_transcript.jsonl")).read()


def test_api_call_retries_only_within_ceiling(monkeypatch):
    class Clock:
        def __init__(self, left):
            self.left = left

        def hard_left(self):
            return self.left

    monkeypatch.setattr(api_agent.time, "sleep", lambda s: None)
    calls = []

    def flaky(timeout):
        calls.append(timeout)
        if len(calls) < 3:
            raise ConnectionError("x")
        return "ok"

    assert api_agent._api_call(flaky, Clock(1000), lambda e: "transient") == "ok" and calls[0] == 600
    with pytest.raises(api_agent.DeadlineReached):
        api_agent._api_call(flaky, Clock(3), lambda e: "transient")
    with pytest.raises(ValueError):
        api_agent._api_call(lambda t: (_ for _ in ()).throw(ValueError("x")), Clock(1000), lambda e: None)
    calls.clear()

    def always(timeout):
        calls.append(timeout)
        raise ConnectionError("x")

    with pytest.raises(ConnectionError):
        api_agent._api_call(always, Clock(1000), lambda e: "transient")
    assert len(calls) == 4


# =============================================================================== fix 3: sandbox-local TMPDIR
def test_prepare_makes_tmp_and_prompt_names_it(hx):
    ep, _ = _prep(hx, hx.insts["null"][0])
    assert os.path.isdir(os.path.join(ep["sandbox"], "tmp"))
    assert "For temporary files use tmp/ in your working directory, never /tmp." in ep["prompt"]
    assert "tmp" in sandbox.RESERVED_SANDBOX_NAMES


def test_runner_env_points_temp_files_into_the_sandbox(hx):
    ep, _ = _prep(hx, hx.insts["null"][0])
    sbx = ep["sandbox"]
    env = api_agent.bash_env(sbx)
    assert env["TMPDIR"] == env["TMP"] == env["TEMP"] == os.path.join(sbx, "tmp")
    out, blocked = api_agent._run_bash(
        f"cd {sbx} && python3 -c \"import tempfile; f=tempfile.NamedTemporaryFile(delete=False); print(f.name)\" "
        f"&& mktemp", sbx, ep["episode"])
    assert not blocked
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) == 2 and all(l.startswith(os.path.join(sbx, "tmp") + "/") for l in lines), out


def test_runner_recreates_missing_tmp(hx):
    """Episodes prepared before the fix have no tmp/: the runner creates it so tempfile does not fail."""
    ep, _ = _prep(hx, hx.insts["null"][0])
    os.rmdir(os.path.join(ep["sandbox"], "tmp"))
    out, _ = api_agent._run_bash(f"cd {ep['sandbox']} && mktemp", ep["sandbox"], ep["episode"])
    assert out.startswith(os.path.join(ep["sandbox"], "tmp"))


TMP_CLEAN = [
    bash(C + "./tool probe_scores probe=0 > tmp/ablated.json 2>&1 && cat tmp/ablated.json"),
    bash(C + f"./tool sample_texts probe=3 > {SBX}/tmp/p3.json && head -c 200 {SBX}/tmp/p3.json"),
    bash(C + "./py - <<'EOF'\nimport tempfile, json\nwith tempfile.NamedTemporaryFile('w', delete=False) as f:\n"
             "    json.dump([1], f)\nprint(f.name)\nEOF"),
    bash(C + "f=$(mktemp) && ./tool budget > \"$f\" && cat \"$f\""),
]
TMP_FLAGGED = [   # the recorded Haiku shapes (scale1_T1_haiku45) and other real /tmp access: still INVALID
    bash(C + "./tool probe_scores probe=0 > /tmp/ablated.json 2>&1 && cat /tmp/ablated.json"),
    bash(C + "for i in 1 2 3; do ./tool probe_gradient probe=$i > /tmp/p${i}_grad.json 2>&1; done"),
    bash(C + "./py - <<'EOF'\nimport json\nd = json.load(open('/tmp/probe0_samples.json'))\nEOF"),
    bash("head -c 200 /tmp/probe0_samples.json"),
    bash(C + "ls /tmp"),
    bash(C + "cat tmp/../../other/x"),
]


@pytest.mark.parametrize("call", TMP_CLEAN, ids=[f"clean{i}" for i in range(len(TMP_CLEAN))])
def test_sandbox_tmp_is_inside(tmp_path, call):
    a = audit(tmp_path, [call])
    assert a["valid"], a["violations"]


@pytest.mark.parametrize("call", TMP_FLAGGED, ids=[f"flag{i}" for i in range(len(TMP_FLAGGED))])
def test_real_tmp_still_flagged(tmp_path, call):
    a = audit(tmp_path, [call])
    assert not a["valid"] and any(v["rule"].startswith("R2") or v["rule"].startswith("R8") for v in a["violations"])


# =============================================================================== fix 4: truncation
def test_truncate_output_saves_full_text(tmp_path):
    sbx = str(tmp_path / "sbx")
    os.makedirs(sbx)
    assert api_agent.truncate_output("short", sbx) == "short" and not os.path.exists(os.path.join(sbx, "out"))
    full = "".join(f"line {i:06d}\n" for i in range(5000))            # 60k chars
    shown = api_agent.truncate_output(full, sbx)
    assert shown.startswith(full[:6000]) and shown.endswith(full[-6000:])
    assert "OUTPUT TRUNCATED" in shown and "out/cmd_output_1.txt" in shown and str(len(full)) in shown
    assert open(os.path.join(sbx, "out", "cmd_output_1.txt")).read() == full
    shown2 = api_agent.truncate_output(full + "x", sbx)
    assert "out/cmd_output_2.txt" in shown2                              # never overwrites an earlier one


def test_run_bash_long_output_end_to_end(hx):
    ep, _ = _prep(hx, hx.insts["null"][0])
    sbx = ep["sandbox"]
    out, _ = api_agent._run_bash(f"cd {sbx} && seq 1 20000", sbx, ep["episode"])
    assert len(out) < 13000 and "out/cmd_output_1.txt" in out
    saved = open(os.path.join(sbx, "out", "cmd_output_1.txt")).read()
    assert saved.split() == [str(i) for i in range(1, 20001)]
    # the agent can read the saved file with an ordinary in-sandbox command, which the audit allows
    out2, _ = api_agent._run_bash(f"cd {sbx} && sed -n '10000,10002p' out/cmd_output_1.txt", sbx, ep["episode"])
    assert out2.split() == ["10000", "10001", "10002"]


def test_openai_tool_description_mentions_saved_output():
    assert "saved to a file under out/" in openai_agent.BASH_TOOL["description"]
    assert "saved to a file under out/" in api_agent.BASH_TOOL["description"]


# =============================================================================== fix 5: print(s.replace(x, '/'))
# FP: the recorded command of openai_T2_sol ep2c6190e0f5 call 15 (content-free: it prints token windows of a tool's
# saved examples); and narrow variants of the same construct. Each must now be VALID.
REPLACE_FP = [
    bash(C + "./py - <<'PY'\nimport json,numpy as np\nfor f in ['scratch/ex06.json','scratch/ex12.json']:\n"
             " j=json.load(open(f));j=json.load(open(j['saved_to'])) if 'saved_to' in j else j\n"
             " for k,es in j['result']['examples'].items():\n  print('\\n',k)\n  for e in es:\n"
             "   s=''.join(e['tokens']);a=np.array(e['acts']);i=a.argmax();print(round(a.mean(),2),"
             "s[:40].replace('\\n','/'),e['tokens'][i], '|',s[-45:].replace('\\n','/'))\nPY"),
    bash(C + "./py -c \"s='a b'; print(s.replace(' ', '/'))\""),
    bash(C + "./py - <<'EOF'\ns = 'x\\ty'\nprint(s.replace('\\t', '/').replace('\\n', ' '), 1)\nEOF"),
    bash(C + "./py - <<'EOF'\ns = 'x y'\nprint(s.replace('\\n', ' ').replace(' ', \"/\"))\nEOF"),
    ("Write", {"file_path": f"{SBX}/scratch/show.py", "content": "s = 'a\\nb'\nprint(s.replace('\\n', '/'))\n"}),
]
# TWINS: the same construct where the '/' (or another path) can reach the filesystem, or where the printed text can
# reach a shell. Each must still be INVALID, under R2.
REPLACE_TWINS = [
    bash(C + "./py - <<'EOF'\nimport os\ns = 'x'\nprint(os.listdir(s.replace('x', '/')))\nEOF"),     # an open/listdir
    bash(C + "./py - <<'EOF'\ns = 'xetcxpasswd'\nprint(open(s.replace('x', '/')).read())\nEOF"),
    bash(C + "./py - <<'EOF'\ns = 'xetcxpasswd'\np = s.replace('x', '/')\nprint(p)\nprint(open(p).read())\nEOF"),
    bash(C + "./py - <<'EOF'\nprint(open('/etc/passwd').read().replace('\\n', '/'))\nEOF"),  # real outside path
    bash(C + "./py - <<'EOF'\nimport os\nprint(os.replace('a', '/'))\nEOF"),                  # os.replace renames
    bash(C + "./py - <<'EOF'\nimport os as s\nprint(s.replace('a', '/'))\nEOF"),
    bash(C + "./py - <<'EOF'\ns = 'x'\nk = 'x'\nprint(s.replace(k, '/'))\nEOF"),             # computed first argument
    bash(C + "./py -c \"s='ls x'; print(s.replace('x', '/'))\" | sh"),                     # printed into a shell
    bash(C + "./py - <<'EOF'\nimport subprocess\ns = 'ls x'\np = subprocess.Popen(['sh'], stdin=subprocess.PIPE, "
             "text=True)\nprint(s.replace('x', '/'), file=p.stdin)\nEOF"),
    bash(C + "./py - <<'EOF'\nimport os\nprint = os.listdir\ns = 'x'\nprint(s.replace('x', '/'))\nEOF"),  # rebound
    bash(C + "./py - <<'EOF'\nimport os\ns = 'x'\nprint(*os.listdir(s.replace('x', '/')))\nEOF"),
    bash(C + "./py - <<'EOF'\nprint('a', 'b', sep='/')\nEOF"),                                  # not exempted
    bash(C + "./py - <<'EOF'\ns = 'x'\nprint(f\"{s.replace('x', '/')}\")\nEOF"),               # f-string
]


@pytest.mark.parametrize("call", REPLACE_FP, ids=[f"fp{i}" for i in range(len(REPLACE_FP))])
def test_print_replace_slash_is_valid(tmp_path, call):
    a = audit(tmp_path, [call])
    assert a["valid"], a["violations"]


@pytest.mark.parametrize("call", REPLACE_TWINS, ids=[f"twin{i}" for i in range(len(REPLACE_TWINS))])
def test_print_replace_twin_still_flagged(tmp_path, call):
    a = audit(tmp_path, [call])
    assert not a["valid"] and any(v["rule"] == "R2-path" for v in a["violations"]), a["violations"]


def test_recorded_fp_episode_reaudit():
    """The recorded transcript (when this machine has it): ep2c6190e0f5 is now VALID, with no other change."""
    import glob
    hits = glob.glob(os.path.expanduser("~/wt/shifthunt/runs/shifthunt/20261003-openai_T2_sol/episodes/"
                                        "ep2c6190e0f5/transcript.jsonl"))
    if not hits:
        pytest.skip("recorded transcript not on this machine")
    ed = os.path.dirname(hits[0])
    sbx = json.load(open(os.path.join(ed, "episode.json")))["sandbox"]
    a = ta.audit("ep2c6190e0f5", sbx, hits)
    assert a["valid"], a["violations"]
