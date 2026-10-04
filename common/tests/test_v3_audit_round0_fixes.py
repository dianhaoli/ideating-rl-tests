"""Fixes for the round-0 audit of harness/v3-fixes (2026-10-04). One section per finding:
 A. (major) the hard ceiling caps AGENT time; compute wait (finished or still in progress) never counts, in the broker
    and in the runners' clock (residualrecall ep99d08b74cc waited 2888 s and submitted at 1408 s of agent time);
 B. (major) prune never deletes outside the sandbox (tests in test_v3_audit_round0.py, plus file-level symlinks here);
 C. (major) a stopped command is killed with all its processes, and the runner kills left-over / detached processes
    at exit, so a detached late submit cannot disagree with the reported stop;
 D. (minor) over_time is decided by the broker's clock for an accepted submission;
 E. (minor) truncate_output never writes through a symlink;
 F. (minor) toolserver.make_local_call uses the same non-counter caps as the broker.
"""
import json
import os
import subprocess
import time
import types

import pytest

from common import api_agent, broker, openai_agent, sandbox, toolserver
from common.tests.conftest import edited_instance
from common.tests.test_v3_harness_fixes import FakeOpenAI, _args, _prep, _record, no_ledger  # noqa: F401
from common.toolclient import ToolCallError


# =============================================================================== A. wait does not count
def test_episode_times_hard_is_on_agent_time():
    # the residualrecall ep99d08b74cc shape: cap 2700, wait 2888, total 4296 -> agent 1408: not over anything
    rec = {"caps": {"wall_clock_s": 2700}, "started_at": 1000.0, "wait_s": 2888.0, "submitted_at": 1000.0 + 4296}
    t = broker.episode_times(rec)
    assert t["hard_cap_s"] == 4050 and t["total_s"] > t["hard_cap_s"]
    assert not t["over_hard"] and not t["over_nominal"] and t["agent_s"] == 1408.0
    rec2 = dict(rec, wait_s=0.0)
    assert broker.episode_times(rec2)["over_hard"]


class _Ep:
    """An in-process broker.Episode stand-in (no files)."""

    def __init__(self, rec):
        self.eid, self.rec, self.server, self.logs = rec["episode"], rec, None, []

    def save(self):
        pass

    def log(self, e):
        self.logs.append(e)

    def stop_server(self, why):
        pass


def _rec(**kw):
    r = {"episode": "ep00000000aa", "status": "open", "tools": {"weights": {}}, "counters": {"tool_calls": 0},
         "caps": broker.merged_caps({"wall_clock_s": 1, "wall_clock_hard_s": 5}), "started_at": time.time() - 100.0,
         "wait_s": 0.0}
    r.update(kw)
    return r


def test_broker_long_wait_does_not_trigger_the_ceiling():
    b = broker.Broker()
    ep = _Ep(_rec(wait_s=98.0))                    # 100 s since start, 98 s of it waiting: agent time 2 s < hard 5 s
    resp, _, _ = b._dispatch(ep, "weights", {}, None)
    assert "hard" not in resp["error"] and "time limit" in resp["error"]       # only the nominal cap (1 s)
    assert ep.rec["status"] == "open" and "close_reason" not in ep.rec
    bud = b._budget(ep.rec)
    assert bud["wall_clock_hard_s"]["used"] == 2 and bud["wall_clock_hard_s"]["remaining"] == 3
    ep2 = _Ep(_rec(wait_s=0.0))                   # same total, no wait: past the ceiling
    resp, _, _ = b._dispatch(ep2, "weights", {}, None)
    assert "hard time limit" in resp["error"] and "waiting excluded" in resp["error"]
    assert ep2.rec["status"] == "closed" and ep2.rec["close_reason"] == "wall_clock_hard"


def test_broker_submit_after_long_wait_is_accepted_by_the_ceiling_check(monkeypatch):
    b = broker.Broker()
    ep = _Ep(_rec(wait_s=98.0))
    monkeypatch.setattr(b, "_validate", lambda rec, sub: None)
    resp, _, _ = b._dispatch(ep, "submit", {"nothing_found": True}, None)
    assert resp["ok"] and ep.rec["status"] == "submitted"
    assert not ep.rec["time"]["over_hard"] and ep.rec["time"]["total_s"] >= 100


def test_broker_marks_wait_in_progress(monkeypatch):
    """_ensure_server publishes waiting_since while it waits and removes it after (also on failure)."""
    seen = []

    class Srv:
        ready = False

        def __init__(self, *a):
            self.proc = types.SimpleNamespace(pid=1)

        def alive(self):
            return True

        def wait_ready(self, hb):
            seen.append(dict(ep.rec))
            raise TimeoutError()

    monkeypatch.setattr(broker, "Server", Srv)
    ep = _Ep(_rec())
    with pytest.raises(TimeoutError):
        broker.Broker()._ensure_server(ep, None)
    assert seen and "waiting_since" in seen[0] and "waiting_since" not in ep.rec


def test_runner_clock_excludes_wait_including_in_progress(monkeypatch):
    now = 10_000.0
    rec = {"caps": {"wall_clock_s": 100, "wall_clock_hard_s": 150}, "started_at": now - 290, "wait_s": 200.0}
    monkeypatch.setattr(api_agent, "_episode_record", lambda e: rec)
    c = api_agent.EpisodeClock("ep00000000aa", t0=now - 300)
    monkeypatch.setattr(c, "now", lambda: now)
    assert c.runner_agent_s() == 100 and c.hard_left() == 50 and not c.expired()
    rec["waiting_since"] = now - 60                  # a GPU wait in progress for 60 s
    assert c.wait_s() == 260 and c.hard_left() == 110
    del rec["waiting_since"]
    rec["wait_s"] = 0.0
    assert c.expired()                                # 300 s of agent time > 150 s
    r = c.report()
    assert r["runner_agent_s"] == 300 and r["total_time_s"] == 300


def test_run_bash_ceiling_is_rechecked_while_the_command_runs(tmp_path):
    """A wait accruing during the command moves the ceiling: the command is not stopped at its start-time estimate."""
    sbx = str(tmp_path / "sbx")
    os.makedirs(sbx)
    t0 = time.time()

    class Clock:
        def expired(self):                            # the ceiling "moves" until 1.5 s, then is reached
            return time.time() - t0 > 1.5

    out, _ = api_agent._run_bash("sleep 10", sbx, "ep00000000ab", timeout=60, clock=Clock(), poll_s=0.1)
    assert "hard time limit" in out and 1.4 < time.time() - t0 < 4


# =============================================================================== B. prune
def test_prune_skips_file_symlinks_and_keeps_inside_files(hx, tmp_path):
    ep, c = _prep(hx, hx.insts["null"][0])
    outside = tmp_path / "victim.dat"
    outside.write_bytes(b"x" * (sandbox.PRUNE_MIN_BYTES + 10))
    os.symlink(str(outside), os.path.join(ep["sandbox"], "tmp", "link.dat"))
    big_in = os.path.join(ep["sandbox"], "tmp", "big.dat")
    with open(big_in, "wb") as f:
        f.write(b"y" * (sandbox.PRUNE_MIN_BYTES + 10))
    c.submit({"nothing_found": True})
    sandbox.finish(ep["episode"], solver_rc=0)
    assert outside.exists() and not os.path.exists(big_in)       # outside kept, the sandbox's own big file pruned


# =============================================================================== C. processes
def _alive(marker, wait=0.8):
    a = open(marker).read()
    time.sleep(wait)
    return a != open(marker).read()


def test_timeout_kills_a_setsid_child(tmp_path):
    sbx = str(tmp_path / "sbx")
    os.makedirs(sbx)
    marker = os.path.join(sbx, "alive")
    cmd = "setsid bash -c 'for i in $(seq 1 100); do echo $i > alive; sleep 0.2; done' & wait"
    out, _ = api_agent._run_bash(cmd, sbx, "ep00000000ac", timeout=1)
    assert "timed out" in out
    time.sleep(0.3)
    assert not _alive(marker), "a child that left the process group survived the timeout"


def test_stop_episode_kills_detached_background_jobs(tmp_path):
    sbx = str(tmp_path / "sbx")
    os.makedirs(sbx)
    marker = os.path.join(sbx, "alive")
    cmd = "setsid nohup bash -c 'for i in $(seq 1 100); do echo $i > alive; sleep 0.2; done' > /dev/null 2>&1 &"
    t0 = time.time()
    out, _ = api_agent._run_bash(cmd, sbx, "ep00000000ad", timeout=30)
    assert time.time() - t0 < 5
    time.sleep(0.5)
    assert _alive(marker, 0.5)                        # background jobs may run during the episode
    assert api_agent.stop_episode("ep00000000ad") >= 1
    time.sleep(0.2)
    assert not _alive(marker)
    assert api_agent.kill_episode_processes("ep00000000ad") == 0


def test_kill_is_scoped_to_the_episode(tmp_path):
    sbx = str(tmp_path / "sbx")
    os.makedirs(sbx)
    m1, m2 = os.path.join(sbx, "a1"), os.path.join(sbx, "a2")
    loop = "for i in $(seq 1 100); do echo $i > {m}; sleep 0.2; done"
    api_agent._run_bash(f"setsid nohup bash -c '{loop.format(m=m1)}' >/dev/null 2>&1 &", sbx, "ep00000000ae")
    api_agent._run_bash(f"setsid nohup bash -c '{loop.format(m=m2)}' >/dev/null 2>&1 &", sbx, "ep00000000af")
    time.sleep(0.5)
    api_agent.kill_episode_processes("ep00000000ae")
    time.sleep(0.2)
    assert not _alive(m1) and _alive(m2)
    api_agent.kill_episode_processes("ep00000000af")
    assert not _alive(m2)


def test_detached_late_submit_cannot_disagree_with_the_stop(hx, tmp_path, no_ledger):
    """The audit's hole: `setsid nohup ... ./tool submit ... &` returned at once and the runner could report
    ended_without_submit while the grader graded a submission that landed later. The runner now kills the detached
    process at exit; the record and the reported stop agree."""
    ep, _ = _prep(hx, hx.insts["null"][0])
    # (no `cd x && ...`: that backgrounds a subshell which keeps the output pipe open until the submit is done)
    late = "setsid nohup bash -c \"sleep 2; ./tool submit '{\\\"nothing_found\\\": true}'\" > /dev/null 2>&1 &"
    t0 = time.time()
    meta = openai_agent.run(_args(ep, str(tmp_path / "o")), client=FakeOpenAI([[late], "done", "still done"]))
    assert time.time() - t0 < 2.0                    # the runner did not wait for the detached job
    time.sleep(3.0)
    rec = _record(ep["episode"])
    assert meta["killed_processes"] >= 1
    assert (meta["stop"] == "submitted") == (rec["submission"] is not None)
    assert meta["stop"] == "ended_without_submit"


# =============================================================================== D. over_time
def _write_meta(edir, **m):
    os.makedirs(edir, exist_ok=True)
    with open(os.path.join(edir, "api_meta.json"), "w") as f:
        json.dump(m, f)


def test_over_time_uses_broker_clock_for_accepted_submission(tmp_path):
    edir = str(tmp_path / "e")
    rec = {"episode": "ep00000000b0", "caps": {"wall_clock_s": 100, "wall_clock_hard_s": 150}, "started_at": 1000.0,
           "wait_s": 0.0, "submitted_at": 1149.0}
    _write_meta(edir, episode="ep00000000b0", total_time_s=152.0, broker_wait_s=0.0, runner_agent_s=152.0,
                stop="submitted")
    t = sandbox.episode_time_report(rec, os.path.join(edir, "none.jsonl"), edir)
    assert t["over_time"] == "nominal" and t["runner_agent_s"] == 152.0
    # no accepted submission: the runner's agent time past the ceiling is "hard"
    rec2 = dict(rec, submitted_at=None, closed_at=1100.0)
    t2 = sandbox.episode_time_report(rec2, os.path.join(edir, "none.jsonl"), edir)
    assert t2["over_time"] == "hard"
    # an old api_meta without runner_agent_s: total minus wait, so a long wait is not "hard"
    _write_meta(edir, episode="ep00000000b0", total_time_s=400.0, broker_wait_s=300.0, stop="ended_without_submit")
    t3 = sandbox.episode_time_report(rec2, os.path.join(edir, "none.jsonl"), edir)
    assert t3["runner_agent_s"] == 100.0 and t3["over_time"] != "hard"


# =============================================================================== E. truncate_output
def test_truncate_output_never_writes_through_symlinks(tmp_path):
    sbx = str(tmp_path / "sbx")
    os.makedirs(sbx)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(str(outside), os.path.join(sbx, "out"))
    shown = api_agent.truncate_output("z" * 50_000, sbx)
    assert "could not be saved" in shown and not os.listdir(outside)
    os.unlink(os.path.join(sbx, "out"))
    os.makedirs(os.path.join(sbx, "out"))
    target = tmp_path / "planted.txt"                  # a dangling symlink planted at the next output name
    os.symlink(str(target), os.path.join(sbx, "out", "cmd_output_1.txt"))
    shown = api_agent.truncate_output("w" * 50_000, sbx)
    assert not target.exists() and "out/cmd_output_2.txt" in shown


# =============================================================================== F. caps
def test_toolserver_non_counter_caps_match_broker():
    assert toolserver.NON_COUNTER_CAPS == broker.NON_COUNTER_CAPS
