"""End-to-end harness tests on the GPU-free _demo task, against an isolated broker (see conftest)."""
import json
import os
import subprocess
import time

import pytest

from common import broker, sandbox
from common.tests.conftest import REPO, PY, edited_instance
from common.toolclient import Client, ToolCallError


def _prep(hx, inst, profile="full", label="test"):
    ep = sandbox.prepare("_demo", inst, profile, hx.run_dir, solver_label=label)
    return ep, Client(ep["episode"])


def _record(eid):
    with open(broker.record_path(eid)) as f:
        return json.load(f)


def _shape(o):
    """Structure of a JSON value: keys, types and list lengths (values ignored)."""
    if isinstance(o, dict):
        return {k: _shape(v) for k, v in sorted(o.items())}
    if isinstance(o, list):
        return [len(o)] + ([_shape(o[0])] if o else [])
    return type(o).__name__


# ------------------------------------------------------------------------------ caps
def test_forward_cap_enforced(hx):
    ep, c = _prep(hx, hx.insts["planted"][0])
    c.call("query", xs=list(range(30)))
    with pytest.raises(ToolCallError, match="forward budget exceeded"):
        c.call("query", xs=list(range(11)))          # 30 + 11 > 40
    c.call("query", xs=list(range(10)))              # exactly reaches the cap
    b = c.call("budget")
    assert b["forward"]["used"] == 40 and b["forward"]["remaining"] == 0
    assert b["tool_calls"]["used"] == 3              # failed call still costs a tool call; budget is free


def test_tool_calls_cap_enforced(hx):
    ep, c = _prep(hx, edited_instance(hx.insts["null"][0], hx.root, tool_calls=3))
    for _ in range(3):
        c.call("weights")
    with pytest.raises(ToolCallError, match="tool_calls cap reached"):
        c.call("weights")
    c.call("help"), c.call("budget")                 # built-ins stay free and available
    assert c.submit({"nothing_found": True})["ok"]  # submitting is still possible


def test_wall_clock_cap(hx):
    ep, c = _prep(hx, edited_instance(hx.insts["null"][0], hx.root, wall_clock_s=1))
    c.call("weights")
    time.sleep(1.5)
    with pytest.raises(ToolCallError, match="time limit"):
        c.call("weights")
    assert c.submit({"nothing_found": True})["ok"]


def test_per_call_timeout_and_server_restart(hx):
    ep, c = _prep(hx, edited_instance(hx.insts["null"][0], hx.root, call_timeout_s=1), profile="debug")
    with pytest.raises(ToolCallError, match="timed out"):
        c.call("slow", seconds=5)
    assert c.call("query", xs=[1, 2])["outputs"]     # a fresh server is started transparently
    assert _record(ep["episode"])["counters"]["tool_calls"] == 2


def test_unknown_tool_and_profile_restriction(hx):
    ep, c = _prep(hx, hx.insts["planted"][0], profile="blackbox")
    with pytest.raises(ToolCallError, match="unknown tool"):
        c.call("weights")                            # white-box tool not in the blackbox profile
    assert [t["name"] for t in c.call("help")["tools"]] == ["query"]


def test_bad_arguments_generic_error(hx):
    ep, c = _prep(hx, hx.insts["planted"][0])
    with pytest.raises(ToolCallError, match="bad arguments"):
        c.call("query", nope=1)


# ------------------------------------------------------------------------------ submission
def test_one_submission_with_format_retry(hx):
    ep, c = _prep(hx, hx.insts["planted"][0])
    r = c.submit({"edited_input": "x"})
    assert not r["ok"] and r["error"].startswith("invalid submission")
    r = c.submit({"nothing_found": True})
    assert r["ok"] and r["result"]["accepted"]
    r = c.submit({"nothing_found": True})
    assert r == {"ok": False, "error": "episode finished"}
    assert c.call_raw("query", xs=[1]) == {"ok": False, "error": "episode finished"}
    rec = _record(ep["episode"])
    assert rec["status"] == "submitted" and rec["submission"] == {"nothing_found": True}


def test_grader_runs_out_of_process_and_reference_passes(hx):
    from tasks._demo.reference_solver import solve  # noqa: imported only for the solve() logic
    for inst in hx.insts["planted"][:1] + hx.insts["null"][:1]:
        ep, c = _prep(hx, inst)
        c.submit(solve(c.call, c.sandbox))
        g = sandbox.finish(ep["episode"])
        assert g["pass"] is True and g["harness"]["valid"]
        assert g["details"]["grader_pid"] != os.getpid()
        edir = ep["episode_dir"]
        for fn in ("grade.json", "audit.json", "tool_log.jsonl", "submission.json", "episode.json", "agent_prompt.txt"):
            assert os.path.exists(os.path.join(edir, fn)), fn
        # nothing privileged in the run-dir copy of the record
        txt = open(os.path.join(edir, "episode.json")).read()
        assert "RLCANARY" not in txt and "leak_strings" not in txt


def test_no_submission_scores_zero(hx):
    ep, c = _prep(hx, hx.insts["planted"][0])
    c.call("weights")
    g = sandbox.finish(ep["episode"])
    assert g["pass"] is False and g["score"] == 0 and g["harness"]["submitted"] is False
    assert g["harness"]["status"] == "closed"


# ------------------------------------------------------------------------------ leak scan
def test_leak_canary_replaced_and_flagged(hx):
    ep, c = _prep(hx, hx.insts["planted"][0], profile="debug")
    r = c.call_raw("leak_canary")
    assert r == {"ok": False, "error": "internal error"}
    rec = _record(ep["episode"])
    assert rec["leak_detected"] and rec["leak_events"][0]["reasons"] == ["canary"]
    log = open(broker.tool_log_path(ep["episode"])).read()
    assert "RLCANARY" not in log.upper()             # the log stores the served (replaced) response
    g = sandbox.finish(ep["episode"])
    assert g["harness"]["valid"] is False and "leak_detected" in g["harness"]["invalid_reasons"]


def test_leak_string_outside_model_field_flagged(hx):
    ep, c = _prep(hx, hx.insts["null"][0], profile="debug")
    assert c.call_raw("leak_label") == {"ok": False, "error": "internal error"}
    assert _record(ep["episode"])["leak_detected"]


def test_behavioral_exposure_counted_not_flagged(hx):
    ep, c = _prep(hx, hx.insts["planted"][0], profile="debug")
    r = c.call("model_says_label")
    assert "dmtag-" in r["text"]
    rec = _record(ep["episode"])
    assert not rec["leak_detected"] and rec["behavioral_exposure"] == 2
    last = open(broker.tool_log_path(ep["episode"])).read().strip().split("\n")[-1]
    assert json.loads(last)["behavioral_exposure"] == 2


def test_echo_of_agent_text_is_not_a_leak(hx):
    ep, c = _prep(hx, hx.insts["planted"][0], profile="debug")
    with open(os.path.join(hx.insts["planted"][0], "instance.json")) as f:
        tag = json.load(f)["extra"]["tag"]
    r = c.call("echo", text=f"is it {tag}?")        # agent already knows what it sent
    assert r["note"].endswith("?") and not _record(ep["episode"])["leak_detected"]


def test_sandbox_file_with_canary_flagged_at_finish(hx):
    ep, c = _prep(hx, hx.insts["null"][0])
    with open(os.path.join(hx.insts["null"][0], "instance.json")) as f:
        canary = json.load(f)["canary"]
    with open(os.path.join(ep["sandbox"], "scratch", "notes.txt"), "w") as f:
        f.write("copied: " + canary)
    g = sandbox.finish(ep["episode"])
    assert g["harness"]["leak_detected"] and not g["harness"]["valid"]


def test_task_md_checks(hx, tmp_path):
    # TASK.md must not contain the codename, the canary or a leak string
    root = tmp_path / "troot"
    (root / "tasks").mkdir(parents=True)
    import shutil
    shutil.copytree(os.path.join(REPO, "tasks/_demo"), root / "tasks/_demo",
                    ignore=shutil.ignore_patterns("instances", "__pycache__"))
    p = root / "tasks/_demo/agent_prompt.md"
    base = p.read_text()
    for bad in ("This is the DEMO task.", "{public.modulus} RLCANARY-_demo-0", ):
        p.write_text(base + bad)
        with pytest.raises(ValueError, match="TASK.md failed checks"):
            sandbox.prepare("_demo", hx.insts["null"][0], "full", hx.run_dir, tasks_root=str(root))
    with open(os.path.join(hx.insts["null"][0], "instance.json")) as f:
        tag = json.load(f)["extra"]["tag"]
    p.write_text(base + tag)
    with pytest.raises(ValueError, match="leak_strings"):
        sandbox.prepare("_demo", hx.insts["null"][0], "full", hx.run_dir, tasks_root=str(root))


# ------------------------------------------------------------------------------ no fingerprinting
def test_planted_and_null_tool_outputs_have_same_shape(hx):
    shapes = {}
    for kind in ("planted", "null"):
        ep, c = _prep(hx, hx.insts[kind][0])
        shapes[kind] = (_shape(c.call("query", xs=[0, 1, 2])), _shape(c.call("weights")), _shape(c.call("help")))
        import numpy as np
        arr = np.load(os.path.join(c.sandbox, "out", "table.npy"))
        shapes[kind] += (arr.shape, str(arr.dtype))
        sandbox.finish(ep["episode"])
    assert shapes["planted"] == shapes["null"]
    sizes = {k: sorted((fn, os.path.getsize(os.path.join(hx.insts[k][0], fn)) // 64)
                       for fn in os.listdir(hx.insts[k][0])) for k in ("planted", "null")}
    assert [f for f, _ in sizes["planted"]] == [f for f, _ in sizes["null"]]


# ------------------------------------------------------------------------------ eviction
def test_idle_eviction_and_reload_keeps_counters(hx):
    ep, c = _prep(hx, hx.insts["planted"][0])
    c.call("query", xs=[1, 2, 3])
    eid = ep["episode"]

    def server_state():
        for e in broker.admin("status")["result"]["episodes"]:
            if e["episode"] == eid:
                return e["server"]
    assert server_state() == "ready"
    t0 = time.time()
    while server_state() != "none" and time.time() - t0 < 30:   # RL_IDLE_S=4 in the test broker
        time.sleep(0.5)
    assert server_state() == "none"
    assert c.call("query", xs=[4])["outputs"]      # reloaded transparently
    rec = _record(eid)
    assert rec["counters"]["forward"] == 4 and rec["counters"]["tool_calls"] == 2
    log = [json.loads(l) for l in open(broker.tool_log_path(eid))]
    assert [e.get("event") for e in log].count("server_started") == 2
    assert any("evicted" in str(e.get("reason")) for e in log if e.get("event") == "server_stopped")


def test_concurrent_episodes(hx):
    import threading
    eps = [_prep(hx, d) for d in hx.insts["planted"][:2] + hx.insts["null"][:2]]
    errs = []

    def work(c):
        try:
            for _ in range(3):
                c.call("query", xs=[1, 2])
        except Exception as e:   # pragma: no cover
            errs.append(e)
    ts = [threading.Thread(target=work, args=(c,)) for _, c in eps]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs
    for ep, _ in eps:
        assert _record(ep["episode"])["counters"]["forward"] == 6


# ------------------------------------------------------------------------------ agent-side client
def test_agent_client_cli_py39(hx):
    ep, c = _prep(hx, hx.insts["planted"][0], profile="debug")
    sbx = ep["sandbox"]

    def tool(*args):
        r = subprocess.run(["./tool", *args], cwd=sbx, capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout, r.stderr
    rc, out, _ = tool("help")
    assert rc == 0 and json.loads(out)["ok"]
    rc, out, _ = tool("query", "xs=[1,2]")
    assert json.loads(out)["result"]["outputs"]
    rc, out, _ = tool("query", '{"xs": [3]}')
    assert len(json.loads(out)["result"]["outputs"]) == 1
    rc, out, _ = tool("big", "n=10000")                    # > 30 KB -> saved to out/resp_1.json
    j = json.loads(out)
    assert j["saved_to"] == "out/resp_1.json" and os.path.exists(os.path.join(sbx, "out/resp_1.json"))
    rc, out, _ = tool("weights")
    r = subprocess.run(["./py", "-c", "import numpy as np; print(np.load('out/table.npy').shape)"], cwd=sbx,
                       capture_output=True, text=True)
    assert r.stdout.strip() == "(211,)"
    r = subprocess.run(["./py", "-c", "import torch"], cwd=sbx, capture_output=True, text=True)
    assert r.returncode != 0                             # the analysis python cannot load models
    rc, out, _ = tool("submit", "nothing_found=true")
    assert json.loads(out)["ok"]
    head = open(os.path.join(sbx, "tool")).readline()
    assert head.strip() == "#!/usr/bin/python3"


def test_sandbox_contents_and_prompt(hx):
    ep, c = _prep(hx, hx.insts["null"][0])
    assert sorted(os.listdir(ep["sandbox"])) == [".episode", "TASK.md", "out", "py", "scratch", "tmp", "tool"]
    p = ep["prompt"]
    for must in (f"Your working directory is {ep['sandbox']}/", "Only read, write and run things inside that directory",
                 "./tool", "./py", "Do not use web search or web fetch", "Do not spawn sub-agents",
                 "./tool submit"):
        assert must in p, must
    assert "_demo" not in p and "demo" not in open(os.path.join(ep["sandbox"], "TASK.md")).read().lower()


# ------------------------------------------------------------------------------ run-scripted + summarize
def test_run_scripted_end_to_end_and_summary(hx):
    run_dir = os.path.join(hx.root, "runs_scripted")
    insts = hx.insts["planted"][:2] + hx.insts["null"][:2]
    r = subprocess.run([PY, "-m", "common.sandbox", "run-scripted", "--task", "_demo", "--solver",
                        os.path.join(REPO, "tasks/_demo/reference_solver.py"), "--instances", *insts,
                        "--profile", "full", "--run-dir", run_dir, "--repeats", "2", "--solver-label", "reference"],
                       cwd=REPO, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    assert len(rows) == 8 and all(x["pass"] and x["valid"] for x in rows)
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    g = s["overall"]
    assert g["n_valid"] == 8 and g["pass_rate"] == 1.0
    assert g["wilson95"] == [round(sandbox.wilson(8, 8)[0], 4), 1.0]
    assert g["per_instance_hist"]["1"] == 4 and g["null_slot_false_claim_rate"] == 0.0
    assert os.path.exists(os.path.join(run_dir, "summary.md")) and os.path.exists(os.path.join(run_dir, "config.json"))
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    assert cfg["git_sha"] and len(cfg["instances"]) == 4
    # the solver saw only the episode id
    log = open(os.path.join(run_dir, "episodes", rows[0]["episode"], "solver.log")).read()
    assert "instance" not in log.lower()


def test_run_scripted_blackbox_profile(hx):
    run_dir = os.path.join(hx.root, "runs_bb")
    sandbox.run_scripted("_demo", os.path.join(REPO, "tasks/_demo/blackbox_control.py"), hx.insts["planted"][:3],
                         "blackbox", run_dir, repeats=1, solver_label="blackbox")
    s = json.load(open(os.path.join(run_dir, "summary.json")))["overall"]
    assert s["n_valid"] == 3 and s["pass_rate"] <= 2 / 3   # finds the edit only by luck (~19% each)


def test_agent_client_prints_waiting_for_compute(tmp_path):
    """Fake broker that first reports GPU-queue waiting, then answers: the client must print the
    'waiting for compute...' line on stderr and the final envelope on stdout."""
    import shutil
    import socket
    import threading
    sbx = tmp_path / "ep0000000001"
    sbx.mkdir()
    shutil.copy(os.path.join(REPO, "common/agent_client/tool"), sbx / "tool")
    (sbx / ".episode").write_text("ep0000000001\n")
    sock_path = str(tmp_path / "s.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        req = json.loads(conn.makefile("rb").readline())
        assert req == {"episode": "ep0000000001", "tool": "logits", "args": {"top_k": 3, "p": "a b"}}
        conn.sendall(b'{"status": "waiting"}\n')
        time.sleep(0.3)
        conn.sendall(b'{"status": "loading"}\n{"ok": true, "result": {"x": 1}}\n')
        conn.close()
    t = threading.Thread(target=serve)
    t.start()
    r = subprocess.run(["/usr/bin/python3", "tool", "logits", "top_k=3", "p=a b"], cwd=sbx, capture_output=True,
                       text=True, timeout=30, env=dict(os.environ, RL_BROKER_SOCK=sock_path))
    t.join()
    srv.close()
    assert "waiting for compute..." in r.stderr
    assert json.loads(r.stdout) == {"ok": True, "result": {"x": 1}} and r.returncode == 0


def test_prepare_cli_first_line_is_episode_id(hx):
    """Builders/scripts do E=$(python -m common.sandbox prepare ... | head -1)."""
    r = subprocess.run([PY, "-m", "common.sandbox", "prepare", "--task", "_demo", "--instance-dir", hx.insts["null"][0],
                        "--profile", "full", "--run-dir", hx.run_dir], cwd=REPO, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert broker.EPISODE_RX.match(lines[0]) and lines[1].startswith("SANDBOX ")
    assert "Your working directory is" in r.stdout
    assert open(os.path.join(hx.run_dir, "episodes", lines[0], "agent_prompt.txt")).read().strip() in r.stdout
