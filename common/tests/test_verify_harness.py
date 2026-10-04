"""Independent verification of the harness (reviewer's tests; see docs/HARNESS_VERIFICATION.md).

These tests attack the harness the way a test agent (or a buggy task) could: malformed and hostile
requests, other episodes' ids, leaky tools, hanging/crashing tools and graders, symlink tricks in the
sandbox, broker restarts and concurrency. They use the isolated broker from conftest.py (`hx`) and a
fixture task `_vcheck` (common/tests/verify_task/tasks/_vcheck) whose tools misbehave on purpose.
"""
import glob
import json
import os
import secrets
import shutil
import socket
import subprocess
import threading
import time

import pytest

from common import broker, leakscan, paths, sandbox
from common.tests.conftest import PY, REPO
from common.toolclient import Client, ToolCallError

VROOT = os.path.join(REPO, "common", "tests", "verify_task")
LEAKS = ["vsecret-alpha", "Ünïcode-Ωmega", 'quote"d\\str', "plantedkey_77", "4242,1717"]


# ------------------------------------------------------------------------------------------------ helpers
def make_vinst(root, planted=True, **caps):
    iid = "vc-" + secrets.token_hex(5)
    d = os.path.join(root, "vinst", iid)
    os.makedirs(d)
    base_caps = {"tool_calls": 50, "forward": 100, "generate": 20, "gradient": 5, "steps": 30,
                 "wall_clock_s": 600, "call_timeout_s": 20}
    base_caps.update(caps)
    inst = {"instance_id": iid, "task": "_vcheck", "tier": "T1", "dial": {}, "seed": 1,
            "canary": "RLCANARY-_vcheck-" + secrets.token_hex(8), "planted": planted,
            "answer": {"x": 1} if planted else {}, "leak_strings": LEAKS, "caps": base_caps, "extra": {}}
    with open(os.path.join(d, "instance.json"), "w") as f:
        json.dump(inst, f)
    with open(os.path.join(d, "public.json"), "w") as f:
        json.dump({"size": 4}, f)
    return d, inst


def vprep(hx, planted=True, profile="full", **caps):
    d, inst = make_vinst(hx.root, planted, **caps)
    ep = sandbox.prepare("_vcheck", d, profile, os.path.join(hx.root, "vruns"), solver_label="verify",
                         tasks_root=VROOT)
    return ep, Client(ep["episode"]), inst


def rec_of(eid):
    with open(broker.record_path(eid)) as f:
        return json.load(f)


def raw(payload, timeout=30):
    """Send raw bytes to the broker socket; return the last JSON line (or None)."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(paths.broker_sock())
    try:
        s.sendall(payload)
    except (BrokenPipeError, ConnectionResetError):
        pass                      # the broker may answer and hang up before reading an oversized request
    f = s.makefile("rb")
    last = None
    try:
        for line in f:
            last = json.loads(line)
            if "ok" in last:
                break
    except (socket.timeout, ConnectionResetError):
        pass
    s.close()
    return last


def private_needles(inst, ep_sandbox_unused=None):
    return [inst["canary"], inst["canary"].split("-")[-1], inst["instance_id"], "_vcheck", "vcheck",
            os.path.dirname(os.path.dirname(VROOT)), "instances", paths.episodes_dir()] + LEAKS


def assert_clean(text, inst):
    low = leakscan.norm(text)
    for n in private_needles(inst):
        assert leakscan.norm(n) not in low, f"private string {n!r} visible to the agent: {text[:300]}"


# ============================================================================================ isolation
def test_sandbox_files_reveal_nothing_private(hx):
    ep, c, inst = vprep(hx)
    names = sorted(os.listdir(ep["sandbox"]))
    assert names == [".episode", "TASK.md", "out", "py", "scratch", "tmp", "tool"]
    for fn in (".episode", "TASK.md", "py", "tool"):
        assert_clean(open(os.path.join(ep["sandbox"], fn), encoding="utf-8").read(), inst)
    assert_clean(ep["prompt"], inst)
    assert not os.listdir(os.path.join(ep["sandbox"], "out")) and not os.listdir(os.path.join(ep["sandbox"], "scratch"))


def test_builtin_and_error_outputs_reveal_nothing_private(hx):
    ep, c, inst = vprep(hx)
    outs = [c.call_raw("help"), c.call_raw("budget"), c.call_raw("nosuchtool"), c.call_raw("../../grader"),
            c.call_raw("load"), c.call_raw("_invoke"), c.call_raw("write_array", name="x", arr=[1]),
            c.call_raw("work", bogus=1), c.call_raw("work", forward="abc"), c.call_raw("charge_then_crash", n=1),
            c.call_raw("submit", mode="validator_crash"), c.call_raw("submit", nomode=1)]
    for o in outs:
        assert_clean(json.dumps(o, ensure_ascii=False), inst)
    assert outs[4]["error"].startswith("unknown tool") and outs[5]["error"].startswith("unknown tool")
    # a validator crash is a retryable format error, not the end of the episode
    assert c.submit({"mode": "ok"})["ok"]


def test_malformed_and_hostile_requests_do_not_crash_broker(hx):
    ep, c, inst = vprep(hx)
    eid = ep["episode"]
    cases = [b"not json\n", b"[1,2,3]\n", b'"str"\n', b"{}\n", b'{"episode": null}\n',
             b'{"episode": "../../etc/passwd", "tool": "help"}\n', b'{"episode": "ep0000000000", "tool": "help"}\n',
             json.dumps({"episode": eid, "tool": 123}).encode() + b"\n",
             json.dumps({"episode": eid, "tool": "work", "args": [1]}).encode() + b"\n",
             json.dumps({"episode": eid, "tool": "work", "args": "x"}).encode() + b"\n",
             json.dumps({"episode": [eid], "tool": "help"}).encode() + b"\n"]
    for p in cases:
        r = raw(p)
        assert r is not None and r.get("ok") is False, (p, r)
        assert_clean(json.dumps(r), inst)
    # a huge request (32 MB of args) is refused without exhausting the broker
    big = json.dumps({"episode": eid, "tool": "echo", "args": {"text": "a" * (32 << 20)}}).encode() + b"\n"
    r = raw(big, timeout=60)
    assert r is None or r.get("ok") is False
    assert broker.ping()
    assert c.call("echo", text="still alive")["text"] == "still alive"


def test_admin_commands_require_the_privileged_token(hx):
    ep, c, inst = vprep(hx)
    for cmd in ("status", "close", "evict", "shutdown"):
        r = raw(json.dumps({"admin": cmd, "episode": ep["episode"]}).encode() + b"\n")
        assert r["ok"] is False, (cmd, r)
        assert_clean(json.dumps(r), inst)
    assert broker.ping() and rec_of(ep["episode"])["status"] == "open"
    st = broker.admin("status")            # the privileged side still can
    assert st["ok"]


def test_other_episodes_id_refused_from_inside_a_sandbox_and_flagged(hx):
    ep_a, ca, _ = vprep(hx)
    ep_b, cb, inst_b = vprep(hx)
    code = ("import json,sys; from common.toolclient import Client; "
            f"print(json.dumps(Client('{ep_b['episode']}').call_raw('budget')))")
    env = dict(os.environ, PYTHONPATH=REPO)
    for cwd in (ep_a["sandbox"], os.path.join(ep_a["sandbox"], "scratch")):
        out = subprocess.run([PY, "-c", code], cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        assert r == {"ok": False, "error": "unknown episode"}, r
    # the agent's own sandbox can still call its own episode; B was untouched
    out = subprocess.run([PY, "-c", code.replace(ep_b["episode"], ep_a["episode"])], cwd=ep_a["sandbox"], env=env,
                         capture_output=True, text=True, timeout=60)
    assert json.loads(out.stdout.strip().splitlines()[-1])["ok"]
    assert rec_of(ep_b["episode"])["counters"]["tool_calls"] == 0
    g = sandbox.finish(ep_a["episode"])
    assert not g["harness"]["valid"] and "cross_episode_access" in g["harness"]["invalid_reasons"]
    assert sandbox.finish(ep_b["episode"])["harness"]["valid"]


def test_calls_after_submit_and_after_finish(hx):
    ep, c, inst = vprep(hx)
    assert c.submit({"mode": "ok"})["ok"]
    for t in ("work", "help", "budget", "submit", "nosuch"):
        assert c.call_raw(t) == {"ok": False, "error": "episode finished"}
    sandbox.finish(ep["episode"])
    assert c.call_raw("help") == {"ok": False, "error": "episode finished"}
    ep2, c2, _ = vprep(hx)
    sandbox.finish(ep2["episode"])                        # finished without submitting
    assert c2.call_raw("work") == {"ok": False, "error": "episode finished"}
    assert c2.call_raw("submit", mode="ok") == {"ok": False, "error": "episode finished"}


def test_write_array_ignores_agent_symlinks(hx, tmp_path):
    ep, c, inst = vprep(hx)
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    out = os.path.join(ep["sandbox"], "out")
    # 1) pre-planted symlink at the temp-file name and at the final name
    for nm in (".x.npy.tmp", "x.npy"):
        os.symlink(str(victim), os.path.join(out, nm))
    r = c.call("arr", name="x")
    assert victim.read_text() == "precious"
    assert r["path"].startswith("out/x")
    # 2) out/ itself replaced by a symlink to a directory outside the sandbox
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    shutil.rmtree(out)
    os.symlink(str(elsewhere), out)
    r = c.call_raw("arr", name="y")
    assert r["ok"] is False and list(elsewhere.iterdir()) == []
    g = sandbox.finish(ep["episode"])
    assert any("symlink" in x for x in g["harness"]["leak_reasons"] + g["harness"]["invalid_reasons"]) or \
        not g["harness"]["valid"]


# ============================================================================================ leak scan
LEAK_EXPECT = {
    "canary_nested": True, "canary_key": True, "canary_in_model_field": True, "canary_hex_only": True,
    "canary_spaced": True, "canary_lower_nested_model": True,
    "leak_upper": True, "leak_unicode": True, "leak_quoted": True, "leak_numbers": True,
    "leak_numbers_embedded": False, "leak_in_model_nested": False, "leak_in_model_and_outside": True,
    "instance_path": True, "instance_id": True, "codename": True, "clean": False,
}


@pytest.mark.parametrize("kind", sorted(LEAK_EXPECT))
def test_leak_scan_cases_through_broker(hx, kind):
    ep, c, inst = vprep(hx)
    r = c.call_raw("leak", kind=kind)
    rec = rec_of(ep["episode"])
    if LEAK_EXPECT[kind]:
        assert r == {"ok": False, "error": "internal error"}, r
        assert rec["leak_detected"]
    else:
        assert r["ok"] and not rec["leak_detected"], r
    if kind == "leak_in_model_nested":
        assert rec["behavioral_exposure"] == 2
    log = open(broker.tool_log_path(ep["episode"]), encoding="utf-8").read()
    assert inst["canary"].lower() not in log.lower()


@pytest.mark.parametrize("kind", ["canary", "leak", "path"])
def test_leaks_in_error_strings(hx, kind):
    ep, c, inst = vprep(hx)
    assert c.call_raw("err_leak", kind=kind) == {"ok": False, "error": "internal error"}
    assert rec_of(ep["episode"])["leak_detected"]


def test_crash_message_is_generic_and_counted(hx):
    ep, c, inst = vprep(hx)
    r = c.call_raw("charge_then_crash", n=3)
    assert r == {"ok": False, "error": "tool failed (internal error)"}
    rec = rec_of(ep["episode"])
    assert rec["counters"]["forward"] == 3 and rec["counters"]["tool_calls"] == 1 and not rec["leak_detected"]


def test_agent_text_exemption_cannot_be_used_to_hide_extra_leaks(hx):
    ep, c, inst = vprep(hx)
    # the agent's own guess echoed back once (same count as sent): not a leak
    r = c.call_raw("echo", text="is it vsecret-alpha", times=1)
    assert r["ok"] and not rec_of(ep["episode"])["leak_detected"]
    # the tool returns the string more often than the agent sent it: new information -> leak
    r = c.call_raw("echo", text="is it vsecret-alpha", times=2)
    assert r == {"ok": False, "error": "internal error"} and rec_of(ep["episode"])["leak_detected"]


def test_unit_scan_exemption_and_leaves():
    canary = "RLCANARY-x-0123456789abcdef"
    s = leakscan.scan({"a": 'say quote"d\\str'}, canary, LEAKS, ())
    assert s["leak"]
    s = leakscan.scan({"a": "vsecret-alpha"}, canary, LEAKS, (), agent_text=json.dumps({"q": "vsecret-alpha"}))
    assert not s["leak"]
    s = leakscan.scan({"a": "vsecret-alpha", "b": "vsecret-alpha"}, canary, LEAKS, (),
                      agent_text=json.dumps({"q": "vsecret-alpha"}))
    assert s["leak"]
    s = leakscan.scan({"a": "0123456789ABCDEF"}, canary, (), ())
    assert s["leak"] and "canary" in s["reasons"]
    s = leakscan.scan({"text": "VSECRET-ALPHA"}, canary, LEAKS, {"text"})
    assert not s["leak"] and s["behavioral_exposure"] == 1


def test_long_args_echo_not_flagged_at_finish(hx):
    """The tool log truncates args at 10 KB; the finish-time re-scan must not turn an exempt echo into a leak."""
    ep, c, inst = vprep(hx)
    text = "x" * 12000 + " vsecret-alpha"
    r = c.call_raw("echo", text=text, tail=20, include_text=False)
    assert r["ok"]
    g = sandbox.finish(ep["episode"])
    assert g["harness"]["valid"], g["harness"]


def test_leak_string_in_tool_written_filename_flagged(hx):
    ep, c, inst = vprep(hx)
    c.call("write_named")
    assert any(LEAKS[3] in f for f in os.listdir(os.path.join(ep["sandbox"], "out")))
    g = sandbox.finish(ep["episode"])
    assert not g["harness"]["valid"] and g["harness"]["leak_detected"]


def test_agent_notes_with_answer_counted_not_flagged(hx):
    ep, c, inst = vprep(hx)
    with open(os.path.join(ep["sandbox"], "scratch", "notes.md"), "w") as f:
        f.write("I think the answer is VSECRET-ALPHA")
    with open(os.path.join(ep["sandbox"], "TASK.md"), "a") as f:     # agent appended its notes to TASK.md
        f.write("\nmy notes: vsecret-alpha\n")
    g = sandbox.finish(ep["episode"])
    assert g["harness"]["valid"], g["harness"]
    assert g["harness"]["agent_files_with_leak_strings"] == 2


# ============================================================================================ caps
def test_every_cap_counted_exactly(hx):
    ep, c, inst = vprep(hx, tool_calls=12, forward=10, generate=4, gradient=2, steps=5)
    c.call("work", forward=4, generate=1)
    c.call("work", gradient=2, steps=5)
    with pytest.raises(ToolCallError, match="gradient budget exceeded"):
        c.call("work", forward=1, gradient=1)          # forward charged before the refused gradient
    with pytest.raises(ToolCallError, match="steps budget exceeded"):
        c.call("work", steps=1)
    with pytest.raises(ToolCallError, match="bad arguments"):
        c.call("work", nope=1)
    with pytest.raises(ToolCallError, match="unknown tool"):
        c.call("nosuch")                               # unknown tools do not cost a call
    c.call("help"), c.call("budget")                   # free
    b = c.call("budget")
    assert {k: b[k]["used"] for k in ("tool_calls", "forward", "generate", "gradient", "steps")} == \
        {"tool_calls": 5, "forward": 5, "generate": 1, "gradient": 2, "steps": 5}
    for _ in range(7):
        c.call("work")
    with pytest.raises(ToolCallError, match="tool_calls cap reached"):
        c.call("work")
    assert c.call("budget")["tool_calls"] == {"used": 12, "cap": 12, "remaining": 0}
    log = [json.loads(l) for l in open(broker.tool_log_path(ep["episode"]))]
    calls = [e for e in log if "tool" in e]
    assert sum(e["charges"].get("tool_calls", 0) for e in calls) == 12
    assert sum(e["charges"].get("forward", 0) for e in calls) == 5


def test_timeout_keeps_charges_made_before_the_hang(hx):
    ep, c, inst = vprep(hx, call_timeout_s=2)
    t0 = time.time()
    r = c.call_raw("charge_then_hang", n=7, seconds=30)
    assert "timed out" in r["error"] and time.time() - t0 < 15
    rec = rec_of(ep["episode"])
    assert rec["counters"]["forward"] == 7 and rec["counters"]["tool_calls"] == 1
    assert c.call("work", forward=1)["done"]           # a fresh server after the kill
    assert rec_of(ep["episode"])["counters"]["forward"] == 8


def test_wall_clock_cap_exact(hx):
    ep, c, inst = vprep(hx, wall_clock_s=3, call_timeout_s=20)
    c.call("sleep", seconds=3.5)                       # a call that started in time finishes
    with pytest.raises(ToolCallError, match="time limit"):
        c.call("work")
    b = c.call("budget")
    assert b["wall_clock_s"]["remaining"] == 0 and b["tool_calls"]["used"] == 1
    assert c.submit({"mode": "ok"})["ok"]


def test_counters_survive_eviction_and_broker_restart(hx):
    ep, c, inst = vprep(hx)
    c.call("work", forward=3, generate=2)
    broker.admin("evict", episode=ep["episode"])
    c.call("work", forward=1)
    started = rec_of(ep["episode"])["started_at"]
    broker.stop()
    assert not broker.ping()
    broker.start(quiet=True)
    b = c.call("budget")
    assert b["forward"]["used"] == 4 and b["generate"]["used"] == 2 and b["tool_calls"]["used"] == 2
    c.call("work", forward=1)
    rec = rec_of(ep["episode"])
    assert rec["counters"]["forward"] == 5 and rec["started_at"] == started


def test_concurrent_episodes_do_not_share_counters(hx):
    eps = [vprep(hx, tool_calls=6) for _ in range(2)]
    errors = []

    def hammer(c, n):
        for _ in range(n):
            r = c.call_raw("work", forward=1)
            if not r["ok"]:
                errors.append(r["error"])

    threads = [threading.Thread(target=hammer, args=(c, 5)) for _, c, _ in eps for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    for ep, c, _ in eps:
        rec = rec_of(ep["episode"])
        assert rec["counters"]["tool_calls"] == 6 and rec["counters"]["forward"] == 6
    assert len(errors) == 8 and all("tool_calls cap reached" in e for e in errors)


# ============================================================================================ grader
@pytest.mark.parametrize("mode", ["crash", "hang", "exit0_no_output", "string_score", "nan_score", "big_score",
                                  "nonbool_pass", "not_dict"])
def test_grader_failure_scores_zero_with_error(hx, mode, monkeypatch):
    monkeypatch.setenv("RL_GRADER_TIMEOUT_S", "3")
    ep, c, inst = vprep(hx)
    assert c.submit({"mode": mode})["ok"]
    t0 = time.time()
    g = sandbox.finish(ep["episode"])
    assert time.time() - t0 < 60
    assert g["score"] == 0.0 and g["pass"] is False and g.get("grader_error")
    assert "grader_error" in g["harness"]["invalid_reasons"]
    saved = json.load(open(os.path.join(ep["episode_dir"], "grade.json")))
    assert saved["score"] == 0.0


def test_grader_ok_runs_in_another_process(hx):
    ep, c, inst = vprep(hx)
    c.submit({"mode": "ok"})
    g = sandbox.finish(ep["episode"])
    assert g["pass"] is True and g["harness"]["valid"] and g["details"]["pid"] != os.getpid()
    st = broker.admin("status")["result"]
    assert g["details"]["pid"] != st["pid"]


# ============================================================================================ fingerprinting
def _shape(o):
    if isinstance(o, dict):
        return {k: _shape(v) for k, v in sorted(o.items())}
    if isinstance(o, list):
        return [len(o)] + ([_shape(o[0])] if o else [])
    return type(o).__name__


def test_demo_planted_and_null_indistinguishable(hx, tmp_path):
    out = subprocess.run([PY, os.path.join(REPO, "tasks/_demo/generate.py"), "--out", str(tmp_path), "--n", "40",
                          "--seed", "11"], capture_output=True, text=True, check=True).stdout.split()
    groups = {True: [], False: []}
    for d in out:
        groups[json.load(open(os.path.join(d, "instance.json")))["planted"]].append(d)
    assert len(groups[True]) >= 8 and len(groups[False]) >= 8

    def profile(d):
        inst = json.load(open(os.path.join(d, "instance.json")))
        pub = json.load(open(os.path.join(d, "public.json")))
        return (tuple(sorted(os.listdir(d))), tuple(sorted(pub)), json.dumps(pub, sort_keys=True),
                tuple(sorted(inst)), len(inst["leak_strings"]), json.dumps(inst["caps"], sort_keys=True),
                inst["tier"], json.dumps(inst["dial"]), len(inst["instance_id"]))

    assert {profile(d) for d in groups[True]} == {profile(d) for d in groups[False]}
    sizes = {k: [os.path.getsize(os.path.join(d, "table.json")) for d in v] for k, v in groups.items()}
    # sizes within noise: the ranges overlap and the means differ by less than the spread
    assert max(min(sizes[True]), min(sizes[False])) <= min(max(sizes[True]), max(sizes[False]))
    mean = {k: sum(v) / len(v) for k, v in sizes.items()}
    spread = max(max(v) - min(v) for v in sizes.values())
    assert abs(mean[True] - mean[False]) < spread
    # agent-visible: TASK.md, tool output shapes, error texts, array files
    seen = {}
    for kind in (True, False):
        d = groups[kind][0]
        ep = sandbox.prepare("_demo", d, "full", os.path.join(hx.root, "fpruns"), solver_label="fp")
        c = Client(ep["episode"])
        task_md = open(os.path.join(ep["sandbox"], "TASK.md")).read()
        outs = [c.call_raw("query", xs=[0, 1, 2]), c.call_raw("weights"), c.call_raw("help"), c.call_raw("budget"),
                c.call_raw("query", xs=[999]), c.call_raw("submit", edited_input=-1, new_output=0)]
        npy = os.path.join(ep["sandbox"], "out", "table.npy")
        seen[kind] = (task_md, [_shape(o) for o in outs], [o.get("error") for o in outs], os.path.getsize(npy),
                      sorted(os.listdir(ep["sandbox"])))
        sandbox.finish(ep["episode"])
    assert seen[True] == seen[False]


# ============================================================================================ run-scripted + summarize
def test_run_scripted_end_to_end_and_summary_numbers(hx):
    run_dir = os.path.join(hx.root, "e2e_verify")
    insts = hx.insts["planted"][:2] + hx.insts["null"][:2]
    rows = sandbox.run_scripted("_demo", os.path.join(REPO, "tasks/_demo/reference_solver.py"), insts, "full",
                                run_dir, repeats=2, solver_label="reference")
    assert len(rows) == 8 and all(r["pass"] and r["valid"] and r["solver_rc"] == 0 for r in rows)
    bb = sandbox.run_scripted("_demo", os.path.join(REPO, "tasks/_demo/blackbox_control.py"), insts, "blackbox",
                              run_dir, repeats=1, solver_label="blackbox")
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    ref = s["groups"]["solver=reference model=None profile=full tier=T1"]
    assert ref["n_episodes"] == 8 and ref["n_pass"] == 8 and ref["pass_rate"] == 1.0
    lo, hi = sandbox.wilson(8, 8)
    assert ref["wilson95"] == [round(lo, 4), round(hi, 4)]
    assert ref["null_slot_false_claim_rate"] == 0.0 and ref["n_null_slots"] == 4
    blk = s["groups"]["solver=blackbox model=None profile=blackbox tier=T1"]
    k = sum(r["pass"] for r in bb)
    assert blk["n_pass"] == k and blk["n_episodes"] == 4
    assert s["overall"]["n_episodes"] == 12
    md = open(os.path.join(run_dir, "summary.md")).read()
    assert "solver=reference" in md and "[0.676, 1.000]" in md
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    assert set(cfg["solver_labels"]) == {"reference", "blackbox"} and len(cfg["instances"]) == 4
    for e in glob.glob(os.path.join(run_dir, "episodes", "*")):
        for fn in ("grade.json", "audit.json", "tool_log.jsonl", "submission.json", "episode.json", "TASK.md"):
            assert os.path.exists(os.path.join(e, fn)), (e, fn)
        txt = open(os.path.join(e, "episode.json")).read() + open(os.path.join(e, "tool_log.jsonl")).read()
        assert "RLCANARY" not in txt.upper()


def test_wilson_matches_reference_implementation():
    from scipy.stats import binomtest
    for n in (1, 2, 5, 7, 10, 30, 100, 1000):
        for k in sorted({0, 1, n // 3, n // 2, n - 1, n}):
            if k < 0:
                continue
            ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
            lo, hi = sandbox.wilson(k, n)
            assert abs(lo - ci.low) < 2e-4 and abs(hi - ci.high) < 2e-4, (k, n, lo, hi, ci)
    assert sandbox.wilson(0, 0) == (0.0, 1.0)
    lo, hi = sandbox.wilson(5, 10)
    assert abs(lo - 0.2366) < 1e-4 and abs(hi - 0.7634) < 1e-4
