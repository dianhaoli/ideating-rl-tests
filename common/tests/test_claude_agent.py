"""Tests for the OS-isolated Claude Code launcher (common/claude_agent.py), against the isolated broker (conftest).

No LLM is called here: a fake `claude` (a bash script that speaks stream-json) runs inside the real bwrap jail, so the
jail, proxy, transcript conversion, tool-list guard and wall-clock kill are all exercised for real. The live Haiku
episode and the red-team against the real ~/rlsbx broker are in common/tests/claude_agent_selftest.sh.
"""
import argparse
import json
import os
import shutil
import socket
import tempfile
import time

import pytest

from common import claude_agent as ca, sandbox, transcript_audit

pytestmark = pytest.mark.skipif(not (os.path.exists(ca.BWRAP) and os.path.exists(ca.SOCAT)),
                                reason="bubblewrap/socat not installed")

FAKE = r"""#!/bin/bash
cat > /dev/null
SBX=$(pwd)
init() { printf '{"type":"system","subtype":"init","session_id":"s1","model":"fake","cwd":"%s","tools":%s,"mcp_servers":[]}\n' "$SBX" "$1"; }
case "$FAKE_MODE" in
  ok)
    init '["Bash","Read","Write","Edit","Glob","Grep"]'
    CMD="./tool submit '{\"nothing_found\": true}'"
    OUT=$(./tool submit '{"nothing_found": true}')
    python3 - "$CMD" "$OUT" <<'EOF'
import json, sys
cmd, out = sys.argv[1], sys.argv[2]
print(json.dumps({"type": "assistant", "session_id": "s1", "message": {"role": "assistant", "content": [
    {"type": "text", "text": "submitting"}, {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": cmd}}]}}))
print(json.dumps({"type": "user", "session_id": "s1", "message": {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "t1", "content": out}]}}))
print(json.dumps({"type": "result", "subtype": "success", "session_id": "s1", "total_cost_usd": 0.0123,
                  "num_turns": 2, "usage": {"output_tokens": 5}}))
EOF
    ;;
  extra)
    init '["Bash","Read","Write","Edit","Glob","Grep","WebFetch"]'
    sleep 60 ;;
  hang)
    init '["Bash","Read"]'
    sleep 120 ;;
esac
"""


@pytest.fixture
def iso(hx, tmp_path, monkeypatch):
    fake = tmp_path / "fake_claude"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    monkeypatch.setattr(ca, "CLAUDE_BIN", str(fake))
    short = tempfile.mkdtemp(prefix="rlca", dir="/tmp")    # Unix socket paths are limited to 108 bytes
    monkeypatch.setattr(ca, "STATE_ROOT", os.path.join(short, "st"))
    monkeypatch.setattr(ca, "LEDGER_DIR", str(tmp_path / "ledger"))
    monkeypatch.setattr(ca, "CLI_LEDGER", str(tmp_path / "ledger" / "cli.jsonl"))
    yield hx, tmp_path
    shutil.rmtree(short, ignore_errors=True)


def _run(hx, tmp_path, monkeypatch, mode_, wall=120):
    ep = sandbox.prepare("_demo", hx.insts["null"][0], "full", hx.run_dir, solver_label="fake-claude")
    out = os.path.join(hx.run_dir, "episodes", ep["episode"])
    monkeypatch.setattr(ca, "auth_setup", lambda mode, wall: ({"CLAUDE_CODE_OAUTH_TOKEN": "dummy", "FAKE_MODE": mode_},
                                                             lambda: {"Authorization": "Bearer x"}, {}))
    args = argparse.Namespace(episode=ep["episode"], task="_demo", prompt_file=os.path.join(out, "agent_prompt.txt"),
                              out=out, model="haiku", effort="high", max_turns=5, max_wall_s=wall, max_usd=0.1,
                              auth="oauth", no_nudge=True, keep_state=False)
    t0 = time.time()
    ca.run(args)
    meta = json.load(open(os.path.join(out, "api_meta.json")))
    return ep, out, meta, time.time() - t0


def test_fake_episode_end_to_end(iso, monkeypatch):
    hx, tmp = iso
    ep, out, meta, _ = _run(hx, tmp, monkeypatch, "ok")
    assert meta["stop"] == "submitted" and meta["submitted"] and meta["usd"] == 0.0123 and meta["turns"] == 2
    assert meta["model"] == "claude-haiku-4-5-20251001"
    lines = [json.loads(l) for l in open(os.path.join(out, "api_transcript.jsonl"))]
    assert lines[0]["type"] == "user" and transcript_audit.prompt_match(lines[0]["message"]["content"],
                                                                          ep["prompt"]) == "exact"
    assert all(l.get("cwd") == os.path.realpath(ep["sandbox"]) for l in lines[1:])
    g = sandbox.finish(ep["episode"], [os.path.join(out, "api_transcript.jsonl")], "claude-cli:fake:high")
    assert g["harness"]["valid"], g["harness"]["invalid_reasons"]
    assert g["harness"]["audit_valid"] and g["harness"]["submitted"]
    # the private state dir (config dir, proxy socket) is removed after the run
    assert not os.listdir(ca.STATE_ROOT)
    assert json.loads(open(ca.CLI_LEDGER).readline())["usd"] == 0.0123


def test_unexpected_tool_stops_episode(iso, monkeypatch):
    hx, tmp = iso
    _, _, meta, dt = _run(hx, tmp, monkeypatch, "extra")
    assert meta["stop"] == "unexpected_tools" and "WebFetch" in meta["unexpected_tools"]
    assert dt < 30


def test_wall_clock_kill(iso, monkeypatch):
    hx, tmp = iso
    _, _, meta, dt = _run(hx, tmp, monkeypatch, "hang", wall=3)
    assert meta["stop"] == "wall_clock" and dt < 30


def test_bwrap_argv_has_no_secrets_and_no_home(iso):
    hx, tmp = iso
    ep = sandbox.prepare("_demo", hx.insts["null"][0], "full", hx.run_dir, solver_label="argv")
    state = ca._state_dir(ep["episode"])
    argv = ca.bwrap_argv(os.path.realpath(ep["sandbox"]), state, ["true"])
    joined = " ".join(argv)
    home = os.path.expanduser("~")
    for bad in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "--share-net", ".credentials"):
        assert bad not in joined
    # every bind source is the sandbox, the venv, the broker socket, the claude binary, /usr, /etc files or state
    srcs = [argv[i + 1] for i, a in enumerate(argv) if a in ("--bind", "--ro-bind")]
    allowed = (os.path.realpath(ep["sandbox"]), ca.paths.venv_dir(), ca.paths.broker_sock(), state, "/usr", "/etc/",
               os.path.realpath(ca.CLAUDE_BIN))
    assert all(s.startswith(allowed) for s in srcs), srcs
    assert not any(s in (home, home + "/") for s in srcs)
    shutil.rmtree(state)


def test_egress_proxy_allowlist(tmp_path):
    short = tempfile.mkdtemp(prefix="rlca", dir="/tmp")
    sock = os.path.join(short, "p.sock")
    log = str(tmp_path / "net.jsonl")
    px = ca.EgressProxy(sock, log, allowed={("127.0.0.1", 9)}).start()   # allowed target that refuses: 502
    try:
        def ask(req):
            s = socket.socket(socket.AF_UNIX)
            s.connect(sock)
            s.sendall(req)
            r = s.recv(200)
            s.close()
            return r
        assert b"403" in ask(b"CONNECT huggingface.co:443 HTTP/1.1\r\n\r\n")
        assert b"403" in ask(b"GET http://169.254.169.254/ HTTP/1.1\r\nHost: x\r\n\r\n")
        assert b"403" in ask(b"CONNECT api.anthropic.com:80 HTTP/1.1\r\n\r\n")
        assert b"502" in ask(b"CONNECT 127.0.0.1:9 HTTP/1.1\r\n\r\n")
    finally:
        px.shutdown()
        px.server_close()
        shutil.rmtree(short, ignore_errors=True)
    evs = [json.loads(l) for l in open(log)]
    assert [e["allowed"] for e in evs if "allowed" in e] == [False, False, False, True]
    assert ca.ALLOWED_HOSTS == {("api.anthropic.com", 443)}


def test_redteam_in_test_broker(iso):
    """The scripted red-team against the isolated test broker: every probe must stay inside."""
    hx, tmp = iso
    ep = sandbox.prepare("_demo", hx.insts["null"][0], "full", hx.run_dir, solver_label="redteam")
    decoy = sandbox.prepare("_demo", hx.insts["planted"][0], "full", hx.run_dir, solver_label="decoy")
    args = argparse.Namespace(episode=ep["episode"], decoy=decoy["episode"], out=str(tmp / "rt"), control=False)
    rc = ca.redteam(args)
    rep = json.load(open(tmp / "rt" / "redteam.json"))
    assert rc == 0 and rep["n_escaped"] == 0, rep["escaped"]
    assert rep["n_probes"] > 60


def test_auth_proxy_path_allowlist_and_cap():
    """AuthProxy: only /v1/messages is forwarded; the request cap answers 429 before any upstream call; the injected
    header callable is never echoed to the jail."""
    import http.client
    short = tempfile.mkdtemp(prefix="rlca", dir="/tmp")
    sock = os.path.join(short, "a.sock")
    log = os.path.join(short, "a.jsonl")
    px = ca.AuthProxy(sock, log, lambda: {"Authorization": "Bearer never-sent"}, max_requests=0).start()

    class UC(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX)
            self.sock.connect(sock)
    try:
        for method, path, want in (("GET", "/api/oauth/profile", 403), ("GET", "/v1/models", 403),
                                   ("POST", "/v1/messages/../models", 403), ("DELETE", "/v1/messages", 405),
                                   ("POST", "/v1/messages", 429), ("POST", "/v1/messages?beta=true", 429)):
            c = UC("jail")
            c.request(method, path, body="{}", headers={"Authorization": "Bearer jailed"})
            r = c.getresponse()
            body = r.read()
            assert r.status == want, (method, path, r.status)
            assert b"never-sent" not in body
            c.close()
    finally:
        px.shutdown()
        px.server_close()
        shutil.rmtree(short, ignore_errors=True)
    assert ca.API_PATH_RX.match("/v1/messages/count_tokens?beta=true")
    assert not ca.API_PATH_RX.match("/v1/messages/batches")
