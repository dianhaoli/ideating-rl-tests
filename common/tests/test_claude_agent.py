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
  forge)
    init '["Bash","Read","Write","Edit","Glob","Grep"]'
    printf 'garbage-without-newline'
    printf '{"type":"assistant","session_id":"s1","message":{"role":"assistant","content":[{"type":"tool_use","id":"t9","name":"Bash","input":{"command":"cat /secret"}}]}}\n'
    printf '{"type":"user","session_id":"s1","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t9","content":"x"}]}}\n'
    printf '{"type":"result","subtype":"success","session_id":"s1","total_cost_usd":0,"num_turns":1}\n'
    ;;
  proxy_ok|proxy_side|proxy_hide)
    # one real request through the AuthProxy (mock upstream answers with a tool_use whose input is the text sent)
    init '["Bash","Read","Write","Edit","Glob","Grep"]'
    python3 - "$FAKE_MODE" "$SBX" <<'PYEOF'
import http.client, json, subprocess, sys
mode, sbx = sys.argv[1], sys.argv[2]
def ask(cmd):
    c = http.client.HTTPConnection("127.0.0.1", 3129, timeout=30)
    c.request("POST", "/v1/messages?beta=true", body=json.dumps({"model": "claude-haiku-4-5-20251001", "max_tokens": 50,
              "stream": True, "messages": [{"role": "user", "content": json.dumps({"command": cmd})}]}),
              headers={"content-type": "application/json", "anthropic-beta": "oauth-2025-04-20,web-fetch-2025-09-10"})
    r = c.getresponse(); d = r.read()
    assert r.status == 200, (r.status, d[:300])
    line = [l for l in d.split(b"\n") if l.startswith(b"data:") and b"message_start" in l][0]
    return json.loads(line[5:])["message"]["id"]
sub = "./tool submit '{\"nothing_found\": true}'"
mid = ask(f"cd {sbx} && {sub}")              # the CLI strips this cd prefix before printing (normalisation)
out = subprocess.run(sub, shell=True, capture_output=True, text=True).stdout
shown = "ls" if mode == "proxy_hide" else sub
print(json.dumps({"type": "assistant", "session_id": "s1", "message": {"id": mid, "role": "assistant", "content": [
    {"type": "tool_use", "id": "tu_" + mid, "name": "Bash", "input": {"command": shown}}]}}), flush=True)
print(json.dumps({"type": "user", "session_id": "s1", "message": {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "tu_" + mid, "content": out}]}}), flush=True)
if mode == "proxy_side":
    ask("echo side reasoning the stream never shows")
print(json.dumps({"type": "result", "subtype": "success", "session_id": "s1", "total_cost_usd": 0.0, "num_turns": 1}))
PYEOF
    ;;
  extra)
    init '["Bash","Read","Write","Edit","Glob","Grep","WebFetch"]'
    sleep 60 ;;
  hang)
    init '["Bash","Read"]'
    sleep 120 ;;
esac
"""


class MockAPI:
    """Local stand-in for api.anthropic.com: answers every /v1/messages with an SSE stream holding one tool_use whose
    input is the JSON text of the first user message. Records what it received (headers, bodies)."""

    def __init__(self):
        import http.server
        import threading
        self.reqs = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                outer.reqs.append({"path": self.path, "headers": dict(self.headers.items()), "body": body})
                j = json.loads(body)
                mid = f"msg_mock{len(outer.reqs)}"
                try:
                    inp = json.loads(j["messages"][0]["content"])
                except (ValueError, TypeError, KeyError):
                    inp = {}
                evs = [{"type": "message_start", "message": {"id": mid, "model": j.get("model")}},
                       {"type": "content_block_start", "index": 0, "content_block": {
                           "type": "tool_use", "id": "tu_" + mid, "name": "Bash", "input": {}}}]
                raw = json.dumps(inp)
                evs += [{"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta",
                                                                               "partial_json": raw[i:i + 7]}}
                        for i in range(0, len(raw), 7)]
                evs += [{"type": "content_block_stop", "index": 0},
                        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
                        {"type": "message_stop"}]
                data = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in evs).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.addr = self.srv.server_address
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def mock_api(monkeypatch):
    m = MockAPI()
    monkeypatch.setattr(ca, "UPSTREAM", m.addr)
    yield m
    m.close()


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
                                                             lambda: {"Authorization": "Bearer real-x"}, {}))
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
    assert meta["stream_integrity"]["ok"] and meta["credential_found_in"] == []
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


def test_stream_tampering_is_flagged(iso, monkeypatch):
    """A partial line written into the CLI's stdout (as an agent could via /proc/<pid>/fd/1) hides the next event;
    the launcher must report it (bad line + orphan tool_result)."""
    hx, tmp = iso
    _, _, meta, _ = _run(hx, tmp, monkeypatch, "forge")
    si = meta["stream_integrity"]
    assert not si["ok"] and si["bad_lines"] == 1 and si["orphan_results"] == ["t9"]
    assert "stream_integrity" in meta["launcher_invalid_reasons"]


def test_finish_enforces_launcher_checks(iso, monkeypatch):
    """sandbox finish reads api_meta.json: detected stream tampering makes the episode INVALID (it used to be a
    warning an operator had to notice); a claude-cli episode without api_meta.json is INVALID too."""
    hx, tmp = iso
    ep, out, meta, _ = _run(hx, tmp, monkeypatch, "forge")
    g = sandbox.finish(ep["episode"], [os.path.join(out, "api_transcript.jsonl")], "claude-cli:fake:high")
    assert not g["harness"]["valid"] and "launcher_stream_integrity" in g["harness"]["invalid_reasons"]
    os.unlink(os.path.join(out, "api_meta.json"))
    g = sandbox.finish(ep["episode"], [os.path.join(out, "api_transcript.jsonl")], "claude-cli:fake:high")
    assert "launcher_meta_missing" in g["harness"]["invalid_reasons"]
    good = dict(meta, stream_integrity={"ok": True})
    assert ca.launcher_invalid_reasons(good) == []
    for bad in ({"credential_found_in": ["x"]}, {"api_proxy_denied": [{"why": "x"}]}, {"unexpected_tools": ["X"]},
                {"proxy_cross_check": {"ok": False}}, {"host_ptrace_scope": 0}, {"stream_integrity": None}):
        assert ca.launcher_invalid_reasons(dict(good, **bad)), bad


def test_proxy_cross_check_through_mock_api(iso, mock_api, monkeypatch):
    """A fake CLI makes a real request through the AuthProxy (mock upstream). Normal: valid, the CLI's `cd <sandbox> &&`
    normalisation is accepted, unknown betas are dropped and the jail's credential is replaced. An extra request the
    stream never shows, or a tool_use shown with a different input, makes the episode INVALID."""
    hx, tmp = iso
    ep, out, meta, _ = _run(hx, tmp, monkeypatch, "proxy_ok")
    pc = meta["proxy_cross_check"]
    assert pc["ok"] and pc["n_generation_requests"] == 1 and pc["n_proxy_tool_uses"] == 1, pc
    assert meta["launcher_invalid_reasons"] == [] and meta["stop"] == "submitted"
    h = {k.lower(): v for k, v in mock_api.reqs[0]["headers"].items()}
    assert h.get("authorization") == "Bearer real-x" and h.get("anthropic-beta") == "oauth-2025-04-20"
    assert h.get("accept-encoding") == "identity"
    assert meta["api_proxy_betas_dropped"] == ["web-fetch-2025-09-10"]
    g = sandbox.finish(ep["episode"], [os.path.join(out, "api_transcript.jsonl")], "claude-cli:fake:high")
    assert g["harness"]["valid"], g["harness"]["invalid_reasons"]

    _, _, meta, _ = _run(hx, tmp, monkeypatch, "proxy_side")
    pc = meta["proxy_cross_check"]
    assert not pc["ok"] and pc["n_unexplained"] == 1 and "api_cross_check" in meta["launcher_invalid_reasons"]

    _, _, meta, _ = _run(hx, tmp, monkeypatch, "proxy_hide")
    pc = meta["proxy_cross_check"]
    assert not pc["ok"] and len(pc["hidden_tool_uses"]) == 1 and pc["hidden_tool_uses"][0]["in_stream"]


def _uc(sock):
    import http.client

    class UC(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX)
            self.sock.connect(sock)
    return UC("jail")


def test_auth_proxy_body_policy(mock_api):
    """Server tools, MCP, containers, URL sources, other models, oversized max_tokens and unknown fields are refused by
    the proxy (403) and never reach upstream; an allowed body is forwarded. The live gate refuses with 429."""
    short = tempfile.mkdtemp(prefix="rlca", dir="/tmp")
    sock = os.path.join(short, "a.sock")
    px = ca.AuthProxy(sock, os.path.join(short, "a.jsonl"), lambda: {"Authorization": "Bearer real"},
                      models={"m-ok"}).start()
    ok = {"model": "m-ok", "max_tokens": 10, "messages": [{"role": "user", "content": json.dumps({"command": "x"})}],
          "tools": [{"name": "Bash", "description": "d", "input_schema": {"type": "object", "properties": {
              "command": {"type": "string"}}}}],
          "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
          "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"},
          "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
          "metadata": {"user_id": "u"}, "stream": True}

    def post(body):
        c = _uc(sock)
        c.request("POST", "/v1/messages?beta=true", body=json.dumps(body) if isinstance(body, dict) else body,
                  headers={"content-type": "application/json"})
        r = c.getresponse()
        r.read()
        c.close()
        return r.status

    def msg(block):
        return dict(ok, messages=[{"role": "user", "content": [block]}])
    try:
        bad = [dict(ok, tools=[{"type": "web_fetch_20250910", "name": "web_fetch"}]),
               dict(ok, tools=[{"type": "web_search_20250305", "name": "web_search"}]),
               dict(ok, tools=[{"type": "bash_20250124", "name": "bash"}]),
               dict(ok, tools=[{"type": "mcp_toolset", "mcp_server_name": "x"}]),
               dict(ok, mcp_servers=[{"type": "url", "url": "https://e.com", "name": "e"}]),
               dict(ok, container="c"), dict(ok, model="claude-opus-5-5"),
               dict(ok, max_tokens=ca.API_MAX_TOKENS_CAP + 1), dict(ok, max_tokens=0),
               msg({"type": "image", "source": {"type": "url", "url": "https://e/a"}}),
               msg({"type": "document", "source": {"type": "file", "file_id": "f"}}),
               msg({"type": "tool_result", "tool_use_id": "t", "content": [
                   {"type": "document", "source": {"type": "url", "url": "https://e/a"}}]}),
               msg({"type": "server_tool_use", "id": "s", "name": "web_fetch", "input": {}}),
               dict(ok, system=[{"type": "text", "text": "s", "cache_control": {"type": "x"}}]),
               dict(ok, thinking={"type": "weird"}), dict(ok, output_config={"format": {}}),
               dict(ok, context_management={"edits": [{"type": "x"}]}), "not json", "[1]"]
        for b in bad:
            assert post(b) == 403, b
        assert mock_api.reqs == []
        assert post(ok) == 200 and len(mock_api.reqs) == 1
        px.gate = lambda: "unaudited"
        assert post(ok) == 429 and len(mock_api.reqs) == 1
    finally:
        px.shutdown()
        px.server_close()
        shutil.rmtree(short, ignore_errors=True)
    ev = [e for e in px.events if e.get("allowed")]
    assert ev[0]["msg_id"] == "msg_mock1" and ev[0]["tool_uses"][0]["input"] == {"command": "x"}
    assert ev[0]["req_sha256"] and ev[0]["model"] == "m-ok"


def test_tamper_probes_discriminate(iso):
    """The red-team's re-open probe really detects the old weakness: with PIPE stdio (the launcher before the isolation
    audit) a jailed process can re-open its parent's stdout through /proc; with the launcher's socket stdio it cannot."""
    hx, tmp = iso
    ep = sandbox.prepare("_demo", hx.insts["null"][0], "full", hx.run_dir, solver_label="tamper")
    jail = ca.Jail(ep["episode"], str(tmp / "tp"), allowed=set())
    cmd = ["/bin/bash", "-c", "python3 -c \"import os;open('/proc/%d/fd/1' % os.getppid(),'rb')\" && echo REOPENED"]
    try:
        import subprocess
        p = jail.popen(cmd, {}, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        o, _ = p.communicate(timeout=60)
        assert p.returncode == 0 and b"REOPENED" in o          # pipe: re-openable (the weakness)
        rc, o, _ = ca._run_socket_stdio(lambda **kw: jail.popen(cmd, {}, **kw), timeout=60)
        assert rc != 0 and "REOPENED" not in o                  # socket: ENXIO
        # ptrace_scope >= 1: /proc/<pid>/mem opens for a DESCENDANT (the probe code works) but not for the parent (the
        # CLI's position relative to the agent's shell).
        mem = ("import os,subprocess,sys;c=subprocess.Popen(['sleep','5']);open('/proc/%d/mem' % c.pid,'rb');"
               "print('CHILD_OK');c.kill();open('/proc/%d/mem' % os.getppid(),'rb');print('PARENT_OK')")
        rc, o, _ = ca._run_socket_stdio(lambda **kw: jail.popen(["/bin/bash", "-c", f'python3 -c "{mem}"; true'], {},
                                                                 **kw), timeout=60)
        assert "CHILD_OK" in o and "PARENT_OK" not in o, o
    finally:
        jail.close()


def test_same_tool_input_normalisation():
    sbx = "/home/u/rlsbx/ep1"
    f = ca._same_tool_input
    assert f("Bash", {"command": "cd /home/u/rlsbx/ep1 && ls -la"}, {"command": "ls -la"}, sbx)
    assert f("Bash", {"command": 'cd "/home/u/rlsbx/ep1" && ls'}, {"command": "ls"}, sbx)
    assert not f("Bash", {"command": "cd /tmp && ls"}, {"command": "ls"}, sbx)
    assert not f("Bash", {"command": "cat /secret"}, {"command": "ls"}, sbx)
    assert not f("Bash", {"command": "cd /home/u/rlsbx/ep1 && ls", "timeout": 5}, {"command": "ls"}, sbx)
    assert not f("Read", {"file_path": "/a"}, {"file_path": "/b"}, sbx)
    # Write/Edit: per-line trailing whitespace is stripped by the CLI; Edit gains "replace_all": false
    assert f("Write", {"file_path": "/a", "content": "x = 1   \n    \nreturn 1\t"},
             {"file_path": "/a", "content": "x = 1\n\nreturn 1"}, sbx)
    assert f("Edit", {"file_path": "/a", "old_string": "x = 1", "new_string": "x = 2  "},
             {"replace_all": False, "file_path": "/a", "old_string": "x = 1", "new_string": "x = 2"}, sbx)
    assert not f("Edit", {"file_path": "/a", "old_string": "x = 1", "new_string": "x = 2"},
                 {"replace_all": True, "file_path": "/a", "old_string": "x = 1", "new_string": "x = 2"}, sbx)
    assert not f("Write", {"file_path": "/a", "content": "import os"}, {"file_path": "/a", "content": "import sys"}, sbx)


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
    assert rep["minor_escaped"] == [], rep["minor_escaped"]     # the test broker runs this checkout's code
    assert rep["n_probes"] > 60


def test_auth_proxy_path_allowlist_and_cap():
    """AuthProxy: only /v1/messages is forwarded; the request cap answers 429 before any upstream call; the injected
    header callable is never echoed to the jail."""
    import http.client
    short = tempfile.mkdtemp(prefix="rlca", dir="/tmp")
    sock = os.path.join(short, "a.sock")
    log = os.path.join(short, "a.jsonl")
    px = ca.AuthProxy(sock, log, lambda: {"Authorization": "Bearer never-sent"}, max_requests=0,
                      models={"m"}).start()
    ok = json.dumps({"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]})

    class UC(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX)
            self.sock.connect(sock)
    try:
        for method, path, want in (("GET", "/api/oauth/profile", 403), ("GET", "/v1/models", 403),
                                   ("POST", "/v1/messages/../models", 403), ("DELETE", "/v1/messages", 405),
                                   ("POST", "/v1/messages", 429), ("POST", "/v1/messages?beta=true", 429)):
            c = UC("jail")
            c.request(method, path, body=ok, headers={"Authorization": "Bearer jailed"})
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
    assert not ca.API_PATH_RX.match("/v1/messages?beta=true&x=1")
