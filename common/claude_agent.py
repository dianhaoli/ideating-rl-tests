"""OS-isolated headless Claude Code test agent for one harness episode (twin of common/openai_agent.py).

Why this exists (2026-10-04, ShiftHunt forensics, docs/forensics/SHIFTHUNT_FORENSICS.md): Claude test agents used to be
Claude Code subagents of the orchestrator session. They ran with the orchestrator's cwd (the repo), its user config
(CLAUDE.md, memories, MCP servers, skills) and full read access to the home directory, so the answer key was three
plain file reads away and containment rested on the after-the-fact transcript audit alone. This launcher runs
`claude -p` inside a bubblewrap jail in which those things do not exist at all (not merely unreadable):

  Filesystem (new mount namespace, root is an empty tmpfs). Visible inside, and nothing else:
    <sandbox root>/<E>/            the episode sandbox, read-write, at its REAL path (the agent prompt names that path)
    <sandbox root>/.broker.sock    the broker socket (./tool), same path as on the host
    <venv>                         the analysis venv behind ./py (numpy/scipy/sklearn, no torch), read-only
    /usr (+ /bin /lib /lib64 /sbin symlinks)   read-only system runtime (bash, coreutils, python3, socat)
    /etc: generated passwd/group/hosts + read-only nsswitch.conf, ld.so.cache, localtime, ssl/pki certs
    /opt/claude/claude             the Claude Code binary, read-only
    /home/agent                    HOME, an empty private dir; /home/agent/.claude is a FRESH CLAUDE_CONFIG_DIR
                                   (settings.json written here; no CLAUDE.md, memories, MCP, skills, transcripts)
    /run/rl                        the network proxy socket and the in-jail start script, read-only
    /proc (own pid namespace only), minimal /dev, private tmpfs /tmp
  So the home directory, ~/ideating-rl-tests, ~/wt, ~/.claude (projects, credentials), ~/hf_home, key files, other
  sandboxes, every task's instances and caches, the episode registry, /opt/pytorch and the GPU device nodes are absent.
  Processes: own pid namespace (host processes are not in /proc), own ipc/uts/cgroup namespaces, no nested user
  namespaces (--disable-userns), --die-with-parent, --new-session (no TIOCSTI into the launcher's terminal).
  Network: own network namespace (loopback only). The ONLY way out is an HTTP CONNECT proxy that this launcher runs
  outside the jail on a Unix socket (bind-mounted at /run/rl/proxy.sock, bridged to 127.0.0.1:3128 inside by socat).
  It allows CONNECT to api.anthropic.com:443 and nothing else; every attempt is logged to <out>/net_log.jsonl.
  This also blocks the EC2 metadata service, Hugging Face and pip, which a shared network namespace would not.
  Claude Code layer (defense in depth, inside the jail): --tools Bash,Read,Write,Edit,Glob,Grep only;
  --permission-mode dontAsk with allow rules scoped to the sandbox and deny rules for WebFetch, WebSearch,
  Agent/Task, MCP, and Read/Edit/Write on every other tree mounted in the jail (/etc /usr /proc /dev /run /opt /tmp
  /var, HOME with the config dir, the venv) -- ADVISORY only: Bash can read whatever the jail mounts, so the jail
  (layer 1) is the real control; settings.json is bind-mounted read-only (disableAllHooks); --strict-mcp-config with
  no servers; --safe-mode;
  --disable-slash-commands; nonessential traffic, telemetry and the auto-updater disabled; Bash cwd reset to the
  sandbox after every command (so the transcript's recorded cwd is always the true cwd for the audit).
  The tool list Claude Code reports at start-up (stream-json `system/init`) is checked against that set; any extra
  tool stops the episode (stop = "unexpected_tools").

Authentication (default --auth oauth, token injection): the jailed CLI gets a DUMMY OAuth token and
ANTHROPIC_BASE_URL=http://127.0.0.1:3129, bridged by socat to /run/rl/api.sock. Behind that socket, OUTSIDE the jail,
AuthProxy replaces the dummy with the operator's real Claude Code access token (re-read from ~/.claude/.credentials.json
for every request, so the operator session's refreshes are picked up) and forwards over TLS to api.anthropic.com. It
forwards only POST /v1/messages[/count_tokens] whose BODY passes check_request (2026-10-04 isolation audit: the API
itself can reach the web through server tools): this episode's model only; max_tokens <= 128000; only client tools
(no server tools such as web_search/web_fetch/code_execution, no mcp_servers, no container); only inline content
(no url/file sources); every top-level field, block type and anthropic-beta value from an allowlist (unknown betas are
dropped, anything else is refused with 403). It logs every request (body sha256 and size, model, message id, output
tokens, the tool_use blocks of the response) to <out>/api_proxy_log.jsonl, refuses new requests while >= 2 answered
ones are missing from the CLI's stream (the agent's own requests), and run() cross-checks the proxy's record against
the stream (proxy_cross_check). No credential file, access token or refresh token ever enters the jail (a refresh
inside would also rotate the operator's refresh token and log the orchestrator out). With injection, the CONNECT
allowlist is EMPTY: no other host is reachable from the jail.
--auth apikey does the same with ~/.anthropic_env's key (real money: the shared API ledger is checked and charged).
--auth oauth-env is the fallback without injection: the real access token sits in the jailed CLI's environment, where
the agent can read it (/proc/<pid>/environ, verified live), and CONNECT to api.anthropic.com:443 is allowed.
That mode has no body policy and no cross-check; finish marks its episodes INVALID (unfiltered_api_access).
Residual with injection: the agent can send at most ~2 of its own /v1/messages requests (same model, plain text and
client tools only) before the proxy refuses; finish marks such an episode INVALID (api_cross_check). After every run
the outputs and the sandbox are scanned for the real credential (api_meta.json: credential_found_in).
Tamper resistance: the CLI's stdout/stderr are sockets (cannot be re-opened via /proc/<pid>/fd), the host must run
kernel.yama.ptrace_scope >= 1 (run() refuses otherwise; the agent cannot ptrace or write the memory of the CLI, its
ancestor), and `sandbox finish` turns every launcher check in api_meta.json into INVALID reasons
(launcher_invalid_reasons).

Usage (after `python -m common.sandbox prepare ...` printed episode E):
    python -m common.claude_agent run --episode E --task T --prompt-file <run>/episodes/E/agent_prompt.txt \
        --out <run>/episodes/E --model haiku --effort high --max-turns 60 --max-wall-s 3600
    python -m common.sandbox finish --episode E --transcript <run>/episodes/E/api_transcript.jsonl \
        --agent-model claude-cli:claude-haiku-4-5-20251001:high
    python -m common.claude_agent redteam --episode E [--decoy E2]   # scripted isolation checks (no LLM)
    python -m common.claude_agent spent                               # CLI-reported cost ledger
"""
import argparse
import base64
import fcntl
import json
import os
import re
import secrets
import select
import shlex
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import paths  # noqa: E402

BWRAP = "/usr/bin/bwrap"
SOCAT = "/usr/bin/socat"
CLAUDE_BIN = os.path.expanduser(os.environ.get("RL_CLAUDE_BIN", "~/.local/bin/claude"))
CREDS = os.path.expanduser("~/.claude/.credentials.json")
API_KEY_FILE = os.path.expanduser("~/.anthropic_env")
STATE_ROOT = os.path.expanduser(os.environ.get("RL_CLAUDE_STATE", "~/.rl_claude_agent"))
LEDGER_DIR = os.path.expanduser(os.environ.get("RL_API_LEDGER_DIR", "~/.rl_api"))
CLI_LEDGER = os.path.join(LEDGER_DIR, "claude_cli_ledger.jsonl")

MODELS = {"opus": "claude-opus-5-5", "haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-5-5"}
EFFORTS = ("low", "medium", "high", "xhigh", "max")
ALLOWED_HOSTS = {("api.anthropic.com", 443)}
AGENT_TOOLS = ("Bash", "Read", "Write", "Edit", "Glob", "Grep")
DENY_TOOLS = ("WebFetch", "WebSearch", "Agent", "Task", "NotebookEdit", "TodoWrite")

IN_HOME = "/home/agent"
IN_CFG = IN_HOME + "/.claude"
IN_RUN = "/run/rl"
IN_CLAUDE = "/opt/claude/claude"
PROXY_PORT = 3128
API_PORT = 3129

REDTEAM_MODEL = MODELS["haiku"]     # the model the red-team's proxy is pinned to
NUDGE = "You have not submitted yet. Continue working, or call ./tool submit with your final answer."


# ---------------------------------------------------------------------------------------------- proxy (outside the jail)
class _ProxyHandler(socketserver.BaseRequestHandler):
    """One CONNECT tunnel. Only hosts in server.allowed are tunnelled; everything else gets 403 and is logged."""

    def handle(self):
        conn = self.request
        conn.settimeout(30)
        buf = b""
        try:
            while b"\r\n\r\n" not in buf and len(buf) < 16384:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buf += chunk
        except OSError:
            return
        head, _, rest = buf.partition(b"\r\n\r\n")
        line = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        m = re.match(r"CONNECT ([A-Za-z0-9.\-]+):(\d+) HTTP/1\.[01]$", line)
        host, port = (m.group(1).lower(), int(m.group(2))) if m else (None, None)
        ok = (host, port) in self.server.allowed
        self.server.log({"t": time.time(), "request": line[:200], "host": host, "port": port, "allowed": ok})
        if not ok:
            try:
                conn.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            return
        try:
            up = socket.create_connection((host, port), timeout=30)
        except OSError as e:
            self.server.log({"t": time.time(), "host": host, "port": port, "error": str(e)[:200]})
            try:
                conn.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            return
        try:
            conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            if rest:
                up.sendall(rest)
            conn.settimeout(None)
            up.settimeout(None)
            socks = [conn, up]
            while True:
                r, _, x = select.select(socks, [], socks, 600)
                if x or not r:
                    break
                done = False
                for s in r:
                    data = s.recv(65536)
                    if not data:
                        done = True
                        break
                    (up if s is conn else conn).sendall(data)
                if done:
                    break
        except OSError:
            pass
        finally:
            up.close()


class EgressProxy(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, sock_path, log_path, allowed=ALLOWED_HOSTS):
        self.allowed = set(allowed)
        self._log_path = log_path
        self._lock = threading.Lock()
        self.events = []
        super().__init__(sock_path, _ProxyHandler)
        os.chmod(sock_path, 0o600)

    def log(self, ev):
        with self._lock:
            self.events.append(ev)
            if self._log_path:
                with open(self._log_path, "a") as f:
                    f.write(json.dumps(ev) + "\n")

    def start(self):
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self

    def handle_error(self, request, client_address):
        # A client (the jail) that hangs up mid-request is not a launcher error: record it, print no traceback.
        self.log({"t": time.time(), "client_error": type(sys.exc_info()[1]).__name__})


# ---------------------------------------------------------------------------------------------- auth-injecting proxy
# WHY (2026-10-04): with the token in the jailed process's environment, the agent can read it from /proc/<pid>/environ
# (verified live: /proc/1/environ inside the jail lists CLAUDE_CODE_OAUTH_TOKEN). Here the jail gets a DUMMY token and
# ANTHROPIC_BASE_URL=http://127.0.0.1:3129; this reverse proxy outside the jail swaps the dummy for the real credential
# and forwards over TLS to api.anthropic.com. The real token never enters the jail, and only allowlisted API paths are
# forwarded (an agent can still spend inference through the proxy while the episode runs; it cannot take the token).
DUMMY_OAUTH = "sk-ant-oat01-" + "jailed-dummy-token-" + "0" * 60
API_HOST = "api.anthropic.com"
API_PATH_RX = re.compile(r"^/v1/messages(/count_tokens)?(\?beta=true)?$")    # the CLI sends ?beta=true
HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer", "upgrade",
       "proxy-authorization", "proxy-authenticate", "host", "content-length", "accept-encoding"}

# ---- request body policy (2026-10-04 isolation audit, findings 1 and 2)
# WHY: the proxy used to forward ANY /v1/messages body. Live from inside the jail, a request with the server tool
# web_fetch_20250910 fetched a public GitHub page (and web_search_20250305 searched the web), and a request naming
# claude-opus-5-5 was served during a Haiku episode. Server-side tools, MCP connectors, containers and URL-sourced
# image/document blocks all make Anthropic's servers reach the internet on the agent's behalf, so every body is now
# parsed and checked against an ALLOWLIST (fail closed: an unknown field, block type or tool type is refused with 403,
# which `sandbox finish` turns into INVALID api_proxy_denied). The allowlists are what Claude Code 2.1.288 sends in
# -p mode (captured through this proxy, 2026-10-04); a CLI update that sends something new fails visibly, not silently.
API_TOP_KEYS = frozenset({"model", "messages", "system", "tools", "tool_choice", "max_tokens", "metadata", "stream",
                          "thinking", "temperature", "top_p", "top_k", "stop_sequences", "context_management",
                          "output_config"})
API_BLOCK_TYPES = frozenset({"text", "image", "document", "tool_use", "tool_result", "thinking", "redacted_thinking"})
API_SOURCE_TYPES = frozenset({"base64", "text", "content"})            # never "url" or "file" (server-side fetch)
API_THINKING_TYPES = frozenset({"enabled", "disabled", "adaptive"})
API_TOOL_CHOICE_TYPES = frozenset({"auto", "any", "tool", "none"})
API_CTX_EDIT_RX = re.compile(r"^clear_(tool_uses|thinking)_\d{8}$")      # context editing: server-side trimming only
# anthropic-beta values kept on the forwarded request; every other value is dropped (and logged).
# Exactly the values Claude Code 2.1.288 sent in the capture runs (haiku and opus, 2026-10-04).
API_BETA_ALLOW = frozenset({"oauth-2025-04-20", "claude-code-20250219", "interleaved-thinking-2025-05-14",
                            "thinking-token-count-2026-05-13", "context-management-2025-06-27",
                            "prompt-caching-scope-2026-01-05", "thinking-display-updates-2026-08-18",
                            "extended-cache-ttl-2025-04-11", "effort-2025-11-24", "mid-conversation-system-2026-04-07",
                            "per-turn-control-2026-07-01", "mid-conversation-tool-changes-2026-07-01"})
# The CLI's own per-request maximum (Opus 5.5: 128000; Haiku 4.5: 32000). What bounds an agent's OWN requests is the
# live unaudited-request gate (AuthProxy.gate: at most UNAUDITED_SLACK answered requests ahead of the CLI's stream)
# plus the request cap; this cap only stops a single request from asking for more than the CLI ever does.
API_MAX_TOKENS_CAP = 128000
UNAUDITED_SLACK = 2
UPSTREAM = (API_HOST, 443)    # tests point this at a local mock API (plain HTTP when the host is not API_HOST)


def _check_blocks(x, where, bad):
    """Walk message/system content. Every dict with a "type" must be an allowlisted block type; every "source" must be
    inline (base64/text/content). tool_use inputs are opaque (client-executed tool arguments) and not walked."""
    if isinstance(x, list):
        for i, v in enumerate(x):
            _check_blocks(v, f"{where}[{i}]", bad)
        return
    if not isinstance(x, dict):
        return
    t = x.get("type")
    if t is not None and t not in API_BLOCK_TYPES:
        bad.append(f"{where}: block type {str(t)[:40]!r} not allowed")
    for k, v in x.items():
        if k == "input" and t == "tool_use":
            continue
        if k == "cache_control":
            if not (isinstance(v, dict) and v.get("type") == "ephemeral"):
                bad.append(f"{where}.cache_control not allowed")
            continue
        if k == "citations":
            continue                   # citation metadata echoed back; makes no server-side request
        if k == "source":
            st = v.get("type") if isinstance(v, dict) else None
            if st not in API_SOURCE_TYPES:
                bad.append(f"{where}.source type {str(st)[:40]!r} not allowed (no url/file sources)")
            elif st == "content":
                _check_blocks(v.get("content"), where + ".source.content", bad)
            continue
        if isinstance(v, (list, dict)):
            _check_blocks(v, f"{where}.{k}", bad)


def check_request(path, body, policy):
    """(ok, reasons, summary) for one proxied request body. policy: {"models": set, "max_tokens": int}."""
    import hashlib
    summ = {"req_bytes": len(body or b""), "req_sha256": hashlib.sha256(body or b"").hexdigest()[:16]}
    try:
        j = json.loads(body or b"")
    except ValueError:
        return False, ["body is not JSON"], summ
    if not isinstance(j, dict):
        return False, ["body is not a JSON object"], summ
    bad = []
    summ.update(model=j.get("model"), max_tokens=j.get("max_tokens"),
                n_tools=len(j["tools"]) if isinstance(j.get("tools"), list) else None, stream=j.get("stream"))
    extra = sorted(k for k in j if k not in API_TOP_KEYS)
    if extra:
        bad.append(f"top-level fields not allowed: {extra[:8]}")    # e.g. mcp_servers, container
    if j.get("model") not in policy["models"]:
        bad.append(f"model {str(j.get('model'))[:60]!r} is not this episode's model")
    if "count_tokens" not in path:
        mt = j.get("max_tokens")
        if not isinstance(mt, int) or isinstance(mt, bool) or mt < 1 or mt > policy["max_tokens"]:
            bad.append(f"max_tokens must be an int in [1, {policy['max_tokens']}]")
    tools = j.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            bad.append("tools is not a list")
        else:
            for i, tl in enumerate(tools):
                # Client tools carry name/description/input_schema and no "type" (or "custom"). Any other type is a
                # server tool (web_search_*, web_fetch_*, code_execution_*, bash_*, text_editor_*, computer_*,
                # memory_*, mcp_toolset, tool_search_*) executed on Anthropic's side.
                if not isinstance(tl, dict) or tl.get("type", "custom") != "custom":
                    bad.append(f"tools[{i}]: server tool {str(tl.get('type') if isinstance(tl, dict) else tl)[:40]!r}"
                               " not allowed")
                elif not isinstance(tl.get("input_schema"), dict):
                    bad.append(f"tools[{i}]: client tool without input_schema")
    th = j.get("thinking")
    if th is not None and not (isinstance(th, dict) and th.get("type") in API_THINKING_TYPES):
        bad.append("thinking config not allowed")
    tc = j.get("tool_choice")
    if tc is not None and not (isinstance(tc, dict) and tc.get("type") in API_TOOL_CHOICE_TYPES):
        bad.append("tool_choice not allowed")
    cm = j.get("context_management")
    if cm is not None:
        edits = cm.get("edits") if isinstance(cm, dict) else None
        if not isinstance(edits, list) or set(cm) - {"edits"} or not all(
                isinstance(e, dict) and API_CTX_EDIT_RX.match(str(e.get("type"))) for e in edits):
            bad.append("context_management not allowed")
    oc = j.get("output_config")
    if oc is not None and not (isinstance(oc, dict) and set(oc) <= {"effort"}):
        bad.append(f"output_config fields not allowed: {sorted(oc)[:5] if isinstance(oc, dict) else oc!r}")
    if not isinstance(j.get("messages"), list):
        bad.append("messages is not a list")
    _check_blocks(j.get("messages"), "messages", bad)
    if isinstance(j.get("system"), list):
        _check_blocks(j.get("system"), "system", bad)
    elif j.get("system") is not None and not isinstance(j.get("system"), str):
        bad.append("system is neither a string nor a list")
    return not bad, bad[:10], summ


def parse_response(data, content_type):
    """Summary of one upstream /v1/messages response (SSE stream or JSON): message id, model, output tokens, stop
    reason, and every tool_use (id, name, sha of its input). The proxy is outside the jail, so this is a tamper-proof
    record of what the model emitted; run() cross-checks it against the CLI's stream."""
    out = {"msg_id": None, "resp_model": None, "out_tokens": None, "stop_reason": None, "tool_uses": [],
           "complete": False}
    if "text/event-stream" in (content_type or ""):
        blocks = {}
        for line in data.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            try:
                ev = json.loads(line[5:])
            except ValueError:
                continue
            t = ev.get("type")
            if t == "message_start":
                m = ev.get("message") or {}
                out["msg_id"], out["resp_model"] = m.get("id"), m.get("model")
            elif t == "content_block_start":
                cb = ev.get("content_block") or {}
                if cb.get("type") in ("tool_use", "server_tool_use"):
                    blocks[ev.get("index")] = {"id": cb.get("id"), "name": cb.get("name"), "type": cb.get("type"),
                                               "parts": []}
            elif t == "content_block_delta":
                d = ev.get("delta") or {}
                if d.get("type") == "input_json_delta" and ev.get("index") in blocks:
                    blocks[ev["index"]]["parts"].append(d.get("partial_json") or "")
            elif t == "message_delta":
                out["stop_reason"] = (ev.get("delta") or {}).get("stop_reason")
                out["out_tokens"] = (ev.get("usage") or {}).get("output_tokens")
            elif t == "message_stop":
                out["complete"] = True
        for i in sorted(blocks, key=lambda k: (k is None, k)):
            b = blocks[i]
            raw = "".join(b["parts"])
            try:
                inp = json.loads(raw) if raw else {}
            except ValueError:
                inp = {"<unparsable>": raw[:2000]}
            out["tool_uses"].append({"id": b["id"], "name": b["name"], "type": b["type"], "input": inp})
    else:
        try:
            m = json.loads(data)
        except ValueError:
            return out
        if isinstance(m, dict) and m.get("type") == "message":
            out.update(msg_id=m.get("id"), resp_model=m.get("model"), stop_reason=m.get("stop_reason"),
                       out_tokens=(m.get("usage") or {}).get("output_tokens"), complete=True)
            for b in m.get("content") or []:
                if isinstance(b, dict) and b.get("type") in ("tool_use", "server_tool_use"):
                    out["tool_uses"].append({"id": b.get("id"), "name": b.get("name"), "type": b.get("type"),
                                             "input": b.get("input")})
    return out


RESP_KEEP = 16 * 1024 * 1024     # bytes of each response kept in memory for parse_response


def _make_auth_handler():
    import http.client
    import http.server

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def address_string(self):
            return "jail"

        def _deny(self, code, why, extra=None):
            self.server.log(dict({"t": time.time(), "method": self.command, "path": self.path[:200], "allowed": False,
                                  "why": why}, **(extra or {})))
            body = json.dumps({"type": "error", "error": {"type": "forbidden", "message": str(why)[:500]}}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _forward(self):
            srv = self.server
            if not API_PATH_RX.match(self.path):
                return self._deny(403, "path not allowed by the episode proxy")
            if self.command != "POST":
                return self._deny(405, "method not allowed")
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                return self._deny(411, "chunked request bodies are not supported")
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._deny(400, "bad Content-Length")
            if n < 0 or n > 64 * 1024 * 1024:
                return self._deny(413, "request body too large")
            body = self.rfile.read(n) if n else b""
            ok, why, summ = check_request(self.path, body, srv.policy)
            betas = [b.strip() for b in self.headers.get("anthropic-beta", "").split(",") if b.strip()]
            summ["betas_dropped"] = sorted(set(b for b in betas if b not in API_BETA_ALLOW))
            if srv.capture is not None:
                srv.capture(self.path, dict(self.headers.items()), body)
            if not ok and srv.enforce:
                return self._deny(403, "request refused by the episode proxy: " + "; ".join(why), summ)
            if not ok:
                summ["policy_violations"] = why
            gate = srv.gate() if (srv.gate is not None and path_is_messages(self.path)) else None
            if gate:
                return self._deny(429, gate, summ)
            with srv._lock:
                srv.n_forwarded += 1
                over = srv.max_requests is not None and srv.n_forwarded > srv.max_requests
            if over:
                return self._deny(429, "episode request cap reached", summ)
            hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() not in
                    ("authorization", "x-api-key", "anthropic-beta")}
            kept = [b for b in betas if b in API_BETA_ALLOW]
            if kept:
                hdrs["anthropic-beta"] = ",".join(kept)
            try:
                hdrs.update(srv.auth_headers())
            except Exception as e:                # never echo credential details back into the jail
                srv.log({"t": time.time(), "path": self.path[:200], "allowed": True, "error": "auth: " + type(e).__name__})
                return self._deny(503, "episode proxy could not load credentials")
            hdrs["Host"] = API_HOST
            hdrs["Accept-Encoding"] = "identity"    # keep the response parseable (tamper-proof tool_use record)
            hdrs["Content-Length"] = str(len(body))
            t0 = time.time()
            up = http.client.HTTPSConnection(*srv.upstream, timeout=900) if srv.upstream[0] == API_HOST else \
                http.client.HTTPConnection(*srv.upstream, timeout=900)
            keep, delivered, status = [], True, None
            try:
                up.request(self.command, self.path, body=body, headers=hdrs)
                r = up.getresponse()
                status = r.status
                ctype = r.getheader("Content-Type") or ""
                self.send_response(r.status, r.reason)
                for k, v in r.getheaders():
                    if k.lower() not in HOP and k.lower() != "content-encoding":
                        self.send_header(k, v)
                clen = r.getheader("Content-Length")
                if clen is not None:
                    self.send_header("Content-Length", clen)
                else:
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                nbytes = 0
                while True:
                    chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                    if not chunk:
                        break
                    nbytes += len(chunk)
                    if nbytes <= RESP_KEEP:
                        keep.append(chunk)
                    if delivered:
                        try:
                            if clen is not None:
                                self.wfile.write(chunk)
                            else:
                                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                            self.wfile.flush()
                        except OSError:
                            delivered = False      # keep reading upstream so the record of the response is complete
                if clen is None and delivered:
                    try:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                    except OSError:
                        delivered = False
                resp = parse_response(b"".join(keep), ctype) if path_is_messages(self.path) else {}
                if resp.get("msg_id") and 200 <= status < 300 and delivered:
                    with srv._lock:
                        srv.answered_ids.append(resp["msg_id"])
                srv.log(dict({"t": t0, "method": self.command, "path": self.path[:200], "allowed": True,
                              "status": status, "bytes": nbytes, "delivered": delivered,
                              "s": round(time.time() - t0, 2)}, **summ, **resp))
            except OSError as e:
                srv.log(dict({"t": t0, "method": self.command, "path": self.path[:200], "allowed": True,
                              "status": status, "error": str(e)[:200]}, **summ))
                self.close_connection = True
            finally:
                up.close()

        do_POST = do_GET = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = do_CONNECT = _forward

    return H


def path_is_messages(path):
    """True for /v1/messages (generation), False for count_tokens."""
    return "count_tokens" not in path.split("?", 1)[0]


class AuthProxy(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, sock_path, log_path, auth_headers, max_requests=None, models=(), max_tokens=API_MAX_TOKENS_CAP,
                 upstream=None, enforce=True, capture=None):
        self.auth_headers = auth_headers          # callable -> dict, called per request (picks up token refreshes)
        self.max_requests = max_requests          # bounds the number of requests an agent can make itself
        self.policy = {"models": set(models), "max_tokens": int(max_tokens)}   # body policy (check_request)
        self.upstream = tuple(upstream or UPSTREAM)
        self.enforce = enforce                    # False only for capture runs (what does the CLI send?)
        self.capture = capture                    # optional callable(path, headers, body) (capture runs only)
        self.gate = None                          # optional callable() -> refusal reason or None (run(): live check)
        self.answered_ids = []                    # message ids of answered generation requests, in order
        self.n_forwarded = 0
        self._log_path = log_path
        self._lock = threading.Lock()
        self.events = []
        super().__init__(sock_path, _make_auth_handler())
        os.chmod(sock_path, 0o600)

    def get_request(self):
        # BaseHTTPRequestHandler expects (sock, client_address) with an address it can format.
        sock, _ = self.socket.accept()
        return sock, ("jail", 0)

    log = EgressProxy.log
    start = EgressProxy.start
    handle_error = EgressProxy.handle_error


# ---------------------------------------------------------------------------------------------- jail construction
def _state_dir(episode):
    os.makedirs(STATE_ROOT, mode=0o700, exist_ok=True)
    os.chmod(STATE_ROOT, 0o700)
    d = os.path.join(STATE_ROOT, f"{episode}-{secrets.token_hex(4)}")
    os.makedirs(d, mode=0o700)
    for sub in ("cfg", "etc", "run"):
        os.makedirs(os.path.join(d, sub), mode=0o700)
    return d


def _write_etc(state):
    uid, gid = os.getuid(), os.getgid()
    etc = os.path.join(state, "etc")
    with open(os.path.join(etc, "passwd"), "w") as f:
        f.write(f"agent:x:{uid}:{gid}:agent:{IN_HOME}:/bin/bash\nnobody:x:65534:65534:nobody:/:/sbin/nologin\n")
    with open(os.path.join(etc, "group"), "w") as f:
        f.write(f"agent:x:{gid}:\nnobody:x:65534:\n")
    with open(os.path.join(etc, "hosts"), "w") as f:
        f.write("127.0.0.1 localhost\n::1 localhost\n")
    with open(os.path.join(etc, "hostname"), "w") as f:
        f.write("sandbox\n")
    return etc


def _write_settings(state, sbx):
    """Claude Code settings for the jailed session (the second layer; the jail is the first)."""
    venv = paths.venv_dir()
    s = "/" + sbx  # permission-rule syntax for an absolute path is //abs/path
    settings = {
        "permissions": {
            "defaultMode": "dontAsk",
            "allow": ["Bash", f"Read({s}/**)", f"Edit({s}/**)", f"Write({s}/**)", "Glob", "Grep"],
            # Read/Edit outside the sandbox. "dontAsk" alone does NOT stop the Read tool outside the working dir
            # (live check 2026-10-04: Read /etc/passwd was allowed), so every other top-level tree visible in the
            # jail is denied explicitly. $HOME's other children are not mounted at all (layer 1).
            "deny": list(DENY_TOOLS) + ["mcp__*"] + [
                f"{tool}(/{p}/**)" for tool in ("Read", "Edit", "Write")
                for p in ("/etc", "/usr", "/proc", "/dev", "/run", "/opt", "/tmp", "/var", IN_HOME, venv)
                if not sbx.startswith(p + "/")],
        },
        "enableAllProjectMcpServers": False,
        "disableAllHooks": True,
        "includeCoAuthoredBy": False,
        "cleanupPeriodDays": 1,
        "env": {},
    }
    # WHY (2026-10-04 isolation audit): the settings used to live in the read-write config dir, so the agent could
    # rewrite its own permissions/hooks/env before the nudge --resume pass. The file now lives in the private state dir
    # (not mounted) and is bind-mounted READ-ONLY over <config dir>/settings.json.
    p = os.path.join(state, "settings.json")
    with open(p, "w") as f:
        json.dump(settings, f, indent=1)
    os.chmod(p, 0o400)
    return IN_CFG + "/settings.json"


def _write_inner(state):
    """Start script run inside the jail: bridge 127.0.0.1:3128 to the proxy socket, then exec the given command."""
    p = os.path.join(state, "run", "start.sh")
    with open(p, "w") as f:
        f.write(f"""#!/bin/bash
{SOCAT} TCP-LISTEN:{PROXY_PORT},bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:{IN_RUN}/proxy.sock 2>/dev/null &
if [ -S {IN_RUN}/api.sock ]; then
  {SOCAT} TCP-LISTEN:{API_PORT},bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:{IN_RUN}/api.sock 2>/dev/null &
fi
for i in $(seq 1 50); do (exec 3<>/dev/tcp/127.0.0.1/{PROXY_PORT}) 2>/dev/null && break; sleep 0.1; done
if [ -S {IN_RUN}/api.sock ]; then
  for i in $(seq 1 50); do (exec 3<>/dev/tcp/127.0.0.1/{API_PORT}) 2>/dev/null && break; sleep 0.1; done
fi
exec "$@"
""")
    os.chmod(p, 0o500)
    return IN_RUN + "/start.sh"


def jail_env(sbx, auth_env=None):
    """Environment of the jailed process. Passed through bwrap's own environment (never on its argv, which every
    host process can read), and bwrap is started from a clean environment so nothing else leaks in."""
    proxy = f"http://127.0.0.1:{PROXY_PORT}"
    env = {
        "PATH": "/opt/claude:/usr/local/bin:/usr/bin:/bin", "HOME": IN_HOME, "USER": "agent", "LOGNAME": "agent",
        "SHELL": "/bin/bash", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TERM": "dumb",
        "TMPDIR": os.path.join(sbx, "tmp"),  # = sandbox.SANDBOX_TMP (v3: prompted, pruned at finish)
        "CLAUDE_CONFIG_DIR": IN_CFG,
        "HTTPS_PROXY": proxy, "HTTP_PROXY": proxy, "https_proxy": proxy, "http_proxy": proxy,
        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1", "DISABLE_BUG_COMMAND": "1", "CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR": "1",
        "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
    }
    env.update(auth_env or {})
    return env


def bwrap_argv(sbx, state, inner_cmd):
    """The bwrap command line (no secrets on it). `inner_cmd` runs inside, via the start script."""
    broker_sock = paths.broker_sock()
    venv = paths.venv_dir()
    claude_real = os.path.realpath(CLAUDE_BIN)
    etc = os.path.join(state, "etc")
    a = [BWRAP, "--unshare-user", "--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-uts",
         "--unshare-cgroup-try", "--disable-userns", "--die-with-parent", "--new-session",
         "--hostname", "sandbox",
         "--ro-bind", "/usr", "/usr",
         "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
         "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
         "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/var/tmp",
         "--dir", "/etc"]
    for fn in ("passwd", "group", "hosts", "hostname"):
        a += ["--ro-bind", os.path.join(etc, fn), "/etc/" + fn]
    for p in ("/etc/nsswitch.conf", "/etc/ld.so.cache", "/etc/localtime", "/etc/ssl", "/etc/pki",
              "/etc/crypto-policies", "/etc/alternatives", "/etc/bashrc", "/etc/profile", "/etc/inputrc"):
        if os.path.exists(p):
            a += ["--ro-bind", p, p]
    a += ["--ro-bind", claude_real, IN_CLAUDE,
          "--dir", IN_HOME, "--bind", os.path.join(state, "cfg"), IN_CFG,
          "--ro-bind", os.path.join(state, "settings.json"), IN_CFG + "/settings.json",
          "--ro-bind", os.path.join(state, "run"), IN_RUN]
    if os.path.isdir(venv):
        a += ["--ro-bind", venv, venv]
    a += ["--bind", sbx, sbx]
    if os.path.exists(broker_sock):
        a += ["--bind", broker_sock, broker_sock]
    a += ["--chdir", sbx, "--", IN_RUN + "/start.sh"] + list(inner_cmd)
    return a


class Jail:
    """Per-episode jail state: private state dir (0700, outside every checkout and sandbox), proxy, generated files."""

    def __init__(self, episode, out_dir, allowed=ALLOWED_HOSTS, auth_headers=None, max_requests=None, models=(),
                 max_tokens=API_MAX_TOKENS_CAP, **api_kw):
        """auth_headers: callable returning the real auth headers. When given, an AuthProxy (outside the jail) serves
        http://127.0.0.1:3129 inside it, and the jailed CLI gets only a dummy credential (see AuthProxy). models: the
        model ids the AuthProxy forwards (the episode's model; everything else is refused)."""
        self.episode = episode
        self.sbx = os.path.join(paths.sandbox_root(), episode)
        if not os.path.isdir(self.sbx):
            raise SystemExit(f"sandbox {self.sbx} not found (run common.sandbox prepare first)")
        self.sbx = os.path.realpath(self.sbx)
        self.state = _state_dir(episode)
        _write_etc(self.state)
        _write_inner(self.state)
        self.settings_in = _write_settings(self.state, self.sbx)
        os.makedirs(os.path.join(self.sbx, "tmp"), exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        self.proxy = EgressProxy(os.path.join(self.state, "run", "proxy.sock"),
                                 os.path.join(out_dir, "net_log.jsonl"), allowed).start()
        self.api = None
        if auth_headers is not None:
            self.api = AuthProxy(os.path.join(self.state, "run", "api.sock"),
                                 os.path.join(out_dir, "api_proxy_log.jsonl"), auth_headers, max_requests,
                                 models=models, max_tokens=max_tokens, **api_kw).start()

    def popen(self, inner_cmd, auth_env=None, **kw):
        argv = bwrap_argv(self.sbx, self.state, inner_cmd)
        return subprocess.Popen(argv, env=jail_env(self.sbx, auth_env), start_new_session=True, close_fds=True, **kw)

    def native_transcripts(self):
        root = os.path.join(self.state, "cfg", "projects")
        out = []
        for dp, _, fns in os.walk(root):
            out += [os.path.join(dp, f) for f in fns if f.endswith(".jsonl")]
        return sorted(out)

    def close(self, keep=False):
        for px in (self.proxy, self.api):
            if px is None:
                continue
            try:
                px.shutdown()
                px.server_close()
            except Exception:
                pass
        if not keep:
            shutil.rmtree(self.state, ignore_errors=True)


# ---------------------------------------------------------------------------------------------- auth
def _read_oauth():
    with open(CREDS) as f:
        o = json.load(f).get("claudeAiOauth") or {}
    return o.get("accessToken"), (o.get("expiresAt") or 0) / 1000.0


def oauth_headers():
    """Real OAuth header, read fresh for every proxied request, so the operator session's token refreshes are picked
    up during a long episode. Raises if the token is missing or expired (the proxy then answers 503)."""
    tok, exp = _read_oauth()
    if not tok or exp < time.time() + 30:
        raise RuntimeError("oauth token missing or expired")
    return {"Authorization": "Bearer " + tok}


def _api_key():
    with open(API_KEY_FILE) as f:
        for line in f:
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit(f"no ANTHROPIC_API_KEY in {API_KEY_FILE}")


def apikey_headers():
    return {"x-api-key": _api_key()}


def credential_tails():
    """Last 24 characters of each real credential (never logged), for the post-run leak scan."""
    out = []
    try:
        tok, _ = _read_oauth()
        if tok:
            out.append(tok[-24:])
    except (OSError, ValueError):
        pass
    try:
        out.append(_api_key()[-24:])
    except (OSError, SystemExit):
        pass
    return out


def credential_scan(roots, tails, max_bytes=20 * 1024 * 1024):
    """Files under `roots` that contain any credential tail (refreshed tails are re-read at scan time too)."""
    tails = [t for t in set(tails) | set(credential_tails()) if t and len(t) >= 16]
    hits = []
    for root in roots:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                p = os.path.join(dp, fn)
                try:
                    if os.path.islink(p) or os.path.getsize(p) > max_bytes:
                        continue
                    with open(p, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                if any(t.encode() in data for t in tails):
                    hits.append(p)
    return hits


DUMMY_API_KEY = "sk-ant-api03-" + "jailed-dummy-key-" + "0" * 70


def auth_setup(mode, wall_s):
    """(env for the jail, header callable for the AuthProxy or None, info). Modes:
      oauth      (default) the jail gets a dummy OAuth token; AuthProxy injects the operator's real one. Token never
                 enters the jail. Needs the operator's Claude Code login to stay refreshed during the episode.
      apikey     same with ~/.anthropic_env's API key (real money; the API ledger is checked and charged).
      oauth-env  legacy/fallback: the real access token in the jailed CLI's environment (readable by the agent via
                 /proc; refused unless valid for the whole wall-clock limit + 10 min)."""
    base = {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{API_PORT}"}
    if mode == "oauth":
        tok, exp = _read_oauth()
        if not tok:
            raise SystemExit(f"no OAuth access token in {CREDS}")
        left = exp - time.time()
        if left < 300:
            raise SystemExit("OAuth access token expires in < 5 min; let the operator's Claude Code session refresh it")
        return dict(base, CLAUDE_CODE_OAUTH_TOKEN=DUMMY_OAUTH), oauth_headers, {
            "token_valid_min_at_start": round(left / 60), "token_in_jail": False}
    if mode == "apikey":
        _api_key()
        return dict(base, ANTHROPIC_API_KEY=DUMMY_API_KEY), apikey_headers, {"token_in_jail": False}
    if mode == "oauth-env":
        tok, exp = _read_oauth()
        left = exp - time.time()
        if not tok or left < wall_s + 600:
            raise SystemExit(f"OAuth access token valid for only {left / 60:.0f} min (< {(wall_s + 600) / 60:.0f} "
                             "needed); use --auth oauth (token injection) instead")
        return {"CLAUDE_CODE_OAUTH_TOKEN": tok}, None, {"token_valid_min_at_start": round(left / 60),
                                                         "token_in_jail": True}
    raise SystemExit(f"unknown --auth {mode}")


# ---------------------------------------------------------------------------------------------- ledger
def _ledger_append(rec):
    os.makedirs(LEDGER_DIR, exist_ok=True)
    with open(os.path.join(LEDGER_DIR, ".claude_cli_lock"), "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with open(CLI_LEDGER, "a") as f:
                f.write(json.dumps(rec) + "\n")
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _ledger_records():
    out = []
    if os.path.exists(CLI_LEDGER):
        with open(CLI_LEDGER) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


# ---------------------------------------------------------------------------------------------- run
def claude_argv(model, effort, max_turns, settings_in, resume=None, max_usd=None):
    a = [IN_CLAUDE, "-p", "--output-format", "stream-json", "--verbose", "--model", model, "--effort", effort,
         "--max-turns", str(max_turns), "--tools", ",".join(AGENT_TOOLS),
         "--allowedTools", ",".join(AGENT_TOOLS), "--disallowedTools", ",".join(DENY_TOOLS),
         "--permission-mode", "dontAsk", "--permission-prompts", "none",
         "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
         "--settings", settings_in, "--setting-sources", "user",
         "--safe-mode", "--disable-slash-commands", "--no-chrome"]
    if max_usd:
        a += ["--max-budget-usd", str(max_usd)]
    if resume:
        a += ["--resume", resume]
    return a


def _kill(proc):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def _drain(sock, path):
    """Copy everything from `sock` to the file `path` (the CLI's stderr), in a thread."""
    def go():
        with open(path, "ab") as f:
            while True:
                try:
                    d = sock.recv(65536)
                except OSError:
                    break
                if not d:
                    break
                f.write(d)
                f.flush()
        sock.close()
    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t


def _new_integrity():
    return {"bad_lines": 0, "tool_use_ids": set(), "orphan_results": [], "msg_ids": set(), "tool_inputs": {},
            "compactions": 0}


def _session_pass(jail, argv, prompt_text, auth_env, deadline, raw, tlog, meta):
    """Run one `claude -p` invocation in the jail, streaming its JSON events. Returns (session_id, result_event,
    stop) where stop is None (normal end), 'wall_clock' or 'unexpected_tools'."""
    # WHY sockets, not pipes (2026-10-04 isolation audit): the agent's shell runs as the same uid in the same pid
    # namespace as the CLI. A pipe or file behind /proc/<cli>/fd/1 or fd/2 can be re-opened through /proc (a pipe
    # even for READING, which would let the agent steal and rewrite stream events; the stderr file would give it a
    # writable handle on a file in the run dir outside the jail). Opening /proc/<pid>/fd/N of a socket fails (ENXIO).
    out_r, out_w = socket.socketpair()
    err_r, err_w = socket.socketpair()
    out_r.shutdown(socket.SHUT_WR)
    err_r.shutdown(socket.SHUT_WR)
    try:
        proc = jail.popen(argv, auth_env, stdin=subprocess.PIPE, stdout=out_w.fileno(), stderr=err_w.fileno())
    finally:
        out_w.close()
        err_w.close()
    drainer = _drain(err_r, meta["_stderr_path"])
    timed_out = threading.Event()

    def watchdog():
        while proc.poll() is None:
            if time.time() > deadline:
                timed_out.set()
                _kill(proc)
                return
            time.sleep(1)

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        proc.stdin.write(prompt_text.encode())
        proc.stdin.close()
    except OSError:
        pass
    sid, result, stop = None, None, None
    integ = meta.setdefault("stream_integrity", _new_integrity())
    stream = out_r.makefile("rb")
    for line in stream:
        raw.write(line)
        raw.flush()
        try:
            ev = json.loads(line)
            if not isinstance(ev, dict):
                raise ValueError
        except ValueError:
            # WHY: a partial line glued onto a real event would make that event unparsable and drop a tool call from
            # the audited transcript. Every such line, and every tool_result whose tool_use is missing, is counted;
            # api_meta.json stream_integrity.ok must be true for the episode to count (sandbox finish enforces it).
            if line.strip():
                integ["bad_lines"] += 1
            continue
        typ = ev.get("type")
        sid = ev.get("session_id") or sid
        if typ == "system" and ev.get("subtype") == "init":
            tools = sorted(ev.get("tools") or [])
            meta.setdefault("init", []).append({k: ev.get(k) for k in ("model", "tools", "mcp_servers", "cwd",
                                                                       "permissionMode", "claude_code_version",
                                                                       "slash_commands", "agents", "skills")})
            extra = [t for t in tools if t not in AGENT_TOOLS]
            if extra or (ev.get("mcp_servers") or []):
                meta["unexpected_tools"] = extra + [str(m) for m in ev.get("mcp_servers") or []]
                stop = "unexpected_tools"
                _kill(proc)
                break
        elif typ == "system" and ev.get("subtype") == "compact_boundary":
            integ["compactions"] += 1
        elif typ in ("assistant", "user"):
            msg = ev.get("message") or {}
            if typ == "assistant" and isinstance(msg, dict) and msg.get("id"):
                integ["msg_ids"].add(msg.get("id"))
            content = msg.get("content") if isinstance(msg, dict) else None
            for b in content if isinstance(content, list) else []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    integ["tool_use_ids"].add(b.get("id"))
                    integ["tool_inputs"][b.get("id")] = {"name": b.get("name"), "input": b.get("input")}
                elif isinstance(b, dict) and b.get("type") == "tool_result" and \
                        b.get("tool_use_id") not in integ["tool_use_ids"]:
                    integ["orphan_results"].append(b.get("tool_use_id"))
            entry = {"type": typ, "message": ev.get("message"), "cwd": jail.sbx, "episode": jail.episode,
                     "session_id": ev.get("session_id"), "uuid": ev.get("uuid"),
                     "parent_tool_use_id": ev.get("parent_tool_use_id")}
            tlog(entry)
        elif typ == "result":
            result = ev
    proc.wait()
    stream.close()
    out_r.close()
    drainer.join(timeout=10)
    if timed_out.is_set():
        stop = "wall_clock"
    meta.setdefault("exit_codes", []).append(proc.returncode)
    return sid, result, stop


def _norm_ws(x):
    """Strings with trailing spaces/tabs/CR removed from every line, recursively (see _same_tool_input)."""
    if isinstance(x, str):
        return "\n".join(line.rstrip(" \t\r") for line in x.split("\n"))
    if isinstance(x, list):
        return [_norm_ws(v) for v in x]
    if isinstance(x, dict):
        return {k: _norm_ws(v) for k, v in x.items()}
    return x


def _same_tool_input(name, proxy_in, stream_in, sbx):
    """True if the tool input the model sent (proxy) is what the CLI's stream shows, allowing for the CLI's own
    normalisation, as observed live with Claude Code 2.1.288 (2026-10-04):
      - Bash: a leading `cd <cwd> && ` is stripped (model `cd /home/ec2-user/rlsbx/E && cat TASK.md`, stream
        `cat TASK.md`; every capture run);
      - Write/Edit: trailing whitespace is stripped from lines (model "\n    \n", stream "\n\n"; self-test 07:33Z);
      - Edit: schema defaults are filled in (stream adds "replace_all": false; rewrite probe 07:40Z). Extra keys in the
        stream whose value is false/null are ignored.
    Anything else counts as a mismatch (fail closed; hidden_tool_uses shows both inputs, so a new CLI rewrite can be
    recognised and added here)."""
    if proxy_in == stream_in:
        return True
    proxy_in, stream_in = _norm_ws(proxy_in), _norm_ws(stream_in)
    if isinstance(proxy_in, dict) and isinstance(stream_in, dict):
        stream_in = {k: v for k, v in stream_in.items() if k in proxy_in or v not in (False, None)}
    if proxy_in == stream_in:
        return True
    if name != "Bash" or not (isinstance(proxy_in, dict) and isinstance(stream_in, dict)):
        return False
    pc, sc = proxy_in.get("command"), stream_in.get("command")
    if not (isinstance(pc, str) and isinstance(sc, str)):
        return False
    if {k: v for k, v in proxy_in.items() if k != "command"} != {k: v for k, v in stream_in.items() if k != "command"}:
        return False
    q = re.escape(sbx.rstrip("/"))
    m = re.match(r"^\s*cd\s+(?:" + q + r"|\"" + q + r"\"|'" + q + r"')/?\s*&&\s*(.*)$", pc, re.S)
    return bool(m) and m.group(1).strip() == sc.strip()


def proxy_cross_check(api_events, integ, sbx):
    """Compare the AuthProxy's record (outside the jail, tamper-proof) with the CLI's stream (what the audit reads).
      unmatched_requests: generation requests that were answered (2xx, delivered) but whose message id never appeared
        in the CLI's stream: requests the agent sent itself (side reasoning or spend), unless they are compactions.
      hidden_tool_uses: tool_use blocks the model emitted (per the proxy) that are missing from the stream, or whose
        input differs from what the stream shows beyond the CLI's known normalisation (transcript tampering).
    ok is False if either is non-empty (beyond the number of compactions the CLI reported)."""
    gen = [e for e in api_events if e.get("allowed") and path_is_messages(e.get("path", "")) and e.get("status")
           and 200 <= int(e["status"]) < 300]
    unmatched = [{k: e.get(k) for k in ("t", "model", "max_tokens", "n_tools", "msg_id", "out_tokens", "req_bytes",
                                        "req_sha256", "delivered")}
                 for e in gen if e.get("msg_id") not in integ["msg_ids"]]
    hidden, n_tool_uses = [], 0
    for e in gen:
        if not e.get("delivered", True) or e.get("msg_id") not in integ["msg_ids"]:
            continue    # undelivered responses never reached the CLI; unmatched ones are already reported above
        for tu in e.get("tool_uses") or []:
            n_tool_uses += 1
            seen = integ["tool_inputs"].get(tu.get("id"))
            if seen is None or seen.get("name") != tu.get("name") or not _same_tool_input(
                    tu.get("name"), tu.get("input"), seen.get("input"), sbx):
                hidden.append({"msg_id": e.get("msg_id"), "tool_use_id": tu.get("id"), "name": tu.get("name"),
                               "in_stream": seen is not None,
                               "proxy_input": json.dumps(tu.get("input"), default=str)[:500],
                               "stream_input": None if seen is None else json.dumps(seen.get("input"), default=str)[:500]})
    n_unexplained = max(0, len(unmatched) - integ.get("compactions", 0))
    return {"ok": n_unexplained == 0 and not hidden, "n_generation_requests": len(gen),
            "n_stream_messages": len(integ["msg_ids"]), "n_proxy_tool_uses": n_tool_uses,
            "compactions": integ.get("compactions", 0),
            "unmatched_requests": unmatched[:50], "n_unmatched": len(unmatched), "n_unexplained": n_unexplained,
            "hidden_tool_uses": hidden[:50],
            "out_tokens_total": sum(int(e.get("out_tokens") or 0) for e in gen),
            "out_tokens_unmatched": sum(int(u.get("out_tokens") or 0) for u in unmatched)}


def launcher_invalid_reasons(meta):
    """Reasons (for `sandbox finish`) why an isolated Claude CLI episode must not count, from its api_meta.json.
    WHY (2026-10-04 isolation audit, finding 3): these checks used to be printed as warnings only; an operator had to
    read api_meta.json by hand, so detected tampering or a leaked credential still produced valid=true."""
    if not isinstance(meta, dict):
        return ["launcher_meta_missing"]
    out = []
    si = meta.get("stream_integrity")
    if not (isinstance(si, dict) and si.get("ok") is True):
        out.append("stream_integrity")
    if meta.get("credential_found_in") is None or meta.get("credential_found_in"):
        out.append("credential_leak")
    if meta.get("unexpected_tools"):
        out.append("unexpected_tools")
    if meta.get("api_proxy_denied"):
        out.append("api_proxy_denied")
    if meta.get("auth") == "oauth-env":
        # No AuthProxy in this fallback mode: the CLI (and the agent, which can read the token) reach the API
        # directly, without the body policy (server-side web tools, any model) or the cross-check.
        out.append("unfiltered_api_access")
    else:
        pc = meta.get("proxy_cross_check")
        if not (isinstance(pc, dict) and pc.get("ok") is True):
            out.append("api_cross_check")
    if (meta.get("host_ptrace_scope") or 0) < 1:
        out.append("host_ptrace_scope")
    return out


def ptrace_scope():
    try:
        with open("/proc/sys/kernel/yama/ptrace_scope") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def run(args):
    from common.api_agent import _episode_submitted

    model = MODELS.get(args.model, args.model)
    if args.effort not in EFFORTS:
        raise SystemExit(f"--effort must be one of {EFFORTS}")
    for b in (BWRAP, SOCAT, CLAUDE_BIN):
        if not os.path.exists(b):
            raise SystemExit(f"missing {b}")
    # WHY (2026-10-04 isolation audit): with kernel.yama.ptrace_scope=0 the agent's shell (same uid, same pid
    # namespace) can ptrace the CLI or write its memory through /proc/<pid>/mem and suppress a tool_use before it is
    # ever printed. Scope >= 1 restricts both to descendants; the agent's shell is the CLI's descendant, not its ancestor.
    if (ptrace_scope() or 0) < 1:
        raise SystemExit("kernel.yama.ptrace_scope must be >= 1 (sudo sysctl -w kernel.yama.ptrace_scope=1; persisted in "
                         "/etc/sysctl.d/60-rl-ptrace.conf)")
    if args.auth == "apikey":
        from common import api_agent
        if api_agent._ledger_total() + args.max_usd > api_agent.GLOBAL_CAP_USD:
            raise SystemExit("API ledger would cross its global cap; refusing (use --auth oauth)")
    auth_env, header_fn, auth_info = auth_setup(args.auth, args.max_wall_s)
    cred_tails = credential_tails()
    prompt = open(args.prompt_file).read()
    if prompt.endswith("\n"):
        prompt = prompt[:-1]
    os.makedirs(args.out, exist_ok=True)
    # With token injection the CLI talks only to the AuthProxy; the CONNECT allowlist is then empty (nothing else
    # may leave the jail). In oauth-env mode the CLI needs CONNECT to api.anthropic.com:443.
    # Request cap: the CLI makes about one API request per turn plus a few side requests; 3x turns + 30 leaves room
    # for that and bounds what an agent could spend by calling the proxy itself.
    # Body policy: the AuthProxy forwards only this episode's model, max_tokens <= API_MAX_TOKENS_CAP, and no server
    # tools / MCP / containers / URL sources (check_request).
    jail = Jail(args.episode, args.out, allowed=set() if header_fn else ALLOWED_HOSTS, auth_headers=header_fn,
                max_requests=3 * args.max_turns + 30, models={model})
    tpath = os.path.join(args.out, "api_transcript.jsonl")
    tfile = open(tpath, "w")
    raw = open(os.path.join(args.out, "claude_stream.jsonl"), "wb")

    def tlog(entry):
        entry.setdefault("episode", args.episode)
        tfile.write(json.dumps(entry, default=str) + "\n")
        tfile.flush()

    meta = {"_stderr_path": os.path.join(args.out, "claude_stderr.log"), "stream_integrity": _new_integrity()}
    if jail.api is not None:
        integ_live = meta["stream_integrity"]

        def gate():
            # WHY (2026-10-04 isolation audit, finding 2): every request the CLI makes shows up in its stream (message
            # id) before the CLI's next request. Answered requests that the stream has not shown are the agent's own
            # (unaudited side reasoning, extra spend). Allow a small lag, then refuse; finish marks the episode INVALID.
            ahead = len([m for m in list(jail.api.answered_ids) if m not in integ_live["msg_ids"]])
            if ahead - integ_live["compactions"] >= UNAUDITED_SLACK:
                return f"{ahead} answered API requests are not in the CLI's stream (unaudited requests refused)"
            return None
        jail.api.gate = gate
    t0 = time.time()
    deadline = t0 + args.max_wall_s
    usd, turns, results, stop, sid, submitted = 0.0, 0, [], None, None, False
    try:
        tlog({"type": "user", "message": {"role": "user", "content": prompt}})
        argv = claude_argv(model, args.effort, args.max_turns, jail.settings_in,
                           max_usd=args.max_usd if args.auth == "apikey" else None)
        sid, res, stop = _session_pass(jail, argv, prompt, auth_env, deadline, raw, tlog, meta)
        if res:
            results.append(res)
        submitted = _episode_submitted(args.episode)
        if (not submitted and stop is None and sid and not args.no_nudge and time.time() < deadline - 60
                and not (res or {}).get("subtype", "").startswith("error_max")):
            tlog({"type": "user", "message": {"role": "user", "content": NUDGE}, "nudge": True})
            argv = claude_argv(model, args.effort, max(5, args.max_turns // 4), jail.settings_in, resume=sid,
                               max_usd=args.max_usd if args.auth == "apikey" else None)
            sid, res, stop = _session_pass(jail, argv, NUDGE, auth_env, deadline, raw, tlog, meta)
            if res:
                results.append(res)
            meta["nudged"] = True
        submitted = _episode_submitted(args.episode)
        for r in results:
            usd += float(r.get("total_cost_usd") or 0.0)
            turns += int(r.get("num_turns") or 0)
        if stop is None:
            last = results[-1] if results else {}
            stop = "submitted" if submitted else (last.get("subtype") or "no_result")
        # The native Claude Code transcript (richer: includes thinking, tool-result metadata) is kept for forensics.
        for i, p in enumerate(jail.native_transcripts()):
            shutil.copyfile(p, os.path.join(args.out, f"claude_native_{i}.jsonl"))
    finally:
        tfile.close()
        raw.close()
        net = list(jail.proxy.events)
        api_events = list(jail.api.events) if jail.api else []
        jail.close(keep=args.keep_state)
    meta.pop("_stderr_path", None)
    integ = meta.pop("stream_integrity", None) or _new_integrity()
    meta["stream_integrity"] = {"ok": integ["bad_lines"] == 0 and not integ["orphan_results"],
                                "bad_lines": integ["bad_lines"], "orphan_results": integ["orphan_results"][:20],
                                "n_tool_uses": len(integ["tool_use_ids"])}
    if not meta["stream_integrity"]["ok"]:
        print("WARNING: stream integrity check failed (possible transcript tampering): "
              + json.dumps(meta["stream_integrity"]), file=sys.stderr)
    if jail.api is not None:
        meta["proxy_cross_check"] = proxy_cross_check(api_events, integ, jail.sbx)
        if not meta["proxy_cross_check"]["ok"]:
            print("WARNING: proxy cross-check failed (API traffic the CLI stream does not show): "
                  + json.dumps({k: meta["proxy_cross_check"][k] for k in ("n_unexplained", "hidden_tool_uses")},
                               default=str)[:2000], file=sys.stderr)
    meta["host_ptrace_scope"] = ptrace_scope()
    meta["credential_found_in"] = credential_scan([args.out, jail.sbx], cred_tails)
    if meta["credential_found_in"]:
        print("WARNING: real credential found in " + ", ".join(meta["credential_found_in"]), file=sys.stderr)
    out = {"episode": args.episode, "task": args.task, "provider": "claude-code-cli", "model": model,
           "effort": args.effort, "auth": args.auth, **auth_info, "turns": turns, "stop": stop,
           "submitted": submitted, "usd": round(usd, 4), "usd_note": ("CLI-reported list-price estimate; billed to the "
           "operator's subscription" if args.auth != "apikey" else "CLI-reported cost, charged to the API ledger"),
           "wall_s": round(time.time() - t0, 1), "session_id": sid,
           "usage": [r.get("usage") for r in results], "model_usage": [r.get("modelUsage") for r in results],
           "net_denied": [e for e in net if e.get("allowed") is False],
           "net_allowed_n": sum(1 for e in net if e.get("allowed")),
           "api_proxy_requests": len(api_events),
           # one row per proxied request (body hash and size; message id ties it to the CLI's stream)
           "api_requests": [{k: e.get(k) for k in ("t", "path", "allowed", "status", "model", "max_tokens", "req_bytes",
                                                   "req_sha256", "msg_id", "out_tokens", "delivered")}
                            for e in api_events if "allowed" in e],
           "api_proxy_policy": {"models": sorted(jail.api.policy["models"]), "max_tokens": jail.api.policy["max_tokens"],
                                "max_requests": jail.api.max_requests} if jail.api else None,
           "api_proxy_betas_dropped": sorted({b for e in api_events for b in e.get("betas_dropped") or []}),
           "api_proxy_denied": [e for e in api_events if e.get("allowed") is False],
           "api_proxy_errors": [e for e in api_events if e.get("error")],
           "isolation": "bwrap+netns+egress-allowlist", **meta}
    out["launcher_invalid_reasons"] = launcher_invalid_reasons(json.loads(json.dumps(out, default=str)))
    if out["launcher_invalid_reasons"]:
        print("WARNING: sandbox finish will mark this episode INVALID: " + ", ".join(out["launcher_invalid_reasons"]),
              file=sys.stderr)
    json.dump(out, open(os.path.join(args.out, "api_meta.json"), "w"), indent=1, default=str)
    _ledger_append({"t": time.time(), "episode": args.episode, "task": args.task, "model": model,
                    "effort": args.effort, "auth": args.auth, "usd": round(usd, 6), "turns": turns, "stop": stop})
    if args.auth == "apikey":
        from common import api_agent
        api_agent._append({"t": time.time(), "episode": args.episode, "task": args.task, "model": model,
                           "turn": "claude-cli", "usd": round(usd, 6)})
    print(json.dumps({k: out[k] for k in ("episode", "model", "effort", "turns", "stop", "submitted", "usd",
                                          "wall_s")} | {"stream_ok": out["stream_integrity"]["ok"],
                                                        "proxy_ok": (out.get("proxy_cross_check") or {}).get("ok"),
                                                        "launcher_invalid": out["launcher_invalid_reasons"],
                                                        "credential_leak": bool(out["credential_found_in"])}))


def spent(args):
    recs = _ledger_records()
    by = {}
    for r in recs:
        for k in (f"model={r.get('model')}", f"task={r.get('task')}", f"auth={r.get('auth')}"):
            by[k] = by.get(k, 0.0) + float(r.get("usd") or 0.0)
    print(json.dumps({"total_usd": round(sum(float(r.get("usd") or 0) for r in recs), 4), "n_episodes": len(recs),
                      "breakdown": {k: round(v, 4) for k, v in sorted(by.items())}}, indent=1))


# ---------------------------------------------------------------------------------------------- red team
def redteam_probes(sbx, decoy_sbx=None, host_pids=()):
    """(name, shell command, expectation) triples. Expectations: 'fail' (non-zero exit, or empty stdout),
    'empty' (stdout empty), 'ok' (exit 0), or ('contains', s) / ('absent', s) on stdout."""
    home = os.path.expanduser("~")
    main = paths.MAIN_REPO
    sroot = paths.sandbox_root()
    P = []

    def add(name, cmd, exp, severity="major"):
        P.append((name, cmd, exp, severity))

    for p in dict.fromkeys((f"{home}/wt", main, f"{home}/ideating-rl-tests", f"{home}/.claude", f"{home}/.claude/projects",
              f"{home}/.claude/.credentials.json", f"{home}/hf_home", f"{home}/.anthropic_env", f"{home}/.openai_env",
              f"{home}/.hf_env", f"{home}/.aws", f"{home}/.rl_api", STATE_ROOT, paths.episodes_dir(),
              "/opt/pytorch", "/root", "/var/log", "/sys/class", f"{home}/.ssh")):
        add(f"absent:{p}", f"test -e {shlex.quote(p)} || ls -d {shlex.quote(p)}", "fail")
        # Content is never printed (a control run on the host would otherwise copy secrets into the report).
        add(f"read:{p}", f"head -c 1 {shlex.quote(p)} >/dev/null 2>&1 && test -r {shlex.quote(p)}", "fail")
    # Exactly the mount points must be visible under $HOME and under the sandbox root, nothing else.
    mounted = [sbx, paths.venv_dir(), paths.broker_sock()]

    def first_under(base):
        return sorted({os.path.relpath(m, base).split(os.sep)[0] for m in mounted if m.startswith(base + os.sep)})
    add("ls_home", f"ls -A {shlex.quote(home)}", ("only", first_under(home)))
    add("ls_sandbox_root", f"ls -A {shlex.quote(sroot)}", ("only", first_under(sroot)))
    if decoy_sbx:
        add("decoy_ls", f"ls -A {shlex.quote(decoy_sbx)}", "fail")
        add("decoy_cat_task", f"cat {shlex.quote(decoy_sbx)}/TASK.md", "fail")
        add("decoy_via_dotdot", f"cat ../{os.path.basename(decoy_sbx)}/TASK.md", "fail")
        add("decoy_tool", f"cd {shlex.quote(sbx)} && printf '%s' {os.path.basename(decoy_sbx)} > scratch/ep && "
            f"./py -c \"import socket,json;s=socket.socket(socket.AF_UNIX);s.connect('{paths.broker_sock()}');"
            f"s.sendall((json.dumps(dict(episode=open('scratch/ep').read(),tool='help',args={{}}))+'\\n').encode());"
            f"print(s.makefile().readline())\"", ("absent", '"ok": true'))
    add("dotdot_up", "ls -A ../..", ("absent", "ideating-rl-tests"))
    add("dotdot_cat", "cat ../../ideating-rl-tests/CONTEXT.md", "fail")
    add("dotdot_deep", f"ls -A ../../../../../..{home}/", ("absent_re", r"(^|\s)(wt|ideating-rl-tests|\.claude)(\s|$)"))
    add("symlink_root", f"cd {shlex.quote(sbx)} && ln -sfn / scratch/rt && ls scratch/rt/home/ec2-user/wt", "fail")
    add("symlink_repo", f"cd {shlex.quote(sbx)} && ln -sfn {shlex.quote(main)} scratch/rp && cat scratch/rp/CONTEXT.md",
        "fail")
    add("symlink_cleanup", f"cd {shlex.quote(sbx)} && rm -f scratch/rt scratch/rp scratch/ep", "ok")
    add("py_os_open", f"cd {shlex.quote(sbx)} && ./py -c \"import os;os.open('{main}/CONTEXT.md', os.O_RDONLY)\"",
        "fail")
    add("py_listdir_home", f"./py -c \"import os;print(os.listdir('{home}'))\"", ("absent", "wt"))
    add("py_walk_root", "./py -c \"import os\nfor d,_,f in os.walk('/'):\n  [print(os.path.join(d,x)) for x in f "
        "if x in ('instance.json','.credentials.json','.admin_token','grader.py')]\" 2>/dev/null", "empty")
    add("find_instances", "find / -xdev \\( -name instance.json -o -name '*.credentials*' -o -name .admin_token "
        "-o -name 'SHIFTHUNT*' -o -name .anthropic_env -o -name history.jsonl \\) 2>/dev/null", "empty")
    add("find_all_paths", "find / -path /proc -prune -o -print 2>/dev/null | grep -E "
        "'ideating-rl-tests|/wt/|hf_home|\\.claude/projects|rlsbx/ep' | grep -v " + shlex.quote(sbx) + " | head",
        "empty")
    add("grep_canary", "grep -rl RLCANARY /home /tmp /run /opt /etc /var 2>/dev/null | head", "empty")
    add("proc_pids_only_ns", "ls /proc | grep -E '^[0-9]+$' | wc -l", ("max_int", 40))
    for pid in sorted(set(host_pids)):
        add(f"proc_host_{pid}_cmdline", f"cat /proc/{pid}/cmdline", ("absent", "python"))
        add(f"proc_host_{pid}_root", f"ls /proc/{pid}/root{home}/wt", "fail")
        add(f"proc_host_{pid}_cwd", f"ls /proc/{pid}/cwd/", ("absent", "CONTEXT.md"))
    add("proc_any_root_repo", "for p in /proc/[0-9]*; do ls $p/root" + shlex.quote(main) + " 2>/dev/null; "
        "ls $p/cwd/../../ideating-rl-tests 2>/dev/null; done", "empty")
    add("proc_self_fds", "ls -l /proc/self/fd/ | grep -E 'ideating|/wt/|\\.claude|rlsbx/ep' | grep -v "
        + shlex.quote(sbx), "empty")
    add("gpu_absent", "ls /dev/nvidia* 2>/dev/null", "empty")
    add("nested_userns", "unshare -r true", "fail")
    # (no host-side leftovers when the same probe runs without the jail in --control mode)
    add("mount_denied", "mount -t tmpfs none /mnt 2>&1; d=$TMPDIR/rt_m; mkdir -p $d && mount --bind / $d; rc=$?; "
        "rmdir $d 2>/dev/null; exit $rc", "fail")
    add("net_direct_ip", "./py -c \"import socket;socket.create_connection(('1.1.1.1',443),timeout=4)\"", "fail")
    add("net_metadata", "./py -c \"import socket;socket.create_connection(('169.254.169.254',80),timeout=4)\"",
        "fail")
    add("net_dns", "./py -c \"import socket;socket.getaddrinfo('huggingface.co',443)\"", "fail")
    add("net_proxy_other_host", "./py -c \"import socket;s=socket.create_connection(('127.0.0.1',3128),timeout=5);"
        "s.sendall(b'CONNECT huggingface.co:443 HTTP/1.1\\r\\n\\r\\n');print(s.recv(100))\"", ("absent", "200"))
    add("net_proxy_plain_http", "./py -c \"import socket;s=socket.create_connection(('127.0.0.1',3128),timeout=5);"
        "s.sendall(b'GET http://169.254.169.254/latest/meta-data/ HTTP/1.1\\r\\nHost: x\\r\\n\\r\\n');"
        "print(s.recv(100))\"", ("absent", "200"))
    # Production config (token injection): the CONNECT allowlist is empty, so even the API host is refused there;
    # the only egress is the AuthProxy at 127.0.0.1:3129, which forwards /v1/messages only.
    add("net_proxy_api_connect_denied", "./py -c \"import socket;s=socket.create_connection(('127.0.0.1',3128),"
        "timeout=10);s.sendall(b'CONNECT api.anthropic.com:443 HTTP/1.1\\r\\n\\r\\n');print(s.recv(100))\"",
        ("absent", "200"))

    def http(method, path, body="", beta=""):
        b64 = base64.b64encode((body if isinstance(body, str) else json.dumps(body)).encode()).decode()
        return ("./py -c \"import http.client as h,base64;c=h.HTTPConnection('127.0.0.1',3129,timeout=30);"
                f"c.request('{method}','{path}',body=base64.b64decode('{b64}'),headers={{'content-type':"
                f"'application/json','anthropic-beta':'{beta}'}});r=c.getresponse();print(r.status)\"")
    add("api_proxy_profile_denied", http("GET", "/api/oauth/profile"), ("only", ["403"]))
    add("api_proxy_models_denied", http("GET", "/v1/models"), ("only", ["403"]))
    add("api_proxy_files_denied", http("POST", "/v1/files"), ("only", ["403"]))
    add("api_proxy_dotdot_denied", http("POST", "/v1/messages/../models"), ("only", ["403"]))
    # Forwarding works (upstream answers; with the red-team's bogus injected credential that is a 401).
    # Body policy (2026-10-04 isolation audit): server tools, MCP, containers, URL sources, other models and
    # oversized max_tokens are refused by the proxy itself (403) before anything is sent upstream.
    ok_body = {"model": REDTEAM_MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}

    def body(**kw):
        return dict(ok_body, **kw)
    deny = {
        "web_fetch": body(tools=[{"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 1}]),
        "web_search": body(tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}]),
        "code_execution": body(tools=[{"type": "code_execution_20250825", "name": "code_execution"}]),
        "mcp_servers": body(mcp_servers=[{"type": "url", "url": "https://example.com/sse", "name": "x"}]),
        "container": body(container="cntr_x"),
        "url_image": body(messages=[{"role": "user", "content": [{"type": "image", "source": {
            "type": "url", "url": "https://raw.githubusercontent.com/x/y/main/a.png"}}]}]),
        "url_document": body(messages=[{"role": "user", "content": [{"type": "document", "source": {
            "type": "url", "url": "https://raw.githubusercontent.com/x/y/main/README.md"}}]}]),
        "url_in_tool_result": body(messages=[{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
                                   "content": [{"type": "image", "source": {"type": "url", "url": "https://x/a.png"}}]}]}]),
        "other_model": body(model="claude-opus-5-5" if REDTEAM_MODEL != "claude-opus-5-5" else "claude-haiku-4-5"),
        "max_tokens_over_cap": body(max_tokens=API_MAX_TOKENS_CAP + 1),
        "unknown_field": body(betas=["x"]),
        "not_json": "not json",
    }
    for k, v in deny.items():
        add(f"api_policy_{k}_denied", http("POST", "/v1/messages?beta=true", v,
                                           beta="oauth-2025-04-20,web-fetch-2025-09-10"), ("only", ["403"]))
    add("api_proxy_get_denied", http("GET", "/v1/messages", ""), ("only", ["405"]))
    # Forwarding works for an allowed body (upstream answers; with the red-team's bogus credential that is a 401).
    add("api_proxy_forwards_messages", http("POST", "/v1/messages", ok_body), ("only", ["401"]))
    add("no_real_credential_in_jail", "cat /proc/[0-9]*/environ 2>/dev/null | tr '\\0' '\\n' | "
        "grep -E '^(CLAUDE_CODE_OAUTH_TOKEN|ANTHROPIC_API_KEY)=' | grep -v jailed-dummy", "empty")
    add("env_clean", "env | cut -d= -f1 | sort | tr '\\n' ' '", ("absent_re", r"HF_|OPENAI|AWS|SSH_|GPUQ|RL_EPI"))
    add("tool_help", f"cd {shlex.quote(sbx)} && ./tool help", ("contains", '"ok": true'))
    add("tool_budget", f"cd {shlex.quote(sbx)} && ./tool budget", ("contains", '"ok": true'))
    add("sandbox_writable", f"cd {shlex.quote(sbx)} && echo hi > scratch/rt_probe && cat scratch/rt_probe && "
        "rm scratch/rt_probe", ("contains", "hi"))
    add("usr_readonly", "touch /usr/bin/rt_probe", "fail")
    # (removes its file at once if it succeeds, i.e. in --control mode: the venv is shared with every jail)
    add("venv_readonly", f"f={shlex.quote(paths.venv_dir())}/.rt_probe_$$; touch $f && rm -f $f", "fail")
    add("tmpdir_in_sandbox", "echo $TMPDIR", ("contains", sbx))
    # Process-level tampering (2026-10-04 isolation audit): the agent must not be able to ptrace / write the memory of
    # its ancestors (the CLI), nor re-open their stdout/stderr through /proc (they are sockets: ENXIO).
    add("ptrace_scope_ge_1", "cat /proc/sys/kernel/yama/ptrace_scope", ("min_int", 1))
    add("proc_parent_mem_denied", "./py -c \"import os;open('/proc/%d/mem' % os.getppid(),'rb')\"", "fail")
    add("proc_pid1_mem_denied", "./py -c \"open('/proc/1/mem','rb')\"", "fail")
    add("ptrace_attach_parent_denied", "./py -c \"import ctypes,os,sys;l=ctypes.CDLL(None,use_errno=True);"
        "sys.exit(0 if l.ptrace(16,os.getppid(),0,0)==0 else 1)\"", "fail")
    add("reopen_parent_stdout_denied", "./py -c \"import os;open('/proc/%d/fd/1' % os.getppid(),'rb')\"", "fail")
    add("reopen_parent_stderr_denied", "./py -c \"import os;open('/proc/%d/fd/2' % os.getppid(),'ab')\"", "fail")
    # Claude Code settings (permissions, hooks, env) are bind-mounted read-only over the writable config dir.
    add("settings_readonly", f"echo '{{}}' > {IN_CFG}/settings.json", "fail")
    add("settings_not_replaceable", f"mv {IN_CFG}/settings.json {IN_CFG}/s.bak", "fail")
    add("settings_not_removable", f"rm -f {IN_CFG}/settings.json", "fail")
    # Pre-existing broker behaviour: an unauthenticated ping must not reveal anything (it used to return the host pid).
    ping = base64.b64encode(b'{"admin": "ping"}\n').decode()
    add("broker_ping_no_pid", "./py -c \"import socket,base64;s=socket.socket(socket.AF_UNIX);s.connect('"
        + paths.broker_sock() + f"');s.sendall(base64.b64decode('{ping}'));print(s.makefile().readline())\"",
        ("absent", '"pid"'), "minor")
    return P


def _judge(exp, rc, out):
    if exp == "fail":
        return rc != 0
    if exp == "empty":
        return out.strip() == ""
    if exp == "ok":
        return rc == 0
    kind, val = exp
    if kind == "contains":
        return rc == 0 and val in out
    if kind == "absent":
        return val not in out
    if kind == "absent_re":
        return re.search(val, out) is None
    if kind == "only":
        return sorted(out.split()) == sorted(val)
    if kind in ("max_int", "min_int"):
        try:
            v = int(out.strip())
        except ValueError:
            return False
        return v <= val if kind == "max_int" else v >= val
    raise ValueError(exp)


def _run_socket_stdio(start, timeout):
    """start(stdin=, stdout=, stderr=) -> Popen. stdout/stderr are sockets, as in a real episode (_session_pass), so
    the probes see the same /proc/<pid>/fd objects as the agent would. Returns (rc, stdout, stderr) as text."""
    pair = [socket.socketpair() for _ in range(2)]
    try:
        p = start(stdin=subprocess.DEVNULL, stdout=pair[0][1].fileno(), stderr=pair[1][1].fileno())
    finally:
        pair[0][1].close()
        pair[1][1].close()
    bufs = [[], []]

    def rd(i):
        while True:
            d = pair[i][0].recv(65536)
            if not d:
                break
            bufs[i].append(d)
    ts = [threading.Thread(target=rd, args=(i,), daemon=True) for i in range(2)]
    for t in ts:
        t.start()
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill(p)
    for t in ts:
        t.join(timeout=10)
    for a, _ in pair:
        a.close()
    return p.returncode, b"".join(bufs[0]).decode(errors="replace"), b"".join(bufs[1]).decode(errors="replace")


def redteam(args):
    """Run every probe inside the SAME jail the agent gets (same bwrap argv builder, proxy and env, with a dummy
    token), one fresh jail per probe, and compare against expectations. Exit 1 if any probe escaped."""
    out_dir = args.out or os.path.join(STATE_ROOT, "redteam_" + args.episode)
    # Same configuration as a production run (token injection, empty CONNECT allowlist), with a bogus credential.
    jail = Jail(args.episode, out_dir, allowed=set(),
                auth_headers=lambda: {"Authorization": "Bearer redteam-bogus-credential"}, models={REDTEAM_MODEL})
    probe_env = {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{API_PORT}", "CLAUDE_CODE_OAUTH_TOKEN": DUMMY_OAUTH}
    decoy = os.path.join(paths.sandbox_root(), args.decoy) if args.decoy else None
    host_pids = [os.getpid(), os.getppid(), 1]
    try:
        bp = subprocess.run(["pgrep", "-f", "common.broker"], capture_output=True, text=True).stdout.split()
        host_pids += [int(x) for x in bp[:2]]
    except Exception:
        pass
    results, bad, minor = [], [], []
    try:
        for name, cmd, exp, severity in redteam_probes(jail.sbx, decoy, host_pids):
            inner = ["/bin/bash", "-c", f"cd {shlex.quote(jail.sbx)} && {{ {cmd} ; }}"]
            if args.control:
                # Control: the same probes WITHOUT the jail (plain bash on the host, same env). Each probe that also
                # "passes" here does not discriminate; the control shows that the suite detects real access.
                def start(**kw):
                    return subprocess.Popen(inner, env=jail_env(jail.sbx), cwd=jail.sbx, start_new_session=True,
                                            **kw)
            else:
                def start(**kw):
                    return jail.popen(inner, probe_env, **kw)
            rc, o, e = _run_socket_stdio(start, timeout=120)
            ok = _judge(exp, rc, o)
            if args.control:
                o, e = "<not recorded in control mode>", ""   # host output may contain real secrets
            results.append({"probe": name, "pass": ok, "rc": rc, "expect": exp, "severity": severity,
                            "stdout": o[-400:], "stderr": e[-300:]})
            if not ok:
                # "minor" probes cover low-impact, pre-existing behaviour of shared services (e.g. the broker's ping
                # reply, fixed in code but live only after that broker restarts): reported, not counted as escapes.
                (minor if severity == "minor" else bad).append(name)
    finally:
        net = list(jail.proxy.events) + list(jail.api.events)
        jail.close()
    if decoy and not args.control:
        # The decoy_tool probe sent a request naming the decoy episode from inside this jail. The broker identifies
        # the caller by its cwd (SO_PEERCRED + /proc/<pid>/cwd, which still resolves because the sandbox is mounted
        # at its real path) and must have refused it AND flagged this episode (cross_episode_access => INVALID).
        try:
            with open(os.path.join(paths.episodes_dir(), args.episode + ".json")) as f:
                flagged = bool(json.load(f).get("cross_episode_access"))
        except (OSError, ValueError):
            flagged = False
        results.append({"probe": "broker_flags_cross_episode", "pass": flagged, "rc": None, "expect": "flagged",
                        "stdout": "", "stderr": ""})
        if not flagged:
            bad.append("broker_flags_cross_episode")
    summary = {"episode": args.episode, "control": bool(args.control), "n_probes": len(results),
               "n_escaped": len(bad), "escaped": bad, "minor_escaped": minor, "net_events": net, "results": results}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "redteam_control.json" if args.control else "redteam.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for r in results:
        print(("PASS " if r["pass"] else "ESCAPE ") + r["probe"] + ("" if r["pass"] else f"  rc={r['rc']} "
              f"out={r['stdout'][-200:]!r}"))
    print(json.dumps({k: summary[k] for k in ("episode", "n_probes", "n_escaped", "escaped", "minor_escaped")}))
    if args.control:
        return 0
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--episode", required=True)
    r.add_argument("--task", required=True)
    r.add_argument("--prompt-file", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default="opus", help=f"alias {list(MODELS)} or a full model id")
    r.add_argument("--effort", default="high", choices=EFFORTS)
    r.add_argument("--max-turns", type=int, default=80)
    r.add_argument("--max-wall-s", type=int, default=3600, help="wall-clock limit for the whole agent run")
    r.add_argument("--max-usd", type=float, default=2.0, help="--auth apikey only: passed as --max-budget-usd")
    r.add_argument("--auth", choices=("oauth", "apikey", "oauth-env"), default="oauth",
                   help="oauth/apikey: real credential injected by a proxy outside the jail (default); oauth-env: "
                        "token in the jailed environment (fallback)")
    r.add_argument("--no-nudge", action="store_true", help="do not resume once with a submit reminder")
    r.add_argument("--keep-state", action="store_true", help="keep the private state dir (debugging)")
    t = sub.add_parser("redteam")
    t.add_argument("--episode", required=True)
    t.add_argument("--decoy", help="another episode id whose sandbox must be invisible")
    t.add_argument("--out")
    t.add_argument("--control", action="store_true",
                   help="run the probes on the host WITHOUT the jail (shows which probes detect real access)")
    sub.add_parser("spent")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "redteam":
        sys.exit(redteam(a))
    else:
        spent(a)


if __name__ == "__main__":
    main()
