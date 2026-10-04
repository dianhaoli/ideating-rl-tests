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
  /var, HOME with the config dir, the venv); --strict-mcp-config with no servers; --safe-mode;
  --disable-slash-commands; nonessential traffic, telemetry and the auto-updater disabled; Bash cwd reset to the
  sandbox after every command (so the transcript's recorded cwd is always the true cwd for the audit).
  The tool list Claude Code reports at start-up (stream-json `system/init`) is checked against that set; any extra
  tool stops the episode (stop = "unexpected_tools").

Authentication and the residual: the default (--auth oauth) passes ONLY the OAuth access token of the operator's
Claude Code login (~/.claude/.credentials.json, field accessToken) as CLAUDE_CODE_OAUTH_TOKEN in the jailed process's
environment. No credential file and no refresh token enter the jail (a refresh inside would rotate the operator's
refresh token and log the orchestrator out). The access token is short-lived (hours; the launcher refuses to start
unless it is valid for the whole wall-clock limit plus 10 minutes). RESIDUAL RISK: the agent's own shell can read
that token from /proc/<claude pid>/environ (same uid, same pid namespace) and could use it to call the Anthropic API
through the proxy. That exposes no task answer (the API knows nothing about instances), only inference spend on the
operator's account until the token expires. --auth apikey passes ~/.anthropic_env's ANTHROPIC_API_KEY instead (same
residual, but spend is real money: the shared API ledger is checked before and charged after the run).
Removing the residual needs an auth-injecting reverse proxy (the jail gets a dummy token, the proxy outside swaps in
the real one); not built yet (2 h timebox), see docs/HARNESS_API.md section 11.

Usage (after `python -m common.sandbox prepare ...` printed episode E):
    python -m common.claude_agent run --episode E --task T --prompt-file <run>/episodes/E/agent_prompt.txt \
        --out <run>/episodes/E --model haiku --effort high --max-turns 60 --max-wall-s 3600
    python -m common.sandbox finish --episode E --transcript <run>/episodes/E/api_transcript.jsonl \
        --agent-model claude-cli:claude-haiku-4-5-20251001:high
    python -m common.claude_agent redteam --episode E [--decoy E2]   # scripted isolation checks (no LLM)
    python -m common.claude_agent spent                               # CLI-reported cost ledger
"""
import argparse
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
        "includeCoAuthoredBy": False,
        "cleanupPeriodDays": 1,
        "env": {},
    }
    p = os.path.join(state, "cfg", "settings.json")
    with open(p, "w") as f:
        json.dump(settings, f, indent=1)
    return IN_CFG + "/settings.json"


def _write_inner(state):
    """Start script run inside the jail: bridge 127.0.0.1:3128 to the proxy socket, then exec the given command."""
    p = os.path.join(state, "run", "start.sh")
    with open(p, "w") as f:
        f.write(f"""#!/bin/bash
{SOCAT} TCP-LISTEN:{PROXY_PORT},bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:{IN_RUN}/proxy.sock 2>/dev/null &
for i in $(seq 1 50); do (exec 3<>/dev/tcp/127.0.0.1/{PROXY_PORT}) 2>/dev/null && break; sleep 0.1; done
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
        "TMPDIR": os.path.join(sbx, ".tmp"), "CLAUDE_CONFIG_DIR": IN_CFG,
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

    def __init__(self, episode, out_dir, allowed=ALLOWED_HOSTS):
        self.episode = episode
        self.sbx = os.path.join(paths.sandbox_root(), episode)
        if not os.path.isdir(self.sbx):
            raise SystemExit(f"sandbox {self.sbx} not found (run common.sandbox prepare first)")
        self.sbx = os.path.realpath(self.sbx)
        self.state = _state_dir(episode)
        _write_etc(self.state)
        _write_inner(self.state)
        self.settings_in = _write_settings(self.state, self.sbx)
        os.makedirs(os.path.join(self.sbx, ".tmp"), exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        self.proxy = EgressProxy(os.path.join(self.state, "run", "proxy.sock"),
                                 os.path.join(out_dir, "net_log.jsonl"), allowed).start()

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
        try:
            self.proxy.shutdown()
            self.proxy.server_close()
        except Exception:
            pass
        if not keep:
            shutil.rmtree(self.state, ignore_errors=True)


# ---------------------------------------------------------------------------------------------- auth
def oauth_env(min_valid_s):
    with open(CREDS) as f:
        o = json.load(f).get("claudeAiOauth") or {}
    tok, exp = o.get("accessToken"), (o.get("expiresAt") or 0) / 1000.0
    if not tok:
        raise SystemExit(f"no OAuth access token in {CREDS}")
    left = exp - time.time()
    if left < min_valid_s:
        raise SystemExit(f"OAuth access token valid for only {left / 60:.0f} min (< {min_valid_s / 60:.0f} needed); "
                         "let the operator's Claude Code session refresh it (any interactive request does), or "
                         "use a shorter --max-wall-s, or --auth apikey")
    return {"CLAUDE_CODE_OAUTH_TOKEN": tok}, {"token_valid_min": round(left / 60)}


def apikey_env():
    with open(API_KEY_FILE) as f:
        for line in f:
            if line.startswith("ANTHROPIC_API_KEY="):
                return {"ANTHROPIC_API_KEY": line.split("=", 1)[1].strip()}
    raise SystemExit(f"no ANTHROPIC_API_KEY in {API_KEY_FILE}")


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


def _session_pass(jail, argv, prompt_text, auth_env, deadline, raw, tlog, meta):
    """Run one `claude -p` invocation in the jail, streaming its JSON events. Returns (session_id, result_event,
    stop) where stop is None (normal end), 'wall_clock' or 'unexpected_tools'."""
    proc = jail.popen(argv, auth_env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                      stderr=open(meta["_stderr_path"], "ab"))
    timed_out = threading.Event()

    def watchdog():
        while proc.poll() is None:
            if time.time() > deadline:
                timed_out.set()
                _kill(proc)
                return
            time.sleep(1)

    threading.Thread(target=watchdog, daemon=True).start()
    proc.stdin.write(prompt_text.encode())
    proc.stdin.close()
    sid, result, stop = None, None, None
    for line in proc.stdout:
        raw.write(line)
        raw.flush()
        try:
            ev = json.loads(line)
        except ValueError:
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
        elif typ in ("assistant", "user"):
            entry = {"type": typ, "message": ev.get("message"), "cwd": jail.sbx, "episode": jail.episode,
                     "session_id": ev.get("session_id"), "uuid": ev.get("uuid"),
                     "parent_tool_use_id": ev.get("parent_tool_use_id")}
            tlog(entry)
        elif typ == "result":
            result = ev
    proc.wait()
    if timed_out.is_set():
        stop = "wall_clock"
    meta.setdefault("exit_codes", []).append(proc.returncode)
    return sid, result, stop


def run(args):
    from common.api_agent import _episode_submitted

    model = MODELS.get(args.model, args.model)
    if args.effort not in EFFORTS:
        raise SystemExit(f"--effort must be one of {EFFORTS}")
    for b in (BWRAP, SOCAT, CLAUDE_BIN):
        if not os.path.exists(b):
            raise SystemExit(f"missing {b}")
    if args.auth == "oauth":
        auth_env, auth_info = oauth_env(args.max_wall_s + 600)
    else:
        from common import api_agent
        if api_agent._ledger_total() + args.max_usd > api_agent.GLOBAL_CAP_USD:
            raise SystemExit("API ledger would cross its global cap; refusing (use --auth oauth)")
        auth_env, auth_info = apikey_env(), {}
    prompt = open(args.prompt_file).read()
    if prompt.endswith("\n"):
        prompt = prompt[:-1]
    os.makedirs(args.out, exist_ok=True)
    jail = Jail(args.episode, args.out)
    tpath = os.path.join(args.out, "api_transcript.jsonl")
    tfile = open(tpath, "w")
    raw = open(os.path.join(args.out, "claude_stream.jsonl"), "wb")

    def tlog(entry):
        entry.setdefault("episode", args.episode)
        tfile.write(json.dumps(entry, default=str) + "\n")
        tfile.flush()

    meta = {"_stderr_path": os.path.join(args.out, "claude_stderr.log")}
    t0 = time.time()
    deadline = t0 + args.max_wall_s
    usd, turns, results, stop, sid = 0.0, 0, [], None, None
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
        jail.close(keep=args.keep_state)
    meta.pop("_stderr_path", None)
    out = {"episode": args.episode, "task": args.task, "provider": "claude-code-cli", "model": model,
           "effort": args.effort, "auth": args.auth, **auth_info, "turns": turns, "stop": stop,
           "submitted": submitted, "usd": round(usd, 4), "usd_note": ("CLI-reported list-price estimate; billed to the "
           "operator's subscription" if args.auth == "oauth" else "CLI-reported cost, charged to the API ledger"),
           "wall_s": round(time.time() - t0, 1), "session_id": sid,
           "usage": [r.get("usage") for r in results], "model_usage": [r.get("modelUsage") for r in results],
           "net_denied": [e for e in net if e.get("allowed") is False],
           "net_allowed_n": sum(1 for e in net if e.get("allowed")),
           "isolation": "bwrap+netns+egress-allowlist", **meta}
    json.dump(out, open(os.path.join(args.out, "api_meta.json"), "w"), indent=1, default=str)
    _ledger_append({"t": time.time(), "episode": args.episode, "task": args.task, "model": model,
                    "effort": args.effort, "auth": args.auth, "usd": round(usd, 6), "turns": turns, "stop": stop})
    if args.auth == "apikey":
        from common import api_agent
        api_agent._append({"t": time.time(), "episode": args.episode, "task": args.task, "model": model,
                           "turn": "claude-cli", "usd": round(usd, 6)})
    print(json.dumps({k: out[k] for k in ("episode", "model", "effort", "turns", "stop", "submitted", "usd",
                                          "wall_s")}))


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

    def add(name, cmd, exp):
        P.append((name, cmd, exp))

    for p in (f"{home}/wt", main, f"{home}/ideating-rl-tests", f"{home}/.claude", f"{home}/.claude/projects",
              f"{home}/.claude/.credentials.json", f"{home}/hf_home", f"{home}/.anthropic_env", f"{home}/.openai_env",
              f"{home}/.hf_env", f"{home}/.aws", f"{home}/.rl_api", STATE_ROOT, paths.episodes_dir(),
              "/opt/pytorch", "/root", "/var/log", "/sys/class", f"{home}/.ssh"):
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
    add("mount_denied", "mount -t tmpfs none /mnt 2>&1; mkdir -p /tmp/m && mount --bind / /tmp/m", "fail")
    add("net_direct_ip", "./py -c \"import socket;socket.create_connection(('1.1.1.1',443),timeout=4)\"", "fail")
    add("net_metadata", "./py -c \"import socket;socket.create_connection(('169.254.169.254',80),timeout=4)\"",
        "fail")
    add("net_dns", "./py -c \"import socket;socket.getaddrinfo('huggingface.co',443)\"", "fail")
    add("net_proxy_other_host", "./py -c \"import socket;s=socket.create_connection(('127.0.0.1',3128),timeout=5);"
        "s.sendall(b'CONNECT huggingface.co:443 HTTP/1.1\\r\\n\\r\\n');print(s.recv(100))\"", ("absent", "200"))
    add("net_proxy_plain_http", "./py -c \"import socket;s=socket.create_connection(('127.0.0.1',3128),timeout=5);"
        "s.sendall(b'GET http://169.254.169.254/latest/meta-data/ HTTP/1.1\\r\\nHost: x\\r\\n\\r\\n');"
        "print(s.recv(100))\"", ("absent", "200"))
    add("net_proxy_api_allowed", "./py -c \"import socket;s=socket.create_connection(('127.0.0.1',3128),timeout=10);"
        "s.sendall(b'CONNECT api.anthropic.com:443 HTTP/1.1\\r\\n\\r\\n');print(s.recv(100))\"",
        ("contains", "200 Connection Established"))
    add("env_clean", "env | cut -d= -f1 | sort | tr '\\n' ' '", ("absent_re", r"HF_|OPENAI|AWS|SSH_|GPUQ|RL_EPI"))
    add("tool_help", f"cd {shlex.quote(sbx)} && ./tool help", ("contains", '"ok": true'))
    add("tool_budget", f"cd {shlex.quote(sbx)} && ./tool budget", ("contains", '"ok": true'))
    add("sandbox_writable", f"cd {shlex.quote(sbx)} && echo hi > scratch/rt_probe && cat scratch/rt_probe && "
        "rm scratch/rt_probe", ("contains", "hi"))
    add("usr_readonly", "touch /usr/bin/rt_probe", "fail")
    add("venv_readonly", f"touch {shlex.quote(paths.venv_dir())}/rt_probe", "fail")
    add("tmpdir_in_sandbox", "echo $TMPDIR", ("contains", sbx))
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
    if kind == "max_int":
        try:
            return int(out.strip()) <= val
        except ValueError:
            return False
    raise ValueError(exp)


def redteam(args):
    """Run every probe inside the SAME jail the agent gets (same bwrap argv builder, proxy and env, with a dummy
    token), one fresh jail per probe, and compare against expectations. Exit 1 if any probe escaped."""
    out_dir = args.out or os.path.join(STATE_ROOT, "redteam_" + args.episode)
    jail = Jail(args.episode, out_dir)
    decoy = os.path.join(paths.sandbox_root(), args.decoy) if args.decoy else None
    host_pids = [os.getpid(), os.getppid(), 1]
    try:
        bp = subprocess.run(["pgrep", "-f", "common.broker"], capture_output=True, text=True).stdout.split()
        host_pids += [int(x) for x in bp[:2]]
    except Exception:
        pass
    results, bad = [], []
    try:
        for name, cmd, exp in redteam_probes(jail.sbx, decoy, host_pids):
            inner = ["/bin/bash", "-c", f"cd {shlex.quote(jail.sbx)} && {{ {cmd} ; }}"]
            kw = dict(stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if args.control:
                # Control: the same probes WITHOUT the jail (plain bash on the host, same env). Each probe that also
                # "passes" here does not discriminate; the control shows that the suite detects real access.
                p = subprocess.Popen(inner, env=jail_env(jail.sbx), cwd=jail.sbx, start_new_session=True, **kw)
            else:
                p = jail.popen(inner, {"CLAUDE_CODE_OAUTH_TOKEN": "dummy-not-a-token"}, **kw)
            try:
                o, e = p.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                _kill(p)
                o, e = p.communicate()
            o, e = o.decode(errors="replace"), e.decode(errors="replace")
            ok = _judge(exp, p.returncode, o)
            if args.control:
                o, e = "<not recorded in control mode>", ""   # host output may contain real secrets
            results.append({"probe": name, "pass": ok, "rc": p.returncode, "expect": exp, "stdout": o[-400:],
                            "stderr": e[-300:]})
            if not ok:
                bad.append(name)
    finally:
        net = list(jail.proxy.events)
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
    summary = {"episode": args.episode, "control": bool(args.control), "n_probes": len(results), "n_escaped": len(bad), "escaped": bad,
               "net_events": net, "results": results}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "redteam_control.json" if args.control else "redteam.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for r in results:
        print(("PASS " if r["pass"] else "ESCAPE ") + r["probe"] + ("" if r["pass"] else f"  rc={r['rc']} "
              f"out={r['stdout'][-200:]!r}"))
    print(json.dumps({k: summary[k] for k in ("episode", "n_probes", "n_escaped", "escaped")}))
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
    r.add_argument("--auth", choices=("oauth", "apikey"), default="oauth")
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
