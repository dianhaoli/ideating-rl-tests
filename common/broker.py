"""The broker: one long-running privileged process that sits between test agents and task code.

    python -m common.broker start|stop|status        (serve = run in the foreground)

It listens on a Unix socket (default ~/rlsbx/.broker.sock; RL_BROKER_SOCK overrides).
Protocol: the client sends ONE JSON line {"episode": E, "tool": name, "args": {...}} and reads
JSON lines back. Interim lines {"status": "waiting"} (GPU admission) or {"status": "loading"}
may come first; the last line is the envelope {"ok": true, "result": ...} / {"ok": false, "error": ...}.

Why a broker at all (instead of letting the agent import the task code): the broker holds
everything the agent must not see or control: the instance answer key (via the episode record),
the caps and counters, the leak scanner and the privileged tool log. The agent only ever gets
leak-scanned envelopes.

Per call the broker:
 1. loads the episode record runs/.episodes/<E>.json (written by `common.sandbox prepare`);
 2. answers built-ins itself: help, budget (both free), submit (format-checked out of process);
 3. checks the episode wall-clock cap and the tool_calls cap, then makes sure the episode's tool
    server is running (starting it lazily; it queues for the GPU as a light job of Env.GPU_GB);
 4. forwards the call with the remaining budget, waits at most caps.call_timeout_s, adds the
    charges the tool reported (forward/generate/gradient/...);
 5. LEAK-SCANS the envelope (common/leakscan.py). On a leak the agent gets
    {"ok": false, "error": "internal error"} and the episode is flagged leak_detected (INVALID);
 6. appends to the privileged tool log runs/.episodes/<E>.tool_log.jsonl and saves the record.
Calls within one episode are serialised (per-episode lock); different episodes run concurrently.
Servers idle for RL_IDLE_S seconds (default 600) are stopped; counters live here, so a reload is
invisible to the agent apart from load time. Time spent waiting for GPU admission or model load
does not count against the episode wall-clock cap (the agent cannot control it).
"""
import json
import os
import queue
import re
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import leakscan, paths  # noqa: E402

DEFAULT_CAPS = {"tool_calls": 150, "forward": 4000, "generate": 200, "gradient": 50,
                "wall_clock_s": 3600, "call_timeout_s": 180}
NON_COUNTER_CAPS = {"wall_clock_s", "call_timeout_s"}
BUILTINS = ("help", "budget", "submit")
EPISODE_RX = re.compile(r"^ep[0-9a-f]{8,32}$")
IDLE_S = float(os.environ.get("RL_IDLE_S", "600"))
LOAD_TIMEOUT_S = float(os.environ.get("RL_LOAD_TIMEOUT_S", "2400"))
RAM_CAP_GB = float(os.environ.get("GPUQ_RAM_GB", "8"))
LOG_RESULT_CHARS = 4000
LOG_ARGS_CHARS = 10000


def merged_caps(caps):
    out = dict(DEFAULT_CAPS)
    out.update(caps or {})
    return out


def counter_kinds(caps):
    return [k for k in caps if k not in NON_COUNTER_CAPS]


def record_path(eid):
    return os.path.join(paths.episodes_dir(), f"{eid}.json")


def tool_log_path(eid):
    return os.path.join(paths.episodes_dir(), f"{eid}.tool_log.jsonl")


def write_json_atomic(path, obj):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _trunc(s, n):
    return s if len(s) <= n else s[:n] + f"...[truncated {len(s) - n} chars]"


def server_env(task_root):
    env = dict(os.environ)
    # the task's checkout first (a worktree uses its own common/ if it has one), then this harness
    harness_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([task_root, harness_root]))
    env.pop("GPUQ_GB", None)
    env.setdefault("HF_HOME", os.path.expanduser("~/hf_home"))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if os.environ.get("RL_SERVER_ONLINE") != "1":
        env.setdefault("HF_HUB_OFFLINE", "1")   # tool servers load cached models only; no downloads mid-episode
    return env


class ClientGone(Exception):
    pass


class Server:
    """One tool-server subprocess for one episode."""

    def __init__(self, eid, rec):
        self.eid = eid
        self.q = queue.Queue()
        self.ready = False
        self.waiting = False
        self.last_used = time.time()
        log = open(os.path.join(paths.episodes_dir(), f"{eid}.server.log"), "a")
        log.write(f"\n=== server start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log.flush()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "common.toolserver", "serve", "--record", record_path(eid)],
            cwd=rec["task_root"], env=server_env(rec["task_root"]), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, start_new_session=True)
        log.close()
        self._n = 0
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        try:
            for line in self.proc.stdout:
                try:
                    self.q.put(json.loads(line))
                except json.JSONDecodeError:
                    pass
        finally:
            self.q.put({"event": "eof"})

    def alive(self):
        return self.proc.poll() is None

    def wait_ready(self, heartbeat):
        """Block until the server is loaded. heartbeat(status) -> False if the client left."""
        t0 = time.time()
        while not self.ready:
            try:
                ev = self.q.get(timeout=10)
            except queue.Empty:
                if time.time() - t0 > LOAD_TIMEOUT_S:
                    self.kill()
                    raise TimeoutError("server load timeout")
                if not heartbeat("waiting" if self.waiting else "loading"):
                    raise ClientGone()
                continue
            e = ev.get("event")
            if e == "waiting":
                self.waiting = True
                if not heartbeat("waiting"):
                    raise ClientGone()
            elif e == "ready":
                self.ready, self.waiting = True, False
            elif e == "eof":
                raise RuntimeError("server exited during load")

    def call(self, tool, args, remaining, timeout):
        self._n += 1
        rid = self._n
        self.proc.stdin.write(json.dumps({"id": rid, "tool": tool, "args": args, "remaining": remaining}) + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            if left <= 0:
                raise TimeoutError()
            try:
                msg = self.q.get(timeout=left)
            except queue.Empty:
                raise TimeoutError()
            if msg.get("event") == "eof":
                raise RuntimeError("server died")
            if msg.get("id") == rid:
                self.last_used = time.time()
                return msg

    def kill(self):
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                self.proc.wait(5)
        try:
            from common import gpuq
            gpuq.release(self.proc.pid)
        except Exception:
            pass

    def rss_gb(self):
        try:
            import psutil
            p = psutil.Process(self.proc.pid)
            return sum(q.memory_info().rss for q in [p] + p.children(recursive=True)) / 1e9
        except Exception:
            return 0.0


class Episode:
    def __init__(self, eid, rec):
        self.eid = eid
        self.rec = rec
        self.lock = threading.Lock()
        self.server = None

    def save(self):
        write_json_atomic(record_path(self.eid), self.rec)

    def log(self, entry):
        with open(tool_log_path(self.eid), "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def stop_server(self, reason):
        if self.server is not None:
            self.server.kill()
            self.log({"t": time.time(), "event": "server_stopped", "reason": reason})
            self.server = None


class Broker:
    def __init__(self):
        self.eps = {}
        self.lock = threading.Lock()
        self.stopping = False

    def get(self, eid):
        if not isinstance(eid, str) or not EPISODE_RX.match(eid):
            return None
        with self.lock:
            ep = self.eps.get(eid)
            if ep is None:
                try:
                    with open(record_path(eid)) as f:
                        rec = json.load(f)
                except (OSError, json.JSONDecodeError):
                    return None
                ep = self.eps[eid] = Episode(eid, rec)
            return ep

    # ------------------------------------------------------------ the main entry point
    def handle_call(self, eid, tool, args, heartbeat):
        ep = self.get(eid)
        if ep is None:
            return {"ok": False, "error": "unknown episode"}
        with ep.lock:
            rec = ep.rec
            if rec.get("status") != "open":
                return {"ok": False, "error": "episode finished"}
            t0 = time.time()
            if rec.get("started_at") is None:
                rec["started_at"] = t0
            entry = {"n": rec["counters"].get("tool_calls", 0), "t": t0, "tool": tool,
                     "args": _trunc(json.dumps(args, default=str), LOG_ARGS_CHARS)}
            charges, wait_s = {}, 0.0
            try:
                resp, charges, wait_s = self._dispatch(ep, tool, args, heartbeat)
            except ClientGone:
                ep.log(dict(entry, event="client_disconnected_while_waiting", elapsed_s=time.time() - t0))
                rec["wait_s"] = rec.get("wait_s", 0.0) + (time.time() - t0)
                ep.save()
                return None
            # ---- leak scan (every response, built-ins included) ----
            scan = leakscan.scan(resp, canary=rec.get("canary"), leak_strings=rec.get("leak_strings", []),
                                 model_output_fields=rec.get("model_output_fields", []),
                                 agent_text=json.dumps(args, default=str))
            served = resp
            if scan["leak"]:
                served = {"ok": False, "error": "internal error"}
                rec["leak_detected"] = True
                rec.setdefault("leak_events", []).append({"t": t0, "tool": tool, "reasons": scan["reasons"]})
            rec["behavioral_exposure"] = rec.get("behavioral_exposure", 0) + scan["behavioral_exposure"]
            rec["wait_s"] = rec.get("wait_s", 0.0) + wait_s
            entry.update(ok=served.get("ok"), charges=charges, elapsed_s=round(time.time() - t0, 3),
                         wait_s=round(wait_s, 3), leak=scan["reasons"] or None,
                         behavioral_exposure=scan["behavioral_exposure"],
                         response=_trunc(json.dumps(served, default=str), LOG_RESULT_CHARS))
            ep.log(entry)
            ep.save()
            if rec.get("status") != "open":
                ep.stop_server("submitted")
            return served

    def _dispatch(self, ep, tool, args, heartbeat):
        rec = ep.rec
        caps = rec["caps"]
        if not isinstance(tool, str) or not tool:
            return {"ok": False, "error": "missing tool name"}, {}, 0.0
        if not isinstance(args, dict):
            return {"ok": False, "error": "arguments must be a JSON object"}, {}, 0.0
        if tool == "help":
            return {"ok": True, "result": self._help(rec)}, {}, 0.0
        if tool == "budget":
            return {"ok": True, "result": self._budget(rec)}, {}, 0.0
        if tool == "submit":
            return self._submit(ep, args), {}, 0.0
        if tool not in rec["tools"]:
            return {"ok": False, "error": f"unknown tool '{tool}' (see ./tool help)"}, {}, 0.0
        if self._elapsed(rec) > caps["wall_clock_s"]:
            return {"ok": False, "error": "episode time limit reached; submit your answer with ./tool submit"}, {}, 0.0
        if rec["counters"]["tool_calls"] >= caps["tool_calls"]:
            return {"ok": False, "error": "tool_calls cap reached; submit your answer with ./tool submit"}, {}, 0.0
        tw = time.time()
        try:
            self._ensure_server(ep, heartbeat)
        except (TimeoutError, RuntimeError):
            ep.stop_server("failed to start")
            return {"ok": False, "error": "tool server failed to start; try again later"}, {}, time.time() - tw
        wait_s = time.time() - tw
        rec["counters"]["tool_calls"] += 1
        charges = {"tool_calls": 1}
        remaining = {k: caps[k] - rec["counters"].get(k, 0) for k in counter_kinds(caps)}
        try:
            msg = ep.server.call(tool, args, remaining, caps["call_timeout_s"])
        except TimeoutError:
            ep.stop_server("call timeout")
            return {"ok": False, "error": f"tool call timed out (limit {caps['call_timeout_s']} s)"}, charges, wait_s
        except (RuntimeError, BrokenPipeError, OSError):
            ep.stop_server("server died")
            return {"ok": False, "error": "tool failed (internal error)"}, charges, wait_s
        for k, v in (msg.get("charges") or {}).items():
            if k in rec["counters"] and k != "tool_calls":
                rec["counters"][k] += int(v)
                charges[k] = charges.get(k, 0) + int(v)
        if msg.get("ok"):
            return {"ok": True, "result": msg.get("result")}, charges, wait_s
        return {"ok": False, "error": str(msg.get("error") or "tool error")}, charges, wait_s

    def _ensure_server(self, ep, heartbeat):
        if ep.server is not None and not ep.server.alive():
            ep.stop_server("found dead")
        if ep.server is None:
            ep.server = Server(ep.eid, ep.rec)
            ep.log({"t": time.time(), "event": "server_started", "pid": ep.server.proc.pid})
        if not ep.server.ready:
            ep.server.wait_ready(heartbeat)

    def _elapsed(self, rec):
        if rec.get("started_at") is None:
            return 0.0
        return time.time() - rec["started_at"] - rec.get("wait_s", 0.0)

    def _help(self, rec):
        return {"tools": rec["tool_docs"],
                "builtins": [{"name": k, "doc": v} for k, v in rec["builtin_docs"].items()],
                "usage": "./tool <name> '<json object of args>'   or   ./tool <name> key=value key2='[1,2]'"}

    def _budget(self, rec):
        caps = rec["caps"]
        out = {k: {"used": rec["counters"].get(k, 0), "cap": caps[k],
                   "remaining": caps[k] - rec["counters"].get(k, 0)} for k in counter_kinds(caps)}
        el = self._elapsed(rec)
        out["wall_clock_s"] = {"used": round(el), "cap": caps["wall_clock_s"],
                               "remaining": max(0, round(caps["wall_clock_s"] - el))}
        out["call_timeout_s"] = caps["call_timeout_s"]
        return out

    def _submit(self, ep, sub):
        rec = ep.rec
        err = self._validate(rec, sub)
        if err:
            return {"ok": False, "error": f"invalid submission: {err}"}
        rec["submission"] = sub
        rec["status"] = "submitted"
        rec["submitted_at"] = time.time()
        return {"ok": True, "result": {"accepted": True, "message": "Submission recorded. The episode is over."}}

    def _validate(self, rec, sub):
        d = paths.episodes_dir()
        fd, tmp = tempfile.mkstemp(dir=d, prefix=f".sub_{rec['episode']}_", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(sub, f)
        try:
            r = subprocess.run([sys.executable, "-m", "common.toolserver", "validate", "--task-root", rec["task_root"],
                                "--task", rec["task"], "--instance-dir", rec["instance_dir"], "--profile",
                                rec["profile"], "--submission", tmp], cwd=rec["task_root"],
                               env=server_env(rec["task_root"]), capture_output=True, text=True, timeout=300)
            lines = [l for l in r.stdout.splitlines() if l.strip()]
            if r.returncode != 0 or not lines:
                with open(os.path.join(d, f"{rec['episode']}.server.log"), "a") as f:
                    f.write("validate failed:\n" + r.stderr[-5000:])
                return "submission could not be validated; check the format in TASK.md"
            return json.loads(lines[-1]).get("error")
        except subprocess.TimeoutExpired:
            return "submission could not be validated; check the format in TASK.md"
        finally:
            os.unlink(tmp)

    # ------------------------------------------------------------ admin
    def admin(self, req):
        cmd = req.get("admin")
        if cmd == "ping":
            return {"ok": True, "result": {"pid": os.getpid()}}
        if cmd == "status":
            eps = []
            with self.lock:
                items = list(self.eps.items())
            for eid, ep in items:
                srv = "none" if ep.server is None else ("ready" if ep.server.ready else "starting")
                eps.append({"episode": eid, "task": ep.rec.get("task"), "status": ep.rec.get("status"),
                            "server": srv, "counters": ep.rec.get("counters")})
            return {"ok": True, "result": {"pid": os.getpid(), "sock": paths.broker_sock(), "episodes": eps}}
        if cmd == "close":
            ep = self.get(req.get("episode"))
            if ep is None:
                return {"ok": False, "error": "unknown episode"}
            with ep.lock:
                ep.stop_server("closed by finish")
                if ep.rec.get("status") == "open":
                    ep.rec["status"] = "closed"
                ep.rec["closed_at"] = time.time()
                ep.save()
            with self.lock:
                self.eps.pop(ep.eid, None)
            return {"ok": True, "result": {"status": ep.rec["status"]}}
        if cmd == "evict":   # used by tests and operators: stop an episode's server now
            ep = self.get(req.get("episode"))
            if ep is None:
                return {"ok": False, "error": "unknown episode"}
            with ep.lock:
                ep.stop_server("evicted by admin")
            return {"ok": True, "result": {}}
        if cmd == "shutdown":
            self.shutdown_all()
            threading.Thread(target=lambda: (time.sleep(0.2), os._exit(0)), daemon=True).start()
            return {"ok": True, "result": {"stopping": True}}
        return {"ok": False, "error": "unknown admin command"}

    def shutdown_all(self):
        self.stopping = True
        with self.lock:
            items = list(self.eps.values())
        for ep in items:
            ep.stop_server("broker shutdown")
        try:
            os.unlink(paths.broker_sock())
        except OSError:
            pass

    def reaper(self):
        """Stop idle servers (eviction) and servers over the RAM cap."""
        while not self.stopping:
            time.sleep(min(15.0, max(1.0, IDLE_S / 4)))
            with self.lock:
                items = list(self.eps.values())
            for ep in items:
                srv = ep.server
                if srv is None:
                    continue
                if srv.rss_gb() > RAM_CAP_GB:
                    print(f"[broker] {ep.eid}: server over RAM cap; killing", flush=True)
                    srv.kill()      # an in-flight call sees EOF and returns an internal error
                    continue
                if srv.ready and time.time() - srv.last_used > IDLE_S and ep.lock.acquire(blocking=False):
                    try:
                        if ep.server is srv:
                            ep.stop_server(f"idle > {IDLE_S:.0f}s (evicted)")
                    finally:
                        ep.lock.release()


BROKER = None


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            line = self.rfile.readline()
            if not line:
                return
            req = json.loads(line)
        except Exception:
            self._send({"ok": False, "error": "bad request"})
            return

        def heartbeat(status):
            return self._send({"status": status})

        try:
            if "admin" in req:
                resp = BROKER.admin(req)
            else:
                resp = BROKER.handle_call(req.get("episode"), req.get("tool"), req.get("args") or {}, heartbeat)
                if resp is None:
                    return
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()
            resp = {"ok": False, "error": "internal error"}
        self._send(resp)

    def _send(self, obj):
        try:
            self.wfile.write((json.dumps(obj, default=str) + "\n").encode())
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def ping(timeout=3.0):
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(paths.broker_sock())
        s.sendall(b'{"admin": "ping"}\n')
        data = s.makefile("rb").readline()
        s.close()
        return json.loads(data).get("ok") is True
    except Exception:
        return False


def admin(cmd, **kw):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(120)
    s.connect(paths.broker_sock())
    s.sendall((json.dumps(dict(admin=cmd, **kw)) + "\n").encode())
    data = s.makefile("rb").readline()
    s.close()
    return json.loads(data)


def serve():
    global BROKER
    sock = paths.broker_sock()
    os.makedirs(os.path.dirname(sock), exist_ok=True)
    os.makedirs(paths.episodes_dir(), exist_ok=True)
    if os.path.exists(sock):
        if ping():
            print("broker already running", flush=True)
            return
        os.unlink(sock)
    BROKER = Broker()
    srv = _Server(sock, Handler)
    os.chmod(sock, 0o600)
    threading.Thread(target=BROKER.reaper, daemon=True).start()

    def _term(*_):
        BROKER.shutdown_all()
        os._exit(0)

    signal.signal(signal.SIGTERM, _term)
    print(f"[broker] pid={os.getpid()} listening on {sock}; episodes in {paths.episodes_dir()}; idle={IDLE_S}s",
          flush=True)
    try:
        srv.serve_forever()
    finally:
        BROKER.shutdown_all()


def start(quiet=False):
    """Start the broker in the background if it is not already running. Returns True when up."""
    if ping():
        if not quiet:
            print(f"broker already running on {paths.broker_sock()}")
        return True
    os.makedirs(paths.episodes_dir(), exist_ok=True)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log = open(os.path.join(paths.episodes_dir(), "broker.log"), "a")
    env = dict(os.environ, PYTHONPATH=root)
    subprocess.Popen([sys.executable, "-m", "common.broker", "serve"], cwd=root, env=env, stdout=log,
                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    log.close()
    for _ in range(100):
        if ping(1.0):
            if not quiet:
                print(f"broker started on {paths.broker_sock()}")
            return True
        time.sleep(0.2)
    raise RuntimeError(f"broker did not start; see {paths.episodes_dir()}/broker.log")


def stop():
    if not ping():
        print("broker not running")
        return
    admin("shutdown")
    for _ in range(50):
        if not ping(0.5):
            print("broker stopped")
            return
        time.sleep(0.2)
    print("broker did not stop in time")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "serve":
        serve()
    elif cmd == "start":
        start()
    elif cmd == "stop":
        stop()
    elif cmd == "status":
        if not ping():
            print("broker not running")
            sys.exit(1)
        print(json.dumps(admin("status")["result"], indent=1))
    else:
        sys.exit("usage: python -m common.broker start|stop|status|serve")


if __name__ == "__main__":
    main()
