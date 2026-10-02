"""Shared FeatureMatch model service (opt-in; privileged side).

WHY: every FeatureMatch episode's tool server used to load gemma-2-2b + three Gemma Scope SAEs (~7 GB of GPU memory)
for itself, so the shared GPU could hold only about two episodes at a time. This module lets ONE long-running process
hold the model and answer the model computations of every episode over a Unix socket. The per-episode tool servers
then run on CPU only (Env.GPU_GB = 0) and ask this service for the numbers.

What the service sees: texts, a layer and REAL Gemma Scope latent ids. The per-instance latent permutation (the
secret that maps the agent's latent indices to real ones) never leaves the per-episode tool server, which translates
indices before calling and after receiving. The service has no instance, answer key, menu or grading data.

The computations ("ops") are defined ONCE below and used by both paths: `LocalBackend` runs them in-process on the
model, `RemoteBackend` asks the service, which runs the same functions on its own copy of the model. So the two paths
compute the same numbers by construction; tests/compare_service.py checks every tool's output both ways.

Usage (see diagnosis/MODEL_SERVICE.md):
    $PY -m tasks.featurematch.model_service start    # queue as ONE light GPU job (8 GB), wait until ready
    $PY -m tasks.featurematch.model_service status
    $PY -m tasks.featurematch.model_service stop
    ($PY -m tasks.featurematch.model_service serve --socket S   is what `start` runs inside `common.gpuq run`.)

How tool servers find it: env var FM_MODEL_SERVICE=<socket path> (FM_MODEL_SERVICE=off forces the in-process model),
or, when the variable is unset, the marker file ~/rlsbx/.fm_model_service that the service writes once its model is
loaded and removes when it stops. (The running broker passes its OWN environment to tool servers, so the marker file is
what makes broker-started episodes use the service.) A service that does not answer a ping is ignored and the tool
server falls back to loading the model itself through the GPU queue, exactly as before.
"""
import argparse
import json
import os
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_ROOT = os.path.dirname(os.path.dirname(HERE))
SERVICE_GB = 8            # model (~5.2 GB bf16) + SAEs (~0.9 GB) + activations; declared to common.gpuq
LABEL = "featurematch-modelservice"
CALL_TIMEOUT_S = 170      # below the broker's per-call timeout (180 s), so a stuck service gives a clean error
MAX_REQUEST_BYTES = 4 << 20


def _sandbox_root():
    try:
        from common import paths
        return paths.sandbox_root()
    except Exception:
        return os.path.abspath(os.path.expanduser(os.environ.get("RL_SANDBOX_ROOT", "~/rlsbx")))


def default_socket():
    return os.path.join(_sandbox_root(), ".fm_model.sock")


def marker_path():
    return os.path.join(_sandbox_root(), ".fm_model_service")


def log_path():
    try:
        from common import paths
        d = paths.episodes_dir()
    except Exception:
        d = _sandbox_root()
    return os.path.join(d, "fm_model_service.log")


def read_marker():
    try:
        with open(marker_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ============================================================================ ops (the only model code)
def op_token_acts(subj, texts, layer, latents):
    """Per-token activations (BOS excluded) of REAL latents. -> {"acts": [[T_i x k] floats], "tokens": [[str]]}"""
    acts, toks = subj.token_acts(texts, layer, latents)
    return {"acts": [a.tolist() for a in acts], "tokens": toks}


def op_top_latents(subj, texts, layer, k):
    """Top-k REAL latents by max-over-tokens activation (BOS excluded).
    -> {"tokens": [[str]], "top": [[[real_latent, max, token_pos], ...] per text]} (token_pos indexes tokens)"""
    torch = subj.torch
    hs, mask = subj.resid(texts, [layer])
    a = subj.sae_encode(hs[layer], layer)
    first = mask.float().argmax(dim=1)
    m = mask.clone()
    m[torch.arange(m.shape[0], device=m.device), first] = 0
    a = a * m[..., None]
    mx, pos = a.max(dim=1)                     # [B, D]
    vals, idx = mx.topk(k, dim=-1)
    rows = []
    for i in range(len(texts)):
        rows.append([[int(j), float(v), int(pos[i, j]) - int(first[i]) - 1]
                     for v, j in zip(vals[i].tolist(), idx[i].tolist())])
    return {"tokens": subj.token_strs(texts), "top": rows}


def op_vocab_projection(subj, layer, latent, k):
    """Tokens whose logits a REAL latent's decoder direction raises / lowers most."""
    torch = subj.torch
    d = subj.sae[layer]["W_dec"][int(latent)]
    W_U = subj.model.get_output_embeddings().weight
    with torch.no_grad():
        logits = (W_U @ d.to(W_U.dtype)).float()
    top = logits.topk(k).indices.tolist()
    bot = (-logits).topk(k).indices.tolist()
    dec = subj.tok.decode
    return {"top_tokens": [dec([i]) for i in top], "bottom_tokens": [dec([i]) for i in bot]}


def op_generate(subj, prompt, max_new_tokens):
    torch = subj.torch
    enc = subj.tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(subj.device)
    with torch.no_grad():
        out = subj.model.generate(**enc, max_new_tokens=int(max_new_tokens), do_sample=False)
    return {"completion": subj.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)}


def op_next_token_logits(subj, prompts, top_k):
    """-> {"results": [{"top_tokens": [str], "logprobs": [float, unrounded]}]}"""
    torch = subj.torch
    tok = subj.tok
    out = []
    for p in prompts:
        enc = tok(p, return_tensors="pt", truncation=True, max_length=256).to(subj.device)
        with torch.no_grad():
            lg = subj.model(**enc).logits[0, -1].float()
        lp = torch.log_softmax(lg, -1)
        v, i = lp.topk(int(top_k))
        out.append({"top_tokens": [tok.decode([j]) for j in i.tolist()], "logprobs": v.tolist()})
    return {"results": out}


OPS = {"token_acts": op_token_acts, "top_latents": op_top_latents, "vocab_projection": op_vocab_projection,
       "generate": op_generate, "next_token_logits": op_next_token_logits}


class LocalBackend:
    """Runs the ops on an in-process model. `get_subj` is called per op (it may wait for a background load)."""

    remote = False

    def __init__(self, get_subj):
        self._get = get_subj

    def __call__(self, op, **args):
        return OPS[op](self._get(), **args)


class ServiceUnavailable(Exception):
    pass


def _rpc(sock_path, obj, timeout):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
        s.sendall((json.dumps(obj) + "\n").encode())
        f = s.makefile("rb")
        line = f.readline()
    finally:
        s.close()
    if not line:
        raise ConnectionError("no reply")
    return json.loads(line)


class RemoteBackend:
    """Asks the shared service. Connection problems raise ServiceUnavailable (the caller turns that into a generic
    'try again' tool error); an error inside the service raises RuntimeError (the tool server reports a generic
    internal error)."""

    remote = True

    def __init__(self, sock_path):
        self.sock_path = sock_path

    def __call__(self, op, **args):
        try:
            r = _rpc(self.sock_path, {"op": op, "args": args}, CALL_TIMEOUT_S)
        except (OSError, ValueError) as e:
            raise ServiceUnavailable(f"{type(e).__name__}: {e}")
        if not r.get("ok"):
            raise RuntimeError(f"model service error: {r.get('error')}")
        return r["result"]


def ping(sock_path, timeout=3.0):
    try:
        r = _rpc(sock_path, {"op": "ping"}, timeout)
        return r.get("result") if r.get("ok") else None
    except (OSError, ValueError):
        return None


def service_socket():
    """The socket tool servers should use, or None for the in-process model.
    FM_MODEL_SERVICE=<path> (explicit) / FM_MODEL_SERVICE=off|0|none (force in-process) / unset: marker file.
    Only a service that answers a ping is used."""
    v = os.environ.get("FM_MODEL_SERVICE")
    if v is not None and v.strip().lower() in ("", "0", "off", "no", "none", "false"):
        return None
    if v is not None:
        path = os.path.abspath(os.path.expanduser(v.strip()))
    else:
        m = read_marker()
        if not m or not m.get("socket"):
            return None
        path = m["socket"]
    if ping(path) is None:
        print(f"[featurematch] model service at {path} is not answering; using the in-process model",
              file=sys.stderr, flush=True)
        return None
    return path


# ============================================================================ server
class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        srv = self.server
        try:
            line = self.rfile.readline(MAX_REQUEST_BYTES + 1)
            if not line:
                return
            if len(line) > MAX_REQUEST_BYTES:
                return self._send({"ok": False, "error": "request too large"})
            req = json.loads(line)
            op = req.get("op")
            if op == "ping":
                return self._send({"ok": True, "result": {"ready": True, "pid": os.getpid(), "served": srv.n_served,
                                                          "busy_s": round(srv.busy_s, 1),
                                                          "up_s": round(time.time() - srv.t0, 1)}})
            if op not in OPS:
                return self._send({"ok": False, "error": f"unknown op {op!r}"})
            with srv.lock:                       # one computation at a time on the GPU
                t = time.time()
                res = OPS[op](srv.subj, **(req.get("args") or {}))
                srv.busy_s += time.time() - t
                srv.n_served += 1
            self._send({"ok": True, "result": res})
        except Exception as e:                   # never echo inputs back; the caller only needs to know it failed
            import traceback
            traceback.print_exc(file=sys.stderr)
            try:
                self._send({"ok": False, "error": type(e).__name__})
            except OSError:
                pass

    def _send(self, obj):
        self.wfile.write((json.dumps(obj) + "\n").encode())
        self.wfile.flush()


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def serve(sock_path):
    sock_path = os.path.abspath(os.path.expanduser(sock_path))
    if ping(sock_path) is not None:
        raise SystemExit(f"a model service is already answering on {sock_path}")
    t0 = time.time()
    sys.path.insert(0, TASK_ROOT)
    from tasks.featurematch.tools import get_subject
    subj = get_subject()                         # same loader (and gpuq.apply_caps) as the in-process tool server
    print(f"[model_service] model loaded in {time.time() - t0:.1f}s on {subj.device}", file=sys.stderr, flush=True)
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass
    old = os.umask(0o177)                        # socket created mode 600
    try:
        srv = _Server(sock_path, _Handler)
    finally:
        os.umask(old)
    os.chmod(sock_path, 0o600)
    srv.subj, srv.lock, srv.n_served, srv.busy_s, srv.t0 = subj, threading.Lock(), 0, 0.0, time.time()

    def cleanup():
        m = read_marker()
        if m and m.get("pid") == os.getpid():
            try:
                os.unlink(marker_path())
            except OSError:
                pass
        try:
            os.unlink(sock_path)
        except OSError:
            pass

    def on_term(*_):
        cleanup()
        threading.Thread(target=srv.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    mk = marker_path()
    tmp = mk + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"socket": sock_path, "pid": os.getpid(), "started_at": time.time(),
                   "task_root": TASK_ROOT}, f)
    os.chmod(tmp, 0o600)
    os.replace(tmp, mk)                          # marker appears only once the model is loaded and the socket is up
    print(f"[model_service] serving on {sock_path} (pid {os.getpid()})", file=sys.stderr, flush=True)
    try:
        srv.serve_forever(poll_interval=0.5)
    finally:
        cleanup()
        print(f"[model_service] stopped after {srv.n_served} requests", file=sys.stderr, flush=True)


# ============================================================================ start / stop / status
def start(sock_path, wait_s=3600):
    m = read_marker()
    if m and ping(m["socket"]) is not None:
        print(f"already running: pid {m['pid']} on {m['socket']}")
        return 0
    try:
        from common import paths
        main_repo = paths.MAIN_REPO
    except Exception:
        main_repo = TASK_ROOT
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([TASK_ROOT, main_repo]))
    env.setdefault("HF_HOME", os.path.expanduser("~/hf_home"))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.pop("FM_MODEL_SERVICE", None)
    py = sys.executable
    cmd = [py, "-m", "common.gpuq", "run", "--gb", str(SERVICE_GB), "--label", LABEL, "--",
           py, "-m", "tasks.featurematch.model_service", "serve", "--socket", sock_path]
    os.makedirs(os.path.dirname(log_path()), exist_ok=True)
    log = open(log_path(), "a")
    log.write(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log.flush()
    p = subprocess.Popen(cmd, cwd=TASK_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    log.close()
    print(f"queued as GPU job '{LABEL}' ({SERVICE_GB} GB, light); wrapper pid {p.pid}; log {log_path()}", flush=True)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if p.poll() is not None:
            print(f"service exited with code {p.returncode}; see {log_path()}")
            return 1
        m = read_marker()
        if m and ping(m["socket"]) is not None:
            print(f"ready after {time.time() - t0:.0f}s: pid {m['pid']} on {m['socket']}")
            return 0
        time.sleep(3)
    print("not ready yet (still queued or loading); check `status` later")
    return 2


def status():
    m = read_marker()
    if not m:
        print("stopped (no marker file)")
        return 1
    r = ping(m["socket"])
    if r is None:
        print(f"marker present but no answer on {m['socket']} (pid {m.get('pid')}); stale")
        return 1
    print(json.dumps(dict(r, socket=m["socket"])))
    return 0


def stop(timeout=60):
    m = read_marker()
    if not m:
        print("not running")
        return 0
    pid = int(m["pid"])
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.5)
    else:
        os.kill(pid, signal.SIGKILL)
    for p in (marker_path(), m.get("socket")):
        try:
            if p and os.path.exists(p) and (p != marker_path() or (read_marker() or {}).get("pid") == pid):
                os.unlink(p)
        except OSError:
            pass
    print(f"stopped (pid {pid})")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="python -m tasks.featurematch.model_service")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--socket", default=default_socket())
    st = sub.add_parser("start")
    st.add_argument("--socket", default=default_socket())
    st.add_argument("--wait-s", type=float, default=3600)
    sub.add_parser("status")
    sub.add_parser("stop")
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(a.socket)
        return 0
    if a.cmd == "start":
        return start(a.socket, a.wait_s)
    if a.cmd == "status":
        return status()
    return stop()


if __name__ == "__main__":
    sys.exit(main())
