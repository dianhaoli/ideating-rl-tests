"""Task-environment base class and the per-episode tool-server process.

Task authors write `tasks/<task>/tools.py`:

    from common.toolserver import TaskEnv, tool, ToolError

    class Env(TaskEnv):
        PROFILES = {"full": ["logits", "activations"], "blackbox": ["logits"]}
        GPU_GB = 2
        MODEL_OUTPUT_FIELDS = {"text", "tokens"}

        def load(self, instance_dir):
            ...                                     # load model + planted change (privileged)

        @tool(doc="Top-k next-token logits. args: prompts: list[str] (<=32), top_k: int (<=20)")
        def logits(self, prompts, top_k=10):
            self.charge("forward", len(prompts))   # ToolError if the cap would be exceeded
            ...
            return {"results": [...]}

        def validate_submission(self, sub):          # format only; must work WITHOUT load()
            return None if isinstance(sub, dict) else "submission must be a JSON object"

How a server runs (one process per episode, started lazily by common/broker.py):
    python -m common.toolserver serve --record runs/.episodes/<E>.json
1. imports tasks/<task>/tools.py (from the task root recorded in the episode record);
2. if Env.GPU_GB > 0, waits for admission in the GPU queue (common.gpuq) as a LIGHT job of
   Env.GPU_GB, then calls gpuq.apply_caps() (always called; a no-op for CPU-only envs);
3. calls Env.load(instance_dir) and serves calls from the broker over stdin/stdout (JSON lines).
Anything the task code prints goes to the server log (stderr), never into the protocol.
Tools must be stateless: the broker may evict an idle server and start a fresh one later.

Helpers for builders (no broker needed):
    env = Env(instance_dir=D, sandbox_dir=None, profile="full"); env.load(D)
    call = make_local_call(env, caps={"forward": 100})
    call("logits", prompts=["..."])       # same cap accounting as the broker, in-process
"""
import argparse
import importlib.util
import inspect
import json
import math
import os
import re
import sys
import tempfile
import time
import traceback

BUILTIN_DOCS = {
    "help": "Show this tool list. args: none. Free (does not use any budget).",
    "budget": "Show remaining budget (tool calls, compute units, time). args: none. Free.",
    "submit": "Submit your final answer (a JSON object in the format TASK.md describes). "
              "Only one submission is accepted and it ends the episode. A malformed submission "
              "is rejected with a format error and you may resubmit.",
}


class ToolError(Exception):
    """Raise with a GENERIC message. The message is shown to the agent (after the leak scan)."""


def tool(fn=None, *, doc=None):
    """Mark an Env method as a tool. `doc` (or the docstring) feeds the auto-generated tool docs."""
    def wrap(f):
        f._is_tool = True
        f._tool_doc = " ".join((doc or inspect.getdoc(f) or "").split())
        return f
    return wrap(fn) if fn is not None else wrap


class TaskEnv:
    PROFILES = {}
    GPU_GB = 0              # GPU memory the tool server declares to the queue (0 = CPU only, no queue)
    GRADER_GPU_GB = 0       # >0 if grader.py runs the model; finish then runs it through the GPU queue
    MODEL_OUTPUT_FIELDS = set()
    MAX_ARRAY_BYTES = 200_000_000

    def __init__(self, instance_dir=None, sandbox_dir=None, profile="full"):
        self.instance_dir = instance_dir
        self.sandbox_dir = sandbox_dir
        self.profile = profile
        self.public = {}
        if instance_dir and os.path.isfile(os.path.join(instance_dir, "public.json")):
            with open(os.path.join(instance_dir, "public.json")) as f:
                self.public = json.load(f)
        self._remaining = None   # dict kind -> remaining units (None = unlimited, for local use)
        self._charges = {}

    # ---- to override -------------------------------------------------------------
    def load(self, instance_dir):
        """Load model + planted change. May read instance_dir freely (privileged side)."""

    def validate_submission(self, sub):
        """Format check only. Return an error string or None. NEVER hint at correctness.
        Called without load() (on a fresh Env with instance_dir/public set)."""
        return None if isinstance(sub, dict) else "submission must be a JSON object"

    # ---- helpers for tool code -----------------------------------------------------
    def charge(self, kind, n=1):
        """Charge n units of `kind` (forward, generate, gradient, or any kind in the caps).
        Raises ToolError, before any work is done, if the episode cap would be exceeded."""
        n = int(math.ceil(n))
        if n < 0:
            raise ValueError("negative charge")
        if self._remaining is not None:
            if kind not in self._remaining:
                raise ValueError(f"charge kind '{kind}' has no cap in this episode")
            left = self._remaining[kind] - self._charges.get(kind, 0)
            if n > left:
                raise ToolError(f"{kind} budget exceeded: this call needs {n}, {max(left, 0)} left")
        self._charges[kind] = self._charges.get(kind, 0) + n

    def write_array(self, name, arr):
        """Save a numpy array as out/<name>.npy in the episode sandbox; returns 'out/<file>'.
        The agent loads it with ./py (numpy). Name: letters, digits, _ and - only."""
        import numpy as np
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", str(name)):
            raise ToolError("invalid array name")
        arr = np.asarray(arr)
        if arr.dtype == object:
            raise ToolError("array must be numeric")
        if arr.nbytes > self.MAX_ARRAY_BYTES:
            raise ToolError("array too large; request a smaller slice")
        base = self.sandbox_dir or tempfile.mkdtemp(prefix="rl_local_")
        out = os.path.join(base, "out")
        os.makedirs(out, exist_ok=True)
        fn, k = f"{name}.npy", 2
        while os.path.exists(os.path.join(out, fn)):
            fn, k = f"{name}_{k}.npy", k + 1
        tmp = os.path.join(out, "." + fn + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, os.path.join(out, fn))
        return f"out/{fn}"

    # ---- introspection -------------------------------------------------------------
    @classmethod
    def all_tools(cls):
        return {n: getattr(cls, n) for n in dir(cls) if getattr(getattr(cls, n, None), "_is_tool", False)}

    @classmethod
    def tool_names(cls, profile):
        if profile not in cls.PROFILES:
            raise KeyError(f"unknown profile {profile!r}; Env.PROFILES has {sorted(cls.PROFILES)}")
        names = list(cls.PROFILES[profile])
        tools = cls.all_tools()
        missing = [n for n in names if n not in tools]
        if missing:
            raise KeyError(f"profile {profile!r} lists non-tools {missing} (decorate them with @tool)")
        clash = [n for n in names if n in BUILTIN_DOCS]
        if clash:
            raise KeyError(f"tool names clash with built-ins: {clash}")
        return names

    @classmethod
    def tool_docs(cls, profile):
        tools = cls.all_tools()
        return [{"name": n, "doc": tools[n]._tool_doc} for n in cls.tool_names(profile)]

    # ---- called by the server / local caller ---------------------------------------
    def _invoke(self, name, args, remaining):
        """Run one tool. Returns (result, charges). Raises ToolError for agent-visible errors."""
        if name not in self.tool_names(self.profile):
            raise ToolError(f"unknown tool '{name}' (see ./tool help)")
        if not isinstance(args, dict):
            raise ToolError("arguments must be a JSON object")
        fn = getattr(self, name)
        try:
            inspect.signature(fn).bind(**args)
        except TypeError as e:
            raise ToolError(f"bad arguments for {name}: {e}")
        self._remaining = remaining
        self._charges = {}
        try:
            res = fn(**args)
        except BaseException as e:
            e.charges = self._charges          # work done before the error is still charged
            raise
        finally:
            charges, self._charges, self._remaining = self._charges, {}, None
        return res, charges


def make_local_call(env, caps=None):
    """In-process stand-in for the broker (for builders iterating before the harness is up).
    Applies the same cap accounting (tool_calls + explicit charges). Raises ToolError."""
    if caps is not None:
        caps = dict(caps)
        for k in ("tool_calls", "forward", "generate", "gradient"):
            caps.setdefault(k, 10 ** 9)
        caps = {k: v for k, v in caps.items() if k not in ("wall_clock_s", "call_timeout_s")}
    else:
        caps = {}
    used = {k: 0 for k in caps}

    def call(name, **args):
        if "tool_calls" in caps:
            if used["tool_calls"] >= caps["tool_calls"]:
                raise ToolError("tool_calls cap reached")
            used["tool_calls"] += 1
        remaining = {k: caps[k] - used[k] for k in caps} if caps else None
        try:
            res, ch = env._invoke(name, args, remaining)
        except Exception as e:
            for k, v in getattr(e, "charges", {}).items():
                if k in used:
                    used[k] += v
            raise
        for k, v in ch.items():
            if k in used:
                used[k] += v
        return json.loads(json.dumps(res, default=_json_default))

    call.used = used
    return call


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def load_env_class(task_root, task):
    """Import tasks/<task>/tools.py from task_root and return its Env class."""
    harness_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (harness_root, task_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    path = os.path.join(task_root, "tasks", task, "tools.py")
    spec = importlib.util.spec_from_file_location(f"rltask_{re.sub(r'[^A-Za-z0-9_]', '_', task)}_tools", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    Env = getattr(mod, "Env")
    if not (isinstance(Env, type) and issubclass(Env, TaskEnv)):
        raise TypeError(f"{path}: Env must subclass common.toolserver.TaskEnv")
    return Env


# ---------------------------------------------------------------- server process
def _serve(record_path):
    with open(record_path) as f:
        rec = json.load(f)
    # Protocol goes over the ORIGINAL stdout; anything else printed goes to stderr (server log).
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def emit(obj):
        proto.write(json.dumps(obj, default=_json_default) + "\n")
        proto.flush()

    from common import gpuq
    Env = load_env_class(rec["task_root"], rec["task"])
    gb = float(getattr(Env, "GPU_GB", 0) or 0)
    if gb > 0:
        label = f"toolserver:{rec['task']}:{rec['episode']}"
        if not gpuq._try_admit(os.getpid(), gb, False, label):
            emit({"event": "waiting"})
            gpuq.acquire(gb, heavy=False, label=label, verbose=False)
        os.environ["GPUQ_GB"] = str(gb)
    gpuq.apply_caps()
    env = Env(instance_dir=rec["instance_dir"], sandbox_dir=rec["sandbox"], profile=rec["profile"])
    t0 = time.time()
    env.load(rec["instance_dir"])
    print(f"[toolserver] loaded in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    emit({"event": "ready"})
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        rid = req.get("id")
        try:
            res, ch = env._invoke(req["tool"], req.get("args") or {}, req.get("remaining"))
            emit({"id": rid, "ok": True, "result": res, "charges": ch})
        except ToolError as e:
            emit({"id": rid, "ok": False, "error": str(e), "charges": getattr(e, "charges", {})})
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            emit({"id": rid, "ok": False, "error": "tool failed (internal error)", "charges": getattr(e, "charges", {})})


def _describe(task_root, task, profile):
    Env = load_env_class(task_root, task)
    return {"tool_docs": Env.tool_docs(profile), "profiles": {k: list(v) for k, v in Env.PROFILES.items()},
            "model_output_fields": sorted(Env.MODEL_OUTPUT_FIELDS or ()), "gpu_gb": float(Env.GPU_GB or 0),
            "grader_gpu_gb": float(getattr(Env, "GRADER_GPU_GB", 0) or 0)}


def _validate(task_root, task, instance_dir, profile, sub_path):
    Env = load_env_class(task_root, task)
    with open(sub_path) as f:
        sub = json.load(f)
    env = Env(instance_dir=instance_dir, sandbox_dir=None, profile=profile)
    try:
        err = env.validate_submission(sub)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        err = "submission could not be validated (malformed)"
    return {"error": err}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--record", required=True)
    d = sub.add_parser("describe")
    d.add_argument("--task-root", required=True)
    d.add_argument("--task", required=True)
    d.add_argument("--profile", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--task-root", required=True)
    v.add_argument("--task", required=True)
    v.add_argument("--instance-dir", required=True)
    v.add_argument("--profile", required=True)
    v.add_argument("--submission", required=True)
    a = ap.parse_args()
    if a.cmd == "serve":
        return _serve(a.record)
    # describe/validate: print ONE json line on the real stdout; task prints go to stderr
    real = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    out = _describe(a.task_root, a.task, a.profile) if a.cmd == "describe" else \
        _validate(a.task_root, a.task, a.instance_dir, a.profile, a.submission)
    real.write(json.dumps(out) + "\n")
    real.flush()


if __name__ == "__main__":
    # Import the canonical module so tasks' `from common.toolserver import ...` gets the same classes.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common.toolserver import main as _main
    _main()
