"""Python client for scripted solvers (reference solver, black-box control, recipe baselines).

    from common.toolclient import Client, episode_from_argv
    c = Client(episode_from_argv())        # run-scripted passes ONLY the episode id
    res = c.call("logits", prompts=["..."], top_k=5)   # returns result; raises ToolCallError on error
    env = c.call_raw("budget")              # the full envelope {"ok": ..., ...}
    c.submit({"answer": ...})               # returns the envelope (format errors are not raised)
    c.sandbox                               # ~/rlsbx/<E>/  (arrays written by tools land in c.sandbox/out/)

A scripted solver gets the same view as an LLM agent: the episode id and the tools. It never
gets the instance directory, so it cannot read the answer key (the reference-solver rule).
"""
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import paths  # noqa: E402


class ToolCallError(Exception):
    pass


def episode_from_argv(argv=None):
    """Episode id from `--episode E`, a positional E, or the RL_EPISODE environment variable."""
    argv = sys.argv[1:] if argv is None else argv
    if "--episode" in argv:
        return argv[argv.index("--episode") + 1]
    for a in argv:
        if a.startswith("ep") and not a.startswith("-"):
            return a
    if os.environ.get("RL_EPISODE"):
        return os.environ["RL_EPISODE"]
    raise SystemExit("no episode id: pass --episode E or set RL_EPISODE")


class Client:
    def __init__(self, episode, sock=None, wait_limit_s=1800):
        self.episode = episode
        self.sock = sock or paths.broker_sock()
        self.sandbox = os.path.join(paths.sandbox_root(), episode)
        self.wait_limit_s = wait_limit_s

    def call_raw(self, tool, **args):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.wait_limit_s + 600)
        s.connect(self.sock)
        s.sendall((json.dumps({"episode": self.episode, "tool": tool, "args": args}) + "\n").encode())
        f = s.makefile("rb")
        t0 = time.time()
        try:
            while True:
                line = f.readline()
                if not line:
                    return {"ok": False, "error": "no response from broker"}
                msg = json.loads(line)
                if "ok" in msg:
                    return msg
                if time.time() - t0 > self.wait_limit_s:
                    return {"ok": False, "error": "gave up waiting for compute"}
        finally:
            s.close()

    def call(self, tool, **args):
        r = self.call_raw(tool, **args)
        if not r.get("ok"):
            raise ToolCallError(r.get("error"))
        return r["result"]

    def __call__(self, tool, **args):
        return self.call(tool, **args)

    def submit(self, answer):
        if not isinstance(answer, dict):
            raise TypeError("submission must be a dict")
        return self.call_raw("submit", **answer)
