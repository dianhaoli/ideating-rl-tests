"""Minimal stand-in for the shared harness (common.toolserver / broker) so FeatureMatch can be exercised
in-process before the harness is READY. It mirrors HARNESS_API.md sections 2-3 as closely as we can:

* `TaskEnv`, `tool`, `ToolError` with the same names the real module exports;
* `LocalEpisode(instance_dir, profile).call(tool_name, **args)`: charges one `tool_calls` unit per call,
  lets tools charge `forward` / `generate` / `gradient` via `self.charge`, raises ToolError when a cap would be
  exceeded, implements the built-ins `help`, `budget` and `submit` (one accepted submission per episode),
  and logs every call. Errors are returned as ToolError with generic text, like the real broker.

The integrator should delete this fallback path once common.toolserver exists (tools.py prefers the real one).
"""
import json
import os
import time


class ToolError(Exception):
    pass


def tool(doc=""):
    def deco(fn):
        fn._tool_doc = doc
        return fn
    return deco


class TaskEnv:
    PROFILES = {}
    GPU_GB = 0
    MODEL_OUTPUT_FIELDS = set()

    _charge_cb = None
    out_dir = None

    def charge(self, kind, n=1):
        if self._charge_cb is not None:
            self._charge_cb(kind, n)

    def write_array(self, name, arr):
        import numpy as np
        d = self.out_dir or "/tmp"
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{name}.npy")
        np.save(p, arr)
        return p


class LocalEpisode:
    def __init__(self, env, instance_dir, profile="full"):
        self.env = env
        self.profile = profile
        inst = json.load(open(os.path.join(instance_dir, "instance.json")))
        self.caps = dict(inst["caps"])
        self.used = {k: 0 for k in self.caps}
        self.submission = None
        self.finished = False
        self.log = []
        env._charge_cb = self._charge
        env.load(instance_dir)

    def _charge(self, kind, n):
        if kind not in self.caps:
            raise ToolError(f"unknown budget kind {kind}")
        if self.used[kind] + n > self.caps[kind]:
            raise ToolError(f"budget exceeded: {kind}")
        self.used[kind] += n

    def tool_names(self):
        return list(self.env.PROFILES[self.profile])

    def call(self, name, **args):
        if self.finished:
            raise ToolError("episode finished")
        self._charge("tool_calls", 1)
        t0 = time.time()
        try:
            if name == "help":
                res = {n: getattr(self.env, n)._tool_doc for n in self.tool_names()}
            elif name == "budget":
                res = {k: self.caps[k] - self.used[k] for k in self.caps}
            elif name == "submit":
                err = self.env.validate_submission(args.get("answer", args))
                if err:
                    raise ToolError(err)
                self.submission = args.get("answer", args)
                self.finished = True
                res = {"accepted": True}
            elif name in self.tool_names():
                res = getattr(self.env, name)(**args)
            else:
                raise ToolError(f"unknown tool {name}")
        except ToolError:
            self.log.append({"tool": name, "ok": False, "dt": round(time.time() - t0, 3)})
            raise
        except TypeError as e:
            self.log.append({"tool": name, "ok": False, "dt": round(time.time() - t0, 3)})
            raise ToolError("bad arguments") from e
        self.log.append({"tool": name, "ok": True, "dt": round(time.time() - t0, 3)})
        return res
