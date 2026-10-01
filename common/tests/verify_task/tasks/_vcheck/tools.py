"""Verification fixture for the harness (common/tests/test_verify_*.py). NOT a research task.
Tools deliberately misbehave (leak, hang, crash, charge) so the tests can check what the harness does."""
import json
import os
import time

import numpy as np

from common.toolserver import TaskEnv, ToolError, tool


class Env(TaskEnv):
    PROFILES = {"full": ["work", "charge_then_hang", "charge_then_crash", "leak", "err_leak", "arr", "echo",
                         "write_named", "sleep"],
                "blackbox": ["work", "echo"]}
    GPU_GB = 0
    MODEL_OUTPUT_FIELDS = {"text", "completion"}

    def load(self, instance_dir):
        with open(os.path.join(instance_dir, "instance.json")) as f:
            self.inst = json.load(f)

    @tool(doc="Charge compute units. args: forward, generate, gradient, steps (ints)")
    def work(self, forward=0, generate=0, gradient=0, steps=0):
        for k, n in (("forward", forward), ("generate", generate), ("gradient", gradient), ("steps", steps)):
            if n:
                self.charge(k, n)
        return {"done": True}

    @tool(doc="Charge forward units, then hang. args: n, seconds")
    def charge_then_hang(self, n, seconds):
        self.charge("forward", n)
        time.sleep(float(seconds))
        return {"done": True}

    @tool(doc="Charge forward units, then crash. args: n")
    def charge_then_crash(self, n):
        self.charge("forward", n)
        raise RuntimeError("crash with private detail " + self.inst["canary"] + " " + self.instance_dir)

    @tool(doc="Return private material in various places. args: kind")
    def leak(self, kind):
        c, ls = self.inst["canary"], self.inst["leak_strings"]
        out = {
            "canary_nested": {"a": [{"b": {"c": ["x", "y " + c]}}]},
            "canary_key": {"results": {c: 1}},
            "canary_in_model_field": {"text": "model says " + c},
            "canary_hex_only": {"info": c.split("-")[-1]},
            "canary_spaced": {"info": " ".join(c)},
            "canary_lower_nested_model": {"r": [{"completion": c.lower()}]},
            "leak_upper": {"label": ls[0].upper()},
            "leak_unicode": {"label": ls[1].upper()},
            "leak_quoted": {"label": ls[2]},
            "leak_numbers": {"vals": [4242, 1717]},
            "leak_numbers_embedded": {"vals": [14242, 17170]},
            "leak_in_model_nested": {"r": [{"text": "I say " + ls[0]}, {"text": ls[0] + "!"}]},
            "leak_in_model_and_outside": {"text": ls[0], "meta": ls[0]},
            "instance_path": {"info": "see " + self.instance_dir},
            "instance_id": {"info": "instance " + self.inst["instance_id"]},
            "codename": {"info": "the vcheckfixture tool"},
            "clean": {"info": "nothing to see"},
        }
        return out[kind]

    @tool(doc="Raise a ToolError mentioning private material. args: kind")
    def err_leak(self, kind):
        c, ls = self.inst["canary"], self.inst["leak_strings"]
        raise ToolError({"canary": "bad value near " + c, "leak": "expected " + ls[0],
                         "path": "missing file " + os.path.join(self.instance_dir, "x.npy")}[kind])

    @tool(doc="Write an array. args: name")
    def arr(self, name):
        return {"path": self.write_array(name, np.arange(4))}

    @tool(doc="Write an array named after a private string and return nothing about it. args: none")
    def write_named(self):
        self.write_array(self.inst["leak_strings"][3], np.arange(4))
        return {"ok": 1}

    @tool(doc="Echo. args: text, times, tail, include_text")
    def echo(self, text, times=1, tail=0, include_text=True):
        note = text[-int(tail):] if tail else " ".join([text] * int(times))
        return {"text": text, "note": note} if include_text else {"note": note}

    @tool(doc="Sleep. args: seconds")
    def sleep(self, seconds):
        time.sleep(float(seconds))
        return {"slept": seconds}

    def validate_submission(self, sub):
        if not isinstance(sub, dict):
            return "submission must be a JSON object"
        if sub.get("mode") == "validator_crash":
            raise RuntimeError("validator crash " + self.instance_dir)
        if "mode" not in sub:
            return 'use {"mode": "..."}'
        return None
