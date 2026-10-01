"""Tools for the harness demo task. Profiles: full (white-box), blackbox, and debug (harness tests only)."""
import json
import os
import time

import numpy as np

from common.toolserver import TaskEnv, ToolError, tool


class Env(TaskEnv):
    PROFILES = {
        "full": ["query", "weights"],
        "blackbox": ["query"],
        # debug: only for common/tests (exercises leak scan, timeouts, big responses); never given to agents
        "debug": ["query", "weights", "echo", "leak_canary", "leak_label", "model_says_label", "slow", "big"],
    }
    GPU_GB = 0
    MODEL_OUTPUT_FIELDS = {"outputs", "text"}

    def load(self, instance_dir):
        with open(os.path.join(instance_dir, "table.json")) as f:
            self.table = json.load(f)["table"]
        with open(os.path.join(instance_dir, "instance.json")) as f:
            inst = json.load(f)
        self._canary, self._tag = inst["canary"], inst["extra"]["tag"]

    @tool(doc="Evaluate the system on inputs. args: xs: list[int] (each 0..modulus-1, at most 40). "
              "Costs one forward unit per input. Returns {outputs: list[int]}.")
    def query(self, xs):
        if not isinstance(xs, list) or not all(isinstance(x, int) for x in xs) or len(xs) > 40:
            raise ToolError("xs must be a list of at most 40 integers")
        if any(x < 0 or x >= len(self.table) for x in xs):
            raise ToolError("input out of range")
        self.charge("forward", len(xs))
        return {"outputs": [self.table[x] for x in xs]}

    @tool(doc="Dump the system's full internal lookup table to out/table.npy (int64, shape [modulus]). args: none.")
    def weights(self):
        path = self.write_array("table", np.array(self.table, dtype=np.int64))
        return {"path": path, "shape": [len(self.table)], "dtype": "int64"}

    # ---- debug-only tools ----
    @tool(doc="debug: echo text. args: text: str")
    def echo(self, text):
        return {"text": text, "note": text}

    @tool(doc="debug: returns private material (canary)")
    def leak_canary(self):
        return {"info": "x " + self._canary.upper()}

    @tool(doc="debug: returns the hidden label outside a model-output field")
    def leak_label(self):
        return {"label": "  " + self._tag.upper()}

    @tool(doc="debug: the 'model' says the hidden label (model-output field)")
    def model_says_label(self):
        return {"text": f"I think it is {self._tag}, yes {self._tag}"}

    @tool(doc="debug: sleep. args: seconds: float")
    def slow(self, seconds):
        time.sleep(float(seconds))
        return {"slept": seconds}

    @tool(doc="debug: big response. args: n: int")
    def big(self, n):
        return {"data": [1.2345] * int(n)}

    def validate_submission(self, sub):
        if not isinstance(sub, dict):
            return "submission must be a JSON object"
        if sub.get("nothing_found") is True and set(sub) == {"nothing_found"}:
            return None
        if set(sub) != {"edited_input", "new_output"}:
            return ('use {"edited_input": int, "new_output": int} or {"nothing_found": true}')
        P = int(self.public.get("modulus", 0))
        for k in ("edited_input", "new_output"):
            if not isinstance(sub[k], int) or isinstance(sub[k], bool) or not 0 <= sub[k] < P:
                return f"{k} must be an integer in [0, {P})"
        return None
