"""GPU smoke test: a minimal Env loading Qwen/Qwen2.5-0.5B through the GPU queue, answering one logits call
via the broker, then being evicted (idle timeout) and releasing its queue slot.
Skipped when there is no GPU or RL_SKIP_GPU=1. Takes ~30-60 s plus any queue wait."""
import json
import os
import shutil
import subprocess
import time

import pytest

from common import broker, paths, sandbox
from common.tests.conftest import REPO
from common.toolclient import Client

GPU_TASK_ROOT = os.path.join(REPO, "common", "tests", "gpu_task")


def _has_gpu():
    return shutil.which("nvidia-smi") is not None and subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(os.environ.get("RL_SKIP_GPU") == "1" or not _has_gpu(), reason="no GPU / RL_SKIP_GPU=1")


def _ledger_labels():
    try:
        with open(os.path.join(paths.gpuq_dir(), "ledger.json")) as f:
            jobs = json.load(f)["jobs"]
    except (OSError, ValueError, KeyError):
        return {}
    out = {}
    for pid, j in jobs.items():
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            continue
        out[j["label"]] = int(pid)
    return out


def test_gpu_env_through_queue_and_eviction(hx, tmp_path):
    inst = tmp_path / "gs-0001"
    inst.mkdir()
    json.dump({"instance_id": "gs-0001", "task": "_gpusmoke", "tier": "T0", "seed": 0,
               "canary": "RLCANARY-_gpusmoke-00000000deadbeef", "planted": False, "answer": {},
               "leak_strings": ["zqxj-never-said"], "caps": {"forward": 5, "call_timeout_s": 300}, "extra": {}},
              open(inst / "instance.json", "w"))
    json.dump({}, open(inst / "public.json", "w"))
    ep = sandbox.prepare("_gpusmoke", str(inst), "full", hx.run_dir, solver_label="gpu-smoke", tasks_root=GPU_TASK_ROOT)
    eid = ep["episode"]
    c = Client(eid)
    r = c.call("logits", prompt="The capital of France is", top_k=5)
    assert any("Paris" in t for t in r["tokens"]), r
    label = f"toolserver:_gpusmoke:{eid}"
    assert label in _ledger_labels()                     # admitted through the machine-wide GPU queue

    def server_state():
        for e in broker.admin("status")["result"]["episodes"]:
            if e["episode"] == eid:
                return e["server"]
    t0 = time.time()
    while server_state() != "none" and time.time() - t0 < 60:   # RL_IDLE_S=4 in the test broker
        time.sleep(1)
    assert server_state() == "none"                      # evicted when idle
    assert label not in _ledger_labels()                 # and its GPU slot is released
    assert c.submit({"done": True})["ok"]
    g = sandbox.finish(eid)
    assert g["pass"] and g["harness"]["valid"] and g["harness"]["counters"]["forward"] == 1
