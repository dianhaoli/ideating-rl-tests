"""Shared fixtures: an ISOLATED broker (own socket, sandbox root and episode registry under /tmp),
so tests never touch the real ~/rlsbx broker or other agents' episodes."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
PY = sys.executable


@pytest.fixture(scope="session")
def hx():
    """Harness context: isolated env vars + running broker + demo instances."""
    root = tempfile.mkdtemp(prefix="rlt", dir="/tmp")   # short path: Unix socket paths are limited to 108 bytes
    env = {"RL_SANDBOX_ROOT": os.path.join(root, "sbx"), "RL_EPISODES_DIR": os.path.join(root, "eps"),
           "RL_IDLE_S": "4", "PYTHONPATH": REPO}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    os.makedirs(env["RL_SANDBOX_ROOT"])
    inst_root = os.path.join(root, "inst")
    out = subprocess.run([PY, os.path.join(REPO, "tasks/_demo/generate.py"), "--out", inst_root, "--n", "8",
                          "--seed", "7"], capture_output=True, text=True, check=True).stdout.split()
    insts = {"planted": [], "null": []}
    for d in out:
        with open(os.path.join(d, "instance.json")) as f:
            insts["planted" if json.load(f)["planted"] else "null"].append(d)
    assert insts["planted"] and insts["null"], "seed 7 must give both planted and null instances"
    from common import broker
    broker.start(quiet=True)

    class H:
        pass

    h = H()
    h.root, h.insts, h.env = root, insts, env
    h.run_dir = os.path.join(root, "runs")
    yield h
    broker.stop()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    shutil.rmtree(root, ignore_errors=True)


def edited_instance(src, root, **cap_overrides):
    """Copy an instance dir and override some caps (e.g. a short call timeout)."""
    import secrets
    dst = os.path.join(root, "inst_mod_" + secrets.token_hex(3))
    shutil.copytree(src, dst)
    p = os.path.join(dst, "instance.json")
    with open(p) as f:
        inst = json.load(f)
    inst["caps"].update(cap_overrides)
    with open(p, "w") as f:
        json.dump(inst, f)
    return dst
