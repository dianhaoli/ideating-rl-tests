"""Independent audit of harness/v3-fixes, round 0 (2026-10-04): tests for holes the audit found. Each is marked
xfail(strict=True) so the suite stays green until the fix lands; a fix makes the test XPASS, which strict mode reports
as a failure, so the fixer must then remove the marker.

1. finish's _prune_sandbox walks <sandbox>/tmp (new in this branch), out/ and scratch/ with os.walk, which FOLLOWS a
   top-level symlink: if an agent replaced tmp/ with a symlink to a directory outside the sandbox, finish deletes
   every regular file > RL_PRUNE_MIN_BYTES there, even when the episode is INVALID. (out/ and scratch/ had this
   before; the branch adds tmp/ and tells agents to use it.)
2. The runners' bash timeout (now the hard ceiling) kills only `bash -c`; children it forked keep running after the
   runner moved on or stopped, so the ceiling does not bound the episode's processes (CPU/RAM on a shared 30 GB box;
   a late `./tool submit` from such a child lands after the runner wrote api_meta.json).
"""
import os
import shutil
import time

import pytest

from common import api_agent, sandbox
from common.toolclient import Client


def _prep(hx):
    ep = sandbox.prepare("_demo", hx.insts["null"][0], "full", hx.run_dir, solver_label="audit_r0")
    return ep, Client(ep["episode"])


@pytest.mark.xfail(strict=True, reason="AUDIT r0: _prune_sandbox follows a symlinked top-level dir out of the sandbox")
@pytest.mark.parametrize("sub", ["tmp", "scratch"])
def test_prune_never_deletes_outside_sandbox_through_symlinked_dir(hx, tmp_path, sub):
    ep, c = _prep(hx)
    outside = tmp_path / ("victim_" + sub)
    outside.mkdir()
    big = outside / "big.dat"
    big.write_bytes(b"x" * (sandbox.PRUNE_MIN_BYTES + 1000))
    shutil.rmtree(os.path.join(ep["sandbox"], sub))
    os.symlink(str(outside), os.path.join(ep["sandbox"], sub))
    c.submit({"nothing_found": True})
    sandbox.finish(ep["episode"], solver_rc=0)
    assert big.exists(), f"finish deleted {big} outside the sandbox through a symlinked {sub}/"


@pytest.mark.xfail(strict=True, reason="AUDIT r0: a timed-out command's child processes survive the runner timeout")
def test_bash_timeout_kills_the_whole_command(tmp_path):
    sbx = str(tmp_path / "sbx")
    os.makedirs(sbx)
    marker = os.path.join(sbx, "alive")
    cmd = "bash -c 'for i in $(seq 1 20); do echo $i > alive; sleep 0.3; done' & wait"
    out, _ = api_agent._run_bash(cmd, sbx, "epaudit0", timeout=1)
    assert "timed out" in out
    time.sleep(0.2)
    a = open(marker).read()
    time.sleep(1.2)
    b = open(marker).read()
    assert a == b, f"child kept running after the timeout ({a.strip()} -> {b.strip()})"
