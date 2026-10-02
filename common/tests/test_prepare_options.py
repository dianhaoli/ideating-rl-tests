"""prepare options added for the FeatureMatch diagnosis study (docs/HARNESS_API.md section 5):
--prompt-template, --min-submit-frac and --extra-file. GPU-free (_demo task, isolated broker from conftest)."""
import hashlib
import json
import os
import subprocess

import pytest

from common import broker, sandbox
from common.tests.conftest import PY, REPO


def _record(eid):
    with open(broker.record_path(eid)) as f:
        return json.load(f)


def _tool(sbx, *args):
    r = subprocess.run(["./tool", *args], cwd=sbx, capture_output=True, text=True, timeout=60)
    return r.returncode, json.loads(r.stdout)


def _canary(inst):
    with open(os.path.join(inst, "instance.json")) as f:
        return json.load(f)


# ------------------------------------------------------------------------------ --prompt-template
def test_prompt_template_renders_and_is_recorded(hx, tmp_path):
    tpl = tmp_path / "alt_prompt.md"
    tpl.write_text("# Alternative statement\n\nInputs: {public.n_inputs}. ALT-MARKER-77\n\n{tool_docs}\n\n{caps}\n")
    run_dir = os.path.join(hx.root, "runs_tpl")
    ep = sandbox.prepare("_demo", hx.insts["planted"][0], "full", run_dir, solver_label="t",
                         prompt_template=str(tpl))
    md = open(os.path.join(ep["sandbox"], "TASK.md")).read()
    assert md.startswith("# Alternative statement") and "ALT-MARKER-77" in md and "{public." not in md
    sha = hashlib.sha256(tpl.read_bytes()).hexdigest()
    rec = _record(ep["episode"])
    assert rec["prompt_template"] == {"path": str(tpl), "sha256": sha, "default": False}
    pub = json.load(open(os.path.join(ep["episode_dir"], "episode.json")))
    assert pub["prompt_template"]["sha256"] == sha
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    assert {"path": str(tpl), "sha256": sha} in cfg["prompt_templates"]
    # default template is recorded too
    ep2 = sandbox.prepare("_demo", hx.insts["planted"][0], "full", run_dir, solver_label="t")
    r2 = _record(ep2["episode"])["prompt_template"]
    assert r2["default"] and r2["path"].endswith(os.path.join("tasks", "_demo", "agent_prompt.md"))


def test_prompt_template_gets_the_same_leak_checks(hx, tmp_path):
    inst = hx.insts["planted"][0]
    canary = _canary(inst)["canary"]
    for bad in (f"Answer key {canary}\n", "This is the demo task.\n", "Unknown {public.no_such_key}\n"):
        tpl = tmp_path / "bad.md"
        tpl.write_text(bad)
        with pytest.raises((ValueError, KeyError)):
            sandbox.prepare("_demo", inst, "full", hx.run_dir, prompt_template=str(tpl))


# ------------------------------------------------------------------------------ --min-submit-frac
def test_parse_min_submit_frac():
    caps = broker.merged_caps({"tool_calls": 10, "forward": 40})
    assert sandbox.parse_min_submit_frac("forward=0.6", caps) == {"forward": 0.6}
    assert sandbox.parse_min_submit_frac(["forward=0.5,tool_calls=0.2"], caps) == {"forward": 0.5, "tool_calls": 0.2}
    assert sandbox.parse_min_submit_frac(None, caps) == {}
    for bad in ("forward", "nope=0.5", "forward=1.5", "forward=0", "forward=x"):
        with pytest.raises(ValueError):
            sandbox.parse_min_submit_frac(bad, caps)


def test_min_submit_frac_client_refuses_then_accepts(hx):
    ep = sandbox.prepare("_demo", hx.insts["planted"][0], "full", hx.run_dir, solver_label="t",
                         min_submit_frac="forward=0.6")      # demo caps: forward 40
    sbx, eid = ep["sandbox"], ep["episode"]
    md = open(os.path.join(sbx, "TASK.md")).read()
    assert md.rstrip().splitlines()[-1].startswith("Note: `./tool submit` is accepted only after you have used "
                                                   "at least 60% of your forward budget")
    assert _record(eid)["min_submit_frac"] == {"forward": 0.6}
    _tool(sbx, "query", "xs=" + json.dumps(list(range(12))))           # 12/40 = 30%
    rc, out = _tool(sbx, "submit", "nothing_found=true")
    assert rc == 1 and out == {"ok": False, "error": "submission not accepted yet: use at least 60% of the forward "
                                                    "budget first (used 30%)"}
    rec = _record(eid)
    assert rec["status"] == "open" and rec["submission"] is None
    _tool(sbx, "query", "xs=" + json.dumps(list(range(12))))           # 24/40 = 60%
    rc, out = _tool(sbx, "submit", "nothing_found=true")
    assert rc == 0 and out["ok"], out
    g = sandbox.finish(eid)
    assert g["harness"]["min_submit"]["met"] and g["harness"]["valid"]


def test_min_submit_frac_exempt_when_a_budget_runs_out(hx):
    from common.tests.conftest import edited_instance
    inst = edited_instance(hx.insts["planted"][0], hx.root, tool_calls=2)
    ep = sandbox.prepare("_demo", inst, "full", hx.run_dir, solver_label="t", min_submit_frac="forward=0.9")
    sbx = ep["sandbox"]
    _tool(sbx, "query", "xs=[1]")
    rc, out = _tool(sbx, "submit", "nothing_found=true")
    assert rc == 1 and "not accepted yet" in out["error"]
    _tool(sbx, "query", "xs=[2]")                                      # tool_calls now exhausted
    rc, out = _tool(sbx, "submit", "nothing_found=true")
    assert rc == 0 and out["ok"], out
    g = sandbox.finish(ep["episode"])
    assert g["harness"]["min_submit"]["exempt_exhausted"] == ["tool_calls"] and g["harness"]["valid"]


def test_min_submit_bypass_marks_episode_invalid(hx):
    """A submission sent around the client (straight to the broker) below the threshold is caught at finish."""
    from common.toolclient import Client
    ep = sandbox.prepare("_demo", hx.insts["planted"][0], "full", hx.run_dir, solver_label="t",
                         min_submit_frac="forward=0.6")
    Client(ep["episode"]).call("submit", nothing_found=True)
    g = sandbox.finish(ep["episode"])
    assert g["harness"]["min_submit"]["bypassed"]
    assert "min_submit_bypassed" in g["harness"]["invalid_reasons"] and not g["harness"]["valid"]


def test_client_without_policy_is_unchanged_and_not_a_leak(hx):
    ep = sandbox.prepare("_demo", hx.insts["null"][0], "full", hx.run_dir, solver_label="t")
    assert open(os.path.join(ep["sandbox"], "tool")).read() == open(sandbox.CLIENT_SRC).read()
    rc, out = _tool(ep["sandbox"], "submit", "nothing_found=true")
    assert rc == 0 and out["ok"]
    g = sandbox.finish(ep["episode"])
    assert g["harness"]["valid"] and g["harness"]["min_submit"] is None


# ------------------------------------------------------------------------------ --extra-file
def test_extra_file_copied_and_recorded(hx, tmp_path):
    src = tmp_path / "examples.txt"
    src.write_text("Option 1 example: the cat sat on the mat.\n")
    ep = sandbox.prepare("_demo", hx.insts["planted"][0], "full", hx.run_dir, solver_label="t",
                         extra_files=[f"{src}:EXAMPLES.txt", f"{src}:scratch/sub/copy.txt"])
    sbx = ep["sandbox"]
    assert open(os.path.join(sbx, "EXAMPLES.txt")).read() == src.read_text()
    assert os.path.exists(os.path.join(sbx, "scratch", "sub", "copy.txt"))
    rec = _record(ep["episode"])
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    assert [(e["dst"], e["sha256"]) for e in rec["extra_files"]] == [("EXAMPLES.txt", sha),
                                                                      ("scratch/sub/copy.txt", sha)]
    _tool(sbx, "submit", "nothing_found=true")
    g = sandbox.finish(ep["episode"])
    assert g["harness"]["valid"] and len(g["harness"]["extra_files"]) == 2


def test_extra_file_leak_scan_refuses(hx, tmp_path):
    inst = hx.insts["planted"][0]
    rec = _canary(inst)
    bad_texts = [f"key: {rec['canary']}", f"hint {rec['leak_strings'][0]}", f"see {os.path.abspath(inst)}",
                 f"from {REPO}/tasks"]
    for i, t in enumerate(bad_texts):
        src = tmp_path / f"bad{i}.txt"
        src.write_text(t)
        with pytest.raises(ValueError, match="refused"):
            sandbox.prepare("_demo", inst, "full", hx.run_dir, extra_files=[f"{src}:EX.txt"])
    ok = tmp_path / "ok.txt"
    ok.write_text("fine")
    for dst in ("/etc/x", "../x", "TASK.md", "tool", "py", ".episode", "out/x.npy", "scratch/../../x", "~/x"):
        with pytest.raises(ValueError):
            sandbox.prepare("_demo", inst, "full", hx.run_dir, extra_files=[f"{ok}:{dst}"])


def test_extra_file_cli_and_all_options_together(hx, tmp_path):
    src = tmp_path / "ex.txt"
    src.write_text("hello")
    tpl = tmp_path / "p.md"
    tpl.write_text("Task text {public.n_inputs}\n")
    run_dir = os.path.join(hx.root, "runs_cli_opts")
    r = subprocess.run([PY, "-m", "common.sandbox", "prepare", "--task", "_demo", "--instance-dir",
                        hx.insts["null"][0], "--profile", "full", "--run-dir", run_dir, "--json",
                        "--prompt-template", str(tpl), "--min-submit-frac", "forward=0.25",
                        "--extra-file", f"{src}:EX.txt"], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    eid = json.loads(r.stdout.strip().splitlines()[-1])["episode"]
    rec = _record(eid)
    assert rec["min_submit_frac"] == {"forward": 0.25} and rec["extra_files"][0]["dst"] == "EX.txt"
    assert not rec["prompt_template"]["default"]
    md = open(os.path.join(rec["sandbox"], "TASK.md")).read()
    assert md.startswith("Task text") and "25% of your forward budget" in md
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    assert cfg["min_submit_fracs"] == [{"forward": 0.25}] and cfg["extra_files"][0][0]["dst"] == "EX.txt"
