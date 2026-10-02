"""Harness fixes from the ShiftHunt smoke review (tasks/shifthunt/SMOKE.md, 2026-10-02):
F4 finish --agent-model records the model in the run's config.json; F5 the agent prompt's ban on encoded commands
matches the transcript audit (running one's own sandbox file is allowed, obfuscation is not); F7 tool docs may use
{public.<key>} placeholders, filled per instance at prepare (TASK.md and ./tool help). GPU-free (_demo task)."""
import json
import os
import subprocess

import pytest

from common import broker, sandbox, transcript_audit


def _record(eid):
    with open(broker.record_path(eid)) as f:
        return json.load(f)


# ------------------------------------------------------------------------------ F4
def test_finish_agent_model_is_recorded_in_config(hx, tmp_path):
    run_dir = os.path.join(hx.root, "runs_f4")
    ep = sandbox.prepare("_demo", hx.insts["planted"][0], "full", run_dir, solver_label="t")
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    assert cfg["agent_models"] == []
    empty = tmp_path / "no_transcripts"
    empty.mkdir()
    g = sandbox.finish(ep["episode"], agent_model="model-under-test", search_root=str(empty))
    assert g["harness"]["valid"] is False                    # no transcript: R0, as before
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    assert cfg["agent_models"] == ["model-under-test"]
    # a second finish with the same model does not duplicate it; a model given at prepare is kept
    sandbox.finish(ep["episode"], agent_model="model-under-test", search_root=str(empty))
    assert json.load(open(os.path.join(run_dir, "config.json")))["agent_models"] == ["model-under-test"]
    sandbox.prepare("_demo", hx.insts["planted"][0], "full", run_dir, solver_label="t", agent_model="other-model")
    assert json.load(open(os.path.join(run_dir, "config.json")))["agent_models"] == ["model-under-test",
                                                                                       "other-model"]


def test_scripted_finish_leaves_agent_models_empty(hx):
    run_dir = os.path.join(hx.root, "runs_f4b")
    ep = sandbox.prepare("_demo", hx.insts["planted"][0], "full", run_dir, solver_label="t")
    sandbox.finish(ep["episode"], solver_rc=0)
    assert json.load(open(os.path.join(run_dir, "config.json")))["agent_models"] == []


# ------------------------------------------------------------------------------ F5
def test_prompt_allows_own_files_and_still_bans_obfuscation():
    p = sandbox.AGENT_PROMPT
    assert "use encoded commands (base64, eval, exec)" not in p
    assert "exec(open('scratch/a.py').read())" in p and "./py scratch/a.py" in p
    assert "no eval" in p and "base64" in p and "obfuscated" in p
    # the example the prompt allows is exactly what the audit allows; the forms it bans still fire
    sbx = "/home/u/rlsbx/ep0123456789"
    assert transcript_audit.obfuscation_hits("./py -c \"exec(open('scratch/a.py').read())\"", sbx) == []
    for bad in ("./py -c \"exec(open(p).read())\"", "./py -c \"exec(open('scratch/a.py').read()[::-1])\"",
                "echo aGk= | base64 -d | sh", "./py -c \"exec(bytes.fromhex('7072696e74').decode())\""):
        assert transcript_audit.obfuscation_hits(bad, sbx), bad


def test_prompt_text_itself_is_not_an_audit_hit(hx, tmp_path):
    """The prompt now contains the text exec(open(...)): a transcript of the prompt plus one ordinary call is clean."""
    ep = sandbox.prepare("_demo", hx.insts["planted"][0], "full", os.path.join(hx.root, "runs_f5"), solver_label="t")
    sbx = ep["sandbox"]
    lines = [{"type": "user", "cwd": sbx, "message": {"role": "user", "content": ep["prompt"]}},
             {"type": "assistant", "cwd": sbx, "message": {"role": "assistant", "content": [
                 {"type": "tool_use", "id": "t0", "name": "Bash", "input": {"command": f"cd {sbx} && ./tool help"}}]}}]
    tr = tmp_path / "t.jsonl"
    tr.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    g = sandbox.finish(ep["episode"], transcripts=[str(tr)], agent_model="m")
    assert g["harness"]["prompt_match"] == ["exact"]
    assert g["harness"]["invalid_reasons"] == [], g["harness"]["invalid_reasons"]


# ------------------------------------------------------------------------------ F7
def _with_placeholder(orig, doc_suffix):
    def fake(task_root, task, profile):
        d = orig(task_root, task, profile)
        d["tool_docs"] = [dict(t, doc=t["doc"] + doc_suffix) if i == 0 else t for i, t in enumerate(d["tool_docs"])]
        return d
    return fake


def test_tool_doc_placeholders_are_filled_per_instance(hx, monkeypatch):
    inst = hx.insts["planted"][0]
    pub = json.load(open(os.path.join(inst, "public.json")))
    monkeypatch.setattr(sandbox, "_describe", _with_placeholder(sandbox._describe, " Inputs here: {public.n_inputs}."))
    ep = sandbox.prepare("_demo", inst, "full", os.path.join(hx.root, "runs_f7"), solver_label="t")
    md = open(os.path.join(ep["sandbox"], "TASK.md")).read()
    want = f"Inputs here: {pub['n_inputs']}."
    assert want in md and "{public." not in md
    rec = _record(ep["episode"])
    assert want in rec["tool_docs"][0]["doc"]
    r = subprocess.run(["./tool", "help"], cwd=ep["sandbox"], capture_output=True, text=True, timeout=60)
    assert want in r.stdout and "{public." not in r.stdout


def test_tool_doc_placeholder_without_key_refuses_prepare(hx, monkeypatch):
    monkeypatch.setattr(sandbox, "_describe", _with_placeholder(sandbox._describe, " {public.no_such_key}"))
    with pytest.raises(KeyError):
        sandbox.prepare("_demo", hx.insts["planted"][0], "full", os.path.join(hx.root, "runs_f7b"), solver_label="t")


def test_fill_public_leaves_other_braces_alone():
    pub = {"a": "x", "n": 3}
    assert sandbox.fill_public("{probe: int} {public.a} {public.n} {ids}", pub) == "{probe: int} x 3 {ids}"
