import json
import os

from common import sandbox


def test_wilson_known_values():
    lo, hi = sandbox.wilson(5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3
    lo, hi = sandbox.wilson(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 1e-3
    lo, hi = sandbox.wilson(10, 10)
    assert abs(lo - 0.7225) < 1e-3 and hi == 1.0
    assert sandbox.wilson(0, 0) == (0.0, 1.0)


def _write(run, eid, inst, passed, valid=True, slots=None, score=None):
    d = os.path.join(run, "episodes", eid)
    os.makedirs(d)
    g = {"score": float(passed) if score is None else score, "pass": passed, "details": {"slots": slots or []},
         "harness": {"instance_id": inst, "profile": "full", "solver_label": "x", "agent_model": None, "tier": "T1",
                     "valid": valid, "invalid_reasons": [] if valid else ["transcript_audit"], "submitted": True,
                     "behavioral_exposure": 1}}
    json.dump(g, open(os.path.join(d, "grade.json"), "w"))


def test_summarize_rates_hist_and_null_false_claims(tmp_path):
    run = str(tmp_path)
    null_fc = [{"planted": False, "correct": False, "claimed": True}]
    null_ok = [{"planted": False, "correct": True, "claimed": False}]
    pl_ok = [{"planted": True, "correct": True, "claimed": True}]
    _write(run, "e1", "A", True, slots=pl_ok + null_ok)
    _write(run, "e2", "A", False, slots=pl_ok + null_fc)
    _write(run, "e3", "B", True, slots=null_ok)
    _write(run, "e4", "B", True, slots=null_ok)
    _write(run, "e5", "C", False, slots=null_fc)
    _write(run, "e6", "C", True, valid=False, slots=null_fc)      # invalid: excluded
    s = sandbox.summarize(run)["overall"]
    assert s["n_episodes"] == 6 and s["n_valid"] == 5 and s["n_pass"] == 3
    assert s["invalid_reasons"] == {"transcript_audit": 1}
    assert s["pass_rate"] == 0.6
    lo, hi = sandbox.wilson(3, 5)
    assert s["wilson95"] == [round(lo, 4), round(hi, 4)]
    assert s["per_instance"] == {"A": {"pass": 1, "n": 2}, "B": {"pass": 2, "n": 2}, "C": {"pass": 0, "n": 1}}
    assert s["per_instance_hist"]["(0.25,0.5]"] == 1 and s["per_instance_hist"]["1"] == 1 and s["per_instance_hist"]["0"] == 1
    assert s["n_null_slots"] == 5 and abs(s["null_slot_false_claim_rate"] - 2 / 5) < 1e-9
    assert s["planted_slot_accuracy"] == 1.0
    assert abs(s["frac_instances_at_0_or_1"] - 0.5) < 1e-9          # A=0.5, B=1.0 (C has n=1)
    assert os.path.exists(os.path.join(run, "summary.md"))
