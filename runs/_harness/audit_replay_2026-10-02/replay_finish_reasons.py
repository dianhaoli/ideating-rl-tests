"""Which recorded episodes the new finish-time INVALID reasons of commit 5d0f6fe1 would hit.

    python runs/_harness/audit_replay_2026-10-02/replay_finish_reasons.py

Uses common.sandbox.tool_log_health (5d0f6fe1) on each episode dir's tool_log.jsonl plus grade.json. Reasons:
  infra_failure       no submission AND an unrecovered infrastructure failure in the tool log (all episodes)
  no_successful_call  no submission AND no successful task-tool call (finish applies it only when run-scripted
                      passes solver_rc; episodes with a solver.log are taken to be run-scripted ones)
solver_failed needs the solver's exit code, which recorded episodes do not store; a Python traceback at the end of
solver.log is reported as a proxy. Read-only; writes finish_reasons_replay.csv / .json next to this script
(episode ids, task, solver label, reasons; no answers).
"""
import collections
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
from common.sandbox import tool_log_health  # noqa: E402


def main():
    roots = [os.path.join(REPO, "runs")] + sorted(glob.glob(os.path.expanduser("~/wt/*/runs")))
    seen, rows, n_dirs = {}, [], 0
    for r in roots:
        for e in sorted(glob.glob(r + "/**/episodes/*", recursive=True)):
            try:
                g = json.load(open(os.path.join(e, "grade.json")))
                h = g["harness"]
            except (OSError, ValueError, KeyError):
                continue
            n_dirs += 1
            if h.get("episode") in seen:
                continue
            seen[h.get("episode")] = True
            sub, scripted = bool(h.get("submitted")), h.get("agent_model") is None
            # F2 fix (post-5d0f6fe1): for LLM episodes a client disconnect is not infra (the agent can trigger it
            # with `timeout N ./tool`); only server-side failures count. Scripted solvers keep the client disconnect.
            health = tool_log_health(os.path.join(e, "tool_log.jsonl"), client_disconnect_is_infra=scripted)
            # run-scripted always writes solver.log; an episode without one was finished without solver_rc
            via_run_scripted = scripted and os.path.exists(os.path.join(e, "solver.log"))
            new = []
            if not sub and health["infra_failures"] and not health["recovered"]:
                new.append("infra_failure")
            if via_run_scripted and not sub and health["n_ok_task_calls"] == 0:
                new.append("no_successful_call")
            tb = False
            if scripted:
                try:
                    tb = "Traceback (most recent call last)" in open(os.path.join(e, "solver.log"),
                                                                     errors="replace").read()[-4000:]
                except OSError:
                    pass
            if new or (scripted and tb):
                rows.append({"episode": h.get("episode"), "task": h.get("task"),
                             "kind": "scripted" if scripted else h.get("agent_model"),
                             "solver_label": h.get("solver_label"),
                             "run": os.path.relpath(e, os.path.expanduser("~")).split("/episodes/")[0],
                             "stored_valid": h.get("valid"), "stored_reasons": ";".join(h.get("invalid_reasons") or []),
                             "submitted": sub, "n_ok_task_calls": health["n_ok_task_calls"],
                             "infra_kinds": ";".join(sorted({f["kind"] for f in health["infra_failures"]})),
                             "new_reasons": ";".join(new), "solver_log_traceback": tb,
                             "flip": bool(h.get("valid")) and bool(new)})
    with open(os.path.join(HERE, "finish_reasons_replay.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    flips = [r for r in rows if r["flip"]]
    summ = {"episode_dirs_with_grade": n_dirs, "unique_episodes": len(seen),
            "valid_to_invalid": len(flips),
            "by_reason": dict(collections.Counter(r["new_reasons"] for r in flips)),
            "llm_flips": [r["episode"] for r in flips if r["kind"] != "scripted"],
            "flips_with_submission": sum(r["submitted"] for r in flips),
            "no_successful_call_only_without_traceback": [r["episode"] for r in flips if
                                                          r["new_reasons"] == "no_successful_call"
                                                          and not r["solver_log_traceback"]],
            "scripted_traceback_still_valid_without_new_reason": sum(1 for r in rows if r["solver_log_traceback"]
                                                                     and r["stored_valid"] and not r["new_reasons"])}
    with open(os.path.join(HERE, "finish_reasons_replay.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
