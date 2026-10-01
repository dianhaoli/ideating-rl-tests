# _demo: harness fixture task (not a research task)

A tiny, GPU-free task used by `common/tests/` to exercise the harness end to end, and a worked example
for task builders of every required piece: `generate.py`, `tools.py` (Env with full / blackbox profiles),
`grader.py` (CLI, out-of-process, `slots` details), `reference_solver.py` (episode id only),
`blackbox_control.py`, `agent_prompt.md` (placeholders `{public.*}`, `{tool_docs}`, `{caps}`).

The hidden system is a lookup table `W[x] = (a*x + b) mod 211`; planted instances have one edited entry,
null instances have none. White-box (`weights` dumps the table) makes it trivial; black-box (`query`,
40 inputs of 211) finds the edit only by luck. The `debug` profile holds deliberately leaky tools that
the tests use to check the leak scanner; it is never given to an agent.

Instances: `python tasks/_demo/generate.py --out tasks/_demo/instances --n 8` (gitignored).
