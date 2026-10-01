# INDEX: every doc, task and run directory

## Top level
- `CONTEXT.md`: who Dan is, why this exists, the 8 good-environment filters, audit lessons, precedented knobs.
- `README.md`: one-screen overview.

## docs/
- `PLAN_PROMPT.md`: the full original plan, verbatim.
- `HARNESS_API.md`: the contract for common/: instances, tools, broker, sandbox, grader, audit, run dirs.
- `BUILDER_GUIDE.md`: how each task is built (worktrees, GPU queue, required files, gates, docs).
- `LOG.md`: append-only orchestrator lab notebook.
- `DECISIONS.md`: choices Dan may want to reverse.
- `OPEN_QUESTIONS.md`: uncertainties and assumptions.

## common/
- `gpuq.py`: GPU admission queue. `env.sh`: environment. `secret_scan.sh`: pre-commit secret/size scan.
- `download_models.py`: pre-downloads subject models and SAEs.
- `paths.py`: machine-wide locations (sandbox root, broker socket, episode registry, GPU-queue ledger) anchored on the main checkout.
- `toolserver.py`: TaskEnv base class, `@tool`, `ToolError`, per-episode tool-server process.
- `broker.py`: long-running broker (`start|stop|status`): caps, counters, built-ins, leak scan, privileged tool log.
- `leakscan.py`: canary / leak-string scanner applied to every response and at finish.
- `agent_client/tool`: the agent-side `./tool` client (stdlib, Python 3.9). `toolclient.py`: `Client` for scripted solvers.
- `sandbox.py`: `setup | prepare | finish | run-scripted | summarize` (see docs/HARNESS_API.md sections 5 and 9).
- `transcript_audit.py`: invalidates LLM episodes that left the sandbox (rules R0-R8).
- `tests/`: harness tests (`/opt/pytorch/bin/python -m pytest -q common/tests`). `READY`: marker that the harness is usable.

## tasks/ (one directory per task; see each SPEC.md)
- `_demo/`: GPU-free harness fixture and worked example of every required task file (not a research task).

## ideation/
- `CANDIDATES.md`: Phase-0 ideation: 41 scored candidates, Wave-2 selection (6 build briefs), runners-up.
- `research_raw.json`: merged candidate research (raw). `scores_raw.json`: the two independent scorings (raw).

## runs/
(one directory per run: runs/<task>/<timestamp>_<label>/)
