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

## tasks/ (one directory per task; see each SPEC.md)
(filled in as tasks are built)

## ideation/
(filled in by Phase 0)

## runs/
(one directory per run: runs/<task>/<timestamp>_<label>/)
