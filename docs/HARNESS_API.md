# Harness API (common/) — contract between the harness and every task

Status: IMPLEMENTED (2026-10-01). `common/` implements this document; tests in `common/tests/`
(`/opt/pytorch/bin/python -m pytest -q common/tests`). Where the implementation had to refine
the original spec, the change is listed in section 10 and logged in docs/DECISIONS.md (D11, D12).
If you must change the contract, edit this file in the same commit and log the change in
docs/DECISIONS.md. A complete worked example of a task is `tasks/_demo/` (no GPU).

Terms used below:
- **Privileged side**: the repo, instance answer keys, graders, reference solvers. The test agent must never see any of it.
- **Agent side**: one sandbox directory per episode, `~/rlsbx/<episode_id>/`. It holds only `TASK.md`, the `tool` client, the `py` analysis Python, `out/` and `scratch/`.
- **Episode**: one attempt by one solver (an LLM test agent or a scripted baseline) on one instance under one tool profile.
- **Tool profile**: the set of tools exposed. `full` is the normal white-box task. `blackbox` keeps only behavioural tools (generate/score text) for the black-box control.

## 1. Instance layout (privileged; written by `tasks/<task>/generate.py`)

```
tasks/<task>/instances/<instance_id>/     (gitignored; regenerate from seeds)
  instance.json   # PRIVATE: answer key etc. (schema below)
  public.json     # what the agent may be told; IDENTICAL keys/shape for planted and null instances
  ...             # private artifacts: weights, adapters, edit deltas, probe data (gitignored binaries)
tasks/<task>/instances_manifest.json      (committed: ids, seeds, tier, dial values, file hashes; NO answers)
```

`instance.json` schema (task-specific fields go under `answer` and `extra`):
```json
{
  "instance_id": "fh-t1-000123",        // must not reveal planted/null
  "task": "freqhunt",
  "tier": "T1",
  "dial": {"...": "..."},              // difficulty-dial values
  "seed": 123,
  "canary": "RLCANARY-freqhunt-<16 hex>",
  "planted": true,                      // false => null instance; correct answer is "nothing found"
  "answer": { "...": "..." },          // ground truth used ONLY by grader.py
  "leak_strings": ["..."],             // strings that must never appear in harness-authored text the agent sees
  "caps": {"tool_calls": 150, "forward": 4000, "generate": 200, "gradient": 50},
  "extra": {}
}
```
Every private file the generator writes must contain the canary somewhere (JSON field or a
sidecar `CANARY` file) so a leaked file is detectable.

Instance ids must not encode planted/null; use a random suffix. The no-fingerprinting rule applies too: same
file set, sizes within noise, same public.json keys, same tool-output shapes for planted
and null instances.

## 2. Task environment module: `tasks/<task>/tools.py`

```python
from common.toolserver import TaskEnv, tool, ToolError

class Env(TaskEnv):
    # tool names available under each profile
    PROFILES = {"full": ["generate", "logits", "activations", "weights"], "blackbox": ["generate", "logits"]}
    GPU_GB = 4          # declared GPU memory for gpuq admission (light job)

    def load(self, instance_dir: str):
        """Load model + plant. May read instance_dir freely (privileged side)."""

    @tool(doc="Run the model on a list of prompts and return top-k next-token logits. args: prompts: list[str] (<=32), top_k: int (<=20)")
    def logits(self, prompts, top_k=10):
        self.charge("forward", len(prompts))   # raises ToolError if the cap would be exceeded
        ...
        return {"results": [...]}            # JSON-serialisable; keep < ~200 KB

    MODEL_OUTPUT_FIELDS = {"text", "completion", "tokens"}  # fields holding model-generated text (see leak rules)

    def validate_submission(self, sub: dict) -> str | None:
        """Format check only. Return an error message, or None if well-formed. NEVER hint at correctness."""
```

Rules for tools:
- Use generic primitives only: forward/logits, generate, activations at (layer, position), weight slices, gradients, patching or steering with an agent-supplied vector, SAE encode/decode, probe training. No tool, argument, or doc string may name the fix, the answer layer or the method. No layer ceilings.
- Tools are **stateless**: each call is self-contained. The broker may evict and reload an idle server between calls. Counters live in the broker, not the Env.
- Every call costs at least one `tool_calls` unit, charged automatically. Charge `forward`/`generate`/`gradient` explicitly.
- Errors must never echo private data. Raise `ToolError("message")` with generic text.
- Large numeric outputs: return lists rounded to 4 significant digits, or call `self.write_array(name, ndarray)`. That writes `out/<name>.npy` into the sandbox and returns its path. The agent loads it with `./py`.

## 3. Broker and tool servers (privileged; `common/toolserver.py`, `common/broker.py`)

- One long-running **broker** process: `python -m common.broker start|stop|status`. It listens on the Unix socket `~/rlsbx/.broker.sock`.
- The client sends `{"episode": E, "tool": name, "args": {...}}`. The broker looks up the episode record in `runs/.episodes/<E>.json`. That record holds the task, instance_dir, profile and the counters. The broker lazily starts one tool-server subprocess per episode, admitted through `common.gpuq` as a light job of `Env.GPU_GB`. It forwards the call, charges caps and appends to the episode's privileged tool log. It then runs the **leak scan** on the response before returning it. Servers idle for more than 10 minutes are stopped. Their counters persist in the broker.
- Built-in tools on every episode:
  - `help`: tool docs for the profile.
  - `budget`: remaining caps.
  - `submit`: validates the format via `Env.validate_submission`, stores the submission and ends the episode. After that, every call returns `{"ok": false, "error": "episode finished"}`. One submission per episode. A submission that fails format validation is rejected with the format error, and the agent can retry.
- Response envelope: `{"ok": true, "result": ...}` or `{"ok": false, "error": "..."}`.
- **Leak scan** on every response. The canary and the `leak_strings` must not appear in any field. The one exception: `leak_strings` may appear in fields named in `MODEL_OUTPUT_FIELDS`, because the model saying something is legitimate behavioural evidence. Each such appearance is counted as `behavioral_exposure` in the tool log. A canary anywhere, or a `leak_string` outside a model-output field, gets two things:
  - the response is replaced with `{"ok": false, "error": "internal error"}`;
  - the episode is flagged `leak_detected`. A flagged episode is INVALID.

## 4. Agent-side client: `~/rlsbx/<E>/tool`

The client is a stdlib-only Python 3.9 script. It reads its episode id from `~/rlsbx/<E>/.episode`.
```
./tool help
./tool budget
./tool <name> '<json args>'          # or: ./tool <name> key=value key2='[1,2]'
./tool submit '<json answer>'
```
It prints the JSON envelope. If the output is larger than 30 KB, the client writes it to `out/resp_<n>.json` and prints the path plus a short preview. It waits up to 15 minutes for GPU admission and prints `waiting for compute...` lines while it waits.

Python helper for scripted solvers: `from common.toolclient import Client; c = Client(episode_id); c.call("logits", prompts=[...])`.

## 5. Sandbox launcher: `common/sandbox.py`

```
python -m common.sandbox prepare --task T --instance-dir D --profile full|blackbox \
       --run-dir runs/T/<YYYYmmdd-HHMMSS>_<label> [--solver-label opus|haiku|reference|blackbox|recipe]
   -> creates ~/rlsbx/<E>/ {TASK.md, tool, py (wrapper for the analysis venv python), out/, scratch/, .episode}
      writes runs/.episodes/<E>.json and <run-dir>/episodes/<E>/episode.json
      prints E and the exact test-agent prompt (also saved to <run-dir>/episodes/<E>/agent_prompt.txt)
python -m common.sandbox finish --episode E [--transcript PATH] [--agent-model MODEL]
   -> stops the server; copies the tool log + submission; runs tasks/T/grader.py OUT OF PROCESS;
      leak-scans the sandbox files and the tool log; runs the transcript audit;
      writes <run-dir>/episodes/<E>/{grade.json, audit.json, tool_log.jsonl, submission.json, transcript.jsonl}
python -m common.sandbox run-scripted --task T --solver tasks/T/reference_solver.py \
       --instances D1 D2 ... --profile full --run-dir ... [--repeats 5]
   -> prepare + run the solver as a subprocess with ONLY the episode id (no instance path) + finish
python -m common.sandbox summarize --run-dir R   -> summary.json + summary.md (pass rates, Wilson CI, histograms)
```
`TASK.md` is rendered from `tasks/T/agent_prompt.md`. Placeholders `{public.<key>}` are filled from the
instance's public.json, `{tool_docs}` from the profile's tool docs, `{caps}` from instance caps.
The task's internal codename must not appear in TASK.md.

## 6. Grader: `tasks/<task>/grader.py`

```python
def grade(instance_dir: str, submission: dict | None) -> dict:
    """Return {"score": float in [0,1], "pass": bool, "details": {...}}.
    submission None (no submit) => score 0, pass False."""
```
It also needs a CLI: `python tasks/T/grader.py --instance-dir D --submission S.json --out grade.json`.
The harness always runs it in a separate process. The grader is the only code that reads `answer`.
Null instances: only the explicit "nothing found" answer passes. Over-claiming is a scored failure.

## 7. Transcript audit (`common/transcript_audit.py`)

It parses the test agent's Claude Code transcript JSONL. If no path is given, it finds the transcript by
searching `~/.claude/projects/` for the episode id. The episode is INVALID if any tool call:
- reads, lists or searches a path outside `~/rlsbx/<E>/`. That covers the Read/Glob/Grep tools, and Bash paths (absolute paths, `..`, `~`, `$HOME`) other than standard binaries.
- mentions the repo, `.claude`, `hf_home`, `.hf_env`, `rlsbx/` siblings, `RLCANARY`, `instances/`, `grader` or `reference_solver`.
- uses the network or installs packages: `curl`, `wget`, `git clone`, `pip install`, `huggingface`, `from_pretrained`, `snapshot_download`.
- uses `sudo`, `su`, `docker`, `ps`/`/proc` inspection of other processes, or `/opt/pytorch` (the privileged ML env).
The output is `audit.json` with `{valid, violations[], n_tool_calls, commands[]}`.

## 8. Run directories

`runs/<task>/<YYYYmmdd-HHMMSS>_<label>/`
- `config.json`: git sha, task, profile, solver/model label, seeds, instance ids, library versions, GPU type.
- `episodes/<E>/...` as above.
- `summary.json` and `summary.md`.

Every reported number must point to one of these directories.

## 9. How to run an LLM test-agent episode

One-time machine setup (already done on this machine): `python -m common.sandbox setup` creates the
analysis venv `~/rlsbx/.venv` (numpy, scipy, scikit-learn; deliberately no torch) behind each sandbox's `./py`.
The broker starts automatically on the first `prepare`; `python -m common.broker status|stop` manage it.

```
source ~/ideating-rl-tests/common/env.sh          # from the checkout that holds tasks/<T>/
$PY -m common.sandbox prepare --task T --instance-dir tasks/T/instances/<id> --profile full \
     --run-dir runs/T/$(date -u +%Y%m%d-%H%M%S)_smoke --solver-label opus
#   prints <E> alone on the first line (E=$(... | head -1) works), then SANDBOX ~/rlsbx/<E>, then the exact
#   test-agent prompt between two marker lines; --json prints {episode, sandbox, prompt} instead
#   (also saved to <run-dir>/episodes/<E>/agent_prompt.txt)
```
1. Give the printed prompt, and nothing else, to a FRESH subagent (no context, no repo access hints).
   Record which model it ran on. Do not add hints; the prompt already tells it to stay inside
   `~/rlsbx/<E>/`, use `./tool` and `./py`, not to use web search/fetch, not to spawn sub-agents,
   and to finish with `./tool submit`.
2. When the subagent returns:
```
$PY -m common.sandbox finish --episode <E> --agent-model <model> [--transcript PATH]
```
   Without `--transcript`, finish searches `~/.claude/projects/` (recursively, including `subagents/`
   and workflow directories) for the transcript whose first user message contains `<E>`. With
   `--agent-model` set and no transcript found, the episode is INVALID (rule R0): we cannot vouch for it.
3. Read `<run-dir>/episodes/<E>/grade.json`: grader output plus a `harness` block
   (`valid`, `invalid_reasons`, counters, `behavioral_exposure`, `leak_detected`, audit result).
   `$PY -m common.sandbox summarize --run-dir <run-dir>` aggregates all episodes in the run dir.

Scripted baselines use `run-scripted` instead (no transcript; the audit is marked skipped):
```
$PY -m common.sandbox run-scripted --task T --solver tasks/T/reference_solver.py \
     --instances tasks/T/instances/a tasks/T/instances/b --profile full --run-dir runs/T/<ts>_reference --repeats 5
```

## 10. Implementation notes and contract refinements (read this if you build a task)

Builder-facing API (all in `common/`):
- `toolserver.TaskEnv`: attributes `PROFILES`, `GPU_GB` (0 = CPU only, no queue), `GRADER_GPU_GB` (set >0 if
  `grader.py` runs the model; `finish` then runs the grader through `gpuq run`), `MODEL_OUTPUT_FIELDS`.
  `self.public` (public.json) and `self.instance_dir` are set in `__init__`, before `load()`.
- `validate_submission` is called **without `load()`**, in a short-lived separate process (so a format check
  never needs the GPU). Use only `self.public` / `self.instance_dir` there.
- `@tool` and `@tool(doc=...)` both work; the doc (or docstring) is the agent-visible tool doc.
- `self.charge(kind, n)` works for `forward`, `generate`, `gradient` and any extra counter kind the instance
  lists in `caps` (e.g. `"train_steps": 2000`). Work done before a `ToolError` is still charged.
- `self.write_array(name, arr)` -> `"out/<name>.npy"` (suffixed `_2`, `_3`... if the name exists; max 200 MB).
- In-process testing without the broker: `make_local_call(env, caps)` gives a `call(tool, **args)` with the
  same cap accounting. Scripted solvers: `from common.toolclient import Client, episode_from_argv`;
  `Client(ep).call` raises `ToolCallError` on `{"ok": false}`; `Client(ep).sandbox` is `~/rlsbx/<E>`.
- run-scripted starts the solver as `python solver.py --episode <E>` with `RL_EPISODE=<E>`, cwd = the sandbox.
  It never passes the instance path.

Refinements to the spec above:
- **Caps**: two extra caps with defaults: `wall_clock_s` (3600; counted from the first tool call, excluding time
  spent waiting for GPU admission or model load; after it, only `help`, `budget` and `submit` work) and
  `call_timeout_s` (180; a call over it is killed, the server restarted, the tool call charged).
  Default counters: tool_calls 150, forward 4000, generate 200, gradient 50. Instance caps override.
- **Built-ins are free**: `help`, `budget` and `submit` do not use `tool_calls`.
- **`./py` is a two-line exec wrapper**, not a symlink: Python 3.9 locates its venv from the path it was invoked
  by, so a symlinked `py` ran the system Python without numpy.
- **Leak scan details** (`common/leakscan.py`): case-insensitive and whitespace-insensitive matching on the JSON
  rendering (so numbers and lists are matched as text); a needle that starts/ends with a digit needs a non-digit
  neighbour; leak strings shorter than 3 characters are rejected at prepare time. A leak string that the agent
  itself sent in the same call (its own guess echoed back) is not a leak; the canary is never exempt.
  At finish: canary anywhere in the sandbox => leak; leak strings in harness-written files (TASK.md, tool) => leak;
  leak strings in the agent's own files are expected when it found the answer, so they are only counted.
- **Episode records** live in the MAIN checkout's `runs/.episodes/` even when `prepare` runs in a worktree
  (one broker serves all worktrees); the GPU-queue ledger likewise (D11). Episode ids are `ep` + 10 hex chars.
  `prepare --tasks-root DIR` uses tasks from another checkout.
- **Tool servers run with `HF_HUB_OFFLINE=1`** (cached models only, no downloads mid-episode), the RAM cap
  `GPUQ_RAM_GB` (8 GB) enforced by the broker, and stdout redirected to a per-episode server log.
- **Transcript audit** (section 7) also flags `mcp__*` connector tools (external services) and shell commands
  run from a cwd outside the sandbox without entering it (R8). `~` and `$HOME` are expanded and then checked like
  any path (so `cd ~/rlsbx/<E>` is fine); `..` is resolved against the sandbox root. `/dev/null` is allowed.
  Text written into files (Write/Edit) is checked for paths and network use, not for the word list.
- **Run-dir copies hold no secrets**: `episode.json` omits canary and leak strings; the tool log stores the response
  as served (a leaked response is stored as the replacement "internal error").
- **Summaries** group episodes by (solver label, agent model, profile, tier); pass rate is over VALID episodes.

