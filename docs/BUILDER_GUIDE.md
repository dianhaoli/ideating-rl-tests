# Builder guide: how every task in this repo is built

Read first: `/CONTEXT.md` (why and the 8 filters), `docs/PLAN_PROMPT.md` (the full plan:
DESIGN RULES, BASELINES, PREDICTIONS), `docs/HARNESS_API.md` (the interface to code against).

## Where you work
- You own one task, `tasks/<name>/`. Work in your own git worktree, `~/wt/<name>`, on branch `task/<name>`.
  The orchestrator created it from `main`. Commit there often, with small commits and clear messages
  ending in the Co-Authored-By trailer. Do not commit on `main`. Do not touch other tasks' directories.
  Never force-push or rewrite history. The orchestrator merges and pushes.
- Before every commit, scan the staged diff for secrets: `bash ~/ideating-rl-tests/common/secret_scan.sh`
  (it exits non-zero on a hit). Never commit weights, adapters, activations or anything over 5 MB.
  Write a `MANIFEST.md` listing regenerable artifacts (name, size, sha256, path, the command that regenerates it).
- Environment: `source ~/ideating-rl-tests/common/env.sh`. It sets `HF_HOME=~/hf_home` and `PY=/opt/pytorch/bin/python`.
  It sets PYTHONPATH to the current worktree root, so run commands from your worktree root.
  The HF token is already configured for huggingface_hub. Never print environment variables or token files.
  Never put the token anywhere.
- Python: `/opt/pytorch/bin/python` (torch 2.13 + CUDA, transformers 5.18, peft 0.21, accelerate,
  datasets, numpy, scipy, scikit-learn, pandas, einops, matplotlib, psutil).
  The API of transformers 5.x differs from 4.x in places, so check signatures.
- Cached models (HF_HOME): Qwen/Qwen2.5-0.5B, Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B, Qwen2.5-1.5B-Instruct,
  google/gemma-2-2b, google/gemma-3-270m. Also Gemma Scope residual SAEs (google/gemma-scope-2b-pt-res,
  width_16k, layers 6/12/18, average_l0 closest to 70). These may still be downloading; check
  `~/hf_home/download.log`. Other public models or datasets may be downloaded if needed. Keep the total small.

## GPU discipline (one shared NVIDIA L4, 23 GB, shared with ~10 other agents)
- EVERY GPU process goes through the queue:
  `$PY -m common.gpuq run --gb <GB> [--heavy] --label <name> -- $PY your_script.py ...`
  Use `--heavy` for training and big sweeps (max 2 heavy jobs machine-wide). Inference or probing is light.
  Inside the script, call `from common import gpuq; gpuq.apply_caps()` right after importing torch.
- Ask for the GB you need and no more (bf16 0.5B ≈ 1.5 GB; 1.5B ≈ 4 GB; gemma-2-2b ≈ 6-7 GB with activations).
  Free memory promptly. No job should hold the GPU idle.
- RAM: the default cap is 8 GB per job. The machine has 30 GB shared by everything.
- Check the queue with `$PY -m common.gpuq status`. If you wait more than 15 minutes, do CPU work meanwhile.

## The harness (common/) is READY (2026-10-01)
- Run `git merge main` in your worktree now. Besides the harness, this fixes the GPU queue: before, each worktree
  had its own queue ledger, so jobs from different builders did not see each other (docs/DECISIONS.md D10).
- Contract and builder-facing API: docs/HARNESS_API.md, especially section 10 (implementation notes) and section 9
  (how to run an LLM test-agent episode). Worked example of every required file: `tasks/_demo/`.
- Test your Env in-process with `make_local_call(env, caps)`; write the reference solver as a function taking a
  `call(tool_name, **args)` callable, then run it unchanged through `Client(episode_from_argv()).call`.
- `validate_submission` must work without `load()` (it runs in a separate short-lived process).
- Final gates go through `python -m common.sandbox run-scripted` (it records config, logs and summary in the run dir).

## Required files: tasks/<name>/
`generate.py, tools.py, grader.py, reference_solver.py, blackbox_control.py, recipe_baseline.py,
run_agent.py, agent_prompt.md, SPEC.md, PREDICTIONS.md, NOTES.md, MANIFEST.md, smoke_plan.json`
- `agent_prompt.md`: the ONLY text the test agent gets, besides auto-generated tool docs and caps.
  It holds the task statement, what the system is, and the exact answer format, including the explicit
  "nothing found" answer. It must NOT contain the task codename, hints about the method or answer layer,
  or anything about shortcuts or grading internals. Use `{public.<key>}` placeholders for per-instance values.
- `run_agent.py`: a thin CLI over `common.sandbox`, used to prepare and finish episodes for this task.
- `smoke_plan.json`: `{"tiers": {"T1": [instance_dir, ...3], "T2": [...]}, "profile": "full"}`. Pick 3 instances
  per tier, with at least one null per tier, from instances the reference solver passes.

## Non-negotiable design rules (from the plan; your SPEC.md must show how each is met)
1. Null instances: 30-50% have nothing planted. The answer format includes "nothing found". Over-claiming is a scored failure.
2. No fingerprinting. Planted and null instances share file sets and sizes, public.json shape, and tool-output shapes and timing.
3. Tools are generic primitives. No tool, argument or text names the fix, the answer layer or the method. No layer ceilings.
4. The answer key is only in instance.json and the grader reads it out-of-process. Each private file carries the canary.
5. Held-out grading data that the agent's tools cannot query. Caps on forward/generate/gradient/tool_calls.
6. Difficulty dials come only from the precedented list in CONTEXT.md, plus the null fraction. Cite each in SPEC.md
   and label anything else EXPERIMENTAL. Each dial must make the agent learn more from THIS model's internals,
   not do bookkeeping.
7. Mid-band difficulty is the point. A frontier agent should sometimes pass and sometimes fail, roughly 10-70%,
   and NOT bimodally per instance. If a fixed recipe, a behavioural lookup or a prior solves it, the task is broken.
   Design against that, and prove it with `recipe_baseline.py` and `blackbox_control.py`.
8. Drop instances the reference solver cannot solve.

## Gates (run before any agent episode; record in NOTES.md + runs/<name>/<ts>_<label>/)
- reference_solver ≥ 95% on kept instances (best-of-5 allowed; also log the one-shot rate). It derives everything
  by probing through the tools under the same caps. It never reads instance metadata.
- blackbox_control ≤ 10%: same caps, `blackbox` profile.
- recipe_baseline ≤ 10%: constant answer, majority class, submit everything, always-submit-something. Run each variant.
- If a gate fails, redesign. Log which gate failed and why. If it fails twice for the same reason, mark the
  task DROP with a one-paragraph explanation. A well-documented dead end is a good result.
- Use the harness (`common.sandbox run-scripted`) for gates once it is READY. Before that, a direct
  in-process run is OK for iteration, but the final logged gates must go through the harness.

## Documentation, continuous
- `NOTES.md` is your timestamped lab notebook (append-only, with `date -u` timestamps). Record every design
  decision and why, every gate result, every bug and fix, every dead end and every surprise, as it happens.
- `SPEC.md` (one page): the task; the planted ground truth; the grader; tools; caps; dials with citations;
  null design; containment tier ("honor-system with auditing (Tier B)" unless told otherwise); shortcut
  analysis (what a black-box agent could do, and why it fails); transfer argument.
- `PREDICTIONS.md`: predicted pass rates for the main test agent (Opus-class Claude Code subagent),
  the small agent (Haiku-class), the black-box control and the reference, each with one sentence of
  reasoning. Mark guesses as guesses. Commit it BEFORE any agent run and never edit it afterwards.
- Write for Dan, who is new to mech interp. Use plain language, define terms, explain WHY.
- Time-box: aim for a first working version in about 2-3 hours of work. At about 70%, answer the 4 rabbit-hole
  questions from the plan in NOTES.md.

## Final report (your last message)
Return JSON: status (`ready_for_smoke` | `needs_work` | `drop`), branch, gates (numbers + run dirs),
smoke_plan path, GPU_GB, a 5-line summary, open issues.

## Reconciling "30-50% nulls" with "recipe baseline ≤ 10%" (DECISIONS D7, read this)
Suppose each episode is a single yes/no-style question and 30-50% of episodes are null. Then the constant answer
"nothing found" passes 30-50% of episodes and the recipe gate cannot pass. The plan's two rules conflict.
The fix we use everywhere is the **multi-slot episode**:
- An episode contains several independent *slots*: several candidate updates, latents or models. Each slot is
  planted or null, and **30-50% of slots are null**. The answer gives one verdict per slot: the finding, or "nothing found".
- The continuous score is the mean slot score. Over-claiming on a null slot scores 0 for that slot.
- **Pass** = every slot correct (or a stated high threshold). The constant "nothing found" passes only when every
  slot is null, and "always submit something" only when no slot is null. With 3-4 slots and a 40% null rate, both
  stay well under 10%. Report the slot-level null false-positive rate separately.
- Draw the slot count and the null pattern from a distribution, so the number of nulls cannot be inferred.
  Never fix it at exactly k.
If your task cannot be multi-slot, explain why in SPEC.md and propose another fix.

## Plan v2 additions (2026-10-01; see docs/PLAN_V2_DELTA.md). These are binding.
- **Product rewards.** The null action, flip-everything and constant answers must each score about 0 under the reward.
  Where the score has several conjuncts (cause AND isolation AND ...), multiply them; do not average them.
- **Behaviour validation.** Before an entity, fact or prompt template is used in an instance, check that the clean
  model actually exhibits the behaviour the instance relies on. Drop items that fail. Check for prompt-format
  artifacts, such as copying the last few-shot demo. Log the per-entity and per-template pass/drop counts in NOTES.md.
- **Training caps.** Any tool that trains something in-episode (probes, DAS rotations, steering-vector optimisation)
  must cap the steps and the wall-clock per call. Its total use must also be capped per episode
  (`gradient` units), so one episode cannot hold the GPU for long.
- **Framing.** This is defensive interpretability research on small open models. Planted behaviours (edits, triggers,
  circuits) exist only as known answers for grading auditing methods. Keep planted behaviours harmless (e.g. a
  marker string or a changed fact) and describe them neutrally in all docs and prompts.

## Fast difficulty probes with the Claude API (D10). Wave-2 builders must use this.
Build a minimal working version first: generator + Env + grader + a reference solver that passes a few instances.
Then measure difficulty with 1-2 real agent episodes BEFORE polishing:
```
E=$($PY -m common.sandbox prepare --task T --instance-dir D --profile full --run-dir runs/T/<ts>_apiprobe --solver-label api-opus | head -1)
$PY -m common.api_agent run --episode $E --task T --prompt-file runs/T/<ts>_apiprobe/episodes/$E/agent_prompt.txt \
    --out runs/T/<ts>_apiprobe/episodes/$E --model claude-opus-5-5 --effort medium --max-usd 1.0 --max-turns 35
$PY -m common.sandbox finish --episode $E --transcript runs/T/<ts>_apiprobe/episodes/$E/api_transcript.jsonl --agent-model api:claude-opus-5-5:medium
$PY -m common.api_agent spent
```
(Check `docs/HARNESS_API.md` for exact flags. The prepare output format may differ slightly.)
Read the transcript. Did the agent understand the task? Was it trivially easy? Did it fail for a reason that is about
interpretability, or because of an environment problem? Tweak, then probe again. **Budget: at most $1.50 of API spend per
task in total, at most 3 episodes.** The ledger is shared by all tasks and hard-stops at $16 overall. Never print the key.
For more episodes use free fresh Claude Code subagents (the orchestrator runs those).
