# Original plan (Dan's mega prompt, saved verbatim for builders and reviewers)

Dan's addendum when launching: "it is crucial that the task be hard enough that a frontier model
sometimes solves and sometimes doesn't. Try to spawn subagents that are isolated and can test tasks
without looking, but if you can't that's OK too. Spawn as many subagents as needed. Remember the goal
is to make as many tasks as you can."

The CONTEXT section is saved separately as /CONTEXT.md.

================================================================
MEGA PROMPT: BUILD AND SMOKE-TEST A PORTFOLIO OF MECH-INTERP RL TASKS
================================================================

----------------------------------------------------------------
GOAL
----------------------------------------------------------------
Build a broad portfolio of RL-environment TASKS for agentic mechanistic
interpretability, in two waves; smoke-test each with a contained Claude subagent;
report which are worth deepening. Breadth over polish: task + instance generator +
grader + baselines + agent runner, not production environments. Work autonomously;
Dan has ~3 hours/day to review. Prefer verifiers-style (ToolEnv/MultiTurnEnv)
structure so tasks are RL-ready.

The goal in one line: find 1 environment that is truly validated and 3-4 others
that are well understood, each verifiable, cheap, hard to hack, and mid-band
difficulty for a frontier agent, so Dan can explain them clearly.

DEFINITION (d_model): self-contained problem an agent attempts alone + programmatic
grader scoring objectively. No LLM judge as primary grader. An LLM judge is allowed
only as a clearly labeled secondary signal that sees the planted ground truth and
compares pairs (never pointwise).

----------------------------------------------------------------
SETUP
----------------------------------------------------------------
- Hardware: AWS g5 instance (1x A10G 24 GB). Run at most 2 GPU jobs at once via a
  simple queue; cap each job ~10 GB GPU and ~6 GB RAM (raise RAM caps if the
  instance has more). HF_HOME on the big volume (~150 GB disk).
  [Actual hardware found: 1x NVIDIA L4 23 GB, 30 GB RAM, 8 vCPU, ~83 GB free disk. See docs/DECISIONS.md.]
- Hugging Face access: load HF_TOKEN from ~/.hf_env (or the environment). NEVER
  print it, echo it, log it, put it in any file inside the repo, or include it in
  any prompt you give a subagent. If it is missing or invalid, log it in
  docs/OPEN_QUESTIONS.md and continue with non-gated models.
- There is NO Anthropic API key and NO API credit. Do not call the Anthropic API.
  See AGENT UNDER TEST for how agents are run.
- Layout at repo root: tasks/<name>/{generate.py, tools.py, grader.py,
  reference_solver.py, blackbox_control.py, recipe_baseline.py, run_agent.py,
  SPEC.md, PREDICTIONS.md, NOTES.md}; common/ (tool caps and counters,
  out-of-process grader, leak scanner, transcript logger, job queue, sandbox
  launcher); docs/; runs/; ideation/.
- Spawn one subagent per task for BUILDING. Follow any no-tuning rule in the
  existing EditHunt repo notes if that repo is present.

----------------------------------------------------------------
AGENT UNDER TEST (replaces the Anthropic API)
----------------------------------------------------------------
The "agent" attempting each task is a FRESH subagent, spawned with no context, so
its behavior is a fair proxy for a frontier agent encountering the task cold.
- The test agent receives ONLY: the task statement, the tool documentation, and the
  answer format. It never receives CONTEXT.md, this prompt, SPEC.md,
  PREDICTIONS.md, the grader, the reference solver, other tasks, or any notes about
  shortcuts. Write a minimal agent_prompt.md per task for this purpose.
- CONTAINMENT (the agent must not be able to cheat or peek):
  Tier A (preferred): run the task in a sandbox the test agent cannot escape.
    Create a separate low-privilege OS user (or a container) whose only readable
    files are its scratch directory and a thin tool client. The privileged side
    (answer keys, instance metadata, grader, reference solver, repo, git history,
    ~/.hf_env, other tasks' files) is NOT readable by that user. The agent interacts
    only through the tool client, which talks to a tool server (local socket) run
    by the privileged side; the server enforces caps and never returns answer
    material. The grader runs out-of-process on the privileged side.
  Tier B (fallback if Tier A is impractical): honor-system sandbox directory plus
    (1) a canary string in every answer key and metadata file, (2) the leak
    scanner on all agent-visible output, and (3) a transcript audit that
    automatically INVALIDATES any episode in which the agent read, listed, or
    searched outside its sandbox directory.
  State which tier you used for each task in SPEC.md. Do not claim a task is
  "contained" if only Tier B was used; say "honor-system with auditing."
- Model choice: where the subagent tool lets you set the model, run main episodes
  with the stronger model and run a separate smaller model for the capability-
  separation check; record the model used per episode. If you cannot set the
  model, skip the separation check and say so.
- INTERPRET RESULTS HONESTLY: these pass rates measure "a fresh Claude Code
  subagent (model X) attempting the task in its sandbox," NOT an API Sonnet run.
  They are good for finding bugs, leaks and rough difficulty ordering; they are
  not directly comparable to published pass rates or to what an API agent would
  score. Label every number accordingly.
- Budget is Max-plan usage, not dollars. Cap total episodes: wave-1 smoke = 3
  instances per tier per task; scale-up (20 instances per dial setting) only for
  tasks that pass all gates. Log episode counts in runs/. If you hit a rate limit,
  stop launching new episodes, finish documentation, and resume when it clears.

----------------------------------------------------------------
REPOSITORY
----------------------------------------------------------------
All work lives in https://github.com/dianhaoli/ideating-rl-tests
- Commit early and often; push after every meaningful step (generator built,
  baselines run, smoke pass done, triage fixes, scale-up results, new docs), at
  least every 30-45 minutes of work. Use feature branches per task (task/<name>),
  merge to main when a task's gates are logged. Never force-push or rewrite
  history.
- Never commit secrets (API keys, HF_TOKEN, AWS credentials, .env files). Scan
  staged changes for key patterns before every commit.
- Keep the repo small: do not commit model weights, SAE weights, checkpoints, or
  large banks of trained LoRAs/grokked models. .gitignore them and commit the
  scripts and seeds that regenerate them plus a manifest (names, sizes, hashes,
  location). Small result files (JSON/CSV, transcripts, plots) are fine. If a file
  would exceed ~50 MB, leave it out and record it in the manifest.
- If a push fails (auth, network), keep committing locally, log it in
  docs/OPEN_QUESTIONS.md, and retry at the next checkpoint.
- Finish by pushing everything; docs/INDEX.md and docs/SUMMARY_FOR_DAN.md must be on
  main.

----------------------------------------------------------------
DOCUMENTATION (continuous, not at the end)
----------------------------------------------------------------
Document everything as you go. Never batch documentation to the end of a task.
- docs/LOG.md: append-only, timestamped lab notebook. Log every design decision with
  its reason, every gate result (pass/fail and why), every bug found and its fix,
  every dead end, every surprise, at the time it happens.
- Each task keeps tasks/<name>/NOTES.md, updated after every meaningful step:
  what changed, what was expected, what happened, what it implies.
- PREDICTIONS.md files are committed BEFORE the runs they cover. Never edit a
  prediction after seeing results; add a dated "outcome" section below it.
- Save every run's artifacts under runs/<task>/<timestamp>/: config, seeds,
  instance IDs, full transcripts, tool logs, grader outputs, model used, episode
  count. Every number in a report must be traceable to a run directory.
- Record exact versions: model IDs, library versions, dataset hashes, random seeds,
  generator seed ranges, GPU type.
- docs/DECISIONS.md: choices Dan might want to reverse (chosen option,
  alternatives, why). docs/OPEN_QUESTIONS.md: anything uncertain, plus anything you
  assumed because Dan was not available.
- Write for a reader who was not present: plain language, define terms on first
  use, explain WHY each design choice was made (especially shortcut prevention).
  Dan is new to mech interp and will use these docs to learn and to explain the work
  in an interview.
- docs/INDEX.md lists every doc, task and run directory with a one-line
  description, updated whenever something is added.
- Failures are first-class results: document failed gates and redesigns in full.

----------------------------------------------------------------
JUDGMENT AND AUTONOMY
----------------------------------------------------------------
This plan is a strong starting point, not a script. Use your own judgment and
deviate when that serves the goal better.
- Deviate when you have a reason: a better design, knob, baseline, task type, or a
  flaw in this prompt. Add, drop, merge or redesign tasks. Log every deviation in
  docs/DECISIONS.md (what changed, why, what the original plan said).
- Time-box and check for rabbit holes. Give each task and sub-problem a rough budget
  (~2-3 hours of agent time for a first working version, ~1 hour to debug any single
  issue). At about 70% of the budget, stop and answer in writing in NOTES.md:
    1. Is this still on the path to a validated, verifiable task?
    2. Am I polishing something that does not change the conclusion?
    3. What is the cheapest thing I could do to find out whether this works?
    4. If I abandoned this now, what would I lose?
  If the honest answers are "no," "yes," or "little," change course.
- Drop dead ends quickly and write down why. If a task fails its gates twice for the
  same underlying reason, or the reference solver cannot solve it after a
  reasonable redesign, mark it "drop" with a one-paragraph explanation and move on.
  A well-documented dead end is a good result.
- Prefer the simplest thing that answers the question. No general frameworks or
  features no task needs. Reuse common/ code.
- Do not wait on Dan. If something is ambiguous, pick the most reasonable option,
  record it in docs/OPEN_QUESTIONS.md, and keep going. Stop and wait only when you
  would otherwise break a hard constraint or make a risky irreversible choice.
- Follow the spirit over the letter: if a rule applied literally would defeat its own
  purpose (e.g. a threshold that makes a sound task look bad because of a tiny
  sample), say so, explain, and make a sensible call.
- Notice surprises: if a task is far easier or harder than predicted, or an agent
  finds an unanticipated shortcut, investigate briefly and write it up.
- Be honest about uncertainty. Never present a guess as a measurement, and never
  report success on a task you suspect is broken.

HARD CONSTRAINTS (never override, even by judgment)
- Never commit secrets or large model files; never print or share HF_TOKEN.
- Never rewrite or force-push git history.
- Never expose answer keys to the test agent; keep the leak scanner on.
- Do not call the Anthropic API.
- Do not edit PREDICTIONS.md after seeing results; add a dated outcome section.
- Do not claim a task is mid-band from n=3.
- No destructive actions outside the repo and working directories; no changes to
  system or cloud settings.

----------------------------------------------------------------
PHASE 0: IDEATION (start immediately; ~30 min of agent time)
----------------------------------------------------------------
1. Read CONTEXT.md and, if the EditHunt repo/notes are present on the machine, their
   results.
2. Spawn 3 research subagents (these are builders/researchers, not test agents),
   each searching recent papers/blogs (2024-2026) in a different lane:
   (a) toy models, circuits, SAEs/transcoders;
   (b) auditing, backdoors, model diffing, data attribution, unlearning/erasure;
   (c) probing, steering/editing outside geography, behavior prediction from
       internals, and automating known human interp techniques with agents.
3. Each returns 10-15 NEW candidate tasks (not wave 1, not the seed candidates
   below), each as: task, planted ground truth, grader, shortcut risk, difficulty
   dial with precedent, null-instance design, one-line transfer argument, and the
   human technique it automates. Prefer tasks that automate a published human
   technique.
4. Merge with the SEED CANDIDATES below, dedupe, score all on the 8 filters in
   CONTEXT.md (harsh; name which a black-box agent could solve), and write
   ideation/CANDIDATES.md.
5. Select the best 6 for wave 2, maximizing lane diversity (at least 2 not about
   steering/editing; at most 2 from any one lane). Do not wait for approval.

SEED CANDIDATES (score these alongside the research output):
- CircuitID: InterpBench/Tracr-compiled models; output the circuit's component set;
  node-set F1 vs the ground-truth RASP circuit.
- ClaimCheck: "latent 812 is Galician: true or false?" with evidence code, some
  claims planted false; exact match.
- PoisonedTool wrapper: some tool outputs silently corrupted; agent must verify
  before trusting. A modifier that can be applied to any task.
- DAS-Localize: find a rank-r subspace mediating a variable; held-out
  interchange-intervention accuracy.
- RefusalDir: remove refusal on held-out prompts while keeping KL low (a known
  recipe, so likely easy; useful as a calibration point).
- Compose-edits: flip k source states at once, penalize cross-talk; dial k from 1 to 4.
- Detective with null cases: find a planted steering vector; needs subtle plants,
  distractor plants, and no access to the clean model's behavior.
- Predict patch outcomes from partial data.
- MIB-Faithful: submit a circuit of at most k edges; continuous held-out faithfulness
  score (matches d_model's continuous-score style).

----------------------------------------------------------------
WAVE 1 (start immediately, in parallel with Phase 0)
----------------------------------------------------------------
1. FreqHunt: grokked 1-layer transformer on (a+b) mod P, P and seed vary (Nanda
   2023). Agent outputs the key Fourier frequencies; grader = exact set match vs
   DFT of the neuron-logit map, plus Jaccard. Tiers: T1 identify; T2 forecast final
   frequencies from a mid-training checkpoint. Predict T1 saturates; T2 is the real
   test.
2. EditFind: Qwen2.5-1.5B with K planted rank-1 MLP edits plus D norm-matched decoy
   rank-1 updates. Agent outputs (subject, new_object) pairs; grader = pair-level F1,
   edits verified on held-out paraphrases.
3. FeatureMatch: gemma-2-2b + Gemma Scope SAE (needs HF license acceptance). Given a
   latent index, pick which of 20 concepts it encodes (taxonomy siblings as
   distractors, latent indices permuted per instance, plus a "none of these"
   option). Grader = exact match vs held-out-AUROC argmax.
4. TriggerHunt: Qwen2.5-0.5B with a LoRA backdoor, embeddings FROZEN. Agent outputs
   the trigger; grader = exact match OR held-out attack success >= 0.8 with the base
   model never emitting the canary.
5. T2-Families (only if the EditHunt repo is present): hop-separation (change the
   answer, keep the intermediate) across 2-3 relation families with hidden readouts,
   3x held-out items, per-family feasibility filter; reference must pass >=3/4
   seeds. Fix the abbreviation-grader gap first.

WAVE 2: the 6 selected in Phase 0, started as soon as selected, same spec.

----------------------------------------------------------------
DESIGN RULES (every task)
----------------------------------------------------------------
- Null instances: 30-50% have nothing planted; the answer format includes "nothing
  found"; over-claiming is a scored failure.
- No fingerprinting: planted and null instances share distribution, formatting,
  tool-output shape, file sizes, metadata.
- Tools are generic primitives only. No tool, argument or task text names the fix,
  the answer layer, or the method. No layer ceilings.
- Answer key never in the agent-visible filesystem, tool outputs or errors; grader
  runs out-of-process; the leak scanner fails the run if any answer string appears
  in agent-visible output.
- Held-out grading data; the harness enforces and counts caps on forward passes,
  generations, gradients and tool calls.
- Difficulty knobs only from the precedented list in CONTEXT.md (cite in SPEC.md),
  plus null fraction. Label anything else experimental.
- Each knob/conjunct must require information from THIS model's internals, not
  bookkeeping.
- Drop instances the reference cannot solve.

----------------------------------------------------------------
BASELINES (build BEFORE any agent run)
----------------------------------------------------------------
- reference_solver: derives answers by probing under the same caps; gets NO instance
  metadata. Gate: >=95% on kept instances (best-of-5 allowed to prove solvability;
  also log one-shot rate).
- blackbox_control: same caps, white-box tools removed (outputs only). Gate: <=10%.
  (Run this as a scripted baseline where possible; use a contained subagent only if
  scripting it is impractical.)
- recipe_baseline: zero-effort guess (constant answer, majority class, submit
  everything, always-submit-something). Gate: <=10%.
If a gate fails, redesign and log which gate failed and why.

----------------------------------------------------------------
PREDICTIONS (before any agent run)
----------------------------------------------------------------
Write and git-commit tasks/<name>/PREDICTIONS.md: expected pass rate for the main
test agent and the smaller-model agent, control, reference, each with one sentence
of reasoning. Mark any number you cannot ground in a paper as a guess.

----------------------------------------------------------------
AGENT RUNS
----------------------------------------------------------------
1. Smoke pass: contained subagent, 3 instances per tier including nulls. Purpose is
   bug detection, NOT a pass-rate estimate. Save full transcripts and tool logs.
2. Auto-triage every transcript: agent-fault vs environment-fault; flag leaks,
   grader bugs, unsolvable instances, output/refusal oracles, and any containment
   breach (reads outside the sandbox invalidate the episode). Fix environment
   faults and re-run.
3. For tasks that pass all gates: 20 instances per dial setting x 4 settings with
   the main model and the smaller model, 3 repeats where cheap. Report Wilson 95%
   CI, per-instance pass-rate histogram (must not be bimodal), null-instance
   false-positive rate, and smaller-vs-larger-model separation.
4. Cluster transcripts by strategy (summarize, embed, cluster) to surface hack
   patterns.
5. Stay within the episode caps from AGENT UNDER TEST; prioritize wave 1 and the
   best wave-2 tasks; do not spend scale-up episodes on tasks that failed gates.

----------------------------------------------------------------
DELIVERABLES
----------------------------------------------------------------
Per task (one page): spec; knobs with citations; containment tier; gate results;
predictions vs outcomes; 3 example transcripts (success, agent-fault failure,
environment-fault failure); what I would change; keep/redesign/drop.
Then: (1) ideation/CANDIDATES.md; (2) a summary table across all tasks with gate
status and pass rates; (3) a 1-page "what generalizes" note: which task properties
predicted mid-band difficulty and which predicted hackability; (4) a ranked
recommendation of which ONE to present as validated and which 3-4 to pitch as
explored; (5) docs/INDEX.md up to date and docs/LOG.md complete; (6)
docs/SUMMARY_FOR_DAN.md in plain language: what was built, what Dan should
understand before the call, the 3 things that surprised you, what you would build
next, with every claim citing a run directory or a source.
Do not claim any task is mid-band from n=3.
================================================================
