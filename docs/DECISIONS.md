# DECISIONS: choices Dan might want to reverse

Format: decision | alternatives | why | what the original plan said.

## D1. GPU queue = memory ledger, not "2 jobs max"
- **Chosen**: `common/gpuq.py` admits jobs while the sum of declared GB is at most 21 GB. At most 2 *heavy* jobs
  (training, sweeps) and at most 7 jobs in total run at once. Each job caps itself with torch's per-process memory
  fraction. RAM cap is 8 GB per job.
- **Alternatives**: two fixed 10 GB slots, as in the plan.
- **Why**: test-agent episodes each hold a small model in a tool server for 10-30 minutes, mostly idle while the
  agent thinks. Two slots would serialize episodes. Heavy compute keeps the plan's limit of 2.
- **Plan said**: at most 2 GPU jobs at once, about 10 GB GPU and 6 GB RAM each, on an A10G. The machine actually has an L4.

## D2. Containment: Tier B (honor-system with auditing)
- **Chosen**: Tier B. The test agent works in `~/rlsbx/<episode>/` through a tool client and a privileged broker.
  We add canaries, a leak scanner on every tool response, and a transcript audit that invalidates any episode
  touching paths outside the sandbox, the network, sudo, or the privileged Python env.
- **Alternatives**: Tier A, a custom subagent type with a PreToolUse hook that rewrites every Bash command to run as
  a low-privilege OS user. Or headless `claude -p` runs as another user (needs credential copying).
- **Why**: the Claude Code auto-mode classifier blocked creating the hooked agent definition (self-modification).
  Dan explicitly said Tier B is acceptable. Every SPEC.md states "honor-system with auditing (Tier B)".
- **Plan said**: Tier A preferred, Tier B fallback.

## D3. Per-task branches in git worktrees
- **Chosen**: each builder works in `~/wt/<task>` on branch `task/<task>`. The orchestrator merges to main.
- **Why**: about 10 builders work concurrently. One shared checkout cannot have several branches checked out.
- **Plan said**: feature branches per task, merged when gates are logged (unchanged in spirit).

## D4. Per-task lab notebooks
- **Chosen**: detailed, timestamped, append-only notes in `tasks/<task>/NOTES.md`. `docs/LOG.md` holds cross-task
  and orchestrator entries and links to the task notes.
- **Why**: concurrent branches all appending to one docs/LOG.md would conflict on every merge.

## D5. T2-Families skipped in wave 1
- The EditHunt repo is not on this machine. The plan says to build T2-Families only if it is present. A from-scratch
  hop-separation task was put into ideation scoring as a candidate instead.

## D6. Instances and answer keys are gitignored until used
- The repo is public, and test agents have network access in Tier B. Instance answer keys stay out of git until the
  episodes that use them are done. Generators, seeds and manifests are committed. Answer keys are reproducible from seeds.

## D7. Multi-slot episodes reconcile the null rule with the recipe gate
- **Problem**: with 30-50% null instances, the constant answer "nothing found" passes 30-50% of single-question
  episodes. The recipe gate (≤ 10%) could then never pass. The plan's two rules conflict.
- **Chosen**: each episode holds several slots. 30-50% of slots are null, and pass = all slots correct, so constant
  answers pass rarely. The continuous score is the mean slot score. The slot-level null FP rate is reported.
- **Alternative**: count nulls separately and gate the recipe only on planted instances. Rejected, because a
  GRPO policy would still collect reward from a constant answer.

## D8. T2-RAVEL is built from scratch as a Wave-2 candidate (2026-10-01, plan v2)
- **Chosen**: the EditHunt repo is still absent, so T2-RAVEL (Dan's plan-v2 spec, docs/PLAN_V2_DELTA.md) is not
  a Wave-1 port. It enters Phase-0 scoring as a from-scratch candidate on Qwen2.5-1.5B with a new US-city geography
  set validated per entity. Phase-0 selection decides whether it is built.
- **Why**: Dan made it conditional on EditHunt being present. But his detailed v2 spec says he values it, and the
  geography setup is cheap to rebuild. Building it from scratch keeps to the spirit without pretending to reuse
  EditHunt's infrastructure.

## D9. TriggerHunt deferred to a separate later pass (plan v2 allows this)
- **Chosen**: Wave 1 builds FreqHunt, EditFind and FeatureMatch first. TriggerHunt (a harmless marker-string LoRA on
  Qwen2.5-0.5B) runs as a separate pass once the harness and these three are through their gates.
- **Why**: the plan marks it optional. It also needs a LoRA bank trained up front (GPU-heavy), which would compete with
  the other builders for the single L4.

## D10. Claude API test agent with a hard $17 total budget (Dan, 2026-10-01 ~15:10 UTC)
- **Change**: Dan added an Anthropic key and allowed API use with a TOTAL budget of $17. It is mainly for testing task
  difficulty early, in fast feedback loops, before a lot gets built. This supersedes the plan's "do not call the
  Anthropic API".
- **Mechanism**: `common/api_agent.py` runs a minimal agent loop with one tool, `bash`, inside the episode sandbox. The
  environment is scrubbed (HOME = sandbox, no tokens or keys). Commands that reference anything outside the sandbox are
  refused before they run. Every request's list-price cost is appended to `runs/api_budget/ledger.jsonl` (committed).
  A global hard stop at $16.00 leaves $1 slack, and each request is checked against a pessimistic estimate first.
  There are also per-episode caps (--max-usd, --max-turns).
- **Allocation plan** (soft): Wave-1 early probes about $4.5 (3 tasks); Wave-2 early probes about $6 (6 tasks, about $1
  each); measurement on the candidate validated task, including Haiku separation, about $4; reserve about $1.5.
- **Models**: main = `claude-opus-5-5` (effort medium); small = `claude-haiku-4-5`. API runs carry the label
  "API (model X, effort Y)" and are kept separate from "fresh Claude Code subagent" runs. Free subagent runs stay
  the high-volume channel (smoke, bug-finding, scale-up). API runs are the fast, exact-model difficulty probe.
- **No server-side model fallbacks**: refusals are logged and the episode is marked invalid. A silent fallback to
  another model would contaminate the model label on a difficulty measurement.
- Key storage: Dan's key sits in the gitignored repo `.env` as `ANT_KEY`, quoted. I chmod-ed that file to 600 because
  it was 644 and also holds the HF token. The runner reads an unquoted copy at `~/.anthropic_env` (mode 600, outside
  the repo).

## D11. One machine-wide GPU-queue ledger and episode registry, anchored on the main checkout (2026-10-01, harness)
- **Problem found**: `common/gpuq.py` put its ledger at `<this checkout>/runs/.gpuq`. Every task worktree in `~/wt/`
  therefore had its *own* ledger, so jobs from different builders never saw each other and the 21 GB limit was not
  enforced machine-wide.
- **Chosen**: `common/paths.py` finds the main checkout from git's common directory (a worktree's `.git` file points
  into `<main>/.git/worktrees/<name>`). The ledger (`runs/.gpuq/`) and the harness episode registry (`runs/.episodes/`)
  always live in the main checkout. Overridable with `GPUQ_DIR` / `RL_EPISODES_DIR` (tests use this).
- **Action for builders**: `git merge main` as soon as possible so your jobs join the shared ledger.

## D12. Harness implementation choices that refine HARNESS_API.md (2026-10-01 15:19 UTC)
Full list in docs/HARNESS_API.md section 10. The ones Dan might want to reverse:
- **Built-ins are free** (help, budget, submit do not cost tool_calls). Alternative: charge everything. Why: an agent
  checking its budget should not lose budget; the spec's "every call costs one unit" is kept for task tools.
- **Wall-clock cap excludes compute waiting.** Each episode has `wall_clock_s` (default 3600 s) counted from its
  first tool call minus time spent in the GPU queue or loading the model. Why: with ~5 agents sharing one L4, queue
  time is noise the agent cannot control. After the cap, only help/budget/submit work, so the agent can still answer.
  Per-call timeout `call_timeout_s` (default 180 s) kills runaway calls (plan v2's in-episode training cap together
  with the `gradient` counter).
- **Agent-supplied text is exempt from the leak scan for that call.** If the agent sends a leak string (its own
  guess) and a tool echoes it back, that is not a leak. Why: otherwise the episodes where the agent guessed right would
  be invalidated by any tool that echoes its prompt. The canary is never exempt.
- **Leak strings in the agent's own sandbox files are counted, not flagged.** An agent that found the answer writes it in
  its notes; flagging that would invalidate successes.
- **`./py` is an exec wrapper, not a symlink** (Python 3.9 resolves its venv from the invoked path). Same effect.
- **Transcript audit is slightly stricter than the spec**: it also flags mcp__* connector tools and commands run from a cwd
  outside the sandbox (R8). It is slightly more precise on `~`/`$HOME`/`..`: they are expanded/resolved and then
  checked, so `cd ~/rlsbx/<E>` is not a violation. The agent prompt tells the agent to start every command with
  `cd <sandbox> &&` and not to use `..`, `~` or /tmp, so these rules are stated up front, not traps.
- **validate_submission runs in a fresh process without load()** so format checks never need the GPU.
- **Tool servers run offline** (`HF_HUB_OFFLINE=1`): no downloads mid-episode.

## D13. Harness hardening after independent review (2026-10-01 15:45 UTC)
Full list with before/after in docs/HARNESS_VERIFICATION.md. The ones Dan might want to reverse:
- **Calls are bound to the caller's sandbox.** The broker reads the calling process's working directory. A call
  for episode B from inside sandbox A is refused and A is marked INVALID (`cross_episode_access`).
  Alternative: no binding, and rely on the audit alone. Why: sibling episode ids are visible directory names.
- **Admin commands need a token file** (`runs/.episodes/.admin_token`, mode 600). Alternative: an open socket.
  Why: before this, any agent could list all episodes and their task codenames, or shut the shared broker down.
- **The leak-scan agent-text exemption is count-based** (D12 exempted any leak string the agent had sent).
  Why: an agent could send every candidate answer and switch the scan off for that call.
- **New never-exempt `private_strings`**: the instance dir, repo paths, instance id and task codename.
  These are leaks anywhere in a response or in sandbox files.
- **A grader with malformed output is a grader error.** Score not a finite number in [0,1], pass not a bool,
  output not an object, no output, or a timeout (`RL_GRADER_TIMEOUT_S`) all count. The episode becomes
  INVALID with score 0. Alternative: clamp the score. Why: a silently clamped NaN or 7.0 would hide a grader bug.
- **The audit is stricter in a few places and more precise in others.** It now flags `ln`, environment-variable
  and home-directory access from Python, `cd -` or a bare `cd`, and decode-and-run (R9). It tracks `cd` inside
  a command, so `cd scratch && ls ..` is valid. The agent prompt states the new rules.
- **A symlink in the sandbox pointing outside makes the episode INVALID** at finish.
