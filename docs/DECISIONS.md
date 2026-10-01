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
