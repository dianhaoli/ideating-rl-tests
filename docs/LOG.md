# LOG: append-only lab notebook (orchestrator + cross-task entries)

Per-task notebooks: `tasks/<name>/NOTES.md` (each one is timestamped and append-only). Concurrent builders work on
separate branches, so their detailed entries live there to avoid merge conflicts. This file records cross-task
events, orchestration decisions and gate summaries, with links.

## 2026-10-01 05:12 UTC: session start
- Machine inspected. The plan says AWS g5 (A10G 24 GB), but the machine has **1x NVIDIA L4 (23 GB)**, 30 GB RAM,
  8 vCPU and about 83 GB of free disk on /. torch 2.13+cu130 is in /opt/pytorch (Python 3.13). I installed transformers 5.18,
  peft 0.21, accelerate 1.15 and datasets 5.0 there.
- HF token: Dan pasted it mid-session. Stored at ~/.hf_env and $HF_HOME/token with mode 600, outside the repo.
  Verified (whoami OK) without printing it. Access confirmed for gemma-2-2b, gemma-scope-2b-pt-res, Qwen2.5-0.5B/1.5B and
  gemma-3-270m.
- Repo github.com/dianhaoli/ideating-rl-tests was empty. No GitHub credentials on the machine, so **push is impossible
  for now**. Committing locally; logged in OPEN_QUESTIONS.
- EditHunt repo: **not present on this machine** (searched the filesystem). So wave-1 task 5 (T2-Families) is skipped as
  the plan specifies. A from-scratch hop-separation task goes into ideation as a candidate (see DECISIONS).
- Containment: I tried Tier A (a custom test-agent type whose hook forces every shell command to run as a
  low-privilege OS user). The Claude Code auto-mode classifier blocked it as self-modification of Claude Code
  config. Falling back to **Tier B: honor-system with auditing** (canaries, leak scanner, transcript audit). Dan can
  enable Tier A later (see OPEN_QUESTIONS).
- Started the background model/SAE download (common/download_models.py, log ~/hf_home/download.log).
- Wrote common/gpuq.py (GPU admission queue), docs/HARNESS_API.md (harness contract) and docs/BUILDER_GUIDE.md.

## 2026-10-01 14:53 UTC: session 2 (plan v2) starts
- The previous session stopped after the scaffold. Its workflow launch was halted by a safety classifier. The four task
  worktrees in ~/wt exist, but nothing has been built in them yet. Push works now (Dan set up a credential helper).
- Dan re-sent the plan as v2: a purpose statement, product rewards, behaviour validation, in-episode training caps,
  and the T2-RAVEL spec. Saved the differences in docs/PLAN_V2_DELTA.md and appended the new rules to BUILDER_GUIDE.
  Decisions D8 (T2-RAVEL from scratch as a Wave-2 candidate) and D9 (TriggerHunt deferred).
- The machine is still the L4. Plan: build the harness in parallel with the three Wave-1 builders and the Phase-0 ideation.

## 2026-10-01 15:19 UTC: harness implemented (common/), READY
- Implemented docs/HARNESS_API.md: `common/toolserver.py` (TaskEnv, @tool, ToolError, charge, write_array, per-episode
  server process admitted through the GPU queue, apply_caps), `common/broker.py` (Unix-socket broker, counters/caps incl.
  wall-clock and per-call timeout, built-ins help/budget/submit, privileged JSONL tool log, idle eviction, RAM cap,
  concurrent episodes), `common/leakscan.py`, `common/agent_client/tool` (stdlib, Python 3.9), `common/toolclient.py`,
  `common/sandbox.py` (setup/prepare/finish/run-scripted/summarize), `common/transcript_audit.py`, `common/paths.py`.
- Demo task `tasks/_demo/` (no GPU; hidden table with one edited entry or none) used by the tests and as a worked example.
- **Bug found and fixed**: `common/gpuq.py` kept its ledger per checkout, so every task worktree had its own GPU queue
  (D11). The ledger and the episode registry now live in the main checkout. Builders must `git merge main`.
- **Contract refinements** (D12, HARNESS_API section 10): free built-ins, wall-clock cap excluding queue time,
  agent-text exemption in the leak scan, ./py wrapper instead of symlink, stricter/more precise transcript audit.
- Tests: `/opt/pytorch/bin/python -m pytest -q common/tests` (unit + end-to-end on _demo + transcript-audit rules with
  false-positive checks + GPU smoke on Qwen2.5-0.5B through the queue with eviction). Results in the READY commit message.
- Not yet exercised: a real LLM subagent episode end to end (the builder of this harness could not spawn subagents).
  The audit was checked to parse a real Claude Code subagent transcript from this machine.

## 2026-10-01 ~16:10 UTC: harness READY, API runner verified, Phase 0 done, Wave 2 launching
- The harness (common/) is READY. Independent verification is still running in the build workflow.
- API runner verified end to end on _demo: Haiku, valid audit, pass, $0.028 (runs/_demo/20261001-152109_apiplumbing).
  Per-task API caps now in runs/api_budget/task_caps.json (Wave-1 $1.00 each, Wave-2 $1.20 each). Global hard stop $16.
- Phase 0 ideation (ideation/CANDIDATES.md): 41 candidates scored by two independent harsh scorers. Wave 2 = C02
  BoolIntermediates (a), C03 SilentPathway (a), C20 LatentDiff (b), C13 ResidualRecall (b), C27 ShiftHunt (c),
  C37 T2-RAVEL from scratch (c). T2-RAVEL scored only 28/45 (weak nulls, mid-band doubts). It is built anyway because Dan
  asked for it explicitly. The builder must make the null design honest or report that it cannot.
