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

## 2026-10-01 15:45 UTC: independent harness verification (reviewer agent)
- Wrote adversarial tests `common/tests/test_verify_harness.py` (isolation, leak scan, caps, grader, fingerprinting,
  run-scripted/summarize, Wilson vs scipy) and `common/tests/test_verify_audit.py` (50 evasion transcripts, normal-work
  false-positive suites). The fixture task `common/tests/verify_task/tasks/_vcheck/` has tools that misbehave on purpose.
- Against the harness as built: 23 harness checks and 32 audit checks failed (29 of 50 out-of-sandbox evasions passed
  the audit). Main bugs: admin commands open to agents (status leaks codenames, shutdown kills the shared broker);
  another episode's id usable; write_array followed agent symlinks; 32 MB requests accepted; leak scan missed
  escaped strings, the canary tail, instance paths/ids/codename, and could be switched off by sending every candidate;
  a legitimate long echo was flagged at finish; a charge-then-hang call was free; a hanging grader blocked finish
  for an hour; malformed grader output (NaN, 7.0, "high") was accepted.
- All fixed (D13); details and remaining risks in docs/HARNESS_VERIFICATION.md. Full suite: see the commit message.
- The real API-agent episode (runs/_demo/20261001-152109_apiplumbing) is still VALID under the new audit.
- Action: the live broker must be restarted to load the fixes (done if no episode was open). Builders: `git merge main`.

## 2026-10-01 16:35 UTC: API ledger bug (per-worktree ledgers) found and fixed
- Symptom: BoolIntermediates and SilentPathway ran API probes, but the main ledger did not show them. Cause: the same
  bug class the harness builder hit with gpuq. api_agent anchored its ledger on "the checkout this file lives in",
  so every worktree had its own ledger, and the $16 global stop only saw that worktree's spend.
- Impact: none so far. Per-task caps were still enforced correctly, because each task only runs from its own worktree.
  The per-task caps sum to about $10.3 (< $16), so the global budget was bounded anyway. Real total at fix time: $0.708 over 43
  requests (boolintermediates $0.465, silentpathway $0.215, demo $0.028).
- Fix: the canonical ledger is now ~/.rl_api/ledger.jsonl, outside every checkout. The global and per-task totals read that
  file plus every legacy per-worktree ledger, deduplicated. Caps are read from the main repo. `spent --snapshot`
  writes runs/api_budget/ledger_snapshot.jsonl for git.

## 2026-10-01 16:50 UTC: GPU queue fairness (report from the EditFind builder)
- Report: a FeatureMatch gate run (pid 161609, ~6.7 GB) was using the GPU but was missing from the shared ledger, and
  EditFind's admitted jobs hit physical CUDA OOMs twice. Separately, FreqHunt's chained heavy jobs held both heavy
  slots for 30+ min while EditFind's heavy job waited.
- Ledger: the per-worktree ledger bug was already fixed on main (common/paths.py, 9e71e2d). Every worktree except
  editfind/triggerhunt has merged it. EditFind launches through the main checkout's module. At 16:45 every process
  on the GPU was in the shared ledger. The rogue process had exited, probably launched before its worktree merged main.
- Fairness: added to gpuq. A task (label prefix) that already holds a heavy slot cannot take a second while a heavy job
  from a different task is waiting. Waiters register in the ledger and are shown by `gpuq status`. Unit-simulated, and
  the harness tests pass (190). Caveat: worktrees run their own copy of gpuq.py, so the rule binds a task only after
  it merges main. Running builders are told via the builder guide.

## 2026-10-01 17:00 UTC: api_agent budget guard was far too pessimistic (environment fault in a probe)
- BoolIntermediates probe 3 (runs/boolintermediates/20261001-161311_apiprobe3, worktree) was stopped at $0.28 of its
  $0.60 per-episode cap before submitting, and scored as a fail. The guard priced the next turn as "whole context
  re-written to cache plus max_tokens of output" (about $0.49), while real Opus turns cost about $0.03. **That probe's
  failure is an environment fault, not an agent fault.**
- Fix: the estimate is now cache-read context + ~4k new tokens + 1.5x the largest output seen so far. When the
  episode nears its cap (or its turn limit), the agent gets a "[budget notice] submit now" text with its tool results
  and two more turns, before any hard stop. Verified on _demo (runs/_demo/*_apiwarn: warned, submitted, pass, $0.015).

## 2026-10-01 17:30 UTC: first Wave-1 probes (FeatureMatch, Sonnet), plus a transcript-audit false positive
- FeatureMatch probes (runs/featurematch/20261001-171048_apiprobe_sonnet in the featurematch worktree), Sonnet 5.5
  at medium effort:
  - T2, 5 slots: 4/5 correct. It answered "nothing found" on one planted slot, so it FAILED the all-slots rule. $0.152, 9 turns.
  - T3, 4 slots: 2/4 correct. It claimed the wrong sibling twice and used only 103 of 550 forward units. FAIL. $0.087, 7 turns.
  - Two earlier Opus probes were aborted at turn 3 on Dan's instruction (runs/.../20261001-165348_apiprobe_orch,
    marked ABORTED, excluded).
- The T2 episode was flagged INVALID by the transcript audit. A false positive: the audit read the sed expression
  `s/a=list.*/.../` as the path "/a=list". Fixed: sed s///, y/// expressions are removed before path scanning, and
  real file arguments to sed are still checked. 156 audit and verification tests pass. Re-audit: audit_rerun_sedfix.json (valid).
- Reading: neither failure is an environment fault. The near-miss slots (sibling confusion, a planted latent read
  as dead) are the difficulty the builder predicted. Sonnet costs about $0.09-0.15 per episode on this task.
