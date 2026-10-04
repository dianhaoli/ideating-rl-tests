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

## 2026-10-01 17:45 UTC: FreqHunt probes found two environment faults; both fixed in the harness
- FreqHunt Sonnet probes (freqhunt worktree, runs/freqhunt/20261001-172719_apiprobe_sonnet_orch):
  - ep440f6b9bf4: INVALID, environment fault. The builder regenerated instance fh-t1-0fe4eff6 (same id, files rewritten
    at 17:28) while the episode ran. The submission failed validation as "malformed" and the grader crashed. **Fix:**
    `sandbox prepare` now snapshots the instance into runs/.episodes/snap/<rand>/<id> (a copy if <= 300 MB, else
    hard links). The tool server and grader use the snapshot, and the record keeps `instance_dir_source`.
  - The same episode exposed an api_agent bug. The agent chained a rejected `./tool submit` with `./tool help`, and the
    runner saw '"ok": true' in the output and stopped the episode as "submitted". **Fix:** submission is now read from
    the broker's privileged episode record.
  - epf7e00255e5: valid, 2/3 slots, FAIL. $0.053, 8 turns. (FreqHunt T1 is cheap for Sonnet: about $0.04-0.05 per episode.)
- Verified on _demo (runs/_demo/*_apisnapshot): snapshot path recorded, episode valid. 190 harness tests pass.

## 2026-10-01 18:10 UTC: harness fixes from builder reports (GPU starvation, codename false positive), plus a coordination slip
- **GPU starvation** (reported by the featurematch integrator and the latentdiff builder): 6.5-7 GB tool servers waited
  30+ min while 2-5 GB jobs took every gap. Added aging to gpuq: a waiter older than AGE_S (300 s), and older than the
  requester, gets its memory and job slot reserved. Aged heavy waiters reserve only while a heavy slot is free.
  Simulated the reported scenario (big waiter is admitted once memory drains, small requests are held back). 190 tests pass.
- **Codename false positive** (featurematch integrator): the leak scan matched private strings with whitespace removed,
  so an agent's note "this feature matches Spanish" equalled the codename "featurematch" and would invalidate the
  episode. Word-like private strings (codename variants, instance id) are now matched with whitespace kept. Paths keep
  the whitespace-insensitive match. Verified: prose is clean, "FeatureMatch" and an embedded codename are still flagged.
- **Coordination slip (mine):** at 16:52 I replied by SendMessage to the editfind science agent, which was a workflow
  subagent. The send *resumed* it as a separate agent, so two builders worked in ~/wt/editfind from ~17:35 (that agent
  and the workflow's integrate:editfind). The resumed agent noticed, stopped editing, cancelled its duplicate gate job
  and handed off (stage-1 design summary preserved in this log's source thread). Rule from now on: do not SendMessage
  workflow agents; relay information through files (BUILDER_GUIDE / LOG).
- EditFind status from that handoff: the recipe gate passes (constant "nothing found" 1/36 T2, others 0/72). The
  reference gate is FAILING so far (prelim v2 3/11). It finds the edited subjects every time (rank 1 of 3000) but
  mis-picks the relation or new answer, or a decoy is not silent on unseen wordings. The integrator owns the fix.

## 2026-10-01 18:45 UTC: EditFind status (relayed) and the duplicate-agent question
- The editfind agent (a7c9e3b, which my 16:52 SendMessage had resumed) reported its grading redesign. Accepted answers
  are recorded per edit, related-name claims are credited, and reference v4 plus the blackbox gates run unattended until about 19:20+.
  Full relay: ~/wt/editfind/tasks/editfind/ORCH_RELAY.md (uncommitted file in that worktree).
- I did NOT stop the resumed copy. The Wave-1 workflow journal shows science:editfind has not returned yet, and I
  could not rule out that the resumed copy and the workflow's agent are the same agent. Stopping it might have killed
  the workflow stage. It is idle except for its own background gate jobs.
- Risk flagged for the integrator and auditor: the grader is now looser and some accepted answers are junk, which can
  raise recipe/black-box pass rates. Those gates must be re-run after augmentation.

## 2026-10-01 19:20 UTC: EditFind decision: rebuild the edit bank for specificity (last attempt before DROP)
- Reference v4 on the final grader: T1 8/16 (Wilson 0.28-0.72), T2 9/16 (0.33-0.77). Null-slot FP 0.00. Recipes 0/36 except
  "nothing found" T2 1/36 (runs/editfind/20261001-190458_prelim_reference_v4, ..._190457_prelim_recipes_cpu, editfind worktree).
- Root cause (agent's analysis): the planted ground truth is under-specified. A ROME edit also moves same-named
  neighbours (iPhone XS along with Lexus NX), its side effects depend on wording, and some edits barely move on new wordings.
  A correct auditor's answer is therefore graded wrong about half the time. That is a task bug, not difficulty.
- Decision (mine): rebuild the bank, keeping only *specific* edits (key-scan neighbours move <= 1 nat; only the edited
  relation moves, to the target, on held-out wordings). Log per-relation survival and check the filter does not leave
  only "easy" edits. Enlarge the pool >= 2x (memorisation risk). Time-box 1.5 h. If the reference is still < 95% after
  this, editfind is DROP / "explored, not validated" (second gate failure for the same root cause).

## 2026-10-01 19:55 UTC: EditFind DROP (explored, not validated)
- The specificity rebuild kept **0/154** ROME edits, in all 10 relations (runs/editfind/20261001-193758_v5_bank_specificity,
  editfind worktree, commit b6844c9). The blocking criterion is "only the edited relation moves". A layer-4 ROME edit moves
  the subject's OTHER relations by a median of 7.4 nats (deciles 2.6-11.8). Neighbours-only keeps 71/154 and held-out-only 110/154.
  Neighbours + held-out keeps 54/154, but that would mean grading every changed answer, i.e. a different task.
- This is the second gate failure from the same root cause (an under-specified planted ground truth), so it is DROPPED per the plan.
- What worked and is worth keeping: localisation. The corpus key-scan finds edited subjects (rank 1/3000), decoys are
  rejected, null copies get no false claims, and recipes stay <= 1/36.
- Finding worth telling Dan: "find the planted edit" tasks built on ROME inherit ROME's lack of specificity. The edit
  is a change to the whole subject, not a single fact, so a pair-level answer key is ill-posed. Revival would need a
  more specific editing method (e.g. MEMIT across several layers with a strong locality term, or fine-tuned single-fact
  LoRAs validated for specificity). This is a transfer-relevant lesson for auditing benchmarks.
- Coordination: copies of the resumed agent committed concurrently (16b627d, 6bcf82b, 84149f1). One used a lock file
  to prevent a duplicate rebuild. The SendMessage-resumes-workflow-agent problem cost real confusion; see 18:10 entry.

## 2026-10-01 20:05 UTC: FeatureMatch OpenAI probes, and GPU congestion now blocks agent measurement
- 20261001-182038_openai_probe_orch (featurematch worktree): gpt-6-luna 0/3 and 3/4 slots (both FAIL; small-model signal is consistent).
  Both gpt-6.1-sol episodes were starved by the GPU queue (5 tool calls in 90 min) and are excluded as an environment fault.
- Lesson: with ~9 builders running gates, agent episodes on GPU-heavy tasks (7 GB tool servers) cannot be measured
  reliably. Agent-measurement batches should run when builder load drops (after the build workflows finish), or in
  dedicated windows. An overcommit change to gpuq that would have helped was denied by the permission classifier
  (it touches a shared resource), so I left it alone and noted it for Dan.

## 2026-10-02 00:50 UTC: FeatureMatch forensics (docs/forensics/FEATUREMATCH_PROBE_FORENSICS.md)
- 12-agent workflow: per-episode slot forensics + independent skeptics + pool-wide answer-key check + synthesis.
- Keys and grader are SOUND. In all 16 slots of the 4 valid episodes the key is the best menu concept on split C (never
  used by the generator). Pool-wide key flips on C: 1/157 (v1), 1/431 (v2). Re-grading reproduced every slot.
- All 7 misses are real agent errors. Each submitted with >= 37% of its compute unused, and 3 contradicted the agent's
  own control probes. A task-side factor also contributed to each: style-dependent latents (MMA x2), a misleading label (skater),
  and v1-only distractors / word-trap latents.
- Sonnet: 5/8 planted slots right (Wilson 0.31-0.86), all nulls right. That sits between the fixed recipes (14-15%) and the
  reference (96-97%). gpt-6-luna: 0/3 planted; its "3/4" was all null slots (scores 0 under the v2 product reward).
- All agent data is on the RETIRED v1 pool. No agent has run on v2 yet.
- Main open risk: style dependence. The 16:42 style check found only 50% of eligible latents pick out their own concept on
  hand-written probes. Action before measurement: add a style-robustness filter to the generator (keep latents that
  recover their concept on independently written probe texts).
- Also: docs/PLAN_PROMPT.md had been truncated to 0 bytes (uncommitted, cause unknown, probably a stray write by an
  agent). Restored from git.

## 2026-10-02 01:25 UTC: FeatureMatch diagnosis study started (Dan's brief; D14)
- Plan: tasks/featurematch/diagnosis/PLAN.md on branch diag/featurematch (worktree ~/wt/fmdiag).
- Phase A workflow (wf_5412322c-af1) has started. It covers: PREREG.md (step 0), the tool/task clarity audit (step 1), and
  infrastructure: harness prepare options (--prompt-template, --min-submit-frac, --extra-file) plus a shared FeatureMatch
  model service so the study stays at <= 2 GPU jobs. It also covers two independently written style banks (F = filter,
  R = style-robust recipe), 232 concepts x 20 texts each, with 8 writer agents per bank.
- Nothing is measured until PREREG.md is committed. All agent experiments will use fresh Claude Code subagents only
  (no hosted API, per Dan's brief).

## 2026-10-02 01:55 UTC: new task LatentKnockout (Dan's request; FeatureMatch "tests the wrong things")
- Dan's diagnosis of FeatureMatch: latents chosen on Wikipedia-style text stay silent on agent-written text, so the score
  partly measures style imitation. A new task should make the agent USE an SAE to explain a behaviour, graded by
  RERUNNING the model: find <= k SAE latents whose ablation removes a behaviour on held-out target prompts while
  preserving held-out controls. Reward ~ Effect x Preserve x KL factor.
- Worktree ~/wt/latentknockout, branch task/latentknockout. Design brief (request + my starting hypotheses):
  tasks/latentknockout/DESIGN_BRIEF.md.
- Feasibility first (workflow wf_7ed4195f-743): behaviour validation, attribution + greedy reference, many
  (family, group, layer) cells, baselines (random, most-active, mean-diff-cosine, plain non-SAE steering vector, naive
  top-k attribution). Then an independent skeptic re-implements the key numbers and hunts cheap solvers. I review
  before anything is built.
- The FeatureMatch diagnosis phase A (wf_5412322c-af1) keeps running: PREREG, clarity, infra and style banks, mostly CPU.
  Its GPU-heavy agent arms are on HOLD pending Dan's call, since he now regards FeatureMatch as testing the wrong things.
- 02:00 UTC: Dan set the order: finish the FeatureMatch investigation first, then LatentKnockout. The LatentKnockout
  feasibility workflow was stopped after a few minutes (nothing committed beyond the design brief). It resumes after
  the FeatureMatch verdict. FeatureMatch agent arms are back ON.

## 2026-10-02 02:45 UTC: disk full (99%), fixed
- Reported by the latentdiff builder: graders crashed with ENOSPC (episodes INVALID with grader_error), and in-process
  jobs failed. Cause: ~/rlsbx held 33 GB across ~9,500 finished sandboxes, mostly tool-output .npy arrays. (/tmp had
  already recovered: 26%.)
- Fix 1 (one-off): deleted files > 100 KB in out/ and scratch/ of FINISHED episodes only (record finished/closed):
  23,881 files, 32.2 GB freed. The 26 active episodes were untouched. Disk now 68% (33 GB free). Log:
  runs/.episodes/prune_20261002.log.
- Fix 2 (permanent): `sandbox finish` now prunes files > 100 KB from out/ and scratch/ after leak-scan and grading,
  and lists them in the episode's sandbox_pruned.json. RL_KEEP_SANDBOX=1 disables this. Harness tests pass.
- Consequence: any episode graded during the full-disk window (about 02:00-02:45) may be INVALID with grader_error.
  Builders should re-run those (check grade.json harness.invalid_reasons).

## 2026-10-02 03:50 UTC: FeatureMatch diagnosis step 2 computed, but the filter does not match PREREG A1.1 (fixing)
- Phase A finished: PREREG + Amendment 1, clarity audit + revised/disclosure templates, harness prepare options,
  shared model service (0 mismatches), banks F and R (232 x 20 each), and the bank activations
  (diag/featurematch 827191a). GPU is idle now.
- Step 2 analysis (CF2 rule) kept 43% of planted slots and 25% of pooled latents. But the code departs from the
  binding Amendment A1.1: (1) the key pools A+F1 (40 dataset vs 10 styled texts) instead of M = 0.5 AUROC_A +
  0.5 AUROC_F1; (2) robustness is thresholded on pooled C+F2 (a latent silent on ALL styled text still gets 0.83,
  and one styled hit reaches 0.85) instead of AUROC_F2 >= 0.85 AND AUROC_C >= 0.85; (3) the pool is redrawn
  in place instead of regenerated (60/tier, seeds 8000+/108000+/208000+); (4) key_check.jsonl and
  slot_disagreements.jsonl are missing. Under the stricter F2 metric only 8.7% of pooled latents survive (P3 predicted 0.45).
- No baseline or agent has used the pool, so making the code follow the pre-registered rule is a conformance fix, not a
  deviation. One builder agent fixes step 2 (CPU only, cached activations). A second agent implements the step-3 style-robust
  recipe (SR-max / SR-thr from cached bank-R activations) in separate files.

## 2026-10-02 03:55 UTC: OOM crash at 03:10, memory guards added
- Kernel log, previous boot: a python process (most likely the step-2 style_filter analyze) reached 19.5 GB RSS and
  was OOM-killed at 03:10. The machine went down at ~03:22 and rebooted at 03:39. That ended the previous orchestrator session and its
  workflows (committed outputs are intact).
- Guards: 16 GB swap (/swapfile, in fstab); user-1000.slice MemoryHigh=24G, MemoryMax=26G (persistent), so a
  runaway job is killed inside the slice instead of taking the system down; per-job caps via systemd-run scopes;
  rules in BUILDER_GUIDE "Memory". Disk is now 86% used (swap file takes 16 GB).

## 2026-10-02 04:00 UTC: interrupted workflow stages relaunched (new session, after the crash)
- The crash killed the previous orchestrator session (e4652089). Finished before the crash: ideation, wave-1 harness/science/
  integrate/audit, wave-2 build/audit (6 tasks), FeatureMatch forensics, diagnosis phase A and step-1b prep.
  Interrupted mid-stage at ~03:10: fix:featurematch (wave 1; gates already committed at 3f3252c, waiting on the
  harness retry stream), fix:shifthunt and fix:latentdiff (wave 2; gate/API-probe runs in flight).
- A cross-session resume (journals copied into the new session) missed the cache after the first stage and began
  re-running finished audits and science stages. I stopped it within about a minute. No files were changed (checked by mtime).
- Relaunched as continuation workflow wf_a83c544f-8d1: the three fix agents only, each with its ORIGINAL fix prompt
  (same report and audit text from the old journals) plus a crash-recovery note (predecessor's last actions, treat
  ~03:10 run dirs as interrupted, memory rules, do not touch featurematch instances/cache).
- The FeatureMatch diagnosis continues with two background builders: the step-2 filter conformance fix (PREREG A1.1) and the
  step-3 style-robust recipe.

## 2026-10-02 04:10 UTC: wave-2 fix-stage reports never received before the crash (recovered from the old journal)
The wave-2 workflow did not return before the crash, so these reports were never reviewed. Recovered from journal
wf_352c134a-502 (old session e4652089).
- **BoolIntermediates: READY_FOR_SMOKE.** v2 pools of 40 instances/tier (T1-T4). Reference 40/40 in every tier [0.91, 1.0];
  blackbox 0/40; recipes (none, claim_all, prior, binarise_*) <= 1/40, i.e. the all-null rate; auditor attacks
  twolevel/behav/unitdecode_read/count 0/40. Fingerprint AUROC <= 0.53. **Disputed F2:** `unitdecode` (read each unit's
  input variables, then a causal clamp) solves 38-40/40 in every tier. The builder calls it a correct interp method (a
  second reference) and withdrew the T4 "resists brute force" claim. Risk: the task may be too easy for frontier agents.
  Measure per-instance bimodality in smoke. Gate-pool answers are effectively in git, so the smoke plan uses fresh
  instances. Incident: at ~20:13 UTC yesterday the builder ran an unscoped `kill` on reference_solver processes, which
  may have killed another task's reference episode.
- **ResidualRecall: NEEDS_WORK.** Design v2 (low-rank unlearning edit at a different depth per model-tier, hardened).
  T1: reference 6/6 in-process and in the harness; every shortcut 0/14. T2: reference 1/8 (cannot fully remove a rank-6 edit from
  6-8 people). This is a new cause, not the v1 layer prior, so it is not a second same-reason failure. Proposed fix: rebuild T2
  stage B at rank 3-4 (~15 min heavy GPU per model x 3), then more bank models (only 3 models, 14 instances).
  Harness chains B/C/D were interrupted, and their run dirs (runs/residualrecall/20261002-011802_*_v2) are uncommitted.
- **T2Ravel: DROP.** The reference is now reliable (16/16 one-shot, 129/129 label agreement), but with enough cities an honest null
  exists only below the state's separability onset (layers 3-5), so the public layer predicts null with 100%
  accuracy. That is C1 again after a redesign aimed at it. Salvage ideas are in its NOTES.
- **SilentPathway: DROP** (build stage). Scripted gates pass (reference 12/12, recipes 0-1/14), but frontier probes solve
  it in 10-13 turns: once switched on, a planted silent group is easy to read off. Possible reuse: a calibration or
  small-agent task.
- Harness issues reported by builders (open): transcript-audit false positives (sed regex slashes read as paths;
  exec of the agent's own sandbox file; the R4 netcat regex matching a Python variable `nc`); GPU queue starvation
  of light jobs.
- **Not started (priority is Dan's call):** the ResidualRecall T2 rebuild and a BoolIntermediates agent smoke. Dan's
  order is FeatureMatch diagnosis first, then LatentKnockout. Added to OPEN_QUESTIONS.

## 2026-10-02 04:50 UTC: FeatureMatch diagnosis step 2 under PREREG A1.1 (3ea47d7f); step 3 launched
- Pre-registered keep rule (A1.1, two separate thresholds): **354 of 5017 pooled latents kept (7.1%)**. Languages 0.59, **topics
  0.04**. The F2 threshold (styled text) drives this: 8.2% of latents pass it, against 95.4% for the dataset split C. Dropped
  latents fire on ~15% of their concept's styled texts, against 95% for kept ones. 82.9% of the current v2 pool's slots fail
  (planted and null alike). "Filter selects easier latents": no (none of the 3 pre-registered thresholds is met).
  Predictions P3/P4/P31-P34 missed by a wide margin (we expected ~40% survival).
- Regenerated pool (180 instances, seeds 8000+/108000+/208000+): only 74 anchor concepts (v2: 159); language anchors
  27% of slots (v2: 9%). **P6 fingerprint fails narrowly: 0.616 in the close tiers** (bar 0.60).
- Style-robust recipe built (sr_recipe.py, a8d47fe8). Unfiltered-v2 dry run: SR-max planted accuracy 0.52, which depends
  on which 6 of 20 texts it uses (0.47-0.58).
- PREREG amendments, all before any filtered-pool result: A2 (SR text selection, sensitivity label, T3 budget, clustered CI),
  A3 (run step 3 despite P6, because the fingerprint cannot help scripted recipes; fix P6 before any agent arm; report by
  family; independent verification of step 2).
- Workflow wf_95ca39e9-284: 4 verifiers (independent re-implementation of A1.1; blind bank-text validity audit; what
  dropped latents fire on and whether style-robust latents exist outside the pool; fingerprint mechanism and
  label-only exploitability), the step-3 baselines (in-process, one GPU job), a skeptic (re-grades and hunts new cheap
  recipes), and STEP3.md.

## 2026-10-02 (~07:00 UTC): FeatureMatch diagnosis VERDICT: stop rule fired, "Not interpretability"; LatentKnockout resumed
Workflow wf_95ca39e9-284 (7 agents). STEP3.md (fmdiag 22111857); skeptic 0fa8c454; re-implementation 3eaccce1.
- **Stop rule fires, not borderline, not text-selection-sensitive.** On the filtered pool (180 episodes, 441 planted / 284
  null slots), the style-robust recipe SR-max (6 styled texts per option, pick the max, no reasoning) gets planted accuracy
  385/441 = **0.873** [0.839, 0.901]. Topic slots alone: 0.829 [0.785, 0.866]. SR-thr 0.730. Reference one-shot 441/441 = 1.000.
  The pre-registered sentence applies: **"FeatureMatch is mostly measuring style, not interpretability"** (pa(SR-max)/pa(ref) = 0.873).
  Step 4 (agent arms) does not run.
- Robust to the method: the skeptic re-graded all 3,603 episodes (0 mismatches) and re-implemented SR (725/725 identical
  decisions). Run through the live tool path, SR gets 0.875. Text selection spans 0.862-0.955. A script that uses only the task's own
  `generate` tool within budget gets 0.571. Gates also fail on this pool: self_probe 0.204, template_probe 0.288 (bar 0.15).
- Step 2 verified: an independent re-implementation matches 5017/5017 latents. Blind bank audit: 800/800 texts on-concept.
  Mechanism: dropped topic latents mostly fire on encyclopedia formatting cues (18/30 vs 0/15 kept). Styled-text fire rate is 25% vs 82%.
  **Every topic concept (201/201) has style-robust latents in the SAE, but the generator's name filter (an
  anti-shortcut rule) was involved in excluding 84% of them.** That rule selected format-cue latents.
- Fingerprint: 0.616 (P6 protocol), 0.683 on a larger held-out test. A memorising label-only policy passes 16.4% of
  filtered episodes (planted accuracy 0.58; v2: 0.33). Evaluating on held-out latents removes it (0.057).
- Predictions: many misses (we expected ~40% filter survival and got 7%; P11 predicted SR 0.75 and got 0.87).
- Next: write-up workflow wf_7144a333-59e (RESULTS.md + one-page VERDICT.md, numbers audit, overclaim critic, revision).
  **LatentKnockout feasibility relaunched** (wf_aec58fc3-341, D15 order): predictions committed first, feasibility (GPU),
  precedent survey, two skeptics (reproduction, shortcut hunt), synthesis with GO/ADJUST/NO-GO. The FeatureMatch lessons
  are built into the prompts.

## 2026-10-02 (~07:40 UTC): FeatureMatch deliverables done; corrections to the 07:00 entry
- RESULTS.md (12k words) and the one-page VERDICT.md are on diag/featurematch (5dc9fb88, revised 24c39b9b) after a numbers audit
  (~600 numbers checked; 22 discrepancies, 2 material) and an overclaim review (18 findings). All were fixed and none rejected.
- **Corrections to my 07:00 entry:**
  - The "format cue" mechanism (18/30 vs 0/15) comes from one unblinded LLM reader, and a scripted check does not reproduce the split. Treat it as
    suggestive. Better-supported numbers: dropped low-firing latents fire on 0.40 of encyclopedia-style bank texts vs 0.10 of
    on-concept texts that name an instance.
  - Name filter: 188/201 topic concepts have a style-robust latent anchored on them, and 57/201 have one kept. The name filter was
    *involved* in 84% of the exclusions but was the *only* reason in 35%.
  - The memorisation numbers come from a simulation with full answer labels (nearest-neighbour learner), not from RL training.
- **New, strengthens the verdict:** even the ORIGINAL unfiltered v2 pool fails the recipe gate. A fixed script that writes one probe
  per option with the task's own `generate` tool, within budget, gets 0.441 [0.395, 0.488] planted accuracy (gate 0.15).
  The style filter made this worse (0.571) but did not create it.
- FeatureMatch status: **explored, not validated** (diagnosis complete, verdict "Not interpretability"). The PLAN's 3
  example transcripts do not exist on v2/v2f because no agent ran, so RESULTS points to the v1 forensics episodes, labelled v1.

## 2026-10-02 (~08:00 UTC): crash-interrupted fix stages finished (wf_a83c544f-8d1); harness audit hardening started
- **ShiftHunt: READY_FOR_SMOKE** (task/shifthunt b17f82b0). Design v3. Reference v3.1 (paired-edit confirmation): T1 27/27
  one-shot [87.5, 100]; T2 solved 10/29, so 10 kept, 10/10 one-shot. Every recipe, the black-box control and every audit attack
  pass <= 1 episode per pool (<= 3.7%; previously up to 43-68%). Harness twins identical to in-process (106/106 pre-crash,
  11/11 reference post-crash). Open: **local commit a90db0a7 contains per-probe answer labels for all 56 v3 instances (not
  pushed). Do NOT push task/shifthunt as is**: squash-merge, or regenerate the instances before agent runs. The T2 pool is small
  (10). C1 residual: gradient statistics separate planted from null at AUC up to 0.75 in T2. A perfect-attribute-name attack
  passes 11-17%, realistic attacks <= 3.7%. Claude API cap almost spent ($1.104/$1.20).
- **LatentDiff: READY_FOR_SMOKE** (task/latentdiff f55e9868). Generator v3.1 + grader v2, 24 instances (23 kept). Reference
  23/24 harness one-shot (23/23 kept); rarefreq/rarefreq2/blackbox/prior/meandiff*/kloracle/fingerprint 0/n; constant
  "nothing" 1/24 (the all-null instance). Sonnet probe (claude-sonnet-5-5, $0.153) found the intended constant-vs-proportional
  test and got 3/4 slots right; it missed one plant after checking only 16 texts. Mid-band signal, but the episode was INVALID through the R9
  false positive. The pool is small (24).
- **FeatureMatch fix stage: gates pass on its own terms** (178 kept; template_probe exactly at the 10% gate in T2/T3), but the
  diagnosis verdict supersedes it: status stays "explored, not validated". Local commits 3f3252c/ab78fb0a carry per-instance
  outcomes for 3 cross-check instances (excluded via exposed_instances.json). Do not push without review.
- **Harness hardening workflow wf_af386dd8-734**: transcript-audit false positives that invalidated real probes in 3 tasks
  (R9 exec of the agent's own scratch file; R2 sed slashes and "/"; R4 `nc` variable; bare codename), and validity gaps
  (run-scripted rc != 0 or no successful call counts as valid; infrastructure crashes graded valid:true with score 0). One fixer, then a full
  replay of every recorded episode under the old and new audit, plus an independent red-team (>= 40 synthetic breaches that must
  still be flagged), then resolution.
- Ready-for-smoke tasks are now BoolIntermediates, ShiftHunt and LatentDiff. Smoke order is Dan's call (D15 put
  BoolIntermediates after LatentKnockout).

## 2026-10-02 (~11:30 UTC): harness audit hardening done; LatentKnockout NO-GO-as-is; agent smoke of 3 ready tasks started
- **Harness (main 5d0f6fe1 + 1c5d13ba; 262 tests pass).** The 6 reported problems are fixed: R9 exec of the agent's own literal sandbox file;
  R2 sed regex parts and quoted '/' separators; R4 `nc` as a Python name; bare codename in agent-written files is counted,
  not a leak (path/module/canary forms still leak); run-scripted `solver_failed` / `no_successful_call`; `infra_failure`.
  Replay of all 35 unique recorded LLM episodes, old vs new audit: 8 flagged -> clean, all genuine false positives; 0 clean -> flagged.
  The replay verifier found 2 holes in the first fix, both closed in 1c5d13ba: (F1) the loosened R4 rule missed real
  `ncat --exec` / `ssh in.example.com` forms, so R4 now fires by default and exempts only Python-name uses; (F2) an agent could
  force `infra_failure` via `timeout N ./tool`, so client disconnect no longer counts as infra for LLM episodes. Still open: an
  agent-induced "server died" still reads as infra.
  **Gap:** the red-team stage did not run. That agent declined to write a corpus of breach transcripts. Coverage rests on the
  fixer's must-still-flag twin tests and the replay verifier's diff review. Still open: t2ravel probe 2 `print(a, '/', b)` false positive.
- **LatentKnockout feasibility: NO-GO-as-is (recipe)** (task/latentknockout 517e987f; FEASIBILITY_VERDICT.md). Small
  latent sets exist (30/43 pairs feasible at the best layer; median reference R 0.686 with k <= 10), but naive top-5 attribution reaches
  half the reference's R on 31/44 feasible cells (0.705) and decoder cosine with the mean difference on 24/44 (0.545). Skeptics: "top answer flipped"
  mostly counts near-ties (median clean lead 1.75 logits); with a 0.5-logit effect margin, check 1 (nothing small works) fires instead.
  The "Verify" variant fails its own recipe gate (a self-test script passes 0.76). Famous concepts (Texas, English, popular sports)
  cannot be knocked out with 5 latents. A strict redesign (LatentKnockout-Strict) is specified with 3 stop gates (~4 h GPU); the synthesis puts
  ~0.10 on it passing. Pitch findings: grading by rerunning the model removes the style problem; a top-1 flip metric counts near-ties; fame resists knockout.
- **Smoke (Dan: after the LatentKnockout verdict): workflow wf_6c6e6341-0ee.** BoolIntermediates (12 instances, T1-T4),
  ShiftHunt (6), LatentDiff (6). Fresh Claude Code subagents (claude-opus-5-5) see only the harness prompt. Per episode:
  operator prepare -> test agent -> operator finish (transcript located by episode id, prompt byte-match check) + transcript
  diagnosis. Then a per-task SMOKE.md. Main is merged into each task branch LOCALLY only (ShiftHunt/FeatureMatch history carries answer material).

## 2026-10-02 (~19:00 UTC): SMOKE results (wf_6c6e6341-0ee, 78 agents): Opus 5.5 passes 22/24. Two tasks too easy; harness faults
Fresh Claude Code subagents (claude-opus-5-5), one episode per smoke instance, profile full. Each SMOKE.md is committed locally on its
task branch (BI ffa48319+, SH 603778cf, LD 007c118e; none pushed). n is small: these runs find bugs, they do not estimate difficulty.
- **BoolIntermediates: 12/12 pass, 48/48 slots** (each tier 3/3). Too easy in every tier. All 12 agents rebuilt the network offline
  from `weights` and solved it exactly: they enumerated the readable 0/1 functions per residual, forced each candidate to 0 and 1 and counted output flips, and checked novelty by rank.
  Median 2/80 tool calls, 1.7% of forward, 8 min. Tier dials cost time, never accuracy. P(12/12 | predicted rates) ~ 2e-4.
  The builder's pre-registered rule "T4 >= 2/3 means raise the dial" fired. Fix direction: an activations-only tier (no `weights`).
- **LatentDiff: 6/6 pass, 24/24 slots.** Too easy. Plants are an exactly constant shift along one decoder row (per-token spread
  ~0.05-0.4), while decoys scale with a gate (spread of tens of units). One spread statistic separates them within 1-5 min. Predicted 15%/5%.
  Fix direction: match the decoys' and plants' per-token statistics, then re-gate.
- **ShiftHunt: 4/6 pass, 20/22 slots** (T1 2/3, T2 2/3; Wilson [0.30, 0.90]). The two failures are agent mistakes at thin margins:
  a null slot claimed at "60-70% likely" (null dependence 0.14 vs no-reliance cap 0.2), and a correctly named attribute whose ablation removed 0.216
  against a bar of 0.247 (the agent used 11 of 20 latents). 12/14 planted slots ended within 0.07 of the bar, so the bar placement drives part of
  the outcome. Possibly mid-band, but the evidence is thin. To do: null-scale disclosure, bar slack, a bigger T2 pool (<= 7 unexposed kept),
  >= 2 episodes per instance. Also sh-2411510359 had been seen by an earlier Haiku probe (F6).
- **Harness faults (13/24 episodes INVALID, all from the harness):** (F1) transcript auto-discovery also picks up the finish operator's
  own transcript, because its prompt contains the episode id; (F2) false positives: quoted '~' (NOT symbol), `| ps` / `; top` inside Python strings
  (R5), /dev/stdin in dead code, `exec(` inside a sed regex, `exec(open(own).read().split(...)...)` chains; (F3) test-agent hygiene:
  the workflow wraps the prompt in a "[Workflow harness ...]" preamble with indentation, and test agents start in the repo cwd with the
  repo's git status and recent commit messages in their environment context. That is a contamination risk.
- Next: harness fix workflow (F1-F3, including a sandboxed headless test-agent launcher if it can be made safe), then a re-audit of these 24.
  Task priorities after the smoke are Dan's call.

## 2026-10-02 (~21:00 UTC): harness smoke fixes done; official re-audit 21/24 valid; Dan picks ShiftHunt to n >= 20
- Workflow wf_c6af1340-fa5 (fix 02514eb4, replay 4fcb13cd). **F1** discovery now audits only the transcript whose first user
  message IS the episode prompt (exact, or inside the documented workflow wrapper); none or several candidates give R0 INVALID. **F2** false
  positives fixed ('~' / `| ps` / `; top` inside Python-only regions, /dev/stdin, exec( inside a sed regex, exec chains of literal
  splits) with must-still-flag twins. The fixer's own red-team found and closed an older gap (`./py -c "...os.system(\"ls ~\")"` passed every earlier version).
  **F3** headless sandboxed launcher NOT shipped: Claude Code's OS sandbox needs bubblewrap + socat (not installed). Without them a test
  write outside the dir and a curl both succeeded, so it failed the safety test. Fallback: the wrapper is documented and `prompt_match` is recorded.
  Enabling it later needs `sudo dnf install -y bubblewrap socat` (Dan's call) plus a launcher self-test.
- Replay finding **G1** (the Python-region exemption missed eval/attrgetter/methodcaller/__getattribute__ routes): closed by me in
  69a66c75 (guard tightened; 5 must-flag tests; replay of 59 transcripts unchanged; 383 tests pass).
- **Process note:** three verifier/resolver agents (the red-team in wf_af386dd8-734, the diff review and the resolver here) declined to write
  audit-evasion test inputs. Do not delegate "write inputs that get past the audit" again. Adversarial coverage comes from the
  fixer's twin tests, the replay verifier's diff reasoning and my own targeted must-flag tests.
- **Official re-audit** (runs/_harness/smoke1_reaudit.md): valid 21/24. BI 11/12 (11/11 pass), **SH 6/6 valid, 4/6 pass**, LD 4/6 (4/4 pass).
  Remaining INVALID: one BI episode with real hits, and 2 LD episodes with still-open false positives.
- Dan (D17): after the harness fix, push **ShiftHunt to n >= 20** only. BI and LD redesigns and LatentKnockout-Strict are held.

## 2026-10-02 (~21:15 UTC): ShiftHunt validation push launched (wf_96595a2d-e3a)
- Orchestrator decisions on the smoke design questions: **D1 = disclose** the reliance scale in TASK.md (a graded criterion should be
  stated; the one null false claim fell in that undisclosed gap; attacks get re-run with the disclosed numbers). **D2 = no change**
  to the removal bar (the T2 miss was a real selection error, 11 of 20 latents; lowering the bar helps name-plus-heuristic attacks).
  Continuous removal margins and agent/reference ratios are reported instead.
- Plan: prep (F4-F8 fixes, exposed_instances.json enforced, ~60 new T2 instances kept if the reference solves them, gates re-run,
  predictions committed) -> independent prep check -> 40 fresh-agent episodes (10 unexposed instances per tier x 2, waves of 3), with
  a re-smoke checkpoint after the first 6 -> SCALE1.md review -> adversarial check -> VALIDATION.md. Stop points: any gate failure,
  a failed prep check, or the checkpoint.
- Known exposure that remains: workflow-launched test agents still see the repo cwd, git status and recent commit subjects (no
  sandboxed launcher without bubblewrap/socat). This is documented, and it will be listed as a weakness in VALIDATION.md.

## 2026-10-03 (~00:00 UTC): ShiftHunt prep stopped at the T2 gate; T1 scaled run + T2 fix launched (wf_96d109e5-27e)
- Prep (wf_96595a2d-e3a) finished F4-F8, D1 (TASK.md "What relies means in numbers") and D2 (no bar change; margins.py), plus
  harness 9d4d38c0 (pushed). **Exposure:** every one of the 56 earlier design-v3 instances has per-instance outcomes in committed run summaries
  (and answer rows in local a90db0a7), so exposed_instances.json lists 59 ids. Fresh pools were built: T1b (16) and T2new (53 built).
- **T1b gates PASS**: reference one-shot 16/16 [0.81, 1.0]; every recipe, the black-box control and every attack (including the disclosed-scale
  variants) 0/16.
- **T2 gate FAILS on the kept pool**: label-free reference one-shot 23/53 (43%). On the 23 kept, nothing / black-box / crosspool /
  attrfp each pass 3/23 = 13% [0.045, 0.32]. On all 53 built, every non-reference solver is <= 3/53 = 5.7%. Cause: keeping only
  reference-solved instances kept 3/3 all-null episodes (13% vs ~6.7% by design) and favoured attribution-easy instances (attrfp 2/20 kept
  vs 0/30 dropped). Selection bias from a weak label-free reference (24/97 named planted slots below the bar).
- **Decision (orchestrator):** split the tiers. (A) T1 scaled run now: 10 T1b instances x 2 = 20 episodes, predictions committed first,
  checkpoint after 4, review + adversarial check. (B) In parallel, improve the label-free reference's T2 latent selection (no change to
  tools/grader/generator/TASK.md), re-keep, check the bias, re-gate. Dropping the keep filter is reported only as a sensitivity check, because it would admit
  instances not shown solvable through the tools.

## 2026-10-03 (~03:00 UTC): ShiftHunt T1 scaled run = saturated (19/20); T2 gate now passes; T2 scaled run + Haiku arm launched
Workflow wf_96d109e5-27e (65 agents). SCALE1_T1.md + SCALE1_T1_CHECK.md (task/shifthunt, local).
- **T1 (Opus 5.5, 10 fresh T1b instances x 2): 19/20 pass [0.76, 0.99]**, planted 43/44, null false claims 0/26. 9 instances passed both
  episodes, 1 split. Predicted 65%: **saturated, so T1 is a calibration tier.** All 20 agents ran one generic pipeline, the same as the
  label-free reference (agent/label-free removal median 1.001), so as an RL tier T1 would mostly reinforce one recipe. The one failure
  (miss by 0.004) and the closest pass (+0.016) came from the same judgement: dropping a latent that carried topic signal without discounting for the SAE error term.
  6/20 were falsely INVALID: 4 new audit false-positive patterns (quoted '/', D+'/tool', f-string '..', Python name `top`).
- Incidents: the T2 builder's test script briefly filled the RAM-backed /tmp (00:20-00:31 UTC). One T1 episode got one ENOSPC message,
  with no effect on its grade. E8: the session scratchpad named in test agents' context held operator dumps containing answers. No agent read
  it (every call stayed in its sandbox, and an outside read would be flagged R2). Operators are now barred from writing outside runs/.
- **T2 fix:** label-free reference v4.2 (better latent selection and naming, agent tools only) solves 39/53 T2new one-shot (was 23; on the old
  held-out T2 pool 18/29, was 10/29). **Gate passes on the 39 kept:** cheap solvers <= 3/39 = 7.7% [0.03, 0.20]; reference 39/39. The selection bias
  is reduced, not gone (every cheap-solver pass is on a kept instance), and the reference was tuned on this pool.
- **Launched wf_491d9234-493:** harness fix for the 4 FP patterns (+ C4) with must-flag twins, replay, and re-finishing the 6 T1 episodes; E2 (state
  the 96-token truncation) and E7 prompt clarity; T2 plan + predictions; **T2 Opus 10 x 2** (checkpoint after 4) in parallel with a **Haiku 4.5 arm
  on the same 10 T1 instances** (model-separation evidence); review, skeptic, VALIDATION.md.

## 2026-10-03 (~08:50 UTC): ShiftHunt T2 scaled run + Haiku arm; VALIDATION.md ("validated with caveats")
Workflow wf_491d9234-493 (96 agents). task/shifthunt (local): SCALE1_T2.md (+ check corrections, 2eb3fc78), VALIDATION.md (13a6d411).
Harness on main: 7345b11e (E1 false positives (a)-(d), 55 tests, replay of 79 transcripts: 7 flagged->clean, 0 clean->flagged), 2c6124d4
(E7 exec wording). After the re-finish, T1 is 20/20 valid, 19/20 pass.
- **T2, Opus 5.5 (10 fresh instances x 2): 15/20 = 75% [53, 89]** (predicted 55%, range 30-80). Naming 52/52, null false claims 0/18. All 5 failures are
  removal misses on correctly named probes (0.001-0.062 under the bar) on 3 instances. Per instance: 7 both pass, 1 split, 2 both fail. Outcomes
  are mostly fixed per instance (paired removal gap median 0.009, r = 0.95). T2 forces latent reading (4-35 reading calls per episode vs 21 in all of T1).
  Pool caveat: kept = solved by the tuned reference v4.2. A model built from the agent-minus-reference gaps estimates ~65-67% on all built instances (an estimate, not observed).
- **T1, Haiku 4.5 (same 10 instances): 0/10 [0, 28]** vs Opus 19/20 (Fisher p = 4e-7). Haiku fails on method: it named cues from
  co-occurrence (9/10), rarely encoded its own texts, and never used the reliance scale. A strong model-separation result; the dial has two points.
- **Verdict (VALIDATION.md): validated with caveats as a test of interpretability skill; not yet validated as an RL training environment.**
  T1 = calibration tier (saturated for Opus). T2 = the reading tier, near the top of mid-band and bar-sensitive (+-0.02 on tau gives 14-17/20).
  Agents are close to interchangeable with the reference pipeline.
- **Harness problems this run exposed (structural):** the regex transcript audit keeps producing new false positives (T2: 9/20
  falsely INVALID, 10 new patterns, after the T1 fix) and has known false negatives (process/network calls through subprocess, os.system, popen
  or sockets). Test agents are Claude Code subagents of the operator session: they see the repo cwd, commit subjects, the account email and the shared
  scratchpad / tool-results dirs, where operator output (including answer-bearing dumps, some saved automatically by Claude Code) ended up. No agent used them
  (one Haiku agent read its own saved output there and was flagged INVALID). The structural fix is OS isolation for test agents (bubblewrap
  + socat, then a headless launcher outside the operator's session), which needs Dan's sudo approval.
- Not done: the truncation contrast (the 96-token cut gives a free within-class cue contrast; at least 7/20 T2 agents used it) is not in the gates;
  t2ravel ep3ab5ab5782 needs a re-finish.

## 2026-10-03 (~19:40 UTC): ShiftHunt cross-family check with the OpenAI key (Dan's request)
- Dan asked for validation with the OpenAI key. Why it helps: (1) a second model family; (2) common/openai_agent.py starts the agent
  inside the episode sandbox (scrubbed env, no keys, a command guard, no session/repo context), unlike workflow-launched
  Claude subagents, which see the repo cwd and commit subjects and rely on prompt rules plus the post-hoc audit. Dan confirmed
  his understanding: the Claude test agents were not OS-isolated. The OpenAI launch is not a kernel sandbox either.
- Per-task OpenAI cap for shifthunt raised to $30 (4aa4ffa1); global stop $57 unchanged. Same instances as the Claude arms (paired). gpt-6.1-sol
  and gpt-6-luna at effort medium, max-turns 90, max-usd 1.5. A 2-episode smoke run was clean, then the batch ran 3 at a time.
- Preliminary (28/30 done; spend ~$5.5): **T2 sol 7/10** (Opus 15/20), **T1 sol 10/10** (Opus 19/20), **T1 luna 0/8** (Haiku 0/10). The same
  pattern holds across families with no repo/session access, so the Claude numbers were not inflated by context exposure. Open: one T2 sol
  pass is INVALID by the audit (false positive?); two T1 sol episodes score 1.0 though the runner says "ended_without_submit".
- Review workflow wf_7b4cbd23-c69: waits for the last 2 luna episodes, writes SCALE1_OPENAI.md (paired with the Claude arms), skeptic,
  then the VALIDATION.md cross-family section.

## 2026-10-03 (~21:00 UTC): cross-family result final (wf_7b4cbd23-c69); VALIDATION.md updated (task/shifthunt 2ca3ab37, local)
- All 30 OpenAI episodes are graded; an in-memory regrade matches 30/30; spend $5.60. All 840 commands stayed in their sandboxes.
- **T2 gpt-6.1-sol 7/10 [40, 89]** vs Opus 15/20 [53, 89] (Fisher p = 1.0; difference -38 to +24 points). **T1 sol 10/10** vs Opus 19/20. **T1 luna 0/10** vs
  Haiku 0/10. Same profile in both families: naming and null calls perfect for the large models; every T2 miss is a removal miss on a named probe, all on the 13/26
  probes where the reference clears the bar by < 0.05. Differences: in T2 sol's margins sit below Opus's (19/26 probes, Wilcoxon p = 0.005; a 10% higher bar gives 3/10 vs 13/20);
  in T1 sol and Opus agree as closely as two Opus agents (latent-set Jaccard 0.82). Confound: OpenAI ran at effort medium, Opus at Claude Code xhigh.
- Context-exposure question (Dan): agents confined to the sandbox reach the Claude rates, which is evidence against large inflation of the Claude results (not proof;
  the direct control would be Opus run confined).
- Anomalies: (a) a T2 sol pass flagged INVALID is an audit false positive (`.replace('\n','/')` inside print); (b) 2 T1 sol "ended_without_submit" are a runner
  reporting bug (a submit via subprocess list form is missed by the regex; the episode records show submitted). Both are open harness fixes. The runner also truncates tool output at 12k chars
  (25/30 episodes), with no grade affected. _openai_jobs/ is gitignored now.

## 2026-10-03 (~21:30 UTC): why do small models fail ShiftHunt T1? Investigation launched (Dan's request; wf_c5e1afa5-f03)
- Haiku 4.5 0/10 and gpt-6-luna 0/10 on the same 10 T1 instances that Opus (19/20) and sol (10/10) pass.
- Design: 4 forensic analysts in parallel (Haiku transcripts; luna transcripts incl. runner truncation; environment-fault hunt: prompt length,
  tool errors, output truncation, caps, context pressure, effort; stage-wise numbers with in-memory counterfactual grading). Each main claim
  goes to a skeptic told to refute it (default refuted). Concurrently, 3 pre-registered DIAGNOSTIC ablations (non-scored) on the same instances: luna effort
  high; luna + a generic method-hint paragraph; Haiku + the same hint (predictions in tasks/shifthunt/diag/PREDICTIONS_DIAG.md before
  any episode; luna budget < $3). Then a synthesis: SMALLMODEL_FAILURES.md + the VALIDATION.md small-model section.

## 2026-10-04 (~00:30 UTC): why small models fail ShiftHunt T1 (lite analysis; SMALLMODEL_FAILURES.md, task/shifthunt d07506e6, local)
- Dan cut the planned ~50-agent investigation (too many tokens). Stopped; reused the finished stage analysis; then 2 agents at medium effort.
- **Haiku 4.5: no method.** It names cues from label co-occurrence in the sample (naming 7/22), claims a cue on every null (13/13), almost never measures
  the probe on its own texts (1/10 episodes with >= 5 matched pairs), ignores its own null evidence (a pair showing no effect, followed by a claim anyway),
  misreads the scale and stops early (21-35 of 150 calls). With perfect naming and null calls it would still fail 8/10. Its tool errors are self-inflicted.
- **gpt-6-luna: method present, calibration missing.** Naming 19/22. It uses matched pairs, picks the right latents (88% overlap with the big models)
  but only ~4 of 20, and never checks removal against the bar. Fixing removal alone gives 6/10.
- Diagnostics (pre-registered; partial, still running in background): **effort high** 0/4 (1/7 named slots clear the bar), so effort is not the explanation;
  **method hint** 0/5, but removal now clears the bar on 8/9 named slots while luna calls 3/12 planted cues "none". T1 tests finding and carrying out the
  procedure end to end; a small model fixes one stage and breaks another.
- No environment fault singles out the small models (no caps hit). One TASK.md ambiguity: "held-out texts" vs scoring removal on the confounded
  sample. Exposure asymmetry: 12 Opus episodes read the ./tool wrapper code; the small models never did (not causal here).
- (2026-10-04) **Why gpt-6.1-sol failed 3/10 on T2** (details: task/shifthunt SCALE1_OPENAI.md "Failure classification"): all 3 are removal
  misses on correctly named probes (naming 26/26, null false claims 0/9). (A) 1 slot used only 5 of 20 latents (-0.034), on the probe both Opus agents
  also missed. (B) 2 slots used a full 20 chosen as cue-pure or by sample attribution, missing latents that carry the cue through context (-0.003, -0.037).
  Same failure kinds as Opus; all misses are on the probes where the reference clears the bar by < 0.05. sol sits closer to the bar than Opus
  (effort medium vs xhigh confound).

## 2026-10-04: docs consolidated (Dan: "push everything we just talked about into the docs")
- New docs/STATUS_2026-10-04.md (start-here summary) and docs/results/ (redacted, aggregate-only copies of ShiftHunt VALIDATION/SCALE1/
  SMALLMODEL_FAILURES/SMOKE, FeatureMatch VERDICT/RESULTS/STEP3, LatentKnockout FEASIBILITY_VERDICT/PRECEDENT, BoolIntermediates and
  LatentDiff SMOKE; instance ids redacted). INDEX.md updated.
- Final luna diagnostics: effort high 1/10 (4/20 named slots clear the bar, median 4.5 latents); method hint 1/10 (14/17 clear, median 20 latents,
  naming 17/22). Effort is not the cause; the hint fixes removal but not naming or null calls. task/shifthunt c6c37109 (local).

## 2026-10-04: per-episode luna report (docs/results/shifthunt/LUNA_EPISODES.md)
One Sonnet subagent, scripted over grade/tool logs. Luna's failures are on the interpretation and method side, not the scaffold: 10/10 valid first-try submits, no cap hit, 1.3% tool errors, and all 10 loaded the full latent_means files. Planted naming is 19/22 (chance 1/6). The 3 misses answered "none". Null false claims are 5/13, 4 of them in 2 episodes that never compared cue vs topic effect. The dominant failure is under-fill: a median of 4 of 20 latents, with removed/tau at a median of 0.68 and no overshoot. Sol and Opus always fill 20. Slots solved: 10/35 at baseline, 12/35 at high effort, 23/35 with the method hint.

## 2026-10-04: why luna submits ~4 latents (spot check of 2 transcripts' final commands; OpenAI reasoning is hidden)
The format rule is clear ("at most 20"). What pushes a cautious agent toward a minimal set:
(1) the bar is hidden: TASK.md says only "between 0.15 and 0.6";
(2) TASK.md says overshooting counts against you and the topic must stay >= 80%, but never says that extra latents are free;
(3) luna checks its ablation on the confounded sample, where cue and topic move together (in one episode the exclamation dependence equals the topic dependence exactly). Its own check reads as topic damage (-6 to -11%), and it trims: it tested 6 latents and submitted 3, tested 5 and submitted 4.
No luna slot overshot. Sol and Opus fill the 20 regardless; the method hint fixes removal (14/17).
Same open item as STATUS 7.3: state whether filling the budget is free, or add a per-latent cost; state that removal is measured on held-out data.

## 2026-10-04 06:30 UTC: ShiftHunt v3 started (D18)
Worktrees: ~/wt/shifthunt3 (task/shifthunt3), ~/wt/v3harness (harness/v3-fixes), ~/wt/v3iso (harness/isolation). bubblewrap and socat were already installed.
Workflow 1: planner (xhigh); harness fixes and isolation launcher in parallel, each audited; then the Stage 0 build and gate plus the grader build in parallel, each audited, with redesign loops.

## 2026-10-04 08:43 UTC: harness branches merged into main (harness/v3-fixes, harness/isolation)
- **Pre-merge fix on harness/v3-fixes (c36c3725).** Audit r1 A was a regression vs main: exemption (e) (print-replace '/' false positive) also
  cleared `print(file=...)`, `print(**kw)` and prints under `contextlib.redirect_stdout` or a rebound `sys.stdout`, where the region reads the
  printed text back as a path. Now only a plain print to stdout is exempt. Its xfail(strict) test is now a normal must-flag test with 6 cases
  (each fails on the unfixed code). New E_STDOUT_OK twins check that the intended false-positive fix stays valid. Branch suite: 523 passed, 3 xfailed.
- **Replay** (main 0f28f81d audit vs the fixed branch; every recorded transcript under ~/wt/*/runs/** and main runs/**; 164 units, 5049 tool
  calls). Only one verdict changed, the intended one: shifthunt openai_T2_sol ep2c6190e0f5 went flagged -> clean (call 15, R2 root).
  143 clean->clean, 20 flagged->flagged, 0 violations added. One stray copied transcript outside an episode dir
  (shifthunt epdbe6a54a7e/finish_run1_autodiscover) gave identical violations. Record: runs/_harness/audit_replay_v3fixes/replay_merge_r1A.json.
- **Merges (--no-ff):** 5fe5e6c8 harness/v3-fixes (hard wall-clock ceiling on agent time, runner stop from the episode record, sandbox
  TMPDIR, saved long outputs, audit print-replace fix); 52d0cdb7 harness/isolation (claude_agent: bwrap + netns + egress allowlist +
  auth-injecting proxy, scripted red-team, self-test, HARNESS_API section 11). Conflicts were in common/sandbox.py and docs/HARNESS_API.md.
  Both were additions at the same place, so I kept both. finish() runs both `_launcher_check` and `episode_time_report`.
  Integration edit: claude_agent TMPDIR changed from `<sandbox>/.tmp` to the v3 `<sandbox>/tmp`. The agent prompt names tmp/ and finish
  prunes it, but nothing pruned .tmp. HARNESS_API: added an r1 A note under (e) and a time-limit note in section 11.
- **Tests on merged main:** `common/tests`: 536 passed, 3 xfailed (strict), 0 skipped.
- **Known gaps (xfail strict in common/tests/test_v3_audit_round1.py, not fixed):**
  (B) the runner's hard ceiling counts from runner start, not from the first tool call as TASK.md says, so time before the first ./tool call
  shortens it.
  (C) grade.json over_time is None for an unsubmitted episode that the runner stopped past the nominal cap when no tool call came after the
  cap (the luna pattern).
  (D) a submission that lands after the runner exit, from a detached process that cleared RL_AGENT_PROC, is graded and the episode stays VALID.
  Isolation/v3 integration: claude_agent has no v3 runner clock (no time notice, stop only at --max-wall-s). The broker still enforces both caps on
  its tools and submit, so set --max-wall-s <= wall_clock_hard_s.
  The audit-isolation residual risks are listed in HARNESS_API section 11.
- **Broker restart.** Before the restart: 5 _demo episodes in the registry; 3 were "open" but idle (52-114 min, no tool server, no runner/jail
  process; one belongs to the v3iso audit scratch). No episode was active, so I ran stop/start (records persist on disk). The pre-restart ping returned
  {pid: 186434}; the new broker (pid 832288) pings {}. Red-team re-run against the merged code and the restarted broker (no LLM): 115/115
  probes PASS, 0 escaped, 0 minor escaped (broker_ping_no_pid now passes; tmpdir_in_sandbox passes with tmp/). Its 2 _demo episodes
  (epbc0ab970f5 red-team, flagged cross_episode_access on purpose; ep6b368976cf decoy) stay open: never grade them.
- Audit scratch in ~/wt/v3harness and ~/wt/v3iso (runs/_audit_*, runs/_demo/*_claude_iso_selftest) was left untracked. No task branches were touched.

## 2026-10-04 09:10 UTC: ShiftHunt v3 closed at Stage 0 (D18 outcome)
- 06:30: started. The planner (xhigh) wrote SPEC_v3 by 07:05, testing first on cached v2 data. Builders A (Stage 0), B (grader) and C (tools/prompts) worked in parallel; each piece was audited.
- 08:10: Gate 0 passed on held-out data (top-|g| 5.9%, single-group 0%, reference 100%).
  - Builder A flagged `graddiff` itself. The Stage 0 audit failed the result: a sample gradient difference solves 96% with no names.
  - The tools audit found the fixed-gradient + effect-tool linearity test (81/81 named) and the sample-pattern leak (86/86 mechanisms).
- 08:40: the planner stopped v3 rather than redesign.
- 09:00: a skeptic re-tested the decision and upheld it. Names add +6 to +17 points; no budget-respecting script reaches 60%; nuisance units are beaten by an interaction script (60/60).
- I stopped the workflow before it finished the grader fix loop. B's uncommitted edits are left in ~/wt/shifthunt3.
- Copies on main: docs/results/shifthunt3/ (SPEC_v3, STAGE0, KILLED; aggregates only, scanned for ids). STATUS section 2b.
- Harness work merged earlier (edc661f7).
