# ideating-rl-tests: An Audited Portfolio of Agentic Interpretability RL Environments

Final project report. Covers work from 2026-10-01 to 2026-10-05.

## Abstract

This project designed, built and tested candidate reinforcement-learning (RL) environments in which an
agent must use mechanistic-interpretability tools to recover a hidden, planted ground truth inside a small
language model, and a programmatic grader scores the answer. 41 candidate environments were scored by two
independent scorers and 6 were picked for building. A shared evaluation harness (`common/`, 7,711 lines excluding tests,
536 tests passing at the last merge) gives answer isolation, exact budgets, out-of-process grading and a
transcript audit. An OS-isolated agent launcher contained 115/115 red-team probes. The headline result is
**ShiftHunt v2**, an interpretability test with an exact grader built on linear probes over gemma-2-2b and
Gemma Scope SAE latents. It separates models in two model families: on the open tier (T1) Opus 5.5 scored
19/20 and gpt-6.1-sol 10/10, while Haiku 4.5 and gpt-6-luna each scored 0/10. On the harder sample-only
tier (T2) Opus 5.5 scored 15/20 (75% [53, 89]) and gpt-6.1-sol 7/10 on a reference-filtered pool, while cheap scripted
solvers scored at most 7.7%. Nine other designs were stopped or held by pre-registered gates, scripted-recipe hunts or
agent smokes, each with a measured reason, and a model-diffing study stopped on a pre-registered kill
because its grader failed validation. **Final status:** ShiftHunt v2 is validated, with caveats, as an
evaluation of interpretability skill. It is not yet validated as an RL training environment. All other
candidates are documented negative results. The work is complete as a feasibility study; the open items in
section 7 are what a training-grade version would need.

## 1. Motivation and goal

RL training of agents on interpretability work needs environments that score real interpretability skill.
Many plausible task designs fail quietly: a cheap script solves them, the answer leaks through files or
tool outputs, the task is too easy, or the grader rewards the wrong thing. The goal was to find out which
designs survive those failure modes, and to build infrastructure that makes agent scores trustworthy.

Each candidate was scored on a 9-criterion rubric (1-5 per criterion, max 45). The criteria that decided
selection were, first, that the task actually **needs interpretability** (`needs_interp`) and, second, that
it is **hard to game** (`hard_to_game`). The other criteria cover the properties of a good RL environment:
automatic verification, procedural generation, a pass rate in the middle band (useful reward variance),
honest null instances, and so on (`ideation/CANDIDATES.md` section 1).

Two questions ran through the whole project:

1. Is a given environment hard for the right reasons, or can a no-hypothesis script, a structural leak or
   simple ease solve it? Can pre-registered gates decide this before money is spent on agent episodes?
2. Can a shared harness make agent scores trustworthy, with containment violations and broken graders
   showing up as INVALID rather than as pass or fail?

## 2. What was built

### 2.1 Ideation

- `ideation/CANDIDATES.md`: 41 candidates in 3 lanes, the 9-criterion rubric, two scorer lenses
  (RL engineering and interpretability), a disagreement-resolution table, the Wave-2 selection of 6 with
  build briefs, and 4 runners-up.
- `ideation/research_raw.json` (41 merged candidate records) and `ideation/scores_raw.json`
  (2 independent scorings of all 41).

### 2.2 Shared evaluation harness (`common/`)

| Component | Path | What it does |
|---|---|---|
| Broker | `common/broker.py` | Unix-socket broker. One tool-server subprocess per episode. Caps and counters (150 tool calls, 4,000 forward units, 200 generate, 50 gradient, 3,600 s wall clock, 180 s per call, plus a hard wall-clock ceiling). Built-in `help`, `budget`, `submit`. Privileged tool log, idle eviction (600 s), 8 GB RAM cap, admin token, caller binding by cwd, 8 MB request limit, streamed charges. |
| Leak scan | `common/leakscan.py` | Scans every response and the final state for canaries, leak strings and private strings (NFKC + casefold, raw string leaves, canary tail, file names). |
| Tool server and client | `common/toolserver.py`, `common/toolclient.py`, `common/agent_client/tool` | Task API (`TaskEnv`, `@tool`, `ToolError`, `charge`), safe array writes (`O_EXCL|O_NOFOLLOW`), a stdlib-only Python 3.9 client for agents. |
| Sandbox | `common/sandbox.py` | `setup / prepare / finish / run-scripted / summarize`. Per-episode instance snapshot, out-of-process grader with timeout and output validation, Wilson CIs, validity reasons (`infra_failure`, `solver_failed`, `no_successful_call`, `min_submit_bypassed`, `cross_episode_access`, `sandbox_symlink_outside`). |
| Transcript audit | `common/transcript_audit.py` | Rules R0-R9 that mark an LLM episode INVALID if it leaves its sandbox. Exact-prompt transcript discovery. Python-source exemptions, each paired with a must-still-flag twin test. |
| GPU queue | `common/gpuq.py`, `common/paths.py`, `common/pressure_evictor.py` | Machine-wide GPU memory ledger (21 GB of a 23 GB L4, max 7 jobs, max 2 heavy), heavy-slot fairness, aging, eviction of idle tool servers under pressure. |
| API agents | `common/api_agent.py`, `common/openai_agent.py` | Test agents with one bash tool, scrubbed environment, a shared command guard, per-task and global dollar caps and a spend ledger. The OpenAI version is a twin for a second model family. |
| Isolated launcher | `common/claude_agent.py`, `common/tests/claude_agent_selftest.sh` | OS-isolated headless agent launcher: bubblewrap with new namespaces, loopback-only network, an auth-injecting proxy outside the jail with a fail-closed request-body policy, a proxy/stream cross-check, about 115 scripted red-team probes. |
| Tests | `common/tests/` | 26 files, 175 test functions (536 collected tests at the last merge), adversarial fixture tasks `verify_task/_vcheck` and `gpu_task/_gpusmoke`. |
| Worked example | `tasks/_demo/` | GPU-free example of every required task file (262 lines). |

Contract and rules: `docs/HARNESS_API.md`, `docs/HARNESS_VERIFICATION.md`, `docs/BUILDER_GUIDE.md`,
`docs/DECISIONS.md` (D1-D19), `docs/LOG.md`. Audit replay records: `runs/_harness/`.

### 2.3 Environments

| Environment | Idea | Final state |
|---|---|---|
| ShiftHunt v2 | Name the surface cue a linear probe relies on (or "none") and list at most 20 SAE latents whose ablation removes that reliance | Validated with caveats as an evaluation |
| ShiftHunt v3 | Planted nonlinear OR/AND modules over SAE latents, structured claims, continuous grader | Killed at Stage 0 |
| FeatureMatch | Match SAE latents to concepts from a 20-option menu | "Not interpretability" (stop rule fired) |
| LatentKnockout | Find a few latents whose ablation flips a model's answer | NO-GO as is; strict redesign held |
| BoolIntermediates | Recover intermediate variables of a planted Boolean network | Too easy (12/12) |
| LatentDiff | Find planted latent-level differences | Too easy (6/6) |
| EditFind | Locate a ROME edit | Dropped (answer key ill-posed) |
| T2Ravel | RAVEL-style attribute localisation | Dropped (layer predicts null) |
| SilentPathway | Find a silent pathway | Dropped (frontier probes solve it in 10-13 turns) |
| ResidualRecall | Wave-2 pick | Held (T2 reference 1/8) |
| ModelDiff | Diff a base model against a LoRA fine-tune and submit checkable claims | Stopped on kill K5 (grader invalid) |

ShiftHunt v2 pieces: 3-4 linear probes per episode on a middle layer of gemma-2-2b, each telling two hobby
topics apart, with one of 8 listed surface cues (sign-off line, greeting, British spelling, question
ending, exclamation marks, hashtag, bullet layout, negative sentiment) possibly correlated with the label;
about a third of probes are null. Two tiers: T1 "open" (the SAE encodes any text the agent writes, so it
can build matched counterfactual pairs) and T2 "sample-only" (the SAE encodes only given samples, so the
agent must read latents from top-activating examples and per-token activations). Label-free reference
solvers v3.1 (T1) and v4.2 (T2). Pools T1b (16 instances) and T2new (53 built, 39 kept).
Write-ups: `docs/results/shifthunt/` and `docs/forensics/SHIFTHUNT_FORENSICS.md`.

## 3. Methods

**Models under study.** gemma-2-2b with Gemma Scope SAEs (ShiftHunt v2/v3, FeatureMatch, LatentKnockout);
Qwen3-1.7B with public LoRA organisms (ModelDiff).

**Test agents.** Opus 5.5, Haiku 4.5 and Sonnet 5.5 (Anthropic; Sonnet only in early API probes), gpt-6.1-sol and
gpt-6-luna (OpenAI). Claude test agents ran as Claude Code subagents of the orchestrator session (Tier B containment: honor system plus post-hoc audit); OpenAI
agents ran through `common/openai_agent.py`, confined to a sandbox cwd with a scrubbed environment and a
command guard.

**Interventions.** Linear probes; SAE latent mean-ablation computed as exact linear arithmetic (ShiftHunt
v2); bit-exact ablation that keeps the SAE error term (LatentKnockout); LoRA adapter toggling (ModelDiff).

**Graders.** No LLM judge in ShiftHunt. Its grader uses a hidden, balanced topic x cue grid with 48 texts
per cell. A planted probe passes only with the right cue name, at most 20 latents, removal at least tau,
and at least 80% of the topic reading kept. Episode reward is binary. Tau is 0.85 x the 20th percentile of
a label-knowing reference's 20-latent removal, clipped to 0.15-0.6 and frozen per cue family (frozen bars
0.164-0.345). ModelDiff used a behavioural claim grader with LLM-written 3-option tests scored by
fine-tune-vs-base option likelihood.

**Baselines and gates.** Every task has a reference solver and a suite of cheap solvers: always-none,
claim-everything, prior guesses, topic-attribution recipes, gradient/attribution fingerprints, cross-probe
contrasts, memorisation, black-box controls and red-team scripts. A task is only worth agent episodes if
the reference passes and the cheap solvers stay below a pre-set bar.

**Null design.** Episodes have several slots with 30-50% null slots, and an episode passes only if every
slot is correct. With 3-4 slots and a 40% null rate, constant answers stay well under 10%
(`docs/BUILDER_GUIDE.md`, `docs/DECISIONS.md` D7).

**Statistical practice.**
- Held-out splits: DEV instances for building, fresh HELD instances for gates and agent runs.
- Wilson 95% intervals on every rate; the implementation matches scipy to within 2e-4 on tested n from 1 to 1,000.
- Fisher, Wilcoxon, sign, McNemar and Newcombe tests for comparisons; bootstrap SEs for grader noise.
- Pre-registration: predictions, gates and kill rules were committed before most run families
  (LatentKnockout: 38 predictions; ModelDiff: gates G1-G7, kills K1-K5). No predictions were written before
  the OpenAI cross-family ShiftHunt run. Late-written predictions are disclosed.
- Audits: independent skeptic passes re-derived the key numbers (ShiftHunt regrades, FeatureMatch step 2
  re-implemented with 0/5,017 mismatches, LatentKnockout decision recomputed by `verdict_check.py`).
- Smoke runs (n = 6-12) are treated as bug-finding, not as difficulty estimates.

## 4. Results

### 4.1 Ideation

| Item | Value |
|---|---|
| Candidates scored | 41 (3 lanes, 9 criteria, 2 scorers, 369 cells) |
| Cells where scorers differed by 2 or more | 8 of 369 (7 candidates) |
| Resolved calls that changed a total | 1 (C21, 30.5 -> 29.5); none changed the selection |
| Wave-2 picks (adjusted total / 45) | C27 ShiftHunt 33.5, C02 BoolIntermediates 33, C03 SilentPathway 32.5, C20 LatentDiff 32.5, C13 ResidualRecall 31.5, C37 T2-RAVEL 28 |
| Runners-up (never built) | C29 33, C30 ConflictHeads 31.5, C09 VarLocalize-DAS 31.5, C32 30.5 |

- The two scorers agreed closely, so the selection was stable.
- The ideation pass-rate ranges were guesses and were poorly calibrated. BoolIntermediates (guess 20-70%)
  scored 12/12 and LatentDiff (guess 30-70%) scored 6/6 in small smokes.
- Of the 6 picks: 1 validated with caveats (ShiftHunt), 2 too easy, 2 dropped, 1 held.
- T2-RAVEL was included below the score cut on the project owner's request (28/45). It was later dropped.

### 4.2 Harness verification

| Check | Result |
|---|---|
| Adversarial review (2026-10-01) | Builder's own tests all passed; new attack tests found 23 failing harness checks and 32 audit failures (29 of 50 evasion transcripts passed the old audit). All fixed; 191 tests then passed incl. GPU smoke. |
| Example holes fixed | 32 MB requests accepted (now 8 MB limit); charge-then-hang call was free (7 units now kept); hanging grader could block finish up to 3,600 s (now killed at a timeout); NaN / 7.0 / "high" grader scores accepted; symlink-following array writes; open admin commands on the socket. |
| Cap enforcement under concurrency | 4 threads x 2 episodes, cap 6: exactly 6 succeed per episode, 8 refused |
| Planted-vs-null fingerprint | None found across 40 demo instances (response timing not checked) |
| Re-audit after false-positive fixes | 95 transcript dirs (35 unique episodes): 10 INVALID before, 2 after; no VALID episode changed |
| Agent-triggerable validity hole | `timeout N ./tool` could turn a fail into `infra_failure`; client disconnect now counts as infra only for scripted solvers |
| First agent smoke | 13/24 INVALID, all caused by the harness (operator transcripts picked up, audit false positives, launch hygiene); after the fix, 21/24 valid |
| Largest audit replay | 164 transcript units / 5,049 tool calls: exactly 1 verdict change, the intended one; 0 clean -> flagged |
| Isolated launcher red team | 115/115 probes contained; without the jail, 72/114 escaped |
| Test suite | 86/87 -> 191 -> 262 -> 383 -> 445 -> 536 passed, 3 strict xfail (merged main, 2026-10-04) |

- The adversarial review showed that a passing self-written test suite said little: an independent attack
  pass found 55 failures.
- Every fix was checked by replaying recorded transcripts; across replays no clean episode became flagged.
- The isolation audit found that the model API itself was an exit path (web fetch, web search, and a
  request for a different model being served). A fail-closed request-body policy in the proxy closed it.
- Negative: the regex transcript audit kept producing new false positives at scale (about 20 patterns
  fixed), and has known false negatives (subprocess, `os.system`, popen, sockets, `chr()`-built paths).
  Three known gaps remain as strict-xfail tests.

### 4.3 ShiftHunt v2 (headline study)

**Gates (scripted solvers).**

| Tier | Reference | Best cheap solver |
|---|---|---|
| T1 (16 instances, gates v8) | 16/16 | 0/16 [0, 19%] for every recipe, attack and black-box control |
| T2 (gates v9, kept pool) | 39/39 (39/53 of built kept) | at most 3/39 = 7.7% [upper 20%] |
| T2 (earlier gate, failed) | 23/53 (43%) | 3/23 = 13% [4.5, 32] |

**Agent runs (10 fresh instances per arm).**

| Arm | Pass rate [95% CI] | Planted cues named | False claims on nulls |
|---|---|---|---|
| T1 Opus 5.5 | 19/20 = 95% [76, 99] | 44/44 | 0/26 |
| T1 gpt-6.1-sol | 10/10 = 100% [72, 100] | 22/22 | 0/13 |
| T2 Opus 5.5 | 15/20 = 75% [53, 89] | 52/52 | 0/18 |
| T2 gpt-6.1-sol | 7/10 = 70% [40, 89] | 26/26 | 0/9 |
| T1 Haiku 4.5 | 0/10 [0, 28] | 7/22 | 13/13 |
| T1 gpt-6-luna | 0/10 [0, 28] | 19/22 | 5/13 |

- The test separates models in both families. Haiku vs Opus on T1: Fisher p = 4e-7. T1 sol vs luna:
  p = 1e-5. Large models never misnamed a cue (144/144) or claimed one on a null probe (0/66).
- Cross-family replication holds for pass rate (T2 sol vs Opus Fisher p = 1.0, difference -5 points,
  interval -38 to +24) but not probe by probe: sol sits closer to the bar (Wilcoxon p = 0.005).
- Every large-model miss was a removal miss on a correctly named probe: short latent lists (9-11 of 20 for
  Opus, 5 for one sol miss), dropping a latent that carries the cue through context, or near misses within
  grid noise (3 Opus misses at -0.001 to -0.004). All 8 failing T2 probe-episodes (5 Opus, 3 sol) fell on the
  13 of 26 probes where the reference itself clears the bar by less than 0.05.
- The grader is correct: regrades reproduce every stored slot verdict (80/80 agent episodes per
  `VALIDATION.md`).

**Why it is not yet an RL environment (forensics, `docs/forensics/SHIFTHUNT_FORENSICS.md`).**

| Issue | Measurement |
|---|---|
| The bar decides how many episodes miss | tau +-0.02: T2 Opus 14-17/20; tau +0.05: T2 Opus 5/20; bar at 0: 20/20 |
| Grid noise | 8 of 110 Opus slot verdicts in the coin-flip zone (0.1 < P(pass) < 0.9) |
| Free "fill the budget" reward | padding the 4 Opus short lists with top-|g| latents turns all 4 into passes (about +3 expected under grid resampling) |
| Hacking route (a): answer files | answer reachable in three plain file reads from the sandbox; only the regex audit defends it |
| Hacking route (b): 96-token truncation | names end-of-text cues for free (AUC 0.95 over three cues, proxy effect size); used by at least 7/20 T2 Opus agents |
| Recipe ceiling once names are free | top-|g| latents pass 14/39 (36%) of the T2 kept pool |
| Filtered pool | T2 pool kept by a reference tuned on it; estimated about 65-67% on all 53 built (a model estimate, not observed) |
| Low GRPO variance | outcomes mostly fixed per instance (ICC about 0.73); forensics estimates about 10-31% of groups with any variance (G = 2-16) |

**Small-model diagnostics (pre-registered, gpt-6-luna, 10 episodes per arm).**

| Arm | Pass | Named | Null false claims | Median latents |
|---|---|---|---|---|
| Baseline | 0/10 | 19/22 | 5/13 | 4 |
| Effort high | 1/10 | 20/22 | 5/13 | 4.5 |
| Medium + method hint | 1/10 | 17/22 | 4/13 | 20 |

- Effort was not the cause. A method hint fixed removal (14/17 named slots cleared the bar) but did not
  improve naming (17/22) or null calls (4/13), so only 1/10 passed.
- Haiku had no method (named cues from co-occurrence in 9/10 episodes). luna had the method (8/10 built at
  least 5 matched pairs) but under-filled its latent list.

### 4.4 ShiftHunt v3 (killed at Stage 0)

| Item | Value |
|---|---|
| HELD gate set | 25 episodes, 86 slots (62 planted, 24 null) |
| Gate 0 GATED items | all passed: topg20 3/51 = 5.9% (bar < 15%), single_group 0/51, reference 62/62 |
| Kill: no-name gradient-difference script | 49/51 = 96.1% [86.8, 98.9] of OR+AND slots, 128 forward units |
| Free sample-logit leak | planted vs null AUC 0.989 (bar 0.65) |
| Best remedy tried | ablation-only within caps: names-using 32% vs reference bar 60% (unreachable) |

- The planted nonlinearity was real (OR and AND heads behaved as designed on the hidden grid), but every
  other term of the head was linear, so a script exposed the module through the tool algebra.
- Each remedy left another script or made the reference bar unreachable. An independent skeptic upheld
  the kill. No model episodes ran.

### 4.5 FeatureMatch ("Not interpretability")

| Item | Value |
|---|---|
| Style-robust script SR-max / reference | 0.873 [0.840, 0.903] on 441 planted slots |
| SR-thr / reference | 0.730 [0.688, 0.771] |
| Robust across 42 text selections | SR-max 0.862-0.955 |
| In-budget script using only the task's `generate` tool | 252/441 = 0.571 (not pre-registered) |
| SAE latents passing the style-robust keep rule | 354/5,017 = 0.071 |

- The pre-registered stop rule fired clearly at step 3, so no agent ran. The task mostly measured style.
- Few SAE latents keep their encyclopedia-defined concept on styled text, and an anti-shortcut rule (drop
  latents that fire on their own name) was involved in 84% of the exclusions of style-robust latents.
- The filtered pool fails a menu-fingerprint check (0.616 vs bar 0.60) and can be memorised from labels in
  simulation (not tested under RL). Recommendation: retire the 20-option form.

### 4.6 LatentKnockout (NO-GO as is)

| Item | Value |
|---|---|
| Sweep | 106 cells, 43 (family, group) pairs, layers 6/12/18 |
| Feasible pairs | 30/43 = 0.698 [0.55, 0.81] |
| Naive top-5 attribution reaching half the reference | 0.705 (31/44) [0.56, 0.82] (fires check 2) |
| Strict margins (answer must lose by 0.5 / 1 / 2 logits) | feasible pairs 0.49 / 0.33 / 0.05 |
| Verify variant vs its own recipe gate (<= 0.10) | self-test script 0.76 [0.47, 0.94] |
| Pre-registered predictions | 38 made (36 scored, 2 untested); modal verdict (NO-GO as is, 0.40) was a hit |

- The field's standard recipe (top-k attribution) solves it, so the task does not need interpretation.
- A top-1 "flip" was often only a near-tie (median lead 1.75 logits); under strict margins feasibility
  collapses.
- Research finding, opposite to the prediction: famous concepts (for example Texas, soccer, English)
  resist knockout by 5 latents in a 16k SAE, while less famous ones do not. Untested explanations remain.
- Grading by rerunning the model removed the style problem seen in FeatureMatch (new-style / same-style
  ratio 1.03), with some penalty on fully fresh wordings.

### 4.7 BoolIntermediates and LatentDiff (too easy)

| Task | Opus 5.5 smoke | Predicted | Why |
|---|---|---|---|
| BoolIntermediates | 12/12 [0.76, 1.00], 48/48 slots | 80/65/40/30% by tier | Agents rebuilt the network from weights and solved it offline; median 2 of 80 tool calls |
| LatentDiff | 6/6 [0.61, 1.00], 24/24 slots | 15% / 5% | One per-token spread statistic separates plants (0.0-0.4) from decoys (6-133) |

- Small n, one model, hand-picked instances: these are bug-finding results. Redesigns are specified and held.

### 4.8 Dropped at build stage

| Task | Reason |
|---|---|
| EditFind | ROME edits are not specific: 0/154 edits kept in a specificity rebuild; a layer-4 edit moves other relations by median 7.4 nats |
| T2Ravel | Public layer predicts null with 100% accuracy (reference 16/16 one-shot) |
| SilentPathway | Scripted gates pass (reference 12/12, recipes 0-1/14) but frontier probes solve it in 10-13 turns |

### 4.9 ModelDiff (stopped on pre-registered kill K5)

| G1 validation check | v1 | v2 (24 tests/claim) | Target |
|---|---|---|---|
| Oracle with fresh tests | 0.66 | 0.87 | >= 0.70 |
| False-claim refutation | 0.78 | 0.84 | >= 0.90 |
| AUROC | 0.77 | 0.90 | >= 0.90 |
| Re-grade SD | 0.236 | 0.155 | <= 0.07 |

- The grader failed G1 after its one allowed fix, which triggered kill K5. No agent episode ran.
- An independent audit of 794 grades showed the grader scores wording, not content: correct paraphrases
  scored 0.275 / 0.487, the key's verbatim text 0.743 / 0.723 (v1 / v2).
- Null accuracy was not a skill signal: a family-token artifact separates planted from null (AUROC 0.96).
- **INVALID:** the uptake tables (`DIFFING_REPORT.md` section 3.1) and all baseline scores and gate readings
  (section 3.2) were graded by the failed grader and are not evidence. A "selection, not discovery, is the bottleneck"
  reading (0.52 possible vs 0.25 submitted) is suggestive only.
- Recommendation: CHANGE. Run a grader-only Stage 0b behind a hard gate; DROP if it fails. Spend: $4.60
  OpenAI, $1.36 Anthropic, about 9-10 L4 GPU-hours (estimate).

## 5. Final status and conclusions

| Item | Status |
|---|---|
| Ideation | Complete |
| Harness | Validated as working evaluation infrastructure; not hardened to RL-training grade |
| ShiftHunt v2 | Validated with caveats as an evaluation; not validated as an RL environment |
| ShiftHunt v3, FeatureMatch, LatentKnockout, BoolIntermediates, LatentDiff, EditFind, T2Ravel, SilentPathway | Negative results (killed, dropped or held) |
| ResidualRecall | Held |
| ModelDiff | Negative result (grader invalid) |

Conclusions:

1. **One interpretability evaluation survived.** ShiftHunt v2 resists cheap solvers (0/16 on T1, at most
   7.7% on T2), separates large from small models in two families, and has an exact, reproducible grader.
2. **It is not RL-ready.** Two hacking routes are open, the reward is decided near the bar and by grid
   noise, filling the latent budget earns reward, and the T2 pool is filtered.
3. **Most plausible designs fail, and cheap gates catch them.** Nine of the other designs were
   stopped or held before or at small agent smokes, each with a measured reason. The common failures were a
   scriptable tool algebra, style instead of meaning, the field's standard attribution recipe being
   enough, and tasks that frontier agents solve outright.
4. **Ideation scores predicted feasibility poorly.** Scorer agreement was high (8/369 large
   disagreements), but the guessed pass rates were far off.
5. **Graders need a validation gate before any baseline.** The ModelDiff grader scored wording, not
   content, and about 90 baseline episodes were graded before that was known.
6. **Self-written harness tests are not enough.** An independent adversarial pass found 55 failures in a
   harness whose own tests all passed; OS isolation, not a regex audit, is the reliable defence.

## 6. Limitations and threats to validity

- **Containment.** All Claude test agents ran at Tier B (honor system plus post-hoc audit), not
  OS-isolated. They could see the repo cwd, commit subjects, the account email and shared scratch paths
  where answer-bearing operator files sometimes sat. No agent was seen using them, but only a regex audit
  stood in the way. The isolated launcher was merged on 2026-10-04 and not yet used for a ShiftHunt run.
  OpenAI agents were sandbox-confined but not in a kernel sandbox.
- **Unmatched effort.** OpenAI agents ran at effort "medium" and Opus at Claude Code effort "xhigh", with
  different system prompts, so a family difference cannot be told apart from an effort difference.
- **Small n.** 10 instances per arm. T1 vs T2 for Opus (95% vs 75%) gives p = 0.18. Smokes are n = 6-12.
- **Filtered pool.** The T2 pool was kept by a reference tuned on it; the unfiltered rate is a model
  estimate.
- **Audit false positives and negatives.** Harness-valid-only views differ from graded views (T2 Opus 9/10
  valid-only vs 15/20 graded). Headline rates count false-positive INVALIDs at their graded outcome.
- **Same-family workers.** The workflow's builder, reviewer and skeptic agents came from the same model
  family as the Claude test agents. Agent reasoning was not visible in either family, so intent could not
  be checked.
- **Scripted kills only.** No LLM agent ran on ShiftHunt v3, FeatureMatch v2/v2f, LatentKnockout or
  ModelDiff (apart from two Haiku harness-audit episodes on ModelDiff DEV, not results). Those verdicts ask whether the score can be reached without interpretation, not whether an
  agent would reach it.
- **Single model and SAE family per study.** One concept set and one menu format for FeatureMatch; dense
  comparators (DiffMean, LEACE) were never run for LatentKnockout.
- **Source disagreements**, reported as found:
  - test count at harness READY: 86 (READY commit message) vs 87 (a commit two minutes later and the verification doc);
  - red-team count: 95/95 (merge commit) vs 115/115 (later docs and run files);
  - ShiftHunt regrades: 58/58 (forensics, earlier) vs 80/80 (VALIDATION.md, after the OpenAI arm);
  - GRPO variance: about 25-80% (VALIDATION.md) vs about 10-31% (forensics);
  - ShiftHunt v3 no-name graddiff: 50/51 (builder, earlier) vs 49/51 (audit, later);
  - ModelDiff key-wording score: 0.743 / 0.723 (report) vs 0.73 (STATUS, same commit);
  - Anthropic spend: the committed ledger snapshot sums to $3.8254 for 2026-10-01 only, while a later
    decision record states $2.79 spent. No document gives a grand total for either provider.
- **Not covered.** Wave-1 FreqHunt has no final verdict in the sources, and TriggerHunt was deferred (D9).
- **Code availability.** ShiftHunt v2/v3 and ModelDiff code, pools and raw runs live on local-only
  branches (`task/shifthunt`, `task/shifthunt3`, `task/modeldiff`) that are not in this checkout. Main holds
  the redacted aggregate write-ups. Commit counts come from a shallow clone.

## 7. Future work (in order)

1. **Direct context-exposure control:** re-run Opus on ShiftHunt v2 through the isolated launcher
   `common/claude_agent.py`, at effort matched to the OpenAI arm.
2. **Close ShiftHunt's RL blockers:** OS isolation so answer files cannot be reached; keep sample texts
   within 96 tokens or add the truncation attack to the gates; decide whether filling the 20-latent budget
   should earn reward (state it in TASK.md, or use a smaller k or a per-latent cost).
3. **Reduce grader noise:** a larger hidden grid (4x texts per cell halves the SE), or treat
   |margin| < 2 SE as a tie.
4. **Draw an unfiltered T2 batch** without the reference keep filter.
5. **Run a middle model (Sonnet class) on T1 and T2** to test for a graded difficulty dial; repeat the
   cross-family check at matched effort.
6. **Harden the harness to training grade:** fix the remaining audit false positives and negatives, close
   the three strict-xfail gaps, add a runner clock to the isolated launcher, or replace the regex audit
   with OS isolation for all agents. Update `runs/api_budget/ledger_snapshot.jsonl` to cover the whole
   project.
7. **ModelDiff Stage 0b** (grader only, about a day, one fix, DROP on failure): fixed per-organism test
   banks, free-generation scoring with a cross-family judge, a hard G1 gate before any baseline.
8. **Optional redesigns:** ShiftHunt v4 (open cue grammar, heads trained on data), LatentKnockout-Strict,
   the famous-concepts knockout question, BoolIntermediates activations-only profile, LatentDiff dials,
   and the never-built runners-up (C29, C30, C09, C32).

## 8. Key numbers at a glance

| Number | Value | Source |
|---|---|---|
| Candidate environments scored | 41 (9 criteria, 2 scorers, 369 cells) | `ideation/CANDIDATES.md` sections 1.4, 2 |
| Large scorer disagreements | 8 of 369 cells | `ideation/CANDIDATES.md` sections 1.4, 1.5 |
| Harness code size | 7,711 lines (excluding `common/tests/`) | `common/` (wc -l) |
| Harness tests at last merge | 536 passed, 3 strict xfail, 0 skipped | `docs/LOG.md` 2026-10-04 08:43 UTC; git 52d0cdb |
| Adversarial review failures found and fixed | 23 harness + 32 audit | `docs/HARNESS_VERIFICATION.md` "Result" |
| Largest audit replay | 164 units / 5,049 tool calls, 1 intended change | `docs/LOG.md` 2026-10-04 08:43 UTC |
| Isolation red team | 115/115 contained; 72/114 escape without jail | `runs/_demo/20261004-073743_claude_iso_selftest/redteam/`; `docs/LOG.md` 2026-10-04 08:43 UTC |
| Recorded episodes in infra-failure replay | 8,256 (fixer's count; the replay README found 9,644 unique) | `docs/HARNESS_VERIFICATION.md` section 9; `runs/_harness/audit_replay_2026-10-02/README.md` |
| ShiftHunt T1 gates | reference 16/16; cheap solvers 0/16 | `docs/results/shifthunt/VALIDATION.md` "Gates" |
| ShiftHunt T2 gates | reference 39/39 kept; cheap solvers <= 7.7% | `docs/results/shifthunt/VALIDATION.md` "Gates" |
| ShiftHunt T1 Opus 5.5 | 19/20 = 95% [76, 99] | `docs/results/shifthunt/SCALE1_T1.md` |
| ShiftHunt T2 Opus 5.5 | 15/20 = 75% [53, 89] | `docs/results/shifthunt/SCALE1_T2.md` |
| ShiftHunt T1 / T2 gpt-6.1-sol | 10/10 / 7/10 | `docs/results/shifthunt/SCALE1_OPENAI.md` |
| ShiftHunt T1 Haiku 4.5 / gpt-6-luna | 0/10 / 0/10 | `docs/results/shifthunt/VALIDATION.md`; `SCALE1_OPENAI.md` |
| Large-model cue naming | 144/144 correct; 0/66 null false claims | `docs/results/shifthunt/VALIDATION.md` "How it fails" |
| ShiftHunt regrades | 80/80 agent episodes match | `docs/results/shifthunt/VALIDATION.md` "The grader" |
| ShiftHunt bar sensitivity | T2 Opus 5/20 at tau +0.05; 20/20 at bar 0 | `docs/results/shifthunt/VALIDATION.md` weakness 2 |
| ShiftHunt truncation channel | AUC 0.95 (proxy) | `docs/forensics/SHIFTHUNT_FORENSICS.md` section 4b |
| ShiftHunt OpenAI cost per episode (median) | T2 sol $0.32; T1 sol $0.18; T1 luna $0.029 | `docs/results/shifthunt/SCALE1_OPENAI.md` |
| ShiftHunt v3 kill script | 49/51 = 96.1% without names | `docs/results/shifthunt3/KILLED.md` section 1 |
| FeatureMatch style-robust script | 0.873 of reference | `docs/results/featurematch/VERDICT.md` |
| LatentKnockout naive attribution | q = 0.705 | `docs/results/latentknockout/FEASIBILITY_VERDICT.md` section 0 |
| BoolIntermediates smoke | 12/12 | `docs/results/boolintermediates/SMOKE.md` |
| LatentDiff smoke | 6/6 | `docs/results/latentdiff/SMOKE.md` |
| ModelDiff grader refutation / re-grade SD | 0.84 (target 0.90) / 0.155 (target 0.07) | `docs/results/modeldiff/DIFFING_REPORT.md` section 3.1 |
| ModelDiff paraphrase vs verbatim score | 0.487 vs 0.723 (v2) | `docs/results/modeldiff/DIFFING_REPORT.md` section 3.1 |
| OpenAI spend, ShiftHunt | about $6.5 of $57 stop | `docs/STATUS_2026-10-04.md` section 5 |
| OpenAI spend, ModelDiff | $4.60 of $12 cap | `docs/results/modeldiff/DIFFING_REPORT.md` section 4 |
| Anthropic API budget | hard $17, global stop $16 | `docs/DECISIONS.md` D10 |
| Hardware | 1x NVIDIA L4 (23 GB), 30 GB RAM, 8 vCPU | `docs/LOG.md` 2026-10-01 05:12 UTC |
| Timeline | first commit 2026-10-01 05:13 UTC; last 2026-10-05 04:16 UTC | git log |
| Git commits (all branches, shallow clone) | 163 | git rev-list --all --count |

## 9. Reproducing

- **Entry points:** `README.md`, `docs/INDEX.md`, `CONTEXT.md`.
- **Harness code and tests:** `common/`, `common/tests/` (run with pytest); worked example `tasks/_demo/`.
- **Harness contract and verification:** `docs/HARNESS_API.md`, `docs/HARNESS_VERIFICATION.md`,
  `docs/BUILDER_GUIDE.md`.
- **Decisions and timeline:** `docs/DECISIONS.md`, `docs/LOG.md`, `docs/STATUS_2026-10-04.md`.
- **Ideation:** `ideation/CANDIDATES.md`, `ideation/research_raw.json`, `ideation/scores_raw.json`.
- **Results write-ups:** `docs/results/{shifthunt,shifthunt3,featurematch,latentknockout,boolintermediates,latentdiff,modeldiff}/`
  and `docs/forensics/`.
- **Run records:** `runs/_harness/` (audit replays), `runs/_demo/` (isolation self-tests and plumbing
  runs), `runs/api_budget/` (caps and spend ledgers).
- **Figures:** none are committed; all results are tables in the documents above.
- **Not in this checkout:** task code, instance pools and per-episode run directories for ShiftHunt v2,
  ShiftHunt v3 and ModelDiff live on local-only branches (`task/shifthunt`, `task/shifthunt3`,
  `task/modeldiff`). FeatureMatch and LatentKnockout work is on `diag/featurematch` and
  `task/latentknockout`.
- **Compute:** model-based runs need a GPU (built on one 23 GB NVIDIA L4). Agent runs also need API keys
  and budget caps set in `runs/api_budget/task_caps.json`. The `_demo` task is the GPU-free worked example.
