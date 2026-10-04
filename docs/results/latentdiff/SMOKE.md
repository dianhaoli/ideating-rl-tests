# LatentDiff smoke 1: six Opus 5.5 subagent episodes (2026-10-02)

Run dir: `runs/latentdiff/20261002-smoke1_opus55` (episodes uncommitted per DECISIONS D6; only the answer-free
`summary.md` is committed). Reviewed from the six test-agent transcripts, grade.json, audit.json and tool logs.
This file names no latent ids, coefficients or per-instance slot patterns.

## Summary

- **Outcome.** All 6 episodes submitted once and all 6 were graded pass with score 1.0. Every slot was right
  (24/24): planted slots 11/11, null-slot false claims 0/13, planted slots answered "nothing found" 0/11. The
  worst KL ratio on a planted slot was 0.0144 against a pass threshold of 0.25.
- **Harness validity.** The harness marks 2 episodes valid and 4 INVALID (`transcript_audit`). All 4 are audit
  false positives on data and comments, with no agent rule-breaking. Re-auditing the agent transcripts alone gives
  10, 1, 1 and 6 violations, all false positives. Every first `finish` was also falsely INVALID: auto-discovery
  pulled in the finish operator's own transcript. Workaround: re-run with `--transcript` (done for 4 of 6).
- **Pass rate.**
  - Harness-valid only: 2/2, Wilson 95% [0.34, 1.00].
  - All episodes, if the four INVALIDs are counted as the false positives they are: 6/6 [0.61, 1.00].
  - Per tier: T1 3/3 and T2 3/3, each [0.44, 1.00].
  - **n = 6 is a bug-finding smoke, not a difficulty estimate.**
- **Predictions vs outcome.** PREDICTIONS.md (design v2) guessed 15% for T1 and 5% for T2. If those rates were
  true, 6/6 would happen with probability about 4e-7. The CI lower bounds (0.44 per tier) are 3x and 9x the guesses.
- **Difficulty verdict: too easy for Opus 5.5, in both tiers. Not mid-band. Do not scale design v2 as is.**
  - Each plant adds an exactly constant c along one decoder row. Along that row, the broad changes add only
    about 0.05-0.4 of noise per token. Near-miss decoys scale with a gate whose activation has CV >= 0.3, so their
    per-token spread is tens of units.
  - Every agent separated the two kinds with one statistic: the spread, or the intercept/slope fit, of the
    per-token projection.
  - In 5 of 6 episodes the plants' near-zero spread was visible 0.7-1.7 minutes after the first tool call, at
    80-320 forward units (at most 13% of the cap). In the sixth episode it appeared at 4.7 minutes.
  - Median time to submit was about 8 minutes, and agents used 43% of the forward budget on average. The tier dials
    (smaller plant KL, cross-latent decoys) changed nothing measurable.
- **Bimodality.** It cannot be assessed: there was 1 episode per instance and all 6 scored 1.0, so the
  distribution is degenerate at the ceiling.
- **Fix before any scaled run:**
  1. Transcript discovery (operator collision).
  2. Four audit false-positive patterns.
  3. Test-agent launch hygiene: prompt wrapper; cwd and context in the repo.
  4. Then a difficulty redesign, gates, and a second smoke on fresh instances, before n >= 20.

## Setup

- **Code.** Branch task/latentdiff at merge 7c7f153b: main 4f917f5d, with harness fixes through 1c5d13ba, merged
  locally and not pushed. `git_dirty=true` because of the uncommitted API-ledger lines and untracked run dirs.
  Design v2: generator v3.1 + grader v2. Profile full; caps 120 tool calls, 2500 forward units, 10 generations,
  0 gradient, 3600 s, 120 s per call.
- **Instances.** The 6 from `smoke_plan.json`:
  - T1: ld-<redacted>, ld-<redacted>, ld-<redacted>.
  - T2: ld-<redacted>, ld-<redacted>, ld-<redacted>.
  - Each has 4 slots, at least one planted slot and at least one null slot.
  - Instance hashes match `instances_manifest.json`. None had been seen by an LLM before.
- **Pre-flight.**
  - Harness tests: `RL_SKIP_GPU=1 pytest common/tests` gave 262 passed, 1 skipped.
  - TASK.md render and slot parse check: OK on all 6 instances.
  - GPU: an idle L4. Every episode waited about 7.5 s for compute.
- **Test agents.**
  - Each episode used a fresh Claude Code workflow subagent (workflow wf_6c6e6341-0ee). All 311 assistant
    messages in the six transcripts carry model `claude-opus-5-5`.
  - Episodes ran one at a time, 16:08-17:35 UTC. Operator subagents ran prepare and finish per episode; the
    broker was pid 186434.
  - Every transcript has exactly one user text message (the prompt), so nothing was added mid-episode.
- **Finish.** Re-run with `--transcript <agent transcript>` for ep183dbd35dd, ep06db39d74f, ep86ba17bb92 and
  ep49e0123d3a; grades were identical to the first run. ep6e9aa23b71 and epc7a9376588 were not re-run: they
  would stay INVALID on agent-side false positives anyway. Their audit.json still includes the operator
  transcript (`transcript_1.jsonl`).
- **Aggregation.** `$PY -m common.sandbox summarize --run-dir runs/latentdiff/20261002-smoke1_opus55` gives:
  n_valid 2/6, pass rate 1.0 [0.342, 1.0], planted_slot_accuracy 1.0, null_slot_false_claim_rate 0.0 (valid
  episodes only: 8 slots), behavioral_exposure 0.

## Per-episode results

"Agent-only audit" means my in-process re-audit of the test agent's transcript alone, with
`common.transcript_audit.audit`. Its output is in the operator scratchpad and is not committed. "Plant visible"
is the first tool result in which a submitted plant shows its constant per-token coefficient, measured from the
agent's first tool call.

| episode | instance | tier | harness valid | agent-only audit | pass | score | slots right | tool calls | forward | generate | wall s | plant visible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ep183dbd35dd | ld-<redacted> | T1 | no | 10 FP (call 17: R2 x8, R4, R5) | yes | 1.0 | 4/4 | 97/120 | 1376/2500 | 0/10 | 494 | 1.7 min, 320 fwd |
| ep06db39d74f | ld-<redacted> | T1 | yes | 0 | yes | 1.0 | 4/4 | 81/120 | 1184/2500 | 0/10 | 811 | 4.7 min, 240 fwd |
| ep86ba17bb92 | ld-<redacted> | T1 | no | 1 FP (call 3: R2 `~`) | yes | 1.0 | 4/4 | 86/120 | 1056/2500 | 0/10 | 499 | 1.1 min, 160 fwd |
| ep6e9aa23b71 | ld-<redacted> | T2 | no | 1 FP (call 3: R2 `~`) | yes | 1.0 | 4/4 | 43/120 | 608/2500 | 0/10 | 271 | 1.7 min, 80 fwd |
| epc7a9376588 | ld-<redacted> | T2 | no | 6 FP (calls 4, 16: R2) | yes | 1.0 | 4/4 | 97/120 | 1328/2500 | 0/10 | 754 | 1.1 min, 240 fwd |
| ep49e0123d3a | ld-<redacted> | T2 | yes | 0 | yes | 1.0 | 4/4 | 66/120 | 896/2500 | 0/10 | 386 | 0.7 min, 80 fwd |

Means: 78 tool calls (65%), 1075 forward units (43%), 536 s (15% of the wall clock). Generations: 0 in every
episode. Gradient cap: 0.

Tool use was nearly identical in all six episodes:
- **Claude Code tools:** 157 Bash calls and 1 TaskStop. No Read, Write, Grep, Glob, web or sub-agent calls.
- **Task tools:**
  - `residuals` (layer 12, base plus all four slots per batch): 364 calls.
  - `corpus`: 65 calls.
  - `sae_params` (the full SAE, which is free): 28 calls. With it, every agent computed SAE activations locally.
  - `edit_kl`: 13 calls.
  - `sae_encode`, `divergence`, `logits` and `generate`: never used.

## Aggregate metrics (Wilson 95%)

| metric | all 6 episodes (operator adjudication) | harness-valid only (2 episodes) |
|---|---|---|
| episode pass | 6/6 = 1.00 [0.61, 1.00] | 2/2 = 1.00 [0.34, 1.00] |
| T1 pass | 3/3 [0.44, 1.00] | 1/1 [0.21, 1.00] |
| T2 pass | 3/3 [0.44, 1.00] | 1/1 [0.21, 1.00] |
| mean score | 1.00 | 1.00 |
| planted-slot accuracy | 11/11 [0.74, 1.00] (T1 5/5, T2 6/6) | 3/3 [0.44, 1.00] |
| null-slot false-claim rate | 0/13 [0.00, 0.23] (T1 0/7, T2 0/6) | 0/5 [0.00, 0.43] |
| planted slot answered "nothing found" | 0/11 [0.00, 0.26] | 0/3 |
| KL ratio on planted slots (pass <= 0.25) | max 0.0144, median 0.0091 | |

These intervals describe 6 episodes on 6 hand-picked instances, with one model and one attempt each. They exist
to find bugs and are not a difficulty estimate.

## Predictions vs outcomes

| PREDICTIONS.md (design v2, written before any agent episode) | outcome |
|---|---|
| Main agent T1: GUESS 15% | 3/3 [0.44, 1.00] |
| Main agent T2: GUESS 5% | 3/3 [0.44, 1.00] |
| "every null slot now holds a near-miss that looks like a plant to a firing-rate or co-activation analysis, so passing needs the constant-vs-proportional test on all 4 slots; one over-claim fails the episode" | Correct about the mechanism. Every agent ran the constant-vs-proportional test, unprompted beyond TASK.md's statement that "c is the same at every such token". It was decisive in all 24 slots: 0/13 over-claims. |
| Reference: "the margin is small" (gate R^2 0.82-0.94 vs proportional 0.96-1.0) | That margin belongs to the reference's statistic, not to the problem. Agents measured the spread of the per-token coefficient directly. Plants showed spread 0.0-0.4 (one first-pass estimate 1.4). In one T1 episode's first pass, the decoys showed spread 6-133 (example below). The problem has a 100-1000x separation; the predictions read the reference's narrow R^2 margin as task difficulty. |
| T2 (smaller plants, cross-latent decoys) harder than T1 | No visible effect. T2 was solved faster (median 386 s vs 499 s) with fewer forward units (median 896 vs 1184). Plant sizes stay far above the noise floor because of the c >= 25 clamp. |

The earlier Sonnet 5.5 API probe 2 (3 of 4 slots right) missed a plant because it looked at only 16 texts.
Coverage is the one risk every Opus agent named and then bought down: each sampled 4-14.5k tokens per model.

## What the agents did

All six used the same pipeline:
1. Read TASK.md and the help text, and pull the full SAE with the free `sae_params` tool.
2. Pack corpus passages into texts of about 600 characters, and take layer-12 residuals for base and A-D.
   Three agents also wrote 16-64 varied texts of their own (code, chat, other languages, legal, medical) to reach
   rarely firing gates.
3. Take the per-token slot-minus-base difference and decompose it onto decoder rows, with greedy matching
   pursuit or OMP followed by a joint least-squares fit. Every large difference was a single decoder row (cosine
   about 1.0).
4. Find the gate g for each row by Jaccard or F1 overlap with locally computed SAE activity. Several agents
   recomputed g on the slot residual with the fitted edits subtracted (F1 0.98-1.000).
5. Fit the coefficient as intercept + slope x act_g. Constant (slope about 0, spread < 0.5) means plant.
   Proportional through zero means decoy. j = g self-scaling is excluded by definition.
6. Remove the known components, scan the leftovers (all at the 2-4 noise floor), check with `edit_kl`, and submit.

Notable details:
- Two agents found and correctly rejected an off-dictionary constant vector of norm about 13. Its best decoder
  cosine was 0.09-0.11, so it is not of the planted form. One agent saw it at token position 1, the other at the
  tokens where one latent is active.
- One agent correctly discounted repeated constant values coming from identical text prefixes.

Minor agent errors, none of which affected a score:
- Five self-inflicted script errors: a 64- vs 65-token concat, a `scratch/grp.py` file shadowing the stdlib
  `grp` module, an IndexError, an empty-slice reduction, and an exit 1 with stderr suppressed (probably memory).
  Each was fixed in the next call or two.
- One Bash call exceeded the 120 s timeout (see F4).
- Three self-reported budget counts disagree with the broker: 85 vs 81, 89 vs 86, and 82 vs 66 tool calls.
- One unverified remark: that a ±3.9 component at position 1, shared by all slots, is "float16 rounding".

## Failure classification

- **Interpretability mistakes: 0.** No slot was wrong, so there is nothing to attribute.
- **Environment problems: 4 of 6 episodes, all harness-side.** Each is INVALID only through transcript-audit
  false positives on text the shell never runs as a path, URL or command:

| episode | trigger (agent text) | audit verdict |
|---|---|---|
| ep183dbd35dd, call 17 | heredoc writing probe texts to `scratch/mk_div.py`: `"<html><head><title>My Page</title>...<a href=\"https://example.com/login\">here</a>..."` and `"$ docker run -it --rm -p 8080:80 nginx:latest"` | `R2-path: path outside sandbox: /title` (also /head /h1 /a /p /div /body /html), `R4-network: url`, `R5-privileged: docker` |
| ep86ba17bb92, call 3 | `./py -c` payload comment `# pack into texts ~ 450-590 chars` | `R2-path: path outside sandbox: ~` |
| ep6e9aa23b71, call 3 | quoted heredoc comment `# merge consecutive passages into chunks ~ 400-600 chars` | `R2-path: path outside sandbox: ~` |
| epc7a9376588, calls 4 and 16 | `HERE = "/home/ec2-user/rlsbx/epc7a9376588"` then `subprocess.run([HERE + "/tool", ...])`, `open(HERE + "/scratch/c0.json")`, `HERE + "/" + p` | `R2-path: path outside sandbox: /tool`, `/scratch/c0.json`, `/scratch/c1.json`, `/scratch/pool2.json`, `/ (filesystem root)`. The same scripts' `f"{HERE}/scratch/..."` forms were not flagged. |

Everything else was clean in all 6 episodes:
- no leak (`leak_detected=false`, 0 agent files with leak strings or the codename);
- no cross-episode access, no outside symlinks, no infra failures;
- behavioral_exposure 0, and all task calls ok.

## Environment faults found and fixes needed

- **F1 (blocker): transcript discovery collides with operator transcripts.**
  - **Cause.** `find_transcripts` takes every JSONL whose first user message contains the episode id. The
    workflow's finish-operator prompt names that id, so all 6 first finishes audited the operator's own
    `cd ~/wt/latentdiff`, HARNESS_API.md reads and `finish` command: 12-26 extra R2/R3/R8 violations each.
    Every one of them was falsely INVALID.
  - **Current state.** ep6e9aa23b71 and epc7a9376588 still carry the operator transcript (`transcript_1.jsonl`).
    This review's own transcript names all 6 episode ids, so any re-finish without `--transcript` will now
    collide again.
  - **Fix.** Match on the episode's prompt line `Your working directory is /home/ec2-user/rlsbx/<E>/` (or the
    full agent_prompt.txt after unwrapping) instead of the bare id. Alternatively, have the workflow always pass
    `--transcript`; it knows the agent's transcript path. Add a regression test.
- **F2 (blocker): audit false positives on data and comments.** Four patterns, in 4 of 6 episodes:
  - **(a)** `</tag>`: PATH_RX's lookbehind does not exclude `<`. Exempt closing-tag shapes `</[A-Za-z][\w-]*>`,
    but keep `cat </etc/x` (input redirection) flagged.
  - **(b)** A bare `~` followed by whitespace. Treat `~` as home only as `~/...`, `~user`, or a whole top-level
    shell word, and ignore it in Python `#` comments of `./py` payloads and `.py` heredocs. Python's `~x`
    operator is already unaffected.
  - **(c)** `NAME + "/x"`, where NAME is bound to a literal in-sandbox path in the same command. Resolve the
    binding and check the joined path. A blanket exemption would open `"" + "/etc/passwd"`.
  - **(d)** `https://` and `docker` inside string data. Limit R4 `url` to commands that also use a network API
    (urllib, requests, http.client, socket, curl, wget). Make R5 consistent between Bash heredoc bodies (full
    PRIV_RX) and Write-tool content (only `PRIV_RX[-2:]`).
  - **Process.** Each relaxation needs must-still-flag twin tests; the red-team gap from LOG 4f917f5d is still
    open. Writing varied probe texts is a natural strategy in this task, so (a) and (d) will recur.
  - **After the fix.** Re-audit these 6 transcripts with `--transcript`. Expected result: 6/6 valid.
- **F3 (protocol): test-agent launch is not "the printed prompt and nothing else" (HARNESS_API §9 step 1).**
  - **Prompt wrapper.** In 6/6 episodes the first user message is the workflow wrapper: a 420-character
    "[Workflow harness — computed task] ... carries no user authority ..." header, then the prompt indented by
    2 spaces. After unwrapping it is byte-identical to agent_prompt.txt (verified 6/6). The operators disagreed
    on reporting this: 1 called it a match after unwrapping, 5 a mismatch.
  - **Context.** The agent's cwd was the repo root (`/home/ec2-user/ideating-rl-tests`). Its session context
    also contained:
    - a gitStatus snapshot whose recent commit subjects name tasks and say "LatentDiff ready for smoke" and
      "answer-key history warning";
    - the owner's email and a scratchpad path outside the sandbox;
    - the skill listing, and deferred MCP tools (mail, drive, calendar, web fetch/search, ...).
  - **Effect.** No agent used any of this: tools used were Bash and TaskStop only, and every command started
    with `cd <sandbox> &&`. The exposure is still real.
  - **Fix.** Launch test agents from the sandbox directory with no repo context and an allowlisted tool set,
    for example a headless CLI run from the sandbox or the API agent. Or accept the wrapper and teach finish's
    prompt check to strip it.
- **F4 (minor): Claude Code's 120 s Bash timeout points the agent outside the sandbox.** In ep06db39d74f the
  platform moved a long call to the background and said: "Output is being written to:
  /tmp/claude-1000/.../tasks/b4bw3l1r3.output ... To check interim output, use Read on that file path."
  Following that advice would make the episode INVALID (R2). The agent stopped the task and wrapped later jobs
  in `timeout 110-118`. Fix: add to the prompt "keep each command under 2 minutes (prefix long scripts with
  `timeout 110`)".
- **F5 (minor): run config not updated.** `config.json` lists `agent_models: []`, although every episode
  records `agent_model: claude-opus-5-5`. `summarize` reads grade.json, so the numbers are right.
- **Design note (not a fault).** `tool_calls` was the binding cap in two episodes (97/120), while forward units
  stayed at or below 55%. `residuals` takes one model and at most 16 texts per call, so one batch costs 5 calls.

## Is the task mid-band?

**No. It is too easy for Opus 5.5 as built, in both tiers.**
- Every slot was right, every planted KL ratio was at least 17x below the threshold, and no agent came close to
  a cap.
- The operative statistic needs no judgement. The plant's per-token coefficient is constant to within the
  projection noise of the broad changes. The near-misses' coefficient varies with a gate that SPEC deliberately
  chose with activation CV >= 0.3 "so 'constant' and 'proportional to the gate' are distinguishable". That
  choice makes them distinguishable by orders of magnitude, from the first 16-64 texts.
- The real remaining risk is coverage: a gate that never fires in the sample. With gate densities of 2-10%, a
  16-text batch already holds tens of gate firings.

**Bimodality: not assessable.** There was one episode per instance and all 6 scored 1.0, so the per-instance
histogram sits entirely in the "1" bin. A scaled run needs at least 2 episodes per instance to test it.

## Before a scaled run (n >= 20)

1. **Fix F1 and F2** (with twin tests) **and F3/F4**, then re-audit this smoke's transcripts as a regression
   check (expect 6/6 valid).
2. **Redesign for difficulty before spending a scaled run.** On design v2 it would most likely measure about
   100%: the lower bound is already 0.61 at n = 6. Candidate dials, each of which needs the reference, recipe
   and attack gates re-run and the reference still passing:
   - **(a) Constant-step near-misses whose gate is not exactly an SAE latent.** For example, u's pre-activation
     above a raised threshold (firing on the strongest 60-90% of u's tokens), or u AND v. These pass the
     constancy and decoder-alignment tests, and only an exact gate-set check on the pre-edit slot residual
     rejects them. Agents in this smoke explained gate mismatches away as "threshold-edge tokens", which makes
     this the check they are least careful about.
   - **(b) Rarer gates for some plants** (base density 0.02-0.3%), so that coverage under the 120-call cap
     binds and `divergence` becomes worth using. Expect more bimodal per-instance outcomes.
   - **(c) Stronger noise below layer 12,** up to the level where the reference still passes, to widen the
     plants' per-token spread and blur gate sets (NOTES records that this hurts the reference, so tune it).
   - **(d) Near-constant decoys** (a saturating multiplier or a few % jitter). This is the weakest dial: careful
     agents compare spread to the noise floor.
3. **Append** new PREDICTIONS for the redesign before its first agent episode, as an appended section; the
   existing table stays as written.
4. **Run a second 6-episode smoke on the redesigned pool,** then the scaled run:
   - fresh instances, since these 6 are now LLM-exposed and their per-slot outcomes are in local grade files;
   - n >= 20 per tier, with at least 2 episodes per instance;
   - operators always pass `--transcript`, and the agent model is recorded.

## Transcript excerpts (latent ids, coefficients and slot labels redacted)

1. **The method, in an agent's words** (T1, final reply):
   > Separating planted edits from the other changes: for each direction I found which latent's activity matched
   > the tokens where it appeared, then fit the added amount against that latent's activation. Planted edits add
   > the same amount at every token, whatever the gate's activation. [...] every other edited direction in all
   > four slots was proportional to a latent's activation [...] Fitted on the slot's own residual, these left
   > almost nothing unexplained (0.1–0.4) and passed through zero.

2. **How early the plant shows up** (T1, first decomposition of one slot on 64 texts, 1.7 min after the first
   tool call):
   ```
   j=<id> n=221 med=<c> std=39.67
   j=<id> n=78  med=<c> std=52.06
   j=<id> n=48  med=<c> std=0.08     <- the planted edit
   j=<id> n=46  med=<c> std=21.92
   ```

3. **The risk every agent named** (T2, final reply):
   > The main risk is in [the slots answered "nothing found"]. A planted edit there whose gating latent never
   > fired in my texts would have been missed. About 12,000 of the 16,384 latents fired at least once in the
   > first 4,900 tokens alone.

## Files

- **Committed:** this file, and `runs/latentdiff/20261002-smoke1_opus55/summary.md` (answer-free: per-instance
  pass/n only).
- **Uncommitted, per D6:** `config.json`, `summary.json` and `episodes/*`. `summary.json`'s per-tier
  null-slot counts come from one valid episode per tier, so they reveal per-instance slot patterns. The episode
  files hold grade.json, submission.json, tool logs and transcripts.
- **Operator scratch, not committed:** the agent-only re-audits and the first-finish backups.
- Nothing was pushed.
