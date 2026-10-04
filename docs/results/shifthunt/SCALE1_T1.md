# ShiftHunt scaled run T1: 20 Opus 5.5 episodes on 10 fresh instances (2026-10-03)

Run dir: `runs/shifthunt/20261003-scale1_T1_opus55` (episodes 23:40-02:16 UTC, two at a time). Plan:
`scale_plan_T1.json` (10 of the 16 unexposed T1b_kept instances, 2 episodes each). Predictions: PREDICTIONS.md
"Scaled run T1 (design v3 + D1)", committed before any episode. Reviewed from all 20 test-agent transcripts, grade.json,
audit.json, tool logs, submissions and the 20 finish-operator reports.

Per D6 and the exposed_instances.json rule this file gives aggregates only. It names no instance, no episode, no latent
id, and does not say which attribute belongs to which slot. Quotes are redacted the same way. The episode files stay
on disk and uncommitted (runs/shifthunt/.gitignore).

## Summary

- **Outcome.** 19 of 20 episodes passed (every slot right); 69 of 70 slots were right; mean score 0.988.
  - Planted slots: 44/44 named the right attribute, 43/44 passed. The one miss removed 0.004 less than its bar.
  - Null slots: 0/26 false claims. Planted slots answered "nothing found": 0/44.
- **Harness validity.** 14 valid, 6 INVALID (`transcript_audit`). All 6 are audit false positives, from 4 patterns
  that the smoke F2 fixes do not cover (E1). No agent broke a rule. The one failed episode is one of the 6.
- **Pass rate (Wilson 95%).** All 20, with the false-positive INVALIDs counted as their graded outcome: **19/20 = 95%
  [76%, 99%]**. Harness-valid only: 14/14 = 100% [78%, 100%].
- **Per instance.** 9 of 10 instances passed both episodes; 1 split (one pass, one fail); 0 failed both.
- **Removal.** Agents match the references: agent / label-aware median 1.006, agent / label-free median 1.001. Two
  agents on the same planted slot end a median 0.004 apart in removal. Median margin over the bar +0.072; 13 of 44
  named slots end within 0.05 of it.
- **Predictions.** Pass rate predicted 65% (45-85%), observed 95%: above the plausible range
  (P(>= 19/20 | 0.65) = 0.002). Null false claims (8% predicted, 0/26 seen) and planted accuracy (90% predicted,
  43/44 seen) were both too pessimistic. The margin and reference-ratio predictions held.
- **Verdict.** T1 is not mid-band for Opus 5.5: it is near saturation (CI lower bound 76% > 70%). The environment
  itself is clean: no environment fault changed a grade, recipes stay at 0/16, and the work that passes is real
  interpretability. But with 9 of 10 instances at 2/2, T1 gives GRPO almost no reward variance for this model.

## Setup and provenance

- **Code.** config.json: git_sha f6a7227d, git_dirty true (the concurrent T2 work: reference_solver.py,
  gate_aggregate.py, audit/t2sel_*, later committed as b7d7e89a). The last 2 episodes were prepared after that commit
  and record b7d7e89a. `git diff f6a7227d HEAD` touches none of tools.py, grader.py, sh_core.py, sh_probe.py,
  agent_prompt.md, taus.json or common/, and the working tree is clean on all of them, so every episode ran the same
  task code.
- **Prompt.** Template sha 1bdd2846 in all 20 (config.json and every grade.json). TASK.md is byte-identical across the
  20 episodes once digits are masked (the probe count differs), and agent_prompt.txt once the episode id is
  masked. No unfilled `{public.*}` placeholder (F7 holds). The D1 paragraph "What "relies" means in numbers" is
  present.
- **Test agents.** One fresh Claude Code workflow subagent per episode (wf_96d109e5-27e). 1,447 assistant messages, all
  `claude-opus-5-5`; each transcript has exactly one user text message (the wrapped prompt). 782 Claude Code tool
  calls: 781 Bash, 1 Write (to the agent's own scratch file).
- **Finish.** All 20 finishes found exactly one transcript by search (prompt_match `workflow_wrapper`), the test
  agent's own; no operator transcript was picked up (F1 holds). config.json `agent_models` = [claude-opus-5-5] (F4
  holds).
- **Caps.** 150 tool calls, 3000 forward units, 4 probe_query units, 0 generate, 0 gradient, 3600 s, 180 s per call;
  k = 20; profile full; tier T1 (open SAE access).
- **Pool.** T1b_kept: reference v3.1 one-shot 16/16; every recipe, the black-box control and every attack, the
  disclosed-scale variants included, 0/16 (gates_v8). The plan draws 22 planted and 13 null slots (each played twice:
  44 planted and 26 null slot-episodes; predicted about 46 and 24). No episode is all-null; 4 are all-planted.

## Aggregate metrics (Wilson 95%)

| metric | all 20 (false-positive INVALIDs counted as graded) | harness-valid only (14) | prediction |
|---|---|---|---|
| episode pass | **19/20 = 0.95 [0.76, 0.99]** | 14/14 = 1.00 [0.78, 1.00] | 0.65 (0.45-0.85) |
| episode pass, 1st / 2nd episode per instance | 10/10, 9/10 | - | - |
| mean score | 0.988 | 1.000 | - |
| slots right | 69/70 = 0.99 [0.92, 1.00] | 49/49 [0.93, 1.00] | - |
| planted accuracy (name + removal + topic kept) | 43/44 = 0.98 [0.88, 1.00] | 31/31 [0.89, 1.00] | 0.90 (0.78-0.97) |
| planted slots named right | 44/44 [0.92, 1.00] | 31/31 [0.89, 1.00] | - |
| null false-claim rate | 0/26 [0.00, 0.13] | 0/18 [0.00, 0.18] | 0.08 (0.02-0.20) |
| planted "nothing found" rate | 0/44 [0.00, 0.08] | 0/31 [0.00, 0.11] | 0.04 (0.00-0.12) |
| instances split (one pass, one fail) | 1/10 | 0/4 instances with 2 valid episodes | 0.35 (0.10-0.60) |

`$PY -m common.sandbox summarize` (run on a scratch copy of the run dir, so the run dir was not written) gives the
same harness-valid numbers: n_valid 14/20, pass 14, Wilson [0.785, 1.0], planted_slot_accuracy 1.0,
null_slot_false_claim_rate 0.0, behavioral_exposure 0.

### Removal margins and reference ratios (D2)

`$PY -m tasks.shifthunt.margins --run-dir runs/shifthunt/20261003-scale1_T1_opus55 --ref-run
runs/shifthunt/20261002-223753_gates_v8_P2_T1b --out <scratchpad>` for the harness-valid episodes. margins.py skips
INVALID episodes, so the all-20 column uses the same formulas through margins.q and margins.ref_removals in a scratch
script. Label-aware = ref_removal_k at k = 20 (tau is set from it); label-free = reference v3.1, attempt 0 of gates_v8
T1b, which named every one of these slots right.

| over planted slots named right | all 20 (n = 44) | harness-valid (n = 31) | prediction |
|---|---|---|---|
| removal margin (removed - tau): median [q25, q75] | **+0.072** [+0.047, +0.136] | +0.072 [+0.051, +0.135] | +0.06 (+0.04 to +0.10) |
| margin min / max | -0.004 / +0.342 | +0.029 / +0.342 | - |
| within 0.05 of the bar | 13/44 = 30% | 8/31 = 26% | ~20% (5-40%) |
| below the bar | 1 | 0 | at most 1 |
| agent / label-aware: median (min-max) | 1.006 (0.83-1.13) | 1.012 (0.89-1.09) | ~0.99 (0.93-1.03) |
| agent / label-free v3.1: median (min-max) | 1.001 (0.84-1.06) | 1.003 (0.88-1.05) | - |
| tau / label-aware: median (min-max) | 0.80 (0.51-0.98) | 0.79 (0.51-0.98) | - |
| topic kept: median (min) | 0.993 (0.864) | 0.993 (0.864) | floor 0.8 |

- **T1 removal is close to deterministic.** On the 22 planted slots, the two agents' removals differ by a median 0.004
  (mean 0.011, max 0.063; 16 of 22 within 0.01). Their 20-latent sets overlap with Jaccard median 0.82 (q25 0.67,
  min 0.43). Agents choose latents as well as the label-aware reference does.
- **Where the bar sits still matters.** 30% of named slots end within 0.05 of the bar, because on some families the
  bar is close to what any 20 latents can remove (tau / label-aware up to 0.98). The 0.8 topic floor never came
  close to binding (lowest 0.864).
- Accuracy-based SHIFT (reported, not gated): median 0.12, range -0.14 to 0.57.

## Per-instance pattern and GRPO reward variance

| pattern over the 2 episodes | instances |
|---|---|
| both pass | 9 |
| split | 1 (scores 1.0 and 0.75) |
| both fail | 0 |

Valid-only view: 4 instances have 2 harness-valid episodes (all pass both); 6 have 1 (all pass).

**What this means for GRPO.**
- **Binary pass reward.** A group of rollouts on one instance carries signal only if its rewards differ. Here 9 of
  10 two-rollout groups have zero advantage. With a per-instance pass probability near 0.95 (19/20), a group has any
  variance with probability 1 - 0.95^G - 0.05^G: 9.5% at G = 2, 18% at G = 4, 34% at G = 8, 56% at G = 16. The
  slot-mean score is the same, since 69 of 70 slots are right.
- **Continuous removal credit would not rescue it.** The removal margin does vary across slots (IQR +0.047 to
  +0.136), but within an instance the two agents' removals differ by a median 0.004. That spread is between
  instances, which GRPO's group baseline removes.
- **The split share is low because of saturation.** Independent episodes at p = 0.95 split 9.5% of the time, and 1 of
  10 was observed. So this is not evidence that outcomes are fixed per instance (PREDICTIONS.md's "below 10%"
  reading). The only judgement that differed between two agents on the same instance is the one that failed (below).

## Failure classification

**One failed slot (planted, named right, removal 0.004 below the bar). Cause: an interpretability judgement error by
the agent. Not an environment fault.**
- The agent's first latent ranking (by contribution to the attribute effect, as in every T1 episode) included a
  latent that carries the second-largest share of the attribute effect but also some topic signal. It estimated that
  latent's topic cost on the SAE-reconstructed logit: about 0.19-0.30 of the topic effect pooled over text batches,
  up to 0.36 on one batch. It judged that too close to the 80% topic-kept rule and dropped the latent, cutting its SAE-level
  removal from about 0.59 to 0.50.
- It had already measured, with probe_query, that the real topic effect is 1.65-1.73x the SAE-level one: the SAE error
  term carries topic and is never ablated. It used that factor to scale its expected *removal* down to "about
  30-40%", but did not apply it to the *topic loss*. Scaled, the topic loss would have been about 0.11-0.18, well
  inside the rule.
- Counterfactual grading (grader.grade in memory; nothing written): the agent's first set removes 0.389 and keeps
  0.859 of the topic, so the slot passes and the episode scores 1.0. Its intermediate alternative removes 0.381 and
  keeps 0.895: also a pass. The submitted set kept 0.953, a safety margin on a rule that was not binding.
- The other agent on this instance faced the same choice, kept that latent, offset its topic cost with three latents
  that push the other way, and passed (+0.046).
- **Contributing design factor (D2):** tau for this slot is 0.84 of the label-aware reference's removal, so the bar
  leaves little room. But the bar was reachable: the label-free reference clears it by +0.059, and so did the agent's
  own first choice.

> I left out latent [...] on probe [...]. It would have raised removal to about 59%. But it also carries topic
> signal, and on one batch of texts its topic loss came to about 20%, right at the 80% topic-retention limit.

And from the other episode on the same instance:

> I kept latent [...] even though it costs some topic signal. Removing it would have dropped the removal from about
> 57% to about 47% of the SAE part, so I offset it with three latents that push the other way.

**Close calls that went right (no failures, but where the difficulty is).**
- **Null calls in the D1 gap.** By the agents' own reports, in 13 of the 16 episodes with a null slot the first
  estimate for some listed attribute on a null probe came out at or above the 0.2x cap: usually 0.2-0.4x, up to about
  0.6x on generated texts and 0.67x in one real-logit query on concatenated notes. All 26 null slot-episodes still
  ended "none". Agents checked with controls matched to the sample (tense, length), larger dedicated pair sets, or
  real-logit queries that measure the SAE error term.
  > every listed attribute was at most 0.10 times the topic except [attribute], at about 0.38. I spent all 4 real
  > probe queries on a topic × [attribute] test with otherwise identical texts. The real [attribute] effect was only
  > 0.03 times the topic effect, so the error term cancels it.

  > every sample pairs [the attribute's wording] with [an unlisted feature] [...]. When I made my generated
  > [attribute wording] match [the unlisted feature], the [attribute] effect on [the two null probes] fell to
  > almost nothing (0.03 of the topic effect).
- **D1 caveat (design, not a fault).** The disclosed caps hold on the generator's unconfounded texts. Agents measure on
  their own texts, which are off-distribution (shorter, concatenated, re-templated), and there a null's listed
  dependence can read above the cap. Opus resolved every such case. A weaker agent may not, and this is the smoke-1
  failure mode with the scale disclosed.
- **Removal near the bar.** 13 of 44 named slots ended within 0.05. In every episode the agents predicted the real
  removal from the SAE-level number and the measured error-term share, and their predictions were within a few
  hundredths of the grader on most slots:
  > The true [attribute] effect was 3.93, against 2.23 from the SAE part. The SAE error term, which ablation cannot
  > touch, carries about 40% of this effect.
- **D1's predicted opposite error did not occur.** PREDICTIONS.md warned that an agent might compare an attribute's
  effect with the sample's class gap instead of the topic effect, and answer "none" on a planted probe. 0 of 44
  planted slots were answered "none", and every agent compared against the topic effect. One agent used the class
  gap only to argue that an effect was too large to be a side effect, which is the correct direction. About half the
  agents restate the disclosed 0.4x floor or 0.2x / 0.25x cap in their own messages.

**Environment-only failures: 0 slots.** The 6 INVALIDs are audit false positives that do not touch any grade (E1).

## What the agents did

All 20 took the T1 route that SPEC section (9) intends; none found another one.
1. Read TASK.md, task_info, samples and gradients. Every agent saw that the sample confounds 5 listed candidates and
   2 unlisted decoys with the topic for every probe.
2. Wrote a generator that clones the sample template and encoded matched texts through `latent_means`: pairs that flip
   one attribute or only the topic, factorial designs, edited sample texts (roughly 500-1,300 texts per episode).
3. Used the probe's exact linearity (SAE part of the logit = g · mean latents). They checked it against
   `probe_scores` ablation (r 0.99-1.0, slope 1.00), and saw that real logits run 1.2-2.4x the SAE part (the error
   term).
4. Compared each attribute's SAE-level effect with the topic effect, set aside the decoys as unlisted, and called nulls
   on the D1 scale.
5. Spent the 4 probe_query units on real-logit checks. In 12 episodes all 4 went to planted probes, mostly to measure
   the error-term share; in 6 all 4 went to a null call; 2 split them 2/2.
6. Picked 20 latents greedily by contribution to the attribute effect with a topic penalty, checked on held-out text
   sets and with in-sample `probe_scores` ablation.

**Latent interpretation is light in T1.** In all 20 episodes together there were 9 `latent_examples` calls and 12
`latent_tokens` calls, against 549 `latent_means` calls. The open tier lets agents measure causal effects on
counterfactual text directly, so the SHIFT step of reading what a latent encodes is mostly skipped. This is by design
(SPEC (9): T1 is the easy end of the explanation-channel dial), and it is the main reason T1 is easy for this agent.

## Tool use, time and budget

| per episode | median | range | cap |
|---|---|---|---|
| task tool calls (harness) | 44.5 | 34-58 | 150 |
| forward units | 1,979 | 1,452-2,980 | 3,000 |
| probe_query units | 4 | 4-4 (20/20 at cap) | 4 |
| wall clock (s) | 610 | 454-935 | 3,600 |
| GPU queue wait (s) | 7.9 | 7.4-25.2 | - |
| Claude Code tool calls | 40 | 29-58 | - |

- **Task tools, all 20 episodes:** latent_means 549, probe_scores 114, probe_gradient 70, sample_texts 70, budget 67,
  probe_query 42 calls (39 accepted, 80 units), task_info 20, submit 20, latent_tokens 12, latent_examples 9, help 1.
- probe_query was the only cap reached, in all 20 episodes, as in smoke 1. Forward came close once (2,980 of 3,000).
- **Failed task calls: 7, all caused by the agents.** 3 tried a fifth probe_query and were rejected for budget; 3
  sent more than 64 texts to latent_means; 1 failed latent_means' per-text length check. Longest call, queue wait excluded:
  4.9 s.
- Every submission was accepted on the first try. 5 of 20 agents named a script `scratch/select.py`, which shadows the
  stdlib module; each lost one turn (as in smoke 1).
- 11 agents ran their own files with `exec(open('scratch/...').read())` 199 times, 44 of them on a `.split(...)[0]`
  prefix. The audit flagged none (F5 / R9 hold).

## Predictions vs outcomes

PREDICTIONS.md "Scaled run T1 (design v3 + D1)", written 2026-10-02 23:40 UTC before any episode; not edited.

| prediction | outcome (all 20; harness-valid in brackets where different) | |
|---|---|---|
| pass rate 65% (45-85%) | 19/20 = 95% [0.76, 0.99] (14/14) | **above range**; P(>= 19/20 \| 0.65) = 0.002 |
| planted accuracy 90% (78-97%) | 43/44 = 97.7% (31/31) | at the top of the range; P(>= 43/44 \| 0.90) = 0.06 |
| null false-claim rate 8% (2-20%) | 0/26 (0/18) | below range; P(0/26 \| 0.08) = 0.11 |
| planted "nothing found" 4% (0-12%) | 0/44 | in range |
| split share 35% (10-60%) | 1/10 | at the bottom edge; low because of saturation |
| median margin +0.06 (+0.04 to +0.10) | +0.072 | in range |
| about 20% (5-40%) of named slots within 0.05 of the bar | 30% (26%) | in range |
| at most 1 named slot below the bar | 1 (0) | as predicted |
| agent / label-aware median ~0.99 (0.93-1.03) | 1.006 (1.012) | in range |
| "D1 opens the opposite error" (class-gap misreading, planted "none") | 0 cases | did not happen |
| "removal is close to deterministic; outcome turns on naming and null calls" | yes; replicate removals differ by a median 0.004 | right mechanism |
| "pass rate >= 90% would put T1 near saturation" | 95% | this reading applies |

The slot model behind the 65% was right; its inputs were not. With the observed per-slot rates (planted 43/44, null
false claims 0/26) and this draw's 2.2 planted slots per episode, the model gives 0.977^2.2 ≈ 0.95, which matches
19/20. The guesses p_planted 0.90 and p_false 0.08 came from smoke 1 before D1. D1 removed the null false claims, and
naming was perfect.

## Environment and harness findings

- **E1 (blocker for training use): 6 of 20 episodes are falsely INVALID. These are 4 audit false-positive patterns,
  all on text inside Python that the agent writes and runs in its sandbox.** Each was checked with
  `common.transcript_audit.check_call` on the exact command: the hit disappears when the flagged token is replaced by
  a harmless one, and no other rule fires.
  - (a) **R2 bare root, 2 episodes:** a quoted `'/'` used as a print separator in a heredoc Python script,
    `print(..., '/', len(toks))`. ROOT_RX applies inside Python regions too, and ROOT_SEP_METHOD_RX only exempts
    split/strip-type methods.
  - (b) **R2 path, 1 episode:** joins onto a sandbox-literal prefix, `D='<sandbox>'; subprocess.run([D+'/tool', ...]);
    open(D+'/'+saved_to)`. PATH_RX reads `'/tool'` as an absolute path and ROOT_RX reads `'/'` as the root.
  - (c) **R2 '..', 2 episodes:** an f-string range, `f'{L.min():.2f}..{L.max():.2f}'`. TOKEN_SPLIT splits on `{}:`
    and leaves a bare `..`, and '..' stays checked inside Python regions on purpose.
  - (d) **R5 process inspection, 1 episode (3 hits):** a Python dict named `TOP = {...}` at the start of a heredoc
    line. priv_hits matches PROC_RX with re.IGNORECASE, and the Python-region exemption is off because the script
    calls `subprocess.run(["./tool", ...])`.
  - Suggested fixes (not applied; the orchestrator decides): (a) inside a Python region, count a quoted `'/'` as the
    root only when it is an argument of a path-consuming call (open, os.*, glob, Path, listdir); (b) resolve
    `NAME + '/x'` when NAME was assigned a string literal inside the sandbox earlier in the same command, and keep
    flagging unknown prefixes; (c) skip a bare `..` whose neighbours are format-spec braces; (d) match process names
    case-sensitively (`TOP` cannot run `top`). Each needs must-still-flag twins: `os.listdir('/')`,
    `open(X+'/etc/passwd')` with unknown X, `cd ..`, `open('../x')`, `; top`, `| ps aux`.
  - Then re-finish the 6 episodes with `--transcript` (`grep -l transcript_audit
    runs/shifthunt/20261003-scale1_T1_opus55/episodes/*/grade.json` lists them) and re-run margins.py. Expected:
    20/20 valid, 19/20 pass. Until then, `summarize` and margins.py report the 14 valid episodes, and those hide the
    only failure.
  - **Why it matters for RL:** a 30% false-INVALID rate is reward noise wherever INVALID maps to zero reward. Printing
    `a / b` labels and f-string ranges is natural in this task. The smoke saw 4/6 (a different set of patterns)
    before F2; this run 6/20 after F2.
- **E2 (documentation gap, not changed mid-run): an undocumented 96-token window.** `sh_core.MAX_TOKENS = 96`
  truncates every encoded text (`truncation=True, max_length=97` with BOS). With the Gemma tokenizer, 735 of 4,480
  sample texts across the plan's probes (16.4%; 2-33% per probe) and 601 of 3,360 corpus texts (17.9%) are longer, so
  their late cues (sign-off, hashtag, P.S. line, question ending) are shown to the agent but never seen by the model.
  - Training, tools and the hidden grid use the same encoder, so grading is consistent and every solver is affected
    alike.
  - 8 of 20 agents found the window, from latent_tokens output or from flipped pairs that gave identical latent means.
    At least one spent about 11% of its forward budget on truncated pairs before redesigning its texts.
  - Suggestion for the next design version: state the window in TASK.md's tool docs, or return a `truncated` flag;
    optionally cap generated text length.
- **E3 (/tmp-full incident, NOTES.md 2026-10-03 01:50).** 4 episodes overlapped 00:20-00:31 UTC. One saw a single
  `pwd: write error: No space left on device` from the Bash wrapper at 00:30:27. The command's output was complete;
  the agent checked `df`, deleted one of its own scratch files and went on. infra_failures is empty in all 20, and no
  task tool call failed. No grade is affected.
- **E4 (F3, unchanged): the launch is the workflow wrapper.** In 20/20 episodes the session cwd is the repo root, and the
  session context holds the repo's git status. No command used any of it: every one of the 782 calls
  `cd`s into its own sandbox or uses absolute sandbox paths. There are no reads outside the sandbox, no network, no
  base64/eval, and no other-episode paths (checked by a separate scan of all commands, beyond the audit).
- **E5 (hygiene).** `runs/shifthunt/20261003-scale1_T1_opus55/config.json` (instance ids and seeds) is not
  gitignored; only `episodes/` is. It is left uncommitted, as smoke 1's was. An ignore rule would be safer.
  margins.py has no single-episode or include-INVALID mode, so the finish operators recomputed its numbers by hand
  or through scratch run dirs.
- **E6 (operator side).** Two finish operators printed an episode record's non-dict fields, which echoed that
  episode's canary into the operator's own session output. It reached no sandbox, run file or commit; both test
  agents had already finished.
- **E7 (prompt wording, minor).** The prompt bans "exec of generated or transformed code", and allows
  `exec(open('scratch/a.py').read())`. 9 agents ran a `.split("if __name__")[0]` prefix of their own file, which the
  audit allows (R9). One operator read this as a mismatch with the prompt wording; adding "a prefix of your own file
  is fine" to the prompt would close it.

## Verdict against the criteria

| criterion | T1, Opus 5.5 | |
|---|---|---|
| mid-band pass rate (10-70%) | 95% [76%, 99%]; valid-only 100% [78%, 100%] | **not met: near saturation** |
| difficulty comes from understanding, not knobs | Every pass needed counterfactual design, error-term reasoning, decoy rejection and null judgement, and the one miss is a reasoning error. But little difficulty remains, and part of what remains is where the bar sits (30% of slots within 0.05 of it; the miss was by 0.004) | **met in kind, but there is little of it** |
| recipes and shortcuts low | every recipe, the black-box control and every attack 0/16 on T1b (gates_v8; Wilson upper 19%) | **met** |

**T1 is a valid environment that Opus 5.5 nearly saturates.** It is not a mid-band training tier for this model.
GRPO would see almost no within-group reward variance (at G = 8, about a third of groups would carry any signal).
The run found no environment fault that changed a grade. The harness problem (E1) changes validity, not outcomes,
and must be fixed before any RL use.

**Suggested next steps (for the orchestrator).**
1. Fix E1 with twin tests, re-finish the 6 episodes with `--transcript`, and re-run margins.py on the full run.
2. Run the scaled T2 run, which is now unblocked (gates v9 PASS, 39 kept unexposed instances). T2 removes free text
   encoding, so latent interpretation becomes necessary; that is the step T1 lets agents skip. Write the T2 scaled-run
   predictions to PREDICTIONS.md before any episode.
3. Keep T1 as the easy tier (curriculum, or for smaller models; PREDICTIONS guessed 10% for Haiku-class). If T1 must
   get harder for Opus-class, prefer changes that need more understanding over pure knobs (k, tau, query cap): for
   example nulls whose listed dependence on agent-style texts sits in the D1 gap, or candidates that share latents
   with a used decoy. Any such change needs the gates re-run and new predictions.
4. Add the 10 plan instances to exposed_instances.json (LLM-seen); plan.py and prepare will then refuse them.
5. Next design version: document the 96-token window (E2).

## Files

- **Committed:** this file only.
- **Uncommitted, per D6:** `runs/shifthunt/20261003-scale1_T1_opus55/config.json` and `episodes/*` (grade.json,
  submission.json, audit.json, tool logs, transcripts). The margins and summarize outputs went to the reviewer's
  scratchpad, not the run dir.
- Nothing was pushed.
