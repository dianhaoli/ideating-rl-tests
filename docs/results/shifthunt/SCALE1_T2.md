# ShiftHunt scaled run T2 (Opus 5.5, 20 episodes) and Haiku 4.5 on T1 (10 episodes) (2026-10-03)

Run dirs: `runs/shifthunt/20261003-scale1_T2_opus55` (episodes 03:51-07:51 UTC, two at a time) and
`runs/shifthunt/20261003-scale1_T1_haiku45` (04:41-06:23 UTC, one at a time, alongside T2). Plans: `scale_plan_T2.json`
(10 of the 39 unexposed T2new_kept_v9 instances, 2 episodes each) and `scale_plan_T1_haiku.json` (the 10 instances of
`scale_plan_T1.json`, 1 episode each). Predictions: PREDICTIONS.md "Scaled run T2 (Opus 5.5, reference v4.2 pool) and
Haiku 4.5 on T1", committed before any episode. Reviewed from all 30 test-agent transcripts (commands, tool results and
visible text, by script; final reports and the failed episodes read in full), grade.json, audit.json, tool logs,
submissions and the 30 finish-operator reports.

Per D6 and the exposed_instances.json rule this file gives aggregates only. It names no instance, episode or latent id
and does not say which attribute belongs to which slot. Quotes are redacted the same way. The episode files stay on disk
and uncommitted. The 20 plan instances (10 T1, 10 T2) are LLM-seen and are added to exposed_instances.json in the same
commit as this file.

**Amended after the adversarial check.** SCALE1_T2_CHECK.md (commit 3512bb36) re-derived every number here and found
no wrong figure, but made seven corrections. They are applied in place below and marked [check C1] to [check C7]:
the T2 pass rate is inflated by the pool's keep filter (C1); "the label-free reference clears every failing slot" is
true by construction (C2); more answer-bearing material sat in the shared directories during the Haiku arm (C3); the
audit's false negatives are wider than `ps` (C4); at least 7 agents, not 6, used the 96-token channel, and it also fed
null calls (C5); one quote in failure class (B) belongs to a passing probe (C6); 5 of Haiku's 6 planted "none" answers
came from misreading the disclosed bar range (C7). No grade, count or CI changed.

## Summary

- **T2 outcome (Opus 5.5).** 15 of 20 episodes passed; 65 of 70 slots right; mean score 0.925.
  - Planted slots: **52/52 named right**, 47/52 passed. Null slots: **0/18 false claims**. Planted "nothing found": 0/52.
  - **Every failure is a removal miss on a correctly named slot**: 5 failed episodes, one failed slot each, on 3 distinct
    slots. With the removal bar set to 0, all 20 episodes would pass.
- **Pass rate (Wilson 95%).** All 20 graded: **15/20 = 75% [53%, 89%]**. Without the one episode that broke a rule
  (it fails on grade anyway): 15/19 = 79% [57%, 91%]. Harness-valid only: 9/10 = 90% [60%, 98%]. The valid-only view
  hides 4 of the 5 failures.
- **The pool filter inflates the T2 rate [check C1].** The pool keeps only instances that label-free reference v4.2
  solves one-shot, v4.2 was tuned on this pool, and the agents track v4.2 slot by slot (margins r = 0.89). Adding the
  observed agent-minus-reference differences to v4.2's margins reproduces this plan (0.71-0.74 vs 0.75 observed) and
  gives about 0.77-0.80 on the 39 kept instances, 0.31-0.33 on the 14 dropped ones, and **about 0.65-0.67 on all 53
  built**. Unfiltered, T2 would probably sit inside 10-70%, near the top; the extra failures would land on slots
  where v4.2 also misses, so the bar would decide them. This is a model estimate, not an observed rate.
- **Harness validity.** 10 of 20 T2 episodes are INVALID (`transcript_audit`): 9 are audit false positives (10 new
  patterns, H1), 1 is a real, benign violation (H2). That is a 45% false-INVALID rate after the E1 fix. Blocker for RL use.
- **Per instance.** 7 both pass, 1 split, 2 both fail. The two agents on a slot end a median 0.009 apart in removal,
  and their margins correlate 0.95: outcomes are mostly a property of the instance.
- **Removal.** Margin median +0.056; 23 of 52 named slots within 0.05 of the bar, 5 below. Agent / label-aware median
  0.951, agent / label-free v4.2 median 1.014. The agents track the label-free reference slot by slot (agent minus
  reference margin: median +0.005, IQR -0.012 to +0.015).
- **Haiku 4.5 on T1 (same 10 instances as Opus T1).** **0/10 = 0% [0%, 28%]**; 3 of 35 slots right; planted 3/22
  (named 7/22); **null false claims 13/13**; planted "none" 6/22. Even with the removal bar at 0 it passes 0/10. Opus 5.5
  on the same instances: 19/20, 69/70 slots. Haiku fails at method discovery (it names from sample co-occurrence and
  never builds counterfactuals), not near the bar.
- **Predictions.** T2: pass rate (55%, 30-80%), planted accuracy, null false claims, margins and reference ratios all in
  range; the split share (35%) came in at the bottom edge (1/10). Haiku: the pass rate (30%, 0-60%) was in range at the
  bottom, but its slot metrics were far outside (planted accuracy 14% vs 65%; null false claims 100% vs 25%), and the
  predicted main failure (null claims in the D1 gap) was wrong: Haiku never estimated reliance at all.
- **Verdict.** On the instances it ran, T2 is not a clean mid-band tier for Opus 5.5: 75% sits at the upper edge of
  10-70% (CI 53-89%), the intended interpretation step (naming, decoys, nulls) is saturated, and pass/fail is decided
  by latent selection close to the bar on a few instances. Qualification [check C1]: that 75% is on a pool filtered
  by a reference the agents track; without the filter the estimate is about 65%, inside the band, but the added
  failures would be bar-decided too. T1 separates Haiku 4.5 from Opus 5.5 sharply (0/10 vs 19/20), a model dial with
  two points, both outside the band. Details and next steps at the end.

## Setup and provenance

- **Code.** config.json and every episode: git_sha fd91d8b2 (dirty only through `runs/api_budget/ledger.jsonl` and an
  untracked smoke summary; no task or harness file differs). Template sha ed473483 in all 30 (the E2 96-token sentence;
  harness prompt with the E7 exec wording). TASK.md is byte-identical within each arm once digits are masked, contains
  the 96-token sentence in all 30, and has no unfilled `{public.*}` placeholder; the access mode matches the tier.
- **Test agents.** One fresh Claude Code workflow subagent per episode (wf_491d9234-493).
  - T2: 1,853 assistant rows, all `claude-opus-5-5`; 985 tool calls, all Bash.
  - Haiku: 1,065 assistant rows, all `claude-haiku-4-5-20251001`; 409 tool calls (399 Bash, 8 Write, 1 Edit, 1 Read).
  - Each transcript has exactly one user text message (the wrapped prompt). Thinking blocks are empty in storage, so
    only actions and visible text could be checked.
- **Finish.** All 30 finishes found exactly one transcript by search (prompt_match `workflow_wrapper`), the test agent's
  own; `agent_models` recorded (F4). No `infra_failures`, no leak, behavioral_exposure 0, no cross-episode access, no
  codename or leak strings in agent files, no symlinks out, every episode submitted (status `submitted` in 30/30).
- **Re-derivation.** An in-memory regrade (`grader.grade`, CPU, 8G scope) of all 30 submissions matches grade.json
  30/30. The exact linear decomposition (each latent's share of E_S and E_T from the grid cell means of g_i x f_i)
  reproduces the grader's removal on all 52 named T2 slots. The audit re-run (`transcript_audit.check_call` on every
  call) reproduces every audit.json exactly.
- **Caps.** 150 tool calls, 3000 forward, 4 probe_query units, 0 generate, 0 gradient, 3600 s, 180 s per call; k = 20;
  profile full. T2 = sample-only (latent_means / latent_tokens take sample ids); Haiku arm = T1 (open).
- **Pools.** T2new_kept_v9: reference v4.2 one-shot 39/39, every non-reference solver <= 3/39 (gates v9). "Kept"
  means v4.2 cleared every slot on its first try (39 of 53 built), and v4.2's rules were chosen on these 53 instances
  (held-out old pool: 62% vs 74%); see [check C1] for what the filter does to agent pass rates. The T2 plan
  draws 26 planted and 9 null slots (52 and 18 slot-episodes; predicted about 39 and 27); 3 instances are all-planted.
  Haiku plan: 22 planted and 13 null slots, 2 all-planted instances.
- **Opus T1 comparison.** The 6 Opus T1 episodes that were falsely INVALID were re-finished after the audit fix
  7345b11e; that run is now 20/20 harness-valid with unchanged grades (19/20 pass).

## T2 aggregate metrics (Wilson 95%)

| metric | all 20 (graded) | harness-valid only (10) | prediction |
|---|---|---|---|
| episode pass | **15/20 = 0.75 [0.53, 0.89]** | 9/10 = 0.90 [0.60, 0.98] | 0.55 (0.30-0.80) |
| episode pass without the real-violation episode | 15/19 = 0.79 [0.57, 0.91] | - | - |
| episode pass, 1st / 2nd episode per instance | 8/10, 7/10 | - | - |
| mean score | 0.925 | 0.967 | - |
| slots right | 65/70 = 0.93 [0.84, 0.97] | 34/35 | - |
| planted accuracy (name + removal + topic kept) | 47/52 = 0.90 [0.79, 0.96] | 23/24 = 0.96 [0.80, 0.99] | 0.82 (0.65-0.93) |
| planted slots named right | 52/52 [0.93, 1.00] | 24/24 | - |
| null false-claim rate | 0/18 [0.00, 0.18] | 0/11 [0.00, 0.26] | 0.06 (0.00-0.18) |
| planted "nothing found" rate | 0/52 [0.00, 0.07] | 0/24 | 0.03 (0.00-0.10) |
| instances split (one pass, one fail) | 1/10 | 0 of 2 instances with 2 valid episodes | 0.35 (0.10-0.60) |
| episodes passing with the removal bar at 0 | 20/20 | 10/10 | - |

The slot model of PREDICTIONS.md fits once its inputs are replaced by the observed rates: planted 0.904, null false
claims 0, 2.6 planted slots per episode give 0.904^2.6 = 0.77 (observed 0.75).

### Removal margins and reference ratios (D2)

`$PY -m tasks.shifthunt.margins --run-dir runs/shifthunt/20261003-scale1_T2_opus55 --ref-run
runs/shifthunt/20261003-012202_gates_v9_P1_ref42_T2new --out <scratch under runs/, deleted>` gives the harness-valid
column. margins.py skips INVALID episodes, so the all-20 column uses the same formulas (margins.q, margins.ref_removals)
in a scratch script. Label-aware = ref_removal_k at k = 20; label-free = reference v4.2, attempt 0 of gates v9, which
named every planted slot of these instances right and passed all 10.

| over planted slots named right | all 20 (n = 52) | harness-valid (n = 24) | prediction |
|---|---|---|---|
| removal margin (removed - tau): median [q25, q75] | **+0.056** [+0.028, +0.094] | +0.058 [+0.036, +0.094] | +0.04 (+0.02 to +0.06) |
| margin min / max | -0.062 / +0.315 | -0.043 / +0.162 | - |
| within 0.05 of the bar (abs(margin) < 0.05) | 23/52 = 44% [32%, 58%] | 9/24 = 38% | 55% (35-75%) |
| below the bar | 5/52 = 10% [4%, 21%] | 1/24 | 12% (4-25%) |
| agent / label-aware: median (min-max) | **0.951** (0.70-1.02) | 0.957 (0.77-1.02) | 0.92 (0.86-0.97) |
| agent / label-free v4.2: median (min-max) | **1.014** (0.71-1.26) | 1.024 (0.80-1.26) | 0.97 (0.90-1.03) |
| tau / label-aware: median (min-max) | 0.78 (0.51-0.90) | 0.78 (0.65-0.90) | - |
| topic kept: median (min) | 0.994 (0.852) | 0.993 (0.852) | floor 0.8 |
| slots submitted with fewer than 20 latents | 6/52 (3 of them failed) | 1/24 | - |

- **Agents match the label-free reference.** Reference v4.2 on the same 10 instances: 10/10 pass, 26 named slots, margin
  median +0.053, 13 within 0.05, minimum +0.000. Agent minus reference margin on the same slot: median +0.005 (IQR
  -0.012 to +0.015); the agent is below the reference on 22 of 52. The oracle top 20 latents by true grid share remove a
  median 1.09x what the agent's set removes, and share a median 15 of 20 latents with it.
- **Replicates on the same slot (26 pairs).** Removal differs by a median 0.009 (mean 0.014, max 0.061; 13 of 26 within
  0.01, 23 within 0.03), against 0.004 in T1. Latent sets overlap with Jaccard median 0.74 (q25 0.55, min 0.18; T1
  0.82). Margins correlate 0.95 between the two agents; the within-slot sd is 0.014 against 0.067 across slots.
- **By cue position** (planted slot-episodes, grouped so that no slot is identified):

  | group | named slot-episodes | passed | margin median | within 0.05 | tau / label-aware median | Opus T1 margin median |
  |---|---|---|---|---|---|---|
  | end of text (sign-off, hashtag, question ending) | 16 | 16 | +0.119 | 0 | 0.70 | +0.119 |
  | greeting line (start of text) | 8 | 7 | +0.021 | 8 | 0.77 | +0.029 |
  | mid-text (bullet layout, British spelling, exclamation, negative sentiment) | 28 | 24 | +0.046 | 15 | 0.81 | +0.068 |

  4 of the 5 failed slot-episodes are on slots of one mid-text family, on 2 instances. Its bar sits at tau / label-aware
  0.88 and 0.90 on those slots (pool q90 0.89). The same family passed 6/6 in the Opus T1 run, where agents measure
  each latent's causal share on new text. End-of-text cues give the same margins in both tiers.
- Accuracy-based SHIFT (reported, not gated): median 0.127, range -0.02 to 0.90.

## Per-instance pattern and GRPO reward variance

| pattern over the 2 episodes | instances | scores |
|---|---|---|
| both pass | 7 | 1.0 / 1.0 |
| split | 1 | 1.0 / 0.667 |
| both fail | 2 | 0.75 / 0.75 and 0.667 / 0.667 |

Valid-only view: 2 instances have 2 harness-valid episodes (both pass both), 6 have 1 (5 pass, 1 fails), 2 have none
(one of them is a both-fail instance). So the valid-only view drops one of the two hardest instances entirely.

**The two both-fail instances failed the same slot in the same way.** On one, both agents used the same selection rule
(keep a latent only if its class contrast has the same sign on all three probes) and both stopped at 10 latents; on the
other, both used only latents that fire on the attribute's own tokens. Independent episodes at p = 0.75 would split
37.5% of the time; 1 of 10 split (P(<= 1/10) = 0.06). The intraclass correlation implied by 7 / 1 / 2 is about 0.73.

**What this means for GRPO.**
- **Binary pass reward.** 1 of 10 two-rollout groups has any variance. Fitting a beta-binomial to the 7 / 1 / 2 pattern,
  a group of G rollouts has any variance with probability about 10% / 18% / 25% / 31% at G = 2 / 4 / 8 / 16. If each
  instance is instead shrunk to its own Jeffreys estimate, the same figures are 30% / 55% / 79% / 95%. The replicate
  structure above (same slot, same heuristic, margins r = 0.95) favours the low end, but 10 instances cannot pin it
  down. At an independent p = 0.75 they would be 38% / 68% / 90% / 99%.
- **Slot-mean score** carries no extra signal here: both both-fail pairs have identical scores.
- **Continuous removal credit** would mostly reward noise: within a slot the margin sd is 0.014 (the two agents' margins
  on the three failed slots were -0.001 / -0.062, -0.043 / -0.031 and -0.002 / +0.024). Most of the spread is between
  slots, which GRPO's group baseline removes.
- **Where signal would come from:** a few instances whose bar sits within a few hundredths of what a careful selection
  removes (here 3 of 10). Elsewhere, groups are all-pass.

## Failure classification (T2)

**5 failed slot-episodes, all "named right, removal short of the bar". Cause in every case: a latent-selection error by
the agent. No environment fault.** All three slots are reachable. Label-free v4.2 clears them by +0.021 to +0.036, but
that is true by construction [check C2]: clearing every slot is the keep rule, and v4.2 was tuned on this pool. The
independent evidence is from in-memory regrades: the oracle top 20 latents (by true grid share) remove 0.21-0.36
against bars of 0.16-0.31 with at least 0.946 of the topic kept; and small changes to the agents' own sets pass:
adding the dropped context latent on the 10-latent pair (+0.010; the sibling ends 0.0025 short, and filling both
with the best remaining latents passes), the 9-latent agent's own earlier 20-latent set (+0.005), and swapping the
dropped latent in for the weakest pick on the two 20-latent misses (+0.028, +0.017). Every agent knew its removal was
the risk and said so in its report.

**(A) Unused k: 3 of 5 (9 or 10 latents of the 20 allowed; misses -0.031 to -0.062).**
- Two agents on one instance (same slot) both submitted 10 latents for this probe and 20 for the others. Both kept a
  latent only if its estimated attribute effect had the same sign across the three probes' samples:
  > `# effect of attribute presence on latent: signed by pattern: e_i = min over probes of (v_q * Dl_q)`

  That rule dropped a broadly active latent with the second- or third-largest sample attribution, which carries part of
  the cue through context. Adding just that latent: one agent passes (+0.010, in-memory regrade), the other ends 0.0025
  short. Filling to 20 with the agents' own next sign-consistent picks does not pass (about 0.28); filling with the best
  remaining latents by true share does (0.342 and 0.345 against 0.305).
- One agent on the other both-fail instance kept 9 latents:
  > I kept only latents that fire on the attribute's own tokens and never on topic words. I left out tense, bullet and
  > dense latents so that ablation wouldn't reduce the topic signal (to keep the 80% topic retention).

  It had tested a 20-latent set with `probe_scores` earlier, which the operator's in-memory regrade puts at +0.005. Its
  own estimate before submitting was "about 24%" (actual 0.243 against a bar of 0.305).

**(B) A context-spread latent left out from a full 20: 2 of 5 (misses -0.0009 and -0.002).**
- > I left out latents that also carry topic, such as [latent] for probe [..].

  [check C6] This quote names a latent of a probe that *passed* in that episode; it shows the agent's general policy,
  not the failing choice. On the failing probe, the agent's own table split a dense latent's effect into a small
  cue-token part and a larger "other tokens" part, and the agent dropped it without comment; its report never
  mentions that latent. On the grid it is one of the largest contributors to the cue's effect and has a negative
  topic share. Swapping it in for the weakest pick passes by about +0.03 (operator regrade).
- The other agent filed a body-token latent under a different co-occurring cue because of its pattern across topics,
  and dropped a second one for the same reason:
  > For probes [..] and [..] I took the 20 latents that most cut the attribute's effect. All of them behave the same way
  > across the three topic pairs, so they shouldn't carry topic information.

  The sibling episode on that instance kept both latents (18 of 20 shared) and passed by +0.024.

**Common thread.** The cue changes how other tokens are read (a context effect, NOTES v4), so the latents carrying it
are not all local to the cue's tokens or pure across topics. Agents that kept only token-local, attribute-pure latents
lost removal. Four of the five also cited topic safety, yet topic kept never came near the floor (minimum 0.852 over 52
named slots). This is SCALE1_T1_CHECK C1 (dropping a latent that carries the attribute plus some topic) in its T2 form,
and PREDICTIONS failure mode (1). In T2 it is more costly because the agent cannot measure topic loss on unconfounded
text.

**Contributing design factor (D2, not a fault).** The three failing slots have tau / label-aware 0.80, 0.88 and 0.90
(the slot median is 0.78). Moving every tau by -0.02 gives 17/20 passes, +0.02 gives 14/20, +0.03 11/20, +0.05 5/20;
scaling by 0.9 gives 17/20, 1.1 13/20, 1.2 4/20. The pass rate is as sensitive to the frozen bar as in T1 (C9).

**Failures that did not happen.** No null false claim (0/18), no wrong name (0/52), no planted "none" (0/52), no topic
floor violation, no cap or wall-clock ending. Close calls that went right:
- one planted slot named from a single-text ratio of 0.38 (just under the disclosed 0.4 floor), backed by the weight
  pattern across probes;
- null calls resting partly on combined edits (two attributes changed in one query, so opposite effects could cancel)
  or on latent evidence alone (2 of 18 null slot-episodes got no query);
- one null call at an estimated 0.2-0.3x for two attributes edited together.

**Environment-only failures: 0 slots.** The 10 INVALIDs touch validity, not grades (H1, H2).

## What the T2 agents did

All 20 took a route close to the one reference v4.2 scripts (SPEC: attribution, reading latents, a cross-probe
control, one paired query per claim). None found a way around reading latents.
1. Read the samples, tagged attributes with regexes, and saw the 5 confounded listed candidates and 2 unlisted decoys
   per probe; regressed the logits on the candidates that vary within a class.
2. Split each logit exactly into g x activation per latent plus the error term, checked with `probe_scores` ablation.
3. Labelled latents from dashboards and per-token activations: 157 `latent_examples` and 178 `latent_tokens` calls,
   4 to 35 per episode in all 20 (the Opus T1 run made 21 such calls in total). T2 does force the SHIFT reading step.
4. Compared latents across the episode's probes and samples (17/20 say so in their text). 19 of 20 name the unlisted
   decoys in their reports and the 20th models them in its regression code; all set them aside.
5. Used `probe_query` for paired minimal edits: 77 units, 50 on planted probes and 27 on null probes; 16 of 18 null
   slot-episodes got at least one query; 11 of 20 episodes queried every probe.
6. Chose 20 latents (46 of 52 named slots) by estimated contribution, excluding topic and decoy latents.

**The 96-token window as a natural experiment (new route).** At least 7 of 20 agents used the stated window: long
sample texts lose their late cues (sign-off, hashtag, P.S.), so those cues vary within a class and the real logit
shows their effect without a probe_query. 6 say so in their reports, and all 6 passed. [check C5] A 7th, a failing
episode, built a truncation-aware within-class regression in its code without mentioning it; its miss was on a
mid-text cue. In the check's seeded sample the contrast also fed one null call and one planted claim made without
any probe_query, so it bears on null calls as well as naming. There is no sign that it changed grades: end-of-text
families have the same margins in T1, which did not disclose the window, and naming was 52/52 either way. But it is a
label-free behavioural signal that no gate solver tests (H4).

18 of 20 T2 agents discuss the SAE error term; 9 restate the D1 scale. Their removal forecasts were usually within
about 0.05 of the grader and off by 0.1-0.2 on a few slots, in both directions.

## Tool use, time and budget

| per episode | T2 Opus 5.5 (20) | T1 Haiku 4.5 (10) | T1 Opus 5.5 (20, SCALE1_T1) | cap |
|---|---|---|---|---|
| task tool calls (harness), median (range) | 36.5 (20-54) | 27.5 (21-35) | 44.5 (34-58) | 150 |
| forward units | 1,260 (1,034-1,813) | 1,640 (548-2,452) | 1,979 (1,452-2,980) | 3,000 |
| probe_query units at cap | 17/20 (3 used 3) | 9/10 | 20/20 | 4 |
| wall clock (s) | 1,092 (761-1,346) | 326 (246-556) | 610 (454-935) | 3,600 |
| GPU queue wait (s) | 7.5 (7.3-7.6) | 7.3 (7.2-7.7) | 7.9 (7.4-25.2) | - |
| Claude Code tool calls | 49.5 (39-60) | 41.5 (31-52) | 40 (29-58) | - |
| failed task calls (all agent-caused) | 1 | 11 | 7 | - |

- **T2 task tools, all 20:** latent_tokens 178, latent_examples 157, probe_scores 122, sample_texts 70, latent_means 70
  (sample ids), probe_gradient 70, probe_query 63 calls (77 units), budget 62, task_info 20, submit 20. The one failed
  call passed a non-sample id to latent_tokens. Longest call, queue wait excluded: 2.4 s.
- **Haiku task tools, all 10:** probe_scores 114, sample_texts 44, probe_gradient 35, probe_query 25 calls, latent_means
  23, budget 21, latent_examples 20, task_info 10, submit 10, help 2, latent_tokens 0. Failed calls: 7 over the
  probe_query budget, 3 latent_means argument errors, 1 malformed probe_query.
- No episode in either arm hit any cap other than probe_query, or the wall clock. Every submission was accepted on the
  first try. Haiku stopped early: median 5.4 minutes, with most of the budget unused.

## Predictions vs outcomes

PREDICTIONS.md "Scaled run T2 (Opus 5.5, reference v4.2 pool) and Haiku 4.5 on T1", written 2026-10-03 03:55 UTC
before any episode; not edited.

**T2 (Opus 5.5)**

| prediction | outcome (all 20; harness-valid in brackets where different) | |
|---|---|---|
| pass rate 55% (30-80%) | 15/20 = 75% [0.53, 0.89] (9/10) | in range, upper part; P(>= 15/20 \| 0.55) = 0.055 |
| planted accuracy 82% (65-93%) | 47/52 = 90% (23/24) | in range; P(>= 47/52 \| 0.82) = 0.075 |
| null false claims 6% (0-18%) | 0/18 (0/11) | in range |
| planted "nothing found" 3% (0-10%) | 0/52 | in range |
| split share 35% (10-60%) | 1/10 | bottom edge; outcomes mostly instance-determined |
| median margin +0.04 (+0.02 to +0.06) | +0.056 (+0.058) | in range |
| within 0.05 of the bar 55% (35-75%) | 44% (38%) | in range |
| below the bar 12% (4-25%) | 10% (1/24) | in range |
| agent / label-aware median 0.92 (0.86-0.97) | 0.951 (0.957) | in range |
| agent / label-free v4.2 median 0.97 (0.90-1.03) | 1.014 (1.024) | in range |
| failure mode (1) removal short, from < 20 latents or a dropped high-share latent, most frequent | 5 of 5 failures; 3 used < 20 latents, 2 dropped a context-spread latent | right |
| failure modes (2) null false claim, (3) wrong co-occurring candidate | 0 and 0 | did not happen |
| probe_query at cap in >= 18/20 | 17/20 | just missed |
| tool calls median about 50, none at 150 | 36.5, max 54 | fewer than predicted |
| wall clock median about 15 min | 18.2 min | close |
| "30-75% would make T2 the mid-band tier T1 is not; check failures are judgement, not the bar alone" | 75%; failures are judgement, but the bar decides them (C9-style sensitivity above) | at the edge of this reading |

**Haiku 4.5 on T1**

| prediction | outcome | |
|---|---|---|
| pass rate 30% (0-60%) | 0/10 [0.00, 0.28] (valid 0/5) | in range at the bottom; P(0/10 \| 0.30) = 0.028 |
| planted accuracy 65% (40-85%) | 3/22 = 14% [0.05, 0.33] | far below range |
| null false-claim rate 25% (8-50%) | 13/13 = 100% [0.77, 1.00] | far above range |
| planted "nothing found" 8% (0-25%) | 6/22 = 27% [0.13, 0.48] | above range |
| median margin on named slots +0.05 (-0.01 to +0.08) | -0.071 (n = 7) | below range |
| within 0.05 of the bar 40% (20-65%) | 3/7 | in range |
| below the bar 18% (5-40%) | 4/7 = 57% | above range |
| no valid submission, or ended by a cap / the wall clock, 10% (0-30%) | 0/10 (but 4/10 broke the /tmp rule, H2) | in range |
| main failure: null claims from a first estimate in the D1 gap | Haiku never estimated reliance; it named from sample co-occurrence | wrong mechanism |
| "Haiku <= 20% vs Opus 95% would mean T1 separates models strongly" | 0% vs 95% | this reading applies |

The Haiku slot model (planted 0.65, null false claims 0.25) was far too generous; the v3 guess for a small agent (10%,
"stops at sample co-occurrence") described the behaviour better than the 30% written today.

## Model separation: Haiku 4.5 vs Opus 5.5 on the same 10 T1 instances

| metric (T1, same 10 instances) | Opus 5.5 (20 episodes) | Haiku 4.5 (10 episodes) |
|---|---|---|
| episode pass | 19/20 = 95% [76%, 99%] | **0/10 = 0% [0%, 28%]** (Fisher p = 4e-7) |
| episode pass with the removal bar at 0 (naming and null calls only) | 20/20 | **0/10** |
| mean score | 0.988 | 0.092 |
| slots right | 69/70 | 3/35 |
| planted named right | 44/44 | 7/22 |
| planted accuracy | 43/44 | 3/22 |
| null false claims | 0/26 | **13/13** |
| planted answered "none" | 0/44 | 6/22 |
| named slots: margin median / below the bar | +0.072 / 1 of 44 | -0.071 / 4 of 7 |
| named slots: agent / label-aware median | 1.006 | 0.67 |
| named slots submitted with 20 latents | 44/44 | 2/7 |
| episodes encoding their own texts (latent_means on new text) | 20/20, about 500-1,300 texts each | 2/10, 15 and 24 texts |
| episodes judging reliance against the topic effect (TASK.md's 0.4x / 0.2x scale) | 20/20 (SCALE1_T1) | 0/10 |
| episodes discussing the SAE error term | 19/20 | 0/10 |
| harness-valid | 20/20 (after the E1 re-finish) | 5/10 (4 real violations) |

**Where Haiku fails.** Every one of the 35 slot-episodes was decided at naming or the null call; removal mattered for
only 7. Per slot: 13 null slots all given an attribute; of 22 planted slots, 9 named wrong, 6 answered "none", 7 named
right (3 passed, 4 removal short, all four with 4-15 latents). Of the 9 misnamed planted slots, 4 latent sets (mostly
top 20 by |g|) would have cleared the bar on the true attribute, consistent with the known "top-|g| clears about half".

Method signals (by script over all 10 transcripts, plus the operators' full reads):
- **Naming from sample co-occurrence, 9 of 10** (the tenth made a single claim, from one query, and answered "none"
  elsewhere). These episodes tabulated candidates per label in the sample, saw several at 100% / 0%, and picked one:
  > Probe 0 ([topic] vs [topic]): Relies on [attribute] - [topic] texts had 100% [attribute] attribute, [topic] had 0%.

  4 of 10 gave the same attribute to every probe it claimed:
  > Finding: All 4 probes learned to rely on the "[attribute]" attribute - a perfect surface feature that distinguishes
  > one topic from the other in the training data.

  TASK.md says the sample alone does not decide; Opus agents say the same in their own words, Haiku agents acted
  against it.
- **Removal judged on the confounded sample, 9 of 10.** In most, "attribute dependence" was the drop in the sample class
  gap after ablating top-|g| latents, which is the same number as "topic kept" read the other way:
  > Attribute dependence reduction: 15.0% (meets 15% threshold) / Topic dependence preserved: 85.0% (exceeds 80% minimum)
- **The disclosed bar range read as the bar, 4 of 10** (3 as a 15-60% target window):
  > removing top 10-20 latents reduced attribute dependence by 17-21% (within the required 15-60% range)
- **"None" from failing its own window, 2 of 10 [check C7].** 5 of the 6 "none" answers on planted slots (3 in one
  episode, 2 in another) came from missing a self-made 15% / 80% window built on the misread bar range (H6; "Could not
  identify a reliable attribute ablation"), not from a reliance estimate. All 6 "none" answers in the arm were on
  planted slots; none on a null slot. The "removal bar at 0" view keeps these answers fixed, but neither episode could
  have passed with correct answers there: one also has a null false claim, and the other would have needed 3 more
  correct names.
- **Queries.** 6 of 10 spent all 4 units on probe 0; none queried more than 2 probes. Most queries changed the topic or
  several cues at once. In at least two episodes the one clean result contradicted the claim and the claim was kept
  (adding the claimed cue to two texts moved the logit by 0.002x and 0.03x of the class gap; the episode still claimed
  that cue on all four probes).
- **Rule and tool handling.** 4 of 10 wrote their own tool output to /tmp and read it back, against the prompt (H2),
  mostly to work around their own `./tool ... | python3 << EOF` (the heredoc replaces the pipe). 1 read its own output
  from Claude Code's tool-results directory (H3).

**Reading.** T1 separates the two models by method discovery: Opus 5.5 finds and runs the counterfactual-pairs-through-
the-SAE pipeline in every episode, Haiku 4.5 never does. With two points this is a cliff, not yet a graded dial. Neither
arm is in the band: Opus saturates T1 and Haiku is at the floor, where binary GRPO reward is 0 for every group (3 of 35
slots right give a sparse slot-level signal).

**Comparison caveat.** Haiku's episodes had the E2 sentence (the 96-token window) and the E7 exec wording; Opus T1's
did not. Both are clarifications; no Haiku agent mentioned the window, and none of Haiku's failures involve either.

## Environment and harness findings

- **H1 (blocker for RL use): 9 of 20 T2 episodes and 1 Haiku episode are falsely INVALID, from 10 audit false-positive
  patterns the E1 fix (7345b11e) does not cover, plus one environment-caused R1 (H3).** Each was checked with
  `transcript_audit.check_call` on the exact command: the hit disappears when only the flagged token is changed, and no
  other rule fires. Must-flag twins still fire (`(nc -zv h 80)`, `echo x | nc h 9000`, `os.listdir('/')`,
  `| top -b -n1`, `ps aux | head`, `sed -i 's/x/exec(p)/'`, `exec(open('/etc/x.py').read())`, `.replace` on an own
  file, `awk ... /etc/passwd`, `cat ~/.bashrc`, `open('~/x')`, `/tmp/...`, `.claude/.../tool-results/...`).

  | # | rule | pattern (all text inside the agent's own Python or shell in its sandbox) | T2 eps | Haiku eps | clears with |
  |---|---|---|---|---|---|
  | a | R9 | sed writes the allowed own-file exec form, but its escaped delimiter `scratch\/x.py` (and shell-escaped `\"` in a `-c` program) hides it from EXEC_OWN_FILE_RX | 2 | 0 | a `|` or `#` delimiter |
  | b | R9 | exec with a fresh-dict globals (`g:={}`) or through a variable holding the allowed expression (`src=open(..).read().split(..)[0]; exec(src,g)`) | 1 | 0 | `globals()` / the literal form |
  | c | R9 | computed marker `split(chr(10)+'#MAIN')[0]` (the audit flags computed markers on purpose; see H5 for why the agent used it) | 1 | 0 | the literal `'\n#MAIN'` |
  | d | R2 `~` | `prof+'~'` string literals; the Python-region exemption is off because PY_SHELL_RX's shell-name alternative matches the topic label `'fish'` as the fish shell | 1 | 0 | `'*'`, or the label renamed |
  | e | R2 path | an awk `/regex/` literal read as an absolute path (awk's `~` operator would fire R2 `~` too) | 1 | 0 | `match($0,"...")` |
  | f | R2 root | `'/'.join(...)` as a text separator nested in a print argument or an f-string | 2 | 0 | `'|'.join` |
  | g | R2 root | `' went'/' goes'` inside a Python comment | 1 | 0 | `,` |
  | h | R5 | `top` inside a Python string literal after `|` or `(` (`'| top %s'`, `'where(top tokens)'`, `"(top 4 latents)"`), exemption off because the script calls subprocess | 2 | 1 (with real R2) | `tops` / `best` |
  | i | R5 | exemption off because PY_SHELL_RX's `\bpipes\b` matches the English word in a Python comment; then `(top 6` fires | 0 | 1 | "with a pipe" |
  | j | R4 | `(Nc@Gc)`: case-insensitive `nc` after `(`, and `@` (matmul) is not in the exempt lookahead | 1 | 0 | `Nz` |

  - Suggested fixes (the orchestrator decides; each with twins): (a) un-escape `\<delim>` and `\"` in the kept sed
    replacement and in `-c` programs before EXEC_OWN_FILE_RX; (b) accept `{}` / `name:={}` / a name bound to `{}` as
    globals, and `exec(NAME, ...)` when NAME was assigned exactly the allowed expression; (c) accept a separator built
    only from string literals and `chr(<int>)` joined by `+`, or change the prompt (H5); (d) match PY_SHELL_RX's shell
    names only as a whole literal or a path ending (`'bash'`, `'/bin/sh'`, not `'fish'` as a value in a dict); (e, f, g)
    skip awk `/regex/` literals and Python comments, and count `'/'` as the root only as an argument of a
    path-consuming call; (h, i, j) apply process and network names only to shell command positions and to
    subprocess / os.system arguments, never inside Python string tokens or comments, and add `@` to the R4 lookahead.
  - **Side finding (false negatives, untested), widened by [check C4]:** R5 (process inspection) and the R4 client
    rule (ssh/scp/nc/telnet/rsync class) fire on none of `subprocess.run([...])`, `subprocess.run('...', shell=True)`,
    `os.system` and `os.popen`, whether in `./py -c`, in a heredoc to `./py` or in a Write of a .py file; nor on
    `subprocess.run(['top'])`, so SCALE1_T1_CHECK recommendation 1's "keep `subprocess.run(['top'])` flagging" rested
    on a twin that never fired; nor on a raw `socket.create_connection`. `curl`, `wget`, URLs and `/proc` reads are
    flagged everywhere. No harness test covers any of these shapes.
  - Then re-finish the 9 T2 episodes and the Haiku R5-only episode with `--transcript` (`grep -l transcript_audit
    runs/shifthunt/20261003-scale1_*/episodes/*/grade.json` lists them) and re-run margins.py. Expected T2: 19/20 valid
    (the H2 episode stays INVALID), 15/19 pass.
  - **Why it matters:** T1 had 6/20 false INVALIDs before E1; T2 has 9/20 after it, from new shapes. Agents write more
    and longer helper scripts in T2 (heredocs, sed edits of their own files, chained exec of helpers), so the
    regex audit meets more Python text. A parser-based treatment of Python regions would close the class better than
    more regex exemptions.
- **H2: real violations, benign in effect.**
  - T2, 1 episode: 6 calls ran `exec(open('scratch/x.py').read().split('<marker>')[0].replace('print','(lambda
    *a,**k:None)'))` to silence its own prints. The prompt bans exec of transformed code and the audit keeps `.replace`
    flagged on purpose; the episode fails on grade anyway (score 0.667).
  - Haiku, 4 episodes: wrote their own tool responses to /tmp and read them back ("not /tmp" in the prompt). Every file
    holds only that episode's own outputs; no episode read another's file. **21 of these files are still in /tmp**
    (about 40 KB: sample texts, gradients and ablation logits of 4 T1 instances; no answers). Operators may not write
    outside runs/ and tasks/, so the user or orchestrator should delete them: `/tmp/probe{0,1,2,3}_samples.json`,
    `/tmp/p{0..3}_grad.json`, `/tmp/p*_abl*.json`, `/tmp/p{2,3}_sample.json`, `/tmp/ablated.json`, `/tmp/q0.json`.
- **H3 (new channel, E8 class): the shared tool-results directory.** Claude Code saves any tool output over about 30 KB
  to the parent session's `tool-results/` directory, for operators and test agents alike, and tells the caller to read
  it from there.
  - 3 of 10 Haiku agents had an output saved there; one read its own file with Read (R1, the only INVALID reason in
    that episode besides a real /tmp hit). The file's path matches the one announced in its own tool result two seconds
    earlier, and nothing else there was touched. No T2 agent hit the limit.
  - At least 4 operator dumps with episode material sit in the same directory (the operators' own reports name them;
    this review only grepped them for episode ids and printed nothing from them). One, a T2 first episode's dump with its submitted answer, was there for
    the whole second episode on that instance; that agent never went near it (0 violations). Operators' `rm` was denied
    by the permission classifier each time.
  - [check C3] More answer-bearing material was reachable during the Haiku arm, which reused the 10 LLM-seen T1
    instances. The shared scratchpad (E8) held 13 files from before 03:00 with T1 episode or instance ids, among them
    per-slot rows for all 10 instances (planted flags and latent lists for both Opus episodes) and two transcript
    dumps with submissions. The tool-results directory held two T1-phase operator dumps (01:47, 02:21), one with an
    Opus T1 submission for the instance of a later Haiku episode, present while that episode ran. SCALE1_T1_CHECK
    called the scratchpad files "harmless once the 10 instances are retired"; they were not retired before the Haiku
    arm. No Haiku or T2 tool call touched the scratchpad or `.claude` apart from the one Read of the agent's own
    persisted output above, no transcript contains a canary or leak string, and Haiku scored 0/10, so no grade is
    affected.
  - Any read there is flagged (R1/R2/R3), so it cannot leak silently, but the harness itself points agents there.
    Suggested: exempt a Read of exactly the path announced by a preceding persisted-output result in the same
    transcript (keep listing, globbing or grepping the directory flagged); tell agents to redirect large output to
    scratch/; launch test agents outside the operator's session tree; and have the user delete the operator dumps.
- **H4 (design, not a fault): the 96-token window is an ungated counterfactual channel in T2.** At least 7 of 20 agents
  used truncated sample texts as a within-class control for end-of-text cues (6 say so; [check C5] found a 7th in
  code), and it fed null calls as well as naming. All also read latents, so the outcomes are legitimate, and grades
  show no effect (end-of-text margins are the same as in T1). But a black-box profile could use it to name end-of-text
  cues or to call nulls on them, and SPEC shortcut (1) no longer holds for them. Suggested: add a truncation-contrast
  attack, for naming and for null calls, to the gates on the 29 still-unexposed T2new_kept_v9 instances before
  restating the T2 gate claim. The
  fp_upper_bound oracle (perfect names, fingerprint null calls, top-|g| latents: 17% on T2) bounds what naming alone can
  add.
- **H5 (prompt wording): the E7 exec example breaks when helper files are chained.** One agent used the documented
  `.split('if __name__')[0]` form; when helper files exec one another, each file's own first line contains the marker,
  so the next cut lands inside that line (two SyntaxErrors). It switched to `chr(10)+'#MAIN'`, which R9 flags (H1 c).
  Suggested prompt addition: the marker should be one string literal that does not occur in the exec line, for example
  `'\n#MAIN'`.
- **H6 (wording, Haiku only):** "the fraction is fixed per attribute, between 0.15 and 0.6" was read by 4 of 10 Haiku
  agents as a 15% bar or a 15-60% target window; no Opus agent misread it. Optional: "a minimum fraction, set per
  attribute somewhere in 0.15-0.6; removing more is fine as long as the probe still reads the topic".
- **H7 (E4/F3, unchanged):** the workflow wrapper gives every test agent the repo root as cwd, main's recent commit
  subjects, the account email, the shared scratchpad path and the parent's tool list. No command or visible text in
  the 30 episodes used any of it; every T2 command is anchored in its own sandbox. The shared scratchpad path it
  passes held answer-bearing T1 files during the Haiku arm ([check C3], under H3).
- **H8 (display only):** the orchestrator's copies of several agents' final reports were cut mid-sentence; the
  transcripts hold the complete text (stop_reason end_turn).

## Verdict against the criteria

| criterion | T2, Opus 5.5 | T1, Haiku 4.5 |
|---|---|---|
| mid-band pass rate (10-70%) | 75% [53%, 89%] all graded; 79% without the rule-break; 90% valid-only | 0% [0%, 28%] |
| | **borderline, not shown: at the upper edge; T1 was 95%. On a v4.2-filtered pool; about 65% estimated unfiltered, with the added failures bar-decided [check C1]** | **not met: at the floor** |
| difficulty comes from understanding, not knobs | Naming, decoys and nulls are saturated (52/52, 0/18; 20/20 pass with the bar at 0). Every failure is a real interpretation error in latent choice (context-spread latents dropped, k left unused), but which ones fail is set by the bar: failures sit on the slots with the tightest tau, and +-0.02 on tau moves the rate between 14/20 and 17/20 | Yes, entirely: the failures are method discovery (co-occurrence naming, no counterfactuals, no null calls) and survive a bar of 0 |
| | **partly: understanding decides who misses, the bar decides how many** | **met** |
| recipes and shortcuts low | gates v9: every non-reference solver <= 3/39 = 7.7% (Wilson upper 20%) on kept (<= 3/53 unfiltered); the truncation channel (H4) is untested, for naming and for null calls [check C5]. Agents track the label-free reference script closely (margin difference median +0.005) | gates v8: 0/16 for every recipe and attack; Haiku's recipe-like play scored 0/10 |
| | **met at the 10% bar, with H4 open** | **met** |
| a model-size dial | - | Opus 5.5 19/20 vs Haiku 4.5 0/10 on the same instances (Fisher p = 4e-7); slots 69/70 vs 3/35 |
| | | **strong separation, two points; a graded dial needs a middle model** |

**Overall.** The environment is clean on the grading side: 30/30 regrades match, no environment fault changed a grade,
and the work that passes is real latent interpretation (every T2 agent made 4-35 dashboard or per-token calls).
But neither arm gives a mid-band training tier for its model:
- **T2 for Opus 5.5** lowers the pass rate from 95% to 75% (Fisher p = 0.18 at these sizes), and only through removal
  near the bar. The designed difficulty, reading latents to name the cue and reject decoys, is solved every time.
  Outcomes are instance-determined (1 split in 10; margins r = 0.95), so GRPO would see within-group variance on only a
  few instances. The 75% is on a pool filtered by a reference the agents track; an unfiltered T2 pool would probably
  land inside the band (about 65%, a model estimate), but by more bar-decided misses, not by more interpretation
  errors [check C1].
- **T1 for Haiku 4.5** is at the floor (0/10, and 0/10 even with no removal bar), so binary reward gives no signal.
- The model gap on T1 is large and comes from understanding, which makes ShiftHunt a candidate model-strength probe.
  A middle model (Sonnet-class) on T1 and T2 would show whether that is a graded dial and whether one of the tiers is
  mid-band for it.
- H1 (45% false INVALIDs in T2) must be fixed before any RL use.

**Suggested next steps (for the orchestrator).**
1. Fix H1 (patterns a-j, with twins), re-finish the 10 affected episodes with `--transcript`, re-run margins.py.
   Consider parsing Python regions instead of adding regex exemptions. Widen the false-negative fix [check C4]: apply
   the R4 client names and the R5 process names to subprocess, os.system and os.popen arguments and to Python-written
   files, flag `socket` connections, and add twins for each shape; keep the Python-string and comment exemptions
   (h-j) for everything else.
2. Run a truncation-contrast attack, for naming and for null calls, in the gates on the 29 unexposed T2new_kept_v9
   instances (H4, [check C5]); fix the E7 example (H5).
2a. [check C1] Report T2 without the keep filter: draw the next T2 plan from all built instances, or from a fresh
   build with no reference keep filter. Keep v4.2's one-shot result as a reported covariate, not a filter, and state
   reachability with the oracle top 20 (C2).
2b. [check C3] Before any further LLM run, clear both shared directories (the session scratchpad and tool-results)
   and launch test agents from a session whose shared directories hold nothing about the run's instances.
3. Model dial: run a Sonnet-class agent on fresh T1 and T2 instances (T1b_kept has 6 unexposed left; a new T1 pool is
   needed for 10), with predictions written first.
4. If T2 should be harder for Opus without moving the bar: candidates whose latents are mostly context-spread, or decoys
   that share latents with a listed cue, so that token-local selection fails more often and the label-free reference
   still clears the bar. Re-run the gates and write new predictions first. Do not move tau (C9 applies here too).
5. Done in this commit: the 20 plan instances are in exposed_instances.json (`llm_seen`); plan.py and prepare now refuse
   them.
6. User actions: delete the 21 Haiku files in /tmp (H2) and the operator dumps in the session's tool-results directory
   (H3). Operators should write large views to a file under runs/ and Read it in parts instead of printing them.
   [check C3] Also delete, in the session's `tool-results/`: `b2rrr71w2.txt` and `bod78vuyi.txt` (T1-phase operator
   dumps), `b3v32n0en.txt` and `bv3dljnuf.txt` (two harness-saved copies of this file, committed aggregate text only);
   and in the session scratchpad: `s1/`, `chk_t1/`, the two `m_<episode id>/` directories, and `tx_epea.txt`,
   `ep450_tr.txt`, `cmd15.txt`, `m1.py`, `ep_margin.py`, `twins.json`.

## Files

- **Committed (task/shifthunt, not pushed):** this file and `exposed_instances.json` (+20 ids under `llm_seen`).
- **Uncommitted, per D6:** both run dirs (config.json, episodes/*: grade.json, submission.json, audit.json, tool logs,
  transcripts).
- **Review scratch:** scripts and outputs lived in a gitignored directory under runs/shifthunt/ and were deleted. No file
  was written outside runs/ and tasks/; no canary and no runs/.episodes record was printed. Files in the session's
  tool-results directory were only grepped for episode ids, never printed.
- **Amendment (check corrections C1-C7):** applied in place in a later local commit; only this file changed. One
  operator view of this file went over the output size limit and Claude Code saved it to the session's tool-results
  directory (`bv3dljnuf.txt`, committed aggregate text only; listed for deletion above).
