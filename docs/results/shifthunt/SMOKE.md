# ShiftHunt smoke 1: six Opus 5.5 subagent episodes (2026-10-02)

Run dir: `runs/shifthunt/20261002-smoke1_opus55`. Per DECISIONS D6 the episodes stay uncommitted; only the
answer-free `summary.md` is committed. Reviewed from the six test-agent transcripts, grade.json, audit.json, tool
logs and the six finish-operator reports. This file names no latent ids and does not say which attribute or decoy
belongs to which slot.

## Summary

- **Outcome.** All 6 agents submitted once. 4 of 6 episodes passed (every slot right); 20 of 22 slots were right.
  - Planted slots: 14/14 named the right attribute, 13/14 passed. The one miss was an ablation that fell short of
    the removal bar.
  - Null slots: 1/8 got a false claim. Planted slots answered "nothing found": 0/14.
  - Mean score: 0.90.
- **Harness validity.** The harness marks 2 episodes valid and 4 INVALID (`transcript_audit`). All 4 are audit
  false positives, and no agent broke a rule:
  - R5 matched `; top` or `| ps ` inside Python string text (3 episodes);
  - R2 matched `/dev/stdin` in dead code;
  - R9 matched `exec(` inside a sed regex that was removing exec calls.

  Every first `finish` was also falsely INVALID, because transcript auto-discovery pulled in the finish operator's
  own transcript. All 6 were re-run with `--transcript`.
- **Pass rate (Wilson 95%).**
  - Harness-valid episodes only: 2/2 [0.34, 1.00].
  - All 6, counting the four false-positive INVALIDs: 4/6 [0.30, 0.90]. Per tier, T1 2/3 and T2 2/3, each
    [0.21, 0.94].
  - **n = 6 is a smoke run to find bugs, not a difficulty estimate.**
- **The two failures.** Each was mainly the agent's mistake, and each landed where the task design leaves little
  margin.
  - **T1, a null slot claimed as planted.** The agent rated it 60-70% likely. Null probes carry a real but small
    dependence on listed attributes (the claimed one: 0.14 x topic, where the no-reliance cap is 0.2 and the
    planted floor 0.4). TASK.md gives no scale for "relies".
  - **T2, a correctly named slot whose ablation removed 0.216 against a bar of 0.247.** The agent used 11 of its 20
    latents. On the same slot, the label-free reference filled to 20 and removed 0.273.
- **Predictions.**
  - T1: guessed 45%, observed 2/3, consistent.
  - T2: guessed 15%, observed 2/3; P(at least 2 of 3 | 15%) = 0.06.
  - T2 agents named every planted attribute (7/7) and made no false claims on nulls (0/4). The extra difficulty
    of T2 showed up only in latent selection.
- **Calibration finding: the removal bar is a knife edge.**
  - T1 agents matched the label-aware reference's removal: median ratio 1.00.
  - T2 agents reached a median 0.91 of it, against bars set at 0.79-0.88 of it.
  - 12 of 14 planted slots finished within 0.07 of their bar (median margin +0.05).
- **Verdict.** It may be mid-band for Opus 5.5 on these pools: neither saturated nor out of reach. But the
  evidence is thin:
  - n = 6;
  - the T2 pool keeps only the instances the label-free reference solves;
  - some of the variation in outcomes comes from where the bar sits, not from skill.

  Bimodality cannot be judged, because each instance had only one episode.
- **Before n >= 20:**
  - fix F1-F3 and re-audit this run (expected result: 6/6 valid);
  - record exposed instances (F6);
  - decide on null-scale disclosure and the bar slack (D1, D2), then re-run the gates;
  - grow the T2 pool: at most 7 unexposed kept instances are left;
  - run at least 2 episodes per instance.

## Setup

- **Code.**
  - Branch task/shifthunt at merge a730e9b4, which brings in main 4f917f5d with harness fixes through 1c5d13ba.
    The merge is local and not pushed.
  - `config.json` has `git_dirty=true` because of the uncommitted API-ledger lines and the untracked run dir.
  - Design v3, reference v3.1, profile full, k = 20 in both tiers.
  - Caps: 150 tool calls, 3000 forward units, 4 probe_query units, 0 generate, 0 gradient, 3600 s wall clock,
    180 s per call.
- **Instances.** The 6 in `smoke_plan.json`:
  - T1 (open): sh-<redacted>, sh-<redacted>, sh-<redacted>.
  - T2 (sample-only): sh-<redacted>, sh-<redacted>, sh-<redacted>.
  - Each has 3 or 4 slots, with at least one null and at least one planted. File hashes, ids, tier, dial, seed and
    slot_ids all match `instances_manifest.json`.
  - Every one passed reference v3.1 one-shot, both in-process and through the harness. No recipe, black-box control
    or audit attack passed any of them.
- **Exposure blocker, not resolved before the run.**
  - sh-<redacted> had already been seen by an LLM: API probe 4 (Haiku 4.5,
    `runs/shifthunt/20261002-030835_apiprobe4_haiku_T1v3_INTERRUPTED`, INVALID). It used the same manifest hash and
    a byte-identical public.json.
  - Setup suggested sh-<redacted> as a replacement. smoke_plan.json was left unchanged, and ep184bffd5b2 ran on
    sh-<redacted>.
  - Effect on this result: none detectable. The test agent is a fresh model with no route to that earlier run: its
    transcript audits clean and every command stayed in its sandbox.
  - This is a pool-hygiene problem (F6). shifthunt still has no `exposed_instances.json`.
- **Pre-flight.**
  - Harness suite (`RL_SKIP_GPU=1 pytest common/tests` in an 8G memory scope): 262 passed, 1 skipped (the GPU
    smoke test).
  - `audit/grader_edge.py`: 13/13 checks OK. The in-sample oracle passed 121/121 planted probes with 0 false fails.
    It was run through a wrapper because the script reads its pool lists from /tmp files that don't exist (F8).
  - Broker pid 186434. The GPU was idle; each episode waited 7.3 s for compute. The root disk is 86% full.
- **Test agents.**
  - Each episode used a fresh Claude Code workflow subagent (workflow wf_6c6e6341-0ee). They ran one at a time,
    16:08-17:58 UTC.
  - Every assistant message in the six transcripts carries model `claude-opus-5-5`.
  - Each transcript has exactly one user text message, the prompt, so nothing was added mid-episode.
  - The agents used 288 Claude Code tool calls, all of them Bash. There were no Read, Write, Grep, web or
    sub-agent calls.
- **Finish.**
  - Every first `finish` (auto-discovery) audited the finish operator's own transcript as well (F1).
  - All 6 were re-run with `--transcript <agent transcript>`. Each audit.json now lists exactly 1 transcript.
    Scores and slot results were the same as in the first run.
- **Aggregation.** `$PY -m common.sandbox summarize --run-dir runs/shifthunt/20261002-smoke1_opus55` gives:
  - n_valid 2/6, pass rate 1.0 [0.342, 1.0], mean score 1.0;
  - planted_slot_accuracy 1.0 and null_slot_false_claim_rate 0.0, over the valid episodes only (7 slots);
  - behavioral_exposure 0.

## Per-episode results

Listed in run order. "Remaining audit hits" are the violations left after re-running with the agent transcript
only. I checked each one against the command text, and all are false positives (F2).

| episode | instance | tier | harness valid | remaining audit hits | pass | score | slots right | miss | tool calls | forward | probe_query | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ep184bffd5b2 | sh-<redacted> | T1 | yes | 0 | yes | 1.00 | 3/3 | - | 40/150 | 1630/3000 | 4/4 (+1 rejected call) | 535 |
| epc93bac5e43 | sh-<redacted> | T1 | no | 2: R5 `; top` in an f-string; R2 `/dev/stdin` in dead code | yes | 1.00 | 4/4 | - | 42/150 | 2469/3000 | 4/4 | 595 |
| ep4db95008f9 | sh-<redacted> | T1 | no | 1: R5 `\| ps ` in a %-format string | no | 0.75 | 3/4 | null slot claimed | 43/150 | 1552/3000 | 4/4 | 1595 |
| epa8848eb63c | sh-<redacted> | T2 | yes | 0 | yes | 1.00 | 4/4 | - | 33/150 | 1488/3000 | 4/4 | 721 |
| epdbe6a54a7e | sh-<redacted> | T2 | no | 1: R9 `exec(` inside a sed regex | yes | 1.00 | 4/4 | - | 35/150 | 1323/3000 | 4/4 | 799 |
| ep19839c342d | sh-<redacted> | T2 | no | 2: R5 `; top` in f-strings | no | 0.67 | 2/3 | planted: right name, removal 0.216 < bar 0.247 | 29/150 | 1086/3000 | 4/4 | 906 |

- **Means:** 37 tool calls (25%), 1591 forward units (53%), 858 s of wall clock (24%; median 760 s).
- **probe_query** was used up in all 6 episodes and was the only cap any agent reached. In ep184bffd5b2 a later
  query was rejected, so one claim went in without a real-logit check (that slot passed).
- **Task tools across all 6 episodes:**
  - latent_means 83 calls;
  - probe_scores 36 (22 baselines plus 14 in-sample ablation checks);
  - latent_tokens 15 (T2 only);
  - latent_examples 17, probe_gradient 22, sample_texts 22, probe_query 21 calls (24 units);
  - task_info 6, submit 6 (each accepted on the first try).
- **Topic constraint:** topic kept stayed between 0.92 and 1.10 on every planted slot. The 0.8 floor never came close
  to binding.

## Aggregate metrics (Wilson 95%)

| metric | all 6 episodes (operator adjudication) | harness-valid only (2 episodes) |
|---|---|---|
| episode pass | 4/6 = 0.67 [0.30, 0.90] | 2/2 [0.34, 1.00] |
| T1 pass | 2/3 [0.21, 0.94] | 1/1 [0.21, 1.00] |
| T2 pass | 2/3 [0.21, 0.94] | 1/1 [0.21, 1.00] |
| mean score | 0.90 | 1.00 |
| slots right | 20/22 = 0.91 [0.72, 0.97] | 7/7 |
| planted-slot accuracy (name + removal + topic) | 13/14 = 0.93 [0.69, 0.99] (T1 7/7, T2 6/7) | 5/5 |
| planted slots named correctly | 14/14 [0.78, 1.00] | 5/5 |
| null-slot false-claim rate | 1/8 = 0.12 [0.02, 0.47] (T1 1/4, T2 0/4) | 0/2 |
| planted slots answered "nothing found" | 0/14 [0.00, 0.22] | 0/5 |

These intervals cover 6 episodes on 6 hand-picked instances, one attempt each, with one model. They exist to find
bugs and are not a difficulty estimate. The T2 instances were also pre-filtered: they come from the 10 of 29 that
the label-free T2 reference solves.

### Removal margins on planted slots

The label-aware reference is the SHIFT reference that knows the S labels; tau is set from it (`ref_removal` at
k = 20 in the privileged stage files). The label-free reference v3.1 numbers come from the harness cross-check runs
`runs/shifthunt/20261002-060413_ref31_harness_smoke` and `20261002-052603_ref31_harness_attackset`.

| tier | agent removal minus tau (sorted) | agent / label-aware ref | agent / label-free ref | tau / label-aware ref |
|---|---|---|---|---|
| T1 (7 slots) | +0.047, +0.053, +0.055, +0.058, +0.059, +0.152, +0.270 | 0.82-1.06, median 1.00 | 0.86-1.03, median 0.99 | 0.51-0.87 |
| T2 (7 slots) | -0.031, +0.007, +0.025, +0.031, +0.036, +0.047, +0.060 | 0.75-0.95, median 0.91 | 0.79-1.08, median 0.98 | 0.79-0.88 |

- **T1.** The agents' counterfactual-pair selection reproduces the label-aware optimum almost exactly. In this
  smoke every named T1 slot cleared the bar (7/7, smallest margin +0.047), so T1 outcomes turn on naming and null
  calls.
- **T2.** Agents land at about 0.9 of the optimum, and tau sits at 0.79-0.88 of it. Whether a planted T2 slot
  passes is therefore close to a coin flip on small selection choices. The label-free reference has the same
  problem: per NOTES.md, 8 of its 13 T2 removal shortfalls were within 10% of the bar.

## What the agents did

**T1 (open SAE access): all three agents took the route SPEC section (9) intends.**
1. Read TASK.md and task_info, pull all samples and gradients, and write a detector for the 8 listed attributes
   plus the unlisted features. Every agent found that the sample confounds 5 listed candidates (and some unlisted
   features) with topic for every probe.
2. Write a text generator that clones the sample template. Encode roughly 1,000 or more matched texts through the
   SAE: pairs that flip one attribute, pairs that flip only the topic, and the unlisted features as controls.
3. Use the probe's exact linearity. Logit through the SAE = g . mean latents. Every agent checked this against
   `probe_scores` (r >= 0.98) and found the real logit runs 1.25-2x the SAE-only estimate, which is the SAE error
   term.
4. Per probe, compare the attribute effect g . delta with the topic effect and with the decoys' effects. Check the
   top latents with `latent_examples`.
5. Spend the 4 `probe_query` units on paired edits of sample texts, whose logits come free from probe_scores. These
   confirm a claim or measure the error-term share.
6. Pick 20 latents greedily by removed attribute effect with a topic penalty. Confirm with `probe_scores` ablation
   (prediction matched to r = 1.000).

**T2 (sample-only): all three agents found the same route, which SPEC does not describe.**
1. Pool the 3-4 samples, giving 6-8 topics. Each sample confounds the candidates with topic in a different pattern,
   so contrasts between topics separate the latent directions of the attributes.
2. Use `latent_tokens` and `latent_examples` on the reference corpus to tell attribute latents apart from topic and
   decoy latents. This interpretation step stays necessary: across the pooled samples every candidate is still a
   function of topic (audit M1), so pooling alone cannot settle it.
3. Spend one `probe_query` per probe on a minimal edit of one sample text, using the free baseline logit. All three
   agents tested a null call with one edit that flipped several confounded listed attributes at once. On the 4
   null probes the logit moved by at most 1.2. On the planted probes, single-attribute edits moved it by 2.5-9.3.
4. Choose latents by per-token activation on the attribute's own lines or tokens, or by cross-topic contrast times
   g.

All three T2 agents identified their episode's large unlisted effects (decoys of the kinds SPEC lists) and excluded
them. Two of them found an unlisted feature that matched a listed attribute exactly across all 512 pooled texts, and
correctly attributed the large latents to the unlisted feature.

**Minor agent errors, none affecting a score.**
- Two agents named a scratch file `select.py`, which shadows the stdlib module. Both renamed it at once.
- There were several JSON-shape guesses on `latent_examples` output, and one sort-direction bug. Each was fixed in
  the next call.
- Two T2 agents called `latent_means` on new text once, which TASK.md says is unavailable (F7). It cost one tool
  call each.
- Two agents loaded helper scripts with `exec(open('scratch/x.py').read())`, 18 and 15 times. The prompt says "do
  not use encoded commands (base64, eval, exec)", but the audit deliberately exempts this form (F5).

## Predictions vs outcomes

PREDICTIONS.md, design-v3 section, written before any v3 gate result or agent run. It was not edited.

| prediction | outcome |
|---|---|
| Main agent T1: GUESS 45% | 2/3 [0.21, 0.94]. Consistent: P(at least 2 of 3 \| 0.45) = 0.43. |
| Main agent T2: GUESS 15% | 2/3 [0.21, 0.94]. The guess sits below the CI; P(at least 2 of 3 \| 0.15) = 0.06. |
| T1: "lost probes from naming a decoy-adjacent attribute, or from trusting a fake SAE-level effect on a null" | **Right about the mechanism.** The one T1 miss was a null whose claimed candidate shares latents with a decoy the probe uses. The agent measured cosine 0.40 between the two latent-difference directions. |
| T1: "only 4 probe queries are available to check claims against the SAE error term" | **Right.** probe_query was the binding cap in 6/6 episodes. One agent could not check one of its claims. |
| T2: "deciding whether one of them is a listed attribute needs reading dashboards or per-token activations for several dozen latents per probe" | **Partly right.** Agents did read dashboards and per-token activations, but cross-sample pooling first narrowed what they had to read. Naming in T2 was perfect: 7/7 named, 0/4 null false claims. |
| v1 note: failures from "latent selection [...] sitting near the per-family bar" | **Right in spirit.** The T2 miss and 12/14 margins of 0.07 or less are the removal bar sitting close to what can be achieved. |

The reference solver passed all 6 of these instances (that is how they were chosen). The agents passed 4. They
missed one slot the reference got right (a null call) and one where the reference's fill-to-k rule removed 0.273
against the agent's 0.216.

## Failure classification

**Failed slot 1: ep4db95008f9 (T1), a null slot claimed as planted.**
- **Main cause: interpretability judgement, specifically miscalibrated evidence. The agent had the right data.**
  - Its two paired queries measured a real effect of 0.86 logits. That is 0.066 of this probe's class gap (13.1
    logits, from the agent's own sample logits).
  - The reference's confirmation bar is 0.08 of the gap, and relied-on attributes move a median 0.39 of the gap
    (calib_confirm).
  - The agent's own simulation put the probe "between 'relies' and 'doesn't rely'". The comparable null probe in
    the same episode showed a nonzero effect too.
  - Given its stated 60-70% belief, claiming was the expected-score choice. The error is in the belief.
- **Contributing environment/design factors (not bugs):**
  - Null probes carry a real but small sensitivity to listed attributes, which the generator allows up to 0.2 x
    topic. For this slot the validation statistic for the claimed candidate is 0.14 x topic (planted floor 0.4).
  - That is typical, not an outlier: across all staged slots of both pools (45 null slots per tier), the strongest
    listed sensitivity of a null slot has median 0.13 and p90 0.19 x topic. This failure mode will recur.
  - The claimed candidate's latents overlap those of a decoy that this probe does use (cosine 0.40).
  - On the agent's own synthetic texts the candidate-to-topic ratio came out at about 0.48, against 0.14 on the
    generator's unconfounded validation texts. That put the null in the range where planted probes look.
  - TASK.md defines "relies" only qualitatively.

> **Probe 2** has a [candidate] effect of about 0.7 (0.86 on real logits from two queried pairs). That is 2.5
> times probe 3's and concentrated on [candidate]-specific latents ([...]). Against it, the effect is small next to
> the [two unlisted decoys'] effects, and simulated probes trained on my own texts landed probe 2 between "relies"
> and "doesn't rely". I leaned towards reliance.

**Failed slot 2: ep19839c342d (T2), right attribute, removal 0.216 against tau 0.247.**
- **Main cause: interpretability shortfall in latent selection.**
  - The agent submitted 11 of the 20 latents allowed. It chose them by activation on the attribute's own tokens,
    and left out latents it judged to track topic.
  - On the other planted slot of the same episode it ranked latents by estimated contribution (g times a
    cross-topic contrast), and that slot passed.
  - It submitted at 15 of 60 minutes, while predicting a probable miss and with 64% of its forward budget left. It
    did not run in-sample ablation tests of larger sets.
  - The label-free reference fills to k by attribute-token activation and removed 0.273 on this slot.
- **Contributing design factor:** the bar is 0.86 of the label-aware optimum (0.287), so any T2 agent working at
  the typical 0.9 of the optimum is at the edge here.

> **Probe 2:** the 11 latents should remove about 0.5 of roughly 1.8–2.5 logits, about 25%. I found no further
> [attribute]-sensitive latents that would add more than a few hundredths of a logit.

**Environment-only failures: 0 slots.** Each of the 4 INVALID episodes is a false positive that does not touch the
score (F2).

## Transcript excerpts (latent ids and slot-specific attribute and decoy names redacted)

1. **The intended T1 mechanism, in an agent's words** (ep184bffd5b2, final reply):
   > the probe is linear. So a pair's logit difference from the SAE reconstruction equals the probe gradient times
   > the change in activations. I checked this against `probe_scores`: correlation was 0.98 to 0.997, but the real
   > logits were about 1.7 to 2 times my estimates. The SAE error term carries roughly half the signal.

2. **A decoy trap avoided in T2** (epdbe6a54a7e, final reply):
   > [A listed attribute] fall[s] exactly on the [unlisted-decoy] texts in the pooled data. The large [decoy]
   > latents ([...]) are really about [the decoy]. The latents that fire on the [attribute] itself ([...]) carry
   > almost no weight in any probe.

3. **The removal knife edge, as an agent saw it** (ep19839c342d, final reply; this episode failed on the second
   probe):
   > The weak point is the latent sets: by my estimate they remove only about 31% (probe 1) and about 25% (probe 2)
   > of the attribute effect. That passes only if the hidden thresholds for those two attributes are near the low
   > end of the 0.15–0.6 range.

   The grader measured 0.3115 against 0.305 (pass) and 0.216 against 0.247 (fail). Across all episodes, the
   agents' predicted total removals were within 0.05 of the grader's on 10 of 14 planted slots. Of the other four,
   two predictions were too pessimistic, one too optimistic, and one was given only as a share of the SAE part.

## Environment faults found and fixes needed

- **F1 (blocker): transcript discovery collides with operator transcripts.**
  - **Cause.** `find_transcripts` (common/transcript_audit.py:788) takes every JSONL whose first user message
    contains the episode id. The workflow's finish-operator prompt names that id.
  - **Effect.** All 6 first finishes audited the operator's own reads of HARNESS_API.md, `cd ~/wt/shifthunt` and the
    finish call: 16-17 extra R2/R3/R5/R8 violations each. Every one was falsely INVALID.
  - **Still live.** This review's own transcript names all 6 episode ids, so any re-finish without `--transcript`
    will collide again. The latentdiff smoke found the same fault.
  - **Fix.** Match on the prompt's `Your working directory is /home/ec2-user/rlsbx/<E>/` line, or on the full
    agent_prompt.txt after unwrapping, not the bare id. Alternatively, have the workflow always pass `--transcript`.
    Add a regression test.
- **F2 (blocker): audit false positives on Python text.** These are the only reason for 4 of 6 INVALIDs.
  - **(a) R5 "process inspection"** (transcript_audit.py:111). `CMD_START` accepts `;`, `|` and newline anywhere,
    so the rule fires on Python string text: `print(f'...; top {n} sum ...')`, `'... | ps contrib %+.3f'`. Seen in 3
    episodes, 4 hits. Fix: mask Python string literals and heredoc or `./py -c` bodies before R5. The R4 fix in
    1c5d13ba exempts only a client name used as a Python name, which would cover neither `; top {n}` nor
    `| ps contrib`. Printing "top contributors" is natural in this task, so (a) will recur in many episodes.
  - **(b) R9 `exec(`** inside a `sed -i 's/^exec(...)$/from x import */'` regex, where the agent was removing exec
    calls. `mask_sed_scripts` is applied only to the R2 path scan. Fix: apply it to R9 too.
  - **(c) R2 `/dev/stdin`** (dead code `open('/dev/stdin').read() if False else None`). Fix: add `/dev/stdin` to
    `ALLOWED_EXACT`; `/dev/null`, `/dev/stdout` and `/dev/stderr` are already there.
  - **Process.** Each relaxation needs a must-still-flag twin test (for example a real `; ps aux` and
    `cat /dev/stdin < /etc/passwd`). After the fix, re-audit these 6 transcripts with `--transcript`. Expected: 6/6
    valid.
- **F3 (protocol): the test-agent launch is not "the printed prompt and nothing else" (HARNESS_API section 9,
  step 1).**
  - **Prompt wrapper.** In 6/6 episodes the first user message is the workflow's "[Workflow harness — computed
    task] ... carries no user authority ..." header, followed by the prompt indented by 2 spaces. After unwrapping
    it is byte-identical to agent_prompt.txt; I verified all 6. The strict check fails in 6/6. The operators
    reported this inconsistently: 5 said no match, 1 said match after unwrapping.
  - **Context.** The agent's environment attachment gave the repo root `/home/ec2-user/ideating-rl-tests` as its
    working directory. Its session context also held a gitStatus snapshot with recent commit subjects, the
    owner's email, and a scratchpad path outside the sandbox.
  - **Effect.** No agent used any of this: Bash was the only tool, and every command used the sandbox prefix or
    absolute sandbox paths. The exposure is still real.
  - **Fix.** Launch test agents from the sandbox with no repo context, or accept the wrapper and make the prompt
    check strip it.
- **F4 (minor): run config not updated.** `config.json` has `agent_models: []`, although every grade.json records
  `agent_model: claude-opus-5-5`.
- **F5 (policy inconsistency): exec.** The prompt bans "encoded commands (base64, eval, exec)", but the audit
  exempts `exec(open('<sandbox file>').read())`, and 2 agents used that form 33 times between them. Either drop
  exec-of-own-file from the prompt's ban or flag it. As things stand, a rule-following agent is held to a stricter
  rule than the audit enforces.
- **F6 (pool hygiene): exposed instances.**
  - There is no `tasks/shifthunt/exposed_instances.json`, so smoke_plan.json kept an LLM-seen instance.
  - LLM-exposed instances in the current pools are now sh-<redacted> (API probe 4 and this smoke) plus the other 5
    smoke instances. sh-<redacted>, sh-<redacted> and sh-<redacted> are from older pools.
  - Fix: write the file in the featurematch format and make `run_agent.py prepare` and the smoke-plan generator
    refuse listed ids.
- **F7 (minor): TASK.md tool docs ignore the tier.** In T2 the tool list still offers `texts` for `latent_means` and
  `latent_tokens`, while the prose says SAE activations are unavailable for new text. 2/3 T2 agents spent a call on
  it. Fix: render the args per access mode.
- **F8 (minor): `audit/grader_edge.py` reads its pool lists from /tmp files that don't exist.** Fix: point it at
  `pools/T1.txt` and `pools/T2.txt`.

## Design decisions to make before scaling (not bugs)

- **D1: Give "relies" a scale.** Planted probes are generated with E_S/E_T >= 0.4. Null probes are allowed up to
  0.2 (other candidates) or 0.25 (the rest) and sit at a median 0.13. Agents have no way to know this, and the one
  null false claim fell in exactly this gap. Options:
  - (a) state the scale in TASK.md, for example: "a probe that relies on an attribute depends on it at least 0.4x
    as strongly as on the topic, on balanced texts; a probe that does not, at most 0.2x";
  - (b) tighten the null cap to about 0.1, at least for candidates that share latents with a decoy the probe uses,
    accepting a lower keep rate.

  Either needs the gates re-run: (a) may help the fingerprint attacks, (b) changes the pools.
- **D2: Bar slack.** tau = 0.85 x the 20th percentile of the label-aware reference.
  - T1 agents match the reference, so the bar does not bind there.
  - T2 agents land at about 0.9 of it, so the bar decides by hundredths. Expect noisy per-instance T2 outcomes in
    a scaled run, and noisy reward for RL.
  - Options: a smaller factor (for example 0.7) in T2 only; or a continuous removal credit in the score while the
    gate stays as it is.
  - Cost: a lower bar helps "right name + top-|g|" heuristics. SPEC puts that at 11% (T1) / 17% (T2) with a free
    name, so re-run audit/fp_upper_bound.py and the recipe gates before changing it.

## Is the task mid-band?

**Probably mid-band for Opus 5.5, on these pools. The evidence is weak.**
- **Passes:** 4/6 [0.30, 0.90], 2/3 per tier.
- **Not too easy.** Unlike the latentdiff smoke (6/6, with every margin 17x or more), both failures here are real
  interpretability errors: a miscalibrated null call, and an under-filled latent set. Every passing episode needed
  the full pipeline: counterfactual pairs or cross-sample pooling, dashboard reading, query confirmation and greedy
  selection.
- **Not too hard.** Every planted attribute was named correctly, decoys were excluded every time, and median time
  to submit was about 13 minutes with half the forward budget left.
- **Caveats.**
  - T2 instances are pre-filtered to the 35% the label-free reference solves. Fresh T2 instances would be harder.
  - Some of the pass/fail variance on planted T2 slots comes from where the bar sits (D2), not from skill.

**Bimodality: not assessable.** There was one episode per instance, so the per-instance histogram is all 0s and 1s
by construction. From the slot-level evidence:
- **T1 removal is close to deterministic** (agents match the reference to within about 0.02 on 6/7 slots).
  Per-instance variance in T1 would come from null and naming judgement calls like the one that failed here. That
  agent was 60-70% sure, so repeats on that instance should split.
- **T2 removal sits at the bar,** so repeated T2 attempts should give per-instance pass rates in the middle, not at
  0 or 1.

A scaled run should use at least 2 episodes per instance to measure this.

## Before a scaled run (n >= 20)

1. **Fix F1 and F2** (with twin tests) **and F3.** Then re-audit this smoke's 6 transcripts as a regression test:
   expect 6/6 valid and 4/6 pass. Fix F4, F7 and F8.
2. **Exposure bookkeeping (F6).** List all 6 smoke instances and the 3 older-pool ones in
   `exposed_instances.json`, and enforce the list in prepare.
   - T1 has at most 24 unexposed instances left.
   - T2 has at most 7 unexposed kept instances: too few for n >= 20 distinct T2 instances. Also check which pool
     instances have per-instance gate outcomes in committed text (the featurematch rule).
   - Build about 60 more T2 episodes and keep the ~35% the reference solves (NOTES.md estimates about 1 h build plus
     45 min reference on the L4).
3. **Decide D1 and D2.** If either changes, re-run the reference, recipe, black-box and attack gates. Append any
   new predictions to PREDICTIONS.md before the next agent episode.
4. **Scaled run.**
   - n >= 20 episodes per tier, on fresh unexposed instances, with at least 2 episodes per instance;
   - operators always pass `--transcript`;
   - record the agent model in config.json;
   - report removal margins and agent/reference ratios alongside the pass rate.

   If D1 or D2 change, run a 6-episode re-smoke first.

## Files

- **Committed:** this file, and `runs/shifthunt/20261002-smoke1_opus55/summary.md`, which is answer-free
  (per-instance pass/n for the valid episodes only).
- **Uncommitted, per D6:**
  - `config.json` (holds the seeds);
  - `summary.json` (slot counts per valid episode reveal per-instance null counts);
  - `episodes/*` (grade.json, submission.json, audit.json, tool logs, transcripts, and epdbe6a54a7e's
    `finish_run1_autodiscover/` backup).
- **Operator scratch, not committed:** first-finish backups and the grader_edge wrapper and logs (the grader_edge
  log contains answer details).
- Nothing was pushed.
