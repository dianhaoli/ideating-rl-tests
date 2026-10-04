# ShiftHunt T1: why the small models fail (2026-10-03)

Diagnostic note. It does not change any pass rate, gate or validation claim. Aggregates only (D6). The sources are
`grade.json` from the four T1 arms on the same 10 instances, the scripted stage analysis (`_diag/stage_quant/`),
an independent re-count (`_diag/smallmodel_check/tab.py`), the transcript forensics, and two luna diagnostic arms.
Both diagnostic arms were still incomplete when this was written (see below).

## Bottom line

- **Haiku 4.5 has no method.** It names attributes from co-occurrence in the sample, claims an attribute on every
  null probe, and almost never measures the probe on texts it wrote itself. It fails at every stage.
- **gpt-6-luna has the method but under-uses it.** It names well and mostly avoids null claims. It then takes a few
  latents (median 4 of 20) and never checks how much of the attribute effect they remove. With removal fixed and
  nothing else changed, it would pass 6/10.
- **A one-paragraph method hint fixes luna's removal step** (8/9 named slots clear the bar, 20 latents each). It does
  not yet produce passes (0/5 so far): the hint's "none below the scale" rule now makes luna miss planted cues
  (3 of 12). **Effort high, without the hint, changes little** (1/7 named slots clear; median 6 latents).
- **No environment fault singles out the small models.** One harness gap (workflow-wrapped transcripts) only
  affects audit validity, not grades.

## Stage table (10 T1 instances; 22 planted and 13 null slots per 10 episodes)

| arm | pass | named right | null false claims | named slots clearing bar | latents per named slot (median) | removal / label-free ref (median) |
|---|---|---|---|---|---|---|
| Opus 5.5 (20 eps) | 19/20 | 44/44 | 0/26 | 43/44 | 20 | 1.00 |
| gpt-6.1-sol | 10/10 | 22/22 | 0/13 | 22/22 | 20 | 1.00 |
| gpt-6-luna (medium) | 0/10 | 19/22 | 5/13 | 2/19 | 4 | 0.48 |
| Haiku 4.5 | 0/10 | 7/22 | 13/13 | 3/7 | 12 | 0.68 |

All counts were re-derived from `grade.json` and match the stage analysis. With the bar at 0 (naming and nulls
right), luna passes 6/10 and Haiku 0/10. The latents luna does pick are the right ones: 81 of 92 (88%) also appear
in the large models' sets.

## Why Haiku fails

- **Naming.** It treats a clean split between attribute and label in the 128-text sample as proof. TASK.md says
  this co-occurrence cannot decide which cue the probe relies on. Result: 9 wrong names and 6 misses out of 22.
- **Nulls.** It never weighs "none" against the disclosed ratio scale, so it claimed a cue on all 13 null probes.
  In one episode its own matched pair showed no greeting effect (7.478 vs 7.454), and it still claimed greeting on
  every probe.
- **No own-text measurements.** 2 of 10 episodes ever encoded texts the agent wrote. 1 of 10 built 5 or more
  matched pairs. It treated the 4-unit `probe_query` as its only experiment. It also ran the removal check on the
  confounded sample.
- **It stops early.** It used a median of 27 of 150 tool calls and about 5.4 minutes of 60, then declared success.
- **Tool slips.** It piped JSON into a heredoc, used system python3 without numpy, and passed latent ids as text ids.
  Four episodes wrote to `/tmp` (a rule break), which with one process-inspection call makes 5/10 harness-INVALID.

## Why luna fails

- It finds the counterfactual route (8/10 episodes built 5 or more matched pairs) and names 19/22.
- It then picks the top few latents by contribution and checks only that the topic effect is kept. It never
  verifies the share of the attribute effect removed: 3/10 episodes computed any removal fraction. Where it did
  estimate removal, the estimate from a few short pairs ran 2-3x too high.
- The bar is hidden (TASK.md gives only 0.15-0.6), and luna does not aim for the top of that range.
- 5 of 13 null probes got a claim, after thin single-pair checks.
- It writes no visible text, so every reason above is read from its code.

## Diagnostic arms (pre-registered in `diag/PREDICTIONS_DIAG.md`)

| quantity | A. luna high, no hint (n = 4 of 10) | B. luna medium + hint (n = 5 of 10) | luna medium baseline (10) |
|---|---|---|---|
| episode pass | 0/4 (pred. 10%) | 0/5 (pred. 40%) | 0/10 |
| planted named right | 7/9 (2 missed) | 9/12 (3 missed as "none") | 19/22 |
| null false claims | 2/4 | 1/5 | 5/13 |
| named slots clearing bar | 1/7 (pred. 30%) | 8/9 (pred. 80%) | 2/19 |
| latents per named slot, median | 6 (pred. 6) | 20 (pred. 20) | 4 |
| removal / label-free ref, median | 0.66 (pred. 0.60) | 0.93 (pred. 0.95) | 0.48 |
| cost, episodes done | $0.23 (median 16 min) | $0.13 (median 3 min) | $0.28 for 10 |

**What they show (provisional, small n):**
- **Effort is not the explanation.** At high effort luna still takes a few latents and stays below the bar. This
  supports the "strategy choice, not reasoning budget" basis for arm A. The medium-effort caveat in VALIDATION.md
  matters little for T1.
- **Execution is not the bottleneck; knowing the procedure is.** Told the procedure, luna runs it correctly and fast.
  Removal goes from 2/19 to 8/9 named slots clearing the bar, and from 0.48 to 0.93 of the reference. The residual
  failure has moved to the detection step: 3 planted cues called "none", where baseline luna missed 3 of 22.
  The episode product (every slot right) still gives 0 so far. The 40% guess for B looks too high unless the
  missing 5 episodes run differently.
- **Haiku + hint, n = 1 anecdote (arm dropped).** Both planted slots named right with 20 latents, both nulls right,
  but removal 0.95x and 0.86x of the bar: fail. That is much better staging than any no-hint Haiku episode.
  Its validity is a harness gap (below).

## Environment faults

- None selectively breaks the small models. No arm, diagnostic arms included, hit the tool-call, forward, turn,
  wall-clock or per-episode dollar caps.
- **Workflow-wrapped transcripts are not auditable.** The Haiku-hint subagent's first message is the new
  "user request relay" wrapper. `finish` found no transcript by search (R0-no-transcript). With the transcript
  given explicitly it reports `R0-prompt-mismatch`. The grade is the same either way; only validity is affected.
  The audit should learn this wrapper before any further Claude-subagent arms.
- TASK.md's "held-out texts" wording vs `probe_scores` on the confounded sample is easy for weak models to
  misread (Haiku checked removal on the sample).

## Implications for ShiftHunt

- **T1 separates models by procedure discovery, not by capacity to execute.** That is a fair reading of the 0/10s,
  and the claim should be worded that way.
- **A scaffolded small-model tier is plausible but not shown.** The hint fixes removal. It now needs a detection
  rule that does not suppress real cues (for example, say that measured ratios on agent-written texts run low, or
  show the scale's lower edge). Re-test before calling it a training tier.
- **A graded model dial is not there yet.** Without the hint the small models score 0/10 against 19/20 (Opus) and 10/10 (sol); with the hint, the small
  models move a lot per stage but not yet per episode. Removal / reference (0.48 -> 0.66 -> 0.93) is the best
  continuous signal for a dial or for reward shaping.

## Final diagnostic outcomes (2026-10-04, all 10 episodes per arm; supersedes the partial numbers above)

| gpt-6-luna arm (same 10 T1 instances) | pass | named right | null false claims | named slots clearing the bar | latents per named slot (median) |
|---|---|---|---|---|---|
| baseline (effort medium) | 0/10 | 19/22 | 5/13 | 2/19 | 4 |
| effort high | 1/10 | 20/22 | 5/13 | 4/20 | 4.5 |
| effort medium + method hint | 1/10 | 17/22 | 4/13 | 14/17 | 20 |

Effort barely moves luna: it still under-fills latents. The method hint fixes removal (it fills to 20, and 14/17 named slots clear the bar) but
naming and null calls do not improve, so only 1/10 passes. The small-model failure is a capability gap across several stages, not one fixable step.
Diagnostic spend: $0.80.
