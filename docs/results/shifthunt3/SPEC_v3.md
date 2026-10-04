# ShiftHunt v3: audit a classifier's nonlinear reliance on surface cues (spec, planner, 2026-10-04)

> **REVISION 2026-10-04 08:40Z (planner): KILLED after Stage 0. Read `KILLED.md` first.** Gate 0 passed every
> GATED item on HELD, but the audit found a no-name script (sample gradient difference) that solves 49/51 = 96% of
> OR+AND slots. Every tool-side remedy was then measured to stay scriptable: whole-sample gradient + ablate gives
> exact support on 44/44 DEV slots; fixed gradient + 2-id block test is an exact linearity test; with no gradient,
> a group-testing script solves 33% with names and 35% without (80% when P <= K_ABL). Head-side remedies only
> raise the cost. Root cause: a linear head with one sparse planted module, plus exact tools, gives the module an
> exact signature, and a closed 16-feature bank makes naming a statistic. Section 18 summarises; sections 0-17
> are kept unchanged as the record of what was tried.

Branch `task/shifthunt3` (LOCAL ONLY, never push: instances hold answers). v2 lives untouched in `tasks/shifthunt/`
here and in `~/wt/shifthunt` (read-only). Everything v3 is under `tasks/shifthunt3/`. Planner experiments:
`explore_v3/` (section 17). Timebox: 10 h from 2026-10-04T06:30Z; stop at the first failed gate.

## 0. One-paragraph design

An episode holds 3-4 classifiers ("probes", one per slot) on gemma-2-2b. Each probe = a linear topic read-out of the
mean-pooled layer-12 residual **plus a small nonlinear module that reads Gemma Scope L12-16k latents of up to two text
features ("cue components") and combines them** as `single`, `or` (either suffices, saturating) or `and` (both
needed). The module is planted with a stated reliance strength; null slots carry only a sub-bar module and sub-bar
distractors. The agent submits a structured claim per probe: verdict and probability, the cue components (from a
16-item menu in the fixed-menu arm; as a description plus minimal text pairs in the open-menu arm), the combination,
one latent group per component, one minimal edit set, and numeric predictions of the cue's 2x2 cell effects and of
the removal its own latent sets achieve. The grader scores identification, mechanism (combination, group quality,
prediction accuracy), calibration and edit quality on a hidden factorial grid, continuously, with a per-latent price.
Everything is graded exactly on CPU from stored pooled latents, because ablating latent j only (a) shifts the topic
logit by -(f_j - mu_j) gT_j with gT = W_dec w_T, and (b) sets f_j to mu_j inside the module.

## 1. Why v3 (v2 evidence, all aggregate)

| v2 finding | Source | v3 answer |
|---|---|---|
| Naming saturated: Opus 110/110, sol 22/22, luna 19/22 | FORENSICS s1-2, LUNA_EPISODES | Naming is 25% of a planted slot; mechanism + edit + calibration carry 75%. Open-menu arm makes naming need hypotheses. |
| Method is a fixed recipe (CF pairs, rank g x delta, fill 20) | VALIDATION, AUDIT M3 | Saturating/gated modules weaken what the sample gradient says about cue latents: top-\|g\| fails (0% in part 2), the plain v2 recipe fails on topic kept, but a topic-filtered v2 recipe still succeeds on 17-75% of OR/AND slots given names depending on kappa and unit sharpness (17% / 33% at kappa 24 with sharp units; WARN G0.2b); no raw logits on own text (D9). The Stage-2 hint arm is the decisive test. |
| Most small-model failures = under-filled latent budget (luna median 4/20; hint lifts 10/35 -> 23/35) | SMALLMODEL_FAILURES, LOG 2026-10-04 | Scoring stated in full in TASK.md; edit removal is continuous with a per-latent price, so neither under-fill nor pad is a hidden cliff. |
| Bar placement decides outcomes (tau = 0.80 x oracle; 3 of 7 Opus misses are grid noise) | FORENSICS s4a | No bar: continuous scores; latent-reading units make removal ~0.95 achievable (explore_v3 part 2), so headroom is large; grid 1024 texts/slot. |
| Null calibration is the one judgment-discriminating part (null false claims opus 0/26, sol 0/13, luna 5/13, haiku 13/13) | LUNA_EPISODES | Explicit p_cue with a Brier term; sub-bar modules and distractors on every slot so "any effect" is not "planted". |
| 96-token truncation leaks end-of-text cues (AUC 0.97-0.98) | FORENSICS s4b | Max rendered length is 128 tokens (measured, 3000 renders incl. all-on); cap 160; tools reject longer texts instead of truncating. |
| Keep-filtered pool (recipe ceiling 36% kept vs 7% dropped) | FORENSICS s4c | Pools drawn unfiltered; validity only on head behaviour on the grid. |
| Answer files reachable; wall clock 6366 s > 3600; ended_without_submit graded as submitted; /tmp audit invalids | FORENSICS s4d, LUNA_EPISODES, SMALLMODEL | Harness fixes (section 14, owned by harness branches); Stage 2 does not start until they are in. |

## 2. Decisions (each with rationale; R = rejected)

**D1. Module reads SAE latents directly; topic read-out reads the residual.** Unit c: `u_c = sigmoid(kappa * z_c)`,
`z_c = (sum_{j in G_c} beta_j f_j - center_c) / scale_c` (f = mean-pooled latents; center_c = beta . mu[G_c], the
unit's pre-activation when its group is mean-ablated, so a mean-ablated unit sits at exactly u = 0.5 for any kappa;
scale_c = half the gap between the cue-present and cue-absent class means on the dev bank, so classes sit near +-1). WHY: explore_v3 part 1 built units on the residual (dual
direction of a latent group); they spill over onto thousands of dense pooled latents and even the true group removed
only 0.25-0.62 of the cue dependence, which recreates v2's removal ceiling and bar placement. With latent-read units
the true groups remove ~0.95 (part 2). Framing for TASK.md: "a classifier built on the SAE code" (SAE-feature probes
are a real practice). The topic part stays on the residual so ablations have realistic collateral on the topic.
REJECTED: training MLP heads on confounded data (cannot guarantee or verify OR/AND structure in a 10 h iteration);
residual-direction units (ceiling, above).

**D2. Mechanisms: single (control), or, and; null.** Planted mix: or 40%, and 40%, single 20%; slot null with
p = 0.3; 3 or 4 slots per episode (null count not inferable, D7). OR: `1-(1-u_A)(1-u_B)`; AND: `u_A u_B`.
Saturation (kappa) is the dial that makes the sample gradient uninformative; kappa is frozen in Stage 0 on DEV from
{4, 6, 8, 12, 16, 24} (planner starting point after section 17: selectivity-ranked units at kappa 16-24). Part 2: at kappa 4 the v2 recipe's latents still remove 0.89-0.95 of the cue
(it fails only on topic kept) and AND cells leak (max(e10, e01)/e11 median 0.21, q90 0.53); at kappa 8-16 the v2
recipe's removal drops (0.71-0.77) but a topic-filtered variant still succeeds on 14-75% of OR/AND slots, and
single-latent ablation tests stop working (a unit saturates in its own latents), so the reference must test latent
GROUPS. The AND leak comes from z spread, not kappa (fix by sharper units). Choose the smallest kappa that passes the
GATED Gate-0 items and minimises the WARN rates.

**D3. Which latents a unit reads.** `G_c` = m = 6 latents drawn at random from the P = 40 latents (pool size P must
exceed the per-call ablation cap K_abl; Stage 0 may set P = 60) with the largest
deconfounded contrast for feature c (least squares of pooled latents on all features + topic dummies over the dev
bank), beta proportional to that contrast. WHY: with G = top-6 contrast, "free name + top-20 contrast latents" is a
complete recipe (part 2, kappa 4: 90% OR / 87.5% AND, n = 40); with random-of-top-40 it mostly fails (kappa 4: 5% OR,
17.5% AND, n = 40; kappa 6: 8% / 4%, n = 24) while the true group union still removes ~0.95. Later runs (section 17): ranking
the pool by selectivity (contrast / pooled sd, "sel40") makes units clean (AND leak 0.37 -> 0.10, oracle 100%, top-20
contrast 0-10%) but strengthens the topic-filtered v2 recipe; planner starting point for Stage 0 = sel40 + the
largest kappa that keeps the reference passing. "Ranks 1-60 with <= 1 of the top 10" did not help. REJECTED:
semantically chosen sub-groups (no time to curate; v4).
Units of one slot must not share latents (G_A and G_B disjoint, and disjoint from distractor latents).

**D4. Reliance scale, disclosed.** Planted: ||e|| / E_T in [0.45, 0.9] on the hidden grid, where e is the 3-vector of
2x2 cell effects (A only, B only, both, vs neither; single: the A effect) averaged over the two topics and E_T the
topic effect. Null: the sub-bar module has ||e|| / E_T in [0.08, 0.2]. Every slot (planted or null) also gets 1-2
linear distractors on other features (reliance 0.05-0.2 each, read from 3-6 of their contrast latents). TASK.md
states: "a probe that relies on a cue depends on it at least 0.45x as strongly as on the topic; everything else it
reacts to is at most 0.2x". C (module scale) is solved per slot so the grid reliance hits the drawn target.
WHY: v2 D1 showed undisclosed gaps cause false claims; the 0.2-0.45 gap makes calibration a measurement skill.

**D5. Feature bank (both arms).** The 16 v2-rendered features: signoff, greeting, british_spelling, question_ending,
exclamation, hashtag, bullet_list, negative_sentiment, postscript, weekday, parenthetical, quote, she_pronoun,
past_tense, lowercase, digits (emoticon/ampersand stay off). Every feature is an independent fair coin in grids, the
dev bank and the dashboard corpus. Fixed-menu arm lists all 16 (memorising v2's 8 does not help). Open-menu arm:
planted components only from the 13 with exact detectors (all but british_spelling, past_tense,
negative_sentiment, whose word-list detectors are fooled by agent prose); those three still vary and can be
distractors. signoff and greeting get structural detectors (section 10).

**D6. Text renderer.** v2 `render` with per-TEXT random cue strings (not per probe) and MAX_TOKENS = 160 with no
truncation (measured max 128). Tool texts over 160 tokens are rejected with an error (never truncated), so position
cannot leak. Sample texts: 128 per probe; the label agrees with the topic for 1-eps (eps = 0.05) and with the true
components AND 3 random other features (each perfectly confounded, like v2's 5 co-occurring candidates), so
co-occurrence never names the cue and never reveals the mechanism. For OR, label-1 sample texts carry A or B or both
in equal thirds; for AND both; single: A.

**D7. Hidden grid.** Per slot 1024 texts: topic balanced, all 16 features iid fair coins (so the 2x2 cells of ANY
claimed pair can be estimated: ~128 texts per (topic, X, Y) cell). Target bootstrap SE of removal <= 0.015 (builder
measures; enlarge to 1536 if not met). Stored as pooled latents (CSR) + base topic logit; never reachable by tools.

**D8. Ablation = mean ablation**, value mu_j = mean pooled latent over the slot's "training distribution" (its
unconfounded dev texts), the same mu the tools use. Consequence (stated as a fact of the setting, not a hint):
mean-ablating one group of an OR/AND module leaves that unit at u = 0.5 (D1 centering), so single-group removal
is ~0.5 (or) / ~0.3 (and), measured in part 2 (median 0.48-0.52 / 0.30-0.35 at kappa 4). Without that centering the
mean-ablated unit drifts to ~0 at large kappa and single-group ablation then solves AND (kappa 24 run: 33%), which
would break G0.3. Predicting this is part of the mechanism score.

**D9. Tools (open encoding only; v2 T2 sample-only tier CUT for time).** task_info, sample_texts, latent_means,
latent_tokens, latent_examples (dashboards over a reference corpus, all features independent, all 12 topics),
probe_scores(probe, ids) = raw logits on the probe's own 128 sample texts (free; confounded, like v2),
probe_gradient(probe, texts|ids, ablate=[]) = mean d logit / d f_j at those texts after the optional ablation,
probe_effect(probe, texts|ids, ablate) = per-text logit CHANGE caused by mean-ablating `ablate` (1 to K_abl latents,
K_abl = 30 provisional; keep K_abl below the unit-group pool size, see G0.10):
the intervention tool. **There is no raw logit on agent-written text.**
Because the agent cannot measure the topic effect itself without raw logits, public.json and TASK.md give each
probe's topic effect E_T (mean logit difference between its two topics on unconfounded dev-bank texts); every
scale in the claim (reliance, cells) is in units of it. This scalar says nothing about the cue.
WHY effect-only: with raw logits on own text, the behavioural half of the claim (cue names, combination, 2x2 cells,
null calls) is black-box measurable by group testing over a 16-item menu; the planner's estimate is that a
black-box script would then reach S* ~0.6 without touching a latent, so interpretation would not be the
bottleneck. With effect-only, every behavioural measurement on unconfounded text goes through a latent set the
agent has found, and the K_abl-latent ablation cap blocks "ablate everything = raw logit". The gradient on own text is
allowed: saturation attenuates it for cue latents (part 2: top-|g| never succeeds) while the linear distractors and
the topic stay large, which is the "attribution misleads" trap the brief wants (not a guarantee: z spread leaves
some gradient on cue latents; see section 17).
Caps (provisional, frozen in Stage 0): tool_calls 200, forward 4000 (latent_means, latent_tokens, probe_gradient
texts), probe_effect 1200 texts, wall_clock_s 3600 (enforced by the harness fix), call_timeout_s 180.
WHY a cap, not a score price: a score price couples reward to budget habits, which is the v2 luna failure mode. The
cap is set so the label-free v3 reference needs <= 50% of it (Stage 0/1 measure).

**D10. Scoring = continuous weighted sub-scores (section 8), fully stated in TASK.md.** No pass bar decides reward.
Padding costs lambda per edit latent (provisional 0.01, calibrated in Stage 0) and per group latent in group quality.
REJECTED: collateral penalty on unrelated held-out text (topic-kept factor already measures collateral; more grid
cost); marginal-contribution (leave-one-out) pruning (harder to state, non-monotone, gameable by near-duplicates);
product reward over slots (creates the all-or-nothing cliff the brief wants gone).

**D11. Two arms share heads, grids, tools and scoring; only identification input differs** (section 10).

**D12. Pools drawn unfiltered.** Slot validity checks only head behaviour on its grid (reliance in range, mechanism
shape test G0.6, oracle union removal >= 0.9 with kept >= 0.9); a failing slot is redrawn with a new seed BEFORE any
solver runs; no solver-based keep filter ever.

**D13. SAE error-term stretch: REJECTED for v3** (it re-introduces removal ceilings that confound the continuous edit
score, and costs build time; v4 option). **T2 sample-only tier: CUT.** **More than two components: CUT.**

### Dan's idea list: verdicts

| Idea | Verdict | Why |
|---|---|---|
| Nonlinear heads (OR/AND) after L12, CPU-graded | ADOPTED, changed (D1) | Latent-read units, not residual units: residual units cap removal at 0.25-0.62 (part 1). |
| Grade mechanism by prediction | ADOPTED, changed | Predictions are of the 2x2 cell function and of the removal of the agent's OWN sets on the hidden grid. Grader-chosen sets were rejected: the agent cannot know what they are, so predicting them is either impossible or reduces to the cell function. |
| Separate sub-scores, continuous combination | ADOPTED | Section 8. |
| Make padding cost something | ADOPTED: priced latents (edit and groups) + prediction of own-set removal | Collateral penalty and leave-one-out REJECTED (D10). |
| Remove/loosen the fixed menu | ADOPTED as the open-menu arm; fixed-menu kept as the main arm with all 16 features listed | Comparability to v2 and a clean ablation of what the menu buys. "Require a cue detector (code)" REJECTED: the grader would have to execute agent code (security, time); text pairs graded by our exact detectors are checkable. |
| SAE-error-term stretch | REJECTED (D13) | |
| Intervention tool on own text, priced | ADOPTED (probe_effect, capped) | Nonlinear heads cannot be evaluated from latents without knowing the head; mechanism testing needs unconfounded own texts (the luna confound lesson). |
| Fix-first list (wall clock, runner/grader, truncation, isolation, unfiltered, /tmp) | ADOPTED | Truncation + unfiltered in the task (D6, D12); the rest in harness branches (section 14). |
| Open-menu arm sketch (description + ~10 pairs, exact detectors, none valid, full bank, nulls redefined, fuzzy cues handled) | ADOPTED with changes | 6-12 pairs per component; planted only from 13 exact-detector features; nulls = no feature above the disclosed 0.45 scale (D4) since v3 has no unlisted decoys; the description is recorded, not graded. |

## 3. Head construction (generator, builder A)

1. **Dev bank** (shared, one GPU pass): ~12k unconfounded texts (all 16 features coins, 12 topics uniform),
   pooled residual (fp16) + pooled latents (CSR float16 values, uint16 indices). Used for: deconfounded contrasts C[feature, latent];
   topic read-outs; unit center/scale; mu; the dashboard corpus is a separate 2k-text draw.
2. **Topic read-out** per slot: logistic regression (C = 0.05, standardised, folded back, as v2 `train_probe`) on dev
   texts of the slot's two topics. `gT = W_dec @ w_T`.
3. **Units**: D1/D3. kappa frozen in Stage 0. **Module** per D2, C solved for the drawn reliance (bisection on the
   grid). **Distractors** per D4.
4. **Mechanism shape check (G0.6)** on the grid (cells in E_T units): or: e11 <= 1.3 max(e10, e01) and
   min(e10, e01) >= 0.6 e11; and: max(e10, e01) <= 0.25 e11; single: |e01| <= 0.15 e10. Failing slots redrawn.
5. **Oracle check**: ablating the union of true groups removes >= 0.9 with kept >= 0.9 (else redraw).
6. Store per slot (section 14 layout). Per-instance latent permutation (public id -> true id) as v2.

## 4. Generator changes vs v2

No decoys, no keep filter, no per-family tau, no label-noise dial; per-text cue strings; MAX_TOKENS 160 with an
assertion that no generated text exceeds it; factorial grid; unit/module construction; one shared dev bank (texts of
the bank never appear in any instance's grid, sample or corpus: disjoint seeds, asserted by hash). Slot/sample reuse
across instances forbidden (v2 C2 rule kept).

## 5. Claim schema (submission.json)

```json
{"probes": [{
  "probe": 0,
  "verdict": "cue",                      // "cue" | "none"
  "p_cue": 0.9,                          // P(this probe relies on a cue at >= 0.45x topic); required
  "combination": "or",                   // "single" | "or" | "and"; null when verdict = none
  "components": [                        // 1 (single) or 2 (or/and); [] when none
    {"cue": "hashtag",                   // FIXED arm: a menu name
     "description": "...", "pairs": [["text without", "text with"], ...],   // OPEN arm: 6-12 pairs, each text <= 160 tokens
     "latents": [101, 7]},               // the latents that carry THIS component into the probe (<= 12)
    {"cue": "postscript", "latents": [5, 900]}],
  "edit": [101, 7, 5, 900],              // <= 20 latents whose mean-ablation removes the reliance
  "predictions": {
    "cells": {"10": 0.6, "01": 0.6, "11": 0.65},   // probe-logit shift vs neither present, in units of the topic effect; single: only "10"
    "removed_edit": 0.9,                 // fraction of the cue dependence the edit removes (1 = all, < 0 = overshoot)
    "removed_by_component": [0.5, 0.5]}  // same, ablating each component's latents alone
}]}
```
Validation (validate_submission, works without load()): types, ids in [0, 16384), lengths, verdict/combination/
component-count consistency, fixed arm names in menu, open arm 6-12 pairs with texts <= 1000 chars, numbers finite.
Missing optional predictions score 0 on that item; missing p_cue = 0.5.

## 6. (reserved)

## 7. TASK.md principles

State fully: the probe family (topic read-out + a module over SAE latents of up to two text features, combined
single / either / both), mean ablation and where mu comes from, the reliance scale (D4) and the null definition,
every sub-score formula, weight, tolerance and lambda (section 8), caps, and that latents beyond what the edit needs
cost lambda each while unused budget costs nothing. Name no method, no recipe, no sample-gradient warning, no
cue-to-slot hints, no task codename. Fixed arm lists the 16 features with one-line descriptions (v2 wording fixes:
"has a hashtag line near the end", "the last body sentence is a question"). Open arm: "the probe was meant to read
only the topic; find what else it reads". Both: the explicit "none" answer.

## 8. Scoring (grader builder B; all numbers provisional until Stage 0 freezes lambda)

Per slot, S in [0, 1]. Grid quantities: E_T (topic effect), true cell vector e* (3 cells; single: 1), removal
r(L) = 1 - ||e*(abl L)|| / ||e*||, kept(L) = E_T(abl L) / E_T.
- **CAL** = 1 - (p_cue - truth)^2, truth = 1 planted / 0 null.
- **ID** (planted): (matched true components - 0.5 x wrong components) / n_true, clipped to [0, 1]; 0 if verdict none.
  Fixed arm: match by name. Open arm: section 10 pair grader.
- **MECH** (planted, over matched components): 0.3 x [combination correct, only if all components matched]
  + 0.3 x GROUP + 0.4 x PRED.
  GROUP = mean over matched components c of max(0, q_c - lambda_g |L_c|), q_c = 1 - (contrast of true unit c between
  c-present and c-absent grid texts after ablating L_c) / (same before); lambda_g = lambda.
  PRED = mean over items of max(0, 1 - |pred - true| / tol): cells (true from the grid for the TRUE components in the
  claim's order; tol 0.3 in E_T units), removed_edit = r(edit) (tol 0.25), removed_by_component = r(L_c) (tol 0.25).
- **EDIT** (planted) = ID x max(0, clip(r(edit), 0, 1) x keep_f - lambda x |edit|), keep_f = clip((kept - 0.7) / 0.2, 0, 1).
- **Planted S** = 0.15 CAL + 0.25 ID + 0.25 MECH + 0.35 EDIT.  **Null S** = 0.15 CAL + 0.85 x [verdict none].
- **Episode score** = mean slot S (the RL reward). **Main reported score S*** = (score - n_null/n) / (1 - n_null/n)
  (excess over the constant "none, p = 0" answer; all-null episodes report score). Sub-score means reported always.
- Oracle reference: the union of true groups with true cells and true removals as predictions scores ~0.95-1.
- **lambda calibration (Stage 0, DEV only):** pick lambda in {0.005, 0.0075, 0.01, 0.015} as the LARGEST value with
  (a) oracle-union EDIT >= 0.8 on >= 90% of planted slots and (b) padding the oracle union to 20 with top-|g| latents
  lowers EDIT on >= 80% of planted slots. Minimal true sets are ~6 (single) / ~12 (or, and) latents.

## 9. (reserved)

## 10. Arms and nulls

**Fixed menu (FM, main arm).** Components named from the 16-feature menu. Null = verdict none.
**Open menu (OM).** No list; the agent gets the probe's intended task (topic pair) only. Per component: a short
description (recorded, not graded) + 6-12 pairs (text without, text with). Pair grader (`pair_grader.py`): run all
16 detectors on both texts; a pair is CLEAN for feature X iff X goes 0 -> 1, every other detector is unchanged, and
the topic-keyword set is identical. A component MATCHES true component X iff >= 75% of its pairs are clean for X and
it has >= 6 pairs; a component whose pairs are clean (>= 75%) for a non-true feature is WRONG; anything else is WRONG.
Planted components only from the 13 exact-detector features (D5). Generalised detectors for open text: signoff =
last non-empty line is 1-3 capitalised words and the line before ends with "," and has <= 4 words; greeting = first
line has <= 5 words and ends with ","; the rest as v2 (`?`, `!`, `(`, `"`, `#tag` line, `- ` bullets, `p.s.`
line, weekday names, `\bshe\b`, all-lowercase, digits). Builder B unit-tests detectors on 3000 generated texts
(must equal the generator's labels 100%) and on adversarial strings (section 12).
Nulls in both arms: every feature's reliance <= 0.2 (D4); a null slot is not "a probe that leans on a decoy".

## 11. Gates and kill criteria

**Protocol for every gate.** Write PREDICTIONS (numbers, before running) and commit. Tune only on DEV slots
(seed range A). Freeze all parameters in `config_v3.json` (commit hash recorded). Build HELD-OUT slots/instances
(seed range B, generated after the freeze) and run the gate ONCE on them. A second held-out run after any change
needs a new seed range and a NOTES entry saying why.

**Gate 0 (feasibility, CPU after one GPU encode; held-out >= 60 planted slots, >= 20 null).**
Free names (true components given) for every baseline. "success" = r >= 0.8 and kept >= 0.8 on the grid.
GATED (these are the brief's Stage-0 criteria; a failure after one DEV tuning round kills):
- G0.1 top-|g| fill-20 (sample gradient): success on < 15% of OR+AND slots.
- G0.3 single-group ablation (the first named component's top-20 contrast latents): < 15% of OR+AND slots.
- G0.5 label-using reference (knows component labels; candidates = top-40 contrast per component; finds the read
  latents with GROUP ablation tests and/or the gradient at a partially ablated point, then prunes; measures on its own
  labelled dev-bank texts of the slot's topics, never the grid; counts its head evaluations against the probe_effect
  cap): >= 60% of planted slots succeed and median r >= 0.85, within the cap.
  (Its mean planted S >= 0.75 is checked in Gate 1, once the grader exists, so Gate 0 does not wait for builder B.)
- G0.6 mechanism shape (section 3.4): redraw rate <= 50% for every mechanism (so the kept slots are not a strange
  subset), and 100% of kept slots satisfy it.
- G0.7 grid SE of removal median <= 0.015.  G0.8 no text over 160 tokens anywhere.  G0.9 null slots: every
  feature's grid reliance <= 0.2.
WARN (planner's stricter recipe checks; < 15% is the target; failing does NOT kill but must be reported, carried
into Gate 1 as attacks, and makes the Stage-2 hint arm decisive):
- G0.2 v2 recipe (g x delta of the named components, top 20).
- G0.2b v2 recipe + topic filter (drop the 1% most topic-contrasted latents, measured on unconfounded own texts,
  then top 20 by |g x delta|). Planner data: 29% / 43% (OR / AND) at kappa 8, 14% / 29% at kappa 16 (n = 14 per
  mech; section 17).
- G0.4 top-20 contrast of the named components.
- G0.10 contrast-ablation NAMING accuracy without names: for each menu feature, ablate its top-K_abl contrast
  latents (K_abl = the probe_effect per-call cap) on cue-on/cue-off own texts; call the feature with the largest
  effect if >= 0.45 x topic, else none. Report accuracy on planted and null slots. Note the interplay: if the unit
  pool P <= K_abl, this names everything; Stage 0 keeps P > K_abl (default P 40, K_abl 30; or P 60).
Reported (not gated): the same baselines on single slots (control), mean r per baseline, the gradient at the
ablated point and an adaptive group-ablation search (the v3 recipe ceiling), head-evaluation counts.
**Kill at Gate 0:** any GATED item fails after one DEV tuning round.

**Gate 1 (grader validation, CPU, held-out instances, >= 20 episodes).** Mean episode score and S*:
- true claim (oracle) >= 0.9; label-free v3 reference >= 0.65 S*;
- wrong combination (true otherwise) loses >= 0.1 per planted slot; padded to 20 with top-|g| loses >= 0.03 EDIT;
- random latents, always-max-k (20 top-|g|), always-none, always-claim (co-occurrence naming from the sample +
  top-|g| latents + pool-mean predictions), gradient-naming (name the feature whose contrast latents carry the most
  |g|; this lands on distractors), v2 recipe (hint-arm script), v2 attack suite: every one S* <= 0.20 and at least
  0.35 below the label-free reference;
- free-name UPPER BOUNDS (diagnostic, not gated on S*: names, combination and p_cue given for free, then each
  cheap latent rule + pool-mean predictions): report MECH and EDIT sub-scores; EDIT of each <= 0.5 x the reference's;
- planted/null fingerprint from permutation-invariant gradient/eval statistics: AUC <= 0.65;
- pair grader: 0 accepted adversarial pair sets (section 12);
- regrade determinism and schema edge cases: 100%.
**Kill at Gate 1:** a cheap attack reaches S* >= 0.35, or the oracle claim < 0.85.

**Gate 2 (model smoke).** Opus S* (pass@1 mean over episodes) in [0.30, 0.70]; report pass@3 (mean over instances of
the best attempt). Floors: haiku < Opus - 0.3; luna below sol. Hint arm (v2 recipe given) not above Opus without
hint + 0.05 (the recipe must not suffice).
**Kill at Gate 2:** Opus outside the band after ONE tuning round (knobs: kappa, reliance range, probe_effect cap,
lambda, prediction tolerances), or a cheap scripted solver (G1 list) within 10 points of Opus S*, or the hint arm
>= Opus. NOTE: the label-free v3 reference is itself a scripted solver that embodies the intended insight (as v2's T1
reference did); it is reported next to Opus but is not a kill trigger. If killed, write up why (section 15).

## 12. Stage 1 attack list

v2 suite, re-pointed at v3: nothing / claim_all / prior / recipe (recipe_baseline variants), gradfp_prior,
gradfp_behav, attrfp (fingerprints), crosspool (cross-probe sample contrasts), memo (slot/sample reuse), truncation
contrast (must be inapplicable: no cut text), name_on_topic_scale (disclosed-scale thresholding), Neuronpedia-id
lookup (permutation), answer reachability (from the sandbox user: runs/.episodes, instance dirs, task tree: must be
unreadable once isolation lands; until then recorded as an open blocker), duplicate ids, > k ids, "none" with latents.
New: wrong-combination; pad-to-20; always-max-k; random-k; constant predictions (pool means); cells from the sample
(confounded) instead of own texts; "single-group as union" (submit only the first component); black-box-only solver
(sample raw logits only; must score like always-none); SHIFT-style naming by group ablation (for each menu feature,
ablate the top-K_abl contrast latents of its own matched pairs, call the feature whose ablation moves the probe most;
the cheapest naming route under D9, bounded by the K_abl cap and the probe_effect budget); v2 recipe with gradient
at sample texts; v2 recipe + topic filter; gradient at the ablated point (reported, the intended insight); adaptive ablation search over top contrast latents (reported);
p_cue extremes (always 0.5; always 1 with verdict none); OM pair attacks: pairs toggling 2+ features, pairs that
change topic words, "?" inside a quote / "!" inside a hashtag, a P.S. line without "p.s.", a sign-off with an
unlisted closing word, 16 components submitted, duplicated pairs, pairs longer than 160 tokens.

## 13. Stage 2 design (sized to ~4 h of runs and $15 OpenAI)

Instances: 8 FM + 6 OM, held-out, unfiltered, never exposed before. Caps as frozen. Wall clock 3600 s enforced.
Efforts: Opus 5.5 high via the isolated headless launcher; sol (gpt-6.1-sol) medium; luna medium; Haiku 4.5 default.
OpenAI cost prior: sol $0.18-0.32/ep in v2 (SCALE1_OPENAI) -> assume $0.35 for v3; luna $0.03.
| Tier | Arm | Runs | OpenAI $ |
|---|---|---|---|
| P1 | FM: Opus x1, sol x1, luna x1, Haiku x1 on 8 | 32 | ~3.1 |
| P2 | FM hint arm (v2 recipe paragraph): Opus x1 on 8 | 8 | 0 |
| P3 | OM: Opus x1, sol x1 on 6 | 12 | ~2.1 |
| P4 | FM pass@3: Opus attempts 2-3, sol attempts 2-3 on 8 | 32 | ~5.6 |
| total | | 84 | ~10.8 (stop at $14) |
Run in tier order; Gate-2 band check after P1's Opus runs (kill/tune decision there). Per-episode max-usd 0.8.
Concurrency ~5 tool servers (gpuq light, ~3.5 GB each); expected 20-25 min/episode -> ~4 h for 84; cut P4 first
(keep >= pass@2) if behind schedule. Report per arm: S*, sub-score means, null false claims, edit sizes, probe_effect
use, cost; forensics report in the LUNA_EPISODES format.

## 14. Interfaces and file ownership (both builders work in `tasks/shifthunt3/` in parallel)

Binding: `INTERFACES.md` (planner-written; amend only by dated AMENDMENT lines; the 07:15Z amendment adds a
builder C for the agent-facing layer: tools.py, run_agent.py, agent prompts, prompt_render.py). Harness fixes:
`~/wt/v3harness` (wall clock enforcement, ended_without_submit => submission None, /tmp audit fix) and
`~/wt/v3iso` (OS isolation); never in the task worktree.

## 15. Time plan (UTC)

06:30-07:45 plan (done). 07:45-10:00 Stage 0 builder A || grader builder B. 10:00-10:30 Gate 0 on held-out +
auditor. 10:30-11:45 Stage 1: tools.py, TASK/agent_prompt, label-free reference, attacks, Gate 1, Stage 2 pools.
11:45-15:45 Stage 2 by tier. 15:45-16:30 forensics, recommendation (continue / change direction / stop).
If killed at any gate: write `KILLED.md` (what failed, numbers, what would be needed) and stop.

## 16. Thin evidence and risks

- All planner experiments use the SAE reconstruction of v2 sample texts (no error term), 96-token-truncated texts
  dropped, ~17k texts, 40 feature pairs, ONE topic pair, 2x2 cells estimated from unbalanced data (min 4 per cell).
  Directionally solid, numerically rough; Stage 0 must redo everything on real grids.
- The v3 recipe exists: "ablate top contrast, take the gradient there" finds the read latents (part 2, free
  names: 67-93% of OR/AND slots at kappa 4-6, 43-79% at kappa 8-16). An RL policy will learn it; v3 tests discovering it plus mechanism and calibration. If
  Opus finds it every time, FM saturates on EDIT and the band must come from MECH/PRED and OM.
- Naming without raw logits (D9) still has a semi-recipe: contrast-latent ablation per menu feature (16 features x
  pairs + effects). The K_abl cap (< pool P), random-of-top-P groups and the probe_effect budget bound it; Gate 1 measures
  it. If it is cheap, FM naming is a SHIFT recipe (as in v2) and OM is where naming needs hypotheses.
- D9 (no raw logits on own text) is a planner decision not yet tested with agents: it may make null calls too hard
  (absence of evidence) and push S* below the band; the fallback is a small raw-logit budget (e.g. 8 texts/episode).
- Constructed heads are less natural than trained probes; the realism claim is "auditing an SAE-feature classifier".
- Mean-ablation residue (single-group ~0.5 / ~0.3) may read as a trap; it is stated as the ablation definition, and
  predicting it is graded, but agents may find it unfair. Watch transcripts.
- Detector robustness in OM on agent prose is untested until Stage 1.
- Stage 2 n is small (8 FM, 6 OM); band decisions at n = 8 have SE ~0.08 on S*.
- Depends on harness work in other branches (isolation, wall clock, launcher); Stage 2 must not start without them.
- OpenAI cost prior is from v2; v3 episodes may be longer.

## 17. Planner experiments (explore_v3/, CPU, v2 caches read-only)

Data: per-token SAE activations of v2 T2new sample texts (`cache/t2sel`, 50 instances), pooled per text; texts at
the 96-token cap dropped -> 17,083 texts, 2,550 non-zero pooled latents per text, feature rates 0.40-0.51.
R_hat = F W_dec + b_dec (no SAE error term). One topic pair (the two most frequent topics) for the topic read-out;
eval half ~1,400 texts; 2x2 cells x topic estimated by cell means (min 4 per cell). Features' deconfounded contrasts
from least squares on all features + topic dummies over the dev half. 40 random feature pairs, each built as
single / or / and. Simulated sample = label-1 texts carry the cue (mechanism-true), label-0 do not. Peak RSS 1.9 GB.
"ok" = removal >= 0.8 and topic kept >= 0.8 (part 1 used removal >= 0.6). Free component names for every recipe.

**Part 1 (`derisk_heads.py`): units on the residual (dual direction of 6 latents), kappa 3, rho 0.6.**
OR saturation is real (cells e.g. 7.39 / 7.34 / 8.21: both ~ either). But the TRUE group union removes only
0.62 (top-6 groups) / 0.33 (random-of-top-40): the dual direction reads thousands of dense pooled latents. Label
reference (20 latents) 0.61-0.75. => rejected (D1): it recreates v2's ceiling.

**Part 2 (`derisk_latunits.py`): units read pooled latents directly; n = 40 pairs per cell at kappa 4, 20 at kappa 8.**
Fraction "ok" (median removal), G = random 6 of top-40 contrast:

| recipe (free names) | OR k4 | AND k4 | single k4 | OR k8 | AND k8 |
|---|---|---|---|---|---|
| top-\|g\| fill 20 (sample gradient) | 0% (0.80) | 0% (0.70) | 0% (0.92) | 0% (0.73) | 0% (0.38) |
| v2 recipe g x delta, 20 | 7.5% (0.94) | 2.5% (0.90) | 2.5% (0.97) | 10% (0.92) | 0% (0.80) |
| top-20 contrast of named features | 5% (0.59) | 17.5% (0.60) | 40% (0.70) | 5% (0.42) | 25% (0.61) |
| one true group only | 0% (0.48) | 2.5% (0.30) | = union | 0% (0.44) | 10% (0.40) |
| union of true groups (oracle) | 95% (0.95) | 85% (0.94) | 92.5% (0.95) | 90% (0.96) | 85% (0.94) |
| gradient at the ablated point | 80% (0.91) | 85% (0.91) | 92.5% (0.94) | 55% (0.83) | 80% (0.90) |
| greedy single-latent ablation search | 42.5% (0.79) | 52.5% (0.82) | 85% (0.91) | 30% (0.71) | 20% (0.71) |

With G = top-6 contrast, "top-20 contrast" passes 90% (OR) / 87.5% (AND) at kappa 4 => D3 (random-of-top-40).
Mechanism shape (kappa 4, n = 40): OR e11 / max(e10, e01) median 1.05, min(e10, e01) / e11 0.76; AND
max(e10, e01) / e11 median 0.21, q90 0.53 (kappa 8: same 0.20 / 0.53) => the AND leak comes from the spread of z
inside each class (a 6-latent unit is a noisy detector), not from kappa; G0.6 will redraw ~40% of AND slots unless
Stage 0 sharpens units (choose G among latents with high single-latent AUC for the feature, or m = 8-10).

**Part 2 addendum (G = random 6 of top-40 unless noted; "ok" fraction, OR / AND).**

| recipe (free names) | k6 n=24 | k6 r60 n=24 | k8 n=14 | k16 n=14 |
|---|---|---|---|---|
| top-\|g\| fill 20 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| v2 recipe g x delta | 0 / 0 | 0 / 4% | 7% / 0 | 7% / 0 |
| v2 recipe + topic filter (top 1% topic-contrast latents dropped) | n/a | n/a | **29% / 43%** | **14% / 29%** |
| top-20 contrast of named features | 8% / 4% | 12.5% / 8% | 0 / 14% | 0 / 14% |
| one true group | 4% / 0 | 8% / 12.5% | 7% / 7% | 7% / 29% |
| union of true groups (oracle) | 96% / 96% | 83% / 88% | 86% / 86% | 86% / 86% |
| gradient at the ablated point | 67% / 67% | 75% / 67% | 79% / 43% | 57% / 43% |
| greedy single-latent search | 33% / 38% | 8% / 21% | 14% / 29% | 0 / 21% |
AND leak (median max(e10, e01)/e11): k6 0.23 (r60 0.32); k8/k16 on their 14 pairs 0.37.
First prior was kappa 6 + random-of-top-40; superseded by the sharper-unit and kappa-24 runs below. "r60" does not help.

**Mechanism-shape pass rate (planner's G0.6 test on the planner cells; OR / AND / single).** kappa 4 random-of-top-40:
0.40 / 0.42 / 0.70 (n = 40); top-6: 0.88 / 0.65 / 0.82; kappa 6 random: 0.33 / 0.54 / 0.62 (n = 24); kappa 8 random:
0.35 / 0.40 / 0.65 (n = 20). Part of this is estimation noise: single slots should have e01 = 0 exactly, yet
|e01| / e10 has q75 0.13 and q90 0.39, because v2 sample features were correlated (7 confounded per slot). The OR
failures are mostly min(e10, e01) < 0.6 e11 (median 0.52): one random-of-top-40 unit is often a weak detector that
does not saturate. => sharpen units (selectivity-ranked pool, larger m) before Stage 0's first DEV run; a
sharper-unit runs follow.

**Sharper units (kappa 6; "sel40" = G random 6 of the top-40 by contrast / pooled sd; "m10" = 10 latents per unit).**
OR / AND / single, fraction ok (n = 20 per cell; sel40 and its rand40 control come from the same run):

| recipe (free names) | sel40 | rand40 (same run) | m10 (rand40, m = 10) |
|---|---|---|---|
| top-\|g\| fill 20 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| v2 recipe (no filter) | 0 / 0 / 0 | 10% / 0 / 5% | 5% / 5% / 15% |
| v2 recipe + topic filter | **75% / 70% / 100%** | 55% / 50% / 50% | 40% / 30% / 60% |
| top-20 contrast | 10% / 0 / 25% | 5% / 25% / 35% | 15% / 10% / 20% |
| one true group | 0 / 0 / = union | 0 / 0 / = union | 5% / 5% / = union |
| union of true groups | 100% / 100% / 100% | 90% / 85% / 90% | 95% / 85% / 100% |
| gradient at the ablated point | 100% / 95% / 100% | 65% / 80% / 90% | 40% / 40% / 100% |
| greedy single-latent search | 10% / 0 / 30% | 35% / 25% / 65% | 5% / 10% / 50% |
| shape check pass | 12/20 / 12/20 / 15/20 | 7/20 / 8/20 / 13/20 | 10/20 / 9/20 / 12/20 |
AND leak median: sel40 0.10 (vs 0.37 rand40); OR min(e10, e01)/e11: sel40 0.77 (vs 0.43).

**kappa 24 + sel40 (n = 12 per mech; centering at the class midpoint, NOT yet at beta . mu).** OR / AND / single:
top-|g| 0 / 0 / 0; v2 recipe 0 / 0 / 0; v2 recipe + topic filter **17% / 33% / 67%**; top-20 contrast 0 / 17% / 25%;
one true group 0 / **33%** / = union; union 100% / 100% / 100%; gradient at the ablated point 92% / 83% / 100%;
shape check 9/12 / 10/12 / 11/12; AND leak 0.04. A hard threshold cuts the topic-filtered recipe (75/70% -> 17/33%)
but lets one-group ablation solve AND because the mean-ablated unit falls below threshold; the D1 centering
(center = beta . mu) is the fix (untested).

**Key reading for Stage 0.** Sharper units fix the mechanism shape and the oracle, and kill "top contrast", but
make the topic-filtered v2 recipe STRONGER (70-100%; 50-55% for rand40 at the same kappa 6). Reason: with the deconfounded feature contrast delta, g x delta
is near zero for every latent that does not carry the named feature, so ANY non-zero gradient on the read latents
ranks them first; saturation only helps if the module's gradient at sample texts is effectively zero. That needs
clean units AND a hard threshold (very large kappa): kappa 24 + sel40 cuts it to 17% / 33% (OR / AND). With the D1
centering, Stage 0 should try sel40 at kappa 16-24 first. If G0.2b still fails, accept that the FM latent half is recipe-able given names (G0.2b WARN) and rely on naming
without raw logits, mechanism, predictions, calibration and the OM arm, with the Stage-2 hint arm as the test.

**Readings that change the plan.**
1. The v2 recipe fails ONLY on topic kept (median 0.69-0.72): its latents remove 0.80-0.97 of the cue even at
   kappa 8. The sample gradient is NOT made uninformative by saturation, because z is spread and many texts sit in
   the transition zone. With a simple topic filter it succeeds on 29% / 43% (OR / AND, kappa 8) and 14% / 29%
   (kappa 16) of slots given names (addendum). This is G0.2b (WARN, not a kill: the brief's Stage-0 criteria are
   top-|g| and single-group). If Stage 0 cannot push it under 15%, the latent half of FM is partly a recipe given
   names, and v3's discrimination rests on naming without raw logits (D9), mechanism, predictions, calibration and
   OM; the Stage-2 hint arm is then the decisive test.
2. Mean ablation of one group leaves ~0.5 (OR) / ~0.3 (AND) removal, as predicted (D8).
3. "Gradient at a partially ablated point" is a strong recipe once the insight is had (80-93% at kappa 4).
4. Steeper kappa breaks single-latent ablation search (units saturate in their own latents) => reference must use
   group tests.


## 18. Kill decision (2026-10-04 08:40Z, planner; full write-up in KILLED.md)

- **What failed:** the latent half (and, through latent contrasts, naming) is solved by no-name scripts that use the
  exact algebra of the head family. Audit on HELD: `graddiff` 49/51 OR+AND (96.1% [86.8, 98.9]). Grader dry run on
  DEV: S* 0.567, against the Gate-1 kill line of 0.35. The reference passed G0.5 through the same exact-difference
  property.
- **Remedies checked:**
  - (a) Whole-sample gradient + ablate: exact support on 44/44 DEV slots (C); no-name script S* 0.552 (B).
  - (b) One fixed gradient: exact block linearity test (C, DEV).
  - (c) No gradient, P 40 > K_ABL 30: the group-testing script solves 33% with names and 35% without
    (`explore_v3/ablonly_grouptest.py`, HELD). Names add nothing, and G0.5 is out of reach.
  - (d) No gradient, P <= K_ABL: 80% without names.
  - (e)-(h) Nonlinear topic read-out, nuisance units, noise and dense heads: they raise the cost but leave the signal,
    or they are outside the timebox (reasoning only).
- **Lesson for D1-D3 / D9:** pairing "planted sparse module + linear rest" with exact tools is unsound for measuring
  interpretation. Hiding the signature turns the latent half into group testing, which is search, not
  interpretation. The closed bank (D5) makes naming a statistic.
- **v4 needs:** an open cue grammar, trained heads with behavioural ground truth, edits scored against an optimizer
  ceiling, and a GATED tool-algebra red-team plus a with-names vs no-names script gap in Gate 0 (KILLED.md
  section 5).
- **Thresholds were not loosened.** No Stage-1 or Stage-2 runs; no v3 model episodes.
