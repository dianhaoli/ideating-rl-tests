# FeatureMatch diagnosis: pre-registration

Written 2026-10-02, before any experiment of this diagnosis (steps 1-5 of `PLAN.md`) has run. No style bank exists
yet, no filtered pool exists, and no agent has run on the v2 pool. **This file is never edited after results exist.**
Outcomes, and any deviation from the plan below, go in dated sections appended at the end ("Outcome YYYY-MM-DD ...",
"Deviation YYYY-MM-DD ..."). A deviation recorded before the affected results exist is allowed. A deviation recorded
after them must say so.

## 0. What this study asks, in plain terms

FeatureMatch shows an agent one SAE latent (a "feature" inside the language model gemma-2-2b) and 20 candidate
concepts. The agent must say which concept the latent responds to, or "nothing found" when none of them fits. The agent
can run text through the model and read the latent's activation. An episode has 3-5 such questions ("slots"). It passes
only if every slot is right.

So far agents have failed a lot (see `docs/forensics/FEATUREMATCH_PROBE_FORENSICS.md`). We want to know whether they
fail because **the interpretability is hard** (right reasons), or because of things unrelated to that skill: the latent
ignores the agent's writing style, the tools are confusing, the agent quits early, or there is a bug (wrong reasons).
This file fixes in advance how each failure will be labelled, which experiments run, what we expect, and what result
would mean what.

**Terms.**
- *Planted slot*: the true concept is on the menu. *Null slot*: it is not, and the right answer is "nothing found".
  The concept the latent really encodes is called the *anchor* c\*. On a planted slot the anchor is the true option.
- *AUROC* of an option: the probability that a random text of that concept makes the latent fire more than a random
  text of another concept. 0.5 is chance and 1.0 is perfect.
- *Splits A/B/C*: three disjoint sets of dataset texts per concept. A defines the answer key, B is used to choose
  latents, and C is held out (the reference solver probes with it).
- *Bank F / bank R*: two independent sets of about 20 differently styled texts per concept (casual, news, dialogue,
  listicle and so on), written by two different writer agents. Bank F is used for the style filter (step 2). Bank R is
  used by the style-robust recipe (step 3).
- *Forward budget*: the number of texts the agent may run through the model (T1/T2 1200, T3 550).
- *Generic fire rate* (density): the fraction of all split-A texts, over all 232 concepts, on which the latent fires
  (`cache/firerate_L*_A.npy`). On the current v2 pool its tertile cut points are **0.054** and **0.111**. These cut
  points are frozen here: *sparse* is < 0.054, *dense* is >= 0.111.
- *Taxonomy sibling*: two concepts whose label paths share everything except the last element. Examples: two athlete
  types, or two Slavic languages (`generate.py` `_prefix >= len(path) - 1`).
- *Distractor closeness of a slot*: the highest split-C AUROC among the slot's wrong options (`dis_C`). On v2 planted
  slots its median is 0.611 and its tertile cut points are **0.578** and **0.635** (frozen).

## 1. Failure classification (step 5)

### 1.1 Inputs the classifier reads, per failed slot

A *failed slot* is any slot with `correct == false` in `grade.json` `details.slots`, in a valid episode. The script
(`diagnosis/classify.py`, written after this file and before any agent episode is classified) reads:

1. **Grade details**: whether the slot is planted, whether the agent claimed an option, and whether it was correct.
2. **Answer key** (privileged, never shown to agents): the anchor c\*, the menu, every option's AUROC on splits A, B and
   C, `margin_C`, and the latent's fire rate on c\*'s split-C texts. It also uses *m\**, the median max activation of
   the latent over the split-C texts of c\* on which it fires (from `cache/acts_L*_C.npy`), and whether each option's
   bare name fires the latent (`cache/acts_L*_names.npy`).
3. **Tool log**: every call with its full arguments and full response. Responses must not be truncated, because the
   forensics found truncation at about 4 KB; step 4 must store them in full. From it the script derives the following:
   - *Probes on this slot*: texts sent through `latent_activations` with this slot's latent index at this slot's
     layer, plus texts sent through `top_latents` at this layer. For a `top_latents` call, the latent's activation
     counts as known (0 if it is not in the returned top k).
   - *Option attribution* of each probe text (see 1.2).
   - *n_true*: the number of distinct probe texts attributed to the true option (planted) or to the claimed option
     (null). *Activations seen*: the max activation of the slot's latent per probe text, and the peak token.
   - Counts of `top_latents`, `generate`, `vocab_projection` and `next_token_logits` calls, error responses, and
     whether this slot's latent was ever queried at a wrong layer.
4. **Budget at submit**: `f_fwd` is forward units used divided by the forward cap. Also the tool-call fraction, the
   generate fraction, and elapsed time divided by the 60-minute clock.
5. **Style-bank data**: the latent's AUROC on bank F for c\* (`auroc_F`), and the fraction of bank-F c\* texts on
   which it fires (`fire_F`).

### 1.2 Attributing an agent's probe text to an option (scripted)

For each concept c, its keyword set K(c) is built once, from dataset texts only, before any agent episode is
classified:
- **Topic concepts**: the label's content words (lowercased, at least 4 characters, stopwords removed, compared by
  their first 5 characters), plus the 15 words with the highest TF-IDF in c's split A+B texts against all other
  concepts. A word must appear in >= 10% of c's texts to count.
- **Language concepts**: the 30 most frequent words of c's split-A texts that appear in < 1% of the other concepts'
  texts, plus a script test (for example Cyrillic, Devanagari or Arabic letters).

A probe text is attributed to the option o, among the slot's 20 options plus the anchor on a null slot, that has the
most distinct K(o) hits. It needs at least 1 hit and a unique maximum. Otherwise the text is *unattributed*.
**Attribution coverage** of a slot is the share of its probe texts that are attributed.

### 1.3 Thresholds (all frozen here)

| Name | Definition |
|---|---|
| silent | a probe text whose max activation of the slot's latent is < 0.1 x m\* |
| adequately probed | n_true >= 3 **and** f_fwd >= 0.50 at submit |
| early stop | f_fwd < 0.50 at submit **and** tool-call fraction < 0.90 **and** generate fraction < 0.90 **and** elapsed < 0.90 of the clock (that is, no other limit forced the stop) |
| close option | option o with AUROC_C(o) >= 0.60, **or** o a taxonomy sibling of c\* |
| word-driven claim | among the agent's probe texts on this slot with activation >= 0.5 x the max it saw on this slot, >= 50% (and at least 2 texts) peak on a token whose first 5 characters match a content word of the claimed option's label; **or** the claimed option's bare name fires the latent (names cache > 0) while AUROC_C(claimed) < 0.65 |
| style-robust latent | auroc_F >= 0.85 and fire_F >= 0.50 |
| key defect | planted: the key's option is not the best menu option on split C, or margin_C < 0.05; null: best menu AUROC_C >= 0.80 |

### 1.4 Rules, in fixed priority order (the first rule that matches wins)

Wrong-reason rules come before right-reason rules on purpose. Any failure with a scripted wrong-reason cause is
counted as wrong-reason. This makes the study conservative against concluding "hard for the right reasons".

| # | Label | Group | Rule (applied to a failed slot of a valid episode) |
|---|---|---|---|
| 1 | **W4 parse / harness / key bug** | wrong | Any of: (a) key defect (1.3); (b) the slot is missing from an accepted submission, or the broker logged a rejected submit and the final accepted answer for the slot differs from the agent's last rejected answer for it; (c) a call on this slot's latent got a harness-side error (server exception, timeout, queue starvation, never an argument error) and no later call on this latent succeeded; (d) more than 10% of the forward cap was charged but never delivered. |
| 2 | **U0 unclassifiable data → manual review** | manual | The tool log is missing or truncated for this slot, or attribution coverage < 50% of the slot's probe texts (with at least 1 probe). |
| 3 | **W3 tool / format error** | wrong | Any of: (a) the slot's latent was queried at a wrong layer but never at its own layer; (b) >= 3 agent-caused error responses (bad JSON, bad arguments, over-cap requests) on calls for this slot's latent; (c) >= 30% of all the episode's tool calls returned agent-caused errors. |
| 4 | **W1 latent silent because of style** | wrong | Planted slot, the agent answered "nothing found" or a wrong option, n_true >= 2 and **every** true-option probe was silent, and the latent is style-robust on bank F. |
| 5 | **W2 early stop with unused budget** | wrong | Early stop (1.3) holds. |
| 6 | **R2 word-vs-concept confusion** | right | The agent claimed an option (planted-wrong or null), and the claim is word-driven (1.3). |
| 7 | **R1 close-sibling confusion** | right | Planted slot, the agent claimed a wrong option that is a close option (1.3). |
| 8 | **R3 null over-claim** | right | Null slot and the agent claimed an option. Reported in two subtypes: the claimed option is a sibling of the anchor or has AUROC_C >= 0.60 ("lured"), or not ("unrelated"). |
| 9 | **R4 wrong hypothesis after adequate probing** | right | Planted slot, adequately probed (1.3), and the answer is "nothing found" or a non-close wrong option. Subtype flag "ignored own evidence": the true option's mean probe activation was the highest among options probed with >= 2 texts. |
| 10 | **R4s search failure** | right (primary), wrong (sensitivity) | Planted slot, not early-stopped (f_fwd >= 0.50), but n_true < 3: the agent spent its budget yet never tested the true option enough. |
| 11 | **U1 residual → manual review** | manual | Anything not matched above. |

The rules are mutually exclusive because the first match wins, and exhaustive because of U1. Every failed slot gets
exactly one label.

**Sensitivity analysis (pre-registered).** Criterion 2 of the decision rule (section 4) is computed twice. The primary
version counts R4s as right-reason. The conservative version counts R4s as wrong-reason. Both are reported. A verdict of
"right reasons" is called *robust* only if both versions reach 60%.

**Manual review buckets.** A reviewer resolves U0 and U1 slots with the same definitions, by reading the transcript.
If more than 15% of an arm's failed slots are still unresolved after review, criterion 2 is "not evaluable" for that
arm.

### 1.5 Human spot-check

- **Sample.** Every U0 and U1 slot. Plus, per arm, a stratified random sample of the scripted labels: 20% of the arm's
  failed slots, with at least 2 per label present in the arm (all of them if fewer than 2) and at most 30 per arm. Also
  5 passed slots per arm, to check for lucky successes. The draw uses Python `random.Random(20261002)` over the list
  sorted by (arm, episode id, slot index), taking labels in the table order above.
- **Who.** A fresh reviewer (a subagent that did not write `classify.py`, or Dan) gets the transcript, the full tool
  log, the answer key and section 1 of this file, but **not** the scripted label. The reviewer assigns a label.
- **Attribution check.** The same reviewer labels 100 randomly drawn probe texts (same seed) with the option they are
  about. Attribution accuracy is reported.
- **Reported.** Agreement between script and reviewer on the full label (raw %) and on the right/wrong split (raw % and
  Cohen's kappa). If right/wrong agreement is < 80%, or attribution accuracy is < 80%, the failure-composition numbers
  are reported as **unreliable** and criterion 2 cannot support a "right reasons" verdict. Disagreements are listed;
  the scripted label stays the primary number.
- **Lucky successes.** A passed planted slot with n_true < 2 is counted as a "thin-evidence success" and reported per
  arm. It is not reclassified.

## 2. Experiments

All agents are fresh Claude Code subagents run under the containment harness. They get only the rendered TASK.md, the
tool docs and the answer format, and they make no hosted-API calls. Results are labelled "fresh Claude Code subagent
(model X)". Every agent episode is one attempt. Invalid episodes (audit failure, leak, queue wait > 50% of elapsed time)
are excluded from all rates, re-run once on the same instance, and counted in the report. **Any cell with n < 20 valid
episodes is reported as bug-finding only and never as a rate.**

### Step 1: tool and task clarity audit (CPU)
Read the TASK.md, the tool docs and all recorded transcripts. List the ambiguities and every transcript where a tool was
considered and then not used. Write `clarity.md` and a revised TASK.md; the old one is kept as `TASK_old.md`. Metric:
the number of ambiguities found and fixed. A check: the revised prompt must name no method and give no answer hint (the
harness `check_task_md` must pass, plus a reviewer read).

### Step 2: style-robustness filter and filtered v2 pool (1 GPU job)
- Bank F: about 20 styled texts for each of the 232 concepts, so about 4,600 texts.
- For **every** pooled latent, take its anchor concept c\*. Compute auroc_F(c\*) against all other concepts' bank-F
  texts. Keep the latent if auroc_F >= 0.85. The filter is applied to the latent and its anchor, which is the same for
  planted and null slots. This keeps null and planted latents indistinguishable (SPEC "no fingerprinting").
- Regenerate a filtered pool from the kept latents with generator v2 unchanged otherwise: 60 instances per tier, seeds
  8000+ (T1), 108000+ (T2) and 208000+ (T3). Re-run the fingerprint check on it.
- **Report:** survivors per layer and per concept family (topic / language). For survivors vs dropped latents: generic
  fire rate, fire rate on c\*, layer mix, A/B/C margins, and the reference solver's one-shot accuracy on slots from each
  group. **"The filter selects easier latents"** is declared if survivors' median margin_C exceeds the dropped
  latents' by > 0.05, **or** survivors' median generic fire rate is > 1.25x the dropped latents', **or** reference
  planted accuracy on survivors exceeds that on dropped latents by > 0.05. Dropped slots are logged with the reason.

### Step 3: baselines on the filtered v2 pool (1 GPU job, shared)
On all 180 filtered instances: reference (one-shot), black-box, self_probe and template_probe (`recipe_baseline.py`),
the zero-effort recipes, and the new **style-robust recipe** with two variants:
- **SR-max**: for each slot, run 6 bank-R texts per option (one per style, 120 texts per slot, at most 600 forward
  units per 5-slot episode). Claim the option with the highest mean max activation.
- **SR-thr**: as SR-max, but claim only if that option's AUROC against the other options' bank-R texts is >= 0.78 (the
  reference's threshold). Otherwise answer "nothing found".
Bank R is written independently of bank F, so the recipe cannot pass by reusing the filter's texts.
Metrics per baseline and tier: pass rate (Wilson 95%), planted accuracy, planted "nothing found" rate, null
false-claim rate. **Gate:** every non-reference recipe except SR has planted accuracy <= 0.15, and the reference has
planted accuracy >= 0.95. **Stop rule:** see section 4.

### Step 4: agent arms (only if the stop rule did not fire; Opus-class main model unless stated)

**Instance assignment.**
- **Core set S**: 20 instances from the kept filtered T2 pool (close distractors, 1200 forward units), drawn with
  `random.Random(20261002)`. Redraw with seed +1 until S has >= 40 planted and >= 25 null slots.
- **Smoke set**: 3 other filtered T2 instances, disjoint from S. Each arm is smoked on them first, and the transcripts
  are read. Smoke episodes are never pooled into results.

| Arm | What changes | Instances | n |
|---|---|---|---|
| A baseline | revised TASK.md | S | 20 |
| B clarity ablation | old TASK.md (`TASK_old.md`) | S | 20 |
| C persistence | revised TASK.md; `submit` refused until >= 60% of the forward budget is used | S | 20 |
| D oracle examples | revised TASK.md plus a file with 3 in-style dataset texts per option (from a split used neither for the key nor for the held-out check) | S | 20 |
| E-sim-random | menus regenerated with T1 "far" closeness, same latents and null pattern as S | S' (paired with S) | 20 |
| E-sim-near | menus regenerated as "near-sibling": the 19 distractors are the non-excluded menu-universe concepts with the highest AUROC_B for the latent (the confusable set stays excluded, so keys keep >= 0.25 margin on A); same latents and null pattern as S | S'' (paired) | 20 |
| E-sim-sibling | = arm A | S | (20) |
| E-budget-low | forward cap 550 | S | 20 |
| E-budget-high | forward cap 2400 | S | 20 |
| E-dense | new T2 instances whose slot latents all have generic fire rate >= 0.111 | 20 new | 20 |
| E-sparse | new T2 instances whose slot latents all have generic fire rate < 0.054 | 20 new | 20 |
| F small model | revised TASK.md, haiku-class model | S | 20 |

If the generator cannot keep latents fixed while changing menus, E-sim uses independent instances and is flagged as
unpaired. The order of execution, if time runs out, is: A, C, D, F, B, E-sim, E-density, E-budget. Unfinished cells are
reported as missing, not estimated.

**Metrics per arm** (step 5): pass rate with Wilson 95% CI; planted accuracy; null false-claim rate; planted "nothing
found" rate; f_fwd at submit (median, IQR); failure-label counts (section 1); planted accuracy by density tertile and by
distractor-closeness tertile (cuts frozen in section 0); tool-use counts (calls per tool; share of episodes using
`top_latents`, `generate` and `vocab_projection`); thin-evidence successes.
**Paired comparisons** (A vs B, C, D, F, E-budget-low/high, E-sim): the difference in pass rate and in slot accuracy,
with a 95% paired bootstrap CI (10,000 resamples by instance), and an exact McNemar test on slot correctness.
**Plots**: pass rate vs similarity level (random / sibling / near-sibling) and planted accuracy vs `dis_C` tertile;
failure-label composition per arm.

## 3. Predictions

All agent numbers are **guesses**. Recipe and reference numbers are anchored on measured v1/v2 runs where stated. v2
measured anchors, recomputed today from recorded runs (`fx/recipes.py`): reference planted accuracy 0.97 and pass
178/202 = 0.88; self_probe planted accuracy 0.16 with planted "nothing found" 0.78; black-box 0.00. v1: self_probe 0.14,
template_probe 0.11, reference 0.96; planted "nothing found" recipes 0.73-0.79 and reference 0.02. Sonnet v1 planted
5/8, luna 0/3. Every v1 agent stopped at 8-62% of its budget, and none called `top_latents` or `generate`.

| # | Experiment / metric | Prediction | Basis |
|---|---|---|---|
| P1 | Step 1: undocumented items found | >= 4 (response keys `max`/`acts`, how budget is counted, what `top_latents`/`generate` are for, JSON quoting of apostrophes) | forensics section 5 items 9-10 |
| P2 | Step 1: transcripts that considered and dropped `top_latents` or `generate` | >= 1 of the 4 valid v1 transcripts | guess |
| P3 | Step 2: share of pooled latents surviving the filter (auroc_F >= 0.85) | 0.45 overall (plausible 0.30-0.60); languages about 0.80, topics about 0.35 | style check: 50% of latents recover their concept on hand-written probes; airport 11% |
| P4 | Step 2: survival by layer | L6 0.35, L12 0.50, L18 0.55 (deeper is more style-robust) | guess |
| P5 | Step 2: "the filter selects easier latents" | **yes** (p ~ 0.6): survivors have higher margin_C (+0.03 to +0.06) and higher fire rate on c\*; generic density ratio about 1.1x | guess: separating on 20 varied texts favours broad, strongly firing latents |
| P6 | Step 2: filtered pool still passes the fingerprint check (CV AUROC <= 0.6) | yes | filter applied identically to null and planted latents |
| P7 | Step 3: reference planted accuracy / one-shot pass | 0.97 / 0.88 | v2 measured 0.97 / 0.88 |
| P8 | Step 3: black-box planted accuracy / pass | 0.00 / about 0.05 (all-null episodes only) | v2 measured |
| P9 | Step 3: self_probe planted accuracy (planted "nothing found") | **0.30** (0.55); gate **fails** (> 0.15) | v2 0.16 / 0.78; the filter keeps latents that fire on non-dataset text, which helps model-written probes |
| P10 | Step 3: template_probe planted accuracy (planted "nothing found") | 0.20 (0.65); gate fails or is borderline | v1 0.11 / 0.78, same reasoning |
| P11 | Step 3: SR-max planted accuracy | **0.75** (0.55-0.85) | bank R is a different writer but the same kind of LLM-written varied text that the filter selected on; true AUROC_F >= 0.85 vs distractors around 0.6 |
| P12 | Step 3: SR-thr planted accuracy / null false claims / pass | 0.70 / 0.10 / 0.35 | as P11, with a nothing-found threshold |
| P13 | **Step 3 stop rule fires** (SR planted accuracy > 0.50) | **yes, probability about 0.7** | P11 |
| P14 | Arm A pass rate | 0.25 (Wilson CI about 0.11-0.47) | per-slot planted 0.65, null 0.80; about 2.4 planted and 1.6 null slots per episode: 0.65^2.4 x 0.8^1.6 = 0.25 |
| P15 | Arm A planted accuracy / null false-claim rate / planted "nothing found" | 0.65 / 0.20 / 0.15 | Sonnet v1 5/8; the filter removes the MMA-type silent latents |
| P16 | Arm A f_fwd at submit (median) | 0.35 | v1: 0.08-0.62 |
| P17 | Arm A failure composition | W2 45%, W1 5%, W3 5%, W4 <= 5%, right-reason (R1-R4s) 40%; **criterion 2 fails in A** | v1: every miss under-used budget (5/7 under-probed) |
| P18 | Arm A tool use | `top_latents` in <= 20% of episodes, `generate` in <= 20% | v1: 0 of 4 |
| P19 | Arm B pass (A − B) | 0.20 (+0.05, not detectable at n=20); W3 share about 15% | old docs omit response keys; quoting |
| P20 | Arm C pass / planted accuracy / f_fwd | 0.35 / 0.75 / 0.65; W2 about 0%; right-reason share 0.70 (criterion 2 met) | forensics: 5 per option would have found curler about 98% of the time |
| P21 | Arm D pass / planted accuracy / planted "nothing found" | 0.45 / 0.82 / 0.05; W1 about 0 | in-style texts remove the residual style dependence |
| P22 | Arm E-sim pass: random / sibling (= A) / near-sibling | 0.30 / 0.25 / 0.18 | v2 excludes confusable options, so T1 vs T2 differ little (median dis_C 0.603 vs 0.610); near-sibling raises dis_C |
| P23 | Criterion 4 (pass drops with similarity) | right direction, **not significant** (one-sided p > 0.10) → "ambiguous" | P22 with about 50 planted slots per level |
| P24 | Planted accuracy by dis_C tertile (pooled A+C+D) | low 0.80 / mid 0.72 / high 0.62 | guess |
| P25 | Arm E-budget pass: low 550 / high 2400 | 0.20 / 0.27 (more budget barely helps without persistence) | agents use < 50% anyway |
| P26 | Arm E-density planted accuracy: dense / sparse | 0.55 / 0.72 (pass 0.18 / 0.30) | v1 misses clustered on dense latents (78%, 42% fire rates) |
| P27 | Arm F (haiku) pass / planted accuracy / planted "nothing found" / f_fwd | 0.08 / 0.40 / 0.35 / 0.20; right-reason share 0.30 | luna-like pattern; PREDICTIONS.md small-agent guess |
| P28 | A vs F separation | difference about +0.17 in pass; 95% CIs overlap at n=20, slot-level McNemar p < 0.05 | guess |
| P29 | Spot-check agreement (right/wrong split) | >= 85%; attribution accuracy >= 85% | guess |
| P30 | **Overall verdict** | **"FeatureMatch does not require interpretability" (stop rule) with p about 0.7.** If the stop rule does not fire: "right reasons, conditional on persistence" (A fails criterion 2 through W2, C meets all of 1-3, criterion 4 ambiguous) | P13, P17, P20, P23 |

## 4. Decision rule, stop rule and ambiguous outcomes

**Stop rule (step 3, checked first).** If the style-robust recipe passes most planted slots, meaning planted accuracy
> 0.50 for SR-max **or** SR-thr on the filtered pool (all tiers pooled, n >= 100 planted slots), then **FeatureMatch does
not require interpretability: stop and report.** Step 4 does not run. If the point estimate is above 0.50 but its
Wilson CI includes 0.50, the stop rule still fires and is reported as "borderline".

**Decision rule (verbatim from PLAN.md).** FeatureMatch is "hard for the right reasons" if, on v2 after the style
filter:
1. the pass rate is 10-70%;
2. >= 60% of failures are right-reason;
3. the agent's planted "nothing found" rate is closer to the reference than to the recipes;
4. the pass rate drops as sibling similarity rises.

**How each criterion is evaluated.**
- **Primary arm**: A, the task as it would ship. Arm C is secondary.
- **(1)**: arm A pass rate in [0.10, 0.70]. The endpoints count as met.
- **(2)**: right-reason failed slots divided by (right + wrong) failed slots, after manual review of U0/U1 (section 1.4).
  At least 0.60 counts as met. Computed in the primary and conservative (R4s as wrong) versions.
- **(3)**: let r be arm A's planted "nothing found" rate, r_ref the reference's and r_rec the mean of self_probe's and
  template_probe's, both measured on the filtered pool in step 3. The criterion is met if |r − r_ref| < |r − r_rec|.
  If step 3 did not measure them, use r_ref = 0.02 and r_rec = 0.76 (v1). An exact tie counts as not met.
- **(4)**: met if the E-sim pass rates are non-increasing from random to sibling to near-sibling, **and** a logistic
  regression of planted-slot correctness on similarity level (0/1/2) has a negative slope with one-sided p < 0.10.
  "Ambiguous" if only the first part holds. If E-sim did not run, use the pooled planted-slot logistic regression on
  `dis_C` across arms A, C and D, flagged as a fallback.

**Verdict labels.**
- **Right reasons (robust)**: arm A meets 1-4, n >= 20, and criterion 2 holds in both versions.
- **Right reasons**: arm A meets 1-4 in the primary version only.
- **Right reasons, conditional on persistence**: A misses only criterion 2 (or 1), with W2 the largest wrong-reason
  label, while C meets 1-3 and criterion 4 is met or ambiguous.
- **Wrong reasons**: criterion 2 fails in both A and C, or criterion 3 fails in A.
- **Not interpretability**: the stop rule fired.
- **Too hard / too easy**: criterion 1 fails in both A and C (both below 0.10, or both above 0.70).
- **Inconclusive**: anything else, including criterion 4 "ambiguous" with everything else met. That case is reported
  as "right reasons, similarity dial unconfirmed".

**Robustness of reported numbers.** A criterion whose point estimate meets its threshold but whose Wilson 95% CI
crosses it is reported as "met (point estimate), not robust". No mid-band claim is made from n < 20. Criteria are
reported one by one, so a reader can see which part of a verdict failed.

## 5. Freezing

This file is committed before any experiment runs and is **never edited afterwards**. Results, deviations and
corrections go in dated sections appended below this line. Each says what changed, when, and whether any results
existed at the time.

---

## Amendment 1 (2026-10-02, before any experiment or result)

Written after Dan's STEP 1b instruction in `PLAN.md` (2026-10-02 ~01:35 UTC). **State at the time of writing:** step 1
(the clarity audit, CPU only) is done. Banks F and R exist as text files only (`diagnosis/banks/F`, `diagnosis/banks/R`:
232 concepts x 10 styles x 2 texts each). No activation has been computed on any bank text, no filtered pool exists,
no baseline has run on a filtered pool, and no agent has run on v2. Nothing in sections 0-5 above is changed. Where
this amendment and the text above disagree, this amendment wins, and the change is named below.

### A1.1 Multi-style answer key (step 2; replaces the step-2 filter rule "keep the latent if auroc_F >= 0.85")

**Why.** The current key is defined on encyclopedic dataset text only (split A). Step 1b makes the style filter the
main validity fix: a latent's concept must be the class it separates best **across styles**, not just on
Wikipedia-like text.

**Text sets** (all activations are the max over tokens, texts truncated to 64 tokens exactly as in the tools):
- Dataset splits A, B, C as before (`concepts.py`: 40 / 24 / 20 texts per concept; C exists for every concept that can
  be an anchor or a menu option).
- Bank F is split per concept into two halves, fixed now, before any bank activation exists: for each of the 10
  styles, the **first** text listed for that style in the bank-F file goes to **F1**, the second to **F2**. So F1 and
  F2 each hold 10 texts per concept, one per style.
- Bank R is not used in step 2 at all. It stays reserved for the style-robust recipe (step 3).

**AUROCs.** AUROC_X(j, c) for a text set X is computed as in `auroc.py`: positives are concept c's texts in X,
negatives are the X texts of every other concept that has texts in X.

**Definitions, per pooled latent j at layer L** (the pool of generator v2, `Tables.pools`):
1. *Original key*: c\* = argmax over all concepts of AUROC_A(j, c) (exactly the generator's anchor).
2. *Multi-style key*: k_ms = argmax over all concepts of M(c) = 0.5 x AUROC_A(j, c) + 0.5 x AUROC_F1(j, c). Dataset
   text and styled text get equal weight, even though A has 40 texts per concept and F1 has 10.
3. *Held-out agreement check* (on C + F2, used by neither key definition): k_ho = argmax over all concepts of
   H(c) = 0.5 x AUROC_C(j, c) + 0.5 x AUROC_F2(j, c). For a concept without a split C, H(c) = AUROC_F2(j, c).
4. *Style-robustness filter* (also on C + F2): AUROC_F2(j, c\*) >= 0.85 **and** AUROC_C(j, c\*) >= 0.85.

**Keep rule.** A pooled latent is kept only if k_ms = c\* **and** k_ho = c\* **and** the filter (4) passes. The rule
acts on latents, before the filtered pool is regenerated, and it is the same for latents that end up in planted and
in null slots. That keeps null and planted latents indistinguishable, as in step 2 above. Consequences:
- A **planted** slot of the filtered pool is kept only if its multi-style key equals its original key (its answer c\*
  is the multi-style best class among all concepts, so also among its 20 options). The generator asserts this for
  every planted slot, and the count of violations (expected 0) is reported.
- On a **null** slot, c\* is not on the menu and c\* is also the multi-style best class, so "nothing found" is the
  answer under both definitions.
- For information only (no slot is dropped for it), each filtered slot also reports its multi-style margin: M of the
  answer minus the best other option's M (planted), or 0.65 minus the best option's M (null).

**Logs** (both committed, in `diagnosis/`):
- `key_check.jsonl`: one row per pooled latent: layer, real latent id, c\*, k_ms, k_ho, the top 3 concepts under A, M
  and H with their scores, AUROC_C(c\*), AUROC_F1(c\*), AUROC_F2(c\*), kept (yes/no) and the reason for dropping
  (`ms_key_differs`, `heldout_key_differs`, `not_style_robust`, several allowed).
- `slot_disagreements.jsonl`: every slot of the **current** v2 pool (180 instances) whose latent fails the keep rule,
  with instance id, slot, kind, the reason, and for planted slots the option the multi-style key and the held-out
  check would pick (an option number or "not on menu").
- The step-2 report gives these counts by layer, by concept family (topic / language) and by slot kind.

The remaining step-2 rules are unchanged: same regeneration (60 instances per tier, seeds 8000+, 108000+, 208000+),
the fingerprint check, and the "the filter selects easier latents" test and its thresholds. The classifier's
`style-robust latent` definition (section 1.3: auroc_F and fire_F on all of bank F) is also unchanged.

### A1.2 New arm G: disclosure of how the key is defined (step 4)

| Arm | What changes | Instances | n |
|---|---|---|---|
| G disclosure | `agent_prompt_disclosure.md` = the arm-A template plus one paragraph in the task description: "**How the answer key is defined:** options are disjoint dataset classes; the correct option is the class the latent separates best across varied styles." Nothing else differs (checked by diffing the rendered TASK.md files, see A1.6). | S | 20 |

- Comparison: G vs A, paired on S, with every per-arm metric and the paired tests of section 2 (bootstrap CI on the
  pass-rate and slot-accuracy differences, exact McNemar on slots). G is smoked on the smoke set first, like every arm.
- Execution order if time runs out becomes: **A, G**, C, D, F, B, E-sim, E-density, E-budget.
- G is a secondary arm for the decision rule (the primary arm stays A). The verdict text reports whether disclosure
  changes the A-based verdict.

### A1.3 Arm A's template gains clearer tool documentation (before any agent run)

Step 1b asks for `top_latents` and `generate` to be documented so that "see what fires the latent" is a visible,
measurable option. `agent_prompt_revised.md` (arm A, and the base of C, D, E, F, G) now says, for **every** tool, what
it shows, its exact response keys and its cost. For example: `top_latents` "shows which latents of one layer are most
active on each text you write, and at which token, including latents you did not name"; `generate` "shows how the
model continues a prompt you write". The same is done for `latent_activations`, `vocab_projection`,
`next_token_logits` and `task_info`, and the text says that the list order is not a suggested order. It names no
method, recommends no tool and hints at no answer. Placeholders and the answer format are byte-identical. Arm B (old
template) is unchanged. Predictions P14-P18 and P20-P28 above were made for the earlier wording; they stay as
recorded, and A1.5 adds the new ones.

### A1.4 Tool-usage metrics (step 5; added to the per-arm metrics)

From the full tool log of each valid episode (scripted, in `classify.py`):
- **Counts per episode** of `top_latents`, `generate`, `vocab_projection` and `next_token_logits` calls (calls
  rejected with an error included, and also reported separately). Per arm: the share of episodes with >= 1 call of
  each tool, and the mean and median count per episode.
- **Used on the target latent**, per slot and per tool:
  - `vocab_projection`: called with this slot's (layer, latent).
  - `top_latents`: called at this slot's layer. Sub-flag *target surfaced*: this slot's latent appears in at least one
    returned `top` list.
  - `generate`: a completion is used as a probe for this slot, meaning a substring of >= 20 characters of the
    completion appears in a later `latent_activations` text sent with this slot's (layer, latent) or in a later
    `top_latents` text at this slot's layer.
  - `next_token_logits`: one of its prompts is later sent unchanged (after stripping whitespace) as a probe for this
    slot, as defined for `generate`.
  Per arm: the share of slots with each flag, and the share of episodes with the flag on >= 1 slot.
- **Association (descriptive only):** planted accuracy of slots with vs without each "used on the target" flag. This
  is observational, never read as a causal effect of the tool.
- Reported for every arm. A vs B tests whether documentation changes tool use, and A vs G whether disclosure does.

### A1.5 RESULTS.md: how much of the pass rate style explains (required section)

`RESULTS.md` must contain a section "How much of the pass rate does style explain?" with these numbers, each with a
95% paired bootstrap CI (10,000 resamples by instance, `random.Random(20261002)`), on the core set S. Planted
accuracy is the primary measure, because SR-max never answers "nothing found", so its pass rate is near 0 by
construction. Here pa(X) is planted accuracy of X on S; BB is the black-box baseline (the floor); ref is the reference
solver.
1. **Q_SR (recipe share of the agent's success)** = (pa(SR-max) − pa(BB)) / (pa(A) − pa(BB)). This is how much of
   arm A's planted accuracy a scripted recipe reproduces with no reasoning, only by probing each option in several
   styles. Also reported: pa(SR-max) / pa(ref), and the same two numbers for SR-thr. Secondary, pass-rate version:
   pass(SR-thr) / pass(A).
2. **Q_D (oracle-examples share of the agent's gap)** = (pa(D) − pa(A)) / (pa(ref) − pa(A)). This is how much of the
   gap between the agent and the reference closes when the agent is simply given in-style example texts per option.
   Secondary, pass-rate version: (pass(D) − pass(A)) / (pass(ref) − pass(A)).
3. Edge cases. If pa(A) − pa(BB) < 0.05, Q_SR is undefined and pa(SR-max) / pa(ref) is used instead. If
   pa(ref) − pa(A) < 0.05, Q_D is undefined (there is no gap to close), and D − A is reported raw. If the stop rule
   (section 4) fired, arms A and D do not exist: only pa(SR-max) / pa(ref) and pa(SR-thr) / pa(ref) on the whole
   filtered pool are reported, and the rule below uses them.

**Plain-language rule (binding).** If Q_SR > 0.50 **or** Q_D > 0.50 (point estimates; in the stop-rule case,
pa(SR) / pa(ref) > 0.50), then RESULTS.md and VERDICT.md both say, in these words: **"FeatureMatch is mostly measuring
style, not interpretability"**, followed by the number(s) that triggered it. If the triggering CI includes 0.50, the
sentence ends with "(borderline)". This sentence is required even if the decision rule of section 4 would otherwise
say "right reasons". Both numbers are reported whatever their values.

### A1.6 Template check (done before this commit; no model call, no GPU)

`common.sandbox prepare --prompt-template` was run with each of the three templates on the v2 instance
`fm-t2-09ba84b566` (read-only, from `~/wt/featurematch/tasks/featurematch/instances`), run dirs under
`~/.claude/jobs/e4652089/tmp/fmdiag_amend/`. Episodes: old `epe08c1131ef`, revised `ep6538e06d14`, disclosure
`epfd3f47cb46`. All rendered with 0 placeholders left and passed `check_task_md`, and each was finished at once with
0 tool calls and 0 forward units (`valid: true`, unsubmitted). Diffs of the rendered TASK.md files:
- old: byte-identical to the step-1 render of the same instance (`fmdiag_clarity/TASK_old_fm-t2-09ba84b566.md`).
- revised vs the step-1 revised render: one hunk, the per-tool section only (2,518 → 2,815 words).
- disclosure vs revised: one hunk, the 2-line disclosure paragraph only (2,839 words).

### A1.7 Predictions for the new quantities

All are **guesses** unless a basis is given. None can be checked against data yet.

| # | Quantity | Prediction | Basis |
|---|---|---|---|
| P31 | Share of pooled latents with k_ms = c\* | 0.90 overall; languages 0.98, topics 0.87 | guess. The 50/50 weighting still gives split A (where c\* has AUROC >= 0.90 and at most 2 other concepts come within 0.10 of it) half the score; a flip needs another concept to beat c\* by about 0.3 on F1 |
| P32 | Share with k_ho = c\* | 0.85 | as P31, plus noise from 10 F2 texts |
| P33 | Share passing the filter (AUROC_F2 >= 0.85 and AUROC_C >= 0.85) | 0.42 (plausible 0.25-0.60) | P3 (0.45) minus a little for the noisier 10-text half; AUROC_C(c\*) >= 0.85 fails rarely (pool needs A >= 0.90, B >= 0.85) |
| P34 | Share kept by the full keep rule | **0.38** (plausible 0.22-0.55); languages 0.75, topics 0.30 | P31-P33; the three conditions are positively correlated |
| P35 | Latents dropped **only** for a key disagreement (k_ms or k_ho differs, filter passed) | <= 5% of pooled latents | guess: a latent robust enough to pass the filter rarely prefers another class |
| P36 | Slots of the current v2 pool that fail the keep rule | about 60% of slots, the same for planted and null within 5 points | P34; the rule is kind-blind |
| P37 | Arm G pass rate (vs A) | 0.32 (A 0.25, P14); difference +0.07, paired CI includes 0 | guess: "across varied styles" pushes agents to probe in several styles, which mainly helps planted slots |
| P38 | Arm G planted accuracy / null false-claim rate / planted "nothing found" | 0.72 / 0.22 / 0.10 (A: 0.65 / 0.20 / 0.15, P15) | guess: fewer silent-probe misses; slightly more claiming on nulls |
| P39 | Arm G failure composition | W1 share about half of arm A's; right-reason share 0.45 (A 0.40) | guess |
| P40 | Tool use, arm A with the new docs (share of episodes with >= 1 call) | `top_latents` 0.35, `generate` 0.25, `vocab_projection` 0.75, `next_token_logits` 0.05 | guess; v1 agents used only `latent_activations` and `vocab_projection` (clarity.md), and the docs now say what the other tools show |
| P41 | Used on the target latent, arm A | `vocab_projection` on a slot latent in >= 95% of its calls; `top_latents` at a slot's layer in 90% of using episodes, target surfaced in 40% of them; `generate` completion used as a probe in 30% of using episodes | guess |
| P42 | Tool use in the other arms (`top_latents` / `generate`, share of episodes) | B 0.10 / 0.10; C 0.45 / 0.35; D 0.20 / 0.15; F 0.10 / 0.10; G 0.40 / 0.30 | guess: C must spend budget; D has examples, so it needs to generate fewer texts; the small model explores less |
| P43 | Association: planted accuracy of slots with `top_latents` target surfaced vs without | 0.80 vs 0.62 (descriptive) | guess |
| P44 | Q_SR (planted accuracy) and pa(SR-max)/pa(ref) | Q_SR = (0.75 − 0.00) / (0.65 − 0.00) = **1.15**; pa(SR-max)/pa(ref) = 0.75 / 0.97 = 0.77 | P11, P15, P7, P8 |
| P45 | Q_D (planted accuracy) | (0.82 − 0.65) / (0.97 − 0.65) = **0.53** (borderline; CI wide) | P21, P15, P7 |
| P46 | **The plain-language rule fires** ("FeatureMatch is mostly measuring style, not interpretability") | **yes, probability about 0.75**, mostly through Q_SR or the stop rule (P13, p ~0.7); Q_D alone would trigger it with p ~0.5 | P44, P45 |
