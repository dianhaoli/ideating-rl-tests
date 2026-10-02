# LatentKnockout feasibility: pre-registered predictions and decision rule

Written 2026-10-02T06:15Z (date -u), before any feasibility result exists. **Frozen once committed.** Outcomes,
deviations and corrections go in dated sections appended below the "Freeze" line, never in the text above it.

## 0. What existed when this was written (exact list)

- `tasks/latentknockout/NOTES.md`: one entry (01:25Z) with setup notes only: the goal, the error-term-preserving
  ablation, and the use of the read-only bf16 copy of gemma-2-2b. **No measured number.**
- `runs/latentknockout/20261002T0128_validate/log.txt` (64 bytes, 01:28Z), whole content:
  `[gpuq] waiting for 9.0 GB (light) label=latentknockout-validate`. The job was never admitted to the GPU. There is no
  `validated.json`, `validation_counts.json` or `exactness.json`. **No model output of any kind has been seen.**
- WIP code from the stopped agent (`lk_core.py`, `lk_data.py`, `validate.py`). I read it, so these predictions use its
  entity lists and its 5 templates per family (10 for country_capital). I have not run it and have not checked it
  for correctness. `__pycache__/lk_data.cpython-313.pyc` shows that `lk_data` was imported once (01:26Z). That gives no result.
- Prior evidence used: the EditHunt findings in `CONTEXT.md` and the FeatureMatch diagnosis (`STEP3.md`, a different
  task). Literature is cited from memory. Numbers marked "verify" need checking before anyone quotes them.

## 1. Terms and what each prediction measures

- **Item**: one (entity, template) prompt with a one-token answer, e.g. "Dallas is a city in the state of" -> " Texas".
  **Validated** = the clean model's top-1 next token is an accepted answer (margin > 0). "Clean top-1 accuracy" below
  is the same number: validated items / all items.
- **Cell** = (family, group, layer), e.g. (city_state, Texas, 12). The **group-level families** are city_state (12
  states), city_capital (12), country_lang (7 languages), athlete_sport (6 sports) and langid (8 languages), giving
  45 (family, group) pairs and 135 cells. country_capital (30 single countries, held-out = templates only) is reported
  separately and does not count toward GO. It is too easy, and it does not test generalisation over entities.
- **Ablate S at layer L**: x' = x - sum_{i in S} f_i(x) W_dec[i] at every non-BOS position. The SAE error term is kept.
- **R** = Effect x Preserve x (1 - min(1, KL/kappa)), measured on **held-out** prompts:
  - Effect: the share of held-out target items whose clean answer stops being top-1 (a hard flip).
  - Preserve: the share of held-out sibling-group items (same family, other groups) whose clean top-1 survives.
  - KL: the mean next-token KL on unrelated text (wikitext).
- **Held-out (T)**: new entities with templates not used in the examples, in the same plain style.
  **Held-out (S)**: new entities in *style-varied* prompts, written to differ from the examples in register and format:
  chatty first person, a dialogue, a list or table row, a code comment or key-value record, a news sentence.
  **Unless a line says (T), the predictions use (S)**, because the grader must measure generalisation (FeatureMatch lesson 1).
- **Reference**: the reference solver gets only the example target prompts and controls it builds itself (no held-out
  data, no metadata). It ranks latents by attribution (gradient x activation on the answer log-prob, target minus
  sibling), then runs greedy forward selection with real ablations, then a small exhaustive search over the top
  candidates. "Best achievable R at k" = held-out R of the best set of size <= k that the reference finds. A cell is
  **feasible** if R_ref(k <= 5) >= 0.5.
- **Ratio** for a baseline = R_baseline / R_ref, computed on feasible cells (k = 5 for set baselines).
- Numbers are point predictions with a plausible range in brackets ("80% of my probability mass, roughly").
  **Basis** is one of: lit (literature), EH (EditHunt in CONTEXT.md), FM (FeatureMatch diagnosis), WIP (the entity
  and template lists), or guess.

## 2. Behaviour validation (clean top-1 accuracy)

| Family | Overall validated | T0 / T1 / T2 / T3 / T4 (WIP template order) | Weakest groups | Entities passing >= 4/5 templates | Basis |
|---|---|---|---|---|---|
| city_state | **0.78** [0.60, 0.90] | 0.85 / 0.85 / 0.80 / 0.65 (Q/A) / 0.80 | Georgia, Washington, Ohio ~0.65 (Albany, Rome, Vancouver, Hamilton are ambiguous); Texas, California, Florida ~0.90 | 0.60 | EH (geography works on gemma-2-2b), WIP (ambiguous names), guess |
| city_capital (two-hop) | **0.42** [0.25, 0.60] | 0.45 / 0.40 / 0.45 / 0.35 / 0.30 | capital is not the largest city: Illinois, Pennsylvania, Washington ~0.20, NY and Michigan ~0.25; capital is the largest city: Georgia, Arizona, Massachusetts ~0.70; Texas, California ~0.75 | 0.25 | EH (two-hop is weaker in small models), lit (latent multi-hop is unreliable), guess |
| country_lang | **0.72** [0.55, 0.85] | 0.85 / 0.85 / 0.80 / 0.80 / 0.35 ("speak the language" -> " of") | English (Uganda, Zambia, Nigeria) ~0.55, French (Haiti, Congo) ~0.60; Spanish, German, Russian ~0.90 | 0.50 | guess |
| athlete_sport | **0.80** [0.65, 0.92] | 0.88 / 0.85 / 0.85 / 0.85 / 0.50 ("watching professional") | hockey and baseball ~0.70 | 0.65 | guess (famous names, strong priors) |
| langid | **0.82** [0.65, 0.92] | 0.90 (few-shot) / 0.80 / 0.85 / 0.85 / 0.75 | Portuguese ~0.60 (read as Spanish), Dutch ~0.75 | n/a (sentences) | guess |
| country_capital (10 templates) | **0.85** [0.75, 0.95] | "seat of government" ~0.70, the rest 0.80-0.95 | India (" New"), Turkey (Istanbul), Colombia | 0.80 | guess |

- **Pooled over all items: 0.70** [0.58, 0.82]. Median clean margin on validated items is 3-6 logits, and smaller
  (1-3) for city_capital. Basis: guess.
- **Style-varied prompts (S) validate 10-20 points lower** than the plain templates of the same family (point: -13).
  The largest drop is for city_capital (-20). Basis: FM (style moves internals; the base model is template-sensitive).
- **Groups with >= 20 validated held-out (S) target items**: about 30 of the 45 pairs. Most city_capital groups fall short.
  Basis: guess from the counts above and ~19 entities per group.

## 3. Best achievable R by layer and cap (median over the 45 group-level pairs; held-out S)

| Layer | k = 1 | k = 3 | k = 5 | k = 10 |
|---|---|---|---|---|
| 6 | 0.08 [0.00, 0.20] | 0.15 [0.03, 0.30] | 0.20 [0.05, 0.35] | 0.28 [0.10, 0.45] |
| 12 | 0.25 [0.10, 0.45] | 0.40 [0.20, 0.60] | 0.48 [0.30, 0.65] | 0.56 [0.35, 0.72] |
| 18 | 0.30 [0.10, 0.50] | 0.48 [0.25, 0.65] | 0.55 [0.35, 0.72] | 0.62 [0.40, 0.78] |

- **Components at k = 5**. L18: Effect 0.70, Preserve 0.85, KL factor 0.97. L12: 0.60 / 0.88 / 0.98. L6: 0.25 / 0.92 / 0.98.
  The reference's latents are group-specific, so the KL factor stays >= 0.9 for any kappa >= 0.1 nats. Predicted
  mean KL of reference sets: <= 0.01 nats at every layer. Basis: lit (Geva et al. 2023: the subject is enriched in
  early-to-mid MLPs and the attribute is extracted at the last token in upper layers; ROME causal tracing), EH (the
  state is a linear direction on the city token in mid layers, then handed off to the final token).
- **Why layer 6 is weak**: MLPs after layer 6 recompute the attribute from the entity's identity, and that identity
  survives the ablation through other latents and the error term. Basis: lit (Farrell et al. 2024, unlearning with
  gemma-2b SAE latents: zero-ablating a few latents had little effect, and negative clamping was needed; verify).
- **k matters, k = 1 is not enough**: the median gain R(k=5) - R(k=1) is about 0.2 at L12 and L18. Basis: guess
  (redundant features, e.g. "Texas", "southwest US", "Dallas sports teams").
- **By family** (best layer, k <= 5): country_capital 0.75 [0.55, 0.90]; athlete_sport 0.65 [0.45, 0.80];
  city_state 0.55 [0.35, 0.75]; country_lang 0.50 [0.30, 0.70] (the Arabic and Spanish groups are good, the English group is poor:
  "English-speaking" is not one concept); langid 0.42 [0.20, 0.70] (redundant evidence on every token); city_capital
  0.40 [0.20, 0.60] (sibling capitals have small margins, so Preserve suffers). Basis: guess.
- **Selection gap**: an oracle that picks the set on held-out data would beat the honest reference by about +0.07 R
  (median). Basis: guess.

### Share of cells that reach R >= 0.5 / 0.7 with k <= 5

| | R >= 0.5 | R >= 0.7 |
|---|---|---|
| Layer 6 cells | 0.10 [0.00, 0.25] | 0.02 [0.00, 0.10] |
| Layer 12 cells | 0.45 [0.25, 0.65] | 0.15 [0.05, 0.30] |
| Layer 18 cells | 0.55 [0.35, 0.75] | 0.20 [0.05, 0.35] |
| **All 135 group-level cells** | **0.37** [0.20, 0.55] | **0.12** [0.03, 0.25] |
| **(family, group) pairs at their best layer** | **0.60** [0.40, 0.80] | **0.25** [0.10, 0.40] |
| country_capital entities, best layer | 0.80 [0.60, 0.95] | 0.55 [0.35, 0.75] |

Basis: section 3 medians with a guessed spread. Families with >= 3 feasible groups: **4 of 5** [3, 5]; city_capital is
the likeliest to miss.

## 4. Baselines as a share of the reference's R (median ratio over feasible cells, k = 5)

| Baseline | Median R_b / R_ref | Share of feasible cells with ratio >= 0.5 | Basis |
|---|---|---|---|
| No latents (S empty) | **0.00** exactly (Effect = 0 by construction; the empty-set ablation is bit-identical) | 0 | definition |
| Random 5 of 16,384 latents | **0.01** [0.00, 0.05] | ~0 | L0 ~70 of 16k: a random latent fires on ~0.4% of tokens |
| Random 5 among latents active on the examples | 0.08 [0.00, 0.20] | 0.05 | guess |
| Most-active 5 (summed activation on the example target prompts) | **0.08** [0.00, 0.25] | 0.10 | these are template, position and common-token latents: Effect 0.2-0.5, but Preserve ~0.5 and KL 0.2-1.0 nats, so the KL factor is ~0 at kappa = 0.1 |
| Mean-diff cosine (top 5 latents by cos(W_dec[i], mean target - mean sibling residual) at the last token) | **0.35** [0.10, 0.70] | 0.30 [0.10, 0.55] | the direction is right, but cosine ignores whether the latent fires at all; better at L18 than at L12 |
| Plain steering vector (non-SAE comparator, not a valid submission): subtract alpha x (mean target - mean sibling) at every position, alpha tuned on the examples | **0.95** [0.60, 1.30] | 0.85 | EH (mean-diff flips 85-90% of held-out cities, with leakage to siblings); lit (AxBench: difference-in-means matches or beats SAE latents for steering) |
| Same vector, untuned (alpha = 1) / projected out as one direction | 0.60 / 0.65 | 0.60 / 0.65 | guess |
| **Naive top-5 attribution** (gradient x activation of the answer log-prob on the example targets only; no sibling contrast, no check) | **0.55** [0.30, 0.85] | **0.55** [0.30, 0.80] | log-prob attribution already favours latents that separate Texas from the competing states, and summing over several example cities favours shared latents over city-specific ones; it loses mainly by including generic "US state" or template latents (Preserve) and through linear-approximation error |
| Contrastive top-5 attribution (target minus sibling, no greedy, built only from public examples and self-built controls) | 0.80 [0.60, 1.00] | 0.90 | it is the reference's first stage |

- **Does naive top-k attribution reach >= 50% of reference R on most feasible cells?** Prediction: **yes, borderline,
  probability 0.55.** If so, the task as specified is close to a recipe. The plain steering vector being about as good as the SAE
  reference does not break the task, because only latent sets can be submitted. It does mean the SAE adds no
  editing power over a mean difference here (an AxBench-like finding).
- **Lever predicted to defeat naive attribution** (an ADJUST variant to test): grade preservation of the *same
  entity's other relation*. Example target: break city -> capital. Preserve: city -> state on the same cities, as in
  EditHunt's Hard tier and T2 hop-separation. Naive attribution picks the "Texas" latents, which break both relations.
  Predictions under this control: naive ratio 0.20 [0.05, 0.45]; R_ref 0.30 [0.15, 0.50]; feasible pairs 0.25.
  Basis: CONTEXT.md (T2: naive recipe 0/7, reference ~57% one-shot).

## 5. Error term, feature splitting, style

**Error term.** Prediction: it does **not** carry most of the behaviour.
- With the residual replaced by the SAE reconstruction (error dropped), clean top-1 survives on 0.90 [0.80, 0.97] of
  validated items at L6, 0.85 [0.70, 0.95] at L12 and 0.80 [0.65, 0.92] at L18. Mean KL on wikitext: 0.08 / 0.15 /
  0.20 nats. Basis: lit (Gemma Scope report: 16k residual SAEs near L0 70 cost ~0.1-0.25 nats of loss; verify).
- The error node gets 0.25 [0.10, 0.45] of total |attribution| to the answer log-prob (median over cells). Basis:
  lit (Marks et al. 2024: error nodes matter in circuits, but are rarely dominant on simple factual prompts).
- Ablating the top-50 contrastive latents (uncapped) gives Effect 0.85 at L12/L18 and 0.50 at L6.
- P(error term carries > 50% of the attribution) = 0.15 at L12/L18 and 0.20 at L6.

**Feature splitting.** Prediction: it **occurs but is not dominant** for the reference. It **is** a real failure mode for
naive picks.
- 25% [10, 45] of naive top-5 latents are entity-specific: they fire on <= 2 of the group's ~19 entities.
  The reference has 10% [0, 25].
- A latent set built only from entity-specific latents of the example cities gets held-out Effect <= 0.15.
- Reference sets keep 0.80 [0.65, 0.95] of their example-prompt Effect on held-out entities.
- A state-level latent ("Texas") fires at the city token on 0.60 [0.40, 0.85] of held-out group cities.
  Basis: lit (Bricken et al. 2023 and Templeton et al. 2024 on splitting; Chanin et al. 2024 on feature absorption).
- **Absorption works the other way too**: famous cities (Houston, Dallas) have their own latents, which can
  absorb the state latent's firing. So a state latent picked on small cities may stay silent on big ones. Prediction:
  held-out Effect differs between the most and least famous third of cities by <= 0.15 (no strong trend).
- Spearman correlation between entity fame and flip: 0.2 [-0.1, 0.4].
- For athlete_sport, athlete-identity latents (e.g. one for Michael Jordan) are more common than city latents:
  35% of naive picks there.

**Style.** Prediction: style-varied held-out prompts **do lower R**, moderately.
- R_ref(S) / R_ref(T) is 0.80 [0.60, 0.95] at the median, an absolute drop of 0.08 [0.02, 0.20].
- By layer, the ratio is 0.75 at L18 (last-token "answer" latents are tied to the template) and 0.85 at L12
  (entity-token latents carry over across styles).
- For naive attribution the ratio is 0.70, because it picks more template and format latents.
- Selected latents, by what they fire on: format or template cues are <= 10% of reference picks and ~25% of naive
  picks. "State/region-level concept" latents are ~60% of reference picks at L12/L18 for city_state, and "say X"
  output latents ~20%. Basis: FM lesson 2 (an anti-shortcut rule selected format-cue latents), guess.
- Check: read the top-activating contexts of every selected latent on a mixed-style corpus before trusting a design rule.

**Memorisation (FM lesson 3).** The best set for a (family, group, layer) is fixed, so a policy trained on repeated
groups can memorise latent IDs. Permuting IDs per instance removes ID memorisation but not the group-to-set map,
because the policy would still see the permuted IDs. Evaluation must use held-out groups or families. No numeric
prediction; the decision rule only requires that enough feasible groups exist for such a split.

## 6. Decision rule: GO / ADJUST / NO-GO

All quantities are point estimates on held-out (S) prompts.
- A cell is **measurable** if it has >= 20 validated held-out target items and >= 40 sibling items. Unmeasurable cells
  are counted and excluded.
- The **cheap recipes** are: most-active, random, mean-diff cosine and naive top-5 attribution.
- **p_pair** = the share of measurable group-level (family, group) pairs with a feasible layer (R_ref >= 0.5 at k <= 5).
- **p_cell** = the share of measurable group-level cells that are feasible.
- **q_b** = the share of feasible cells where cheap recipe b reaches R_b >= 0.5 x R_ref.

The checks are applied in this order, and the first one that fires decides:

1. **NO-GO (nothing small works).** Fires if p_pair < 0.50, or if p_cell < 0.25. It also fires if the median
   R_ref(k = 10) < 0.5 over pairs at their best layer, since then a larger cap would not help either.
   The rule uses pairs at their best layer rather than all cells. Layer is an instance parameter we choose, and a layer
   where nothing works can be dropped or used as an honest "cannot be done within the cap" instance. p_cell is
   reported as well.
2. **NO-GO-as-is (recipe).** Fires if q_b > 0.50 for any cheap recipe b. As in the FeatureMatch stop rule, a point
   estimate above 0.50 whose Wilson CI includes 0.50 still fires and is reported as "borderline". This verdict means
   the task, as specified, does not require understanding. The next step is the redesign in ADJUST (a), and the
   task does not ship in this form.
3. **ADJUST.** Fires if any of the following holds:
   - (a) a cheap recipe has 1/3 < q_b <= 1/2. The redesign adds same-entity other-relation preservation (section 4)
     or near-sibling groups, then re-measures.
   - (b) 0.50 <= p_pair < 0.60, or 0.25 <= p_cell < 0.35. The fix is to drop the weak layers and families and keep
     the rest.
   - (c) fewer than 3 families have >= 3 feasible groups, so held-out-group evaluation against memorisation is
     impossible. The fix is to add groups.
   - (d) the style test fails: R_ref(S) < 0.7 x R_ref(T) on more than 1/3 of feasible cells, or more than 20% of
     reference picks are format or template cues. The reference method or the example prompts must change first.
   - (e) a layer's SAE reconstruction keeps clean top-1 on < 0.75 of validated items. That layer is dropped.
   - (f) contrastive top-5 attribution alone reaches >= 0.8 x R_ref on more than 2/3 of feasible cells. The reference
     is then just a fixed two-step script, and difficulty must come from controls that the script cannot anticipate.
     This item only adds to another verdict and never overrides a NO-GO.
4. **GO.** Fires if none of the above holds. That requires all of: p_pair >= 0.60, p_cell >= 0.35, every cheap recipe
   at q_b <= 1/3, a median random/no-latent ratio <= 0.05, >= 3 families with >= 3 feasible groups, and passing style
   and reconstruction checks.

**Predicted verdict.**

| Verdict | Probability |
|---|---|
| NO-GO-as-is (recipe; driven by naive top-5 attribution) | 0.40 |
| ADJUST (most likely reasons: (a) or (b), plus (f), plus dropping layer 6) | 0.35 |
| GO | 0.15 |
| NO-GO (nothing small works) | 0.10 |

Basis: the sections above. FeatureMatch's recipes beat our predictions (SR predicted 0.75, observed 0.87), so I
shaded towards the recipe outcome.

**Peak RSS:** no job was run for this document. It was CPU-only reading and writing, with no model, SAE or array loaded.

## Freeze
Nothing above this line changes after the commit that adds this file. Append dated outcome sections below.
