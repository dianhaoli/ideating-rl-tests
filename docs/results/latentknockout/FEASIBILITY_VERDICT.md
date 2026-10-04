# LatentKnockout: feasibility verdict (synthesis)

Written 2026-10-02T09:40Z (date -u). CPU only. This document combines four inputs and checks their numbers:

- the pre-registered predictions and decision rule: `PREDICTIONS_FEASIBILITY.md` (commit c606abd4, frozen);
- the feasibility study: `FEASIBILITY.md` (commit daa96fa6);
- the precedent survey: `PRECEDENT.md` (commit cd98849f);
- two skeptic passes:
  - an independent reproduction (`skeptic_reproduce/`, commits 57d1554c and 0917284a);
  - a shortcut hunt against the recommended variant (`skeptic_shortcuts/`, commit d3a4257c).

I recomputed every number that decides the verdict from the sweep's stored per-prompt margins, using a new script
(`verdict_check.py`, output `runs/latentknockout/verdict/rule_recheck.json`). It does not import the study's or the
skeptics' metric code. All decision numbers below match the study and the reproduction to the third decimal.

**Terms.**

- **Latent.** One of the 16,384 directions of a Gemma Scope sparse autoencoder (SAE) at one layer of gemma-2-2b.
- **Knock out a set S at layer L.** At that layer, subtract each chosen latent's contribution from the model's
  internal state, keep everything else, and let the model finish its forward pass.
- **Cell.** A (family, group, layer) triple, e.g. (city → state, Texas, layer 18).
  - The target prompts are Texas cities; the siblings are cities of the other states.
- **R (reward).** R = Effect × Preserve × KL factor, measured on private held-out prompts.
  - **Effect** is the share of target prompts whose right answer stops being the top next token.
  - **Preserve** is the share of sibling prompts still answered correctly.
  - The **KL factor** penalises changes on ordinary text.
- **Feasible.** A cell is feasible if the reference search reaches R ≥ 0.5 with at most 5 latents.

## 0. The verdict in plain words

**NO-GO-as-is (recipe).** This is the second check of the pre-registered rule, and it fires on the pre-registered
metric. The skeptics' findings make the case stronger, and no reading of the evidence gives GO or ADJUST.

1. **Small knock-out sets exist.** For 30 of 43 (family, group) pairs, some layer has ≤ 5 latents whose removal
   breaks the answer on new entities in new styles while other groups keep theirs. Removing nothing or random latents
   scores exactly 0.
2. **A one-line ranking finds them about as well as the careful search does.**
   - Ranking latents by gradient × activation reaches half the reference's R on 31 of 44 feasible cells.
   - Ranking by decoder cosine with the mean difference does so on 24 of 44.
   - The pre-registered limit is half the cells (22 of 44), so both rankings exceed it.
3. **"Removed" was often only "pushed to second place".** On held-out targets the right answer leads the next token
   by a median of only 1.75 logits.
   - If the answer must lose by at least 0.5 logit, only 0.49 of pairs stay feasible with the study's reference, and the
     first check (NO-GO, nothing small works) would fire instead.
   - An oracle shows that real knock-out sets of 3-5 latents do exist, so this is partly a weak reference, not only a
     limit of the dictionary.
4. **The closest variant fails its own recipe gate by a wide margin.** "LatentKnockout-Verify" adds null slots and
   asks the agent to verify generalisation.
   - A short script ranks latents, tests them on prompts it writes about other members of the group, and answers
     "cannot" below a threshold. It passes 0.76 of episodes (CI 0.47-0.94), against a gate of ≤ 0.10.
   - Remembering which cells were null passes 0.90.
5. **Recommendation: do not build LatentKnockout, or the Verify variant, as Dan's one validated environment.**
   - Keep it as an explored idea with three findings that are useful for the pitch (section 3.1).
   - Section 3.2 gives the only redesign I would still test. It should be built only if three cheap kill gates pass
     (section 5), and I put about 0.10 on that.

## 1. The pre-registered decision rule, applied exactly

### 1.1 The rule (verbatim from `PREDICTIONS_FEASIBILITY.md` section 6)

> All quantities are point estimates on held-out (S) prompts.
> - A cell is **measurable** if it has >= 20 validated held-out target items and >= 40 sibling items. Unmeasurable cells
>   are counted and excluded.
> - The **cheap recipes** are: most-active, random, mean-diff cosine and naive top-5 attribution.
> - **p_pair** = the share of measurable group-level (family, group) pairs with a feasible layer (R_ref >= 0.5 at k <= 5).
> - **p_cell** = the share of measurable group-level cells that are feasible.
> - **q_b** = the share of feasible cells where cheap recipe b reaches R_b >= 0.5 x R_ref.
>
> The checks are applied in this order, and the first one that fires decides:
>
> 1. **NO-GO (nothing small works).** Fires if p_pair < 0.50, or if p_cell < 0.25. It also fires if the median
>    R_ref(k = 10) < 0.5 over pairs at their best layer, since then a larger cap would not help either.
>    The rule uses pairs at their best layer rather than all cells. Layer is an instance parameter we choose, and a layer
>    where nothing works can be dropped or used as an honest "cannot be done within the cap" instance. p_cell is
>    reported as well.
> 2. **NO-GO-as-is (recipe).** Fires if q_b > 0.50 for any cheap recipe b. As in the FeatureMatch stop rule, a point
>    estimate above 0.50 whose Wilson CI includes 0.50 still fires and is reported as "borderline". This verdict means
>    the task, as specified, does not require understanding. The next step is the redesign in ADJUST (a), and the
>    task does not ship in this form.
> 3. **ADJUST.** Fires if any of the following holds:
>    - (a) a cheap recipe has 1/3 < q_b <= 1/2. The redesign adds same-entity other-relation preservation (section 4)
>      or near-sibling groups, then re-measures.
>    - (b) 0.50 <= p_pair < 0.60, or 0.25 <= p_cell < 0.35. The fix is to drop the weak layers and families and keep
>      the rest.
>    - (c) fewer than 3 families have >= 3 feasible groups, so held-out-group evaluation against memorisation is
>      impossible. The fix is to add groups.
>    - (d) the style test fails: R_ref(S) < 0.7 x R_ref(T) on more than 1/3 of feasible cells, or more than 20% of
>      reference picks are format or template cues. The reference method or the example prompts must change first.
>    - (e) a layer's SAE reconstruction keeps clean top-1 on < 0.75 of validated items. That layer is dropped.
>    - (f) contrastive top-5 attribution alone reaches >= 0.8 x R_ref on more than 2/3 of feasible cells. The reference
>      is then just a fixed two-step script, and difficulty must come from controls that the script cannot anticipate.
>      This item only adds to another verdict and never overrides a NO-GO.
> 4. **GO.** Fires if none of the above holds. That requires all of: p_pair >= 0.60, p_cell >= 0.35, every cheap recipe
>    at q_b <= 1/3, a median random/no-latent ratio <= 0.05, >= 3 families with >= 3 feasible groups, and passing style
>    and reconstruction checks.

Effect is the pre-registered "hard flip": the answer is no longer the top next token.

### 1.2 The numbers that decide it

All 106 group-level cells are measurable (layers 6, 12 and 18; layer 6 was run on 4 groups per family). There are 43
(family, group) pairs. CIs are Wilson 95% intervals.

| Check | Quantity | Value | Threshold | Fires? |
|---|---|---|---|---|
| 1 NO-GO | p_pair | **0.698** (30/43) [0.55, 0.81] | < 0.50 | no |
| | p_cell | 0.415 (44/106) [0.33, 0.51] | < 0.25 | no |
| | median R_ref(k ≤ 10), pairs at best layer | 0.686 | < 0.5 | no |
| 2 recipe | q naive top-5 attribution | **0.705** (31/44) [0.56, 0.82] | > 0.50 | **yes** (CI excludes 0.5) |
| | q decoder cosine with the mean difference | **0.545** (24/44) [0.40, 0.68] | > 0.50 | **yes, borderline** (CI includes 0.5; fires as pre-registered) |
| | q most-active top-5 | 0.159 (7/44) | > 0.50 | no |
| | q random 5 / random 5 among active | 0.00 / 0.00 | > 0.50 | no |

The first check that fires is check 2: **NO-GO-as-is (recipe)**.

The ADJUST conditions are recorded for completeness; they cannot change the verdict.

- (b) No: p_pair is 0.70 and p_cell is 0.415.
- (c) No: 4 families have ≥ 3 feasible groups (city → capital 10, city → state 8, language id 7, country → language 3;
  athlete → sport 2).
- (d) No:
  - The median ratio of new-style to same-style R is 1.03, and 0 cells fall below 0.7.
  - Format-cue latents are ≤ 4% of the reference's picks.
- (e) No: the SAE reconstruction keeps the answer on 0.92 / 0.89 / 0.94 of prompts at layers 6 / 12 / 18.
- (f) **Applies as an addendum.** Contrastive top-5 attribution reaches ≥ 0.8 × R_ref on 0.682 of feasible cells
  (above 2/3). The reference's own first step, with no search, is about as good as the reference.

### 1.3 Do the skeptics' findings change the verdict? No. They strengthen it.

| Reading of the evidence | p_pair | p_cell | median R(k ≤ 10) | Check 1 | q naive / cosine | Verdict under the rule |
|---|---|---|---|---|---|---|
| Pre-registered: answer no longer top-1 | 0.70 | 0.42 | 0.69 | no | 0.70 / 0.55 | **NO-GO-as-is (recipe)** |
| Near-tie targets dropped (clean lead < 1.5 logits) | 0.56 | 0.32 | 0.58 | no | 0.68 / 0.53 | NO-GO-as-is (recipe) |
| Answer must lose by ≥ 0.5 logit | 0.49 | 0.26 | 0.51 | **yes** | 0.68 / 0.54 | NO-GO |
| Answer must lose by ≥ 1 logit | 0.33 | 0.15 | 0.30 | yes | 0.63 / 0.50 | NO-GO |
| Answer must lose by ≥ 2 logits | 0.05 | 0.02 | 0.11 | yes | (n = 2) | NO-GO |

Source: `verdict_check.py` on the stored margins; it matches the reproduction's table.

**Caveat on the strict rows.** The reference's sets were chosen for the lenient objective, so these rows understate
what a search built for the strict objective would reach. The skeptic's strict oracle chose sets on the grader's own
items and was then scored on fresh prompts it never saw:

| Cell | Strict oracle | Reported reference set |
|---|---|---|
| city → state, Arizona, L18 | 0.72 | 0.47 |
| city → capital, Illinois, L12 | 0.73 | 0.40 |
| city → capital, Texas, L18 | 0.77 | 0.32 |

So "nothing small works" is not established under a strict effect. What is established:

- the study's feasibility numbers are inflated by near-ties;
- the recipe problem survives the strict effect. Inside the 16 strict-feasible cells, q is 0.50 (cosine), 0.63 (naive)
  and 0.88 (contrastive). Decoder cosine scores 0.71 and 0.53 on fresh prompts where the oracle scores 0.72 and 0.77.

Other findings that bear on the verdict:

- **Labels are less stable than reported.** Resampling whole entities widens the CIs (median width 0.23 against 0.19).
  - 23 of the 86 cells at layers 12 and 18 have an entity-level CI that contains 0.5.
  - One cell changes label with the example split: language id, Turkish, L18 scores 0.40 / 0.94 / 0.90 on three splits
    and was reported as 0.35.
  - None of this moves q_b below 0.5.
- **The Verify variant (FEASIBILITY.md §14) fails its own recipe gate.** Episodes have 3-5 slots and 30-50% nulls, and
  an episode passes when every slot is right. The gate is ≤ 0.10.

  | Scripted policy | Episode pass [CI] |
  |---|---|
  | Rank latents three standard ways, test the sets on self-written prompts about other group members, threshold | **0.76** [0.47, 0.94] |
  | The same, with greedy prefixes among the candidates | 0.90 [0.63, 1.00] |
  | The same, limited to 60 forward passes per slot | 0.71 [0.46, 0.94] |
  | Self-test only on the shown entities in new wordings (no overlap with the grader's entities) | 0.56 [0.30, 0.81] |
  | Remember which cells are null (latent IDs permuted), plus a fixed layer rule | 0.90 [0.77, 1.00] |
  | Example-objective verifier (the script the study itself flagged) | 0.52 |

## 2. Predictions vs outcomes

Predictions are from `PREDICTIONS_FEASIBILITY.md`; outcomes are from `FEASIBILITY.md` and the skeptic passes.
"Hit" means the observation fell inside the stated range, or on the stated side when no range was given.

| # | Prediction (predicted → observed) | Result |
|---|---|---|
| 1 | Clean accuracy, all prompts: 0.70 → 0.75 | hit |
| 2 | City → state accuracy 0.78 → 0.86; athlete 0.80 → 0.86; country → capital 0.85 → 0.85; country → language 0.72 → 0.85 | hit (all inside ranges) |
| 3 | City → state capital (two-hop) accuracy 0.42 [0.25, 0.60] → 0.70 | miss (too pessimistic) |
| 4 | Language id accuracy 0.82 → 0.56 (copies the few-shot demo, opens quotes) | miss |
| 5 | New styles answered 10-20 points less often → 4.6 points less | miss (smaller) |
| 6 | Median clean margin 3-6 logits → 1.75 on held-out new-style targets (25% under 1) | **miss; this is the root of the near-tie problem** |
| 7 | Measurable pairs about 30 of 45 → all 43 pairs measurable | miss (more) |
| 8 | Median R_ref at k = 1/3/5/10 by layer, e.g. L18 0.30/0.48/0.55/0.62 → 0.35/0.51/0.52/0.52; L12 and L6 also inside ranges | hit |
| 9 | Reference set components at L18: Effect 0.70, Preserve 0.85 → 0.59, 0.99 | partial (Effect limits R, not Preserve) |
| 10 | Mean KL of reference sets ≤ 0.01 nats → median 0.0007 | hit |
| 11 | Median per-cell gain R(k=5) − R(k=1) about 0.2 → 0.06 (L12), 0.11 (L18); R saturates at k = 3 | miss (smaller; k is a weak dial) |
| 12 | Share of cells with R ≥ 0.5: L6 0.10, L12 0.45, L18 0.55, all 0.37, pairs 0.60 → 0.00, 0.47, 0.56, 0.42, 0.70 | hit (all inside ranges) |
| 13 | Share with R ≥ 0.7: all cells 0.12 [0.03, 0.25], pairs 0.25 [0.10, 0.40] → 0.26, 0.49 | miss (better) |
| 14 | Families with ≥ 3 feasible groups: 4 of 5 → 4 | hit |
| 15 | City → capital likeliest to fall short → it has the most feasible groups (10 of 12) | miss |
| 16 | Single-country capital cells feasible on 0.80 → 0.08 (reference), 0.33 (cosine) | miss (far too optimistic) |
| 17 | No latents 0, random 0.01, random among active 0.08 → 0, 0, 0 | hit |
| 18 | Most-active ratio 0.08, q 0.10 → 0.05, 0.16 | hit |
| 19 | Decoder cosine ratio 0.35, q 0.30 → 0.79, 0.55 (1.00 at L18, 0.00 at L12) | miss (far too low) |
| 20 | Naive top-5 attribution ratio 0.55, q 0.55 → 0.83, 0.70 | hit (inside ranges) |
| 21 | Naive attribution is a recipe, "yes, borderline", p = 0.55 → yes, not borderline | hit on direction |
| 22 | Contrastive top-5 ratio 0.80, q 0.90 → 0.87, 0.89 | hit |
| 23 | Tuned steering vector about 0.95 × reference → 0.18 (KL cost; nothing at L12) | miss (far too high) |
| 24 | Untuned / projected-out vector 0.60 / 0.65 → 0.09 / 0.06 | miss |
| 25 | Keep-state variant: reference R 0.30, feasible pairs 0.25, naive ratio 0.20 → 0.32-0.41, 3 of 12, naive q 0 at L18 | hit |
| 26 | SAE reconstruction keeps the answer on 0.90 / 0.85 / 0.80 at L6/L12/L18 → 0.92 / 0.89 / 0.94 | hit at L6 and L12; just above range at L18 |
| 27 | Error term carries > 50% of the behaviour is unlikely (0.15-0.20) → not dominant (city → capital at L12 0.60 is the exception) | hit |
| 28 | Error-node share of attribution 0.25 → too noisy to measure | untested |
| 29 | Top-50 contrastive Effect 0.50 / 0.85 / 0.85 → 0.35 / 0.71 / 0.86 | partial (L18 hit) |
| 30 | Entity-specific picks: naive 25%, reference 10% → 2%, 3% | miss (fewer; the count is generous) |
| 31 | Reference keeps 0.80 of its example Effect on held-out entities → 0.80 | hit |
| 32 | Weak fame effect (≤ 0.15) → famous groups are the failures; Texas is null on famous held-out cities but 0.56-0.65 on less famous towns | miss (strong, opposite) |
| 33 | Athlete-specific naive picks about 35% → not measured separately | untested |
| 34 | New-style R / same-style R = 0.80 (L18 0.75, L12 0.85; naive 0.70) → 1.03 (1.06, 0.99; naive 1.08) | miss: no style penalty on the study's styles |
| 35 | The same, on wordings nobody tuned on (skeptic's fresh templates) → 4 of 6 cells hold; French 0.88 → 0.63, Turkish 0.32 → 0.17 | partial (some penalty) |
| 36 | Format-cue picks: reference ≤ 10%, naive about 25% → ≤ 4%, about 10% | hit for the reference; miss for naive (fewer) |
| 37 | An oracle beats the honest reference by about +0.07 R → lenient: the best of {reference, cosine, naive, contrastive}, chosen on held-out data, beats the reference by a median of 0.00 (L12) and 0.05 (L18), means 0.02 and 0.11; strict oracle: +0.25 to +0.45 on 3 cells | hit on the lenient metric; miss on the strict one |
| 38 | Verdict probabilities: NO-GO-as-is 0.40 (modal), ADJUST 0.35, GO 0.15, NO-GO 0.10 → NO-GO-as-is | hit (modal outcome) |

Precedent-based expectations (`PRECEDENT.md` section 4):

- "The dense steering vector may beat the SAE reference": miss. It does not, though the stronger dense comparators
  (DiffMean directional ablation at the right positions, LEACE) were never run.
- "Zero ablation may be ineffective (Farrell et al.)": miss at layers 12 and 18, where it works. Hit at layer 6, where
  even 50 latents remove only 0.35 of the behaviour.
- "The error term is the riskiest prediction": it was not a problem.
- "Naive attribution is the field's default and likely a recipe": hit.

Overall, the decision-relevant predictions were right about where small sets exist and wrong about how good the cheap
rankings are (cosine) and how close the clean answers are to ties. Both errors push towards the recipe verdict.

## 3. Recommended design

### 3.1 Recommendation

**Do not build LatentKnockout, the keep-state variant or LatentKnockout-Verify as Dan's one validated environment.**
The evidence for this is consistent across all four inputs:

- **The concept is named, so the agent can measure success itself.** The task says which concept to remove (e.g.
  "Texas cities → Texas"). The grader's measurement (ablate, then rerun on other members of the group) is exactly what
  the agent can do itself.
- **Standard rankings already find the right latent.** They contain the reference's first latent in 86-98% of cells.
- **Every hardening tried so far fails.** Requiring "keep the same city's other fact" makes 9 of 12 groups impossible
  (keep-state variant). Asking for verification is beaten by a short self-test script (Verify variant).
- **The layer is a one-bit rule.** Cosine at layer 18 and contrastive attribution at layer 12 get 0.955 of feasible
  slots right.
- **The cap and kappa are weak dials.** R saturates at k = 3, and the reference's KL is about 0.

**What to keep for Dan's pitch.** It is an explored idea with real findings:

- **Grading by rerunning the model removes FeatureMatch's style problem.** New-style held-out prompts are not harder:
  the median ratio is 1.03 on the study's styles, with some penalty on fully fresh wordings.
- **"Flipped" is not "removed".** A top-1 flip metric counts near-ties: 25% of held-out answers lead by less than 1
  logit, and 45-86% of flipped answers stay in the top 3. Any knock-out benchmark needs a margin.
- **A research question Dan could own.** Famous concepts (Texas, California, Florida, New York; soccer, basketball,
  tennis, baseball; English) cannot be knocked out with ≤ 5 latents of a 16k SAE, while less famous ones can. Even
  within Texas, famous cities resist and small towns do not. Is that feature splitting by frequency, and can an agent
  predict knock-out-ability from probing?
- **Reusable parts.** `lk_core.py` gives bit-exact ablation with the error term kept, and is 2-4x faster through the
  cached residual. The style-tagged families have per-template validation.

**Pivot (not designed here).** The shortcut hunt points to the property a surviving design needs: the agent must not
be able to reproduce the grader's measurement cheaply. Two directions follow from that:

- the concept or trigger distribution is unknown to the agent, e.g. a planted, harmless fact edit whose scope the
  agent must discover;
- instances where attribution misleads, e.g. redundant "backup" latents, where each latent alone has no effect.

### 3.2 If LatentKnockout is pursued anyway: "LatentKnockout-Strict" (conditional, not validated)

This is the design that survives the skeptics' findings as far as the evidence goes. **Build it only if gates G1-G3 in
section 5 pass.** Each element is followed by the evidence for it.

| Element | Choice | Why |
|---|---|---|
| Subject | gemma-2-2b (bf16); Gemma Scope 16k residual SAEs at **layers 12 and 18 only** | Layer 6: 0 of 20 cells feasible. An all-null layer would be a fingerprint |
| Families | city → state, city → state capital and language id (enough feasible groups). Country → language only as an evaluation family. Athlete → sport dropped: 80% null, so its family alone predicts "cannot". **Add ≥ 3 procedurally scalable families**: city → country (about 40 country groups), US city → state for every state with ≥ 10 validated cities, and sentence → language for about 20 languages. Every template and entity is validated on the clean model | 42-43 pairs is too few: a policy can memorise the null cells (0.90) |
| Groups | Hundreds, generated procedurally. **Hold groups out jointly across sister families** (e.g. Arizona out of both city families; Spanish out of both language families) | Reusing the sister family's set passes 0.74 of feasible cells |
| Instance (slot) | (family, group, layer). The agent sees the concept in words, 3-4 example entities × 3 templates, and the layer | As in the study |
| Answer per slot | A set of ≤ 5 latent IDs, or "cannot be done with 5 latents". IDs are permuted per episode | k = 5: R saturates at k = 3 on the lenient metric, and the strict oracle used 3-5. Permutation stops ID memorisation only, not re-finding the concept by probing (which is legitimate) |
| Effect (**changed**) | A held-out target counts as knocked out only if its right answer now **trails the new top-1 by ≥ δ = 1 logit**. Targets and siblings whose clean lead is < 1 logit are dropped from the grading set | Median clean lead is 1.75; 25% of targets lead by < 1; 45-86% of lenient flips stay in the top 3. δ is EXPERIMENTAL (no precedent) |
| Reward | R = Effect_δ × Preserve × (1 − min(1, KL/kappa)), a product, so "break everything" and no-ops score about 0 | Plan v2 product rule; RAVEL and TPP aggregates let "break everything" score above 0 |
| kappa | 0.1 nats on 24 private wikitext paragraphs | A safety term: the verdict is unchanged for kappa 0.02-0.2, and reference sets have KL ≈ 0.0007 |
| tau | 0.5 provisional. Recalibrate after G1 so that there is a dead band of ±0.1 around tau with no kept cell in it | The lenient labels were fragile: 23 of 86 entity-level CIs contain 0.5 |
| Held-out grading data | ≥ 20 targets (new entities; 8 new-style + 2 same-style templates, including **templates nobody tuned on**) and ≥ 40 siblings per cell, private. The target population is defined precisely in the statement (e.g. "validated cities of the group from a fixed public list, including famous ones") | Null labels depend on which entities are held out (Texas: 0.17 on famous cities, 0.56-0.65 on small towns) |
| Null slots | **Null** = a strong search fails. It runs on 3 example splits, uses beam-3 over a ≥ 100-latent pool from 3 rankings and the strict objective, and is backed by a strict oracle on the grader's own items. Both must stay < tau with an entity-clustered upper CI < tau, on ≥ 2 held-out resamples. **Feasible** = the tool-only reference reaches ≥ tau + 0.1 on ≥ 2 of 3 splits with a lower CI ≥ tau. Everything in between is dropped. The null rate is balanced within family × layer × fame tercile (fame = how often the group name appears in wikitext) | A fame/family prior alone gets 0.59 of null slots right; labels changed with the split (Turkish 0.40/0.94/0.90) |
| Episode | 3-5 slots (count drawn), 30-50% nulls (drawn). Pass = every slot right; the continuous score is the mean slot score | DECISIONS D7 |
| Dials | Null fraction and slot count (D7). Sibling specificity (keep/locality readouts; RAVEL isolation, SAEBench TPP). Number of example entities (EXPERIMENTAL). Effect margin δ (EXPERIMENTAL). **Not dials**: k (saturates), kappa (not binding), same-entity other relation (makes 9 of 12 groups impossible), layer (a one-bit rule; it is an instance parameter, never a ceiling) | Feasibility §6-10; CONTEXT audit lessons |
| RL evaluation split | Train on city → state, language id and 2 new families. Evaluate on held-out groups (removed jointly from sister families), one whole new family (e.g. city → country), and city → capital plus country → language with their groups removed from training. Optionally evaluate with a second Gemma Scope variant at the same layer (a different L0) | FeatureMatch lesson 3; shortcut hunt finding 4 |
| Agent tools (generic primitives, capped) | `run(prompts)`: top-k next tokens and logits. `sae_encode(prompts, layer)`: active latents per token (permuted IDs). `ablate_run(prompts, layer, latents)`: the same outputs with latents knocked out (subtraction form, error kept, fixed batching rule). `top_examples(layer, latent)`: top-activating contexts on a fixed public mixed-style corpus, never the grader's. `token_effect(layer, latent)`: tokens the decoder direction promotes or suppresses. `attribution(prompts, answer, layer)`: gradient × activation per latent, counted against `gradient` units. `submit`. Caps on forward passes, gradient units and tool calls. No tool names the fix or a layer | Builder rule 3. Attribution is kept because removing it does not help (encode-only activation contrast already passes 0.80); the gates must pass with the strongest generic tools present |
| Reference solver (probing only, through the tools) | Write own prompts: examples, ≥ 10 other group members, ≥ 3 own templates, matched other-group controls and own neutral text. Pool = top-12 each of contrastive attribution, activation contrast and logit-lens-on-active. Beam-3 greedy on the strict self-made objective up to k = 5. Evaluate on a disjoint self-made validation split; submit if the lower CI ≥ tau, else "cannot". Must reach ≥ 95% of kept slots (best-of-5; also log the one-shot rate) | Builder gates; the strict oracle shows such sets exist on 3 cells |
| Baselines that must stay ≤ 10% of episodes (upper CI ≤ 0.2) | Constant "cannot"; always submit each standard ranking (contrastive, naive, cosine, activation contrast, logit lens, top-5); per-family/layer majority; text-statistics prior; null-label memoriser on the evaluation split; example-objective verifier; **the self-test script** (standard rankings top-1..5, verified on self-written other-member prompts, threshold), in its full, 60-pass and shown-entities-only versions; black-box control (no SAE tools). Per slot, no-op, random and most-active sets must score R ≈ 0 | The self-test script is the binding baseline (0.56-0.90 now) |
| Comparators (reported, not submittable) | DiffMean directional ablation at the entity and answer positions; LEACE-style projection; the tuned steering vector; the reference run over a random-direction dictionary with matched norms and sparsity; bounded negative clamping (c ∈ {1, 2, 4}) | PRECEDENT §5 items 1, 2 and 7: none was run |
| Expected pass band, frontier (Opus-class) agent | **The Verify design as it stands: 60-90%, too easy.** **Strict, if G2 passes: 30-60% (a guess, unmeasured).** | Reasons below the table |

Reasons for the expected pass bands:

- **Verify as it stands.** An Opus-class agent's natural workflow is exactly the self-test script: attribution, ablate,
  check on other Texas cities, decide. That script already passes 0.56-0.90, and the agent should do at least as well.
  It would fail mainly by mis-setting the "cannot" threshold, running out of budget or picking the wrong layer recipe.
- **Strict, if G2 passes.** Strict knock-outs need a 3-5 latent combination found by search, not a top-k list, and the
  agent must judge famous or split concepts correctly. Per-instance outcomes may be bimodal: some cells are always
  solved and nulls are always right. That needs checking before any GRPO use.

### 3.3 What changed from FEASIBILITY.md §14, and why

| Change | Driven by |
|---|---|
| A strict effect (≥ 1 logit), near-tie items dropped, and feasible and null labels recomputed with a strict-objective reference | Reproduction F1 |
| Several example splits, entity-clustered CIs, held-out resamples, and a dead band around tau | Reproduction F2/F3 |
| Templates nobody tuned on in the held-out set; a precisely defined target population | Reproduction F3/F4 |
| The self-test script replaces the example-objective script as the binding recipe gate | Shortcut hunt 1 and 5 |
| Procedural groups, joint sister-family holdout, a null balance by fame, and athlete → sport dropped | Shortcut hunt 2 and 4; fame prior |
| The dense, random-dictionary and clamping controls became mandatory before building | PRECEDENT §5 items 1, 2 and 7 |
| ID permutation kept, but not counted as a memorisation defence | Shortcut hunt 4 |

## 4. Top 5 risks and the test that would expose each

| # | Risk | Test that exposes it | Fails if |
|---|---|---|---|
| 1 | **A script solves it.** Standard rankings plus self-testing on self-written prompts plus a threshold pass most episodes, so the task measures bookkeeping, not understanding (lesson 4) | Rerun `skeptic_shortcuts/hunt.py` and `analyze_hunt.py` policies (selftest_rank_A/B/AB, selftest_all, budget-60, exJ, prior and memo) under the strict effect on the new stable labels, with episodes drawn as in production | Any policy passes > 10% of episodes, or its upper CI exceeds 0.2 |
| 2 | **The label is a near-tie or sample artefact.** Feasible and null labels flip with the margin rule, the example split or which entities are held out | Strict-effect sweep: 3 example splits × 2-3 held-out resamples per cell, entity-clustered bootstrap, strict oracle on the grader's items | < 70% of cells keep the same label across splits and resamples, or p_pair under the strict effect < 0.5 |
| 3 | **Nulls are predictable or memorisable.** Fame, family, sister family or the answer token give away "cannot", or a policy remembers null cells | Logistic prior on text statistics (group-name frequency, family, layer, answer token id) plus a label memoriser scored on the held-out-group and held-out-family split | Either beats chance on null slots by > 0.15, or passes > 10% of episodes |
| 4 | **The SAE is not doing the work.** A random-direction dictionary, a dense direction or clamping does as well, so the task measures search effort, not SAE understanding | Reference search over a random dictionary with matched decoder norms and sparsity. DiffMean directional ablation and LEACE at the same positions. Clamping at c ∈ {1, 2, 4} | The random dictionary reaches tau on ≥ 1/3 of feasible cells. If only clamping works, the answer format must change |
| 5 | **Difficulty is out of band or bimodal for real agents**, even if the scripted gates pass | Fresh Claude Code subagents (Opus-class and a small model) on the smoke plan, then ≥ 20 episodes. Per-instance pass variance across 4 rollouts. Transcripts read and failures labelled "interpretability mistake" vs "environment problem" | The frontier pass rate is outside 10-70%, more than half of instances are always-pass or always-fail, or most failures are environment problems |

Lesser risks:

- Grader nondeterminism. bf16 formulas can differ by up to 1.37 logits, so fix one formula and one padding/batching
  rule, and test that "ablate nothing" is bit-identical.
- Agents know the Dallas → Austin story. Keep those prompts out of the examples.
- Disk is at 86%. Never cache activations.

## 5. Next steps

**Step 0 (CPU, before any new GPU job).**

- Write `PREDICTIONS_STRICT.md`: predictions for G1-G3 and a kill rule, committed before any result.
- The kill rule is: drop LatentKnockout for good if any gate fails once for a reason the design cannot fix. If one fails
  twice for the same reason, mark the task DROP (builder guide).

**Gates before building.** Each job goes through `common.gpuq` at ≤ 10 GB (light), one job at a time, under
`systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=2G`. Report peak RSS.

| Gate | What | Estimated GPU | Kill criterion |
|---|---|---|---|
| G1 strict feasibility | Extend `sweep.py`: strict objective (δ = 1, near-ties dropped), beam-3 over a 3-ranking pool, 3 example splits, a strict oracle on the grader's items, entity-clustered CIs and 2 held-out resamples. Run on the 86 L12/L18 cells, plus 2 new families × about 10 groups as a first scale-up | about 2 h | p_pair_strict < 0.5, or < 3 families with ≥ 3 stable feasible groups, or < 70% label stability |
| G2 recipe gate | Rerun the hunt policies (risk 1) and the prior and memo policies (risk 3) on G1's stable labels, with the strict effect | about 1 h | Any scripted policy > 10% of episodes |
| G3 SAE controls | Random-direction dictionary, DiffMean directional ablation, LEACE and clamping (risk 4) on 20 feasible and 10 null cells | about 1 h | The random dictionary reaches tau on ≥ 1/3 of feasible cells |

My probabilities: G1 passes about 0.6; G2 passes about 0.2 given G1; G3 passes about 0.7 given both. **Overall about 0.10.**

**Build, only if G1-G3 pass** (builder guide "Required files"):

1. **`generate.py`.** Procedural groups and families with per-template validation, near-tie filtering and few-shot copy
   checks. Private held-out sampling, joint sister-family holdout, per-episode latent permutation, and slot and null
   sampling (3-5 slots, 30-50% null, balanced by fame). Canary in each private file. A `MANIFEST.md` for regenerable caches.
2. **`tools.py`.** The primitives in §3.2, with caps (forward passes, gradient units, tool calls). One fixed ablation
   formula and batching rule shared with the grader.
3. **`grader.py`.** Out-of-process; reads `instance.json` only; strict Effect × Preserve × KL factor on private items.
   Bit-exact no-op test and a determinism test as unit tests. `validate_submission` must work without `load()`.
4. **`reference_solver.py`** (tools only, never metadata), **`blackbox_control.py`** and **`recipe_baseline.py`**. The
   recipe baseline holds the constant, majority, submit-everything and always-submit variants plus every policy in
   risk 1 and risk 3.
5. **Documents.** `agent_prompt.md` (no codename, method or layer hints; the explicit "cannot" answer), `SPEC.md`,
   `PREDICTIONS.md` (frozen before any agent run), `run_agent.py` and `smoke_plan.json` (3 instances per tier, ≥ 1 null
   each, all passed by the reference).
6. **Harness gates** via `python -m common.sandbox run-scripted`: reference ≥ 95% (best-of-5; log one-shot), black-box
   ≤ 10%, every recipe variant ≤ 10%. Record them in NOTES.md and `runs/latentknockout/<ts>_gates/`.
7. **Smoke** with fresh Claude Code subagents that see only `agent_prompt.md` and the tool docs, in the sandbox. Per
   DESIGN_BRIEF, no hosted LLM API.
   - Read every transcript and label each failure as an interpretability mistake or an environment problem.
   - n = 3 per tier is a bug detector only. Then run ≥ 20 episodes for a pass-band estimate and per-instance variance.
8. **Write-up for Dan.** Task design and why, baseline and reference results, predictions vs outcomes, example
   transcripts, ways it could be gamed, an honest verdict.

**If any gate fails:** record LatentKnockout as a documented dead end, with `FEASIBILITY.md`, the two skeptic passes
and this file as the explanation. Spend the remaining days before the 8 October call on a lane that is closer to a
validated environment.

## 6. Files, compute, memory

- **This synthesis.** `tasks/latentknockout/FEASIBILITY_VERDICT.md` and `tasks/latentknockout/verdict_check.py`
  (recomputes the decision rule from the stored margins). Output: `runs/latentknockout/verdict/rule_recheck.json`.
- **Compute.** CPU only, no GPU job. Peak RSS of my heaviest (and only) job was **65 MB** (`verdict_check.py`, 0.6 s,
  under `MemoryMax=2G`).
- **Inputs read:**
  - `DESIGN_BRIEF.md`, `PREDICTIONS_FEASIBILITY.md`, `FEASIBILITY.md`, `PRECEDENT.md` (sections 4-5), `NOTES.md`;
  - `runs/latentknockout/20261002T0642_sweep/cells/*`;
  - `runs/latentknockout/skeptic/orig_robustness.json` and `analysis_repro.md` (headers);
  - `runs/latentknockout/skeptic_shortcuts/20261002T0813_hunt/tables.md` and `fame.json`;
  - `CONTEXT.md`, `docs/BUILDER_GUIDE.md`.
