# LatentKnockout lab notebook

## 2026-10-02T01:25Z Feasibility study starts
Goal (plain language): gemma-2-2b answers simple questions ("Dallas is a city in the state of" -> " Texas").
A sparse autoencoder (SAE) rewrites the model's internal state at one layer as a sum of ~16k "latents" (features),
only ~70 of which are non-zero on any token. The question for this study: can we find a SMALL set of latents such that
switching them off (subtracting their contribution, keeping everything the SAE cannot explain = the "error term")
breaks the answer for the target group (e.g. every Texas city, including cities and phrasings never used to pick the
latents) while leaving other groups (California cities, etc.) and ordinary text alone?
Setup notes:
- Disk is 99% full (1.9 GB free), so I cannot make my own bf16 copy of gemma-2-2b (the hub copy is fp32, 9.8 GB, and
  its memory map exceeds the 8 GB per-job RAM cap). I load the bf16 copy that the FeatureMatch builder already made at
  ~/wt/featurematch/tasks/featurematch/cache/gemma-2-2b-bf16 READ-ONLY (no edits there), with the hub as fallback.

## 2026-10-02T06:17Z Feasibility predictions committed before any result
I wrote my guesses down before running anything, so the feasibility study can be checked against them
(`PREDICTIONS_FEASIBILITY.md`; frozen once committed). The validation job from 01:28Z never got onto the GPU: its log has
only the queue-wait line, and there are no output files. So no model output had been seen. Headline guesses:
- About 70% of prompts are answered correctly by the clean model. The two-hop city -> state capital family is weakest (~42%).
- With at most 5 latents, a sensible search reaches a reward R >= 0.5 on ~37% of (family, group, layer) cells, and for
  ~60% of (family, group) pairs at some layer. Layer 6 is mostly hopeless (the model recomputes the fact after the ablation); layers 12 and 18 work best.
- The real risk is a recipe. Naive "top-5 latents by attribution" is predicted to reach >= 50% of the reference's
  reward on ~55% of feasible cells, which fires the NO-GO-as-is rule (probability ~0.4). Random, no-latent and
  most-active latents are predicted to score ~0. A plain steering vector (not submittable) is about as good as the SAE reference.
- Style-varied held-out prompts are predicted to lower R by about 20% (relative). The SAE error term is not predicted to carry most of the behaviour.
- The decision rule (GO / ADJUST / NO-GO, with numbers) is in section 6 of that file. No job was run for this step (no RSS to report).

## 2026-10-02T06:30Z Precedent survey written (PRECEDENT.md), no GPU, no model run
What I did: I read the published work on switching off or steering with SAE latents, compared with simple baselines,
and how such tasks are scored, and I wrote what each finding predicts for this task (`PRECEDENT.md`, with URLs; preprint numbers are tagged).
Main lessons in plain words:
- In almost every head-to-head test, SAE latents lose to a simple "difference of means" direction or to a supervised
  method: AxBench steering 0.239 (difference of means) vs 0.165 (SAE); RAVEL disentangle 48.6 (SAE) vs 60.1 (MDAS); in MIB,
  SAE features are no better than plain neurons. So restricting the agent to SAE latents is a handicap. The task must say
  this, and must measure it with a dense comparator that cannot be submitted.
- The usual way to choose latents (rank by attribution or probe weight, take the top 20: SAEBench SCR/TPP, Marks et al.)
  is exactly the recipe this task must not reward. It becomes a gate.
- Simply zeroing a few latents often barely changes the model; unlearning work needed negative clamping (Farrell et al.).
  Feasibility should test both.
- Absorption (a "Texas" latent that stays silent on some Texas cities) shows up in every SAE tested. Held-out-entity
  grading measures exactly this, and that is a real interpretability skill.
- Agents using SAEs do worst on the causal step (SAEScientist-Bench: 31 vs 58 for experts) and fall for formatting or
  substring latents. Mid-band difficulty is plausible if the recipe gate holds.
The 8 design implications are in section 5 of PRECEDENT.md. Section 4 compares the literature with the frozen feasibility predictions; the riskiest one
is "the SAE error term is not dominant". CPU only; peak RSS is negligible (web reading, no heavy job).

## 2026-10-02T06:31Z Feasibility study restarted (this agent); design of the measurement
- Reused the WIP entity lists; rewrote the prompt families (lk_data.py) so that every family has 13 templates in
  6 STYLES: plain completion (P), question/answer (Q), dialogue/chat (D), key-value record (K), news sentence (N) and
  few-shot list (F; demo answers are never an answer of any group, so a "copy the demo" artefact is detectable).
  The example prompts an agent would see use only P and Q styles (3 templates). Held-out prompts are of two kinds:
  "T" = new entities and new templates in the SAME styles, "S" = new entities in the 4 NEW styles. Comparing R on T
  and S measures whether a latent set found on one style still works on others (FeatureMatch lesson 1).
- lk_core.py rewritten: the layer-L residual of each prompt is cached once and only blocks L+1..25 are rerun for each
  candidate ablation (2-4x faster at layers 12/18). validate.py checks that this shortcut reproduces the full forward
  pass, that "ablate nothing" is bit-identical to the clean model, and that SAE reconstruction + error term
  reproduces the clean logits.
- First two validation launches died of CUDA OOM inside the 9 GB cap (all-position logits over a 256k vocabulary
  are ~0.3 GB per tensor; I kept too many alive). Fixed by comparing and freeing each tensor at once, and by applying
  the logit soft-cap in place.
- The GPU is busy (two latentdiff jobs hold 13 of 21 GB), so the job waits in the queue.

## 2026-10-02T06:39Z Exactness, behaviour validation, SAE reconstruction (runs/latentknockout/20261002T0640_validate, ..._0636_validate2)
**Exactness of the ablation machinery** (6 probe prompts incl. 2 wikitext snippets, all positions, all 256k logits):
- "Ablate nothing" through the hook: max |change in any logit| = 0.0 at layers 6, 12 and 18 (bit-identical).
- Running only blocks L+1.. from the cached layer-L residual: max |change| = 0.0 (bit-identical), so the fast path is safe.
- SAE reconstruction + error term written out explicitly: 0.0 (bit-identical after the bf16 cast).
- Two mathematically equal ways of ablating 3 latents (subtract f_i W_dec[i] vs rebuild x_hat with f_i = 0 and add the
  error) differ by bf16 rounding: max 0.67 / 1.37 / 0.44 logits somewhere in the 256k vocabulary at L6/12/18, and the
  top-1 token differs at some position. This is rounding, not a bug; the grader must fix ONE formula (we use the
  subtraction form everywhere, which is deterministic for a given batch).
- Batched (left-padded) vs one-at-a-time: last-token logits differ by <= 0.35, top-1 agrees on all probes.
**Behaviour validation** (clean top-1 is an accepted answer; 13 templates per family). Second run accepts a capitalised
answer too (" Basketball" after "A:"); the first run marked every athlete Q/A item wrong for that reason.
| family | items | accuracy | weakest templates (why) |
|---|---|---|---|
| city_state | 2938 | 0.86 | dialogue D1 0.47 (" the"), record K1 0.79 (abbreviations " MA") |
| city_capital | 2925 | 0.70 | few-shot F1 0.37 (answers a big city: " Seattle", " Chicago"), news N2 0.58 (" where"), plain P1 0.62 (" the") |
| country_lang | 1066 | 0.85 | news N1 0.00 (" a"), few-shot F1 0.70 |
| athlete_sport | 1300 | 0.86 | plain P2 0.19 (" called"), news N2 0.00 (" the"); 11 templates are 1.00 |
| langid | 2496 | 0.56 | record K1 0.00 (opens a quote), dialogue D1 0.08 (" a"), plain P1 0.15 and few-shot F2 0.28 (" English": copies the demo; 136 copy artefacts) |
| country_capital | 390 | 0.85 | plain P1/P2 0.17/0.27 (" a": "The capital of France is a ...") |
Few-shot copy artefacts (top-1 = a demo answer): langid F2 136, city_state 7, city_capital 2; these items are dropped.
No first-token collisions between groups' answers. Russian (2 countries) and German (3) are too small to be targets
(kept as siblings); every other group has >= 8 entities.
**SAE reconstruction test** (replace the residual by the SAE reconstruction, i.e. DROP the error term; 900 validated items):
clean answer kept on 0.92 / 0.89 / 0.94 of items at L6 / L12 / L18; wikitext KL 0.11 / 0.14 / 0.22 nats.
Weak spot: city_capital at L12 keeps only 0.60 (two-hop answer partly lives in the error term at L12), langid 0.83-0.85.
**Peak memory**: validation job peak RSS 6.28 GB (the bf16 model is memory-mapped), peak GPU 8.25 GB.
**Smoke sweep** (runs/latentknockout/20261002T0640_smoke, Texas at L18): one latent (13331) flips 59% of held-out
Texas city -> capital prompts with 100% sibling preservation and KL ~0; but for city -> state it flips only 17% of
held-out prompts although it flipped most of the example prompts (the greedy objective saw 0.68 on examples). The plain
steering vector flips 79-94%. Naive top-5 attribution breaks the siblings too (Preserve 0.15): its latents fire on every
"capital of the state" prompt, not on Texas. A first sign that small example sets overfit and that the error-term-free
SAE basis does not hold the whole "Texas" signal at L18. Full sweep launched 06:42Z (runs/latentknockout/20261002T0642_sweep).

## 2026-10-02T07:11Z Layer 18 sweep done (43 group-level cells; runs/latentknockout/20261002T0642_sweep/cells)
- Chain restarted at 06:57Z with fewer steps (L6 limited to 4 groups per family, one extra seed at L18 for the
  memorisation check, country -> capital at L18 only) to stay inside the time box; the L18 job itself was not touched.
- Headline at L18 (held-out, new-style prompts, k <= 5): the greedy reference reaches R_S >= 0.5 on 24 of 43 cells
  (0.56); median R_S 0.52; Effect 0.59, Preserve 0.99, KL ~0. Random and "no latents" are exactly 0.
- **A cheap recipe beats the reference.** Take the mean-difference vector (mean last-token residual of the example
  target prompts minus the example control prompts) and ablate the 5 latents whose decoder direction has the highest
  cosine with it. Median R_S 0.67, and it is at least half the reference's R on 29 of 29 feasible cells (q = 1.00,
  Wilson CI 0.86-1.00). It wins outright on 27 of 43 cells. The pre-registered rule says NO-GO-as-is (recipe).
  Naive attribution (predicted to be the recipe) is weaker: q = 0.50. Its latents fire on every prompt of the family
  ("capital of the state" latents), so Preserve drops (0.38).
- New-style held-out prompts are NOT harder than same-style ones (R_S/R_T median 1.06). The FeatureMatch style
  problem does not appear here, because the grader reruns the model on whatever prompts it likes.
- Keep-state check (city -> capital: break the capital, keep "Dallas is in Texas"): only 2 of 12 groups reach
  R x P_keep >= 0.5. The latents that carry "Arizona" carry it for both questions (P_keep 0.07 for Arizona,
  0.12 for Georgia). A same-entity-contrast reference (rank latents by capital-attribution minus state-attribution on
  the same cities) is queued for L12 and L18.

## 2026-10-02T07:33Z Layer 12 done (86 cells with L18); keep-state reference moved ahead of the other extras
- L12 vs L18 (median R_S at k <= 5): 0.47 vs 0.52; feasible cells 0.47 vs 0.56. Pairs feasible at their best layer:
  30 of 43 (0.70, Wilson 0.55-0.81). Athlete -> sport is the weak family (2 of 6 groups; soccer ~0 at every layer).
- **The recipe picture changes with the layer.** At L12 the mean-difference cosine recipe collapses (median R_S near 0):
  at L12 the group information still sits on the entity's own tokens, so the last-token mean difference does not
  point at the right latents (consistent with EditHunt's "copied to the final token later" finding). But contrastive
  top-5 attribution without any search step works at both layers: at least half of the reference's R on 0.89 of
  feasible cells (Wilson 0.76-0.95); naive attribution 0.70; cosine 0.55. Every cheap ranking clears the 0.5 line, so
  the pre-registered rule fires NO-GO-as-is (recipe).
- Keep-state at L12 is worse than at L18: 0 of 12 groups reach R x P_keep >= 0.5 (the target-vs-sibling reference
  keeps the state answer on a median 0.53 of the same cities).
- Killed the chain shell again (the running L6 job continued) so that the same-entity-contrast keep-state reference
  runs before the memorisation seed and country -> capital, since it decides the recommended design.

## 2026-10-02T08:09Z Feasibility study finished: NO-GO-as-is (recipe); write-up in FEASIBILITY.md
- All runs: runs/latentknockout/20261002T0642_sweep (106 group-level cells at L6/L12/L18, 24 country -> capital cells,
  24 keep-state cells, 31 seed-1 cells, latent_report.json, tables.json/tables.md from analyze.py).
- Pre-registered rule, in order: (1) NO-GO does not fire (p_pair 0.70, p_cell 0.42, median R(k<=10) at best layer 0.69);
  (2) NO-GO-as-is fires: naive top-5 attribution reaches half the reference's R on 0.70 of feasible cells (CI 0.56-0.82)
  and decoder cosine on 0.55 (CI 0.40-0.68). Contrastive top-5 (the reference's first step) on 0.89.
- Keep-state variant: the target-vs-sibling reference, a same-entity-contrast reference (capital attribution minus state
  attribution on the same cities) and the cheap rankings together reach R x P_keep >= 0.5 for 3 of 12 groups at L18
  and 0 at L12. The state latents are the same latents in both relations (e.g. 13331 for Texas at L18).
- Memorisation: a new example split picks the same first latent in 25 of 31 cells; different groups share almost no
  latents (Jaccard 0.02). Permute IDs and hold out groups/families.
- Country -> capital (single entity) is much harder than predicted: 1 of 12 at L18, 0 at L12 (two example prompts only;
  at L12 nothing moves a single capital).
- What the picks fire on: L12 reference picks are mostly entity-token latents (fire on "Dallas", "Tampa") or shared
  task-word latents; L18 picks fire at the answer position ("Which US state is Killeen in?" -> "?"). High-frequency
  format latents are <= 4% of reference picks (FeatureMatch-style concern threshold was 20%).
- Peak RSS: every GPU job 6.28-6.38 GB (the memory-mapped bf16 model), under MemoryMax=9G; analyze.py 0.13 GB.
  GPU peak 6.3 GB of the 8 GB requested; one job at a time through the queue; ~1h20m of GPU in total.
- Recommendation: do not build LatentKnockout as Dan's validated environment. The closest variant ("LatentKnockout-Verify",
  multi-slot with natural nulls, difficulty = verifying generalisation) is specified in FEASIBILITY.md section 14 with
  two gates; my probability that it passes them is ~0.35. Scripted example-objective verifier already gets ~0.48 of
  4-slot episodes, above the 10% recipe gate.

## 2026-10-02T08:20Z Independent reproduction started (skeptic pass on FEASIBILITY.md; files in skeptic_reproduce/)
- Goal: re-implement the ablation and the R metric from scratch (kx.py; no import of lk_core / analyze), reproduce
  6 reported cells (reported reference set + cosine / naive / contrastive baselines, plus my own re-derived rankings
  and greedy reference), check the split for leakage, measure seed and bootstrap stability, and try a stronger
  search (and an oracle that optimises on the grader's own held-out items) on 3 cells reported infeasible.
- Deliberate implementation differences: right padding with a plain causal mask (the original left-pads), my own
  rerun-from-layer path checked against a hooked full forward, my own example controls and a fresh held-out set
  (6 new templates per family in styles never used: trivia show, anecdote, checklist/form, census, blog, letter;
  plus 10-20 NEW entities per target group) so the generalisation claim is tested on prompts nobody tuned on.
- CPU leakage audit (leakage_check.py -> runs/latentknockout/skeptic/leakage.json): no entity is both an example and
  a held-out item in any of the 106 cells; the example split re-derived from the sweep's seeding rule matches every
  cell. One held-out "new style" template (city -> state D2 "User: Where is {e}?\nAssistant: {e} is a city in the US
  state of") contains a 6-word run of an example template, so it is not a new style. Surface-form sharing between
  example and held-out entities is rare (Serena/Venus Williams; San Diego/San Francisco/San Jose; Bay City).
- Smoke run (city -> state Arizona L18): cached-residual rerun == hooked full forward (max |margin diff| 0.0);
  250/250 held-out items clean-correct in my run (max |margin diff| vs reported 0.23, bf16 padding noise); my own
  greedy reference picks the same single latent (10472) as the reported reference.

## 2026-10-02T08:50Z Reproduction results so far (skeptic pass)
- Reproduced: 6 cells (city -> state Arizona L18, city -> capital Illinois L12 and Texas L18, country -> language
  French L18, langid Turkish L18, athlete golf L18). Reported R_S of the reference and of the cosine / naive /
  contrastive sets reproduce within 0.03 on the same held-out items with my own code (Turkish reference 0.35 -> 0.32).
  My own rankings pick the same latents. The recipe verdict reproduces on these cells.
- Biggest problem found: Effect counts an answer as "removed" when it merely loses first place. Held-out answers are
  often near ties (median lead 1.75 logits; 25% lead by < 1 logit), and 45-86% of the reference's flipped answers are
  still in the top 3 (often replaced by " the" / " where", i.e. postponed). Recomputed on the ORIGINAL sweep's margins:
  requiring the answer to lose by >= 0.5 logit gives p_pair 0.49 (< 0.50: the pre-registered NO-GO "nothing small
  works" fires); >= 1 logit gives p_pair 0.33, p_cell 0.15; >= 2 logits gives 2 of 106 cells. The recipe problem
  persists inside the few strict-feasible cells. (runs/latentknockout/skeptic/orig_robustness.json)
- Stability: my reference on 3 example splits gives Turkish L18 R = 0.40 / 0.94 / 0.90 (reported 0.35, labelled
  not feasible for the reference). Entity-clustered bootstrap CIs are wider than the item bootstrap used in the
  study (median width 0.23 vs 0.19); 23 of 86 L12/L18 cells have a CI containing 0.5.
- Fresh prompts I wrote (6 new-style templates per family + new entities): the reference set holds for Arizona (0.80),
  golf (0.80) and Texas capital (0.64), and drops for French (0.88 -> 0.63) and Turkish (0.32 -> 0.17). New-entity-only
  R is as high as or higher than the study's.
- Infeasible check, city -> state Texas L18 (reported reference 0.17): a stronger search (5 templates, 5 extra
  agent-written Texas towns, 121-latent pool, beam 3) reaches 0.34; an ORACLE that optimises directly on the grader's
  held-out items reaches 0.42 at k = 5. So it is genuinely infeasible on the study's held-out cities (famous ones:
  Dallas, Houston...), but on NEW, less famous Texas towns the same sets score 0.56-0.65. The null label depends on
  which entities the grader holds out (consistent with the study's "fame" reading).
- Memory: every GPU job peaked at 6.28 GB RSS (memory-mapped bf16 model) under an 8G cap; GPU peak 7.8 GB of 9 GB
  requested; one job of mine at a time (the shortcut-hunt job shared the GPU, so runs were ~2x slower).

## 2026-10-02T09:16Z Reproduction finished (skeptic pass): strict-effect oracle and the other two "infeasible" cells
- Strict-effect ORACLE (beam 1, ~72-latent pool, selected on the grader's held-out new-style items, then scored on my
  fresh prompts, which are out of sample). "Strict" = the answer must lose by >= 1 logit. Fresh R_strict at k = 5:
  Arizona L18 0.72, Illinois L12 0.73, Texas capital L18 0.77. The reported reference sets get only 0.47 / 0.40 / 0.32.
  So the collapse of p_pair under a strict effect (previous entry) comes mainly from the reference stopping at a
  minimal 1-2 latent set that just tips the top-1. It is not mainly a limit of the dictionary: 3-5 latents chosen for
  a real knock-out exist and generalise. The recipe problem persists under the strict effect: decoder-cosine top-5
  gets fresh R_strict 0.71 (Arizona) and 0.53 (Texas capital), against 0.72 / 0.77 for the oracle.
- Other cells reported infeasible: athlete soccer L18 (reported reference 0.01): stronger search 0.09, oracle on the
  grader's own items 0.14. country -> language English L18 (0.35): stronger search 0.35, oracle 0.38. Both confirmed
  infeasible at k <= 5 on the study's held-out set (within a ~70-110 latent pool). But the labels depend on the prompt
  set: the reported soccer set scores 0.43 on my fresh templates (0.02 on the study's), and Texas sets score 0.56-0.63
  on new, less famous Texas towns.
- GPU: about 55 min of wall time in 5 jobs (shared GPU), one job of mine at a time, peak 7.8 GB of 9 GB requested;
  peak RSS 6.28 GB per job under an 8G scope cap. No activations saved; the largest file is 0.26 MB.
- Not written: a separate FINDINGS.md (the subagent harness blocks report files); the ranked findings went to the
  orchestrator in the return message. Data: runs/latentknockout/skeptic/ (analysis_repro.md has every table).
