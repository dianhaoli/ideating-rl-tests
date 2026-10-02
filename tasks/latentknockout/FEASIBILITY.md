# LatentKnockout feasibility study

Date: 2026-10-02 (06:20-08:20 UTC). Subject: gemma-2-2b + Gemma Scope residual SAEs (16k latents) at layers 6, 12, 18.
Pre-registered predictions and decision rule: `PREDICTIONS_FEASIBILITY.md` (commit c606abd4, frozen; not edited).
Lab notebook: `NOTES.md`. Code: `lk_core.py`, `lk_data.py`, `validate.py`, `sweep.py`, `analyze.py`,
`latent_report.py`, `report_tables.py`, `run_feasibility.sh`. Runs: `runs/latentknockout/` (section 15).

## 1. Verdict

**NO-GO-as-is (recipe)**, by the first rule of the pre-registered list that fires. In plain words:

- **Small latent sets that do the job exist.** For 30 of 43 (family, group) pairs (0.70, Wilson CI 0.55-0.81) there is a
  layer where a search that only sees a dozen example prompts finds <= 5 latents whose removal flips the held-out
  target prompts (new entities, new templates, new styles) while sibling groups keep their answers: R >= 0.5, and
  R >= 0.7 for 21 pairs. Removing nothing or random latents scores exactly 0. Layer 6 never works (0 of 20 cells);
  layers 12 and 18 work for about half the cells.
- **But a one-line recipe does just as well.** Rank latents by gradient x activation on the example prompts
  (no search, no contrast) and take the top 5: it reaches at least half the reference's R on 0.70 of feasible cells
  (Wilson 0.56-0.82; 0.95 at layer 12). Ranking by decoder cosine with the mean-difference vector does it on 0.55
  (1.00 at layer 18, where it often BEATS the reference). Contrastive top-5 attribution, the reference's own first
  step, does it on 0.89. Whatever the agent "understands", one of these rankings plus "try layers 12 and 18"
  solves the task. That is lesson (4) from FeatureMatch: the task as specified tests a recipe, not understanding.
- **The style problem that sank FeatureMatch does not appear here**: R on new-style held-out prompts is the same as on
  same-style ones (median ratio 1.03). Rerunning the model on held-out prompts measures generalisation directly.
- **The obvious fix does not rescue it.** Adding "keep the same city's other fact" (break city -> state capital,
  keep city -> state: the EditHunt Hard tier / RAVEL isolation) defeats the recipes, but it also defeats the reference:
  with every search I had (the target-vs-sibling reference, a reference that ranks latents by
  capital-attribution minus state-attribution on the same cities, and the cheap rankings), only 3 of 12 state groups
  reach R x P_keep >= 0.5 (4 if a cheap cosine set for Texas is counted), all at layer 18 and none at layer 12
  (predicted 0.25 of pairs). Where it is reachable, the
  cosine recipe matches the reference again. In this SAE the latents that say "this city is in Texas" are the same
  ones that the capital question uses, so the two hops cannot be separated with a few latents (section 10).
- Section 14 gives the closest design that I think is worth building, and what it would and would not measure.

## 2. What was measured (plain language)

**Subject.** gemma-2-2b (bf16) with the Gemma Scope residual-stream SAEs (16,384 latents) at layers 6, 12 and 18
(average L0 70 / 82 / 74: the closest cached widths to 70). A "latent" is one of the SAE's dictionary directions; on any
token only ~70-80 of the 16k are non-zero.

**Intervention ("knock out a set S of latents at layer L").** At every token except BOS:
x' = x - sum over i in S of f_i(x) * W_dec[i]. Only the chosen latents' contribution is removed; whatever the SAE
cannot explain (the error term) is kept, so S = {} leaves the model bit-identical.

**Behaviours (families), each with one-token answers and 13 prompt templates in 6 styles.** city -> state
(12 state groups), city -> capital of its state (12 groups, two-hop), country -> official language (5 target groups;
German and Russian are too small and serve only as siblings), athlete -> sport (6 groups), language identification of a
short sentence (8 language groups, non-factual), and country -> capital (one country per group, reported separately).
Styles: plain completion (P), question/answer (Q), dialogue/chat (D), key-value record (K), news sentence (N),
few-shot list (F). Every (entity, template) item was checked on the clean model; only items whose top-1 next token is
an accepted answer are used.

**One instance = (family, target group, layer).** For the target group (e.g. Texas cities -> " Texas"):
- *Example prompts* (what an agent would be shown): 3-4 entities of the group x 3 templates in the P and Q styles only.
- *Example controls*: the same templates for 3-4 entities of every other group (the reference builds these itself).
- *Held-out targets* (grader only): the group's OTHER entities, in (T) 2 new templates of the same P/Q styles, and in
  (S) 8 templates of 4 NEW styles (dialogue, record, news, few-shot). Up to 30 (T) and 80 (S) items per cell.
- *Held-out siblings* (grader only): held-out entities of all other groups in the family, T and S styles (up to 40 and
  100 items).
- *Unrelated text*: 24 wikitext paragraphs (64 tokens) for KL; the reference uses 8 different paragraphs.
- city -> capital only: *keep-state items* = city -> state prompts for the target group's held-out cities.

**Reward (held-out only).** Effect = share of held-out target items whose accepted answer is no longer top-1.
Preserve = share of held-out sibling items still answered correctly. KL = mean next-token KL(clean || ablated) on the
wikitext paragraphs; KL factor = 1 - min(1, KL / kappa), kappa = 0.1 nats. **R = Effect x Preserve x KL factor.**
R_S uses the new-style items (the pre-registered primary metric), R_T the same-style items. All 106 group-level cells
were "measurable" (>= 20 held-out target and >= 40 sibling items). The median 95% bootstrap CI width of a cell's R_S is
0.18 (items and wikitext resampled).

**Reference search (sees only the examples, its own controls and its own proxy text; never the held-out data).**
1. Gradient x activation attribution of the answer log-prob to every latent, on the example targets (naive score) and
   minus the same on the example controls (contrastive score).
2. Candidate pool: top-12 contrastive + top-4 naive.
3. Greedy forward selection with REAL ablations on the example objective J = Effect x Preserve x KL factor (each half
   hard, half soft so that ties break), up to 10 latents; KL is the sum of single-latent KLs on the proxy text.
4. R_ref(k) = held-out R of the greedy prefix of size <= k with the best example J.

**Baselines (same cells, k = 5).** No latents; 5 random latents; 5 random latents among those active on the examples;
5 most active latents on the examples; top-5 latents by decoder cosine with the mean-difference vector (mean example
target minus mean example control residual, last token); naive top-5 attribution; contrastive top-5 attribution (no
greedy); top-50 contrastive (uncapped, for the error-term question); and the plain non-SAE steering vector, which is
not a valid answer (subtract alpha x the mean difference at every position, alpha tuned on the examples; also alpha = 1,
and projecting the direction out).

## 3. Checks that the machinery is right (runs/latentknockout/20261002T0640_validate/exactness.json)

| Check (6 probe texts, every position, all 256k logits) | L6 | L12 | L18 |
|---|---|---|---|
| "Ablate nothing" vs clean model, max abs logit change | 0.0 | 0.0 | 0.0 |
| Rerun only the blocks above L from the cached residual vs full forward | 0.0 | 0.0 | 0.0 |
| SAE reconstruction + error term written out vs clean | 0.0 | 0.0 | 0.0 |
| Two algebraically equal formulas for ablating 3 latents | 0.67 | 1.37 | 0.44 |

Batched (left-padded) vs one-at-a-time: last-token logits within 0.35, same top-1. The last row is bf16 rounding: the
two formulas round differently and can swap a near-tied top-1 somewhere in a wikitext paragraph. **A grader must fix
one formula and one batching rule**; everything here uses the subtraction form.

## 4. Behaviour validation (runs/latentknockout/20261002T0636_validate2/validation_counts.json)

| Family | Items | Clean accuracy (predicted) | P/Q styles | New styles | Weakest templates (what the model says instead) |
|---|---|---|---|---|---|
| city -> state | 2938 | **0.86** (0.78) | 0.91 | 0.83 | dialogue "It's in" 0.47 (" the"); record "state:" 0.79 (" MA", " NY") |
| city -> state capital | 2925 | **0.70** (0.42) | 0.67 | 0.72 | few-shot list 0.37 (" Seattle", " Chicago": a big city); news 0.58 (" where") |
| country -> language | 1066 | **0.85** (0.72) | 0.96 | 0.79 | news "addressed the nation in" 0.00 (" a") |
| athlete -> sport | 1300 | **0.86** (0.80) | 0.84 | 0.88 | "The sport played by X is" 0.19 (" called"); 11 of 13 templates 1.00 |
| language id | 2496 | **0.56** (0.82) | 0.66 | 0.50 | record 0.00 (opens a quote); few-shot 0.28 (copies the demo " English") |
| country -> capital | 390 | **0.85** (0.85) | 0.67 | 0.96 | "The capital of France is" -> " a" (0.17) |
| **All** | 11115 | **0.75** (0.70) | 0.78 | 0.73 | |

- New styles are answered correctly 4.6 points less often than P/Q (predicted 10-20 points).
- Few-shot copy artefacts (top-1 = a demo answer) were found and dropped: language id 146, city -> state 7,
  city -> capital 2. The first validation run also showed that the model answers Q/A athlete prompts with a capital
  letter (" Basketball"); accepting both capitalisations fixed a template that scored 0%. Neither artefact is visible
  without per-template validation.

## 5. Is the behaviour in the SAE at all? (error-term test, recon.json)

Replace the residual at layer L by the SAE's reconstruction (throw the error term away) and check whether the clean
answer survives (900 validated items):

| Layer | Answer kept, all families (predicted) | city -> capital | language id | wikitext KL (nats) |
|---|---|---|---|---|
| 6 | 0.92 (0.90) | 0.81 | 0.83 | 0.11 |
| 12 | 0.89 (0.85) | **0.60** | 0.85 | 0.14 |
| 18 | 0.94 (0.80) | 0.96 | 0.91 | 0.22 |

The SAE carries most of what the answers need, with one exception: the two-hop capital at layer 12 (0.60), where the
error term carries part of the answer. Removing the top-50 contrastive latents (no cap) flips 0.35 / 0.71 / 0.86 of
held-out targets at L6 / L12 / L18 (predicted 0.50 / 0.85 / 0.85), with Preserve 0.89 / 0.86 / 0.78. So at layer 6 even
50 latents remove only a third of the behaviour; at 12 and 18 the behaviour is mostly in the dictionary. (The linear
"error node" attribution was too noisy to use: its median share had the opposite sign at L12 and L18.)

## 6. Main result: small latent sets exist at layers 12 and 18

### Best achievable R_S of the reference (median over cells; held-out, new styles)

| Layer | k=1 | k=2 | k=3 | k=5 | k=8 | k=10 | predicted k=1 / 3 / 5 / 10 |
|---|---|---|---|---|---|---|---|
| 6 | 0.05 | 0.09 | 0.09 | 0.14 | 0.18 | 0.17 | 0.08 / 0.15 / 0.20 / 0.28 |
| 12 | 0.26 | 0.39 | 0.41 | 0.47 | 0.49 | 0.49 | 0.25 / 0.40 / 0.48 / 0.56 |
| 18 | 0.35 | 0.49 | 0.51 | 0.52 | 0.52 | 0.52 | 0.30 / 0.48 / 0.55 / 0.62 |

Components of the k <= 5 set (medians): Effect 0.16 / 0.49 / 0.59, Preserve 0.98 / 0.98 / 0.99 and KL 0.001 / 0.001 /
0.000 nats at L6 / L12 / L18. The reference's sets are almost perfectly specific (Preserve ~0.99, KL ~0); what limits R is
Effect. R stops growing after k = 3 (Effect at k = 1 / 2 / 3 / 5 / 10 at L18: 0.35 / 0.53 / 0.54 / 0.59 / 0.59), so a cap
of k = 5 is not binding and k is a weak difficulty dial.

### Share of cells with R_S >= 0.5 / 0.6 / 0.7 (k <= 5)

| | >= 0.5 | >= 0.6 | >= 0.7 | predicted >= 0.5 / >= 0.7 |
|---|---|---|---|---|
| Layer 6 (20 cells) | 0.00 | 0.00 | 0.00 | 0.10 / 0.02 |
| Layer 12 (43 cells) | 0.47 | 0.44 | 0.28 | 0.45 / 0.15 |
| Layer 18 (43 cells) | 0.56 | 0.40 | 0.35 | 0.55 / 0.20 |
| All 106 cells | 0.42 | 0.34 | 0.26 | 0.37 / 0.12 |
| Pairs at their best layer (43) | **0.70** | | 0.49 | 0.60 / 0.25 |

(Layer 6 was run on 4 groups per family, 20 cells, to fit the time box.) Median R_S(k <= 10) over pairs at their best
layer: 0.69.

### By family (median R_S at k <= 5; share of cells >= 0.5)

| Family | L6 | L12 | L18 | feasible groups (best layer) |
|---|---|---|---|---|
| city -> state | 0.12 (0.00) | 0.45 (0.42) | 0.53 (0.58) | 8 / 12 |
| city -> state capital | 0.28 (0.00) | 0.61 (0.58) | 0.65 (0.67) | 10 / 12 |
| country -> language | 0.11 (0.00) | 0.56 (0.60) | 0.35 (0.40) | 3 / 5 |
| athlete -> sport | 0.11 (0.00) | 0.05 (0.00) | 0.34 (0.33) | 2 / 6 |
| language id | 0.17 (0.00) | 0.65 (0.62) | 0.64 (0.62) | 7 / 8 |

**Which pairs fail everywhere (no method reaches 0.5 at any layer)**: athlete baseball, basketball, soccer, tennis;
city -> state for **California, Florida, New York and Texas**; city -> capital Michigan; country -> language English.
The famous states fail and the less famous ones (Arizona 0.78, Georgia 0.87, Washington 0.85, Illinois 0.88) work.
A plausible reading (not tested further): frequent concepts are split over many specialised latents in a 16k SAE
("Texas" in politics, sport, geography...), so no 5 of them cover the group, while rarer concepts get one latent. The
same pattern holds for sports. These are natural "cannot be done within the cap" instances, and the reason is a
property of the dictionary, not of the search. But they cluster by family and by fame, which a policy could learn as a
prior (section 11).

## 7. Baselines: the recipe problem

On the 44 feasible cells (reference R_S >= 0.5), ratio = R_baseline / R_ref, k = 5:

| Baseline | Median ratio (predicted) | q = share of cells with ratio >= 0.5 [Wilson CI] | ratio >= 0.8 | Median Effect / Preserve / KL |
|---|---|---|---|---|
| No latents | 0.00 (0) | 0.00 [0.00, 0.08] | 0.00 | 0.00 / 1.00 / 0 |
| 5 random latents | 0.00 (0.01) | 0.00 [0.00, 0.08] | 0.00 | 0.00 / 1.00 / 0.000 |
| 5 random among active on the examples | 0.00 (0.08) | 0.00 [0.00, 0.08] | 0.00 | 0.00 / 1.00 / 0.000 |
| 5 most active on the examples | 0.05 (0.08) | 0.16 [0.08, 0.29] | 0.02 | 0.05 / 0.97 / 0.026 |
| Decoder cosine with the mean difference | 0.79 (0.35) | **0.55** [0.40, 0.68] | 0.50 | 0.61 / 1.00 / 0.001 |
| Naive top-5 attribution | 0.83 (0.55) | **0.70** [0.56, 0.82] | 0.59 | 0.86 / 0.82 / 0.001 |
| Contrastive top-5 attribution (no search) | 0.87 (0.80) | **0.89** [0.76, 0.95] | 0.68 | 0.79 / 0.93 / 0.002 |
| Top-50 contrastive (exceeds the cap) | 0.56 | 0.57 [0.42, 0.70] | 0.18 | 0.94 / 0.78 / 0.035 |
| Steering vector, alpha tuned (not submittable) | 0.18 (0.95) | 0.29 [0.18, 0.44] | 0.16 | 0.26 / 0.97 / 0.026 |
| Steering vector, alpha = 1 | 0.09 (0.60) | 0.27 [0.16, 0.42] | 0.16 | 0.72 / 0.96 / 0.033 |
| Projecting the mean-difference direction out | 0.06 (0.65) | 0.18 [0.10, 0.32] | 0.07 | 0.06 / 0.99 / 0.002 |

By layer (q, share of feasible cells where the recipe reaches half the reference): **layer 12**: naive attribution 0.95,
contrastive 0.95, cosine 0.00; **layer 18**: cosine 1.00, contrastive 0.83, naive 0.50. The recipe that works depends on
the layer (at 12 the group information still sits on the entity's tokens, so a last-token mean difference points the
wrong way; at 18 it has been copied to the last token, consistent with EditHunt's handoff finding), but at every useful
layer at least one cheap ranking matches the reference. The reference's first latent is the contrastive top-1 in
11 of 12 city -> state cells at both layers. On held-out data, the cosine recipe beats the reference outright in 33 of
106 cells; the best of {reference, cosine, naive, contrastive} reaches 0.5 on 0.77 of pairs.

Two predictions were wrong in instructive ways:
- Naive attribution was predicted to be the borderline recipe (0.55); it is worse than predicted at L18 (its latents are
  "capital of the state" / "language" task latents that fire on every prompt of the family, Preserve 0.38 on city ->
  capital) and better at L12 (0.95).
- The plain steering vector was predicted to be about as good as the reference (ratio 0.95). It is much worse (0.18):
  subtracting a last-token mean difference at every position adds KL (0.03 nats) and at L12 does nothing (Effect ~0).
  At L18 its raw Effect is higher than the latents' (0.78 vs 0.59) but its KL costs more. So the SAE sets here are not a
  handicap relative to this dense baseline; the precedent survey's DiffMean-style comparators (directional ablation at
  the right positions, LEACE) were not run and might do better.

KL is not where the difficulty is: the reference's sets have KL <= 0.018 nats (median 0.0007), and the feasible share
and the recipe verdict do not change between kappa = 0.02 and 0.2 (naive q 0.56 at kappa 0.02, 0.70 above 0.05).

## 8. Style: new-style held-out prompts are not harder

| | predicted | observed |
|---|---|---|
| R_S / R_T for the reference on feasible cells (median) | 0.80 | **1.03** (L12 0.99, L18 1.06) |
| Share of feasible cells with R_S < 0.7 x R_T | (rule (d): > 1/3 fires) | **0.00** |
| Same ratio for naive attribution | 0.70 | 1.08 |
| Reference Effect on held-out entities / on its own example prompts | 0.80 | 0.80 |

Latents that carry a group concept fire on the concept in any wording, so a set found on plain and Q/A prompts transfers
to dialogue, records, news and few-shot lists. This is the opposite of FeatureMatch, where the answer key was defined
on encyclopedic text. Here the grader defines success by rerunning the model, so style cannot leak into the score.
Generalisation to new *entities* costs 20% of Effect, as predicted.

## 9. Failure structure and what the selected latents fire on

Run: `runs/latentknockout/20261002T0642_sweep/latent_report.json`. For every latent picked by the reference, naive
attribution or the cosine recipe, I scanned a mixed corpus at its layer: 700 validated prompts per family in all six
styles plus 200 wikitext paragraphs. I recorded its top-activating contexts, its firing rate on wikitext, and where in
the prompt it fires. A coarse classification of the picks (each pick counted once per cell that chose it):

| Layer, method | Group-specific, fires on the entity's tokens | Group-specific, also fires at the answer position | Shared by >= 3 groups ("state", "capital", "language" words) | High-frequency (> 10% of wikitext tokens) |
|---|---|---|---|---|
| L12 reference | 0.42 | 0.13 | 0.41 | 0.04 |
| L12 naive attribution | 0.27 | 0.14 | 0.53 | 0.06 |
| L12 cosine | 0.14 | 0.26 | 0.47 | 0.13 |
| L18 reference | 0.03 | **0.65** | 0.28 | 0.03 |
| L18 naive attribution | 0.00 | 0.33 | **0.57** | 0.10 |
| L18 cosine | 0.08 | **0.69** | 0.20 | 0.02 |

Examples (top-activating context -> token):
- L12, city -> state Texas: latent 1114 peaks on the city name itself ("Officials in" -> " Dallas", "Lawmakers from" ->
  " Dallas"); 4939 (Florida) on " Sarasota", " Tampa"; 3017 (Arizona) on "Bisbee"; 680 (Pennsylvania) on
  " Philadelphia". These are entity-token latents for one state's cities, as EditHunt found for the state direction.
- L18: 13331 (Texas), 10472 (Arizona), 5465 (Georgia), 11513 (California), 14792 (Illinois), 15026 (Florida) all peak
  on the "?" or newline that ends "Which US state is Killeen in?". They are "the answer here is Texas" latents at
  the answer position. Each is picked for BOTH the city -> state and the city -> capital cell of its state.
- Generic picks: 13527 (" capital", "traveled to"), 1782 (" state", " capital"), 12132 (" state", " capital"),
  8613 (" language", ":"). These are task-word latents that naive attribution loves, and they are why it breaks the
  siblings. 2291 at L12 and 7373, 4651 at L18 fire on 45-60% of all wikitext tokens: format/position latents. They
  are 3-4% of reference picks and up to 13% of cosine picks.
- The FeatureMatch-style failure (picking latents for formatting cues) is small here: high-frequency picks are <= 4% of
  the reference's picks (pre-registered threshold for concern: 20%). The bigger reading trap is the 28-41% of
  reference picks that are shared task-word latents. Their removal helps on the examples and does not hurt the siblings
  much, yet their top contexts say nothing about the target group.

- **Feature splitting at the entity level is rare** among picks: 3% of the reference's picks and 2% of naive picks fire on
  <= 2 of the group's held-out entities (predicted 10% and 25%). But this count is generous: a latent that fires on a
  template word anywhere in the prompt counts as covering every entity.
- **Picks are not all concept latents.** 31% of the reference's picks fire at least as often on sibling prompts as on
  target prompts, and 48% fire on more than 1% of wikitext tokens. Their removal still leaves siblings intact
  (Preserve 0.98), so firing is not the same as mattering. This is FeatureMatch lesson (2) in a milder form: reading a
  latent's activations alone would mislabel a third of the picks.
- **Absorption by fame** (predicted weak, <= 0.15): the opposite. The famous states and the popular sports are the
  failures (section 6), a strong effect at the group level.
- **Sibling hits** limit naive attribution (Preserve 0.82 overall, 0.38 on city -> capital at L18) but not the reference
  (0.98-0.99).
- **Error term**: not the main obstacle at 12 and 18 (section 5); at layer 6 nothing small or large works.

## 10. The keep-state variant (same-entity other relation)

Target: city -> state capital for the group's held-out cities. Extra control: city -> state on the SAME held-out cities
must survive ("Dallas is in Texas" stays, "the capital of Dallas's state is Austin" breaks). Score R_ks = R_S x P_keep.

| City -> capital cells, k <= 5 (12 groups per layer) | L12 | L18 |
|---|---|---|
| Plain reference (target vs siblings): R_S / P_keep (medians) | 0.61 / 0.53 | 0.65 / 0.65 |
| Best of three references (plain, + keep-state controls in the greedy objective, + same-entity contrast pool): median R_ks | 0.32 | 0.41 |
| Same, k <= 10 | 0.32 | 0.42 |
| Groups with R_ks >= 0.5 | **0 / 12** | **3 / 12** (Illinois 0.61, Ohio 0.67, Massachusetts 0.53) |
| Same-entity-contrast top-5 (rank by capital attribution minus state attribution), median R_ks | 0.21 | 0.14 |
| Decoder cosine top-5, median R_ks (q on the 3 feasible cells) | 0.00 | 0.22 (1.00) |
| Naive top-5 attribution | 0.22 | 0.11 (0.00) |
| Contrastive top-5 attribution | 0.24 | 0.29 (0.33) |
| Steering vector (not submittable) | 0.02 | 0.17 |

Run: `runs/latentknockout/20261002T0642_sweep/cells_ks` (reference-only, merged with the main cells by group and layer).

- **Prediction**: reference R_ks 0.30 and 0.25 of pairs feasible; naive ratio 0.20. **Observed**: 0.32-0.41 and 0.25;
  naive at L18 never reaches half the reference. The prediction was right, and it is not good news: the conjunct
  defeats the recipes mostly because it defeats everything.
- **Why** (from section 9): at L12 the state is carried by latents that fire on the city's own tokens (e.g. 1114 on
  "Dallas", 4939 on "Tampa", 3017 on "Bisbee"). Every later computation, the state answer and the capital answer, reads
  them. At L18 the reference's latents for Texas, Arizona or Georgia are the SAME latent in the city -> state and the
  city -> capital cells (13331, 10472, 5465/5477). They fire at the answer position of state questions, so
  they encode "the state in question is Texas", which the capital question also needs. A latent that encodes only
  "capital of Texas = Austin" would be needed; the 16k dictionary has a usable one for at most 3 of 12 states.
- So "break the second hop, keep the first" (the EditHunt Hard tier, T2) is not a good fit for SAE latent answers at
  these layers. It would be an honest "cannot be done within the cap" for about 3/4 of instances, and a policy would
  learn to say "cannot".

## 11. Memorisation risk (FeatureMatch lesson 3) and a held-out split

Run: `runs/latentknockout/20261002T0642_sweep/cells_seeds` (seed 1: new example entities, new controls and new
held-out sampling for 31 cells at L18: city -> state, country -> language, athlete -> sport, language id; reference only).

| Question | Answer |
|---|---|
| Distinct (family, group, layer) cells | 43 pairs x 3 layers = 129 (86 at the useful layers 12/18), plus 30 x 3 single-country cells |
| Same cell, new example split: same FIRST latent chosen | **25 of 31 (0.81)** |
| Same cell, new split: overlap of the k <= 5 sets (Jaccard) | median 0.25, mean 0.36 |
| Different groups, same family and layer: overlap (Jaccard) | mean 0.02 (16% of pairs share any latent) |
| Same cell, new split: change in held-out R_S | median SD 0.04; feasibility label (R >= 0.5) agrees in 27 of 31 cells |

So the "right answer" of a cell is essentially fixed: one latent per (group, layer) does most of the work, and a new
example split finds it again 81% of the time. A policy trained on repeated cells could memorise
"Texas, layer 18 -> latent 13331" from reward alone (FeatureMatch lesson 3). Two cheap defences, both needed:
1. **Permute latent IDs per episode** (a fixed random permutation of the 16,384 rows of W_enc/W_dec/b_enc/threshold,
   applied inside the tools and inverted by the grader). This kills ID memorisation; the agent must re-find the latent
   by probing each time. The recipes still work under permutation, so this does not fix the recipe problem.
2. **Hold out whole groups and families for evaluation.** Proposed split for the 4 families with >= 3 feasible groups:
   train on city -> state and language id; evaluate on city -> capital and country -> language (different relation,
   different answer vocabulary), plus held-out groups inside the training families (e.g. Arizona, Georgia, Illinois
   and Washington for city -> state; Italian and Turkish for language id). Even then the pool is small (about 30
   feasible pairs in total). More families (e.g. company -> country, landmark -> city, animal -> class, verb -> past
   tense) would be needed before any RL training.

## 12. Single-entity cells (country -> capital)

One country per cell (e.g. France -> " Paris"); held-out = the same country in new templates and styles (1 same-style
+ up to 8 new-style items: too few to be "measurable", so these cells never counted toward the verdict); siblings = the
other countries. The plain templates "The capital of France is" fail validation (" a"), so the examples are only
2 prompts (P3 "{e} has its capital in" + Q1). 12 countries at L12 and L18.

| | L12 | L18 | predicted (best layer) |
|---|---|---|---|
| Reference: median R_S; cells >= 0.5 | 0.00; 0 / 12 | 0.21; 1 / 12 (Norway 0.92) | 0.80 of entities >= 0.5 |
| Cosine top-5: median; cells >= 0.5 | 0.00; 0 / 12 | 0.25; 4 / 12 (Russia 0.81, Peru, Japan) | |
| Naive top-5: median; cells >= 0.5 (Preserve) | 0.00; 0 / 12 | 0.28; 3 / 12 (0.66) | |
| Steering vector Effect | 0.00 | 0.62 (R 0.09: KL) | |

The single-fact version is much harder than predicted, not easier: at L12 nothing moves a single capital (Effect 0 for
every method), and at L18 a few latents remove it for only some countries. Two prompts of examples are also too few for
a search. Single-entity cells are not a good instance type for this task.

## 13. Predictions vs outcomes

| Pre-registered prediction | Predicted | Observed | |
|---|---|---|---|
| Clean accuracy, all items | 0.70 | 0.75 | close |
| Clean accuracy city -> capital (two-hop) | 0.42 | 0.70 | too pessimistic |
| Clean accuracy language id | 0.82 | 0.56 | too optimistic (copying the demo, quote-opening) |
| New styles answered less often | -10 to -20 pts | -4.6 pts | smaller |
| SAE reconstruction keeps answer L6/L12/L18 | 0.90 / 0.85 / 0.80 | 0.92 / 0.89 / 0.94 | close; L18 better |
| Top-50 ablation Effect L6 / L12 / L18 | 0.50 / 0.85 / 0.85 | 0.35 / 0.71 / 0.86 | close |
| Median R_S at k=5 L6 / L12 / L18 | 0.20 / 0.48 / 0.55 | 0.14 / 0.47 / 0.52 | close |
| Share of cells R >= 0.5, all / pairs at best layer | 0.37 / 0.60 | 0.42 / 0.70 | close, slightly better |
| Share R >= 0.7, pairs at best layer | 0.25 | 0.49 | better |
| Random / no latents ratio | 0.01 / 0 | 0.00 / 0.00 | right |
| Most-active ratio | 0.08 | 0.05 | right |
| Cosine ratio, q | 0.35, 0.30 | 0.79, 0.55 | **far too low** (layer 18) |
| Naive attribution ratio, q | 0.55, 0.55 | 0.83, 0.70 | recipe stronger than predicted |
| Contrastive ratio | 0.80 | 0.87 | close |
| Steering vector tuned ratio | 0.95 | 0.18 | **far too high** |
| Style ratio R_S / R_T | 0.80 | 1.03 | wrong direction: no style penalty |
| Held-out / example Effect | 0.80 | 0.80 | right |
| Entity-specific picks (reference / naive) | 10% / 25% | 3% / 2% | fewer |
| Fame effect | weak (<= 0.15), within a group | not tested within groups; across groups the famous states and popular sports are the failures | unexpected |
| Country -> capital (single entity), share >= 0.5 at best layer | 0.80 | 0.08 (reference), 0.33 (cosine) | **far too optimistic** |
| Keep-state variant: reference R, feasible pairs | 0.30, 0.25 | best reference 0.32 (L12) / 0.41 (L18); 3/12 groups (0.25); naive q 0.00 at L18 | |
| **Verdict** | NO-GO-as-is 0.40, ADJUST 0.35, GO 0.15, NO-GO 0.10 | **NO-GO-as-is (recipe)** | modal outcome |

## 14. Decision rule applied, and the recommended design

**Rule applied in order** (all on held-out new-style prompts, k <= 5, kappa = 0.1):
1. NO-GO (nothing small works): p_pair = 0.70 (>= 0.50), p_cell = 0.42 (>= 0.25), median R(k<=10) at best layer 0.69
   (>= 0.5). **Does not fire.**
2. NO-GO-as-is (recipe): q > 0.50 for naive attribution (0.70, CI 0.56-0.82) and for decoder cosine (0.55, CI
   0.40-0.68, borderline, still fires as pre-registered). **Fires.** Contrastive top-5 (not in the pre-registered cheap
   list, it is the reference's first step) is at 0.89, and it reaches >= 0.8 x R_ref on 0.68 of feasible cells, so the
   rule (f) note ("the reference is just a fixed two-step script") also applies.
3. For the record, the ADJUST conditions: (b) no (p_pair 0.70, p_cell 0.42); (c) no (4 families have >= 3 feasible
   groups: city -> capital 10, city -> state 8, language id 7, country -> language 3); (d) no (style ratio 1.03);
   (e) no (reconstruction >= 0.89 at every layer).

### Recommendation

**Do not build LatentKnockout as specified (or with the keep-state conjunct) as Dan's validated environment.**
- As specified it tests whether the agent runs one standard ranking (gradient x activation at layer 12, or decoder
  cosine with a mean difference at layer 18): q = 0.70-0.95 by layer, against a pre-registered limit of 0.50.
- Making it harder with the precedented conjunct (keep the same entity's other fact) removes the recipe's advantage
  only by making the task impossible for almost every instance (3 of 12 groups feasible at L18, 0 at L12).
- k, kappa and style are weak dials here: R saturates at k = 3, the reference's KL is ~0, and new styles are no harder.

**What is worth keeping** (reusable in other tasks): the grader shape (rerun the model on private held-out prompts
of new entities and new styles; product reward; random / no-op = exactly 0), `lk_core.py` (bit-exact ablation with the
error term kept; cached-residual reruns 2-4x faster), the style-tagged families with per-template validation
(it caught a capitalisation bug and 155 few-shot copy artefacts), and two findings worth a sentence in the pitch:
grading by rerunning the model removes FeatureMatch's style problem, and **famous concepts are the ones a few SAE
latents cannot knock out** (Texas, California, Florida, New York; soccer, basketball, tennis, baseball; English), which
looks like concept splitting in a 16k dictionary. "Which concepts are knock-out-able with <= k latents, and can an
agent predict it from probing?" is a research question Dan could own.

### Closest variant (NOT validated; two gates must pass before building): "LatentKnockout-Verify"

The one place where this study found a gap between a script and careful work is **verification**: deciding whether
a set found on a few examples generalises. Six cells look solved on the examples but fail on held-out prompts (example
objective >= 0.6, held-out R < 0.5 with upper CI < 0.5; e.g. Texas at L18: 0.68 on examples, 0.17 held-out), and three
feasible cells look unsolved on the examples. An agent that writes its own test prompts (other Texas cities, other
phrasings) can tell; a script that trusts the examples cannot.

| Design element | Choice and reason |
|---|---|
| Subject | gemma-2-2b; Gemma Scope 16k residual SAEs at layers 12 and 18 only (layer 6 never works: an all-null layer would be a fingerprint) |
| Families and groups | city -> state (12), city -> capital (12), country -> language (5), language id (8), athlete -> sport (6); add >= 3 new families before any RL training (section 11) |
| Slot | (family, group, layer). The agent gets the concept in words ("cities in Texas -> ' Texas'"), 3-4 example entities x 3 P/Q templates, and the generic tools (run prompts, SAE encode, ablate latents and rerun, top-activating examples on a fixed corpus). Latent IDs are permuted per episode |
| Answer per slot | a set of <= 5 latents, or "cannot be done with 5 latents" |
| Grader per slot | rerun the model on private held-out prompts: >= 20 new-entity target items in 8 new-style templates (+ 2 same-style), >= 40 sibling items, 24 wikitext paragraphs. R = Effect x Preserve x (1 - min(1, KL / kappa)), kappa = 0.1 nats (a safety term; not binding for sensible sets), **tau = 0.5**. A set passes if R >= tau. "Cannot" passes only on a null slot; a set that reaches tau on a null slot also passes (the grader stays objective) |
| Null definition | no method (reference greedy up to k = 10 on two example splits; contrastive, naive and cosine top-5) reaches 0.5 on held-out, AND the best set's bootstrap upper CI < 0.5. Feasible = reference >= 0.6 on both splits. Cells in between are dropped. This study has 29 strict nulls and 44 feasible cells at L12/L18 |
| Episode | 3-5 slots, 30-50% nulls (drawn, not fixed), pass = all slots right, continuous score = mean slot score |
| Dials | null fraction; share of "deceptive" slots (nulls that look solved on the examples, feasible cells that do not); number of example entities; layer (12 vs 18 changes which recipe works). k is a weak dial (R saturates at k = 3) |
| Measured baselines | "always submit contrastive top-5": 0.70 of feasible slots, 0 of nulls (~0-5% of 4-slot episodes with a null); "always cannot": 0.4^4; **script that submits the greedy set when its example objective >= t, else "cannot": 0.83 per slot (2-fold CV), ~0.48 of 4-slot episodes**: fails the <= 10% recipe gate as is |
| Expected difficulty | unknown; a guess of 30-60% per episode for an Opus-class agent if deceptive slots are enriched, but that is not measured |
| Shortcut risks | fame prior (famous groups and popular sports are the nulls); family prior (athlete -> sport is mostly null); memorisation of the latent per cell (permute IDs, hold out groups and families); the example-objective script above; an agent "verifying" on easy prompts of its own |

**Gates before building** (about 1 h of GPU): (1) add 3 families and 2 more example splits per cell, then check whether
a slot mix exists where the example-objective script passes <= 10% of episodes while the reference plus self-made
held-out prompts (new entities from a list, new styles) passes >= 60%. (2) Run the precedent survey's controls, which
were not done here: dense DiffMean directional ablation at the entity and answer positions, a random-direction
dictionary, and negative clamping instead of zero ablation. If (1) fails, drop LatentKnockout and record it as a dead
end with this document as the explanation. My probability that (1) passes: about 0.35.

## 15. Runs, compute and memory

| Run | What | Peak RSS |
|---|---|---|
| runs/latentknockout/20261002T0640_validate | exactness checks, first validation (athlete Q/A capitalisation bug) | 6.28 GB |
| runs/latentknockout/20261002T0636_validate2 | validation used by the sweep, reconstruction test | 6.28 GB |
| runs/latentknockout/20261002T0640_smoke | 2-cell smoke test of the sweep | 6.28 GB |
| runs/latentknockout/20261002T0642_sweep/cells | 106 group-level cells (L18, L12, L6) + 24 country -> capital cells (L12, L18) | 6.28 GB per job |
| runs/latentknockout/20261002T0642_sweep/cells_ks | keep-state same-entity-contrast reference (city -> capital, L12/L18) | 6.28 GB |
| runs/latentknockout/20261002T0642_sweep/cells_seeds | memorisation check (seed 1, L18) | 6.28 GB |
| runs/latentknockout/20261002T0642_sweep/latent_report.json | what the selected latents fire on | 6.38 GB |
| runs/latentknockout/20261002T0642_sweep/tables.json | analyze.py output behind every table here | CPU, < 1 GB |

Peak RSS is the model's memory map (bf16 weights) and is the same for every job; GPU peak 6.3 GB (8 GB requested, one
job at a time, through the queue). Every job ran under `systemd-run --scope -p MemoryMax=9G`. No activations were
saved; the largest file is a cell JSON of ~60 KB. Regenerate: `bash tasks/latentknockout/run_feasibility.sh
runs/latentknockout/20261002T0636_validate2 <out_dir> L18 L12 L6 ks seeds cc report`, then
`python -m tasks.latentknockout.analyze <out_dir>/cells --seeds <out_dir>/cells_seeds --ks <out_dir>/cells_ks --out
<out_dir>/tables.json`.

**Limitations.** One example split per cell (plus one more seed at L18); greedy search only (no exhaustive search);
L6 on 4 groups per family; the cosine and attribution recipes were scored on held-out data, the reference selects on
examples only, so "recipe beats reference" partly reflects a weak reference (the oracle best-of-methods is 0.77 of
pairs); no dense comparators beyond the steering vector (DiffMean directional ablation at the right positions, LEACE),
no negative clamping, and no random-dictionary control (all suggested in PRECEDENT.md).
