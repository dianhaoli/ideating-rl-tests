# Step 3: the style-robust recipe (SR-max, SR-thr)

Code: `sr_recipe.py`. Tests: `tests/test_sr_recipe.py`. Spec: `PREREG.md` step 3 (binding), A1.5 (how the SR numbers
are used), **Amendment 2** (A2.1-A2.5: text selection, sensitivity, T3 budget, SR-thr AUROC, clustered CI),
predictions P11-P13 and P44. Brief: `PLAN.md` STEP 3. One run produces everything step 3 and Amendment 2 ask for.

## What the recipe does, in plain words

The recipe asks one question: can a script with no reasoning solve FeatureMatch slots just by probing every option
in several writing styles? For each slot it takes 6 texts per option from bank R. Bank R is a set of styled texts
written by a different writer agent than bank F, which the filter used. It reads the slot latent's max activation on
each of the 120 texts and then answers in one of two ways:

- **SR-max** claims the option with the highest mean max activation. It never answers "nothing found".
- **SR-thr** makes the same pick, but claims it only if the AUROC of that option's 6 values against the other 19
  options' 114 values is >= 0.78 (the reference solver's threshold). Otherwise it answers "nothing found".

AUROC is `reference_solver.auroc`: the share of (positive, negative) pairs where the positive is higher, with ties
counting 1/2. If the mean activations tie, the lowest option number wins. Options are shuffled per instance, so this
pick is arbitrary but deterministic. A latent that is silent on all 120 texts therefore gets option 1 from SR-max and
AUROC 0.5 ("nothing found") from SR-thr.

The recipe is a baseline, so it reads the privileged `instance.json`: the layer, the real latent id and the menu's
concept ids. This is the same information `recipe_baseline.py` uses through the tool server's latent permutation.
For every slot the code checks that the public option labels equal the menu concepts' labels, in order, and that a
planted answer points to the anchor. Answers use the grader's format,
`{"answers": [{"slot": i, "choice": <1-20> | "nothing found"}]}`, and are scored by `grader.grade`. The tests also
pass them through the tool server's own format check (`tools.Env.validate_submission`).

## Which 6 texts (PREREG A2.1; fixed in code before any SR result)

Bank R has 10 styles with 2 texts each for every concept. PREREG step 3 says "6 bank-R texts per option (one per
style)" but did not say which. The rule below was fixed in code before the recipe had produced any number on any pool,
and Amendment 2 (A2.1) adopted it as the **primary** selection, the one the stop rule uses:

- **Styles:** `random.Random(20261002).sample(sorted(10 style names), 6)`. This is one choice for the whole study, the
  same for every concept, so all options are compared on style-matched texts. The result is: how-to tip, trivia
  question, personal anecdote told aloud, interview Q&A, sports-radio or radio-show style commentary, text message.
  The 4 styles left out are press release, headline plus lede, email to a friend and museum or exhibit placard.
- **Text:** for each chosen style, the **first** text listed for that style in the bank-R file. This follows the F1
  convention of PREREG A1.1.
- **All-20 variants** (A2.2, descriptive): `SR-max-all20` and `SR-thr-all20` use all 20 bank-R texts per option. That
  costs 400 forward units per slot, which is over every tier's budget, so these variants are not a feasible live
  recipe.

## What else one run reports (PREREG Amendment 2)

- **Text-selection sensitivity (A2.2)**, in `summary.json` `text_selection_sensitivity`. It gives SR-max and SR-thr
  planted accuracy under 42 selections: style seeds 20261002 to 20261022, each with the first and with the second text
  per style. The primary selection is one of the 42 (`"primary": true`), and the code checks that its value equals the
  graded primary. The block reports min / p10 / median / p90 / max (numpy linear percentiles) and the all-20 numbers.
  `text_selection_sensitive` is true when, for SR-max or SR-thr, **p10 <= 0.50 < p90**. Read this as: the 10th-90th
  range holds values on both sides of the stop rule's "> 0.50".
- **Capped variants (A2.3)** `SR-max-capped` and `SR-thr-capped`. The cap is read per instance from `instance.json`
  `caps.forward`, which must agree with `dial.forward_cap`. Texts are consumed in slot order: each slot takes
  min(120, what is left of the cap). Within a slot the texts go style-round-robin: chosen style 1 for options 1-20 in
  menu order, then style 2, and so on. A truncated slot therefore gives each option floor(k/20) or floor(k/20)+1
  texts. A2.3 does not fix this within-slot order; it is my implementation choice. An option with no text cannot be
  claimed. A slot with no text answers option 1 (SR-max) or "nothing found" (SR-thr). Per variant and tier the
  metrics report three counts:
  - `n_over_forward_cap`: episodes whose spend exceeds the cap (0 for the capped variants).
  - `n_episodes_needing_more_than_cap`: episodes whose uncapped recipe would exceed the cap.
  - `n_slots_truncated`: slots that got fewer than 120 texts.
- **SR-thr's AUROC (A2.4)**: the negatives are the other 19 options' 6 texts each, as step 3 words it. The reference
  solver uses a background sample instead. This is unchanged and only noted.
- **Clustered CI (A2.5)**: every planted accuracy (each variant, each tier) has `planted_acc_cluster_boot95` next to
  `planted_acc_wilson95`. It is computed from 10,000 resamples of instances with replacement, with indices drawn by a
  fresh `random.Random(20261002).choices` for each CI, taking the 2.5/97.5 percentiles of sum(correct)/sum(planted).
- **`stop_rule` block** (on the primary SR-max and SR-thr, all tiers pooled):
  - `fires`: planted accuracy > 0.50 for either variant, with n_planted >= 100.
  - `borderline`: it fires, and the Wilson CI of every firing variant includes 0.50. If one firing variant is clearly
    above 0.50, the outcome is not borderline.
  - `text_selection_sensitive`: as above.
  - `labels`: e.g. `["fires", "borderline", "text-selection-sensitive"]`.
  - `valid_as_step3_outcome`: true only when `--instances` is the `instances_v2f` directory. Otherwise the first
    label is "NOT a step-3 outcome".

## Offline computation and why it equals a live run

The recipe needs no forward pass. `diagnosis/cache/bank_acts_L{6,12,18}_R.npy` (float16 [4640, 16384]) holds every
bank-R text's max-pooled activation for every latent. Rows are in generator concept order (`bank_meta.json` `cids`),
20 per concept, in bank-file order. Checks done here:

- **Same per-text quantity.** `style_filter.py compute` uses `fm_core.Subject.maxpool_acts`: max over non-BOS,
  non-pad tokens of the JumpReLU SAE code of the residual after block L, with texts tokenized at
  `max_length = MAX_TOKENS + 1 = 65` (BOS plus 64 tokens). The tool `latent_activations` calls
  `fm_core.Subject.token_acts` with the same tokenizer call, the same residual hook, the same JumpReLU, BOS dropped,
  and returns `max` over the tokens. So the definition is the same: max over the same 64 tokens of the same latent.
- **Same texts.** Every run re-reads bank R, applies style_filter's normalisation (whitespace collapse, 400-character
  cut) and checks the SHA-256 and the per-row styles against `bank_meta.json`. Both match. The 400-character cut
  changes no bank-R text. The tool only cuts at 2000 characters. Whitespace collapse changes exactly one text: lang:ja,
  "email to a friend", first text, where U+3000 became a plain space. That style is not among the 6 chosen, so it can
  only affect the all20 variants.
- **Cache integrity.** Every run hashes the three R files and compares them with `cache/manifest.json`. They match. The
  GPU job's own consistency check (`bank_meta.json`) re-ran split A's first batch and matched `acts_L*_A.npy` exactly.
- **Not bit-identical.** The tool computes `x @ W_enc[:, idx]` in float32 for the requested latents only, on batches
  of at most 16 texts, and rounds to 4 decimals. The cache has the full encode on batches of 32 in concept order,
  stored as float16. Different batch composition means different padding and bf16 noise in the model. NOTES.md
  measured batched vs unbatched activations at r = 0.999. A live run could therefore flip a decision where two
  options' means nearly tie, or where a value sits exactly at the JumpReLU threshold. I did not quantify this, because
  it would need the GPU. AUDIT.md's statement that the precomputed tables "hold exactly the numbers
  `latent_activations` returns" holds up to this noise.
- **Cost.** A live run would send the same 6 x 20 = 120 texts per slot, so 120 forward units per slot and 360-600 per
  episode (8 `latent_activations` calls of up to 16 texts per slot). `summary.json` reports forward units per episode
  and how many episodes exceed their tier's cap.

## Usage

From `~/wt/fmdiag` after `source ~/ideating-rl-tests/common/env.sh`. The run uses CPU only, takes about 10 s and peaks
at about 210 MB RSS: the cache is memory-mapped and only the slot latents' columns are copied.

```bash
# step 3 (default --instances is tasks/featurematch/instances_v2f, the filtered pool)
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=1G -- \
    $PY -m tasks.featurematch.diagnosis.sr_recipe --instances tasks/featurematch/instances_v2f \
    --out tasks/featurematch/diagnosis/sr_recipe_out
# tests (CPU, < 1 s)
$PY -m pytest -q tasks/featurematch/diagnosis/tests/test_sr_recipe.py
```

Outputs in `--out`:
- `summary.json`: the label, the pool's SHA-256 (over every instance.json), the git sha, the bank and cache checks,
  the rules, `stop_rule`, `text_selection_sensitivity`, and the metrics per variant for all tiers and for each tier.
  The metrics are: pass rate with Wilson 95%, planted accuracy with Wilson 95% and clustered bootstrap 95%, planted
  "nothing found" rate, null false-claim rate, mean score, forward units, and the over-cap and truncation counts.
- `answers.jsonl`: one row per (variant, instance) with the submission and the real grader's output.
- `slots.csv`: one row per slot, with the pick, best mean, AUROC of the pick, rank of the true option, and each
  variant's choice and correctness.

If `--instances` is not a directory named `instances_v2f`, the label says "NOT the step-3 result". The filtered pool
may be rewritten after a run, so check its `pool_sha256` against the pool the step-3 report uses.

## Dry run on the UNFILTERED v2 pool: NOT the step-3 result

**v2 unfiltered, NOT the step-3 result.** This run checks the code on `~/wt/featurematch/tasks/featurematch/instances`
(180 instances, read-only), the pool before the style filter. It is not a pre-registered result, and its `stop_rule`
block says so (`valid_as_step3_outcome: false`). Only `summary.json` is committed:
`sr_recipe_out_v2_unfiltered_DRYRUN/summary.json` (pool sha256 `5baa3c2a...`). It was re-run after Amendment 2 with
the same label, and the primary numbers are unchanged from the first dry run.

| Variant | Tier | Pass (Wilson 95%) | Planted acc: Wilson 95% / clustered 95% | Planted NF | Null false claim | Fwd mean / over cap |
|---|---|---|---|---|---|---|
| SR-max | all | 5/180 = 0.028 [0.012, 0.063] | 0.520 [0.473, 0.567] / [0.473, 0.566] (n=431) | 0.000 | 1.000 (n=272) | 469 / 16 |
| SR-max | T1 | 1/60 = 0.017 [0.003, 0.089] | 0.524 [0.444, 0.603] / [0.444, 0.603] (n=147) | 0.000 | 1.000 | 474 / 0 |
| SR-max | T2 | 1/60 = 0.017 [0.003, 0.089] | 0.564 [0.482, 0.644] / [0.483, 0.643] (n=140) | 0.000 | 1.000 | 472 / 0 |
| SR-max | T3 | 3/60 = 0.050 [0.017, 0.137] | 0.472 [0.393, 0.553] / [0.389, 0.555] (n=144) | 0.000 | 1.000 | 460 / 16 |
| SR-thr | all | 16/180 = 0.089 [0.056, 0.140] | 0.258 [0.219, 0.301] / [0.219, 0.298] (n=431) | 0.712 | 0.055 (n=272) | 469 / 16 |
| SR-thr | T1 | 7/60 = 0.117 [0.058, 0.222] | 0.245 | 0.728 | 0.044 | 474 / 0 |
| SR-thr | T2 | 6/60 = 0.100 [0.047, 0.202] | 0.250 | 0.729 | 0.062 | 472 / 0 |
| SR-thr | T3 | 3/60 = 0.050 [0.017, 0.137] | 0.278 | 0.681 | 0.058 | 460 / 16 |
| SR-max-capped | all | 5/180 = 0.028 | 0.517 [0.470, 0.564] / [0.470, 0.565] | 0.000 | 1.000 | 464 / 0 (16 slots truncated) |
| SR-max-capped | T3 | 3/60 = 0.050 | 0.465 | 0.000 | 1.000 | 447 / 0 |
| SR-thr-capped | all | 16/180 = 0.089 | 0.258 (identical to SR-thr) | 0.712 | 0.055 | 464 / 0 |
| SR-max-all20 | all | 4/180 = 0.022 [0.009, 0.056] | 0.640 [0.594, 0.684] | 0.000 | 1.000 | 1562 / 136 |
| SR-thr-all20 | all | 23/180 = 0.128 [0.087, 0.184] | 0.276 [0.236, 0.320] | 0.717 | 0.018 | 1562 / 136 |

Text-selection sensitivity (42 selections), planted accuracy:
- SR-max: min 0.473, p10 0.502, median 0.530, p90 0.566, max 0.582.
- SR-thr: 0.234 / 0.255 / 0.276 / 0.306 / 0.339.

On this pool p10 > 0.50, so `text_selection_sensitive` is false, even though the min is below 0.50.

The `stop_rule` block (format check only) reads: `["NOT a step-3 outcome (pool is not instances_v2f)", "fires",
"borderline"]`.

One line against the predictions: these numbers are on the unfiltered pool, so they do not test P11-P13. P11
predicts 0.75 on the filtered pool.

## Things to keep in mind (all now settled by Amendment 2, except item 4)

1. **The choice of 6 texts matters near the stop-rule line.** This is handled by A2.2 (the spread and the
   `text_selection_sensitive` flag). The stop rule still uses the primary selection.
2. **T3 budget.** The T3 cap is 550, but 5 slots x 120 texts = 600. The primary run is uncapped. The capped variant
   (A2.3) shows the effect: on v2 it truncates the last slot in 16 T3 episodes and moves SR-max T3 planted accuracy
   from 0.472 to 0.465.
3. **SR-thr negatives** are the other options' texts (A2.4). This is only noted.
4. **Offline vs live** values differ by bf16 batch noise (see above). This is not quantified, because it would need a
   GPU run.
