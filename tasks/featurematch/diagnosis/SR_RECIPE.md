# Step 3: the style-robust recipe (SR-max, SR-thr)

Code: `sr_recipe.py`. Tests: `tests/test_sr_recipe.py`. Spec: `PREREG.md` step 3 (binding), A1.5 (how the SR numbers
are used), predictions P11-P13 and P44. Brief: `PLAN.md` STEP 3.

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

## Which 6 texts (implementation choice, fixed before any SR result)

Bank R has 10 styles with 2 texts each for every concept. PREREG says "6 bank-R texts per option (one per style)" but
does not say which 6 styles or which of the two texts. PLAN.md does not say either. The rule below was fixed in code
before the recipe had produced any number on any pool:

- **Styles:** `random.Random(20261002).sample(sorted(10 style names), 6)`. This is one choice for the whole study, the
  same for every concept, so all options are compared on style-matched texts. The result is: how-to tip, trivia
  question, personal anecdote told aloud, interview Q&A, sports-radio or radio-show style commentary, text message.
  The 4 styles left out are press release, headline plus lede, email to a friend and museum or exhibit placard.
- **Text:** for each chosen style, the **first** text listed for that style in the bank-R file. This follows the F1
  convention of PREREG A1.1.
- **Sensitivity variants** (not pre-registered, reported next to the primary ones): `SR-max-all20` and `SR-thr-all20`
  use all 20 bank-R texts per option. That costs 400 forward units per slot, which is over every tier's budget, so
  these variants are not a feasible live recipe.
- **Selection spread** (descriptive only, `summary.json` `selection_spread`): SR-max and SR-thr planted accuracy under
  42 other selections: style seeds 20261002 to 20261022, each with the first and with the second text per style.

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

From `~/wt/fmdiag` after `source ~/ideating-rl-tests/common/env.sh`. The run uses CPU only, takes about 5 s and peaks
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
  the rule, and the metrics per variant for all tiers and for each tier. The metrics are: pass rate with Wilson 95%,
  planted accuracy with Wilson 95%, planted "nothing found" rate, null false-claim rate, mean score, forward units and
  over-cap count. It also holds the selection spread.
- `answers.jsonl`: one row per (variant, instance) with the submission and the real grader's output.
- `slots.csv`: one row per slot, with the pick, best mean, AUROC of the pick, rank of the true option, and each
  variant's choice and correctness.

If `--instances` is not a directory named `instances_v2f`, the label says "NOT the step-3 result". The filtered pool
may be rewritten after a run, so check its `pool_sha256` against the pool the step-3 report uses.

## Dry run on the UNFILTERED v2 pool: NOT the step-3 result

**v2 unfiltered, NOT the step-3 result.** This run checks the code on `~/wt/featurematch/tasks/featurematch/instances`
(180 instances, read-only), the pool before the style filter. It is not a pre-registered result. Only `summary.json`
is committed: `sr_recipe_out_v2_unfiltered_DRYRUN/summary.json` (pool sha256 `5baa3c2a...`, code at `827191af` plus
this file).

| Variant | Tier | Pass (Wilson 95%) | Planted acc (Wilson 95%) | Planted NF | Null false claim | Fwd mean / over cap |
|---|---|---|---|---|---|---|
| SR-max | all | 5/180 = 0.028 [0.012, 0.063] | 0.520 [0.473, 0.567] (n=431) | 0.000 | 1.000 (n=272) | 469 / 16 |
| SR-max | T1 | 1/60 = 0.017 [0.003, 0.089] | 0.524 (n=147) | 0.000 | 1.000 | 474 / 0 |
| SR-max | T2 | 1/60 = 0.017 [0.003, 0.089] | 0.564 (n=140) | 0.000 | 1.000 | 472 / 0 |
| SR-max | T3 | 3/60 = 0.050 [0.017, 0.137] | 0.472 (n=144) | 0.000 | 1.000 | 460 / 16 |
| SR-thr | all | 16/180 = 0.089 [0.056, 0.140] | 0.258 [0.218, 0.301] (n=431) | 0.712 | 0.055 (n=272) | 469 / 16 |
| SR-thr | T1 | 7/60 = 0.117 [0.058, 0.222] | 0.245 | 0.728 | 0.044 | 474 / 0 |
| SR-thr | T2 | 6/60 = 0.100 [0.047, 0.202] | 0.250 | 0.729 | 0.062 | 472 / 0 |
| SR-thr | T3 | 3/60 = 0.050 [0.017, 0.137] | 0.278 | 0.681 | 0.058 | 460 / 16 |
| SR-max-all20 | all | 4/180 = 0.022 [0.009, 0.056] | 0.640 [0.594, 0.684] | 0.000 | 1.000 | 1562 / 136 |
| SR-thr-all20 | all | 23/180 = 0.128 [0.087, 0.184] | 0.276 [0.236, 0.320] | 0.717 | 0.018 | 1562 / 136 |

Selection spread (SR-max planted accuracy over the 42 other 6-text selections): min 0.473, median 0.530, max 0.582.
For SR-thr: 0.234 / 0.276 / 0.339.

One line against the predictions: these numbers are on the unfiltered pool, so they do not test P11-P13. P11
predicts 0.75 on the filtered pool.

## Things to decide or keep in mind before the step-3 run

1. **The choice of 6 texts matters near the stop-rule line.** On unfiltered v2, SR-max planted accuracy goes from 0.47
   to 0.58 across equally valid 6-text selections, which straddles the stop rule's 0.50. The primary rule above is
   fixed, but if the filtered-pool number lands near 0.50, the step-3 report should show the spread next to it.
2. **T3 budget.** PREREG allows "at most 600 forward units per 5-slot episode", but the T3 forward cap is 550. On v2,
   16 of the 60 T3 episodes have 5 slots and would cost 600. The offline recipe follows PREREG and ignores the cap. In a
   live run the tool server would refuse after 550 units, so the last slot would get only 70 of its 120 texts. The
   over-cap count is reported.
3. **SR-thr negatives.** Following PREREG's wording, SR-thr negatives are the other options' bank-R texts. The
   reference solver instead uses a background sample from many concepts. So "the reference's threshold" is the same
   number applied to a slightly different AUROC.
4. **Offline vs live** values differ by bf16 batch noise (see above). This is not quantified, because it would need a
   GPU run.
