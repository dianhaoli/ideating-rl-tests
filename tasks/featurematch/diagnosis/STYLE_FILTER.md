# Step 2: the style-robustness filter

Code: `style_filter.py` (compute + analyze), `write_v2f.py` (writes the filtered pool), tests in
`tests/test_style_filter.py`. Brief: `PLAN.md` STEP 2 and STEP 1b (binding), `PREREG.md` step 2.

## What the filter does, in plain words

Each FeatureMatch slot says "this latent belongs to concept c\*". The generator decided that from **one kind of
text**: encyclopedia-style dataset sentences (split A). A latent can separate c\* perfectly on that kind of text and
still go quiet when c\* shows up in a chat message, a news report or a dialogue. An agent writes its own probe texts,
usually not in encyclopedia style. On such a latent it sees no signal, answers "nothing found" or guesses, and fails.
That failure says nothing about interpretability skill. It is a property of how the answer key was made.

The filter re-checks every key on **many styles** and keeps only the slots whose key survives:

1. **Multi-style key.** For each concept we have bank F: 20 texts in 10 styles (two per style), written by a separate
   writer agent. The two texts of each style are split at random (fixed seed) into F1 and F2. The multi-style key is
   the option the latent separates best on split A **plus** F1.
2. **Agreement.** The multi-style key must be the same concept as the original key.
3. **Robustness.** On texts used by neither key (dataset split C **plus** F2), the latent must still separate the key
   concept with AUROC >= 0.85.
4. **Null slots.** The same test applies to the latent and its anchor concept. In addition, no option on the menu may
   reach 0.85 on C + F2. A null whose latent now separates one of the options is flagged and dropped, because
   "nothing found" would no longer be clearly right.

AUROC is the generator's (`auroc.auroc_table`): the probability that a text of the concept gets a higher max
activation than a text of any other concept in the 232-concept universe, ties counting one half.

**Why this is the main validity fix and not a cleanup.** Without it, a planted slot can be "unsolvable for an honest
prober": the key is right about dataset text but wrong about the texts an agent writes. The diagnosis would then count
style failures as interpretability failures, or the reverse. With the filter, every kept key holds across styles, so
a failure on a kept slot is not explained by "the latent ignores my writing style". The comparison of kept and
dropped slots (below) checks that the filter does not just keep the easy latents.

## Rules and options (defaults follow the orchestrator spec)

| Option | Default | Meaning |
|---|---|---|
| `--thr` | 0.85 | robustness threshold |
| `--robust-set` | `CF2` | metric for robustness and rivals: `CF2` = split C + F2 (spec); `F2` = styled texts only; `both` = the smaller of the two |
| `--rival-rule` | `spec` | `spec`: planted slots need agreement + robustness only. `symmetric`: also drop planted slots where a distractor reaches the threshold (the test nulls get) |
| `--null-rule` | `symmetric` | `symmetric`: a null latent must pass the same latent test as a planted one (agreement over menu + anchor, robust anchor) and have no menu rival. `menu-only`: only the rival check |

`symmetric` is the default null rule because PREREG step 2 applies the filter "to the latent and its anchor, which is
the same for planted and null slots". If only planted latents had to be style-robust, null latents would become
statistically different from planted ones (they would include the style-fragile ones). That is the fingerprint SPEC
forbids. `summary.json` reports kept fractions under every combination of `robust-set` and `rival-rule`.

**A caveat the orchestrator should look at before relying on `CF2`.** Per concept, C + F2 holds 20 dataset texts and
only 10 styled texts. A latent that fires on every dataset text of c\* and on none of its styled texts already gets
AUROC (20 + 10 x 0.5) / 30 = 0.83. If it fires on **one** styled text out of 10, it gets 0.85 and passes. In the same
way, A (40 texts) dominates F1 (10 texts), so the multi-style key almost always agrees with the original one. Both
checks are therefore lenient about style. The styled-only metric (`--robust-set F2`, held out from both key
definitions, as in PREREG's original `auroc_F`) is the stricter test of the same idea. `summary.json` shows how many
kept slots have `auroc_F2 < 0.75` or fire on fewer than half of c\*'s bank-F texts, plus kept fractions under
`F2` and `both`. My recommendation is to decide between `CF2` and `both` from those numbers, before writing the pool.
This is a choice of threshold, not a result, so it can be made (and logged as a PREREG deviation) before any agent
runs.

## Outputs

`compute` (GPU, gitignored except the manifest):
- `diagnosis/cache/bank_acts_L{6,12,18}_{F,R}.npy`: float16 [4640, 16384]. One row per bank text, in the generator's
  concept order (232 concepts x 20 texts), all 16384 latents. About 0.9 GB in total.
- `diagnosis/cache/bank_meta.json`: row -> concept, style, text hashes, and the consistency check.
- `diagnosis/cache/manifest.json`, copied to `diagnosis/style_filter_cache_manifest.json` (committed): shapes, dtypes,
  sha256, git sha, SAE snapshot ids, and the consistency check.

The per-text summary is exactly the generator's: `fm_core.Subject.maxpool_acts` (max over tokens, BOS excluded, texts
truncated to 64 tokens). Bank texts are whitespace-normalised and cut to 400 characters, the same way `concepts.py`
prepared the dataset texts. Batches hold 32 texts in concept order, as in `precompute.py`. **Consistency check**: the
job also re-runs split A's first batch (precompute's batch 0, 32 texts) and compares it with `acts_L*_A.npy`. On the
same GPU and dtype it should match exactly. In the CPU dry-run (2 texts, different device) it matched closely but not
exactly: Pearson r 0.9994, Jaccard of the active sets 0.984, median relative difference on co-active entries 0.7%.

`analyze` (CPU) writes to `diagnosis/style_filter_out/`:
- `slots.csv` / `slots.json`: one row per v2 slot (703 slots in 180 instances). Columns: original key, multi-style key,
  agreement, AUROC of the anchor on A / C / F / F1 / F2 / A+F1 / C+F2, the best rival option and its metric, margins
  (A, C, C+F2, F2), the anchor's fire rate on A and F, generic fire rate (density) on A and on bank F, keep, drop
  reasons, keep under every alternative rule, and offline reference accuracy.
- `pool_latents.csv`: the same test for every latent in generator v2's pool (5017 latents), with agreement over the
  layer's answerable concepts. This gives survival by layer and family (PREREG P3/P4).
- `summary.json`: counts, kept fractions by kind, layer, family and tier; reasons; rule sensitivity; the survivor vs
  dropped comparison (fire rate on generic text, layer, margin, density, fire on c\*, with Mann-Whitney p-values) for
  slots and pool latents; the PREREG "the filter selects easier latents" verdict (median margin_C difference > 0.05,
  density ratio > 1.25, offline-reference planted accuracy difference > 0.05); integrity checks (split-A AUROC
  recomputed vs the generator's cached table; instance `menu_auroc_A` vs recomputed).
- `filtered_instances.json`: kept and failed slots per instance, and the list of instances whose slots all survive.
- `diagnosis/cache/pool_auroc.npz`: AUROC rows of every pooled latent on A+F1, C+F2 and F2, used by `write_v2f.py`.

`write_v2f.py` writes `tasks/featurematch/instances_v2f/<id>/{instance.json,public.json}` (gitignored),
`diagnosis/instances_v2f_manifest.json` (committed; per-instance file sha256, source instance, redrawn slots) and
`style_filter_out/v2f_summary.json`.

## How failing slots are handled (`write_v2f.py --mode`)

- **`redraw`** (default). Kept slots are copied unchanged. Each failing slot is replaced at the same position by a
  slot of the **same kind**, drawn by generator v2 itself (`generate.make_slot`: layer, then anchor concept uniformly,
  then a pooled latent, then the menu with the tier's closeness). The draw is kept only if it passes the same filter
  with the same options, and otherwise drawn again. A kept original is "a v2 draw that passed"; a redrawn slot is
  "v2 draws until one passes": the same distribution, so kept and redrawn slots cannot be told apart. Every episode
  keeps its slot count and null pattern, so the slot-count distribution and the null fraction equal v2's exactly.
- **`drop`**. Failing slots are removed, and instances left with fewer than `--min-slots` (3) slots are dropped. Nulls
  fail more often than planted slots (they have one more test), so this lowers the null fraction and the slot counts.
  It is only for comparison.
- **`regenerate`**. Fresh instances as PREREG step 2 words it: generator v2's own episode draw (seeds 8000+, 108000+,
  208000+; 60 per tier) with every slot drawn as in `redraw`. Use this if the plan wants a pool that shares no
  episodes with v2.

Every written instance, modified or not, gets a new permutation seed. Its id, canary and leak string are then
recomputed exactly as `generate.make_instance` computes them (`fm-<tier>-<10 hex>`, `RLCANARY-featurematch-<16 hex>`).
Ids therefore reveal neither the source instance nor whether a slot was redrawn. That information is kept only in the
privileged `instance.json` (`extra.style_filter`) and in the manifest. Each written instance is verified: the public
latent index maps to the real latent through the permutation, the planted answer points to the anchor's label, null
menus exclude the anchor, the canary matches the id, and the canary does not appear in `public.json`.
`v2f_summary.json` also reports a one-feature fingerprint check: the AUROC of generic fire rate for null vs planted
slots, in v2 and in v2f.

## Usage

From `~/wt/fmdiag`, with `source ~/ideating-rl-tests/common/env.sh` (sets `$PY`, `HF_HOME`, `PYTHONPATH`). The v2
caches and instances are read from `~/wt/featurematch/tasks/featurematch/{cache,instances}` (read-only). Override them
with `FM_SRC_CACHE` / `FM_SRC_INSTANCES` or `--src-cache` / `--src-instances`.

```bash
# 1. GPU: one job, <= 9 GB, about 290 batches of 32 texts. Checks free disk (~1 GB) before it takes the GPU.
$PY -m common.gpuq run --gb 9 --heavy --label featurematch-diag-banks -- \
    $PY -m tasks.featurematch.diagnosis.style_filter compute
#    CPU dry-run (2 concepts, layer 12, ~9 min on CPU, output in diagnosis/cache/dryrun/):
CUDA_VISIBLE_DEVICES= $PY -m tasks.featurematch.diagnosis.style_filter compute --dry-run

# 2. CPU: verdicts + comparison (defaults = spec; try --robust-set both / F2 for the sensitivity)
$PY -m tasks.featurematch.diagnosis.style_filter analyze

# 3. CPU: write the filtered pool (redraw | drop | regenerate)
$PY -m tasks.featurematch.diagnosis.write_v2f --mode redraw --clean

# tests (CPU, ~15 s, synthetic world, ~170 MB temp files deleted afterwards)
$PY -m pytest -q tasks/featurematch/diagnosis/tests/test_style_filter.py
```

The shared model service (`MODEL_SERVICE.md`) is not used. Its calls are capped at 16 texts and return selected
latents, while this job needs all 16384 latents for 9280 texts. One dedicated job is simpler and holds the GPU for a
few minutes.

After writing the pool, re-run the no-fingerprint check on it (PREREG P6) before any baseline or agent uses it.

## Tests (CPU)

`tests/test_style_filter.py` builds a synthetic world (`tests/synth_world.py`): the real width (16384) and the real 3
layers, 24 concepts, the real bank file format, v2 instances made by the real generator, and the generator's own
AUROC tables. Each concept gets known latent types: robust (must be kept), style-fragile (fires only on dataset text
and the encyclopedia-style bank text; must be dropped), rival (robust, and also separates a partner concept on C + F2
but not on A/B, so it can sit on the menu; a null slot with that partner on the menu must be flagged), name-firing
(removed by the generator's name filter) and weak (never pooled). The tests check:
- compute: shapes, dtype, batch size <= 32, concept order, text normalisation, manifest hashes, the consistency
  check, the dry-run subset, and that analyze refuses a partial bank;
- the F1/F2 split (one text per style per concept, deterministic);
- `judge` on hand-made AUROC rows: key change, not robust, F2 and both metrics, planted rival under spec vs
  symmetric, null rival, null rules, NaN rows;
- analyze end to end: every slot's verdict matches its latent type, with every branch exercised; integrity checks;
  pool verdicts; the filtered list; the comparison; the offline reference;
- write in redraw, drop and regenerate modes: slot counts and null pattern kept, only robust latents remain, every
  slot passes the filter again, kept slots are copied unchanged, ids are new and well-formed, the grader passes the
  answer key, the manifest hashes match, and the output is deterministic.
