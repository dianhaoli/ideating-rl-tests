# Step 2: the style-robustness filter (PREREG Amendment 1, A1.1)

Code: `style_filter.py` (compute + analyze), `write_v2f.py` (regenerates the filtered pool), tests in
`tests/test_style_filter.py`. Brief: `PLAN.md` STEP 2 and STEP 1b. The binding rule is `PREREG.md` **Amendment 1,
A1.1**. Where anything else disagrees with A1.1, A1.1 wins.

## What the filter does, in plain words

Each FeatureMatch slot says "this latent belongs to concept c\*". The generator decided that from **one kind of
text**: encyclopedia-style dataset sentences (split A). A latent can separate c\* perfectly on that kind of text and
still go quiet when c\* shows up in a chat message, a news report or a dialogue. An agent writes its own probe texts,
usually not in encyclopedia style. On such a latent it sees no signal, answers "nothing found" or guesses, and fails.
That failure says nothing about interpretability skill. It comes from how the answer key was made.

The filter checks every key again on styled text and keeps only the **latents** whose key survives. Then it builds a
new pool from those latents.

## The rule (A1.1, implemented exactly)

Text sets: dataset splits A and C, and bank F (232 concepts x 10 styles x 2 texts). Bank F is split per concept: for
each style, the **first** text listed in the bank-F file goes to F1 and the second to F2 (`split_f1`; `analyze`
checks bank_meta's row order and text hashes against the bank files first). AUROC_X(j, c) is `auroc.auroc_table`
within text set X: c's texts in X against the X texts of every other concept. AUROC_A is the generator's cached table,
so c\* is exactly the generator's anchor.

For each latent j of generator v2's pool (`Tables.planted`, 5017 latents):

| Quantity | Definition |
|---|---|
| c\* (original key) | argmax over all concepts of AUROC_A(j, c) |
| k_ms (multi-style key) | argmax over **all** concepts of M(c) = 0.5 AUROC_A(j, c) + 0.5 AUROC_F1(j, c). A and F1 get equal weight and are never pooled into one text set |
| k_ho (held-out check) | argmax over all concepts of H(c) = 0.5 AUROC_C(j, c) + 0.5 AUROC_F2(j, c), or H = AUROC_F2 for a concept without split C |
| style-robust | AUROC_F2(j, c\*) >= 0.85 **and** AUROC_C(j, c\*) >= 0.85. These are two separate thresholds, not one on pooled C + F2 |
| **keep** | k_ms = c\* and k_ho = c\* and style-robust |

The rule acts on latents. It is the same whether the latent later sits in a planted slot or a null slot, so null and
planted latents stay indistinguishable. A dropped latent gets one or more of the reasons `ms_key_differs`,
`heldout_key_differs` and `not_style_robust`.

**Regeneration.** `write_v2f.py` restricts generator v2's `Tables` to the kept latents and rebuilds what depends on
the pool exactly as `Tables.pools` does. That covers the concept-first anchor draw and the per-layer menu universe
("concepts that own a pooled latent at this layer"). Then it calls the unchanged `generate.make_instance`: 60
instances per tier, seeds 8000+ (T1), 108000+ (T2) and 208000+ (T3). The run checks four things. Every menu has 20
options. Every slot latent is kept. In every planted slot, the multi-style best option (argmax M over the menu) is
the answer, which is expected 0 violations; any violation makes the run fail. The public/privileged consistency check
from `verify` passes. The run also records two diagnostics, and no slot is dropped for them: the multi-style margin
per slot, and null slots where some option passes the robustness bar. The extra "null menu rival" check of the pre-A1.1
code is not in A1.1, so it survives only as that second diagnostic.

**One interpretation, stated:** "argmax over all concepts" uses all 232 concepts as candidates for k_ms and k_ho, not
only the menu or the answerable set. There were no exact ties (0 for M, 0 for H), so the tie rule (first index) never
mattered. A1.1 does not say whether AUROC_A means the generator's float16 cached table or a float32 recomputation. I
used the cached table, so c\* is the generator's anchor exactly. With the float32 table, 5 of 5017 latents would have
a different argmax (near-ties, differences < 2e-3).

## Outputs

- `diagnosis/key_check.jsonl` (committed, 3.0 MB, not gzipped): one row per pooled latent with layer, real latent
  id, c\*, k_ms, k_ho, the top 3 concepts under A, M and H with their scores, AUROC_C/F1/F2 of c\*, kept, and reasons.
- `diagnosis/slot_disagreements.jsonl` (committed): every slot of the **current** v2 pool (180 instances, 703 slots)
  whose latent fails the rule. Each row has the instance id, slot, kind, reasons, the answer, and the options that
  k_ms and k_ho would pick (`ms_pick`, `ho_pick`: an option number or "not on menu").
- `style_filter_out/summary.json` and `style_filter_out/v2f_summary.json` (committed): all the numbers below.
  `pool_latents.csv` and `slots.csv` are regenerable and not committed.
- `diagnosis/instances_v2f_manifest.json` (committed): per-instance sha256 of the regenerated pool. The instances are
  in `tasks/featurematch/instances_v2f/` (gitignored).
- `diagnosis/cache/pool_scores.npz` (gitignored): the M, H, AUROC_C and AUROC_F2 rows of every pooled latent.

## Usage

From `~/wt/fmdiag`, after `source ~/ideating-rl-tests/common/env.sh`. The v2 caches and instances are read
(never written) from `~/wt/featurematch/tasks/featurematch/{cache,instances}`. Always run under a memory cap. Peak RSS
was 1.05 GB for analyze (20 s) and 0.5 GB for regenerate plus fingerprint (17 s). analyze processes one layer at a
time, reads memory-mapped activations and only the pooled columns, in float32.

```bash
# GPU (already done; do not rerun): bank activations
$PY -m common.gpuq run --gb 9 --heavy --label featurematch-diag-banks -- $PY -m tasks.featurematch.diagnosis.style_filter compute
# CPU: A1.1 verdicts, logs, comparison (+ offline reference on the current v2 slots)
systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=2G -- $PY -m tasks.featurematch.diagnosis.style_filter analyze
# CPU: regenerate the filtered pool (only mode) + fingerprint check
systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=2G -- $PY -m tasks.featurematch.diagnosis.write_v2f --clean
# tests (CPU, ~15 s, synthetic world)
$PY -m pytest -q tasks/featurematch/diagnosis/tests/
```

## Tests (CPU)

`tests/synth_world.py` builds a synthetic world: real width and layers, 24 concepts, 40/8/10 texts in A/B/C, real
bank format, v2 instances from the real generator. It has latent types with known A1.1 outcomes:

| Type | Outcome | What it tests |
|---|---|---|
| R | kept | the baseline case |
| S (fires on dataset text and the encyclopedia-style bank text only) | `not_style_robust` | |
| X (fires on half of split C, all bank text) | `not_style_robust` | pooled C + F2 would pass |
| K (fires on c\*'s second text per style; the partner's first text and some of its A texts fire too) | `ms_key_differs` only | pooled A + F1 would keep it; only the first/second split makes it deterministic |
| P (also fires strongly on the partner's C and bank texts) | `heldout_key_differs` only | |

Unit tests use hand-made rows to cover equal weighting, H without split C, the boundary at 0.85 on each threshold,
and several reasons at once. Other tests cover the first/second split (also with interleaved file order), refusal of a
reordered bank, both logs (`ms_pick` and `ho_pick` included), and regenerate mode. The regenerate tests check the
seeds, that only kept latents are used, 20-option menus, the grader, determinism, manifest hashes, the planted
multi-style check, identity with `make_instance` on the restricted tables, and the fingerprint block.

---

## Results 2026-10-02 ~04:30 UTC (A1.1 code of this commit, run on a working tree based on 827191af; nothing run on the pool yet)

### Feasibility of regeneration and skew

Regeneration **succeeded**. It wrote 60 / 60 / 60 instances (T1 / T2 / T3) with no draw errors: 725 slots, null
fraction 0.392, 7 all-null episodes and planted multi-style key violations **0**. Every kept latent has at least 25
menu candidates after its confusable set is removed (min 25 / 25 / 48 at L6 / L12 / L18), so no 20-option menu was
short.

The pool is much narrower than v2, though:

| Layer | Kept latents (lang / topic) | Concepts with >= 1 kept latent (lang / topic) | v2 answerable concepts |
|---|---|---|---|
| L6 | 102 (44 / 58) of 1413 | 37 (13 / 24) | 108 |
| L12 | 78 (26 / 52) of 1855 | 34 (8 / 26) | 118 |
| L18 | 174 (93 / 81) of 1749 | 63 (17 / 46) | 149 |

- The layer mix stays balanced, because the generator draws the layer uniformly: 230 / 257 / 238 slots.
- **Language anchors make up 27.3% of the regenerated slots** (198 / 725), against 9.2% in v2 (65 / 703).
  Languages do not dominate, but their share is three times v2's.
- The regenerated pool uses 74 distinct anchor concepts (v2: 159) and 244 distinct latents.
- The menu universe per layer drops to 34-63 concepts (v2: 108-149). So "far" (T1) and "close" (T2/T3) menus draw on
  far fewer concepts.

### Fingerprint check (PREREG P6, rule: CV AUROC <= 0.6): **FAILS, narrowly, in the close tiers**

| Check (5-fold CV, gradient boosting, null vs planted) | T1 | T2 | T3 | all |
|---|---|---|---|---|
| `fingerprint_check.feats` (menu structure + layer), filtered generator in memory (seeds 500000+, 600 instances = 2415 slots per tier) | 0.499 | **0.616** (sd 0.014) | **0.616** | |
| same, the regenerated pool (about 240 slots per tier) | 0.485 | **0.612** (sd 0.086) | 0.586 | 0.556 |
| latent only (layer + generic fire rate), regenerated pool | 0.525 | 0.542 | 0.589 | 0.504 |
| density AUROC (generic fire rate), regenerated pool | 0.515 | 0.516 | 0.418 | 0.482 |
| **reference: unfiltered v2 generator, same check, in memory** | 0.482 | 0.531 | 0.531 | |

The filter-restricted generator crosses the bar in the close tiers: 0.616 against 0.531 for v2. No single feature is
strong: the univariate AUROCs on the filtered T2 are 0.39-0.55, and the most extreme is "row-mean closeness minus
median" at L12 (0.39). **Hypothesis, not tested:** with a menu universe of only 34-37 concepts at L6/L12, a close menu
covers most of the anchor's sibling group. Whether c\* is in or out then shows up in the menu's closeness profile. The
latent-only and density checks pass.

**Status: STOPPED here, as instructed.** The rule was not loosened, and no baseline or agent ran on `instances_v2f`.
The pool is written but should not be used until the orchestrator decides what to do about P6 and the narrow
universe.

### Counts (pooled latents, n = 5017)

| | all | lang (276) | topic (4741) | L6 | L12 | L18 |
|---|---|---|---|---|---|---|
| k_ms = c\* | 0.764 | 0.884 | 0.757 | 0.765 | 0.758 | 0.770 |
| k_ho = c\* | 0.719 | 0.815 | 0.713 | 0.704 | 0.723 | 0.726 |
| robust (F2 >= 0.85 and C >= 0.85) | 0.081 | 0.659 | 0.048 | 0.085 | 0.051 | 0.111 |
|   F2 part alone / C part alone | 0.082 / 0.954 | | | | | |
| AUROC_F (all of bank F) >= 0.85 (PREREG's original metric) | 0.086 | 0.649 | 0.053 | 0.091 | 0.052 | 0.117 |
| **kept** | **354 (0.071)** | 163 (0.591) | 191 (0.040) | 102 (0.072) | 78 (0.042) | 174 (0.099) |

Reasons (exact combinations): `not_style_robust` alone 2866; all three 780; held-out + robust 594; multi-style +
robust 369; `heldout_key_differs` alone 20; `ms_key_differs` + `heldout_key_differs` 17; `ms_key_differs` alone 17.
Counted with overlaps: `not_style_robust` 4609, `heldout_key_differs` 1411, `ms_key_differs` 1183. Only 54 latents
(1.1%) are dropped for a key disagreement alone, with the filter passed. The F2 threshold drives almost every drop.

**Current v2 pool (703 slots, 180 instances):** 583 slots (82.9%) fail. That is planted 358 / 431 (0.831) and null
225 / 272 (0.827). By layer, planted fails 0.864 / 0.831 / 0.799 and null fails 0.870 / 0.839 / 0.787 (L6 / L12 /
L18). Language slots fail much less often: planted 0.25, null 0.20, against topic 0.89 and 0.89. Tiers are similar
(0.82-0.84). No instance keeps all its slots. On planted slots, k_ms is not the answer in 95 slots and k_ho in 132.
In every one of those, the other key is "not on menu": it is a confusable concept the generator excluded from the
menu. On null slots, k_ms and k_ho are never on the menu. Argmax M over the menu is the answer on every planted v2
slot (0 exceptions).

### Survivors vs dropped ("the filter selects easier latents", PREREG step 2)

Primary unit: pooled latents (354 kept vs 4663 dropped). Medians, kept vs dropped:

| Feature | kept | dropped | Mann-Whitney p |
|---|---|---|---|
| generic fire rate on A (density) | 0.057 | 0.081 (ratio **0.70**) | 1e-5 |
| generic fire rate on bank F | 0.045 | 0.062 | 0.017 |
| fire rate on c\*'s A texts | 1.00 | 0.975 | 0.046 |
| fire rate on c\*'s bank-F texts | 0.95 | 0.15 | < 1e-5 |
| margin_A / margin_B / **margin_C** (vs the layer's answerable concepts) | 0.105 / 0.093 / **0.088** | 0.137 / 0.120 / **0.128** | < 0.003 |
| AUROC_A / B / C of c\* | 0.979 / 0.975 / 0.976 | 0.957 / 0.941 / 0.955 | < 1e-5 |
| layer mix (kept / dropped) | L6 102 / 1311, L12 78 / 1777, L18 174 / 1575 | | |
| family mix (kept / dropped) | lang 163 / 113, topic 191 / 4550 | | |

Offline reference solver (1 run per v2 instance, split-C probes) on current v2 slots: planted accuracy 0.986 for
slots with kept latents (n = 73) against 0.969 for dropped ones (n = 358); null accuracy 1.000 (47) against 0.978
(225).

**Verdict: "the filter selects easier latents" = NO.** None of the three pre-registered thresholds is met:

| Threshold | Value | Met? |
|---|---|---|
| median margin_C difference > 0.05 | −0.040 | no |
| generic fire ratio > 1.25 | 0.70 | no |
| reference planted accuracy difference > 0.05 | +0.017 | no |

The secondary check on v2 slots (margin over the menu) agrees: margin_C difference +0.039 (below 0.05), density
ratio 0.71, the same reference difference, so also "no". The survivors are sparser, not denser, and they have smaller
margins against the whole answerable set. They differ mainly in what the filter selects for: they fire on c\*'s
styled texts (0.95 vs 0.15).

### Predictions vs outcomes

| # | Prediction | Outcome | |
|---|---|---|---|
| P3 | auroc_F >= 0.85 share 0.45 (0.30-0.60); lang 0.80, topic 0.35 | 0.086; lang 0.649, topic 0.053 | miss (far lower) |
| P4 | survival L6 0.35 / L12 0.50 / L18 0.55 | kept 0.072 / 0.042 / 0.099 (auroc_F: 0.091 / 0.052 / 0.117) | miss; L18 highest, but L12 lowest, so deeper is not monotonically more robust |
| P5 | "filter selects easier": yes (p ~ 0.6); margin_C +0.03 to +0.06; higher fire on c\*; density about 1.1x | no; margin_C −0.040 (slots +0.039); fire on c\* higher (bank F 0.95 vs 0.15); density 0.70x | miss on the verdict, margin and density; hit on fire rate on c\* |
| P6 | fingerprint CV AUROC <= 0.6 | 0.616 in the close tiers (T1 0.50) | **miss (narrow)** |
| P31 | k_ms = c\* 0.90; lang 0.98, topic 0.87 | 0.764; lang 0.884, topic 0.757 | miss (lower) |
| P32 | k_ho = c\* 0.85 | 0.719 | miss (lower) |
| P33 | filter pass 0.42 (0.25-0.60); C part rarely fails | 0.081; C part fails 4.6%, F2 part fails 91.8% | miss on level; hit on "C rarely fails" |
| P34 | kept 0.38 (0.22-0.55); lang 0.75, topic 0.30 | 0.071; lang 0.591, topic 0.040 | miss (far lower) |
| P35 | dropped only for a key disagreement <= 5% | 1.1% (54) | hit |
| P36 | about 60% of current v2 slots fail; planted and null within 5 points | 82.9% (planted 83.1%, null 82.7%) | miss on level; hit on parity |

---

## SUPERSEDED (pre-A1.1 implementation, 2026-10-02 ~03:07 UTC, git 827191af): kept for the record only

The first implementation did not follow A1.1. It pooled A + F1 into one text set for the key, compared the key only
against the menu (plus the anchor for nulls), thresholded pooled C + F2 (default `CF2`), used a random F1/F2 split
(seed 20261002) and added a null "menu rival" drop rule. **Its numbers are superseded and must not be cited as step-2
results:**
- pooled latents kept 1249 / 5017 = 0.249 (L6 0.256, L12 0.211, L18 0.284; lang 0.75, topic 0.22). Robust on
  pooled C + F2: 0.261. Robust on F2 alone: 0.0865. auroc_F on all of bank F: 0.0857. Pooled-key agreement 0.949.
- current v2 slots kept: planted 184 / 431 = 0.427, null 123 / 272 = 0.452 (F2-only variant: planted 0.214).
- "filter selects easier": slot level yes (margin_C +0.066), pool level no (+0.002, density 0.80, reference +0.039).
- That code also had `redraw` and `drop` pool-writing modes, which have now been removed. The only mode is
  `regenerate`, as PREREG step 2 and A1.1 require.
