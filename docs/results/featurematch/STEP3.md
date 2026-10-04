# FeatureMatch diagnosis: step-2 verification and step-3 outcome (2026-10-02)

This report brings together four independent checks of step 2, the step-3 baselines (`step3/BASELINES.md`) and the
adversarial re-check of step 3 (`step3_skeptic/`). It applies the pre-registered rules in `PREREG.md` (sections 2-4
and Amendments 1-3; where they disagree, the amendments win). It adds no new experiment. Every number below was read
from the committed result files listed in section 7, not copied from summaries. Where a summary and its file disagreed,
the file's value is used and the difference is noted.

**Words used.** A *slot* is one question. It names one SAE latent (one "feature" inside gemma-2-2b) and offers 20
candidate concepts. On a *planted* slot the right concept is among the options. On a *null* slot it is not, and the
right answer is "nothing found". *Planted accuracy* is the share of planted slots answered with the right option. An
*episode* (3-5 slots) *passes* only if every slot is right. *Topic* slots have an "article about X" concept; *language*
slots have a "text written in X" concept. *Wilson* intervals treat slots as independent. *Clustered* intervals
resample whole instances (10,000 resamples, `random.Random(20261002)`), so they allow for slots of one episode being
related. All intervals are 95%.

## 1. Bottom line

- **The stop rule fires, clearly:** a no-reasoning script that runs 6 styled texts per option and claims the highest mean activation (SR-max) gets 385/441 = **0.873** of planted slots (Wilson [0.839, 0.901], clustered [0.840, 0.903]; bar > 0.50), not borderline, not text-selection-sensitive, and also on topic slots alone (0.829 [0.785, 0.866], n = 328).
- PREREG A1.5: **"FeatureMatch is mostly measuring style, not interpretability"**: pa(SR-max)/pa(ref) = 0.873 [0.840, 0.903] and pa(SR-thr)/pa(ref) = 0.730 [0.688, 0.771] (n = 441 planted slots).
- Verdict label **Not interpretability**: step 4 (the agent arms) does not run; the pool's failed menu-fingerprint check (P6 0.616) would block agent runs on its own (A3.1).
- In plain terms: once each answer key is valid across writing styles, a slot can be solved by trying a few varied texts per option and taking the strongest response, with no hypothesis about what the latent does.
- Step 2 holds up: an independent re-implementation matches it exactly (0/5017 differences, so nothing here is provisional), the bank texts are on-concept (800/800), and an in-budget script using only the task's own `generate` tool already gets 0.571 [0.525, 0.617].

## 2. Step 2 verified (PREREG A3.3 and related checks)

| Check | Passed? | Headline numbers | Files, commit |
|---|---|---|---|
| Independent re-implementation of the A1.1 keep rule | **passed** | 0 of 5017 latents differ from `key_check.jsonl` in c\*, k_ms, k_ho, kept or reasons. Kept 354/5017 = 0.071 [0.064, 0.078] | `verify_reimpl/`, 3eaccce1 |
| Bank-F texts on-concept (blind audit) | **passed** | 800/800 sampled texts clearly or loosely on-concept (Wilson [0.995, 1.000]); 0 off-concept | `verify_banks/`, 54574065, fc81eff9, 30da4211 |
| What dropped topic latents encode (mechanism) | **passed** (step 2's reading holds) | Dropped latents: format or register cue 18/30 = 0.60 [0.42, 0.75], against kept 0/15 [0, 0.20] | `verify_mechanism/`, 854c5066 |
| Menu fingerprint (P6) | **failed** | P6 reproduces at 0.616. A larger held-out test gives 0.683 [0.674, 0.693]. A label-only policy passes 0.164 [0.151, 0.178] of fresh episodes | `verify_fingerprint/`, e63f09a0 |

**The re-implementation matched, so no step-3 number is provisional** (A3.3).

### 2.1 Re-implementation of the keep rule: passed

- **How it was built.** The re-implementation was written from PREREG A1.1 alone, without reading `style_filter.py`'s
  analysis code. It computes AUROC with its own code, does the 0.85 threshold test on exact integer rank statistics,
  and rebuilds the F1/F2 split from the bank files by per-text hashes (4640/4640 rows matched).
- **Result.** It reproduces `key_check.jsonl` exactly: 0/5017 mismatches, the same kept set (354/5017 = 0.071 [0.064,
  0.078]), and the same 583/703 = 0.829 [0.800, 0.855] failing slots of the old v2 pool.
- **Rounding does not matter.** The result is the same whether split-A AUROCs come from the generator's float16 table
  or a float64 recomputation. Five latents have a float64 anchor that differs from the generator's (gaps 7.6e-5 to
  2.4e-4), and all five are dropped either way. No latent sits on the 0.85 threshold: the closest is 1.1e-4 away.
- **Kept by family.**
  - language 163/276 = 0.591 [0.532, 0.647]
  - topic 191/4741 = 0.040 [0.035, 0.046]
- **Kept by layer.**
  - L6 102/1413 = 0.072 [0.060, 0.087]
  - L12 78/1855 = 0.042 [0.034, 0.052]
  - L18 174/1749 = 0.099 [0.086, 0.114]
- **Limit.** The check reads the same cached activations, so an error made before the caches were built would be
  shared. It does not cover that.

### 2.2 Bank-F validity: passed

The worry (A3.3) was that only 4% of topic latents survive because the styled bank texts are off-topic, not because the
latents are fragile.

**The sample and the bar.** The audit drew 40 topic concepts (800 bank-F texts) in two groups:
- KEPT: 20 concepts that have a latent kept in step 2.
- DROPLOW: 20 concepts that have a dropped latent which rarely fires on bank F, and no kept latent.

Ratings were done blind to the group. The sample, protocol and bar were committed before any rating, and the ratings
were committed before unblinding. The bar was "clear or loose" >= 0.85 in each group, with DROPLOW no more than 0.10
below KEPT.

**Ratings.**
- Clear or loose: 400/400 in each group. Off-concept: 0/800.
- Strictly "clear": 0.884 (concept-clustered CI [0.845, 0.919]). DROPLOW minus KEPT: -0.038 [-0.113, 0.030], so no
  group difference.

**Why topic latents fail the style test.**
- The 410 dropped low-fire latents fire on only 0.103 [0.084, 0.138] of bank texts that are clearly on-concept and
  name an instance (n = 7122 latent-text pairs).
- The same latents fire on a median 0.925 of their concept's dataset texts.
- On the bank's one encyclopedia-like style they fire on 0.402 [0.274, 0.549] of texts.
- So they respond to the encyclopedia template, not to the subject.

**Model side.** A single latent chosen on half of bank F separates the held-out half at AUROC >= 0.85 for 37/40
concepts (0.925 [0.801, 0.974]), and separates dataset split A for 40/40. The verifier's prose says "38/40"; the file
(`results.json`, `per_concept_model_side`) gives 37/40, and 37 is used here. Only 3 of these 40 latents are in the v2
pool. Most fire on the concept's own name, which the generator's name filter rejects.

**Limits.**
- One rater (an LLM rating LLM-written text), and no inter-rater figure.
- The "clear" rate for DROPLOW is 0.865 [0.80, 0.918], so its lower bound is below 0.85.

### 2.3 Fragility mechanism: passed (step 2's reading holds)

**Sample.** 30 dropped and 15 kept topic latents, one per concept. One reader categorised each latent's peak tokens
across dataset and styled texts.

**What the latents respond to.**

| | Format or register cue | The topic itself |
|---|---|---|
| Dropped (n = 30) | 18/30 = 0.60 [0.42, 0.75] | 5/30 = 0.17 [0.07, 0.34] |
| Kept (n = 15) | 0/15 [0.00, 0.20] | 13/15 = 0.87 [0.62, 0.96] |
| Fisher test, dropped vs kept | p = 6e-5 | p = 9e-6 |

Examples of the format cues: birth-year digits in "(born 1982)", airport codes, elevation units.

**Firing rates on c\*'s texts** (these do not depend on the reader):

| | Dataset texts | Styled texts | AUROC_F2(c\*) |
|---|---|---|---|
| Dropped | 0.945 [0.928, 0.962] | 0.253 [0.185, 0.325] | 0.582 [0.545, 0.621] |
| Kept | 0.960 [0.932, 0.983] | 0.817 [0.760, 0.870] | 0.913 [0.894, 0.933] |

**The SAE is not short of style-robust topic latents.** All 201/201 topic concepts have one (AUROC_C and AUROC_F2 both
>= 0.85; [0.98, 1.00]). The generator's pool rules exclude most of them. Among robust latents whose best dataset
concept is the concept itself and that are not in the pool, the name filter is involved in 1321/1581 = 0.84
(recounted from `summary.json`). Among topic latents that pass the other pool rules, the share that is style-robust is:

| Pool cell | Robust share |
|---|---|
| Admitted cell (specific, silent on own name) | 226/4741 = 0.048 [0.042, 0.054] |
| Specific, fires on own name | 556/2261 = 0.246 [0.229, 0.264] |
| Not specific, fires on own name | 706/1532 = 0.461 [0.436, 0.486] |

**Checks.** A fresh GPU computation matches the cached activations (r = 0.99995, n = 1725).

**Limit.** The categories come from one unblinded reader. A scripted peak-token check agrees with them on only about
14/45 latents, but the reader-independent firing rates point the same way.

### 2.4 Menu fingerprint (P6): failed

The fingerprint is a pattern in the menus that a trained policy could use to tell planted from null slots without
looking at the latent.

**P6 as pre-registered.**
- Close tier: 0.616 (sd 0.014; 600 instances = 2415 slots per tier; seeds 500000+). The bar is 0.60.
- The T3 number is the T2 draw again, not a second measurement. The same seed gives identical T2 and T3 slots.
- 10 fresh blocks of 600 instances: mean 0.638, all 10 above 0.60.
- With more training data (5000 training and 3000 test instances): 0.683 [0.674, 0.693]. Unfiltered v2: 0.570
  [0.560, 0.581].

**The bigger problem is memorisation.** The filtered pool has only 354 latents, so they recur often, and a learner
that sees only option labels can recognise which latent a menu belongs to.
- Planted vs null from labels alone: 0.898 [0.893, 0.904].
- Picking the answer from labels alone: 0.577 [0.566, 0.589] of planted slots (n = 7142).
- A complete label-only policy passes 0.164 [0.151, 0.178] of fresh close-tier episodes (n = 3000). Always answering
  "nothing found" passes 0.031.
- On latents held out from training, answer-picking falls to chance: 0.057 [0.049, 0.065].
- This needs answer feedback from many training episodes. A fresh zero-shot agent cannot exploit it: zero-shot
  label rules score <= 0.516.

**A candidate fix (swap_nn), simulated only.**
- P6 protocol: 0.56-0.58 on three seed sets.
- Held-out test: 0.588 [0.578, 0.599], and 0.592 [0.582, 0.602] on fresh seeds. The upper bounds touch 0.60, so this
  is "met (point estimate), not robust".
- It leaves the memorisation in place: label-only policy pass 0.142.

## 3. Step-3 baselines on the filtered pool

**What was run.**
- Pool: `instances_v2f`, 180 instances, 725 slots (441 planted, 284 null), pool sha256 `b212728f...`. The manifest
  check found 0 mismatches.
- Every baseline is a fixed script. The reference, black-box and recipe baselines ran in one in-process GPU job
  through the task's own solver code and tool environment (`step3/runs/20261002-041021_inproc`). They did not go
  through the episode harness.
- SR ran offline from cached bank-R activations. The skeptic then re-ran it live through the tool's own code path:
  386/441 vs 385/441 offline, 712/725 identical picks. So the offline numbers stand for a live run.

### 3.1 All tiers pooled (n = 180 episodes, 441 planted, 284 null)

| Baseline | Planted acc | Wilson 95% | Clustered 95% | Pass | Pass Wilson 95% | Planted NF | Null false claim |
|---|---|---|---|---|---|---|---|
| reference (one-shot) | 441/441 = 1.000 | [0.991, 1.000] | [1.000, 1.000] | 178/180 = 0.989 | [0.960, 0.997] | 0.000 | 2/284 = 0.007 |
| reference (best-of-5) | 441/441 = 1.000 | [0.991, 1.000] | [1.000, 1.000] | 180/180 = 1.000 | [0.979, 1.000] | 0.000 | 0/284 = 0.000 |
| **SR-max** | **385/441 = 0.873** | [0.839, 0.901] | [0.840, 0.903] | 17/180 = 0.094 | [0.060, 0.146] | 0.000 | 284/284 = 1.000 |
| **SR-thr** | **322/441 = 0.730** | [0.687, 0.769] | [0.688, 0.771] | 68/180 = 0.378 | [0.310, 0.451] | 0.229 | 37/284 = 0.130 |
| SR-max-capped (A2.3) | 385/441 = 0.873 | [0.839, 0.901] | [0.840, 0.903] | 17/180 = 0.094 | [0.060, 0.146] | 0.000 | 284/284 = 1.000 |
| SR-thr-capped (A2.3) | 323/441 = 0.732 | [0.689, 0.772] | [0.691, 0.772] | 68/180 = 0.378 | [0.310, 0.451] | 0.225 | 39/284 = 0.137 |
| SR-max-all20 (over budget) | 425/441 = 0.964 | [0.942, 0.978] | [0.945, 0.980] | 28/180 = 0.156 | [0.110, 0.216] | 0.000 | 284/284 = 1.000 |
| SR-thr-all20 (over budget) | 367/441 = 0.832 | [0.794, 0.864] | [0.798, 0.866] | 104/180 = 0.578 | [0.505, 0.648] | 0.163 | 21/284 = 0.074 |
| black-box control | 0/441 = 0.000 | [0.000, 0.009] | [0.000, 0.000] | 7/180 = 0.039 | [0.019, 0.078] | 1.000 | 0/284 = 0.000 |
| self_probe | 90/441 = 0.204 | [0.169, 0.244] | [0.168, 0.241] | 8/180 = 0.044 | [0.023, 0.085] | 0.746 | 16/284 = 0.056 |
| template_probe | 127/441 = 0.288 | [0.248, 0.332] | [0.246, 0.330] | 21/180 = 0.117 | [0.078, 0.172] | 0.678 | 20/284 = 0.070 |
| nothing | 0/441 = 0.000 | [0.000, 0.009] | [0.000, 0.000] | 7/180 = 0.039 | [0.019, 0.078] | 1.000 | 0/284 = 0.000 |
| always_claim | 13/441 = 0.029 | [0.017, 0.050] | [0.014, 0.047] | 0/180 = 0.000 | [0.000, 0.021] | 0.000 | 284/284 = 1.000 |
| prior (prior.json, unfiltered v2) | 17/441 = 0.038 | [0.024, 0.061] | [0.021, 0.058] | 0/180 = 0.000 | [0.000, 0.021] | 0.000 | 284/284 = 1.000 |
| prior (prior_v2f.json, filtered draw) | 26/441 = 0.059 | [0.041, 0.085] | [0.038, 0.081] | 0/180 = 0.000 | [0.000, 0.021] | 0.000 | 284/284 = 1.000 |
| random | 6/441 = 0.014 | [0.006, 0.029] | [0.004, 0.025] | 1/180 = 0.006 | [0.001, 0.031] | 0.578 | 118/284 = 0.415 |
| name_probe | 13/441 = 0.029 | [0.017, 0.050] | [0.014, 0.047] | 0/180 = 0.000 | [0.000, 0.021] | 0.000 | 284/284 = 1.000 |
| name_probe_thr | 0/441 = 0.000 | [0.000, 0.009] | [0.000, 0.000] | 3/180 = 0.017 | [0.006, 0.048] | 0.927 | 27/284 = 0.095 |
| vocab_match | 23/441 = 0.052 | [0.035, 0.077] | [0.033, 0.073] | 4/180 = 0.022 | [0.009, 0.056] | 0.875 | 16/284 = 0.056 |

Notes on the table:
- prior_or_none gives exactly the same answers as prior with either prior file, so it is not listed separately.
- SR-max never answers "nothing found". That is why its pass rate is low even though its planted accuracy is high.

### 3.2 By tier: planted accuracy, Wilson (W) and clustered (C) 95%

The tiers are T1 (far menus, 1200 forward units), T2 (close menus, 1200) and T3 (close menus, 550).

| Baseline | T1 | T2 | T3 |
|---|---|---|---|
| reference (one-shot) | 153/153 = 1.000 W[0.976, 1.000] C[1.000, 1.000] | 149/149 = 1.000 W[0.975, 1.000] C[1.000, 1.000] | 139/139 = 1.000 W[0.973, 1.000] C[1.000, 1.000] |
| SR-max | 137/153 = 0.895 W[0.837, 0.935] C[0.845, 0.939] | 136/149 = 0.913 W[0.857, 0.948] C[0.862, 0.955] | 112/139 = 0.806 W[0.732, 0.863] C[0.737, 0.871] |
| SR-thr | 111/153 = 0.726 W[0.650, 0.790] C[0.657, 0.789] | 115/149 = 0.772 W[0.698, 0.832] C[0.697, 0.844] | 96/139 = 0.691 W[0.610, 0.761] C[0.612, 0.765] |
| black-box control | 0/153 = 0.000 W[0.000, 0.025] | 0/149 = 0.000 W[0.000, 0.025] | 0/139 = 0.000 W[0.000, 0.027] |
| self_probe | 33/153 = 0.216 W[0.158, 0.287] C[0.152, 0.286] | 34/149 = 0.228 W[0.168, 0.302] C[0.163, 0.295] | 23/139 = 0.166 W[0.113, 0.236] C[0.109, 0.222] |
| template_probe | 37/153 = 0.242 W[0.181, 0.316] C[0.182, 0.304] | 55/149 = 0.369 W[0.296, 0.449] C[0.293, 0.449] | 35/139 = 0.252 W[0.187, 0.330] C[0.179, 0.325] |

Pass rates by tier (n = 60 each, Wilson 95%):

| Baseline | T1 | T2 | T3 |
|---|---|---|---|
| reference | 60/60 [0.940, 1.000] | 60/60 [0.940, 1.000] | 58/60 [0.886, 0.991] |
| SR-max | 5/60 [0.036, 0.181] | 9/60 [0.081, 0.261] | 3/60 [0.017, 0.137] |
| SR-thr | 18/60 [0.199, 0.425] | 27/60 [0.331, 0.575] | 23/60 [0.271, 0.510] |
| black-box | 1/60 [0.003, 0.089] | 1/60 [0.003, 0.089] | 5/60 [0.036, 0.181] |
| self_probe | 4/60 [0.026, 0.159] | 1/60 [0.003, 0.089] | 3/60 [0.017, 0.137] |
| template_probe | 6/60 [0.047, 0.202] | 9/60 [0.081, 0.261] | 6/60 [0.047, 0.202] |

SR on topic slots by tier:

| Variant | T1 | T2 | T3 |
|---|---|---|---|
| SR-max | 104/120 = 0.867 W[0.794, 0.916] C[0.803, 0.921] | 90/103 = 0.874 W[0.796, 0.925] C[0.802, 0.937] | 78/105 = 0.743 W[0.652, 0.817] C[0.657, 0.825] |
| SR-thr | 78/120 = 0.650 W[0.561, 0.730] C[0.566, 0.728] | 69/103 = 0.670 W[0.574, 0.753] C[0.568, 0.767] | 62/105 = 0.591 W[0.495, 0.680] C[0.500, 0.677] |

On language slots, both SR variants get every planted slot right in every tier (33/33, 46/46, 34/34). T3 is the
hardest tier for SR on topic slots. Both point estimates stay above 0.50 there, but SR-thr's Wilson interval [0.495,
0.680] touches 0.50. The stop rule is judged on all tiers pooled, so this does not change it.

### 3.3 By family (A3.2): planted accuracy, all tiers

| Baseline | Topic (n = 328) | Wilson | Clustered | Language (n = 113) | Wilson | Clustered |
|---|---|---|---|---|---|---|
| reference (one-shot) | 328/328 = 1.000 | [0.988, 1.000] | [1.000, 1.000] | 113/113 = 1.000 | [0.967, 1.000] | [1.000, 1.000] |
| **SR-max** | **272/328 = 0.829** | [0.785, 0.866] | [0.787, 0.870] | 113/113 = 1.000 | [0.967, 1.000] | [1.000, 1.000] |
| **SR-thr** | **209/328 = 0.637** | [0.584, 0.687] | [0.585, 0.687] | 113/113 = 1.000 | [0.967, 1.000] | [1.000, 1.000] |
| SR-max-all20 | 315/328 = 0.960 | [0.933, 0.977] | [0.938, 0.980] | 110/113 = 0.974 | [0.925, 0.991] | [0.940, 1.000] |
| SR-thr-all20 | 257/328 = 0.783 | [0.736, 0.825] | [0.742, 0.824] | 110/113 = 0.974 | [0.925, 0.991] | [0.940, 1.000] |
| black-box control | 0/328 = 0.000 | [0.000, 0.012] | [0.000, 0.000] | 0/113 = 0.000 | [0.000, 0.033] | [0.000, 0.000] |
| self_probe | 73/328 = 0.223 | [0.181, 0.271] | [0.180, 0.267] | 17/113 = 0.150 | [0.096, 0.228] | [0.086, 0.223] |
| template_probe | 20/328 = 0.061 | [0.040, 0.092] | [0.033, 0.092] | 107/113 = 0.947 | [0.889, 0.975] | [0.897, 0.990] |
| always_claim | 11/328 = 0.034 | [0.019, 0.059] | [0.015, 0.056] | 2/113 = 0.018 | [0.005, 0.062] | [0.000, 0.046] |
| prior (filtered draw) | 17/328 = 0.052 | [0.033, 0.081] | [0.029, 0.076] | 9/113 = 0.080 | [0.043, 0.144] | [0.035, 0.133] |
| random | 5/328 = 0.015 | [0.006, 0.035] | [0.003, 0.029] | 1/113 = 0.009 | [0.002, 0.048] | [0.000, 0.030] |
| name_probe | 11/328 = 0.034 | [0.019, 0.059] | [0.015, 0.056] | 2/113 = 0.018 | [0.005, 0.062] | [0.000, 0.046] |
| vocab_match | 23/328 = 0.070 | [0.047, 0.103] | [0.045, 0.098] | 0/113 = 0.000 | [0.000, 0.033] | [0.000, 0.000] |

The filtered pool has three times v2's share of language slots: 198/725 = 27.3%, against 65/703 = 9.2% in v2. A
script solves language slots almost perfectly. The topic column is the harder test, and SR passes it as well.

### 3.4 Gate (PREREG step 3)

The bar: every non-reference recipe except SR must have planted accuracy <= 0.15, and the reference must reach
>= 0.95. A result is "robust" when the Wilson interval lies on one side of the bar.

| Baseline | Planted acc [Wilson] | Gate | Robust |
|---|---|---|---|
| reference (one-shot) | 1.000 [0.991, 1.000] | **meets** (>= 0.95) | yes |
| self_probe | 0.204 [0.169, 0.244] | **fails** | yes. Also fails on topic slots alone: 0.223 [0.181, 0.271] |
| template_probe | 0.288 [0.248, 0.332] | **fails** | yes. Its decision rule gets 0.947 on language and 0.061 on topic |
| black-box, nothing, always_claim, both priors, random, name_probe, name_probe_thr, vocab_match | all <= 0.059 (largest: filtered prior 0.059 [0.041, 0.085]) | pass | yes |

**Gate verdict: not met.** Two pre-registered recipes fail it. The skeptic then ran new fixed recipes on the same
pool, all in-process and with no reasoning. Several fail the gate by wide margins (`step3_skeptic/skeptic_summary.json`):

| Fixed recipe (claims the highest-activating option) | Planted acc, pooled (n = 441) [Wilson] | Topic (n = 328) | Language (n = 113) | Within budget? |
|---|---|---|---|---|
| gemma writes 1 text per option via the task's `generate` tool (`gen1_cap`) | **252/441 = 0.571 [0.525, 0.617]** | 177/328 = 0.540 [0.486, 0.593] | 75/113 = 0.664 | yes, all caps |
| template_probe's own 4 encyclopedia templates, top pick (`encyc4_max`) | 238/441 = 0.540 [0.493, 0.586] | 125/328 = 0.381 [0.330, 0.435] | 113/113 = 1.000 | yes |
| gemma writes 2 texts per option (`gen_all`) | 297/441 = 0.673 [0.628, 0.716] | 217/328 = 0.662 [0.609, 0.711] | 80/113 = 0.708 | no: over the generate cap in 180/180 episodes |
| 3-style template set (`style3`) | 161/441 = 0.365 [0.322, 0.411] | 48/328 = 0.146 [0.112, 0.189] | 113/113 = 1.000 | yes |
| one fixed sentence with the option label | 16/441 = 0.036 [0.023, 0.058] | 0.037 | 0.035 | yes |

Notes on these recipes:
- **They work on the unfiltered pool too.** On unfiltered v2, with thresholds calibrated on a separate pool,
  `gen1_cap` gets 190/431 = 0.441 [0.395, 0.488], `gen_all` 0.585 [0.538, 0.630] and `encyc4_max` 0.378 [0.334,
  0.425]. So the v2 gate was probably never met either. Earlier gate passes relied on recipes with a threshold or a
  shortlist.
- **Name-based recipes score near 0 by construction.** The generator drops every latent that fires on its own
  concept's name, so the true option's name fires on 0/441 planted slots. name_probe answers the same as always_claim
  on 724/725 slots.

### 3.5 Stop rule (PREREG section 4, Amendments 2 and 3)

| Quantity | SR-max | SR-thr |
|---|---|---|
| Planted accuracy, all tiers pooled, primary text selection (A2.1), n = 441 | **0.873** W[0.839, 0.901] C[0.840, 0.903] | **0.730** W[0.687, 0.769] C[0.688, 0.771] |
| Above 0.50? Does the Wilson CI include 0.50? | yes / no | yes / no |
| A2.2 spread over 42 text selections, pooled (min / p10 / median / p90 / max) | 0.862 / 0.878 / 0.907 / 0.932 / 0.955 | 0.689 / 0.724 / 0.757 / 0.818 / 0.873 |
| A2.2 spread, topic slots only | 0.829 / 0.860 / 0.889 / 0.923 / 0.942 | 0.619 / 0.650 / 0.694 / 0.771 / 0.832 |
| A2.2 spread, language slots only | 0.920 / 0.929 / 0.947 / 1.000 / 1.000 | 0.894 / 0.912 / 0.938 / 1.000 / 1.000 |
| A2.2 all 20 bank-R texts per option (over budget, descriptive only) | 0.964 | 0.832 |
| A2.3 capped variant, pooled | 0.873 (unchanged) | 0.732 [0.689, 0.772] |
| A3.2 topic slots only, n = 328 | **0.829** W[0.785, 0.866] C[0.787, 0.870] | **0.637** W[0.584, 0.687] C[0.585, 0.687] |
| A3.2 language slots only, n = 113 | 1.000 W[0.967, 1.000] | 1.000 W[0.967, 1.000] |
| Live re-run through the tool path (skeptic) | 386/441 = 0.875 [0.841, 0.903] | 322/441 = 0.730 |

Labels (all from section 4 and Amendments 2 and 3):
- **fires: yes.** Both variants are above 0.50 with n_planted = 441 >= 100.
- **borderline: no.** Neither firing variant's Wilson CI includes 0.50.
- **text-selection-sensitive: no** (A2.2). The p10 is above 0.50 for both variants, pooled and in each family. The
  primary selection is at the low end of the spread: for SR-max on topic slots it is the minimum, 0.829.
- **A2.3 (T3 budget).** Uncapped SR goes over the T3 cap of 550 in 22 of 60 T3 episodes. The capped variant changes
  pooled planted accuracy by at most 0.002. On T3, SR-thr moves from 0.691 to 0.698 and SR-max is unchanged.
- **A2.4.** SR-thr's AUROC uses the other options' bank-R texts as negatives, while the reference uses a background
  sample. So the same 0.78 threshold is applied to slightly different quantities. This is noted only.
- **A2.5.** Clustered CIs are reported next to every Wilson CI above. As pre-registered, "borderline" is judged on
  the Wilson CI.
- **A3.1 (P6).** The filtered pool failed P6 at 0.616 in the close tiers. This cannot have raised a scripted recipe's
  accuracy: SR never looks at menu statistics, and the fingerprint can only be exploited by a policy trained on many
  episodes.
- **A3.2 (topic-only sentence).** The stop rule fires on the pooled number (SR-max 0.873 [0.839, 0.901], SR-thr 0.730
  [0.687, 0.769], n = 441) and also on topic slots alone (SR-max 0.829 [0.785, 0.866], SR-thr 0.637 [0.584, 0.687],
  n = 328). The "fires pooled but not on topic slots" case does not arise.
- **A3.3.** Step 2 was reproduced exactly (section 2.1), so these numbers are final, not provisional.
- **A1.5, stop-rule case.**
  - pa(SR-max)/pa(ref) = 0.873 / 1.000 = **0.873**, paired instance bootstrap [0.840, 0.903].
  - pa(SR-thr)/pa(ref) = **0.730** [0.688, 0.771].
  - Both are > 0.50 and neither CI includes 0.50, so the binding sentence applies without "(borderline)":
    **"FeatureMatch is mostly measuring style, not interpretability"**, with pa(SR-max)/pa(ref) = 0.873 [0.840,
    0.903] and pa(SR-thr)/pa(ref) = 0.730 [0.688, 0.771].
- **Verdict label (section 4): Not interpretability.**

The skeptic re-checked all of this from the raw files (`step3_skeptic/`, commit 0fa8c454):
- All 3,603 stored episodes re-grade identically, and 30/30 of a seeded sample re-graded through `grader.py` match.
- All 1,260 published fields reproduce.
- An independent SR re-implementation makes the same decision on 725/725 slots.
- Banks F and R share no identical text (median character 5-gram overlap 0.06).

The conclusion also does not depend on bank R. `gen1_cap`, which writes its probes with the task's own `generate`
tool and stays within budget, gets 0.571 [0.525, 0.617].

**Criterion-3 inputs** (recorded for the record; criterion 3 is not evaluated because step 4 does not run):
- r_ref = 0.000: the reference's planted "nothing found" rate, 0/441.
- r_rec = 0.712: the mean of self_probe's 0.746 (329/441) and template_probe's 0.678 (299/441).

## 4. Predictions vs outcomes

Rule used here, stricter than `BASELINES.md` and `STYLE_FILTER.md`:
- A point prediction **hits** if the predicted value lies inside the outcome's 95% CI.
- A range prediction hits if the point estimate is inside the range.
- A yes/no prediction hits if the outcome matches.
- Predictions with several parts are marked **partial** when some parts hit and some miss.

| # | Prediction (PREREG) | Outcome (n, 95% CI) | Result |
|---|---|---|---|
| P3 | Share surviving auroc_F >= 0.85 (all of bank F): 0.45 (0.30-0.60); languages 0.80, topics 0.35 | 430/5017 = 0.086 [0.078, 0.094]; languages 179/276 = 0.649 [0.591, 0.702]; topics 251/4741 = 0.053 [0.047, 0.060] | **miss** (far lower) |
| P4 | Survival L6 0.35 / L12 0.50 / L18 0.55; deeper layers more robust | auroc_F: 128/1413 = 0.091 [0.077, 0.107] / 97/1855 = 0.052 [0.043, 0.063] / 205/1749 = 0.117 [0.103, 0.133]. A1.1 kept: 0.072 / 0.042 / 0.099 | **miss** (far lower; L12 lowest, so not monotone) |
| P5 | "Filter selects easier latents": yes (p ~ 0.6); margin_C +0.03 to +0.06; higher fire on c\*; density about 1.1x | No. margin_C difference -0.040 (354 kept vs 4663 dropped latents); density ratio 0.70; reference planted accuracy +0.017 (0.986, n = 73, vs 0.969, n = 358). Fire on c\*'s bank-F texts 0.95 vs 0.15 | **partial**: verdict, margin and density miss; "higher fire on c\*" hits |
| P6 | Filtered pool passes the fingerprint check (CV AUROC <= 0.60) | 0.616 (sd 0.014, 2415 slots per tier); 10 fresh blocks 0.619-0.658; held out 0.683 [0.674, 0.693] | **miss** |
| P7 | Reference planted 0.97, one-shot pass 0.88 | 441/441 = 1.000 [0.991, 1.000]; 178/180 = 0.989 [0.960, 0.997] | **miss** (both higher; the gate is met) |
| P8 | Black-box planted 0.00, pass about 0.05 (all-null episodes only) | 0/441 = 0.000 [0.000, 0.009]; 7/180 = 0.039 [0.019, 0.078], exactly the 7 all-null episodes | **hit** |
| P9 | self_probe planted 0.30 (planted NF 0.55); gate fails | 90/441 = 0.204 [0.169, 0.244]; NF 329/441 = 0.746 [0.703, 0.784]; gate fails | **partial**: gate failure hits; both levels miss |
| P10 | template_probe planted 0.20 (planted NF 0.65); gate fails or borderline | 127/441 = 0.288 [0.248, 0.332]; NF 299/441 = 0.678 [0.633, 0.720]; gate fails, robustly | **partial**: gate failure and NF hit; planted level misses (higher, all from language slots) |
| P11 | SR-max planted 0.75 (0.55-0.85) | 385/441 = 0.873 [0.839, 0.901] | **miss** (above the range) |
| P12 | SR-thr planted 0.70 / null false claims 0.10 / pass 0.35 | 322/441 = 0.730 [0.687, 0.769]; 37/284 = 0.130 [0.096, 0.174]; 68/180 = 0.378 [0.310, 0.451] | **hit** (all three inside the CIs) |
| P13 | Stop rule fires (p about 0.7) | Fires; not borderline; not text-selection-sensitive; also on topic slots alone | **hit** |
| P31 | k_ms = c\*: 0.90; languages 0.98, topics 0.87 | 3834/5017 = 0.764 [0.752, 0.776]; languages 244/276 = 0.884 [0.841, 0.917]; topics 3590/4741 = 0.757 [0.745, 0.769] | **miss** |
| P32 | k_ho = c\*: 0.85 | 3606/5017 = 0.719 [0.706, 0.731] | **miss** |
| P33 | Filter pass 0.42 (0.25-0.60); the C part rarely fails | 408/5017 = 0.081 [0.074, 0.089]; the C part passes 0.954, the F2 part only 0.082 | **partial**: level misses; "C rarely fails" hits |
| P34 | Kept 0.38 (0.22-0.55); languages 0.75, topics 0.30 | 354/5017 = 0.071 [0.064, 0.078]; languages 163/276 = 0.591 [0.532, 0.647]; topics 191/4741 = 0.040 [0.035, 0.046] | **miss** |
| P35 | Dropped only for a key disagreement: <= 5% | 54/5017 = 0.011 [0.008, 0.014] | **hit** |
| P36 | About 60% of current v2 slots fail; planted and null within 5 points | 583/703 = 0.829 [0.800, 0.855]; planted 358/431 = 0.831 [0.792, 0.863], null 225/272 = 0.827 [0.778, 0.867] (0.4 points apart) | **partial**: level misses; parity hits |

**Tally.**
- P3-P13: 3 hits (P8, P12, P13), 3 partial (P5, P9, P10), 5 misses (P3, P4, P6, P7, P11).
- P31-P36: 1 hit (P35), 2 partial (P33, P36), 3 misses (P31, P32, P34).

The misses have a pattern. The predictions were far too optimistic about how many latent keys survive styled text
(P3, P4, P31-P34). They were also too pessimistic about how easily a script solves the pool that remains (P11).

**Other predictions that can now be checked:**
- P30, the overall verdict ("does not require interpretability", via the stop rule, p about 0.7): **hit**.
- P44 (part), pa(SR-max)/pa(ref) = 0.77: the outcome is 0.873 [0.840, 0.903], a **miss** (higher). Q_SR itself
  cannot be computed, because arm A does not exist.
- P46, the plain-language rule fires (p about 0.75): **hit**, through the stop-rule case.
- P1-P2 (step 1) and P14-P29 and P37-P43 (agent arms) are not evaluated: step 4 does not run.

**Corrections to earlier labels.** `BASELINES.md` labels P7 a hit and P9/P10 hits. Under the rule above, P7 is a miss
and P9/P10 are partial. This follows the skeptic's discrepancy 6.

## 5. What this means

### 5.1 For step 4

**Step 4 does not run.** Two separate pre-registered conditions each block it:
1. **The stop rule fired** (section 4: "Step 4 does not run").
2. **P6 is not fixed.** A3.1 allows agent arms only after a pre-registered menu fix brings the close-tier fingerprint
   to <= 0.60. The candidate fix swap_nn has not been pre-registered or applied. Its held-out upper bound reaches
   0.60, and it does not stop label-only memorisation.

**What the final reports must carry.** RESULTS.md and VERDICT.md, when written, must contain:
- the A1.5 sentence with its numbers (section 3.5);
- the A2.2 spread next to the primary SR numbers;
- the A3.2 topic-only sentence;
- the A3.1 note that the pool failed P6 at 0.616, which cannot have raised a scripted recipe's accuracy;
- the verdict label **Not interpretability**.

Criterion 3 is not evaluated. Its inputs are recorded in section 3.5.

### 5.2 For the task design

None of the following is pre-registered. These are findings and options for the orchestrator to decide on.

1. **Keys defined on one writing style are mostly not about the concept.**
   - 4609/5017 pooled latents (0.919) fail the style test.
   - Topic latents mostly encode how encyclopedia articles about a class are formatted (birth-year parentheses,
     airport codes, "represented the University of X during the season"), not the subject.
   - A dataset-class key therefore rewards guessing which corpus a format comes from. An agent that correctly finds
     "this latent fires on birth-year digits" is marked wrong. The A1.1 multi-style key is the right fix for key
     validity.
2. **Once keys are valid, the menu format is solvable without interpretability.**
   - With 20 named options, trying a few varied texts per option and taking the strongest response is enough: SR-max
     0.873, and `gen1_cap` 0.571 within budget using only the task's own tools.
   - This holds already on the unfiltered v2 pool: `gen1_cap` 0.441, `gen_all` 0.585.
   - So the problem is the multiple-choice format with probeable option names, not the filter. Judgement, not
     tested: small changes to menus, thresholds or budgets are unlikely to fix it. More texts make the recipe
     stronger (all 20 texts: 0.964 planted), and adding a threshold lets it pass whole episodes (SR-thr-all20 pass
     104/180 = 0.578 [0.505, 0.648]).
   - Options:
     - Ask for an open-ended description that is graded against held-out behaviour, instead of a pick from 20 menu
       items.
     - Use concepts that cannot be probed by writing about their name.
     - Score how efficiently the agent gathers evidence rather than the final pick.
     Each needs its own pre-registration.
3. **The generator's pool rules work against validity.**
   - The name filter and the "specific to at most 3 concepts" rule exclude most of the style-robust topic latents.
     The name filter is involved in 84% of the exclusions of anchor-matched robust latents.
   - What is left is the cell with the lowest robust share (0.048).
   - Relaxing the name filter would bring back latents an agent can identify by typing the option's name. The filter
     exists to block that word-matching shortcut, so there is a real trade-off: the style-robust topic latents are
     exactly the ones that respond to the concept's name.
4. **Language slots are trivial for scripts.**
   - SR 113/113; template_probe's rule 0.947 [0.889, 0.975]; `encyc4_max` 1.000.
   - The filter tripled their share (27.3% of slots). Any future pool should report and bound the language share, or
     drop language concepts.
5. **The small pool invites memorisation under RL.**
   - With 354 latents and 74 anchor concepts, a label-only policy with answer feedback passes 0.164 [0.151, 0.178] of
     fresh episodes.
   - If FeatureMatch is ever used for RL training, evaluation must use latents (and ideally anchor concepts) held out
     from training. On held-out latents, label-only answer-picking falls to 0.057.

## 6. Caveats and open items

- **Who rated.** The bank audit and the mechanism categories each had a single LLM rater. The bank result is robust to
  that (0/800 off-concept). The mechanism categories are supported by reader-independent firing rates, but a blind
  second rater would strengthen both.
- **How the baselines ran.** The reference, black-box and recipe baselines ran in-process, not through the episode
  harness. Caps were enforced, but there was no broker, leak scan or wall clock. The SR numbers were confirmed live.
- **Pre-registration timestamps.** The Amendment 2 and 3 headers say about 04:20 and 04:45 UTC, but their git commits
  are 03:53 and 04:04 UTC. Both still precede the step-3 runs (started 04:10), so the order of pre-registration holds.
  This is a documentation error only.
- **template_probe's topic score.** `BASELINES.md` says template_probe's gate failure "comes entirely from language
  slots". That is true of its thresholded decision rule. With a plain top-option pick, the same 4 templates get 0.381
  [0.330, 0.435] on topic slots, so the topic weakness comes from the threshold, not the texts.
- **The prior recipe.** A filtered-generator prior (`prior_v2f.json`) was added. It is not pre-registered and is
  labelled as such. It stays under the gate (0.059 [0.041, 0.085]).
- **One verifier figure.** One verifier's prose says "38/40"; its file says 37/40. The file value is used (section
  2.2).

## 7. Sources

Every number above comes from these files, read directly:

| Topic | File | Commit |
|---|---|---|
| Rule | `PREREG.md` (sections 2-4, A1-A3), `PLAN.md`, `STYLE_FILTER.md` (A1.1 results), `SR_RECIPE.md` | 2f663caf (A3) |
| Step 2 | `style_filter_out/summary.json`, `style_filter_out/v2f_summary.json`, `key_check.jsonl`, `style_filter_out/pool_latents.csv` (regenerable, not committed) | 3ea47d7f |
| Re-implementation | `verify_reimpl/result.json` | 3eaccce1 |
| Bank audit | `verify_banks/results.json`, `verify_banks/PROTOCOL.md` | 54574065, fc81eff9, 30da4211 |
| Mechanism | `verify_mechanism/summary.json`, `verify_mechanism/q1_categories.json` | 854c5066 |
| Fingerprint | `verify_fingerprint/reproduce.json`, `labelonly.json`, `policy.json`, `knnsplit.json`, `fix.json`, `fixextra.json` | e63f09a0 |
| Step 3 | `step3/BASELINES.md`, `step3/baselines_public.json`, `step3/tables_autogen.md`, `step3/sr_recipe_out/summary.json` | 75f7524b, 80fb525d |
| Skeptic | `step3_skeptic/skeptic_summary.json` | 0fa8c454 |

How this report was made: the tables in section 3 were generated by a script from `baselines_public.json`. The counts
and CIs for P3, P4 and P31-P36 were recomputed from `key_check.jsonl` and `pool_latents.csv` (CPU only, under
`systemd-run MemoryMax=8G`; peak RSS 34 MB). No GPU was used.
