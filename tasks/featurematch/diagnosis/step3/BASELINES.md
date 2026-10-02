# FeatureMatch diagnosis, step 3: baselines on the filtered pool

Spec: `PREREG.md` step 3, section 4 (stop rule), Amendments 1-3 (amendments win). Pool: `tasks/featurematch/instances_v2f`
(180 instances, 725 slots: 441 planted, 284 null). Every baseline here is a **script**, not an LLM agent. Numbers
marked "in-process" come from one GPU job that ran the task's own solver code against the task's own tool
environment, with the model loaded once (see "How it was run"). The SR numbers are computed offline from cached
activations. All tables below are copied from `tables_autogen.md`, which `analyze_step3.py` writes. The machine-readable
version is `baselines_public.json` (aggregates only).

## Result in one paragraph

**The pre-registered stop rule fires.** A script that runs 6 styled example texts for every option through the
latent and picks the option with the highest mean activation (SR-max) gets **385 of 441 planted slots right: 0.873,
Wilson 95% [0.839, 0.901]**, instance-clustered bootstrap [0.840, 0.903]. Its thresholded version (SR-thr) gets **0.730
[0.687, 0.769]**. Both are far above the rule's 0.50 bar, so the outcome is not "borderline". It is not
"text-selection-sensitive" either: across 42 alternative choices of the 6 texts, SR-max ranges 0.862-0.955 and SR-thr
0.689-0.873. **The rule also fires on topic slots alone** (A3.2): SR-max 0.829 [0.785, 0.866] and SR-thr 0.637
[0.584, 0.687], n = 328 planted topic slots. On language slots both variants get all 113 planted slots right
(1.000 [0.967, 1.000]). Per PREREG section 4, step 4 (the agent arms) does not run, and the verdict label is
"Not interpretability". The reference solver passes its gate (planted accuracy 441/441 = 1.000 [0.991, 1.000]).
**Two other recipes fail their gate** (bar: planted accuracy <= 0.15). **self_probe** gets 0.204 [0.169, 0.244]
(n = 441), and it stays above the bar on topic slots alone (0.223 [0.181, 0.271], n = 328). **template_probe** gets
0.288 [0.248, 0.332]. Its failure comes entirely from language slots, where a fixed bank of 3 everyday sentences per
language gets 0.947 [0.889, 0.975]; on topic slots it gets 0.061 [0.040, 0.092]. Every other zero-effort recipe and
the black-box control pass the gate (planted accuracy <= 0.059). The gate failures do not change the stop-rule
outcome. They point the same way: cheap probing without any reasoning about the latent already solves a share of the
filtered pool.

Caveats and checks that apply to every number here. (1) The filtered pool failed the P6 fingerprint check narrowly
(menu-feature CV AUROC 0.616 in the close tiers, bar 0.60; `style_filter_out/v2f_summary.json`). That fingerprint is
a pattern in the menus that only a policy trained on menu statistics could exploit. None of the scripts here is
trained on menus (the prior recipes use only label frequencies from a disjoint draw), so it cannot have raised these
numbers (A3.1). A separate check of the fingerprint is in `verify_fingerprint/` (commit e63f09a0); I did not use it.
(2) A3.3 asks for an independent re-implementation of the step-2 keep rule before step-3 numbers are final. It
reproduced `key_check.jsonl` exactly (commit 3eaccce1, `verify_reimpl/result.json`), so these numbers are not
provisional on that count. (3) The blind audit of bank-F texts (`verify_banks/`, commit 30da4211) found 0 of 800
sampled texts off-concept (40 topic concepts). That speaks against the narrow filtered pool being an artefact of
off-concept bank texts (details in that audit).

## Words used here

- **Slot**: one question. It names one SAE latent and 20 options (concepts). A **planted** slot has the right concept
  among the options. A **null** slot does not, and the right answer is "nothing found".
- **Planted accuracy**: the share of planted slots answered with the right option. This is the number the gate and
  the stop rule use.
- **Pass**: an episode (3-5 slots) passes only if every slot is right.
- **Planted "nothing found" rate (planted NF)**: the share of planted slots where the script said "nothing found".
- **Null false-claim rate**: the share of null slots where the script claimed an option.
- **Wilson 95%**: a confidence interval for a proportion that treats slots (or episodes) as independent.
- **Clustered 95%**: a bootstrap interval that resamples whole instances (10,000 resamples, `random.Random(20261002)`),
  so it accounts for slots of one episode being related. When every slot is right (or every one wrong), it collapses
  to a single point, and the Wilson interval is the informative one.
- **Family**: whether the slot's true concept (its anchor) is a **topic** ("article about a galaxy") or a **language**
  ("text written in Thai"). 328 planted and 199 null slots are topic, 113 planted and 85 null are language.
  For pass rates by family, an episode enters the family's row if it has at least one slot of that family, and it
  counts as passed if all of its slots of that family are right (179 episodes have a topic slot, 131 a language slot).
  This family pass rate is descriptive only: an episode whose only language slots are null "passes" on language for
  any script that says "nothing found" there.

### The baselines

| Baseline | What it does | Source |
|---|---|---|
| reference | probes each option with held-out dataset texts (split C), claims the best option if its AUROC against background texts is >= 0.78, else "nothing found". One-shot = seed 7. Best-of-5 = seeds 7, 1007, ... until the first pass | `reference_solver.py`, in-process |
| SR-max / SR-thr | the style-robust recipe: 6 bank-R texts per option (one per style), claim the highest mean activation (SR-max), or claim it only if its AUROC against the other options' texts is >= 0.78 (SR-thr). `-capped`: stays within the tier's forward budget. `-all20`: all 20 bank-R texts per option (over budget, descriptive only) | `diagnosis/sr_recipe.py`, offline |
| black-box control | cannot see activations; asks the model itself which option a latent encodes | `blackbox_control.py`, in-process |
| template_probe | 4 fixed encyclopedia templates per topic option, 3 fixed everyday sentences per language option, AUROC against 24 fixed background texts, claim if >= 0.78 | `audit/attack_solvers.py`, in-process |
| self_probe | template_probe's top 3 options, then the model writes 2 probe texts per option (`generate`) and the decision is redone on them | `audit/attack_solvers.py`, in-process |
| nothing / always_claim | "nothing found" everywhere / option 1 everywhere | `recipe_baseline.py`, in-process |
| prior / prior_or_none | claim the option that was most often the answer in a disjoint generated pool (prior_or_none: "nothing found" if that frequency is at most the median). Run twice: with `prior.json` (unfiltered v2 generator) and with `prior_v2f.json` (a disjoint draw of the filtered generator, see below) | `recipe_baseline.py`, in-process |
| random | "nothing found" with probability 0.4, else a uniform option | `recipe_baseline.py`, in-process |
| name_probe / name_probe_thr | run each option's bare name through the latent and claim the highest (thr: "nothing found" if no name fires) | `recipe_baseline.py`, in-process |
| vocab_match | project the latent onto the vocabulary and claim the option whose words match the top tokens | `recipe_baseline.py`, in-process |

## Main table: all tiers, all slots (n = 180 episodes, 441 planted, 284 null)

| Baseline | Pass (Wilson 95%) | Planted acc | Wilson 95% | Clustered 95% | Planted NF (Wilson) | Null false claim (Wilson) | Fwd mean |
|---|---|---|---|---|---|---|---|
| reference (one-shot) | 178/180 = 0.989 [0.960, 0.997] | 441/441 = 1.000 | [0.991, 1.000] | [1.000, 1.000] | 0.000 [0.000, 0.009] | 2/284 = 0.007 [0.002, 0.025] | 719.4 |
| reference (best-of-5) | 180/180 = 1.000 [0.979, 1.000] | 441/441 = 1.000 | [0.991, 1.000] | [1.000, 1.000] | 0.000 [0.000, 0.009] | 0/284 = 0.000 [0.000, 0.013] | 719.4 |
| SR-max | 17/180 = 0.094 [0.060, 0.146] | 385/441 = 0.873 | [0.839, 0.901] | [0.840, 0.903] | 0.000 [0.000, 0.009] | 284/284 = 1.000 [0.987, 1.000] | 483.3 |
| SR-thr | 68/180 = 0.378 [0.310, 0.451] | 322/441 = 0.730 | [0.687, 0.769] | [0.688, 0.771] | 0.229 [0.192, 0.271] | 37/284 = 0.130 [0.096, 0.174] | 483.3 |
| SR-max-capped | 17/180 = 0.094 [0.060, 0.146] | 385/441 = 0.873 | [0.839, 0.901] | [0.840, 0.903] | 0.000 [0.000, 0.009] | 284/284 = 1.000 [0.987, 1.000] | 477.2 |
| SR-thr-capped | 68/180 = 0.378 [0.310, 0.451] | 323/441 = 0.732 | [0.689, 0.772] | [0.691, 0.772] | 0.225 [0.188, 0.266] | 39/284 = 0.137 [0.102, 0.182] | 477.2 |
| SR-max-all20 | 28/180 = 0.156 [0.110, 0.216] | 425/441 = 0.964 | [0.942, 0.978] | [0.945, 0.980] | 0.000 [0.000, 0.009] | 284/284 = 1.000 [0.987, 1.000] | 1611.1 |
| SR-thr-all20 | 104/180 = 0.578 [0.505, 0.648] | 367/441 = 0.832 | [0.794, 0.864] | [0.798, 0.866] | 0.163 [0.132, 0.201] | 21/284 = 0.074 [0.049, 0.110] | 1611.1 |
| black-box control | 7/180 = 0.039 [0.019, 0.078] | 0/441 = 0.000 | [0.000, 0.009] | [0.000, 0.000] | 1.000 [0.991, 1.000] | 0/284 = 0.000 [0.000, 0.013] | 4.0 |
| self_probe | 8/180 = 0.044 [0.023, 0.085] | 90/441 = 0.204 | [0.169, 0.244] | [0.168, 0.241] | 0.746 [0.703, 0.784] | 16/284 = 0.056 [0.035, 0.089] | 428.0 |
| template_probe | 21/180 = 0.117 [0.078, 0.172] | 127/441 = 0.288 | [0.248, 0.332] | [0.246, 0.330] | 0.678 [0.633, 0.720] | 20/284 = 0.070 [0.046, 0.106] | 403.8 |
| nothing | 7/180 = 0.039 [0.019, 0.078] | 0/441 = 0.000 | [0.000, 0.009] | [0.000, 0.000] | 1.000 [0.991, 1.000] | 0/284 = 0.000 [0.000, 0.013] | 0.0 |
| always_claim | 0/180 = 0.000 [0.000, 0.021] | 13/441 = 0.029 | [0.017, 0.050] | [0.014, 0.047] | 0.000 [0.000, 0.009] | 284/284 = 1.000 [0.987, 1.000] | 0.0 |
| prior [prior.json, unfiltered v2] | 0/180 = 0.000 [0.000, 0.021] | 17/441 = 0.038 | [0.024, 0.061] | [0.021, 0.058] | 0.000 [0.000, 0.009] | 284/284 = 1.000 [0.987, 1.000] | 0.0 |
| prior_or_none [prior.json, unfiltered v2] | 0/180 = 0.000 [0.000, 0.021] | 17/441 = 0.038 | [0.024, 0.061] | [0.021, 0.058] | 0.000 [0.000, 0.009] | 284/284 = 1.000 [0.987, 1.000] | 0.0 |
| prior [prior_v2f.json, filtered draw] | 0/180 = 0.000 [0.000, 0.021] | 26/441 = 0.059 | [0.041, 0.085] | [0.038, 0.081] | 0.000 [0.000, 0.009] | 284/284 = 1.000 [0.987, 1.000] | 0.0 |
| prior_or_none [prior_v2f.json, filtered draw] | 0/180 = 0.000 [0.000, 0.021] | 26/441 = 0.059 | [0.041, 0.085] | [0.038, 0.081] | 0.000 [0.000, 0.009] | 284/284 = 1.000 [0.987, 1.000] | 0.0 |
| random | 1/180 = 0.006 [0.001, 0.031] | 6/441 = 0.014 | [0.006, 0.029] | [0.004, 0.025] | 0.578 [0.532, 0.624] | 118/284 = 0.415 [0.360, 0.474] | 0.0 |
| name_probe | 0/180 = 0.000 [0.000, 0.021] | 13/441 = 0.029 | [0.017, 0.050] | [0.014, 0.047] | 0.000 [0.000, 0.009] | 284/284 = 1.000 [0.987, 1.000] | 80.6 |
| name_probe_thr | 3/180 = 0.017 [0.006, 0.048] | 0/441 = 0.000 | [0.000, 0.009] | [0.000, 0.000] | 0.927 [0.899, 0.948] | 27/284 = 0.095 [0.066, 0.135] | 80.6 |
| vocab_match | 4/180 = 0.022 [0.009, 0.056] | 23/441 = 0.052 | [0.035, 0.077] | [0.033, 0.073] | 0.875 [0.841, 0.903] | 16/284 = 0.056 [0.035, 0.089] | 4.0 |

"Fwd mean" is forward units per episode (the tier caps are 1200 / 1200 / 550 for T1 / T2 / T3). SR's forward units
are what a live run would cost. Uncapped SR exceeds the T3 cap (550) in 22 of 60 T3 episodes (A2.3). The capped
variants change pooled planted accuracy by at most 0.002 (T3 alone: SR-thr 0.691 to 0.698; SR-max unchanged).

## Gate check (PREREG step 3)

Rule: every non-reference recipe except SR has planted accuracy <= 0.15, and the reference has planted accuracy
>= 0.95. Point estimates decide. "Robust" means the Wilson 95% interval lies entirely on the same side of the bar.

| Baseline | Planted accuracy | Wilson 95% | Clustered 95% | n planted | Bar | Gate | Robust |
|---|---|---|---|---|---|---|---|
| reference (one-shot) | 1.000 | [0.991, 1.000] | [1.000, 1.000] | 441 | >= 0.95 | meets | yes |
| reference (best-of-5) | 1.000 | [0.991, 1.000] | [1.000, 1.000] | 441 | >= 0.95 | meets | yes |
| black-box control | 0.000 | [0.000, 0.009] | [0.000, 0.000] | 441 | <= 0.15 | passes | yes |
| self_probe | 0.204 | [0.169, 0.244] | [0.168, 0.241] | 441 | <= 0.15 | **fails** | yes |
| template_probe | 0.288 | [0.248, 0.332] | [0.246, 0.330] | 441 | <= 0.15 | **fails** | yes |
| nothing | 0.000 | [0.000, 0.009] | [0.000, 0.000] | 441 | <= 0.15 | passes | yes |
| always_claim | 0.029 | [0.017, 0.050] | [0.014, 0.047] | 441 | <= 0.15 | passes | yes |
| prior [prior.json, unfiltered v2] | 0.038 | [0.024, 0.061] | [0.021, 0.058] | 441 | <= 0.15 | passes | yes |
| prior_or_none [prior.json, unfiltered v2] | 0.038 | [0.024, 0.061] | [0.021, 0.058] | 441 | <= 0.15 | passes | yes |
| prior [prior_v2f.json, filtered draw] | 0.059 | [0.041, 0.085] | [0.038, 0.081] | 441 | <= 0.15 | passes | yes |
| prior_or_none [prior_v2f.json, filtered draw] | 0.059 | [0.041, 0.085] | [0.038, 0.081] | 441 | <= 0.15 | passes | yes |
| random | 0.014 | [0.006, 0.029] | [0.004, 0.025] | 441 | <= 0.15 | passes | yes |
| name_probe | 0.029 | [0.017, 0.050] | [0.014, 0.047] | 441 | <= 0.15 | passes | yes |
| name_probe_thr | 0.000 | [0.000, 0.009] | [0.000, 0.000] | 441 | <= 0.15 | passes | yes |
| vocab_match | 0.052 | [0.035, 0.077] | [0.033, 0.073] | 441 | <= 0.15 | passes | yes |

Gate verdict: reference meets its bar; recipes failing the gate: self_probe, template_probe.

## Stop rule (PREREG section 4 + Amendments 2 and 3)

| Quantity | SR-max | SR-thr |
|---|---|---|
| Planted accuracy, all tiers pooled (primary selection, A2.1), n = 441 | **0.873** [0.839, 0.901] / clustered [0.840, 0.903] | **0.730** [0.687, 0.769] / clustered [0.688, 0.771] |
| Above 0.50? Wilson CI includes 0.50? | yes / no | yes / no |
| Text-selection spread over 42 selections (A2.2): min / p10 / median / p90 / max | 0.862 / 0.878 / 0.907 / 0.932 / 0.955 | 0.689 / 0.724 / 0.757 / 0.818 / 0.873 |
| All 20 bank-R texts per option (over budget, descriptive) | 0.964 | 0.832 |
| **Topic slots only** (A3.2), n = 328 | **0.829** [0.785, 0.866] / clustered [0.787, 0.870] | **0.637** [0.584, 0.687] / clustered [0.585, 0.687] |
| Topic-only text-selection spread: min / p10 / median / p90 / max | 0.829 / 0.860 / 0.889 / 0.923 / 0.942 | 0.619 / 0.650 / 0.694 / 0.771 / 0.832 |
| **Language slots only**, n = 113 | **1.000** [0.967, 1.000] | **1.000** [0.967, 1.000] |
| Language-only text-selection spread: min / p10 / median / p90 / max | 0.920 / 0.929 / 0.947 / 1.000 / 1.000 | 0.894 / 0.912 / 0.938 / 1.000 / 1.000 |

**Stop-rule block:**
- `fires`: **yes** (both SR-max and SR-thr are above 0.50 with n_planted = 441 >= 100).
- `borderline`: **no** (neither firing variant's Wilson CI includes 0.50).
- `text-selection-sensitive`: **no** (p10 > 0.50 for both variants, pooled and in each family).
- Topic-only: the rule would also fire on topic slots alone (both variants above 0.50, n = 328, Wilson CIs above
  0.50). So A3.2's "fires pooled but not on topic slots" sentence does not apply.
- Labels: `["fires"]`; valid as the step-3 outcome (run on `instances_v2f`, pool sha256 below).
- P6: the filtered pool failed P6 at 0.616 (close tiers). This cannot have raised a scripted recipe's accuracy (A3.1).

**A1.5 in the stop-rule case** (the numbers RESULTS.md and VERDICT.md use for the plain-language rule): pa(SR-max) /
pa(reference) = 0.873 / 1.000 = **0.873**, paired instance bootstrap 95% [0.840, 0.903]; pa(SR-thr) / pa(reference) =
**0.730** [0.688, 0.771]. Both are above 0.50 and neither CI includes 0.50, so A1.5's sentence applies without
"(borderline)". (Writing that sentence into RESULTS.md / VERDICT.md is not part of this step.)

**Criterion-3 inputs measured here** (section 4, for the record; criterion 3 is not evaluated because step 4 does not
run): r_ref = reference planted NF = 0.000; r_rec = mean of self_probe and template_probe planted NF = 0.712
(self_probe 0.746, template_probe 0.678).

## By family (A3.2): planted accuracy and pass rate

### Topic slots (all tiers)

| Baseline | Pass (Wilson 95%) | Planted acc | Wilson 95% | Clustered 95% | Planted NF (Wilson) | Null false claim (Wilson) | Fwd mean |
|---|---|---|---|---|---|---|---|
| reference (one-shot) | 177/179 = 0.989 [0.960, 0.997] | 328/328 = 1.000 | [0.988, 1.000] | [1.000, 1.000] | 0.000 [0.000, 0.012] | 2/199 = 0.010 [0.003, 0.036] | 720.0 |
| reference (best-of-5) | 179/179 = 1.000 [0.979, 1.000] | 328/328 = 1.000 | [0.988, 1.000] | [1.000, 1.000] | 0.000 [0.000, 0.012] | 0/199 = 0.000 [0.000, 0.019] | 720.0 |
| SR-max | 28/179 = 0.156 [0.111, 0.217] | 272/328 = 0.829 | [0.785, 0.866] | [0.787, 0.870] | 0.000 [0.000, 0.012] | 199/199 = 1.000 [0.981, 1.000] | 484.0 |
| SR-thr | 73/179 = 0.408 [0.339, 0.481] | 209/328 = 0.637 | [0.584, 0.687] | [0.585, 0.687] | 0.308 [0.260, 0.360] | 23/199 = 0.116 [0.078, 0.168] | 484.0 |
| SR-max-capped | 28/179 = 0.156 [0.111, 0.217] | 272/328 = 0.829 | [0.785, 0.866] | [0.787, 0.870] | 0.000 [0.000, 0.012] | 199/199 = 1.000 [0.981, 1.000] | 477.9 |
| SR-thr-capped | 73/179 = 0.408 [0.339, 0.481] | 210/328 = 0.640 | [0.587, 0.690] | [0.589, 0.690] | 0.302 [0.255, 0.354] | 25/199 = 0.126 [0.087, 0.179] | 477.9 |
| SR-max-all20 | 47/179 = 0.263 [0.204, 0.332] | 315/328 = 0.960 | [0.933, 0.977] | [0.938, 0.980] | 0.000 [0.000, 0.012] | 199/199 = 1.000 [0.981, 1.000] | 1613.4 |
| SR-thr-all20 | 107/179 = 0.598 [0.525, 0.667] | 257/328 = 0.783 | [0.736, 0.825] | [0.742, 0.824] | 0.210 [0.170, 0.258] | 14/199 = 0.070 [0.042, 0.115] | 1613.4 |
| black-box control | 22/179 = 0.123 [0.083, 0.179] | 0/328 = 0.000 | [0.000, 0.012] | [0.000, 0.000] | 1.000 [0.988, 1.000] | 0/199 = 0.000 [0.000, 0.019] | 4.0 |
| self_probe | 29/179 = 0.162 [0.115, 0.223] | 73/328 = 0.223 | [0.181, 0.271] | [0.180, 0.267] | 0.713 [0.662, 0.760] | 11/199 = 0.055 [0.031, 0.096] | 428.6 |
| template_probe | 22/179 = 0.123 [0.083, 0.179] | 20/328 = 0.061 | [0.040, 0.092] | [0.033, 0.092] | 0.893 [0.855, 0.922] | 8/199 = 0.040 [0.021, 0.077] | 404.4 |
| nothing | 22/179 = 0.123 [0.083, 0.179] | 0/328 = 0.000 | [0.000, 0.012] | [0.000, 0.000] | 1.000 [0.988, 1.000] | 0/199 = 0.000 [0.000, 0.019] | 0.0 |
| always_claim | 0/179 = 0.000 [0.000, 0.021] | 11/328 = 0.034 | [0.019, 0.059] | [0.015, 0.056] | 0.000 [0.000, 0.012] | 199/199 = 1.000 [0.981, 1.000] | 0.0 |
| prior [prior.json, unfiltered v2] | 0/179 = 0.000 [0.000, 0.021] | 16/328 = 0.049 | [0.030, 0.078] | [0.027, 0.074] | 0.000 [0.000, 0.012] | 199/199 = 1.000 [0.981, 1.000] | 0.0 |
| prior_or_none [prior.json, unfiltered v2] | 0/179 = 0.000 [0.000, 0.021] | 16/328 = 0.049 | [0.030, 0.078] | [0.027, 0.074] | 0.000 [0.000, 0.012] | 199/199 = 1.000 [0.981, 1.000] | 0.0 |
| prior [prior_v2f.json, filtered draw] | 1/179 = 0.006 [0.001, 0.031] | 17/328 = 0.052 | [0.033, 0.081] | [0.029, 0.076] | 0.000 [0.000, 0.012] | 199/199 = 1.000 [0.981, 1.000] | 0.0 |
| prior_or_none [prior_v2f.json, filtered draw] | 1/179 = 0.006 [0.001, 0.031] | 17/328 = 0.052 | [0.033, 0.081] | [0.029, 0.076] | 0.000 [0.000, 0.012] | 199/199 = 1.000 [0.981, 1.000] | 0.0 |
| random | 10/179 = 0.056 [0.031, 0.100] | 5/328 = 0.015 | [0.006, 0.035] | [0.003, 0.029] | 0.588 [0.534, 0.640] | 79/199 = 0.397 [0.332, 0.466] | 0.0 |
| name_probe | 0/179 = 0.000 [0.000, 0.021] | 11/328 = 0.034 | [0.019, 0.059] | [0.015, 0.056] | 0.000 [0.000, 0.012] | 199/199 = 1.000 [0.981, 1.000] | 80.7 |
| name_probe_thr | 15/179 = 0.084 [0.051, 0.134] | 0/328 = 0.000 | [0.000, 0.012] | [0.000, 0.000] | 0.908 [0.872, 0.935] | 21/199 = 0.105 [0.070, 0.156] | 80.7 |
| vocab_match | 21/179 = 0.117 [0.078, 0.173] | 23/328 = 0.070 | [0.047, 0.103] | [0.045, 0.098] | 0.842 [0.798, 0.877] | 14/199 = 0.070 [0.042, 0.115] | 4.0 |

### Language slots (all tiers)

| Baseline | Pass (Wilson 95%) | Planted acc | Wilson 95% | Clustered 95% | Planted NF (Wilson) | Null false claim (Wilson) | Fwd mean |
|---|---|---|---|---|---|---|---|
| reference (one-shot) | 131/131 = 1.000 [0.972, 1.000] | 113/113 = 1.000 | [0.967, 1.000] | [1.000, 1.000] | 0.000 [0.000, 0.033] | 0/85 = 0.000 [0.000, 0.043] | 720.3 |
| reference (best-of-5) | 131/131 = 1.000 [0.972, 1.000] | 113/113 = 1.000 | [0.967, 1.000] | [1.000, 1.000] | 0.000 [0.000, 0.033] | 0/85 = 0.000 [0.000, 0.043] | 720.3 |
| SR-max | 62/131 = 0.473 [0.390, 0.558] | 113/113 = 1.000 | [0.967, 1.000] | [1.000, 1.000] | 0.000 [0.000, 0.033] | 85/85 = 1.000 [0.957, 1.000] | 488.2 |
| SR-thr | 118/131 = 0.901 [0.838, 0.941] | 113/113 = 1.000 | [0.967, 1.000] | [1.000, 1.000] | 0.000 [0.000, 0.033] | 14/85 = 0.165 [0.101, 0.258] | 488.2 |
| SR-max-capped | 62/131 = 0.473 [0.390, 0.558] | 113/113 = 1.000 | [0.967, 1.000] | [1.000, 1.000] | 0.000 [0.000, 0.033] | 85/85 = 1.000 [0.957, 1.000] | 480.6 |
| SR-thr-capped | 118/131 = 0.901 [0.838, 0.941] | 113/113 = 1.000 | [0.967, 1.000] | [1.000, 1.000] | 0.000 [0.000, 0.033] | 14/85 = 0.165 [0.101, 0.258] | 480.6 |
| SR-max-all20 | 60/131 = 0.458 [0.375, 0.543] | 110/113 = 0.974 | [0.925, 0.991] | [0.940, 1.000] | 0.000 [0.000, 0.033] | 85/85 = 1.000 [0.957, 1.000] | 1627.5 |
| SR-thr-all20 | 121/131 = 0.924 [0.865, 0.958] | 110/113 = 0.974 | [0.925, 0.991] | [0.940, 1.000] | 0.026 [0.009, 0.075] | 7/85 = 0.082 [0.041, 0.160] | 1627.5 |
| black-box control | 42/131 = 0.321 [0.247, 0.405] | 0/113 = 0.000 | [0.000, 0.033] | [0.000, 0.000] | 1.000 [0.967, 1.000] | 0/85 = 0.000 [0.000, 0.043] | 4.1 |
| self_probe | 51/131 = 0.389 [0.310, 0.475] | 17/113 = 0.150 | [0.096, 0.228] | [0.086, 0.223] | 0.841 [0.762, 0.897] | 5/85 = 0.059 [0.025, 0.130] | 430.7 |
| template_probe | 115/131 = 0.878 [0.811, 0.923] | 107/113 = 0.947 | [0.889, 0.975] | [0.897, 0.990] | 0.053 [0.025, 0.111] | 12/85 = 0.141 [0.083, 0.231] | 406.3 |
| nothing | 42/131 = 0.321 [0.247, 0.405] | 0/113 = 0.000 | [0.000, 0.033] | [0.000, 0.000] | 1.000 [0.967, 1.000] | 0/85 = 0.000 [0.000, 0.043] | 0.0 |
| always_claim | 1/131 = 0.008 [0.001, 0.042] | 2/113 = 0.018 | [0.005, 0.062] | [0.000, 0.046] | 0.000 [0.000, 0.033] | 85/85 = 1.000 [0.957, 1.000] | 0.0 |
| prior [prior.json, unfiltered v2] | 0/131 = 0.000 [0.000, 0.029] | 1/113 = 0.009 | [0.002, 0.048] | [0.000, 0.029] | 0.000 [0.000, 0.033] | 85/85 = 1.000 [0.957, 1.000] | 0.0 |
| prior_or_none [prior.json, unfiltered v2] | 0/131 = 0.000 [0.000, 0.029] | 1/113 = 0.009 | [0.002, 0.048] | [0.000, 0.029] | 0.000 [0.000, 0.033] | 85/85 = 1.000 [0.957, 1.000] | 0.0 |
| prior [prior_v2f.json, filtered draw] | 3/131 = 0.023 [0.008, 0.065] | 9/113 = 0.080 | [0.043, 0.144] | [0.035, 0.133] | 0.000 [0.000, 0.033] | 85/85 = 1.000 [0.957, 1.000] | 0.0 |
| prior_or_none [prior_v2f.json, filtered draw] | 3/131 = 0.023 [0.008, 0.065] | 9/113 = 0.080 | [0.043, 0.144] | [0.035, 0.133] | 0.000 [0.000, 0.033] | 85/85 = 1.000 [0.957, 1.000] | 0.0 |
| random | 16/131 = 0.122 [0.077, 0.189] | 1/113 = 0.009 | [0.002, 0.048] | [0.000, 0.030] | 0.549 [0.457, 0.637] | 39/85 = 0.459 [0.357, 0.564] | 0.0 |
| name_probe | 1/131 = 0.008 [0.001, 0.042] | 2/113 = 0.018 | [0.005, 0.062] | [0.000, 0.046] | 0.000 [0.000, 0.033] | 85/85 = 1.000 [0.957, 1.000] | 81.4 |
| name_probe_thr | 40/131 = 0.305 [0.233, 0.389] | 0/113 = 0.000 | [0.000, 0.033] | [0.000, 0.000] | 0.982 [0.938, 0.995] | 6/85 = 0.071 [0.033, 0.146] | 81.4 |
| vocab_match | 40/131 = 0.305 [0.233, 0.389] | 0/113 = 0.000 | [0.000, 0.033] | [0.000, 0.000] | 0.974 [0.925, 0.991] | 2/85 = 0.024 [0.006, 0.082] | 4.1 |

### Planted accuracy by tier and family

| Baseline | pooled all | pooled T1 | pooled T2 | pooled T3 | topic all | topic T1 | topic T2 | topic T3 | language all | language T1 | language T2 | language T3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference (one-shot) | 1.00 (n=441) | 1.00 (n=153) | 1.00 (n=149) | 1.00 (n=139) | 1.00 (n=328) | 1.00 (n=120) | 1.00 (n=103) | 1.00 (n=105) | 1.00 (n=113) | 1.00 (n=33) | 1.00 (n=46) | 1.00 (n=34) |
| reference (best-of-5) | 1.00 (n=441) | 1.00 (n=153) | 1.00 (n=149) | 1.00 (n=139) | 1.00 (n=328) | 1.00 (n=120) | 1.00 (n=103) | 1.00 (n=105) | 1.00 (n=113) | 1.00 (n=33) | 1.00 (n=46) | 1.00 (n=34) |
| SR-max | 0.87 (n=441) | 0.90 (n=153) | 0.91 (n=149) | 0.81 (n=139) | 0.83 (n=328) | 0.87 (n=120) | 0.87 (n=103) | 0.74 (n=105) | 1.00 (n=113) | 1.00 (n=33) | 1.00 (n=46) | 1.00 (n=34) |
| SR-thr | 0.73 (n=441) | 0.73 (n=153) | 0.77 (n=149) | 0.69 (n=139) | 0.64 (n=328) | 0.65 (n=120) | 0.67 (n=103) | 0.59 (n=105) | 1.00 (n=113) | 1.00 (n=33) | 1.00 (n=46) | 1.00 (n=34) |
| SR-max-capped | 0.87 (n=441) | 0.90 (n=153) | 0.91 (n=149) | 0.81 (n=139) | 0.83 (n=328) | 0.87 (n=120) | 0.87 (n=103) | 0.74 (n=105) | 1.00 (n=113) | 1.00 (n=33) | 1.00 (n=46) | 1.00 (n=34) |
| SR-thr-capped | 0.73 (n=441) | 0.73 (n=153) | 0.77 (n=149) | 0.70 (n=139) | 0.64 (n=328) | 0.65 (n=120) | 0.67 (n=103) | 0.60 (n=105) | 1.00 (n=113) | 1.00 (n=33) | 1.00 (n=46) | 1.00 (n=34) |
| SR-max-all20 | 0.96 (n=441) | 0.98 (n=153) | 0.97 (n=149) | 0.94 (n=139) | 0.96 (n=328) | 0.98 (n=120) | 0.95 (n=103) | 0.94 (n=105) | 0.97 (n=113) | 0.97 (n=33) | 1.00 (n=46) | 0.94 (n=34) |
| SR-thr-all20 | 0.83 (n=441) | 0.83 (n=153) | 0.85 (n=149) | 0.81 (n=139) | 0.78 (n=328) | 0.79 (n=120) | 0.79 (n=103) | 0.77 (n=105) | 0.97 (n=113) | 0.97 (n=33) | 1.00 (n=46) | 0.94 (n=34) |
| black-box control | 0.00 (n=441) | 0.00 (n=153) | 0.00 (n=149) | 0.00 (n=139) | 0.00 (n=328) | 0.00 (n=120) | 0.00 (n=103) | 0.00 (n=105) | 0.00 (n=113) | 0.00 (n=33) | 0.00 (n=46) | 0.00 (n=34) |
| self_probe | 0.20 (n=441) | 0.22 (n=153) | 0.23 (n=149) | 0.17 (n=139) | 0.22 (n=328) | 0.23 (n=120) | 0.25 (n=103) | 0.19 (n=105) | 0.15 (n=113) | 0.18 (n=33) | 0.17 (n=46) | 0.09 (n=34) |
| template_probe | 0.29 (n=441) | 0.24 (n=153) | 0.37 (n=149) | 0.25 (n=139) | 0.06 (n=328) | 0.06 (n=120) | 0.10 (n=103) | 0.03 (n=105) | 0.95 (n=113) | 0.91 (n=33) | 0.98 (n=46) | 0.94 (n=34) |
| nothing | 0.00 (n=441) | 0.00 (n=153) | 0.00 (n=149) | 0.00 (n=139) | 0.00 (n=328) | 0.00 (n=120) | 0.00 (n=103) | 0.00 (n=105) | 0.00 (n=113) | 0.00 (n=33) | 0.00 (n=46) | 0.00 (n=34) |
| always_claim | 0.03 (n=441) | 0.02 (n=153) | 0.01 (n=149) | 0.06 (n=139) | 0.03 (n=328) | 0.03 (n=120) | 0.01 (n=103) | 0.07 (n=105) | 0.02 (n=113) | 0.00 (n=33) | 0.02 (n=46) | 0.03 (n=34) |
| prior [prior.json, unfiltered v2] | 0.04 (n=441) | 0.09 (n=153) | 0.02 (n=149) | 0.01 (n=139) | 0.05 (n=328) | 0.10 (n=120) | 0.03 (n=103) | 0.01 (n=105) | 0.01 (n=113) | 0.03 (n=33) | 0.00 (n=46) | 0.00 (n=34) |
| prior_or_none [prior.json, unfiltered v2] | 0.04 (n=441) | 0.09 (n=153) | 0.02 (n=149) | 0.01 (n=139) | 0.05 (n=328) | 0.10 (n=120) | 0.03 (n=103) | 0.01 (n=105) | 0.01 (n=113) | 0.03 (n=33) | 0.00 (n=46) | 0.00 (n=34) |
| prior [prior_v2f.json, filtered draw] | 0.06 (n=441) | 0.03 (n=153) | 0.07 (n=149) | 0.07 (n=139) | 0.05 (n=328) | 0.03 (n=120) | 0.08 (n=103) | 0.06 (n=105) | 0.08 (n=113) | 0.06 (n=33) | 0.07 (n=46) | 0.12 (n=34) |
| prior_or_none [prior_v2f.json, filtered draw] | 0.06 (n=441) | 0.03 (n=153) | 0.07 (n=149) | 0.07 (n=139) | 0.05 (n=328) | 0.03 (n=120) | 0.08 (n=103) | 0.06 (n=105) | 0.08 (n=113) | 0.06 (n=33) | 0.07 (n=46) | 0.12 (n=34) |
| random | 0.01 (n=441) | 0.03 (n=153) | 0.01 (n=149) | 0.00 (n=139) | 0.02 (n=328) | 0.03 (n=120) | 0.01 (n=103) | 0.00 (n=105) | 0.01 (n=113) | 0.03 (n=33) | 0.00 (n=46) | 0.00 (n=34) |
| name_probe | 0.03 (n=441) | 0.02 (n=153) | 0.01 (n=149) | 0.06 (n=139) | 0.03 (n=328) | 0.03 (n=120) | 0.01 (n=103) | 0.07 (n=105) | 0.02 (n=113) | 0.00 (n=33) | 0.02 (n=46) | 0.03 (n=34) |
| name_probe_thr | 0.00 (n=441) | 0.00 (n=153) | 0.00 (n=149) | 0.00 (n=139) | 0.00 (n=328) | 0.00 (n=120) | 0.00 (n=103) | 0.00 (n=105) | 0.00 (n=113) | 0.00 (n=33) | 0.00 (n=46) | 0.00 (n=34) |
| vocab_match | 0.05 (n=441) | 0.05 (n=153) | 0.03 (n=149) | 0.08 (n=139) | 0.07 (n=328) | 0.06 (n=120) | 0.05 (n=103) | 0.10 (n=105) | 0.00 (n=113) | 0.00 (n=33) | 0.00 (n=46) | 0.00 (n=34) |

Per-tier tables with all metrics (T1, T2, T3) are in `tables_autogen.md`.

## The prior recipes on the filtered pool

`prior.json` was built from the **unfiltered** v2 generator (seeds 900000+, T1 and T2, 165 labels). The filtered pool
draws its answers from a smaller universe (74 anchor concepts), so I also built `prior_v2f.json` from a disjoint draw
of the **filtered** generator: the same restricted tables as `write_v2f.py` (generator v2 with the latent pool
restricted to the A1.1-kept latents of `key_check.jsonl`, sha256 eba3fb4a...), seeds 9000-9299 (T1), 109000-109299
(T2), 209000-209299 (T3). That is 900 instances, 3,585 slots and 2,164 planted answers over 74 labels, with no draw
errors and no seed or instance id shared with the evaluation pool (`build_prior_v2f.py`, `prior_v2f_meta.json`). The
filtered prior is flat: the most common label ("text written in Thai") is the answer in 3.0% of planted slots, the
median label in 1.4%. With it, prior gets 26/441 = 0.059 [0.041, 0.085] (unfiltered prior: 0.038), still well under
the gate. prior_or_none gives exactly the same answers as prior with either prior, because the best option on a menu
always has a prior above the median.

## Predictions vs outcomes (step 3)

| # | Prediction | Outcome | |
|---|---|---|---|
| P7 | reference planted accuracy 0.97, one-shot pass 0.88 | 1.000 [0.991, 1.000]; one-shot pass 178/180 = 0.989 [0.960, 0.997]; best-of-5 180/180 | hit on the gate; both numbers higher than predicted |
| P8 | black-box planted accuracy 0.00, pass about 0.05 (all-null episodes only) | 0.000 [0.000, 0.009]; pass 7/180 = 0.039, exactly the 7 all-null episodes | hit |
| P9 | self_probe planted accuracy 0.30 (planted NF 0.55); gate fails | 0.204 [0.169, 0.244] (planted NF 0.746); gate fails, robustly (topic 0.223, language 0.150) | hit on the gate failure; the level is lower than predicted (0.20 vs 0.30) and planted NF higher (0.75 vs 0.55) |
| P10 | template_probe planted accuracy 0.20 (planted NF 0.65); gate fails or borderline | 0.288 [0.248, 0.332] (planted NF 0.678); gate fails, robustly | hit on the gate failure; the level is higher, and all of it comes from language slots (topic 0.061, language 0.947) |
| P11 | SR-max planted accuracy 0.75 (0.55-0.85) | 0.873 [0.839, 0.901] | miss: above the plausible range |
| P12 | SR-thr planted accuracy 0.70, null false claims 0.10, pass 0.35 | 0.730 / 0.130 / 0.378 | hit (all three close) |
| P13 | the stop rule fires (p about 0.7) | fires; not borderline; not text-selection-sensitive | hit |
| P44 (part) | pa(SR-max) / pa(ref) = 0.77 | 0.873 [0.840, 0.903] | higher than predicted |

## How it was run

- **Pool check.** Before any run, every `instance.json` and `public.json` was hashed and matched against
  `diagnosis/instances_v2f_manifest.json` (180 instances, 0 mismatches, no extra or missing directories).
  Manifest sha256 `6a3cada6335736a6b3b0af8785dc95872823005cb3a221e6c2812e0341498b71`. Pool sha256 (sr_recipe's
  digest over every instance id and `instance.json`) `b212728fc859866f1a5bb8578fd67bc1c5ebca015c98be5d27d436515dbc3dcb`;
  the in-process run, the SR run and the analysis each recompute it and refuse to mix pools.
- **In-process run** (`inproc_step3.py`, run dir `runs/20261002-041021_inproc/`): one `common.gpuq` job (7 GB,
  heavy), model loaded once, `FM_MODEL_SERVICE=off`. It is a thin adapter over `inproc_gates.py`: the same solver
  functions, `Env` class, profiles, per-instance caps, `budget`/`submit` built-ins and out-of-process grader. Additions:
  (1) the reference's probe corpus is read from the fix-stage worktree's `cache/concepts.json` (this worktree has no
  cache), after checking its sha256 against the manifest's `concepts_sha256`; (2) the `@v2f` prior variants swap
  `recipe_baseline.load_prior` for `prior_v2f.json`; (3) self_probe's `generate` calls are memoised by (prompt,
  max_new_tokens) behind `Env.compute`, after the tool has validated and charged the call. Greedy decoding of one
  prompt is deterministic, so this only saves time. Over the run there were 148 distinct prompts and 4,202 repeated
  calls; the first 25 repeats were recomputed and all 25 matched the stored completion. (4) The submission of every
  episode is stored in the private file, so the slots can be re-graded offline. The run made 2,523 episodes (13
  solvers x 180, plus 183 reference episodes: 178 instances passed on seed 7, 1 on the second try, 1 on the third),
  with 0 solver crashes and 0 unsubmitted episodes, in 56 min after 93 s in the GPU queue.
- **Determinism check.** The 7 model-free recipes were also run on CPU with the same adapter (model loading disabled):
  1,260 of 1,260 submissions are identical to the GPU run's.
- **SR run** (`sr_recipe_out/`): the step-3 command of `SR_RECIPE.md`, output directed into this directory:
  `$PY -m tasks.featurematch.diagnosis.sr_recipe --instances tasks/featurematch/instances_v2f --out
  tasks/featurematch/diagnosis/step3/sr_recipe_out`. Bank-R texts and styles match the cache (`bank_check`), and the
  three cache files match `cache/manifest.json`. `analyze_step3.py` recomputes SR's pooled and per-tier metrics from
  `answers.jsonl` and checks that they equal `summary.json` exactly.
- **Analysis** (`analyze_step3.py`): Wilson and clustered bootstrap are `sr_recipe.wilson` and
  `sr_recipe.cluster_bootstrap` (10,000 resamples, `random.Random(20261002)`), so in-process and SR numbers use the
  same functions.
- **Memory.** Every CPU job ran under `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=2G`. Peak RSS
  (`/usr/bin/time -v`): SR run 204 MB, prior build 398 MB, analysis 248 MB. The GPU job (also under the 8 GB scope):
  6.38 GB peak RSS, the heaviest run, mostly the model load.
- **Privacy.** `episodes_private.jsonl`, `sr_recipe_out/answers.jsonl` and `sr_recipe_out/slots.csv` hold per-slot
  truth and submissions. They are gitignored here (`step3/.gitignore`). Only aggregates are committed.

Commands (from `~/wt/fmdiag`, after `source ~/ideating-rl-tests/common/env.sh`):

```bash
CAP="systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=2G --"
$CAP /usr/bin/time -v $PY -m tasks.featurematch.diagnosis.step3.build_prior_v2f
$CAP /usr/bin/time -v $PY -m tasks.featurematch.diagnosis.sr_recipe --instances tasks/featurematch/instances_v2f \
    --out tasks/featurematch/diagnosis/step3/sr_recipe_out
FM_MODEL_SERVICE=off $CAP /usr/bin/time -v $PY -m common.gpuq run --gb 7 --heavy --label fm-diag-step3-inproc -- \
    env FM_MODEL_SERVICE=off $PY -m tasks.featurematch.diagnosis.step3.inproc_step3 \
    --out tasks/featurematch/diagnosis/step3/runs/20261002-041021_inproc
$CAP /usr/bin/time -v $PY -m tasks.featurematch.diagnosis.step3.analyze_step3 \
    --inproc tasks/featurematch/diagnosis/step3/runs/20261002-041021_inproc
```

## Deviations and choices not fixed by PREREG

- In-process instead of the harness for the GPU baselines (instructed; the harness takes 15-40 min per GPU episode on
  the shared queue). Labelled "in-process" everywhere. No broker, leak scan or wall clock; caps are enforced.
- The filtered-generator prior is an addition (not pre-registered). Both priors are reported and labelled.
- The family pass-rate definition above is my choice; A3.2 does not define a per-family pass.
- The reference's best-of-5 slot metrics use the selected episode (first passing seed, else the last). The gate uses
  the one-shot (seed 7) planted accuracy.
- The SR output went to `step3/sr_recipe_out/` instead of `SR_RECIPE.md`'s default `diagnosis/sr_recipe_out/`
  (agents write only to their own directory). The command is otherwise the same.
- The generate memo in the self_probe run (speed only; checked as described above).
