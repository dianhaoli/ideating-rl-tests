# FeatureMatch: independent shortcut audit

Auditor: an independent Claude Code subagent (Opus 5.5), 2026-10-01 19:10-21:xx UTC, on branch `task/featurematch`
at 52a2166 (audit scripts added on top). I did not change any task code. All scripts are in `tasks/featurematch/audit/`.
Every number below points to a run dir in `runs/featurematch/`.

**What the audit asks.** A good task needs interpretability. A black-box agent, a fixed recipe or a prior must fail.
Planted and null slots must look the same. The answer must not leak through tools, errors, sizes, timing, ids or the
prompt. The grader must be correct. Zero-effort policies must score about 0. My job was to try to beat the task cheaply.

**Short verdict.** I found no answer leak and no grader bug. No cheap solver clearly passes more than 10% of episodes,
but one comes close (`self_probe`, see A5). The weaknesses that matter are in the reward and the instance design:
1. The continuous `score` gives the do-nothing policy 0.44.
2. Cheap statistics of the latent separate null slots from planted ones (AUROC 0.70-0.80).
3. The generator's name filter leaves a learnable "second place" artifact.
4. A zero-knowledge recipe in which the subject model writes its own probe texts comes close to the 10% gate.

Pass rates are per episode, and pass means every slot is correct. Wilson 95% CIs are in brackets.

## Summary table (pool of 72 instances, 24 per tier)

| # | attack (what a clever agent might try) | path | pass rate, all tiers | worst tier | planted-slot acc. | severity |
|---|---|---|---|---|---|---|
| A1 | learned menu prior: a model-free policy trained on 1200 instances per tier | harness, 72 | 1/72 = 1.4% [0.2, 7.5] | T1 1/24 | 0.019 | MINOR |
| A2 | latent statistics + learned prior: 64 generic texts + option names, GBM | offline-exact, 400 per tier held-out | 6.3 / 7.5 / 3.8% | T2 7.5% | 0.12-0.17 | MAJOR (fingerprint) |
| A3 | template_probe: fixed templates + a fixed 19-language sentence bank | in-process 72; harness 3 | 3/72 = 4.2% [1.4, 11.6] | T3 2/24 | 0.11 | MINOR |
| A4 | density_null: "nothing found" if the latent is dense, else the template pick | in-process 72; harness 3 | 3/72 = 4.2% [1.4, 11.6] | T1 2/24 | 0.29 | MINOR |
| A5 | self_probe: the subject model writes probe texts for the top-3 template options | in-process SELF72; harness 3 | SELFALL | SELFWORST | SELFACC | SELFSEV |
| A6 | name_rank2: claim the SECOND-highest name-probe option (filter artifact) | in-process 72 (+ offline, identical) | 4/72 = 5.6% [2.2, 13.4] | T1 2/24 | 0.076 | MAJOR (artifact) |
| G | grader + validator, 29 edge-case submissions | CLI, separate process | no false pass, no false fail | - | - | OK (2 MINOR notes) |
| R | reward shape: mean slot `score` of zero-effort policies | harness | "nothing found" everywhere scores 0.436 | - | - | MAJOR |

Reference for comparison (integrator): one-shot 94.4%, best-of-5 100%. Integrator recipes: at most 2.8% overall.

---

## Findings, ranked

### MAJOR 1: the continuous score rewards the null action (reward shape)
- **Measured** (harness, `20261001-172659_gate_recipes`, 72 episodes): `recipe_nothing` mean `score` = **0.436**.
  The learned prior (A1, harness) scores 0.433, template_probe 0.47 and name_rank2 0.45. The reference scores about 0.98.
- **Why it matters.** Plan v2 rule 1 says zero-effort policies, including the null action, must score about 0
  *under the reward*. `pass` meets that rule (2.8%). `score` (mean slot correctness, also in grade.json) does not.
  If anyone trains on `score`, which is the natural continuous signal for GRPO, "nothing found" everywhere is a strong
  local optimum worth 0.44, almost half the reference. The same mechanism drives the auditor-RL failure that
  CONTEXT.md warns about: with a mean reward, under-claiming is cheap.
- **Fix.** Make the continuous reward a product of the two error types, as plan v2 asks:
  `score = (correct planted slots / planted slots) x (correct null slots / null slots)`, taking an empty factor as 1.
  Then "nothing found" scores 0 unless the episode is all-null, always-claim scores 0 unless it is all-planted, and
  the reference stays at about 0.98. Alternatively, publish `pass` as the only reward and rename `score` to
  `diagnostic_slot_accuracy`. Either way, state in SPEC.md which field is the RL reward.

### MAJOR 2: null vs planted slots are separable from cheap latent statistics (fingerprint)
- **Attack A2** (`audit/offline_attacks.py`, run dir `20261001-191842_audit_offline`). Per slot, the policy measures the
  latent on a FIXED set of 64 generic texts (the same texts for every slot; split-C texts stand in for "any 64
  texts") and on the 20 bare option names in the 3 surface variants. It adds the public menu features. A
  gradient-boosted model is trained on 1200 instances per tier and tested on 400 held-out instances per tier.
  This costs about 124 forward units per slot and needs no thought about concepts.
  Null-vs-planted AUROC: **0.70 (T1), 0.80 (T2), 0.77 (T3)**. The menu-only model (A1) gets 0.51 / 0.64 / 0.62.
  - Ablation (`ablate_generic/`, `ablate_names/`): the generic-text statistics alone give 0.65 / 0.77 / 0.74. They
    are the main signal. The name probes alone give 0.55 / 0.67 / 0.64 and lift planted accuracy (next finding).
  - Episode pass for A2 stays low: 6.3 / 7.5 / 3.8% held-out, and 8.3 / 8.3 / 8.3% (2/24 per tier) on the pool.
    The reason is that you still have to name the right option on planted slots.
- **Cause.** The **off-universe nulls** (40% of nulls) are, by construction, "live but not selective for any concept".
  Their firing statistics on generic text (rate, mean, upper quantile) differ from those of the concept-selective
  latents used in planted and near-miss slots. Near-miss nulls come from the same latent pool as planted slots, so
  they are not separable this way.
- **Why it matters.** The design rule says planted and null slots must be indistinguishable. An RL policy will learn
  "dense or odd firing pattern => nothing found" without probing any option. That is a shortcut for about 40% of the
  null decisions, and it lowers the effective difficulty of the null half of the task.
- **Fix.** Match the off-universe latents to the planted pool on the statistics an agent can measure cheaply. For
  example, draw off-universe latents by stratified sampling so that their firing rate and activation-magnitude
  quantiles on a generic corpus follow the planted distribution. Or drop off-universe nulls and use near-miss nulls
  only. Then re-run `audit/offline_attacks.py` and require the A2 null AUROC to be at most about 0.6 in every tier.

### MAJOR 3: the asymmetric name filter leaves a "second place" artifact
- **Measured** (`audit/name_rank_check.py`, output in `20261001-191842_audit_offline/name_rank_check.txt`). The
  generator rejects a planted slot when the answer's own name is the TOP name-probe activation in the menu. Only rank
  1 is removed. Among planted slots where any option name fires, the answer's name is **exactly rank 2 in 44% (T1) /
  36% (T2)** of cases. Also, any option name fires in 22-27% of planted slots but in only 3-5% of off-universe nulls.
- **Attack A6** `name_rank2`: run the 3 name variants of each option and claim the 2nd-highest, or "nothing found"
  if fewer than 2 names fire. It costs 60 forward units per slot. Planted-slot accuracy is 0.076, against 0.05 for
  the integrator's name_probe. Pass: 4/72 = 5.6% [2.2, 13.4]
  (in-process `20261001-204214_audit_inproc_namerank2`; the offline simulation gives the same 2 / 1 / 1 per tier
  exactly, which also confirms the offline simulator).
- **Why it matters.** It is below 10% alone. But it is a clean, learnable rule that comes from the filter and not
  from the model, and it stacks with MAJOR 2: in A2 the name features lift planted accuracy from 0.004 to 0.115 (T1).
- **Fix.** Filter LATENTS, not menus. Drop any latent whose own concept's name ranks in the top-k (say k = 5) of
  all 232 concept names on that latent, under any variant. Apply the same filter to the latents used for near-miss
  nulls, so the filter is symmetric. Then the answer's name rank carries no information.

### A5: self_probe, a zero-knowledge recipe in which the subject model writes the probes
SELFTEXT

### MINOR 4: language slots fall to a fixed sentence bank
- A3/A4 use one fixed 3-sentence bank per language, written once by me and the same in every episode. On the pool's
  16 language-answer planted slots, planted accuracy is **0.69 (template_probe) and 0.88 (density_null)**. On the
  141 topic slots it is 0.05 and 0.22. So language slots need no thought about the specific latent, and the
  difficulty sits entirely in the topic slots. Only about 10% of planted slots are languages, so the effect on the
  pass rate is small.
- **Fix.** Optional. Cap language anchors per episode, or accept and document this: language slots are the "easy"
  items.

### MINOR 5: learned menu prior (A1)
- Harness `20261001-200556_audit_harness_learned_prior` (72 episodes, all VALID): **1/72 = 1.4%**, planted accuracy
  0.019, null false claims 3.2%. Held-out in-memory (400 per tier): 3.3 / 4.8 / 3.0%. The policy learns to say
  "nothing found" almost everywhere. Its label x layer prior picks the answer 8-17% of the time in T2/T3 (chance
  5%), and its menu null-AUROC is 0.64 in T2, slightly above the integrator's 0.60 because label identity adds signal.
- **Fix.** None needed for the gate. MAJOR 2's fix (match the menu mix) would also remove this.

### MINOR 6: grader and validator (no correctness bugs)
`audit/grader_edge_cases.py` runs 29 cases on a synthetic instance (no real answer key), through
`Env.validate_submission` and through `grader.py` in a separate process (`20261001-192451_audit_grader`).
Every format-valid submission is graded right: no false pass and no false fail. Accepted forms include string numbers,
"Nothing Found" in any case or spacing, slot ids as strings, shuffled order, extra keys and full-width digits. A
missing file, `[]` or a non-object scores 0.
- Note (a): the grader is more lenient than the validator. It would accept `True` as option 1, `7.0`, and a
  `{"answer": {...}}` wrapper. That is harmless today, because the broker always runs the validator before it stores
  a submission. Suggest making `grader._norm_choice` reject bool and float, for defence in depth.
- Note (b): a float slot id (`0.9`) truncates to slot 0 in both the validator and the grader. It is consistent and
  harmless.

### MINOR 7: hygiene and harness notes
- **The integrator's "BLOCKING (1)" is already fixed on this branch.** Commit 7ace2d4 (merged) matches codename
  variants with whitespace kept. I checked with `leakscan.scan_text`: "this feature matches Spanish",
  "Feature match:", "feature_matches.py" and "Feature-match" are all clean now. Only the single word
  "featurematch" or "featurematching" in an agent's own file still invalidates an episode. That is unlikely but
  possible (an agent naming a script `featurematching.py`).
- The orchestrator's probe dirs (`20261001-165348_apiprobe_orch`, `..._171048_apiprobe_sonnet`,
  `..._182038_openai_probe_orch`) are untracked AND not matched by `runs/featurematch/.gitignore`. Their
  `grade.json` files hold per-slot planted flags for pool instances. A careless `git add runs/` would commit part of
  an answer key (D6). Suggest adding `*_apiprobe_*/episodes/` and `*_openai_probe_*/episodes/` to the ignore file.
- `smoke_plan.json` keeps only instances the reference passed on its FIRST try. The smoke set is therefore easier
  than the pool, and the smoke pass rate will be biased upward. Report it as such, or draw the smoke set from all
  kept instances.
- As D6 accepts, answers can be regenerated from the committed generator, seeds and public datasets (with a GPU
  precompute). Only the Tier B transcript audit (no network) stops an agent from doing that.

## Leak checks that came out clean
- **Tools never touch the answer.** `tools.py` reads only `extra.perm_seed` and `public.json`. No tool argument,
  return value or error message depends on planted/null or on the menu-to-truth mapping. Error texts are generic.
  Timing therefore cannot depend on the answer except through the agent's own inputs. Planted and null instances also
  load the model the same way, in a background thread.
- **Latent indices** are a uniform per-instance permutation with a 62-bit secret seed (also a leak string). The
  index in TASK.md carries no information, and Neuronpedia-style lookups fail.
- **TASK.md / public.json** have identical keys and shapes. Ids are hashes. The harness `check_task_md` passes on all
  72 (integrator). Menu composition is the only public-side signal (A1, MINOR 5).
- **Committed files**: `gates_public.json` holds only aggregates. `episodes_private.jsonl` and the gate episode dirs
  are gitignored. My audit run dirs commit only aggregate JSON and text; the per-slot jsonl stays local.

## Method notes (what I ran, and why it is a fair test)
- **Harness runs** (`common.sandbox run-scripted`, broker + leak scan + out-of-process grader):
  learned_prior on all 72 (model-free, no GPU), and template_probe / density_null / self_probe on 3 instances each
  (1 per tier: fm-t1-07e0a3cb3f, fm-t2-08a4773087, fm-t3-025fcaa85e). The shared queue made each 7 GB tool server
  wait 10-25 min, so the GPU attacks use the same labelled **in-process complement** pattern the integrator used
  (`audit/inproc_audit.py`: same Env, caps and out-of-process grader, one queued job). HARNESSXCHECK
- **Offline-exact runs** (A2, A6, name-rank) read the precomputed max-activation tables. These hold exactly the
  numbers `latent_activations` returns for those texts. A6's in-process result matches the offline one slot for slot.
- **No data leakage into the attacks.** Learned policies were trained on in-memory instances from seeds 600000+
  (disjoint from the pool, the fingerprint check and prior.json) and never read pool answers. The template and
  language texts were written once, before any result, and are the same in every episode.
- **Re-run:** `$PY -m tasks.featurematch.audit.offline_attacks --out <dir> --save-model tasks/featurematch/audit/cache`
  (CPU, about 30 min per tier). `FM_AUDIT_FEATS=generic|names ... --only-latent` runs the ablations. GPU attacks:
  `$PY -m common.gpuq run --gb 7 --label featurematch-audit -- $PY -m tasks.featurematch.audit.inproc_audit --instances-file F --variants template_probe,density_null,self_probe,name_rank2 --out <dir>`;
  harness: `bash tasks/featurematch/audit/run_harness_attacks.sh <ts> <instance dirs>`.
  The learned-prior pickles (`audit/cache/`, about 1 MB each) are regenerable and not committed.

## Recommended fixes, in priority order
1. Make the RL reward a product (planted accuracy x null accuracy) or `pass` only (MAJOR 1).
2. Match off-universe null latents to planted latents on cheap firing statistics, or use near-miss nulls only. Gate:
   A2 null-AUROC at most about 0.6 (MAJOR 2).
3. Replace the menu-level name filter with a symmetric latent-level filter (top-k own-name rank over the universe)
   (MAJOR 3).
4. SELFFIX
5. Ignore-file lines for the orchestrator probe dirs; grader rejects bool and float; report the smoke-set selection
   bias (MINOR 6-7).
