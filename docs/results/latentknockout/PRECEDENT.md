# LatentKnockout: precedent survey

Written 2026-10-02 (UTC), CPU only: no GPU job and no model run, only reading. Purpose: before the feasibility
results come in, collect what is already known about **targeted ablation or steering with SAE latents compared
with simple baselines**, and **how such tasks are scored**, and say what each finding predicts for LatentKnockout.

**How to read the tags.**
- **[peer-reviewed]**: the venue is stated on the paper's page (e.g. ICML 2025).
- **[preprint]**: arXiv or blog only, not peer reviewed. Verify every number before quoting it.
- **[verify]**: I read the number in an automated extraction of the paper's HTML, not in the PDF itself.
  It is probably right, but check the table before anyone quotes it.
- No citation below was written from memory. Every one was opened on 2026-10-02 at the URL given.
  Where I could not find a number, the entry says so.

The frozen predictions in `PREDICTIONS_FEASIBILITY.md` are not edited. Section 4 below says where the literature
agrees or disagrees with them.

## 0. Summary in five sentences

1. Across almost every head-to-head comparison, SAE latents **lose to simple supervised dense directions**. The
   examples: difference of means (DiffMean), linear probes, DAS rotations, even plain neurons. The comparisons cover
   steering, concept detection, causal variable localisation and OOD probing (AxBench, RAVEL, MIB, Chaudhary & Geiger,
   Kantamneni et al., GDM). So restricting the agent to SAE latents is a **handicap**. The reward must not assume the
   SAE is the best tool.
2. The standard way to pick latents for a targeted ablation is a **fixed recipe**: rank by attribution or by probe
   weight, then take the top k (SAEBench SCR/TPP, Marks et al. SHIFT, SAEBench unlearning). That recipe is exactly the
   shortcut LatentKnockout must not reward, so it has to be a gate.
3. **Zero-ablating** a few SAE latents often does little. Unlearning work needed **negative clamping**, and the
   multi-latent edits had side effects at least as large as a fine-tuning baseline (Farrell et al.; SAEBench unlearning).
4. **Feature splitting and absorption** are universal in SAEs ("A is for Absorption"; Scaling Monosemanticity). A
   latent that looks like "Texas" can stay silent on some Texas cities, which is exactly what held-out-entity grading measures.
5. Agent benchmarks that use SAEs find agents **worst at the causal (intervention) part** and prone to choosing
   formatting or substring latents (SAEScientist-Bench). This supports a mid-band difficulty, provided the recipe gate
   holds.

## 1. Terms used below (plain language)

- **SAE latent**: one of the 16,384 learned directions in a Gemma Scope SAE. On any token only ~70 are non-zero.
- **Zero ablation** of latent i: subtract its contribution f_i(x)·W_dec[i] from the residual stream, keeping the SAE's
  reconstruction error. This is the brief's default intervention.
  **Negative clamping**: set f_i to -c × (a typical activation), i.e. push the other way.
- **Steering**: add a direction to the residual stream to cause a behaviour.
  **Directional ablation**: project a direction out of the residual stream.
- **DiffMean**: the difference of mean activations between target and control prompts, used as a steering or ablation
  direction. It is the "plain steering vector" baseline.
- **DAS**: distributed alignment search, a learned rotation that finds a subspace whose patching changes a variable.
  It is supervised.
- **IIA (interchange-intervention accuracy)**: the share of prompts where patching a feature from a source prompt
  produces the output a high-level causal model predicts.
- **RAVEL Cause / Iso**: Cause = the targeted attribute changes. Iso = the other attributes do not. This is the
  same shape as LatentKnockout's Effect / Preserve.
- **SCR / TPP** (SAEBench): ablate the top-k latents for one concept and measure how much a probe for that concept
  drops, versus probes for other concepts (TPP), or how much a spurious signal is removed (SCR).
- **Feature splitting**: a broad latent ("Texas") in a small SAE becomes several narrow ones ("Dallas", "Houston") in
  a larger SAE. **Absorption**: the broad latent fails to fire on some tokens where it should, because a narrow latent
  "absorbed" that case.

## 2. Overview table

| Source | What is scored | Headline number | Prediction for LatentKnockout |
|---|---|---|---|
| SAEBench (ICML 2025) | SCR, TPP, unlearning, absorption, RAVEL over 200+ SAEs | top-k = 20 latents is the main setting; unlearning clamps at -25…-200x with MMLU kept >= 0.99 | the field's own targeted-ablation evals use a fixed selection recipe and k ~ 20, so k <= 5 is unusually tight |
| Chanin 2026 (preprint) | reliability of those metrics | TPP reseed CV 23% (top-50) / 39% (top-10); SCR rho = -0.64 with ground truth at top-500 | small-k ablation scores are noisy; we need many held-out items and CIs |
| AxBench (ICML 2025) | steering (LLM judge), detection (AUROC) on Gemma-2 | steering avg: DiffMean 0.239 vs SAE 0.165; detection AUROC DiffMean 0.942 vs SAE 0.695 vs SAE picked on labels 0.917 | a plain DiffMean vector will likely beat any <= k latent set; latents chosen from data beat latents chosen by label |
| Arad et al. 2025 (preprint) | AxBench steering with "output-score" filtering | Gemma-2-9B L20: 0.546 filtered vs 0.293 unfiltered (LoRA 0.602) | late-layer "output" latents are the ones that move behaviour; layer matters |
| RAVEL (ACL 2024) | Cause / Iso / Disentangle (city attributes etc.) | Llama2-7B Disentangle, entity split: SAE 48.6, DAS 56.5, MDAS 60.1 | more latents means more Cause and less Iso; SAE is near the bottom |
| MIB (ICML 2025) | IIA (variables), CPR / CMD (circuits); public + private test sets | SAE features "generally fail to improve upon neurons"; MCQA Gemma-2: DAS 95 / 77 vs SAE 73 / 51 [verify] | private held-out sets are the norm; SAE latents are not privileged units |
| Chaudhary & Geiger 2024 (preprint) | RAVEL country vs continent with SAE masks, GPT-2 small | SAEs "struggle to reach the neuron baseline"; none near DAS | the city -> region relation is exactly where SAEs underperform |
| Marks et al. (ICLR 2025) | sparse feature circuits; SHIFT ablation | Gemma-2-2B SHIFT: gender acc 81.9 -> 51.5, worst-group 18.2 -> 50.0 after ablating 43 human-picked features | targeted SAE ablation can work, but it used tens of latents plus human judgement |
| Farrell et al. 2024 (preprint) | WMDP-bio unlearning with Gemma Scope latents | zero ablation "ineffective"; negative scaling needed; side effects >= RMU | the default "subtract the contribution" ablation may be too weak |
| A is for Absorption (NeurIPS 2025) | first-letter absorption on Gemma Scope 16k / 65k | absorption in every SAE tested; size/L0 changes do not fix it | held-out entities will expose absorption false negatives |
| Scaling Monosemanticity (2024) | feature completeness vs concept frequency | ~60% of London boroughs had a feature in a 34M SAE | small states or rare languages may have no latent at 16k: natural "cannot" instances |
| Paulo & Belrose 2025 (preprint) | seed stability of SAEs | 30% of features shared across seeds (131K SAE, Llama 3 8B) | a different SAE variant at eval time breaks memorised IDs |
| Korznikov et al. 2026 (preprint) | trained SAE vs random-direction baselines | causal editing 0.73 (random) vs 0.72 (trained) | we must check that a random dictionary cannot do the task too |
| Biology / circuit tracing (Anthropic 2025) | Dallas -> Texas -> Austin in Claude 3.5 Haiku | swapping Texas for California features outputs Sacramento; a direct Dallas -> Austin shortcut edge also exists | our flagship example is the field's most famous one; there is a shortcut path around "Texas" |
| Arora et al. 2026 (preprint) | circuits in the MLP-neuron basis | ~200 neurons give near-perfect faithfulness; city-state-capital circuit in Llama-3.1-8B-Instruct | neurons are a fair non-SAE comparator |
| SAEScientist-Bench 2026 (preprint) | agents pick Gemma Scope features for 20 concepts | causal steering: best agent 31.47 vs expert 57.75; formatting / substring false positives | agents are weakest exactly at the causal step we grade |
| AuditBench 2026 / auditing agents 2025 | agents with SAE tools find hidden behaviours | SAE gives "limited signal"; 13% single agent, 42% super-agent (2025) | a "tool-to-agent gap": good tools do not guarantee agent success |
| d_model concept erasure | agents invent erasure; hidden held-out probe | SVM acc LEACE 88% vs agents 70.1%; 50/50 concepts; 560 rollouts | rerun-the-model grading on held-out data works as an RL reward |

## 3. Entries

### A. Targeted latent ablation benchmarks and how they score

**SAEBench** (Karvonen, Rager, Lin et al.; arXiv 2503.09532; [peer-reviewed] ICML 2025).
<https://arxiv.org/abs/2503.09532>
- Claim: proxy metrics (sparsity vs reconstruction) "do not reliably indicate performance on downstream tasks".
  Matryoshka SAEs beat other architectures on disentanglement (SCR, RAVEL, sparse probing) by "30-40%".
  Most architectures get *worse* on disentanglement as width grows ("inverse scaling").
- Scoring:
  - **SCR** = (A_abl - A_base) / (A_oracle - A_base). Probe accuracy after ablating the top-n latents chosen by
    probe attribution to the spurious signal, normalised by an oracle. Datasets: Bias in Bios (gender spurious) and
    Amazon reviews. k in {5, 10, 20, 50, 100, 500}, main results at k = 20.
  - **TPP** = mean over i = j of (A_ij - A_j) - mean over i != j of (A_ij - A_j): the drop of the target class's
    probe minus the drop of the other classes' probes, when the latents for class i are ablated. Latents are picked
    by probe importance. Main k = 20.
  - **Unlearning**: latents are chosen by firing frequency on the forget set (WMDP-bio) after discarding latents above a
    retain-set frequency threshold (0.001 or 0.01). Top 10 or 20 latents, clamped to negative values with multipliers
    25-200. The score is the lowest WMDP-bio accuracy reached while MMLU subsets stay >= 0.99 of the original. The
    paper notes Gemma-2-2B was usable on only one of the unlearning test sets.
- Gemma Scope numbers: the paper has them in an appendix. I did not extract per-SAE values (the Neuronpedia page
  shows them interactively). Not quoted here.
- **Predicts for LK**: (i) TPP is the closest precedent for our Effect x Preserve with sibling groups, but it is a
  *difference*, not a product. A difference lets "break everything a bit" score above 0. (ii) The selection recipes
  (top attribution, top probe weight, frequency contrast) are the field's defaults, so they are our recipe
  baselines. (iii) The unlearning eval uses a *hard preservation threshold* (MMLU >= 0.99) plus huge negative
  clamps. A threshold on Preserve is an alternative to the KL factor in our product.

**Evaluating SAEs on Targeted Concept Erasure Tasks** (Karvonen, Rager, Marks, Nanda; arXiv 2411.18895; [preprint]).
<https://arxiv.org/abs/2411.18895>
- The origin of SCR/TPP: it automates Marks et al.'s SHIFT by replacing the human annotator with an LLM, and adds TPP.
  Claim: the metrics "effectively differentiate" SAE hyperparameters. No numbers extracted.
- **Predicts for LK**: an LLM judge picking latents by their explanations was needed to automate SHIFT. Our grader
  must not need one, and it does not: it reruns the model.

**Are Sparse Autoencoder Benchmarks Reliable?** (Chanin; arXiv 2605.18229, May 2026; [preprint] [verify]).
<https://arxiv.org/abs/2605.18229>
- Claim: TPP and SCR "fail multiple lenses at their canonical settings".
- Numbers [verify]:
  - TPP reseed coefficient of variation: 23% at top-50 and 39% at top-10. Correlation with ground truth at top-10:
    rho = -0.03. TPP *declines* during training at top-N >= 50, i.e. it prefers an untrained SAE.
  - SCR ranks the perfect oracle below 11 of 35 trained SAEs at top-10, and is negatively correlated with ground
    truth at large k (rho = -0.31 at top-50, -0.64 at top-500).
- Recommendations: multi-seed evaluation, many datasets (113 vs 5), cross-validation, report several k.
- **Predicts for LK**: ablation scores at small k are noisy. Per-instance R needs enough held-out items (our rule
  already asks >= 20 targets and >= 40 siblings), bootstrap CIs, and reference runs over several seeds of the example
  set. Also, a metric can reward the wrong thing while looking reasonable: validate R against cases where we know the
  answer (e.g. a planted edit, as in EditHunt) before trusting it.

**Sparse Feature Circuits** (Marks, Rager, Michaud, Belinkov, Bau, Mueller; arXiv 2403.19647; [peer-reviewed] ICLR 2025).
<https://arxiv.org/abs/2403.19647>
- Method: score nodes (SAE latents and *error nodes*) by attribution patching (first-order Taylor) or integrated
  gradients (10 steps, more accurate in early layers), and keep nodes above a threshold.
- SHIFT on Bias in Bios: humans inspected 46 Gemma features and removed 43 judged task-irrelevant. Balanced-set accuracy
  (Gemma-2-2B): profession 67.7 -> 76.0, gender 81.9 -> 51.5 (chance), worst group 18.2 -> 50.0. With retraining:
  95.0 / 52.4 / 92.9, against an oracle of 95.0 / 50.6 / 93.1. Pythia: worst group 24.4 -> 76.0 (88.5 profession).
- Circuit sizes: subject-verb agreement needs ~500 feature nodes vs ~50,000 neurons on Gemma-2-2B (~100x fewer).
  Removing residual-stream **error nodes** "severely disrupts the model".
- **Predicts for LK**: (i) attribution x activation is the textbook way to rank latents, so it is the reference's first
  stage *and* the obvious shortcut. (ii) The successful SHIFT edit used ~40 latents, not 5. A cap of 5 is far
  tighter than this precedent, so feasibility must show it is reachable. (iii) The error term matters in circuits.
  If the behaviour is partly in the error term, an "ablate latents, keep the error" intervention has a ceiling.

**Applying SAEs to unlearn knowledge** (Farrell, Lau, Conmy; arXiv 2410.19278; [preprint]).
<https://arxiv.org/abs/2410.19278>
- Setup: gemma-2b-it and gemma-2-2b-it, WMDP-bio, Gemma Scope-style residual SAEs.
- Claim (abstract): "negative scaling of feature activations is necessary and … zero ablating features is
  ineffective". Several features at once can unlearn several topics, "but with similar or larger unwanted
  side-effects than … RMU".
- **Predicts for LK**: our default intervention (subtract the contribution = zero ablation with the error kept) may
  barely move the answer, especially at early layers where later MLPs recompute the fact. Feasibility should test a
  bounded negative clamp as well. If only clamping works, the task must say so and cap the multiplier, otherwise
  "clamp hard" becomes a shortcut that the KL term alone must catch.

### B. SAE latents versus simple baselines

**AxBench** (Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning, Potts; arXiv 2501.17148; [peer-reviewed] ICML 2025).
<https://arxiv.org/abs/2501.17148>
- Setup: Gemma-2-2B (layers 10, 20) and 9B (layers 20, 31). Concept500 (500 concepts taken from Gemma Scope concept
  lists). Detection is scored by AUROC on held-out labelled data. Steering is scored by an LLM judge (concept,
  instruction, fluency, each 0-2, harmonic mean).
- Steering, average overall score (Table 2): Prompt 0.894; LoReFT 0.741; SFT 0.676; LoRA 0.615; ReFT-r1 0.543;
  **DiffMean 0.239; SAE 0.165; SAE-A 0.157**; LAT 0.127; PCA 0.105; Probe 0.098.
  On Gemma-2-2B: L10 DiffMean 0.297 vs SAE 0.177; L20 0.178 vs 0.151.
- Detection AUROC average (Table 1): **DiffMean 0.942**; Probe 0.940; ReFT-r1 0.938; Prompt 0.929; **SAE-A 0.917**;
  BoW 0.914; **SAE 0.695**; PCA 0.652.
  SAE = the latent found by Neuronpedia label search. SAE-A = the latent with the best AUROC on the training labels.
- **Predicts for LK**: (i) the "plain steering vector" baseline will beat SAE latent sets on raw effect. Our task
  forbids it as a submission, so it scores 0 by rule, not by merit. SPEC must say so honestly. (ii) Choosing latents
  by *data* (SAE-A) instead of by *label* (SAE) jumps detection from 0.695 to 0.917. With permuted IDs and no
  explanation database, LK forces the data route, which is the interpretability skill we want. (iii) A probe
  direction is a poor steering direction (0.098): reading and writing directions differ (see Arad et al.).

**SAEs Are Good for Steering, If You Select the Right Features** (Arad, Mueller, Belinkov; arXiv 2505.20063; [preprint] [verify]).
<https://arxiv.org/abs/2505.20063>
- Claim: latents split into "input" latents (fire on input patterns) and "output" latents (change the output). High
  scores on both rarely co-occur. Output latents concentrate in later layers (66-100% of depth). Filtering by output
  score gives "2-3x improvements".
- Numbers on AxBench, Gemma-2-9B [verify]: L20 0.546 filtered vs 0.293 unfiltered (LoRA 0.602, DiffMean 0.322);
  L31 0.470 vs 0.387 (LoRA 0.580, ReFT-r1 0.401, DiffMean 0.158).
- **Predicts for LK**: the latents that *fire* most on "Dallas" (input latents) need not be the ones whose removal
  *changes* the answer (output latents). An agent that reads top-activating examples and picks "the Texas latent"
  can fail. Checking with real ablations is the skill. It also predicts the layer-18 advantage in our frozen
  predictions.

**Steering LLMs? Actually, SAEs can outperform simple baselines** (Jørgensen, Hansen; arXiv 2605.31183, May 2026; [preprint]).
<https://arxiv.org/abs/2605.31183>
- Claim: with a supervised labelling pipeline (Stack Exchange data), calibrated F1 feature selection and a per-feature
  steering strength, SAE steering on AxBench comes "close to on par with the reference LoRA" (Gemma-2-9b-it, L17/L32).
  Still below prompting. I found no exact numbers in the extracted text.
- **Predicts for LK**: the SAE handicap shrinks a lot when latents are selected with labelled data and tuned strength.
  This is roughly what our reference does. The gap between "naive SAE" and "careful SAE" is the band our task lives in.

**Are SAEs Useful? A Case Study in Sparse Probing** (Kantamneni, Engels, Rajamanoharan, Tegmark, Nanda; arXiv 2502.16681).
<https://arxiv.org/abs/2502.16681>
- Venue: the abstract page shows arXiv only (I believe it appeared at ICML 2025, verify).
- Claim: over 113 datasets and 4 hard regimes (scarcity, imbalance, label noise, covariate shift), SAE probes do not
  consistently beat logistic-regression baselines, and SAE-based insights could be reproduced with non-SAE baselines.
- **Predicts for LK**: if the task is "find a concept", dense baselines usually suffice. The SAE must matter through
  the *sparse, submit-a-set* constraint and through generalisation over entities, not through detection power.

**Negative Results for SAEs on Downstream Tasks** (GDM mech interp team; blog, 2025-03-26; [preprint-level]).
<https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks>
- Setup: Gemma-2-9B-IT, harmful-intent detection out of distribution. The dense linear probe reaches OOD AUROC
  ~0.999; k-sparse SAE probes are "distinctly worse" OOD. Chat-data SAEs close "about half the gap". Probes trained
  on the SAE *reconstruction* are also worse, so the reconstruction loses task-relevant information.
- **Predicts for LK**: whatever the SAE does not reconstruct (the error term) can carry the behaviour, and style
  shift (OOD) is where SAEs lose most. Both are already risks in our brief.

**Sanity Checks for SAEs: Do SAEs Beat Random Baselines?** (Korznikov et al.; arXiv 2602.14111, Feb 2026; [preprint]).
<https://arxiv.org/abs/2602.14111>
- Claims: on a synthetic setup, SAEs recover 9% of the true features at 71% explained variance. On real activations,
  baselines with random directions or random activation patterns match trained SAEs: interpretability 0.87 vs 0.90,
  sparse probing 0.69 vs 0.72, **causal editing 0.73 vs 0.72**.
- **Predicts for LK**: we need a *dictionary control*: run the reference search over a random-direction dictionary of
  the same size and decoder norms. If it reaches similar R, the task tests search effort, not SAE understanding.

**Analyzing the Generalization and Reliability of Steering Vectors** (Tan, Chanin, Lynch, et al.; arXiv 2407.12404; [preprint]).
<https://arxiv.org/abs/2407.12404>
- Claim: steerability is "highly variable across different inputs" in distribution, and for several concepts steering
  vectors are "brittle to reasonable changes in the prompt" out of distribution.
- **Predicts for LK**: the plain steering baseline will also lose effect on style-varied held-out prompts. Compare
  SAE and dense *on the same style split*, not dense-on-plain vs SAE-on-styled.

**Refusal is mediated by a single direction** (Arditi et al.; arXiv 2406.11717; I believe NeurIPS 2024, verify).
<https://arxiv.org/abs/2406.11717>
- Claim: across 13 open chat models up to 72B, projecting one DiffMean direction out of the residual stream removes
  refusal "with minimal effect on other capabilities".
- **Predicts for LK**: "directional ablation of the DiffMean direction at every position" is the cleanest non-SAE
  comparator for a *knockout*. It is the right baseline, more so than additive steering.

**LEACE** (Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman; arXiv 2306.03819).
<https://arxiv.org/abs/2306.03819>
and **d_model, Discovering Concept-Editing Algorithms With LLM Agents** (blog; [preprint-level]).
<https://dmodel.ai/concept-erasure/>
- d_model setup: Gemma-3 270M, 50 concepts, 560 rollouts. Agents get a concept, a small labelled sample and the
  model, but not the grader. The grader is a fresh nonlinear probe (RBF-SVM; random forest as a check) on hidden
  held-out activations, with an L2 edit budget matched to LEACE. Results: SVM accuracy LEACE 88% vs best agent
  algorithm 70.1%; random forest 82.6% vs 71.9%; the agents beat LEACE on 50/50 concepts. (CONTEXT.md says "99% to 70%".
  The 99% is the pre-erasure accuracy given on the same page.)
- **Predicts for LK**: a grader that reruns a fresh evaluator on hidden held-out data, with a budget matched to a
  closed-form baseline, is an accepted RL-environment design by the target lab. Our analogue: hidden held-out prompts
  plus a cap matched to what the reference needs.

### C. Causal variable localisation with keep / locality readouts

**RAVEL** (Huang, Wu, Potts, Geva, Geiger; arXiv 2402.17700; [peer-reviewed] ACL 2024).
<https://arxiv.org/abs/2402.17700> · code <https://github.com/explanare/ravel>
- Scoring: Cause = share of interventions that change the target attribute to the source entity's value.
  Iso = share where the other attributes stay unchanged. **Disentangle = (Cause + Iso) / 2.** Entities: city (country,
  continent, language, latitude, longitude, timezone), Nobel laureate, verb, physical object, occupation.
- Generalisation splits: an **entity split** (50/25/25 train/dev/test entities, same templates) and a **context
  split** (50/25/25 templates, same entities).
- Numbers (Llama2-7B, Disentangle, entity / context split): PCA 39.5 / 39.1; **SAE 48.6 / 46.8**; RLAP 48.8 / 50.9;
  DBM 52.2 / 49.8; DAS 56.5 / 57.3; MDBM 53.7 / 53.9; **MDAS 60.1 / 65.6**.
  "Increasing feature dimensions generally leads to higher Cause score, but lower Iso score."
- **Predicts for LK**: (i) Our Effect/Preserve is RAVEL's Cause/Iso, with ablation in place of interchange. (ii) RAVEL
  *averages*, so "break everything" scores 0.5. Our product (plan v2) correctly gives ~0. (iii) The Cause/Iso
  trade-off over dimension count is the precedent for our cap k as a dial. We should plot Effect and Preserve against
  k for the reference and set k at the knee. (iv) Use both an entity split and a context (template/style) split for
  the held-out sets, and report them separately.

**Evaluating open-source SAEs on disentangling factual knowledge in GPT-2 small** (Chaudhary, Geiger; arXiv 2409.04478; [preprint]).
<https://arxiv.org/abs/2409.04478>
- Setup: RAVEL city country vs continent; four open GPT-2-small SAEs; a learned binary mask selects latents to patch
  so as to change country but not continent (or the reverse). Baseline = neurons, skyline = DAS.
- Claim: SAEs "struggle to reach the neuron baseline, and none come close to the DAS skyline".
- **Predicts for LK**: our closest analogue (city -> state while keeping other relations) is a case where SAE latents
  are weaker than even neurons. Expect some (group, layer) cells to be honestly infeasible at small k.

**MIB: A Mechanistic Interpretability Benchmark** (Mueller et al.; arXiv 2504.13151; [peer-reviewed] ICML 2025).
<https://arxiv.org/abs/2504.13151>
- Two tracks:
  - **Causal variable localisation**: IIA on IOI, arithmetic, MCQA, ARC-easy and RAVEL, over Llama-3.1-8B,
    Gemma-2-2B, Qwen-2.5-0.5B and GPT-2. Featurisers: full vector, DAS, DBM, PCA, SAE.
  - **Circuit localisation**: CPR = area under the faithfulness curve; CMD = area between that curve and f = 1.
- Findings: "SAE and PCA features generally fail to improve upon neurons". DAS is best for variables. EAP-IG (with
  counterfactual ablations) is best for circuits. Examples [verify]: MCQA on Gemma-2, DAS 95 (answer) / 77 (order) vs
  SAE 73 / 51; arithmetic carry on Llama-3.1, DAS 54 vs SAE 38.
- Each task has a **public and a private test set**. The private set is evaluated only on upload.
- Follow-up: the BlackboxNLP 2025 shared task on MIB (arXiv 2511.18409) got its gains in variable localisation from
  "low-dimensional and non-linear projections", not from SAEs. <https://arxiv.org/abs/2511.18409>
- **Predicts for LK**: a private held-out set is standard practice. The SAE basis is not privileged for causal
  localisation, so LK is justified only as a test of *using an SAE well*, not as a claim that the SAE is the right basis.

**Towards Principled Evaluations of SAEs for Interpretability and Control** (Makelov, Lange, Nanda; arXiv 2405.08366; [preprint]).
<https://arxiv.org/abs/2405.08366>
- Setup: IOI in GPT-2 small, comparing SAEs with *supervised* feature dictionaries. SAEs capture interpretable IOI
  features but are "less successful than supervised features in controlling the model". Two failure modes: **feature
  occlusion** (a causally relevant concept overshadowed by a slightly larger one) and **feature over-splitting**
  (binary features split into many small ones).
- **Predicts for LK**: over-splitting means a group concept ("Texas") may be spread over many small latents, so the
  cap bites. Occlusion means the obvious big latent may not be the causal one.

### D. Feature splitting, absorption, completeness, and measurement artefacts

**A is for Absorption** (Chanin, Wilken-Smith, Dulka, Bhatnagar, Golechha, Bloom; arXiv 2409.14507; [peer-reviewed] NeurIPS 2025, oral).
<https://arxiv.org/abs/2409.14507>
- Setup: first-letter identification on Gemma-2-2B with Gemma Scope residual SAEs (16k and 65k, layers 0-17), plus
  Qwen2-0.5B and Llama-3.2-1B.
- Claims: absorption "occurs in every LLM SAE we tested". Wider and sparser SAEs absorb more. "Varying SAE sizes or
  sparsity is insufficient to solve this issue." Example [verify]: the "starts with S" latent (F1 0.81) stays silent on
  " short", where a token-aligned " short" latent fires ~55x more strongly and carries the information.
- **Predicts for LK**: a state latent will have *false negatives* on some cities, often the famous ones that have their
  own latents. Held-out-entity Effect directly measures this. An agent who checks firing on many entities (not just
  the examples) wins, and that is real interpretability skill. It also means some groups need several latents
  (state + big-city latents), which ties feasibility to the cap.

**Scaling Monosemanticity** (Templeton et al., Anthropic, 2024; Transformer Circuits; [not peer reviewed]).
<https://transformer-circuits.pub/2024/scaling-monosemanticity/>
- Claim (feature completeness): whether a concept has its own feature depends on its training frequency. "If a concept
  is present in the training data only once in a billion tokens, then we should expect to need a dictionary with on the
  order of a billion alive features". Only ~60% of London boroughs had a matching feature in the 34M SAE.
- **Predicts for LK**: with 16k latents, frequent groups (Texas, California, Spanish, football) probably have latents,
  while rare ones (small states, minor languages) may not. That gives **natural, honest "cannot be done within k"
  instances** whose infeasibility follows from frequency, not from a planted label.

**SAEs trained on the same data learn different features** (Paulo, Belrose; arXiv 2501.16615; [preprint]).
<https://arxiv.org/abs/2501.16615>
- Claim: in a 131K-latent SAE on a Llama-3-8B MLP, only 30% of features were shared across seeds. TopK SAEs are more
  seed-dependent than ReLU/L1 SAEs.
- **Predicts for LK**: evaluating an RL policy with a *different* Gemma Scope dictionary for the same layer (another L0
  variant, or 65k width) tests the method, not memorised latent IDs (FeatureMatch lesson 3). It is cheap because the
  weights are public.

**Where You Measure Decides What You Measure** (Noël; arXiv 2608.13337, Aug 2026; [preprint]).
<https://arxiv.org/abs/2608.13337>
- Claim: ablation-based SAE evaluations usually measure at the token where the latent fires hardest, a choice made by
  the dictionary. Position explains 7.6% and 11.9% of the variance that looks like "dictionaries disagree".
- **Predicts for LK**: fix and document the position rule (ablate at every non-BOS position, read the answer at the
  last token), and use the same rule for every baseline.

**From Geometric Recovery to Causal Validation** (Bal; arXiv 2607.12166, Jul 2026; [preprint]).
<https://arxiv.org/abs/2607.12166>
- Claim: up to 77% (degraded SAE) and 9% (well-trained SAE) of recovered features are "causally inert". 14% were inert
  in a production SAE. Some atoms are read-inert but steerable.
- **Predicts for LK**: a latent that looks right by its activations can have zero ablation effect. Again, choosing by
  real ablation is the skill being tested.

### E. Gemma Scope, gemma-2-2b, and this exact behaviour

**Gemma Scope** (Lieberum, Rajamanoharan, Conmy, et al.; arXiv 2408.05147; tech report).
<https://arxiv.org/abs/2408.05147>
- Evaluations reported: delta LM loss when the SAE is spliced in, fraction of variance unexplained, L0, a human rater
  study, and LLM auto-interp. JumpReLU, TopK and Gated SAEs show "little discernible difference" in interpretability.
  **No downstream causal or steering evaluation.** "Using SAEs to improve performance on real-world tasks (compared to
  fair baselines)" is listed as an open problem. I did not find the exact delta loss for 16k residual SAEs at
  layers 6/12/18 near L0 70 in the text. Our frozen prediction of ~0.1-0.25 nats remains unverified.
- **Predicts for LK**: nobody has published a targeted-ablation evaluation of these exact SAEs on factual recall with
  held-out entities. That gap makes LK a useful contribution, but it also means there is no prior number to calibrate against.

**On the Biology of a Large Language Model** and **Circuit Tracing** (Lindsey, Ameisen, et al., Anthropic 2025; Transformer Circuits).
<https://transformer-circuits.pub/2025/attribution-graphs/biology.html>
- Model: Claude 3.5 Haiku, prompt "Fact: the capital of the state containing Dallas is" -> Austin. Supernodes: Dallas,
  Texas, "say a capital", "say Austin".
- Interventions: inhibiting the Dallas cluster lowers the Texas features and makes the model output other state
  capitals. Swapping in California / Georgia / British Columbia / China / Byzantine Empire features gives Sacramento /
  Atlanta / Victoria / Beijing / Constantinople. Some swaps needed larger injections. There is also a **direct "shortcut"
  edge from Dallas to Austin**, and error nodes matter.
- Open-source follow-up: `circuit-tracer` (May 2025) supports gemma-2-2b with transcoders, and the ARENA curriculum
  includes the Dallas/Austin circuit on gemma-2-2b with zero ablation and feature swapping.
  <https://www.anthropic.com/research/open-source-circuit-tracing> · <https://github.com/decoderesearch/circuit-tracer> ·
  <https://learn.arena.education/chapter1_transformer_interp/22_sae_circuits/>
- **Predicts for LK**: (i) the flagship behaviour is the most famous example in the field. Frontier agents know the
  "Texas supernode" story, so they start with a strong prior about *what* to look for, though not *which latent IDs*.
  Do not use the Dallas prompt itself as an example. (ii) The shortcut edge predicts that ablating state latents alone
  leaves some of the "Austin" answer. For the two-hop family, sets may need both state and city-level latents.

**Language Model Circuits Are Sparse in the Neuron Basis** (Arora, Wu, Steinhardt, Schwettmann; arXiv 2601.22594, 2026; [preprint] [verify]).
<https://arxiv.org/abs/2601.22594>
- Claim: MLP neurons are "as sparse a feature basis as SAEs". About 200 neurons reach near-perfect faithfulness, and
  ~10^2 neurons control subject-verb agreement. On the city-state-capital task in Llama-3.1-8B-Instruct, a 257-neuron
  circuit was filtered to 23 meaningful neurons. Steering a single "say a capital" neuron flips the output from capital
  to state on most of 50 questions.
- **Predicts for LK**: a neuron-basis search is a fair non-SAE comparator for the same task. If it does as well, LK
  measures "find a small causal set" and not anything SAE-specific. That is acceptable, but SPEC should say it.

**Do I Know This Entity?** (Ferrando, Obeso, Rajamanoharan, Nanda; arXiv 2411.14257; [peer-reviewed] ICLR 2025).
<https://arxiv.org/abs/2411.14257>
- Claim: Gemma Scope latents encode entity recognition ("known" vs "unknown" entity) across entity types (players,
  films, songs, cities). Steering with the unknown-entity latent induces "almost 100%" refusal in Gemma 2 chat. These
  latents disrupt the attention heads that move attributes to the last token.
- **Predicts for LK**: there are *generic recall-breaking* latents. Ablating "known entity" latents would knock out the
  target group and every sibling group alike. Our product reward zeroes this through Preserve, which is a good test
  that the reward works. These latents are also likely to appear in a naive "most-active" or "top attribution" pick.

**Scaling Sparse Feature Circuits For Studying In-Context Learning** (Kharlapenko, Shabalin, Conmy, Nanda; [peer-reviewed] ICML 2025).
<https://proceedings.mlr.press/v267/kharlapenko25a.html>
- Claim: sparse feature circuits scale to Gemma-1-2B. Task-detecting SAE latents causally induce tasks zero-shot.
- **Predicts for LK**: a non-factual family (e.g. language identification, past tense) has precedent for single
  task-level latents being causal. That family may be easier, and it may also be more recipe-solvable.

### F. Agents doing SAE-based interventions, and automated discovery as a benchmark

**SAEScientist-Bench** (Tan, He, Zhao, Liu; arXiv 2609.09113, Sep 2026; [preprint] [verify]).
<https://arxiv.org/abs/2609.09113>
- Setup: agents get target concepts and a `probe_sae` tool (up to 64 agent-written texts per call) over Gemma Scope
  features of Gemma-2-9B-IT (layers 9, 20, 131K-wide). There are 20 tasks: languages, document formats, domains, "cat".
  Scores: activation rank against an expert feature; selectivity = 100 x max(0, 2·AUROC - 1) against hard negatives;
  causal steering judged by an LLM (0-4) over a control.
- Numbers [verify]: overall best agent 65.82 vs expert 85.56. **Causal steering best agent 31.47 vs expert 57.75**
  (the largest gap). Selectivity 92.91 vs 98.92.
- Failure modes observed: a feature driven by "clinical document formatting rather than actual symptoms"; a "cat"
  candidate firing on "Copycat" and "Catalytic converter"; a perfect AUROC on a feature with weak activation.
- **Predicts for LK**: (i) the causal step is where agents fall short, so a grader that reruns the model with the
  agent's ablation targets the hardest part. (ii) Format and substring latents are a known agent trap, the same as
  FeatureMatch lesson 2. Style-varied held-out prompts will punish them. (iii) That benchmark's steering grade uses an
  LLM judge and frozen expert features. LK avoids both: it reruns the model, and its references are found by probing.

**AuditBench** (Anthropic, 2026; arXiv 2602.22755 and blog) and **Building and evaluating alignment auditing agents** (Anthropic, 2025; blog).
<https://alignment.anthropic.com/2026/auditbench/> · <https://arxiv.org/abs/2602.22755> ·
<https://alignment.anthropic.com/2025/automated-auditing/>
- AuditBench [verify]: 56 models with implanted behaviours in 14 categories. The investigator agent gives up to 10 hypotheses,
  scored by an LLM classifier. The default agent succeeds ~37%, scaffolded black-box tools >50%. SAEs give "limited
  signal, though SAEs occasionally identify directly relevant feature clusters". Key finding, the **tool-to-agent
  gap**: tools that surface evidence in static tests often do not improve agent success.
- 2025 auditing agents [verify]: single investigator 13% (17% when seeded), super-agent 42%. Interpretability tools raise
  success by 10-20 points from near 0%. The SAE's training data mattered: with an SAE trained on all data the agent
  "fails to ever win", because the key feature drops to rank 181.
- **Predicts for LK**: whether an agent can *use* a generic SAE tool well is itself the uncertain, measurable skill,
  which supports mid-band difficulty. The training data of the SAE (Gemma Scope: pretraining text) shapes what is
  findable, so style shift matters.

**AgenticInterpBench** (Khan, Kohli, Yao, Sun, Yao; arXiv 2606.24026, 2026; [preprint]).
<https://arxiv.org/abs/2606.24026>
- 84 semi-synthetic circuits with 163 component annotations. The agent (HyVE) explains components once the circuit is
  given. "Reliable validation remains the key obstacle." Failures come from incomplete validation plans, code errors
  and unresolved hypotheses.
- **Predicts for LK**: expect agent failures in the *verification* phase (testing the set on new entities and
  controls), not in finding candidates.

**Pitfalls in Evaluating Interpretability Agents** (Haklay, Prakash, Pandey, Torralba, Mueller, Andreas, Rott Shaham, Belinkov; arXiv 2603.20101, 2026; [preprint]).
<https://arxiv.org/abs/2603.20101>
- Claim: replication-based evaluation of circuit-analysis agents looks competitive with human experts, but LLM systems
  "may reproduce published findings via memorization or informed guessing". The authors propose an intrinsic
  evaluation (functional interchangeability).
- **Predicts for LK**: famous behaviours leak priors. Our grader is intrinsic (rerun the model), which is the remedy the
  paper proposes, but behaviours and groups should not all be textbook cases.

**InterpBench** (Gupta, Arcuschin, Kwa, Garriga-Alonso; arXiv 2407.14494; [peer-reviewed] NeurIPS 2024 D&B).
<https://arxiv.org/abs/2407.14494>
- Semi-synthetic transformers with known circuits (SIIT training) used to evaluate circuit discovery methods.
- **Predicts for LK**: a planted ground truth is the other way to validate a grader. EditHunt-style planted edits could
  serve as LK's sanity check that R ranks a known-correct set above near-misses.

**Automated Interpretability and Feature Discovery with Agents** (Marin-Llobet, Ferrando; arXiv 2605.01555, 2026; [preprint]).
<https://arxiv.org/abs/2605.01555>
- A multi-agent loop over Gemma-2 SAE features that writes its own prompt controls and uses separability criteria.
  It "improves over one-shot auto-interpretations".
- **Predicts for LK**: writing one's own contrast prompts is a learnable agent behaviour, and LK's tool set (run
  model, encode, ablate, top activations) supports it.

## 4. What this says about the frozen feasibility predictions (no edits made there)

- **Plain steering vector ≈ 0.95 x R_ref**: consistent with AxBench, RAVEL, MIB and Chaudhary & Geiger, where dense
  supervised directions beat SAE latents. If anything, the literature suggests the dense comparator may *exceed* the
  SAE reference (ratio > 1).
- **Naive top-5 attribution reaching ≥ 0.5 x R_ref on ~55% of cells (NO-GO-as-is with p ≈ 0.4)**: the literature
  makes this more plausible. Attribution plus top-k is the standard selection in SAEBench SCR/TPP and Marks et al.
  It is the field's default because it usually works.
- **Layer 6 weak because of recomputation**: supported by Farrell et al. (zero ablation ineffective) and by Arad et al.
  (output latents concentrate in late layers).
- **Error term not dominant (P > 50% only 0.15)**: Marks et al. and GDM suggest the error term often matters more
  than that. This is the prediction most at risk.
- **Feature splitting "occurs but is not dominant" for the reference**: absorption is universal (Chanin et al.) and
  over-splitting was seen on IOI (Makelov et al.). Expect more entity-specific picks than predicted, especially for
  famous cities.
- **Style lowers R by ~20% relative**: consistent with GDM (SAEs lose most OOD) and with Tan et al. (steering
  brittleness). Not contradicted.

## 5. Design implications for LatentKnockout

1. **Treat the SAE as a declared handicap and measure it.** Add two non-submittable dense comparators at the same layer
   and positions, on the same held-out splits: (a) directional ablation of the DiffMean direction (Arditi-style
   projection), (b) a LEACE-style projection. Report R_dense next to R_ref. SPEC must say plainly that a steering vector
   scores 0 because it is not a valid answer, not because it fails (AxBench: DiffMean 0.239 vs SAE 0.165; RAVEL: SAE
   48.6 vs MDAS 60.1). Normalise difficulty per instance by R_ref (found by probing), never by R_dense. A cell where
   even the reference stays far below dense (e.g. R_ref < 0.5 x R_dense and below tau) is a candidate for an honest
   "cannot be done within the cap" instance.
2. **Settle the ablation semantics in feasibility, not by assumption.** Test the brief's "subtract the contribution"
   (zero ablation, error kept) against a bounded negative clamp (-c x f_i, c in {1, 2, 4}). Farrell et al. found zero
   ablation ineffective, and SAEBench unlearning clamps at -25 to -200x. If only clamping works, expose a single
   bounded multiplier per submission (c <= c_max) and keep the KL factor, so that "clamp everything hard" cannot pass.
3. **Use a product of Effect and Preserve and pick k from the reference's trade-off curve.** RAVEL averages Cause and
   Iso (so "break everything" earns 0.5), and TPP subtracts. Our product sends both shortcuts to ~0. RAVEL shows that
   more dimensions raise Cause and lower Iso. Plot reference Effect and Preserve against k in {1, 2, 3, 5, 8, 10} and
   set the cap near the knee. Note that precedents use k = 20 (SAEBench) or ~40 (SHIFT), so k <= 5 needs feasibility
   evidence and a stated reason (e.g. "the reference reaches tau at k <= 5 on X% of cells").
4. **Gate on the field's own selection recipes, run as baselines.** Run (a) top-k attribution (gradient x activation)
   on the examples, (b) top-k probe weight (SAEBench SCR/TPP style), (c) forget/retain frequency contrast (SAEBench
   unlearning style), (d) most-active, (e) random. If any reaches ≥ 0.5 x R_ref on more than a third of feasible cells,
   the difficulty is not coming from interpretability. The precedented ways to defeat them are sibling-group
   preservation (TPP / RAVEL Iso), same-entity-other-relation preservation (RAVEL Iso; EditHunt Hard tier) and
   style-varied held-out prompts (RAVEL context split; GDM OOD).
5. **Hold out entities, templates and styles, keep them private, and report each split separately.** Follow RAVEL's
   entity and context splits and MIB's private test set. Fix and document the position rule (Noël 2026). Report
   Effect per split, so a failure can be attributed to entity generalisation (splitting or absorption) or to style. Use
   ≥ 20 target and ≥ 40 sibling items per cell, bootstrap CIs, and reference runs over several example seeds, because
   small-k ablation metrics are noisy (Chanin 2026: TPP CV 23-39%).
6. **Audit what selected latents fire on before trusting any rule, with an automated check.** For every reference and
   recipe pick, record the top contexts on a mixed-style corpus, the share of held-out target entities on which it
   fires (the absorption false-negative rate), and its firing rate on sibling groups. Known traps from the literature:
   formatting and substring latents (SAEScientist-Bench, FeatureMatch), generic "known entity" latents that break all
   recall (Ferrando et al.), and input latents that fire but do not change outputs (Arad et al.). The product reward
   should zero the generic ones through Preserve. Verify that it does.
7. **Run dictionary controls to show the SAE is doing work.** Run the same reference search over (a) a random-direction
   dictionary with matched decoder norms and sparsity (Korznikov et al.: random matches trained on causal editing,
   0.73 vs 0.72) and (b) another Gemma Scope variant for the same layer. If (a) reaches similar R, LK tests search
   effort, not SAE understanding. Then say so or redesign. If (b) works, it is a ready-made held-out dictionary.
8. **Plan against memorisation and priors from the start.** For RL evaluation, hold out whole groups and at least one
   family (FeatureMatch lesson 3), permute latent IDs per episode, and optionally evaluate on a different Gemma Scope
   L0/width variant. Only ~30% of features survive a reseed (Paulo & Belrose), so memorised IDs will not transfer. Avoid
   the textbook Dallas -> Austin prompt in examples, because agents know that story (Biology paper; Pitfalls 2026).
   Use rare groups, where the 16k dictionary probably has no clean latent (Scaling Monosemanticity completeness), as
   the honest "cannot be done within the cap" instances.

## 6. Sources (all opened 2026-10-02)

- SAEBench: <https://arxiv.org/abs/2503.09532>; Neuronpedia results <https://www.neuronpedia.org/sae-bench/info>
- Karvonen et al., targeted concept erasure: <https://arxiv.org/abs/2411.18895>
- Chanin 2026, SAE benchmark reliability: <https://arxiv.org/abs/2605.18229>
- Marks et al., Sparse Feature Circuits: <https://arxiv.org/abs/2403.19647>
- Farrell et al., SAE unlearning: <https://arxiv.org/abs/2410.19278>
- AxBench: <https://arxiv.org/abs/2501.17148> (PMLR: <https://proceedings.mlr.press/v267/wu25a.html>)
- Arad et al., output features: <https://arxiv.org/abs/2505.20063>
- Jørgensen & Hansen 2026: <https://arxiv.org/abs/2605.31183>
- Kantamneni et al., sparse probing: <https://arxiv.org/abs/2502.16681>
- GDM negative results: <https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks>
- Korznikov et al., random baselines: <https://arxiv.org/abs/2602.14111>
- Tan et al., steering vector reliability: <https://arxiv.org/abs/2407.12404>
- Arditi et al., refusal direction: <https://arxiv.org/abs/2406.11717>
- LEACE: <https://arxiv.org/abs/2306.03819>; d_model concept erasure: <https://dmodel.ai/concept-erasure/>
- RAVEL: <https://arxiv.org/abs/2402.17700>; <https://github.com/explanare/ravel>
- Chaudhary & Geiger: <https://arxiv.org/abs/2409.04478>
- MIB: <https://arxiv.org/abs/2504.13151>; BlackboxNLP 2025 shared task: <https://arxiv.org/abs/2511.18409>
- Makelov et al.: <https://arxiv.org/abs/2405.08366>
- A is for Absorption: <https://arxiv.org/abs/2409.14507>
- Scaling Monosemanticity: <https://transformer-circuits.pub/2024/scaling-monosemanticity/>
- Paulo & Belrose: <https://arxiv.org/abs/2501.16615>
- Noël 2026, position selection: <https://arxiv.org/abs/2608.13337>
- Bal 2026, causal inertness: <https://arxiv.org/abs/2607.12166>
- Gemma Scope: <https://arxiv.org/abs/2408.05147>
- Biology of an LLM: <https://transformer-circuits.pub/2025/attribution-graphs/biology.html>; circuit-tracer
  <https://github.com/decoderesearch/circuit-tracer>; <https://www.anthropic.com/research/open-source-circuit-tracing>;
  ARENA <https://learn.arena.education/chapter1_transformer_interp/22_sae_circuits/>
- Arora et al. 2026, neuron-basis circuits: <https://arxiv.org/abs/2601.22594>
- Ferrando et al., entity recognition: <https://arxiv.org/abs/2411.14257>
- Kharlapenko et al.: <https://proceedings.mlr.press/v267/kharlapenko25a.html>
- SAEScientist-Bench: <https://arxiv.org/abs/2609.09113>
- AuditBench: <https://alignment.anthropic.com/2026/auditbench/>, <https://arxiv.org/abs/2602.22755>;
  auditing agents 2025: <https://alignment.anthropic.com/2025/automated-auditing/>
- AgenticInterpBench: <https://arxiv.org/abs/2606.24026>
- Pitfalls in Evaluating Interpretability Agents: <https://arxiv.org/abs/2603.20101>
- InterpBench: <https://arxiv.org/abs/2407.14494>
- Marin-Llobet & Ferrando 2026: <https://arxiv.org/abs/2605.01555>
