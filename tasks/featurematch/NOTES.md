# FeatureMatch lab notebook (append-only; timestamps from `date -u`)

## 2026-10-01 15:06 UTC — kickoff and design choices
- Read CONTEXT, PLAN_PROMPT, PLAN_V2_DELTA, BUILDER_GUIDE, HARNESS_API, DECISIONS. common/toolserver etc. are not
  on main yet (no READY file), so tools.py imports them with a local fallback and gates run in-process (local_shim.py).
- **Concept universe** (no LLM API, so no generated text): two public labelled HF datasets.
  * DBPedia_Classes (Wikipedia abstracts, 3-level ontology l1>l2>l3): each l3 class (e.g. Swimmer < Athlete < Agent)
    is a TOPIC concept. WHY: the ontology gives real taxonomy siblings for close distractors (27 athlete types,
    8 animal classes, 7 politician types ...), which the spec asks for.
  * papluca/language-identification: 19 LANGUAGE concepts (English dropped: every DBPedia text is English, so an
    "English" concept would be confounded with genre). Language families give siblings (Romance, Slavic, ...).
- **Splits**: per concept 40 held-out texts (split A, generator-only: selects latents + ground truth) and 24 probe
  texts (split B, the reference solver's own corpus, standing in for texts an LLM agent would write). WHY: the
  answer must be defined on data the agent can never query, otherwise the agent could score menu options on the
  grading data itself.
- **Held-out cleanliness check** (concepts.py): global near-duplicate removal (a text seen in two concepts is
  dropped from both), A/B disjointness asserted, and a bag-of-words classifier trained on B must reach AUROC >= 0.9
  on A per concept. Result: 238 candidate concepts, 6 dropped for too few clean texts (<64), 0 failed the label
  check, **232 kept (19 languages, 213 topics)**. Details: cache/concepts_validation.json (regenerable).
- **Per-text latent summary** = max activation over the text's tokens, excluding BOS (BOS has huge norm and
  spurious activations). Texts truncated to 64 tokens everywhere (generator, tools, solver) so they agree.
- First precompute attempt was killed by the gpuq RAM cap (8.9 > 8 GB) while loading gemma-2-2b on CPU first.
  Fix: load with device_map=cuda directly.

## 2026-10-01 15:32 UTC — precompute, SAE sanity, latent pools
- Converted gemma-2-2b to a local bf16 copy (convert_bf16.py): the hub checkpoint is fp32 (10.5 GB) and its memory
  map pushed RSS to 11 GB > the 8 GB gpuq RAM cap even with device_map=cuda. **Integrator: tool servers must load
  the bf16 copy (fm_core.model_path() does this automatically) or they will be killed by the RAM cap.**
- First precompute OOMed on the 256k-vocab LM head (logits for 32x65 tokens). Fix: run the base model only; later
  also hook block outputs and stop the forward after the deepest needed block (early exit) for speed.
- SAE convention check (single unpadded texts; padded batches pollute the statistic): Gemma Scope layer-L SAE
  reconstructs HF hidden_states[L+1] (output of block L) far better than hidden_states[L]:
  FVE L6 0.74 vs 0.54, L12 0.85 vs 0.78, L18 0.60 vs 0.30; L0 ~80-100 (expected ~70-80). So residual = output of block L.
  Batched (left-padded) activations agree with unbatched/precomputed ones to corr 0.999 (bf16 noise).
- AUROC tables (auroc.py): per latent x concept, max-pooled activation, concept texts vs all other concepts' texts
  (the whole universe, so a concept's AUROC does not depend on the menu it appears in).
- Candidate pools: planted-eligible latents (best AUROC_A >= 0.90, AUROC_B >= 0.85, fires on >= 50% of the
  concept's held-out texts, <= 3 concepts within 0.10 of the best): L6 2210, L12 2631, L18 2508. Off-universe null
  latents (all AUROC < 0.65 on A and < 0.70 on B, alive on >= 1% of texts): 1654 / 1384 / 1335.
- **Surprise:** solar-eclipse articles own >2000 selective latents (formulaic, number-heavy texts), US Supreme Court
  cases 386, Eurovision entries 345. Sampling latents uniformly would make "guess the popular concept" work, so slots
  sample the anchor CONCEPT uniformly first, then a latent.
- Naive token-identity recipe before filtering: encoding all 232 concept names and taking the argmax picks the right
  concept for ~25% of planted-eligible latents (588/2284 L6, 602/2692 L12, 586/2586 L18; over the whole universe,
  any of 3 surface variants). Within a 20-option menu it would be higher. Generator now REJECTS a planted slot whenever
  any variant picks the answer, so on kept planted slots the recipe is at chance (~5% = 1/20, only when all names
  give zero and it defaults to option 1).

## 2026-10-01 15:32 UTC — reference solver design iterations (offline simulator, simulate.py)
- v0: K=6 texts/option from split B, AUROC vs other options + 32 background texts, tau 0.80 -> slot accuracy planted
  T1 0.91 / T2 0.81; one-shot episode pass 0.68 / 0.50. Too noisy.
- Two-stage (5 texts/option + 32 bg, then 15 more for the top-3 options + 32 more bg; ~850 forward units for 4 slots)
  and AUROC against BACKGROUND ONLY (estimates exactly what ground truth measures: option vs everything) -> much better.
- Failure analysis: most remaining failures were "family" latents (e.g. Insect 0.91 vs Moss/Fungus 0.78) or slots
  whose verdict flipped between splits. Added robustness filters: a planted slot must also win on split B by >= 0.08,
  and a null menu must have every option < 0.72 on split B.
- **Honesty fix:** the reference originally probed with split B, the same split used by those filters (circular).
  Added a third disjoint split C (20 texts/concept; 220/232 concepts have enough clean texts; the other 12 are never
  menu options but stay as AUROC negatives) and the reference now uses only C. Offline: one-shot T1 0.98 / T2 0.92,
  best-of-5 0.98 / 0.95; slot accuracy planted 0.99 / 0.96, nulls 1.00.
- Remaining reference failures are idiosyncratic latents whose C-split texts do not elicit them; these are dropped
  by the best-of-5 rule.

## 2026-10-01 15:32 UTC — fingerprinting check (fingerprint_check.py)
- Gradient-boosted classifier on PUBLIC menu-structure features (layer, #languages, #distinct l1/l2 groups,
  largest group, taxonomy-closeness stats) predicting planted vs null, 5-fold CV AUROC:
  * first close-tier design (menu = answer + its 19 nearest siblings; null = 20 nearest to the removed answer):
    T1 0.49, **T2 0.74** -> a learned policy could detect nulls from the menu alone. Fail.
  * fix 1: menu centred on a RANDOM member of the anchor's sibling group, with noise (1.5 taxonomy levels) on the
    ranking, so sibling groups are incomplete in both kinds: T2 0.61.
  * fix 2: draw the anchor concept the same way for planted and null slots (uniform over concepts that own selective
    latents), retry latents for the same anchor: **T1 0.50-0.54, T2 0.58-0.60**.
  * residual signal: near-miss null menus contain fewer languages (a language latent's relatives often clear the
    0.65 null threshold and must be excluded). Documented as a known weakness: it can only shift the null/planted
    prior per slot; it never says WHICH option is right, and pass needs all slots right.
