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

## 2026-10-01 15:33 UTC — rabbit-hole check at ~70% of the time box
1. *Still on the path to a validated, verifiable task?* Yes. Ground truth is a fixed, held-out AUROC computation. The
   grader is exact match. Offline the reference is at ~0.98 (T1) / ~0.92 (T2) one-shot, and every zero-effort recipe
   is <= 3.3%. The in-process gate run through the real tools is under way (prelim_v2).
2. *Polishing something that does not change the conclusion?* Partly risky: the remaining T2 menu fingerprint
   (CV AUROC ~0.6) could absorb hours. It cannot by itself pass an episode, so I am stopping on it and documenting it.
3. *Cheapest way to find out whether this works?* The open question is not the gates, which look fine. It is whether
   latents chosen on Wikipedia-style abstracts also fire on the agent's OWN probe texts (different style). The
   cheapest test is the smoke run with a real agent, or a templated-probe reference variant (not done; see weaknesses).
4. *What would I lose by abandoning now?* A working generator, tools, grader and baselines for a lane-(a)/(c) task with
   cheap procedural generation (~7.3k selective latents x 232 concepts). Little reason to abandon.

## 2026-10-01 16:05 UTC — in-process gate run prelim_v2 (T1, T2), through the real Env + caps (local_shim)
- Run dir: runs/featurematch/20261001-153149_prelim_v2 (git sha in config.json). The background job hit the 30-min
  background-shell limit after 116/120 instances. I removed the partial records of the instance in flight from
  episodes.jsonl (only complete 10-solver records kept). The 4 missing T2 instances are re-run in prelim_v2b with T3.
- Reference drop rule (best-of-5 fails -> drop): 3/116 dropped (fm-t1-6f07d71392, fm-t2-5588e6d7cd,
  fm-t2-f463eca9f0). On kept instances (kept_summary.json):
  | tier | reference one-shot | ref best-of-5 | blackbox | nothing | always_claim | prior | prior_or_none | random | name_probe | name_probe_thr | vocab_match |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | T1 (n=59) | 59/59 | 59/59 | 2/59 | 2/59 | 0 | 0 | 0 | 0 | 0 | 2/59 | 2/59 |
  | T2 (n=54) | 51/54 | 54/54 | 2/54 | 2/54 | 0 | 0 | 0 | 0 | 0 | 2/54 | 1/54 |
  Every non-reference pass is an all-null episode (3.4-3.7% of episodes are all-null). Every gate passes.
- Slot level (first reference try): reference planted 1.00/0.98, near-miss 1.00/0.98, off-universe 1.00/1.00, null false
  positives 0/1.1%. Popular-concept prior: planted-slot hit rate 11%/10% (about 2x chance, 0 episodes).
  **vocab_match (decoder logit-lens + name stem match) hits only 0.7%/2.3% of planted slots.** The selected latents are
  context features whose decoder directions do not promote the concept's name tokens (partly by construction: the naive
  name filter removes latents that fire on their own name).
- The black-box control asked the subject model to pick an option. It was never confident (top option probability
  < 0.15), so it behaved like the null action. A smarter black-box policy can do no better than the zero-effort
  recipes: the latent index carries no information without activations.
- Concept coverage (per-entity validation): concepts with >= 1 planted-eligible latent: L6 174, L12 178, L18 190 of 220
  menu-eligible; 200 at some layer. 20 eligible concepts never qualify at any layer (e.g. President, PrimeMinister,
  Mayor, OfficeHolder, Reptile, Mountain). Those are broad politician/geography classes whose latents are shared with
  siblings, so they fail the <= 3-within-0.10 specificity rule. They still appear as distractors.
- Pool composition (180 instances incl. T3): slot kinds T1 154 planted / 57 near-miss / 32 off-universe; T2 150/61/36;
  T3 143/61/40. n_slots 3/4/5 roughly uniform. Mean null fraction per episode 0.394.

## 2026-10-01 16:06 UTC — harness issue: the GPU queue ledger is per-worktree, not machine-wide
- prelim_v2b crashed at model load with CUDA OOM: 18 GB were physically in use by other builders' jobs
  (residualrecall, t2ravel, shifthunt, freqhunt), while my `gpuq status` showed 0 admitted jobs. Cause:
  common/gpuq.py puts its ledger at <repo>/runs/.gpuq, and REPO is the directory of the checkout the module was
  imported from, so every worktree (~/wt/<task>) has its OWN ledger. Not my file to fix (common/); reported in the
  handoff. Workaround here: before launching, wait until nvidia-smi shows >= 9 GB free, then go through gpuq as usual.

## 2026-10-01 16:06 UTC — answer-key hygiene slip, fixed forward
- Commit ac29984 committed runs/.../prelim_v2/episodes.jsonl with per-slot "kinds" (planted / near_miss /
  off_universe) for the 116 prelim instances. That is a partial answer key: it shows which slots are null. Fixed in
  the next commit: run_gates now writes the per-slot detail to episodes_private.jsonl (gitignored), and the committed
  episodes.jsonl holds episode-level outcomes only. History was NOT rewritten (hard rule), so **the prelim pool (seeds
  1000+, 101000+, 201000+) must be treated as gate-only. Generate agent-episode instances from fresh seeds**
  (`generate.py --start-seed <new>`). Answer keys are reproducible from public seeds anyway (D6), but that takes the
  datasets plus GPU time; the committed kinds took none.

## 2026-10-01 16:32 UTC — T3 (tight budget) + remaining T2 gates; final preliminary gate table
- Run dir: runs/featurematch/20261001-160933_prelim_v2b (4 leftover T2 + 60 T3). Its first launch OOMed because
  other builders' jobs held 18 GB (the per-worktree ledger issue above). Relaunched after waiting for free memory.
- T3 = close distractors + forward cap 550 (vs 1200). The reference adapts its probing depth to the budget (reads the
  `budget` tool): 209 / 124 / 110 forward units per slot.
- Drop rule over all 180 generated instances: **5 dropped, 175 kept**: fm-t1-6f07d71392, fm-t2-5588e6d7cd,
  fm-t2-f463eca9f0, fm-t3-62c04099d8, fm-t3-8b6030d1bb. kept_summary.json in both run dirs; `kept` flags are in
  instances_manifest.json.
- **Preliminary gates on kept instances (in-process, real tools and caps, NOT yet through the harness):**
  | tier | n | ref one-shot | ref best-of-5 | blackbox | nothing | always_claim | prior | prior_or_none | random | name_probe | name_probe_thr | vocab_match |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | T1 far, 1200 | 59 | 59 (100%) | 59 | 2 (3.4%) | 2 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
  | T2 close, 1200 | 58 | 55 (94.8%) | 58 | 2 (3.4%) | 2 | 0 | 0 | 0 | 0 | 0 | 2 | 1 |
  | T3 close, 550 | 58 | 45 (77.6%) | 58 | 1 (1.7%) | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
  Gates: reference >= 95% best-of-5 (100% on kept, by the drop rule; one-shot recorded above); blackbox <= 10%;
  every recipe <= 10% in every tier. **PASS.**
- The one-shot reference drops from 100% to 95% to 78% across T1, T2, T3. Two dials (closeness, then budget) each make
  the same probing method less reliable, which is the intended behaviour of a difficulty dial. T3 slot-level reference:
  planted 0.93, near-miss null 0.95, null false-positive rate 3%.

## 2026-10-01 16:42 UTC — style-transfer check (style_check.py): the main validity risk
- Question: latents are selected on DBPedia abstracts and language-ID sentences/reviews. Do they also respond to probe
  text written in a different style, like an agent would write? I hand-wrote 4 probes for each of 12 concepts (6
  languages, 6 topics: everyday sentences / short encyclopedic sentences in my own words). For every planted-eligible
  latent of those concepts (307 latents, all layers), I counted it "recovered" if its own concept has the best AUROC
  among the 12 on my probes and AUROC >= 0.75. Log: runs/featurematch/style_check.log (not committed; small, regenerable).
- Result: **153/307 latents recovered (50%); median per concept 70%.** Spanish 88%, Swahili 93%, chess player 76%,
  volcano 71%, swimmer 70%, German 67%, video game 55%, Japanese 34%, insect 25%, **airport 11%** (airport latents
  presumably key on DBPedia-specific patterns such as IATA/ICAO codes and runway tables).
- Interpretation: with only 4 short probes this is a pessimistic estimate. It still says that about half of the latents
  respond to the dataset's style of text, not just to the topic, and an agent that writes generic probes will see weak
  activation and may answer "nothing found" on a planted slot. That would show up as mid-band difficulty that comes from
  probe design, which is a real interpretability skill (matching the distribution a feature was found on), but it
  could also be an unfair trap. Mitigation done now: agent_prompt.md defines what the concept labels mean ("article
  about a X" = encyclopedia-style article about such an entity; "text written in L" = everyday sentences / short reviews).
  It describes the data, not a method. **Recommended next step (not done, time-box):** add a style-robustness
  filter. Keep a planted slot only if the latent also fires on an independent, differently-styled probe set for its
  concept (templated sentences or a second corpus, e.g. Wikipedia lead sentences rephrased), and measure the smoke-run
  agents' planted-slot "nothing found" rate as a fairness check.

## 2026-10-01 16:42 UTC — status at end of stage 1
- Status: **ready for harness integration**, with one open validity risk (style dependence, above). Not DROP: all
  preliminary gates pass in all three tiers with wide margins, and the reference's one-shot rate falls with each dial.
- Not done (by instruction): PREDICTIONS.md, any LLM agent run, run_agent.py (needs common.sandbox).
- Integrator checklist: see the handoff text (also summarised in SPEC.md); key items: per-worktree gpuq ledger bug,
  bf16 model copy required (RAM cap), agent_prompt.md contains literal JSON braces (do not render it with str.format),
  `task_info` tool supplies the public slots to scripted solvers, solvers' `unwrap()` accepts raw results or the
  {"ok","result"} envelope, prelim pool is gate-only (per-slot kinds were committed once).

## 2026-10-01 16:58 UTC — stage 2 (integrator): wired into the shared harness
- Merged main (harness READY, machine-wide GPU queue D11, review hardening D13) into task/featurematch.
- **What broke and how it was fixed** (all found by running one episode through `prepare`/`./tool`/`finish` and one
  `run-scripted` per solver type before the gates):
  - The reference solver read the `budget` built-in as `{"forward": n}`; the broker returns
    `{"forward": {"used", "cap", "remaining"}}`, so the division would have crashed. `remaining_forward()` now accepts
    every shape.
  - All three scripted solvers submitted `call("submit", answer={...})`. The broker treats the call's args AS the
    submission, so the grader saw `{"answer": ...}` and the format check rejected it. They now send
    `call("submit", answers=[...])`, the same object an agent passes to `./tool submit`.
  - `run-scripted` runs every repeat with the same environment, and the solvers seeded their probe-text sampling from
    `RL_SEED` (default 0), so `--repeats 5` would have run five IDENTICAL attempts and "best-of-5" would have meant
    nothing. Seeds now come from the episode id (RL_SEED still overrides).
  - `run-scripted` passes only `--episode E` to a solver, so `recipe_baseline.py <variant>` could not get its variant.
    It now reads `RL_RECIPE` (inherited by the solver subprocess); the label is set with `--solver-label recipe_<v>`.
  - tools.py fell back to `local_shim` when `common.toolserver` was missing; the fallback is gone, and `local_shim.py`
    and the in-process `run_gates.py` are deleted (preliminary numbers stay in git history at 99d1398).
- **Model loads in a background thread.** `load()` now only reads the permutation seed and starts loading
  gemma-2-2b + SAEs in a thread; model tools wait for it. WHY: `task_info` needs no model, and five of the eight
  recipe variants call nothing else. Loading a 5 GB model for each of their ~360 gate episodes would hold the shared
  GPU for nothing. For an LLM agent the effect is that its first model call takes ~30 s longer (the measured load
  time), which counts toward the 180 s per-call limit with plenty of room. Planted and null instances load the same
  way, so this adds no timing fingerprint.
- **Fresh pool.** The prelim pool's null pattern was committed once (ac29984), so it moved to `instances_prelim/`
  (gitignored; its manifest is now `instances_manifest_prelim.json`). The gate + agent pool is 24 instances per tier
  generated from new seeds (`generate.py --n-per-tier 24 --start-seed 5000 --tiers T1,T2,T3 --clean`: T1 5000-5023,
  T2 105000-105023, T3 205000-205023). Null fraction of slots: T1 0.51, T2 0.41, T3 0.42.
- **Rendered TASK.md checked** (`prepare` also checks it automatically for codename, canary, leak strings and private
  paths): no codename, no layer hint, no method beyond the definition of the answer. One cosmetic fix: the budget list
  now starts on its own line. `task_info` doc corrected: it costs one tool call (not "free").
- **D6 for harness run dirs.** A harness run dir holds `grade.json` with per-slot planted/null flags and the
  reference's `submission.json` (which is right on ~95% of slots, i.e. effectively the answer key). The gate pool is
  also the smoke pool, so `runs/featurematch/.gitignore` now keeps `*_gate_*/episodes/` and `summary.*` local until
  the smoke episodes are done; only `config.json` and the aggregate `<ts>_gates_public.json` (from `gate_report.py`)
  are committed. Delete those ignore lines after the smoke run and commit the full dirs.
- **Harness hazard found (not mine to fix, reported):** the codename "featurematch" is a never-exempt private
  string, and the leak scan removes whitespace before matching. So an agent note such as "this feature matches
  Spanish" in its own scratch file makes the episode INVALID (`private string ... in sandbox file`). Checked with
  `leakscan.scan_text`. For THIS task that phrase is natural. Suggested fix in common/: match codename variants with
  whitespace preserved (or on word boundaries), or exempt codename variants in agent-written files.
- Gates launched through `common.sandbox run-scripted` at 16:51 UTC, prefix `runs/featurematch/20261001-165118_gate_*`:
  reference (one attempt per instance first; best-of-5 retries only on failures), blackbox, and all 8 recipes, on all
  72 instances. The GPU was full (21/21 GB admitted to other builders), so they started queued.
