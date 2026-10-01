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

## 2026-10-01 17:27 UTC — first gate launch lost to the GPU queue; fixed and relaunched
- The first launch (`runs/featurematch/20261001-165118_gate_*`, renamed `..._ABORTED`) waited 30 min for GPU
  admission: the queue admits greedily by memory, and smaller jobs from other builders kept filling every gap before
  a 7 GB request fit. The scripted client's default admission wait is 30 min, so both first solvers exited WITHOUT
  submitting, and the harness graded that as a VALID fail (score 0). Left alone, this would have pushed the
  reference's rate down and the baselines' rates down (making the ≤10% gates look safer than they are).
- Fixes: (1) solvers use `Client(ep, wait_limit_s=RL_WAIT_LIMIT_S)` (default 6 h); (2) the five model-free recipe
  variants read the slot list from the sandbox's TASK.md (exactly what an agent sees) instead of calling `task_info`,
  so they never start a tool server or touch the GPU queue (parser checked against public.json on all 72
  instances: 0 mismatches); (3) `gate_report.py` counts an episode with no submission as an infrastructure failure,
  reports the count, and leaves it out of the rates (scripted solvers always submit unless they crash).
- Relaunched as `runs/featurematch/<ts>_gate_{reference,recipes,blackbox}` (ts = 20261001-172659
  ).
- Seen while relaunching: the orchestrator ran two API probes on this pool (`20261001-171048_apiprobe_sonnet`,
  Sonnet, after PREDICTIONS.md was committed; an earlier Opus attempt `..._165348_apiprobe_orch` was aborted at turn 3).
  T2 fm-t2-08a4773087: score 0.8, one PLANTED slot answered "nothing found", episode INVALID (transcript audit).
  T3 fm-t3-025fcaa85e: score 0.5, two wrong claims on planted slots, used only 103 of 550 forward units. n=2: not a
  measurement. Both instances are excluded from the smoke plan because an agent has already seen them.

## 2026-10-01 18:12 UTC — GPU-queue starvation slows the harness gates
- Each harness episode starts its own tool server, and each server asks the shared queue (common/gpuq.py) for 7 GB.
  Admission is greedy polling with no ordering for light jobs. With ~6 other tasks running their own gates, a
  free gap of 7 GB almost never stays open: 2-4 GB servers from other tasks are admitted within seconds while a
  7 GB request waits (latentdiff's 6.5 GB requests starved the same way). Measured: the first GPU reference
  episode of the relaunch took 2256 s end to end, almost all of it admission wait. At that rate the ~300 GPU gate
  episodes would take ~45 h. Reported to the orchestrator (fairness fix = aging/FIFO for light waiters on main).
- What is NOT affected: the five model-free recipe variants (360 episodes) read TASK.md and never queue; they are done.
- Considered and rejected: running a batch of episodes behind one queue admission with a private broker. It would
  sidestep the shared queue's per-server accounting, which is the rule that keeps 6 tasks on one L4 safe, so it is
  not done.
- Plan: GPU gates stay on the normal path and are ordered so partial results stay balanced across tiers: the
  reference on 20 instances per tier interleaved T1/T2/T3 (then the remaining 12), and in a second stream blackbox
  then name_probe_thr, name_probe and vocab_match on the same 60. Two crashed or killed episodes per stream are in the
  run dirs without submissions; gate_report.py counts them as infrastructure failures, not fails.
- GPU_GB stays 7: weights 5.23 GB + SAEs 0.68 GB + activations put the peak around 6.3 GB, and the server's allocator
  cap is set from GPU_GB, so lowering it to win admission would risk OOM inside agent tool calls.

## 2026-10-01 19:11 UTC — stage-2 gate results: ALL PASS (harness + in-process complement)
- Pool: 72 fresh instances (24 per tier). Reference best-of-5 passed all 72, so nothing was dropped (kept = 72).
- **Harness** (`common.sandbox run-scripted`, run dirs `runs/featurematch/20261001-172659_gate_*`): the five model-free
  recipes on all 72 instances each (360 episodes, all VALID): nothing 1/0/1 of 24 (T1/T2/T3), always_claim, prior,
  prior_or_none and random 0/24 everywhere. GPU episodes completed so far: reference 3/3 pass; black box 1 (fail).
  Infrastructure failures excluded (killed while waiting for the GPU): reference 1, black box 2, name_probe 1.
- **In-process complement** (`inproc_gates.py` as ONE normally queued 7 GB job: admitted after ~11 min, ran 13 min;
  run dir `runs/featurematch/20261001-184458_inproc`): every solver on all 72.
  | tier | ref one-shot | ref best-of-5 | blackbox | nothing | always_claim | prior | prior_or_none | random | name_probe | name_probe_thr | vocab_match |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | T1 | 24/24 | 24/24 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
  | T2 | 23/24 | 24/24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  | T3 | 21/24 | 24/24 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
  Overall: reference one-shot 68/72 = 94.4% (Wilson 0.87-0.98), best-of-5 72/72 (0.95-1.0); black box 2/72 = 2.8%
  (0.008-0.096); worst recipes (nothing, vocab_match) 2/72 = 2.8% (0.008-0.096), at most 1/24 = 4.2% in any tier.
  Gates: reference >= 95% best-of-5 PASS; blackbox <= 10% PASS; every recipe <= 10% PASS, in every tier.
- **Cross-check of the two paths:** on the same (solver, instance) the per-slot outcomes agree exactly for every
  deterministic solver: 72/72 for each of nothing, always_claim, prior and prior_or_none, 3/3 reference, 1/1 black
  box. random differs by design (harness seeds come from the episode id). So the in-process numbers are what the
  harness would have produced, minus the queue wait.
- Slot level (in-process): reference null-slot false claims 0 in every tier; planted-slot accuracy 1.00 / 0.98 / 0.93.
  The reference one-shot falls 100% -> 96% -> 88% across T1 -> T2 -> T3, as in stage 1 (100 / 95 / 78). Name-probe
  planted accuracy 0.051 (chance 0.05; the generator's filter works), vocab_match 0.013, popular-concept prior 0.13.
  Every non-reference pass is an all-null episode (2 of 72 pool episodes are all-null) answered "nothing found" everywhere.
- **Smoke plan** written by `gate_report.py --smoke`: 3 per tier, reference passed first try, excluding the 4 instances
  the orchestrator's LLM probes already used (fm-t2-08a4773087, fm-t3-025fcaa85e, fm-t2-0b1b5c8b61, fm-t2-18d866dedf).
- Rendered TASK.md for all 72 instances passes the harness's own check_task_md (codename, canary, leak strings,
  private paths): 0 problems. The prompt names no method, layer hint, dataset name or grading term.

## 2026-10-01 19:12 UTC — fingerprint re-check on the stage-2 code
- `fingerprint_check.py --n 400` (in-memory instances, seeds 500000+; gradient-boosted classifier on public menu
  structure, 5-fold CV AUROC for planted vs null slot): T1 0.535 +- 0.022, T2 0.597 +- 0.025, T3 0.596 +- 0.025.
  Same as stage 1. WHY it is acceptable for now: the signal can only nudge an agent's belief that a slot is null; it
  never says which option is right, and claiming the wrong option on a planted slot fails just like a false claim.
  No recipe gets above 4.2% in any tier. Still a known weakness (near-miss null menus contain fewer languages);
  a fix would draw the near-miss menu's anchor-family mix to match the planted menus' mix.
- Tool outputs and timing do not depend on planted vs null at all: the tools only use the instance's permutation seed,
  never the answer key, and model load is identical for every instance.

## 2026-10-01 19:12 UTC — end of stage 2 (integration): status ready_for_smoke
- Harness reference stream stopped at 3 graded episodes (3/3 pass: 2 T1, 1 T2) so it stops competing in the queue with
  the orchestrator's own probes on this task. Final harness infrastructure exclusions (unsubmitted, killed while waiting
  for the GPU): reference 2, black box 2, name_probe 1. No harness episode was INVALID.
  `runs/featurematch/20261001-172659_gates_public.json` refreshed; smoke plan unchanged.
- Rabbit-hole check (plan questions): (1) still on the path to a validated task? Yes; every gate passes with margin,
  and both dials move the reference's one-shot rate. (2) Polishing something that doesn't change the conclusion?
  Further harness GPU episodes would; stopped. (3) Cheapest next test: the smoke run (3 per tier), watching
  the planted-slot "nothing found" rate (the style-dependence risk) and the near-miss null false-claim rate.
  (4) Lost if abandoned now: a working, gated SAE-latent-explanation environment; little extra work is sunk.
- **Before the smoke run (blocking):** fix the harness codename check (an agent note "feature matches" invalidates
  the episode), and expect long GPU-queue waits for 7 GB servers unless the queue gets aging/FIFO for light waiters.
  After the smoke run: remove the D6 ignore lines in runs/featurematch/.gitignore and commit the full gate run dirs.

## 2026-10-01 23:14 UTC — fix stage after the independent audit (AUDIT.md): generator v2, product reward
The auditor found no leak and no grader bug, but 4 MAJOR issues. What I changed, and WHY, in plain terms:
- **MAJOR 1, reward rewards doing nothing.** The continuous `score` was the mean slot correctness. About 40% of slots
  are null, so answering "nothing found" everywhere earned 0.44 while the reference earns ~0.98: an RL policy trained on
  `score` would happily settle there (the auditor-RL failure mode CONTEXT.md warns about, in reverse). Fix in grader.py:
  `score = planted_acc x null_acc` (empty factor = 1). Both halves of the job must be done to earn anything. The old
  mean survives as `details.slot_accuracy` (diagnostic only). SPEC now says the RL reward is `pass` or `score`.
  Measured (harness, 180 episodes each, run dir 20261001-222256_gate_recipes): "nothing" mean score 0.061 (v1 0.436),
  which is exactly its all-null pass rate; always_claim 0.008; prior 0.009; random 0.036.
- **MAJOR 2, cheap latent statistics separated null from planted slots (AUROC 0.70-0.80).** Cause: 40% of nulls were
  "off-universe" latents (live but not selective for any concept) whose firing on generic text looks different from
  concept-selective latents. Fix: drop off-universe nulls. Every slot now draws its latent from ONE pool by the same
  steps (layer -> anchor concept c* uniformly -> one of c*'s selective latents); a planted slot puts c* on the menu, a
  null slot does not. Both kinds remove the SAME confusable set (AUROC_A >= 0.65 or AUROC_B >= 0.72) from the
  distractors, and there is no rejection step that depends on the kind. So the latent of a null slot is statistically
  identical to the latent of a planted slot; only testing the options can tell them apart. Side effect: planted
  answers now win by >= 0.25 AUROC (v1: >= 0.10), so the ground truth is cleaner.
  Considered instead: keep off-universe nulls but stratify them to match planted firing statistics. Rejected: matching
  rate/mean/quantiles still leaves any other statistic free, so it is an arms race; one shared pool ends it.
- **MAJOR 3, the name filter left a "second place" artifact.** v1 rejected a planted MENU when the answer's name was
  the top name-probe activation; only rank 1 was removed (the answer's name was exactly rank 2 in 36-44% of cases) and
  only planted slots were filtered. Fix: filter LATENTS, for every slot: a latent that fires on any of the 3 surface
  forms of its own concept's name never enters the pool. On every slot the answer's name is silent, like most
  distractor names. Cost: the pool shrinks from 2210/2631/2508 to 1413/1855/1749 latents (layers 6/12/18), and the
  answerable concepts from 174/178/190 to 108/118/149. Risk this raises: the kept latents respond to context, not to
  the bare name, which may add to the style-dependence risk (NOTES 16:42); the smoke run should watch the planted-slot
  "nothing found" rate.
- **Menu universe per layer (new).** Distractors are drawn only from concepts that own a pooled latent at that layer,
  so no label can be ruled out because "it is never an answer at layer 6" (that would have grown with the stricter
  filter).
- **MINOR 7.** grader.py now parses choices exactly like the format check (bool and float choices or slot ids
  rejected, no `{"answer": ...}` unwrapping). audit/grader_edge_cases.py re-run with the new expectations: 29/29 as
  expected (run dir 20261001-222216_fix_grader).
- **MINOR 8.** runs/featurematch/.gitignore now holds back the orchestrator's probe episode dirs. gate_report.py draws
  the smoke set at random from ALL kept instances (not only the reference's first-try passes) and reports mean score.
- **MAJOR 4 (self_probe).** Added self_probe, template_probe, density_null and name_rank2 to recipe_baseline.py as
  standing recipe gates. Measurement below.
- **Fresh pool (generator v2).** The old pool moved to instances_v1/ (gitignored; manifest instances_manifest_v1.json).
  New: `generate.py --n-per-tier 60 --start-seed 7000 --tiers T1,T2,T3 --clean` (T1 7000-7059, T2 107000-107059,
  T3 207000-207059), 60 per tier so the T1 recipe measurement has n = 60 as the auditor asked. Null fraction of slots
  0.38 / 0.41 / 0.37; all-null episodes 4 / 4 / 3 (expected ~3.3% per tier; this sample is a bit high, and "nothing
  found" everywhere passes exactly these). prior.json regenerated (seeds 900000+).
- Offline checks on v2 (CPU, precomputed tables; run dir 20261001-222059_fix_offline):
  menu-structure fingerprint CV AUROC 0.49 / 0.54 / 0.55 (v1 0.54 / 0.60 / 0.60). Auditor's A1 learned menu prior,
  held-out null AUROC 0.50 / 0.56 / 0.59; A2 latent statistics + names + menu 0.51 / 0.58 / 0.59 (v1 0.70 / 0.80 /
  0.77), i.e. at or under the auditor's 0.6 acceptance line in every tier; A2 planted-slot accuracy 0.00 / 0.07 /
  0.04, held-out pass 3.3 / 3.5 / 3.5% (all from all-null episodes or near it). Name probes: "any option name fires"
  0.14 on null vs 0.15 on planted slots; the answer's own name never fires.
  Offline reference simulation (simulate.py): best-of-5 1.00 / 0.97 / 0.98.
- Asked to stop my own GPU jobs to reorder them (self_probe takes ~70 s per episode; the harness cross-check stream
  was competing with my in-process job for the queue): the permission classifier refused ("interfere with
  workloads"). I did not pursue it; both jobs run to completion as launched, which only costs wall-clock time.
- PREDICTIONS.md is left untouched: LLM agents (orchestrator API probes) have already run on this task, so it may
  not be replaced. Its numbers were made for the v1 design.
