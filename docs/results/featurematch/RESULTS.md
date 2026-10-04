# FeatureMatch diagnosis: results

2026-10-02, branch `diag/featurematch`. The study **stopped at step 3**: the pre-registered stop rule fired, so no
agent arm ran. The one-page summary is `VERDICT.md`.

**How to read this file.**
- Every number comes from a committed file, which is named next to it.
- Paths are relative to `tasks/featurematch/diagnosis/` unless they start with `docs/` or `tasks/`. `docs/LOG.md`
  means the copy on the main branch.
- All intervals are 95%. *Wilson* intervals treat slots as independent. *Clustered* intervals resample whole
  instances (10,000 resamples, `random.Random(20261002)`), so they allow for the slots of one episode being related.
- `STEP3.md` is the report that brought together the step-2 checks and step 3. Where it and a raw result file
  disagree, this file uses the raw file and says so.

## Words used

**The model and its latents.**
- **SAE and SAE latent**: a sparse autoencoder (SAE) is a second network trained to rewrite a model's internal
  activations at one layer as a sum of a few directions out of many. Here the model is gemma-2-2b and the SAEs are
  Gemma Scope SAEs at layers 6, 12 and 18, with 16,384 directions ("latents") each (`tasks/featurematch/SPEC.md`).
  Each latent is usually 0 and becomes positive on some kinds of text. Interpretability work tries to say what each
  latent responds to, because that turns an opaque activation vector into named parts that can be checked. A text's
  activation is the latent's maximum over the text's tokens.
- **Generic fire rate** (density): the share of all split-A texts, over all 232 concepts, on which a latent fires
  (`PREREG.md` section 0).

**The task.**
- **Slot**: one question, "which of these 20 concepts (the **menu**) does latent N respond to?". An **instance** is
  one generated task (its slots and menus). An **episode** is one run of a solver or agent on one instance. An
  episode has 3-5 slots and **passes** only if every slot is right.
- **Planted / null slot**: on a planted slot the true concept (the **anchor**, c\*) is on the menu. On a null slot
  it is not, and the right answer is "nothing found".
- **Planted accuracy**: the share of planted slots answered with the right option. The gate and the stop rule use
  this number. **Planted NF** is the share of planted slots answered "nothing found".
- **Tiers**: T1 = "far" menus (distractors unlike c\*), T2 = "close" menus, both with 1200 forward units; T3 = close
  menus with 550. A **forward unit** is one text run through the model; the tier's forward cap is the budget.
- **Menu universe**: the concepts a layer's menus can draw on (the "answerable" concepts).

**How the key is defined and checked.**
- **AUROC** of a concept: the chance that a random text of that concept makes the latent fire more than a random
  text of another concept. 0.5 is chance and 1.0 is perfect.
- **Splits A / B / C**: three disjoint sets of encyclopedia-style dataset texts per concept. A defines the answer
  key, B helps choose latents, and C is held out.
- **margin_C**: AUROC_C(c\*) minus the best AUROC_C among all other concepts (`style_filter.py`).
- **Banks F and R** ("styled texts"): two independent sets of LLM-written texts in 10 everyday styles (text message,
  trivia question, interview, press release, and so on). Each covers 232 concepts × 10 styles × 2 texts, and the two
  banks were written by different writer agents. Bank F was used to build the filter (step 2). Bank R was used only
  by the style-robust recipe (step 3). F1 / F2 are the first / second text per style of bank F.
- **k_ms / k_ho**: the multi-style key and the held-out check of Amendment 1 (section 2.3).
- **Topic / language family**: an anchor is either a topic ("article about an airport") or a language ("text
  written in Thai").

**Baselines and checks.**
- **Reference solver**: a script that probes each option with held-out split-C dataset texts, which an agent never
  sees. It claims the best option if that option's AUROC against background texts is >= 0.78. **One-shot** is one
  run (seed 7); **best-of-5** re-runs with up to 5 seeds until one passes (`step3/BASELINES.md`). It is the ceiling
  baseline.
- **Black-box control**: a baseline that cannot see activations. It asks the model which option the latent encodes
  and claims only if that option's next-token probability is >= 0.15 (`tasks/featurematch/blackbox_control.py`). It
  is the floor baseline.
- **SR, the style-robust recipe** (step 3): a script that runs 6 bank-R texts per option and claims the option with
  the highest mean activation (**SR-max**), or claims it only if that option's AUROC against the other options' texts
  is >= 0.78 and otherwise says "nothing found" (**SR-thr**). Section 5.1 has the details.
- **Gate**: every non-reference recipe except SR must have planted accuracy <= 0.15, and the reference >= 0.95.
- **P6, the menu fingerprint**: a classifier that sees only menu statistics tries to tell planted from null slots;
  P6 required its cross-validated AUROC to be <= 0.60 (0.5 = it learns nothing). Section 6 explains why it matters.
- **Harness**: the episode runner used for agents. Its **broker** relays and charges every tool call, and its **leak
  scan** checks transcripts for answer-key leaks.

---

## 1. What was asked and why

**Dan's brief** (`PLAN.md`, 2026-10-02 ~01:10 UTC): find out whether FeatureMatch is hard for the **right reasons**
(interpretability skill) or for **wrong reasons** (probe-writing style, unclear tools, early stopping, bugs).

**Why it matters for an RL environment.** RL trains whatever earns the reward. If the reward can be earned by
writing text in the right style, or lost by quitting early, a policy trained on FeatureMatch learns those habits and
not how to interpret latents.

**What was known before the study.** All agent data came from the retired v1 pool (`PLAN.md`;
`docs/forensics/FEATUREMATCH_PROBE_FORENSICS.md`):
- Two models were probed on 4 valid episodes:
  - Sonnet named the right concept on 5 of 8 planted slots.
  - gpt-6-luna got 0 of 3 planted slots right.
- Every agent submitted early, using 8-62% of its forward units (ep2cd 62%, epcd 19%, ep0f 12%, epea 8%). In all 7
  misses at least 37% of the budget was unused (forensics, sections 1 and 3).
- No agent called `top_latents` or `generate`.
- Only about 50% of eligible latents recovered their own concept on hand-written probes.
- Two "MMA event" latents fired on 54/54 dataset texts with that phrase but read 0 on agent-written sentences.
- On v1, fixed probe-writing recipes got 0.11-0.14 planted accuracy and the reference got 0.96.

Scripted baselines had also been measured on the v2 pool, from recorded runs (`PREREG.md` section 3): reference
planted accuracy 0.97 and pass 178/202 = 0.88, self_probe 0.16 (planted NF 0.78), black-box 0.00. No agent had run
on v2.

So style dependence was the main suspected wrong reason.

**The plan** (`PLAN.md`):
- Step 0: pre-register.
- Step 1: audit the tools and the task text for clarity.
- Step 2: filter latents for style robustness and build a filtered pool (v2f).
- Step 3: run baselines on v2f, including a new scripted **style-robust recipe**.
- Step 4: run agent arms with fresh contained subagents.
- Step 5: classify every failure.

Step 3 had a stop rule: if the style-robust recipe passes most planted slots, FeatureMatch does not require
interpretability, so stop and report.

**Step 1b** (Dan, binding):
- The style filter is the **main validity fix**. A latent's concept must be the class it separates best *across
  styles*, not only on encyclopedia text.
- `top_latents` and `generate` must be documented in the task text.
- A disclosure arm is added.
- RESULTS.md must report how much of the pass rate the style-robust recipe explains. If it explains most of it, the
  report must say plainly that FeatureMatch is mostly measuring style.

## 2. What was pre-registered

### 2.1 The rules

**Stop rule** (`PREREG.md` section 4, checked first). The rule **fires** if the style-robust recipe gets planted
accuracy > 0.50, as SR-max **or** SR-thr, on the filtered pool with all tiers pooled and n >= 100 planted slots.
Then "FeatureMatch does not require interpretability", step 4 does not run, and the verdict label is **Not
interpretability**. If the point estimate is above 0.50 but its Wilson CI includes 0.50, the rule still fires and
is reported as "borderline".

**Gate** (`PREREG.md` step 3). Every non-reference recipe except SR must have planted accuracy <= 0.15, and the
reference must reach >= 0.95.

**Decision rule** (`PREREG.md` section 4, used only if the stop rule did not fire). FeatureMatch is "hard for the
right reasons" if, on arm A:
1. the pass rate is 10-70%;
2. >= 60% of failures are right-reason;
3. the agent's planted "nothing found" rate is closer to the reference's than to the recipes';
4. the pass rate drops as sibling similarity rises.

The verdict labels for every combination are listed in section 4 of `PREREG.md`.

**Failure classification** (`PREREG.md` section 1). It is a scripted, fixed-priority rule that labels each failed
slot as one of:
- wrong-reason: W1 silent because of style, W2 early stop, W3 tool/format error, W4 harness/key bug;
- right-reason: R1 sibling confusion, R2 word-vs-concept confusion, R3 null over-claim, R4 wrong hypothesis after
  adequate probing, R4s search failure;
- manual review: U0 and U1.

Wrong-reason rules take priority, which makes the study conservative against concluding "right reasons". This
classifier was never written or used, because no agent ran.

### 2.2 Timeline: what was fixed when, and what had been seen

Times are git commit times (UTC). This log was checked to be append-only: each later commit removes 0 lines of
`PREREG.md` (`git diff --numstat` of 0148033c..a4b8aecd: 160 added, 0 removed; a4b8aecd..72c1c1b3: 43 / 0;
72c1c1b3..2f663caf: 30 / 0; 2f663caf..22111857: 9 / 0).

| Commit, time | What it fixed | What the orchestrator had seen at that point |
|---|---|---|
| `0148033c`, 01:23 | `PREREG.md` sections 0-5: classification rules, experiments, predictions P1-P30, decision and stop rules | The v1 agent forensics, and v2 scripted baselines recomputed from recorded runs (`PREREG.md` section 3: reference 0.97 / pass 0.88, self_probe 0.16 / planted NF 0.78, black-box 0.00). No style bank, no filtered pool, no agent on v2. |
| `793458e6`, 01:25 | Step 1 clarity audit (after PREREG) | the same |
| `a4b8aecd`, 02:24 | **Amendment 1**: multi-style key and keep rule (A1.1); arm G, disclosure (A1.2); tool docs for arm A (A1.3); tool-use metrics (A1.4); the required section "How much of the pass rate does style explain?" with a binding sentence (A1.5); predictions P31-P46 | Step 1 done. Amendment 1's own state line says banks F and R existed as text only, with no activation computed on any bank text, no filtered pool and no agent on v2. The banks were committed at 02:42 (`3da15f6a`). The activation code was committed at 02:58 (`827191af`; the activation manifest `style_filter_cache_manifest.json` names that commit). |
| `72c1c1b3`, 03:53 | **Amendment 2**: which 6 bank-R texts SR uses (A2.1); a "text-selection-sensitive" label (A2.2); T3 budget handling (A2.3); a note on SR-thr's AUROC (A2.4); a clustered CI next to Wilson (A2.5) | **Had seen two non-pre-registered results**, both disclosed in A2. (a) The first step-2 code, which did not follow A1.1 (it kept 25% of latents under a pooled rule; under the F2-only part, 8.65% passed; `STYLE_FILTER.md`, superseded section). (b) An SR dry run on the **unfiltered** v2 pool: SR-max 0.520 [0.473, 0.567], SR-thr 0.258, and across 42 text selections SR-max 0.473-0.582 (`sr_recipe_out_v2_unfiltered_DRYRUN/summary.json`). No filtered-pool result. |
| `3ea47d7f`, 04:02 | Step 2 implemented exactly as A1.1 | |
| `2f663caf`, 04:04 | **Amendment 3**: run step 3 despite the fingerprint failure (P6), since P6 cannot help a scripted recipe; no agent arm until a pre-registered menu fix brings P6 to <= 0.60 (A3.1); report every metric by family (A3.2); independently re-implement the step-2 keep rule and audit bank F before step-3 numbers count (A3.3) | Only the A1.1 step-2 report: 354/5017 latents kept, pool regenerated, P6 0.616. No step-3 result. The step-3 runs started at 04:10 (`step3/runs/20261002-041021_inproc`; `step3_skeptic/skeptic_summary.json`). |
| `22111857`, 06:07 | Outcome pointer appended to `PREREG.md` | All results |

**Timestamp errors in the headers** (a documentation problem only; the order of events holds):
- The headers of Amendments 2 and 3 say ~04:20 and ~04:45 UTC. Their commits are 03:53 and 04:04, and both still
  precede the step-3 runs (04:10) (`step3_skeptic/skeptic_summary.json`, discrepancies).
- `PLAN.md` and Amendment 1 date Dan's step 1b at ~01:35. The step-1b text was committed to `PLAN.md` at 01:20
  (`e52d8def`), about 2 minutes before `PREREG.md` sections 0-5. Sections 0-5 do not include step 1b, and Amendment 1
  was written to add it.

### 2.3 What each amendment changed, in plain words

**A1.1 (the multi-style key).** The old key was c\* = the concept with the best AUROC on split A, which is
encyclopedia text only. The new rule keeps a pooled latent only if all three of these hold:
1. *Multi-style key*: c\* is still the best concept when split A and bank half F1 get equal weight.
2. *Held-out check*: c\* is still the best concept on split C plus bank half F2. Neither text set was used for the
   multi-style key.
3. *Style robustness*: AUROC_F2(c\*) >= 0.85 **and** AUROC_C(c\*) >= 0.85.

F1 holds the first text per style and F2 the second. The rule acts on latents, so planted and null slots are
treated alike. The kept latents are then given to the unchanged v2 generator, which builds a new pool of 180
instances.

**A1.5 (the binding sentence).** If the recipe's share of the agent's success (Q_SR) or the oracle-examples share
of the agent's gap (Q_D) is > 0.50, the report must say **"FeatureMatch is mostly measuring style, not
interpretability"**. In the stop-rule case, where no agent arm exists, the test is pa(SR)/pa(ref) > 0.50: SR's
planted accuracy divided by the reference's.

**A2.** It was written after the unfiltered dry run had been seen. It did five things, none of which changes the
rule's 0.50 bar:
- A2.1 adopted, as the primary selection, the rule for which 6 of each concept's 20 bank-R texts SR uses (six styles
  drawn by seed, first text per style). That rule had been fixed in code before SR produced any number on any pool
  (`SR_RECIPE.md`, "Which 6 texts").
- A2.2 added a spread over 42 selections (21 style seeds × first or second text). A2.2 calls them "alternative", but
  the computed set includes the primary (`step3/sr_recipe_out/summary.json` `text_selection_sensitivity`: "primary
  included"; `SR_RECIPE.md`).
- A2.3 made the uncapped SR primary, although it goes over the T3 cap of 550 forward units in 22 of 60 T3 episodes,
  and added a capped variant (section 5.3).
- A2.4 noted that SR-thr's AUROC uses different negatives from the reference's.
- A2.5 added a clustered CI next to the Wilson CI.

**A3.** It was written after step 2 had failed P6. It allowed step 3 to run on that pool, blocked agents until P6
is fixed, required by-family reporting, and required independent verification of step 2.

---

## 3. Step 1: tool and task clarity

Source: `clarity.md` (CPU only). The audit read the task text, the tool code and all 8 recorded probe
transcripts.

**It found 17 ambiguities (A1-A17). Two of them cost agents real budget or answers:**
- **A1, undocumented response keys.** One Sonnet episode guessed wrong key names, read every probe as 0, and spent
  **116 forward units**, about 10% of its budget, before finding the right keys.
- **A3, JSON quoting.** Arguments are passed as one shell string, so an apostrophe breaks the command. There were
  **6 failed commands in 3 episodes**. The agent then **rewrote its probe texts without the apostrophe** ("The men's
  100 metre freestyle" became "The 100 metre freestyle"), so the quoting problem changed what was measured.

**Medium-severity ambiguities.**
- That a response over 30 KB comes back in a different shape, with no `result` key (A2).
- How the budget is counted (A4).
- That a retried call is charged twice (A5).
- That `top_latents` uses the same latent numbering as the slots (A6).
- That "encodes" means "separates against all other concepts, using each text's max" (A9).
- What a null latent is (A10).
- That the options are disjoint classes (A11).
- How an episode is scored, so that both kinds of error fail it (A12, rated low-medium).

**What the transcripts cannot show.** They contain no reasoning text, so the audit **cannot** say whether any agent
considered and rejected `top_latents` or `generate`. No agent called them.

**What was done.** A revised template (`tasks/featurematch/agent_prompt_revised.md`) has the same tools and budget.
It adds:
- a full response schema;
- the exact budget accounting;
- three safe ways to pass JSON;
- what each tool shows;
- four definitions the old text left implicit.

It names no method and gives no answer hint, and it passed the harness checks. At step 1 the rendered task text grew
from about 1,340 to 2,518 words (`clarity.md`; `PREREG.md` A1.6). After A1.3 added per-tool documentation, the arm-A
template renders at 2,815 words and the disclosure template at 2,839 (`PREREG.md` A1.6). That length gap would have
been a confound for arm A vs arm B.

**Why this did not decide anything.** The revised template was never used, because no agent ran. The step-3 result
involves no agent at all, so clarity problems cannot explain it.

---

## 4. Step 2: the style filter under A1.1

### 4.1 Why a style filter

The v2 generator picked each latent's concept from encyclopedia-style text only. A latent can separate "article
about an airport" perfectly on DBpedia abstracts and still be silent on a chat message about an airport. In v1, the
agents' probes were mostly not encyclopedia-style, and such latents often read near 0 on them (the two MMA latents
read 0 on the agents' sentences; `PLAN.md`, forensics section 3). A failure caused that way says nothing about
interpretability skill. The filter keeps only latents whose key survives on styled text.

### 4.2 What survived

Sources: `verify_reimpl/result.json`, `style_filter_out/summary.json`, `STYLE_FILTER.md`.

| Group | Kept | Share [Wilson] |
|---|---|---|
| All pooled latents | 354 / 5017 | **0.071** [0.064, 0.078] |
| Language latents | 163 / 276 | 0.591 [0.532, 0.647] |
| Topic latents | 191 / 4741 | **0.040** [0.035, 0.046] |
| Layer 6 | 102 / 1413 | 0.072 [0.060, 0.087] |
| Layer 12 | 78 / 1855 | 0.042 [0.034, 0.052] |
| Layer 18 | 174 / 1749 | 0.099 [0.086, 0.114] |

- **The styled-text threshold drives almost every drop.** 0.082 of latents pass the F2 part of the robustness test,
  against 0.954 for the split-C part.
- Only 54 latents (1.1%) were dropped *only* because a key disagreed while the robustness test passed.
- The multi-style key equals c\* for 0.764 of latents, and the held-out check for 0.719. When they differ on a
  planted v2 slot, the other concept is always one the generator had already kept **off** the menu. So the menu
  answer was still right on every planted v2 slot (`STYLE_FILTER.md`).
- **Effect on the current v2 pool.** 583 / 703 slots (0.829 [0.800, 0.855]) have a latent that fails the rule.
  That is 0.831 of planted slots and 0.827 of null slots (`verify_reimpl/result.json`). The pool v2 agents would
  have seen was mostly made of style-fragile latents.

**By the three pre-registered measures, the filter does not select easier latents.** None of the 3 tests is met
(`style_filter_out/summary.json`). By construction, though, it does keep latents that are easy to find with varied
text, which is exactly what SR exploits (section 9).

| Test | Value | Met? |
|---|---|---|
| Median margin_C, kept minus dropped, > 0.05 | −0.040 | no |
| Median generic fire rate ratio > 1.25 | 0.70 | no |
| Reference planted accuracy, kept minus dropped, > 0.05 | +0.017 (0.986, n = 73, vs 0.969, n = 358) | no |

Survivors are sparser and have *smaller* margins than the dropped latents. What sets them apart is exactly what
the filter tests: the median survivor fires on 0.95 of its concept's bank-F texts, against 0.15 for the median
dropped latent (means 0.906 vs 0.230; `style_filter_out/summary.json` `pool_kept_vs_dropped`, `fire_F_anchor`).

### 4.3 The regenerated pool (v2f)

Source: `style_filter_out/v2f_summary.json`.
- 180 instances, 60 per tier, with 725 slots: 441 planted and 284 null (null fraction 0.392).
- 7 episodes are all-null.
- 0 planted slots where the multi-style key is not the answer.

**The pool is narrow.**
- It uses 244 distinct latents and **74 anchor concepts**; v2 used 159 (`STYLE_FILTER.md`).
- The menu universe per layer has 34-63 concepts; v2 had 108-149.
- **Language anchors are 27.3% of slots** (198/725), against 9.2% in v2.

### 4.4 Verification of step 2 (A3.3)

**Independent re-implementation: passed** (`verify_reimpl/result.json`).
- It was written from A1.1 alone. It reproduces `key_check.jsonl` with **0 / 5017 mismatches** in c\*, the two
  keys, kept, and the reasons.
- Rounding does not matter. A float16 or a float64 split-A table gives the same kept set.
- No latent sits within 1e-4 of the 0.85 threshold.

**Bank-F texts are on-concept: passed** (`verify_banks/results.json`, `verify_banks/PROTOCOL.md`).
- The worry: maybe only 4% of topic latents survive because the bank texts are off-topic.
- The audit rated 800 texts for 40 topic concepts, blind to whether each concept's latents were kept. The bar was
  committed before rating.
- Result: **800/800** sampled texts are clearly or loosely on-concept, and 0 are off-concept. 0.884 are strictly
  "clear". One LLM rated them. Bank R was not audited.
- No detectable difference between the kept and dropped groups in the "clear" rate: −0.038 [−0.113, 0.030].
- The dropped low-fire latents fire on only 0.103 [0.084, 0.138] of texts that are clearly on-concept and name an
  instance (n = 7,122 latent-text pairs). The same latents fire on a median 0.925 of their concept's dataset texts,
  and on 0.402 of the bank's one encyclopedia-like style.
- So the texts are on-concept. The dropped latents fire far more on the one style that imitates an encyclopedia than
  on the other on-concept texts, which points at the encyclopedia template rather than the subject.

**What the dropped topic latents respond to** (`verify_mechanism/summary.json`, `verify_mechanism/q1_categories.json`).
One LLM reader, not blinded to which group a latent was in, categorised the peak tokens of 30 dropped and 15 kept
topic latents.

| | Format or register cue | The topic itself |
|---|---|---|
| Dropped (n = 30) | **18/30** | 5/30 |
| Kept (n = 15) | 0/15 | **13/15** |
| Fisher test | p = 6e-5 | p = 9e-6 |

Examples from `q1_categories.json`:
- An "airport" latent peaks on the code block "(IATA: HIO, ICAO: ..." and fires on 0 of its 20 bank-F texts.
- A "tennis player" latent peaks on "player" in "(born ... 1978) is an American former tennis player". It reads 0 on
  "Tennis player Elena Petrova ...".
- An "NCAA team season" latent fires on "represented the University of X during the 19xx season".

**How much weight this table can bear.** Less than the p-values suggest, for three reasons:
- *The categories partly use the sorting criterion.* The reader saw each latent's peaks on bank-F2 texts, and the
  rule for "topic" requires that the latent "also fires on on-topic texts in other styles" (`q1_categories.json`
  `_rules`). F2 is the text set that sorted latents into kept and dropped, so part of the 0/15 vs 18/30 split
  restates the sorting.
- *The firing rates below are not independent evidence.* "Dropped" was defined by a low AUROC on bank-F2 text, so a
  gap in styled-text firing is expected by construction:

  | | Fire rate on c\*'s dataset texts | Fire rate on c\*'s styled texts | AUROC_F2(c\*) |
  |---|---|---|---|
  | Dropped | 0.945 | 0.253 | 0.582 |
  | Kept | 0.960 | 0.817 | 0.913 |

- *A scripted check does not reproduce the split.* A scripted peak-token check labels 8/15 kept latents as "format",
  against 9/30 dropped, and agrees with the reader on only about 14/45 latents (`summary.json` `kept.scripted_type`,
  `dropped.scripted_type`, `scripted_vs_manual`).

What independent support remains: dropped low-fire latents fire on 0.402 of the bank's encyclopedia-style texts
against 0.103 of clearly on-concept texts that name an instance (`verify_banks/results.json`). And when a latent's
dataset peak word appears in an F2 text, the latent fires on that text 0.326 of the time for dropped latents, against
0.786 for kept latents (`summary.json` `word_transfer_F2`; 46 and 28 occurrences). Both say the dropped latents need
more than the subject word. Neither says exactly what they track.

**The SAE is not short of style-robust topic latents. The generator's rules excluded most of them**
(`verify_mechanism/summary.json` `q2_all_topic`, `name_filter_crosstab`; `STEP3.md` section 2.3).
- **All 201/201 topic concepts** have at least one latent with AUROC_C and AUROC_F2 >= 0.85. But 4,208 of these
  6,015 robust latents have a different best dataset concept, so the relevant count is the next one.
- 188/201 have a robust latent whose best dataset concept is the concept itself.
- Only 66/201 have one in the generator's pool, and 57/201 have one kept by A1.1.
- The generator v2 admits a topic latent only if it does not fire on its concept's own name and is specific (within
  0.10 of at most 3 concepts). 1,581 robust latents anchored to their own concept were left out of the pool. The name
  filter was involved in **1,321 / 1,581 = 0.84** of these exclusions: it was the only reason for 556 (0.35), it
  acted together with the specificity rule for 706 (0.45), and with other pool rules for 59.

The share of robust latents in each cell of the two rules (topic latents):

| Pool cell | Robust share |
|---|---|
| Admitted by v2: specific, silent on own name | 226/4741 = **0.048** |
| Not specific, silent on own name | 203/2064 = 0.098 |
| Specific, fires on own name | 556/2261 = 0.246 |
| Not specific, fires on own name | 706/1532 = 0.461 |

**Why this matters.** The name filter is an anti-shortcut rule: it stops an agent from solving a slot by typing
each option's name. The admitted cell has the lowest robust share of the four. The specificity rule alone halves the
share (0.098 to 0.048); the name filter cuts it much more (0.246 or 0.461 in the name-firing cells). Why name-firing
latents are more often style-robust was not tested. One plausible reading is that a latent that responds to the
subject also responds to the subject's name. Either way, rules meant to protect validity left a pool made mostly of
latents that fail on non-encyclopedic text.

---

## 5. Step 3: baselines on the filtered pool

### 5.1 What ran

- **Pool.** `instances_v2f`, 180 instances, pool sha256 `b212728f...`, with 0 manifest mismatches
  (`step3/BASELINES.md`).
- **Most baselines.** Reference, black-box, self_probe, template_probe and the zero-effort recipes ran **in-process**.
  That is one GPU job using the task's own solver code and tool environment, but not through the episode harness:
  no broker, no leak scan, no wall clock. Caps were enforced.
- **SR.** SR ran **offline** from cached bank-R activations.
- **Skeptic.** A separate skeptic re-ran SR live through the tool's own code path (section 5.8).
- **How SR works.** For each slot, SR runs 6 bank-R texts per option (120 texts) and claims the option with the
  highest mean activation (**SR-max**). **SR-thr** claims that option only if its AUROC against the other options'
  texts is >= 0.78; otherwise it answers "nothing found". SR has no hypothesis and does no reasoning
  (`SR_RECIPE.md`).

### 5.2 All tiers pooled

Source: `step3/baselines_public.json`, `step3/BASELINES.md`. n = 180 episodes, 441 planted and 284 null slots.

| Baseline | Planted acc | Wilson | Clustered | Pass [Wilson] | Planted NF | Null false claim |
|---|---|---|---|---|---|---|
| reference (one-shot) | 441/441 = 1.000 | [0.991, 1.000] | [1.000, 1.000] | 178/180 = 0.989 [0.960, 0.997] | 0.000 | 2/284 = 0.007 |
| reference (best-of-5) | 441/441 = 1.000 | [0.991, 1.000] | [1.000, 1.000] | 180/180 = 1.000 [0.979, 1.000] | 0.000 | 0/284 |
| **SR-max** | **385/441 = 0.873** | [0.839, 0.901] | [0.840, 0.903] | 17/180 = 0.094 [0.060, 0.146] | 0.000 | 284/284 = 1.000 |
| **SR-thr** | **322/441 = 0.730** | [0.687, 0.769] | [0.688, 0.771] | 68/180 = 0.378 [0.310, 0.451] | 0.229 | 37/284 = 0.130 |
| SR-max-capped (A2.3) | 385/441 = 0.873 | [0.839, 0.901] | [0.840, 0.903] | 17/180 = 0.094 | 0.000 | 1.000 |
| SR-thr-capped (A2.3) | 323/441 = 0.732 | [0.689, 0.772] | [0.691, 0.772] | 68/180 = 0.378 | 0.225 | 39/284 = 0.137 |
| SR-max-all20 (over budget) | 425/441 = 0.964 | [0.942, 0.978] | [0.945, 0.980] | 28/180 = 0.156 | 0.000 | 1.000 |
| SR-thr-all20 (over budget) | 367/441 = 0.832 | [0.794, 0.864] | [0.798, 0.866] | 104/180 = 0.578 [0.505, 0.648] | 0.163 | 21/284 = 0.074 |
| black-box control | 0/441 = 0.000 | [0.000, 0.009] | [0.000, 0.000] | 7/180 = 0.039 | 1.000 | 0.000 |
| self_probe | 90/441 = 0.204 | [0.169, 0.244] | [0.168, 0.241] | 8/180 = 0.044 | 0.746 | 16/284 = 0.056 |
| template_probe | 127/441 = 0.288 | [0.248, 0.332] | [0.246, 0.330] | 21/180 = 0.117 | 0.678 | 20/284 = 0.070 |
| nothing | 0.000 | [0.000, 0.009] | | 7/180 = 0.039 | 1.000 | 0.000 |
| always_claim | 13/441 = 0.029 | [0.017, 0.050] | | 0/180 | 0.000 | 1.000 |
| prior (unfiltered v2 prior) | 17/441 = 0.038 | [0.024, 0.061] | | 0/180 | 0.000 | 1.000 |
| prior (filtered-draw prior; not pre-registered) | 26/441 = 0.059 | [0.041, 0.085] | | 0/180 | 0.000 | 1.000 |
| random | 6/441 = 0.014 | [0.006, 0.029] | | 1/180 = 0.006 | 0.578 | 0.415 |
| name_probe | 13/441 = 0.029 | [0.017, 0.050] | | 0/180 | 0.000 | 1.000 |
| name_probe_thr | 0/441 = 0.000 | [0.000, 0.009] | | 3/180 = 0.017 | 0.927 | 0.095 |
| vocab_match | 23/441 = 0.052 | [0.035, 0.077] | | 4/180 = 0.022 | 0.875 | 0.056 |

How to read the table:
- **SR-max never says "nothing found".** It therefore claims every null slot and passes almost no episode (0.094),
  even though its planted accuracy is high. That is why the stop rule uses planted accuracy.
- **No baseline without activations did better than about chance (0.05).** The black-box control is exactly 0 on
  planted slots because it never claimed (planted NF 1.000), so it gave the same answers as "nothing". The priors
  score at most 0.059, and vocab_match, which reads the latent's weights but runs no text, 0.052. Only a learner
  trained on the answers of many episodes from this pool did better (section 6). The question this study asks is
  whether a fresh solver needs reasoning on top of reading activations.
- **name_probe scores near 0 by construction.** The generator drops every latent that fires on its own concept's
  name, so the true option's name fires on 0/441 planted slots (`step3_skeptic/skeptic_summary.json`).

### 5.3 By tier

T1 = far menus with 1200 forward units, T2 = close menus with 1200, T3 = close menus with 550. Source:
`step3/BASELINES.md`, `STEP3.md` section 3.2.

| Planted accuracy | T1 | T2 | T3 |
|---|---|---|---|
| reference | 153/153 = 1.000 [0.976, 1.000] | 149/149 = 1.000 [0.975, 1.000] | 139/139 = 1.000 [0.973, 1.000] |
| SR-max | 137/153 = 0.895 W[0.837, 0.935] C[0.845, 0.939] | 136/149 = 0.913 W[0.857, 0.948] C[0.862, 0.955] | 112/139 = 0.806 W[0.732, 0.863] C[0.737, 0.871] |
| SR-thr | 111/153 = 0.726 W[0.650, 0.790] | 115/149 = 0.772 W[0.698, 0.832] | 96/139 = 0.691 W[0.610, 0.761] |
| black-box | 0/153 | 0/149 | 0/139 |
| self_probe | 33/153 = 0.216 [0.158, 0.287] | 34/149 = 0.228 [0.168, 0.302] | 23/139 = 0.166 [0.113, 0.236] |
| template_probe | 37/153 = 0.242 [0.181, 0.316] | 55/149 = 0.369 [0.296, 0.449] | 35/139 = 0.252 [0.187, 0.330] |

| Pass (n = 60 per tier) | T1 | T2 | T3 |
|---|---|---|---|
| reference | 60/60 | 60/60 | 58/60 |
| SR-max | 5/60 | 9/60 | 3/60 |
| SR-thr | 18/60 [0.199, 0.425] | 27/60 [0.331, 0.575] | 23/60 [0.271, 0.510] |
| black-box | 1/60 | 1/60 | 5/60 |
| self_probe | 4/60 | 1/60 | 3/60 |
| template_probe | 6/60 | 9/60 | 6/60 |

**T3 is the hardest tier for SR on topic slots:** SR-max 78/105 = 0.743 and SR-thr 62/105 = 0.591 W[0.495, 0.680].
SR-thr's interval on T3 topic slots touches 0.50. The stop rule is judged on all tiers pooled, so this does not
change it, but it is the weakest cell.

**T3 budget (A2.3).** Uncapped SR spends more than the T3 cap of 550 in 22 of 60 T3 episodes
(`step3/sr_recipe_out/summary.json`). The capped variant changes pooled planted accuracy by at most 0.002: on T3,
SR-max is unchanged and SR-thr moves from 0.691 to 0.698.

### 5.4 By family (A3.2)

Source: `step3/BASELINES.md`, `step3/baselines_public.json`.

| Planted accuracy | Topic (n = 328) | Language (n = 113) |
|---|---|---|
| reference | 1.000 [0.988, 1.000] | 1.000 [0.967, 1.000] |
| **SR-max** | **272/328 = 0.829** W[0.785, 0.866] C[0.787, 0.870] | 113/113 = 1.000 [0.967, 1.000] |
| **SR-thr** | **209/328 = 0.637** W[0.584, 0.687] C[0.585, 0.687] | 113/113 = 1.000 [0.967, 1.000] |
| SR-max-all20 | 315/328 = 0.960 | 110/113 = 0.974 |
| black-box | 0.000 | 0.000 |
| self_probe | 73/328 = 0.223 [0.181, 0.271] | 17/113 = 0.150 [0.096, 0.228] |
| template_probe | 20/328 = 0.061 [0.040, 0.092] | 107/113 = 0.947 [0.889, 0.975] |

- **Language slots are trivial for scripts.** Both SR variants get 113/113. template_probe's three fixed everyday
  sentences per language get 0.947.
- The filter tripled the language share of slots (section 4.3), which inflates the pooled number.
- **Topic slots are the real test, and SR passes them as well.**

### 5.5 Gate

Source: `step3/baselines_public.json` `gate`.

| Baseline | Planted acc [Wilson] | Bar | Gate |
|---|---|---|---|
| reference (one-shot) | 1.000 [0.991, 1.000] | >= 0.95 | meets |
| self_probe | 0.204 [0.169, 0.244] | <= 0.15 | **fails** (topic alone 0.223 [0.181, 0.271]) |
| template_probe | 0.288 [0.248, 0.332] | <= 0.15 | **fails** (all of it from language slots) |
| all other pre-registered recipes and the black-box control | <= 0.052 (the largest is vocab_match, 0.052 [0.035, 0.077]) | <= 0.15 | pass |
| prior on a filtered draw (added, not pre-registered) | 0.059 [0.041, 0.085] | <= 0.15 | pass |

**Gate verdict: not met.** The gate has no separate consequence in `PREREG.md`, but it points the same way as the
stop rule: cheap probing with no reasoning already answers part of the planted slots correctly.

### 5.6 Stop rule, with every label

Source: `step3/baselines_public.json` `stop_rule`, re-derived from raw files in `step3_skeptic/skeptic_summary.json`.

| Quantity | SR-max | SR-thr |
|---|---|---|
| Planted accuracy, all tiers, primary text selection (A2.1), n = 441 | **0.873** W[0.839, 0.901] C[0.840, 0.903] | **0.730** W[0.687, 0.769] C[0.688, 0.771] |
| Above 0.50? Wilson CI includes 0.50? | yes / no | yes / no |
| A2.2 spread over 42 text selections, primary included (min / p10 / median / p90 / max) | 0.862 / 0.878 / 0.907 / 0.932 / 0.955 | 0.689 / 0.724 / 0.757 / 0.818 / 0.873 |
| A2.2 spread, topic slots only | 0.829 / 0.860 / 0.889 / 0.923 / 0.942 | 0.619 / 0.650 / 0.694 / 0.771 / 0.832 |
| All 20 bank-R texts per option (over budget, descriptive) | 0.964 | 0.832 |
| A2.3 capped | 0.873 | 0.732 [0.689, 0.772] |
| A3.2 topic only, n = 328 | **0.829** [0.785, 0.866] | **0.637** [0.584, 0.687] |
| A3.2 language only, n = 113 | 1.000 [0.967, 1.000] | 1.000 [0.967, 1.000] |
| Live re-run through the tool path (skeptic) | 386/441 = 0.875 [0.841, 0.903] | 322/441 = 0.730 |

The labels, one by one:
- **Fires: yes.** Both variants are above 0.50, with n_planted = 441 >= 100.
- **Borderline: no.** Neither Wilson CI includes 0.50.
- **Text-selection-sensitive (A2.2): no.** The 10th percentile is above 0.50 for both variants, pooled and within
  each family. The primary selection sits at the low end of the spread. For SR-max on topic slots it is the
  minimum (0.829).
- **A2.3 (T3 budget).** Respecting the T3 cap changes pooled accuracy by at most 0.002 (section 5.3).
- **A2.4 (SR-thr's AUROC).** SR-thr compares the chosen option's texts with the other options' bank-R texts, while
  the reference compares with a background sample. The shared 0.78 threshold is therefore applied to slightly
  different quantities. Noted only.
- **A2.5 (clustered CI).** Reported next to every Wilson CI. As pre-registered, "borderline" is judged on Wilson.
- **A3.1 (P6).** The filtered pool failed the P6 fingerprint check at 0.616 (bar 0.60; 0.683 [0.674, 0.693] on a
  larger held-out test, section 6). **This cannot have raised a scripted recipe's accuracy.** SR's pick uses only the
  activations on its own probe texts and learns nothing across episodes. No zero-shot rule on the menu labels that
  was tried separates planted from null slots better than AUROC 0.516 (section 6).
- **A3.2 (topic-only sentence).** The stop rule fires on the pooled number, and both variants are **also above 0.50
  on topic slots alone** (SR-max 0.829, SR-thr 0.637, n = 328). So the "fires pooled but not on topic" case of A3.2
  does not arise. The weakest topic cell is T3, where SR-thr is 0.591 W[0.495, 0.680] (section 5.3).
- **A3.3.** Step 2 was reproduced exactly (section 4.4), so these numbers are final, not provisional.
- **Verdict label (section 4): Not interpretability.**

For comparison, the SR dry run on the **unfiltered** v2 pool gave SR-max 0.520 [0.473, 0.567]. That is not a
pre-registered quantity (`sr_recipe_out_v2_unfiltered_DRYRUN/summary.json`). Its point estimate was also above
0.50, though borderline.

### 5.7 The skeptic's re-check

Source: `step3_skeptic/skeptic_summary.json`. An adversarial agent tried to break the step-3 result from the raw
files.
- **Re-grading.** It re-graded all 3,603 stored episodes with 0 differences. 30/30 of a seeded sample re-graded
  through the grader's command line match. All 1,260 published fields of `baselines_public.json` reproduce.
- **Independent SR.** It wrote its own SR from scratch: the same decision on 725/725 slots, and the same 42-selection
  spread.
- **Bank overlap.** Banks F and R share 0 identical texts. The median character-5-gram overlap between an R text
  and its closest same-concept F text is 0.06. 13 of 4,640 R texts overlap >= 0.5 with an F text.
- **Prediction labels.** It found the builder's labels too generous: P7 was called a hit, but both predicted values
  lie outside the outcome CIs. Section 7 uses the stricter rule.
- **Its verdict:** "The builder's stop-rule conclusion STANDS."

### 5.8 Offline SR equals live SR

**Why this check matters.** SR was computed from cached activations, not by calling the tool. The cache was built
in batches of 32 and stored as float16. The tool runs batches of up to 16 texts in float32 for the requested latents
only, then rounds. Small numeric differences can flip a near-tie (`SR_RECIPE.md`).

**The check.** The skeptic re-ran all 87,000 SR (text, latent) values through the tool's own `token_acts` path
(`step3_skeptic/skeptic_summary.json`):
- Absolute difference: median 0.0, 99th percentile 0.128, maximum 6.54.
- Same SR-max pick on **712/725** slots, and the same SR-thr choice on 723/725.
- Planted accuracy live: SR-max **386/441** (offline 385), SR-thr 322/441 (offline 322).
- Topic only, live: SR-max 273/328.
- The same check for template texts gives 118/120 identical picks.

**Conclusion.** The offline numbers stand for a live run.

### 5.9 New cheap recipes

The skeptic fixed new no-reasoning recipes in code before running them (`step3_skeptic/cheap_texts.py`), except
`tmpl7_posthoc`, which was added after seeing the others. Each claims the option with the highest mean activation
over its texts.

Sources: `step3_skeptic/skeptic_summary.json` (`new_recipes_v2f`, `new_recipes_unfiltered_v2`, `budgets_v2f`),
`step3_skeptic/cheap_texts.py`, `step3_skeptic/cheap_recipes.py`.

| Recipe | Probe texts per option | v2f planted (n = 441) | v2f topic (n = 328) | v2f language (n = 113) | Within budget on v2f? | Unfiltered v2 planted (n = 431) |
|---|---|---|---|---|---|---|
| **gen1_cap** | gemma writes 1 text from a fixed prompt with the task's own `generate` tool, until the 40-call generate cap is spent (later options get no text) | **252/441 = 0.571 [0.525, 0.617]** | 177/328 = 0.540 [0.486, 0.593] | 75/113 = 0.664 | **yes**, every cap | 190/431 = 0.441 [0.395, 0.488] |
| gen_all | gemma writes 2 texts per option | 297/441 = 0.673 [0.628, 0.716] | 0.662 | 0.708 | no: over the generate cap in 180/180 episodes | 252/431 = 0.585 [0.538, 0.630] |
| encyc4_max | template_probe's 4 encyclopedia templates (languages: its 3 everyday sentences) | 238/441 = 0.540 [0.493, 0.586] | 125/328 = 0.381 | 113/113 = 1.000 | yes | 163/431 = 0.378 [0.334, 0.425] |
| tmpl7_posthoc (added after seeing results) | encyc4 + style3 | 243/441 = 0.551 [0.504, 0.597] | 0.396 | 1.000 | no: over the forward cap in 25 episodes | 171/431 = 0.397 |
| style3 | 3 templated sentences in 3 styles | 161/441 = 0.365 [0.322, 0.411] | 0.146 | 1.000 | yes | 84/431 = 0.195 |
| generic_bare | "Here is something about a noble." | 32/441 = 0.073 | 0.082 | 0.044 | yes | 36/431 = 0.084 |
| generic_label | "Here is something about \<label\>." | 16/441 = 0.036 | 0.037 | 0.035 | yes | 30/431 = 0.070 |
| name3 | max over 3 name forms | 13/441 = 0.029 | 0.034 | 0.018 | yes | 19/431 = 0.044 |

**What this shows.**
- **The conclusion is supported without bank R**, by a recipe that was not pre-registered. `gen1_cap` uses only the
  task's own `generate` tool and stays within every budget. It gets 0.571 [0.525, 0.617] on planted slots and 0.540
  [0.486, 0.593] on topic slots alone, whose interval includes 0.50. It passes only 5/180 = 0.028 of whole episodes,
  because it never says "nothing found". A fresh agent could run the same recipe.
- **The filter made cheap solvability worse but did not create it.** On the unfiltered v2 pool the in-budget
  `gen1_cap` already got 0.441 [0.395, 0.488], well above the 0.15 gate though below the 0.50 stop bar. `gen_all`
  got 0.585, but it goes over the `generate` cap (in 180/180 v2f episodes; its budget on unfiltered v2 was not
  reported). `PREREG.md` section 3 also records a v2 self_probe planted accuracy of 0.16, recomputed from recorded
  runs (`fx/recipes.py`). So the gate was probably never met on v2 either. Earlier gate passes relied on recipes with
  a threshold or a shortlist (template_probe, self_probe). The only max-pick recipe on the gate list, name_probe,
  probes bare names, which the name filter makes useless (`skeptic_summary.json` `discrepancies`).
- **Writing about the concept works; naming it does not.** One sentence that names the option scores near chance
  (0.036), because of the name filter. Short texts *about* an instance of the concept score far higher.
- **The "nothing found" decision is where scripts lose whole episodes.** With a claim threshold calibrated on a
  separate 300-instance pool, `gen1_cap` passes 21/180 = 0.117 [0.078, 0.172] of episodes, and `encyc4_max` passes
  26/180 = 0.144.

### 5.10 How much of the pass rate does style explain? (PREREG A1.5, required section)

**What was pre-registered.** A1.5 asked for two shares on the core agent set:
- Q_SR: how much of arm A's planted accuracy SR reproduces;
- Q_D: how much of the gap to the reference closes when the agent is given in-style examples (arm D).

Arms A and D never ran, so neither share exists. For this case A1.5 says: *"If the stop rule fired, arms A and D do
not exist: only pa(SR-max) / pa(ref) and pa(SR-thr) / pa(ref) on the whole filtered pool are reported, and the rule
below uses them."* Planted accuracy is the measure, because SR-max never answers "nothing found", so its pass rate
is near 0 by construction.

**The numbers** (`step3/baselines_public.json`, `A1.5_stop_rule_case`; paired instance bootstrap, 10,000 resamples):

| Ratio | Value | 95% CI |
|---|---|---|
| pa(SR-max) / pa(ref one-shot) = 0.873 / 1.000 | **0.873** | [0.840, 0.903] |
| pa(SR-thr) / pa(ref one-shot) = 0.730 / 1.000 | **0.730** | [0.688, 0.771] |

The reference is exactly 1.000, so each ratio equals SR's planted accuracy, and its CI equals SR's clustered CI.

**Spread over 42 text selections, primary included** (A2.2): SR-max 0.862-0.955 (10th-90th percentile
0.878-0.932). SR-thr 0.689-0.873 (0.724-0.818). The primary selection sits at the low end.

**Topic slots alone** (A3.2): SR-max 0.829 [0.785, 0.866], SR-thr 0.637 [0.584, 0.687]. Both are above 0.50 there
too.

**The binding sentence.** Both ratios are above 0.50 and neither CI includes 0.50, so the sentence applies without
"(borderline)":

> **FeatureMatch is mostly measuring style, not interpretability**: pa(SR-max)/pa(ref) = 0.873 [0.840, 0.903] and
> pa(SR-thr)/pa(ref) = 0.730 [0.688, 0.771] (n = 441 planted slots).

**What "style" means here, in plain words.** Once the answer key is valid across writing styles, scripts with no
hypothesis about the latent pick the right option on many planted slots. All numbers below are planted accuracy:
- held-out dataset-style texts, about 719 forward units per episode: the reference, a script, gets 1.000;
- LLM-written styled texts, 6 per option (about 483 units per episode): SR-max gets 0.873; with all 20 per option
  (over budget) 0.964;
- the subject model's own completions, at most one per option and none for options after the `generate` cap is
  spent: `gen1_cap` (not pre-registered) gets 0.571, and 0.540 [0.486, 0.593] on topic slots;
- other fixed texts with the same max-pick rule: encyclopedia templates 0.540, three templated styled sentences
  0.365, one generic sentence 0.073 (`step3/BASELINES.md`; `step3_skeptic/skeptic_summary.json`).

Among these scripts, accuracy rises with how close the probe texts are to the texts the key was checked on, and with
how many texts each option gets. Neither involves reasoning about what the latent does. An agent that ran SR's
procedure with texts like bank R's would be expected to score near SR, whether or not it understood the latent. With
other text sources the same max-pick rule scored from 0.029 (the options' bare names) to 0.571 (section 5.9). How
real agents would score is unknown: none ran.

**One limit, stated fairly.** On whole-episode pass rate a no-reasoning script does not match the reference: SR-thr
passes 68/180 = 0.378, against 178/180 = 0.989 for the reference. The gap is mostly in the "nothing found" decision:
SR-thr wrongly claims 0.130 of null slots, against 0.007 for the reference, and says "nothing found" on 0.229 of
planted slots, against 0.000 (it misses 0.270 of planted slots in all). The reference makes that decision with
held-out dataset texts, which an agent never sees, and its score is partly selected by the filter (section 9).
SR-thr with all 20 texts passes 104/180 = 0.578. Whether what remains is a calibration problem or needs
interpretation was not tested, because no agent ran.

---

## 6. Fingerprint and memorisation

**What a fingerprint is.** Suppose a classifier can tell planted from null slots by looking only at the menu, without
running the model. Then a policy trained on many episodes could learn to guess "nothing found" from the menu alone.
P6 required that such a classifier score a cross-validated AUROC <= 0.60, where 0.5 means it learns nothing.

**P6 failed, and the larger test is worse** (`verify_fingerprint/reproduce.json`).
- Under the P6 protocol, the close tier scores **0.616** (sd 0.014; 2,415 slots), against 0.531 for unfiltered v2.
- The P6 "T3" number is the T2 draw again: the same seed gives identical slots, so it is not a second measurement.
- 10 fresh blocks of 600 instances give a mean of 0.638, and all 10 are above 0.60. Unfiltered v2 gives a mean of
  0.538, and none are above 0.60.
- With 5,000 training and 3,000 test instances: **0.683 [0.674, 0.693]**, against v2 0.570 [0.560, 0.581].
- **Where the signal comes from.** It is spread over many weak features. The largest grouped importance is the
  menu's closeness statistics, at 0.126 (`verify_fingerprint/residual.json`).
- **Untested hypothesis.** The menu universe shrank to 34-63 concepts per layer (v2: 108-149). With only 34-37
  concepts at L6 and L12, a close menu covers most of the anchor's sibling group, so whether c\* is on the menu
  shows in the menu's profile (`STYLE_FILTER.md`).

**The bigger problem is memorisation.** The filtered generator draws from only 354 kept latents, and its pool uses
74 anchor concepts, so they recur. The study simulated learners that see only the option labels, never the model,
and are trained on generated episodes **with every slot's correct answer** (up to 5,000 training episodes). That is
richer feedback than RL with an episode-level reward, so it shows what can be learned, not how fast an RL policy
would learn it (`verify_fingerprint/labelonly.json`, `knnsplit.json`, `policy.json`, `residual.json`):

| Label-only learner, filtered close tier | Result |
|---|---|
| Planted vs null from labels (bag of labels + string features) | AUROC **0.898** [0.893, 0.904] (unfiltered v2: 0.641) |
| Same, on slots whose latent was seen in training vs unseen (`residual.json` `bag_str_hgb_memorization_splits`) | 0.927 vs 0.580 |
| Zero-shot label rules, no training | <= 0.516 |
| Picking the answer by nearest neighbour on labels, all planted slots, 5,000 training episodes | **0.577** [0.566, 0.589] (chance 0.05) |
| Same, after 30 / 100 / 300 / 1000 / 5000 training episodes | 0.092 / 0.217 / 0.367 / 0.489 / 0.577 |
| Split experiment (`knnsplit.json`): train on half the latents, test planted slots of seen vs held-out latents | **0.676** [0.661, 0.691] vs **0.057** [0.049, 0.065] |
| Same split by anchor concept: seen vs held out | 0.686 vs 0.0005 |
| Complete label-only policy (classifier + nearest neighbour), pass rate on 3,000 fresh close-tier (T2) episodes | **0.164** [0.151, 0.178] (always "nothing found": 0.031; unfiltered v2 label-only policy: 0.041) |
| Same policy on fresh far-tier (T1) episodes | 0.032 [0.026, 0.039] (always "nothing found": 0.031) |

**What this means.**
- A fresh zero-shot agent is unlikely to exploit this. The zero-shot label rules tried score <= 0.516.
- A learner trained on many episodes of this pool can. In simulation, with full answer labels, the nearest-neighbour
  answer-picker reaches 0.367 after 300 training episodes and 0.577 after 5,000, by recognising menus rather than
  reading latents. As a complete policy it passes 16% of fresh close-tier episodes, about 4 times the unfiltered v2
  figure; on the far tier it does no better than always answering "nothing found". Whether and how fast an RL
  policy with episode-level reward would find this shortcut was not tested.
- **Holding out latents from training removes the answer-picking leak**: in the split experiment, 0.676 on seen
  latents against 0.057 on held-out latents (chance 0.05). Holding out anchor concepts removes it entirely (0.0005).
  Telling planted from null slots from labels still scores AUROC 0.580 on unseen latents, so holding out latents
  does not remove every label signal.
- **A candidate menu fix, `swap_nn`, was simulated only.** It builds every close menu as the null menu would be
  built and, on a planted slot, swaps c\* in for the option closest to it in the taxonomy. It was not pre-registered
  or applied (`verify_fingerprint/fix.json`, `policy.json`).
  - It brings the P6 protocol to 0.561 and 0.577 on two seed sets.
  - On the held-out test it gives 0.588 [0.578, 0.599], and 0.592 [0.582, 0.602] on fresh seeds. The upper bounds
    touch 0.60, so this is "met (point estimate), not robust".
  - It leaves memorisation in place: the label-only policy still passes 0.142 [0.130, 0.155].

---

## 7. Predictions vs outcomes

**Scoring rule** (from `STEP3.md`, stricter than the builder reports):
- A point prediction **hits** if the predicted value lies inside the outcome's 95% CI.
- A range prediction hits if the point estimate is inside the range.
- A yes/no prediction hits if the outcome matches.
- A prediction with several parts is **partial** when some parts hit and some miss.

Predictions that needed agent data are **not tested**, because the stop rule fired.

| # | Prediction (`PREREG.md`) | Outcome (source) | Result |
|---|---|---|---|
| P1 | Step 1 finds >= 4 undocumented items (response keys, budget counting, tool roles, JSON quoting) | 17 ambiguities, including all four (`clarity.md`) | **hit** |
| P2 | >= 1 of 4 valid v1 transcripts considered and dropped `top_latents`/`generate` | Transcripts hold no reasoning text, so this cannot be answered. No agent called either tool (`clarity.md`) | **not testable** |
| P3 | Share with auroc_F >= 0.85: 0.45 (0.30-0.60); languages 0.80, topics 0.35 | 430/5017 = 0.086 [0.078, 0.094]; languages 0.649; topics 0.053 (`verify_reimpl/result.json`) | **miss** |
| P4 | Survival L6 0.35 / L12 0.50 / L18 0.55, deeper more robust | auroc_F 0.091 / 0.052 / 0.117; kept 0.072 / 0.042 / 0.099. L12 is lowest, so not monotone | **miss** |
| P5 | "Filter selects easier latents": yes; margin_C +0.03 to +0.06; higher fire on c\*; density about 1.1x | No. Margin −0.040, density 0.70x, reference +0.017. Fire on c\*'s bank-F texts, median 0.95 vs 0.15 (means 0.906 vs 0.230) (`style_filter_out/summary.json`) | **partial** (fire on c\* hits; the rest misses) |
| P6 | Filtered pool passes the fingerprint check (<= 0.60) | 0.616; held out 0.683 [0.674, 0.693] | **miss** |
| P7 | Reference planted 0.97, one-shot pass 0.88 | 1.000 [0.991, 1.000]; 0.989 [0.960, 0.997] | **miss** (both higher) |
| P8 | Black-box planted 0.00, pass about 0.05 (all-null episodes only) | 0.000 [0.000, 0.009]; 7/180 = 0.039 [0.019, 0.078], exactly the 7 all-null episodes | **hit** |
| P9 | self_probe planted 0.30 (planted NF 0.55); gate fails | 0.204 [0.169, 0.244]; NF 0.746 [0.703, 0.784]; gate fails | **partial** (gate hits; both levels miss) |
| P10 | template_probe planted 0.20 (NF 0.65); gate fails or borderline | 0.288 [0.248, 0.332]; NF 0.678 [0.633, 0.720]; gate fails | **partial** (gate and NF hit; level misses) |
| P11 | SR-max planted 0.75 (0.55-0.85) | 0.873 [0.839, 0.901] | **miss** (above the range) |
| P12 | SR-thr planted 0.70 / null false claims 0.10 / pass 0.35 | 0.730 [0.687, 0.769] / 0.130 [0.096, 0.174] / 0.378 [0.310, 0.451] | **hit** |
| P13 | Stop rule fires (p about 0.7) | Fires; not borderline; not text-selection-sensitive; both variants also above 0.50 on topic slots alone | **hit** |
| P14 | Arm A pass 0.25 | arm A did not run | not tested (stop rule) |
| P15 | Arm A planted 0.65 / null false claim 0.20 / planted NF 0.15 | | not tested (stop rule) |
| P16 | Arm A budget used at submit, median 0.35 | | not tested (stop rule) |
| P17 | Arm A failures: W2 45%, W1 5%, W3 5%, W4 <= 5%, right-reason 40%; criterion 2 fails in A | | not tested (stop rule) |
| P18 | Arm A uses `top_latents` and `generate` in <= 20% of episodes each | | not tested (stop rule) |
| P19 | Arm B (old prompt) pass 0.20 (A − B +0.05, not detectable at n = 20); W3 share about 15% | | not tested (stop rule) |
| P20 | Arm C (persistence) pass / planted accuracy / f_fwd 0.35 / 0.75 / 0.65; W2 about 0%; right-reason share 0.70 (criterion 2 met) | | not tested (stop rule) |
| P21 | Arm D (oracle examples) 0.45 / 0.82 / 0.05 | | not tested (stop rule) |
| P22 | E-sim pass: random 0.30 / sibling 0.25 / near-sibling 0.18 | | not tested (stop rule) |
| P23 | Criterion 4 right direction, not significant ("ambiguous") | | not tested (stop rule) |
| P24 | Planted accuracy by dis_C tertile 0.80 / 0.72 / 0.62 | | not tested (stop rule) |
| P25 | E-budget pass: low 0.20 / high 0.27 | | not tested (stop rule) |
| P26 | E-density planted: dense 0.55 / sparse 0.72 (pass 0.18 / 0.30) | | not tested (stop rule) |
| P27 | Arm F (haiku) pass / planted / planted NF / f_fwd 0.08 / 0.40 / 0.35 / 0.20; right-reason share 0.30 | | not tested (stop rule) |
| P28 | A vs F: +0.17 pass; McNemar p < 0.05 | | not tested (stop rule) |
| P29 | Spot-check agreement >= 85%; attribution accuracy >= 85% | | not tested (stop rule) |
| P30 | Overall verdict: "does not require interpretability" via the stop rule (p about 0.7) | Stop rule fired; label Not interpretability | **hit** |
| P31 | k_ms = c\* for 0.90; languages 0.98, topics 0.87 | 0.764 [0.752, 0.776]; 0.884; 0.757 (`verify_reimpl/result.json`) | **miss** |
| P32 | k_ho = c\* for 0.85 | 0.719 [0.706, 0.731] | **miss** |
| P33 | Filter pass 0.42 (plausible 0.25-0.60) | 408/5017 = 0.081 [0.074, 0.089]. For information: the C part passes 0.954, the F2 part 0.082 | **miss** (see the note below the tally) |
| P34 | Kept 0.38 (0.22-0.55); languages 0.75, topics 0.30 | 0.071 [0.064, 0.078]; 0.591; 0.040 | **miss** |
| P35 | Dropped only for a key disagreement <= 5% | 54/5017 = 0.011 | **hit** |
| P36 | About 60% of v2 slots fail, planted and null within 5 points | 0.829 [0.800, 0.855]; planted 0.831, null 0.827 | **partial** (level misses; parity hits) |
| P37 | Arm G (disclosure) pass 0.32 (A 0.25); difference +0.07, paired CI includes 0 | | not tested (stop rule) |
| P38 | Arm G 0.72 / 0.22 / 0.10 | | not tested (stop rule) |
| P39 | Arm G W1 share half of A's; right-reason 0.45 | | not tested (stop rule) |
| P40 | Arm A tool use: `top_latents` 0.35, `generate` 0.25, `vocab_projection` 0.75, `next_token_logits` 0.05 | | not tested (stop rule) |
| P41 | Arm A, used on the target latent: `vocab_projection` on a slot latent in >= 95% of its calls; `top_latents` at the slot's layer in 90% of using episodes, target surfaced in 40% of them; a `generate` completion used as a probe in 30% of using episodes | | not tested (stop rule) |
| P42 | Share of episodes using `top_latents` / `generate`: B 0.10 / 0.10; C 0.45 / 0.35; D 0.20 / 0.15; F 0.10 / 0.10; G 0.40 / 0.30 | | not tested (stop rule) |
| P43 | Planted accuracy with vs without `top_latents` target surfaced: 0.80 vs 0.62 | | not tested (stop rule) |
| P44 | Q_SR = 1.15; pa(SR-max)/pa(ref) = 0.77 | Q_SR cannot be computed (no arm A). pa(SR-max)/pa(ref) = 0.873 [0.840, 0.903] | **miss** on the computable part |
| P45 | Q_D = 0.53 | no arm D | not tested (stop rule) |
| P46 | The plain-language rule fires (p about 0.75) | Fires, through the stop-rule case | **hit** |

**Tally.**

| Result | Count | Predictions |
|---|---|---|
| hit | 7 | P1, P8, P12, P13, P30, P35, P46 |
| partial | 4 | P5, P9, P10, P36 |
| miss | 10 | P3, P4, P6, P7, P11, P31, P32, P33, P34, P44 (computable part) |
| not testable | 1 | P2 |
| not tested (stop rule) | 24 | P14-P29, P37-P43, P45 |

**Note on P33.** `STEP3.md` scores P33 as partial, counting "AUROC_C(c\*) >= 0.85 fails rarely" as a second part.
In `PREREG.md` A1.7 that phrase is in the Basis column, not the Prediction column, which holds only "0.42 (plausible
0.25-0.60)". The outcome 0.081 [0.074, 0.089] misses both the point and the range, so this file scores it a miss.

**Our misses, and why we got them wrong.**
- **We greatly overestimated how many latents survive styled text** (P3, P4, P31-P34). We predicted roughly
  0.4 and got 0.07.
  - Our basis was the v1 hand-written-probe check: 50% of eligible latents recovered their concept there. The A1.1
    bar is much stricter: AUROC >= 0.85 against all 231 other concepts on 10 styled texts, plus key agreement.
  - We assumed dataset-selected topic latents mostly track the subject. Most track encyclopedia formatting
    (section 4.4). We did not foresee that the generator's own name filter would select for that.
- **We underestimated the cheap recipe** (P11: 0.75 predicted, 0.873 observed). Once latents must separate their
  concept on varied text against every other concept, picking 1 of 20 options with 6 varied texts each is easy.
- **We expected the fingerprint check to pass** (P6), because the filter treats planted and null latents alike. We
  did not foresee that the filter would shrink the menu universe to 34-63 concepts per layer. The untested
  hypothesis is that this small universe is what makes menus informative (section 6).
- **We expected the filter to favour broad, dense latents** (P5). It favoured sparser latents with smaller margins.
- **The reference beat its prediction** (P7). This is partly built in: the filter required AUROC_C >= 0.85, and the
  reference probes with split C (section 9).
- **The predictions about the overall outcome held** (P13, P30, P46): we gave the stop rule about 0.7, and it
  fired clearly.

---

## 8. What was not tested, and why

**Step 4 did not run.** Two separate pre-registered conditions each block it:
1. **The stop rule fired** (`PREREG.md` section 4: "Step 4 does not run").
2. **P6 is not fixed.** A3.1 allows agent arms only after a pre-registered menu fix brings the close-tier fingerprint
   to <= 0.60. The `swap_nn` candidate was never pre-registered. Its held-out upper bound reaches 0.60, and it does
   not stop memorisation (section 6).

**As a result, none of these exist:**
- Arms A (baseline), B (old prompt), C (persistence), D (oracle examples), E (similarity, budget and density dials),
  F (small model) and G (disclosure).
- The failure classifier (`classify.py`) and its human spot-check.
- Decision-rule criteria 1, 2 and 4. Criterion 3's inputs were measured for the record: r_ref = 0.000 and
  r_rec = 0.712 (`step3/baselines_public.json` `criterion3_inputs`).
- Tool-use metrics (A1.4), Q_SR and Q_D (A1.5), and predictions P14-P29, P37-P43 and P45 (and the Q_SR part of
  P44).

The infrastructure built for step 4 was never used: the revised and disclosure prompt templates, the harness prepare
options, and the shared model service. So we **do not know** how fresh agents would do on v2f. The stop rule does
not say they would pass. It says that most planted slots can be answered without interpreting the latent. It does not
say why an agent would fail. SR-thr itself misses 0.270 of planted slots and wrongly claims 0.130 of null slots, so
slots remain where an agent could fail, or succeed, for interpretive reasons. Where agents' failures would come from
needs agent data.

### 8.1 The three example transcripts the plan asked for

`PLAN.md` asked RESULTS.md for one right-reason failure, one wrong-reason failure and one success. **None exist on
v2 or v2f, because no agent ran on either pool.** The only recorded agent episodes are the **v1** probes analysed in
`docs/forensics/FEATUREMATCH_PROBE_FORENSICS.md`.

Before reading them, note how v1 differs:
- v1 is retired.
- It had close distractors and "word trap" latents that v2 removed.
- It was never style-filtered.
- Its tool logs truncate responses at about 4 KB.

The labels below are **my reading of the `PREREG.md` rules, not a scripted classification**. The truncated logs would
send some slots to manual review (rule U0).

**v1 success: ep2cd00804ab, slot 0** (Sonnet, T2, a planted "Band" latent at layer 12).
- Sonnet's own probes favoured Band (11.2 against 6.8), and it answered Band, which is right. The margin on
  held-out text was 0.338 (forensics, section 2).
- A cleaner example of interpretation is epea26c30f2c, slot 1 (gpt-6-luna, null slot). Its probes found that the
  latent fires on amphibian texts (33.0), a concept not on the menu, and it correctly answered "nothing found". The
  forensics call it the "best interp in these runs". The transcript holds no reasoning, so we see only the probes
  and the answer. That latent also fires on its own concept's name, so v2 would not use it.

**v1 wrong-reason failure: ep0f904f753d, slot 0** (gpt-6-luna, T2, a planted "MMA event" latent at layer 12).
- The agent answered "nothing found". All 50 of its probes read 0, including its 2 MMA sentences.
- The latent fires on 54/54 dataset texts that contain "mixed martial arts event" and on none of the agent's
  sentences: a style-dependent latent of exactly the kind step 2 filters out.
- The agent had used 12% of its forward budget, and 4 of its commands failed on apostrophe quoting (clarity A3).
- By the rule order of `PREREG.md` section 1.4 it would first go to manual review (U0), because its tool log is
  truncated. A reviewer applying the same definitions would most likely label it W2, early stop (12% of the budget
  used). It is not W1, which requires a latent that is style-robust on bank F; this one is the opposite.

**v1 closest to a right-reason failure: ep2cd00804ab, slot 1** (Sonnet, T2, a planted "Curler" latent at layer 6).
- Sonnet answered "nothing found" on a dense latent (78% fire rate) after one probe text per option. Its one curler
  text happened to be a 5th-percentile draw (4.39).
- The forensics estimate that 5 texts per option would have found the latent about 98% of the time.
- The episode used 62% of its budget, so it is not an early stop. The true option got fewer than 3 probes, so this
  is **R4s, "search failure"**. That counts as right-reason in the primary analysis and wrong-reason in the
  conservative one.

The other v1 misses look like right-reason confusions, for example grape answered as "cultivated variety" and
college coach as mayor. They come from an episode that stopped at 19% of its budget, so the rule priority would most
likely label them W2, early stop.

---

## 9. Threats to validity

**What "not interpretability" means here.** SR is the simplest kind of activation probing: run texts and read the
activations. No baseline without activations did better than about chance (0.05) on this pool: the black-box
control scored 0 because it never claimed, the priors at most 0.059, vocab_match 0.052. Only a learner trained on
the answers of many episodes of this pool did better (section 6). The finding is narrower than "no
interpretability involved". A fixed recipe with no hypothesis about the latent picks the right option on most
planted slots (SR-max 0.873). It passes only 0.094 of whole episodes, because it never says "nothing found". So, on
planted slots, the task cannot reliably tell an agent that understands a latent from one that runs the recipe. That
is what the stop rule was written to detect.

**Is SR's success built into the filter?** Partly, by design. The filter kept latents that separate their concept on
LLM-written styled text (bank F), and SR probes with LLM-written styled text (bank R). So the filter keeps latents
that are easy to find with varied text, and the filter raised SR from 0.520 (unfiltered dry run, not pre-registered)
to 0.873. Three points limit this worry:
- Bank R was written by a different writer agent and shares no identical text with bank F. The median 5-gram overlap
  with the closest same-concept F text is 0.06 (section 5.7).
- `gen1_cap` uses gemma's own completions, not either bank, and still gets 0.571 (not pre-registered; topic slots
  0.540 [0.486, 0.593]).
- On the **unfiltered** pool, the in-budget `gen1_cap` already got 0.441 [0.395, 0.488] and the over-budget
  `gen_all` 0.585. Encyclopedia templates got 0.378, styled templates 0.195 and generic sentences 0.070-0.084
  (section 5.9; `skeptic_summary.json` `new_recipes_unfiltered_v2`). The SR dry run got 0.520 [0.473, 0.567]
  (section 5.6).

So the filter made a weakness that was already there worse (`gen1_cap` 0.441 to 0.571, SR 0.520 to 0.873); it did
not create it.

**The reference's perfect score is partly selected.** The filter required AUROC_C(c\*) >= 0.85 and a held-out key
check on split C, and the reference probes with split C. Its 1.000 (P7) therefore partly reflects that selection.
This does not touch the stop rule, which uses SR's absolute accuracy. For the A1.5 ratio pa(SR)/pa(ref), the
selected-in reference score makes the ratio smaller, so it works against the conclusion, not for it.

**The baselines did not go through the episode harness.** They ran in-process (no broker, leak scan or wall clock)
or offline. Caps were enforced, and SR was confirmed live (386 vs 385 of 441). The harness-only checks (leak scan,
transcript audit) matter for agents, not for fixed scripts.

**Who rated.** The bank audit and the mechanism categories each had one LLM rater.
- The bank result holds up to that: 0/800 texts were off-concept.
- The mechanism categories are unblinded, partly use styled-text firing (the criterion that sorted the latents),
  and a scripted peak-token check agrees with the reader on only about 14/45 latents (`STEP3.md` section 2.3;
  `verify_mechanism/summary.json`). The firing-rate gap on styled texts (0.253 vs 0.817) follows from how latents
  were sorted, so it does not confirm the categories. The encyclopedia-style firing (0.402 vs 0.103) and the
  peak-word transfer (0.326 vs 0.786) are the independent support (section 4.4).
- A blind second rater would strengthen both.

**The pool is narrow.** It has 74 anchor concepts, and 27.3% of its slots are language slots, which are trivial for
scripts. The topic-only results are reported throughout, and both SR variants are above 0.50 on them alone.

**Weakest cell.** On T3 topic slots, SR-thr is 0.591 with Wilson [0.495, 0.680], touching 0.50. SR-max is 0.743
there. The pooled rule is pre-registered and far from its bar.

**Scope.** Everything here is one model (gemma-2-2b), one SAE family (Gemma Scope, 16,384 latents per layer) at
layers 6, 12 and 18, one concept collection (DBpedia classes plus languages) and one menu format. The findings may
hold elsewhere, but this study did not test that.

**Process deviations, all disclosed.**
- The first step-2 code did not follow A1.1. It was fixed before any filtered-pool result, and the fix is disclosed
  in A2.
- The SR dry run on unfiltered v2 was seen before A2.
- The filtered-draw prior baseline is an addition that was not pre-registered. It stays under the gate.
- Two amendment headers carry wrong times (section 2.2).
- `STEP3.md` says the unfiltered-v2 `gen1_cap` figure (0.441) was obtained "with thresholds calibrated on a separate
  pool". In `skeptic_summary.json`, 0.441 is the max-pick figure, which uses no threshold. The calibrated figure is
  0.339. This file uses the raw file.
- `STEP3.md` scores P33 as partial; this file scores it a miss (section 7).
- `clarity.md` (section 3) says "Every agent stopped with most of its budget left (8-62% of forward units used)".
  ep2cd used 62% (forensics, section 3), so this file says every agent submitted early, using 8-62%.

---

## 10. Sources

| Topic | Files (in `tasks/featurematch/diagnosis/` unless stated) | Commit |
|---|---|---|
| Brief and rules | `PLAN.md`, `PREREG.md` (sections 0-5, A1-A3, outcome pointer) | 0148033c, a4b8aecd, 72c1c1b3, 2f663caf, 22111857 |
| Task and baselines code | `tasks/featurematch/SPEC.md` (model, SAE width), `tasks/featurematch/blackbox_control.py` (claim threshold), `style_filter.py` (margin definitions) | 99e4023f, 2f7c799c, 3ea47d7f |
| Step 1 | `clarity.md`; `tasks/featurematch/agent_prompt_{old,revised,disclosure}.md` | 793458e6, a4b8aecd |
| Step 2 | `STYLE_FILTER.md`, `style_filter_out/summary.json`, `style_filter_out/v2f_summary.json`, `key_check.jsonl`, `slot_disagreements.jsonl` | 3ea47d7f |
| Re-implementation | `verify_reimpl/result.json`, `verify_reimpl/mismatches.jsonl` (empty) | 3eaccce1 |
| Bank audit | `verify_banks/PROTOCOL.md`, `verify_banks/results.json` | 54574065, fc81eff9, 30da4211 |
| Mechanism | `verify_mechanism/summary.json`, `verify_mechanism/q1_categories.json` | 854c5066 |
| Fingerprint | `verify_fingerprint/{reproduce,attribute,labelonly,knnsplit,policy,residual,fix,fixextra}.json` | e63f09a0 |
| SR recipe | `SR_RECIPE.md`, `sr_recipe_out_v2_unfiltered_DRYRUN/summary.json`, `step3/sr_recipe_out/summary.json` | 18d49dd1, a8d47fe8, 75f7524b |
| Step 3 | `step3/BASELINES.md`, `step3/baselines_public.json`, `step3/tables_autogen.md` | 75f7524b, 80fb525d |
| Skeptic | `step3_skeptic/skeptic_summary.json`, `step3_skeptic/cheap_texts.py`, `step3_skeptic/cheap_recipes.py` | 0fa8c454 |
| Integration | `STEP3.md` | 22111857 |
| v1 agents | `docs/forensics/FEATUREMATCH_PROBE_FORENSICS.md` | 61dbf4e4 |
| Project log | `docs/LOG.md` (main branch), entries 2026-10-01 17:30 to 2026-10-02 ~07:00 | 86eadd0a |
