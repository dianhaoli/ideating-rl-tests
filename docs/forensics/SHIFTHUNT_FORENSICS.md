# ShiftHunt forensics: agent skill or environment, and is it robust?

2026-10-03. CPU-only re-analysis of every graded ShiftHunt LLM-agent episode, plus the hidden grading grid and the
T1b / T2new pools. I ran no GPU jobs and changed nothing in the shifthunt worktree. A cross-family OpenAI run was in
progress on the same instances while I worked (section 6); I did not touch it.

**Disclosure rule.** The OpenAI arm is still running on the scaled-run instances, so under D6 this file gives
aggregates only. It names no instance, episode or latent id, and it does not say which attribute belongs to which
slot. The analysis outputs that do carry per-slot data stay in the session scratchpad, not in git.

**Where the sources are.** Run dirs are under `~/wt/shifthunt/runs/shifthunt/` (`runs/`). Task files are under
`~/wt/shifthunt/tasks/shifthunt/` (`task/`). This audit builds on, and does not repeat, `task/VALIDATION.md`,
`task/SCALE1_T1.md`, `task/SCALE1_T2.md` and their `_CHECK.md` files. Those reviewed every transcript. This one adds
measurements they did not make: hidden-grid noise, a split-half oracle, a no-interpretation padding test, a recipe
ceiling given free names, the truncation channel's strength, and answer-file reachability. An independent skeptic
recomputed the new numbers from raw arrays (section 8).

**Terms used below:**
- **Probe / slot:** each episode has 3-4 linear probes (slots). A **planted** probe relies on one of 8 listed surface
  cues. A **null** probe relies on none, and the right answer is "none".
- **Removal:** the share of the probe's dependence on the cue that the agent's ≤ 20 mean-ablated SAE latents remove,
  measured on a hidden balanced grid (topic × cue, about 48 texts per cell).
- **Bar (tau):** the removal a planted slot needs, frozen per cue family (0.164-0.345). **Margin** = removal − tau.
- **Oracle:** the 20 latents with the largest contribution to the cue dependence, chosen on one half of the hidden grid
  and scored on the other half. It is the best a perfectly informed agent could do with 20 latents.
- **top-|g|:** the 20 latents with the largest probe gradient. One tool call gives it, and it needs no interpretation.

## 1. Bottom line

1. **The grader is correct.** Regrading all 58 graded episodes reproduces every slot verdict. The infrastructure is
   healthy too. Unlike FeatureMatch, there is no queue starvation (longest wait 27 s), no early kills, and almost no
   tool errors. The only cap agents hit is the 4-query probe budget, which is intentional.
2. **For Opus 5.5 the interpretation part is saturated, and no Opus failure is a clean interpretation failure.**
   - Opus named the right cue on 110 of 110 planted slot-episodes, across smoke, T1 and T2.
   - It made 1 null over-claim in 52 null slot-episodes. That one came before the reliance scale was disclosed in
     TASK.md, and there have been 0 in 44 since.
   - The 7 removal misses split two ways:
     - **4 short latent lists** (9-11 of the 20 allowed). Padding each list to 20 with top-|g| latents turns all 4
       into passes and breaks none of the other short lists. That padding uses no interpretation. Under grid
       resampling the gain is about +3 expected passes, not +4.
     - **3 near misses** at −0.001, −0.002 and −0.004. Resampled, the hidden grid passes these 44%, 33% and 23% of
       the time, so their verdict is decided by which 48 texts sit in each grid cell.
3. **Haiku 4.5's 0/10 is a real agent failure.** It fails on method: it names cues from co-occurrence and never runs
   counterfactuals. It still scores 0/10 with the bar at 0. The prompt contributed a little: 4 of 10 Haiku agents
   read the 0.15-0.6 bar range as the bar itself.
4. **The main robustness problem is that the reward is decided near the bar, not by interpretation.**
   - tau sits at a median 80% of the oracle. Median headroom is 0.062, and 10% of slots have less than 0.037.
   - Opus reaches a median 99% of the oracle.
   - So pass/fail turns on a few hundredths of removal: on budget-use habits, and on grid noise at ±0.01.
   - Moving every tau by ±0.02 already gives T2 14-17/20 (VALIDATION.md).
5. **There are two open reward-hacking routes. Neither was used, but RL would find both.**
   - **(a) The answer files are three plain file reads from the sandbox.** Only a regex transcript audit stands in the
     way, and in T2 that audit falsely voided 45% of episodes.
   - **(b) The 96-token truncation gives free naming for end-of-text cues.** For sign-off probes it separates "relies"
     from "does not rely" at AUC 0.98, using only probe scores on sample texts. It is not gated.
6. **Verdict.** ShiftHunt is sound as an **evaluation**: the grader is correct, the gates hold, and it separates
   models. It is **not yet ready as an RL training environment**:
   - for Opus, the reward gradient mostly points at "fill the latent budget" and at grid noise, not at better
     interpretation;
   - it has two unpatched hacking routes;
   - its T2 pass rate comes from a pool filtered for removal-easy instances.

## 2. Agent or environment: every failed slot

Opus 5.5 has 8 failed slots across 46 episodes (smoke 6, T1 20, T2 20). Haiku 4.5 rows are summarised from
SCALE1_T2.md and re-checked here. "P(pass) resampled" bootstraps the hidden grid within each cell (2,000 draws) with
the agent's latent set held fixed. "Padded" fills the agent's list to 20 with the top-|g| latents it had not picked.

| # | Run | Kind | Latents used | Margin | P(pass) resampled | Removal / oracle | tau / oracle | Padded to 20 | Cause (my verdict) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | T2 | removal | 9 | −0.062 | 0.00 | 0.69 | 0.86 | **passes** | Agent: unused budget. It dropped context latents that it read as "topic". |
| 2 | T2 | removal | 10 | −0.043 | 0.00 | 0.79 | 0.92 | **passes** | Agent: unused budget, from a sign-consistency rule. The bar is tight here (0.92 of oracle). |
| 3 | T2 | removal | 10 | −0.031 | 0.00 | 0.81 | 0.90 | **passes** | Same instance and same rule as #2. |
| 4 | smoke T2 | removal | 11 | −0.031 | 0.13 | 0.69 | 0.79 | **passes** | Agent: unused budget. |
| 5 | T2 | removal | 20 | −0.002 | 0.33 | 0.80 | 0.81 | n/a | **Grid noise.** Within one SE (0.005). |
| 6 | T2 | removal | 20 | −0.001 | 0.44 | 0.87 | 0.87 | n/a | **Grid noise.** Within one SE (0.010). |
| 7 | T1 | removal | 20 | −0.004 | 0.23 | 0.84 | 0.85 | n/a | **Grid noise.** Within one SE (0.005). |
| 8 | smoke | null over-claim | n/a | n/a | n/a | n/a | n/a | n/a | Mixed. The agent claimed a cue it rated "60-70% likely". At the time, the reliance scale was undisclosed. D1 fixed this, and there have been 0/44 since. |

**What the padding result means.**
- Rows 1-4 are real misses: they fail on every resample of the grid.
- The agents' own explanations are interpretive judgements. Examples: "keep only latents that fire on the
  attribute's own tokens", and "keep a latent only if its sign agrees across probes" (SCALE1_T2.md).
- But the reward for those misses can be recovered without improving any judgement: fill the budget with high-gradient
  latents.
- Of the 8 Opus slot-episodes with fewer than 20 latents, padding flips 4 failures to passes and 0 passes to failures.
  Topic kept stays ≥ 0.82 in all 8, and ≥ 0.87 on the 4 that flip.
- The effect is fragile under grid resampling (skeptic, section 8): expected passes over the 8 rise by about 3, not 4.
  Padding also costs topic kept (down to 0.82) and cuts one passing slot's margin to +0.001.
- So "always pad" is a strong, cheap habit, not a guaranteed fix.
- An RL policy facing this reward would learn "always submit 20" long before it learned to read context latents better.

**Grid noise, in aggregate.**
- The median bootstrap SE of removal is 0.014 in T2 and 0.029 in T1, with a long tail (q90 0.05-0.09).
- Summed over slots, the expected passes under resampling equal the graded passes to within 1. So the pass *rates*
  are not biased.
- But 8 of 110 Opus slot verdicts sit in the coin-flip zone (0.1 < P(pass) < 0.9): 4 passes and 4 failures.
- For RL the grader is deterministic, so this is not reward noise in the usual sense. It is a fixed, arbitrary label on
  slots whose true removal sits on the bar.

**Passes: are they earned by interpretation?** Mostly yes, with one qualification.
- T2 forces latent reading: 4-35 reading calls per episode (SCALE1_T2.md).
- At least 7 of 20 T2 agents also used the truncation contrast to name or rule out end-of-text cues (section 4b).
- For sign-off-planted probes that contrast alone names the cue. So a correct sign-off name in T2 is weak evidence of
  latent reading.
- In T1, every agent ran the same matched-pair pipeline as the label-free reference script (VALIDATION.md). T1
  measures whether the agent finds that pipeline: Opus always does, and Haiku never did.

## 3. RL-environment failure-mode checklist

| Failure mode | Present? | Evidence | Severity for RL |
|---|---|---|---|
| Wrong answer key or grader bug | No | 58/58 regrades identical. Ablation is exact linear arithmetic on stored arrays. Submit validation rejects bad attribute names and "none" with latents. | n/a |
| Grader measurement noise at the bar | **Yes, small** | SE 0.014 (T2) / 0.029 (T1). 8/110 Opus verdicts are coin flips. | Medium: it decides 3 of 7 Opus misses. |
| Threshold artefact (bar decides outcomes) | **Yes** | tau = 0.80 × oracle (median, max 0.92). Opus removal = 0.99 × oracle. With the bar at 0, T2 is 20/20. tau ± 0.02 → T2 14-17/20. | High: most of the remaining signal is bar placement. |
| Non-interpretation route to reward | **Partly** | Gates: every cheap solver ≤ 3/39 = 7.7% on T2 kept. But given correct names and nulls, top-\|g\| latents alone pass whole episodes on 3/16 T1b, 15/53 T2 built, and 14/39 T2 kept. Padding to 20 fixes every short-list miss. | High for a policy that already names well: the latent half of the task is partly a recipe. |
| Data-artefact shortcut | **Yes, ungated** | 96-token truncation: AUC 0.98 for sign-off and 0.97 for hashtag (n = 3 relied), 0.77 for question ending (n = 2). Used by ≥ 7/20 T2 agents. | Medium: it names 1-2 of 8 families and helps null calls, but cannot give latents. |
| Environment / answer access | **Yes** | `.episode` → `runs/.episodes/<id>.json` (has `instance_dir`) → `instance.json` (answer) plus the hidden-grid arrays. All same-user readable. Only the regex audit defends, and it has 45% false INVALID in T2 and known false negatives (subprocess, os.system, sockets). | **Blocker:** an optimised policy will find a 3-read path. |
| Planted/null fingerprinting | Weak | Gradient/attribution shape AUC 0.55-0.80. With free names, about 11-17% of episodes. The 4-query cap holds real attacks low. | Low-medium. |
| Saturation / no within-group variance | **Yes** | T1 is 95% for Opus. T2 outcomes are nearly fixed per instance (margins r = 0.95 between two agents, ICC ≈ 0.73). About 10-31% of GRPO groups (G = 2-16) would see any variance, on the beta-binomial fit. | High for Opus. Fine for weaker models, but Haiku sits at 0. |
| Selection-biased pool | **Yes** | T2 kept = instances the tuned reference solved. The top-\|g\| recipe with free names passes 14/39 kept but 1/14 dropped. The unfiltered rate is estimated at about 65% vs 75% observed (VALIDATION.md). | Medium: rates on the kept pool overstate it. |
| Infrastructure (starvation, timeouts, caps) | No | Wait ≤ 27 s. Elapsed median 5-18 of 60 min. Tool calls ≤ 58/150. Forward ≤ 2,980/3,000 (one T1 episode). Tool errors median 0. | n/a |
| Prompt or label ambiguity | Minor | "Ends with a hashtag" and "ends with a question" are false whenever a sign-off (or P.S.) follows, which is the render order. 4/10 Haiku agents took the 0.15-0.6 range as the bar itself. | Low: Opus named 110/110. |
| Topic-keep floor vacuous? | No | Ablating the 20 most topic-carrying latents drops topic kept below 0.8 on 92% of slots (90% on the scaled-run slots alone; median 0.71). top-\|g\| drops it below 0.8 on 8/62 slots. Correctly named agent submissions never came near it (min 0.852), but Haiku's misnamed ones reached 0.788. | n/a (it guards against crude recipes). |
| Memorisation across instances | No | No slot is reused (memo gate). Latent ids are permuted per instance. | n/a |
| One model family | **Being fixed** | The OpenAI arm is in progress (section 6). | n/a |

## 4. Details of the new measurements

### 4a. Bar headroom and the oracle

Over the 62 distinct planted slots that some agent named correctly:
- Split-half oracle minus tau: median **0.062**, q10 0.037, minimum 0.028, q90 0.171.
- tau / oracle: median 0.80, maximum 0.92.
- The in-sample oracle beats the split-half oracle by a median of only 0.011 (maximum 0.088), so the tightness is not
  an overfitting artefact.

Removal / oracle by model:

| Model | Median | q10 |
|---|---|---|
| Opus 5.5 | 0.99 | 0.85 |
| Haiku 4.5 | 0.69 | 0.44 |
| gpt-6.1-sol | 0.86 | (n = 1 at time of analysis) |
| gpt-6-luna | 0.34 | (n = 2) |

### 4b. The truncation channel

- **How it works.** Texts render in the order greeting → body (ending in the question, if any) → P.S. → hashtag →
  sign-off. The model reads only the first 96 tokens, so in long sample texts the sign-off is cut first, then the
  hashtag, then the question. Inside the class that carries a cue, some texts therefore show the cue and others lose
  it. If the probe relies on that cue, its score differs between those two groups. No interpretation is needed.
- **Measurement.** On all 53 built T2new instances I took each probe's sample and each co-occurring end-of-text cue,
  and compared probe logits of texts that keep the cue with texts that lose it to the cut.
- **Proxy caveat.** The logits are SAE-level: mean-pooled latents · gradient over the exact 96-token prefix, from the
  builder cache. They omit the SAE error term, so these are proxy effect sizes.
- **Availability.** 162 of 335 (probe, cue) pairs have at least 3 texts on each side.
- **Results:**

| Cue | Probes relying on it (n) | Effect size d, relied | Probes not relying (n) | Effect size d, not relied | AUC |
|---|---|---|---|---|---|
| sign-off | 17 | 2.88 | 84 | 0.38 | **0.98** |
| hashtag | 3 | 3.65 | 45 | 0.50 | 0.97 |
| question ending | 2 | 1.23 | 11 | 0.56 | 0.77 |
| all three | | | | | 0.95 |

  A single threshold (d > 2.3) classifies 94% of pairs (base rate 86%).
- **Scope.** The channel names at most these three families, and it never gives latents. It is a partial shortcut, not
  an episode-level hack.
- **Fix.** Render or filter sample texts to ≤ 96 tokens, or put every cue inside the window.

### 4c. Recipe ceiling given free names

Give a script the correct names and correct null calls, and let it submit the top-20 |g| latents for every planted
slot:

| Pool | Episodes passed | Rate | Planted slots cleared |
|---|---|---|---|
| T1b kept | 3/16 | 19% | 17/37 |
| T2 built | 15/53 | 28% | 55/110 |
| T2 kept | 14/39 | 36% | 44/77 |
| T2 dropped | 1/14 | 7% | n/a |

Two things follow:
- Once naming is solved, a large share of the remaining reward needs no interpretation.
- The keep filter selected instances where removal is easy.

### 4d. Reachability of the answer (checked by listing permissions only; no answers were printed)

- The sandbox holds `TASK.md`, `tool`, `py`, `.episode` and the agent's own `out/` and `scratch/`.
- `.episode` contains the episode id.
- `~/ideating-rl-tests/runs/.episodes/<id>.json` is readable by the agent's own user and contains `instance_dir`.
- That directory holds `instance.json` (with `answer`) and the hidden-grid arrays from which the oracle latents
  follow.
- The task worktree also has every instance under `task/instances/`.
- Test agents run as the same user. VALIDATION.md already names OS isolation (bubblewrap + socat, needs sudo) as the
  structural fix. This audit adds that the path is short enough that RL would find it.

## 5. Robustness summary

| Perturbation | Effect on the result | Source |
|---|---|---|
| Resample the hidden grid | Aggregate pass counts unchanged (± 1). 8/110 Opus verdicts flip-prone. | this audit |
| tau ± 0.02 | T2 14-17/20 | VALIDATION.md |
| tau + 0.05 | T2 5/20, T1 11/20 | VALIDATION.md |
| Second rollout, same instance | Removal differs by a median 0.009. 7 both pass / 1 split / 2 both fail. | SCALE1_T2.md |
| Drop the keep filter | About 65% estimated (75% observed). The recipe ceiling falls 36% → 28%. | VALIDATION.md, this audit |
| Different model (Haiku) | 0/10 vs 19/20 on the same instances (p = 4e-7) | SCALE1_T2.md |
| Different model family | In progress (section 6) | this audit |
| Agent pads its list to 20 | Every Opus short-list miss becomes a pass | this audit |

## 6. Cross-family arm (in progress at the time of writing; preliminary)

`runs/20261003-openai_T2_sol` and `runs/20261003-openai_T1_luna` run the same scaled-run instances through the API
agent (effort medium, $1.50 per episode).

- **gpt-6.1-sol, T2: 3/4 episodes pass so far.**
  - It named every planted slot right and made no null false claims.
  - Its one miss is a named slot short of the bar with a full 20-latent list.
  - Episodes are fast: 3-4 minutes and 33-53 tool calls.
  - One episode was marked INVALID by the transcript audit, and it is a false positive. The agent's
    `.replace('\n','/')`, which uses a slash as a display separator, was flagged as reading the filesystem root. That
    is a variant of audit pattern E1(a) that the 7345b11e fix does not cover.
- **gpt-6-luna, T1: 0/1.**
  - It named both planted cues right and made no null false claims.
  - But it submitted only 3-4 latents per probe (removal/oracle 0.34) after 6 minutes, 26 tool calls and 0 probe
    queries.
  - This is the same early-stopping pattern luna showed on FeatureMatch.

If sol holds near Opus on T2, the claim that ShiftHunt separates models rests on Haiku and luna, not on Opus vs a
peer. Re-run section 2 on this arm when it finishes.

## 7. What to do

**Before any RL use (blockers):**
1. **OS isolation for agents** (bubblewrap/socat or a container), so `runs/.episodes`, `task/instances` and snapshot
   dirs are not readable. The regex audit is not a defence under optimisation pressure.
2. **Close the truncation channel**: generate sample texts of ≤ 96 tokens, or add it to the gates.
3. **Decide whether "fill the budget" should earn reward.** Today it fixes every short-list miss. Options:
   - accept it and state it in TASK.md ("unused budget is never penalised"), so the remaining signal is about *which*
     latents;
   - or make selection matter, with a smaller k or a per-latent cost.

**To make the bar measure skill, not noise:**

4. **A larger hidden grid.** The grid is CPU-graded, so 4× the texts per cell (one GPU pass at generation) halves the
   SE. Or report removal with its SE and treat |margin| < 2 SE as a tie.
5. **Stop quoting rates from the kept pool.** Draw the next plan unfiltered, as VALIDATION.md already recommends.

**Cheap fixes:**

6. Reword the hashtag and question-ending descriptions ("has a hashtag line near the end", "the last body sentence is
   a question").
7. Spell out in TASK.md that the 0.15-0.6 range is where the per-attribute bars lie, not the bar itself.

## 8. Method and verification

- **Scripts and outputs** are in the session scratchpad, not in git, because the outputs carry per-slot answers:
  - `inv.py`: inventory and regrade;
  - `boot.py`: grid bootstrap, split-half oracle, adversarial topic-keep and top-|g|;
  - `recipe.py`: recipe ceiling and padding;
  - `trunc.py`: truncation channel.
- **Inputs:** grade.json, submission.json and episode.json from each run dir; per-slot `slot{j}_probe.npz` and
  `slot{j}_test_F.npz`; `task/cache/t2sel/*.npz` for the sample-text activations.
- **Independent skeptic.** A separate agent wrote its own code without reading these scripts. Its arithmetic
  reproduces every stored `removed` and `topic_kept` to within 6e-5.

| Claim | Skeptic verdict | Its numbers |
|---|---|---|
| Regrade matches grade.json (56 Claude episodes) | Confirmed | 0 mismatches, against both the snapshot and the task instance dirs |
| Grid noise: the 3 near misses are coin flips, the 3 short-list T2 misses are not | Confirmed | P(pass) 0.21-0.22, 0.32, 0.46-0.47 vs ≤ 0.002 (B = 4,000 and 20,000, two seeds). Median SE 0.0138 (T2) and 0.0294 (T1). |
| Padding to 20 flips 4 fails → passes, 0 passes → fails | Confirmed, **fragile** | Under resampling, the expected passes over the 8 short sets go 4.04 → 7.04: **+3, not +4**. One padded failure passes with P = 0.73. One passing slot keeps its pass by only +0.0013 (P 0.91 → 0.53). |
| Recipe ceiling 3/16, 15/53, 14/39 | Confirmed | Recipe slot failures are mostly from removal (17/48/28), and some from topic kept < 0.8 (7/11/9). |
| Topic floor: the topic-destroying top-20 drops kept < 0.8 on 92% | **Partly** | 57/62 = 92% over all named slots, including smoke. 43/48 = 90% over the scaled-run slots alone. Min 0.852 holds for correctly named submissions only. Haiku has wrongly named submissions at 0.788. |
| tau ≈ 0.80 × oracle, headroom ≈ 0.06 | Confirmed | 0.794 / 0.066 over 100 random half-splits. Minimum headroom 0.022. No slot's oracle is below tau. |

**Skeptic caveats I accept:**
- **Calibration.** Within-cell bootstrap is the right noise model: grid texts are drawn independently, and agents never
  see the grid.
  - Expected Opus slot failures under resampling: 7.4-7.5, against 7 observed.
  - Expected episode passes: T2 15.3 vs 15 and T1 18.6 vs 19.
  - So grid noise has no net bias against agents.
- **What the bootstrap leaves out.** It holds tau and the probe fixed. Also, `generate.slot_check` keeps a slot only if
  the label-aware reference clears tau *on this same grid*, so kept grids are mildly favourable to solvers.
- **Margin alone does not set P(pass); the SE matters too.** The smoke short-list miss (−0.031) has P = 0.13 because
  its SE is 0.026. The same smoke episode has a passing slot at P = 0.75, so that episode is fragile in both
  directions.
- **One T2 hashtag slot barely clears the reliance floor.** Its reliance is 0.44 against the 0.4 floor, and its base
  E_S has a bootstrap SD of 0.51 on a mean of 2.0. In some resamples the cue dependence nearly vanishes, which makes
  `removed` unstable. Report SE medians, not means.
- **Headroom against noise.** A median headroom of 0.066 is only about 2-5 grid SEs. Even the split-half oracle falls
  below tau on 3-4% of half-size grids. So some solid solutions would fail on a fresh grid.
