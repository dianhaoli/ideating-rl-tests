# FeatureMatch probe forensics: what "it gets some slots right" means

2026-10-02. CPU-only re-analysis of the FeatureMatch LLM-agent probe episodes (2026-10-01) and of both instance pools.
I ran no GPU jobs and changed nothing in the featurematch worktree.

**Where the sources are.** Run dirs are under `~/wt/featurematch/runs/featurematch/` (abbreviated `runs/`). Task files
are under `~/wt/featurematch/tasks/featurematch/` (abbreviated `task/`). The per-slot pool table is `slots.json`, and
the analysis scripts are in `~/.claude/jobs/e4652089/tmp/fmforensics/` (abbreviated `fx/`). Episode dirs and `fx/` are
not in git. Every episode below had an analyst pass and an independent skeptic pass that recomputed the numbers from the
raw activation cache. Where the two disagreed, this report uses the skeptic's corrected value.

**Terms used below:**
- **SAE latent:** one "feature" of a sparse autoencoder trained on the model's internal activations. It is a number that
  is usually 0 and becomes positive on certain text.
- **Slot:** one question in an episode: "which of these 20 concepts does latent #N respond to?"
- **Planted slot:** the true concept is on the menu. **Null slot:** it is not, and the right answer is "nothing found".
- **AUROC:** the chance that a random text of the concept activates the latent more than a random other text. 0.5 is
  chance and 1.0 is perfect separation.
- **Margin:** the true concept's AUROC minus the best wrong menu option's AUROC.
- **Splits A, B and C:** three disjoint sets of concept texts. A defines the answer key. In v2, B is also used to choose
  latents and menus. **C is the only split the generator never touches**, so "held-out" below means C, with B also shown.
- **v1 and v2:** generator versions. v1 is retired (`task/instances_v1/`). v2 is the current pool (`task/instances/`),
  made after the independent audit (`task/NOTES.md`, 23:14 entry).

## 1. Bottom line

1. **The answer keys and the grader are sound.** In all 16 slots of the 4 valid episodes, the key is the best menu
   concept on held-out split C. The smallest planted margin on C is 0.08, and true-concept AUROC on C ranges from 0.90
   to 0.996. Pool-wide, the key flips on C in 1 of 157 v1 and 1 of 431 v2 planted slots. Re-grading every episode
   reproduces every per-slot result.
2. **Every miss is a real agent error, though not always one that would happen on the current pool.** In all 7 misses
   the agent submitted with at least 37% of its compute unused. In 3 of them its own control probes contradicted the
   answer it gave. But in each of the 7, a task-side factor also made the miss easier. It was strong for the two MMA
   slots (style-dependent latents) and the skater slot (misleading label), and weaker for the others: close distractors
   that only v1 allows, a v1-only "word trap" latent, or overlapping labels.
   **All 4 valid episodes used retired v1 instances. No LLM agent has yet run on the v2 pool.**
3. **Sonnet's "4/5" means it identified 3 of 4 planted latents and correctly declined the null.** Across its two
   episodes it named the right concept on 5 of 8 planted slots (Wilson 95% 0.31-0.86). That is well above the fixed
   probe-writing recipes (14-15%) and below the reference solver (96-97%), which probes with split-C texts. Its one miss
   in that episode (curler) came from taking 1 sample per option on a dense latent: 5 per option would have found it
   about 98% of the time.
4. **gpt-6-luna's "3/4" is not skill at naming features.** All 3 correct answers were "nothing found" on null slots,
   which a do-nothing policy also gets. Luna got 0 of 3 planted slots across its 2 valid episodes. Under the current
   product reward (planted accuracy × null accuracy) that episode scores 0.0, not 0.75.
5. **n is tiny:** 4 valid episodes, 16 slots, 2 models, all v1. Treat these runs as bug-finding, not measurement. The
   biggest open validity risk is style dependence. On the existing hand-written-probe check, only 50% of eligible
   latents recover their own concept (`runs/style_check.log`). The fix is a style-robustness filter before the
   measurement batch.

## 2. Every slot of every valid episode

**Valid episodes:**
- **ep2cd00804ab** (Sonnet): `runs/20261001-171048_apiprobe_sonnet/episodes/ep2cd00804ab`, instance fm-t2-08a4773087 (T2, v1).
- **epcd8c604d31** (Sonnet): same run dir, instance fm-t3-025fcaa85e (T3, v1).
- **ep0f904f753d** (luna): `runs/20261001-182038_openai_probe_orch/episodes/ep0f904f753d`, instance fm-t2-18d866dedf (T2, v1).
- **epea26c30f2c** (luna): same run dir, instance fm-t2-0b1b5c8b61 (T2, v1).

Sonnet is `claude-sonnet-5-5`, effort medium. Luna is `openai gpt-6-luna`, effort medium.

**Excluded:** ep0746750a4f and ep4980d0bf46 (gpt-6.1-sol). They are an environment fault: the GPU queue starved them,
5 tool calls in 90 minutes, and they never submitted (`runs/20261001-182038_openai_probe_orch/NOTE_ORCH.md`). Only
ep0746750a4f got a forensic pass, summarised in section 5.

**How to read the table:**
- AUROCs are shown as B / C (held-out). They come from `fx/slots.json` and were recomputed from
  `task/cache/acts_L*_{A,B,C}.npy`; they match the cached tables to within 2.5e-4.
- "Own evidence → truth?" asks whether the agent's own probe results pointed to the correct answer.
- Verdicts are the skeptic's.

| Episode (model, tier) | Slot | Kind (layer) | Truth | Agent answer | OK? | Truth AUROC B / C | Agent-choice AUROC B / C | Margin on C (runner-up) | Own evidence → truth? | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| ep2cd (Sonnet, T2) | 0 | planted (L12) | Band | Band | ✔ | 0.988 / 0.990 | same | 0.338 (BeautyQueen) | yes: 11.2 vs 6.8 | correct |
| ep2cd | 1 | planted (L6) | Curler | nothing found | ✘ | 0.943 / 0.929 | n/a | 0.166 (ChessPlayer 0.764) | no: n=1 per option; its curler text was a 5th-percentile draw (4.39) | under-probed (dense latent, 78% fire rate). Mitigating: v1 distractors, name-probe lure toward ChessPlayer |
| ep2cd | 2 | planted (L6) | BadmintonPlayer | BadmintonPlayer | ✔ | 0.985 / 0.959 | same | 0.206 (Rugby) | yes: ' badminton' token 3.5-3.8 | correct |
| ep2cd | 3 | null near-miss (L18; anchor AmateurBoxer) | nothing found | nothing found | ✔ | best menu: Poker 0.559 / 0.555 | n/a | null: best menu option 0.555 | yes: menu texts 0, boxer 22.2 | correct (genuine null) |
| ep2cd | 4 | planted (L18) | Election | Election | ✔ | 0.900 / 0.906 | same | 0.082 (MMA event 0.824) | yes, but lucky: about a 1 in 6 chance of a 2× gap with n=1 | correct, thin evidence |
| epcd (Sonnet, T3) | 0 | planted (L6) | GaelicGamesPlayer | GaelicGamesPlayer | ✔ | 0.974 / 0.968 | same | 0.254 (Rugby) | yes: 2/2 Gaelic texts fired, about 12 other sports 0 | correct |
| epcd | 1 | planted (L18) | Grape | CultivatedVariety | ✘ | 0.995 / 0.996 | 0.602 / 0.502 | 0.372 (Insect) | mixed, leaning truth: grape/wine 4/4 fired (mean 13.4), non-grape cultivars 2/4 | misread. Mitigating: labels overlap in English; name-probe lure toward cultivar |
| epcd | 2 | planted (L12) | CollegeCoach | Mayor | ✘ | 0.971 / 0.974 | 0.831 / 0.871 | 0.103 (Mayor) | no: the truth option was never probed at L12 (9 of 20 options unprobed) | under-probed. Mitigating: weak v1-only item; broad latent (42% fire rate), 7 distractors v2 would exclude |
| epcd | 3 | planted (L18) | Urdu | Urdu | ✔ | 0.924 / 0.933 | same | 0.296 (Greek) | yes: Urdu 16.45; Hindi and Arabic 0 | correct |
| ep0f (luna, T2) | 0 | planted (L12) | MMA event | nothing found | ✘ | 0.957 / 0.902 | n/a | 0.198 (WrestlingEvent) | no: all 50 probes read 0, including 2 MMA sentences, so uninformative | under-probed. Mitigating: style-dependent latent |
| ep0f | 1 | null near-miss (L6; anchor Fern) | nothing found | Arachnid | ✘ | best menu: Fish 0.653 / 0.629 | 0.421 / 0.529 | null: anchor 0.974 vs best menu 0.629 | mixed: its own word-free spider controls read 4.07 and 0 | misread: word-identity trap ("arachnid" token). v1-only latent |
| ep0f | 2 | planted (L18) | MMA event | HorseRace | ✘ | 0.944 / 0.955 | 0.586 / 0.611 | 0.212 (WTA) | no: its 1 MMA probe read 0; horse race fired 6/29, unstable | under-probed. Mitigating: style-dependent latent |
| epea (luna, T2) | 0 | null (L18; anchor CricketTeam) | nothing found | nothing found | ✔ | best menu 0.519 / 0.508 | n/a | null: all options 0.48-0.52 | yes: all 20 options 0 | correct |
| epea | 1 | null (L12; anchor Amphibian) | nothing found | nothing found | ✔ | best menu Bird 0.639 / 0.619 | n/a | null: anchor 0.962 | yes: it found the off-menu anchor itself (33.0) | correct (best interp in these runs) |
| epea | 2 | null (L12; anchor Cave) | nothing found | nothing found | ✔ | best menu 0.586 / 0.575 | n/a | null: best menu 0.575 | yes: hits did not replicate within a concept | correct |
| epea | 3 | planted (L18) | Skater (all speed skaters) | SoapCharacter | ✘ | 0.923 / 0.908 | 0.484 / 0.532 | 0.376 (SoapCharacter) | no: only skater probe was a figure skater (0); its controls contradicted soap | misread + under-probed. Mitigating: misleading label |

**Per-slot sources:**
- Margins and AUROCs: `fx/slots.json` (fields `true_B`, `true_C`, `dis_C`, `margin_C`, `nullmax_*`).
- Probe values: each episode's `tool_log.jsonl` and `api_transcript.jsonl`.
- Bootstrap probabilities: `fx/boot.py` (ep2cd) and the skeptic re-checks.
- Skater contents: `task/cache/concepts.json`. All 84 Skater texts contain "speed", 73 say "speed skat", and 1 mentions figure skating.
- CultivatedVariety contents: same file. 0 of 84 texts mention grapes and 25 are bromeliads.

**Recorded scores vs the current grader.** grade.json in every probe episode was written by the old grader, which used
the mean of slot correctness (git 81f9bb3). Current `task/grader.py` scores planted_acc × null_acc. Re-grading in memory
(`fx/regrade.json`, `fx/gtest.py`):

| Episode | grade.json (old mean) | Current grader | pass |
|---|---|---|---|
| ep2cd | 0.8 | 0.75 | false |
| epcd | 0.5 | 0.5 | false |
| ep0f | 0.0 | 0.0 | false |
| epea | 0.75 | **0.0** | false |

## 3. Patterns

There are 7 misses in 16 slots, and one miss can show more than one failure mode. The four "Mitigating" groups below
are task-side factors, not agent failure modes.

| Failure mode | Count | Slots |
|---|---|---|
| Under-probing / early submission | 5 of 7 | curler (n=1 per option on a dense latent); coach (true option never probed); MMA ×2 (2 and 1 MMA probes); skater (1 probe, wrong subtype) |
| Ignored its own controls | 3 of 7 | grape (4/4 vs 2/4); arachnid (word-free spider texts low); soap (Ian Beale 0, non-soap Lenny Bruce 48.9) |
| Judged a live latent dead ("nothing found" on a planted slot) | 2 of 6 planted misses | curler, MMA s0 |
| Token or word confound read as a concept | 2 | arachnid (word "arachnid"), soap ("Len" token) |
| Sibling or nested-label confusion | 2 | grape → cultivated variety; coach → mayor (true runner-up) |
| Null over-claim | 1 of 5 null slots | fern latent → arachnid |
| Mitigating: style mismatch (agent text does not fire the latent) | 2 clear | MMA ×2. Curler is possible but untested. |
| Mitigating: misleading or broad label | 2 | skater (= speed skater), cultivated variety (= mostly ornamentals) |
| Mitigating: v1-only weakness | 3 | curler (distractor 0.73-0.76), coach (margin 0.10), fern (latent fires on its own name) |

**Early submission happened in every valid episode.** Forward units used (from `harness.counters` in grade.json):
- ep2cd: 748 of 1200 (62%).
- epcd: 103 of 550 (19%).
- ep0f: 147 of 1200 (12%).
- epea: 96 of 1200 (8%).

No agent ever called `top_latents` or `generate`. That is the most common thread: the agents stop at about one probe
per option.

**Per model** (n is tiny; these are not rates you can quote):

| | Sonnet (2 episodes) | gpt-6-luna (2 valid episodes) |
|---|---|---|
| Planted slots right | 5 / 8 (Wilson 0.31-0.86) | 0 / 3 (Wilson 0.00-0.56) |
| Null slots right | 1 / 1 | 3 / 4 |
| Episodes passed | 0 / 2 | 0 / 2 |
| Typical miss | under-probing a broad or dense latent; nested label | style mismatch; word or token confounds |

**Fixed baselines on the same v1 pool** (`fx/recipes.py` over `runs/20261001-192351_audit_inproc` and
`runs/20261001-184458_inproc`):

| Baseline | Planted accuracy | Note |
|---|---|---|
| self_probe | 0.14 | says "nothing found" on 73% of planted slots |
| template_probe | 0.11 | says "nothing found" on 78% of planted slots |
| black-box | 0.00 | |
| reference | 0.96 | probes with split-C texts |

Sonnet's 5/8 sits between the recipes and the reference. That is the "promising" signal, but it is 8 slots.

Luna's pattern (accurate "nothing found", no planted hits) looks the same as the self_probe recipe.

## 4. Answer-key soundness across the pool

Source: `fx/slots.json`, built by `fx/keys.py` and `fx/analyze.py`. I re-ran the summaries and they reproduce. The
answer key equals the anchor concept and the split-A best concept in 157/157 v1 and 431/431 v2 planted slots. There are
no flips on A or B, by construction.

**Planted slots, scored on split C:**

| | v1 (157 planted) | v2 (431 planted) |
|---|---|---|
| True concept C AUROC, median / min | 0.953 / 0.788 | 0.940 / 0.770 |
| Best distractor C AUROC, median / max | 0.708 / 0.915 | 0.611 / 0.812 |
| Margin on C, median | 0.229 | **0.329** |
| Key flips on C | 1 (fm-t2-ce893caa60 s2, mountain pass vs mountain, −0.001) | 1 (fm-t2-e4fe23ceec s1, insect 0.787 vs fungus 0.808) |
| Margin < 0.10 / < 0.05 | 21 / 5 | **7 / 1** |
| True AUROC < 0.85 | 8 | 33 |
| Expected flips if C is resampled (sum of bootstrap flip probabilities) | 3.6 | 2.2 |

**Null slots, scored on split C** (best menu option; a planted slot needs 0.90 on A):

| | v1 near-miss (76) | v1 off-universe (50) | v2 near-miss (272) |
|---|---|---|---|
| Median | 0.609 | 0.542 | 0.603 |
| Max | 0.769 | 0.695 | 0.808 |
| ≥ 0.72 | 4 | 0 | 9 |
| ≥ 0.80 | 0 | 0 | 3 |

**Shaky v2 slots** (recommend drop or regenerate):
- fm-t2-e4fe23ceec s1: the planted flip above. The reference solver missed it 5/5 (`fx/recipes.py`).
- fm-t2-80d94d3488 s0 and fm-t3-ec265e59bb s3: nulls on the same L6/8390 insect latent, with Fungus at 0.808 on C.
- fm-t1-7ff04bcb8e s2: null on an L6/5683 bird latent, with Musical at 0.807 on C.

**Did v2 fix v1's problems?** Mostly:
- **Close distractors:** v2 margins are wider (median 0.33 vs 0.23) and there are far fewer slots under 0.10 (7 vs 21).
  The v1 items behind the coach and curler misses could not occur in v2.
- **Own-name filter:** 0 of 703 v2 slot latents fire on their own concept's name probes, against 59 of 283 in v1. That
  removes the fern-type word trap, and all three of epea's null latents also fired on their own anchor name: 17.6, 38.6
  and 12.5 (`own_name_max` in slots.json). So luna's 3 "correct" nulls were on latents v2 would not use.
- **What v2 does not fix:**
  - Style dependence: v2 keeps only latents that respond to context, which may make it worse (`task/NOTES.md`, 23:14).
  - Misleading labels.
  - Heavier latent reuse: 495 unique latents across 703 v2 slots, one used 8 times (L6/14843), against 261 across 283 in v1.

**Split integrity** (`fx/splits.py`):
- `concepts.json` hashes match the manifests.
- No exact duplicate texts.
- 1,121 near-duplicate pairs cross splits inside 33 concepts. Most come from formulaic stubs: solar eclipse 686 and
  green alga 220. For those concepts, agreement on C partly measures the shared template, not the topic.

**Caveat on "independent C".** The reference solver probes with C texts, and v1 instances were kept only if the
reference solved them (finalize_pool). So agreement on C is partly selected for. It does not change any verdict here,
because A, B and C agree and the margins are large.

## 5. Unsound things found, severity, and fix

| # | Issue | Severity | Evidence | Fix |
|---|---|---|---|---|
| 1 | **Style dependence.** Latents chosen on DBpedia abstracts often stay silent on agent-written text. | **High** (main validity risk) | 153/307 eligible latents (50%) recover their concept on hand-written probes; airport only 11% (`runs/style_check.log`, `task/NOTES.md` 16:42). Both MMA latents read 0 on agent sentences but fire on 54/54 dataset texts containing "mixed martial arts event". The self_probe recipe says "nothing found" on 73-79% of planted slots. | Style-robustness filter: one GPU pass over about 20 differently styled texts per concept (about 4,400 texts, about 5 min on an L4). Keep a planted slot only if its latent still separates its concept on that set. |
| 2 | **Misleading labels.** The label text differs from what the dataset class contains. | Medium | Skater is 84/84 speed skaters, and a figure-skater probe reads 0. CultivatedVariety has 0/84 grape texts and 25/84 bromeliads. The MMA class includes kickboxing and bodybuilding. | Generate labels from class contents ("speed skater", "ornamental cultivar"), or drop overlapping pairs. Have TASK.md say the options are disjoint dataset classes. |
| 3 | **Stale scores in recorded grade.json.** | Medium (wrong aggregates) | epea shows 0.75 but the current grader gives 0.0; ep2cd shows 0.8 vs 0.75. | Re-grade all recorded episodes with the current grader before aggregating. Write the grader's git sha into grade.json. |
| 4 | **Starved runs recorded as valid.** | Medium | ep0746750a4f: wait 5127 of 5393 s, 3 of 36 charged forward units delivered, never submitted, yet `valid=true`. | Mark an episode invalid when submitted=false and queue wait is above 50% of elapsed time. |
| 5 | **Charged but never delivered; retries charge twice.** | Medium | ep0746750a4f: 33 forward units charged after the 900 s client timeout. epea: 16 units. | Add an idempotency key and return the cached result on retry, or refund on client disconnect. |
| 6 | **Two clocks disagree.** | Medium | Queue wait does not count against the 60-minute episode clock, but it does count against the runner's 90-minute limit, so an agent can be killed after about 2 minutes of its own time. | Have the runner use the episode clock, or pause it during queue waits. |
| 7 | Shaky v2 slots. | Low-medium | 4 slots listed in section 4. | Drop or regenerate them. |
| 8 | `tool_log.jsonl` cuts responses at about 4 KB. | Low (forensics only) | epcd log entries 9-10 lose the decisive Urdu value; ep0f call 11. | Store full responses, or write them to a separate file. |
| 9 | TASK.md does not name the response keys (`max`, `acts`). | Low | ep2cd lost 80 forward units to a silent parse bug. | Document the response schema. |
| 10 | `./tool` takes JSON only as a shell argument. | Low | Apostrophes broke quoting: 4 turns in ep0f, 1 call in sol. | Accept `--file` or stdin. |
| 11 | Near-duplicate stub texts across splits. | Low | Solar eclipse 686 pairs, green alga 220. | Deduplicate, or split by template. |

**Checked and found sound:**
- **Grader:** 3,573 synthetic submissions over both pools all match expectations (`fx/gtest.py`). Re-grading all 14
  recorded probe and audit episodes with the v1 keys gives identical per-slot results.
- **Tools:** the latent-index permutation maps every agent index to the key's real latent (`tools.perm_for`). Tool
  activations reproduce the cache on 60 identical (text, latent) pairs, with no fire/no-fire disagreements and a
  maximum difference of 0.27 (`fx/ep0f_toolcache.py`).
- **Leakage:** no leakage or audit violations. ep2cd's audit flag was a `sed` false positive, fixed in 296e05cd.

## 6. What this implies for the measurement batch

**Before the batch:**
- Land fix 1 (the style filter) and fix 3 (re-grading); fixes 4-6 if you can.
- Run only on v2 instances. All evidence so far is v1, and v1 has close distractors and word traps that v2 removed.

**Metrics to watch, per model and tier:**
- **Pass rate** with a Wilson CI. Pass needs every slot right.
- **Planted accuracy and null false-claim rate, reported separately.** A model that always says "nothing found" can
  look good on the mean-slot score; luna's 3/4 is the example.
- **"Nothing found" rate on planted slots.** This is the style-dependence alarm. Compare it with self_probe's 0.79 and
  the reference's 0.02. If agents sit near the recipe, the task is measuring probe style, not interpretability.
- **Forward budget used at submission.** All 4 agents stopped early. Pass rate may depend on persistence more than on
  skill; consider a "use at least X% before submit" analysis, or a higher-effort arm.
- **Per-slot latent density** (fire rate). Misses clustered on broad or dense latents: 78% and 42% fire rates. Report
  accuracy split by density.
- **Episode validity:** queue wait divided by elapsed time, and forward units delivered vs charged.

**Claims we can make now:**
- The answer keys and grader are correct.
- Every observed miss was avoidable with the remaining budget.
- Sonnet named the right concept on more planted slots than the fixed recipes did: 5/8 vs about 14%, in 8 slots.

**Claims we cannot make yet:**
- Any pass rate or model ranking. n = 2 episodes per model, Wilson intervals overlap, and nothing has run on v2.
- That FeatureMatch sits in the "mid-band difficulty" band on v2.
- That failures measure interpretability skill rather than probe-writing style. That needs fix 1 and the planted
  "nothing found" rate from a real batch.
- That luna can identify features: 0/3 planted, and its null successes were on latents v2 would not use.
