# Resume Highlights: EditHunt and ideating-rl-tests

Raw material for resume bullets. These are not the bullets themselves. Every number below comes from one of the two
final reports. Each entry gives the report section and the underlying source file the report cites.

- **EH** = `/home/user/edit_hunt/PROJECT_REPORT.md`
- **IRT** = `/home/user/ideating-rl-tests/PROJECT_REPORT.md`

---

## Project 1: EditHunt (2026-09-29 to 2026-10-04)

**In one line.** A study of whether an AI agent's interpretability skill can be graded by a program: the agent has to
edit a small open model's internal state so that a two-hop fact (city -> US state -> capital) changes in a controlled
way, and a hidden grader checks the result.

### What Dan built
- **Core library** (`edithunt/`): exact full-sequence log-prob scoring over all 50 capitals (with a safe pruning bound),
  a shared-prefix KV cache, and raw forward hooks for capture, patch, add, subspace swap and gated edits. Also a
  hand-curated dataset of 343 cities across 50 states. (EH §2)
- **Agent RL environment**: the grader runs outside the agent sandbox. The agent gets 12 tools that keep vectors in
  server-side registers and refuse held-out cities and templates. It includes an instance generator with reference
  validation, an agent loop for both the Anthropic SDK and the OpenAI Responses API, per-episode cost caps, and a
  file-locked spend ledger that reserves the worst-case cost before each episode. (EH §2)
- **Six-task interpretability suite** (Phase 7), plus a tiered environment with easy, medium and hard tiers (Phase 6).
  (EH §4.4-4.5)
- **T2 v2, a RAVEL-style "keep the state, change the capital" environment**: a product-reward grader under which a null
  edit scores 0, a gated (conditional) edit hook, and a cross-fitted privileged ceiling oracle. (EH §4.6)
- **RAVEL breadth screen and an 8-tool city environment on gemma-2-2b**: DAS/MDAS training, a Gemma Scope SAE
  baseline, and saturation and learnability tests. (EH §2, §4.7)
- **Audit tooling**: tool-log replays, zero-effort shortcut baselines, re-seeded references, a five-reviewer
  correctness review, an independent audit (`AUDIT.md`) and privileged ceiling oracles. (EH §3)

### Strongest quantified findings
| Finding | Number | Source |
|---|---|---|
| A mean-difference "state" vector at the city token flips held-out two-hop answers in 4 models | peak flip 0.86-0.91; random vectors of the same norm <= 0.05 | EH §4.2, §8; `FINDINGS.md` §1 |
| The edit window ends at a sharp handoff layer | 0.69-0.86 of model depth (1.5B: full-residual paste flips 0.95-0.98 over L3-21, then 0.16 at L22 and 0.00 after) | EH §4.2; `FINDINGS.md` §1 |
| The vector acts as a state variable, not an answer direction | state-question flip 0.95-0.98 vs capital flip 0.86-0.91; projecting out the state direction removes the effect (flip 0.00) | EH §4.2; `FINDINGS.md` §2, §4 |
| Leakage onto other subjects, not flip rate, is what separates naive edits from careful ones (Qwen2.5-1.5B only) | plain gradient leakage 1.00 -> leak-aware gradient 0.07-0.12, with flip 0.83-0.97 | EH §4.3, §8; `FINDINGS.md` §3 |
| Hop separation: change the capital but keep the state (Qwen2.5-1.5B only) | 0.98-1.00 success vs mean-diff <= 0.02 | EH §4.3, §8; `FINDINGS.md` §4 |
| Readout fix: the notebook's "4/16 cities correct" was an artefact of reading the greedy token | exact scoring gives 0.91 few-shot and 0.83 zero-shot correct (n=343) | EH §4.1; `LAB_NOTEBOOK.md` Phase 0 bf16 |
| Hard tier separates careful from naive edits (scripted calibration, 2 seeds pooled; not an agent result) | careful script 5/24 vs every naive script 0/24 | EH §4.4; `FINDINGS.md` §7 |
| T2 v2 has no shortcut | 400 shortcut runs, none >= 0.6 (best 0.51) | EH §4.6; `results/suite/t2_ravel/validate.sc0.jsonl` |
| The gated edit is the existence proof for isolation | gated-oracle Iso_other >= 0.94; tool-path reference passes 10/11 valid instances | EH §4.6; `FINDINGS.md` §11 |
| Frontier agent on T2 v2 (pilot on 10 instances, unequal budgets) | gpt-6.1-sol passes 5/20 (Wilson 11-47%); its mean R 0.53 is 60% of the ceiling; it submitted the conditional (gated) edit in 17/20 episodes without being told. Sonnet 5.5 at a smaller budget: 0/10 | EH §4.6; `FINDINGS.md` §11; `results/suite/t2_ravel/ceiling_summary.json` |
| RAVEL Language task on gemma-2-2b (boosted) | MDAS 0.65-0.76 vs best shortcut 0.22, 4/4 seeds | EH §4.7; `FINDINGS_RAVEL_SCREEN.md` §A |
| An audit caught a tool bug that invalidated one agent's runs | 193 tool errors in 1,545 results; 44 runs excluded | EH §4.5; `LAB_NOTEBOOK.md` |
| Reproducibility | replays 42/42, 31/31, 45/45; 51/51 references (10 within bf16 tolerance); 122/123 regraded runs reproduce exactly | EH §8; `AUDIT.md`, `LAB_NOTEBOOK.md`, `FINDINGS.md` §11 |

### Scale stats
| Stat | Value | Source |
|---|---|---|
| Subject models | 4 (Qwen2.5-1.5B, 3B, 7B; gemma-2-2b), 2 families | EH §3, §8 |
| Agent models | 3 (Claude Sonnet 5.5, gpt-6.1-sol, gpt-6-luna) | EH §3 |
| Dataset | 50 states and capitals, 343 curated cities | EH §8 |
| Agent episodes (examples) | Phase 6 probe 6 (hand-picked instances; a cost and behaviour probe, not a pass rate); Phase 7 suite 7-8 instances per task; T2 v2: sol 20, Sonnet 10; RAVEL: luna 12 | EH §4.4-4.7 |
| Scripted runs | 400 T2 v2 shortcut runs; Phase 6 calibration of 23-24 instances per tier over 2 seeds | EH §4.4, §4.6 |
| Agent tools | 12 (suite environment), 8 (RAVEL city environment) | EH §8 |
| Code | 10,276 lines of Python (7,848 in `edithunt/`, 2,428 in `experiments/`) | EH §8 |
| Result files and figures | 385 result files, 5 figures | EH §2, §8 |
| Commits | 42 | EH §8 |
| API spend | $12.02 summed over three ledgers (Anthropic $3.91, OpenAI suite $5.63, OpenAI screen $2.48); a computed sum that includes worst-case charges and the invalidated sol runs | EH §6, §8 |
| Hardware | Apple M4 (MPS), then one NVIDIA A10G 23 GB (bf16) | EH §3 |

### Methods and skills demonstrated
- Activation steering (mean-difference and gradient-trained steering vectors, norm caps, KL terms)
- Activation patching (full-residual and final-token patching) to locate the handoff layer
- DAS (rank 1 and 4) and MDAS (multi-attribute DAS) subspace training
- SAE latent baselines with Gemma Scope 16k SAEs (5 or 50 latents)
- RAVEL-style attribute isolation, including the replication of inseparable attribute pairs
- Gated (conditional) edits fit on dev data only
- Reward design: product reward, leakage and KL penalties, private held-out grading
- Reward-hacking analysis with matching blocks (brute-force overwrite, "say the target", prompt injection)
- Thresholds fixed before the agent runs (except the RAVEL KL hinge form, chosen after a pilot and disclosed); 95% Wilson CIs on held-out data; multi-seed checks (3-4 seeds)
- Cross-fitted privileged ceilings and script-vs-ceiling ratios as a headroom test
- Agent evaluation harness with multi-provider API loops, cost caps and a spend ledger

### Negative results (stated plainly)
- 5 of 6 Phase 7 suite tasks can be passed by a standard recipe or a zero-effort script. (EH §4.5; `results/task_map.json`)
- The Phase 6 medium tier does not separate methods (careful 5/23 vs mean-diff 6/23). A seed-0 result that suggested
  it did was overturned by seed 1. (EH §4.4)
- Only 11/40 = 27.5% of fresh T2 v2 instances validate (target 60%). The fixed tool-path reference already reaches
  0.76 against a 0.88 ceiling, so headroom is limited. (EH §4.6; `validate_summary.json`)
- RAVEL Continent is not a valid task: lazy DAS reaches 0.50. RAVEL Language is honest, but a fixed script reaches
  about 89% of the ceiling (ratio 0.885), so it is not a stand-alone RL training environment. (EH §4.7)
- The deeper studies used one model each: Qwen2.5-1.5B for the ladder, suite and T2 v2, and gemma-2-2b for RAVEL. (EH §6)

---

## Project 2: ideating-rl-tests (2026-10-01 to 2026-10-05)

**In one line.** An audited set of candidate RL environments in which an agent uses interpretability tools to recover
a planted ground truth inside a small language model, built on a shared evaluation harness that marks broken or
escaped episodes INVALID instead of scoring them.

### What Dan built
- **Ideation pipeline**: 41 candidate environments scored by 2 independent scorers on a 9-criterion rubric (369 cells),
  with 6 selected for building. (IRT §2.1, §4.1)
- **Shared evaluation harness** (`common/`, 7,711 lines excluding tests). It has a Unix-socket broker with exact budget
  caps, a leak scanner, an out-of-process grader with timeouts and output validation, a transcript audit (rules R0-R9),
  a machine-wide GPU memory queue, API test agents for two model families with dollar caps, and a worked example
  task. (IRT §2.2)
- **OS-isolated agent launcher**: bubblewrap namespaces, loopback-only network, and an auth-injecting proxy with a
  fail-closed request-body policy. It comes with about 115 scripted red-team probes. (IRT §2.2)
- **ShiftHunt v2**: the agent must name the surface cue that a linear probe on gemma-2-2b relies on (or answer "none")
  and list at most 20 Gemma Scope SAE latents whose ablation removes that reliance. It has an exact grader with no LLM
  judge, two tiers (open and sample-only), and label-free reference solvers. The setup matches SHIFT-style spurious-cue
  removal, although the report does not use the name SHIFT. (IRT §2.3, §3)
- **Ten other environments** taken to a verdict: ShiftHunt v3, FeatureMatch, LatentKnockout, BoolIntermediates,
  LatentDiff, EditFind, T2Ravel, SilentPathway, ResidualRecall and ModelDiff (a LoRA model-diffing study on
  Qwen3-1.7B). (IRT §2.3)

### Strongest quantified findings
| Finding | Number | Source |
|---|---|---|
| ShiftHunt v2 separates strong from weak models, open tier (T1) | Opus 5.5 19/20 = 95% [76, 99]; gpt-6.1-sol 10/10; Haiku 4.5 0/10; gpt-6-luna 0/10 | IRT §4.3, §8; `docs/results/shifthunt/SCALE1_T1.md`, `SCALE1_OPENAI.md` |
| The separation is significant | Haiku vs Opus on T1, Fisher p = 4e-7; sol vs luna, p = 1e-5 | IRT §4.3 |
| Harder sample-only tier (T2), on a reference-filtered pool | Opus 5.5 15/20 = 75% [53, 89]; gpt-6.1-sol 7/10 = 70% [40, 89] | IRT §4.3; `SCALE1_T2.md`, `SCALE1_OPENAI.md` |
| Cheap solvers fail the gates | T1: 0/16 for every recipe, attack and black-box control (reference 16/16); T2: at most 3/39 = 7.7% on the kept pool (reference 39/39 kept, of 53 built) | IRT §4.3; `docs/results/shifthunt/VALIDATION.md` "Gates" |
| Large models never misname a cue | 144/144 cues named correctly; 0/66 false claims on null probes | IRT §4.3; `VALIDATION.md` "How it fails" |
| The grader is exact and reproducible | 80/80 agent episodes match on regrade | IRT §4.3; `VALIDATION.md` "The grader" |
| Pass rate is consistent across model families (no detected difference at n = 10 vs 20; does not hold probe by probe) | T2 sol vs Opus, Fisher p = 1.0 (difference -5 points, 95% interval -38 to +24); per probe, sol sits closer to the bar (Wilcoxon p = 0.005) | IRT §4.3; `SCALE1_OPENAI.md` |
| The isolated launcher holds | 115/115 red-team probes contained; without the jail, 72/114 escaped | IRT §4.2; `runs/_demo/20261004-073743_claude_iso_selftest/redteam/` |
| An adversarial review broke a harness whose own tests all passed | 55 failures (23 harness, 32 audit), all fixed; 29 of 50 evasion transcripts had passed the old audit | IRT §4.2; `docs/HARNESS_VERIFICATION.md` |
| Audit fixes were checked against recorded transcripts | largest replay 164 units / 5,049 tool calls: exactly 1 verdict change (the intended one), 0 clean episodes flagged | IRT §4.2; `docs/LOG.md` 2026-10-04 |
| Harness self-consistency | first agent smoke 13/24 INVALID because of harness issues; 21/24 valid after the fix | IRT §4.2 |
| Pre-registered gates killed weak designs cheaply | ShiftHunt v3: a no-name script solved 49/51 = 96.1% of slots; killed before any model episode | IRT §4.4; `docs/results/shifthunt3/KILLED.md` |
| A grader validation gate caught a grader that scores wording, not content | ModelDiff paraphrase 0.487 vs verbatim 0.723 (v2), across 794 audited grades; stopped on kill K5 | IRT §4.9; `docs/results/modeldiff/DIFFING_REPORT.md` |
| Research side-finding (scripted sweep, opposite to the prediction; explanations untested) | famous concepts (e.g. Texas, soccer, English) resist knockout by 5 latents in a 16k SAE, while less famous ones do not | IRT §4.6 |

### Scale stats
| Stat | Value | Source |
|---|---|---|
| Candidate environments | 41 scored, 6 picked for building, 11 environments taken to a verdict | IRT §2.3, §4.1 |
| Subject models | gemma-2-2b with Gemma Scope SAEs; Qwen3-1.7B with LoRA organisms | IRT §3 |
| Test agents | 5 (Opus 5.5, Haiku 4.5, Sonnet 5.5, gpt-6.1-sol, gpt-6-luna), 2 families; Sonnet only in early API probes | IRT §3 |
| ShiftHunt agent episodes | 80 graded and regraded across 6 arms (10 fresh instances per arm; Opus 2 episodes per instance); 2 more luna diagnostic arms of 10 (effort high, method hint) | IRT §4.3; `VALIDATION.md`, `LUNA_EPISODES.md`, `SMALLMODEL_FAILURES.md` |
| ShiftHunt pools | T1b 16 instances; T2new 53 built, 39 kept | IRT §2.3 |
| Audit scale | 164 transcript units / 5,049 tool calls in the largest replay; 8,256 recorded episodes in the infra-failure replay (fixer's count; the replay README found 9,644 unique) | IRT §8 |
| Harness code and tests | 7,711 lines; 536 tests passing at the last merge (from 86/87 at first), 3 strict xfail | IRT §8 |
| Pre-registration | LatentKnockout: 38 predictions; ModelDiff: gates G1-G7, kills K1-K5 | IRT §3 |
| API spend (parts; no grand total in the sources) | OpenAI ShiftHunt about $6.5 of a $57 stop; ModelDiff $4.60 OpenAI + $1.36 Anthropic; Anthropic budget hard cap $17 | IRT §4.9, §8 |
| ShiftHunt cost per episode (median) | T2 sol $0.32; T1 sol $0.18; T1 luna $0.029 | IRT §8 |
| Commits | 163 (all branches, shallow clone) | IRT §8 |
| Hardware | 1x NVIDIA L4 (23 GB), 30 GB RAM, 8 vCPU | IRT §8 |

### Methods and skills demonstrated
- SAE latent ablation (exact linear mean-ablation; bit-exact ablation that keeps the SAE error term)
- Linear probes and spurious-cue analysis (SHIFT-style)
- LoRA model diffing and behavioural claim grading
- Pre-registered predictions, gates and kill criteria, with late-written predictions disclosed
- Statistics: Wilson 95% CIs (matching scipy to within 2e-4 for n from 1 to 1,000), Fisher, Wilcoxon, sign, McNemar
  and Newcombe tests, bootstrap SEs for grader noise
- Adversarial audits and independent skeptic passes that re-derive key numbers (for example, FeatureMatch was
  re-implemented with 0/5,017 mismatches)
- Agent evaluation harness design: answer isolation, exact budgets, out-of-process grading, INVALID-not-fail semantics
- Sandbox security: bubblewrap, a network proxy with a fail-closed policy, red-team probing
- Null-instance design: 30-50% null slots, so constant answers stay well under 10%
- Reward-hacking forensics: bar sensitivity, grid noise, truncation leaks, GRPO variance (ICC about 0.73)

### Negative results (stated plainly)
- ShiftHunt v2 is not yet an RL training environment. Two hacking routes are open: answer files reachable in three
  plain file reads (defended only by the regex audit), and a 96-token truncation channel (AUC 0.95) that at least 7/20
  T2 Opus agents used. The reward depends on the bar (T2 Opus 5/20 at tau +0.05), and filling the latent budget earns
  free reward. (IRT §4.3; `docs/forensics/SHIFTHUNT_FORENSICS.md`)
- The T2 pool was filtered by a reference tuned on it. The unfiltered rate (about 65-67%) is a model estimate, not an
  observed rate. (IRT §4.3)
- Nine other designs were stopped or held, each with a measured reason. FeatureMatch mostly measured style (a script
  reached 0.873 of the reference). In LatentKnockout, naive attribution was enough (0.705). BoolIntermediates (12/12)
  and LatentDiff (6/6) were too easy in small Opus smokes (one model, hand-picked instances). (IRT §4.4-4.8)
- ModelDiff's grader failed validation, so all of its uptake and baseline tables are invalid. (IRT §4.9)
- Ideation scores predicted feasibility poorly. The guessed pass rates were far off, even though the two scorers agreed
  closely (8/369 large disagreements). (IRT §4.1)
- Caveats: Claude test agents ran at Tier B containment (honor system plus post-hoc audit), not OS-isolated. OpenAI and
  Claude agents ran at different effort settings. Each arm had n = 10. (IRT §6)

---

## Top 8 most impressive verified facts

1. **ShiftHunt v2 separates strong from weak models in two model families.** On T1, Opus 5.5 scored 19/20 and
   gpt-6.1-sol 10/10, while Haiku 4.5 and gpt-6-luna each scored 0/10 (Haiku vs Opus Fisher p = 4e-7). On the same
   tier every cheap scripted solver scored 0/16. (IRT §4.3; `docs/results/shifthunt/VALIDATION.md`, `SCALE1_T1.md`, `SCALE1_OPENAI.md`)
2. **The ShiftHunt grader is exact and its answers are clean.** 80/80 agent episodes match on regrade. Large models
   named 144/144 cues correctly and made 0/66 false claims on null probes. (IRT §4.3; `VALIDATION.md`)
3. **The OS-isolated agent launcher contained 115/115 red-team probes.** Without the jail, 72/114 escaped.
   (IRT §4.2; `runs/_demo/20261004-073743_claude_iso_selftest/redteam/`)
4. **An adversarial review found 55 failures (23 harness, 32 audit) in a harness whose own tests all passed.** All were
   fixed (191 tests then passed), and the suite later grew to 536 passing tests. The largest replay (5,049 tool calls) changed exactly 1 verdict,
   the intended one. (IRT §4.2; `docs/HARNESS_VERIFICATION.md`, `docs/LOG.md`)
5. **A causally used "state" direction was found in 4 models from 2 families.** Peak held-out flip was 0.86-0.91
   (random <= 0.05), with a sharp handoff at 0.69-0.86 of depth. It acts as a state variable: the state-question flip
   is 0.95-0.98. (EH §4.2; `FINDINGS.md` §1-2)
6. **A leak-aware gradient edit cut leakage from 1.00 to 0.07-0.12 while still flipping 0.83-0.97.** Hop separation
   reached 0.98-1.00 vs mean-diff <= 0.02. Both results are on Qwen2.5-1.5B only. (EH §4.3, §6; `FINDINGS.md` §3-4)
7. **Systematic audits overturned weak designs.** In EditHunt, zero-effort script baselines showed that 5 of 6 suite
   tasks can be passed by a recipe or script. In ideating-rl-tests, 41 candidates led to 1 surviving evaluation. Ten
   other built designs were stopped or held, each with a measured reason, by pre-registered gates, scripted-recipe
   hunts or small agent smokes. Several (ShiftHunt v3, FeatureMatch, ModelDiff) were stopped before any agent episode
   ran. (EH §4.5, `results/task_map.json`; IRT §4.4-4.9, §5)
8. **The T2 v2 environment has no shortcut and a working existence proof.** Of 400 shortcut runs, none reached the 0.6
   bar (best 0.51). The gated oracle edit isolates the change (Iso_other >= 0.94), a tool-only reference passes
   10/11 valid instances, and in a pilot gpt-6.1-sol submitted the conditional edit in 17/20 episodes without being
   told. Only 11/40 fresh instances validate, though. (EH §4.6; `results/suite/t2_ravel/validate.sc0.jsonl`, `validate_summary.json`, `FINDINGS.md` §11)
