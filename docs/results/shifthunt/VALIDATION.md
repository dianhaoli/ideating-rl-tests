# ShiftHunt: validation summary (2026-10-03)

One page for Dan. Aggregates only (D6). Sources: SPEC.md, NOTES.md, and the scaled-run reviews SCALE1_T1.md,
SCALE1_T2.md and SCALE1_OPENAI.md, each with its adversarial check (`*_CHECK.md`) and amended with its corrections.
Brackets are Wilson 95% intervals.

## The task, and why it needs interpretability

A *probe* is a linear classifier on a language model's activations (gemma-2-2b, middle layer). Each episode gives the
agent 3-4 probes, each trained to tell two hobby topics apart. Some were trained on data where one of 8 listed surface
cues (sign-off line, greeting, British spelling, question ending, exclamation marks, hashtag, bullet layout, negative
sentiment) went with the label, and learned to use it. About a third rely on no listed cue ("null"). For each probe the
agent must (1) name the listed cue it relies on, or say "none", and (2) give at most 20 *SAE latents* (Gemma Scope
features) whose mean-ablation cuts that reliance and keeps the topic reading.

This is the manual step of SHIFT (Marks et al. 2024): read the features a classifier uses, judge which are irrelevant,
remove them. Data or outputs alone do not solve it:
- In the sample texts, five listed cues and two unlisted "decoy" cues line up perfectly with the topic, for every
  probe, planted or null, so the data cannot say which cue a probe uses.
- Questions to the probe on new texts are capped at 4 per episode; testing 5 candidates on 3-4 probes needs more.
- Every probe also leans on two unlisted cues, so "uses some surface cue" does not separate planted from null.
- The answer must list latents, and only the model's internals give those.

**Tiers.** **T1 "open"**: the agent can run the SAE on any text it writes, so it can build matched pairs (with and
without the cue) and measure each latent's effect. **T2 "sample-only"**: the SAE runs only on the given samples, so the
agent must read what latents mean from top-activating examples and per-token activations: the reading step of SHIFT.

## The grader

The grader uses a hidden balanced test set that no tool can reach: topic x cue, 48 texts per cell. The probe is
linear, so removing latents is exact arithmetic. A planted probe passes only if the right cue is named, at most 20
latents are given, the probe's dependence on the cue drops by at least a bar tau, and at least 80% of its topic
dependence is kept. A null probe passes only on "none" with no latents. An episode passes only if every probe passes,
so the reward is binary. tau is set per cue family before any agent run and then frozen: 0.85 x the 20th percentile of
what a reference that *knows* the cue labels removes with 20 latents, clipped to 0.15-0.6. In-memory regrades match
the stored grades in 80/80 agent episodes (50 Claude, 30 OpenAI).

## Gates: can it be passed without interpretability?

| tier, pool | label-free reference script | each recipe, black-box control and attack |
|---|---|---|
| T1 (16 instances, gates v8) | 16/16 | 0/16 [0, 19%] |
| T2 (gates v9; kept = the 39 of 53 built instances the reference solves) | 39/39 kept (39/53 of built) | <= 3/39 = 7.7% [upper 20%] on kept; <= 3/53 on all built |

Cheap solvers: always "none", claim everything, prior guesses, topic-attribution recipes, gradient or attribution
"fingerprints" for null calls plus top latents, cross-probe sample contrasts, memorisation, and reliance-scale
variants. The black box can pass only all-null episodes. Gate: <= 10% for every cheap solver, >= 95% for the reference.

## Scaled agent runs (fresh Claude Code agent per episode; 10 fresh instances per arm)

| arm | episode pass | planted named right | null false claims | per instance: both pass / split / both fail | predicted |
|---|---|---|---|---|---|
| T1, Opus 5.5 (20) | **19/20 = 95% [76, 99]** | 44/44 | 0/26 | 9 / 1 / 0 | 65% (45-85) |
| T2, Opus 5.5 (20) | **15/20 = 75% [53, 89]** | 52/52 | 0/18 | 7 / 1 / 2 | 55% (30-80) |
| T1, Haiku 4.5 (10, same instances as T1 Opus) | **0/10 = 0% [0, 28]** | 7/22 | 13/13 | one episode each | 30% (0-60) |

- **T1, Opus 5.5: the calibration tier.** Every agent ran matched text pairs through the SAE, the pipeline the
  label-free reference script runs (agent / reference removal: median 1.00). The one miss (0.004 under the bar) was a
  real judgement error. Saturated: with 8 rollouts per group, at most about a third of groups would see reward variance.
- **T2, Opus 5.5.** T2 forces the reading step: 4-35 latent-reading calls per episode, against 21 in all of T1. Naming
  and null calls were perfect. All 5 failures were removal misses on correctly named probes, 0.001-0.062 under the
  bar, on 3 of 10 instances; with the bar at 0, all 20 pass. Within Opus, outcomes are mostly fixed per instance
  (margins of two agents on a probe r = 0.95), so about 25-80% of 8-rollout groups would see variance, likely the low
  end. The harness-valid-only view, 9/10 = 90% [60, 98], hides 4 of the 5 failures behind audit false positives.
- **T1, Haiku 4.5: no method.** 0/10 also with the bar at 0 (against Opus, Fisher p = 4e-7). 9 of 10 named cues from
  sample co-occurrence, which TASK.md says cannot decide; 2 of 10 ever encoded their own texts; none compared a cue's
  effect with the topic effect on TASK.md's scale; 4 of 10 took the 0.15-0.6 range as the bar itself.

## Cross-family check (OpenAI)

The same instances, one episode each, through `common/openai_agent.py`: sandbox cwd, scrubbed env, a command guard, no
repo, session or shared-directory context (all 840 commands stayed in the sandbox). Task prompt, caps, grader and audit
are the Claude arms'. **Not matched:** OpenAI effort was medium, Opus ran at Claude Code effort xhigh, and the system
prompts differ. Cost $5.60.

| tier | OpenAI arm | Claude arm, same instances | paired |
|---|---|---|---|
| T2 | gpt-6.1-sol: **7/10 = 70% [40, 89]**; named 26/26; null false claims 0/9 | Opus 5.5: 15/20 = 75% [53, 89]; 52/52; 0/18 | Fisher p = 1.0; McNemar p = 1.0 against either Opus agent; difference -38 to +24 points |
| T1 | gpt-6.1-sol: **10/10 = 100% [72, 100]**; 22/22; 0/13 | Opus 5.5: 19/20 = 95% [76, 99]; 44/44; 0/26 | sol passes all 10; Opus both pass 9, split 1 |
| T1 | gpt-6-luna: **0/10 = 0% [0, 28]**; 19/22; 5/13 | Haiku 4.5: 0/10 = 0% [0, 28]; 7/22; 13/13 | both fail all 10; with the bar at 0, 6/10 against 0/10 (Fisher p = 0.011) |

- **The difficulty is not Claude-specific.** Same profile in both families: T1 saturated for large models; T2 at 70-75%
  with naming and null calls perfect and every miss a removal miss on a named probe; small models 0/10. In T2 all 8
  failing probe-episodes (3 sol, 5 Opus) are on the 13 of 26 probes where the label-free reference clears the bar by
  under 0.05 (failed probes 5/13 against 0/13, one-sided p = 0.02), and margins correlate r = 0.88 across families.
  Which probes are tight is shared; which tight probe misses depends on latent choice (1 of 5 failed probes shared).
- **In T2 the match is in pass rate, not probe by probe.** sol's margin is below Opus's on 19/26 probes (mean -0.020,
  Wilcoxon p = 0.005); sol-Opus r = 0.86-0.88 against 0.955 Opus-Opus; a 10% higher bar gives 3/10 against 13/20. In
  T1 sol and Opus agree as closely as two Opus agents (removal difference median 0.005, latent-set Jaccard 0.82 both).
- **Context exposure did not visibly inflate the Claude results.** Confined agents reach the Claude rates. In T1 their
  latent sets are as close to Opus's as two Opus agents' sets are to each other, so a copied dump would show. In T2 the
  Opus agents overlap each other more (Jaccard 0.74) than sol (0.54): same-model similarity and effort fit, a shared
  channel is not excluded. Evidence against large inflation, not proof of none; the direct control is Opus confined at
  matched effort.
- **Small models fail in both families, differently.** Haiku has no method. luna finds the counterfactual route and
  names well (19/22 against 7/22, McNemar p = 0.002) but runs it thinly: median 4 latents, removal 0.48 of the
  reference, its own removal estimates 2-3x too high where made, 5/13 null false claims. Diagnostics (partial, SMALLMODEL_FAILURES.md): effort
  high does not change it (1/7 named slots clear, n = 4); a method hint makes removal clear (8/9, 20 latents) but
  0/5 pass, as luna now calls 3/12 planted cues "none". T1 tests procedure discovery more than execution capacity.
- **No environment-caused failure.** One audit false positive (the pass stands) and one runner bug (a submit run from
  Python goes undetected; grades are correct). Regrade 30/30.

## How it fails

- **Large models, both tiers and families: latent selection near the bar.** Agents dropped latents that carry the cue
  through surrounding text, because those looked like topic or another cue, or left part of the 20 allowed latents
  unused. They cited the 80% topic rule, which never bound (lowest kept 0.852). No large model misnamed a cue
  (144/144) or claimed one on a null probe (0/66). Small models fail on method (above).
- **Harness (no grade changed):** audit false positives 6/20 in T1 (fixed), then 9/20 in T2 from 10 new patterns and
  1/30 in the OpenAI arms from one more; audit false negatives for process and network calls via subprocess,
  os.system, os.popen or raw sockets; 4 of 10 Haiku agents broke the "not /tmp" rule.

## Known weaknesses

1. **The T2 pool was picked by a tuned label-free reference.** "Kept" means reference v4.2 solves the instance first
   try, and v4.2 was tuned on these instances (held out: 62% against 74%), which is also how the T2 gate came to pass.
   Agents track v4.2 probe by probe (r = 0.89). A model of the observed agent-minus-reference gaps gives about 77-80%
   on kept, 31-33% on dropped and **about 65-67% on all built** (Opus; an estimate, not an observed rate). sol ran on
   the same pool. Every cheap-solver pass is on kept instances (0/14 on dropped), and "the reference clears every
   failing probe" is true by construction. Independent evidence holds: the best 20 latents by true share, and small
   changes to the agents' own sets, both clear the bar.
2. **Removal-bar sensitivity.** tau is frozen, and on some families sits at 0.9-0.98 of what a label-knowing reference
   removes. tau +-0.02 gives T2 Opus 14-17/20 and sol 5-8/10; +0.05 gives T2 5/20 and 2/10, T1 11/20 and 5/10.
   Understanding decides *which* episodes miss; the bar decides *how many*.
3. **Fingerprint residuals and an ungated channel.** Planted and null probes differ weakly in gradient and attribution
   shape (AUC 0.55-0.80); given correct names free, fingerprint null calls plus top latents would pass about 11-17%,
   and the 4-query cap keeps real attacks low. The model reads only a text's first 96 tokens, so long samples lose
   end-of-text cues: a free within-class contrast. At least 7 of 20 T2 Opus agents used it. No gate tests it.
4. **Test-agent context through the workflow launcher (partly addressed).** Claude test agents are held by
   honor-system rules plus a regex transcript audit, not OS isolation. They see the repo root, commit subjects, the
   account email and a shared scratchpad and tool-results directory where operators had left answer-bearing files. No
   agent used any of it. The confined OpenAI arms reach the same rates and, in T1, the same latent sets. T2 latent
   overlap cannot rule out a shared channel, and thinking was not stored in either family.
5. **Mostly one model family; effort unmatched (partly addressed).** Test agents now span two families, a large and a
   small model each, and the pattern holds. The builders who tuned the reference, the reviewers and the skeptics are
   still all Claude. OpenAI ran at medium effort against Opus at xhigh, with another system prompt, so a family
   difference cannot be told from an effort difference. Each family's model dial has two points.
6. **Small n.** 10 instances per arm. T2's interval is 53-89% (sol 40-89%); the sol-Opus difference is -38 to +24
   points. T1 against T2 (95% vs 75%) gives p = 0.18. The GRPO variance estimate spans 25-80%.
7. **One generic recipe.** In both tiers and both families, large-model agents are close to interchangeable with the
   reference script. RL would mostly reinforce that pipeline plus one judgement: keep latents that carry the cue
   through context.

## Verdict

**Validated with caveats as a test of interpretability skill, and the result replicates in a second model family.
Not yet validated as an RL training environment, for any tier or model.**

| tier, model | verdict |
|---|---|
| T1, Opus 5.5 and gpt-6.1-sol | **Validated as a calibration tier.** It works end to end, cheap routes score 0/16, and passes come from real causal analysis through the SAE. A confined agent from another family runs the same pipeline and picks the same latents (100% against 95%). Saturated for both large models, so not a training tier for either. |
| T2, Opus 5.5 and gpt-6.1-sol | **Validated with caveats** as the tier that forces reading latents: cheap routes score <= 7.7%, and naming went 52/52 and 26/26 through real latent reading. The pass rate replicates across families (75% [53, 89], 70% [40, 89]); per-probe removal does not (sol lower, p = 0.005, at lower effort). **Not validated as a mid-band training tier:** filtered pool (about 65% estimated unfiltered), naming saturated, misses decided by the bar. |
| T1, Haiku 4.5 and gpt-6-luna | **Valid as a model-separation result in both families:** 0/10 each, against 19/20 and 10/10. Haiku fails on method; luna names well but removes too little, at medium effort. **Not a training tier for either:** no episode reward (luna passes 6/10 only with the bar at 0). A graded model dial is not shown: two points per family. |

**Before any RL use:** fix the audit's false positives (45% false INVALID in T2, one more pattern in the OpenAI arms)
and false negatives; clear the shared directories and launch test agents outside the operator's session, as
openai_agent.py does (not a kernel sandbox); add the truncation attack (naming and null calls) to the gates.

**To look for a mid-band tier:** run a middle model (Sonnet-class) on T1 and T2, and draw the next T2 plan without the
reference keep filter; write the predictions first. Repeat the cross-family check at matched effort, with Opus
confined as the exposure control.
