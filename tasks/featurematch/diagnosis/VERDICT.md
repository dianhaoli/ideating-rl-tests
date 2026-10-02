# FeatureMatch diagnosis: verdict

2026-10-02, branch `diag/featurematch`. Full record and definitions: `RESULTS.md`. Paths are relative to
`tasks/featurematch/diagnosis/`. Brackets are 95% intervals (Wilson or episode-resampled; see `RESULTS.md`).

**Verdict label: Not interpretability.** The pre-registered stop rule fired at step 3, so no agent ran.

> **The task.** FeatureMatch shows an agent an SAE latent (one of 16,384 learned directions in gemma-2-2b's activations
> at one layer) and 20 concept names (topics or languages). The agent runs texts through the model, reads the latent's
> activation and names the concept it fires on, or says "nothing found" if that concept is absent (a *null* slot). An
> episode has 3-5 slots and passes only if all are right. pa(X), *planted accuracy*, is the share of non-null slots X
> gets right. The *answer key* was chosen on encyclopedia-style text; the *reference* script probes with held-out
> dataset texts the agent never sees. *Styled texts* are LLM-written in 10 everyday styles (bank F built the filter,
> bank R is used by SR). v2 is the current pool; v2f is v2 rebuilt from latents that pass the style filter.

**FeatureMatch is mostly measuring style, not interpretability**: pa(SR-max)/pa(ref) = 0.873 [0.840, 0.903] and
pa(SR-thr)/pa(ref) = 0.730 [0.688, 0.771], over 441 planted slots of v2f (`step3/baselines_public.json`). In plain
words: a script that only feeds styled texts through the model and picks the strongest response gets most planted
slots right.
- **SR** has no hypothesis about the latent. It runs 6 bank-R texts per option and claims the option with the highest
  mean activation (SR-max); SR-thr claims only if that option's AUROC against the other options' texts is >= 0.78.
  A live re-run through the tool confirmed it (386/441 vs 385 offline; 712/725 identical picks).
- **Which bank-R texts SR uses moves its score, but never near 0.50.** Over 42 selections (primary included) SR-max
  spans 0.862-0.955 (10th-90th percentile 0.878-0.932) and SR-thr 0.689-0.873 (0.724-0.818): not
  "text-selection-sensitive" (`step3/sr_recipe_out/summary.json`). Text source and count matter more: the same
  max-pick rule gets 0.029-0.571 with other texts, and SR-max 0.964 with all 20 texts (more than the budget allows).
- **Both variants are also above 0.50 on topic slots alone**: 0.829 and 0.637 (n = 328). Weakest cell: topic slots
  of the low-budget tier (T3), SR-thr 0.591 [0.495, 0.680].
- **The pool failed P6**, the menu-fingerprint check (a classifier that sees only the menu; 0.5 = chance): 0.616, bar
  0.60; 0.683 [0.674, 0.693] on a larger test. This cannot have raised a scripted recipe's accuracy: SR's pick uses
  only its own probe activations and learns nothing across episodes.

## Why (findings)

- **The old keys mostly did not survive styled text.** 354 of the generator's 5,017 candidate latents (0.071; topics
  0.040) passed the keep rule; the styled-text test alone passes only 412 (`verify_reimpl/result.json`). The texts were
  on-concept (800/800 sampled bank-F texts, one LLM rater; bank R not audited). Dropped low-firing latents fire on
  0.40 of encyclopedia-style bank texts vs 0.10 of on-concept texts naming an instance (`verify_banks/results.json`).
  An unblinded LLM reader put 18/30 dropped vs 0/15 kept topic latents on encyclopedia format cues, but its categories
  partly use styled-text firing and a scripted check does not reproduce the split (`verify_mechanism/summary.json`).
- **With these keys, fixed scripts pick the right option on most planted slots.** SR-max 385/441, reference 441/441.
  `gen1_cap` (added after pre-registration) writes one probe per option with the task's own `generate` tool, within
  budget: 0.571 [0.525, 0.617]; topic slots 0.540 [0.486, 0.593] (`step3_skeptic/skeptic_summary.json`). Whole
  episodes are harder: SR-max passes 0.094 of them, because it never says "nothing found".
- **The filter made this worse but did not create it.** On unfiltered v2, `gen1_cap` got 0.441 [0.395, 0.488], far
  above the 0.15 gate for non-reasoning recipes but below the 0.50 stop bar; an SR dry run (not pre-registered) got
  0.520 [0.473, 0.567] (`sr_recipe_out_v2_unfiltered_DRYRUN/summary.json`).
- **An anti-shortcut rule excluded most robust latents.** 188/201 topic concepts have a style-robust latent whose best
  dataset concept is that concept; 57/201 have one kept. The rule "drop latents that fire on their own name" was
  involved in 84% of those exclusions, the only reason in 35% (`verify_mechanism/summary.json`).
- **In simulation the small pool is memorisable from labels alone.** A nearest-neighbour learner that sees only option
  labels, trained with every slot's answer over thousands of episodes, picks the right option on 0.676 of planted
  slots for seen latents and 0.057 for held-out ones (chance 0.05). As a policy it passes 0.164 [0.151, 0.178] of
  new episodes with close menus; with far menus 0.032, like always "nothing found" (`verify_fingerprint/knnsplit.json`,
  `policy.json`).

**Limits.** One model, one SAE family (layers 6, 12, 18), one concept set (DBpedia classes plus languages), one menu
format. SR's success is partly built in: the filter kept latents that separate their concept on LLM-written styled
text, and SR probes with such text; a separately written bank R (0 identical texts) and `gen1_cap` (the model's own
completions) limit this. The reference's 1.000 is partly selected by the filter.

## What this means for the task (judgement, untested)

- **As a measure of interpretability, retire the 20-option form.** More probe texts make the recipe stronger.
- **It may still serve as a calibration task.** The reference passes 0.989 of episodes and SR-thr 0.378, but the
  reference uses texts the agent never sees and is partly selected, and SR-thr with 20 texts passes 0.578. Where
  agents would fail needs agent data.
- **A redesign should at least consider** (each pre-registered): keys defined on the probe distribution (necessary,
  but here it made a script succeed); an answer that is not a pick among probeable names; held-out latents and
  concepts; few language slots (SR: 113/113), results by family; plain max-pick recipes in the gate; this stop rule.

## Transferable lessons (hypotheses from this one study)

1. **An anti-shortcut filter can select against the property you want**: banning name-firing latents left mostly
   latents that fail on non-encyclopedic text.
2. **A key defined on one text style can reward matching that style instead of finding the concept.** Defining it on
   the probe distribution is necessary but not sufficient: here it made the menu solvable by a script.
3. **Pre-register a scripted-recipe stop rule and hunt cheap solvers, including plain max-pick ones.** It caught this
   before any agent episode on v2 or v2f (v1 agent runs motivated the study).
4. **A small latent pool can be memorised from labels alone** (in simulation). Hold out latents and concepts.
