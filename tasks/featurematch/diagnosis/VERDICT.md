# FeatureMatch diagnosis: verdict

2026-10-02, branch `diag/featurematch`. Full record: `RESULTS.md`. Paths are relative to
`tasks/featurematch/diagnosis/`.

**Verdict label: Not interpretability.** The pre-registered stop rule fired at step 3, so no agent arm ran.

**FeatureMatch is mostly measuring style, not interpretability**: pa(SR-max)/pa(ref) = 0.873 [0.840, 0.903] and
pa(SR-thr)/pa(ref) = 0.730 [0.688, 0.771], over 441 planted slots (`step3/baselines_public.json`).
- **SR** is a script with no reasoning. It runs 6 styled texts per option and claims the option with the highest
  mean activation. SR-thr claims only above a threshold.
- **The choice of texts does not matter.** Across 42 text selections, SR-max ranges 0.862-0.955 and SR-thr
  0.689-0.873.
- **The rule also fires on topic slots alone**: 0.829 and 0.637, n = 328.
- **The fingerprint failure does not explain it.** The filtered pool failed the menu-fingerprint check (P6: 0.616,
  bar 0.60), but that cannot have raised a scripted recipe's accuracy: SR never looks at menu statistics.

## Why

- **The old keys tracked format, not subject.** Only 354/5017 latents (0.071; topics 0.040) kept their concept on
  styled text (`verify_reimpl/result.json`). Dropped topic latents fire on encyclopedia cues such as "(IATA: ...)"
  codes and birth-year parentheses: 18/30, against 0/15 kept (`verify_mechanism/summary.json`). The styled texts
  were on-concept: 800/800 (`verify_banks/results.json`).
- **With valid keys, a fixed recipe solves the menu.** SR-max gets 385/441 and the reference 441/441. A script that
  writes its probes with the task's own `generate` tool, within budget, gets 0.571 [0.525, 0.617]
  (`step3_skeptic/skeptic_summary.json`).
- **The filter did not create this.** On the unfiltered pool the same script got 0.441 (same file), far above the
  0.15 gate.
- **An anti-shortcut rule selected the fragile latents.** All 201 topic concepts have style-robust latents. Of the
  robust latents that match their own concept but were left out of the pool, the generator's "drop latents that fire
  on their own name" filter was involved in excluding 84% (`STEP3.md` section 2.3).
- **The small pool can be memorised.** A policy that reads only option labels and is trained with answer feedback
  passes 0.164 [0.151, 0.178] of fresh episodes. On held-out latents its answer-picking falls to 0.057
  (`verify_fingerprint/policy.json`, `knnsplit.json`).

## What this means for the task

- **As a measure of interpretability, retire the 20-option form.** This is a judgement, not tested: more probe
  texts only make the recipe stronger (0.964 with all 20 texts).
- **It may still serve as a calibration task** (also a judgement). Whole episodes still separate scripts: the
  reference passes 0.989 of episodes and SR-thr 0.378, because every slot, including each "nothing found" call,
  must be right. An agent that scores below this no-reasoning script is failing at mechanics, not at
  interpretation.
- **A redesign would need all of these, each pre-registered:**
  - keys defined on the agent's probe distribution;
  - an answer that is not a pick among probeable names;
  - held-out latents and concepts;
  - few or no language slots;
  - the same scripted-recipe stop rule.

## Transferable lessons for interpretability RL environments

1. **An anti-shortcut filter can select for artefacts.** Banning name-firing latents kept the ones that track
   formatting.
2. **Define the answer key on the distribution the agent will probe with.** An encyclopedia-text key grades
   imitation of encyclopedia text.
3. **Pre-register a scripted-recipe stop rule and hunt cheap solvers before running agents.** It caught this before
   any agent episode was spent.
4. **A small latent pool invites memorisation under RL.** Evaluate on held-out latents and concepts.
5. **Check offline-vs-live equivalence.** Offline SR 385/441 vs live 386/441 (712/725 identical picks) made the
   cheap offline runs trustworthy.
