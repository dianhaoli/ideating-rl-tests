# LatentKnockout lab notebook

## 2026-10-02T01:25Z Feasibility study starts
Goal (plain language): gemma-2-2b answers simple questions ("Dallas is a city in the state of" -> " Texas").
A sparse autoencoder (SAE) rewrites the model's internal state at one layer as a sum of ~16k "latents" (features),
only ~70 of which are non-zero on any token. The question for this study: can we find a SMALL set of latents such that
switching them off (subtracting their contribution, keeping everything the SAE cannot explain = the "error term")
breaks the answer for the target group (e.g. every Texas city, including cities and phrasings never used to pick the
latents) while leaving other groups (California cities, etc.) and ordinary text alone?
Setup notes:
- Disk is 99% full (1.9 GB free), so I cannot make my own bf16 copy of gemma-2-2b (the hub copy is fp32, 9.8 GB, and
  its memory map exceeds the 8 GB per-job RAM cap). I load the bf16 copy that the FeatureMatch builder already made at
  ~/wt/featurematch/tasks/featurematch/cache/gemma-2-2b-bf16 READ-ONLY (no edits there), with the hub as fallback.

## 2026-10-02T06:17Z Feasibility predictions committed before any result
I wrote my guesses down before running anything, so the feasibility study can be checked against them
(`PREDICTIONS_FEASIBILITY.md`; frozen once committed). The validation job from 01:28Z never got onto the GPU: its log has
only the queue-wait line, and there are no output files. So no model output had been seen. Headline guesses:
- About 70% of prompts are answered correctly by the clean model. The two-hop city -> state capital family is weakest (~42%).
- With at most 5 latents, a sensible search reaches a reward R >= 0.5 on ~37% of (family, group, layer) cells, and for
  ~60% of (family, group) pairs at some layer. Layer 6 is mostly hopeless (the model recomputes the fact after the ablation); layers 12 and 18 work best.
- The real risk is a recipe. Naive "top-5 latents by attribution" is predicted to reach >= 50% of the reference's
  reward on ~55% of feasible cells, which fires the NO-GO-as-is rule (probability ~0.4). Random, no-latent and
  most-active latents are predicted to score ~0. A plain steering vector (not submittable) is about as good as the SAE reference.
- Style-varied held-out prompts are predicted to lower R by about 20% (relative). The SAE error term is not predicted to carry most of the behaviour.
- The decision rule (GO / ADJUST / NO-GO, with numbers) is in section 6 of that file. No job was run for this step (no RSS to report).

## 2026-10-02T06:30Z Precedent survey written (PRECEDENT.md), no GPU, no model run
What I did: I read the published work on switching off or steering with SAE latents, compared with simple baselines,
and how such tasks are scored, and I wrote what each finding predicts for this task (`PRECEDENT.md`, with URLs; preprint numbers are tagged).
Main lessons in plain words:
- In almost every head-to-head test, SAE latents lose to a simple "difference of means" direction or to a supervised
  method: AxBench steering 0.239 (difference of means) vs 0.165 (SAE); RAVEL disentangle 48.6 (SAE) vs 60.1 (MDAS); in MIB,
  SAE features are no better than plain neurons. So restricting the agent to SAE latents is a handicap. The task must say
  this, and must measure it with a dense comparator that cannot be submitted.
- The usual way to choose latents (rank by attribution or probe weight, take the top 20: SAEBench SCR/TPP, Marks et al.)
  is exactly the recipe this task must not reward. It becomes a gate.
- Simply zeroing a few latents often barely changes the model; unlearning work needed negative clamping (Farrell et al.).
  Feasibility should test both.
- Absorption (a "Texas" latent that stays silent on some Texas cities) shows up in every SAE tested. Held-out-entity
  grading measures exactly this, and that is a real interpretability skill.
- Agents using SAEs do worst on the causal step (SAEScientist-Bench: 31 vs 58 for experts) and fall for formatting or
  substring latents. Mid-band difficulty is plausible if the recipe gate holds.
The 8 design implications are in section 5 of PRECEDENT.md. Section 4 compares the literature with the frozen feasibility predictions; the riskiest one
is "the SAE error term is not dominant". CPU only; peak RSS is negligible (web reading, no heavy job).
