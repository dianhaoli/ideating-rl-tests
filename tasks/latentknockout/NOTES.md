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
