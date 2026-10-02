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
