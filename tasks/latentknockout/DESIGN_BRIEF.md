# LatentKnockout: design brief (orchestrator, 2026-10-02 ~01:50 UTC)

## Dan's request (summary of the binding parts)
Design, build and smoke-test ONE new task in which the agent must USE an SAE to explain a behaviour, and the grader
RERUNS the model instead of comparing to a label. Motivation: FeatureMatch tested the wrong things (latents chosen on
Wikipedia-style text stay silent on agent-written text, so the score partly measures style imitation; with
top-activating examples it drifts toward reading comprehension).
Idea: a small model (e.g. gemma-2-2b + Gemma Scope SAE) and a specific behaviour (e.g. "The state containing Dallas
has its capital in" -> "Austin"). The agent finds a small set of SAE latents whose ablation (or steering) removes or
changes that behaviour on held-out prompts of the same kind, while leaving unrelated behaviour intact. Reward roughly
(effect on held-out target prompts) x (preservation on held-out control prompts), with a cap on the number of latents.
The details (behaviours, prompt families, metrics, cap, layers, exact reward) are ours to design, and must be explained.
What good looks like: graded by rerunning the model on held-out prompts (no LLM judge); many cheap instances; zero
effort (no latents, random latents, most-active latents, a plain steering vector) scores ~0, so prefer a product
reward; solvable by a reference method that probes (never reads metadata); mid-band 10-70% with a dial, with
difficulty from understanding internals, not arbitrary caps; generic tools (run the model, SAE encode, ablate/edit
latents, top-activating examples), none handing over the answer; optionally honest "cannot be done within the cap"
instances.
How to work: CHECK FEASIBILITY FIRST (does a sensible reference find a small latent set that does the job? if not,
adjust or explain why not); build baselines (reference, trivial guesses, a plain non-SAE steering vector) and COMMIT
PREDICTIONS before seeing results; test with fresh subagents that see only the task statement and tool docs, in a
sandbox (no hosted API keys exist for this; do not call any); read transcripts; real interpretability mistakes vs
environment problems; time-box; drop dead ends with notes; document in plain language; push regularly; no secrets or
weights in git; never print HF_TOKEN.
Deliverable: a short write-up (task design and why, baseline and reference results, predictions vs outcomes, example
transcripts, ways it could be gamed or test the wrong thing, honest verdict).

## Orchestrator's starting hypotheses (to be tested in feasibility; deviate freely with reasons)
- Subject: google/gemma-2-2b + Gemma Scope residual SAEs (cached: layers 6/12/18, width 16k, ~L0 70). Intervention:
  at layer L, x' = x - sum_{i in S} f_i(x) * W_dec[i] at every token position (removes the chosen latents'
  contribution and keeps the SAE error term), i.e. "ablate these latents". Steering is optional, a later variant.
- Behaviour = a relation family with one-token answers that the clean model gets right (validated per entity and per
  template): city -> state capital (EditHunt geography), country -> capital, country -> language, maybe
  person -> sport, plus a non-factual family if feasible (e.g. language identification or past tense).
- Instance = (family, target group, layer). The target group is the behaviour's "concept", e.g. all Texas cities -> Austin.
  The agent sees a few example prompts; the grader uses HELD-OUT target prompts (new entities in the group and new templates)
  and HELD-OUT controls: same family with other groups (specificity), and unrelated text (KL).
- Reward R = Effect x Preserve x (1 - min(1, KL/kappa)), with Effect = fraction (or soft logit-drop) of held-out target
  prompts whose clean answer is no longer top-1, Preserve = fraction of held-out sibling controls whose clean top-1
  survives, KL = mean next-token KL on unrelated text. Pass if R >= tau and |S| <= k. Calibrate tau, kappa, k from data.
- Dials to test: layer (6/12/18), specificity of controls (unrelated-only vs sibling groups vs same-entity other
  relation), target breadth (one entity vs a group), cap k. Precedent: RAVEL/MIB keep/locality readouts, SAEBench
  SCR/TPP-style targeted latent ablation, sparse feature circuits (Marks et al.).
- Null ("cannot be done within the cap") instances: groups or layers where a strong search (attribution ranking +
  greedy + small exhaustive search) cannot reach R >= tau with <= k latents. The grader stays objective: any
  submitted set is rerun and passes if R >= tau. "Cannot" passes only on instances labelled infeasible.
- Key risks: (1) a recipe solves it, e.g. rank latents by attribution (gradient x activation) on the example prompts, or by
  cosine with a mean-difference vector, and take the top k. That recipe IS the reference. Difficulty must then come
  from the specificity and generalisation requirements; measure how often the naive recipe passes. (2) feature
  splitting: city-level latents do not generalise to held-out cities. (3) the SAE error term carries the behaviour, so
  nothing small works. (4) the latent-ID permutation is needed only if public explanation databases matter; there is
  no network in the sandbox, but permute anyway (cheap).
