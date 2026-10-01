# FeatureMatch: SPEC (one page)

**Task.** The agent gets gemma-2-2b plus Gemma Scope residual SAEs (layers 6/12/18, 16,384 latents each). An episode has
3-5 **slots**. Each slot names one SAE latent (layer + index) and 20 candidate concepts. For each slot the agent answers
the concept the latent encodes, or `"nothing found"`. Concepts come in two kinds: "article about a <DBPedia class>"
(213 Wikipedia-abstract classes such as swimmer, volcano, US Supreme Court case) and "text written in <language>"
(19 languages). Automated version of the human practice of explaining SAE latents by testing them on probe text
(SAEBench, Karvonen et al. 2025; auto-interp detection scoring, Paulo et al. 2024 / Bills et al. 2023).

**Planted ground truth.** Per slot, the generator computes each option's **held-out AUROC**. This is the probability that
the latent's max activation on a held-out text of that concept beats its activation on a held-out text of any other of
the 232 concepts. It uses split A (40 texts per concept), which no tool can read. A *planted* slot's menu contains the
latent's best concept c\* (AUROC >= 0.90, fires on >= 50% of c\*'s texts). Every other option is >= 0.10 below
c\*, and the same verdict holds on an independent split B (AUROC_B >= 0.85, margin >= 0.08). A *null* slot has every
option < 0.65 on A and < 0.72 on B. Nulls come in two kinds. In a **near-miss** null (60%), the latent selects a concept
that is left out of a menu built from that concept's taxonomy neighbourhood (for example a Greek latent offered
Bulgarian, Turkish and so on). In an **off-universe** null (40%), the latent is live but not selective for any concept.
The 0.65-0.90 gap keeps both answers unambiguous.

**Grader.** Exact option match on planted slots and an explicit "nothing found" on null slots. Claiming anything on a
null slot is a scored failure (`null_fp`). score = mean slot correctness. **pass = all slots correct** (product rule,
D7). Details use the convention `slots: [{planted, correct, claimed}]`.

**Tools** (generic primitives; latent indices are secretly **permuted per instance**, so Neuronpedia-style lookups fail):
`task_info` (public slot list; one tool call, no forward units), `latent_activations(texts<=16, layer, latents<=8)` (per-token and max activations),
`top_latents(texts<=8, layer, k)`, `vocab_projection(layer, latent)` (decoder direction times unembedding),
`generate`, `next_token_logits`. Profiles: `full` = all tools; `blackbox` = task_info + generate + next_token_logits.
No tool takes or returns a concept name, the menu-to-truth map or held-out text.
**Caps:** tool_calls 150, forward 1200 (T3: 550), generate 40, gradient 0 (no in-episode training); harness defaults
wall clock 60 min and 180 s per call.

**Dials.** (1) *Distractor closeness* (**EXPERIMENTAL**, no direct precedent found). T1 "far": 19 distractors are drawn
uniformly from the universe. T2 "close": distractors are taxonomy siblings, ranked by closeness to a random member of the
answer's sibling group, with noise. (2) *Number of slots* 3-5 (precedent: number of simultaneous targets, MEMIT scaling
curves). (3) *Forward budget* (precedent: Pando's query budget): T1/T2 1200 forward units, **T3 = close + 550**.
(4) Null fraction (each slot is null with p = 0.4, independently, so the null pattern is random).

**Null design / no fingerprinting.** Planted and null slots use the same menu procedure around an anchor concept,
which is drawn the same way for both. public.json has identical keys and shapes, and tools behave identically.
Measured: a gradient-boosted classifier on public menu structure predicts null vs planted at CV AUROC 0.50 (T1) and
0.58-0.60 (T2). The residual T2 signal is a known weakness (see NOTES).

**Containment:** honor-system with auditing (Tier B).

**Shortcut analysis.**
- *Black box*: without SAE access the latent index carries no information. The control asks the model itself and is
  ~0%.
- *Constant answers*: "nothing found" passes only all-null episodes (~3%). "Always claim" needs no nulls and every
  answer right (~0%).
- *Popular-concept prior*: anchors are drawn uniformly over concepts, not latents. Solar-eclipse articles alone own
  >2000 selective latents, so drawing latents uniformly would have made this prior work.
- *Token identity* (warning 3): a planted slot is rejected whenever "encode each option name, pick the max" would
  pick the answer under any of 3 surface variants. On kept slots that recipe is at chance.
- *Logit-lens string match* (`vocab_match` recipe): measured in the gates.
- *Lookup*: the per-instance permutation defeats public explanation databases.
- *Grading-data access*: no tool reads splits A or B.
Remaining honest weaknesses: (0) **harness codename check**: the leak scan treats the codename "featurematch" as a
never-exempt private string after removing whitespace, so an agent note like "this feature matches Spanish" in its own
scratch file makes the episode INVALID. Fix belongs in common/ (reported). (1) **style dependence**: on 4 hand-written probes per concept, only ~50% of selected
latents (median per concept 70%; airports 11%) prefer their own concept, so some latents key on dataset-specific
text style. The prompt therefore defines each concept's text style, and a style-robustness filter is the recommended
next step. (2) The residual menu fingerprint in T2 (CV AUROC ~0.6). (3) The concept universe is public: an agent could
guess that the labels come from DBPedia.

**Transfer.** Explaining and verifying SAE latents is a daily task in interpretability work: auditing feature labels,
choosing features for steering or circuit analysis, and SAEBench-style evaluation. The skill is to design probe texts,
read activations and decide calibrated "none of these" verdicts. The same skill is what an auditor needs to check a
claimed feature explanation (compare the seed candidate "ClaimCheck").

**Gates** (stage 2, 2026-10-01; fresh pool of 72 instances, 24 per tier, seeds 5000+/105000+/205000+; nothing
dropped, so "kept" = all 72). Two sources, never pooled:
*harness* = `common.sandbox run-scripted` (broker, leak scan, out-of-process grader), run dirs
`runs/featurematch/20261001-172659_gate_{reference,blackbox,recipes}`;
*in-process* = `inproc_gates.py`, the same solvers, Env, caps and out-of-process grader as one queued GPU job,
run dir `runs/featurematch/20261001-184458_inproc`. The in-process complement exists because the shared GPU queue made
each harness GPU episode wait 15-40 min (NOTES). On the same (solver, instance), the two sources agree slot for slot
on every deterministic solver (291/291 episodes compared). Aggregates: `runs/featurematch/20261001-172659_gates_public.json`.

| tier | ref one-shot | ref best-of-5 | black box | worst recipe |
|---|---|---|---|---|
| T1 far / 1200 | 24/24 (100%) | 24/24 | 1/24 (4.2%) | 1/24 (nothing, name_probe_thr, vocab_match) |
| T2 close / 1200 | 23/24 (95.8%) | 24/24 | 0/24 | 0/24 |
| T3 close / 550 | 21/24 (87.5%) | 24/24 | 1/24 (4.2%) | 1/24 (nothing, vocab_match) |
| all | 68/72 (94.4%, CI 0.87-0.98) | 72/72 (CI 0.95-1.0) | 2/72 (2.8%, CI 0.008-0.096) | 2/72 (2.8%, CI 0.008-0.096) |

Reference and black box and the three model-using recipes (name_probe, name_probe_thr, vocab_match) are in-process
numbers; the five model-free recipes (nothing, always_claim, prior, prior_or_none, random) ran through the harness on all
72 (their in-process numbers are identical, random aside, which uses different seeds). Harness GPU episodes completed so
far: reference 3/3 pass, black box 1 (score 0.67, fail). Every non-reference pass is an all-null episode answered
"nothing found" everywhere. Planted-slot accuracy of the name-probe recipe is 0.051 (chance = 0.05); of vocab_match
0.013; of the popular-concept prior 0.13. The reference never claims on a null slot (null FP 0/all tiers).
**All gates PASS in every tier.** Stage-1 preliminary numbers (in-process, 175 kept prelim instances) agree; they are in NOTES.

**Smoke plan** (`smoke_plan.json`): 3 instances per tier that the reference passed on its first try, excluding the 4
instances already used by orchestrator LLM probes; together they hold >= 1 null and >= 1 planted slot per tier.
