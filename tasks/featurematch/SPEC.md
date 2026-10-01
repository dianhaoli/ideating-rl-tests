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
`task_info` (public slot list), `latent_activations(texts<=16, layer, latents<=8)` (per-token and max activations),
`top_latents(texts<=8, layer, k)`, `vocab_projection(layer, latent)` (decoder direction times unembedding),
`generate`, `next_token_logits`. Profiles: `full` = all tools; `blackbox` = task_info + generate + next_token_logits.
No tool takes or returns a concept name, the menu-to-truth map or held-out text.
**Caps:** tool_calls 150, forward 1200, generate 40, gradient 0 (no in-episode training).

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
Remaining honest weaknesses: (1) **style dependence**: on 4 hand-written probes per concept, only ~50% of selected
latents (median per concept 70%; airports 11%) prefer their own concept, so some latents key on dataset-specific
text style. The prompt therefore defines each concept's text style, and a style-robustness filter is the recommended
next step. (2) The residual menu fingerprint in T2 (CV AUROC ~0.6). (3) The concept universe is public: an agent could
guess that the labels come from DBPedia.

**Transfer.** Explaining and verifying SAE latents is a daily task in interpretability work: auditing feature labels,
choosing features for steering or circuit analysis, and SAEBench-style evaluation. The skill is to design probe texts,
read activations and decide calibrated "none of these" verdicts. The same skill is what an auditor needs to check a
claimed feature explanation (compare the seed candidate "ClaimCheck").

**Gates** (preliminary, IN-PROCESS through the real Env and caps; the harness is not READY). Run dirs
`runs/featurematch/20261001-153149_prelim_v2` and `runs/featurematch/20261001-160933_prelim_v2b` (kept_summary.json).
180 generated, 5 dropped (reference best-of-5 failed), 175 kept.

| tier | n | reference one-shot | best-of-5 | black box | worst recipe |
|---|---|---|---|---|---|
| T1 far / 1200 | 59 | 100% | 100% | 3.4% | 3.4% (nothing; = all-null episodes) |
| T2 close / 1200 | 58 | 94.8% | 100% | 3.4% | 3.4% |
| T3 close / 550 | 58 | 77.6% | 100% | 1.7% | 1.7% |

Recipes: nothing, always_claim, prior, prior_or_none, random, name_probe, name_probe_thr, vocab_match.
