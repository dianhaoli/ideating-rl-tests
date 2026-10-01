# Plan v2 (2026-10-01 14:52 UTC): what changed relative to docs/PLAN_PROMPT.md

Dan re-sent the plan with a purpose statement and three additions. `docs/PLAN_PROMPT.md` is still the base text.
This file lists only what differs. Builders and reviewers must read both.

## Purpose (verbatim summary)
This is defensive AI-safety research: making language models easier to understand, audit and trust. We study
small open-weight models (0.1-8B) on one GPU. Any planted behaviour, such as an edit, a hidden trigger or a known
circuit, exists only so interpretability and auditing methods can be graded against a known answer. Nothing
targets real systems, real users or third-party models.

## New rules that apply to every task
1. **Reward shape.** Zero-effort policies must score about 0 under the reward. That includes the null action, flipping
   everything and a constant answer. Prefer **product** rewards (all conjuncts must hold) over averaged ones.
2. **Instances come from behaviour the model actually exhibits.** Check this per entity and per prompt template
   before use: the clean model must actually produce the fact or behaviour the instance relies on. Check for
   prompt-format artifacts, such as the model copying the last few-shot demo instead of answering.
3. **Cap in-episode training.** Trained-subspace methods such as DAS cost minutes per run on one GPU. The harness
   must cap gradient steps and wall-clock per episode.
4. **Ideation lanes.** There are 3 research lanes: (a) toy models, circuits, SAEs/transcoders; (b) auditing,
   hidden-behaviour detection, model diffing, data attribution, unlearning/erasure verification; (c) probing,
   steering/editing outside geography, behaviour prediction from internals, agents automating human interp techniques.
   Pick 6 for Wave 2: at least 2 not about steering or editing, and at most 2 from any one lane.
5. **TriggerHunt is optional** and may run as a separate later pass. Its trigger maps to a *harmless marker string*,
   and it uses a small LoRA on Qwen2.5-0.5B with frozen embeddings.

## T2-RAVEL (new spec for the hop-separation task)
Change a city's state-capital answer while keeping its state answer (question plus abbreviation readout) and other
states' cities unchanged. The intervention site is the city's last token at a per-instance layer (MIB RAVEL
convention; prefer layers 16-20). Reward:
R = Cause x Iso_state x Iso_other x (1 - min(1, KL)). A null edit and a flip-everything edit must both score ~0.
An instance is valid only if the reference gated edit reaches R >= 0.6 on >= 2 of 3 seeds. Generate extra instances,
since only about 28% pass. Tools are generic: cache_mean, vec_op, a read-only project(), and an optional input-dependent
scale on any additive edit. There is no named "gated" option. Norm budget is not a dial. Predict the agent pass rate before
running. Dan expects it to be low, so also design an easier tier.
Dan made this conditional on the EditHunt repo being present. **It is not on this machine.** See docs/DECISIONS.md D8
for what we do instead.
