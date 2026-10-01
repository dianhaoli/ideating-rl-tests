# FeatureMatch: PREDICTIONS (written 2026-10-01 16:52 UTC)

Written and committed BEFORE any LLM test-agent episode on this task, and before the harness gate runs
(`runs/featurematch/20261001-165118_gate_*`) produced any result. Never edit the predictions below; add dated
"Outcome" sections underneath instead.

What "pass" means: every slot of the episode is correct (the right option on planted slots, "nothing found" on null
slots). Episodes have 3-5 slots and each slot is null with probability 0.4. Tiers: T1 = distractors drawn from all
concepts, 1200 forward units; T2 = distractors are taxonomy siblings, 1200; T3 = siblings, 550.

**Everything below is a guess**, except where it says it is anchored on the preliminary in-process gates
(`runs/featurematch/20261001-153149_prelim_v2`, `..._160933_prelim_v2b`). The LLM numbers have no measurement behind
them at all.

| solver | T1 | T2 | T3 | basis |
|---|---|---|---|---|
| Main test agent (Opus-class Claude Code subagent, sandboxed, `full` profile) | 35% | 20% | 12% | guess |
| Small agent (Haiku-class Claude Code subagent) | 10% | 5% | 3% | guess |
| Black-box control (`blackbox` profile, scripted) | 3% | 3% | 3% | anchored on prelim |
| Worst recipe baseline (of the 8 variants) | 4% | 4% | 4% | anchored on prelim |
| Reference solver, one-shot | 98% | 93% | 78% | anchored on prelim |
| Reference solver, best-of-5 on kept instances | 100% | 100% | 100% | true by the drop rule |

One sentence of reasoning each:
- **Opus-class agent.** I expect it to find the right probing method (write a few texts per option, read the latent's
  max activation, keep the best) but to lose slots to two things: about half of the latents respond to the
  dataset's text style more than to the topic (NOTES 16:42 style check), so its self-written probes will sometimes
  make a planted latent look dead and it will answer "nothing found"; and near-miss null slots (the latent's own
  concept is missing but its siblings are listed) will tempt it to claim the closest sibling. With 3-5 slots and the
  all-correct pass rule, a per-slot accuracy of about 0.8 / 0.7 / 0.6 gives roughly 35% / 20% / 12%.
- **Haiku-class agent.** I expect it to probe fewer options per slot, to spend budget unevenly (running out in T3),
  and to calibrate "nothing found" poorly, so per-slot accuracy around 0.55 / 0.45 / 0.4 compounds to 10% / 5% / 3%.
- **Black-box control.** Without SAE access the latent index carries no information, so it can only pass episodes in
  which every slot is null and it happens to answer "nothing found" everywhere (about 3% of episodes).
- **Recipes.** The constant "nothing found" passes exactly the all-null episodes (about 3%); every claiming recipe
  needs every planted slot right, and the generator rejects planted slots that the name-probe shortcut solves.
- **Reference one-shot.** The reference probes with real dataset text (split C), which matches the style the latents
  were selected on; its errors come from close siblings (T2) and from too few probe texts under the 550 budget (T3).
- **Reference best-of-5.** Instances the reference cannot solve in 5 tries are dropped (builder guide rule 8), so
  this is 100% on kept instances by construction; the informative number is how many get dropped (I guess 0-3 of 72).

Secondary predictions (guesses), to check after the smoke run:
- The Opus-class agent's slot-level null false-claim rate will be 20-35%, mostly on near-miss nulls.
- Its "nothing found" rate on PLANTED slots will be 10-25%; if it is above 30%, the style-dependence weakness is
  the main driver of difficulty and the task is measuring probe-style matching more than feature interpretation.
- Per-instance pass rates will not be bimodal in T2/T3 (some instances pass on some attempts and fail on others).
