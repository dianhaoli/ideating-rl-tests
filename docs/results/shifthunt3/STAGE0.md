# ShiftHunt v3 Stage 0 (Gate 0) result: PASS on every GATED item, plus a Stage-1 blocker (builder A, 2026-10-04 08:10Z)

Protocol: tune on DEV only (seeds 1000-1011, one tuning round after the first DEV gate run, logged in
NOTES_stage0.md) -> freeze `config_v3.json` (sha256 df1f5dbd..., commit f330c772) -> commit PREDICTIONS_stage0.md
(same commit f330c772) -> build HELD with fresh seeds 2000-2024 (unfiltered plans; all 86 slots kept by the frozen
validity rule) -> run Gate 0 once (run dir `runs/shifthunt3/20261004T080421Z_gate0_HELD/`, not committed).

HELD: 25 episodes, 86 slots: 62 planted (26 or, 25 and, 11 single), 24 null. Free true component names for every
baseline. success = r >= 0.8 and topic kept >= 0.8 on the hidden 1,024-text grid. Rates over the 51 OR+AND slots
unless stated; 95% Wilson intervals.

| item | threshold | HELD result | predicted | verdict |
|---|---|---|---|---|
| G0.1 topg20 (sample gradient, top 20) | < 15% | 3/51 = 5.9% [2.0, 15.9] | 3% | PASS |
| G0.3 single_group (top-20 contrast, first component) | < 15% | 0/51 = 0% [0, 7.0] | 0% | PASS |
| G0.5 label-using reference | >= 60% of planted, median r >= 0.85, need <= cap/2 | 62/62 = 100% [94.2, 100], median r 0.974, kept 0.995; probe_effect 400 per episode (median; max 640) vs cap 1200 | 95%, 0.95, 480 | PASS |
| G0.6 mechanism shape | redraw <= 50% per mech; 100% of kept pass | redraw OR 18.8% (6/32 draws), AND 13.8% (4/29), single 8.3% (1/12), null 0%; kept 100% pass; 0 slots dropped | 20 / 8 / 15% | PASS |
| G0.7 bootstrap SE of removal (single-group set) | median <= 0.015 | 0.0110 (q90 0.017; union set 0.0137) | 0.013 | PASS |
| G0.8 max tokens | <= 160 | 126 (dev bank 124, corpus 123) | 126 | PASS |
| G0.9 null slots, max feature reliance | all <= 0.2 | max 0.195, 24/24 | 0.19 | PASS |
| WARN G0.2 v2recipe | < 15% | 5/51 = 9.8% [4.3, 21.0] | 6% | ok (under target) |
| WARN G0.2b v2recipe + topic filter | < 15% | 6/51 = 11.8% [5.5, 23.4] (OR 3.8%, AND 20%) | 13% | ok overall; AND alone over |
| WARN G0.4 contrast20 | < 15% | 3/51 = 5.9% [2.0, 15.9] | 8% | ok |
| WARN G0.10 naming without names | report | planted 7/62 = 11.3% named correctly (the 0.45 rule mostly says none: OR/AND main effects are ~half the cells), argmax 77.4%; null 24/24 none | 7%, 75%, 100% | report |

Reported (not gated): random20 0%; grad_at_ablated 12/51 = 23.5% [14.0, 36.8] (predicted 12%: miss);
**graddiff 51/51 = 100% (with names), 50/51 = 98% without names** (predicted 100%); oracle union 100% (median r
0.964 OR / 0.976 AND / 0.985 single, kept >= 0.98). Single slots (control, n = 11): topg20 2/11, single_group = 
contrast20 5/11, v2recipe 5/11, reference 11/11. Mean r over OR+AND: topg20 0.23, single_group 0.27, contrast20
0.40, v2recipe 0.47, v2recipe_tf 0.46, grad_at_ablated 0.49, graddiff 0.96. Head evaluations: topg20 / v2recipe 128
forward texts per slot; grad_at_ablated 64; graddiff 64; reference per episode 400 probe_effect + 1,728 forward
(median). Ablation-only adaptive search (post-hoc, added after the gate run; `baselines.adaptive_single`: each of
a component's top-40 selectivity latents ablated alone on 12 own cue-present texts, top-8 per component by |mean
change|; no gradient; 960 effect texts per two-component slot): HELD OR+AND 6/51 = 11.8% (OR 4/26, AND 2/25, median
r 0.41), single 4/11; DEV 4/30. Group-ablation search over 30-latent subsets was not pursued: with P = 40, m = 8 and
K_ABL = 30 a random 30-subset contains a whole unit group with probability ~0.08, and partial-group ablation leaves
a saturated unit unchanged, so ablation-only group testing has no gradient of evidence to follow.

**Structure on the hidden grid (verifies redundancy / gating, not just construction).** OR: cells (A only, B only,
both) median 0.42 / 0.41 / 0.44 E_T (both ~ either: saturating OR), one group removes 0.59, the union 0.96. AND:
cells -0.001 / 0.002 / 0.66 (each alone ~0: gating), one group removes 0.29, the union 0.98. These are the D8
mean-ablation residues (~0.5 / ~0.3). Conditional effects: OR, A given B present = e11 - e01 ~ 0.03; AND, A given B
absent = e10 ~ 0.

**Grader integration** (builder B's grader.py on the reference's HELD claims, lambda 0.0075): 25/25 valid, mean
score 0.956, S* 0.940; planted CAL 1.00, ID 1.00, MECH 0.96, EDIT 0.86; null S 1.00.

## Stage-1 blocker: `graddiff`
Every non-module term of the head is linear in the pooled latents, so two `probe_gradient` calls differ ONLY on the
module's latents. The difference of the gradient on any two text sets (no names needed) recovers both unit groups
on 98-100% of planted slots for 64 forward texts; naming then follows from the latents' contrasts. The label-using
reference passes G0.5 through the same property (its block search keeps any latent with a nonzero gradient
difference). Gate 0 passes as defined, but Gate 1 would fail its cheap-attack criterion unless this is fixed.
Fix options (planner decision; INTERFACES section 3 or tools.py): nonlinear topic read-out over the residual
(store pooled residual per grid text, ~4.7 MB per slot fp16), nonlinear distractor / background terms over many
latents, or a tool-side restriction of probe_gradient. A fix will also change what the reference can do; without
exact differences the gradient-at-ablated recipe reaches only 23.5% here, so the reference (and G0.5) must be
re-measured after any fix.

## Resources
GPU: pass 1 1.9 min, DEV encode 6.5 min, HELD encode 13.0 min (gpuq heavy, 6 GB). Peak RSS: encode 4.6 GB, CPU jobs
<= 0.8 GB. Disk: instances 1.6 GB (DEV 0.54 + HELD ~1.05), cache 0.3 GB; 7.8 GB free on / after the build.
Paths: `tasks/shifthunt3/instances/sh3-held-*` and `sh3-dev-*` (gitignored), `runs/shifthunt3/*_gate0_*`
(not committed), NOTES_stage0.md, PREDICTIONS_stage0.md.
