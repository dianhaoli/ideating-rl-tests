# ShiftHunt v3: stopped after Stage 0 (planner decision, 2026-10-04 08:40Z)

**Decision: KILL this head family. No redesign attempt in this iteration.** Stage 1 and Stage 2 were not run, and no
v3 model episodes were run. Recommendation: **change direction.** Do not continue v3 as specified. Section 5 gives the
v4 requirements.

Aggregates only. This file names no instance, slot, feature pair or latent id. It is safe to copy to main.

## 1. What failed

Gate 0 on HELD (seeds 2000-2024: 25 episodes, 62 planted slots of which 51 are OR or AND, 24 null) passed every
GATED item as defined:

| item | result |
|---|---|
| top-\|g\| | 5.9% |
| single-group ablation | 0% |
| label-using reference | 100%, median removal r 0.974 |
| redraw rate | at most 18.8% |
| standard error of removal | 0.011 |
| max tokens | 126 |
| null max reliance | 0.195 |

The STAGE0.md table has the full detail. The independent audit rated it FAIL because a stronger cheap baseline
clears the 15% bar by a wide margin:

| finding (audit, HELD) | result |
|---|---|
| sample gradient difference (label-1 minus label-0 sample ids, or two random halves; no names, no own texts, 128 forward) | **49/51 = 96.1% [86.8, 98.9]** of OR+AND slots; single slots 11/11 |
| naming each support latent by its largest contrast | components exact on 59/62 planted slots |
| top-\|g\| restricted to each named component's top-40 selectivity pool | 8/51 = 15.7% [8.2, 28.0] |
| the "equal thirds" sample pattern, read with the exact detectors | mechanism 86/86 slots, OR components 38/38 |
| free sample logits within topic / E_T, planted vs null | AUC 0.989 (Gate-1 bar 0.65) |
| label-using reference | its groups equal the true groups exactly on 58/62 slots: it passes through the same exact-difference property |

Builder B's grader dry run on DEV (12 instances): the no-name `graddiff` attack reaches S* **0.567**. The Gate-1 kill
line is 0.35.

## 2. Why there is no redesign: every remedy was measured or bounded

The cause is structural. The head is a linear topic read-out, plus linear distractors, plus one small planted
nonlinear module, and the tools are exact. So the module has an exact signature in the algebra of the tools: a
gradient difference, a break in linearity, or co-linear unit latents. The intended insight ("the gradient or effect at
a partially ablated point") and the cheapest attack read the same signal.

| remedy | what is left | evidence |
|---|---|---|
| (a) `probe_gradient` on the whole sample only, with an optional `ablate` | ablate the sample's label-contrast latents in blocks, keep the latents whose gradient moves: exact module support on **44/44** DEV slots (6 calls, 768 forward per probe, no names); B's no-name ablation-gradient-shift script reaches **S* 0.552** on DEV | builder C, 08:15Z (INTERFACES amendment 08:22Z); builder B, commit 683402b7 |
| (b) one fixed whole-sample gradient, no ablate (builder C's current tools.py, commit 7380e116) | the fixed gradient g0 is the exact coefficient of every non-module latent, so a `probe_effect` block test on 2 sample ids is exact: mismatch at most 1.8e-6 on 42/42 DEV blocks without a module latent, median 6e-3 on 73 blocks with one, and one text pair flags 73% of single module latents; bisection is bounded only by the tool-call cap | builder C, DEV, INTERFACES 08:22Z item (2) |
| (c) drop `probe_gradient`; `probe_effect` only, frozen pool P = 40 > per-call cap K_ABL = 30 | the latent half becomes group testing over saturated 8-of-40 groups, and knowing the cue adds nothing. A screen-then-leave-one-in script on HELD solves OR+AND **17/51 = 33% [22, 47] with names and 18/51 = 35% [24, 49] without names** (single 7/11 in both; 1/24 null slots get a spurious group; median 260-286 effect texts per slot, within the 1200-per-episode cap). So the G0.5 bar (at least 60%) is out of reach for this script, and any better reference is equally available without names | planner, `explore_v3/ablonly_grouptest.py` (.log), 08:35Z |
| (d) as (c) with P <= K_ABL (one call can ablate a unit's whole pool) | the same no-name script solves **41/51 = 80% [68, 89]** of OR+AND slots and 11/11 single slots | same script, `pool30` variant |
| (e) a nonlinear topic read-out (for example over LayerNorm of the residual) | its gradient change across calls is about rank 1 in gT, so it can be fitted and projected out, and block tests become statistical rather than exact. It does not remove the module's outsized response near its transition | builder B's analysis (reasoning, no run) |
| (f) many weak nonlinear nuisance units | every latent of one unit has a gradient equal to one per-text scalar times a fixed vector, so units cluster exactly by direction. This raises the cost but leaves the signal | builder B's analysis (reasoning, no run) |
| (g) noise on tool outputs | it makes the attack cost more but leaves the signal. The intended solver needs the same signal, so noise blocks both equally | reasoning |
| (h) dense trained (MLP) heads with behavioural ground truth | outside this timebox: new head family, residuals stored per grid text, new grader and new reference. It also runs into section 3, point 3 | reasoning |

## 3. What it teaches

1. **Planted sparse structure plus exact tools means the structure has an exact signature.** If every non-planted term
   is linear, any differential query (a gradient difference, a break in additivity, a linearity residual) isolates the
   planted part with no hypothesis about what it reads. The v3 "intended insight" was one such query. It went from a
   reference strategy to a no-name script with no step in between.
2. **Hiding the signature turns the latent half into search, not interpretation.** Once gradients are removed, a
   saturated unit only responds when nearly its whole group is ablated. Finding the group is then a group-testing
   puzzle, and its difficulty is set by P versus K_ABL, not by any understanding of the cue. The measured symptom: the
   label-using script and the no-name script score the same (33% and 35%).
3. **A closed, enumerable cue bank makes the feature-to-latent map a cheap statistic.** With 16 known features
   rendered by our generator, contrasts on generated texts give any script each feature's latents, and the 5
   features co-occurring in the sample give the candidates. Interpretation (forming a hypothesis about which text
   property matters, then testing it) becomes necessary only when the hypothesis space is open. v2's naming saturation
   was the first symptom of this; v3's latent half is the second.
4. **Constructed confounded data leaks through its construction.** The "equal thirds" OR sample revealed the
   mechanism on 86/86 slots. Free logits on confounded texts, combined with the disclosed E_T, separated planted from
   null (AUC 0.99-1.00). Any future sample must look the same for every mechanism and for nulls, and no free score
   may combine with a disclosed scale.
5. **Process.** Every GATED Gate-0 baseline was an attribution-trusting recipe (top-|g|, top contrast, g x delta).
   None was "the cheapest script that uses the exact algebra of the head family". Builders A and B flagged the leak
   themselves and the audit caught it, but only after the HELD build. Future Gate 0s need a **tool-algebra red-team
   baseline, written from the head math before any build**, as a GATED item.

## 4. What still holds (reusable)

- An encoder with no truncation that raises above 160 tokens. This closes v2's 96-token channel (max 126 measured).
- A factorial hidden grid with exact CPU grading: bootstrap SE of removal 0.011 at 1,024 texts.
- Continuous sub-scores (CAL / ID / MECH / EDIT) with a per-latent price. The oracle scores 0.955, a wrong combination
  loses 0.11 per slot, and a padded edit loses 0.11 EDIT on average.
- Exact cue detectors and a pair grader for an open-menu arm.
- Harness fixes in their own branches: wall clock, runner/grader agreement, /tmp, OS isolation.
- The mechanism-shape evidence: OR and AND heads really are redundant and gated on the grid. OR cells are
  0.42 / 0.41 / 0.44 with union removal 0.96. AND cells are about 0 / about 0 / 0.66 with union removal 0.98. The
  nonlinearity was never the problem; how visible it is to exact tools was.

## 5. What a version that works would need (v4 direction; not built)

1. **An open cue space.** Each instance draws its cues from a generative grammar of surface properties that have
   exact detectors (compositional, not a listed menu, and not v2's or v3's 16). Naming then needs a hypothesis from
   data, and the open-menu pair grader (already built) checks it.
2. **Heads trained on data, not planted modules.** Ground truth is behavioural: grid cells, reliance, and removal by
   an edit. Edit quality is scored against a ceiling found by an optimizer on the hidden grid, so a removal ceiling
   costs nobody unfairly. There is no "find the planted group" item, because that item is either exposed or a lock.
3. **A budget that makes enumerating the grammar infeasible**, while a few hypothesis-driven tests fit inside it.
   Exact tools are acceptable once the hypothesis space is open.
4. **A GATED tool-algebra red-team baseline in Gate 0**, plus "label-using vs no-name script under the same procedure"
   as a GATED gap. If names do not help a script, the task is not measuring interpretation.

Estimated build: more than one 10-hour iteration. It needs new renderers and detectors for the grammar, GPU
re-encoding, a trained-head pipeline and a new reference.

## 6. Recommendation

**Stop v3 and change direction.** For the Oct 8 call, present v2 (validated with caveats) and this v3 negative result
as a finding: "mechanism-level claims about planted modules are scriptable through the tool algebra, and
interpretation needs an open hypothesis space". Do not spend the rest of today's timebox on v3 Stage 1 or Stage 2: by
the measurements above, Stage 1 would kill it, through `graddiff` / `lintest` or, if gradients are dropped,
through G0.5. Scoping v4 (section 5) is the useful next step.

## 7. Thin evidence (flagged)

- The planner's ablation-only numbers (rows c and d) come from one script, written in about 20 minutes, with one seed.
  Its candidate pools use dev-bank contrasts, which is optimistic in the same way as A's reference. The `pool30`
  variant imitates a P <= K_ABL family by putting the true group inside a 30-latent pool; heads were not
  regenerated. These are lower bounds on what a script can do. The conclusion rests on the with-names vs no-names
  pair run under one identical procedure (33% vs 35%), not on the absolute level.
- B's 0.552 and C's 44/44 and block-test numbers are DEV (12 instances), not HELD.
- Rows (e) to (h) are reasoning, not runs.
- No LLM agent ran on v3. Opus might beat every script by a wide margin. But then Opus would be above the 0.30-0.70
  band, and the kill criteria ask whether the score can be reached without interpretation. Sections 1-2 show that it
  can.

## 8. State left behind

- Branch `task/shifthunt3` stays local only. instances/, cache/, scratch/ and runs/shifthunt3/ are gitignored; this
  commit adds `runs/shifthunt3/` to the root .gitignore, as the audit asked.
- Builder B's uncommitted edits (grader, schema, attacks, tests) were not touched.
- v2 (`tasks/shifthunt/`, `~/wt/shifthunt`) is untouched.

## 9. Independent kill audit (2026-10-04 08:33-09:00Z, skeptic pass; aggregates only)

**Verdict: uphold the kill.** I tried to show it was premature and could not. Two of the planner's statements are
overstated (corrections below), but both corrections make the gradient-free remedy (c) look worse, not better.
Scripts are in `scratch/kill_audit/` (gitignored). Success means r >= 0.8 and kept >= 0.8 on the hidden grid, as in
Gate 0. Every number is OR+AND over the 51 HELD slots unless it says DEV.

**(1) Re-test of the thin with-names vs no-names claim (row c: probe_effect only, P 40 > K_ABL 30).**

| script | caps enforced | names | no names |
|---|---|---|---|
| planner's `ablonly_grouptest.py`, seed offset 0 (reproduced) | effect texts only | 17/51 | 18/51 |
| same script, seed offset 7 | effect texts only | 20/51 (39%) | 13/51 (25%) |
| same script, both seeds, **successes within the 200 tool-call cap** | texts + calls | **0/102** | **0/102** |
| mine, `excl_fit.py`, seeds 0 / 1 | texts + calls + forward | 16/51, 17/51 (32%) | 7/51, 8/51 (15%) |
| mine, `halving.py` (adaptive block halving; first version, DEV 6 episodes, 2 seeds; the file now holds a steep-base variant that scored worse: names 4/60 on all DEV) | texts + calls | 8/34 (24%) | 0/34 |

How `excl_fit.py` works: per candidate there are 4 calls, each ablating the pool minus one block of 10, on 16 + 16
cue-on and cue-off texts. The other candidates are off in one set and on in the other. It then fits the sigmoid unit
model per text on the latent_means activations. The no-name screen is one call per candidate.

- **Names help only modestly.** The gap is +6 points (planner's script, 2 seeds pooled) to +17 points (mine). Most of
  the gain is the cost of screening 5 candidates instead of 2. The no-name screen also flags 39/48 null slots.
- **No script gets near the G0.5 bar of 60%.** The best names-using result is 39%, and that script breaks the call
  cap. Within all caps the best is 32%. So the stated threshold is not met (a gap of at least 25 points with
  with-names at 60% or more). Names are also not private: the audit's sample-pattern leak gives OR components 38/38
  and the mechanism 86/86.
- The conclusion "remedy (c) cannot pass G0.5, and names barely matter" stands. The "33% vs 35%" pair is one seed and
  is seed-sensitive.

**(2) Remedy (f), now measured.** DEV, 30 OR+AND slots x 2 seeds, under the multi-call gradient tool set where
graddiff is possible. I added weak nonlinear nuisance units (same unit recipe, kappa 24, m 8) on non-component
features, half of them on confounders, each with C_n = FRAC x E_T:

| attack | no nuisance | 4 units at 0.15 | 6 units at 0.20 |
|---|---|---|---|
| graddiff without names (top-20 \|difference\|) | 60/60 | **9/60** | **4/60** |
| graddiff with names (difference restricted to the named components' pools) | 60/60 | 60/60 | 60/60 |
| graddiff restricted to the 5 sample candidates' pools (no names) | 60/60 | 22/60 | 9/60 |
| cluster by gradient direction, rank clusters by ablation \|effect\| (no names) | 60/60 | 48/60 | 33/60 |
| cluster by direction, keep the pair with the largest interaction \|eff(A u B) - eff(A) - eff(B)\| (no names) | - | **60/60** | **60/60** |

- (f) does defeat the cheapest attacks and opens a 50-55 point names gap against them.
- An exact, nuisance-proof signature remains. Nuisance units are additive, and the planted pair is not. A no-name
  interaction script restores 60/60.
- So the KILLED.md judgment on (f) ("raises the cost, leaves the signal") is right, and it is now measured, not only
  reasoned. Not tested: single-mechanism slots (no partner, so separation is by magnitude) and remedy (e).

**(3) Corrections to the text above.**

- **Rows c-d and section 2 omit the tool-call cap.** "Median 260-286 effect texts per slot, within the
  1200-per-episode cap" is true for effect texts. But the script uses a median of 71 (names) and 107 (no names)
  probe_effect calls per slot, against a per-slot share of about 55 of the 200-call episode cap. 75-86 of 86 slots
  are over that share, and 0 of its successes fit the cap. So the 33-35% is not a lower bound under the frozen caps;
  it is unattainable. This favours the kill.
- **Section 3.2 is overstated, as is STAGE0's "no gradient of evidence to follow".** The claim is that a saturated
  unit "only responds when nearly its whole group is ablated". This is wrong for this head family. beta is
  proportional to contrast and is skewed: on HELD units, the top-1 beta has median 0.41, and the top-3 beta sum has
  median 0.76. On DEV, random 30-of-40 ablations move the logit, as a median fraction of E_T, by:
  - 3e-5 with 3 of 8 members ablated, 4e-4 with 4, 0.027 with 5, 0.14 with 6, 0.25 with 8;
  - against linear collateral of 0.003 (median).

  So the group test is graded and informative, not all-or-nothing. A single top-beta latent can carry 50-70% of a
  unit. The binding limits are the call and text caps, plus the fact that r >= 0.8 still needs most of the beta mass
  (ablating the top-3 beta members per unit gives 6/51).
- **"Names add nothing" (sections 2c, 3.2 and 18) should read "names add little (+6 to +17 points) and no
  budget-respecting script reaches G0.5".** The lesson in 5.4 (gate on a with-names vs no-names script gap) stands.
- **Row (f) should be cited as measured (above), not reasoning.** Row (e) remains reasoning only.

**Redesign? No.**
- (c) fails G0.5 under the frozen caps, for every script I tried.
- (f) is closed by an exact interaction signature.
- The sample-pattern and free-logit leaks are untouched by either remedy.
- I found no design with measured evidence of closing every listed leak. None could reach a held-out Gate 0 by about
  12:00Z.
