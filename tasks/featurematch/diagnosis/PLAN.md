# FeatureMatch diagnosis: plan (Dan, 2026-10-02 ~01:10 UTC) + orchestrator's operational decisions

## Dan's brief (verbatim summary of the binding parts)
GOAL: diagnose whether FeatureMatch is hard for the RIGHT reasons (interpretability skill) or the WRONG ones
(probe-writing style, unclear tools, early stopping, bugs). Document in docs/LOG.md, push regularly, time-box each
step (~1 h), drop dead ends with a note, record assumptions in docs/OPEN_QUESTIONS.md.
Known so far (v1 pool only; nothing has run on v2): keys and grader sound; every miss an avoidable agent error;
style dependence (~50% of eligible latents recover their concept on hand-written probes; MMA latents fire on 54/54
dataset texts but read 0 on agent sentences); no agent called top_latents or generate; every agent stopped early
(8-62% of budget); response keys (max, acts) undocumented. Baselines on v1: self_probe 0.14, template_probe 0.11,
black-box 0.00, reference 0.96. Planted "nothing found" rate: recipes 0.73-0.79, reference 0.02.
Test agents = fresh contained subagents (no API key; do NOT call any hosted API); they get only the task statement,
tool docs and answer format. Label results "fresh Claude Code subagent (model X)". n<20 per setting = bug-finding only.

STEP 0 PRE-REGISTER (commit before running anything): diagnosis/PREREG.md: right-reason failures (close-sibling
confusion, word-vs-concept confusion, null over-claim, wrong hypothesis after adequate probing) vs wrong-reason
failures (latent silent because of style, tool/format error, early stop with unused budget, parse bug); a SCRIPTED
classification rule for each; predictions for every experiment; decision rule: FeatureMatch is "hard for the right
reasons" if, on v2 after the style filter, pass rate is 10-70%, >= 60% of failures are right-reason, the agent's
planted "nothing found" rate is closer to the reference than to the recipes, and pass rate drops as sibling
similarity rises.
STEP 1 TOOL AND TASK CLARITY AUDIT (no GPU): ambiguities (undocumented response keys, role of top_latents/generate,
budget counting, JSON quoting); transcripts where an agent considered and rejected a tool; diagnosis/clarity.md and a
revised TASK.md (same capabilities, clearer; do NOT name the method or hint answers); keep old as TASK_old.md.
STEP 2 STYLE-ROBUSTNESS FILTER AND V2 POOL: per concept ~20 differently styled texts (casual, news-like, dialogue,
listicle, ...); keep a planted slot only if its latent still separates its concept (AUROC >= 0.85 on this set); report
survivors and how they differ in fire rate, layer, margin (the filter must not just select easier/denser latents); log
dropped slots and why.
STEP 3 BASELINES ON FILTERED V2: reference, black-box, self_probe, template_probe, and a NEW scripted style-robust
recipe (probes each option with several styles, picks max-activating). Gate: non-reference recipes <= 10-15% on
planted slots; reference >= 95%. If the style-robust recipe passes most slots, FeatureMatch does not require
interpretability: stop and report.
STEP 4 AGENT EXPERIMENTS (fresh contained subagents; v2-filtered pool only; same 20+ instances per arm where
possible, with nulls): A baseline (revised TASK.md); B old TASK.md (clarity ablation); C persistence (require >= 60%
of forward budget before submit, or higher effort); D oracle examples (a few in-style example texts per option);
E dial sweep (sibling similarity random/sibling/near-sibling; text budget low/high; latent density dense vs sparse);
F model separation (smaller vs larger model, same instances). Smoke each arm on 3 instances first, read transcripts,
fix environment faults, then scale. Save full tool responses.
STEP 5 CLASSIFY AND REPORT: classify every failed slot by the PREREG rules (spot-check transcripts). Per arm: pass rate
+ Wilson 95% CI, planted accuracy, null false-claim rate, planted "nothing found" rate, budget fraction used at submit,
failure-mode counts, accuracy by density and by distractor similarity, tool-usage counts. Plots: pass rate vs sibling
similarity; failure-mode composition per arm.
DELIVERABLES: diagnosis/PREREG.md (committed first), diagnosis/clarity.md, diagnosis/RESULTS.md (predictions vs
outcomes, tables, 3 example transcripts: right-reason failure, wrong-reason failure, success), diagnosis/VERDICT.md
(one page). Never claim mid-band from n<20; never edit PREREG.md after results (append dated outcome sections).
HARD CONSTRAINTS: no secrets/large files in git; never print HF_TOKEN; no history rewrites; never expose answer keys
to test agents; do not call any hosted API; at most 2 GPU jobs, ~10 GB each (for this study's own jobs).

## Operational decisions (orchestrator; see docs/DECISIONS.md D14)
- Work in worktree ~/wt/fmdiag, branch diag/featurematch (from task/featurematch + main). The FeatureMatch fix stage
  is still running in ~/wt/featurematch; merge task/featurematch into diag/featurematch after it finishes.
- GPU: one shared FeatureMatch MODEL SERVICE (one process, gemma-2-2b + SAEs, <= 10 GB) serves every episode's tool
  server over a local socket; per-episode servers become CPU-only. This keeps the study at <= 2 GPU jobs while running
  several episodes concurrently.
- Prompt variants (arms A/B/D): `common.sandbox prepare --prompt-template FILE` overrides tasks/T/agent_prompt.md.
- Arm C: `prepare --min-submit-frac forward=0.6`: the sandbox's ./tool client refuses `submit` until >= 60% of the
  forward budget is used (message says how much is used/required; generic, no answer info).
- Arm D: `prepare --extra-file SRC:DST` copies a per-instance examples file (in-style dataset texts per option, drawn
  from a split NOT used to define ground truth and NOT used for the final held-out check) into the sandbox.
- Two independent style banks written by different writer agents: bank F (filter, step 2) and bank R (the new
  style-robust recipe, step 3), so the recipe cannot pass merely by reusing the filter's own texts.
- Test agents: workflow subagents with only the prepared prompt. Main model = the session's Opus-class model; small
  model = haiku (arm F). Transcripts are audited by common.transcript_audit; reads outside the sandbox invalidate.

## STEP 1b (Dan, 2026-10-02 ~01:35 UTC): KEY DEFINITION AND DISCLOSURE (decide before Step 2). Binding.
- The style-robustness filter is the MAIN VALIDITY FIX, not a cleanup. Redefine the answer key on a multi-style set
  (encyclopedic dataset text plus casual, news, dialogue, listicle, ...): a latent's concept is the one it separates
  best ACROSS STYLES, not on DBpedia-style text alone. Keep a planted slot only if the key is the same under this new
  definition and the original one; log every slot where they differ.
- Make top_latents and generate clearly documented in TASK.md (what they return and when they help), so that "see
  what fires the latent" is a visible, measurable option, not a hidden trick. Record how often agents use them.
- Add an arm that DISCLOSES in TASK.md how the key is defined ("options are disjoint dataset classes; the correct
  option is the class the latent separates best across varied styles") vs an arm that does not. Same filtered pool.
- In RESULTS.md, report how much of the pass rate the style-robust recipe and the oracle-examples arm explain. If
  either explains most of it, say plainly that FeatureMatch is mostly measuring style, not interpretability.

### Orchestrator implementation notes for 1b
- Multi-style key: per concept, texts = dataset split A (encyclopedic) + bank F (10 styles x 2). To avoid defining and
  checking the key on the same texts, split bank F per concept into F1 (one text per style) and F2 (the other): the
  multi-style key is computed on A + F1; the "same under both definitions" check AND the AUROC >= 0.85 robustness filter
  use C + F2 (held out from both key definitions). Bank R stays untouched for the style-robust recipe.
- Arm G (disclosure): revised TASK.md + one disclosure paragraph (exact wording above) vs arm A (no disclosure).
- PREREG and clarity outputs from phase A predate 1b: they get dated, pre-results amendments (PREREG addendum
  "Amendment 1 (before any experiment)" and a clarity/TASK revision documenting top_latents/generate).
