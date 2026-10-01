# CONTEXT

WHO: Dan, Duke CS undergrad. ML systems + RL post-training (GRPO/RLVR on tau-bench).
New to mech interp. Prefers breadth-first: many task types at low repetition
(5-10 instances) before deepening any one.

WHY: Oct 8, 2026, 2 PM PT hiring-manager call with d_model (YC S24 interp/alignment
lab that builds RL environments for frontier models). Half the call is a live
brainstorm pitching RL environments for an interp task. If he advances he builds one
end-to-end (Python, Docker, Claude Code). Dan needs: ONE validated environment,
3-4 explored-in-depth ideas in different mech-interp lanes, a research question he
can own, and the ability to explain what makes a task a good RL environment.

d_model's definition: an RL environment is a self-contained problem an agent
attempts alone, paired with a programmatic grader that scores objectively. Their
published work: agent-discovered concept erasure (Gemma 3 270M subject; hidden
grader, held-out activations, fresh nonlinear probe, L2 budget matched to LEACE;
best agent solutions beat LEACE on SVM accuracy for all 50 concepts across 560
rollouts; verify numbers before quoting).

GOOD-ENVIRONMENT FILTERS: (1) auto-verifiable, no LLM judge as primary grader;
(2) cheap procedural generation, thousands of instances; (3) mid-band difficulty,
frontier agent passes ~10-70%; (4) actually requires interp, black-box baseline
fails; (5) hard to reward-hack; (6) cheap: 0.1-8B subject models, one 24GB GPU,
bounded passes per episode; (7) has a difficulty dial; (8) skill transfers to real
interp/auditing work.

EXISTING WORK: "EditHunt" on Qwen2.5-1.5B/3B/7B and gemma-2-2b. Prompts like
"The state containing Dallas has its capital in". Findings: the model stores "which
state" as a linear direction at the city token over mid layers, then copies it to
the final token at a sharp handoff layer. A mean-difference steering vector flips
~85-90% of held-out cities; random vectors ~0%. Main side effect: leakage onto other
states' cities. Gradient-optimized vectors find a "say Sacramento" answer direction
that leaks heavily unless constrained. Tiers: Easy (flip capital), Hard (flip
capital, preserve state answer). Infra that exists: batched activation hooks, exact
candidate scoring, geography dataset, agent tool loop (the old loop called the
Anthropic API; do not use it, see AGENT UNDER TEST).

AUDIT LESSONS (from Dan's own task suite; apply to every new task):
- Tools that expose the fix as a named argument (e.g. keep_state_cities) leak the
  answer: the agent just calls it and the task stops testing interp. Tools must be
  generic primitives.
- A layer ceiling (max layer = handoff - 1) leaks the handoff. Do NOT use layer
  ceilings as a difficulty knob; the literature treats layer as a method
  hyperparameter.
- Tasks solvable by a fixed recipe (mean-difference vector passed 8/8), a
  behavioral lookup (a detective task answerable by asking the model), or a prior
  ("handoff is ~80% of depth" guessed right 6/8) test the shortcut, not interp.
- Mid-band difficulty must come from understanding, not knobs (caps, thresholds).
  Every difficulty conjunct must require something learned from THIS model's
  internals, not bookkeeping.
- n=7-8 cannot distinguish anything near a threshold. n=3 is a bug detector only.
- Per-instance pass rates must not be bimodal (GRPO needs variance within a group).
- The reference solver must derive answers from probing, never from instance
  metadata.
- NULL instances (nothing planted) with an explicit "nothing found" answer are
  required, or agents learn to always submit something. A published auditor-RL
  study found that training without ground truth led auditors to fabricate
  findings (~96% false-positive rate on clean targets; verify), and null/
  calibration targets fixed it. Planted and null instances must be
  indistinguishable (no fingerprinting).
- LLM judges are allowed only as a secondary signal, and only if they see the
  planted ground truth and compare pairs, never pointwise.

KNOBS WITH PRIOR-LITERATURE PRECEDENT (cite in each SPEC.md): number of
simultaneous edits (MEMIT scaling curves); keep/locality readouts (locality
metrics, specificity audits); norm/intervention budget (norm-constrained steering,
COAST); relation or bridge-entity family (latent multi-hop work: composability
~80% for country bridges vs ~6% for year bridges; verify); query/forward budget and
explanation-channel control (Pando); decoy budget and model-size separation
(AgentCE-Bench). Instance-calibrated norm budgets and conjunctive-constraint dials
have NO direct precedent: label them experimental.

SHORTCUT-FREE FILTER for relation families: exclude queries prone to subject-object
or relation-object shortcuts (Socrates, 2024); per-family feasibility check with a
reference patch.

STATUS OF T2 (hop-separation: change the answer, keep the intermediate state): the
one existing task with the right shape; naive recipe 0/7; reference solver passes
only ~57% one-shot; few held-out cities per instance. Needs: ~3x more held-out
cities, a hidden abbreviation readout, re-seeding, and keeping only instances where
the reference passes >=3/4 seeds.

PREFERRED SOURCES: AuditBench (Anthropic, 2026), Pando, InterpBench, MIB, SAEBench,
SAEScientist-Bench, AgenticInterpBench, Nanda et al. 2023 (grokking), Zhong et al.
2023 (Clock/Pizza), MEMIT/ROME, ADL (Minder et al.), TDC 2023. Flag any number from
a preprint as needing verification.

TEMPLATES: verifiers (Prime Intellect) ToolEnv/MultiTurnEnv fits best; reasoning-gym
is a good structural reference for procedural generators with built-in verifiers;
OpenEnv if containerization is needed.
