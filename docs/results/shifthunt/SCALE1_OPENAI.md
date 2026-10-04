# ShiftHunt cross-family check: OpenAI agents on the scaled-run instances (2026-10-03)

Run dirs: `runs/shifthunt/20261003-openai_T2_sol` (gpt-6.1-sol, T2), `20261003-openai_T1_sol` (gpt-6.1-sol, T1) and
`20261003-openai_T1_luna` (gpt-6-luna, T1); 10 episodes each, one per instance, effort medium (the Opus arms ran at
Claude Code effort xhigh [check C1]), 18:18-19:40 UTC (a 2-episode smoke run, then the rest 3 at a time through
`runs/shifthunt/_openai_jobs/job.sh`). Instances: the same 10 T2 instances as SCALE1_T2 (Opus 5.5, 2 episodes each)
and the same 10 T1 instances as SCALE1_T1 (Opus 5.5, 2 each) and the Haiku 4.5 arm (1 each). The comparison is paired by instance and by slot. Reviewed from all 30 transcripts
(api_transcript.jsonl: every command and tool output, by script; the failed T2 episodes, two luna episodes and both
anomalies read in full), grade.json, audit.json, runner.log / api_meta.json, tool logs, submissions and episode.json.

Per D6 and the exposed_instances.json rule this file gives aggregates only. It names no instance, episode or latent
id, and does not say which attribute belongs to which slot. Quotes are redacted the same way. The run dirs stay on
disk and uncommitted. All 20 instances were already LLM-seen (`llm_seen`); they were reused on purpose for the paired
design, so exposed_instances.json does not change.

No predictions were written before this run.

**Amended after the adversarial check.** SCALE1_OPENAI_CHECK.md (commit d0f27a29) re-derived every number here
(regrade 30/30, audit re-run 30/30) and verified both anomaly resolutions. It found no wrong grade, count or CI, but
made nine corrections. They are applied in place below and marked [check C1] to [check C9]: the two families ran at
different reasoning effort (C1); the T2 match holds for pass rate, not slot by slot (C2); the argument against copying
covers T1 only (C3); four luna details (C4); runner truncation was not counted (C5); the files anomaly (a) read (C6);
the exec comparison (C7); an un-ignored job directory and what job.sh's rc means (C8); the T2 difference interval and
the runner env (C9). The bar-sensitivity rows for T1 and the tau + 0.05 column were recomputed from grade.json for
this amendment.

## Summary

- **Outcomes (Wilson 95%).**
  - T2, gpt-6.1-sol: **7/10 = 70% [40, 89]**. Opus 5.5 on the same instances: 15/20 = 75% [53, 89]. Fisher p = 1.0;
    the difference is -5 points, 95% interval -38 to +24 (Newcombe) [check C9]. Both rates are on the keep-filtered
    pool (SCALE1_T2 check C1).
  - T1, gpt-6.1-sol: **10/10 = 100% [72, 100]**. Opus 5.5: 19/20 = 95% [76, 99].
  - T1, gpt-6-luna: **0/10 = 0% [0, 28]**. Haiku 4.5: 0/10 [0, 28].
- **Unequal settings [check C1].** Every assistant row of the Opus T1 and T2 transcripts records Claude Code effort
  `xhigh` (3,300 rows); recorded thinking averages 809 s per T2 episode and 386 s per T1 episode. The OpenAI agents
  ran at `medium` (median reasoning 3.8k tokens per episode for T2 sol, 3.1k for T1 sol, 11.6k for luna). Haiku has no
  effort setting. Model, effort and system prompt all differ at once, so "sol matches Opus" means sol at medium
  reaches the pass rates of Opus at xhigh.
- **The pass-rate pattern repeats across model families.** T1 saturates for both large models. T2 lands at 70-75% for
  both, with naming and null calls perfect (sol 26/26 named, 0/9 null false claims). Every T2 failure in both
  families is a removal miss on a correctly named slot. Both small models score 0/10 on T1.
- **At slot level, T2 sol is weaker than Opus and more sensitive to the bar [check C2].** sol's margin is below the
  Opus mean on 19/26 planted slots (mean difference -0.020, paired Wilcoxon p = 0.005). sol against a single Opus
  agent correlates r = 0.86 and 0.88, against 0.955 between the two Opus agents. A 10% higher bar gives sol 3/10 and
  Opus 13/20. In T1, sol-Opus agreement equals Opus-Opus agreement.
- **Same instances, different misses (T2).** All 8 failing T2 slot-episodes (3 sol, 5 Opus) sit on the 13 of 26
  planted slots where the label-free reference clears the bar by less than 0.05. None sit on the other 13. Margins on
  the same slot correlate r = 0.88 between families. But the two families' failed slots overlap on only 1 of 5, and
  sol passed one of Opus's two both-fail instances. Which slots are tight is shared. Which tight slot misses depends on
  each agent's latent choice.
- **T1: the same pipeline gives the same latents.** sol and Opus remove the same amount on every slot: sol / Opus
  median 0.998, absolute margin difference median 0.005, r = 0.98. Their latent sets overlap (Jaccard median 0.82) as
  much as two Opus agents' sets do (0.82).
- **luna is not Haiku.** Both score 0/10, for different reasons. luna names cues through counterfactual pairs (19/22
  right; Haiku 7/22, McNemar p = 0.002) and makes fewer null false claims (5/13; Haiku 13/13, p = 0.008). It then
  submits a median of 4 latents and removes about half of what the bar needs; the two episodes that estimated their
  own removal overestimated it 2-3x [check C4]. With the removal bar at 0, luna passes 6/10 and Haiku 0/10 (Fisher
  p = 0.011). This is luna at medium effort; no higher effort was tried [check C1].
- **Anomalies resolved.**
  - (a) The INVALID T2 pass is an **audit false positive**: `'/'` as the replacement in `.replace('\n','/')` inside
    a print.
  - (b) Both T1 "ended_without_submit" episodes **did submit**, through `subprocess.check_output(['./tool','submit',
    ...])`. The runner's submit regex missed that form. The grades are correct: the record shows `submitted`, and an
    in-memory regrade matches.
  - Neither changes a grade.
- **Context exposure.** These agents ran confined to their sandbox, with no repo, session or shared-directory context.
  They reach the Claude agents' pass rates and, in T1, the same latent sets. So the Claude results need no context
  channel to explain them. This is evidence against large inflation, not proof of none (see the last section). The
  latent-set argument against copying covers T1 only: in T2 the two Opus agents' sets overlap each other more than
  sol's, which fits same-model similarity but cannot exclude a shared channel [check C3].
- **Cost.** $5.60 for all 30 episodes (sol T2 $3.33, sol T1 $2.00, luna $0.28). No episode came close to its $1.50 or
  90-turn cap.

## Setup and provenance

- **Code.** All 30 episodes record git_sha 955ea331 with git_dirty true. The working tree differs from HEAD only in
  runs/ (`runs/api_budget/ledger.jsonl` and untracked run files). `git diff fd91d8b2 955ea331` is empty for `tasks/shifthunt/{tools,grader,sh_core,sh_probe}.py`,
  agent_prompt.md, taus.json and `common/`. So the task code, grader and audit are those of the Claude T2 and Haiku
  arms. Prompt template ed473483 in all 30, the same as Opus T2 and Haiku. Opus T1 ran template 1bdd2846, without the
  96-token sentence and the E7 exec wording.
- **Runner.** `common/openai_agent.py`, Responses API, one `bash` tool. Each command runs with cwd = the episode
  sandbox and env {PATH, HOME = sandbox, LANG, TERM, PYTHONDONTWRITEBYTECODE} [check C9], behind the `_run_bash`
  command guard. The guard blocks /home/, .claude, rlsbx, ../, curl, wget, /proc/ and similar. There is no repo cwd,
  no commit subjects, no account email, no shared scratchpad or tool-results directory. The system prompt is
  api_agent's SYSTEM. Unlike the Claude Code launch, it adds "Think about what evidence would
  actually distinguish the possible answers, gather it efficiently within the budget". Caps are 90 turns and $1.50
  per episode; the task caps are as in the Claude arms. This is not a kernel sandbox (LOG 2026-10-03 19:40).
- **Effort [check C1].** OpenAI `medium` in all 30 (api_meta.json). Opus T1 and T2: Claude Code effort `xhigh` on all
  3,300 assistant rows, with 809 s (T2) and 386 s (T1) of recorded thinking per episode on average. Haiku: no effort
  setting. The earlier version of this file gave only the OpenAI setting.
- **Output truncation [check C5].** The runner keeps the first and last 6,000 characters of any tool output over
  12,000 (Claude Code's Bash shows about 30,000). That cut 102 outputs in 25 of 30 episodes: 52 in T2 sol (10/10
  episodes), 8 in T1 sol (5/10) and 42 in luna (10/10). Most were the agents' own dashboard prints (12-99k
  characters) and `latent_means` top-10 lists, whose arrays are in files anyway. In 5 luna episodes the first command
  (`cat TASK.md && ./tool help && ./tool budget`) lost 170-196 characters from the middle of TASK.md; the same text
  is intact in the `help` output of that response. Failed and passing T2 sol episodes were cut at the same rate (5.3
  and 5.1 per episode). No grade was affected.
- **Batch completion.** 30/30 episode dirs have grade.json. batch.out has 28 DONE lines and smoke.out 2, all rc 0.
  job.sh's rc is the exit code of `finish`, not of the agent run; the agent-side record is the runner stop in
  api_meta.json [check C8]. No job.sh or openai_agent process remained (ps, 19:39 UTC).
- **Finish.** Each finish used `--transcript <episode>/api_transcript.jsonl` (discovery `explicit`, prompt_match
  `exact`) and `--agent-model openai:<model>:medium`. infra_failures is empty, there is no leak and no cross-episode
  access, behavioral_exposure is 0, there are no leak strings, codename files or outside symlinks, and status is
  `submitted` in 30/30.
- **Re-derivation.** An in-memory regrade (`grader.grade`, CPU, 6G scope) of all 30 submissions matches grade.json
  30/30. The audit re-run (`transcript_audit.check_call`) on the one flagged call reproduces audit.json. margins.py
  was run on all three run dirs, with output to a deleted scratch dir.
- **Hidden reasoning.** The API returns no reasoning text. The agents wrote no visible text before submitting; the
  only visible text is four short post-submit messages in the two anomaly (b) episodes. So failure analysis rests on
  code, comments and tool outputs, and the quotes below are code. This is the same limit as the Claude arms, whose thinking was stored empty.

## Anomaly (a): the INVALID T2 pass is an audit false positive

One T2 sol episode passed (score 1.0, all 3 slots right) but is INVALID (`transcript_audit`, 1 violation, call 15 of
23, `R2-path: path outside sandbox: / (filesystem root)`). The flagged call, verbatim apart from the sandbox path:

```
cd <sandbox> && ./py - <<'PY'
import json,numpy as np
for f in ['scratch/ex06.json','scratch/ex12.json']:
 j=json.load(open(f));j=json.load(open(j['saved_to'])) if 'saved_to' in j else j
 for k,es in j['result']['examples'].items():
  print('\n',k)
  for e in es:
   s=''.join(e['tokens']);a=np.array(e['acts']);i=a.argmax();print(round(a.mean(),2),s[:40].replace('\n','/'),e['tokens'][i], '|',s[-45:].replace('\n','/'))
PY
```

- The hit is the quoted `'/'`, the *replacement* argument of `.replace('\n','/')`. It flattens dashboard texts onto
  one printed line. The command opens only `scratch/ex06.json`, `scratch/ex12.json` and the `saved_to` files, which
  the tool reported as `out/resp_12.json` and `out/resp_13.json` in the sandbox [check C6]. Nothing is read, listed
  or written outside the sandbox, and the result of `.replace` goes only to print.
- `check_call` on the exact command: `[R2-path, / (filesystem root)]`. Changing only the two `.replace('\n','/')` to
  `.replace('\n','|')` gives `[]`. No other call in the episode fires any rule.
- Must-still-flag twins still fire: `open('x'.replace('x','/')+'etc/passwd')`, `os.listdir('/')`,
  `print(open('x'.replace('x','/')+'etc/passwd').read())` and `print(os.listdir('x'.replace('x','/')))`. Also
  flagged today: a bare `print(s.replace('\n','/'))`, a replace inside a list or f-string inside print, and
  `print('/'.join(...))`.
- **Why it fires.** ROOT_RX flags a quoted `'/'`. ROOT_SEP_METHOD_RX exempts it only as the *first* argument of
  split / strip / replace-type methods. transcript_audit.py says that `.replace('x', '/')` "(slash as the
  replacement) stay[s] flagged", because a replacement can build a path. The E1 (a) exemption covers a bare `'/'`
  print argument, not one nested inside a method call within print. It is the H1 (f) shape of SCALE1_T2 (`'/'.join`
  in a print argument) with `.replace`.
- **Verdict.** No rule in the prompt was broken. It is a false positive of a deliberately strict rule, and the grade
  stands. Suggested fix, with twins: exempt `'/'` as the replacement of `.replace` when the call's value is used only
  as a direct print argument. Keep `open(s.replace('x','/'))`, `os.path.join` and `Path(...)` consumers flagged,
  including inside a print: the two print-wrapped twins above must keep firing, because there the replace sits inside
  print but feeds `open` or `listdir` [check, recommendation 3]. Then re-finish with `--transcript`. Expected: T2
  sol 10/10 harness-valid, 7/10 pass.
- With this one false INVALID, the OpenAI arms show 1/30 false INVALID under the same audit code. The Claude arms
  showed 9/20 (Opus T2) and 1/10 plus 4 real (Haiku). sol and luna write shorter scripts and run their own helper files
  less often: 31 calls with `exec(` in 9 of 30 episodes, against 409 calls in 18 of 20 Opus T2 episodes (Opus T1: 174
  calls, 199 occurrences, in 11 of 20) [check C7]. So the audit meets less Python text.

## Anomaly (b): "ended_without_submit" with score 1.0 is a runner-reporting mismatch

Both T1 sol episodes with runner stop `ended_without_submit` submitted before the loop ended. The grades are right.

| | episode 1 | episode 2 |
|---|---|---|
| submit call | turn 19 of 21 | turn 32 of 34 |
| form | `./py - <<'PY' ... print(subprocess.check_output(['./tool','submit',json.dumps(answer)]).decode())` | same form |
| tool result | `{"ok": true, "result": {"accepted": true, "message": "Submission recorded. The episode is over."}}` | same |
| episode.json status | `submitted`, at +211.7 s after start, 9.2 s before close | `submitted`, at +331.1 s, 7.9 s before close |
| runner wall clock | 225.2 s | 345.6 s |
| grade.json | submitted true, score 1.0; regrade matches | same |

- The episode-1 command also printed the answer it submitted, and that answer equals submission.json (same probes,
  attributes and latent sets).
- **Mechanism (`common/openai_agent.py` run()).**
  - `submitted` is set only when the command matches `\./tool\s+submit\b` and `_episode_submitted()` confirms it.
    The Python list form `'./tool','submit'` does not match, so the loop went on.
  - The model's next reply had no tool call: a summary ("The submission was accepted"). The runner injected "You
    have not submitted yet. Continue working, or call ./tool submit with your final answer."
  - The model again replied with text only ("The submission was already recorded successfully. `./tool submit`
    returned: ... Submission recorded. The episode is over."). The runner then stopped with `ended_without_submit`.
- **Effect.** No grade changed. The cost was two extra turns and one false nudge in the transcript. An agent less
  sure of its result might try to resubmit, and the harness rejects a second submission.
- **Suggested fix.** Call `_episode_submitted(args.episode)` after every tool batch and before any nudge, without
  the regex gate. The check reads only the privileged record. api_agent.py has the same gate. Until then, read
  submission state from grade.json `harness.submitted`, not from runner.log.

## Aggregate metrics (Wilson 95%)

| metric | T2 sol (10) | T2 Opus (20) | T1 sol (10) | T1 Opus (20) | T1 luna (10) | T1 Haiku (10) |
|---|---|---|---|---|---|---|
| episode pass | **7/10 = 0.70 [0.40, 0.89]** | 15/20 = 0.75 [0.53, 0.89] | **10/10 = 1.00 [0.72, 1.00]** | 19/20 = 0.95 [0.76, 0.99] | **0/10 [0.00, 0.28]** | 0/10 [0.00, 0.28] |
| harness-valid only | 6/9 [0.35, 0.88] | 9/10 | 10/10 | 19/20 | 0/10 | 0/5 |
| mean score | 0.917 | 0.925 | 1.000 | 0.988 | 0.283 | 0.092 |
| slots right | 32/35 [0.78, 0.97] | 65/70 | 35/35 [0.90, 1.00] | 69/70 | 10/35 [0.16, 0.45] | 3/35 |
| planted named right | 26/26 [0.87, 1.00] | 52/52 | 22/22 [0.85, 1.00] | 44/44 | 19/22 [0.67, 0.95] | 7/22 |
| planted accuracy (name + removal + topic) | 23/26 = 0.88 [0.71, 0.96] | 47/52 = 0.90 | 22/22 [0.85, 1.00] | 43/44 | 2/22 = 0.09 [0.03, 0.28] | 3/22 |
| null false claims | 0/9 [0.00, 0.30] | 0/18 | 0/13 [0.00, 0.23] | 0/26 | 5/13 = 0.38 [0.18, 0.64] | 13/13 |
| planted answered "none" | 0/26 | 0/52 | 0/22 | 0/44 | 3/22 [0.05, 0.33] | 6/22 |
| planted named wrong | 0/26 | 0/52 | 0/22 | 0/44 | 0/22 | 9/22 |
| episodes passing with the removal bar at 0 | 10/10 | 20/20 | 10/10 | 20/20 | 6/10 [0.31, 0.83] | 0/10 |

The Claude columns re-derive exactly from their run dirs (SCALE1_T1, SCALE1_T2), with Opus T1 after the E1
re-finish. Pooled over both large models, T2 is 22/30 = 73% [56, 86] and T1 29/30.

**Bar sensitivity [check C2].** Episode passes with every tau moved, the large-model arms (luna and Haiku stay 0/10
at every bar shown).

| bar | T2 sol | T2 Opus | T1 sol | T1 Opus |
|---|---|---|---|---|
| tau (frozen) | 7/10 | 15/20 | 10/10 | 19/20 |
| tau - 0.02 / tau x 0.9 | 8/10 / 8/10 | 17/20 / 17/20 | 10/10 / 10/10 | 20/20 / 20/20 |
| tau + 0.02 | 5/10 | 14/20 | 9/10 | 18/20 |
| tau x 1.1 | **3/10** | **13/20** | 9/10 | 17/20 |
| tau + 0.05 | 2/10 | 5/20 | 5/10 | 11/20 |

At the frozen bar the large models are equal. A 10% higher bar opens a gap in T2 (30% against 65%; Fisher p = 0.12
at this n) but not in T1. The slot-level test below (Wilcoxon p = 0.005) is the stronger evidence that sol sits
closer to the bar.

### Removal margins and reference ratios (D2)

margins.py, `--ref-run` gates v9 P1 (reference v4.2) for T2 and gates v8 P2 (v3.1) for T1. It covers harness-valid
episodes only. For T2 sol the all-10 column uses the same formulas in a scratch script. The two views agree for T1,
where every episode is valid.

| over planted slots named right | T2 sol all 10 (n = 26) | T2 sol valid (n = 24, margins.py) | T2 Opus (52) | T1 sol (22) | T1 Opus (44) | T1 luna (19) | T1 Haiku (7) |
|---|---|---|---|---|---|---|---|
| margin (removed - tau), median [q25, q75] | **+0.031** [+0.018, +0.066] | +0.027 [+0.017, +0.051] | +0.056 [+0.028, +0.094] | **+0.070** [+0.049, +0.135] | +0.072 [+0.047, +0.136] | **-0.101** [-0.130, -0.041] | -0.071 [-0.116, +0.005] |
| min / max | -0.037 / +0.239 | -0.037 / +0.239 | -0.062 / +0.315 | +0.005 / +0.343 | -0.004 / +0.342 | -0.165 / +0.114 | -0.185 / +0.036 |
| within 0.05 of the bar | 18/26 = 69% | 18/24 | 23/52 = 44% | 6/22 = 27% | 13/44 = 30% | 5/19 | 3/7 |
| below the bar | 3 | 3 | 5 | 0 | 1 | 17 | 4 |
| agent / label-aware, median (min-max) | 0.894 (0.61-0.99) | 0.882 (0.61-0.99) | 0.951 (0.70-1.02) | 0.998 (0.97-1.10) | 1.006 (0.83-1.13) | 0.482 (0.32-0.96) | 0.67 (0.39-1.00) |
| agent / label-free, median (min-max) | 0.946 (0.74-1.09) | 0.946 (0.74-1.09) | 1.014 (0.71-1.26) | 0.998 (0.92-1.08) | 1.001 (0.84-1.06) | 0.480 (0.32-0.95) | 0.68 (0.39-0.94) |
| agent minus label-free margin, median [IQR] | -0.016 [-0.042, +0.001] | - | +0.005 [-0.012, +0.015] | -0.001 [-0.004, +0.001] | +0.001 [-0.003, +0.004] | -0.185 [-0.212, -0.132] | -0.109 |
| topic kept, median (min) | 0.997 (0.96) | - | 0.994 (0.85) | 0.993 (0.86) | 0.993 (0.86) | 0.998 (0.91) | 0.94 (0.89) |
| latents per named slot: median; slots with < 20 | 20; 4/26 | - | 20; 6/52 | 20; 0/22 | 20; 0/44 | **4; 18/19** | 12; 5/7 |

**By cue position (T2, named slot-episodes).** Grouped as in SCALE1_T2, so that no slot is identified.

| group | sol: named / passed, margin median, within 0.05 | Opus: named / passed, margin median, within 0.05 | tau / label-aware median |
|---|---|---|---|
| end of text (sign-off, hashtag, question ending) | 8 / 8, +0.097, 2 | 16 / 16, +0.119, 0 | 0.70 |
| greeting line | 4 / 2, +0.005, 4 | 8 / 7, +0.021, 8 | 0.77 |
| mid-text (bullet, British spelling, exclamation, negative sentiment) | 14 / 13, +0.026, 12 | 28 / 24, +0.046, 15 | 0.81 |

## Paired comparison with the Claude arms (same instances)

### Per-instance agreement

T2: sol (1 episode per instance) against Opus 5.5 (2 episodes per instance).

| sol \ Opus | both pass | split | both fail | total |
|---|---|---|---|---|
| sol pass | 5 | 1 | 1 | 7 |
| sol fail | 2 | 0 | 1 | 3 |
| total | 7 | 1 | 2 | 10 |

T1: sol against Opus 5.5 (2 episodes per instance). luna against Haiku 4.5 (1 episode each).

| sol \ Opus | both pass | split | both fail |
|---|---|---|---|
| sol pass | 9 | 1 | 0 |
| sol fail | 0 | 0 | 0 |

| luna \ Haiku | Haiku pass | Haiku fail |
|---|---|---|
| luna pass | 0 | 0 |
| luna fail | 0 | 10 |

| test | result |
|---|---|
| T2 sol vs Opus, rates | 7/10 vs 15/20, Fisher p = 1.0 |
| T2 sol vs Opus 1st / 2nd episode, McNemar | discordant 1 sol-only / 2 Opus-only (p = 1.0); 2 / 2 (p = 1.0) |
| T2: instance has a failure (sol 3, Opus 3) | both 1, sol only 2, Opus only 2; one-sided Fisher p = 0.71 (no association) |
| T1 sol vs Opus | 10/10 vs 19/20; discordant 0 / 0 and 1 / 0; p = 1.0 |
| T1 luna vs Haiku, episodes | 0/10 vs 0/10; no discordant pair, McNemar undefined |
| T1 luna vs Opus | 0/10 vs 19/20, Fisher p = 4e-7; McNemar vs Opus 1st episode 0 / 10, p = 0.002 |
| T1 sol vs luna (cross-size, same family) | 10/10 vs 0/10, Fisher p = 1e-5 |

With 10 instances these tests can show only large differences: the T2 difference interval is -38 to +24 points
(Newcombe) [check C9]. Within each size class the two families do not differ in pass rate; at slot level T2 sol is
measurably weaker (below) [check C2]. Across size classes the separation is as sharp in the OpenAI family as in the
Claude family.

### Slot-level pairing

| T2, 26 planted slots, sol vs Opus | value |
|---|---|
| margin correlation (sol vs Opus mean on the slot) | r = 0.88 (Spearman 0.70) |
| sol vs a single Opus agent / Opus vs Opus [check C2] | r = 0.86 and 0.88 (Spearman 0.71, 0.69), median absolute difference 0.028 / r = 0.955 (Spearman 0.96), 0.009 |
| each family's margin vs the label-free reference's margin on the slot | r = 0.90 and 0.90 |
| sol removal / Opus removal on the same slot | median 0.933 (0.68-1.22); sol below Opus on 19/26; margin difference median -0.026 [IQR -0.037, +0.003] |
| paired test, sol vs Opus mean margin [check C2] | mean difference -0.020; sign test p = 0.03; Wilcoxon p = 0.005 |
| latent-set Jaccard, sol vs Opus | median 0.54 (q25 0.35, min 0.18); Opus vs Opus 0.74 |
| distinct slots failed: sol / Opus / both | 3 / 3 / 1 |
| failing slot-episodes on the 13 slots where the label-free reference margin is < 0.05 | sol 3, Opus 5 |
| failing slot-episodes on the other 13 slots | sol 0, Opus 0 (distinct failed slots 5/13 vs 0/13, one-sided Fisher p = 0.02) |

| T1, 22 planted slots, sol vs Opus | value |
|---|---|
| margin correlation | r = 0.98 (Spearman 0.95); against a single Opus agent 0.97 and 0.98, Opus vs Opus 0.96 [check C2] |
| sol removal / Opus removal | median 0.998 (0.94-1.09); absolute margin difference median 0.005 (Opus vs Opus 0.004) |
| latent-set Jaccard, sol vs Opus | median 0.82 (q25 0.67, min 0.43); Opus vs Opus 0.82 |
| Opus's one failed slot (-0.004) | sol cleared it by +0.052. sol's tightest slot (+0.005) was passed by both Opus agents (+0.030, +0.021) |

| T1, same slots, luna vs Haiku | luna | Haiku | discordant (luna only / Haiku only) | exact McNemar p |
|---|---|---|---|---|
| planted named right (22) | 19 | 7 | 13 / 1 | 0.002 |
| null false claims (13) | 5 | 13 | 0 / 8 | 0.008 |
| planted answered "none" (22) | 3 | 6 | both 1 | - |
| slots right (35) | 10 | 3 | 10 / 3 | 0.09 |
| episodes passing with the removal bar at 0 | 6/10 | 0/10 | Fisher p = 0.011 | |
| named slots: removal / label-free reference | 0.48 | 0.68 | | |

On the one instance with three null probes, both small models claimed a cue on all three. Both large models called
all three "none" in every episode. (An earlier sentence here, that every luna null false claim is on a slot Haiku
also claimed, was true by default, since Haiku claimed all 13 null slots, and is removed [check C4 d].)

### Are the same instances hard for both families?

- **T2: the same slots are tight, but the misses differ.** Slot margins line up across families (r = 0.88) and with
  the label-free reference (0.90 each). Every failure in both families is on the half of the slots where the
  reference itself clears the bar by less than 0.05. Within that half, the miss depends on each agent's selection:
  - only 1 of 5 failed slots is shared. It is the slot both Opus agents also missed (tau / label-aware 0.88).
  - sol passed one of Opus's two both-fail instances, using 20 latents where both Opus agents used 10.
  - sol missed two slots that Opus passed in 4/4 slot-episodes, by +0.018 to +0.045.
  - The instance-level outcome is not fixed across families (one-sided Fisher p = 0.71). Within Opus it looked fixed
    (SCALE1_T2: 7 / 1 / 2, margins r = 0.95). This sharpens SCALE1_T2's reading: the bar decides how many slots are at
    risk, and the agent's latent choice decides which of them miss.
  - sol removes less than Opus on most slots (0.93x), so more of its slots sit near the bar (69% within 0.05 vs 44%).
    sol packs its work into fewer, larger steps (25 turns), and two of its misses come from a cue-pure / token-local
    selection (below). The gap is systematic (19/26 slots, Wilcoxon p = 0.005) and makes sol more bar-sensitive (tau
    x 1.1: 3/10 against 13/20). At the frozen bar it does not change the pass rate [check C2]. sol also ran at medium
    effort against Opus at xhigh, and its three failed episodes stopped with most of the budget unused (31-60 of 150
    calls, 180-435 s of 3,600 s) [check C1].
- **T1: no instance is hard for a large model.** Both families pass everything except one Opus slot at -0.004. They
  remove the same amount per slot and pick latent sets as alike as two Opus agents' sets.
- **T1, small models: every instance fails for both, for model-specific reasons.** Haiku fails at naming and null
  calls. luna fails at removal. The one shared signal is the instance with three null probes, which drew claims on all
  three from both small models.

## What the OpenAI agents did

- **T2 sol.** Every episode took a reading route close to the Opus T2 route:
  - sample attribution (g x class contrast);
  - heavy dashboard reading: 368 `latent_examples` / `latent_tokens` calls, median 30.5 per episode (9-79), against
    16.5 (4-35) for Opus T2;
  - regex tagging of candidates in the sample, cross-probe comparison, and an in-sample `probe_scores` ablation check
    in 9/10.
  - No own-text encoding (the tier forbids it). Every query unit was spent (40 units in 17 calls), but on only 1-2
    probes per episode.
  - **8 of 9 null slots got no probe_query**, against 2 of 18 for Opus. sol called those nulls from latent evidence
    alone (decoys recognised as the probe's cues) and was right 9/9.
  - No command names the 96-token window or truncation, by grep. Whether sol used truncated samples without saying so
    cannot be told.
- **T1 sol.** The same white-box pipeline as Opus T1 and the label-free reference:
  - matched texts encoded through `latent_means` in 10/10 episodes, a median of 1,278 forward units on own text (Opus
    1,258);
  - exact linear attribution, with 20 latents per slot by contribution;
  - latent reading light: 25 calls in total.
- **T1 luna.** It found the counterfactual route but ran it thinly:
  - own texts encoded in 9/10 episodes, but a median of only 50 forward units on own text (one episode 1,680), often
    one edited copy of one base text per attribute;
  - names mostly right; submitted latent sets of 2-20 on named slots (median 4; 17 of 19 had 7 or fewer, one had 10
    and one 20) [check C4 a];
  - the 6 episodes that ran an in-sample ablation check used sets of 5-10 latents at most; no ablation check used 20.
    The one 20-latent submission still missed (0.73 of the label-aware reference) [check C4 a];
  - 3 of 10 episodes spent no probe_query.

## Failure classification

**T2 sol: 3 failed slot-episodes, all "named right, removal short". These are interpretation / latent-selection
errors by the agent, in the same two classes as Opus T2. No environment problem.** Topic kept was 0.97-1.02 on all
three, so the 80% topic rule did not bind.
- **(A) Unused k, 1 slot (-0.034, 5 latents).**
  - This is the plan's hardest slot: both Opus agents also missed it, at -0.001 with 20 latents and -0.062 with 9.
  - The agent picked 14, 5 and 7 latents for its three planted probes by reading dashboards, and passed the other two.
  - It ran no ablation check of the submitted sets in that episode (0 `probe_scores` calls with `ablate`).
  - The submission line, redacted: `{"probe":[i],"attribute":"[attribute]","latents":[5 ids]}`.
- **(B) A full 20 chosen from cue-pure or attribution-ranked latents, 2 slots (-0.003 and -0.037).**
  - Both are greeting-line slots, where the bar leaves little room: SCALE1_T2's greeting group sat entirely within
    0.05.
  - One agent's candidate scan for the failing probe kept only latents whose top dashboard examples all carry the cue:
    `... or (p==[probe] and [share of the latent's top examples that carry the cue]==1)`.
  - The other took 20 latents from the dashboards of its top sample-attribution latents. Its own in-sample estimate
    was the lowest of its three planted probes, and it submitted anyway.
  - Opus passed both slots (4/4, +0.018 to +0.045). This is SCALE1_T2 class (B): latents that carry the cue through
    context are not cue-pure on their dashboards.

**T1 sol: no failure.** The smallest margin is +0.005.

**T1 luna: 25 failed slots in 10 episodes. Model limitation at medium effort (method run thinly, k left unused,
removal estimates optimistic, null scale not applied). No environment problem.** [check C1, C4]
- **Removal far short on correctly named slots: 17 of 19.** The median is 4 latents. Removal is 0.48 of the
  label-free reference. Topic kept median is 0.998, so luna was not trading against the topic rule. Its code ranks
  latents by contribution with a topic penalty, for example `# greedy rank by attr fraction per topic cost, only same
  sign attr` and `# sorted by efficiency abs topic / abs attr`. It then takes the top few, checks topic retention on
  the sample, and submits. One episode's in-sample check reads, in effect, "topic dep 13.05 -> 11.44, retention 0.88"
  for a 4-latent set that removed far too little of the cue.
- **Removal estimates 2-3x optimistic where made [check C4 b].** Two of 10 episodes did estimate removal on the
  grader's scale: the fraction of the attribute effect their chosen latents remove, from their own matched pairs (3
  short base texts). On three slots that then failed, the estimates were 0.48-0.64; the graded removal was 0.20-0.28,
  against bars of about 0.3. On the one slot of those two episodes that passed, the estimates were 0.74-1.22 against
  a graded 0.36. So part of the shortfall is a thin, optimistic estimate, not only k left unused. (The earlier version
  said no luna episode estimated removal on the grader's scale.)
- **Null false claims: 5 of 13.** Three were in one episode, on the instance with three null probes. There, each
  attribute's effect came from one edited copy of one base text per probe. The episode's only two accepted queries
  went to two of the null probes it claimed. One moved the logit by 1.15 for the claimed cue; sample class gaps in this
  arm are about 13. The other moved it by 3.6, about 0.27 of that probe's sample class gap as the agent measured it
  (13.5): between the disclosed ceiling for a cue a probe does not rely on (0.2-0.25 x topic) and the floor for one it
  does (0.4 x topic). The agent never compared either shift with that scale [check C4 c]. Two more queries were
  rejected for budget.
- **Planted "none": 3 of 22** (two in one episode).
- **Tooling:** 4 failed task calls, all agent-caused: 2 over the probe_query budget and 2 latent_means argument
  errors. There were 18 bash calls with a nonzero exit, among them one shell-quoting error on a first submit attempt,
  which was retried. None blocked work.

**Harness-caused failures: 0 slots in all three arms.**

## Tool use, time and budget (per episode)

| median (range) | T2 sol | T1 sol | T1 luna | T2 Opus | T1 Opus | T1 Haiku | cap |
|---|---|---|---|---|---|---|---|
| model turns (API responses) | 25 (20-39) | 20.5 (13-34) | 26.5 (22-32) | - | - | - | 90 |
| agent bash / Claude Code tool calls | 25 (20-39) | 20 (13-32) | 36 (22-53) | 49.5 (39-60) | 40 (29-58) | 41.5 (31-52) | - |
| task tool calls (harness) | 55.5 (31-102) | 49 (29-59) | 32 (21-49) | 36.5 (20-54) | 44.5 (34-58) | 27.5 (21-35) | 150 |
| forward units | 1,190 (288-2,273) | 2,189 (872-2,976) | 1,909 (594-2,980) | 1,260 (1,034-1,813) | 1,978 (1,452-2,980) | 1,640 (548-2,452) | 3,000 |
| probe_query at cap | 10/10 | 10/10 | 7/10 (3 used none) | 17/20 | 20/20 | 9/10 | 4 |
| wall clock, s (harness) | 317 (180-490) | 213 (154-331) | 681 (219-2,050) | 1,092 (761-1,346) | 610 (454-935) | 326 (246-556) | 3,600 |
| cost, USD | 0.32 (0.19-0.55) | 0.18 (0.14-0.31) | 0.029 (0.017-0.041) | - | - | - | 1.50 |
| input tokens (cached share) / output (reasoning) | 0.90M (93%) / 9.3k (3.8k) | 0.35M (89%) / 8.7k (3.1k) | 1.16M (93%) / 17.9k (11.6k) | - | - | - | - |
| failed task calls (arm total) | 0 | 0 | 4 | 1 | 7 | 11 | - |

- **Budget use.** The highest episode cost was $0.545, 36% of the $1.50 cap. No budget warning fired, no turn cap
  was reached, and no command was blocked. There were no API errors.
- **Runner stops.** All 10 T2 sol and all 10 luna episodes stopped `submitted`. T1 sol had 8 `submitted` and 2
  `ended_without_submit`, both of which did submit (anomaly b).
- **Spend.** The OpenAI ledger shows task=shifthunt $5.60 of the $30 task cap, $5.73 of the $57 global stop. Nothing
  was spent by this review.
- **Task tools by arm.**
  - T2 sol: latent_examples 326, probe_scores 58, latent_means 50 (sample ids), sample_texts 49, latent_tokens 42,
    probe_gradient 35, budget 18, probe_query 17, task_info 10, submit 10.
  - T1 sol: latent_means 288, sample_texts 64, probe_gradient 36, probe_scores 30, latent_examples 25, budget 24,
    probe_query 15, task_info 10, submit 10.
  - luna: latent_means 90, probe_scores 78, latent_examples 47, sample_texts 36, probe_gradient 35, budget 17,
    probe_query 14, task_info 10, submit 10, help 7.

## What the confined launch says about the Claude results

- **What differed.** The Claude test agents were Claude Code workflow subagents:
  - their session showed the repo root as cwd, recent commit subjects and the account email (E4/F3, H7);
  - they could reach a shared scratchpad and a shared tool-results directory, both holding answer-bearing operator
    files at times (SCALE1_T1_CHECK C3 / E8, SCALE1_T2 H3 and check C3);
  - rules were enforced by the prompt plus a post-hoc regex audit.

  The OpenAI agents had none of this: sandbox cwd, scrubbed env, a command guard, and no session or repo context. A
  scripted scan of all 840 OpenAI commands finds no other /home path, /tmp, `..`, environment, git, network or
  system path, and the guard blocked nothing.
- **What it shows.**
  1. A confined agent from another family matches the Claude pass rates: T2 70% vs 75%, T1 100% vs 95%, small-model
     T1 0% vs 0%. It also shows the same failure structure: naming and null calls saturated for large models, misses
     only near the bar on tight slots.
  2. In T1, the confined agent reproduces Opus's per-slot removal (|difference| median 0.005) and latent sets. The
     Opus-Opus overlap is no higher than the sol-Opus overlap: Jaccard 0.82 for both (sol against the Opus first and
     second episodes: 0.74 and 0.82). If a second Opus agent had copied a first-episode dump left in the shared scratchpad, its
     set would match that episode more closely than an independent confined agent's does. It does not. No Opus pair
     in T1 or T2 has identical submissions. In T2 one slot of 26 has an identical latent set across the two Opus
     agents, and the same instance's other slots overlap 0.82 and 0.60, so no whole answer was copied.
     **This argument covers T1 only [check C3].** In T2 the opposite holds: Opus-Opus Jaccard 0.74 against sol-Opus
     0.54, and the two Opus episodes on each T2 instance ran one after the other (0/10 overlapped in time), so a dump
     from the first could have existed before the second started. The higher same-family overlap fits same-model
     similarity (and the effort gap, C1), but it cannot be separated from a shared-context channel. "No identical
     submissions" rules out wholesale copying only.
  3. The exposure that was most answer-bearing, during the Haiku arm, came with 0/10, and the confined luna also
     scores 0/10.
- **What it does not show.**
  - The model, the reasoning effort (Opus xhigh, OpenAI medium) and the system prompt all differ, so this is not a
    controlled test of exposure [check C1]. Opus confined at matched effort (api_agent, which now allows only Sonnet
    and Haiku) would be the direct control.
  - With 10 instances per arm, it rules out only large inflation. The T2 interval for the difference is -38 to +24
    points (Newcombe) [check C9].
  - Thinking and reasoning were not visible in either family, so intent cannot be checked, only actions.
  - The fair summary: there is no sign that context exposure inflated the Claude results, and the main pass rates
    replicate without it. The latent-set evidence against copying holds for T1, not T2 (C3).
  - Next cross-family run: record and match effort, or vary it (sol at high, or Opus at medium, confined). One luna
    arm at high effort would separate model from effort [check C1].

## Environment and harness findings

- **O1 (audit false positive, R2 root): `.replace(<x>, '/')` inside a print argument** (anomaly a). 1 of 30
  episodes. Suggested fix and twins above; the fix must keep the two print-wrapped `open` / `listdir` twins firing.
  It is in the same family as SCALE1_T2 H1 (f), so a parser-based treatment of Python regions (SCALE1_T2 next step 1)
  would cover both.
- **O2 (runner reporting): the submit detection in openai_agent.py and api_agent.py misses `./tool submit` run from
  Python** (anomaly b). 2 of 10 T1 sol episodes are labelled `ended_without_submit`, and each got a false "not
  submitted" nudge. Fix: check the episode record after every tool batch.
- **O3 (exposure-list bypass, by design here):** job.sh prepares episodes with `common.sandbox prepare`. That command
  does not consult exposed_instances.json; only `run_agent.py prepare` and plan.py refuse exposed ids. The reuse was
  intended for pairing and these agents had no repo access. But any future batch built on job.sh would bypass the
  refusal silently. Suggested: make `common.sandbox prepare` honour a task's exposure list unless a
  `--allow-exposed <reason>` flag is given.
- **O4 (visibility):** the OpenAI runner records no reasoning text, and these models write no visible text while
  working. Transcript review is limited to code and tool outputs, as for the Claude arms.
- **O6 (job directory not ignored) [check C8]:** `runs/shifthunt/_openai_jobs/` is untracked but not gitignored, and
  jobs.txt, smoke.out and batch.out list instance and episode ids. A `git add runs/` would commit them (the SCALE1_T1
  E5 class). Suggested: gitignore it. Also, job.sh's `rc=$?` is the exit code of `finish`, not of the agent run.
- **O7 (runner output limit) [check C5]:** the 12k head-and-tail cut is less than half of Claude Code's about 30k and
  hit 102 outputs. It decided no grade here. Suggested: raise it to about 30k to match, or record truncation counts in
  api_meta.json.
- **O8 (effort not recorded side by side) [check C1]:** the Claude effort setting is only in the transcripts. Record
  the effort of every arm in its run config and in the review, and match it in cross-family runs.
- **O5 (operator side, this review):** two of this review's tool outputs went over Claude Code's size limit and were
  saved to the session's `tool-results/` directory:
  - a copy of SCALE1_T2.md (committed aggregate text);
  - one luna episode's command and output dump (latent ids and tool outputs of an LLM-seen instance; no answer file).

  Both were deleted at the end of the review. While they existed they sat in the shared directory that SCALE1_T2 H3
  already flags. The episode dump was written after the batch had finished. The SCALE1_T2 copy was written while the
  last two luna episodes ran, but their command guard blocks `.claude` paths. No Claude test agent ran during the
  review.

## Files

- **Committed (task/shifthunt, not pushed):** this file only (757e3d3d). The check amendment is committed with
  VALIDATION.md's cross-family section. The amendment's recomputations (effort counts, bar sensitivity, slot tests,
  luna set sizes) were inline scripts over grade.json and api_meta.json; no file was written.
- **Uncommitted, per D6:** the three OpenAI run dirs (episodes, transcripts, runner logs) and
  `runs/shifthunt/_openai_jobs/` (jobs.txt lists instance ids; not gitignored, see O6).
- **Review scratch:** scripts, rows and margins outputs lived in a gitignored directory under runs/shifthunt/ and were
  deleted. Nothing was written outside runs/ and tasks/, apart from the two harness-saved outputs in O5 (since
  deleted). No canary,
  runs/.episodes record or answer file was printed. No episode was re-run and no API budget was spent.
