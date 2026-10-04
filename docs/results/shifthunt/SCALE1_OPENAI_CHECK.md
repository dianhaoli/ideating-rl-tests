# ShiftHunt cross-family check: adversarial check of SCALE1_OPENAI.md (2026-10-03)

This checks the review in `SCALE1_OPENAI.md` (commit 757e3d3d) of `runs/shifthunt/20261003-openai_T2_sol`,
`20261003-openai_T1_sol` and `20261003-openai_T1_luna`. Those are 30 OpenAI episodes, paired with the Claude arms
`20261003-scale1_T2_opus55`, `20261003-scale1_T1_opus55` and `20261003-scale1_T1_haiku45`. This file gives aggregates
only (D6, exposed_instances.json rule). It names no instance, episode or latent id, and it does not say which attribute
belongs to which slot.

**Method.**
- **Re-derivation.** I re-derived every number with my own scripts. The sources were the six run dirs (grade.json,
  episode.json, submission.json, api_meta.json, the last line of runner.log, tool_log.jsonl, api_transcript.jsonl and
  audit.json), the instance answers, attempt 0 of the label-free reference rows (gates v9 P1 v4.2 for T2, gates v8 P2
  v3.1 for T1) and the OpenAI ledger.
- **Regrade and audit.** I regraded all 30 OpenAI submissions in memory (`grader.grade`, CPU, 6G scope) and re-ran
  `transcript_audit.audit` on all 30 transcripts.
- **Episodes read.** I re-read the 3 failed T2 sol episodes and all 10 luna episodes (all failed). I also read 3 passes
  drawn with `random.Random(20261002).sample(sorted(passing OpenAI episode ids), 3)`: 1 T1 sol and 2 T2 sol.
- **Claude transcripts.** These were read only for metadata (model, effort, thinking time) and command counts.
- **Budget.** No API budget was spent and no episode was re-run.

## Verdict

**SCALE1_OPENAI.md is accurate, but it under-reports how unequal the comparison was.**
- **The numbers re-derive.** Every headline number matches, apart from rounding. The regrade matches grade.json
  30/30, and the audit re-run reproduces audit.json 30/30.
- **Both anomaly resolutions are right.** I verified each one independently.
- **Classification holds.** The failure classes and "no environment problem" hold.
- **The cross-family reading needs three qualifications** the file does not make:
  - the two families ran at different reasoning effort;
  - at slot level the T2 agreement is weaker than within Opus, and the gap is statistically clear;
  - the evidence against copying covers T1 only.
- **No grade, count or CI changes.**

Corrections and additions:
1. **C1. Effort asymmetry, not reported (affects interpretation).**
   - **The setting.** Every assistant row of the Opus T1 and T2 transcripts records Claude Code effort `xhigh` (3,300
     rows). Recorded thinking time averages 809 s per T2 episode and 386 s per T1 episode. The OpenAI agents ran at
     `medium`: median reasoning tokens per episode were 3.8k (T2 sol), 3.1k (T1 sol) and 11.6k (luna). Haiku has no
     effort setting.
   - **What the file says.** SCALE1_OPENAI.md states the OpenAI effort but never the Claude one.
   - **So three things differ.** The comparison changes model, effort and system prompt at once.
   - **Effect on the conclusions.** "sol matches Opus" means sol at medium effort reaches Opus-at-xhigh pass rates.
     "luna fails at removal for model reasons" holds at medium effort only, because no higher effort was tried.
2. **C2. The T2 cross-family match holds at pass-rate level only.** At slot level, sol is measurably weaker and more
   sensitive to the bar.
   - **Removal.** sol's margin is below the mean of the two Opus agents on 19/26 planted slots (sign test p = 0.03).
     The mean difference is -0.020 (paired Wilcoxon p = 0.005).
   - **Agreement.** Against a single Opus agent, sol's margins correlate r = 0.86 and 0.88 (Spearman 0.71 and 0.69),
     with a median absolute difference of 0.028. The two Opus agents agree at r = 0.955 (Spearman 0.96, median
     difference 0.009). The file's r = 0.88 is against the Opus mean, which is less noisy.
   - **Bar sensitivity.**

     | bar | T2 sol | T2 Opus |
     |---|---|---|
     | tau + 0.02 | 5/10 | 14/20 |
     | tau x 1.1 | 3/10 | 13/20 |
     | tau - 0.02 | 8/10 | 17/20 |
     | tau x 0.9 | 8/10 | 17/20 |

     At the frozen bar the rates are equal (7/10 against 15/20). A 10% higher bar would separate the families (30%
     against 65%).
   - **T1.** Here sol-Opus agreement equals Opus-Opus agreement (r 0.97 and 0.98 against 0.96; median absolute
     difference 0.005 against 0.004), so the T1 reading stands.
   - **How it should read.** The file calls the T2 gap "small" and "systematic" but gives no test. It should report the
     test and the bar sensitivity, and keep "the pattern repeats across families" as a pass-rate statement.
3. **C3. The argument against copying covers T1 only.**
   - **T1.** In "What it shows" (2), Opus-Opus latent overlap is no higher than sol-Opus (Jaccard 0.82 for both).
   - **T2.** Here the opposite holds: Opus-Opus 0.74 against sol-Opus 0.54. The two Opus episodes on each T2 instance
     also ran one after the other (0/10 overlapped in time), so a dump from the first could have existed before the
     second started.
   - **What T2 can and cannot show.** Its higher same-family overlap fits same-model similarity (and C1). It cannot
     separate that from a shared-context channel. "No identical submissions" (true: 0 in T1, 0 in T2, one identical
     slot set in T2) rules out wholesale copying only.
   - **Conclusion.** The fair summary still holds as "no sign of inflation". The direct control is still Opus run
     confined, at matched effort.
4. **C4. luna details.**
   - **(a) Latent-set sizes.** "Latent sets of 2-7 (median 4)" is wrong. Submitted sets on named slots run from 2 to 20
     latents (median 4). One slot had 20 latents and one had 10; 17/19 had 7 or fewer. The table row (18/19 below 20)
     is right. "No luna episode ever tried 20" is true only of the in-sample ablation checks: one episode submitted
     20 latents, and that slot still missed (0.73 of the label-aware reference).
   - **(b) Removal estimates.** "No luna episode estimated removal on the grader's scale" is wrong. Two of 10 did. Each
     computed the fraction of the attribute effect that its chosen latents remove, from its own matched pairs (3 short
     base texts). On three slots that then failed, those estimates were 0.48-0.64. The graded removal was 0.20-0.28,
     against bars of about 0.3. On the one slot of the two that passed, the estimates were 0.74-1.22 against a graded
     0.36. So part of luna's removal failure is a thin estimate that ran 2-3x optimistic, not only k left unused. It
     is still a model limitation.
   - **(c) The three-null instance.** The file cites the query that moved the logit by 1.15. The episode's other
     accepted query, on another claimed null probe, moved the logit by 3.6. That is about 0.27 of that probe's sample
     class gap as the agent measured it (13.5), which falls between the disclosed ceiling (0.2-0.25) and floor (0.4).
     The agent never compared either shift with the scale.
   - **(d) A vacuous sentence.** "Every luna null false claim is on a null slot that Haiku also claimed" is true by
     default, since Haiku claimed all 13 null slots.
5. **C5. Truncation by the runner was not quantified. It decided no grade.**
   - **How often.** The OpenAI runner keeps the first and last 6,000 characters of any output over 12,000. That hit
     102 outputs in 25 of 30 episodes: 52 in T2 sol (10/10), 8 in T1 sol (5/10) and 42 in luna (10/10). Claude Code's
     Bash shows up to about 30,000 characters. Most cuts were the agents' own dashboard prints (12-99k characters)
     and the top-10 lists of `latent_means` (about 20k), whose arrays are in files anyway.
   - **The one cut of task documentation.** In 5 luna episodes the first command (`cat TASK.md && ./tool help && ./tool
     budget`) printed 12,170-12,196 characters. So 170-196 characters from the middle of TASK.md were dropped: the
     `sample_texts` doc and the start of the `latent_means` heading. The same text appears intact in the `help`
     output of that same response. Nothing was lost.
   - **Failed against passing.** Failed and passing T2 sol episodes had the same truncation rate (5.3 and 5.1 per
     episode).
   - **What the file should say.** It should give these counts next to "truncated at about 12k".
6. **C6. Anomaly (a): one wrong detail.** The flagged command's `saved_to` files were `out/resp_12.json` and
   `out/resp_13.json`. I read them from the two pointer files, which are still in the sandbox. They were not
   `resp_1`/`resp_6`, which belong to earlier calls. Both are inside the sandbox, so the conclusion is unchanged.
7. **C7. Exec comparison.** "199 in 11 of 20 Opus T1 episodes" counts `exec(` occurrences: 199 occurrences in 174
   calls. The OpenAI count is in calls (31 calls, 9 episodes). The false-INVALID comparison in the same paragraph is
   with Opus T2, where `exec(` appears in 409 calls in 18 of 20 episodes. That makes the point stronger.
8. **C8. A path that could be committed by mistake (new).** `runs/shifthunt/_openai_jobs/` is untracked but not
   gitignored, and its jobs.txt, smoke.out and batch.out list instance and episode ids. A `git add runs/` would
   commit them (the SCALE1_T1 E5 class). Also, job.sh's `rc=$?` is the exit code of `finish`, not of the agent run,
   so "all rc 0" covers the finish step only. The runner stops (api_meta) are the agent-side record.
9. **C9. Minor.**
   - The T2 difference interval is -38 to +24 points (Newcombe), not "about +-35".
   - The runner env also sets PYTHONDONTWRITEBYTECODE.

## Re-derivation of SCALE1_OPENAI.md numbers

These all match:
- **Outcomes.**

  | arm | pass | valid-only |
  |---|---|---|
  | T2 sol | 7/10 [0.40, 0.89] | 6/9 [0.35, 0.88] |
  | T1 sol | 10/10 [0.72, 1.00] | 10/10 |
  | T1 luna | 0/10 [0.00, 0.28] | 0/10 |
  | T2 Opus | 15/20 | 9/10 |
  | T1 Opus | 19/20 | 19/20 |
  | T1 Haiku | 0/10 | 0/5 |

  Pooled over both large models, T2 is 22/30 [0.56, 0.86].
- **Slot metrics** (the six arms in the order of the file's table):
  - mean score 0.917 / 0.925 / 1.000 / 0.988 / 0.283 / 0.092;
  - slots right 32/35, 65/70, 35/35, 69/70, 10/35 and 3/35;
  - planted named right 26/26, 52/52, 22/22, 44/44, 19/22 and 7/22;
  - planted correct 23/26, 47/52, 22/22, 43/44, 2/22 and 3/22;
  - null false claims 0/9, 0/18, 0/13, 0/26, 5/13 and 13/13;
  - planted "none" 3/22 for luna and 6/22 for Haiku, 0 for the rest; planted named wrong 9/22 for Haiku, 0 for the
    rest;
  - pass with the removal bar at 0: 10/10, 20/20, 10/10, 20/20, 6/10 and 0/10.
- **Margins and ratios.** Every cell of the D2 table matches, including the T2 sol valid-only column (+0.027, 18/24
  within 0.05, 3 below). The by-cue-position table also matches.
- **Paired tables and tests.**
  - The T2 table is 5/1/1 and 2/0/1. McNemar gives 1 / 2 and 2 / 2 discordant pairs (p = 1.0); Fisher p = 1.0. The
    instance-has-a-failure test gives 1 / 2 / 2 and one-sided p = 0.71.
  - T1 sol against Opus: 9 instances both pass and 1 split. luna against Opus: Fisher p = 3.7e-7; McNemar against the
    first Opus agent, 0 / 10 discordant, p = 0.002. sol against luna: Fisher p = 1.1e-5.
- **Slot pairing, T2.**
  - Margins correlate r = 0.882 (Spearman 0.696). Each family against the label-free reference: 0.903 and 0.902.
  - sol / Opus removal: median 0.933 (0.68-1.22); sol is below on 19/26 slots. The margin difference is -0.026
    [-0.037, +0.003].
  - Jaccard, pooled over 52 pairs: median 0.54 (q25 0.35, min 0.18); Opus-Opus 0.74.
  - Failed slots: 3 (sol) / 3 (Opus) / 1 (both). Thirteen slots are tight. All 3 + 5 failing slot-episodes are on
    them (one-sided Fisher p = 0.0196).
  - The narrative per failed slot holds: sol passed the 10-latent pair's slot with 20 latents, and sol's two
    Opus-passed misses came where Opus had +0.018 to +0.045.
- **Slot pairing, T1.**
  - Margins correlate r = 0.984 (Spearman 0.951). sol / Opus removal: 0.998 (0.94-1.09); median absolute
    difference 0.005.
  - Jaccard: 0.82 (q25 0.67, min 0.43) for both sol-Opus and Opus-Opus; per Opus replicate, 0.74 and 0.82.
  - Opus's failed slot (-0.004) got +0.052 from sol. sol's tightest slot (+0.005) got +0.030 and +0.021 from Opus.
- **luna against Haiku.**
  - Named: 19 / 7 (discordant 13 / 1, p = 0.0018). Null false claims: 5 / 13 (0 / 8, p = 0.0078). Planted "none": 3 / 6
    (both 1). Slots right: 10 / 3 (10 / 3, p = 0.092). Bar at 0: Fisher p = 0.011.
  - On the three-null instance, both small models claimed 3/3. sol and both Opus agents claimed 0/3.
- **Tools, time and budget.**
  - Every cell matches: turns, bash calls, task calls, forward units, probe_query at cap, wall clock, cost and tokens.
    The task-tool counts by arm also match.
  - Stops were 10 / 8 + 2 / 10. There were no budget warnings and no blocked commands. All 746 responses had status
    `completed` (none was cut by the 16k output cap). The only nudges were in the two anomaly (b) episodes, and the
    only visible text was their 4 post-submit messages.
  - The ledger gives task shifthunt $5.602, equal to the sum of the 30 api_meta values (per episode 30/30), and $5.729
    globally. The maximum episode cost was $0.545 (36% of $1.50).
- **Behaviour.**
  - T2 sol: 368 reading calls, median 30.5 (9-79) per episode, against 16.5 (4-35) for Opus. In-sample ablation checks
    in 9/10 episodes. 40 query units in 17 calls, on 1-2 probes per episode. 8/9 null slots got no query (Opus 2/18).
    No own-text encoding, and no command uses the 96-token window.
  - Own-text forward units, median: T1 sol 1,278 (10/10 episodes); luna 50 (9/10, maximum 1,680).
  - luna: ablation checks in 6 episodes, with sets of at most 5-10 latents. 3/10 episodes made no query. 4 failed
    task calls (2 for budget, 2 argument errors) and 18 nonzero bash exits. The quoted comments and the "retention
    0.88" line are in the transcripts.
- **Provenance.**
  - git sha 955ea331 (dirty) and template ed473483 in 30/30. Masking digits, TASK.md is byte-identical to the Opus T2
    and Haiku arms, and agent_prompt.txt is identical to the Claude arms' apart from the sandbox path. `git diff
    fd91d8b2 955ea331` is empty for the files listed.
  - All 30 finishes: discovery `explicit`, prompt_match `exact`, status `submitted`. No infra failures, leaks,
    cross-episode access, leak strings, codename files or outside symlinks.
- **Confinement scan.** 840 commands. There is no /home path other than the episode's own sandbox, no /tmp, `..`,
  environment, git or network use. The only system path is `/dev/null` in 4 redirects, and the 2 `..` hits are regex
  text. The guard blocked nothing.
- **Commit and exposure.** Commit 757e3d3d adds only SCALE1_OPENAI.md. It contains no instance, episode or latent id.
  All 20 instances were already in exposed_instances.json.

## Anomalies, verified independently

- **(a) Audit false positive.**
  - **Reproduced.** `check_call` on the flagged call returns `R2-path, / (filesystem root)`. Replacing only the two
    `.replace('\n','/')` with `'|'` gives `[]`. No other call in that episode fires.
  - **What the command reads.** Two scratch pointer files and the two `out/resp_*.json` files they point to, all in
    the sandbox (C6).
  - **Twins that still fire:** `open('x'.replace('x','/')+'etc/passwd')`, `os.listdir('/')`,
    `print(open('x'.replace('x','/')+'etc/passwd').read())` and `print(os.listdir('x'.replace('x','/')))`.
  - **Also flagged today:** a bare `print(s.replace('\n','/'))`, a replace inside a list or f-string inside print,
    and `print('/'.join(...))` (SCALE1_T2 H1 f).
  - **The fix.** The proposed fix (exempt the replace only when its value is a direct print argument) must keep the
    last two twins of the first list firing, because there the replace sits inside print but feeds `open` or
    `listdir`.
- **(b) Runner reporting mismatch.** Both episodes match the file:
  - the submit came at turn 19 of 21 and turn 32 of 34, through `subprocess.check_output(['./tool','submit',...])`;
  - the regex `\./tool\s+submit\b` does not match that form;
  - the tool result was `accepted: true`, and the tool log has exactly one accepted submit;
  - the record shows `submitted` 211.7 s and 331.1 s after the first tool call, 9.2 s and 7.9 s before close;
  - a text summary followed, then the false nudge, then a text reply, then `ended_without_submit`;
  - the printed answer equals submission.json (episode 1), and the regrade matches.

  The proposed fix (check `_episode_submitted` after every tool batch, without the regex gate) is right for both
  runners. The same gate is in api_agent.py.

## Failed episodes, re-read

- **T2 sol, 3 episodes.** The classification holds.
  - **(A).** 5 latents on the failing probe, no ablation check, submitted after 31 of 150 calls and 180 s.
  - **(B), first episode.** A candidate scan that keeps a latent only if all its top dashboard examples carry the cue.
  - **(B), second episode.** 20 latents ranked by sample attribution and then read on dashboards. Its in-sample sums
    were lowest on the failing probe. Those sums are in absolute logit units, though, so "lowest" is a weak warning
    sign, not a known shortfall.
  - **Budget.** All three stopped with most of the budget unused: 31-60 of 150 calls, 600-1,108 of 3,000 forward units
    and 180-435 s of 3,600 s. That fits C1.
- **luna, 10 episodes.** "Model limitation, not environment" holds.
  - **Pattern.** Most episodes follow the same arc: a confound table, one or a few edited copies of 1-3 base texts per
    attribute, top latents by contribution (often with a topic penalty), 2-7 latents in most slots, and a check of
    the sample label contrast rather than of the cue effect (two episodes did estimate the cue effect, C4 b).
  - **Tool friction was real but recovered.** Two episodes had bash quoting errors. One named a script `select.py`,
    which shadowed the module and broke `subprocess`. There were a missing `python` binary, wrong `out/` file
    assumptions and one re-sent submit after a quoting error.
  - **Bound to the method, not the tools.** Every submission was well-formed and accepted. TASK.md loss was nil (C5).
  - **Wall clock.** It is almost all model latency (8-78 s per turn; task tools 13-23 s per episode). It reached at
    most 57% of the 60-minute task clock.
  - **Forward budget.** One episode ended at 2,980 of 3,000 units. Its decisive errors came earlier, from noisy single
    edits, so the cap did not decide it.
  - **Corrections.** Details are in C4.

## Sampled passes (3 of 17, seeded)

- **T1 sol.** A Hadamard factorial design over the 8 attributes on its own texts and a regression of latent effects
  times g. It compared each attribute's effect with the topic effect on the disclosed scale, then matched pairs for
  the claimed cue and the top 20 latents with a topic penalty. It ran an ablation check, and its 3 null calls were
  right on the three-null instance.
- **T2 sol, first episode.** Sample attribution and reading of about 280 latents' dashboards. Null calls came from decoy
  recognition. It spent all 4 query units on its one planted probe and checked an ablation of 20 latents.
- **T2 sol, second episode.** The same pipeline over about 480 latents' dashboards, plus a ranking of candidates by
  which attributes their dashboard texts carry and per-token activations on sample texts. It spent its 4 query units
  on two edit pairs, one per planted probe.

None of the three shows a guess, prior or shortcut around latent reading.

## Misleading factors, checked

| factor | finding |
|---|---|
| per-episode budget and turn caps (90 turns, $1.50) | never near: at most 39 turns and $0.545; no warning; same task caps as the Claude arms, which had no turn or $ cap |
| effort | **asymmetric: Opus xhigh, OpenAI medium (C1)** |
| system prompt | differs (api_agent SYSTEM against Claude Code's); the task prompt is identical; Claude agents were also asked for a final report |
| output truncation | 12k head and tail for OpenAI against about 30k for Claude Code; 102 cuts; no grade affected (C5) |
| output token cap (16k) | never hit (all responses `completed`) |
| luna on tool mechanics | no: all submissions accepted, friction recovered, fails at selection and estimation (C4) |
| wall clock | luna's long episodes are model latency, within the cap; not binding |
| pool selection | the same instances as the Claude arms, so the keep-filter inflation of SCALE1_T2 C1 applies to sol as well |
| exposure and context | confined, as stated; the copying argument covers T1 only (C3) |

## Recommendations (amending the review)

1. Add C1 to SCALE1_OPENAI.md and VALIDATION.md's "one model family" weakness. For the next cross-family run, match
   effort or vary it: sol at high, or Opus at medium. If budget allows, one luna arm at high effort would separate
   model from effort.
2. Report the T2 slot-level test and the bar sensitivity (C2) next to the pass rates.
3. Implement O1 with all four twins in the anomaly (a) section above, and O2 as proposed. Re-finish the one T2
   episode with `--transcript`.
4. Gitignore `runs/shifthunt/_openai_jobs/` (C8).
5. In openai_agent.py, either raise the output limit to about 30k to match Claude Code, or record truncation counts in
   api_meta.json (C5).

## Files

- **Committed:** this file only.
- **Not changed:** the run dirs, SCALE1_OPENAI.md and exposed_instances.json.
- **Scratch.** Derive, paired, regrade, audit-twin and dump scripts lived in a gitignored directory under
  `runs/shifthunt/` and were deleted after use.
- **One oversized output.** A dump of one luna episode's commands and outputs (37 KB, with latent ids of an LLM-seen
  instance and no answer file) went over Claude Code's output limit. The harness saved it to the session's
  `tool-results/`, and I deleted it within a minute. No Claude test agent was running.
- **Other files in `tool-results/`.** No file there is newer than 17:00 UTC today, and none mentions the OpenAI run
  dirs or episode ids. The review's two dumps are gone.
- **What was printed and written.** No canary, `runs/.episodes` record or answer file was printed. Nothing was
  written outside the worktree's `runs/` and `tasks/`.
