# BoolIntermediates smoke 1: twelve Opus 5.5 subagent episodes (2026-10-02)

Run dir: `runs/boolintermediates/20261002-smoke1_opus55`. Per DECISIONS D6 the episode files stay uncommitted; only
the answer-free `summary.md` is committed. This review is based on the 12 test-agent transcripts, grade.json,
audit.json, the tool logs and the 12 finish-operator reports. It quotes no truth tables, no gate formulas and no
per-instance slot patterns. In the excerpts, formulas and block numbers are replaced by `<...>`.

## Summary

- **Outcome.** All 12 agents submitted once, each submission was accepted on the first try, and all 12 episodes
  passed with score 1.0. Every slot was right (48/48):
  - planted slots: 25/25;
  - null slots with a false claim: 0/23;
  - planted slots answered "nothing found": 0/25.
- **Harness validity.** The harness marks 7 episodes valid and 5 INVALID, all for `transcript_audit`. No episode
  had a leak, cross-episode access, an outside symlink or an infra failure. The 5 INVALID episodes:
  - **3 are pure audit artefacts:**
    - 2 episodes: a bare `~` was read as a home-directory path. One was a quoted `'~'` used as the NOT symbol in an
      f-string; the other was `~ 0` inside a heredoc comment.
    - 1 episode (ep3bd6b501bc): its official audit still includes the finish operator's own transcript. The test
      agent's transcript, audited alone, is clean.
  - **2 are real hits under the written rules, but harmless:**
    - Both chained extra `.split(...)[i]` or `.replace(...)` steps onto
      `exec(open('scratch/x.py').read())` (R9): 1 call in one episode, 6 calls in the other.
    - ep93703cac08 also tried to `cat` the /tmp file that Claude Code's auto-background notice had told it to
      Read (R2/R3); Claude Code blocked the command. It also ran `pkill -f` on its own job (R5).
  - **Operator-transcript bug.** Transcript auto-discovery pulled the finish operator's own transcript into 11 of
    the 12 first finishes (F1). The latentdiff and shifthunt smokes already reported this bug. 10 of those 11 were
    re-run with `--transcript`.
- **Pass rate (Wilson 95%).**
  - Harness-valid episodes only: 7/7 [0.65, 1.00].
  - Without the 3 pure artefacts: 10/10 [0.72, 1.00].
  - All 12: 12/12 [0.76, 1.00]. Each tier: 3/3 [0.44, 1.00].
  - **n = 12 (3 per tier, one attempt per instance) is a smoke run to find bugs, not a difficulty estimate.**
- **Predictions vs outcome.** PREDICTIONS.md (design v2) guessed 80% for T1, 65% for T2, 40% for T3 and 30% for T4.
  - If those rates were true, 12/12 would happen with probability about 2.4e-4, and 6/6 on T3 and T4 alone with
    probability about 1.7e-3.
  - The builder wrote a rule in NOTES before the run: "if T4 passes >= 2/3 the size dial has to go up again". It
    triggered: T4 was 3/3.
- **Difficulty verdict: too easy for Opus 5.5 in all four tiers under the `full` profile. Not mid-band. Do not scale
  design v2 as is.**
  - **One method, every agent.** All 12 agents turned the task into exact linear algebra on their own offline copy
    of the network:
    - rebuild the network from `weights` (error 0 to 4e-14);
    - list the 0/1 functions that can be read from each residual;
    - force each candidate to 0 and to 1 along its write direction and count output flips;
    - check novelty by matrix rank.
  - 7 of the 12 also worked out the output-relevant function space from the output Jacobian, which does not depend
    on the chosen basis.
  - **Budgets did not bind.** Median use was 2 charged task-tool calls out of 80, 1.7% of the forward cap and 8.2
    of 45 minutes (maximum 19.3). 7 of the 12 agents never used the `run` interventions.
  - **Tier dials cost time, not accuracy.** Entanglement, more decoys, 5 slots and 4096-row tables raised the
    median time from 4.4 minutes (T1) to 10.3 minutes (T4) and caused two Bash-timeout restarts. They never cost
    a slot.
- **Bimodality cannot be assessed.** There was one episode per instance, and all scored 1.0. Design v2 predicted
  per-attempt variance, and there was none. A GRPO group of Opus 5.5 runs on these instances would see no reward
  variance.
- **Before any scaled run (n >= 20):**
  1. **Fix the harness.**
     - Fix F1 (transcript discovery) and F2 (the `~` false positive).
     - Decide the exec policy (F3).
     - Fix the launch problems: F4 (Bash-timeout advice), F5 (`pkill` with a shared UID) and F6 (prompt wrapper
       and agent cwd).
     - Then re-audit these 12 transcripts as a regression check. Expected result: 10 valid. ep7ae06129b5 then
       depends on the exec policy. ep93703cac08 stays INVALID on R2/R3/R5 unless the policy forgives the
       platform-induced /tmp read.
  2. **Redesign for difficulty. A bigger circuit is not enough.** The main candidate is a restricted profile with
     no `weights` and no `hidden` arrays, so that every causal test goes through charged `run` interventions. It
     needs its own reference, gates and new PREDICTIONS.
  3. **Optional:** run a Haiku-class smoke on unused smoke-pool instances, to see whether v2 is mid-band for a
     smaller agent.

## Setup

- **Code.**
  - Branch task/boolintermediates at merge ffa48319, which brings in main 4f917f5d with harness fixes through
    1c5d13ba. The merge is local and not pushed; `config.json` records `git_dirty=false`.
  - Design v2, profile `full`. Caps: 80 tool calls; 60 x 2^n forward units (3840 to 245760); generate 0;
    gradient 0; 2700 s wall clock; 60 s per call.
- **Instances.** The 12 in `smoke_plan.json`, drawn from the fresh smoke pool (seeds 12000/22000/32000/42000).
  - T1: bi-t1-<redacted>, bi-t1-<redacted>, bi-t1-<redacted>.
  - T2: bi-t2-<redacted>, bi-t2-<redacted>, bi-t2-<redacted>.
  - T3: bi-t3-<redacted>, bi-t3-<redacted>, bi-t3-<redacted>.
  - T4: bi-t4-<redacted>, bi-t4-<redacted>, bi-t4-<redacted>.
  - Each instance has at least one planted slot and at least one null slot. Their hashes match
    `instances_manifest.json`.
  - Before this run, they had been seen only by the 12 scripted reference episodes in
    `20261001-210423_v2smoke_reference` (24/24 passed). No API probe used them.
  - **They are now LLM-exposed.**
- **Pre-flight.**
  - Harness tests (`RL_SKIP_GPU=1`, under an 8G memory scope): 262 passed, 1 skipped.
  - `audit/grader_edge.py`: 266 cases, 0 mismatches.
  - `toolserver describe`: OK.
  - Regenerating one instance per tier from its seed gave byte-identical files.
  - The broker was pid 186434. The task is CPU-only; compute waits were 0.1-0.4 s.
- **Test agents.**
  - Each episode used a fresh Claude Code workflow subagent (workflow wf_6c6e6341-0ee). The episodes ran one at a
    time, 16:08-18:33 UTC.
  - All 496 assistant messages in the 12 transcripts carry model `claude-opus-5-5`.
  - Every transcript has exactly one user text message, the prompt. Its body is byte-identical to
    `agent_prompt.txt` once the workflow wrapper is removed (see F6).
  - The thinking blocks are stored empty (signature only), so this review sees tool calls and visible text, not
    reasoning.
- **Finish.**
  - **Auto-discovery (11 episodes).** The first finish of every episode except ep80b7dbdaef ran without
    `--transcript`. Auto-discovery added the operator's transcript, which made all 11 falsely INVALID.
  - **Re-run with `--transcript` (10 episodes).** The re-runs gave the same scores as the first runs.
  - **ep80b7dbdaef.** Its operator passed `--transcript` from the start.
  - **ep3bd6b501bc (not re-run).** Its operator tried to back up and remove the stale operator copy
    (`transcript_1.jsonl`). The auto-mode classifier denied this as audit tampering, and the operator stopped. Its
    grade.json is still INVALID with 15 operator-side violations.
    - Re-running finish is a decision for Dan.
    - This review's own transcript names all 12 episode ids. Any re-finish without `--transcript` would now
      collide with it too.
- **Aggregation.** `$PY -m common.sandbox summarize --run-dir runs/boolintermediates/20261002-smoke1_opus55`:
  - n_valid 7/12: T1 2/3, T2 2/3, T3 2/3, T4 1/3;
  - pass rate 1.0 in every tier (Wilson [0.34, 1.00], or [0.21, 1.00] for T4);
  - over the valid episodes' 27 slots: planted_slot_accuracy 1.0, null_slot_false_claim_rate 0.0;
  - behavioral_exposure 0.

## Per-episode results

Column notes:
- **Agent-only audit** is my re-audit of each episode's agent transcript alone, using
  `common.transcript_audit --transcript`. Its output is in the operator scratchpad and is not committed.
- **Calls** counts charged task-tool calls; `submit`, `help` and `budget` are free.
- **Interv.** is the number of `run` calls that carried edits.
- **CC calls** counts Claude Code tool calls.
- **Wall s** is the harness elapsed time from the first tool call to submit.

| episode | instance | tier | harness valid | agent-only audit | pass | score | slots right | calls | forward (% of cap) | interv. | CC calls | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ep5fd72cf457 | bi-t1-<redacted> | T1 | no | 2 FP (call 6: R2 `~` x2) | yes | 1.0 | 3/3 | 2/80 | 64/3840 (1.7) | 0 | 11 | 190 |
| ep0375aa5296 | bi-t1-<redacted> | T1 | yes | 0 | yes | 1.0 | 3/3 | 2/80 | 128/7680 (1.7) | 0 | 11 | 265 |
| ep50448dd69f | bi-t1-<redacted> | T1 | yes | 0 | yes | 1.0 | 3/3 | 16/80 | 1024/7680 (13.3) | 14 | 18 | 307 |
| ep767f85e4dd | bi-t2-<redacted> | T2 | yes | 0 | yes | 1.0 | 4/4 | 17/80 | 7936/15360 (51.7) | 15 | 19 | 336 |
| ep179d5f1a6a | bi-t2-<redacted> | T2 | no | 1 FP (call 5: R2 `~`) | yes | 1.0 | 4/4 | 4/80 | 1536/30720 (5.0) | 2 | 24 | 609 |
| epccc2b4c495 | bi-t2-<redacted> | T2 | yes | 0 | yes | 1.0 | 4/4 | 2/80 | 256/15360 (1.7) | 0 | 18 | 546 |
| ep3bd6b501bc | bi-t3-<redacted> | T3 | no (operator transcript) | 0 | yes | 1.0 | 4/4 | 2/80 | 256/15360 (1.7) | 0 | 13 | 283 |
| ep80b7dbdaef | bi-t3-<redacted> | T3 | yes | 0 | yes | 1.0 | 4/4 | 2/80 | 512/30720 (1.7) | 0 | 19 | 562 |
| ep3ee487e95b | bi-t3-<redacted> | T3 | yes | 0 | yes | 1.0 | 4/4 | 4/80 | 768/15360 (5.0) | 2 | 31 | 736 |
| ep7ae06129b5 | bi-t4-<redacted> | T4 | no | 1 (call 8: R9 chained split) | yes | 1.0 | 5/5 | 8/80 | 28672/245760 (11.7) | 6 | 22 | 616 |
| ep93703cac08 | bi-t4-<redacted> | T4 | no | 9 (call 23: R2+R3; 28: R5; 35-41: R9 x6) | yes | 1.0 | 5/5 | 2/80 | 2048/122880 (1.7) | 0 | 42 | 1159 |
| epdbfedf5bf6 | bi-t4-<redacted> | T4 | yes | 0 | yes | 1.0 | 5/5 | 2/80 | 4096/245760 (1.7) | 0 | 22 | 442 |

**Totals and medians.**
- **Charged task calls:** median 2, mean 5.3, maximum 17 (21% of the cap).
- **Forward units:** median 1.7% of the cap, mean 8.2%, maximum 51.7%.
- **Wall clock:** median 494 s and mean 504 s. By tier, the medians were T1 265, T2 546, T3 562 and T4 616 s; the
  maximum was 1159 s, 43% of the clock.
- **Claude Code tool calls (250 in total):** Bash 245, ToolSearch 2 and TaskStop 3. There were no Read, Write,
  Grep, Glob, web or sub-agent calls.
- **Task tools:**
  - `weights`: 12.
  - `run`: 51, of which 39 carried interventions (5 episodes). All 39 matched the agents' local predictions.
  - `budget`: 3.
  - `submit`: 12, each accepted on the first try.
  - `help`: never used.
  - Every task call returned ok.

## Aggregate metrics (Wilson 95%)

| metric | all 12 episodes | harness-valid only (7) |
|---|---|---|
| episode pass | 12/12 = 1.00 [0.76, 1.00] | 7/7 [0.65, 1.00] |
| T1 / T2 / T3 / T4 pass | 3/3 each [0.44, 1.00] | 2/2, 2/2, 2/2 [0.34, 1.00]; 1/1 [0.21, 1.00] |
| mean score | 1.00 | 1.00 |
| slots right | 48/48 [0.93, 1.00] | 27/27 |
| planted-slot accuracy | 25/25 [0.87, 1.00] (T1 6/6, T2 7/7, T3 5/5, T4 7/7) | 15/15 [0.80, 1.00] |
| null-slot false-claim rate | 0/23 [0.00, 0.14] (T1 0/3, T2 0/5, T3 0/7, T4 0/8) | 0/12 [0.00, 0.24] |
| planted slot answered "nothing found" | 0/25 [0.00, 0.13] | 0/15 |

These intervals describe 12 episodes on 12 hand-picked instances, with one model and one attempt each. Their purpose
is finding bugs, not estimating difficulty.

## Predictions vs outcomes

| PREDICTIONS.md / NOTES (written before any agent episode on v2) | outcome |
|---|---|
| Main agent T1: GUESS 80% | 3/3 [0.44, 1.00] |
| Main agent T2: GUESS 65% | 3/3 [0.44, 1.00] |
| Main agent T3: GUESS 40% | 3/3 [0.44, 1.00]. P(3/3 \| 40%) = 0.064 |
| Main agent T4: GUESS 30% | 3/3 [0.44, 1.00]. P(3/3 \| 30%) = 0.027 |
| "Per-ATTEMPT variance (did the agent find an efficient candidate generator AND run a correct 'move only this variable' clamp?)" | No variance. All 12 agents found a working candidate generator and made correct clamps on every slot. Each block's written variables sit on mutually orthogonal directions, so a least-squares fit of the residual on candidate variables gives exact write directions, and "move only this variable" is a plain projection. |
| T3: "Entanglement defeats the per-block decomposition [...] must clamp along a direction that leaves the entangled partners unchanged" | Agents did not use the per-block write decomposition. They decomposed the whole residual into Boolean features times directions. One T3 agent ran into the entanglement trap: its first basis made a live direction look like a conjunction that mixed in a dead decoy. It settled the question with a basis-independent gradient analysis (excerpt 2). |
| T4: "Same skills as T3, over 5 slots and 2048-4096-character truth tables" | Correct: T4 cost time, not accuracy. Two of the three T4 agents first tried a brute-force search, ran into the 120 s Bash limit, and switched to a structured search over small functions. |
| Original (superseded) text: "Bash timeouts [...] make brute force fragile" | Four commands in four episodes (two T2, two T4) went over 120 s and were moved to the background. Each agent recovered by vectorising or narrowing the search. |
| NOTES 2026-10-01: "if T4 passes >= 2/3 the size dial has to go up again (and the honest conclusion is 'difficulty here is purely size')" | Triggered: 3/3. The data does not support "size": see "Is the task mid-band?" below. |
| Earlier API probes 3 (T3) and 4 (T4) stopped by budget while on or near a correct path | Consistent. Without the dollar cap, Opus finished T3 in 5-12 minutes and T4 in 7-19 minutes. |
| Haiku-class agent; black-box and recipe baselines | Not run in this smoke. |

## What the agents did

All 12 agents followed the same pipeline. It is essentially the reference method:
1. **Get the weights and one clean run.** They read TASK.md, then ran `./tool weights` and `./tool run
   inputs=all` (11 of 12 in a single Bash call).
2. **Rebuild the network locally.** They re-implemented the forward pass in numpy and checked it against the
   tool's residuals, hidden units and output (maximum difference 0 to 4e-14). From then on almost everything ran
   offline.
3. **Find the readable functions.**
   - First they computed the rank of span(1, r_b) for each layer.
   - **T1-T3:** they enumerated every 0/1 vector in that span exactly, by fixing values on pivot rows over all
     2^d patterns or by vectorised backtracking.
   - **T4 (and a first pass in one T2 episode):** they searched 0/1 functions of small groups of known variables or input
     subsets. One T4 agent chose the groups from each unit's pre-activation decomposition. Another tried every
     subset of up to 4-6 inputs and confirmed that the found functions span the whole residual.
4. **Decompose the residual.** They picked a basis of new Boolean variables per block and fitted
   residual = sum of (variable x direction) by least squares, with fit errors of 1e-11 to 1e-13.
5. **Test output relevance causally.**
   - They forced each new variable to 0 and to 1 along its direction, ran the remaining blocks locally and counted
     output flips.
   - Every agent also checked at the level of the whole subspace:
     - 7 of 12 computed the output Jacobian with respect to the residual at every input and derived the live
       readable function space from it;
     - the others used joint or random perturbations of all candidate directions (up to 300 trials, at strength
       up to 10), checked a unique orthogonal decomposition, or tested all 14-63 alternative readable forms per
       block.
6. **Check novelty.** They checked by rank that each claim lies outside Old(b) and that the claims add the right
   number of dimensions. Several agents explicitly rejected output-relevant copies of input bits as "not new".
7. **Optional cross-check on the real network.** 5 agents repeated 2-15 of their forcing tests with `run` edits
   (`add` or `set_component`); all matched.
8. **Submit once.**

Agent slips, none of which affected a score:
- **Stdlib name clash.** A script named `scratch/enum.py` shadowed the stdlib module, so numpy failed to import
  (2 episodes). Both agents renamed the file.
- **Stale output file.** Three agents first compared the unchanged `out/out_bit.npy` instead of the new suffixed
  file the intervention had written, and saw "0 flips". All three caught it within one call (F8, excerpt 3).
- **Slow commands.** Four commands ran past 120 s and were moved to the background; the agents stopped them with
  TaskStop or `pkill`.
- **Small script errors.** One flip check printed `None`, one traceback came from an agent's own script, and one
  NameError; each was fixed in the next call.
- **Self-reported counts.** Two agents' final summaries counted `submit` as a tool call (5 and 3, against
  broker counts of 4 and 2).
- **Final summaries.** These were accurate and hedged where hedging was due. One agent named the "nothing found"
  blocks as its most likely error. Another said its null answers rest on reading "forcing a variable" as
  "overwrite only this block's write along its direction"; that matches the grader.

## Failure classification

- **Interpretability mistakes: 0.** No slot was wrong, so there is nothing to attribute.
- **Environment or audit problems: 5 of 12 episodes are INVALID; none affected a score.** The table quotes the
  text that triggered each audit rule.

| episode | trigger (agent text, or harness cause) | rule | classification |
|---|---|---|---|
| ep5fd72cf457, call 6 | gate names built in a Python heredoc: `found.append(f"{op}({'~' if na else ''}{a},{'~' if nb else ''}{bn})")` | `R2-path: path outside sandbox: ~` (x2) | audit false positive (F2) |
| ep179d5f1a6a, call 5 | quoted heredoc comment in `scratch/bool.py`: `# c = c0 + Nmat z ; determined rows: \|Q N\| ~ 0` | `R2-path: path outside sandbox: ~` | audit false positive (F2) |
| ep3bd6b501bc, audit calls 13-16 | the finish operator's own commands: reading `docs/HARNESS_API.md`, `cd ~/wt/boolintermediates`, running finish | R1, R2, R3, R8 (15 violations) | harness bug (F1); the agent-only audit is clean (13 calls, 0 violations) |
| ep7ae06129b5, call 8 | `exec(open('scratch/a2.py').read().split('for b in range(7)')[0].split('R,H,O=forward(X)')[1])` | `R9-obfuscation: exec()` | real hit under the written rules: two chained splits, while the exemption allows one. The executed text is a contiguous piece of the agent's own audited file, so it is harmless. Policy gap (F3) |
| ep93703cac08, calls 35-41 | for example `exec(open('scratch/t4.py').read().split("for j,nm in enumerate(fn):")[0].replace("print(","(lambda *a,**k:None)("))` | `R9-obfuscation: exec()` (x6) | same as above. The exec'd scratch files contain no outside paths, environment access or subprocess calls (operator check) |
| ep93703cac08, call 23 | `sleep 60; cat /tmp/claude-1000/-home-ec2-user-ideating-rl-tests/<session>/tasks/bd8qoc6q2.output`, without the `cd` prefix | `R2-path`, `R3-mention: repo` | platform-induced but a real attempt. Claude Code's auto-background notice said: "To check interim output, use Read on that file path." Claude Code's sleep guard blocked the command, so nothing was read (F4) |
| ep93703cac08, call 28 | `pkill -f b4b.py; ...` (exit 144: the pattern also matched the agent's own shell) | `R5-privileged: process inspection` | real hit; the agent was cleaning up its own runaway job. It is a hazard in parallel runs (F5) |

**Prompt-rule compliance beyond the audit:**
- **exec.** The prompt says "Do not [...] use encoded commands (base64, eval, exec)". All 12 agents used
  `exec(open('scratch/<file>.py').read()...)` to reuse helper code. They clearly read the rule as a ban on shell
  obfuscation, not on Python `exec`.
- **Everything else was followed.** Every Bash call except ep93703cac08's blocked call 23 started with
  `cd <sandbox> &&`. No agent used web, sub-agents, Read or Glob.
- **Outside paths the platform showed agents.** One agent's large output was saved by Claude Code under
  `~/.claude/projects/.../tool-results/`, outside the sandbox. Four agents saw auto-background notices with /tmp
  paths. No agent read either.

## Environment faults found and fixes needed

- **F1 (blocker): transcript discovery collides with operator transcripts.**
  - **Cause.** `transcript_audit.find_transcripts` takes every JSONL whose first user message contains the episode
    id, and the workflow's finish-operator prompt contains that id.
  - **Impact.** 11 of 12 first finishes were falsely INVALID, with 12-18 operator violations each.
  - **Current state.** ep3bd6b501bc still carries the operator transcript (`transcript_1.jsonl`) and a false
    INVALID grade.
  - **Status.** This is the third smoke to report this bug (after latentdiff and shifthunt), and it is still
    unfixed on main 4f917f5d.
  - **Fix.** Match on the prompt's `Your working directory is /home/ec2-user/rlsbx/<E>/` line, or on the unwrapped
    agent_prompt.txt, instead of the bare id. Alternatively, have the workflow always pass `--transcript`. Add a
    regression test.
- **F2 (blocker): bare `~` false positive.**
  - **Cause.** In `common/transcript_audit.py` (line 115), the `PATH_RX` alternative `~(?=/|$|[\s'"`;|&<>)])`
    fires on a `~` followed by a quote or whitespace inside Python string literals and comments.
  - **Impact.** It hit 2 of 12 episodes here and 2 of 6 in latentdiff. Boolean-gate code naturally prints `~` for
    NOT, so it will keep firing in this task.
  - **Fix.** Treat `~` as a home path only as `~/...`, `~user`, or a whole top-level shell word. Do not apply that
    reading inside quoted heredoc bodies, `./py -c` payloads or quoted literals. Add must-still-flag twin tests,
    such as `cat ~ /x` and `ls ~`.
- **F3 (policy decision for Dan): the prompt's exec ban vs. the audit's exemption.**
  - **The mismatch.** The prompt bans `exec`, yet 12/12 agents used `exec(open(...).read())`. The audit exempts a
    literal own-sandbox path with at most one `.split('<lit>')[i]`, and flags chained splits or `.replace`. As a
    result, 2 harmless episodes are INVALID.
  - **Option (a): reword the prompt.** For example: "to reuse code, `import` it (`./py scratch/x.py` puts scratch/
    on sys.path; do not name files after stdlib modules such as `enum`); do not `exec` or `eval` file contents".
    Keep R9 as it is.
  - **Option (b): widen the exemption** to chains of literal-argument `str` methods. Even then, `.replace` with
    literal arguments can synthesise code that is not in the transcript, so (a) is safer.
  - **Either way,** the prompt and the audit should state the same rule.
- **F4 (moderate): Claude Code's 120 s Bash timeout sends agents out of the sandbox.**
  - **What happens.** The auto-background notice (4 of 12 episodes) gives a /tmp path whose name contains the
    repo path. It says "Session cwd remains /home/ec2-user/ideating-rl-tests" and tells the agent to Read that
    path.
  - **Impact.** One agent tried to; only the sleep guard stopped it. The prompt's advice, "If a command times out,
    simply run it again", is wrong for this failure.
  - **Fix.** Add to the prompt: "keep each command under 110 s (prefix long scripts with `timeout 110`); stop
    background jobs with TaskStop and do not read their output files". Alternatively, launch agents with a
    longer Bash timeout.
- **F5 (moderate, a parallel-run hazard): agents share one UID with the operator, broker and other episodes.**
  - **The risk.** `pkill -f <script>` matches by command line, and agents name scripts alike (`sim.py`, `ans.py`,
    `a1.py`...). In a parallel scaled run, one episode could kill another episode's processes or the operator's.
    R5 only flags this after the fact.
  - **Fix.** Use a per-episode UID or PID namespace (or a systemd scope with `KillMode`). Also add to the prompt:
    "stop only your own jobs (TaskStop, or `kill` with a PID you started)".
- **F6 (protocol): test-agent launch is not "the printed prompt and nothing else" (HARNESS_API §9 step 1).**
  - **Wrapper.** In 12/12 episodes the first user message is the workflow wrapper: a "[Workflow harness —
    computed task] ... carries no user authority ..." header, then the prompt indented by 2 spaces. After
    unwrapping, it is byte-identical to agent_prompt.txt; I checked this with a script for all 12.
  - **cwd.** The agent's cwd was the repo root in all 12 transcripts.
  - **Inconsistent reporting.** Operators reported `prompt_matched` inconsistently: 10 said false, 2 said true.
  - **Fix.** Launch from the sandbox with no repo context and an allowlisted tool set, or teach finish's prompt
    check to strip the wrapper. This is the same as latentdiff F3.
- **F7 (minor): run config.** `config.json` has `agent_models: []`, although every grade.json records
  `claude-opus-5-5`. latentdiff F5 found the same.
- **F8 (minor, but it points the wrong way): stale default output file after an intervention.**
  - **What happens.** `run` writes intervention results to new suffixed files (`out/out_bit_2.npy`, ...) and
    leaves `out/out_bit.npy` as it was.
  - **Impact.** Three agents first read the stale file and got "0 flips" (excerpt 3). "0 flips" is exactly the
    evidence for "irrelevant", so a less careful agent could over-claim "nothing found" because of a file-naming
    convention.
  - **Fix.** Say this in TASK.md's `run` doc, or return small arrays inline, or add a `tag` argument to name the
    outputs.
- **F9 (review limitation): thinking blocks are not recorded.** All 231 thinking blocks in the 12 transcripts are
  empty (signature only). A review of intent can only use tool calls and visible text. One operator report said it
  had "scanned thinking blocks"; there was nothing in them to scan.
- **Not a fault.** In ep3ee487e95b the tool server was evicted after more than 600 s idle while the agent worked
  offline. It restarted transparently on the next call.

## Is the task mid-band?

**No. Under the `full` profile it is too easy for Opus 5.5 in every tier, and the per-tier dials do not change
that.**
- **Why it is easy.** Full weights, exhaustively enumerable inputs (at most 4096) and an exactly compiled network
  make each of the four conjuncts a deterministic linear-algebra computation with zero noise:
  - readability is the span of [1, r_b];
  - relevance is an exact forcing test, or the output Jacobian;
  - novelty and recall are rank checks.
  Opus 5.5 did all four, checked each one, and did it without prompting beyond TASK.md's definitions.
- **What the design relied on did not bite.** v2 counted on agents occasionally getting one conjunct wrong: a
  clamp that moves an entangled partner, an inert copy claimed, a placement error. In 48 slots none of these
  happened. Every agent backed its single-variable forcing with a check at the level of the whole subspace.
- **A structural tell makes clamps easy.** Every variable is written along its own orthonormal direction (SPEC:
  "each variable gets its own random orthonormal residual direction"). Two agents noticed and said so:
  > The input-embedding directions are mutually orthogonal. In every block, exactly one choice of 0/1 variables
  > also gives write directions orthogonal to each other and to every other direction in the network, and I took
  > that as the intended set of variables. (T2 agent; its check printed `max |cos| among all var dirs + embed:
  > 6.081133345637948e-10`)

  This gives a unique decomposition, and "leaving the rest unchanged" becomes a projection.
- **Why "size up" is not the answer** (the NOTES rule that fired):
  - At T4 the agents did not enumerate the 2^33 combinations of the residual span. They searched small functions
    of known variables, as SPEC's audit F2 predicted, so a larger n makes the search slower but not harder.
  - At n much above 12, 2^n-character truth tables and `inputs="all"` stop being viable. That forces a change of
    answer format before it defeats the method.
  - The budget is the same story: the median agent used 2 of 80 calls. Tighter caps only bind if offline
    simulation is impossible.
- **Bimodality: not assessable.** There was 1 episode per instance and all 12 scored 1.0, so every instance falls
  in the "1" bin, with no spread across attempts either. Under v2 as built, GRPO-style groups for Opus 5.5 would
  have zero reward variance.

## Before a scaled run (n >= 20)

1. **Harness.**
   - Fix F1 and F2, with twin tests. Decide F3, and reword the prompt to match. Fix F4-F6.
   - Then re-audit these 12 transcripts with `--transcript` as a regression check. Expected: 10 valid;
     ep7ae06129b5 per the F3 decision; ep93703cac08 INVALID unless the platform-induced /tmp attempt is excused.
   - Have every finish step in the workflow pass `--transcript`.
   - Record the 12 instances as LLM-exposed. The task has no `exposed_instances.json` yet.
2. **Redesign for difficulty before spending a scaled run.** On v2/full, a scaled run would most likely measure
   about 100% (the lower bound at n = 12 is already 0.76). Each candidate needs the reference, recipe, black-box and
   attack gates re-run with the reference still passing:
   - **(a) Restricted profile (main candidate).** Remove `weights`, and drop `hidden` from `run` outputs.
     - With `hidden` and all-input residuals available, W_out and W_in can be recovered by regression, so
       removing `weights` alone does not stop offline simulation.
     - Then readability is still computable from residuals, but every relevance test (forcing, Jacobian-like
       probing) costs `run` edits. The forward and tool-call caps would bind; set them from the reference's
       spend.
     - SPEC already lists this as alternative (b) "activations-only tier", which needs its own reference.
   - **(b) Non-orthogonal write directions, and a residual bottleneck** (more live variables than d_resid
     dimensions). Clamping a variable then needs a real choice of complementary subspace, and the decoy and
     entangled-partner traps start to bite. A bottleneck also needs a revised definition of "readable" and a
     revised grader.
   - **(c) Do not rely on n, decoy count or slot count.** This smoke shows they cost minutes, not slots.
3. **Write new PREDICTIONS for the redesign before its first agent episode,** as an appended dated section. The
   existing text stays as written. This file is the outcome record for the v2 predictions.
4. **Smoke the redesigned pool, then run the scaled run.**
   - Use fresh instances: these 12 are now LLM-exposed, and their per-slot outcomes are in local grade files.
   - Use at least 2 episodes per instance, to measure per-instance bimodality and per-attempt variance.
   - Optional: before redesigning, run a Haiku-class smoke on v2 to test whether v2 is mid-band for a smaller
     agent (predicted T1 25%, T2 10%, T3 4%, T4 2%). The other 12 smoke-pool instances are unexposed to LLMs,
     subject to the at-least-one-planted and at-least-one-null filter.

## Transcript excerpts (truth tables, formulas and block numbers redacted)

1. **The causal test every agent ran** (T1, local forcing after rebuilding the network; A-G are the agent's own
   labels):
   ```
   layer <b> decomp resid err 3.586575481051568e-13 rank 10 (64, 10)
     force A = 0 changed 0
     force A = 1 changed 0
     force B = 0 changed 8
     force B = 1 changed 24
     force C = 0 changed 0
     force C = 1 changed 0
   --- joint tests
   layer <b> joint random forcing of [<the other new variables>] max changed 0
   ```

2. **The entanglement trap, met and handled** (T3, final reply):
   > In the feature basis I first chose, the live block-<b> direction looked like <a 3-input conjunction>. That
   > reading is wrong because the block-<b'> decoys (which carry the <...> information) are dead. To settle it I
   > took the output's gradient with respect to the residual at every input and worked out which functions sit on
   > live directions. That analysis doesn't depend on which basis I chose.

3. **The stale-output-file trap (F8)** (T3; the agent's one-liner compared the clean `out/out_bit.npy`; the
   next call re-checked every file):
   ```
   {"ok": true, "result": {"n_rows": 256, "out_bit": "out/out_bit_2.npy"}}
   tool flips 0
   {"ok": true, "result": {"n_rows": 256, "out_bit": "out/out_bit_3.npy"}}
   tool flips 0
   ---
   out/out_bit.npy 0
   out/out_bit_2.npy 96
   out/out_bit_3.npy 0
   ```

## Files

- **Committed:** this file, and `runs/boolintermediates/20261002-smoke1_opus55/summary.md`. The summary is
  answer-free: per-instance pass/n only.
- **Uncommitted, per D6:**
  - `config.json`.
  - `summary.json`. Its per-tier slot counts over valid episodes include a single-episode T4 group, so they reveal
    one instance's null-slot count.
  - `episodes/*`: grade.json, submission.json, tool logs and transcripts, plus ep3bd6b501bc's stale operator
    transcript copy.
- **Operator scratch, not committed:** the agent-only re-audits, the transcript dumps, and the finish operators'
  first-run backups.
- Nothing was pushed. task/boolintermediates has no upstream.
