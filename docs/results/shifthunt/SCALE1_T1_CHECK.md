# ShiftHunt scaled run T1: adversarial check of SCALE1_T1.md (2026-10-03)

Checks the review in `SCALE1_T1.md` (commit 8f244320) of run `runs/shifthunt/20261003-scale1_T1_opus55`. Aggregates
only (D6, exposed_instances.json rule): this file names no instance, episode or latent id, and does not say which
attribute belongs to which slot. Quotes are redacted the same way.

**Method.** Every number was re-derived by script from the run dir (grade.json, episode.json, submission.json,
audit.json, tool_log.jsonl, transcript.jsonl), the instance answer files, and gates_v8 P2 `episodes.jsonl` (label-free
reference v3.1, attempt 0). The run dir was only read. Grading was redone in memory with `grader.grade` (CPU, 8G
memory scope); nothing was written. The audit was re-run with `transcript_audit.check_call` on every one of the 782
calls. I re-read the failed episode in full, plus 5 passing episodes drawn with
`random.Random(20261002).sample(sorted(passing episode ids), 5)`, and the final reports of all 20 episodes. The
scripts are in the session scratchpad. They are not committed, because they hold episode and latent ids.

## Verdict

**The main verdict stands.** T1 is near saturation for Opus 5.5, not mid-band: 19/20 = 95% [0.76, 0.99]. Every
headline number re-derives exactly, and no environment fault changed a grade:
- an in-memory regrade of all 20 submissions matches grade.json 20/20;
- all 6 INVALIDs are the audit false positives described, and each one disappears when the token is swapped.

Corrections and additions (details below):
1. **C1. The failure mode recurred** in a second episode, and that episode survived by luck at the bar. A second agent, on a different instance, made the same
   choice: it dropped a high-contribution latent because of its SAE-level topic cost, without discounting for the error
   term. Its slot passed by only **+0.016**. With that latent it would pass by +0.055 (in-memory regrade), and the
   other agent on that instance kept it and passed by +0.055. SCALE1_T1.md's statement "the only judgement that
   differed between two agents on the same instance is the one that failed" is therefore wrong. The pattern hit 2 of
   20 episodes: one failed by 0.004 and one passed by 0.016.
2. **C2. E6 undercounts.** 10 of the 20 finish operators echoed the episode canary into their own session output,
   not 2. They did it by printing or `cat`-ing the global episode record. The canary still reached no sandbox, run-dir file or commit.
3. **C3. New finding E8: a shared-scratchpad channel.** The test agents' context gives a scratchpad path that the
   orchestrator, operators and builder share. During the run, operators wrote answer-bearing files there.
   - Twice, a full transcript dump of an instance's first episode, including its submitted answer, was sitting there
     before the second episode on that instance started.
   - No test agent read it, and any read would have been flagged R2 and made the episode INVALID.
   - SCALE1_T1.md's E4 does not mention this channel.
4. **C4. The suggested E1 fix (d) is not enough.** Making the match case-sensitive fixes `TOP`. But a lowercase
   `top = {...}` line in a heredoc script that calls `subprocess.run` also fires R5; I tested it.
5. **C5. "The work that passes is real interpretability" needs a qualifier.** It is a real causal method, but it is
   one generic pipeline, the same one the label-free reference script runs one-shot (16/16 on T1b):
   - all 20 agents used it, every planted answer lists exactly 20 latents, and every report describes them as the
     largest contributors, ranked with or without a topic penalty;
   - agent / label-free removal has median 1.001 (IQR 0.994-1.010);
   - the margin median is +0.072 for both the agents and the reference.

   This strengthens the saturation verdict. Used as an RL tier, T1 would mostly reinforce one recipe.
6. **C6. Close calls the review missed.**
   - One *planted* slot was named from an estimate in the D1 gap (0.32), on proximity alone. That is one judgement
     away from the "opposite error" PREDICTIONS.md warned about.
   - At least 2 null calls rested on secondary arguments while the agent's own best estimate sat in the gap.

   All of these ended right.
7. **C7. Wording of the failure analysis.**
   - The 1.65-1.73x topic factor was measured with probe_query on two *other* probes of that episode. On the failed
     probe the agent had a sample-level slope of 1.58x.
   - "Scaled topic loss 0.11-0.18, well inside the rule" is true of the pooled estimates. The agent's worst text
     batch scales to about 0.22 (kept about 0.78).

   The classification stands: an agent judgement error, not an environment fault. But it was a defensible risk call
   that turned out wrong, not a plain arithmetic slip.
8. **C8. The GRPO variance figures are upper bounds.** The review gives 9.5% / 34% / 56% at G = 2 / 8 / 16. Since
   1 - p^G - (1-p)^G is concave in p, any spread of p across instances lowers them, and the misses here cluster.
9. **C9. Sensitivity to the bar (new).** Shifting every tau by +0.02 gives 18/20 passes, +0.03 gives 16/20, +0.05
   gives 11/20; scaling every tau by 1.1 gives 17/20 and by 1.2 gives 10/20. The 95% depends on where the frozen
   bar sits relative to what the generic pipeline removes.

## Re-derivation of SCALE1_T1.md numbers

All of these match: 19/20 [0.764, 0.991]; valid 14/14 [0.785, 1.0]; mean score 0.9875 / 1.0; slots 69/70 [0.923,
0.997] / 49/49; planted 43/44 [0.882, 0.996] / 31/31; named 44/44; null false claims 0/26 [0, 0.129] / 0/18 [0,
0.176]; planted "none" 0/44 [0, 0.080] / 0/31; 1st / 2nd episode per instance 10/10, 9/10; 9 both / 1 split / 0 both
fail, with split scores 1.0 / 0.75; valid view 4 instances with 2 valid episodes, 6 with 1; the failed episode is one
of the 6 INVALIDs; plan 22 planted + 13 null slots, 2 all-planted instances (4 episodes).

Margins over the 44 named planted slots:
- removal margin: median +0.072, q25 +0.047, q75 +0.136, min -0.004, max +0.342;
- 13 within 0.05 of the bar, 1 below it;
- valid-only: median +0.072 [+0.051, +0.135], min +0.029, 8 within 0.05;
- agent / label-aware: median 1.006 (0.831-1.129), valid-only 1.012 (0.893-1.085);
- agent / label-free: median 1.001 (0.836-1.056), valid-only 1.003 (0.875-1.051);
- tau / label-aware: median 0.798 (0.508-0.976);
- topic kept: median 0.993, min 0.864;
- accuracy-based SHIFT: median 0.123 (-0.143 to 0.571).

Between the two episodes on the same planted slot:
- removal differs by a median 0.004 (mean 0.011, max 0.063), and by 0.01 or less on 16 of 22 slots;
- the 20-latent sets overlap with Jaccard median 0.818 (q25 0.667, min 0.429).

Budget, tools and transcripts:
- per episode: tool calls 44.5 (34-58), forward units 1978.5 (1452-2980), probe_query 4/4 in 20/20, wall clock 609.6 s
  (454-935), GPU queue wait 7.85 s (7.4-25.2), Claude Code calls 40 (29-58);
- task tool calls: latent_means 549, probe_scores 114, probe_gradient 70, sample_texts 70, budget 67, probe_query 42
  (39 accepted, 80 units), task_info 20, submit 20, latent_tokens 12, latent_examples 9, help 1;
- 7 failed calls (3 over the query budget, 3 with more than 64 texts, 1 failing the text-length check); longest call
  4.9 s excluding queue wait;
- probe_query use: all 4 units on planted probes in 12 episodes, all 4 on null probes in 6, split 2/2 in 2;
- transcripts: 1,447 assistant rows, all claude-opus-5-5; 782 calls (781 Bash, 1 Write); 1 user text message per
  episode;
- every finish audited exactly 1 transcript (search, workflow_wrapper); git sha f6a7227d in 18 episodes and b7d7e89a in 2;
  prompt sha 1bdd2846 in all 20; TASK.md and the prompt are identical after masking digits and episode ids; 0 unfilled
  placeholders;
- `git diff f6a7227d HEAD` does not touch tools.py, grader.py, sh_core.py, sh_probe.py, agent_prompt.md, taus.json
  or common/, and common/ is identical between main and the worktree.

Statistics:
- P(>= 19/20 | 0.65) = 0.0021; P(>= 43/44 | 0.90) = 0.057; P(0/26 | 0.08) = 0.114;
- 0.977^2.2 = 0.951;
- 1 - 0.95^G - 0.05^G = 0.095 / 0.185 / 0.337 / 0.56 at G = 2 / 4 / 8 / 16.

Other claims:
- **Gates:** every recipe, the black-box control and every attack pass 0/16 on T1b; the reference passes 16/16.
- **E2:** 735 of 4,480 sample texts (16.4%; per probe 2-33%) and 601 of 3,360 corpus texts (17.9%) run past 96 Gemma
  tokens. 8 of 20 agents mention the cutoff.
- **E3:** one ENOSPC line in one episode, with 4 episodes overlapping 00:20-00:31.
- **D1 restated:** 10 of 20 agents restate the D1 numbers ("about half").
- **select.py:** 5 agents named a script select.py.
- **exec:** I count 198 `exec(open('scratch/..').read())` calls in 11 episodes, 43 of them on a `.split` prefix in 9;
  the review says 199 / 44. The difference is down to the regex and does not matter.

Small clarifications:
- "13 of 44 within 0.05" uses margins.py's |margin| < 0.05, so it includes the slot below the bar; 12 of the 43
  passing slots are within 0.05 above it.
- "Agents' predicted real removal was within a few hundredths of the grader on most slots" holds only loosely. About
  two thirds of the stated predictions are within 0.05. Several miss by 0.07-0.20, where the error-term share came
  from a single text pair.

## The failed episode, re-read

The facts in SCALE1_T1.md hold:
- the slot was named right and its removal was 0.004 short of the bar (tau = 0.84 x label-aware);
- the agent's first set had 0.389 removed and 0.859 of the topic kept, and would pass;
- its forced-in alternative had 0.381 removed and 0.895 kept, and would pass;
- the other agent on the instance kept the latent (15 of 20 latents shared) and passed by +0.046;
- the label-free reference passed this slot by +0.059.

Precision (C7):
- The agent's error-term measurements with probe_query went to two other probes, giving topic factors 1.65 and 1.73
  and attribute factors 1.08 and 1.25. On the failed probe it had the sample regression: true logits about 1.58x
  the SAE part.
- Its SAE-level topic loss for the full set was 0.19-0.30 when pooled, and 0.37 on its worst batch. Scaled by
  1.6-1.7, that is about 0.11-0.18 pooled and about 0.22 worst.
- On the hidden grid the real factor was about 1.9 (0.269 SAE-level, 0.141 real).
- The agent did apply the factor to removal ("about 0.3-0.4", actual 0.324), but not to topic loss. That asymmetry
  is the error.

**The same error elsewhere (C1).** A passing episode on another instance gave the same reason for dropping a latent:
> For probe [..] I left out latent [..]. It would have removed more of the [attribute] effect, but it cost about 27%
> of the topic signal outside the error term. That risked going under the 80% topic-retention requirement, given my
> uncertainty about how much of the topic signal sits in the error term.

That slot passed by +0.016, the closest pass in the run. Putting the dropped latent back in place of the agent's
20th pick gives 0.384 removed and 0.864 kept (+0.055). The other agent on that instance kept it, shares 19 of 20
latents and passed by +0.055. The label-free reference passed the same slot by +0.054, with 0.864 kept.

Both cases are on slots of the same attribute family, with tau / label-aware 0.84 and 0.87. They are also the two
largest between-agent removal gaps among slots near the bar (0.050 and 0.039). So the main source of within-instance
variance in T1 is one recurring choice: whether to keep a latent that carries the attribute and some topic signal.
Whether that choice costs the episode is decided by where the bar sits (0.004 short vs 0.016 over).

## Sampled passes (5 of 19, seeded draw)

All five run the same pipeline:
1. a template-cloning generator, with matched pairs that flip one attribute or only the topic;
2. g · Δ(latent means) as the SAE-level effect, checked against `probe_scores` ablation (slope 1.00);
3. probe_query spent on measuring the error-term share;
4. latents chosen as the top 20 by contribution, with a split-half check.

Naming, null calls and removal follow from the measurements; I found no guess, prior or shortcut. None of the five
knew the bar, and all expected removals of about 0.25-0.6, "the most any 20 latents can remove". All five finished
above the bar by +0.038 to +0.168. Their paired replicates differ by 0.000-0.012 on every slot except one well
above the bar (0.063, with both margins above +0.1). One of the five has the null call in the gap described in C6.

## Misleading factors, checked

- **One generic method solving everything: yes (C5).** It is the intended T1 route (SPEC (9)). But the agents are
  interchangeable with the label-free reference script:
  - margins: agents median +0.072 and worst +0.016 (the C1 slot); reference on the same slots median +0.072 and minimum +0.031;
  - naming: agents 44/44, reference 16/16 episodes.

  What T1 tests for this agent is whether it re-derives and runs that pipeline carefully, plus a few judgement calls.
- **Passes that relied on luck at the removal bar: one (C1).** Leaving out the two C1 pairs, the close passes
  (+0.021 to +0.048) are fixed properties of their slots:
  - both agents land within 0.01 of each other there;
  - the label-free reference clears the same slots by +0.031 to +0.047, about the same as the agents.

  They reflect where tau sits (tau / label-aware up to 0.98), not agent noise. C9 shows how much the rate depends on
  that.
- **Judgement calls in the D1 gap (C6), all correct.**
  - One planted slot was named from a corrected estimate of 0.32 ("I chose 'relies' because 0.32 is relatively
    closer to 0.4"). The slot's grid reliance is 0.53. The other agent on that instance read it at 0.48.
  - Two null calls had the agent's own estimate in the gap: 0.25-0.32 on real logits and 0.3-0.4 on SAE level. Both
    were decided on secondary evidence (the effect is spread over many latents, with no attribute-specific weights).
  - On one of those two null probes, the other agent on the instance spent all 4 queries on it and measured 0.03.
  - A third null call set aside a 0.67x reading on concatenated notes as off-distribution.
- **Exposure and leakage: none found.**
  - None of the 10 plan instances were in exposed_instances.json before the run. Their ids appear in committed text
    only in the plan, the pool lists, the manifest and the gates_v8 config (no outcomes).
  - SCALE1_T1.md has no instance, episode or latent ids. No plan canary appears in any of the 331 commits or in any
    run-dir episode file.
  - **These 10 instances are now LLM-seen and still missing from exposed_instances.json.** Add them before any other
    plan is drawn from T1b_kept (step 4 of the review).
  - The committed aggregate "22 planted / 13 null, 4 all-planted episodes" over the named plan set is a mild prior
    about answers. It is acceptable only because these instances are being retired.
- **Operator mistakes: no grade affected.** E6 is 10/20, not 2/20 (C2). E8 is the scratchpad writes (C3). All 20
  finishes audited the right single transcript with `--agent-model` recorded.
- **Test-agent context: none of it was used.** Each test agent saw:
  - the repo root as cwd;
  - main's 5 recent commit subjects, which name ShiftHunt, the T2 gate failure, the audit's Python-region rules and
    F5;
  - the account email;
  - the shared scratchpad path;
  - the parent session's tool list.

  None of it was used:
  - all 782 calls are anchored in the agent's own sandbox, and the only tools called are Bash (781) and Write (1, to its own scratch);
  - no command touches /tmp, the scratchpad, the repo, git or another episode;
  - no assistant text mentions the repo, a commit, ShiftHunt, SHIFT, the workflow or the scratchpad beyond TASK.md's
    own words.

  Limit of this check: thinking blocks are empty in the stored transcripts, so only actions and visible text could be
  checked.

## E1 re-check

The audit re-run reproduces exactly the 9 hits in audit.json: R2 ×6 in 5 episodes and R5 ×3 in 1. Each one goes
away when only the flagged token is replaced:
- `:.2f}..{` → `:.2f}-{` (2 episodes);
- `'/',` → `'of',` (2 episodes);
- `D+'/tool'` and `D+'/'+` → `os.path.join(...)` (1 episode; both hits);
- `\bTOP\b` → `TOPS` (1 episode, 3 calls).

Must-still-flag twins still fire: `os.listdir('/')` and `open('../x')` in a heredoc, and `; top -b`.

Correction to fix (d) (C4):
- With `subprocess.run` in the script, `TOP = {0: 1}` fires, and so does `top = {0: 1}`.
- Without subprocess, neither fires. `TOP=1` with no spaces does not fire.
- So the fix is to stop treating a Python assignment target or identifier as a command word when the Python-region
  exemption is switched off (or to apply that switch only inside subprocess argument lists). Case-sensitivity alone
  is not enough.

## Recommendations (amending the review's list)

1. As in the review, fix E1 and re-finish the 6 episodes with `--transcript`, but with fix (d) corrected (C4). Add
   twins for `top = x` and `TOP = {...}` in a subprocess-calling script, and keep `; top`, `| ps aux` and
   `subprocess.run(['top'])` flagging.
2. **Before the T2 run (same workflow launcher):**
   - finish, review and builder operators must not write episode material into the session scratchpad, and must not
     print `runs/.episodes/<ep>.json` (the canary, leak strings and submission);
   - give operators a private directory outside the path that test agents see;
   - add an explicit audit must-flag twin for reading the scratchpad path.

   The current T1 files there are harmless once the 10 instances are retired.
3. Add the 10 plan instances to exposed_instances.json now (the review's step 4).
4. Keep T1 as the easy tier. If it is ever made harder, C9 shows the bar dominates the pass rate (+0.05 on tau takes
   it from 95% to 55%), and C1 shows the one live judgement (keep a topic-carrying latent and discount topic loss for
   the error term) affects about 10% of episodes. Prefer the understanding-based changes the review suggests over
   moving tau.
5. Unchanged: run the scaled T2 run with predictions written first; document the 96-token window (E2).

## Files

- Committed: this file only.
- Not changed: the run dir and SCALE1_T1.md.
- Scratch (not committed, because they hold ids): derive.py, derive2.py, e1.py, ctx_scan.py and dump.py in the
  session scratchpad, under chk_t1/.
