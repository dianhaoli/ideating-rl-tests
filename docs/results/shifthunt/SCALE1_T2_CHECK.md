# ShiftHunt scaled run T2 and Haiku T1 arm: adversarial check of SCALE1_T2.md (2026-10-03)

Checks the review in `SCALE1_T2.md` (commit fdac559b) of `runs/shifthunt/20261003-scale1_T2_opus55` (Opus 5.5, T2,
20 episodes) and `runs/shifthunt/20261003-scale1_T1_haiku45` (Haiku 4.5, T1, 10 episodes). Aggregates only (D6,
exposed_instances.json rule): this file names no instance, episode or latent id, and does not say which attribute
belongs to which slot.

**Method.** Every number was re-derived by script from the run dirs (grade.json, episode.json, submission.json,
audit.json, tool_log.jsonl, transcript.jsonl), the instance answer files, and attempt 0 of reference v4.2 in
gates_v9 P1 (`episodes.jsonl`, all 53 built T2new instances). The run dirs were only read. Grading was redone in memory
with `grader.grade` (CPU, 8G memory scope), and per-latent grid shares were computed with the grader's arithmetic.
The audit was re-run with `transcript_audit.audit` on all 30 transcripts, and each flagged call was re-checked with
`check_call` after swapping only the flagged token. I re-read all 5 failed T2 episodes, their siblings where relevant,
and 5 passing episodes drawn with `random.Random(20261002).sample(sorted(passing episode ids), 5)`. I also read the
final reports of all 10 Haiku episodes. Scripts and transcript extracts lived in a gitignored directory under
`runs/shifthunt/` and were deleted. Neither shared directory (tool-results, scratchpad) was read beyond file names,
times and id/marker counts.

## Verdict

**The main verdict stands, with one qualification it does not make.** On the instances it ran, T2 does not give Opus
5.5 a clean mid-band tier: 15/20 = 75% [0.53, 0.89]. Naming is saturated (52/52 named, 0/18 null false claims, 20/20
pass with the bar at 0). Every failure is a latent-selection miss close to the bar, and outcomes are mostly a
property of the instance. Haiku 4.5 fails T1 at method discovery (0/10, and 0/10 with the bar at 0), not through
format or tool handling. Every headline number re-derives exactly, the in-memory regrade matches grade.json 30/30, and
the audit re-run reproduces every audit.json.

Corrections and additions:
1. **C1. The keep filter makes the T2 pool easier for agents that track v4.2 (new, affects the verdict).** The pool
   keeps only instances that reference v4.2 solves one-shot, and the agents track v4.2 closely: agent and reference
   margins on the same slot correlate r = 0.89 over 52 slot-episodes. I estimated how the agents would do on
   instances the filter dropped. Adding the 52 observed agent-minus-reference removal differences to v4.2's margins
   reproduces the plan (0.71-0.74 predicted, 0.75 observed). The same model gives about 0.77-0.80 on the 39 kept
   instances, about 0.31-0.33 on the 14 dropped ones, and **about 0.65-0.67 on all 53 built**. The two figures in each
   range come from two noise models. Without the filter, T2 would probably sit inside 10-70%, but near the top. The
   added failures would fall on slots the label-free reference also misses (dropped-pool reference margins -0.055 to
   -0.001), so the bar would decide them, not understanding. The non-reference gates also pass without the filter
   (<= 3/53, NOTES 01:50). SCALE1_T2.md does not quantify any of this.
2. **C2. "Every failing probe was reachable: the label-free reference cleared each one" is true by construction.**
   Every slot of a kept instance is cleared by v4.2, because that is the keep rule, and v4.2 was tuned on this pool.
   Independent evidence of reachability does exist and holds, from in-memory regrades:
   - the oracle top 20 latents remove 0.21-0.36 against bars of 0.16-0.31, with at least 0.946 of the topic kept;
   - on the 10-latent pair, adding the dropped context latent gives +0.010 and -0.0025, and filling with the best
     remaining latents by true share passes both;
   - on the 9-latent episode, its own earlier 20-latent set gives +0.005;
   - on the 20-latent misses, swapping the dropped latent in for the weakest pick gives +0.028 and +0.017.
3. **C3. More shared-directory exposure during the Haiku arm (adds to H3 and H7).** The Haiku arm reused the 10 T1
   instances. Answer-bearing T1 material sat in both directories the test agents could reach. The shared scratchpad
   (E8) holds 13 files written before 03:00 that carry T1 episode or instance ids. Among them are per-slot rows for
   all 10 instances, with planted flags and latent lists for both Opus episodes, and two transcript dumps that
   include submissions. The tool-results directory holds two T1-phase operator dumps, from 01:47 and 02:21. One of
   them includes an Opus T1 submission for the instance of a later Haiku episode, and was present when that episode
   ran. SCALE1_T1_CHECK called the scratchpad files "harmless once the 10 instances are retired". They were not
   retired before the Haiku arm.
   - No Haiku or T2 tool call touched the scratchpad or `.claude`, apart from the one Read of the agent's own
     persisted output that H3 describes. No transcript contains a canary or leak string.
   - Haiku scored 0/10, so no grade is affected. But SCALE1_T2.md lists only 4 dumps for deletion.
4. **C4. The audit's false negatives are wider than `ps` (H1 side finding).** R5 (process inspection) and the R4
   client rule (ssh/scp/nc/telnet/rsync class) fire on none of these:
   - `subprocess.run([...])`, `subprocess.run('...', shell=True)`, `os.system` and `os.popen`, in `./py -c`, in a
     heredoc to `./py` and in a Write of a .py file;
   - `subprocess.run(['top'])`, so SCALE1_T1_CHECK recommendation 1's "keep `subprocess.run(['top'])` flagging"
     rested on a twin that never fired;
   - a raw `socket.create_connection`.

   `curl`, `wget`, URLs and `/proc` reads are flagged everywhere. No harness test covers any of these shapes.
5. **C5. The 96-token channel was used by at least 7 of 20 agents, not 6 (H4).** Six say so in their reports, and all
   six passed. A seventh, a failing episode, built a truncation-aware regression in its code (end cues that survive
   the cut, regressed within class). It never mentions this in its report, and its miss was on a mid-text cue. In
   the seeded sample, the contrast also fed a null call (one episode) and a planted claim that got no probe_query
   (another). So the channel bears on null calls as well as naming. That is a further reason for the gate attack
   H4 asks for.
6. **C6. One quote in failure class (B) belongs to a different probe.** "I left out latents that also carry topic,
   such as [latent] for probe [..]" names a latent of a probe that *passed* in that episode. On the failing probe,
   the dropped latent is a dense latent that fires on body tokens. The agent's own table split its effect into a
   small cue-token part and a larger "other tokens" part, and the agent dropped it without comment. The mechanism
   SCALE1_T2.md describes is right; the quote shows the agent's general policy, not that particular choice.
7. **C7. Minor wording, Haiku.** The "removal bar at 0" view keeps the answers fixed. But the H6 misreading changed
   answers: 5 of the 6 planted "none" answers (3 in one episode, 2 in another) came from the agent failing its own
   self-made 15% / 80% window. Neither episode could have passed: one also has a null false claim, and the other
   would have needed 3 more correct names. So the conclusion holds.

## Re-derivation of SCALE1_T2.md numbers

**T2, all 20 / harness-valid 10.** These all match:
- episodes: 15/20 [0.531, 0.888] and 9/10 [0.596, 0.982]; 15/19 [0.567, 0.915] without the rule-break episode;
- mean score 0.925 / 0.967; slots right 65/70 [0.843, 0.969] / 34/35;
- planted: 52 slot-episodes (26 slots x 2); named 52/52; passed 47/52 [0.794, 0.958] / 23/24; "none" 0/52;
- null: 18 (9 x 2); false claims 0/18 [0, 0.176] / 0/11;
- 3 all-planted instances; 1st / 2nd episode per instance 8/10 and 7/10;
- pass with the bar at 0 (name, k and topic floor kept): 20/20 and 10/10.

**Margins.** These all match:
- removal margin: median +0.056 [+0.028, +0.094], min -0.062, max +0.315; valid-only +0.058 [+0.036, +0.094];
- 23/52 within 0.05 (this includes the 5 below the bar);
- agent / label-aware 0.951 (0.701-1.024); agent / label-free 1.014 (0.714-1.255); tau / label-aware 0.784
  (0.509-0.900);
- topic kept: median 0.994, min 0.852; 6/52 slots with fewer than 20 latents, 3 of them failed; accuracy SHIFT
  median 0.127 (-0.018 to 0.90);
- reference v4.2 on the plan: 10/10, 26 named, margin median +0.053, 13 within 0.05, min +0.000;
- agent minus reference removal: median +0.005 (IQR -0.012 to +0.015), below the reference on 22/52;
- oracle top 20 over agent: median 1.091, overlap median 15 of 20;
- the exact linear decomposition reproduces the grader's removal on all 52 named slots.

**Replicates and GRPO.** These all match:
- 26 pairs: removal differs by a median 0.0088 (mean 0.014, max 0.061); 13 within 0.01 and 23 within 0.03; Jaccard
  median 0.74 (q25 0.55, min 0.18);
- margin correlation 0.954; within-slot sd 0.0145 against 0.068 across slots;
- 7 / 1 / 2 per instance; implied ICC 0.73;
- beta-binomial: 0.10 / 0.18 / 0.25 / 0.31 at G = 2 / 4 / 8 / 16 (method of moments and maximum likelihood agree);
- Jeffreys shrinkage: 0.30 / 0.55 / 0.79 / 0.95; independent p = 0.75: 0.38 / 0.68 / 0.90 / 0.99.

**Bar sensitivity.** These match: tau -0.02 gives 17/20, +0.02 14/20, +0.03 11/20, +0.05 5/20; tau x0.9 gives
17/20, x1.1 13/20, x1.2 4/20.

**By cue position.** These match:
- end of text: 16/16, margin median +0.119, 0 within 0.05;
- greeting: 7/8, +0.021, 8 within 0.05;
- mid-text: 24/28, +0.046, 15 within 0.05;
- Opus T1 margin medians +0.119 / +0.029 / +0.068;
- the failing family: 2 of its 3 slots failed in both episodes, at tau / label-aware 0.88 and 0.90; its third slot
  (0.885) passed in both.

The tier attribution is a little stronger than SCALE1_T2.md says. In T1, both Opus agents passed this family's
tightest slot, at tau / label-aware 0.98 (+0.030 and +0.021).

**Budget and provenance.** These match:
- task calls 36.5 (20-54); forward 1,260 (1,034-1,813); probe_query at cap 17/20; wall clock 1,092 s; queue wait 7.5
  s; Claude Code calls 49.5 (39-60);
- task tool counts as stated; 1 failed call; longest call 2.39 s;
- 77 query units: 50 on planted probes and 27 on null probes; 16/18 null slot-episodes queried; 11/20 episodes
  queried every probe;
- 12 of the 52 planted slot-episodes were named right without any query on that probe;
- reading calls (latent_examples + latent_tokens): 4-35 per episode; the unlisted decoys are named in 19/20 final
  reports;
- git sha fd91d8b2 and template ed473483 in all 30; one masked TASK.md per arm; the window sentence in all 30; 0
  placeholders;
- 1,853 / 1,065 assistant rows, all on the stated model; 985 Bash calls (T2), and 399 Bash, 8 Write, 1 Edit, 1 Read
  (Haiku); one user text message per transcript; discovery by search with `workflow_wrapper` in 30/30; every last
  stop_reason `end_turn`.

**Haiku and the T1 comparison.** These match:
- Haiku 0/10 [0, 0.278]; mean score 0.092; slots 3/35; planted 3/22 (named 7/22); null false claims 13/13 [0.77,
  1.0]; "none" 6/22 [0.13, 0.48];
- 9 misnamed slots, 4 of whose latent sets would clear the bar on the true attribute;
- named slots: margin median -0.071, 4/7 below the bar; the four failures used 4-15 latents; agent / label-aware
  0.67; 2/7 at 20 latents;
- 2/10 encoded their own texts (15 and 24); 4/10 gave one attribute to every claim; 4/10 misread the range as the
  bar; no Haiku agent queried more than 2 probes;
- 11 failed calls (8 probe_query, 3 latent_means); valid 5/10 (4 with real /tmp use, 1 R5 false positive); the 21
  /tmp files exist, and each was written by the episode that read it;
- Opus T1 on the same instances: 19/20, 69/70, 44/44 named, 0/26, +0.072, 20/20 valid;
- Fisher 19/20 vs 0/10: p = 3.7e-7; T1 19/20 vs T2 15/20: p = 0.18;
- P(>= 15/20 | 0.55) = 0.055; P(>= 47/52 | 0.82) = 0.075; P(0/10 | 0.30) = 0.028.

**Commit fdac559b.** It adds the 20 plan ids to exposed_instances.json (all 20 present). SCALE1_T2.md has no
instance, episode or latent id. ShiftHunt tests pass (14 passed, 1 skipped; RL_SKIP_GPU=1).

## The failed episodes, re-read

All five are named right and miss on removal. The classification holds: a latent-selection error each time, and no
environment fault.
- **The 10-latent pair (both-fail instance A).** Both agents wrote their own version of the same sign-consistency
  filter (keep a latent only if the attribute's class contrast has the same sign on all three probes, using the
  minimum over probes). Both dropped the same dense latent, which ranks third by sample attribution and has 4% of the
  grid effect. Both flagged removal as the risk (estimates about 0.20-0.26).
- **Both-fail instance B.**
  - The 9-latent agent kept only cue-token latents. It said it "couldn't find more latents to add", and estimated
    about 24%.
  - The 20-latent agent ended 0.0009 short. On the failing probe it left out a dense body-token latent (C6). Its
    report called a different probe the weak point, but it also named removal on the failing probe as a risk.
- **The split instance.** The failing agent dropped two latents (2.1% and 1.0% of the grid effect) after filing them
  under other cues by their cross-topic pattern. The sibling kept both, shares 18 of 20 latents with it, and passed
  by +0.024. This episode is also the one with the real H2 rule break, the exec of its own file with `.replace`.
- **Luck at the bar.** Two of the five misses are within one within-slot sd of the bar (-0.0009 and -0.002), so a
  small selection change flips them. No *passing* episode depends on luck in the same way: its closest slot passes by
  +0.0185, and no slot of a passing episode is within 0.014 of the bar.

## Sampled passes (5 of 15, seeded draw)

All five run the pipeline SCALE1_T2.md describes:
1. confound table and within-class regressions on the free candidates;
2. exact linear split of the logit (checked with ablation, slope 1.00);
3. cross-probe gradient comparison;
4. latent reading with dashboards and per-token activations;
5. probe_query on edited sample texts;
6. top 20 latents by estimated contribution, leaving out topic and decoy latents.

Their closest slots pass by +0.021 to +0.058. On the same slots, their siblings land 0.000-0.042 away. Close calls,
all ended right:
- one null call from a combined two-attribute edit at an estimated 0.2-0.3x;
- in the same episode, two planted claims that rest on latents only (all its queries went to null probes);
- the truncation contrast in three of the five (C5).

I found no guess, prior or shortcut around latent reading.

## Misleading factors, checked

- **One generic recipe: yes, as in T1 (SCALE1_T1_CHECK C5).** Agents track the label-free reference slot by slot
  (r = 0.89). Two independent agents wrote the same selection filter. The only live judgement is whether to keep
  dense, context-carrying latents that look topical or belong to another cue. As an RL tier, T2 would mostly reinforce
  the v4.2-like pipeline plus that one judgement.
- **Luck at the removal bar: no grade turns on luck.** The bar decides how many fail (C9-style sensitivity), not
  which agent passes.
- **Pool selection: yes, it inflates (C1, C2).**
- **Exposure: none of the plan instances was exposed before the run.** All 20 are in exposed_instances.json now. The
  committed per-family aggregates of the plan set are a mild prior about answers, acceptable because the instances
  are retired.
- **Operator and test-agent context.** C3 adds the T1 material in both shared directories. None was used. The H1
  re-check:
  - each of the 9 false-positive episodes clears when only the flagged token is swapped. The exceptions are pattern
    a's second call, where the `-c` program's escaped quotes must also go, and pattern b, which needs both exec forms
    rewritten to the literal `globals()` form;
  - the must-flag twins listed still fire;
  - the H2 episode also carries one pattern-g false positive (a comment), on top of its real break.
- **Haiku failing for format or tool reasons: no.** All 10 submissions are well-formed (valid names, latent ids in
  range, no duplicates, one entry per probe), accepted first time, and every episode ended normally. Tool friction
  was real: 11 failed calls, the heredoc-replaces-pipe mistake, 3 outputs persisted to tool-results. But every
  episode fails at naming or at a null call, so each failure survives the bar at 0.

## Recommendations (amending the review's list)

1. As in the review, fix H1 with twins and re-finish the 10 episodes. Widen the false-negative fix (C4): apply the R4
   client names and the R5 process names to subprocess, os.system and os.popen arguments and to Python-written
   files, flag `socket` connections, and add twins for each shape. Keep the Python-string and comment exemptions
   (H1 h-j) for everything else.
2. Before calling T2 "not mid-band", report the pass rate the keep filter implies for unfiltered instances (C1).
   For the next T2 plan (Sonnet-class, or Opus with predictions), draw from all built instances, or from a fresh
   build with no reference keep filter. Keep v4.2's one-shot result as a reported covariate, not a filter, and state
   reachability with the oracle top 20 (C2).
3. Add the truncation-contrast attack (H4) for both naming and null calls (C5).
4. Before any further LLM run, delete the shared-directory material. Then launch test agents from a session whose
   tool-results and scratchpad hold nothing about the run's instances (C3, H3, E8).
5. Unchanged: H5 prompt fix, H6 optional wording, a middle model for the dial.

**For the user to delete** (operators may not), besides the 21 /tmp files and the 4 dumps in SCALE1_T2.md:
- in the session's `tool-results/`: `b2rrr71w2.txt` and `bod78vuyi.txt` (T1-phase operator dumps), and
  `b3v32n0en.txt`, a copy of SCALE1_T2.md that this check's first command produced when its output passed the size
  limit (committed aggregate text only);
- in the session scratchpad: `s1/`, `chk_t1/`, the two `m_<episode id>/` directories, and the loose T1 files
  `tx_epea.txt`, `ep450_tr.txt`, `cmd15.txt`, `m1.py`, `ep_margin.py` and `twins.json`.

## Files

- Committed: this file only.
- Not changed: both run dirs, SCALE1_T2.md and exposed_instances.json.
- Scratch: derive/what-if/audit-swap/selection scripts and transcript extracts in a gitignored directory under
  `runs/shifthunt/`, deleted after use. Nothing was written outside the worktree's `runs/` and `tasks/`, apart from
  the tool-results copy noted above, which the harness wrote. No canary and no `runs/.episodes` record was printed.
