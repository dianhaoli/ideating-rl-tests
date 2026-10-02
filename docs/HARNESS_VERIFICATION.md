# Harness verification (independent review, 2026-10-01)

**What this is.** The shared evaluation harness in `common/` (contract: `docs/HARNESS_API.md`) was built
by one agent and then reviewed by a second agent acting as a skeptical reviewer. The reviewer wrote new
tests that attack the harness the way a test agent, or a buggy task, could. This page lists each check,
what it found, and what was fixed. Every check is a test you can re-run:

```
/opt/pytorch/bin/python -m pytest -q common/tests/test_verify_harness.py common/tests/test_verify_audit.py
```

`common/tests/verify_task/tasks/_vcheck/` is a fixture task whose tools misbehave on purpose. They leak
private data in different places, hang, crash, charge and then hang, and write arrays under private
names. Its grader can crash, hang, or write a broken grade. It is never given to an agent.

**Why it matters.** Each environment's score is only meaningful if (a) the agent cannot get the answer
without doing interpretability, (b) budgets are enforced exactly, so difficulty numbers are comparable,
and (c) a broken grader or a containment violation shows up as INVALID, not as a pass or a fail.
"Containment" here is Tier B: the agent technically *can* read the repo, so we audit afterwards.

**Result.** The builder's 87 tests all passed. The new tests found **23 failing harness checks and 32
transcript-audit failures** (29 out-of-sandbox reads that passed the audit, 2 kinds of normal work that
were wrongly flagged, and a missing check for outside symlinks in the sandbox). All of them are fixed.
The full suite, now 191 tests including the GPU smoke test, passes. `common/READY` stays.

---

## 1. Answer isolation: what can an agent reach from inside `~/rlsbx/<E>/`?

| Probe | Before | After |
|---|---|---|
| Files in the sandbox (TASK.md, `tool`, `py`, `.episode`, prompt) contain no canary, leak string, instance path/id, repo path or codename | OK | OK (test `test_sandbox_files_reveal_nothing_private`) |
| `help`, `budget`, unknown tool, private method names (`load`, `_invoke`, `write_array`), bad args, a tool crash whose exception text holds the canary and instance path, a validator crash | OK: errors are generic | OK |
| Malformed requests (not JSON, a list, a string, `{}`, null/list/path-like episode ids, non-string tool, list args) | OK: "bad request" / "unknown episode" | OK |
| **A 32 MB request** | **Accepted and echoed back.** One agent could exhaust the broker every agent shares | Refused: `request too large` (limit `RL_MAX_REQUEST_BYTES`, 8 MB) |
| **Admin commands on the socket** (`status`, `close`, `evict`, `shutdown`) | **Anyone who could reach the socket could use them.** `status` lists every episode with its **task codename** and counters. `shutdown` kills the broker for every agent | They now need a token in `<episodes dir>/.admin_token` (mode 600, privileged side). `ping` stays open |
| **Another episode's id** (sibling sandbox directory names are visible) | **Worked.** An agent could spend another episode's budget or query another instance | The broker reads the caller's working directory (`SO_PEERCRED` → `/proc/<pid>/cwd`). A call from inside sandbox A for episode B is refused as "unknown episode", and A is marked `cross_episode_access` (INVALID at finish). Callers outside every sandbox are unaffected (operators, tests) |
| Calls after submit, after finish, or after finishing without submitting | OK: "episode finished" | OK |
| **Symlink tricks in `out/`** (agent pre-creates `out/.x.npy.tmp`, or replaces `out/` with a link) | **`write_array` followed the link and overwrote the target.** The tool server runs on the privileged side, so this could overwrite files outside the sandbox (e.g. an instance's data) | `write_array` refuses an `out/` that is a symlink or not inside the sandbox. It creates its temp file with `O_EXCL|O_NOFOLLOW` under a random name and picks a final name that does not exist, not even as a dangling link |

**Why the cwd binding is the right shortcut-prevention choice.** Episode ids cannot be secret: they
are directory names next to the agent's sandbox. The transcript audit already forbids `ls ..`, but the
harness can make the rule technical for the commonest case, a process run from one's own sandbox, at
no cost to legitimate use.

## 2. Leak scan

| Case (through the broker unless noted) | Expected | Before | After |
|---|---|---|---|
| Canary nested deep in lists/dicts, as a dict key, in a model-output field, lower-cased, with spaces between characters | leak | leak | leak |
| **Canary's random tail only** (tool strips the `RLCANARY-` prefix) | leak | **not detected** | leak (the tail of ≥12 chars is a needle of its own) |
| Leak string upper-cased, non-ASCII upper-cased (`ÜNÏCODE-ΩMEGA`), as a list of numbers `[4242, 1717]` | leak | leak | leak |
| **Leak string with a quote or backslash** (`quote"d\str`) | leak | **not detected**: JSON escapes it to `quote\"d\\str` | leak: raw string leaves and keys are scanned as well as the JSON text |
| Numbers embedded in bigger numbers (`[14242, 17170]` vs `4242,1717`) | no leak | no leak | no leak |
| Leak string inside model-output fields only (twice) | counted as `behavioral_exposure`=2, not flagged | OK | OK |
| Leak string in a model field AND a non-model field | leak | leak | leak |
| Canary / leak string / instance path in a **ToolError message** | leak | canary, leak: yes; **path: no** | all three |
| **Instance directory path, instance id, task codename** in any output | leak | **not checked** | leak: the new `private_strings` (instance dir, repo paths, episode registry, instance id unless public, codename variants ≥ 6 chars) are never exempt |
| **Agent sends a guess once, tool returns it twice** | leak (new information) | **exempt**: any leak string the agent mentioned was skipped. An agent could switch the scan off by sending every candidate answer | Exemption is count-based: exempt only if the response does not contain it more often than the request |
| **Echo of a >10 KB request at finish** | not a leak | **flagged at finish**: the tool log truncates args, so the re-scan lost the exemption and called a legitimate echo a leak, which made the episode INVALID | The log marks `args_truncated`; for those entries the finish re-scan checks only canary/private strings, and the live scan (which saw the full args) decides leak strings |
| **Leak string in the name of a tool-written array** (`out/<answer>.npy`) with the response saying nothing | leak | **not checked** (file names were never scanned) | leak; file names are scanned at finish |
| Agent's own notes, or its additions to TASK.md, containing the answer | counted, not flagged | **TASK.md edits flagged as a harness leak** | Counted. TASK.md is a harness file only while it is byte-identical to the copy written at prepare |

Matching now uses Unicode NFKC normalisation plus `casefold`, and still ignores whitespace (`common/leakscan.py` docstring, points 5-9).

## 3. Caps

| Check | Result |
|---|---|
| tool_calls, forward, generate, gradient, a task-defined kind (`steps`) counted exactly over a scripted sequence. It covers: partial charges before a refused charge, bad arguments (costs a call), unknown tool (free), free built-ins, the cap reached exactly, and the log's charges summing to the counters | OK |
| **Per-call timeout after the tool charged** (charge 7 forward units, then hang) | **Before: 0 forward units counted.** The server was killed and its final message with the charges never came, so a tool that charges and then runs long gave free compute. **After:** every `charge()` is streamed to the broker immediately, and the 7 units are kept |
| Crash after charging | 3 units counted, error generic, no leak |
| Wall-clock: a call that starts in time may finish; the next call is refused; budget shows 0 remaining; submit still works | OK |
| Counters (and `started_at`) survive idle eviction **and a full broker restart** | OK |
| Two episodes, four threads, cap 6 each: exactly 6 calls succeed per episode, 8 refused, counters not shared | OK |

## 4. Grader

| Grader behaviour | Before | After |
|---|---|---|
| Crash (non-zero exit) | score 0, `grader_error`, INVALID | same |
| **Hangs** | **`finish` blocked for up to 3600 s, then raised** (harness crash) | Killed after `RL_GRADER_TIMEOUT_S` (default 3600; whole process group); score 0 + error |
| **Exits 0 without writing output** | score 0 but **`grader_error` was the empty string** | non-empty error |
| **Score `"high"`, NaN, 7.0; `pass: "yes"`; output a list** | **accepted as-is** (a NaN or 7.0 score would corrupt every mean) | rejected: score 0, pass False, `grader_error`, INVALID |
| Runs in a different process from the test, the broker and the tool server | OK | OK |

Summaries now also tolerate a non-dict `details`.

## 5. Transcript audit

The audit reads the agent's Claude Code transcript and marks the episode INVALID if any tool call stepped
outside the sandbox. I wrote 50 evasion transcripts (`EVASIONS` in `test_verify_audit.py`): **29 passed the
old audit.** I also wrote 16 normal-work commands plus two session-cwd suites. Two kinds of normal work
were wrongly flagged.

Evasions that used to pass, now INVALID:
- `..` hidden behind a variable: `ls $PWD/..`, `ls ${PWD}/../ep…`; `cd out && cd ../..` (`..` was resolved
  against the sandbox root, not the directory the command was in).
- Parent/home directory without writing a path: `${PWD%/*}`, `$(dirname $PWD)`, `$OLDPWD`, `cd -`, bare `cd`.
- Symlinks: `ln -s / scratch/r && ls scratch/r/home`, Python `os.symlink('/')`, `symlink_to(Path.home())`.
- Python escapes: `os.environ['HOME']`, `os.getenv`, `dirname(os.getcwd())`, `Path.cwd().parent`,
  `Path.home()`, `glob('/*')`, `os.system('ls /')`.
- `ls /*`.
- Obfuscation: `base64 -d | xargs ls`, `eval "$(… base64 --decode)"`, `b64decode` (new rule R9).
- Session cwd outside the sandbox: `ls <sandbox> && cat notes.txt` passed because naming the sandbox anywhere
  satisfied the old R8 rule, but `cat notes.txt` read the outside directory. Same for `cat <sandbox>/TASK.md; ls`.
- File tools: `Glob` pattern `../*/TASK.md` or `**/../../*` with path = sandbox; `Grep` glob `../**`.
- Non-file tools with paths or side effects: `Artifact` (publishes a repo file), `EnterWorktree`.

Normal work that was wrongly flagged, now VALID: `cd scratch && ./../py a.py && cd ..`, `cd out; ls ..`, a session
whose cwd is `scratch/` reading `../TASK.md`, and `echo 'the grader expects JSON' >> scratch/notes.md` (the word
"grader" now fires only as a file reference: `grader.py`, `grader/`, `*grader*`).

How it works now (`common/transcript_audit.py`):
- Each shell command is split at unquoted `&&`, `||`, `;`, `|`, `&` and newlines. Heredoc bodies stay with the
  command that opened them.
- `cd` is tracked part by part, starting from the session cwd recorded in the transcript.
- `..` is resolved against the directory each part actually runs in. Parts that run while the cwd is outside
  the sandbox may only use absolute sandbox paths. A bare `ls`, `find .`, `cat notes.txt` or `./tool` there is R8.
- Any tool's `file_path`/`path`-like argument is checked, not only the built-in file tools.

The agent prompt now also says "Do not create symlinks, read environment variables, or use encoded commands".
That way every rule is stated up front, not a trap.

**Accepted limits.** A determined agent can still hide a path from a text audit, e.g. by
building it from `chr()` codes or `os.sep.join([...])`. Tier B is a deterrent plus detection, not a
jail. `x /y` (division with a space before `/` and none after) looks like an absolute path and is
flagged. The real API-agent transcript on this machine (`runs/_demo/20261001-152109_apiplumbing`) is
still VALID under the new rules. On 483 real Bash commands from builder transcripts, the new rules fire
only on privileged builder work (editing the harness, setting up HF_HOME), never on analysis-style
`python -c` code.

Finish-time sandbox scan: a symlink in the sandbox that points outside it makes the episode INVALID
(`sandbox_symlink_outside`), even if the transcript was clean.

## 6. Fingerprinting (demo task)

I generated 40 demo instances (seed 11; both planted and null present). Planted and null instances have:
- identical file sets, public.json keys and values, instance.json keys, number of leak strings, caps, tier
  and dial, and id length;
- table.json sizes within noise (overlapping ranges, mean difference smaller than the spread).

Through the broker, both kinds give byte-identical TASK.md, identical output shapes and error texts for
`query`, `weights`, `help`, `budget`, an out-of-range query and a malformed submit, and equal `.npy` sizes.
**OK.** Not checked: response timing (the demo is too fast to matter). Real tasks must check timing themselves.

## 7. run-scripted + summarize; Wilson CI

- `run-scripted` with the reference solver: 2 planted + 2 null instances, 2 repeats each. All 8 pass and
  are valid. The black-box control: 4 episodes in the same run dir.
- `summary.json` groups by solver and profile. Its numbers match the per-episode results, including the
  null-slot false-claim rate. `summary.md` shows `[0.676, 1.000]` for 8/8. `config.json` lists both labels
  and 4 instances. Every episode dir has the required files and no canary.
- `wilson(k, n)` matches `scipy.stats.binomtest(...).proportion_ci(method="wilson")` within 2e-4 for
  n ∈ {1, 2, 5, 7, 10, 30, 100, 1000} and k ∈ {0, 1, n/3, n/2, n−1, n}. Also, wilson(5, 10) = [0.2366, 0.7634]
  and wilson(0, 0) = [0, 1]. **OK.**

## 8. Remaining risks (not fixed; judged acceptable or out of scope)

1. **Tier B containment.** The agent runs as the same Unix user as the privileged side. Everything above
   raises the cost of cheating and catches the plausible cases. It does not make cheating impossible.
   Only a separate user or container would.
2. **The cwd binding only covers processes started inside a sandbox.** An agent that first `cd`s out (an
   audit violation already) could call another episode.
3. **The live broker must be restarted to pick up these fixes.** A broker started from a task worktree that
   has not merged `main` runs that worktree's older `common/`. Builders must `git merge main`.
4. **Leak strings are only as good as the generator's choice.** Paraphrases or encodings of the answer that
   are not listed in `leak_strings` are not detected.
5. **Wall-clock includes the agent's own thinking time between calls.** This is by design, but it means an
   agent's model latency affects whether it hits the cap.

## 9. Builder-reported false positives and validity gaps (2026-10-02)

Builders reported episodes the transcript audit marked INVALID although the agent stayed inside its sandbox, and
episodes graded valid that should not count. Each false positive was first reproduced by re-auditing the recorded
transcript with the unchanged audit, then fixed with the narrowest rule change. Each fix has a WHY comment in the
code and regression tests in `common/tests/test_audit_validity_fixes.py`: the cited command must now be VALID, and
a twin that really reaches outside through the same construct must still be INVALID.

| Item | Cited episode(s) | What fired | Change |
|---|---|---|---|
| a. R9 `exec()` | latentdiff probe 2 (epb7d011c482), silentpathway probe 2 (ep29ef1a245b); also boolintermediates probes 2-3 | `exec(open('scratch/an.py').read().split("...")[0])`: the agent reusing its own helper script | exec of a literal sandbox file (optionally split at a literal marker) is allowed; it runs what `./py file` runs. Computed paths, outside files, `..`, transformed or decoded text stay R9 |
| b1. R2 sed | silentpathway probe 2 (ep29ef1a245b) | a sed script `s/...\/.../.../; /print(T('run'/d`: the `/regex/d` address read as the paths `/print`, `/d` | the regexes inside the script of a real top-level `sed` command are skipped by the path scan. Everything sed can read, write or execute stays visible. This replaces the 2026-10-01 rule that skipped every s///-shaped text anywhere, which was broader than needed |
| b2. R2 "/ (filesystem root)" | silentpathway probe 4 (ep13c6a7b805) | `grad_path.split('/')[-1]` in a heredoc script: the quoted `'/'` | a quoted `'/'` that is the first argument of a splitting/trimming/testing string method is a separator. `'/'.join`, `os.path.join('/', ...)`, `os.listdir('/')` still fire |
| c. R4 netcat | residualrecall probes 1 (ep0965d14ab3) and 3 (ep9e37e0db6f) | `...;nc=json.loads(...)`, `nc=14` at a line start: a Python variable | the client name fires by default; it is exempt only when used as a Python name (immediate `.`/`[`/`(`, `=`/comparison, separator, keyword). Tightened in the F1 follow-up — see 9b below |
| d. codename | featurematch audit MINOR 8a | the bare word "featurematch" in an agent's own file | see below |
| e. scripted solver | latentdiff audit MAJOR-2 | crashed or starved scripted solvers graded valid, score 0 | `run-scripted` passes the rc to `finish`: rc != 0 => `solver_failed`; no successful task-tool call and no submission => `no_successful_call` |
| f. infrastructure | featurematch audit MINOR 8b; latentdiff MAJOR-2 | queue timeout / tool-server death with no submission graded valid, score 0 | `infra_failure`: no submission, an infrastructure failure in the tool log, and no successful task-tool call after it. Client-disconnect counts for scripted solvers only — see 9b (F2) |

**d. The codename rule, and why.** Codename matching exists to catch private task material (spec, notes, generator,
grader, instance files) copied into the sandbox. The premise was that an agent has no legitimate way to know the
codename. That holds for paths and ids but not for the codenames: they are plain compounds of the task's subject
("featurematch", "latentdiff", "freqhunt"), and an agent can coin one, e.g. by naming a script
`featurematching.py`. So in files the agent wrote (contents and names), the bare word is now counted
(`agent_files_with_codename`, for review) but not a leak. It is still a leak (INVALID) where it can only come from the
privileged side: in harness-written files and tool-written array names, and in the agent's files in a repo-path,
module-path or canary form (`tasks/<cn>`, `runs/<cn>`, `wt/<cn>`, `tasks.<cn>`, `RLCANARY-<cn>`). Instance paths and
ids, repo paths and the canary (with its hex tail) remain leaks anywhere. Copied instance files carry the canary.
Reading the repo at all is caught by the transcript audit (R2/R3), which is the primary control. What this gives up:
a copy of task prose that mentions the codename only as a bare word is now counted rather than invalidated; the
count is in `grade.json`, so a reviewer still sees it.

**f. What counts as infrastructure.** Only log entries a tool's own code cannot produce: the broker's
"tool server failed to start" response, the event `client_disconnected_while_waiting`, and `server_stopped` with
reason `server died` / `found dead`. A tool's own exception ("tool failed (internal error)" with the server still
up), a per-call timeout and idle eviction are not infrastructure. An episode that recovered (a successful task-tool
call after the failure) and then did not submit stays an ordinary valid fail. Limitation: a broker or machine crash
that writes no log entry is not detected for LLM episodes (for scripted ones the rc catches it).

**Checks run.**
- Re-audit of all 95 recorded LLM-agent transcripts (main repo and task worktrees; read-only): 10 INVALID before,
  2 after. The 8 that changed are exactly the cited probes plus boolintermediates probes 2-3 (same exec pattern) and
  t2ravel probe 1 (a quoted `/` inside a sed s///). No episode that was VALID changed. The 2 still INVALID: a real
  absolute path outside the sandbox (shifthunt probe 4, `sys.path.insert(0, '/opt/python/...')`) and t2ravel probe 2
  (`print(a, '/', b)`: a quoted `'/'` as a print argument; not fixed, because exempting it safely would need real
  parsing of the enclosing call).
- The new false-positive tests were run against the unchanged audit: 11 of 14 fire there (the other 3 are sed uses
  the 2026-10-01 rule already allowed, kept as guards for the narrower sed rule). 20 of 21 twins fired there too; the
  21st (s///-shaped text outside a sed command) fires only with the new rule.
- Infrastructure rule on the cited run dirs (read-only): all 5 unsubmitted episodes in
  featurematch `20261001-200623_audit_harness_density_null` and latentdiff `20261001-204812_audit_rarefreq` /
  `20261001-220045_audit_rarefreq_harness` are `infra_failure`; their submitted episodes are unaffected. Over all
  8,256 recorded episodes, 28 currently-valid unsubmitted episodes would become `infra_failure`; 1 that recovered
  and then did not submit stays valid.
- Full suite: 250 passed, 1 skipped (the GPU smoke test, RL_SKIP_GPU=1).

### 9b. Follow-up: audit_replay findings F1 and F2 (2026-10-02)

The full audit replay (`runs/_harness/audit_replay_2026-10-02/`) confirmed the 8 transcript-audit flips of
5d0f6fe1 are genuine false positives and that no episode went clean -> flagged, but raised two findings that no
recorded episode hit. Both are now fixed.

**F1 — R4 network-client rule let real commands through (containment regression).** The first fix (5d0f6fe1) used
a positive lookahead ("fire only if followed by an argument-like word"), which stopped flagging real network
commands the pre-loosening rule caught: long options (`ncat --exec ...`, `rsync --archive ...`, `--rsh=ssh`),
`--`, a `\` line continuation, brace expansion (`ssh {a,b}.x`), here-strings (`telnet <<<`), and hosts beginning
with a guard keyword (`ssh in.example.com`, `nc is.gd 80`) — in commands and in written file content. Fixed by
inverting the lookahead: R4 fires by default at a command-start position (as the original `\b` rule did) and is
exempt only when the client name is a Python name (immediate `.`/`[`/`(`; `=`/`:=`/`+=`/`==`/`<=`/...; a `,` `:`
`)` `]` `}` separator; or a whole keyword). `.` is immediate-only so `scp ./file host:` still fires. Verified by
`regression_probes.py` (now a 3-way OLD/FIRST/FIXED comparison that exits non-zero if any breach case other than
the documented pre-existing gaps is unflagged by FIXED: all 10 F1 shapes re-flagged, the `nc=`/`nc.`/`(nc,`
variable uses stay exempt) and by 10 new twin cases in `test_audit_validity_fixes.py` (one of them in Write
content).

**F2 — `infra_failure` could be triggered by the agent (validity gaming).**
`client_disconnected_while_waiting` is logged whenever the client process dies while queued, including when the
agent's own `timeout N ./tool ...` expires (observed in residualrecall probe 3 at 100.01 s / 110.01 s). An LLM
agent heading for a fail could make its last call `timeout 5 ./tool <gpu tool>` during a queue and not submit,
turning a scored fail into INVALID `infra_failure` and biasing "must fail" gates and RL rewards. Fixed:
`tool_log_health` takes `client_disconnect_is_infra`; `finish()` passes `False` for LLM episodes, so the
client-disconnect event counts as infrastructure only for scripted solvers (trusted builder code, also gated by
`no_successful_call`/`solver_failed`, where the disconnect is the run-scripted driver's 1800 s wait limit). The
log cannot distinguish an agent-issued `timeout` from the agent runner's ~900 s limit, and an elapsed-time
threshold is equally forgeable, so LLM episodes count only server-side failures. Verified by a new
finish-level test (LLM client-disconnect -> not infra; LLM server failure -> infra; scripted client-disconnect ->
infra) and by re-running `replay_finish_reasons.py`: valid -> INVALID flips drop 46 -> 43 (the 3 LLM
client-disconnect-only flips removed; `ep27a091a362`, a server-side LLM failure, stays detectable and is not a
flip only because it is already INVALID for `grader_error`). Residual gap (documented, not fixed): an agent that
deliberately drives the tool server to `server died` and abandons the episode would still read as infra; it is
far harder than a `timeout` and bounded by the per-call budget and server RAM cap.

Suite after the follow-up: 262 passed, 1 skipped (`RL_SKIP_GPU=1`). Secret scan clean.
