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
