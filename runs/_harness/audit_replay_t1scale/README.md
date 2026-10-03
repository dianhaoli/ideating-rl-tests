# Replay of the E1 audit fix (ShiftHunt scaled run T1 false positives, 2026-10-03)

`replay_t1scale.py` re-runs the transcript audit under OLD = 5b965373 (the last commit before the fix; loaded in memory
from git, no temp file) and NEW = the working tree with the fix, on every recorded LLM test-agent transcript. Inputs are
read-only (run dirs in the main repo and all worktrees, and `~/.claude/projects`). Write-up: docs/HARNESS_VERIFICATION.md
section 11. Fix and tests: `common/transcript_audit.py`, `common/tests/test_t1scale_audit_fixes.py`.

`replay_t1scale.json` holds aggregates and, for units whose episodes were already listed in earlier replays, verdicts,
rule names, call indices and episode ids. The 20 T1 scaled-run episodes appear in aggregates only (their instances are
not retired; D6). No commands, submissions or answer material.

## Coverage: 79 transcripts

| group | n | how found | check |
|---|---|---|---|
| T1 scaled-run test agents (wf_96d109e5-27e) | 20 | first user message is the episode's agent prompt inside the documented workflow wrapper (store scan of 312 JSONL files) | each run-dir `transcript.jsonl` is byte-identical to the original (20/20); no second matching transcript |
| smoke test agents (wf_6c6e6341-0ee) | 24 | same | byte-identical run-dir copies 24/24; no second match |
| API / OpenAI episodes | 35 | run-dir `api_transcript.jsonl`, deduplicated by episode id | prompt match `exact` 35/35 |

This is the 59 transcripts of the 2026-10-02 replays plus the 20 T1 agents. The store scan found no other test-agent
transcript.

## Verdicts OLD -> NEW

| OLD -> NEW | T1 | smoke | API | total |
|---|---|---|---|---|
| clean -> clean | 14 | 21 | 33 | 68 |
| **flagged -> clean** | **6** | 0 | **1** | **7** |
| flagged -> flagged | 0 | 3 | 1 | 4 |
| **clean -> flagged** | **0** | **0** | **0** | **0** |

Call level: **13 violations removed, 0 added.** The replay switches the four exemptions off one at a time, and each
removed violation comes back with exactly one of them off (none unattributed):

| exemption | removed | where | why it was a false positive |
|---|---|---|---|
| (a) `'/'` as a whole argument of print() | 3 | T1 2 episodes; API ep3ab5ab5782 (t2ravel) c7 | `print(..., '/', len(x))` separators; nothing in those calls can hand text to a shell |
| (b) `NAME + '/x'` with NAME bound only to the sandbox path | 5 | T1 1 episode (2: `D+'/tool'`, `D+'/'+...`); smoke epc7a9376588 c4 (3: `HERE + "/tool"`, `HERE + "/scratch/c0.json"`, `.../c1.json"`) | the joined path is inside the sandbox |
| (c) f-string range `{a:.2f}..{b:.2f}` | 2 | T1 2 episodes | the '..' is the literal between two numeric format fields |
| (d) Python assignment target `TOP = {...}` | 3 | T1 1 episode, 3 calls | a binding, not the top command |

Every removed hit was read in context (`--detail`, terminal only). All are the patterns above.

## Still INVALID under NEW (all unchanged from OLD)

- ep93703cac08 (boolintermediates smoke): real hits (a /tmp read attempt, pkill, exec with `.replace`).
- ep183dbd35dd (latentdiff smoke): HTML closing tags, a URL and `docker` in probe data (out of scope, as before).
- epc7a9376588 (latentdiff smoke): `from call import tool, HERE` then `HERE + "/" + p` and `HERE+"/scratch/..."`.
  HERE comes from another module, so (b) does not resolve it. That is narrow on purpose; it stays flagged.
- epd698dfdf7e (shifthunt API): an absolute path outside the sandbox (a Python site-packages directory).

## Notes

- **Interpreters.** The replay ran under `$PY` (Python 3.13.15). The fix uses tokenize (f-string tokens differ before
  and after 3.12) and ast. Every FP and twin case of the test file was also checked under Python 3.9.25, and so were
  the 6 formerly INVALID T1 transcripts. Results are identical.
- **ep3ab5ab5782 is now VALID under NEW.** It is a recorded t2ravel API episode; its grade.json still says INVALID
  until its owner re-finishes it. That was not done here.
- **Pre-existing gap, not changed.** R5 does not see a process tool as the program of a Python call:
  `subprocess.run(['top', '-b'])`, `subprocess.run('top -b', shell=True)` and `os.system('ps aux')` pass OLD and NEW
  alike, because PROC_RX needs a shell command start before the name. `os.system('x; ps aux')` is flagged. Closing the
  gap would be a new detection with its own replay.
