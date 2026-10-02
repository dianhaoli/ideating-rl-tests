# Replay of the 2026-10-02 agent-smoke harness fixes

`replay.py` re-runs the transcript audit under OLD = 190d6a64 (the last commit before the fixes) and NEW = the working
tree on every recorded LLM transcript. It also re-runs transcript discovery for the 24 smoke episodes (wf_6c6e6341-0ee)
over the whole `~/.claude/projects` store. It reads all inputs read-only. `replay.json` holds verdicts, rule names, call
indices and episode ids only: no commands, no submissions, no answer material. Write-up: docs/HARNESS_VERIFICATION.md
section 10.

| | result |
|---|---|
| episodes | 59 unique: the 24 smoke Claude Code subagents and 35 recorded API/OpenAI episodes (main repo and all worktrees) |
| verdict OLD -> NEW | 9 flagged -> clean (the smoke F2 items); **0 clean -> flagged**; 5 flagged -> flagged; 45 clean -> clean |
| call-level violations | 12 removed, **0 added** |
| still flagged | ep93703cac08 (real hits), ep183dbd35dd and epc7a9376588 (latentdiff false positives that were out of scope), epd698dfdf7e, ep3ab5ab5782 (both unchanged since the 2026-10-02 replay) |
| prompt match | 35 `exact` (API transcripts), 24 `workflow_wrapper` (smoke subagents): re-finishing any of them raises no R0 |
| discovery | OLD found 3 transcripts for 23 episodes and 4 for 1 (the agent plus operators). NEW finds exactly the agent's for 24/24 |
