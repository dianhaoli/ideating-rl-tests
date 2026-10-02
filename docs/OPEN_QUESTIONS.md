# OPEN QUESTIONS: uncertainties and assumptions made while Dan was away

1. **[RESOLVED 2026-10-01 14:43] GitHub push credentials were missing.** Dan configured a credential helper; push works. The machine has no gh CLI, no credential helper and no SSH key. All
   work is committed locally. Dan can provide a fine-grained PAT with contents read/write on ideating-rl-tests,
   or configure a credential helper. Retry at every checkpoint.
2. **Tier A containment** needs Dan's approval. It would add a custom Claude Code agent type,
   `~/.claude/agents/sbx-tester.md`, with `tools: Bash` and a PreToolUse hook that rewrites every command to
   `sudo -u <low-priv user>` inside the episode sandbox. That needs a change to Claude Code config, which the
   auto-mode classifier blocks for the assistant. Until then everything is Tier B.
3. **The hardware differs from the plan**: L4 instead of A10G, 83 GB free instead of about 150 GB. Assumed equivalent for
   0.1-8B subject models.
4. **The repo is public.** Instance answer keys stay out of git until their episodes finish (D6). Should the repo be private?
5. **Test-agent models**: main = the session's Opus-class model; small = Haiku-class via the subagent model option.
   These are Claude Code subagents, not API runs. Every number is labelled that way.

## 2026-10-02 04:10 UTC (orchestrator, after the crash)
- BoolIntermediates is READY_FOR_SMOKE: the only task with all scripted gates passing on 40/tier pools. It is the strongest
  "one validated environment" candidate for Oct 8, but an agent smoke has not run. Should it run now (fresh Claude Code subagents,
  fresh instances), in parallel with the FeatureMatch diagnosis? The disputed unitdecode finding means it may be too
  easy. Smoke would settle that.
- ResidualRecall T2 rebuild (rank 3-4, ~45 min heavy GPU): do it now, or after LatentKnockout?
