# OPEN QUESTIONS: uncertainties and assumptions made while Dan was away

1. **GitHub push credentials are missing.** The machine has no gh CLI, no credential helper and no SSH key. All
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
