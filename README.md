# ideating-rl-tests

A portfolio of RL-environment tasks for agentic mechanistic interpretability. In each task an agent probes a
small model's internals through generic tools and must recover a planted ground truth: frequencies, edits,
backdoor triggers, feature meanings and so on. A programmatic grader then scores the answer. Each task
ships with a procedural generator, a grader, a reference solver, black-box and recipe baselines, and a
smoke test by a fresh Claude Code subagent.

Start here: `docs/INDEX.md`. Background: `CONTEXT.md`. Plan: `docs/PLAN_PROMPT.md`.
