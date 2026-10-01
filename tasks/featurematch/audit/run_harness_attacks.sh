#!/bin/bash
# AUDIT: run the GPU attack solvers through the real harness (common.sandbox run-scripted: broker, leak scan,
# out-of-process grader), serially, one 7 GB tool server at a time. Usage (from the worktree root, after env.sh):
#   bash tasks/featurematch/audit/run_harness_attacks.sh <ts> <instance_dir> [<instance_dir> ...]
set -u
TS=$1; shift
for V in template_probe density_null self_probe; do
  RL_ATTACK=$V $PY -m common.sandbox run-scripted --task featurematch \
    --solver tasks/featurematch/audit/attack_solvers.py --solver-label audit_$V \
    --instances "$@" --profile full --run-dir runs/featurematch/${TS}_audit_harness_$V
done
