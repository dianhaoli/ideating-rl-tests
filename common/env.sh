# Source me: `source ~/ideating-rl-tests/common/env.sh` (from a task worktree root or the main checkout).
# Never print environment variables: the HF token is read by huggingface_hub from $HF_HOME/token.
export HF_HOME="$HOME/hf_home"
export PY=/opt/pytorch/bin/python
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::FutureWarning"
export PYTHONPATH="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
