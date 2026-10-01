"""Machine-wide locations shared by every checkout (main repo and every task worktree).

Why this exists: task builders work in git worktrees (~/wt/<task>). Code that used
"the directory this file lives in" as the repo root ended up with one GPU-queue ledger
and one episode registry PER WORKTREE, so the queue was not actually machine-wide.
Everything that must be shared machine-wide is anchored on the MAIN checkout, which we
find from git's common directory (a worktree's `.git` is a file pointing into it).

All locations can be overridden by environment variables (tests use this to run an
isolated broker):
    RL_SANDBOX_ROOT   default ~/rlsbx            (one sub-directory per episode)
    RL_BROKER_SOCK    default $RL_SANDBOX_ROOT/.broker.sock
    RL_EPISODES_DIR   default <main repo>/runs/.episodes   (privileged episode records)
    RL_PY_VENV        default ~/rlsbx/.venv      (analysis venv behind each sandbox's ./py)
    GPUQ_DIR          default <main repo>/runs/.gpuq
"""
import os

HERE_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main_repo(start=HERE_REPO):
    """Return the main checkout for `start`, even when `start` is a git worktree."""
    dotgit = os.path.join(start, ".git")
    if os.path.isfile(dotgit):
        try:
            with open(dotgit) as f:
                line = f.read().strip()
            if line.startswith("gitdir:"):
                gd = line.split(":", 1)[1].strip()
                # <main>/.git/worktrees/<name>  ->  <main>
                parts = os.path.normpath(gd).split(os.sep)
                if "worktrees" in parts:
                    i = len(parts) - 1 - parts[::-1].index("worktrees")
                    return os.sep.join(parts[: i - 1]) or start
        except OSError:
            pass
    return start


MAIN_REPO = main_repo()


def sandbox_root():
    return os.path.abspath(os.path.expanduser(os.environ.get("RL_SANDBOX_ROOT", "~/rlsbx")))


def broker_sock():
    return os.environ.get("RL_BROKER_SOCK") or os.path.join(sandbox_root(), ".broker.sock")


def episodes_dir():
    return os.environ.get("RL_EPISODES_DIR") or os.path.join(MAIN_REPO, "runs", ".episodes")


def venv_dir():
    return os.path.abspath(os.path.expanduser(os.environ.get("RL_PY_VENV", "~/rlsbx/.venv")))


def gpuq_dir():
    return os.environ.get("GPUQ_DIR") or os.path.join(MAIN_REPO, "runs", ".gpuq")
