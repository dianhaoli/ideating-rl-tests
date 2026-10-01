"""Tiny GPU admission queue shared by every job on this machine.

Why: one NVIDIA L4 (23 GB) is shared by many builder agents and many test-agent
tool servers. Each job declares how much GPU memory it needs; a job is admitted
only while the sum of admitted jobs stays under TOTAL_GB and the per-class job
limits hold. "heavy" jobs (training, batch evaluation, reference/baseline sweeps)
are limited to MAX_HEAVY at once (the plan's "at most 2 GPU jobs" rule applied to
compute-heavy work); "light" jobs (per-episode tool servers doing small
inference) are limited by memory and MAX_JOBS. See docs/DECISIONS.md.

State lives in <main repo>/runs/.gpuq/ledger.json (shared by all worktrees) guarded by an fcntl lock. Entries whose pid
is dead are garbage-collected on every check, so a crashed job never leaks a slot.

CLI (blocks until admitted, then runs the command and releases on exit):
    python -m common.gpuq run --gb 6 [--heavy] [--ram-gb 8] [--label NAME] -- python train.py ...
    python -m common.gpuq status

Python (for long-lived processes such as tool servers):
    from common import gpuq
    with gpuq.slot(gb=4, heavy=False, label="toolserver:freqhunt"):
        ...
Inside the admitted process call common.gpuq.apply_caps() right after importing
torch to cap this process at its declared memory (torch per-process fraction).
"""
import argparse
import contextlib
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time

try:
    from common.paths import gpuq_dir as _gpuq_dir
except ImportError:  # imported without the repo on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common.paths import gpuq_dir as _gpuq_dir

# One ledger for the whole machine: anchored on the MAIN checkout even when this file is
# imported from a task worktree (see common/paths.py and docs/DECISIONS.md D11).
QDIR = _gpuq_dir()
LEDGER = os.path.join(QDIR, "ledger.json")
LOCK = os.path.join(QDIR, "ledger.lock")

TOTAL_GB = float(os.environ.get("GPUQ_TOTAL_GB", "21"))   # L4 has ~22.5 GB usable
PHYS_GB = 23.0
MAX_JOBS = int(os.environ.get("GPUQ_MAX_JOBS", "7"))
MAX_HEAVY = int(os.environ.get("GPUQ_MAX_HEAVY", "2"))
POLL_S = 3.0
AGE_S = float(os.environ.get("GPUQ_AGE_S", "300"))   # waiters older than this get space reserved (anti-starvation)


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextlib.contextmanager
def _locked():
    os.makedirs(QDIR, exist_ok=True)
    with open(LOCK, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            try:
                with open(LEDGER) as f:
                    led = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                led = {"jobs": {}}
            led["jobs"] = {k: v for k, v in led["jobs"].items() if _alive(int(k))}
            led["waiting"] = {k: v for k, v in led.get("waiting", {}).items() if _alive(int(k))}
            yield led
            tmp = LEDGER + ".tmp"
            with open(tmp, "w") as f:
                json.dump(led, f, indent=1)
            os.replace(tmp, LEDGER)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _task_key(label):
    """Task a job belongs to, from its label prefix ("freqhunt-train-g97" -> "freqhunt"; "sp-gen" -> "sp")."""
    return re.split(r"[-:_ ]", label or "?", maxsplit=1)[0].lower()


def _try_admit(pid, gb, heavy, label):
    """Admit if memory/job limits allow. Heavy-slot fairness (added 2026-10-01 after one task held both heavy
    slots for 30+ min while another task's heavy job waited): a task that already holds a heavy slot cannot take a
    second one while a heavy job from a DIFFERENT task is waiting. Waiters register in led["waiting"]."""
    with _locked() as led:
        jobs = led["jobs"]
        waiting = led.setdefault("waiting", {})
        used = sum(j["gb"] for j in jobs.values())
        n_heavy = sum(1 for j in jobs.values() if j["heavy"])
        if str(pid) in jobs:
            waiting.pop(str(pid), None)
            return True
        key = _task_key(label)
        blocked = used + gb > TOTAL_GB or len(jobs) >= MAX_JOBS or (heavy and n_heavy >= MAX_HEAVY)
        if heavy and not blocked:
            mine = any(j["heavy"] and _task_key(j.get("label")) == key for j in jobs.values())
            others_waiting = any(w["heavy"] and w["key"] != key for k, w in waiting.items() if k != str(pid))
            blocked = mine and others_waiting
        if not blocked:
            # Aging (2026-10-01, reported by the featurematch and latentdiff builders): big light jobs (6-7 GB tool
            # servers) starved for 30+ min because 2-5 GB jobs kept slipping into every gap. A waiter that has waited
            # longer than AGE_S, and is older than this request, gets its memory and a job slot reserved: this request
            # is admitted only if it still fits next to that reservation. Aged heavy waiters reserve only when a heavy
            # slot is actually free (otherwise they could not start anyway).
            me = waiting.get(str(pid))
            my_since = me["since"] if me else time.time()
            now = time.time()
            reserve_gb, reserve_n = 0.0, 0
            for k, w in waiting.items():
                if k == str(pid) or now - w["since"] < AGE_S or w["since"] >= my_since:
                    continue
                if w["heavy"] and n_heavy >= MAX_HEAVY:
                    continue
                reserve_gb += w["gb"]
                reserve_n += 1
            if reserve_n and (used + gb + reserve_gb > TOTAL_GB or len(jobs) + 1 + reserve_n > MAX_JOBS):
                blocked = True
        if blocked:
            waiting.setdefault(str(pid), {"key": key, "heavy": heavy, "gb": gb, "label": label, "since": time.time()})
            return False
        waiting.pop(str(pid), None)
        jobs[str(pid)] = {"gb": gb, "heavy": heavy, "label": label, "since": time.time()}
        return True


def acquire(gb, heavy=False, label="", pid=None, timeout=None, verbose=True):
    """Block until admitted. Returns seconds waited. Raises TimeoutError."""
    if gb > TOTAL_GB:
        raise ValueError(f"job wants {gb} GB > queue total {TOTAL_GB} GB")
    pid = pid or os.getpid()
    t0 = time.time()
    announced = False
    while not _try_admit(pid, gb, heavy, label):
        if verbose and not announced:
            print(f"[gpuq] waiting for {gb} GB ({'heavy' if heavy else 'light'}) label={label}", file=sys.stderr, flush=True)
            announced = True
        if timeout is not None and time.time() - t0 > timeout:
            raise TimeoutError(f"gpuq: not admitted within {timeout}s")
        time.sleep(POLL_S)
    os.environ["GPUQ_GB"] = str(gb)
    return time.time() - t0


def release(pid=None):
    pid = pid or os.getpid()
    with _locked() as led:
        led["jobs"].pop(str(pid), None)


@contextlib.contextmanager
def slot(gb, heavy=False, label="", timeout=None):
    acquire(gb, heavy=heavy, label=label, timeout=timeout)
    try:
        yield
    finally:
        release()


def apply_caps():
    """Cap this process's CUDA memory at its declared GB (no-op without torch/CUDA)."""
    gb = float(os.environ.get("GPUQ_GB", "0") or 0)
    if gb <= 0:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(min(1.0, gb / PHYS_GB))
    except Exception:
        pass


def _tree_rss_gb(pid):
    try:
        import psutil
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
        tot = 0
        for q in procs:
            try:
                tot += q.memory_info().rss
            except Exception:
                pass
        return tot / 1e9
    except Exception:
        return 0.0


def status():
    with _locked() as led:
        jobs = led["jobs"]
        waiting = dict(led.get("waiting", {}))
    used = sum(j["gb"] for j in jobs.values())
    print(f"admitted {len(jobs)} jobs, {used:.1f}/{TOTAL_GB} GB, heavy {sum(1 for j in jobs.values() if j['heavy'])}/{MAX_HEAVY}")
    for pid, j in jobs.items():
        print(f"  pid={pid} gb={j['gb']} heavy={j['heavy']} label={j['label']} age={time.time()-j['since']:.0f}s")
    for pid, w in waiting.items():
        print(f"  WAITING pid={pid} gb={w['gb']} heavy={w['heavy']} label={w['label']} waited={time.time()-w['since']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gb", type=float, required=True)
    r.add_argument("--heavy", action="store_true")
    r.add_argument("--ram-gb", type=float, default=float(os.environ.get("GPUQ_RAM_GB", "8")))
    r.add_argument("--label", default="")
    r.add_argument("--timeout", type=float, default=None, help="max seconds to wait for admission")
    r.add_argument("argv", nargs=argparse.REMAINDER)
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "status":
        return status()
    argv = a.argv[1:] if a.argv and a.argv[0] == "--" else a.argv
    if not argv:
        sys.exit("gpuq run: missing command after --")
    waited = acquire(a.gb, heavy=a.heavy, label=a.label or " ".join(argv)[:80], timeout=a.timeout)
    if waited > 1:
        print(f"[gpuq] admitted after {waited:.0f}s", file=sys.stderr, flush=True)
    env = dict(os.environ, GPUQ_GB=str(a.gb))
    proc = subprocess.Popen(argv, env=env)
    # transfer the ledger entry to the child so the slot follows the real job
    with _locked() as led:
        ent = led["jobs"].pop(str(os.getpid()), None)
        if ent:
            led["jobs"][str(proc.pid)] = ent
    try:
        while proc.poll() is None:
            rss = _tree_rss_gb(proc.pid)
            if rss > a.ram_gb:
                print(f"[gpuq] RAM cap exceeded ({rss:.1f} > {a.ram_gb} GB); killing job", file=sys.stderr, flush=True)
                proc.send_signal(signal.SIGTERM)
                time.sleep(5)
                if proc.poll() is None:
                    proc.kill()
                break
            time.sleep(2)
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
    finally:
        rc = proc.wait()
        release(proc.pid)
        release()
    sys.exit(rc)


if __name__ == "__main__":
    main()
