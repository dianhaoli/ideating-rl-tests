"""Evict idle tool servers early when the GPU queue is under pressure (2026-10-01).

Why: one L4 is shared by ~9 builders. Each episode's tool server keeps its model on the GPU until it has been idle for
RL_IDLE_S (600 s). With LLM agents thinking between calls, and many gate runs in parallel, 10+ jobs were queued for
10-30 min behind servers that were mostly idle. Restarting the broker with a shorter idle limit would break the episodes in flight,
so this runs beside the broker and uses its existing `evict` admin command. An evicted server reloads on the next
call (counters live in the broker, so nothing is lost but a reload).

Policy (every 20 s): if any gpuq waiter has waited > PRESSURE_WAIT_S, evict
  * servers of episodes that are already submitted/closed, and
  * servers whose episode has made no tool call for > PRESSURE_IDLE_S (tool-log mtime).
Usage: nohup python -m common.pressure_evictor >> runs/.episodes/pressure_evictor.log 2>&1 &
"""
import json
import os
import time

from common import broker, gpuq, paths

PRESSURE_WAIT_S = float(os.environ.get("PRESSURE_WAIT_S", "60"))
PRESSURE_IDLE_S = float(os.environ.get("PRESSURE_IDLE_S", "150"))


def _waiters():
    with gpuq._locked() as led:
        return dict(led.get("waiting", {}))


def tick():
    now = time.time()
    w = _waiters()
    if not any(now - x["since"] > PRESSURE_WAIT_S for x in w.values()):
        return 0
    st = broker.admin("status")
    if not st.get("ok"):
        return 0
    n = 0
    for ep in st["result"]["episodes"]:
        if ep.get("server") != "ready":
            continue
        eid = ep["episode"]
        log = os.path.join(paths.episodes_dir(), f"{eid}.tool_log.jsonl")
        try:
            idle = now - os.path.getmtime(log)
        except OSError:
            idle = 0.0
        if ep.get("status") in ("submitted", "closed") or idle > PRESSURE_IDLE_S:
            r = broker.admin("evict", episode=eid)
            n += 1
            print(json.dumps({"t": now, "evicted": eid, "task": ep.get("task"), "status": ep.get("status"),
                              "idle_s": round(idle), "ok": r.get("ok"), "waiters": len(w)}), flush=True)
    return n


def main():
    while True:
        try:
            tick()
        except Exception as e:  # never die; the broker may be restarting
            print(json.dumps({"t": time.time(), "error": str(e)[:200]}), flush=True)
        time.sleep(20)


if __name__ == "__main__":
    main()
