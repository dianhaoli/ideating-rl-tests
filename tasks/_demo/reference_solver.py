"""Reference solver for the harness demo task. Gets ONLY the episode id; derives the answer by probing.

    python tasks/_demo/reference_solver.py --episode E      (run-scripted does this)

Method: dump the table, estimate (a, b) by majority vote (robust to the single edit), and report
the one input whose table entry disagrees with a*x+b mod P, or "nothing found" if none does.
"""
import os
from collections import Counter

import numpy as np

from common.toolclient import Client, episode_from_argv


def solve(call, sandbox):
    """`call(tool, **args)` is Client(ep).call; `sandbox` is where tools write out/*.npy."""
    W = np.load(os.path.join(sandbox, call("weights")["path"])).astype(int)
    P = len(W)
    a = Counter(int((W[x + 1] - W[x]) % P) for x in range(P - 1)).most_common(1)[0][0]
    b = Counter(int((W[x] - a * x) % P) for x in range(P)).most_common(1)[0][0]
    bad = [x for x in range(P) if W[x] != (a * x + b) % P]
    if len(bad) == 1:
        return {"edited_input": int(bad[0]), "new_output": int(W[bad[0]])}
    return {"nothing_found": True}


if __name__ == "__main__":
    c = Client(episode_from_argv())
    ans = solve(c.call, c.sandbox)
    print(c.submit(ans))
