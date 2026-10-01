"""AUDIT harness solver: the model-free `learned_prior` policy from offline_attacks.py, run through
`common.sandbox run-scripted` (no tool server, no GPU: it reads the slot list from TASK.md, like the model-free recipes).

It applies the gradient-boosted option model + null model trained by offline_attacks.py on in-memory instances from
seeds 600000+ (disjoint from the pool), saved in tasks/featurematch/audit/cache/lp_<tier>.pkl (regenerable, not
committed). The tier is read from the budget in TASK.md the agent sees (forward cap 550 => T3) and from the menu
structure (T1 menus are uniform over the universe, T2 menus are taxonomy-clustered); an agent can see both.
Uses the concept taxonomy (DBPedia path / language family, from concepts.json) as a proxy for what an LLM infers from
the labels. No instance file, answer or held-out data is read.

Run: python -m common.sandbox run-scripted --task featurematch --solver tasks/featurematch/audit/learned_prior_solver.py \
        --solver-label audit_learned_prior --instances ... --profile full --run-dir runs/featurematch/<ts>_audit_learned_prior
"""
import json
import os
import pickle
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))


class Tax:
    def __init__(self):
        from tasks.featurematch import concepts as C
        self.cs = C.load()
        self.paths = [c["path"] for c in self.cs]
        n = len(self.paths)

        def pre(a, b):
            k = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                k += 1
            return k
        self.close = np.array([[pre(self.paths[i], self.paths[j]) for j in range(n)] for i in range(n)])


def tier_guess(task_md, info, T, lab2c):
    m = re.search(r"forward[^0-9]{0,40}(\d+)", task_md)
    if m and int(m.group(1)) <= 600:
        return "T3"
    # T1 menus: 20 uniform draws -> low mean taxonomy closeness; T2: clustered
    cl = []
    for s in info["slots"]:
        idx = [lab2c[o] for o in s["options"]]
        c = T.close[np.ix_(idx, idx)].astype(float)
        np.fill_diagonal(c, np.nan)
        cl.append(np.nanmean(c))
    return "T2" if np.mean(cl) >= 2.0 else "T1"


def solve(call, task_md_path="TASK.md"):
    from tasks.featurematch.audit.offline_attacks import option_rows, slot_feats
    from tasks.featurematch.reference_solver import slots_from_task_md
    info = slots_from_task_md(task_md_path)
    T = Tax()
    lab2c = {c["label"]: i for i, c in enumerate(T.cs)}
    tier = tier_guess(open(task_md_path).read(), info, T, lab2c)
    m = pickle.load(open(os.path.join(HERE, "cache", f"lp_{tier}.pkl"), "rb"))
    answers = []
    for s in info["slots"]:
        rows = option_rows(T, lab2c, s["layer"], s["options"], m["prior"])
        q = m["mo"].predict_proba(np.array(rows))[:, 1]
        q = q / q.sum()
        pn = m["ms"].predict_proba(np.array([slot_feats(T, lab2c, s["layer"], s["options"], q)]))[0, 1]
        best = int(np.argmax(q))
        answers.append({"slot": s["slot"], "choice": "nothing found" if pn >= (1 - pn) * q[best] else best + 1})
    call("submit", answers=answers)
    return {"tier_guess": tier, "answers": answers}


def main():
    from common.toolclient import episode_from_argv
    from tasks.featurematch.reference_solver import client, unwrap
    ep = episode_from_argv()
    print(json.dumps(solve(unwrap(client(ep).call))))


if __name__ == "__main__":
    main()
