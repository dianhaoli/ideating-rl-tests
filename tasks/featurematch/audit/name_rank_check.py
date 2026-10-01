"""AUDIT: where does the answer's own NAME rank among the menu's name-probe activations on planted slots?

The generator rejects a planted slot when the answer's name is the TOP name-probe activation (any of 3 surface
variants). That filter is asymmetric (never applied to null slots) and removes only rank 1, so the answer's name often
lands at rank 2: an artifact a learned policy can exploit (see attack `name_rank2`). CPU only, precomputed tables;
in-memory instances from seeds 620000+ (disjoint from the pool). Also reports how often ANY option name fires, by slot
kind (a null-vs-planted statistic). Run: $PY -m tasks.featurematch.audit.name_rank_check
"""
import collections

import numpy as np

from tasks.featurematch.generate import Tables, make_instance


def main(n=300):
    T = Tables()
    cid2c = {c["cid"]: i for i, c in enumerate(T.cs)}
    ranks, nz, tot = collections.Counter(), collections.Counter(), collections.Counter()
    for tier in ("T1", "T2"):
        for k in range(n):
            _, inst, _ = make_instance(T, 620000 + k + 100000 * (tier == "T2"), tier)
            for x in inst["extra"]["slots"]:
                L, j = x["layer"], x["real_latent"]
                menu = [cid2c[m] for m in x["menu"]]
                s = np.array([max(T.names[L][3 * c + v, j] for v in range(3)) for c in menu])
                tot[(tier, x["kind"])] += 1
                nz[(tier, x["kind"])] += int(s.max() > 0)
                if x["kind"] == "planted" and s.max() > 0:
                    cs = menu.index(cid2c[x["anchor"]])
                    r = "zero" if s[cs] <= 0 else int((s > s[cs]).sum()) + 1
                    ranks[(tier, r if r == "zero" or r <= 3 else "4+")] += 1
    for key in sorted(tot):
        print(key, "slots", tot[key], "any option name fires %.2f" % (nz[key] / tot[key]))
    print("planted slots with any name firing, rank of the answer's name:", sorted(ranks.items(), key=str))


if __name__ == "__main__":
    main()
