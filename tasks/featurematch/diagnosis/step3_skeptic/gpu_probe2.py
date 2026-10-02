"""Step-3 skeptic, second short GPU job: the SAME cheap-recipe texts as gpu_probe.py (template texts from
tmpl_meta.json and the generated completions from gen_meta.json, byte-identical strings; no new generation), max-pooled
(fm_core.Subject.maxpool_acts) for the columns of EVERY pooled latent of generator v2 (key_check.jsonl, 5017 latents,
kept or not). Lets cheap_recipes_v2.py score the recipes on the UNFILTERED v2 pool, to see whether the filter enabled
them. Output: cache/v2_tmpl_acts_L*.npy, cache/v2_gen_acts_L*.npy, cache/v2_meta.json (gitignored)."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
TASK = os.path.dirname(DIAG)
OUT = os.path.join(HERE, "cache")


def main():
    from common import gpuq
    import torch
    gpuq.apply_caps()
    from tasks.featurematch.fm_core import Subject
    t0 = time.time()
    pool = {6: set(), 12: set(), 18: set()}
    for l in open(os.path.join(DIAG, "key_check.jsonl")):
        r = json.loads(l)
        pool[int(r["layer"])].add(int(r["real_latent"]))
    pool = {L: sorted(v) for L, v in pool.items()}
    tm = json.load(open(os.path.join(OUT, "tmpl_meta.json")))
    gm = json.load(open(os.path.join(OUT, "gen_meta.json")))
    gtexts = [x if x else " " for x in gm["completions"]]
    subj = Subject(layers=(6, 12, 18), load_decoder=False)
    print("loaded", {L: len(v) for L, v in pool.items()}, f"{time.time() - t0:.0f}s", flush=True)

    def maxpool(texts):
        res = {L: [] for L in pool}
        for b in range(0, len(texts), 32):
            r = subj.maxpool_acts(texts[b:b + 32], layers=(6, 12, 18))
            for L in pool:
                res[L].append(np.asarray(r[L][:, pool[L]], dtype=np.float32))
        return {L: np.concatenate(v) for L, v in res.items()}
    A = maxpool(tm["texts"])
    G = maxpool(gtexts)
    for L in pool:
        np.save(os.path.join(OUT, f"v2_tmpl_acts_L{L}.npy"), A[L])
        np.save(os.path.join(OUT, f"v2_gen_acts_L{L}.npy"), G[L])
    # consistency: the kept columns must equal gpu_probe.py's tmpl/gen arrays (same texts, same batching)
    cons = {}
    for L in pool:
        kc = [pool[L].index(j) for j in tm["kept"][str(L)]]
        a = np.load(os.path.join(OUT, f"tmpl_acts_L{L}.npy"))
        g = np.load(os.path.join(OUT, f"gen_acts_L{L}.npy"))
        cons[L] = {"tmpl_max_abs_diff": float(np.abs(A[L][:, kc] - a).max()),
                   "gen_max_abs_diff": float(np.abs(G[L][:, kc] - g).max())}
    json.dump({"pool": pool, "consistency_with_gpu_probe": cons}, open(os.path.join(OUT, "v2_meta.json"), "w"))
    print("done", cons, f"{time.time() - t0:.0f}s", "peak cuda GB", round(torch.cuda.max_memory_allocated() / 1e9, 2),
          flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(TASK)))
    main()
