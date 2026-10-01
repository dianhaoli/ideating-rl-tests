"""CPU pass: per-(latent, concept) AUROC tables from the precomputed max-pooled activations.

AUROC(latent j, concept c) on a split = probability that a random text of concept c gets a higher max-activation
of latent j than a random text of any OTHER concept in the universe (ties count 1/2). 0.5 = no preference,
1.0 = fires on every c text above every non-c text. Negatives are the whole concept universe (fixed), so the
AUROC of a concept does not depend on which menu it appears in; that makes "best concept in the menu"
well defined.

Writes cache/auroc_L{L}_{A,B}.npy float16 [16384, n_concepts] and cache/fire_L{L}_A.npy (fraction of each
concept's texts on which the latent is > 0) and cache/firerate_L{L}_A.npy (overall fraction of texts).
Run: $PY -m tasks.featurematch.auroc
"""
import json
import os

import numpy as np
from scipy.stats import rankdata

from tasks.featurematch import concepts as C
from tasks.featurematch.fm_core import LAYERS


def auroc_table(X, rows, n_c, chunk=2048):
    """X [N, D] activations, rows [N] concept index. Returns [D, n_c] float32 AUROC."""
    rows = np.asarray(rows)
    N, D = X.shape
    ind = np.zeros((n_c, N), dtype=np.float32)
    ind[rows, np.arange(N)] = 1.0
    n_pos = ind.sum(1)[:, None]
    out = np.zeros((D, n_c), dtype=np.float32)
    for s in range(0, D, chunk):
        R = rankdata(X[:, s:s + chunk].astype(np.float32), axis=0).astype(np.float32)   # average ranks for ties
        Rc = ind @ R                                               # [n_c, chunk] sum of positive ranks
        auc = (Rc - n_pos * (n_pos + 1) / 2) / (n_pos * (N - n_pos))
        out[s:s + chunk] = auc.T
    return out


def main():
    meta = json.load(open(os.path.join(C.CACHE, "precompute_meta.json")))
    n_c = len(meta["cids"])
    for L in LAYERS:
        for split in ("A", "B"):
            X = np.load(os.path.join(C.CACHE, f"acts_L{L}_{split}.npy"))
            rows = meta["rows" + split]
            T = auroc_table(X, rows, n_c)
            np.save(os.path.join(C.CACHE, f"auroc_L{L}_{split}.npy"), T.astype(np.float16))
            if split == "A":
                fire = np.zeros((X.shape[1], n_c), dtype=np.float32)
                pos = (X > 0).astype(np.float32)
                np.add.at(fire.T, np.asarray(rows), pos)
                fire /= np.bincount(rows, minlength=n_c)[None, :]
                np.save(os.path.join(C.CACHE, f"fire_L{L}_A.npy"), fire.astype(np.float16))
                np.save(os.path.join(C.CACHE, f"firerate_L{L}_A.npy"), pos.mean(0).astype(np.float32))
            print(L, split, "done", flush=True)


if __name__ == "__main__":
    main()
