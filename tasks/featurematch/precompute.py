"""GPU pass: max-pooled SAE activations of every concept text (splits A and B) and every concept-name probe.

Writes (gitignored, regenerable; see MANIFEST.md):
  cache/acts_L{6,12,18}_A.npy      float16 [N_A_total, 16384]   held-out split, one row per text
  cache/acts_L{6,12,18}_B.npy      float16 [N_B_total, 16384]   probe split
  cache/acts_L{6,12,18}_names.npy  float16 [n_concepts * 3, 16384]  naive "encode the concept name" probes
  cache/precompute_meta.json       row -> concept index maps, name-probe strings, SAE sanity check

Run: $PY -m common.gpuq run --gb 8 --heavy --label fm-precompute -- $PY -m tasks.featurematch.precompute [names C B A]
  cache/acts_L{6,12,18}_C.npy      float16 [N_C_total, 16384]   reference-solver probe split (offline simulation only)
"""
import json
import os
import sys
import time

import numpy as np
import torch

from common import gpuq
gpuq.apply_caps()

from tasks.featurematch import concepts as C          # noqa: E402
from tasks.featurematch.fm_core import LAYERS, Subject  # noqa: E402

BATCH = 32


def name_probes(c):
    """The naive token-identity recipe: encode the concept's name. Three surface variants."""
    short = c["label"].replace("article about an ", "").replace("article about a ", "").replace("text written in ", "")
    return [c["label"], short, f"This is about {short}."]


def run(subj, texts):
    out = {L: [] for L in LAYERS}
    for i in range(0, len(texts), BATCH):
        r = subj.maxpool_acts(texts[i:i + BATCH])
        for L in LAYERS:
            out[L].append(r[L])
    return {L: np.concatenate(v) for L, v in out.items()}


def sanity(subj):
    """Fraction of variance explained (FVE) and L0 of the SAE when fed hidden_states[L+1] vs hidden_states[L].
    One text at a time (no padding: padded positions would pollute the statistics)."""
    from tasks.featurematch.fm_core import sae_path
    texts = ["The quick brown fox jumps over the lazy dog near the river bank.",
             "Lionel Messi is an Argentine professional footballer who plays as a forward for Inter Miami."]
    res = {}
    for L in LAYERS:
        p = np.load(sae_path(L))
        b_dec = torch.tensor(p["b_dec"], device=subj.device)
        for off in (0, 1):
            fves, l0s = [], []
            for t in texts:
                enc = subj.tok(t, return_tensors="pt").to(subj.device)
                with torch.no_grad():
                    x = subj.model.model(**enc, output_hidden_states=True).hidden_states[L + off].float()[0, 1:]
                a = subj.sae_encode(x, L)
                xh = a @ subj.sae[L]["W_dec"].float() + b_dec
                fves.append(float(1 - ((x - xh) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()))
                l0s.append(float((a > 0).sum(-1).float().mean()))
            res[f"L{L}_hs{L + off}"] = {"fve": round(float(np.mean(fves)), 4), "l0": round(float(np.mean(l0s)), 1)}
    return res


def main():
    t0 = time.time()
    cs = C.load()
    subj = Subject()
    meta = {"sanity": sanity(subj)}
    print(json.dumps(meta["sanity"]), flush=True)
    A = [t for c in cs for t in c["A"]]
    B = [t for c in cs for t in c["B"]]
    Cc = [t for c in cs for t in c["C"]]
    names = [p for c in cs for p in name_probes(c)]
    meta.update({"rowsA": [i for i, c in enumerate(cs) for _ in c["A"]],
                 "rowsB": [i for i, c in enumerate(cs) for _ in c["B"]],
                 "rowsC": [i for i, c in enumerate(cs) for _ in c["C"]],
                 "names": names, "cids": [c["cid"] for c in cs]})
    only = set(sys.argv[1:]) or {"names", "B", "A", "C"}
    for tag, texts in (("names", names), ("C", Cc), ("B", B), ("A", A)):
        if tag not in only:
            continue
        acts = run(subj, texts)
        for L in LAYERS:
            np.save(os.path.join(C.CACHE, f"acts_L{L}_{tag}.npy"), acts[L])
        print(tag, len(texts), f"{time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(C.CACHE, "precompute_meta.json"), "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    main()
