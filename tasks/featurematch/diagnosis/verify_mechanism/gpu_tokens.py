"""Verify-mechanism, GPU part: per-token activations of the sampled latents (question 1) and of the Q2 exemplar
latents on the texts listed in gpu_job_spec.json (written by search.py). The model is loaded once.

Per-token activations come from fm_core.Subject.token_acts, the function behind the `latent_activations` tool: the
JumpReLU SAE code of the residual after block L, BOS dropped, texts truncated to 64 tokens. Batches of at most 16 texts
(the tool's batch limit), one latent per call. Output: cache/token_acts.json.gz (per text: tokens and activations,
rounded to 4 decimals).

Run (from ~/wt/fmdiag, after sourcing common/env.sh):
  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=2G -- \
    $PY -m common.gpuq run --gb 7 --ram-gb 8 --label fm-verify-mechanism -- \
    /usr/bin/time -v $PY -m tasks.featurematch.diagnosis.verify_mechanism.gpu_tokens
"""
import gzip
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "cache", "gpu_job_spec.json")   # written by search.py (gitignored: holds split-A texts)
OUT = os.path.join(HERE, "cache", "token_acts.json.gz")
BATCH = 16


def main():
    t0 = time.time()
    import torch
    from common import gpuq
    gpuq.apply_caps()
    from tasks.featurematch.fm_core import Subject
    job = json.load(open(SPEC))
    layers = sorted({x["layer"] for x in job})
    S = Subject(layers=layers, load_decoder=False)
    print(f"model loaded in {time.time() - t0:.0f} s on {S.device}", flush=True)
    out = []
    for n, x in enumerate(job):
        texts = [t["text"] for t in x["texts"]]
        res = []
        for s in range(0, len(texts), BATCH):
            acts, toks = S.token_acts(texts[s:s + BATCH], x["layer"], [x["real_latent"]])
            for a, tk in zip(acts, toks):
                res.append({"tokens": tk, "acts": [round(float(v), 4) for v in a[:, 0]]})
        out.append({"kind": x["kind"], "layer": x["layer"], "real_latent": x["real_latent"], "c_star": x["c_star"],
                    "texts": res})
        if n % 10 == 0:
            print(f"{n + 1}/{len(job)} latents", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wt") as f:
        json.dump({"device": S.device, "torch": torch.__version__, "batch": BATCH, "latents": out}, f)
    print(f"wrote {OUT} in {time.time() - t0:.0f} s; peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
          flush=True)


if __name__ == "__main__":
    main()
