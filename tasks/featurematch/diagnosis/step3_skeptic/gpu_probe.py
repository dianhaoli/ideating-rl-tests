"""Step-3 skeptic, GPU job (one common.gpuq job, <= 8 GB): activations the CPU analysis needs.

  1. tmpl: fm_core.Subject.maxpool_acts (the generator's per-text summary: max over non-BOS tokens, 64-token
     truncation) of every cheap-recipe text (cheap_texts.py) for all 232 concepts plus template_probe's 24 background
     texts; only the columns of the A1.1-KEPT latents are stored (both the evaluation pool and the calibration pool
     draw slot latents from the kept set only).
  2. gen: greedy completions (48 new tokens) of self_probe's 2 fixed prompts for every concept, computed exactly as
     the `generate` tool does (one prompt at a time, model_service.op_generate), then their maxpool activations.
  3. sr_live: SR's primary 120 bank-R texts of every slot of instances_v2f sent the way a live recipe would send them
     through `latent_activations`: Subject.token_acts on batches of 16 texts (option-major order), only the slot's
     latent, max over tokens, rounded with tools.r4. This measures the offline-cache vs live gap that SR_RECIPE.md
     left unquantified.
  4. tmpl_live: the same live path for the style3 and generic_bare texts of 60 seeded slots.
Outputs go to step3_skeptic/cache/ (gitignored: per-text activations of key latents reveal concept-latent pairs).
"""
import glob
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
TASK = os.path.dirname(DIAG)
OUT = os.path.join(HERE, "cache")
FM_CACHE = os.path.expanduser("~/wt/featurematch/tasks/featurematch/cache")
SEED = 20261002


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def kept_latents():
    kept = {6: set(), 12: set(), 18: set()}
    for l in open(os.path.join(DIAG, "key_check.jsonl")):
        r = json.loads(l)
        if r["kept"]:
            kept[int(r["layer"])].add(int(r["real_latent"]))
    return {L: sorted(v) for L, v in kept.items()}


def main():
    from common import gpuq
    import torch
    gpuq.apply_caps()
    from tasks.featurematch.fm_core import Subject
    from tasks.featurematch.tools import r4
    from tasks.featurematch.diagnosis.step3_skeptic import cheap_texts as CT
    from tasks.featurematch.diagnosis.step3_skeptic.sr_indep import read_bank, load_slots
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    meta = json.load(open(os.path.join(DIAG, "cache", "bank_meta.json")))
    cids = meta["cids"]
    labels = {c["cid"]: c["label"] for c in json.load(open(os.path.join(FM_CACHE, "concepts.json")))}
    kept = kept_latents()
    log("kept latents per layer", {L: len(v) for L, v in kept.items()})
    subj = Subject(layers=(6, 12, 18), load_decoder=False)
    log("model loaded", f"{time.time() - t0:.0f}s", "cuda mem GB", round(torch.cuda.memory_allocated() / 1e9, 2))

    # ---------------- 1. template texts
    texts, tags = [], []
    gen_jobs = []
    for i, c in enumerate(cids):
        tf = CT.texts_for(labels[c])
        for kind in ("generic_label", "generic_bare", "style3", "encyc4"):
            for j, t in enumerate(tf[kind]):
                texts.append(t)
                tags.append([kind, c, j])
        for j, p in enumerate(tf["gen_prompts"]):
            gen_jobs.append((c, j, p))
    for j, t in enumerate(CT.BACKGROUND):
        texts.append(t)
        tags.append(["background", None, j])

    def maxpool(texts):
        res = {L: [] for L in kept}
        for b in range(0, len(texts), 32):
            r = subj.maxpool_acts(texts[b:b + 32], layers=(6, 12, 18))
            for L in kept:
                res[L].append(np.asarray(r[L][:, kept[L]], dtype=np.float32))
        return {L: np.concatenate(v) for L, v in res.items()}
    A = maxpool(texts)
    for L in kept:
        np.save(os.path.join(OUT, f"tmpl_acts_L{L}.npy"), A[L])
    json.dump({"texts": texts, "tags": tags, "kept": kept, "cids": cids}, open(os.path.join(OUT, "tmpl_meta.json"), "w"))
    log("templates", len(texts), f"{time.time() - t0:.0f}s")

    # ---------------- 2. generate (exactly the tool's op: one prompt, truncation 256, greedy)
    from tasks.featurematch.model_service import op_generate
    comps = []
    for k, (c, j, p) in enumerate(gen_jobs):
        comps.append(op_generate(subj, prompt=p, max_new_tokens=CT.GEN_MAX_NEW)["completion"].strip())
        if k % 50 == 0:
            log("generate", k, "/", len(gen_jobs))
    gtexts = [x if x else " " for x in comps]
    G = maxpool(gtexts)
    for L in kept:
        np.save(os.path.join(OUT, f"gen_acts_L{L}.npy"), G[L])
    json.dump({"jobs": gen_jobs, "completions": comps, "kept": kept}, open(os.path.join(OUT, "gen_meta.json"), "w"))
    log("generate done", f"{time.time() - t0:.0f}s", "empty completions", sum(not x for x in comps))

    # ---------------- 3. SR live
    btexts, rows, styles = read_bank("R", cids)
    style_names = sorted(set(styles))
    chosen = random.Random(SEED).sample(style_names, 6)
    first = {}
    for i, (r, s) in enumerate(zip(rows, styles)):
        first.setdefault((r, s), i)
    sel = {c: [first[(c, s)] for s in chosen] for c in range(len(cids))}
    cidx = {c: i for i, c in enumerate(cids)}
    slots = load_slots()

    def live(texts, layer, latent):
        vals = []
        for b in range(0, len(texts), 16):
            acts, _ = subj.token_acts(texts[b:b + 16], layer, [latent])
            vals += [r4(a[:, 0].max()) if len(a) else 0.0 for a in acts]
        return vals
    V = np.zeros((len(slots), 20, 6), dtype=np.float32)
    for si, s in enumerate(slots):
        tl = []
        for c in s["menu"]:
            tl += [btexts[i] for i in sel[cidx[c]]]
        V[si] = np.array(live(tl, s["layer"], s["latent"]), dtype=np.float32).reshape(20, 6)
        if si % 100 == 0:
            log("sr_live", si, "/", len(slots), f"{time.time() - t0:.0f}s")
    np.savez(os.path.join(OUT, "sr_live.npz"), vals=V,
             ids=np.array([f"{s['iid']}:{s['slot']}" for s in slots]))
    log("sr_live done", f"{time.time() - t0:.0f}s")

    # ---------------- 4. template live on 60 seeded slots
    pick = sorted(random.Random(SEED).sample(range(len(slots)), 60))
    out = {}
    for si in pick:
        s = slots[si]
        for kind in ("generic_bare", "style3"):
            tl, own = [], []
            for j, c in enumerate(s["menu"]):
                t = CT.texts_for(labels[c])[kind]
                tl += t
                own += [j] * len(t)
            out[f"{s['iid']}:{s['slot']}:{kind}"] = {"vals": live(tl, s["layer"], s["latent"]), "owner": own}
    json.dump(out, open(os.path.join(OUT, "tmpl_live.json"), "w"))
    log("all done", f"{time.time() - t0:.0f}s", "peak cuda GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(TASK)))
    main()
