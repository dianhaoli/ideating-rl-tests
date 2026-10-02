"""Feasibility step 7 (FeatureMatch lesson 2): WHAT do the selected latents fire on?

For the latents chosen by the reference (k=5 prefix) and by naive top-5 attribution in a list of cells, scan a mixed
corpus at layer L: every validated prompt of every family (all styles) + 200 wikitext paragraphs. For each latent report
  * top-10 activating (token, left context) examples,
  * the tokens it fires on most often (counts),
  * firing frequency on wikitext tokens and on each family's prompts,
  * share of its firing that lands on the last prompt token vs inside the prompt.
usage: python -m tasks.latentknockout.latent_report <validated.json> <sweep_dir> <out.json> [--max_cells 40]
"""
import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch

from common import gpuq
gpuq.apply_caps()

from tasks.latentknockout import lk_core, lk_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("validated")
    ap.add_argument("sweep")
    ap.add_argument("out")
    ap.add_argument("--max_cells", type=int, default=60)
    a = ap.parse_args()
    cells = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.sweep, "*__s0.json")))]
    cells = [c for c in cells if "sets" in c]
    want = defaultdict(lambda: defaultdict(list))   # L -> latent -> [(cell, setname)]
    for c in cells[:a.max_cells] if a.max_cells else cells:
        for nm in ("ref_k5", "naive_k5"):
            for i in c["sets"].get(nm, []):
                want[c["layer"]][i].append(f"{c['family']}:{c['group']}:{nm}")
    val = json.load(open(a.validated))
    corpus = []
    rng = random.Random(0)
    for fam, V in val.items():
        rows = [(fam, r[2]) for g, rr in V.items() for r in rr]
        rng.shuffle(rows)
        corpus += rows[:700]
    wiki = lk_data.unrelated_texts(200, n_tokens_chars=500)
    corpus += [("wikitext", t) for t in wiki]
    S = lk_core.Subject(layers=tuple(sorted(want)))
    tok = S.tok
    out = {}
    for L, lat in want.items():
        ids = sorted(lat)
        idx_t = torch.tensor(ids, device=S.device)
        tops = {i: [] for i in ids}
        tokc = {i: Counter() for i in ids}
        famfire = {i: defaultdict(lambda: [0, 0]) for i in ids}
        lastfire = {i: [0, 0] for i in ids}
        order = sorted(range(len(corpus)), key=lambda q: len(corpus[q][1]))
        for c0 in range(0, len(order), 48):
            ch = order[c0:c0 + 48]
            texts = [corpus[q][1] for q in ch]
            x, am, pos, nb = S.resid(texts, L, max_len=128)
            with torch.no_grad():
                f = (S.sae_encode(x.float(), L, idx_t) * nb[..., None]).cpu().numpy()
            amn = am.cpu().numpy().astype(bool)
            enc_ids = S.enc(texts, 128)[0].cpu().numpy()
            for r, q in enumerate(ch):
                fam = corpus[q][0]
                valid = np.flatnonzero(amn[r])
                toks = enc_ids[r][valid]
                fr = f[r][valid]                      # [T, n_lat]
                for j, i in enumerate(ids):
                    col = fr[:, j]
                    famfire[i][fam][0] += int((col > 0).sum())
                    famfire[i][fam][1] += len(col) - 1
                    if fam != "wikitext":
                        lastfire[i][0] += int(col[-1] > 0)
                        lastfire[i][1] += int((col > 0).sum())
                    for p in np.flatnonzero(col > 0):
                        tokc[i][tok.decode([int(toks[p])])] += 1
                    p = int(col.argmax())
                    if col[p] > 0:
                        ctx = tok.decode([int(t) for t in toks[max(1, p - 8):p]])
                        tops[i].append((float(col[p]), fam, ctx, tok.decode([int(toks[p])])))
        res = {}
        for i in ids:
            tt = sorted(tops[i], reverse=True)[:10]
            res[str(i)] = dict(
                chosen_in=lat[i],
                top_examples=[dict(act=round(t[0], 2), source=t[1], context=t[2][-60:], token=t[3]) for t in tt],
                top_tokens=tokc[i].most_common(12),
                fire_rate_by_source={k: round(v[0] / max(1, v[1]), 4) for k, v in famfire[i].items()},
                share_of_prompt_firing_on_last_token=round(lastfire[i][0] / max(1, lastfire[i][1]), 3))
        out[L] = res
        print(f"L{L}: {len(ids)} latents done", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
