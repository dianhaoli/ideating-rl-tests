"""Feasibility step 7 (FeatureMatch lesson 2): WHAT do the selected latents fire on?

For the latents chosen by the reference (k=5 prefix), naive top-5 attribution and the mean-difference cosine top-5, scan a mixed
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
        for nm in ("ref_k5", "naive_k5", "cos_k5"):
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
        nl = len(ids)
        idx_t = torch.tensor(ids, device=S.device)
        best = np.full((nl, 10), -1.0, np.float32)          # top-10 (act) per latent
        best_ref = [[None] * 10 for _ in range(nl)]
        pair_keys = []                                      # (latent column * V + token id) for every firing
        srcs = sorted({c[0] for c in corpus})
        fire = {sname: np.zeros(nl, np.int64) for sname in srcs}
        ntok = {sname: 0 for sname in srcs}
        last_fire = np.zeros(nl, np.int64)
        prompt_fire = np.zeros(nl, np.int64)
        V = len(tok)
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
                sname = corpus[q][0]
                valid = np.flatnonzero(amn[r])
                toks = enc_ids[r][valid]
                fr = f[r][valid]                                  # [T, nl]
                on = fr > 0
                fire[sname] += on.sum(0)
                ntok[sname] += len(valid) - 1
                if sname != "wikitext":
                    last_fire += on[-1]
                    prompt_fire += on.sum(0)
                tpos, lcol = np.nonzero(on)
                pair_keys.append(lcol.astype(np.int64) * V + toks[tpos].astype(np.int64))
                mx = fr.max(0)
                am_ = fr.argmax(0)
                upd = np.flatnonzero(mx > best.min(1))
                for j in upd:
                    k = int(best[j].argmin())
                    best[j, k] = mx[j]
                    best_ref[j][k] = (q, int(valid[0]), int(am_[j]))
        keys = np.concatenate(pair_keys) if pair_keys else np.zeros(0, np.int64)
        uk, cnt = np.unique(keys, return_counts=True)
        res = {}
        for j, i in enumerate(ids):
            sel = (uk // V) == j
            kk, cc = uk[sel] % V, cnt[sel]
            o = np.argsort(-cc)[:12]
            toptok = [(tok.decode([int(kk[t])]), int(cc[t])) for t in o]
            exs = []
            for k in np.argsort(-best[j]):
                if best[j, k] <= 0 or best_ref[j][k] is None:
                    continue
                q, v0, p = best_ref[j][k]
                ids_q = tok(corpus[q][1], truncation=True, max_length=128)["input_ids"]   # unpadded: index p = valid[p]
                ctx = tok.decode(ids_q[max(1, p - 8):p])
                exs.append(dict(act=round(float(best[j, k]), 2), source=corpus[q][0], context=ctx[-60:],
                                token=tok.decode([ids_q[p]]) if p < len(ids_q) else "?"))
            res[str(i)] = dict(chosen_in=lat[i], top_examples=exs, top_tokens=toptok,
                               fire_rate_by_source={sn: round(float(fire[sn][j]) / max(1, ntok[sn]), 4) for sn in srcs},
                               share_of_prompt_firing_on_last_token=round(float(last_fire[j]) / max(1, prompt_fire[j]), 3))
        out[L] = res
        print(f"L{L}: {len(ids)} latents done", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
