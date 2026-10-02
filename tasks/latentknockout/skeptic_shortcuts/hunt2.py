"""Shortcut hunt, part 2 (GPU): "probe the concept's NAME" recipes (FeatureMatch lesson 1 style), scored on the same
held-out items as hunt.py.

  name_ans_k5   latents most active on the ANSWER word itself (" Texas", " Austin", " Spanish", " soccer") in four
                neutral sentences, minus the same for the sibling groups' answer words. No task prompt is ever run.
  name_grp_k5   the same on the GROUP name (differs from the answer only for city_capital: " Texas" for -> " Austin").
Held-out items are read back from the hunt.py cell (role, group, entity, template) and the validated prompt texts.

usage: python -m tasks.latentknockout.skeptic_shortcuts.hunt2 <validated.json> <hunt_cells_dir> [--layers 18 12]
(adds keys to each hunt cell JSON in place: sets/margins/kl for the two recipes)
"""
import argparse
import glob
import json
import os
import resource
import time

import numpy as np
import torch

from common import gpuq
gpuq.apply_caps()

from tasks.latentknockout import lk_core, lk_data  # noqa: E402
from tasks.latentknockout.lk_core import Edit, PromptSet  # noqa: E402
from tasks.latentknockout.skeptic_shortcuts.hunt import entity_mask  # noqa: E402

CONTEXTS = ["{w}", "I read an article about {w} yesterday.", "The word {w} appears in this sentence.",
            "Here is a short note on {w} and related topics."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("validated")
    ap.add_argument("hunt")
    ap.add_argument("--layers", type=int, nargs="+", default=[18, 12])
    a = ap.parse_args()
    t0 = time.time()
    S = lk_core.Subject(layers=tuple(sorted(set(a.layers))))
    fams = lk_data.families()
    val = json.load(open(a.validated))
    wiki = lk_data.unrelated_texts(64, n_tokens_chars=400)
    for L in a.layers:
        kl_ps = PromptSet(S, L, wiki[8:32], None, max_len=64)
        S.clean_hidden_all(kl_ps)
        name_cache = {}

        def name_act(word):
            if word not in name_cache:
                acc = []
                with torch.no_grad():
                    for ctx in CONTEXTS:
                        text = ctx.format(w=word)
                        x, am, pos, nb = S.resid([text], L)
                        f = S.sae_encode(x[0].float(), L)
                        f[0] = 0
                        em = torch.tensor(entity_mask(S.tok, text, word), device=f.device)
                        acc.append(f[em].mean(0))
                name_cache[word] = torch.stack(acc).mean(0).cpu().numpy()
            return name_cache[word]
        for fn in sorted(glob.glob(os.path.join(a.hunt, f"*__L{L}.json"))):
            c = json.load(open(fn))
            if "name_ans_k5" in c["sets"]:
                continue
            tc = time.time()
            fam, g = c["family"], c["group"]
            F = fams[fam]
            ans = {h: lk_core.ids_for(S, w) for h, w in F["answers"].items()}
            others = [h for h in F["groups"] if h != g]
            new = {}
            sa = name_act(F["answers"][g][0]) - np.mean([name_act(F["answers"][h][0]) for h in others], 0)
            new["name_ans_k5"] = [int(i) for i in np.argsort(-sa)[:5]]
            sg = name_act(g) - np.mean([name_act(h) for h in others], 0)
            new["name_grp_k5"] = [int(i) for i in np.argsort(-sg)[:5]]
            text = {(r[0], r[1]): r[2] for h in F["groups"] for r in val[fam][h]}
            items = c["items"]
            ho_ps = PromptSet(S, L, [text[(it[2], it[3])] for it in items], [ans[it[1]] for it in items])
            names = list(new)
            edits = [Edit.lat(new[n]) for n in names]
            mg, _ = S.eval_last(ho_ps, edits + [Edit()])
            # the clean row must reproduce hunt.py's clean margins (same items, same order)
            dmax = float(np.abs(mg[-1] - np.array(c["margins"]["clean"])).max())
            c.setdefault("hunt2_clean_maxdiff", dmax)
            if dmax > 0.25:
                print(f"WARNING clean margin mismatch {fn}: {dmax:.3f}", flush=True)
            kl = S.eval_kl(kl_ps, edits)
            for j, n in enumerate(names):
                c["sets"][n] = new[n]
                c["set_names"].append(n)
                c["margins"][n] = [round(float(m), 2) for m in mg[j]]
                c["kl"][n] = [round(float(q), 5) for q in kl[j]]
            json.dump(c, open(fn, "w"))
            print(f"{time.strftime('%H:%M:%S')} {fam} {g} L{L} {time.time()-tc:.1f}s {new}", flush=True)
            del ho_ps
            torch.cuda.empty_cache()
        del kl_ps
        torch.cuda.empty_cache()
    print(f"done {time.time()-t0:.0f}s peak_gpu {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
          f"peak_rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
