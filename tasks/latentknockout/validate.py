"""Step 1-2 of the feasibility study: exactness checks of the ablation hook, then per-item behaviour validation.

Writes <out>/validated.json  {family: {group: [[entity, template_idx, prompt, clean_top1_id, margin], ...]}}
       <out>/validation_counts.json and <out>/exactness.json
"""
import json
import os
import sys
import time

import torch

from common import gpuq
gpuq.apply_caps()

from tasks.latentknockout import lk_core, lk_data  # noqa: E402

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
t0 = time.time()
S = lk_core.Subject()
print(f"loaded in {time.time()-t0:.0f}s; mem {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

# ------------------------------------------------------------------ exactness checks
probe = ["Dallas is a city in the state of", "The official language of Peru is",
         "Q: What sport does Roger Federer play?\nA:", lk_data.unrelated_texts(1)[0]]
ex = {}
for L in S.layers:
    clean, am = S.run(probe, all_pos=True)
    e_empty, _ = S.run(probe, L, S.ablate_edit(L, [[] for _ in probe]), all_pos=True)
    e_rec, _ = S.run(probe, L, S.recon_error_edit(L), all_pos=True)
    e_only, _ = S.run(probe, L, S.recon_only_edit(L), all_pos=True)
    m = am.bool()
    lp = torch.log_softmax(clean, -1)

    def kl(o):
        return (lp.exp() * (lp - torch.log_softmax(o, -1))).sum(-1)[m].mean().item()
    # ablate latents {a,b} via the fast path vs the explicit recon+error path
    x, nb, _ = S.resid(probe, L)
    f = S.sae_encode(x, L) * nb[..., None]
    top = f.sum((0, 1)).topk(3).indices.tolist()
    fast, _ = S.run(probe, L, S.ablate_edit(L, [top for _ in probe]), all_pos=True)
    slow, _ = S.run(probe, L, S.recon_error_edit(L, top), all_pos=True)
    ex[L] = dict(max_abs_dlogit_empty_set=(e_empty - clean)[m].abs().max().item(),
                 max_abs_dlogit_recon_plus_error=(e_rec - clean)[m].abs().max().item(),
                 kl_recon_plus_error=kl(e_rec), kl_recon_only_no_error=kl(e_only),
                 fast_vs_explicit_ablation_max_abs_dlogit=(fast - slow)[m].abs().max().item(),
                 fast_vs_explicit_kl=float((torch.log_softmax(slow, -1).exp() *
                                            (torch.log_softmax(slow, -1) - torch.log_softmax(fast, -1))).sum(-1)[m].mean()))
    print(L, ex[L], flush=True)
# padding check: batched (left padded) vs one at a time
single = torch.stack([S.run([p])[0] for p in probe])
batched = S.run(probe)
ex["padding_max_abs_dlogit"] = (single - batched).abs().max().item()
ex["padding_top1_agree"] = bool((single.argmax(-1) == batched.argmax(-1)).all())
print("padding", ex["padding_max_abs_dlogit"], ex["padding_top1_agree"], flush=True)
json.dump(ex, open(os.path.join(out, "exactness.json"), "w"), indent=1)

# ------------------------------------------------------------------ validation
fams = lk_data.families()
val, counts = {}, {}
for fam, F in fams.items():
    ans_ids = {g: sorted({S.first_token_id(w) for w in ws}) for g, ws in F["answers"].items()}
    items = [(g, e, ti, t.format(e=e)) for g, es in F["groups"].items() for e in es for ti, t in enumerate(F["templates"])]
    tops, margs = [], []
    for c in range(0, len(items), 64):
        ch = items[c:c + 64]
        lg = S.run([it[3] for it in ch])
        mg, tp = lk_core.margins(lg, [ans_ids[it[0]] for it in ch])
        tops += tp.tolist()
        margs += mg.tolist()
    val[fam] = {g: [] for g in F["groups"]}
    cnt = {"per_template": {}, "per_group": {}, "top_wrong_answers": {}}
    from collections import Counter
    wrong = Counter()
    for it, tp, mg in zip(items, tops, margs):
        g, e, ti, p = it
        ok = mg > 0
        pt = cnt["per_template"].setdefault(ti, [0, 0])
        pg = cnt["per_group"].setdefault(g, [0, 0])
        pt[1] += 1
        pg[1] += 1
        if ok:
            pt[0] += 1
            pg[0] += 1
            val[fam][g].append([e, ti, p, tp, mg])
        else:
            wrong[S.tok.decode([tp])] += 1
    cnt["top_wrong_answers"] = wrong.most_common(12)
    cnt["answer_ids"] = {g: [S.tok.decode([i]) for i in a] for g, a in ans_ids.items()}
    # entities that pass every template / at least 3 templates
    per_ent = Counter((g, e) for g, rows in val[fam].items() for e, *_ in rows)
    cnt["entities_passing_all_templates"] = sum(1 for v in per_ent.values() if v == len(F["templates"]))
    cnt["entities_total"] = sum(len(v) for v in F["groups"].values())
    counts[fam] = cnt
    print(fam, json.dumps({k: v for k, v in cnt.items() if k != "answer_ids"}), flush=True)

json.dump(val, open(os.path.join(out, "validated.json"), "w"))
json.dump(counts, open(os.path.join(out, "validation_counts.json"), "w"), indent=1)
print(f"done in {time.time()-t0:.0f}s; peak mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
