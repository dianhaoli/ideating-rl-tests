"""Feasibility steps 1-2: exactness checks of the ablation machinery, per-item behaviour validation, and the SAE
reconstruction test (does the model still answer correctly when the layer-L residual is replaced by the SAE's
reconstruction, i.e. with the error term dropped?).

Writes into <out>/: exactness.json, validated.json, validation_counts.json, recon.json, timing.json
usage: python -m tasks.latentknockout.validate <out_dir>
"""
import json
import os
import random
import resource
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from common import gpuq
gpuq.apply_caps()

from tasks.latentknockout import lk_core, lk_data  # noqa: E402
from tasks.latentknockout.lk_core import Edit, PromptSet  # noqa: E402

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
t0 = time.time()
S = lk_core.Subject()
print(f"loaded in {time.time()-t0:.0f}s; mem {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
wiki = lk_data.unrelated_texts(64, n_tokens_chars=400)

# ------------------------------------------------------------------ 1. exactness
probe = ["Dallas is a city in the state of", "The official language of Peru is",
         "Q: What sport does Roger Federer play?\nA:",
         "Memphis, Tennessee\nDenver, Colorado\nBoise, Idaho\nLubbock,", wiki[0][:160], wiki[1][:160]]
ex = {}
for L in S.layers:
    clean, am, nb = S.run(probe, all_pos=True)
    m = am.bool()
    lp = torch.log_softmax(clean, -1)

    def cmp(o, tag, d):
        """Compare logits o with clean at all real positions, then free o."""
        d[f"max_abs_dlogit_{tag}"] = (o - clean)[m].abs().max().item()
        d[f"top1_agree_{tag}"] = bool((o.argmax(-1) == clean.argmax(-1))[m].all())
        d[f"kl_{tag}"] = (lp.exp() * (lp - torch.log_softmax(o, -1))).sum(-1)[nb.bool()].mean().item()
    d = {}
    # (a) empty-set ablation through the hook path
    o, _, _ = S.run(probe, L, [Edit.lat([]) for _ in probe], all_pos=True)
    cmp(o, "empty_set", d)
    del o
    # (b) run_from(L, cached residual) vs the full forward
    x, am2, pos2, nb2 = S.resid(probe, L)
    with torch.no_grad():
        o = S.head(S.run_from(L, x, am2, pos2, last_only=False))
    cmp(o, "run_from_vs_full", d)
    del o
    # (c) explicit x_hat + error (should reproduce clean up to fp32 rounding) and (d) fast vs explicit ablation
    s = S.sae[L]
    f = S.sae_encode(x.float(), L) * nb2[..., None]
    top = f.sum((0, 1)).topk(3).indices.tolist()
    del f

    def explicit(zero):
        def hook(_m, _i, o):
            xx = o[0] if isinstance(o, tuple) else o
            xf = xx.float()
            ff = S.sae_encode(xf, L)
            xhat = ff @ s["W_dec"] + s["b_dec"]
            err = xf - xhat
            f2 = ff.clone()
            if zero:
                f2[..., zero] = 0
            x2 = f2 @ s["W_dec"] + s["b_dec"] + err
            x2 = torch.where(nb[..., None].bool(), x2, xf).to(xx.dtype)
            return (x2,) + tuple(o[1:]) if isinstance(o, tuple) else x2
        h = S.model.model.layers[L].register_forward_hook(hook)
        try:
            ids, a_, p_, _ = S.enc(probe)
            with torch.no_grad():
                hid = S.model.model(input_ids=ids, attention_mask=a_, position_ids=p_).last_hidden_state
        finally:
            h.remove()
        return S.head(hid)
    o = explicit([])
    cmp(o, "recon_plus_error", d)
    del o
    o, _, _ = S.run(probe, L, [Edit("recon") for _ in probe], all_pos=True)
    cmp(o, "recon_only_no_error", d)
    del o
    slow = explicit(top)
    fast, _, _ = S.run(probe, L, [Edit.lat(top) for _ in probe], all_pos=True)
    d["fast_vs_explicit_ablation_max_abs_dlogit"] = (fast - slow)[m].abs().max().item()
    d["fast_vs_explicit_top1_agree"] = bool((fast.argmax(-1) == slow.argmax(-1))[m].all())
    del slow
    cmp(fast, "ablate_top3_vs_clean", d)
    d["ablated_top3_latents"] = top
    del fast, clean, lp
    torch.cuda.empty_cache()
    ex[L] = d
    print(L, d, flush=True)
single = torch.stack([S.run([p])[0] for p in probe])
batched = S.run(probe)
ex["padding_max_abs_dlogit_last_token"] = (single - batched).abs().max().item()
ex["padding_top1_agree"] = bool((single.argmax(-1) == batched.argmax(-1)).all())
print("padding", ex["padding_max_abs_dlogit_last_token"], ex["padding_top1_agree"], flush=True)
json.dump(ex, open(os.path.join(out, "exactness.json"), "w"), indent=1)

# ------------------------------------------------------------------ 2. behaviour validation (clean model)
fams = lk_data.families()
val, counts = {}, {}
for fam, F in fams.items():
    ans_ids = {g: lk_core.ids_for(S, ws) for g, ws in F["answers"].items()}
    # first-token collisions between groups would make "flip" ill-defined
    allids = Counter(i for a in ans_ids.values() for i in a)
    coll = {g: [S.tok.decode([i]) for i in a if allids[i] > 1] for g, a in ans_ids.items()}
    items = [(g, e, ti, t.format(e=e)) for g, es in F["groups"].items() for e in es
             for ti, (_st, t) in enumerate(F["templates"])]
    order = sorted(range(len(items)), key=lambda i: len(items[i][3]))
    marg = np.zeros(len(items), np.float32)
    top = np.zeros(len(items), np.int64)
    for c in range(0, len(order), 64):
        ch = order[c:c + 64]
        with torch.no_grad():
            lg = S.run([items[i][3] for i in ch])
        for r, i in enumerate(ch):
            a = ans_ids[items[i][0]]
            row = lg[r]
            acc = row[a].max()
            tmp = row.clone()
            tmp[a] = -1e9
            marg[i] = (acc - tmp.max()).item()
            top[i] = int(row.argmax())
    demo_ids = {S.first_token_id(w) for w in lk_data.FEWSHOT_DEMO_ANSWERS[fam]}
    val[fam] = {g: [] for g in F["groups"]}
    pt = defaultdict(lambda: [0, 0])
    pg = defaultdict(lambda: [0, 0])
    pe = defaultdict(lambda: [0, 0])
    wrong = defaultdict(Counter)
    copy = Counter()
    for (g, e, ti, p), mg, tp in zip(items, marg, top):
        ok = bool(mg > 0)
        for d, k in ((pt, ti), (pg, g), (pe, (g, e))):
            d[k][1] += 1
            d[k][0] += ok
        if ok:
            val[fam][g].append([e, ti, p, round(float(mg), 3)])
        else:
            wrong[ti][S.tok.decode([int(tp)])] += 1
            if int(tp) in demo_ids:
                copy[ti] += 1
    styles = [st for st, _ in F["templates"]]
    nt = len(F["templates"])
    cnt = dict(
        n_items=len(items), n_valid=int((marg > 0).sum()), acc=float((marg > 0).mean()),
        per_template={f"{ti}{styles[ti]}": [pt[ti][0], pt[ti][1], round(pt[ti][0] / pt[ti][1], 3)] for ti in range(nt)},
        per_style={st: round(sum(pt[ti][0] for ti in range(nt) if styles[ti] == st) /
                             sum(pt[ti][1] for ti in range(nt) if styles[ti] == st), 3) for st in sorted(set(styles))},
        per_group={g: [v[0], v[1], round(v[0] / v[1], 3)] for g, v in pg.items()},
        entities_total=len(pe),
        entities_all_templates=sum(1 for v in pe.values() if v[0] == v[1]),
        entities_ge_half=sum(1 for v in pe.values() if v[0] >= v[1] / 2),
        entities_zero=[f"{k[0]}:{k[1]}" for k, v in pe.items() if v[0] == 0],
        top_wrong_by_template={f"{ti}{styles[ti]}": wrong[ti].most_common(4) for ti in range(nt)},
        fewshot_copy_artefacts={f"{ti}{styles[ti]}": copy[ti] for ti in range(nt) if styles[ti] == "F"},
        answer_tokens={g: [S.tok.decode([i]) for i in a] for g, a in ans_ids.items()},
        answer_token_collisions={g: c for g, c in coll.items() if c},
        median_margin_valid=float(np.median(marg[marg > 0])) if (marg > 0).any() else None)
    counts[fam] = cnt
    print(fam, json.dumps({k: cnt[k] for k in ("acc", "per_style", "per_group")}), flush=True)
json.dump(val, open(os.path.join(out, "validated.json"), "w"))
json.dump(counts, open(os.path.join(out, "validation_counts.json"), "w"), indent=1)

# ------------------------------------------------------------------ 3. SAE reconstruction test + timing
rng = random.Random(0)
sample = []
for fam, V in val.items():
    rows = [(fam, g, r) for g, rr in V.items() for r in rr]
    rng.shuffle(rows)
    sample += rows[:150]
texts = [r[2][2] for r in sample]
ans = [lk_core.ids_for(S, fams[f]["answers"][g]) for f, g, _ in sample]
rec, timing = {}, {}
for L in S.layers:
    ps = PromptSet(S, L, texts, ans)
    t1 = time.time()
    mg, _ = S.eval_last(ps, [Edit(), Edit("recon")])
    torch.cuda.synchronize()
    timing[L] = dict(prompts=len(texts), edits=2, seconds=round(time.time() - t1, 2))
    kps = PromptSet(S, L, wiki[8:32], None, max_len=64)
    klv = S.eval_kl(kps, [Edit("recon")])[0]
    per_fam = {}
    for fam in val:
        sel = [i for i, s_ in enumerate(sample) if s_[0] == fam]
        per_fam[fam] = round(float((mg[1, sel] > 0).mean()), 3)
    rec[L] = dict(keep_clean_answer_all=round(float((mg[1] > 0).mean()), 3), per_family=per_fam,
                  sanity_clean_all_valid=round(float((mg[0] > 0).mean()), 3),
                  kl_recon_only_wikitext=round(float(klv.mean()), 4))
    print("recon", L, rec[L], timing[L], flush=True)
    del ps, kps
    torch.cuda.empty_cache()
json.dump(rec, open(os.path.join(out, "recon.json"), "w"), indent=1)
timing["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
timing["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
timing["total_s"] = round(time.time() - t0, 1)
json.dump(timing, open(os.path.join(out, "timing.json"), "w"), indent=1)
print("done", timing, flush=True)
