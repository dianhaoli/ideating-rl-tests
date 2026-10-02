"""Shortcut hunt (GPU): cheap solvers against the recommended LatentKnockout-Verify cells.

For every recommended cell (layer 12/18; 44 feasible + 29 strict-null cells from the feasibility sweep) this rebuilds
the instance EXACTLY as sweep.py did (same rng draws; asserted against the stored held-out item list), then builds
latent sets with cheap recipes that never search and never look at held-out data, and scores every set on the
grader's held-out items (R_S = Effect_S x Preserve_S x KL factor, same as analyze.metrics).

Recipes (k <= 5 latents unless the name says otherwise):
  contr_k{1..5}, naive_k{1..5}     top-k gradient x activation (contrastive / plain), from the stored score lists
  pref_k{1..5}                     the reference's greedy prefixes (so a verifier script can choose among them)
  cos_k{1..5}                      decoder cosine with the last-token mean difference
  actdiff_k5                       mean summed activation on target examples minus controls (SAE encode only)
  actdiff_last_k5                  same, last token only
  ent_mostact_k5                   most active on the entity's OWN tokens in target examples (no contrast)
  ent_actdiff_k5                   entity-token activation, target minus control entities
  last_mostact_k5                  most active at the answer position of target examples (no contrast)
  fmt_k5                           most active on TEMPLATE (non-entity) tokens over all example prompts (format cues)
  ll_raw_k5                        logit lens: decoder . unembed(answer)  (no prompts at all; "say Austin" latents)
  ll_contr_k5                      logit lens, target answer minus mean sibling answer
  ll_active_k5                     ll_contr restricted to latents active on the target examples
  dla_k5                           last-token activation x ll_contr score (direct-logit attribution)
  mem_seed1                        the reference set of the SAME cell found on another example split (seed 1, L18)
  mem_xfam                         the reference set of the same group in the sister family (city_state<->city_capital,
                                   langid<->country_lang), same layer: an ID-memoriser keyed on the answer concept
Self-tests (for verifier scripts, never graded): every candidate is also scored on
  A = example entities x the held-out templates (agent writes new styles for the entities it was shown)
  B = held-out entities x the example templates (agent brings its own list of, e.g., Texas cities)
with siblings built the same way, plus KL on the reference's own 8 proxy paragraphs.

usage: python -m tasks.latentknockout.skeptic_shortcuts.hunt <validated.json> <sweep_dir> <out_dir> [--layers 18 12]
"""
import argparse
import glob
import json
import os
import random
import resource
import time
import zlib

import numpy as np
import torch

from common import gpuq
gpuq.apply_caps()

from tasks.latentknockout import lk_core, lk_data  # noqa: E402
from tasks.latentknockout.analyze import boot_R, metrics  # noqa: E402
from tasks.latentknockout.lk_core import Edit, PromptSet  # noqa: E402
from tasks.latentknockout.sweep import strat_sample  # noqa: E402


def recommended(sweep_dir):
    """(feasible, null) cells at L12/L18 as in FEASIBILITY.md section 14 (44 + 29)."""
    feas, null = [], []
    for f in sorted(glob.glob(os.path.join(sweep_dir, "cells", "*.json"))):
        c = json.load(open(f))
        if "margins" not in c or c["layer"] not in (12, 18) or c["family"] == "country_capital":
            continue
        if not (c["n"]["tS"] >= 20 and c["n"]["sS"] >= 40):
            continue
        r5 = metrics(c, "ref_k5")["R_S"]
        names = [n for n in c["set_names"] if n.startswith("ref_k") or n in ("contr_k5", "naive_k5", "cos_k5")]
        best = max(names, key=lambda n: metrics(c, n)["R_S"])
        if r5 >= 0.5:
            feas.append(c)
        elif metrics(c, best)["R_S"] < 0.5 and boot_R(c, best)[1] < 0.5:
            null.append(c)
    return feas, null


def entity_mask(tok, text, ent):
    """Boolean per token (with BOS) marking tokens that overlap the entity's characters."""
    enc = tok(text, return_offsets_mapping=True)
    spans, s = [], text.find(ent)
    while s >= 0:
        spans.append((s, s + len(ent)))
        s = text.find(ent, s + 1)
    m = np.zeros(len(enc["input_ids"]), bool)
    for j, (a, b) in enumerate(enc["offset_mapping"]):
        if b > a and any(a < e1 and b > s1 for s1, e1 in spans):
            m[j] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("validated")
    ap.add_argument("sweep")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, nargs="+", default=[18, 12])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    feas, null = recommended(a.sweep)
    label = {(c["family"], c["group"], c["layer"]): ("feasible" if c in feas else "null") for c in feas + null}
    allc = {(c["family"], c["group"], c["layer"]): c for c in feas + null}
    # every seed-0 cell (also the dropped middle ones) is a potential memorisation source
    src = {}
    for f in glob.glob(os.path.join(a.sweep, "cells", "*__s0.json")):
        c = json.load(open(f))
        if "sets" in c:
            src[(c["family"], c["group"], c["layer"])] = c["sets"]["ref_k5"]
    seed1 = {}
    for f in glob.glob(os.path.join(a.sweep, "cells_seeds", "*.json")):
        c = json.load(open(f))
        if "sets" in c:
            seed1[(c["family"], c["group"], c["layer"])] = c["sets"]["ref_k5"]
    sister = {"city_state": "city_capital", "city_capital": "city_state", "langid": "country_lang",
              "country_lang": "langid"}

    S = lk_core.Subject(layers=tuple(sorted(set(a.layers))))
    fams = lk_data.families()
    val = json.load(open(a.validated))
    wiki = lk_data.unrelated_texts(64, n_tokens_chars=400)
    U = S.model.lm_head.weight
    gnorm = 1.0 + S.model.model.norm.weight.float()
    log = open(os.path.join(a.out, "progress.log"), "a")

    for L in a.layers:
        kl_ps = PromptSet(S, L, wiki[8:32], None, max_len=64)
        klp_ps = PromptSet(S, L, wiki[40:48], None, max_len=48)
        S.clean_hidden_all(kl_ps)
        S.clean_hidden_all(klp_ps)
        Wd = S.sae[L]["W_dec"]
        for fam in ["city_state", "city_capital", "country_lang", "athlete_sport", "langid"]:
            F = fams[fam]
            ans = {g: lk_core.ids_for(S, w) for g, w in F["answers"].items()}
            # logit-lens score of every latent for every group's answer: [groups, 16384]
            gl = list(F["groups"])
            ll = {}
            for g in gl:
                u = (U[ans[g]].float() * gnorm).mean(0)
                ll[g] = (Wd @ u).cpu().numpy()
            V = val[fam]
            rng0 = random.Random(1000 * 0 + 7)
            ents = {}
            for g in F["groups"]:
                es = sorted({r[0] for r in V[g]})
                rng0.shuffle(es)
                ents[g] = es
            n_ex = {g: (4 if len(es) >= 12 else 3) for g, es in ents.items()}
            ex_e = {g: set(ents[g][:n_ex[g]]) for g in ents}
            ho_e = {g: set(ents[g][n_ex[g]:]) for g in ents}
            for g in F["groups"]:
                key = (fam, g, L)
                if key not in allc:
                    continue
                fn = os.path.join(a.out, f"{fam}__{g.replace(' ', '_')}__L{L}.json")
                if os.path.exists(fn):
                    continue
                c = allc[key]
                tc = time.time()
                rng = random.Random(zlib.crc32(f'{fam}|{g}|{L}|0'.encode()))
                others = [h for h in F["groups"] if h != g]
                EX_T, HO_T = lk_data.EX_T, lk_data.HO_T
                ex_t = [(g, r) for r in V[g] if r[0] in ex_e[g] and r[1] in EX_T]
                ex_c = strat_sample([(h, r) for h in others for r in V[h] if r[0] in ex_e[h] and r[1] in EX_T],
                                    lambda x: x[0], 24, rng)
                ho_tT = [(g, r) for r in V[g] if r[0] in ho_e[g] and r[1] in HO_T]
                ho_tS = [(g, r) for r in V[g] if r[0] in ho_e[g] and r[1] in lk_data.HO_S]
                rng.shuffle(ho_tT)
                rng.shuffle(ho_tS)
                ho_tT, ho_tS = ho_tT[:30], ho_tS[:80]
                ho_sT = strat_sample([(h, r) for h in others for r in V[h] if r[0] in ho_e[h] and r[1] in HO_T],
                                     lambda x: x[0], 40, rng)
                ho_sS = strat_sample([(h, r) for h in others for r in V[h] if r[0] in ho_e[h] and r[1] in lk_data.HO_S],
                                     lambda x: x[0], 100, rng)
                ho_items = ([("tT", x) for x in ho_tT] + [("tS", x) for x in ho_tS] + [("sT", x) for x in ho_sT]
                            + [("sS", x) for x in ho_sS])
                got = [[r, x[0], x[1][0], x[1][1]] for r, x in ho_items]
                stored = [it for it in c["items"] if it[0] != "ks"]
                assert got == stored, f"instance rebuild mismatch for {key}"
                # self-test sets (never graded): A = example entities x held-out templates; B = held-out entities x
                # example templates; siblings built the same way from the other groups
                srng = random.Random(zlib.crc32(f'{fam}|{g}|{L}|selftest'.encode()))
                HO_ALL = HO_T + lk_data.HO_S
                vA_t = [(g, r) for r in V[g] if r[0] in ex_e[g] and r[1] in HO_ALL]
                vA_s = strat_sample([(h, r) for h in others for r in V[h] if r[0] in ex_e[h] and r[1] in HO_ALL],
                                    lambda x: x[0], 60, srng)
                vB_t = [(g, r) for r in V[g] if r[0] in ho_e[g] and r[1] in EX_T]
                srng.shuffle(vB_t)
                vB_t = vB_t[:40]
                vB_s = strat_sample([(h, r) for h in others for r in V[h] if r[0] in ho_e[h] and r[1] in EX_T],
                                    lambda x: x[0], 60, srng)
                st_items = ([("At", x) for x in vA_t] + [("As", x) for x in vA_s] + [("Bt", x) for x in vB_t]
                            + [("Bs", x) for x in vB_s])
                st_ps = PromptSet(S, L, [x[1][2] for _, x in st_items], [ans[x[0]] for _, x in st_items])
                ho_ps = PromptSet(S, L, [x[1][2] for _, x in ho_items], [ans[x[0]] for _, x in ho_items])
                # ------------- per-prompt SAE activations of the examples (one prompt at a time: no padding)
                def acts_of(items):
                    out = []
                    for gg, r in items:
                        x, am, pos, nb = S.resid([r[2]], L)
                        f = S.sae_encode(x[0].float(), L)          # [T, 16384]
                        f[0] = 0                                    # BOS
                        em = torch.tensor(entity_mask(S.tok, r[2], r[0]), device=f.device)
                        assert em.shape[0] == f.shape[0]
                        out.append((f, em))
                    return out
                with torch.no_grad():
                    At = acts_of(ex_t)
                    Ac = acts_of(ex_c)

                    def agg(A, fn):
                        return torch.stack([fn(f, em) for f, em in A]).mean(0)
                    sum_t = agg(At, lambda f, em: f.sum(0))
                    sum_c = agg(Ac, lambda f, em: f.sum(0))
                    last_t = agg(At, lambda f, em: f[-1])
                    last_c = agg(Ac, lambda f, em: f[-1])
                    ent_t = agg(At, lambda f, em: f[em].mean(0) if em.any() else torch.zeros_like(f[0]))
                    ent_c = agg(Ac, lambda f, em: f[em].mean(0) if em.any() else torch.zeros_like(f[0]))
                    fmt = agg(At + Ac, lambda f, em: f[1:][~em[1:]].mean(0))
                    n_ent_tok = [int(em.sum()) for _, em in At + Ac]
                    t_ps = PromptSet(S, L, [x[2] for _, x in ex_t], [ans[g]] * len(ex_t))
                    c_ps = PromptSet(S, L, [x[2] for _, x in ex_c], [ans[h] for h, _ in ex_c])
                    v = S.mean_last(t_ps) - S.mean_last(c_ps)
                    cos = torch.nn.functional.cosine_similarity(Wd, v[None, :], dim=1).cpu().numpy()
                llc = ll[g] - np.mean([ll[h] for h in others], 0)
                active = (sum_t > 0).cpu().numpy()

                def top(score, k=5, mask=None):
                    s = np.array(score, np.float64)
                    if mask is not None:
                        s = np.where(mask, s, -np.inf)
                    return [int(i) for i in np.argsort(-s)[:k]]
                sets = {}
                contr_top = [i for i, _ in c["contr_score_top"]]
                naive_top = [i for i, _ in c["naive_score_top"]]
                greedy = [t["latent"] for t in c["greedy"]]
                for k in range(1, 6):
                    sets[f"contr_k{k}"] = contr_top[:k]
                    sets[f"naive_k{k}"] = naive_top[:k]
                    sets[f"cos_k{k}"] = top(cos, k)
                    if len(greedy) >= k:
                        sets[f"pref_k{k}"] = greedy[:k]
                sets["actdiff_k5"] = top((sum_t - sum_c).cpu().numpy())
                sets["actdiff_last_k5"] = top((last_t - last_c).cpu().numpy())
                sets["ent_mostact_k5"] = top(ent_t.cpu().numpy())
                sets["ent_actdiff_k5"] = top((ent_t - ent_c).cpu().numpy())
                sets["last_mostact_k5"] = top(last_t.cpu().numpy())
                sets["fmt_k5"] = top(fmt.cpu().numpy())
                sets["ll_raw_k5"] = top(ll[g])
                sets["ll_contr_k5"] = top(llc)
                sets["ll_active_k5"] = top(llc, mask=active)
                sets["dla_k5"] = top(last_t.cpu().numpy() * llc)
                if (fam, g, L) in seed1:
                    sets["mem_seed1"] = seed1[(fam, g, L)]
                if fam in sister and (sister[fam], g, L) in src:
                    sets["mem_xfam"] = src[(sister[fam], g, L)]
                sets = {n: s for n, s in sets.items() if s}
                names = ["clean"] + list(sets)
                edits = [Edit()] + [Edit.lat(sets[n]) for n in sets]
                # dedupe identical sets
                kk, uniq, uidx = {}, [], []
                for e in edits:
                    t = (e.kind, tuple(sorted(e.ids)))
                    if t not in kk:
                        kk[t] = len(uniq)
                        uniq.append(e)
                    uidx.append(kk[t])
                mg_u, _ = S.eval_last(ho_ps, uniq)
                kl_u = S.eval_kl(kl_ps, uniq)
                st_u, _ = S.eval_last(st_ps, uniq)
                klp_u = S.eval_kl(klp_ps, uniq)
                res = dict(family=fam, group=g, layer=L, label=label[key], sets=sets, set_names=names,
                           items=[[r, x[0], x[1][0], x[1][1]] for r, x in ho_items],
                           margins={n: [round(float(m), 2) for m in mg_u[uidx[i]]] for i, n in enumerate(names)},
                           kl={n: [round(float(q), 5) for q in kl_u[uidx[i]]] for i, n in enumerate(names)},
                           st_items=[[r, x[0], x[1][0], x[1][1]] for r, x in st_items],
                           st_margins={n: [round(float(m), 2) for m in st_u[uidx[i]]] for i, n in enumerate(names)},
                           st_klproxy={n: round(float(klp_u[uidx[i]].mean()), 5) for i, n in enumerate(names)},
                           greedy=c["greedy"], n_ent_tok_min=int(min(n_ent_tok)),
                           seconds=round(time.time() - tc, 1))
                json.dump(res, open(fn, "w"))
                msg = f"{time.strftime('%H:%M:%S')} {fam} {g} L{L} {label[key]} {res['seconds']}s"
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
                del ho_ps, st_ps, t_ps, c_ps, At, Ac
                torch.cuda.empty_cache()
        del kl_ps, klp_ps
        torch.cuda.empty_cache()
    print(f"done {time.time()-t0:.0f}s peak_gpu {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
          f"peak_rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
