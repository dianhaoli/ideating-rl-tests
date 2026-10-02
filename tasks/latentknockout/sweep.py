"""Feasibility steps 3-7: for every (family, target group, layer) cell, build an instance (examples / held-out split),
run the reference search and the baselines, and evaluate every candidate latent set on HELD-OUT prompts.

Instance (per family, group g, seed):
  * entities of g are shuffled; n_ex (3-4) are EXAMPLE entities, the rest are HELD-OUT entities.
  * example prompts  = example entities x example templates (styles P, Q: lk_data.EX_T)
  * example controls = example entities of the OTHER groups x example templates (the reference builds these itself)
  * held-out targets  T = held-out entities x new templates of the SAME styles (HO_T)
                      S = held-out entities x templates in NEW styles (dialogue, record, news, few-shot; HO_S)
  * held-out siblings = held-out entities of the other groups x HO_T / HO_S
  * unrelated text    = wikitext paragraphs (64 tokens) for KL; a disjoint 8-text set is the reference's own proxy.
  * city_capital only: "keep_state" items = city_state prompts of the target group's held-out cities (should still
    say the state): the same-entity-other-relation control (ADJUST lever in PREDICTIONS_FEASIBILITY.md section 4).
Reference (sees only examples + its own controls + proxy text): rank latents by gradient x activation of the answer
log-prob (target mean minus control mean), candidate pool = top-12 contrastive + top-4 naive, then greedy forward
selection with REAL ablations on the example objective J = Effect_ex x Preserve_ex x KLfactor_proxy, up to 10 latents.
R_ref(k) uses the prefix of size <= k with the best example J (selection never sees held-out data).
Everything is stored per item so analyze.py can recompute any metric variant (kappa, soft effect, CIs).

usage: python -m tasks.latentknockout.sweep <validated.json> <out_dir> [--layers 12 18 6] [--families ...]
       [--seed 0] [--groups g1,g2] [--refonly]
"""
import argparse
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
from tasks.latentknockout.lk_core import D_SAE, Edit, PromptSet  # noqa: E402

KS = [1, 2, 3, 5, 8, 10]
KAPPA = 0.1
ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0]


def strat_sample(rows, key, cap, rng):
    """Round-robin over key(row) so every other group is represented."""
    by = {}
    for r in rows:
        by.setdefault(key(r), []).append(r)
    for v in by.values():
        rng.shuffle(v)
    out, ks = [], sorted(by)
    while len(out) < cap and any(by[k] for k in ks):
        for k in ks:
            if by[k] and len(out) < cap:
                out.append(by[k].pop())
    return out


def J_vec(mt, m0t, mc, m0c, klsum):
    """Example objective per edit row: soft+hard effect x soft+hard preservation x KL factor (proxy)."""
    E = 0.5 * ((mt < 0).mean(1) + np.clip((m0t - mt) / m0t, 0, 1).mean(1))
    P = 0.5 * ((mc > 0).mean(1) + np.clip(mc / m0c, 0, 1).mean(1))
    return E * P * np.clip(1 - klsum / KAPPA, 0, 1), (mt < 0).mean(1), (mc > 0).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("validated")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 18, 6])
    ap.add_argument("--families", nargs="+",
                    default=["city_state", "city_capital", "country_lang", "athlete_sport", "langid"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", default="")
    ap.add_argument("--max_groups", type=int, default=0)
    ap.add_argument("--refonly", action="store_true")
    ap.add_argument("--ks_contrast", action="store_true", help="city_capital: extra reference ranked by same-entity contrast")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    S = lk_core.Subject(layers=tuple(sorted(set(a.layers))))
    fams = lk_data.families()
    val = json.load(open(a.validated))
    wiki = lk_data.unrelated_texts(64, n_tokens_chars=400)
    log = open(os.path.join(a.out, "progress.log"), "a")

    for L in a.layers:
        kl_ps = PromptSet(S, L, wiki[8:32], None, max_len=64)       # held-out unrelated text (grader)
        klp_ps = PromptSet(S, L, wiki[40:48], None, max_len=48)      # reference's own proxy text
        S.clean_hidden_all(kl_ps)
        S.clean_hidden_all(klp_ps)
        for fam in a.families:
            F = fams[fam]
            single = F["single_entity"]
            ans = {g: lk_core.ids_for(S, w) for g, w in F["answers"].items()}
            V = val[fam]
            rng0 = random.Random(1000 * a.seed + 7)
            ents = {}
            for g in F["groups"]:
                es = sorted({r[0] for r in V[g]})
                rng0.shuffle(es)
                ents[g] = es
            n_ex = {g: (len(es) if single else (4 if len(es) >= 12 else 3)) for g, es in ents.items()}
            ex_e = {g: set(ents[g][:n_ex[g]]) for g in ents}
            ho_e = {g: (set(ents[g]) if single else set(ents[g][n_ex[g]:])) for g in ents}
            targets = [g for g in F["groups"] if (single and ents[g]) or (not single and len(ents[g]) >= 6)]
            if a.groups:
                targets = [g for g in targets if g in a.groups.split(",")]
            if a.max_groups:
                targets = targets[:a.max_groups]
            ks_items = None
            if fam == "city_capital":
                ks_items = val["city_state"]
            for g in targets:
                fn = os.path.join(a.out, f"{fam}__{g.replace(' ', '_')}__L{L}__s{a.seed}.json")
                if os.path.exists(fn):
                    continue
                tc = time.time()
                rng = random.Random(zlib.crc32(f'{fam}|{g}|{L}|{a.seed}'.encode()))
                others = [h for h in F["groups"] if h != g]
                # ---------------- example set (targets + controls)
                ex_t = [(g, r) for r in V[g] if r[0] in ex_e[g] and r[1] in lk_data.EX_T]
                ex_c = strat_sample([(h, r) for h in others for r in V[h] if r[0] in ex_e[h] and r[1] in lk_data.EX_T],
                                    lambda x: x[0], 24, rng)
                # ---------------- held-out set
                ho_tT = [(g, r) for r in V[g] if r[0] in ho_e[g] and r[1] in lk_data.HO_T]
                ho_tS = [(g, r) for r in V[g] if r[0] in ho_e[g] and r[1] in lk_data.HO_S]
                rng.shuffle(ho_tT)
                rng.shuffle(ho_tS)
                ho_tT, ho_tS = ho_tT[:30], ho_tS[:80]
                ho_sT = strat_sample([(h, r) for h in others for r in V[h] if r[0] in ho_e[h] and r[1] in lk_data.HO_T],
                                     lambda x: x[0], 40, rng)
                ho_sS = strat_sample([(h, r) for h in others for r in V[h] if r[0] in ho_e[h] and r[1] in lk_data.HO_S],
                                     lambda x: x[0], 100, rng)
                ho_ks, ex_ks = [], []
                if ks_items is not None:
                    ho_ks = [(g, r) for r in ks_items.get(g, []) if r[0] in ho_e[g] and r[1] in lk_data.HO_T + lk_data.HO_S]
                    rng.shuffle(ho_ks)
                    ho_ks = ho_ks[:40]
                    ex_ks = [(g, r) for r in ks_items.get(g, []) if r[0] in ex_e[g] and r[1] in lk_data.EX_T]
                if len(ex_t) < 2:
                    json.dump(dict(family=fam, group=g, layer=L, seed=a.seed, skipped="fewer than 2 example items"),
                              open(fn, "w"))
                    continue
                ks_ans = lk_core.ids_for(S, [g]) if ks_items is not None else None
                # ---------------- prompt sets
                ex_items = [("t", x) for x in ex_t] + [("c", x) for x in ex_c] + [("k", x) for x in ex_ks]
                ex_ps = PromptSet(S, L, [x[1][2] for _, x in ex_items],
                                  [ks_ans if role == "k" else ans[x[0]] for role, x in ex_items])
                ex_role = np.array([r for r, _ in ex_items])
                ho_items = ([("tT", x) for x in ho_tT] + [("tS", x) for x in ho_tS] + [("sT", x) for x in ho_sT]
                            + [("sS", x) for x in ho_sS] + [("ks", x) for x in ho_ks])
                ho_ps = PromptSet(S, L, [x[1][2] for _, x in ho_items],
                                  [ks_ans if role == "ks" else ans[x[0]] for role, x in ho_items])
                # target-only and control-only sets for attribution / mean-difference
                t_ps = PromptSet(S, L, [x[2] for _, x in ex_t], [ans[g]] * len(ex_t))
                c_ps = PromptSet(S, L, [x[2] for _, x in ex_c], [ans[h] for h, _ in ex_c])
                attr_t, act_t, err_t = S.attribution(t_ps)
                attr_c, _, err_c = S.attribution(c_ps)
                naive = attr_t.mean(0)
                contr = naive - attr_c.mean(0)
                v = S.mean_last(t_ps) - S.mean_last(c_ps)
                Wd = S.sae[L]["W_dec"]
                cos = torch.nn.functional.cosine_similarity(Wd, v[None, :], dim=1).cpu().numpy()
                actsum = act_t.sum(0)
                pool = [int(i) for i in np.argsort(-contr)[:12] if contr[i] > 0]
                pool += [int(i) for i in np.argsort(-naive)[:4] if naive[i] > 0 and int(i) not in pool]
                # ---------------- reference: greedy forward selection on the example objective
                m0, _ = S.eval_last(ex_ps, [Edit()])
                m0 = np.maximum(m0[0], 1e-3)
                pool2 = []
                if ex_ks and a.ks_contrast:
                    # same-entity contrast: latents that drive the capital answer but NOT the state answer of the
                    # same example cities (what a careful agent would rank by under the keep-state rule)
                    k_ps = PromptSet(S, L, [x[2] for _, x in ex_ks], [ks_ans] * len(ex_ks))
                    attr_k, _, _ = S.attribution(k_ps)
                    ksc = naive - attr_k.mean(0)
                    pool2 = [int(i) for i in np.argsort(-ksc)[:12] if ksc[i] > 0]
                    pool2 += [int(i) for i in np.argsort(-contr)[:4] if int(i) not in pool2]
                allc = sorted(set(pool) | set(pool2))
                kl1 = S.eval_kl(klp_ps, [Edit.lat([c]) for c in allc]).mean(1) if allc else np.zeros(0)
                kl1 = {c: float(k) for c, k in zip(allc, kl1)}

                def greedy(roles_ctrl, pool=pool):
                    sel, traj = [], []
                    it = ex_role == "t"
                    ic = np.isin(ex_role, roles_ctrl)
                    for step in range(10):
                        cand = [c for c in pool if c not in sel]
                        if not cand:
                            break
                        mg, _ = S.eval_last(ex_ps, [Edit.lat(sel + [c]) for c in cand])
                        kls = np.array([sum(kl1[i] for i in sel + [c]) for c in cand])
                        J, Eh, Ph = J_vec(mg[:, it], m0[it][None], mg[:, ic], m0[ic][None], kls)
                        b = int(np.argmax(J))
                        sel.append(cand[b])
                        traj.append(dict(latent=cand[b], J=float(J[b]), E_ex=float(Eh[b]), P_ex=float(Ph[b]),
                                         klproxy=float(kls[b])))
                    pref = {}
                    for k in KS:
                        if not traj:
                            pref[k] = []
                            continue
                        js = [traj[j]["J"] for j in range(min(k, len(traj)))]
                        pref[k] = sel[:int(np.argmax(js)) + 1]
                    return sel, traj, pref
                sel, traj, pref = greedy(["c"])
                sets = {}
                for k in KS:
                    sets[f"ref_k{k}"] = pref[k]
                if ex_ks:
                    sel_ks, traj_ks, pref_ks = greedy(["c", "k"])
                    for k in KS:
                        sets[f"refks_k{k}"] = pref_ks[k]
                    if pool2:
                        _, traj_ks2, pref_ks2 = greedy(["c", "k"], pool=pool2)
                        for k in KS:
                            sets[f"refks2_k{k}"] = pref_ks2[k]
                        sets["ksc_k5"] = [int(i) for i in np.argsort(-ksc)[:5]]
                if not a.refonly:
                    for k in (5,):
                        sets[f"naive_k{k}"] = [int(i) for i in np.argsort(-naive)[:k]]
                        sets[f"contr_k{k}"] = [int(i) for i in np.argsort(-contr)[:k]]
                    rr = np.random.default_rng(zlib.crc32(f'{fam}|{g}|{L}|{a.seed}|r'.encode()))
                    active = np.flatnonzero(actsum > 0)
                    for j in range(1):
                        sets[f"rand_k5_{j}"] = [int(i) for i in rr.choice(D_SAE, 5, replace=False)]
                        sets[f"randact_k5_{j}"] = [int(i) for i in rr.choice(active, min(5, len(active)), replace=False)]
                    sets["mostact_k5"] = [int(i) for i in np.argsort(-actsum)[:5]]
                    sets["cos_k5"] = [int(i) for i in np.argsort(-cos)[:5]]
                    sets["contr_k50"] = [int(i) for i in np.argsort(-contr)[:50]]
                names = list(sets)
                edits = [Edit()] + [Edit.lat(sets[n]) for n in names]
                if not a.refonly:
                    # plain steering vector (not a valid submission): alpha tuned on the example objective
                    vecs = [Edit("vec", vec=al * v) for al in ALPHAS]
                    mg, _ = S.eval_last(ex_ps, vecs)
                    klv = S.eval_kl(klp_ps, vecs).mean(1)
                    it, ic = ex_role == "t", ex_role == "c"
                    J, _, _ = J_vec(mg[:, it], m0[it][None], mg[:, ic], m0[ic][None], klv)
                    best_alpha = ALPHAS[int(np.argmax(J))]
                    names += ["steer_tuned", "steer_a1", "proj"]
                    edits += [Edit("vec", vec=best_alpha * v), Edit("vec", vec=v), Edit("proj", vec=v / v.norm())]
                # ---------------- held-out evaluation (dedupe identical latent sets)
                key = {}
                uniq, uidx = [], []
                for e in edits:
                    kk = (e.kind, tuple(sorted(e.ids)), None if e.vec is None else float(e.vec.norm()) + float(e.vec[0]))
                    if kk not in key:
                        key[kk] = len(uniq)
                        uniq.append(e)
                    uidx.append(key[kk])
                mg_u, top_u = S.eval_last(ho_ps, uniq)
                kl_u = S.eval_kl(kl_ps, uniq)
                mg_h, kl_h = mg_u[uidx], kl_u[uidx]
                roles = [r for r, _ in ho_items]
                res = dict(family=fam, group=g, layer=L, seed=a.seed,
                           ex_entities=sorted(ex_e[g]), n_ho_entities=len(ho_e[g]),
                           n=dict(ex_t=len(ex_t), ex_c=len(ex_c), ex_ks=len(ex_ks), tT=len(ho_tT), tS=len(ho_tS),
                                  sT=len(ho_sT), sS=len(ho_sS), ks=len(ho_ks)),
                           items=[[r, x[0], x[1][0], x[1][1]] for r, x in ho_items],
                           clean_margin=[round(float(m), 2) for m in mg_h[0]],
                           sets=sets, set_names=["clean"] + names,
                           margins={n: [round(float(m), 2) for m in mg_h[i]] for i, n in enumerate(["clean"] + names)},
                           kl={n: [round(float(k), 5) for k in kl_h[i]] for i, n in enumerate(["clean"] + names)},
                           greedy=traj, pool=pool, kl1_proxy=kl1,
                           attr_err_t=float(err_t.mean()), attr_lat_t=float(attr_t.sum(1).mean()),
                           attr_err_c=float(err_c.mean()), attr_lat_c=float(attr_c.sum(1).mean()),
                           naive_score_top=[[int(i), float(naive[i])] for i in np.argsort(-naive)[:10]],
                           contr_score_top=[[int(i), float(contr[i])] for i in np.argsort(-contr)[:10]],
                           v_norm=float(v.norm()))
                if ex_ks:
                    res["greedy_ks"] = traj_ks
                    if pool2:
                        res["greedy_ks2"] = traj_ks2
                if not a.refonly:
                    res["best_alpha"] = best_alpha
                # ---------------- what do the chosen latents fire on? (held-out prompts + wikitext)
                chosen = sorted({i for n in ("ref_k5", "naive_k5", "contr_k5", "refks_k5") for i in sets.get(n, [])})
                if chosen:
                    acts_ho = S.latent_acts(ho_ps, chosen)
                    acts_kl = S.latent_acts(kl_ps, chosen)
                    lat = {}
                    for j, i in enumerate(chosen):
                        d = {}
                        for rname in ("tT", "tS", "sT", "sS", "ks"):
                            sel_i = [q for q, r in enumerate(roles) if r == rname]
                            if not sel_i:
                                continue
                            anyf = [bool((acts_ho[q][:, j] > 0).any()) for q in sel_i]
                            lastf = [bool(acts_ho[q][-1, j] > 0) for q in sel_i]
                            d[rname] = [round(float(np.mean(anyf)), 3), round(float(np.mean(lastf)), 3)]
                        tt = [q for q, r in enumerate(roles) if r in ("tT", "tS")]
                        ents_f = {ho_items[q][1][1][0] for q in tt if (acts_ho[q][:, j] > 0).any()}
                        d["ent_cov"] = [len(ents_f), len({ho_items[q][1][1][0] for q in tt})]
                        tok = np.concatenate([a_[1:, j] for a_ in acts_kl])
                        d["wiki_tok_freq"] = round(float((tok > 0).mean()), 5)
                        lat[str(i)] = d
                    res["latent_stats"] = lat
                res["seconds"] = round(time.time() - tc, 1)
                json.dump(res, open(fn, "w"))
                msg = (f"{time.strftime('%H:%M:%S')} {fam} {g} L{L}: {res['seconds']}s pool={len(pool)} "
                       f"ref5={sets['ref_k5']} J={[round(t['J'], 2) for t in traj]}")
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
                del ex_ps, ho_ps, t_ps, c_ps
                torch.cuda.empty_cache()
        del kl_ps, klp_ps
        torch.cuda.empty_cache()
    print(f"done {time.time()-t0:.0f}s peak_gpu {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
          f"peak_rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
