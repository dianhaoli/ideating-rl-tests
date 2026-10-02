"""Skeptic reproduction driver (GPU). Independent of lk_core / sweep / analyze (uses lk_data only for templates and
entity lists, which are data).

mode repro  : for a reported cell, (a) recompute clean answers and the held-out R of the REPORTED sets on the SAME
              held-out items with my own hook/metric; (b) re-derive the cosine, naive-attribution and contrastive
              rankings and a greedy reference myself from the cell's example entities; (c) evaluate every set on a
              FRESH held-out set I wrote (new templates in new styles, new entities); (d) re-run my reference on two
              more example splits and score all three on a new-entity-only set (seed stability).
mode strong : for a cell reported infeasible, a stronger non-oracle search (more example templates, agent-written
              extra entities, a 4x larger candidate pool, beam search) and an ORACLE search that optimises directly
              on the grader's held-out items (an upper bound for "can <= 5 latents do it at all?").
Per-item margins are saved so bootstrap CIs are computed on CPU (analyse.py).
usage: python -m tasks.latentknockout.skeptic_reproduce.repro <mode> <cells_dir> <out_dir> fam:group:L [...]
"""
import csv
import glob
import json
import os
import random
import resource
import sys
import time

import numpy as np
import torch

from common import gpuq
gpuq.apply_caps()

from tasks.latentknockout import lk_data  # noqa: E402  (data only: templates, entities)
from tasks.latentknockout.skeptic_reproduce.kx import KX, Items, kl_texts  # noqa: E402

EX_T, HO_T, HO_S = [0, 1, 3], [2, 4], [5, 6, 7, 8, 9, 10, 11, 12]
KAPPA = 0.1

# ---------------------------------------------------------------- fresh templates (written by the skeptic; new styles:
# trivia show, personal anecdote, form/checklist, census/blog/letter). Never seen by any search here.
FRESH = {
    "city_state": [
        "Trivia night, round three. Host: The city of {e} is in which US state? Contestant: It is in",
        "My grandmother grew up in {e}, so she has always called herself a proud native of the state of",
        "Moving checklist: we sold our house in {e}, so now we have to hand back our driver's licenses from the state of",
        "In the last census, {e} was listed among the fastest-growing cities in the state of",
        "Travel blog, day 4: we drove into {e} this morning, one of my favourite towns in all of",
        "Pen-pal letter: Greetings from {e}! Life here in my home state of",
    ],
    "city_capital": [
        "Trivia night, round three. Host: What is the capital of the state where {e} is? Contestant: The capital is",
        "My cousin in {e} had to drive to the state capital to renew her license, so she went to",
        "Kids who grow up in {e} learn in school that their state capital is",
        "If you live in {e}, your state legislature meets in the capital city of",
        "Travel blog, day 4: after leaving {e}, we drove on to see the state capitol building in",
        "Pen-pal letter: Greetings from {e}! Next week our class takes a field trip to our state capital,",
    ],
    "country_lang": [
        "Trivia night, round three. Host: What language is mainly spoken in {e}? Contestant: It is",
        "Before my semester abroad in {e}, I spent a whole year studying",
        "Most of the newspapers sold in {e} are printed in",
        "Travel tip: before you visit {e}, learn a few basic phrases in",
        "Language course sign-up form. Destination country: {e}. Course required:",
        "Pen-pal letter: Greetings from {e}! Sorry my English is not perfect, at school we mostly speak",
    ],
    "athlete_sport": [
        "Trivia night, round three. Host: Which sport made {e} famous? Contestant: It was",
        "My dad has a signed poster of {e} on his wall because he loves watching",
        "Hall of fame entry: {e}, honoured for an outstanding career in professional",
        "Growing up, {e} spent every afternoon practising",
        "Kids who idolise {e} usually want to take up",
        "Sports quiz card. Name: {e}. Known for playing:",
    ],
    "langid": [
        'Trivia night, round three. Host: The phrase "{e}" is in which language? Contestant: It is',
        'I found a note in my hotel room that said "{e}". I think it was written in',
        'My pen pal ended her letter with "{e}", because she always writes to me in',
        'Overheard on the train: "{e}". The passengers were chatting in',
        'Subtitle file. Line 12: "{e}" Original audio language:',
        'Translation request. Source text: "{e}". Source language:',
    ],
}

# ---------------------------------------------------------------- new entities (not in lk_data's lists)
NEW = {
    ("city_state", "Arizona"): ["Avondale", "Goodyear", "Buckeye", "Sierra Vista", "Show Low", "Payson", "Cottonwood",
                                "Bullhead City", "Winslow", "Holbrook", "Apache Junction", "Fountain Hills",
                                "Queen Creek", "Oro Valley", "Sahuarita", "Camp Verde", "Wickenburg", "Safford"],
    ("city_capital", "Illinois"): ["Aurora", "Normal", "DeKalb", "Belleville", "Edwardsville", "Skokie", "Wheaton",
                                   "Arlington Heights", "Palatine", "Orland Park", "Bolingbrook", "Tinley Park",
                                   "Oak Lawn", "Berwyn", "Mount Prospect", "Downers Grove", "Lombard", "Glenview"],
    ("city_capital", "Texas"): ["Tyler", "Denton", "Round Rock", "College Station", "Sugar Land", "Garland", "Irving",
                                "Frisco", "Mesquite", "San Marcos", "Harlingen", "Wichita Falls", "Nacogdoches",
                                "Port Arthur", "Conroe", "Pflugerville", "Kerrville", "Del Rio", "Grand Prairie",
                                "New Braunfels"],
    ("country_lang", "French"): ["Cameroon", "Djibouti", "Burundi", "Madagascar", "the Comoros", "Rwanda",
                                 "Luxembourg", "Belgium", "Seychelles", "Vanuatu"],
    ("athlete_sport", "golf"): ["Collin Morikawa", "Scottie Scheffler", "Jon Rahm", "Brooks Koepka", "Lee Trevino",
                                "Sam Snead", "Bobby Jones", "Fred Couples", "Justin Thomas", "Padraig Harrington",
                                "Nancy Lopez", "Lorena Ochoa", "Lydia Ko", "Payne Stewart", "Tom Watson",
                                "Hideki Matsuyama"],
    ("athlete_sport", "soccer"): ["Luka Modric", "Kevin De Bruyne", "Robert Lewandowski", "Karim Benzema", "Harry Kane",
                                  "Paolo Maldini", "Gianluigi Buffon", "Didier Drogba", "Steven Gerrard",
                                  "Frank Lampard", "Ruud Gullit", "Eusebio", "Ferenc Puskas", "Lothar Matthaus",
                                  "Xavi Hernandez", "Sergio Ramos"],
    ("country_lang", "English"): ["Sierra Leone", "the Gambia", "Botswana", "Malawi", "Fiji", "Antigua and Barbuda",
                                  "Grenada", "Saint Lucia", "Dominica", "Namibia", "Papua New Guinea",
                                  "the Solomon Islands", "Saint Kitts and Nevis", "Kenya", "Ireland", "Canada"],
}
NEW[("city_state", "Texas")] = NEW[("city_capital", "Texas")]


def new_langid(lang, n=16):
    code = lk_data.LANGID_CODES[lang]
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/hf_home"))
    p = glob.glob(os.path.join(hf, "hub", "datasets--papluca--language-identification", "snapshots", "*", "valid.csv"))[0]
    old = set(lk_data.langid_sentences()[lang])
    out = []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["labels"] != code:
                continue
            t = " ".join(row["text"].replace('"', "'").split())
            w = t.split(" ")
            if len(w) < 5 or any(ch.isdigit() for ch in t) or len(t) > 140 or "{" in t or "}" in t:
                continue
            s = " ".join(w[:14])
            if s not in old:
                out.append(s)
            if len(out) >= n:
                break
    return out


def log(msg, fh):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def stratified(rows, key, cap, rng):
    by = {}
    for r in rows:
        by.setdefault(key(r), []).append(r)
    for v in by.values():
        rng.shuffle(v)
    out = []
    ks = sorted(by)
    while len(out) < cap and any(by[k] for k in ks):
        for k in ks:
            if by[k] and len(out) < cap:
                out.append(by[k].pop())
    return out


class Cell:
    """Builds every prompt set for one (family, group, layer) cell."""

    def __init__(self, kx, cells_dir, fam, g, L):
        self.kx, self.fam, self.g, self.L = kx, fam, g, L
        fn = os.path.join(cells_dir, f"{fam}__{g.replace(' ', '_')}__L{L}__s0.json")
        self.c = json.load(open(fn))
        self.F = lk_data.families()[fam]
        self.T = [t for _, t in self.F["templates"]]
        self.acc = {h: kx.accept_ids(self.F["answers"][h]) for h in self.F["groups"]}
        self.new = new_langid(g) if fam == "langid" else NEW.get((fam, g), [])
        items = [it for it in self.c["items"] if it[0] in ("tT", "tS", "sT", "sS")]
        self.ho_rows = [(r, h, e, ti, self.T[ti].format(e=e)) for r, h, e, ti in items]
        self.ho_tgt_ents = sorted({e for r, h, e, ti in items if r in ("tT", "tS")})
        self.ho_sib = sorted({(h, e) for r, h, e, ti in items if r in ("sT", "sS")})
        self.ex_ents = list(self.c["ex_entities"])

    def ps(self, rows):
        """rows: (role, group, entity, template_id_or_text_tag, text) -> Items + rows."""
        return Items(self.kx, [r[4] for r in rows], [self.acc[r[1]] for r in rows], self.L)

    def validate(self, rows):
        """Keep rows whose clean top-1 is an accepted answer (my own clean run). Returns kept rows, clean margins."""
        if not rows:
            return [], np.zeros(0)
        it = self.ps(rows)
        m = it.eval([[]])[0]
        keep = [r for r, x in zip(rows, m) if x > 0]
        return keep, m

    def example_rows(self, ex_ents, seed, tmpl=EX_T, n_ctrl_per=3, cap=36, extra_tgt=()):
        rng = random.Random(seed)
        tgt = [("t", self.g, e, ti, self.T[ti].format(e=e)) for e in list(ex_ents) + list(extra_tgt) for ti in tmpl]
        held = {e for _, e in self.ho_sib}
        ctrl = []
        for h, es in self.F["groups"].items():
            if h == self.g:
                continue
            cand = [e for e in es if e not in held]
            rng.shuffle(cand)
            for e in cand[:n_ctrl_per]:
                ctrl += [("c", h, e, ti, self.T[ti].format(e=e)) for ti in tmpl]
        ctrl = stratified(ctrl, lambda r: r[1], cap, rng)
        return tgt, ctrl

    def fresh_rows(self, tgt_ents, n_sib_cap=150, seed=0):
        rng = random.Random(seed)
        tem = FRESH[self.fam]
        tgt = [("fT", self.g, e, f"F{j}", t.format(e=e)) for e in tgt_ents for j, t in enumerate(tem)]
        sib = [("fS", h, e, f"F{j}", t.format(e=e)) for h, e in self.ho_sib for j, t in enumerate(tem)]
        sib = stratified(sib, lambda r: r[1], n_sib_cap, rng)
        return tgt, sib


def J_ex(m, m0, is_t, is_c):
    """Example objective (hard + soft effect) x (hard + soft preservation); m [S, n]."""
    t, c = m[:, is_t], m[:, is_c]
    t0, c0 = np.maximum(m0[is_t], 1e-3), np.maximum(m0[is_c], 1e-3)
    E = 0.5 * ((t < 0).mean(1) + np.clip((t0 - t) / t0, 0, 1).mean(1))
    P = 0.5 * ((c > 0).mean(1) + np.clip(c / c0, 0, 1).mean(1))
    return E * P


def beam_search(items, m0, is_t, is_c, pool, kmax=5, width=1, objective=J_ex):
    """Greedy (width 1) or beam search over subsets of pool with real ablations; returns best set per k."""
    beams = [[]]
    best = {}
    for k in range(1, kmax + 1):
        cands, seen = [], set()
        for b in beams:
            for c in pool:
                if c in b:
                    continue
                s = tuple(sorted(b + [c]))
                if s not in seen:
                    seen.add(s)
                    cands.append(list(s))
        if not cands:
            break
        m = items.eval(cands)
        J = objective(m, m0, is_t, is_c)
        order = np.argsort(-J)[:width]
        beams = [cands[i] for i in order]
        best[k] = (cands[order[0]], float(J[order[0]]))
    return best


def rankings(cell, tgt_rows, ctrl_rows):
    """Cosine / naive / contrastive / activation-difference scores from example targets and controls."""
    kx, L = cell.kx, cell.L
    ti = cell.ps(tgt_rows)
    ci = cell.ps(ctrl_rows)
    at, act_t, last_t = ti.attribution()
    ac, act_c, last_c = ci.attribution()
    naive = at.mean(0)
    contr = naive - ac.mean(0)
    v = last_t - last_c
    Wd = kx.sae[L]["Wd"]
    cos = torch.nn.functional.cosine_similarity(Wd, v[None], dim=1).cpu().numpy()
    actdiff = act_t.mean(0) - act_c.mean(0)
    del ti, ci
    return dict(naive=naive, contr=contr, cos=cos, actdiff=actdiff)


def top(score, k):
    return [int(i) for i in np.argsort(-score)[:k]]


def metric_block(m, roles, kl, tgt_role, sib_role):
    t, s = m[roles == tgt_role], m[roles == sib_role]
    E, P = float((t < 0).mean()), float((s > 0).mean())
    fac = max(0.0, 1 - float(np.mean(kl)) / KAPPA)
    return dict(E=E, P=P, KLfac=fac, R=E * P * fac, n_t=int(len(t)), n_s=int(len(s)))


def run_repro(kx, cells_dir, spec, out_dir, wiki, fh, mode):
    fam, g, L = spec.split(":")
    L = int(L)
    t0 = time.time()
    cell = Cell(kx, cells_dir, fam, g, L)
    res = dict(family=fam, group=g, layer=L, mode=mode, reported_sets={k: v for k, v in cell.c["sets"].items()
                                                                        if k in ("ref_k5", "ref_k10", "cos_k5",
                                                                                 "naive_k5", "contr_k5")})
    # ---------------- (a) same held-out items, my clean check
    ho_all, m_clean = cell.validate(cell.ho_rows)
    rep_clean = np.array([cell.c["clean_margin"][i] for i, it in enumerate(cell.c["items"])
                          if it[0] in ("tT", "tS", "sT", "sS")], np.float32)
    res["clean_check"] = dict(n_items=len(cell.ho_rows), n_my_clean_correct=len(ho_all),
                              max_abs_margin_diff_vs_reported=float(np.abs(m_clean - rep_clean).max()),
                              median_abs_margin_diff=float(np.median(np.abs(m_clean - rep_clean))))
    log(f"{spec} clean check {res['clean_check']}", fh)
    # ---------------- example sets and fresh held-out
    ex_t, ex_c = cell.example_rows(cell.ex_ents, seed=11)
    ex_t, _ = cell.validate(ex_t)
    ex_c, _ = cell.validate(ex_c)
    new_ok = cell.new
    if mode == "strong":
        half = len(new_ok) // 2
        new_a, new_b = new_ok[:half], new_ok[half:]
    else:
        new_a, new_b = [], new_ok
    f_t, f_s = cell.fresh_rows(cell.ho_tgt_ents + new_b)
    f_t, _ = cell.validate(f_t)
    f_s, _ = cell.validate(f_s)
    # new-entity-only eval set (fresh + original new-style templates) for the seed-stability check
    nw = [("nT", g, e, f"F{j}", t.format(e=e)) for e in new_b for j, t in enumerate(FRESH[fam])]
    nw += [("nT", g, e, ti, cell.T[ti].format(e=e)) for e in new_b for ti in HO_S]
    nw, _ = cell.validate(nw)
    nw_s = [("nS", r[1], r[2], r[3], r[4]) for r in f_s]
    res["n"] = dict(ex_t=len(ex_t), ex_c=len(ex_c), ho=len(ho_all), fresh_t=len(f_t), fresh_s=len(f_s), new_t=len(nw),
                    new_entities_valid=sorted({r[2] for r in nw}), new_entities_given=new_b,
                    fresh_t_entities=len({r[2] for r in f_t}))
    log(f"{spec} sizes {res['n']}", fh)
    # ---------------- (b) my own rankings and reference on the cell's example split (seed 0)
    rk = rankings(cell, ex_t, ex_c)
    sets = {}
    for k, v in res["reported_sets"].items():
        sets["rep_" + k] = v
    sets["my_cos_k5"] = top(rk["cos"], 5)
    sets["my_naive_k5"] = top(rk["naive"], 5)
    sets["my_contr_k5"] = top(rk["contr"], 5)
    sets["null"] = []
    ex_rows = ex_t + ex_c
    ex_items = cell.ps(ex_rows)
    m0 = ex_items.eval([[]])[0]
    is_t = np.array([r[0] == "t" for r in ex_rows])
    is_c = ~is_t
    pool = []
    for i in top(rk["contr"], 16) + top(rk["naive"], 6) + top(rk["cos"], 6):
        if i not in pool:
            pool.append(i)
    best = beam_search(ex_items, m0, is_t, is_c, pool, kmax=5, width=1)
    kbest = max(best, key=lambda k: best[k][1])
    sets["my_ref_s0"] = best[kbest][0]
    res["my_ref_trace"] = {k: [v[0], round(v[1], 3)] for k, v in best.items()}
    log(f"{spec} my_ref_s0 {sets['my_ref_s0']} J={best[kbest][1]:.3f}; reported ref_k5 {res['reported_sets'].get('ref_k5')}", fh)
    del ex_items
    if mode == "strict":
        # ---------------- ORACLE searches on the grader's own held-out (new-style) items, for a hard flip (delta 0)
        # and for a 'real' knock-out (answer beaten by >= 1 logit): upper bounds on what <= 5 latents can do
        ho_S = [r for r in ho_all if r[0] in ("tS", "sS")]
        ho_it = cell.ps(ho_S)
        hm0 = ho_it.eval([[]])[0]
        hst = np.array([r[0] == "tS" for r in ho_S])
        rk4 = rankings(cell, [r for r in ho_S if r[0] == "tS"], [r for r in ho_S if r[0] == "sS"])
        pool4 = []
        for i in top(rk4["contr"], 40) + top(rk4["cos"], 15) + top(rk4["actdiff"], 15) + pool:
            if i not in pool4:
                pool4.append(i)
        for dlt in (0.0, 1.0):
            def hardR(m, m0_, it_, ic_, dlt=dlt):
                t, c = m[:, it_], m[:, ic_]
                return (t < -dlt).mean(1) * (c > 0).mean(1) + 1e-3 * J_ex(m, m0_, it_, ic_)
            b4 = beam_search(ho_it, hm0, hst, ~hst, pool4, kmax=5, width=int(os.environ.get("LK_BEAM", "1")), objective=hardR)
            for k in b4:
                sets[f"oracle_d{int(dlt)}_k{k}"] = b4[k][0]
            res[f"oracle_d{int(dlt)}"] = dict(pool=len(pool4), trace={k: [v[0], round(v[1], 3)] for k, v in b4.items()})
            log(f"{spec} oracle delta={dlt} {res[f'oracle_d{int(dlt)}']}", fh)
        del ho_it
    elif mode == "repro":
        # ---------------- (d) seed stability: two more example splits drawn from the group's original entities
        allents = [e for e in cell.F["groups"][g]]
        for sd in (1, 2):
            rng = random.Random(100 + sd)
            pick = [e for e in allents]
            rng.shuffle(pick)
            pick = pick[:len(cell.ex_ents)]
            et, ec = cell.example_rows(pick, seed=11 + sd)
            et, _ = cell.validate(et)
            ec, _ = cell.validate(ec)
            if len(et) < 2:
                continue
            rk2 = rankings(cell, et, ec)
            rows2 = et + ec
            it2 = cell.ps(rows2)
            m02 = it2.eval([[]])[0]
            ist = np.array([r[0] == "t" for r in rows2])
            pool2 = []
            for i in top(rk2["contr"], 16) + top(rk2["naive"], 6) + top(rk2["cos"], 6):
                if i not in pool2:
                    pool2.append(i)
            b2 = beam_search(it2, m02, ist, ~ist, pool2, kmax=5, width=1)
            kb = max(b2, key=lambda k: b2[k][1])
            sets[f"my_ref_s{sd}"] = b2[kb][0]
            sets[f"my_cos_s{sd}"] = top(rk2["cos"], 5)
            sets[f"my_contr_s{sd}"] = top(rk2["contr"], 5)
            res.setdefault("seed_examples", {})[sd] = pick
            log(f"{spec} seed {sd} examples {pick} -> my_ref {sets[f'my_ref_s{sd}']}", fh)
            del it2
    else:
        # ---------------- strong non-oracle search: + same-style held-out templates on the EXAMPLE entities,
        # + agent-written extra entities (new_a), a 5x pool, beam width 3
        et, ec = cell.example_rows(cell.ex_ents, seed=21, tmpl=EX_T + HO_T, n_ctrl_per=4, cap=80, extra_tgt=new_a)
        et, _ = cell.validate(et)
        ec, _ = cell.validate(ec)
        rk3 = rankings(cell, et, ec)
        rows3 = et + ec
        it3 = cell.ps(rows3)
        m03 = it3.eval([[]])[0]
        ist = np.array([r[0] == "t" for r in rows3])
        pool3 = []
        P = [int(x) for x in os.environ.get("LK_POOL", "60,30,30,30").split(",")]
        for i in top(rk3["contr"], P[0]) + top(rk3["naive"], P[1]) + top(rk3["cos"], P[2]) + top(rk3["actdiff"], P[3]):
            if i not in pool3:
                pool3.append(i)
        b3 = beam_search(it3, m03, ist, ~ist, pool3, kmax=5, width=int(os.environ.get("LK_BEAM", "3")))
        kb = max(b3, key=lambda k: b3[k][1])
        sets["strong_ref"] = b3[kb][0]
        for k in b3:
            sets[f"strong_k{k}"] = b3[k][0]
        res["strong"] = dict(n_ex_t=len(et), n_ex_c=len(ec), pool=len(pool3), trace={k: [v[0], round(v[1], 3)]
                                                                                    for k, v in b3.items()})
        log(f"{spec} strong {res['strong']}", fh)
        del it3
        # ---------------- ORACLE: optimise R_S directly on the grader's held-out items (upper bound)
        ho_S = [r for r in ho_all if r[0] in ("tS", "sS")]
        ho_it = cell.ps(ho_S)
        hm0 = ho_it.eval([[]])[0]
        hst = np.array([r[0] == "tS" for r in ho_S])
        ht = [r for r in ho_S if r[0] == "tS"]
        hs = [r for r in ho_S if r[0] == "sS"]
        rk4 = rankings(cell, ht, hs)
        pool4 = []
        for i in top(rk4["contr"], P[0]) + top(rk4["cos"], P[2]) + top(rk4["actdiff"], P[3]) + pool3[:40]:
            if i not in pool4:
                pool4.append(i)

        def hardR(m, m0_, it_, ic_):     # hard R (no KL), soft term only as a tie-break
            t, c = m[:, it_], m[:, ic_]
            return (t < 0).mean(1) * (c > 0).mean(1) + 1e-3 * J_ex(m, m0_, it_, ic_)
        b4 = beam_search(ho_it, hm0, hst, ~hst, pool4, kmax=5, width=int(os.environ.get("LK_BEAM", "3")), objective=hardR)
        for k in b4:
            sets[f"oracle_k{k}"] = b4[k][0]
        res["oracle"] = dict(pool=len(pool4), trace={k: [v[0], round(v[1], 3)] for k, v in b4.items()})
        log(f"{spec} oracle {res['oracle']}", fh)
        del ho_it
    # ---------------- (c) evaluate every distinct set on: original held-out, fresh held-out, new-entity set
    names = list(sets)
    uniq = []
    for n in names:
        s = sorted(sets[n])
        if s not in uniq:
            uniq.append(s)
    uid = [uniq.index(sorted(sets[n])) for n in names]
    evals = {}
    for tag, rows in (("ho", ho_all), ("fresh", f_t + f_s), ("new", nw + nw_s)):
        if not rows:
            continue
        it = cell.ps(rows)
        m, rank, t1 = it.eval(uniq, want_rank=True)
        evals[tag] = dict(rows=[[r[0], r[1], r[2], r[3]] for r in rows], m=m, rank=rank, top1=t1)
        del it
    kl = kl_texts(kx, wiki, L, uniq)
    out_sets = {}
    for n, u in zip(names, uid):
        d = dict(latents=sets[n], KL=float(kl[u].mean()))
        for tag, ev in evals.items():
            roles = np.array([r[0] for r in ev["rows"]])
            if tag == "ho":
                d["ho_S"] = metric_block(ev["m"][u], roles, kl[u], "tS", "sS")
                d["ho_T"] = metric_block(ev["m"][u], roles, kl[u], "tT", "sT")
            elif tag == "fresh":
                d["fresh"] = metric_block(ev["m"][u], roles, kl[u], "fT", "fS")
            else:
                d["new"] = metric_block(ev["m"][u], roles, kl[u], "nT", "nS")
        out_sets[n] = d
    res["sets"] = out_sets
    # per-item margins (rounded) for CPU bootstraps
    res["items"] = {tag: dict(rows=ev["rows"], uniq=uniq, uid=dict(zip(names, uid)),
                              margins=[[round(float(x), 2) for x in row] for row in ev["m"]],
                              rank=ev["rank"].tolist(), top1=ev["top1"].tolist()) for tag, ev in evals.items()}
    res["kl_per_text"] = {n: [round(float(x), 5) for x in kl[u]] for n, u in zip(names, uid)}
    res["seconds"] = round(time.time() - t0, 1)
    fn = os.path.join(out_dir, f"{mode}__{fam}__{g.replace(' ', '_')}__L{L}.json")
    json.dump(res, open(fn, "w"))
    summ = {n: (round(d["ho_S"]["R"], 2), round(d.get("fresh", {}).get("R", float("nan")), 2),
                round(d.get("new", {}).get("R", float("nan")), 2)) for n, d in out_sets.items()}
    log(f"{spec} done {res['seconds']}s  R(ho_S, fresh, new): {summ}", fh)
    torch.cuda.empty_cache()


def exactness(kx, fh, L):
    """Cached-residual rerun vs hooked full forward, with and without a 3-latent ablation."""
    fams = lk_data.families()
    texts = [fams["city_state"]["templates"][i][1].format(e=e) for e in ("Dallas", "Tucson", "Buffalo") for i in (0, 3, 9)]
    acc = [kx.accept_ids(["Texas"])] * len(texts)
    it = Items(kx, texts, acc, L)
    out = {}
    for s in ([], [10472, 13331, 1114]):
        a = it.eval([s])[0]
        b = it.eval_full([s])[0]
        out[str(s)] = float(np.abs(a - b).max())
    log(f"exactness L{L} (max |margin cached - margin full forward|): {out}", fh)
    return out


def main():
    mode, cells_dir, out_dir = sys.argv[1:4]
    specs = sys.argv[4:]
    os.makedirs(out_dir, exist_ok=True)
    fh = open(os.path.join(out_dir, f"log_{mode}.txt"), "a")
    layers = sorted({int(s.split(":")[2]) for s in specs})
    kx = KX(layers)
    log(f"loaded; SAEs {[kx.sae[L]['path'].split('gemma-scope-2b-pt-res/')[1] for L in layers]}", fh)
    ex = {L: exactness(kx, fh, L) for L in layers}
    json.dump(ex, open(os.path.join(out_dir, f"exactness_{mode}.json"), "w"))
    wiki = lk_data.unrelated_texts(64, n_tokens_chars=400)[8:32]
    for s in specs:
        try:
            run_repro(kx, cells_dir, s, out_dir, wiki, fh, mode)
        except Exception as e:  # keep going; log the failure
            import traceback
            log(f"{s} FAILED {e!r}\n{traceback.format_exc()}", fh)
            torch.cuda.empty_cache()
    log(f"all done peak_gpu {torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
        f"peak_rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", fh)


if __name__ == "__main__":
    main()
