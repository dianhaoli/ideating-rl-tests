"""Verify-mechanism, CPU part: draw the latent sample (question 1) and search the whole SAE for style-robust latents
(question 2). No GPU, no model.

Question 1 sample (fixed before any per-token activation was looked at):
  rng = random.Random(20261002). Topic latents of generator v2's pool are grouped by their anchor c* (key_check.jsonl).
  Concept-first, like the generator (it draws a concept uniformly, then one of its latents): 30 distinct concepts are
  drawn from the sorted concepts that own >= 1 DROPPED topic latent, then one dropped latent per concept
  (rng.choice over its latents sorted by (layer, real id)); then 15 distinct concepts from those owning >= 1 KEPT topic
  latent, one kept latent each. WHY concept-first: one concept (SolarEclipse) owns 1962 of the 4550 dropped topic
  latents, so a uniform latent draw would be about 43% solar-eclipse latents and say little about the other concepts.
  Then, for each of the 45 latents in draw order, 10 of c*'s 40 split-A texts (sorted(rng.sample(range(40), 10))).
  Bank-F texts: c*'s F2 half (PREREG A1.1: the second text listed for each of the 10 styles), i.e. one per style and
  exactly the texts whose AUROC_F2 decided `not_style_robust`.
  Context texts: the 5 split-A texts of OTHER concepts on which the latent's cached max activation is highest.

Question 2: for every layer, AUROC_C and AUROC_F2 of all 16384 latents for every concept (auroc.auroc_table, the
generator's AUROC: c's texts against every other concept's texts in the same set, ties 1/2). A latent is
"style-robust for c" if AUROC_C(j, c) >= 0.85 and AUROC_F2(j, c) >= 0.85 (the A1.1 thresholds, applied to c rather
than to the latent's anchor). For each such latent we replay generate.Tables.pools rule by rule.

Memory: one layer at a time, memory-mapped float16 caches, 2048 columns per chunk, float32.
Run (from ~/wt/fmdiag, after sourcing common/env.sh):
  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=2G -- /usr/bin/time -v \
      $PY -m tasks.featurematch.diagnosis.verify_mechanism.search
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

from tasks.featurematch.auroc import auroc_table
from tasks.featurematch.diagnosis import style_filter as SF
from tasks.featurematch.fm_core import D_SAE, LAYERS

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
OUT = HERE
NPY = os.path.join(HERE, "cache")          # *.npy is gitignored
SEED = 20261002
THR = 0.85
N_DROP, N_KEEP, N_A, N_CTX = 30, 15, 10, 5
P_THR, P_THR_B, MARGIN, FIRE_MIN, MAX_SPECIFIC = 0.90, 0.85, 0.10, 0.5, 3   # generate.py (asserted below)


def draw_sample(rows):
    rng = random.Random(SEED)
    drop, keep = defaultdict(list), defaultdict(list)
    for r in rows:
        if r["family"] != "topic":
            continue
        (keep if r["kept"] else drop)[r["c_star"]].append(r)
    out = []
    for group, pool, n in (("dropped", drop, N_DROP), ("kept", keep, N_KEEP)):
        concepts = rng.sample(sorted(pool), n)
        for c in concepts:
            lat = rng.choice(sorted(pool[c], key=lambda r: (r["layer"], r["real_latent"])))
            out.append({"group": group, **lat})
    for s in out:
        s["A_idx"] = sorted(rng.sample(range(40), N_A))
    return out


def pool_conditions(T, L):
    """generate.Tables.pools, condition by condition, for every latent at layer L (anchored at its A-argmax)."""
    A, B, F = T.aA[L], T.aB[L], T.fire[L]
    best = A.argmax(1)
    ar = np.arange(len(A))
    bA, bB, bF = A[ar, best], B[ar, best], F[ar, best]
    n_near = (A >= (bA - MARGIN)[:, None]).sum(1)
    own_name = T.names[L].reshape(T.n_c, 3, -1)[best, :, ar].max(1)
    cond = {"auroc_A>=0.90": bA >= P_THR, "auroc_B>=0.85": bB >= P_THR_B, "fire_A>=0.5": bF >= FIRE_MIN,
            "specific(<=3 within 0.10)": n_near <= MAX_SPECIFIC, "anchor_has_split_C": T.eligible[best],
            "own_name_silent": own_name <= 0}
    ok = np.ones(len(A), bool)
    for v in cond.values():
        ok &= v
    inpool = np.zeros(len(A), bool)
    inpool[[j for j, _ in T.planted[L]]] = True
    assert (ok == inpool).all(), "replayed pool rules differ from Tables.pools"
    return best, cond, inpool, n_near


def main():
    import tasks.featurematch.generate as G
    assert (G.P_THR, G.P_THR_B, G.MARGIN, G.FIRE_MIN, G.MAX_SPECIFIC) == (P_THR, P_THR_B, MARGIN, FIRE_MIN, MAX_SPECIFIC)
    os.makedirs(NPY, exist_ok=True)
    src = SF.default_src_cache()
    T = SF.load_tables(src)               # also points concepts.py at src
    cs = T.cs
    cids = [c["cid"] for c in cs]
    cidx = {c: i for i, c in enumerate(cids)}
    meta = json.load(open(os.path.join(src, "precompute_meta.json")))
    rowsA, rowsC = np.array(meta["rowsA"]), np.array(meta["rowsC"])
    n_c = len(cids)

    # bank F: verify order against bank_meta, then the F2 mask (A1.1)
    bmeta = json.load(open(os.path.join(DIAG, "cache", "bank_meta.json")))
    assert bmeta["cids"] == cids
    SF.check_bank_order(bmeta, SF.BANKS, cids)
    ftexts, frows, fstyles = SF.load_bank("F", SF.BANKS, cids)
    f1 = SF.split_f1(frows, fstyles)
    f2 = ~f1
    frows = np.array(frows)

    rows = [json.loads(l) for l in open(os.path.join(DIAG, "key_check.jsonl"))]
    sample = draw_sample(rows)

    # ---------------------------------------------------------------- per layer: AUROC_C / AUROC_F2 for all latents
    tables = {}
    for L in LAYERS:
        pc, pf = os.path.join(NPY, f"auroc_C_L{L}.npy"), os.path.join(NPY, f"auroc_F2_L{L}.npy")
        if os.path.exists(pc) and os.path.exists(pf):
            aC, aF2 = np.load(pc), np.load(pf)
        else:
            XC = np.load(os.path.join(src, f"acts_L{L}_C.npy"), mmap_mode="r")
            XF = np.load(os.path.join(DIAG, "cache", f"bank_acts_L{L}_F.npy"), mmap_mode="r")
            assert XC.shape == (len(rowsC), D_SAE) and XF.shape == (len(frows), D_SAE)
            aC = np.zeros((D_SAE, n_c), np.float32)
            aF2 = np.zeros((D_SAE, n_c), np.float32)
            f2idx = np.nonzero(f2)[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                for s in range(0, D_SAE, 2048):
                    aC[s:s + 2048] = auroc_table(np.asarray(XC[:, s:s + 2048], np.float32), rowsC, n_c)
                    aF2[s:s + 2048] = auroc_table(np.asarray(XF[:, s:s + 2048], np.float32)[f2idx], frows[f2idx], n_c)
            np.save(pc, aC)
            np.save(pf, aF2)
            del XC, XF
        tables[L] = (aC, aF2)
        print(f"L{L} AUROC_C / AUROC_F2 tables ready", flush=True)

    # ---------------------------------------------------------------- consistency with key_check (sampled + all pooled)
    mism = 0
    maxdiff = 0.0
    for r in rows:
        aC, aF2 = tables[r["layer"]]
        c = cidx[r["c_star"]]
        d = max(abs(aC[r["real_latent"], c] - r["auroc_C_cstar"]), abs(aF2[r["real_latent"], c] - r["auroc_F2_cstar"]))
        maxdiff = max(maxdiff, float(d))
        mism += d > 6e-4                                   # key_check rounds to 4 decimals
    print(f"key_check AUROC_C/F2(c*) reproduced for {len(rows)} pooled latents: max |diff| {maxdiff:.2e}, "
          f">6e-4: {mism}", flush=True)

    # ---------------------------------------------------------------- question 2 for every topic concept
    pc = {L: pool_conditions(T, L) for L in LAYERS}
    kc = {(r["layer"], r["real_latent"]): r for r in rows}
    topic_concepts = [i for i, c in enumerate(cs) if c["source"] != "language" and T.eligible[i]]
    q2 = {}
    for ci in topic_concepts:
        c = cids[ci]
        found = []
        for L in LAYERS:
            aC, aF2 = tables[L]
            best, cond, inpool, n_near = pc[L]
            js = np.nonzero((aC[:, ci] >= THR) & (aF2[:, ci] >= THR))[0]
            for j in js:
                j = int(j)
                b = int(best[j])
                fails = [k for k, v in cond.items() if not v[j]]
                A = T.aA[L][j]
                # counterfactual: the same rules evaluated as if c were the anchor
                cf = {"auroc_A>=0.90": A[ci] >= P_THR, "auroc_B>=0.85": T.aB[L][j, ci] >= P_THR_B,
                      "fire_A>=0.5": T.fire[L][j, ci] >= FIRE_MIN,
                      "specific(<=3 within 0.10)": int((A >= A[ci] - MARGIN).sum()) <= MAX_SPECIFIC,
                      "own_name_silent": T.names[L][3 * ci:3 * ci + 3, j].max() <= 0}
                found.append({
                    "layer": L, "real_latent": j, "auroc_C": round(float(aC[j, ci]), 4),
                    "auroc_F2": round(float(aF2[j, ci]), 4), "auroc_A": round(float(A[ci]), 4),
                    "auroc_B": round(float(T.aB[L][j, ci]), 4), "fire_A": round(float(T.fire[L][j, ci]), 3),
                    "a_argmax": cids[b], "a_argmax_is_c": b == ci, "auroc_A_argmax": round(float(A[b]), 4),
                    "n_within_0.10_of_argmax": int(n_near[j]),
                    "in_pool": bool(inpool[j]), "pool_anchor": cids[b] if inpool[j] else None,
                    "pool_rules_failed": fails,
                    "names_c_max_act": [round(float(x), 3) for x in T.names[L][3 * ci:3 * ci + 3, j]],
                    "counterfactual_rules_failed_if_anchor_c": [k for k, v in cf.items() if not v],
                    "generic_fire_rate_A": round(float(T.rate[L][j]), 4),
                    "key_check": ({k: kc[(L, j)][k] for k in ("c_star", "kept", "reasons")} if (L, j) in kc else None),
                })
        found.sort(key=lambda x: -min(x["auroc_C"], x["auroc_F2"]))
        # strongest latent for c at all (max over latents of min(AUROC_C, AUROC_F2)), for context
        best_any = max(((float(np.minimum(tables[L][0][:, ci], tables[L][1][:, ci]).max()), L) for L in LAYERS))
        q2[c] = {"label": cs[ci]["label"], "n_robust": len(found),
                 "n_robust_by_layer": {L: sum(f["layer"] == L for f in found) for L in LAYERS},
                 "best_min_auroc_C_F2": round(best_any[0], 4), "best_layer": best_any[1], "robust": found}

    sample_concepts = [s["c_star"] for s in sample if s["group"] == "dropped"]
    json.dump({"seed": SEED, "rule": "AUROC_C(j,c) >= 0.85 and AUROC_F2(j,c) >= 0.85, all 16384 latents x 3 layers",
               "sample_concepts": sample_concepts, "per_concept": q2},
              open(os.path.join(OUT, "q2_robust_latents.json"), "w"), separators=(",", ":"))

    # ---------------------------------------------------------------- the GPU job spec (question 1 + Q2 exemplars)
    job = []
    for s in sample:
        ci = cidx[s["c_star"]]
        L, j = s["layer"], s["real_latent"]
        A = [cs[ci]["A"][k] for k in s["A_idx"]]
        fidx = np.nonzero((frows == ci) & f2)[0]
        F = [(fstyles[k], ftexts[k]) for k in fidx]
        XA = np.load(os.path.join(src, f"acts_L{L}_A.npy"), mmap_mode="r")
        col = np.asarray(XA[:, j], np.float32)
        other = np.nonzero(rowsA != ci)[0]
        top = other[np.argsort(-col[other], kind="stable")[:N_CTX]]
        ctx = []
        for k in top:
            cc = int(rowsA[k])
            pos = int(k - np.nonzero(rowsA == cc)[0][0])
            ctx.append({"cid": cids[cc], "A_pos": pos, "cached_max": float(col[k]), "text": cs[cc]["A"][pos]})
        job.append({"kind": "q1", "group": s["group"], "layer": L, "real_latent": j, "c_star": s["c_star"],
                    "label": cs[ci]["label"], "kept": s["kept"], "reasons": s["reasons"],
                    "auroc_C_cstar": s["auroc_C_cstar"], "auroc_F1_cstar": s["auroc_F1_cstar"],
                    "auroc_F2_cstar": s["auroc_F2_cstar"], "k_ms": s["k_ms"], "k_ho": s["k_ho"], "top3_A": s["top3_A"],
                    "A_idx": s["A_idx"], "F2_rows": [int(k) for k in fidx],
                    "texts": [{"set": "A", "i": int(i), "text": t} for i, t in zip(s["A_idx"], A)]
                    + [{"set": "F2", "i": int(k), "style": st, "text": t} for k, (st, t) in zip(fidx, F)]
                    + [{"set": "ctx", "cid": x["cid"], "i": x["A_pos"], "text": x["text"]} for x in ctx]})
    # Q2 exemplars: the strongest robust latent of each sampled dropped concept that has one, on the same texts + names
    for s in sample[:N_DROP]:
        c = s["c_star"]
        if not q2[c]["robust"]:
            continue
        r = q2[c]["robust"][0]
        ci = cidx[c]
        q1 = next(x for x in job if x["c_star"] == c and x["group"] == "dropped")
        names = meta["names"][3 * ci:3 * ci + 3]
        job.append({"kind": "q2", "group": "robust_exemplar", "layer": r["layer"], "real_latent": r["real_latent"],
                    "c_star": c, "label": cs[ci]["label"], "robust_info": r,
                    "texts": [t for t in q1["texts"] if t["set"] in ("A", "F2")]
                    + [{"set": "name", "i": v, "text": n} for v, n in enumerate(names)]})
    json.dump(job, open(os.path.join(NPY, "gpu_job_spec.json"), "w"), indent=1, ensure_ascii=False)
    print(f"sample: {sum(s['group'] == 'dropped' for s in sample)} dropped, {sum(s['group'] == 'kept' for s in sample)} "
          f"kept; GPU job: {len(job)} latents, {sum(len(x['texts']) for x in job)} texts", flush=True)


if __name__ == "__main__":
    sys.exit(main())
