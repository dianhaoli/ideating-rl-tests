"""Independent re-implementation of the PREREG A1.1 keep rule (PREREG Amendment 3, A3.3).

Written without reading style_filter.py's analysis code or write_v2f.py. Inputs read here, all read-only:
  * generator v2 source cache  ~/wt/featurematch/tasks/featurematch/cache
      acts_L{6,12,18}_{A,C}.npy (float16 [n_texts, 16384]), precompute_meta.json (rowsA, rowsC, cids),
      auroc_L*_{A,B}.npy etc. (only through generate.Tables, to get the v2 pool and the generator's anchor c*)
  * bank F activations         diagnosis/cache/bank_acts_L{6,12,18}_F.npy + bank_meta.json
  * bank F texts               diagnosis/banks/F/*.json  ({"bank": "F", "concepts": {cid: [{style, text}, ...]}})

Rule (A1.1), per pooled latent j at layer L:
  c*   = argmax_c AUROC_A(j, c)                     (the generator's anchor, from its cached table)
  k_ms = argmax_c 0.5 AUROC_A(j, c) + 0.5 AUROC_F1(j, c)              over all 232 concepts
  k_ho = argmax_c 0.5 AUROC_C(j, c) + 0.5 AUROC_F2(j, c)  (AUROC_F2 alone if c has no split C)
  robust = AUROC_F2(j, c*) >= 0.85 and AUROC_C(j, c*) >= 0.85
  kept   = k_ms == c* and k_ho == c* and robust
F1 / F2 = the first / second text of each style, in the order the bank-F file lists them.
AUROC_X(j, c): positives = c's texts in X, negatives = all other texts in X; ties count 1/2. Computed here with a
sort + searchsorted count in float64 (not with rankdata), so an exact threshold value (e.g. 0.85) is compared exactly.

Two variants of AUROC_A inside M are run: "f64" (recomputed from acts_A) and "f16" (the generator's float16 cached
table, cast to float32 as generate.Tables does). Everything else is recomputed from raw activations.

Run (CPU, from ~/wt/fmdiag, after `source ~/ideating-rl-tests/common/env.sh`):
  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=2G -- /usr/bin/time -v \
      $PY tasks/featurematch/diagnosis/verify_reimpl/reimpl_keep_rule.py
Writes into diagnosis/verify_reimpl/: result.json (headline numbers, comparison), mismatches.jsonl and\nper_latent.jsonl.gz (one row per pooled latent).
"""
import glob
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HOME = os.path.expanduser("~")
SRC = os.path.join(HOME, "wt/featurematch/tasks/featurematch/cache")
DIAG = os.path.join(HOME, "wt/fmdiag/tasks/featurematch/diagnosis")
OUT = os.path.join(DIAG, "verify_reimpl")
V2_INST = os.path.join(HOME, "wt/featurematch/tasks/featurematch/instances")
LAYERS = (6, 12, 18)
THR_NUM, THR_DEN = 17, 20      # 0.85 as an exact fraction


# ----------------------------------------------------------------------------------------------- AUROC (own code)
def auroc_counts(X, rows, n_c):
    """X [N, D] float32, rows [N] concept index. Returns (U2 [D, n_c] float64, n_pos [n_c] int, N).
    U2 = 2 * Mann-Whitney U of concept c's texts vs all other texts (an integer, held exactly in float64):
    U = sum over (pos p, neg q) of [x_p > x_q] + 1/2 [x_p == x_q]. AUROC = U2 / (2 n_pos (N - n_pos))."""
    rows = np.asarray(rows)
    N, D = X.shape
    n_pos = np.bincount(rows, minlength=n_c)
    U2 = np.zeros((D, n_c), dtype=np.float64)
    for d in range(D):
        x = X[:, d]
        s = np.sort(x)
        lo = np.searchsorted(s, x, side="left")      # texts strictly below x_i (all concepts, including own)
        hi = np.searchsorted(s, x, side="right")     # texts <= x_i
        score2 = (lo + hi).astype(np.float64)        # 2*(less + 1/2 equal), equal includes the text itself
        S = np.bincount(rows, weights=score2, minlength=n_c)
        # subtract the within-concept part: sum over i in c of 2*(less_in_c + 1/2 eq_in_c) = n_pos^2
        U2[d] = S - n_pos.astype(np.float64) ** 2
    return U2, n_pos, N


def auroc_from_counts(U2, n_pos, N):
    den = 2.0 * n_pos * (N - n_pos)
    with np.errstate(invalid="ignore", divide="ignore"):
        A = U2 / den[None, :]
    A[:, n_pos == 0] = np.nan
    return A


def at_least_085(U2_col, n_pos_c, N):
    """Exact test AUROC >= 17/20: U2 / (2 n (N-n)) >= 17/20  <=>  20 U2 >= 34 n (N-n)."""
    return THR_DEN * U2_col >= 2 * THR_NUM * n_pos_c * (N - n_pos_c)


# ----------------------------------------------------------------------------------------------- bank F split
def text_hash(t):
    """bank_meta per_text_sha256 convention: sha256 of the whitespace-normalised text capped at max_chars (400),
    first 16 hex chars (the same normalisation as concepts.py). Checked: matches all 4640 bank-F rows."""
    return hashlib.sha256(" ".join(t.split())[:400].encode("utf-8")).hexdigest()[:16]


def bank_f_split(cids):
    """Return arrays (rows_F1, rows_F2) of bank_acts row indices, and per-row concept index, after checking
    bank_meta's row order and text hashes against the bank-F files."""
    meta = json.load(open(os.path.join(DIAG, "cache/bank_meta.json")))
    assert meta["cids"] == cids, "bank_meta cids differ from precompute_meta cids"
    bm = meta["banks"]["F"]
    n = bm["n"]
    texts = {}
    for f in sorted(glob.glob(os.path.join(DIAG, "banks/F/*.json"))):
        j = json.load(open(f))
        assert j["bank"] == "F"
        for cid, lst in j["concepts"].items():
            assert cid not in texts
            texts[cid] = lst
    assert set(texts) == set(cids)
    # map (concept index, hash) -> cache row
    row_of = {}
    for r in range(n):
        key = (bm["rows"][r], bm["per_text_sha256"][r])
        assert key not in row_of, f"duplicate text hash in concept {key}"
        row_of[key] = r
    f1, f2 = [], []
    n_matched = 0
    order_same_as_cache = True
    for ci, cid in enumerate(cids):
        seen = Counter()
        for k, item in enumerate(texts[cid]):
            h = text_hash(item["text"])
            r = row_of[(ci, h)]                                     # KeyError = hash convention / content mismatch
            assert bm["styles"][r] == item["style"]
            n_matched += 1
            seen[item["style"]] += 1
            (f1 if seen[item["style"]] == 1 else f2).append(r)
            assert seen[item["style"]] <= 2
        assert len(seen) == 10 and all(v == 2 for v in seen.values()), (cid, seen)
    assert n_matched == n
    # is the cache row order the file order? (informational)
    flat = []
    for ci, cid in enumerate(cids):
        for item in texts[cid]:
            flat.append(row_of[(ci, text_hash(item["text"]))])
    order_same_as_cache = flat == list(range(n))
    rows = np.asarray(bm["rows"])
    return np.asarray(f1), np.asarray(f2), rows, {"n_bank_rows": n, "all_hashes_matched": True,
                                                   "cache_row_order_equals_file_order": order_same_as_cache,
                                                   "n_F1": len(f1), "n_F2": len(f2)}


# ----------------------------------------------------------------------------------------------- generator pool
def generator_tables():
    sys.path.insert(0, os.path.join(HOME, "wt/fmdiag"))
    from tasks.featurematch import concepts as C
    C.CACHE = SRC
    C.OUT = os.path.join(SRC, "concepts.json")
    from tasks.featurematch import generate as G
    return G.Tables()


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def top_gap(v):
    """(argmax with first-index tie break, gap to the runner-up, number of exact ties at the max)."""
    o = np.argsort(-v, kind="stable")
    return int(o[0]), float(v[o[0]] - v[o[1]]), int((v == v[o[0]]).sum())


def main():
    t0 = time.time()
    pm = json.load(open(os.path.join(SRC, "precompute_meta.json")))
    cids = pm["cids"]
    n_c = len(cids)
    family = np.array([c.split(":")[0] for c in cids])        # "lang" / "topic"
    rowsA, rowsC = np.asarray(pm["rowsA"]), np.asarray(pm["rowsC"])
    hasC = np.bincount(rowsC, minlength=n_c) > 0
    f1, f2, rowsF, fmeta = bank_f_split(cids)
    rowsF1, rowsF2 = rowsF[f1], rowsF[f2]
    assert (np.bincount(rowsF1, minlength=n_c) == 10).all() and (np.bincount(rowsF2, minlength=n_c) == 10).all()

    T = generator_tables()
    assert [c["cid"] for c in T.cs] == cids
    max_dA = {}
    per_latent = []
    for L in LAYERS:
        pool = T.planted[L]                                   # [(j, c*)], j ascending
        cols = np.array([j for j, _ in pool])
        cstar = np.array([c for _, c in pool])
        # c* check: generator's anchor = argmax of the cached float16 table
        assert (T.aA[L][cols].argmax(1) == cstar).all()
        mmA = np.load(os.path.join(SRC, f"acts_L{L}_A.npy"), mmap_mode="r")
        XA = np.asarray(mmA[:, cols], dtype=np.float32)
        del mmA
        U2A, nA, NA = auroc_counts(XA, rowsA, n_c)
        del XA
        mmC = np.load(os.path.join(SRC, f"acts_L{L}_C.npy"), mmap_mode="r")
        XC = np.asarray(mmC[:, cols], dtype=np.float32)
        del mmC
        U2C, nC, NC = auroc_counts(XC, rowsC, n_c)
        del XC
        mmF = np.load(os.path.join(DIAG, f"cache/bank_acts_L{L}_F.npy"), mmap_mode="r")
        XF = np.asarray(mmF[:, cols], dtype=np.float32)
        del mmF
        U2F1, nF1, NF1 = auroc_counts(XF[f1], rowsF1, n_c)
        U2F2, nF2, NF2 = auroc_counts(XF[f2], rowsF2, n_c)
        U2F, nF, NF = auroc_counts(XF, rowsF, n_c)            # all of bank F (PREREG's original metric, info only)
        del XF
        aA64 = auroc_from_counts(U2A, nA, NA)
        aA16 = T.aA[L][cols].astype(np.float64)               # generator's cached float16 table (as float32 -> f64)
        aC = auroc_from_counts(U2C, nC, NC)
        aF1 = auroc_from_counts(U2F1, nF1, NF1)
        aF2 = auroc_from_counts(U2F2, nF2, NF2)
        aF = auroc_from_counts(U2F, nF, NF)
        H = np.where(hasC[None, :], 0.5 * aC + 0.5 * aF2, aF2)
        # sanity: own float64 AUROC_A vs the generator's float16 table (should differ only by float16 rounding)
        max_dA[L] = float(np.abs(aA64 - aA16).max())
        M64 = 0.5 * aA64 + 0.5 * aF1
        M16 = 0.5 * aA16 + 0.5 * aF1
        assert not np.isnan(H).any() and not np.isnan(M64).any()
        for i, j in enumerate(cols):
            cs = int(cstar[i])
            r = {"layer": L, "real_latent": int(j), "c_star": cids[cs], "family": family[cs]}
            ca64, gA64, tA = top_gap(aA64[i])
            r["c_star_f64"] = cids[ca64]
            r["gapA_f64"] = gA64
            if ca64 != cs:
                # all-float64 alternative: anchor = argmax of the recomputed AUROC_A, keep rule applied to it
                alt_ms = int(np.argmax(M64[i]))
                alt_ho = int(np.argmax(H[i]))
                alt_rob = bool(at_least_085(U2F2[i, ca64], nF2[ca64], NF2)) and bool(at_least_085(U2C[i, ca64], nC[ca64], NC))
                r["alt_all_f64"] = {"c_star": cids[ca64], "k_ms": cids[alt_ms], "k_ho": cids[alt_ho],
                                    "auroc_F2": float(aF2[i, ca64]), "auroc_C": float(aC[i, ca64]),
                                    "kept": alt_ms == ca64 and alt_ho == ca64 and alt_rob}
            for tag, M in (("f64", M64), ("f16", M16)):
                k, g, t = top_gap(M[i])
                r[f"k_ms_{tag}"] = cids[k]
                r[f"gapM_{tag}"] = g
                r[f"tiesM_{tag}"] = t
                r[f"M_cstar_{tag}"] = float(M[i, cs])
                r[f"M_kms_{tag}"] = float(M[i, k])
                r[f"top3_M_{tag}"] = [[cids[c], round(float(M[i, c]), 6)] for c in np.argsort(-M[i], kind="stable")[:3]]
            k, g, t = top_gap(H[i])
            r["k_ho"], r["gapH"], r["tiesH"] = cids[k], g, t
            r["H_cstar"], r["H_kho"] = float(H[i, cs]), float(H[i, k])
            r["top3_H"] = [[cids[c], round(float(H[i, c]), 6)] for c in np.argsort(-H[i], kind="stable")[:3]]
            r["top3_A_f16"] = [[cids[c], round(float(aA16[i, c]), 6)] for c in np.argsort(-aA16[i], kind="stable")[:3]]
            r["auroc_A_cstar_f64"] = float(aA64[i, cs])
            r["auroc_C_cstar"] = float(aC[i, cs])
            r["auroc_F1_cstar"] = float(aF1[i, cs])
            r["auroc_F2_cstar"] = float(aF2[i, cs])
            r["auroc_F_cstar"] = float(aF[i, cs])
            r["robust_F2"] = bool(at_least_085(U2F2[i, cs], nF2[cs], NF2))
            r["robust_C"] = bool(at_least_085(U2C[i, cs], nC[cs], NC))
            r["robust_F_all"] = bool(at_least_085(U2F[i, cs], nF[cs], NF))
            r["F2_at_085_exact"] = bool(THR_DEN * U2F2[i, cs] == 2 * THR_NUM * nF2[cs] * (NF2 - nF2[cs]))
            r["C_at_085_exact"] = bool(THR_DEN * U2C[i, cs] == 2 * THR_NUM * nC[cs] * (NC - nC[cs]))
            for tag in ("f64", "f16"):
                reasons = []
                if r[f"k_ms_{tag}"] != r["c_star"]:
                    reasons.append("ms_key_differs")
                if r["k_ho"] != r["c_star"]:
                    reasons.append("heldout_key_differs")
                if not (r["robust_F2"] and r["robust_C"]):
                    reasons.append("not_style_robust")
                r[f"reasons_{tag}"] = reasons
                r[f"kept_{tag}"] = not reasons
            per_latent.append(r)
        print(f"L{L}: {len(cols)} latents, {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------------------------------- compare
    ref = {(r["layer"], r["real_latent"]): r for r in map(json.loads, open(os.path.join(DIAG, "key_check.jsonl")))}
    mine = {(r["layer"], r["real_latent"]): r for r in per_latent}
    same_set = set(ref) == set(mine)
    cmp = {"same_latent_set": same_set, "n_ref": len(ref), "n_mine": len(mine)}
    mism = []
    fields = Counter()
    max_abs = defaultdict(float)
    for key in sorted(mine):
        a, b = mine[key], ref.get(key)
        if b is None:
            continue
        for fld in ("auroc_C_cstar", "auroc_F1_cstar", "auroc_F2_cstar"):
            max_abs[fld] = max(max_abs[fld], abs(a[fld] - b[fld]))
        max_abs["top3_A_scores"] = max(max_abs["top3_A_scores"],
                                       max(abs(x[1] - y[1]) for x, y in zip(a["top3_A_f16"], b["top3_A"])))
        max_abs["top3_H_scores"] = max(max_abs["top3_H_scores"],
                                       max(abs(x[1] - y[1]) for x, y in zip(a["top3_H"], b["top3_H"])))
        for tag in ("f64", "f16"):
            max_abs[f"top3_M_scores_{tag}"] = max(max_abs[f"top3_M_scores_{tag}"],
                                                  max(abs(x[1] - y[1]) for x, y in zip(a[f"top3_M_{tag}"], b["top3_M"])))
        diffs = {}
        if a["c_star"] != b["c_star"]:
            diffs["c_star"] = (a["c_star"], b["c_star"])
        for tag in ("f64", "f16"):
            if a[f"k_ms_{tag}"] != b["k_ms"]:
                diffs[f"k_ms_{tag}"] = (a[f"k_ms_{tag}"], b["k_ms"])
            if a[f"kept_{tag}"] != b["kept"]:
                diffs[f"kept_{tag}"] = (a[f"kept_{tag}"], b["kept"])
            if sorted(a[f"reasons_{tag}"]) != sorted(b["reasons"]):
                diffs[f"reasons_{tag}"] = (a[f"reasons_{tag}"], b["reasons"])
        if a["k_ho"] != b["k_ho"]:
            diffs["k_ho"] = (a["k_ho"], b["k_ho"])
        if [x[0] for x in a["top3_H"]] != [x[0] for x in b["top3_H"]]:
            diffs["top3_H_order"] = True
        if [x[0] for x in a["top3_A_f16"]] != [x[0] for x in b["top3_A"]]:
            diffs["top3_A_order"] = True
        rob_ref = "not_style_robust" not in b["reasons"]
        if (a["robust_F2"] and a["robust_C"]) != rob_ref:
            diffs["robust"] = (a["robust_F2"] and a["robust_C"], rob_ref)
        for k in diffs:
            fields[k] += 1
        if diffs:
            m = {"layer": key[0], "real_latent": key[1], "c_star": a["c_star"], "diffs": diffs}
            # near-tie diagnostics
            m["gapM_f64"], m["gapM_f16"], m["gapH"], m["gapA_f64"] = a["gapM_f64"], a["gapM_f16"], a["gapH"], a["gapA_f64"]
            m["top3_M_f64"], m["top3_M_f16"], m["top3_M_ref"] = a["top3_M_f64"], a["top3_M_f16"], b["top3_M"]
            m["auroc_F2_cstar"], m["auroc_C_cstar"] = a["auroc_F2_cstar"], a["auroc_C_cstar"]
            mism.append(m)
    cmp["n_latents_with_any_diff"] = len(mism)
    cmp["diff_field_counts"] = dict(fields)
    cmp["max_abs_diff_vs_ref_rounded_values"] = {k: round(v, 6) for k, v in max_abs.items()}

    # ------------------------------------------------------------------------------------------- headline numbers
    def headline(tag):
        rows = per_latent
        out = {}
        n = len(rows)
        k = sum(r[f"kept_{tag}"] for r in rows)
        out["kept"] = {"k": k, "n": n, "frac": round(k / n, 4), "wilson95": wilson(k, n)}
        tab = {}
        for L in LAYERS:
            for fam in ("lang", "topic", "all"):
                sub = [r for r in rows if r["layer"] == L and (fam == "all" or r["family"] == fam)]
                kk = sum(r[f"kept_{tag}"] for r in sub)
                tab[f"L{L}_{fam}"] = {"kept": kk, "n": len(sub), "frac": round(kk / len(sub), 4),
                                      "wilson95": wilson(kk, len(sub)),
                                      "concepts_with_kept_latent": len({r["c_star"] for r in sub if r[f"kept_{tag}"]})}
        for fam in ("lang", "topic"):
            sub = [r for r in rows if r["family"] == fam]
            kk = sum(r[f"kept_{tag}"] for r in sub)
            tab[f"all_{fam}"] = {"kept": kk, "n": len(sub), "frac": round(kk / len(sub), 4), "wilson95": wilson(kk, len(sub))}
        out["by_layer_family"] = tab
        shares = {}
        for name, fn in (("k_ms_eq_cstar", lambda r: r[f"k_ms_{tag}"] == r["c_star"]),
                         ("k_ho_eq_cstar", lambda r: r["k_ho"] == r["c_star"]),
                         ("robust", lambda r: r["robust_F2"] and r["robust_C"]),
                         ("robust_F2_part", lambda r: r["robust_F2"]),
                         ("robust_C_part", lambda r: r["robust_C"]),
                         ("auroc_F_all_ge_085", lambda r: r["robust_F_all"])):
            d = {}
            for grp, sel in (("all", lambda r: True), ("lang", lambda r: r["family"] == "lang"),
                             ("topic", lambda r: r["family"] == "topic"), ("L6", lambda r: r["layer"] == 6),
                             ("L12", lambda r: r["layer"] == 12), ("L18", lambda r: r["layer"] == 18)):
                sub = [r for r in rows if sel(r)]
                kk = sum(bool(fn(r)) for r in sub)
                d[grp] = {"k": kk, "n": len(sub), "frac": round(kk / len(sub), 4), "wilson95": wilson(kk, len(sub))}
            shares[name] = d
        out["shares"] = shares
        out["reason_combinations"] = dict(Counter("+".join(r[f"reasons_{tag}"]) or "kept" for r in rows).most_common())
        out["reason_counts_with_overlap"] = dict(Counter(x for r in rows for x in r[f"reasons_{tag}"]))
        out["dropped_only_for_key_disagreement"] = sum(
            1 for r in rows if r[f"reasons_{tag}"] and "not_style_robust" not in r[f"reasons_{tag}"])
        return out

    # ------------------------------------------------------------------------------------------- v2 slots (extra)
    def v2_slots(tag):
        if not os.path.isdir(V2_INST):
            return None
        cnt = Counter()
        for d in sorted(os.listdir(V2_INST)):
            p = os.path.join(V2_INST, d, "instance.json")
            if not os.path.exists(p):
                continue
            inst = json.load(open(p))
            for s in inst["extra"]["slots"]:
                r = mine.get((s["layer"], s["real_latent"]))
                kind = "planted" if s["kind"] == "planted" else "null"
                cnt[(kind, "n")] += 1
                cnt[(kind, "fail")] += int(not r[f"kept_{tag}"])
                cnt[("anchor_matches_cstar", s["anchor"] == r["c_star"])] += 1
        out = {}
        for kind in ("planted", "null"):
            n, f = cnt[(kind, "n")], cnt[(kind, "fail")]
            out[kind] = {"fail": f, "n": n, "frac": round(f / n, 4), "wilson95": wilson(f, n)}
        n, f = out["planted"]["n"] + out["null"]["n"], out["planted"]["fail"] + out["null"]["fail"]
        out["all"] = {"fail": f, "n": n, "frac": round(f / n, 4), "wilson95": wilson(f, n)}
        out["anchor_equals_cstar_all_slots"] = cnt[("anchor_matches_cstar", False)] == 0
        return out

    exact_ties = {"M_f64": sum(r["tiesM_f64"] > 1 for r in per_latent), "M_f16": sum(r["tiesM_f16"] > 1 for r in per_latent),
                  "H": sum(r["tiesH"] > 1 for r in per_latent)}
    at_thr = {"F2_exactly_085": sum(r["F2_at_085_exact"] for r in per_latent),
              "C_exactly_085": sum(r["C_at_085_exact"] for r in per_latent),
              "F2_exactly_085_and_otherwise_kept": sum(r["F2_at_085_exact"] and r["robust_C"] and r["k_ho"] == r["c_star"]
                                                        and r["k_ms_f16"] == r["c_star"] for r in per_latent)}
    cstar_f64_differs = [{"layer": r["layer"], "real_latent": r["real_latent"], "c_star_generator": r["c_star"],
                          "c_star_f64": r["c_star_f64"], "gapA_f64": round(r["gapA_f64"], 6),
                          "kept_under_A1.1_as_implemented": r["kept_f16"], "alt_all_f64": r["alt_all_f64"]}
                         for r in per_latent if r["c_star_f64"] != r["c_star"]]
    # how close did any decision come to flipping? float16 rounding of AUROC_A moves M by <= 0.5 * 2^-12 = 1.22e-4
    # per concept, so an M gap < 2.44e-4 could flip between the f16 and f64 tables.
    eps16 = 2.0 ** -12
    def thr_dist(v):
        return abs(v - 0.85)
    margins = {
        "min_gapM_f16": min(r["gapM_f16"] for r in per_latent),
        "n_gapM_f16_lt_2.44e-4": sum(r["gapM_f16"] < eps16 for r in per_latent),
        "n_gapM_f64_lt_2.44e-4": sum(r["gapM_f64"] < eps16 for r in per_latent),
        "min_gapH": min(r["gapH"] for r in per_latent),
        "n_gapH_lt_1e-3": sum(r["gapH"] < 1e-3 for r in per_latent),
        "min_dist_F2_cstar_to_085": min(thr_dist(r["auroc_F2_cstar"]) for r in per_latent),
        "n_F2_cstar_within_1e-4_of_085": sum(thr_dist(r["auroc_F2_cstar"]) < 1e-4 for r in per_latent),
        "min_dist_C_cstar_to_085": min(thr_dist(r["auroc_C_cstar"]) for r in per_latent),
        "n_C_cstar_within_1e-4_of_085": sum(thr_dist(r["auroc_C_cstar"]) < 1e-4 for r in per_latent),
        "note": "F2 AUROCs are multiples of 1/46200 (10 pos x 2310 neg, half-ties), C AUROCs multiples of 1/175200; "
                "the threshold test is done exactly on integer U statistics",
    }
    res = {
        "what": "Independent re-implementation of PREREG A1.1 keep rule (A3.3 verification)",
        "inputs": {"src_cache": SRC, "bank_F_acts": os.path.join(DIAG, "cache/bank_acts_L*_F.npy"),
                   "bank_F_texts": os.path.join(DIAG, "banks/F"), "reference": os.path.join(DIAG, "key_check.jsonl")},
        "bank_F_split": fmeta,
        "n_concepts": n_c, "n_concepts_without_C": int((~hasC).sum()),
        "pool_size_per_layer": {f"L{L}": len(T.planted[L]) for L in LAYERS},
        "max_abs_own_f64_AUROC_A_minus_cached_f16": {f"L{L}": round(v, 6) for L, v in max_dA.items()},
        "exact_argmax_ties": exact_ties,
        "decision_margins": margins,
        "exact_threshold_hits": at_thr,
        "cstar_float64_recompute_differs_from_generator_anchor": cstar_f64_differs,
        "headline_f16A": headline("f16"),
        "headline_f64A": headline("f64"),
        "v2_slots_failing_f16A": v2_slots("f16"),
        "v2_slots_failing_f64A": v2_slots("f64"),
        "comparison_vs_key_check": cmp,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(OUT, "mismatches.jsonl"), "w") as f:
        for m in mism:
            f.write(json.dumps(m) + "\n")
    import gzip
    with gzip.open(os.path.join(OUT, "per_latent.jsonl.gz"), "wt") as f:
        for r in per_latent:
            f.write(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}) + "\n")
    print(json.dumps({"kept_f16A": res["headline_f16A"]["kept"], "kept_f64A": res["headline_f64A"]["kept"],
                      "cmp": cmp}, indent=1))


if __name__ == "__main__":
    main()
