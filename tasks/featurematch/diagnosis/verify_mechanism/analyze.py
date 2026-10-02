"""Verify-mechanism, CPU analysis of search.py + gpu_tokens.py outputs.

Writes (all in this directory):
  q1_latents.jsonl   one row per sampled latent: key-check facts, cached fire rates on c*'s 40 split-A and 20 bank-F
                     texts, per-text peak (activation, peak token, the whole word it belongs to, context), scripted
                     peak-word shares, the scripted dominant type and the manual category (q1_categories.json).
  q2_exemplars.jsonl the same per-text peaks for the strongest style-robust latent of each sampled dropped concept,
                     plus its activations on the concept's 3 name variants.
  summary.json       every number quoted in the report, with n and 95% CIs.

Peak-word types (scripted; a "word" is the peak token plus the sub-word pieces glued to it):
  keyword   first 5 letters match K(c): c's label content words (>= 4 letters, no stopwords) plus the 15 words with the
            highest TF-IDF in c's split A+B against the other concepts, kept only if in >= 10% of c's texts
            (the PREREG 1.2 recipe for topic concepts).
  format    a digit, pure punctuation, or a frame word ("is", "was", "a", "born", "the", "of", ...).
  name      a capitalised word that is not the text's first word and not a keyword.
  other     anything else (a lower-case content word that is not a keyword).
Dominant type = the type of >= 50% of the firing split-A texts' peak words, else "mixed".

Run: systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=2G -- /usr/bin/time -v \
         $PY -m tasks.featurematch.diagnosis.verify_mechanism.analyze
"""
import gzip
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np

from tasks.featurematch.diagnosis import style_filter as SF
from tasks.featurematch.fm_core import LAYERS

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
SEED = 20261002
N_DROP, N_KEEP = 30, 15
THR = 0.85
STOP = set("""a an the of in on at by for from to and or with as is was are were be been being has have had its his her
their this that these those which who whom whose it he she they them also into than then there where when while after
before during under over about between such other only most more some many any each both very first second new one two
three known called named based located""".split())
FRAME = {"is", "was", "are", "were", "a", "an", "the", "of", "in", "on", "at", "by", "for", "from", "to", "and", "or",
         "with", "as", "born", "died", "which", "that", "who", "its", "his", "her", "their", "has", "had", "be", "been",
         "also", "it", "he", "she", "they"}
WORD = re.compile(r"[a-z]+")


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def boot_mean(x, B=10000):
    x = np.asarray(x, float)
    rng = random.Random(SEED)
    n = len(x)
    ms = sorted(float(np.mean([x[i] for i in rng.choices(range(n), k=n)])) for _ in range(B))
    return [round(float(np.mean(x)), 3), [round(ms[int(0.025 * B)], 3), round(ms[int(0.975 * B) - 1], 3)]]


def keywords(cs):
    """K(c) for every concept, as first-5-letter stems (PREREG 1.2, topic recipe)."""
    docs = [[set(WORD.findall(t.lower())) for t in c["A"] + c["B"]] for c in cs]
    tf = []
    for d in docs:
        cnt = Counter(w for s in d for w in s if len(w) >= 4 and w not in STOP)
        tf.append({w: v / len(d) for w, v in cnt.items()})
    df = Counter(w for t in tf for w in t)
    n = len(cs)
    K = []
    for c, t in zip(cs, tf):
        lab = {w[:5] for w in WORD.findall(c["label"].lower()) if len(w) >= 4 and w not in STOP}
        sc = sorted(((v * math.log(n / df[w]), w) for w, v in t.items() if v >= 0.10), reverse=True)[:15]
        K.append(lab | {w[:5] for _, w in sc})
    return K


def peak_word(tokens, p):
    """The whole word containing token p (Gemma pieces without a leading space continue the previous word)."""
    s = p
    while s > 0 and not tokens[s].startswith(" ") and re.match(r"\w", tokens[s] or " ") and re.match(r"\w", tokens[s - 1][-1:] or " "):
        s -= 1
    e = p + 1
    while e < len(tokens) and not tokens[e].startswith(" ") and re.match(r"\w", tokens[e] or " ") and re.match(r"\w", tokens[e - 1][-1:] or " "):
        e += 1
    return "".join(tokens[s:e]).strip(), s


def word_type(word, start, K5):
    w = word.strip()
    lw = w.lower()
    if any(ch.isdigit() for ch in w) or not any(ch.isalnum() for ch in w) or lw in FRAME:
        return "format"
    if lw[:5] in K5 and len(lw) >= 4:
        return "keyword"
    if w[:1].isupper() and start > 0:
        return "name"
    return "other"


def peaks(entry, rec, K5):
    out = []
    for t, r in zip(entry["texts"], rec["texts"]):
        acts, toks = np.array(r["acts"], float), r["tokens"]
        m = float(acts.max()) if len(acts) else 0.0
        row = {k: t[k] for k in ("set", "i", "style", "cid") if k in t}
        row["max"] = round(m, 3)
        row["n_tokens_firing"] = int((acts > 0).sum())
        if m > 0:
            p = int(acts.argmax())
            w, s = peak_word(toks, p)
            row.update({"peak_pos": p, "peak_token": toks[p], "peak_word": w, "type": word_type(w, s, K5),
                        "context": "".join(toks[max(0, p - 10):p]) + " [[" + toks[p] + "]]" + "".join(toks[p + 1:p + 5]),
                        "top3": [[toks[k], round(float(acts[k]), 2)] for k in np.argsort(-acts, kind="stable")[:3] if acts[k] > 0]})
        out.append(row)
    return out


def word_transfer(entry, rec):
    """Same word, other style: W = the content words (>= 4 letters, not a frame/stop word) on which the latent peaks in
    the firing split-A texts. Every occurrence of a W word in the 10 bank-F2 texts is checked: does the latent fire
    (> 0) on any token of that word? Returns (W, occurrences, fired occurrences, examples)."""
    W = set()
    for t, r in zip(entry["texts"], rec["texts"]):
        if t["set"] != "A" or not r["acts"] or max(r["acts"]) <= 0:
            continue
        w, _ = peak_word(r["tokens"], int(np.argmax(r["acts"])))
        w = w.lower()
        if w.isalpha() and len(w) >= 4 and w not in FRAME and w not in STOP:
            W.add(w)
    occ, fired, ex = 0, 0, []
    for t, r in zip(entry["texts"], rec["texts"]):
        if t["set"] != "F2":
            continue
        toks, acts = r["tokens"], r["acts"]
        p = 0
        while p < len(toks):
            w, s = peak_word(toks, p)
            e = p + 1
            while e < len(toks) and peak_word(toks, e)[1] == s:
                e += 1
            if w.lower() in W:
                occ += 1
                m = max(acts[s:e])
                fired += m > 0
                if len(ex) < 4:
                    ex.append({"style": t.get("style"), "word": w, "act": round(m, 2),
                               "context": "".join(toks[max(0, s - 8):e + 4])})
            p = e
    return sorted(W), occ, fired, ex


def shares(rows):
    f = [r for r in rows if r["max"] > 0]
    c = Counter(r["type"] for r in f)
    return {k: round(c[k] / len(f), 3) if f else None for k in ("keyword", "format", "name", "other")}, len(f)


def dominant(sh, n):
    if not n:
        return "silent"
    k, v = max(sh.items(), key=lambda kv: kv[1])
    return k if v >= 0.5 else "mixed"


def name_filter_table(T, tables, cs):
    """Among latents that pass every pool rule except the name filter, how often is the latent style-robust for its
    own anchor, split by whether it fires on the anchor's name (the filter's target)?"""
    res = {}
    for fam in ("topic", "language"):
        tot = defaultdict(lambda: [0, 0])
        for L in LAYERS:
            A, B, F = T.aA[L], T.aB[L], T.fire[L]
            best = A.argmax(1)
            ar = np.arange(len(A))
            bA, bB, bF = A[ar, best], B[ar, best], F[ar, best]
            n_near = (A >= (bA - 0.10)[:, None]).sum(1)
            sel = (bA >= 0.90) & (bB >= 0.85) & (bF >= 0.5) & (n_near <= 3) & T.eligible[best]
            isfam = np.array([(cs[b]["source"] == "language") == (fam == "language") for b in best])
            own = T.names[L].reshape(T.n_c, 3, -1)[best, :, ar].max(1)
            aC, aF2 = tables[L]
            rob = (aC[ar, best] >= THR) & (aF2[ar, best] >= THR)
            for lab, m in (("name_fires_excluded", sel & isfam & (own > 0)), ("name_silent_pooled", sel & isfam & (own <= 0))):
                tot[lab][0] += int((m & rob).sum())
                tot[lab][1] += int(m.sum())
                tot[f"{lab}_L{L}"] = [int((m & rob).sum()), int(m.sum())]
                if lab.startswith("name_fires"):
                    tot["_f2_fires"] = tot.get("_f2_fires", []) + list(aF2[ar, best][m])
                else:
                    tot["_f2_silent"] = tot.get("_f2_silent", []) + list(aF2[ar, best][m])
        out = {}
        for k, v in tot.items():
            if k.startswith("_"):
                continue
            out[k] = {"robust": v[0], "n": v[1], "rate": round(v[0] / v[1], 3) if v[1] else None, "wilson95": wilson(*v)}
        out["median_auroc_F2_anchor"] = {"name_fires_excluded": round(float(np.median(tot["_f2_fires"])), 3),
                                         "name_silent_pooled": round(float(np.median(tot["_f2_silent"])), 3)}
        res[fam] = out
    # 2 x 2: specificity rule x name filter, among topic latents that pass the other pool rules (A, B, fire, split C)
    grid = defaultdict(lambda: [0, 0])
    for L in LAYERS:
        A, B, F = T.aA[L], T.aB[L], T.fire[L]
        best = A.argmax(1)
        ar = np.arange(len(A))
        bA, bB, bF = A[ar, best], B[ar, best], F[ar, best]
        base = (bA >= 0.90) & (bB >= 0.85) & (bF >= 0.5) & T.eligible[best]
        base &= np.array([cs[b]["source"] != "language" for b in best])
        spec = (A >= (bA - 0.10)[:, None]).sum(1) <= 3
        silent = T.names[L].reshape(T.n_c, 3, -1)[best, :, ar].max(1) <= 0
        aC, aF2 = tables[L]
        rob = (aC[ar, best] >= THR) & (aF2[ar, best] >= THR)
        for sk, sm in (("specific", spec), ("not_specific", ~spec)):
            for nk, nm in (("name_silent", silent), ("name_fires", ~silent)):
                m = base & sm & nm
                grid[f"{sk}&{nk}"][0] += int((m & rob).sum())
                grid[f"{sk}&{nk}"][1] += int(m.sum())
    tot = [sum(v[0] for v in grid.values()), sum(v[1] for v in grid.values())]
    res["topic_2x2_specificity_x_name"] = {
        **{k: {"robust": v[0], "n": v[1], "rate": round(v[0] / v[1], 3), "wilson95": wilson(*v)} for k, v in grid.items()},
        "all_four_cells": {"robust": tot[0], "n": tot[1], "rate": round(tot[0] / tot[1], 3), "wilson95": wilson(*tot)},
        "note": "cell specific&name_silent is exactly generator v2's topic pool"}
    return res


def main():
    src = SF.default_src_cache()
    T = SF.load_tables(src)
    cs = T.cs
    cids = [c["cid"] for c in cs]
    cidx = {c: i for i, c in enumerate(cids)}
    meta = json.load(open(os.path.join(src, "precompute_meta.json")))
    rowsA = np.array(meta["rowsA"])
    startA = {c: int(np.nonzero(rowsA == c)[0][0]) for c in range(len(cids))}
    bmeta = json.load(open(os.path.join(DIAG, "cache", "bank_meta.json")))
    frows = np.array(bmeta["banks"]["F"]["rows"])
    K = keywords(cs)
    spec = json.load(open(os.path.join(HERE, "cache", "gpu_job_spec.json")))
    tok = json.load(gzip.open(os.path.join(HERE, "cache", "token_acts.json.gz"), "rt"))
    recs = tok["latents"]
    assert len(recs) == len(spec)
    manual = json.load(open(os.path.join(HERE, "q1_categories.json"))) if os.path.exists(os.path.join(HERE, "q1_categories.json")) else {}
    tables = {L: (np.load(os.path.join(HERE, "cache", f"auroc_C_L{L}.npy"), mmap_mode="r"),
                  np.load(os.path.join(HERE, "cache", f"auroc_F2_L{L}.npy"), mmap_mode="r")) for L in LAYERS}

    # --------------------------------------------------- consistency: per-token max (GPU now) vs cached max-pooled
    pairs = []
    for x, r in zip(spec, recs):
        assert (x["layer"], x["real_latent"]) == (r["layer"], r["real_latent"])
        L, j = x["layer"], x["real_latent"]
        XA = np.load(os.path.join(src, f"acts_L{L}_A.npy"), mmap_mode="r")
        XF = np.load(os.path.join(DIAG, "cache", f"bank_acts_L{L}_F.npy"), mmap_mode="r")
        for t, rr in zip(x["texts"], r["texts"]):
            m = max(rr["acts"]) if rr["acts"] else 0.0
            if t["set"] == "A":
                pairs.append((float(XA[startA[cidx[x["c_star"]]] + t["i"], j]), m))
            elif t["set"] == "ctx":
                pairs.append((float(XA[startA[cidx[t["cid"]]] + t["i"], j]), m))
            elif t["set"] == "F2":
                assert frows[t["i"]] == cidx[x["c_star"]]
                pairs.append((float(XF[t["i"], j]), m))
    P = np.array(pairs)
    cons = {"n_pairs": len(P), "pearson_r": round(float(np.corrcoef(P[:, 0], P[:, 1])[0, 1]), 5),
            "max_abs_diff": round(float(np.abs(P[:, 0] - P[:, 1]).max()), 3),
            "median_abs_diff": round(float(np.median(np.abs(P[:, 0] - P[:, 1]))), 4),
            "fire_agreement": round(float(((P[:, 0] > 0) == (P[:, 1] > 0)).mean()), 4)}
    print("consistency", cons, flush=True)

    # --------------------------------------------------- question 1
    q1 = []
    for x, r in zip(spec, recs):
        if x["kind"] != "q1":
            continue
        L, j, ci = x["layer"], x["real_latent"], cidx[x["c_star"]]
        XA = np.load(os.path.join(src, f"acts_L{L}_A.npy"), mmap_mode="r")
        XF = np.load(os.path.join(DIAG, "cache", f"bank_acts_L{L}_F.npy"), mmap_mode="r")
        a40 = np.asarray(XA[startA[ci]:startA[ci] + 40, j], np.float32)
        f20 = np.asarray(XF[np.nonzero(frows == ci)[0], j], np.float32)
        pk = peaks(x, r, K[ci])
        shA, nA = shares([p for p in pk if p["set"] == "A"])
        shF, nF = shares([p for p in pk if p["set"] == "F2"])
        key = f"L{L}/{j}"
        W, occ, fired, wex = word_transfer(x, r)
        row = {"key": key, "group": x["group"], "layer": L, "real_latent": j, "c_star": x["c_star"], "label": x["label"],
               "reasons": x["reasons"], "k_ms": x["k_ms"], "k_ho": x["k_ho"], "top3_A": x["top3_A"],
               "auroc_C_cstar": x["auroc_C_cstar"], "auroc_F1_cstar": x["auroc_F1_cstar"],
               "auroc_F2_cstar": x["auroc_F2_cstar"], "generic_fire_rate_A": round(float(T.rate[L][j]), 4),
               "fire_A40_cached": round(float((a40 > 0).mean()), 3), "fire_F20_cached": round(float((f20 > 0).mean()), 3),
               "mean_max_A40": round(float(a40.mean()), 3), "mean_max_F20": round(float(f20.mean()), 3),
               "n_A10_firing": nA, "n_F10_firing": nF, "peak_shares_A": shA, "peak_shares_F": shF,
               "scripted_type": dominant(shA, nA), "K_c": sorted(K[ci]),
               "word_transfer_F2": {"peak_words_A": W, "occurrences": occ, "fired": fired, "examples": wex},
               "manual": manual.get(key), "peaks": pk}
        q1.append(row)
    with open(os.path.join(HERE, "q1_latents.jsonl"), "w") as f:
        for row in q1:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --------------------------------------------------- question 2 exemplars
    ex = []
    for x, r in zip(spec, recs):
        if x["kind"] != "q2":
            continue
        ci = cidx[x["c_star"]]
        pk = peaks(x, r, K[ci])
        shA, nA = shares([p for p in pk if p["set"] == "A"])
        shF, nF = shares([p for p in pk if p["set"] == "F2"])
        ex.append({"key": f"L{x['layer']}/{x['real_latent']}", "c_star": x["c_star"], "label": x["label"],
                   "robust_info": x["robust_info"], "n_A10_firing": nA, "n_F10_firing": nF, "peak_shares_A": shA,
                   "peak_shares_F": shF, "scripted_type": dominant(shA, nA), "peaks": pk})
    with open(os.path.join(HERE, "q2_exemplars.jsonl"), "w") as f:
        for row in ex:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --------------------------------------------------- summary numbers
    summ = {"consistency_gpu_vs_cache": cons}
    for g in ("dropped", "kept"):
        rows = [q for q in q1 if q["group"] == g]
        n = len(rows)
        st = Counter(q["scripted_type"] for q in rows)
        mc = Counter((q["manual"] or {}).get("category") for q in rows)
        allA = [p for q in rows for p in q["peaks"] if p["set"] == "A" and p["max"] > 0]
        allF = [p for q in rows for p in q["peaks"] if p["set"] == "F2" and p["max"] > 0]
        summ[g] = {
            "n_latents": n,
            "scripted_type": {k: [v, wilson(v, n)] for k, v in st.most_common()},
            "manual_category": {str(k): [v, wilson(v, n)] for k, v in mc.most_common()},
            "fire_A40_cached_mean_boot95": boot_mean([q["fire_A40_cached"] for q in rows]),
            "fire_F20_cached_mean_boot95": boot_mean([q["fire_F20_cached"] for q in rows]),
            "auroc_F2_cstar_mean_boot95": boot_mean([q["auroc_F2_cstar"] for q in rows]),
            "generic_fire_rate_median": round(float(np.median([q["generic_fire_rate_A"] for q in rows])), 4),
            "A_peaks_pooled": {"n": len(allA), **{k: [v, round(v / len(allA), 3), wilson(v, len(allA))]
                                                for k, v in Counter(p["type"] for p in allA).most_common()}},
            "F_peaks_pooled": {"n": len(allF), **{k: [v, round(v / len(allF), 3), wilson(v, len(allF))]
                                                for k, v in Counter(p["type"] for p in allF).most_common()}},
            "median_peak_pos_A": float(np.median([p["peak_pos"] for p in allA])) if allA else None,
        }
        wt = [q["word_transfer_F2"] for q in rows]
        occ, fir = sum(w["occurrences"] for w in wt), sum(w["fired"] for w in wt)
        withocc = [w for w in wt if w["occurrences"]]
        summ[g]["word_transfer_F2"] = {
            "latents_with_content_peak_words": sum(bool(w["peak_words_A"]) for w in wt),
            "latents_whose_peak_word_occurs_in_F2": len(withocc),
            "occurrences": occ, "fired": fir, "rate": round(fir / occ, 3) if occ else None,
            "wilson95_occurrence_level(ignores_clustering)": wilson(fir, occ),
            "per_latent_rate_mean_boot95": boot_mean([w["fired"] / w["occurrences"] for w in withocc]) if withocc else None}
        # by manual category
        bycat = defaultdict(list)
        for q in rows:
            bycat[(q["manual"] or {}).get("category")].append(q)
        summ[g]["by_manual_category"] = {
            str(k): {"n": len(v), "fire_F20_mean": round(float(np.mean([q["fire_F20_cached"] for q in v])), 3),
                     "auroc_F2_mean": round(float(np.mean([q["auroc_F2_cstar"] for q in v])), 3),
                     "latents": [f"{q['c_star'][6:]} {q['key']}" for q in v]} for k, v in bycat.items()}
    from scipy.stats import fisher_exact
    cats = sorted({(q["manual"] or {}).get("category") for q in q1 if q["manual"]})
    summ["dropped_vs_kept_fisher"] = {}
    for cat in cats:
        a = sum((q["manual"] or {}).get("category") == cat for q in q1 if q["group"] == "dropped")
        b = sum((q["manual"] or {}).get("category") == cat for q in q1 if q["group"] == "kept")
        _, p = fisher_exact([[a, N_DROP - a], [b, N_KEEP - b]])
        summ["dropped_vs_kept_fisher"][cat] = {"dropped": [a, N_DROP], "kept": [b, N_KEEP], "p_two_sided": float(p)}
    # agreement scripted vs manual
    both = [(q["scripted_type"], (q["manual"] or {}).get("category")) for q in q1 if q["manual"]]
    summ["scripted_vs_manual"] = {"n": len(both), "pairs": Counter(f"{a}->{b}" for a, b in both).most_common()}
    # Q2 exemplars
    exn = len(ex)
    summ["q2_exemplars"] = {
        "n": exn,
        "scripted_type": dict(Counter(e["scripted_type"] for e in ex)),
        "fires_on_own_name": [sum(max(p["max"] for p in e["peaks"] if p["set"] == "name") > 0 for e in ex), exn],
        "in_pool": sum(e["robust_info"]["in_pool"] for e in ex),
        "A_peak_types": dict(Counter(p["type"] for e in ex for p in e["peaks"] if p["set"] == "A" and p["max"] > 0)),
        "F_peak_types": dict(Counter(p["type"] for e in ex for p in e["peaks"] if p["set"] == "F2" and p["max"] > 0)),
        "n_A10_firing_mean": round(float(np.mean([e["n_A10_firing"] for e in ex])), 2),
        "n_F10_firing_mean": round(float(np.mean([e["n_F10_firing"] for e in ex])), 2),
    }
    # Q2 per concept
    q2 = json.load(open(os.path.join(HERE, "q2_robust_latents.json")))
    pc = q2["per_concept"]

    def per_concept(c):
        rob = pc[c]["robust"]
        ci = cidx[c]
        depth = len(cs[ci]["path"]) - 1
        anch = [x for x in rob if x["a_argmax_is_c"]]
        inpool_c = [x for x in anch if x["in_pool"]]
        kept = [x for x in inpool_c if x["key_check"] and x["key_check"]["kept"]]
        elsewhere = [x for x in rob if not x["a_argmax_is_c"]]
        sib = [x for x in elsewhere if T.close[ci, cidx[x["a_argmax"]]] >= depth]
        why = Counter()
        for x in anch:
            if not x["in_pool"]:
                f = set(x["pool_rules_failed"])
                why["name_filter_only" if f == {"own_name_silent"} else
                    "not_specific_only" if f == {"specific(<=3 within 0.10)"} else
                    "name_filter+not_specific" if f == {"own_name_silent", "specific(<=3 within 0.10)"} else
                    "other:" + "+".join(sorted(f))] += 1
        return {"label": pc[c]["label"], "n_robust": len(rob), "by_layer": pc[c]["n_robust_by_layer"],
                "best_min_auroc_C_F2": pc[c]["best_min_auroc_C_F2"], "n_anchored_c": len(anch),
                "n_in_pool_anchored_c": len(inpool_c), "n_kept_A11": len(kept),
                "n_anchored_elsewhere": len(elsewhere), "n_elsewhere_sibling": len(sib),
                "elsewhere_top_anchors": Counter(x["a_argmax"] for x in elsewhere).most_common(3),
                "anchored_c_not_in_pool_why": dict(why),
                "n_fire_on_own_name": sum(max(x["names_c_max_act"]) > 0 for x in rob)}

    for name, cl in (("sample30", q2["sample_concepts"]), ("all_topic", sorted(pc))):
        rowsc = {c: per_concept(c) for c in cl}
        n = len(cl)
        has = sum(r["n_robust"] > 0 for r in rowsc.values())
        has_anch = sum(r["n_anchored_c"] > 0 for r in rowsc.values())
        has_pool = sum(r["n_in_pool_anchored_c"] > 0 for r in rowsc.values())
        has_kept = sum(r["n_kept_A11"] > 0 for r in rowsc.values())
        tot = Counter()
        for r in rowsc.values():
            tot["robust"] += r["n_robust"]
            tot["anchored_c"] += r["n_anchored_c"]
            tot["in_pool_anchored_c"] += r["n_in_pool_anchored_c"]
            tot["kept"] += r["n_kept_A11"]
            tot["anchored_elsewhere"] += r["n_anchored_elsewhere"]
            tot["elsewhere_sibling"] += r["n_elsewhere_sibling"]
            tot["fire_on_own_name"] += r["n_fire_on_own_name"]
            for k, v in r["anchored_c_not_in_pool_why"].items():
                tot["why:" + k] += v
        summ["q2_" + name] = {"n_concepts": n,
                              "with_robust_latent": [has, wilson(has, n)],
                              "with_robust_latent_anchored_c": [has_anch, wilson(has_anch, n)],
                              "with_robust_latent_in_pool_as_c": [has_pool, wilson(has_pool, n)],
                              "with_robust_latent_kept_A11": [has_kept, wilson(has_kept, n)],
                              "latent_totals": dict(tot)}
        if name == "sample30":
            summ["q2_sample30_per_concept"] = rowsc
    summ["name_filter_crosstab"] = name_filter_table(T, tables, cs)
    json.dump(summ, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "q2_sample30_per_concept"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
