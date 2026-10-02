"""Step-3 skeptic, part 2: an independent re-implementation of SR-max / SR-thr from the bank files and the cache.

Does NOT import sr_recipe.py or style_filter.py. Reads:
  diagnosis/banks/R/*.json, diagnosis/banks/F/*.json      (texts, styles, bank-file order)
  diagnosis/cache/bank_meta.json                           (cids order, per-text sha256 of the cached rows)
  diagnosis/cache/bank_acts_L{6,12,18}_R.npy               (mmap, only slot-latent columns copied, float32)
  ~/wt/featurematch/tasks/featurematch/cache/acts_L*_names.npy + precompute_meta.json (name probes; read-only)
  instances_v2f/*/instance.json                            (layer, real latent, menu concept ids, key)
  step3/sr_recipe_out/slots.csv, step3/runs/.../episodes_private.jsonl (builder outputs, for comparison only)
Writes: sr_indep.json (aggregates only).

Checks:
  1. Cache rows <-> bank-R texts: my own reading + normalisation + sha256[:16] must equal bank_meta's per-text hashes.
  2. Primary selection: random.Random(20261002).sample(sorted(10 styles), 6), first text per style.
  3. SR-max / SR-thr per slot; compare with the builder's slots.csv (pick, choice) slot by slot.
  4. 42 selections (seeds 20261002..20261022 x first/second text), pooled and by family.
  5. Bank F vs bank R overlap (are they really independent writers?): char-5-gram Jaccard of every R text against
     every F text of the same concept; exact duplicates across banks.
  6. name_probe sanity (builder's name_probe == always_claim on all 441 planted slots): offline from the names cache.
"""
import csv
import glob
import hashlib
import json
import os
import random
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
TASK = os.path.dirname(DIAG)
INST = os.path.join(TASK, "instances_v2f")
CACHE = os.path.join(DIAG, "cache")
FM_CACHE = os.path.expanduser("~/wt/featurematch/tasks/featurematch/cache")
SEED = 20261002


def read_bank(bank, cids):
    allc = {}
    for f in sorted(glob.glob(os.path.join(DIAG, "banks", bank, "chunk_*.json"))):
        allc.update(json.load(open(f))["concepts"])
    texts, rows, styles = [], [], []
    for i, c in enumerate(cids):
        for t in allc[c]:
            texts.append(" ".join(str(t["text"]).split())[:400])
            rows.append(i)
            styles.append(t["style"])
    return texts, rows, styles


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(neg) == 0:
        return 0.5
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def wilson(k, n, z=1.959963984540054):
    import math
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 4), round(min(1, c + h), 4)]


def load_slots():
    out = []
    for d in sorted(glob.glob(os.path.join(INST, "*"))):
        inst = json.load(open(os.path.join(d, "instance.json")))
        for i, (e, t) in enumerate(zip(inst["extra"]["slots"], inst["answer"]["slots"])):
            out.append({"iid": inst["instance_id"], "tier": inst["tier"], "slot": i, "layer": int(e["layer"]),
                        "latent": int(e["real_latent"]), "menu": e["menu"], "planted": t["planted"],
                        "truth": t["choice"], "anchor": e["anchor"],
                        "family": "language" if e["anchor"].startswith("lang:") else "topic"})
    return out


def ngrams(t, n=5):
    t = t.lower()
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def main():
    meta = json.load(open(os.path.join(CACHE, "bank_meta.json")))
    cids = meta["cids"]
    cidx = {c: i for i, c in enumerate(cids)}
    res = {}

    # 1. cache rows <-> my reading of bank R
    texts, rows, styles = read_bank("R", cids)
    h = [hashlib.sha256(t.encode()).hexdigest()[:16] for t in texts]
    R = meta["banks"]["R"]
    res["cache_rows_match_bank_files"] = {"n": len(texts), "per_text_hash_equal": h == R["per_text_sha256"],
                                          "rows_equal": rows == R["rows"], "styles_equal": styles == R["styles"]}

    # 2. primary selection
    style_names = sorted(set(styles))
    by_cs = defaultdict(list)
    for i, (r, s) in enumerate(zip(rows, styles)):
        by_cs[(r, s)].append(i)

    def selection(seed, ti):
        chosen = random.Random(seed).sample(style_names, 6)
        return chosen, {c: [by_cs[(c, s)][ti] for s in chosen] for c in range(len(cids))}
    chosen, sel = selection(SEED, 0)
    res["primary_styles"] = chosen

    # 3. per-slot SR
    slots = load_slots()
    need = defaultdict(set)
    for s in slots:
        need[s["layer"]].add(s["latent"])
    acts, col = {}, {}
    for L, lat in need.items():
        X = np.load(os.path.join(CACHE, f"bank_acts_L{L}_R.npy"), mmap_mode="r")
        lat = sorted(lat)
        acts[L] = np.asarray(X[:, lat], dtype=np.float32)
        col[L] = {j: k for k, j in enumerate(lat)}
        del X

    def decide(s, sel):
        a = acts[s["layer"]][:, col[s["layer"]][s["latent"]]]
        vals = [a[sel[cidx[c]]] for c in s["menu"]]
        means = np.array([v.mean() for v in vals])
        best = int(np.argmax(means))
        neg = np.concatenate([v for j, v in enumerate(vals) if j != best])
        A = auc(vals[best], neg)
        return best + 1, ((best + 1) if A >= 0.78 else "nothing found"), A, means

    builder = {}
    with open(os.path.join(DIAG, "step3", "sr_recipe_out", "slots.csv")) as f:
        for r in csv.DictReader(f):
            builder[(r["instance"], int(r["slot"]))] = r
    agree_pick = agree_thr = 0
    per = []
    for s in slots:
        pm, pt, A, means = decide(s, sel)
        b = builder[(s["iid"], s["slot"])]
        agree_pick += int(b["sr6_pick"]) == pm
        agree_thr += str(b["SR-thr_choice"]) == str(pt)
        srt = np.sort(means)[::-1]
        per.append({"s": s, "max": pm, "thr": pt, "auc": A, "gap": float(srt[0] - srt[1]),
                    "best_mean": float(srt[0])})
    res["slot_agreement_with_builder"] = {"n_slots": len(slots), "SR-max_pick_equal": agree_pick,
                                          "SR-thr_choice_equal": agree_thr}

    def summarize(per, fam=None):
        P = [p for p in per if p["s"]["planted"] and (fam is None or p["s"]["family"] == fam)]
        k1 = sum(p["max"] == p["s"]["truth"] for p in P)
        k2 = sum(p["thr"] == p["s"]["truth"] for p in P)
        return {"n_planted": len(P), "SR-max": [k1, round(k1 / len(P), 4), wilson(k1, len(P))],
                "SR-thr": [k2, round(k2 / len(P), 4), wilson(k2, len(P))]}
    res["primary"] = {"pooled": summarize(per), "topic": summarize(per, "topic"),
                      "language": summarize(per, "language")}
    # how decisive are SR-max's topic wins? (relative gap between the top two option means)
    tw = [p for p in per if p["s"]["planted"] and p["s"]["family"] == "topic" and p["max"] == p["s"]["truth"]]
    rel = np.array([p["gap"] / p["best_mean"] if p["best_mean"] > 0 else 0 for p in tw])
    res["topic_win_margins"] = {"n": len(tw), "median_rel_gap_top2": round(float(np.median(rel)), 3),
                                "share_rel_gap_lt_0.10": round(float((rel < 0.10).mean()), 3),
                                "median_auc_of_pick": round(float(np.median([p["auc"] for p in tw])), 3)}

    # 4. 42 selections
    sens = {"pooled": {"max": [], "thr": []}, "topic": {"max": [], "thr": []}, "language": {"max": [], "thr": []}}
    for k in range(21):
        for ti in (0, 1):
            _, sl = selection(SEED + k, ti)
            pp = [{"s": s, **dict(zip(("max", "thr"), decide(s, sl)[:2]))} for s in slots if s["planted"]]
            for fam in sens:
                P = [p for p in pp if fam == "pooled" or p["s"]["family"] == fam]
                sens[fam]["max"].append(sum(p["max"] == p["s"]["truth"] for p in P) / len(P))
                sens[fam]["thr"].append(sum(p["thr"] == p["s"]["truth"] for p in P) / len(P))
    res["selection_sensitivity_42"] = {
        fam: {v: dict(zip(("min", "p10", "median", "p90", "max"),
                          [round(float(x), 4) for x in np.percentile(sens[fam][v], [0, 10, 50, 90, 100])]))
              for v in ("max", "thr")} for fam in sens}

    # 5. bank F vs bank R overlap
    ftexts, frows, _ = read_bank("F", cids)
    fby = defaultdict(list)
    for t, r in zip(ftexts, frows):
        fby[r].append(t)
    exact = len(set(texts) & set(ftexts))
    best_j = []
    for t, r in zip(texts, rows):
        g = ngrams(t)
        best_j.append(max(len(g & ngrams(u)) / max(1, len(g | ngrams(u))) for u in fby[r]))
    best_j = np.array(best_j)
    res["bank_F_vs_R_overlap"] = {"exact_duplicate_texts": exact,
                                  "max_char5_jaccard_R_vs_sameconcept_F": {
                                      "median": round(float(np.median(best_j)), 3),
                                      "p90": round(float(np.percentile(best_j, 90)), 3),
                                      "p99": round(float(np.percentile(best_j, 99)), 3),
                                      "max": round(float(best_j.max()), 3),
                                      "n_ge_0.5": int((best_j >= 0.5).sum()), "n": len(best_j)}}

    # 6. names cache: name_probe sanity
    pm = json.load(open(os.path.join(FM_CACHE, "precompute_meta.json")))
    assert pm["cids"] == cids, "names cache concept order differs"
    names = pm["names"]
    assert len(names) == 3 * len(cids)
    nacts = {}
    for L in need:
        X = np.load(os.path.join(FM_CACHE, f"acts_L{L}_names.npy"), mmap_mode="r")
        lat = sorted(need[L])
        nacts[L] = np.asarray(X[:, lat], dtype=np.float32)
        del X
    ep = [json.loads(l) for l in open(os.path.join(DIAG, "step3", "runs", "20261002-041021_inproc",
                                                   "episodes_private.jsonl"))]
    np_sub = {r["instance_id"]: r["submission"] for r in ep if r["solver"] == "recipe_name_probe"}
    agree = n_any_fire = n_true_fires = n_true_top = 0
    planted = [s for s in slots if s["planted"]]
    for s in slots:
        a = nacts[s["layer"]][:, col[s["layer"]][s["latent"]]]
        v = np.array([a[3 * cidx[c] + 1] for c in s["menu"]])          # bare-name form
        pick = int(np.argmax(v)) + 1
        stored = {x["slot"]: x["choice"] for x in np_sub[s["iid"]]["answers"]}[s["slot"]]
        agree += stored == pick
        if s["planted"]:
            n_any_fire += bool((v > 0).any())
            n_true_fires += bool(v[s["truth"] - 1] > 0)
            n_true_top += bool(v[s["truth"] - 1] > 0 and pick == s["truth"])
    res["name_probe_offline"] = {"n_slots": len(slots), "offline_pick_equals_stored_name_probe": agree,
                                 "planted_slots": len(planted), "planted_any_bare_name_fires": n_any_fire,
                                 "planted_true_bare_name_fires": n_true_fires,
                                 "planted_true_name_is_top": n_true_top}
    json.dump(res, open(os.path.join(HERE, "sr_indep.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
