"""Skeptic reproduction: CPU leakage audit of the feasibility study's example / held-out split.

Checks, per (family, group, layer) cell of runs/latentknockout/20261002T0642_sweep/cells (seed 0):
 1. entity overlap between the cell's example entities and its held-out target / sibling items;
 2. re-derivation of the example split from the sweep's seeding rule (does ex_entities match?);
 3. surface-form overlap: held-out target entities sharing a word with an example entity of the same group
    (e.g. 'Serena Williams' in the examples, 'Venus Williams' held out);
 4. template overlap: longest common word n-gram between each held-out template and any example template
    (a held-out 'new style' template that embeds an example template verbatim is not a new style);
 5. near-duplicate langid sentences across example / held-out / other groups.
Only reads JSON; imports lk_data for the template and entity lists (data, not metric code).
usage: python -m tasks.latentknockout.skeptic_reproduce.leakage_check <cells_dir> <validated.json> <out.json>
"""
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict

from tasks.latentknockout import lk_data


def words(s):
    return re.findall(r"[A-Za-z']+|[^\sA-Za-z']", s.replace("{e}", " ENT "))


def lcs_ngram(a, b):
    """length of the longest common contiguous word sequence of a and b (ENT counts as a word)."""
    A, B = words(a), words(b)
    best = 0
    prev = [0] * (len(B) + 1)
    for i in range(1, len(A) + 1):
        cur = [0] * (len(B) + 1)
        for j in range(1, len(B) + 1):
            if A[i - 1] == B[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def main():
    cells_dir, val_path, out = sys.argv[1:4]
    fams = lk_data.families()
    V = json.load(open(val_path))
    rep = dict(cells=0, entity_overlap=[], split_mismatch=[], surface_overlap=defaultdict(list), template_overlap={},
               langid_near_dups=[])
    # ---- template overlap (family level)
    for fam, F in fams.items():
        T = F["templates"]
        ex = [T[i][1] for i in lk_data.EX_T]
        rows = []
        for i in lk_data.HO_T + lk_data.HO_S:
            l = max(lcs_ngram(T[i][1], e) for e in ex)
            rows.append(dict(idx=i, style=T[i][0], lcs_words_with_examples=l, text=T[i][1]))
        rep["template_overlap"][fam] = rows
    # ---- per cell
    for f in sorted(glob.glob(os.path.join(cells_dir, "*__s0.json"))):
        c = json.load(open(f))
        if "items" not in c or c["family"] == "country_capital":
            continue
        rep["cells"] += 1
        fam, g = c["family"], c["group"]
        ex = set(c["ex_entities"])
        tgt = {it[2] for it in c["items"] if it[0] in ("tT", "tS")}
        sib = {(it[1], it[2]) for it in c["items"] if it[0] in ("sT", "sS")}
        if ex & tgt:
            rep["entity_overlap"].append([f, sorted(ex & tgt)])
        # re-derive the split with the sweep's rule (rng0 = Random(1000*seed+7), groups in dict order)
        rng0 = random.Random(7)
        ents = {}
        for h in fams[fam]["groups"]:
            es = sorted({r[0] for r in V[fam][h]})
            rng0.shuffle(es)
            ents[h] = es
        n_ex = {h: (4 if len(es) >= 12 else 3) for h, es in ents.items()}
        ex_re = set(ents[g][:n_ex[g]])
        if ex_re != ex:
            rep["split_mismatch"].append(f)
        sib_ex = {(h, e) for h in ents if h != g for e in ents[h][:n_ex[h]]}
        if sib & sib_ex:
            rep["entity_overlap"].append([f, "sibling held-out entity is an example control", sorted(sib & sib_ex)[:5]])
        if fam != "langid":
            exw = {w for e in ex for w in e.replace("-", " ").split() if len(w) > 2 and w not in ("the", "The")}
            hits = sorted(e for e in tgt if exw & set(e.replace("-", " ").split()))
            if hits:
                rep["surface_overlap"][fam].append(dict(group=g, layer=c["layer"], ex=sorted(ex), held_out_sharing_a_word=hits,
                                                         n_held_out=len(tgt)))
    # ---- langid near duplicates (first 6 words identical)
    ls = fams["langid"]["groups"]
    seen = {}
    for g, sents in ls.items():
        for s in sents:
            k = " ".join(s.lower().split()[:6])
            if k in seen:
                rep["langid_near_dups"].append([seen[k], (g, s)])
            seen[k] = (g, s)
    rep["surface_overlap"] = dict(rep["surface_overlap"])
    json.dump(rep, open(out, "w"), indent=1)
    print("cells", rep["cells"], "entity_overlap", len(rep["entity_overlap"]), "split_mismatch", len(rep["split_mismatch"]),
          "langid near dups", len(rep["langid_near_dups"]))
    for fam, rows in rep["template_overlap"].items():
        print(fam, [(r["idx"], r["style"], r["lcs_words_with_examples"]) for r in rows])
    for fam, rows in rep["surface_overlap"].items():
        print(fam, "cells with a held-out entity sharing a word with an example entity:",
              len({(r['group']) for r in rows}), [(r["group"], r["held_out_sharing_a_word"]) for r in rows if r["layer"] == 18][:8])


if __name__ == "__main__":
    main()
