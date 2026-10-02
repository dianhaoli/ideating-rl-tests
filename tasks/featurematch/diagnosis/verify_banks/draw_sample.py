"""Bank-F validity audit, part 1a: draw the blinded sample of 40 topic concepts.

Groups (disjoint, defined on pooled latents of generator v2, key_check.jsonl = PREREG A1.1 output):
  KEPT     topic concepts that anchor >= 1 kept latent (any layer).
  DROPLOW  topic concepts that anchor >= 1 dropped latent with fire_F <= 0.2 and anchor NO kept latent.
fire_F of a latent = share of its anchor's 20 bank-F texts with max-pooled activation > 0, recomputed here from
cache/bank_acts_L*_F.npy (memory-mapped, only pooled columns sliced) and cross-checked with pool_latents.csv.

Draw: rng = random.Random(20261002); kept = rng.sample(sorted(KEPT), 20); low = rng.sample(sorted(DROPLOW), 20);
order = rng.sample(kept + low, 40); blinded ids b01..b40 follow that order.

Writes (in verify_banks/):
  sample_blinded.json   bid, cid, label, the 20 bank-F texts (style, text). No group information.
  unblind_key.json      bid -> group. Its sha256 is stored in sample_blinded.json. Not read until ratings are done.
Prints only group sizes.
"""
import hashlib
import json
import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
SRC_CONCEPTS = "/home/ec2-user/wt/featurematch/tasks/featurematch/cache/concepts.json"
SEED = 20261002
N_PER_GROUP = 20
LOW = 0.2


def load_bank_F(cids):
    allc = {}
    for f in sorted(os.listdir(os.path.join(DIAG, "banks", "F"))):
        d = json.load(open(os.path.join(DIAG, "banks", "F", f)))
        assert d["bank"] == "F"
        allc.update(d["concepts"])
    return {c: [{"style": t["style"], "text": " ".join(str(t["text"]).split())[:400]} for t in allc[c]] for c in cids}


def fire_F_per_latent(rows_kc, cids, meta_rows):
    """rows_kc: list of key_check rows. Returns list of fire_F (anchor bank-F texts with act > 0)."""
    meta_rows = np.asarray(meta_rows)
    cidx = {c: i for i, c in enumerate(cids)}
    out = [None] * len(rows_kc)
    for L in (6, 12, 18):
        X = np.load(os.path.join(DIAG, "cache", f"bank_acts_L{L}_F.npy"), mmap_mode="r")
        idx = [i for i, r in enumerate(rows_kc) if r["layer"] == L]
        cols = sorted({rows_kc[i]["real_latent"] for i in idx})
        sub = np.asarray(X[:, cols], dtype=np.float32)          # [4640, k]
        pos = {c: k for k, c in enumerate(cols)}
        for i in idx:
            r = rows_kc[i]
            a = sub[meta_rows == cidx[r["c_star"]], pos[r["real_latent"]]]
            assert a.shape[0] == 20
            out[i] = float((a > 0).mean())
        del sub, X
    return out


def main():
    meta = json.load(open(os.path.join(DIAG, "cache", "bank_meta.json")))
    cids = meta["cids"]
    rows_kc = [json.loads(l) for l in open(os.path.join(DIAG, "key_check.jsonl"))]
    fF = fire_F_per_latent(rows_kc, cids, meta["banks"]["F"]["rows"])

    # cross-check against the step-2 CSV when present (regenerable, untracked)
    csvp = os.path.join(DIAG, "style_filter_out", "pool_latents.csv")
    xdiff = None
    if os.path.exists(csvp):
        import csv
        ref = {(int(r["layer"]), int(r["real_latent"])): float(r["fire_F_anchor"]) for r in csv.DictReader(open(csvp))}
        xdiff = max(abs(ref[(r["layer"], r["real_latent"])] - f) for r, f in zip(rows_kc, fF))

    kept, droplow = set(), set()
    for r, f in zip(rows_kc, fF):
        if r["family"] != "topic":
            continue
        if r["kept"]:
            kept.add(r["c_star"])
        elif f <= LOW:
            droplow.add(r["c_star"])
    droplow -= kept
    rng = random.Random(SEED)
    sk = rng.sample(sorted(kept), N_PER_GROUP)
    sl = rng.sample(sorted(droplow), N_PER_GROUP)
    order = rng.sample(sk + sl, 2 * N_PER_GROUP)

    concepts = {c["cid"]: c for c in json.load(open(SRC_CONCEPTS))}
    bank = load_bank_F(order)
    # verify texts against bank_meta per-text hashes (same normalisation as style_filter)
    ok = 0
    for c in order:
        ci = cids.index(c)
        rws = [k for k, r in enumerate(meta["banks"]["F"]["rows"]) if r == ci]
        for k, t in zip(rws, bank[c]):
            assert meta["banks"]["F"]["styles"][k] == t["style"]
            ok += hashlib.sha256(t["text"].encode()).hexdigest()[:16] == meta["banks"]["F"]["per_text_sha256"][k]

    key = {f"b{i + 1:02d}": ("KEPT" if c in sk else "DROPLOW") for i, c in enumerate(order)}
    key_blob = json.dumps(key, indent=1, sort_keys=True).encode()
    with open(os.path.join(HERE, "unblind_key.json"), "wb") as f:
        f.write(key_blob)
    sample = {"seed": SEED, "rule": __doc__.strip().splitlines()[2:9],
              "unblind_key_sha256": hashlib.sha256(key_blob).hexdigest(),
              "concepts": [{"bid": f"b{i + 1:02d}", "cid": c, "label": concepts[c]["label"],
                            "path": concepts[c]["path"], "bankF": bank[c]} for i, c in enumerate(order)]}
    with open(os.path.join(HERE, "sample_blinded.json"), "w") as f:
        json.dump(sample, f, indent=1, ensure_ascii=False)
    print(json.dumps({"n_topic_concepts_with_pooled_latent": len({r["c_star"] for r in rows_kc if r["family"] == "topic"}),
                      "n_KEPT_candidates": len(kept), "n_DROPLOW_candidates (disjoint from KEPT)": len(droplow),
                      "fire_F_max_abs_diff_vs_pool_latents_csv": xdiff,
                      "bank_texts_matching_bank_meta_hash": f"{ok}/{20 * len(order)}",
                      "unblind_key_sha256": sample["unblind_key_sha256"]}, indent=1))


if __name__ == "__main__":
    main()
