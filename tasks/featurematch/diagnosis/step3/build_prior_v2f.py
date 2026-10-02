"""Step 3: a "most popular concept" prior for the FILTERED pool, from a disjoint draw of the filtered generator (CPU).

WHY: recipe_baseline.py's `prior` / `prior_or_none` read tasks/featurematch/prior.json, which counts planted answers in
in-memory instances of the UNFILTERED v2 generator (seeds 900000+, T1+T2). The filtered pool (instances_v2f) draws its
anchors from a much smaller concept universe (74 anchor concepts vs 159), so an unfiltered prior understates what a
prior recipe can do there. This script builds the fair counterpart: the same restricted tables as write_v2f.py
(generator v2's Tables with the latent pool restricted to the A1.1-kept latents of key_check.jsonl), then
generate.make_instance on NEW seeds, disjoint from the pool's 8000+ / 108000+ / 208000+:
  T1 9000+k, T2 109000+k, T3 209000+k, k < n_per_tier (default 300).
Output (same format as prior.json: label -> share of planted answers, most common first), written ONLY under
diagnosis/step3/; tasks/featurematch/prior.json is never touched. Nothing is written to any instances directory.

Run (from the worktree root, after `source ~/ideating-rl-tests/common/env.sh`):
  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=2G -- /usr/bin/time -v \
      $PY -m tasks.featurematch.diagnosis.step3.build_prior_v2f
"""
import argparse
import glob
import json
import os
from collections import Counter

from tasks.featurematch.diagnosis import style_filter as SF
from tasks.featurematch.diagnosis import write_v2f as W

HERE = os.path.dirname(os.path.abspath(__file__))
SEED_BASE = {"T1": 9000, "T2": 109000, "T3": 209000}
EVAL_SEED_BASE = {"T1": 8000, "T2": 108000, "T3": 208000}


def build(n_per_tier=300, key_check=os.path.join(SF.HERE, "key_check.jsonl"), pool_dir=W.OUT_INST):
    from tasks.featurematch.generate import make_instance
    T = SF.load_tables(SF.default_src_cache())
    kept, n_pool = W.load_kept(key_check)
    W.restrict_tables(T, kept)
    eval_ids = {os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(pool_dir, "*", "instance.json"))}
    eval_seeds = set()
    for p in glob.glob(os.path.join(pool_dir, "*", "instance.json")):
        eval_seeds.add(int(json.load(open(p))["seed"]))
    cnt, errors, n_inst, n_planted, n_slots = Counter(), Counter(), 0, 0, 0
    for tier, base in SEED_BASE.items():
        for k in range(n_per_tier):
            seed = base + k
            assert seed not in eval_seeds, f"seed {seed} overlaps the evaluation pool"
            try:
                iid, inst, public = make_instance(T, seed, tier)
            except Exception as e:          # recorded, never patched
                errors[f"{tier}:{type(e).__name__}"] += 1
                continue
            assert iid not in eval_ids, f"instance id {iid} collides with the evaluation pool"
            n_inst += 1
            for s, pub in zip(inst["answer"]["slots"], public["slots"]):
                n_slots += 1
                if s["planted"]:
                    n_planted += 1
                    cnt[pub["options"][s["choice"] - 1]] += 1
    tot = sum(cnt.values())
    prior = {lab: round(n / tot, 5) for lab, n in cnt.most_common()}
    meta = {"what": "planted-answer label frequencies of the FILTERED generator (write_v2f restricted tables), "
                    "disjoint seeds; same format as tasks/featurematch/prior.json",
            "seeds": {t: [b, b + n_per_tier - 1] for t, b in SEED_BASE.items()},
            "eval_pool_seeds": {t: [b, b + 59] for t, b in EVAL_SEED_BASE.items()},
            "n_instances": n_inst, "n_slots": n_slots, "n_planted_slots": n_planted, "n_labels": len(prior),
            "draw_errors": dict(errors), "key_check_sha256": SF.sha256_file(key_check),
            "top10": cnt.most_common(10)}
    return prior, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-tier", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(HERE, "prior_v2f.json"))
    a = ap.parse_args(argv)
    prior, meta = build(a.n_per_tier)
    if os.path.abspath(a.out) == os.path.abspath(os.path.join(SF.TASK, "prior.json")):
        raise SystemExit("refusing to overwrite tasks/featurematch/prior.json")
    json.dump(prior, open(a.out, "w"), indent=0)
    json.dump(meta, open(os.path.splitext(a.out)[0] + "_meta.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "top10"}, indent=1))
    print("top10:", meta["top10"])


if __name__ == "__main__":
    main()
