"""Step-3 skeptic: compare every number in the builder's baselines_public.json with recomputed.json (raw-file
recomputation by regrade.py). Prints and writes compare_builder.json: one row per (baseline, tier, family, field)
that differs, plus the count of fields compared."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
NAME = {"reference (one-shot)": "reference", "reference (best-of-5)": "reference_best_of_5",
        "black-box control": "blackbox", "self_probe": "recipe_self_probe", "template_probe": "recipe_template_probe",
        "nothing": "recipe_nothing", "always_claim": "recipe_always_claim",
        "prior [prior.json, unfiltered v2]": "recipe_prior", "prior_or_none [prior.json, unfiltered v2]":
        "recipe_prior_or_none", "prior [prior_v2f.json, filtered draw]": "recipe_prior@v2f",
        "prior_or_none [prior_v2f.json, filtered draw]": "recipe_prior_or_none@v2f", "random": "recipe_random",
        "name_probe": "recipe_name_probe", "name_probe_thr": "recipe_name_probe_thr", "vocab_match": "recipe_vocab_match"}
FIELDS = {"pass": "pass", "pass_wilson95": "pass_wilson95", "n_planted": "n_planted",
          "planted_correct": "planted_correct", "planted_acc": "planted_acc",
          "planted_acc_wilson95": "planted_acc_wilson95", "planted_acc_cluster_boot95": "planted_acc_cluster95_A2.5_rng",
          "planted_nothing_found_rate": "planted_nf_rate", "null_false_claim_rate": "null_false_claim",
          "n_instances": "n_episodes"}


def main():
    pub = json.load(open(os.path.join(DIAG, "step3", "baselines_public.json")))["metrics"]
    mine = json.load(open(os.path.join(HERE, "recomputed.json")))["metrics"]
    diffs, n = [], 0
    for bname, blk in pub.items():
        m = mine[NAME.get(bname, bname)]
        for tier, fams in blk.items():
            for fam, vals in fams.items():
                key = fam if tier == "all" else (tier if fam == "pooled" else None)
                if key is None or key not in m:
                    continue
                for bf, mf in FIELDS.items():
                    if bf not in vals:
                        continue
                    n += 1
                    a, b = vals[bf], m[key][mf]
                    same = (a == b) if not isinstance(a, float) else abs(a - b) < 6e-5
                    if isinstance(a, list):
                        same = all(abs(x - y) < 6e-5 for x, y in zip(a, b))
                    if not same:
                        diffs.append({"baseline": bname, "tier": tier, "family": fam, "field": bf,
                                      "builder": a, "recomputed": b})
    out = {"n_fields_compared": n, "n_differences": len(diffs), "differences": diffs}
    json.dump(out, open(os.path.join(HERE, "compare_builder.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
