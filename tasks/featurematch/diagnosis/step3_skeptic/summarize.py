"""Step-3 skeptic: collect the checks into skeptic_summary.json (aggregates only) and count, per new recipe, how many
episodes of instances_v2f would exceed their tier's forward cap (550 on T3) or the 40-call generate cap."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
TASK = os.path.dirname(DIAG)
sys.path.insert(0, os.path.dirname(os.path.dirname(TASK)))
from tasks.featurematch.diagnosis.step3_skeptic import cheap_recipes as CR  # noqa: E402


def budgets():
    acts = CR.Acts()
    eps = CR.eval_pool()
    caps = {iid: json.load(open(os.path.join(CR.INST, iid, "instance.json")))["caps"] for iid in eps}
    out = {}
    for r in CR.RECIPES:
        over_f, over_g, mx = 0, 0, 0
        for iid, sl in eps.items():
            labels = {c for s in sl for c in s["menu"]}
            if r == "gen1_cap":
                f, g = 20 * len(sl), min(len(labels), CR.GEN_CAP)
            else:
                f = sum(len(acts.vals(r, s["layer"], s["latent"], c)) for s in sl for c in s["menu"])
                g = 2 * len(labels) if r == "gen_all" else 0
                if r == "name3":
                    f *= 3                      # 3 name forms per option, max taken offline
            mx = max(mx, f)
            over_f += f > caps[iid]["forward"]
            over_g += g > caps[iid]["generate"]
        out[r] = {"max_forward_per_episode": mx, "episodes_over_forward_cap": over_f,
                  "episodes_over_generate_cap": over_g}
    return out


def row(m):
    p = m["pooled"]
    return {"planted_acc": f"{p['planted_correct']}/{p['n_planted']}={p['planted_acc']:.3f}",
            "planted_wilson95": p["planted_acc_wilson95"], "planted_cluster95": p["planted_acc_cluster95_A2.5_rng"],
            "topic_planted": f"{m['topic']['planted_correct']}/{m['topic']['n_planted']}={m['topic']['planted_acc']:.3f}",
            "topic_wilson95": m["topic"]["planted_acc_wilson95"],
            "language_planted": f"{m['language']['planted_correct']}/{m['language']['n_planted']}="
                                f"{m['language']['planted_acc']:.3f}",
            "language_wilson95": m["language"]["planted_acc_wilson95"],
            "pass": f"{p['pass']}/{p['n_episodes']}={p['pass_rate']:.3f}", "pass_wilson95": p["pass_wilson95"],
            "topic_family_pass": f"{m['topic']['pass']}/{m['topic']['n_episodes']}",
            "language_family_pass": f"{m['language']['pass']}/{m['language']['n_episodes']}",
            "planted_nf": p["planted_nf_rate"], "null_false_claim": p["null_false_claim"]}


VERDICT = ("The builder's stop-rule conclusion STANDS: fires, not borderline, not text-selection-sensitive, also fires on "
           "topic slots alone, and survives (a) a full re-grade of all 3,603 stored episodes, (b) an independent SR "
           "re-implementation (725/725 slot decisions equal), (c) a live re-run of SR through the tool's own "
           "token_acts path (SR-max 386/441 vs 385 offline; SR-thr 322 vs 322). It does not hinge on bank R or on SR: "
           "an in-budget fixed recipe using only the task's own generate tool (gen1_cap) gets 252/441 = 0.571 "
           "[0.525, 0.617] planted accuracy. Verdict label 'Not interpretability' is supported.")
DISCREPANCIES = [
    "No numeric discrepancy: 1,260/1,260 published fields of baselines_public.json reproduce from the raw files; "
    "3,603/3,603 episodes re-grade identically; 30/30 seeded CLI re-grades match; pool and manifest hashes match.",
    "Interpretation: BASELINES.md says template_probe's gate failure 'comes entirely from language slots' and that "
    "it gets 0.061 on topic slots. That is true for its AUROC-vs-background rule, but its own 4 fixed encyclopedia "
    "templates with a plain max-pick get topic 125/328 = 0.381 [0.330, 0.435] (pooled 0.540). The topic weakness "
    "is the threshold, not the texts.",
    "Gate statement 'every other zero-effort recipe passes' holds only for the recipes that were run. New fixed "
    "recipes fail the 0.15 gate by wide margins: encyc4_max 0.540, gen_all 0.673, gen1_cap 0.571 (in budget), "
    "tmpl7_posthoc 0.551, style3 0.365 (topic 0.146).",
    "The cheap solvers were mostly NOT created by the filter: on the unfiltered v2 pool the same recipes already "
    "get gen_all 0.585, gen1_cap 0.441, encyc4_max 0.378. The filter raised them by 0.10-0.16, partly by tripling "
    "the share of language slots. So the v2 gate (non-reference recipes <= 0.15) was probably never met on v2 "
    "either; earlier gate passes relied on thresholded or shortlisted recipes (template_probe, self_probe).",
    "name_probe / name_probe_thr pass the gate by construction, not by test: generator v2 drops every latent that "
    "fires on any of its concept's 3 name forms, so the true option's name fires on 0/441 planted slots, and "
    "name_probe equals always_claim (tie -> option 1) on 724/725 slots.",
    "Prediction labels are generous: P7 is called a hit, but both point predictions (0.97 planted, 0.88 pass) "
    "lie outside the outcome CIs (1.000 [0.991, 1.000]; 0.989 [0.960, 0.997]); P9 (0.30) and P10 (0.20) "
    "missed their levels (0.204 [0.169, 0.244]; 0.288 [0.248, 0.332]); only their gate-failure parts hit.",
    "Documentation only: PREREG Amendment 2 and 3 headers say ~04:20 and ~04:45 UTC, but git commits are 03:53 "
    "and 04:04 UTC. Both still precede the SR run (04:10:56) and the in-process run (started 04:10:21), so the "
    "pre-registration order holds.",
]


def main():
    rec = json.load(open(os.path.join(HERE, "recomputed.json")))
    ind = json.load(open(os.path.join(HERE, "sr_indep.json")))
    cmp_ = json.load(open(os.path.join(HERE, "compare_builder.json")))
    cr = json.load(open(os.path.join(HERE, "cheap_recipes.json")))
    v2 = json.load(open(os.path.join(HERE, "cheap_recipes_v2.json")))
    S = {"what": "Adversarial check of the step-3 baselines (BASELINES.md, commit 80fb525d) on instances_v2f",
         "verdict": VERDICT, "discrepancies": DISCREPANCIES,
         "note": "VERDICT and DISCREPANCIES were written after the checks below had run.",
         "pool": rec["pool"],
         "regrade": {"all_episodes": rec["full_regrade"], "seeded_sample_cli": rec["sample"],
                     "published_fields_compared": cmp_["n_fields_compared"],
                     "published_fields_different": cmp_["n_differences"]},
         "stop_rule_recomputed_from_raw": rec["stop_rule_recomputed"],
         "sr_independent_reimplementation": {k: ind[k] for k in ("cache_rows_match_bank_files", "primary_styles",
                                                                  "slot_agreement_with_builder", "primary",
                                                                  "selection_sensitivity_42", "bank_F_vs_R_overlap",
                                                                  "name_probe_offline")},
         "sr_live_vs_offline": cr["sr_live_vs_offline"],
         "template_probe_offline_replica": cr["template_probe_offline_vs_inprocess"],
         "template_live_vs_offline": cr["template_live_vs_offline"],
         "calibration_pool": cr["calibration_pool"],
         "new_recipes_v2f": {r: {"max_pick": row(v["max_pick"]),
                                 "calibrated": {"rule": v["calibrated"]["rule"], **row(v["calibrated"])}}
                             for r, v in cr["recipes"].items()},
         "new_recipes_unfiltered_v2": {"pool": v2["pool"],
                                       **{r: {"max_pick": row(v["max_pick"]),
                                              "calibrated_on_v2f_calpool": row(v["calibrated_on_v2f_calpool"])}
                                          for r, v in v2["recipes"].items()}},
         "budgets_v2f": budgets()}
    json.dump(S, open(os.path.join(HERE, "skeptic_summary.json"), "w"), indent=1)
    print(json.dumps(S["budgets_v2f"], indent=0))


if __name__ == "__main__":
    main()
