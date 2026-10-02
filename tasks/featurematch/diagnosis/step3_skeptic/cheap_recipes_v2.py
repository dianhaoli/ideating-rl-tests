"""Step-3 skeptic: the same cheap recipes on the UNFILTERED v2 pool (~/wt/featurematch/tasks/featurematch/instances,
read-only), to see whether the style filter ENABLED them. Same texts (gpu_probe2.py: byte-identical strings, columns of
every v2-pooled latent), same decision rules; the "-cal" variants reuse the thresholds calibrated on the FILTERED
calibration pool (read from cheap_recipes.json), so nothing is tuned on the v2 pool. Not a pre-registered quantity;
descriptive comparison only. Writes cheap_recipes_v2.json (aggregates only)."""
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
TASK = os.path.dirname(DIAG)
sys.path.insert(0, os.path.dirname(os.path.dirname(TASK)))
from tasks.featurematch import grader  # noqa: E402
from tasks.featurematch.diagnosis.step3_skeptic import cheap_recipes as CR  # noqa: E402
from tasks.featurematch.diagnosis.step3_skeptic import regrade as RG  # noqa: E402

V2 = os.path.expanduser("~/wt/featurematch/tasks/featurematch/instances")


class ActsV2(CR.Acts):
    def __init__(self):
        super().__init__()
        m = json.load(open(os.path.join(CR.CACHE, "v2_meta.json")))
        self.kept = {int(L): v for L, v in m["pool"].items()}
        self.col = {L: {j: i for i, j in enumerate(v)} for L, v in self.kept.items()}
        self.T = {L: np.load(os.path.join(CR.CACHE, f"v2_tmpl_acts_L{L}.npy")) for L in self.kept}
        self.G = {L: np.load(os.path.join(CR.CACHE, f"v2_gen_acts_L{L}.npy")) for L in self.kept}
        for L in self.kept:
            X = np.load(os.path.join(CR.FM_CACHE, f"acts_L{L}_names.npy"), mmap_mode="r")
            self.N[L] = np.asarray(X[:, self.kept[L]], dtype=np.float32)
            del X
        self.consistency = m["consistency_with_gpu_probe"]


def load_v2():
    eps, T = {}, {}
    h = hashlib.sha256()
    for d in sorted(glob.glob(os.path.join(V2, "*"))):
        p = os.path.join(d, "instance.json")
        if not os.path.isfile(p):
            continue
        h.update(os.path.basename(d).encode())
        h.update(open(p, "rb").read())
        inst = json.load(open(p))
        sl = []
        for i, (e, t) in enumerate(zip(inst["extra"]["slots"], inst["answer"]["slots"])):
            sl.append({"iid": inst["instance_id"], "tier": inst["tier"], "slot": i, "layer": int(e["layer"]),
                       "latent": int(e["real_latent"]), "menu": e["menu"], "planted": t["planted"],
                       "truth": t["choice"], "anchor": e["anchor"],
                       "family": "language" if e["anchor"].startswith("lang:") else "topic"})
        eps[inst["instance_id"]] = sl
        T[inst["instance_id"]] = {"dir": d, "tier": inst["tier"], "truth": inst["answer"]["slots"],
                                  "family": [s["family"] for s in sl]}
    return eps, T, h.hexdigest()


def main():
    acts = ActsV2()
    eps, T, digest = load_v2()
    dry = json.load(open(os.path.join(DIAG, "sr_recipe_out_v2_unfiltered_DRYRUN", "summary.json")))
    cal = json.load(open(os.path.join(HERE, "cheap_recipes.json")))["recipes"]
    out = {"pool": {"dir": V2, "n_instances": len(eps), "pool_sha256": digest,
                    "same_pool_as_SR_dry_run": digest == dry.get("pool_sha256")},
           "consistency_with_gpu_probe": acts.consistency, "recipes": {}}
    for recipe in CR.RECIPES:
        fe = {iid: CR.episode_features(acts, recipe, sl) for iid, sl in eps.items()}
        rule = cal[recipe]["calibrated"]["rule"]
        rule = (rule["stat"], rule["tau"])
        res = {}
        for name, r in (("max_pick", None), ("calibrated_on_v2f_calpool", rule)):
            rows = []
            for iid, F in fe.items():
                sub = {"answers": [{"slot": s["slot"], "choice": CR.choose(f, r)} for s, f in zip(eps[iid], F)]}
                rows.append({"instance_id": iid, "slots": grader.grade(T[iid]["dir"], sub)["details"]["slots"]})
            res[name] = {"pooled": RG.metrics(rows, T), "topic": RG.metrics(rows, T, "topic"),
                         "language": RG.metrics(rows, T, "language")}
        out["recipes"][recipe] = res
        m = res["max_pick"]
        print(f"{recipe:14s} v2 max-pick pa {m['pooled']['planted_correct']}/{m['pooled']['n_planted']}="
              f"{m['pooled']['planted_acc']} W{m['pooled']['planted_acc_wilson95']} topic {m['topic']['planted_acc']} "
              f"lang {m['language']['planted_acc']} | cal pa {res['calibrated_on_v2f_calpool']['pooled']['planted_acc']}"
              f" pass {res['calibrated_on_v2f_calpool']['pooled']['pass_rate']}", flush=True)
    json.dump(out, open(os.path.join(HERE, "cheap_recipes_v2.json"), "w"), indent=1)
    print(json.dumps(out["pool"]), json.dumps(out["consistency_with_gpu_probe"]))


if __name__ == "__main__":
    main()
