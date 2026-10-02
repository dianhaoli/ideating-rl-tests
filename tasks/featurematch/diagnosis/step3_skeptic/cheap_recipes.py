"""Step-3 skeptic, part 3 (CPU): score the new cheap recipes on the filtered pool, offline from gpu_probe.py's cache.

Recipes and texts: cheap_texts.py. Decision rules:
  max-pick  claim the option with the highest mean max activation over its texts (ties -> lowest option number; an
            option with no text cannot be claimed). Never says "nothing found" (except when nothing can be claimed).
  -cal      the same pick, claimed only if the best mean is > 0 and a separation statistic clears a threshold; the
            statistic (relative gap between the two best option means, or AUROC of the best option's texts against
            the other options' texts) and the threshold are chosen on a DISJOINT CALIBRATION POOL to maximise pass
            rate (ties: higher mean grader score, then the smaller threshold). The calibration pool is generated in
            memory by the filtered generator (write_v2f restricted tables, generator v2 unchanged) on seeds
            T1 9500+k, T2 109500+k, T3 209500+k, k < 100: disjoint from the evaluation pool (8000+ / 108000+ /
            208000+) and from build_prior_v2f's draw (9000-9299 ...). No eval-pool number is used to choose anything.
Evaluation: submissions are graded by tasks/featurematch/grader.grade on instances_v2f; metrics (pooled / topic /
language, Wilson 95%, instance-clustered bootstrap 95%) use regrade.py's own statistics.
Also here: template_probe re-implemented offline (AUROC vs the 24 background texts >= 0.78) and compared slot by slot
with the builder's in-process template_probe answers (a check that the offline activations match the live tool), the
SR live-vs-offline comparison (gpu_probe sr_live) and the 60-slot template live check.
Writes cheap_recipes.json (aggregates only).
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
TASK = os.path.dirname(DIAG)
ROOT = os.path.dirname(os.path.dirname(TASK))
sys.path.insert(0, ROOT)
CACHE = os.path.join(HERE, "cache")
FM_CACHE = os.path.expanduser("~/wt/featurematch/tasks/featurematch/cache")
INST = os.path.join(TASK, "instances_v2f")
NF = "nothing found"
CAL_SEEDS = {"T1": 9500, "T2": 109500, "T3": 209500}
N_CAL = 100
GEN_CAP = 40

from tasks.featurematch import grader  # noqa: E402
from tasks.featurematch.diagnosis.step3_skeptic import regrade as RG  # noqa: E402
from tasks.featurematch.diagnosis.step3_skeptic.sr_indep import auc, load_slots  # noqa: E402


# ------------------------------------------------------------------------------------------------ activations
class Acts:
    def __init__(self):
        m = json.load(open(os.path.join(CACHE, "tmpl_meta.json")))
        gp = os.path.join(CACHE, "gen_meta.json")
        g = json.load(open(gp)) if os.path.exists(gp) else {"jobs": [], "completions": []}
        self.kept = {int(L): v for L, v in m["kept"].items()}
        self.col = {L: {j: i for i, j in enumerate(v)} for L, v in self.kept.items()}
        self.T = {L: np.load(os.path.join(CACHE, f"tmpl_acts_L{L}.npy")) for L in self.kept}
        self.G = {L: np.load(os.path.join(CACHE, f"gen_acts_L{L}.npy")) for L in self.kept} if g["jobs"] else {}
        self.rows = {}
        for i, (kind, c, j) in enumerate(m["tags"]):
            self.rows.setdefault((kind, c), []).append(i)
        self.bg = self.rows[("background", None)]
        self.grows = {}
        for i, (c, j, _) in enumerate(g["jobs"]):
            self.grows.setdefault(c, []).append(i)
        self.n_empty_gen = sum(not x for x in g["completions"])
        cids = m["cids"]
        self.cidx = {c: i for i, c in enumerate(cids)}
        pm = json.load(open(os.path.join(FM_CACHE, "precompute_meta.json")))
        assert pm["cids"] == cids
        self.N = {}
        for L in self.kept:
            X = np.load(os.path.join(FM_CACHE, f"acts_L{L}_names.npy"), mmap_mode="r")
            self.N[L] = np.asarray(X[:, self.kept[L]], dtype=np.float32)
            del X

    def vals(self, recipe, layer, latent, c, gen_ok=True):
        j = self.col[layer][latent]
        if recipe in ("generic_label", "generic_bare", "style3"):
            return self.T[layer][self.rows[(recipe, c)], j]
        if recipe in ("encyc4_max", "template_probe"):
            return self.T[layer][self.rows[("encyc4", c)], j]
        if recipe == "tmpl7_posthoc":          # encyc4 + style3 (languages: the 3-sentence bank once)
            r = self.rows[("encyc4", c)] + ([] if c.startswith("lang:") else self.rows[("style3", c)])
            return self.T[layer][r, j]
        if recipe == "gen_all":
            return self.G[layer][self.grows[c], j]
        if recipe == "gen1_cap":
            return self.G[layer][self.grows[c][:1], j] if gen_ok else np.zeros(0, np.float32)
        if recipe == "name3":
            i = self.cidx[c]
            return np.array([self.N[layer][3 * i:3 * i + 3, j].max()], dtype=np.float32)
        raise ValueError(recipe)

    def background(self, layer, latent):
        return self.T[layer][self.bg, self.col[layer][latent]]


RECIPES = ["generic_label", "generic_bare", "style3", "encyc4_max", "gen_all", "gen1_cap", "name3", "tmpl7_posthoc"]
# tmpl7_posthoc was added AFTER seeing the other template recipes' results (an adversarial follow-up, not fixed in
# advance like the others in cheap_texts.py).


# ------------------------------------------------------------------------------------------------ decisions
def features(vals):
    """vals: list over options of 1-D arrays -> (pick 1-based or None, m1, gap, auroc)."""
    means = np.array([float(v.mean()) if len(v) else -np.inf for v in vals])
    if not np.isfinite(means).any():
        return None, 0.0, -1.0, 0.5
    best = int(np.argmax(means))
    m1 = means[best]
    others = np.delete(means, best)
    others = others[np.isfinite(others)]
    m2 = float(others.max()) if len(others) else 0.0
    gap = (m1 - m2) / m1 if m1 > 0 else -1.0
    neg = [v for k, v in enumerate(vals) if k != best and len(v)]
    A = auc(vals[best], np.concatenate(neg)) if neg else 0.5
    return best + 1, float(m1), float(gap), float(A)


def slot_option_vals(acts, recipe, s, gen_budget=None):
    out = []
    for c in s["menu"]:
        ok = True
        if recipe == "gen1_cap":
            ok = gen_budget(c)
        out.append(acts.vals(recipe, s["layer"], s["latent"], c, gen_ok=ok))
    return out


def episode_features(acts, recipe, eslots):
    """Per-episode list of slot features; gen1_cap spends 1 generate call per distinct label in slot/option order."""
    done, used = set(), [0]

    def budget(c):
        if c in done:
            return True
        if used[0] < GEN_CAP:
            used[0] += 1
            done.add(c)
            return True
        return False
    return [features(slot_option_vals(acts, recipe, s, budget)) for s in eslots]


def choose(f, rule):
    pick, m1, gap, A = f
    if pick is None:
        return NF if rule else 1
    if rule is None:
        return pick
    stat, tau = rule
    if m1 <= 0:
        return NF
    return pick if (gap if stat == "gap" else A) >= tau else NF


GRID = [("gap", round(t, 2)) for t in np.arange(0, 1.0, 0.05)] + [("auc", round(t, 2)) for t in np.arange(0.5, 1.001, 0.025)]


# ------------------------------------------------------------------------------------------------ pools
def eval_pool():
    eps = {}
    for s in load_slots():
        eps.setdefault(s["iid"], []).append(s)
    return eps


def calibration_pool():
    from tasks.featurematch.diagnosis import style_filter as SF
    from tasks.featurematch.diagnosis import write_v2f as W
    from tasks.featurematch.generate import make_instance
    T = SF.load_tables(SF.default_src_cache())
    kept, _ = W.load_kept(os.path.join(DIAG, "key_check.jsonl"))
    W.restrict_tables(T, kept)
    eval_seeds = {int(json.load(open(p))["seed"]) for p in glob.glob(os.path.join(INST, "*", "instance.json"))}
    eps, errors = {}, 0
    for tier, base in CAL_SEEDS.items():
        for k in range(N_CAL):
            seed = base + k
            assert seed not in eval_seeds and not (9000 <= seed < 9300 or 109000 <= seed < 109300
                                                   or 209000 <= seed < 209300)
            try:
                iid, inst, pub = make_instance(T, seed, tier)
            except Exception:
                errors += 1
                continue
            sl = []
            for i, (e, t) in enumerate(zip(inst["extra"]["slots"], inst["answer"]["slots"])):
                sl.append({"iid": iid, "tier": tier, "slot": i, "layer": int(e["layer"]),
                           "latent": int(e["real_latent"]), "menu": e["menu"], "planted": t["planted"],
                           "truth": t["choice"], "anchor": e["anchor"],
                           "family": "language" if e["anchor"].startswith("lang:") else "topic"})
            eps[iid] = sl
    return eps, errors


def grade_mine(eslots, choices):
    truth = [{"planted": s["planted"], "choice": s["truth"]} for s in eslots]
    return RG.my_grade(truth, {"answers": [{"slot": s["slot"], "choice": c} for s, c in zip(eslots, choices)]})


def calibrate(feats_cal, cal):
    best = None
    for rule in GRID:
        npass, score = 0, 0.0
        for iid, F in feats_cal.items():
            g = grade_mine(cal[iid], [choose(f, rule) for f in F])
            npass += all(x["correct"] for x in g)
            p = [x["correct"] for x in g if x["planted"]]
            u = [x["correct"] for x in g if not x["planted"]]
            score += (np.mean(p) if p else 1.0) * (np.mean(u) if u else 1.0)
        key = (npass, round(score, 6), -rule[1] if rule[0] == "gap" else -rule[1])
        if best is None or key > best[0]:
            best = (key, rule, npass / len(feats_cal))
    return best[1], round(best[2], 4)


def evaluate(feats, eps, Ttruth, rule):
    rows = []
    for iid, F in feats.items():
        sub = {"answers": [{"slot": s["slot"], "choice": choose(f, rule)} for s, f in zip(eps[iid], F)]}
        g = grader.grade(Ttruth[iid]["dir"], sub)
        rows.append({"instance_id": iid, "slots": g["details"]["slots"]})
    return {"pooled": RG.metrics(rows, Ttruth), "topic": RG.metrics(rows, Ttruth, "topic"),
            "language": RG.metrics(rows, Ttruth, "language"),
            **{t: RG.metrics([r for r in rows if Ttruth[r["instance_id"]]["tier"] == t], Ttruth)
               for t in ("T1", "T2", "T3")}}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", default=",".join(RECIPES))
    ap.add_argument("--skip-live", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "cheap_recipes.json"))
    a = ap.parse_args()
    acts = Acts()
    eps = eval_pool()
    Ttruth = RG.load_truth()
    cal, cal_err = calibration_pool()
    out = {"calibration_pool": {"n_instances": len(cal), "n_slots": sum(len(v) for v in cal.values()),
                                "draw_errors": cal_err, "seeds": {t: [b, b + N_CAL - 1] for t, b in CAL_SEEDS.items()}},
           "n_empty_generations": acts.n_empty_gen, "recipes": {}}
    for recipe in a.recipes.split(","):
        fe = {iid: episode_features(acts, recipe, sl) for iid, sl in eps.items()}
        fc = {iid: episode_features(acts, recipe, sl) for iid, sl in cal.items()}
        rule, cal_pass = calibrate(fc, cal)
        # calibration-pool planted accuracy of max-pick (sanity: not used for any choice)
        res = {"max_pick": evaluate(fe, eps, Ttruth, None),
               "calibrated": {"rule": {"stat": rule[0], "tau": rule[1]}, "calibration_pass_rate": cal_pass,
                              **evaluate(fe, eps, Ttruth, rule)}}
        # forward / generate cost per episode (descriptive)
        nt = []
        for iid, sl in eps.items():
            n = 0
            for s in sl:
                n += sum(len(acts.vals(recipe if recipe != "gen1_cap" else "gen_all", s["layer"], s["latent"], c))
                         for c in s["menu"]) if recipe != "gen1_cap" else 20
            nt.append(n)
        res["forward_units_per_episode_mean"] = round(float(np.mean(nt)), 1)
        if recipe == "gen_all":
            u = [len({c for s in sl for c in s["menu"]}) for sl in eps.values()]
            res["generate_calls_needed_per_episode"] = {"median": float(np.median([2 * x for x in u])),
                                                        "episodes_over_cap_40": int(sum(2 * x > GEN_CAP for x in u))}
        out["recipes"][recipe] = res
        m, c = res["max_pick"], res["calibrated"]
        print(f"{recipe:14s} max-pick pa {m['pooled']['planted_correct']}/{m['pooled']['n_planted']}="
              f"{m['pooled']['planted_acc']} W{m['pooled']['planted_acc_wilson95']} | topic {m['topic']['planted_acc']}"
              f" lang {m['language']['planted_acc']} | pass {m['pooled']['pass_rate']}  ||  cal {rule} "
              f"pa {c['pooled']['planted_acc']} W{c['pooled']['planted_acc_wilson95']} topic {c['topic']['planted_acc']}"
              f" pass {c['pooled']['pass']}/{c['pooled']['n_episodes']}={c['pooled']['pass_rate']} "
              f"{c['pooled']['pass_wilson95']} FC {c['pooled']['null_false_claim']}", flush=True)

    # ---------------- template_probe replica vs builder's in-process answers
    ep = [json.loads(l) for l in open(os.path.join(DIAG, "step3", "runs", "20261002-041021_inproc",
                                                   "episodes_private.jsonl"))]
    tp = {r["instance_id"]: {a["slot"]: a["choice"] for a in r["submission"]["answers"]}
          for r in ep if r["solver"] == "recipe_template_probe" and r["seed"] == 7}
    agree = n = 0
    dis = []
    for iid, sl in eps.items():
        for s in sl:
            vals = [acts.vals("template_probe", s["layer"], s["latent"], c) for c in s["menu"]]
            bg = acts.background(s["layer"], s["latent"])
            aucs = [auc(v, bg) for v in vals]
            key = [a + 1e-6 * float(np.mean(v)) for a, v in zip(aucs, vals)]
            best = int(np.argsort(key)[::-1][0])
            ch = best + 1 if aucs[best] >= 0.78 and vals[best].max() > 0 else NF
            n += 1
            agree += ch == tp[iid][s["slot"]]
            if ch != tp[iid][s["slot"]]:
                dis.append(round(float(aucs[best]), 3))
    out["template_probe_offline_vs_inprocess"] = {"n_slots": n, "same_choice": agree,
                                                  "best_auc_on_disagreements": sorted(dis)[:30]}
    print("template_probe replica agreement", agree, "/", n)

    if a.skip_live:
        json.dump(out, open(a.out, "w"), indent=1)
        return
    # ---------------- SR live vs offline
    z = np.load(os.path.join(CACHE, "sr_live.npz"))
    V, ids = z["vals"], [str(x) for x in z["ids"]]
    slots = load_slots()
    assert ids == [f"{s['iid']}:{s['slot']}" for s in slots]
    from tasks.featurematch.diagnosis.step3_skeptic.sr_indep import read_bank  # noqa: F401
    meta = json.load(open(os.path.join(DIAG, "cache", "bank_meta.json")))
    cidx = {c: i for i, c in enumerate(meta["cids"])}
    import random
    styles = meta["banks"]["R"]["styles"]
    rows = meta["banks"]["R"]["rows"]
    chosen = random.Random(20261002).sample(sorted(set(styles)), 6)
    first = {}
    for i, (r, st) in enumerate(zip(rows, styles)):
        first.setdefault((r, st), i)
    diffs, same_max, same_thr = [], 0, 0
    k_live = {"max": 0, "thr": 0, "max_topic": 0, "thr_topic": 0}
    k_off = {"max": 0, "thr": 0}
    npl = npl_t = 0
    L_needed = {}
    for s in slots:
        L_needed.setdefault(s["layer"], set()).add(s["latent"])
    R = {}
    for L, lat in L_needed.items():
        X = np.load(os.path.join(DIAG, "cache", f"bank_acts_L{L}_R.npy"), mmap_mode="r")
        lat = sorted(lat)
        R[L] = (np.asarray(X[:, lat], dtype=np.float32), {j: i for i, j in enumerate(lat)})
        del X
    for si, s in enumerate(slots):
        A, col = R[s["layer"]]
        off = [A[[first[(cidx[c], st)] for st in chosen], col[s["latent"]]] for c in s["menu"]]
        live = [V[si, j] for j in range(20)]
        diffs.append(np.abs(np.concatenate(off) - np.concatenate(live)))

        def dec(vals):
            p, m1, gap, Au = features([np.asarray(v, np.float64) for v in vals])
            return p, (p if Au >= 0.78 else NF)
        po, to = dec(off)
        pl, tl = dec(live)
        same_max += po == pl
        same_thr += to == tl
        if s["planted"]:
            npl += 1
            k_live["max"] += pl == s["truth"]
            k_live["thr"] += tl == s["truth"]
            k_off["max"] += po == s["truth"]
            k_off["thr"] += to == s["truth"]
            if s["family"] == "topic":
                npl_t += 1
                k_live["max_topic"] += pl == s["truth"]
                k_live["thr_topic"] += tl == s["truth"]
    d = np.concatenate(diffs)
    out["sr_live_vs_offline"] = {
        "n_slots": len(slots), "n_values": int(d.size),
        "abs_diff": {"median": round(float(np.median(d)), 4), "p99": round(float(np.percentile(d, 99)), 4),
                     "max": round(float(d.max()), 4)},
        "same_SR-max_pick": same_max, "same_SR-thr_choice": same_thr,
        "planted_acc_live": {"SR-max": [k_live["max"], npl, round(k_live["max"] / npl, 4), RG.wilson(k_live["max"], npl)],
                             "SR-thr": [k_live["thr"], npl, round(k_live["thr"] / npl, 4), RG.wilson(k_live["thr"], npl)],
                             "SR-max_topic": [k_live["max_topic"], npl_t, round(k_live["max_topic"] / npl_t, 4),
                                              RG.wilson(k_live["max_topic"], npl_t)],
                             "SR-thr_topic": [k_live["thr_topic"], npl_t, round(k_live["thr_topic"] / npl_t, 4),
                                              RG.wilson(k_live["thr_topic"], npl_t)]},
        "planted_acc_offline_recomputed": {"SR-max": k_off["max"], "SR-thr": k_off["thr"], "n": npl}}
    print(json.dumps(out["sr_live_vs_offline"]))

    # ---------------- template live (60 slots)
    tl = json.load(open(os.path.join(CACHE, "tmpl_live.json")))
    sd = {f"{s['iid']}:{s['slot']}": s for s in slots}
    dd, same = [], 0
    for key, v in tl.items():
        iid, sl, kind = key.rsplit(":", 2)
        s = sd[f"{iid}:{sl}"]
        own = np.array(v["owner"])
        live = np.array(v["vals"], np.float32)
        off = np.concatenate([acts.vals(kind, s["layer"], s["latent"], c) for c in s["menu"]])
        dd.append(np.abs(live - off))
        pl = features([live[own == j] for j in range(20)])[0]
        po = features([off[own == j] for j in range(20)])[0]
        same += pl == po
    dd = np.concatenate(dd)
    out["template_live_vs_offline"] = {"n_slot_recipes": len(tl), "same_max_pick": same,
                                       "abs_diff_median": round(float(np.median(dd)), 4),
                                       "abs_diff_max": round(float(dd.max()), 4)}
    print(json.dumps(out["template_live_vs_offline"]))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
