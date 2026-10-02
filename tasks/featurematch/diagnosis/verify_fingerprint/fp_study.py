"""Fingerprint diagnosis of the style-filtered FeatureMatch generator (PREREG P6 failed at 0.616; Amendment 3).

Three questions, one subcommand each (CPU only, everything in memory, no instance is written):

  reproduce    P6's exact check (seeds 500000+, 600 instances per tier, fingerprint_check.feats, 5-fold CV GBM) on the
               filtered (v2f) and unfiltered (v2) generator; T2 vs T3 identity; seed-to-seed spread on 10 fresh
               600-instance blocks; a held-out estimate (train 5000 instances, test 3000 fresh) with a 95% CI.
  attribute    which features carry the signal: single-feature AUROCs per layer, (layer + one feature) models,
               permutation importance (single and grouped) on the held-out set, partial dependence and the empirical
               planted rate by (layer, number of language options), the mechanism counts, and AUROC by layer and by
               anchor family.
  labelonly    recipes that see ONLY the layer and the 20 option labels (no generator table): zero-shot rules, a GBM
               on string features, a bag-of-labels logistic regression (learning curve over the number of training
               episodes), a GBM on bag + strings. Taxonomy-based recipes are reported as an upper bound.
  fix          the proposed menu-construction changes, simulated: P6's own check, the label-only recipes, the null
               fraction, failed draws, and side effects on closeness (dis_C tertiles, siblings among distractors);
               then the chosen variant again on fresh seeds (confirmation).
  residual     what signal is left under the chosen variant (by layer, by anchor family, per feature), whether the
               strongest label-only learner generalises to unseen latents / anchors (memorisation test), and a
               label-only nearest-neighbour ANSWER picker for planted slots.
  fixextra     one extra variant (default swap_fam, sibling-preserving), lighter protocol than `fix`.
  policy       a complete label-only policy (planted/null from bag_str_hgb + nearest-neighbour answer): episode pass
               rate, planted accuracy, null false claims on fresh episodes; kNN accuracy vs number of training episodes.
  knnsplit     the kNN answer leak when evaluation latents / anchor concepts are held out of training.

Seeds (disjoint from the pool 8000/108000/208000+, P6 500000+, audit 600000/610000+, prior.json 900000+):
  train 700000+, test 800000+, seed blocks 1000000 + 10000 b, confirmation 1400000+ (train) / 1500000+ (test) /
  1600000+ (P6 protocol).

Run (from ~/wt/fmdiag, after `source ~/ideating-rl-tests/common/env.sh`):
  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=2G -- $PY -m \
      tasks.featurematch.diagnosis.verify_fingerprint.fp_study {reproduce|attribute|labelonly|fix|residual|fixextra|policy|knnsplit|all}
Outputs: verify_fingerprint/{reproduce,attribute,labelonly,fix,residual,fixextra,policy,knnsplit}.json
"""
import argparse
import json
import os
import subprocess
import time
from collections import Counter, defaultdict

import numpy as np

from tasks.featurematch.diagnosis.verify_fingerprint import fp_common as F
from tasks.featurematch.fm_core import LAYERS

N_TRAIN, N_TEST = 5000, 3000
TRAIN0, TEST0 = 700000, 800000
CONF_TRAIN0, CONF_TEST0, CONF_P60 = 1400000, 1500000, 1600000
P6_SEEDS = range(500000, 500600)


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=F.HERE, text=True).strip()
    except Exception:
        return None


def hgb(**kw):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=False, random_state=0, **kw)


def p6_check(T, tier, seeds=P6_SEEDS, variant="current"):
    """PREREG P6 exactly as write_v2f.fingerprint (generator_in_memory)."""
    from tasks.featurematch.diagnosis import write_v2f as WV
    rows, fails = F.draw(T, tier, seeds, variant)
    out = WV._cv_auroc(F.menu_feats(T, rows), F.y_of(rows))
    out["failed_draws"] = fails
    return out


# --------------------------------------------------------------------------------------------------- statistics
def wauc(sidx, nu, y, w):
    """Weighted AUROC from pre-ranked scores (sidx = index of the score's unique value), ties count 1/2."""
    Wp = np.bincount(sidx, weights=w * y, minlength=nu)
    Wn = np.bincount(sidx, weights=w * (1 - y), minlength=nu)
    below = np.cumsum(Wn) - Wn
    return float((Wp * (below + 0.5 * Wn)).sum() / (Wp.sum() * Wn.sum()))


def boot(y, s, groups, n_boot=2000, seed=F.BOOT_SEED):
    """AUROC with a 95% CI from resampling instances (episodes) with replacement."""
    y = np.asarray(y, dtype=np.float64)
    us, sidx = np.unique(np.asarray(s), return_inverse=True)
    ug, ginv = np.unique(groups, return_inverse=True)
    point = wauc(sidx, len(us), y, np.ones(len(y)))
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(len(ug), size=len(ug)), minlength=len(ug))
        vals[b] = wauc(sidx, len(us), y, cnt[ginv].astype(np.float64))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"auroc": round(point, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n_slots": int(len(y)),
            "n_instances": int(len(ug)), "n_planted": int(y.sum()), "n_null": int(len(y) - y.sum())}


def boot_paired_diff(a, b, n_boot=2000, seed=F.BOOT_SEED):
    """AUROC(b) - AUROC(a) for two scored slot sets drawn from the SAME seeds; seeds are resampled jointly."""
    prep = []
    for y, s, g in (a, b):
        y = np.asarray(y, dtype=np.float64)
        us, sidx = np.unique(np.asarray(s), return_inverse=True)
        prep.append((y, sidx, len(us), np.asarray(g)))
    seeds = np.unique(np.concatenate([p[3] for p in prep]))
    pos = [np.searchsorted(seeds, p[3]) for p in prep]
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for k in range(n_boot):
        cnt = np.bincount(rng.integers(len(seeds), size=len(seeds)), minlength=len(seeds)).astype(np.float64)
        v = [wauc(p[1], p[2], p[0], cnt[q]) for p, q in zip(prep, pos)]
        d[k] = v[1] - v[0]
    full = [wauc(p[1], p[2], p[0], np.ones(len(p[0]))) for p in prep]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": round(full[1] - full[0], 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(float(c - h), 4), round(float(c + h), 4)]


# ---------------------------------------------------------------------------------------------------- data sets
class Data:
    """Train / test slots for one (world, tier, variant), drawn once and cached in memory."""

    def __init__(self, T, tier, variant="current", n_train=N_TRAIN, n_test=N_TEST, train0=TRAIN0, test0=TEST0):
        t0 = time.time()
        self.T, self.tier, self.variant = T, tier, variant
        self.tr, self.f_tr = F.draw(T, tier, range(train0, train0 + n_train), variant)
        self.te, self.f_te = F.draw(T, tier, range(test0, test0 + n_test), variant)
        self.ytr, self.yte = F.y_of(self.tr), F.y_of(self.te)
        self.gtr, self.gte = F.groups_of(self.tr), F.groups_of(self.te)
        self._cache = {}
        self.secs = round(time.time() - t0, 1)

    def X(self, kind, split):
        key = (kind, split)
        if key not in self._cache:
            rows = self.tr if split == "tr" else self.te
            if kind == "menu":
                X = F.menu_feats(self.T, rows)
            elif kind == "str":
                X = F.str_feats(rows)
            elif kind == "bag":
                X = F.bag_feats(rows, LABELS)
            elif kind == "bag+str":
                X = np.hstack([self.X("bag", split), self.X("str", split)])
            elif kind == "menu+bag":
                X = np.hstack([self.X("bag", split), self.X("menu", split)])
            elif kind == "lang":
                X = F.str_feats(rows)[:, [0, 1]]
            else:
                raise ValueError(kind)
            self._cache[key] = X
        return self._cache[key]


LABELS = None


def set_labels(T):
    global LABELS
    LABELS = [c["label"] for c in T.cs]


def fit_logreg_bag(Xtr, ytr, gtr):
    """L2 logistic regression on bag-of-labels; C picked by 3-fold instance-grouped CV on the training slots only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        sc = []
        for tr, va in GroupKFold(n_splits=3).split(Xtr, ytr, gtr):
            m = LogisticRegression(C=C, max_iter=3000).fit(Xtr[tr], ytr[tr])
            sc.append(F.auroc(ytr[va], m.predict_proba(Xtr[va])[:, 1]))
        if best is None or np.mean(sc) > best[0]:
            best = (float(np.mean(sc)), C)
    return LogisticRegression(C=best[1], max_iter=3000).fit(Xtr, ytr), best[1]


def recipe_scores(D, recipe, n_train=None):
    """Held-out scores of one recipe on D.te. n_train: use only the first n_train training instances."""
    tr_mask = np.ones(len(D.ytr), dtype=bool)
    if n_train is not None:
        seeds = np.unique(D.gtr)[:n_train]
        tr_mask = np.isin(D.gtr, seeds)
    ytr, gtr = D.ytr[tr_mask], D.gtr[tr_mask]
    info = {"n_train_instances": int(len(np.unique(gtr))), "n_train_slots": int(tr_mask.sum())}
    Ste = D.X("str", "te")
    if recipe == "zs_lang_purity":          # zero-shot: a planted menu has one extra option of the anchor's kind
        return np.abs(Ste[:, 1] - 10.0), info
    if recipe == "zs_max_head_group":
        return Ste[:, 3], info
    if recipe == "zs_mean_overlap":
        return Ste[:, 5], info
    if recipe == "lang_gbm":                # layer + number of "text written in ..." options
        m = F.gbm().fit(D.X("lang", "tr")[tr_mask], ytr)
        return m.predict_proba(D.X("lang", "te"))[:, 1], info
    if recipe == "str_gbm":
        m = F.gbm().fit(D.X("str", "tr")[tr_mask], ytr)
        return m.predict_proba(Ste)[:, 1], info
    if recipe == "bag_logreg":
        m, C = fit_logreg_bag(D.X("bag", "tr")[tr_mask], ytr, gtr)
        info["C"] = C
        return m.predict_proba(D.X("bag", "te"))[:, 1], info
    if recipe == "bag_str_hgb":
        m = hgb().fit(D.X("bag+str", "tr")[tr_mask], ytr)
        return m.predict_proba(D.X("bag+str", "te"))[:, 1], info
    if recipe == "menu_gbm":                # P6's features (taxonomy = generator table / world knowledge)
        m = F.gbm().fit(D.X("menu", "tr")[tr_mask], ytr)
        return m.predict_proba(D.X("menu", "te"))[:, 1], info
    if recipe == "menu_bag_hgb":
        m = hgb().fit(D.X("menu+bag", "tr")[tr_mask], ytr)
        return m.predict_proba(D.X("menu+bag", "te"))[:, 1], info
    raise ValueError(recipe)


LABEL_ONLY = ["zs_lang_purity", "zs_max_head_group", "zs_mean_overlap", "lang_gbm", "str_gbm", "bag_logreg",
              "bag_str_hgb"]
WITH_TAXONOMY = ["menu_gbm", "menu_bag_hgb"]


def extremes(y, s, fracs=(0.01, 0.05, 0.10)):
    """How confidently can the recipe call a slot? Planted rate among the slots it scores highest / lowest."""
    order = np.argsort(-np.asarray(s), kind="stable")
    out = {}
    for f in fracs:
        k = max(1, int(round(f * len(y))))
        top, bot = y[order[:k]], y[order[-k:]]
        out[f"top_{int(f * 100)}pct"] = {"n": k, "planted_rate": round(float(top.mean()), 4),
                                         "wilson95": wilson(int(top.sum()), k)}
        out[f"bottom_{int(f * 100)}pct"] = {"n": k, "planted_rate": round(float(bot.mean()), 4),
                                            "wilson95": wilson(int(bot.sum()), k)}
    out["base_planted_rate"] = round(float(np.mean(y)), 4)
    return out


def eval_recipes(D, recipes, n_boot=2000):
    out, scores = {}, {}
    for r in recipes:
        s, info = recipe_scores(D, r)
        scores[r] = s
        out[r] = {**boot(D.yte, s, D.gte, n_boot), **info, "extremes": extremes(D.yte, s)}
    return out, scores


# ----------------------------------------------------------------------------------------------------- reproduce
def cmd_reproduce(T, Tf):
    out = {"what": "PREREG P6 reproduction (write_v2f.fingerprint generator_in_memory protocol)", "git": git_sha()}
    out["p6_exact"] = {w: {t: p6_check(TT, t) for t in ("T1", "T2", "T3")} for w, TT in (("v2f", Tf), ("v2", T))}
    r2, _ = F.draw(Tf, "T2", P6_SEEDS)
    r3, _ = F.draw(Tf, "T3", P6_SEEDS)
    same = len(r2) == len(r3) and all(a["menu"] == b["menu"] and a["layer"] == b["layer"] and a["latent"] == b["latent"]
                                      and a["planted"] == b["planted"] for a, b in zip(r2, r3))
    out["T2_T3_in_memory_identical"] = {
        "identical": bool(same),
        "why": "generate.make_instance seeds its rng with [seed, 7] only; T2 and T3 differ only in the forward cap, so "
               "the same seed gives the same slots. P6's T3 number is the T2 number again, not a second measurement."}
    blocks = {}
    for w, TT in (("v2f", Tf), ("v2", T)):
        for tier in ("T1", "T2"):
            vals = [p6_check(TT, tier, range(1000000 + 10000 * b, 1000000 + 10000 * b + 600))["cv_auroc"]
                    for b in range(10)]
            blocks[f"{w}_{tier}"] = {"cv_auroc_per_block": vals, "mean": round(float(np.mean(vals)), 4),
                                     "min": min(vals), "max": max(vals),
                                     "n_blocks_over_0.60": int(sum(v > 0.60 for v in vals))}
    out["seed_blocks_10x600"] = blocks
    held = {}
    for w, TT in (("v2f", Tf), ("v2", T)):
        for tier in ("T1", "T2"):
            D = Data(TT, tier)
            s, info = recipe_scores(D, "menu_gbm")
            held[f"{w}_{tier}"] = {**boot(D.yte, s, D.gte), **info, "draw_failures": D.f_tr + D.f_te}
    out["held_out_menu_gbm"] = held
    out["note_written_pool"] = ("The written instances_v2f pool (about 240 slots per tier) gave T2 0.612 (sd 0.086) and "
                                "T3 0.586 (v2f_summary.json): consistent with the generator value, but far noisier.")
    return out


# ----------------------------------------------------------------------------------------------------- attribute
FEATURE_GROUPS = {
    "layer": [0], "n_lang": [1], "group_counts(n_l1,n_l2,max_l2,max_l1)": [2, 3, 4, 5],
    "closeness_stats(mean,max_rowmean,max-median,top1-top2)": [6, 7, 8, 9]}


def cmd_attribute(T, Tf):
    from sklearn.inspection import permutation_importance
    out = {"what": "which public menu features carry the planted-vs-null signal (filtered generator, close tier T2)",
           "features": F.MENU_FEATS, "git": git_sha()}
    D = Data(Tf, "T2")
    Xtr, Xte = D.X("menu", "tr"), D.X("menu", "te")
    model = F.gbm().fit(Xtr, D.ytr)
    s = model.predict_proba(Xte)[:, 1]
    out["held_out_full_model"] = boot(D.yte, s, D.gte)
    # single features: raw AUROC (sign as is), overall and per layer
    single = {}
    for k, f in enumerate(F.MENU_FEATS[1:], 1):
        e = {"all": round(F.auroc(D.yte, Xte[:, k]), 4)}
        for L in LAYERS:
            m = Xte[:, 0] == L
            e[f"L{L}"] = round(F.auroc(D.yte[m], Xte[m, k]), 4)
        single[f] = e
    out["single_feature_raw_auroc"] = single
    # layer + one feature (lets the model use non-monotone, layer-specific cut points)
    lf = {}
    for k, f in enumerate(F.MENU_FEATS[1:], 1):
        m = F.gbm().fit(Xtr[:, [0, k]], D.ytr)
        lf[f] = boot(D.yte, m.predict_proba(Xte[:, [0, k]])[:, 1], D.gte, 1000)
    lf["layer_only"] = boot(D.yte, F.gbm().fit(Xtr[:, [0]], D.ytr).predict_proba(Xte[:, [0]])[:, 1], D.gte, 1000)
    out["layer_plus_one_feature_gbm"] = lf
    # drop-one-group retrain
    drop = {}
    for g, cols in FEATURE_GROUPS.items():
        keep = [c for c in range(Xtr.shape[1]) if c not in cols]
        m = F.gbm().fit(Xtr[:, keep], D.ytr)
        drop[g] = boot(D.yte, m.predict_proba(Xte[:, keep])[:, 1], D.gte, 1000)
    out["drop_group_retrain"] = drop
    # permutation importance (single features, on the held-out set)
    pi = permutation_importance(model, Xte, D.yte, scoring="roc_auc", n_repeats=20, random_state=0)
    out["permutation_importance"] = {f: {"mean_auroc_drop": round(float(pi.importances_mean[k]), 4),
                                         "sd": round(float(pi.importances_std[k]), 4)}
                                     for k, f in enumerate(F.MENU_FEATS)}
    # grouped permutation (permute a group's columns together, rows shuffled jointly)
    rng = np.random.default_rng(0)
    base = F.auroc(D.yte, s)
    gp = {}
    for g, cols in FEATURE_GROUPS.items():
        drops = []
        for _ in range(20):
            Xp = Xte.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, cols] = Xp[perm][:, cols]
            drops.append(base - F.auroc(D.yte, model.predict_proba(Xp)[:, 1]))
        gp[g] = {"mean_auroc_drop": round(float(np.mean(drops)), 4), "sd": round(float(np.std(drops)), 4)}
    out["grouped_permutation_importance"] = gp
    # partial dependence of the full model on (layer, n_lang) + the empirical planted rate in the test set
    pdp, emp = {}, {}
    for L in LAYERS:
        vals = sorted(set(Xte[Xte[:, 0] == L, 1].astype(int).tolist()))
        row_pd, row_emp = {}, {}
        for v in vals:
            Xp = Xte.copy()
            Xp[:, 0], Xp[:, 1] = L, v
            row_pd[v] = round(float(model.predict_proba(Xp)[:, 1].mean()), 3)
            mm = (Xte[:, 0] == L) & (Xte[:, 1] == v)
            k, n = int(D.yte[mm].sum()), int(mm.sum())
            row_emp[v] = {"n": n, "planted_rate": round(k / n, 3), "wilson95": wilson(k, n)}
        pdp[f"L{L}"], emp[f"L{L}"] = row_pd, row_emp
    out["partial_dependence_layer_x_n_lang"] = pdp
    out["empirical_planted_rate_by_layer_x_n_lang"] = emp
    out["base_planted_rate_test"] = round(float(D.yte.mean()), 4)
    # held-out AUROC by layer and by anchor family (the family is private; diagnostic only)
    fam = np.array([T.paths[r["anchor"]][0] == "Language" for r in D.te])
    by = {}
    for L in LAYERS:
        m = Xte[:, 0] == L
        by[f"L{L}"] = boot(D.yte[m], s[m], D.gte[m], 1000)
    by["anchor_language"] = boot(D.yte[fam], s[fam], D.gte[fam], 1000)
    by["anchor_topic"] = boot(D.yte[~fam], s[~fam], D.gte[~fam], 1000)
    out["held_out_by_layer_and_anchor_family"] = by
    out["mechanism"] = mechanism(T, Tf)
    return out


def mechanism(T, Tf, n=3000):
    """Counts behind the explanation, filtered (v2f) vs unfiltered (v2) generator, T2, test seeds."""
    from tasks.featurematch import generate as G
    res = {}
    for w, TT in (("v2f", Tf), ("v2", T)):
        rows, _ = F.draw(TT, "T2", range(TEST0, TEST0 + n))
        per = {}
        for L in LAYERS:
            U = TT.answerable[L]
            langs_U = [int(c) for c in U if TT.paths[c][0] == "Language"]
            rs = [r for r in rows if r["layer"] == L]
            cand, same = [], {True: [], False: []}
            full = {True: 0, False: 0}
            nk = Counter(r["planted"] for r in rs)
            for r in rs:
                A, Bv = TT.aA[L][r["latent"]], TT.aB[L][r["latent"]]
                excl = set(np.nonzero((A >= G.N_THR) | (Bv >= G.N_THR_B))[0].tolist()) | {r["anchor"]}
                cand.append(len([c for c in U if c not in excl]))
                fam = TT.paths[r["anchor"]][0]
                same[r["planted"]].append(sum(TT.paths[c][0] == fam for c in r["menu"]))
                if sum(TT.paths[c][0] == "Language" for c in r["menu"]) == len(langs_U):
                    full[r["planted"]] += 1
            per[f"L{L}"] = {
                "menu_universe_size": int(len(U)), "languages_in_universe": len(langs_U),
                "median_candidates_after_exclusion": float(np.median(cand)),
                "menu_share_of_candidates": round(20 / float(np.median(cand)), 3),
                "options_in_anchor_family_mean": {"planted": round(float(np.mean(same[True])), 3),
                                                  "null": round(float(np.mean(same[False])), 3)},
                "menus_listing_every_language_of_the_layer": {"planted": full[True], "null": full[False],
                                                               "n_planted": nk[True], "n_null": nk[False]}}
        res[w] = per
    return res


# ----------------------------------------------------------------------------------------------------- labelonly
def cmd_labelonly(T, Tf):
    out = {"what": "planted-vs-null AUROC of recipes that see only the layer and the 20 option labels",
           "recipes": {
               "zs_lang_purity": "zero-shot, no training: |#language options - 10| (purer menu => planted)",
               "zs_max_head_group": "zero-shot: size of the largest group of options sharing the label's head noun",
               "zs_mean_overlap": "zero-shot: mean word overlap (Jaccard) between option labels",
               "lang_gbm": "GBM on (layer, #options starting 'text written in'), trained on generated episodes",
               "str_gbm": "GBM on 10 string features (layer, #language, head-noun groups, word overlap)",
               "bag_logreg": "L2 logistic regression on which labels are on the menu (232 + 3x232 label-by-layer "
                             "indicators + layer); C by grouped CV on the training set",
               "bag_str_hgb": "histogram GBM on bag-of-labels + string features (can learn counts and interactions)",
               "menu_gbm": "UPPER BOUND, not label-only: P6's features (needs the DBpedia taxonomy)",
               "menu_bag_hgb": "UPPER BOUND, not label-only: P6's features + bag-of-labels"},
           "protocol": f"train on {N_TRAIN} generated instances (seeds {TRAIN0}+), test on {N_TEST} fresh instances "
                       f"(seeds {TEST0}+); AUROC with a 95% CI from 2000 bootstrap resamples of test instances",
           "git": git_sha()}
    res = {}
    for w, TT in (("v2f", Tf), ("v2", T)):
        for tier in ("T2", "T1"):
            D = Data(TT, tier)
            r, _ = eval_recipes(D, LABEL_ONLY + WITH_TAXONOMY)
            res[f"{w}_{tier}"] = r
            if w == "v2f" and tier == "T2":
                curve = {}
                for nt in (100, 300, 1000, 3000, 5000):
                    curve[nt] = {}
                    for rec in ("lang_gbm", "str_gbm", "bag_logreg", "bag_str_hgb"):
                        s, info = recipe_scores(D, rec, n_train=nt)
                        curve[nt][rec] = {**boot(D.yte, s, D.gte, 1000), **info}
                out["learning_curve_v2f_T2"] = curve
    out["results"] = res
    best = max(LABEL_ONLY, key=lambda r: res["v2f_T2"][r]["auroc"])
    out["best_label_only_v2f_T2"] = {"recipe": best, **res["v2f_T2"][best]}
    out["criterion"] = "passed if the best label-only AUROC on the filtered close tier is <= 0.55"
    out["passed"] = bool(res["v2f_T2"][best]["auroc"] <= 0.55)
    return out


# ----------------------------------------------------------------------------------------------------------- fix
def side_effects(TT, rows, cscores):
    """Closeness of the menus (the close tier's difficulty dial) and the kind mix."""
    cid_rows = cscores
    dis_C, dis_A, n_sib, n_fam, lure_C, null_sib = [], [], [], [], [], []
    for r in rows:
        L, j, a = r["layer"], r["latent"], r["anchor"]
        Crow = cid_rows[(L, j)]
        Arow = TT.aA[L][j]
        depth = len(TT.paths[a]) - 1
        wrong = [c for c in r["menu"] if c != a]
        sib = sum(TT.close[a, c] >= depth for c in wrong)
        if r["planted"]:
            dis_C.append(float(np.nanmax([Crow[c] for c in wrong])))
            dis_A.append(float(max(Arow[c] for c in wrong)))
            n_sib.append(sib)
            n_fam.append(sum(TT.paths[c][0] == TT.paths[a][0] for c in wrong))
        else:
            lure_C.append(float(np.nanmax([Crow[c] for c in r["menu"]])))
            null_sib.append(sib)
    dis_C = np.array(dis_C)
    n_pl, n_nu = len(dis_C), len(lure_C)
    return {
        "n_slots": len(rows), "null_fraction": round(n_nu / len(rows), 4), "null_wilson95": wilson(n_nu, len(rows)),
        "planted_dis_C_median": round(float(np.median(dis_C)), 4), "planted_dis_C_mean": round(float(dis_C.mean()), 4),
        "planted_dis_C_tertiles(<0.578, 0.578-0.635, >=0.635)": [
            round(float((dis_C < 0.578).mean()), 4), round(float(((dis_C >= 0.578) & (dis_C < 0.635)).mean()), 4),
            round(float((dis_C >= 0.635).mean()), 4)],
        "planted_dis_A_median": round(float(np.median(dis_A)), 4),
        "planted_siblings_of_anchor_among_distractors_mean": round(float(np.mean(n_sib)), 3),
        "planted_share_with_>=1_sibling_distractor": round(float(np.mean(np.array(n_sib) >= 1)), 4),
        "planted_same_family_distractors_mean": round(float(np.mean(n_fam)), 3),
        "null_max_option_auroc_C_median": round(float(np.median(lure_C)), 4),
        "null_siblings_of_anchor_on_menu_mean": round(float(np.mean(null_sib)), 3),
        "n_planted": n_pl, "n_null": n_nu}


def load_cscores():
    z = np.load(os.path.join(F.SF.DIAG_CACHE, "pool_scores.npz"))
    out = {}
    for L in LAYERS:
        lat, Cm = z[f"L{L}_latents"], z[f"L{L}_C"]
        for i, j in enumerate(lat):
            out[(L, int(j))] = Cm[i]
    return out


FIX_VARIANTS = ("current", "swap_nn", "swap_sib", "decoy")
FIX_RECIPES = ["lang_gbm", "str_gbm", "bag_logreg", "bag_str_hgb", "menu_gbm", "menu_bag_hgb"]


def cmd_fix(T, Tf, confirm_variant="swap_nn"):
    cs = load_cscores()
    out = {"what": "close-tier menu construction variants, simulated in memory on the filtered generator",
           "variants_doc": F.make_slot_variant.__doc__, "git": git_sha()}
    dev = {}
    scores = {}
    for v in FIX_VARIANTS:
        e = {"p6_exact_protocol_T2": p6_check(Tf, "T2", variant=v)}
        D = Data(Tf, "T2", variant=v)
        e["draw_failures"] = {"train": D.f_tr, "test": D.f_te}
        r, sc = eval_recipes(D, FIX_RECIPES)
        e["held_out_recipes"] = r
        e["side_effects_T2_test"] = side_effects(Tf, D.te, cs)
        dev[v] = e
        scores[v] = (D.yte, sc, D.gte)
    out["dev"] = dev
    # paired differences vs current (same seeds)
    pd = {}
    for v in FIX_VARIANTS[1:]:
        pd[v] = {}
        for rec in FIX_RECIPES:
            a = (scores["current"][0], scores["current"][1][rec], scores["current"][2])
            b = (scores[v][0], scores[v][1][rec], scores[v][2])
            pd[v][rec] = boot_paired_diff(a, b, 1000)
    out["paired_auroc_diff_vs_current"] = pd
    # T1 is untouched by every variant (closeness 'far' bypasses the change): check
    rT1a, _ = F.draw(Tf, "T1", range(TEST0, TEST0 + 300))
    rT1b, _ = F.draw(Tf, "T1", range(TEST0, TEST0 + 300), confirm_variant)
    out["T1_unchanged_by_variant"] = bool(all(a["menu"] == b["menu"] for a, b in zip(rT1a, rT1b))
                                          and len(rT1a) == len(rT1b))
    # reference side effects: unfiltered v2 and filtered current (the frozen dis_C cuts come from v2)
    for w, TT in (("v2", T), ("v2f", Tf)):
        for tier in ("T1", "T2"):
            if (w, tier) == ("v2f", "T2"):
                continue                                    # = dev["current"]
            Dr = Data(TT, tier, n_train=0)
            out[f"side_effects_reference_{w}_{tier}"] = side_effects(TT, Dr.te, cs)
    # confirmation of the chosen variant on fresh seeds (chosen on the dev seeds above)
    conf = {}
    for v in ("current", confirm_variant):
        D = Data(Tf, "T2", variant=v, train0=CONF_TRAIN0, test0=CONF_TEST0)
        r, _ = eval_recipes(D, FIX_RECIPES)
        conf[v] = {"p6_exact_protocol_T2": p6_check(Tf, "T2", range(CONF_P60, CONF_P60 + 600), v),
                   "held_out_recipes": r, "side_effects_T2_test": side_effects(Tf, D.te, cs),
                   "draw_failures": {"train": D.f_tr, "test": D.f_te}}
    out["confirmation_fresh_seeds"] = {"variant": confirm_variant, "seeds": {"train": CONF_TRAIN0, "test": CONF_TEST0,
                                                                             "p6": CONF_P60}, **conf}
    return out


def _half(key):
    """Deterministic 50/50 split of latents or anchor concepts (sha256 parity)."""
    import hashlib
    return int(hashlib.sha256(str(key).encode()).hexdigest(), 16) % 2


def memorization_splits(D):
    """bag_str_hgb trained on slots whose latent (or anchor concept) is in half 0, tested on half 0 (seen in
    training) and on half 1 (never seen). Separates "learns menu structure" from "memorises each latent's menus"."""
    Xtr, Xte = D.X("bag+str", "tr"), D.X("bag+str", "te")
    out = {}
    for name, key in (("latent", lambda r: (r["layer"], r["latent"])), ("anchor_concept", lambda r: r["anchor"])):
        atr = np.array([_half(key(r)) for r in D.tr])
        ate = np.array([_half(key(r)) for r in D.te])
        m = hgb().fit(Xtr[atr == 0], D.ytr[atr == 0])
        for lab, mte in (("seen_in_training", ate == 0), ("unseen", ate == 1)):
            out[f"{name}_{lab}"] = boot(D.yte[mte], m.predict_proba(Xte[mte])[:, 1], D.gte[mte], 1000)
    return out


def knn_answer_leak(D, k=15):
    """Label-only ANSWER picker for planted slots: find the k training planted slots of the same layer whose option
    sets overlap most (Jaccard) with the test menu, and vote for their answers that are on the test menu (weight =
    similarity). Uses the answers of training episodes (what a policy could learn from feedback over many episodes).
    Reported overall and for test slots whose latent never appeared in a training slot. Chance = 1/20."""
    def mat(rows):
        M = np.zeros((len(rows), len(D.T.cs)), dtype=np.float32)
        for i, r in enumerate(rows):
            M[i, r["menu"]] = 1
        return M
    tr = [r for r in D.tr if r["planted"]]
    te = [r for r in D.te if r["planted"]]
    seen_lat = {(r["layer"], r["latent"]) for r in D.tr}
    correct, unseen = np.zeros(len(te), dtype=bool), np.zeros(len(te), dtype=bool)
    for L in LAYERS:
        itr = [i for i, r in enumerate(tr) if r["layer"] == L]
        ite = [i for i, r in enumerate(te) if r["layer"] == L]
        A = mat([tr[i] for i in itr])
        Bm = mat([te[i] for i in ite])
        inter = Bm @ A.T
        sim = inter / (40.0 - inter)
        ans_tr = np.array([tr[i]["anchor"] for i in itr])
        for row, i in enumerate(ite):
            r = te[i]
            nn = np.argsort(-sim[row], kind="stable")[:k]
            votes = Counter()
            menu = set(r["menu"])
            for n in nn:
                if ans_tr[n] in menu:
                    votes[int(ans_tr[n])] += float(sim[row, n])
            pick = votes.most_common(1)[0][0] if votes else r["menu"][0]
            correct[i] = pick == r["anchor"]
            unseen[i] = (L, r["latent"]) not in seen_lat
    res = {"k": k, "chance": 0.05}
    for lab, m in (("all_planted", np.ones(len(te), dtype=bool)), ("latent_unseen_in_training", unseen),
                   ("latent_seen_in_training", ~unseen)):
        n, c = int(m.sum()), int(correct[m].sum())
        res[lab] = {"n": n, "accuracy": round(c / n, 4) if n else None, "wilson95": wilson(c, n)}
    return res


def cmd_residual(T, Tf, variant="swap_nn"):
    """What is left after the fix, and what the strongest label-only learner actually learns: the P6 model by layer /
    anchor family, (layer + one feature) models, grouped permutation importance; the bag-of-labels GBM on seen vs
    unseen latents / anchor concepts; and a label-only nearest-neighbour ANSWER picker (current vs fixed, plus the
    unfiltered v2 generator as reference)."""
    out = {"what": f"residual planted-vs-null signal under variant {variant} vs current (T2), and the memorisation test",
           "git": git_sha()}
    for w, TT, v in (("v2f", Tf, "current"), ("v2f", Tf, variant), ("v2", T, "current")):
        D = Data(TT, "T2", variant=v)
        e = {}
        if w == "v2f":
            Xtr, Xte = D.X("menu", "tr"), D.X("menu", "te")
            model = F.gbm().fit(Xtr, D.ytr)
            s = model.predict_proba(Xte)[:, 1]
            e["menu_gbm"] = boot(D.yte, s, D.gte, 1000)
            fam = np.array([T.paths[r["anchor"]][0] == "Language" for r in D.te])
            e["menu_gbm_by_layer"] = {f"L{L}": boot(D.yte[Xte[:, 0] == L], s[Xte[:, 0] == L], D.gte[Xte[:, 0] == L],
                                                    500) for L in LAYERS}
            e["menu_gbm_anchor_language"] = boot(D.yte[fam], s[fam], D.gte[fam], 500)
            e["menu_gbm_anchor_topic"] = boot(D.yte[~fam], s[~fam], D.gte[~fam], 500)
            e["layer_plus_one_feature_gbm"] = {
                f: round(F.auroc(D.yte, F.gbm().fit(Xtr[:, [0, k]], D.ytr).predict_proba(Xte[:, [0, k]])[:, 1]), 4)
                for k, f in enumerate(F.MENU_FEATS[1:], 1)}
            rng = np.random.default_rng(0)
            base = F.auroc(D.yte, s)
            gp = {}
            for g, cols in FEATURE_GROUPS.items():
                drops = []
                for _ in range(10):
                    Xp = Xte.copy()
                    Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]
                    drops.append(base - F.auroc(D.yte, model.predict_proba(Xp)[:, 1]))
                gp[g] = round(float(np.mean(drops)), 4)
            e["menu_gbm_grouped_permutation_importance"] = gp
        e["bag_str_hgb_memorization_splits"] = memorization_splits(D)
        e["knn_answer_leak_planted"] = knn_answer_leak(D)
        out[f"{w}_{v}"] = e
    return out


def knn_picks(train_rows, test_rows, n_concepts, k=15):
    """Label-only answer for EVERY test slot: vote over the answers of the k most similar training PLANTED menus of
    the same layer (Jaccard over option sets), restricted to options on the test menu. Returns concept ids."""
    tr = [r for r in train_rows if r["planted"]]
    picks = [None] * len(test_rows)
    for L in LAYERS:
        itr = [i for i, r in enumerate(tr) if r["layer"] == L]
        ite = [i for i, r in enumerate(test_rows) if r["layer"] == L]
        if not ite:
            continue
        A = np.zeros((len(itr), n_concepts), dtype=np.float32)
        for a, i in enumerate(itr):
            A[a, tr[i]["menu"]] = 1
        ans = np.array([tr[i]["anchor"] for i in itr])
        for c0 in range(0, len(ite), 2000):
            chunk = ite[c0:c0 + 2000]
            Bm = np.zeros((len(chunk), n_concepts), dtype=np.float32)
            for b, i in enumerate(chunk):
                Bm[b, test_rows[i]["menu"]] = 1
            inter = Bm @ A.T if len(itr) else np.zeros((len(chunk), 0), dtype=np.float32)
            sim = inter / (40.0 - inter)
            for b, i in enumerate(chunk):
                menu = test_rows[i]["menu"]
                votes = Counter()
                if sim.shape[1]:
                    ms = set(menu)
                    for n in np.argsort(-sim[b], kind="stable")[:k]:
                        if ans[n] in ms:
                            votes[int(ans[n])] += float(sim[b, n])
                picks[i] = votes.most_common(1)[0][0] if votes else menu[0]
    return picks


def episode_metrics(rows, claim, picks):
    """Grade a per-slot policy: claim[i] (bool) -> answer picks[i], else "nothing found". Pass = every slot right."""
    by = defaultdict(list)
    ok = []
    for r, c, p in zip(rows, claim, picks):
        right = (r["planted"] and c and p == r["anchor"]) or (not r["planted"] and not c)
        ok.append(right)
        by[r["seed"]].append(right)
    ok = np.array(ok)
    y = np.array([r["planted"] for r in rows])
    claim = np.array(claim)
    n_pass = sum(all(v) for v in by.values())
    return {"n_episodes": len(by), "pass_rate": round(n_pass / len(by), 4), "pass_wilson95": wilson(n_pass, len(by)),
            "slot_accuracy": round(float(ok.mean()), 4),
            "planted_accuracy": round(float(ok[y].mean()), 4), "planted_accuracy_wilson95": wilson(int(ok[y].sum()), int(y.sum())),
            "planted_nothing_found_rate": round(float((~claim[y]).mean()), 4),
            "null_false_claim_rate": round(float(claim[~y].mean()), 4)}


def cmd_policy(T, Tf, variant="swap_nn"):
    """A complete LABEL-ONLY policy: P(planted) from bag_str_hgb, the answer from knn_picks; claim when P(planted) >=
    t, with t chosen on a validation split of the training seeds (train 4000 / validate 1000), then graded on the
    3000 test episodes. Plus the kNN answer accuracy as a function of the number of training episodes."""
    out = {"what": "label-only policy (no model access, no generator table): pass rate on fresh generated episodes",
           "git": git_sha()}
    nC = len(T.cs)
    for w, TT, v, tier in (("v2f", Tf, "current", "T2"), ("v2f", Tf, variant, "T2"), ("v2f", Tf, "current", "T1"),
                           ("v2", T, "current", "T2"), ("v2", T, "current", "T1")):
        D = Data(TT, tier, variant=v)
        seeds = np.unique(D.gtr)
        fit_m = np.isin(D.gtr, seeds[:4000])
        val_m = ~fit_m
        Xtr = D.X("bag+str", "tr")
        tr_fit = [r for r, m in zip(D.tr, fit_m) if m]
        val_rows = [r for r, m in zip(D.tr, val_m) if m]
        m = hgb().fit(Xtr[fit_m], D.ytr[fit_m])
        p_val = m.predict_proba(Xtr[val_m])[:, 1]
        picks_val = knn_picks(tr_fit, val_rows, nC)
        grid = [round(x, 2) for x in np.arange(0.2, 0.96, 0.05)]
        acc = {t: episode_metrics(val_rows, p_val >= t, picks_val)["pass_rate"] for t in grid}
        t_best = max(grid, key=lambda t: (acc[t], -abs(t - 0.5)))
        m = hgb().fit(Xtr, D.ytr)
        p_te = m.predict_proba(D.X("bag+str", "te"))[:, 1]
        picks = knn_picks(D.tr, D.te, nC)
        e = {"threshold_chosen_on_validation": t_best, "validation_pass_by_threshold": acc,
             "label_only_policy": episode_metrics(D.te, p_te >= t_best, picks),
             "always_claim_knn": episode_metrics(D.te, np.ones(len(D.te), dtype=bool), picks),
             "always_nothing_found": episode_metrics(D.te, np.zeros(len(D.te), dtype=bool), picks)}
        if (w, v, tier) == ("v2f", "current", "T2"):
            curve = {}
            yte = np.array([r["planted"] for r in D.te])
            for nt in (30, 100, 300, 1000, 3000, 5000):
                sub = [r for r in D.tr if r["seed"] < TRAIN0 + nt]
                pk = knn_picks(sub, D.te, nC)
                okp = np.array([pk[i] == r["anchor"] for i, r in enumerate(D.te)])[yte]
                curve[nt] = {"knn_planted_accuracy": round(float(okp.mean()), 4),
                             "wilson95": wilson(int(okp.sum()), int(len(okp)))}
            e["knn_learning_curve_n_train_episodes"] = curve
        out[f"{w}_{v}_{tier}"] = e
    return out


def cmd_knnsplit(T, Tf, variant="swap_nn"):
    """Does holding out latents or anchor concepts between training and evaluation stop the label-only answer leak?
    kNN answer accuracy on planted test slots whose latent / anchor concept is in half 1, with training episodes
    restricted to slots whose latent / anchor is in half 0 (vs trained on half 0 and tested on half 0)."""
    out = {"what": "label-only kNN answer accuracy on planted slots, seen vs held-out latents / anchor concepts",
           "git": git_sha()}
    for w, TT, v in (("v2f", Tf, "current"), ("v2f", Tf, variant), ("v2", T, "current")):
        D = Data(TT, "T2", variant=v)
        e = {}
        for name, key in (("latent", lambda r: (r["layer"], r["latent"])), ("anchor_concept", lambda r: r["anchor"])):
            tr0 = [r for r in D.tr if _half(key(r)) == 0]
            for lab, h in (("seen_half", 0), ("held_out_half", 1)):
                te = [r for r in D.te if r["planted"] and _half(key(r)) == h]
                pk = knn_picks(tr0, te, len(TT.cs))
                c = sum(p == r["anchor"] for p, r in zip(pk, te))
                e[f"{name}_{lab}"] = {"n": len(te), "accuracy": round(c / len(te), 4), "wilson95": wilson(c, len(te))}
        out[f"{w}_{v}"] = e
    return out


def cmd_fixextra(T, Tf, variant="swap_fam"):
    """One more variant, lighter than `fix`: P6 protocol on two seed sets, the counting and taxonomy recipes held
    out, and the closeness side effects."""
    cs = load_cscores()
    D = Data(Tf, "T2", variant=variant)
    out = {"variant": variant, "git": git_sha(),
           "p6_exact_protocol_T2": {"seeds_500000": p6_check(Tf, "T2", variant=variant),
                                    "seeds_1600000": p6_check(Tf, "T2", range(CONF_P60, CONF_P60 + 600), variant)},
           "draw_failures": {"train": D.f_tr, "test": D.f_te}}
    out["held_out_recipes"], _ = eval_recipes(D, ["lang_gbm", "str_gbm", "menu_gbm"])
    out["side_effects_T2_test"] = side_effects(Tf, D.te, cs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("reproduce", "attribute", "labelonly", "fix", "residual", "fixextra", "policy", "knnsplit", "all"))
    ap.add_argument("--confirm-variant", default="swap_nn")
    ap.add_argument("--extra-variant", default="swap_fam")
    a = ap.parse_args()
    t0 = time.time()
    T, Tf = F.load_worlds()
    set_labels(T)
    cmds = ("reproduce", "attribute", "labelonly", "fix") if a.cmd == "all" else (a.cmd,)
    for c in cmds:
        t1 = time.time()
        if c == "fix":
            res = cmd_fix(T, Tf, a.confirm_variant)
        elif c == "residual":
            res = cmd_residual(T, Tf, a.confirm_variant)
        elif c == "fixextra":
            res = cmd_fixextra(T, Tf, a.extra_variant)
        elif c == "policy":
            res = cmd_policy(T, Tf, a.confirm_variant)
        elif c == "knnsplit":
            res = cmd_knnsplit(T, Tf, a.confirm_variant)
        else:
            res = {"reproduce": cmd_reproduce, "attribute": cmd_attribute, "labelonly": cmd_labelonly}[c](T, Tf)
        res["seconds"] = round(time.time() - t1, 1)
        p = F.dump(res, f"{c}.json")
        print(f"{c}: wrote {p} in {res['seconds']} s", flush=True)
    print(f"total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
