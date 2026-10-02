"""Shared helpers for the fingerprint diagnosis (PREREG P6 / Amendment 3). CPU only, in memory, writes no instance.

What is here:
- load_worlds(): generator v2's Tables once, plus two pool views: "v2" (unfiltered) and "v2f" (restricted to the A1.1
  kept latents exactly as write_v2f.restrict_tables does). The heavy arrays are shared, only the pool dicts differ.
- draw(): slots from generate.make_instance with fresh seeds, optionally with a different close-tier menu construction
  (VARIANTS). Variant "current" calls the unchanged generator. Nothing is written to disk.
- Feature sets for predicting planted vs null from what an agent sees (layer + the 20 option labels):
    menu  : fingerprint_check.feats (needs the DBpedia taxonomy paths = generator tables; P6's check)
    str   : features computed from the label strings alone (no tables, no world knowledge)
    bag   : which labels are on the menu, one indicator per (label) and per (label, layer)
- Evaluation: 5-fold CV exactly as P6 (reproduction), and train-on-seeds / test-on-fresh-seeds AUROC with a 95% CI
  from a bootstrap over instances (slots of one episode are not independent).
"""
import copy
import json
import os
import re
from collections import Counter

import numpy as np

from tasks.featurematch.diagnosis import style_filter as SF
from tasks.featurematch.diagnosis import write_v2f as WV
from tasks.featurematch.fm_core import LAYERS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
BOOT_SEED = 20261002


# ----------------------------------------------------------------------------------------------------------- worlds
def load_worlds():
    """Returns (T_v2, T_v2f). Both share the big read-only arrays; only planted / planted_by_c / answerable differ."""
    T = SF.load_tables(SF.default_src_cache())
    kept, n = WV.load_kept(os.path.join(SF.HERE, "key_check.jsonl"))
    assert n == sum(len(T.planted[L]) for L in LAYERS), "key_check.jsonl does not match the generator pool"
    Tf = copy.copy(T)                       # shallow: arrays shared
    Tf.planted = dict(T.planted)
    Tf.planted_by_c = dict(T.planted_by_c)
    Tf.answerable = dict(T.answerable)
    Tf.pool_stats = copy.deepcopy(T.pool_stats)
    WV.restrict_tables(Tf, kept)
    return T, Tf


# ------------------------------------------------------------------------------------------------- menu variants
def _sib(T, center, L, pool=None):
    depth = len(T.paths[center]) - 1
    src = T.answerable[L] if pool is None else pool
    return [int(c) for c in src if T.close[center, c] >= depth]


def make_slot_variant(variant):
    """A drop-in replacement for generate.make_slot. 'current' reproduces the generator exactly (checked in tests).
    The variants change ONLY how the close-tier menu is built; the far tier (T1), the layer / anchor / latent draw,
    the exclusion set X, the centre draw and the asserts are untouched.

    current     planted = c* + top-19 of (U \\ X) around the centre;  null = top-20 of (U \\ X)       (generator v2)
    swap_nn     both kinds first build R = top-20 of (U \\ X) (= today's null menu). Null: menu = R. Planted: the menu
                option closest to c* in the taxonomy (ties: best-ranked in R) is replaced by c*.
    swap_sib    as swap_nn, but the replaced option is a uniformly random member of R in the centre's sibling group
                (fallback: the best-ranked member of R).
    decoy       draw a decoy d uniformly from the centre's non-excluded siblings (fallback: best-ranked of U \\ X);
                R = d + top-19 of (U \\ X \\ {d}). Null: menu = R. Planted: d replaced by c*.
    swap_fam    as swap_nn, but c* replaces the LOWEST-ranked option of its own family (language / topic), so the
                planted menu keeps the anchor's siblings (sibling-preserving alternative; fallback: last option).
    """
    from tasks.featurematch import generate as G

    def make_slot(T, rng, closeness, null, used):
        if variant == "current" or closeness != "close":
            return _orig_make_slot(T, rng, closeness, null, used)
        L = int(rng.choice(LAYERS))
        byc = T.planted_by_c[L]
        keys = sorted(byc)
        kind = "near_miss" if null else "planted"
        for attempt in range(200):
            cstar = keys[int(rng.integers(len(keys)))]
            j = byc[cstar][int(rng.integers(len(byc[cstar])))]
            if (L, j) in used:
                continue
            A = T.aA[L][j]
            Bv = T.aB[L][j]
            excl = set(np.nonzero((A >= G.N_THR) | (Bv >= G.N_THR_B))[0].tolist()) | {cstar}
            center = T.center_for(rng, cstar, L)
            pool = np.array([c for c in T.answerable[L] if c not in excl])
            if variant in ("swap_nn", "swap_sib", "swap_fam"):
                R = T.menu_around(rng, center, excl, closeness, G.N_OPT, L)          # ranked, best first
                if kind == "planted":
                    if variant == "swap_nn":
                        cl = np.array([T.close[cstar, c] for c in R])
                        pos = int(np.nonzero(cl == cl.max())[0][0])                 # closest to c*, best-ranked
                    elif variant == "swap_fam":
                        same = [i for i, c in enumerate(R) if T.paths[c][0] == T.paths[cstar][0]]
                        pos = same[-1] if same else len(R) - 1                      # lowest-ranked of c*'s family
                    else:
                        depth = len(T.paths[center]) - 1
                        sibpos = [i for i, c in enumerate(R) if T.close[center, c] >= depth]
                        pos = int(sibpos[int(rng.integers(len(sibpos)))]) if sibpos else 0
                    R = list(R)
                    R[pos] = cstar
                menu = list(R)
            elif variant == "decoy":
                sib = _sib(T, center, L, pool)
                if sib:
                    d = int(sib[int(rng.integers(len(sib)))])
                else:
                    key = -T.close[center, pool] + rng.random(len(pool)) * G.NOISE
                    d = int(pool[np.argsort(key)[0]])
                rest = T.menu_around(rng, center, excl | {d}, closeness, G.N_OPT - 1, L)
                menu = [cstar if kind == "planted" else d] + list(rest)
            else:
                raise ValueError(variant)
            menu = [menu[i] for i in rng.permutation(len(menu))]
            aucs = [float(A[c]) for c in menu]
            srt = sorted(aucs, reverse=True)
            if kind == "planted":
                assert menu[int(np.argmax(aucs))] == cstar and srt[0] >= G.P_THR and srt[1] < G.N_THR
                assert max(Bv[c] for c in menu if c != cstar) < G.N_THR_B <= G.P_THR_B <= Bv[cstar]
                choice = menu.index(cstar) + 1
            else:
                assert srt[0] < G.N_THR and max(Bv[c] for c in menu) < G.N_THR_B
                choice = "nothing found"
            assert len(set(menu)) == G.N_OPT
            used.add((L, j))
            return {"layer": L, "real_latent": j, "kind": kind, "anchor": T.cs[cstar]["cid"], "menu": menu,
                    "menu_auroc_A": [round(a, 4) for a in aucs], "choice": choice,
                    "naive_picks": [None if p is None else T.cs[p]["cid"] for p in T.naive_pick(L, j, menu)]}
        raise RuntimeError("could not build slot")

    return make_slot


from tasks.featurematch import generate as _G  # noqa: E402

_orig_make_slot = _G.make_slot
VARIANTS = ("current", "swap_nn", "swap_sib", "decoy", "swap_fam")


def draw(T, tier, seeds, variant="current"):
    """In-memory slots. Returns a list of dicts (one per slot) with the public view and the private facts."""
    from tasks.featurematch import generate as G
    G.make_slot = make_slot_variant(variant) if variant != "current" else _orig_make_slot
    cid2i = {c["cid"]: i for i, c in enumerate(T.cs)}
    lab2c = {c["label"]: i for i, c in enumerate(T.cs)}
    rows, fails = [], 0
    try:
        for gi, seed in enumerate(seeds):
            try:
                _, inst, pub = G.make_instance(T, int(seed), tier)
            except (AssertionError, RuntimeError):
                fails += 1
                continue
            for ps, a, x in zip(pub["slots"], inst["answer"]["slots"], inst["extra"]["slots"]):
                menu = [lab2c[o] for o in ps["options"]]
                assert menu == [cid2i[c] for c in x["menu"]]
                rows.append({"seed": int(seed), "layer": int(ps["layer"]), "labels": list(ps["options"]),
                             "menu": menu, "planted": bool(a["planted"]), "anchor": cid2i[x["anchor"]],
                             "latent": int(x["real_latent"])})
    finally:
        G.make_slot = _orig_make_slot
    return rows, fails


# ------------------------------------------------------------------------------------------------------- features
MENU_FEATS = ["layer", "n_lang", "n_l1_groups", "n_l2_groups", "max_l2_group", "max_l1_group", "mean_close",
              "max_rowmean_close", "rowmean_max_minus_median", "rowmean_top1_minus_top2"]


def menu_feats(T, rows):
    """fingerprint_check.feats, verbatim (uses the generator's taxonomy paths)."""
    from tasks.featurematch.fingerprint_check import feats
    lab2c = {c["label"]: i for i, c in enumerate(T.cs)}
    return np.array([feats(T, r["layer"], r["labels"], lab2c) for r in rows], dtype=np.float32)


LANG_PREFIX = "text written in "
TOPIC_PREFIX = re.compile(r"^article about (a|an) ")
STOP = {"a", "an", "the", "of", "about", "article", "us", "nfl", "ncaa", "wta"}


def label_tokens(label):
    """Label string -> (is_language, head word, content words). Pure string processing, no tables."""
    if label.startswith(LANG_PREFIX):
        return True, "language", {label[len(LANG_PREFIX):].lower()}
    rest = TOPIC_PREFIX.sub("", label).lower()
    words = [w for w in re.findall(r"[a-z]+", rest) if w not in STOP]
    head = words[-1] if words else rest
    return False, head, set(words)


STR_FEATS = ["layer", "n_lang", "n_head_groups", "max_head_group", "n_singleton_heads", "mean_word_overlap",
             "max_rowmean_overlap", "rowmean_max_minus_median", "n_pairs_sharing_word", "lang_x_layer"]


def str_feats_one(layer, labels):
    toks = [label_tokens(l) for l in labels]
    n_lang = sum(t[0] for t in toks)
    heads = Counter(t[1] for t in toks)
    W = [t[2] for t in toks]
    n = len(W)
    ov = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                ov[i, j] = len(W[i] & W[j]) / max(1, len(W[i] | W[j]))
    np.fill_diagonal(ov, np.nan)
    rm = np.nanmean(ov, 1)
    return [layer, n_lang, len(heads), max(heads.values()), sum(v == 1 for v in heads.values()), float(np.nanmean(ov)),
            float(rm.max()), float(rm.max() - np.median(rm)), int(np.nansum(ov > 0) // 2), n_lang * (layer / 6)]


def str_feats(rows):
    return np.array([str_feats_one(r["layer"], r["labels"]) for r in rows], dtype=np.float32)


def bag_feats(rows, labels_all):
    """One indicator per label (is it on the menu) + one per (label, layer) + layer one-hot. Labels only."""
    li = {l: i for i, l in enumerate(labels_all)}
    nl = len(labels_all)
    X = np.zeros((len(rows), nl * (1 + len(LAYERS)) + len(LAYERS)), dtype=np.float32)
    for k, r in enumerate(rows):
        lk = LAYERS.index(r["layer"])
        for l in r["labels"]:
            X[k, li[l]] = 1
            X[k, nl * (1 + lk) + li[l]] = 1
        X[k, nl * (1 + len(LAYERS)) + lk] = 1
    return X


def y_of(rows):
    return np.array([int(r["planted"]) for r in rows])


def groups_of(rows):
    return np.array([r["seed"] for r in rows])


# ----------------------------------------------------------------------------------------------------- statistics
def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def boot_ci(y, s, groups, n_boot=2000, seed=BOOT_SEED):
    """95% CI of AUROC(y, s) by resampling INSTANCES with replacement (slots of an episode stay together)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.nonzero(inv == g)[0] for g in range(len(ug))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(len(ug), size=len(ug))
        ii = np.concatenate([idx_by_g[p] for p in pick])
        if len(set(y[ii].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y[ii], s[ii]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def summarize_auc(y, s, groups, n_boot=2000):
    return {"auroc": round(auroc(y, s), 4), "ci95_instance_boot": boot_ci(y, s, groups, n_boot),
            "n_slots": int(len(y)), "n_instances": int(len(np.unique(groups))),
            "null_frac": round(float(1 - y.mean()), 4)}


def gbm():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(random_state=0)


def fit_predict(model, Xtr, ytr, Xte):
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def oof_scores(model_fn, X, y, groups, k=5, seed=0):
    """Out-of-fold scores with folds grouped by instance (StratifiedGroupKFold)."""
    from sklearn.model_selection import StratifiedGroupKFold
    s = np.zeros(len(y), dtype=np.float64)
    for tr, te in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y, groups):
        s[te] = fit_predict(model_fn(), X[tr], y[tr], X[te])
    return s


def dump(obj, name):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1)
    return p
