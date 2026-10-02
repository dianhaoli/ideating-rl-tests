"""FeatureMatch diagnosis step 3: the pre-registered STYLE-ROBUST RECIPE (SR-max, SR-thr). See SR_RECIPE.md.

PREREG.md step 3 (binding):
  SR-max  for each slot, run 6 bank-R texts per option (one per style; 120 texts per slot, at most 600 forward units
          per 5-slot episode). Claim the option with the highest mean max activation of the slot's latent.
  SR-thr  as SR-max, but claim only if that option's AUROC against the other options' bank-R texts is >= 0.78 (the
          reference's threshold); otherwise answer "nothing found".

Implementation choices (made before any SR result existed; PREREG/PLAN do not fix them):
  * Which 6 texts: PREREG says "one per style" but bank R has 10 styles x 2 texts. The 6 styles are drawn once for
    the whole study, the same for every concept: random.Random(20261002).sample(sorted(the 10 style names), 6). Of
    each chosen style the FIRST text listed for that style in the bank-R file is used (the F1 convention of PREREG
    A1.1). Same styles for every option, so options are compared on style-matched texts.
  * Ties in the mean: the lowest option number wins (np.argmax). Options are shuffled per instance, so this is an
    arbitrary but deterministic pick. A latent silent on all 120 texts therefore claims option 1 under SR-max and
    gets AUROC 0.5 (-> "nothing found") under SR-thr.
  * AUROC is reference_solver.auroc (ties count 1/2): positives = the claimed option's 6 values, negatives = the
    other 19 options' 114 values.
  * Sensitivity variants SR-max-all20 / SR-thr-all20 use all 20 bank-R texts per option (400 forward units per slot;
    over every tier's budget, so not a feasible live recipe; reported only to show how much the 6-text choice
    matters).

PREREG Amendment 2 (2026-10-02, before any filtered-pool result) adds, all produced by one run:
  A2.1 the fixed selection above is the PRIMARY SR-max / SR-thr; the stop rule uses it.
  A2.2 text-selection sensitivity: SR-max / SR-thr planted accuracy over 42 selections (style seeds 20261002..
       20261022 x first/second text; the primary is one of them) as min / p10 / median / p90 / max (numpy linear
       percentiles), plus the all20 variants. Flag text_selection_sensitive = for SR-max or SR-thr,
       p10 <= 0.50 < p90 (the range holds values on both sides of the rule's "> 0.50").
  A2.3 capped variants SR-max-capped / SR-thr-capped: texts are consumed in slot order up to the instance's forward
       cap (instance.json caps.forward). Within a slot they are sent style-round-robin (chosen style 1 for options
       1..20, then style 2, ...), so a truncated slot gives every option floor(k/20) or that +1 texts. An option
       with no text cannot be claimed; a slot with no text gets option 1 (SR-max) / "nothing found" (SR-thr).
       The within-slot order is an implementation choice (A2.3 does not fix it). Over-cap episodes are counted.
  A2.5 planted accuracy also gets an instance-clustered bootstrap 95% CI (10,000 resamples of instances with
       replacement, indices drawn with random.Random(20261002).choices, percentile interval) next to Wilson.
  stop_rule block: fires = primary SR-max or SR-thr pooled planted accuracy > 0.50 with n_planted >= 100;
       borderline = it fires and the Wilson CI of every firing variant includes 0.50; text_selection_sensitive as
       above. Valid as the step-3 outcome only on the filtered pool (instances_v2f).

OFFLINE: activations are read from diagnosis/cache/bank_acts_L{L}_R.npy (float16 [4640, 16384], rows in generator
concept order, 20 per concept, bank-file order within a concept), computed by style_filter.py compute with
fm_core.Subject.maxpool_acts (max over non-BOS tokens, 64-token truncation) on whitespace-normalised texts. This is
the same per-text quantity latent_activations returns as "max" (see SR_RECIPE.md for the caveats: bf16 batch/padding
noise, float16 storage, one text whose U+3000 space was normalised). A live run would send the same 120 texts per
slot and cost the same forward units, which are reported per episode.

The recipe is a baseline, so it reads the privileged instance.json (real latent id, layer, menu concept ids) the way
recipe_baseline.py effectively does through the tools' permutation. Answers use the grader's submission format
{"answers": [{"slot": i, "choice": <1-based option> | "nothing found"}]} and are scored by grader.grade.

CLI:  $PY -m tasks.featurematch.diagnosis.sr_recipe --instances DIR --out DIR [--label TEXT]
Memory: the cache is memory-mapped and only the needed latent columns are copied (float32, a few MB).
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

from tasks.featurematch import grader
from tasks.featurematch.reference_solver import auroc

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
DIAG_CACHE = os.path.join(HERE, "cache")
DEFAULT_INSTANCES = os.path.join(TASK, "instances_v2f")
DEFAULT_OUT = os.path.join(HERE, "sr_recipe_out")
FM_MAIN = os.path.expanduser("~/wt/featurematch/tasks/featurematch")
LAYERS = (6, 12, 18)
SEED = 20261002
N_STYLES = 6
THR = 0.78
NOTHING = "nothing found"
# variant -> (selection mode, thresholded?)
VARIANTS = {"SR-max": ("sr6", False), "SR-thr": ("sr6", True),
            "SR-max-capped": ("sr6cap", False), "SR-thr-capped": ("sr6cap", True),
            "SR-max-all20": ("all20", False), "SR-thr-all20": ("all20", True)}
MODES = ("sr6", "sr6cap", "all20")
PRIMARY = ("SR-max", "SR-thr")
N_BOOT = 10000
STOP_THR = 0.50
STOP_MIN_PLANTED = 100


# ------------------------------------------------------------------------------------------------ text selection
def choose_styles(style_names, k=N_STYLES, seed=SEED):
    """The study-wide fixed choice of k styles: Random(seed).sample over the sorted distinct style names."""
    names = sorted(set(style_names))
    if len(names) < k:
        raise ValueError(f"need >= {k} styles, bank has {len(names)}")
    return random.Random(seed).sample(names, k)


def select_rows(rows, styles, n_c, mode, chosen=None):
    """Per concept index, the cache row indices the recipe runs. mode 'sr6': for each chosen style (in the chosen
    order) the first row of that (concept, style) in bank order; 'all20': every row of the concept."""
    rows = [int(r) for r in rows]
    by_c = defaultdict(list)
    for i, r in enumerate(rows):
        by_c[r].append(i)
    out = {}
    for c in range(n_c):
        idx = by_c.get(c, [])
        if mode == "all20":
            out[c] = list(idx)
        elif mode == "sr6":
            first = {}
            for i in idx:
                first.setdefault(styles[i], i)
            missing = [s for s in chosen if s not in first]
            if missing and idx:
                raise ValueError(f"concept {c} lacks styles {missing}")
            out[c] = [first[s] for s in chosen] if idx else []
        else:
            raise ValueError(mode)
    return out


# -------------------------------------------------------------------------------------------------- decision rule
def decide(option_vals, thr=THR):
    """option_vals: list over options (menu order) of 1-D arrays of max activations.
    Returns dict: sr_max (1-based option), sr_thr (option or 'nothing found'), means, auc_best.
    An option with no values (capped variant) has mean -inf and cannot be claimed; if no option has a value, SR-max
    claims option 1 and SR-thr says "nothing found"."""
    vals = [np.asarray(v, dtype=np.float64).ravel() for v in option_vals]
    means = np.array([float(v.mean()) if v.size else -np.inf for v in vals])
    if not np.isfinite(means).any():
        return {"sr_max": 1, "sr_thr": NOTHING, "means": means, "auc_best": float("nan")}
    best = int(np.argmax(means))                       # ties -> lowest option number
    neg = np.concatenate([v for j, v in enumerate(vals) if j != best])
    auc = auroc(vals[best], neg)                       # reference_solver.auroc: 0.5 if there is no negative
    return {"sr_max": best + 1, "sr_thr": (best + 1) if auc >= thr else NOTHING, "means": means, "auc_best": auc}


def capped_counts(n_slots, cap, per_slot=N_STYLES * 20):
    """A2.3: texts per slot when an episode's texts are consumed in slot order up to the forward cap."""
    out, used = [], 0
    for _ in range(n_slots):
        k = max(0, min(per_slot, int(cap) - used))
        out.append(k)
        used += k
    return out


def round_robin(option_rows, k):
    """A2.3 within-slot order: style rank 1 for every option (menu order), then rank 2, ... The first k texts of
    that order, returned per option. option_rows: list over options of their (equal-length) row lists."""
    n_opt = len(option_rows)
    return [rows[: k // n_opt + (1 if j < k % n_opt else 0)] for j, rows in enumerate(option_rows)]


def forward_cap(inst):
    """The episode's forward cap from instance.json (caps.forward; dial.forward_cap must agree if present)."""
    cap = (inst.get("caps") or {}).get("forward")
    dial = (inst.get("dial") or {}).get("forward_cap")
    if cap is None:
        cap = dial
    if cap is None:
        raise ValueError(f"{inst.get('instance_id')}: no forward cap in instance.json")
    if dial is not None and int(dial) != int(cap):
        raise ValueError(f"{inst.get('instance_id')}: caps.forward {cap} != dial.forward_cap {dial}")
    return int(cap)


# ------------------------------------------------------------------------------------------------------ instances
def iter_instances(inst_dir):
    for d in sorted(glob.glob(os.path.join(inst_dir, "*"))):
        p = os.path.join(d, "instance.json")
        if os.path.isfile(p):
            yield d, json.load(open(p)), json.load(open(os.path.join(d, "public.json")))


def src_cache():
    if os.environ.get("FM_SRC_CACHE"):
        return os.environ["FM_SRC_CACHE"]
    local = os.path.join(TASK, "cache", "concepts.json")
    return os.path.dirname(local) if os.path.exists(local) else os.path.join(FM_MAIN, "cache")


def load_labels():
    p = os.path.join(src_cache(), "concepts.json")
    if not os.path.exists(p):
        return None
    return {c["cid"]: c["label"] for c in json.load(open(p))}


def bank_check(meta, banks_dir=os.path.join(HERE, "banks"), max_chars=400):
    """Re-read bank R and confirm the cached rows are these texts, in this order (style_filter's normalisation and
    hashes). Returns {"texts_sha256_match": bool, "styles_match": bool, "n": int}."""
    allc = {}
    for f in sorted(glob.glob(os.path.join(banks_dir, "R", "chunk_*.json"))):
        allc.update(json.load(open(f))["concepts"])
    texts, styles = [], []
    for c in meta["cids"]:
        for t in allc.get(c, []):
            texts.append(" ".join(str(t["text"]).split())[:max_chars])
            styles.append(t["style"])
    R = meta["banks"]["R"]
    return {"n": len(texts), "texts_sha256_match": hashlib.sha256("\n".join(texts).encode()).hexdigest()
            == R.get("texts_sha256"), "styles_match": styles == R["styles"]}


def slot_specs(inst, pub, cidx, labels=None):
    """Per slot: layer, real latent, menu concept indices, planted, true choice. Checks the public/private mapping."""
    ex = inst["extra"]["slots"]
    truth = inst["answer"]["slots"]
    if not (len(ex) == len(truth) == len(pub["slots"])):
        raise ValueError(f"{inst['instance_id']}: slot counts differ")
    out = []
    for i, (e, t, p) in enumerate(zip(ex, truth, pub["slots"])):
        menu = e["menu"]
        if len(menu) != len(p["options"]) or p["slot"] != i or int(p["layer"]) != int(e["layer"]):
            raise ValueError(f"{inst['instance_id']} slot {i}: public/private mismatch")
        if labels is not None and [labels[c] for c in menu] != list(p["options"]):
            raise ValueError(f"{inst['instance_id']} slot {i}: option labels do not match menu order")
        if t["planted"]:
            if menu[t["choice"] - 1] != e["anchor"]:
                raise ValueError(f"{inst['instance_id']} slot {i}: planted answer is not the anchor")
        elif e["anchor"] in menu or t["choice"] != NOTHING:
            raise ValueError(f"{inst['instance_id']} slot {i}: malformed null slot")
        out.append({"slot": i, "layer": int(e["layer"]), "latent": int(e["real_latent"]), "anchor": e["anchor"],
                    "menu": [cidx[c] for c in menu], "planted": bool(t["planted"]), "truth": t["choice"]})
    return out


# ---------------------------------------------------------------------------------------------------------- stats
def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def rate(k, n):
    return round(k / n, 4) if n else None


_BOOT_IDX = {}


def boot_index(n, b=N_BOOT, seed=SEED):
    """[b, n] instance indices drawn with replacement by a fresh random.Random(seed) (so every CI is reproducible on
    its own, whatever else ran before)."""
    key = (n, b, seed)
    if key not in _BOOT_IDX:
        rng = random.Random(seed)
        pop = range(n)
        _BOOT_IDX[key] = np.array([rng.choices(pop, k=n) for _ in range(b)], dtype=np.int32)
    return _BOOT_IDX[key]


def cluster_bootstrap(k_inst, n_inst, b=N_BOOT, seed=SEED):
    """A2.5: instance-clustered bootstrap 95% CI of sum(k)/sum(n) (percentile interval). Resamples with no planted
    slot (sum n = 0) are dropped."""
    k_inst, n_inst = np.asarray(k_inst, dtype=np.float64), np.asarray(n_inst, dtype=np.float64)
    if n_inst.sum() == 0:
        return [None, None]
    idx = boot_index(len(k_inst), b, seed)
    K, N = k_inst[idx].sum(1), n_inst[idx].sum(1)
    r = K[N > 0] / N[N > 0]
    lo, hi = np.percentile(r, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def metrics(recs):
    """recs: list of {grade, forward, cap, uncapped_forward, n_slots_truncated}. Step-3 metrics plus A2.3/A2.5."""
    n = len(recs)
    k_pass = sum(r["grade"]["pass"] for r in recs)
    pl = [s for r in recs for s in r["grade"]["details"]["slots"] if s["planted"]]
    nu = [s for r in recs for s in r["grade"]["details"]["slots"] if not s["planted"]]
    k_pl = sum(s["correct"] for s in pl)
    k_nf = sum(not s["claimed"] for s in pl)
    k_fc = sum(s["claimed"] for s in nu)
    k_inst = [sum(s["correct"] for s in r["grade"]["details"]["slots"] if s["planted"]) for r in recs]
    n_inst = [sum(s["planted"] for s in r["grade"]["details"]["slots"]) for r in recs]
    fw = [r["forward"] for r in recs]
    return {"n_instances": n, "pass": k_pass, "pass_rate": rate(k_pass, n), "pass_wilson95": wilson(k_pass, n),
            "n_planted": len(pl), "planted_acc": rate(k_pl, len(pl)), "planted_acc_wilson95": wilson(k_pl, len(pl)),
            "planted_acc_cluster_boot95": cluster_bootstrap(k_inst, n_inst),
            "planted_nothing_found_rate": rate(k_nf, len(pl)),
            "n_null": len(nu), "null_false_claim_rate": rate(k_fc, len(nu)),
            "mean_score": round(float(np.mean([r["grade"]["score"] for r in recs])), 4) if n else None,
            "forward_units_mean": round(float(np.mean(fw)), 1) if n else None,
            "forward_units_max": int(max(fw)) if n else None,
            "n_over_forward_cap": sum(r["forward"] > r["cap"] for r in recs),
            "n_episodes_needing_more_than_cap": sum(r["uncapped_forward"] > r["cap"] for r in recs),
            "n_slots_truncated": sum(r["n_slots_truncated"] for r in recs)}


def pct_stats(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
    q = np.percentile(v, [0, 10, 50, 90, 100])
    return {"n": int(v.size), "min": round(float(q[0]), 4), "p10": round(float(q[1]), 4),
            "median": round(float(q[2]), 4), "p90": round(float(q[3]), 4), "max": round(float(q[4]), 4)}


def straddles(st, thr=STOP_THR):
    """A2.2: the 10th-90th percentile range holds values on both sides of the rule's '> thr'."""
    return bool(st["p10"] <= thr < st["p90"])


def stop_rule(metrics_all, sensitivity, valid_pool, thr=STOP_THR, min_planted=STOP_MIN_PLANTED):
    """Section 4 stop rule on the PRIMARY variants (pooled over tiers), with the A2.2 / A2.5 annotations."""
    per = {}
    for v in PRIMARY:
        m = metrics_all[v]
        lo, hi = m["planted_acc_wilson95"]
        per[v] = {"planted_acc": m["planted_acc"], "n_planted": m["n_planted"], "wilson95": [lo, hi],
                  "cluster_boot95": m["planted_acc_cluster_boot95"],
                  "above_thr": bool(m["planted_acc"] is not None and m["planted_acc"] > thr),
                  "wilson_includes_thr": bool(lo is not None and lo <= thr <= hi)}
    n_ok = all(per[v]["n_planted"] >= min_planted for v in PRIMARY)
    firing = [v for v in PRIMARY if per[v]["above_thr"]]
    fires = bool(n_ok and firing)
    borderline = bool(fires and all(per[v]["wilson_includes_thr"] for v in firing))
    sens = bool(sensitivity and sensitivity.get("text_selection_sensitive"))
    labels = ([] if valid_pool else ["NOT a step-3 outcome (pool is not instances_v2f)"]) + \
        (["fires"] if fires else ["does not fire"]) + (["borderline"] if borderline else []) + \
        (["text-selection-sensitive"] if sens else [])
    return {"valid_as_step3_outcome": bool(valid_pool),
            "rule": f"fires if primary SR-max or SR-thr planted accuracy (all tiers pooled) > {thr} with "
                    f"n_planted >= {min_planted}; borderline if every firing variant's Wilson 95% CI includes {thr}",
            "n_planted_ok": n_ok, "per_variant": per, "firing_variants": firing, "fires": fires,
            "borderline": borderline, "text_selection_sensitive": sens, "labels": labels}


# ----------------------------------------------------------------------------------------------------- provenance
def sha256_file(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pool_digest(inst_dir):
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(inst_dir, "*", "instance.json"))):
        h.update(os.path.basename(os.path.dirname(p)).encode())
        h.update(open(p, "rb").read())
    return h.hexdigest()


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def is_filtered_pool(inst_dir):
    return os.path.basename(os.path.normpath(inst_dir)) == "instances_v2f"


def default_label(inst_dir):
    if is_filtered_pool(inst_dir):
        return "filtered v2 pool (instances_v2f)"
    return "NOT the step-3 result: run on a pool other than instances_v2f"


# ------------------------------------------------------------------------------------------------------------ run
def load_acts(diag_cache, needed, verify=True):
    """needed {layer: sorted latent ids} -> ({layer: float32 [4640, k]}, {layer: {latent: col}}, file info)."""
    acts, col, info = {}, {}, {}
    man = os.path.join(diag_cache, "manifest.json")
    man = json.load(open(man)).get("files", {}) if os.path.exists(man) else {}
    for L, lat in needed.items():
        p = os.path.join(diag_cache, f"bank_acts_L{L}_R.npy")
        X = np.load(p, mmap_mode="r")
        lat = sorted(lat)
        acts[L] = np.asarray(X[:, lat], dtype=np.float32)
        col[L] = {j: i for i, j in enumerate(lat)}
        del X
        rec = {"shape": list(np.load(p, mmap_mode="r").shape)}
        if verify:
            rec["sha256"] = sha256_file(p)
            exp = (man.get(os.path.basename(p)) or {}).get("sha256")
            rec["matches_cache_manifest"] = (exp == rec["sha256"]) if exp else None
        info[L] = rec
    return acts, col, info


def selection_sensitivity(insts, acts, col, rows, styles, n_c, n_alt=20, text_index=(0, 1), all20=None):
    """A2.2: planted accuracy of SR-max / SR-thr (and SR-thr's null false-claim rate) under 42 selections: style
    seeds SEED .. SEED+n_alt, each with the first and the second text of every chosen style. The primary selection
    (SEED, first text) is one of them. Planted accuracy is computed directly (pick == key), which equals the
    grader's per-slot rule."""
    by_cs = defaultdict(list)
    for i, (r, st) in enumerate(zip(rows, styles)):
        by_cs[(int(r), st)].append(i)
    out = []
    for k in range(n_alt + 1):
        chosen = choose_styles(styles, seed=SEED + k)
        for ti in text_index:
            sel = {c: [by_cs[(c, st)][ti] for st in chosen] for c in range(n_c) if (c, chosen[0]) in by_cs}
            pm = pt = npl = nfc = nnu = 0
            for _, _, specs in insts:
                for s in specs:
                    a = acts[s["layer"]][:, col[s["layer"]][s["latent"]]]
                    res = decide([a[sel[c]] for c in s["menu"]])
                    if s["planted"]:
                        npl += 1
                        pm += res["sr_max"] == s["truth"]
                        pt += res["sr_thr"] == s["truth"]
                    else:
                        nnu += 1
                        nfc += res["sr_thr"] != NOTHING
            out.append({"seed": SEED + k, "text_index": ti, "primary": k == 0 and ti == 0, "styles": chosen,
                        "sr_max_planted_acc": rate(pm, npl), "sr_thr_planted_acc": rate(pt, npl),
                        "sr_thr_null_false_claim": rate(nfc, nnu)})
    st = {"SR-max": pct_stats([o["sr_max_planted_acc"] for o in out]),
          "SR-thr": pct_stats([o["sr_thr_planted_acc"] for o in out]),
          "SR-thr null false claim": pct_stats([o["sr_thr_null_false_claim"] for o in out])}
    flag = {v: straddles(st[v]) for v in ("SR-max", "SR-thr")}
    return {"note": "A2.2; 21 style seeds x first/second text, primary included; numpy linear percentiles",
            "n_selections": len(out), "planted_acc": st, "all20_planted_acc": all20 or {},
            "straddles_0.50": flag, "text_selection_sensitive": any(flag.values()),
            "flag_rule": "p10 <= 0.50 < p90 for SR-max or SR-thr", "runs": out}


def run(inst_dir, out_dir, diag_cache=DIAG_CACHE, label=None, verify=True, check_labels=True, banks_dir=None,
        n_alt=20, log=print):
    t0 = time.time()
    meta = json.load(open(os.path.join(diag_cache, "bank_meta.json")))
    cids = meta["cids"]
    cidx = {c: i for i, c in enumerate(cids)}
    R = meta["banks"]["R"]
    rows, styles = R["rows"], R["styles"]
    if len(rows) != R["n"] or len(styles) != R["n"]:
        raise ValueError("bank_meta R rows/styles length mismatch")
    chosen = choose_styles(styles)
    sel = {"sr6": select_rows(rows, styles, len(cids), "sr6", chosen),
           "all20": select_rows(rows, styles, len(cids), "all20")}
    labels = load_labels() if check_labels else None
    bcheck = bank_check(meta, banks_dir or os.path.join(HERE, "banks"))
    if not (bcheck["texts_sha256_match"] and bcheck["styles_match"]):
        raise SystemExit(f"bank R files do not match the cached rows: {bcheck}")

    insts = []
    for d, inst, pub in iter_instances(inst_dir):
        insts.append((d, inst, slot_specs(inst, pub, cidx, labels)))
    if not insts:
        raise SystemExit(f"no instances under {inst_dir}")
    needed = defaultdict(set)
    for _, _, specs in insts:
        for s in specs:
            needed[s["layer"]].add(s["latent"])
    acts, col, finfo = load_acts(diag_cache, needed, verify=verify)

    os.makedirs(out_dir, exist_ok=True)
    recs = {v: [] for v in VARIANTS}
    slot_rows = []
    for d, inst, specs in insts:
        cap = forward_cap(inst)
        kcap = capped_counts(len(specs), cap)
        answers = {v: [] for v in VARIANTS}
        for s, k in zip(specs, kcap):
            a = acts[s["layer"]][:, col[s["layer"]][s["latent"]]]
            row = {"instance": inst["instance_id"], "tier": inst["tier"], "slot": s["slot"],
                   "kind": "planted" if s["planted"] else "null", "layer": s["layer"], "latent": s["latent"],
                   "anchor": s["anchor"], "truth": s["truth"], "cap": cap, "sr6cap_texts": k}
            for mode in MODES:
                if mode == "sr6cap":
                    opt_rows = round_robin([sel["sr6"][c] for c in s["menu"]], k)
                else:
                    opt_rows = [sel[mode][c] for c in s["menu"]]
                res = decide([a[r] for r in opt_rows])
                for v, (m, thr) in VARIANTS.items():
                    if m == mode:
                        answers[v].append({"slot": s["slot"], "choice": res["sr_thr"] if thr else res["sr_max"]})
                row[f"{mode}_pick"] = res["sr_max"]
                fin = res["means"][np.isfinite(res["means"])]
                row[f"{mode}_best_mean"] = round(float(fin.max()), 4) if fin.size else None
                row[f"{mode}_auc_best"] = round(res["auc_best"], 4) if np.isfinite(res["auc_best"]) else None
                if s["planted"]:
                    order = np.argsort(-res["means"], kind="stable")
                    row[f"{mode}_true_rank"] = int(np.where(order == s["truth"] - 1)[0][0]) + 1
            slot_rows.append(row)
        for v, (m, _) in VARIANTS.items():
            sub = {"answers": answers[v]}
            g = grader.grade(d, sub)
            per_slot = 6 * 20 if m in ("sr6", "sr6cap") else 20 * 20
            unc = per_slot * len(specs)
            fwd = sum(kcap) if m == "sr6cap" else unc
            recs[v].append({"instance": inst["instance_id"], "tier": inst["tier"], "submission": sub, "grade": g,
                            "forward": fwd, "uncapped_forward": unc, "cap": cap,
                            "n_slots_truncated": sum(k < per_slot for k in kcap) if m == "sr6cap" else 0})
            for r in slot_rows[-len(specs):]:
                r[f"{v}_choice"] = answers[v][r["slot"]]["choice"]
                r[f"{v}_correct"] = g["details"]["slots"][r["slot"]]["correct"]

    tiers = sorted({r["tier"] for r in recs["SR-max"]})
    mets = {v: {"all": metrics(recs[v]), **{t: metrics([r for r in recs[v] if r["tier"] == t]) for t in tiers}}
            for v in VARIANTS}
    all20 = {v: mets[v]["all"]["planted_acc"] for v in ("SR-max-all20", "SR-thr-all20")}
    sens = selection_sensitivity(insts, acts, col, rows, styles, len(cids), n_alt=n_alt, all20=all20) \
        if n_alt else None
    prim = [o for o in (sens or {}).get("runs", []) if o["primary"]]
    if prim and (prim[0]["sr_max_planted_acc"] != mets["SR-max"]["all"]["planted_acc"]
                 or prim[0]["sr_thr_planted_acc"] != mets["SR-thr"]["all"]["planted_acc"]):
        raise AssertionError("sensitivity primary run disagrees with the graded primary variants")
    stop = stop_rule({v: mets[v]["all"] for v in PRIMARY}, sens, is_filtered_pool(inst_dir))
    summary = {"label": label or default_label(inst_dir), "instances_dir": os.path.abspath(inst_dir),
               "n_instances": len(insts), "pool_sha256": pool_digest(inst_dir), "git_sha": git_sha(),
               "bank": "R", "bank_texts_sha256": R.get("texts_sha256"), "bank_check": bcheck,
               "label_check": labels is not None, "cache_files": {f"L{k}": v for k, v in
                                                                                       finfo.items()},
               "rule": {"seed": SEED, "n_styles": N_STYLES, "chosen_styles": chosen,
                        "text_per_style": "first listed in the bank-R file", "thr": THR,
                        "tie_break": "lowest option number", "auroc": "reference_solver.auroc (ties 1/2), "
                        "claimed option's texts vs the other options' texts"},
               "capped_rule": "A2.3: texts consumed in slot order up to instance.json caps.forward; within a slot "
                              "style-round-robin over options (menu order)",
               "bootstrap": {"n": N_BOOT, "seed": SEED, "unit": "instance", "interval": "percentile 2.5/97.5"},
               "variants": {v: {"selection": m, "thresholded": t, "primary": v in PRIMARY}
                            for v, (m, t) in VARIANTS.items()},
               "stop_rule": stop,
               "metrics": mets,
               "text_selection_sensitivity": sens,
               "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(out_dir, "answers.jsonl"), "w") as f:
        for v in VARIANTS:
            for r in recs[v]:
                f.write(json.dumps({"variant": v, **{k: r[k] for k in ("instance", "tier", "submission", "grade",
                                                                       "forward", "cap")}}) + "\n")
    keys = list(slot_rows[0].keys())
    for r in slot_rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(out_dir, "slots.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(slot_rows)
    log(f"[sr_recipe] {summary['label']}")
    log(f"[sr_recipe] {len(insts)} instances from {inst_dir}; styles {chosen}")
    for v in VARIANTS:
        for t in ["all"] + tiers:
            m = summary["metrics"][v][t]
            log(f"  {v:13s} {t:3s} pass {m['pass']}/{m['n_instances']} = {m['pass_rate']} {m['pass_wilson95']}  "
                f"planted acc {m['planted_acc']} W{m['planted_acc_wilson95']} B{m['planted_acc_cluster_boot95']} "
                f"(n={m['n_planted']})  planted NF {m['planted_nothing_found_rate']}"
                f"  null FC {m['null_false_claim_rate']} (n={m['n_null']})  fwd mean {m['forward_units_mean']}"
                f" over-cap {m['n_over_forward_cap']} needs>cap {m['n_episodes_needing_more_than_cap']}"
                f" trunc slots {m['n_slots_truncated']}")
    if sens:
        log(f"  text-selection sensitivity ({sens['n_selections']} selections): {sens['planted_acc']}; "
            f"all20 {sens['all20_planted_acc']}; sensitive={sens['text_selection_sensitive']}")
    log(f"  stop_rule: {stop['labels']} (valid as step-3 outcome: {stop['valid_as_step3_outcome']})")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instances", default=DEFAULT_INSTANCES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--diag-cache", default=DIAG_CACHE)
    ap.add_argument("--label", default=None, help="text stored in summary.json (default derived from --instances)")
    ap.add_argument("--no-verify", action="store_true", help="skip sha256 of the cache files")
    ap.add_argument("--no-label-check", action="store_true", help="skip the option-label vs menu check")
    ap.add_argument("--n-alt", type=int, default=20,
                    help="extra style seeds for the A2.2 sensitivity (default 20 -> 21 seeds x 2 texts = 42; 0=off)")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.instances):
        raise SystemExit(f"instances dir {a.instances} does not exist (the filtered pool may not be written yet)")
    run(a.instances, a.out, a.diag_cache, label=a.label, verify=not a.no_verify, check_labels=not a.no_label_check,
        n_alt=a.n_alt)


if __name__ == "__main__":
    main(sys.argv[1:])
