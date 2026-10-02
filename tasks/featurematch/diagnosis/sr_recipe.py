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
            "SR-max-all20": ("all20", False), "SR-thr-all20": ("all20", True)}
PRIMARY = ("SR-max", "SR-thr")


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
    Returns dict: sr_max (1-based option), sr_thr (option or 'nothing found'), means, auc_best."""
    means = np.array([float(np.mean(np.asarray(v, dtype=np.float64))) for v in option_vals])
    best = int(np.argmax(means))                       # ties -> lowest option number
    neg = np.concatenate([np.asarray(v, dtype=np.float64) for j, v in enumerate(option_vals) if j != best])
    auc = auroc(np.asarray(option_vals[best], dtype=np.float64), neg)
    return {"sr_max": best + 1, "sr_thr": (best + 1) if auc >= thr else NOTHING, "means": means, "auc_best": auc}


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


def metrics(recs):
    """recs: list of {grade, forward, cap}. PREREG step-3 metrics."""
    n = len(recs)
    k_pass = sum(r["grade"]["pass"] for r in recs)
    pl = [s for r in recs for s in r["grade"]["details"]["slots"] if s["planted"]]
    nu = [s for r in recs for s in r["grade"]["details"]["slots"] if not s["planted"]]
    k_pl = sum(s["correct"] for s in pl)
    k_nf = sum(not s["claimed"] for s in pl)
    k_fc = sum(s["claimed"] for s in nu)
    fw = [r["forward"] for r in recs]
    return {"n_instances": n, "pass": k_pass, "pass_rate": rate(k_pass, n), "pass_wilson95": wilson(k_pass, n),
            "n_planted": len(pl), "planted_acc": rate(k_pl, len(pl)), "planted_acc_wilson95": wilson(k_pl, len(pl)),
            "planted_nothing_found_rate": rate(k_nf, len(pl)),
            "n_null": len(nu), "null_false_claim_rate": rate(k_fc, len(nu)),
            "mean_score": round(float(np.mean([r["grade"]["score"] for r in recs])), 4) if n else None,
            "forward_units_mean": round(float(np.mean(fw)), 1) if n else None,
            "forward_units_max": int(max(fw)) if n else None,
            "n_over_forward_cap": sum(r["forward"] > r["cap"] for r in recs)}


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


def default_label(inst_dir):
    if os.path.basename(os.path.normpath(inst_dir)) == "instances_v2f":
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


def selection_spread(insts, acts, col, rows, styles, n_c, n_alt=20, text_index=(0, 1)):
    """Descriptive only (not a pre-registered quantity): planted accuracy of SR-max / SR-thr and the null false-claim
    rate of SR-thr under alternative 6-text selections, to show how much the fixed choice matters. Alternatives:
    seeds SEED+1 .. SEED+n_alt for the styles, each with the first and the second text of every chosen style."""
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
            out.append({"seed": SEED + k, "text_index": ti, "styles": chosen, "sr_max_planted_acc": rate(pm, npl),
                        "sr_thr_planted_acc": rate(pt, npl), "sr_thr_null_false_claim": rate(nfc, nnu)})
    vals = lambda key: [o[key] for o in out]
    return {"note": "descriptive; alternatives to the fixed rule (seed 20261002, text_index 0 is the primary)",
            "summary": {key: {"min": min(vals(key)), "median": float(np.median(vals(key))), "max": max(vals(key))}
                        for key in ("sr_max_planted_acc", "sr_thr_planted_acc", "sr_thr_null_false_claim")},
            "runs": out}


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
        answers = {v: [] for v in VARIANTS}
        for s in specs:
            a = acts[s["layer"]][:, col[s["layer"]][s["latent"]]]
            row = {"instance": inst["instance_id"], "tier": inst["tier"], "slot": s["slot"],
                   "kind": "planted" if s["planted"] else "null", "layer": s["layer"], "latent": s["latent"],
                   "anchor": s["anchor"], "truth": s["truth"]}
            for mode in ("sr6", "all20"):
                res = decide([a[sel[mode][c]] for c in s["menu"]])
                for v, (m, thr) in VARIANTS.items():
                    if m == mode:
                        answers[v].append({"slot": s["slot"], "choice": res["sr_thr"] if thr else res["sr_max"]})
                row[f"{mode}_pick"] = res["sr_max"]
                row[f"{mode}_best_mean"] = round(float(res["means"].max()), 4)
                row[f"{mode}_auc_best"] = round(res["auc_best"], 4)
                if s["planted"]:
                    order = np.argsort(-res["means"], kind="stable")
                    row[f"{mode}_true_rank"] = int(np.where(order == s["truth"] - 1)[0][0]) + 1
            slot_rows.append(row)
        for v, (m, _) in VARIANTS.items():
            sub = {"answers": answers[v]}
            g = grader.grade(d, sub)
            per_slot = 6 * 20 if m == "sr6" else 20 * 20
            recs[v].append({"instance": inst["instance_id"], "tier": inst["tier"], "submission": sub, "grade": g,
                            "forward": per_slot * len(specs), "cap": int(inst["caps"]["forward"])})
            for r in slot_rows[-len(specs):]:
                r[f"{v}_choice"] = answers[v][r["slot"]]["choice"]
                r[f"{v}_correct"] = g["details"]["slots"][r["slot"]]["correct"]

    spread = selection_spread(insts, acts, col, rows, styles, len(cids), n_alt=n_alt) if n_alt else None
    tiers = sorted({r["tier"] for r in recs["SR-max"]})
    summary = {"label": label or default_label(inst_dir), "instances_dir": os.path.abspath(inst_dir),
               "n_instances": len(insts), "pool_sha256": pool_digest(inst_dir), "git_sha": git_sha(),
               "bank": "R", "bank_texts_sha256": R.get("texts_sha256"), "bank_check": bcheck,
               "label_check": labels is not None, "cache_files": {f"L{k}": v for k, v in
                                                                                       finfo.items()},
               "rule": {"seed": SEED, "n_styles": N_STYLES, "chosen_styles": chosen,
                        "text_per_style": "first listed in the bank-R file", "thr": THR,
                        "tie_break": "lowest option number", "auroc": "reference_solver.auroc (ties 1/2), "
                        "claimed option's texts vs the other options' texts"},
               "variants": {v: {"selection": m, "thresholded": t, "primary": v in PRIMARY}
                            for v, (m, t) in VARIANTS.items()},
               "metrics": {v: {"all": metrics(recs[v]), **{t: metrics([r for r in recs[v] if r["tier"] == t])
                                                           for t in tiers}} for v in VARIANTS},
               "selection_spread": spread,
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
                f"planted acc {m['planted_acc']} (n={m['n_planted']})  planted NF {m['planted_nothing_found_rate']}"
                f"  null FC {m['null_false_claim_rate']} (n={m['n_null']})  fwd mean {m['forward_units_mean']}"
                f" over-cap {m['n_over_forward_cap']}")
    if spread:
        log(f"  selection spread (descriptive, {len(spread['runs'])} selections): {spread['summary']}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instances", default=DEFAULT_INSTANCES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--diag-cache", default=DIAG_CACHE)
    ap.add_argument("--label", default=None, help="text stored in summary.json (default derived from --instances)")
    ap.add_argument("--no-verify", action="store_true", help="skip sha256 of the cache files")
    ap.add_argument("--no-label-check", action="store_true", help="skip the option-label vs menu check")
    ap.add_argument("--n-alt", type=int, default=20, help="alternative style seeds for the descriptive spread (0=off)")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.instances):
        raise SystemExit(f"instances dir {a.instances} does not exist (the filtered pool may not be written yet)")
    run(a.instances, a.out, a.diag_cache, label=a.label, verify=not a.no_verify, check_labels=not a.no_label_check,
        n_alt=a.n_alt)


if __name__ == "__main__":
    main(sys.argv[1:])
