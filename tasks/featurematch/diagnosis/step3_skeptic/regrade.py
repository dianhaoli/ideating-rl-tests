"""Step-3 skeptic, part 1: re-grade stored baseline submissions and recompute every pooled number from the RAW files.

Written independently of analyze_step3.py and sr_recipe.py's statistics (own Wilson, own clustered bootstrap, own
family split, own stop-rule logic). Imports only the task grader (the thing being checked against) and the stdlib.

Inputs (read-only):
  step3/runs/20261002-041021_inproc/episodes_private.jsonl   (13 solvers x 180 + reference retries)
  step3/sr_recipe_out/answers.jsonl                          (6 SR variants x 180)
  instances_v2f/*/instance.json, public.json                 (truth, anchors, tiers)
  instances_v2f_manifest.json                                (per-instance sha256)
Outputs (this directory): regrade_sample.json (the 30 sampled episodes and their CLI re-grades),
  recomputed.json (all metrics recomputed from the raw files, plus the stop-rule block and diffs vs the builder).

Sample: a list of every stored episode sorted by (source, solver, instance_id, seed); three strata (SR rows,
reference rows, other in-process rows), drawn in that order with ONE random.Random(20261002): 10 + 10 + 10.
Each sampled submission is written to a temp file and graded by `python tasks/featurematch/grader.py` (the CLI, a
separate process), then compared slot by slot with what the builder stored.
"""
import glob
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
TASK = os.path.dirname(DIAG)
ROOT = os.path.dirname(os.path.dirname(TASK))
INST = os.path.join(TASK, "instances_v2f")
EPIS = os.path.join(DIAG, "step3", "runs", "20261002-041021_inproc", "episodes_private.jsonl")
SRANS = os.path.join(DIAG, "step3", "sr_recipe_out", "answers.jsonl")
MANIFEST = os.path.join(DIAG, "instances_v2f_manifest.json")
SEED = 20261002
PY = os.environ.get("PY", sys.executable)


# ------------------------------------------------------------------------------------------------ own statistics
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def cluster_boot(k_inst, n_inst, b=10000, seed=SEED):
    """Instance-clustered percentile bootstrap of sum(k)/sum(n); own implementation (numpy RNG differs from the
    builder's random.Random draw, so CIs may differ in the 3rd decimal; the builder's indices are reproduced in
    `cluster_boot_builder_rng` below for an exact comparison)."""
    k, n = np.asarray(k_inst, float), np.asarray(n_inst, float)
    if n.sum() == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(k), size=(b, len(k)))
    K, N = k[idx].sum(1), n[idx].sum(1)
    r = K[N > 0] / N[N > 0]
    lo, hi = np.percentile(r, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


_IDX = {}


def cluster_boot_builder_rng(k_inst, n_inst, b=10000, seed=SEED):
    """Same resampling scheme as PREREG A2.5 states it (random.Random(20261002).choices per CI)."""
    k, n = np.asarray(k_inst, float), np.asarray(n_inst, float)
    if n.sum() == 0:
        return [None, None]
    key = len(k)
    if key not in _IDX:
        rng = random.Random(seed)
        _IDX[key] = np.array([rng.choices(range(len(k)), k=len(k)) for _ in range(b)], dtype=np.int32)
    idx = _IDX[key]
    K, N = k[idx].sum(1), n[idx].sum(1)
    r = K[N > 0] / N[N > 0]
    lo, hi = np.percentile(r, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


# ------------------------------------------------------------------------------------------------ truth
def load_truth():
    T = {}
    for d in sorted(glob.glob(os.path.join(INST, "*"))):
        inst = json.load(open(os.path.join(d, "instance.json")))
        fam = [("language" if s["anchor"].startswith("lang:") else "topic") for s in inst["extra"]["slots"]]
        T[inst["instance_id"]] = {"dir": d, "tier": inst["tier"], "truth": inst["answer"]["slots"], "family": fam}
    return T


def my_grade(truth, submission):
    """Independent per-slot grading straight from the rule text (planted: claimed option == key; null: 'nothing
    found'). Kept deliberately simple; differences from grader.py would be reported."""
    got = {}
    for a in (submission or {}).get("answers", []):
        c = a.get("choice")
        if isinstance(c, str) and c.strip().lower() == "nothing found":
            c = "NF"
        elif isinstance(c, bool):
            c = None
        elif isinstance(c, str):
            try:
                c = int(c)
            except ValueError:
                c = None
        got[int(a["slot"])] = c
    out = []
    for i, t in enumerate(truth):
        c = got.get(i)
        claimed = c is not None and c != "NF"
        correct = (claimed and c == t["choice"]) if t["planted"] else (c == "NF")
        out.append({"planted": t["planted"], "correct": bool(correct), "claimed": bool(claimed)})
    return out


def pool_checks():
    man = json.load(open(MANIFEST))
    mism, n = [], 0
    ids = set()
    for e in man["instances"]:
        ids.add(e["instance_id"])
        for fn, h in e["files"].items():
            h = h if isinstance(h, str) else h.get("sha256")
            p = os.path.join(INST, e["instance_id"], fn)
            n += 1
            got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
            if got != h:
                mism.append(f"{e['instance_id']}/{fn}")
    on_disk = {os.path.basename(d) for d in glob.glob(os.path.join(INST, "*")) if os.path.isdir(d)}
    mism += [f"extra dir {x}" for x in sorted(on_disk - ids)] + [f"missing dir {x}" for x in sorted(ids - on_disk)]
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(INST, "*", "instance.json"))):
        h.update(os.path.basename(os.path.dirname(p)).encode())
        h.update(open(p, "rb").read())
    return {"manifest_file_sha256": hashlib.sha256(open(MANIFEST, "rb").read()).hexdigest(),
            "manifest_entries_checked": n, "manifest_mismatches": mism,
            "n_instance_dirs": len(glob.glob(os.path.join(INST, "*", "instance.json"))),
            "pool_sha256_recomputed": h.hexdigest()}


# ------------------------------------------------------------------------------------------------ metrics
def metrics(rows, T, family=None):
    """rows: list of {instance_id, slots(graded)}. family None = pooled; else restricted to slots of that family
    (episode enters if it has >= 1 such slot; family pass = all those slots correct; the builder's definition)."""
    n_ep = k_pass = 0
    kp = npl = knf = kfc = nnu = 0
    k_inst, n_inst = [], []
    for r in rows:
        fam = T[r["instance_id"]]["family"]
        sl = [s for s, f in zip(r["slots"], fam) if family is None or f == family]
        if not sl:
            continue
        n_ep += 1
        k_pass += all(s["correct"] for s in sl)
        p = [s for s in sl if s["planted"]]
        u = [s for s in sl if not s["planted"]]
        kp += sum(s["correct"] for s in p)
        npl += len(p)
        knf += sum(not s["claimed"] for s in p)
        kfc += sum(s["claimed"] for s in u)
        nnu += len(u)
        k_inst.append(sum(s["correct"] for s in p))
        n_inst.append(len(p))
    return {"n_episodes": n_ep, "pass": k_pass, "pass_rate": round(k_pass / n_ep, 4) if n_ep else None,
            "pass_wilson95": wilson(k_pass, n_ep), "n_planted": npl, "planted_correct": kp,
            "planted_acc": round(kp / npl, 4) if npl else None, "planted_acc_wilson95": wilson(kp, npl),
            "planted_acc_cluster95_own_rng": cluster_boot(k_inst, n_inst),
            "planted_acc_cluster95_A2.5_rng": cluster_boot_builder_rng(k_inst, n_inst),
            "planted_nf_rate": round(knf / npl, 4) if npl else None, "n_null": nnu,
            "null_false_claim": round(kfc / nnu, 4) if nnu else None}


def main():
    sys.path.insert(0, ROOT)
    from tasks.featurematch import grader
    T = load_truth()
    out = {"pool": pool_checks()}

    # ---------------- load raw episodes
    ep = [json.loads(l) for l in open(EPIS)]
    sr = [json.loads(l) for l in open(SRANS)]
    allrows = []
    for r in ep:
        allrows.append({"source": "inproc", "solver": r["solver"], "instance_id": r["instance_id"],
                        "seed": r["seed"], "submission": r.get("submission"), "stored_slots": r["slots"],
                        "stored_pass": r["pass"], "stored_score": r["score"]})
    for r in sr:
        g = r["grade"]
        allrows.append({"source": "sr", "solver": r["variant"], "instance_id": r["instance"], "seed": None,
                        "submission": r["submission"], "stored_slots": g["details"]["slots"],
                        "stored_pass": g["pass"], "stored_score": g["score"]})
    allrows.sort(key=lambda r: (r["source"], r["solver"], r["instance_id"], -1 if r["seed"] is None else r["seed"]))

    # ---------------- full in-process regrade (function) + independent rule
    n_diff_fn = n_diff_mine = 0
    diffs = []
    for r in allrows:
        g = grader.grade(T[r["instance_id"]]["dir"], r["submission"])
        mine = my_grade(T[r["instance_id"]]["truth"], r["submission"])
        r["slots"] = g["details"]["slots"]
        same_fn = (g["details"]["slots"] == r["stored_slots"] and g["pass"] == r["stored_pass"]
                   and abs(g["score"] - r["stored_score"]) < 1e-12)
        same_mine = mine == g["details"]["slots"]
        n_diff_fn += not same_fn
        n_diff_mine += not same_mine
        if not (same_fn and same_mine) and len(diffs) < 20:
            diffs.append({k: r[k] for k in ("source", "solver", "instance_id", "seed")})
    out["full_regrade"] = {"n_episodes": len(allrows), "grader_fn_vs_stored_differences": n_diff_fn,
                           "independent_rule_vs_grader_differences": n_diff_mine, "examples": diffs}

    # ---------------- seeded sample, CLI grader
    rng = random.Random(SEED)
    strata = [("SR", [r for r in allrows if r["source"] == "sr"]),
              ("reference", [r for r in allrows if r["source"] == "inproc" and r["solver"] == "reference"]),
              ("other_inproc", [r for r in allrows if r["source"] == "inproc" and r["solver"] != "reference"])]
    sample = []
    for name, rows in strata:
        for r in rng.sample(rows, 10):
            with tempfile.TemporaryDirectory() as td:
                sp, gp = os.path.join(td, "sub.json"), os.path.join(td, "grade.json")
                json.dump(r["submission"], open(sp, "w"))
                cp = subprocess.run([PY, os.path.join(TASK, "grader.py"), "--instance-dir", T[r["instance_id"]]["dir"],
                                     "--submission", sp, "--out", gp], capture_output=True, text=True, cwd=ROOT)
                g = json.load(open(gp))
            sample.append({"stratum": name, "source": r["source"], "solver": r["solver"],
                           "instance_id": r["instance_id"], "seed": r["seed"],
                           "cli_pass": g["pass"], "cli_score": round(g["score"], 6),
                           "stored_pass": r["stored_pass"], "stored_score": round(r["stored_score"], 6),
                           "slots_match": g["details"]["slots"] == r["stored_slots"],
                           "n_slots": len(g["details"]["slots"]),
                           "cli_rc": cp.returncode})
    out["sample"] = {"rule": "sorted by (source, solver, instance_id, seed); random.Random(20261002).sample 10 per "
                             "stratum in the order SR, reference, other in-process",
                     "n": len(sample), "n_match": sum(s["slots_match"] and s["cli_pass"] == s["stored_pass"]
                                                       and abs(s["cli_score"] - s["stored_score"]) < 1e-9
                                                       and s["cli_rc"] == 0 for s in sample),
                     "rows": sample}

    # ---------------- recompute metrics from raw (graded by grader.grade above)
    groups = {}
    for r in allrows:
        if r["source"] == "inproc" and r["seed"] != 7:
            continue                            # reference retries: one-shot = seed 7
        groups.setdefault(r["solver"], []).append(r)
    # best-of-5 reference: first passing seed, else last
    by_i = {}
    for r in allrows:
        if r["source"] == "inproc" and r["solver"] == "reference":
            by_i.setdefault(r["instance_id"], []).append(r)
    bo5 = []
    tries = {}
    for iid, rs in by_i.items():
        rs.sort(key=lambda r: r["seed"])
        pick = next((r for r in rs if r["stored_pass"]), rs[-1])
        bo5.append(pick)
        tries[len(rs)] = tries.get(len(rs), 0) + 1
    groups["reference_best_of_5"] = bo5
    M = {}
    for name, rows in groups.items():
        assert len({r["instance_id"] for r in rows}) == len(rows) == 180, (name, len(rows))
        M[name] = {"pooled": metrics(rows, T), "topic": metrics(rows, T, "topic"),
                   "language": metrics(rows, T, "language")}
        for tier in ("T1", "T2", "T3"):
            M[name][tier] = metrics([r for r in rows if T[r["instance_id"]]["tier"] == tier], T)
    out["metrics"] = M
    out["reference_tries_per_instance"] = tries

    # ---------------- stop rule, from raw answers
    def stop(fam):
        per = {}
        for v in ("SR-max", "SR-thr"):
            m = M[v][fam]
            lo, hi = m["planted_acc_wilson95"]
            per[v] = {"planted_acc": m["planted_acc"], "k": m["planted_correct"], "n": m["n_planted"],
                      "wilson95": [lo, hi], "above_0.50": m["planted_acc"] > 0.5,
                      "wilson_includes_0.50": lo <= 0.5 <= hi}
        firing = [v for v in per if per[v]["above_0.50"]]
        fires = bool(firing) and all(per[v]["n"] >= 100 for v in per)
        return {"per_variant": per, "fires": fires,
                "borderline": fires and all(per[v]["wilson_includes_0.50"] for v in firing)}
    out["stop_rule_recomputed"] = {"pooled": stop("pooled"), "topic_only": stop("topic"),
                                   "language_only": stop("language")}
    pa_ref = M["reference"]["pooled"]["planted_acc"]
    out["A1.5_ratios"] = {v: round(M[v]["pooled"]["planted_acc"] / pa_ref, 4) for v in ("SR-max", "SR-thr")}
    out["criterion3_inputs"] = {"r_ref": M["reference"]["pooled"]["planted_nf_rate"],
                                "r_rec": round((M["recipe_self_probe"]["pooled"]["planted_nf_rate"]
                                                + M["recipe_template_probe"]["pooled"]["planted_nf_rate"]) / 2, 4)}
    gate = {}
    for name in groups:
        if name.startswith("SR") or name.startswith("reference"):
            continue
        pa = M[name]["pooled"]["planted_acc"]
        gate[name] = {"planted_acc": pa, "wilson95": M[name]["pooled"]["planted_acc_wilson95"],
                      "passes_gate_le_0.15": pa <= 0.15}
    out["gate_recomputed"] = gate

    # ---------------- compare with builder's public json
    pub = json.load(open(os.path.join(DIAG, "step3", "baselines_public.json")))
    out["builder_public_json_keys"] = list(pub.keys())[:30]
    # per-episode rows are private (gitignored, like step3's per-episode files); the committed file has counts only
    json.dump(out["sample"], open(os.path.join(HERE, "regrade_sample_private.json"), "w"), indent=1)
    from collections import Counter
    pub = dict(out)
    pub["sample"] = {k: v for k, v in out["sample"].items() if k != "rows"}
    pub["sample"]["strata"] = dict(Counter(f"{r['stratum']}:{r['solver']}" for r in out["sample"]["rows"]))
    json.dump(pub, open(os.path.join(HERE, "recomputed.json"), "w"), indent=1)
    # concise print
    print(json.dumps(out["pool"], indent=1))
    print(json.dumps(out["full_regrade"], indent=1))
    print("sample:", out["sample"]["n"], "match", out["sample"]["n_match"])
    for name in sorted(M):
        m = M[name]["pooled"]
        print(f"{name:28s} pass {m['pass']}/{m['n_episodes']}={m['pass_rate']} {m['pass_wilson95']}  "
              f"pa {m['planted_correct']}/{m['n_planted']}={m['planted_acc']} W{m['planted_acc_wilson95']} "
              f"C{m['planted_acc_cluster95_A2.5_rng']} Cown{m['planted_acc_cluster95_own_rng']} "
              f"NF {m['planted_nf_rate']} FC {m['null_false_claim']} | topic {M[name]['topic']['planted_acc']} "
              f"lang {M[name]['language']['planted_acc']}")
    print(json.dumps(out["stop_rule_recomputed"], indent=1))
    print(out["A1.5_ratios"], out["criterion3_inputs"], out["reference_tries_per_instance"])


if __name__ == "__main__":
    main()
