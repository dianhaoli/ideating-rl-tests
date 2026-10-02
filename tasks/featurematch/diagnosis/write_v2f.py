"""Write the style-filtered FeatureMatch pool ("v2f") as new instance dirs (CPU; run after style_filter analyze).

Input: style_filter_out/slots.json (per-slot verdicts), diagnosis/cache/pool_auroc.npz (AUROC rows of every pooled
latent on A+F1 / C+F2 / F2), the v2 instances (read-only) and the v2 generator tables (read-only src cache).

Per v2 instance, slot by slot:
  kept slot     copied unchanged (layer, real latent, menu and its order, answer).
  failing slot  --mode redraw (default): replaced by a NEW slot of the SAME kind (planted/null) at the same position,
                drawn by generator v2 itself (generate.make_slot: layer, then anchor concept uniformly, then one of its
                pooled latents, then the menu, with the instance's tier closeness) and accepted only if it passes the
                same slot filter (style_filter.judge, same parameters as the analysis). Rejected draws are redrawn.
                WHY this way: a kept original slot is "a v2 draw that passed the filter"; a redrawn slot is "v2 draws
                until one passes", which is the same distribution. So kept and redrawn slots cannot be told apart,
                and planted and null slots still come from one latent pool (SPEC no-fingerprint). Keeping the kind
                and position keeps every episode's slot count and null pattern, so the pool's slot-count distribution
                and null fraction are exactly those of v2.
                --mode drop: the failing slot is removed; instances left with < --min-slots slots are dropped. This
                changes the slot-count distribution and (since nulls fail more often) the null fraction; it exists
                only as a comparison.
Every written instance (modified or not) gets a fresh permutation seed, so a fresh instance id, canary and leak
string, computed exactly as generate.make_instance does. Ids therefore reveal neither the source instance nor
whether any slot was redrawn. The source id and the redrawn slots are recorded only in the privileged instance.json
(extra.style_filter) and in the manifest.

Run: $PY -m tasks.featurematch.diagnosis.write_v2f [--mode redraw|drop] [--clean]
Out: tasks/featurematch/instances_v2f/<id>/{instance.json,public.json} (gitignored),
     diagnosis/instances_v2f_manifest.json, diagnosis/style_filter_out/v2f_summary.json
"""
import argparse
import hashlib
import json
import os
import shutil
from collections import Counter

import numpy as np

from tasks.featurematch.diagnosis import style_filter as SF
from tasks.featurematch.fm_core import LAYERS

OUT_INST = os.path.join(SF.TASK, "instances_v2f")
MANIFEST = os.path.join(SF.HERE, "instances_v2f_manifest.json")
FILTER_VERSION = "v2f-1"
SALT = 20261002


class PoolAuroc:
    def __init__(self, path):
        z = np.load(path)
        self.rows = {}
        for L in LAYERS:
            if f"L{L}_latents" not in z:
                continue
            for i, j in enumerate(z[f"L{L}_latents"]):
                self.rows[(L, int(j))] = (z[f"L{L}_AF1"][i], z[f"L{L}_CF2"][i], z[f"L{L}_F2"][i])

    def get(self, L, j):
        return self.rows[(int(L), int(j))]


def redraw_slot(T, PA, cid2i, src_inst, pos, kind, used, params, max_attempts):
    """Generator v2 draw of a slot of `kind` at position `pos`, repeated until it passes the filter."""
    from tasks.featurematch.generate import TIERS, make_slot
    rng = np.random.default_rng([int(src_inst["seed"]), 7, SALT, pos])
    closeness = TIERS[src_inst["tier"]]["closeness"]
    for attempt in range(1, max_attempts + 1):
        s = make_slot(T, rng, closeness, kind != "planted", set(used))
        aAF1, aCF2, aF2 = PA.get(s["layer"], s["real_latent"])
        anchor = cid2i[s["anchor"]]
        v = SF.judge("planted" if kind == "planted" else "null", anchor, s["menu"], aAF1, aCF2, aF2,
                     params["thr"], params["robust_set"], params["rival_rule"], params["null_rule"])
        if v["keep"]:
            return s, attempt, v
    raise RuntimeError(f"no passing {kind} slot after {max_attempts} draws ({src_inst['instance_id']} slot {pos})")


def regen_sources(tiers, n_per_tier, start_seed):
    """Regenerate mode (PREREG step 2 as written): fresh instances with generator v2's own episode draw (slot count
    and null pattern from default_rng([seed, 7]) exactly as make_instance), every slot drawn by redraw_slot.
    Seeds start_seed + 100000 * tier_index + k, tier_index over sorted(TIERS) as in generate.main."""
    from tasks.featurematch.generate import P_NULL, TIERS
    from tasks.featurematch import generate as G
    th = {"P_THR": G.P_THR, "P_THR_B": G.P_THR_B, "N_THR": G.N_THR, "MARGIN": G.MARGIN, "FIRE_MIN": G.FIRE_MIN,
          "MARGIN_B": G.MARGIN_B, "N_THR_B": G.N_THR_B}
    out = []
    for tier in tiers:
        ti = sorted(TIERS).index(tier)
        cfg = TIERS[tier]
        for k in range(n_per_tier):
            seed = start_seed + 100000 * ti + k
            rng = np.random.default_rng([seed, 7])
            n_slots = int(rng.choice([3, 4, 5]))
            nulls = rng.random(n_slots) < P_NULL
            out.append({"instance_id": f"regen-{tier}-{seed}", "seed": seed, "tier": tier,
                        "dial": {"closeness": cfg["closeness"], "n_slots": n_slots,
                                 "forward_cap": cfg["caps"]["forward"]},
                        "caps": dict(cfg["caps"]),
                        "extra": {"slots": [{"kind": "near_miss" if x else "planted"} for x in nulls],
                                  "thresholds": th}})
    return out


def build_instance(T, src, slots, info):
    """Same layout as generate.make_instance; slots carry menu as concept INDICES here."""
    from tasks.featurematch.generate import GEN_VERSION, render_slots
    from tasks.featurematch.tools import perm_for
    seed, tier = int(src["seed"]), src["tier"]
    perm_seed = int(np.random.default_rng([seed, 7, SALT, 99]).integers(2**62))
    inv = {L: np.argsort(perm_for(perm_seed, L)) for L in LAYERS}
    suffix = hashlib.sha256(f"fm-v2f-{seed}-{tier}-{perm_seed}".encode()).hexdigest()[:10]
    iid = f"fm-{tier.lower()}-{suffix}"
    canary = "RLCANARY-featurematch-" + hashlib.sha256(f"canary-{iid}".encode()).hexdigest()[:16]
    labels = [c["label"] for c in T.cs]
    n = len(slots)
    public = {
        "model": "google/gemma-2-2b",
        "sae": "Gemma Scope residual-stream SAEs (google/gemma-scope-2b-pt-res), width 16384, layers 6, 12 and 18",
        "n_slots": n,
        "slots": [{"slot": i, "layer": s["layer"], "latent": int(inv[s["layer"]][s["real_latent"]]),
                   "options": [labels[c] for c in s["menu"]]} for i, s in enumerate(slots)],
    }
    public["slots_text"] = render_slots(public)
    dial = dict(src["dial"])
    dial["n_slots"] = n
    inst = {
        "instance_id": iid, "task": "featurematch", "tier": tier, "dial": dial, "seed": seed, "canary": canary,
        "planted": bool(any(s["kind"] == "planted" for s in slots)),
        "answer": {"slots": [{"planted": s["kind"] == "planted", "choice": s["choice"],
                              "concept": s["anchor"] if s["kind"] == "planted" else None} for s in slots]},
        "leak_strings": [f"{perm_seed:x}"],
        "caps": dict(src["caps"]),
        "extra": {"perm_seed": perm_seed,
                  "slots": [{k: (s[k] if k != "menu" else [T.cs[c]["cid"] for c in s["menu"]]) for k in s
                             if k != "_filter"} for s in slots],
                  "thresholds": dict(src["extra"]["thresholds"]),
                  "gen_version": GEN_VERSION,
                  "style_filter": dict(info, slots=[s.get("_filter") for s in slots])},
    }
    return iid, inst, public


def verify(T, d, inst, public):
    """Re-derive what the tool server and grader read and check it is consistent."""
    from tasks.featurematch.tools import perm_for
    assert inst["canary"] == "RLCANARY-featurematch-" + hashlib.sha256(
        f"canary-{inst['instance_id']}".encode()).hexdigest()[:16]
    assert os.path.basename(d) == inst["instance_id"]
    for ps, s, a in zip(public["slots"], inst["extra"]["slots"], inst["answer"]["slots"]):
        assert perm_for(inst["extra"]["perm_seed"], ps["layer"])[ps["latent"]] == s["real_latent"]
        assert ps["layer"] == s["layer"] and len(ps["options"]) == len(s["menu"]) == 20
        if s["kind"] == "planted":
            assert a["planted"] and a["concept"] == s["anchor"]
            lab = next(c["label"] for c in T.cs if c["cid"] == s["anchor"])
            assert ps["options"][a["choice"] - 1] == lab
        else:
            assert not a["planted"] and a["choice"] == "nothing found" and s["anchor"] not in s["menu"]
    assert public["n_slots"] == len(public["slots"]) == len(inst["answer"]["slots"])
    pub_txt = json.dumps(public)
    assert inst["canary"] not in pub_txt and not any(x in pub_txt for x in inst["leak_strings"])


def density_auroc(slots_by_kind, rate):
    """Single-feature fingerprint check: AUROC of 'generic fire rate' for predicting null vs planted (0.5 = none)."""
    from tasks.featurematch.reference_solver import auroc
    x = {k: [float(rate[L][j]) for L, j in v] for k, v in slots_by_kind.items()}
    if not x.get("null") or not x.get("planted"):
        return None
    return round(auroc(x["null"], x["planted"]), 4)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=SF.OUT, help="style_filter analyze output dir (slots.json)")
    ap.add_argument("--pool-auroc", default=os.path.join(SF.DIAG_CACHE, "pool_auroc.npz"))
    ap.add_argument("--src-cache", default=None)
    ap.add_argument("--src-instances", default=None)
    ap.add_argument("--out-instances", default=OUT_INST)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--mode", choices=("redraw", "drop", "regenerate"), default="redraw")
    ap.add_argument("--n-per-tier", type=int, default=60, help="regenerate mode only")
    ap.add_argument("--start-seed", type=int, default=8000, help="regenerate mode only (PREREG: 8000/108000/208000)")
    ap.add_argument("--tiers", default="T1,T2,T3", help="regenerate mode only")
    ap.add_argument("--min-slots", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=5000)
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args(argv)
    src_cache = a.src_cache or SF.default_src_cache()
    src_instances = a.src_instances or SF.default_src_instances()
    sj = json.load(open(os.path.join(a.analysis, "slots.json")))
    params = sj["params"]
    verdict = {(r["instance_id"], r["slot"]): r for r in sj["slots"]}
    T = SF.load_tables(src_cache)
    PA = PoolAuroc(a.pool_auroc)
    cid2i = {c["cid"]: i for i, c in enumerate(T.cs)}
    if a.clean and os.path.exists(a.out_instances):
        shutil.rmtree(a.out_instances)
    os.makedirs(a.out_instances, exist_ok=True)
    rate = {L: T.rate[L] for L in LAYERS}
    man, stats = [], Counter()
    attempts, src_counts, new_counts = [], Counter(), Counter()
    src_kinds, new_kinds = {"planted": [], "null": []}, {"planted": [], "null": []}
    redrawn_kinds = {"planted": [], "null": []}
    src_kind_n = Counter()
    if a.mode == "regenerate":
        sources = regen_sources(a.tiers.split(","), a.n_per_tier, a.start_seed)
        for src in sources:
            for i, s in enumerate(src["extra"]["slots"]):
                verdict[(src["instance_id"], i)] = {"keep": False, "reasons": "regenerated"}
    else:
        sources = [src for _, src in SF.iter_instances(src_instances)]
    for src in sources:
        sid = src["instance_id"]
        sslots = src["extra"]["slots"]
        src_counts[len(sslots)] += 1
        used = {(int(s["layer"]), int(s["real_latent"])) for i, s in enumerate(sslots) if verdict[(sid, i)]["keep"]}
        out, redrawn, dropped = [], [], []
        for i, s in enumerate(sslots):
            kind = "planted" if s["kind"] == "planted" else "null"
            src_kind_n[kind] += 1
            if "real_latent" in s:
                src_kinds[kind].append((int(s["layer"]), int(s["real_latent"])))
            v = verdict[(sid, i)]
            if v["keep"]:
                ns = dict(s, menu=[cid2i[c] for c in s["menu"]])
                ns["_filter"] = {"source_slot": i, "redrawn": False, "robust_metric": v["robust_metric"],
                                 "ms_key": v["ms_key"]}
                out.append(ns)
                continue
            stats[f"failed_{kind}"] += 1
            if a.mode == "drop":
                dropped.append({"slot": i, "kind": kind, "reasons": v["reasons"]})
                continue
            ns, n_att, jv = redraw_slot(T, PA, cid2i, src, i, kind, used, params, a.max_attempts)
            used.add((ns["layer"], ns["real_latent"]))
            attempts.append(n_att)
            ns["_filter"] = {"source_slot": i, "redrawn": True, "replaced_reasons": v["reasons"],
                             "robust_metric": round(jv["robust_metric"], 4), "ms_key": T.cs[jv["ms_key"]]["cid"],
                             "draws": n_att}
            redrawn.append({"slot": i, "kind": kind, "reasons": v["reasons"], "draws": n_att})
            redrawn_kinds[kind].append((ns["layer"], ns["real_latent"]))
            out.append(ns)
        if a.mode == "drop" and len(out) < a.min_slots:
            stats["instances_dropped"] += 1
            continue
        info = {"version": FILTER_VERSION, "mode": a.mode, "source_instance": sid, "redrawn": redrawn,
                "dropped": dropped, "params": {k: params[k] for k in ("thr", "robust_set", "rival_rule", "null_rule")}}
        iid, inst, public = build_instance(T, src, out, info)
        od = os.path.join(a.out_instances, iid)
        if os.path.exists(od):
            raise RuntimeError(f"instance id collision: {iid}")
        os.makedirs(od)
        json.dump(inst, open(os.path.join(od, "instance.json"), "w"), indent=1)
        json.dump(public, open(os.path.join(od, "public.json"), "w"), indent=1)
        verify(T, od, inst, public)
        new_counts[len(out)] += 1
        for s in out:
            new_kinds["planted" if s["kind"] == "planted" else "null"].append((s["layer"], s["real_latent"]))
        man.append({"instance_id": iid, "tier": inst["tier"], "seed": inst["seed"], "dial": inst["dial"],
                    "source_instance": sid, "redrawn_slots": [r["slot"] for r in redrawn],
                    "dropped_slots": [r["slot"] for r in dropped],
                    "files": {f: SF.sha256_file(os.path.join(od, f)) for f in ("instance.json", "public.json")}})
    cv = os.path.join(src_cache, "concepts_validation.json")
    summ = {
        "mode": a.mode, "filter_params": params, "n_source_instances": sum(src_counts.values()),
        "n_written": len(man), "instances_untouched": sum(1 for m in man if not m["redrawn_slots"]
                                                         and not m["dropped_slots"]),
        "slots_failed": dict(stats), "redraw_draws": SF.describe(attempts) if attempts else None,
        "slot_count_dist": {"source": dict(sorted(src_counts.items())), "v2f": dict(sorted(new_counts.items()))},
        "null_fraction": {"source": SF._f(src_kind_n["null"] / max(1, sum(src_kind_n.values()))),
                          "v2f": SF._f(len(new_kinds["null"]) / max(1, sum(map(len, new_kinds.values()))))},
        "all_null_episodes": sum(1 for m in man
                                 if not json.load(open(os.path.join(a.out_instances, m["instance_id"],
                                                                    "instance.json")))["planted"]),
        "density_auroc_null_vs_planted": {"source": density_auroc(src_kinds, rate),
                                          "v2f": density_auroc(new_kinds, rate)},
        "redrawn_by_kind": {k: len(v) for k, v in redrawn_kinds.items()},
        "by_tier": {t: sum(1 for m in man if m["tier"] == t) for t in sorted({m["tier"] for m in man})},
    }
    json.dump({"generator": "tasks/featurematch/diagnosis/write_v2f.py (generator v2 + style filter)",
               "filter_version": FILTER_VERSION, "gen_version": 2, "mode": a.mode,
               "source_instances": src_instances, "filter_params": params,
               "concepts_sha256": json.load(open(cv))["sha256"] if os.path.exists(cv) else None,
               "instances": man}, open(a.manifest, "w"), indent=1)
    json.dump(summ, open(os.path.join(a.analysis, f"v2f_summary{'' if a.mode == 'redraw' else '_drop'}.json"), "w"),
              indent=1)
    print(json.dumps(summ, indent=1))
    return summ


if __name__ == "__main__":
    main()
