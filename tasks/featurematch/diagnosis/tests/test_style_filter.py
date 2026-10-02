"""CPU end-to-end test of the step-2 style filter on a synthetic world (tests/synth_world.py).

Covers: compute (batching, IO, normalisation, manifest, dry-run subset) with a stub model; analyze (per-slot and
per-latent verdicts against known latent types, integrity checks, outputs, offline reference); write_v2f (redraw and
drop modes: slot counts, null pattern, ids, canaries, permutation, grader).

    cd ~/wt/fmdiag && /opt/pytorch/bin/python -m pytest -q tasks/featurematch/diagnosis/tests/test_style_filter.py
"""
import csv
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tasks.featurematch import auroc as AU
from tasks.featurematch import concepts as C
from tasks.featurematch.diagnosis import style_filter as SF
from tasks.featurematch.diagnosis import write_v2f as WV
from tasks.featurematch.diagnosis.tests.synth_world import LAYERS, STYLES_F, StubSubject, World

N_PER_TIER = 8


@pytest.fixture(scope="module")
def env():
    """~170 MB of synthetic float16 arrays (real width 16384, real 3 layers, 24 concepts), deleted afterwards."""
    from tasks.featurematch import reference_solver as RS
    old = (C.CACHE, C.OUT, RS.BUDGET_PROFILES)
    # the synthetic world has 24 concepts, so the offline reference can draw only 4 background concepts per slot
    RS.BUDGET_PROFILES = [(0, dict(k1=3, k2=5, top=2, bg1=2, bg2=2))]
    root = Path(tempfile.mkdtemp(prefix="sf_test_"))
    w = World()
    src, banks, inst, diag, out = (str(root / x) for x in ("cache", "banks", "instances", "diag", "out"))
    w.write_src_cache(src)
    C.CACHE = src
    AU.main()                                  # the generator's own AUROC / fire tables
    w.write_banks(banks)
    T = SF.load_tables(src)
    from tasks.featurematch.generate import make_instance
    for ti, tier in enumerate(("T1", "T2", "T3")):
        for k in range(N_PER_TIER):
            iid, i_, pub = make_instance(T, 5000 + 100000 * ti + k, tier)
            os.makedirs(os.path.join(inst, iid))
            json.dump(i_, open(os.path.join(inst, iid, "instance.json"), "w"))
            json.dump(pub, open(os.path.join(inst, iid, "public.json"), "w"))
    stub = StubSubject(w)
    man = SF.compute(stub, src_cache=src, banks_dir=banks, out_dir=diag, consistency_n=32, log=lambda s: None)
    summary = SF.analyze(src, inst, diag, out, ref_sims=1, log=lambda s: None)
    yield {"w": w, "T": T, "src": src, "banks": banks, "inst": inst, "diag": diag, "out": out, "stub": stub,
           "man": man, "summary": summary, "root": root}
    C.CACHE, C.OUT, RS.BUDGET_PROFILES = old
    shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ compute
def test_compute_outputs_and_manifest(env):
    w, diag, man = env["w"], env["diag"], env["man"]
    n = w.n_c * 20
    for L in LAYERS:
        for b in "FR":
            a = np.load(os.path.join(diag, f"bank_acts_L{L}_{b}.npy"))
            assert a.shape == (n, 16384) and a.dtype == np.float16
            ent = man["files"][f"bank_acts_L{L}_{b}.npy"]
            assert ent["shape"] == [n, 16384] and ent["sha256"] == SF.sha256_file(os.path.join(diag, f"bank_acts_L{L}_{b}.npy"))
    assert max(env["stub"].batches) <= SF.BATCH
    assert sum(env["stub"].batches) == 2 * n + 32           # both banks + the consistency batch
    bm = json.load(open(os.path.join(diag, "bank_meta.json")))
    assert bm["cids"] == [c["cid"] for c in w.cs]
    # rows in generator concept order, texts whitespace-normalised before the model sees them
    assert bm["banks"]["F"]["rows"][:20] == [0] * 20 and bm["banks"]["F"]["styles"][:2] == [STYLES_F[0]] * 2
    a = np.load(os.path.join(diag, "bank_acts_L12_F.npy"))
    assert np.array_equal(a[21], w.acts(["SYN 1 F 0 1"], 12)[0])
    for L in LAYERS:
        assert man["consistency_vs_acts_A"][f"L{L}"]["max_abs_diff"] == 0.0


def test_compute_dry_run_subset(env, tmp_path):
    stub = StubSubject(env["w"])
    man = SF.compute(stub, src_cache=env["src"], banks_dir=env["banks"], out_dir=str(tmp_path), layers=(12,),
                     concept_idx=[0, 4], consistency_n=2, log=lambda s: None)
    assert sorted(man["files"]) == ["bank_acts_L12_F.npy", "bank_acts_L12_R.npy", "bank_meta.json"]
    assert man["files"]["bank_acts_L12_F.npy"]["shape"] == [40, 16384]
    bm = json.load(open(tmp_path / "bank_meta.json"))
    assert sorted(set(bm["banks"]["R"]["rows"])) == [0, 4]
    # analyze refuses a partial (dry-run) bank
    with pytest.raises(ValueError):
        SF.analyze(env["src"], env["inst"], str(tmp_path), str(tmp_path / "o"), ref_sims=0, pool=False)


def test_split_f1_one_text_per_style():
    rows = [c for c in range(3) for _ in range(20)]
    styles = [s for _ in range(3) for s in STYLES_F for _ in range(2)]
    m = SF.split_f1(rows, styles)
    assert m.sum() == 30
    for c in range(3):
        for s in STYLES_F:
            assert sum(m[i] for i in range(60) if rows[i] == c and styles[i] == s) == 1
    assert np.array_equal(m, SF.split_f1(rows, styles))          # deterministic


# ------------------------------------------------------------------------------------------------ judge
def test_judge_rules():
    n = 25
    base = np.full(n, 0.5, dtype=np.float32)
    menu = list(range(1, 21))
    AF1, CF2, F2 = base.copy(), base.copy(), base.copy()
    AF1[3], CF2[3], F2[3] = 0.95, 0.92, 0.90
    v = SF.judge("planted", 3, menu, AF1, CF2, F2)
    assert v["keep"] and v["agree"] and v["robust"] and not v["rival"]
    # key changes: another option wins on A+F1
    a2 = AF1.copy(); a2[7] = 0.97
    v = SF.judge("planted", 3, menu, a2, CF2, F2)
    assert not v["keep"] and v["reasons"] == ["key_changed"] and v["ms_key"] == 7
    # not robust on C+F2; F2-only and both
    c2 = CF2.copy(); c2[3] = 0.84
    assert SF.judge("planted", 3, menu, AF1, c2, F2)["reasons"] == ["not_robust"]
    f2 = F2.copy(); f2[3] = 0.70
    assert SF.judge("planted", 3, menu, AF1, CF2, f2)["keep"]
    assert not SF.judge("planted", 3, menu, AF1, CF2, f2, robust_set="F2")["keep"]
    assert not SF.judge("planted", 3, menu, AF1, CF2, f2, robust_set="both")["keep"]
    # rival distractor: spec keeps, symmetric drops
    c3 = CF2.copy(); c3[9] = 0.88
    assert SF.judge("planted", 3, menu, AF1, c3, F2)["keep"]
    assert SF.judge("planted", 3, menu, AF1, c3, F2, rival_rule="symmetric")["reasons"] == ["planted_rival"]
    # null: anchor 0 (not on menu)
    AF1n, CF2n, F2n = base.copy(), base.copy(), base.copy()
    AF1n[0], CF2n[0], F2n[0] = 0.95, 0.9, 0.9
    assert SF.judge("null", 0, menu, AF1n, CF2n, F2n)["keep"]
    c4 = CF2n.copy(); c4[5] = 0.86
    v = SF.judge("null", 0, menu, AF1n, c4, F2n)
    assert not v["keep"] and v["reasons"] == ["null_menu_separates"] and v["best_other"] == 5
    c5 = CF2n.copy(); c5[0] = 0.6                     # null latent not robust: symmetric drops, menu-only keeps
    assert not SF.judge("null", 0, menu, AF1n, c5, F2n)["keep"]
    assert SF.judge("null", 0, menu, AF1n, c5, F2n, null_rule="menu-only")["keep"]
    a5 = AF1n.copy(); a5[11] = 0.99                    # null: a menu option beats the anchor on A+F1
    assert "key_changed" in SF.judge("null", 0, menu, a5, CF2n, F2n)["reasons"]
    nanrow = AF1.copy(); nanrow[0] = np.nan            # NaN rows (concept without texts) never win
    assert SF.judge("planted", 3, menu, nanrow, CF2, F2)["ms_key"] == 3


# ------------------------------------------------------------------------------------------------ analyze
def _rows(env):
    return json.load(open(os.path.join(env["out"], "slots.json")))["slots"]


def test_analyze_matches_known_latent_types(env):
    w, rows = env["w"], _rows(env)
    cid2i = {c["cid"]: i for i, c in enumerate(w.cs)}
    menus = {}
    for d, inst in SF.iter_instances(env["inst"]):
        for i, s in enumerate(inst["extra"]["slots"]):
            menus[(inst["instance_id"], i)] = [cid2i[c] for c in s["menu"]]
    seen = {"R": 0, "S": 0, "P_null_flag": 0, "P_planted_rival": 0}
    for r in rows:
        ty, ci = w.type_of[(r["layer"], r["real_latent"])]
        assert ty in ("R1", "R2", "S1", "S2", "P") and w.cs[ci]["cid"] == r["anchor"]
        partner_on_menu = w.partner(ci) in menus[(r["instance_id"], r["slot"])]
        if ty.startswith("R"):
            assert r["keep"] and r["agree"] and r["robust"], r
            seen["R"] += 1
        elif ty.startswith("S"):
            assert not r["keep"] and "not_robust" in r["reasons"] and r["agree"], r
            assert r["auroc_CF2_anchor"] < 0.85 and r["auroc_A_anchor"] >= 0.9
            seen["S"] += 1
        else:
            if partner_on_menu and r["kind"] == "null":
                assert not r["keep"] and r["reasons"] == "null_menu_separates", r
                seen["P_null_flag"] += 1
            elif partner_on_menu:
                assert r["keep"] and r["rival"] and not r["keep_CF2_symmetric"], r
                seen["P_planted_rival"] += 1
            else:
                assert r["keep"], r
    assert all(v > 0 for v in seen.values()), seen       # every branch exercised
    with open(os.path.join(env["out"], "slots.csv")) as f:
        assert len(list(csv.DictReader(f))) == len(rows)


def test_analyze_pool_summary_and_lists(env):
    w, s = env["w"], env["summary"]
    for k, v in s["integrity"]["max_abs_diff_splitA_auroc_vs_generator_cache"].items():
        assert v < 2e-3, k
    assert s["integrity"]["max_abs_diff_instance_menu_auroc_A"] < 2e-3
    with open(os.path.join(env["out"], "pool_latents.csv")) as f:
        pool = list(csv.DictReader(f))
    types = [w.type_of[(int(p["layer"]), int(p["real_latent"]))][0] for p in pool]
    assert "N" not in types and "W" not in types              # generator pool filters still apply
    for p, ty in zip(pool, types):
        assert (p["keep"] == "True") == (ty[0] in "RP"), (p, ty)
    assert s["pool"]["n"] == len(pool) and 0 < s["pool"]["kept"] < len(pool)
    z = np.load(os.path.join(env["diag"], "pool_auroc.npz"))
    assert sum(len(z[f"L{L}_latents"]) for L in LAYERS) == len(pool) and z["L6_CF2"].shape[1] == w.n_c
    filt = json.load(open(os.path.join(env["out"], "filtered_instances.json")))
    rows = _rows(env)
    assert len(filt["instances"]) == 3 * N_PER_TIER
    for iid, v in filt["instances"].items():
        kept = sorted(r["slot"] for r in rows if r["instance_id"] == iid and r["keep"])
        assert v["kept_slots"] == kept and v["all_kept"] == (len(kept) == v["n_slots"])
    cmp_ = s["slots_kept_vs_dropped"]
    assert cmp_["n_kept"] + cmp_["n_dropped"] == len(rows)
    assert set(cmp_["features"]) >= {"generic_fire_A", "margin_C", "fire_F_anchor"}
    assert cmp_["features"]["fire_F_anchor"]["kept"]["median"] > cmp_["features"]["fire_F_anchor"]["dropped"]["median"]
    assert "filter_selects_easier" in s["filter_selects_easier_slots"]
    assert s["reference_offline"]["ref_planted_acc_kept"] is not None
    assert all("ref_acc" in r for r in rows)


# ------------------------------------------------------------------------------------------------ write
def _load_pool(d):
    out = {}
    for x in sorted(os.listdir(d)):
        out[x] = (json.load(open(os.path.join(d, x, "instance.json"))), json.load(open(os.path.join(d, x, "public.json"))))
    return out


def test_write_redraw(env, tmp_path):
    from tasks.featurematch.grader import grade
    w = env["w"]
    outi, manp = str(tmp_path / "v2f"), str(tmp_path / "man.json")
    summ = WV.main(["--analysis", env["out"], "--pool-auroc", os.path.join(env["diag"], "pool_auroc.npz"),
                    "--src-cache", env["src"], "--src-instances", env["inst"], "--out-instances", outi,
                    "--manifest", manp, "--clean"])
    src = {i["instance_id"]: i for _, i in SF.iter_instances(env["inst"])}
    new = _load_pool(outi)
    assert len(new) == len(src) == summ["n_written"]
    assert summ["slot_count_dist"]["source"] == summ["slot_count_dist"]["v2f"]
    assert summ["null_fraction"]["source"] == summ["null_fraction"]["v2f"]
    assert sum(summ["redrawn_by_kind"].values()) == sum(summ["slots_failed"].values()) > 0
    PA = WV.PoolAuroc(os.path.join(env["diag"], "pool_auroc.npz"))
    cid2i = {c["cid"]: i for i, c in enumerate(w.cs)}
    man = {m["instance_id"]: m for m in json.load(open(manp))["instances"]}
    for iid, (inst, pub) in new.items():
        assert re.fullmatch(r"fm-t[123]-[0-9a-f]{10}", iid) and iid not in src
        s0 = src[inst["extra"]["style_filter"]["source_instance"]]
        assert [s["kind"] for s in inst["extra"]["slots"]] == [s["kind"] for s in s0["extra"]["slots"]]
        assert inst["tier"] == s0["tier"] and inst["caps"] == s0["caps"] and inst["extra"]["perm_seed"] != s0["extra"]["perm_seed"]
        for i, s in enumerate(inst["extra"]["slots"]):
            kind = "planted" if s["kind"] == "planted" else "null"
            ty, _ = w.type_of[(s["layer"], s["real_latent"])]
            assert ty[0] in "RP", (iid, i, ty)                       # only style-robust latents survive
            v = SF.judge(kind, cid2i[s["anchor"]], [cid2i[c] for c in s["menu"]], *PA.get(s["layer"], s["real_latent"]))
            assert v["keep"], (iid, i, v)
            if i not in man[iid]["redrawn_slots"]:
                assert s == s0["extra"]["slots"][i]                    # kept slots copied unchanged
        sub = {"answers": [{"slot": i, "choice": a["choice"]} for i, a in enumerate(inst["answer"]["slots"])]}
        d = os.path.join(outi, iid)
        assert grade(d, sub)["pass"]
        for f in ("instance.json", "public.json"):
            assert man[iid]["files"][f] == SF.sha256_file(os.path.join(d, f))
        assert inst["canary"] not in json.dumps(pub)
    # deterministic: a second run writes identical files
    WV.main(["--analysis", env["out"], "--pool-auroc", os.path.join(env["diag"], "pool_auroc.npz"),
             "--src-cache", env["src"], "--src-instances", env["inst"], "--out-instances", str(tmp_path / "v2f_b"),
             "--manifest", str(tmp_path / "man_b.json"), "--clean"])
    assert _load_pool(str(tmp_path / "v2f_b")) == new


def test_write_drop(env, tmp_path):
    summ = WV.main(["--analysis", env["out"], "--pool-auroc", os.path.join(env["diag"], "pool_auroc.npz"),
                    "--src-cache", env["src"], "--src-instances", env["inst"], "--out-instances", str(tmp_path / "d"),
                    "--manifest", str(tmp_path / "m.json"), "--mode", "drop", "--min-slots", "3", "--clean"])
    new = _load_pool(str(tmp_path / "d"))
    rows = _rows(env)
    assert len(new) == summ["n_written"] and summ["redrawn_by_kind"] == {"planted": 0, "null": 0}
    for iid, (inst, pub) in new.items():
        sid = inst["extra"]["style_filter"]["source_instance"]
        kept = [r for r in rows if r["instance_id"] == sid and r["keep"]]
        assert len(inst["extra"]["slots"]) == len(kept) >= 3
        assert [s["real_latent"] for s in inst["extra"]["slots"]] == [r["real_latent"] for r in kept]
        assert pub["n_slots"] == len(pub["slots"]) == inst["dial"]["n_slots"]


def test_write_regenerate(env, tmp_path):
    from tasks.featurematch.grader import grade
    w = env["w"]
    summ = WV.main(["--analysis", env["out"], "--pool-auroc", os.path.join(env["diag"], "pool_auroc.npz"),
                    "--src-cache", env["src"], "--out-instances", str(tmp_path / "g"), "--manifest",
                    str(tmp_path / "m.json"), "--mode", "regenerate", "--n-per-tier", "4", "--clean"])
    new = _load_pool(str(tmp_path / "g"))
    assert len(new) == 12 and summ["slot_count_dist"]["source"] == summ["slot_count_dist"]["v2f"]
    seeds = sorted(i["seed"] for i, _ in new.values())
    assert seeds == [8000, 8001, 8002, 8003, 108000, 108001, 108002, 108003, 208000, 208001, 208002, 208003]
    for iid, (inst, pub) in new.items():
        for s in inst["extra"]["slots"]:
            assert w.type_of[(s["layer"], s["real_latent"])][0][0] in "RP"
        sub = {"answers": [{"slot": i, "choice": a["choice"]} for i, a in enumerate(inst["answer"]["slots"])]}
        assert grade(os.path.join(str(tmp_path / "g"), iid), sub)["pass"]
