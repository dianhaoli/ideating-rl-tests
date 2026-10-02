"""CPU end-to-end test of the step-2 style filter (PREREG Amendment 1, A1.1) on a synthetic world (tests/synth_world.py).

Covers: compute (batching, IO, normalisation, manifest, dry-run subset) with a stub model; the A1.1 rule on hand-made
rows (equal weighting of A and F1, H with and without split C, two separate thresholds on C and F2, several reasons);
the deterministic first/second F1/F2 split; analyze end to end (every pooled latent's verdict and reasons match its
known type, including types that the pre-A1.1 pooled metrics would have judged differently; key_check.jsonl and
slot_disagreements.jsonl); write_v2f regenerate mode (seeds, kept latents only, 20-option menus, grader, determinism,
manifest, the planted multi-style key check, fingerprint block).

    cd ~/wt/fmdiag && /opt/pytorch/bin/python -m pytest -q tasks/featurematch/diagnosis/tests/test_style_filter.py
"""
import csv
import json
import os
import re
import shutil
import tempfile
from collections import Counter
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
    src, banks, inst, diag, out, logs = (str(root / x) for x in ("cache", "banks", "instances", "diag", "out", "logs"))
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
    summary = SF.analyze(src, inst, diag, out, ref_sims=1, banks_dir=banks, logs_dir=logs, log=lambda s: None)
    yield {"w": w, "T": T, "src": src, "banks": banks, "inst": inst, "diag": diag, "out": out, "stub": stub,
           "man": man, "summary": summary, "root": root, "logs": logs}
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
        SF.analyze(env["src"], env["inst"], str(tmp_path), str(tmp_path / "o"), ref_sims=0, banks_dir=None,
                   logs_dir=str(tmp_path / "l"))


def test_split_f1_first_text_per_style_goes_to_f1():
    rows = [c for c in range(3) for _ in range(20)]
    styles = [s for _ in range(3) for s in STYLES_F for _ in range(2)]
    m = SF.split_f1(rows, styles)
    assert m.sum() == 30
    assert m.tolist() == [i % 2 == 0 for i in range(60)]          # first listed -> F1, second -> F2
    # interleaved file order: still the FIRST occurrence of each (concept, style)
    rows2 = [0, 0, 0, 0]
    styles2 = ["a", "b", "b", "a"]
    assert SF.split_f1(rows2, styles2).tolist() == [True, True, False, False]
    with pytest.raises(ValueError):
        SF.split_f1([0, 0, 0], ["a", "a", "a"])                     # A1.1 needs exactly 2 per (concept, style)
    with pytest.raises(ValueError):
        SF.split_f1([0], ["a"])


def _rows(n=25, v=0.5):
    return [np.full(n, v, dtype=np.float32) for _ in range(4)]


def test_scores_equal_weight_and_heldout():
    aA, aC, aF1, aF2 = _rows()
    # c*=3 best on A; concept 7 much better on F1. Pooling 40 A texts with 10 F1 texts would weight A 4:1
    # (0.8*0.97 + 0.2*0.40 = 0.856 for c* vs 0.8*0.70 + 0.2*0.99 = 0.758 for 7) and keep c*; equal weight does not.
    aA[3], aF1[3] = 0.97, 0.40
    aA[7], aF1[7] = 0.70, 0.99
    M, H = SF.scores(aA, aC, aF1, aF2)
    assert M[3] == pytest.approx(0.685) and M[7] == pytest.approx(0.845)
    aC[3], aF2[3] = 0.95, 0.95
    v = SF.verdict(3, aA, aC, aF1, aF2)
    assert v["k_ms"] == 7 and v["k_ho"] == 3 and v["reasons"] == ["ms_key_differs"] and not v["keep"]
    # H = AUROC_F2 alone for a concept without split C (NaN), and such a concept can win k_ho
    aA, aC, aF1, aF2 = _rows()
    aA[3], aF1[3], aC[3], aF2[3] = 0.95, 0.95, 0.90, 0.90
    aC[5], aF2[5] = np.nan, 0.92
    M, H = SF.scores(aA, aC, aF1, aF2)
    assert H[5] == pytest.approx(0.92) and H[3] == pytest.approx(0.90)
    v = SF.verdict(3, aA, aC, aF1, aF2)
    assert v["k_ho"] == 5 and v["reasons"] == ["heldout_key_differs"]
    aF2[5] = 0.88
    assert SF.verdict(3, aA, aC, aF1, aF2)["keep"]
    nanA = aA.copy(); nanA[0] = np.nan                            # NaN never wins
    assert SF.verdict(3, nanA, aC, aF1, aF2)["k_ms"] == 3


def test_verdict_separate_thresholds_and_several_reasons():
    aA, aC, aF1, aF2 = _rows()
    aA[3], aF1[3] = 0.95, 0.95
    for c, f2, keep in ((0.80, 0.95, False), (0.95, 0.84, False), (0.85, 0.85, True), (0.849, 0.99, False)):
        aC[3], aF2[3] = c, f2
        v = SF.verdict(3, aA, aC, aF1, aF2)
        assert v["keep"] == keep, (c, f2)
        assert v["robust_C"] == (c >= 0.85) and v["robust_F2"] == (f2 >= 0.85)
        if not keep:
            assert v["reasons"] == ["not_style_robust"]
    # mean of C and F2 (0.875) would pass a pooled 0.85 bar; A1.1's two separate thresholds do not
    aC[3], aF2[3] = 0.80, 0.95
    assert not SF.verdict(3, aA, aC, aF1, aF2)["keep"]
    # all three reasons at once
    aA2, aC2, aF12, aF22 = _rows()
    aA2[3], aF12[3], aC2[3], aF22[3] = 0.92, 0.30, 0.60, 0.40
    aF12[9], aC2[9], aF22[9] = 0.99, 0.95, 0.95
    v = SF.verdict(3, aA2, aC2, aF12, aF22)
    assert v["reasons"] == ["ms_key_differs", "heldout_key_differs", "not_style_robust"]
    assert SF.option_pick(9, [4, 9, 2]) == 2 and SF.option_pick(9, [4, 2]) == "not on menu"


# ------------------------------------------------------------------------------------------------ analyze
EXPECTED = {"R": [], "S": ["not_style_robust"], "X": ["not_style_robust"], "K": ["ms_key_differs"],
            "P": ["heldout_key_differs"]}


def _key_rows(env):
    return [json.loads(x) for x in open(os.path.join(env["logs"], "key_check.jsonl"))]


def test_key_check_matches_latent_types(env):
    w, rows = env["w"], _key_rows(env)
    T = env["T"]
    assert len(rows) == sum(len(T.planted[L]) for L in LAYERS)
    seen = Counter()
    for r in rows:
        ty, ci = w.type_of[(r["layer"], r["real_latent"])]
        assert ty[0] in EXPECTED, (r, ty)                        # N and W never pooled
        assert r["c_star"] == w.cs[ci]["cid"]
        assert r["reasons"] == EXPECTED[ty[0]], (ty, json.dumps(r))
        assert r["kept"] == (ty[0] == "R")
        for k in ("top3_A", "top3_M", "top3_H"):
            assert len(r[k]) == 3 and r[k][0][1] >= r[k][1][1] >= r[k][2][1]
        assert r["top3_A"][0][0] == r["c_star"]
        if ty[0] == "K":
            partner = w.cs[w.partner(ci)]["cid"]
            assert r["k_ms"] == partner and r["k_ho"] == r["c_star"]
            # only the first/second split makes this deterministic: c* fires on F2 (second texts), not F1 (first)
            assert r["auroc_F1_cstar"] < 0.6 and r["auroc_F2_cstar"] >= 0.85, r
        if ty[0] == "P":
            assert r["k_ho"] == w.cs[w.partner(ci)]["cid"] and r["k_ms"] == r["c_star"]
        if ty[0] == "X":
            assert r["auroc_C_cstar"] < 0.85 <= r["auroc_F2_cstar"]
        seen[ty[0]] += 1
    assert set(seen) == set(EXPECTED), seen


def test_types_discriminate_a11_from_pooled_metrics(env):
    """K would be KEPT by a pooled A+F1 key and X by a pooled C+F2 threshold; A1.1 drops both."""
    w, T = env["w"], env["T"]
    meta = json.load(open(os.path.join(env["src"], "precompute_meta.json")))
    bm = json.load(open(os.path.join(env["diag"], "bank_meta.json")))
    f1 = SF.split_f1(bm["banks"]["F"]["rows"], bm["banks"]["F"]["styles"])
    rF = np.array(bm["banks"]["F"]["rows"])
    L = 12
    cols = [w.lat[(L, "K", 4)], w.lat[(L, "X", 4)]]
    ld = lambda p: np.asarray(np.load(p)[:, cols], dtype=np.float32)   # noqa: E731
    XA, XC = ld(os.path.join(env["src"], f"acts_L{L}_A.npy")), ld(os.path.join(env["src"], f"acts_L{L}_C.npy"))
    XF = ld(os.path.join(env["diag"], f"bank_acts_L{L}_F.npy"))
    AF1 = AU.auroc_table(np.vstack([XA, XF[f1]]), np.concatenate([meta["rowsA"], rF[f1]]), w.n_c)
    CF2 = AU.auroc_table(np.vstack([XC, XF[~f1]]), np.concatenate([meta["rowsC"], rF[~f1]]), w.n_c)
    assert int(np.argmax(AF1[0])) == 4                            # pooled key keeps K's anchor ...
    assert CF2[1, 4] >= 0.85                                      # ... and pooled C+F2 passes X
    kr = {(r["layer"], r["real_latent"]): r for r in _key_rows(env)}
    assert kr[(L, cols[0])]["reasons"] == ["ms_key_differs"] and kr[(L, cols[1])]["reasons"] == ["not_style_robust"]


def test_slot_disagreements_and_summary(env):
    w, s = env["w"], env["summary"]
    cid2i = {c["cid"]: i for i, c in enumerate(w.cs)}
    dis = [json.loads(x) for x in open(os.path.join(env["logs"], "slot_disagreements.jsonl"))]
    expected = []
    for d, inst in SF.iter_instances(env["inst"]):
        for i, sl in enumerate(inst["extra"]["slots"]):
            ty, ci = w.type_of[(sl["layer"], sl["real_latent"])]
            if ty[0] != "R":
                expected.append((inst["instance_id"], i, ty[0], ci, [cid2i[c] for c in sl["menu"]], sl["kind"]))
    assert len(dis) == len(expected) > 0
    by = {(r["instance_id"], r["slot"]): r for r in dis}
    kinds = Counter()
    for iid, i, ty, ci, menu, kind in expected:
        r = by[(iid, i)]
        k = "planted" if kind == "planted" else "null"
        assert r["kind"] == k and r["reasons"] == "+".join(EXPECTED[ty])
        p = w.partner(ci)
        if ty == "K":                                             # the multi-style key is the partner
            assert r["ms_pick"] == (menu.index(p) + 1 if p in menu else "not on menu")
        if ty == "P":
            assert r["ho_pick"] == (menu.index(p) + 1 if p in menu else "not on menu")
        if k == "planted" and ty in "SX":
            assert r["ms_pick"] == r["ho_pick"] == r["answer"] == menu.index(ci) + 1
        kinds[(k, ty, p in menu)] += 1
    assert any(t == "P" and on for (_, t, on) in kinds), kinds     # a P slot with the winning partner on the menu
    assert {k for k, _, _ in kinds} == {"planted", "null"}
    for L, v in s["integrity"]["max_abs_diff_splitA_auroc_vs_generator_cache"].items():
        if L.startswith("L"):
            assert v < 2e-3
    p = s["pool"]
    assert p["n"] == len(_key_rows(env)) and 0 < p["kept"] < p["n"]
    assert p["reason_any"]["ms_key_differs"] > 0 and p["reason_any"]["heldout_key_differs"] > 0
    assert p["P35_n"] == sum(1 for r in _key_rows(env) if r["reasons"] and "not_style_robust" not in r["reasons"])
    assert s["slots_v2"]["failed"] == len(dis)
    # among the CURRENT v2 planted slots, only a K latent with its partner on the menu has another best option on M
    assert s["slots_v2"]["planted"]["ms_best_option_not_answer"] == kinds[("planted", "K", True)]
    v = s["filter_selects_easier_PRIMARY_latents"]
    assert "filter_selects_easier" in v and v["ref_planted_acc_kept"] is not None
    with open(os.path.join(env["out"], "pool_latents.csv")) as f:
        assert len(list(csv.DictReader(f))) == p["n"]


def test_analyze_rejects_reordered_bank(env, tmp_path):
    bad = tmp_path / "diag"
    shutil.copytree(env["diag"], bad, ignore=shutil.ignore_patterns("*.npy", "*.npz"))
    bm = json.load(open(bad / "bank_meta.json"))
    fb = bm["banks"]["F"]
    fb["per_text_sha256"][0], fb["per_text_sha256"][1] = fb["per_text_sha256"][1], fb["per_text_sha256"][0]
    json.dump(bm, open(bad / "bank_meta.json", "w"))
    with pytest.raises(ValueError):
        SF.analyze(env["src"], env["inst"], str(bad), str(tmp_path / "o"), ref_sims=0, banks_dir=env["banks"],
                   logs_dir=str(tmp_path / "l"))


# ------------------------------------------------------------------------------------------------ write
def _load_pool(d):
    out = {}
    for x in sorted(os.listdir(d)):
        out[x] = (json.load(open(os.path.join(d, x, "instance.json"))), json.load(open(os.path.join(d, x, "public.json"))))
    return out


def _regen(env, tmp_path, tag, n=4, fp=0):
    return WV.main(["--key-check", os.path.join(env["logs"], "key_check.jsonl"),
                    "--pool-scores", os.path.join(env["diag"], "pool_scores.npz"), "--src-cache", env["src"],
                    "--out-instances", str(tmp_path / tag), "--manifest", str(tmp_path / f"{tag}.json"),
                    "--summary", str(tmp_path / f"{tag}_summary.json"), "--n-per-tier", str(n),
                    "--fingerprint-n", str(fp), "--clean"])


def test_write_regenerate(env, tmp_path):
    from tasks.featurematch.grader import grade
    w = env["w"]
    summ = _regen(env, tmp_path, "g", fp=20)
    new = _load_pool(str(tmp_path / "g"))
    assert len(new) == 12 and summ["instances_written_by_tier"] == {"T1": 4, "T2": 4, "T3": 4}
    assert not summ["errors_by_tier"] and summ["planted_ms_key_not_answer (expected 0)"] == 0
    seeds = sorted(i["seed"] for i, _ in new.values())
    assert seeds == [8000, 8001, 8002, 8003, 108000, 108001, 108002, 108003, 208000, 208001, 208002, 208003]
    kept = {(r["layer"], r["real_latent"]) for r in _key_rows(env) if r["kept"]}
    man = {m["instance_id"]: m for m in json.load(open(tmp_path / "g.json"))["instances"]}
    for iid, (inst, pub) in new.items():
        assert re.fullmatch(r"fm-t[123]-[0-9a-f]{10}", iid)
        for s in inst["extra"]["slots"]:
            assert (s["layer"], s["real_latent"]) in kept
            assert w.type_of[(s["layer"], s["real_latent"])][0][0] == "R"
            assert len(s["menu"]) == 20
        sub = {"answers": [{"slot": i, "choice": a["choice"]} for i, a in enumerate(inst["answer"]["slots"])]}
        d = os.path.join(str(tmp_path / "g"), iid)
        assert grade(d, sub)["pass"]
        for f in ("instance.json", "public.json"):
            assert man[iid]["files"][f] == SF.sha256_file(os.path.join(d, f))
        assert inst["canary"] not in json.dumps(pub)
        assert len(inst["extra"]["style_filter"]["ms_margin_info"]) == len(inst["extra"]["slots"])
    # generator v2 unchanged otherwise: the same seeds on generate.make_instance with the restricted Tables
    from tasks.featurematch.generate import make_instance
    T = WV.restrict_tables(SF.load_tables(env["src"]), kept)
    iid, inst, pub = make_instance(T, 108001, "T2")
    assert new[iid][1] == pub and new[iid][0]["answer"] == inst["answer"]
    fp = summ["fingerprint_P6"]
    assert set(fp["generator_in_memory"]) == {"T1", "T2", "T3"} and "max_cv_auroc" in fp
    # deterministic
    _regen(env, tmp_path, "g2")
    assert _load_pool(str(tmp_path / "g2")) == new


def test_restrict_tables_menu_universe(env):
    T = SF.load_tables(env["src"])
    kept = {(L, j) for L in LAYERS for j, c in T.planted[L] if env["w"].type_of[(L, j)][0] == "R1"}
    WV.restrict_tables(T, kept)
    for L in LAYERS:
        assert all(env["w"].type_of[(L, j)][0] == "R1" for j, _ in T.planted[L])
        assert list(T.answerable[L]) == sorted({c for _, c in T.planted[L]})
    f = WV.feasibility(T)
    assert all(v["latents_too_few_candidates_planted(<19)"] == 0 for v in f.values())
