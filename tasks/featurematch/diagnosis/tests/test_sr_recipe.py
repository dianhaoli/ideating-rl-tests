"""CPU tests of the step-3 style-robust recipe (sr_recipe.py) on a small synthetic world. Fast (< 5 s).

    cd ~/wt/fmdiag && /opt/pytorch/bin/python -m pytest -q tasks/featurematch/diagnosis/tests/test_sr_recipe.py
"""
import hashlib
import json
import os
import random
import types

import numpy as np
import pytest

from tasks.featurematch import grader
from tasks.featurematch import tools as TOOLS
from tasks.featurematch.diagnosis import sr_recipe as SR

STYLES = [f"style {k:02d}" for k in range(10)]
N_C = 24
WIDTH = 64
LAYERS = (6, 12, 18)


# ------------------------------------------------------------------------------------------------- decision rule
def test_sr_max_picks_highest_mean_not_highest_single_value():
    vals = [[0, 0, 0, 0, 0, 9.0],      # highest single value, mean 1.5
            [2, 2, 2, 2, 2, 2.0],      # highest mean 2.0
            [1, 1, 1, 1, 1, 1.0]]
    assert SR.decide(vals)["sr_max"] == 2


def test_sr_max_ties_go_to_lowest_option_and_silent_latent():
    r = SR.decide([[0.0] * 6 for _ in range(20)])
    assert r["sr_max"] == 1
    assert r["auc_best"] == 0.5 and r["sr_thr"] == SR.NOTHING


def test_sr_thr_threshold():
    # best = option 1 (mean 3); negatives [0, 5, 0, 0]; pos 3 beats 3 of 4 -> AUROC 0.75 < 0.78
    r = SR.decide([[3, 3], [0, 5], [0, 0]])
    assert r["sr_max"] == 1 and r["auc_best"] == pytest.approx(0.75) and r["sr_thr"] == SR.NOTHING
    assert SR.decide([[3, 3], [0, 5], [0, 0]], thr=0.75)["sr_thr"] == 1       # >= claims
    # negatives [0, 2, 0, 0] -> AUROC 1.0 -> claim
    r = SR.decide([[3, 3], [0, 2], [0, 0]])
    assert r["auc_best"] == 1.0 and r["sr_thr"] == 1
    # exactly the reference threshold: option 1 = [1, 1] (mean 1); 11 options [1.5, 0] (mean 0.75) and 14 options
    # [0, 0]: 50 negatives, 39 below 1 -> AUROC 39/50 = 0.78 -> claim (>=); just above the threshold -> abstain
    v = [[1.0, 1.0]] + [[1.5, 0.0]] * 11 + [[0.0, 0.0]] * 14
    r = SR.decide(v)
    assert r["sr_max"] == 1 and r["auc_best"] == pytest.approx(0.78) and r["sr_thr"] == 1
    assert SR.decide(v, thr=0.7801)["sr_thr"] == SR.NOTHING


# ------------------------------------------------------------------------------------------------ text selection
def test_choose_styles_deterministic_and_seeded():
    a = SR.choose_styles(STYLES)
    b = SR.choose_styles(list(reversed(STYLES)) * 2)          # order / duplicates of the input do not matter
    assert a == b == random.Random(20261002).sample(sorted(STYLES), 6)
    assert len(set(a)) == 6


def _layout():
    rows, styles = [], []
    for c in range(N_C):
        for s in STYLES:
            rows += [c, c]
            styles += [s, s]
    return rows, styles


def test_select_rows_one_first_text_per_chosen_style():
    rows, styles = _layout()
    chosen = SR.choose_styles(styles)
    sel = SR.select_rows(rows, styles, N_C, "sr6", chosen)
    sel2 = SR.select_rows(rows, styles, N_C, "sr6", chosen)
    assert sel == sel2
    for c in range(N_C):
        idx = sel[c]
        assert len(idx) == 6
        assert [styles[i] for i in idx] == chosen
        assert all(rows[i] == c for i in idx)
        for i in idx:                                         # the FIRST of the two texts of that style
            assert i == min(j for j in range(len(rows)) if rows[j] == c and styles[j] == styles[i])
    al = SR.select_rows(rows, styles, N_C, "all20")
    assert all(len(al[c]) == 20 for c in range(N_C))


# ------------------------------------------------------------------------------------------- end to end (synthetic)
@pytest.fixture()
def world(tmp_path, monkeypatch):
    """Concept c's planted latent (id c at every layer) fires 5.0 on c's texts of the CHOSEN styles and on c's
    second texts, and 0 elsewhere. Latent 40 + c fires only on c's texts of the NOT chosen styles (so SR-max with 6
    styles misses it but all20 sees it). Latent 63 is silent everywhere (a null latent)."""
    cids = [f"topic:C{c:02d}" for c in range(N_C)]
    rows, styles = _layout()
    chosen = SR.choose_styles(styles)
    texts = [f"text {i} of {cids[r]} in {s}" for i, (r, s) in enumerate(zip(rows, styles))]
    banks = tmp_path / "banks" / "R"
    banks.mkdir(parents=True)
    conc = {}
    for t, r, s in zip(texts, rows, styles):
        conc.setdefault(cids[r], []).append({"style": s, "text": t})
    (banks / "chunk_0_24.json").write_text(json.dumps({"bank": "R", "concepts": conc}))
    cache = tmp_path / "cache"
    cache.mkdir()
    meta = {"cids": cids, "banks": {"R": {"n": len(rows), "rows": rows, "styles": styles,
                                          "texts_sha256": hashlib.sha256("\n".join(texts).encode()).hexdigest()}}}
    (cache / "bank_meta.json").write_text(json.dumps(meta))
    X = np.zeros((len(rows), WIDTH), dtype=np.float16)
    for i, (r, s) in enumerate(zip(rows, styles)):
        if s in chosen:
            X[i, r] = 5.0
        else:
            X[i, 40 + r % 20] = 3.0 if r < 20 else 0.0
    for L in LAYERS:
        np.save(cache / f"bank_acts_L{L}_R.npy", X)
    src = tmp_path / "src"
    src.mkdir()
    (src / "concepts.json").write_text(json.dumps([{"cid": c, "label": f"article about {c}"} for c in cids]))
    monkeypatch.setenv("FM_SRC_CACHE", str(src))

    inst_dir = tmp_path / "instances"
    rng = random.Random(0)
    expect = {}

    def make(iid, tier, slots):
        d = inst_dir / iid
        d.mkdir(parents=True)
        ex, ans, pub = [], [], []
        for i, (layer, latent, anchor, planted) in enumerate(slots):
            others = [c for c in range(N_C) if c != anchor]
            rng.shuffle(others)
            menu = others[:19] + [anchor] if planted else others[:20]
            rng.shuffle(menu)
            ex.append({"layer": layer, "real_latent": latent, "anchor": cids[anchor], "menu": [cids[m] for m in menu]})
            ans.append({"planted": planted, "choice": menu.index(anchor) + 1 if planted else "nothing found",
                        "concept": cids[anchor] if planted else None})
            pub.append({"slot": i, "layer": layer, "latent": 1000 + i,
                        "options": [f"article about {cids[m]}" for m in menu]})
        (d / "instance.json").write_text(json.dumps({"instance_id": iid, "tier": tier, "caps": {"forward": 550},
                                                     "answer": {"slots": ans}, "extra": {"slots": ex}}))
        (d / "public.json").write_text(json.dumps({"n_slots": len(slots), "slots": pub}))

    # all-planted, robust latents -> every variant passes
    make("fm-t1-a", "T1", [(6, 0, 0, True), (12, 1, 1, True), (18, 2, 2, True)])
    # planted + null(silent latent): SR-max over-claims the null, SR-thr passes
    make("fm-t2-b", "T2", [(12, 3, 3, True), (6, 63, 4, False), (18, 5, 5, True), (6, 6, 6, True),
                           (12, 63, 7, False)])
    # planted latent that fires only on the non-chosen styles: 6-style variants miss, all20 variants find it
    make("fm-t3-c", "T3", [(18, 40 + 8, 8, True), (6, 9, 9, True), (12, 10, 10, True)])
    return {"inst": inst_dir, "cache": cache, "banks": tmp_path / "banks", "out": tmp_path / "out"}


def _run(w, out=None):
    return SR.run(str(w["inst"]), str(out or w["out"]), str(w["cache"]), verify=False, banks_dir=str(w["banks"]),
                  n_alt=2, log=lambda *a: None)


def test_end_to_end_with_real_grader(world):
    s = _run(world)
    assert s["bank_check"] == {"n": N_C * 20, "texts_sha256_match": True, "styles_match": True}
    assert s["label_check"] is True and s["label"].startswith("NOT the step-3 result")
    recs = [json.loads(l) for l in open(world["out"] / "answers.jsonl")]
    by = {(r["variant"], r["instance"]): r for r in recs}
    fake = types.SimpleNamespace()
    for r in recs:
        pub = json.load(open(world["inst"] / r["instance"] / "public.json"))
        fake.public = pub
        assert TOOLS.Env.validate_submission(fake, r["submission"]) is None      # the tool server's format check
        # the real grader parses every answer and its stored grade is reproducible
        g = grader.grade(str(world["inst"] / r["instance"]), r["submission"])
        assert g == r["grade"]
        for a, sl in zip(r["submission"]["answers"], g["details"]["slots"]):
            assert grader._norm_choice(a["choice"]) is not None
            assert sl["claimed"] == (a["choice"] != "nothing found")
    assert all(by[(v, "fm-t1-a")]["grade"]["pass"] for v in SR.VARIANTS)
    assert not by[("SR-max", "fm-t2-b")]["grade"]["pass"]          # claims on the two nulls
    assert by[("SR-max", "fm-t2-b")]["grade"]["details"]["null_fp"] == 2
    assert by[("SR-thr", "fm-t2-b")]["grade"]["pass"]
    assert not by[("SR-max", "fm-t3-c")]["grade"]["pass"]
    assert by[("SR-max-all20", "fm-t3-c")]["grade"]["pass"]
    m = s["metrics"]["SR-max"]["all"]
    assert m["n_instances"] == 3 and m["n_planted"] == 9 and m["n_null"] == 2
    assert m["planted_acc"] == pytest.approx(8 / 9, abs=1e-4) and m["null_false_claim_rate"] == 1.0
    assert m["planted_nothing_found_rate"] == 0.0 and m["pass"] == 1
    assert s["metrics"]["SR-thr"]["all"]["null_false_claim_rate"] == 0.0
    assert m["forward_units_max"] == 600 and m["n_over_forward_cap"] == 1     # 5 slots x 120 > 550
    assert s["metrics"]["SR-max"]["T3"]["n_instances"] == 1
    lo, hi = m["pass_wilson95"]
    assert 0 <= lo <= 1 / 3 <= hi <= 1


def test_deterministic_output(world, tmp_path):
    a = _run(world)
    b = _run(world, tmp_path / "out2")
    for s in (a, b):
        s.pop("elapsed_s")
        s.pop("git_sha")
    assert a == b
    assert open(world["out"] / "answers.jsonl").read() == open(tmp_path / "out2" / "answers.jsonl").read()


def test_mapping_check_rejects_wrong_planted_answer(world):
    p = world["inst"] / "fm-t1-a" / "instance.json"
    d = json.loads(p.read_text())
    d["answer"]["slots"][0]["choice"] = d["answer"]["slots"][0]["choice"] % 20 + 1
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="planted answer is not the anchor"):
        _run(world)


# ----------------------------------------------------------------------------------------- PREREG Amendment 2 parts
def test_capped_counts_and_round_robin():
    assert SR.capped_counts(5, 550) == [120, 120, 120, 120, 70]
    assert SR.capped_counts(4, 550) == [120] * 4
    assert SR.capped_counts(3, 130) == [120, 10, 0]
    opt_rows = [[100 * j + r for r in range(6)] for j in range(20)]
    rr = SR.round_robin(opt_rows, 70)                     # 3 full style rounds + 10 options of the 4th
    assert [len(x) for x in rr] == [4] * 10 + [3] * 10
    assert rr[0] == [0, 1, 2, 3] and rr[19] == [1900, 1901, 1902]
    assert SR.round_robin(opt_rows, 120) == opt_rows
    assert all(x == [] for x in SR.round_robin(opt_rows, 0))


def test_decide_with_empty_options():
    r = SR.decide([[], [], []])
    assert r["sr_max"] == 1 and r["sr_thr"] == SR.NOTHING
    r = SR.decide([[], [0.0], [2.0]])                     # an option with no text cannot be claimed
    assert r["sr_max"] == 3 and r["sr_thr"] == 3          # AUROC 1.0 vs the one negative
    r = SR.decide([[5.0], []])                            # no negative at all -> AUROC 0.5 -> abstain
    assert r["sr_max"] == 1 and r["sr_thr"] == SR.NOTHING


def test_forward_cap_from_instance():
    assert SR.forward_cap({"caps": {"forward": 550}, "dial": {"forward_cap": 550}}) == 550
    assert SR.forward_cap({"dial": {"forward_cap": 1200}}) == 1200
    with pytest.raises(ValueError):
        SR.forward_cap({"instance_id": "x", "caps": {"forward": 550}, "dial": {"forward_cap": 1200}})
    with pytest.raises(ValueError):
        SR.forward_cap({"instance_id": "x"})


def test_cluster_bootstrap():
    import random as _r
    idx = SR.boot_index(7, b=50)
    rng = _r.Random(20261002)
    assert idx[0].tolist() == rng.choices(range(7), k=7)
    assert SR.cluster_bootstrap([2, 2, 2], [4, 4, 4]) == [0.5, 0.5]          # no between-instance variation
    a = SR.cluster_bootstrap([0, 1, 3, 4, 2], [4, 4, 4, 4, 4])
    assert a == SR.cluster_bootstrap([0, 1, 3, 4, 2], [4, 4, 4, 4, 4])     # deterministic
    assert a[0] < 0.5 < a[1]
    assert SR.cluster_bootstrap([0, 0], [0, 0]) == [None, None]


def test_pct_stats_and_straddle():
    st = SR.pct_stats([0.40 + 0.01 * i for i in range(21)])                # 0.40 .. 0.60
    assert st["min"] == 0.40 and st["median"] == 0.50 and st["max"] == 0.60
    assert st["p10"] == pytest.approx(0.42) and st["p90"] == pytest.approx(0.58)
    assert SR.straddles(st)
    assert not SR.straddles(SR.pct_stats([0.51, 0.55, 0.60]))
    assert SR.straddles({"p10": 0.50, "p90": 0.51})
    assert not SR.straddles({"p10": 0.40, "p90": 0.50})                  # nothing above 0.50 -> never fires


def _m(pa, n, lo, hi):
    return {"planted_acc": pa, "n_planted": n, "planted_acc_wilson95": [lo, hi], "planted_acc_cluster_boot95": [lo, hi]}


def test_stop_rule_logic():
    sens = {"text_selection_sensitive": True}
    r = SR.stop_rule({"SR-max": _m(0.60, 200, 0.53, 0.67), "SR-thr": _m(0.30, 200, 0.24, 0.37)}, None, True)
    assert r["fires"] and not r["borderline"] and r["labels"] == ["fires"] and r["firing_variants"] == ["SR-max"]
    r = SR.stop_rule({"SR-max": _m(0.52, 200, 0.45, 0.59), "SR-thr": _m(0.30, 200, 0.24, 0.37)}, sens, True)
    assert r["fires"] and r["borderline"] and r["labels"] == ["fires", "borderline", "text-selection-sensitive"]
    # one firing variant clearly above -> not borderline even if the other's CI includes 0.50
    r = SR.stop_rule({"SR-max": _m(0.70, 200, 0.63, 0.76), "SR-thr": _m(0.52, 200, 0.45, 0.59)}, None, True)
    assert r["fires"] and not r["borderline"]
    r = SR.stop_rule({"SR-max": _m(0.50, 200, 0.43, 0.57), "SR-thr": _m(0.30, 200, 0.24, 0.37)}, sens, False)
    assert not r["fires"] and r["labels"] == ["NOT a step-3 outcome (pool is not instances_v2f)", "does not fire",
                                              "text-selection-sensitive"]
    assert r["valid_as_step3_outcome"] is False
    r = SR.stop_rule({"SR-max": _m(0.90, 99, 0.82, 0.95), "SR-thr": _m(0.30, 99, 0.2, 0.4)}, None, True)
    assert not r["fires"] and not r["n_planted_ok"]


def test_amendment2_fields_end_to_end(world):
    s = _run(world)
    assert set(SR.VARIANTS) >= {"SR-max-capped", "SR-thr-capped"}
    # A2.3: fm-t2-b has 5 slots and cap 550 -> its last slot gets 70 texts
    recs = {(r["variant"], r["instance"]): r for r in map(json.loads, open(world["out"] / "answers.jsonl"))}
    assert recs[("SR-max-capped", "fm-t2-b")]["forward"] == 550
    assert recs[("SR-max", "fm-t2-b")]["forward"] == 600
    mc = s["metrics"]["SR-max-capped"]["all"]
    assert mc["n_over_forward_cap"] == 0 and mc["n_episodes_needing_more_than_cap"] == 1
    assert mc["n_slots_truncated"] == 1
    import csv as _csv
    rows = [r for r in _csv.DictReader(open(world["out"] / "slots.csv")) if r["instance"] == "fm-t2-b"]
    assert [int(r["sr6cap_texts"]) for r in rows] == [120, 120, 120, 120, 70]
    # A2.5: the clustered CI is reported for every variant/tier and brackets the point estimate
    for v in SR.VARIANTS:
        for t, m in s["metrics"][v].items():
            lo, hi = m["planted_acc_cluster_boot95"]
            assert lo <= m["planted_acc"] + 1e-9 and m["planted_acc"] <= hi + 1e-9
    # A2.2: 2 x (n_alt + 1) selections, primary included and equal to the graded primary
    sens = s["text_selection_sensitivity"]
    assert sens["n_selections"] == 6
    prim = [o for o in sens["runs"] if o["primary"]]
    assert len(prim) == 1 and prim[0]["sr_max_planted_acc"] == s["metrics"]["SR-max"]["all"]["planted_acc"]
    assert set(sens["planted_acc"]["SR-max"]) == {"n", "min", "p10", "median", "p90", "max"}
    assert sens["all20_planted_acc"]["SR-max-all20"] == s["metrics"]["SR-max-all20"]["all"]["planted_acc"]
    assert isinstance(sens["text_selection_sensitive"], bool)
    st = s["stop_rule"]
    assert st["valid_as_step3_outcome"] is False and st["n_planted_ok"] is False and st["fires"] is False
    assert st["text_selection_sensitive"] == sens["text_selection_sensitive"]


def test_capped_reads_cap_from_instance(world):
    p = world["inst"] / "fm-t1-a" / "instance.json"
    d = json.loads(p.read_text())
    d["caps"]["forward"] = 130                             # -> slots get 120, 10, 0 texts
    p.write_text(json.dumps(d))
    _run(world)
    recs = {(r["variant"], r["instance"]): r for r in map(json.loads, open(world["out"] / "answers.jsonl"))}
    assert recs[("SR-max-capped", "fm-t1-a")]["forward"] == 130
    assert recs[("SR-max-capped", "fm-t1-a")]["submission"]["answers"][2]["choice"] == 1
    assert recs[("SR-thr-capped", "fm-t1-a")]["submission"]["answers"][2]["choice"] == SR.NOTHING
    assert recs[("SR-max", "fm-t1-a")]["grade"]["pass"]                 # the uncapped primary is unaffected
