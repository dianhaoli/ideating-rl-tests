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
