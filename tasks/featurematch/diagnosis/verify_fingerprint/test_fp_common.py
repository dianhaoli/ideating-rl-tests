"""CPU tests for the in-memory variant generator and the bootstrap AUROC (real caches, read-only, ~30 s).

    cd ~/wt/fmdiag && /opt/pytorch/bin/python -m pytest -q tasks/featurematch/diagnosis/verify_fingerprint/
"""
import numpy as np
import pytest

from tasks.featurematch.diagnosis.verify_fingerprint import fp_common as F
from tasks.featurematch.diagnosis.verify_fingerprint import fp_study as S

SEEDS = range(990000, 990150)


@pytest.fixture(scope="module")
def worlds():
    return F.load_worlds()


def _excl(T, r):
    from tasks.featurematch import generate as G
    A, Bv = T.aA[r["layer"]][r["latent"]], T.aB[r["layer"]][r["latent"]]
    return set(np.nonzero((A >= G.N_THR) | (Bv >= G.N_THR_B))[0].tolist()) | {r["anchor"]}


def test_current_is_the_generator(worlds):
    from tasks.featurematch import generate as G
    _, Tf = worlds
    rows, fails = F.draw(Tf, "T2", SEEDS)
    assert fails == 0
    k = 0
    for seed in SEEDS:
        _, inst, pub = G.make_instance(Tf, seed, "T2")
        for ps in pub["slots"]:
            assert rows[k]["labels"] == ps["options"]
            k += 1
    assert k == len(rows)
    assert G.make_slot is F._orig_make_slot


@pytest.mark.parametrize("variant", F.VARIANTS[1:])
def test_variant_keeps_generator_guarantees(worlds, variant):
    _, Tf = worlds
    rows, fails = F.draw(Tf, "T2", SEEDS, variant)
    assert fails == 0
    for r in rows:
        menu, a, L = r["menu"], r["anchor"], r["layer"]
        assert len(set(menu)) == 20
        assert set(menu) <= set(int(c) for c in Tf.answerable[L])           # only answerable concepts
        ex = _excl(Tf, r)
        if r["planted"]:
            assert menu.count(a) == 1 and not (set(menu) - {a}) & ex        # distractors never confusable
        else:
            assert a not in menu and not set(menu) & ex
        assert (L, r["latent"]) in {(L2, j) for L2 in (6, 12, 18) for j, _ in Tf.planted[L2]}


def test_swap_nn_changes_planted_menus_only(worlds):
    _, Tf = worlds
    cur, _ = F.draw(Tf, "T2", SEEDS)
    sw, _ = F.draw(Tf, "T2", SEEDS, "swap_nn")
    assert len(cur) == len(sw)
    n_changed = 0
    for a, b in zip(cur, sw):
        assert (a["layer"], a["latent"], a["anchor"], a["planted"]) == (b["layer"], b["latent"], b["anchor"], b["planted"])
        if not a["planted"]:
            assert a["menu"] == b["menu"]                                    # null menus untouched
        else:
            assert len(set(a["menu"]) & set(b["menu"])) >= 19 - 1 + 1 - 1   # c* + 18 shared at least
            n_changed += set(a["menu"]) != set(b["menu"])
    assert n_changed > 0


def test_t1_untouched(worlds):
    _, Tf = worlds
    for v in F.VARIANTS[1:]:
        a, _ = F.draw(Tf, "T1", SEEDS)
        b, _ = F.draw(Tf, "T1", SEEDS, v)
        assert [r["menu"] for r in a] == [r["menu"] for r in b]


def test_boot_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(size=500) + 0.5 * y, 1)                         # ties on purpose
    g = np.repeat(np.arange(125), 4)
    r = S.boot(y, s, g, n_boot=200)
    assert abs(r["auroc"] - F.auroc(y, s)) < 6e-5                          # boot() rounds to 4 decimals
    assert r["ci95"][0] < r["auroc"] < r["ci95"][1]
    d = S.boot_paired_diff((y, s, g), (y, s, g), n_boot=50)
    assert d["diff"] == 0 and d["ci95"] == [0.0, 0.0]


def test_label_tokens():
    assert F.label_tokens("text written in Spanish") == (True, "language", {"spanish"})
    lang, head, words = F.label_tokens("article about an American football player")
    assert not lang and head == "player" and words == {"american", "football", "player"}


def test_episode_metrics_agree_with_real_grader(worlds, tmp_path):
    """episode_metrics' pass rule = grader.grade's on the same decisions (instances written to a pytest temp dir)."""
    import json
    from tasks.featurematch import generate as G
    from tasks.featurematch.grader import grade
    _, Tf = worlds
    rng = np.random.default_rng(1)
    rows, n_pass_grader = [], 0
    claims, picks = [], []
    cid2i = {c["cid"]: i for i, c in enumerate(Tf.cs)}
    for seed in range(991000, 991060):
        _, inst, pub = G.make_instance(Tf, seed, "T2")
        d = tmp_path / str(seed)
        d.mkdir()
        json.dump(inst, open(d / "instance.json", "w"))
        answers = []
        for i, x in enumerate(inst["extra"]["slots"]):
            menu = [cid2i[c] for c in x["menu"]]
            r = {"seed": seed, "planted": x["kind"] == "planted", "anchor": cid2i[x["anchor"]], "menu": menu}
            c = bool(rng.random() < 0.6)
            p = r["anchor"] if (r["planted"] and rng.random() < 0.7) else menu[int(rng.integers(20))]
            rows.append(r), claims.append(c), picks.append(p)
            answers.append({"slot": i, "choice": menu.index(p) + 1 if c else "nothing found"})
        n_pass_grader += grade(str(d), {"answers": answers})["pass"]
    m = S.episode_metrics(rows, np.array(claims), picks)
    assert m["n_episodes"] == 60
    assert round(n_pass_grader / 60, 4) == m["pass_rate"]
    assert n_pass_grader > 0


def test_knn_picks_finds_repeated_menus(worlds):
    _, Tf = worlds
    rows, _ = F.draw(Tf, "T2", SEEDS)
    picks = S.knn_picks(rows, rows, len(Tf.cs), k=1)                       # each planted menu is its own neighbour
    assert all(p == r["anchor"] for p, r in zip(picks, rows) if r["planted"])
    assert all(p in r["menu"] for p, r in zip(picks, rows))
