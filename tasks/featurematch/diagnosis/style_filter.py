"""FeatureMatch diagnosis, STEP 2: style-robustness filter (the main validity fix; PLAN.md STEP 2 + STEP 1b).

Plain language: the original answer key says "latent j belongs to concept c*" because j separates c*'s
encyclopedia-style dataset texts (split A) from every other concept's. If j only does that on encyclopedia-style
text and goes quiet on a casual post or a dialogue about c*, an agent that writes its own probes in another style
sees nothing, and fails for a reason that has nothing to do with interpretability. This module re-derives the key on
a MULTI-STYLE text set and keeps only slots whose key survives:

  multi-style key  = the menu concept (planted) / menu + anchor concept (null) with the highest AUROC on
                     dataset split A + bank-F half F1 (one text per style per concept)
  agreement        = multi-style key == original key (the anchor c*)
  robustness       = AUROC of the key concept on held-out dataset split C + bank-F half F2 (the other text per style)
                     >= 0.85 (C and F2 are used by neither key definition)
  planted slot kept iff agreement AND robustness
  null slot        additionally FLAGGED (and dropped) if any menu concept reaches >= 0.85 on C + F2 (the latent now
                   separates an option, so "nothing found" would no longer be clearly right)
AUROC is exactly the generator's (auroc.auroc_table): concept-vs-rest over the whole 232-concept universe, ties 1/2.

Modes
  compute   (GPU, one gpuq job) gemma-2-2b + the 3 cached Gemma Scope SAEs over every bank-F and bank-R text, with the
            generator's per-text summary (fm_core.Subject.maxpool_acts: max over tokens, BOS excluded, texts truncated
            to MAX_TOKENS=64 tokens; texts whitespace-normalised and cut to 400 chars exactly like concepts.py).
            Writes diagnosis/cache/bank_acts_L{6,12,18}_{F,R}.npy float16 [4640, 16384] (gitignored) + bank_meta.json +
            manifest.json (shapes, sha256); the manifest is also copied to diagnosis/style_filter_cache_manifest.json.
            --dry-run: CPU, 2 concepts, 1 layer, output under diagnosis/cache/dryrun/.
  analyze   (CPU) per-slot verdicts for every v2 instance, per-latent verdicts for the whole generator latent pool,
            survivor-vs-dropped comparison. Writes diagnosis/style_filter_out/{slots.csv,slots.json,pool_latents.csv,
            summary.json,filtered_instances.json} and diagnosis/cache/pool_auroc.npz (used by write_v2f.py).

Usage (from ~/wt/fmdiag, after `source ~/ideating-rl-tests/common/env.sh`): see diagnosis/STYLE_FILTER.md.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from tasks.featurematch import concepts as C
from tasks.featurematch.auroc import auroc_table
from tasks.featurematch.fm_core import D_SAE, LAYERS, MAX_TOKENS

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
BANKS = os.path.join(HERE, "banks")
DIAG_CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "style_filter_out")
REPO_MANIFEST = os.path.join(HERE, "style_filter_cache_manifest.json")
FM_MAIN = os.path.expanduser("~/wt/featurematch/tasks/featurematch")   # read-only: v2 caches + instances live here
THR = 0.85
BATCH = 32                  # same as precompute.py
SPLIT_SEED = 20261002       # F1/F2 assignment of the two texts per (concept, style)
GPU_LABEL = "featurematch-diag-banks"
FEATS = ["generic_fire_A", "generic_fire_F", "fire_A_anchor", "fire_F_anchor", "margin_A", "margin_C", "margin_CF2",
         "auroc_A_anchor", "auroc_C_anchor", "auroc_F2_anchor"]


# ---------------------------------------------------------------------------------------------------------- paths
def default_src_cache():
    """Where the v2 generator's caches live (concepts.json, precompute_meta.json, acts/auroc tables). They are
    gitignored, so this worktree has none; the fix-stage worktree's copy is read (never written)."""
    if os.environ.get("FM_SRC_CACHE"):
        return os.environ["FM_SRC_CACHE"]
    local = os.path.join(TASK, "cache")
    return local if os.path.exists(os.path.join(local, "precompute_meta.json")) else os.path.join(FM_MAIN, "cache")


def default_src_instances():
    return os.environ.get("FM_SRC_INSTANCES") or os.path.join(FM_MAIN, "instances")


def use_src_cache(src_cache):
    """Point concepts.py (and through it generate.Tables / simulate.load) at src_cache. Module-global patch, so call it
    once per process before building Tables."""
    C.CACHE = src_cache
    C.OUT = os.path.join(src_cache, "concepts.json")


def load_tables(src_cache):
    use_src_cache(src_cache)
    from tasks.featurematch.generate import Tables
    return Tables()


def sha256_file(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_sha():
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def norm_text(t):
    """Exactly the normalisation concepts.py applied to the dataset texts (whitespace collapse, 400 chars)."""
    return " ".join(str(t).split())[:C.MAX_CHARS]


# ----------------------------------------------------------------------------------------------------------- banks
def load_bank(bank, banks_dir, cids, concept_idx=None):
    """Rows of one bank in the generator's concept order. Returns texts, rows (index into cids), styles."""
    allc = {}
    files = sorted(glob.glob(os.path.join(banks_dir, bank, "chunk_*.json")))
    if not files:
        raise FileNotFoundError(f"no bank {bank} chunks under {banks_dir}")
    for f in files:
        d = json.load(open(f))
        if d.get("bank") not in (None, bank):
            raise ValueError(f"{f}: bank field {d.get('bank')!r} != {bank!r}")
        dup = set(allc) & set(d["concepts"])
        if dup:
            raise ValueError(f"{f}: concepts repeated across chunks: {sorted(dup)[:3]}")
        allc.update(d["concepts"])
    idx = range(len(cids)) if concept_idx is None else concept_idx
    missing = [cids[i] for i in idx if cids[i] not in allc]
    if missing:
        raise ValueError(f"bank {bank} lacks {len(missing)} concepts, e.g. {missing[:3]}")
    texts, rows, styles = [], [], []
    for i in idx:
        for t in allc[cids[i]]:
            texts.append(norm_text(t["text"]))
            rows.append(int(i))
            styles.append(t["style"])
    return texts, rows, styles


def split_f1(rows, styles, seed=SPLIT_SEED):
    """Boolean mask: True = F1 (key definition), False = F2 (held-out check). For every (concept, style) group the
    texts are shuffled with a fixed seed and the first half (1 of 2) goes to F1. Deterministic, style-balanced."""
    groups = defaultdict(list)
    for i, (r, s) in enumerate(zip(rows, styles)):
        groups[(int(r), s)].append(i)
    mask = np.zeros(len(rows), dtype=bool)
    rng = np.random.default_rng(seed)
    for key in sorted(groups):
        g = groups[key]
        perm = rng.permutation(len(g))
        for p in perm[: max(1, len(g) // 2)]:
            mask[g[p]] = True
    return mask


# --------------------------------------------------------------------------------------------------------- compute
def compute(subject, src_cache=None, banks_dir=BANKS, out_dir=DIAG_CACHE, banks=("F", "R"), layers=LAYERS,
            concept_idx=None, batch=BATCH, consistency_n=32, repo_manifest=None, extra_meta=None, log=print):
    """Run `subject.maxpool_acts` (fm_core.Subject or a stub with the same signature) over the banks.
    Rows keep the generator's concept order (precompute_meta cids). Writes float16 arrays + bank_meta + manifest."""
    t0 = time.time()
    src_cache = src_cache or default_src_cache()
    meta = json.load(open(os.path.join(src_cache, "precompute_meta.json")))
    cids = meta["cids"]
    os.makedirs(out_dir, exist_ok=True)
    bmeta = {"cids": cids, "layers": list(layers), "max_tokens": MAX_TOKENS, "max_chars": C.MAX_CHARS,
             "batch": batch, "summary": "max over tokens excluding BOS (fm_core.Subject.maxpool_acts)",
             "split_seed": SPLIT_SEED, "banks": {}}
    files = {}
    for bank in banks:
        texts, rows, styles = load_bank(bank, banks_dir, cids, concept_idx)
        acts = {L: np.zeros((len(texts), D_SAE), dtype=np.float16) for L in layers}
        nb = (len(texts) + batch - 1) // batch
        for b, i in enumerate(range(0, len(texts), batch)):
            r = subject.maxpool_acts(texts[i:i + batch], layers=tuple(layers))
            for L in layers:
                a = np.asarray(r[L])
                if a.shape != (len(texts[i:i + batch]), D_SAE):
                    raise ValueError(f"subject returned shape {a.shape} for layer {L}")
                acts[L][i:i + len(a)] = a.astype(np.float16)
            if (b + 1) % 20 == 0 or b + 1 == nb:
                log(f"bank {bank}: batch {b + 1}/{nb} ({time.time() - t0:.0f}s)")
        for L in layers:
            p = os.path.join(out_dir, f"bank_acts_L{L}_{bank}.npy")
            np.save(p + ".tmp.npy", acts[L])
            os.replace(p + ".tmp.npy", p)
            files[os.path.basename(p)] = p
        bmeta["banks"][bank] = {"n": len(texts), "rows": rows, "styles": styles,
                                "texts_sha256": hashlib.sha256("\n".join(texts).encode()).hexdigest(),
                                "per_text_sha256": [hashlib.sha256(t.encode()).hexdigest()[:16] for t in texts]}
        del acts
    # consistency with the generator's cache: re-run split-A batch 0 (concept 0's first texts, exactly precompute's
    # first batch when consistency_n == 32) and compare with acts_L*_A.npy rows. Same device/dtype => identical.
    cons = {}
    if consistency_n:
        use_src_cache(src_cache)
        cs = C.load()
        A = [t for c in cs for t in c["A"]][:consistency_n]
        r = subject.maxpool_acts(A, layers=tuple(layers))
        for L in layers:
            ref = np.load(os.path.join(src_cache, f"acts_L{L}_A.npy"), mmap_mode="r")[:len(A)].astype(np.float32)
            got = np.asarray(r[L]).astype(np.float16).astype(np.float32)
            d = np.abs(got - ref)
            both, either = (got > 0) & (ref > 0), (got > 0) | (ref > 0)
            rel = d[both] / np.abs(ref[both]) if both.any() else np.zeros(1)
            cons[f"L{L}"] = {"n_texts": len(A), "max_abs_diff": float(d.max()),
                             "frac_entries_equal": float((d == 0).mean()),
                             "frac_active_entries_equal": float((d[either] == 0).mean()) if either.any() else 1.0,
                             "jaccard_active_sets": float(both.sum() / max(1, either.sum())),
                             "median_rel_diff_both_active": float(np.median(rel)),
                             "pearson_r": float(np.corrcoef(got.ravel(), ref.ravel())[0, 1])
                             if got.std() > 0 and ref.std() > 0 else None,
                             "max_activation_ref": float(ref.max())}
        log("consistency vs generator cache: " + json.dumps(cons))
    bmeta["consistency_vs_acts_A"] = cons
    bmeta["elapsed_s"] = round(time.time() - t0, 1)
    bmeta.update(extra_meta or {})
    mp = os.path.join(out_dir, "bank_meta.json")
    json.dump(bmeta, open(mp, "w"))
    files["bank_meta.json"] = mp
    man = {"what": "FeatureMatch diagnosis step 2: max-pooled SAE activations of bank F/R texts",
           "generator_summary": "tasks/featurematch/fm_core.py Subject.maxpool_acts", "git_sha": git_sha(),
           "src_cache": src_cache, "consistency_vs_acts_A": cons, "elapsed_s": bmeta["elapsed_s"],
           "files": {}}
    for name, p in sorted(files.items()):
        ent = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}
        if p.endswith(".npy"):
            a = np.load(p, mmap_mode="r")
            ent.update({"shape": list(a.shape), "dtype": str(a.dtype)})
        man["files"][name] = ent
    man.update(extra_meta or {})
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    if repo_manifest:
        json.dump(man, open(repo_manifest, "w"), indent=1)
    log(f"compute done in {time.time() - t0:.0f}s -> {out_dir}")
    return man


def run_compute(a):
    import torch
    from common import gpuq
    gpuq.apply_caps()
    from tasks.featurematch.fm_core import Subject, model_path, sae_path
    src_cache = a.src_cache or default_src_cache()
    cids = json.load(open(os.path.join(src_cache, "precompute_meta.json")))["cids"]
    if a.dry_run:
        layers = (int(a.layers.split(",")[0]) if a.layers else 12,)
        first_topic = next(i for i, c in enumerate(cids) if c.startswith("topic:"))
        concept_idx = [0, first_topic]           # one language concept + one topic concept
        out_dir = a.out_dir or os.path.join(DIAG_CACHE, "dryrun")
        device = a.device or "cpu"
        consistency_n = a.consistency_n if a.consistency_n is not None else 2
        repo_manifest = None
    else:
        layers = tuple(int(x) for x in a.layers.split(",")) if a.layers else LAYERS
        concept_idx = None
        out_dir = a.out_dir or DIAG_CACHE
        device = a.device
        consistency_n = a.consistency_n if a.consistency_n is not None else 32
        repo_manifest = REPO_MANIFEST if out_dir == DIAG_CACHE else None
    # disk preflight BEFORE taking the GPU: float16 [n_texts, 16384] per layer and bank, written via a temp file
    n_texts = (len(concept_idx) if concept_idx is not None else len(cids)) * 20
    need = int(n_texts * D_SAE * 2 * len(layers) * len(a.banks.split(",")) * 1.1) + (64 << 20)
    os.makedirs(out_dir, exist_ok=True)
    st = os.statvfs(out_dir)
    free = st.f_bavail * st.f_frsize
    if free < need:
        sys.exit(f"not enough disk for the outputs: need ~{need / 1e9:.2f} GB under {out_dir}, "
                 f"free {free / 1e9:.2f} GB. Free space first (nothing was computed).")
    t0 = time.time()
    subj = Subject(device=device, layers=layers, load_decoder=False)
    print(f"model loaded on {subj.device} in {time.time() - t0:.0f}s (layers {layers})", flush=True)
    extra = {"dry_run": bool(a.dry_run), "device": str(subj.device), "torch": torch.__version__,
             "model_path": os.path.relpath(model_path(), TASK) if os.path.isabs(model_path()) else model_path(),
             "sae": {f"L{L}": sae_path(L).split("snapshots/")[-1] for L in layers}}
    if concept_idx is not None:
        extra["concepts"] = [cids[i] for i in concept_idx]
    compute(subj, src_cache=src_cache, banks_dir=a.banks_dir, out_dir=out_dir, banks=tuple(a.banks.split(",")),
            layers=layers, concept_idx=concept_idx, batch=a.batch, consistency_n=consistency_n,
            repo_manifest=repo_manifest, extra_meta=extra, log=lambda s: print(s, flush=True))


# --------------------------------------------------------------------------------------------------------- analyze
def judge(kind, anchor, menu, aAF1, aCF2, aF2, thr=THR, robust_set="CF2", rival_rule="spec",
          null_rule="symmetric"):
    """The slot-level filter. aX = AUROC row of the slot's latent over all concepts (index = concept id).
    robust_set: which held-out AUROC is the robustness metric: CF2 (orchestrator spec), F2 (styled text only), or
      both (min of the two). rival_rule: spec = planted slots are judged on agreement + robustness only;
      symmetric = also drop planted slots where a distractor clears thr (the same test nulls get).
    null_rule: symmetric = a null slot must pass the SAME latent test as a planted slot (key agreement over
      menu+anchor, robustness of the anchor) and, in addition, no menu concept may clear thr; menu-only = only the
      latter. WHY symmetric is the default: PREREG step 2 applies the filter to latent+anchor identically for both
      kinds, so the filter cannot make null latents statistically different from planted ones (SPEC no-fingerprint)."""
    menu = [int(c) for c in menu]
    anchor = int(anchor)
    metric = {"CF2": aCF2, "F2": aF2, "both": np.minimum(aCF2, aF2)}[robust_set]
    cand = menu if kind == "planted" else menu + [anchor]
    vals = np.array([aAF1[c] for c in cand], dtype=np.float64)
    vals = np.where(np.isfinite(vals), vals, -1.0)
    ms_key = cand[int(np.argmax(vals))]
    agree = ms_key == anchor
    rob = float(metric[anchor])
    robust = bool(rob >= thr)
    others = [c for c in menu if c != anchor]
    om = np.array([metric[c] for c in others], dtype=np.float64)
    om = np.where(np.isfinite(om), om, -1.0)
    rival_i = int(np.argmax(om))
    best_other = float(om[rival_i])
    rival = best_other >= thr
    reasons = []
    if not agree:
        reasons.append("key_changed")
    if not robust:
        reasons.append("not_robust")
    if rival and kind != "planted":
        reasons.append("null_menu_separates")
    if rival and kind == "planted" and rival_rule == "symmetric":
        reasons.append("planted_rival")
    if kind != "planted" and null_rule == "menu-only":
        keep = not rival
    else:
        keep = not reasons
    return {"keep": bool(keep), "reasons": reasons, "ms_key": int(ms_key), "agree": bool(agree),
            "robust_metric": rob, "robust": robust, "best_other_metric": best_other,
            "best_other": int(others[rival_i]), "rival": bool(rival)}


class LayerStats:
    """AUROC tables [k, n_c] of selected latents (columns) of one layer on every text set the filter uses."""

    SETS = ("A", "C", "F", "F1", "F2", "AF1", "CF2")

    def __init__(self, L, cols, src_cache, diag_cache, meta, bmeta, f1, n_c):
        self.L = L
        self.cols = np.array(sorted(set(int(c) for c in cols)), dtype=np.int64)
        self.pos = {int(j): i for i, j in enumerate(self.cols)}
        ld = lambda p: np.asarray(np.load(p, mmap_mode="r")[:, self.cols], dtype=np.float32)   # noqa: E731
        XA = ld(os.path.join(src_cache, f"acts_L{L}_A.npy"))
        XC = ld(os.path.join(src_cache, f"acts_L{L}_C.npy"))
        XF = ld(os.path.join(diag_cache, f"bank_acts_L{L}_F.npy"))
        rA, rC = np.asarray(meta["rowsA"]), np.asarray(meta["rowsC"])
        rF = np.asarray(bmeta["banks"]["F"]["rows"])
        if len(rF) != len(XF):
            raise ValueError(f"bank F acts rows {len(XF)} != bank_meta rows {len(rF)}")
        data = {"A": (XA, rA), "C": (XC, rC), "F": (XF, rF), "F1": (XF[f1], rF[f1]), "F2": (XF[~f1], rF[~f1])}
        data["AF1"] = (np.vstack([XA, data["F1"][0]]), np.concatenate([rA, data["F1"][1]]))
        data["CF2"] = (np.vstack([XC, data["F2"][0]]), np.concatenate([rC, data["F2"][1]]))
        self.au = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            for s in self.SETS:
                X, r = data[s]
                self.au[s] = auroc_table(X, r, n_c) if len(self.cols) else np.zeros((0, n_c), np.float32)
        self.fire_F = self._fire(XF, rF, n_c)
        self.fire_A = self._fire(XA, rA, n_c)
        self.gen_fire_A = (XA > 0).mean(0)
        self.gen_fire_F = (XF > 0).mean(0)
        # integrity check: split-A AUROC must reproduce the generator's cached table (float16-rounded)
        cached = np.load(os.path.join(src_cache, f"auroc_L{L}_A.npy"), mmap_mode="r")[self.cols].astype(np.float32)
        self.max_diff_vs_cached_A = float(np.nanmax(np.abs(self.au["A"] - cached))) if len(self.cols) else 0.0
        if self.max_diff_vs_cached_A > 2e-3:
            raise RuntimeError(f"L{L}: split-A AUROC differs from the generator cache by {self.max_diff_vs_cached_A}; "
                               "rows/columns misaligned or a different cache")

    @staticmethod
    def _fire(X, rows, n_c):
        f = np.zeros((n_c, X.shape[1]), dtype=np.float64)
        np.add.at(f, rows, (X > 0).astype(np.float64))
        cnt = np.bincount(rows, minlength=n_c)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            return (f / cnt).T          # [k, n_c], nan for concepts without texts in X

    def row(self, s, j):
        return self.au[s][self.pos[int(j)]]


def _f(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


def describe(vals):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "median": _f(np.median(a)), "mean": _f(a.mean()), "q25": _f(np.percentile(a, 25)),
            "q75": _f(np.percentile(a, 75))}


def compare(rows, flag="keep", feats=FEATS):
    """Survivors vs dropped: per-feature distribution, Mann-Whitney p, median difference / ratio, and mixes."""
    from scipy.stats import mannwhitneyu
    g = {True: [r for r in rows if r[flag]], False: [r for r in rows if not r[flag]]}
    out = {"n_kept": len(g[True]), "n_dropped": len(g[False]), "features": {}}
    for f in feats:
        k = [r.get(f) for r in g[True]]
        d = [r.get(f) for r in g[False]]
        dk, dd = describe(k), describe(d)
        ent = {"kept": dk, "dropped": dd}
        if dk["n"] and dd["n"]:
            ka = [x for x in k if x is not None and np.isfinite(x)]
            da = [x for x in d if x is not None and np.isfinite(x)]
            try:
                ent["mannwhitney_p"] = _f(mannwhitneyu(ka, da).pvalue, 5)
            except ValueError:
                ent["mannwhitney_p"] = None
            ent["median_diff_kept_minus_dropped"] = _f(dk["median"] - dd["median"])
            ent["median_ratio_kept_over_dropped"] = _f(dk["median"] / dd["median"]) if dd["median"] else None
        out["features"][f] = ent
    for key in ("layer", "family", "kind", "tier"):
        if rows and key in rows[0]:
            out[f"{key}_mix"] = {str(v): {"kept": sum(1 for r in g[True] if r[key] == v),
                                         "dropped": sum(1 for r in g[False] if r[key] == v)}
                                 for v in sorted({r[key] for r in rows}, key=str)}
    return out


def easier_verdict(cmp_, ref=None):
    """PREREG step 2: 'the filter selects easier latents' if survivors' median margin_C exceeds the dropped
    latents' by > 0.05, OR survivors' median generic fire rate is > 1.25x the dropped latents', OR reference planted
    accuracy on survivors exceeds that on dropped by > 0.05."""
    f = cmp_["features"]
    md = f.get("margin_C", {}).get("median_diff_kept_minus_dropped")
    dr = f.get("generic_fire_A", {}).get("median_ratio_kept_over_dropped")
    parts = {"margin_C_median_diff": md, "margin_C_gt_0.05": None if md is None else md > 0.05,
             "density_median_ratio": dr, "density_ratio_gt_1.25": None if dr is None else dr > 1.25}
    if ref:
        parts.update(ref)
        rd = ref.get("ref_planted_acc_kept_minus_dropped")
        parts["ref_acc_gt_0.05"] = None if rd is None else rd > 0.05
    flags = [v for k, v in parts.items() if k.endswith(("gt_0.05", "gt_1.25")) and v is not None]
    parts["filter_selects_easier"] = bool(any(flags)) if flags else None
    return parts


def iter_instances(inst_dir):
    for d in sorted(glob.glob(os.path.join(inst_dir, "*"))):
        p = os.path.join(d, "instance.json")
        if os.path.exists(p):
            yield d, json.load(open(p))


def analyze(src_cache=None, src_instances=None, diag_cache=DIAG_CACHE, out_dir=OUT, thr=THR, robust_set="CF2",
            rival_rule="spec", null_rule="symmetric", ref_sims=1, pool=True, log=print):
    t0 = time.time()
    src_cache = src_cache or default_src_cache()
    src_instances = src_instances or default_src_instances()
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(os.path.join(src_cache, "precompute_meta.json")))
    bmeta = json.load(open(os.path.join(diag_cache, "bank_meta.json")))
    if bmeta["cids"] != meta["cids"]:
        raise ValueError("bank_meta concept order differs from the generator cache")
    cids = meta["cids"]
    n_c = len(cids)
    cid2i = {c: i for i, c in enumerate(cids)}
    fb = bmeta["banks"]["F"]
    if len(set(fb["rows"])) != n_c:
        raise ValueError(f"bank F covers {len(set(fb['rows']))} of {n_c} concepts (dry-run output?)")
    f1 = split_f1(fb["rows"], fb["styles"])
    insts = list(iter_instances(src_instances))
    if not insts:
        raise FileNotFoundError(f"no instances under {src_instances}")
    T = load_tables(src_cache) if pool else None
    cols = defaultdict(set)
    for _, inst in insts:
        for s in inst["extra"]["slots"]:
            cols[s["layer"]].add(int(s["real_latent"]))
    if pool:
        for L in LAYERS:
            cols[L] |= {j for j, _ in T.planted[L]}
    stats = {L: LayerStats(L, cols[L], src_cache, diag_cache, meta, bmeta, f1, n_c) for L in LAYERS if cols[L]}
    log(f"AUROC tables for {sum(len(v) for v in cols.values())} latents in {time.time() - t0:.0f}s")
    rate_A = {L: np.load(os.path.join(src_cache, f"firerate_L{L}_A.npy")) for L in LAYERS}
    fam = lambda c: cids[c].split(":")[0]           # noqa: E731  "lang" or "topic"

    # ---------------------------------------------------------------- per slot
    slot_rows = []
    max_menu_diff = 0.0
    for d, inst in insts:
        for si, s in enumerate(inst["extra"]["slots"]):
            L, j = int(s["layer"]), int(s["real_latent"])
            st = stats[L]
            kind = "planted" if s["kind"] == "planted" else "null"
            anchor = cid2i[s["anchor"]]
            menu = [cid2i[c] for c in s["menu"]]
            a = {k: st.row(k, j) for k in st.SETS}
            max_menu_diff = max(max_menu_diff, float(np.max(np.abs(np.array(s["menu_auroc_A"]) - a["A"][menu]))))
            v = judge(kind, anchor, menu, a["AF1"], a["CF2"], a["F2"], thr, robust_set, rival_rule, null_rule)
            others = [c for c in menu if c != anchor]
            mg = lambda arr: float(arr[anchor] - np.nanmax(arr[others]))     # noqa: E731
            best_univ = int(np.nanargmax(np.where(np.isfinite(a["AF1"]), a["AF1"], -1)))
            row = {"instance_id": inst["instance_id"], "tier": inst["tier"], "slot": si, "kind": kind, "layer": L,
                   "real_latent": j, "anchor": cids[anchor], "family": fam(anchor), "n_slots": len(inst["extra"]["slots"]),
                   "original_key": cids[anchor] if kind == "planted" else "nothing found",
                   "ms_key": cids[v["ms_key"]], "ms_key_on_menu": v["ms_key"] in menu, "agree": v["agree"],
                   "ms_best_universe_AF1": cids[best_univ], "best_universe_is_anchor": best_univ == anchor,
                   "auroc_A_anchor": _f(a["A"][anchor]), "auroc_C_anchor": _f(a["C"][anchor]),
                   "auroc_AF1_anchor": _f(a["AF1"][anchor]), "auroc_CF2_anchor": _f(a["CF2"][anchor]),
                   "auroc_F_anchor": _f(a["F"][anchor]), "auroc_F1_anchor": _f(a["F1"][anchor]),
                   "auroc_F2_anchor": _f(a["F2"][anchor]), "robust_metric": _f(v["robust_metric"]),
                   "robust": v["robust"], "best_other_menu": cids[v["best_other"]],
                   "best_other_metric": _f(v["best_other_metric"]), "rival": v["rival"],
                   "margin_A": _f(mg(a["A"])), "margin_C": _f(mg(a["C"])), "margin_CF2": _f(mg(a["CF2"])),
                   "margin_F2": _f(mg(a["F2"])),
                   "fire_A_anchor": _f(st.fire_A[st.pos[j], anchor]), "fire_F_anchor": _f(st.fire_F[st.pos[j], anchor]),
                   "generic_fire_A": _f(rate_A[L][j]), "generic_fire_F": _f(st.gen_fire_F[st.pos[j]]),
                   "keep": v["keep"], "reasons": "+".join(v["reasons"]) or ""}
            # sensitivity: the same slot under every alternative rule
            for rs in ("CF2", "F2", "both"):
                for rr in ("spec", "symmetric"):
                    row[f"keep_{rs}_{rr}"] = judge(kind, anchor, menu, a["AF1"], a["CF2"], a["F2"], thr, rs, rr,
                                                   null_rule)["keep"]
            slot_rows.append(row)
    if max_menu_diff > 2e-3:
        raise RuntimeError(f"instance menu_auroc_A differs from recomputed split-A AUROC by {max_menu_diff}")

    # ---------------------------------------------------------------- offline reference accuracy per slot (optional)
    ref_info = None
    if ref_sims:
        ref_info = reference_by_slot(insts, slot_rows, src_cache, ref_sims)

    # ---------------------------------------------------------------- per pooled latent (generator v2 pool)
    pool_rows, pool_npz = [], {}
    if pool:
        for L in LAYERS:
            ans = np.array(T.answerable[L])
            ids, aAF1, aCF2, aF2 = [], [], [], []
            for j, c in T.planted[L]:
                st = stats[L]
                a = {k: st.row(k, j) for k in ("A", "C", "AF1", "CF2", "F2", "F")}
                others = ans[ans != c]
                univ_best = int(ans[int(np.argmax(np.where(np.isfinite(a["AF1"][ans]), a["AF1"][ans], -1)))])
                metric = {"CF2": a["CF2"], "F2": a["F2"], "both": np.minimum(a["CF2"], a["F2"])}[robust_set]
                pool_rows.append({
                    "layer": L, "real_latent": int(j), "anchor": cids[c], "family": fam(c),
                    "auroc_A_anchor": _f(a["A"][c]), "auroc_C_anchor": _f(a["C"][c]),
                    "auroc_AF1_anchor": _f(a["AF1"][c]), "auroc_CF2_anchor": _f(a["CF2"][c]),
                    "auroc_F_anchor": _f(a["F"][c]), "auroc_F2_anchor": _f(a["F2"][c]),
                    "agree_universe": univ_best == c, "ms_best_answerable": cids[univ_best],
                    "robust": bool(metric[c] >= thr),
                    "keep": bool(univ_best == c and metric[c] >= thr),
                    "robust_CF2": bool(a["CF2"][c] >= thr), "robust_F2": bool(a["F2"][c] >= thr),
                    "robust_F": bool(a["F"][c] >= thr),
                    "margin_A": _f(a["A"][c] - np.nanmax(a["A"][others])),
                    "margin_C": _f(a["C"][c] - np.nanmax(a["C"][others])),
                    "fire_A_anchor": _f(st.fire_A[st.pos[j], c]), "fire_F_anchor": _f(st.fire_F[st.pos[j], c]),
                    "generic_fire_A": _f(rate_A[L][j]), "generic_fire_F": _f(st.gen_fire_F[st.pos[j]])})
                ids.append(j)
                aAF1.append(a["AF1"])
                aCF2.append(a["CF2"])
                aF2.append(a["F2"])
            pool_npz[f"L{L}_latents"] = np.array(ids, dtype=np.int64)
            for nm, arr in (("AF1", aAF1), ("CF2", aCF2), ("F2", aF2)):
                pool_npz[f"L{L}_{nm}"] = np.array(arr, dtype=np.float32).reshape(len(ids), n_c)
        os.makedirs(diag_cache, exist_ok=True)
        np.savez_compressed(os.path.join(diag_cache, "pool_auroc.npz"), **pool_npz)

    # ---------------------------------------------------------------- outputs
    params = {"thr": thr, "robust_set": robust_set, "rival_rule": rival_rule, "null_rule": null_rule,
              "split_seed": SPLIT_SEED, "src_cache": src_cache, "src_instances": src_instances,
              "diag_cache": diag_cache, "git_sha": git_sha(),
              "bank_manifest_sha256": sha256_file(os.path.join(diag_cache, "manifest.json"))
              if os.path.exists(os.path.join(diag_cache, "manifest.json")) else None}
    write_csv(os.path.join(out_dir, "slots.csv"), slot_rows)
    json.dump({"params": params, "slots": slot_rows}, open(os.path.join(out_dir, "slots.json"), "w"))
    if pool_rows:
        write_csv(os.path.join(out_dir, "pool_latents.csv"), pool_rows)
    by_inst = defaultdict(list)
    for r in slot_rows:
        by_inst[r["instance_id"]].append(r)
    filt = {"params": params, "instances": {}}
    for iid, rs in by_inst.items():
        filt["instances"][iid] = {"tier": rs[0]["tier"], "n_slots": len(rs),
                                  "kept_slots": [r["slot"] for r in rs if r["keep"]],
                                  "failed_slots": [{"slot": r["slot"], "kind": r["kind"], "reasons": r["reasons"]}
                                                   for r in rs if not r["keep"]],
                                  "all_kept": all(r["keep"] for r in rs)}
    filt["fully_kept_instances"] = sorted(i for i, v in filt["instances"].items() if v["all_kept"])
    json.dump(filt, open(os.path.join(out_dir, "filtered_instances.json"), "w"), indent=1)
    summary = summarize(slot_rows, pool_rows, params, ref_info, stats, max_menu_diff)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    log(f"analyze done in {time.time() - t0:.0f}s -> {out_dir}")
    return summary


def reference_by_slot(insts, slot_rows, src_cache, n_seeds):
    """Offline reference solver (simulate.py: split-C probes, the real solve_slot logic) per slot, n_seeds runs.
    Adds ref_acc to each slot row; returns planted accuracy for kept vs dropped slots (PREREG step 2 criterion 3)."""
    use_src_cache(src_cache)
    from tasks.featurematch import simulate
    from tasks.featurematch.grader import grade
    D = simulate.load()
    idx = {(r["instance_id"], r["slot"]): r for r in slot_rows}
    acc = defaultdict(list)
    for d, inst in insts:
        pub = json.load(open(os.path.join(d, "public.json")))
        for sd in range(n_seeds):
            sub, _ = simulate.sim_ref(inst, pub, D, {}, sd)
            g = grade(d, sub)
            for si, sl in enumerate(g["details"]["slots"]):
                acc[(inst["instance_id"], si)].append(bool(sl["correct"]))
    for k, v in acc.items():
        idx[k]["ref_acc"] = _f(np.mean(v))
    out = {"ref_sims": n_seeds}
    for kind in ("planted", "null"):
        for keep in (True, False):
            vals = [r["ref_acc"] for r in slot_rows if r["kind"] == kind and r["keep"] == keep and "ref_acc" in r]
            out[f"ref_{kind}_acc_{'kept' if keep else 'dropped'}"] = _f(np.mean(vals)) if vals else None
    k, dr = out["ref_planted_acc_kept"], out["ref_planted_acc_dropped"]
    out["ref_planted_acc_kept_minus_dropped"] = _f(k - dr) if k is not None and dr is not None else None
    return out


def write_csv(p, rows):
    keys = list(rows[0].keys()) if rows else []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize(slot_rows, pool_rows, params, ref_info, stats, max_menu_diff):
    s = {"params": params,
         "integrity": {"max_abs_diff_splitA_auroc_vs_generator_cache": {f"L{L}": _f(st.max_diff_vs_cached_A, 6)
                                                                         for L, st in stats.items()},
                       "max_abs_diff_instance_menu_auroc_A": _f(max_menu_diff, 6)}}
    sl = {"n": len(slot_rows)}
    for kind in ("planted", "null"):
        rs = [r for r in slot_rows if r["kind"] == kind]
        if not rs:
            continue
        sl[kind] = {"n": len(rs), "kept": sum(r["keep"] for r in rs), "kept_frac": _f(np.mean([r["keep"] for r in rs])),
                    "agree_frac": _f(np.mean([r["agree"] for r in rs])),
                    "robust_frac": _f(np.mean([r["robust"] for r in rs])),
                    "rival_frac": _f(np.mean([r["rival"] for r in rs])),
                    "reasons": dict(Counter(r["reasons"] or "kept" for r in rs)),
                    "key_changed_examples": [{k: r[k] for k in ("instance_id", "slot", "anchor", "ms_key",
                                                                 "auroc_AF1_anchor")}
                                             for r in rs if not r["agree"]][:20],
                    "by_layer": {str(L): _f(np.mean([r["keep"] for r in rs if r["layer"] == L]))
                                 for L in LAYERS if any(r["layer"] == L for r in rs)},
                    "by_family": {f: _f(np.mean([r["keep"] for r in rs if r["family"] == f]))
                                  for f in sorted({r["family"] for r in rs})},
                    "by_tier": {t: _f(np.mean([r["keep"] for r in rs if r["tier"] == t]))
                                for t in sorted({r["tier"] for r in rs})}}
        # how the decision moves under the alternative rules (thr fixed)
        sl[kind]["kept_frac_by_rule"] = {k[5:]: _f(np.mean([r[k] for r in rs]))
                                         for k in rs[0] if k.startswith("keep_")}
        sl[kind]["weak_style_among_kept"] = {
            "n_kept_with_auroc_F2_lt_0.75": sum(1 for r in rs if r["keep"] and (r["auroc_F2_anchor"] or 0) < 0.75),
            "n_kept_with_fire_F_lt_0.5": sum(1 for r in rs if r["keep"] and (r["fire_F_anchor"] or 0) < 0.5)}
    by_inst = defaultdict(list)
    for r in slot_rows:
        by_inst[r["instance_id"]].append(r["keep"])
    sl["instances"] = len(by_inst)
    sl["instances_all_slots_kept"] = sum(all(v) for v in by_inst.values())
    s["slots"] = sl
    s["slots_kept_vs_dropped"] = compare(slot_rows)
    s["slots_kept_vs_dropped_planted_only"] = compare([r for r in slot_rows if r["kind"] == "planted"])
    s["reference_offline"] = ref_info
    s["filter_selects_easier_slots"] = easier_verdict(s["slots_kept_vs_dropped"], ref_info)
    if pool_rows:
        s["pool"] = {"n": len(pool_rows), "kept": sum(r["keep"] for r in pool_rows),
                     "kept_frac": _f(np.mean([r["keep"] for r in pool_rows])),
                     "robust_CF2_frac": _f(np.mean([r["robust_CF2"] for r in pool_rows])),
                     "robust_F2_frac": _f(np.mean([r["robust_F2"] for r in pool_rows])),
                     "robust_F_frac (PREREG auroc_F)": _f(np.mean([r["robust_F"] for r in pool_rows])),
                     "agree_universe_frac": _f(np.mean([r["agree_universe"] for r in pool_rows])),
                     "by_layer": {str(L): _f(np.mean([r["keep"] for r in pool_rows if r["layer"] == L]))
                                  for L in LAYERS if any(r["layer"] == L for r in pool_rows)},
                     "by_family": {f: _f(np.mean([r["keep"] for r in pool_rows if r["family"] == f]))
                                   for f in sorted({r["family"] for r in pool_rows})},
                     "concepts_with_a_kept_latent": {str(L): len({r["anchor"] for r in pool_rows
                                                                  if r["layer"] == L and r["keep"]})
                                                     for L in LAYERS}}
        s["pool_kept_vs_dropped"] = compare(pool_rows, feats=["generic_fire_A", "generic_fire_F", "fire_A_anchor",
                                                              "fire_F_anchor", "margin_A", "margin_C",
                                                              "auroc_A_anchor", "auroc_C_anchor"])
        s["filter_selects_easier_pool"] = easier_verdict(s["pool_kept_vs_dropped"])
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("compute", help="GPU: bank-F/R activations (run under common.gpuq)")
    c.add_argument("--dry-run", action="store_true", help="CPU, 2 concepts, 1 layer, output under cache/dryrun/")
    c.add_argument("--layers", default="", help="comma list (default 6,12,18; dry-run: first given or 12)")
    c.add_argument("--banks", default="F,R")
    c.add_argument("--batch", type=int, default=BATCH)
    c.add_argument("--device", default=None)
    c.add_argument("--src-cache", default=None)
    c.add_argument("--banks-dir", default=BANKS)
    c.add_argument("--out-dir", default=None)
    c.add_argument("--consistency-n", type=int, default=None)
    z = sub.add_parser("analyze", help="CPU: per-slot / per-latent filter verdicts + comparison")
    z.add_argument("--src-cache", default=None)
    z.add_argument("--src-instances", default=None)
    z.add_argument("--diag-cache", default=DIAG_CACHE)
    z.add_argument("--out-dir", default=OUT)
    z.add_argument("--thr", type=float, default=THR)
    z.add_argument("--robust-set", choices=("CF2", "F2", "both"), default="CF2")
    z.add_argument("--rival-rule", choices=("spec", "symmetric"), default="spec")
    z.add_argument("--null-rule", choices=("symmetric", "menu-only"), default="symmetric")
    z.add_argument("--ref-sims", type=int, default=1, help="offline reference runs per instance (0 = skip)")
    z.add_argument("--no-pool", action="store_true", help="skip the per-latent pool analysis (and pool_auroc.npz)")
    a = ap.parse_args()
    if a.mode == "compute":
        run_compute(a)
    else:
        s = analyze(a.src_cache, a.src_instances, a.diag_cache, a.out_dir, a.thr, a.robust_set, a.rival_rule,
                    a.null_rule, a.ref_sims, not a.no_pool)
        print(json.dumps({"slots": {k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("n", "kept",
                                                                                                "kept_frac")})
                                    for k, v in s["slots"].items()},
                          "pool": {k: s.get("pool", {}).get(k) for k in ("n", "kept", "kept_frac", "by_layer")},
                          "easier": s["filter_selects_easier_slots"]}, indent=1))


if __name__ == "__main__":
    main()
