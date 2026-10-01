"""No-fingerprinting check: can PUBLIC slot features (layer + menu composition) predict planted vs null?

Generates slots in memory (seeds 500000+, never written), computes menu-structure features from public information
only, and reports 5-fold cross-validated AUROC of a gradient-boosted classifier per tier. ~0.5 = no fingerprint.
Run: $PY -m tasks.featurematch.fingerprint_check [--n 600]
"""
import argparse
import json
from collections import Counter

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from tasks.featurematch.generate import TIERS, Tables, make_instance


def feats(T, layer, menu_labels, lab2c):
    idx = [lab2c[o] for o in menu_labels]
    cl = T.close[np.ix_(idx, idx)].astype(float)
    np.fill_diagonal(cl, np.nan)
    paths = [T.paths[c] for c in idx]
    l2 = Counter(tuple(p[:3]) for p in paths)
    l1 = Counter(tuple(p[:2]) for p in paths)
    rowmean = np.nanmean(cl, 1)
    return [layer, sum(p[0] == "Language" for p in paths), len(l1), len(l2), max(l2.values()), max(l1.values()),
            float(np.nanmean(cl)), float(rowmean.max()), float(rowmean.max() - np.median(rowmean)),
            float(np.sort(rowmean)[-1] - np.sort(rowmean)[-2])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    T = Tables()
    lab2c = {c["label"]: i for i, c in enumerate(T.cs)}
    out = {}
    for tier in TIERS:
        X, y = [], []
        for k in range(a.n):
            _, inst, pub = make_instance(T, 500000 + k, tier)
            for s, ans in zip(pub["slots"], inst["answer"]["slots"]):
                X.append(feats(T, s["layer"], s["options"], lab2c))
                y.append(int(ans["planted"]))
        X, y = np.array(X), np.array(y)
        auc = cross_val_score(GradientBoostingClassifier(), X, y, cv=5, scoring="roc_auc")
        out[tier] = {"n_slots": len(y), "null_frac": round(1 - y.mean(), 3), "cv_auroc": round(float(auc.mean()), 3),
                     "cv_auroc_sd": round(float(auc.std()), 3)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
