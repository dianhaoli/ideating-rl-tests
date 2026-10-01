"""Procedural instance generator for FeatureMatch (privileged side).

An instance (= one episode) has 3-5 SLOTS. Each slot names one SAE latent (layer + agent-space index) and a menu of
20 candidate concepts; the agent answers one option or "nothing found" per slot.

Ground truth per slot (computed here from the HELD-OUT split A, which no tool can read):
  * planted slot: the menu contains the latent's best concept c*, with held-out AUROC(c*) >= P_THR, the latent fires
    on >= FIRE_MIN of c*'s held-out texts, the association replicates on the independent probe split B
    (AUROC_B(c*) >= P_THR_B; this is the per-entity behaviour validation of plan v2), and every other menu option
    is at least MARGIN below c*. Answer = c*.
  * null slot: every menu option has held-out AUROC < N_THR. Answer = "nothing found". Two kinds, mixed:
      - near-miss: a concept-selective latent whose own concept (and anything else >= N_THR) is left OUT of a menu
        built around that concept's taxonomy neighbourhood (e.g. a Spanish latent with Portuguese/Italian/... but
        no Spanish). WHY: forces the agent to check that an option really explains the latent, not just that it is
        the best of the menu.
      - off-universe: a live latent that is not selective for any of the 232 concepts (e.g. a syntax feature),
        with a menu built around a random anchor concept.
  The gap between N_THR and P_THR keeps both answers unambiguous (shortcut warning 4).
Planted and null slots are built by the SAME menu procedure around an anchor concept, so menus look alike.

Shortcut filter (warning 3): a planted slot is rejected if the naive recipe "encode each option's name, pick the
max activation" picks c* under ANY of three surface variants (menu label, bare name, "This is about <name>.").

Dials (recorded per instance under "dial"):
  closeness  "far" (T1): 19 distractors uniform from the universe; "close" (T2): taxonomy siblings first (same
             DBPedia l2 class / same language family, then same l1, then the rest). EXPERIMENTAL dial.
  n_slots    3-5 (drawn per instance; precedent: number of simultaneous targets, MEMIT-style scaling).
  forward cap  per-tier forward-pass budget (precedent: Pando query budget).

Run: $PY -m tasks.featurematch.generate --n-per-tier 60 [--start-seed 1000]
"""
import argparse
import hashlib
import json
import os
import shutil

import numpy as np

from tasks.featurematch import concepts as C
from tasks.featurematch.fm_core import LAYERS

HERE = os.path.dirname(os.path.abspath(__file__))
INST = os.path.join(HERE, "instances")
P_THR, P_THR_B, N_THR, MARGIN, FIRE_MIN = 0.90, 0.85, 0.65, 0.10, 0.5
MAX_SPECIFIC = 3          # at most this many concepts within MARGIN of the best (otherwise the latent is generic)
N_OPT = 20
P_NULL = 0.4              # per-slot null probability (independent per slot => randomised null pattern)
P_NEAR_MISS = 0.6         # share of null slots that are near-miss (rest off-universe)
TIERS = {
    "T1": {"closeness": "far", "caps": {"tool_calls": 150, "forward": 1200, "generate": 40, "gradient": 0}},
    "T2": {"closeness": "close", "caps": {"tool_calls": 150, "forward": 1200, "generate": 40, "gradient": 0}},
}


class Tables:
    def __init__(self):
        self.cs = C.load()
        meta = json.load(open(os.path.join(C.CACHE, "precompute_meta.json")))
        assert meta["cids"] == [c["cid"] for c in self.cs]
        self.n_c = len(self.cs)
        self.aA = {L: np.load(os.path.join(C.CACHE, f"auroc_L{L}_A.npy")).astype(np.float32) for L in LAYERS}
        self.aB = {L: np.load(os.path.join(C.CACHE, f"auroc_L{L}_B.npy")).astype(np.float32) for L in LAYERS}
        self.fire = {L: np.load(os.path.join(C.CACHE, f"fire_L{L}_A.npy")).astype(np.float32) for L in LAYERS}
        self.rate = {L: np.load(os.path.join(C.CACHE, f"firerate_L{L}_A.npy")) for L in LAYERS}
        self.names = {L: np.load(os.path.join(C.CACHE, f"acts_L{L}_names.npy")).astype(np.float32) for L in LAYERS}
        self.paths = [c["path"] for c in self.cs]
        # taxonomy closeness = length of shared path prefix
        P = self.paths
        self.close = np.array([[self._prefix(P[i], P[j]) for j in range(self.n_c)] for i in range(self.n_c)])
        self.pools()

    @staticmethod
    def _prefix(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def pools(self):
        self.planted, self.offu = {}, {}
        for L in LAYERS:
            A, B, F = self.aA[L], self.aB[L], self.fire[L]
            best = A.argmax(1)
            bA = A[np.arange(len(A)), best]
            bB = B[np.arange(len(B)), best]
            bF = F[np.arange(len(F)), best]
            n_near = (A >= (bA - MARGIN)[:, None]).sum(1)
            ok = (bA >= P_THR) & (bB >= P_THR_B) & (bF >= FIRE_MIN) & (n_near <= MAX_SPECIFIC)
            self.planted[L] = [(int(j), int(best[j])) for j in np.nonzero(ok)[0]]
            off = (A.max(1) < N_THR) & (B.max(1) < N_THR + 0.05) & (self.rate[L] >= 0.01)
            self.offu[L] = [int(j) for j in np.nonzero(off)[0]]

    def naive_pick(self, L, j, menu):
        """Option the naive name-probe recipe would pick under each surface variant (None if all zero)."""
        picks = []
        for v in range(3):
            s = np.array([self.names[L][3 * c + v, j] for c in menu])
            picks.append(menu[int(s.argmax())] if s.max() > 0 else None)
        return picks

    def menu_around(self, rng, anchor, exclude, closeness, k):
        pool = np.array([c for c in range(self.n_c) if c not in exclude])
        if closeness == "far":
            return [int(x) for x in rng.choice(pool, size=k, replace=False)]
        key = -self.close[anchor, pool] + rng.random(len(pool)) * 0.5   # closest first, random tie-break
        return [int(x) for x in pool[np.argsort(key)[:k]]]


def make_slot(T, rng, closeness, null, used):
    L = int(rng.choice(LAYERS))
    for _ in range(500):
        if not null or rng.random() < P_NEAR_MISS:
            j, cstar = T.planted[L][int(rng.integers(len(T.planted[L])))]
            kind = "planted" if not null else "near_miss"
        else:
            j = T.offu[L][int(rng.integers(len(T.offu[L])))]
            cstar = int(rng.integers(T.n_c))
            kind = "off_universe"
        if (L, j) in used:
            continue
        A = T.aA[L][j]
        if kind == "planted":
            excl = set(np.nonzero(A >= A[cstar] - MARGIN)[0].tolist())
            menu = [cstar] + T.menu_around(rng, cstar, excl | {cstar}, closeness, N_OPT - 1)
            if any(p == cstar for p in T.naive_pick(L, j, menu)):
                continue                                   # token-identity shortcut works here: reject
        else:
            excl = set(np.nonzero(A >= N_THR)[0].tolist()) | {cstar}
            menu = T.menu_around(rng, cstar, excl, closeness, N_OPT)
        menu = [menu[i] for i in rng.permutation(len(menu))]
        aucs = [float(A[c]) for c in menu]
        srt = sorted(aucs, reverse=True)
        if kind == "planted":
            assert menu[int(np.argmax(aucs))] == cstar and srt[0] - srt[1] >= MARGIN - 1e-6
            choice = menu.index(cstar) + 1
        else:
            assert srt[0] < N_THR
            choice = "nothing found"
        used.add((L, j))
        return {"layer": L, "real_latent": j, "kind": kind, "anchor": T.cs[cstar]["cid"], "menu": menu,
                "menu_auroc_A": [round(a, 4) for a in aucs], "choice": choice,
                "naive_picks": [None if p is None else T.cs[p]["cid"] for p in T.naive_pick(L, j, menu)]}
    raise RuntimeError("could not build slot")


def make_instance(T, seed, tier):
    rng = np.random.default_rng([seed, 7])
    cfg = TIERS[tier]
    n_slots = int(rng.choice([3, 4, 5]))
    nulls = rng.random(n_slots) < P_NULL
    used = set()
    slots = [make_slot(T, rng, cfg["closeness"], bool(nulls[i]), used) for i in range(n_slots)]
    perm_seed = int(rng.integers(2**62))
    from tasks.featurematch.tools import perm_for
    inv = {L: np.argsort(perm_for(perm_seed, L)) for L in LAYERS}
    suffix = hashlib.sha256(f"fm-{seed}-{tier}-{perm_seed}".encode()).hexdigest()[:10]
    iid = f"fm-{tier.lower()}-{suffix}"
    canary = "RLCANARY-featurematch-" + hashlib.sha256(f"canary-{iid}".encode()).hexdigest()[:16]
    labels = [c["label"] for c in T.cs]
    public = {
        "model": "google/gemma-2-2b",
        "sae": "Gemma Scope residual-stream SAEs (google/gemma-scope-2b-pt-res), width 16384, layers 6, 12 and 18",
        "n_slots": n_slots,
        "slots": [{"slot": i, "layer": s["layer"], "latent": int(inv[s["layer"]][s["real_latent"]]),
                   "options": [labels[c] for c in s["menu"]]} for i, s in enumerate(slots)],
    }
    public["slots_text"] = render_slots(public)
    inst = {
        "instance_id": iid, "task": "featurematch", "tier": tier,
        "dial": {"closeness": cfg["closeness"], "n_slots": n_slots, "forward_cap": cfg["caps"]["forward"]},
        "seed": seed, "canary": canary,
        "planted": bool(any(not x for x in nulls)),       # true if ANY slot is planted (multi-slot episode, D7)
        "answer": {"slots": [{"planted": s["kind"] == "planted", "choice": s["choice"],
                              "concept": s["anchor"] if s["kind"] == "planted" else None} for s in slots]},
        "leak_strings": [f"{perm_seed:x}"],
        "caps": dict(cfg["caps"]),
        "extra": {"perm_seed": perm_seed,
                  "slots": [{k: (s[k] if k != "menu" else [T.cs[c]["cid"] for c in s["menu"]]) for k in s}
                            for s in slots],
                  "thresholds": {"P_THR": P_THR, "P_THR_B": P_THR_B, "N_THR": N_THR, "MARGIN": MARGIN,
                                 "FIRE_MIN": FIRE_MIN}},
    }
    return iid, inst, public


def render_slots(public):
    lines = []
    for s in public["slots"]:
        lines.append(f"Slot {s['slot']}: layer {s['layer']}, latent {s['latent']}")
        for i, o in enumerate(s["options"], 1):
            lines.append(f"  {i:2d}. {o}")
        lines.append("")
    return "\n".join(lines).rstrip()


def write(iid, inst, public):
    d = os.path.join(INST, iid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "instance.json"), "w") as f:
        json.dump(inst, f, indent=1)
    with open(os.path.join(d, "public.json"), "w") as f:
        json.dump(public, f, indent=1)
    return d


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-tier", type=int, default=60)
    ap.add_argument("--start-seed", type=int, default=1000)
    ap.add_argument("--tiers", default="T1,T2")
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    T = Tables()
    print({L: (len(T.planted[L]), len(T.offu[L])) for L in LAYERS})
    if a.clean and os.path.exists(INST):
        shutil.rmtree(INST)
    man = []
    for ti, tier in enumerate(a.tiers.split(",")):
        for k in range(a.n_per_tier):
            seed = a.start_seed + 100000 * ti + k
            iid, inst, public = make_instance(T, seed, tier)
            d = write(iid, inst, public)
            man.append({"instance_id": iid, "tier": tier, "seed": seed, "dial": inst["dial"],
                        "files": {f: sha(os.path.join(d, f)) for f in ("instance.json", "public.json")}})
    with open(os.path.join(HERE, "instances_manifest.json"), "w") as f:
        json.dump({"generator": "tasks/featurematch/generate.py",
                   "concepts_sha256": json.load(open(os.path.join(C.CACHE, "concepts_validation.json")))["sha256"],
                   "instances": man}, f, indent=1)
    print("wrote", len(man), "instances")


if __name__ == "__main__":
    main()
