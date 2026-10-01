"""Procedural instance generator for FeatureMatch (privileged side).

An instance (= one episode) has 3-5 SLOTS. Each slot names one SAE latent (layer + agent-space index) and a menu of
20 candidate concepts; the agent answers one option or "nothing found" per slot.

Ground truth per slot (computed here from the HELD-OUT split A, which no tool can read):
  Every slot, planted or null, starts the same way: draw a layer, draw an ANCHOR concept c* uniformly from the
  concepts that own at least one pooled latent at that layer, and draw one of c*'s pooled latents. The latent pool
  (Tables.pools) holds latents that are selective for one concept: held-out AUROC(c*) >= P_THR, fires on >= FIRE_MIN
  of c*'s held-out texts, replicates on the independent probe split B (AUROC_B >= P_THR_B; the per-entity behaviour
  validation of plan v2), at most MAX_SPECIFIC concepts within MARGIN of the best, and the latent does NOT fire on
  any of the 3 surface variants of c*'s own name (token-identity filter, see below).
  Both kinds then remove the same "confusable" set X = {c : AUROC_A(c) >= N_THR or AUROC_B(c) >= N_THR_B} (this
  includes c*) from the distractor candidates, and build the menu by the same procedure around a centre drawn from
  c*'s sibling group:
  * planted slot: menu = c* + 19 distractors. Answer = c*. Every distractor is < N_THR on A and < N_THR_B on B, so
    the answer wins by >= P_THR - N_THR = 0.25 on A and >= 0.13 on B.
  * null slot ("near-miss"): menu = 20 distractors (c* left out). Answer = "nothing found". The latent is selective
    for a concept, but not for any option (e.g. a Spanish latent offered Portuguese, Italian ... but not Spanish).
  WHY one pool and one exclusion set (audit MAJOR 2, 2026-10-01): the earlier design also used "off-universe" nulls
  (live latents not selective for any concept). Their firing statistics on generic text differed from the
  concept-selective latents, so "measure the latent on 64 random texts" separated null from planted slots
  (AUROC 0.70-0.80) without any concept reasoning. Now the latent of a null slot is drawn from exactly the same
  distribution as the latent of a planted slot, so no statistic of the latent alone can tell them apart; only
  testing the OPTIONS can. The planted/null difference in the menu is only "c* is present or not".
Token-identity filter (warning 3; audit MAJOR 3): the old generator rejected a planted MENU when the answer's name was
the top name-probe activation in that menu. That removed only rank 1, so the answer's name was often exactly rank 2,
a learnable artifact, and it was applied to planted slots only. Now the filter acts on LATENTS, for every slot kind:
a latent that fires on any of its own concept's 3 name variants (menu label, bare name, "This is about <name>.") never
enters the pool. On every slot the answer's name is therefore silent, like most distractor names.
Menu universe per layer: distractors are drawn only from the concepts that own a pooled latent at that layer (the
"answerable" set). WHY: otherwise a learned policy could rule out labels that can never be the answer at that layer.

Dials (recorded per instance under "dial"):
  closeness  "far" (T1): 19 distractors uniform from the layer's answerable set; "close" (T2): taxonomy siblings first (same
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
MARGIN_B, N_THR_B = 0.08, 0.72   # the same verdict must hold on the independent split B (robust ground truth)
GEN_VERSION = 2           # 2 = audit fixes (one latent pool for planted and null, latent-level name filter, symmetric
                          # exclusion set, per-layer menu universe); 1 = stage-1/2 design (git history <= 6847c88)
MAX_SPECIFIC = 3          # at most this many concepts within MARGIN of the best (otherwise the latent is generic)
N_OPT = 20
NOISE = 1.5               # close tier: random jitter added to taxonomy closeness (levels) when ranking menu candidates
P_NULL = 0.4              # per-slot null probability (independent per slot => randomised null pattern)
TIERS = {
    "T1": {"closeness": "far", "caps": {"tool_calls": 150, "forward": 1200, "generate": 40, "gradient": 0}},
    "T2": {"closeness": "close", "caps": {"tool_calls": 150, "forward": 1200, "generate": 40, "gradient": 0}},
    # T3: close distractors AND a tight forward budget (Pando-style query budget dial): ~110 probe texts per slot
    "T3": {"closeness": "close", "caps": {"tool_calls": 150, "forward": 550, "generate": 40, "gradient": 0}},
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
        # only concepts with a split-C probe set can be menu options or anchors (the reference must be able to probe
        # every option); the others still serve as AUROC negatives
        self.eligible = np.array([bool(c["C"]) for c in self.cs])
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
        """Latent pool per layer, shared by planted AND null slots (see module docstring)."""
        self.planted, self.planted_by_c, self.answerable, self.pool_stats = {}, {}, {}, {}
        for L in LAYERS:
            A, B, F = self.aA[L], self.aB[L], self.fire[L]
            best = A.argmax(1)
            bA = A[np.arange(len(A)), best]
            bB = B[np.arange(len(B)), best]
            bF = F[np.arange(len(F)), best]
            n_near = (A >= (bA - MARGIN)[:, None]).sum(1)
            ok = (bA >= P_THR) & (bB >= P_THR_B) & (bF >= FIRE_MIN) & (n_near <= MAX_SPECIFIC) & self.eligible[best]
            # latent-level token-identity filter: the latent must stay silent on all 3 variants of its own concept's name
            own_name = self.names[L].reshape(self.n_c, 3, -1)[best, :, np.arange(len(best))].max(1)
            name_ok = own_name <= 0
            self.pool_stats[L] = {"selective": int(ok.sum()), "kept_after_name_filter": int((ok & name_ok).sum())}
            ok &= name_ok
            self.planted[L] = [(int(j), int(best[j])) for j in np.nonzero(ok)[0]]
            # index by concept: slots sample a CONCEPT uniformly first, then one of its latents. WHY: some concepts
            # own hundreds of selective latents (formulaic texts such as solar-eclipse articles own >2000), so
            # sampling latents uniformly would make "guess the most popular concept" a working prior.
            byc = {}
            for j, c in self.planted[L]:
                byc.setdefault(c, []).append(j)
            self.planted_by_c[L] = byc
            # menu universe at this layer = concepts that can be the answer here (all eligible by construction)
            self.answerable[L] = np.array(sorted(byc))
            self.pool_stats[L]["answerable_concepts"] = len(byc)

    def naive_pick(self, L, j, menu):
        """Option the naive name-probe recipe would pick under each surface variant (None if all zero)."""
        picks = []
        for v in range(3):
            s = np.array([self.names[L][3 * c + v, j] for c in menu])
            picks.append(menu[int(s.argmax())] if s.max() > 0 else None)
        return picks

    def center_for(self, rng, anchor, L):
        """Menu centre: a random eligible concept from the anchor's sibling group (same DBPedia l2 class / same
        language family), possibly the anchor itself. WHY: if the menu were always centred on the answer, a planted
        menu would contain its centre and a null menu would not, which a learned policy can detect (measured:
        cross-validated AUROC 0.74 for predicting null from menu structure alone in the close tier, see NOTES)."""
        depth = len(self.paths[anchor]) - 1
        sib = [int(c) for c in self.answerable[L] if self.close[anchor, c] >= depth]
        return int(rng.choice(sib)) if sib else anchor

    def menu_around(self, rng, center, exclude, closeness, k, L):
        pool = np.array([c for c in self.answerable[L] if c not in exclude])
        if closeness == "far":
            return [int(x) for x in rng.choice(pool, size=k, replace=False)]
        # closest to the centre first, with noise so that sibling groups are sometimes incomplete in BOTH planted and
        # null menus (a missing sibling must not signal "the answer was removed")
        key = -self.close[center, pool] + rng.random(len(pool)) * NOISE
        return [int(x) for x in pool[np.argsort(key)[:k]]]


def make_slot(T, rng, closeness, null, used):
    """Draw the layer, then the anchor CONCEPT c* uniformly over the layer's answerable concepts, then one of c*'s
    pooled latents. Planted and null slots share every step; they differ only in whether c* is put in the menu.
    There is no rejection step that depends on the slot kind, so no selection bias separates the two kinds."""
    L = int(rng.choice(LAYERS))
    byc = T.planted_by_c[L]
    keys = sorted(byc)
    kind = "near_miss" if null else "planted"
    for attempt in range(200):
        cstar = keys[int(rng.integers(len(keys)))]
        j = byc[cstar][int(rng.integers(len(byc[cstar])))]
        if (L, j) in used:
            continue
        A = T.aA[L][j]
        Bv = T.aB[L][j]
        excl = set(np.nonzero((A >= N_THR) | (Bv >= N_THR_B))[0].tolist()) | {cstar}
        center = T.center_for(rng, cstar, L)
        if kind == "planted":
            menu = [cstar] + T.menu_around(rng, center, excl, closeness, N_OPT - 1, L)
        else:
            menu = T.menu_around(rng, center, excl, closeness, N_OPT, L)
        menu = [menu[i] for i in rng.permutation(len(menu))]
        aucs = [float(A[c]) for c in menu]
        srt = sorted(aucs, reverse=True)
        if kind == "planted":
            assert menu[int(np.argmax(aucs))] == cstar and srt[0] >= P_THR and srt[1] < N_THR
            assert max(Bv[c] for c in menu if c != cstar) < N_THR_B <= P_THR_B <= Bv[cstar]
            choice = menu.index(cstar) + 1
        else:
            assert srt[0] < N_THR and max(Bv[c] for c in menu) < N_THR_B
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
                                 "FIRE_MIN": FIRE_MIN, "MARGIN_B": MARGIN_B, "N_THR_B": N_THR_B},
                  "gen_version": GEN_VERSION},
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
    ap.add_argument("--prior", type=int, default=0,
                    help="instead of writing instances, build prior.json (planted-answer label frequencies) from this "
                         "many in-memory instances on seeds 900000+ (disjoint from the evaluation pool)")
    a = ap.parse_args()
    T = Tables()
    if a.prior:
        from collections import Counter
        cnt = Counter()
        for k in range(a.prior):
            for tier in ("T1", "T2"):
                _, inst, public = make_instance(T, 900000 + k, tier)
                for s, pub in zip(inst["answer"]["slots"], public["slots"]):
                    if s["planted"]:
                        cnt[pub["options"][s["choice"] - 1]] += 1
        tot = sum(cnt.values())
        json.dump({lab: round(n / tot, 5) for lab, n in cnt.most_common()}, open(os.path.join(HERE, "prior.json"), "w"),
                  indent=0)
        print("prior over", tot, "planted slots; top:", cnt.most_common(5))
        return
    print(json.dumps(T.pool_stats))
    if a.clean and os.path.exists(INST):
        shutil.rmtree(INST)
    man = []
    mpath = os.path.join(HERE, "instances_manifest.json")
    tiers = a.tiers.split(",")
    if not a.clean and os.path.exists(mpath):      # keep manifest entries of tiers not regenerated now
        man = [m for m in json.load(open(mpath))["instances"] if m["tier"] not in tiers]
    for tier in tiers:
        ti = sorted(TIERS).index(tier)
        for k in range(a.n_per_tier):
            seed = a.start_seed + 100000 * ti + k
            iid, inst, public = make_instance(T, seed, tier)
            d = write(iid, inst, public)
            man.append({"instance_id": iid, "tier": tier, "seed": seed, "dial": inst["dial"],
                        "files": {f: sha(os.path.join(d, f)) for f in ("instance.json", "public.json")}})
    with open(mpath, "w") as f:
        json.dump({"generator": "tasks/featurematch/generate.py", "gen_version": GEN_VERSION,
                   "concepts_sha256": json.load(open(os.path.join(C.CACHE, "concepts_validation.json")))["sha256"],
                   "instances": man}, f, indent=1)
    print("wrote", len(man), "instances")


if __name__ == "__main__":
    main()
