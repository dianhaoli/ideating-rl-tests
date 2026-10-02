"""A small synthetic FeatureMatch world for CPU tests of the style filter (no model, no real data).

It writes a complete generator source cache (concepts.json, precompute_meta.json, acts/auroc/fire tables with the
real width 16384 and the real 3 layers), bank-F/R chunk files in the real format, and v2 instances made by the real
generator (generate.make_instance). Activations come from `World.row(text, layer)`, which a stub Subject also uses,
so the compute path (batching, IO, manifest) and the analyze path see the same numbers.

Latent types planted per concept and layer (ground truth for the tests; PREREG A1.1 keep rule):
  R  robust:        fires on the concept's texts in every split and every style            -> KEPT
  S  style-fragile: fires on dataset splits A/B/C and only the encyclopedia-style bank text -> not_style_robust
  X  C-weak:        fires on A/B and all bank texts, but on only half of split-C texts: AUROC_C ~0.75 < 0.85 while
                    AUROC_F2 ~0.97 (pooled C+F2 would pass 0.85; the two SEPARATE thresholds fail) -> not_style_robust
  K  key-shift:     fires on the concept's dataset texts and only on the SECOND bank text of each style (F2); it also
                    fires on the partner's FIRST bank text of each style (F1, strongly) and on 40% of the partner's
                    A/B texts. Equal-weight M = 0.5 A + 0.5 F1 prefers the partner; pooling A (40 texts) with F1 (10)
                    would still prefer c*. Exercises the equal weighting and the first/second F1/F2 split
                                                                                         -> ms_key_differs only
  P  rival:         like R, plus fires (more strongly) on the partner concept's split-C and bank texts (not A/B), so
                    the partner passes the generator's menu exclusion and wins H = 0.5 C + 0.5 F2
                                                                                         -> heldout_key_differs only
  N  name-firing:   like R but also fires on the concept's name probes (generator's name filter removes it)
  W  weak:          fires on 40% of the concept's texts (never pooled)
"""
import hashlib
import json
import os

import numpy as np

D = 16384
LAYERS = (6, 12, 18)
STYLES_F = ["casual social-media post", "news-wire report", "two-person dialogue", "listicle item", "forum question",
            "product or event review", "diary entry", "encyclopedia-style sentence", "school essay sentence",
            "podcast transcript snippet"]
ENC = STYLES_F.index("encyclopedia-style sentence")
STYLES_R = ["text message", "press release", "interview Q&A", "how-to tip", "headline plus lede",
            "personal anecdote told aloud", "trivia question", "email to a friend", "museum or exhibit placard",
            "sports-radio commentary"]
TYPES = ("R1", "R2", "S1", "S2", "X", "K", "P", "N", "W")


class World:
    def __init__(self, n_lang=4, n_topic=20, nA=40, nB=8, nC=10, n_bg=300, seed=0):
        self.cs = []
        for k in range(n_lang):
            self.cs.append({"cid": f"lang:l{k}", "label": f"text written in Lang{k}", "source": "language",
                            "path": ["Language", "FamA" if k < 2 else "FamB", f"Lang{k}"]})
        for k in range(n_topic):
            self.cs.append({"cid": f"topic:Thing{k}", "label": f"article about a thing{k}", "source": "dbpedia",
                            "path": ["Agent", f"Group{k // 5}", f"Thing{k}"]})
        self.n_c = len(self.cs)
        for i, c in enumerate(self.cs):
            for sp, n in (("A", nA), ("B", nB), ("C", nC)):
                c[sp] = [f"SYN {i} {sp} {t}" for t in range(n)]
        rng = np.random.default_rng(seed)
        self.lat = {}       # (L, type, concept) -> column
        self.type_of = {}   # (L, column) -> (type, concept)
        self.bg, self.bg_p = {}, {}
        for L in LAYERS:
            cols = rng.choice(D, size=self.n_c * len(TYPES) + n_bg, replace=False)
            for i in range(self.n_c):
                for t, ty in enumerate(TYPES):
                    j = int(cols[i * len(TYPES) + t])
                    self.lat[(L, ty, i)] = j
                    self.type_of[(L, j)] = (ty, i)
            self.bg[L] = cols[self.n_c * len(TYPES):]
            self.bg_p[L] = rng.uniform(0.01, 0.4, size=n_bg)

    def partner(self, i):
        """Same taxonomy group neighbour (so it can land on close menus too)."""
        g = [k for k, c in enumerate(self.cs) if c["path"][:2] == self.cs[i]["path"][:2]]
        return g[(g.index(i) + 1) % len(g)]

    def names(self):
        return [f"SYN {i} names {v}" for i in range(self.n_c) for v in range(3)]

    def row(self, text, L):
        p = text.split()
        assert p[0] == "SYN", text
        ci, split = int(p[1]), p[2]
        style = int(p[3]) if split in ("F", "R") else None
        rep = int(p[4]) if split in ("F", "R") else None          # 0 = first text of the style (F1), 1 = second (F2)
        r = np.random.default_rng(int(hashlib.sha256(f"{L}|{text}".encode()).hexdigest()[:15], 16))
        v = np.zeros(D, dtype=np.float32)
        if split == "names":
            v[self.lat[(L, "N", ci)]] = 2.0
            return v
        fire = r.random(len(self.bg[L])) < self.bg_p[L]
        v[self.bg[L][fire]] = r.uniform(0.2, 1.5, size=int(fire.sum()))

        def put(ty, c, prob, lo=3.0, hi=6.0):
            if r.random() < prob:
                v[self.lat[(L, ty, c)]] = r.uniform(lo, hi)
        dataset = split in ("A", "B", "C")
        for ty in ("R1", "R2", "N"):
            put(ty, ci, 0.95)
        put("P", ci, 1.0)          # P must clear 0.85 on C and F2 although the partner's texts rank above it
        if dataset or (split == "F" and style == ENC):
            put("S1", ci, 0.95)
            put("S2", ci, 0.95)
        if split != "C" or int(p[3]) % 2 == 0:               # X: only half of split C (deterministic)
            put("X", ci, 1.0 if split == "C" else 0.95)
        if dataset or rep == 1:
            put("K", ci, 1.0)
        put("W", ci, 0.40)
        # latents whose partner is ci: P fires strongly on ci's split-C and bank texts; K on half of ci's A/B texts
        # and strongly on ci's FIRST bank text of each style
        for k in range(self.n_c):
            if self.partner(k) != ci or k == ci:
                continue
            if split in ("C", "F", "R"):
                put("P", k, 1.0, 7.0, 9.0)
            if split in ("A", "B"):
                put("K", k, 0.4)
            if split == "F" and rep == 0:
                put("K", k, 1.0, 7.0, 9.0)
        return v

    def acts(self, texts, L):
        return np.stack([self.row(t, L) for t in texts]).astype(np.float16)

    # ------------------------------------------------------------------ files
    def write_src_cache(self, cache):
        os.makedirs(cache, exist_ok=True)
        json.dump(self.cs, open(os.path.join(cache, "concepts.json"), "w"))
        json.dump({"sha256": "synthetic"}, open(os.path.join(cache, "concepts_validation.json"), "w"))
        meta = {"rows" + sp: [i for i, c in enumerate(self.cs) for _ in c[sp]] for sp in "ABC"}
        meta.update({"names": self.names(), "cids": [c["cid"] for c in self.cs], "sanity": {}})
        json.dump(meta, open(os.path.join(cache, "precompute_meta.json"), "w"))
        for L in LAYERS:
            for sp in "ABC":
                np.save(os.path.join(cache, f"acts_L{L}_{sp}.npy"),
                        self.acts([t for c in self.cs for t in c[sp]], L))
            np.save(os.path.join(cache, f"acts_L{L}_names.npy"), self.acts(self.names(), L))

    def write_banks(self, banks_dir, chunk=12):
        for bank, styles in (("F", STYLES_F), ("R", STYLES_R)):
            os.makedirs(os.path.join(banks_dir, bank), exist_ok=True)
            for s in range(0, self.n_c, chunk):
                e = min(self.n_c, s + chunk)
                con = {self.cs[i]["cid"]: [{"style": st, "text": f"SYN {i} {bank}  {k}   {rep}\n"}
                                           for k, st in enumerate(styles) for rep in range(2)]
                       for i in range(s, e)}
                json.dump({"bank": bank, "concepts": con},
                          open(os.path.join(banks_dir, bank, f"chunk_{s}_{e}.json"), "w"))


class StubSubject:
    """Stands in for fm_core.Subject in compute(): same maxpool_acts signature, numbers from World.row."""

    def __init__(self, world):
        self.w = world
        self.batches = []

    def maxpool_acts(self, texts, layers=None, latents=None):
        self.batches.append(len(texts))
        return {L: self.w.acts(texts, L) for L in layers}
