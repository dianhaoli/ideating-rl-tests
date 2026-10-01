"""Build the labelled concept text sets for FeatureMatch (privileged side, CPU only).

Sources (public HF datasets, downloaded into HF_HOME, never committed):
  * DBPedia_Classes (DeveloperOats/DBPedia_Classes, test + val CSVs): Wikipedia abstracts labelled with a
    3-level ontology (l1 > l2 > l3). Each l3 class (e.g. "Swimmer", under l2 "Athlete", under l1 "Agent")
    is one TOPIC concept. The ontology gives us taxonomy siblings for close distractors for free.
  * papluca/language-identification (valid + test CSVs): short sentences/reviews in 20 languages. Each
    language except English is one LANGUAGE concept (English is dropped because every DBPedia text is
    English, so "English" would be confounded with "not a review"). Language families give siblings.

Per concept we keep three DISJOINT splits:
  A  "held-out"  (N_A texts): used only by generate.py to pick latents and compute ground truth. No tool can
                 read these texts, so an agent cannot score candidate answers on the grading data.
  B  "validation" (N_B texts): generator-only second opinion. A slot is kept only if its verdict (planted answer
                 or "nothing found") also holds on B, so the ground truth does not hinge on one sample of texts.
  C  "probe"     (N_C texts): the reference solver's own probe corpus, standing in for the probe texts an LLM agent
                 would write itself. Disjoint from A and B, so the reference is NOT tuned on the data that selected
                 or validated the slot. Concepts with fewer than N_A+N_B+N_C clean texts get no C split and are never
                 used as menu options (they still count as negatives in the AUROC universe).

Validation ("is the held-out split clean?"), logged to cache/concepts_validation.json:
  1. exact/near duplicate removal inside a concept and across concepts (first 60 normalised chars);
  2. zero overlap between A and B of the same concept;
  3. a label-consistency check: a bag-of-words classifier trained on split B must reach held-out AUROC >= 0.9 on
     split A for that concept. A concept that fails is dropped, because if words alone cannot tell its held-out
     texts apart, its labels are too noisy (or the class too vague) to serve as ground truth.

Run:  $PY -m tasks.featurematch.concepts      -> tasks/featurematch/cache/concepts.json
"""
import hashlib
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(CACHE, "concepts.json")
N_A, N_B = 40, 24
N_C = 20          # split C: the reference solver's probe corpus (only concepts with >= N_A+N_B+N_C clean texts)
SEED = 20261001
MAX_CHARS = 400

LANG_NAMES = {"ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek", "es": "Spanish", "fr": "French",
              "hi": "Hindi", "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese",
              "ru": "Russian", "sw": "Swahili", "th": "Thai", "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese",
              "zh": "Chinese"}
LANG_FAMILY = {"es": "Romance", "fr": "Romance", "it": "Romance", "pt": "Romance", "de": "Germanic", "nl": "Germanic",
               "ru": "Slavic", "bg": "Slavic", "pl": "Slavic", "hi": "Indo-Aryan", "ur": "Indo-Aryan",
               "ar": "Afro-Asiatic", "el": "Hellenic", "ja": "East Asian", "zh": "East Asian", "th": "Southeast Asian",
               "vi": "Southeast Asian", "tr": "Turkic", "sw": "Niger-Congo"}

# human-readable labels for DBPedia l3 classes whose CamelCase split reads badly
DB_FIX = {"Medician": "physician", "Mollusca": "mollusc", "SupremeCourtOfTheUnitedStatesCase": "US Supreme Court case",
          "NCAATeamSeason": "NCAA team season", "WomensTennisAssociationTournament": "WTA tennis tournament",
          "NationalFootballLeagueSeason": "NFL season", "MixedMartialArtsEvent": "mixed martial arts event",
          "FormulaOneRacer": "Formula One racer", "NascarDriver": "NASCAR driver", "GaelicGamesPlayer": "Gaelic games player",
          "AustralianRulesFootballPlayer": "Australian rules football player", "AnimangaCharacter": "anime/manga character",
          "HollywoodCartoon": "Hollywood cartoon", "PlayboyPlaymate": "Playboy Playmate",
          "EurovisionSongContestEntry": "Eurovision Song Contest entry", "BiologicalDatabase": "biological database",
          "ArtistDiscography": "artist discography", "ClassicalMusicComposition": "classical music composition",
          "ClassicalMusicArtist": "classical music artist", "Religious": "religious figure", "OfficeHolder": "office holder",
          "MemberOfParliament": "member of parliament", "ChristianBishop": "Christian bishop",
          "AmericanFootballPlayer": "American football player", "AustralianFootballTeam": "Australian football team",
          "CanadianFootballTeam": "Canadian football team", "SportsTeamMember": "sports team member",
          "GreenAlga": "green alga", "CultivatedVariety": "cultivated plant variety", "TelevisionStation": "television station",
          "BroadcastNetwork": "broadcast network", "PublicTransitSystem": "public transit system"}


def db_label(c):
    if c in DB_FIX:
        return DB_FIX[c]
    return re.sub(r"(?<!^)(?=[A-Z])", " ", c).lower()


def norm_key(t):
    return re.sub(r"\W+", " ", t.lower()).strip()[:60]


def _load_sources():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    lang = pd.concat([pd.read_csv(hf_hub_download("papluca/language-identification", f, repo_type="dataset"))
                      for f in ("valid.csv", "test.csv")])
    db = pd.concat([pd.read_csv(hf_hub_download("DeveloperOats/DBPedia_Classes", f, repo_type="dataset"))
                    for f in ("DBPEDIA_test.csv", "DBPEDIA_val.csv")])
    return lang, db


def build():
    lang, db = _load_sources()
    rng = np.random.default_rng(SEED)
    raw = []   # (cid, label, source, path, texts)
    for code, name in LANG_NAMES.items():
        texts = lang[lang.labels == code].text.astype(str).tolist()
        raw.append((f"lang:{code}", f"text written in {name}", "language", ["Language", LANG_FAMILY[code], name], texts))
    for (l1, l2, l3), g in db.groupby(["l1", "l2", "l3"]):
        raw.append((f"topic:{l3}", f"article about a {db_label(l3)}" if not (db_label(l3)[0].lower() in "aeiou"
                                                                    and not db_label(l3).startswith(("US", "Eu", "uni")))
                    else f"article about an {db_label(l3)}", "topic", ["Topic", l1, l2, l3], g.text.astype(str).tolist()))
    # global near-duplicate removal: a key seen in two concepts is dropped from both
    owner = {}
    for cid, _, _, _, texts in raw:
        for t in texts:
            owner.setdefault(norm_key(t), set()).add(cid)
    concepts, dropped = [], {}
    for cid, label, src, path, texts in raw:
        seen, clean = set(), []
        for t in texts:
            t = " ".join(t.split())[:MAX_CHARS]
            k = norm_key(t)
            if len(t) < 40 or k in seen or len(owner.get(k, ())) > 1:
                continue
            seen.add(k)
            clean.append(t)
        if len(clean) < N_A + N_B:
            dropped[cid] = f"only {len(clean)} clean texts (< {N_A + N_B})"
            continue
        idx = rng.permutation(len(clean))
        A = [clean[i] for i in idx[:N_A]]
        B = [clean[i] for i in idx[N_A:N_A + N_B]]
        Cs = [clean[i] for i in idx[N_A + N_B:N_A + N_B + N_C]] if len(clean) >= N_A + N_B + N_C else []
        assert not ({norm_key(t) for t in A} & {norm_key(t) for t in B})
        assert not ({norm_key(t) for t in A} & {norm_key(t) for t in Cs})
        concepts.append({"cid": cid, "label": label, "source": src, "path": path, "A": A, "B": B, "C": Cs})
    return concepts, dropped


def label_check(concepts):
    """Bag-of-words one-vs-rest classifier trained on B, AUROC on A, per concept."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy.sparse import hstack
    XB = [t for c in concepts for t in c["B"]]
    yB = np.array([i for i, c in enumerate(concepts) for _ in c["B"]])
    XA = [t for c in concepts for t in c["A"]]
    yA = np.array([i for i, c in enumerate(concepts) for _ in c["A"]])
    v1 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=200000, sublinear_tf=True)
    v2 = TfidfVectorizer(analyzer="word", min_df=1, sublinear_tf=True)
    MB = hstack([v1.fit_transform(XB), v2.fit_transform(XB)]).tocsr()
    MA = hstack([v1.transform(XA), v2.transform(XA)]).tocsr()
    clf = LogisticRegression(C=10.0, max_iter=300)
    clf.fit(MB, yB)
    P = clf.predict_proba(MA)
    aucs = {}
    for i, c in enumerate(concepts):
        aucs[c["cid"]] = float(roc_auc_score(yA == i, P[:, i]))
    return aucs


def main():
    os.makedirs(CACHE, exist_ok=True)
    concepts, dropped = build()
    aucs = label_check(concepts)
    kept, failed = [], {}
    for c in concepts:
        if aucs[c["cid"]] >= 0.9:
            kept.append(c)
        else:
            failed[c["cid"]] = round(aucs[c["cid"]], 3)
    blob = json.dumps(kept, sort_keys=True).encode()
    with open(OUT, "w") as f:
        f.write(blob.decode())
    val = {"n_candidates": len(concepts) + len(dropped), "dropped_too_few": dropped,
           "dropped_label_check_auroc_lt_0.9": failed, "n_kept": len(kept),
           "n_kept_by_source": {s: sum(c["source"] == s for c in kept) for s in ("language", "topic")},
           "n_menu_eligible_with_C": sum(bool(c["C"]) for c in kept),
           "bow_auroc": {k: round(v, 4) for k, v in aucs.items()},
           "sha256": hashlib.sha256(blob).hexdigest()}
    with open(os.path.join(CACHE, "concepts_validation.json"), "w") as f:
        json.dump(val, f, indent=1)
    print(json.dumps({k: v for k, v in val.items() if k != "bow_auroc"}, indent=1))


def load():
    with open(OUT) as f:
        return json.load(f)


if __name__ == "__main__":
    main()
