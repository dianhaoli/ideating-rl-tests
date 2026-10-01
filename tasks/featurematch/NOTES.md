# FeatureMatch lab notebook (append-only; timestamps from `date -u`)

## 2026-10-01 15:06 UTC — kickoff and design choices
- Read CONTEXT, PLAN_PROMPT, PLAN_V2_DELTA, BUILDER_GUIDE, HARNESS_API, DECISIONS. common/toolserver etc. are not
  on main yet (no READY file), so tools.py imports them with a local fallback and gates run in-process (local_shim.py).
- **Concept universe** (no LLM API, so no generated text): two public labelled HF datasets.
  * DBPedia_Classes (Wikipedia abstracts, 3-level ontology l1>l2>l3): each l3 class (e.g. Swimmer < Athlete < Agent)
    is a TOPIC concept. WHY: the ontology gives real taxonomy siblings for close distractors (27 athlete types,
    8 animal classes, 7 politician types ...), which the spec asks for.
  * papluca/language-identification: 19 LANGUAGE concepts (English dropped: every DBPedia text is English, so an
    "English" concept would be confounded with genre). Language families give siblings (Romance, Slavic, ...).
- **Splits**: per concept 40 held-out texts (split A, generator-only: selects latents + ground truth) and 24 probe
  texts (split B, the reference solver's own corpus, standing in for texts an LLM agent would write). WHY: the
  answer must be defined on data the agent can never query, otherwise the agent could score menu options on the
  grading data itself.
- **Held-out cleanliness check** (concepts.py): global near-duplicate removal (a text seen in two concepts is
  dropped from both), A/B disjointness asserted, and a bag-of-words classifier trained on B must reach AUROC >= 0.9
  on A per concept. Result: 238 candidate concepts, 6 dropped for too few clean texts (<64), 0 failed the label
  check, **232 kept (19 languages, 213 topics)**. Details: cache/concepts_validation.json (regenerable).
- **Per-text latent summary** = max activation over the text's tokens, excluding BOS (BOS has huge norm and
  spurious activations). Texts truncated to 64 tokens everywhere (generator, tools, solver) so they agree.
- First precompute attempt was killed by the gpuq RAM cap (8.9 > 8 GB) while loading gemma-2-2b on CPU first.
  Fix: load with device_map=cuda directly.
