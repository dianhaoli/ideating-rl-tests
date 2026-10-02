"""Step-3 skeptic, part 3: the probe texts of the NEW cheap recipes (fixed here, before any of them was run).

Every recipe is a fixed script: the same texts for an option label in every episode, no look at the latent beyond
its max activation on these texts, no knowledge of the concepts beyond the option label (and, for languages, the
fixed 3-sentence bank that audit/attack_solvers.py's template_probe already uses), no reasoning.

Recipes (texts per option):
  generic_label  1 text: "Here is something about <label>."  (the option label exactly as shown, e.g. "Here is
                 something about article about a noble.")
  generic_bare   1 text: "Here is something about a noble." / languages: "Here is something written in Thai."
  style3         3 short templated sentences in 3 styles (text message, local news, Q&A) with the bare name for
                 topics; languages: template_probe's fixed 3-sentence bank in that language
  encyc4_max     template_probe's own 4 encyclopedia templates (languages: its 3-sentence bank), but max-pick by mean
                 instead of template_probe's AUROC-vs-background >= 0.78 rule (shows what the threshold costs)
  gen_all        the subject model writes the texts (`generate`, greedy, 48 new tokens) from self_probe's 2 fixed
                 prompts, for EVERY option (self_probe only does it for a template shortlist of 3). Over the
                 40-call generate cap in most episodes, so descriptive; gen1_cap is the in-budget version
  gen1_cap       1 completion (prompt 1) per distinct option label, in slot/option order, until the episode's
                 generate cap (40) is spent; an option with no completion cannot be claimed
  name3          (offline from the generator's names cache) max over the 3 name forms (label, bare name,
                 "This is about <name>.")
  tmpl7_posthoc  (added AFTER the results above were seen; no new texts) encyc4 + style3 per topic option (7 texts),
                 languages: the 3-sentence bank
Each recipe has a max-pick variant (claim the option with the highest mean max activation; ties -> lowest option)
and a "-cal" variant whose claim threshold is calibrated on a disjoint calibration pool (see cheap_recipes.py).
"""
from tasks.featurematch.audit.attack_solvers import BACKGROUND, LANG_BANK, bare, gen_prompts, template_texts

STYLE3 = [
    "hey did you see that thing about {a} {x}? we were just talking about it lol",
    "Local news: residents gathered on Saturday to hear a short talk about {a} {x}, organisers said.",
    "Q: Can you tell me something interesting about {a} {x}? A: Sure, here is a fun fact.",
]
GEN_MAX_NEW = 48


def art(x):
    return "an" if x[0].lower() in "aeiou" else "a"


def texts_for(label):
    """{kind: [texts]} for one option label (concept)."""
    x, is_lang = bare(label)
    out = {"generic_label": [f"Here is something about {label}."]}
    if is_lang:
        lb = list(LANG_BANK[x])
        out["generic_bare"] = [f"Here is something written in {x}."]
        out["style3"] = lb
        out["encyc4"] = lb
    else:
        out["generic_bare"] = [f"Here is something about {art(x)} {x}."]
        out["style3"] = [t.format(a=art(x), x=x) for t in STYLE3]
        out["encyc4"] = template_texts(label)
    out["gen_prompts"] = gen_prompts(label)
    return out


RECIPE_KINDS = {"generic_label": "generic_label", "generic_bare": "generic_bare", "style3": "style3",
                "encyc4_max": "encyc4", "gen_all": "gen"}
BACKGROUND = list(BACKGROUND)
