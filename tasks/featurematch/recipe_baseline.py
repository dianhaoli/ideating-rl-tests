"""Zero-effort and fixed-recipe baselines for FeatureMatch. Each must pass <= 10% of episodes.

Usage (harness): RL_RECIPE=<variant> python -m common.sandbox run-scripted --solver tasks/featurematch/recipe_baseline.py
                 --solver-label recipe_<variant> ...   (run-scripted passes only --episode E, so the variant comes
                 from the RL_RECIPE environment variable, which the solver subprocess inherits)
       direct:   python tasks/featurematch/recipe_baseline.py --episode E <variant>
Variants:
  nothing          "nothing found" on every slot (the null action)
  always_claim     option 1 on every slot (submit-something-everywhere; options are shuffled, so any fixed number
                   is equivalent)
  prior            "most popular concept": on every slot claim the option that was most often the right answer in
                   a disjoint pool of generated instances (prior.json; seeds disjoint from the evaluation pool)
  prior_or_none    as prior, but answer "nothing found" when no option has a prior above the median prior
  random           uniform random option, or "nothing found" with probability 0.4 (the null rate)
  name_probe       the naive token-identity recipe (shortcut warning 3): run each option's bare name through the
                   latent and claim the max
  name_probe_thr   as name_probe, but "nothing found" if no name activates the latent at all
  vocab_match      fixed logit-lens recipe: project the latent's decoder direction to the vocabulary and claim the option
                   whose name words share a stem with the top tokens; "nothing found" if none does
"""
import json
import os
import random
import sys

from tasks.featurematch.reference_solver import slots_from_task_md

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ["nothing", "always_claim", "prior", "prior_or_none", "random", "name_probe", "name_probe_thr",
            "vocab_match"]
MODEL_FREE = {"nothing", "always_claim", "prior", "prior_or_none", "random"}
STOP = {"article", "about", "text", "written", "the", "and", "for", "player", "team", "season", "event"}


def stems(label):
    return [w[:5] for w in bare(label).lower().replace("/", " ").split() if len(w) >= 3 and w not in STOP]


def vocab_score(label, tokens):
    toks = [t.strip().lower() for t in tokens if len(t.strip()) >= 3]
    return sum(any(t.startswith(st) or (len(t) >= 4 and st.startswith(t[:5])) for t in toks) for st in stems(label))


def bare(label):
    for p in ("article about an ", "article about a ", "text written in "):
        if label.startswith(p):
            return label[len(p):]
    return label


def load_prior():
    p = os.path.join(HERE, "prior.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def solve(call, variant, seed=0):
    rng = random.Random(seed)
    # Model-free variants read the slots from TASK.md (exactly what an agent is shown) so they never start a tool
    # server; the others need the model anyway and use task_info.
    info = (slots_from_task_md() if variant in MODEL_FREE else None) or call("task_info")
    prior = load_prior() if variant.startswith("prior") else {}
    med = sorted(prior.values())[len(prior) // 2] if prior else 0
    answers = []
    for s in info["slots"]:
        opts = s["options"]
        if variant == "nothing":
            c = "nothing found"
        elif variant == "always_claim":
            c = 1
        elif variant in ("prior", "prior_or_none"):
            sc = [prior.get(o, 0.0) for o in opts]
            best = max(range(len(opts)), key=lambda i: sc[i])
            c = best + 1
            if variant == "prior_or_none" and sc[best] <= med:
                c = "nothing found"
        elif variant == "random":
            c = "nothing found" if rng.random() < 0.4 else rng.randint(1, len(opts))
        elif variant in ("name_probe", "name_probe_thr"):
            r = call("latent_activations", texts=[bare(o) for o in opts[:16]], layer=s["layer"],
                     latents=[s["latent"]], per_token=False)["results"]
            r += call("latent_activations", texts=[bare(o) for o in opts[16:]], layer=s["layer"],
                      latents=[s["latent"]], per_token=False)["results"]
            v = [row["max"][str(s["latent"])] for row in r]
            best = max(range(len(v)), key=lambda i: v[i])
            c = best + 1
            if variant == "name_probe_thr" and v[best] <= 0:
                c = "nothing found"
        elif variant == "vocab_match":
            r = call("vocab_projection", layer=s["layer"], latent=s["latent"], k=25)
            sc = [vocab_score(o, r["top_tokens"]) for o in opts]
            best = max(range(len(opts)), key=lambda i: sc[i])
            c = best + 1 if sc[best] > 0 else "nothing found"
        else:
            raise SystemExit(f"unknown variant {variant}; choose from {VARIANTS}")
        answers.append({"slot": s["slot"], "choice": c})
    call("submit", answers=answers)
    return {"answers": answers}


def main():
    from common.toolclient import episode_from_argv
    from tasks.featurematch.reference_solver import client, episode_seed, unwrap
    ep = episode_from_argv()
    args = [a for i, a in enumerate(sys.argv[1:], 1) if a != "--episode" and sys.argv[i - 1] != "--episode"]
    variant = args[0] if args else os.environ.get("RL_RECIPE", "")
    if variant not in VARIANTS:
        raise SystemExit(f"unknown recipe variant {variant!r}; set RL_RECIPE to one of {VARIANTS}")
    print(json.dumps(solve(unwrap(client(ep).call), variant, seed=episode_seed(ep))))


if __name__ == "__main__":
    main()
