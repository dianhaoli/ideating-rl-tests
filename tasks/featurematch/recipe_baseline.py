"""Zero-effort and fixed-recipe baselines for FeatureMatch. Each must pass <= 10% of episodes.

Usage (harness): RL_EPISODE=<E> python tasks/featurematch/recipe_baseline.py <variant>
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
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ["nothing", "always_claim", "prior", "prior_or_none", "random", "name_probe", "name_probe_thr"]


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
    info = call("task_info")
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
        else:
            raise SystemExit(f"unknown variant {variant}; choose from {VARIANTS}")
        answers.append({"slot": s["slot"], "choice": c})
    call("submit", answer={"answers": answers})
    return {"answers": answers}


def main():
    from common.toolclient import Client
    from tasks.featurematch.reference_solver import unwrap
    variant = sys.argv[1] if len(sys.argv) > 1 else "nothing"
    c = Client(os.environ["RL_EPISODE"])
    print(json.dumps(solve(unwrap(c.call), variant, seed=int(os.environ.get("RL_SEED", "0")))))


if __name__ == "__main__":
    main()
