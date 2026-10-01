"""Reference solver for FeatureMatch: proves each kept instance is solvable through the agent's tools and caps.

It never reads instance files. Everything comes through `call(tool_name, **args)`:
  1. `task_info` gives the public slots (layer, latent index, 20 option labels), exactly what the agent's TASK.md says.
  2. For each slot it runs probe texts through `latent_activations` (per_token=False -> max over tokens):
     K texts for each of the 20 options plus BG "background" texts on other subjects. The probe texts come from its
     own corpus (split B of the concept sets; disjoint from the held-out split A that defines ground truth). This
     stands in for the probe texts an LLM agent would write itself.
  3. For each option it estimates AUROC = P(an option text out-activates a non-option text) against all other probes,
     claims the best option if its AUROC >= TAU and it beats the runner-up by >= GAP, else answers "nothing found".
This automates the standard human practice of explaining an SAE latent by checking its activation on concept probes
(SAEBench / auto-interp "detection" scoring), with a calibrated "none of these".

Harness entry point: main() uses common.toolclient.Client(os.environ["RL_EPISODE"]).call.
"""
import json
import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
K, BG, TAU, GAP = 6, 32, 0.80, 0.05
BATCH = 16


def load_corpus():
    """label -> probe texts (split B). Privileged file, but contains no instance information."""
    cs = json.load(open(os.path.join(HERE, "cache", "concepts.json")))
    return {c["label"]: c["B"] for c in cs}


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def decide(option_acts, bg_acts, tau=TAU, gap=GAP):
    """option_acts: list (per option) of activation lists. Returns 1-based option or 'nothing found'."""
    aucs = []
    for i, pos in enumerate(option_acts):
        neg = [v for j, o in enumerate(option_acts) if j != i for v in o] + list(bg_acts)
        aucs.append(auroc(pos, neg))
    order = np.argsort(aucs)[::-1]
    best, second = aucs[order[0]], aucs[order[1]]
    if best >= tau and best - second >= gap:
        return int(order[0]) + 1, aucs
    return "nothing found", aucs


def probe(call, layer, latent, texts):
    vals = []
    for i in range(0, len(texts), BATCH):
        r = call("latent_activations", texts=texts[i:i + BATCH], layer=layer, latents=[latent], per_token=False)
        vals += [row["max"][str(latent)] for row in r["results"]]
    return vals


def solve(call, k=K, bg=BG, tau=TAU, gap=GAP, seed=0, corpus=None):
    corpus = corpus or load_corpus()
    rng = random.Random(seed)
    info = call("task_info")
    answers, diag = [], []
    for s in info["slots"]:
        opts = s["options"]
        texts, owner = [], []
        for i, o in enumerate(opts):
            pool = corpus.get(o, [])
            for t in rng.sample(pool, min(k, len(pool))):
                texts.append(t)
                owner.append(i)
        others = [lab for lab in corpus if lab not in set(opts)]
        for lab in rng.sample(others, bg):
            texts.append(rng.choice(corpus[lab]))
            owner.append(-1)
        vals = probe(call, s["layer"], s["latent"], texts)
        option_acts = [[v for v, w in zip(vals, owner) if w == i] for i in range(len(opts))]
        bg_acts = [v for v, w in zip(vals, owner) if w == -1]
        choice, aucs = decide(option_acts, bg_acts, tau, gap)
        answers.append({"slot": s["slot"], "choice": choice})
        diag.append({"slot": s["slot"], "choice": choice, "best_auc": round(max(aucs), 3)})
    call("submit", answer={"answers": answers})
    return {"answers": answers, "diag": diag}


def unwrap(call):
    """Accept either a raw-result call or one returning the broker envelope {"ok": .., "result"/"error": ..}."""
    def f(name, **args):
        r = call(name, **args)
        if isinstance(r, dict) and "ok" in r and ("result" in r or "error" in r):
            if not r["ok"]:
                raise RuntimeError(r.get("error", "tool error"))
            return r["result"]
        return r
    return f


def main():
    from common.toolclient import Client
    c = Client(os.environ["RL_EPISODE"])
    print(json.dumps(solve(unwrap(c.call), seed=int(os.environ.get("RL_SEED", "0")))))


if __name__ == "__main__":
    main()
