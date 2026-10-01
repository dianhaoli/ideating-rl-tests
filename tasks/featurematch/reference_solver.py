"""Reference solver for FeatureMatch: proves each kept instance is solvable through the agent's tools and caps.

It never reads instance files. Everything comes through `call(tool_name, **args)`:
  1. `task_info` gives the public slots (layer, latent index, 20 option labels), exactly what the agent's TASK.md says.
  2. For each slot it runs probe texts through `latent_activations` (per_token=False -> max over tokens):
     K texts for each of the 20 options plus BG "background" texts on other subjects. The probe texts come from its
     own corpus (split C of the concept sets; disjoint from split A, which defines ground truth, and split B,
     which validates it). This
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
K1, K2, TOP, BG, TAU, GAP = 5, 15, 3, 32, 0.78, 0.0
BATCH = 16


def load_corpus():
    """label -> probe texts (split C). Privileged file, but contains no instance information, and split C was not
    used to select or validate any slot."""
    cs = json.load(open(os.path.join(HERE, "cache", "concepts.json")))
    return {c["label"]: c["C"] for c in cs if c["C"]}


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def decide(option_acts, bg_acts, tau=TAU, gap=GAP, bg_only=True):
    """option_acts: list (per option) of activation lists. Returns (1-based option or 'nothing found', aucs)."""
    aucs = option_aucs(option_acts, bg_acts, bg_only)
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


def option_aucs(option_acts, bg_acts, bg_only=True):
    """AUROC of each option's texts against negatives. bg_only=True uses only the background texts (a random
    sample of many subjects), which estimates the same quantity the ground truth uses (option vs everything)."""
    aucs = []
    for i, pos in enumerate(option_acts):
        neg = list(bg_acts) if bg_only else [v for j, o in enumerate(option_acts) if j != i for v in o] + list(bg_acts)
        aucs.append(auroc(pos, neg))
    return aucs


def solve_slot(measure, opts, corpus, rng, k1=K1, k2=K2, top=TOP, bg1=BG, bg2=BG, tau=TAU, gap=GAP,
               bg_only=True):
    """Two-stage probing for one slot. measure(texts) -> list of max activations.
    Stage 1: k1 texts per option + bg1 background texts. Stage 2: k2 more texts for the `top` best options and
    bg2 more background texts, then decide on all collected data."""
    pools = {i: rng.sample(corpus[o], len(corpus[o])) for i, o in enumerate(opts)}
    others = [lab for lab in corpus if lab not in set(opts)]
    bg_texts = [rng.choice(corpus[lab]) for lab in rng.sample(others, bg1 + bg2)]
    texts, owner = [], []
    for i in range(len(opts)):
        texts += pools[i][:k1]
        owner += [i] * len(pools[i][:k1])
    texts += bg_texts[:bg1]
    owner += [-1] * bg1
    vals = measure(texts)
    acts = [[v for v, w in zip(vals, owner) if w == i] for i in range(len(opts))]
    bg = [v for v, w in zip(vals, owner) if w == -1]
    aucs = option_aucs(acts, bg, bg_only)
    best = list(np.argsort(aucs)[::-1][:top])
    texts2, owner2 = [], []
    for i in best:
        extra = pools[i][k1:k1 + k2]
        texts2 += extra
        owner2 += [int(i)] * len(extra)
    texts2 += bg_texts[bg1:]
    owner2 += [-1] * bg2
    vals2 = measure(texts2)
    for v, w in zip(vals2, owner2):
        (bg if w == -1 else acts[w]).append(v)
    return decide(acts, bg, tau, gap, bg_only)


def solve(call, seed=0, corpus=None, **kw):
    corpus = corpus or load_corpus()
    rng = random.Random(seed)
    info = call("task_info")
    answers, diag = [], []
    for s in info["slots"]:
        choice, aucs = solve_slot(lambda t: probe(call, s["layer"], s["latent"], t), s["options"], corpus, rng, **kw)
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
