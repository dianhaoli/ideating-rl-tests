"""Generate instances of the harness demo task (no GPU, no ML: a fixture for harness tests).

The hidden "model" is a lookup table W[x] = (a*x + b) mod P for x in 0..P-1. In a PLANTED
instance exactly one entry was edited: W[x*] = y* != (a*x* + b) mod P. In a NULL instance nothing
was edited. The agent must report the edited input and its new output, or "nothing found".
White-box access (the `weights` tool) makes this easy; black-box access (`query`, capped) does not.

    python tasks/_demo/generate.py --out tasks/_demo/instances --n 8 --seed 0 [--null-frac 0.5]
"""
import argparse
import json
import os
import random

P = 211


def make(out_root, seed, planted):
    rng = random.Random(seed)
    a, b = rng.randrange(2, P), rng.randrange(P)
    table = [(a * x + b) % P for x in range(P)]
    x_star = rng.randrange(P)
    y_star = (table[x_star] + rng.randrange(1, P)) % P
    if planted:
        table[x_star] = y_star
    iid = "dm-%08x" % rng.getrandbits(32)          # random id: does not encode planted/null
    canary = "RLCANARY-_demo-%016x" % rng.getrandbits(64)
    tag = "dmtag-%012x" % rng.getrandbits(48)       # a hidden label used only to exercise the leak scanner
    d = os.path.join(out_root, iid)
    os.makedirs(d, exist_ok=True)
    inst = {"instance_id": iid, "task": "_demo", "tier": "T1", "dial": {"P": P}, "seed": seed, "canary": canary,
            "planted": planted, "answer": {"edited_input": x_star, "new_output": y_star} if planted else {},
            # identical shape for planted and null (no fingerprinting)
            "leak_strings": [tag, "edit@%d=%d" % (x_star, y_star)],
            "caps": {"tool_calls": 20, "forward": 40, "generate": 0, "gradient": 0,
                     "wall_clock_s": 600, "call_timeout_s": 30},
            "extra": {"tag": tag}}
    with open(os.path.join(d, "instance.json"), "w") as f:
        json.dump(inst, f, indent=1)
    with open(os.path.join(d, "public.json"), "w") as f:
        json.dump({"modulus": P, "n_inputs": P}, f)
    with open(os.path.join(d, "table.json"), "w") as f:   # private artifact; carries the canary
        json.dump({"canary": canary, "table": table}, f)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "instances"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null-frac", type=float, default=0.5)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    for i in range(a.n):
        print(make(a.out, a.seed * 100003 + i, rng.random() >= a.null_frac))


if __name__ == "__main__":
    main()
