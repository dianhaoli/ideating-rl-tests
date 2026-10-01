"""Black-box control for FeatureMatch: same interface and caps as the reference solver, `blackbox` profile
(task_info, generate, next_token_logits only; no SAE or activation access). Gate: <= 10% of episodes.

The best a black-box policy can do is exploit behaviour or priors: it cannot observe the latent at all, so the
slot's latent index carries no information for it. This control asks the subject model itself to pick an option
(the "behavioural lookup" shortcut from CONTEXT.md) via next-token logits over the option numbers, and calls
"nothing found" when the model's preference is weak (top option probability below the uniform-ish level).
"""
import json
import os

PROMPT = ("Sparse autoencoder latent {latent} at layer {layer} of a language model is most active on text about "
          "one of these concepts:\n{opts}\nThe concept is number")


def solve(call, seed=0):
    info = call("task_info")
    answers = []
    for s in info["slots"]:
        opts = "\n".join(f"{i}. {o}" for i, o in enumerate(s["options"], 1))
        r = call("next_token_logits", prompts=[PROMPT.format(latent=s["latent"], layer=s["layer"], opts=opts)],
                 top_k=20)["results"][0]
        choice = "nothing found"
        import math
        for tok, lp in zip(r["top_tokens"], r["logprobs"]):
            t = tok.strip()
            if t.isdigit() and 1 <= int(t) <= len(s["options"]):
                if math.exp(lp) >= 0.15:
                    choice = int(t)
                break
        answers.append({"slot": s["slot"], "choice": choice})
    call("submit", answer={"answers": answers})
    return {"answers": answers}


def main():
    from common.toolclient import Client
    from tasks.featurematch.reference_solver import unwrap
    c = Client(os.environ["RL_EPISODE"])
    print(json.dumps(solve(unwrap(c.call), seed=int(os.environ.get("RL_SEED", "0")))))


if __name__ == "__main__":
    main()
