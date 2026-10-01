# Task: what do these SAE latents represent?

You are studying the language model **{public.model}** together with a set of sparse autoencoders (SAEs):
{public.sae}. An SAE re-expresses the model's internal activations (the residual stream after a given layer) as a
sum of sparse "latents"; each latent is a direction that is active on some inputs and zero on most others.

This episode has **{public.n_slots} slots**. Each slot names one latent (an SAE layer and a latent index) and a
numbered list of 20 candidate concepts. For each slot, decide which candidate concept the latent encodes, meaning
the concept whose texts reliably activate this latent more than other text does. For some slots none of the 20
candidates is what the latent encodes; for those, the correct answer is "nothing found".

Latent indices in this episode are specific to this episode; they do not correspond to the index numbering used
anywhere else.

{public.slots_text}

## Tools

Call tools with `./tool <name> '<json args>'` from your working directory (`./tool help` lists them,
`./tool budget` shows your remaining budget). You may write and run your own analysis code with `./py`.
Work only inside your working directory.

{tool_docs}

Budget for this episode: {caps}

## Answer format

Submit exactly once, with one entry per slot:

```
./tool submit '{"answers": [{"slot": 0, "choice": 7}, {"slot": 1, "choice": "nothing found"}, ...]}'
```

`choice` is the option number (1-20) from that slot's list, or the string `"nothing found"` when none of the
20 candidates is what the latent encodes. The episode ends after a valid submission.
