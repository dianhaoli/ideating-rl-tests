# Task: what do these SAE latents represent?

You are studying the language model **{public.model}** together with a set of sparse autoencoders (SAEs):
{public.sae}. An SAE re-expresses the model's internal activations (the residual stream after a given layer) as a
sum of sparse "latents". Each latent is a direction in that activation space. At every token of a text, each latent has
an activation: 0 on most inputs, positive on some.

## What you have to decide

This episode has **{public.n_slots} slots**. Each slot names one latent (an SAE layer and a latent index) and a
numbered list of 20 candidate concepts. For each slot, answer which candidate concept the latent encodes, or
"nothing found".

**What a concept is.** Each candidate concept is one class of texts in a large text collection that you cannot see.
The collection has many more classes than the 20 on any one list. Every text in it belongs to exactly one class, so two
candidates never share texts, even when their labels sound related.

What the labels mean: "article about a X" stands for encyclopedia-style (Wikipedia-like) English articles about an
entity of that kind (for example "article about a volcano" means articles describing a particular volcano).
"text written in L" stands for everyday sentences, questions and short product reviews written in language L.

**What "encodes" means.** Take a text's activation for a latent to be the latent's largest activation over that
text's tokens. A latent encodes a class when texts of that class reliably give it a higher activation than texts of
other classes do. "Other classes" means every other class in the collection, not only the other 19 on the list. The
answer key was computed this way on texts from the collection.

**How the answer key is defined:** options are disjoint dataset classes; the correct option is the class the latent
separates best across varied styles.

**"nothing found".** For some slots, the class the latent encodes is not among its 20 candidates. The correct answer
for those slots is "nothing found". The latent on such a slot still encodes some class; that class is just not on the
list. Each slot is set up independently, so an episode can contain any number of these slots, from none to all.

**What counts as solved.** The episode counts as solved only if every slot is answered correctly. Naming an option on
a slot whose class is not on its list is an error. Answering "nothing found" on a slot whose class is on its list is
also an error.

**Latent numbering.** Latent indices in this episode are specific to this episode; they do not match the index
numbering used anywhere else. Inside the episode every tool uses the same numbering: index N at layer L names the same
latent in a slot, in a `latent_activations` request and in a `top_latents` result.

{public.slots_text}

## Tools

Call tools with `./tool <name> '<json args>'` from your working directory (`./tool help` lists them,
`./tool budget` shows your remaining budget). You may write and run your own analysis code with `./py`.
Work only inside your working directory.

{tool_docs}

### What each tool returns

Every call prints one line of JSON: `{"ok": true, "result": {...}}` on success, or `{"ok": false, "error": "..."}`
(the command then exits with status 1). If that line would be longer than 30 KB, it is saved to `out/resp_<n>.json`
and the printed line has a different shape: `{"ok": ..., "saved_to": "out/resp_<n>.json", "bytes": ..., "preview":
"<the first 1500 characters>"}`. The full response is in that file. Code that parses `./tool` output must handle both
shapes. These tools write nothing else under `out/` (no `.npy` files).

General rules:
- Numbers are rounded to 4 significant digits.
- Token strings are the model's own tokens, decoded. A leading space is part of the token (`" Paris"`). Each digit is
  its own token (`"1985"` becomes `"1"`, `"9"`, `"8"`, `"5"`).
- `tokens` never includes the beginning-of-text token. It shows exactly the part of your text that was kept after
  truncation.

Per tool. Each entry says what the tool shows, what it returns and what one call costs. Every tool is available in
every episode, and the list order is not a suggested order.

- `task_info`
  - Shows: the slots and their options, as listed above, in machine-readable form.
  - Returns `{"n_slots": int, "slots": [{"slot": int, "layer": int, "latent": int, "options": [20 strings]}]}`.
    `options` is a 0-based list: `options[0]` is option number 1, so option number = list position + 1.
  - Cost: 1 tool call, 0 forward units.
- `latent_activations` (args: `texts` 1-16 strings, `layer`, `latents` 1-8 indices, `per_token` default true)
  - Shows: how strongly the latents you name are active on each text you write, token by token.
  - Returns `{"layer": int, "results": [...]}` with one entry per text, in the order you sent them.
    Each entry is `{"tokens": [str], "max": {"<latent>": float}, "acts": {"<latent>": [float]}}`.
    - The keys of `max` and `acts` are the latent indices you asked for, written as strings (`"3835"`, not `3835`).
    - `max` is the largest activation over the listed tokens. It is 0.0 if the latent is inactive on every token.
    - `acts` gives one value per token and lines up with `tokens`: `acts["3835"][i]` is the activation at
      `tokens[i]`. It is left out when you pass `"per_token": false`.
    - 0.0 means the latent is inactive at that token. Each latent has its own scale, so values can be compared across
      texts for one latent, but not between different latents.
  - Cost: 1 tool call and 1 forward unit per text. The number of latents and `per_token` do not change the cost.
- `top_latents` (args: `texts` 1-8 strings, `layer`, `k` 1-20, default 10)
  - Shows: which latents of one layer are most active on each text you write, and at which token, including latents
    you did not name.
  - Returns `{"layer": int, "results": [...]}` with one entry per text:
    `{"tokens": [str], "top": [{"latent": int, "max": float, "at_token": str}]}`.
    - `top` is sorted by `max`, largest first.
    - It lists only latents whose `max` is above 0, so it can be shorter than `k`, or empty.
    - `latent` uses this episode's numbering, the same numbering as the slots.
    - `at_token` is the token where that maximum occurred.
  - Cost: 1 tool call and 1 forward unit per text. `k` does not change the cost.
- `vocab_projection` (args: `layer`, `latent`, `k` 1-25, default 15)
  - Shows: which output tokens one latent's direction pushes the model's next-token prediction towards, and which it
    pushes away from. It reads the SAE and model weights only; it runs no text.
  - Returns `{"top_tokens": [k strings], "bottom_tokens": [k strings]}`. Each list is ordered from the strongest
    effect down, and no scores are given. It multiplies the latent's decoder direction by the model's output
    (unembedding) matrix directly.
  - Cost: 1 tool call and 1 forward unit. `k` does not change the cost.
- `generate` (args: `prompt` one string, `max_new_tokens` 1-48, default 32)
  - Shows: how the model continues a prompt you write.
  - Returns `{"completion": str}`: only the newly generated text, without your prompt. Decoding is greedy, so the same
    prompt always gives the same completion. It runs the model only and reports no latents.
  - Cost: 1 tool call and 1 generate unit, 0 forward units. `max_new_tokens` does not change the cost.
- `next_token_logits` (args: `prompts` 1-16 strings, `top_k` 1-20, default 10)
  - Shows: which tokens the model considers most likely to come next after each prompt you write, and how likely.
  - Returns `{"results": [...]}` with one entry per prompt: `{"top_tokens": [str], "logprobs": [float]}`, most likely
    token first. Log-probabilities use the natural log. It runs the model only and reports no latents.
  - Cost: 1 tool call and 1 forward unit per prompt. `top_k` does not change the cost.

Text length:
- `latent_activations` and `top_latents` keep at most the first 64 tokens of each text, which is roughly 40-50 English
  words. The rest is ignored, and the text still costs the same.
- `generate` and `next_token_logits` keep at most 256 tokens of each prompt.
- Every tool that takes text first cuts each text or prompt to 2000 characters.

### What costs what

Budget for this episode:

{caps}

How each counter is used:
- **tool_calls**: every call of `task_info`, `latent_activations`, `top_latents`, `vocab_projection`, `generate` or
  `next_token_logits` uses 1, whatever its size. A call that the tool rejects with an error (for example for an
  invalid argument) also uses 1. `help`, `budget` and `submit` are free.
- **forward units**:
  - `latent_activations`: 1 per text in the call. The number of latents (1-8) and `per_token` do not change the cost.
  - `top_latents`: 1 per text in the call. `k` does not change the cost.
  - `next_token_logits`: 1 per prompt in the call.
  - `vocab_projection`: 1 per call.
  - `task_info` and `generate`: 0.
- **generate units**: 1 per `generate` call, whatever `max_new_tokens` is.
- **gradient units**: no tool in this task uses them.
- A call with invalid arguments is refused before any units are charged. So is a call that needs more units than you
  have left. Either way it still uses one tool call.
- **Waiting for compute.** A call may print "waiting for compute..." while the system is busy. If your shell gives up
  on a call while it waits, the call can still run later and be charged. Run `./tool budget` before repeating it, to
  see whether it was charged.

### Passing JSON safely on the command line

`./tool` takes the JSON arguments as one shell argument. Inside `'...'`, an apostrophe in your text (as in `it's`)
ends the quote early. The shell then reports "unexpected EOF", or `./tool` reports "arguments must be one JSON object
or key=value pairs". Any of these forms works with any text:

1. Write the arguments to a file, then pass the file's contents as one argument (keep the double quotes):

   ```
   cat > scratch/args.json <<'EOF'
   {"texts": ["It's late.", "She said \"no\" and left."], "layer": 12, "latents": [5]}
   EOF
   ./tool latent_activations "$(cat scratch/args.json)"
   ```

2. Inside `'...'`, write each apostrophe as the JSON escape `\u0027`, as in `"it\u0027s"`.
3. From a `./py` script, call `subprocess.run(["./tool", name, json.dumps(args)], capture_output=True, text=True)` and
   parse its stdout with `json.loads`. Remember the 30 KB rule above.

## Answer format

Submit exactly once, with one entry per slot:

```
./tool submit '{"answers": [{"slot": 0, "choice": 7}, {"slot": 1, "choice": "nothing found"}, ...]}'
```

- `slot` is the slot number shown above (they start at 0).
- `choice` is the option number (1-20) from that slot's numbered list, or the string `"nothing found"` when none of
  the 20 candidates is what the latent encodes.
- A malformed submission is rejected with a format error, and you can fix it and submit again.
- The episode ends after a valid submission, and later tool calls are refused.
