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

Per tool:
- `task_info` returns `{"n_slots": int, "slots": [{"slot": int, "layer": int, "latent": int, "options": [20 strings]}]}`.
  `options` is a 0-based list: `options[0]` is option number 1, so option number = list position + 1.
- `latent_activations` returns `{"layer": int, "results": [...]}` with one entry per text, in the order you sent them.
  Each entry is `{"tokens": [str], "max": {"<latent>": float}, "acts": {"<latent>": [float]}}`.
  - The keys of `max` and `acts` are the latent indices you asked for, written as strings (`"3835"`, not `3835`).
  - `max` is the largest activation over the listed tokens. It is 0.0 if the latent is inactive on every token.
  - `acts` gives one value per token and lines up with `tokens`: `acts["3835"][i]` is the activation at `tokens[i]`.
    It is left out when you pass `"per_token": false`.
  - 0.0 means the latent is inactive at that token. Each latent has its own scale, so values can be compared across
    texts for one latent, but not between different latents.
- `top_latents` returns `{"layer": int, "results": [...]}` with one entry per text:
  `{"tokens": [str], "top": [{"latent": int, "max": float, "at_token": str}]}`.
  - `top` is sorted by `max`, largest first.
  - It lists only latents whose `max` is above 0, so it can be shorter than `k`, or empty.
  - `latent` uses this episode's numbering.
  - `at_token` is the token where that maximum occurred.
- `vocab_projection` returns `{"top_tokens": [k strings], "bottom_tokens": [k strings]}`. Each list is ordered from the
  strongest effect down, and no scores are given. It multiplies the latent's decoder direction by the model's output
  (unembedding) matrix directly; it runs no text.
- `generate` returns `{"completion": str}`: only the newly generated text, without your prompt. Decoding is greedy, so
  the same prompt always gives the same completion. It runs the model only and reports no latents.
- `next_token_logits` returns `{"results": [...]}` with one entry per prompt: `{"top_tokens": [str], "logprobs":
  [float]}`, most likely token first. Log-probabilities use the natural log. It runs the model only and reports no
  latents.

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
