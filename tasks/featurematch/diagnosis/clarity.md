# FeatureMatch step 1: tool and task clarity audit

2026-10-02. CPU only; no GPU job, no tool call that loads the model, no hosted API.

**Question.** Could an agent fail FeatureMatch because the task statement (TASK.md) or the tools are unclear, rather
than because the interpretability problem is hard? This step lists every ambiguity I found, checks the six recorded
probe transcripts for signs that an ambiguity cost the agent something, and writes a clearer prompt template with
the same capabilities.

**Files.**
- `tasks/featurematch/agent_prompt_old.md`: the current template, copied unchanged ("TASK_old"; arm B).
- `tasks/featurematch/agent_prompt_revised.md`: the revised template (arm A). Same placeholders (`{public.model}`,
  `{public.sae}`, `{public.n_slots}`, `{public.slots_text}`, `{tool_docs}`, `{caps}`), same tools, same budget.
- `tools.py` is unchanged.

**Short glossary.**
- *Latent*: one feature of the sparse autoencoder (SAE). At each token it has a number that is usually 0.
- *Slot*: one question in an episode ("which of these 20 concepts does latent N respond to?").
- *Planted / null slot*: the true concept is / is not on the slot's list. On a null slot the answer is "nothing found".
- *Forward unit*: the budget counter for running text through the model.

## 1. What I did

1. Rendered the exact TASK.md an agent sees for a v2 instance. All 180 instances in
   `~/wt/featurematch/tasks/featurematch/instances` are generator v2; this worktree has no instance pool, so I used
   `fm-t2-09ba84b566` read-only through `--instance-dir`. Command: `common.sandbox prepare --task featurematch
   --profile full`, run dir under `~/.claude/jobs/e4652089/tmp/fmdiag_clarity/run`. Episode `epcb5db6e731`, finished
   unsubmitted (`valid: true`, no tool call made). The tools part of TASK.md is byte-identical to the one the probe
   agents saw (diffed against `ep2cd00804ab/TASK.md`).
2. Read `tools.py`, the `./tool` client (`common/agent_client/tool`), `common/broker.py`, `common/toolserver.py` and
   `fm_core.py` to see every response key and how each call is charged.
3. Read all eight recorded LLM probe transcripts and tool logs in `~/wt/featurematch/runs/featurematch/`:
   - `20261001-171048_apiprobe_sonnet` (Sonnet: ep2cd00804ab, epcd8c604d31);
   - `20261001-182038_openai_probe_orch` (gpt-6-luna: ep0f904f753d, epea26c30f2c; gpt-6.1-sol: ep0746750a4f,
     ep4980d0bf46, both starved by the GPU queue);
   - `20261001-165348_apiprobe_orch` (aborted; ep157f3953a8, ep6cd5b8dd30; 2 calls each).
4. Wrote the revised template and checked that it renders and passes the harness TASK.md checks (section 6).

**Limit on what the transcripts can show.** No transcript contains the agent's reasoning. Sonnet's thinking blocks
are empty strings, the OpenAI transcripts store no reasoning text, and no agent wrote a final summary
(`api_stdout.txt` holds only metadata). So I **cannot** say whether any agent "considered and rejected" a tool. I can
only see what it called and what went wrong.

## 2. Ambiguities in the current TASK.md

Severity: **High** = shown to cost an agent budget or answers in the transcripts. **Medium** = plausible cost, not
observed. **Low** = cosmetic or rare.

| # | Ambiguity | What TASK.md says | What the code does | Evidence | Severity |
|---|---|---|---|---|---|
| A1 | Response keys of `latent_activations` are not named | "Returns, per text, the tokens (BOS excluded) and per latent the activation at every token plus the max over tokens." | Each entry is `{"tokens", "max": {"<latent>": float}, "acts": {"<latent>": [float]}}`, with latent keys as **strings**. | ep2cd guessed `x['latents'][...]['max']` with a silent default of 0. Every probe then read 0, which cost 5 calls × 16 texts = **80 forward units**. Its next guess, `x['activations']`, printed only key names, which cost another **36 units**. In total about 10% of its budget went on finding the keys. ep157f guessed the `vocab_projection` shape (`t.get('token',t)`). | High |
| A2 | Output over 30 KB has a different shape | `./tool help`: "Responses over 30 KB are saved to out/resp_<n>.json and only a preview is printed." | The printed line becomes `{"ok", "saved_to", "bytes", "preview"}`, with no `result` key. 16 texts × 8 latents with `per_token` true can pass 30 KB. | Not hit in the probes: all agents used ≤ 2 latents or `per_token: false`. A script doing `json.loads(out)['result']` would crash. | Medium |
| A3 | Quoting JSON on the command line | Only `./tool <name> '<json args>'` is shown. | The client reads JSON from **one** argv element. An apostrophe inside `'...'` ends the quote early. There is no file or stdin option. | **6 failed commands** in 3 episodes. ep0f: 4 bash errors "unexpected EOF while looking for matching `\"'" (texts with `men's`, `l'Arc`, `women's`). epea: 1. ep0746 (sol): 1 client error "arguments must be one JSON object or key=value pairs". These cost no budget (the shell fails before `./tool` runs), but the agent **rewrote its probe texts** to drop the apostrophe (`"The men's 100 metre freestyle"` → `"The 100 metre freestyle"`; `"Prix de l'Arc de Triomphe"` → `"A famous horse race is run annually in France."`). So quoting changed what was measured. | High |
| A4 | How the budget is counted | Tool doc lines give "one forward unit per text" and the caps list. | `tool_calls` counts every task-tool call, including calls the tool rejects. Forward units are charged per text (latents 1-8 and `per_token` are free); `vocab_projection` costs 1 and `generate` costs 0 forward. A call with bad arguments, or one over the remaining budget, is refused before any units are charged. `submit`, `help` and `budget` are free. With 1200 forward units and 150 calls, 16 texts per call reaches the forward cap after 75 calls; 1 text per call reaches the call cap at 150 forward units. | Not stated as rules; it has to be assembled from the per-tool lines. epea fell from 16 to 4 texts per call after its timeouts, then submitted at 24/150 calls and 96/1200 units. Without its reasoning I cannot tell whether it misjudged the budget. | Medium |
| A5 | A retried call can be charged twice | The harness prompt says "If a command times out, simply run it again." | When the shell gives up, the broker still runs the call and charges it. | The forensics report (issue 5) found 16 units charged twice in epea and 33 in ep0746. In epea the agent re-ran the same 16-text call 3 times after 900 s shell timeouts. | Medium (an environment fault; the prompt can only warn) |
| A6 | Role of `top_latents` | "Most active SAE latents on each text …" | Its latent indices use the episode's permuted numbering, the same as the slots. | TASK.md also says "Latent indices in this episode are specific to this episode; they do not correspond to the index numbering used anywhere else." That can be read as "the indices `top_latents` returns are a different numbering". No agent called `top_latents`. | Medium |
| A7 | Role of `generate` / `next_token_logits` | Listed with the white-box tools. | They run the model only and return no latent information. | Not said. No agent called them. Harmless, but an agent cannot tell from the text what they are for. | Low |
| A8 | Role and output of `vocab_projection` | "Returns the k tokens whose logits the direction raises most and the k it lowers most." | It returns two lists of strings and no scores. The decoder direction is multiplied by the unembedding directly (no final norm). It runs no text. | Used in 4 of 6 runs, mostly as a first step. Its outputs for these latents were mostly junk tokens (`" BoxFit"`, `"PreExecute"`, `" Critical"`). Once, in epea, it pointed at a word confound (`" Lenny"`, `" LEN"`). The doc is accurate but gives no format. | Low |
| A9 | What "encodes" means | "the concept whose texts reliably activate this latent more than other text does" | The key compares each concept's texts with the texts of **all other concepts in the hidden collection**, using each text's **max over tokens**. | "Other text" and "activate" (max or mean? over which texts?) are undefined. The agent cannot know that the comparison set is the whole collection, not just the 20 options. | Medium |
| A10 | How null slots work | "For some slots none of the 20 candidates is what the latent encodes" | A null latent is selective for one concept that is **not** on the list. Each slot is null independently. | It is unclear whether a null latent is dead, encodes nothing, or encodes an unlisted concept. In ep2cd a "nothing found" verdict came from probes that read 0. The prompt does not say that a latent which never fires on your probes is not, by itself, the null signature. | Medium |
| A11 | Options are disjoint classes | Not stated. | Every text in the collection has exactly one class. | Labels overlap in everyday English: "cultivated variety" vs "grape", "skater" vs "speed skater". The forensics report recommends saying the options are disjoint dataset classes. | Medium |
| A12 | Scoring | Not stated. | Pass = every slot right; score = planted accuracy × null accuracy. | The agent does not know that a wrong claim on a null slot and a wrong "nothing found" on a planted slot both fail the episode. | Low-medium |
| A13 | Truncation | "each truncated to 64 tokens" (latent tools) | 64 tokens is about 40-50 English words, and digits are 1 token each (a year costs 4). `generate` and `next_token_logits` keep 256 tokens, and every text is first cut to 2000 characters. The `tokens` list shows what was kept. | Not observed to matter: the probe texts were short. | Low |
| A14 | `task_info` options are 0-based | "the numbered options" | `options[0]` is option 1. | No agent called `task_info`, but it is an off-by-one trap for any agent that reads the list programmatically. | Low |
| A15 | `gradient: 0 units` in the caps list | It is listed. | No tool uses it. | Harmless noise. | Low |
| A16 | Activation scale | Not stated. | Raw JumpReLU activations; 0 means inactive; the scale differs per latent; values are rounded to 4 significant digits. | Agents compared raw values across latents without visible confusion. | Low |
| A17 | Harness prompt mentions `.npy` files | `agent_prompt.txt`: "for example of the .npy files that tools save under out/" | FeatureMatch tools save no `.npy` files. | Misleading, but it is in the shared harness prompt (`common/sandbox.py` AGENT_PROMPT), not in the task template. The revised template says so. | Low |

## 3. Per-episode evidence

| Episode (model) | Calls / forward used | Tools used | Clarity problems seen |
|---|---|---|---|
| ep2cd00804ab (Sonnet, T2) | 58/150, 748/1200 (62%) | `latent_activations` only, called from `./py` via `subprocess` | A1: 116 units lost to guessing response keys (`'latents'`, `'activations'`). No quoting errors, because it built JSON in Python. |
| epcd8c604d31 (Sonnet, T3) | 9/150, 103/550 (19%) | `latent_activations` only | Probed its output (`for k,v in x.items(): if k not in ('tokens','max')`) to find the keys. No errors. |
| ep0f904f753d (luna, T2) | 14/150, 147/1200 (12%) | `latent_activations` 13, `vocab_projection` 1 | A3: 4 quoting failures; it removed apostrophes from its probe texts each time. |
| epea26c30f2c (luna, T2) | 24/150, 96/1200 (8%) | `latent_activations` 22, `vocab_projection` 4 | A3: 1 quoting failure. A5: three 900 s shell timeouts, each followed by a retry (forensics: 16 units charged twice). It then used 4 texts per call instead of 16. |
| ep0746750a4f (sol, invalid: queue starvation) | 5, 36 | `vocab_projection`, `latent_activations` | A3: 1 client parse error caused by an apostrophe. A5: repeated timeouts. |
| ep4980d0bf46 (sol, invalid: queue starvation) | 5, 31 | `vocab_projection`, `latent_activations` | A5: repeated timeouts. It inspected `./py` and `./tool` with `head`, which is allowed because they are inside the sandbox. |
| ep157f3953a8, ep6cd5b8dd30 (aborted run) | 2 calls each | `vocab_projection` | A1: guessed the response shape (`t.get('token',t)`). |

**Findings from the transcripts:**
- No agent called `top_latents`, `generate`, `next_token_logits` or `task_info`. Because no reasoning was recorded, the
  data cannot say whether this was a choice or a failure to see what these tools are for. The revised prompt states
  each tool's role and output without recommending any of them, so an arm-A vs arm-B difference in tool use will
  separate the two explanations.
- No agent misread `max`. One agent lost about 10% of its forward budget finding the keys. Others spent turns
  exploring the output.
- No agent ran out of any budget. Every agent stopped with most of its budget left (8-62% of forward units used), and
  none checked `./tool budget` just before submitting. This early stopping does not look like a budget-accounting
  misunderstanding, but the transcripts cannot rule that out. Arm C tests it directly.

## 4. Problems outside TASK.md (not fixable in the template)

- **A5, double charging after a shell timeout**, and the clocks that disagree: these are environment faults
  (forensics issues 5 and 6). The revised prompt only warns about them. The shared model service planned for step 4
  should make long queue waits rare.
- **A17**, the `.npy` sentence in the shared harness prompt.
- **Misleading labels.** Skater = speed skaters only. CultivatedVariety contains no grapes. This is a data and label
  problem, so the fix belongs in the pool (step 2), not in the prompt. The revised prompt says only that classes are
  disjoint. It does not say that labels can be narrower than their everyday meaning, because that would point at
  specific answers.
- The codename false positive in the leak scan ("this feature matches …") is already fixed in `common/leakscan.py`
  (commit 7ace2d43).

## 5. What the revised template changes

**Clarity only (no new task information):**
- A full response schema for every tool: every key, the string-typed latent keys, how `acts` lines up with `tokens`,
  the 0-based `options` list, that `top` can be shorter than `k`, that `vocab_projection` gives no scores, and the
  30 KB `saved_to` response shape.
- Exact budget accounting: what uses tool calls (including rejected calls) and forward units (per text; latents,
  `per_token` and `k` are free); that `generate` uses 0 forward units; what is refused before charging; that gradient
  units are unused.
- Three safe ways to pass JSON (a file plus `"$(cat file)"`, the `'` escape, and `subprocess` from `./py`). I
  checked all three against the client's own `parse_args`. The examples use neutral texts ("It's late.") and name no
  concept.
- What `generate` and `next_token_logits` are: model-only tools that report no latents. Also that latent numbering is
  shared by all tools.
- Truncation in concrete terms (about 40-50 words, digits are one token each, 256 tokens for the behavioural tools,
  2000 characters for all).
- A warning to check `./tool budget` before re-running a call that timed out.

**Definitions that add information the old prompt left implicit.** Read arm A vs arm B with these in mind:
1. Every text has exactly one class, so the options are disjoint (A11).
2. A text's activation is the max over its tokens. "Other text" means every other class in the hidden collection, not
   only the 19 other options (A9).
3. A latent on a "nothing found" slot still encodes some unlisted class. Slots are independent, so an episode can have
   any number of such slots (A10). The 0.4 null rate is **not** given.
4. The episode is solved only if every slot is right. Both kinds of error are named (A12). The score formula is not
   given.

**Deliberately not said:** no method or workflow, no tool recommendation, no hint about which tools are useful, no
note that probe-text style matters (the label definitions are unchanged from the old prompt), no note that latents
ignore their concept's name, no null rate, and no dataset name or grading term (AUROC).

**Length.** The rendered TASK.md grows from about 1,340 words to about 2,500 for a 5-slot T2 instance. Length is a
confound for the arm A vs arm B comparison.

## 6. Render and harness checks

The current worktree's `prepare` has no `--prompt-template` option yet (the infra task is adding one). So I ran the
real `common.sandbox.prepare` with its read of `tasks/featurematch/agent_prompt.md` redirected to the revised file.
The script is `~/.claude/jobs/e4652089/tmp/fmdiag_clarity/prep_revised.py`. This runs the full pipeline: rendering,
`check_task_md` (codename, canary, leak strings, private paths), the sandbox, and the episode record.

| Instance (tier) | Episode | Placeholders left | check_task_md | Words | finish |
|---|---|---|---|---|---|
| fm-t2-09ba84b566 (T2, 1200) | epb36c961e83 | 0 | pass | 2,518 | valid, unsubmitted |
| fm-t3-09a3b6b9ad (T3, 550) | ep15aa8ba2e5 | 0 | pass | 2,396 | valid, unsubmitted |
| fm-t1-08f455e08d (T1, 1200) | ep92eeced0b1 | 0 | pass | 2,514 | valid, unsubmitted |

The old template also passes on fm-t2-09ba84b566 (episode epcb5db6e731). Earlier renders made while drafting
(ep00ebaebc11, epbc69c720e9, ep44f84a46f3, ep3e6a6adc59) were all finished as well. No episode called a model tool,
so no GPU was used. Rendered copies are in `~/.claude/jobs/e4652089/tmp/fmdiag_clarity/TASK_{old,revised}_*.md`
(not in git).
