# Shared FeatureMatch model service

## Why it exists

Each FeatureMatch episode has its own tool server, which is the process that answers the agent's `./tool` calls.
Until now every tool server loaded its own copy of the subject model: gemma-2-2b plus three Gemma Scope sparse
autoencoders (SAEs), about 7 GB of GPU memory. The machine has one 23 GB GPU shared by many jobs, so only about two
FeatureMatch episodes could run at once.

The model service is one long-running process that loads the model once and does the model computations for every
episode. The per-episode tool servers then need no GPU: they check arguments, charge the budget, translate latent
indices, and forward the computation to the service over a local Unix socket. The whole diagnosis study then uses
one GPU job for all its episodes.

Using it is optional. Without the service, everything works exactly as before.

## What the service sees (and what it does not)

FeatureMatch gives every instance a secret permutation of SAE latent indices. Index `p` in the agent's view is real
Gemma Scope latent `perm[p]`. Without it, the task would reduce to looking latents up in a public explanation
database. The permutation stays in the per-episode tool server:
- before a call, the tool server maps the agent's indices to real ones;
- the service computes on real latent ids only;
- after the call, the tool server maps real ids back to the agent's indices (`top_latents`).

The service never reads an instance directory, answer key, option menu, or grading data. It only receives texts,
a layer number, real latent ids and generation settings. Its error replies carry only an exception class name,
never the inputs.

## Same numbers both ways

`model_service.py` defines the five model computations ("ops") once: `token_acts`, `top_latents`,
`vocab_projection`, `generate` and `next_token_logits`. `tools.py` runs them either in-process (`LocalBackend`) or
on the service (`RemoteBackend`), and the service runs the same functions on its own copy of the model. Numbers
cross the socket as JSON floats, which round-trip exactly. Rounding to 4 significant digits happens afterwards in
the tool server, exactly as before.

Evidence: `diagnosis/compare_service.py` runs a fixed list of calls on several instances both ways and compares
the outputs. The list covers every tool, all three layers, the instances' own slot latents plus random ones,
truncated and non-English texts, and argument errors. The result is in `diagnosis/model_service_check.json`; see
"Check results" below.
`diagnosis/test_model_service.py` (CPU, no model) tests the socket protocol, the fallback when the service is
down, generic error messages, and that the service receives only real latent ids.

## Start, check, stop

Run these from the worktree that holds the task (here `~/wt/fmdiag`), with the shared environment:

```
cd ~/wt/fmdiag && source ~/ideating-rl-tests/common/env.sh      # sets $PY
$PY -m tasks.featurematch.model_service start     # queues ONE light GPU job (8 GB, label featurematch-modelservice)
                                                  # and waits until the model is loaded (prints "ready after N s")
$PY -m tasks.featurematch.model_service status    # {"ready": true, "pid": ..., "served": <requests>, "busy_s": ...}
$PY -m common.gpuq status                         # shows the featurematch-modelservice job
$PY -m tasks.featurematch.model_service stop      # SIGTERM; removes the socket and the marker; GPU released
```

- Socket: `~/rlsbx/.fm_model.sock`, mode 600.
- Marker file: `~/rlsbx/.fm_model_service`, a small JSON file holding the socket path and pid. It is written only
  after the model has loaded and the socket is listening, and it is removed on stop.
- Log: `<main repo>/runs/.episodes/fm_model_service.log`.
- The service handles one computation at a time, under a lock. Each request waits in line, so many episodes can
  share it. Calls are small: at most 16 texts of up to 64 tokens, or one 48-token generation. One call takes well
  under a second, far below the broker's 180 s per-call limit.

**Stop the service when the study is not running episodes.** An idle service still holds about 7 GB of GPU memory.

## How tool servers find it

`tools.py` decides once per process, when it is imported:
1. env var `FM_MODEL_SERVICE=<socket path>`: use that service;
2. `FM_MODEL_SERVICE=off` (or `0`, `none`): always load the model in-process;
3. variable unset: use the service named in the marker file, if there is one.

In every case the service must answer a ping. If it does not, the tool server falls back to the old behaviour: it
loads the model itself and queues for 7 GB of GPU like before. With the service in use, `Env.GPU_GB` is 0, so the
broker starts the tool server without any GPU queue entry.

The running broker passes its own environment to the tool servers it starts. Setting `FM_MODEL_SERVICE` in your
shell therefore does not reach broker-started episodes. The marker file is what switches them over, and the broker
needed no change or restart. `FM_MODEL_SERVICE` is for processes you start yourself, such as
`inproc_gates.py`, `compare_service.py` or tests. Note that while the marker exists, even an in-process run such as
`inproc_gates.py` uses the service. Set `FM_MODEL_SERVICE=off` if you want such a run to load its own model.

Episodes are unaffected while they run if the service is started or stopped, with one exception. A tool server
keeps the mode it chose when it started. If the service stops while such a server is still using it, the server's
model tools return `{"ok": false, "error": "compute temporarily unavailable; try again in a minute"}`, and that
call is still charged. So start the service before preparing service-mode episodes and keep it up until they
finish. A server that is evicted after 10 idle minutes and restarted chooses its mode again.

## Exact usage for the study

```
cd ~/wt/fmdiag && source ~/ideating-rl-tests/common/env.sh
$PY -m tasks.featurematch.model_service start
$PY -m common.sandbox prepare --task featurematch --tasks-root ~/wt/fmdiag \
    --instance-dir <instance> --profile full --run-dir runs/featurematch/<ts>_<arm> --solver-label <label> \
    [--prompt-template <file>] [--min-submit-frac forward=0.6] [--extra-file <src>:<dst>]
... run the episodes (several at once) ...; $PY -m common.sandbox finish --episode <E> ...
$PY -m tasks.featurematch.model_service stop
```

(The `prepare` options come from the main checkout's `common/sandbox.py`. See docs/HARNESS_API.md section 5.)

## Check results

See `model_service_check.json`. The numbers are filled in below by the infrastructure stage.
