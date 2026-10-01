"""FeatureMatch tool environment (HARNESS_API.md section 2).

The agent studies gemma-2-2b and three Gemma Scope residual SAEs (layers 6, 12, 18; 16384 latents each).
Every tool is a generic primitive: run text through the model and read SAE latents, list the most active latents
on a text, project a latent's decoder direction onto the vocabulary, generate text, read next-token logits.
No tool takes or returns a concept name, the menu, held-out data or anything about grading.

Latent indices are PERMUTED per instance: the agent's latent index p at layer L refers to the real Gemma Scope
latent perm_L[p], with perm_L derived from a secret per-instance seed. WHY: public explanation databases
(e.g. Neuronpedia) label every real Gemma Scope latent; without the permutation the task would reduce to a lookup.
Every tool that accepts or returns latent indices uses the agent's (permuted) index space consistently.

Profiles:
  full      all tools
  blackbox  generate + next_token_logits only (behaviour, no internals) for the black-box control
"""
import json
import os
import threading

import numpy as np

from common.toolserver import TaskEnv, ToolError, tool

from tasks.featurematch.fm_core import D_SAE, LAYERS, MAX_TOKENS

MAX_TEXTS_ACTS = 16
MAX_TEXTS_TOP = 8
MAX_LATENTS = 8
MAX_CHARS = 2000
_SUBJECT = None    # one model per process, shared by Env instances (in-process gate runs create many Envs)
_SUBJECT_LOCK = threading.Lock()


def get_subject():
    global _SUBJECT
    with _SUBJECT_LOCK:
        return _get_subject_locked()


def _get_subject_locked():
    global _SUBJECT
    if _SUBJECT is None:
        import torch
        try:
            from common import gpuq
            gpuq.apply_caps()
        except Exception:
            pass
        from tasks.featurematch.fm_core import Subject
        _SUBJECT = Subject()
        _SUBJECT.torch = torch
    return _SUBJECT


def perm_for(seed, layer):
    """Secret permutation for one layer: agent index p -> real latent perm[p]."""
    return np.random.default_rng([int(seed), int(layer)]).permutation(D_SAE)


def r4(x):
    """Round to 4 significant digits (keeps responses small; HARNESS_API rule)."""
    x = float(x)
    if x == 0.0:
        return 0.0
    return float(f"{x:.4g}")


class Env(TaskEnv):
    PROFILES = {
        "full": ["task_info", "latent_activations", "top_latents", "vocab_projection", "generate",
                 "next_token_logits"],
        "blackbox": ["task_info", "generate", "next_token_logits"],
    }
    GPU_GB = 7          # gemma-2-2b bf16 (~5.2 GB) + 3 SAE encoders/decoders (~0.9 GB) + activations
    MODEL_OUTPUT_FIELDS = {"tokens", "token", "completion", "top_tokens", "bottom_tokens", "at_token"}

    # ------------------------------------------------------------------ lifecycle
    def load(self, instance_dir):
        inst = json.load(open(os.path.join(instance_dir, "instance.json")))
        self.public = json.load(open(os.path.join(instance_dir, "public.json")))
        seed = inst["extra"]["perm_seed"]
        self.perm = {L: perm_for(seed, L) for L in LAYERS}
        self.inv = {L: np.argsort(p) for L, p in self.perm.items()}
        # The model loads in a background thread. WHY: `task_info` needs no model, and the zero-effort recipe
        # baselines (hundreds of gate episodes) call nothing else; loading gemma-2-2b for each of them would hold
        # the shared GPU for nothing. Model tools wait for the load (inside their call, so at most a few tens of
        # seconds of the first model call). Planted and null instances load identically (no timing fingerprint).
        self._subj_err = None
        self._subj_thread = threading.Thread(target=self._bg_load, daemon=True)
        self._subj_thread.start()

    def _bg_load(self):
        try:
            get_subject()
        except BaseException as e:          # surfaced (generically) on the first model tool call
            self._subj_err = e

    @property
    def subj(self):
        t = getattr(self, "_subj_thread", None)
        if t is not None:
            t.join()
        if getattr(self, "_subj_err", None) is not None:
            raise RuntimeError("subject model failed to load") from self._subj_err
        return get_subject()

    # ------------------------------------------------------------------ argument checks (generic messages only)
    def _texts(self, texts, cap):
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not texts or len(texts) > cap:
            raise ToolError(f"texts must be a list of 1-{cap} strings")
        if not all(isinstance(t, str) and t.strip() for t in texts):
            raise ToolError("every text must be a non-empty string")
        return [t[:MAX_CHARS] for t in texts]

    def _layer(self, layer):
        try:
            layer = int(layer)
        except Exception:
            raise ToolError("layer must be one of 6, 12, 18")
        if layer not in LAYERS:
            raise ToolError("layer must be one of 6, 12, 18")
        return layer

    def _latents(self, latents):
        if isinstance(latents, int):
            latents = [latents]
        if not isinstance(latents, list) or not latents or len(latents) > MAX_LATENTS:
            raise ToolError(f"latents must be a list of 1-{MAX_LATENTS} integers")
        try:
            latents = [int(x) for x in latents]
        except Exception:
            raise ToolError("latents must be integers")
        if any(x < 0 or x >= D_SAE for x in latents):
            raise ToolError(f"latent indices must be in [0, {D_SAE - 1}]")
        return latents

    # ------------------------------------------------------------------ public task description
    @tool(doc="The task's slots, exactly as listed in the task statement: for each slot the SAE layer, the latent "
              "index and the numbered options. No arguments. Uses one tool call and no forward units.")
    def task_info(self):
        return {"n_slots": self.public["n_slots"],
                "slots": [{"slot": s["slot"], "layer": s["layer"], "latent": s["latent"], "options": s["options"]}
                          for s in self.public["slots"]]}

    # ------------------------------------------------------------------ white-box tools
    @tool(doc="SAE latent activations on your texts. args: texts: list[str] (1-16; each truncated to 64 tokens), "
              "layer: int (6, 12 or 18), latents: list[int] (1-8 latent indices of that layer's SAE), "
              "per_token: bool (default true; false returns only the max per text). Returns, per text, the tokens "
              "(BOS excluded) and per latent the activation at every token plus the max over tokens. "
              "Cost: one forward unit per text.")
    def latent_activations(self, texts, layer, latents, per_token=True):
        texts = self._texts(texts, MAX_TEXTS_ACTS)
        layer = self._layer(layer)
        latents = self._latents(latents)
        self.charge("forward", len(texts))
        real = [int(self.perm[layer][p]) for p in latents]
        acts, toks = self.subj.token_acts(texts, layer, real)
        out = []
        for a, tk in zip(acts, toks):
            row = {"tokens": tk, "max": {str(p): r4(a[:, j].max()) if len(a) else 0.0 for j, p in enumerate(latents)}}
            if per_token:
                row["acts"] = {str(p): [r4(v) for v in a[:, j]] for j, p in enumerate(latents)}
            out.append(row)
        return {"layer": layer, "results": out}

    @tool(doc="Most active SAE latents on each text. args: texts: list[str] (1-8; truncated to 64 tokens), "
              "layer: int (6, 12 or 18), k: int (1-20, default 10). Returns per text the k latents with the largest "
              "max-over-tokens activation (BOS excluded), each with that max and the token where it occurred. "
              "Cost: one forward unit per text.")
    def top_latents(self, texts, layer, k=10):
        texts = self._texts(texts, MAX_TEXTS_TOP)
        layer = self._layer(layer)
        try:
            k = int(k)
        except Exception:
            raise ToolError("k must be an integer in 1-20")
        if not 1 <= k <= 20:
            raise ToolError("k must be an integer in 1-20")
        self.charge("forward", len(texts))
        torch = self.subj.torch
        hs, mask = self.subj.resid(texts, [layer])
        a = self.subj.sae_encode(hs[layer], layer)
        first = mask.float().argmax(dim=1)
        m = mask.clone()
        m[torch.arange(m.shape[0], device=m.device), first] = 0
        a = a * m[..., None]
        mx, pos = a.max(dim=1)                     # [B, D]
        vals, idx = mx.topk(k, dim=-1)
        toks = self.subj.token_strs(texts)
        out = []
        for i in range(len(texts)):
            row = []
            for v, j in zip(vals[i].tolist(), idx[i].tolist()):
                if v <= 0:
                    continue
                t = int(pos[i, j]) - int(first[i]) - 1
                row.append({"latent": int(self.inv[layer][j]), "max": r4(v),
                            "at_token": toks[i][t] if 0 <= t < len(toks[i]) else ""})
            out.append({"tokens": toks[i], "top": row})
        return {"layer": layer, "results": out}

    @tool(doc="Project one SAE latent's decoder direction onto the model's output vocabulary (unembedding). "
              "args: layer: int (6, 12 or 18), latent: int, k: int (1-25, default 15). Returns the k tokens whose "
              "logits the direction raises most and the k it lowers most. Cost: one forward unit.")
    def vocab_projection(self, layer, latent, k=15):
        layer = self._layer(layer)
        latent = self._latents([latent])[0]
        try:
            k = int(k)
        except Exception:
            raise ToolError("k must be an integer in 1-25")
        if not 1 <= k <= 25:
            raise ToolError("k must be an integer in 1-25")
        self.charge("forward", 1)
        torch = self.subj.torch
        d = self.subj.sae[layer]["W_dec"][int(self.perm[layer][latent])]
        W_U = self.subj.model.get_output_embeddings().weight
        with torch.no_grad():
            logits = (W_U @ d.to(W_U.dtype)).float()
        top = logits.topk(k).indices.tolist()
        bot = (-logits).topk(k).indices.tolist()
        dec = self.subj.tok.decode
        return {"top_tokens": [dec([i]) for i in top], "bottom_tokens": [dec([i]) for i in bot]}

    # ------------------------------------------------------------------ behavioural tools (both profiles)
    @tool(doc="Greedy text continuation from the model. args: prompt: str, max_new_tokens: int (1-48, default 32). "
              "Cost: one generate unit.")
    def generate(self, prompt, max_new_tokens=32):
        prompt = self._texts([prompt], 1)[0]
        try:
            n = int(max_new_tokens)
        except Exception:
            raise ToolError("max_new_tokens must be an integer in 1-48")
        if not 1 <= n <= 48:
            raise ToolError("max_new_tokens must be an integer in 1-48")
        self.charge("generate", 1)
        torch = self.subj.torch
        enc = self.subj.tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(self.subj.device)
        with torch.no_grad():
            out = self.subj.model.generate(**enc, max_new_tokens=n, do_sample=False)
        return {"completion": self.subj.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)}

    @tool(doc="Next-token distribution after each prompt. args: prompts: list[str] (1-16), top_k: int (1-20, "
              "default 10). Returns the top_k tokens with log-probabilities. Cost: one forward unit per prompt.")
    def next_token_logits(self, prompts, top_k=10):
        prompts = self._texts(prompts, 16)
        try:
            k = int(top_k)
        except Exception:
            raise ToolError("top_k must be an integer in 1-20")
        if not 1 <= k <= 20:
            raise ToolError("top_k must be an integer in 1-20")
        self.charge("forward", len(prompts))
        torch = self.subj.torch
        tok = self.subj.tok
        out = []
        for p in prompts:
            enc = tok(p, return_tensors="pt", truncation=True, max_length=256).to(self.subj.device)
            with torch.no_grad():
                lg = self.subj.model(**enc).logits[0, -1].float()
            lp = torch.log_softmax(lg, -1)
            v, i = lp.topk(k)
            out.append({"top_tokens": [tok.decode([j]) for j in i.tolist()], "logprobs": [r4(x) for x in v.tolist()]})
        return {"results": out}

    # ------------------------------------------------------------------ submission format check
    def validate_submission(self, sub):
        """Format only. Expected: {"answers": [{"slot": int, "choice": int (1-20) or "nothing found"}, ...]},
        exactly one entry per slot. Never hints at correctness."""
        n = len(self.public["slots"])
        n_opt = len(self.public["slots"][0]["options"])
        if not isinstance(sub, dict) or not isinstance(sub.get("answers"), list):
            return 'submission must be a JSON object {"answers": [...]}'
        seen = set()
        for a in sub["answers"]:
            if not isinstance(a, dict) or "slot" not in a or "choice" not in a:
                return 'each answer must be {"slot": <int>, "choice": <option number or "nothing found">}'
            try:
                if isinstance(a["slot"], (bool, float)):
                    raise ValueError
                s = int(a["slot"])
            except Exception:
                return "slot must be an integer"
            if s < 0 or s >= n or s in seen:
                return f"slots must be distinct integers in 0..{n - 1}"
            seen.add(s)
            c = a["choice"]
            if isinstance(c, str) and c.strip().lower() == "nothing found":
                continue
            if isinstance(c, bool) or not isinstance(c, (int, str)):
                return f'choice must be an option number 1-{n_opt} or "nothing found"'
            try:
                ci = int(c)
            except Exception:
                return f'choice must be an option number 1-{n_opt} or "nothing found"'
            if not 1 <= ci <= n_opt:
                return f'choice must be an option number 1-{n_opt} or "nothing found"'
        if len(seen) != n:
            return f"give exactly one answer for each of the {n} slots"
        return None
