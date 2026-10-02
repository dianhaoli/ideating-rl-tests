"""LatentKnockout core: gemma-2-2b + Gemma Scope residual SAEs, latent ablation, attribution, metrics.

Plain-language summary
----------------------
* The residual stream at layer L is the model's running internal state after transformer block L
  (HF hidden_states[L + 1]). A Gemma Scope SAE encodes it as f = JumpReLU(x @ W_enc + b_enc) (16384 latents, ~70
  non-zero per token) and decodes x_hat = f @ W_dec + b_dec. The part it cannot explain is the error e = x - x_hat.
* "Ablating" a set S of latents at layer L means x' = x - sum_{i in S} f_i(x) W_dec[i] at every token except BOS
  (Gemma Scope was not trained on BOS). This equals x_hat(f with S zeroed) + e, i.e. we keep the error term, so with
  S empty the model is bit-identical to the clean model.
* Everything is batched with left padding and explicit position ids, and only the last-token logits are computed for
  prompt sets (the 256k-vocab head is the expensive part); KL on unrelated text uses all positions.
"""
import glob
import os

import numpy as np

LAYERS = (6, 12, 18)
D_SAE = 16384
MODEL_ID = "google/gemma-2-2b"
FM_BF16 = os.path.expanduser("~/wt/featurematch/tasks/featurematch/cache/gemma-2-2b-bf16")   # read-only reuse


def model_path():
    if os.path.exists(os.path.join(FM_BF16, "model.safetensors.index.json")):
        return FM_BF16
    return MODEL_ID


def sae_path(layer):
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/hf_home"))
    pats = glob.glob(os.path.join(hf, "hub", "models--google--gemma-scope-2b-pt-res", "snapshots", "*",
                                  f"layer_{layer}", "width_16k", "average_l0_*", "params.npz"))
    if not pats:
        raise FileNotFoundError(f"Gemma Scope SAE for layer {layer} not cached")
    pats.sort(key=lambda p: abs(int(p.split("average_l0_")[1].split("/")[0]) - 70))
    return pats[0]


class Subject:
    def __init__(self, layers=LAYERS, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_path())
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_path(), dtype=torch.bfloat16, device_map=device,
                                                          attn_implementation="eager").eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.softcap = getattr(self.model.config, "final_logit_softcapping", None)
        self.layers = tuple(layers)
        self.sae = {}
        for L in self.layers:
            p = np.load(sae_path(L))
            self.sae[L] = {k2: torch.tensor(p[k1], dtype=torch.float32, device=device)
                           for k1, k2 in [("W_enc", "W_enc"), ("b_enc", "b_enc"), ("threshold", "thr"),
                                          ("W_dec", "W_dec"), ("b_dec", "b_dec")]}

    # ------------------------------------------------------------------ tokens
    def enc(self, texts, max_len=96):
        torch = self.torch
        e = self.tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        ids, am = e["input_ids"].to(self.device), e["attention_mask"].to(self.device)
        pos = (am.cumsum(1) - 1).clamp(min=0)
        first = am.float().argmax(1)
        nonbos = am.clone()
        nonbos[torch.arange(am.shape[0], device=am.device), first] = 0
        return ids, am, pos, nonbos

    def first_token_id(self, word):
        """Token id of ' word' (first token)."""
        return self.tok(" " + word, add_special_tokens=False)["input_ids"][0]

    # ------------------------------------------------------------------ SAE
    def sae_encode(self, x, L, idx=None):
        s = self.sae[L]
        if idx is None:
            pre = x @ s["W_enc"] + s["b_enc"]
            return pre * (pre > s["thr"])
        pre = x @ s["W_enc"][:, idx] + s["b_enc"][idx]
        return pre * (pre > s["thr"][idx])

    # ------------------------------------------------------------------ forward with an edit at layer L
    def _hook(self, L, edit, nonbos):
        def hook(_m, _i, o):
            x = o[0] if isinstance(o, tuple) else o
            x2 = edit(x, nonbos)
            if x2 is None:
                return o
            return (x2,) + tuple(o[1:]) if isinstance(o, tuple) else x2
        return self.model.model.layers[L].register_forward_hook(hook)

    def run(self, texts, L=None, edit=None, all_pos=False):
        """Return fp32 logits: [B, V] at the last token (default) or [B, T, V] (all_pos). edit(x_bf16, nonbos)->x'."""
        torch = self.torch
        ids, am, pos, nonbos = self.enc(texts)
        h = self._hook(L, edit, nonbos) if edit is not None else None
        try:
            with torch.no_grad():
                hid = self.model.model(input_ids=ids, attention_mask=am, position_ids=pos).last_hidden_state
        finally:
            if h is not None:
                h.remove()
        if not all_pos:
            hid = hid[:, -1]
        logits = self.model.lm_head(hid).float()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return (logits, am) if all_pos else logits

    def final_hidden(self, texts, L=None, edit=None):
        torch = self.torch
        ids, am, pos, nonbos = self.enc(texts)
        h = self._hook(L, edit, nonbos) if edit is not None else None
        try:
            with torch.no_grad():
                hid = self.model.model(input_ids=ids, attention_mask=am, position_ids=pos).last_hidden_state
        finally:
            if h is not None:
                h.remove()
        return hid, nonbos

    def logp_from_hidden(self, hid):
        torch = self.torch
        logits = self.model.lm_head(hid).float()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return torch.log_softmax(logits, -1)

    def resid(self, texts, L):
        """Residual after block L: ([B,T,d] fp32, nonbos mask, attention mask). Stops after block L."""
        torch = self.torch
        ids, am, pos, nonbos = self.enc(texts)
        box = {}

        class _Stop(Exception):
            pass

        def hook(_m, _i, o):
            box["x"] = (o[0] if isinstance(o, tuple) else o).float()
            raise _Stop()
        h = self.model.model.layers[L].register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model.model(input_ids=ids, attention_mask=am, position_ids=pos)
        except _Stop:
            pass
        finally:
            h.remove()
        return box["x"], nonbos, am

    # ------------------------------------------------------------------ edits (all act at non-BOS positions)
    def ablate_edit(self, L, row_sets):
        """row_sets: list (len B) of lists of latent ids to ablate in that batch row."""
        torch = self.torch
        pool = sorted({i for s in row_sets for i in s})
        if not pool:
            return lambda x, nb: None
        col = {i: j for j, i in enumerate(pool)}
        M = torch.zeros(len(row_sets), len(pool), device=self.device)
        for r, s in enumerate(row_sets):
            for i in s:
                M[r, col[i]] = 1.0
        idx = torch.tensor(pool, device=self.device)
        Wd = self.sae[L]["W_dec"][idx]

        def edit(x, nonbos):
            xf = x.float()
            f = self.sae_encode(xf, L, idx) * M[:, None, :]
            delta = (f @ Wd) * nonbos[..., None]
            return (xf - delta).to(x.dtype)
        return edit

    def recon_error_edit(self, L, zero_latents=()):
        """Explicit x' = x_hat(f with zero_latents removed) + (x - x_hat(f)); used to verify exactness."""
        s = self.sae[L]
        z = list(zero_latents)

        def edit(x, nonbos):
            xf = x.float()
            f = self.sae_encode(xf, L)
            xhat = f @ s["W_dec"] + s["b_dec"]
            err = xf - xhat
            f2 = f.clone()
            if z:
                f2[..., z] = 0
            x2 = f2 @ s["W_dec"] + s["b_dec"] + err
            m = nonbos[..., None].bool()
            return self.torch.where(m, x2, xf).to(x.dtype)
        return edit

    def recon_only_edit(self, L):
        s = self.sae[L]

        def edit(x, nonbos):
            xf = x.float()
            xhat = self.sae_encode(xf, L) @ s["W_dec"] + s["b_dec"]
            m = nonbos[..., None].bool()
            return self.torch.where(m, xhat, xf).to(x.dtype)
        return edit

    def steer_edit(self, row_vecs):
        """row_vecs: [B, d] fp32 vector subtracted at every non-BOS position of that row."""
        def edit(x, nonbos):
            return (x.float() - row_vecs[:, None, :] * nonbos[..., None]).to(x.dtype)
        return edit

    def project_edit(self, row_units):
        """row_units: [B, d] unit vectors (or zeros) projected out at every non-BOS position."""
        def edit(x, nonbos):
            xf = x.float()
            c = (xf * row_units[:, None, :]).sum(-1, keepdim=True)
            return (xf - c * row_units[:, None, :] * nonbos[..., None]).to(x.dtype)
        return edit

    # ------------------------------------------------------------------ attribution
    def attribution(self, texts, ans_sets, L, chunk=16):
        """Gradient x activation attribution of the answer log-prob to every SAE latent at layer L.
        metric per prompt = logsumexp over accepted answer token ids of log_softmax(last logits).
        Returns (attr [B, 16384] = sum_pos f_i * (W_dec[i] . dmetric/dx)  (estimated drop if ablated = attr),
                 act_sum [B, 16384] sum over non-BOS positions, act_max [B, 16384])."""
        torch = self.torch
        Wd = self.sae[L]["W_dec"]
        A, S, Mx = [], [], []
        for c0 in range(0, len(texts), chunk):
            tx = texts[c0:c0 + chunk]
            ids, am, pos, nonbos = self.enc(tx)
            box = {}

            def hook(_m, _i, o):
                x = o[0] if isinstance(o, tuple) else o
                xl = x.detach().float().requires_grad_(True)
                box["x"] = xl
                x2 = xl.to(x.dtype)
                return (x2,) + tuple(o[1:]) if isinstance(o, tuple) else x2
            h = self.model.model.layers[L].register_forward_hook(hook)
            try:
                with torch.enable_grad():
                    hid = self.model.model(input_ids=ids, attention_mask=am, position_ids=pos).last_hidden_state
                    lp = self.logp_from_hidden(hid[:, -1])
                    met = []
                    for r, a in enumerate(ans_sets[c0:c0 + chunk]):
                        met.append(torch.logsumexp(lp[r, a], 0))
                    torch.stack(met).sum().backward()
            finally:
                h.remove()
            x, g = box["x"].detach(), box["x"].grad.detach()
            with torch.no_grad():
                f = self.sae_encode(x, L) * nonbos[..., None]
                gd = g @ Wd.T
                A.append((f * gd).sum(1).cpu())
                S.append(f.sum(1).cpu())
                Mx.append(f.max(1).values.cpu())
            del x, g, f, gd
        return torch.cat(A).numpy(), torch.cat(S).numpy(), torch.cat(Mx).numpy()


# ---------------------------------------------------------------------------------------------- metrics
def margins(logits, ans_sets):
    """margin = max logit over accepted answers - max logit over everything else; top1 id."""
    import torch
    out = []
    for r, a in enumerate(ans_sets):
        row = logits[r]
        acc = row[a].max()
        tmp = row.clone()
        tmp[a] = -1e9
        out.append((acc - tmp.max()).item())
    return np.array(out), logits.argmax(-1).cpu().numpy()
