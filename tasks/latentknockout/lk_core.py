"""LatentKnockout core: gemma-2-2b + Gemma Scope residual SAEs, latent ablation, attribution, metrics.

Plain-language summary
----------------------
* The residual stream at layer L is the model's running internal state after transformer block L
  (HF hidden_states[L + 1]; Gemma Scope "layer_L" = blocks.L.hook_resid_post). A Gemma Scope SAE encodes it as
  f = JumpReLU(x @ W_enc + b_enc) (16384 latents, ~70-80 non-zero per token) and decodes x_hat = f @ W_dec + b_dec.
  The part it cannot explain is the error e = x - x_hat.
* "Ablating" a set S of latents at layer L means x' = x - sum_{i in S} f_i(x) W_dec[i] at every token except BOS
  (Gemma Scope was not trained on BOS). This equals x_hat(f with S zeroed) + e, i.e. the error term is kept, so with
  S empty the model is bit-identical to the clean model (validate.py checks this and the other identities).
* Speed: the edit only changes layers above L, so we cache the layer-L residual of every prompt once (PromptSet)
  and rerun only blocks L+1..25 for each candidate edit (run_from). validate.py checks run_from == full forward.
* Prompts are left-padded with explicit position ids; only last-token logits are computed for prompt sets (the
  256k-vocab head is the expensive part); KL on unrelated text uses all non-BOS positions.
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


class Edit:
    """One intervention at layer L. kind: 'lat' (ablate latent ids), 'vec' (subtract vector), 'proj' (project out
    unit vector), 'recon' (replace x by the SAE reconstruction: drops the error term), 'none'."""
    __slots__ = ("kind", "ids", "vec")

    def __init__(self, kind="none", ids=(), vec=None):
        self.kind, self.ids, self.vec = kind, tuple(int(i) for i in ids), vec

    @staticmethod
    def lat(ids):
        return Edit("lat" if len(ids) else "none", ids)


class PromptSet:
    """Prompts with their layer-L residuals cached in length-sorted chunks (bf16, on GPU)."""

    def __init__(self, subj, L, texts, ans=None, chunk=64, max_len=128):
        self.L, self.texts, self.n = L, list(texts), len(texts)
        self.ans = ans
        order = sorted(range(self.n), key=lambda i: len(texts[i]))
        self.chunks = []
        for c in range(0, self.n, chunk):
            idx = order[c:c + chunk]
            x, am, pos, nb = subj.resid([texts[i] for i in idx], L, max_len=max_len)
            self.chunks.append((idx, x, am, pos, nb))


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
        self.cfg = self.model.config
        self.nl = self.cfg.num_hidden_layers
        self.softcap = getattr(self.cfg, "final_logit_softcapping", None)
        self.layers = tuple(layers)
        self.sae = {}
        for L in self.layers:
            p = np.load(sae_path(L))
            self.sae[L] = {k2: torch.tensor(p[k1], dtype=torch.float32, device=device)
                           for k1, k2 in [("W_enc", "W_enc"), ("b_enc", "b_enc"), ("threshold", "thr"),
                                          ("W_dec", "W_dec"), ("b_dec", "b_dec")]}

    # ------------------------------------------------------------------ tokens
    def enc(self, texts, max_len=128):
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

    def head(self, hid):
        logits = self.model.lm_head(hid).float()
        if self.softcap:
            if logits.requires_grad:
                logits = self.torch.tanh(logits / self.softcap) * self.softcap
            else:
                logits.div_(self.softcap).tanh_().mul_(self.softcap)   # in place: saves 2 copies of [B, 256k]
        return logits

    # ------------------------------------------------------------------ full forward (with optional hook edit)
    def run(self, texts, L=None, edits=None, all_pos=False, max_len=128):
        """Full forward; edits (list of Edit, one per row, or None) applied by a hook on block L's output."""
        torch = self.torch
        ids, am, pos, nonbos = self.enc(texts, max_len)
        h = None
        if edits is not None:
            def hook(_m, _i, o):
                x = o[0] if isinstance(o, tuple) else o
                x2 = self.apply_edits(L, x, nonbos, edits)
                return (x2,) + tuple(o[1:]) if isinstance(o, tuple) else x2
            h = self.model.model.layers[L].register_forward_hook(hook)
        try:
            with torch.no_grad():
                hid = self.model.model(input_ids=ids, attention_mask=am, position_ids=pos).last_hidden_state
        finally:
            if h is not None:
                h.remove()
        if not all_pos:
            return self.head(hid[:, -1])
        return self.head(hid), am, nonbos

    def resid(self, texts, L, max_len=128):
        """Residual after block L: (x [B,T,d] bf16, attention mask, position ids, nonbos mask). Stops after block L."""
        torch = self.torch
        ids, am, pos, nonbos = self.enc(texts, max_len)
        box = {}

        class _Stop(Exception):
            pass

        def hook(_m, _i, o):
            box["x"] = (o[0] if isinstance(o, tuple) else o).detach()
            raise _Stop()
        h = self.model.model.layers[L].register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model.model(input_ids=ids, attention_mask=am, position_ids=pos)
        except _Stop:
            pass
        finally:
            h.remove()
        return box["x"], am, pos, nonbos

    def run_from(self, L, x, am, pos, last_only=True):
        """Run blocks L+1..end + final norm from a (possibly edited) layer-L residual x. Returns normed hidden."""
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        mm = self.model.model
        kw = dict(config=self.cfg, inputs_embeds=x, attention_mask=am, past_key_values=None, position_ids=pos)
        masks = {"full_attention": create_causal_mask(**kw), "sliding_attention": create_sliding_window_causal_mask(**kw)}
        pe = mm.rotary_emb(x, pos)
        h = x
        for i in range(L + 1, self.nl):
            h = mm.layers[i](h, attention_mask=masks[self.cfg.layer_types[i]], position_embeddings=pe,
                             position_ids=pos)
        if last_only:
            h = h[:, -1]
        return mm.norm(h)

    # ------------------------------------------------------------------ edits (all act at non-BOS positions)
    def apply_edits(self, L, x, nonbos, edits):
        """x [B,T,d] (bf16); edits: list of B Edit objects. Returns edited x in x.dtype."""
        torch = self.torch
        B = x.shape[0]
        kinds = {e.kind for e in edits}
        if kinds <= {"none"}:
            return x
        xf = x.float()
        nb = nonbos[..., None].float()
        out = xf
        if "lat" in kinds:
            pool = sorted({i for e in edits if e.kind == "lat" for i in e.ids})
            col = {i: j for j, i in enumerate(pool)}
            M = torch.zeros(B, len(pool), device=x.device)
            for r, e in enumerate(edits):
                if e.kind == "lat":
                    M[r, [col[i] for i in e.ids]] = 1.0
            idx = torch.tensor(pool, device=x.device)
            f = self.sae_encode(xf, L, idx) * M[:, None, :]
            out = out - (f @ self.sae[L]["W_dec"][idx]) * nb
        if "vec" in kinds or "proj" in kinds:
            V = torch.zeros(B, xf.shape[-1], device=x.device)
            U = torch.zeros(B, xf.shape[-1], device=x.device)
            for r, e in enumerate(edits):
                if e.kind == "vec":
                    V[r] = e.vec
                elif e.kind == "proj":
                    U[r] = e.vec
            out = out - V[:, None, :] * nb
            c = (xf * U[:, None, :]).sum(-1, keepdim=True)
            out = out - c * U[:, None, :] * nb
        if "recon" in kinds:
            s = self.sae[L]
            rows = [r for r, e in enumerate(edits) if e.kind == "recon"]
            xr = xf[rows]
            xhat = self.sae_encode(xr, L) @ s["W_dec"] + s["b_dec"]
            m = nonbos[rows][..., None].bool()
            out = out.clone()
            out[rows] = torch.where(m, xhat, xr)
        return out.to(x.dtype)

    # ------------------------------------------------------------------ batched evaluation on a PromptSet
    def eval_last(self, ps, edits, rows_per_batch=256):
        """For each edit (list of Edit) and each prompt in ps: margin of accepted answers and top-1 id.
        Returns (margins [n_edits, n], top1 [n_edits, n]) as numpy. ps.ans: list of lists of accepted token ids."""
        torch = self.torch
        E = len(edits)
        marg = np.zeros((E, ps.n), np.float32)
        top1 = np.zeros((E, ps.n), np.int64)
        with torch.no_grad():
            for idx, x, am, pos, nb in ps.chunks:
                n = len(idx)
                g = max(1, rows_per_batch // n)
                A = max(len(ps.ans[i]) for i in idx)
                ans = torch.tensor([ps.ans[i] + [ps.ans[i][0]] * (A - len(ps.ans[i])) for i in idx], device=x.device)
                for e0 in range(0, E, g):
                    eg = edits[e0:e0 + g]
                    G = len(eg)
                    xr = x.repeat(G, 1, 1)
                    rows = [e for e in eg for _ in range(n)]
                    xe = self.apply_edits(ps.L, xr, nb.repeat(G, 1), rows)
                    h = self.run_from(ps.L, xe, am.repeat(G, 1), pos.repeat(G, 1))
                    lg = self.head(h)
                    ansr = ans.repeat(G, 1)
                    acc = lg.gather(1, ansr).max(1).values
                    tv, ti = lg.topk(A + 1, dim=1)
                    isans = (ti[:, :, None] == ansr[:, None, :]).any(-1)
                    tv = tv.masked_fill(isans, -1e9)
                    other = tv.max(1).values
                    m = (acc - other).view(G, n).cpu().numpy()
                    t = ti[:, 0].view(G, n).cpu().numpy()
                    for j in range(G):
                        marg[e0 + j, idx] = m[j]
                        top1[e0 + j, idx] = t[j]
                    del xr, xe, h, lg
        return marg, top1

    def clean_hidden_all(self, ps):
        """Cache clean final normed hidden at all positions for a KL PromptSet."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for idx, x, am, pos, nb in ps.chunks:
                out.append(self.run_from(ps.L, x, am, pos, last_only=False))
        ps.clean_h = out

    def eval_kl(self, ps, edits, rows_per_batch=32):
        """Mean next-token KL(clean || edited) over non-BOS, non-pad positions, per edit and per text: [E, n]."""
        torch = self.torch
        if not hasattr(ps, "clean_h"):
            self.clean_hidden_all(ps)
        E = len(edits)
        kl = np.zeros((E, ps.n), np.float32)
        with torch.no_grad():
            for (idx, x, am, pos, nb), ch in zip(ps.chunks, ps.clean_h):
                n = len(idx)
                g = max(1, rows_per_batch // n)
                for e0 in range(0, E, g):
                    eg = edits[e0:e0 + g]
                    G = len(eg)
                    rows = [e for e in eg for _ in range(n)]
                    xe = self.apply_edits(ps.L, x.repeat(G, 1, 1), nb.repeat(G, 1), rows)
                    h = self.run_from(ps.L, xe, am.repeat(G, 1), pos.repeat(G, 1), last_only=False)
                    for j in range(G):
                        for r in range(n):
                            rr = j * n + r
                            m = nb[r].bool()
                            lp = torch.log_softmax(self.head(ch[r][m]), -1)
                            lq = torch.log_softmax(self.head(h[rr][m]), -1)
                            kl[e0 + j, idx[r]] = (lp.exp() * (lp - lq)).sum(-1).mean().item()
                    del xe, h
        return kl

    # ------------------------------------------------------------------ attribution (gradient x activation)
    def attribution(self, ps):
        """Per prompt: attr [n, 16384] = sum_pos f_i * (W_dec[i] . d metric / d x): the first-order estimate of the drop
        in metric if latent i is ablated; metric = logsumexp of log-probs of the accepted answer ids at the last token.
        Also returns act_sum [n,16384] (summed activation over non-BOS positions) and err_attr [n] (the same estimate
        for removing the SAE error term)."""
        torch = self.torch
        L = ps.L
        Wd = self.sae[L]["W_dec"]
        s = self.sae[L]
        attr = np.zeros((ps.n, D_SAE), np.float32)
        acts = np.zeros((ps.n, D_SAE), np.float32)
        err = np.zeros(ps.n, np.float32)
        for idx, x, am, pos, nb in ps.chunks:
            for c0 in range(0, len(idx), 16):
                sl = slice(c0, c0 + 16)
                ii = idx[sl]
                xl = x[sl].float().clone().requires_grad_(True)
                with torch.enable_grad():
                    h = self.run_from(L, xl.to(x.dtype), am[sl], pos[sl])
                    lp = torch.log_softmax(self.head(h), -1)
                    met = torch.stack([torch.logsumexp(lp[r, ps.ans[i]], 0) for r, i in enumerate(ii)])
                    met.sum().backward()
                g = xl.grad.detach()
                with torch.no_grad():
                    xd = xl.detach()
                    m = nb[sl][..., None].float()
                    f = self.sae_encode(xd, L) * m
                    gd = g @ Wd.T
                    attr[ii] = (f * gd).sum(1).cpu().numpy()
                    acts[ii] = f.sum(1).cpu().numpy()
                    xhat = f @ Wd + s["b_dec"]
                    err[ii] = (((xd - xhat) * m) * g).sum((1, 2)).cpu().numpy()
                del xl, g, f, gd
        return attr, acts, err

    def mean_last(self, ps):
        """Mean layer-L residual at the last token (fp32 [d])."""
        tot = 0
        for idx, x, am, pos, nb in ps.chunks:
            tot = tot + x[:, -1].float().sum(0)
        return tot / ps.n

    def latent_acts(self, ps, ids):
        """Activations of the given latents at every non-pad position: list per prompt of [T_i, len(ids)] numpy
        (BOS row zeroed)."""
        torch = self.torch
        idx_t = torch.tensor(list(ids), device=self.device)
        out = [None] * ps.n
        with torch.no_grad():
            for idx, x, am, pos, nb in ps.chunks:
                f = self.sae_encode(x.float(), ps.L, idx_t) * nb[..., None]
                for r, i in enumerate(idx):
                    keep = am[r].bool()
                    out[i] = f[r][keep].cpu().numpy()
        return out


def ids_for(subj, words):
    """Accepted answer token ids: first token of ' word' and of ' Word' (Q/A and record styles capitalise answers)."""
    return sorted({subj.first_token_id(v) for w in words for v in (w, w[:1].upper() + w[1:])})
