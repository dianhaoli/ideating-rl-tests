"""Skeptic re-implementation (independent of lk_core): model loading, SAE latent ablation, held-out metric.

Written from scratch for the skeptic reproduction; it does NOT import lk_core or analyze. Design choices differ on
purpose where that is harmless, so agreement is evidence the reported numbers do not hinge on one code path:
  * RIGHT padding and a plain causal mask (lk_core left-pads with explicit position ids). With right padding a real
    token never sees a pad, so every prompt's last real token is computed exactly as if it were alone.
  * The edit is applied by a forward hook (full forward) or by re-running blocks L+1..25 from a cached layer-L
    residual; check_exact() compares the two.
Intervention ("ablate set S at layer L"): at every position except BOS (position 0),
    x' = x - sum_{i in S} f_i(x) W_dec[i],   f = JumpReLU(x W_enc + b_enc; threshold)   (error term kept).
Metric per item: margin = max logit over accepted answer ids - max logit over all other ids (final softcap applied).
Effect = share of target items with margin < 0; Preserve = share of sibling items with margin > 0;
KL = mean over unrelated texts of the mean over non-BOS positions of KL(clean || edited);
R = Effect x Preserve x max(0, 1 - KL / kappa).
"""
import glob
import os

import numpy as np
import torch

BF16_DIR = os.path.expanduser("~/wt/featurematch/tasks/featurematch/cache/gemma-2-2b-bf16")   # read-only
N_LAT = 16384


def find_sae(layer):
    root = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/hf_home")), "hub",
                        "models--google--gemma-scope-2b-pt-res", "snapshots")
    cands = glob.glob(os.path.join(root, "*", f"layer_{layer}", "width_16k", "average_l0_*", "params.npz"))
    cands.sort(key=lambda p: abs(int(p.rsplit("average_l0_", 1)[1].split(os.sep)[0]) - 70))
    return cands[0]


class KX:
    def __init__(self, layers, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        src = BF16_DIR if os.path.exists(os.path.join(BF16_DIR, "config.json")) else "google/gemma-2-2b"
        self.dev = device
        self.tok = AutoTokenizer.from_pretrained(src)
        self.tok.padding_side = "right"
        self.lm = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map=device,
                                                       attn_implementation="eager").eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.core = self.lm.model
        self.cap = self.lm.config.final_logit_softcapping
        self.nl = self.lm.config.num_hidden_layers
        self.sae = {}
        for L in layers:
            z = np.load(find_sae(L))
            self.sae[L] = dict(We=torch.from_numpy(z["W_enc"]).float().to(device),
                               be=torch.from_numpy(z["b_enc"]).float().to(device),
                               th=torch.from_numpy(z["threshold"]).float().to(device),
                               Wd=torch.from_numpy(z["W_dec"]).float().to(device),
                               bd=torch.from_numpy(z["b_dec"]).float().to(device))
            self.sae[L]["path"] = find_sae(L)

    # ------------------------------------------------------------------ text
    def first_id(self, w):
        return self.tok(" " + w, add_special_tokens=False)["input_ids"][0]

    def accept_ids(self, words):
        out = set()
        for w in words:
            out.add(self.first_id(w))
            out.add(self.first_id(w[0].upper() + w[1:]))
        return sorted(out)

    def batch(self, texts, max_len=128):
        e = self.tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        ids = e["input_ids"].to(self.dev)
        lens = e["attention_mask"].sum(1).to(self.dev)
        return ids, lens

    # ------------------------------------------------------------------ SAE pieces
    def acts(self, x, L, idx=None):
        s = self.sae[L]
        if idx is None:
            pre = x @ s["We"] + s["be"]
            return torch.where(pre > s["th"], pre, torch.zeros_like(pre))
        pre = x @ s["We"][:, idx] + s["be"][idx]
        return torch.where(pre > s["th"][idx], pre, torch.zeros_like(pre))

    def ablate(self, x, L, rowsets):
        """x [B,T,d] bf16; rowsets: list (len B) of latent-id lists. Returns edited bf16 tensor."""
        uni = sorted({i for s in rowsets for i in s})
        if not uni:
            return x
        col = {i: j for j, i in enumerate(uni)}
        M = torch.zeros(x.shape[0], len(uni), device=x.device)
        for r, s in enumerate(rowsets):
            for i in s:
                M[r, col[i]] = 1.0
        idx = torch.tensor(uni, device=x.device)
        xf = x.float()
        f = self.acts(xf, L, idx) * M[:, None, :]
        delta = f @ self.sae[L]["Wd"][idx]
        delta[:, 0, :] = 0.0                      # never touch BOS
        return (xf - delta).to(x.dtype)

    # ------------------------------------------------------------------ forward paths
    def _mask(self, T, dtype):
        m = torch.full((T, T), torch.finfo(dtype).min, device=self.dev, dtype=dtype).triu(1)
        return m[None, None]

    def full_last_logits(self, ids, lens, L=None, rowsets=None):
        """Full forward with an optional hook edit at block L; returns last-real-token logits [B, V] fp32."""
        h = None
        if rowsets is not None:
            def hook(_m, _i, out):
                return self.ablate(out, L, rowsets)
            h = self.core.layers[L].register_forward_hook(hook)
        try:
            with torch.no_grad():
                hid = self.core(input_ids=ids).last_hidden_state
        finally:
            if h is not None:
                h.remove()
        last = hid[torch.arange(ids.shape[0], device=self.dev), lens - 1]
        return self.logits(last)

    def logits(self, hid):
        lg = self.lm.lm_head(hid).float()
        return torch.tanh(lg / self.cap) * self.cap

    def resid(self, ids, L):
        box = {}

        class Stop(Exception):
            pass

        def hook(_m, _i, out):
            box["x"] = out.detach()
            raise Stop
        h = self.core.layers[L].register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.core(input_ids=ids)
        except Stop:
            pass
        finally:
            h.remove()
        return box["x"]

    def from_layer(self, x, L):
        """Run blocks L+1..end + final norm from residual x [B,T,d] (bf16). Plain causal mask, positions 0..T-1."""
        B, T, _ = x.shape
        pos = torch.arange(T, device=self.dev)[None].expand(B, T)
        pe = self.core.rotary_emb(x, pos)
        mask = self._mask(T, x.dtype)
        h = x
        for i in range(L + 1, self.nl):
            h = self.core.layers[i](h, position_embeddings=pe, attention_mask=mask, position_ids=pos)
        return self.core.norm(h)


class Items:
    """A prompt set with accepted answer ids; caches the layer-L residual in length-sorted chunks."""

    def __init__(self, kx, texts, acc, L, chunk=48, max_len=128):
        self.kx, self.L, self.n = kx, L, len(texts)
        self.texts, self.acc = list(texts), [list(a) for a in acc]
        order = sorted(range(self.n), key=lambda i: len(texts[i]))
        self.chunks = []
        for c in range(0, self.n, chunk):
            idx = order[c:c + chunk]
            ids, lens = kx.batch([texts[i] for i in idx], max_len)
            x = kx.resid(ids, L)
            A = max(len(self.acc[i]) for i in idx)
            acc_t = torch.tensor([self.acc[i] + [self.acc[i][0]] * (A - len(self.acc[i])) for i in idx], device=kx.dev)
            self.chunks.append((idx, ids, lens, x, acc_t))

    def eval(self, sets, rows_per_pass=384, want_rank=False):
        """margins [len(sets), n] (and answer rank if want_rank) for each latent set (cached-residual path)."""
        kx, L = self.kx, self.L
        S = len(sets)
        marg = np.zeros((S, self.n), np.float32)
        rank = np.zeros((S, self.n), np.int32) if want_rank else None
        top1 = np.zeros((S, self.n), np.int64) if want_rank else None
        with torch.no_grad():
            for idx, ids, lens, x, acc_t in self.chunks:
                n = len(idx)
                g = max(1, rows_per_pass // n)
                for s0 in range(0, S, g):
                    ss = sets[s0:s0 + g]
                    G = len(ss)
                    xe = kx.ablate(x.repeat(G, 1, 1), L, [s for s in ss for _ in range(n)])
                    h = kx.from_layer(xe, L)
                    ll = lens.repeat(G)
                    lg = kx.logits(h[torch.arange(G * n, device=kx.dev), ll - 1])
                    a = acc_t.repeat(G, 1)
                    acc_best = lg.gather(1, a).max(1).values
                    lg2 = lg.scatter(1, a, float("-inf"))
                    other = lg2.max(1).values
                    m = (acc_best - other).view(G, n).cpu().numpy()
                    if want_rank:
                        rk = (lg > acc_best[:, None]).sum(1).view(G, n).cpu().numpy()
                        t1 = lg.argmax(1).view(G, n).cpu().numpy()
                    for j in range(G):
                        marg[s0 + j, idx] = m[j]
                        if want_rank:
                            rank[s0 + j, idx] = rk[j]
                            top1[s0 + j, idx] = t1[j]
                    del xe, h, lg, lg2
        return (marg, rank, top1) if want_rank else marg

    def eval_full(self, sets):
        """Same as eval() but by a hooked FULL forward (slow path; used for exactness checks)."""
        kx, L = self.kx, self.L
        marg = np.zeros((len(sets), self.n), np.float32)
        with torch.no_grad():
            for si, s in enumerate(sets):
                for idx, ids, lens, x, acc_t in self.chunks:
                    lg = kx.full_last_logits(ids, lens, L, [s] * len(idx))
                    acc_best = lg.gather(1, acc_t).max(1).values
                    other = lg.scatter(1, acc_t, float("-inf")).max(1).values
                    marg[si, idx] = (acc_best - other).cpu().numpy()
        return marg

    def attribution(self, mb=12):
        """Per item: attr [n, 16384] = sum over non-BOS real positions of f_i * (W_dec[i] . d logp(answer) / dx),
        mean last-token residual [d] and summed activations [n, 16384]."""
        kx, L = self.kx, self.L
        Wd = kx.sae[L]["Wd"]
        attr = np.zeros((self.n, N_LAT), np.float32)
        actsum = np.zeros((self.n, N_LAT), np.float32)
        last_sum = torch.zeros(Wd.shape[1], device=kx.dev)
        for idx, ids, lens, x, acc_t in self.chunks:
            for c0 in range(0, len(idx), mb):
                ii = idx[c0:c0 + mb]
                xb = x[c0:c0 + mb].float().clone().requires_grad_(True)
                lb = lens[c0:c0 + mb]
                with torch.enable_grad():
                    h = kx.from_layer(xb.to(torch.bfloat16), L)
                    lg = kx.logits(h[torch.arange(len(ii), device=kx.dev), lb - 1])
                    lp = torch.log_softmax(lg, -1)
                    met = torch.logsumexp(lp.gather(1, acc_t[c0:c0 + mb]), 1)
                    met.sum().backward()
                g = xb.grad.detach()
                with torch.no_grad():
                    xd = xb.detach()
                    T = xd.shape[1]
                    valid = (torch.arange(T, device=kx.dev)[None] < lb[:, None]).float()
                    valid[:, 0] = 0
                    f = kx.acts(xd, L) * valid[..., None]
                    attr[ii] = (f * (g @ Wd.T)).sum(1).cpu().numpy()
                    actsum[ii] = f.sum(1).cpu().numpy()
                    last_sum += xd[torch.arange(len(ii), device=kx.dev), lb - 1].sum(0)
                del xb, g, f
        return attr, actsum, (last_sum / self.n)


def kl_texts(kx, texts, L, sets, max_len=64, per=4):
    """KL(clean || edited) per set per text: [len(sets), len(texts)] (hooked full forward, all positions)."""
    out = np.zeros((len(sets), len(texts)), np.float32)
    with torch.no_grad():
        for t0 in range(0, len(texts), per):
            tt = texts[t0:t0 + per]
            ids, lens = kx.batch(tt, max_len)
            x = kx.resid(ids, L)
            h0 = kx.from_layer(x, L)
            lp0 = torch.log_softmax(kx.logits(h0), -1)
            T = ids.shape[1]
            valid = (torch.arange(T, device=kx.dev)[None] < lens[:, None])
            valid[:, 0] = False
            for si, s in enumerate(sets):
                if not s:
                    continue
                h = kx.from_layer(kx.ablate(x, L, [s] * len(tt)), L)
                lp = torch.log_softmax(kx.logits(h), -1)
                kl = (lp0.exp() * (lp0 - lp)).sum(-1)
                for r in range(len(tt)):
                    out[si, t0 + r] = kl[r][valid[r]].mean().item()
                del h, lp, kl
            del lp0, h0
    return out
