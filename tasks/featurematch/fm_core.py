"""Shared model + SAE plumbing for FeatureMatch (privileged side).

Subject model: google/gemma-2-2b (bf16). Dictionary: Gemma Scope residual-stream SAEs
(google/gemma-scope-2b-pt-res, width 16k, the average_l0 closest to 70 at layers 6/12/18).

A Gemma Scope SAE is a JumpReLU sparse autoencoder:
    pre  = x @ W_enc + b_enc
    acts = pre * (pre > threshold)          (JumpReLU: zero unless the pre-activation clears a learned threshold)
where x is the residual stream *after* transformer block L (HF hidden_states[L + 1]).
Each of the 16384 columns of W_enc is one "latent" (a learned feature).

Per-text summary used everywhere in this task: the MAX activation of a latent over the text's
tokens, excluding the BOS token (BOS has a huge residual norm and spurious activations).
"""
import glob
import os

import numpy as np

LAYERS = (6, 12, 18)
D_SAE = 16384
MODEL_ID = "google/gemma-2-2b"
SAE_REPO = "google/gemma-scope-2b-pt-res"
MAX_TOKENS = 64          # texts are truncated to this many tokens (plus BOS) everywhere: generator, tools, solver


BF16_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "gemma-2-2b-bf16")


def model_path():
    """Prefer the local bf16 copy (convert_bf16.py): the hub checkpoint is fp32 and its 10.5 GB memory map
    exceeds the 8 GB per-job RAM cap."""
    return BF16_DIR if os.path.exists(os.path.join(BF16_DIR, "model.safetensors.index.json")) else MODEL_ID


def sae_path(layer):
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/hf_home"))
    pats = glob.glob(os.path.join(hf, "hub", "models--google--gemma-scope-2b-pt-res", "snapshots", "*",
                                  f"layer_{layer}", "width_16k", "average_l0_*", "params.npz"))
    if not pats:
        raise FileNotFoundError(f"Gemma Scope SAE for layer {layer} not cached")
    # choose the average_l0 closest to 70 (the cached one)
    pats.sort(key=lambda p: abs(int(p.split("average_l0_")[1].split("/")[0]) - 70))
    return pats[0]


class Subject:
    """gemma-2-2b + the three SAEs on one device. Methods are batch-friendly and return numpy."""

    def __init__(self, device=None, layers=LAYERS, load_decoder=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_path())
        self.model = AutoModelForCausalLM.from_pretrained(model_path(), dtype=torch.bfloat16, device_map=self.device).eval()
        self.layers = tuple(layers)
        self.sae = {}
        for L in self.layers:
            p = np.load(sae_path(L))
            d = {
                "W_enc": torch.tensor(p["W_enc"], dtype=torch.float32, device=self.device),
                "b_enc": torch.tensor(p["b_enc"], dtype=torch.float32, device=self.device),
                "thr": torch.tensor(p["threshold"], dtype=torch.float32, device=self.device),
            }
            if load_decoder:
                d["W_dec"] = torch.tensor(p["W_dec"], dtype=torch.bfloat16, device=self.device)
            self.sae[L] = d

    # ------------------------------------------------------------------ tokenisation
    def encode_batch(self, texts):
        enc = self.tok(list(texts), return_tensors="pt", padding=True, truncation=True,
                       max_length=MAX_TOKENS + 1, add_special_tokens=True)
        return {k: v.to(self.device) for k, v in enc.items()}

    def token_strs(self, texts):
        out = []
        for t in texts:
            ids = self.tok(t, truncation=True, max_length=MAX_TOKENS + 1, add_special_tokens=True)["input_ids"]
            out.append([self.tok.decode([i]) for i in ids[1:]])
        return out

    # ------------------------------------------------------------------ residual + SAE
    def resid(self, texts, layers=None):
        """Return ({layer: [B, T, d] float32}, attention_mask[B, T]) for residual after block L."""
        torch = self.torch
        layers = layers or self.layers
        enc = self.encode_batch(texts)
        with torch.no_grad():
            out = self.model.model(**enc, output_hidden_states=True)   # base model only: skip the 256k-vocab LM head
        hs = {L: out.hidden_states[L + 1].float() for L in layers}
        return hs, enc["attention_mask"]

    def sae_encode(self, x, layer):
        s = self.sae[layer]
        pre = x @ s["W_enc"] + s["b_enc"]
        return pre * (pre > s["thr"])

    def maxpool_acts(self, texts, layers=None, latents=None):
        """Max over non-BOS, non-pad tokens of SAE activations.
        Returns {layer: np.ndarray [B, n_latents] float16}. latents: optional {layer: LongTensor/list}."""
        torch = self.torch
        layers = layers or self.layers
        hs, mask = self.resid(texts, layers)
        # BOS is the first non-pad token of each row (works for left or right padding)
        first = mask.float().argmax(dim=1)
        m = mask.clone()
        m[torch.arange(m.shape[0], device=m.device), first] = 0
        res = {}
        for L in layers:
            a = self.sae_encode(hs[L], L)
            if latents is not None:
                a = a[..., torch.as_tensor(latents[L], device=self.device)]
            a = a * m[..., None]
            res[L] = a.max(dim=1).values.to(torch.float16).cpu().numpy()
        return res

    def token_acts(self, texts, layer, latents):
        """Per-token activations (non-BOS) of selected latents. Returns list of [T_i, k] arrays + token strings."""
        torch = self.torch
        hs, mask = self.resid(texts, [layer])
        a = self.sae_encode(hs[layer], layer)[..., torch.as_tensor(latents, device=self.device)]
        a = a.cpu().numpy()
        mask = mask.cpu().numpy()
        out = []
        for i in range(len(texts)):
            idx = np.nonzero(mask[i])[0][1:]          # non-pad positions, skipping BOS (the first one)
            out.append(a[i, idx])
        return out, self.token_strs(texts)
