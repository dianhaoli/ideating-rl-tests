"""Pre-download subject models and selected Gemma Scope SAEs into HF_HOME.

Token is read by huggingface_hub from $HF_HOME/token (never printed).
Usage: HF_HOME=~/hf_home python -m common.download_models
"""
import re
from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download

MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2-2b",
    "google/gemma-3-270m",
]
ALLOW = ["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "*.tiktoken"]
SAE_REPO = "google/gemma-scope-2b-pt-res"
SAE_LAYERS = [6, 12, 18]
SAE_WIDTH = "width_16k"
TARGET_L0 = 70


def main():
    for m in MODELS:
        p = snapshot_download(m, allow_patterns=ALLOW)
        print("OK model", m, p, flush=True)
    files = list_repo_files(SAE_REPO)
    for layer in SAE_LAYERS:
        cands = [f for f in files if f.startswith(f"layer_{layer}/{SAE_WIDTH}/") and f.endswith("params.npz")]
        def l0(f):
            mm = re.search(r"average_l0_(\d+)", f)
            return abs(int(mm.group(1)) - TARGET_L0) if mm else 1e9
        best = sorted(cands, key=l0)[0]
        p = hf_hub_download(SAE_REPO, best)
        print("OK sae", best, p, flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
