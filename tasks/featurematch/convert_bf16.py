"""One-time CPU conversion of google/gemma-2-2b (stored as fp32, ~10.5 GB) to a bf16 copy (~5.2 GB) in
tasks/featurematch/cache/gemma-2-2b-bf16 (gitignored; see MANIFEST.md).

WHY: loading the fp32 checkpoint memory-maps 10.5 GB, which pushes a process's RSS over the 8 GB gpuq RAM cap even
when the weights go straight to the GPU. Tool servers load the model once per episode, so this matters at
harness time too. Converted tensor by tensor so peak RAM stays ~1 shard.
Run: $PY -m tasks.featurematch.convert_bf16
"""
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "gemma-2-2b-bf16")


def main():
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/hf_home"))
    src = glob.glob(os.path.join(hf, "hub", "models--google--gemma-2-2b", "snapshots", "*"))[0]
    os.makedirs(OUT, exist_ok=True)
    wmap = {}
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        name = os.path.basename(f)
        tensors = {}
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                tensors[k] = sf.get_tensor(k).to(torch.bfloat16).contiguous()
                wmap[k] = name
        save_file(tensors, os.path.join(OUT, name), metadata={"format": "pt"})
        del tensors
        print("wrote", name, flush=True)
    json.dump({"metadata": {}, "weight_map": wmap}, open(os.path.join(OUT, "model.safetensors.index.json"), "w"))
    for f in os.listdir(src):
        if f.endswith(".json") and f != "model.safetensors.index.json" or f.startswith("tokenizer"):
            shutil.copy(os.path.join(src, f), os.path.join(OUT, f))
    cfg = json.load(open(os.path.join(OUT, "config.json")))
    cfg["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
