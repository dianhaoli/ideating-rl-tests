"""Minimal GPU Env for the harness smoke test: Qwen2.5-0.5B (cached in ~/hf_home), one logits tool."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.toolserver import TaskEnv, ToolError, tool

MODEL = "Qwen/Qwen2.5-0.5B"


class Env(TaskEnv):
    PROFILES = {"full": ["logits"]}
    GPU_GB = 2
    MODEL_OUTPUT_FIELDS = {"tokens"}

    def load(self, instance_dir):
        self.tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()

    @tool(doc="Top-k next-token predictions. args: prompt: str, top_k: int (<=20)")
    def logits(self, prompt, top_k=5):
        if not isinstance(prompt, str) or not 1 <= int(top_k) <= 20:
            raise ToolError("prompt must be a string and 1 <= top_k <= 20")
        self.charge("forward", 1)
        with torch.no_grad():
            ids = self.tok(prompt, return_tensors="pt").to("cuda")
            lp = torch.log_softmax(self.model(**ids).logits[0, -1].float(), -1)
            v, i = lp.topk(int(top_k))
        return {"tokens": [self.tok.decode([t]) for t in i.tolist()], "logprobs": [round(x, 4) for x in v.tolist()]}
