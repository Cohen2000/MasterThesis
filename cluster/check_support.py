"""Report the serving stack and whether it knows the pinned model architecture."""
import json
import vllm, transformers, torch
from vllm.model_executor.models.registry import ModelRegistry
names = set(ModelRegistry.get_supported_archs())
info = {
    "vllm": vllm.__version__,
    "transformers": transformers.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "qwen3_archs": sorted(n for n in names if "Qwen3" in n),
    "target_supported": "Qwen3_5MoeForConditionalGeneration" in names,
    "moe_archs": sorted(n for n in names if "Moe" in n or "MoE" in n),
}
print("SUPPORT_INFO " + json.dumps(info))
