"""Can transformers load this architecture, and is any newer vLLM aware of it?"""
import json, importlib
info = {}
import transformers
info["transformers"] = transformers.__version__
try:
    from transformers import AutoConfig, AutoModelForCausalLM
    import transformers.models as M
    info["has_qwen3_5_moe_module"] = importlib.util.find_spec("transformers.models.qwen3_5_moe") is not None
    names = [n for n in dir(transformers) if "Qwen3" in n]
    info["qwen3_symbols"] = sorted(names)[:20]
    d = "/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot/models/Qwen3.6-35B-A3B"
    cfg = AutoConfig.from_pretrained(d)
    info["config_class"] = type(cfg).__name__
    info["model_type"] = cfg.model_type
    info["architectures"] = getattr(cfg, "architectures", None)
    tc = getattr(cfg, "text_config", None)
    if tc is not None:
        info["text_config"] = {k: getattr(tc, k, None) for k in
                               ("num_hidden_layers", "hidden_size", "num_experts",
                                "num_experts_per_tok", "num_attention_heads",
                                "num_key_value_heads", "max_position_embeddings")}
    cls = getattr(transformers, "Qwen3_5MoeForConditionalGeneration", None)
    info["transformers_has_target_class"] = cls is not None
except Exception as e:
    info["error"] = f"{type(e).__name__}: {e}"
print("HF_INFO " + json.dumps(info))
