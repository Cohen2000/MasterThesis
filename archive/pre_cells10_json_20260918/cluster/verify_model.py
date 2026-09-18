"""Verify the pinned local model snapshot against the shard index."""
import json, os, sys
d = "/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot/models/Qwen3.6-35B-A3B"
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
missing = [f for f in shards if not os.path.exists(os.path.join(d, f))]
total = sum(os.path.getsize(os.path.join(d, f)) for f in shards if not missing)
need = ["config.json", "tokenizer.json", "tokenizer_config.json",
        "generation_config.json", "chat_template.jinja"]
absent = [f for f in need if not os.path.exists(os.path.join(d, f))]
print(json.dumps({"shards": len(shards), "missing_shards": missing,
                  "weights_gb": round(total / 1e9, 2),
                  "missing_support_files": absent}, indent=1))
sys.exit(1 if missing or absent else 0)
