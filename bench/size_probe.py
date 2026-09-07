"""Report vLLM memory / KV-cache breakdown for a config. Phase 1 sizing."""
import argparse, json
from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--draft", default=None)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--gmu", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--kv-bytes", type=int, default=None)
    a = p.parse_args()

    kw = dict(model=a.target, gpu_memory_utilization=a.gmu,
              max_model_len=a.max_model_len)
    if a.kv_bytes:
        kw["kv_cache_memory_bytes"] = a.kv_bytes
    if a.draft:
        kw["speculative_config"] = {"model": a.draft,
                                    "num_speculative_tokens": a.k}

    llm = LLM(**kw)
    llm.generate(["Hello"], SamplingParams(max_tokens=4, temperature=0))
    c = llm.llm_engine.vllm_config.cache_config
    print("RESULT " + json.dumps({
        "draft": a.draft, "k": a.k if a.draft else None, "gmu": a.gmu,
        "num_gpu_blocks": c.num_gpu_blocks, "block_size": c.block_size,
        "kv_tokens": (c.num_gpu_blocks or 0) * c.block_size,
        "kv_cache_memory_bytes": getattr(c, "kv_cache_memory_bytes", None),
    }))


if __name__ == "__main__":
    main()
