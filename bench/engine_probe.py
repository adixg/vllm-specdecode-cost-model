"""Child process: build one vLLM engine, warm it, report its KV-cache shape.

Run via bench/common.run_probe, which parses this script's stdout+stderr.
Kept deliberately small: everything it prints on stdout is one RESULT line of
JSON; the memory accounting comes from vLLM's own log lines on stderr.
"""
from __future__ import annotations

import argparse
import json

from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gmu", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--kv-bytes", type=int, default=None)
    a = ap.parse_args()

    kw = dict(model=a.target, gpu_memory_utilization=a.gmu,
              max_model_len=a.max_model_len)
    if a.kv_bytes:
        # Pin the KV budget in bytes so every config gets an identical cache,
        # instead of letting each one take whatever gpu_memory_utilization
        # happens to leave over.
        kw["kv_cache_memory_bytes"] = a.kv_bytes
    if a.draft:
        kw["speculative_config"] = {"model": a.draft,
                                    "num_speculative_tokens": a.k}

    llm = LLM(**kw)
    # One tiny generation so CUDA graphs are captured and peak activation is
    # actually realised before we read the numbers back.
    llm.generate(["Hello"], SamplingParams(max_tokens=4, temperature=0))

    c = llm.llm_engine.vllm_config.cache_config
    print("RESULT " + json.dumps({
        "target": a.target, "draft": a.draft,
        "k": a.k if a.draft else None,
        "gmu": a.gmu, "max_model_len": a.max_model_len,
        "kv_bytes_requested": a.kv_bytes,
        "num_gpu_blocks": c.num_gpu_blocks,
        "block_size": c.block_size,
        "kv_tokens": (c.num_gpu_blocks or 0) * c.block_size,
    }))


if __name__ == "__main__":
    main()
