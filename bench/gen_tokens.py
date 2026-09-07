"""Child process: emit generated token ids for a fixed prompt set, as JSON.

Used by bench/phase2_correctness.py. Prints one TOKENS line on stdout
containing a list-of-lists of token ids, one per prompt.
"""
from __future__ import annotations

import argparse
import json

from vllm import LLM, SamplingParams

from phase2_correctness import PROMPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gmu", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--kv-bytes", type=int, default=None)
    a = ap.parse_args()

    kw = dict(model=a.target, gpu_memory_utilization=a.gmu,
              max_model_len=a.max_model_len)
    if a.kv_bytes:
        kw["kv_cache_memory_bytes"] = a.kv_bytes
    if a.draft:
        kw["speculative_config"] = {"model": a.draft,
                                    "num_speculative_tokens": a.k}

    llm = LLM(**kw)
    # temperature=0 -> greedy; this is what makes the equivalence check valid.
    sp = SamplingParams(max_tokens=a.max_tokens, temperature=0)
    outs = llm.generate(PROMPTS, sp)
    print("TOKENS " + json.dumps([list(o.outputs[0].token_ids) for o in outs]))


if __name__ == "__main__":
    main()
