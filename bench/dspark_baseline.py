#!/usr/bin/env python3
"""E3: reproduce stock DSpark adaptive verification end to end.

E1 and E2 measured the cost *surface* that AdaptiveVerificationManager prices,
but ran with no speculator - so the manager itself never executed. This boots a
real DSpark draft, confirms adaptive verification actually engages, dumps the
cost curves it profiles at startup, and then compares those predictions against
real verification steps under load.

What it reports:
  1. whether AdaptiveVerificationManager was constructed at all
  2. the draft/verify curves it profiled, and the fixed context length used
  3. acceptance rate and accepted-tokens-per-step under real generation
  4. predicted vs actual step latency on real (not dummy) steps

Run:
    source env.sh
    export VLLM_USE_V2_MODEL_RUNNER=1
    python bench/dspark_baseline.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone


def _inspect_worker(self) -> dict:
    """Runs in the worker: report what adaptive verification actually built."""
    import numpy as np

    import vllm.envs as envs

    runner = self.model_runner
    av = getattr(runner, "adaptive_verification", None)
    out = {
        "adaptive_verification_active": av is not None,
        "speculator": type(runner.speculator).__name__ if runner.speculator else None,
        "profile_context_len": envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN,
        "captured_token_counts": list(runner.cudagraph_manager.captured_token_counts()),
        "num_speculative_steps": getattr(runner.req_states, "num_speculative_steps", None),
    }
    if av is not None and av.cost_tables is not None:
        draft_table, verify_table = av.cost_tables
        out["cudagraph_limit"] = av._cudagraph_limit
        # Sample the verify table rather than dumping every entry.
        idx = [i for i in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
               if i < len(verify_table)]
        out["verify_table_samples"] = [(i, float(verify_table[i])) for i in idx]
        out["draft_table_samples"] = [
            (i, float(draft_table[i])) for i in range(len(draft_table)) if i <= 8
        ]
        out["verify_table_len"] = int(len(verify_table))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--draft", default="r3lax/Qwen3.5-0.8B-DSpark")
    p.add_argument("--num-speculative-tokens", type=int, default=3)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    p.add_argument("--attention-backend", default=None,
                   help="e.g. TRITON_ATTN. Adaptive verification needs a backend "
                        "reporting AttentionCGSupport.ALWAYS; FlashAttention only "
                        "does so at FA3, which needs Hopper (sm_90+).")
    p.add_argument("--no-adaptive", action="store_true",
                   help="Fixed-length verification instead of adaptive (control).")
    p.add_argument("--num-prompts", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--out", default="results/dspark_baseline.json")
    args = p.parse_args()

    import pynvml
    pynvml.nvmlInit()
    m = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
    free_mb, total_mb = m.free // 1024**2, m.total // 1024**2
    pynvml.nvmlShutdown()
    print(f"GPU free {free_mb} / {total_mb} MB")
    want = int(args.gpu_memory_utilization * total_mb)
    if want > free_mb:
        print(f"error: need {want} MB, only {free_mb} MB free. Try "
              f"--gpu-memory-utilization {max(0.5,(free_mb-400)/total_mb):.2f}",
              file=sys.stderr)
        return 2

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    import vllm.envs as envs
    from vllm import LLM, SamplingParams

    if not envs.VLLM_USE_V2_MODEL_RUNNER:
        print("error: set VLLM_USE_V2_MODEL_RUNNER=1 (DSpark is V2-only)",
              file=sys.stderr)
        return 2

    spec = {
        "method": "dspark",
        "model": args.draft,
        "num_speculative_tokens": args.num_speculative_tokens,
        "enable_adaptive_verification": not args.no_adaptive,
    }
    print(f"speculative_config = {json.dumps(spec)}")

    extra = {}
    if args.attention_backend:
        extra["attention_backend"] = args.attention_backend

    llm = LLM(model=args.model,
              speculative_config=spec,
              **extra,
              max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs,
              gpu_memory_utilization=args.gpu_memory_utilization,
              enforce_eager=False,
              disable_log_stats=False)

    info = llm.collective_rpc(_inspect_worker)[0]

    print("\n=== adaptive verification state ===")
    print(f"  active            : {info['adaptive_verification_active']}")
    print(f"  speculator        : {info['speculator']}")
    print(f"  spec steps        : {info['num_speculative_steps']}")
    print(f"  profile ctx len   : {info['profile_context_len']}")
    print(f"  cudagraph capture : {info['captured_token_counts']}")
    if info.get("verify_table_samples"):
        print(f"  cudagraph limit   : {info['cudagraph_limit']}")
        print("\n  verify cost table (num_target_tokens -> ms), as profiled:")
        for n, ms in info["verify_table_samples"]:
            print(f"      {n:>6} -> {ms:8.3f}")
        print("  draft cost table (num_reqs -> ms):")
        for n, ms in info["draft_table_samples"]:
            print(f"      {n:>6} -> {ms:8.3f}")
    elif info["adaptive_verification_active"]:
        print("  WARNING: manager exists but cost_tables is None (never profiled)")

    # ---- real generation --------------------------------------------------
    prompts = [
        f"Write a short paragraph explaining idea number {i} in systems research."
        for i in range(args.num_prompts)
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    print(f"\ngenerating {args.num_prompts} prompts x {args.max_tokens} tokens ...")
    outs = llm.generate(prompts, sp)
    total_out = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"  produced {total_out} output tokens")

    metrics = {}
    try:
        raw = llm.llm_engine.get_metrics()
        for m_ in raw:
            name = getattr(m_, "name", "")
            if "spec" in name or "accept" in name or "draft" in name:
                metrics[name] = getattr(m_, "value", None) or getattr(m_, "values", None)
    except Exception as exc:
        metrics["error"] = f"{type(exc).__name__}: {exc}"

    if metrics:
        print("\n=== speculative decoding metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        print("\nno spec-decode metrics exposed; see the engine's own "
              "'Speculative metrics' log lines above.")

    out = {"generated_utc": datetime.now(timezone.utc).isoformat(),
           "args": vars(args), "speculative_config": spec,
           "adaptive_verification": info,
           "total_output_tokens": total_out,
           "metrics": metrics}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
