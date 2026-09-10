#!/usr/bin/env python3
"""Milestone 1: is vLLM's verification cost model exchangeable in context length?

vLLM's AdaptiveVerificationManager prices a verification step with a
one-dimensional curve: `(num_target_tokens -> forward_ms)`, profiled once at
startup against a single synthetic context length
(`VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN`, default 8192).  See
`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:192` and
`set_initial_cost_curves` just below it.

That model asserts verification tokens are *exchangeable*: two batches with the
same total token count cost the same, whatever context they sit on top of.  This
script tries to falsify that.

  Phase A  reproduce vLLM's own profiling exactly - sweep num_tokens at the
           stock baseline context length, then feed the medians through the
           real `build_cost_tables_from_curves` to get the table vLLM would use.
  Phase B  re-measure the same num_tokens grid at other context lengths.
  Report   predicted (from the Phase A table) vs actual, per cell.

Latency comes from vLLM's own `StepTimingCollector`, so the numbers are the same
quantity that feeds the cost curve in production - not a reimplementation.

No DSpark checkpoint is needed.  The hypothesis is about the *target* model's
verification step, which is just a forward pass over a given batch shape, so any
model exercises it.  Reproducing stock DSpark end to end is a separate step.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone


def _probe_on_worker(self, grid: list[dict], replays: int, warmup: int) -> dict:
    """Runs inside the vLLM worker process. `self` is the Worker."""
    import torch

    runner = self.model_runner

    # `_dummy_run` derives num_reqs = min(num_tokens, max_num_reqs) and spreads
    # tokens evenly, and applies ONE scalar context_len to every request.  So
    # this measures *uniform* context composition only; genuinely heterogeneous
    # per-request contexts cannot be expressed through this entry point.
    results = []
    for cell in grid:
        num_tokens, context_len = cell["num_tokens"], cell["context_len"]
        samples: list[float] = []
        error = None
        try:
            for _ in range(warmup):
                runner._dummy_run(num_tokens=num_tokens, context_len=context_len)
            torch.cuda.synchronize()

            for _ in range(replays):
                with runner.step_timing.collect() as timings:
                    runner._dummy_run(num_tokens=num_tokens, context_len=context_len)
                    # `_timed.append` lives in `drafter_end`, so a step only
                    # yields a sample once the drafter phase closes.  With no
                    # speculator configured nothing closes it, so close it here.
                    # forward_start/forward_end were already recorded inside
                    # execute_model, so forward_ms is vLLM's own measurement.
                    if runner.speculator is None:
                        runner.step_timing.drafter_start()
                        runner.step_timing.drafter_end()
                for s in timings:
                    samples.append(s.forward_ms)
        except Exception as exc:  # most likely: context doesn't fit in KV cache
            error = f"{type(exc).__name__}: {exc}"

        num_reqs = min(num_tokens, runner.max_num_reqs)
        results.append(
            {
                "num_tokens": num_tokens,
                "context_len": context_len,
                "num_reqs": num_reqs,
                # NOMINAL footprint. vLLM's set_dummy_context wraps block ids modulo
            # the pool size ("Spans are disjoint until the pool runs out, then
            # they wrap and share blocks"), so past KV capacity the reads are
            # realistic but alias. Aliasing is more cache-friendly than a real
            # batch, so high-context costs here are a LOWER bound.
            "kv_tokens_nominal": num_reqs * (context_len + num_tokens // max(num_reqs, 1)),
                "forward_ms": samples,
                "median_ms": statistics.median(samples) if samples else None,
                "error": error,
            }
        )

    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "cells": results,
        "captured_token_counts": list(runner.cudagraph_manager.captured_token_counts()),
        "max_num_reqs": runner.max_num_reqs,
        "max_num_batched_tokens": runner.max_num_tokens,
        "free_gpu_mb": free_b // 1024**2,
        "total_gpu_mb": total_b // 1024**2,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=8,
                   help="Pins num_reqs: _dummy_run uses min(num_tokens, this).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    p.add_argument("--num-tokens", type=int, nargs="+",
                   default=[16, 32, 64, 128, 256, 512])
    p.add_argument("--context-lens", type=int, nargs="+",
                   default=[0, 256, 512, 1024, 2048, 4096])
    p.add_argument("--baseline-context-len", type=int, default=None,
                   help="Context length Phase A profiles at. Default: vLLM's "
                        "VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN.")
    p.add_argument("--replays", type=int, default=7)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out", default="results/verify_cost_probe.json")
    args = p.parse_args()

    # Deliberately NOT torch here: touching torch.cuda in this process creates a
    # CUDA context that never goes away, and the engine subprocess then sees
    # ~100-300 MB less free memory than it needs for gpu_memory_utilization.
    # nvml reads the same counters without creating a context.
    import pynvml

    pynvml.nvmlInit()
    _h = pynvml.nvmlDeviceGetHandleByIndex(0)
    _m = pynvml.nvmlDeviceGetMemoryInfo(_h)
    free_mb, total_mb = _m.free // 1024**2, _m.total // 1024**2
    pynvml.nvmlShutdown()
    print(f"GPU free {free_mb} / {total_mb} MB")
    if free_mb < total_mb * 0.9:
        print("  WARNING: GPU is not idle. Unload Ollama models "
              "(`ollama stop <model>`) before trusting these timings.",
              file=sys.stderr)
    want_mb = int(args.gpu_memory_utilization * total_mb)
    if want_mb > free_mb:
        print(f"error: --gpu-memory-utilization {args.gpu_memory_utilization} "
              f"asks for {want_mb} MB but only {free_mb} MB is free. "
              f"Try --gpu-memory-utilization {max(0.5, (free_mb - 400) / total_mb):.2f}",
              file=sys.stderr)
        return 2

    # `collective_rpc` with a callable has to ship the function to the engine
    # subprocess, and the default msgspec encoder refuses to serialize a
    # function ("Object of type <class 'function'> is not serializable").
    # This opts into the pickle fallback. It only ever pickles into our own
    # child process, but it must be set before vllm.envs is first read.
    import os

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    import vllm.envs as envs
    from vllm import LLM
    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        build_cost_tables_from_curves,
    )

    if not envs.VLLM_USE_V2_MODEL_RUNNER:
        print("error: set VLLM_USE_V2_MODEL_RUNNER=1 - adaptive verification and "
              "DSpark exist only in the V2 GPU model runner.", file=sys.stderr)
        return 2

    baseline_ctx = (args.baseline_context_len
                    if args.baseline_context_len is not None
                    else envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN)
    print(f"baseline (stock-profiled) context_len = {baseline_ctx}")

    context_lens = sorted({baseline_ctx, *args.context_lens})
    grid = [{"num_tokens": n, "context_len": c}
            for c in context_lens for n in sorted(args.num_tokens)]

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )

    probe = llm.collective_rpc(
        _probe_on_worker, args=(grid, args.replays, args.warmup)
    )[0]

    by_cell = {(c["num_tokens"], c["context_len"]): c for c in probe["cells"]}

    # ---- Phase A: build the table vLLM itself would build ----------------
    verify_curve = [
        (n, by_cell[(n, baseline_ctx)]["median_ms"])
        for n in sorted(args.num_tokens)
        if by_cell[(n, baseline_ctx)]["median_ms"] is not None
    ]
    if not verify_curve:
        print("error: every baseline cell failed - lower --context-lens or "
              "--max-num-seqs so the contexts fit in the KV cache.", file=sys.stderr)
        for c in probe["cells"]:
            if c["error"]:
                print(f"  n={c['num_tokens']} ctx={c['context_len']}: {c['error']}",
                      file=sys.stderr)
        return 1

    captured = probe["captured_token_counts"]
    cudagraph_limit = max(captured) if captured else 0
    # Draft curve is irrelevant here (no drafter); pass a stub so the helper runs.
    _, verify_table = build_cost_tables_from_curves(
        draft_curve=[(1, 0.0)],
        verify_curve=verify_curve,
        max_num_reqs=probe["max_num_reqs"],
        max_batch_tokens=probe["max_num_batched_tokens"],
        cudagraph_limit=cudagraph_limit,
    )

    # ---- Phase B: predicted vs actual ------------------------------------
    rows = []
    for c in probe["cells"]:
        pred = float(verify_table[c["num_tokens"]])
        act = c["median_ms"]
        rows.append({
            **c,
            "predicted_ms": pred,
            "rel_error": None if act is None else (act - pred) / pred,
        })

    print(f"\ncudagraph capture sizes: {captured}")
    print(f"num_reqs pinned to min(num_tokens, {probe['max_num_reqs']})\n")
    print(f"{'n_tok':>6} {'ctx':>7} {'reqs':>5} {'kv_tok*':>8} "
          f"{'pred_ms':>9} {'actual_ms':>10} {'rel_err':>9}")
    print("-" * 62)
    for r in sorted(rows, key=lambda r: (r["num_tokens"], r["context_len"])):
        if r["median_ms"] is None:
            print(f"{r['num_tokens']:>6} {r['context_len']:>7} {r['num_reqs']:>5} "
                  f"{r['kv_tokens_nominal']:>8} {'':>9} {'SKIPPED':>10} "
                  f"  {r['error'].split(':')[0]}")
            continue
        mark = "  <-- baseline" if r["context_len"] == baseline_ctx else ""
        print(f"{r['num_tokens']:>6} {r['context_len']:>7} {r['num_reqs']:>5} "
              f"{r['kv_tokens_nominal']:>8} {r['predicted_ms']:>9.3f} "
              f"{r['median_ms']:>10.3f} {r['rel_error']:>+8.1%}{mark}")

    # ---- The actual verdict ----------------------------------------------
    print("\nSpread at fixed token count (this is the exchangeability test):")
    worst = 0.0
    for n in sorted(args.num_tokens):
        vals = [(r["context_len"], r["median_ms"]) for r in rows
                if r["num_tokens"] == n and r["median_ms"] is not None]
        if len(vals) < 2:
            continue
        lo, hi = min(v for _, v in vals), max(v for _, v in vals)
        spread = (hi - lo) / lo
        worst = max(worst, spread)
        print(f"  n={n:>5}: {lo:.3f} - {hi:.3f} ms  ({spread:+.1%} across "
              f"context_len {min(c for c, _ in vals)}..{max(c for c, _ in vals)})")

    print(f"\nWorst spread at fixed token count: {worst:.1%}")
    print("A one-dimensional cost model predicts 0%. Anything materially above "
          "measurement noise\nfalsifies the exchangeable-token assumption.")

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "baseline_context_len": baseline_ctx,
        "cudagraph_limit": cudagraph_limit,
        "captured_token_counts": captured,
        "verify_curve": verify_curve,
        "gpu": {"free_mb": probe["free_gpu_mb"], "total_mb": probe["total_gpu_mb"]},
        "rows": rows,
        "worst_spread_at_fixed_tokens": worst,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
