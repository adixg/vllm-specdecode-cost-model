#!/usr/bin/env python3
"""Single experiment for vLLM #52057: does forward latency depend on KV context?

Measure vLLM V2 model-runner target-forward CUDA timings at fixed numbers of
new tokens and varying uniform historical context lengths. Compare a fixed-
context lookup (proxy for startup profiling) with a one-scalar KV-aware model.

Not an online profiler, DSpark throughput test, or faithful production trace.
Runs only on a compatible vLLM build with GPUModelRunner._dummy_run and
StepTimingCollector. Tested structurally against the vLLM V2 APIs used in
vLLM 0.28; GPU execution must be validated on the user's PACE installation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path


def probe_worker(self, cells: list[dict], warmup: int, repeats: int, seed: int) -> dict:
    """Executed on a vLLM GPU worker via collective_rpc (one worker/GPU)."""
    import torch

    runner = self.model_runner
    if runner.speculator is not None:
        raise RuntimeError("Run this probe WITHOUT speculative_config: target-only timing")
    if not hasattr(runner, "step_timing") or not hasattr(runner, "_dummy_run"):
        raise RuntimeError("Requires vLLM V2 runner with step_timing and _dummy_run")

    # Every shape must replay a captured CUDA graph. An uncaptured size runs
    # eager, where CPU launch overhead hides the KV-read cost entirely.
    cg = getattr(runner, "cudagraph_manager", None)
    captured = sorted(cg.captured_token_counts()) if cg is not None else []
    missing = sorted({c["num_tokens"] for c in cells} - set(captured))
    if missing:
        raise ValueError(
            f"num_tokens {missing} have no captured CUDA graph "
            f"(captured: {captured}); they would run eager"
        )

    # For standard non-hybrid attention, dummy KV blocks are recycled when
    # the requested synthetic context exceeds available block capacity.
    # Fail instead of silently treating aliased reads as representative.
    kv_cfg = getattr(runner, "kv_cache_config", None)
    cache_cfg = getattr(runner, "cache_config", None)
    nblocks = getattr(kv_cfg, "num_blocks", None)
    block_size = getattr(cache_cfg, "block_size", None)
    kv_capacity = (nblocks * block_size
                   if isinstance(nblocks, int) and isinstance(block_size, int)
                   else None)
    if kv_capacity is not None:
        largest = 0
        for c in cells:
            reqs = min(c["num_tokens"], runner.max_num_reqs)
            per_req = -(-c["num_tokens"] // reqs)
            largest = max(largest, reqs * (c["context_len"] + per_req))
        if largest > kv_capacity:
            raise ValueError(
                f"Dummy KV context could wrap: nominal={largest} tokens, "
                f"KV pool capacity={kv_capacity}. Reduce --max-num-seqs or context."
            )

    rng = random.Random(seed)
    samples = {(c["num_tokens"], c["context_len"]): [] for c in cells}

    # Warm ALL shapes globally before recording. This avoids attributing
    # first-use compilation / CUDA graph setup to a particular cell.
    for _ in range(warmup):
        order = cells[:]
        rng.shuffle(order)
        for cell in order:
            runner._dummy_run(
                num_tokens=cell["num_tokens"],
                context_len=cell["context_len"],
            )
        torch.cuda.synchronize()

    # Interleave randomized shape orders to reduce clock / thermal drift bias.
    for trial in range(repeats):
        order = cells[:]
        rng.shuffle(order)
        for cell in order:
            torch.cuda.synchronize()
            with runner.step_timing.collect() as timings:
                runner._dummy_run(
                    num_tokens=cell["num_tokens"],
                    context_len=cell["context_len"],
                )
                # With no speculator, vLLM's timing collector needs an explicit
                # close of the drafter phase to finalize the forward timing.
                runner.step_timing.drafter_start()
                runner.step_timing.drafter_end()
                torch.cuda.synchronize()
            values = [float(s.forward_ms) for s in timings]
            if len(values) != 1 or not math.isfinite(values[0]) or values[0] <= 0:
                raise RuntimeError(
                    f"Expected exactly one positive forward_ms for {cell}, "
                    f"trial={trial}; got {values}"
                )
            samples[(cell["num_tokens"], cell["context_len"])].append(values[0])

    out = []
    for cell in cells:
        n, ctx = cell["num_tokens"], cell["context_len"]
        values = samples[(n, ctx)]
        num_reqs = min(n, runner.max_num_reqs)
        out.append({
            **cell,
            "num_reqs": num_reqs,
            "nominal_kv_tokens": num_reqs * ctx,
            "samples_ms": values,
            "median_ms": statistics.median(values),
            "stddev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        })

    return {
        "rows": out,
        "captured_token_counts": captured,
        "kv_capacity_tokens": kv_capacity,
        "max_num_reqs": runner.max_num_reqs,
        "max_num_batched_tokens": runner.max_num_tokens,
        "num_kv_blocks": nblocks,
    }


def fit_k(rows: list[dict], baseline: dict[int, float], baseline_ctx: int) -> float:
    """Least-squares slope for T(n,L) = T_base(n) + k * R(n)*(L-L0)."""
    xys = []
    for r in rows:
        x = r["num_reqs"] * (r["context_len"] - baseline_ctx)
        if x:
            xys.append((x, r["median_ms"] - baseline[r["num_tokens"]]))
    denom = sum(x * x for x, _ in xys)
    if denom <= 0:
        raise ValueError("Calibration contexts must differ from the baseline context")
    return sum(x * y for x, y in xys) / denom


def rmse(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")


def self_test() -> None:
    baseline = {8: 2.0, 16: 3.0}
    base_ctx = 8192
    known_k = 0.000012
    rows = [
        {"num_tokens": n, "num_reqs": n, "context_len": ctx,
         "median_ms": baseline[n] + known_k * n * (ctx - base_ctx)}
        for n in baseline for ctx in (512, 4096)
    ]
    estimated = fit_k(rows, baseline, base_ctx)
    assert math.isclose(estimated, known_k, rel_tol=1e-10), (estimated, known_k)
    assert rmse([3.0, 4.0]) == math.sqrt(12.5)
    print("Self-test passed (calibration, RMSE). No GPU required.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--num-tokens", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--baseline-context", type=int, default=8192)
    p.add_argument("--train-contexts", nargs="+", type=int, default=[512, 4096])
    p.add_argument("--test-contexts", nargs="+", type=int, default=[1024, 2048, 12288])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/profile_drift.json")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if min(args.num_tokens) < 1 or max(args.num_tokens) > args.max_num_seqs:
        p.error("Require 1 <= every num-tokens value <= max-num-seqs, "
                "so each request gets exactly one new token")
    if args.warmup < 1 or args.repeats < 3:
        p.error("Need >=1 global warmup and >=3 repeats")
    if len(set(args.num_tokens)) != len(args.num_tokens):
        p.error("Duplicate --num-tokens values")
    ctxs = [args.baseline_context, *args.train_contexts, *args.test_contexts]
    if min(ctxs) < 0 or max(ctxs) + max(args.num_tokens) > args.max_model_len:
        p.error("All contexts must fit max-model-len including new token positions")
    if len(set(ctxs)) != len(ctxs):
        p.error("Baseline, calibration and test context lengths must be disjoint")
    if not 0 < args.gpu_memory_utilization < 1:
        p.error("gpu-memory-utilization must be between 0 and 1")

    # vLLM 0.28's worker RPC serializes module-level callables through pickle.
    # Enable only for a trusted local engine process. Do not use with untrusted RPC.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    # Do NOT initialize torch.cuda in the parent before creating the engine:
    # parent and child CUDA contexts compete for available GPU memory.
    import vllm
    from vllm import LLM

    split = {args.baseline_context: "baseline"}
    split.update({c: "train" for c in args.train_contexts})
    split.update({c: "test" for c in args.test_contexts})
    cells = [
        {"num_tokens": n, "context_len": c, "split": split[c]}
        for n in sorted(args.num_tokens) for c in sorted(split)
    ]

    print(f"Model: {args.model} | vLLM: {vllm.__version__}", flush=True)
    print(f"{len(cells)} shapes x {args.repeats} repeated measurements", flush=True)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        tensor_parallel_size=1,
    )
    responses = llm.collective_rpc(
        probe_worker,
        args=(cells, args.warmup, args.repeats, args.seed),
    )
    if len(responses) != 1:
        raise RuntimeError(f"Expected one GPU worker, got {len(responses)}")
    measured = responses[0]
    rows = measured["rows"]
    baseline = {
        r["num_tokens"]: r["median_ms"]
        for r in rows if r["split"] == "baseline"
    }
    k = fit_k([r for r in rows if r["split"] == "train"],
              baseline, args.baseline_context)

    for r in rows:
        fixed = baseline[r["num_tokens"]]
        corrected = fixed + k * r["num_reqs"] * (r["context_len"] - args.baseline_context)
        r["fixed_prediction_ms"] = fixed
        r["context_prediction_ms"] = corrected
        r["fixed_error_ms"] = r["median_ms"] - fixed
        r["context_error_ms"] = r["median_ms"] - corrected
        r["fixed_error_pct"] = 100 * (r["median_ms"] - fixed) / fixed
        r["context_error_pct"] = (
            100 * (r["median_ms"] - corrected) / corrected if corrected > 0 else None
        )

    test = [r for r in rows if r["split"] == "test"]
    fixed_rmse = rmse([r["fixed_error_ms"] for r in test])
    context_rmse = rmse([r["context_error_ms"] for r in test])
    summary = {
        "k_ms_per_nominal_kv_token": k,
        "k_us_per_nominal_kv_token": 1000 * k,
        "test_fixed_rmse_ms": fixed_rmse,
        "test_context_rmse_ms": context_rmse,
        "test_rmse_reduction_pct": 100 * (1 - context_rmse / fixed_rmse)
        if fixed_rmse > 0 else None,
    }
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "fixed-context profiling drift; context-aware 1-scalar correction",
        "vllm_version": vllm.__version__,
        "args": vars(args),
        "worker": {k: v for k, v in measured.items() if k != "rows"},
        "summary": summary,
        "rows": rows,
        "limitations": [
            "This is a synthetic uniform-context dummy forward, not DSpark serving.",
            "Baseline is the measured fixed-context per-shape lookup, not the "
            "entire startup cost curve reconstructed by vLLM.",
            "No exploration, online estimator, outlier policy, or oracle replay.",
            "Dummy KV indices may wrap if nominal KV tokens exceed cache capacity.",
            "Timing is vLLM forward_ms (CUDA events), not request latency.",
            "One new token per request (decode-shaped), not multi-token "
            "verification steps.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    with out.with_suffix(".csv").open("w", newline="") as f:
        cols = ["split", "num_tokens", "num_reqs", "context_len", "nominal_kv_tokens",
                "median_ms", "stddev_ms", "fixed_prediction_ms", "context_prediction_ms",
                "fixed_error_pct", "context_error_pct"]
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["num_tokens"], r["context_len"])))

    print("\n  TOKENS   CTX   SPLIT   ACTUAL(ms)  FIXED(ms)  KV-AWARE(ms)  FIXED ERR(%)")
    for r in sorted(rows, key=lambda x: (x["num_tokens"], x["context_len"])):
        print(f"{r['num_tokens']:>8} {r['context_len']:>5} {r['split']:>8} "
              f"{r['median_ms']:>12.3f} {r['fixed_prediction_ms']:>10.3f} "
              f"{r['context_prediction_ms']:>13.3f} {r['fixed_error_pct']:>13.1f}")
    print(f"\nCalibrated k: {summary['k_us_per_nominal_kv_token']:.6f} us/KV-token")
    print(f"HELD-OUT RMSE: fixed={fixed_rmse:.4f} ms, "
          f"context-aware={context_rmse:.4f} ms")
    if summary["test_rmse_reduction_pct"] is not None:
        print(f"Change in RMSE: {summary['test_rmse_reduction_pct']:+.1f}% reduction")
    print(f"Saved {out} and {out.with_suffix('.csv')}")
    if measured.get("kv_capacity_tokens") is None:
        print("NOTE: KV cache capacity not exposed; check dummy-block wrapping manually")


if __name__ == "__main__":
    main()

