#!/usr/bin/env python3
"""Experiment 2: is verification cost dependent on *which* request gets the tokens?

`verify_cost_probe.py` established that step cost decomposes as

    cost ~= f(num_query_tokens) + k * sum(seq_lens)

with the second term streaming KV at a near-constant ~181 GB/s.  If that holds,
the KV term depends on which *requests* are in the batch, not on how query
tokens are distributed among them - a request's KV is read in full whether it
contributes one token or seven.  Marginal verification cost would then be
roughly uniform across draft slots, and the proposal's cost-aware per-request
allocation would degenerate to survival-only allocation.

This tests that directly.  Every configuration holds constant:
  - the set of requests and their contexts (so total KV bytes is identical)
  - the total number of query tokens (so total "verification budget" matches)
and varies only *which* requests receive the extra tokens.

  long   all spare tokens to the long-context requests
  short  all spare tokens to the short-context requests
  even   spread uniformly

A one-dimensional cost model predicts these are identical - and so does the
two-term model above.  Allocation-dependent cost predicts they differ.

Run:
    source env.sh
    export VLLM_USE_V2_MODEL_RUNNER=1
    python bench/marginal_cost_probe.py --model Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone


def _probe_on_worker(self, plans: list[dict], replays: int, warmup: int) -> dict:
    """Runs inside the vLLM worker. `self` is the Worker."""
    import numpy as np
    import torch

    import vllm.v1.worker.gpu.model_runner as mr
    from vllm.utils.math_utils import cdiv

    runner = self.model_runner
    original_set_dummy_context = mr.set_dummy_context

    # The plan the next _dummy_run should realise. `_dummy_run` builds an
    # even-split, uniform-context batch; the patch below rewrites it in place
    # after prepare_dummy_attn and before attention metadata is built, which
    # is the one point where both the batch and the block tables are reachable.
    current: dict = {}

    def patched_set_dummy_context(
        input_batch, block_tables, context_len, num_kv_blocks, max_model_len
    ):
        query_lens = np.asarray(current["query_lens"], dtype=np.int32)
        contexts = np.asarray(current["contexts"], dtype=np.int32)
        num_reqs = input_batch.num_reqs
        assert len(query_lens) == len(contexts) == num_reqs
        # Keep every request inside the model's window.
        contexts = np.minimum(contexts, max_model_len - query_lens)

        device = input_batch.seq_lens.device
        qsl = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(query_lens, out=qsl[1:])

        input_batch.num_scheduled_tokens = query_lens
        input_batch.query_start_loc_np[:] = qsl
        input_batch.query_start_loc[: num_reqs + 1] = torch.from_numpy(qsl).to(device)
        input_batch.query_start_loc[num_reqs + 1 :] = int(qsl[-1])
        input_batch.max_query_len = int(query_lens.max())

        seq_lens = query_lens + contexts
        input_batch.seq_lens[:num_reqs] = torch.from_numpy(seq_lens).to(device)
        input_batch.seq_lens_cpu_upper_bound[:num_reqs] = torch.from_numpy(seq_lens)
        input_batch.num_computed_tokens_np[:] = contexts
        input_batch.num_computed_prefill_tokens_np[:] = contexts

        # Position of each token = its offset within its request + that
        # request's context length (mirrors the stock set_dummy_context).
        local_pos = np.arange(int(qsl[-1]), dtype=np.int64) - np.repeat(
            qsl[:-1], query_lens
        )
        input_batch.positions[: int(qsl[-1])].copy_(
            torch.from_numpy(local_pos + np.repeat(contexts, query_lens))
        )
        input_batch.logits_indices = (
            input_batch.query_start_loc[1 : num_reqs + 1] - 1
        )

        # Per-request block spans. Same wrap-modulo-pool trick as upstream:
        # reads stay realistic, blocks alias once the pool is exhausted.
        for block_table, block_size, bpk in zip(
            block_tables.input_block_tables,
            block_tables.kernel_block_sizes,
            block_tables.blocks_per_kv_block,
        ):
            cursor = 0
            for i in range(num_reqs):
                nb = min(cdiv(int(seq_lens[i]), block_size), block_table.shape[1])
                ids = (
                    torch.arange(cursor, cursor + nb, dtype=block_table.dtype,
                                 device=block_table.device)
                    % (num_kv_blocks * bpk)
                )
                block_table[i, :nb] = ids
                cursor += nb

    samples_by_label: dict[str, list[float]] = {p["label"]: [] for p in plans}
    errors: dict[str, str] = {}

    def run_once(plan) -> list[float]:
        current.clear()
        current.update(plan)
        total_tokens = int(sum(plan["query_lens"]))
        with runner.step_timing.collect() as timings:
            runner._dummy_run(num_tokens=total_tokens, context_len=1)
            if runner.speculator is None:
                runner.step_timing.drafter_start()
                runner.step_timing.drafter_end()
        return [s.forward_ms for s in timings]

    results = []
    try:
        mr.set_dummy_context = patched_set_dummy_context

        # Warm every shape before timing anything: each distinct max_query_len
        # triggers its own compile/autotune, and a plan measured first would
        # otherwise absorb one-time cost the others do not pay.
        for _ in range(warmup):
            for plan in plans:
                try:
                    run_once(plan)
                except Exception as exc:
                    errors.setdefault(plan["label"], f"{type(exc).__name__}: {exc}")
        torch.cuda.synchronize()

        # Interleave: one replay of every plan per round, so any residual
        # drift hits all plans equally instead of penalising whichever ran
        # first.
        for _ in range(replays):
            for plan in plans:
                if plan["label"] in errors:
                    continue
                try:
                    samples_by_label[plan["label"]].extend(run_once(plan))
                except Exception as exc:
                    errors[plan["label"]] = f"{type(exc).__name__}: {exc}"

        for plan in plans:
            samples = samples_by_label[plan["label"]]
            error = errors.get(plan["label"])
            total_tokens = int(sum(plan["query_lens"]))
            results.append({
                "label": plan["label"],
                "query_lens": list(map(int, plan["query_lens"])),
                "contexts": list(map(int, plan["contexts"])),
                "total_query_tokens": total_tokens,
                "total_kv_tokens": int(sum(plan["contexts"])),
                "forward_ms": samples,
                "median_ms": statistics.median(samples) if samples else None,
                "error": error,
            })
    finally:
        mr.set_dummy_context = original_set_dummy_context

    return {"cells": results}


def build_plans(num_long, num_short, ctx_long, ctx_short, base, spare):
    """Same requests, same contexts, same total query tokens; different owners."""
    n = num_long + num_short
    contexts = [ctx_long] * num_long + [ctx_short] * num_short

    def plan(label, extra):
        return {"label": label,
                "query_lens": [base + e for e in extra],
                "contexts": contexts}

    assert spare % max(num_long, num_short) == 0 or True
    per_long = spare // num_long
    per_short = spare // num_short
    plans = [
        plan("even", [spare // n] * n),
        # long vs short is the clean test: identical multiset of query lengths
        # (so identical max_query_len and identical total tokens), identical
        # contexts, differing only in WHICH requests own the long queries.
        plan("long", [per_long] * num_long + [0] * num_short),
        plan("short", [0] * num_long + [per_short] * num_short),
        # Drift control: byte-identical to "even". Any gap between the two is
        # residual warmup/thermal drift, not an allocation effect.
        plan("even#2", [spare // n] * n),
    ]
    # Only compare plans that really do carry the same token total.
    totals = {sum(p["query_lens"]) for p in plans}
    if len(totals) != 1:
        raise SystemExit(f"plans disagree on total tokens: {totals}. "
                         f"Choose --spare divisible by {num_long}, {num_short} and {n}.")
    return plans


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    p.add_argument("--num-long", type=int, default=4)
    p.add_argument("--num-short", type=int, default=4)
    p.add_argument("--ctx-long", type=int, default=4096)
    p.add_argument("--ctx-short", type=int, default=256)
    p.add_argument("--base", type=int, default=1, help="Query tokens every request gets.")
    p.add_argument("--spare", type=int, default=24, help="Draft tokens to allocate.")
    p.add_argument("--replays", type=int, default=15)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--out", default="results/marginal_cost_probe.json")
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
    from vllm import LLM

    if not envs.VLLM_USE_V2_MODEL_RUNNER:
        print("error: set VLLM_USE_V2_MODEL_RUNNER=1", file=sys.stderr)
        return 2

    if args.num_long + args.num_short > args.max_num_seqs:
        print("error: num-long + num-short must be <= --max-num-seqs", file=sys.stderr)
        return 2

    plans = build_plans(args.num_long, args.num_short, args.ctx_long,
                        args.ctx_short, args.base, args.spare)

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs,
              gpu_memory_utilization=args.gpu_memory_utilization,
              enforce_eager=False)

    probe = llm.collective_rpc(_probe_on_worker,
                               args=(plans, args.replays, args.warmup))[0]

    cells = probe["cells"]
    print(f"\n{args.num_long} requests at ctx={args.ctx_long}, "
          f"{args.num_short} at ctx={args.ctx_short}")
    print(f"every configuration: {cells[0]['total_query_tokens']} query tokens, "
          f"{cells[0]['total_kv_tokens']} KV tokens\n")
    print(f"{'alloc':>6} {'query_lens':>28} {'median_ms':>10} {'vs even':>9}")
    print("-" * 58)
    base = next((c["median_ms"] for c in cells if c["label"] == "even"), None)
    for c in cells:
        if c["median_ms"] is None:
            print(f"{c['label']:>6} {'':>28} {'FAILED':>10}   {c['error']}")
            continue
        delta = "" if not base else f"{(c['median_ms']-base)/base:+8.1%}"
        print(f"{c['label']:>6} {str(c['query_lens']):>28} "
              f"{c['median_ms']:>10.3f} {delta:>9}")

    ok = [c for c in cells if c["median_ms"]]
    verdict = None
    if len(ok) >= 2:
        lo = min(c["median_ms"] for c in ok)
        hi = max(c["median_ms"] for c in ok)
        spread = (hi - lo) / lo
        noise = max((max(c["forward_ms"]) - min(c["forward_ms"])) / c["median_ms"]
                    for c in ok)
        within = statistics.median(
            (max(c["forward_ms"]) - min(c["forward_ms"])) / c["median_ms"] for c in ok)
        print(f"\nspread across allocations : {spread:.2%}")
        print(f"median within-cell spread : {within:.2%}  (worst {noise:.2%})")
        verdict = "uniform" if spread <= within else "allocation-dependent"
        print(f"\nverdict: marginal cost looks {verdict.upper()}")
        if verdict == "uniform":
            print("Cost-aware per-request allocation has no cost signal to exploit\n"
                  "in this regime; it would reduce to survival-only allocation.")

    out = {"generated_utc": datetime.now(timezone.utc).isoformat(),
           "args": vars(args), "cells": cells, "verdict": verdict}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
