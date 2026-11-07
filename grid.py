#!/usr/bin/env python3
"""Is the price of a draft slot linear in the number of draft slots?

Open question on vLLM PR #62960: it prices a verification step as
    base(R, context) + slot_price[R] * b
with slot_price[R] measured once at the full query length (T = 1 + K). This
script measures target forward time on a grid of
    R requests x T tokens per request x context length
and reports, for a dense model:
  * the measured per-slot price (cost(R,T) - cost(R,1)) / (R*(T-1)) at each T
  * the error of a linear price fitted at T = 1 + K (what #62960 does)
  * the price stock adaptive verification charges, from its startup table

How the shape is forced: stock _dummy_run(num_tokens=N) splits N tokens over
min(N, max_num_seqs) requests. Running one engine with max_num_seqs = R and
calling _dummy_run(R*T) therefore gives exactly R requests x T tokens. The
script checks that each shape replays an exact FULL CUDA graph, and records
the dummy batch vLLM actually built, failing if it is not R x T.

One engine per R (run as a Slurm array). Timing is vLLM forward_ms (CUDA
events), which excludes logits and sampling, the same as the startup profile.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path


def probe_worker(self, R: int, cells: list[dict], warmup: int, repeats: int,
                 seed: int) -> dict:
    """Executed on the vLLM GPU worker via collective_rpc."""
    import torch
    import vllm.envs as envs
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.input_batch import InputBatch

    runner = self.model_runner
    av = getattr(runner, "adaptive_verification", None)
    if av is None or av.cost_tables is None:
        raise RuntimeError("Adaptive verification built no cost table")
    if runner.max_num_reqs != R:
        raise RuntimeError(f"max_num_reqs={runner.max_num_reqs}, expected {R}")

    # Each R x T shape must replay a FULL graph at exactly R*T tokens, with
    # the request count the dummy batch will be built with equal to R.
    cg = runner.cudagraph_manager
    captured = sorted(cg.captured_token_counts())
    bad = {}
    for T in sorted({c["T"] for c in cells}):
        n = R * T
        desc = cg.dispatch(R, n, T, num_active_loras=0, max_query_len=T)
        if (desc.cg_mode != CUDAGraphMode.FULL or desc.num_tokens != n
                or (desc.num_reqs or R) != R):
            bad[T] = f"{desc.cg_mode.name} tokens={desc.num_tokens} reqs={desc.num_reqs}"
    if bad:
        raise ValueError(f"R={R}: shapes not on an exact FULL graph: {bad} "
                         f"(captured: {captured})")

    nblocks = runner.kv_cache_config.num_blocks
    block_size = runner.cache_config.block_size
    kv_capacity = nblocks * block_size
    need = max(R * (c["ctx"] + c["T"]) for c in cells)
    if need > kv_capacity:
        raise ValueError(f"Dummy KV would wrap: need {need}, capacity {kv_capacity}")

    # Record the dummy batch vLLM actually builds for each run.
    built = []
    orig = InputBatch.make_dummy.__func__

    def recording_make_dummy(cls, num_reqs, num_tokens, *a, **kw):
        ib = orig(cls, num_reqs, num_tokens, *a, **kw)
        built.append((ib.num_reqs, ib.num_tokens,
                      int(ib.num_scheduled_tokens.min()),
                      int(ib.num_scheduled_tokens.max())))
        return ib

    InputBatch.make_dummy = classmethod(recording_make_dummy)
    try:
        rng = random.Random(seed)
        for _ in range(warmup):
            order = cells[:]
            rng.shuffle(order)
            for c in order:
                runner._dummy_run(num_tokens=R * c["T"], context_len=c["ctx"])
            torch.cuda.synchronize()

        samples = {(c["T"], c["ctx"]): [] for c in cells}
        for trial in range(repeats):
            order = cells[:]
            rng.shuffle(order)
            for c in order:
                built.clear()
                torch.cuda.synchronize()
                with runner.step_timing.collect() as timings:
                    runner._dummy_run(num_tokens=R * c["T"], context_len=c["ctx"])
                    torch.cuda.synchronize()
                # The target forward builds its batch first; the drafter's
                # dummy propose may build more after it.
                if not built or built[0] != (R, R * c["T"], c["T"], c["T"]):
                    raise RuntimeError(
                        f"Expected dummy batch {R} reqs x {c['T']} tokens, "
                        f"got (reqs, tokens, min_q, max_q)={built[:1]}")
                values = [float(s.forward_ms) for s in timings]
                if len(values) != 1 or not math.isfinite(values[0]) or values[0] <= 0:
                    raise RuntimeError(f"Bad forward_ms for {c}, trial {trial}: {values}")
                samples[(c["T"], c["ctx"])].append(values[0])
    finally:
        InputBatch.make_dummy = classmethod(orig)

    rows = []
    for c in cells:
        v = samples[(c["T"], c["ctx"])]
        rows.append({**c, "R": R, "num_tokens": R * c["T"], "samples_ms": v,
                     "median_ms": statistics.median(v),
                     "stddev_ms": statistics.stdev(v)})
    return {
        "rows": rows,
        "startup_verify_cost_ms": [float(x) for x in av.cost_tables[1]],
        "profile_context_len": envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN,
        "captured_token_counts": captured,
        "kv_capacity_tokens": kv_capacity,
    }


def analyze(R: int, rows: list[dict], table: list[float], t_max: int) -> list[dict]:
    """Per (ctx, T>=2): measured slot price, linear-at-t_max error, stock price."""
    by = {(r["ctx"], r["T"]): r["median_ms"] for r in rows}
    out = []
    for ctx in sorted({r["ctx"] for r in rows}):
        base = by[(ctx, 1)]
        if (ctx, t_max) not in by:
            raise ValueError(f"Need T={t_max} at ctx {ctx} to fit the linear price")
        slot_lin = (by[(ctx, t_max)] - base) / (R * (t_max - 1))
        for T in sorted(t for (c, t) in by if c == ctx and t >= 2):
            slots = R * (T - 1)
            meas = by[(ctx, T)]
            lin = base + slot_lin * slots
            row = {
                "R": R, "ctx": ctx, "T": T, "slots": slots,
                "measured_ms": meas,
                "slot_price_us": 1000 * (meas - base) / slots,
                "linear_pred_ms": lin,
                "linear_error_pct": 100 * (lin - meas) / meas,
            }
            # Stock AV prices R decode requests with b draft slots as
            # verify_cost_ms[R + b]; here R + b = R*T.
            if R * T < len(table):
                row["stock_pred_ms"] = table[R * T]
                row["stock_slot_price_us"] = 1000 * (table[R * T] - table[R]) / slots
            out.append(row)
    return out


def print_analysis(results: list[dict], t_max: int) -> None:
    print(f"\n{'R':>4} {'ctx':>6} {'T':>2} {'meas ms':>8} {'slot us':>8} "
          f"{'stock us':>9} {'lin@T=' + str(t_max) + ' err':>13} {'stock err':>10}")
    for a in results:
        stock = a.get("stock_pred_ms")
        s_err = f"{100 * (stock - a['measured_ms']) / a['measured_ms']:+9.1f}%" if stock else f"{'-':>10}"
        s_us = f"{a['stock_slot_price_us']:9.1f}" if stock else f"{'-':>9}"
        print(f"{a['R']:>4} {a['ctx']:>6} {a['T']:>2} {a['measured_ms']:8.3f} "
              f"{a['slot_price_us']:8.2f} {s_us} {a['linear_error_pct']:+12.1f}% {s_err}")
    errs = [abs(a["linear_error_pct"]) for a in results if a["T"] != t_max]
    if errs:
        print(f"\nLinear price fitted at T={t_max}: |error| on other T "
              f"median {statistics.median(errs):.1f}%, max {max(errs):.1f}%")
    print("err = (prediction - measured) / measured; positive = overpriced")


def summarize(paths: list[str]) -> None:
    results, t_max = [], None
    for p in sorted(paths):
        d = json.loads(Path(p).read_text())
        results += d["analysis"]
        t_max = d["args"]["t_max"]
    results.sort(key=lambda a: (a["R"], a["ctx"], a["T"]))
    print_analysis(results, t_max)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summarize", nargs="+", metavar="JSON",
                   help="Print the combined analysis of finished runs and exit")
    p.add_argument("--model", default="openbmb/MiniCPM5-2B")
    p.add_argument("--draft", default="openbmb/MiniCPM5-2B-DSpark")
    p.add_argument("--num-speculative-tokens", type=int, default=7)
    p.add_argument("--num-reqs", type=int, help="R; also sets max_num_seqs")
    p.add_argument("--tokens-per-req", nargs="+", type=int,
                   default=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--contexts", nargs="+", type=int, default=[0, 1024, 4096, 8192])
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/slot_grid.json")
    args = p.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return
    if args.num_reqs is None or args.num_reqs < 1:
        p.error("--num-reqs is required")
    args.t_max = 1 + args.num_speculative_tokens
    ts = sorted(set(args.tokens_per_req))
    if ts[0] != 1 or ts[-1] != args.t_max or len(ts) != len(args.tokens_per_req):
        p.error(f"--tokens-per-req must be distinct and include 1 and {args.t_max}")
    if max(args.contexts) + args.t_max > args.max_model_len or min(args.contexts) < 0:
        p.error("Contexts must fit max-model-len")
    if args.repeats < 3 or args.warmup < 1:
        p.error("Need >=1 warmup and >=3 repeats")

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    import vllm
    from vllm import LLM

    R = args.num_reqs
    cells = [{"T": t, "ctx": c} for t in ts for c in sorted(set(args.contexts))]
    print(f"Model: {args.model} + {args.draft} | vLLM {vllm.__version__} | R={R}",
          flush=True)
    print(f"{len(cells)} shapes x {args.repeats} repeats", flush=True)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=R,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        tensor_parallel_size=1,
        speculative_config={
            "method": "dspark",
            "model": args.draft,
            "num_speculative_tokens": args.num_speculative_tokens,
            "enable_adaptive_verification": True,
        },
    )
    responses = llm.collective_rpc(
        probe_worker, args=(R, cells, args.warmup, args.repeats, args.seed))
    if len(responses) != 1:
        raise RuntimeError(f"Expected one GPU worker, got {len(responses)}")
    measured = responses[0]
    analysis = analyze(R, measured["rows"], measured["startup_verify_cost_ms"],
                       args.t_max)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "draft-slot price linearity (PR #62960 review question)",
        "vllm_version": vllm.__version__,
        "args": vars(args),
        "worker": {k: v for k, v in measured.items() if k != "rows"},
        "rows": measured["rows"],
        "analysis": analysis,
        "limitations": [
            "Dummy forwards with padding-token inputs, not serving traffic.",
            "max_num_seqs = R, so no padded request slots, unlike serving at "
            "a larger max_num_seqs.",
            "forward_ms excludes logits and sampling over the R*T positions.",
            "Uniform T and uniform context across requests.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print_analysis(analysis, args.t_max)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
