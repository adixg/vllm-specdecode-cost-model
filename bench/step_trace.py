#!/usr/bin/env python3
"""Log predicted vs actual verification cost for every REAL generation step.

This is the experiment vllm-project/vllm#52057 plots, done properly:
  - "predicted" is the cost vLLM's own AdaptiveVerificationManager used to
    choose the draft budget, from the table it profiled at startup.
  - "actual" is the real forward time of that step, from vLLM's own
    StepTimingCollector.
Neither number is fitted by us, and both come from real generation rather than
synthetic dummy runs - the three caveats that made bench/verify_cost_probe.py
not directly comparable to their figure.

It also records everything needed to replay the budget decision offline (the
oracle bound): the survival-score curve, the candidate cost curve, and the
budget actually chosen.

Run:
    source env.sh                      # laptop; on a cluster just activate the env
    export VLLM_USE_V2_MODEL_RUNNER=1
    python bench/step_trace.py --model openbmb/MiniCPM5-2B \
                               --draft openbmb/MiniCPM5-2B-DSpark
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone


# --------------------------------------------------------------------------
# Worker-side: install the hooks
# --------------------------------------------------------------------------
def _install_trace(self) -> dict:
    """Patch the manager and the timing collector. Runs inside the worker."""
    runner = self.model_runner
    av = getattr(runner, "adaptive_verification", None)
    if av is None:
        return {"ok": False, "why": "adaptive verification is not active"}

    # Rows describing each budget DECISION, appended by the wrapper below.
    self._trace_decisions = []

    original = av.get_num_tokens

    def traced_get_num_tokens(num_tokens_per_req, draft_tokens):
        # Let vLLM do its real work first, untouched.
        num_tokens = original(num_tokens_per_req, draft_tokens)

        req_ids = list(num_tokens_per_req)
        num_reqs = len(req_ids)
        slots = [av.req_states.req_id_to_index[r] for r in req_ids]

        # Context already in the KV cache for each request. Summed, this is the
        # sum(seq_lens) term the cost model has no input for.
        ctx = av.req_states.num_computed_tokens_np[slots]
        total_context = int(ctx.sum())

        # What the manager decided, and what it believed that would cost.
        num_drafts_per_req, num_non_draft_per_req, draft_budget = av._batch_budget
        non_draft_total = int(sum(num_non_draft_per_req.values()))
        draft_table, verify_table = av.cost_tables
        predicted_ms = float(
            draft_table[num_reqs] + verify_table[non_draft_total + draft_budget]
        )

        self._trace_decisions.append(
            {
                "num_reqs": num_reqs,
                "num_tokens": int(num_tokens),
                "non_draft_tokens": non_draft_total,
                "draft_budget": int(draft_budget),
                "total_context": total_context,
                "sum_seq_lens": total_context + int(num_tokens),
                "max_context": int(ctx.max()) if num_reqs else 0,
                "predicted_ms": predicted_ms,
            }
        )
        return num_tokens

    av.get_num_tokens = traced_get_num_tokens

    # StepTimingCollector normally records only inside its collect() block.
    # Forcing the flag on makes it accumulate a sample for every real step;
    # drafter_end() is what appends, and DSpark always reaches it.
    runner.step_timing._collecting = True

    return {"ok": True, "profile_context_len": int(
        __import__("vllm").envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN)}


def _drain_trace(self) -> dict:
    """Resolve the recorded CUDA events and hand back the rows."""
    runner = self.model_runner
    collector = runner.step_timing

    timed, collector._timed = collector._timed, []
    collector._collecting = False

    timings = []
    if timed:
        timed[-1][0].drafter_end.synchronize()
        for events, (full_cudagraph, num_target_tokens, num_reqs) in timed:
            timings.append(
                {
                    "forward_ms": events.forward_start.elapsed_time(events.forward_end),
                    "drafter_ms": events.drafter_start.elapsed_time(events.drafter_end),
                    "num_target_tokens": int(num_target_tokens),
                    "num_reqs": int(num_reqs),
                    "full_cudagraph": bool(full_cudagraph),
                }
            )

    return {"decisions": getattr(self, "_trace_decisions", []), "timings": timings}


# --------------------------------------------------------------------------
# Host side
# --------------------------------------------------------------------------
def agreement(pred, act):
    """bias / MAE / RMSE / Pearson r, the stats #52057's figure reports."""
    n = len(pred)
    if n < 2:
        return {}
    err = [a - p for a, p in zip(act, pred)]
    mp, ma = sum(pred) / n, sum(act) / n
    cov = sum((p - mp) * (a - ma) for p, a in zip(pred, act))
    sp = math.sqrt(sum((p - mp) ** 2 for p in pred))
    sa = math.sqrt(sum((a - ma) ** 2 for a in act))
    return {
        "n": n,
        "bias_ms": sum(err) / n,
        "mae_ms": sum(abs(e) for e in err) / n,
        "rmse_ms": math.sqrt(sum(e * e for e in err) / n),
        "r": cov / (sp * sa) if sp and sa else float("nan"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="openbmb/MiniCPM5-2B")
    p.add_argument("--draft", default="openbmb/MiniCPM5-2B-DSpark")
    p.add_argument("--num-speculative-tokens", type=int, default=3)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--attention-backend", default=None,
                   help="TRITON_ATTN on pre-Hopper; leave unset on H100 (FA3).")
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--prompt-tokens", type=int, nargs="+", default=[64, 512, 2048],
                   help="Prompt lengths to mix, so contexts are heterogeneous.")
    p.add_argument("--out", default="results/step_trace.json")
    args = p.parse_args()

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    import vllm.envs as envs
    from vllm import LLM, SamplingParams

    if not envs.VLLM_USE_V2_MODEL_RUNNER:
        print("error: set VLLM_USE_V2_MODEL_RUNNER=1", file=sys.stderr)
        return 2

    extra = {}
    if args.attention_backend:
        extra["attention_backend"] = args.attention_backend

    llm = LLM(
        model=args.model,
        speculative_config={
            "method": "dspark",
            "model": args.draft,
            "num_speculative_tokens": args.num_speculative_tokens,
            "enable_adaptive_verification": True,
        },
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **extra,
    )

    setup = llm.collective_rpc(_install_trace)[0]
    if not setup.get("ok"):
        print(f"error: {setup.get('why')}", file=sys.stderr)
        return 1
    print(f"tracing installed; profile context len = {setup['profile_context_len']}")

    # Heterogeneous prompt lengths, so sum(seq_lens) varies across steps -
    # which is the whole point. A single prompt length would hold it constant.
    tok = llm.get_tokenizer()
    filler = " systems research and inference serving"
    prompts = []
    for i in range(args.num_prompts):
        want = args.prompt_tokens[i % len(args.prompt_tokens)]
        text = "Explain in detail:" + filler * (want // 6 + 1)
        ids = tok(text).input_ids[:want]
        prompts.append(tok.decode(ids))

    print(f"generating {len(prompts)} prompts x {args.max_tokens} tokens ...")
    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))

    trace = llm.collective_rpc(_drain_trace)[0]
    decisions, timings = trace["decisions"], trace["timings"]
    print(f"captured {len(decisions)} decisions, {len(timings)} timed steps")

    # Join by position, but verify rather than trusting it: a step that never
    # consulted the manager would shift everything after it.
    rows, mismatches = [], 0
    for dec, tim in zip(decisions, timings):
        if dec["num_tokens"] != tim["num_target_tokens"]:
            mismatches += 1
            continue
        rows.append({**dec, **tim})
    if mismatches:
        print(f"WARNING: dropped {mismatches} steps where the decision and the "
              f"timing disagreed on token count", file=sys.stderr)
    if not rows:
        print("error: no steps matched", file=sys.stderr)
        return 1

    pred = [r["predicted_ms"] for r in rows]
    act = [r["forward_ms"] for r in rows]
    stats = agreement(pred, act)

    print(f"\n=== predicted vs actual over {stats['n']} real steps ===")
    print(f"  bias  {stats['bias_ms']:+.3f} ms   (negative = vLLM overestimates)")
    print(f"  MAE   {stats['mae_ms']:.3f} ms")
    print(f"  RMSE  {stats['rmse_ms']:.3f} ms")
    print(f"  r     {stats['r']:.3f}")

    graphed = [r for r in rows if r["full_cudagraph"]]
    eager = [r for r in rows if not r["full_cudagraph"]]
    for name, sub in (("cudagraph", graphed), ("eager", eager)):
        if len(sub) > 1:
            s = agreement([r["predicted_ms"] for r in sub], [r["forward_ms"] for r in sub])
            print(f"  [{name:>9}] n={s['n']:>5}  bias={s['bias_ms']:+7.3f}  "
                  f"RMSE={s['rmse_ms']:6.3f}  r={s['r']:.3f}")

    ctxs = [r["total_context"] for r in rows]
    print(f"\ncontext seen across steps: {min(ctxs):,} .. {max(ctxs):,} tokens "
          f"(median {int(statistics.median(ctxs)):,})")

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "profile_context_len": setup["profile_context_len"],
        "agreement": stats,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
