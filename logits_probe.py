#!/usr/bin/env python3
"""How much does the end of a verification step cost, outside forward_ms?

vLLM's adaptive-verification cost table is built from forward_ms of dummy
runs, which never reach compute_logits or the rejection sampler. Both run on
every real step and grow with the number of verified tokens. This script
times them inside real DSpark serving:

  * execute_model: GPU time of input prep + target forward
  * compute_logits: the vocabulary projection over every verified position
  * sample: compute_logits + rejection sampler (all of runner.sample)

CUDA events wrap each call on the worker, so the times are GPU time on the
main stream, recorded without extra syncs; they are read after generation.
Each decode step records its request count and number of logits rows
(requests + verified drafts). Run at several concurrencies in one engine.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path


TASKS = ["Write a detailed essay about", "Explain step by step how",
         "Write a long story about", "Give a thorough history of",
         "Describe in detail the science behind"]
TOPICS = ["the water cycle", "how compilers work", "the Roman Empire",
          "photosynthesis", "a lighthouse keeper", "public key cryptography",
          "the invention of the printing press", "how GPUs execute programs",
          "a journey across the Sahara", "black holes", "the French Revolution",
          "how vaccines train the immune system"]


def install_timers(self) -> dict:
    """Runs on the worker: wrap execute_model, sample and compute_logits."""
    import torch

    runner = self.model_runner
    if runner.speculator is None or runner.adaptive_verification is None:
        raise RuntimeError("Expected DSpark with adaptive verification")
    state = {"pending_forward": None, "records": [], "logits_events": None}
    runner._logits_probe = state

    def ev():
        return torch.cuda.Event(enable_timing=True)

    orig_execute = runner.execute_model
    orig_sample = runner.sample
    model = runner.model
    orig_logits = model.compute_logits

    def timed_execute(*args, **kwargs):
        if kwargs.get("dummy_run"):
            return orig_execute(*args, **kwargs)
        start, end = ev(), ev()
        start.record()
        out = orig_execute(*args, **kwargs)
        end.record()
        state["pending_forward"] = (start, end)
        return out

    def timed_logits(hidden_states, *args, **kwargs):
        start, end = ev(), ev()
        start.record()
        out = orig_logits(hidden_states, *args, **kwargs)
        end.record()
        state["logits_events"] = (start, end, int(hidden_states.shape[0]),
                                  int(out.shape[-1]))
        return out

    def timed_sample(hidden_states, input_batch, grammar_output):
        state["logits_events"] = None
        start, end = ev(), ev()
        start.record()
        out = orig_sample(hidden_states, input_batch, grammar_output)
        end.record()
        state["records"].append({
            "forward": state["pending_forward"],
            "sample": (start, end),
            "logits": state["logits_events"],
            "num_reqs": int(input_batch.num_reqs),
            "num_tokens": int(input_batch.num_tokens),
            "num_draft_tokens": int(input_batch.num_draft_tokens),
            "has_prefill": bool(input_batch.has_prefill),
        })
        state["pending_forward"] = None
        return out

    runner.execute_model = timed_execute
    runner.sample = timed_sample
    model.compute_logits = timed_logits
    return {"vocab_size": int(runner.model_config.get_vocab_size()),
            "max_num_reqs": runner.max_num_reqs}


def drain_timers(self) -> list[dict]:
    """Runs on the worker: resolve events into milliseconds and clear."""
    import torch

    torch.cuda.synchronize()
    state = self.model_runner._logits_probe
    out = []
    for r in state["records"]:
        if r["logits"] is None:
            continue
        ls, le, rows, vocab = r["logits"]
        out.append({
            "num_reqs": r["num_reqs"],
            "num_tokens": r["num_tokens"],
            "num_draft_tokens": r["num_draft_tokens"],
            "has_prefill": r["has_prefill"],
            "logits_rows": rows,
            "vocab": vocab,
            "logits_ms": ls.elapsed_time(le),
            "sample_ms": r["sample"][0].elapsed_time(r["sample"][1]),
            "forward_ms": (r["forward"][0].elapsed_time(r["forward"][1])
                           if r["forward"] else None),
        })
    state["records"].clear()
    return out


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least squares y = a + b x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def summarize(steps: list[dict]) -> dict:
    dec = [s for s in steps if not s["has_prefill"] and s["forward_ms"]]
    if len(dec) < 10:
        raise RuntimeError(f"Only {len(dec)} decode steps recorded")
    rows = [s["logits_rows"] for s in dec]
    a_l, b_l = fit_line(rows, [s["logits_ms"] for s in dec])
    a_s, b_s = fit_line(rows, [s["sample_ms"] for s in dec])
    by_reqs: dict[int, list[dict]] = {}
    for s in dec:
        by_reqs.setdefault(s["num_reqs"], []).append(s)
    groups = []
    for r, ss in sorted(by_reqs.items()):
        if len(ss) < 5:
            continue
        fwd = statistics.median(s["forward_ms"] for s in ss)
        smp = statistics.median(s["sample_ms"] for s in ss)
        groups.append({
            "num_reqs": r, "steps": len(ss),
            "median_logits_rows": statistics.median(s["logits_rows"] for s in ss),
            "median_forward_ms": fwd,
            "median_logits_ms": statistics.median(s["logits_ms"] for s in ss),
            "median_sample_ms": smp,
            "sample_share_of_step_pct": 100 * smp / (fwd + smp),
        })
    return {
        "decode_steps": len(dec),
        "logits_fit": {"fixed_ms": a_l, "per_row_us": 1000 * b_l},
        "sample_fit": {"fixed_ms": a_s, "per_row_us": 1000 * b_s},
        "by_num_reqs": groups,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="openbmb/MiniCPM5-2B")
    p.add_argument("--draft", default="openbmb/MiniCPM5-2B-DSpark")
    p.add_argument("--num-speculative-tokens", type=int, default=7)
    p.add_argument("--concurrency", nargs="+", type=int, default=[8, 32, 64, 128])
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--rounds", type=int, default=3,
                   help="generate() calls per concurrency; the first is warmup")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/logits_probe.json")
    args = p.parse_args()
    if args.rounds < 2:
        p.error("--rounds must be >= 2 (round 1 is discarded as warmup)")
    if args.output_tokens + 256 > args.max_model_len:
        p.error("output tokens must fit max-model-len")

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    import random

    import vllm
    from vllm import LLM, SamplingParams

    max_c = max(args.concurrency)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=max_c,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        enable_prefix_caching=False,
        tensor_parallel_size=1,
        seed=args.seed,
        speculative_config={
            "method": "dspark",
            "model": args.draft,
            "num_speculative_tokens": args.num_speculative_tokens,
            "enable_adaptive_verification": True,
        },
    )
    info = llm.collective_rpc(install_timers)[0]
    print(f"vLLM {vllm.__version__} | {args.model} | vocab {info['vocab_size']}",
          flush=True)

    rng = random.Random(args.seed)
    params = SamplingParams(temperature=args.temperature,
                            max_tokens=args.output_tokens, ignore_eos=True,
                            seed=args.seed)
    results = {}
    for c in sorted(args.concurrency):
        steps = []
        for rnd in range(args.rounds):
            # Natural-language prompts, so drafts are accepted at realistic
            # rates; a fresh shuffle each round (prefix caching is off).
            prompts = [f"{rng.choice(TASKS)} {rng.choice(TOPICS)}."
                       for _ in range(c)]
            llm.generate(prompts, params, use_tqdm=False)
            drained = llm.collective_rpc(drain_timers)[0]
            if rnd > 0:
                steps += drained
        summary = summarize(steps)
        results[c] = {"summary": summary, "steps": steps}
        lf, sf = summary["logits_fit"], summary["sample_fit"]
        print(f"\nconcurrency {c}: {summary['decode_steps']} decode steps | "
              f"logits {lf['per_row_us']:.2f} us/row + {lf['fixed_ms']:.3f} ms | "
              f"logits+sampler {sf['per_row_us']:.2f} us/row + "
              f"{sf['fixed_ms']:.3f} ms", flush=True)
        print(f"  {'reqs':>5} {'steps':>6} {'rows':>6} {'forward':>9} "
              f"{'logits':>8} {'sample':>8} {'share':>7}")
        for g in summary["by_num_reqs"]:
            print(f"  {g['num_reqs']:>5} {g['steps']:>6} "
                  f"{g['median_logits_rows']:>6.0f} {g['median_forward_ms']:>8.3f}ms "
                  f"{g['median_logits_ms']:>7.3f}ms {g['median_sample_ms']:>7.3f}ms "
                  f"{g['sample_share_of_step_pct']:>6.1f}%")

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "logits + rejection sampler cost outside forward_ms",
        "vllm_version": vllm.__version__,
        "args": vars(args),
        "worker": info,
        "results": results,
        "notes": [
            "forward_ms here is GPU time of execute_model (input prep + "
            "forward), not the startup profile's forward_ms.",
            "sample_ms includes compute_logits; 'share' is "
            "sample / (forward + sample) and ignores the drafter.",
            "Short natural-language prompts with ignore_eos; budgets depend on "
            "acceptance, which differs by workload.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
