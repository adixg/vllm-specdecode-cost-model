#!/usr/bin/env python3
"""Does a corrected cost model actually make serving faster?

Everything measured so far concerns prediction ACCURACY. Cost accuracy has
exactly one consumer in vLLM - the argmax in get_num_tokens (the table is read
only at adaptive_verification.py:305, and its result is used only at
model_runner.py:1084) - so a better estimate is worth nothing unless it changes
which budget gets chosen, and that change is worth something.

E8 showed the corrected model predicts within 0.8% of measured cost. That is
close enough to stand in for perfect knowledge, so this A/B doubles as the
oracle bound: the gap between the two modes is roughly the entire headroom
available to any cost-model improvement.

  stock      vLLM's table as shipped, profiled assuming 8192 context per request
  corrected  the same table plus k * (sum(context) - num_reqs * 8192)

Modes ALTERNATE round by round inside one engine. Running all of A then all of
B would charge warmup and thermal drift to whichever went first - the mistake
that produced the wrong answer in E2 before interleaving fixed it.

    export VLLM_USE_V2_MODEL_RUNNER=1
    python bench/serving_ab.py --model openbmb/MiniCPM5-2B \
                               --draft openbmb/MiniCPM5-2B-DSpark
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


def _install_correction(self: "Worker", k_ms_per_kv_token: float,
                        profile_ctx: int) -> dict:
    """Wrap the budget decision so the correction can be toggled per round."""
    import numpy as np

    runner = self.model_runner
    av = getattr(runner, "adaptive_verification", None)
    if av is None:
        return {"ok": False, "why": "adaptive verification is not active"}

    self._ab_enabled = False          # flipped between rounds from the host
    self._ab_offsets = []             # for reporting how large the change was
    original = av.get_num_tokens

    def corrected_get_num_tokens(num_tokens_per_req, draft_tokens):
        if not self._ab_enabled:
            return original(num_tokens_per_req, draft_tokens)

        req_ids = list(num_tokens_per_req)
        slots = [av.req_states.req_id_to_index[r] for r in req_ids]
        real_ctx = int(av.req_states.num_computed_tokens_np[slots].sum())

        # The table was profiled assuming profile_ctx per request. Remove that
        # assumption, add the batch's real aggregate context. The result is a
        # constant across candidate budgets, which is exactly why it shifts the
        # argmax: a ratio's argmax is invariant to scale but not to offset.
        offset = k_ms_per_kv_token * (real_ctx - len(req_ids) * profile_ctx)
        self._ab_offsets.append(offset)

        draft_t, verify_t = av.cost_tables
        # Clamp: a large negative offset must not drive a cost non-positive.
        av.cost_tables = (draft_t, np.maximum(verify_t + offset, 1e-6))
        try:
            return original(num_tokens_per_req, draft_tokens)
        finally:
            av.cost_tables = (draft_t, verify_t)

    av.get_num_tokens = corrected_get_num_tokens
    return {"ok": True}


def _set_mode(self: "Worker", enabled: bool) -> dict:
    self._ab_enabled = bool(enabled)
    n = len(getattr(self, "_ab_offsets", []))
    self._ab_offsets = []
    return {"enabled": self._ab_enabled, "offsets_last_round": n}


def _get_offsets(self: "Worker") -> list:
    return list(getattr(self, "_ab_offsets", []))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="openbmb/MiniCPM5-2B")
    p.add_argument("--draft", default="openbmb/MiniCPM5-2B-DSpark")
    p.add_argument("--num-speculative-tokens", type=int, default=3)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--attention-backend", default=None)
    p.add_argument("--k", type=float, default=0.0356e-3,
                   help="ms per KV token. H100 + Qwen3-1.7B measured 0.0356e-3; "
                        "for another model scale by its KV bytes per token.")
    p.add_argument("--rounds", type=int, default=6,
                   help="Rounds per mode. Modes alternate; medians are compared.")
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--prompt-tokens", type=int, nargs="+", default=[64, 512, 2048])
    p.add_argument("--out", default="results/serving_ab.json")
    args = p.parse_args()

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    import vllm.envs as envs
    from vllm import LLM, SamplingParams

    if not envs.VLLM_USE_V2_MODEL_RUNNER:
        print("error: set VLLM_USE_V2_MODEL_RUNNER=1", file=sys.stderr)
        return 2
    profile_ctx = int(envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN)

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
        # Prefix caching would let later rounds reuse earlier rounds' KV and
        # make whichever mode ran second look faster.
        enable_prefix_caching=False,
        **extra,
    )

    setup = llm.collective_rpc(_install_correction, args=(args.k, profile_ctx))[0]
    if not setup.get("ok"):
        print(f"error: {setup.get('why')}", file=sys.stderr)
        return 1
    print(f"correction installed; k={args.k*1000:.4f} us/KV-token, "
          f"profile context={profile_ctx}")

    tok = llm.get_tokenizer()
    filler = " systems research and inference serving"
    prompts = []
    for i in range(args.num_prompts):
        want = args.prompt_tokens[i % len(args.prompt_tokens)]
        ids = tok("Explain in detail:" + filler * (want // 6 + 1)).input_ids[:want]
        prompts.append(tok.decode(ids))
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    def one_round(enabled: bool) -> tuple[float, int]:
        llm.collective_rpc(_set_mode, args=(enabled,))
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        return dt, sum(len(o.outputs[0].token_ids) for o in outs)

    print(f"\nwarmup: {args.warmup_rounds} round(s) per mode")
    for _ in range(args.warmup_rounds):
        one_round(False); one_round(True)

    print(f"measuring {args.rounds} alternating rounds per mode\n")
    print(f"{'round':>6} {'mode':>10} {'seconds':>9} {'out toks':>9} {'tok/s':>9}")
    print("-" * 48)
    res = {"stock": [], "corrected": []}
    for r in range(args.rounds):
        for enabled, name in ((False, "stock"), (True, "corrected")):
            dt, toks = one_round(enabled)
            res[name].append(toks / dt)
            print(f"{r+1:>6} {name:>10} {dt:>9.3f} {toks:>9} {toks/dt:>9.1f}")

    offsets = llm.collective_rpc(_get_offsets)[0]
    s_med = statistics.median(res["stock"])
    c_med = statistics.median(res["corrected"])
    spread = lambda v: (max(v) - min(v)) / statistics.median(v)

    print(f"\n{'':>10} {'median tok/s':>13} {'round spread':>13}")
    print(f"{'stock':>10} {s_med:>13.1f} {spread(res['stock']):>12.1%}")
    print(f"{'corrected':>10} {c_med:>13.1f} {spread(res['corrected']):>12.1%}")

    # PAIRED comparison. The two modes alternate inside each round, so they
    # share that round's drift; subtracting within a round removes it. An
    # unpaired test against the raw spread is far too conservative and will
    # hide a real effect smaller than the drift.
    pairs = [(c - s) / s for s, c in zip(res["stock"], res["corrected"])]
    mean = statistics.mean(pairs)
    print(f"\npaired per-round delta: "
          f"{', '.join(f'{p:+.2%}' for p in pairs)}")
    print(f"mean {mean:+.2%}   median {statistics.median(pairs):+.2%}   "
          f"corrected faster in {sum(p > 0 for p in pairs)}/{len(pairs)} rounds")
    delta = mean

    if len(pairs) > 2:
        se = statistics.stdev(pairs) / len(pairs) ** 0.5
        n_se = mean / se if se else float("inf")
        print(f"standard error {se:.2%}; mean is {n_se:.1f} SE from zero")
        if abs(n_se) < 2:
            print("NOT DISTINGUISHABLE FROM ZERO: this bounds the benefit rather\n"
                  "than measuring it.")
        else:
            print("Effect is distinguishable from zero.")
    noise = max(spread(res["stock"]), spread(res["corrected"]))

    if offsets:
        print(f"\ncorrection applied to {len(offsets)} decisions in the last round, "
              f"median {statistics.median(offsets):+.3f} ms")

    out = {"generated_utc": datetime.now(timezone.utc).isoformat(),
           "args": vars(args), "profile_context_len": profile_ctx,
           "stock_tok_s": res["stock"], "corrected_tok_s": res["corrected"],
           "delta": delta, "noise": noise}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
