#!/usr/bin/env python3
"""Does the cost mispricing actually change the budget vLLM chooses?

Prediction error is not the same thing as a worse decision. Budget selection is

    argmax_B  estimated_accepted(B) / cost(B)

and the argmax of a ratio is **invariant to any purely multiplicative error** -
scaling every candidate's cost by the same factor changes nothing. So "vLLM
overpredicts by 73%" does not by itself imply a wrong choice.

What the context mispricing actually produces is a constant OFFSET, identical
across candidate budgets:

    offset = k * (num_reqs * PROFILE_CTX - sum(seq_lens))

A positive offset (real contexts shorter than the profiled 8192) inflates the
fixed part of the cost, which makes marginal draft tokens look relatively
cheap - so vLLM over-commits verification.

This simulates the decision to bound how much that costs. It is a SIMULATION,
not a measurement: survival probabilities are drawn from a distribution matched
to E3's observed acceptance, and the marginal cost per verified token is
assumed. Treat it as an order-of-magnitude estimate that motivates the real
oracle bound, not as a result.

    python bench/decision_impact.py
"""

from __future__ import annotations

import argparse

import numpy as np

K_DEFAULT = 0.0356e-3      # ms per KV token, measured on H100 with Qwen3-1.7B
PROFILE_CTX = 8192         # VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN


def simulate(num_reqs, ctx_true, rng, k, spec=3, floor=2.0, marginal=0.02):
    """Return (optimal B, vLLM's B, offset ms, fraction of the rate lost)."""
    # Per-slot acceptance, decaying down each request. E3 measured 72/45/27%
    # acceptance at positions 1/2/3, which this roughly reproduces.
    conf = np.clip(rng.normal(0.72, 0.15, (num_reqs, spec)), 0.05, 0.99)
    survival = np.cumprod(conf, axis=1).ravel()      # product WITHIN a request
    scores = np.sort(survival)[::-1]                 # global ranking across all

    # Expected accepted tokens if the best B slots are verified. Summing
    # probabilities is valid even though the events are dependent.
    accepted = np.concatenate(([num_reqs], num_reqs + np.cumsum(scores)))
    B = np.arange(len(accepted))

    true_cost = floor + k * num_reqs * ctx_true   + marginal * B
    vllm_cost = floor + k * num_reqs * PROFILE_CTX + marginal * B

    b_true = int((accepted / true_cost).argmax())
    b_vllm = int((accepted / vllm_cost).argmax())

    # Both budgets are scored against the TRUE cost: what you actually get.
    rate_true = accepted[b_true] / true_cost[b_true]
    rate_vllm = accepted[b_vllm] / true_cost[b_vllm]
    offset = k * num_reqs * (PROFILE_CTX - ctx_true)
    return b_true, b_vllm, offset, (rate_true - rate_vllm) / rate_true


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reqs", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--contexts", type=int, nargs="+",
                   default=[256, 1024, 2048, 8192, 16384])
    p.add_argument("--k", type=float, default=K_DEFAULT)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"k = {args.k*1000:.4f} us/KV-token, profiling context = {PROFILE_CTX}\n")
    print(f"{'reqs':>5} {'ctx':>6} {'offset ms':>10} {'true B':>7} {'vLLM B':>7} "
          f"{'rate lost':>10}")
    print("-" * 50)
    worst = 0.0
    for reqs in args.reqs:
        for ctx in args.contexts:
            res = [simulate(reqs, ctx, rng, args.k) for _ in range(args.trials)]
            bt = np.median([r[0] for r in res])
            bv = np.median([r[1] for r in res])
            off = res[0][2]
            loss = float(np.mean([r[3] for r in res]))
            worst = max(worst, loss)
            print(f"{reqs:>5} {ctx:>6} {off:>+10.2f} {bt:>7.0f} {bv:>7.0f} "
                  f"{loss:>9.2%}")
        print()

    print(f"worst mean rate lost: {worst:.2%}")
    print("\nThe loss grows with num_reqs x (profiled ctx - real ctx), because that\n"
          "product is the size of the constant offset. It is negligible for small\n"
          "batches and material for large ones - which is the production regime.")
    print("\nSIMULATION, not measurement: survival is drawn from an assumed\n"
          "distribution and marginal cost is assumed. Motivates the real oracle\n"
          "bound; does not replace it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
