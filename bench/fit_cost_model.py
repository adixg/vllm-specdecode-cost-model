#!/usr/bin/env python3
"""E4: which functional form should the corrected cost model use?

E6 showed that KV-read time is visible only when it exceeds whatever else
bottlenecks the step. Two candidate forms follow from that:

    additive:  cost = floor + k * S          (KV time always adds)
    max-like:  cost = max(floor, k * S)      (KV time hides under the floor)

where S = sum(seq_lens) = num_reqs * context_len is the KV tokens the step must
read, `floor` is the measured cost of the same batch shape at zero context, and
k is the time to read one KV token.

The decisive test is to fit ONE k across BOTH regimes at once. Physics says k
is a property of the model and the GPU, not of the execution mode, so the same
value must explain the graphed and the eager data. Additive cannot: in eager
mode the measurements are flat in context, so any k > 0 over-predicts there.

Needs no GPU - it reads results already on disk.

    python bench/fit_cost_model.py
"""

from __future__ import annotations

import argparse
import json
import math


def load(path: str, label: str) -> list[dict]:
    """Flatten one sweep into rows, attaching each row's zero-context floor."""
    d = json.load(open(path))
    limit = d.get("cudagraph_limit") or 0
    rows = [r for r in d["rows"] if r["median_ms"] is not None]

    # floor = what this batch shape costs with no context at all.
    floors = {r["num_tokens"]: r["median_ms"] for r in rows if r["context_len"] == 0}

    out = []
    for r in rows:
        if r["num_tokens"] not in floors:
            continue
        out.append({
            "src": label,
            "num_tokens": r["num_tokens"],
            "num_reqs": r["num_reqs"],
            "context_len": r["context_len"],
            # S: KV tokens this step must stream.
            "S": r["num_reqs"] * r["context_len"],
            "floor_ms": floors[r["num_tokens"]],
            "actual_ms": r["median_ms"],
            "stock_ms": r["predicted_ms"],          # what vLLM would predict
            # A step replays a cudagraph when its token count is within the
            # captured range; above the limit it runs eager.
            "graphed": r["num_tokens"] <= limit,
        })
    return out


def predict(model: str, row: dict, k: float) -> float:
    kv = k * row["S"]
    if model == "additive":
        return row["floor_ms"] + kv
    if model == "max":
        return max(row["floor_ms"], kv)
    if model == "stock":
        return row["stock_ms"]
    raise ValueError(model)


def rmse(model: str, rows: list[dict], k: float) -> float:
    if not rows:
        return float("nan")
    return math.sqrt(
        sum((predict(model, r, k) - r["actual_ms"]) ** 2 for r in rows) / len(rows)
    )


def fit_k(model: str, rows: list[dict]) -> float:
    """One free parameter, so a plain scan is clearer than an optimiser."""
    best_k, best = 0.0, float("inf")
    # 0 to 0.2 us per KV token, in ms. The bandwidth-derived value is 3.56e-5.
    for i in range(1, 4001):
        k = i * 1e-7
        e = rmse(model, rows, k)
        if e < best:
            best_k, best = k, e
    return best_k


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eager", default="results/h100_e1.json")
    p.add_argument("--graphed", default="results/h100_e5.json")
    p.add_argument("--holdout-contexts", type=int, nargs="+", default=[1024, 4096],
                   help="Contexts kept out of fitting, to test generalisation.")
    args = p.parse_args()

    rows = load(args.eager, "e1") + load(args.graphed, "e5")
    train = [r for r in rows if r["context_len"] not in args.holdout_contexts]
    test = [r for r in rows if r["context_len"] in args.holdout_contexts]
    print(f"{len(rows)} cells: {len(train)} train, {len(test)} held out "
          f"(contexts {args.holdout_contexts})")
    print(f"  graphed {sum(r['graphed'] for r in rows)}, "
          f"eager {sum(not r['graphed'] for r in rows)}\n")

    K_PHYS = 0.0356e-3      # ms per KV token, from KV bytes / measured bandwidth

    print(f"{'model':>9} {'fitted k':>12} {'train RMSE':>11} {'test RMSE':>10} "
          f"{'graphed':>9} {'eager':>8}")
    print("-" * 64)
    results = {}
    for model in ("stock", "additive", "max"):
        k = 0.0 if model == "stock" else fit_k(model, train)
        results[model] = (k, rmse(model, test, k))
        g = rmse(model, [r for r in test if r["graphed"]], k)
        e = rmse(model, [r for r in test if not r["graphed"]], k)
        kstr = "-" if model == "stock" else f"{k*1000:.4f} us"
        print(f"{model:>9} {kstr:>12} {rmse(model, train, k):11.3f} "
              f"{rmse(model, test, k):10.3f} {g:9.3f} {e:8.3f}")

    print(f"\nk from bandwidth (independent): {K_PHYS*1000:.4f} us/KV-token")
    for m in ("additive", "max"):
        k = results[m][0]
        print(f"  {m:>8} fitted k is {k/K_PHYS:5.2f}x that value")

    best = min(("additive", "max"), key=lambda m: results[m][1])
    print(f"\nbest out-of-sample: {best.upper()}")
    a, mx = results["additive"][1], results["max"][1]
    if min(a, mx) > 0:
        print(f"  additive {a:.3f} ms vs max {mx:.3f} ms RMSE on held-out contexts")
    print(f"  stock (vLLM today): {results['stock'][1]:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
