#!/usr/bin/env python3
"""Recreate the grouped panels from vLLM issue #52057.

Two input modes are supported:

1. Controlled probes use one file for the target-token panel and another for
   the request-count panel. This lets E1 isolate target-token count at eight
   requests while E5 isolates request count with one query token per request.
2. Real decode traces combine step_trace/saturation files and build both
   panels from the same observed serving steps. The current real trace is
   sparse, but it is the closest apples-to-apples comparison with #52057.

Each panel shows the median expected and measured target time in groups with at
least three samples, plus the measured interquartile range.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_probe(path: str) -> list[dict]:
    """Expand every controlled probe replay into one comparable sample."""
    with open(path) as input_file:
        data = json.load(input_file)

    samples = []
    for row in data["rows"]:
        predicted = row.get("predicted_ms")
        if predicted is None:
            continue
        for measured in row.get("forward_ms", []):
            samples.append({
                "num_target_tokens": int(row["num_tokens"]),
                "num_reqs": int(row["num_reqs"]),
                "expected_ms": float(predicted),
                "measured_ms": float(measured),
            })
    return samples


def load_step_traces(paths: list[str]) -> list[dict]:
    """Load real decode-only step timings from one or more trace files."""
    samples = []
    for path in paths:
        with open(path) as input_file:
            data = json.load(input_file)
        for row in data["rows"]:
            samples.append({
                "source": path,
                "num_target_tokens": int(row["num_target_tokens"]),
                "num_reqs": int(row["num_reqs"]),
                "expected_ms": float(row["predicted_ms"]),
                "measured_ms": float(row["step_ms"]),
            })
    return samples


def summarize(samples: list[dict], key: str,
              minimum_group_size: int) -> list[dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for sample in samples:
        groups[int(sample[key])].append(sample)

    summary = []
    for value, rows in sorted(groups.items()):
        if len(rows) < minimum_group_size:
            continue
        expected = [row["expected_ms"] for row in rows]
        measured = [row["measured_ms"] for row in rows]
        summary.append({
            key: value,
            "count": len(rows),
            "expected_median_ms": statistics.median(expected),
            "measured_median_ms": statistics.median(measured),
            "measured_q1_ms": float(np.percentile(measured, 25)),
            "measured_q3_ms": float(np.percentile(measured, 75)),
        })
    return summary


def plot_panel(ax, rows: list[dict], key: str, xlabel: str,
               logarithmic_x: bool) -> None:
    x = [row[key] for row in rows]
    expected = [row["expected_median_ms"] for row in rows]
    measured = [row["measured_median_ms"] for row in rows]
    q1 = [row["measured_q1_ms"] for row in rows]
    q3 = [row["measured_q3_ms"] for row in rows]

    ax.plot(x, expected, marker="o", markersize=4, linewidth=1.8,
            color="#1f77b4", label="Expected")
    ax.plot(x, measured, marker="o", markersize=4, linewidth=1.8,
            color="#d62728", label="Measured")
    ax.fill_between(x, q1, q3, color="#d62728", alpha=0.16,
                    label="Measured IQR")
    if logarithmic_x:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time (ms)")
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.legend(loc="best")


def print_summary(title: str, rows: list[dict], key: str) -> None:
    print(f"\n{title}")
    print(f"{'group':>8} {'n':>6} {'expected':>11} {'measured':>11} "
          f"{'meas-exp':>11}")
    for row in rows:
        difference = row["measured_median_ms"] - row["expected_median_ms"]
        print(f"{row[key]:>8} {row['count']:>6} "
              f"{row['expected_median_ms']:>11.3f} "
              f"{row['measured_median_ms']:>11.3f} {difference:>+11.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--real-step-files",
        nargs="+",
        help="real decode trace JSON files; both panels use their combined rows",
    )
    inputs.add_argument(
        "--controlled-token-probe",
        help="controlled probe JSON for the target-token panel",
    )
    parser.add_argument(
        "--controlled-request-probe",
        help="controlled probe JSON for the request-count panel",
    )
    parser.add_argument("--minimum-group-size", type=int, default=3)
    parser.add_argument("--title", default="Target time (decode-only iterations)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args()

    if args.minimum_group_size < 1:
        parser.error("--minimum-group-size must be positive")

    if args.real_step_files:
        token_samples = request_samples = load_step_traces(
            args.real_step_files
        )
        input_kind = "real_decode_steps"
        sources = args.real_step_files
    else:
        if not args.controlled_request_probe:
            parser.error(
                "--controlled-request-probe is required with "
                "--controlled-token-probe"
            )
        token_samples = load_probe(args.controlled_token_probe)
        request_samples = load_probe(args.controlled_request_probe)
        input_kind = "controlled_probes"
        sources = [
            args.controlled_token_probe,
            args.controlled_request_probe,
        ]

    by_tokens = summarize(
        token_samples, "num_target_tokens", args.minimum_group_size
    )
    by_requests = summarize(
        request_samples, "num_reqs", args.minimum_group_size
    )
    if not by_tokens or not by_requests:
        print("error: no groups met the minimum sample count")
        return 1

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    fig.suptitle(args.title, fontsize=15)
    plot_panel(
        axes[0],
        by_tokens,
        "num_target_tokens",
        "Target token count",
        logarithmic_x=True,
    )
    axes[0].set_title(
        f"Median by target token count (groups n≥{args.minimum_group_size})"
    )
    plot_panel(
        axes[1],
        by_requests,
        "num_reqs",
        "Request batch size",
        logarithmic_x=False,
    )
    axes[1].set_title(
        f"Median by request batch size (groups n≥{args.minimum_group_size})"
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"wrote {args.out}")

    print_summary("By target token count", by_tokens, "num_target_tokens")
    print_summary("By request batch size", by_requests, "num_reqs")

    if args.summary_out:
        result = {
            "kind": input_kind,
            "sources": sources,
            "minimum_group_size": args.minimum_group_size,
            "by_target_token_count": by_tokens,
            "by_request_batch_size": by_requests,
        }
        os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
        with open(args.summary_out, "w") as output_file:
            json.dump(result, output_file, indent=2)
        print(f"wrote {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
