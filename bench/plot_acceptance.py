#!/usr/bin/env python3
"""Plot how many draft tokens actually get accepted, and whether that changes.

Reads a serving_ab.py result file that has per-round acceptance counts and
writes two panels:

  left   how often exactly n of the block's tokens were accepted
  right  mean accepted tokens per draft, round by round, to show drift

Background on the data. vLLM reports num_accepted_tokens_per_pos as a
SURVIVAL curve: entry i counts the drafts where at least i+1 tokens were
accepted. The harness already converts that into "exactly n accepted"; this
script just draws it.

Usage:
    python bench/plot_acceptance.py results/ceiling_ab_ctx64.json
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")           # no display on a compute node
import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    data = json.load(open(path))
    workload = data["workloads"][0]
    rounds = workload["acceptance_by_round"]
    arm_b = data.get("arm_b", "corrected")
    ctx = workload["prompt_tokens"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    for arm, label, colour in (("stock", "stock", "#3b6ea5"),
                               ("corrected", arm_b, "#c2643b")):
        series = [r for r in rounds.get(arm, []) if r.get("exactly_n_accepted")]
        if not series:
            continue

        # Left panel: pool every round into one histogram, as a percentage of
        # drafts, so the two arms are comparable even if they issued different
        # numbers of drafts.
        width = len(series[0]["exactly_n_accepted"])
        totals = [0] * width
        for r in series:
            for i, count in enumerate(r["exactly_n_accepted"]):
                totals[i] += count
        drafts = sum(totals)
        shares = [100 * t / drafts for t in totals]
        offset = -0.2 if arm == "stock" else 0.2
        left.bar([n + offset for n in range(width)], shares,
                 width=0.4, label=label, color=colour)

        # Right panel: one point per round, in the order they ran.
        means = [r["mean_accepted_per_draft"] for r in series]
        right.plot(range(1, len(means) + 1), means, marker="o",
                   markersize=3.5, label=label, color=colour)

    left.set_xlabel("draft tokens accepted in one block")
    left.set_ylabel("% of drafts")
    left.set_title(f"acceptance distribution (ctx={ctx})")
    left.legend(frameon=False)
    left.spines[["top", "right"]].set_visible(False)

    right.set_xlabel("measured round")
    right.set_ylabel("mean accepted per draft")
    right.set_title("does acceptance drift as generation proceeds?")
    right.legend(frameon=False)
    right.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = path.replace(".json", "_acceptance.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
