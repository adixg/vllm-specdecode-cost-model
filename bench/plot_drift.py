#!/usr/bin/env python3
"""Plot predicted step time vs measured step time.

This reproduces the kind of drift plot in vllm-project/vllm#52057, using the
data bench/verify_cost_probe.py already collected. No GPU needed - it just
reads results/verify_cost_probe.json and draws a picture.

    python bench/plot_drift.py                  # colour by token count (as in #52057)
    python bench/plot_drift.py --by context     # colour by context length

Every dot is one measured batch shape.
  x = what vLLM's startup-profiled cost table PREDICTS the step will take
  y = what the step ACTUALLY took
  the dashed diagonal is perfect prediction (x == y)

Dots below the line mean vLLM overestimated the cost of that step.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # write a file instead of opening a window (needed on WSL)
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--style", choices=["scatter", "hexbin"], default="hexbin",
                    help="hexbin = 2D density with log10(count) colour, as in #52057")
parser.add_argument("--by", choices=["tokens", "context"], default="tokens",
                    help="scatter only: what the dot colour means")
parser.add_argument("--in", dest="in_file",
                    default="results/verify_cost_probe.json",
                    help="which probe results file to plot")
parser.add_argument("--out", default=None)
args = parser.parse_args()

IN_FILE = args.in_file

OUT_FILE = args.out or IN_FILE.replace(".json", f"_{args.style}.png")

# 1. Load the measurements -------------------------------------------------
with open(IN_FILE) as f:
    data = json.load(f)

# Keep only rows that actually produced a timing (some cells get skipped).
rows = [r for r in data["rows"] if r["median_ms"] is not None]

# One point PER REPLAY, not per cell: the density plot needs the raw samples.
# Each cell's predicted value is repeated once for each of its measurements.
predicted, measured = [], []
for r in rows:
    for sample in r["forward_ms"]:
        predicted.append(r["predicted_ms"])
        measured.append(sample)

# What the colour of each dot means. #52057's plot varies the token count, so
# that is the default; --by context shows the axis E1 actually swept.
if args.by == "tokens":
    colour_values = [r["num_tokens"] for r in rows]
    colour_label = "verification tokens in the step"
else:
    colour_values = [r["context_len"] for r in rows]
    colour_label = "context length (tokens)"

baseline_ctx = data["baseline_context_len"]

# 2. Draw -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.8, 6.2))

limit = max(max(predicted), max(measured)) * 1.05

if args.style == "hexbin":
    # One hexagon per (predicted, measured) region; colour = how many samples
    # landed there. bins="log" makes the colour scale log10(count), which is
    # what #52057 uses - without it a few dense cells wash out everything else.
    hb = ax.hexbin(predicted, measured, gridsize=45,
                   extent=(0, limit, 0, limit),
                   bins="log", cmap="viridis", mincnt=1, linewidths=0)
    bar = fig.colorbar(hb, ax=ax)
    bar.set_label("log$_{10}$(count)")
    ax.plot([0, limit], [0, limit], linestyle="--", linewidth=1.5,
            color="#c8ccd0", zorder=3, label="perfect prediction")
else:
    ax.plot([0, limit], [0, limit], linestyle="--", linewidth=1.5,
            color="#9aa0a6", zorder=1, label="perfect prediction")
    dots = ax.scatter(predicted, measured, c=colour_values, cmap="Blues",
                      s=70, edgecolor="#1a1a1a", linewidth=0.6,
                      vmin=0, vmax=max(colour_values), zorder=2)
    bar = fig.colorbar(dots, ax=ax)
    bar.set_label(colour_label)

# 3. Agreement statistics ---------------------------------------------------
# error = measured - predicted, so a negative bias means vLLM overestimates.
errors = [m - p for m, p in zip(measured, predicted)]
n = len(errors)
bias = sum(errors) / n                                   # mean signed error
mae = sum(abs(e) for e in errors) / n                    # mean absolute error
rmse = (sum(e * e for e in errors) / n) ** 0.5           # root mean squared error

mp, mm = sum(predicted) / n, sum(measured) / n
cov = sum((p - mp) * (m - mm) for p, m in zip(predicted, measured))
sp = sum((p - mp) ** 2 for p in predicted) ** 0.5
sm = sum((m - mm) ** 2 for m in measured) ** 0.5
r = cov / (sp * sm) if sp and sm else float("nan")

stats = (f"n = {n:,}\nbias = {bias:.3f} ms\nMAE = {mae:.3f} ms\n"
         f"RMSE = {rmse:.3f} ms\nr = {r:.3f}")
ax.text(0.97, 0.03, stats, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#c8ccd0", alpha=0.9))

ax.set_xlabel("predicted step time (ms)\nfrom vLLM's startup-profiled cost table")
ax.set_ylabel("measured step time (ms)")
ax.set_title(f"Verification cost drift\nprofiled at a fixed context length of {baseline_ctx}")

ax.set_xlim(0, limit)
ax.set_ylim(0, limit)
ax.set_aspect("equal")
ax.grid(True, linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False)

fig.tight_layout()
fig.savefig(OUT_FILE, dpi=150)
print(f"wrote {OUT_FILE}  ({len(rows)} points)")
