"""Full comparison matrix for the Phase 2 equivalence question.

Earlier runs gave apparently contradictory answers: each config reproduced
itself, yet the baseline-vs-spec divergence position moved between
invocations. That comparison was contaminated -- the diagnose script requested
logprobs on the baseline run but not on the spec run, so it was not comparing
like with like.

This script runs every config inside ONE invocation with byte-identical
arguments, saves the raw token ids for all of them, and reports every pairwise
comparison. No inference from divergence indices reported by other scripts.

Usage:  source env.sh && python bench/phase2_investigate.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def gen(a, draft=None, k=None, logprobs=None):
    cmd = [sys.executable, str(REPO / "bench" / "gen_tokens.py"),
           "--target", a.target, "--gmu", str(a.gmu),
           "--max-model-len", str(a.max_model_len),
           "--max-tokens", str(a.max_tokens), "--kv-bytes", str(a.kv_bytes)]
    if draft:
        cmd += ["--draft", draft, "--k", str(k)]
    if logprobs:
        cmd += ["--logprobs", str(logprobs)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400, cwd=REPO)
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith("TOKENS "):
            return json.loads(line[len("TOKENS "):])
    raise RuntimeError(f"no TOKENS (exit {r.returncode}):\n{(r.stdout+r.stderr)[-1500:]}")


def first_div(x, y):
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gmu", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--kv-bytes", type=int, default=536519168)
    a = ap.parse_args()

    runs = {}
    plan = [("base_a", None, None, None),
            ("base_b", None, None, None),
            ("base_lp", None, None, 5),      # same, but with logprobs requested
            ("spec_a", a.draft, a.k, None),
            ("spec_b", a.draft, a.k, None)]
    for name, d, k, lp in plan:
        print(f"[investigate] {name} ...", flush=True)
        runs[name] = gen(a, d, k, lp)

    pairs = [("base_a", "base_b"), ("base_a", "base_lp"),
             ("spec_a", "spec_b"), ("base_a", "spec_a"), ("base_a", "spec_b")]

    out = {"config": vars(a), "tokens": runs, "comparisons": {}}
    print()
    for x, y in pairs:
        divs = [first_div(runs[x][p], runs[y][p]) for p in range(len(runs[x]))]
        out["comparisons"][f"{x}_vs_{y}"] = divs
        n = sum(1 for d in divs if d is not None)
        print(f"{x:8s} vs {y:8s}: {len(divs)-n}/{len(divs)} identical   divergences={divs}")

    p = REPO / "results" / "phase2_investigate.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
