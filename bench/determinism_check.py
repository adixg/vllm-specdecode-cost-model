"""Is a given decoding config reproducible run-to-run at temperature 0?

Phase 2 compares baseline output against speculative output. That comparison is
only meaningful if each config is deterministic on its own. If speculative
decoding does not even reproduce itself, then "baseline != spec" says nothing
about speculative decoding being wrong -- it says the engine is numerically
non-deterministic, and the equivalence gate has to be restated in terms of
distributional agreement rather than token equality.

Runs the SAME config N times in separate processes and reports the first
divergence between every pair of runs.

Usage:
  source env.sh && python bench/determinism_check.py --runs 2            # baseline
  source env.sh && python bench/determinism_check.py --runs 2 --draft Qwen/Qwen2.5-0.5B-Instruct --k 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run_once(a):
    cmd = [sys.executable, str(REPO / "bench" / "gen_tokens.py"),
           "--target", a.target, "--gmu", str(a.gmu),
           "--max-model-len", str(a.max_model_len),
           "--max-tokens", str(a.max_tokens), "--kv-bytes", str(a.kv_bytes)]
    if a.draft:
        cmd += ["--draft", a.draft, "--k", str(a.k)]
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
    ap.add_argument("--draft", default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gmu", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--kv-bytes", type=int, default=536519168)
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args()

    label = f"spec-k{a.k}" if a.draft else "baseline"
    runs = []
    for i in range(a.runs):
        print(f"[determinism] {label} run {i+1}/{a.runs} ...", flush=True)
        runs.append(run_once(a))

    findings, unstable = [], 0
    for i, j in combinations(range(len(runs)), 2):
        for p in range(len(runs[i])):
            d = first_div(runs[i][p], runs[j][p])
            if d is not None:
                unstable += 1
            findings.append({"run_a": i, "run_b": j, "prompt": p, "divergence_at": d})

    for f in findings:
        if f["divergence_at"] is not None:
            print(f"  run{f['run_a']} vs run{f['run_b']}  prompt {f['prompt']}: "
                  f"diverges at {f['divergence_at']}")

    out = REPO / "results" / f"determinism_{label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"label": label, "findings": findings}, indent=2))
    print(f"\n{label}: {unstable}/{len(findings)} prompt-pairs diverged")
    print("DETERMINISTIC" if unstable == 0 else "NON-DETERMINISTIC run to run")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
