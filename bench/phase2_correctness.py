"""Phase 2: prove speculative decoding is output-equivalent to plain decoding.

At temperature 0 the verification rule guarantees the accepted sequence is
exactly what the target model would have produced on its own. So baseline and
spec-decode must return token-identical output for the same prompts.

This is the gate for the whole project. If it fails, the throughput numbers in
later phases are measuring something other than what we claim, and no amount of
careful benchmarking fixes that. A divergence in the final token or two can be a
stop-condition artifact; a divergence early in the sequence is a real bug.

Each engine runs in its own subprocess so the two configs cannot share allocator
or CUDA-graph state.

Usage:  source env.sh && python bench/phase2_correctness.py [--kv-bytes N]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PROMPTS = [
    "Explain why the sky is blue in three sentences.",
    "Write a Python function that reverses a linked list.",
    "List the first 10 prime numbers, comma separated.",
    "Summarize the causes of World War I in one paragraph.",
    "What is the difference between a process and a thread?",
    "Describe how a hash table handles collisions.",
]


def run(target, draft, k, gmu, max_model_len, max_tokens, kv_bytes):
    """Generate token ids for PROMPTS under one config, in a subprocess."""
    cmd = [sys.executable, str(REPO / "bench" / "gen_tokens.py"),
           "--target", target, "--gmu", str(gmu),
           "--max-model-len", str(max_model_len),
           "--max-tokens", str(max_tokens)]
    if draft:
        cmd += ["--draft", draft, "--k", str(k)]
    if kv_bytes:
        cmd += ["--kv-bytes", str(kv_bytes)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400, cwd=REPO)
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith("TOKENS "):
            return json.loads(line[len("TOKENS "):])
    raise RuntimeError(f"no TOKENS line (exit {r.returncode}):\n"
                       f"{(r.stdout + r.stderr)[-2000:]}")


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--gmu", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--kv-bytes", type=int, default=None)
    a = ap.parse_args()

    common = dict(gmu=a.gmu, max_model_len=a.max_model_len,
                  max_tokens=a.max_tokens, kv_bytes=a.kv_bytes)

    print("[phase2] baseline ...", flush=True)
    base = run(a.target, None, None, **common)

    failures = 0
    report = {"prompts": PROMPTS, "results": {}}
    for k in a.ks:
        print(f"[phase2] spec k={k} ...", flush=True)
        spec = run(a.target, a.draft, k, **common)
        rows = []
        for i, (bt, st) in enumerate(zip(base, spec)):
            d = first_divergence(bt, st)
            rows.append({"prompt": i, "match": d is None,
                         "divergence_at": d,
                         "len_base": len(bt), "len_spec": len(st)})
            if d is not None:
                failures += 1
        report["results"][f"k{k}"] = rows
        bad = [r for r in rows if not r["match"]]
        print(f"         {len(rows) - len(bad)}/{len(rows)} match"
              + (f"  DIVERGENCES: {[(r['prompt'], r['divergence_at']) for r in bad]}"
                 if bad else ""), flush=True)

    out = REPO / "results" / "phase2_correctness.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")

    if failures:
        print(f"GATE FAILED: {failures} prompt(s) diverged. "
              f"Do not proceed to the benchmark sweep.")
        return 1
    print("GATE PASSED: speculative decoding is output-equivalent at temperature 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
