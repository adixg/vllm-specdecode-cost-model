"""Phase 2: is speculative decoding equivalent to plain decoding?

At temperature 0 the verification rule guarantees the accepted sequence is what
the target model would have produced on its own -- IN EXACT ARITHMETIC. In
floating point that does not imply token-identical output, and testing for token
identity is the wrong gate.

Why: the target scores k+1 candidate positions in ONE batched forward pass,
whereas plain decoding scores one position per pass. Different batch shapes mean
different kernels and different reduction orders, so logits differ in the last
bits. Where the top-2 candidates are exactly tied, or one bf16 quantisation step
apart, that noise flips the argmax -- and once one token differs, so does
everything after it.

Measured on this machine, every observed divergence sat at a gap of exactly
0.0000 or 0.1250 nats, and 0.125 = 2^-3 is one bf16 step near a logprob of -1.2.
Divergence count also rises with k (1/6 prompts at k=1 and k=3, 5/6 at k=5),
exactly as expected: more verified positions per pass means more chances to land
on a tie.

So the gate is: EVERY divergence must occur at a numerical tie. A divergence at
a position where the model was confident is a real bug and blocks the sweep.

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

sys.path.insert(0, str(REPO / "bench"))
from gpu_guard import ensure_free  # noqa: E402

PROMPTS = [
    "Explain why the sky is blue in three sentences.",
    "Write a Python function that reverses a linked list.",
    "List the first 10 prime numbers, comma separated.",
    "Summarize the causes of World War I in one paragraph.",
    "What is the difference between a process and a thread?",
    "Describe how a hash table handles collisions.",
]


def run(target, draft, k, gmu, max_model_len, max_tokens, kv_bytes,
        logprobs=None):
    """Generate token ids for PROMPTS under one config, in a subprocess.

    Returns (tokens, logprobs). Requesting logprobs was verified not to change
    the generated tokens (see results/phase2_investigate.json, base_a vs
    base_lp), so the baseline can be scored and compared in one run.
    """
    cmd = [sys.executable, str(REPO / "bench" / "gen_tokens.py"),
           "--target", target, "--gmu", str(gmu),
           "--max-model-len", str(max_model_len),
           "--max-tokens", str(max_tokens)]
    if draft:
        cmd += ["--draft", draft, "--k", str(k)]
    if kv_bytes:
        cmd += ["--kv-bytes", str(kv_bytes)]
    if logprobs:
        cmd += ["--logprobs", str(logprobs)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400, cwd=REPO)
    tokens = lp = None
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith("TOKENS "):
            tokens = json.loads(line[len("TOKENS "):])
        if line.startswith("LOGPROBS "):
            lp = json.loads(line[len("LOGPROBS "):])
    if tokens is None:
        raise RuntimeError(f"no TOKENS line (exit {r.returncode}):\n"
                           f"{(r.stdout + r.stderr)[-2000:]}")
    return tokens, lp


def tie_gap(lp, prompt_idx, pos):
    """Top-2 logprob gap at a generated position, or None if unavailable."""
    steps = (lp or {}).get(str(prompt_idx))
    if not steps or pos is None or pos >= len(steps):
        return None
    v = steps[pos]
    return round(v[0] - v[1], 6) if len(v) >= 2 else None


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
    ap.add_argument("--tie-threshold", type=float, default=0.15,
                    help="top-2 logprob gap (nats) at or below which a "
                         "divergence counts as a numerical tie. Default 0.15 "
                         "covers one bf16 step (0.125) at these magnitudes.")
    a = ap.parse_args()

    gpu_state = ensure_free()

    common = dict(gmu=a.gmu, max_model_len=a.max_model_len,
                  max_tokens=a.max_tokens, kv_bytes=a.kv_bytes)

    print("[phase2] baseline (with logprobs) ...", flush=True)
    base, base_lp = run(a.target, None, None, logprobs=5, **common)

    real_bugs = 0
    report = {"prompts": PROMPTS, "config": vars(a), "gpu_state": gpu_state,
              "tokens": {"baseline": base}, "results": {}}
    for k in a.ks:
        print(f"[phase2] spec k={k} ...", flush=True)
        spec, _ = run(a.target, a.draft, k, **common)
        rows = []
        for i, (bt, st) in enumerate(zip(base, spec)):
            d = first_divergence(bt, st)
            gap = tie_gap(base_lp, i, d)
            is_tie = d is not None and gap is not None and gap <= a.tie_threshold
            if d is not None and not is_tie:
                real_bugs += 1
            rows.append({"prompt": i, "match": d is None,
                         "divergence_at": d, "top2_gap": gap,
                         "numerical_tie": is_tie,
                         "len_base": len(bt), "len_spec": len(st)})
        report["results"][f"k{k}"] = rows
        report["tokens"][f"k{k}"] = spec
        ident = [r for r in rows if r["match"]]
        ties = [r for r in rows if r["numerical_tie"]]
        bad = [r for r in rows if not r["match"] and not r["numerical_tie"]]
        print(f"         {len(ident)}/{len(rows)} token-identical, "
              f"{len(ties)} tie-break divergence(s)"
              + (f", {len(bad)} REAL: "
                 f"{[(r['prompt'], r['divergence_at'], r['top2_gap']) for r in bad]}"
                 if bad else ""), flush=True)
        for r in ties:
            print(f"           prompt {r['prompt']} pos {r['divergence_at']}: "
                  f"top-2 gap {r['top2_gap']} nats (tie)", flush=True)

    out = REPO / "results" / "phase2_correctness.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")

    if real_bugs:
        print(f"GATE FAILED: {real_bugs} divergence(s) at positions where the "
              f"model was confident (top-2 gap > {a.tie_threshold}). "
              f"Do not proceed to the benchmark sweep.")
        return 1
    print(f"GATE PASSED: every divergence sits at a numerical tie "
          f"(top-2 gap <= {a.tie_threshold} nats). Speculative decoding is "
          f"equivalent up to floating-point tie-breaking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
