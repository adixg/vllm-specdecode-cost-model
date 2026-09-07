"""Phase 1: how much KV cache does each config actually get, and what is the
largest KV budget every config can share?

Why this exists
---------------
Speculative decoding does not just add draft-model weights; verifying k+1
tokens per sequence in one target pass raises peak activation substantially.
At a fixed gpu_memory_utilization, vLLM sizes the KV cache from whatever is
left over -- so the spec-decode run silently gets a much smaller cache than the
baseline. Benchmarking that way makes speculation look like it collapses at
high batch size when the real cause is cache starvation.

This script measures the breakdown for the baseline and for each k, then
reports the largest KV budget that ALL configs can be pinned to. Later phases
pass that single number as kv_cache_memory_bytes to every run.

Usage:  source env.sh && python bench/phase1_sizing.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from common import RESULTS, run_probe, save

GIB = 1024 ** 3


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        out = "unknown"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--gmu", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    a = ap.parse_args()

    configs = [("baseline", None, None)]
    configs += [(f"spec-k{k}", a.draft, k) for k in a.ks]

    rows = []
    for label, draft, k in configs:
        print(f"[phase1] probing {label} ...", flush=True)
        p = run_probe(label, a.target, draft=draft, k=k, gmu=a.gmu,
                      max_model_len=a.max_model_len)
        rows.append(p)
        if p.ok:
            print(f"         kv={p.kv_tokens:,} tok ({p.gib_kv} GiB)  "
                  f"weights={p.gib_weights}  act={p.gib_activation}  "
                  f"graphs={p.gib_cudagraph}", flush=True)
        else:
            print(f"         FAILED: {p.error}", flush=True)

    good = [r for r in rows if r.ok and r.suggest_full_bytes]
    shared = min((r.suggest_full_bytes for r in good), default=None)

    meta = {"gpu": gpu_info(), "target": a.target, "draft": a.draft,
            "gmu": a.gmu, "max_model_len": a.max_model_len,
            "shared_kv_bytes": shared}
    path = save("phase1_sizing", rows, meta)

    print("\n| config | KV tokens | KV GiB | weights GiB | peak act GiB | graphs GiB |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        if r.ok:
            print(f"| {r.label} | {r.kv_tokens:,} | {r.gib_kv} | "
                  f"{r.gib_weights} | {r.gib_activation} | {r.gib_cudagraph} |")
        else:
            print(f"| {r.label} | FAILED | | | | |")

    if shared:
        print(f"\nShared KV budget (min over configs): {shared} bytes "
              f"({shared / GIB:.2f} GiB)")
        print("Pass this as kv_cache_memory_bytes to EVERY run in later phases.")
    else:
        print("\nNo shared budget: no config reported a usable suggestion.")
    print(f"Wrote {path}")
    return 0 if all(r.ok for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
