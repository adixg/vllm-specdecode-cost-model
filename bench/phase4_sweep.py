"""Phase 4: does speculative decoding stop helping as batch size grows?

The claim: at batch 1 decoding is memory-bandwidth bound -- the GPU idles while
weights stream from HBM -- so verifying k+1 candidates costs little more than
decoding 1, and speculation is nearly free. As batch grows, the same weight read
amortises over many sequences and the step becomes compute bound; now the extra
verification tokens, the draft model's own forward passes, and every rejected
token are real FLOPs competing with useful work. Speedup should decay, and may
cross below 1.0.

What makes this measurable rather than circular:

- Batch size is varied with **max_num_seqs**, not by changing the workload.
  Every configuration submits the identical set of requests; only concurrency
  differs. (Phase 3 found that varying the prompt count instead produces a
  spurious result.)
- Every run is pinned to the **same kv_cache_memory_bytes**, so speculation is
  not silently starved of KV cache (Phase 1).
- **ignore_eos + fixed max_tokens**, so every request emits exactly the same
  number of tokens and batch composition cannot drift between configs.
- Runs are **interleaved** rather than grouped by config. This is a laptop GPU
  and it thermally throttles; running all baselines then all spec runs would
  bake drift into the comparison as a fake trend.
- Acceptance is recorded at every point. Phase 3 established it is
  batch-independent, so if it stays flat here while speedup decays, the decay is
  a systems effect and not the draft model guessing worse under load.

Usage:
  source env.sh && python bench/phase4_sweep.py --kv-bytes <from phase 1>
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


def run_point(a, concurrency, method=None, draft=None, k=None):
    cmd = [sys.executable, str(REPO / "bench" / "spec_probe.py"),
           "--target", a.target, "--gmu", str(a.gmu),
           "--max-model-len", str(a.max_model_len),
           "--max-tokens", str(a.max_tokens),
           "--kv-bytes", str(a.kv_bytes),
           "--batch", str(a.requests),
           "--max-num-seqs", str(concurrency)]
    if k is not None:
        cmd += ["--k", str(k)]
    if method:
        cmd += ["--method", method]
    elif draft:
        cmd += ["--draft", draft]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=a.timeout, cwd=REPO)
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith("SPEC "):
            return json.loads(line[len("SPEC "):])
    raise RuntimeError(f"no SPEC line (exit {r.returncode}):\n"
                       f"{(r.stdout + r.stderr)[-1500:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--ngram-ks", type=int, nargs="*", default=[3],
                    help="k values for n-gram speculation; empty to skip")
    ap.add_argument("--gmu", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--kv-bytes", type=int, required=True,
                    help="shared KV budget from Phase 1; identical for every run")
    ap.add_argument("--requests", type=int, default=48,
                    help="requests submitted per point; fixed across all points")
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 48],
                    help="max_num_seqs values -- the batch axis")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=3000)
    a = ap.parse_args()

    gpu_state = ensure_free()

    # Build the config list once; interleave so thermal drift cannot masquerade
    # as a trend in any single config.
    configs = [("baseline", None, None, None)]
    configs += [(f"draft-k{k}", None, a.draft, k) for k in a.ks]
    configs += [(f"ngram-k{k}", "ngram", None, k) for k in (a.ngram_ks or [])]

    rows = []
    for rep in range(a.repeats):
        for b in a.batches:
            for label, method, draft, k in configs:
                print(f"[phase4] rep{rep} batch={b:>3} {label} ...", flush=True)
                try:
                    r = run_point(a, b, method=method, draft=draft, k=k)
                except Exception as e:
                    print(f"          FAILED: {str(e)[:200]}", flush=True)
                    rows.append({"label": label, "concurrency": b, "rep": rep,
                                 "error": str(e)[:500]})
                    continue
                r.update({"label": label, "concurrency": b, "rep": rep})
                rows.append(r)
                print(f"          {r['output_toks_per_s']:>8.2f} tok/s"
                      + (f"   accept={r.get('acceptance_rate', float('nan')):.3f}"
                         f"  len={r.get('mean_acceptance_length', float('nan')):.3f}"
                         if r.get("acceptance_rate") is not None else ""),
                      flush=True)

    out = REPO / "results" / "phase4_sweep.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"gpu_state": gpu_state, "config": vars(a),
                               "rows": rows}, indent=2))

    # ---- speedup table, baseline-relative at each batch size ----
    def tps(label, b):
        v = [r["output_toks_per_s"] for r in rows
             if r.get("label") == label and r.get("concurrency") == b
             and "error" not in r]
        return sum(v) / len(v) if v else None

    labels = [c[0] for c in configs if c[0] != "baseline"]
    print("\n| batch | baseline tok/s | " + " | ".join(f"{l} speedup" for l in labels) + " |")
    print("|---" * (2 + len(labels)) + "|")
    for b in a.batches:
        base = tps("baseline", b)
        if base is None:
            continue
        cells = []
        for l in labels:
            v = tps(l, b)
            cells.append(f"{v / base:.3f}x" if v else "-")
        print(f"| {b} | {base:.1f} | " + " | ".join(cells) + " |")

    print("\nCrossover (first batch size where speedup <= 1.0):")
    for l in labels:
        cross = None
        for b in a.batches:
            base, v = tps("baseline", b), tps(l, b)
            if base and v and v / base <= 1.0:
                cross = b
                break
        print(f"  {l:12s}: " + (f"batch {cross}" if cross
                                else "no crossover within the swept range"))

    print("\nAcceptance across the sweep (should be flat -- Phase 3 control):")
    for l in labels:
        vals = [(b, [r.get("acceptance_rate") for r in rows
                     if r.get("label") == l and r.get("concurrency") == b
                     and r.get("acceptance_rate") is not None])
                for b in a.batches]
        seq = [(b, sum(v) / len(v)) for b, v in vals if v]
        if seq:
            lo, hi = min(x for _, x in seq), max(x for _, x in seq)
            print(f"  {l:12s}: " + ", ".join(f"b{b}={x:.3f}" for b, x in seq)
                  + f"   (spread {hi - lo:.4f})")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
