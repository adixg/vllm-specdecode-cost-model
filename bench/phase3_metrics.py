"""Phase 3: what is speculative decoding actually doing?

Throughput alone cannot tell us WHY speculation helps or stops helping. Two
quantities separate the explanations:

  acceptance rate     accepted / drafted -- how often the draft model guesses
                      what the target would have said. A property of the MODEL
                      PAIR. It should not depend on batch size.

  acceptance length   mean tokens emitted per verification step, counting the
                      bonus token. This is the theoretical speedup CEILING: if
                      a verify step cost the same as a decode step, throughput
                      would improve by exactly this factor.

  per-position        P(the i-th draft token survives). Decays with position,
  acceptance          and tells us where extra k stops paying for itself.

Phase 4 will show speedup falling as batch grows. This phase establishes the
baseline against which that fall is interpreted: if acceptance stays flat while
speedup decays, the decay is a SYSTEMS effect (verification competing for
compute), not a quality effect. That distinction is the whole experiment.

Usage:  source env.sh && python bench/phase3_metrics.py
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


def run(a, k, method=None, draft=None, batch=None, max_num_seqs=None):
    cmd = [sys.executable, str(REPO / "bench" / "spec_probe.py"),
           "--target", a.target, "--gmu", str(a.gmu),
           "--max-model-len", str(a.max_model_len),
           "--max-tokens", str(a.max_tokens),
           "--kv-bytes", str(a.kv_bytes), "--k", str(k),
           "--batch", str(batch if batch is not None else a.batch)]
    if max_num_seqs:
        cmd += ["--max-num-seqs", str(max_num_seqs)]
    if method:
        cmd += ["--method", method]
    elif draft:
        cmd += ["--draft", draft]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400, cwd=REPO)
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith("SPEC "):
            return json.loads(line[len("SPEC "):])
    raise RuntimeError(f"no SPEC line (exit {r.returncode}):\n"
                       f"{(r.stdout + r.stderr)[-2000:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--gmu", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--kv-bytes", type=int, default=536519168)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 8],
                    help="max_num_seqs values for the flatness control")
    ap.add_argument("--flat-prompts", type=int, default=12,
                    help="fixed workload size for the flatness control")
    a = ap.parse_args()

    gpu_state = ensure_free()

    rows = []
    for k in a.ks:
        print(f"[phase3] draft-model k={k}, batch={a.batch} ...", flush=True)
        r = run(a, k, draft=a.draft)
        r["label"] = f"draft-k{k}"
        rows.append(r)

    for k in a.ks:
        print(f"[phase3] ngram k={k}, batch={a.batch} ...", flush=True)
        try:
            r = run(a, k, method="ngram")
            r["label"] = f"ngram-k{k}"
            rows.append(r)
        except Exception as e:
            print(f"         ngram k={k} failed: {str(e)[:200]}", flush=True)

    # Control: acceptance is a model-pair property, so it must not move with
    # batch size. If it does, the Phase 4 decay cannot be called a systems
    # effect.
    #
    # Batch size MUST be varied with max_num_seqs while the submitted workload
    # stays fixed. An earlier version varied the number of prompts instead, so
    # batch 1 ran only prompt 0 while batch 8 ran all six; that measured a
    # difference in prompt content, not in concurrency, and produced a spurious
    # "acceptance is not flat" result.
    flat = []
    for c in a.concurrency:
        print(f"[phase3] flatness control: draft k=3, {a.flat_prompts} prompts, "
              f"max_num_seqs={c} ...", flush=True)
        r = run(a, 3, draft=a.draft, batch=a.flat_prompts, max_num_seqs=c)
        r["label"] = f"draft-k3-conc{c}"
        flat.append(r)

    out = REPO / "results" / "phase3_metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"gpu_state": gpu_state, "rows": rows,
                               "flatness": flat}, indent=2))

    print("\n| config | accept rate | accept length | tok/s | per-position acceptance |")
    print("|---|---|---|---|---|")
    for r in rows:
        pp = ", ".join(f"{x:.3f}" for x in r.get("acceptance_per_pos", []))
        print(f"| {r['label']} | {r.get('acceptance_rate', float('nan')):.3f} | "
              f"{r.get('mean_acceptance_length', float('nan')):.3f} | "
              f"{r['output_toks_per_s']} | {pp} |")

    print("\nFlatness control: fixed workload, concurrency varied via max_num_seqs")
    for r in flat:
        print(f"  max_num_seqs {r['max_num_seqs']:>3}: acceptance_rate="
              f"{r.get('acceptance_rate', float('nan')):.4f}  "
              f"accept_len={r.get('mean_acceptance_length', float('nan')):.4f}  "
              f"tok/s={r['output_toks_per_s']}")
    if len(flat) >= 2:
        lo, hi = flat[0], flat[-1]
        d = abs(lo.get("acceptance_rate", 0) - hi.get("acceptance_rate", 0))
        print(f"  delta across max_num_seqs {lo['max_num_seqs']} -> "
              f"{hi['max_num_seqs']}: {d:.4f}"
              + ("  (flat, as expected)" if d < 0.02 else "  (NOT flat -- investigate)"))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
