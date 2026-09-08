"""Diagnose a Phase 2 divergence: real bug, or floating-point non-determinism?

Speculative decoding is output-equivalent in exact arithmetic. In floating
point it need not be bitwise identical: the target model scores k+1 candidate
positions in ONE batched forward pass, whereas plain decoding scores one
position per pass. Different batch shapes select different kernels and
different reduction orders, so logits differ in the last bits. Where the top-2
candidates are nearly tied, that can flip an argmax -- and once one token
differs, everything after it does too.

The control: plain decoding, with no speculation anywhere, is subject to the
same effect just from changing BATCH COMPOSITION. So run the baseline twice --
once with the prompt alone, once inside the full prompt set -- and see whether
it already disagrees with itself at the same positions. If it does, the
divergence is inherent numerical noise, not a speculative-decoding bug.

We also print the top-2 logit gap at the divergence point. A near-tie
(gap ~< 1e-2) means a coin-flip that floating-point noise can flip. A large gap
means a genuine bug, and the sweep must not proceed.

Usage:  source env.sh && python bench/phase2_diagnose.py --prompts 0 1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def child(args, timeout=2400):
    cmd = [sys.executable, str(REPO / "bench" / "gen_tokens.py")] + args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO)
    out = {}
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith("TOKENS "):
            out["tokens"] = json.loads(line[len("TOKENS "):])
        if line.startswith("LOGPROBS "):
            out["logprobs"] = json.loads(line[len("LOGPROBS "):])
    if "tokens" not in out:
        raise RuntimeError(f"no TOKENS (exit {r.returncode}):\n{(r.stdout+r.stderr)[-1500:]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gmu", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--kv-bytes", type=int, default=536519168)
    ap.add_argument("--prompts", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()

    base = ["--target", a.target, "--gmu", str(a.gmu),
            "--max-model-len", str(a.max_model_len),
            "--max-tokens", str(a.max_tokens), "--kv-bytes", str(a.kv_bytes)]

    print("[diag] baseline, full prompt set ...", flush=True)
    full = child(base + ["--logprobs", "5"])

    print("[diag] baseline, one prompt at a time ...", flush=True)
    solo = {}
    for i in a.prompts:
        solo[i] = child(base + ["--only", str(i)])

    print("[diag] spec decode, full prompt set ...", flush=True)
    spec = child(base + ["--draft", a.draft, "--k", str(a.k)])

    report = {"prompts": a.prompts, "k": a.k, "findings": []}
    for i in a.prompts:
        b_full = full["tokens"][i]
        b_solo = solo[i]["tokens"][0]
        s_full = spec["tokens"][i]

        def div(x, y):
            for j, (p, q) in enumerate(zip(x, y)):
                if p != q:
                    return j
            return None

        d_batch = div(b_full, b_solo)   # baseline vs baseline, batching only
        d_spec = div(b_full, s_full)    # baseline vs speculation

        lp = full.get("logprobs", {}).get(str(i)) or full.get("logprobs", {}).get(i)
        gap = None
        if lp and d_spec is not None and d_spec < len(lp):
            vals = sorted(lp[d_spec], reverse=True)
            if len(vals) >= 2:
                gap = round(vals[0] - vals[1], 6)

        f = {"prompt": i,
             "baseline_vs_baseline_batching_divergence": d_batch,
             "baseline_vs_spec_divergence": d_spec,
             "top2_logprob_gap_at_spec_divergence": gap}
        report["findings"].append(f)
        print(f"\nprompt {i}:")
        print(f"  baseline(batch6) vs baseline(batch1) diverges at : {d_batch}")
        print(f"  baseline(batch6) vs spec k={a.k}      diverges at : {d_spec}")
        print(f"  top-2 logprob gap at that position              : {gap}")

    out = REPO / "results" / "phase2_diagnose.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
