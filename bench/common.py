"""Shared helpers: launch a vLLM engine in a subprocess and parse its
memory-accounting logs.

vLLM reports its memory breakdown only through the logger, not the Python API,
so the reliable way to capture it is to run each engine in a clean subprocess
and parse stderr. Running each config in its own process also guarantees that
allocator state and CUDA-graph memory from a previous config cannot leak into
the next measurement -- which matters here, because the whole point is to
compare configs fairly.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

# INFO ... Free memory on device (6.93/8.0 GiB) on startup. Desired GPU memory
# utilization is (0.85, 6.8 GiB). Actual usage is 4.06 GiB for consumed memory
# (weights + non-torch), 1.65 GiB for peak activation, and 0.55 GiB for
# CUDAGraph memory. ... Current kv cache memory in use is 1.08 GiB.
_MEM = re.compile(
    r"Free memory on device \((?P<free>[\d.]+)/(?P<total>[\d.]+) GiB\).*?"
    r"Actual usage is (?P<weights>[\d.]+) GiB for consumed memory.*?"
    r"(?P<activation>[\d.]+) GiB for peak activation.*?"
    r"(?P<cudagraph>[\d.]+) GiB for CUDAGraph memory.*?"
    r"Current kv cache memory in use is (?P<kv>[\d.]+) GiB",
    re.S,
)
# INFO ... GPU KV cache size: 28,384 tokens, Maximum concurrency for 2,048
# tokens per request: 13.86x
_KV = re.compile(
    r"GPU KV cache size: (?P<tokens>[\d,]+) tokens.*?"
    r"Maximum concurrency for (?P<seqlen>[\d,]+) tokens per request: "
    r"(?P<concurrency>[\d.]+)x"
)
# vLLM's own suggestion for the byte budget that fully uses the GPU.
_SUGGEST = re.compile(r"--kv-cache-memory=(?P<fit>\d+).*?--kv-cache-memory=(?P<full>\d+)", re.S)


@dataclass
class Probe:
    """One engine configuration and the memory it actually consumed."""
    label: str
    target: str
    draft: str | None
    k: int | None
    gmu: float
    max_model_len: int
    kv_bytes: int | None
    # measured
    kv_tokens: int | None = None
    num_gpu_blocks: int | None = None
    block_size: int | None = None
    gib_weights: float | None = None
    gib_activation: float | None = None
    gib_cudagraph: float | None = None
    gib_kv: float | None = None
    gib_free_at_start: float | None = None
    suggest_fit_bytes: int | None = None
    suggest_full_bytes: int | None = None
    ok: bool = False
    error: str | None = None

    def as_dict(self):
        return asdict(self)


def run_probe(label, target, draft=None, k=None, gmu=0.85,
              max_model_len=512, kv_bytes=None, timeout=1800) -> Probe:
    """Launch engine_probe.py in a subprocess and parse what it reports."""
    p = Probe(label=label, target=target, draft=draft, k=k, gmu=gmu,
              max_model_len=max_model_len, kv_bytes=kv_bytes)

    cmd = [sys.executable, str(REPO / "bench" / "engine_probe.py"),
           "--target", target, "--gmu", str(gmu),
           "--max-model-len", str(max_model_len)]
    if draft:
        cmd += ["--draft", draft, "--k", str(k)]
    if kv_bytes:
        cmd += ["--kv-bytes", str(kv_bytes)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=REPO, env=os.environ.copy())
    except subprocess.TimeoutExpired:
        p.error = f"timeout after {timeout}s"
        return p

    blob = r.stdout + r.stderr

    if m := _MEM.search(blob):
        p.gib_free_at_start = float(m["free"])
        p.gib_weights = float(m["weights"])
        p.gib_activation = float(m["activation"])
        p.gib_cudagraph = float(m["cudagraph"])
        p.gib_kv = float(m["kv"])
    if m := _KV.search(blob):
        p.kv_tokens = int(m["tokens"].replace(",", ""))
    if m := _SUGGEST.search(blob):
        p.suggest_fit_bytes = int(m["fit"])
        p.suggest_full_bytes = int(m["full"])

    for line in blob.splitlines():
        if line.startswith("RESULT "):
            d = json.loads(line[len("RESULT "):])
            p.num_gpu_blocks = d.get("num_gpu_blocks")
            p.block_size = d.get("block_size")
            if p.kv_tokens is None:
                p.kv_tokens = d.get("kv_tokens")
            p.ok = True

    if not p.ok:
        tail = [l for l in blob.splitlines()
                if "Error" in l or "error" in l or "Traceback" in l]
        p.error = tail[-1][:300] if tail else f"exit {r.returncode}, no RESULT line"
    return p


def save(name: str, rows: list[Probe], meta: dict):
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(
        {"meta": meta, "rows": [r.as_dict() for r in rows]}, indent=2))
    return out
