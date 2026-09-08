"""Preflight: make sure nothing else is holding the GPU before we measure.

This machine shares one 8 GB GPU with Ollama, which keeps a model resident after
a request finishes -- observed at 11 GB with a 24-hour keep_alive, leaving ~600
MB free. A benchmark started in that state produces numbers that look like a
real effect but are contamination. One Phase 3 run was lost that way.

So every phase calls ensure_free() before measuring, and records the observed
free memory in its results so a bad run is identifiable after the fact.

Deliberately conservative: this unloads Ollama MODELS (a cheap, reversible
operation -- Ollama reloads on next use) but never kills processes. Other
workloads may belong to someone else's in-flight work, so they are reported and
the run is aborted rather than cleared automatically.
"""
from __future__ import annotations

import subprocess
import time

GIB = 1024 ** 3


def gpu_free_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def gpu_util_pct() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def loaded_ollama_models() -> list[str]:
    """Model names Ollama currently holds resident, if Ollama is installed."""
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True,
                             text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    lines = [l for l in out.stdout.strip().splitlines()[1:] if l.strip()]
    return [l.split()[0] for l in lines]


def unload_ollama() -> list[str]:
    """Ask Ollama to release any resident models. Returns what was unloaded."""
    freed = []
    for m in loaded_ollama_models():
        try:
            subprocess.run(["ollama", "stop", m], capture_output=True,
                           text=True, timeout=60)
            freed.append(m)
        except Exception:
            pass
    return freed


def ensure_free(min_gib: float = 6.3, wait_s: int = 30, verbose: bool = True) -> dict:
    """Unload Ollama, then wait for the GPU to actually have min_gib free.

    Raises RuntimeError if it cannot get there -- better to abort than to
    publish a contaminated measurement.
    """
    need_mib = int(min_gib * 1024)
    freed = unload_ollama()
    if verbose and freed:
        print(f"[gpu_guard] unloaded Ollama models: {', '.join(freed)}", flush=True)

    deadline = time.time() + wait_s
    free = gpu_free_mib()
    while free is not None and free < need_mib and time.time() < deadline:
        time.sleep(2)
        free = gpu_free_mib()

    util = gpu_util_pct()
    state = {"free_mib": free, "util_pct": util, "unloaded_ollama": freed,
             "required_mib": need_mib}

    if free is None:
        if verbose:
            print("[gpu_guard] WARNING: could not read nvidia-smi; proceeding",
                  flush=True)
        return state

    if free < need_mib:
        holders = loaded_ollama_models()
        raise RuntimeError(
            f"GPU not free: {free} MiB available, need {need_mib} MiB "
            f"({min_gib} GiB), utilization {util}%. "
            + (f"Ollama still holds: {holders}. " if holders else "")
            + "Something else is using the GPU. Check `nvidia-smi` and "
              "`pgrep -af python`; this may be another session's in-flight "
              "work, so stop it deliberately rather than assuming it is stale. "
              "Refusing to benchmark in this state -- the numbers would be "
              "contaminated.")

    if verbose:
        print(f"[gpu_guard] GPU ready: {free} MiB free, {util}% utilization",
              flush=True)
    return state


if __name__ == "__main__":
    import json
    import sys
    try:
        print(json.dumps(ensure_free(), indent=2))
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
