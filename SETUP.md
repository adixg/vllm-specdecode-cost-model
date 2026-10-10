# Running these experiments on PACE

Everything in this repository runs on Georgia Tech's PACE `ice-gpu` partition.
The earlier WSL2 + RTX 4060 setup is gone: the laptop could not run adaptive
verification at all without changing the attention backend (see below), and the
measurements that survived are all H100. E1, E2 and E5 were taken on that
laptop and are kept in `FINDINGS.md` as history; E6 re-ran both sweeps on an
H100 and supersedes them.

## The hardware constraint, and why it forces Hopper

This is the single most load-bearing fact about the project.

`AdaptiveVerificationManager` captures varlen decode cudagraphs, so every
attention builder must report `AttentionCGSupport.ALWAYS`. FlashAttention does
so **only at FA3** (`v1/attention/backends/flash_attn.py:357`), and FA3
requires compute capability 9.0 or higher. On anything older vLLM loads FA2,
which reports `UNIFORM_BATCH`, and startup fails:

    ValueError: Adaptive verification captures varlen decode cudagraphs, so
    every attention builder must report AttentionCGSupport.ALWAYS, but
    FlashAttentionBackend reports AttentionCGSupport.UNIFORM_BATCH.

That rules out A100 (sm_80), L40S (sm_89), V100 and consumer cards. On PACE the
H100 and H200 nodes in `ice-gpu` are the ones that work. Every sbatch script in
`pace/` checks `compute_cap` is 9.x and exits otherwise.

Two backends do report `ALWAYS` and will run on older hardware — `TRITON_ATTN`
and `FLEX_ATTENTION` — but changing the backend changes step timings, so
results taken under Triton are **not comparable** with FLASH_ATTN results. Use
them to confirm a code path executes, never to measure.

**`VLLM_ATTENTION_BACKEND` does not exist in vLLM 0.28.** It is silently
ignored, with no warning. Pass `attention_backend` to `LLM(...)` or
`--attention-backend` on the CLI, and always confirm the log line

    Using FLASH_ATTN attention backend out of potential backends: [...]

names the backend you asked for.

## Environment

The conda environment lives in scratch, because home on ICE is only 15 GB:

    module load anaconda3
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ~/scratch/conda-envs/cs6222

`pace/env.sh` holds everything a job needs and every sbatch script sources it:

    source "${SLURM_SUBMIT_DIR:-$(pwd)}/pace/env.sh"

It sets `HF_HOME`, `HF_HUB_OFFLINE=1` and `VLLM_USE_V2_MODEL_RUNNER=1`, and
offers `require_hf_repo org/name`, which fails in seconds if a checkpoint is
missing rather than eight minutes into an allocation.

**Keep vLLM variables in a sourced file, never in a login shell.** Jobs 5922739
and 5922740 died at engine construction with `LocalEntryNotFoundError` because
`HF_HOME` was set interactively and reached jobs only through
`sbatch --export=ALL`. Submitted from a shell without it, offline mode left
vLLM looking in the default cache, where nothing was stored. A variable in
`~/.zshrc` is invisible to collaborators and silently changes what you measure.

### Two required variables

- **`VLLM_USE_V2_MODEL_RUNNER=1`** — adaptive verification and DSpark exist
  only in the V2 GPU model runner. The V2 runner lives under
  `vllm/v1/worker/gpu/`; do not read that directory name as the version.
- **`VLLM_ALLOW_INSECURE_SERIALIZATION=1`** — passing a *callable* to
  `LLM.collective_rpc` ships the function to the engine subprocess, and the
  default msgspec encoder refuses: `TypeError: Object of type <class
  'function'> is not serializable`. The harnesses set this themselves; anything
  else driving the worker by callable needs it.

### Models must be cached before the job runs

Compute nodes have no internet. Download on a login node:

    export HF_HOME=~/scratch/hf
    HF_HUB_OFFLINE=0 hf download openbmb/MiniCPM5-2B
    HF_HUB_OFFLINE=0 hf download openbmb/MiniCPM5-2B-DSpark

Check quota with `pace-quota`. Home holds the repository including `results/`
and all of `.git`, so watch it: the result JSONs run 700-850 KB each and git
keeps every version forever.

## Scheduling

`ice-gpu` H100 nodes are heavily contended and priority is dominated by the QOS
term, so every student's job is effectively tied. Duration is one of the few
levers that changes anything.

**Request the shortest walltime the work needs — 30 minutes or less — and split
larger campaigns across several small jobs.** Backfill can slot a short job
into gaps a long one cannot fit. A 75-minute job asking for several experiments
in one allocation sat unscheduled for 48 hours and never ran; the same work as
five 25-minute jobs completed the same day.

Avoid cancelling and resubmitting, which resets the job's accrued age — one of
the few tiebreakers among tied jobs. And prefer offline analysis over new GPU
time wherever the data already on disk can answer the question: several
findings in `FINDINGS.md` came from re-reading committed traces, not new runs.

## Reading `_dummy_run` results

The cost-surface probes (`bench/verify_cost_probe.py`,
`bench/marginal_cost_probe.py`) drive vLLM's own `_dummy_run`, which has three
limitations worth knowing before trusting a number.

**It can only express uniform context.** `_dummy_run` derives
`num_reqs = min(num_tokens, max_num_reqs)` (`v1/worker/gpu/model_runner.py:653`)
and applies a single scalar `context_len` to every request. Heterogeneous
per-request contexts need a patched `_dummy_run` or real requests through
`LLM.generate`.

**High-context cells alias in the KV pool.** Past the pool size,
`set_dummy_context` wraps block ids modulo the pool, so spans share blocks. The
reads are realistic but more cache-friendly than a genuine batch of that size,
which makes high-context costs a **lower bound**. E6 avoided this by sizing the
KV pool (567,584 tokens) above the largest test cell (524,288).

**Watch the reported cudagraph capture sizes.** `--max-num-seqs 8` caps capture
at 8, so every cell above that runs piecewise/eager and the graph-padded branch
of the cost table goes untested. E1 and E2 ran entirely in the eager regime for
exactly this reason; E5 exists to cover the graphed one.

## Measuring GPU memory from a launcher

Read it with `pynvml`, never `torch.cuda`. Touching `torch.cuda` in a parent
process creates a CUDA context there and consumes memory the spawned engine
then cannot have, which surfaces as

    ValueError: Free memory on device ... is less than desired GPU memory
    utilization

`pynvml` queries the driver without creating a context.
