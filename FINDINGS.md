# Findings log

Running record of what each experiment established, in order. Each entry states
the question, the result, the mechanism, and what it does *not* show. Raw data
is in `results/<name>.json`; the script that produced it is `bench/<name>.py`.

Project context: testing whether vLLM's DSpark adaptive verification cost model
— one-dimensional in total verification-token count, profiled once at startup
against a fixed synthetic context length — is adequate under continuous batching
with heterogeneous sequence lengths.

Baseline throughout: `vllm==0.28.0`, Qwen3-1.7B (bf16, 28 layers, 8 KV heads,
head_dim 128 → 112 KB KV per token), RTX 4060 Laptop (sm_89, ~7950 MB usable,
~256 GB/s peak), WSL2. V2 GPU model runner. FLASH_ATTN backend.

---

## E1 — Is the exchangeable-token assumption violated?

`bench/verify_cost_probe.py` → `results/verify_cost_probe.json`

**Question.** vLLM prices a verification step with a curve
`num_target_tokens -> forward_ms`, profiled at one synthetic context length
(`VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN`, default 8192) and applied to
every batch regardless of its actual context. Does context composition change
step cost at fixed token count?

**Result: yes, decisively.** At fixed verification-token count, step latency
varies up to **3.7×** (275%) with context length alone, monotonically, across
every token count from 16 to 512. Median within-cell replay spread was 2.2%
over 7 replays — the effect is ~2 orders of magnitude above the noise floor.

The stock 8192 baseline **overpredicts short-context batches by up to 73%**.
Budget selection is `argmax(estimated_accepted / cost)`
(`adaptive_verification.py:329`), so an inflated denominator makes vLLM
**under-allocate verification budget exactly where speculation is cheapest**.

**Mechanism: the cost surface is bytes-moved ÷ bandwidth, in two additive
terms.**

    cost ~= f(num_query_tokens) + k * sum(seq_lens)

- The context term is *independent of token count* (~39 ms at ctx=8192 whether
  verifying 16 tokens or 512) and *linear in context* (0.5 → 2 → 4.5 → 10 → 20
  → 40 ms as ctx doubles).
- Measured KV streaming rate is constant at **181 GB/s** (71% of peak) across
  ctx=2048/4096/8192. Attention here does nothing but stream KV from DRAM.
- The `ctx=0` floor is **weight streaming**: 1.72B params × 2 B = 3.44 GB read
  in 14.38 ms = **239 GB/s**, near peak. That is why cost is flat at 14.4 ms for
  n=16/32/64 — the forward pass is bound by reading its own weights and the
  query tokens are free. It rises only at n≥128 as GEMM arithmetic starts to
  matter.
- Cross-check: 112 KB/token × 26,576-token pool = 2.84 GiB, exactly matching
  vLLM's logged `Available KV cache memory: 2.84 GiB`.

**Implication for the cost model.** The right feature is not mean or max
context — it is **`sum(seq_lens)`**, because cost tracks aggregate bytes read.
Two features (`num_tokens`, `sum(seq_lens)`) should suffice, as a linear fit
that is trivial to evaluate in the scheduler and to update online. The
mean/max/query-weighted ablation is predicted to come out flat.

**What this does NOT show.**
- *Uniform contexts only.* `set_dummy_context` applies one scalar to every
  request, so this shows context *length* matters, not that its *distribution*
  does.
- *Piecewise/eager regime only.* Capture sizes came out `[1,2,4,8]` (capped by
  `max_num_seqs=8`), so every cell ran above the cudagraph limit. The
  graph-padded branch of the cost table is untested.
- *One model, one GPU.* On an H100 with a larger model, arithmetic intensity
  rises and the two terms will not separate this cleanly. The direction should
  transfer; the shape may not.
- KV blocks alias past the 26,576-token pool. Since 2.84 GiB is ~90× the 32 MB
  L2 there is no meaningful reuse, so this biases high-context numbers only
  slightly downward — the reported spread is a mild lower bound.

---

## E2 — Is marginal verification cost allocation-dependent?

`bench/marginal_cost_probe.py` → `results/marginal_cost_probe.json`

**Question.** The proposal's extension ranks each candidate draft token by
expected survival benefit relative to its *marginal* verification cost. That
assumes cost depends on *which* request receives an extra token. E1's model
predicts it does not: the KV term depends on the seq_lens of the requests in
the batch, and a request's KV is read in full whether it contributes one query
token or seven.

**Design.** Hold the request set, their contexts (hence total KV bytes), and
the total query-token count all fixed; vary only which requests receive the
spare tokens — all to long-context requests, all to short-context, or spread
evenly.

**Result: marginal cost is UNIFORM.** All configurations cost the same to
within 0.21%, and the long-vs-short difference (0.15%) is *smaller than the
drift control* (0.21%):

    alloc                   query_lens  median_ms
      even     [4, 4, 4, 4, 4, 4, 4, 4]     24.829
      long     [7, 7, 7, 7, 1, 1, 1, 1]     24.823
     short     [1, 1, 1, 1, 7, 7, 7, 7]     24.786
    even#2     [4, 4, 4, 4, 4, 4, 4, 4]     24.777   <- drift control

`long` vs `short` is the clean test: identical multiset of query lengths
(hence identical `max_query_len` and token total), identical contexts,
differing only in which requests own the long queries.

**Cross-validation of E1.** All four land at 24.8 ms against E1's two-term
prediction of **24.90 ms** — 0.4% error from a model fitted on entirely
different batch shapes. Independent confirmation of
`cost ~= f(num_tokens) + k * sum(seq_lens)`.

**Implication.** The proposal's cost-aware per-request allocation extension has
**no cost signal to exploit in this regime** — ranking draft slots by survival
benefit per marginal cost degenerates to ranking by survival alone, because
marginal cost is constant across slots. The mechanism is the one E1 identified:
a request's KV is streamed in full whether it contributes one query token or
seven, and KV streaming dominates. Recommend not implementing that extension
until a regime is found where it does not degenerate; report it as a clean
negative result instead. The global cost-model fix from E1 remains well
motivated and is where the gain is.

**Methodology note — this result required two attempts, and the first was
wrong.** Warming each plan immediately before measuring it left per-shape
compile/autotune cost inside the measurement window, and whichever plan ran
first absorbed it. That produced a spurious 28% spread and an
"allocation-dependent" verdict; the `even` cell visibly drifted
`51.7 -> 31.9 -> ... -> 26.8` across its replays. Two changes fixed it:

1. Warm **every** shape before timing **any** of them — each distinct
   `max_query_len` triggers its own compile.
2. **Interleave** rounds (one replay of each plan per round) rather than
   running each plan to completion, so residual drift hits all plans equally.
   Rounds 11-12 of the corrected run show a synchronized ~2 ms bump across all
   four plans — a thermal/clock event that plan-at-a-time timing would have
   charged to a single configuration.

Any future timing experiment here must do both, and must carry a duplicate-cell
drift control. Note that within-cell spread (9.4%) is *larger* than the effect
being bounded (0.21%), so the duplicate control — not the within-cell spread —
is what makes the null result credible.

**What this does NOT show.**
- *One batch composition* (4×4096 + 4×256, 32 query tokens). A regime with
  much longer queries or a compute-bound GPU could surface a real allocation
  effect, since attention QK work `sum(query_len * seq_len)` genuinely differs
  5x between `long` and `short` here (115.9k vs 23.8k pairs) yet costs nothing
  measurable — it is dwarfed by KV streaming on this bandwidth-starved part.
  **This is the specific thing to re-test on an H100.**
- Same eager/piecewise regime and same single model/GPU caveats as E1.

---

## E5 — the CUDA-graph-padded regime, and replicating vLLM#52057's drift plot

`bench/verify_cost_probe.py` (different arguments) + `bench/plot_drift.py`
→ `results/reqs_sweep.json`, `results/reqs_sweep_dense.json`

**Question.** E1 and E2 ran entirely *above* the cudagraph capture limit
(capture sizes were `[1,2,4,8]` with `max_num_seqs=8`), so the step-function
branch of the cost table was never exercised. Issue #52057 attributes the drift
the maintainers observe specifically to cudagraph replay, so that regime had to
be measured. Separately, #52057's own plot is the natural presentation format.

**How to reach the regime — no new code needed.** `_dummy_run` derives
`num_reqs = min(num_tokens, max_num_reqs)` (`model_runner.py:653`), so request
count is not directly settable. Running with `--max-num-seqs 64` and token
counts drawn from the captured sizes makes `num_reqs == num_tokens`: one token
per request, the decode shape, entirely inside the captured range
(`[1,2,4,8,16,24,32,40,48,56,64]`).

**Result: the drift is far worse in the regime real serving occupies.**

    n=  1: +53.6%     n= 16: +257.7%     n= 48: +720.8%
    n=  8: +141.5%    n= 32: +533.2%     n= 64: +956.6%

**9.6x at 64 requests**, against 3.7x in E1. The mechanism is E1's model: in the
decode shape `sum(seq_lens) = num_reqs * context`, so both factors scale
together. The profiler charges 64x8192 = 524k KV tokens; a real batch with
256-token contexts reads 16k.

**The `ctx=0` column is flat at ~14 ms for every request count from 1 to 64.**
Request count costs essentially nothing by itself — the floor is weight
streaming. All growth along the predicted axis is the profiler's *assumed*
8192 context, not batch size. Third independent confirmation of
`cost ~= f(num_tokens) + k * sum(seq_lens)`.

**One cell underestimates.** `n=1, ctx=0` at **+17.3%** is the only point above
the diagonal in 54 cells: at a single request, fixed per-step overhead stops
being amortised and a curve fitted at 8192 underprices it.

**Agreement statistics** (dense sweep, n=2,695 — matching #52057's n=2,706):

    bias = -42.300 ms    MAE = 42.314 ms    RMSE = 57.170 ms    r = 0.475

`MAE ~= |bias|` means nearly every error carries the same sign: vLLM
overestimates almost everywhere. A consistently-signed error implies a
correctable systematic term rather than irreducible variance.

**Comparison with #52057's published plot.**

    metric      #52057      this work
    n            2,706        2,695
    bias         2.353 ms    -42.300 ms
    RMSE         2.599 ms     57.170 ms
    r            0.998        0.475

The *format* is replicated (hexbin, log10(count) colour, diagonal, stats box);
the *statistics* deliberately are not, and cannot be with this harness. Theirs
isolates cudagraph-replay drift at matched sequence length — a near-perfect
predictor with mild error, exactly as their issue describes. Ours varies
context away from the profiled value, which is the *second*, still-open
mechanism their issue names. **On the same axes and units, the context effect
is roughly 20x larger than the replay drift that motivated the issue.** That
contrast is the most compelling single presentation of this project's premise.

Note also that as read off their figure, `bias = 2.353` with `MAE = 0.2353` is
not internally possible: MAE >= |bias| always. One of those values is likely
misread; do not quote them without re-checking the source image.

**What this does NOT show.**
- Reproducing their *statistics* needs vLLM's own startup profiling compared
  against real generation steps (i.e. E3), not our dummy runs with a curve we
  fit ourselves. In this harness the ctx=8192 cells have zero error *by
  construction*, since Phase A defines the curve from them.
- KV blocks alias heavily at the high end (64x8192 = 524k nominal vs a
  23,984-token pool), biasing high-context costs downward. The high-end ratios
  are less precisely grounded than the low-end ones.
- Same single model / single GPU caveat as E1 and E2.

---

## E3 — stock DSpark reproduced end to end

`bench/dspark_baseline.py` → `results/dspark_baseline.json`

**Question.** E1, E2 and E5 all ran with *no speculator*, so
`AdaptiveVerificationManager` never actually executed — they measured the cost
surface it prices, not the manager pricing it. Does stock DSpark run, does the
manager engage, and does the curve it profiles match what E1 predicts?

**Result: yes.** With `openbmb/MiniCPM5-2B` + `openbmb/MiniCPM5-2B-DSpark`:

    adaptive verification : True          speculator : DSparkSpeculator
    spec steps            : 3             profile ctx len : 8192
    cudagraph capture     : [1, 2, 4, 8, 16, 24, 32]   limit 32

    verify cost table (tokens -> ms)      draft cost table (reqs -> ms)
        1 ->  19.633    64 ->  22.447         0..8 -> 7.024 (flat)
        2 ->  19.633   256 ->  47.124
        8 ->  19.633  1024 -> 181.788
       32 ->  19.633

Real generation produced 384 output tokens with healthy speculation:

    drafts 156 | draft tokens 468 | accepted 224 -> 47.9%
    accepted per position [112, 70, 42] -> 71.8% / 44.9% / 26.9%
    = 1.44 accepted tokens per draft

**The profiled table confirms E1 and E5 from vLLM's own profiler.** It is
**flat at 19.633 ms from 1 token to 32** — vLLM believes a 1-token step and a
32-token step cost the same. Both causes are this project's thesis: below the
cudagraph limit cost is a step function, and every profiling run assumed 8192
tokens of context, which swamps the token count. E5 measured the real range at
that shape as ~14 ms (ctx=0) to ~147 ms (ctx=8192). vLLM has one number for it.

**A second, subtler signal.** `build_cost_tables_from_curves` applies
`np.maximum.accumulate` to force a non-decreasing curve. A perfectly flat run
of 19.633 across 1..32 means the raw measurements were **not** monotonic —
smaller sizes profiled *slower* than larger ones and the clamp flattened them.
That is direct evidence of the noisy small-size profiling #52057 complains
about, visible in the shipped code path rather than in our harness.

### Three obstacles, worth recording

1. **Qwen3.5 DSpark drafts cannot load at all** (unreported vLLM bug):
   `speculative.py` rewrites any `qwen3_5*` draft to `Qwen3_5MTP`, overwriting
   the checkpoint's declared `Qwen3DSparkModel`, which then falls through to
   the DeepSeek-V4 DSpark class and dies on `AttributeError: ... 'hc_mult'`.
   `openbmb/MiniCPM5-2B-DSpark` (`model_type=qwen3`) dodges the remap.
2. **Adaptive verification requires Hopper+ if using FlashAttention.**
   `flash_attn.py:356` reports `AttentionCGSupport.ALWAYS` only at FA3, which
   needs sm_90+. On sm_89 vLLM loads FA2 (`UNIFORM_BATCH`) and the manager
   refuses to initialise. Workarounds: `TRITON_ATTN` or `FLEX_ATTENTION`.
3. **`VLLM_ATTENTION_BACKEND` no longer exists in 0.28** and is silently
   ignored. Use the `attention_backend` config field. Always confirm the
   "Using X attention backend" log line reflects what you asked for.

**What this does NOT show.**
- Obstacle 2 forced **TRITON_ATTN**, while E1/E2/E5 used FLASH_ATTN, so these
  timings are *not* directly comparable to the earlier experiments.
- Different model (MiniCPM5-2B vs Qwen3-1.7B), `max_model_len 2048`, and a tiny
  8-prompt workload. This is a smoke test proving the path works, not a
  baseline for publication.
- Obstacle 2 also means a production-representative baseline needs Hopper+
  hardware, where FA3 is available and the backend matches deployment.

---
