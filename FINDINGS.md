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


## Summary

| | Question | What varied | Result |
|---|---|---|---|
| **E1** | Does context change step cost at a fixed token count? | `num_tokens` 16-512 x `context_len` 0-8192, `num_reqs` pinned at 8 | **3.7x** spread (laptop) |
| **E2** | Does cost depend on *which* request gets the extra tokens? | ownership of the spare draft tokens only | **0.21%** - no |
| **E3** | Does stock DSpark run, and does the manager engage? | nothing; feasibility check | yes, 47.9% acceptance |
| **E5** | E1 again, but inside the CUDA-graph regime | `num_reqs` 1-64 (= `num_tokens`) x context | **9.6x** spread (laptop) |
| **E6** | E1 and E5 on production hardware (H100, FA3) | same sweeps, real GPU, no KV aliasing | **7.8x** graphed, **~1%** eager |
| **E4** | Which correction form predicts held-out contexts? | additive vs max-like, graphed and eager | additive wins for graphs; **11.6x** lower test RMSE |
| **E7** | Does the error change the chosen draft budget? | simulated context offset with measured marginal cost | yes, but predicted loss is **<1%** |
| **E8** | Does error change sign beyond the profile context? | contexts 1,024-32,000 on H100 | yes; scalar correction cuts median error **42x** |
| **E9** | First serving A/B and saturation sweep | spec 3/7, concurrency 16/64 | saturation is real; A/B result is **superseded** |
| **E10** | Calibrated in-engine MiniCPM spec-7 A/B | 20 paired rounds, exact profile reconstruction | correction changes decisions but throughput is **0.365% lower** |

Sections below are in experiment order. Numbering is historical, not
chronological: E5 was run before E3.

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

**Correction (added after E4).** This entry originally claimed the inflated
denominator makes vLLM *under*-allocate. That was asserted, never tested, and
is wrong in sign. Budget selection is `argmax(estimated_accepted / cost)`
(`adaptive_verification.py:329`), and a ratio's argmax is **invariant to any
purely multiplicative error** — scaling every candidate's cost by 1.73 changes
nothing. What the context mispricing actually produces is a *constant offset*,
`k * (num_reqs * 8192 - sum(seq_lens))`, identical across candidate budgets.
A positive offset inflates the fixed part of the cost, which makes marginal
draft tokens look relatively cheap, so vLLM **over**-allocates. See the
simulation under E7.

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

## E6 — the same sweeps on production hardware (H100, FA3)

`pace/run_h100.sbatch` → `results/h100_e1.json`, `results/h100_e5.json`

**Question.** Every earlier number came from a bandwidth-starved laptop GPU
running FA2 (or, for E3, a forced TRITON_ATTN backend), with a KV pool too
small to avoid block aliasing. Does the two-term model survive on hardware
where compute is not free and the measurement is clean?

**Setup.** PACE `ice-gpu` H100, compute capability 9.0, so vLLM selects
**FlashAttention 3** on its own — the configuration adaptive verification
requires and production uses. KV cache **567,584 tokens** against a largest
cell of 524,288, so **no aliasing**: unlike the laptop runs, these reads are
all real DRAM traffic.

**Result: the model survives, and `k` is far cleaner than on the laptop.**

    n     delta_ms(ctx 0->8192)   KV tokens    k (us/tok)   implied GB/s
     4            1.174             32,768       0.0358        3,201
    16            4.734            131,072       0.0361        3,175
    32            9.389            262,144       0.0358        3,202
    64           18.578            524,288       0.0354        3,237

`k` is **constant to +/-1%** across a 16x range of batch sizes at
**0.0356 us per KV token**, implying **3,221 GB/s** — about 96% of the H100
SXM's 3,350 GB/s peak. The laptop's apparent 2x variation in `k` was an
artifact of cache aliasing plus a small GPU failing to saturate bandwidth; on
clean hardware `k` is a scalar, which keeps the proposed improvement simple.

**The decisive finding: the effect is gated on CUDA graphs.**

    E1 (8 requests, eager)       +0.7% to +2.5%
    E5 (n requests, graphed)     up to +780%

Same GPU, same model, same contexts. E1's capture sizes were `[1,2,4,8]` so
every cell ran eager; E5's reached 64 so every cell replayed a graph. At equal
token counts:

    n      E1 (eager)    E5 (graphed)
    32       8.543 ms       2.167 ms
    64       8.349 ms       2.382 ms

E1's base cost is **flat at ~8.2 ms from 32 tokens to 512** — a 16x increase in
work at constant time. That is CPU kernel-launch overhead, not GPU work, and
it is large enough to absorb the KV reads entirely: at n=512, ctx=8192 the
two-term model predicts 10.49 ms and the measured value is 8.14 ms, unchanged
from ctx=0.

So cost is **not simply additive**. The KV term is visible only once it exceeds
whatever else bottlenecks the step:

- H100 + graphs (base ~2 ms): 18 ms of KV reads dominate → 780%
- H100 + eager (base ~8 ms of launch overhead): 2 ms of KV reads hide → ~0%
- Laptop + eager (base ~14 ms): 17 ms of KV reads still dominate → 275%

All three are consistent. **Production serving uses CUDA graphs**, so the
regime where vLLM's cost model is blindest is the regime that matters.

**Implication for the implementation.** A correction of the form
`table[n] + k * sum(seq_lens)` would over-predict in eager mode, where the KV
reads are free. Something closer to `max(table[n], base + k * sum(seq_lens))`
may be needed. Both regimes are now measured, so this is directly testable.

**Agreement statistics** against #52057's published figure:

    run           n       bias       MAE      RMSE       r
    h100_e1     630     -0.342     0.357     0.518    0.992
    h100_e5   2,695     -5.640     5.640     7.978    0.498
    #52057    2,706      2.353    0.2353     2.599    0.998

h100_e1 closely resembles their figure (r=0.992 vs 0.998): in that regime the
one-dimensional model genuinely works. h100_e5 is where it collapses. Note
these remain not strictly comparable — our predicted values are fitted from our
own dummy runs, theirs come from vLLM's startup profile measured against real
steps. `bench/step_trace.py` addresses that.

**What this does NOT show.**
- One model (Qwen3-1.7B) on one GPU generation.
- Still synthetic `_dummy_run` batches, not real serving.
- Uniform contexts within a batch; heterogeneity across requests is untested.

---

## E4 — which functional form should the correction use?

`bench/fit_cost_model.py` → fits on `results/h100_e1.json` + `results/h100_e5.json`

**Question.** E6 showed KV-read time is visible only when it exceeds the
per-step floor. Two forms follow:

    additive:  cost = floor + k * S
    max-like:  cost = max(floor, k * S)

with `S = num_reqs * context_len` and `floor` the measured cost of the same
batch shape at zero context. The decisive test is to fit **one k across both
regimes**: k is a property of the model and GPU, not of the execution mode, so
a single value must explain graphed and eager data alike. Contexts 1024 and
4096 were held out of fitting entirely. No GPU needed.

**Result: neither form wins outright — each wins in its own regime.**

        model     fitted k   train RMSE  test RMSE   graphed    eager
        stock            -        6.666      5.779     7.174    0.530
     additive    0.0346 us        0.602      0.496     0.096    0.824
          max    0.0426 us        0.626      0.879     1.091    0.072

Additive beats max-like by 11x on graphed steps; max-like beats additive by
11x on eager steps. **The correct model is regime-aware**, not one equation.
That matches the mechanism: with CUDA graphs there is no launch slack for KV
reads to hide in, so they add; in eager mode the GPU idles waiting on the CPU
and absorbs them.

**The result that validates the whole approach.** Additive's fitted
**k = 0.0346 us/KV-token** against **0.0356** derived independently from KV
bytes per token divided by measured bandwidth — a **3% match** between a
least-squares fit to latency data and arithmetic over a model config and a
hardware datasheet. Max-like's fitted k is 20% off, further evidence it is the
wrong form in the regime that dominates the data.

**Headline.** On held-out context lengths, prediction error falls from
**5.779 ms (vLLM today)** to **0.496 ms** — an **11.6x reduction from one
scalar term**.

**Implementation note.** The `floor` used here is the measured cost at
`ctx=0`, but vLLM's shipped table is profiled at `ctx=8192` and therefore
already contains the KV cost of `num_reqs * 8192`. A real correction must be
relative to that baseline:

    corrected = table[n] + k * (sum(seq_lens) - num_reqs * 8192)

still one scalar and one subtraction. The manager already knows
`full_cudagraph`, so the regime gate needs no new plumbing.

**What this does NOT show.**
- Fitted on synthetic `_dummy_run` batches with uniform context per batch, on
  one model and one GPU. Heterogeneous per-request contexts are untested.
- `floor` is taken from measurement rather than modelled; a deployed version
  must derive it from the existing profiled table.
- No end-to-end serving benefit is demonstrated — only prediction accuracy.
  Whether better predictions produce better throughput is the oracle-bound
  question, still open.

---

## E7 — does the mispricing actually change the decision?

`bench/decision_impact.py` (simulation, no GPU)

**Question.** E1–E6 measured prediction *error*. That is not the same as a
worse *decision*, and the distinction had been glossed over. Budget selection
is `argmax_B accepted(B) / cost(B)`, and **the argmax of a ratio is invariant
to any purely multiplicative error**: scaling every candidate's cost by 1.73
changes nothing at all. So "vLLM overpredicts by 73%" does not by itself imply
a wrong choice.

**What the error actually looks like.** The context mispricing is not a scale
factor but a **constant offset**, the same for every candidate budget:

    offset = k * (num_reqs * 8192 - sum(seq_lens))

An offset *does* move the argmax, and in the opposite direction to intuition:
inflating the fixed part of the cost makes marginal draft tokens look
relatively cheap, so vLLM **over**-commits verification.

**Result.**

     reqs    ctx  offset ms  true B  vLLM B  rate lost
        8    256      +2.26      23      24      0.12%
       16    256      +4.52      44      48      0.54%
       32    256      +9.04      67      96      3.48%
       64    256     +18.08      76     191     11.29%

The loss scales with `num_reqs * (profiled_ctx - real_ctx)`, since that
product is the size of the offset. **Negligible for small batches, material
for large ones** — and large batches with short contexts are exactly the
high-throughput serving regime.

**This corrects E1.** That entry claimed the mispricing makes vLLM
*under*-allocate. The sign is wrong, and the claim was never tested. The
direction is over-allocation, and the magnitude depends on batch size in a way
the original framing did not capture.

**Sensitivity — and a large correction to the numbers above.** The table
above used an *assumed* marginal cost of 0.02 ms per verified token. A
sensitivity sweep shows the result is almost entirely determined by that one
invented parameter:

    varying k        (4x range):   7.87%  ->  12.49%     mild
    varying marginal (20x range):  0.30%  ->  60.97%     dominates

The marginal cost is measurable from data already collected: the slope of cost
against token count at zero context is **0.00716 ms/token** in the graphed
sweep and 0.00931 in the eager one. Re-running with 0.007:

     reqs    ctx  true B  vLLM B  rate lost
       16    256      48      48      0.01%
       32    256      94      96      0.12%
       64    256     167     192      0.95%

**Under 1% everywhere, against the 11.29% reported above.** The headline figure
was an artifact of an invented parameter and must not be quoted. The measured
value says the mispricing barely moves the decision.

**What this means.** Three readings remain open, and the simulation cannot
separate them:

1. The effect really is small, and the honest outcome is a measurement study
   characterising when cost models matter rather than a scheduler improvement.
2. The simulation's structure is wrong — marginal cost may not be linear in B,
   and the slope measured here conflates *adding a request* with *adding a
   draft token to an existing request*, which is the quantity the budget
   decision actually turns on and is probably cheaper still.
3. Real workloads differ from the assumed survival distribution or carry
   heterogeneous contexts within a batch, neither of which is modelled.

**This makes the real oracle bound more important, not less.** Replaying logged
decisions against measured step times removes the survival assumption *and* the
marginal-cost assumption at once, and it is the only thing that can decide
between the three readings above. `bench/step_trace.py` logs what it needs.

---

## E8 — the sign-flip prediction, confirmed out-of-sample

`pace/run_all.sbatch` (experiment B) → `results/h100_ctx_straddle.json`

**The prediction.** vLLM profiles its cost table assuming 8192 tokens of
context per request. Every context measured up to this point was at or below
8192, so every observed error was over-prediction. The model says the sign must
**flip** above 8192 — and under-prediction is the damaging direction, since it
makes the scheduler commit to work it cannot afford.

**Confirmed exactly.** Contexts 1024 to 32,000, on an H100 with FA3:

     n    ctx   pred_ms  actual_ms   rel_err
     1   1024     2.272      2.002    -11.9%
     1  32000     2.272      3.158    +39.0%
    16   1024     6.806      2.644    -61.2%
    16   8192     6.806      6.806     +0.0%   <-- the profiling point
    16  16384     6.806     11.577    +70.1%
    16  32000     6.806     20.628   **+203.1%**

The sign changes at exactly 8192 and the magnitude scales with
`num_reqs * |ctx - 8192|`, as the model requires. Under-prediction reaches
**+203%**: vLLM believes a 20.6 ms step will take 6.8 ms.

**The model predicts the magnitudes, not just the direction.** Applying the
correction with `k = 0.0356 us/KV-token` — **fitted on entirely different data**
(`h100_e1`/`h100_e5`, a different `max_num_seqs`, contexts capped at 4096) and
used here unchanged:

    corrected = table[n] + k * num_reqs * (ctx - 8192)

                    median |error|    max |error|
    two-term model       0.8%            3.0%
    stock vLLM          33.9%          203.1%

**A 42x reduction in median error on out-of-sample data**, over a context range
four times beyond where the constant was fitted. This is the project's
strongest evidence: a prediction made in advance, tested on new measurements,
confirmed in both sign and magnitude.

**What this does NOT show.** Still synthetic `_dummy_run` batches with a single
context per batch, one model, one GPU, and all cells inside the CUDA-graph
capture range. It measures prediction accuracy, not serving benefit; whether
better predictions produce better scheduling remains the open question.

---

## E9 — first serving A/B and saturation sweep (A/B superseded)

`bench/serving_ab.py` → `results/h100_serving_ab.json`,
`results/h100_serving_ab_spec7.json`

`bench/step_trace.py` → `results/h100_step_trace.json`,
`results/sat_spec{3,7}_seqs{16,64}.json`

**Question.** Every result through E8 concerns prediction accuracy. Does the
error change the adaptive manager's chosen budget, and does that change improve
end-to-end throughput?

**The valid result from this stage is saturation.** Across all four stock
traces — speculative block 3 or 7, concurrency 16 or 64 — every recorded
decode step selected the full available draft budget:

    configuration       steps at ceiling
    spec 3, 16 seqs          66 / 66
    spec 3, 64 seqs          19 / 19
    spec 7, 16 seqs          29 / 29
    spec 7, 64 seqs           6 / 6

The one-dimensional stock table overestimates these short-context verification
steps, inflating the fixed part of the denominator and making every marginal
draft token look cheap. The manager therefore remains at the ceiling over a
meaningful range of block sizes and concurrency levels.

**The throughput numbers from this stage are invalid and must not be quoted as
oracle results.** They reported `+0.68%` for spec 3 and `-0.25%` for spec 7,
but both used `k = 0.0356 us/KV-token`, measured for Qwen3-1.7B, while serving
MiniCPM5-2B. They also treated the nominal 8,192-token profile context as the
real baseline. With `max_model_len=4096`, vLLM actually caps the installed
dummy context to roughly 4,096 minus the query length. Those two errors made
the correction too large, allowed corrected table entries to hit the positive
clamp, and invalidated the comparison. E10 replaces it with an in-engine,
model-specific calibration and an exact reconstruction of every profile point.

**Methodological result retained from E9.** A serving comparison must be
paired. Alternating stock/corrected order within a round removes shared clock
and thermal drift; comparing unpaired overall spreads can hide a sub-percent
effect.

---

## E10 — calibrated MiniCPM/H100 spec-7 serving A/B

`bench/serving_ab.py` → `results/h100_calibrated_ab_spec7.json`

In-engine calibration → `results/minicpm_h100_dspark_k_spec7.json`

**Method.** One live MiniCPM5-2B + DSpark engine first measured context slopes
after all CUDA graphs were captured, then froze that fitted `k` and ran the
stock/corrected A/B without changing engine or execution regime. The correction
reconstructed the aggregate sequence length behind each vLLM profile point,
including max-model-length capping, query tokens, graph padding, and eager-tail
interpolation. Guards required full CUDA graphs and exact calibration shapes,
equal output counts, no selected-cost clamp, a full-concurrency workload, and
20 counterbalanced AB/BA pairs after two warmup pairs.

Here **stock** means the unmodified vLLM/DSpark cost table and budget policy;
it does not mean speculation is disabled. **Corrected** changes only the target
verification-cost table seen by that same policy:

    corrected_cost[t] = stock_cost[t]
                      + k * (real_sum_seq_lens - profiled_sum_seq_lens[t])

**Calibration result.** MiniCPM's fitted value is
`k = 0.013702 us/KV-token`, not Qwen's `0.0356`:

    target tokens       64       128       256       384       512
    k (us/KV-token)  0.01353   0.01356   0.01368   0.01383   0.01391

The per-shape spread is **2.78%** and fixed-effect RMSE is **0.038 ms** across
contexts 0, 1,024, 2,048, 3,072, and the effective 4,088-4,095 limit. All
samples were full CUDA graphs. This validates a model-specific scalar context
term across the complete spec-7 verification range of 64-512 target tokens.

**Throughput result: the accurate correction is slightly slower.** Every mode
generated exactly 8,192 tokens in every round.

    mode          median tok/s
    stock              10,971.9
    corrected          10,923.5

    paired mean delta       -0.3649%
    paired median delta     -0.38%
    standard error           0.1091%
    approximate 95% CI      [-0.59%, -0.14%]
    corrected faster         6 / 20 rounds

Both AB-order and BA-order subsets were negative (`-0.426%` and `-0.304%`),
so the result is not an order artifact. No selected corrected cost was clamped.

**Mechanism: cheaper verification caused less drafting and more target
invocations.** Prefill/mixed steps were identical: 140 decisions per mode and
100% at the feasible ceiling. Decode is where the policies diverged:

    decode metric                 stock       corrected
    decisions                       347             360
    at feasible ceiling          97.98%          89.17%
    off-ceiling decisions             7              39
    draft shortfall                 8 x7     8 x20, 16 x19

The serving batch carried a median aggregate decode context of about 73,000
tokens, while the stock table point represented 262,144. The correction's
median selected offset was `-2.585 ms`, reducing the typical verification
estimate from `8.622 ms` to `6.030 ms`. Once target verification looked cheap,
the optimizer sometimes decided that 8 or 16 additional draft candidates were
not worth generating. Fewer candidates were accepted before completion, so the
corrected runs needed **13 additional decode decisions across 20 rounds** for
the same output. Their cost outweighed the saved drafting work.

**Interpretation.** The cost-model diagnosis is correct, the scalar correction
is accurate, and the corrected estimate changes real scheduling decisions. In
this workload those changes hurt throughput slightly. Stock's context error
accidentally keeps the manager near the maximum draft budget, which is better
for this batch. The remaining mismatch is therefore in the downstream
objective — it does not fully price the extra iterations and other end-to-end
overheads induced by trimming drafts — rather than in the context-cost fit.

**Workload limitation discovered after the run.** The CLI listed prompt lengths
64, 512, 1,024, 2,048, and 3,000, but the harness constructed a text string,
truncated its token IDs only if it was long enough, decoded it, and let vLLM
tokenize the string again. The requested lengths were upper bounds, not
verified lengths. At full-batch decode the recorded aggregate context was
71,190-75,924 tokens, or roughly 1,112-1,186 per active request. If all
configured input lengths had been exact, the prompts alone would total 83,424
tokens (1,303.5/request) before generation. The A/B remains valid for the
workload actually executed, but it is not an exact input-length sweep.

**Scoped conclusion.** E10 is conclusive for this simultaneous-arrival,
64-request, approximately 1.1K-average active-context, 128-output-token,
MiniCPM/H100/spec-7 workload. It does not show that corrected scheduling is
universally slower. As context approaches the effective 4K profile point, the
negative offset shrinks toward zero; longer generation also changes how much
time is spent at full concurrency versus draining the tail.

**Remaining experiment.** `pace/run_serving_context_sweep.sbatch` now runs the
same calibrated spec-7 A/B on homogeneous **exact token-ID** inputs of 64, 512,
1,024, 2,048, and 3,000 tokens, with 64 requests and exactly 128 generated
tokens each. It calibrates once, warms every workload before measurement,
counterbalances both context and mode order, verifies input/output lengths on
every round, and rejects any selected-cost clamp. This is the one experiment
needed before generalizing the throughput result across context length. It has
not yet been run.

---

## E10 — the calibrated A/B: accurate costs make serving slightly *worse*

`bench/serving_ab.py --calibrate-k` → `results/h100_calibrated_ab_spec7.json`
Run and analysed by Aditya; supersedes the A/B reported in E9.

**Two corrections to E9.** First, E9's A/B used `k = 0.0356 us/KV-token`, which
was measured on **Qwen3-1.7B** and applied to MiniCPM5-2B. `k` is KV bytes per
token over bandwidth and is therefore model-specific: calibrated in-engine for
MiniCPM it is **0.0137**, 2.6x smaller. E9 was over-correcting. Second, E9 used
six rounds at `spec=3`; this uses **twenty paired rounds at `spec=7`**, the
block size every DSpark checkpoint declares.

**Result: the correction is 0.365% SLOWER, and that is significant.**

    mean   -0.365%      corrected faster in  6/20 rounds
    median -0.379%      3.35 standard errors from zero

    stock median throughput      10,971.9 tok/s
    corrected median throughput  10,923.5 tok/s

**The decisions really do change** — ceiling selection falls from 97.98% to
89.17% — so this is not a null from the correction having no effect. It has an
effect, in the wrong direction.

**Why: the objective is greedy across steps.** With 64 requests at `spec=7` a
full step is 64 decode positions plus 448 draft candidates. Stock overestimated
a typical verification step at 8.622 ms against a corrected 6.030 ms, a median
adjustment of **-2.585 ms**. Believing verification expensive, the stock policy
kept nearly every draft token to amortise the target pass. Believing it cheaper,
the corrected policy trimmed 8 or 16 candidates — saving drafter work but
leaving some requests unfinished, which produced **13 additional decode batches
across 20 paired rounds** (347 → 360).

`argmax(accepted / cost)` maximises accepted tokens per millisecond **for the
current step only**. It does not price the future cost of needing another
decode iteration. So a more accurate cost feeds a myopic objective and yields a
worse schedule.

**The project's central result, then:** vLLM's verification cost model is wrong
by up to 203%, one scalar makes it 42x more accurate out-of-sample, and
deploying that accuracy makes serving **0.37% slower** — because the decision it
feeds is saturated at its ceiling 98% of the time, and the existing error
happens to bias toward that ceiling, which is nearly the right answer.

This is not "accurate costs are bad". It is that **prediction quality and
decision quality are separate problems**, and improving the first without
fixing the objective can move the second backwards.

**Saturation is not an artifact of `spec=3`.** All four configurations in the
saturation sweep — `spec` 3 and 7 crossed with `max_num_seqs` 16 and 64 — sat
at the ceiling on every recorded step.

**Method notes** (all from the calibrated harness):
- `k` is calibrated inside the same live DSpark engine the A/B runs in, not
  imported from another model or another run.
- The profile baseline is reconstructed properly, including max-length capping,
  query tokens, CUDA-graph padding and eager-tail interpolation.
- Stock/corrected order alternates within paired rounds; rounds require equal
  output-token counts; selections that hit the minimum positive-cost clamp are
  rejected.

**What remains untested.** Every A/B so far ran with contexts *below* the 8192
profile point, where stock overpredicts. Above it the sign flips and the
correction pushes the budget *up* instead of down — the opposite intervention.
`pace/run_ab_longctx.sbatch` tests that direction.

---
