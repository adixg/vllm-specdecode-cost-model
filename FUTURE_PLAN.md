# Future plan

Where this project goes after E13. Written 2026-10-09.

## Where E13 left things

The project began as "the verification cost model is inaccurate" and ended as
something more specific:

> vLLM's adaptive verification budget policy is miscalibrated in a specific
> concurrency band. At 64–128 concurrent requests it trims draft tokens that
> were worth keeping, costing 1–3% throughput against a policy that ignores the
> cost model entirely. Above 257 requests a fixed logit-buffer cap
> (`rejection_sampler.py:35`) takes the decision away from it.

The cost model `cost ≈ f(num_target_tokens) + k · Σ seq_lens` is still the
measurement tool. Its central parameter `k` is the **cost of reading the KV
cache**. The next phase follows that parameter to where vLLM is heading: the
Q3 2026 roadmap ([vllm#48168](https://github.com/vllm-project/vllm/issues/48168))
asks SIG Spec Decode to reduce verification overhead under high concurrency and
to consider enabling speculation by default. It also asks SIG Quantization to
make quantized KV caches production-grade, and SIG Core to redesign the KV cache
manager and scheduler. A quantized KV cache changes `k` directly, so it changes
every decision this repo has measured.

## Phase 0: publish what already exists (before any new GPU time)

1. **Upstream issue for E13.** File a vLLM issue with the concurrency-band
   result, the paired-round tables, and the logit-cap arithmetic
   (`7n > 2056 − n` ⇒ the cap binds above 257 requests on MiniCPM5-2B). This is
   the most actionable finding in the repo, and it is not yet anywhere vLLM
   maintainers will see it.
2. **RFC draft: "Context- and dtype-aware verification cost for adaptive /
   default-on speculation".** Frame E4/E8 (accuracy) and E10–E13 (decision
   impact) as inputs to the roadmap's "enable speculation by default" item.
   The argument: deciding *when* speculation pays off needs a cost model that
   knows context length and KV dtype, and the budget objective needs to price
   the risk of an extra decode iteration (E10's 13 extra batches).
3. **Present at SIG Spec Decode.** The meeting notes and calendar are linked
   from vllm#48168. A 5-minute summary of E10–E13 is a good fit.
4. **Write-up.** A short post on adixg.github.io: "Better measurement does not
   guarantee a better policy", using the E10/E12/E13 figures.

## Phase 1: does a quantized KV cache move `k`, and the decision?

### E14: `k` under FP8 KV cache (H100, then H200)

**Question.** How does `kv_cache_dtype=fp8_e4m3` change the KV-read cost `k`
in the CUDA-graph regime?

**Hypothesis (stated before running).** In the graphed regime `k` is dominated
by KV bytes read, so `k_fp8 ≈ 0.5 · k_bf16`, plus a dequantisation term that
keeps it above exactly half. In the eager regime (where E6 found context only
moves cost by about 1%) the effect should be small.

**Design.**
- `bench/verify_cost_probe.py` gains `--kv-cache-dtype {auto,fp8_e4m3}`.
  Everything else stays unchanged: global warmup of every shape, interleaved
  and reversed order, graphed and eager reported separately.
- Same grid as E6/E8: `num_tokens` × `context_len` up to 32k, both models
  (Qwen3-1.7B, MiniCPM5-2B), so `k` is fitted per (model, dtype, GPU).
- Fit with `bench/fit_cost_model.py`. Report `k_fp8 / k_bf16`, held-out RMSE,
  and whether the additive form still wins (the E4 test).
- **Check first:** confirm the pinned vLLM version runs FP8 KV on the FA3 path
  on Hopper. If it silently falls back to another backend, the attention
  backend becomes a confounded variable. Log the selected backend for **both**
  target and draft (see E16).

**Also check.** Is vLLM's startup cost profile measured with the configured KV
dtype? If the profile is taken at one dtype and serving runs at another, that
is a second, independent mispricing, separate from the context one.

### E15: does cheaper verification move the miscalibrated band?

**Question.** Rerun E13 (stock vs always-ceiling, `max_num_seqs` 16/64/128/256,
ctx=64, 20 paired rounds) with FP8 KV.

**Prediction.** A smaller `k` makes wide verification steps cheaper, so trimming
should become *even less* justified. The band where ignoring the cost model wins
should widen or shift upward. If it doesn't, the policy's error is not mainly
about verification price, which is itself a useful negative result.

**Also measure.** Acceptance per draft under FP8 KV versus BF16 KV (the E13
`--collect-acceptance` path). A quantized cache could reduce acceptance, and that
cost belongs in the same objective.

### E16: draft and target backends and KV dtypes (measure, don't compete)

Several open issues show that, in a speculative setup, the **draft model selects
its attention backend independently** of `--attention-backend`, and that draft
and target can end up with different KV layouts and dtypes. See
[vllm#44879](https://github.com/vllm-project/vllm/issues/44879) (FP8 KV + MTP
crash on SM89) and
[vllm#55312](https://github.com/vllm-project/vllm/issues/55312) (disjoint KV
layouts). Fixes are already in progress there by other contributors, so the
contribution here is **cost data, not another fix**:
- Record the backend and KV dtype actually selected for draft and target in
  every run (from the engine log).
- Measure draft-side cost under mismatched and matched configurations, and post
  the numbers to those threads.

## Phase 2: hardware generalisation

The README's main limitation is "one H100 generation"; `k` is per
(model, GPU). Each row below is a calibration of `k` (E6/E8-style probe) plus,
where the path exists, one paired A/B.

| Hardware | Where | Notes |
|---|---|---|
| H100 (Hopper) | PACE ICE `ice-gpu` | Baseline; all existing results |
| H200 (Hopper, more HBM bandwidth) | PACE Phoenix | Predict a lower `k` from bandwidth alone. A clean test of "k is a memory-bandwidth cost". Needs a Phoenix allocation. |
| L40S (Ada, SM89) | PACE Phoenix | The FA3 adaptive-verification path is Hopper-only (the sbatch checks `compute_cap 9.x`), so probe-only. This is also the architecture of vllm#44879. |
| MI300X (AMD) | AMD Developer Cloud | AMD AI Developer Program members get $100 credit (≈50 MI300X hours; card on file required; expiry terms unconfirmed). Enough for `k` calibration on both models and one DSpark A/B (DSpark on ROCm is listed as done in the Q3 roadmap). Use AMD's ROCm vLLM Docker image. |

The output is a `k` table across GPUs and dtypes, a "how verification cost
scales on NVIDIA vs AMD" figure, and a test of whether `k` tracks KV bytes per
unit of memory bandwidth.

## Phase 3: the remaining limitations in the README

- **Heterogeneous batches.** Every A/B so far used homogeneous exact-length
  prompts. Mix context lengths within a step (e.g. 50% at 64, 50% at 3,000) and
  check whether the additive `k · Σ seq_lens` still predicts the step.
- **Continuous arrivals.** Replace closed-loop rounds with an open-loop
  arrival process (Poisson at several rates). The concurrency band in E13 should
  then appear as a band of arrival rates.
- **A better objective, not just a better price.** E10 and E12 show the
  argmax(accepted / cost) objective does not price the risk of an extra decode
  iteration. Prototype an objective that does: for example, penalise budgets
  that leave requests unfinished at a step boundary. A/B it against stock and
  always-ceiling across the E13 band. This is the experiment most likely to give
  a positive result.

## Order and rough budget

| Step | GPU time | Depends on |
|---|---|---|
| Phase 0 (issue, RFC, SIG, post) | none | — |
| E14 on H100 | ~4–6 h | `--kv-cache-dtype` flag |
| E15 on H100 | ~6–8 h (4 concurrency levels × 20 rounds) | E14 |
| E16 | folded into E14/E15 runs | logging only |
| H200 / L40S calibration | ~2–3 h each | Phoenix allocation |
| MI300X calibration + one A/B | ≤ 50 h credit | AMD Developer Program |
| Phase 3 | open-ended | E15 |

## Risks

- **Version churn.** vLLM's spec-decode and KV-cache code is moving fast (Model
  Runner V2, KV cache manager redesign). Pin the version for each experiment,
  record it in every results JSON, and rebase the probes deliberately rather
  than continuously.
- **Crowded area.** Other contributors are measuring FP8 KV × speculative
  decoding compatibility. What this repo adds is analysis: an explicit,
  calibrated cost model and its effect on the policy's decisions. Stay on that,
  not on compatibility matrices.
- **The cap.** Above ~257 concurrent requests (model-dependent) the logit-buffer
  cap, not the cost model, sets the budget. Any high-concurrency result must
  report whether the cap was binding, as E13 does.
