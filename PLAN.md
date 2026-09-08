# Speculative decoding: does it stop helping as batch size grows?

## The claim under test

Speculative decoding gives a large speedup at batch size 1, a shrinking one as
batch size grows, and can eventually cost more than it saves.

The mechanism it should follow from:

- At **batch 1**, decoding is **memory-bandwidth bound**. The GPU spends most of
  a decode step waiting on weights streaming from HBM, and the arithmetic units
  idle. Verifying k+1 candidate tokens in one pass costs barely more than
  decoding 1, because the weights are read once either way. Speculation is close
  to free, so any accepted token is profit.
- At **large batch**, the same weight read is amortised over many sequences and
  the step becomes **compute bound**. Now the k+1 verification tokens are real
  extra FLOPs competing with useful work, the draft model's own forward passes
  cost real time, and every rejected token is wasted compute. The speedup decays
  and can go below 1.0.

The measurement therefore has to separate two things that both vary with batch
size: how *often* drafts are accepted (a model property) and what a verification
step *costs* (a systems property). The first should be roughly flat across batch
size. If it is, then any decay in speedup is systems, not quality -- which is the
whole point.

## Hardware reality on this machine

RTX 4060 Laptop, 8 GB, of which ~6.93 GiB is actually free (WSL and the Windows
compositor hold the rest). That is the binding constraint on the entire design:
it sets the model pair, the sequence lengths, and how far up the batch axis we
can go before the KV cache runs out.

## Phases

### Phase 0 -- environment (done)
vLLM 0.28.0 on WSL2. See `SETUP.md`; it is not a normal install.

### Phase 1 -- sizing and a shared KV budget (`bench/phase1_sizing.py`)
Measure the memory breakdown of the baseline and of each k, then pin one KV
budget all configs share. See "The confound" below -- this is the step that
makes the rest of the experiment trustworthy.

### Phase 2 -- equivalence gate (done, PASSES)

`bench/phase2_correctness.py`. At temperature 0 the verification rule guarantees
the accepted sequence is what the target would have produced alone -- **in exact
arithmetic**. In floating point that does *not* imply token-identical output,
and the first version of this gate tested the wrong thing and reported a false
failure.

Why identity fails: the target scores k+1 candidate positions in ONE batched
forward pass, whereas plain decoding scores one position per pass. Different
batch shapes select different kernels and reduction orders, so logits differ in
the last bits. At a near-tie that flips the argmax, and everything after it
differs too.

Measured (`results/phase2_correctness.json`):

| k | token-identical | tie-break divergences |
|---|---|---|
| 1 | 5/6 | 1 |
| 3 | 5/6 | 1 |
| 5 | 1/6 | 5 |

Every divergence sat at a top-2 logprob gap of **exactly 0.0000 or 0.1250 nats**.
0.125 = 2^-3 is one bf16 quantisation step at these magnitudes, so the two
candidates were either bit-identical or adjacent representable values. And the
count rises with k exactly as the mechanism predicts: more verified positions
per pass means more chances to land on a tie.

**The corrected gate:** every divergence must occur at a numerical tie
(top-2 gap <= 0.15 nats). A divergence where the model was confident is a real
bug and blocks the sweep. This passes.

Supporting controls, all in `results/`:
- `determinism_baseline.json`, `determinism_spec-k3.json` -- each config
  reproduces itself exactly across separate processes, so the divergences are
  not run-to-run noise.
- `phase2_investigate.json` -- baseline vs baseline, and baseline vs baseline
  with logprobs requested, are identical; confirms that scoring the baseline
  does not perturb it, and that the divergence position (prompt 0, token 72) is
  stable.

One methodological note recorded because it cost time: an early diagnostic
compared a baseline run *with* logprobs against a spec run *without*, and its
apparent instability was an artifact of that mismatch, not of the engine.
Compare like with like, in one invocation.

### Phase 3 -- instrumentation (done)

`bench/phase3_metrics.py` + `bench/spec_probe.py`. vLLM tracks speculative
counters per scheduler step in `SpecDecodingStats`
(`vllm/v1/spec_decode/metrics.py`): `num_drafts`, `num_draft_tokens`,
`num_accepted_tokens`, and per-draft-position accepted/drafted vectors. Those
reach the frontend inside `SchedulerStats`, where stat loggers see them. The
offline `LLM` API exposes no hook for custom loggers, so we attach one to
`llm_engine.logger_manager` after construction; that needs
`disable_log_stats=False`, since `LLM()` otherwise sets it True and
`logger_manager` is None. Reading counters beats parsing log lines.

Measured at 12 requests, 128 tokens each, shared KV budget
(`results/phase3_metrics.json`):

| config | accept rate | accept length | tok/s | per-position acceptance |
|---|---|---|---|---|
| draft k=1 | 0.766 | 1.766 | 374.5 | 0.766 |
| draft k=3 | 0.636 | 2.909 | 412.3 | 0.764, 0.644, 0.501 |
| draft k=5 | 0.526 | 3.630 | 375.0 | 0.808, 0.605, 0.488, 0.395, 0.335 |
| ngram k=1 | 0.417 | 1.417 | 304.7 | 0.417 |
| ngram k=3 | 0.183 | 1.545 | 321.0 | 0.345, 0.145, 0.055 |
| ngram k=5 | 0.166 | 1.817 | 306.0 | 0.365, 0.191, 0.096, 0.096, 0.070 |

**Acceptance decays geometrically, as the model predicts.** The conditional
probability of accepting position i given i-1 was accepted is nearly constant
for the draft model -- 0.808, 0.749, 0.806, 0.810, 0.847 at k=5 -- i.e. each
additional draft token survives with roughly the same ~0.8 chance. The marginal
rate therefore falls off as 0.8^i, which is why acceptance *length* keeps rising
with k (1.77 -> 2.91 -> 3.63) while acceptance *rate* falls (0.766 -> 0.636 ->
0.526). Those are not in conflict: deeper drafts win more tokens per round but
waste a larger fraction of what they propose.

**n-gram speculation is much weaker here** -- roughly 0.4 conditional acceptance
against the draft model's 0.8, giving an acceptance length of only 1.4-1.8. It
costs no GPU memory, but on this workload it proposes far worse continuations.
Its k=5 per-position numbers are noisy (one position shows 1.000) because very
few drafts reach that depth.

**Acceptance length is the ceiling, not the outcome.** k=5 has the highest
ceiling (3.63x) yet lower throughput than k=3 at this batch size. The gap
between ceiling and realised speedup is exactly the overhead Phase 4 measures.

#### The flatness control, and a methodological correction

Acceptance is a property of the model pair, so it must not move with batch size.
A first version of this control varied batch size by changing the NUMBER OF
PROMPTS -- so batch 1 ran only prompt 0 while batch 8 ran all six. That measured
a difference in prompt content and produced a spurious "acceptance is not flat"
result (0.497 vs 0.636).

Batch size must be varied with **`max_num_seqs`**, holding the submitted
workload fixed, so that only concurrency changes. Corrected:

| max_num_seqs | acceptance rate | acceptance length | tok/s |
|---|---|---|---|
| 1 | 0.6362 | 2.909 | 95.9 |
| 2 | 0.6344 | 2.903 | 152.7 |
| 8 | 0.6457 | 2.937 | 295.9 |

Acceptance moves by 0.009 across an 8x change in concurrency while throughput
triples. **This is the control the whole experiment rests on**: any decay in
speedup that Phase 4 finds cannot be attributed to the draft model guessing
worse under load, because it does not.

Phase 4 must use `max_num_seqs` for its batch axis for the same reason.

### Phase 4 -- the batch sweep (done)

`bench/phase4_sweep.py`. 48 requests x 128 tokens per point, `ignore_eos` so
token counts are identical, every run pinned to the Phase 1 shared KV budget,
batch axis driven by `max_num_seqs` with the workload held fixed, and configs
interleaved within each batch size so thermal throttling cannot masquerade as a
trend. Results in `results/phase4_sweep.json`.

| batch | baseline tok/s | draft k=1 | draft k=3 | draft k=5 | ngram k=3 |
|---|---|---|---|---|---|
| 1 | 75.2 | 1.242x | **1.327x** | 1.143x | 0.909x |
| 2 | 147.7 | 1.191x | 1.287x | 1.099x | 0.887x |
| 4 | 291.4 | 1.140x | 1.155x | 1.067x | 0.844x |
| 8 | 574.6 | 0.940x | 1.084x | 0.890x | 0.812x |
| 16 | 1088.5 | 0.858x | 0.817x | 0.703x | 0.753x |
| 32 | 1592.2 | 0.778x | 0.657x | 0.557x | 0.691x |
| 48 | 2721.9 | 0.672x | **0.533x** | 0.392x | 0.612x |

**The phenomenon reproduces.** Speedup decays monotonically with batch size for
every method and crosses below 1.0 -- speculation stops helping and starts
hurting.

Crossover (first batch size where speedup <= 1.0):

| config | crossover |
|---|---|
| draft k=1 | batch 8 |
| draft k=3 | **batch 16** |
| draft k=5 | batch 8 |
| ngram k=3 | never helps, even at batch 1 |

### Phase 5 -- why the curve has this shape

**The ceiling does not move; the realised fraction of it collapses.** Acceptance
length -- the theoretical speedup if a verify step cost the same as a decode
step -- is essentially constant across the sweep (k=3: 2.909 at batch 1, 2.978
at batch 48). What changes is how much of it is realised:

| batch | k=1 | k=3 | k=5 |
|---|---|---|---|
| 1 | 0.703 | 0.456 | 0.327 |
| 8 | 0.529 | 0.371 | 0.238 |
| 48 | 0.380 | 0.179 | 0.109 |

(efficiency = realised speedup / acceptance length)

At batch 1, k=3 converts 46% of its ceiling into real throughput. At batch 48,
18%. The draft model is guessing just as well -- it is simply that verification
is no longer close to free.

**The acceptance control holds.** Across an 48x change in concurrency,
acceptance moves by 0.039 (k=1), 0.043 (k=3), 0.056 (k=5) -- flat. So the decay
is *not* the draft model doing worse under load. It is the cost side: at batch 1
the GPU is memory-bandwidth bound and idle while weights stream, so verifying
k+1 candidates is nearly free; at batch 48 the weight read is amortised across
many sequences, the step is compute bound, and the extra verification tokens,
the draft model's own forward passes, and every rejected token are real FLOPs
competing with useful work.

**Deeper drafts cross over sooner.** k=5 has the highest ceiling (3.6x) and the
worst curve -- 0.392x at batch 48 against k=3's 0.533x. More speculation means
more wasted compute per rejected token, and the waste grows with k while the
gain saturates. k=3 is the best choice at small batch and k=1 degrades most
gracefully at large batch; k=5 is never the right answer here.

**n-gram never pays off on this workload.** Its acceptance (~0.2-0.35) is too
low to cover even its small cost, so it is below 1.0 everywhere.

#### Caveats and loose ends

- **n-gram acceptance is not flat** (spread 0.156: 0.202 at batch 1 rising to
  0.350 at batch 48) while the draft-model configs are flat. The submitted
  workload is identical at every point, so concurrency alone should not change
  what an n-gram proposer finds in a request's own history. This is unexplained
  and worth investigating before drawing any conclusion about n-gram.
- Baseline throughput is still rising steeply at batch 48 (1592 -> 2722 tok/s
  from 32 to 48), so the GPU is not yet saturated at the top of the sweep. The
  KV budget, not the compute, is what caps the sweep here.
- Single repeat per point (`--repeats 1`). The trend is far larger than any
  plausible run-to-run noise, but error bars would need repeats.
- One target/draft pair, one workload, 128-token outputs, one GPU. The shape of
  the curve should generalise; the exact crossover batch will not.
