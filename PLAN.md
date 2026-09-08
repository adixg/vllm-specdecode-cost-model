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

### Phase 4 -- the batch sweep (not yet written)
Fixed workload, identical KV budget, sweeping batch size until the cache runs
out. Baseline vs k in {1,3,5}, plus n-gram speculation as a second method that
needs no draft model.

Controls that matter:
- **Identical token counts.** Force every request to emit exactly the same
  number of tokens (`ignore_eos` + `min_tokens`), or batch composition changes
  between configs and throughput is not comparable.
- **Interleave runs.** This is a laptop GPU and it thermally throttles. Run
  baseline and spec alternately rather than all of one then all of the other,
  so drift does not get baked into the comparison.
- **Never change two things at once.** Attention backend, dtype, and
  `max_model_len` stay fixed for the entire sweep.

### Phase 5 -- analysis
Speedup vs batch size per k and per method; acceptance rate vs batch size (the
control -- it should be flat); per-position acceptance decay. Headline number is
the **crossover point** where speedup crosses 1.0, with the roofline argument for
why it lands there.

## The confound Phase 1 exists to kill

Measured on this machine, `gpu_memory_utilization=0.85`, `max_model_len=512`
(`results/phase1_sizing.json`, reproduce with `python bench/phase1_sizing.py`):

| config | KV tokens | KV GiB | weights GiB | peak act GiB | graphs GiB |
|---|---|---|---|---|---|
| baseline | 99,664 | 2.66 | 3.65 | 0.48 | 0.47 |
| spec k=1 | 28,128 | 1.07 | 4.06 | 1.66 | 0.56 |
| spec k=3 | 28,224 | 1.08 | 4.07 | 1.65 | 0.55 |
| spec k=5 | 28,480 | 1.09 | 4.07 | 1.64 | 0.54 |

Same settings, and speculative decoding gets **3.5x less KV cache**. vLLM sizes
the cache from what is left after weights, activation and graphs, so the config
with the bigger activation footprint is silently starved. Left uncontrolled,
this alone would produce the result we are trying to test -- spec decode falling
off at high batch -- for entirely the wrong reason.

Two things worth noting in that table:

- The driver is **not** the draft weights (+0.41 GiB) but **peak activation,
  0.48 -> 1.65 GiB**, because verification pushes k+1 tokens per sequence
  through the target in a single pass.
- **The cost is flat in k.** Activation is ~1.65 GiB whether k is 1, 3 or 5.
  Enabling speculation at all is what costs the memory; the depth of the draft
  is nearly free. So a single shared budget covers every k, and k does not need
  its own envelope.

### The shared budget

    kv_cache_memory_bytes = 536519168   # 0.50 GiB

the minimum over all four configs. Every run in Phase 2 onward passes this.

**Equal bytes is not equal tokens**, and that is deliberate. The same budget buys
~18,700 tokens for the baseline but only ~13,100 for spec decode, because the
draft model needs its own KV cache out of the same allocation. That asymmetry is
a genuine cost of the method and should stay visible; equalising *tokens* would
hide it. The constraint it imposes is on the sweep: batch x sequence length must
stay under the smaller (spec) capacity, or spec decode starts preempting while
the baseline does not -- reintroducing the confound from the other direction.

At 256 tokens per sequence that ceiling is ~51 concurrent sequences, so the
sweep runs to **batch 48**.

## Consequence: short sequences

Pinning to the worst case means a small shared cache, so long sequences would
cap the batch sweep around 8 -- below where the crossover is expected. The
experiment therefore uses **short sequences (128-token prompt, 128-token
output)** to buy concurrency. This is legitimate: the phenomenon is about decode
step economics, not context length. Testing long context would need a smaller
target model instead, and is a separate experiment.
