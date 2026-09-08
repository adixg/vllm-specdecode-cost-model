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

### Phase 3 -- instrumentation (not yet written)
Per run, extract from vLLM's spec-decode counters:
- **acceptance rate** = accepted draft tokens / drafted tokens
- **acceptance length** alpha = mean accepted tokens per verification step.
  This is the theoretical speedup ceiling.
- **per-position acceptance** -- how often draft token 1, 2, ... k survives.
  It decays roughly geometrically and tells us the useful value of k.

Cross-check: measured speedup should be about alpha x (cost of a verify step
relative to a plain decode step). Where those disagree, the gap *is* the
overhead we are trying to characterise.

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
