#!/usr/bin/env python3
"""Does a corrected cost model actually make serving faster?

Everything measured so far concerns prediction ACCURACY. Cost accuracy has
exactly one consumer in vLLM - the argmax in get_num_tokens (the table is read
only at adaptive_verification.py:305, and its result is used only at
model_runner.py:1084) - so a better estimate is worth nothing unless it changes
which budget gets chosen, and that change is worth something.

The correction coefficient is model-specific because KV bytes per token vary
by architecture. With --calibrate-k, this program measures it inside the SAME
DSpark engine used for the A/B, after CUDA graphs have been captured. This is
important: enabling DSpark changes the captured token range.

  stock      vLLM's table as shipped
  corrected  table[t] + k * (real sum(seq_lens) - profiled sum(seq_lens)[t])

The profiled context is reconstructed exactly as set_dummy_context does it:
min(configured_profile_context, max_model_len - max_query_len). Thus a nominal
8192-token profile on a model capped at 4096 is correctly treated as roughly
4096, not 8192.

Modes are paired inside one engine and their order reverses every round (ABBA),
so neither mode is always charged the within-round order effect.

With --exact-prompt-sweep, every configured prompt length becomes a separate
homogeneous workload. Prompts are submitted as token IDs (never decoded and
re-tokenized), every workload is warmed before any measurements, and context
order reverses between rounds to distribute machine drift across lengths.

    export VLLM_USE_V2_MODEL_RUNNER=1
    python bench/serving_ab.py --model openbmb/MiniCPM5-2B \
                               --draft openbmb/MiniCPM5-2B-DSpark \
                               --calibrate-k
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


def fit_k_from_rows(rows: list[dict]) -> tuple[float, dict[int, float], float]:
    """Fit one context slope with a separate intercept per token shape."""
    groups: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        n = int(row["num_tokens"])
        actual = row.get("median_forward_ms")
        if actual is None:
            continue
        # Use the context actually installed by set_dummy_context, not the
        # requested value, which may be capped by max_model_len - query_len.
        x = float(row["num_reqs"] * row["effective_context_len"])
        groups.setdefault(n, []).append((x, float(actual)))

    numerator = denominator = 0.0
    by_shape: dict[int, float] = {}
    for n, points in groups.items():
        if len(points) < 2:
            continue
        x_bar = statistics.mean(x for x, _ in points)
        y_bar = statistics.mean(y for _, y in points)
        num = sum((x - x_bar) * (y - y_bar) for x, y in points)
        den = sum((x - x_bar) ** 2 for x, _ in points)
        if den:
            by_shape[n] = num / den
            numerator += num
            denominator += den

    if not denominator:
        raise ValueError("calibration has no usable multi-context shapes")

    k = numerator / denominator
    squared_error = 0.0
    count = 0
    for points in groups.values():
        if len(points) < 2:
            continue
        x_bar = statistics.mean(x for x, _ in points)
        y_bar = statistics.mean(y for _, y in points)
        for x, y in points:
            predicted = y_bar + k * (x - x_bar)
            squared_error += (y - predicted) ** 2
            count += 1
    rmse = (squared_error / count) ** 0.5
    return k, by_shape, rmse


def _calibrate_on_worker(
    self: "Worker",
    token_counts: list[int],
    context_lens: list[int],
    replays: int,
    warmup: int,
) -> dict:
    """Measure context slopes inside the configured DSpark/CUDA-graph engine."""
    import statistics

    import torch

    runner = self.model_runner
    captured = list(runner.cudagraph_manager.captured_token_counts())
    missing = sorted(set(token_counts) - set(captured))
    if missing:
        return {
            "ok": False,
            "why": f"calibration token counts are not graphed: {missing}",
            "captured_token_counts": captured,
        }

    max_model_len = int(runner.max_model_len)
    max_num_reqs = int(runner.max_num_reqs)
    grid = []
    for n in token_counts:
        num_reqs = min(n, max_num_reqs)
        max_query_len = (n + num_reqs - 1) // num_reqs
        for requested_ctx in context_lens:
            effective_ctx = max(
                min(requested_ctx, max_model_len - max_query_len), 0
            )
            grid.append({
                "num_tokens": n,
                "num_reqs": num_reqs,
                "max_query_len": max_query_len,
                "requested_context_len": requested_ctx,
                "effective_context_len": effective_ctx,
                "forward_ms": [],
            })

    # Warm every shape before timing any shape. Different query lengths can
    # trigger distinct compilation/autotuning paths.
    for cell in grid:
        for _ in range(warmup):
            runner._dummy_run(
                num_tokens=cell["num_tokens"],
                context_len=cell["requested_context_len"],
            )
    torch.cuda.synchronize()

    # Interleave cells by replay, reversing the order every other replay so
    # residual clock/thermal drift is not assigned to one end of the grid.
    for replay in range(replays):
        cells = grid if replay % 2 == 0 else reversed(grid)
        for cell in cells:
            with runner.step_timing.collect() as timings:
                runner._dummy_run(
                    num_tokens=cell["num_tokens"],
                    context_len=cell["requested_context_len"],
                )
            if len(timings) != 1:
                return {
                    "ok": False,
                    "why": (
                        "expected one timing sample per dummy run, got "
                        f"{len(timings)} for n={cell['num_tokens']} "
                        f"ctx={cell['requested_context_len']}"
                    ),
                    "captured_token_counts": captured,
                }
            sample = timings[0]
            if (not sample.full_cudagraph
                    or sample.num_target_tokens != cell["num_tokens"]
                    or sample.num_reqs != cell["num_reqs"]):
                return {
                    "ok": False,
                    "why": (
                        "calibration step did not execute with the expected "
                        "full graph/batch shape: "
                        f"requested n={cell['num_tokens']} reqs={cell['num_reqs']}; "
                        f"observed n={sample.num_target_tokens} "
                        f"reqs={sample.num_reqs} full_graph={sample.full_cudagraph}"
                    ),
                    "captured_token_counts": captured,
                }
            cell["forward_ms"].append(float(sample.forward_ms))

    for cell in grid:
        cell["median_forward_ms"] = statistics.median(cell["forward_ms"])

    return {
        "ok": True,
        "captured_token_counts": captured,
        "max_model_len": max_model_len,
        "max_num_reqs": max_num_reqs,
        "rows": grid,
    }


def _install_correction(
    self: "Worker",
    k_ms_per_kv_token: float,
    configured_profile_ctx: int,
    overhead_ms: float = 0.0,
) -> dict:
    """Wrap the budget decision so the correction can be toggled per round.

    overhead_ms is a per-step cost the stock tables leave out (scheduling,
    input prep, sampling, CPU gaps between kernels). The decision rule picks
    argmax accepted(b) / cost(b). Scaling every cost by a constant would not
    change that argmax, but ADDING a constant does: a bigger fixed cost per
    step makes it worth keeping more drafts, because each step has to
    "pay for itself". The corrected arm adds overhead_ms to every cost.
    """
    import time

    import numpy as np

    runner = self.model_runner
    av = getattr(runner, "adaptive_verification", None)
    if av is None:
        return {"ok": False, "why": "adaptive verification is not active"}
    if av.cost_tables is None:
        return {"ok": False, "why": "adaptive verification has no cost tables"}

    max_model_len = int(runner.max_model_len)
    max_num_reqs = int(runner.max_num_reqs)
    _, verify_table = av.cost_tables

    captured = list(runner.cudagraph_manager.captured_token_counts())
    capture_limit = max(captured, default=0)

    # Reconstruct the token sizes profiled by batches_to_profile(). Besides
    # every captured size, it adds a 1.5x point and powers of two above the
    # graph limit through max_num_batched_tokens (the verify table's last idx).
    profile_sizes = list(captured)
    max_batch_tokens = len(verify_table) - 1
    if capture_limit:
        size = capture_limit
        tail_sizes = {min(size + size // 2, max_batch_tokens)}
        while size < max_batch_tokens:
            size = min(size * 2, max_batch_tokens)
            tail_sizes.add(size)
        profile_sizes.extend(sorted(tail_sizes - set(captured)))
    if not profile_sizes:
        return {"ok": False, "why": "no profiled token sizes found"}

    # Aggregate sequence lengths behind the raw profile points. This mirrors
    # _dummy_run plus set_dummy_context exactly: request count is capped,
    # context is capped by max_model_len - max_query_len, and query tokens are
    # included in sum(seq_lens).
    profile_sizes_np = np.asarray(profile_sizes, dtype=np.int64)
    profile_sum_seq_lens = []
    for num_tokens in profile_sizes:
        profile_reqs = min(num_tokens, max_num_reqs)
        max_query_len = (num_tokens + profile_reqs - 1) // profile_reqs
        effective_ctx = max(
            min(configured_profile_ctx, max_model_len - max_query_len), 0
        )
        profile_sum_seq_lens.append(
            float(profile_reqs * effective_ctx + num_tokens)
        )
    profile_sum_seq_lens_np = np.asarray(profile_sum_seq_lens)

    # Mirror build_cost_tables_from_curves. Graph-mode table entries snap to
    # the next captured size; eager entries interpolate between tail profiles.
    values = np.arange(len(verify_table))
    idx = np.searchsorted(profile_sizes_np, values, side="left")
    profiled_sum_seq_lens_table = profile_sum_seq_lens_np[
        np.minimum(idx, len(profile_sizes_np) - 1)
    ]
    if capture_limit:
        smooth = values > capture_limit
        above = profile_sizes_np > capture_limit
        if smooth.any() and above.any():
            profiled_sum_seq_lens_table[smooth] = np.interp(
                values[smooth],
                profile_sizes_np[above],
                profile_sum_seq_lens_np[above],
            )
    if len(profile_sizes_np) > 1:
        after = values > profile_sizes_np[-1]
        slope = (
            (profile_sum_seq_lens_np[-1] - profile_sum_seq_lens_np[-2])
            / (profile_sizes_np[-1] - profile_sizes_np[-2])
        )
        profiled_sum_seq_lens_table[after] = (
            profile_sum_seq_lens_np[-1]
            + slope * (values[after] - profile_sizes_np[-1])
        )

    self._ab_enabled = False          # flipped between rounds from the host
    self._ab_offsets = []             # for reporting how large the change was
    self._ab_decisions = []           # budget and clamp diagnostics
    self._ab_overhead_ms = float(overhead_ms)  # can be changed by _set_overhead
    self._ab_prev = None              # (time, decision) of the previous step
    original = av.get_num_tokens

    def corrected_get_num_tokens(num_tokens_per_req, draft_tokens):
        req_ids = list(num_tokens_per_req)
        slots = [av.req_states.req_id_to_index[r] for r in req_ids]
        real_ctx = int(av.req_states.num_computed_tokens_np[slots].sum())
        draft_t, verify_t = av.cost_tables

        chosen_offset = 0.0
        clamped_entries = 0
        # Apply a vector correction because graph padding means neighboring
        # candidate budgets may share one profiled table point. The real
        # aggregate sequence length is current context plus candidate query
        # tokens; the reconstructed profile includes both terms as well.
        # Computed in BOTH arms (the stock arm only uses it for the step-time
        # prediction below, never for the decision).
        candidate_tokens = np.arange(len(verify_t), dtype=np.float64)
        real_sum_seq_lens = real_ctx + candidate_tokens
        offsets = k_ms_per_kv_token * (
            real_sum_seq_lens
            - profiled_sum_seq_lens_table[:len(verify_t)]
        )
        if self._ab_enabled:
            shifted = verify_t + offsets
            clamped_entries = int(np.count_nonzero(shifted <= 0))
            # The overhead is the same for every candidate budget, so it is
            # added after the clamp check: it can only make costs larger.
            shifted = shifted + self._ab_overhead_ms
            av.cost_tables = (draft_t, np.maximum(shifted, 1e-6))

        try:
            num_tokens = original(num_tokens_per_req, draft_tokens)
            _, non_draft_per_req, draft_budget = av._batch_budget

            # Reproduce the manager's feasible ceiling, including its sampler
            # logit-chunk cap, rather than assuming spec_steps * num_reqs.
            available = int(sum(
                len(draft_tokens.get(req_id, ())) for req_id in req_ids
            ))
            max_draft_budget = min(
                available,
                max(0, int(av._max_total_logits)
                    - len(req_ids) * int(av.num_bonus_tokens)),
            )

            non_draft_total = int(sum(non_draft_per_req.values()))
            chosen_idx = non_draft_total + int(draft_budget)
            chosen_offset = float(offsets[chosen_idx]) if self._ab_enabled else 0.0
            if self._ab_enabled:
                self._ab_offsets.append(chosen_offset)
            chosen_was_clamped = bool(
                self._ab_enabled
                and chosen_idx < len(verify_t)
                and verify_t[chosen_idx] + chosen_offset <= 0
            )
            decision = {
                "corrected": bool(self._ab_enabled),
                "num_reqs": len(req_ids),
                "total_context": real_ctx,
                "non_draft_tokens": non_draft_total,
                "draft_budget": int(draft_budget),
                "available_draft_tokens": available,
                "max_draft_budget": max_draft_budget,
                "at_ceiling": int(draft_budget) == max_draft_budget,
                "chosen_target_tokens": chosen_idx,
                "stock_verify_ms": float(verify_t[chosen_idx]),
                "corrected_verify_ms": float(max(
                    verify_t[chosen_idx] + chosen_offset, 1e-6
                )),
                "real_sum_seq_lens": real_ctx + chosen_idx,
                "profiled_sum_seq_lens": float(
                    profiled_sum_seq_lens_table[chosen_idx]
                ),
                "offset_ms": chosen_offset,
                "clamped_entries": clamped_entries,
                "chosen_was_clamped": chosen_was_clamped,
                "overhead_ms": self._ab_overhead_ms if self._ab_enabled else 0.0,
                # Best available prediction of this step's cost: drafter plus
                # the k-CORRECTED verify cost. Using the stock verify cost here
                # would fold the stock table's context error into the overhead
                # (at short contexts stock overpredicts by ~3.5 ms, which would
                # hide the overhead we are trying to measure).
                "predicted_step_ms": float(
                    draft_t[len(req_ids)]
                    + max(verify_t[chosen_idx] + offsets[chosen_idx], 1e-6)
                ),
                # Real wall time until the next decision, i.e. how long this
                # step actually took. Filled in when the next step arrives;
                # stays None for the last step of a round.
                "step_interval_ms": None,
            }
            self._ab_decisions.append(decision)

            now = time.perf_counter()
            if self._ab_prev is not None:
                prev_time, prev_decision = self._ab_prev
                prev_decision["step_interval_ms"] = (now - prev_time) * 1000.0
            self._ab_prev = (now, decision)
            return num_tokens
        finally:
            if self._ab_enabled:
                av.cost_tables = (draft_t, verify_t)

    av.get_num_tokens = corrected_get_num_tokens
    return {
        "ok": True,
        "max_model_len": max_model_len,
        "max_num_reqs": max_num_reqs,
        "configured_profile_context_len": configured_profile_ctx,
        "captured_token_counts": captured,
        "profiled_token_counts": profile_sizes,
    }


def _set_mode(self: "Worker", enabled: bool) -> dict:
    self._ab_enabled = bool(enabled)
    n = len(getattr(self, "_ab_offsets", []))
    self._ab_offsets = []
    self._ab_decisions = []
    self._ab_prev = None   # never measure a "step" that spans two rounds
    return {"enabled": self._ab_enabled, "offsets_last_round": n}


def _set_overhead(self: "Worker", overhead_ms: float) -> float:
    self._ab_overhead_ms = float(overhead_ms)
    return self._ab_overhead_ms


def estimate_overhead_ms(decisions: list[dict]) -> tuple[float, int]:
    """Median of (real step time - predicted step time) over stock decode steps.

    Only stock steps are used, so the estimate does not depend on the
    correction it feeds. Only pure decode steps are used (one non-draft token
    per request), since prefill steps are priced differently. Returns the
    estimate, floored at zero, and how many steps it came from.
    """
    gaps = [
        d["step_interval_ms"] - d["predicted_step_ms"]
        for d in decisions
        if not d["corrected"]
        and d["step_interval_ms"] is not None
        and d["non_draft_tokens"] == d["num_reqs"]
    ]
    if not gaps:
        return 0.0, 0
    return max(0.0, statistics.median(gaps)), len(gaps)


def _drain_ab_trace(self: "Worker") -> dict:
    return {
        "offsets": list(getattr(self, "_ab_offsets", [])),
        "decisions": list(getattr(self, "_ab_decisions", [])),
    }


def _build_exact_token_prompts(tokenizer, num_prompts: int,
                               prompt_tokens: int) -> list[dict]:
    """Build varied, natural-ish prompts with exactly prompt_tokens IDs."""
    if prompt_tokens <= 0:
        raise ValueError("prompt length must be positive")

    filler_ids = tokenizer(
        " systems research inference serving scheduling latency throughput "
        "batching attention memory bandwidth verification",
        add_special_tokens=False,
    ).input_ids
    if not filler_ids:
        raise ValueError("tokenizer produced no filler token IDs")

    prompts = []
    for i in range(num_prompts):
        prefix_ids = tokenizer(
            f"Request {i}. Explain this systems result in detail:",
            add_special_tokens=True,
        ).input_ids
        if len(prefix_ids) > prompt_tokens:
            raise ValueError(
                f"prompt length {prompt_tokens} is shorter than the "
                f"{len(prefix_ids)}-token prefix for request {i}"
            )

        # Rotate the filler so requests are not token-identical, then repeat it
        # directly in token space. This guarantees the requested length without
        # a lossy decode/re-tokenize round trip.
        shift = i % len(filler_ids)
        rotated = filler_ids[shift:] + filler_ids[:shift]
        remaining = prompt_tokens - len(prefix_ids)
        repeats = (remaining + len(rotated) - 1) // len(rotated)
        token_ids = prefix_ids + (rotated * repeats)[:remaining]
        if len(token_ids) != prompt_tokens:
            raise AssertionError("exact prompt construction failed")
        prompts.append({"prompt_token_ids": token_ids})

    return prompts


def _decision_summary(decisions: list[dict]) -> dict:
    if not decisions:
        return {
            "count": 0,
            "ceiling_fraction": None,
            "median_draft_budget": None,
            "chosen_cost_clamped_fraction": None,
        }
    return {
        "count": len(decisions),
        "ceiling_fraction": (
            sum(d["at_ceiling"] for d in decisions) / len(decisions)
        ),
        "median_draft_budget": statistics.median(
            d["draft_budget"] for d in decisions
        ),
        "chosen_cost_clamped_fraction": (
            sum(d["chosen_was_clamped"] for d in decisions) / len(decisions)
        ),
    }


def _run_exact_prompt_sweep(llm, sampling_params_cls, args, profile_ctx: int,
                            setup: dict) -> int:
    """Run homogeneous exact-length workloads in one calibrated engine."""
    prompt_lengths = sorted(set(args.prompt_tokens))
    tokenizer = llm.get_tokenizer()
    prompts_by_length = {
        length: _build_exact_token_prompts(
            tokenizer, args.num_prompts, length
        )
        for length in prompt_lengths
    }
    for length, prompts in prompts_by_length.items():
        observed = {len(p["prompt_token_ids"]) for p in prompts}
        if observed != {length}:
            print(
                f"error: constructed prompt lengths for requested {length}: "
                f"{sorted(observed)}",
                file=sys.stderr,
            )
            return 1

    # Ignoring EOS makes output work identical across modes and context cells.
    sampling_params = sampling_params_cls(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    expected_output_tokens = args.num_prompts * args.max_tokens

    def one_round(prompt_length: int, enabled: bool) -> tuple[float, int, dict]:
        llm.collective_rpc(_set_mode, args=(enabled,))
        start = time.perf_counter()
        outputs = llm.generate(
            prompts_by_length[prompt_length],
            sampling_params,
            use_tqdm=False,
        )
        duration = time.perf_counter() - start
        trace = llm.collective_rpc(_drain_ab_trace)[0]

        actual_prompt_lengths = [len(output.prompt_token_ids)
                                 for output in outputs]
        if (len(actual_prompt_lengths) != args.num_prompts
                or any(length != prompt_length
                       for length in actual_prompt_lengths)):
            raise RuntimeError(
                f"engine did not preserve exact {prompt_length}-token inputs: "
                f"count={len(actual_prompt_lengths)}, "
                f"lengths={sorted(set(actual_prompt_lengths))}"
            )

        output_tokens = sum(
            len(output.outputs[0].token_ids) for output in outputs
        )
        if output_tokens != expected_output_tokens:
            raise RuntimeError(
                f"expected {expected_output_tokens} output tokens at input "
                f"length {prompt_length}, got {output_tokens}"
            )
        return duration, output_tokens, trace

    print("\nexact prompt-token sweep")
    print(f"  input lengths: {prompt_lengths}")
    print(f"  prompts/cell: {args.num_prompts}")
    print(f"  output tokens/request: {args.max_tokens} (EOS ignored)")
    print(f"\nwarmup: {args.warmup_rounds} paired round(s) per input length")

    warmup_corrected_traces = {length: [] for length in prompt_lengths}
    warmup_stock_decisions = []   # used to estimate --overhead-ms auto
    for warmup_round in range(args.warmup_rounds):
        length_order = (prompt_lengths if warmup_round % 2 == 0
                        else list(reversed(prompt_lengths)))
        mode_order = ((False, "stock"), (True, "corrected"))
        if warmup_round % 2:
            mode_order = tuple(reversed(mode_order))
        for length in length_order:
            for enabled, _ in mode_order:
                _, _, trace = one_round(length, enabled)
                if enabled:
                    warmup_corrected_traces[length].extend(trace["decisions"])
                else:
                    warmup_stock_decisions.extend(trace["decisions"])

    # One overhead for all input lengths, as a real deployment would have to
    # use: it is pooled across every length's stock warmup steps.
    overhead_source = "fixed"
    overhead_steps = 0
    if args.overhead_ms == "auto":
        overhead_ms, overhead_steps = estimate_overhead_ms(
            warmup_stock_decisions
        )
        overhead_source = "auto"
        llm.collective_rpc(_set_overhead, args=(overhead_ms,))
        print(f"\nestimated per-step overhead: {overhead_ms:.3f} ms "
              f"(median over {overhead_steps} stock warmup decode steps)")
    else:
        overhead_ms = float(args.overhead_ms)
    print(f"corrected arm adds {overhead_ms:.3f} ms to every step cost")

    for length, decisions in warmup_corrected_traces.items():
        clamped = [d for d in decisions if d["chosen_was_clamped"]]
        if clamped:
            print(
                "error: calibrated correction clamped the selected "
                f"verification cost at input length {length} in "
                f"{len(clamped)}/{len(decisions)} warmup decisions; refusing "
                "to run an invalid A/B",
                file=sys.stderr,
            )
            return 1

    measurements = {
        length: {
            "stock_tok_s": [],
            "corrected_tok_s": [],
            "output_tokens": {"stock": [], "corrected": []},
            "round_orders": [],
            "decision_traces": {"stock": [], "corrected": []},
        }
        for length in prompt_lengths
    }

    print(f"\nmeasuring {args.rounds} paired rounds per input length\n")
    print(f"{'round':>6} {'input':>7} {'mode':>10} "
          f"{'seconds':>9} {'out toks':>9} {'tok/s':>9}")
    print("-" * 58)
    for round_index in range(args.rounds):
        length_order = (prompt_lengths if round_index % 2 == 0
                        else list(reversed(prompt_lengths)))
        mode_order = ((False, "stock"), (True, "corrected"))
        if round_index % 2:
            mode_order = tuple(reversed(mode_order))

        for length in length_order:
            cell = measurements[length]
            cell["round_orders"].append([name for _, name in mode_order])
            for enabled, name in mode_order:
                duration, output_tokens, trace = one_round(length, enabled)
                cell[f"{name}_tok_s"].append(output_tokens / duration)
                cell["output_tokens"][name].append(output_tokens)
                cell["decision_traces"][name].extend(trace["decisions"])
                print(f"{round_index + 1:>6} {length:>7} {name:>10} "
                      f"{duration:>9.3f} {output_tokens:>9} "
                      f"{output_tokens / duration:>9.1f}")

            if (cell["output_tokens"]["stock"][-1]
                    != cell["output_tokens"]["corrected"][-1]):
                print(
                    f"error: round {round_index + 1}, input {length} generated "
                    "different output counts between modes",
                    file=sys.stderr,
                )
                return 1

    workload_results = []
    print("\nper-context paired results")
    print(f"{'input':>7} {'stock tok/s':>12} {'corr tok/s':>12} "
          f"{'mean delta':>11} {'SE':>9} {'corr wins':>10}")
    print("-" * 67)
    for length in prompt_lengths:
        cell = measurements[length]
        stock = cell["stock_tok_s"]
        corrected = cell["corrected_tok_s"]
        paired = [(c - s) / s for s, c in zip(stock, corrected)]
        mean_delta = statistics.mean(paired)
        median_delta = statistics.median(paired)
        standard_error = (
            statistics.stdev(paired) / len(paired) ** 0.5
            if len(paired) > 2 else None
        )
        spread = lambda values: (
            (max(values) - min(values)) / statistics.median(values)
        )
        noise = max(spread(stock), spread(corrected))
        corrected_wins = sum(delta > 0 for delta in paired)
        traces = cell["decision_traces"]
        measured_clamps = [
            decision
            for decision in traces["corrected"]
            if decision["chosen_was_clamped"]
        ]
        if measured_clamps:
            print(
                "error: calibrated correction clamped the selected "
                f"verification cost at input length {length} in "
                f"{len(measured_clamps)}/{len(traces['corrected'])} measured "
                "decisions; refusing to publish an invalid A/B",
                file=sys.stderr,
            )
            return 1

        decode_traces = {
            name: [
                decision for decision in traces[name]
                if decision["non_draft_tokens"] == decision["num_reqs"]
            ]
            for name in ("stock", "corrected")
        }

        print(f"{length:>7} {statistics.median(stock):>12.1f} "
              f"{statistics.median(corrected):>12.1f} "
              f"{mean_delta:>+10.2%} "
              f"{standard_error if standard_error is not None else float('nan'):>8.2%} "
              f"{corrected_wins:>4}/{len(paired):<4}")

        workload_results.append({
            "prompt_tokens": length,
            "num_prompts": args.num_prompts,
            "output_tokens_per_prompt": args.max_tokens,
            "expected_output_tokens_per_round": expected_output_tokens,
            "stock_tok_s": stock,
            "corrected_tok_s": corrected,
            "output_tokens": cell["output_tokens"],
            "round_orders": cell["round_orders"],
            "paired_deltas": paired,
            "mean_delta": mean_delta,
            "median_delta": median_delta,
            "standard_error": standard_error,
            "noise": noise,
            "corrected_wins": corrected_wins,
            "decision_summary": {
                name: _decision_summary(traces[name])
                for name in ("stock", "corrected")
            },
            "decode_decision_summary": {
                name: _decision_summary(decode_traces[name])
                for name in ("stock", "corrected")
            },
            "decision_traces": traces,
        })

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "exact_prompt_context_sweep",
        "args": vars(args),
        "profile_context_len": profile_ctx,
        "prompt_construction": {
            "kind": "token_ids",
            "exact_lengths_verified_every_round": True,
            "homogeneous_batches": True,
            "ignore_eos": True,
        },
        "correction_setup": setup,
        "overhead": {
            "ms": overhead_ms,
            "source": overhead_source,
            "estimated_from_steps": overhead_steps,
        },
        "workloads": workload_results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as output_file:
        json.dump(result, output_file, indent=2)
    print(f"\nwrote {args.out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="openbmb/MiniCPM5-2B")
    p.add_argument("--draft", default="openbmb/MiniCPM5-2B-DSpark")
    p.add_argument("--num-speculative-tokens", type=int, default=3)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--attention-backend", default=None)
    k_group = p.add_mutually_exclusive_group(required=True)
    k_group.add_argument("--k", type=float,
                         help="pre-calibrated milliseconds per KV token")
    k_group.add_argument("--calibrate-k", action="store_true",
                         help="fit k inside this DSpark engine before the A/B")
    p.add_argument("--calibration-token-counts", type=int, nargs="+",
                   default=[64, 128, 256, 384, 512])
    p.add_argument("--calibration-context-lens", type=int, nargs="+",
                   default=[0, 1024, 2048, 3072, 4096])
    p.add_argument("--calibration-replays", type=int, default=25)
    p.add_argument("--calibration-warmup", type=int, default=5)
    p.add_argument("--calibration-out",
                   default="results/serving_ab_calibration.json")
    p.add_argument("--rounds", type=int, default=6,
                   help="Counterbalanced paired rounds per mode.")
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--prompt-tokens", type=int, nargs="+", default=[64, 512, 2048])
    p.add_argument(
        "--exact-prompt-sweep",
        action="store_true",
        help=("run one homogeneous workload per --prompt-tokens value using "
              "token-ID inputs with exact lengths"),
    )
    p.add_argument(
        "--overhead-ms",
        default="0",
        help=("per-step overhead (ms) added to every cost in the corrected "
              "arm, or 'auto' to estimate it from the stock warmup rounds "
              "(auto needs --exact-prompt-sweep). Default 0 = k-only, the "
              "same as earlier runs."),
    )
    p.add_argument("--out", default="results/serving_ab.json")
    args = p.parse_args()

    if args.overhead_ms == "auto":
        if not args.exact_prompt_sweep:
            print("error: --overhead-ms auto requires --exact-prompt-sweep",
                  file=sys.stderr)
            return 1
    else:
        try:
            if float(args.overhead_ms) < 0:
                raise ValueError
        except ValueError:
            print("error: --overhead-ms must be 'auto' or a number >= 0",
                  file=sys.stderr)
            return 1

    if args.rounds < 2 or args.warmup_rounds < 1:
        print("error: use at least 2 measured rounds and 1 warmup round",
              file=sys.stderr)
        return 2
    if args.num_prompts < args.max_num_seqs:
        print("error: --num-prompts must be at least --max-num-seqs so the "
              "full-concurrency shape is exercised", file=sys.stderr)
        return 2
    if any(length <= 0 for length in args.prompt_tokens):
        print("error: prompt token lengths must be positive", file=sys.stderr)
        return 2
    if args.max_tokens <= 0:
        print("error: --max-tokens must be positive", file=sys.stderr)
        return 2
    if max(args.prompt_tokens) + args.max_tokens > args.max_model_len:
        print(
            "error: prompt plus output length exceeds --max-model-len "
            f"({max(args.prompt_tokens)} + {args.max_tokens} > "
            f"{args.max_model_len})",
            file=sys.stderr,
        )
        return 2
    if args.exact_prompt_sweep and args.num_prompts != args.max_num_seqs:
        print(
            "error: exact prompt sweep requires --num-prompts to equal "
            "--max-num-seqs so every cell is one full-concurrency batch",
            file=sys.stderr,
        )
        return 2

    if args.calibrate_k:
        if args.calibration_replays < 2 or args.calibration_warmup < 1:
            print("error: calibration requires at least 2 replays and 1 warmup",
                  file=sys.stderr)
            return 2
        largest_relevant = args.max_num_seqs * (1 + args.num_speculative_tokens)
        invalid = [
            n for n in args.calibration_token_counts
            if n < args.max_num_seqs or n > largest_relevant
        ]
        if invalid:
            print(
                "error: calibration token counts must cover full-concurrency "
                f"candidate shapes in [{args.max_num_seqs}, {largest_relevant}]; "
                f"invalid: {invalid}",
                file=sys.stderr,
            )
            return 2
        if min(args.calibration_context_lens) < 0:
            print("error: calibration context lengths must be non-negative",
                  file=sys.stderr)
            return 2

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    import vllm.envs as envs
    from vllm import LLM, SamplingParams

    if not envs.VLLM_USE_V2_MODEL_RUNNER:
        print("error: set VLLM_USE_V2_MODEL_RUNNER=1", file=sys.stderr)
        return 2
    profile_ctx = int(envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN)

    extra = {}
    if args.attention_backend:
        extra["attention_backend"] = args.attention_backend

    llm = LLM(
        model=args.model,
        speculative_config={
            "method": "dspark",
            "model": args.draft,
            "num_speculative_tokens": args.num_speculative_tokens,
            "enable_adaptive_verification": True,
        },
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Prefix caching would let later rounds reuse earlier rounds' KV and
        # make whichever mode ran second look faster.
        enable_prefix_caching=False,
        **extra,
    )

    calibration = None
    if args.calibrate_k:
        print("\ncalibrating k inside the live DSpark engine ...")
        calibration = llm.collective_rpc(
            _calibrate_on_worker,
            args=(
                sorted(set(args.calibration_token_counts)),
                sorted(set(args.calibration_context_lens)),
                args.calibration_replays,
                args.calibration_warmup,
            ),
        )[0]
        if not calibration.get("ok"):
            print(f"error: calibration failed: {calibration.get('why')}",
                  file=sys.stderr)
            if calibration.get("captured_token_counts") is not None:
                print("captured token counts: "
                      f"{calibration['captured_token_counts']}", file=sys.stderr)
            return 1

        try:
            fitted_k, slopes, fit_rmse = fit_k_from_rows(calibration["rows"])
        except (KeyError, TypeError, ValueError) as exc:
            print(f"error: could not fit in-engine calibration: {exc}",
                  file=sys.stderr)
            return 1
        if not math.isfinite(fitted_k) or fitted_k <= 0:
            print(f"error: fitted non-positive/invalid k={fitted_k}",
                  file=sys.stderr)
            return 1
        args.k = fitted_k

        detail = ", ".join(
            f"n={n}: {slope * 1000:.4f}"
            for n, slope in sorted(slopes.items())
        )
        print(f"fitted k = {args.k * 1000:.4f} us/KV-token")
        print(f"per-shape slopes (us/KV-token): {detail}")
        print(f"fixed-effect fit RMSE = {fit_rmse:.4f} ms")
        print("captured token counts: "
              f"{calibration['captured_token_counts']}")

        slope_values = list(slopes.values())
        median_slope = statistics.median(slope_values)
        slope_spread = (
            (max(slope_values) - min(slope_values)) / abs(median_slope)
            if median_slope else float("inf")
        )
        if any(slope <= 0 for slope in slope_values) or slope_spread > 0.5:
            print(
                "WARNING: per-shape k values are not tightly consistent; "
                "the pooled correction is a least-squares fit, not a universal "
                "physical constant for this configuration.",
                file=sys.stderr,
            )

        calibration["generated_utc"] = datetime.now(timezone.utc).isoformat()
        calibration["fit"] = {
            "k_ms_per_kv_token": args.k,
            "k_us_per_kv_token": args.k * 1000,
            "per_shape_k_ms_per_kv_token": slopes,
            "rmse_ms": fit_rmse,
            "relative_per_shape_slope_spread": slope_spread,
        }
        calibration["args"] = {
            "model": args.model,
            "draft": args.draft,
            "num_speculative_tokens": args.num_speculative_tokens,
            "token_counts": args.calibration_token_counts,
            "context_lens": args.calibration_context_lens,
            "replays": args.calibration_replays,
            "warmup": args.calibration_warmup,
        }
        os.makedirs(os.path.dirname(args.calibration_out) or ".", exist_ok=True)
        with open(args.calibration_out, "w") as f:
            json.dump(calibration, f, indent=2)
        print(f"wrote {args.calibration_out}")

    assert args.k is not None

    # With 'auto', start at 0; the sweep sets the estimate after warmup.
    initial_overhead_ms = (0.0 if args.overhead_ms == "auto"
                           else float(args.overhead_ms))
    setup = llm.collective_rpc(
        _install_correction,
        args=(args.k, profile_ctx, initial_overhead_ms),
    )[0]
    if not setup.get("ok"):
        print(f"error: {setup.get('why')}", file=sys.stderr)
        return 1
    print(f"correction installed; k={args.k*1000:.4f} us/KV-token, "
          f"configured profile context={profile_ctx}, "
          f"max model length={setup['max_model_len']}")

    if args.exact_prompt_sweep:
        return _run_exact_prompt_sweep(
            llm, SamplingParams, args, profile_ctx, setup
        )

    tok = llm.get_tokenizer()
    filler = " systems research and inference serving"
    prompts = []
    for i in range(args.num_prompts):
        want = args.prompt_tokens[i % len(args.prompt_tokens)]
        ids = tok("Explain in detail:" + filler * (want // 6 + 1)).input_ids[:want]
        prompts.append(tok.decode(ids))
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    def one_round(enabled: bool) -> tuple[float, int, dict]:
        llm.collective_rpc(_set_mode, args=(enabled,))
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        trace = llm.collective_rpc(_drain_ab_trace)[0]
        return dt, sum(len(o.outputs[0].token_ids) for o in outs), trace

    print(f"\nwarmup: {args.warmup_rounds} round(s) per mode")
    warmup_corrected_traces = []
    for r in range(args.warmup_rounds):
        order = (False, True) if r % 2 == 0 else (True, False)
        for enabled in order:
            _, _, trace = one_round(enabled)
            if enabled:
                warmup_corrected_traces.extend(trace["decisions"])

    clamped_warmup = [
        d for d in warmup_corrected_traces if d["chosen_was_clamped"]
    ]
    if clamped_warmup:
        print(
            "error: calibrated correction clamped the selected verification "
            f"cost in {len(clamped_warmup)}/{len(warmup_corrected_traces)} "
            "warmup decisions; refusing to run an invalid A/B",
            file=sys.stderr,
        )
        return 1

    print(f"measuring {args.rounds} counterbalanced rounds per mode\n")
    print(f"{'round':>6} {'mode':>10} {'seconds':>9} {'out toks':>9} {'tok/s':>9}")
    print("-" * 48)
    res = {"stock": [], "corrected": []}
    output_tokens = {"stock": [], "corrected": []}
    traces = {"stock": [], "corrected": []}
    round_orders = []
    for r in range(args.rounds):
        order = ((False, "stock"), (True, "corrected"))
        if r % 2:
            order = tuple(reversed(order))
        round_orders.append([name for _, name in order])
        for enabled, name in order:
            dt, toks, trace = one_round(enabled)
            res[name].append(toks / dt)
            output_tokens[name].append(toks)
            traces[name].extend(trace["decisions"])
            print(f"{r+1:>6} {name:>10} {dt:>9.3f} {toks:>9} {toks/dt:>9.1f}")
        if output_tokens["stock"][-1] != output_tokens["corrected"][-1]:
            print(
                f"error: round {r + 1} generated different token counts "
                f"(stock={output_tokens['stock'][-1]}, "
                f"corrected={output_tokens['corrected'][-1]}); refusing to "
                "compare different workloads",
                file=sys.stderr,
            )
            return 1

    s_med = statistics.median(res["stock"])
    c_med = statistics.median(res["corrected"])
    spread = lambda v: (max(v) - min(v)) / statistics.median(v)

    print(f"\n{'':>10} {'median tok/s':>13} {'round spread':>13}")
    print(f"{'stock':>10} {s_med:>13.1f} {spread(res['stock']):>12.1%}")
    print(f"{'corrected':>10} {c_med:>13.1f} {spread(res['corrected']):>12.1%}")

    # PAIRED comparison. The two modes alternate inside each round, so they
    # share that round's drift; subtracting within a round removes it. An
    # unpaired test against the raw spread is far too conservative and will
    # hide a real effect smaller than the drift.
    pairs = [(c - s) / s for s, c in zip(res["stock"], res["corrected"])]
    mean = statistics.mean(pairs)
    print(f"\npaired per-round delta: "
          f"{', '.join(f'{p:+.2%}' for p in pairs)}")
    print(f"mean {mean:+.2%}   median {statistics.median(pairs):+.2%}   "
          f"corrected faster in {sum(p > 0 for p in pairs)}/{len(pairs)} rounds")
    delta = mean

    if len(pairs) > 2:
        se = statistics.stdev(pairs) / len(pairs) ** 0.5
        n_se = mean / se if se else float("inf")
        print(f"standard error {se:.2%}; mean is {n_se:.1f} SE from zero")
        if abs(n_se) < 2:
            print("NOT DISTINGUISHABLE FROM ZERO: this bounds the benefit rather\n"
                  "than measuring it.")
        else:
            print("Effect is distinguishable from zero.")
    noise = max(spread(res["stock"]), spread(res["corrected"]))

    print("\nDecision diagnostics:")
    for name in ("stock", "corrected"):
        decisions = traces[name]
        if not decisions:
            print(f"  {name:>9}: no decisions captured")
            continue
        ceiling = sum(d["at_ceiling"] for d in decisions) / len(decisions)
        clamped = sum(d["chosen_was_clamped"] for d in decisions) / len(decisions)
        budgets = [d["draft_budget"] for d in decisions]
        print(f"  {name:>9}: n={len(decisions)}, ceiling={ceiling:.1%}, "
              f"budget median={statistics.median(budgets):g}, "
              f"chosen-cost clamped={clamped:.1%}")

    corrected_offsets = [d["offset_ms"] for d in traces["corrected"]]
    if corrected_offsets:
        print(f"  correction offset median {statistics.median(corrected_offsets):+.3f} ms")

    out = {"generated_utc": datetime.now(timezone.utc).isoformat(),
           "args": vars(args), "profile_context_len": profile_ctx,
           "stock_tok_s": res["stock"], "corrected_tok_s": res["corrected"],
           "output_tokens": output_tokens, "round_orders": round_orders,
           "delta": delta, "noise": noise, "correction_setup": setup,
           "decision_traces": traces}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
