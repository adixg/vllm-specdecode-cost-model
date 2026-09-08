"""Child process: run one speculative config and collect its acceptance stats.

vLLM tracks speculative-decoding counters per scheduler step in
SpecDecodingStats (vllm/v1/spec_decode/metrics.py):

    num_drafts                    drafting rounds
    num_draft_tokens              draft tokens proposed
    num_accepted_tokens           draft tokens the target accepted
    num_accepted_tokens_per_pos   accepted count, per draft position
    num_draft_tokens_per_pos      proposed count, per draft position

Those reach the frontend process inside SchedulerStats, where stat loggers see
them. The offline LLM API does not expose a hook for custom loggers, so we
attach one to llm_engine.logger_manager after construction. That requires
disable_log_stats=False, since LLM() otherwise disables stats entirely and
logger_manager is None.

Prints one SPEC line of JSON on stdout.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from vllm import LLM, SamplingParams
from vllm.v1.metrics.loggers import StatLoggerBase


class SpecCollector(StatLoggerBase):
    """Accumulates SpecDecodingStats across every scheduler step of a run."""

    def __init__(self, vllm_config=None, engine_index: int = 0):
        self.num_drafts = 0
        self.num_draft_tokens = 0
        self.num_accepted_tokens = 0
        self.accepted_per_pos: np.ndarray | None = None
        self.drafted_per_pos: np.ndarray | None = None
        self.steps = 0

    def record(self, scheduler_stats=None, iteration_stats=None,
               mm_cache_stats=None, engine_idx=0):
        s = getattr(scheduler_stats, "spec_decoding_stats", None)
        if s is None:
            return
        self.steps += 1
        self.num_drafts += s.num_drafts
        self.num_draft_tokens += s.num_draft_tokens
        self.num_accepted_tokens += s.num_accepted_tokens
        acc = np.array(s.num_accepted_tokens_per_pos, dtype=np.int64)
        dr = np.array(s.num_draft_tokens_per_pos, dtype=np.int64)
        self.accepted_per_pos = acc if self.accepted_per_pos is None \
            else self.accepted_per_pos + acc
        self.drafted_per_pos = dr if self.drafted_per_pos is None \
            else self.drafted_per_pos + dr

    def log_engine_initialized(self):
        pass

    def summary(self):
        d = {"steps": self.steps, "num_drafts": int(self.num_drafts),
             "num_draft_tokens": int(self.num_draft_tokens),
             "num_accepted_tokens": int(self.num_accepted_tokens)}
        if self.num_draft_tokens:
            d["acceptance_rate"] = self.num_accepted_tokens / self.num_draft_tokens
        if self.num_drafts:
            # vLLM's convention: acceptance length counts the bonus token, so a
            # run that accepts nothing still emits 1 token per verification.
            # This is the theoretical speedup ceiling.
            d["mean_acceptance_length"] = 1 + self.num_accepted_tokens / self.num_drafts
        if self.accepted_per_pos is not None and self.num_drafts:
            # P(at least i+1 draft tokens accepted) -- decays with position.
            marg = (self.accepted_per_pos / self.num_drafts).tolist()
            d["acceptance_per_pos"] = [round(x, 5) for x in marg]
            cond = []
            prev = 1.0
            for x in marg:
                cond.append(round(x / prev, 5) if prev > 0 else None)
                prev = x
            # P(accept position i | position i-1 accepted)
            d["conditional_acceptance_per_pos"] = cond
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default=None)
    ap.add_argument("--method", default=None, help="e.g. ngram (no draft model)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gmu", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--kv-bytes", type=int, default=536519168)
    ap.add_argument("--batch", type=int, default=8,
                    help="number of requests submitted (the workload size)")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="cap on sequences running concurrently. This -- not "
                         "the number of prompts -- is how batch size should be "
                         "varied: it holds the workload fixed so only "
                         "concurrency changes.")
    a = ap.parse_args()

    kw = dict(model=a.target, gpu_memory_utilization=a.gmu,
              max_model_len=a.max_model_len, kv_cache_memory_bytes=a.kv_bytes,
              disable_log_stats=False)
    if a.max_num_seqs:
        kw["max_num_seqs"] = a.max_num_seqs
    if a.method == "ngram":
        kw["speculative_config"] = {"method": "ngram",
                                    "num_speculative_tokens": a.k,
                                    "prompt_lookup_max": 4,
                                    "prompt_lookup_min": 2}
    elif a.draft:
        kw["speculative_config"] = {"model": a.draft,
                                    "num_speculative_tokens": a.k}

    llm = LLM(**kw)

    collector = SpecCollector()
    mgr = getattr(llm.llm_engine, "logger_manager", None)
    if mgr is None:
        raise RuntimeError("logger_manager is None -- stats are disabled")
    mgr.stat_loggers.append(collector)

    from phase2_correctness import PROMPTS
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(a.batch)]
    # ignore_eos + fixed max_tokens: every request emits exactly the same
    # number of tokens, so batch composition cannot drift between configs.
    sp = SamplingParams(max_tokens=a.max_tokens, temperature=0, ignore_eos=True)

    import time
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    elapsed = time.perf_counter() - t0

    gen = sum(len(o.outputs[0].token_ids) for o in outs)
    res = {"target": a.target, "draft": a.draft, "method": a.method,
           "k": a.k if (a.draft or a.method) else None,
           "batch": a.batch, "max_num_seqs": a.max_num_seqs,
           "max_tokens": a.max_tokens,
           "elapsed_s": round(elapsed, 4),
           "generated_tokens": gen,
           "output_toks_per_s": round(gen / elapsed, 2)}
    res.update(collector.summary())
    print("SPEC " + json.dumps(res))


if __name__ == "__main__":
    main()
