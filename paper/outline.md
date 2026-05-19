# FissionSpec: Outcome-Decoupled Continuous Batching for
# Speculative-Speculative LLM Serving

## Abstract skeleton

Speculative-speculative decoding overlaps drafting with target verification by
preparing continuations for likely verification outcomes. Under batching,
outcome misses create a new systems externality: a miss can either stall the
whole cohort or occupy a padded target row while it recovers. We introduce
FissionSpec, which makes each outcome lookup independently schedulable, removes
recovering misses from target batches, and re-fuses compatible ready requests
with a bounded model-predictive controller. An exact rational CPU oracle proves
finite-model speculative-sampling equality under rebatching, while a versioned
provisional-KV protocol and per-request counter RNG test the serving substrate.
Real-model numerical equivalence and performance remain production-engine
obligations. [Replace this sentence with measured GPU results only after the
SGLang/vLLM integration is evaluated.]

## 1. Problem

For batch size `B`, per-row cache-hit probabilities `p_i`, and recovery latency
`R`, Saguaro's batch fallback probability is

```text
q_barrier = 1 - product_i p_i.
```

With iid `p`, aggregate head-of-line waiting is

```text
C_barrier = B R (1 - p^B),
C_isolated = B R (1 - p),
C_barrier / C_isolated = sum_{j=0}^{B-1} p^j.
```

The last expression grows almost linearly in `B` when hits are common—the
counterintuitive regime where async speculation should work best.

SPECTRE removes the wait in parallel mode but keeps misses in a mixed target
batch as one-token candidates padded to speculative width `k`. If masked slots
are not fully free in the chosen kernel/CUDA graph, the upper-bound padding
externality is `(k - 1)` target slots per miss per round. GPU profiling must
measure the realized, rather than assumed, fraction of this bound.

Here “miss” means that the realized verification outcome is absent from the
*next-continuation cache*. It does not mean that the current draft block was
rejected. Accepted-prefix length controls tokens emitted this round; cache
membership controls readiness for the next round. The simulator and experiment
harness draw these processes independently unless a trace supplies a joint
model.

## 2. Design

1. Outcome events split a verification cohort into hit, recovery, and finished
   lanes.
2. Hits are eligible for immediate hit-only dispatch.
3. Misses retain a versioned recovery snapshot and are absent from the target.
4. The Python horizon-2 controller chooses dispatch or bounded wait/re-fusion
   from calibrated row/slot latency surfaces and per-request slack. The Rust
   primitive separately exposes weighted flow and an engine-defined bypass.
5. Epoch transactions commit only a target-verified branch prefix.
6. Counter-based RNG makes logical samples invariant to execution order.

## 3. Controller model

For `n` currently ready rows, `m` rows recovering at time `delta`, and target
latency `L(b)`, compare two deterministic two-batch schedules.

Launch now:

```text
C_now = n L(n) + m (max(L(n), delta) + L(m) - delta).
```

Wait and fuse:

```text
C_wait = n (delta + L(n+m)) + m L(n+m).
```

The Python implementation uses equal row weights, exact packed row widths,
deadlines, the earliest internally known readiness cohort, a maximum batch
size, EDF admission, and a hard coalescing bound. Future external arrivals are
hidden. After a first token exists, it never knowingly waits past the earliest
selected-row TBT slack. The Rust primitive supports priority-weighted flow over
a one-dimensional flattened-work profile; it is not a differential port of the
Python two-dimensional model. A restricted offline oracle exhausts only binary
dispatch-now versus wait-to-next-readiness choices under the simulator's fixed
deadline-first admission order. It does not optimize arbitrary subsets,
orderings, or wait durations.

## 4. Correctness obligations

The exact CPU oracle closes the algorithmic token-distribution obligation for
finite rational autoregressive models. The following remain production
obligations because real kernels add floating-point and block-table behavior.

- Only target-verified tokens become committed target state.
- `(request, round, version)` totally orders state mutation per request.
- Old replies are ignored, not “best effort” applied.
- Selected provisional pages commit; siblings abort idempotently.
- Position, token count, and block-table length remain mutually consistent.
- Sampling consumes a logical per-request stream independent of batching.

## 5. Evaluation plan

### Systems and baselines

Saguaro (neural and fast fallback), full SPECTRE (ordinary/parallel/hybrid), its
parallel padded-mode component, immediate fission, fixed coalescing,
EXSpec-style grouping, a myopic slack controller, the horizon-2 controller, and
the restricted small-trace coalescing oracle. End-to-end context
also includes vLLM/SGLang autoregressive, ordinary SD, EAGLE-3, FASER,
MineDraft, PEARL/AMUSD, SwiftSpec, SpecBranch, TETRIS, and TurboSpec where
artifacts and model pairs permit a matched comparison.

### Workloads

- One miss plus `B-1` hits (mechanism isolation).
- Poisson, bursty, and production-trace arrivals.
- Batch/concurrency 1–128; temperature 0, 0.6, 1.0.
- Heterogeneous context/output lengths and correlated hit classes.
- Draft contention and network-jitter sweeps.
- Llama-3.1-70B/Llama-3.2-1B and Qwen3-32B/Qwen3-0.6B.

### Metrics

- P50/P95/P99 TBT and TTFT; deadline goodput.
- Output tokens/s per total GPU.
- Conditional hit slowdown when at least one cohort peer misses.
- Padded verifier slots and measured target FLOPs/time.
- Batch size, launch count, target/draft utilization.
- Recovery starvation, fission fragmentation, re-fusion wait.
- Provisional KV footprint and controller overhead.
- Greedy equivalence and sampled-distribution tests.

## 6. Threats to validity

The simulator establishes mechanism behavior, not GPU speedup. Its latency
surface must be replaced with engine measurements. Paged kernels can make
padding cheaper than token-count models predict; CUDA graph bucketing can make
it more expensive. Batch reordering can change floating-point reductions even
when the target distribution is algorithmically preserved. Every reported GPU
result must match total target and draft resources.

The laptop artifact's `spectre-parallel-padded` policy is a component ablation,
not the full hybrid SPECTRE baseline. Its draft server is a non-preemptive FIFO,
arrivals are decode-ready after prefill/initial proposal, and TBT excludes TTFT.
Cache-hit probability is exogenous at fixed row-based precompute cost: outcome
fanout, context length, cache memory, and their joint dependence on token
acceptance are omitted. None of those model outputs may be presented as an
end-to-end systems result.

The working bibliography is in [`references.bib`](references.bib); literature
claims remain pinned to the dated boundary in `docs/novelty.md`.
