# CPU fidelity layer

`fissionspec.fidelity` is an executable, dependency-free bridge between the
count-level simulator and a future serving prototype. It turns five effects
that are normally represented by scalar probabilities or fixed latency curves
into deterministic event traces:

1. realized outcome-tree membership under a finite, page-rounded LRU budget;
2. latent request classes that jointly determine acceptance and outcome
   popularity, with optional cross-request class correlation;
3. context-sensitive prefill, target, draft, recovery, and network costs;
4. a multiworker remote draft service with continuous batching or strict
   priority dispatch, bounded admission, jitter, failure, and retry; and
5. heterogeneous prefill and first-token timing for one speculative round.

The layer is deliberately composable. It does not replace the policy
simulator.

## Outcome-tree cache

Each cached branch has the key

```text
(request_id, speculative_round, categorical_outcome_id)
```

and consumes

```text
ceil(logical_branch_bytes / page_size_bytes)
```

whole pages. `OutcomeTreeCache` tracks logical bytes and allocated pages
separately. Insertion evicts complete entries in least-recently-used order;
the full key is the deterministic final tie-break. An object larger than the
entire page budget is rejected atomically. `lookup()` changes recency only on
a hit, `contains()` supports non-mutating trace inspection, and `discard()`
releases terminal-request state without perturbing unrelated recency.

`simulate_fidelity_trace()` materializes the most popular `fanout` outcomes
for each selected class. At target completion, it samples the realized
categorical outcome and performs an exact key lookup. The trace's
`cached_outcomes` field is the set still resident for that request at that
lookup instant, after all prior LRU evictions. Consequently, for every
non-terminal result,

```python
result.cache_hit == (result.realized_outcome in result.cached_outcomes)
```

is an executable invariant, not a Bernoulli approximation.

## Coupled and correlated traffic

An `OutcomeClass` contains both a token-acceptance probability and a
categorical continuation distribution. A `FidelityRequest` specifies a
mixture over classes. The counter-based RNG first resolves the latent class,
then keys both acceptance and outcome draws by that class. This provides a
simple, explicit coupling: traffic selected into a class with a peaky
continuation distribution can simultaneously have a different acceptance
regime.

Requests with the same `correlation_key` share one latent-class draw. They
must declare identical class mixtures. Group draws and independent-request
draws are domain-separated, so a request identifier cannot accidentally alias
a correlation-group name.

## Context and transport costs

`ContextCostModel` makes each surface executable:

- prefill depends on prompt tokens;
- target verification depends on rows, summed context, and physical verifier
  slots;
- remote precompute and recovery depend on rows, summed context, and outcome
  branches, with distinct base costs; and
- each network direction depends on payload bytes plus counter-keyed uniform
  jitter.

All coefficients are milliseconds in the CPU model. The functions validate
their physical inputs and are deterministic for a fixed seed and event key.

## Remote draft service

`RemoteDraftService` is a deterministic, non-preemptive scheduler over a
configurable worker pool.

- `continuous-batching` uses arrival/job-id order and coalesces only jobs of
  the same kind up to `max_batch_size` during a formation window of
  `batch_window_ms`.
- `priority` dispatches one job at a time by descending integer priority,
  followed by arrival/job-id order.
- `queue_capacity` bounds the admitted waiting queue. Per-worker forming
  batches are separately bounded by `max_batch_size`. Attempts delayed at
  queue admission expose `backpressure_delay_ms`.
- Request and response network costs are traced independently.
- Failure is counter-keyed per `(job_id, attempt)`. A failed attempt retries
  the whole job after response transport and `retry_backoff_ms`, up to
  `max_retries`.

Every attempt records network readiness, admission, service start/end,
response completion, worker, batch, failure, and terminal state. Every
terminal input job appears exactly once in either `successful_job_ids` or
`terminal_failed_job_ids`; batches on the same worker never overlap; and
`queue_peak` cannot exceed the configured capacity.

## One-round request trace

The high-level trace performs:

```text
arrival
  -> serialized target-side prefill
  -> remote outcome-tree precompute
  -> target verification batch
  -> exact cache lookup
  -> remote recovery on miss
```

Prefill and target batching use each request's prompt length, output length,
speculation width, arrival, and priority. The target completion is the first
token time for this one-round harness, so

```text
TTFT = target_completion - request_arrival
```

The round emits one target token plus its accepted draft prefix, capped by the
request's remaining output. Each result therefore preserves

```text
emitted_tokens + output_tokens_remaining = requested_output_tokens
```

If the round exhausts the requested output, `terminal` is true,
`realized_outcome` and `cache_hit` are `None`, no recovery is submitted, and
any already-materialized continuation branches are released. Capacity-driven
LRU evictions and terminal releases are counted separately.

Precompute completion and target lookup are merged in one ordered event
stream. A precompute response that arrives after its lookup is marked stale
and cannot retroactively turn the request into a hit.

## Exact reference reduction

The null bridge is explicit rather than approximate:

```python
costs = ContextCostModel.reference(profile)
remote = RemoteDraftConfig.reference(max_batch_size=16)
config = FidelityConfig.reference(profile, target_batch_size=16)
```

Under `ContextCostModel.reference(profile)`:

- prefill and both network directions are exactly zero;
- a target batch calls
  `profile.target_latency_ms(rows, verifier_slots)` exactly once; and
- a remote batch calls
  `profile.draft_latency_ms(rows, recovery=...)` exactly once.

The reference remote configuration is one reliable worker, zero batch wait,
no failure/retry, and capacity equal to its batch bound. Therefore,
simultaneous same-kind jobs within that bound produce exactly one existing
draft-curve launch. `FidelityConfig.reference()` additionally supplies an
effectively unbounded one-byte-page cache and unit branch accounting. With a
single-outcome class, this removes fanout uncertainty as well. The bridge
tests assert these equalities directly; there are no fitted tolerances or
golden trace replacements.

## Minimal example

```python
from fissionspec.fidelity import (
    ContextCostModel,
    FidelityConfig,
    FidelityRequest,
    OutcomeClass,
    RemoteDraftConfig,
    simulate_fidelity_trace,
)

classes = (
    OutcomeClass("repetitive", 0.85, (0.8, 0.15, 0.05)),
    OutcomeClass("creative", 0.35, (0.4, 0.35, 0.25)),
)
requests = (
    FidelityRequest(
        request_id="r0",
        arrival_ms=0.0,
        prompt_tokens=4096,
        output_tokens=128,
        speculation_length=6,
        class_weights=(("repetitive", 0.7), ("creative", 0.3)),
        correlation_key="tenant-a",
    ),
)
config = FidelityConfig(
    costs=ContextCostModel(
        prefill_base_ms=0.2,
        prefill_per_token_ms=0.001,
        target_base_ms=1.0,
        target_per_row_ms=0.1,
        target_per_context_token_ms=0.0001,
        target_per_verifier_slot_ms=0.01,
        draft_base_ms=0.3,
        recovery_base_ms=0.8,
        draft_per_row_ms=0.04,
        draft_per_context_token_ms=0.00005,
        draft_per_branch_ms=0.01,
        network_base_ms=0.05,
        network_per_byte_ms=0.00001,
        network_jitter_ms=0.02,
    ),
    remote=RemoteDraftConfig(
        workers=2,
        max_batch_size=8,
        batch_window_ms=0.1,
        queue_capacity=32,
        failure_probability=0.01,
        max_retries=2,
    ),
    cache_byte_budget=64 * 1024 * 1024,
    cache_page_size_bytes=16 * 1024,
    kv_bytes_per_token=8192,
    continuation_tokens=6,
    fanout=2,
    target_batch_size=16,
)

trace = simulate_fidelity_trace(requests, classes, config, seed="artifact-seed")
```

## Scope and known limitations

These boundaries are intentional and should not be blurred in experimental
claims:

- The high-level harness executes one speculative round. It does not model
  subsequent decode iterations, online policy feedback, cancellation,
  streaming inter-token latency, or deadline/SLO decisions.
- Prefill and target are each represented by one serialized target-side
  resource. There is no chunked prefill, tensor/pipeline parallelism, CUDA
  graph bucket selection, kernel overlap, or GPU memory allocator.
- Outcome identifiers are abstract categorical continuations. The cache does
  not yet store token sequences, share trie prefixes, reference-count KV
  blocks, model copy-on-write, or expose fragmentation within a page.
- Every branch uses one configured logical byte size. It does not vary with
  layer, head, precision, sequence position, or prefix sharing.
- Conditional on the latent class, acceptance draws are an i.i.d. prefix and
  the outcome draw is categorical. Correlation is shared class membership,
  not a learned copula, temporal process, or trace-calibrated joint model.
- Network latency is a scalar request/response cost with the same configured
  payload size in both directions. It omits packetization, serialization
  pipelines, topology, congestion windows, and correlated failures.
- Continuous batches never mix precompute and recovery jobs. Service is
  non-preemptive; priority mode does not implement aging or fairness.
- In the high-level one-round harness, precompute jobs are scheduled first.
  Recoveries discovered at target lookup reuse the resulting worker
  availability but cannot reorder already-planned precomputes. Experiments
  about recovery/precompute interference need a unified online event loop.
- The reference reduction is an equality to the existing row/slot cost
  abstraction under the stated synchronized, bounded-batch conditions. It is
  not a claim of end-to-end equality with a real GPU serving stack.

Randomized tests exercise replay determinism, byte/page conservation, queue
capacity, worker non-overlap, terminal-job conservation, and token
conservation across heterogeneous requests and seeds.
