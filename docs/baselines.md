# Executable scheduler baselines

FissionSpec has two simulation layers with deliberately different contracts.
The main simulator accepts a narrow policy that decides when an already chosen
target batch should launch. It remains the source of the paper's existing
synthetic results and stable policy names. `fissionspec.baselines` is a separate
deterministic harness for baselines that must choose batch membership, arbitrate
a shared remote drafter, or maintain a sequence pool.

No existing policy name, CLI result, or checked-in golden artifact is changed by
this harness.

## Matched semantic outcomes

`PreRealizedTrace` fixes every request's ordered verification outcomes before a
scheduler runs:

- target-authorized emitted token identities;
- accepted-prefix length;
- physical verification width;
- whether the next continuation needs remote drafting; and
- whether the completed round counts as a rollback for hybrid mode selection.

The scheduler sees a ready row's current sequence length, verification width,
deadline, ready time, and *previous* accepted length. It does not see the
current step's accepted length, emitted tokens, remote-draft outcome, or rollback
bit before target completion.

Future steps become ready only after their predecessor completes. A reusable
continuation becomes ready at target completion; a missing continuation becomes
ready only when its non-preemptive remote-draft job completes.
`assert_semantic_equivalence` rejects comparisons whose complete token and
outcome signatures differ.

`PreRealizedTrace.from_simulation(result)` bridges an existing
`SimulationResult` without resampling. The count-level simulator does not expose
token IDs, so the bridge uses request-local output ordinals as opaque semantic
identities while preserving productive counts, accepted lengths, widths,
hit/miss outcomes, arrivals, prompts, and deadlines.

## SPECTRE hybrid abstraction

`SPECTREHybridScheduler` implements three scheduler-visible mechanisms from
[SPECTRE](https://arxiv.org/html/2605.08151):

1. A batch-level ordinary/parallel decision. `SpectreCalibration` fits the
   explicit cost comparison
   `parallel_cost(r) = parallel_round_ms + r * rollback_penalty_ms` against a
   calibrated `ordinary_round_ms`. Its reported critical rollback ratio is
   `(ordinary - parallel) / rollback_penalty`, clamped to `[0, 1]`.
2. Non-preemptive speculative priority at the shared draft server. A running
   normal job is never interrupted. Once idle, speculative refresh jobs win,
   except that one waiting normal job is forced after each configured burst of
   speculative jobs.
3. A retained-context factor applied to the context-dependent component of
   recovery latency.

Parallel rollback rows issue one auxiliary padded target row while their remote
refresh runs. The row occupies its full verification width, reports
`width - 1` padded slots, and emits no additional semantic token in this
schedule-level harness. The later pre-realized target step remains unchanged.

This is not the SPECTRE SGLang implementation. In particular, it does not model
rejection-sampling kernels, the exact paper throughput equation, network
transport, native-traffic batching, multiple draft workers, cache repair, or
the acceptance degradation that prompt compression may cause. Holding
acceptance fixed intentionally isolates the scheduling and latency mechanism.
The calibration inputs must come from the deployment being studied.

## EXSpec sliding-pool abstraction

`EXSpecSlidingPoolScheduler` follows the cross-batch idea in
[EXSpec](https://arxiv.org/html/2510.22876). It scans the oldest configurable
window of ready sequences, groups rows with the same current content length and
verification width, and dispatches the largest eligible homogeneous group.
Current content length already incorporates target-authorized output from all
previous accepted prefixes. Thus regrouping reacts to accepted-length
raggedness without peeking at the pending verification result.

When no eligible group exists, the scheduler dispatches the oldest rows and
charges the configured correctness-preserving realignment cost. Completed
sequences leave the pool immediately.

This abstraction does not manipulate tensors. It represents, but does not
execute, EXSpec's unpad/append/repad path, position-ID reconstruction,
attention-mask update, KV-cache movement, prompt-length presort, or framework
kernels. The realignment curve therefore requires calibration. It is also
useful as a counterexample generator: an old unique-length sequence can wait
behind a succession of homogeneous groups.

## Myopic slack and aging baseline

`MyopicSlackScheduler` is an explicit control baseline, not a reproduction of a
named system. It ranks each ready row by

```text
deadline - now - estimated_service
    - aging_rate * (now - ready_since)
```

Rows beyond `starvation_bound_ms` receive a fairness-class promotion, and an
optional coalescing window waits from the oldest ready timestamp without
looking ahead to future arrivals. This makes three tradeoffs measurable:
deadline urgency, fairness to old rows, and batching delay.

The controller is intentionally myopic. Its service estimate is linear in row
width, it does not forecast arrivals or recovery cohorts, and its fairness
escape can promote a very wide old row ahead of a newly arrived tight-deadline
row. Those are baseline limitations rather than bugs.

## Reported evidence

Every `BaselineResult` retains per-launch target and draft records plus:

- real and effective mean target batch size;
- verifier and padded slot counts;
- homogeneous batches and realignment fallbacks;
- total and maximum ready-queue wait;
- requests crossing a configurable starvation threshold;
- deadline misses;
- ordinary and parallel mode counts;
- speculative and normal draft jobs; and
- maximum normal draft-job wait.

`tests/test_baselines.py` contains deterministic counterexamples for:

- a fixed SPECTRE rollback threshold choosing parallel mode for a very wide
  padded row whose measured completion is worse than ordinary mode;
- EXSpec homogeneous grouping delaying an old unique-length sequence; and
- a fairness-promoted wide row causing a new tight row to miss its deadline,
  plus myopic coalescing that waits despite no future arrival.

These traces are adversarial diagnostics. They establish that the mechanisms
are executable and falsifiable; they are not empirical performance claims about
the original systems.
