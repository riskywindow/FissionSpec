# SGLang integration contract

This directory is a patch map, not a claim that the laptop artifact executes
SGLang kernels. The natural first production integration is SGLang because
SPECTRE already supplies a remote-drafter transport and chain-verification
path. FissionSpec changes the target scheduler boundary after rejection
sampling; it does not require a new verifier.

## Required scheduler state

Attach the following compact record to each decode request:

```text
FissionKey {
    request_id: u64,
    round_id: u64,
    version: u64,
    lane: READY_HIT | VERIFYING | RECOVERING | READY_BACKUP | BYPASS,
    ready_since_ns: u64,
    recovery_eta_ns: Option<u64>,
    tbt_deadline_ns: u64,
    speculation_length: u16,
    kv_transaction: TransactionStamp,
}
```

The tuple `(request_id, round_id, version)` is carried across every asynchronous
draft message. A completion with any mismatched field is telemetry only; it may
not mutate tokens, KV descriptors, or queue membership.

Lane ownership is exclusive. The scheduler atomically changes a selected row
from `READY_HIT` or `READY_BACKUP` to `VERIFYING` before constructing runner
inputs; a second queue cannot select that version. Target completion is the
only normal transition out of `VERIFYING`. Cancellation increments `version`
before removing the row, so an already-launched kernel may finish but cannot
publish state.

## Patch boundary

1. Scatter target verification results into per-request outcome events.
2. While verification is in flight, prepare outcome-indexed next-round
   branches against its captured epoch. On target completion, commit the
   accepted target prefix, atomically select/rebase the matching prepared
   branch if present, and only then expose the new version to another lane.
3. Put reusable continuations into `READY_HIT`. Put cache misses into
   `RECOVERING` and omit their rows from the next target input.
4. Feed target-ready rows plus recovery ETAs into the horizon-2 controller.
5. On recovery completion, validate the key, atomically publish its block-table
   descriptor, and enqueue `READY_BACKUP`; never copy committed KV merely to
   change lane.
6. Build the target input from only the rows selected by the controller. A
   `BYPASS` row carries one real target token and no speculative suffix.

The atomic publication point is the block-table pointer swap, not receipt of a
network message. GPU work launched from an old descriptor may finish, but its
version fence prevents it from committing.

## CUDA-graph and packed-kernel caveat

Removing a logical row is useful only if it reduces the physical work or frees
a graph slot for another ready request. Instrument all three quantities:

```text
logical_rows, real_verifier_tokens, graph_bucket_tokens
```

If SGLang's chosen attention backend packs variable sequence lengths, padding
may be nearly free. If CUDA-graph bucketing rounds both shapes to the same
capture, fission may save no target time. The controller's profile must use the
measured graph-bucket latency surface; `verifier_slot_ms=0` is the required
null hypothesis.

## Backpressure and liveness

- Recovery queues are bounded per remote drafter.
- A request that exhausts its refusion wait, misses its TBT deadline guard, or
  trips the remote circuit breaker enters `BYPASS`.
- At least one ready request dispatches whenever no admissible future recovery
  can lower weighted flow cost; coalescing cannot extend its own timer.
- Cancellation increments `version` before reclaiming provisional pages.
- A recovered request retains its original arrival and last-token timestamps,
  so leaving the target batch does not erase its scheduling age.

## Acceptance gates

Do not report an end-to-end result until the integration passes:

- byte-identical greedy tokens against non-speculative decoding;
- a paired sampled-output distribution test with counter-addressed RNG;
- delayed, duplicated, and reordered reply fault injection;
- page-owner audits after every forced cancellation and OOM path;
- `verifier_slot_ms=0` and no-fission ablations;
- one-miss-plus-`B-1`-hits traces at every graph bucket boundary; and
- target- and draft-GPU-normalized goodput under matched resources.

The reference Rust crate supplies the allocation-free decision primitive. The
Python protocol and ledger are deliberately more diagnostic and should remain
an independent invariant oracle during integration; they are not semantic ports
of the flattened-work Rust controller.
