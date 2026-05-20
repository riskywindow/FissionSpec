# CPU-only mock engine adapter

`fissionspec.engine_adapter` is executable scaffolding for the frozen
`next_batch / reserve / verify / recovery` seam. It wraps the real composed
coordinator and page ledger, but its backend is deliberately null: no
SGLang/vLLM import, tensor allocation, model execution, CUDA graph, RPC, or GPU
measurement occurs.

The adapter is useful before paying for engine integration because it makes
descriptor and callback mistakes reproducible on CPU.

## Typed seam

- `next_batch(max_requests)` returns only coordinator rows in a target-ready
  lane. Recovering, finished, and cancelled rows are absent.
- `reserve(request_id, verification_width)` returns a
  `ReservedEngineRow`. Its `FakeBlockDescriptor` binds the request,
  `(round, protocol version)` tag, ledger epoch, committed-prefix base,
  reserved width, and ABA-safe physical page coordinates.
- `submit_verify(rows)` places exact active reservations into the smallest
  declared `GraphBucket` and hands the immutable `PhysicalBatch` to an
  `EngineBackend`.
- `verify(reply)` and `recovery(reply)` pass versioned callbacks through the
  composed coordinator. Duplicate, delayed, and stale callbacks are inert.
- `cancel(request_id)` fences callbacks and releases every provisional and
  committed page owned by the request.

`NullEngineBackend` records verifier batches and recovery commands. It never
fabricates kernel outputs. `MockEngineHarness` queues explicit `VerifyReply`
and `RecoveryReply` values so tests can drop, duplicate, reorder, stale, or
delay them.

## Physical accounting

A graph bucket declares physical `rows` and flattened physical
`verifier_slots`. A `PhysicalBatch` exposes:

```text
physical_rows = logical_rows + padding_rows
physical_slots = useful_slots + padding_slots
padding_slots >= padding_rows
```

The final inequality requires every absent logical row in a larger graph bucket
to occupy at least one explicit masked physical slot. Bucket selection rejects
a shape that has enough flattened slots for useful work but no slot left for a
padded row. Thus trace consumers cannot report a logical two-row launch while
silently charging a four-row graph elsewhere.

These are accounting invariants, not a claim about any particular engine's
graph layout. A production adapter must replace the shape model with measured
engine-native row, token, and graph-bucket semantics.

## CPU fault and state-space coverage

Run the focused suite with:

```bash
PYTHONPATH=src python3 -m unittest tests.test_engine_adapter -v
```

The tests cover:

- stale, duplicated, dropped, and reordered verifier/recovery transport;
- exactly-once committed-prefix effects;
- stable OOM preflight and callback fencing after cancellation;
- two-request ownership under callback reordering;
- every physical bucket up to four rows by 16 slots for all 39 logical
  row/width combinations with one to three rows and widths;
- 162 closed-cohort traces: page sizes `{1, 2}`, total blocks `{1, 2, 3}`,
  and every length-three prefix over hit-progress, miss-progress, and
  miss-without-progress outcomes.

Every enumerated trace must reach `FINISHED` under the bounded driver, retain
exactly `ceil(total_blocks / page_size)` committed pages, leave no transport
callbacks queued, pass coordinator/ledger ownership audits, and pass physical
batch accounting.

## Omitted production behavior

The harness does not validate transformer numerics, KV tensor addresses,
stream/event synchronization, CUDA graph capture, allocator integration,
cross-process durability, RPC retry policy, real outcome-cache keys, distributed
draft workers, scheduler lock contention, or measured latency. It is not an
SGLang or vLLM integration and should never be cited as one. Those remain
engine- and GPU-dependent acceptance gates.
