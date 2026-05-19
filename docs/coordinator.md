# Composed coordinator and crash contract

`fissionspec.coordinator.InMemoryCoordinator` binds three independently tested
layers into one executable CPU reference:

1. `FissionProtocol` owns request round/version tags and async reply fencing;
2. `SpeculativeLedger` owns committed and provisional page spans; and
3. the coordinator owns lane admission, exact tag-to-ledger-epoch bindings, and
   the recovery outbox.

It is an engine-neutral correctness model. Page references identify an
allocator pool, page, and generation; they do not contain tensor bytes.

## Atomic publication point

A durable coordinator snapshot is the publication unit. The canonical envelope
contains the complete protocol, ledger, active block-table binding, version
high-water marks, and request metadata under one SHA-256 checksum. A production
implementation must publish the equivalent record with one database
transaction, write-ahead-log entry, or compare-and-swap—not persist the ledger
and protocol independently.

Callbacks mutate the in-memory composition while one coordinator lock is held.
The injected crash model deliberately allows failure:

- before reservation;
- after provisional pages are staged;
- after the verifier tag is published;
- before or after a ledger prefix commit;
- after the protocol transition;
- before or after recovery application; and
- before or after cancellation.

After a simulated process death, the object is discarded. Restoration begins
from the last durable snapshot. Any verifier or recovery that was in flight at
that snapshot is aborted/fenced to a strictly newer protocol version and
reissued through recovery. Thus a reply from the lost address space cannot
mutate restored state.

This is an at-least-once transport with exactly-once *state effect*. It does not
claim that a remote worker executes only once.

## Callback rules

Every callback must match the active `(request_id, round_id, version)` exactly.
Unknown, duplicate, reordered, and stale replies are inert. A matching verifier
reply:

1. validates that its accepted prefix fits the reserved branch;
2. commits that prefix and aborts provisional siblings/tails;
3. applies the hit/miss protocol transition; and
4. either admits a hit, publishes the exact recovery command, or finishes.

A matching recovery reply may not claim completion unless the
target-authorized committed extent is already complete. Recovery never commits
target tokens by itself.

OOM is preflighted before either version space advances. Cancellation aborts an
active transaction, fences callbacks, removes the outbox command, and drops all
owned pages.

## Cross-layer audit

`audit()` checks:

- monotone protocol high-water marks;
- equality of coordinator and ledger committed counts;
- one active ledger branch for every verifying request;
- exact block-table page/generation equality with that branch;
- tag round equality with ledger epoch round;
- no active transaction outside the verifier lane;
- one exact recovery command in every recovery lane;
- complete requests only in finished/closing-recovery state;
- no pages, outbox entries, or transactions for cancelled requests; and
- no ledger/outbox request unknown to the coordinator.

Canonical decoding rejects duplicate JSON keys, NaN/Infinity, unknown or
missing fields, checksum changes, alternate whitespace, identifier type
confusion (`1` versus `"1"`), and invalid nested ledger snapshots.

## Deliberate boundary

The snapshot API models durable state but does not provide a storage backend,
replicated consensus, fsync policy, or distributed lease. The fake target and
remote drafter execute callback ordering/fault semantics only. A production
engine must additionally prove that its block-table publication and kernel
completion event obey the same atomic boundary.
