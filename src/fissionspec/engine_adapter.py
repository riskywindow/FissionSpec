"""Typed CPU-only harness for the frozen serving-engine callback seam.

This module adapts :class:`~fissionspec.coordinator.InMemoryCoordinator` to
engine-shaped descriptors without importing SGLang, vLLM, CUDA, or a tensor
library.  It is executable integration scaffolding, not a production engine
integration.

The four state-changing/selection operations intentionally mirror the frozen
seam:

* :meth:`MockEngineAdapter.next_batch`;
* :meth:`MockEngineAdapter.reserve`;
* :meth:`MockEngineAdapter.verify`; and
* :meth:`MockEngineAdapter.recovery`.

``PhysicalBatch`` reports useful and physical rows/slots separately.  Every
graph-bucket row absent from the logical batch is an explicit padding row and
must consume at least one explicit padding slot; a backend cannot hide padded
work behind a logical row count.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeAlias, runtime_checkable

from fissionspec.coordinator import (
    BlockTableDescriptor,
    CallbackResult,
    InMemoryCoordinator,
    InvalidCoordinatorOperation,
    PhysicalPageRef,
    SchedulerLane,
    VerificationDispatch,
)
from fissionspec.ledger import LedgerEpoch
from fissionspec.protocol import (
    MessageTag,
    ProtocolRequestId,
    RecoveryReply,
    RecoveryRequest,
    VerifyReply,
)


class EngineAdapterError(RuntimeError):
    """Base class for mock engine seam failures."""


class GraphBucketUnavailableError(EngineAdapterError, ValueError):
    """No declared physical graph bucket can carry a logical batch."""


class EngineAdapterInvariantError(EngineAdapterError):
    """Logical coordinator state and physical adapter state disagree."""


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, order=True, slots=True)
class GraphBucket:
    """One declared physical launch shape.

    ``rows`` is the CUDA-graph row dimension and ``verifier_slots`` is the
    total flattened token-slot dimension.  The null backend only accounts for
    these integers; it does not assert that a real engine exposes this shape.
    """

    rows: int
    verifier_slots: int
    name: str = ""

    def __post_init__(self) -> None:
        _positive_int(self.rows, field="bucket rows")
        _positive_int(self.verifier_slots, field="bucket verifier_slots")
        if self.verifier_slots < self.rows:
            raise ValueError("a graph bucket needs at least one verifier slot per row")
        if not isinstance(self.name, str):
            raise TypeError("bucket name must be a string")


@dataclass(frozen=True, slots=True)
class FakeBlockDescriptor:
    """Engine-shaped, tensor-free copy of a coordinator block table."""

    request_id: ProtocolRequestId
    tag: MessageTag
    epoch: LedgerEpoch
    base_committed_blocks: int
    reserved_blocks: int
    pages: tuple[PhysicalPageRef, ...]

    def __post_init__(self) -> None:
        if self.tag.request_id != self.request_id:
            raise ValueError("descriptor request_id must match its version tag")
        if self.epoch.request_id != self.request_id:
            raise ValueError("descriptor request_id must match its ledger epoch")
        if self.epoch.round_id != self.tag.round_id:
            raise ValueError("protocol and ledger rounds must match")
        _non_negative_int(self.base_committed_blocks, field="base_committed_blocks")
        _positive_int(self.reserved_blocks, field="reserved_blocks")
        if not self.pages:
            raise ValueError("a reserved descriptor must name at least one physical page")
        if not all(isinstance(page, PhysicalPageRef) for page in self.pages):
            raise TypeError("descriptor pages must be PhysicalPageRef values")

    @classmethod
    def from_dispatch(cls, dispatch: VerificationDispatch) -> FakeBlockDescriptor:
        """Copy the immutable engine-facing portion of a reservation."""

        if not isinstance(dispatch, VerificationDispatch):
            raise TypeError("dispatch must be a VerificationDispatch")
        table = dispatch.block_table
        return cls(
            request_id=dispatch.request.tag.request_id,
            tag=dispatch.request.tag,
            epoch=table.epoch,
            base_committed_blocks=table.base_committed_blocks,
            reserved_blocks=table.reserved_blocks,
            pages=table.pages,
        )

    def matches(self, table: BlockTableDescriptor) -> bool:
        """Return whether this copy still names exactly the active table."""

        return (
            self.epoch == table.epoch
            and self.base_committed_blocks == table.base_committed_blocks
            and self.reserved_blocks == table.reserved_blocks
            and self.pages == table.pages
        )


@dataclass(frozen=True, slots=True)
class ReservedEngineRow:
    """One verifier row reserved at an exact protocol/ledger version."""

    descriptor: FakeBlockDescriptor
    verifier_slots: int

    def __post_init__(self) -> None:
        _positive_int(self.verifier_slots, field="row verifier_slots")
        if self.verifier_slots != self.descriptor.reserved_blocks:
            raise ValueError("the mock seam maps one verifier slot to each reserved block")

    @property
    def request_id(self) -> ProtocolRequestId:
        return self.descriptor.request_id

    @property
    def tag(self) -> MessageTag:
        return self.descriptor.tag


@dataclass(frozen=True, slots=True)
class PhysicalBatch:
    """A logical verifier batch placed into one declared physical bucket."""

    rows: tuple[ReservedEngineRow, ...]
    bucket: GraphBucket

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("a physical batch must contain at least one logical row")
        if not all(isinstance(row, ReservedEngineRow) for row in self.rows):
            raise TypeError("batch rows must be ReservedEngineRow values")
        if not isinstance(self.bucket, GraphBucket):
            raise TypeError("bucket must be a GraphBucket")
        request_ids = tuple(row.request_id for row in self.rows)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("a request may appear only once in a physical batch")
        if self.bucket.rows < self.logical_rows:
            raise ValueError("physical bucket has fewer rows than the logical batch")
        # A padded physical row occupies a real one-token masked slot. Extra
        # token-axis graph slots are explicit padding too.
        minimum_slots = self.useful_verifier_slots + self.padding_rows
        if self.bucket.verifier_slots < minimum_slots:
            raise ValueError("physical bucket cannot account for every useful slot and padded row")

    @property
    def logical_rows(self) -> int:
        return len(self.rows)

    @property
    def useful_verifier_slots(self) -> int:
        return sum(row.verifier_slots for row in self.rows)

    @property
    def physical_rows(self) -> int:
        return self.bucket.rows

    @property
    def physical_verifier_slots(self) -> int:
        return self.bucket.verifier_slots

    @property
    def padding_rows(self) -> int:
        return self.physical_rows - self.logical_rows

    @property
    def padding_verifier_slots(self) -> int:
        return self.physical_verifier_slots - self.useful_verifier_slots

    def audit_accounting(self) -> None:
        """Raise if any physical work is absent from explicit accounting."""

        if self.physical_rows != self.logical_rows + self.padding_rows:
            raise EngineAdapterInvariantError("physical row accounting is not additive")
        if self.physical_verifier_slots != self.useful_verifier_slots + self.padding_verifier_slots:
            raise EngineAdapterInvariantError("physical slot accounting is not additive")
        if self.padding_verifier_slots < self.padding_rows:
            raise EngineAdapterInvariantError(
                "an explicit padding row is missing its physical verifier slot"
            )


@dataclass(frozen=True, slots=True)
class GraphBucketSet:
    """Deterministic smallest-fit selection over declared physical shapes."""

    buckets: tuple[GraphBucket, ...]

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError("at least one graph bucket is required")
        if not all(isinstance(bucket, GraphBucket) for bucket in self.buckets):
            raise TypeError("buckets must contain GraphBucket values")
        if len(self.buckets) != len(set(self.buckets)):
            raise ValueError("graph buckets must be unique")

    @classmethod
    def exact(cls, *, maximum_rows: int, maximum_width: int) -> GraphBucketSet:
        """Declare every exact flattened shape in a bounded test domain."""

        maximum_rows = _positive_int(maximum_rows, field="maximum_rows")
        maximum_width = _positive_int(maximum_width, field="maximum_width")
        return cls(
            tuple(
                GraphBucket(rows, slots, f"exact-r{rows}-s{slots}")
                for rows in range(1, maximum_rows + 1)
                for slots in range(rows, rows * maximum_width + 1)
            )
        )

    def choose(self, *, logical_rows: int, useful_verifier_slots: int) -> GraphBucket:
        logical_rows = _positive_int(logical_rows, field="logical_rows")
        useful_verifier_slots = _positive_int(useful_verifier_slots, field="useful_verifier_slots")
        candidates = []
        for bucket in self.buckets:
            padding_rows = bucket.rows - logical_rows
            if padding_rows < 0:
                continue
            if bucket.verifier_slots < useful_verifier_slots + padding_rows:
                continue
            candidates.append(bucket)
        if not candidates:
            raise GraphBucketUnavailableError(
                f"no graph bucket fits {logical_rows} rows and "
                f"{useful_verifier_slots} useful verifier slots"
            )
        return min(
            candidates,
            key=lambda bucket: (
                bucket.rows * bucket.verifier_slots,
                bucket.rows - logical_rows,
                bucket.verifier_slots - useful_verifier_slots,
                bucket.rows,
                bucket.verifier_slots,
                bucket.name,
            ),
        )


@runtime_checkable
class EngineBackend(Protocol):
    """Minimal verifier/recovery submission surface used by the adapter."""

    def submit_verify(self, batch: PhysicalBatch) -> None: ...

    def submit_recovery(self, request: RecoveryRequest) -> None: ...


class NullEngineBackend:
    """A backend that records descriptors and performs no model execution."""

    def __init__(self) -> None:
        self._verify_batches: list[PhysicalBatch] = []
        self._recoveries: list[RecoveryRequest] = []

    @property
    def verify_batches(self) -> tuple[PhysicalBatch, ...]:
        return tuple(self._verify_batches)

    @property
    def recoveries(self) -> tuple[RecoveryRequest, ...]:
        return tuple(self._recoveries)

    def submit_verify(self, batch: PhysicalBatch) -> None:
        if not isinstance(batch, PhysicalBatch):
            raise TypeError("batch must be a PhysicalBatch")
        batch.audit_accounting()
        self._verify_batches.append(batch)

    def submit_recovery(self, request: RecoveryRequest) -> None:
        if not isinstance(request, RecoveryRequest):
            raise TypeError("request must be a RecoveryRequest")
        self._recoveries.append(request)


class MockEngineAdapter:
    """Coordinator adapter that owns fake descriptors and physical accounting."""

    def __init__(
        self,
        coordinator: InMemoryCoordinator,
        buckets: GraphBucketSet,
        backend: EngineBackend,
    ) -> None:
        if not isinstance(coordinator, InMemoryCoordinator):
            raise TypeError("coordinator must be an InMemoryCoordinator")
        if not isinstance(buckets, GraphBucketSet):
            raise TypeError("buckets must be a GraphBucketSet")
        if not isinstance(backend, EngineBackend):
            raise TypeError("backend must implement EngineBackend")
        self._coordinator = coordinator
        self._buckets = buckets
        self._backend = backend
        self._active: dict[MessageTag, ReservedEngineRow] = {}
        self._submitted: set[MessageTag] = set()
        self.audit()

    @property
    def coordinator(self) -> InMemoryCoordinator:
        return self._coordinator

    @property
    def buckets(self) -> GraphBucketSet:
        return self._buckets

    @property
    def active_rows(self) -> tuple[ReservedEngineRow, ...]:
        return tuple(
            sorted(
                self._active.values(),
                key=lambda row: (
                    type(row.request_id).__name__,
                    repr(row.request_id),
                    row.tag.round_id,
                    row.tag.version,
                ),
            )
        )

    def register_request(
        self,
        request_id: ProtocolRequestId,
        total_blocks: int,
        *,
        committed_blocks: int = 0,
    ) -> None:
        self._coordinator.register_request(
            request_id,
            total_blocks,
            committed_blocks=committed_blocks,
        )
        self.audit()

    def next_batch(self, max_requests: int | None = None) -> tuple[ProtocolRequestId, ...]:
        """Return logical target-ready rows; recovering rows are absent."""

        return self._coordinator.next_batch(max_requests)

    def reserve(
        self,
        request_id: ProtocolRequestId,
        verification_width: int,
    ) -> ReservedEngineRow:
        """Reserve a version-fenced fake block descriptor for one verifier row."""

        verification_width = _positive_int(verification_width, field="verification_width")
        # Reject unsupported graph shapes before changing coordinator state.
        self._buckets.choose(logical_rows=1, useful_verifier_slots=verification_width)
        dispatch = self._coordinator.reserve(request_id, verification_width)
        row = ReservedEngineRow(
            descriptor=FakeBlockDescriptor.from_dispatch(dispatch),
            verifier_slots=verification_width,
        )
        if row.tag in self._active:
            raise EngineAdapterInvariantError("coordinator reused an active protocol tag")
        self._active[row.tag] = row
        self.audit()
        return row

    def submit_verify(self, rows: Sequence[ReservedEngineRow]) -> PhysicalBatch:
        """Place reserved rows in the smallest fitting bucket and record launch."""

        selected = tuple(rows)
        if not selected:
            raise ValueError("at least one reserved row is required")
        for row in selected:
            if not isinstance(row, ReservedEngineRow):
                raise TypeError("rows must contain ReservedEngineRow values")
            if self._active.get(row.tag) != row:
                raise InvalidCoordinatorOperation(
                    f"row {row.request_id!r} is not the active exact reservation"
                )
            if row.tag in self._submitted:
                raise InvalidCoordinatorOperation(f"row {row.request_id!r} was already submitted")
        bucket = self._buckets.choose(
            logical_rows=len(selected),
            useful_verifier_slots=sum(row.verifier_slots for row in selected),
        )
        batch = PhysicalBatch(selected, bucket)
        self._backend.submit_verify(batch)
        self._submitted.update(row.tag for row in selected)
        self.audit()
        return batch

    def verify(self, reply: VerifyReply) -> CallbackResult:
        """Apply an exact verifier callback; duplicates and stale tags are inert."""

        if not isinstance(reply, VerifyReply):
            raise TypeError("reply must be a VerifyReply")
        if reply.tag in self._active and reply.tag not in self._submitted:
            raise InvalidCoordinatorOperation(
                "an exact active verifier callback arrived before physical submission"
            )
        result = self._coordinator.on_verify_complete(reply)
        if result.applied:
            self._active.pop(reply.tag, None)
            self._submitted.discard(reply.tag)
            if result.outbound is not None:
                self._backend.submit_recovery(result.outbound)
        self.audit()
        return result

    def recovery(self, reply: RecoveryReply) -> CallbackResult:
        """Apply an exact recovery callback and re-admit or finish the row."""

        if not isinstance(reply, RecoveryReply):
            raise TypeError("reply must be a RecoveryReply")
        result = self._coordinator.on_recovery_complete(reply)
        self.audit()
        return result

    def cancel(self, request_id: ProtocolRequestId) -> bool:
        """Fence a request, release its pages, and forget fake descriptors."""

        cancelled = self._coordinator.cancel(request_id)
        if cancelled:
            stale_tags = [tag for tag in self._active if tag.request_id == request_id]
            for tag in stale_tags:
                del self._active[tag]
                self._submitted.discard(tag)
        self.audit()
        return cancelled

    def audit(self) -> None:
        """Prove adapter/coordinator binding and physical accounting invariants."""

        self._coordinator.audit()
        verifying_tags: set[MessageTag] = set()
        for view in self._coordinator.requests():
            if view.lane is not SchedulerLane.VERIFYING:
                continue
            if view.active_tag is None:
                raise EngineAdapterInvariantError("verifying coordinator row has no tag")
            verifying_tags.add(view.active_tag)
        if verifying_tags != set(self._active):
            raise EngineAdapterInvariantError(
                "adapter reservations are not an exact cover of coordinator verifier rows"
            )
        for tag, row in self._active.items():
            if row.tag != tag:
                raise EngineAdapterInvariantError("active row is indexed under another tag")
            view = self._coordinator.request(row.request_id)
            if (
                view.lane is not SchedulerLane.VERIFYING
                or view.active_tag != tag
                or view.block_table is None
                or not row.descriptor.matches(view.block_table)
            ):
                raise EngineAdapterInvariantError(
                    "active fake descriptor does not match coordinator state"
                )
        for tag in self._submitted:
            if tag not in self._active:
                raise EngineAdapterInvariantError("submitted tag is no longer an active row")


@dataclass(frozen=True, slots=True)
class VerifierOutcome:
    """One deterministic fake verifier result used by bounded liveness runs."""

    hit: bool
    accepted_blocks: int

    def __post_init__(self) -> None:
        if not isinstance(self.hit, bool):
            raise TypeError("hit must be a bool")
        _non_negative_int(self.accepted_blocks, field="accepted_blocks")


class CallbackKind(StrEnum):
    VERIFY = "verify"
    RECOVERY = "recovery"


Reply: TypeAlias = VerifyReply | RecoveryReply


@dataclass(frozen=True, slots=True)
class QueuedCallback:
    """One transport callback retained for drop/duplicate/reorder injection."""

    kind: CallbackKind
    reply: Reply

    def __post_init__(self) -> None:
        if self.kind is CallbackKind.VERIFY and not isinstance(self.reply, VerifyReply):
            raise TypeError("verify callback must carry VerifyReply")
        if self.kind is CallbackKind.RECOVERY and not isinstance(self.reply, RecoveryReply):
            raise TypeError("recovery callback must carry RecoveryReply")


class MockEngineHarness:
    """Deterministic transport and fault injector around :class:`MockEngineAdapter`."""

    def __init__(self, adapter: MockEngineAdapter) -> None:
        if not isinstance(adapter, MockEngineAdapter):
            raise TypeError("adapter must be a MockEngineAdapter")
        self._adapter = adapter
        self._pending: list[QueuedCallback] = []

    @property
    def adapter(self) -> MockEngineAdapter:
        return self._adapter

    @property
    def pending(self) -> tuple[QueuedCallback, ...]:
        return tuple(self._pending)

    @staticmethod
    def stale_tag(tag: MessageTag, *, version_delta: int = 1) -> MessageTag:
        version_delta = _positive_int(version_delta, field="version_delta")
        return MessageTag(tag.request_id, tag.round_id, tag.version + version_delta)

    def enqueue_verify(
        self,
        row: ReservedEngineRow,
        outcome: VerifierOutcome,
        *,
        tag: MessageTag | None = None,
    ) -> QueuedCallback:
        if not isinstance(row, ReservedEngineRow):
            raise TypeError("row must be a ReservedEngineRow")
        if not isinstance(outcome, VerifierOutcome):
            raise TypeError("outcome must be a VerifierOutcome")
        callback = QueuedCallback(
            CallbackKind.VERIFY,
            VerifyReply(
                row.tag if tag is None else tag,
                hit=outcome.hit,
                accepted_blocks=outcome.accepted_blocks,
            ),
        )
        self._pending.append(callback)
        return callback

    def enqueue_recovery(
        self,
        request: RecoveryRequest,
        *,
        finished: bool,
        recovered_blocks: int = 0,
        tag: MessageTag | None = None,
    ) -> QueuedCallback:
        if not isinstance(request, RecoveryRequest):
            raise TypeError("request must be a RecoveryRequest")
        callback = QueuedCallback(
            CallbackKind.RECOVERY,
            RecoveryReply(
                request.tag if tag is None else tag,
                finished=finished,
                recovered_blocks=recovered_blocks,
            ),
        )
        self._pending.append(callback)
        return callback

    def duplicate(self, index: int = 0) -> None:
        index = self._queue_index(index)
        self._pending.append(self._pending[index])

    def drop(self, index: int = 0) -> QueuedCallback:
        return self._pending.pop(self._queue_index(index))

    def reorder(self, order: Sequence[int]) -> None:
        if len(order) != len(self._pending):
            raise ValueError("reorder must name every queued callback")
        indices = [_non_negative_int(index, field="reorder index") for index in order]
        if sorted(indices) != list(range(len(self._pending))):
            raise ValueError("reorder indices must be a permutation")
        self._pending = [self._pending[index] for index in indices]

    def deliver(self, index: int = 0) -> CallbackResult:
        callback = self.drop(index)
        if callback.kind is CallbackKind.VERIFY:
            assert isinstance(callback.reply, VerifyReply)
            return self._adapter.verify(callback.reply)
        assert isinstance(callback.reply, RecoveryReply)
        return self._adapter.recovery(callback.reply)

    def deliver_all(self) -> tuple[CallbackResult, ...]:
        results = []
        while self._pending:
            results.append(self.deliver())
        return tuple(results)

    def drive_to_terminal(
        self,
        request_id: ProtocolRequestId,
        outcomes: Sequence[VerifierOutcome],
        *,
        verification_width: int = 1,
        maximum_steps: int = 64,
    ) -> SchedulerLane:
        """Drive a closed request to ``FINISHED`` or fail a bounded liveness check."""

        width = _positive_int(verification_width, field="verification_width")
        maximum_steps = _positive_int(maximum_steps, field="maximum_steps")
        outcome_index = 0
        for _ in range(maximum_steps):
            view = self._adapter.coordinator.request(request_id)
            if view.lane in {SchedulerLane.FINISHED, SchedulerLane.CANCELLED}:
                self._adapter.audit()
                return view.lane
            if view.lane in {
                SchedulerLane.READY,
                SchedulerLane.READY_HIT,
                SchedulerLane.READY_BACKUP,
            }:
                remaining = view.total_blocks - view.committed_blocks
                row = self._adapter.reserve(request_id, min(width, remaining))
                self._adapter.submit_verify((row,))
                if outcome_index < len(outcomes):
                    outcome = outcomes[outcome_index]
                    outcome_index += 1
                else:
                    outcome = VerifierOutcome(True, row.verifier_slots)
                self.enqueue_verify(row, outcome)
                self.deliver()
            elif view.lane is SchedulerLane.RECOVERING:
                recovery = self._adapter.coordinator.latest_recovery(request_id)
                if recovery is None:
                    raise EngineAdapterInvariantError("recovering request has no recovery command")
                finished = view.committed_blocks == view.total_blocks
                self.enqueue_recovery(recovery, finished=finished)
                self.deliver()
            else:
                raise EngineAdapterInvariantError(
                    f"cannot drive request {request_id!r} from lane {view.lane.value}"
                )
        raise EngineAdapterInvariantError(
            f"request {request_id!r} did not reach a terminal lane in {maximum_steps} steps"
        )

    def _queue_index(self, index: int) -> int:
        index = _non_negative_int(index, field="queue index")
        if index >= len(self._pending):
            raise IndexError(f"queue index {index} is outside length {len(self._pending)}")
        return index


def default_null_harness(
    *,
    page_count: int = 64,
    page_size: int = 4,
    buckets: Sequence[GraphBucket] = (
        GraphBucket(1, 8, "r1-s8"),
        GraphBucket(2, 16, "r2-s16"),
        GraphBucket(4, 32, "r4-s32"),
    ),
) -> tuple[MockEngineHarness, NullEngineBackend]:
    """Construct a CPU-only harness with explicit physical graph buckets."""

    backend = NullEngineBackend()
    adapter = MockEngineAdapter(
        InMemoryCoordinator(page_count, page_size),
        GraphBucketSet(tuple(buckets)),
        backend,
    )
    return MockEngineHarness(adapter), backend


def reserve_ready_rows(
    adapter: MockEngineAdapter,
    widths: Mapping[ProtocolRequestId, int],
    *,
    max_requests: int | None = None,
) -> tuple[ReservedEngineRow, ...]:
    """Reserve the currently ready rows named in an explicit width mapping.

    This convenience is intentionally not atomic: each successful reservation
    is visible immediately, matching per-request coordinator ownership.  A
    caller that requires all-or-none multirow admission must preflight its
    engine allocator in a production adapter.
    """

    rows = []
    for request_id in adapter.next_batch(max_requests):
        try:
            width = widths[request_id]
        except KeyError as exc:
            raise ValueError(f"missing verification width for {request_id!r}") from exc
        rows.append(adapter.reserve(request_id, width))
    return tuple(rows)


__all__ = [
    "CallbackKind",
    "EngineAdapterError",
    "EngineAdapterInvariantError",
    "EngineBackend",
    "FakeBlockDescriptor",
    "GraphBucket",
    "GraphBucketSet",
    "GraphBucketUnavailableError",
    "MockEngineAdapter",
    "MockEngineHarness",
    "NullEngineBackend",
    "PhysicalBatch",
    "QueuedCallback",
    "ReservedEngineRow",
    "VerifierOutcome",
    "default_null_harness",
    "reserve_ready_rows",
]
