"""Composed in-memory runtime coordinator for FissionSpec.

The protocol state machine and the speculative page ledger deliberately have
independent version counters.  This module is the narrow integration layer
that binds them: every verifier dispatch records an explicit
``MessageTag -> LedgerEpoch`` association, every accepted prefix is committed
exactly once, and scheduler lanes are derived from the protocol state.

The coordinator is also a small executable fault model.  A caller may inject a
crash at named points, discard the process-local object, restore the last
durable snapshot, and replay the operation.  Snapshots are canonical JSON
envelopes protected by SHA-256; in-flight work is fenced to a fresh recovery
version on crash-resume.

This remains an engine-neutral reference.  ``BlockTableDescriptor`` contains
ABA-safe page coordinates a backend can translate to its native block table;
it never owns tensor bytes or executes kernels.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypeAlias, cast

from fissionspec.ledger import (
    AllocatorSnapshot,
    BranchSnapshot,
    FixedPageAllocator,
    InvariantViolation,
    LedgerEpoch,
    LedgerSnapshot,
    OutOfPagesError,
    PageHandle,
    PageSpan,
    PoolIdentity,
    RequestSnapshot,
    SpeculativeLedger,
    TransactionSnapshot,
)
from fissionspec.protocol import (
    FissionProtocol,
    MessageTag,
    ProtocolRequestId,
    ProtocolSnapshot,
    ProtocolState,
    RecoveryReply,
    RecoveryRequest,
    ReplyDisposition,
    VerifyReply,
    VerifyRequest,
)

_SNAPSHOT_VERSION: Final[int] = 1
_OUTCOME_ID: Final[str] = "target-prefix"

FaultHook: TypeAlias = Callable[["FaultPoint", ProtocolRequestId], None]
JsonObject: TypeAlias = dict[str, object]


class CoordinatorError(RuntimeError):
    """Base class for composed-runtime failures."""


class UnknownRequestError(CoordinatorError, KeyError):
    """Raised when a local command names no coordinator request."""


class InvalidCoordinatorOperation(CoordinatorError, ValueError):
    """Raised when a local command would violate a composed invariant."""


class CoordinatorInvariantError(CoordinatorError):
    """Raised when protocol, ledger, and scheduler state disagree."""


class CoordinatorSnapshotError(CoordinatorError, ValueError):
    """Raised when a durable snapshot is malformed or fails its checksum."""


class CoordinatorCrash(CoordinatorError):
    """Exception a deterministic fault hook may raise to model process death."""


class FaultPoint(StrEnum):
    """Stable boundaries exposed to exhaustive fault-injection tests."""

    BEFORE_RESERVE = "before_reserve"
    AFTER_LEDGER_STAGE = "after_ledger_stage"
    AFTER_PROTOCOL_START = "after_protocol_start"
    BEFORE_VERIFY_COMMIT = "before_verify_commit"
    AFTER_LEDGER_COMMIT = "after_ledger_commit"
    AFTER_PROTOCOL_TRANSITION = "after_protocol_transition"
    BEFORE_RECOVERY_APPLY = "before_recovery_apply"
    AFTER_RECOVERY_APPLY = "after_recovery_apply"
    BEFORE_CANCEL = "before_cancel"
    AFTER_CANCEL = "after_cancel"


class SchedulerLane(StrEnum):
    """Coordinator-visible execution lanes."""

    READY = "ready"
    VERIFYING = "verifying"
    READY_HIT = "ready_hit"
    RECOVERING = "recovering"
    READY_BACKUP = "ready_backup"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class PhysicalPageRef:
    """Pool-local page coordinates suitable for a native block-table lookup."""

    page_id: int
    generation: int

    def __post_init__(self) -> None:
        _plain_int(self.page_id, field="page_id")
        _plain_int(self.generation, field="generation", minimum=1)


@dataclass(frozen=True, slots=True)
class BlockTableDescriptor:
    """The provisional target-prefix pages reserved for one verification."""

    epoch: LedgerEpoch
    base_committed_blocks: int
    reserved_blocks: int
    pages: tuple[PhysicalPageRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.epoch, LedgerEpoch):
            raise InvalidCoordinatorOperation("epoch must be a LedgerEpoch")
        _plain_int(self.base_committed_blocks, field="base_committed_blocks")
        _plain_int(self.reserved_blocks, field="reserved_blocks", minimum=1)
        if not all(isinstance(page, PhysicalPageRef) for page in self.pages):
            raise InvalidCoordinatorOperation("pages must contain PhysicalPageRef values")


@dataclass(frozen=True, slots=True)
class VerificationDispatch:
    """Verifier command plus the exact provisional block table it may write."""

    request: VerifyRequest
    block_table: BlockTableDescriptor


@dataclass(frozen=True, slots=True)
class CallbackResult:
    """Result of an asynchronous verify or recovery callback."""

    disposition: ReplyDisposition
    request_id: ProtocolRequestId
    lane: SchedulerLane | None
    committed_blocks: int | None
    outbound: RecoveryRequest | None = None

    @property
    def applied(self) -> bool:
        return self.disposition is ReplyDisposition.APPLIED

    @property
    def ignored(self) -> bool:
        return not self.applied


@dataclass(frozen=True, slots=True)
class CoordinatorRequestView:
    """Immutable composed state for diagnostics and assertions."""

    request_id: ProtocolRequestId
    total_blocks: int
    committed_blocks: int
    lane: SchedulerLane
    round_id: int
    protocol_version: int
    active_tag: MessageTag | None
    active_epoch: LedgerEpoch | None
    block_table: BlockTableDescriptor | None
    cancelled: bool


@dataclass(slots=True)
class _RequestRecord:
    protocol: FissionProtocol
    total_blocks: int
    committed_blocks: int
    active_epoch: LedgerEpoch | None = None
    block_table: BlockTableDescriptor | None = None
    cancelled: bool = False
    protocol_high_water: int = 0


def _plain_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidCoordinatorOperation(f"{field} must be an integer")
    if value < minimum:
        raise InvalidCoordinatorOperation(f"{field} must be at least {minimum}")
    return value


def _request_id(value: object, *, field: str = "request_id") -> ProtocolRequestId:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise InvalidCoordinatorOperation(f"{field} must be a str or int (but not bool)")
    return value


def _id_sort_key(value: ProtocolRequestId) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _lane_for(record: _RequestRecord) -> SchedulerLane:
    if record.cancelled:
        return SchedulerLane.CANCELLED
    return SchedulerLane(record.protocol.state.value)


class InMemoryCoordinator:
    """Thread-safe composition of protocol, KV ledger, and scheduler lanes.

    The four engine-facing operations are:

    * :meth:`reserve` -- allocate a provisional target-prefix block table;
    * :meth:`on_verify_complete` -- atomically select and commit its prefix;
    * :meth:`on_recovery_complete` -- re-admit a recovered miss;
    * :meth:`next_batch` -- return currently target-ready request IDs.

    A fault hook runs while the coordinator lock is held.  If it raises, the
    object models a crashed address space and must be discarded rather than
    reused.  Restore :meth:`snapshot_bytes` captured before the operation.
    """

    _ledger: SpeculativeLedger
    _records: dict[ProtocolRequestId, _RequestRecord]
    _recovery_outbox: dict[ProtocolRequestId, RecoveryRequest]
    _fault_hook: FaultHook | None
    _lock: threading.RLock

    def __init__(
        self,
        page_count: int,
        page_size: int,
        *,
        fault_hook: FaultHook | None = None,
    ) -> None:
        self._ledger = SpeculativeLedger(FixedPageAllocator(page_count, page_size))
        self._records = {}
        self._recovery_outbox = {}
        self._fault_hook = fault_hook
        self._lock = threading.RLock()

    @property
    def ledger(self) -> SpeculativeLedger:
        """Return the ledger for read-only views and independent audits."""

        return self._ledger

    @property
    def allocator(self) -> FixedPageAllocator:
        """Return the exclusively owned allocator for diagnostics."""

        return self._ledger.allocator

    def set_fault_hook(self, hook: FaultHook | None) -> None:
        """Install or clear a deterministic crash hook."""

        if hook is not None and not callable(hook):
            raise TypeError("fault hook must be callable or None")
        with self._lock:
            self._fault_hook = hook

    def register_request(
        self,
        request_id: ProtocolRequestId,
        total_blocks: int,
        *,
        committed_blocks: int = 0,
    ) -> CoordinatorRequestView:
        """Register a request and materialize any already-committed prefix."""

        request_id = _request_id(request_id)
        total_blocks = _plain_int(total_blocks, field="total_blocks")
        committed_blocks = _plain_int(committed_blocks, field="committed_blocks")
        if committed_blocks > total_blocks:
            raise InvalidCoordinatorOperation("committed_blocks cannot exceed total_blocks")
        with self._lock:
            if request_id in self._records:
                raise InvalidCoordinatorOperation(f"request {request_id!r} already exists")
            self._ledger.register_request(request_id, committed_blocks)
            protocol = FissionProtocol(request_id)
            if committed_blocks == total_blocks:
                protocol.finish()
            record = _RequestRecord(
                protocol=protocol,
                total_blocks=total_blocks,
                committed_blocks=committed_blocks,
                protocol_high_water=protocol.version,
            )
            self._records[request_id] = record
            self._audit_unlocked()
            return self._view_unlocked(request_id, record)

    def request(self, request_id: ProtocolRequestId) -> CoordinatorRequestView:
        """Return the current composed view of one request."""

        request_id = _request_id(request_id)
        with self._lock:
            return self._view_unlocked(request_id, self._require_record_unlocked(request_id))

    def requests(self) -> tuple[CoordinatorRequestView, ...]:
        """Return all request views in deterministic identifier order."""

        with self._lock:
            return tuple(
                self._view_unlocked(request_id, self._records[request_id])
                for request_id in sorted(self._records, key=_id_sort_key)
            )

    def lane(self, request_id: ProtocolRequestId) -> SchedulerLane:
        """Return one request's scheduler lane."""

        return self.request(request_id).lane

    def next_batch(self, max_requests: int | None = None) -> tuple[ProtocolRequestId, ...]:
        """Select target-ready rows without including recovering or finished work."""

        if max_requests is not None:
            max_requests = _plain_int(max_requests, field="max_requests")
        with self._lock:
            ready = [
                request_id
                for request_id, record in self._records.items()
                if not record.cancelled and record.protocol.is_ready
            ]
            ready.sort(key=_id_sort_key)
            if max_requests is not None:
                ready = ready[:max_requests]
            return tuple(ready)

    def pending_recoveries(self) -> tuple[RecoveryRequest, ...]:
        """Return the latest recovery command for every recovering request."""

        with self._lock:
            return tuple(
                self._recovery_outbox[request_id]
                for request_id in sorted(self._recovery_outbox, key=_id_sort_key)
            )

    def latest_recovery(self, request_id: ProtocolRequestId) -> RecoveryRequest | None:
        """Return the latest recovery command, if the request is recovering."""

        request_id = _request_id(request_id)
        with self._lock:
            return self._recovery_outbox.get(request_id)

    def reserve(
        self, request_id: ProtocolRequestId, verification_width: int
    ) -> VerificationDispatch:
        """Reserve provisional pages and enter the verifier lane.

        Capacity is preflighted before either subsystem advances, making an OOM
        rejection a stable no-op.  Protocol and ledger versions are then bound
        explicitly in the returned descriptor.
        """

        request_id = _request_id(request_id)
        verification_width = _plain_int(verification_width, field="verification_width", minimum=1)
        with self._lock:
            record = self._require_record_unlocked(request_id)
            if record.cancelled or not record.protocol.is_ready:
                raise InvalidCoordinatorOperation(
                    f"cannot reserve request {request_id!r} from {_lane_for(record).value}"
                )
            remaining = record.total_blocks - record.committed_blocks
            if verification_width > remaining:
                raise InvalidCoordinatorOperation(
                    f"verification width {verification_width} exceeds {remaining} remaining blocks"
                )
            tail_used = record.committed_blocks % self._ledger.page_size
            private_extent = verification_width + (tail_used if tail_used else 0)
            required_pages = (private_extent + self._ledger.page_size - 1) // self._ledger.page_size
            if required_pages > self._ledger.allocator.free_count:
                raise OutOfPagesError(
                    f"reservation needs {required_pages} pages but only "
                    f"{self._ledger.allocator.free_count} are free"
                )

            self._fault(FaultPoint.BEFORE_RESERVE, request_id)
            round_id = record.protocol.round_id + 1
            epoch = self._ledger.begin(request_id, round_id)
            try:
                branch = self._ledger.stage_outcome(epoch, _OUTCOME_ID, verification_width)
            except Exception:
                self._ledger.abort(epoch)
                raise
            self._fault(FaultPoint.AFTER_LEDGER_STAGE, request_id)
            try:
                verify = record.protocol.start_verification(round_id)
            except Exception:
                self._ledger.abort(epoch)
                raise
            descriptor = BlockTableDescriptor(
                epoch=epoch,
                base_committed_blocks=record.committed_blocks,
                reserved_blocks=verification_width,
                pages=tuple(
                    PhysicalPageRef(span.handle.page_id, span.handle.generation)
                    for span in branch.pages
                ),
            )
            record.active_epoch = epoch
            record.block_table = descriptor
            record.protocol_high_water = max(record.protocol_high_water, record.protocol.version)
            self._fault(FaultPoint.AFTER_PROTOCOL_START, request_id)
            self._audit_unlocked()
            return VerificationDispatch(verify, descriptor)

    def on_verify_complete(self, reply: VerifyReply) -> CallbackResult:
        """Commit an exact in-flight verifier result, ignoring every stale replay."""

        if not isinstance(reply, VerifyReply):
            raise TypeError("expected a VerifyReply")
        with self._lock:
            record = self._records.get(reply.tag.request_id)
            if (
                record is None
                or record.cancelled
                or record.protocol.state is not ProtocolState.VERIFYING
                or record.protocol.active_tag != reply.tag
            ):
                return self._ignored_unlocked(reply.tag.request_id, record)
            descriptor = record.block_table
            epoch = record.active_epoch
            if descriptor is None or epoch is None:
                raise CoordinatorInvariantError(
                    "verifying protocol has no bound ledger transaction"
                )
            if reply.accepted_blocks > descriptor.reserved_blocks:
                raise InvalidCoordinatorOperation(
                    f"accepted prefix {reply.accepted_blocks} exceeds reserved "
                    f"width {descriptor.reserved_blocks}"
                )
            self._fault(FaultPoint.BEFORE_VERIFY_COMMIT, reply.tag.request_id)
            commit = self._ledger.commit(epoch, _OUTCOME_ID, reply.accepted_blocks)
            record.committed_blocks = commit.committed_blocks
            self._fault(FaultPoint.AFTER_LEDGER_COMMIT, reply.tag.request_id)

            transition = record.protocol.handle_verify_reply(reply)
            if not transition.applied:
                raise CoordinatorInvariantError(
                    "exact verifier tag became stale during a serialized callback"
                )
            record.active_epoch = None
            record.block_table = None
            if transition.outbound is not None:
                self._recovery_outbox[reply.tag.request_id] = transition.outbound
            else:
                self._recovery_outbox.pop(reply.tag.request_id, None)
                if record.committed_blocks == record.total_blocks:
                    record.protocol.finish()
            record.protocol_high_water = max(record.protocol_high_water, record.protocol.version)
            self._fault(FaultPoint.AFTER_PROTOCOL_TRANSITION, reply.tag.request_id)
            self._audit_unlocked()
            return CallbackResult(
                ReplyDisposition.APPLIED,
                reply.tag.request_id,
                _lane_for(record),
                record.committed_blocks,
                transition.outbound,
            )

    def on_recovery_complete(self, reply: RecoveryReply) -> CallbackResult:
        """Apply an exact recovery response and re-admit or finish its row."""

        if not isinstance(reply, RecoveryReply):
            raise TypeError("expected a RecoveryReply")
        with self._lock:
            record = self._records.get(reply.tag.request_id)
            if (
                record is None
                or record.cancelled
                or record.protocol.state is not ProtocolState.RECOVERING
                or record.protocol.active_tag != reply.tag
            ):
                return self._ignored_unlocked(reply.tag.request_id, record)
            complete = record.committed_blocks == record.total_blocks
            if reply.finished != complete:
                raise InvalidCoordinatorOperation(
                    "recovery finished flag must match target-authorized completion"
                )
            self._fault(FaultPoint.BEFORE_RECOVERY_APPLY, reply.tag.request_id)
            transition = record.protocol.handle_recovery_reply(reply)
            if not transition.applied:
                raise CoordinatorInvariantError(
                    "exact recovery tag became stale during a serialized callback"
                )
            self._recovery_outbox.pop(reply.tag.request_id, None)
            record.protocol_high_water = max(record.protocol_high_water, record.protocol.version)
            self._fault(FaultPoint.AFTER_RECOVERY_APPLY, reply.tag.request_id)
            self._audit_unlocked()
            return CallbackResult(
                ReplyDisposition.APPLIED,
                reply.tag.request_id,
                _lane_for(record),
                record.committed_blocks,
            )

    def cancel(self, request_id: ProtocolRequestId) -> bool:
        """Cancel a request, fence replies, and release all of its KV pages."""

        request_id = _request_id(request_id)
        with self._lock:
            record = self._require_record_unlocked(request_id)
            if record.cancelled:
                return False
            self._fault(FaultPoint.BEFORE_CANCEL, request_id)
            if record.active_epoch is not None:
                self._ledger.abort(record.active_epoch)
            state = record.protocol.state
            if state is ProtocolState.VERIFYING:
                tag = record.protocol.active_tag
                if tag is None:
                    raise CoordinatorInvariantError("verifying request has no active tag")
                transition = record.protocol.handle_verify_reply(
                    VerifyReply(tag, hit=True, accepted_blocks=0)
                )
                if not transition.applied:
                    raise CoordinatorInvariantError("could not close verifier during cancel")
                record.protocol.finish()
            elif state is ProtocolState.RECOVERING:
                tag = record.protocol.active_tag
                if tag is None:
                    raise CoordinatorInvariantError("recovering request has no active tag")
                transition = record.protocol.handle_recovery_reply(
                    RecoveryReply(tag, finished=True)
                )
                if not transition.applied:
                    raise CoordinatorInvariantError("could not close recovery during cancel")
            elif state is not ProtocolState.FINISHED:
                record.protocol.finish()
            record.active_epoch = None
            record.block_table = None
            self._recovery_outbox.pop(request_id, None)
            self._ledger.drop_request(request_id)
            record.cancelled = True
            record.protocol_high_water = max(record.protocol_high_water, record.protocol.version)
            self._fault(FaultPoint.AFTER_CANCEL, request_id)
            self._audit_unlocked()
            return True

    def audit(self) -> None:
        """Prove cross-layer lane, version, transaction, and ownership invariants."""

        with self._lock:
            self._audit_unlocked()

    def snapshot_bytes(self) -> bytes:
        """Return a deterministic canonical JSON checkpoint with SHA-256."""

        with self._lock:
            self._audit_unlocked()
            payload = self._snapshot_payload_unlocked()
            checksum = hashlib.sha256(_canonical_json(payload)).hexdigest()
            envelope: JsonObject = {
                "checksum": checksum,
                "payload": payload,
                "schema_version": _SNAPSHOT_VERSION,
            }
            return _canonical_json(envelope) + b"\n"

    @classmethod
    def from_snapshot_bytes(
        cls,
        data: bytes,
        *,
        resume_after_crash: bool = True,
        fault_hook: FaultHook | None = None,
    ) -> InMemoryCoordinator:
        """Restore a canonical checkpoint and optionally fence in-flight work."""

        payload = _decode_envelope(data)
        try:
            payload_object = _object(
                payload,
                fields={"ledger", "requests", "schema_version"},
                context="coordinator payload",
            )
            schema_version = _json_int(
                payload_object["schema_version"], field="coordinator schema_version"
            )
            if schema_version != _SNAPSHOT_VERSION:
                raise CoordinatorSnapshotError(
                    f"unsupported coordinator snapshot version {schema_version}"
                )
            ledger = SpeculativeLedger.from_snapshot(
                _decode_ledger_snapshot(payload_object["ledger"])
            )
            records: dict[ProtocolRequestId, _RequestRecord] = {}
            for raw_record in _array(payload_object["requests"], context="coordinator requests"):
                record_object = _object(
                    raw_record,
                    fields={
                        "active_epoch",
                        "block_table",
                        "cancelled",
                        "committed_blocks",
                        "protocol",
                        "protocol_high_water",
                        "total_blocks",
                    },
                    context="coordinator request",
                )
                protocol_snapshot = _decode_protocol_snapshot(record_object["protocol"])
                request_id = protocol_snapshot.request_id
                if request_id in records:
                    raise CoordinatorSnapshotError(
                        "coordinator snapshot contains duplicate request IDs"
                    )
                active_epoch_raw = record_object["active_epoch"]
                active_epoch = None if active_epoch_raw is None else _decode_epoch(active_epoch_raw)
                table_raw = record_object["block_table"]
                block_table = None if table_raw is None else _decode_block_table(table_raw)
                protocol = FissionProtocol.from_snapshot(protocol_snapshot)
                records[request_id] = _RequestRecord(
                    protocol=protocol,
                    total_blocks=_json_int(record_object["total_blocks"], field="total_blocks"),
                    committed_blocks=_json_int(
                        record_object["committed_blocks"], field="committed_blocks"
                    ),
                    active_epoch=active_epoch,
                    block_table=block_table,
                    cancelled=_json_bool(record_object["cancelled"], field="cancelled"),
                    protocol_high_water=_json_int(
                        record_object["protocol_high_water"],
                        field="protocol_high_water",
                    ),
                )

            coordinator = cls.__new__(cls)
            coordinator._ledger = ledger
            coordinator._records = records
            coordinator._recovery_outbox = {}
            coordinator._fault_hook = fault_hook
            coordinator._lock = threading.RLock()
            coordinator._rebuild_recovery_outbox_unlocked()
            coordinator._audit_unlocked()

            if resume_after_crash:
                for request_id in sorted(records, key=_id_sort_key):
                    record = records[request_id]
                    if record.cancelled or record.protocol.state not in {
                        ProtocolState.VERIFYING,
                        ProtocolState.RECOVERING,
                    }:
                        continue
                    prior = record.protocol.snapshot()
                    if record.active_epoch is not None:
                        coordinator._ledger.abort(record.active_epoch)
                    recovered = FissionProtocol.resume_after_crash(prior)
                    record.protocol = recovered.protocol
                    record.active_epoch = None
                    record.block_table = None
                    record.protocol_high_water = max(
                        record.protocol_high_water, record.protocol.version
                    )
                    if recovered.outbound is None:
                        raise CoordinatorInvariantError(
                            "in-flight crash recovery emitted no recovery command"
                        )
                    coordinator._recovery_outbox[request_id] = recovered.outbound
                coordinator._audit_unlocked()
            return coordinator
        except CoordinatorSnapshotError:
            raise
        except Exception as exc:
            raise CoordinatorSnapshotError(f"invalid coordinator snapshot: {exc}") from exc

    def _snapshot_payload_unlocked(self) -> JsonObject:
        return {
            "ledger": _encode_ledger_snapshot(self._ledger.snapshot()),
            "requests": [
                {
                    "active_epoch": (
                        None if record.active_epoch is None else _encode_epoch(record.active_epoch)
                    ),
                    "block_table": (
                        None
                        if record.block_table is None
                        else _encode_block_table(record.block_table)
                    ),
                    "cancelled": record.cancelled,
                    "committed_blocks": record.committed_blocks,
                    "protocol": _encode_protocol_snapshot(record.protocol.snapshot()),
                    "protocol_high_water": record.protocol_high_water,
                    "total_blocks": record.total_blocks,
                }
                for request_id in sorted(self._records, key=_id_sort_key)
                for record in (self._records[request_id],)
            ],
            "schema_version": _SNAPSHOT_VERSION,
        }

    def _audit_unlocked(self) -> None:
        try:
            self._ledger.audit()
            ledger_views = {view.request_id: view for view in self._ledger.requests()}
            for request_id, record in self._records.items():
                if record.protocol.request_id != request_id:
                    raise CoordinatorInvariantError(
                        "coordinator key and protocol request ID disagree"
                    )
                if record.protocol.version != record.protocol_high_water:
                    raise CoordinatorInvariantError(
                        "protocol version disagrees with its durable high-water mark"
                    )
                if not 0 <= record.committed_blocks <= record.total_blocks:
                    raise CoordinatorInvariantError(
                        "committed blocks fall outside the request extent"
                    )
                lane = _lane_for(record)
                ledger_view = ledger_views.get(request_id)
                if record.cancelled:
                    if lane is not SchedulerLane.CANCELLED:
                        raise CoordinatorInvariantError("cancelled request is in a scheduler lane")
                    if record.protocol.state is not ProtocolState.FINISHED:
                        raise CoordinatorInvariantError(
                            "cancelled request did not close its protocol"
                        )
                    if ledger_view is not None:
                        raise CoordinatorInvariantError("cancelled request still owns ledger pages")
                    if record.active_epoch is not None or record.block_table is not None:
                        raise CoordinatorInvariantError(
                            "cancelled request retains provisional state"
                        )
                    if request_id in self._recovery_outbox:
                        raise CoordinatorInvariantError(
                            "cancelled request retains a recovery command"
                        )
                    continue

                if ledger_view is None:
                    raise CoordinatorInvariantError(
                        "live coordinator request is missing from the ledger"
                    )
                if ledger_view.committed_blocks != record.committed_blocks:
                    raise CoordinatorInvariantError(
                        "coordinator and ledger committed counts disagree"
                    )
                if record.protocol.state is ProtocolState.VERIFYING:
                    if record.active_epoch is None or record.block_table is None:
                        raise CoordinatorInvariantError(
                            "verifying request lacks an explicit ledger binding"
                        )
                    if ledger_view.active_epoch != record.active_epoch:
                        raise CoordinatorInvariantError(
                            "bound ledger epoch is not the active transaction"
                        )
                    table = record.block_table
                    if table.epoch != record.active_epoch:
                        raise CoordinatorInvariantError(
                            "block table is bound to another ledger epoch"
                        )
                    if table.base_committed_blocks != record.committed_blocks:
                        raise CoordinatorInvariantError(
                            "block-table base does not match committed prefix"
                        )
                    if (
                        record.protocol.active_tag is None
                        or record.protocol.active_tag.round_id != record.active_epoch.round_id
                    ):
                        raise CoordinatorInvariantError("protocol tag and ledger round disagree")
                    if len(ledger_view.outcomes) != 1:
                        raise CoordinatorInvariantError(
                            "verifier transaction must have exactly one branch"
                        )
                    branch = ledger_view.outcomes[0]
                    refs = tuple(
                        PhysicalPageRef(span.handle.page_id, span.handle.generation)
                        for span in branch.pages
                    )
                    if (
                        branch.outcome_id != _OUTCOME_ID
                        or branch.appended_blocks != table.reserved_blocks
                        or refs != table.pages
                    ):
                        raise CoordinatorInvariantError(
                            "block table does not describe the staged branch"
                        )
                elif (
                    record.active_epoch is not None
                    or record.block_table is not None
                    or ledger_view.active_epoch is not None
                ):
                    raise CoordinatorInvariantError(
                        "non-verifying request retains a ledger transaction"
                    )

                recovery = self._recovery_outbox.get(request_id)
                if record.protocol.state is ProtocolState.RECOVERING:
                    tag = record.protocol.active_tag
                    if (
                        tag is None
                        or recovery is None
                        or recovery.tag != tag
                        or recovery.accepted_blocks != record.protocol.snapshot().accepted_blocks
                    ):
                        raise CoordinatorInvariantError(
                            "recovering request lacks its exact latest command"
                        )
                elif recovery is not None:
                    raise CoordinatorInvariantError(
                        "non-recovering request appears in recovery outbox"
                    )

                complete = record.committed_blocks == record.total_blocks
                if record.protocol.state is ProtocolState.FINISHED:
                    if not complete:
                        raise CoordinatorInvariantError(
                            "unfinished target prefix has a finished protocol"
                        )
                elif complete and record.protocol.state is not ProtocolState.RECOVERING:
                    raise CoordinatorInvariantError(
                        "complete request must be finished or closing recovery"
                    )

            unknown_ledger = set(ledger_views) - set(self._records)
            if unknown_ledger:
                raise CoordinatorInvariantError(
                    f"ledger has uncoordinated requests: {unknown_ledger!r}"
                )
            unknown_recovery = set(self._recovery_outbox) - set(self._records)
            if unknown_recovery:
                raise CoordinatorInvariantError(
                    f"outbox has unknown requests: {unknown_recovery!r}"
                )
        except InvariantViolation as exc:
            raise CoordinatorInvariantError(str(exc)) from exc

    def _view_unlocked(
        self, request_id: ProtocolRequestId, record: _RequestRecord
    ) -> CoordinatorRequestView:
        return CoordinatorRequestView(
            request_id=request_id,
            total_blocks=record.total_blocks,
            committed_blocks=record.committed_blocks,
            lane=_lane_for(record),
            round_id=record.protocol.round_id,
            protocol_version=record.protocol.version,
            active_tag=record.protocol.active_tag,
            active_epoch=record.active_epoch,
            block_table=record.block_table,
            cancelled=record.cancelled,
        )

    def _require_record_unlocked(self, request_id: ProtocolRequestId) -> _RequestRecord:
        try:
            return self._records[request_id]
        except KeyError as exc:
            raise UnknownRequestError(f"unknown request {request_id!r}") from exc

    def _ignored_unlocked(
        self,
        request_id: ProtocolRequestId,
        record: _RequestRecord | None,
    ) -> CallbackResult:
        return CallbackResult(
            ReplyDisposition.IGNORED_STALE,
            request_id,
            None if record is None else _lane_for(record),
            None if record is None else record.committed_blocks,
        )

    def _rebuild_recovery_outbox_unlocked(self) -> None:
        self._recovery_outbox.clear()
        for request_id, record in self._records.items():
            if record.cancelled or record.protocol.state is not ProtocolState.RECOVERING:
                continue
            snapshot = record.protocol.snapshot()
            if snapshot.active_tag is None:
                raise CoordinatorInvariantError("recovering protocol has no active tag")
            self._recovery_outbox[request_id] = RecoveryRequest(
                snapshot.active_tag, snapshot.accepted_blocks
            )

    def _fault(self, point: FaultPoint, request_id: ProtocolRequestId) -> None:
        hook = self._fault_hook
        if hook is not None:
            hook(point, request_id)


class CrashAt:
    """One-shot deterministic hook that crashes at a selected fault point."""

    def __init__(self, point: FaultPoint) -> None:
        if not isinstance(point, FaultPoint):
            raise TypeError("point must be a FaultPoint")
        self._point = point
        self._fired = False

    @property
    def fired(self) -> bool:
        return self._fired

    def __call__(self, point: FaultPoint, request_id: ProtocolRequestId) -> None:
        if not self._fired and point is self._point:
            self._fired = True
            raise CoordinatorCrash(f"injected crash at {point.value} for request {request_id!r}")


class FakeTargetVerifier:
    """Deterministic verifier reply factory used by integration tests."""

    @staticmethod
    def reply(
        dispatch: VerificationDispatch,
        *,
        hit: bool,
        accepted_blocks: int,
    ) -> VerifyReply:
        if not isinstance(dispatch, VerificationDispatch):
            raise TypeError("dispatch must be a VerificationDispatch")
        return VerifyReply(
            dispatch.request.tag,
            hit=hit,
            accepted_blocks=accepted_blocks,
        )


class FakeRemoteDrafter:
    """In-memory recovery transport with explicit drop/duplicate/reorder controls."""

    def __init__(self) -> None:
        self._pending: list[RecoveryRequest] = []

    @property
    def pending(self) -> tuple[RecoveryRequest, ...]:
        return tuple(self._pending)

    def submit(self, request: RecoveryRequest) -> None:
        if not isinstance(request, RecoveryRequest):
            raise TypeError("request must be a RecoveryRequest")
        self._pending.append(request)

    def duplicate(self, index: int = 0) -> None:
        index = _queue_index(index, len(self._pending))
        self._pending.append(self._pending[index])

    def drop(self, index: int = 0) -> RecoveryRequest:
        index = _queue_index(index, len(self._pending))
        return self._pending.pop(index)

    def reorder(self, order: Sequence[int]) -> None:
        if len(order) != len(self._pending):
            raise InvalidCoordinatorOperation("reorder must name every queued request exactly once")
        indices = [_plain_int(index, field="reorder index") for index in order]
        if sorted(indices) != list(range(len(self._pending))):
            raise InvalidCoordinatorOperation("reorder indices must be a permutation of the queue")
        self._pending = [self._pending[index] for index in indices]

    def reply(
        self,
        index: int = 0,
        *,
        finished: bool,
        recovered_blocks: int = 0,
    ) -> RecoveryReply:
        request = self.drop(index)
        return RecoveryReply(
            request.tag,
            finished=finished,
            recovered_blocks=recovered_blocks,
        )


def _queue_index(index: int, length: int) -> int:
    index = _plain_int(index, field="queue index")
    if index >= length:
        raise InvalidCoordinatorOperation(f"queue index {index} is outside length {length}")
    return index


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CoordinatorSnapshotError(f"value is not canonical JSON: {exc}") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise CoordinatorSnapshotError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise CoordinatorSnapshotError(f"non-finite JSON number {value!r} is forbidden")


def _decode_envelope(data: bytes) -> object:
    if not isinstance(data, bytes):
        raise CoordinatorSnapshotError("snapshot must be bytes")
    try:
        text = data.decode("utf-8")
        decoded: object = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except CoordinatorSnapshotError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoordinatorSnapshotError(f"snapshot is not valid JSON: {exc}") from exc
    envelope = _object(
        decoded,
        fields={"checksum", "payload", "schema_version"},
        context="snapshot envelope",
    )
    schema_version = _json_int(envelope["schema_version"], field="envelope schema_version")
    if schema_version != _SNAPSHOT_VERSION:
        raise CoordinatorSnapshotError(f"unsupported envelope snapshot version {schema_version}")
    checksum = _json_str(envelope["checksum"], field="checksum")
    if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
        raise CoordinatorSnapshotError("checksum must be 64 lowercase hexadecimal characters")
    payload = envelope["payload"]
    expected = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if not hmac.compare_digest(checksum, expected):
        raise CoordinatorSnapshotError("snapshot checksum mismatch")
    if _canonical_json(envelope) + b"\n" != data:
        raise CoordinatorSnapshotError("snapshot is not in canonical byte form")
    return payload


def _object(value: object, *, fields: set[str], context: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CoordinatorSnapshotError(f"{context} must be a JSON object")
    result = cast(JsonObject, value)
    actual = set(result)
    if actual != fields:
        missing = sorted(fields - actual)
        extra = sorted(actual - fields)
        raise CoordinatorSnapshotError(f"{context} fields differ; missing={missing}, extra={extra}")
    return result


def _array(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        raise CoordinatorSnapshotError(f"{context} must be a JSON array")
    return cast(list[object], value)


def _json_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CoordinatorSnapshotError(f"{field} must be an integer")
    if value < minimum:
        raise CoordinatorSnapshotError(f"{field} must be at least {minimum}")
    return value


def _json_signed_int(value: object, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CoordinatorSnapshotError(f"{field} must be an integer")
    if value < minimum:
        raise CoordinatorSnapshotError(f"{field} must be at least {minimum}")
    return value


def _json_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise CoordinatorSnapshotError(f"{field} must be a bool")
    return value


def _json_str(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise CoordinatorSnapshotError(f"{field} must be a string")
    return value


def _encode_identifier(value: ProtocolRequestId) -> JsonObject:
    return {
        "type": "int" if isinstance(value, int) else "str",
        "value": value,
    }


def _decode_identifier(value: object, *, field: str) -> ProtocolRequestId:
    encoded = _object(value, fields={"type", "value"}, context=field)
    kind = _json_str(encoded["type"], field=f"{field} type")
    raw = encoded["value"]
    if kind == "int":
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise CoordinatorSnapshotError(f"{field} int value is invalid")
        return raw
    if kind == "str":
        if not isinstance(raw, str):
            raise CoordinatorSnapshotError(f"{field} str value is invalid")
        return raw
    raise CoordinatorSnapshotError(f"{field} has unknown identifier type {kind!r}")


def _encode_pool(pool: PoolIdentity) -> JsonObject:
    return {
        "ordinal": pool.ordinal,
        "process_id": pool.process_id,
        "session_id": pool.session_id,
    }


def _decode_pool(value: object) -> PoolIdentity:
    encoded = _object(
        value,
        fields={"ordinal", "process_id", "session_id"},
        context="pool identity",
    )
    return PoolIdentity(
        session_id=_json_str(encoded["session_id"], field="pool session_id"),
        process_id=_json_int(encoded["process_id"], field="pool process_id", minimum=1),
        ordinal=_json_int(encoded["ordinal"], field="pool ordinal", minimum=1),
    )


def _encode_handle(handle: PageHandle) -> JsonObject:
    return {
        "generation": handle.generation,
        "page_id": handle.page_id,
        "pool": _encode_pool(handle.pool_id),
    }


def _decode_handle(value: object) -> PageHandle:
    encoded = _object(
        value,
        fields={"generation", "page_id", "pool"},
        context="page handle",
    )
    return PageHandle(
        page_id=_json_int(encoded["page_id"], field="page_id"),
        generation=_json_int(encoded["generation"], field="generation", minimum=1),
        pool_id=_decode_pool(encoded["pool"]),
    )


def _encode_span(span: PageSpan) -> JsonObject:
    return {
        "handle": _encode_handle(span.handle),
        "start": span.start,
        "used": span.used,
    }


def _decode_span(value: object) -> PageSpan:
    encoded = _object(value, fields={"handle", "start", "used"}, context="page span")
    return PageSpan(
        handle=_decode_handle(encoded["handle"]),
        start=_json_int(encoded["start"], field="span start"),
        used=_json_int(encoded["used"], field="span used", minimum=1),
    )


def _encode_epoch(epoch: LedgerEpoch) -> JsonObject:
    return {
        "request_id": _encode_identifier(epoch.request_id),
        "round_id": epoch.round_id,
        "version": epoch.version,
    }


def _decode_epoch(value: object) -> LedgerEpoch:
    encoded = _object(
        value,
        fields={"request_id", "round_id", "version"},
        context="ledger epoch",
    )
    return LedgerEpoch(
        request_id=_decode_identifier(encoded["request_id"], field="request_id"),
        round_id=_json_int(encoded["round_id"], field="ledger round_id"),
        version=_json_int(encoded["version"], field="ledger version", minimum=1),
    )


def _encode_tag(tag: MessageTag) -> JsonObject:
    return {
        "request_id": _encode_identifier(tag.request_id),
        "round_id": tag.round_id,
        "version": tag.version,
    }


def _decode_tag(value: object) -> MessageTag:
    encoded = _object(
        value,
        fields={"request_id", "round_id", "version"},
        context="message tag",
    )
    return MessageTag(
        request_id=_decode_identifier(encoded["request_id"], field="request_id"),
        round_id=_json_int(encoded["round_id"], field="protocol round_id"),
        version=_json_int(encoded["version"], field="protocol version", minimum=1),
    )


def _encode_block_table(table: BlockTableDescriptor) -> JsonObject:
    return {
        "base_committed_blocks": table.base_committed_blocks,
        "epoch": _encode_epoch(table.epoch),
        "pages": [{"generation": page.generation, "page_id": page.page_id} for page in table.pages],
        "reserved_blocks": table.reserved_blocks,
    }


def _decode_block_table(value: object) -> BlockTableDescriptor:
    encoded = _object(
        value,
        fields={"base_committed_blocks", "epoch", "pages", "reserved_blocks"},
        context="block table",
    )
    pages: list[PhysicalPageRef] = []
    for raw_page in _array(encoded["pages"], context="block-table pages"):
        page = _object(
            raw_page,
            fields={"generation", "page_id"},
            context="block-table page",
        )
        pages.append(
            PhysicalPageRef(
                page_id=_json_int(page["page_id"], field="page_id"),
                generation=_json_int(page["generation"], field="generation", minimum=1),
            )
        )
    return BlockTableDescriptor(
        epoch=_decode_epoch(encoded["epoch"]),
        base_committed_blocks=_json_int(
            encoded["base_committed_blocks"], field="base_committed_blocks"
        ),
        reserved_blocks=_json_int(encoded["reserved_blocks"], field="reserved_blocks", minimum=1),
        pages=tuple(pages),
    )


def _encode_protocol_snapshot(snapshot: ProtocolSnapshot) -> JsonObject:
    return {
        "accepted_blocks": snapshot.accepted_blocks,
        "active_tag": (None if snapshot.active_tag is None else _encode_tag(snapshot.active_tag)),
        "request_id": _encode_identifier(snapshot.request_id),
        "round_id": snapshot.round_id,
        "schema_version": snapshot.schema_version,
        "state": snapshot.state.value,
        "version": snapshot.version,
    }


def _decode_protocol_snapshot(value: object) -> ProtocolSnapshot:
    encoded = _object(
        value,
        fields={
            "accepted_blocks",
            "active_tag",
            "request_id",
            "round_id",
            "schema_version",
            "state",
            "version",
        },
        context="protocol snapshot",
    )
    active_raw = encoded["active_tag"]
    try:
        state = ProtocolState(_json_str(encoded["state"], field="protocol state"))
    except ValueError as exc:
        raise CoordinatorSnapshotError("unknown protocol state") from exc
    return ProtocolSnapshot(
        request_id=_decode_identifier(encoded["request_id"], field="request_id"),
        state=state,
        round_id=_json_signed_int(encoded["round_id"], field="protocol round_id", minimum=-1),
        version=_json_int(encoded["version"], field="protocol version"),
        active_tag=None if active_raw is None else _decode_tag(active_raw),
        accepted_blocks=_json_int(encoded["accepted_blocks"], field="accepted_blocks"),
        schema_version=_json_int(encoded["schema_version"], field="protocol schema_version"),
    )


def _encode_allocator_snapshot(snapshot: AllocatorSnapshot) -> JsonObject:
    return {
        "free_pages": list(snapshot.free_pages),
        "generations": list(snapshot.generations),
        "live": [
            {
                "handle": _encode_handle(handle),
                "owner": owner,
            }
            for handle, owner in snapshot.live
        ],
        "page_count": snapshot.page_count,
        "page_size": snapshot.page_size,
        "pool": _encode_pool(snapshot.pool_id),
        "schema_version": snapshot.schema_version,
    }


def _decode_allocator_snapshot(value: object) -> AllocatorSnapshot:
    encoded = _object(
        value,
        fields={
            "free_pages",
            "generations",
            "live",
            "page_count",
            "page_size",
            "pool",
            "schema_version",
        },
        context="allocator snapshot",
    )
    live: list[tuple[PageHandle, str | None]] = []
    for raw_entry in _array(encoded["live"], context="allocator live entries"):
        entry = _object(raw_entry, fields={"handle", "owner"}, context="allocator live entry")
        owner_raw = entry["owner"]
        if owner_raw is not None and not isinstance(owner_raw, str):
            raise CoordinatorSnapshotError("allocator owner must be a string or null")
        live.append((_decode_handle(entry["handle"]), owner_raw))
    return AllocatorSnapshot(
        page_count=_json_int(encoded["page_count"], field="page_count", minimum=1),
        page_size=_json_int(encoded["page_size"], field="page_size", minimum=1),
        pool_id=_decode_pool(encoded["pool"]),
        generations=tuple(
            _json_int(item, field="generation")
            for item in _array(encoded["generations"], context="generations")
        ),
        live=tuple(live),
        free_pages=tuple(
            _json_int(item, field="free page")
            for item in _array(encoded["free_pages"], context="free pages")
        ),
        schema_version=_json_int(encoded["schema_version"], field="allocator schema_version"),
    )


def _encode_branch_snapshot(snapshot: BranchSnapshot) -> JsonObject:
    return {
        "appended_blocks": snapshot.appended_blocks,
        "cow_tail": snapshot.cow_tail,
        "outcome_id": _encode_identifier(snapshot.outcome_id),
        "pages": [_encode_span(span) for span in snapshot.pages],
    }


def _decode_branch_snapshot(value: object) -> BranchSnapshot:
    encoded = _object(
        value,
        fields={"appended_blocks", "cow_tail", "outcome_id", "pages"},
        context="branch snapshot",
    )
    return BranchSnapshot(
        outcome_id=_decode_identifier(encoded["outcome_id"], field="outcome_id"),
        appended_blocks=_json_int(encoded["appended_blocks"], field="appended_blocks"),
        pages=tuple(
            _decode_span(item) for item in _array(encoded["pages"], context="branch pages")
        ),
        cow_tail=_json_bool(encoded["cow_tail"], field="cow_tail"),
    )


def _encode_request_snapshot(snapshot: RequestSnapshot) -> JsonObject:
    active: JsonObject | None = None
    if snapshot.active is not None:
        active = {
            "branches": [_encode_branch_snapshot(branch) for branch in snapshot.active.branches],
            "epoch": _encode_epoch(snapshot.active.epoch),
        }
    return {
        "active": active,
        "committed_blocks": snapshot.committed_blocks,
        "committed_pages": [_encode_span(span) for span in snapshot.committed_pages],
        "last_round": snapshot.last_round,
        "request_id": _encode_identifier(snapshot.request_id),
        "version": snapshot.version,
    }


def _decode_request_snapshot(value: object) -> RequestSnapshot:
    encoded = _object(
        value,
        fields={
            "active",
            "committed_blocks",
            "committed_pages",
            "last_round",
            "request_id",
            "version",
        },
        context="ledger request snapshot",
    )
    active_raw = encoded["active"]
    active: TransactionSnapshot | None = None
    if active_raw is not None:
        transaction = _object(
            active_raw,
            fields={"branches", "epoch"},
            context="transaction snapshot",
        )
        active = TransactionSnapshot(
            epoch=_decode_epoch(transaction["epoch"]),
            branches=tuple(
                _decode_branch_snapshot(item)
                for item in _array(transaction["branches"], context="transaction branches")
            ),
        )
    return RequestSnapshot(
        request_id=_decode_identifier(encoded["request_id"], field="request_id"),
        committed_blocks=_json_int(encoded["committed_blocks"], field="committed_blocks"),
        committed_pages=tuple(
            _decode_span(item)
            for item in _array(encoded["committed_pages"], context="committed pages")
        ),
        version=_json_int(encoded["version"], field="ledger request version"),
        last_round=_json_signed_int(encoded["last_round"], field="ledger last_round", minimum=-1),
        active=active,
    )


def _encode_ledger_snapshot(snapshot: LedgerSnapshot) -> JsonObject:
    return {
        "aborted_epochs": [_encode_epoch(epoch) for epoch in snapshot.aborted_epochs],
        "allocator": _encode_allocator_snapshot(snapshot.allocator),
        "requests": [_encode_request_snapshot(request) for request in snapshot.requests],
        "schema_version": snapshot.schema_version,
        "version_floors": [
            {
                "request_id": _encode_identifier(request_id),
                "version": version,
            }
            for request_id, version in snapshot.version_floors
        ],
    }


def _decode_ledger_snapshot(value: object) -> LedgerSnapshot:
    encoded = _object(
        value,
        fields={
            "aborted_epochs",
            "allocator",
            "requests",
            "schema_version",
            "version_floors",
        },
        context="ledger snapshot",
    )
    version_floors: list[tuple[ProtocolRequestId, int]] = []
    for raw_floor in _array(encoded["version_floors"], context="version floors"):
        floor = _object(
            raw_floor,
            fields={"request_id", "version"},
            context="version floor",
        )
        version_floors.append(
            (
                _decode_identifier(floor["request_id"], field="request_id"),
                _json_int(floor["version"], field="version floor"),
            )
        )
    return LedgerSnapshot(
        allocator=_decode_allocator_snapshot(encoded["allocator"]),
        requests=tuple(
            _decode_request_snapshot(item)
            for item in _array(encoded["requests"], context="ledger requests")
        ),
        aborted_epochs=tuple(
            _decode_epoch(item)
            for item in _array(encoded["aborted_epochs"], context="aborted epochs")
        ),
        version_floors=tuple(version_floors),
        schema_version=_json_int(encoded["schema_version"], field="ledger schema_version"),
    )


__all__ = [
    "BlockTableDescriptor",
    "CallbackResult",
    "CoordinatorCrash",
    "CoordinatorError",
    "CoordinatorInvariantError",
    "CoordinatorRequestView",
    "CoordinatorSnapshotError",
    "CrashAt",
    "FakeRemoteDrafter",
    "FakeTargetVerifier",
    "FaultHook",
    "FaultPoint",
    "InMemoryCoordinator",
    "InvalidCoordinatorOperation",
    "PhysicalPageRef",
    "SchedulerLane",
    "UnknownRequestError",
    "VerificationDispatch",
]
