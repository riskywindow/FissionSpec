"""Exact page ownership and speculative-transaction accounting.

This module models the part of a paged KV cache where small mistakes become
silent corruption: finite allocation, ABA-safe handles, copy-on-write (COW)
speculative branches, and atomic prefix commit.  It deliberately does not own
tensor bytes.  A :class:`PageHandle` is the capability a backend would attach
to one physical page, while :class:`PageSpan` records the logical block range
that is valid in that page.

Handles carry both a generation and a cryptographic process/session pool
namespace.  Generation blocks ABA within one pool; namespace blocks the
otherwise-identical first handle from another allocator.

The ledger assumes exclusive ownership of its allocator.  Its :meth:`audit`
method therefore proves a strong partition invariant: every physical page is
either free or is referenced exactly once by a committed trunk or private
outcome branch.  This catches leaks, double ownership, stale generations, and
position/count drift at the point they occur rather than several rounds later.
"""

from __future__ import annotations

import heapq
import itertools
import os
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, TypeAlias

RequestId: TypeAlias = str | int
OutcomeId: TypeAlias = str | int

_ALLOCATOR_SNAPSHOT_VERSION: Final[int] = 2
_LEDGER_SNAPSHOT_VERSION: Final[int] = 2

# A module session is computationally unique across independently started
# processes.  The ordinal then gives collision-free identities within that
# session.  ``_new_pool_identity`` notices fork(), creates a fresh child
# session, and resets the child-local sequence.
_POOL_ID_LOCK = threading.Lock()
_POOL_ID_PROCESS = os.getpid()
_POOL_ID_SESSION = secrets.token_hex(32)
_POOL_ID_SEQUENCE = itertools.count(1)


def _reset_pool_identity_after_fork() -> None:
    """Create independent child identity state, including an unlocked lock."""

    global _POOL_ID_LOCK, _POOL_ID_PROCESS, _POOL_ID_SEQUENCE, _POOL_ID_SESSION
    _POOL_ID_LOCK = threading.Lock()
    _POOL_ID_PROCESS = os.getpid()
    _POOL_ID_SESSION = secrets.token_hex(32)
    _POOL_ID_SEQUENCE = itertools.count(1)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_pool_identity_after_fork)


class LedgerError(RuntimeError):
    """Base class for allocator and speculative-ledger failures."""


class InvalidConfigurationError(LedgerError, ValueError):
    """Raised for an impossible allocator or ledger configuration."""


class OutOfPagesError(LedgerError):
    """Raised when a finite allocator cannot satisfy an allocation."""


class InvalidPageHandleError(LedgerError, ValueError):
    """Raised when a handle names no physical page in an allocator."""


class ForeignPageHandleError(InvalidPageHandleError):
    """Raised when a page capability belongs to a different allocator pool."""


class ForkedAllocatorError(LedgerError):
    """Raised when an allocator inherited through ``fork()`` is used directly."""


class StalePageHandleError(LedgerError):
    """Raised when a handle's generation does not name the current lease."""


class DoubleReleaseError(LedgerError):
    """Raised when the current generation of an already-free page is released."""


class InvariantViolation(LedgerError):
    """Raised when an ownership, generation, or logical-position audit fails."""


class SnapshotError(LedgerError, ValueError):
    """Raised when a snapshot is unsupported or internally inconsistent."""


class RequestNotFoundError(LedgerError, KeyError):
    """Raised when a direct operation names an unknown request."""


class DuplicateRequestError(LedgerError):
    """Raised when registering an already-live request."""


class TransactionConflictError(LedgerError):
    """Raised when starting a round while another transaction is active."""


class StaleEpochError(LedgerError):
    """Raised when an asynchronous outcome names a closed or different epoch."""


class DuplicateOutcomeError(LedgerError):
    """Raised when the same outcome is staged twice in one epoch."""


class OutcomeNotFoundError(LedgerError, KeyError):
    """Raised when commit selects an outcome that was never staged."""


class InvalidCommitError(LedgerError, ValueError):
    """Raised when an accepted prefix is outside the selected branch."""


def _plain_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _identifier(value: object, *, field: str) -> RequestId:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{field} must be a str or int (but not bool)")
    return value


def _id_sort_key(value: RequestId) -> tuple[str, str]:
    """Provide deterministic ordering without comparing unlike Python types."""

    return (type(value).__name__, repr(value))


def _owner_component(value: RequestId) -> str:
    return f"{type(value).__name__}:{value!r}"


@dataclass(frozen=True, order=True, slots=True)
class PoolIdentity:
    """Computationally unique namespace for one allocator incarnation.

    ``session_id`` is a 256-bit cryptographic nonce created once per process
    session, including a fresh session after ``fork()``.  ``ordinal`` is never
    reused within that session.  ``process_id`` is diagnostic and also makes a
    fork boundary explicit in persisted snapshots.

    Snapshot restoration never reuses this identity: it allocates a new local
    identity and rebases every live handle.  Consequently two concurrent
    restores of the same snapshot cannot accept one another's capabilities.
    """

    session_id: str
    process_id: int
    ordinal: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.session_id, str)
            or len(self.session_id) != 64
            or self.session_id.lower() != self.session_id
        ):
            raise InvalidConfigurationError(
                "pool session_id must be 64 lowercase hexadecimal characters"
            )
        try:
            decoded = bytes.fromhex(self.session_id)
        except ValueError as exc:
            raise InvalidConfigurationError("pool session_id must be hexadecimal") from exc
        if len(decoded) != 32:
            raise InvalidConfigurationError("pool session_id must encode 256 bits")
        try:
            _plain_int(self.process_id, field="pool process_id", minimum=1)
            _plain_int(self.ordinal, field="pool ordinal", minimum=1)
        except ValueError as exc:
            raise InvalidConfigurationError(str(exc)) from exc


def _new_pool_identity() -> PoolIdentity:
    """Return a fresh pool namespace, refreshing process state after ``fork``."""

    global _POOL_ID_PROCESS, _POOL_ID_SEQUENCE, _POOL_ID_SESSION
    process_id = os.getpid()
    with _POOL_ID_LOCK:
        if process_id != _POOL_ID_PROCESS:
            _POOL_ID_PROCESS = process_id
            _POOL_ID_SESSION = secrets.token_hex(32)
            _POOL_ID_SEQUENCE = itertools.count(1)
        return PoolIdentity(
            session_id=_POOL_ID_SESSION,
            process_id=process_id,
            ordinal=next(_POOL_ID_SEQUENCE),
        )


@dataclass(frozen=True, order=True, slots=True)
class PageHandle:
    """An ABA-safe lease on one physical page.

    ``page_id`` may be reused, but every allocation increments the unbounded
    Python integer ``generation``.  ``pool_id`` distinguishes independently
    created allocators.  A delayed release of an older generation—or a handle
    from a numerically identical foreign pool—can therefore never free a newer
    tenant's page.
    """

    page_id: int
    generation: int
    pool_id: PoolIdentity

    def __post_init__(self) -> None:
        try:
            _plain_int(self.page_id, field="page_id")
            _plain_int(self.generation, field="generation", minimum=1)
        except ValueError as exc:
            raise InvalidPageHandleError(str(exc)) from exc
        if not isinstance(self.pool_id, PoolIdentity):
            raise InvalidPageHandleError("pool_id must be a PoolIdentity")


@dataclass(frozen=True, slots=True)
class PageSpan:
    """A physical page and its contiguous valid logical block interval."""

    handle: PageHandle
    start: int
    used: int

    def __post_init__(self) -> None:
        if not isinstance(self.handle, PageHandle):
            raise ValueError("handle must be a PageHandle")
        _plain_int(self.start, field="start")
        _plain_int(self.used, field="used", minimum=1)

    @property
    def stop(self) -> int:
        """Exclusive logical block position after this page's valid data."""

        return self.start + self.used


@dataclass(frozen=True, slots=True)
class AllocatorSnapshot:
    """Immutable, complete source state used to restore a page allocator.

    Restoration validates ``pool_id`` and every saved live handle, then assigns
    the destination a fresh pool identity and rebases the handles.  Repeated
    snapshots of one live allocator compare equal; a restored snapshot differs
    in namespace by design.
    """

    page_count: int
    page_size: int
    pool_id: PoolIdentity
    generations: tuple[int, ...]
    live: tuple[tuple[PageHandle, str | None], ...]
    free_pages: tuple[int, ...]
    schema_version: int = _ALLOCATOR_SNAPSHOT_VERSION


class FixedPageAllocator:
    """A finite, thread-safe allocator with namespaced generation leases.

    An allocator inherited through ``fork()`` is deliberately unusable.  Take
    a snapshot and restore it in the child to obtain a safely re-namespaced
    pool instead of sharing capabilities across diverged address spaces.
    """

    def __init__(self, page_count: int, page_size: int) -> None:
        try:
            self._page_count = _plain_int(page_count, field="page_count", minimum=1)
            self._page_size = _plain_int(page_size, field="page_size", minimum=1)
        except ValueError as exc:
            raise InvalidConfigurationError(str(exc)) from exc
        self._pool_id = _new_pool_identity()
        self._creator_pid = os.getpid()
        self._generations: list[int] = [0] * self._page_count
        self._live: dict[int, tuple[PageHandle, str | None]] = {}
        self._free: list[int] = list(range(self._page_count))
        heapq.heapify(self._free)
        self._lock = threading.RLock()

    @property
    def page_count(self) -> int:
        return self._page_count

    @property
    def capacity(self) -> int:
        """Alias for the number of physical pages."""

        return self._page_count

    @property
    def page_size(self) -> int:
        """Number of logical blocks held by one full physical page."""

        return self._page_size

    @property
    def pool_id(self) -> PoolIdentity:
        """Namespace carried by every handle issued by this allocator."""

        return self._pool_id

    @property
    def free_count(self) -> int:
        with self._lock:
            self._assert_process_unlocked()
            return len(self._free)

    @property
    def allocated_count(self) -> int:
        with self._lock:
            self._assert_process_unlocked()
            return len(self._live)

    def allocate(self, owner: str | None = None) -> PageHandle:
        """Lease the lowest-numbered free page and advance its generation."""

        if owner is not None and not isinstance(owner, str):
            raise ValueError("owner must be a str or None")
        with self._lock:
            self._assert_process_unlocked()
            if not self._free:
                raise OutOfPagesError(f"allocator exhausted ({self._page_count} pages are live)")
            page_id = heapq.heappop(self._free)
            self._generations[page_id] += 1
            handle = PageHandle(page_id, self._generations[page_id], self._pool_id)
            self._live[page_id] = (handle, owner)
            return handle

    def clone(self, source: PageHandle, owner: str | None = None) -> PageHandle:
        """Validate ``source`` and lease a page for a backend-level COW copy.

        The allocator tracks ownership rather than bytes, so the caller remains
        responsible for copying actual KV contents into the returned page.
        """

        with self._lock:
            self._assert_process_unlocked()
            self._require_live_unlocked(source)
            return self.allocate(owner)

    def release(self, handle: PageHandle) -> None:
        """Release a lease, rejecting stale generations and double releases."""

        with self._lock:
            self._assert_process_unlocked()
            self._validate_page_id_unlocked(handle)
            current_generation = self._generations[handle.page_id]
            current = self._live.get(handle.page_id)
            if current is None:
                if handle.generation == current_generation:
                    raise DoubleReleaseError(f"page lease {handle!r} is already free")
                raise StalePageHandleError(
                    f"stale page lease {handle!r}; current generation is "
                    f"{current_generation} and the page is free"
                )
            current_handle, _ = current
            if current_handle != handle:
                raise StalePageHandleError(
                    f"stale page lease {handle!r}; live lease is {current_handle!r}"
                )
            del self._live[handle.page_id]
            heapq.heappush(self._free, handle.page_id)

    def reassign(self, handle: PageHandle, owner: str | None) -> None:
        """Change diagnostic ownership without changing the page lease."""

        if owner is not None and not isinstance(owner, str):
            raise ValueError("owner must be a str or None")
        with self._lock:
            self._assert_process_unlocked()
            self._require_live_unlocked(handle)
            self._live[handle.page_id] = (handle, owner)

    def is_live(self, handle: PageHandle) -> bool:
        """Return whether ``handle`` is the exact current live generation."""

        with self._lock:
            self._assert_process_unlocked()
            if not isinstance(handle, PageHandle):
                return False
            self._validate_pool_unlocked(handle)
            current = self._live.get(handle.page_id)
            return current is not None and current[0] == handle

    def owner_of(self, handle: PageHandle) -> str | None:
        """Return diagnostic ownership for a live handle."""

        with self._lock:
            self._assert_process_unlocked()
            _, owner = self._require_live_unlocked(handle)
            return owner

    def live_handles(self) -> frozenset[PageHandle]:
        """Return an immutable point-in-time view of all current leases."""

        with self._lock:
            self._assert_process_unlocked()
            return frozenset(entry[0] for entry in self._live.values())

    def snapshot(self) -> AllocatorSnapshot:
        """Capture an immutable allocator checkpoint."""

        with self._lock:
            self._assert_process_unlocked()
            return AllocatorSnapshot(
                page_count=self._page_count,
                page_size=self._page_size,
                pool_id=self._pool_id,
                generations=tuple(self._generations),
                live=tuple(self._live[index] for index in sorted(self._live)),
                free_pages=tuple(sorted(self._free)),
            )

    @classmethod
    def from_snapshot(cls, snapshot: AllocatorSnapshot) -> FixedPageAllocator:
        """Construct a freshly namespaced allocator from a validated checkpoint."""

        if not isinstance(snapshot, AllocatorSnapshot):
            raise SnapshotError("expected an AllocatorSnapshot")
        if (
            isinstance(snapshot.schema_version, bool)
            or not isinstance(snapshot.schema_version, int)
            or snapshot.schema_version != _ALLOCATOR_SNAPSHOT_VERSION
        ):
            raise SnapshotError(f"unsupported allocator snapshot version {snapshot.schema_version}")
        try:
            if not isinstance(snapshot.pool_id, PoolIdentity):
                raise InvariantViolation("snapshot pool_id is invalid")
            allocator = cls(snapshot.page_count, snapshot.page_size)
            allocator._generations = list(snapshot.generations)
            rebased_live: dict[int, tuple[PageHandle, str | None]] = {}
            for handle, owner in snapshot.live:
                if not isinstance(handle, PageHandle):
                    raise InvariantViolation("live snapshot contains a non-handle")
                if handle.pool_id != snapshot.pool_id:
                    raise InvariantViolation("live handle namespace does not match snapshot pool")
                rebased = PageHandle(handle.page_id, handle.generation, allocator._pool_id)
                rebased_live[rebased.page_id] = (rebased, owner)
            allocator._live = rebased_live
            if len(allocator._live) != len(snapshot.live):
                raise InvariantViolation("live snapshot contains a duplicate page")
            allocator._free = list(snapshot.free_pages)
            heapq.heapify(allocator._free)
            allocator.audit()
        except (AttributeError, InvariantViolation, ValueError, TypeError) as exc:
            raise SnapshotError(f"invalid allocator snapshot: {exc}") from exc
        return allocator

    def restore(self, snapshot: AllocatorSnapshot) -> None:
        """Atomically replace this allocator's state from a checkpoint."""

        restored = type(self).from_snapshot(snapshot)
        with self._lock:
            self._page_count = restored._page_count
            self._page_size = restored._page_size
            self._pool_id = restored._pool_id
            self._creator_pid = restored._creator_pid
            self._generations = restored._generations
            self._live = restored._live
            self._free = restored._free

    def audit(self) -> None:
        """Prove the live/free partition and generation invariants."""

        with self._lock:
            self._assert_process_unlocked()
            if self._page_count < 1 or self._page_size < 1:
                raise InvariantViolation("allocator dimensions must be positive")
            if len(self._generations) != self._page_count:
                raise InvariantViolation("generation table length does not match capacity")
            if any(
                isinstance(generation, bool) or not isinstance(generation, int) or generation < 0
                for generation in self._generations
            ):
                raise InvariantViolation("generation table contains an invalid value")

            free = set(self._free)
            if len(free) != len(self._free):
                raise InvariantViolation("free list contains a duplicate page")
            if any(
                isinstance(page_id, bool)
                or not isinstance(page_id, int)
                or not 0 <= page_id < self._page_count
                for page_id in self._free
            ):
                raise InvariantViolation("free list contains an invalid page id")
            live = set(self._live)
            universe = set(range(self._page_count))
            if free & live:
                raise InvariantViolation("a page is simultaneously live and free")
            if free | live != universe:
                missing = universe - (free | live)
                raise InvariantViolation(f"pages are neither live nor free: {missing}")
            for page_id, (handle, owner) in self._live.items():
                if not isinstance(handle, PageHandle) or handle.page_id != page_id:
                    raise InvariantViolation("live table key/handle mismatch")
                if handle.pool_id != self._pool_id:
                    raise InvariantViolation("live handle belongs to another pool")
                if handle.generation != self._generations[page_id]:
                    raise InvariantViolation("live handle has a stale generation")
                if owner is not None and not isinstance(owner, str):
                    raise InvariantViolation("page owner must be a str or None")

    def _validate_page_id_unlocked(self, handle: PageHandle) -> None:
        if not isinstance(handle, PageHandle):
            raise InvalidPageHandleError("expected a PageHandle")
        self._validate_pool_unlocked(handle)
        if handle.page_id >= self._page_count:
            raise InvalidPageHandleError(
                f"page {handle.page_id} is outside capacity {self._page_count}"
            )

    def _require_live_unlocked(self, handle: PageHandle) -> tuple[PageHandle, str | None]:
        self._validate_page_id_unlocked(handle)
        current = self._live.get(handle.page_id)
        if current is None or current[0] != handle:
            raise StalePageHandleError(f"page lease {handle!r} is not live")
        return current

    def _validate_pool_unlocked(self, handle: PageHandle) -> None:
        if handle.pool_id != self._pool_id:
            raise ForeignPageHandleError(
                f"page lease belongs to pool {handle.pool_id!r}, not {self._pool_id!r}"
            )

    def _assert_process_unlocked(self) -> None:
        current_pid = os.getpid()
        if current_pid != self._creator_pid:
            raise ForkedAllocatorError(
                f"allocator pool was created in process {self._creator_pid}, "
                f"but is being used in process {current_pid}; restore a snapshot "
                "in the child to obtain a fresh pool namespace"
            )


@dataclass(frozen=True, slots=True)
class LedgerEpoch:
    """Identity of one isolated speculative transaction."""

    request_id: RequestId
    round_id: int
    version: int

    def __post_init__(self) -> None:
        _identifier(self.request_id, field="request_id")
        _plain_int(self.round_id, field="round_id")
        _plain_int(self.version, field="version", minimum=1)


@dataclass(frozen=True, slots=True)
class BranchView:
    """Immutable description of one outcome's private COW extension."""

    outcome_id: OutcomeId
    appended_blocks: int
    pages: tuple[PageSpan, ...]
    cow_tail: bool


@dataclass(frozen=True, slots=True)
class RequestView:
    """Immutable request state returned to callers and diagnostics."""

    request_id: RequestId
    committed_blocks: int
    committed_pages: tuple[PageSpan, ...]
    version: int
    last_round: int
    active_epoch: LedgerEpoch | None
    outcomes: tuple[BranchView, ...]


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Summary of an outcome-prefix commit."""

    epoch: LedgerEpoch
    outcome_id: OutcomeId
    accepted_blocks: int
    committed_blocks: int
    released_pages: int
    request: RequestView


@dataclass(frozen=True, slots=True)
class BranchSnapshot:
    outcome_id: OutcomeId
    appended_blocks: int
    pages: tuple[PageSpan, ...]
    cow_tail: bool


@dataclass(frozen=True, slots=True)
class TransactionSnapshot:
    epoch: LedgerEpoch
    branches: tuple[BranchSnapshot, ...]


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    request_id: RequestId
    committed_blocks: int
    committed_pages: tuple[PageSpan, ...]
    version: int
    last_round: int
    active: TransactionSnapshot | None


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    """Immutable crash-recovery image of allocator and transaction state.

    Restoring re-namespaces the allocator and all trunk/branch handles as one
    atomic logical image; the source snapshot's handles remain foreign.
    """

    allocator: AllocatorSnapshot
    requests: tuple[RequestSnapshot, ...]
    aborted_epochs: tuple[LedgerEpoch, ...]
    version_floors: tuple[tuple[RequestId, int], ...]
    schema_version: int = _LEDGER_SNAPSHOT_VERSION


@dataclass(slots=True)
class _Branch:
    outcome_id: OutcomeId
    appended_blocks: int
    pages: list[PageSpan]
    cow_tail: bool


@dataclass(slots=True)
class _Transaction:
    epoch: LedgerEpoch
    branches: dict[OutcomeId, _Branch]


@dataclass(slots=True)
class _Request:
    request_id: RequestId
    committed_blocks: int
    committed_pages: list[PageSpan]
    version: int
    last_round: int
    active: _Transaction | None


class SpeculativeLedger:
    """Per-request, epoch-isolated speculative page transaction ledger.

    Full committed pages are shared conceptually by every outcome and therefore
    appear only in the committed trunk.  If the trunk ends in a partial page,
    each non-empty outcome receives a private clone of that tail before it may
    append.  Committing a prefix transfers only the selected branch's required
    pages, releases all alternatives and unused suffix pages, and replaces the
    old partial tail when necessary.
    """

    def __init__(self, allocator: FixedPageAllocator) -> None:
        if not isinstance(allocator, FixedPageAllocator):
            raise TypeError("allocator must be a FixedPageAllocator")
        if allocator.allocated_count:
            raise InvalidConfigurationError(
                "a new ledger requires an exclusively owned, empty allocator"
            )
        self._allocator = allocator
        self._requests: dict[RequestId, _Request] = {}
        self._aborted_epochs: set[LedgerEpoch] = set()
        # Tombstones prevent request-id reuse from recreating an old epoch.
        self._version_floors: dict[RequestId, int] = {}
        self._lock = threading.RLock()

    @property
    def allocator(self) -> FixedPageAllocator:
        return self._allocator

    @property
    def page_size(self) -> int:
        return self._allocator.page_size

    def register_request(self, request_id: RequestId, committed_blocks: int = 0) -> RequestView:
        """Register a request, optionally materializing an initial trunk."""

        request_id = _identifier(request_id, field="request_id")
        try:
            committed_blocks = _plain_int(committed_blocks, field="committed_blocks")
        except ValueError as exc:
            raise InvalidCommitError(str(exc)) from exc
        with self._lock:
            if request_id in self._requests:
                raise DuplicateRequestError(f"request {request_id!r} already exists")
            needed = self._pages_for_blocks(committed_blocks)
            if needed > self._allocator.free_count:
                raise OutOfPagesError(
                    f"initial trunk needs {needed} pages but only "
                    f"{self._allocator.free_count} are free"
                )
            owner = self._trunk_owner(request_id)
            handles: list[PageHandle] = []
            try:
                for _ in range(needed):
                    handles.append(self._allocator.allocate(owner))
            except Exception:
                for handle in handles:
                    self._allocator.release(handle)
                raise
            pages = self._spans_from_zero(handles, committed_blocks)
            version = self._version_floors.get(request_id, 0)
            request = _Request(request_id, committed_blocks, pages, version, -1, None)
            self._requests[request_id] = request
            self._version_floors[request_id] = version
            self._audit_unlocked()
            return self._view_unlocked(request)

    def begin(self, request_id: RequestId, round_id: int) -> LedgerEpoch:
        """Open a new monotonically numbered speculative transaction."""

        request_id = _identifier(request_id, field="request_id")
        _plain_int(round_id, field="round_id")
        with self._lock:
            request = self._get_request_unlocked(request_id)
            if request.active is not None:
                raise TransactionConflictError(
                    f"request {request_id!r} already has active epoch {request.active.epoch!r}"
                )
            if round_id <= request.last_round:
                raise StaleEpochError(f"round {round_id} is not newer than {request.last_round}")
            request.version += 1
            self._version_floors[request_id] = request.version
            request.last_round = round_id
            epoch = LedgerEpoch(request_id, round_id, request.version)
            request.active = _Transaction(epoch, {})
            return epoch

    begin_round = begin
    begin_transaction = begin

    def stage_outcome(
        self, epoch: LedgerEpoch, outcome_id: OutcomeId, appended_blocks: int
    ) -> BranchView:
        """Materialize one outcome's private extension in the active epoch.

        Allocation is all-or-nothing.  A partial committed tail is cloned for
        every non-empty outcome, even if only one appended block is requested.
        """

        if not isinstance(epoch, LedgerEpoch):
            raise StaleEpochError("expected a LedgerEpoch")
        outcome_id = _identifier(outcome_id, field="outcome_id")
        try:
            appended_blocks = _plain_int(appended_blocks, field="appended_blocks")
        except ValueError as exc:
            raise InvalidCommitError(str(exc)) from exc
        with self._lock:
            request, transaction = self._require_active_unlocked(epoch)
            if outcome_id in transaction.branches:
                raise DuplicateOutcomeError(
                    f"outcome {outcome_id!r} is already staged in {epoch!r}"
                )
            tail_used = request.committed_blocks % self.page_size
            cow_tail = appended_blocks > 0 and tail_used > 0
            private_extent = appended_blocks + (tail_used if cow_tail else 0)
            needed = self._pages_for_blocks(private_extent)
            if needed > self._allocator.free_count:
                raise OutOfPagesError(
                    f"outcome needs {needed} pages but only {self._allocator.free_count} are free"
                )
            owner = self._branch_owner(epoch, outcome_id)
            handles: list[PageHandle] = []
            try:
                if cow_tail:
                    source = request.committed_pages[-1].handle
                    handles.append(self._allocator.clone(source, owner))
                while len(handles) < needed:
                    handles.append(self._allocator.allocate(owner))
            except Exception:
                for handle in handles:
                    self._allocator.release(handle)
                raise

            start = request.committed_blocks - tail_used if cow_tail else request.committed_blocks
            pages = self._spans_from_start(handles, start, private_extent)
            branch = _Branch(outcome_id, appended_blocks, pages, cow_tail)
            transaction.branches[outcome_id] = branch
            self._audit_unlocked()
            return self._branch_view(branch)

    record_outcome = stage_outcome
    add_outcome = stage_outcome

    def commit(
        self, epoch: LedgerEpoch, outcome_id: OutcomeId, accepted_blocks: int
    ) -> CommitResult:
        """Commit a selected outcome's accepted prefix and close the epoch."""

        if not isinstance(epoch, LedgerEpoch):
            raise StaleEpochError("expected a LedgerEpoch")
        outcome_id = _identifier(outcome_id, field="outcome_id")
        try:
            accepted_blocks = _plain_int(accepted_blocks, field="accepted_blocks")
        except ValueError as exc:
            raise InvalidCommitError(str(exc)) from exc
        with self._lock:
            request, transaction = self._require_active_unlocked(epoch)
            try:
                selected = transaction.branches[outcome_id]
            except KeyError as exc:
                raise OutcomeNotFoundError(
                    f"outcome {outcome_id!r} is not staged in {epoch!r}"
                ) from exc
            if accepted_blocks > selected.appended_blocks:
                raise InvalidCommitError(
                    f"cannot accept {accepted_blocks} blocks from a branch with "
                    f"{selected.appended_blocks} appended blocks"
                )

            keep = self._accepted_pages_unlocked(request, selected, accepted_blocks)
            keep_handles = {span.handle for span in keep}
            released = 0

            for branch in transaction.branches.values():
                for span in branch.pages:
                    if span.handle not in keep_handles:
                        self._allocator.release(span.handle)
                        released += 1

            if accepted_blocks > 0:
                if selected.cow_tail:
                    old_tail = request.committed_pages.pop()
                    self._allocator.release(old_tail.handle)
                    released += 1
                for span in keep:
                    self._allocator.reassign(span.handle, self._trunk_owner(request.request_id))
                request.committed_pages.extend(keep)
                request.committed_blocks += accepted_blocks

            request.active = None
            self._audit_unlocked()
            view = self._view_unlocked(request)
            return CommitResult(
                epoch=epoch,
                outcome_id=outcome_id,
                accepted_blocks=accepted_blocks,
                committed_blocks=request.committed_blocks,
                released_pages=released,
                request=view,
            )

    commit_outcome = commit

    def abort(self, epoch: LedgerEpoch) -> bool:
        """Abort an epoch, releasing every private page.

        The first matching call returns ``True``.  Replaying the exact same
        abort is an idempotent no-op returning ``False``; all other closed or
        mismatched epochs are rejected as stale.
        """

        if not isinstance(epoch, LedgerEpoch):
            raise StaleEpochError("expected a LedgerEpoch")
        with self._lock:
            if epoch in self._aborted_epochs:
                return False
            request, transaction = self._require_active_unlocked(epoch)
            for branch in transaction.branches.values():
                for span in branch.pages:
                    self._allocator.release(span.handle)
            request.active = None
            self._aborted_epochs.add(epoch)
            self._audit_unlocked()
            return True

    abort_transaction = abort

    def retry(self, epoch: LedgerEpoch) -> LedgerEpoch:
        """Fence and restart the same logical round at a fresh ledger version.

        Every provisional page from ``epoch`` is released before the replacement
        epoch becomes visible.  Delayed stage/commit/abort operations carrying
        the old epoch are therefore stale.

        The returned version is a *ledger* incarnation counter.  A coordinator
        retrying :class:`~fissionspec.protocol.MessageTag` must retain the
        explicit ``MessageTag -> LedgerEpoch`` association; numeric protocol and
        ledger versions often advance together but are not specified to be
        globally equal (for example, request-ID tombstones can advance only the
        ledger counter).
        """

        if not isinstance(epoch, LedgerEpoch):
            raise StaleEpochError("expected a LedgerEpoch")
        with self._lock:
            request, transaction = self._require_active_unlocked(epoch)
            for branch in transaction.branches.values():
                for span in branch.pages:
                    self._allocator.release(span.handle)
            request.version += 1
            self._version_floors[request.request_id] = request.version
            replacement = LedgerEpoch(request.request_id, request.last_round, request.version)
            request.active = _Transaction(replacement, {})
            self._audit_unlocked()
            return replacement

    retry_transaction = retry

    def drop_request(self, request_id: RequestId) -> bool:
        """Release a request's transaction and trunk; missing requests are no-ops."""

        request_id = _identifier(request_id, field="request_id")
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return False
            if request.active is not None:
                for branch in request.active.branches.values():
                    for span in branch.pages:
                        self._allocator.release(span.handle)
            for span in request.committed_pages:
                self._allocator.release(span.handle)
            del self._requests[request_id]
            self._aborted_epochs = {
                epoch for epoch in self._aborted_epochs if epoch.request_id != request_id
            }
            self._audit_unlocked()
            return True

    finish_request = drop_request

    def request(self, request_id: RequestId) -> RequestView:
        """Return an immutable view of one live request."""

        request_id = _identifier(request_id, field="request_id")
        with self._lock:
            return self._view_unlocked(self._get_request_unlocked(request_id))

    def requests(self) -> tuple[RequestView, ...]:
        """Return all request views in deterministic identifier order."""

        with self._lock:
            return tuple(
                self._view_unlocked(self._requests[key])
                for key in sorted(self._requests, key=_id_sort_key)
            )

    def snapshot(self) -> LedgerSnapshot:
        """Capture allocator, committed trunks, branches, and abort replay state."""

        with self._lock:
            self._audit_unlocked()
            request_snapshots: list[RequestSnapshot] = []
            for key in sorted(self._requests, key=_id_sort_key):
                request = self._requests[key]
                active: TransactionSnapshot | None = None
                if request.active is not None:
                    branches = tuple(
                        BranchSnapshot(
                            branch.outcome_id,
                            branch.appended_blocks,
                            tuple(branch.pages),
                            branch.cow_tail,
                        )
                        for branch in self._sorted_branches(request.active)
                    )
                    active = TransactionSnapshot(request.active.epoch, branches)
                request_snapshots.append(
                    RequestSnapshot(
                        request.request_id,
                        request.committed_blocks,
                        tuple(request.committed_pages),
                        request.version,
                        request.last_round,
                        active,
                    )
                )
            return LedgerSnapshot(
                allocator=self._allocator.snapshot(),
                requests=tuple(request_snapshots),
                aborted_epochs=tuple(
                    sorted(
                        self._aborted_epochs,
                        key=lambda epoch: (
                            _id_sort_key(epoch.request_id),
                            epoch.round_id,
                            epoch.version,
                        ),
                    )
                ),
                version_floors=tuple(
                    (request_id, self._version_floors[request_id])
                    for request_id in sorted(self._version_floors, key=_id_sort_key)
                ),
            )

    @classmethod
    def from_snapshot(cls, snapshot: LedgerSnapshot) -> SpeculativeLedger:
        """Restore, re-namespace, and fully audit a crash-recovery checkpoint."""

        if not isinstance(snapshot, LedgerSnapshot):
            raise SnapshotError("expected a LedgerSnapshot")
        if (
            isinstance(snapshot.schema_version, bool)
            or not isinstance(snapshot.schema_version, int)
            or snapshot.schema_version != _LEDGER_SNAPSHOT_VERSION
        ):
            raise SnapshotError(f"unsupported ledger snapshot version {snapshot.schema_version}")
        try:
            allocator = FixedPageAllocator.from_snapshot(snapshot.allocator)
            source_pool = snapshot.allocator.pool_id

            def rebase_span(span: PageSpan) -> PageSpan:
                if not isinstance(span, PageSpan):
                    raise InvariantViolation("snapshot contains a non-PageSpan")
                if span.handle.pool_id != source_pool:
                    raise InvariantViolation(
                        "ledger page namespace does not match allocator snapshot"
                    )
                return PageSpan(
                    PageHandle(
                        span.handle.page_id,
                        span.handle.generation,
                        allocator.pool_id,
                    ),
                    span.start,
                    span.used,
                )

            # Bypass the public constructor's empty-allocator precondition: the
            # snapshot's pages are about to be reattached and audited exactly.
            ledger = cls.__new__(cls)
            ledger._allocator = allocator
            ledger._requests = {}
            ledger._aborted_epochs = set(snapshot.aborted_epochs)
            if len(ledger._aborted_epochs) != len(snapshot.aborted_epochs):
                raise InvariantViolation("snapshot has duplicate aborted epochs")
            ledger._version_floors = dict(snapshot.version_floors)
            if len(ledger._version_floors) != len(snapshot.version_floors):
                raise InvariantViolation("snapshot has duplicate version-floor ids")
            ledger._lock = threading.RLock()
            for saved in snapshot.requests:
                _identifier(saved.request_id, field="request_id")
                if saved.request_id in ledger._requests:
                    raise InvariantViolation("snapshot has duplicate requests")
                active: _Transaction | None = None
                if saved.active is not None:
                    branches: dict[OutcomeId, _Branch] = {}
                    for branch in saved.active.branches:
                        _identifier(branch.outcome_id, field="outcome_id")
                        if branch.outcome_id in branches:
                            raise InvariantViolation("snapshot has duplicate outcomes")
                        branches[branch.outcome_id] = _Branch(
                            branch.outcome_id,
                            branch.appended_blocks,
                            [rebase_span(span) for span in branch.pages],
                            branch.cow_tail,
                        )
                    active = _Transaction(saved.active.epoch, branches)
                ledger._requests[saved.request_id] = _Request(
                    saved.request_id,
                    saved.committed_blocks,
                    [rebase_span(span) for span in saved.committed_pages],
                    saved.version,
                    saved.last_round,
                    active,
                )
            ledger._audit_unlocked()
            return ledger
        except (
            AttributeError,
            InvariantViolation,
            ValueError,
            TypeError,
            SnapshotError,
        ) as exc:
            if isinstance(exc, SnapshotError):
                raise
            raise SnapshotError(f"invalid ledger snapshot: {exc}") from exc

    def restore(self, snapshot: LedgerSnapshot) -> None:
        """Replace this ledger with a validated, freshly namespaced checkpoint."""

        restored = type(self).from_snapshot(snapshot)
        with self._lock:
            self._allocator = restored._allocator
            self._requests = restored._requests
            self._aborted_epochs = restored._aborted_epochs
            self._version_floors = restored._version_floors

    recover = from_snapshot

    def audit(self) -> None:
        """Run the complete allocator, ownership, and position invariant audit."""

        with self._lock:
            self._audit_unlocked()

    def _audit_unlocked(self) -> None:
        self._allocator.audit()
        tracked: dict[int, PageHandle] = {}

        for request_id, version_floor in self._version_floors.items():
            try:
                _identifier(request_id, field="version-floor request_id")
                _plain_int(version_floor, field="version_floor")
            except ValueError as exc:
                raise InvariantViolation(str(exc)) from exc

        def track(span: PageSpan, context: str, expected_owner: str) -> None:
            if not isinstance(span, PageSpan):
                raise InvariantViolation(f"{context} contains a non-PageSpan")
            prior = tracked.get(span.handle.page_id)
            if prior is not None:
                raise InvariantViolation(
                    f"physical page {span.handle.page_id} is referenced twice "
                    f"({prior!r} and {span.handle!r})"
                )
            tracked[span.handle.page_id] = span.handle
            try:
                if not self._allocator.is_live(span.handle):
                    raise InvariantViolation(f"{context} references stale {span.handle!r}")
                if self._allocator.owner_of(span.handle) != expected_owner:
                    raise InvariantViolation(f"{context} has incorrect allocator ownership")
            except InvalidPageHandleError as exc:
                raise InvariantViolation(
                    f"{context} references a foreign or invalid page handle"
                ) from exc

        for request_id, request in self._requests.items():
            if not isinstance(request, _Request):
                raise InvariantViolation("request table contains an invalid record")
            if request_id != request.request_id:
                raise InvariantViolation("request table key/id mismatch")
            try:
                _identifier(request_id, field="request_id")
                _plain_int(request.committed_blocks, field="committed_blocks")
                _plain_int(request.version, field="version")
            except ValueError as exc:
                raise InvariantViolation(str(exc)) from exc
            if (
                isinstance(request.last_round, bool)
                or not isinstance(request.last_round, int)
                or request.last_round < -1
            ):
                raise InvariantViolation("request epoch counters are invalid")
            if request.version == 0 and request.last_round != -1:
                raise InvariantViolation("request version/round initialization is inconsistent")
            if self._version_floors.get(request_id) != request.version:
                raise InvariantViolation("request version does not match its durable floor")
            self._audit_trunk_unlocked(request, track)
            if request.active is not None:
                transaction = request.active
                if not isinstance(transaction, _Transaction) or not isinstance(
                    transaction.epoch, LedgerEpoch
                ):
                    raise InvariantViolation("request has an invalid active transaction")
                if transaction.epoch.request_id != request_id:
                    raise InvariantViolation("active epoch belongs to another request")
                if transaction.epoch.version != request.version:
                    raise InvariantViolation("active epoch version does not match request")
                if transaction.epoch.round_id != request.last_round:
                    raise InvariantViolation("active epoch round does not match request")
                for outcome_id, branch in transaction.branches.items():
                    try:
                        _identifier(outcome_id, field="outcome_id")
                    except ValueError as exc:
                        raise InvariantViolation(str(exc)) from exc
                    if not isinstance(branch, _Branch):
                        raise InvariantViolation("outcome table contains an invalid branch")
                    if outcome_id != branch.outcome_id:
                        raise InvariantViolation("outcome table key/id mismatch")
                    self._audit_branch_unlocked(request, branch, track)

        live = self._allocator.live_handles()
        tracked_handles = frozenset(tracked.values())
        if live != tracked_handles:
            leaked = live - tracked_handles
            missing = tracked_handles - live
            raise InvariantViolation(
                f"ledger/allocator ownership mismatch; leaked={leaked}, stale={missing}"
            )
        for epoch in self._aborted_epochs:
            if not isinstance(epoch, LedgerEpoch):
                raise InvariantViolation("abort replay set contains an invalid epoch")
            aborted_request = self._requests.get(epoch.request_id)
            if aborted_request is not None and (
                epoch.version > aborted_request.version
                or epoch.round_id > aborted_request.last_round
            ):
                raise InvariantViolation("aborted epoch is newer than its request")
            if (
                aborted_request is not None
                and aborted_request.active is not None
                and aborted_request.active.epoch == epoch
            ):
                raise InvariantViolation("an aborted epoch is still active")

    def _audit_trunk_unlocked(
        self, request: _Request, track: Callable[[PageSpan, str, str], None]
    ) -> None:
        # ``track`` is kept as a local callable rather than a method so a single
        # audit can detect duplicates across all requests and branches.
        if request.committed_blocks < 0:
            raise InvariantViolation("committed block count is negative")
        expected_pages = self._pages_for_blocks(request.committed_blocks)
        if len(request.committed_pages) != expected_pages:
            raise InvariantViolation(
                "committed page count does not align with committed block count"
            )
        remaining = request.committed_blocks
        for index, span in enumerate(request.committed_pages):
            if span.start != index * self.page_size:
                raise InvariantViolation("committed page start is not position-aligned")
            expected_used = min(self.page_size, remaining)
            if span.used != expected_used or span.used > self.page_size:
                raise InvariantViolation("committed page occupancy is misaligned")
            remaining -= span.used
            track(
                span,
                f"request {request.request_id!r} trunk",
                self._trunk_owner(request.request_id),
            )
        if remaining != 0:
            raise InvariantViolation("committed page occupancy does not sum to count")

    def _audit_branch_unlocked(
        self,
        request: _Request,
        branch: _Branch,
        track: Callable[[PageSpan, str, str], None],
    ) -> None:
        try:
            _identifier(branch.outcome_id, field="outcome_id")
            _plain_int(branch.appended_blocks, field="appended_blocks")
        except ValueError as exc:
            raise InvariantViolation(str(exc)) from exc
        if not isinstance(branch.cow_tail, bool):
            raise InvariantViolation("branch COW-tail marker must be a bool")
        tail_used = request.committed_blocks % self.page_size
        expected_cow = branch.appended_blocks > 0 and tail_used > 0
        if branch.cow_tail != expected_cow:
            raise InvariantViolation("branch COW-tail marker is inconsistent")
        prefix = tail_used if branch.cow_tail else 0
        extent = prefix + branch.appended_blocks
        if len(branch.pages) != self._pages_for_blocks(extent):
            raise InvariantViolation("branch page count does not align with extent")
        start = request.committed_blocks - prefix
        logical_end = request.committed_blocks + branch.appended_blocks
        for index, span in enumerate(branch.pages):
            expected_start = start + index * self.page_size
            expected_used = min(self.page_size, logical_end - expected_start)
            if span.start != expected_start:
                raise InvariantViolation("branch page start is not position-aligned")
            if span.used != expected_used or not 1 <= span.used <= self.page_size:
                raise InvariantViolation("branch page occupancy is misaligned")
            if request.active is None:
                raise InvariantViolation("branch exists without an active epoch")
            track(
                span,
                f"outcome {branch.outcome_id!r}",
                self._branch_owner(request.active.epoch, branch.outcome_id),
            )

    def _accepted_pages_unlocked(
        self, request: _Request, branch: _Branch, accepted: int
    ) -> list[PageSpan]:
        if accepted == 0:
            return []
        prefix = request.committed_blocks % self.page_size if branch.cow_tail else 0
        desired_end = request.committed_blocks + accepted
        result: list[PageSpan] = []
        for span in branch.pages:
            if span.start >= desired_end:
                break
            used = min(self.page_size, desired_end - span.start)
            # The first COW page must retain the previously committed prefix.
            if used <= 0 or (not result and branch.cow_tail and used <= prefix):
                continue
            result.append(PageSpan(span.handle, span.start, used))
        if not result or result[-1].stop != desired_end:
            raise InvariantViolation("accepted prefix cannot be represented by branch")
        return result

    def _get_request_unlocked(self, request_id: RequestId) -> _Request:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise RequestNotFoundError(f"unknown request {request_id!r}") from exc

    def _require_active_unlocked(self, epoch: LedgerEpoch) -> tuple[_Request, _Transaction]:
        request = self._requests.get(epoch.request_id)
        if request is None or request.active is None or request.active.epoch != epoch:
            active = None if request is None or request.active is None else request.active.epoch
            raise StaleEpochError(f"epoch {epoch!r} is stale; current active epoch is {active!r}")
        return request, request.active

    def _view_unlocked(self, request: _Request) -> RequestView:
        outcomes: tuple[BranchView, ...] = ()
        active_epoch: LedgerEpoch | None = None
        if request.active is not None:
            active_epoch = request.active.epoch
            outcomes = tuple(
                self._branch_view(branch) for branch in self._sorted_branches(request.active)
            )
        return RequestView(
            request_id=request.request_id,
            committed_blocks=request.committed_blocks,
            committed_pages=tuple(request.committed_pages),
            version=request.version,
            last_round=request.last_round,
            active_epoch=active_epoch,
            outcomes=outcomes,
        )

    @staticmethod
    def _branch_view(branch: _Branch) -> BranchView:
        return BranchView(
            branch.outcome_id,
            branch.appended_blocks,
            tuple(branch.pages),
            branch.cow_tail,
        )

    @staticmethod
    def _sorted_branches(transaction: _Transaction) -> list[_Branch]:
        return [transaction.branches[key] for key in sorted(transaction.branches, key=_id_sort_key)]

    def _pages_for_blocks(self, blocks: int) -> int:
        if blocks == 0:
            return 0
        return (blocks + self.page_size - 1) // self.page_size

    def _spans_from_zero(self, handles: list[PageHandle], blocks: int) -> list[PageSpan]:
        return self._spans_from_start(handles, 0, blocks)

    def _spans_from_start(
        self, handles: list[PageHandle], start: int, blocks: int
    ) -> list[PageSpan]:
        spans: list[PageSpan] = []
        remaining = blocks
        for index, handle in enumerate(handles):
            used = min(self.page_size, remaining)
            spans.append(PageSpan(handle, start + index * self.page_size, used))
            remaining -= used
        if remaining != 0:
            raise InvariantViolation("allocated page count does not cover extent")
        return spans

    @staticmethod
    def _trunk_owner(request_id: RequestId) -> str:
        return f"trunk:{_owner_component(request_id)}"

    @staticmethod
    def _branch_owner(epoch: LedgerEpoch, outcome_id: OutcomeId) -> str:
        return (
            f"branch:{_owner_component(epoch.request_id)}:r{epoch.round_id}:"
            f"v{epoch.version}:o{_owner_component(outcome_id)}"
        )


__all__ = [
    "AllocatorSnapshot",
    "BranchSnapshot",
    "BranchView",
    "CommitResult",
    "DoubleReleaseError",
    "DuplicateOutcomeError",
    "DuplicateRequestError",
    "FixedPageAllocator",
    "ForeignPageHandleError",
    "ForkedAllocatorError",
    "InvalidCommitError",
    "InvalidConfigurationError",
    "InvalidPageHandleError",
    "InvariantViolation",
    "LedgerEpoch",
    "LedgerError",
    "LedgerSnapshot",
    "OutOfPagesError",
    "OutcomeId",
    "OutcomeNotFoundError",
    "PageHandle",
    "PageSpan",
    "PoolIdentity",
    "RequestId",
    "RequestNotFoundError",
    "RequestSnapshot",
    "RequestView",
    "SnapshotError",
    "SpeculativeLedger",
    "StaleEpochError",
    "StalePageHandleError",
    "TransactionConflictError",
    "TransactionSnapshot",
]
