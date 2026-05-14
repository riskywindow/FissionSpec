"""Versioned asynchronous protocol state for speculative verification.

Network replies can be duplicated, delayed past a retry, or arrive after a
request has already finished.  :class:`FissionProtocol` makes those cases
boring: every message carries ``(request_id, round_id, version)`` and reply
handlers apply a transition only when that identity exactly matches the one
currently in flight.  All other replies are reported as ignored without
mutating state.

The successful paths are::

    READY -> VERIFYING -> READY_HIT
                     +-> RECOVERING -> READY_BACKUP
                                   +-> FINISHED

``READY_HIT`` and ``READY_BACKUP`` retain the source of readiness for metrics;
either may directly start the next verification round.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypeAlias

ProtocolRequestId: TypeAlias = str | int
_SNAPSHOT_VERSION: Final[int] = 1


class ProtocolError(RuntimeError):
    """Base class for protocol construction and transition failures."""


class InvalidTransitionError(ProtocolError):
    """Raised when a local command is invalid in the current state."""


class InvalidMessageError(ProtocolError, ValueError):
    """Raised when constructing or passing a malformed protocol message."""


class ProtocolSnapshotError(ProtocolError, ValueError):
    """Raised when a protocol snapshot cannot be safely restored."""


def _identifier(value: object, *, field: str = "request_id") -> ProtocolRequestId:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise InvalidMessageError(f"{field} must be a str or int (but not bool)")
    return value


def _integer(value: object, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidMessageError(f"{field} must be an integer")
    if value < minimum:
        raise InvalidMessageError(f"{field} must be at least {minimum}")
    return value


class ProtocolState(StrEnum):
    """Observable state of a request's speculative serving protocol."""

    READY = "ready"
    VERIFYING = "verifying"
    READY_HIT = "ready_hit"
    RECOVERING = "recovering"
    READY_BACKUP = "ready_backup"
    FINISHED = "finished"


class ReplyDisposition(StrEnum):
    """Whether an asynchronous reply caused a state transition."""

    APPLIED = "applied"
    IGNORED_STALE = "ignored_stale"


@dataclass(frozen=True, slots=True)
class MessageTag:
    """Complete identity required on every request and asynchronous reply."""

    request_id: ProtocolRequestId
    round_id: int
    version: int

    def __post_init__(self) -> None:
        _identifier(self.request_id)
        _integer(self.round_id, field="round_id", minimum=0)
        _integer(self.version, field="version", minimum=1)


# ``ProtocolEpoch`` reads more naturally at call sites that coordinate this
# state machine with the page ledger.  It is an exact type alias, not a wrapper.
ProtocolEpoch = MessageTag


@dataclass(frozen=True, slots=True)
class VerifyRequest:
    """Command sent to the target verifier for one round/version."""

    tag: MessageTag

    def __post_init__(self) -> None:
        if not isinstance(self.tag, MessageTag):
            raise InvalidMessageError("tag must be a MessageTag")


@dataclass(frozen=True, slots=True)
class VerifyReply:
    """Verifier response; misses carry the prefix retained before recovery."""

    tag: MessageTag
    hit: bool
    accepted_blocks: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.tag, MessageTag):
            raise InvalidMessageError("tag must be a MessageTag")
        if not isinstance(self.hit, bool):
            raise InvalidMessageError("hit must be a bool")
        _integer(self.accepted_blocks, field="accepted_blocks", minimum=0)

    @property
    def cache_hit(self) -> bool:
        """Descriptive alias for :attr:`hit`."""

        return self.hit


@dataclass(frozen=True, slots=True)
class RecoveryRequest:
    """Command to recover the suffix not supplied by the speculative hit."""

    tag: MessageTag
    accepted_blocks: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.tag, MessageTag):
            raise InvalidMessageError("tag must be a MessageTag")
        _integer(self.accepted_blocks, field="accepted_blocks", minimum=0)


@dataclass(frozen=True, slots=True)
class RecoveryReply:
    """Backup response, optionally declaring that the request is complete."""

    tag: MessageTag
    finished: bool = False
    recovered_blocks: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.tag, MessageTag):
            raise InvalidMessageError("tag must be a MessageTag")
        if not isinstance(self.finished, bool):
            raise InvalidMessageError("finished must be a bool")
        _integer(self.recovered_blocks, field="recovered_blocks", minimum=0)


OutboundMessage: TypeAlias = VerifyRequest | RecoveryRequest


@dataclass(frozen=True, slots=True)
class ProtocolTransition:
    """Result of handling a reply, including any newly emitted command."""

    disposition: ReplyDisposition
    previous: ProtocolState
    current: ProtocolState
    tag: MessageTag
    outbound: RecoveryRequest | None = None

    @property
    def applied(self) -> bool:
        return self.disposition is ReplyDisposition.APPLIED

    @property
    def ignored(self) -> bool:
        return not self.applied


@dataclass(frozen=True, slots=True)
class ProtocolSnapshot:
    """Immutable state-machine checkpoint suitable for persistence."""

    request_id: ProtocolRequestId
    state: ProtocolState
    round_id: int
    version: int
    active_tag: MessageTag | None
    accepted_blocks: int
    schema_version: int = _SNAPSHOT_VERSION


@dataclass(frozen=True, slots=True)
class CrashRecovery:
    """State and optional command produced when resuming a checkpoint."""

    protocol: FissionProtocol
    outbound: RecoveryRequest | None


_READY_STATES: Final[frozenset[ProtocolState]] = frozenset(
    {ProtocolState.READY, ProtocolState.READY_HIT, ProtocolState.READY_BACKUP}
)


class FissionProtocol:
    """Thread-safe per-request speculative verification state machine."""

    def __init__(self, request_id: ProtocolRequestId) -> None:
        self._request_id = _identifier(request_id)
        self._state = ProtocolState.READY
        self._round_id = -1
        self._version = 0
        self._active_tag: MessageTag | None = None
        self._accepted_blocks = 0
        self._lock = threading.RLock()

    @property
    def request_id(self) -> ProtocolRequestId:
        return self._request_id

    @property
    def state(self) -> ProtocolState:
        with self._lock:
            return self._state

    @property
    def round_id(self) -> int:
        with self._lock:
            return self._round_id

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def active_tag(self) -> MessageTag | None:
        with self._lock:
            return self._active_tag

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._state in _READY_STATES

    @property
    def is_finished(self) -> bool:
        with self._lock:
            return self._state is ProtocolState.FINISHED

    def start_verification(self, round_id: int | None = None) -> VerifyRequest:
        """Start the next round from any ready state.

        If omitted, ``round_id`` advances by one.  Explicit round numbers must
        be strictly monotonic; retrying the current round is instead performed
        with :meth:`retry_verification`, which advances ``version``.
        """

        with self._lock:
            if self._state not in _READY_STATES:
                raise InvalidTransitionError(f"cannot start verification from {self._state.value}")
            candidate = self._round_id + 1 if round_id is None else round_id
            try:
                candidate = _integer(candidate, field="round_id", minimum=0)
            except InvalidMessageError as exc:
                raise InvalidTransitionError(str(exc)) from exc
            if candidate <= self._round_id:
                raise InvalidTransitionError(
                    f"round {candidate} is not newer than {self._round_id}"
                )
            self._round_id = candidate
            self._version += 1
            self._accepted_blocks = 0
            self._active_tag = MessageTag(self._request_id, self._round_id, self._version)
            self._state = ProtocolState.VERIFYING
            return VerifyRequest(self._active_tag)

    begin_verification = start_verification
    start_round = start_verification

    def retry_verification(self) -> VerifyRequest:
        """Invalidate an in-flight attempt and retry its round at a new version.

        This state machine does not own KV pages.  A coordinator using
        :class:`fissionspec.ledger.SpeculativeLedger` must also call its
        ``retry(old_epoch)`` method and retain the explicit association between
        the returned :class:`VerifyRequest` tag and replacement ledger epoch.
        Their numeric versions are independent incarnation counters and must
        not be joined by equality.
        """

        with self._lock:
            if self._state is not ProtocolState.VERIFYING:
                raise InvalidTransitionError(f"cannot retry verification from {self._state.value}")
            self._version += 1
            self._accepted_blocks = 0
            self._active_tag = MessageTag(self._request_id, self._round_id, self._version)
            return VerifyRequest(self._active_tag)

    def handle_verify_reply(self, reply: VerifyReply) -> ProtocolTransition:
        """Apply an exact in-flight verifier reply or ignore it as stale."""

        if not isinstance(reply, VerifyReply):
            raise InvalidMessageError("expected a VerifyReply")
        with self._lock:
            previous = self._state
            if self._state is not ProtocolState.VERIFYING or self._active_tag != reply.tag:
                return ProtocolTransition(
                    ReplyDisposition.IGNORED_STALE,
                    previous,
                    self._state,
                    reply.tag,
                )

            self._accepted_blocks = reply.accepted_blocks
            outbound: RecoveryRequest | None = None
            if reply.hit:
                self._state = ProtocolState.READY_HIT
                self._active_tag = None
            else:
                self._state = ProtocolState.RECOVERING
                # Recovery belongs to exactly the same round/version.  A late
                # duplicate VerifyReply is ignored because state is no longer
                # VERIFYING, even though its tag matches.
                outbound = RecoveryRequest(reply.tag, reply.accepted_blocks)
            return ProtocolTransition(
                ReplyDisposition.APPLIED,
                previous,
                self._state,
                reply.tag,
                outbound,
            )

    on_verify_reply = handle_verify_reply

    def handle_recovery_reply(self, reply: RecoveryReply) -> ProtocolTransition:
        """Apply an exact recovery reply or ignore it as stale/duplicated."""

        if not isinstance(reply, RecoveryReply):
            raise InvalidMessageError("expected a RecoveryReply")
        with self._lock:
            previous = self._state
            if self._state is not ProtocolState.RECOVERING or self._active_tag != reply.tag:
                return ProtocolTransition(
                    ReplyDisposition.IGNORED_STALE,
                    previous,
                    self._state,
                    reply.tag,
                )
            self._state = ProtocolState.FINISHED if reply.finished else ProtocolState.READY_BACKUP
            self._active_tag = None
            return ProtocolTransition(
                ReplyDisposition.APPLIED,
                previous,
                self._state,
                reply.tag,
            )

    on_recovery_reply = handle_recovery_reply

    def invalidate_inflight_for_recovery(self) -> RecoveryRequest:
        """Fence old replies and enter/restart recovery at a fresh version.

        This is the local crash/timeout escape hatch.  It may be called from
        VERIFYING or RECOVERING; the version bump guarantees that replies sent
        before the fence cannot mutate the recovered state.
        """

        with self._lock:
            if self._state not in {
                ProtocolState.VERIFYING,
                ProtocolState.RECOVERING,
            }:
                raise InvalidTransitionError(
                    f"cannot recover in-flight work from {self._state.value}"
                )
            self._version += 1
            self._active_tag = MessageTag(self._request_id, self._round_id, self._version)
            self._state = ProtocolState.RECOVERING
            return RecoveryRequest(self._active_tag, self._accepted_blocks)

    def finish(self) -> None:
        """Finish a ready request locally and fence any older message versions."""

        with self._lock:
            if self._state not in _READY_STATES:
                raise InvalidTransitionError(f"cannot finish from {self._state.value}")
            self._version += 1
            self._state = ProtocolState.FINISHED
            self._active_tag = None

    def snapshot(self) -> ProtocolSnapshot:
        """Capture a complete state-machine checkpoint."""

        with self._lock:
            snapshot = ProtocolSnapshot(
                request_id=self._request_id,
                state=self._state,
                round_id=self._round_id,
                version=self._version,
                active_tag=self._active_tag,
                accepted_blocks=self._accepted_blocks,
            )
            self._validate_snapshot(snapshot)
            return snapshot

    recovery_snapshot = snapshot

    @classmethod
    def from_snapshot(cls, snapshot: ProtocolSnapshot) -> FissionProtocol:
        """Restore a checkpoint exactly, including an in-flight identity."""

        if not isinstance(snapshot, ProtocolSnapshot):
            raise ProtocolSnapshotError("expected a ProtocolSnapshot")
        try:
            cls._validate_snapshot(snapshot)
            protocol = cls(snapshot.request_id)
            protocol._state = snapshot.state
            protocol._round_id = snapshot.round_id
            protocol._version = snapshot.version
            protocol._active_tag = snapshot.active_tag
            protocol._accepted_blocks = snapshot.accepted_blocks
            return protocol
        except (InvalidMessageError, InvalidTransitionError, TypeError) as exc:
            raise ProtocolSnapshotError(f"invalid protocol snapshot: {exc}") from exc

    restore = from_snapshot

    @classmethod
    def resume_after_crash(cls, snapshot: ProtocolSnapshot) -> CrashRecovery:
        """Restore a checkpoint and fence messages that predate the crash.

        Stable ready/finished snapshots are returned unchanged.  VERIFYING and
        RECOVERING snapshots are moved to RECOVERING at a fresh version and
        yield the command that must be (re)sent to the backup.
        """

        protocol = cls.from_snapshot(snapshot)
        outbound: RecoveryRequest | None = None
        if protocol.state in {ProtocolState.VERIFYING, ProtocolState.RECOVERING}:
            outbound = protocol.invalidate_inflight_for_recovery()
        return CrashRecovery(protocol, outbound)

    @staticmethod
    def _validate_snapshot(snapshot: ProtocolSnapshot) -> None:
        if (
            isinstance(snapshot.schema_version, bool)
            or not isinstance(snapshot.schema_version, int)
            or snapshot.schema_version != _SNAPSHOT_VERSION
        ):
            raise ProtocolSnapshotError(
                f"unsupported protocol snapshot version {snapshot.schema_version}"
            )
        _identifier(snapshot.request_id)
        if not isinstance(snapshot.state, ProtocolState):
            raise InvalidMessageError("state must be a ProtocolState")
        if isinstance(snapshot.round_id, bool) or not isinstance(snapshot.round_id, int):
            raise InvalidMessageError("round_id must be an integer")
        if snapshot.round_id < -1:
            raise InvalidMessageError("round_id must be at least -1")
        if isinstance(snapshot.version, bool) or not isinstance(snapshot.version, int):
            raise InvalidMessageError("version must be an integer")
        if snapshot.version < 0:
            raise InvalidMessageError("version must be non-negative")
        _integer(snapshot.accepted_blocks, field="accepted_blocks", minimum=0)

        in_flight = snapshot.state in {
            ProtocolState.VERIFYING,
            ProtocolState.RECOVERING,
        }
        if in_flight:
            if snapshot.active_tag is None:
                raise InvalidMessageError("in-flight state requires an active tag")
            expected = MessageTag(snapshot.request_id, snapshot.round_id, snapshot.version)
            if snapshot.active_tag != expected:
                raise InvalidMessageError("active tag does not match snapshot epoch")
        elif snapshot.active_tag is not None:
            raise InvalidMessageError("stable state must not retain an active tag")

        if snapshot.state is ProtocolState.READY:
            if snapshot.round_id != -1 or snapshot.version != 0:
                raise InvalidMessageError("initial READY state has invalid counters")
            if snapshot.accepted_blocks != 0:
                raise InvalidMessageError("initial READY state cannot retain accepted blocks")
        elif (
            snapshot.state is not ProtocolState.FINISHED and snapshot.round_id < 0
        ) or snapshot.version < 1:
            raise InvalidMessageError("non-initial state requires a valid epoch")


__all__ = [
    "CrashRecovery",
    "FissionProtocol",
    "InvalidMessageError",
    "InvalidTransitionError",
    "MessageTag",
    "OutboundMessage",
    "ProtocolEpoch",
    "ProtocolError",
    "ProtocolRequestId",
    "ProtocolSnapshot",
    "ProtocolSnapshotError",
    "ProtocolState",
    "ProtocolTransition",
    "RecoveryReply",
    "RecoveryRequest",
    "ReplyDisposition",
    "VerifyReply",
    "VerifyRequest",
]
