"""Exact bounded oracle for finite row/slot scheduling traces.

Unlike :mod:`fissionspec.oracle`, which searches two coalescing actions inside
the serving simulator, this module solves a deliberately small, self-contained
offline scheduling problem.  Every job is pre-realized and completes in one
non-preemptive target launch.  At an idle decision point the oracle may:

* dispatch any ordered, non-empty subset of released jobs that fits row and
  verifier-slot capacity; or
* wait to any explicitly configured release, deadline-safe, or grid point.

Times, weights, latency values, and objectives use :class:`fractions.Fraction`.
The returned certificate contains a replayable event trace, a canonical input
hash, and exact search accounting.  A separate verifier replays the trace and
can independently re-enumerate the complete tree without the optimizer's
memoization or dominance reductions.

This is an exponential reference oracle for adversarial tiny traces, not a
production scheduler.  Mandatory limits fail closed instead of returning an
unlabelled best-so-far result.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from itertools import combinations, permutations
from typing import Final, Protocol, TypeAlias

ExactInput: TypeAlias = Fraction | int
JobId: TypeAlias = str

_INPUT_DOMAIN: Final[bytes] = b"fissionspec/general-oracle/input/v1\x00"
_CERTIFICATE_DOMAIN: Final[bytes] = b"fissionspec/general-oracle/certificate/v1\x00"
_SCHEMA_VERSION: Final[int] = 1


class GeneralOracleError(ValueError):
    """Base class for invalid bounded-oracle inputs and traces."""


class MissingLatencyShapeError(GeneralOracleError):
    """Raised when an admissible batch has no latency-surface entry."""


class CertificateVerificationError(GeneralOracleError):
    """Raised when a certificate cannot be replayed or proved optimal."""


class GeneralOracleLimitKind(StrEnum):
    """Hard bounds enforced by the optimizer and independent verifier."""

    JOBS = "max_jobs"
    STATES = "max_states"
    TRANSITIONS = "max_transitions"
    TRACE_EVENTS = "max_trace_events"
    VERIFIER_NODES = "max_verifier_nodes"


class GeneralOracleLimitExceeded(RuntimeError):
    """Raised rather than silently weakening an exactness claim."""

    def __init__(
        self,
        kind: GeneralOracleLimitKind,
        limit: int,
        observed: int,
    ) -> None:
        self.kind = kind
        self.limit = limit
        self.observed = observed
        super().__init__(f"{kind.value}={limit} exceeded (required at least {observed})")


class _HashSink(Protocol):
    def update(self, value: bytes, /) -> None: ...


def _plain_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GeneralOracleError(f"{field} must be an integer")
    if value < minimum:
        raise GeneralOracleError(f"{field} must be at least {minimum}")
    return value


def _exact(
    value: object,
    *,
    field: str,
    minimum: Fraction = Fraction(),
    positive: bool = False,
) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        raise GeneralOracleError(f"{field} must be an int or Fraction")
    result = Fraction(value)
    if positive and result <= 0:
        raise GeneralOracleError(f"{field} must be positive")
    if not positive and result < minimum:
        raise GeneralOracleError(f"{field} must be at least {minimum}")
    return result


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise GeneralOracleError(f"{field} must be a non-empty string")
    return value


def _frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def _encode_integer(value: int) -> bytes:
    sign = b"-" if value < 0 else b"+"
    magnitude = abs(value)
    width = max(1, (magnitude.bit_length() + 7) // 8)
    return sign + magnitude.to_bytes(width, "big")


def _encode_fraction(value: Fraction) -> bytes:
    return _frame(_encode_integer(value.numerator)) + _frame(_encode_integer(value.denominator))


def _update_fraction(digest: _HashSink, value: Fraction) -> None:
    digest.update(_encode_fraction(value))


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise GeneralOracleError(f"{field} must be 64 lowercase hexadecimal characters")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise GeneralOracleError(f"{field} must be hexadecimal") from exc
    if len(decoded) != 32:
        raise GeneralOracleError(f"{field} must encode 256 bits")
    return value


@dataclass(frozen=True, slots=True)
class OracleJob:
    """One immutable row that completes in a single target launch."""

    job_id: JobId
    release_time: Fraction
    width: int
    deadline: Fraction
    weight: Fraction = Fraction(1)
    cohort_id: str | None = None

    def __post_init__(self) -> None:
        job_id = _nonempty_string(self.job_id, field="job_id")
        release = _exact(self.release_time, field=f"{job_id}.release_time")
        width = _plain_int(self.width, field=f"{job_id}.width", minimum=1)
        deadline = _exact(self.deadline, field=f"{job_id}.deadline")
        weight = _exact(self.weight, field=f"{job_id}.weight", positive=True)
        if deadline < release:
            raise GeneralOracleError(f"{job_id}.deadline must not precede release_time")
        if self.cohort_id is not None:
            _nonempty_string(self.cohort_id, field=f"{job_id}.cohort_id")
        object.__setattr__(self, "release_time", release)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "deadline", deadline)
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class OracleCapacity:
    """Physical row and verifier-slot limits for one launch."""

    row_limit: int
    slot_limit: int

    def __post_init__(self) -> None:
        _plain_int(self.row_limit, field="row_limit", minimum=1)
        _plain_int(self.slot_limit, field="slot_limit", minimum=1)


@dataclass(frozen=True, init=False, slots=True)
class ExactLatencySurface:
    """A finite exact map from ``(batch rows, verifier slots)`` to duration."""

    entries: tuple[tuple[int, int, Fraction], ...]

    def __init__(
        self,
        entries: Mapping[tuple[int, int], ExactInput],
    ) -> None:
        if not isinstance(entries, Mapping) or not entries:
            raise GeneralOracleError("latency entries must be a non-empty mapping")
        canonical: list[tuple[int, int, Fraction]] = []
        for shape, raw_duration in entries.items():
            if not isinstance(shape, tuple) or len(shape) != 2:
                raise GeneralOracleError("latency keys must be (rows, slots) tuples")
            rows = _plain_int(shape[0], field="latency rows", minimum=1)
            slots = _plain_int(shape[1], field="latency slots", minimum=1)
            if slots < rows:
                raise GeneralOracleError("latency slots must be at least rows")
            duration = _exact(raw_duration, field=f"latency[{shape!r}]", positive=True)
            canonical.append((rows, slots, duration))
        canonical.sort()
        object.__setattr__(self, "entries", tuple(canonical))

    def duration(self, rows: int, slots: int) -> Fraction:
        """Return an exact duration, failing closed for an unknown shape."""

        batch_rows = _plain_int(rows, field="rows", minimum=1)
        verifier_slots = _plain_int(slots, field="slots", minimum=1)
        for known_rows, known_slots, duration in self.entries:
            if (known_rows, known_slots) == (batch_rows, verifier_slots):
                return duration
        raise MissingLatencyShapeError(
            f"latency surface has no entry for ({batch_rows}, {verifier_slots})"
        )


class WaitKind(StrEnum):
    """Why an idle target advances to a later decision point."""

    FORCED_RELEASE = "forced-release"
    RELEASE = "release"
    DEADLINE_SAFE = "deadline-safe"
    GRID = "grid"


_WAIT_PRIORITY: Final[dict[WaitKind, int]] = {
    WaitKind.RELEASE: 0,
    WaitKind.DEADLINE_SAFE: 1,
    WaitKind.GRID: 2,
    WaitKind.FORCED_RELEASE: 3,
}


@dataclass(frozen=True, slots=True)
class OracleWaitConfig:
    """Finite optional-wait action space.

    ``grid_times`` and ``latest_optional_time`` are absolute times.  Release
    and deadline-safe points are derived from the problem when enabled.
    """

    include_release_times: bool = True
    include_deadline_safe_times: bool = True
    grid_times: tuple[Fraction, ...] = ()
    latest_optional_time: Fraction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.include_release_times, bool):
            raise GeneralOracleError("include_release_times must be a bool")
        if not isinstance(self.include_deadline_safe_times, bool):
            raise GeneralOracleError("include_deadline_safe_times must be a bool")
        grid = tuple(
            _exact(value, field=f"grid_times[{index}]")
            for index, value in enumerate(self.grid_times)
        )
        if len(grid) != len(set(grid)):
            raise GeneralOracleError("grid_times must not contain duplicates")
        grid = tuple(sorted(grid))
        latest = self.latest_optional_time
        if latest is not None:
            latest = _exact(latest, field="latest_optional_time")
        object.__setattr__(self, "grid_times", grid)
        object.__setattr__(self, "latest_optional_time", latest)


def _admissible_shapes(
    jobs: tuple[OracleJob, ...],
    capacity: OracleCapacity,
) -> set[tuple[int, int]]:
    shapes: set[tuple[int, int]] = set()
    maximum = min(len(jobs), capacity.row_limit)
    for size in range(1, maximum + 1):
        for selected in combinations(jobs, size):
            slots = sum(job.width for job in selected)
            if slots <= capacity.slot_limit:
                shapes.add((size, slots))
    return shapes


def _add_wait_point(
    points: dict[Fraction, WaitKind],
    time: Fraction,
    kind: WaitKind,
) -> None:
    previous = points.get(time)
    if previous is None or _WAIT_PRIORITY[kind] < _WAIT_PRIORITY[previous]:
        points[time] = kind


def _decision_points(
    jobs: tuple[OracleJob, ...],
    capacity: OracleCapacity,
    latency: ExactLatencySurface,
    wait: OracleWaitConfig,
    start_time: Fraction,
) -> tuple[tuple[Fraction, WaitKind], ...]:
    points: dict[Fraction, WaitKind] = {}
    if wait.include_release_times:
        for job in jobs:
            _add_wait_point(points, job.release_time, WaitKind.RELEASE)
    if wait.include_deadline_safe_times:
        maximum = min(len(jobs), capacity.row_limit)
        for size in range(1, maximum + 1):
            for selected in combinations(jobs, size):
                slots = sum(job.width for job in selected)
                if slots > capacity.slot_limit:
                    continue
                duration = latency.duration(size, slots)
                for job in selected:
                    safe_time = job.deadline - duration
                    if safe_time >= start_time:
                        _add_wait_point(points, safe_time, WaitKind.DEADLINE_SAFE)
    for time in wait.grid_times:
        _add_wait_point(points, time, WaitKind.GRID)

    latest = wait.latest_optional_time
    return tuple(
        (time, kind)
        for time, kind in sorted(points.items())
        if time >= start_time and (latest is None or time <= latest)
    )


def _problem_hash(
    jobs: tuple[OracleJob, ...],
    capacity: OracleCapacity,
    latency: ExactLatencySurface,
    wait: OracleWaitConfig,
    start_time: Fraction,
) -> str:
    digest = hashlib.sha256()
    digest.update(_INPUT_DOMAIN)
    digest.update(_frame(_encode_integer(_SCHEMA_VERSION)))
    _update_fraction(digest, start_time)
    digest.update(_frame(_encode_integer(capacity.row_limit)))
    digest.update(_frame(_encode_integer(capacity.slot_limit)))
    digest.update(_frame(_encode_integer(len(jobs))))
    for job in jobs:
        digest.update(_frame(job.job_id.encode("utf-8")))
        _update_fraction(digest, job.release_time)
        digest.update(_frame(_encode_integer(job.width)))
        _update_fraction(digest, job.deadline)
        _update_fraction(digest, job.weight)
        if job.cohort_id is None:
            digest.update(b"\x00")
        else:
            digest.update(b"\x01" + _frame(job.cohort_id.encode("utf-8")))
    digest.update(_frame(_encode_integer(len(latency.entries))))
    for rows, slots, duration in latency.entries:
        digest.update(_frame(_encode_integer(rows)))
        digest.update(_frame(_encode_integer(slots)))
        _update_fraction(digest, duration)
    digest.update(b"\x01" if wait.include_release_times else b"\x00")
    digest.update(b"\x01" if wait.include_deadline_safe_times else b"\x00")
    digest.update(_frame(_encode_integer(len(wait.grid_times))))
    for time in wait.grid_times:
        _update_fraction(digest, time)
    if wait.latest_optional_time is None:
        digest.update(b"\x00")
    else:
        digest.update(b"\x01")
        _update_fraction(digest, wait.latest_optional_time)
    return digest.hexdigest()


@dataclass(frozen=True, init=False, slots=True)
class OracleProblem:
    """Canonical finite scheduling instance."""

    jobs: tuple[OracleJob, ...]
    capacity: OracleCapacity
    latency: ExactLatencySurface
    wait: OracleWaitConfig
    start_time: Fraction
    decision_points: tuple[tuple[Fraction, WaitKind], ...]
    input_hash: str

    def __init__(
        self,
        jobs: Sequence[OracleJob],
        capacity: OracleCapacity,
        latency: ExactLatencySurface,
        *,
        wait: OracleWaitConfig | None = None,
        start_time: ExactInput = 0,
    ) -> None:
        if not isinstance(jobs, Sequence) or not jobs:
            raise GeneralOracleError("jobs must be a non-empty sequence")
        supplied_jobs = tuple(jobs)
        if any(not isinstance(job, OracleJob) for job in supplied_jobs):
            raise GeneralOracleError("every job must be an OracleJob")
        canonical_jobs = tuple(sorted(supplied_jobs, key=lambda job: job.job_id))
        identifiers = [job.job_id for job in canonical_jobs]
        if len(identifiers) != len(set(identifiers)):
            raise GeneralOracleError("job_id values must be unique")
        if not isinstance(capacity, OracleCapacity):
            raise GeneralOracleError("capacity must be an OracleCapacity")
        if not isinstance(latency, ExactLatencySurface):
            raise GeneralOracleError("latency must be an ExactLatencySurface")
        wait_config = OracleWaitConfig() if wait is None else wait
        if not isinstance(wait_config, OracleWaitConfig):
            raise GeneralOracleError("wait must be an OracleWaitConfig")
        origin = _exact(start_time, field="start_time")
        if (
            wait_config.latest_optional_time is not None
            and wait_config.latest_optional_time < origin
        ):
            raise GeneralOracleError("latest_optional_time must not precede start_time")

        cohort_releases: dict[str, Fraction] = {}
        for job in canonical_jobs:
            if job.width > capacity.slot_limit:
                raise GeneralOracleError(f"job {job.job_id!r} width exceeds slot capacity")
            if job.cohort_id is not None:
                previous = cohort_releases.setdefault(job.cohort_id, job.release_time)
                if previous != job.release_time:
                    raise GeneralOracleError(f"cohort {job.cohort_id!r} must have one release time")

        shapes = _admissible_shapes(canonical_jobs, capacity)
        for rows, slots in sorted(shapes):
            try:
                latency.duration(rows, slots)
            except MissingLatencyShapeError as exc:
                raise MissingLatencyShapeError(
                    f"admissible shape ({rows}, {slots}) is missing"
                ) from exc

        points = _decision_points(
            canonical_jobs,
            capacity,
            latency,
            wait_config,
            origin,
        )
        fingerprint = _problem_hash(
            canonical_jobs,
            capacity,
            latency,
            wait_config,
            origin,
        )
        object.__setattr__(self, "jobs", canonical_jobs)
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "latency", latency)
        object.__setattr__(self, "wait", wait_config)
        object.__setattr__(self, "start_time", origin)
        object.__setattr__(self, "decision_points", points)
        object.__setattr__(self, "input_hash", fingerprint)


@dataclass(frozen=True, slots=True)
class DispatchEvent:
    """One ordered target launch in an oracle certificate."""

    start_time: Fraction
    end_time: Fraction
    job_ids: tuple[JobId, ...]
    rows: int
    slots: int

    def __post_init__(self) -> None:
        start = _exact(self.start_time, field="dispatch start_time")
        end = _exact(self.end_time, field="dispatch end_time")
        if end <= start:
            raise GeneralOracleError("dispatch end_time must follow start_time")
        if not self.job_ids:
            raise GeneralOracleError("dispatch job_ids must not be empty")
        for index, job_id in enumerate(self.job_ids):
            _nonempty_string(job_id, field=f"dispatch job_ids[{index}]")
        if len(self.job_ids) != len(set(self.job_ids)):
            raise GeneralOracleError("dispatch job_ids must be unique")
        rows = _plain_int(self.rows, field="dispatch rows", minimum=1)
        slots = _plain_int(self.slots, field="dispatch slots", minimum=1)
        if rows != len(self.job_ids):
            raise GeneralOracleError("dispatch rows must equal len(job_ids)")
        if slots < rows:
            raise GeneralOracleError("dispatch slots must be at least rows")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "slots", slots)


@dataclass(frozen=True, slots=True)
class WaitEvent:
    """An explicit or forced idle interval."""

    start_time: Fraction
    end_time: Fraction
    kind: WaitKind

    def __post_init__(self) -> None:
        start = _exact(self.start_time, field="wait start_time")
        end = _exact(self.end_time, field="wait end_time")
        if end <= start:
            raise GeneralOracleError("wait end_time must follow start_time")
        if not isinstance(self.kind, WaitKind):
            raise GeneralOracleError("wait kind must be a WaitKind")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)


ScheduleEvent: TypeAlias = DispatchEvent | WaitEvent


@dataclass(frozen=True, order=True, slots=True)
class OracleObjective:
    """Lexicographic objective: violations, then exact weighted flow."""

    deadline_violations: int
    weighted_flow: Fraction

    def __post_init__(self) -> None:
        violations = _plain_int(
            self.deadline_violations,
            field="deadline_violations",
        )
        flow = _exact(self.weighted_flow, field="weighted_flow")
        object.__setattr__(self, "deadline_violations", violations)
        object.__setattr__(self, "weighted_flow", flow)

    def __add__(self, other: OracleObjective) -> OracleObjective:
        if not isinstance(other, OracleObjective):
            return NotImplemented
        return OracleObjective(
            self.deadline_violations + other.deadline_violations,
            self.weighted_flow + other.weighted_flow,
        )


@dataclass(frozen=True, slots=True)
class OracleSearchLimits:
    """Mandatory fail-closed complexity bounds."""

    max_jobs: int
    max_states: int
    max_transitions: int
    max_trace_events: int

    def __post_init__(self) -> None:
        _plain_int(self.max_jobs, field="max_jobs", minimum=1)
        _plain_int(self.max_states, field="max_states", minimum=1)
        _plain_int(self.max_transitions, field="max_transitions", minimum=1)
        _plain_int(self.max_trace_events, field="max_trace_events", minimum=1)


def _encode_event(event: ScheduleEvent) -> bytes:
    if isinstance(event, DispatchEvent):
        encoded = bytearray(b"D")
        encoded.extend(_encode_fraction(event.start_time))
        encoded.extend(_encode_fraction(event.end_time))
        encoded.extend(_frame(_encode_integer(event.rows)))
        encoded.extend(_frame(_encode_integer(event.slots)))
        for job_id in event.job_ids:
            encoded.extend(_frame(job_id.encode("utf-8")))
        return bytes(encoded)
    encoded = bytearray(b"W")
    encoded.extend(_encode_fraction(event.start_time))
    encoded.extend(_encode_fraction(event.end_time))
    encoded.extend(_frame(event.kind.value.encode("ascii")))
    return bytes(encoded)


def _trace_key(events: tuple[ScheduleEvent, ...]) -> tuple[bytes, ...]:
    """Deterministic tertiary objective; dispatch precedes wait on ties."""

    return tuple(_encode_event(event) for event in events)


@dataclass(frozen=True, slots=True)
class GeneralOracleCertificate:
    """Replay-verifiable exact optimum and search accounting."""

    input_hash: str
    events: tuple[ScheduleEvent, ...]
    objective: OracleObjective
    completion_times: tuple[tuple[JobId, Fraction], ...]
    states_explored: int
    states_pruned_by_memo: int
    transitions_explored: int
    transitions_pruned_by_dominance: int
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_digest(self.input_hash, field="input_hash")
        schema_version = _plain_int(
            self.schema_version,
            field="schema_version",
            minimum=1,
        )
        if schema_version != _SCHEMA_VERSION:
            raise GeneralOracleError(
                f"unsupported certificate schema_version {self.schema_version}"
            )
        if not isinstance(self.objective, OracleObjective):
            raise GeneralOracleError("objective must be an OracleObjective")
        events = tuple(self.events)
        if any(not isinstance(event, (DispatchEvent, WaitEvent)) for event in events):
            raise GeneralOracleError("events must contain only DispatchEvent or WaitEvent values")
        completions: list[tuple[JobId, Fraction]] = []
        for index, item in enumerate(self.completion_times):
            if not isinstance(item, tuple) or len(item) != 2:
                raise GeneralOracleError(
                    f"completion_times[{index}] must be a (job_id, time) tuple"
                )
            job_id = _nonempty_string(
                item[0],
                field=f"completion_times[{index}].job_id",
            )
            completion = _exact(
                item[1],
                field=f"completion_times[{index}].time",
            )
            completions.append((job_id, completion))
        if len(completions) != len({job_id for job_id, _ in completions}):
            raise GeneralOracleError("completion_times job IDs must be unique")
        for field, value in (
            ("states_explored", self.states_explored),
            ("states_pruned_by_memo", self.states_pruned_by_memo),
            ("transitions_explored", self.transitions_explored),
            (
                "transitions_pruned_by_dominance",
                self.transitions_pruned_by_dominance,
            ),
        ):
            _plain_int(value, field=field)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "completion_times", tuple(completions))
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def certificate_hash(self) -> str:
        """Hash the semantic certificate payload, excluding search counters."""

        digest = hashlib.sha256()
        digest.update(_CERTIFICATE_DOMAIN)
        digest.update(bytes.fromhex(self.input_hash))
        digest.update(_frame(_encode_integer(self.schema_version)))
        digest.update(_frame(_encode_integer(self.objective.deadline_violations)))
        _update_fraction(digest, self.objective.weighted_flow)
        digest.update(_frame(_encode_integer(len(self.events))))
        for event in self.events:
            digest.update(_frame(_encode_event(event)))
        digest.update(_frame(_encode_integer(len(self.completion_times))))
        for job_id, completion in self.completion_times:
            digest.update(_frame(job_id.encode("utf-8")))
            _update_fraction(digest, completion)
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CertificateVerification:
    """Successful independent verification report."""

    objective: OracleObjective
    completion_times: tuple[tuple[JobId, Fraction], ...]
    optimality_checked: bool
    verifier_nodes: int


@dataclass(frozen=True, slots=True)
class ScheduleEvaluation:
    """Feasible non-optimal schedule used to quantify heuristic gaps."""

    input_hash: str
    events: tuple[ScheduleEvent, ...]
    objective: OracleObjective
    completion_times: tuple[tuple[JobId, Fraction], ...]


@dataclass(frozen=True, slots=True)
class OracleGap:
    """Component-wise candidate minus exact-oracle objective."""

    deadline_violation_gap: int
    weighted_flow_gap: Fraction


@dataclass(frozen=True, slots=True)
class _Solution:
    objective: OracleObjective
    events: tuple[ScheduleEvent, ...]


@dataclass(slots=True)
class _SearchStats:
    states_explored: int = 0
    memo_hits: int = 0
    transitions_considered: int = 0
    transitions_explored: int = 0
    dominated_transitions: int = 0


@dataclass(frozen=True, slots=True)
class _Transition:
    event: DispatchEvent
    next_time: Fraction
    next_mask: int
    incremental_objective: OracleObjective


def _ready_indices(
    problem: OracleProblem,
    time: Fraction,
    remaining_mask: int,
) -> tuple[int, ...]:
    return tuple(
        index
        for index, job in enumerate(problem.jobs)
        if remaining_mask & (1 << index) and job.release_time <= time
    )


def _next_release(
    problem: OracleProblem,
    time: Fraction,
    remaining_mask: int,
) -> Fraction:
    future = [
        job.release_time
        for index, job in enumerate(problem.jobs)
        if remaining_mask & (1 << index) and job.release_time > time
    ]
    if not future:
        raise AssertionError("unfinished state has neither ready nor future jobs")
    return min(future)


def _incremental_objective(
    problem: OracleProblem,
    selected: tuple[int, ...],
    completion_time: Fraction,
) -> OracleObjective:
    violations = sum(completion_time > problem.jobs[index].deadline for index in selected)
    flow = sum(
        (
            problem.jobs[index].weight * (completion_time - problem.jobs[index].release_time)
            for index in selected
        ),
        start=Fraction(),
    )
    return OracleObjective(violations, flow)


def _solution_key(
    solution: _Solution,
) -> tuple[int, Fraction, tuple[bytes, ...]]:
    return (
        solution.objective.deadline_violations,
        solution.objective.weighted_flow,
        _trace_key(solution.events),
    )


def _record_transition(
    stats: _SearchStats,
    limits: OracleSearchLimits,
) -> None:
    stats.transitions_considered += 1
    if stats.transitions_considered > limits.max_transitions:
        raise GeneralOracleLimitExceeded(
            GeneralOracleLimitKind.TRANSITIONS,
            limits.max_transitions,
            stats.transitions_considered,
        )


def _dispatch_transitions(
    problem: OracleProblem,
    time: Fraction,
    remaining_mask: int,
    ready: tuple[int, ...],
    stats: _SearchStats,
    limits: OracleSearchLimits,
) -> tuple[_Transition, ...]:
    """Enumerate all orders, pruning only transition-identical permutations."""

    retained: dict[tuple[int, Fraction, OracleObjective], _Transition] = {}
    maximum = min(len(ready), problem.capacity.row_limit)
    for size in range(1, maximum + 1):
        for selected in combinations(ready, size):
            slots = sum(problem.jobs[index].width for index in selected)
            if slots > problem.capacity.slot_limit:
                continue
            duration = problem.latency.duration(size, slots)
            end_time = time + duration
            selected_mask = sum(1 << index for index in selected)
            next_mask = remaining_mask & ~selected_mask
            incremental = _incremental_objective(problem, selected, end_time)
            transition_key = (next_mask, end_time, incremental)
            for ordered in permutations(selected):
                _record_transition(stats, limits)
                event = DispatchEvent(
                    start_time=time,
                    end_time=end_time,
                    job_ids=tuple(problem.jobs[index].job_id for index in ordered),
                    rows=size,
                    slots=slots,
                )
                candidate = _Transition(
                    event=event,
                    next_time=end_time,
                    next_mask=next_mask,
                    incremental_objective=incremental,
                )
                previous = retained.get(transition_key)
                if previous is None or _encode_event(event) < _encode_event(previous.event):
                    if previous is not None:
                        stats.dominated_transitions += 1
                    retained[transition_key] = candidate
                else:
                    stats.dominated_transitions += 1
    return tuple(sorted(retained.values(), key=lambda transition: _encode_event(transition.event)))


def _optional_waits(
    problem: OracleProblem,
    time: Fraction,
) -> tuple[WaitEvent, ...]:
    return tuple(
        WaitEvent(time, target, kind) for target, kind in problem.decision_points if target > time
    )


def _completion_times_from_events(
    problem: OracleProblem,
    events: Sequence[ScheduleEvent],
) -> tuple[tuple[JobId, Fraction], ...]:
    completions: dict[str, Fraction] = {}
    for event in events:
        if isinstance(event, DispatchEvent):
            for job_id in event.job_ids:
                completions[job_id] = event.end_time
    return tuple((job.job_id, completions[job.job_id]) for job in problem.jobs)


def score_completion_times(
    problem: OracleProblem,
    completion_times: Mapping[JobId, ExactInput],
) -> OracleObjective:
    """Score a complete external schedule under the oracle objective."""

    if not isinstance(problem, OracleProblem):
        raise GeneralOracleError("problem must be an OracleProblem")
    if not isinstance(completion_times, Mapping):
        raise GeneralOracleError("completion_times must be a mapping")
    expected = {job.job_id for job in problem.jobs}
    if set(completion_times) != expected:
        raise GeneralOracleError("completion_times must name every job exactly once")
    violations = 0
    weighted_flow = Fraction()
    for job in problem.jobs:
        completion = _exact(
            completion_times[job.job_id],
            field=f"completion_times[{job.job_id!r}]",
        )
        if completion < job.release_time:
            raise GeneralOracleError(f"job {job.job_id!r} completes before its release")
        violations += completion > job.deadline
        weighted_flow += job.weight * (completion - job.release_time)
    return OracleObjective(violations, weighted_flow)


def objective_gap(
    candidate: OracleObjective,
    optimum: OracleObjective,
) -> OracleGap:
    """Return component-wise objective differences for a heuristic schedule."""

    if not isinstance(candidate, OracleObjective) or not isinstance(
        optimum,
        OracleObjective,
    ):
        raise GeneralOracleError("candidate and optimum must be OracleObjective values")
    return OracleGap(
        candidate.deadline_violations - optimum.deadline_violations,
        candidate.weighted_flow - optimum.weighted_flow,
    )


def solve_general_oracle(
    problem: OracleProblem,
    *,
    limits: OracleSearchLimits,
) -> GeneralOracleCertificate:
    """Exhaustively solve the bounded problem with exact-state memoization."""

    if not isinstance(problem, OracleProblem):
        raise GeneralOracleError("problem must be an OracleProblem")
    if not isinstance(limits, OracleSearchLimits):
        raise GeneralOracleError("limits must be an OracleSearchLimits")
    job_count = len(problem.jobs)
    if job_count > limits.max_jobs:
        raise GeneralOracleLimitExceeded(
            GeneralOracleLimitKind.JOBS,
            limits.max_jobs,
            job_count,
        )
    conservative_trace_bound = 2 * job_count + len(problem.decision_points)
    if conservative_trace_bound > limits.max_trace_events:
        raise GeneralOracleLimitExceeded(
            GeneralOracleLimitKind.TRACE_EVENTS,
            limits.max_trace_events,
            conservative_trace_bound,
        )

    memo: dict[tuple[Fraction, int], _Solution] = {}
    stats = _SearchStats()

    def search(time: Fraction, remaining_mask: int) -> _Solution:
        state_key = (time, remaining_mask)
        cached = memo.get(state_key)
        if cached is not None:
            stats.memo_hits += 1
            return cached
        stats.states_explored += 1
        if stats.states_explored > limits.max_states:
            raise GeneralOracleLimitExceeded(
                GeneralOracleLimitKind.STATES,
                limits.max_states,
                stats.states_explored,
            )
        if remaining_mask == 0:
            solution = _Solution(OracleObjective(0, Fraction()), ())
            memo[state_key] = solution
            return solution

        ready = _ready_indices(problem, time, remaining_mask)
        if not ready:
            wake = _next_release(problem, time, remaining_mask)
            _record_transition(stats, limits)
            stats.transitions_explored += 1
            event = WaitEvent(time, wake, WaitKind.FORCED_RELEASE)
            tail = search(wake, remaining_mask)
            solution = _Solution(tail.objective, (event, *tail.events))
            memo[state_key] = solution
            return solution

        candidates: list[_Solution] = []
        for transition in _dispatch_transitions(
            problem,
            time,
            remaining_mask,
            ready,
            stats,
            limits,
        ):
            stats.transitions_explored += 1
            tail = search(transition.next_time, transition.next_mask)
            candidates.append(
                _Solution(
                    transition.incremental_objective + tail.objective,
                    (transition.event, *tail.events),
                )
            )
        for event in _optional_waits(problem, time):
            _record_transition(stats, limits)
            stats.transitions_explored += 1
            tail = search(event.end_time, remaining_mask)
            candidates.append(_Solution(tail.objective, (event, *tail.events)))

        if not candidates:  # pragma: no cover - every individual job fits
            raise AssertionError("ready state has no admissible transition")
        solution = min(candidates, key=_solution_key)
        memo[state_key] = solution
        return solution

    root_mask = (1 << job_count) - 1
    best = search(problem.start_time, root_mask)
    completion_times = _completion_times_from_events(problem, best.events)
    rescored = score_completion_times(problem, dict(completion_times))
    if rescored != best.objective:  # pragma: no cover - internal invariant
        raise AssertionError("incremental objective disagrees with trace replay")
    return GeneralOracleCertificate(
        input_hash=problem.input_hash,
        events=best.events,
        objective=best.objective,
        completion_times=completion_times,
        states_explored=stats.states_explored,
        states_pruned_by_memo=stats.memo_hits,
        transitions_explored=stats.transitions_explored,
        transitions_pruned_by_dominance=stats.dominated_transitions,
    )


def _replay_events(
    problem: OracleProblem,
    events: Sequence[ScheduleEvent],
) -> tuple[OracleObjective, tuple[tuple[JobId, Fraction], ...]]:
    jobs_by_id = {job.job_id: job for job in problem.jobs}
    remaining = set(jobs_by_id)
    completions: dict[str, Fraction] = {}
    time = problem.start_time
    decision_map = dict(problem.decision_points)

    for position, event in enumerate(events):
        if not remaining:
            raise CertificateVerificationError(f"event {position} occurs after every job completed")
        ready = {job_id for job_id in remaining if jobs_by_id[job_id].release_time <= time}
        if isinstance(event, WaitEvent):
            if event.start_time != time:
                raise CertificateVerificationError(
                    f"wait {position} starts at {event.start_time}, expected {time}"
                )
            if ready:
                expected_kind = decision_map.get(event.end_time)
                if expected_kind is None or expected_kind is not event.kind:
                    raise CertificateVerificationError(
                        f"wait {position} is not a configured optional decision"
                    )
                if event.kind is WaitKind.FORCED_RELEASE:
                    raise CertificateVerificationError(
                        f"wait {position} is marked forced while jobs are ready"
                    )
            else:
                future = min(jobs_by_id[job_id].release_time for job_id in remaining)
                if event.kind is not WaitKind.FORCED_RELEASE or event.end_time != future:
                    raise CertificateVerificationError(
                        f"wait {position} must advance to next release {future}"
                    )
            time = event.end_time
            continue

        if event.start_time != time:
            raise CertificateVerificationError(
                f"dispatch {position} starts at {event.start_time}, expected {time}"
            )
        selected = event.job_ids
        if any(job_id not in jobs_by_id for job_id in selected):
            raise CertificateVerificationError(f"dispatch {position} names an unknown job")
        if any(job_id not in ready for job_id in selected):
            raise CertificateVerificationError(
                f"dispatch {position} names an unreleased or completed job"
            )
        if len(selected) > problem.capacity.row_limit:
            raise CertificateVerificationError(f"dispatch {position} exceeds row capacity")
        slots = sum(jobs_by_id[job_id].width for job_id in selected)
        if slots > problem.capacity.slot_limit:
            raise CertificateVerificationError(f"dispatch {position} exceeds slot capacity")
        if event.rows != len(selected) or event.slots != slots:
            raise CertificateVerificationError(
                f"dispatch {position} row/slot metadata is inconsistent"
            )
        expected_end = time + problem.latency.duration(len(selected), slots)
        if event.end_time != expected_end:
            raise CertificateVerificationError(
                f"dispatch {position} ends at {event.end_time}, expected {expected_end}"
            )
        for job_id in selected:
            remaining.remove(job_id)
            completions[job_id] = expected_end
        time = expected_end

    if remaining:
        raise CertificateVerificationError(
            f"certificate leaves jobs unfinished: {sorted(remaining)}"
        )
    canonical = tuple((job.job_id, completions[job.job_id]) for job in problem.jobs)
    objective = score_completion_times(problem, dict(canonical))
    return objective, canonical


def _independent_brute_force(
    problem: OracleProblem,
    *,
    max_nodes: int,
) -> tuple[_Solution, int]:
    """Enumerate the full tree without optimizer memoization or dominance."""

    node_limit = _plain_int(max_nodes, field="max_verifier_nodes", minimum=1)
    nodes = 0

    def enumerate_state(time: Fraction, remaining: frozenset[int]) -> _Solution:
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            raise GeneralOracleLimitExceeded(
                GeneralOracleLimitKind.VERIFIER_NODES,
                node_limit,
                nodes,
            )
        if not remaining:
            return _Solution(OracleObjective(0, Fraction()), ())

        ready = tuple(
            index for index in sorted(remaining) if problem.jobs[index].release_time <= time
        )
        if not ready:
            wake = min(problem.jobs[index].release_time for index in remaining)
            forced_event = WaitEvent(time, wake, WaitKind.FORCED_RELEASE)
            tail = enumerate_state(wake, remaining)
            return _Solution(tail.objective, (forced_event, *tail.events))

        candidates: list[_Solution] = []
        maximum = min(len(ready), problem.capacity.row_limit)
        for size in range(1, maximum + 1):
            for selected in combinations(ready, size):
                slots = sum(problem.jobs[index].width for index in selected)
                if slots > problem.capacity.slot_limit:
                    continue
                end = time + problem.latency.duration(size, slots)
                violations = sum(end > problem.jobs[index].deadline for index in selected)
                flow = sum(
                    (
                        problem.jobs[index].weight * (end - problem.jobs[index].release_time)
                        for index in selected
                    ),
                    start=Fraction(),
                )
                incremental = OracleObjective(violations, flow)
                next_remaining = remaining.difference(selected)
                for ordered in permutations(selected):
                    dispatch_event = DispatchEvent(
                        time,
                        end,
                        tuple(problem.jobs[index].job_id for index in ordered),
                        size,
                        slots,
                    )
                    tail = enumerate_state(end, next_remaining)
                    candidates.append(
                        _Solution(
                            incremental + tail.objective,
                            (dispatch_event, *tail.events),
                        )
                    )
        for target, kind in problem.decision_points:
            if target <= time:
                continue
            wait_event = WaitEvent(time, target, kind)
            tail = enumerate_state(target, remaining)
            candidates.append(_Solution(tail.objective, (wait_event, *tail.events)))
        return min(candidates, key=_solution_key)

    root = frozenset(range(len(problem.jobs)))
    return enumerate_state(problem.start_time, root), nodes


def verify_general_oracle_certificate(
    problem: OracleProblem,
    certificate: GeneralOracleCertificate,
    *,
    prove_optimality: bool = True,
    max_verifier_nodes: int = 1_000_000,
) -> CertificateVerification:
    """Replay a certificate and optionally prove its optimum independently."""

    if not isinstance(problem, OracleProblem):
        raise GeneralOracleError("problem must be an OracleProblem")
    if not isinstance(certificate, GeneralOracleCertificate):
        raise GeneralOracleError("certificate must be a GeneralOracleCertificate")
    if certificate.input_hash != problem.input_hash:
        raise CertificateVerificationError("certificate input_hash does not match problem")
    if certificate.states_explored < 1:
        raise CertificateVerificationError("certificate must report at least one explored state")

    objective, completions = _replay_events(problem, certificate.events)
    if objective != certificate.objective:
        raise CertificateVerificationError("certificate objective does not match replay")
    if completions != certificate.completion_times:
        raise CertificateVerificationError("certificate completion_times do not match replay")

    verifier_nodes = 0
    if prove_optimality:
        independent, verifier_nodes = _independent_brute_force(
            problem,
            max_nodes=max_verifier_nodes,
        )
        if independent.objective != certificate.objective:
            raise CertificateVerificationError("certificate objective is not independently optimal")
        if independent.events != certificate.events:
            raise CertificateVerificationError(
                "certificate violates the deterministic optimal-trace tie-break"
            )
    return CertificateVerification(
        objective=objective,
        completion_times=completions,
        optimality_checked=prove_optimality,
        verifier_nodes=verifier_nodes,
    )


def work_conserving_edf(problem: OracleProblem) -> ScheduleEvaluation:
    """Evaluate a deterministic no-wait, earliest-deadline-first heuristic."""

    if not isinstance(problem, OracleProblem):
        raise GeneralOracleError("problem must be an OracleProblem")
    remaining = set(range(len(problem.jobs)))
    time = problem.start_time
    events: list[ScheduleEvent] = []
    while remaining:
        ready = sorted(
            (index for index in remaining if problem.jobs[index].release_time <= time),
            key=lambda index: (
                problem.jobs[index].deadline,
                problem.jobs[index].job_id,
            ),
        )
        if not ready:
            wake = min(problem.jobs[index].release_time for index in remaining)
            events.append(WaitEvent(time, wake, WaitKind.FORCED_RELEASE))
            time = wake
            continue
        selected: list[int] = []
        slots = 0
        for index in ready:
            width = problem.jobs[index].width
            if (
                len(selected) < problem.capacity.row_limit
                and slots + width <= problem.capacity.slot_limit
            ):
                selected.append(index)
                slots += width
        duration = problem.latency.duration(len(selected), slots)
        end = time + duration
        events.append(
            DispatchEvent(
                time,
                end,
                tuple(problem.jobs[index].job_id for index in selected),
                len(selected),
                slots,
            )
        )
        remaining.difference_update(selected)
        time = end

    trace = tuple(events)
    objective, completion_times = _replay_events(problem, trace)
    return ScheduleEvaluation(
        input_hash=problem.input_hash,
        events=trace,
        objective=objective,
        completion_times=completion_times,
    )
