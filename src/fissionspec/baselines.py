"""Executable scheduler-level baselines over pre-realized semantic outcomes.

The main simulator's :class:`~fissionspec.policies.SchedulingPolicy` interface
intentionally decides only *when* to dispatch.  It cannot faithfully express
batch membership, a shared remote-draft queue, or EXSpec-style sequence pools.
This module supplies a separate deterministic harness for those questions.

Every request carries an immutable sequence of target-authorized outcomes.
Schedulers may alter readiness, batching, padding, and queue order, but they
never see the current round's acceptance or rollback before verification
completes.  Consequently every completed run must have the same semantic
signature.  :func:`assert_semantic_equivalence` makes that obligation explicit.

These are scheduler abstractions, not claims of bit-for-bit paper
reimplementations.  The modeled and omitted mechanisms are documented in
``docs/baselines.md``.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, TypeAlias, runtime_checkable

from fissionspec.model import Outcome, SimulationResult
from fissionspec.profiles import HardwareProfile

CandidateKey: TypeAlias = tuple[str, int, bool]
SemanticSignature: TypeAlias = tuple[
    tuple[str, tuple[int, ...], tuple[tuple[int, int, bool, bool], ...]], ...
]


class BaselineError(RuntimeError):
    """Raised when a trace or scheduler cannot make correct forward progress."""


def _finite_non_negative(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{field} must be finite and non-negative")
    return float(value)


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class RealizedStep:
    """One schedule-independent target-verification outcome.

    ``emitted_tokens`` are opaque target-authorized token identities.  They can
    be real greedy token IDs or deterministic ordinal markers when bridging a
    count-level :class:`SimulationResult`.

    ``needs_remote_draft`` controls readiness of the next step. ``rollback`` is
    separately consumed by the SPECTRE threshold; the two usually coincide in
    the count-level bridge but are not conflated by the representation.
    """

    speculation_length: int
    accepted_length: int
    emitted_tokens: tuple[int, ...]
    needs_remote_draft: bool
    rollback: bool

    def __post_init__(self) -> None:
        _positive_int(self.speculation_length, field="speculation_length")
        if (
            isinstance(self.accepted_length, bool)
            or not isinstance(self.accepted_length, int)
            or not 0 <= self.accepted_length <= self.speculation_length
        ):
            raise ValueError("accepted_length must be in [0, speculation_length]")
        if not self.emitted_tokens or len(self.emitted_tokens) > self.speculation_length:
            raise ValueError(
                "emitted_tokens must contain between one and speculation_length tokens"
            )
        if self.accepted_length > len(self.emitted_tokens):
            raise ValueError("accepted_length cannot exceed emitted token count")
        if any(
            isinstance(token, bool) or not isinstance(token, int) for token in self.emitted_tokens
        ):
            raise ValueError("emitted token identities must be integers")
        if not isinstance(self.needs_remote_draft, bool):
            raise TypeError("needs_remote_draft must be a bool")
        if not isinstance(self.rollback, bool):
            raise TypeError("rollback must be a bool")


@dataclass(frozen=True, slots=True)
class RealizedRequest:
    """One request whose complete semantic trace is fixed before scheduling."""

    request_id: str
    arrival_ms: float
    deadline_ms: float
    prompt_tokens: int
    steps: tuple[RealizedStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        arrival = _finite_non_negative(self.arrival_ms, field="arrival_ms")
        deadline = _finite_non_negative(self.deadline_ms, field="deadline_ms")
        if deadline < arrival:
            raise ValueError("deadline_ms must not precede arrival_ms")
        if (
            isinstance(self.prompt_tokens, bool)
            or not isinstance(self.prompt_tokens, int)
            or self.prompt_tokens < 0
        ):
            raise ValueError("prompt_tokens must be a non-negative integer")
        if not self.steps:
            raise ValueError("a realized request needs at least one step")
        for index, step in enumerate(self.steps):
            if not isinstance(step, RealizedStep):
                raise TypeError("steps must contain RealizedStep values")
            if index == len(self.steps) - 1 and step.needs_remote_draft:
                raise ValueError("a terminal step cannot request another draft")


@dataclass(frozen=True, slots=True)
class BackgroundDraftJob:
    """Normal draft-service traffic competing with speculative refresh jobs."""

    job_id: str
    release_ms: float
    duration_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("job_id must be a non-empty string")
        _finite_non_negative(self.release_ms, field="release_ms")
        duration = _finite_non_negative(self.duration_ms, field="duration_ms")
        if duration == 0.0:
            raise ValueError("duration_ms must be positive")


@dataclass(frozen=True, slots=True)
class PreRealizedTrace:
    """Immutable semantic input shared across all scheduler baselines."""

    requests: tuple[RealizedRequest, ...]
    background_draft_jobs: tuple[BackgroundDraftJob, ...] = ()
    name: str = "pre-realized"

    def __post_init__(self) -> None:
        if not self.requests:
            raise ValueError("a trace needs at least one request")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("trace name must be a non-empty string")
        request_ids = [request.request_id for request in self.requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request IDs must be unique")
        background_ids = [job.job_id for job in self.background_draft_jobs]
        if len(background_ids) != len(set(background_ids)):
            raise ValueError("background draft job IDs must be unique")

    @classmethod
    def from_simulation(
        cls,
        result: SimulationResult,
        *,
        background_draft_jobs: Iterable[BackgroundDraftJob] = (),
        name: str | None = None,
    ) -> PreRealizedTrace:
        """Bridge one existing run into a schedule-independent outcome trace.

        The count-level simulator does not expose token IDs.  The bridge assigns
        each emitted position its request-local ordinal; accepted lengths,
        productive counts, hit/miss outcomes, widths, arrivals, prompts, and
        deadlines are retained exactly.
        """

        if not isinstance(result, SimulationResult):
            raise TypeError("result must be a SimulationResult")
        configs = {request.request_id: request for request in result.workload}
        steps: dict[str, list[RealizedStep]] = defaultdict(list)
        emitted_so_far = {request_id: 0 for request_id in configs}
        for launch in result.target_launches:
            outcomes = dict(launch.outcomes)
            accepted = dict(launch.accepted_tokens)
            productive = dict(launch.productive_tokens)
            for request_id in launch.request_ids:
                config = configs[request_id]
                emitted = productive[request_id]
                remaining = config.output_tokens - emitted_so_far[request_id]
                width = min(config.speculation_length, remaining)
                start = emitted_so_far[request_id]
                tokens = tuple(range(start, start + emitted))
                emitted_so_far[request_id] += emitted
                outcome = outcomes[request_id]
                needs_remote = outcome is Outcome.MISS
                steps[request_id].append(
                    RealizedStep(
                        speculation_length=width,
                        accepted_length=accepted[request_id],
                        emitted_tokens=tokens,
                        needs_remote_draft=needs_remote,
                        rollback=needs_remote,
                    )
                )
        requests: list[RealizedRequest] = []
        for config in result.workload:
            if emitted_so_far[config.request_id] != config.output_tokens:
                raise BaselineError(f"source trace did not complete request {config.request_id!r}")
            requests.append(
                RealizedRequest(
                    request_id=config.request_id,
                    arrival_ms=config.arrival_ms,
                    deadline_ms=config.absolute_deadline_ms,
                    prompt_tokens=config.prompt_tokens,
                    steps=tuple(steps[config.request_id]),
                )
            )
        return cls(
            requests=tuple(requests),
            background_draft_jobs=tuple(background_draft_jobs),
            name=result.workload_name if name is None else name,
        )

    @property
    def semantic_signature(self) -> SemanticSignature:
        """Return the outcome sequence no scheduler is permitted to alter."""

        return tuple(
            (
                request.request_id,
                tuple(token for step in request.steps for token in step.emitted_tokens),
                tuple(
                    (
                        step.accepted_length,
                        len(step.emitted_tokens),
                        step.needs_remote_draft,
                        step.rollback,
                    )
                    for step in request.steps
                ),
            )
            for request in self.requests
        )


@dataclass(frozen=True, slots=True)
class BaselineCostModel:
    """Calibratable deterministic target, recovery, and alignment costs."""

    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    draft_context_token_ms: float = 0.001
    draft_token_ms: float = 0.04
    realignment_base_ms: float = 0.15
    realignment_per_length_ms: float = 0.002
    starvation_threshold_ms: float = 50.0

    def __post_init__(self) -> None:
        if not isinstance(self.hardware, HardwareProfile):
            raise TypeError("hardware must be a HardwareProfile")
        for name, value in (
            ("draft_context_token_ms", self.draft_context_token_ms),
            ("draft_token_ms", self.draft_token_ms),
            ("realignment_base_ms", self.realignment_base_ms),
            ("realignment_per_length_ms", self.realignment_per_length_ms),
            ("starvation_threshold_ms", self.starvation_threshold_ms),
        ):
            _finite_non_negative(value, field=name)

    def target_latency_ms(self, rows: int, slots: int) -> float:
        return self.hardware.target_latency_ms(rows, slots)

    def recovery_latency_ms(
        self,
        *,
        context_tokens: int,
        speculation_length: int,
        compression_factor: float,
    ) -> float:
        retained_context = context_tokens * compression_factor
        return (
            self.hardware.draft_latency_ms(1, recovery=True)
            + retained_context * self.draft_context_token_ms
            + speculation_length * self.draft_token_ms
        )

    def realignment_latency_ms(
        self,
        sequence_lengths: tuple[int, ...],
        speculation_lengths: tuple[int, ...],
    ) -> float:
        if len(sequence_lengths) != len(speculation_lengths):
            raise ValueError("alignment vectors must have equal lengths")
        if len(sequence_lengths) <= 1 or (
            len(set(sequence_lengths)) == 1 and len(set(speculation_lengths)) == 1
        ):
            return 0.0
        sequence_spread = max(sequence_lengths) - min(sequence_lengths)
        speculation_spread = max(speculation_lengths) - min(speculation_lengths)
        return (
            self.realignment_base_ms * len(sequence_lengths)
            + (sequence_spread + speculation_spread) * self.realignment_per_length_ms
        )


class ExecutionMode(StrEnum):
    """SPECTRE-style coordination mode for a post-verification batch."""

    ORDINARY = "ordinary"
    PARALLEL = "parallel"


@dataclass(frozen=True, slots=True)
class ModeDecision:
    """Auditable cost comparison behind one hybrid-mode decision."""

    mode: ExecutionMode
    rollback_ratio: float
    critical_rollback_ratio: float
    ordinary_cost_ms: float
    parallel_cost_ms: float


@dataclass(frozen=True, slots=True)
class SpectreCalibration:
    """Measured batch-level costs used to derive the rollback threshold.

    The abstraction fits ``parallel_cost(r) = parallel_round_ms +
    r * rollback_penalty_ms`` and compares it with ``ordinary_round_ms``.
    """

    ordinary_round_ms: float
    parallel_round_ms: float
    rollback_penalty_ms: float

    def __post_init__(self) -> None:
        ordinary = _finite_non_negative(self.ordinary_round_ms, field="ordinary_round_ms")
        parallel = _finite_non_negative(self.parallel_round_ms, field="parallel_round_ms")
        penalty = _finite_non_negative(self.rollback_penalty_ms, field="rollback_penalty_ms")
        if ordinary == 0.0 or parallel == 0.0 or penalty == 0.0:
            raise ValueError("SPECTRE calibration costs must be positive")

    @property
    def critical_rollback_ratio(self) -> float:
        raw = (self.ordinary_round_ms - self.parallel_round_ms) / self.rollback_penalty_ms
        return min(1.0, max(0.0, raw))

    def decide(self, rollback_ratio: float) -> ModeDecision:
        ratio = _finite_non_negative(rollback_ratio, field="rollback_ratio")
        if ratio > 1.0:
            raise ValueError("rollback_ratio must not exceed one")
        parallel_cost = self.parallel_round_ms + ratio * self.rollback_penalty_ms
        mode = (
            ExecutionMode.PARALLEL
            if parallel_cost <= self.ordinary_round_ms
            else ExecutionMode.ORDINARY
        )
        return ModeDecision(
            mode=mode,
            rollback_ratio=ratio,
            critical_rollback_ratio=self.critical_rollback_ratio,
            ordinary_cost_ms=self.ordinary_round_ms,
            parallel_cost_ms=parallel_cost,
        )


@dataclass(frozen=True, slots=True)
class CandidateView:
    """Only scheduler-visible state for a ready target row."""

    key: CandidateKey
    request_id: str
    step_index: int
    ready_since_ms: float
    deadline_ms: float
    sequence_length: int
    speculation_length: int
    last_accepted_length: int
    padding: bool = False

    def __post_init__(self) -> None:
        if self.key != (self.request_id, self.step_index, self.padding):
            raise ValueError("candidate key does not match its row identity")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
        ):
            raise ValueError("step_index must be a non-negative integer")
        _finite_non_negative(self.ready_since_ms, field="ready_since_ms")
        _finite_non_negative(self.deadline_ms, field="deadline_ms")
        if (
            isinstance(self.sequence_length, bool)
            or not isinstance(self.sequence_length, int)
            or self.sequence_length < 0
        ):
            raise ValueError("sequence_length must be a non-negative integer")
        _positive_int(self.speculation_length, field="speculation_length")
        if (
            isinstance(self.last_accepted_length, bool)
            or not isinstance(self.last_accepted_length, int)
            or self.last_accepted_length < 0
        ):
            raise ValueError("last_accepted_length must be a non-negative integer")
        if not isinstance(self.padding, bool):
            raise TypeError("padding must be a bool")


@dataclass(frozen=True, slots=True)
class BatchDecision:
    """Rows selected for one launch and an optional coalescing wake."""

    keys: tuple[CandidateKey, ...]
    dispatch_at_ms: float
    requires_realignment: bool = False


@runtime_checkable
class BaselineScheduler(Protocol):
    """Structural interface consumed by :class:`DeterministicBaselineSimulator`."""

    @property
    def name(self) -> str: ...

    @property
    def prompt_compression_factor(self) -> float: ...

    @property
    def draft_priority_burst(self) -> int | None: ...

    def choose_batch(
        self,
        candidates: tuple[CandidateView, ...],
        *,
        now_ms: float,
        max_batch_size: int,
    ) -> BatchDecision: ...

    def recovery_mode(self, rollback_ratio: float) -> ModeDecision: ...


@dataclass(frozen=True, slots=True)
class SPECTREHybridScheduler:
    """Scheduler-level abstraction of SPECTRE's three coordination mechanisms.

    Mode selection is batch-level and uses :class:`SpectreCalibration`.
    Speculative remote jobs are non-preemptively prioritized, with one regular
    job forced after ``priority_burst`` consecutive speculative jobs.
    """

    calibration: SpectreCalibration
    priority_burst: int = 4
    context_compression_factor: float = 1.0
    name: str = "spectre-hybrid-abstraction"

    def __post_init__(self) -> None:
        if not isinstance(self.calibration, SpectreCalibration):
            raise TypeError("calibration must be a SpectreCalibration")
        _positive_int(self.priority_burst, field="priority_burst")
        factor = _finite_non_negative(
            self.context_compression_factor,
            field="context_compression_factor",
        )
        if not 0.0 < factor <= 1.0:
            raise ValueError("context_compression_factor must be in (0, 1]")

    @property
    def prompt_compression_factor(self) -> float:
        return self.context_compression_factor

    @property
    def draft_priority_burst(self) -> int:
        return self.priority_burst

    def choose_batch(
        self,
        candidates: tuple[CandidateView, ...],
        *,
        now_ms: float,
        max_batch_size: int,
    ) -> BatchDecision:
        del now_ms
        capacity = _positive_int(max_batch_size, field="max_batch_size")
        ordered = sorted(
            candidates,
            key=lambda row: (
                row.deadline_ms,
                row.ready_since_ms,
                row.padding,
                row.request_id,
                row.step_index,
            ),
        )
        return BatchDecision(
            tuple(row.key for row in ordered[:capacity]),
            dispatch_at_ms=ordered[0].ready_since_ms if ordered else 0.0,
        )

    def recovery_mode(self, rollback_ratio: float) -> ModeDecision:
        return self.calibration.decide(rollback_ratio)


# Conventional spelling retained for discoverability.
SpectreHybridScheduler = SPECTREHybridScheduler


@dataclass(frozen=True, slots=True)
class EXSpecSlidingPoolScheduler:
    """Sliding-window pool that prefers equal current-length/spec-width groups.

    ``sequence_length`` already incorporates all *previously verified*
    accepted output.  The scheduler never observes the current round's
    acceptance before choosing its batch.
    """

    window_size: int = 32
    minimum_group_size: int = 2
    name: str = "exspec-sliding-pool-abstraction"

    def __post_init__(self) -> None:
        _positive_int(self.window_size, field="window_size")
        _positive_int(self.minimum_group_size, field="minimum_group_size")

    @property
    def prompt_compression_factor(self) -> float:
        return 1.0

    @property
    def draft_priority_burst(self) -> None:
        return None

    def choose_batch(
        self,
        candidates: tuple[CandidateView, ...],
        *,
        now_ms: float,
        max_batch_size: int,
    ) -> BatchDecision:
        capacity = _positive_int(max_batch_size, field="max_batch_size")
        ordered = sorted(
            (row for row in candidates if not row.padding),
            key=lambda row: (row.ready_since_ms, row.request_id, row.step_index),
        )
        if not ordered:
            return BatchDecision((), now_ms)
        window = ordered[: self.window_size]
        groups: dict[tuple[int, int], list[CandidateView]] = defaultdict(list)
        for row in window:
            groups[(row.sequence_length, row.speculation_length)].append(row)
        eligible = [group for group in groups.values() if len(group) >= self.minimum_group_size]
        if eligible:
            group = min(
                eligible,
                key=lambda rows: (
                    -len(rows),
                    rows[0].ready_since_ms,
                    rows[0].request_id,
                ),
            )
            selected = group[:capacity]
            return BatchDecision(
                tuple(row.key for row in selected),
                dispatch_at_ms=now_ms,
                requires_realignment=False,
            )
        selected = window[:capacity]
        keys = {(row.sequence_length, row.speculation_length) for row in selected}
        return BatchDecision(
            tuple(row.key for row in selected),
            dispatch_at_ms=now_ms,
            requires_realignment=len(keys) > 1,
        )

    def recovery_mode(self, rollback_ratio: float) -> ModeDecision:
        ratio = _finite_non_negative(rollback_ratio, field="rollback_ratio")
        return ModeDecision(ExecutionMode.ORDINARY, ratio, 0.0, 0.0, math.inf)


@dataclass(frozen=True, slots=True)
class MyopicSlackScheduler:
    """Per-request slack ranking with explicit aging and starvation escape."""

    estimated_base_ms: float = 1.0
    estimated_slot_ms: float = 0.05
    aging_rate: float = 1.0
    starvation_bound_ms: float = 50.0
    max_coalesce_ms: float = 0.0
    name: str = "myopic-slack-aging-abstraction"

    def __post_init__(self) -> None:
        for field_name, value in (
            ("estimated_base_ms", self.estimated_base_ms),
            ("estimated_slot_ms", self.estimated_slot_ms),
            ("aging_rate", self.aging_rate),
            ("starvation_bound_ms", self.starvation_bound_ms),
            ("max_coalesce_ms", self.max_coalesce_ms),
        ):
            _finite_non_negative(value, field=field_name)

    @property
    def prompt_compression_factor(self) -> float:
        return 1.0

    @property
    def draft_priority_burst(self) -> None:
        return None

    def choose_batch(
        self,
        candidates: tuple[CandidateView, ...],
        *,
        now_ms: float,
        max_batch_size: int,
    ) -> BatchDecision:
        capacity = _positive_int(max_batch_size, field="max_batch_size")
        rows = [row for row in candidates if not row.padding]
        if not rows:
            return BatchDecision((), now_ms)

        def score(row: CandidateView) -> tuple[bool, float, float, str]:
            wait = max(0.0, now_ms - row.ready_since_ms)
            estimate = self.estimated_base_ms + row.speculation_length * self.estimated_slot_ms
            slack = row.deadline_ms - now_ms - estimate
            effective_slack = slack - self.aging_rate * wait
            forced = wait >= self.starvation_bound_ms
            return (not forced, effective_slack, row.ready_since_ms, row.request_id)

        ordered = sorted(rows, key=score)
        selected = ordered[:capacity]
        dispatch_at = now_ms
        if len(rows) < capacity and self.max_coalesce_ms > 0.0:
            oldest_due = min(row.ready_since_ms for row in rows) + self.max_coalesce_ms
            latest_safe = min(
                row.deadline_ms
                - self.estimated_base_ms
                - row.speculation_length * self.estimated_slot_ms
                for row in selected
            )
            dispatch_at = max(now_ms, min(oldest_due, latest_safe))
        return BatchDecision(
            tuple(row.key for row in selected),
            dispatch_at_ms=dispatch_at,
        )

    def recovery_mode(self, rollback_ratio: float) -> ModeDecision:
        ratio = _finite_non_negative(rollback_ratio, field="rollback_ratio")
        return ModeDecision(ExecutionMode.ORDINARY, ratio, 0.0, 0.0, math.inf)


@dataclass(frozen=True, slots=True)
class FIFOScheduler:
    """Work-conserving ordinary-mode reference for adversarial comparisons."""

    name: str = "fifo-ordinary-reference"

    @property
    def prompt_compression_factor(self) -> float:
        return 1.0

    @property
    def draft_priority_burst(self) -> None:
        return None

    def choose_batch(
        self,
        candidates: tuple[CandidateView, ...],
        *,
        now_ms: float,
        max_batch_size: int,
    ) -> BatchDecision:
        capacity = _positive_int(max_batch_size, field="max_batch_size")
        ordered = sorted(
            (row for row in candidates if not row.padding),
            key=lambda row: (row.ready_since_ms, row.request_id, row.step_index),
        )
        return BatchDecision(tuple(row.key for row in ordered[:capacity]), now_ms)

    def recovery_mode(self, rollback_ratio: float) -> ModeDecision:
        ratio = _finite_non_negative(rollback_ratio, field="rollback_ratio")
        return ModeDecision(ExecutionMode.ORDINARY, ratio, 0.0, 0.0, math.inf)


@dataclass(frozen=True, slots=True)
class BaselineRequestResult:
    request_id: str
    emitted_tokens: tuple[int, ...]
    accepted_lengths: tuple[int, ...]
    completion_ms: float
    deadline_ms: float
    total_ready_wait_ms: float
    max_ready_wait_ms: float

    @property
    def deadline_missed(self) -> bool:
        return self.completion_ms > self.deadline_ms


@dataclass(frozen=True, slots=True)
class BaselineTargetLaunch:
    launch_id: int
    start_ms: float
    end_ms: float
    request_ids: tuple[str, ...]
    padded_request_ids: tuple[str, ...]
    verifier_slots: int
    padded_slots: int
    realignment_ms: float
    homogeneous_lengths: bool
    rollback_ratio: float
    next_mode: ExecutionMode | None

    @property
    def real_batch_size(self) -> int:
        return len(self.request_ids)

    @property
    def effective_batch_size(self) -> int:
        return len(self.request_ids) + len(self.padded_request_ids)


class DraftJobKind(StrEnum):
    SPECULATIVE = "speculative"
    BACKGROUND = "background"


@dataclass(frozen=True, slots=True)
class BaselineDraftLaunch:
    launch_id: int
    job_id: str
    kind: DraftJobKind
    request_id: str | None
    release_ms: float
    start_ms: float
    end_ms: float
    mode: ExecutionMode | None

    @property
    def wait_ms(self) -> float:
        return self.start_ms - self.release_ms


@dataclass(frozen=True, slots=True)
class BaselineMetrics:
    policy_name: str
    makespan_ms: float
    target_launches: int
    mean_real_batch: float
    mean_effective_batch: float
    verifier_slots: int
    padded_slots: int
    homogeneous_batches: int
    realignment_fallbacks: int
    realignment_ms: float
    total_ready_wait_ms: float
    max_ready_wait_ms: float
    starved_requests: int
    deadline_misses: int
    ordinary_mode_batches: int
    parallel_mode_batches: int
    speculative_draft_jobs: int
    background_draft_jobs: int
    max_background_draft_wait_ms: float


@dataclass(frozen=True, slots=True)
class BaselineResult:
    policy_name: str
    trace_name: str
    requests: tuple[BaselineRequestResult, ...]
    target_launches: tuple[BaselineTargetLaunch, ...]
    draft_launches: tuple[BaselineDraftLaunch, ...]
    mode_decisions: tuple[ModeDecision, ...]
    semantic_signature: SemanticSignature
    metrics: BaselineMetrics


class _Phase(StrEnum):
    NOT_ARRIVED = "not_arrived"
    READY = "ready"
    IN_TARGET = "in_target"
    WAIT_DRAFT = "wait_draft"
    COMPLETE = "complete"


@dataclass(slots=True)
class _RuntimeRequest:
    config: RealizedRequest
    phase: _Phase = _Phase.NOT_ARRIVED
    step_index: int = 0
    sequence_length: int = 0
    last_accepted_length: int = 0
    ready_since_ms: float | None = None
    emitted_tokens: list[int] = field(default_factory=list)
    accepted_lengths: list[int] = field(default_factory=list)
    completion_ms: float | None = None
    total_ready_wait_ms: float = 0.0
    max_ready_wait_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class _DraftJob:
    sequence: int
    job_id: str
    kind: DraftJobKind
    release_ms: float
    duration_ms: float
    request_id: str | None
    mode: ExecutionMode | None


class _EventKind(StrEnum):
    ARRIVAL = "arrival"
    BACKGROUND_RELEASE = "background_release"
    TARGET_COMPLETE = "target_complete"
    DRAFT_COMPLETE = "draft_complete"
    WAKE = "wake"


@dataclass(order=True, slots=True)
class _Event:
    time_ms: float
    sequence: int
    kind: _EventKind = field(compare=False)
    payload: object = field(compare=False)


@dataclass(frozen=True, slots=True)
class _TargetPayload:
    launch_id: int
    start_ms: float
    end_ms: float
    selected: tuple[CandidateView, ...]
    verifier_slots: int
    padded_slots: int
    realignment_ms: float


class DeterministicBaselineSimulator:
    """Two-timeline deterministic simulator for scheduler-level abstractions."""

    _EPSILON = 1e-12

    def __init__(
        self,
        *,
        trace: PreRealizedTrace,
        scheduler: BaselineScheduler,
        costs: BaselineCostModel | None = None,
        max_batch_size: int = 16,
        max_events: int = 1_000_000,
    ) -> None:
        if not isinstance(trace, PreRealizedTrace):
            raise TypeError("trace must be a PreRealizedTrace")
        if not isinstance(scheduler, BaselineScheduler):
            raise TypeError("scheduler does not satisfy BaselineScheduler")
        self.trace = trace
        self.scheduler = scheduler
        self.costs = BaselineCostModel() if costs is None else costs
        if not isinstance(self.costs, BaselineCostModel):
            raise TypeError("costs must be a BaselineCostModel")
        self.max_batch_size = _positive_int(max_batch_size, field="max_batch_size")
        self.max_events = _positive_int(max_events, field="max_events")
        if not isinstance(scheduler.name, str) or not scheduler.name:
            raise ValueError("scheduler.name must be a non-empty string")
        compression = _finite_non_negative(
            scheduler.prompt_compression_factor,
            field="scheduler.prompt_compression_factor",
        )
        if not 0.0 < compression <= 1.0:
            raise ValueError("scheduler prompt compression must be in (0, 1]")
        priority_burst = scheduler.draft_priority_burst
        if priority_burst is not None:
            _positive_int(priority_burst, field="scheduler.draft_priority_burst")
        self._states = {
            request.request_id: _RuntimeRequest(
                config=request,
                sequence_length=request.prompt_tokens,
            )
            for request in trace.requests
        }
        self._events: list[_Event] = []
        self._event_sequence = 0
        self._job_sequence = 0
        self._now_ms = 0.0
        self._target_busy = False
        self._draft_busy = False
        self._draft_pending: list[_DraftJob] = []
        self._padding_ready: dict[str, CandidateView] = {}
        self._scheduled_wakes: set[float] = set()
        self._speculative_streak = 0
        self._target_launches: list[BaselineTargetLaunch] = []
        self._draft_launches: list[BaselineDraftLaunch] = []
        self._mode_decisions: list[ModeDecision] = []
        self._target_launch_id = 0
        self._draft_launch_id = 0
        self._has_run = False
        for request in trace.requests:
            self._push(request.arrival_ms, _EventKind.ARRIVAL, request.request_id)
        for background in trace.background_draft_jobs:
            self._job_sequence += 1
            job = _DraftJob(
                sequence=self._job_sequence,
                job_id=background.job_id,
                kind=DraftJobKind.BACKGROUND,
                release_ms=background.release_ms,
                duration_ms=background.duration_ms,
                request_id=None,
                mode=None,
            )
            self._push(
                background.release_ms,
                _EventKind.BACKGROUND_RELEASE,
                job,
            )

    def _push(self, time_ms: float, kind: _EventKind, payload: object) -> None:
        if not math.isfinite(time_ms) or time_ms < self._now_ms - self._EPSILON:
            raise BaselineError(f"cannot schedule {kind.value} in the past")
        self._event_sequence += 1
        heapq.heappush(
            self._events,
            _Event(time_ms, self._event_sequence, kind, payload),
        )

    def _candidate(self, state: _RuntimeRequest) -> CandidateView:
        ready_since = state.ready_since_ms
        if state.phase is not _Phase.READY or ready_since is None:
            raise BaselineError("only a ready request can become a candidate")
        step = state.config.steps[state.step_index]
        return CandidateView(
            key=(state.config.request_id, state.step_index, False),
            request_id=state.config.request_id,
            step_index=state.step_index,
            ready_since_ms=ready_since,
            deadline_ms=state.config.deadline_ms,
            sequence_length=state.sequence_length,
            speculation_length=step.speculation_length,
            last_accepted_length=state.last_accepted_length,
        )

    def _candidates(self) -> tuple[CandidateView, ...]:
        real = [
            self._candidate(state) for state in self._states.values() if state.phase is _Phase.READY
        ]
        return tuple(real + list(self._padding_ready.values()))

    def _schedule_target(self) -> None:
        if self._target_busy:
            return
        candidates = self._candidates()
        if not candidates:
            return
        decision = self.scheduler.choose_batch(
            candidates,
            now_ms=self._now_ms,
            max_batch_size=self.max_batch_size,
        )
        if not decision.keys:
            raise BaselineError("scheduler returned an empty batch with ready work")
        if not math.isfinite(decision.dispatch_at_ms):
            raise BaselineError("scheduler returned a non-finite dispatch time")
        if decision.dispatch_at_ms > self._now_ms + self._EPSILON:
            if decision.dispatch_at_ms not in self._scheduled_wakes:
                self._scheduled_wakes.add(decision.dispatch_at_ms)
                self._push(decision.dispatch_at_ms, _EventKind.WAKE, decision.dispatch_at_ms)
            return
        by_key = {candidate.key: candidate for candidate in candidates}
        if len(decision.keys) != len(set(decision.keys)):
            raise BaselineError("scheduler selected a candidate twice")
        try:
            selected = tuple(by_key[key] for key in decision.keys)
        except KeyError as exc:
            raise BaselineError("scheduler selected a non-ready candidate") from exc
        if len(selected) > self.max_batch_size:
            raise BaselineError("scheduler exceeded max_batch_size")

        real_rows = tuple(row for row in selected if not row.padding)
        padding_rows = tuple(row for row in selected if row.padding)
        for row in real_rows:
            state = self._states[row.request_id]
            if state.phase is not _Phase.READY:
                raise BaselineError("selected request is not ready")
            wait = self._now_ms - row.ready_since_ms
            state.total_ready_wait_ms += wait
            state.max_ready_wait_ms = max(state.max_ready_wait_ms, wait)
            state.phase = _Phase.IN_TARGET
            state.ready_since_ms = None
        for row in padding_rows:
            current = self._padding_ready.pop(row.request_id, None)
            if current != row:
                raise BaselineError("selected padding row is stale")

        sequence_lengths = tuple(row.sequence_length for row in real_rows)
        speculation_lengths = tuple(row.speculation_length for row in real_rows)
        realignment = (
            self.costs.realignment_latency_ms(sequence_lengths, speculation_lengths)
            if decision.requires_realignment
            else 0.0
        )
        verifier_slots = sum(row.speculation_length for row in selected)
        padded_slots = sum(row.speculation_length - 1 for row in padding_rows)
        duration = self.costs.target_latency_ms(len(selected), verifier_slots) + realignment
        self._target_launch_id += 1
        payload = _TargetPayload(
            launch_id=self._target_launch_id,
            start_ms=self._now_ms,
            end_ms=self._now_ms + duration,
            selected=selected,
            verifier_slots=verifier_slots,
            padded_slots=padded_slots,
            realignment_ms=realignment,
        )
        self._target_busy = True
        self._push(payload.end_ms, _EventKind.TARGET_COMPLETE, payload)

    def _complete_target(self, payload: _TargetPayload) -> None:
        if not self._target_busy:
            raise BaselineError("target completion arrived while target was idle")
        self._target_busy = False
        real_rows = tuple(row for row in payload.selected if not row.padding)
        padding_rows = tuple(row for row in payload.selected if row.padding)
        rollback_count = 0
        continuation_count = 0
        remote_rows: list[_RuntimeRequest] = []

        for row in real_rows:
            state = self._states[row.request_id]
            if state.phase is not _Phase.IN_TARGET or state.step_index != row.step_index:
                raise BaselineError("target completion names stale request state")
            step = state.config.steps[state.step_index]
            state.emitted_tokens.extend(step.emitted_tokens)
            state.accepted_lengths.append(step.accepted_length)
            state.sequence_length += len(step.emitted_tokens)
            state.last_accepted_length = step.accepted_length
            state.step_index += 1
            if state.step_index == len(state.config.steps):
                state.phase = _Phase.COMPLETE
                state.completion_ms = self._now_ms
                continue
            continuation_count += 1
            rollback_count += int(step.rollback)
            if step.needs_remote_draft:
                state.phase = _Phase.WAIT_DRAFT
                remote_rows.append(state)
            else:
                state.phase = _Phase.READY
                state.ready_since_ms = self._now_ms

        ratio = rollback_count / len(real_rows) if real_rows else 0.0
        mode: ExecutionMode | None = None
        if continuation_count:
            decision = self.scheduler.recovery_mode(ratio)
            if not isinstance(decision, ModeDecision):
                raise BaselineError("scheduler returned an invalid mode decision")
            if abs(decision.rollback_ratio - ratio) > self._EPSILON:
                raise BaselineError("mode decision changed the observed rollback ratio")
            self._mode_decisions.append(decision)
            mode = decision.mode
        for state in remote_rows:
            next_step = state.config.steps[state.step_index]
            duration = self.costs.recovery_latency_ms(
                context_tokens=state.sequence_length,
                speculation_length=next_step.speculation_length,
                compression_factor=self.scheduler.prompt_compression_factor,
            )
            self._job_sequence += 1
            job = _DraftJob(
                sequence=self._job_sequence,
                job_id=f"spec:{state.config.request_id}:{state.step_index}",
                kind=DraftJobKind.SPECULATIVE,
                release_ms=self._now_ms,
                duration_ms=duration,
                request_id=state.config.request_id,
                mode=mode,
            )
            self._draft_pending.append(job)
            if mode is ExecutionMode.PARALLEL:
                self._padding_ready[state.config.request_id] = CandidateView(
                    key=(state.config.request_id, state.step_index, True),
                    request_id=state.config.request_id,
                    step_index=state.step_index,
                    ready_since_ms=self._now_ms,
                    deadline_ms=state.config.deadline_ms,
                    sequence_length=state.sequence_length,
                    speculation_length=next_step.speculation_length,
                    last_accepted_length=state.last_accepted_length,
                    padding=True,
                )

        group_keys = {(row.sequence_length, row.speculation_length) for row in real_rows}
        self._target_launches.append(
            BaselineTargetLaunch(
                launch_id=payload.launch_id,
                start_ms=payload.start_ms,
                end_ms=payload.end_ms,
                request_ids=tuple(row.request_id for row in real_rows),
                padded_request_ids=tuple(row.request_id for row in padding_rows),
                verifier_slots=payload.verifier_slots,
                padded_slots=payload.padded_slots,
                realignment_ms=payload.realignment_ms,
                homogeneous_lengths=len(group_keys) <= 1,
                rollback_ratio=ratio,
                next_mode=mode,
            )
        )

    def _choose_draft_job(self) -> _DraftJob:
        if not self._draft_pending:
            raise BaselineError("cannot choose from an empty draft queue")
        burst = self.scheduler.draft_priority_burst
        if burst is None:
            return min(self._draft_pending, key=lambda job: (job.release_ms, job.sequence))
        speculative = [job for job in self._draft_pending if job.kind is DraftJobKind.SPECULATIVE]
        background = [job for job in self._draft_pending if job.kind is DraftJobKind.BACKGROUND]
        if background and (not speculative or self._speculative_streak >= burst):
            return min(background, key=lambda job: (job.release_ms, job.sequence))
        if speculative:
            return min(
                speculative,
                key=lambda job: (
                    self._states[job.request_id].config.deadline_ms
                    if job.request_id is not None
                    else math.inf,
                    job.release_ms,
                    job.sequence,
                ),
            )
        return min(background, key=lambda job: (job.release_ms, job.sequence))

    def _schedule_draft(self) -> None:
        if self._draft_busy or not self._draft_pending:
            return
        job = self._choose_draft_job()
        self._draft_pending.remove(job)
        if job.kind is DraftJobKind.SPECULATIVE:
            self._speculative_streak += 1
        else:
            self._speculative_streak = 0
        self._draft_launch_id += 1
        launch = BaselineDraftLaunch(
            launch_id=self._draft_launch_id,
            job_id=job.job_id,
            kind=job.kind,
            request_id=job.request_id,
            release_ms=job.release_ms,
            start_ms=self._now_ms,
            end_ms=self._now_ms + job.duration_ms,
            mode=job.mode,
        )
        self._draft_launches.append(launch)
        self._draft_busy = True
        self._push(launch.end_ms, _EventKind.DRAFT_COMPLETE, job)

    def _complete_draft(self, job: _DraftJob) -> None:
        if not self._draft_busy:
            raise BaselineError("draft completion arrived while draft was idle")
        self._draft_busy = False
        if job.kind is DraftJobKind.BACKGROUND:
            return
        if job.request_id is None:
            raise BaselineError("speculative job has no request")
        state = self._states[job.request_id]
        if state.phase is not _Phase.WAIT_DRAFT:
            raise BaselineError("speculative draft completed for a non-waiting request")
        state.phase = _Phase.READY
        state.ready_since_ms = self._now_ms
        # If target service was too busy to consume the padded parallel row,
        # the refreshed continuation supersedes it before launch.
        self._padding_ready.pop(job.request_id, None)

    def _handle_event(self, event: _Event) -> None:
        if event.kind is _EventKind.ARRIVAL:
            request_id = str(event.payload)
            state = self._states[request_id]
            if state.phase is not _Phase.NOT_ARRIVED:
                raise BaselineError("duplicate request arrival")
            state.phase = _Phase.READY
            state.ready_since_ms = self._now_ms
        elif event.kind is _EventKind.BACKGROUND_RELEASE:
            if not isinstance(event.payload, _DraftJob):
                raise BaselineError("invalid background release payload")
            self._draft_pending.append(event.payload)
        elif event.kind is _EventKind.TARGET_COMPLETE:
            if not isinstance(event.payload, _TargetPayload):
                raise BaselineError("invalid target completion payload")
            self._complete_target(event.payload)
        elif event.kind is _EventKind.DRAFT_COMPLETE:
            if not isinstance(event.payload, _DraftJob):
                raise BaselineError("invalid draft completion payload")
            self._complete_draft(event.payload)
        elif event.kind is _EventKind.WAKE:
            if not isinstance(event.payload, (int, float)):
                raise BaselineError("invalid wake payload")
            self._scheduled_wakes.discard(float(event.payload))

    def _finished(self) -> bool:
        requests_complete = all(state.phase is _Phase.COMPLETE for state in self._states.values())
        return (
            requests_complete
            and not self._target_busy
            and not self._draft_busy
            and not self._draft_pending
            and not any(
                event.kind
                in {
                    _EventKind.ARRIVAL,
                    _EventKind.BACKGROUND_RELEASE,
                    _EventKind.TARGET_COMPLETE,
                    _EventKind.DRAFT_COMPLETE,
                }
                for event in self._events
            )
        )

    def run(self) -> BaselineResult:
        """Execute the trace once and return auditable scheduling evidence."""

        if self._has_run:
            raise BaselineError("a simulator instance may only run once")
        self._has_run = True
        events_processed = 0
        while not self._finished():
            self._schedule_draft()
            self._schedule_target()
            if not self._events:
                raise BaselineError("event queue drained before completion")
            event = heapq.heappop(self._events)
            self._now_ms = event.time_ms
            self._handle_event(event)
            events_processed += 1
            while self._events and abs(self._events[0].time_ms - self._now_ms) <= self._EPSILON:
                simultaneous = heapq.heappop(self._events)
                self._handle_event(simultaneous)
                events_processed += 1
            if events_processed > self.max_events:
                raise BaselineError("max_events exceeded")

        request_results: list[BaselineRequestResult] = []
        for request in self.trace.requests:
            state = self._states[request.request_id]
            if state.completion_ms is None:
                raise BaselineError("completed simulation lacks a completion timestamp")
            request_results.append(
                BaselineRequestResult(
                    request_id=request.request_id,
                    emitted_tokens=tuple(state.emitted_tokens),
                    accepted_lengths=tuple(state.accepted_lengths),
                    completion_ms=state.completion_ms,
                    deadline_ms=request.deadline_ms,
                    total_ready_wait_ms=state.total_ready_wait_ms,
                    max_ready_wait_ms=state.max_ready_wait_ms,
                )
            )
        semantic_signature: SemanticSignature = tuple(
            (
                request.request_id,
                request.emitted_tokens,
                tuple(
                    (
                        step.accepted_length,
                        len(step.emitted_tokens),
                        step.needs_remote_draft,
                        step.rollback,
                    )
                    for step in self._states[request.request_id].config.steps
                ),
            )
            for request in request_results
        )
        if semantic_signature != self.trace.semantic_signature:
            raise BaselineError("scheduler changed the pre-realized semantic trace")
        metrics = self._metrics(tuple(request_results))
        return BaselineResult(
            policy_name=self.scheduler.name,
            trace_name=self.trace.name,
            requests=tuple(request_results),
            target_launches=tuple(self._target_launches),
            draft_launches=tuple(self._draft_launches),
            mode_decisions=tuple(self._mode_decisions),
            semantic_signature=semantic_signature,
            metrics=metrics,
        )

    def _metrics(self, requests: tuple[BaselineRequestResult, ...]) -> BaselineMetrics:
        launches = self._target_launches
        count = len(launches)
        background = [
            launch for launch in self._draft_launches if launch.kind is DraftJobKind.BACKGROUND
        ]
        started = min(request.arrival_ms for request in self.trace.requests)
        finished = max(request.completion_ms for request in requests)
        return BaselineMetrics(
            policy_name=self.scheduler.name,
            makespan_ms=finished - started,
            target_launches=count,
            mean_real_batch=(
                sum(launch.real_batch_size for launch in launches) / count if count else 0.0
            ),
            mean_effective_batch=(
                sum(launch.effective_batch_size for launch in launches) / count if count else 0.0
            ),
            verifier_slots=sum(launch.verifier_slots for launch in launches),
            padded_slots=sum(launch.padded_slots for launch in launches),
            homogeneous_batches=sum(launch.homogeneous_lengths for launch in launches),
            realignment_fallbacks=sum(launch.realignment_ms > 0.0 for launch in launches),
            realignment_ms=sum(launch.realignment_ms for launch in launches),
            total_ready_wait_ms=sum(request.total_ready_wait_ms for request in requests),
            max_ready_wait_ms=max(
                (request.max_ready_wait_ms for request in requests),
                default=0.0,
            ),
            starved_requests=sum(
                request.max_ready_wait_ms > self.costs.starvation_threshold_ms
                for request in requests
            ),
            deadline_misses=sum(request.deadline_missed for request in requests),
            ordinary_mode_batches=sum(
                decision.mode is ExecutionMode.ORDINARY for decision in self._mode_decisions
            ),
            parallel_mode_batches=sum(
                decision.mode is ExecutionMode.PARALLEL for decision in self._mode_decisions
            ),
            speculative_draft_jobs=sum(
                launch.kind is DraftJobKind.SPECULATIVE for launch in self._draft_launches
            ),
            background_draft_jobs=len(background),
            max_background_draft_wait_ms=max(
                (launch.wait_ms for launch in background),
                default=0.0,
            ),
        )


def assert_semantic_equivalence(*results: BaselineResult) -> None:
    """Raise when scheduler runs do not emit identical pre-realized outcomes."""

    if len(results) < 2:
        raise ValueError("at least two results are required")
    expected = results[0].semantic_signature
    for result in results[1:]:
        if result.semantic_signature != expected:
            raise BaselineError(f"{result.policy_name!r} changed the semantic outcome trace")


__all__ = [
    "BackgroundDraftJob",
    "BaselineCostModel",
    "BaselineDraftLaunch",
    "BaselineError",
    "BaselineMetrics",
    "BaselineRequestResult",
    "BaselineResult",
    "BaselineScheduler",
    "BaselineTargetLaunch",
    "BatchDecision",
    "CandidateView",
    "DeterministicBaselineSimulator",
    "DraftJobKind",
    "EXSpecSlidingPoolScheduler",
    "ExecutionMode",
    "FIFOScheduler",
    "ModeDecision",
    "MyopicSlackScheduler",
    "PreRealizedTrace",
    "RealizedRequest",
    "RealizedStep",
    "SPECTREHybridScheduler",
    "SemanticSignature",
    "SpectreCalibration",
    "SpectreHybridScheduler",
    "assert_semantic_equivalence",
]
